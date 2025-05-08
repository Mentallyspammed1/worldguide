# -*- coding: utf-8 -*-
# pylint: disable=logging-fstring-interpolation, too-many-instance-attributes, too-many-arguments, too-many-locals, too-many-public-methods, invalid-name, unused-argument, too-many-lines, wrong-import-order, wrong-import-position, unnecessary-pass, unnecessary-lambda-assignment, bad-option-value, line-too-long
# fmt: off
#   ____        _       _   _                  _            _         _
#  |  _ \\ _   _| |_ ___| | | | __ ___   ____ _| |_ ___  ___| |_ _ __ | |__   ___ _ __ ___  _ __
#  | |_) | | | | __/ _ \\ | | |/ _` |\\ V / _` | __/ _ \\/ __| __| '_ \\| '_ \\ / _ \\ '_ ` _ \\| '_ \\
#  |  __/| |_| | ||  __/ | | | (_| |\\ V / (_| | ||  __/\\__ \\ |_| |_) | | | |  __/ | | | | | |_) |
#  |_|   \\__,_|\\__\\___|_|_|_|\\__,_| \\_/ \\__,_|\\__\\___||___/\\__| .__/|_| |_|\\___|_| |_| |_| .__/
#                                                             |_|                       |_|
# fmt: on
"""
Pyrmethus - Termux Trading Spell (v4.5.7 - Neon Nexus Edition)

Conjures market insights and executes trades on Bybit Futures using the
V5 Unified Account API via CCXT. Refactored into classes for better structure
and utilizing V5 position-based stop-loss/take-profit/trailing-stop features.

Enhancements in this version (4.5.7+):
- Corrected colorama usage (Fore.RED + Style.BRIGHT instead of Fore.BRIGHT_RED).
- Merged v4.5.7 Exit Logic (EMA priority, Stoch reversal confirmation).
- Corrected Bybit V5 private method call for setting trading stops (`private_post_position_set_trading_stop`).
- Added `stoch_k_prev` calculation and passing in IndicatorCalculator.
- Integrated Enhanced Configuration Handling (_get_env, Snippet 1).
- Integrated Enhanced Retry Logic (fetch_with_retries, Snippet 2).
- Integrated Safer Indicator Data Conversion (calculate_indicators, Snippet 3).
- Integrated Robust V5 Parameter Formatting Helper (_format_v5_param, Snippet 4).
- Integrated Enhanced Position Verification Logic (_verify_position_state, Snippet 5).
- Applied PEP8 formatting and Neon Recolorization.
- Improved Docstrings and Comments.
- Ensured consistency in logging and variable names.
- Optimized minor parts for clarity and robustness.

Original v2.4.1 Base Features & Enhancements:
- Robust configuration loading.
- Multi-condition signal generation.
- V5 Position-based SL/TP/TSL management.
- Signal-based exit mechanism.
- Enhanced error handling with retries.
- Decimal type for high precision.
- Trade journaling.
- Termux notifications.
- Graceful shutdown handling.
- Rich library integration for terminal output.
- Fixed Termux notification command.
- Fixed Decimal conversion errors from API strings.
- Implemented robust `safe_decimal` utility.
- Corrected V5 order cancellation logic.
- Ensured numeric params for V5 stops passed as strings.
- Handled NaN values better.
- Replaced deprecated pandas `applymap` with `map`.
- Simplified previous indicator value fetching.
"""

# Standard Library Imports
import copy
import csv
import logging
import os
import platform
import signal
import subprocess
import sys
import textwrap
import time
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
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Type
import types # For signal frame type hint

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

    COMMON_PACKAGES = [
        "ccxt",
        "python-dotenv",
        "pandas",
        "numpy",
        "rich",
        "colorama",
        "requests",
    ]
except ImportError as e:
    # Initialize colorama here for error message formatting if import fails
    if "colorama" in str(e):
        print("Missing essential package: colorama. Cannot display colored output.")
        print("Attempting basic error message...")
        print(f"Missing essential spell component: {e.name}")
        print(f"To conjure it, cast: pip install {e.name}")
        print("\nOr, to ensure all scrolls are present, cast:")
        print(f"pip install {' '.join(COMMON_PACKAGES)}")
        sys.exit(1)
    else:
        # Attempt basic colorama init for the error message itself
        try:
            colorama_init(autoreset=True)
            missing_pkg = e.name
            print(
                f"{Fore.RED}{Style.BRIGHT}Missing essential spell component: {missing_pkg}{Style.RESET_ALL}"
            )
            print(
                f"{Fore.YELLOW}To conjure it, cast: {Style.BRIGHT}pip install {missing_pkg}{Style.RESET_ALL}"
            )
            print(f"\n{Fore.CYAN}Or, to ensure all scrolls are present, cast:")
            if os.getenv("TERMUX_VERSION"):
                print(
                    f"{Style.BRIGHT}pkg install python python-pandas python-numpy && pip install {' '.join([p for p in COMMON_PACKAGES if p not in ['pandas', 'numpy']])}{Style.RESET_ALL}"
                )
                print(
                    f"{Fore.YELLOW}Note: pandas and numpy often installed via pkg in Termux.{Style.RESET_ALL}"
                )
            else:
                print(
                    f"{Style.BRIGHT}pip install {' '.join(COMMON_PACKAGES)}{Style.RESET_ALL}"
                )
        except Exception: # Fallback to plain print if colorama init failed even here
             print(f"Missing essential spell component: {e.name}")
             print(f"To conjure it, cast: pip install {e.name}")
             print("\nOr, to ensure all scrolls are present, cast:")
             print(f"pip install {' '.join(COMMON_PACKAGES)}")
        sys.exit(1)

# --- Constants ---
DECIMAL_PRECISION = 50
POSITION_QTY_EPSILON = Decimal("1E-12")  # Threshold for considering a position 'flat'
DEFAULT_PRICE_DP = 4
DEFAULT_AMOUNT_DP = 6
DEFAULT_OHLCV_LIMIT = 200
DEFAULT_LOOP_SLEEP = 15
DEFAULT_RETRY_DELAY = 3
DEFAULT_MAX_RETRIES = 3
DEFAULT_RISK_PERCENT = Decimal("0.01")
DEFAULT_SL_MULT = Decimal("1.5")
DEFAULT_TP_MULT = Decimal("3.0")
DEFAULT_TSL_ACT_MULT = Decimal("1.0")
DEFAULT_TSL_PERCENT = Decimal("0.5")
DEFAULT_STOCH_OVERSOLD = Decimal("30")
DEFAULT_STOCH_OVERBOUGHT = Decimal("70")
DEFAULT_MIN_ADX = Decimal("20")
DEFAULT_JOURNAL_FILE = "pyrmethus_trading_journal.csv"
V5_UNIFIED_ACCOUNT_TYPE = "UNIFIED"
V5_HEDGE_MODE_POSITION_IDX = 0 # Default index for hedge mode (0=One-Way, 1=Buy Hedge, 2=Sell Hedge)
V5_TPSL_MODE_FULL = "Full" # Apply SL/TP to entire position
V5_SUCCESS_RETCODE = 0
TERMUX_NOTIFY_TIMEOUT = 10 # Increased timeout for termux-toast

# Initialize Colorama & Rich Console
colorama_init(autoreset=True)
console = Console(log_path=False) # Disable Rich logging interception

# Set Decimal precision context
getcontext().prec = DECIMAL_PRECISION

# --- Logging Setup ---
# Custom logging level for trade actions
TRADE_LEVEL_NUM = 25  # Between INFO and WARNING
if not hasattr(logging.Logger, "trade"):
    logging.addLevelName(TRADE_LEVEL_NUM, "TRADE")

    def trade_log(self, message, *args, **kws):
        """Logs a message with level TRADE."""
        if self.isEnabledFor(TRADE_LEVEL_NUM):
            # pylint: disable=protected-access
            self._log(TRADE_LEVEL_NUM, message, args, **kws)

    logging.Logger.trade = trade_log  # type: ignore[attr-defined]

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

    def __init__(self, env_file: str = ".env"):
        logger.debug(f"Loading configuration from environment variables / {env_file}...")
        # Use pathlib for robustness
        env_path = Path(env_file)
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path)
            logger.info(f"Loaded configuration from {env_path}")
        else:
            logger.warning(f"Environment file '{env_path}' not found. Relying on system environment variables.")

        # Core Trading Parameters
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT", Style.DIM)
        self.market_type: str = self._get_env(
            "MARKET_TYPE",
            "linear",
            Style.DIM,
            allowed_values=["linear", "inverse", "swap"],
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
        self.journal_file_path: str = self._get_env("JOURNAL_FILE_PATH", DEFAULT_JOURNAL_FILE, Style.DIM)
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

            if self.market_type == "inverse":
                category = "inverse" # e.g., BTC/USD:BTC
            elif self.market_type in ["linear", "swap"]:
                # Linear includes USDT/USDC perpetuals/futures
                category = "linear"
            else:
                # This shouldn't happen due to _get_env validation, but as a safeguard:
                raise ValueError(f"Unsupported MARKET_TYPE '{self.market_type}'")

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

    def _validate_config(self):
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

    # --- Enhanced _get_env and helpers from Snippet 1 ---
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
                if dec_val.as_tuple().exponent < 0: # Check if fractional part exists
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
            if default is None:
                log_msg = f"{Fore.RED}{Style.BRIGHT}Required {'secret ' if is_secret else ''}configuration '{key}' not found and no default provided. Halting.{Style.RESET_ALL}"
                logger.critical(log_msg)
                sys.exit(1)
            use_default = True
            value_str_to_process = str(default) # Use string representation of default for casting/logging
            source = f"default ({default})"
            log_value_display = default # Display original default
        else:
            value_str_to_process = value_str
            log_value_display = "****" if is_secret else value_str_to_process

        # Log the found/default value
        log_method = logger.warning if use_default else logger.info
        log_method(f"{color}Using {key}: {log_value_display} (from {source}){Style.RESET_ALL}")

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
    # --- End Enhanced _get_env ---

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
        else:
             # Critical failure already logged in _initialize_exchange, exiting handled there
             pass # Exit already occurred

    def _initialize_exchange(self):
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
                    "brokerId": "TermuxNeonV5", # Custom ID for tracking via Bybit referrals
                    "createMarketBuyOrderRequiresPrice": False, # V5 specific option
                    # "defaultMarginMode": "isolated", # Or 'cross', relevant for some calls
                    "defaultTimeInForce": "GTC", # Good-Till-Cancelled default
                    # 'enableRateLimit': True, # CCXT default is usually True
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
                exc_info=False,
            )
            sys.exit(1)
        except (ccxt.NetworkError, requests.exceptions.RequestException) as e:
            # Corrected color usage here
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
                if prec_dec < 1: # Likely a step size (e.g., 0.01)
                     # Calculate dp from step size exponent
                     exponent = prec_dec.as_tuple().exponent
                     # Ensure exponent is integer before abs()
                     return abs(int(exponent)) if isinstance(exponent, int) else default_dp
                else: # Likely number of decimal places directly
                    try:
                        return int(prec_dec)
                    except (ValueError, TypeError):
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

    def format_price(
        self, price: Union[Decimal, str, float, int]
    ) -> str:
        """Formats price according to market precision using ROUND_HALF_EVEN."""
        price_decimal = safe_decimal(price)
        if price_decimal.is_nan():
            return "NaN" # Return NaN string if input was bad

        precision = DEFAULT_PRICE_DP # Default
        if (
            self.market_info
            and "precision_dp" in self.market_info
            and "price" in self.market_info["precision_dp"]
        ):
            precision = self.market_info["precision_dp"]["price"]

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

        precision = DEFAULT_AMOUNT_DP # Default
        if (
            self.market_info
            and "precision_dp" in self.market_info
            and "amount" in self.market_info["precision_dp"]
        ):
            precision = self.market_info["precision_dp"]["amount"]

        try:
            quantizer = Decimal("1e-" + str(precision))
            formatted_amount = amount_decimal.quantize(quantizer, rounding=rounding_mode)
            # Ensure the string representation matches the precision exactly
            return f"{formatted_amount:.{precision}f}"
        except (InvalidOperation, ValueError) as e:
             logger.error(f"Error formatting amount {amount_decimal} to {precision}dp: {e}")
             return "ERR" # Indicate formatting error

    # --- V5 Parameter Formatting Helper (Snippet 4) ---
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
            formatter = self.format_price if param_type in ["price", "distance"] else self.format_amount
            # Format zero itself to ensure correct number of decimal places if needed by API
            formatted_zero = formatter(Decimal("0"))
            # Ensure it's not "ERR" or "NaN" after formatting zero
            return formatted_zero if formatted_zero not in ["ERR", "NaN"] else None
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
            formatter = self.format_amount # Pass rounding mode when calling if needed
        else:
            logger.error(f"V5 Param Formatting: Unknown param_type '{param_type}'.")
            return None

        # Format the positive Decimal value
        # Apply ROUND_DOWN specifically for amount type here if needed, or rely on format_amount default
        rounding = ROUND_DOWN if param_type == "amount" else ROUND_HALF_EVEN
        formatted_str = formatter(decimal_value) # format_amount default is ROUND_DOWN

        # Final check: ensure the formatted string isn't an error/NaN indicator
        if formatted_str in ["ERR", "NaN"]:
            logger.error(f"V5 Param Formatting: Failed to format valid string for '{value}' (type: {param_type}). Result: {formatted_str}")
            return None

        # logger.debug(f"V5 Param Formatted: Input='{value}', Type='{param_type}', Output='{formatted_str}'")
        return formatted_str
    # --- End V5 Parameter Formatting Helper ---

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
            for col in ["open", "high", "low", "close", "volume"]:
                # Use safe_decimal utility via map for efficiency
                df[col] = df[col].map(safe_decimal)
                # Check if any conversion resulted in NaN, which indicates bad data
                if df[col].apply(lambda x: isinstance(x, Decimal) and x.is_nan()).any():
                     logger.warning(f"Column '{col}' contains NaN values after conversion. Check API data.")

            # Drop rows where essential price data is NaN after conversion
            initial_len = len(df)
            df.dropna(subset=["open", "high", "low", "close"], inplace=True)
            if len(df) < initial_len:
                logger.warning(f"Dropped {initial_len - len(df)} rows with NaN OHLC values.")

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
            if (total_equity.is_nan() or available_balance.is_nan()) and "info" in balance_data:
                logger.debug("Parsing balance from 'info' field as fallback...")
                info_result = balance_data["info"].get("result", {})
                account_list = info_result.get("list", [])
                if account_list and isinstance(account_list, list):
                    # Find the Unified account details
                    unified_acc_info = next((item for item in account_list if item.get("accountType") == V5_UNIFIED_ACCOUNT_TYPE), None)
                    if unified_acc_info:
                        if total_equity.is_nan():
                            total_equity = safe_decimal(unified_acc_info.get("totalEquity"))
                        if available_balance.is_nan():
                            available_balance = safe_decimal(unified_acc_info.get("totalAvailableBalance"))

                        # If still NaN, check the specific coin details within the account
                        if available_balance.is_nan() and "coin" in unified_acc_info:
                             coin_list = unified_acc_info.get("coin", [])
                             if coin_list and isinstance(coin_list, list):
                                 settle_coin_info = next((c for c in coin_list if c.get("coin") == settle_currency), None)
                                 if settle_coin_info:
                                      available_balance = safe_decimal(settle_coin_info.get("availableToWithdraw"))
                                      if total_equity.is_nan():
                                          total_equity = safe_decimal(settle_coin_info.get("equity"))

            # Final Validation and Logging
            if total_equity.is_nan():
                logger.error(
                    f"Could not extract valid total equity for {settle_currency}. Balance data might be incomplete or unexpected format. Raw snippet: {str(balance_data)[:500]}"
                )
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
        """Fetches current position details for the symbol using V5 API."""
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
            }
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

            # --- Find the specific position matching symbol AND index from the list ---
            target_pos_info = None
            for pos in fetched_positions_list:
                info = pos.get("info", {})
                pos_symbol_info = info.get("symbol")
                pos_idx_str = info.get("positionIdx")
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

            # --- Parse the found position details using safe_decimal ---
            qty = safe_decimal(target_pos_info.get("size", "0"))
            side = target_pos_info.get("side", "None").lower() # 'Buy' -> 'buy', 'Sell' -> 'sell', 'None' -> 'none'
            entry_price = safe_decimal(target_pos_info.get("avgPrice", "0"))
            liq_price_raw = safe_decimal(target_pos_info.get("liqPrice", "0"))
            unrealized_pnl = safe_decimal(target_pos_info.get("unrealisedPnl", "0"))
            sl_price_raw = safe_decimal(target_pos_info.get("stopLoss", "0"))
            tp_price_raw = safe_decimal(target_pos_info.get("takeProfit", "0"))
            tsl_trigger_price_raw = safe_decimal(target_pos_info.get("trailingStop", "0"))

            # --- Validate and clean up parsed values ---
            qty_abs = qty.copy_abs() if not qty.is_nan() else Decimal("0")
            entry_price = entry_price if not entry_price.is_nan() and entry_price > 0 else Decimal("NaN")
            liq_price = liq_price_raw if not liq_price_raw.is_nan() and liq_price_raw > 0 else Decimal("NaN")
            sl_price = sl_price_raw if not sl_price_raw.is_nan() and sl_price_raw > 0 else None
            tp_price = tp_price_raw if not tp_price_raw.is_nan() and tp_price_raw > 0 else None
            is_tsl_active = not tsl_trigger_price_raw.is_nan() and tsl_trigger_price_raw > 0
            tsl_trigger_price = tsl_trigger_price_raw if is_tsl_active else None
            is_position_open = qty_abs >= POSITION_QTY_EPSILON

            if not is_position_open:
                logger.debug(f"Position size {qty} is negligible or zero. Considered flat.")
                return positions_dict # Return empty dict

            # Determine position side based on API 'side' field or posIdx
            position_side_key = None
            if side == "buy": position_side_key = "long"
            elif side == "sell": position_side_key = "short"
            elif side == "none" and self.config.position_idx == 1: position_side_key = "long"
            elif side == "none" and self.config.position_idx == 2: position_side_key = "short"
            # Add case for one-way mode (posIdx=0) where side 'None' might mean flat (already handled by qty check)
            # Or side 'Buy'/'Sell' indicates the one-way position direction.
            elif side in ["buy", "sell"] and self.config.position_idx == 0:
                position_side_key = "long" if side == "buy" else "short"


            if position_side_key:
                position_details = {
                    "qty": qty_abs, # Store absolute quantity
                    "entry_price": entry_price,
                    "liq_price": liq_price,
                    "unrealized_pnl": unrealized_pnl if not unrealized_pnl.is_nan() else Decimal("0"),
                    "side": side, # Store original API side ('buy'/'sell'/'None')
                    "info": target_pos_info, # Store raw info for debugging
                    "stop_loss_price": sl_price,
                    "take_profit_price": tp_price,
                    "is_tsl_active": is_tsl_active,
                    "tsl_trigger_price": tsl_trigger_price, # Store the trigger price if active
                }
                positions_dict[position_side_key] = position_details
                entry_str = entry_price.normalize() if not entry_price.is_nan() else "[dim]N/A[/]"
                logger.debug(
                    f"Found {position_side_key.upper()} position: Qty={qty_abs.normalize()}, Entry={entry_str}"
                )
            else:
                 logger.warning(f"Position found with size {qty} but side is '{side}' and posIdx is {self.config.position_idx}. Could not determine long/short state reliably. Treating as flat for safety.")
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

    # --- Enhanced calculate_indicators from Snippet 3 (with stoch_k_prev added) ---
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

        required_cols = ["open", "high", "low", "close"]
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
                def safe_to_float(x):
                    if isinstance(x, (float, int)):
                        return float(x)
                    if isinstance(x, Decimal):
                        return float('nan') if x.is_nan() else float(x)
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
            df_calc.dropna(subset=required_cols, inplace=True)
            rows_dropped = initial_len - len(df_calc)
            if rows_dropped > 0:
                 logger.debug(f"Dropped {rows_dropped} rows with NaN in OHLC after float conversion.")

            if df_calc.empty:
                logger.error(f"{Fore.RED}DataFrame empty after NaN drop during indicator calculation.{Style.RESET_ALL}")
                return None

            # --- Check Data Length ---
            max_period = max(
                self.config.slow_ema_period, self.config.trend_ema_period,
                self.config.stoch_period + self.config.stoch_smooth_k + self.config.stoch_smooth_d,
                self.config.atr_period, self.config.adx_period * 2,
            )
            min_required_len = max_period + 10 # Buffer for stability
            if len(df_calc) < min_required_len:
                logger.error(f"{Fore.RED}Insufficient data ({len(df_calc)} rows < required ~{min_required_len}) for indicators.{Style.RESET_ALL}")
                return None

            # --- Indicator Calculations (using df_calc with float types) ---
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
            stoch_k_raw = np.where(stoch_range > 1e-12, 100 * (close_s - low_min) / stoch_range, 50.0)
            stoch_k_raw_s = pd.Series(stoch_k_raw, index=df_calc.index).fillna(50)
            stoch_k_s = stoch_k_raw_s.rolling(window=self.config.stoch_smooth_k).mean().fillna(50)
            stoch_d_s = stoch_k_s.rolling(window=self.config.stoch_smooth_d).mean().fillna(50)

            # ATR
            prev_close = close_s.shift(1)
            tr1 = high_s - low_s
            tr2 = (high_s - prev_close).abs()
            tr3 = (low_s - prev_close).abs()
            tr_s = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(0)
            atr_s = tr_s.ewm(alpha=1 / self.config.atr_period, adjust=False).mean()

            # ADX, +DI, -DI
            adx_s, pdi_s, mdi_s = self._calculate_adx(
                high_s, low_s, close_s, atr_s, self.config.adx_period
            )

            # --- Extract Latest Values & Convert back to Decimal ---
            def get_latest_decimal(series: pd.Series, name: str) -> Decimal:
                """Safely get the last valid value from a Series and convert to Decimal."""
                valid_series = series.dropna()
                if valid_series.empty: return Decimal("NaN")
                last_valid = valid_series.iloc[-1]
                try: return Decimal(str(last_valid))
                except (InvalidOperation, TypeError, ValueError):
                    logger.error(f"Failed converting latest {name} value '{last_valid}' (type: {type(last_valid).__name__}) to Decimal.")
                    return Decimal("NaN")

            indicators_out = {
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

            # --- Calculate Previous Stochastic K for Exit Logic ---
            stoch_k_valid = stoch_k_s.dropna()
            k_prev = Decimal("NaN")
            if len(stoch_k_valid) >= 2:
                k_prev = safe_decimal(str(stoch_k_valid.iloc[-2]))
            indicators_out["stoch_k_prev"] = k_prev # Add previous K to output

            # --- Calculate Stochastic Cross Signals ---
            k_last = indicators_out["stoch_k"]
            d_last = indicators_out["stoch_d"]
            d_prev = Decimal("NaN")
            stoch_d_valid = stoch_d_s.dropna()
            if len(stoch_d_valid) >= 2:
                 d_prev = safe_decimal(str(stoch_d_valid.iloc[-2]))

            stoch_kd_bullish = False
            stoch_kd_bearish = False
            if not any(v.is_nan() for v in [k_last, d_last, k_prev, d_prev]):
                crossed_above = (k_last > d_last) and (k_prev <= d_prev)
                crossed_below = (k_last < d_last) and (k_prev >= d_prev)
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
                if indicators_out.get("atr", Decimal("NaN")).is_nan():
                     logger.error(f"{Fore.RED}ATR is NaN, cannot proceed with risk calculations. Aborting indicator calc.{Style.RESET_ALL}")
                     return None

            logger.info(f"{Fore.GREEN}{Style.BRIGHT}Indicator patterns woven successfully.{Style.RESET_ALL}")
            return indicators_out

        except Exception as e:
            logger.error(f"{Fore.RED}Error weaving indicator patterns: {e}{Style.RESET_ALL}", exc_info=True)
            return None
    # --- End Enhanced calculate_indicators ---

    def _calculate_adx(
        self,
        high_s: pd.Series,
        low_s: pd.Series,
        close_s: pd.Series, # Included for context, though not directly used in this implementation part
        atr_s: pd.Series, # Pre-calculated ATR series
        period: int,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Helper to calculate ADX, +DI, -DI using Wilder's smoothing (EMA)."""
        if period <= 0:
            raise ValueError("ADX period must be positive")
        if atr_s.empty or atr_s.isnull().all(): # Check if ATR is empty or all NaN
             logger.error("ATR series is empty or all NaN, cannot calculate ADX.")
             nan_series = pd.Series(np.nan, index=high_s.index)
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
        plus_dm_s = pd.Series(plus_dm, index=high_s.index).ewm(alpha=alpha, adjust=False).mean().fillna(0)
        minus_dm_s = pd.Series(minus_dm, index=high_s.index).ewm(alpha=alpha, adjust=False).mean().fillna(0)

        # Calculate Directional Indicators (+DI, -DI)
        pdi_s_raw = np.where((atr_s > 1e-12) & (~atr_s.isnull()), 100 * plus_dm_s / atr_s, 0.0)
        mdi_s_raw = np.where((atr_s > 1e-12) & (~atr_s.isnull()), 100 * minus_dm_s / atr_s, 0.0)
        pdi_s = pd.Series(pdi_s_raw, index=high_s.index).fillna(0)
        mdi_s = pd.Series(mdi_s_raw, index=high_s.index).fillna(0)

        # Calculate Directional Movement Index (DX)
        di_diff_abs = (pdi_s - mdi_s).abs()
        di_sum = pdi_s + mdi_s
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

        # --- Pre-checks ---
        if not indicators:
            result["reason"] = "No Signal: Indicators missing"
            logger.debug(result["reason"])
            return result
        if df_last_candles is None or len(df_last_candles) < 2:
            reason = f"No Signal: Insufficient candle data (<2, got {len(df_last_candles) if df_last_candles is not None else 0})"
            result["reason"] = reason
            logger.debug(reason)
            return result

        try:
            # --- Extract Data & Indicators Safely ---
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
            ind_values = {}
            nan_keys = []
            for key in required_indicator_keys:
                val = indicators.get(key)
                if isinstance(val, Decimal) and val.is_nan():
                    nan_keys.append(key)
                elif val is None: # Consider None as missing/invalid too
                    nan_keys.append(key)
                else:
                    ind_values[key] = val

            if nan_keys:
                result["reason"] = f"No Signal: Required indicator(s) NaN/Missing: {', '.join(nan_keys)}"
                logger.warning(result["reason"])
                return result

            k, fast_ema, slow_ema, trend_ema, atr, adx, pdi, mdi = (
                ind_values["stoch_k"], ind_values["fast_ema"], ind_values["slow_ema"],
                ind_values["trend_ema"], ind_values["atr"], ind_values["adx"],
                ind_values["pdi"], ind_values["mdi"]
            )
            kd_bull = indicators.get("stoch_kd_bullish", False)
            kd_bear = indicators.get("stoch_kd_bearish", False)

            # --- Define Conditions ---
            ema_bullish_cross = fast_ema > slow_ema
            ema_bearish_cross = fast_ema < slow_ema

            trend_buffer = trend_ema.copy_abs() * (self.config.trend_filter_buffer_percent / 100)
            price_above_trend_ema = current_price > (trend_ema - trend_buffer)
            price_below_trend_ema = current_price < (trend_ema + trend_buffer)
            trend_allows_long = price_above_trend_ema if self.config.trade_only_with_trend else True
            trend_allows_short = price_below_trend_ema if self.config.trade_only_with_trend else True
            trend_reason = f"Trend(P:{current_price:.{DEFAULT_PRICE_DP}f} vs EMA:{trend_ema:.{DEFAULT_PRICE_DP}f})" if self.config.trade_only_with_trend else "TrendFilter OFF"

            stoch_long_cond = (k < self.config.stoch_oversold_threshold) or kd_bull
            stoch_short_cond = (k > self.config.stoch_overbought_threshold) or kd_bear
            stoch_reason = f"Stoch(K:{k:.1f} {'BullX' if kd_bull else ''}{'BearX' if kd_bear else ''})"

            significant_move = True
            atr_reason = "ATR Filter OFF"
            if self.config.atr_move_filter_multiplier > 0:
                if atr.is_nan() or atr <= 0:
                    atr_reason = f"ATR Filter Skipped (Invalid ATR: {atr})"
                    significant_move = False
                elif prev_close.is_nan():
                    atr_reason = "ATR Filter Skipped (Prev Close NaN)"
                    significant_move = False
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

            # --- Build Detailed Reason String ---
            if final_long_signal:
                result["long"] = True
                result["reason"] = f"Long Signal: EMA Bull | {stoch_reason} | {trend_reason} | {atr_reason} | {adx_reason}"
            elif final_short_signal:
                result["short"] = True
                result["reason"] = f"Short Signal: EMA Bear | {stoch_reason} | {trend_reason} | {atr_reason} | {adx_reason}"
            else:
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
                else:
                    reason_parts.append(f"Conditions unmet (EMA:{ema_bullish_cross}/{ema_bearish_cross}, Stoch:{stoch_long_cond}/{stoch_short_cond}, Trend:{trend_allows_long}/{trend_allows_short}, ATR:{significant_move}, ADX:{adx_allows_long}/{adx_allows_short})")
                result["reason"] = " | ".join(reason_parts)

            log_level_sig = logging.INFO if result["long"] or result["short"] or "Blocked" in result["reason"] else logging.DEBUG
            logger.log(log_level_sig, f"Signal Check: {result['reason']}")

        except Exception as e:
            logger.error(f"{Fore.RED}Error generating signals: {e}{Style.RESET_ALL}", exc_info=True)
            result["reason"] = f"No Signal: Exception ({type(e).__name__})"
            result["long"] = False
            result["short"] = False

        return result

    # --- Updated check_exit_signals method from v4.5.7 ---
    def check_exit_signals(
        self,
        position_side: str, # 'long' or 'short'
        indicators: Dict[str, Union[Decimal, bool, int]],
    ) -> Optional[str]:
        """Checks for signal-based exits (EMA cross, Stoch reversal). Returns reason string or None."""
        if not indicators:
            logger.warning("Cannot check exit signals: indicators missing.")
            return None

        # --- Inside check_exit_signals, after extracting indicators ---
        # Ensure necessary indicators are present and valid
        fast_ema = indicators.get("fast_ema", Decimal("NaN"))
        slow_ema = indicators.get("slow_ema", Decimal("NaN"))
        stoch_k = indicators.get("stoch_k", Decimal("NaN"))
        # IMPORTANT: This logic requires the *previous* Stochastic K value.
        # Ensure 'stoch_k_prev' is correctly calculated and included in the 'indicators' dict passed to this method.
        stoch_k_prev = indicators.get("stoch_k_prev", Decimal("NaN"))

        # Assume EMA cross signals are calculated just before this block based on fast_ema and slow_ema
        # Example calculation (should happen before this snippet):
        ema_bullish_cross = fast_ema > slow_ema
        ema_bearish_cross = fast_ema < slow_ema
        # These variables are used below

        # Check for NaN values in required indicators
        if fast_ema.is_nan() or slow_ema.is_nan() or stoch_k.is_nan() or stoch_k_prev.is_nan():
            logger.warning(
                "Cannot check exit signals due to NaN indicators (EMA/Stoch/StochPrev)."
            )
            return None # Cannot proceed without valid indicator values

        exit_reason: Optional[str] = None

        # --- Define Overbought/Oversold levels from config ---
        # Ensure these config values are valid Decimals during initialization
        oversold_level = self.config.stoch_oversold_threshold
        overbought_level = self.config.stoch_overbought_threshold

        # --- Evaluate Exit Conditions based on Position Side ---
        if position_side == "long":
            # Priority 1: EMA Bearish Cross
            if ema_bearish_cross:
                exit_reason = "Exit Signal: EMA Bearish Cross"
                logger.trade(f"{Fore.YELLOW}{exit_reason} detected for LONG position.{Style.RESET_ALL}") # Use TRADE level log

            # Priority 2: Stochastic Reversal Confirmation from Overbought
            elif stoch_k_prev >= overbought_level and stoch_k < overbought_level:
                exit_reason = f"Exit Signal: Stoch Reversal Confirmation (K {stoch_k_prev:.1f} -> {stoch_k:.1f} crossed below {overbought_level})"
                logger.trade(f"{Fore.YELLOW}{exit_reason} detected for LONG position.{Style.RESET_ALL}") # Use TRADE level log

            # Informational Logging: Stochastic is in the zone but hasn't reversed yet
            elif stoch_k >= overbought_level:
                logger.debug(f"Exit Check (Long): Stoch K ({stoch_k:.1f}) >= Overbought ({overbought_level}), awaiting cross below for exit signal.")

        elif position_side == "short":
            # Priority 1: EMA Bullish Cross
            if ema_bullish_cross:
                exit_reason = "Exit Signal: EMA Bullish Cross"
                logger.trade(f"{Fore.YELLOW}{exit_reason} detected for SHORT position.{Style.RESET_ALL}") # Use TRADE level log

            # Priority 2: Stochastic Reversal Confirmation from Oversold
            elif stoch_k_prev <= oversold_level and stoch_k > oversold_level:
                exit_reason = f"Exit Signal: Stoch Reversal Confirmation (K {stoch_k_prev:.1f} -> {stoch_k:.1f} crossed above {oversold_level})"
                logger.trade(f"{Fore.YELLOW}{exit_reason} detected for SHORT position.{Style.RESET_ALL}") # Use TRADE level log

            # Informational Logging: Stochastic is in the zone but hasn't reversed yet
            elif stoch_k <= oversold_level:
                logger.debug(f"Exit Check (Short): Stoch K ({stoch_k:.1f}) <= Oversold ({oversold_level}), awaiting cross above for exit signal.")

        # Return the reason string if an exit condition was met, otherwise None
        return exit_reason
    # --- End updated check_exit_signals ---


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
        # --- Input Validation ---
        if atr.is_nan() or atr <= 0:
            logger.error(f"Invalid ATR ({atr}) for parameter calculation.")
            return None
        if total_equity.is_nan() or total_equity <= 0:
            logger.error(f"Invalid equity ({total_equity}) for parameter calculation.")
            return None
        if current_price.is_nan() or current_price <= 0:
            logger.error(f"Invalid current price ({current_price}) for parameter calculation.")
            return None
        if not self.market_info or "tick_size" not in self.market_info or "contract_size" not in self.market_info:
             logger.error("Market info (tick_size, contract_size) missing for parameter calculation.")
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

            if sl_price <= 0:
                logger.error(f"Calculated SL price ({sl_price:.{DEFAULT_PRICE_DP}f}) is invalid (<=0).")
                return None

            sl_distance_price = (current_price - sl_price).copy_abs()
            min_tick_size = self.market_info['tick_size']
            if sl_distance_price < min_tick_size:
                sl_distance_price = min_tick_size
                if side == "buy": sl_price = current_price - sl_distance_price
                else: sl_price = current_price + sl_distance_price
                logger.warning(f"Initial SL distance < tick size. Adjusted SL price to {sl_price:.{DEFAULT_PRICE_DP}f} (dist {sl_distance_price}).")
                if sl_price <= 0:
                     logger.error(f"Adjusted SL price ({sl_price:.{DEFAULT_PRICE_DP}f}) is still invalid (<=0).")
                     return None

            if sl_distance_price <= 0:
                logger.error(f"Calculated SL distance ({sl_distance_price}) is invalid (<=0).")
                return None

            # --- Quantity Calculation ---
            contract_size = self.market_info['contract_size']
            value_per_point: Decimal
            if self.config.market_type == "inverse":
                 if current_price <= 0:
                      logger.error("Cannot calculate inverse contract value: Current price is zero/negative.")
                      return None
                 value_per_point = contract_size / current_price
            else: # Linear/Swap
                 value_per_point = contract_size

            if value_per_point <= 0:
                 logger.error(f"Calculated invalid value_per_point ({value_per_point}).")
                 return None

            risk_per_contract = sl_distance_price * value_per_point
            if risk_per_contract <= 0:
                logger.error(f"Calculated zero or negative risk per contract ({risk_per_contract}).")
                return None

            quantity = risk_amount_per_trade / risk_per_contract
            quantity_str = self.exchange_manager.format_amount(quantity, rounding_mode=ROUND_DOWN)
            quantity_decimal = safe_decimal(quantity_str)

            if quantity_decimal.is_nan() or quantity_decimal <= 0:
                 logger.error(f"Calculated quantity ({quantity_str}) is invalid or zero.")
                 return None

            min_order_size = self.market_info.get('min_order_size', Decimal('NaN'))
            if not min_order_size.is_nan() and quantity_decimal < min_order_size:
                logger.error(f"Calculated qty {quantity_decimal.normalize()} < minimum {min_order_size.normalize()}.")
                return None

            # --- Take Profit Calculation ---
            tp_price: Optional[Decimal] = None
            if self.config.tp_atr_multiplier > 0:
                tp_distance_atr = atr * self.config.tp_atr_multiplier
                if side == "buy": tp_price = current_price + tp_distance_atr
                else: tp_price = current_price - tp_distance_atr
                if tp_price <= 0:
                    logger.warning(f"Calculated TP price ({tp_price:.{DEFAULT_PRICE_DP}f}) invalid (<=0). Disabling TP.")
                    tp_price = None

            # --- Trailing Stop Calculation (Distance) ---
            tsl_distance_price = current_price * (self.config.trailing_stop_percent / 100)
            if tsl_distance_price < min_tick_size:
                 tsl_distance_price = min_tick_size
            tsl_distance_str = self.exchange_manager.format_price(tsl_distance_price)
            tsl_distance_decimal = safe_decimal(tsl_distance_str)
            if tsl_distance_decimal.is_nan() or tsl_distance_decimal <= 0:
                 logger.warning(f"Calculated invalid TSL distance ({tsl_distance_str}). TSL might fail.")
                 tsl_distance_decimal = Decimal('NaN')

            # --- Format Final SL/TP Prices ---
            sl_price_str = self.exchange_manager.format_price(sl_price)
            sl_price_decimal = safe_decimal(sl_price_str)
            tp_price_decimal: Optional[Decimal] = None
            if tp_price is not None:
                 tp_price_str = self.exchange_manager.format_price(tp_price)
                 tp_price_decimal = safe_decimal(tp_price_str)
                 if tp_price_decimal.is_nan() or tp_price_decimal <= 0:
                      logger.warning(f"Failed to format valid TP price ({tp_price_str}). Disabling TP.")
                      tp_price_decimal = None

            params_out: Dict[str, Optional[Decimal]] = {
                "qty": quantity_decimal,
                "sl_price": sl_price_decimal,
                "tp_price": tp_price_decimal, # Will be None if disabled or invalid
                "tsl_distance": tsl_distance_decimal, # Price difference for TSL
            }

            log_tp_str = f"{params_out['tp_price'].normalize()}" if params_out['tp_price'] else "None"
            log_tsl_str = f"{params_out['tsl_distance'].normalize()}" if params_out['tsl_distance'] and not params_out['tsl_distance'].is_nan() else "Invalid"
            logger.info(
                f"Trade Params Calculated: Side={side.upper()}, "
                f"Qty={params_out['qty'].normalize()}, "
                f"Entry~={current_price.normalize()}, "
                f"SL={params_out['sl_price'].normalize()}, "
                f"TP={log_tp_str}, "
                f"TSLDist~={log_tsl_str}, "
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
                "timeInForce": "ImmediateOrCancel", # Try IOC for market orders
            }
            amount_float = float(final_qty_decimal)

            order = fetch_with_retries(
                self.exchange.create_market_order,
                symbol=symbol,
                side=side,
                amount=amount_float,
                params=params,
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
            )

            if order is None:
                logger.error(f"{Fore.RED}Market order submission failed after retries (returned None).{Style.RESET_ALL}")
                return None

            order_id = order.get("id", "[N/A]")
            order_status = order.get("status", "[unknown]")
            filled_qty_str = order.get("filled", "0")
            avg_fill_price_str = order.get("average", "0")
            filled_qty = safe_decimal(filled_qty_str)
            avg_fill_price = safe_decimal(avg_fill_price_str)

            avg_price_log_str = '[N/A]'
            if not avg_fill_price.is_nan() and avg_fill_price > 0:
                try:
                    avg_price_log_str = avg_fill_price.normalize()
                except (InvalidOperation, ValueError):
                    logger.warning(f"Could not normalize avg_fill_price {avg_fill_price} for logging.")
                    avg_price_log_str = str(avg_fill_price)

            logger.trade(
                f"{Fore.GREEN}{Style.BRIGHT}Market order submitted: ID {order_id}, Side {side.upper()}, Qty {final_qty_decimal.normalize()}, Status: {order_status}, Filled: {filled_qty.normalize()}, AvgPx: {avg_price_log_str}{Style.RESET_ALL}"
            )
            termux_notify(
                f"{symbol} Order Submitted", f"Market {side.upper()} {final_qty_decimal.normalize()} ID:{order_id}"
            )

            if order_status in ["rejected", "canceled", "expired"]:
                 logger.error(f"{Fore.RED}Market order {order_id} was {order_status}. Check exchange reason if available in order info: {order.get('info')}{Style.RESET_ALL}")
                 return None

            # Short delay, robust checks done later by _verify_position_state
            logger.debug(f"Waiting short delay ({self.config.order_check_delay_seconds}s) after order {order_id}...")
            time.sleep(self.config.order_check_delay_seconds)

            return order # Return submitted order info

        except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e:
            logger.error(f"{Fore.RED}Order placement failed ({type(e).__name__}): {e}{Style.RESET_ALL}")
            termux_notify(f"{symbol} Order FAILED", f"Market {side.upper()} failed: {str(e)[:50]}")
            return None
        except Exception as e:
            logger.error(f"{Fore.RED}Unexpected error placing market order: {e}{Style.RESET_ALL}", exc_info=True)
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

        # Use the parameter formatting helper
        sl_str = self.exchange_manager._format_v5_param(sl_price, "price", allow_zero=True) or "0"
        tp_str = self.exchange_manager._format_v5_param(tp_price, "price", allow_zero=True) or "0"
        # TSL distance requires formatting as price difference, allow_zero depends on API clear logic (False safer)
        tsl_distance_str = self.exchange_manager._format_v5_param(tsl_distance, "distance", allow_zero=False)
        # TSL activation price needs price formatting, disallow zero
        tsl_activation_price_str = self.exchange_manager._format_v5_param(tsl_activation_price, "price", allow_zero=False)


        # --- Prepare Base Parameters for V5 private_post_position_set_trading_stop ---
        params: Dict[str, Any] = {
            "category": self.config.bybit_v5_category,
            "symbol": market_id,
            "positionIdx": self.config.position_idx,
            "tpslMode": V5_TPSL_MODE_FULL,
            "stopLoss": "0", "takeProfit": "0", "trailingStop": "0", "activePrice": "0", # Defaults
            "slTriggerBy": self.config.sl_trigger_by,
            "tpTriggerBy": self.config.sl_trigger_by, # Use same trigger for TP as SL
            "tslTriggerBy": self.config.tsl_trigger_by,
        }

        action_desc = ""
        new_tracker_state: Optional[str] = None

        # --- Logic Branch: Activate TSL ---
        if is_tsl and tsl_distance_str and tsl_activation_price_str:
            # _format_v5_param already handled validation/formatting
            params["trailingStop"] = tsl_distance_str
            params["activePrice"] = tsl_activation_price_str
            params["stopLoss"] = "0" # Clear fixed SL/TP
            params["takeProfit"] = "0"
            action_desc = f"ACTIVATE TSL (Dist: {tsl_distance_str}, ActPx: {tsl_activation_price_str})"
            new_tracker_state = "ACTIVE_TSL"
            logger.debug(f"TSL Params Prepared: {params}")

        # --- Logic Branch: Set Fixed SL/TP ---
        elif not is_tsl and (sl_str != "0" or tp_str != "0"):
            params["stopLoss"] = sl_str
            params["takeProfit"] = tp_str
            params["trailingStop"] = "0" # Clear TSL
            params["activePrice"] = "0"
            action_desc = f"SET SL={params['stopLoss']} TP={params['takeProfit']}"
            new_tracker_state = "ACTIVE_SLTP"

        # --- Logic Branch: Clear All Stops ---
        else:
            action_desc = "CLEAR SL/TP/TSL"
            # Defaults already set params to clear all
            new_tracker_state = None
            logger.debug(f"Clearing all stops for {position_side.upper()} position.")

        # --- Execute API Call ---
        symbol = self.config.symbol # Use the display symbol for logging
        logger.trade(f"{Fore.CYAN}Attempting to {action_desc} for {position_side.upper()} {symbol}...{Style.RESET_ALL}")

        # --- Corrected Private Method Call ---
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
            )
            # logger.debug(f"SetTradingStop Response: {response}") # Verbose

            # --- Process Response ---
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
                termux_notify(f"{symbol} Protection FAILED", f"{action_desc[:30]} {position_side.upper()} failed: {ret_msg[:50]}")
                return False
        except Exception as e:
            logger.error(f"{Fore.RED}Unexpected error during {action_desc} for {position_side.upper()} {symbol}: {e}{Style.RESET_ALL}", exc_info=True)
            termux_notify(f"{symbol} Protection ERROR", f"{action_desc[:30]} {position_side.upper()} error.")
            return False

    # --- Enhanced Position Verification Method (Snippet 5) ---
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

            # --- Check against expected state ---
            is_currently_flat = True
            actual_side = None
            actual_qty = Decimal("0")
            active_pos_data = None

            long_pos_data = current_positions.get("long")
            short_pos_data = current_positions.get("short")

            if long_pos_data:
                 long_qty = safe_decimal(long_pos_data.get("qty", "0"))
                 if long_qty.copy_abs() >= POSITION_QTY_EPSILON:
                     is_currently_flat = False
                     actual_side = "long"
                     actual_qty = long_qty
                     active_pos_data = long_pos_data
            # Check short only if not actively long
            if not active_pos_data and short_pos_data:
                 short_qty = safe_decimal(short_pos_data.get("qty", "0"))
                 if short_qty.copy_abs() >= POSITION_QTY_EPSILON:
                     is_currently_flat = False
                     actual_side = "short"
                     actual_qty = short_qty
                     active_pos_data = short_pos_data

            # --- Evaluate success based on expectation ---
            verification_met = False
            if expected_side is None: # Expecting flat
                verification_met = is_currently_flat
                log_msg = f"Expected FLAT, Actual: {'FLAT' if is_currently_flat else f'{actual_side.upper()} Qty={actual_qty.normalize()}'}"
            elif expected_side == actual_side: # Expecting a specific side
                qty_met = actual_qty.copy_abs() >= expected_qty_min
                verification_met = qty_met
                log_msg = f"Expected {expected_side.upper()} (MinQty~{expected_qty_min}), Actual: {actual_side.upper()} Qty={actual_qty.normalize()} ({'OK' if qty_met else 'QTY MISMATCH'})"
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
    # --- End Enhanced Position Verification Method ---

    # --- Updated place_risked_market_order using _verify_position_state ---
    def place_risked_market_order(
        self,
        side: str, # 'buy' or 'sell'
        atr: Decimal,
        total_equity: Decimal,
        current_price: Decimal,
    ) -> bool:
        """Calculates parameters, places market order, verifies position, and sets initial SL/TP."""
        if not self.exchange or not self.market_info: return False
        if side not in ["buy", "sell"]: logger.error(f"Invalid side '{side}'."); return False
        if atr.is_nan() or atr <= 0: logger.error("Entry Aborted: Invalid ATR."); return False
        if total_equity is None or total_equity.is_nan() or total_equity <= 0: logger.error("Entry Aborted: Invalid Equity."); return False
        if current_price.is_nan() or current_price <= 0: logger.error("Entry Aborted: Invalid Price."); return False

        position_side = "long" if side == "buy" else "short"
        logger.info(f"{Fore.MAGENTA}{Style.BRIGHT}--- Initiating Entry Sequence for {position_side.upper()} ---{Style.RESET_ALL}")

        # 1. Calculate Trade Parameters
        logger.debug("Calculating trade parameters...")
        trade_params = self._calculate_trade_parameters(side, atr, total_equity, current_price)
        if not trade_params or not trade_params.get("qty"): # Check qty exists
            logger.error("Entry Aborted: Failed to calculate valid trade parameters (qty missing?).")
            return False

        qty_to_order = trade_params["qty"]
        initial_sl_price = trade_params.get("sl_price") # Can be None if calc fails, though unlikely if qty valid
        initial_tp_price = trade_params.get("tp_price") # Can be None

        # Validate SL price before proceeding
        if initial_sl_price is None or initial_sl_price.is_nan() or initial_sl_price <= 0:
             logger.error(f"Entry Aborted: Invalid SL price ({initial_sl_price}) calculated.")
             return False

        # 2. Execute Market Order
        logger.debug(f"Executing market {side} order for {qty_to_order.normalize()}...")
        order_info = self._execute_market_order(side, qty_to_order)
        if not order_info:
            logger.error("Entry Aborted: Market order execution failed or rejected.")
            self._handle_entry_failure(side, qty_to_order) # Attempt cleanup
            return False
        order_id = order_info.get("id", "[N/A]")

        # 3. Verify Position Establishment using _verify_position_state
        logger.info(f"Verifying position establishment after market order {order_id}...")
        # Define verification parameters
        min_expected_qty = qty_to_order * Decimal("0.90") # Allow 10% slippage/fee diff? Adjust as needed.
        verification_ok, final_pos_state = self._verify_position_state(
            expected_side=position_side,
            expected_qty_min=min_expected_qty,
            max_attempts=5, # More attempts for verification
            delay_seconds=self.config.order_check_delay_seconds + 0.5, # Use configured delay + buffer
            action_context=f"Post-{position_side.upper()}-Entry"
        )

        if not verification_ok:
            logger.error(f"{Fore.RED}Entry Failed: Position verification FAILED after market order {order_id}. Manual check required! Attempting cleanup...{Style.RESET_ALL}")
            self._handle_entry_failure(side, qty_to_order) # Attempt cleanup
            return False

        # Position verified, extract details from the final verified state
        active_pos_data = final_pos_state.get(position_side) if final_pos_state else None
        if not active_pos_data: # Should not happen if verification_ok is True, but safeguard
            logger.error(f"{Fore.RED}Internal Error: Position verified OK but data missing for {position_side} in final state {final_pos_state}. Halting entry.{Style.RESET_ALL}")
            self._handle_entry_failure(side, qty_to_order)
            return False

        filled_qty = safe_decimal(active_pos_data.get("qty", "0"))
        avg_entry_price = safe_decimal(active_pos_data.get("entry_price", "NaN"))

        # Log confirmation with verified details
        logger.info(
            f"{Fore.GREEN}{Style.BRIGHT}Position {position_side.upper()} confirmed via verification: Qty={filled_qty.normalize()}, AvgEntry={avg_entry_price.normalize() if not avg_entry_price.is_nan() else '[N/A]'}{Style.RESET_ALL}"
        )

        # Handle partial fill logging (already done in verification logic)
        if filled_qty < qty_to_order * Decimal("0.99"): # Log if less than 99% filled
             logger.warning(f"{Fore.YELLOW}Filled qty {filled_qty.normalize()} less than ordered {qty_to_order.normalize()}.{Style.RESET_ALL}")

        # 4. Set Initial Position SL/TP
        logger.info(f"Setting initial SL/TP for {position_side.upper()} position...")
        set_stops_ok = self._set_position_protection(
            position_side, sl_price=initial_sl_price, tp_price=initial_tp_price
        )

        if not set_stops_ok:
            logger.error(f"{Fore.RED}Entry Failed: Failed to set initial SL/TP after establishing position {position_side.upper()}. Attempting emergency close!{Style.RESET_ALL}")
            self.close_position(position_side, filled_qty, reason="EmergencyClose:FailedStopSet")
            return False

        # 5. Log Entry to Journal
        if self.config.enable_journaling:
            if avg_entry_price.is_nan():
                logger.warning("Logging entry to journal with N/A entry price.")
            self.log_trade_entry_to_journal(side, filled_qty, avg_entry_price, order_id)

        logger.info(f"{Fore.GREEN}{Style.BRIGHT}--- Entry Sequence for {position_side.upper()} Completed Successfully ---{Style.RESET_ALL}")
        return True
    # --- End updated place_risked_market_order ---

    def manage_trailing_stop(
        self,
        position_side: str, # 'long' or 'short'
        entry_price: Decimal,
        current_price: Decimal,
        atr: Decimal,
    ) -> None:
        """Checks if TSL activation conditions are met and activates it using V5 position TSL."""
        if not self.exchange or not self.market_info: return
        tracker_key = position_side.lower()

        current_protection_state = self.protection_tracker.get(tracker_key)
        if current_protection_state != "ACTIVE_SLTP":
            log_msg = f"TSL already active ({current_protection_state})." if current_protection_state == "ACTIVE_TSL" else f"No active SL/TP ({current_protection_state}), cannot activate TSL yet."
            logger.debug(f"TSL Check ({position_side.upper()}): {log_msg}")
            return

        if atr.is_nan() or atr <= 0: logger.debug("Invalid ATR for TSL."); return
        if entry_price.is_nan() or entry_price <= 0: logger.debug("Invalid entry price for TSL."); return
        if current_price.is_nan() or current_price <= 0: logger.debug("Invalid current price for TSL."); return

        try:
            activation_distance_atr = atr * self.config.tsl_activation_atr_multiplier
            activation_price: Decimal
            if position_side == "long": activation_price = entry_price + activation_distance_atr
            else: activation_price = entry_price - activation_distance_atr

            if activation_price.is_nan() or activation_price <= 0:
                logger.warning(f"Calculated invalid TSL activation price ({activation_price}). Skipping TSL."); return

            tsl_distance_price = current_price * (self.config.trailing_stop_percent / 100)
            min_tick_size = self.market_info.get('tick_size', Decimal('1e-8'))
            if tsl_distance_price < min_tick_size:
                logger.debug(f"TSL distance ({tsl_distance_price}) < min tick {min_tick_size}. Adjusting.")
                tsl_distance_price = min_tick_size
            if tsl_distance_price <= 0:
                 logger.warning(f"Invalid TSL distance ({tsl_distance_price}) after adjustment. Skipping TSL."); return

            should_activate_tsl = False
            if position_side == "long" and current_price >= activation_price: should_activate_tsl = True
            elif position_side == "short" and current_price <= activation_price: should_activate_tsl = True

            if should_activate_tsl:
                logger.trade(f"{Fore.MAGENTA}TSL Activation condition met for {position_side.upper()}!{Style.RESET_ALL}")
                logger.trade(f"  Entry={entry_price.normalize()}, Current={current_price.normalize()}, ActTarget~={activation_price.normalize():.{DEFAULT_PRICE_DP}f}")
                logger.trade(f"Attempting TSL activation (Dist ~{tsl_distance_price.normalize()}, ActPx {activation_price.normalize():.{DEFAULT_PRICE_DP}f})...")

                activation_success = self._set_position_protection(
                    position_side,
                    is_tsl=True,
                    tsl_distance=tsl_distance_price,
                    tsl_activation_price=activation_price,
                )
                if activation_success:
                    logger.trade(f"{Fore.GREEN}{Style.BRIGHT}TSL activated successfully for {position_side.upper()}.{Style.RESET_ALL}")
                else:
                    logger.error(f"{Fore.RED}Failed to activate TSL for {position_side.upper()} via API.{Style.RESET_ALL}")
            else:
                logger.debug(f"TSL check ({position_side.upper()}): Condition not met (Current: {current_price.normalize()}, Target: ~{activation_price.normalize():.{DEFAULT_PRICE_DP}f})")

        except Exception as e:
            logger.error(f"Error managing TSL for {position_side.upper()}: {e}", exc_info=True)

    # --- Updated close_position using _verify_position_state ---
    def close_position(
        self, position_side: str, qty_to_close: Decimal, reason: str = "Signal"
    ) -> bool:
        """Closes the specified position: clears stops first, places closing order, verifies closure."""
        if not self.exchange or not self.market_info: return False
        if position_side not in ["long", "short"]: logger.error(f"Invalid side '{position_side}'."); return False

        if qty_to_close.is_nan() or qty_to_close.copy_abs() < POSITION_QTY_EPSILON:
            logger.warning(f"Close requested for zero/negligible qty ({qty_to_close}). Skipping close for {position_side}.")
            self.protection_tracker[position_side] = None
            return True

        symbol = self.config.symbol
        closing_order_side = "sell" if position_side == "long" else "buy"
        tracker_key = position_side.lower()

        logger.trade(f"{Fore.YELLOW}Attempting to CLOSE {position_side.upper()} pos ({qty_to_close.normalize()} {symbol}) | Reason: {reason}...{Style.RESET_ALL}")

        # 1. Clear Existing Position Protection
        logger.info(f"Clearing protection for {position_side.upper()} before closing...")
        clear_stops_ok = self._set_position_protection(
            position_side, sl_price=None, tp_price=None, is_tsl=False
        )
        if not clear_stops_ok:
            logger.warning(f"{Fore.YELLOW}Failed to confirm protection clear via API for {position_side.upper()}. Proceeding cautiously...{Style.RESET_ALL}")
        else:
            logger.info(f"Protection cleared successfully via API for {position_side.upper()}.")
        self.protection_tracker[tracker_key] = None # Clear tracker state

        # 2. Place Closing Market Order
        logger.info(f"Submitting MARKET {closing_order_side.upper()} order to close {position_side.upper()}...")
        close_order_info = self._execute_market_order(closing_order_side, qty_to_close)

        if not close_order_info:
            logger.error(f"{Fore.RED}Failed to submit closing market order for {position_side.upper()}. MANUAL INTERVENTION REQUIRED!{Style.RESET_ALL}")
            termux_notify(f"{symbol} CLOSE FAILED", f"Market {closing_order_side.upper()} order failed!")
            return False

        close_order_id = close_order_info.get("id", "[N/A]")
        avg_close_price_str = close_order_info.get("average")
        avg_close_price = safe_decimal(avg_close_price_str, default=Decimal("NaN"))
        logger.trade(f"{Fore.YELLOW}Closing market order ({close_order_id}) submitted for {position_side.upper()}. AvgClosePrice: {avg_close_price.normalize() if not avg_close_price.is_nan() else '[Pending/N/A]'}{Style.RESET_ALL}")
        termux_notify(f"{symbol} Position Closing", f"{position_side.upper()} close order {close_order_id} submitted.")

        # 3. Verify Position Closed using _verify_position_state
        logger.info(f"Verifying position closure...")
        verification_ok, final_pos_state = self._verify_position_state(
            expected_side=None, # Expecting flat
            max_attempts=5,
            delay_seconds=self.config.order_check_delay_seconds + 1.0, # Wait slightly longer for close verification
            action_context=f"Post-{position_side.upper()}-Close"
        )

        # 4. Log Exit to Journal (regardless of verification outcome, use available info)
        if self.config.enable_journaling:
            self.log_trade_exit_to_journal(
                position_side, qty_to_close, avg_close_price, close_order_id, reason
            )

        # 5. Handle Verification Outcome
        if not verification_ok:
            lingering_side = None
            lingering_qty = Decimal(0)
            if final_pos_state: # Check the last known state if verification failed
                if final_pos_state.get("long") and safe_decimal(final_pos_state["long"].get("qty")).copy_abs() >= POSITION_QTY_EPSILON:
                    lingering_side, lingering_qty = "long", safe_decimal(final_pos_state["long"].get("qty"))
                elif final_pos_state.get("short") and safe_decimal(final_pos_state["short"].get("qty")).copy_abs() >= POSITION_QTY_EPSILON:
                    lingering_side, lingering_qty = "short", safe_decimal(final_pos_state["short"].get("qty"))

            logger.error(
                f"{Fore.RED}Position {position_side.upper()} closure verification FAILED. "
                f"Last state showed: {'FLAT' if not lingering_side else f'{lingering_side.upper()} Qty={lingering_qty.normalize()}'}. MANUAL CHECK REQUIRED!{Style.RESET_ALL}"
            )
            termux_notify(f"{symbol} CLOSE VERIFY FAILED", f"{position_side.upper()} may still be open!")
            return False # Indicate closure verification failed
        else:
            logger.trade(f"{Fore.GREEN}{Style.BRIGHT}Position {position_side.upper()} confirmed closed via verification.{Style.RESET_ALL}")
            return True # Closure confirmed
    # --- End updated close_position ---

    def _handle_entry_failure(
        self, failed_entry_side: str, attempted_qty: Decimal
    ):
        """Attempts to close any potentially opened position after a failed entry sequence step."""
        logger.warning(
            f"{Fore.YELLOW}Handling potential entry failure for {failed_entry_side.upper()} (intended qty: {attempted_qty.normalize()}). Checking for lingering position...{Style.RESET_ALL}"
        )
        position_side_to_check = "long" if failed_entry_side == "buy" else "short"

        # Wait briefly before checking
        time.sleep(self.config.order_check_delay_seconds + 1)
        logger.debug(f"Checking position status after {failed_entry_side} entry failure...")

        # Use verification logic to check state robustly
        verification_ok, current_positions = self._verify_position_state(
            expected_side=None, # We don't know what to expect, just get the state
            max_attempts=2, # Quick check
            delay_seconds=1,
            action_context=f"Entry-Fail-Check-{failed_entry_side.upper()}"
        )

        if current_positions is None:
            # If fetch fails during failure handling, requires manual check
            logger.error(f"{Fore.RED}Could not fetch positions during entry failure handling for {failed_entry_side}. MANUAL CHECK REQUIRED!{Style.RESET_ALL}")
            termux_notify(f"{self.config.symbol} Check Needed", "Failed pos check after entry fail")
            return

        lingering_pos_data = current_positions.get(position_side_to_check)
        if lingering_pos_data:
            current_qty = safe_decimal(lingering_pos_data.get("qty", "0"))
            if current_qty.copy_abs() >= POSITION_QTY_EPSILON:
                logger.error(f"{Fore.RED}Lingering {position_side_to_check.upper()} pos (Qty: {current_qty.normalize()}) found after entry fail. Attempting emergency close.{Style.RESET_ALL}")
                termux_notify(f"{self.config.symbol} Emergency Close", f"Lingering {position_side_to_check} pos")
                close_success = self.close_position(position_side_to_check, current_qty, reason="EmergencyClose:EntryFail")
                if close_success:
                    logger.info(f"Emergency close order submitted/confirmed for lingering {position_side_to_check} position.")
                else:
                    logger.critical(f"{Fore.RED}{Style.BRIGHT}EMERGENCY CLOSE FAILED for {position_side_to_check.upper()}. MANUAL INTERVENTION URGENT!{Style.RESET_ALL}")
                    termux_notify(f"{self.config.symbol} URGENT CHECK", f"Emergency close FAILED!")
            else:
                logger.info(f"Lingering pos ({position_side_to_check}) found but qty negligible ({current_qty}). No emergency close needed.")
                self.protection_tracker[position_side_to_check] = None
        else:
            logger.info(f"No lingering {position_side_to_check} position detected after entry failure.")
            self.protection_tracker[position_side_to_check] = None

    def _write_journal_row(self, data: Dict[str, Any]):
        """Helper function to write a row to the CSV journal."""
        if not self.config.enable_journaling: return
        file_path = Path(self.config.journal_file_path)
        file_exists = file_path.is_file()
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open("a", newline="", encoding="utf-8") as csvfile:
                fieldnames = ["TimestampUTC", "Symbol", "Action", "Side", "Quantity", "AvgPrice", "OrderID", "Reason", "Notes"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
                if not file_exists or file_path.stat().st_size == 0: writer.writeheader()

                row_to_write = {}
                for field in fieldnames:
                    value = data.get(field)
                    if isinstance(value, Decimal): row_to_write[field] = 'NaN' if value.is_nan() else value.normalize()
                    elif value is None: row_to_write[field] = 'N/A'
                    else: row_to_write[field] = str(value)
                row_to_write['Notes'] = data.get('Notes', '')
                writer.writerow(row_to_write)
            logger.debug(f"Trade {data.get('Action', '').lower()} logged to {file_path}")
        except IOError as e: logger.error(f"I/O error writing {data.get('Action', '').lower()} to journal {file_path}: {e}")
        except Exception as e: logger.error(f"Unexpected error writing {data.get('Action', '').lower()} to journal: {e}", exc_info=True)

    def log_trade_entry_to_journal(self, side: str, qty: Decimal, avg_price: Decimal, order_id: Optional[str]):
        """Logs trade entry details to the CSV journal."""
        position_side = "long" if side == "buy" else "short"
        data = {
            "TimestampUTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "Symbol": self.config.symbol, "Action": "ENTRY", "Side": position_side.upper(),
            "Quantity": qty, "AvgPrice": avg_price, "OrderID": order_id, "Reason": "Strategy Signal",
        }
        self._write_journal_row(data)

    def log_trade_exit_to_journal(self, position_side: str, qty: Decimal, avg_price: Decimal, order_id: Optional[str], reason: str):
        """Logs trade exit details to the CSV journal."""
        data = {
            "TimestampUTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "Symbol": self.config.symbol, "Action": "EXIT", "Side": position_side.upper(),
            "Quantity": qty, "AvgPrice": avg_price, "OrderID": order_id, "Reason": reason,
        }
        self._write_journal_row(data)


# --- Status Display Class ---
class StatusDisplay:
    """Handles displaying the bot status using the Rich library."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self._default_price_dp = DEFAULT_PRICE_DP
        self._default_amount_dp = DEFAULT_AMOUNT_DP

    def _format_decimal(
        self, value: Optional[Decimal], precision: Optional[int] = None,
        default_precision: int = 2, add_commas: bool = False, highlight_negative: bool = False,
        style_override: Optional[str] = None
    ) -> Text:
        """Formats Decimal values for Rich display with enhanced styling."""
        if value is None or (isinstance(value, Decimal) and value.is_nan()):
            return Text("N/A", style="dim")

        dp = precision if precision is not None else default_precision
        try:
            quantizer = Decimal("1e-" + str(dp))
            formatted_value = value.quantize(quantizer, rounding=ROUND_HALF_EVEN)
            format_spec = f"{{:{',' if add_commas else ''}.{dp}f}}"
            display_str = format_spec.format(formatted_value)

            # Determine style
            if style_override:
                style = style_override
            elif highlight_negative and formatted_value < 0:
                style = "bright_red" # Use Rich color name
            elif highlight_negative and formatted_value > 0:
                 style = "bright_green" # Use Rich color name
            else:
                style = "white" # Default non-highlighted style

            return Text(display_str, style=style)
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.error(f"Error formatting decimal {value}: {e}")
            return Text("ERR", style="bold bright_red") # Use Rich color name

    def print_status_panel(
        self, cycle: int, timestamp: Optional[datetime], price: Optional[Decimal],
        indicators: Optional[Dict], positions: Optional[Dict], equity: Optional[Decimal],
        signals: Dict, protection_tracker: Dict, market_info: Optional[Dict]
    ):
        """Prints the status panel to the console using Rich Panel and Text."""
        price_dp = self._default_price_dp
        amount_dp = self._default_amount_dp
        if market_info and "precision_dp" in market_info:
             price_dp = market_info["precision_dp"].get("price", self._default_price_dp)
             amount_dp = market_info["precision_dp"].get("amount", self._default_amount_dp)

        panel_content = Text()
        ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S %Z") if timestamp else "[dim]Timestamp N/A[/]"
        title_text = f" Cycle {cycle} | {self.config.symbol} ({self.config.interval}) | {ts_str} "

        # --- Price & Equity ---
        price_text = self._format_decimal(price, precision=price_dp, default_precision=self._default_price_dp, style_override="bright_white")
        settle_curr = self.config.symbol.split(":")[-1] if ":" in self.config.symbol else "QUOTE"
        equity_text = self._format_decimal(equity, precision=2, default_precision=2, add_commas=True, style_override="bright_yellow")
        panel_content.append("Price: ", style="bold bright_cyan")
        panel_content.append(price_text)
        panel_content.append(" | ", style="dim")
        panel_content.append("Equity: ", style="bold bright_yellow")
        panel_content.append(equity_text)
        panel_content.append(f" {settle_curr}\n", style="bright_yellow")
        panel_content.append("---\n", style="dim")

        # --- Indicators ---
        panel_content.append("Indicators: ", style="bold bright_cyan")
        if indicators:
            ind_parts = []
            def fmt_ind(key: str, prec: int = 1, default_prec: int = 1, style: str = "white") -> Text:
                 val = indicators.get(key)
                 dec_val = safe_decimal(val) if isinstance(val, (Decimal, int, float, str)) else Decimal('NaN') # Added str
                 return self._format_decimal(dec_val, precision=prec, default_precision=default_prec, style_override=style)

            # EMAs
            ema_text = Text("EMA(F/S/T): ")
            ema_text.append(fmt_ind('fast_ema', prec=price_dp, default_prec=self._default_price_dp, style="cyan"))
            ema_text.append("/")
            ema_text.append(fmt_ind('slow_ema', prec=price_dp, default_prec=self._default_price_dp, style="magenta"))
            ema_text.append("/")
            ema_text.append(fmt_ind('trend_ema', prec=price_dp, default_prec=self._default_price_dp, style="yellow"))
            ind_parts.append(ema_text)

            # Stochastic
            stoch_text = Text("Stoch(K/D/PrevK): ")
            stoch_text.append(fmt_ind('stoch_k', prec=1, style="bright_blue"))
            stoch_text.append("/")
            stoch_text.append(fmt_ind('stoch_d', prec=1, style="blue"))
            stoch_text.append("/")
            stoch_text.append(fmt_ind('stoch_k_prev', prec=1, style="dim blue")) # Display Prev K
            if indicators.get('stoch_kd_bullish'): stoch_text.append(" [b green]▲[/]", style="green")
            elif indicators.get('stoch_kd_bearish'): stoch_text.append(" [b red]▼[/]", style="red")
            ind_parts.append(stoch_text)

            # ATR
            atr_text = Text(f"ATR({indicators.get('atr_period', '?')}): ")
            atr_text.append(fmt_ind('atr', prec=price_dp + 1, default_prec=self._default_price_dp + 1, style="bright_magenta")) # More precision for ATR
            ind_parts.append(atr_text)

            # ADX
            adx_val = indicators.get('adx', Decimal('NaN'))
            adx_val_dec = safe_decimal(adx_val) # Ensure Decimal
            adx_is_valid = not adx_val_dec.is_nan()
            adx_text = Text(f"ADX({self.config.adx_period}): ")
            adx_text.append(fmt_ind('adx', prec=1, style="yellow" if adx_is_valid and adx_val_dec > self.config.min_adx_level else "dim yellow"))
            adx_text.append(f" [+DI:")
            adx_text.append(fmt_ind('pdi', prec=1, style="bright_green"))
            adx_text.append(f" -DI:")
            adx_text.append(fmt_ind('mdi', prec=1, style="bright_red"))
            adx_text.append("]")
            ind_parts.append(adx_text)

            separator = Text(" | ", style="dim")
            for i, part in enumerate(ind_parts):
                panel_content.append(part)
                if i < len(ind_parts) - 1: panel_content.append(separator)
            panel_content.append("\n")
        else:
            panel_content.append("[dim]Calculating...[/]\n", style="dim")
        panel_content.append("---\n", style="dim")

        # --- Position ---
        panel_content.append("Position: ", style="bold bright_cyan")
        pos_text = Text("FLAT", style="bold bright_green")
        active_position_data = None
        position_side_str = None

        if positions:
             long_pos = positions.get("long"); short_pos = positions.get("short")
             if long_pos and long_pos.get('qty', Decimal(0)).copy_abs() >= POSITION_QTY_EPSILON:
                 active_position_data = long_pos; position_side_str = "long"
             elif short_pos and short_pos.get('qty', Decimal(0)).copy_abs() >= POSITION_QTY_EPSILON:
                 active_position_data = short_pos; position_side_str = "short"

        if active_position_data and position_side_str:
            pos_style = "bold bright_green" if position_side_str == "long" else "bold bright_red"
            pos_text = Text(f"{position_side_str.upper()}: ", style=pos_style)
            qty_text = self._format_decimal(active_position_data.get("qty"), precision=amount_dp, default_precision=self._default_amount_dp)
            pos_text.append(f"Qty="); pos_text.append(qty_text)
            entry_text = self._format_decimal(active_position_data.get("entry_price"), precision=price_dp, default_precision=self._default_price_dp)
            pos_text.append(" | Entry=", style="dim"); pos_text.append(entry_text)
            pnl_text = self._format_decimal(active_position_data.get("unrealized_pnl"), precision=4, default_precision=4, highlight_negative=True)
            pos_text.append(" | PnL=", style="dim"); pos_text.append(pnl_text)

            # Protection Status
            tracked_protection = protection_tracker.get(position_side_str)
            sl_from_pos = active_position_data.get("stop_loss_price")
            tp_from_pos = active_position_data.get("take_profit_price")
            tsl_active_from_pos = active_position_data.get("is_tsl_active", False)
            tsl_trigger_from_pos = active_position_data.get("tsl_trigger_price")

            pos_text.append(" | Prot: ", style="dim")
            prot_status_text = Text("None", style="dim"); prot_details = Text("")

            if tsl_active_from_pos:
                 prot_status_text = Text("TSL", style="bright_magenta")
                 tsl_trigger_text = self._format_decimal(tsl_trigger_from_pos, precision=price_dp, default_precision=self._default_price_dp)
                 prot_details = Text(" (Trig:", style="dim").append(tsl_trigger_text).append(")", style="dim")
                 if tracked_protection != "ACTIVE_TSL": prot_status_text.append("?", style="bright_yellow")
            elif sl_from_pos or tp_from_pos:
                 prot_status_text = Text("SL/TP", style="bright_yellow")
                 sl_text = self._format_decimal(sl_from_pos, precision=price_dp, default_precision=self._default_price_dp)
                 tp_text = self._format_decimal(tp_from_pos, precision=price_dp, default_precision=self._default_price_dp)
                 prot_details = Text(" (S:", style="dim").append(sl_text).append(" T:", style="dim").append(tp_text).append(")", style="dim")
                 if tracked_protection != "ACTIVE_SLTP": prot_status_text.append("?", style="bright_yellow")
            elif tracked_protection:
                 prot_status_text = Text(f"Tracked:{tracked_protection}", style="bright_yellow")
                 prot_details = Text(" (Pos:None?)", style="dim")

            pos_text.append(prot_status_text); pos_text.append(prot_details)

        panel_content.append(pos_text)
        panel_content.append("\n")
        panel_content.append("---\n", style="dim")

        # --- Signals ---
        panel_content.append("Signal: ", style="bold bright_cyan")
        sig_reason = signals.get("reason", "[dim]No signal info[/]")
        sig_style = "dim"
        if signals.get("long"): sig_style = "bold bright_green"
        elif signals.get("short"): sig_style = "bold bright_red"
        elif "Blocked" in str(sig_reason): sig_style = "yellow"
        elif "No Signal:" not in str(sig_reason) and "Initializing" not in str(sig_reason): sig_style = "white"

        wrapped_reason = "\n        ".join(textwrap.wrap(str(sig_reason), width=100))
        panel_content.append(Text(wrapped_reason, style=sig_style))

        console.print(
            Panel(panel_content, title=f"[bold bright_magenta]{title_text}[/]",
                  border_style="bright_blue", expand=False, padding=(1, 2))
        )


# --- Trading Bot Class ---
class TradingBot:
    """Main orchestrator for the trading bot."""

    def __init__(self):
        logger.info(
            f"{Fore.MAGENTA}{Style.BRIGHT}--- Initializing Pyrmethus v4.5.7 (Neon Nexus Edition) ---{Style.RESET_ALL}"
        )
        self.config = TradingConfig()
        self.exchange_manager = ExchangeManager(self.config)

        if not self.exchange_manager.exchange or not self.exchange_manager.market_info:
            logger.critical(f"{Fore.RED}{Style.BRIGHT}TradingBot init failed: ExchangeManager issues. Halting.{Style.RESET_ALL}")
            sys.exit(1)

        self.indicator_calculator = IndicatorCalculator(self.config)
        self.signal_generator = SignalGenerator(self.config)
        try:
            self.order_manager = OrderManager(self.config, self.exchange_manager)
        except ValueError as e:
            logger.critical(f"{Fore.RED}{Style.BRIGHT}TradingBot init failed: {e}. Halting.{Style.RESET_ALL}")
            sys.exit(1)
        self.status_display = StatusDisplay(self.config)
        self.shutdown_requested = False
        self._setup_signal_handlers()
        logger.info(f"{Fore.GREEN}{Style.BRIGHT}Pyrmethus components initialized successfully.{Style.RESET_ALL}")

    def _setup_signal_handlers(self):
        """Sets up OS signal handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGINT, self._signal_handler) # Handle Ctrl+C
            signal.signal(signal.SIGTERM, self._signal_handler) # Handle kill/system shutdown
            logger.debug("Signal handlers for SIGINT and SIGTERM set up.")
        except (ValueError, OSError, Exception) as e: # Catch more potential errors
             logger.warning(f"{Fore.YELLOW}Could not set signal handlers (OS/thread issue?): {e}{Style.RESET_ALL}")

    def _signal_handler(self, sig: int, frame: Optional[types.FrameType]):
        """Internal signal handler to initiate graceful shutdown."""
        if not self.shutdown_requested:
            sig_name = signal.Signals(sig).name if isinstance(sig, int) and sig in signal.Signals else str(sig)
            console.print(f"\n[bold yellow]Signal {sig_name} received. Initiating graceful shutdown...[/]")
            logger.warning(f"Signal {sig_name} received. Initiating graceful shutdown...")
            self.shutdown_requested = True
        else:
            logger.warning("Shutdown already in progress. Ignoring additional signal.")

    def run(self):
        """Starts the main trading loop."""
        self._display_startup_info()
        termux_notify(f"Pyrmethus Started", f"{self.config.symbol} @ {self.config.interval}")
        cycle_count = 0
        while not self.shutdown_requested:
            cycle_count += 1
            cycle_start_time = time.monotonic()
            logger.debug(f"{Fore.BLUE}--- Starting Cycle {cycle_count} ---{Style.RESET_ALL}")

            try:
                self.trading_spell_cycle(cycle_count)
            except KeyboardInterrupt:
                logger.warning("\nCtrl+C detected during main loop execution. Initiating shutdown.")
                self.shutdown_requested = True
                break
            except ccxt.AuthenticationError as e:
                logger.critical(f"{Fore.RED}{Style.BRIGHT}CRITICAL AUTH ERROR in cycle {cycle_count}: {e}. Halting.{Style.RESET_ALL}", exc_info=False)
                termux_notify("Pyrmethus CRITICAL ERROR", f"Auth failed: {e}")
                self.shutdown_requested = True
                break
            except SystemExit as e:
                 logger.warning(f"SystemExit called with code {e.code}. Terminating.")
                 self.shutdown_requested = True
                 break
            except Exception as e:
                logger.error(f"{Fore.RED}{Style.BRIGHT}Unhandled exception in main trading cycle {cycle_count}: {e}{Style.RESET_ALL}", exc_info=True)
                termux_notify("Pyrmethus Cycle Error", f"Exception cycle {cycle_count}. Check logs.")
                sleep_duration = self.config.loop_sleep_seconds * 2 # Longer sleep after error
            else:
                cycle_duration = time.monotonic() - cycle_start_time
                sleep_duration = max(0, self.config.loop_sleep_seconds - cycle_duration)
                logger.debug(f"Cycle {cycle_count} completed in {cycle_duration:.2f}s.")

            # Interruptible Sleep
            if not self.shutdown_requested and sleep_duration > 0:
                logger.debug(f"Sleeping for {sleep_duration:.2f} seconds...")
                sleep_end_time = time.monotonic() + sleep_duration
                try:
                    while time.monotonic() < sleep_end_time and not self.shutdown_requested:
                        time.sleep(min(0.5, sleep_duration)) # Sleep in small chunks to check flag
                except KeyboardInterrupt:
                    logger.warning("\nCtrl+C detected during sleep.")
                    self.shutdown_requested = True

            if self.shutdown_requested:
                logger.info("Shutdown requested, exiting main loop.")
                break

        self.graceful_shutdown()
        console.print(f"\n[bold bright_cyan]Pyrmethus {self.config.symbol} has returned to the ether.[/]")
        sys.exit(0)

    def trading_spell_cycle(self, cycle_count: int) -> None:
        """Executes one cycle of the trading logic: fetch, analyze, act, display."""
        start_time = time.monotonic()
        cycle_status = "OK"

        # 1. Fetch Data & Basic Info
        logger.debug("Fetching market data (OHLCV)...")
        df = self.exchange_manager.fetch_ohlcv()
        if df is None or df.empty:
            logger.error(f"{Fore.RED}Cycle Aborted (Cycle {cycle_count}): Market data fetch failed.{Style.RESET_ALL}")
            cycle_status = "FAIL:FETCH_OHLCV"
            self.status_display.print_status_panel(cycle_count, None, None, None, None, None, {"reason": cycle_status}, {}, self.exchange_manager.market_info)
            return

        try:
            last_candle = df.iloc[-1]
            current_price = safe_decimal(last_candle["close"])
            last_timestamp = df.index[-1].to_pydatetime()
            if current_price.is_nan() or current_price <= 0: raise ValueError(f"Invalid latest close: {current_price}")
            logger.debug(f"Latest Candle: {last_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}, Close={current_price.normalize()}")
        except (IndexError, KeyError, ValueError, TypeError) as e:
            logger.error(f"{Fore.RED}Cycle Aborted (Cycle {cycle_count}): Error processing latest candle: {e}{Style.RESET_ALL}", exc_info=False)
            cycle_status = "FAIL:PROCESS_CANDLE"
            self.status_display.print_status_panel(cycle_count, None, None, None, None, None, {"reason": cycle_status}, {}, self.exchange_manager.market_info)
            return

        # 2. Calculate Indicators
        logger.debug("Calculating indicators...")
        indicators = self.indicator_calculator.calculate_indicators(df)
        if indicators is None:
            logger.warning(f"{Fore.YELLOW}Indicator calculation failed (Cycle {cycle_count}).{Style.RESET_ALL}")
            cycle_status = "WARN:INDICATORS_FAILED"
        current_atr = indicators.get("atr", Decimal("NaN")) if indicators else Decimal("NaN")

        # 3. Get Current State
        logger.debug("Fetching balance and position state...")
        total_equity, _ = self.exchange_manager.get_balance()
        current_positions = self.exchange_manager.get_current_position()

        # Validate State
        can_run_trade_logic = True
        if total_equity is None or total_equity.is_nan() or total_equity <= 0:
            logger.error(f"{Fore.RED}Invalid equity ({total_equity}). Logic skipped (Cycle {cycle_count}).{Style.RESET_ALL}")
            cycle_status = "FAIL:FETCH_EQUITY"; can_run_trade_logic = False
        if current_positions is None:
            logger.error(f"{Fore.RED}Failed fetching positions. Logic skipped (Cycle {cycle_count}).{Style.RESET_ALL}")
            cycle_status = "FAIL:FETCH_POSITIONS"; can_run_trade_logic = False
        if current_atr.is_nan() or current_atr <= 0:
            # Only fail if indicators were expected to succeed
            if cycle_status != "WARN:INDICATORS_FAILED":
                logger.error(f"{Fore.RED}Invalid ATR ({current_atr}). Logic skipped (Cycle {cycle_count}).{Style.RESET_ALL}")
                cycle_status = "FAIL:INVALID_ATR"; can_run_trade_logic = False
            else:
                logger.warning(f"{Fore.YELLOW}Invalid ATR ({current_atr}) due to indicator failure. Logic skipped.{Style.RESET_ALL}")
                can_run_trade_logic = False


        # Prepare Data Snapshots for display
        protection_tracker_snapshot = copy.deepcopy(self.order_manager.protection_tracker)
        live_positions_state = current_positions if current_positions is not None else {"long": {}, "short": {}}
        final_positions_for_panel = copy.deepcopy(live_positions_state) # Start with current state
        signals: Dict[str, Union[bool, str]] = {"long": False, "short": False, "reason": f"Skipped ({cycle_status})"}

        # 4. Execute Core Trading Logic
        if can_run_trade_logic:
            logger.debug("Executing core trading logic...")
            # Extract Current Position Details from live_positions_state
            active_long_pos = live_positions_state.get("long", {}); active_short_pos = live_positions_state.get("short", {})
            long_qty = safe_decimal(active_long_pos.get("qty", "0")); short_qty = safe_decimal(active_short_pos.get("qty", "0"))
            long_entry = safe_decimal(active_long_pos.get("entry_price", "NaN")); short_entry = safe_decimal(active_short_pos.get("entry_price", "NaN"))
            has_long_pos = long_qty.copy_abs() >= POSITION_QTY_EPSILON; has_short_pos = short_qty.copy_abs() >= POSITION_QTY_EPSILON
            is_flat = not has_long_pos and not has_short_pos
            current_pos_side = "long" if has_long_pos else "short" if has_short_pos else None
            logger.debug(f"State Before Actions: Flat={is_flat}, Side={current_pos_side}, LongQty={long_qty.normalize()}, ShortQty={short_qty.normalize()}")

            # Manage Trailing Stops
            if current_pos_side and indicators:
                entry_price = long_entry if current_pos_side == "long" else short_entry
                if not entry_price.is_nan() and entry_price > 0:
                     logger.debug(f"Managing TSL for {current_pos_side} position...")
                     self.order_manager.manage_trailing_stop(current_pos_side, entry_price, current_price, current_atr)
                     protection_tracker_snapshot = copy.deepcopy(self.order_manager.protection_tracker) # Update snapshot after TSL
                else: logger.warning(f"Cannot manage TSL for {current_pos_side}: Invalid entry price ({entry_price})")

                # Re-fetch position state AFTER TSL check as it might change protection status
                logger.debug("Re-fetching position state after TSL management...")
                positions_after_tsl = self.exchange_manager.get_current_position()
                if positions_after_tsl is None:
                    logger.error(f"{Fore.RED}Failed re-fetching positions after TSL check (Cycle {cycle_count}). Using potentially stale state.{Style.RESET_ALL}")
                    cycle_status = "WARN:POS_REFETCH_TSL_FAIL"
                else:
                    live_positions_state = positions_after_tsl # Update live state
                    final_positions_for_panel = copy.deepcopy(live_positions_state) # Update panel state
                    # Re-evaluate position state based on potentially updated info
                    active_long_pos = live_positions_state.get("long", {}); active_short_pos = live_positions_state.get("short", {})
                    long_qty = safe_decimal(active_long_pos.get("qty", "0")); short_qty = safe_decimal(active_short_pos.get("qty", "0"))
                    has_long_pos = long_qty.copy_abs() >= POSITION_QTY_EPSILON; has_short_pos = short_qty.copy_abs() >= POSITION_QTY_EPSILON
                    is_flat = not has_long_pos and not has_short_pos
                    current_pos_side = "long" if has_long_pos else "short" if has_short_pos else None
                    logger.debug(f"State After TSL Check: Flat={is_flat}, Side={current_pos_side}")
                    if is_flat and any(self.order_manager.protection_tracker.values()):
                        logger.warning(f"{Fore.YELLOW}Position became flat after TSL logic, clearing protection tracker.{Style.RESET_ALL}")
                        self.order_manager.protection_tracker = {"long": None, "short": None}
                        protection_tracker_snapshot = copy.deepcopy(self.order_manager.protection_tracker)

            # Generate Trading Signals (Entry)
            can_gen_signals = indicators is not None and not current_price.is_nan() and len(df) >= 2
            if can_gen_signals:
                logger.debug("Generating entry signals...")
                signals = self.signal_generator.generate_signals(df.iloc[-2:], indicators)
            else:
                reason = "Skipped Signal Gen: " + ("Indicators missing" if indicators is None else f"Need >=2 candles ({len(df)} found)")
                signals = {"long": False, "short": False, "reason": reason}; logger.warning(reason)

            # Check for Signal-Based Exits
            exit_triggered_by_signal = False; exit_side = None; qty_to_close_on_exit = Decimal(0)
            if can_gen_signals and current_pos_side: # Only check exit if in position and signals are valid
                logger.debug(f"Checking exit signals for {current_pos_side} position...")
                exit_reason = self.signal_generator.check_exit_signals(current_pos_side, indicators)
                if exit_reason:
                    exit_side = current_pos_side
                    qty_to_close_on_exit = long_qty if exit_side == "long" else short_qty
                    logger.trade(f"{Fore.YELLOW}Attempting signal exit for {exit_side.upper()} (Qty: {qty_to_close_on_exit.normalize()}) | Reason: {exit_reason}{Style.RESET_ALL}")
                    close_success = self.order_manager.close_position(exit_side, qty_to_close_on_exit, reason=exit_reason)
                    exit_triggered_by_signal = close_success
                    if not exit_triggered_by_signal:
                        cycle_status = "FAIL:EXIT_ORDER_FAILED"
                        logger.error(f"{Fore.RED}Failed to execute closing order for {exit_side} based on exit signal.{Style.RESET_ALL}")
                    else:
                        logger.info(f"Signal-based exit initiated/confirmed for {exit_side}.")
                        # Re-fetch state AGAIN if an exit was triggered successfully
                        logger.debug(f"Re-fetching state after signal exit attempt ({exit_side})...")
                        positions_after_exit = self.exchange_manager.get_current_position()
                        if positions_after_exit is None:
                            logger.error(f"{Fore.RED}Failed re-fetching positions after signal exit (Cycle {cycle_count}).{Style.RESET_ALL}")
                            cycle_status = "WARN:POS_REFETCH_EXIT_FAIL"
                        else:
                            live_positions_state = positions_after_exit # Update live state
                            final_positions_for_panel = copy.deepcopy(live_positions_state) # Update panel state
                            # Re-evaluate position state
                            active_long_pos = live_positions_state.get("long", {}); active_short_pos = live_positions_state.get("short", {})
                            long_qty = safe_decimal(active_long_pos.get("qty", "0")); short_qty = safe_decimal(active_short_pos.get("qty", "0"))
                            has_long_pos = long_qty.copy_abs() >= POSITION_QTY_EPSILON; has_short_pos = short_qty.copy_abs() >= POSITION_QTY_EPSILON
                            is_flat = not has_long_pos and not has_short_pos
                            current_pos_side = "long" if has_long_pos else "short" if has_short_pos else None
                            logger.debug(f"State After Signal Exit Attempt: Flat={is_flat}, Side={current_pos_side}")
                            if is_flat: # Ensure tracker clear if flat
                                logger.debug("Position became flat after signal exit, ensuring tracker clear.")
                                self.order_manager.protection_tracker = {"long": None, "short": None}
                                protection_tracker_snapshot = copy.deepcopy(self.order_manager.protection_tracker)

            # Execute Entry Trades (Only if flat and no exit was just triggered)
            if is_flat and not exit_triggered_by_signal and can_gen_signals and (signals.get("long") or signals.get("short")):
                # Double-check essential parameters
                if total_equity is None or total_equity.is_nan() or total_equity <= 0 or current_atr.is_nan() or current_atr <= 0:
                     logger.error("Entry skipped: Invalid equity/ATR detected pre-entry.")
                     cycle_status = "FAIL:INVALID_STATE_PRE_ENTRY"
                else:
                    entry_side = "buy" if signals.get("long") else "sell"; signal_reason = signals.get('reason', '')
                    log_color = Fore.GREEN if entry_side == 'buy' else Fore.RED
                    logger.trade(f"{log_color}{Style.BRIGHT}Entry Signal Detected: {entry_side.upper()}! {signal_reason}. Attempting entry...{Style.RESET_ALL}")
                    entry_successful = self.order_manager.place_risked_market_order(entry_side, current_atr, total_equity, current_price)
                    if entry_successful:
                        logger.info(f"{Fore.GREEN}{Style.BRIGHT}Entry sequence completed successfully for {entry_side}.{Style.RESET_ALL}")
                        logger.debug("Re-fetching state after successful entry...")
                        positions_after_entry = self.exchange_manager.get_current_position()
                        if positions_after_entry is not None:
                            final_positions_for_panel = copy.deepcopy(positions_after_entry) # Update panel state
                            protection_tracker_snapshot = copy.deepcopy(self.order_manager.protection_tracker) # Update tracker snapshot
                        else: logger.warning("Failed re-fetching positions post-entry."); cycle_status = "WARN:POS_REFETCH_ENTRY_FAIL"
                    else:
                        logger.error(f"{Fore.RED}Entry sequence failed for {entry_side}.{Style.RESET_ALL}")
                        cycle_status = "FAIL:ENTRY_SEQUENCE"
                        logger.debug("Re-fetching state after failed entry...")
                        positions_after_failed_entry = self.exchange_manager.get_current_position()
                        if positions_after_failed_entry is not None:
                            final_positions_for_panel = copy.deepcopy(positions_after_failed_entry) # Update panel state
                            if not (final_positions_for_panel.get("long") or final_positions_for_panel.get("short")):
                                self.order_manager.protection_tracker = {"long": None, "short": None} # Ensure tracker clear if flat
                            protection_tracker_snapshot = copy.deepcopy(self.order_manager.protection_tracker) # Update tracker snapshot
                        else: logger.warning("Failed re-fetching positions post-failed-entry."); cycle_status += "|POS_REFETCH_FAIL"

            elif is_flat: logger.debug("Position flat, no entry signal.")
            elif current_pos_side: logger.debug(f"Position ({current_pos_side.upper()}) open, skipping entry.")
        else:
             logger.warning(f"Core logic skipped (Cycle {cycle_count}) due to earlier failure/invalid state ({cycle_status}).")

        # 5. Display Status Panel (using final_positions_for_panel and protection_tracker_snapshot)
        logger.debug("Displaying status panel...")
        self.status_display.print_status_panel(
            cycle_count, last_timestamp, current_price, indicators,
            final_positions_for_panel, total_equity, signals,
            protection_tracker_snapshot, self.exchange_manager.market_info,
        )

        end_time = time.monotonic()
        logger.info(f"{Fore.BLUE}--- Cycle {cycle_count} Status: {cycle_status} (Duration: {end_time - start_time:.2f}s) ---{Style.RESET_ALL}")

    def graceful_shutdown(self) -> None:
        """Handles cleaning up: cancelling orders and closing positions before exiting."""
        console.print(f"\n[bold yellow]Initiating Graceful Shutdown Sequence...[/]")
        logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}Initiating Graceful Shutdown Sequence...{Style.RESET_ALL}")
        termux_notify("Pyrmethus Shutdown", f"Closing {self.config.symbol}...")

        if not self.exchange_manager or not self.exchange_manager.exchange or not self.exchange_manager.market_info:
            logger.error(f"{Fore.RED}Cannot shutdown: Exchange Manager/Instance/Market Info missing.{Style.RESET_ALL}")
            termux_notify("Shutdown Warning!", f"{self.config.symbol}: Cannot shutdown cleanly!")
            return

        exchange = self.exchange_manager.exchange
        symbol = self.config.symbol

        # --- 1. Cancel All Open Non-Positional Orders ---
        logger.info(f"{Fore.CYAN}Cancelling active non-positional orders for {symbol} (Category: {self.config.bybit_v5_category})...{Style.RESET_ALL}")
        try:
            params = {"category": self.config.bybit_v5_category, "symbol": symbol} # V5 might need symbol too
            cancel_resp = fetch_with_retries(
                exchange.cancel_all_orders, symbol=symbol, params=params, max_retries=1, delay_seconds=1
            )
            logger.info(f"Cancel all orders response: {str(cancel_resp)[:200]}...")
            if isinstance(cancel_resp, dict) and cancel_resp.get('retCode') == V5_SUCCESS_RETCODE:
                 cancelled_list = cancel_resp.get('result', {}).get('list', [])
                 logger.info(f"Cancel all orders command successful (API). Found {len(cancelled_list)} items in response.")
            elif isinstance(cancel_resp, list):
                 logger.info(f"Cancelled {len(cancel_resp)} active orders (CCXT list response).")
            else: logger.warning("Cancel all orders response format unexpected/failed.")
        except ccxt.NotSupported: logger.warning(f"Exchange does not support cancel_all_orders w/ params. Skipping.")
        except Exception as e: logger.error(f"{Fore.RED}Error cancelling orders: {e}{Style.RESET_ALL}", exc_info=False)

        logger.info("Clearing local protection tracker state...")
        self.order_manager.protection_tracker = {"long": None, "short": None}
        logger.info("Waiting briefly after cancel/clear...")
        time.sleep(max(self.config.order_check_delay_seconds, 2))

        # --- 2. Check and Close Any Lingering Positions ---
        logger.info(f"{Fore.CYAN}Checking for lingering positions for {symbol} to close...{Style.RESET_ALL}")
        closed_count = 0; positions_to_close: List[Tuple[str, Decimal]] = []
        try:
            final_positions = self.exchange_manager.get_current_position()
            if final_positions is not None:
                for side in ["long", "short"]:
                    pos_data = final_positions.get(side)
                    if pos_data:
                        qty = safe_decimal(pos_data.get('qty', '0'))
                        if qty.copy_abs() >= POSITION_QTY_EPSILON:
                            logger.warning(f"{Fore.YELLOW}Lingering {side.upper()} position (Qty: {qty.normalize()}) found.{Style.RESET_ALL}")
                            positions_to_close.append((side, qty))

                if not positions_to_close: logger.info(f"{Fore.GREEN}No lingering positions found.{Style.RESET_ALL}")
                else:
                    logger.warning(f"Attempting emergency close for {len(positions_to_close)} position(s)...")
                    for side, qty in positions_to_close:
                        logger.info(f"Closing {side.upper()} (Qty: {qty.normalize()})...")
                        # Note: close_position already clears stops and verifies
                        close_success = self.order_manager.close_position(side, qty, reason="GracefulShutdown")
                        if close_success: closed_count += 1; logger.info(f"{Fore.GREEN}{Style.BRIGHT}Closure initiated/confirmed for {side.upper()}.{Style.RESET_ALL}")
                        else: logger.error(f"{Fore.RED}Closure FAILED for {side.upper()}. MANUAL INTERVENTION REQUIRED.{Style.RESET_ALL}")

                    if closed_count == len(positions_to_close): logger.info(f"{Fore.GREEN}{Style.BRIGHT}All detected positions ({closed_count}) closed/initiated.{Style.RESET_ALL}")
                    else: logger.warning(f"{Fore.YELLOW}Attempted {len(positions_to_close)} closures, {closed_count} succeeded/initiated. MANUAL CHECK NEEDED.{Style.RESET_ALL}")
            else:
                logger.error(f"{Fore.RED}Failed fetching positions during shutdown check. MANUAL CHECK REQUIRED for {symbol}!{Style.RESET_ALL}")
                termux_notify(f"{symbol} Shutdown Issue", "Failed pos check! Manual verify.")
        except Exception as e:
            logger.error(f"{Fore.RED}{Style.BRIGHT}Error during position closure shutdown: {e}. MANUAL CHECK REQUIRED.{Style.RESET_ALL}", exc_info=True)
            termux_notify(f"{symbol} Shutdown Issue", f"Error closing pos: {e}")

        console.print(f"[bold yellow]Graceful Shutdown Sequence Complete.[/]")
        logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}Graceful Shutdown Complete. Pyrmethus rests.{Style.RESET_ALL}")
        termux_notify("Shutdown Complete", f"{self.config.symbol} shutdown finished.")

    def _display_startup_info(self):
        """Prints initial configuration details using Rich."""
        console.print(f"[bold bright_cyan] Summoning Pyrmethus [bright_magenta]v4.5.7 (Neon Nexus Edition)[/]...")
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="bright_yellow") # Neon Yellow for keys
        table.add_column(style="bright_white") # Bright White for values

        table.add_row("Trading Symbol:", f"{self.config.symbol}")
        table.add_row("Interval:", f"{self.config.interval}")
        table.add_row("Market Type:", f"{self.config.market_type}")
        table.add_row("V5 Category:", f"{self.config.bybit_v5_category}")
        table.add_row("---", "---", style="dim")
        table.add_row("Risk %:", f"{self.config.risk_percentage:.3%}")
        table.add_row("SL ATR Mult:", f"{self.config.sl_atr_multiplier.normalize()}x")
        table.add_row("TP ATR Mult:", f"{self.config.tp_atr_multiplier.normalize()}x" if self.config.tp_atr_multiplier > 0 else "[dim]Disabled[/]")
        table.add_row("TSL Act Mult:", f"{self.config.tsl_activation_atr_multiplier.normalize()}x")
        table.add_row("TSL Trail %:", f"{self.config.trailing_stop_percent.normalize()}%")
        table.add_row("---", "---", style="dim")
        table.add_row("Trend Filter:", f"[bright_green]ON[/]" if self.config.trade_only_with_trend else "[dim]OFF[/]")
        table.add_row("ATR Move Filter:", f"{self.config.atr_move_filter_multiplier.normalize()}x" if self.config.atr_move_filter_multiplier > 0 else "[dim]Disabled[/]")
        table.add_row("ADX Filter Lvl:", f">{self.config.min_adx_level.normalize()}")
        table.add_row("---", "---", style="dim")
        journal_status = f"[bright_green]Enabled[/] ([dim]{self.config.journal_file_path}[/])" if self.config.enable_journaling else "[dim]Disabled[/]"
        table.add_row("Journaling:", journal_status)
        v5_stops_info = (f"SLTrig:[bright_cyan]{self.config.sl_trigger_by}[/], "
                         f"TSLTrig:[bright_cyan]{self.config.tsl_trigger_by}[/], "
                         f"PosIdx:[bright_cyan]{self.config.position_idx}[/]")
        table.add_row("[dim]V5 Pos Stops:[/]", f"[dim]{v5_stops_info}[/]")

        console.print(table)
        console.print("-" * 60, style="dim bright_blue")


# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        bot = TradingBot()
        bot.run() # Contains the main loop and graceful shutdown
    except SystemExit as e:
        log_level_exit = logging.INFO if e.code == 0 else logging.WARNING
        logger.log(log_level_exit, f"Pyrmethus process terminated (Exit Code: {e.code}).")
    except Exception as main_exception:
        # Corrected color usage here
        logger.critical(f"{Fore.RED}{Style.BRIGHT}CRITICAL UNHANDLED ERROR during bot execution: {main_exception}{Style.RESET_ALL}", exc_info=True)
        try: termux_notify("Pyrmethus CRITICAL ERROR", f"Bot failed: {str(main_exception)[:100]}")
        except Exception as notify_exc: logger.error(f"Failed to send critical error notification: {notify_exc}")
        sys.exit(1) # Ensure non-zero exit code on critical failure


