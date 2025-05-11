# -*- coding: utf-8 -*-
# pylint: disable=logging-fstring-interpolation, too-many-instance-attributes, too-many-arguments, too-many-locals, too-many-public-methods, invalid-name, unused-argument, too-many-lines, unnecessary-pass, unnecessary-lambda-assignment, line-too-long
# fmt: off
#   ____        _       _   _                  _            _         _
#  |  _ \\ _   _| |_ ___| | | | __ ___   ____ _| |_ ___  ___| |_ _ __ | |__   ___ _ __ ___  _ __
#  | |_) | | | | __/ _ \\ | | |/ _` \\ \\ / / _` | __/ _ \\/ __| __| '_ \\| '_ \\ / _ \\ '_ ` _ \\| '_ \\
#  |  __/| |_| | ||  __/ | | | (_| |\\ V / (_| | ||  __/\\__ \\ |_| |_) | | | |  __/ | | | | | |_) |
#  |_|    \\__, |\\__\\___|_|_|_|\\__,_| \\_/ \\__,_|\\__\\___||___/\\__| .__/|_| |_|\\___|_| |_| |_| .__/
#         |___/                                                |_|                      |_|
# Pyrmethus v4.5.7 - Neon Nexus Edition
# fmt: on
"""
Pyrmethus - Termux Trading Spell (v4.5.7 - Neon Nexus Edition)

Conjures market insights and executes trades on Bybit Futures.
This version is specifically refactored to use the V5 Unified Account API via CCXT,
employing classes for better structure and leveraging V5 position-based
stop-loss, take-profit, and trailing-stop features.
"""

# Standard Library Imports
import csv
import logging
import os
import signal
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from decimal import (
    ROUND_DOWN,
    ROUND_HALF_EVEN,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    getcontext,
)
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

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
    if e.name == "colorama": # Special handling if colorama itself is missing
        print("Missing essential package: colorama. Cannot display colored output.")
        print("Attempting basic error message...")
        print(f"Missing essential spell component: {e.name}")
        print(f"To conjure it, cast: pip install {e.name}")
        print("\nOr, to ensure all scrolls are present, cast:")
        print(f"pip install {' '.join(COMMON_PACKAGES)}")
        sys.exit(1)
    else: # Colorama is available, use it for a nicer error message
        colorama_init(autoreset=True)
        missing_pkg = e.name
        print(
            f"{Fore.RED}{Style.BRIGHT}Missing essential spell component: {Style.BRIGHT}{missing_pkg}{Style.NORMAL}"
        )
        print(
            f"{Fore.YELLOW}To conjure it, cast: {Style.BRIGHT}pip install {missing_pkg}{Style.RESET_ALL}"
        )
        print(f"\n{Fore.CYAN}Or, to ensure all scrolls are present, cast:")
        # More robust Termux check by looking for PREFIX environment variable common in Termux
        is_termux = "com.termux" in os.environ.get("PREFIX", "")

        if is_termux:
            termux_pkgs_to_install = []
            pip_pkgs_to_install = list(COMMON_PACKAGES) # Start with all packages for pip

            # For Termux, pandas and numpy are often better installed via pkg after python
            # Check if they are in COMMON_PACKAGES before suggesting 'pkg install'
            if "pandas" in COMMON_PACKAGES:
                termux_pkgs_to_install.append("python-pandas")
                if 'pandas' in pip_pkgs_to_install: pip_pkgs_to_install.remove('pandas')
            if "numpy" in COMMON_PACKAGES:
                termux_pkgs_to_install.append("python-numpy")
                if 'numpy' in pip_pkgs_to_install: pip_pkgs_to_install.remove('numpy')

            install_cmd_parts = []
            if termux_pkgs_to_install:
                install_cmd_parts.append(f"pkg install python {' '.join(termux_pkgs_to_install)}")
            if pip_pkgs_to_install: # Install remaining packages via pip
                install_cmd_parts.append(f"pip install {' '.join(pip_pkgs_to_install)}")

            install_cmd = " && ".join(install_cmd_parts) if install_cmd_parts else f"pip install {' '.join(COMMON_PACKAGES)}" # Fallback
            print(f"{Style.BRIGHT}{install_cmd}{Style.RESET_ALL}")
            if termux_pkgs_to_install:
                print(
                    f"{Fore.YELLOW}Note: In Termux, {' and '.join(pkg.replace('python-','') for pkg in termux_pkgs_to_install)} are often best installed via 'pkg' for compatibility.{Style.RESET_ALL}"
                )
        else: # Standard pip install for other systems
            print(
                f"{Style.BRIGHT}pip install {' '.join(COMMON_PACKAGES)}{Style.RESET_ALL}"
            )
        sys.exit(1)

# --- Constants ---
DECIMAL_PRECISION = 50 # Global precision for Decimal context
POSITION_QTY_EPSILON = Decimal("1E-12")  # Threshold for considering a position 'flat' or qty negligible
DEFAULT_PRICE_DP = 4  # Default decimal places for price formatting if market info unavailable
DEFAULT_AMOUNT_DP = 6 # Default decimal places for amount/quantity formatting
DEFAULT_OHLCV_LIMIT = 200
DEFAULT_LOOP_SLEEP = 15 # Seconds
DEFAULT_RETRY_DELAY = 3   # Seconds
DEFAULT_MAX_RETRIES = 3
DEFAULT_RISK_PERCENT = Decimal("0.01") # 1% risk per trade
DEFAULT_SL_MULT = Decimal("1.5")    # ATR Multiplier for Stop Loss
DEFAULT_TP_MULT = Decimal("3.0")    # ATR Multiplier for Take Profit
DEFAULT_TSL_ACT_MULT = Decimal("1.0") # ATR Multiplier for Trailing Stop Activation
DEFAULT_TSL_PERCENT = Decimal("0.5")  # Percentage for Trailing Stop Loss distance from current price
DEFAULT_STOCH_OVERSOLD = Decimal("30")
DEFAULT_STOCH_OVERBOUGHT = Decimal("70")
DEFAULT_MIN_ADX = Decimal("20")     # Minimum ADX level to consider a trend strong enough
DEFAULT_JOURNAL_FILE = "pyrmethus_trading_journal.csv"
V5_UNIFIED_ACCOUNT_TYPE = "UNIFIED"
V5_HEDGE_MODE_POSITION_IDX = 0 # Default index for position mode (0=One-Way, 1=Buy Hedge, 2=Sell Hedge)
V5_TPSL_MODE_FULL = "Full" # Apply SL/TP to the entire position for V5
V5_SUCCESS_RETCODE = 0     # Standard success return code for Bybit V5 API
TERMUX_NOTIFY_TIMEOUT = 10 # Seconds, increased timeout for termux-toast command

# Initialize Colorama & Rich Console
colorama_init(autoreset=True)
console = Console(log_path=False) # Disable Rich's own log file handling to use Python's logging

# Set Decimal precision context globally
getcontext().prec = DECIMAL_PRECISION

# --- Logging Setup ---
# Custom logging level for trade actions (e.g., order placement, closure)
TRADE_LEVEL_NUM = 25  # Between INFO (20) and WARNING (30)
if not hasattr(logging, "TRADE"): # Ensure it's not already defined as a level name
    logging.addLevelName(TRADE_LEVEL_NUM, "TRADE")
if not hasattr(logging.Logger, "trade"): # Ensure method not already defined on Logger
    def trade_log(self, message, *args, **kws):
        """Logs a message with custom level TRADE."""
        if self.isEnabledFor(TRADE_LEVEL_NUM):
            # pylint: disable=protected-access
            self._log(TRADE_LEVEL_NUM, message, args, **kws)
    logging.Logger.trade = trade_log # type: ignore[attr-defined]

# Base logger configuration
logger = logging.getLogger(__name__) # Get logger for this module
log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)-8s] (%(filename)s:%(lineno)d) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
# Ensure log_level_str is a valid level name before getattr
valid_log_levels = ["DEBUG", "INFO", "TRADE", "WARNING", "ERROR", "CRITICAL"]
log_level_to_set: int # Variable to hold the final integer log level
if log_level_str.isdigit() and int(log_level_str) == TRADE_LEVEL_NUM:
    log_level_to_set = TRADE_LEVEL_NUM
elif log_level_str in valid_log_levels:
    log_level_to_set = getattr(logging, log_level_str)
else:
    # Early print, logger not fully set up, but colorama is
    print(f"{Fore.YELLOW}Warning: Invalid LOG_LEVEL '{log_level_str}'. Defaulting to INFO.{Style.RESET_ALL}")
    log_level_str = "INFO" # For display in startup info
    log_level_to_set = logging.INFO

logger.setLevel(log_level_to_set)

# Ensure handler is added only once to prevent duplicate logs in different environments/reloads
if not logger.hasHandlers():
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
logger.propagate = False # Prevent passing logs to the root logger, which might have other handlers


# --- Utility Functions ---
def safe_decimal(
    value: Any, default: Decimal = Decimal("NaN")
) -> Decimal:
    """Safely converts a value to Decimal, handling None, empty strings, and invalid formats."""
    if value is None:
        return default
    try:
        # Convert potential floats or numeric types to string first for precise Decimal conversion
        str_value = str(value).strip()
        if not str_value:  # Handle empty string after stripping
            return default
        # Handle common non-numeric strings that might appear in API responses or configs
        if str_value.lower() in ["nan", "none", "null"]: # "null" for JSON
            return default
        return Decimal(str_value)
    except (InvalidOperation, ValueError, TypeError):
        # Optional: logger.debug(f"Could not convert '{value}' (type: {type(value).__name__}) to Decimal, using default {default}")
        return default


def termux_notify(title: str, content: str) -> None:
    """Sends a notification via Termux API (toast), if available. Title is ignored by termux-toast."""
    # Check if running in Termux environment by checking for common Termux env var PREFIX
    if "com.termux" in os.environ.get("PREFIX", ""):
        try:
            # termux-toast expects only the content argument; title is effectively ignored.
            # Using check=False to manually handle non-zero exit codes.
            result = subprocess.run(
                ["termux-toast", content],
                check=False,
                timeout=TERMUX_NOTIFY_TIMEOUT,
                capture_output=True, # Capture stdout/stderr
                text=True, # Decode output as text
            )
            if result.returncode != 0:
                # Log stderr if available, otherwise stdout, for debugging failed toasts
                error_output = result.stderr.strip() if result.stderr else result.stdout.strip()
                logger.warning(
                    f"Termux toast command failed (code {result.returncode}): {error_output}"
                )
            # Optional: logger.debug(f"Termux toast sent: '{content}' (Title '{title}' ignored by toast)")
        except FileNotFoundError:
            logger.warning(
                "Termux notify failed: 'termux-toast' command not found. Is Termux:API installed and setup?"
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"Termux notify failed: command timed out after {TERMUX_NOTIFY_TIMEOUT} seconds.")
        except Exception as e: # Catch any other unexpected errors during subprocess run
            logger.warning(f"Termux notify failed unexpectedly: {e}")
    # else: # Optional: logger.debug("Not in Termux environment, skipping notification.")


def fetch_with_retries(
    fetch_function: Callable[..., Any],
    *args: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
    delay_seconds: int = DEFAULT_RETRY_DELAY,
    retry_on_exceptions: Tuple[Type[Exception], ...] = (
        ccxt.DDoSProtection, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable,
        ccxt.NetworkError, ccxt.RateLimitExceeded, requests.exceptions.ConnectionError,
        requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError,
        requests.exceptions.ReadTimeout,
    ),
    fatal_exceptions: Tuple[Type[Exception], ...] = (
        ccxt.AuthenticationError, ccxt.PermissionDenied # Errors that should halt immediately
    ),
    fail_fast_exceptions: Tuple[Type[Exception], ...] = (
         ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.OrderNotFound # Errors where retrying is pointless for this call
    ),
    **kwargs: Any,
) -> Any:
    """Wraps a function call with enhanced retry logic and specific error handling."""
    last_exception: Optional[Exception] = None
    func_name = getattr(fetch_function, "__name__", "Unnamed function")

    for attempt in range(max_retries + 1): # Total attempts = max_retries + 1 (initial attempt)
        try:
            result = fetch_function(*args, **kwargs)
            if attempt > 0: # Log success only if it's a retry that succeeded
                logger.info(f"{Style.BRIGHT}{Fore.GREEN}Successfully executed {func_name} on attempt {attempt + 1}/{max_retries + 1} after previous failures.{Style.RESET_ALL}")
            return result
        except fatal_exceptions as e:
            logger.critical(f"{Style.BRIGHT}{Fore.RED}Fatal error ({type(e).__name__}) executing {func_name}: {e}. Halting immediately.{Style.RESET_ALL}", exc_info=False) # No stack trace for auth
            raise e # Re-raise critical error to be handled by higher-level logic (e.g., bot shutdown)
        except fail_fast_exceptions as e:
            logger.error(f"{Fore.RED}Fail-fast error ({type(e).__name__}) executing {func_name}: {e}. Not retrying this call.{Style.RESET_ALL}")
            last_exception = e
            break # Break loop, don't retry for these specific errors
        except retry_on_exceptions as e:
            last_exception = e
            # Truncate long error messages for cleaner logs
            error_summary = str(e)[:150] + "..." if len(str(e)) > 150 else str(e)
            retry_msg = f"{Fore.YELLOW}Retryable error ({type(e).__name__}) on attempt {attempt + 1}/{max_retries + 1} for {func_name}: {error_summary}.{Style.RESET_ALL}"
            if attempt < max_retries:
                logger.warning(f"{retry_msg} Retrying in {delay_seconds}s...")
                time.sleep(delay_seconds)
            else:
                logger.error(f"{Fore.RED}Max retries ({max_retries + 1}) reached for {func_name} after retryable error. Last error: {e}{Style.RESET_ALL}")
                # Loop ends, last_exception will be raised below
        except ccxt.ExchangeError as e: # Catch other generic exchange errors
            last_exception = e
            logger.error(f"{Fore.RED}Unhandled ExchangeError during {func_name}: {e}{Style.RESET_ALL}")
            # Decide if specific ExchangeErrors are retryable - here we retry generic ones as a fallback
            if attempt < max_retries:
                logger.warning(f"Retrying generic exchange error in {delay_seconds}s...")
                time.sleep(delay_seconds)
            else:
                logger.error(f"Max retries reached after generic exchange error for {func_name}.")
                break
        except Exception as e: # Catch truly unexpected errors not covered above
            last_exception = e
            logger.error(f"{Fore.RED}Unexpected error during {func_name}: {e}{Style.RESET_ALL}", exc_info=True) # Include stack trace
            break # Don't retry unknown errors, break loop

    # If loop finished without returning (i.e., all retries failed or a break occurred), raise the last captured exception
    if last_exception:
        raise last_exception
    else:
        # This path should ideally not be hit if logic is correct (e.g., max_retries = 0 and first attempt fails without exception type match)
        # Or if fetch_function returns None and it's not handled as an error above.
        raise RuntimeError(f"Function {func_name} failed after {max_retries + 1} attempts without raising a recognized or captured exception.")


# --- Configuration Class ---
class TradingConfig:
    """Loads, validates, and holds trading configuration parameters from .env file or environment variables."""

    # pylint: disable=too-many-statements
    def __init__(self, env_file: str = ".env"):
        logger.debug(f"Loading configuration from environment variables / '{env_file}'...")
        env_path = Path(env_file)
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path, override=True) # override=True ensures .env takes precedence
            logger.info(f"Loaded configuration from {env_path}")
        else:
            logger.warning(f"Environment file '{env_path}' not found. Relying solely on system environment variables.")

        # Core Trading Parameters
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT", Style.DIM)
        self.market_type: str = self._get_env(
            "MARKET_TYPE", "linear", Style.DIM, allowed_values=["linear", "inverse", "swap"]
        ).lower()
        # bybit_v5_category is determined after symbol and market_type are loaded
        self.bybit_v5_category: str = self._determine_v5_category()
        self.interval: str = self._get_env("INTERVAL", "1m", Style.DIM)

        # Financial Parameters (Decimal for precision)
        self.risk_percentage: Decimal = self._get_env(
            "RISK_PERCENTAGE", DEFAULT_RISK_PERCENT, Fore.YELLOW, cast_type=Decimal,
            min_val=Decimal("0.00001"), max_val=Decimal("0.5") # Allow 0.001% to 50% risk
        )
        self.sl_atr_multiplier: Decimal = self._get_env(
            "SL_ATR_MULTIPLIER", DEFAULT_SL_MULT, Fore.YELLOW, cast_type=Decimal,
            min_val=Decimal("0.1"), max_val=Decimal("20.0")
        )
        self.tp_atr_multiplier: Decimal = self._get_env(
            "TP_ATR_MULTIPLIER", DEFAULT_TP_MULT, Fore.YELLOW, cast_type=Decimal,
            min_val=Decimal("0.0"), max_val=Decimal("50.0") # Allow TP=0 to disable ATR-based TP
        )
        self.tsl_activation_atr_multiplier: Decimal = self._get_env(
            "TSL_ACTIVATION_ATR_MULTIPLIER", DEFAULT_TSL_ACT_MULT, Fore.YELLOW, cast_type=Decimal,
            min_val=Decimal("0.1"), max_val=Decimal("20.0")
        )
        self.trailing_stop_percent: Decimal = self._get_env(
            "TRAILING_STOP_PERCENT", DEFAULT_TSL_PERCENT, Fore.YELLOW, cast_type=Decimal,
            min_val=Decimal("0.001"), max_val=Decimal("10.0") # Allow 0.1% to 10% TSL
        )

        # V5 Position Stop Parameters
        self.sl_trigger_by: str = self._get_env(
            "SL_TRIGGER_BY", "LastPrice", Style.DIM, allowed_values=["LastPrice", "MarkPrice", "IndexPrice"]
        )
        self.tsl_trigger_by: str = self._get_env( # TP trigger usually follows SL trigger type in Bybit V5 TPSL settings
            "TSL_TRIGGER_BY", "LastPrice", Style.DIM, allowed_values=["LastPrice", "MarkPrice", "IndexPrice"]
        )
        self.position_idx: int = self._get_env(
            "POSITION_IDX", V5_HEDGE_MODE_POSITION_IDX, Style.DIM, cast_type=int, allowed_values=[0, 1, 2]
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
        self.stoch_oversold_threshold: Decimal = self._get_env(
            "STOCH_OVERSOLD_THRESHOLD", DEFAULT_STOCH_OVERSOLD, Fore.CYAN, cast_type=Decimal,
            min_val=Decimal("0"), max_val=Decimal("45")
        )
        self.stoch_overbought_threshold: Decimal = self._get_env(
            "STOCH_OVERBOUGHT_THRESHOLD", DEFAULT_STOCH_OVERBOUGHT, Fore.CYAN, cast_type=Decimal,
            min_val=Decimal("55"), max_val=Decimal("100")
        )
        self.trend_filter_buffer_percent: Decimal = self._get_env(
            "TREND_FILTER_BUFFER_PERCENT", Decimal("0.5"), Fore.CYAN, cast_type=Decimal,
            min_val=Decimal("0"), max_val=Decimal("5") # Buffer as percentage of trend EMA
        )
        self.atr_move_filter_multiplier: Decimal = self._get_env(
            "ATR_MOVE_FILTER_MULTIPLIER", Decimal("0.5"), Fore.CYAN, cast_type=Decimal,
            min_val=Decimal("0"), max_val=Decimal("5") # Multiplier for ATR; 0 disables filter
        )
        self.min_adx_level: Decimal = self._get_env(
            "MIN_ADX_LEVEL", DEFAULT_MIN_ADX, Fore.CYAN, cast_type=Decimal,
            min_val=Decimal("0"), max_val=Decimal("90")
        )

        # API Keys (Secrets) - Handled by _get_env with is_secret=True
        self.api_key: str = self._get_env("BYBIT_API_KEY", None, Fore.RED, is_secret=True)
        self.api_secret: str = self._get_env("BYBIT_API_SECRET", None, Fore.RED, is_secret=True)

        # Operational Parameters
        self.ohlcv_limit: int = self._get_env("OHLCV_LIMIT", DEFAULT_OHLCV_LIMIT, Style.DIM, cast_type=int, min_val=50, max_val=1000)
        self.loop_sleep_seconds: int = self._get_env("LOOP_SLEEP_SECONDS", DEFAULT_LOOP_SLEEP, Style.DIM, cast_type=int, min_val=1)
        self.order_check_delay_seconds: int = self._get_env("ORDER_CHECK_DELAY_SECONDS", 2, Style.DIM, cast_type=int, min_val=1)
        self.order_fill_timeout_seconds: int = self._get_env( # Used in verification logic implicitly by number of attempts
            "ORDER_FILL_TIMEOUT_SECONDS", 20, Style.DIM, cast_type=int, min_val=5
        )
        self.max_fetch_retries: int = self._get_env("MAX_FETCH_RETRIES", DEFAULT_MAX_RETRIES, Style.DIM, cast_type=int, min_val=0, max_val=10)
        self.retry_delay_seconds: int = self._get_env("RETRY_DELAY_SECONDS", DEFAULT_RETRY_DELAY, Style.DIM, cast_type=int, min_val=1)
        self.trade_only_with_trend: bool = self._get_env("TRADE_ONLY_WITH_TREND", True, Style.DIM, cast_type=bool)

        # Journaling
        self.journal_file_path: str = self._get_env("JOURNAL_FILE_PATH", DEFAULT_JOURNAL_FILE, Style.DIM)
        self.enable_journaling: bool = self._get_env("ENABLE_JOURNALING", True, Style.DIM, cast_type=bool)

        # Final Checks (API keys are checked within _get_env if default is None)
        self._validate_config() # Perform cross-parameter validations
        logger.debug("Configuration loaded and validated successfully.")

    def _determine_v5_category(self) -> str:
        """Determines the Bybit V5 API category based on symbol and market type."""
        try:
            category: str
            if self.market_type == "inverse":
                category = "inverse" # e.g., BTC/USD (settled in BTC)
            elif self.market_type in ["linear", "swap"]: # 'swap' is usually linear for Bybit V5 category
                category = "linear"  # e.g., BTC/USDT (settled in USDT)
            else: # Should be caught by _get_env validation for MARKET_TYPE
                raise ValueError(f"Unsupported MARKET_TYPE '{self.market_type}' for category determination.")

            # Log the symbol format for clarity, as it impacts CCXT's behavior
            # CCXT symbol format for Bybit V5 futures:
            # Linear: BASE/QUOTE:SETTLE (e.g., BTC/USDT:USDT)
            # Inverse: BASE/QUOTE:SETTLE (e.g., BTC/USD:BTC)
            # The :SETTLE part is crucial for CCXT to correctly identify the contract.
            if ":" not in self.config.symbol:
                 logger.warning(f"Symbol '{self.config.symbol}' does not explicitly include the settle currency (e.g., :USDT or :BTC). "
                                f"CCXT might default or infer, but explicit format (BASE/QUOTE:SETTLE) is recommended for V5 API clarity.")

            logger.info(
                f"Determined Bybit V5 API category: '{category}' for symbol '{self.config.symbol}' and market type '{self.market_type}'"
            )
            return category
        except ValueError as e:
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Could not determine V5 category: {e}. Halting.{Style.RESET_ALL}",
                exc_info=True, # Show traceback for this critical error
            )
            sys.exit(1)
        # Add a return statement here to satisfy linters, though it should not be reached.
        return "" # Should be unreachable due to sys.exit()

    def _validate_config(self):
        """Performs post-load validation of related configuration parameters."""
        if self.fast_ema_period >= self.slow_ema_period:
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Validation failed: FAST_EMA_PERIOD ({self.fast_ema_period}) must be less than SLOW_EMA_PERIOD ({self.slow_ema_period}). Halting.{Style.RESET_ALL}"
            )
            sys.exit(1)
        if self.trend_ema_period <= self.slow_ema_period: # More of a strategy warning
            logger.warning(
                f"{Fore.YELLOW}Config Warning: TREND_EMA_PERIOD ({self.trend_ema_period}) is less than or equal to SLOW_EMA_PERIOD ({self.slow_ema_period}). Trend filter might lag short-term EMA signals.{Style.RESET_ALL}"
            )
        if self.stoch_oversold_threshold >= self.stoch_overbought_threshold:
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Validation failed: STOCH_OVERSOLD_THRESHOLD ({self.stoch_oversold_threshold.normalize()}) must be less than STOCH_OVERBOUGHT_THRESHOLD ({self.stoch_overbought_threshold.normalize()}). Halting.{Style.RESET_ALL}"
            )
            sys.exit(1)
        if self.tsl_activation_atr_multiplier < self.sl_atr_multiplier: # Strategy warning
            logger.warning(
                f"{Fore.YELLOW}Config Warning: TSL_ACTIVATION_ATR_MULTIPLIER ({self.tsl_activation_atr_multiplier.normalize()}) is less than SL_ATR_MULTIPLIER ({self.sl_atr_multiplier.normalize()}). TSL may activate before initial SL distance is fully established by price movement.{Style.RESET_ALL}"
            )
        # Check TP vs SL only if TP is enabled (multiplier > 0)
        if self.tp_atr_multiplier > Decimal("0") and self.tp_atr_multiplier <= self.sl_atr_multiplier: # Strategy warning
            logger.warning(
                f"{Fore.YELLOW}Config Warning: TP_ATR_MULTIPLIER ({self.tp_atr_multiplier.normalize()}) is less than or equal to SL_ATR_MULTIPLIER ({self.sl_atr_multiplier.normalize()}). This implies a Risk:Reward ratio of 1:1 or less.{Style.RESET_ALL}"
            )

    def _cast_value(self, key: str, value_str: str, cast_type: Type, default: Any) -> Any:
        """Helper to cast string value to target type, returning default on failure. Handles empty strings."""
        val_to_cast = value_str.strip() # Strip whitespace before casting
        if not val_to_cast: # Handle empty string after strip
            # If default is also None or empty for a string type, this might be intended.
            # Otherwise, using the actual default value passed is safer.
            if default is None or (isinstance(default, str) and not default):
                logger.debug(f"Empty value string for '{key}' after stripping. Default is also empty/None. Returning original default.")
                return default # Return the original default (which might be None or "")
            else:
                logger.warning(f"Empty value string for '{key}' after stripping. Using default '{default}'.")
                return default

        try:
            if cast_type == bool:
                return val_to_cast.lower() in ["true", "1", "yes", "y", "on"]
            elif cast_type == Decimal:
                # Check for common non-numeric strings before attempting Decimal conversion
                if val_to_cast.lower() in ["nan", "none", "null"]:
                    return Decimal("NaN") # Consistent NaN representation
                return Decimal(val_to_cast)
            elif cast_type == int:
                # Attempt conversion to Decimal first to check for fractional parts
                dec_val = Decimal(val_to_cast)
                if dec_val.to_integral_value(rounding=ROUND_DOWN) != dec_val:
                    raise ValueError(f"Decimal value '{val_to_cast}' with fractional part cannot be cast to int without loss.")
                return int(dec_val)
            # Add other specific casts if needed
            else: # Includes str type, which is the default cast_type
                return cast_type(val_to_cast) # Use constructor directly (e.g., str())
        except (ValueError, TypeError, InvalidOperation) as e: # Catch errors from Decimal, int, bool, etc.
            logger.error(
                f"{Fore.RED}Cast failed for '{key}' (value: '{value_str}', target_type: {cast_type.__name__}): {e}. Using default '{default}'.{Style.RESET_ALL}"
            )
            return default

    def _validate_value(
        self, key: str, value: Any,
        min_val: Optional[Union[int, float, Decimal]],
        max_val: Optional[Union[int, float, Decimal]],
        allowed_values: Optional[List[Any]]
    ) -> bool:
        """Helper to validate a value against min/max constraints and allowed values. Logs and returns False on failure (for non-critical). Halts for critical."""
        is_numeric_comparable = isinstance(value, (int, float, Decimal))

        # Min/Max checks (critical, halts if violated)
        if min_val is not None:
            if not is_numeric_comparable:
                logger.critical(f"{Style.BRIGHT}{Fore.RED}Validation failed for '{key}': Non-numeric value '{value}' (type: {type(value).__name__}) cannot be compared with min_val '{min_val}'. Halting.{Style.RESET_ALL}")
                sys.exit(1)
            if value < min_val: # type: ignore
                logger.critical(f"{Style.BRIGHT}{Fore.RED}Validation failed for '{key}': Value '{value}' is less than minimum '{min_val}'. Halting.{Style.RESET_ALL}")
                sys.exit(1)
        if max_val is not None:
            if not is_numeric_comparable:
                logger.critical(f"{Style.BRIGHT}{Fore.RED}Validation failed for '{key}': Non-numeric value '{value}' (type: {type(value).__name__}) cannot be compared with max_val '{max_val}'. Halting.{Style.RESET_ALL}")
                sys.exit(1)
            if value > max_val: # type: ignore
                logger.critical(f"{Style.BRIGHT}{Fore.RED}Validation failed for '{key}': Value '{value}' is greater than maximum '{max_val}'. Halting.{Style.RESET_ALL}")
                sys.exit(1)

        # Allowed values check (non-critical, logs error and returns False if validation fails here)
        if allowed_values:
            # Normalize string values for case-insensitive comparison if value and allowed_values are strings
            comp_value = str(value).lower() if isinstance(value, str) else value
            lower_allowed = [str(v).lower() if isinstance(v, str) else v for v in allowed_values]
            if comp_value not in lower_allowed:
                logger.error(f"{Fore.RED}Validation failed for '{key}': Invalid value '{value}'. Allowed values are: {allowed_values}.{Style.RESET_ALL}")
                return False # Value not in allowed list

        return True # All checks passed or not applicable

    def _get_env(
        self,
        key: str,
        default: Any,
        color: str, # For logging color
        cast_type: Type = str,
        min_val: Optional[Union[int, float, Decimal]] = None,
        max_val: Optional[Union[int, float, Decimal]] = None,
        allowed_values: Optional[List[Any]] = None,
        is_secret: bool = False
    ) -> Any:
        """Streamlined fetching, casting, validating, and defaulting for environment variables."""
        value_str = os.getenv(key)
        source_info = "environment variable"
        use_default_flag = False
        value_to_process_str: str # Will hold the string value to be cast

        if value_str is None or value_str.strip() == "": # Check if env var is not set or is empty string
            if default is None: # Required config, no default
                 # Secrets are critical, non-secrets with no default are also critical
                 log_msg_type = "secret " if is_secret else ""
                 logger.critical(f"{Style.BRIGHT}{Fore.RED}Required {log_msg_type}configuration '{key}' not found in environment and no default provided. Halting.{Style.RESET_ALL}")
                 sys.exit(1)

            use_default_flag = True
            value_to_process_str = str(default) # Use string representation of default for casting
            source_info = f"default value ('{default}')"
            # For logging, display the original default value, not its string representation if it's not a string
            log_value_display = default if not is_secret else "****"
        else:
            value_to_process_str = value_str
            log_value_display = "****" if is_secret else value_to_process_str # Mask secrets

        # Log the found/default value being used
        log_method = logger.warning if use_default_flag and default is not None else logger.info
        # Colorize the log message part
        colored_key_value = f"{color}{key}: {log_value_display}{Style.RESET_ALL}"
        log_method(f"Using {colored_key_value} (from {source_info})")

        # Attempt to cast the value string (either from env or from stringified default)
        casted_value = self._cast_value(key, value_to_process_str, cast_type, default)

        # Validate the casted value. Min/max validation will sys.exit if critical.
        # _validate_value returns False for non-critical issues (e.g., allowed_values mismatch).
        if not self._validate_value(key, casted_value, min_val, max_val, allowed_values):
            # This path is hit if validation failed due to allowed_values or a type error pre-min/max.
            # If min/max failed, _validate_value would have exited.
            # Revert to the original default value provided to _get_env if the casted value from env var failed non-critical validation.
            logger.warning(
                f"{color}Reverting '{key}' to its original default '{default}' due to non-critical validation failure of processed value '{casted_value}'.{Style.RESET_ALL}"
            )
            casted_value = default # Use the original default value passed to the function

            # Critical: Re-validate the original default value itself. This ensures defaults in code are valid.
            if not self._validate_value(key, casted_value, min_val, max_val, allowed_values):
                logger.critical(
                    f"{Style.BRIGHT}{Fore.RED}FATAL: The hardcoded default value '{default}' for '{key}' itself failed validation. Halting.{Style.RESET_ALL}"
                )
                sys.exit(1)
        return casted_value

# --- Exchange Manager Class ---
class ExchangeManager:
    """Handles CCXT exchange interactions, data fetching, formatting, and market information."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.exchange: Optional[ccxt.Exchange] = None
        self.market_info: Optional[Dict[str, Any]] = None
        self._initialize_exchange() # Initializes self.exchange
        if self.exchange: # Load markets only if exchange was successfully initialized
             self.market_info = self._load_market_info() # Initializes self.market_info
        # If initialization failed, critical errors would have been logged and sys.exit called.

    def _initialize_exchange(self):
        """Initializes the CCXT exchange instance for Bybit V5."""
        logger.info(f"Initializing Bybit exchange interface (V5 API, Market Type: {self.config.market_type})...")
        try:
            exchange_params: Dict[str, Any] = {
                "apiKey": self.config.api_key,
                "secret": self.config.api_secret,
                "options": {
                    "defaultType": self.config.market_type, # e.g., 'linear', 'inverse'
                    "adjustForTimeDifference": True, # CCXT handles time sync with server
                    "recvWindow": 10000, # Optional: Increased receive window
                    "brokerId": "PyrmV5NEXUS", # Custom broker ID for Bybit referral/tracking
                    "defaultTimeInForce": "GTC", # Good-Till-Cancelled
                },
            }
            if os.getenv("USE_BYBIT_TESTNET", "false").lower() == "true":
                logger.warning(f"{Fore.YELLOW}Using Bybit Testnet endpoint.{Style.RESET_ALL}")
                exchange_params['urls'] = {'api': 'https://api-testnet.bybit.com'}

            self.exchange = ccxt.bybit(exchange_params)
            logger.debug("Testing exchange connection by fetching server time...")
            self.exchange.fetch_time() # Throws on error
            logger.info(
                f"{Style.BRIGHT}{Fore.GREEN}Bybit V5 interface initialized and connection tested successfully.{Style.RESET_ALL}"
            )

        except ccxt.AuthenticationError as e:
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Authentication failed: {e}. Check API keys and permissions. Halting.{Style.RESET_ALL}",
                exc_info=False, # No need for full stack trace for common auth errors
            )
            sys.exit(1)
        except (ccxt.NetworkError, requests.exceptions.RequestException) as e:
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Network error initializing exchange: {e}. Check internet connection and endpoint. Halting.{Style.RESET_ALL}",
                exc_info=True,
            )
            sys.exit(1)
        except Exception as e: # Catch-all for other unexpected errors
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Unexpected error initializing exchange: {e}. Halting.{Style.RESET_ALL}",
                exc_info=True,
            )
            sys.exit(1)

    def _load_market_info(self) -> Optional[Dict[str, Any]]:
        """Loads and caches market information for the configured symbol, including precision details."""
        if not self.exchange:
            logger.error("Exchange not initialized, cannot load market info.")
            return None # Should not be reached due to checks in __init__
        try:
            logger.info(f"Loading market info for symbol: {self.config.symbol}...")
            # Force reload of markets to get latest info, especially precision
            self.exchange.load_markets(reload=True)
            market = self.exchange.market(self.config.symbol)
            if not market:
                # This can happen if the symbol string is malformed or not supported by the exchange
                raise ccxt.ExchangeError(
                    f"Market {self.config.symbol} not found on exchange after loading markets. "
                    f"Ensure symbol format is correct (e.g., BTC/USDT:USDT for linear, BTC/USD:BTC for inverse)."
                )

            # Precision from CCXT market structure (usually step sizes)
            amount_precision_raw = market.get("precision", {}).get("amount") # e.g., 0.001
            price_precision_raw = market.get("precision", {}).get("price")   # e.g., 0.01

            def get_dp_from_precision_step(precision_val: Optional[Union[str, float, int]], default_dp: int) -> int:
                """Helper to determine decimal places from CCXT precision (step size)."""
                if precision_val is None: return default_dp
                prec_dec = safe_decimal(precision_val) # Convert to Decimal
                if prec_dec.is_nan() or prec_dec.is_zero(): # Invalid or zero step implies integer or error
                    return 0 if prec_dec.is_zero() else default_dp # 0 dp for integer precision

                if prec_dec > 0 and prec_dec < 1: # e.g., 0.1, 0.001 (typical step sizes)
                     exponent = prec_dec.as_tuple().exponent # Negative exponent gives number of DPs
                     return abs(exponent)
                elif prec_dec >= 1: # e.g., 1 (for price like JPY pairs), 10.
                    # If precision is an integer like 1 or 10 (e.g. price must be multiple of 1 or 10),
                    # this means 0 decimal places for the fractional part.
                    if prec_dec.to_integral_value() == prec_dec:
                        return 0
                    else: # e.g. 1.5, 0.5 (uncommon for typical price/amount precision steps)
                        exponent = prec_dec.as_tuple().exponent
                        return abs(exponent) if exponent < 0 else default_dp # Fallback if logic is complex
                else: # Negative or other unexpected precision values
                    return default_dp

            amount_dp_for_formatting = get_dp_from_precision_step(amount_precision_raw, DEFAULT_AMOUNT_DP)
            price_dp_for_formatting = get_dp_from_precision_step(price_precision_raw, DEFAULT_PRICE_DP)

            market["precision_dp"] = {"amount": amount_dp_for_formatting, "price": price_dp_for_formatting}
            # Store actual tick size (price step) and amount step from raw precision values
            market["tick_size"] = safe_decimal(price_precision_raw, default=Decimal('NaN'))
            market["amount_step"] = safe_decimal(amount_precision_raw, default=Decimal('NaN'))

            # Min order size and contract size
            min_amount_raw = market.get("limits", {}).get("amount", {}).get("min")
            market["min_order_size"] = safe_decimal(min_amount_raw, default=Decimal("NaN"))
            # Contract size is crucial for PnL and quantity calculations, especially for inverse.
            # CCXT usually provides this. Default to 1 if missing, but log a warning.
            contract_size_raw = market.get("contractSize")
            market["contract_size"] = safe_decimal(contract_size_raw, default=Decimal("1"))
            if contract_size_raw is None:
                 logger.warning(f"Contract size not found in market info for {self.config.symbol}. Defaulting to 1. This may affect PnL/risk calculations if incorrect.")


            min_amt_str = market["min_order_size"].normalize() if not market["min_order_size"].is_nan() else "N/A"
            tick_size_str = market["tick_size"].normalize() if not market["tick_size"].is_nan() else "N/A"
            amount_step_str = market["amount_step"].normalize() if not market["amount_step"].is_nan() else "N/A"

            logger.info(
                f"Market info for {self.config.symbol} (ID: {market.get('id', 'N/A')}): "
                f"FormattingDP(Amount={amount_dp_for_formatting}, Price={price_dp_for_formatting}), "
                f"ActualSteps(TickSize={tick_size_str}, AmountStep={amount_step_str}), "
                f"Limits(MinAmount={min_amt_str}), ContractSize={market['contract_size'].normalize()}, "
                f"SettleCurrency: {market.get('settle', 'N/A')}"
            )
            return market
        except (ccxt.ExchangeError, KeyError, ValueError, TypeError, Exception) as e:
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Failed to load or parse market info for {self.config.symbol}: {e}. Halting.{Style.RESET_ALL}",
                exc_info=True,
            )
            sys.exit(1)
        return None # Should not be reached if sys.exit is called

    def format_price(self, price: Union[Decimal, str, float, int]) -> str:
        """Formats a price value to a string according to market precision using ROUND_HALF_EVEN."""
        price_decimal = safe_decimal(price)
        if price_decimal.is_nan():
            return "NaN" # Consistent NaN string

        precision_dp = DEFAULT_PRICE_DP # Fallback
        if self.market_info and "precision_dp" in self.market_info and "price" in self.market_info["precision_dp"]:
            precision_dp = self.market_info["precision_dp"]["price"]

        # For API calls, it's often better to use exchange.price_to_precision(symbol, price_float).
        # For display or internal use, quantize is fine.
        try:
            # Using f-string formatting for fixed decimal places, which inherently rounds (usually half-even for Python 3).
            # For exact control over rounding, quantize first.
            quantizer = Decimal("1e-" + str(precision_dp))
            formatted_price_decimal = price_decimal.quantize(quantizer, rounding=ROUND_HALF_EVEN)
            return f"{formatted_price_decimal:.{precision_dp}f}"
        except (InvalidOperation, ValueError) as e: # Should be rare with safe_decimal input
             logger.error(f"Error formatting price {price_decimal} to {precision_dp}dp: {e}")
             return "ERR" # Consistent error string

    def format_amount(
        self, amount: Union[Decimal, str, float, int], rounding_mode=ROUND_DOWN
    ) -> str:
        """Formats an amount (quantity) to a string according to market precision, default ROUND_DOWN."""
        amount_decimal = safe_decimal(amount)
        if amount_decimal.is_nan():
            return "NaN"

        precision_dp = DEFAULT_AMOUNT_DP # Fallback
        if self.market_info and "precision_dp" in self.market_info and "amount" in self.market_info["precision_dp"]:
            precision_dp = self.market_info["precision_dp"]["amount"]

        # Similar to price, exchange.amount_to_precision(symbol, amount_float) is an option for API calls.
        try:
            quantizer = Decimal("1e-" + str(precision_dp))
            formatted_amount_decimal = amount_decimal.quantize(quantizer, rounding=rounding_mode)
            return f"{formatted_amount_decimal:.{precision_dp}f}"
        except (InvalidOperation, ValueError) as e:
             logger.error(f"Error formatting amount {amount_decimal} to {precision_dp}dp: {e}")
             return "ERR"

    def _format_v5_param(
            self,
            value: Optional[Union[Decimal, str, float, int]],
            param_type: str = "price", # 'price', 'amount', or 'distance' (distance uses price precision)
            allow_zero: bool = False # Whether "0" or "0.00..." is a valid formatted parameter
        ) -> Optional[str]:
        """
        Formats a numeric value as a string suitable for Bybit V5 API parameters.
        Returns None if the value is invalid, cannot be formatted positively (unless allow_zero=True),
        or if formatting itself fails. For clearing stops, "0" is often used.
        """
        if value is None:
            return None # None input means no parameter should be sent or use API default

        decimal_value = safe_decimal(value, default=Decimal("NaN"))

        if decimal_value.is_nan():
            logger.warning(f"V5 Param Formatting: Input '{value}' (type: {type(value).__name__}) converted to NaN Decimal. Cannot format.")
            return None

        is_zero_val = decimal_value.is_zero() # is_zero() is True for +0, -0
        if is_zero_val:
            if allow_zero:
                # Format zero according to the parameter type's precision (typically price for stops)
                # Bybit API expects "0" as a string, not "0.00" for clearing stops.
                return "0"
            else: # Zero not allowed for this parameter (e.g., a TSL distance that must be positive)
                logger.debug(f"V5 Param Formatting: Input value '{value}' is zero, but zero is not allowed for this parameter type ('{param_type}').")
                return None
        # For non-zero values, ensure they are positive for price/amount/distance parameters
        elif decimal_value < 0:
            logger.warning(f"V5 Param Formatting: Input value '{value}' is negative ({decimal_value}), which is typically invalid for API price/amount/distance parameters.")
            return None

        # Proceed with formatting for positive values
        rounding_for_amount = ROUND_DOWN # Default rounding for amounts (conservative)

        formatted_str: str
        if param_type == "price" or param_type == "distance":
            # Price and distance (which is a price delta) use price formatting rules
            # Use CCXT's built-in formatter for API calls if available and reliable,
            # otherwise use custom formatter that matches exchange requirements.
            # Here, using our custom formatter for consistency.
            # Ensure the exchange is available for ccxt methods
            if self.exchange and self.config.symbol:
                try:
                    # Using `decimal_to_precision` which takes string output format.
                    # It applies tick size rounding.
                    if param_type == "price":
                        formatted_str = self.exchange.price_to_precision(self.config.symbol, float(decimal_value))
                    else: # distance is a price delta, so also use price_to_precision.
                        formatted_str = self.exchange.price_to_precision(self.config.symbol, float(decimal_value))
                    # Verify it's not "NaN" or "ERR"
                    if safe_decimal(formatted_str).is_nan(): raise ValueError("CCXT formatting resulted in NaN")
                except Exception as e_ccxt_format:
                    logger.warning(f"CCXT {param_type}_to_precision failed ({e_ccxt_format}), falling back to custom format_price for V5 param.")
                    formatted_str = self.format_price(decimal_value) # Fallback
            else:
                formatted_str = self.format_price(decimal_value) # Fallback if exchange not ready
        elif param_type == "amount":
            if self.exchange and self.config.symbol:
                try:
                    formatted_str = self.exchange.amount_to_precision(self.config.symbol, float(decimal_value))
                    if safe_decimal(formatted_str).is_nan(): raise ValueError("CCXT formatting resulted in NaN")
                except Exception as e_ccxt_format:
                    logger.warning(f"CCXT amount_to_precision failed ({e_ccxt_format}), falling back to custom format_amount for V5 param.")
                    formatted_str = self.format_amount(decimal_value, rounding_mode=rounding_for_amount) # Fallback
            else:
                formatted_str = self.format_amount(decimal_value, rounding_mode=rounding_for_amount) # Fallback
        else:
            logger.error(f"V5 Param Formatting: Unknown param_type '{param_type}'. Cannot format '{value}'.")
            return None

        if formatted_str in ["ERR", "NaN"] or safe_decimal(formatted_str).is_nan(): # Double check output
            logger.error(f"V5 Param Formatting: Failed to produce a valid string for '{value}' (type: {param_type}). Formatter returned: {formatted_str}")
            return None
        return formatted_str

    def fetch_ohlcv(self) -> Optional[pd.DataFrame]:
        """Fetches OHLCV data with retries, converts to DataFrame, and processes numeric columns."""
        if not self.exchange:
            logger.error("Exchange not initialized, cannot fetch OHLCV.")
            return None
        logger.debug(
            f"Fetching up to {self.config.ohlcv_limit} OHLCV candles for {self.config.symbol} (Timeframe: {self.config.interval})..."
        )
        try:
            ohlcv_data = fetch_with_retries(
                self.exchange.fetch_ohlcv,
                symbol=self.config.symbol,
                timeframe=self.config.interval,
                limit=self.config.ohlcv_limit,
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
            )
            if not ohlcv_data: # Handles empty list return
                logger.error(f"fetch_ohlcv for {self.config.symbol} returned no data (empty list).")
                return None
            if len(ohlcv_data) < 20: # Warn if insufficient for longer lookback indicators
                 logger.warning(f"Fetched only {len(ohlcv_data)} candles. This might be insufficient for some indicators requiring longer lookbacks (e.g., >20 periods).")

            df = pd.DataFrame(
                ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)

            # Convert OHLCV columns to Decimal, handling potential NaNs robustly
            for col in ["open", "high", "low", "close", "volume"]:
                # Apply safe_decimal to each element. It returns Decimal('NaN') for unparseable values.
                df[col] = df[col].apply(safe_decimal) # Use .apply for element-wise conversion
                # Check if any conversion resulted in Decimal('NaN')
                if df[col].apply(lambda x: isinstance(x, Decimal) and x.is_nan()).any():
                     logger.warning(f"Column '{col}' in OHLCV data contains NaN values after Decimal conversion. Check API data source if issues persist.")

            initial_len = len(df)
            # Drop rows if any of O/H/L/C is Decimal('NaN') after conversion
            # A row is dropped if any of these critical columns is NaN
            df.dropna(subset=["open", "high", "low", "close"], inplace=True, how="any",
                      # Custom checker for Decimal('NaN')
                      # This is tricky with dropna which expects np.nan.
                      # A better way is to replace Decimal('NaN') with np.nan before dropna, or filter manually.
                     )
            # Manual filter for Decimal('NaN')
            for col in ["open", "high", "low", "close"]:
                 df = df[~df[col].apply(lambda x: isinstance(x, Decimal) and x.is_nan())]

            if len(df) < initial_len:
                logger.warning(f"Dropped {initial_len - len(df)} rows from OHLCV data due to NaN values in critical O/H/L/C columns.")

            if df.empty: # Check if DataFrame became empty after NaN drop or was initially empty and unparseable
                 logger.error("OHLCV DataFrame is empty after processing (NaN drop or initial empty). Cannot proceed with this data.")
                 return None

            logger.debug(
                f"Fetched and processed {len(df)} OHLCV candles. Last timestamp: {df.index[-1]}"
            )
            return df
        except Exception as e: # Catch-all for fetch_with_retries issues or DataFrame processing
            logger.error(f"Failed to fetch or process OHLCV data for {self.config.symbol}: {e}", exc_info=True)
            return None

    def get_balance(self) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Fetches total equity and available balance for the settlement currency using V5 API."""
        if not self.exchange or not self.market_info:
            logger.error("Exchange or market info not available, cannot fetch balance.")
            return None, None

        settle_currency = self.market_info.get("settle")
        if not settle_currency:
            logger.error("Settle currency not found in market info. Cannot determine balance currency.")
            return None, None

        logger.debug(
            f"Fetching balance for {settle_currency} (Account: {V5_UNIFIED_ACCOUNT_TYPE}, Category: {self.config.bybit_v5_category})..."
        )
        try:
            params = {
                "accountType": V5_UNIFIED_ACCOUNT_TYPE,
                "coin": settle_currency, # Request specific coin balance for V5
            }
            # For V5, fetch_balance with accountType=UNIFIED and coin might give specific coin details,
            # or sometimes the overall account balance. The 'info' field is key.
            balance_data = fetch_with_retries(
                self.exchange.fetch_balance,
                params=params,
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
            )

            total_equity = Decimal("NaN")
            available_balance = Decimal("NaN")

            # Bybit V5 `GET /v5/account/wallet-balance` structure is usually in `balance_data['info']`
            if "info" in balance_data and "result" in balance_data["info"] and "list" in balance_data["info"]["result"]:
                account_list = balance_data["info"]["result"]["list"]
                if account_list and isinstance(account_list, list):
                    # Find the UNIFIED account type details
                    unified_acc_info = next((item for item in account_list if item.get("accountType") == V5_UNIFIED_ACCOUNT_TYPE), None)
                    if unified_acc_info:
                        # Overall equity for the UNIFIED account
                        total_equity_raw = unified_acc_info.get("totalEquity")
                        if total_equity_raw is not None: total_equity = safe_decimal(total_equity_raw)

                        # Try to get coin-specific available balance first
                        coin_details_list = unified_acc_info.get("coin", [])
                        if coin_details_list and isinstance(coin_details_list, list):
                            settle_coin_info = next((c for c in coin_details_list if c.get("coin") == settle_currency), None)
                            if settle_coin_info:
                                # availableToWithdraw is usually the most relevant for new trades for that specific coin
                                available_balance_raw = settle_coin_info.get("availableToWithdraw")
                                if available_balance_raw is not None: available_balance = safe_decimal(available_balance_raw)
                                # If specific coin equity is available and totalEquity was not, use it
                                if total_equity.is_nan() and settle_coin_info.get("equity") is not None:
                                    total_equity = safe_decimal(settle_coin_info.get("equity"))


                        # Fallback to total available balance if coin-specific not found or preferred
                        if available_balance.is_nan() and unified_acc_info.get("totalAvailableBalance") is not None:
                            available_balance = safe_decimal(unified_acc_info.get("totalAvailableBalance"))
                            logger.debug(f"Used 'totalAvailableBalance' for {settle_currency} as coin-specific 'availableToWithdraw' was not found/parsed.")

            # If parsing from 'info' failed, try CCXT's standardized structure as a last resort (less common for V5 UTA details)
            if total_equity.is_nan() and balance_data.get("total", {}).get(settle_currency) is not None:
                total_equity = safe_decimal(balance_data["total"].get(settle_currency))
                logger.debug("Used CCXT standardized 'total' balance field as fallback.")
            if available_balance.is_nan() and balance_data.get("free", {}).get(settle_currency) is not None:
                available_balance = safe_decimal(balance_data["free"].get(settle_currency))
                logger.debug("Used CCXT standardized 'free' balance field as fallback.")


            if total_equity.is_nan():
                logger.error(
                    f"Could not extract valid total equity for {settle_currency}. Balance data format might be unexpected or field missing. Raw 'info.result.list' snippet: {str(balance_data.get('info',{}).get('result',{}).get('list',''))[:300]}"
                ) # Log part of the relevant structure
                # Return None for total_equity to signal failure, but provide available_balance if found
                return None, available_balance if not available_balance.is_nan() else Decimal("0")

            if available_balance.is_nan():
                logger.warning(
                    f"Could not extract valid available balance for {settle_currency}. Defaulting to 0. Check raw balance data if issues persist."
                )
                available_balance = Decimal("0") # Safe default

            logger.debug(
                f"Balance Fetched ({settle_currency}): Total Equity = {total_equity.normalize()}, Available Balance = {available_balance.normalize()}"
            )
            return total_equity, available_balance
        except Exception as e: # Catch-all for fetch_with_retries issues or parsing
            logger.error(f"Failed to fetch or parse balance: {e}", exc_info=True)
            return None, None

    def get_current_position(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Fetches current position details for the configured symbol using V5 API.
        Returns a dictionary structured as {'long': {details}, 'short': {details}} or None on error.
        Empty dicts for 'long'/'short' if no position or if position for the configured positionIdx is flat.
        """
        if not self.exchange or not self.market_info:
            logger.error("Exchange or market info not available, cannot fetch position.")
            return None

        market_id = self.market_info.get("id") # CCXT market ID, e.g., BTCUSDT
        if not market_id:
            logger.error("Market ID not found in market info. Cannot fetch position.")
            return None

        logger.debug(
            f"Fetching position for {self.config.symbol} (API ID: {market_id}, Category: {self.config.bybit_v5_category}, Target PositionIdx: {self.config.position_idx})..."
        )
        positions_summary: Dict[str, Dict[str, Any]] = {"long": {}, "short": {}}

        try:
            # For Bybit V5, fetch_positions requires category.
            # CCXT's `fetch_positions` for Bybit V5 should ideally handle the `symbol` and `category` to query `/v5/position/list`.
            params = {
                "category": self.config.bybit_v5_category,
                "symbol": market_id, # Explicitly pass market_id for V5
                # "settleCoin": self.market_info.get("settle") # Optional: can specify settleCoin if needed
            }
            # `fetch_positions` returns a list of positions. We are interested in the one matching our symbol and configured positionIdx.
            fetched_positions_list_ccxt = fetch_with_retries(
                self.exchange.fetch_positions,
                symbols=[self.config.symbol], # Request for our specific symbol
                params=params,
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
            )

            if not fetched_positions_list_ccxt:
                logger.debug("No position data returned from fetch_positions (empty list). Assuming flat for this symbol/category.")
                return positions_summary

            # The `info` field in CCXT's unified position structure contains the raw API response for that position entry.
            # We need to iterate through the list and find the entry matching our `config.position_idx`.
            target_pos_info_raw_api = None # This will hold the raw API data for the matched position
            for pos_data_ccxt_unified in fetched_positions_list_ccxt:
                raw_api_entry = pos_data_ccxt_unified.get("info", {})
                pos_idx_from_api_str = raw_api_entry.get("positionIdx") # positionIdx from API as string "0", "1", "2"

                try:
                    pos_idx_from_api_int = int(pos_idx_from_api_str) if pos_idx_from_api_str is not None else -1 # Default if missing
                except ValueError:
                    logger.warning(f"Could not parse positionIdx '{pos_idx_from_api_str}' from an API position entry. Skipping this entry.")
                    continue

                # Check if this entry's positionIdx matches our configured one
                if pos_idx_from_api_int == self.config.position_idx:
                    target_pos_info_raw_api = raw_api_entry
                    logger.debug(f"Found position entry in API list matching configured positionIdx={self.config.position_idx}: {target_pos_info_raw_api}")
                    break # Found the specific position context we're interested in

            if not target_pos_info_raw_api:
                logger.debug(
                    f"No position entry found in API list matching configured positionIdx={self.config.position_idx} for symbol {market_id}. Assuming flat for this specific position index."
                )
                return positions_summary # Return empty summary, meaning no position for this idx

            # Parse details from the matched raw_pos_info_raw_api
            qty_str = target_pos_info_raw_api.get("size", "0")
            api_side_str = target_pos_info_raw_api.get("side", "None").lower() # 'Buy', 'Sell', or 'None' if flat for that idx

            qty_decimal = safe_decimal(qty_str)
            # Position size must be positive for our logic; API 'side' determines direction for one-way mode.
            # For hedge mode, positionIdx determines direction.
            qty_abs = qty_decimal.copy_abs() if not qty_decimal.is_nan() else Decimal("0")

            # Consider position effectively closed if quantity is very small
            is_position_effectively_open = qty_abs >= POSITION_QTY_EPSILON

            if not is_position_effectively_open:
                logger.debug(f"Position size {qty_abs.normalize()} for Idx {self.config.position_idx} is negligible or zero. Considered flat for this index.")
                return positions_summary # Return empty summary

            # Position is open for this index, now determine logical side ('long' or 'short') for our internal representation
            entry_price = safe_decimal(target_pos_info_raw_api.get("avgPrice", "0"))
            liq_price = safe_decimal(target_pos_info_raw_api.get("liqPrice", "0"))
            unrealized_pnl = safe_decimal(target_pos_info_raw_api.get("unrealisedPnl", "0"))
            # V5 specific fields for stops on position (from /v5/position/list)
            sl_price_api = safe_decimal(target_pos_info_raw_api.get("stopLoss", "0"))
            tp_price_api = safe_decimal(target_pos_info_raw_api.get("takeProfit", "0"))
            tsl_trigger_price_api = safe_decimal(target_pos_info_raw_api.get("trailingStop", "0")) # This is TSL activation price if TSL is set
            # Note: 'tpslMode' ('Full' or 'Partial') and 'tpTriggerBy'/'slTriggerBy' are also in raw_pos_info.

            position_side_key: Optional[str] = None
            if self.config.position_idx == 0: # One-Way Mode
                # 'side' field from API ('Buy' or 'Sell') determines if it's long or short
                if api_side_str == "buy": position_side_key = "long"
                elif api_side_str == "sell": position_side_key = "short"
                # If side is "None" but size > 0, this is an inconsistent state or specific to exchange logic.
                # For Bybit V5 one-way, a position with size > 0 should have 'side' as 'Buy' or 'Sell'.
                elif api_side_str == "none" and qty_abs > 0:
                     logger.warning(f"Inconsistent state for One-Way mode (Idx 0): API side is 'None' but size is {qty_abs.normalize()}. Check exchange position details.")
                     # Cannot reliably determine side, treat as error / flat for safety here.
                     return positions_summary


            elif self.config.position_idx == 1: # Hedge Mode - Buy Side Position (Long)
                position_side_key = "long"
                 # In hedge mode, 'side' from API should be 'Buy' if size > 0 for positionIdx 1
                if api_side_str != "buy" and qty_abs > 0:
                    logger.warning(f"Potential mismatch for Hedge Mode Buy (Idx 1): API side is '{api_side_str}' but expected 'Buy' with size {qty_abs.normalize()}. Assuming long.")

            elif self.config.position_idx == 2: # Hedge Mode - Sell Side Position (Short)
                position_side_key = "short"
                # In hedge mode, 'side' from API should be 'Sell' if size > 0 for positionIdx 2
                if api_side_str != "sell" and qty_abs > 0:
                    logger.warning(f"Potential mismatch for Hedge Mode Sell (Idx 2): API side is '{api_side_str}' but expected 'Sell' with size {qty_abs.normalize()}. Assuming short.")


            if position_side_key:
                position_details_for_summary = {
                    "qty": qty_abs, # Always positive quantity for our tracking
                    "entry_price": entry_price if not entry_price.is_nan() and entry_price > 0 else Decimal("NaN"),
                    "liq_price": liq_price if not liq_price.is_nan() and liq_price > 0 else Decimal("NaN"),
                    "unrealized_pnl": unrealized_pnl if not unrealized_pnl.is_nan() else Decimal("0"),
                    "api_side": api_side_str, # Original 'side' from API ('Buy', 'Sell', 'None')
                    "info": target_pos_info_raw_api, # Raw data for debugging or further use
                    "stop_loss_price": sl_price_api if not sl_price_api.is_nan() and sl_price_api > 0 else None,
                    "take_profit_price": tp_price_api if not tp_price_api.is_nan() and tp_price_api > 0 else None,
                    # TSL is active if 'trailingStop' (activation price) is set and positive
                    "is_tsl_active": not tsl_trigger_price_api.is_nan() and tsl_trigger_price_api > 0,
                    "tsl_trigger_price": tsl_trigger_price_api if not tsl_trigger_price_api.is_nan() and tsl_trigger_price_api > 0 else None,
                }
                positions_summary[position_side_key] = position_details_for_summary
                entry_str = position_details_for_summary["entry_price"].normalize() if position_details_for_summary["entry_price"] and not position_details_for_summary["entry_price"].is_nan() else "N/A"
                logger.debug(
                    f"Identified {position_side_key.upper()} position (Idx {self.config.position_idx}): Qty={qty_abs.normalize()}, Entry={entry_str}"
                )
            else:
                 # This case might happen if positionIdx is valid, size > 0, but logic couldn't map to long/short (e.g. one-way side="None")
                 logger.warning(f"Position found with size {qty_abs.normalize()} for Idx {self.config.position_idx} but could not determine logical long/short state reliably (api_side: '{api_side_str}'). Treating as flat for safety.")
                 return positions_summary # Return empty summary

            return positions_summary

        except Exception as e: # Catch-all for fetch_with_retries issues or parsing
            logger.error(
                f"Failed to fetch or parse positions for {self.config.symbol}: {e}", exc_info=True
            )
            return None


# --- Indicator Calculator Class ---
class IndicatorCalculator:
    """Calculates technical indicators (EMAs, Stochastic, ATR, ADX) needed for the trading strategy."""

    def __init__(self, config: TradingConfig):
        self.config = config

    def calculate_indicators(
        self, df: pd.DataFrame
    ) -> Optional[Dict[str, Union[Decimal, bool, int]]]:
        """
        Calculates EMAs, Stochastic (%K, %D, prev %K), ATR, and ADX from OHLCV DataFrame.
        Uses robust data conversion to float for calculations and Decimal for output.
        Returns a dictionary of indicators or None on critical failure.
        """
        logger.info(
            f"{Fore.CYAN}# Weaving indicator patterns (EMA, Stoch, ATR, ADX)...{Style.RESET_ALL}"
        )
        if df is None or df.empty:
            logger.error(f"{Fore.RED}No DataFrame provided for indicator calculation.{Style.RESET_ALL}")
            return None

        required_ohlc_cols = ["open", "high", "low", "close"]
        if not all(c in df.columns for c in required_ohlc_cols):
            missing_cols = [c for c in required_ohlc_cols if c not in df.columns]
            logger.error(f"{Fore.RED}DataFrame missing required columns for indicators: {missing_cols}{Style.RESET_ALL}")
            return None

        try:
            # Work with a copy for calculations, ensure original Decimal types are not altered if df is passed around
            df_calc = df[required_ohlc_cols].copy()

            # Convert Decimal columns to float for TA library compatibility / performance
            # Handles various input types within the DataFrame cells (Decimal, str, float, int, None)
            def safe_to_float(x: Any) -> float:
                if isinstance(x, (float, int)): return float(x)
                if isinstance(x, Decimal): return float('nan') if x.is_nan() else float(x)
                if isinstance(x, str):
                    try:
                        val_stripped = x.strip().lower()
                        if val_stripped in ["nan", "none", "null", ""]: return float('nan')
                        return float(val_stripped)
                    except ValueError:
                        logger.debug(f"Could not convert string '{x}' to float for TA calculation.")
                        return float('nan')
                if x is None: return float('nan')
                # For any other unexpected types
                logger.warning(f"Unexpected type {type(x).__name__} ('{x}'), converting to NaN for TA calculation.")
                return float('nan')

            for col in required_ohlc_cols:
                if df_calc[col].empty: # Handle empty series if a column was all NaNs initially
                    logger.warning(f"Column '{col}' is empty before conversion. Ensuring float type.")
                    df_calc[col] = pd.Series(dtype=float) # Ensure it's float type even if empty
                    continue
                df_calc[col] = df_calc[col].apply(safe_to_float) # Use .apply for element-wise
                df_calc[col] = df_calc[col].astype(float) # Final cast to float, handles if map returned objects

            # Drop rows with NaN in any critical OHLC column *after* float conversion
            initial_len = len(df_calc)
            df_calc.dropna(subset=required_ohlc_cols, inplace=True, how='any')
            rows_dropped = initial_len - len(df_calc)
            if rows_dropped > 0:
                 logger.debug(f"Dropped {rows_dropped} rows with NaN in OHLC columns after float conversion for TA.")

            if df_calc.empty:
                logger.error(f"{Fore.RED}DataFrame became empty after NaN drop during indicator pre-processing.{Style.RESET_ALL}")
                return None

            # Ensure sufficient data for the longest period indicator + some buffer
            max_period_needed = max(
                self.config.slow_ema_period, self.config.trend_ema_period,
                self.config.stoch_period + self.config.stoch_smooth_k + self.config.stoch_smooth_d, # Sum for full Stoch history
                self.config.atr_period,
                self.config.adx_period * 2, # ADX typically needs 2*period for smoothing internal DX
            )
            min_required_data_length = max_period_needed + 20 # Buffer for stable calculation
            if len(df_calc) < min_required_data_length:
                logger.error(f"{Fore.RED}Insufficient data ({len(df_calc)} rows) for robust indicator calculation (requires ~{min_required_data_length} rows).{Style.RESET_ALL}")
                return None

            # Access columns as Series
            close_s = df_calc["close"]
            high_s = df_calc["high"]
            low_s = df_calc["low"]

            # --- EMAs ---
            fast_ema_s = close_s.ewm(span=self.config.fast_ema_period, adjust=False).mean()
            slow_ema_s = close_s.ewm(span=self.config.slow_ema_period, adjust=False).mean()
            trend_ema_s = close_s.ewm(span=self.config.trend_ema_period, adjust=False).mean()

            # --- Stochastic ---
            # Rolling min/max over stoch_period
            low_min_stoch = low_s.rolling(window=self.config.stoch_period, min_periods=max(1, self.config.stoch_period // 2)).min()
            high_max_stoch = high_s.rolling(window=self.config.stoch_period, min_periods=max(1, self.config.stoch_period // 2)).max()
            stoch_range = high_max_stoch - low_min_stoch
            # Calculate %K raw, handle division by zero if range is 0 (set to 50 as neutral)
            # Use a small epsilon for float comparison to avoid division by tiny numbers that are effectively zero
            stoch_k_raw_values = np.where(stoch_range > 1e-12,
                                      100 * (close_s - low_min_stoch) / stoch_range,
                                      50.0) # Default to 50 if range is zero or too small
            stoch_k_raw_s = pd.Series(stoch_k_raw_values, index=df_calc.index).fillna(50) # Fill initial NaNs from rolling
            # Smooth %K raw to get final %K
            stoch_k_s = stoch_k_raw_s.rolling(window=self.config.stoch_smooth_k, min_periods=1).mean().fillna(50)
            # Smooth final %K to get %D
            stoch_d_s = stoch_k_s.rolling(window=self.config.stoch_smooth_d, min_periods=1).mean().fillna(50)


            # --- ATR (Wilder's ATR) ---
            prev_close_s = close_s.shift(1)
            tr1 = high_s - low_s
            tr2 = (high_s - prev_close_s).abs()
            tr3 = (low_s - prev_close_s).abs()
            true_range_s = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(0) # Fill initial NaN TR with 0
            # Standard EMA for ATR (alpha = 1/N)
            atr_s = true_range_s.ewm(span=self.config.atr_period, adjust=False).mean()

            # --- ADX ---
            adx_s, pdi_s, mdi_s = self._calculate_adx(
                high_s, low_s, close_s, atr_s, self.config.adx_period
            )

            # Helper to get latest valid Decimal value from a Series
            def get_latest_decimal_from_series(series: pd.Series, indicator_name: str) -> Decimal:
                valid_series = series.dropna() # Drop NaNs before taking last value
                if valid_series.empty:
                    logger.warning(f"Indicator series '{indicator_name}' is empty or all NaNs after dropna.")
                    return Decimal("NaN")
                last_valid_float = valid_series.iloc[-1]
                # Convert float to string, then to Decimal for precision, robustly
                return safe_decimal(str(last_valid_float))


            # Prepare output dictionary
            indicators_out: Dict[str, Union[Decimal, bool, int]] = {
                "fast_ema": get_latest_decimal_from_series(fast_ema_s, "fast_ema"),
                "slow_ema": get_latest_decimal_from_series(slow_ema_s, "slow_ema"),
                "trend_ema": get_latest_decimal_from_series(trend_ema_s, "trend_ema"),
                "stoch_k": get_latest_decimal_from_series(stoch_k_s, "stoch_k"),
                "stoch_d": get_latest_decimal_from_series(stoch_d_s, "stoch_d"),
                "atr": get_latest_decimal_from_series(atr_s, "atr"),
                "atr_period": self.config.atr_period, # Include period for context
                "adx": get_latest_decimal_from_series(adx_s, "adx"),
                "pdi": get_latest_decimal_from_series(pdi_s, "pdi"),
                "mdi": get_latest_decimal_from_series(mdi_s, "mdi"),
            }

            # Previous Stochastic %K value
            stoch_k_valid_series = stoch_k_s.dropna()
            stoch_k_prev_val = Decimal("NaN")
            if len(stoch_k_valid_series) >= 2: # Need at least two valid points for a previous value
                stoch_k_prev_val = safe_decimal(str(stoch_k_valid_series.iloc[-2]))
            indicators_out["stoch_k_prev"] = stoch_k_prev_val

            # Stochastic K/D Cross
            k_now = indicators_out["stoch_k"]
            d_now = indicators_out["stoch_d"]
            k_prev = indicators_out["stoch_k_prev"] # K at t-1

            stoch_d_valid_series = stoch_d_s.dropna()
            d_prev_val = Decimal("NaN") # D at t-1
            if len(stoch_d_valid_series) >=2:
                d_prev_val = safe_decimal(str(stoch_d_valid_series.iloc[-2]))

            stoch_kd_bullish_cross = False
            stoch_kd_bearish_cross = False
            if not any(v.is_nan() for v in [k_now, d_now, k_prev, d_prev_val]): # Ensure all values are valid
                if (k_prev <= d_prev_val) and (k_now > d_now): stoch_kd_bullish_cross = True
                if (k_prev >= d_prev_val) and (k_now < d_now): stoch_kd_bearish_cross = True

            indicators_out["stoch_kd_bullish"] = stoch_kd_bullish_cross
            indicators_out["stoch_kd_bearish"] = stoch_kd_bearish_cross


            # Check critical indicators for NaN, especially ATR
            critical_indicator_keys_for_check = [
                "fast_ema", "slow_ema", "trend_ema", "atr",
                "stoch_k", "stoch_d", # k_prev can be NaN if not enough history, less critical for initial check
                "adx", "pdi", "mdi",
            ]
            failed_indicators = [
                k for k in critical_indicator_keys_for_check if indicators_out.get(k, Decimal("NaN")).is_nan()
            ]
            if failed_indicators:
                # ATR is particularly critical for risk calculation
                if indicators_out.get("atr", Decimal("NaN")).is_nan():
                     logger.error(f"{Fore.RED}CRITICAL: ATR calculated as NaN. Risk calculations will fail. Aborting indicator calculation result.{Style.RESET_ALL}")
                     return None # Cannot proceed without ATR
                logger.warning( # Changed to warning if only non-ATR critical indicators are NaN
                    f"{Fore.YELLOW}Warning: Some critical indicators calculated as NaN: {', '.join(failed_indicators)}. This may impair signal generation.{Style.RESET_ALL}"
                )


            logger.info(f"{Style.BRIGHT}{Fore.GREEN}Indicator patterns woven successfully.{Style.RESET_ALL}")
            return indicators_out

        except Exception as e: # Catch-all for unexpected errors during calculation
            logger.error(f"{Fore.RED}Error weaving indicator patterns: {e}{Style.RESET_ALL}", exc_info=True)
            return None

    def _calculate_adx(
        self,
        high_s: pd.Series, low_s: pd.Series, close_s: pd.Series,
        atr_s: pd.Series, # ATR Series needed for DI calculation
        period: int,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Helper to calculate ADX, +DI, -DI using Wilder's smoothing (EMA with alpha = 1/N)."""
        if period <= 0:
            logger.error("ADX period must be a positive integer.")
            nan_series = pd.Series(np.nan, index=high_s.index)
            return nan_series, nan_series, nan_series
        if atr_s.empty or atr_s.isnull().all():
             logger.error("ATR series is empty or all NaN. Cannot calculate ADX components.")
             nan_series = pd.Series(np.nan, index=high_s.index)
             return nan_series, nan_series, nan_series

        # Calculate +DM and -DM
        move_up = high_s.diff()
        move_down = -low_s.diff() # low_s.diff() is L(t) - L(t-1), so -low_s.diff() is L(t-1) - L(t)

        plus_dm_values = np.where((move_up > move_down) & (move_up > 0), move_up, 0.0)
        minus_dm_values = np.where((move_down > move_up) & (move_down > 0), move_down, 0.0)

        plus_dm_s = pd.Series(plus_dm_values, index=high_s.index).fillna(0) # Fill initial NaN from diff
        minus_dm_s = pd.Series(minus_dm_values, index=high_s.index).fillna(0) # Fill initial NaN from diff

        # Smooth DMs using Wilder's EMA (alpha = 1/period)
        alpha = 1.0 / period
        smoothed_plus_dm_s = plus_dm_s.ewm(alpha=alpha, adjust=False, min_periods=period).mean().fillna(0)
        smoothed_minus_dm_s = minus_dm_s.ewm(alpha=alpha, adjust=False, min_periods=period).mean().fillna(0)

        # Calculate +DI and -DI, ensure ATR is not zero or NaN to prevent division errors
        # Use a small epsilon for ATR in denominator
        pdi_values = np.where((atr_s > 1e-12) & (~atr_s.isnull()), 100 * smoothed_plus_dm_s / atr_s, 0.0)
        mdi_values = np.where((atr_s > 1e-12) & (~atr_s.isnull()), 100 * smoothed_minus_dm_s / atr_s, 0.0)
        pdi_s_out = pd.Series(pdi_values, index=high_s.index).fillna(0)
        mdi_s_out = pd.Series(mdi_values, index=high_s.index).fillna(0)

        # Calculate DX
        di_diff_abs = (pdi_s_out - mdi_s_out).abs()
        di_sum = pdi_s_out + mdi_s_out
        # Ensure di_sum is not zero (or very small) to prevent division errors
        dx_values = np.where(di_sum > 1e-12, 100 * di_diff_abs / di_sum, 0.0)
        dx_s = pd.Series(dx_values, index=high_s.index).fillna(0)

        # Calculate ADX (smoothed DX using Wilder's EMA)
        adx_s_out = dx_s.ewm(alpha=alpha, adjust=False, min_periods=period).mean().fillna(0)

        return adx_s_out, pdi_s_out, mdi_s_out


# --- Signal Generator Class ---
class SignalGenerator:
    """Generates trading entry and exit signals based on indicator conditions."""

    def __init__(self, config: TradingConfig):
        self.config = config

    def generate_signals(
        self,
        df_last_candles: pd.DataFrame, # Needs at least last 2 candles for prev_close for ATR move filter
        indicators: Dict[str, Union[Decimal, bool, int]],
    ) -> Dict[str, Union[bool, str]]:
        """Generates 'long'/'short' entry signals and provides a detailed reason string."""
        result: Dict[str, Union[bool, str]] = {
            "long": False, "short": False, "reason": "Initializing signal check",
        }

        if not indicators:
            result["reason"] = "No Signal: Indicators data missing."
            logger.debug(result["reason"])
            return result
        if df_last_candles is None or len(df_last_candles) < 2: # Need 2 for current and previous close for ATR move filter
            reason = f"No Signal: Insufficient candle data (requires >=2 for ATR move filter, got {len(df_last_candles) if df_last_candles is not None else 0})."
            result["reason"] = reason
            logger.debug(reason)
            return result

        try:
            latest_candle = df_last_candles.iloc[-1]
            prev_candle = df_last_candles.iloc[-2] # For ATR move filter using price change from prev_close
            current_price = safe_decimal(latest_candle["close"])
            prev_close = safe_decimal(prev_candle["close"]) # Used for ATR move filter

            if current_price.is_nan() or current_price <= 0:
                result["reason"] = f"No Signal: Invalid current price ({current_price.normalize() if not current_price.is_nan() else 'NaN'})."
                logger.warning(result["reason"])
                return result

            # Extract necessary indicator values, ensuring they are Decimals and not NaN
            required_indicator_keys = [
                "stoch_k", "fast_ema", "slow_ema", "trend_ema", "atr", "adx", "pdi", "mdi"
            ]
            ind_values: Dict[str, Decimal] = {} # Store valid Decimal indicator values
            nan_keys = []
            for key in required_indicator_keys:
                val = indicators.get(key)
                # Ensure value is Decimal and not NaN. Booleans (like K/D cross) are handled separately.
                if isinstance(val, Decimal) and not val.is_nan():
                    ind_values[key] = val
                elif not isinstance(val, bool): # Booleans are fine, other non-Decimals/NaNs are issues
                    nan_keys.append(key)


            if nan_keys: # If any required indicator is missing or NaN
                result["reason"] = f"No Signal: Required indicator(s) are NaN/Missing: {', '.join(nan_keys)}."
                logger.warning(result["reason"])
                return result

            # Unpack validated Decimal indicators
            k, fast_ema, slow_ema, trend_ema, atr, adx, pdi, mdi = (
                ind_values["stoch_k"], ind_values["fast_ema"], ind_values["slow_ema"],
                ind_values["trend_ema"], ind_values["atr"], ind_values["adx"],
                ind_values["pdi"], ind_values["mdi"]
            )
            # Stochastic K/D crosses (booleans from indicators dict)
            stoch_kd_bull_cross = bool(indicators.get("stoch_kd_bullish", False))
            stoch_kd_bear_cross = bool(indicators.get("stoch_kd_bearish", False))

            # --- Condition Checks ---
            # 1. EMA Cross
            ema_bullish_cross = fast_ema > slow_ema
            ema_bearish_cross = fast_ema < slow_ema
            ema_cross_state = "Bullish" if ema_bullish_cross else "Bearish" if ema_bearish_cross else "Neutral"

            # 2. Trend Filter (Price vs Trend EMA with buffer)
            trend_buffer_abs = trend_ema.copy_abs() * (self.config.trend_filter_buffer_percent / 100)
            # For long: price should be above trend EMA (minus buffer for some leniency if price dips slightly below)
            price_above_trend_ema_for_long = current_price > (trend_ema - trend_buffer_abs)
            # For short: price should be below trend EMA (plus buffer for leniency if price spikes slightly above)
            price_below_trend_ema_for_short = current_price < (trend_ema + trend_buffer_abs)
            trend_allows_long = price_above_trend_ema_for_long if self.config.trade_only_with_trend else True
            trend_allows_short = price_below_trend_ema_for_short if self.config.trade_only_with_trend else True
            trend_reason_suffix = f"(P:{current_price:.{DEFAULT_PRICE_DP}f} vs TrendEMA:{trend_ema:.{DEFAULT_PRICE_DP}f} ±{trend_buffer_abs:.{DEFAULT_PRICE_DP}f})" if self.config.trade_only_with_trend else "(TrendFilter OFF)"

            # 3. Stochastic Condition (Oversold/Overbought or K/D Cross)
            stoch_long_entry_cond = (k < self.config.stoch_oversold_threshold) or stoch_kd_bull_cross
            stoch_short_entry_cond = (k > self.config.stoch_overbought_threshold) or stoch_kd_bear_cross
            stoch_state_reason = f"K:{k:.1f} (OS:{self.config.stoch_oversold_threshold.normalize()}/OB:{self.config.stoch_overbought_threshold.normalize()}) KD_Cross(Bull:{stoch_kd_bull_cross}/Bear:{stoch_kd_bear_cross})"

            # 4. ATR Move Filter (Significant price movement compared to ATR)
            significant_price_move = True # Default to true if filter disabled or invalid ATR/prev_close
            atr_filter_reason_suffix = "(ATR MoveFilter OFF)"
            if self.config.atr_move_filter_multiplier > 0: # Only apply if multiplier is positive
                if atr.is_nan() or atr <= 0: # Should have been caught by ind_values check, but defensive
                    atr_filter_reason_suffix = f"(ATR Filter Skipped: Invalid ATR {atr.normalize() if not atr.is_nan() else 'NaN'})"
                    significant_price_move = False # Cannot evaluate, so treat as not significant
                elif prev_close.is_nan() or prev_close <= 0: # Check prev_close for ATR move calc
                    atr_filter_reason_suffix = f"(ATR Filter Skipped: Invalid Previous Close {prev_close.normalize() if not prev_close.is_nan() else 'NaN'})"
                    significant_price_move = False
                else:
                    atr_move_threshold_abs = atr * self.config.atr_move_filter_multiplier
                    price_move_abs = (current_price - prev_close).copy_abs()
                    significant_price_move = price_move_abs > atr_move_threshold_abs
                    atr_filter_reason_suffix = f"(Move:{price_move_abs:.{DEFAULT_PRICE_DP}f} {'OK' if significant_price_move else 'LOW'} vs Thr:{atr_move_threshold_abs:.{DEFAULT_PRICE_DP}f})"

            # 5. ADX Filter (Trend strength and direction)
            adx_is_trending_strong = adx > self.config.min_adx_level
            adx_long_direction_favored = pdi > mdi # +DI above -DI
            adx_short_direction_favored = mdi > pdi # -DI above +DI
            adx_allows_long = adx_is_trending_strong and adx_long_direction_favored
            adx_allows_short = adx_is_trending_strong and adx_short_direction_favored
            adx_filter_reason_suffix = f"(ADX:{adx:.1f} {'STRONG' if adx_is_trending_strong else 'WEAK'} vs Min:{self.config.min_adx_level.normalize()} | Dir: {'PDI>MDI' if adx_long_direction_favored else 'MDI>PDI' if adx_short_direction_favored else 'Neutral'})"


            # --- Combine Conditions for Final Signal ---
            base_long_signal_met = ema_bullish_cross and stoch_long_entry_cond
            base_short_signal_met = ema_bearish_cross and stoch_short_entry_cond

            final_long_signal = base_long_signal_met and trend_allows_long and significant_price_move and adx_allows_long
            final_short_signal = base_short_signal_met and trend_allows_short and significant_price_move and adx_allows_short

            # --- Construct Reason String ---
            if final_long_signal:
                result["long"] = True
                result["reason"] = f"Long Signal: EMA_X {ema_cross_state} & StochOK {stoch_state_reason} & TrendOK {trend_reason_suffix} & ATRMoveOK {atr_filter_reason_suffix} & ADX_OK {adx_filter_reason_suffix}"
            elif final_short_signal:
                result["short"] = True
                result["reason"] = f"Short Signal: EMA_X {ema_cross_state} & StochOK {stoch_state_reason} & TrendOK {trend_reason_suffix} & ATRMoveOK {atr_filter_reason_suffix} & ADX_OK {adx_filter_reason_suffix}"
            else: # No final signal, provide detailed reason why
                reason_parts = ["No Signal:"]
                # Log individual filter states for easier debugging
                reason_parts.append(f"Base(EMA_X:{ema_cross_state},Stoch:{stoch_state_reason}) -> LongBase:{base_long_signal_met}, ShortBase:{base_short_signal_met}.")
                if base_long_signal_met or base_short_signal_met: # If a base signal was met, show why filters blocked it
                    if not trend_allows_long and base_long_signal_met: reason_parts.append(f"Long Blocked: TrendFail {trend_reason_suffix}.")
                    if not trend_allows_short and base_short_signal_met: reason_parts.append(f"Short Blocked: TrendFail {trend_reason_suffix}.")
                    if not significant_price_move and (base_long_signal_met or base_short_signal_met) : reason_parts.append(f"Blocked: ATRMoveFail {atr_filter_reason_suffix}.")
                    if not adx_allows_long and base_long_signal_met: reason_parts.append(f"Long Blocked: ADXFail {adx_filter_reason_suffix}.")
                    if not adx_allows_short and base_short_signal_met: reason_parts.append(f"Short Blocked: ADXFail {adx_filter_reason_suffix}.")
                result["reason"] = " ".join(reason_parts)

            # Log signal check result: INFO for actual signals or blocks, DEBUG for "no signal met"
            log_level_for_signal = logging.INFO if result["long"] or result["short"] or "Blocked" in result["reason"] else logging.DEBUG
            logger.log(log_level_for_signal, f"Signal Check Result: {result['reason']}")

        except Exception as e: # Catch-all for unexpected errors during signal generation
            logger.error(f"{Fore.RED}Error generating entry signals: {e}{Style.RESET_ALL}", exc_info=True)
            result["reason"] = f"No Signal: Exception during generation ({type(e).__name__})"
            result["long"] = False; result["short"] = False # Ensure signals are false on error
        return result

    def check_exit_signals(
        self,
        position_side: str, # "long" or "short"
        indicators: Dict[str, Union[Decimal, bool, int]],
    ) -> Optional[str]:
        """
        Checks for signal-based exits:
        1. EMA Cross against the position.
        2. Stochastic Reversal Confirmation: %K crossing back from Overbought/Oversold, using previous K.
        Returns an exit reason string if conditions met, otherwise None.
        """
        if not indicators:
            logger.warning("Cannot check exit signals: indicators data missing.")
            return None

        # Extract necessary indicator values, ensuring they are Decimals and not NaN
        fast_ema_val = indicators.get("fast_ema")
        slow_ema_val = indicators.get("slow_ema")
        stoch_k_current_val = indicators.get("stoch_k")
        stoch_k_previous_val = indicators.get("stoch_k_prev") # K value from the previous candle

        # Validate all are Decimals and not NaN
        required_for_exit_check = {
            "fast_ema": fast_ema_val, "slow_ema": slow_ema_val,
            "stoch_k_current": stoch_k_current_val, "stoch_k_previous": stoch_k_previous_val
        }
        for name, val in required_for_exit_check.items():
            if not isinstance(val, Decimal) or val.is_nan():
                logger.warning(
                    f"Cannot check exit signals: Required indicator '{name}' is missing, not Decimal, or NaN (value: {val})."
                )
                return None
        # Cast to Decimal explicitly for type hinting after validation
        fast_ema: Decimal = fast_ema_val # type: ignore
        slow_ema: Decimal = slow_ema_val # type: ignore
        stoch_k_current: Decimal = stoch_k_current_val # type: ignore
        stoch_k_previous: Decimal = stoch_k_previous_val # type: ignore

        ema_is_bullish_crossed: bool = fast_ema > slow_ema
        ema_is_bearish_crossed: bool = fast_ema < slow_ema
        exit_reason: Optional[str] = None
        oversold_level = self.config.stoch_oversold_threshold
        overbought_level = self.config.stoch_overbought_threshold

        if position_side == "long":
            # Exit Long if EMA bearish cross
            if ema_is_bearish_crossed:
                exit_reason = f"Exit Signal (Long): EMA Bearish Cross (Fast {fast_ema.normalize()} < Slow {slow_ema.normalize()})"
            # Exit Long if Stochastic %K (current) crosses down from overbought, confirming with previous K above OB.
            elif stoch_k_previous >= overbought_level and stoch_k_current < overbought_level:
                exit_reason = (
                    f"Exit Signal (Long): Stoch Reversal from Overbought "
                    f"(PrevK {stoch_k_previous.normalize():.1f} >= OB {overbought_level.normalize()} -> CurrK {stoch_k_current.normalize():.1f} < OB)"
                )
            elif stoch_k_current >= overbought_level: # Still in overbought, log for info
                logger.debug(f"Exit Check (Long): Stoch K ({stoch_k_current.normalize():.1f}) is at/above Overbought ({overbought_level.normalize()}), awaiting bearish cross from above OB for potential exit signal.")

        elif position_side == "short":
            # Exit Short if EMA bullish cross
            if ema_is_bullish_crossed:
                exit_reason = f"Exit Signal (Short): EMA Bullish Cross (Fast {fast_ema.normalize()} > Slow {slow_ema.normalize()})"
            # Exit Short if Stochastic %K (current) crosses up from oversold, confirming with previous K below OS.
            elif stoch_k_previous <= oversold_level and stoch_k_current > oversold_level:
                exit_reason = (
                    f"Exit Signal (Short): Stoch Reversal from Oversold "
                    f"(PrevK {stoch_k_previous.normalize():.1f} <= OS {oversold_level.normalize()} -> CurrK {stoch_k_current.normalize():.1f} > OS)"
                )
            elif stoch_k_current <= oversold_level: # Still in oversold, log for info
                logger.debug(f"Exit Check (Short): Stoch K ({stoch_k_current.normalize():.1f}) is at/below Oversold ({oversold_level.normalize()}), awaiting bullish cross from below OS for potential exit signal.")

        if exit_reason:
            logger.trade(f"{Fore.YELLOW}{exit_reason}{Style.RESET_ALL}") # Use TRADE log level for actual exit signals
        return exit_reason


# --- Order Manager Class ---
class OrderManager:
    """
    Handles order placement (market), position protection (SL/TP/TSL using V5 API),
    and position closing. Manages a local tracker for protection status.
    """

    def __init__(
        self, config: TradingConfig, exchange_manager: ExchangeManager
    ):
        self.config = config
        self.exchange_manager = exchange_manager
        if not exchange_manager or not exchange_manager.exchange or not exchange_manager.market_info: # Critical dependency
            err_msg = "OrderManager cannot initialize: Valid ExchangeManager instance with initialized exchange and loaded market_info is required."
            logger.critical(f"{Style.BRIGHT}{Fore.RED}{err_msg}{Style.RESET_ALL}")
            raise ValueError(err_msg) # Raise error to halt bot creation
        self.exchange = exchange_manager.exchange # Convenience access
        self.market_info = exchange_manager.market_info # Convenience access
        # Tracks current protection state for each logical side ('long', 'short')
        # Possible states: None (no protection), 'ACTIVE_SLTP', 'ACTIVE_TSL'
        self.protection_tracker: Dict[str, Optional[str]] = {"long": None, "short": None}

    def _calculate_trade_parameters(
        self,
        side: str, # "buy" or "sell"
        atr: Decimal,
        total_equity: Decimal,
        current_price: Decimal,
    ) -> Optional[Dict[str, Optional[Decimal]]]:
        """Calculates SL price, TP price, order quantity, and TSL distance based on risk, ATR, and market info."""
        # Validate inputs
        if atr.is_nan() or atr <= 0:
            logger.error(f"Invalid ATR ({atr.normalize() if not atr.is_nan() else 'NaN'}) for trade parameter calculation.")
            return None
        if total_equity.is_nan() or total_equity <= 0:
            logger.error(f"Invalid total equity ({total_equity.normalize() if not total_equity.is_nan() else 'NaN'}) for parameter calculation.")
            return None
        if current_price.is_nan() or current_price <= 0:
            logger.error(f"Invalid current price ({current_price.normalize() if not current_price.is_nan() else 'NaN'}) for parameter calculation.")
            return None
        if not self.market_info or \
           self.market_info.get('tick_size', Decimal('NaN')).is_nan() or \
           self.market_info.get('contract_size', Decimal('NaN')).is_nan() or \
           self.market_info.get('min_order_size', Decimal('NaN')).is_nan():
             logger.error("Market info (tick_size, contract_size, min_order_size) missing, NaN, or incomplete for parameter calculation.")
             return None
        if side not in ["buy", "sell"]:
            logger.error(f"Invalid side '{side}' specified for trade parameter calculation.")
            return None

        try:
            # 1. Calculate Risk Amount in Settle Currency
            # total_equity is assumed to be in the settlement currency of the traded market.
            risk_amount_per_trade_settle_ccy = total_equity * self.config.risk_percentage

            # 2. Calculate Stop Loss Price
            sl_distance_atr_points = atr * self.config.sl_atr_multiplier # This is a price delta
            sl_price_calculated: Decimal
            if side == "buy": sl_price_calculated = current_price - sl_distance_atr_points
            else: sl_price_calculated = current_price + sl_distance_atr_points

            if sl_price_calculated <= 0: # SL price must be positive
                logger.error(f"Calculated SL price ({sl_price_calculated:.{DEFAULT_PRICE_DP}f}) is invalid (<=0). Cannot proceed.")
                return None

            # Ensure SL distance from current price is at least one tick size
            sl_distance_from_current_abs = (current_price - sl_price_calculated).copy_abs()
            min_tick_size = self.market_info['tick_size'] # Should be a valid Decimal from _load_market_info
            if min_tick_size.is_nan() or min_tick_size <= 0:
                logger.error("Market tick_size is invalid or zero. Cannot validate SL distance.")
                return None

            if sl_distance_from_current_abs < min_tick_size:
                logger.warning(f"Initial SL distance ({sl_distance_from_current_abs.normalize()}) < min tick size ({min_tick_size.normalize()}). Adjusting SL distance to min tick size.")
                sl_distance_from_current_abs = min_tick_size
                # Recalculate SL price with adjusted distance
                if side == "buy": sl_price_calculated = current_price - sl_distance_from_current_abs
                else: sl_price_calculated = current_price + sl_distance_from_current_abs
                if sl_price_calculated <= 0: # Re-check after adjustment
                     logger.error(f"Adjusted SL price ({sl_price_calculated:.{DEFAULT_PRICE_DP}f}) is still invalid (<=0).")
                     return None

            if sl_distance_from_current_abs <= 0: # Should not happen if current_price and sl_price_calculated are valid
                logger.error(f"Calculated SL distance ({sl_distance_from_current_abs.normalize()}) is invalid (<=0).")
                return None

            # 3. Calculate Quantity (in base asset, e.g., BTC for BTC/USDT or BTC/USD)
            # Bybit V5 API `qty` field is always in base currency for both linear and inverse.
            market_contract_size = self.market_info['contract_size'] # Value of 1 contract in quote currency (for linear) or base (for inverse, but usually 1 USD)
            quantity_calculated_base_asset: Decimal

            if self.config.market_type == "inverse":
                # Risk is in Settle (Base) currency. SL distance from current is in Quote.
                # Qty_base = (Risk_Settle_CCY * Price_Quote/Settle) / SL_Distance_Quote
                # If Settle=Base (e.g. BTC for BTC/USD), this is: (Risk_BTC * Price_USD/BTC) / SL_USD_per_BTC = Risk_USD / SL_USD_per_BTC = Qty_BTC
                if current_price <= 0: logger.error("Invalid current_price for inverse quantity calc."); return None
                risk_amount_in_quote_ccy = risk_amount_per_trade_settle_ccy * current_price
                quantity_calculated_base_asset = risk_amount_in_quote_ccy / sl_distance_from_current_abs
            else: # Linear/Swap (e.g., BTC/USDT:USDT)
                # Risk is in Settle (Quote) currency. SL distance from current is in Quote.
                # Qty_base = Risk_Quote_CCY / (SL_Distance_Quote_per_Base * ContractSize_Multiplier)
                # If contractSize is 1 (e.g. 1 unit of base per contract), then SL_Distance is effectively per unit of base.
                # PnL per point move for 1 unit of base = contract_size (if it represents this, usually 1 for USDT perps)
                # For linear, contract_size is often 1 (e.g. BTC for BTC/USDT).
                # Value per point for 1 unit of base asset. If market_contract_size is 1, this is 1.
                value_change_per_point_per_base_unit = market_contract_size # Typically 1 for linear futures where qty is base
                if value_change_per_point_per_base_unit <= 0: logger.error("Invalid contract size for linear quantity."); return None

                risk_per_unit_of_base = sl_distance_from_current_abs * value_change_per_point_per_base_unit
                if risk_per_unit_of_base <= 0:
                    logger.error(f"Calculated zero or negative risk per unit of base ({risk_per_unit_of_base.normalize()}). Cannot determine quantity.")
                    return None
                quantity_calculated_base_asset = risk_amount_per_trade_settle_ccy / risk_per_unit_of_base

            # Format quantity and check against minimum order size
            # Use ROUND_DOWN for quantity to be conservative.
            quantity_str_formatted = self.exchange_manager.format_amount(quantity_calculated_base_asset, rounding_mode=ROUND_DOWN)
            quantity_decimal_final = safe_decimal(quantity_str_formatted)

            if quantity_decimal_final.is_nan() or quantity_decimal_final <= 0:
                 logger.error(f"Calculated quantity ({quantity_str_formatted}) is invalid or zero after formatting. Original calc: {quantity_calculated_base_asset.normalize()}")
                 return None

            min_order_size_market = self.market_info.get('min_order_size', Decimal('NaN')) # Should be valid Decimal
            if min_order_size_market.is_nan(): logger.error("Min order size is NaN."); return None
            if quantity_decimal_final < min_order_size_market:
                logger.error(f"Calculated quantity {quantity_decimal_final.normalize()} is less than market minimum order size {min_order_size_market.normalize()}.")
                return None

            # 4. Calculate Take Profit Price (Optional)
            tp_price_calculated: Optional[Decimal] = None
            if self.config.tp_atr_multiplier > 0: # Only if TP is enabled
                tp_distance_atr_points = atr * self.config.tp_atr_multiplier
                if side == "buy": tp_price_calculated = current_price + tp_distance_atr_points
                else: tp_price_calculated = current_price - tp_distance_atr_points
                if tp_price_calculated <= 0: # TP must be positive
                    logger.warning(f"Calculated TP price ({tp_price_calculated:.{DEFAULT_PRICE_DP}f}) is invalid (<=0). Disabling TP for this trade.")
                    tp_price_calculated = None # Effectively disables TP

            # 5. Calculate Trailing Stop Loss (TSL) Distance (for activation later)
            # TSL distance is a price delta, use price precision from market_info.
            tsl_distance_price_points = current_price * (self.config.trailing_stop_percent / 100)
            if tsl_distance_price_points < min_tick_size: # TSL distance should also be >= tick size
                 logger.debug(f"TSL distance ({tsl_distance_price_points.normalize()}) < min tick ({min_tick_size.normalize()}). Adjusting TSL distance to min tick size.")
                 tsl_distance_price_points = min_tick_size
            # Formatting TSL distance uses price formatting rules
            tsl_distance_str_formatted = self.exchange_manager.format_price(tsl_distance_price_points)
            tsl_distance_decimal_final = safe_decimal(tsl_distance_str_formatted)
            if tsl_distance_decimal_final.is_nan() or tsl_distance_decimal_final <= 0:
                 logger.warning(f"Calculated invalid TSL distance ('{tsl_distance_str_formatted}'). TSL might fail. Original calc: {tsl_distance_price_points.normalize()}")
                 tsl_distance_decimal_final = Decimal('NaN') # Mark as invalid for later checks

            # Format SL and TP prices using market price formatting
            sl_price_str_formatted = self.exchange_manager.format_price(sl_price_calculated)
            sl_price_decimal_final = safe_decimal(sl_price_str_formatted)
            if sl_price_decimal_final.is_nan() or sl_price_decimal_final <= 0:
                logger.error(f"Formatted SL price ('{sl_price_str_formatted}') is invalid. Aborting parameter calculation.")
                return None

            tp_price_decimal_final: Optional[Decimal] = None
            if tp_price_calculated is not None:
                 tp_price_str_formatted = self.exchange_manager.format_price(tp_price_calculated)
                 tp_price_decimal_final = safe_decimal(tp_price_str_formatted)
                 if tp_price_decimal_final.is_nan() or tp_price_decimal_final <= 0:
                      logger.warning(f"Failed to format a valid TP price ('{tp_price_str_formatted}'). Disabling TP for this trade.")
                      tp_price_decimal_final = None # Disable TP if formatting failed

            # Prepare output dictionary
            params_out: Dict[str, Optional[Decimal]] = {
                "qty": quantity_decimal_final,
                "sl_price": sl_price_decimal_final,
                "tp_price": tp_price_decimal_final,
                "tsl_distance": tsl_distance_decimal_final if not tsl_distance_decimal_final.is_nan() else None, # Store None if NaN
            }

            log_tp_str = f"{params_out['tp_price'].normalize()}" if params_out['tp_price'] else "Disabled"
            log_tsl_str = f"{params_out['tsl_distance'].normalize()}" if params_out['tsl_distance'] else "Invalid/Not Set"
            settle_ccy_display = self.market_info.get('settle', self.config.symbol.split(':')[-1] if ':' in self.config.symbol else 'SETTLE')
            logger.info(
                f"Trade Parameters Calculated for {side.upper()} entry: "
                f"Qty={params_out['qty'].normalize()} {self.market_info.get('base', 'BASE')}, "
                f"EntryPrice (approx.)={current_price.normalize():.{DEFAULT_PRICE_DP}f}, "
                f"SLPrice={params_out['sl_price'].normalize()}, "
                f"TPPrice={log_tp_str}, "
                f"TSLDistance (for future TSL activation)={log_tsl_str}, "
                f"RiskAmountSettle={risk_amount_per_trade_settle_ccy.normalize():.{DEFAULT_PRICE_DP}f} {settle_ccy_display}, ATR={atr.normalize():.{DEFAULT_PRICE_DP+1}f}"
            )
            return params_out

        except (InvalidOperation, DivisionByZero, TypeError, Exception) as e:
            logger.error(f"Error calculating trade parameters for {side.upper()} side: {e}", exc_info=True)
            return None

    def _execute_market_order(
        self, side: str, qty_decimal: Decimal
    ) -> Optional[Dict]:
        """Executes a market order with retries and basic confirmation logging."""
        if not self.exchange or not self.market_info:
            logger.error("Cannot execute market order: Exchange or Market info missing.")
            return None

        symbol_to_trade = self.config.symbol
        # Format quantity according to market rules (e.g., amount precision)
        # Use ROUND_DOWN for order quantity to be conservative.
        # CCXT create_market_order expects amount as float.
        # First, format to string with correct precision and rounding.
        qty_str_for_api = self.exchange_manager.format_amount(qty_decimal, rounding_mode=ROUND_DOWN)
        final_qty_decimal_for_log = safe_decimal(qty_str_for_api) # For logging and comparison

        if final_qty_decimal_for_log.is_nan() or final_qty_decimal_for_log <= 0:
            logger.error(f"Attempted market order with zero/invalid formatted quantity: '{qty_str_for_api}' (Original Decimal: {qty_decimal.normalize()}). Order aborted.")
            return None

        # Convert the precisely formatted string quantity to float for CCXT API
        try:
            amount_float_for_ccxt = float(qty_str_for_api)
        except ValueError:
            logger.error(f"Could not convert formatted quantity string '{qty_str_for_api}' to float for API. Order aborted.")
            return None


        logger.trade(
            f"{Fore.CYAN}Attempting MARKET {side.upper()} order: {final_qty_decimal_for_log.normalize()} {self.market_info.get('base', '')} for {symbol_to_trade}...{Style.RESET_ALL}"
        )
        try:
            # Bybit V5 specific parameters for creating orders
            params_v5 = {
                "category": self.config.bybit_v5_category,
                "positionIdx": self.config.position_idx, # 0 for one-way, 1 for hedge buy, 2 for hedge sell
                "timeInForce": "ImmediateOrCancel", # Market orders are effectively IOC. FOK also possible.
                # "reduceOnly": False, # This would be True for closing orders if not opening new.
                                      # For Bybit V5, `closeOnTrigger` can be used with SL/TP orders.
                                      # Market orders to close positions just need opposite side and correct qty.
            }

            order_response = fetch_with_retries(
                self.exchange.create_market_order,
                symbol=symbol_to_trade,
                side=side, # "buy" or "sell"
                amount=amount_float_for_ccxt, # Pass float here
                params=params_v5,
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
            )

            if order_response is None: # Should be caught by fetch_with_retries if it raises, but defensive
                logger.error(f"{Fore.RED}Market order submission failed after retries (returned None unexpectedly).{Style.RESET_ALL}")
                return None

            # Parse common fields from CCXT unified order response
            order_id = order_response.get("id", "[N/A]")
            order_status = order_response.get("status", "[unknown]") # e.g., 'open', 'closed', 'canceled', 'rejected'
            filled_qty_str = order_response.get("filled", "0") # Amount filled
            avg_fill_price_str = order_response.get("average", "0") # Average fill price

            filled_qty_decimal = safe_decimal(filled_qty_str)
            avg_fill_price_decimal = safe_decimal(avg_fill_price_str)
            avg_price_log_str = avg_fill_price_decimal.normalize() if not avg_fill_price_decimal.is_nan() and avg_fill_price_decimal > 0 else "[N/A]"

            logger.trade(
                f"{Style.BRIGHT}{Fore.GREEN}Market order submitted: ID {order_id}, Side {side.upper()}, "
                f"Ordered Qty {final_qty_decimal_for_log.normalize()}, Status: {order_status}, "
                f"Filled Qty: {filled_qty_decimal.normalize()}, AvgFillPx: {avg_price_log_str}{Style.RESET_ALL}"
            )
            termux_notify(
                f"{symbol_to_trade} Order Submitted",
                f"Market {side.upper()} {final_qty_decimal_for_log.normalize()} ID:{order_id}, Status:{order_status}"
            )

            # Handle immediate rejection or failure status from the order submission response
            # For IOC, 'canceled' with 0 filled means it didn't fill. 'closed' means it filled.
            if order_status == "rejected":
                 rejection_reason = order_response.get("info", {}).get("rejectReason", "No reason provided by exchange")
                 logger.error(f"{Fore.RED}Market order {order_id} was REJECTED. Reason: '{rejection_reason}'. Full info: {order_response.get('info')}{Style.RESET_ALL}")
                 return None # Treat as failed order
            elif order_status == "canceled" and filled_qty_decimal == 0 and params_v5.get("timeInForce") == "ImmediateOrCancel":
                 logger.error(f"{Fore.RED}Market order {order_id} (IOC) was CANCELED with 0 filled. Order did not execute.{Style.RESET_ALL}")
                 return None # Treat as failed order
            elif order_status == "expired": # Should not happen for market IOC, but good to check
                 logger.error(f"{Fore.RED}Market order {order_id} EXPIRED. This is unexpected for market orders.{Style.RESET_ALL}")
                 return None

            # Short delay to allow order processing and position state update on the exchange
            logger.debug(f"Short delay ({self.config.order_check_delay_seconds}s) after market order {order_id} submission for propagation...")
            time.sleep(self.config.order_check_delay_seconds)

            return order_response # Return the full CCXT order response

        except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e: # Fail-fast exceptions for orders
            logger.error(f"{Fore.RED}Order placement failed ({type(e).__name__}): {e}{Style.RESET_ALL}")
            termux_notify(f"{symbol_to_trade} Order FAILED", f"Market {side.upper()} failed: {str(e)[:50]}")
            return None
        except Exception as e: # Catch-all for other unexpected errors
            logger.error(f"{Fore.RED}Unexpected error placing market order: {e}{Style.RESET_ALL}", exc_info=True)
            termux_notify(f"{symbol_to_trade} Order ERROR", f"Market {side.upper()} unexpected error.")
            return None

    def _set_position_protection(
        self,
        position_side: str, # "long" or "short" (logical side of the position being protected)
        sl_price: Optional[Decimal] = None,
        tp_price: Optional[Decimal] = None,
        is_tsl: bool = False, # True if setting/activating TSL
        tsl_distance: Optional[Decimal] = None, # TSL distance from market price (price points)
        tsl_activation_price: Optional[Decimal] = None, # Price at which TSL should activate
    ) -> bool:
        """
        Sets SL, TP, or TSL for a position using Bybit V5's setTradingStop endpoint.
        This endpoint manages SL, TP, and TSL for an entire position based on `positionIdx`.
        To clear stops, send "0" for the respective price/distance fields or omit them if API allows.
        """
        if not self.exchange: logger.error("Cannot set position protection: Exchange not initialized."); return False
        if not self.market_info: logger.error("Cannot set position protection: Market info missing."); return False
        market_id = self.market_info.get("id")
        if not market_id: logger.error("Cannot set position protection: Market ID missing."); return False

        tracker_key = position_side.lower() # 'long' or 'short'
        if tracker_key not in self.protection_tracker:
             logger.error(f"Invalid position_side '{position_side}' for protection tracker update."); return False

        # Format parameters for API. "0" string to clear/disable.
        # _format_v5_param(value, type, allow_zero)
        sl_price_api_str = self.exchange_manager._format_v5_param(sl_price, "price", allow_zero=True) if sl_price else "0"
        tp_price_api_str = self.exchange_manager._format_v5_param(tp_price, "price", allow_zero=True) if tp_price else "0"
        tsl_distance_api_str = self.exchange_manager._format_v5_param(tsl_distance, "distance", allow_zero=False) if tsl_distance else "0"
        tsl_activation_price_api_str = self.exchange_manager._format_v5_param(tsl_activation_price, "price", allow_zero=False) if tsl_activation_price else "0"

        # Base parameters for Bybit V5 POST /v5/position/trading-stop
        api_params: Dict[str, Any] = {
            "category": self.config.bybit_v5_category,
            "symbol": market_id,
            "positionIdx": self.config.position_idx,
            "tpslMode": V5_TPSL_MODE_FULL, # Apply to the full position
            # Bybit API: stopLoss, takeProfit, trailingStop (distance), activePrice (TSL activation)
            # slTriggerBy, tpTriggerBy, triggerBy (for TSL trail type)
        }

        action_description = ""
        new_tracker_state: Optional[str] = None # New state for self.protection_tracker

        if is_tsl: # Setting or modifying TSL
            if tsl_distance_api_str and tsl_distance_api_str != "0" and \
               tsl_activation_price_api_str and tsl_activation_price_api_str != "0":
                api_params["trailingStop"] = tsl_distance_api_str
                api_params["activePrice"] = tsl_activation_price_api_str
                api_params["triggerBy"] = self.config.tsl_trigger_by # Price type for TSL to trail
                # When TSL is set, Bybit V5 API might require SL/TP to be "0" or it manages them implicitly.
                # Explicitly setting SL/TP to "0" is safer if TSL is the primary mechanism.
                api_params["stopLoss"] = "0"
                api_params["takeProfit"] = "0"
                action_description = f"ACTIVATE/MODIFY TSL (Dist: {tsl_distance_api_str}, ActPx: {tsl_activation_price_api_str})"
                new_tracker_state = "ACTIVE_TSL"
            else:
                logger.error(f"Cannot activate TSL for {position_side.upper()}: Invalid TSL distance ('{tsl_distance_api_str}') or activation price ('{tsl_activation_price_api_str}'). Must be positive values.")
                return False
        elif sl_price_api_str != "0" or tp_price_api_str != "0": # Setting fixed SL and/or TP
            if sl_price_api_str != "0": api_params["stopLoss"] = sl_price_api_str
            if tp_price_api_str != "0": api_params["takeProfit"] = tp_price_api_str
            api_params["slTriggerBy"] = self.config.sl_trigger_by
            api_params["tpTriggerBy"] = self.config.sl_trigger_by # Bybit V5 often uses same trigger for TP as SL
            # Ensure TSL fields are "0" when setting fixed SL/TP
            api_params["trailingStop"] = "0"
            api_params["activePrice"] = "0"
            action_description = f"SET SL={api_params.get('stopLoss','0')} TP={api_params.get('takeProfit','0')}"
            new_tracker_state = "ACTIVE_SLTP"
        else: # Clearing all stops (all price/distance inputs were None or resulted in "0")
            api_params["stopLoss"] = "0"
            api_params["takeProfit"] = "0"
            api_params["trailingStop"] = "0"
            api_params["activePrice"] = "0"
            action_description = "CLEAR ALL SL/TP/TSL"
            new_tracker_state = None # No active protection

        logger.trade(f"{Fore.CYAN}Attempting to {action_description} for {position_side.upper()} {self.config.symbol}...{Style.RESET_ALL}")
        logger.debug(f"Calling V5 setTradingStop with parameters: {api_params}")

        # CCXT method for Bybit V5 POST /v5/position/trading-stop
        private_method_name = "privatePostPositionTradingStop"
        if not hasattr(self.exchange, private_method_name):
            logger.critical(f"{Style.BRIGHT}{Fore.RED}Fatal Error: CCXT private method '{private_method_name}' not found. Cannot manage position protection.{Style.RESET_ALL}")
            return False # Critical failure

        method_to_call = getattr(self.exchange, private_method_name)
        try:
            response = fetch_with_retries(
                method_to_call,
                params=api_params,
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
            )

            if response and response.get("retCode") == V5_SUCCESS_RETCODE:
                logger.trade(f"{Style.BRIGHT}{Fore.GREEN}{action_description} successful for {position_side.upper()} {self.config.symbol}.{Style.RESET_ALL}")
                termux_notify(f"{self.config.symbol} Protection {('Set' if new_tracker_state else 'Cleared')}", f"{action_description} for {position_side.upper()}")
                self.protection_tracker[tracker_key] = new_tracker_state # Update local tracker
                return True
            else:
                ret_code = response.get("retCode", "[N/A]") if response else "[No Response]"
                ret_msg = response.get("retMsg", "[No error message]") if response else "[No Response]"
                logger.error(f"{Fore.RED}{action_description} failed for {position_side.upper()} {self.config.symbol}. API Response: Code={ret_code}, Msg='{ret_msg}'.{Style.RESET_ALL}")
                logger.debug(f"Full response from failed {private_method_name}: {response}")
                termux_notify(f"{self.config.symbol} Protection FAILED", f"{action_description[:30]}... failed: {ret_msg[:50]}")
                return False
        except Exception as e:
            logger.error(f"{Fore.RED}Unexpected error during '{action_description}' for {position_side.upper()} {self.config.symbol}: {e}{Style.RESET_ALL}", exc_info=True)
            termux_notify(f"{self.config.symbol} Protection ERROR", f"{action_description[:30]}... error.")
            return False

    def _verify_position_state(
            self,
            expected_side_logical: Optional[str], # "long", "short", or None (if expecting flat)
            expected_qty_min_abs: Decimal = POSITION_QTY_EPSILON, # Min expected absolute qty if position should be open
            max_attempts: int = 4, # Number of times to check
            delay_seconds: float = 1.5, # Delay between checks
            action_context: str = "Position Verification" # For logging
        ) -> Tuple[bool, Optional[Dict[str, Dict[str, Any]]]]:
        """
        Fetches current position state repeatedly to verify if it matches the expected state.
        Returns (verification_success: bool, final_position_state_summary: Optional[Dict]).
        The final_position_state_summary is the last successfully fetched state, or state from last attempt.
        """
        logger.debug(f"{action_context}: Verifying position state. Expecting side: '{expected_side_logical}', MinAbsQty (if open): {expected_qty_min_abs.normalize()}. Max attempts: {max_attempts}.")
        last_known_position_summary: Optional[Dict[str, Dict[str, Any]]] = None

        for attempt in range(max_attempts):
            logger.debug(f"{action_context}: Verification attempt {attempt + 1}/{max_attempts}...")
            current_positions_summary_fetched = self.exchange_manager.get_current_position()
            last_known_position_summary = current_positions_summary_fetched # Update with latest attempt

            if current_positions_summary_fetched is None: # Failed to fetch position state
                logger.warning(f"{action_context} Warning: Failed to fetch position state on attempt {attempt + 1}.")
                if attempt < max_attempts - 1:
                    time.sleep(delay_seconds); continue
                else:
                    logger.error(f"{Fore.RED}{action_context} FAILED: Could not fetch position state after {max_attempts} attempts.{Style.RESET_ALL}")
                    return False, last_known_position_summary

            # Determine current actual state from the fetched summary
            actual_is_flat = True
            actual_open_side_logical: Optional[str] = None
            actual_open_qty_abs = Decimal("0")

            long_pos_data = current_positions_summary_fetched.get("long", {})
            short_pos_data = current_positions_summary_fetched.get("short", {})

            if long_pos_data and safe_decimal(long_pos_data.get("qty", "0")).copy_abs() >= POSITION_QTY_EPSILON:
                actual_is_flat = False; actual_open_side_logical = "long"
                actual_open_qty_abs = safe_decimal(long_pos_data.get("qty", "0")).copy_abs()
            elif short_pos_data and safe_decimal(short_pos_data.get("qty", "0")).copy_abs() >= POSITION_QTY_EPSILON:
                actual_is_flat = False; actual_open_side_logical = "short"
                actual_open_qty_abs = safe_decimal(short_pos_data.get("qty", "0")).copy_abs()

            # Compare actual state with expected state
            verification_succeeded = False
            log_message_suffix = ""

            if expected_side_logical is None: # Expecting flat
                verification_succeeded = actual_is_flat
                log_message_suffix = f"Expected FLAT, Actual: {'FLAT' if actual_is_flat else f'{actual_open_side_logical.upper()} Qty={actual_open_qty_abs.normalize()}'}"
            elif actual_open_side_logical == expected_side_logical: # Expected side matches actual open side
                quantity_matches_expectation = actual_open_qty_abs >= expected_qty_min_abs
                verification_succeeded = quantity_matches_expectation
                log_message_suffix = (f"Expected {expected_side_logical.upper()} (MinAbsQty~{expected_qty_min_abs.normalize()}), "
                                      f"Actual: {actual_open_side_logical.upper()} Qty={actual_open_qty_abs.normalize()} "
                                      f"({'QTY OK' if quantity_matches_expectation else 'QTY MISMATCH'})")
            else: # Side mismatch
                 verification_succeeded = False
                 log_message_suffix = (f"Expected {expected_side_logical.upper() if expected_side_logical else 'FLAT'}, "
                                       f"Actual: {'FLAT' if actual_is_flat else (actual_open_side_logical.upper() + ' Qty=' + actual_open_qty_abs.normalize()) if actual_open_side_logical else 'UNKNOWN/ERROR'} "
                                       f"(SIDE MISMATCH)")

            logger.debug(f"{action_context} Check {attempt + 1}: {log_message_suffix}")

            if verification_succeeded:
                logger.info(f"{Style.BRIGHT}{Fore.GREEN}{action_context} SUCCEEDED on attempt {attempt + 1}. State confirmed: {log_message_suffix}{Style.RESET_ALL}")
                return True, current_positions_summary_fetched

            if attempt < max_attempts - 1:
                logger.debug(f"State not as expected. Waiting {delay_seconds}s for next attempt...")
                time.sleep(delay_seconds)
            else: # Max attempts reached, verification failed
                 logger.error(f"{Fore.RED}{action_context} FAILED after {max_attempts} attempts. Final state check: {log_message_suffix}{Style.RESET_ALL}")
                 return False, current_positions_summary_fetched

        return False, last_known_position_summary # Fallback, should not be reached

    def place_risked_market_order(
        self,
        side: str, # "buy" or "sell" (order side)
        atr: Decimal,
        total_equity: Decimal,
        current_price: Decimal,
    ) -> bool:
        """Orchestrates a risked market order entry sequence: calculate, execute, verify, protect."""
        if not self.exchange or not self.market_info: logger.critical("OrderManager not fully initialized for placing order."); return False
        if side not in ["buy", "sell"]: logger.error(f"Invalid side '{side}' for place_risked_market_order."); return False
        if atr.is_nan() or atr <= 0: logger.error("Entry Aborted: Invalid ATR value for risk calculation."); return False
        if total_equity is None or total_equity.is_nan() or total_equity <= 0: logger.error("Entry Aborted: Invalid Equity value for risk calculation."); return False
        if current_price.is_nan() or current_price <= 0: logger.error("Entry Aborted: Invalid Current Price for calculations."); return False

        logical_position_side = "long" if side == "buy" else "short"
        logger.info(f"{Style.BRIGHT}{Fore.MAGENTA}--- Initiating Entry Sequence for {logical_position_side.upper()} Position ---{Style.RESET_ALL}")

        # 1. Calculate Trade Parameters (Qty, SL, TP)
        trade_params = self._calculate_trade_parameters(side, atr, total_equity, current_price)
        if not trade_params or not trade_params.get("qty") or trade_params["qty"] <= 0:
            logger.error("Entry Aborted: Failed to calculate valid trade parameters (qty, SL, etc.).")
            return False
        qty_to_order = trade_params["qty"] # This is a positive Decimal
        initial_sl_price = trade_params.get("sl_price")
        initial_tp_price = trade_params.get("tp_price") # Can be None
        # TSL distance is also in trade_params, will be used by manage_trailing_stop later

        if initial_sl_price is None or initial_sl_price.is_nan() or initial_sl_price <= 0:
             logger.error(f"Entry Aborted: Invalid Stop Loss price ({initial_sl_price}) calculated. Cannot place order without SL.")
             return False

        # 2. Execute Market Order
        market_order_info = self._execute_market_order(side, qty_to_order)
        if not market_order_info:
            logger.error(f"Entry Aborted: Market order execution failed for {side.upper()} {qty_to_order.normalize()}.")
            self._handle_entry_failure(side, qty_to_order)
            return False
        entry_order_id = market_order_info.get("id", "[N/A_ORDER_ID]")
        # Average fill price from market order response (might be delayed or not precise initially)
        avg_entry_price_from_order_resp = safe_decimal(market_order_info.get("average", "NaN"))


        # 3. Verify Position State Post-Order
        # Expect at least e.g. 90% fill for verification, adjust as needed.
        min_expected_filled_qty_abs = qty_to_order * Decimal("0.90")
        verification_ok, final_verified_pos_state = self._verify_position_state(
            expected_side_logical=logical_position_side,
            expected_qty_min_abs=min_expected_filled_qty_abs,
            max_attempts=6, # Allow more attempts for position state to update
            delay_seconds=max(self.config.order_check_delay_seconds, 1.0),
            action_context=f"Post-{logical_position_side.upper()}-Entry Verification"
        )

        if not verification_ok:
            logger.error(f"{Fore.RED}Entry FAILED: Position verification failed after market order {entry_order_id}. Manual check required! Attempting cleanup...{Style.RESET_ALL}")
            self._handle_entry_failure(side, qty_to_order)
            return False

        # Extract details from the verified position state for logging and protection
        active_pos_details = final_verified_pos_state.get(logical_position_side) if final_verified_pos_state else {}
        if not active_pos_details: # Should not happen if verification_ok is True
            logger.error(f"{Fore.RED}Internal Error: Position {logical_position_side} verified OK, but details missing. Aborting entry sequence.{Style.RESET_ALL}")
            self._handle_entry_failure(side, qty_to_order)
            return False

        actual_filled_qty_abs = safe_decimal(active_pos_details.get("qty", "0")).copy_abs()
        actual_avg_entry_price = safe_decimal(active_pos_details.get("entry_price", "NaN"))
        # If avg_entry_price from position is still NaN, use the one from order response as fallback for logging
        if actual_avg_entry_price.is_nan() and not avg_entry_price_from_order_resp.is_nan():
            actual_avg_entry_price = avg_entry_price_from_order_resp
            logger.debug(f"Used avg entry price from order response ({avg_entry_price_from_order_resp.normalize()}) as position data was NaN.")


        logger.info(
            f"{Style.BRIGHT}{Fore.GREEN}Position {logical_position_side.upper()} confirmed: "
            f"Actual Qty={actual_filled_qty_abs.normalize()}, AvgEntryPx={actual_avg_entry_price.normalize() if not actual_avg_entry_price.is_nan() else '[N/A]'}{Style.RESET_ALL}"
        )
        if actual_filled_qty_abs < qty_to_order * Decimal("0.99"): # Warn if fill is slightly off
             logger.warning(f"Filled quantity {actual_filled_qty_abs.normalize()} is notably less than ordered {qty_to_order.normalize()}. This might be due to slippage or partial fill.")

        # 4. Set Initial Stop Loss and Take Profit
        set_stops_successful = self._set_position_protection(
            logical_position_side,
            sl_price=initial_sl_price,
            tp_price=initial_tp_price # Pass None if TP is disabled
        )

        if not set_stops_successful:
            logger.error(f"{Fore.RED}Entry Alert: Failed to set initial SL/TP for {logical_position_side.upper()} position. Attempting emergency close!{Style.RESET_ALL}")
            self.close_position(logical_position_side, actual_filled_qty_abs, reason="EmergencyClose:FailedInitialStopSet")
            return False

        # 5. Log Trade Entry to Journal
        if self.config.enable_journaling:
            if actual_avg_entry_price.is_nan():
                logger.warning("Logging trade entry to journal with N/A average entry price due to fetch/parse issue.")
            self.log_trade_entry_to_journal(side, actual_filled_qty_abs, actual_avg_entry_price, entry_order_id)

        logger.info(f"{Style.BRIGHT}{Fore.GREEN}--- Entry Sequence for {logical_position_side.upper()} Completed Successfully ---{Style.RESET_ALL}")
        return True

    def manage_trailing_stop(
        self,
        position_side: str, # "long" or "short"
        entry_price: Decimal,
        current_market_price: Decimal,
        current_atr: Decimal,
    ) -> None:
        """Checks TSL activation conditions and attempts to activate TSL if position is protected by fixed SL/TP."""
        if not self.exchange or not self.market_info: logger.error("Cannot manage TSL: Exchange/Market info missing."); return
        tracker_key = position_side.lower()

        current_protection_state_local = self.protection_tracker.get(tracker_key)
        if current_protection_state_local != "ACTIVE_SLTP":
            log_msg_tsl_check = (f"TSL already active or protection not SL/TP (Tracker: {current_protection_state_local})."
                                 if current_protection_state_local == "ACTIVE_TSL"
                                 else f"No active SL/TP protection tracked locally (Tracker: {current_protection_state_local}). Cannot evaluate TSL activation yet.")
            logger.debug(f"TSL Management Check ({position_side.upper()}): {log_msg_tsl_check}")
            return

        if current_atr.is_nan() or current_atr <= 0: logger.debug(f"TSL Check ({position_side.upper()}): Invalid ATR. Skipping."); return
        if entry_price.is_nan() or entry_price <= 0: logger.debug(f"TSL Check ({position_side.upper()}): Invalid entry price. Skipping."); return
        if current_market_price.is_nan() or current_market_price <= 0: logger.debug(f"TSL Check ({position_side.upper()}): Invalid current market price. Skipping."); return

        try:
            # 1. Calculate TSL Activation Target Price (price level where TSL should kick in)
            activation_distance_points = current_atr * self.config.tsl_activation_atr_multiplier
            tsl_activation_target_price: Decimal
            if position_side == "long":
                tsl_activation_target_price = entry_price + activation_distance_points
            else: # short
                tsl_activation_target_price = entry_price - activation_distance_points

            if tsl_activation_target_price.is_nan() or tsl_activation_target_price <= 0:
                logger.warning(f"Invalid TSL activation price ({tsl_activation_target_price.normalize()}). Skipping TSL for {position_side.upper()}.")
                return

            # 2. Calculate TSL Distance (actual trail distance from current price once active)
            tsl_actual_distance_points = current_market_price * (self.config.trailing_stop_percent / 100)
            min_tick_size = self.market_info.get('tick_size', Decimal('1e-8')) # Default if not found
            if min_tick_size.is_nan() or min_tick_size <= 0:
                logger.warning(f"TSL Check ({position_side.upper()}): Invalid market tick_size. Skipping TSL distance adjustment.")
            elif tsl_actual_distance_points < min_tick_size:
                logger.debug(f"TSL distance ({tsl_actual_distance_points.normalize()}) < min tick ({min_tick_size.normalize()}). Adjusting to min tick.")
                tsl_actual_distance_points = min_tick_size

            if tsl_actual_distance_points <= 0:
                 logger.warning(f"Invalid TSL distance ({tsl_actual_distance_points.normalize()}). Skipping TSL for {position_side.upper()}.")
                 return

            # 3. Check Activation Condition (current price must have moved beyond activation target)
            should_activate_tsl = False
            if position_side == "long" and current_market_price >= tsl_activation_target_price:
                should_activate_tsl = True
            elif position_side == "short" and current_market_price <= tsl_activation_target_price:
                should_activate_tsl = True

            if should_activate_tsl:
                logger.trade(
                    f"{Fore.MAGENTA}Trailing Stop Loss (TSL) activation condition MET for {position_side.upper()}!{Style.RESET_ALL}"
                )
                logger.trade(
                    f"  Details: EntryPx={entry_price.normalize():.{DEFAULT_PRICE_DP}f}, CurrentPx={current_market_price.normalize():.{DEFAULT_PRICE_DP}f}, "
                    f"TSLActivationTargetPx~={tsl_activation_target_price.normalize():.{DEFAULT_PRICE_DP}f}, "
                    f"TSLDistanceToSet~={tsl_actual_distance_points.normalize():.{DEFAULT_PRICE_DP}f}"
                )
                # Activate TSL. Bybit's `activePrice` is the price that, if breached, activates the trail.
                # `trailingStop` is the distance.
                # The `tsl_activation_target_price` calculated is the price that *triggers* this logic.
                # For Bybit API, `activePrice` is the price the market needs to reach for TSL to become active.
                # So, `tsl_activation_target_price` seems correct for `activePrice` API param.
                activation_successful = self._set_position_protection(
                    position_side,
                    is_tsl=True,
                    tsl_distance=tsl_actual_distance_points,
                    tsl_activation_price=tsl_activation_target_price,
                )
                if activation_successful: # protection_tracker updated inside _set_position_protection
                    logger.trade(f"{Style.BRIGHT}{Fore.GREEN}TSL activated successfully for {position_side.upper()} position.{Style.RESET_ALL}")
                else:
                    logger.error(f"{Fore.RED}Failed to activate TSL for {position_side.upper()} position via API.{Style.RESET_ALL}")
            else:
                logger.debug(
                    f"TSL Check ({position_side.upper()}): Activation NOT MET. "
                    f"(CurrentPx: {current_market_price.normalize():.{DEFAULT_PRICE_DP}f}, TargetActivationPx: ~{tsl_activation_target_price.normalize():.{DEFAULT_PRICE_DP}f})"
                )
        except Exception as e:
            logger.error(f"Error managing TSL for {position_side.upper()} position: {e}", exc_info=True)

    def close_position(
        self, position_side_to_close: str, # "long" or "short"
        qty_abs_to_close: Decimal,
        reason: str = "Strategy Exit Signal"
    ) -> bool:
        """Orchestrates position closure: clear stops, execute market order, verify closure."""
        if not self.exchange or not self.market_info: logger.critical("OrderManager not fully initialized for closing position."); return False
        if position_side_to_close not in ["long", "short"]: logger.error(f"Invalid side '{position_side_to_close}' for close_position."); return False

        if qty_abs_to_close.is_nan() or qty_abs_to_close.copy_abs() < POSITION_QTY_EPSILON:
            logger.warning(f"Close requested for zero/negligible quantity ({qty_abs_to_close.normalize()}). Skipping close for {position_side_to_close.upper()}.")
            self.protection_tracker[position_side_to_close.lower()] = None # Ensure tracker is cleared
            return True # Considered successful as no action needed

        symbol_to_trade = self.config.symbol
        closing_order_side = "sell" if position_side_to_close == "long" else "buy" # Opposite side to close
        tracker_key = position_side_to_close.lower()

        logger.trade(
            f"{Fore.YELLOW}Attempting to CLOSE {position_side_to_close.upper()} position (Qty: {qty_abs_to_close.normalize()} {self.market_info.get('base', '')}) "
            f"for {symbol_to_trade} | Reason: {reason}...{Style.RESET_ALL}"
        )

        # 1. Clear any existing SL/TP/TSL protection before sending close order
        logger.debug(f"Clearing any existing protection for {position_side_to_close.upper()} before closing...")
        clear_stops_successful = self._set_position_protection(
            position_side_to_close, sl_price=None, tp_price=None, is_tsl=False # This sends "0" for all stops
        )
        if not clear_stops_successful:
            logger.warning(f"{Fore.YELLOW}Failed to explicitly confirm protection clear for {position_side_to_close.upper()}. Proceeding with close cautiously...{Style.RESET_ALL}")
            # Don't update tracker here, as API call failed. It might be cleared by the close order itself.
        else:
            logger.info(f"Protection cleared (or was already clear) for {position_side_to_close.upper()} position.")
            self.protection_tracker[tracker_key] = None # Update local tracker

        # 2. Execute Market Order to Close Position
        # For Bybit V5, to close a position, send an order of the opposite side with the same quantity.
        # The `reduceOnly` parameter is not directly available in `create_market_order` via CCXT in a simple way for all exchanges.
        # Bybit V5 API for orders has `reduceOnly` flag. If CCXT doesn't expose it easily for market orders,
        # the opposite side market order should still function to close/reduce.
        # Here, we rely on the opposite side market order.
        close_market_order_info = self._execute_market_order(closing_order_side, qty_abs_to_close)

        if not close_market_order_info:
            logger.error(f"{Fore.RED}Failed to submit closing market order for {position_side_to_close.upper()}. MANUAL INTERVENTION REQUIRED!{Style.RESET_ALL}")
            termux_notify(f"{symbol_to_trade} CLOSE ORDER FAILED", f"Market {closing_order_side.upper()} order failed!")
            # State of protection_tracker is uncertain. An external SL/TP might still exist if clear failed.
            return False

        close_order_id = close_market_order_info.get("id", "[N/A_CLOSE_ORDER_ID]")
        avg_close_price_str = close_market_order_info.get("average") # Average price of fills
        avg_close_price_decimal = safe_decimal(avg_close_price_str, default=Decimal("NaN"))

        logger.trade(
            f"{Fore.YELLOW}Closing market order ({close_order_id}) submitted for {position_side_to_close.upper()}. "
            f"Reported AvgClosePrice: {avg_close_price_decimal.normalize() if not avg_close_price_decimal.is_nan() else '[Pending/N/A]'}{Style.RESET_ALL}"
        )
        termux_notify(f"{symbol_to_trade} Position Closing", f"{position_side_to_close.upper()} close order {close_order_id} submitted.")

        # 3. Verify Position is Flat
        verification_ok, _ = self._verify_position_state( # Don't need final_verified_pos_state_after_close here
            expected_side_logical=None, # Expecting flat
            max_attempts=6,
            delay_seconds=max(self.config.order_check_delay_seconds + 0.5, 1.5),
            action_context=f"Post-{position_side_to_close.upper()}-Close Verification"
        )

        # 4. Log Trade Exit to Journal (regardless of verification, log the attempt)
        if self.config.enable_journaling:
            self.log_trade_exit_to_journal(
                position_side_to_close, qty_abs_to_close, avg_close_price_decimal, close_order_id, reason
            )

        if not verification_ok:
            logger.error(
                f"{Fore.RED}Position {position_side_to_close.upper()} closure verification FAILED. "
                f"Position may still be open. MANUAL INTERVENTION REQUIRED!{Style.RESET_ALL}"
            )
            termux_notify(f"{symbol_to_trade} CLOSE VERIFY FAILED", f"{position_side_to_close.upper()} position may still be open!")
            return False # Closure failed verification

        logger.trade(f"{Style.BRIGHT}{Fore.GREEN}Position {position_side_to_close.upper()} confirmed closed (flat) via verification.{Style.RESET_ALL}")
        self.protection_tracker[tracker_key] = None # Ensure tracker is None after confirmed close
        return True

    def _handle_entry_failure(
        self, failed_entry_order_side: str, # "buy" or "sell"
        attempted_qty_abs: Decimal
    ):
        """Handles cleanup after a failed entry sequence step, checking for and closing lingering positions."""
        logger.warning(
            f"{Fore.YELLOW}Handling potential entry failure for {failed_entry_order_side.upper()} order (intended qty: {attempted_qty_abs.normalize()}). "
            f"Checking for lingering position...{Style.RESET_ALL}"
        )
        logical_pos_side_to_check = "long" if failed_entry_order_side == "buy" else "short"

        time.sleep(max(self.config.order_check_delay_seconds, 1.0) + 1.0) # Wait for state to settle
        logger.debug(f"Checking current position status after {failed_entry_order_side.upper()} entry attempt failure...")

        # Fetch current position state. We don't have a strong expectation, just want to see what's open.
        _, current_positions_summary = self._verify_position_state(
            expected_side_logical=None, # No specific expectation
            max_attempts=2, delay_seconds=1.0,
            action_context=f"Entry-Failure-Cleanup-Check-{logical_pos_side_to_check.upper()}"
        )

        if current_positions_summary is None:
            logger.error(f"{Fore.RED}Could not fetch positions during entry failure handling for {logical_pos_side_to_check.upper()}. MANUAL CHECK URGENTLY REQUIRED!{Style.RESET_ALL}")
            termux_notify(f"{self.config.symbol} URGENT CHECK", "Failed to get position state during entry failure cleanup!")
            return

        lingering_pos_details = current_positions_summary.get(logical_pos_side_to_check, {})
        current_lingering_qty_abs = safe_decimal(lingering_pos_details.get("qty", "0")).copy_abs()

        if current_lingering_qty_abs >= POSITION_QTY_EPSILON: # If a position exists
            logger.error(
                f"{Fore.RED}Lingering {logical_pos_side_to_check.upper()} position (Qty: {current_lingering_qty_abs.normalize()}) "
                f"found after failed entry. Attempting emergency close...{Style.RESET_ALL}"
            )
            termux_notify(f"{self.config.symbol} Emergency Close", f"Lingering {logical_pos_side_to_check.upper()} pos found.")
            close_success = self.close_position(
                logical_pos_side_to_check, current_lingering_qty_abs, reason="EmergencyClose:LingeringAfterEntryFail"
            )
            if close_success:
                logger.info(f"Emergency close for lingering {logical_pos_side_to_check.upper()} position submitted/confirmed.")
            else:
                logger.critical(f"{Style.BRIGHT}{Fore.RED}EMERGENCY CLOSE FAILED for lingering {logical_pos_side_to_check.upper()}. MANUAL INTERVENTION URGENTLY REQUIRED!{Style.RESET_ALL}")
                termux_notify(f"{self.config.symbol} URGENT CHECK", f"Emergency close of lingering {logical_pos_side_to_check.upper()} FAILED!")
        else:
            logger.info(f"No significant lingering {logical_pos_side_to_check.upper()} position detected. Current qty: {current_lingering_qty_abs.normalize()}.")
            self.protection_tracker[logical_pos_side_to_check] = None # Ensure tracker is clear

    def _write_journal_row(self, trade_data: Dict[str, Any]):
        """Helper function to write a single row to the CSV trading journal."""
        if not self.config.enable_journaling: return
        journal_file = Path(self.config.journal_file_path)
        file_already_exists_and_has_content = journal_file.is_file() and journal_file.stat().st_size > 0

        try:
            journal_file.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            with journal_file.open("a", newline="", encoding="utf-8") as csvfile:
                fieldnames = [ # Define consistent field order
                    "TimestampUTC", "Symbol", "Action", "Side", "Quantity",
                    "AvgPrice", "OrderID", "Reason", "Notes"
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)

                if not file_already_exists_and_has_content: # Write header if new file or empty file
                    writer.writeheader()

                row_to_write = {}
                for field in fieldnames:
                    value = trade_data.get(field)
                    if isinstance(value, Decimal):
                        row_to_write[field] = 'NaN' if value.is_nan() else f"{value.normalize()}"
                    elif value is None:
                        row_to_write[field] = 'N/A'
                    else:
                        row_to_write[field] = str(value)
                row_to_write['Notes'] = trade_data.get('Notes', '') # Ensure 'Notes' is present

                writer.writerow(row_to_write)
            logger.debug(f"Trade action '{trade_data.get('Action', 'Unknown')}' logged to journal: {journal_file}")
        except IOError as e:
            logger.error(f"I/O error writing trade action '{trade_data.get('Action', '')}' to journal '{journal_file}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error writing trade action '{trade_data.get('Action', '')}' to journal: {e}", exc_info=True)

    def log_trade_entry_to_journal(
        self, order_side: str, # "buy" or "sell" (order side)
        filled_qty_abs: Decimal, avg_fill_price: Decimal, order_id: Optional[str]
    ):
        """Logs trade entry details to the CSV journal."""
        logical_position_side = "long" if order_side == "buy" else "short"
        entry_data = {
            "TimestampUTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "Symbol": self.config.symbol,
            "Action": "ENTRY",
            "Side": logical_position_side.upper(),
            "Quantity": filled_qty_abs,
            "AvgPrice": avg_fill_price, # Can be Decimal('NaN')
            "OrderID": order_id,
            "Reason": "Strategy Entry Signal",
        }
        self._write_journal_row(entry_data)

    def log_trade_exit_to_journal(
        self, position_side_closed: str, # "long" or "short" (logical side of closed position)
        closed_qty_abs: Decimal, avg_close_price: Decimal,
        order_id: Optional[str], exit_reason: str
    ):
        """Logs trade exit details to the CSV journal."""
        exit_data = {
            "TimestampUTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "Symbol": self.config.symbol,
            "Action": "EXIT",
            "Side": position_side_closed.upper(),
            "Quantity": closed_qty_abs,
            "AvgPrice": avg_close_price, # Can be Decimal('NaN')
            "OrderID": order_id,
            "Reason": exit_reason,
        }
        self._write_journal_row(exit_data)


# --- Status Display Class ---
class StatusDisplay:
    """Handles displaying the bot's status and key information using the Rich library."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self._default_price_dp_display = DEFAULT_PRICE_DP
        self._default_amount_dp_display = DEFAULT_AMOUNT_DP

    def _format_decimal_for_rich(
        self,
        value: Optional[Decimal],
        precision: Optional[int] = None,
        default_precision_fallback: int = 2,
        add_commas: bool = False,
        highlight_negative: bool = False,
        default_style: str = "white",
        style_override: Optional[str] = None
    ) -> Text:
        """Formats Decimal values for Rich Text display with styling options."""
        if value is None or (isinstance(value, Decimal) and value.is_nan()):
            return Text("N/A", style="dim")

        dp_to_use = precision if precision is not None else default_precision_fallback

        try:
            quantizer = Decimal("1e-" + str(dp_to_use))
            # Use ROUND_HALF_EVEN for display rounding, common for financial values
            formatted_decimal_val = value.quantize(quantizer, rounding=ROUND_HALF_EVEN)
            format_spec = f"{{:{',' if add_commas else ''}.{dp_to_use}f}}"
            display_string = format_spec.format(formatted_decimal_val)

            current_style = style_override if style_override else default_style
            if highlight_negative and not style_override:
                if formatted_decimal_val < 0: current_style = "bright_red"
                elif formatted_decimal_val > 0: current_style = "bright_green"
            return Text(display_string, style=current_style)
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.error(f"Error formatting decimal '{value}' for Rich display: {e}")
            return Text("ERR", style="bold bright_red")

    def print_status_panel(
        self,
        cycle_num: int,
        current_timestamp: Optional[datetime],
        current_market_price: Optional[Decimal],
        indicators_data: Optional[Dict],
        current_positions_summary: Optional[Dict],
        account_equity: Optional[Decimal],
        signal_check_result_or_status: Dict, # Can be signal dict or custom status dict
        protection_status_tracker: Dict, # OrderManager.protection_tracker
        market_specific_info: Optional[Dict] # ExchangeManager.market_info for precision
    ):
        """Prints the main status panel to the console using Rich Panel and Text objects."""
        price_display_dp = self._default_price_dp_display
        amount_display_dp = self._default_amount_dp_display
        if market_specific_info and "precision_dp" in market_specific_info:
             price_display_dp = market_specific_info["precision_dp"].get("price", self._default_price_dp_display)
             amount_display_dp = market_specific_info["precision_dp"].get("amount", self.config.ohlcv_limit) # Fallback to a large number for amount if needed, though DEFAULT_AMOUNT_DP is better.

        panel_content = Text()
        timestamp_str = current_timestamp.strftime("%Y-%m-%d %H:%M:%S %Z") if current_timestamp else Text("Timestamp N/A", style="dim").plain
        panel_title_str = f" Cycle {cycle_num} | {self.config.symbol} ({self.config.interval}) | {timestamp_str} "

        price_text = self._format_decimal_for_rich(current_market_price, precision=price_display_dp, style_override="bright_white")
        settle_ccy = market_specific_info.get("settle", "SETTLE") if market_specific_info else "SETTLE"
        equity_text = self._format_decimal_for_rich(account_equity, precision=2, add_commas=True, style_override="bright_yellow")

        panel_content.append("Price: ", style="bold bright_cyan"); panel_content.append(price_text)
        panel_content.append(" | ", style="dim")
        panel_content.append("Equity: ", style="bold bright_yellow"); panel_content.append(equity_text)
        panel_content.append(f" {settle_ccy}\n", style="bright_yellow")
        panel_content.append("---\n", style="dim")

        panel_content.append("Indicators: ", style="bold bright_cyan")
        if indicators_data:
            parts = []
            def fmt_ind(key: str, prec: int = 1, style: str = "white") -> Text:
                 val = indicators_data.get(key)
                 dec_val = val if isinstance(val, Decimal) else safe_decimal(str(val) if val is not None else "NaN")
                 return self._format_decimal_for_rich(dec_val, precision=prec, default_style=style)

            parts.append(Text("EMA(F/S/T): ").append(fmt_ind('fast_ema', price_display_dp, "cyan")).append("/")
                         .append(fmt_ind('slow_ema', price_display_dp, "magenta")).append("/")
                         .append(fmt_ind('trend_ema', price_display_dp, "yellow")))

            stoch_text = Text("Stoch(K/D/PrevK): ").append(fmt_ind('stoch_k', 1, "bright_blue")).append("/")
            stoch_text.append(fmt_ind('stoch_d', 1, "blue")).append("/")
            stoch_text.append(fmt_ind('stoch_k_prev', 1, "dim blue"))
            if indicators_data.get('stoch_kd_bullish'): stoch_text.append(" [b green]▲BullX[/]", style="green")
            elif indicators_data.get('stoch_kd_bearish'): stoch_text.append(" [b red]▼BearX[/]", style="red")
            parts.append(stoch_text)

            parts.append(Text(f"ATR({indicators_data.get('atr_period', self.config.atr_period)}): ")
                         .append(fmt_ind('atr', price_display_dp + 1, "bright_magenta"))) # ATR often needs more precision

            adx_val_dec = indicators_data.get('adx') if isinstance(indicators_data.get('adx'), Decimal) else safe_decimal(str(indicators_data.get('adx')))
            adx_style = "yellow" if not adx_val_dec.is_nan() and adx_val_dec > self.config.min_adx_level else "dim yellow"
            parts.append(Text(f"ADX({self.config.adx_period}): ")
                         .append(self._format_decimal_for_rich(adx_val_dec, 1, default_style=adx_style))
                         .append(" [+DI:", style="dim").append(fmt_ind('pdi', 1, "bright_green"))
                         .append(" -DI:", style="dim").append(fmt_ind('mdi', 1, "bright_red")).append("]", style="dim"))

            panel_content.append(Text(" | ", style="dim").join(parts)); panel_content.append("\n")
        else:
            panel_content.append(Text("Calculating or data unavailable...", style="dim")); panel_content.append("\n")
        panel_content.append("---\n", style="dim")

        panel_content.append("Position: ", style="bold bright_cyan")
        pos_display_text = Text("FLAT", style="bold bright_green")
        active_pos_side: Optional[str] = None
        active_pos_data: Optional[Dict] = None

        if current_positions_summary:
            long_data = current_positions_summary.get("long", {})
            short_data = current_positions_summary.get("short", {})
            if long_data and safe_decimal(long_data.get('qty', Decimal(0))).copy_abs() >= POSITION_QTY_EPSILON:
                active_pos_side, active_pos_data = "long", long_data
            elif short_data and safe_decimal(short_data.get('qty', Decimal(0))).copy_abs() >= POSITION_QTY_EPSILON:
                active_pos_side, active_pos_data = "short", short_data

        if active_pos_side and active_pos_data:
            style = "bold bright_green" if active_pos_side == "long" else "bold bright_red"
            pos_display_text = Text(f"{active_pos_side.upper()}: ", style=style)
            pos_display_text.append("Qty=", style=style).append(self._format_decimal_for_rich(active_pos_data.get("qty"), amount_display_dp))
            pos_display_text.append(" | EntryPx=", style="dim").append(self._format_decimal_for_rich(active_pos_data.get("entry_price"), price_display_dp))
            pos_display_text.append(" | PnL=", style="dim").append(self._format_decimal_for_rich(active_pos_data.get("unrealized_pnl"), 4, highlight_negative=True))

            # Protection Status (From Exchange vs. Local Tracker)
            local_tracker_state = protection_status_tracker.get(active_pos_side)
            exch_sl = active_pos_data.get("stop_loss_price")
            exch_tp = active_pos_data.get("take_profit_price")
            exch_tsl_active = active_pos_data.get("is_tsl_active", False)
            exch_tsl_trigger_px = active_pos_data.get("tsl_trigger_price")

            prot_status_text = Text(" | Protection: ", style="dim")
            exchange_prot_desc = Text("None", style="dim")
            if exch_tsl_active:
                exchange_prot_desc = Text("TSL Active", style="bright_magenta")
                if exch_tsl_trigger_px: exchange_prot_desc.append(f" (ActPx:{self._format_decimal_for_rich(exch_tsl_trigger_px, price_display_dp).plain})", style="dim")
            elif exch_sl or exch_tp:
                exchange_prot_desc = Text("SL/TP Active", style="bright_yellow")
                sl_tp_parts = []
                if exch_sl: sl_tp_parts.append(f"S:{self._format_decimal_for_rich(exch_sl, price_display_dp).plain}")
                if exch_tp: sl_tp_parts.append(f"T:{self._format_decimal_for_rich(exch_tp, price_display_dp).plain}")
                if sl_tp_parts: exchange_prot_desc.append(f" ({' '.join(sl_tp_parts)})", style="dim")

            prot_status_text.append("Exch:").append(exchange_prot_desc)
            prot_status_text.append(" LocalTrk:").append(Text(str(local_tracker_state) if local_tracker_state else "None", style="blue" if local_tracker_state else "dim"))

            # Consistency Check visual cue
            mismatch = False
            if exch_tsl_active and local_tracker_state != "ACTIVE_TSL": mismatch = True
            elif (exch_sl or exch_tp) and not exch_tsl_active and local_tracker_state != "ACTIVE_SLTP": mismatch = True
            elif not exch_tsl_active and not exch_sl and not exch_tp and local_tracker_state is not None: mismatch = True
            if mismatch: prot_status_text.append(Text(" [TrackerMismatch?]", style="bold bright_yellow"))

            pos_display_text.append(prot_status_text)

        panel_content.append(pos_display_text); panel_content.append("\n")
        panel_content.append("---\n", style="dim")

        panel_content.append("Signal/Status: ", style="bold bright_cyan")
        status_reason = signal_check_result_or_status.get("reason", Text("No status info", style="dim").plain)
        status_style_key = "dim"
        if signal_check_result_or_status.get("long") or "Long Signal" in status_reason or "ENTERED_long" in status_reason: status_style_key = "bold bright_green"
        elif signal_check_result_or_status.get("short") or "Short Signal" in status_reason or "ENTERED_short" in status_reason: status_style_key = "bold bright_red"
        elif "Blocked" in status_reason or "FAIL:" in status_reason or "EmergencyClose" in status_reason: status_style_key = "yellow"
        elif "CLOSED_" in status_reason or "HOLDING_" in status_reason or "INFO:" in status_reason: status_style_key = "bright_blue"
        elif "No Signal:" not in status_reason and "Initializing" not in status_reason: status_style_key = "white"

        wrapped_status_reason = "\n             ".join(textwrap.wrap(status_reason, width=100, subsequent_indent=""))
        panel_content.append(Text(wrapped_status_reason, style=status_style_key))

        console.print(Panel(panel_content, title=f"[bold bright_magenta]{panel_title_str}[/]", border_style="bright_blue", expand=False, padding=(1, 2)))


# --- Trading Bot Class ---
class TradingBot:
    """Main orchestrator class for the Pyrmethus trading bot."""

    def __init__(self):
        logger.info(
            f"{Style.BRIGHT}{Fore.MAGENTA}--- Initializing Pyrmethus v4.5.7 (Neon Nexus Edition) ---{Style.RESET_ALL}"
        )
        self.config = TradingConfig() # Load configuration first
        try:
            self.exchange_manager = ExchangeManager(self.config)
            # ExchangeManager init includes critical checks for exchange and market_info
            self.indicator_calculator = IndicatorCalculator(self.config)
            self.signal_generator = SignalGenerator(self.config)
            self.order_manager = OrderManager(self.config, self.exchange_manager)
            # OrderManager init includes critical checks for exchange_manager validity
        except ValueError as ve: # Specific catch for component init value errors (e.g. from OrderManager)
            logger.critical(f"{Style.BRIGHT}{Fore.RED}TradingBot initialization failed (Component Init Error): {ve}. Halting.{Style.RESET_ALL}")
            sys.exit(1)
        except Exception as e: # Catch-all for other unexpected init errors
            logger.critical(f"{Style.BRIGHT}{Fore.RED}Unexpected critical error during TradingBot component initialization: {e}. Halting.{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)

        self.status_display = StatusDisplay(self.config)
        self.shutdown_requested = False
        self._setup_signal_handlers()
        logger.info(f"{Style.BRIGHT}{Fore.GREEN}Pyrmethus components initialized successfully. Ready to conjure trades.{Style.RESET_ALL}")

    def _setup_signal_handlers(self):
        """Sets up OS signal handlers for graceful shutdown."""
        signals_to_handle = [signal.SIGINT, signal.SIGTERM]
        for sig in signals_to_handle:
            try:
                signal.signal(sig, self._signal_handler_callback)
                logger.debug(f"Signal handler for {signal.Signals(sig).name} set up.")
            except (ValueError, OSError, AttributeError, Exception) as e:
                 # AttributeError if on non-Unix platform without signals, ValueError if invalid signal for platform
                 logger.warning(f"{Fore.YELLOW}Could not set OS signal handler for {sig} (e.g., running on Windows or restricted environment): {e}{Style.RESET_ALL}")

    def _signal_handler_callback(self, sig_num: int, frame: Optional[Any]):
        """Internal callback for OS signals to initiate shutdown."""
        if not self.shutdown_requested:
            try: sig_name = signal.Signals(sig_num).name
            except (ValueError, AttributeError): sig_name = f"Signal {sig_num}"
            console.print(f"\n[bold yellow]Signal {sig_name} received. Initiating graceful shutdown... Please wait.[/]")
            logger.warning(f"Signal {sig_name} received. Initiating graceful shutdown...")
            self.shutdown_requested = True
        else:
            logger.warning("Shutdown sequence already in progress. Ignoring additional signal.")

    def _display_startup_info(self):
        """Displays key configuration parameters at startup using Rich Panel."""
        # log_level_str is globally defined after parsing ENV
        console.print(Panel(
            Text(
                f"Symbol: {self.config.symbol}\n"
                f"Interval: {self.config.interval}\n"
                f"Market Type: {self.config.market_type} (Category: {self.config.bybit_v5_category})\n"
                f"Position Index: {self.config.position_idx} (0=One-Way, 1=HedgeBuy, 2=HedgeSell)\n"
                f"Risk Per Trade: {self.config.risk_percentage * 100:.3f}%\n"
                f"SL/TP Multipliers (ATR): SL={self.config.sl_atr_multiplier.normalize()}, TP={self.config.tp_atr_multiplier.normalize()}\n"
                f"TSL Activation (ATR Mult): {self.config.tsl_activation_atr_multiplier.normalize()}, TSL Percent: {self.config.trailing_stop_percent.normalize()}%\n"
                f"Trade Only With Trend: {self.config.trade_only_with_trend}\n"
                f"Journaling Enabled: {self.config.enable_journaling} (File: '{self.config.journal_file_path}')\n"
                f"Log Level: {log_level_str}" # Display actual log level string used
                , style="bright_white"
            ),
            title="[bold cyan]Pyrmethus Configuration Summary[/]",
            border_style="cyan",
            expand=False
        ))

    def run(self):
        """Starts the main trading loop."""
        self._display_startup_info()
        termux_notify("Pyrmethus Started", f"Trading {self.config.symbol} on {self.config.interval} interval.")
        cycle_count = 0

        while not self.shutdown_requested:
            cycle_count += 1
            cycle_start_time_monotonic = time.monotonic()
            logger.debug(f"{Fore.BLUE}--- Starting Trading Cycle {cycle_count} ---{Style.RESET_ALL}")

            try:
                self.trading_spell_cycle(cycle_count)
            except KeyboardInterrupt:
                logger.warning("\nKeyboardInterrupt detected in main loop. Initiating shutdown.")
                self.shutdown_requested = True; break
            except ccxt.AuthenticationError as auth_err:
                logger.critical(f"{Style.BRIGHT}{Fore.RED}CRITICAL AUTH ERROR in cycle {cycle_count}: {auth_err}. Halting.{Style.RESET_ALL}", exc_info=False)
                termux_notify("Pyrmethus CRITICAL ERROR", f"Auth failed: {str(auth_err)[:100]}")
                self.shutdown_requested = True; break
            except SystemExit as se:
                 logger.warning(f"SystemExit (code {se.code}) encountered in trading cycle. Terminating.")
                 self.shutdown_requested = True; break
            except Exception as cycle_err:
                logger.error(f"{Style.BRIGHT}{Fore.RED}Unhandled exception in trading cycle {cycle_count}: {cycle_err}{Style.RESET_ALL}", exc_info=True)
                termux_notify("Pyrmethus Cycle Error", f"Exception in cycle {cycle_count}. Check logs.")
                sleep_duration_after_error = self.config.loop_sleep_seconds * 2
                logger.info(f"Sleeping for {sleep_duration_after_error}s after cycle error before retrying cycle logic.")
                time.sleep(sleep_duration_after_error) # Longer sleep after error
                continue # Continue to next cycle attempt

            cycle_duration_seconds = time.monotonic() - cycle_start_time_monotonic
            sleep_needed_seconds = max(0, self.config.loop_sleep_seconds - cycle_duration_seconds)
            logger.debug(f"Cycle {cycle_count} completed in {cycle_duration_seconds:.2f}s.")

            if not self.shutdown_requested and sleep_needed_seconds > 0:
                logger.debug(f"Sleeping for {sleep_needed_seconds:.2f} seconds until next cycle...")
                sleep_end_time = time.monotonic() + sleep_needed_seconds
                try:
                    while time.monotonic() < sleep_end_time and not self.shutdown_requested:
                        time.sleep(min(0.5, sleep_needed_seconds)) # Sleep in small increments
                except KeyboardInterrupt:
                    logger.warning("\nKeyboardInterrupt during sleep. Initiating shutdown.")
                    self.shutdown_requested = True

            if self.shutdown_requested:
                logger.info("Shutdown requested. Exiting main trading loop.")
                break

        self.graceful_shutdown()
        console.print(f"\n[bold bright_cyan]Pyrmethus ({self.config.symbol}) has completed its session and returned to the ether.[/]")

    def trading_spell_cycle(self, cycle_num: int) -> None:
        """Executes one complete cycle of the trading logic."""
        current_cycle_status_dict = {"reason": f"Cycle {cycle_num} Processing..."}

        # 1. Fetch Market Data (OHLCV)
        logger.debug("Fetching latest market data (OHLCV)...")
        ohlcv_df = self.exchange_manager.fetch_ohlcv()
        if ohlcv_df is None or ohlcv_df.empty:
            logger.error(f"{Fore.RED}Cycle {cycle_num} Aborted: Failed to fetch OHLCV data.{Style.RESET_ALL}")
            current_cycle_status_dict = {"reason": f"FAIL_CYCLE_{cycle_num}:FETCH_OHLCV_DATA"}
            self.status_display.print_status_panel(cycle_num, None, None, None, None, None, current_cycle_status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info)
            return

        try:
            latest_candle_data = ohlcv_df.iloc[-1]
            current_market_price = safe_decimal(latest_candle_data["close"])
            last_candle_timestamp = ohlcv_df.index[-1].to_pydatetime()
            if current_market_price.is_nan() or current_market_price <= 0:
                raise ValueError(f"Invalid latest close price from OHLCV: {current_market_price.normalize() if not current_market_price.is_nan() else 'NaN'}")
            logger.debug(f"Latest Candle: Ts={last_candle_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}, Price={current_market_price.normalize()}")
        except (IndexError, KeyError, ValueError, TypeError) as e:
            logger.error(f"{Fore.RED}Cycle {cycle_num} Aborted: Error processing latest candle data: {e}{Style.RESET_ALL}")
            current_cycle_status_dict = {"reason": f"FAIL_CYCLE_{cycle_num}:PROCESS_LATEST_CANDLE ({e})"}
            self.status_display.print_status_panel(cycle_num, None, None, None, None, None, current_cycle_status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info)
            return

        # 2. Calculate Indicators
        indicators = self.indicator_calculator.calculate_indicators(ohlcv_df)
        if not indicators:
            logger.error(f"{Fore.RED}Cycle {cycle_num} Aborted: Failed to calculate indicators.{Style.RESET_ALL}")
            current_cycle_status_dict = {"reason": f"FAIL_CYCLE_{cycle_num}:CALCULATE_INDICATORS"}
            self.status_display.print_status_panel(cycle_num, last_candle_timestamp, current_market_price, None, None, None, current_cycle_status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info)
            return

        # 3. Fetch Account Balance and Current Position State
        total_equity, _ = self.exchange_manager.get_balance()
        if total_equity is None or total_equity.is_nan() or total_equity <= 0:
            logger.error(f"{Fore.RED}Cycle {cycle_num} Aborted: Failed to fetch valid total equity (value: {total_equity}) or equity is zero/negative.{Style.RESET_ALL}")
            current_cycle_status_dict = {"reason": f"FAIL_CYCLE_{cycle_num}:FETCH_EQUITY_INVALID"}
            self.status_display.print_status_panel(cycle_num, last_candle_timestamp, current_market_price, indicators, None, total_equity, current_cycle_status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info)
            return

        current_positions_summary = self.exchange_manager.get_current_position()
        if current_positions_summary is None:
            logger.error(f"{Fore.RED}Cycle {cycle_num} Aborted: Failed to fetch current position state.{Style.RESET_ALL}")
            current_cycle_status_dict = {"reason": f"FAIL_CYCLE_{cycle_num}:FETCH_POSITION"}
            self.status_display.print_status_panel(cycle_num, last_candle_timestamp, current_market_price, indicators, None, total_equity, current_cycle_status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info)
            return

        # Determine active position from summary for this cycle
        active_pos_side_logical: Optional[str] = None; active_pos_details: Optional[Dict] = None
        if current_positions_summary.get("long", {}): active_pos_side_logical, active_pos_details = "long", current_positions_summary["long"]
        elif current_positions_summary.get("short", {}): active_pos_side_logical, active_pos_details = "short", current_positions_summary["short"]

        # 4. If in an Active Position: Manage Exits and TSL
        if active_pos_side_logical and active_pos_details:
            pos_qty_abs = safe_decimal(active_pos_details.get("qty"))
            pos_entry_price = safe_decimal(active_pos_details.get("entry_price"))
            current_atr = indicators.get("atr") # Should be Decimal

            # 4a. Manage Trailing Stop Loss (TSL) Activation
            # Only if current protection is SLTP and all data is valid
            if (self.order_manager.protection_tracker.get(active_pos_side_logical) == "ACTIVE_SLTP" and
                not pos_entry_price.is_nan() and pos_entry_price > 0 and
                not current_market_price.is_nan() and current_market_price > 0 and
                isinstance(current_atr, Decimal) and not current_atr.is_nan() and current_atr > 0):
                self.order_manager.manage_trailing_stop(
                    active_pos_side_logical, pos_entry_price, current_market_price, current_atr
                )
                # If TSL activated, protection_tracker is updated. Re-fetch position for display.
                if self.order_manager.protection_tracker.get(active_pos_side_logical) == "ACTIVE_TSL":
                    logger.debug("Re-fetching position summary after TSL management for display.")
                    current_positions_summary = self.exchange_manager.get_current_position()
                    # Re-evaluate active_pos_details based on new summary
                    if current_positions_summary:
                        if active_pos_side_logical == "long": active_pos_details = current_positions_summary.get("long", {})
                        elif active_pos_side_logical == "short": active_pos_details = current_positions_summary.get("short", {})
                    else: active_pos_details = None # Fetch failed, clear details

            # 4b. Check for Signal-Based Exits (only if TSL is not yet primary)
            if self.order_manager.protection_tracker.get(active_pos_side_logical) != "ACTIVE_TSL":
                exit_reason_signal = self.signal_generator.check_exit_signals(active_pos_side_logical, indicators)
                if exit_reason_signal:
                    logger.trade(f"Attempting to close {active_pos_side_logical.upper()} position due to: {exit_reason_signal}")
                    if not pos_qty_abs.is_nan() and pos_qty_abs > 0:
                        close_success = self.order_manager.close_position(active_pos_side_logical, pos_qty_abs, reason=exit_reason_signal)
                        current_cycle_status_dict = {"reason": f"CLOSED_{active_pos_side_logical.upper()}_BY_SIGNAL" if close_success else f"FAIL:CLOSE_SIGNAL_{active_pos_side_logical.upper()}"}
                        current_positions_summary = self.exchange_manager.get_current_position() # Refresh for display
                        self.status_display.print_status_panel(cycle_num, last_candle_timestamp, current_market_price, indicators, current_positions_summary, total_equity, current_cycle_status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info)
                        return # Action taken (close attempt), end cycle.
                    else:
                        logger.warning(f"Exit signal for {active_pos_side_logical.upper()} but position quantity invalid ({pos_qty_abs}). Cannot close.")

            # 4c. Re-check position: Might have been closed by exchange SL/TP/TSL
            # This is important if an exchange stop was hit during this cycle's processing time.
            logger.debug(f"Re-fetching position state for {active_pos_side_logical.upper()} after TSL/exit checks to confirm current status before proceeding to entry checks (if it became flat).")
            current_positions_summary_after_actions = self.exchange_manager.get_current_position()
            if current_positions_summary_after_actions is None:
                 logger.warning(f"Failed to re-fetch position state for {active_pos_side_logical.upper()} after TSL/exit checks. Status may be stale for entry decision.")
            else: # Successfully re-fetched
                current_positions_summary = current_positions_summary_after_actions
                # Re-evaluate active_pos_side_logical and active_pos_details
                new_long_pos_data = current_positions_summary.get("long", {})
                new_short_pos_data = current_positions_summary.get("short", {})
                if new_long_pos_data and safe_decimal(new_long_pos_data.get("qty", "0")).copy_abs() >= POSITION_QTY_EPSILON:
                    active_pos_side_logical = "long"; active_pos_details = new_long_pos_data
                elif new_short_pos_data and safe_decimal(new_short_pos_data.get("qty", "0")).copy_abs() >= POSITION_QTY_EPSILON:
                    active_pos_side_logical = "short"; active_pos_details = new_short_pos_data
                else: # Position became flat (e.g., hit SL/TP on exchange during this cycle)
                    if active_pos_side_logical: # If it *was* active and now isn't
                        logger.info(f"Position {active_pos_side_logical.upper()} appears to have been closed by exchange (e.g., SL/TP/TSL hit) during cycle processing.")
                        current_cycle_status_dict = {"reason": f"INFO:POS_{active_pos_side_logical.upper()}_CLOSED_BY_EXCH_STOP"}
                        self.order_manager.protection_tracker[active_pos_side_logical.lower()] = None # Clear tracker
                    active_pos_side_logical = None; active_pos_details = None


        # 5. If Flat (or became flat): Check for New Entry Signals
        if not active_pos_side_logical:
            logger.debug("Currently flat. Checking for new entry signals...")
            entry_signals = self.signal_generator.generate_signals(ohlcv_df, indicators)
            current_cycle_status_dict = entry_signals # Use signal reason as current status for display

            target_entry_order_side: Optional[str] = None
            if entry_signals.get("long"): target_entry_order_side = "buy"
            elif entry_signals.get("short"): target_entry_order_side = "sell"

            if target_entry_order_side:
                current_atr = indicators.get("atr") # Should be Decimal
                if (not total_equity.is_nan() and total_equity > 0 and
                    isinstance(current_atr, Decimal) and not current_atr.is_nan() and current_atr > 0 and
                    not current_market_price.is_nan() and current_market_price > 0):
                    entry_success = self.order_manager.place_risked_market_order(
                        target_entry_order_side, current_atr, total_equity, current_market_price
                    )
                    entered_logical_side = "long" if target_entry_order_side == "buy" else "short"
                    current_cycle_status_dict = {"reason": f"ENTERED_{entered_logical_side.upper()}" if entry_success else f"FAIL:ENTRY_{entered_logical_side.upper()}"}
                    current_positions_summary = self.exchange_manager.get_current_position() # Refresh for display
                    self.status_display.print_status_panel(cycle_num, last_candle_timestamp, current_market_price, indicators, current_positions_summary, total_equity, current_cycle_status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info)
                    return # Action taken (entry attempt), end cycle.
                else:
                    logger.warning(f"Cannot attempt {target_entry_order_side.upper()} entry: Missing critical data (Equity, ATR, or Price).")
                    current_cycle_status_dict = {"reason": f"FAIL:ENTRY_DATA_MISSING_{target_entry_order_side.upper()}"}
        else: # Still in an active position, and no exit signal triggered
            current_cycle_status_dict = {"reason": f"HOLDING_{active_pos_side_logical.upper()}"}


        # 6. Display Current Status (if no early return from an action)
        self.status_display.print_status_panel(
            cycle_num, last_candle_timestamp, current_market_price, indicators,
            current_positions_summary, total_equity, current_cycle_status_dict,
            self.order_manager.protection_tracker, self.exchange_manager.market_info
        )

    def graceful_shutdown(self):
        """Performs cleanup actions before the bot exits."""
        logger.info(f"{Style.BRIGHT}{Fore.YELLOW}--- Pyrmethus Graceful Shutdown Sequence Initiated ---{Style.RESET_ALL}")
        # For this V5 strategy, SL/TP/TSL are set on the exchange for the entire position.
        # These protections will remain active on the exchange even if the bot shuts down.
        # If there were client-side managed orders (e.g., limit orders not yet filled),
        # they might be cancelled here. This strategy uses market orders for entry/exit,
        # and position-level stops, so no specific order cancellation is typically needed on shutdown.
        logger.info("Graceful shutdown: Positions with exchange-side protection (SL/TP/TSL) will remain active. No other specific cleanup actions configured for this strategy beyond logging.")
        termux_notify("Pyrmethus Shutdown", f"Bot for {self.config.symbol} is shutting down.")
        logger.info(f"{Style.BRIGHT}{Fore.YELLOW}--- Pyrmethus Shutdown Complete ---{Style.RESET_ALL}")


if __name__ == "__main__":
    try:
        bot = TradingBot()
        bot.run()
    except SystemExit as e:
        if e.code == 0: logger.info("Pyrmethus terminated normally.")
        else: logger.warning(f"Pyrmethus terminated with exit code: {e.code}")
        sys.exit(e.code) # Propagate the exit code
    except Exception as main_exception:
        # Use already configured logger if available, otherwise print
        log_func = logger.critical if 'logger' in globals() else print
        # Ensure colorama is initialized for this final error message if logger isn't fully set up
        if 'colorama_init' in globals() and 'Fore' in globals() and 'Style' in globals():
             colorama_init(autoreset=True)
             err_msg = f"{Style.BRIGHT}{Fore.RED}CRITICAL UNHANDLED EXCEPTION in Pyrmethus main execution: {main_exception}{Style.RESET_ALL}"
        else:
             err_msg = f"CRITICAL UNHANDLED EXCEPTION in Pyrmethus main execution: {main_exception}"

        if 'logger' in globals() and hasattr(logger, 'critical'):
            logger.critical(err_msg, exc_info=True)
        else: # Fallback to print if logger is not available
            print(err_msg, file=sys.stderr)
            import traceback
            traceback.print_exc() # Print stack trace to stderr

        if 'termux_notify' in globals(): # termux_notify might not be defined if imports failed early
            termux_notify("Pyrmethus CRASHED", "Critical unhandled exception. Check logs!")
        sys.exit(1)

