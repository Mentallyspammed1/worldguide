# -*- coding: utf-8 -*-
# pylint: disable=logging-fstring-interpolation, too-many-instance-attributes, too-many-arguments, too-many-locals, too-many-public-methods, invalid-name, unused-argument, too-many-lines, unnecessary-pass, unnecessary-lambda-assignment, bad-option-value, line-too-long
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

Conjures market insights and executes trades on Bybit Futures using the
V5 Unified Account API via CCXT. Refactored into classes for better structure
and utilizing V5 position-based stop-loss/take-profit/trailing-stop features.


# Standard Library Imports
import copy
import csv
import logging
import os
import platform # Not actively used, but kept for potential future use
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
    if "colorama" in str(e): # Special handling if colorama itself is missing
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
        if os.getenv("TERMUX_VERSION"): # Termux specific instructions
            # For Termux, pandas and numpy are often better installed via pkg
            termux_pkgs_to_install = []
            pip_pkgs_to_install = []
            if "pandas" in COMMON_PACKAGES: termux_pkgs_to_install.append("python-pandas")
            if "numpy" in COMMON_PACKAGES: termux_pkgs_to_install.append("python-numpy")
            pip_pkgs_to_install = [p for p in COMMON_PACKAGES if p not in ['pandas', 'numpy']]

            install_cmd = "pkg install python"
            if termux_pkgs_to_install: install_cmd += f" {' '.join(termux_pkgs_to_install)}"
            if pip_pkgs_to_install: install_cmd += f" && pip install {' '.join(pip_pkgs_to_install)}"

            print(f"{Style.BRIGHT}{install_cmd}{Style.RESET_ALL}")
            print(
                f"{Fore.YELLOW}Note: pandas and numpy are often best installed via 'pkg' in Termux for compatibility.{Style.RESET_ALL}"
            )
        else: # Standard pip install for other systems
            print(
                f"{Style.BRIGHT}pip install {' '.join(COMMON_PACKAGES)}{Style.RESET_ALL}"
            )
        sys.exit(1)

# --- Constants ---
DECIMAL_PRECISION = 50
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
if not hasattr(logging.Logger, "trade"): # Ensure it's not already defined
    logging.addLevelName(TRADE_LEVEL_NUM, "TRADE")

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
if log_level_str not in valid_log_levels and log_level_str != str(TRADE_LEVEL_NUM):
    # If it's not a standard name or our custom level number, default to INFO
    if log_level_str.isdigit() and int(log_level_str) == TRADE_LEVEL_NUM:
        pass # It's our custom level number, allow it
    else:
        print(f"Warning: Invalid LOG_LEVEL '{log_level_str}'. Defaulting to INFO.") # Early print, logger not fully set up
        log_level_str = "INFO"

log_level = getattr(logging, log_level_str, logging.INFO) if log_level_str not in [str(TRADE_LEVEL_NUM)] else TRADE_LEVEL_NUM
logger.setLevel(log_level)

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
        if str_value.lower() in ["nan", "none", "null", ""]: # "null" for JSON, "" for empty env var
            return default
        return Decimal(str_value)
    except (InvalidOperation, ValueError, TypeError):
        # Optional: logger.debug(f"Could not convert '{value}' (type: {type(value).__name__}) to Decimal, using default {default}")
        return default


def termux_notify(title: str, content: str) -> None:
    """Sends a notification via Termux API (toast), if available. Title is ignored by termux-toast."""
    # Check if running in Termux environment more reliably by checking PREFIX
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
        requests.exceptions.ReadTimeout, # Added ReadTimeout for requests
    ),
    fatal_exceptions: Tuple[Type[Exception], ...] = (
        ccxt.AuthenticationError, ccxt.PermissionDenied # Errors that should halt immediately
    ),
    fail_fast_exceptions: Tuple[Type[Exception], ...] = (
         ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.OrderNotFound # Errors where retrying is pointless
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
            logger.error(f"{Fore.RED}Fail-fast error ({type(e).__name__}) executing {func_name}: {e}. Not retrying.{Style.RESET_ALL}")
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
            # Symbol format for CCXT is typically BASE/QUOTE:SETTLE (e.g., BTC/USDT:USDT for USDT linear)
            # or BASE/QUOTE (e.g., BTC/USD for BTC inverse if CCXT implies :BTC)
            if ":" not in self.symbol:
                # Attempt to infer if not explicit, though explicit is better.
                # This part might need adjustment based on how symbols are defined for inverse.
                # For Bybit V5, category is often 'linear' or 'inverse'.
                logger.warning(f"Symbol '{self.symbol}' does not explicitly state settle currency. Inferring category based on MARKET_TYPE.")
                if self.market_type == "inverse":
                    category = "inverse" # e.g., BTC/USD (settled in BTC)
                elif self.market_type in ["linear", "swap"]:
                    category = "linear"  # e.g., BTC/USDT (settled in USDT)
                else: # Should be caught by _get_env validation
                    raise ValueError(f"Unsupported MARKET_TYPE '{self.market_type}' for category determination.")
            else: # Symbol includes settle currency
                # settle_currency = self.symbol.split(":")[-1]
                # base_currency = self.symbol.split("/")[0]
                if self.market_type == "inverse":
                    category = "inverse" # e.g., BTC/USD:BTC
                elif self.market_type in ["linear", "swap"]:
                    category = "linear" # e.g., BTC/USDT:USDT or BTC/USDC:USDC
                else:
                    raise ValueError(f"Unsupported MARKET_TYPE '{self.market_type}' for category determination.")

            logger.info(
                f"Determined Bybit V5 API category: '{category}' for symbol '{self.symbol}' and type '{self.market_type}'"
            )
            return category
        except ValueError as e:
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Could not determine V5 category: {e}. Halting.{Style.RESET_ALL}",
                exc_info=True, # Show traceback for this critical error
            )
            sys.exit(1)

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

    # --- Enhanced _get_env and helpers (integrated from Snippet 1) ---
    def _cast_value(self, key: str, value_str: str, cast_type: Type, default: Any) -> Any:
        """Helper to cast string value to target type, returning default on failure. Handles empty strings."""
        val_to_cast = value_str.strip() # Strip whitespace before casting
        if not val_to_cast: # Handle empty string after strip
            logger.warning(f"Empty value string for '{key}' after stripping. Using default '{default}'.")
            return default
        try:
            if cast_type == bool:
                return val_to_cast.lower() in ["true", "1", "yes", "y", "on"]
            elif cast_type == Decimal:
                # Check for common non-numeric strings before attempting Decimal conversion
                if val_to_cast.lower() in ["nan", "none", "null"]: # "null" for JSON, "" already handled
                    raise ValueError("Non-numeric string cannot be cast to Decimal")
                return Decimal(val_to_cast)
            elif cast_type == int:
                # Use Decimal as an intermediary for robust int conversion (e.g., "10.0" -> 10)
                dec_val = Decimal(val_to_cast)
                if dec_val.as_tuple().exponent < 0: # Check if there's a fractional part
                    raise ValueError("Decimal value with fractional part cannot be cast to int without loss.")
                return int(dec_val)
            # Add other specific casts if needed (e.g., float, though Decimal is preferred for finance)
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
        """Helper to validate a value against min/max constraints and allowed values. Logs and returns False on failure."""
        # Type check for numeric comparison (min_val/max_val)
        is_numeric_comparable = isinstance(value, (int, float, Decimal))
        if (min_val is not None or max_val is not None) and not is_numeric_comparable:
            logger.error(f"Validation failed for '{key}': Non-numeric value '{value}' (type: {type(value).__name__}) cannot be compared with min/max.")
            return False # Cannot perform min/max validation

        # Min/Max checks (critical, halts if violated)
        if min_val is not None and is_numeric_comparable and value < min_val: # type: ignore
            logger.critical(f"{Style.BRIGHT}{Fore.RED}Validation failed for '{key}': Value '{value}' is less than minimum '{min_val}'. Halting.{Style.RESET_ALL}")
            sys.exit(1)
        if max_val is not None and is_numeric_comparable and value > max_val: # type: ignore
            logger.critical(f"{Style.BRIGHT}{Fore.RED}Validation failed for '{key}': Value '{value}' is greater than maximum '{max_val}'. Halting.{Style.RESET_ALL}")
            sys.exit(1)

        # Allowed values check (non-critical, logs error and returns False)
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

        if value_str is None or value_str.strip() == "": # Check if env var is not set or is empty string
            if default is None and not is_secret: # Required config, no default (non-secret)
                 logger.critical(f"{Style.BRIGHT}{Fore.RED}Required configuration '{key}' not found in environment and no default provided. Halting.{Style.RESET_ALL}")
                 sys.exit(1)
            elif default is None and is_secret: # Required secret, no default
                 logger.critical(f"{Style.BRIGHT}{Fore.RED}Required secret configuration '{key}' not found. Halting.{Style.RESET_ALL}")
                 sys.exit(1)

            use_default_flag = True
            value_to_process_str = str(default) # Use string representation of default for casting
            source_info = f"default value ({default})"
            # For logging, display the original default value, not its string representation if it's not a string
            log_value_display = default if not is_secret else "****"
        else:
            value_to_process_str = value_str
            log_value_display = "****" if is_secret else value_str_to_process # Mask secrets

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
            # So, if we are here, it's likely an allowed_values mismatch for a non-default value
            # or a type issue that didn't get caught by _cast_value returning default.
            # Revert to the original default value provided to _get_env.
            logger.warning(
                f"{color}Reverting '{key}' to its original default '{default}' due to validation failure of processed value '{casted_value}'.{Style.RESET_ALL}"
            )
            casted_value = default # Use the original default value passed to the function

            # Critical: Re-validate the original default value itself. This ensures defaults in code are valid.
            if not self._validate_value(key, casted_value, min_val, max_val, allowed_values):
                logger.critical(
                    f"{Style.BRIGHT}{Fore.RED}FATAL: The hardcoded default value '{default}' for '{key}' itself failed validation. Halting.{Style.RESET_ALL}"
                )
                sys.exit(1)
        return casted_value
    # --- End Enhanced _get_env ---

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
                    "brokerId": "TermuxNeonV5", # Custom broker ID for Bybit referral/tracking
                    # V5 specific options if needed, though CCXT often handles these:
                    # 'createMarketBuyOrderRequiresPrice': False, # Bybit V5 default behavior
                    # 'defaultMarginMode': "isolated", # Or 'cross'
                    "defaultTimeInForce": "GTC", # Good-Till-Cancelled
                },
            }
            # Example for using a testnet/sandbox endpoint via an environment variable
            if os.getenv("USE_BYBIT_TESTNET", "false").lower() == "true":
                logger.warning(f"{Fore.YELLOW}Using Bybit Testnet endpoint.{Style.RESET_ALL}")
                exchange_params['urls'] = {'api': 'https://api-testnet.bybit.com'}

            self.exchange = ccxt.bybit(exchange_params)
            # Test connectivity by fetching server time (common practice)
            logger.debug("Testing exchange connection by fetching server time...")
            self.exchange.fetch_time()
            logger.info(
                f"{Style.BRIGHT}{Fore.GREEN}Bybit V5 interface initialized and connection tested successfully.{Style.RESET_ALL}"
            )

        except ccxt.AuthenticationError as e:
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Authentication failed: {e}. Check API keys and permissions. Halting.{Style.RESET_ALL}",
                exc_info=False, # No need for full traceback for auth errors
            )
            sys.exit(1)
        except (ccxt.NetworkError, requests.exceptions.RequestException) as e: # Catch network-related issues
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Network error initializing exchange: {e}. Check internet connection and endpoint. Halting.{Style.RESET_ALL}",
                exc_info=True,
            )
            sys.exit(1)
        except Exception as e: # Catch any other unexpected error during initialization
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Unexpected error initializing exchange: {e}. Halting.{Style.RESET_ALL}",
                exc_info=True,
            )
            sys.exit(1)

    def _load_market_info(self) -> Optional[Dict[str, Any]]:
        """Loads and caches market information for the configured symbol, including precision details."""
        if not self.exchange: # Should not happen if _initialize_exchange was successful
            logger.error("Exchange not initialized, cannot load market info.")
            return None
        try:
            logger.info(f"Loading market info for symbol: {self.config.symbol}...")
            # Force reload of markets to ensure up-to-date information
            self.exchange.load_markets(reload=True)
            market = self.exchange.market(self.config.symbol)
            if not market: # Should be caught by ccxt.BadSymbol if symbol is invalid
                # This is a safeguard; ccxt.market() usually raises if symbol not found after load_markets
                raise ccxt.ExchangeError(
                    f"Market {self.config.symbol} not found on exchange after loading markets."
                )

            # Safely extract precision and convert to integer decimal places (dp)
            amount_precision_raw = market.get("precision", {}).get("amount")
            price_precision_raw = market.get("precision", {}).get("price")

            # Helper to determine decimal places from CCXT precision value (which can be step size or dp)
            def get_dp_from_precision(precision_val: Optional[Union[str, float, int]], default_dp: int) -> int:
                if precision_val is None: return default_dp
                prec_dec = safe_decimal(precision_val) # Convert to Decimal safely
                if prec_dec.is_nan(): return default_dp
                if prec_dec == 0: return 0 # Precision is exactly 0, so 0 dp

                if prec_dec < 1: # Likely a step size (e.g., 0.01 for 2 dp, 0.001 for 3 dp)
                     exponent = prec_dec.as_tuple().exponent # e.g., Decimal('0.01').as_tuple().exponent is -2
                     # Exponent is already an int for finite Decimals. abs() gives dp.
                     return abs(exponent)
                else: # Likely number of decimal places directly (e.g., 3 for 3 dp)
                    try:
                        # Ensure it's a whole number if it's >= 1
                        if prec_dec % 1 == 0: return int(prec_dec)
                        else: return default_dp # Not a simple integer, use default
                    except (ValueError, TypeError, InvalidOperation):
                        return default_dp

            amount_dp = get_dp_from_precision(amount_precision_raw, DEFAULT_AMOUNT_DP)
            price_dp = get_dp_from_precision(price_precision_raw, DEFAULT_PRICE_DP)

            market["precision_dp"] = {"amount": amount_dp, "price": price_dp}
            market["tick_size"] = Decimal("1e-" + str(price_dp)) # Smallest price increment
            min_amount_raw = market.get("limits", {}).get("amount", {}).get("min")
            market["min_order_size"] = safe_decimal(min_amount_raw, default=Decimal("NaN"))
            market["contract_size"] = safe_decimal(market.get("contractSize", "1"), default=Decimal("1")) # Usually 1 for linear

            min_amt_str = market["min_order_size"].normalize() if not market["min_order_size"].is_nan() else "N/A"
            logger.info(
                f"Market info for {self.config.symbol} (ID: {market.get('id')}): "
                f"Precision(AmountDP={amount_dp}, PriceDP={price_dp}), TickSize={market['tick_size'].normalize()}, "
                f"Limits(MinAmount={min_amt_str}), ContractSize={market['contract_size'].normalize()}"
            )
            return market
        except (ccxt.ExchangeError, KeyError, ValueError, TypeError, Exception) as e: # Catch broad errors during market loading
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Failed to load or parse market info for {self.config.symbol}: {e}. Halting.{Style.RESET_ALL}",
                exc_info=True,
            )
            sys.exit(1)
        return None # Should be unreachable if sys.exit occurs

    def format_price(self, price: Union[Decimal, str, float, int]) -> str:
        """Formats a price value to a string according to market precision using ROUND_HALF_EVEN."""
        price_decimal = safe_decimal(price)
        if price_decimal.is_nan():
            return "NaN" # Return "NaN" string if input was invalid

        precision_dp = DEFAULT_PRICE_DP # Default if market info or precision not found
        if self.market_info and "precision_dp" in self.market_info and "price" in self.market_info["precision_dp"]:
            precision_dp = self.market_info["precision_dp"]["price"]

        try:
            quantizer = Decimal("1e-" + str(precision_dp)) # e.g., Decimal('0.01') for 2 dp
            # ROUND_HALF_EVEN is standard ("round half to even", unbiased rounding)
            formatted_price_decimal = price_decimal.quantize(quantizer, rounding=ROUND_HALF_EVEN)
            # Ensure the string representation has exactly 'precision_dp' decimal places
            return f"{formatted_price_decimal:.{precision_dp}f}"
        except (InvalidOperation, ValueError) as e: # Catch errors during quantize or formatting
             logger.error(f"Error formatting price {price_decimal} to {precision_dp}dp: {e}")
             return "ERR" # Indicate a formatting error

    def format_amount(
        self, amount: Union[Decimal, str, float, int], rounding_mode=ROUND_DOWN
    ) -> str:
        """Formats an amount (quantity) to a string according to market precision, default ROUND_DOWN."""
        amount_decimal = safe_decimal(amount)
        if amount_decimal.is_nan():
            return "NaN"

        precision_dp = DEFAULT_AMOUNT_DP
        if self.market_info and "precision_dp" in self.market_info and "amount" in self.market_info["precision_dp"]:
            precision_dp = self.market_info["precision_dp"]["amount"]

        try:
            quantizer = Decimal("1e-" + str(precision_dp))
            # Default rounding_mode is ROUND_DOWN for order quantities to avoid over-ordering due to precision limits
            formatted_amount_decimal = amount_decimal.quantize(quantizer, rounding=rounding_mode)
            return f"{formatted_amount_decimal:.{precision_dp}f}"
        except (InvalidOperation, ValueError) as e:
             logger.error(f"Error formatting amount {amount_decimal} to {precision_dp}dp: {e}")
             return "ERR"

    # --- V5 Parameter Formatting Helper (integrated from Snippet 4) ---
    def _format_v5_param(
            self,
            value: Optional[Union[Decimal, str, float, int]],
            param_type: str = "price", # 'price', 'amount', or 'distance' (distance uses price precision)
            allow_zero: bool = False # Whether "0" or "0.00" is a valid formatted parameter
        ) -> Optional[str]:
        """
        Formats a numeric value as a string suitable for Bybit V5 API parameters.
        Returns None if the value is invalid, cannot be formatted positively (unless allow_zero=True),
        or if formatting itself fails.
        """
        if value is None:
            # logger.debug("V5 Param Formatting: Input value is None.")
            return None # None input results in None output, typically to omit the parameter

        decimal_value = safe_decimal(value, default=Decimal("NaN"))

        if decimal_value.is_nan():
            logger.warning(f"V5 Param Formatting: Input '{value}' (type: {type(value).__name__}) converted to NaN Decimal. Cannot format.")
            return None

        # Handle zero based on allow_zero flag and ensure it's not negative zero
        is_zero_val = decimal_value.is_zero() and not decimal_value.is_signed() # Check for +0 or 0
        if is_zero_val:
            if allow_zero:
                # Format "0" using the appropriate precision (price or amount)
                formatter = self.format_price if param_type in ["price", "distance"] else self.format_amount
                # Ensure rounding mode for amount is appropriate if formatting zero (usually doesn't matter for 0)
                formatted_zero_str = formatter(Decimal("0"))
                # Check if formatting zero resulted in an error string
                return formatted_zero_str if formatted_zero_str not in ["ERR", "NaN"] else None
            else: # Zero is not allowed
                # logger.debug(f"V5 Param Formatting: Input value '{value}' is zero, but zero is not allowed for this parameter.")
                return None
        elif decimal_value < 0: # Negative values are usually invalid for price/amount/distance parameters
            logger.warning(f"V5 Param Formatting: Input value '{value}' is negative ({decimal_value}), which is typically invalid for API parameters.")
            return None

        # Select appropriate formatter based on param_type for positive, non-zero values
        formatter_func: Callable[[Union[Decimal, str, float, int]], str]
        rounding_for_amount = ROUND_DOWN # Default for amounts

        if param_type == "price" or param_type == "distance": # distance often refers to a price difference
            formatter_func = self.format_price
            # For price/distance, format_price uses ROUND_HALF_EVEN by default
            formatted_str = formatter_func(decimal_value)
        elif param_type == "amount":
            # Use format_amount with specified rounding (default ROUND_DOWN)
            formatted_str = self.format_amount(decimal_value, rounding_mode=rounding_for_amount)
        else:
            logger.error(f"V5 Param Formatting: Unknown param_type '{param_type}'. Cannot format '{value}'.")
            return None

        # Final check: ensure the formatted string isn't an error/NaN indicator from formatter
        if formatted_str in ["ERR", "NaN"]:
            logger.error(f"V5 Param Formatting: Failed to produce a valid string for '{value}' (type: {param_type}). Formatter returned: {formatted_str}")
            return None

        # logger.debug(f"V5 Param Formatted: Input='{value}', Type='{param_type}', Output='{formatted_str}'")
        return formatted_str
    # --- End V5 Parameter Formatting Helper ---

    def fetch_ohlcv(self) -> Optional[pd.DataFrame]:
        """Fetches OHLCV data with retries, converts to DataFrame, and processes numeric columns."""
        if not self.exchange: # Should be initialized
            logger.error("Exchange not initialized, cannot fetch OHLCV.")
            return None
        logger.debug(
            f"Fetching up to {self.config.ohlcv_limit} OHLCV candles for {self.config.symbol} (Timeframe: {self.config.interval})..."
        )
        try:
            # Use fetch_with_retries utility for robust API call
            ohlcv_data = fetch_with_retries(
                self.exchange.fetch_ohlcv, # Pass the CCXT method
                symbol=self.config.symbol,
                timeframe=self.config.interval,
                limit=self.config.ohlcv_limit,
                # Retry parameters are taken from defaults or can be customized here
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
            )
            if not ohlcv_data: # Empty list from API
                logger.error(f"fetch_ohlcv for {self.config.symbol} returned no data (empty list).")
                return None
            if len(ohlcv_data) < 20: # Arbitrary threshold for "too little" data
                 logger.warning(f"Fetched only {len(ohlcv_data)} candles. This might be insufficient for some indicators requiring longer lookbacks.")

            df = pd.DataFrame(
                ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            # Convert timestamp to UTC localized datetime objects and set as index
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)

            # Convert OHLCV columns to Decimal robustly using safe_decimal
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].map(safe_decimal) # Apply safe_decimal to each element
                # Check if any conversion resulted in NaN, which indicates problematic data from API
                if df[col].apply(lambda x: isinstance(x, Decimal) and x.is_nan()).any():
                     logger.warning(f"Column '{col}' in OHLCV data contains NaN values after Decimal conversion. Check API data source if issues persist.")

            # Drop rows where essential price data (O, H, L, C) is NaN after conversion
            initial_len = len(df)
            df.dropna(subset=["open", "high", "low", "close"], inplace=True, how="any") # Drop if any of these are NaN
            if len(df) < initial_len:
                logger.warning(f"Dropped {initial_len - len(df)} rows from OHLCV data due to NaN values in critical O/H/L/C columns.")

            if df.empty: # If all rows were dropped or df was empty to begin with
                 logger.error("OHLCV DataFrame is empty after processing (NaN drop or initial empty). Cannot proceed with this data.")
                 return None

            logger.debug(
                f"Fetched and processed {len(df)} OHLCV candles. Last timestamp: {df.index[-1]}"
            )
            return df
        except Exception as e: # Catch exceptions from fetch_with_retries or DataFrame processing
            logger.error(f"Failed to fetch or process OHLCV data for {self.config.symbol}: {e}", exc_info=True)
            return None

    def get_balance(self) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Fetches total equity and available balance for the settlement currency using V5 API."""
        if not self.exchange or not self.market_info:
            logger.error("Exchange or market info not available, cannot fetch balance.")
            return None, None

        settle_currency = self.market_info.get("settle") # e.g., USDT, BTC
        if not settle_currency:
            logger.error("Settle currency not found in market info. Cannot determine balance currency.")
            return None, None

        logger.debug(
            f"Fetching balance for {settle_currency} (Account: {V5_UNIFIED_ACCOUNT_TYPE}, Category: {self.config.bybit_v5_category})..."
        )
        try:
            # For Bybit V5, fetch_balance might require specific params for Unified accounts.
            # CCXT usually abstracts this, but providing `accountType` can be beneficial.
            params = {
                "accountType": V5_UNIFIED_ACCOUNT_TYPE,
                "coin": settle_currency, # Request specific coin balance to narrow down response
            }
            balance_data = fetch_with_retries(
                self.exchange.fetch_balance,
                params=params,
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
            )
            # logger.debug(f"Raw balance data from exchange: {balance_data}") # Very verbose

            total_equity = Decimal("NaN")
            available_balance = Decimal("NaN")

            # --- Parse V5 Response Structure (Unified Account) ---
            # CCXT aims to unify the response, but structure can vary.
            # Primary check: Standard CCXT 'total' and 'free' dictionaries for the settle currency.
            if settle_currency in balance_data.get("total", {}):
                total_equity = safe_decimal(balance_data["total"].get(settle_currency))
            if settle_currency in balance_data.get("free", {}):
                available_balance = safe_decimal(balance_data["free"].get(settle_currency))

            # Secondary check: If primary parsing yields NaN, look into `info` for Bybit's raw V5 structure.
            # This is common for Unified accounts where overall equity vs. coin-specific balance might differ.
            if (total_equity.is_nan() or available_balance.is_nan()) and "info" in balance_data:
                logger.debug("Parsing balance from 'info' field as fallback (V5 structure)...")
                info_result = balance_data["info"].get("result", {})
                account_list = info_result.get("list", []) # V5 returns a list of account details
                if account_list and isinstance(account_list, list):
                    # Find the Unified account details within the list
                    unified_acc_info = next((item for item in account_list if item.get("accountType") == V5_UNIFIED_ACCOUNT_TYPE), None)
                    if unified_acc_info:
                        if total_equity.is_nan(): # Try to get overall total equity for the account
                            total_equity = safe_decimal(unified_acc_info.get("totalEquity"))
                        if available_balance.is_nan(): # Try to get overall available balance
                            available_balance = safe_decimal(unified_acc_info.get("totalAvailableBalance"))

                        # If available_balance is still NaN, check coin-specific details within the Unified account
                        if available_balance.is_nan() and "coin" in unified_acc_info:
                             coin_list_details = unified_acc_info.get("coin", [])
                             if coin_list_details and isinstance(coin_list_details, list):
                                 settle_coin_info = next((c for c in coin_list_details if c.get("coin") == settle_currency), None)
                                 if settle_coin_info:
                                      # Use 'availableToWithdraw' or similar for the specific coin
                                      available_balance = safe_decimal(settle_coin_info.get("availableToWithdraw"))
                                      # If total_equity was also NaN, coin-specific equity might be a fallback
                                      if total_equity.is_nan():
                                          total_equity = safe_decimal(settle_coin_info.get("equity"))
            # Final Validations
            if total_equity.is_nan():
                logger.error(
                    f"Could not extract valid total equity for {settle_currency}. Balance data format might be unexpected. Raw snippet: {str(balance_data)[:500]}"
                ) # Log part of raw data for debug
                # Return None for equity, but try to return available_balance if found
                return None, available_balance if not available_balance.is_nan() else Decimal("0")

            if available_balance.is_nan(): # If available is still NaN, default to 0 with a warning
                logger.warning(
                    f"Could not extract valid available balance for {settle_currency}. Defaulting to 0. Check raw balance data if issues persist."
                )
                available_balance = Decimal("0")

            logger.debug(
                f"Balance Fetched ({settle_currency}): Total Equity = {total_equity.normalize()}, Available Balance = {available_balance.normalize()}"
            )
            return total_equity, available_balance
        except Exception as e:
            logger.error(f"Failed to fetch or parse balance: {e}", exc_info=True)
            return None, None

    def get_current_position(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Fetches current position details for the configured symbol using V5 API.
        Returns a dictionary structured as {'long': {details}, 'short': {details}} or None on error.
        Empty dicts for 'long'/'short' if no position.
        """
        if not self.exchange or not self.market_info:
            logger.error("Exchange or market info not available, cannot fetch position.")
            return None

        market_id = self.market_info.get("id") # Exchange-specific symbol ID
        if not market_id:
            logger.error("Market ID not found in market info. Cannot fetch position.")
            return None

        logger.debug(
            f"Fetching position for {self.config.symbol} (ID: {market_id}, Category: {self.config.bybit_v5_category}, PositionIdx: {self.config.position_idx})..."
        )
        # Initialize structure to return: keys 'long', 'short' with empty dicts if no position
        positions_summary: Dict[str, Dict[str, Any]] = {"long": {}, "short": {}}

        try:
            # Bybit V5 fetch_positions requires 'category' and optionally 'symbol' in params for filtering
            params = {
                "category": self.config.bybit_v5_category,
                "symbol": market_id, # Filter by symbol on the server-side
                # "positionIdx": self.config.position_idx, # Not directly used by fetch_positions, filtering is client-side on response
            }
            # CCXT fetch_positions might take a list of symbols or be general
            # Providing the symbol in `symbols` list for CCXT's `fetch_positions`
            fetched_positions_list = fetch_with_retries(
                self.exchange.fetch_positions,
                symbols=[self.config.symbol], # CCXT standard way to request for a specific symbol
                params=params, # Additional params for Bybit V5
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
            )
            # logger.debug(f"Raw position data list from exchange: {fetched_positions_list}") # Verbose

            if not fetched_positions_list: # API returned an empty list
                logger.debug("No position data returned from fetch_positions (empty list). Assuming flat.")
                return positions_summary # Return empty summary, means flat

            # --- Find the specific position matching symbol AND configured positionIdx from the list ---
            target_pos_info = None
            for pos_data_ccxt_unified in fetched_positions_list:
                # `pos_data_ccxt_unified` is CCXT's unified structure.
                # We need to look into `info` for Bybit-specific fields like `positionIdx`.
                raw_pos_info = pos_data_ccxt_unified.get("info", {}) # Bybit's raw position data
                pos_symbol_api = raw_pos_info.get("symbol") # Symbol from API response
                pos_idx_api_str = raw_pos_info.get("positionIdx")

                try: # Convert positionIdx from API (string) to int for comparison
                    pos_idx_api = int(pos_idx_api_str) if pos_idx_api_str is not None else -1 # Use -1 if not present
                except ValueError:
                    logger.warning(f"Could not parse positionIdx '{pos_idx_api_str}' from API for a position. Skipping this entry.")
                    continue # Skip this position entry if index is invalid

                # Match symbol (exchange ID) and the configured position index
                symbol_matches = (pos_symbol_api == market_id)
                index_matches = (pos_idx_api == self.config.position_idx)

                if symbol_matches and index_matches:
                    target_pos_info = raw_pos_info # Found the position matching our config
                    logger.debug(f"Found matching position info in list: Symbol={pos_symbol_api}, Idx={pos_idx_api}")
                    break # Stop searching once found

            if not target_pos_info: # No position in the list matched our symbol and positionIdx
                logger.debug(
                    f"No position found matching symbol {market_id} and positionIdx {self.config.position_idx} in the returned data. Assuming flat for this configuration."
                )
                return positions_summary # Return empty summary

            # --- Parse details from the found target_pos_info (Bybit's raw data) ---
            qty = safe_decimal(target_pos_info.get("size", "0"))
            # API 'side' can be 'Buy', 'Sell', or 'None'. 'None' typically means flat for that idx,
            # but size should be the definitive factor.
            api_side_str = target_pos_info.get("side", "None").lower()
            entry_price = safe_decimal(target_pos_info.get("avgPrice", "0"))
            liq_price = safe_decimal(target_pos_info.get("liqPrice", "0")) # Use safe_decimal
            unrealized_pnl = safe_decimal(target_pos_info.get("unrealisedPnl", "0"))
            # Stop Loss / Take Profit / Trailing Stop from position data
            sl_price_api = safe_decimal(target_pos_info.get("stopLoss", "0"))
            tp_price_api = safe_decimal(target_pos_info.get("takeProfit", "0"))
            tsl_trigger_price_api = safe_decimal(target_pos_info.get("trailingStop", "0")) # This is the TSL distance if active, or trigger price for some APIs

            # --- Validate and clean up parsed values ---
            qty_abs = qty.copy_abs() if not qty.is_nan() else Decimal("0")
            is_position_open = qty_abs >= POSITION_QTY_EPSILON # Check if quantity is significant

            if not is_position_open:
                logger.debug(f"Position size {qty_abs.normalize()} is negligible or zero. Considered flat.")
                return positions_summary # Return empty summary

            # Determine effective position side ('long' or 'short')
            # For Bybit V5 with positionIdx:
            # - If positionIdx=0 (One-Way): API 'side' ('Buy'/'Sell') dictates long/short.
            # - If positionIdx=1 (Hedge Buy): This is inherently a long position. API 'side' should be 'Buy' or 'None'.
            # - If positionIdx=2 (Hedge Sell): This is inherently a short position. API 'side' should be 'Sell' or 'None'.
            position_side_key: Optional[str] = None
            if self.config.position_idx == 0: # One-Way Mode
                if api_side_str == "buy": position_side_key = "long"
                elif api_side_str == "sell": position_side_key = "short"
            elif self.config.position_idx == 1: # Hedge Mode Buy Side
                position_side_key = "long" # If size > 0, it must be long
            elif self.config.position_idx == 2: # Hedge Mode Sell Side
                position_side_key = "short" # If size > 0, it must be short

            if position_side_key:
                position_details = {
                    "qty": qty_abs, # Store absolute quantity
                    "entry_price": entry_price if not entry_price.is_nan() and entry_price > 0 else Decimal("NaN"),
                    "liq_price": liq_price if not liq_price.is_nan() and liq_price > 0 else Decimal("NaN"),
                    "unrealized_pnl": unrealized_pnl if not unrealized_pnl.is_nan() else Decimal("0"),
                    "api_side": api_side_str, # Store original API side ('buy'/'sell'/'None')
                    "info": target_pos_info, # Store raw info for debugging or further use
                    "stop_loss_price": sl_price_api if not sl_price_api.is_nan() and sl_price_api > 0 else None,
                    "take_profit_price": tp_price_api if not tp_price_api.is_nan() and tp_price_api > 0 else None,
                    "is_tsl_active": not tsl_trigger_price_api.is_nan() and tsl_trigger_price_api > 0,
                    "tsl_trigger_price": tsl_trigger_price_api if not tsl_trigger_price_api.is_nan() and tsl_trigger_price_api > 0 else None,
                }
                positions_summary[position_side_key] = position_details
                entry_str = position_details["entry_price"].normalize() if not position_details["entry_price"].is_nan() else "N/A"
                logger.debug(
                    f"Identified {position_side_key.upper()} position: Qty={qty_abs.normalize()}, Entry={entry_str}"
                )
            else: # Should not happen if position_idx is valid and qty > 0
                 logger.warning(f"Position found with size {qty_abs.normalize()} but could not determine long/short state reliably (api_side: '{api_side_str}', positionIdx: {self.config.position_idx}). Treating as flat for safety.")
                 return positions_summary # Return empty

            return positions_summary

        except Exception as e: # Catch any error during position fetching or parsing
            logger.error(
                f"Failed to fetch or parse positions for {self.config.symbol}: {e}", exc_info=True
            )
            return None # Indicate error in fetching/processing


# --- Indicator Calculator Class ---
class IndicatorCalculator:
    """Calculates technical indicators (EMAs, Stochastic, ATR, ADX) needed for the trading strategy."""

    def __init__(self, config: TradingConfig):
        self.config = config

    # --- Enhanced calculate_indicators (integrated from Snippet 3, with stoch_k_prev added) ---
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

        required_ohlc_cols = ["open", "high", "low", "close"] # Volume not strictly needed for these indicators
        if not all(c in df.columns for c in required_ohlc_cols):
            missing_cols = [c for c in required_ohlc_cols if c not in df.columns]
            logger.error(f"{Fore.RED}DataFrame missing required columns for indicators: {missing_cols}{Style.RESET_ALL}")
            return None

        try:
            # Work with a copy, select only necessary columns to avoid modifying original DataFrame
            # Ensure input df has Decimal types for OHLC from fetch_ohlcv
            df_calc = df[required_ohlc_cols].copy()

            # Robust conversion to numeric (float) for TA calculations, handling Decimal, strings, None, etc.
            for col in required_ohlc_cols:
                if df_calc[col].empty: # Handle empty series case
                    logger.warning(f"Column '{col}' is empty before conversion. Ensuring float type.")
                    df_calc[col] = pd.Series(dtype=float) # Create empty series of float type
                    continue

                # Define a safe conversion function to float
                def safe_to_float(x: Any) -> float:
                    if isinstance(x, (float, int)): return float(x)
                    if isinstance(x, Decimal): return float('nan') if x.is_nan() else float(x)
                    if isinstance(x, str):
                        try: return float(x.strip()) # Try direct float conversion after strip
                        except ValueError: # Handle common non-numeric strings
                            if x.strip().lower() in ["nan", "none", "null", ""]: return float('nan')
                            logger.debug(f"Could not convert string '{x}' in column '{col}' to float for TA calculation.")
                            return float('nan')
                    if x is None: return float('nan')
                    # For other unexpected types, log and return NaN
                    logger.warning(f"Unexpected type {type(x)} ('{x}') in column '{col}', converting to NaN for TA calculation.")
                    return float('nan')

                df_calc[col] = df_calc[col].map(safe_to_float) # Apply safe conversion
                df_calc[col] = df_calc[col].astype(float) # Ensure final dtype is float

            # Drop rows with NaN in essential OHLC columns *after* float conversion
            initial_len = len(df_calc)
            df_calc.dropna(subset=required_ohlc_cols, inplace=True, how='any')
            rows_dropped = initial_len - len(df_calc)
            if rows_dropped > 0:
                 logger.debug(f"Dropped {rows_dropped} rows with NaN in OHLC columns after float conversion for TA.")

            if df_calc.empty:
                logger.error(f"{Fore.RED}DataFrame became empty after NaN drop during indicator pre-processing.{Style.RESET_ALL}")
                return None

            # --- Check Data Length: Ensure enough data for the longest indicator period ---
            # Max period considers lookbacks for smoothing (e.g., Stoch K, D, ADX smoothing)
            max_period_needed = max(
                self.config.slow_ema_period, self.config.trend_ema_period,
                self.config.stoch_period + self.config.stoch_smooth_k + self.config.stoch_smooth_d, # Sum for full Stoch formation
                self.config.atr_period,
                self.config.adx_period * 2, # ADX often needs more data for stable calculation (DI smoothing + DX smoothing)
            )
            min_required_data_length = max_period_needed + 20 # Add a buffer for stability and to get previous values
            if len(df_calc) < min_required_data_length:
                logger.error(f"{Fore.RED}Insufficient data ({len(df_calc)} rows) for robust indicator calculation (requires ~{min_required_data_length} rows).{Style.RESET_ALL}")
                return None

            # --- Indicator Calculations (using df_calc which now has float types) ---
            close_s = df_calc["close"] # pd.Series
            high_s = df_calc["high"]   # pd.Series
            low_s = df_calc["low"]     # pd.Series

            # Exponential Moving Averages (EMAs)
            fast_ema_s = close_s.ewm(span=self.config.fast_ema_period, adjust=False).mean()
            slow_ema_s = close_s.ewm(span=self.config.slow_ema_period, adjust=False).mean()
            trend_ema_s = close_s.ewm(span=self.config.trend_ema_period, adjust=False).mean()

            # Stochastic Oscillator (%K, %D)
            low_min_stoch = low_s.rolling(window=self.config.stoch_period).min()
            high_max_stoch = high_s.rolling(window=self.config.stoch_period).max()
            stoch_range = high_max_stoch - low_min_stoch
            # Calculate %K raw, handle division by zero by setting to 50 (neutral)
            stoch_k_raw_values = np.where(stoch_range > 1e-12, # Avoid division by zero or tiny range
                                      100 * (close_s - low_min_stoch) / stoch_range,
                                      50.0) # Default to 50 if range is zero/too small
            stoch_k_raw_s = pd.Series(stoch_k_raw_values, index=df_calc.index).fillna(50) # Fill NaNs from rolling
            stoch_k_s = stoch_k_raw_s.rolling(window=self.config.stoch_smooth_k).mean().fillna(50)
            stoch_d_s = stoch_k_s.rolling(window=self.config.stoch_smooth_d).mean().fillna(50)

            # Average True Range (ATR)
            prev_close_s = close_s.shift(1)
            tr1 = high_s - low_s
            tr2 = (high_s - prev_close_s).abs()
            tr3 = (low_s - prev_close_s).abs()
            true_range_s = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(0) # Max of the three, fill initial NaN
            atr_s = true_range_s.ewm(alpha=1 / self.config.atr_period, adjust=False).mean() # Wilder's ATR

            # ADX, +DI, -DI
            adx_s, pdi_s, mdi_s = self._calculate_adx(
                high_s, low_s, close_s, atr_s, self.config.adx_period
            )

            # --- Extract Latest Values & Convert back to Decimal for precision in strategy logic ---
            def get_latest_decimal_from_series(series: pd.Series, indicator_name: str) -> Decimal:
                """Safely gets the last valid (non-NaN) value from a Pandas Series and converts to Decimal."""
                valid_series = series.dropna()
                if valid_series.empty: return Decimal("NaN") # No valid data points
                last_valid_float = valid_series.iloc[-1]
                try:
                    # Convert float to string first for precise Decimal conversion
                    return Decimal(str(last_valid_float))
                except (InvalidOperation, TypeError, ValueError):
                    logger.error(f"Failed to convert latest {indicator_name} value '{last_valid_float}' (type: {type(last_valid_float).__name__}) to Decimal.")
                    return Decimal("NaN")

            indicators_out: Dict[str, Union[Decimal, bool, int]] = {
                "fast_ema": get_latest_decimal_from_series(fast_ema_s, "fast_ema"),
                "slow_ema": get_latest_decimal_from_series(slow_ema_s, "slow_ema"),
                "trend_ema": get_latest_decimal_from_series(trend_ema_s, "trend_ema"),
                "stoch_k": get_latest_decimal_from_series(stoch_k_s, "stoch_k"),
                "stoch_d": get_latest_decimal_from_series(stoch_d_s, "stoch_d"),
                "atr": get_latest_decimal_from_series(atr_s, "atr"),
                "atr_period": self.config.atr_period, # Store period used for ATR, for context
                "adx": get_latest_decimal_from_series(adx_s, "adx"),
                "pdi": get_latest_decimal_from_series(pdi_s, "pdi"), # Positive Directional Indicator
                "mdi": get_latest_decimal_from_series(mdi_s, "mdi"), # Negative Directional Indicator
            }

            # --- Calculate Previous Stochastic K (%K) for Exit Logic ---
            stoch_k_valid_series = stoch_k_s.dropna()
            stoch_k_prev_val = Decimal("NaN")
            if len(stoch_k_valid_series) >= 2: # Need at least two valid points to get a "previous"
                stoch_k_prev_val = safe_decimal(str(stoch_k_valid_series.iloc[-2])) # Second to last
            indicators_out["stoch_k_prev"] = stoch_k_prev_val

            # --- Calculate Stochastic K/D Cross Signals (using latest and previous values) ---
            k_now = indicators_out["stoch_k"]
            d_now = indicators_out["stoch_d"]
            k_prev = indicators_out["stoch_k_prev"] # This is K at t-2, D at t-1 uses K at t-1
            # For K/D cross, we need D previous. D previous depends on K previous.
            stoch_d_valid_series = stoch_d_s.dropna()
            d_prev_val = Decimal("NaN")
            if len(stoch_d_valid_series) >=2:
                d_prev_val = safe_decimal(str(stoch_d_valid_series.iloc[-2]))


            stoch_kd_bullish_cross = False
            stoch_kd_bearish_cross = False
            if not any(v.is_nan() for v in [k_now, d_now, k_prev, d_prev_val]): # Ensure all needed values are valid
                # Bullish: K crosses above D from below
                crossed_above = (k_prev <= d_prev_val) and (k_now > d_now)
                # Bearish: K crosses below D from above
                crossed_below = (k_prev >= d_prev_val) and (k_now < d_now)

                # Confirm cross happens in oversold/overbought zones for stronger signal
                # (Using k_prev and d_prev_val to check zone status *before* the cross)
                # Or simply use current K being in the zone after cross. Strategy choice.
                # Here, we check if K crossed D *and* the cross originated from or moved into OB/OS.
                # A common interpretation: cross + K now in OB/OS (or k_prev was).
                # For this example, we'll use a simpler K/D cross logic for the boolean flag.
                # The signal generator can then combine this with OB/OS levels.
                # For this implementation: a cross is bullish if K(t-1) <= D(t-1) AND K(t) > D(t)
                # To be more precise with actual Stoch K/D cross:
                # Bullish cross: K_prev <= D_prev AND K_now > D_now
                # We need K_prev (which is stoch_k_s.iloc[-2]) and D_prev (stoch_d_s.iloc[-2])
                # The `stoch_k_prev` in `indicators_out` is K at t-1 relative to K_now at t.
                # So, `k_prev` is K(t-1), `d_prev_val` is D(t-1). `k_now` is K(t), `d_now` is D(t).
                if (k_prev <= d_prev_val) and (k_now > d_now): stoch_kd_bullish_cross = True
                if (k_prev >= d_prev_val) and (k_now < d_now): stoch_kd_bearish_cross = True


            indicators_out["stoch_kd_bullish"] = stoch_kd_bullish_cross
            indicators_out["stoch_kd_bearish"] = stoch_kd_bearish_cross

            # Final check for critical NaNs in output that would break strategy logic
            critical_indicator_keys = [
                "fast_ema", "slow_ema", "trend_ema", "atr",
                "stoch_k", "stoch_d", "stoch_k_prev",
                "adx", "pdi", "mdi",
            ]
            failed_indicators = [
                k for k in critical_indicator_keys if indicators_out.get(k, Decimal("NaN")).is_nan()
            ]
            if failed_indicators:
                logger.error(
                    f"{Fore.RED}Critical indicators calculated as NaN: {', '.join(failed_indicators)}. This may prevent or impair signal generation.{Style.RESET_ALL}"
                )
                # ATR is especially critical for risk calculations
                if indicators_out.get("atr", Decimal("NaN")).is_nan():
                     logger.error(f"{Fore.RED}ATR is NaN. Risk calculations will fail. Aborting indicator calculation result.{Style.RESET_ALL}")
                     return None # Cannot proceed without ATR

            logger.info(f"{Style.BRIGHT}{Fore.GREEN}Indicator patterns woven successfully.{Style.RESET_ALL}")
            return indicators_out

        except Exception as e:
            logger.error(f"{Fore.RED}Error weaving indicator patterns: {e}{Style.RESET_ALL}", exc_info=True)
            return None
    # --- End Enhanced calculate_indicators ---

    def _calculate_adx(
        self,
        high_s: pd.Series, low_s: pd.Series, close_s: pd.Series, # Close is for context, not directly used here but often part of ADX inputs
        atr_s: pd.Series, # Pre-calculated ATR series (Wilder's smoothed)
        period: int,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]: # Returns (ADX, +DI, -DI)
        """Helper to calculate ADX, +DI, -DI using Wilder's smoothing (EMA)."""
        if period <= 0:
            raise ValueError("ADX period must be a positive integer.")
        if atr_s.empty or atr_s.isnull().all(): # Check if ATR series is unusable
             logger.error("ATR series is empty or all NaN. Cannot calculate ADX components.")
             nan_series = pd.Series(np.nan, index=high_s.index) # Create NaN series of same length
             return nan_series, nan_series, nan_series

        # Calculate Directional Movement (+DM, -DM)
        move_up = high_s.diff() # Current high - previous high
        move_down = -low_s.diff() # Previous low - current low (positive if current low is lower)

        # +DM is UpMove if UpMove > DownMove and UpMove > 0, else 0
        plus_dm_values = np.where((move_up > move_down) & (move_up > 0), move_up, 0.0)
        # -DM is DownMove if DownMove > UpMove and DownMove > 0, else 0
        minus_dm_values = np.where((move_down > move_up) & (move_down > 0), move_down, 0.0)

        # Convert to Series for EWM
        plus_dm_s = pd.Series(plus_dm_values, index=high_s.index)
        minus_dm_s = pd.Series(minus_dm_values, index=high_s.index)

        # Smoothed +DM, -DM using Wilder's method (equivalent to EMA with alpha = 1/period)
        alpha = 1.0 / period
        smoothed_plus_dm_s = plus_dm_s.ewm(alpha=alpha, adjust=False).mean().fillna(0)
        smoothed_minus_dm_s = minus_dm_s.ewm(alpha=alpha, adjust=False).mean().fillna(0)

        # Calculate Directional Indicators (+DI, -DI)
        # DI = 100 * (Smoothed DM / ATR)
        # Handle ATR being zero or NaN to prevent division errors
        pdi_values = np.where((atr_s > 1e-12) & (~atr_s.isnull()), 100 * smoothed_plus_dm_s / atr_s, 0.0)
        mdi_values = np.where((atr_s > 1e-12) & (~atr_s.isnull()), 100 * smoothed_minus_dm_s / atr_s, 0.0)
        pdi_s_out = pd.Series(pdi_values, index=high_s.index).fillna(0)
        mdi_s_out = pd.Series(mdi_values, index=high_s.index).fillna(0)

        # Calculate Directional Movement Index (DX)
        # DX = 100 * (abs(+DI - -DI) / (+DI + -DI))
        di_diff_abs = (pdi_s_out - mdi_s_out).abs()
        di_sum = pdi_s_out + mdi_s_out
        dx_values = np.where(di_sum > 1e-12, 100 * di_diff_abs / di_sum, 0.0) # Handle sum being zero
        dx_s = pd.Series(dx_values, index=high_s.index).fillna(0)

        # Calculate Average Directional Index (ADX - Smoothed DX using Wilder's EMA)
        adx_s_out = dx_s.ewm(alpha=alpha, adjust=False).mean().fillna(0)

        return adx_s_out, pdi_s_out, mdi_s_out


# --- Signal Generator Class ---
class SignalGenerator:
    """Generates trading entry and exit signals based on indicator conditions."""

    def __init__(self, config: TradingConfig):
        self.config = config

    def generate_signals(
        self,
        df_last_candles: pd.DataFrame, # Expecting DataFrame with at least last 2 candles
        indicators: Dict[str, Union[Decimal, bool, int]],
    ) -> Dict[str, Union[bool, str]]: # Returns {'long': bool, 'short': bool, 'reason': str}
        """Generates 'long'/'short' entry signals and provides a detailed reason string."""
        # Initialize default result
        result: Dict[str, Union[bool, str]] = {
            "long": False, "short": False, "reason": "Initializing signal check",
        }

        # --- Pre-checks for data validity ---
        if not indicators:
            result["reason"] = "No Signal: Indicators data missing."
            logger.debug(result["reason"])
            return result
        if df_last_candles is None or len(df_last_candles) < 2: # Need at least 2 candles for prev_close
            reason = f"No Signal: Insufficient candle data (requires >=2, got {len(df_last_candles) if df_last_candles is not None else 0})."
            result["reason"] = reason
            logger.debug(reason)
            return result

        try:
            # --- Extract Current Price & Previous Close Safely ---
            # Assuming df_last_candles contains the most recent candles, last row is current forming/closed.
            latest_candle = df_last_candles.iloc[-1]
            prev_candle = df_last_candles.iloc[-2]
            current_price = safe_decimal(latest_candle["close"])
            prev_close = safe_decimal(prev_candle["close"])

            if current_price.is_nan() or current_price <= 0:
                result["reason"] = f"No Signal: Invalid current price ({current_price.normalize() if not current_price.is_nan() else 'NaN'})."
                logger.warning(result["reason"])
                return result

            # --- Extract Indicator Values Safely ---
            required_indicator_keys = [
                "stoch_k", "fast_ema", "slow_ema", "trend_ema", "atr", "adx", "pdi", "mdi"
            ]
            ind_values: Dict[str, Decimal] = {} # Store valid Decimal indicators
            nan_keys = []
            for key in required_indicator_keys:
                val = indicators.get(key)
                if not isinstance(val, Decimal) or val.is_nan(): # Check if not Decimal or is NaN Decimal
                    nan_keys.append(key)
                else:
                    ind_values[key] = val

            if nan_keys: # If any required indicator is missing or NaN
                result["reason"] = f"No Signal: Required indicator(s) are NaN/Missing: {', '.join(nan_keys)}."
                logger.warning(result["reason"])
                return result

            # Unpack validated indicators
            k, fast_ema, slow_ema, trend_ema, atr, adx, pdi, mdi = (
                ind_values["stoch_k"], ind_values["fast_ema"], ind_values["slow_ema"],
                ind_values["trend_ema"], ind_values["atr"], ind_values["adx"],
                ind_values["pdi"], ind_values["mdi"]
            )
            # Get K/D cross signals (booleans)
            stoch_kd_bull_cross = indicators.get("stoch_kd_bullish", False)
            stoch_kd_bear_cross = indicators.get("stoch_kd_bearish", False)

            # --- Define Trading Conditions ---
            # EMA Cross
            ema_bullish_cross = fast_ema > slow_ema
            ema_bearish_cross = fast_ema < slow_ema
            ema_cross_state = "Bullish" if ema_bullish_cross else "Bearish" if ema_bearish_cross else "Neutral"

            # Trend Filter (Price vs Trend EMA with buffer)
            trend_buffer_abs = trend_ema.copy_abs() * (self.config.trend_filter_buffer_percent / 100)
            price_above_trend_ema_with_buffer = current_price > (trend_ema - trend_buffer_abs)
            price_below_trend_ema_with_buffer = current_price < (trend_ema + trend_buffer_abs)
            trend_allows_long = price_above_trend_ema_with_buffer if self.config.trade_only_with_trend else True
            trend_allows_short = price_below_trend_ema_with_buffer if self.config.trade_only_with_trend else True
            trend_reason_suffix = f"(P:{current_price:.{DEFAULT_PRICE_DP}f} vs TrendEMA:{trend_ema:.{DEFAULT_PRICE_DP}f} ±{trend_buffer_abs:.{DEFAULT_PRICE_DP}f})" if self.config.trade_only_with_trend else "(TrendFilter OFF)"

            # Stochastic Condition (K in zone OR K/D cross in respective direction)
            stoch_long_entry_cond = (k < self.config.stoch_oversold_threshold) or stoch_kd_bull_cross
            stoch_short_entry_cond = (k > self.config.stoch_overbought_threshold) or stoch_kd_bear_cross
            stoch_state_reason = f"K:{k:.1f} (Oversold:{self.config.stoch_oversold_threshold.normalize()}/Overbought:{self.config.stoch_overbought_threshold.normalize()}) KD_Cross(Bull:{stoch_kd_bull_cross}/Bear:{stoch_kd_bear_cross})"

            # ATR Move Filter (Price move > ATR threshold)
            significant_price_move = True # Default to true if filter is off or ATR invalid
            atr_filter_reason_suffix = "(ATR Filter OFF)"
            if self.config.atr_move_filter_multiplier > 0:
                if atr.is_nan() or atr <= 0:
                    atr_filter_reason_suffix = f"(ATR Filter Skipped: Invalid ATR value {atr.normalize()})"
                    significant_price_move = False # Cannot evaluate, treat as not significant
                elif prev_close.is_nan():
                    atr_filter_reason_suffix = "(ATR Filter Skipped: Previous close price NaN)"
                    significant_price_move = False
                else:
                    atr_move_threshold_abs = atr * self.config.atr_move_filter_multiplier
                    price_move_abs = (current_price - prev_close).copy_abs()
                    significant_price_move = price_move_abs > atr_move_threshold_abs
                    atr_filter_reason_suffix = f"(Move:{price_move_abs:.{DEFAULT_PRICE_DP}f} {'OK' if significant_price_move else 'LOW'} vs Threshold:{atr_move_threshold_abs:.{DEFAULT_PRICE_DP}f})"

            # ADX Trend Strength and Direction Filter
            adx_is_trending_strong = adx > self.config.min_adx_level
            adx_long_direction_favored = pdi > mdi # +DI above -DI
            adx_short_direction_favored = mdi > pdi # -DI above +DI
            adx_allows_long = adx_is_trending_strong and adx_long_direction_favored
            adx_allows_short = adx_is_trending_strong and adx_short_direction_favored
            adx_filter_reason_suffix = f"(ADX:{adx:.1f} {'Trending' if adx_is_trending_strong else 'Weak'} vs Min:{self.config.min_adx_level.normalize()} | Dir: {'+DI > -DI' if adx_long_direction_favored else '-DI > +DI' if adx_short_direction_favored else 'Neutral'})"

            # --- Combine Conditions for Final Entry Signals ---
            # Base signal: EMA cross + Stochastic confirmation
            base_long_signal_met = ema_bullish_cross and stoch_long_entry_cond
            base_short_signal_met = ema_bearish_cross and stoch_short_entry_cond

            # Final signal: Base signal + All filters pass
            final_long_signal = base_long_signal_met and trend_allows_long and significant_price_move and adx_allows_long
            final_short_signal = base_short_signal_met and trend_allows_short and significant_price_move and adx_allows_short

            # --- Build Detailed Reason String for Logging/Display ---
            if final_long_signal:
                result["long"] = True
                result["reason"] = f"Long Signal: EMA {ema_cross_state} & Stoch OK {stoch_state_reason} & Trend OK {trend_reason_suffix} & ATR OK {atr_filter_reason_suffix} & ADX OK {adx_filter_reason_suffix}"
            elif final_short_signal:
                result["short"] = True
                result["reason"] = f"Short Signal: EMA {ema_cross_state} & Stoch OK {stoch_state_reason} & Trend OK {trend_reason_suffix} & ATR OK {atr_filter_reason_suffix} & ADX OK {adx_filter_reason_suffix}"
            else: # No final signal, provide reason why
                reason_parts = ["No Signal:"]
                if not base_long_signal_met and not base_short_signal_met: # Neither base signal met
                     reason_parts.append(f"Base (EMA {ema_cross_state} or Stoch {stoch_state_reason}) not met.")
                elif base_long_signal_met: # Base long met, but a filter blocked it
                    if not trend_allows_long: reason_parts.append(f"Long Blocked by Trend Filter {trend_reason_suffix}.")
                    elif not significant_price_move: reason_parts.append(f"Long Blocked by ATR Filter {atr_filter_reason_suffix}.")
                    elif not adx_allows_long: reason_parts.append(f"Long Blocked by ADX Filter {adx_filter_reason_suffix}.")
                    else: reason_parts.append("Long filters passed but final logic error.") # Fallback
                elif base_short_signal_met: # Base short met, but a filter blocked it
                    if not trend_allows_short: reason_parts.append(f"Short Blocked by Trend Filter {trend_reason_suffix}.")
                    elif not significant_price_move: reason_parts.append(f"Short Blocked by ATR Filter {atr_filter_reason_suffix}.")
                    elif not adx_allows_short: reason_parts.append(f"Short Blocked by ADX Filter {adx_filter_reason_suffix}.")
                    else: reason_parts.append("Short filters passed but final logic error.") # Fallback
                else: # General "conditions unmet"
                    reason_parts.append(f"Conditions unmet (EMA:{ema_cross_state},Stoch:{stoch_state_reason},Trend:{trend_reason_suffix},ATR:{atr_filter_reason_suffix},ADX:{adx_filter_reason_suffix})")
                result["reason"] = " ".join(reason_parts)

            # Log signal check result (INFO for actual signals or blocks, DEBUG for no signal)
            log_level_for_signal = logging.INFO if result["long"] or result["short"] or "Blocked" in result["reason"] else logging.DEBUG
            logger.log(log_level_for_signal, f"Signal Check Result: {result['reason']}")

        except Exception as e: # Catch any unexpected error during signal generation
            logger.error(f"{Fore.RED}Error generating entry signals: {e}{Style.RESET_ALL}", exc_info=True)
            result["reason"] = f"No Signal: Exception during generation ({type(e).__name__})"
            result["long"] = False; result["short"] = False # Ensure signals are false on error
        return result

    # --- Updated check_exit_signals method (from v4.5.7 exit logic) ---
    def check_exit_signals(
        self,
        position_side: str, # 'long' or 'short'
        indicators: Dict[str, Union[Decimal, bool, int]], # Expects full indicator dict
    ) -> Optional[str]: # Returns reason string for exit, or None if no exit signal
        """
        Checks for signal-based exits:
        1. EMA Cross against the position.
        2. Stochastic Reversal Confirmation: %K crossing back from Overbought/Oversold.
        Returns an exit reason string if conditions met, otherwise None.
        """
        if not indicators:
            logger.warning("Cannot check exit signals: indicators data missing.")
            return None

        # --- Safely Extract Required Indicator Values ---
        fast_ema = indicators.get("fast_ema")
        slow_ema = indicators.get("slow_ema")
        stoch_k_current = indicators.get("stoch_k")
        stoch_k_previous = indicators.get("stoch_k_prev") # %K from the bar before current %K

        # Validate that all necessary indicators are available and are Decimals
        required_for_exit = {
            "fast_ema": fast_ema, "slow_ema": slow_ema,
            "stoch_k": stoch_k_current, "stoch_k_prev": stoch_k_previous
        }
        for name, val in required_for_exit.items():
            if not isinstance(val, Decimal) or val.is_nan():
                logger.warning(
                    f"Cannot check exit signals: Required indicator '{name}' is missing, not Decimal, or NaN (value: {val})."
                )
                return None # Cannot proceed without valid indicator values

        # --- Determine EMA Cross State ---
        # These are asserted as non-NaN Decimals by checks above
        ema_is_bullish_crossed: bool = fast_ema > slow_ema # type: ignore
        ema_is_bearish_crossed: bool = fast_ema < slow_ema # type: ignore

        exit_reason: Optional[str] = None

        # --- Get Stochastic OB/OS Thresholds from Config ---
        oversold_level = self.config.stoch_oversold_threshold
        overbought_level = self.config.stoch_overbought_threshold

        # --- Evaluate Exit Conditions based on Current Position Side ---
        if position_side == "long":
            # Priority 1: EMA Bearish Cross (Fast EMA crosses below Slow EMA)
            if ema_is_bearish_crossed:
                exit_reason = f"Exit Signal (Long): EMA Bearish Cross (Fast {fast_ema.normalize()} < Slow {slow_ema.normalize()})"
                logger.trade(f"{Fore.YELLOW}{exit_reason}{Style.RESET_ALL}")

            # Priority 2: Stochastic Reversal Confirmation from Overbought
            # (Stoch %K was >= Overbought and now has crossed below Overbought level)
            # Asserting stoch_k_previous and stoch_k_current are valid Decimals
            elif stoch_k_previous >= overbought_level and stoch_k_current < overbought_level: # type: ignore
                exit_reason = (
                    f"Exit Signal (Long): Stoch Reversal from Overbought "
                    f"(Prev K {stoch_k_previous.normalize():.1f} -> Curr K {stoch_k_current.normalize():.1f} crossed below {overbought_level.normalize()})"
                )
                logger.trade(f"{Fore.YELLOW}{exit_reason}{Style.RESET_ALL}")
            # Informational: Stoch is in overbought but hasn't confirmed reversal yet
            elif stoch_k_current >= overbought_level: # type: ignore
                logger.debug(f"Exit Check (Long): Stoch K ({stoch_k_current.normalize():.1f}) is at/above Overbought ({overbought_level.normalize()}), awaiting bearish cross for potential exit signal.")

        elif position_side == "short":
            # Priority 1: EMA Bullish Cross (Fast EMA crosses above Slow EMA)
            if ema_is_bullish_crossed:
                exit_reason = f"Exit Signal (Short): EMA Bullish Cross (Fast {fast_ema.normalize()} > Slow {slow_ema.normalize()})"
                logger.trade(f"{Fore.YELLOW}{exit_reason}{Style.RESET_ALL}")

            # Priority 2: Stochastic Reversal Confirmation from Oversold
            # (Stoch %K was <= Oversold and now has crossed above Oversold level)
            elif stoch_k_previous <= oversold_level and stoch_k_current > oversold_level: # type: ignore
                exit_reason = (
                    f"Exit Signal (Short): Stoch Reversal from Oversold "
                    f"(Prev K {stoch_k_previous.normalize():.1f} -> Curr K {stoch_k_current.normalize():.1f} crossed above {oversold_level.normalize()})"
                )
                logger.trade(f"{Fore.YELLOW}{exit_reason}{Style.RESET_ALL}")
            # Informational: Stoch is in oversold but hasn't confirmed reversal yet
            elif stoch_k_current <= oversold_level: # type: ignore
                logger.debug(f"Exit Check (Short): Stoch K ({stoch_k_current.normalize():.1f}) is at/below Oversold ({oversold_level.normalize()}), awaiting bullish cross for potential exit signal.")

        return exit_reason # Returns the reason string if an exit condition was met, otherwise None
    # --- End updated check_exit_signals ---


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
        if not exchange_manager or not exchange_manager.exchange: # Critical dependency
            # This error should ideally be caught earlier, but as a safeguard:
            logger.critical(f"{Style.BRIGHT}{Fore.RED}OrderManager cannot initialize: Valid ExchangeManager instance with initialized exchange is required.{Style.RESET_ALL}")
            raise ValueError("OrderManager requires a valid ExchangeManager with an initialized CCXT exchange instance.")
        self.exchange = exchange_manager.exchange # Convenience accessor for the CCXT exchange object
        self.market_info = exchange_manager.market_info # Convenience accessor for market details

        # Tracks active protection STATUS (e.g., "ACTIVE_SLTP", "ACTIVE_TSL") for V5 position stops.
        # Keys are 'long', 'short'. Value is None if no protection is thought to be active.
        self.protection_tracker: Dict[str, Optional[str]] = {"long": None, "short": None}

    def _calculate_trade_parameters(
        self,
        side: str, # 'buy' or 'sell' for the entry order
        atr: Decimal,
        total_equity: Decimal, # In settlement currency
        current_price: Decimal,
    ) -> Optional[Dict[str, Optional[Decimal]]]: # Returns dict with qty, sl_price, tp_price, tsl_distance
        """Calculates SL price, TP price, order quantity, and TSL distance based on risk, ATR, and market info."""
        # --- Input Validation ---
        if atr.is_nan() or atr <= 0:
            logger.error(f"Invalid ATR ({atr.normalize() if not atr.is_nan() else 'NaN'}) for trade parameter calculation.")
            return None
        if total_equity.is_nan() or total_equity <= 0:
            logger.error(f"Invalid total equity ({total_equity.normalize() if not total_equity.is_nan() else 'NaN'}) for parameter calculation.")
            return None
        if current_price.is_nan() or current_price <= 0:
            logger.error(f"Invalid current price ({current_price.normalize() if not current_price.is_nan() else 'NaN'}) for parameter calculation.")
            return None
        if not self.market_info or 'tick_size' not in self.market_info or 'contract_size' not in self.market_info or 'min_order_size' not in self.market_info:
             logger.error("Market info (tick_size, contract_size, min_order_size) missing or incomplete for parameter calculation.")
             return None
        if side not in ["buy", "sell"]:
            logger.error(f"Invalid side '{side}' specified for trade parameter calculation.")
            return None

        try:
            risk_amount_per_trade = total_equity * self.config.risk_percentage # In settlement currency
            sl_distance_atr_points = atr * self.config.sl_atr_multiplier # Price distance for SL in quote currency points

            # Calculate initial Stop Loss Price
            sl_price_calculated: Decimal
            if side == "buy": sl_price_calculated = current_price - sl_distance_atr_points
            else: sl_price_calculated = current_price + sl_distance_atr_points

            if sl_price_calculated <= 0: # SL price cannot be zero or negative
                logger.error(f"Calculated SL price ({sl_price_calculated:.{DEFAULT_PRICE_DP}f}) is invalid (<=0). Cannot proceed.")
                return None

            # Ensure SL distance is at least one tick size
            sl_distance_from_current = (current_price - sl_price_calculated).copy_abs()
            min_tick_size = self.market_info['tick_size']
            if sl_distance_from_current < min_tick_size:
                logger.warning(f"Initial SL distance ({sl_distance_from_current.normalize()}) < min tick size ({min_tick_size.normalize()}). Adjusting SL distance to min tick size.")
                sl_distance_from_current = min_tick_size
                # Recalculate SL price based on adjusted distance
                if side == "buy": sl_price_calculated = current_price - sl_distance_from_current
                else: sl_price_calculated = current_price + sl_distance_from_current
                if sl_price_calculated <= 0: # Re-check after adjustment
                     logger.error(f"Adjusted SL price ({sl_price_calculated:.{DEFAULT_PRICE_DP}f}) is still invalid (<=0).")
                     return None

            if sl_distance_from_current <= 0: # Should be caught by sl_price <=0, but safeguard
                logger.error(f"Calculated SL distance ({sl_distance_from_current.normalize()}) is invalid (<=0).")
                return None

            # --- Quantity Calculation ---
            # `contract_size` from CCXT market info:
            #   - For linear (e.g., BTC/USDT): typically 1 (meaning 1 unit of base, e.g., 1 BTC).
            #   - For inverse (e.g., BTC/USD): typically the quote currency value of one contract (e.g., 1 USD).
            market_contract_size = self.market_info['contract_size'] # This is Decimal
            value_per_point_move: Decimal # Value of a 1-point price move per unit of quantity, in settlement currency

            if self.config.market_type == "inverse": # e.g., BTC/USD, settled in BTC
                # Quantity is in contracts (e.g., USD amount). Risk is in Base currency (BTC).
                # Value of 1 contract (e.g., 1 USD) changing by 1 point (1 USD)
                # PnL per contract = ContractSize_Quote * (1/Entry - 1/Exit)
                # Risk per contract (in Base) = ContractSize_Quote * |1/CurrentPrice - 1/SLPrice|
                # This can be approximated for small moves or calculated precisely.
                # Simpler: Qty_contracts = (Risk_Base * CurrentPrice_Quote) / (SL_distance_Quote * ContractSize_Quote)
                # Assuming ContractSize_Quote is 1 (e.g. 1 USD per contract for BTCUSD)
                # Qty_contracts (USD amount) = (Risk_BTC * Price_USD) / SL_distance_USD
                if market_contract_size <= 0: # Should be positive (e.g. 1 USD)
                    logger.error(f"Invalid market_contract_size ({market_contract_size}) for inverse quantity calculation.")
                    return None
                # `risk_amount_per_trade` is in base (BTC), `sl_distance_from_current` is in quote (USD)
                # `current_price` is in quote (USD)
                quantity_calculated = (risk_amount_per_trade * current_price) / (sl_distance_from_current * market_contract_size)
            else: # Linear/Swap (e.g., BTC/USDT, settled in USDT)
                # Quantity is in Base currency (e.g., BTC). Risk is in Quote currency (USDT).
                # Risk per unit of Base = SL_distance_Quote * ContractSize_Base (usually 1 for linear)
                # Qty_Base = Risk_Quote / (SL_distance_Quote * ContractSize_Base)
                value_per_point_move = market_contract_size # e.g., 1 USDT profit if price moves 1 USD for 1 BTC unit if contract_size=1
                if value_per_point_move <= 0:
                    logger.error(f"Invalid value_per_point_move ({value_per_point_move}) for linear quantity calculation.")
                    return None
                risk_per_unit_base = sl_distance_from_current * value_per_point_move
                if risk_per_unit_base <= 0:
                    logger.error(f"Calculated zero or negative risk per unit of base ({risk_per_unit_base.normalize()}). Cannot determine quantity.")
                    return None
                quantity_calculated = risk_amount_per_trade / risk_per_unit_base

            # Format quantity according to market precision (amount), rounding down
            quantity_str_formatted = self.exchange_manager.format_amount(quantity_calculated, rounding_mode=ROUND_DOWN)
            quantity_decimal_final = safe_decimal(quantity_str_formatted)

            if quantity_decimal_final.is_nan() or quantity_decimal_final <= 0:
                 logger.error(f"Calculated quantity ({quantity_str_formatted}) is invalid or zero after formatting. Original calc: {quantity_calculated.normalize()}")
                 return None

            # Check against minimum order size
            min_order_size_market = self.market_info.get('min_order_size', Decimal('NaN'))
            if not min_order_size_market.is_nan() and quantity_decimal_final < min_order_size_market:
                logger.error(f"Calculated quantity {quantity_decimal_final.normalize()} is less than market minimum order size {min_order_size_market.normalize()}.")
                return None

            # --- Take Profit Calculation (Optional) ---
            tp_price_calculated: Optional[Decimal] = None
            if self.config.tp_atr_multiplier > 0: # TP is enabled
                tp_distance_atr_points = atr * self.config.tp_atr_multiplier
                if side == "buy": tp_price_calculated = current_price + tp_distance_atr_points
                else: tp_price_calculated = current_price - tp_distance_atr_points
                if tp_price_calculated <= 0: # TP price invalid
                    logger.warning(f"Calculated TP price ({tp_price_calculated:.{DEFAULT_PRICE_DP}f}) is invalid (<=0). Disabling TP for this trade.")
                    tp_price_calculated = None # Effectively disable TP

            # --- Trailing Stop Calculation (Distance for V5 API) ---
            # TSL distance is a price difference, calculated as a percentage of current price
            tsl_distance_price_points = current_price * (self.config.trailing_stop_percent / 100)
            if tsl_distance_price_points < min_tick_size: # Ensure TSL distance is at least one tick
                 tsl_distance_price_points = min_tick_size
            # Format this distance like a price (it's a price difference)
            tsl_distance_str_formatted = self.exchange_manager.format_price(tsl_distance_price_points)
            tsl_distance_decimal_final = safe_decimal(tsl_distance_str_formatted)
            if tsl_distance_decimal_final.is_nan() or tsl_distance_decimal_final <= 0:
                 logger.warning(f"Calculated invalid TSL distance ({tsl_distance_str_formatted}). TSL might fail if attempted. Original calc: {tsl_distance_price_points.normalize()}")
                 tsl_distance_decimal_final = Decimal('NaN') # Mark as invalid

            # --- Format Final SL/TP Prices ---
            sl_price_str_formatted = self.exchange_manager.format_price(sl_price_calculated)
            sl_price_decimal_final = safe_decimal(sl_price_str_formatted)
            if sl_price_decimal_final.is_nan() or sl_price_decimal_final <= 0: # Should be caught earlier
                logger.error(f"Formatted SL price ({sl_price_str_formatted}) is invalid. Aborting parameter calculation.")
                return None

            tp_price_decimal_final: Optional[Decimal] = None
            if tp_price_calculated is not None: # If TP was calculated and not disabled
                 tp_price_str_formatted = self.exchange_manager.format_price(tp_price_calculated)
                 tp_price_decimal_final = safe_decimal(tp_price_str_formatted)
                 if tp_price_decimal_final.is_nan() or tp_price_decimal_final <= 0:
                      logger.warning(f"Failed to format a valid TP price ({tp_price_str_formatted}). Disabling TP for this trade.")
                      tp_price_decimal_final = None # Disable TP if formatting failed

            # --- Compile Output ---
            params_out: Dict[str, Optional[Decimal]] = {
                "qty": quantity_decimal_final,
                "sl_price": sl_price_decimal_final,
                "tp_price": tp_price_decimal_final, # Will be None if TP disabled or invalid
                "tsl_distance": tsl_distance_decimal_final, # Price difference for TSL, can be NaN
            }

            log_tp_str = f"{params_out['tp_price'].normalize()}" if params_out['tp_price'] else "Disabled"
            log_tsl_str = f"{params_out['tsl_distance'].normalize()}" if params_out['tsl_distance'] and not params_out['tsl_distance'].is_nan() else "Invalid/Not Set"
            logger.info(
                f"Trade Parameters Calculated for {side.upper()} entry: "
                f"Qty={params_out['qty'].normalize()}, "
                f"EntryPrice (approx.)={current_price.normalize()}, "
                f"SLPrice={params_out['sl_price'].normalize()}, "
                f"TPPrice={log_tp_str}, "
                f"TSLDistance (approx.)={log_tsl_str}, "
                f"RiskAmount={risk_amount_per_trade.normalize():.{DEFAULT_PRICE_DP}f} {self.market_info.get('settle', '')}, ATR={atr.normalize():.{DEFAULT_PRICE_DP+1}f}"
            )
            return params_out

        except (InvalidOperation, DivisionByZero, TypeError, Exception) as e:
            logger.error(f"Error calculating trade parameters for {side.upper()} side: {e}", exc_info=True)
            return None

    def _execute_market_order(
        self, side: str, qty_decimal: Decimal # 'buy' or 'sell'
    ) -> Optional[Dict]: # Returns CCXT order dict if successful, else None
        """Executes a market order with retries and basic confirmation logging."""
        if not self.exchange or not self.market_info: # Should be pre-checked by caller
            logger.error("Cannot execute market order: Exchange or Market info missing.")
            return None

        symbol_to_trade = self.config.symbol
        # Format quantity according to market rules (amount precision, round down)
        qty_str_formatted = self.exchange_manager.format_amount(qty_decimal, rounding_mode=ROUND_DOWN)
        final_qty_to_order_decimal = safe_decimal(qty_str_formatted)

        if final_qty_to_order_decimal.is_nan() or final_qty_to_order_decimal <= 0:
            logger.error(f"Attempted market order with zero/invalid formatted quantity: '{qty_str_formatted}' (Original Decimal: {qty_decimal.normalize()}). Order aborted.")
            return None

        logger.trade(
            f"{Fore.CYAN}Attempting MARKET {side.upper()} order: {final_qty_to_order_decimal.normalize()} {self.market_info.get('base', '')} for {symbol_to_trade}...{Style.RESET_ALL}"
        )
        try:
            # V5 specific parameters for creating an order
            params_v5 = {
                "category": self.config.bybit_v5_category,
                "positionIdx": self.config.position_idx, # Crucial for hedge mode vs one-way
                "timeInForce": "ImmediateOrCancel", # Or "FillOrKill"; market orders are usually FOK by nature
                # Bybit V5 usually doesn't require 'price' for market orders if 'createMarketBuyOrderRequiresPrice' is false (default)
            }
            # CCXT expects amount as float for create_market_order
            amount_float_for_ccxt = float(final_qty_to_order_decimal)

            order_response = fetch_with_retries(
                self.exchange.create_market_order,
                symbol=symbol_to_trade,
                side=side,
                amount=amount_float_for_ccxt,
                params=params_v5,
                max_retries=self.config.max_fetch_retries, # Use configured retries
                delay_seconds=self.config.retry_delay_seconds,
            )

            if order_response is None: # Should be caught by fetch_with_retries raising an error
                logger.error(f"{Fore.RED}Market order submission failed after retries (returned None unexpectedly).{Style.RESET_ALL}")
                return None

            # Parse common fields from CCXT unified order response
            order_id = order_response.get("id", "[N/A]")
            order_status = order_response.get("status", "[unknown]") # e.g., 'open', 'closed', 'canceled'
            filled_qty_str = order_response.get("filled", "0") # Amount filled
            avg_fill_price_str = order_response.get("average", "0") # Average fill price

            filled_qty_decimal = safe_decimal(filled_qty_str)
            avg_fill_price_decimal = safe_decimal(avg_fill_price_str)
            avg_price_log_str = avg_fill_price_decimal.normalize() if not avg_fill_price_decimal.is_nan() and avg_fill_price_decimal > 0 else "[N/A]"

            logger.trade(
                f"{Style.BRIGHT}{Fore.GREEN}Market order submitted: ID {order_id}, Side {side.upper()}, "
                f"Ordered Qty {final_qty_to_order_decimal.normalize()}, Status: {order_status}, "
                f"Filled Qty: {filled_qty_decimal.normalize()}, AvgFillPx: {avg_price_log_str}{Style.RESET_ALL}"
            )
            termux_notify(
                f"{symbol_to_trade} Order Submitted",
                f"Market {side.upper()} {final_qty_to_order_decimal.normalize()} ID:{order_id}, Status:{order_status}"
            )

            # Check for immediate rejection or cancellation
            if order_status in ["rejected", "canceled", "expired"]:
                 rejection_reason = order_response.get("info", {}).get("rejectReason", "No reason provided") # Bybit specific field
                 logger.error(f"{Fore.RED}Market order {order_id} was {order_status}. Reason: '{rejection_reason}'. Full info: {order_response.get('info')}{Style.RESET_ALL}")
                 return None # Order failed

            # Short delay to allow order processing on exchange before further checks
            # More robust verification is handled by _verify_position_state after this call.
            logger.debug(f"Short delay ({self.config.order_check_delay_seconds}s) after market order {order_id} submission...")
            time.sleep(self.config.order_check_delay_seconds)

            return order_response # Return the CCXT unified order dictionary

        except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e: # Fail-fast exceptions
            logger.error(f"{Fore.RED}Order placement failed ({type(e).__name__}): {e}{Style.RESET_ALL}")
            termux_notify(f"{symbol_to_trade} Order FAILED", f"Market {side.upper()} failed: {str(e)[:50]}")
            return None
        except Exception as e: # Other unexpected errors
            logger.error(f"{Fore.RED}Unexpected error placing market order: {e}{Style.RESET_ALL}", exc_info=True)
            termux_notify(f"{symbol_to_trade} Order ERROR", f"Market {side.upper()} unexpected error.")
            return None

    def _set_position_protection(
        self,
        position_side: str, # 'long' or 'short' (logical side of the position)
        sl_price: Optional[Decimal] = None,
        tp_price: Optional[Decimal] = None,
        is_tsl: bool = False, # True if activating/modifying TSL
        tsl_distance: Optional[Decimal] = None, # Trailing stop distance (price points)
        tsl_activation_price: Optional[Decimal] = None, # Price at which TSL should activate
    ) -> bool:
        """
        Sets SL, TP, or TSL for a position using Bybit V5's setTradingStop endpoint (via private CCXT call).
        Ensures numeric parameters are passed as formatted strings. Clears other stops when activating TSL.
        Updates internal protection_tracker on success.
        To clear all stops, call with all price/distance params as None and is_tsl=False.
        """
        if not self.exchange: logger.error("Cannot set position protection: Exchange not initialized."); return False
        if not self.market_info: logger.error("Cannot set position protection: Market info missing."); return False
        market_id = self.market_info.get("id")
        if not market_id: logger.error("Cannot set position protection: Market ID missing."); return False

        tracker_key = position_side.lower() # 'long' or 'short'
        if tracker_key not in self.protection_tracker: # Should not happen
             logger.error(f"Invalid position_side '{position_side}' for protection tracker update."); return False

        # --- Format API Parameters using _format_v5_param helper ---
        # Pass allow_zero=True for SL/TP as "0" means to cancel that specific stop.
        # Pass allow_zero=False for TSL distance/activation as "0" is usually not a valid value for these.
        sl_price_str = self.exchange_manager._format_v5_param(sl_price, "price", allow_zero=True)
        tp_price_str = self.exchange_manager._format_v5_param(tp_price, "price", allow_zero=True)
        tsl_distance_str = self.exchange_manager._format_v5_param(tsl_distance, "distance", allow_zero=False)
        tsl_activation_price_str = self.exchange_manager._format_v5_param(tsl_activation_price, "price", allow_zero=False)

        # --- Prepare Base Parameters for V5 private_post_position_set_trading_stop ---
        # These are common to all protection setting actions.
        base_api_params: Dict[str, Any] = {
            "category": self.config.bybit_v5_category,
            "symbol": market_id,
            "positionIdx": self.config.position_idx,
            "tpslMode": V5_TPSL_MODE_FULL, # Apply to entire position
            # Trigger prices for SL/TP/TSL (e.g., LastPrice, MarkPrice, IndexPrice)
            "slTriggerBy": self.config.sl_trigger_by,
            "tpTriggerBy": self.config.sl_trigger_by, # Typically TP trigger matches SL trigger
            "tslTriggerBy": self.config.tsl_trigger_by,
            # Initialize protection fields to "0" (meaning "do not set" or "cancel if existing")
            "stopLoss": "0", "takeProfit": "0", "trailingStop": "0", "activePrice": "0",
        }

        action_description = ""
        new_tracker_state: Optional[str] = None # What to set in self.protection_tracker on success

        # --- Logic Branch: Activate or Modify Trailing Stop Loss (TSL) ---
        if is_tsl:
            if tsl_distance_str and tsl_activation_price_str: # Both distance and activation price must be valid
                base_api_params["trailingStop"] = tsl_distance_str
                base_api_params["activePrice"] = tsl_activation_price_str
                # When setting TSL, Bybit V5 typically requires SL/TP to be "0" (cleared)
                base_api_params["stopLoss"] = "0"
                base_api_params["takeProfit"] = "0"
                action_description = f"ACTIVATE/MODIFY TSL (Distance: {tsl_distance_str}, ActivationPx: {tsl_activation_price_str})"
                new_tracker_state = "ACTIVE_TSL"
                logger.debug(f"Prepared TSL parameters: {base_api_params}")
            else: # Missing TSL parameters
                logger.error(f"Cannot activate TSL for {position_side.upper()}: Missing or invalid TSL distance ('{tsl_distance_str}') or activation price ('{tsl_activation_price_str}').")
                return False

        # --- Logic Branch: Set Fixed Stop Loss (SL) and/or Take Profit (TP) ---
        elif sl_price_str or tp_price_str: # At least one of SL or TP is being set/modified
            if sl_price_str: base_api_params["stopLoss"] = sl_price_str
            if tp_price_str: base_api_params["takeProfit"] = tp_price_str
            # When setting fixed SL/TP, ensure TSL is cleared by setting its params to "0"
            base_api_params["trailingStop"] = "0"
            base_api_params["activePrice"] = "0"
            action_description = f"SET SL={base_api_params['stopLoss']} TP={base_api_params['takeProfit']}"
            new_tracker_state = "ACTIVE_SLTP"
            logger.debug(f"Prepared SL/TP parameters: {base_api_params}")

        # --- Logic Branch: Clear All Protection (SL, TP, TSL) ---
        else: # No SL, TP, or TSL parameters provided, means clear all existing protection
            # base_api_params already defaults to "0" for all protection fields
            action_description = "CLEAR ALL SL/TP/TSL"
            new_tracker_state = None # No protection active
            logger.debug(f"Preparing to clear all stops for {position_side.upper()} position.")

        # --- Execute API Call using the correct private method ---
        logger.trade(f"{Fore.CYAN}Attempting to {action_description} for {position_side.upper()} {self.config.symbol}...{Style.RESET_ALL}")

        # Bybit V5 endpoint: POST /v5/position/set-trading-stop
        # CCXT private method name convention: privatePost<PathWithoutSlashesInCamelCase>
        # e.g., /v5/position/set-trading-stop -> privatePostPositionSetTradingStop
        private_method_name = "privatePostPositionSetTradingStop" # Corrected name

        if not hasattr(self.exchange, private_method_name):
            logger.error(
                f"{Style.BRIGHT}{Fore.RED}Fatal Error: CCXT private method '{private_method_name}' for setting position protection not found in exchange instance. "
                f"This might be due to an outdated CCXT version or an issue with its Bybit V5 implementation. Cannot manage position protection.{Style.RESET_ALL}"
            )
            return False # Cannot proceed

        method_to_call = getattr(self.exchange, private_method_name)
        logger.debug(f"Calling CCXT private V5 method '{private_method_name}' with parameters: {base_api_params}")

        try:
            response = fetch_with_retries(
                method_to_call, # Pass the bound private method
                params=base_api_params, # Pass the V5 parameters dictionary
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
            )
            # logger.debug(f"SetTradingStop API Response: {response}") # Verbose, enable if needed

            # --- Process API Response ---
            if response and response.get("retCode") == V5_SUCCESS_RETCODE:
                logger.trade(f"{Style.BRIGHT}{Fore.GREEN}{action_description} successful for {position_side.upper()} {self.config.symbol}.{Style.RESET_ALL}")
                termux_notify(f"{self.config.symbol} Protection Set", f"{action_description} for {position_side.upper()}")
                self.protection_tracker[tracker_key] = new_tracker_state # Update local tracker
                return True
            else: # API call failed or returned an error code
                ret_code = response.get("retCode", "[N/A]") if response else "[No Response]"
                ret_msg = response.get("retMsg", "[No error message]") if response else "[No Response]"
                logger.error(f"{Fore.RED}{action_description} failed for {position_side.upper()} {self.config.symbol}. API Response: Code={ret_code}, Msg='{ret_msg}'.{Style.RESET_ALL}")
                logger.debug(f"Full response from failed SetTradingStop: {response}") # Log full failing response
                termux_notify(f"{self.config.symbol} Protection FAILED", f"{action_description[:30]}... for {position_side.upper()} failed: {ret_msg[:50]}")
                return False
        except Exception as e: # Catch errors from fetch_with_retries or other unexpected issues
            logger.error(f"{Fore.RED}Unexpected error during '{action_description}' for {position_side.upper()} {self.config.symbol}: {e}{Style.RESET_ALL}", exc_info=True)
            termux_notify(f"{self.config.symbol} Protection ERROR", f"{action_description[:30]}... for {position_side.upper()} error.")
            return False

    # --- Enhanced Position Verification Method (integrated from Snippet 5) ---
    def _verify_position_state(
            self,
            expected_side: Optional[str], # 'long', 'short', or None (if expecting to be flat)
            expected_qty_min: Decimal = POSITION_QTY_EPSILON, # Min absolute qty if expecting an open position
            max_attempts: int = 4, # Number of verification attempts
            delay_seconds: float = 1.5, # Delay between attempts
            action_context: str = "Position Verification" # Context for logging (e.g., "Post-Entry", "Post-Close")
        ) -> Tuple[bool, Optional[Dict[str, Dict[str, Any]]]]:
        """
        Fetches current position state repeatedly to verify if it matches the expected state after an action.
        Returns a tuple: (verification_success: bool, final_position_state_summary: Optional[Dict]).
        `final_position_state_summary` is the last fetched position summary from `get_current_position()`.
        """
        logger.debug(f"{action_context}: Verifying position state. Expecting side: '{expected_side}', MinQty (if open): {expected_qty_min.normalize()}. Max attempts: {max_attempts}.")
        last_known_position_summary: Optional[Dict[str, Dict[str, Any]]] = None

        for attempt in range(max_attempts):
            logger.debug(f"{action_context}: Verification attempt {attempt + 1}/{max_attempts}...")
            current_positions_summary = self.exchange_manager.get_current_position()
            last_known_position_summary = current_positions_summary # Store the latest fetched state

            if current_positions_summary is None: # Failed to fetch position data
                logger.warning(f"{action_context} Warning: Failed to fetch position state on attempt {attempt + 1}.")
                if attempt < max_attempts - 1:
                    time.sleep(delay_seconds)
                    continue # Retry fetching
                else: # Max attempts reached for fetching
                    logger.error(f"{Fore.RED}{action_context} FAILED: Could not fetch position state after {max_attempts} attempts.{Style.RESET_ALL}")
                    return False, last_known_position_summary # Verification failed, return last attempt's state (which is None here)

            # --- Analyze Fetched Position State ---
            is_currently_flat = True
            actual_open_side: Optional[str] = None
            actual_open_qty = Decimal("0")
            # active_pos_details = None # Not strictly needed here, we operate on summary

            long_pos_data = current_positions_summary.get("long", {})
            short_pos_data = current_positions_summary.get("short", {})

            if long_pos_data: # Check for active long position
                 long_qty_val = safe_decimal(long_pos_data.get("qty", "0"))
                 if long_qty_val.copy_abs() >= POSITION_QTY_EPSILON: # Position Qty Epsilon
                     is_currently_flat = False
                     actual_open_side = "long"
                     actual_open_qty = long_qty_val
                     # active_pos_details = long_pos_data

            if not actual_open_side and short_pos_data: # Check for active short if not already found long
                 short_qty_val = safe_decimal(short_pos_data.get("qty", "0"))
                 if short_qty_val.copy_abs() >= POSITION_QTY_EPSILON:
                     is_currently_flat = False
                     actual_open_side = "short"
                     actual_open_qty = short_qty_val
                     # active_pos_details = short_pos_data

            # --- Evaluate if current state matches expected state ---
            verification_succeeded = False
            log_message_suffix = ""

            if expected_side is None: # Expecting to be flat
                verification_succeeded = is_currently_flat
                log_message_suffix = f"Expected FLAT, Actual: {'FLAT' if is_currently_flat else f'{actual_open_side.upper()} Qty={actual_open_qty.normalize()}'}"
            elif actual_open_side == expected_side: # Expecting a specific side, and it matches
                # Check if quantity meets minimum expectation
                quantity_matches_expectation = actual_open_qty.copy_abs() >= expected_qty_min
                verification_succeeded = quantity_matches_expectation
                log_message_suffix = (f"Expected {expected_side.upper()} (MinQty~{expected_qty_min.normalize()}), "
                                      f"Actual: {actual_open_side.upper()} Qty={actual_open_qty.normalize()} "
                                      f"({'QTY OK' if quantity_matches_expectation else 'QTY MISMATCH'})")
            else: # Expecting a specific side, but actual side is different or flat
                 verification_succeeded = False # Side mismatch
                 log_message_suffix = (f"Expected {expected_side.upper()}, "
                                       f"Actual: {'FLAT' if is_currently_flat else (actual_open_side.upper() + ' Qty=' + actual_open_qty.normalize()) if actual_open_side else 'UNKNOWN/ERROR'} "
                                       f"(SIDE MISMATCH)")

            logger.debug(f"{action_context} Check {attempt + 1}: {log_message_suffix}")

            if verification_succeeded:
                logger.info(f"{Style.BRIGHT}{Fore.GREEN}{action_context} SUCCEEDED on attempt {attempt + 1}. State confirmed: {log_message_suffix}{Style.RESET_ALL}")
                return True, current_positions_summary # Verification succeeded, return final confirmed state

            # Verification not met, wait for next attempt if any remain
            if attempt < max_attempts - 1:
                logger.debug(f"State not as expected. Waiting {delay_seconds}s for next attempt...")
                time.sleep(delay_seconds)
            else: # Max attempts reached, verification failed
                 logger.error(f"{Fore.RED}{action_context} FAILED after {max_attempts} attempts. Final state check: {log_message_suffix}{Style.RESET_ALL}")
                 return False, current_positions_summary # Verification failed, return last known state

        # Fallback, should ideally not be reached if max_attempts >= 1
        logger.error(f"{action_context} Verification loop completed unexpectedly without returning. This indicates a logic flaw.")
        return False, last_known_position_summary
    # --- End Enhanced Position Verification Method ---

    # --- Updated place_risked_market_order using _verify_position_state ---
    def place_risked_market_order(
        self,
        side: str, # 'buy' or 'sell' (for the entry order)
        atr: Decimal,
        total_equity: Decimal,
        current_price: Decimal,
    ) -> bool: # Returns True if entry sequence (order, verification, protection) is successful
        """
        Orchestrates a risked market order entry:
        1. Calculates trade parameters (qty, SL, TP) based on risk.
        2. Executes the market order.
        3. Verifies that the position is established correctly.
        4. Sets initial Stop Loss and Take Profit for the position.
        Logs entry to journal on success. Handles failures at each step.
        """
        if not self.exchange or not self.market_info: return False # Should be caught by TradingBot init
        if side not in ["buy", "sell"]: logger.error(f"Invalid side '{side}' for place_risked_market_order."); return False
        # Critical inputs for calculation must be valid
        if atr.is_nan() or atr <= 0: logger.error("Entry Aborted: Invalid ATR value."); return False
        if total_equity is None or total_equity.is_nan() or total_equity <= 0: logger.error("Entry Aborted: Invalid Equity value."); return False
        if current_price.is_nan() or current_price <= 0: logger.error("Entry Aborted: Invalid Current Price."); return False

        logical_position_side = "long" if side == "buy" else "short"
        logger.info(f"{Style.BRIGHT}{Fore.MAGENTA}--- Initiating Entry Sequence for {logical_position_side.upper()} Position ---{Style.RESET_ALL}")

        # 1. Calculate Trade Parameters
        logger.debug(f"Calculating trade parameters for {logical_position_side.upper()} entry...")
        trade_params = self._calculate_trade_parameters(side, atr, total_equity, current_price)
        if not trade_params or not trade_params.get("qty") or trade_params["qty"] <= 0:
            logger.error("Entry Aborted: Failed to calculate valid trade parameters (quantity is missing, zero, or negative).")
            return False
        # Extract calculated parameters
        qty_to_order = trade_params["qty"] # This is validated positive by _calculate_trade_parameters
        initial_sl_price = trade_params.get("sl_price")
        initial_tp_price = trade_params.get("tp_price") # Optional, can be None

        if initial_sl_price is None or initial_sl_price.is_nan() or initial_sl_price <= 0:
             logger.error(f"Entry Aborted: Invalid Stop Loss price ({initial_sl_price}) calculated. Cannot proceed without valid SL.")
             return False

        # 2. Execute Market Order
        logger.debug(f"Executing market {side.upper()} order for {qty_to_order.normalize()}...")
        market_order_info = self._execute_market_order(side, qty_to_order)
        if not market_order_info: # Order submission failed or was rejected
            logger.error(f"Entry Aborted: Market order execution failed or was rejected for {side.upper()} {qty_to_order.normalize()}.")
            # Attempt cleanup in case of partial fill or unknown state, though _execute_market_order should indicate clear failure
            self._handle_entry_failure(side, qty_to_order)
            return False
        entry_order_id = market_order_info.get("id", "[N/A_ORDER_ID]")

        # 3. Verify Position Establishment robustly using _verify_position_state
        logger.info(f"Verifying position establishment after market order {entry_order_id} for {logical_position_side.upper()}...")
        # Define verification parameters: expect the logical_position_side, with quantity close to ordered_qty
        # Allow some tolerance for fees/slippage, e.g., 90% of ordered quantity
        min_expected_filled_qty = qty_to_order * Decimal("0.90")
        verification_ok, final_verified_pos_state = self._verify_position_state(
            expected_side=logical_position_side,
            expected_qty_min=min_expected_filled_qty,
            max_attempts=6, # Allow more attempts for position state to update post-order
            delay_seconds=max(self.config.order_check_delay_seconds, 1.0), # Use configured delay or 1s
            action_context=f"Post-{logical_position_side.upper()}-Entry Verification"
        )

        if not verification_ok:
            logger.error(f"{Fore.RED}Entry FAILED: Position verification indicated failure after market order {entry_order_id}. Manual check of position state is required! Attempting cleanup...{Style.RESET_ALL}")
            # _handle_entry_failure will check current state and try to close if unexpectedly open
            self._handle_entry_failure(side, qty_to_order)
            return False

        # Position verified successfully. Extract details from the final verified state.
        # `final_verified_pos_state` is the summary dict from get_current_position.
        active_pos_details = final_verified_pos_state.get(logical_position_side) if final_verified_pos_state else {}
        if not active_pos_details: # Should not happen if verification_ok is True and expected_side was not None
            logger.error(f"{Fore.RED}Internal Error: Position for {logical_position_side} verified OK, but details missing in final state: {final_verified_pos_state}. Aborting entry sequence.{Style.RESET_ALL}")
            self._handle_entry_failure(side, qty_to_order) # Attempt cleanup
            return False

        # Get actual filled quantity and average entry price from verified position
        actual_filled_qty = safe_decimal(active_pos_details.get("qty", "0"))
        actual_avg_entry_price = safe_decimal(active_pos_details.get("entry_price", "NaN"))

        logger.info(
            f"{Style.BRIGHT}{Fore.GREEN}Position {logical_position_side.upper()} confirmed via verification: "
            f"Actual Qty={actual_filled_qty.normalize()}, AvgEntryPx={actual_avg_entry_price.normalize() if not actual_avg_entry_price.is_nan() else '[N/A]'}{Style.RESET_ALL}"
        )
        # Log if filled quantity significantly differs from ordered quantity
        if actual_filled_qty < qty_to_order * Decimal("0.99"): # e.g., less than 99% filled
             logger.warning(f"Filled quantity {actual_filled_qty.normalize()} is notably less than ordered quantity {qty_to_order.normalize()}.")

        # 4. Set Initial Position SL/TP using the verified position details
        logger.info(f"Setting initial SL/TP for confirmed {logical_position_side.upper()} position...")
        set_stops_successful = self._set_position_protection(
            logical_position_side,
            sl_price=initial_sl_price,
            tp_price=initial_tp_price
            # is_tsl=False, tsl params default to None
        )

        if not set_stops_successful:
            logger.error(f"{Fore.RED}Entry Sequence Alert: Failed to set initial SL/TP after establishing {logical_position_side.upper()} position. Attempting emergency close of position!{Style.RESET_ALL}")
            # If stops cannot be set, it's safer to close the unprotected position
            self.close_position(logical_position_side, actual_filled_qty, reason="EmergencyClose:FailedInitialStopSet")
            return False # Entry sequence ultimately failed

        # 5. Log Successful Entry to Journal
        if self.config.enable_journaling:
            if actual_avg_entry_price.is_nan(): # Should have valid entry price if position verified
                logger.warning("Logging trade entry to journal with N/A average entry price. Position data might be incomplete.")
            self.log_trade_entry_to_journal(side, actual_filled_qty, actual_avg_entry_price, entry_order_id)

        logger.info(f"{Style.BRIGHT}{Fore.GREEN}--- Entry Sequence for {logical_position_side.upper()} Completed Successfully ---{Style.RESET_ALL}")
        return True
    # --- End updated place_risked_market_order ---

    def manage_trailing_stop(
        self,
        position_side: str, # 'long' or 'short'
        entry_price: Decimal, # Verified entry price of the current position
        current_market_price: Decimal, # Latest market price
        current_atr: Decimal, # Current ATR value
    ) -> None:
        """Checks if TSL activation conditions are met and attempts to activate TSL using V5 position TSL."""
        if not self.exchange or not self.market_info: return # Should not happen
        tracker_key = position_side.lower() # 'long' or 'short'

        # Only attempt to activate TSL if fixed SL/TP is currently believed to be active
        current_protection_state_local = self.protection_tracker.get(tracker_key)
        if current_protection_state_local != "ACTIVE_SLTP":
            log_msg_tsl_check = (f"TSL already active or in transition (Tracker: {current_protection_state_local})."
                                 if current_protection_state_local == "ACTIVE_TSL"
                                 else f"No active SL/TP protection tracked (Tracker: {current_protection_state_local}). Cannot activate TSL yet.")
            logger.debug(f"TSL Management Check ({position_side.upper()}): {log_msg_tsl_check}")
            return

        # Validate inputs for TSL calculation
        if current_atr.is_nan() or current_atr <= 0: logger.debug(f"TSL Check ({position_side.upper()}): Invalid ATR ({current_atr.normalize()}). Skipping TSL management."); return
        if entry_price.is_nan() or entry_price <= 0: logger.debug(f"TSL Check ({position_side.upper()}): Invalid entry price ({entry_price.normalize()}). Skipping."); return
        if current_market_price.is_nan() or current_market_price <= 0: logger.debug(f"TSL Check ({position_side.upper()}): Invalid current market price ({current_market_price.normalize()}). Skipping."); return

        try:
            # Calculate TSL Activation Price: Entry Price + (ATR * Activation Multiplier)
            activation_distance_points = current_atr * self.config.tsl_activation_atr_multiplier
            tsl_activation_target_price: Decimal
            if position_side == "long":
                tsl_activation_target_price = entry_price + activation_distance_points
            else: # short position
                tsl_activation_target_price = entry_price - activation_distance_points

            if tsl_activation_target_price.is_nan() or tsl_activation_target_price <= 0:
                logger.warning(f"Calculated invalid TSL activation price ({tsl_activation_target_price.normalize()}). Skipping TSL activation for {position_side.upper()}.")
                return

            # Calculate TSL Distance: Percentage of current market price
            tsl_actual_distance_points = current_market_price * (self.config.trailing_stop_percent / 100)
            min_tick_size = self.market_info.get('tick_size', Decimal('1e-8')) # Default to small if not found
            if tsl_actual_distance_points < min_tick_size: # Ensure TSL distance is at least one tick
                logger.debug(f"TSL distance ({tsl_actual_distance_points.normalize()}) < min tick ({min_tick_size.normalize()}). Adjusting to min tick size.")
                tsl_actual_distance_points = min_tick_size
            if tsl_actual_distance_points <= 0: # Should be positive
                 logger.warning(f"Invalid TSL distance calculated ({tsl_actual_distance_points.normalize()}) after adjustments. Skipping TSL activation for {position_side.upper()}.")
                 return

            # Check if TSL activation condition is met
            should_activate_tsl = False
            if position_side == "long" and current_market_price >= tsl_activation_target_price:
                should_activate_tsl = True
            elif position_side == "short" and current_market_price <= tsl_activation_target_price:
                should_activate_tsl = True

            if should_activate_tsl:
                logger.trade(
                    f"{Fore.MAGENTA}Trailing Stop Loss (TSL) activation condition MET for {position_side.upper()} position!{Style.RESET_ALL}"
                )
                logger.trade(
                    f"  Details: EntryPx={entry_price.normalize()}, CurrentPx={current_market_price.normalize()}, "
                    f"TSLActivationTargetPx~={tsl_activation_target_price.normalize():.{DEFAULT_PRICE_DP}f}"
                )
                logger.trade(
                    f"  Attempting TSL activation with Distance~={tsl_actual_distance_points.normalize()} and ActivationPx={tsl_activation_target_price.normalize():.{DEFAULT_PRICE_DP}f}..."
                )

                # Attempt to set TSL on the exchange, this will also clear existing SL/TP
                activation_successful = self._set_position_protection(
                    position_side,
                    is_tsl=True,
                    tsl_distance=tsl_actual_distance_points, # The calculated trailing distance
                    tsl_activation_price=tsl_activation_target_price, # The price that triggered activation
                )
                if activation_successful:
                    logger.trade(f"{Style.BRIGHT}{Fore.GREEN}TSL activated successfully for {position_side.upper()} position.{Style.RESET_ALL}")
                    # self.protection_tracker already updated by _set_position_protection
                else:
                    logger.error(f"{Fore.RED}Failed to activate TSL for {position_side.upper()} position via API.{Style.RESET_ALL}")
            else: # TSL activation condition not met
                logger.debug(
                    f"TSL Check ({position_side.upper()}): Activation condition NOT MET. "
                    f"(CurrentPx: {current_market_price.normalize()}, TargetActivationPx: ~{tsl_activation_target_price.normalize():.{DEFAULT_PRICE_DP}f})"
                )
        except Exception as e:
            logger.error(f"Error managing TSL for {position_side.upper()} position: {e}", exc_info=True)

    # --- Updated close_position using _verify_position_state ---
    def close_position(
        self, position_side: str, # 'long' or 'short'
        qty_to_close: Decimal,
        reason: str = "Strategy Exit Signal"
    ) -> bool: # Returns True if closure sequence (clear protection, order, verification) is successful
        """
        Closes the specified position:
        1. Clears any existing position protection (SL/TP/TSL).
        2. Places a market order to close the position.
        3. Verifies that the position is actually closed (flat).
        Logs exit to journal. Handles failures.
        """
        if not self.exchange or not self.market_info: return False
        if position_side not in ["long", "short"]: logger.error(f"Invalid side '{position_side}' for close_position."); return False

        if qty_to_close.is_nan() or qty_to_close.copy_abs() < POSITION_QTY_EPSILON:
            logger.warning(f"Close requested for zero or negligible quantity ({qty_to_close.normalize()}). Skipping actual close for {position_side.upper()}. Marking protection as None.")
            self.protection_tracker[position_side.lower()] = None # Clear protection tracker
            return True # Considered successful as no action needed for zero/small qty

        symbol_to_trade = self.config.symbol
        # Determine market order side to close the position
        closing_order_side = "sell" if position_side == "long" else "buy"
        tracker_key = position_side.lower()

        logger.trade(
            f"{Fore.YELLOW}Attempting to CLOSE {position_side.upper()} position (Qty: {qty_to_close.normalize()} {self.market_info.get('base', '')}) "
            f"for {symbol_to_trade} | Reason: {reason}...{Style.RESET_ALL}"
        )

        # 1. Clear Existing Position Protection (SL/TP/TSL)
        logger.info(f"Clearing any existing protection for {position_side.upper()} position before closing...")
        clear_stops_successful = self._set_position_protection(
            position_side, sl_price=None, tp_price=None, is_tsl=False # Clears all
        )
        if not clear_stops_successful:
            # This is a warning because the closing order might still succeed.
            # However, it's a risk if stops remain and interfere.
            logger.warning(f"{Fore.YELLOW}Failed to confirm protection clear via API for {position_side.upper()}. Proceeding with close order cautiously...{Style.RESET_ALL}")
        else:
            logger.info(f"Protection cleared successfully (or was already clear) for {position_side.upper()} position.")
        # self.protection_tracker already updated by _set_position_protection to None on success

        # 2. Place Closing Market Order
        logger.info(f"Submitting MARKET {closing_order_side.upper()} order to close {position_side.upper()} position...")
        close_market_order_info = self._execute_market_order(closing_order_side, qty_to_close)

        if not close_market_order_info: # Closing order failed to submit or was rejected
            logger.error(f"{Fore.RED}Failed to submit closing market order for {position_side.upper()} position. MANUAL INTERVENTION REQUIRED!{Style.RESET_ALL}")
            termux_notify(f"{symbol_to_trade} CLOSE ORDER FAILED", f"Market {closing_order_side.upper()} order for {qty_to_close.normalize()} failed!")
            # Position might still be open and unprotected. State is uncertain.
            return False # Closure sequence failed

        close_order_id = close_market_order_info.get("id", "[N/A_CLOSE_ORDER_ID]")
        avg_close_price_str = close_market_order_info.get("average") # Avg fill price from order response
        avg_close_price_decimal = safe_decimal(avg_close_price_str, default=Decimal("NaN")) # May be NaN if not filled yet

        logger.trade(
            f"{Fore.YELLOW}Closing market order ({close_order_id}) submitted for {position_side.upper()}. "
            f"Reported AvgClosePrice: {avg_close_price_decimal.normalize() if not avg_close_price_decimal.is_nan() else '[Pending/N/A]'}{Style.RESET_ALL}"
        )
        termux_notify(f"{symbol_to_trade} Position Closing", f"{position_side.upper()} close order {close_order_id} submitted.")

        # 3. Verify Position is Actually Closed (Flat) using _verify_position_state
        logger.info(f"Verifying position closure for {position_side.upper()} (expecting flat)...")
        # Expecting None (flat) for the side that was closed.
        # expected_qty_min is irrelevant when expecting flat.
        verification_ok, final_verified_pos_state_after_close = self._verify_position_state(
            expected_side=None, # Expecting to be flat
            max_attempts=6, # More attempts for close verification
            delay_seconds=max(self.config.order_check_delay_seconds + 0.5, 1.5), # Slightly longer delay
            action_context=f"Post-{position_side.upper()}-Close Verification"
        )

        # 4. Log Exit to Journal (use avg_close_price from order if available, otherwise NaN)
        if self.config.enable_journaling:
            # If verification failed but order was submitted, log with available price.
            # If verification succeeded (flat), avg_close_price might be from initial order response.
            self.log_trade_exit_to_journal(
                position_side, qty_to_close, avg_close_price_decimal, close_order_id, reason
            )

        # 5. Handle Verification Outcome
        if not verification_ok:
            # Attempt to determine if the position is still open from `final_verified_pos_state_after_close`
            lingering_side_details = None
            if final_verified_pos_state_after_close:
                lingering_side_details = final_verified_pos_state_after_close.get(position_side) # Check the side we tried to close
                if not lingering_side_details or safe_decimal(lingering_side_details.get("qty", "0")).copy_abs() < POSITION_QTY_EPSILON:
                    # If verification said "failed" but detailed check shows it's flat, consider it a soft failure/success.
                    logger.warning(f"{Fore.YELLOW}Closure verification for {position_side.upper()} reported failure, but detailed check of last known state suggests it might be flat or qty negligible. Proceed with caution.{Style.RESET_ALL}")
                    # Could return True here if confident, or stick with False due to verification discrepancy.
                    # For safety, if _verify_position_state returns False, we honor that.
            logger.error(
                f"{Fore.RED}Position {position_side.upper()} closure verification FAILED. "
                f"The position may still be open or partially open. MANUAL CHECK AND INTERVENTION REQUIRED!{Style.RESET_ALL}"
            )
            termux_notify(f"{symbol_to_trade} CLOSE VERIFY FAILED", f"{position_side.upper()} position may still be open after close attempt!")
            return False # Indicate closure sequence ultimately failed due to verification

        # Verification successful, position is confirmed flat
        logger.trade(f"{Style.BRIGHT}{Fore.GREEN}Position {position_side.upper()} confirmed closed (flat) via verification.{Style.RESET_ALL}")
        # Ensure protection tracker is cleared again, though _set_position_protection should have done it.
        self.protection_tracker[tracker_key] = None
        return True # Closure sequence fully successful
    # --- End updated close_position ---

    def _handle_entry_failure(
        self, failed_entry_order_side: str, # 'buy' or 'sell' (the side of the market order that failed)
        attempted_qty: Decimal
    ):
        """
        Handles cleanup after a failed entry sequence step.
        Checks for any inadvertently opened position corresponding to the failed entry
        and attempts to close it as an emergency measure.
        """
        logger.warning(
            f"{Fore.YELLOW}Handling potential entry failure for {failed_entry_order_side.upper()} order (intended qty: {attempted_qty.normalize()}). "
            f"Checking for any lingering position...{Style.RESET_ALL}"
        )
        # Determine the logical position side that might have been created
        logical_pos_side_to_check = "long" if failed_entry_order_side == "buy" else "short"

        # Wait briefly to allow exchange state to settle before checking
        time.sleep(max(self.config.order_check_delay_seconds, 1.0) + 1) # Add a bit more buffer
        logger.debug(f"Checking current position status after {failed_entry_order_side.upper()} entry attempt failure...")

        # Use _verify_position_state to robustly check the current state.
        # We don't have a strong expectation here, just want to see *if* a position exists.
        # So, expected_side=None is not right. We check if the *intended* side is open.
        # If it is, that's the lingering position.
        _, current_positions_summary = self._verify_position_state(
            expected_side=None, # Pass None, as we just want the current state, not to verify a specific state.
                                # The analysis of this state happens below.
            max_attempts=2, # Quick check with a few attempts
            delay_seconds=1.0,
            action_context=f"Entry-Failure-Cleanup-Check-{logical_pos_side_to_check.upper()}"
        )

        if current_positions_summary is None:
            # If fetching positions fails during cleanup, this is a critical situation.
            logger.error(f"{Fore.RED}Could not fetch current positions during entry failure handling for {logical_pos_side_to_check.upper()}. MANUAL CHECK AND INTERVENTION URGENTLY REQUIRED!{Style.RESET_ALL}")
            termux_notify(f"{self.config.symbol} URGENT CHECK", "Failed to get position state during entry failure cleanup!")
            return

        # Check if the intended position side has an active quantity
        lingering_pos_details = current_positions_summary.get(logical_pos_side_to_check, {})
        current_lingering_qty = safe_decimal(lingering_pos_details.get("qty", "0"))

        if current_lingering_qty.copy_abs() >= POSITION_QTY_EPSILON: # If a significant quantity exists
            logger.error(
                f"{Fore.RED}Lingering {logical_pos_side_to_check.upper()} position (Qty: {current_lingering_qty.normalize()}) "
                f"found after failed entry attempt. Attempting emergency close...{Style.RESET_ALL}"
            )
            termux_notify(f"{self.config.symbol} Emergency Close", f"Lingering {logical_pos_side_to_check.upper()} pos found after entry fail.")
            # Attempt to close this lingering, unintended position
            close_success = self.close_position(
                logical_pos_side_to_check, current_lingering_qty, reason="EmergencyClose:LingeringAfterEntryFail"
            )
            if close_success:
                logger.info(f"Emergency close order for lingering {logical_pos_side_to_check.upper()} position submitted/confirmed.")
            else: # Emergency close itself failed
                logger.critical(f"{Style.BRIGHT}{Fore.RED}EMERGENCY CLOSE FAILED for lingering {logical_pos_side_to_check.upper()} position. MANUAL INTERVENTION URGENTLY REQUIRED!{Style.RESET_ALL}")
                termux_notify(f"{self.config.symbol} URGENT CHECK", f"Emergency close of lingering {logical_pos_side_to_check.upper()} FAILED!")
        else: # No significant lingering position found for the intended side
            logger.info(f"No significant lingering {logical_pos_side_to_check.upper()} position detected after entry failure attempt. Current qty: {current_lingering_qty.normalize()}.")
            # Ensure protection tracker is clear for this side if it's indeed flat
            self.protection_tracker[logical_pos_side_to_check] = None

    def _write_journal_row(self, trade_data: Dict[str, Any]):
        """Helper function to write a single row to the CSV trading journal."""
        if not self.config.enable_journaling: return # Do nothing if journaling is disabled
        journal_file = Path(self.config.journal_file_path)
        file_already_exists = journal_file.is_file() and journal_file.stat().st_size > 0

        try:
            journal_file.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            with journal_file.open("a", newline="", encoding="utf-8") as csvfile:
                # Define consistent field names for the journal
                fieldnames = [
                    "TimestampUTC", "Symbol", "Action", "Side", "Quantity",
                    "AvgPrice", "OrderID", "Reason", "Notes"
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
                if not file_already_exists: # Write header only if file is new or empty
                    writer.writeheader()

                # Prepare row data, formatting Decimals and handling None
                row_to_write = {}
                for field in fieldnames:
                    value = trade_data.get(field)
                    if isinstance(value, Decimal):
                        row_to_write[field] = 'NaN' if value.is_nan() else f"{value.normalize()}"
                    elif value is None:
                        row_to_write[field] = 'N/A' # Use 'N/A' for None values
                    else:
                        row_to_write[field] = str(value) # Convert other types to string
                # Ensure 'Notes' field is present, even if empty
                row_to_write['Notes'] = trade_data.get('Notes', '') # Default to empty string if no notes

                writer.writerow(row_to_write)
            logger.debug(f"Trade action '{trade_data.get('Action', 'Unknown')}' logged to journal: {journal_file}")
        except IOError as e:
            logger.error(f"I/O error writing trade action '{trade_data.get('Action', '')}' to journal '{journal_file}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error writing trade action '{trade_data.get('Action', '')}' to journal: {e}", exc_info=True)

    def log_trade_entry_to_journal(
        self, order_side: str, # 'buy' or 'sell' (of the entry order)
        filled_qty: Decimal, avg_fill_price: Decimal, order_id: Optional[str]
    ):
        """Logs trade entry details to the CSV journal."""
        logical_position_side = "long" if order_side == "buy" else "short"
        entry_data = {
            "TimestampUTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "Symbol": self.config.symbol,
            "Action": "ENTRY",
            "Side": logical_position_side.upper(),
            "Quantity": filled_qty,
            "AvgPrice": avg_fill_price,
            "OrderID": order_id,
            "Reason": "Strategy Entry Signal", # Default reason for entry
        }
        self._write_journal_row(entry_data)

    def log_trade_exit_to_journal(
        self, position_side_closed: str, # 'long' or 'short' (the position that was closed)
        closed_qty: Decimal, avg_close_price: Decimal,
        order_id: Optional[str], exit_reason: str
    ):
        """Logs trade exit details to the CSV journal."""
        exit_data = {
            "TimestampUTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "Symbol": self.config.symbol,
            "Action": "EXIT",
            "Side": position_side_closed.upper(),
            "Quantity": closed_qty,
            "AvgPrice": avg_close_price,
            "OrderID": order_id,
            "Reason": exit_reason, # Specific reason for exit (e.g., SL, TP, Signal)
        }
        self._write_journal_row(exit_data)


# --- Status Display Class ---
class StatusDisplay:
    """Handles displaying the bot's status and key information using the Rich library."""

    def __init__(self, config: TradingConfig):
        self.config = config
        # Store default precisions, can be overridden by market_info later
        self._default_price_dp_display = DEFAULT_PRICE_DP
        self._default_amount_dp_display = DEFAULT_AMOUNT_DP

    def _format_decimal_for_rich(
        self,
        value: Optional[Decimal],
        precision: Optional[int] = None, # Specific precision for this value
        default_precision_fallback: int = 2, # Fallback if market/specific precision not set
        add_commas: bool = False, # For large numbers like equity
        highlight_negative: bool = False, # For PnL
        default_style: str = "white", # Default Rich style
        style_override: Optional[str] = None # Force a specific style
    ) -> Text:
        """Formats Decimal values for Rich Text display with styling options."""
        if value is None or (isinstance(value, Decimal) and value.is_nan()):
            return Text("N/A", style="dim") # Display "N/A" for None or NaN

        # Determine the number of decimal places for formatting
        dp_to_use = precision if precision is not None else default_precision_fallback

        try:
            # Quantize to the desired decimal places using ROUND_HALF_EVEN for display
            quantizer = Decimal("1e-" + str(dp_to_use))
            formatted_decimal_val = value.quantize(quantizer, rounding=ROUND_HALF_EVEN)

            # Create format spec string for f-string formatting (e.g., "{:,.2f}" or "{:.4f}")
            format_spec = f"{{:{',' if add_commas else ''}.{dp_to_use}f}}"
            display_string = format_spec.format(formatted_decimal_val)

            # Determine Rich style
            current_style = style_override if style_override else default_style
            if highlight_negative and not style_override: # Only apply if no override
                if formatted_decimal_val < 0: current_style = "bright_red"
                elif formatted_decimal_val > 0: current_style = "bright_green"
                # If zero, keeps default_style or style_override

            return Text(display_string, style=current_style)
        except (ValueError, TypeError, InvalidOperation) as e: # Catch formatting errors
            logger.error(f"Error formatting decimal '{value}' for Rich display: {e}")
            return Text("ERR", style="bold bright_red") # Error indicator

    def print_status_panel(
        self,
        cycle_num: int,
        current_timestamp: Optional[datetime],
        current_market_price: Optional[Decimal],
        indicators_data: Optional[Dict], # Full indicators dict
        current_positions_summary: Optional[Dict], # From ExchangeManager.get_current_position()
        account_equity: Optional[Decimal],
        trading_signals: Dict, # From SignalGenerator.generate_signals()
        protection_status_tracker: Dict, # OrderManager.protection_tracker snapshot
        market_specific_info: Optional[Dict] # ExchangeManager.market_info
    ):
        """Prints the main status panel to the console using Rich Panel and Text objects."""
        # Determine price and amount decimal places from market_info if available
        price_display_dp = self._default_price_dp_display
        amount_display_dp = self._default_amount_dp_display
        if market_specific_info and "precision_dp" in market_specific_info:
             price_display_dp = market_specific_info["precision_dp"].get("price", self._default_price_dp_display)
             amount_display_dp = market_specific_info["precision_dp"].get("amount", self._default_amount_dp_display)

        # --- Panel Title and Basic Info ---
        panel_content = Text() # Main container for all text in the panel
        timestamp_str = current_timestamp.strftime("%Y-%m-%d %H:%M:%S %Z") if current_timestamp else Text("Timestamp N/A", style="dim")
        panel_title_str = f" Cycle {cycle_num} | {self.config.symbol} ({self.config.interval}) | {timestamp_str} "

        # Price & Equity Display
        price_text_formatted = self._format_decimal_for_rich(current_market_price, precision=price_display_dp, style_override="bright_white")
        settle_currency_code = self.config.symbol.split(":")[-1] if ":" in self.config.symbol else market_specific_info.get("settle", "QUOTE") if market_specific_info else "QUOTE"
        equity_text_formatted = self._format_decimal_for_rich(account_equity, precision=2, add_commas=True, style_override="bright_yellow")

        panel_content.append("Price: ", style="bold bright_cyan"); panel_content.append(price_text_formatted)
        panel_content.append(" | ", style="dim")
        panel_content.append("Equity: ", style="bold bright_yellow"); panel_content.append(equity_text_formatted)
        panel_content.append(f" {settle_currency_code}\n", style="bright_yellow")
        panel_content.append("---\n", style="dim") # Separator line

        # --- Indicators Display ---
        panel_content.append("Indicators: ", style="bold bright_cyan")
        if indicators_data:
            indicator_parts_text = [] # List to hold Rich Text objects for each indicator group
            # Helper to format an indicator value from the dict
            def format_indicator_value(key: str, prec: int = 1, style: str = "white") -> Text:
                 val = indicators_data.get(key)
                 dec_val = val if isinstance(val, Decimal) else safe_decimal(val) # Ensure Decimal for formatting
                 return self._format_decimal_for_rich(dec_val, precision=prec, default_style=style)

            # EMAs (Fast/Slow/Trend)
            ema_text_group = Text("EMA(F/S/T): ")
            ema_text_group.append(format_indicator_value('fast_ema', prec=price_display_dp, style="cyan"))
            ema_text_group.append("/"); ema_text_group.append(format_indicator_value('slow_ema', prec=price_display_dp, style="magenta"))
            ema_text_group.append("/"); ema_text_group.append(format_indicator_value('trend_ema', prec=price_display_dp, style="yellow"))
            indicator_parts_text.append(ema_text_group)

            # Stochastic (K/D/PrevK), with K/D cross symbols
            stoch_text_group = Text("Stoch(K/D/PrevK): ")
            stoch_text_group.append(format_indicator_value('stoch_k', prec=1, style="bright_blue"))
            stoch_text_group.append("/"); stoch_text_group.append(format_indicator_value('stoch_d', prec=1, style="blue"))
            stoch_text_group.append("/"); stoch_text_group.append(format_indicator_value('stoch_k_prev', prec=1, style="dim blue")) # Display Previous K
            if indicators_data.get('stoch_kd_bullish'): stoch_text_group.append(" [b green]▲ BullX[/]", style="green") # Bullish K/D cross
            elif indicators_data.get('stoch_kd_bearish'): stoch_text_group.append(" [b red]▼ BearX[/]", style="red")   # Bearish K/D cross
            indicator_parts_text.append(stoch_text_group)

            # ATR
            atr_period_val = indicators_data.get('atr_period', self.config.atr_period) # Get ATR period used
            atr_text_group = Text(f"ATR({atr_period_val}): ")
            atr_text_group.append(format_indicator_value('atr', prec=price_display_dp + 1, style="bright_magenta")) # ATR often needs more precision
            indicator_parts_text.append(atr_text_group)

            # ADX (+DI, -DI)
            adx_text_group = Text(f"ADX({self.config.adx_period}): ")
            adx_val_dec = indicators_data.get('adx') if isinstance(indicators_data.get('adx'), Decimal) else safe_decimal(indicators_data.get('adx'))
            adx_style = "yellow" if not adx_val_dec.is_nan() and adx_val_dec > self.config.min_adx_level else "dim yellow"
            adx_text_group.append(self._format_decimal_for_rich(adx_val_dec, precision=1, default_style=adx_style))
            adx_text_group.append(" [+DI:", style="dim"); adx_text_group.append(format_indicator_value('pdi', prec=1, style="bright_green"))
            adx_text_group.append(" -DI:", style="dim"); adx_text_group.append(format_indicator_value('mdi', prec=1, style="bright_red"))
            adx_text_group.append("]", style="dim")
            indicator_parts_text.append(adx_text_group)

            # Join indicator parts with a separator
            separator_text = Text(" | ", style="dim")
            for i, part_text in enumerate(indicator_parts_text):
                panel_content.append(part_text)
                if i < len(indicator_parts_text) - 1: panel_content.append(separator_text)
            panel_content.append("\n")
        else: # Indicators not available
            panel_content.append(Text("Calculating indicators or data unavailable...", style="dim"))
            panel_content.append("\n")
        panel_content.append("---\n", style="dim")

        # --- Position Display ---
        panel_content.append("Position: ", style="bold bright_cyan")
        position_display_text = Text("FLAT", style="bold bright_green") # Default if no position
        active_pos_side_logical: Optional[str] = None
        active_pos_details: Optional[Dict] = None

        if current_positions_summary: # Check if position data is available
             long_pos = current_positions_summary.get("long", {})
             short_pos = current_positions_summary.get("short", {})
             if long_pos and safe_decimal(long_pos.get('qty', Decimal(0))).copy_abs() >= POSITION_QTY_EPSILON:
                 active_pos_details = long_pos; active_pos_side_logical = "long"
             elif short_pos and safe_decimal(short_pos.get('qty', Decimal(0))).copy_abs() >= POSITION_QTY_EPSILON:
                 active_pos_details = short_pos; active_pos_side_logical = "short"

        if active_pos_details and active_pos_side_logical: # If a position is active
            pos_color_style = "bold bright_green" if active_pos_side_logical == "long" else "bold bright_red"
            position_display_text = Text(f"{active_pos_side_logical.upper()}: ", style=pos_color_style)
            # Qty
            qty_text = self._format_decimal_for_rich(active_pos_details.get("qty"), precision=amount_display_dp)
            position_display_text.append("Qty=", style=pos_color_style); position_display_text.append(qty_text)
            # Entry Price
            entry_px_text = self._format_decimal_for_rich(active_pos_details.get("entry_price"), precision=price_display_dp)
            position_display_text.append(" | EntryPx=", style="dim"); position_display_text.append(entry_px_text)
            # Unrealized PnL
            pnl_text = self._format_decimal_for_rich(active_pos_details.get("unrealized_pnl"), precision=4, highlight_negative=True)
            position_display_text.append(" | PnL=", style="dim"); position_display_text.append(pnl_text)

            # Protection Status (SL/TP/TSL from position data vs. local tracker)
            local_tracker_status = protection_status_tracker.get(active_pos_side_logical)
            # Get protection details from the position data itself (exchange state)
            sl_from_pos_data = active_pos_details.get("stop_loss_price")
            tp_from_pos_data = active_pos_details.get("take_profit_price")
            tsl_is_active_on_pos = active_pos_details.get("is_tsl_active", False)
            tsl_trigger_px_on_pos = active_pos_details.get("tsl_trigger_price") # Or TSL distance, depends on API

            position_display_text.append(" | Protection: ", style="dim")
            protection_status_rich_text = Text("None", style="dim"); protection_details_rich_text = Text("")

            if tsl_is_active_on_pos: # TSL is active on exchange
                 protection_status_rich_text = Text("TSL Active", style="bright_magenta")
                 tsl_trigger_text = self._format_decimal_for_rich(tsl_trigger_px_on_pos, precision=price_display_dp)
                 protection_details_rich_text = Text(" (TrigPx:", style="dim").append(tsl_trigger_text).append(")", style="dim")
                 if local_tracker_status != "ACTIVE_TSL": # Mismatch with local tracker
                     protection_status_rich_text.append(" [TrackerMismatch?]", style="bright_yellow")
            elif sl_from_pos_data or tp_from_pos_data: # Fixed SL/TP active on exchange
                 protection_status_rich_text = Text("SL/TP Active", style="bright_yellow")
                 sl_text = self._format_decimal_for_rich(sl_from_pos_data, precision=price_display_dp) if sl_from_pos_data else Text("N/A", style="dim")
                 tp_text = self._format_decimal_for_rich(tp_from_pos_data, precision=price_display_dp) if tp_from_pos_data else Text("N/A", style="dim")
                 protection_details_rich_text = Text(" (S:", style="dim").append(sl_text).append(" T:", style="dim").append(tp_text).append(")", style="dim")
                 if local_tracker_status != "ACTIVE_SLTP": # Mismatch
                     protection_status_rich_text.append(" [TrackerMismatch?]", style="bright_yellow")
            elif local_tracker_status: # Local tracker thinks protection is active, but exchange shows none
                 protection_status_rich_text = Text(f"Tracked:{local_tracker_status}", style="yellow")
                 protection_details_rich_text = Text(" (Exchange:None?)", style="dim")

            position_display_text.append(protection_status_rich_text); position_display_text.append(protection_details_rich_text)

        panel_content.append(position_display_text); panel_content.append("\n")
        panel_content.append("---\n", style="dim")

        # --- Trading Signals Display ---
        panel_content.append("Signal: ", style="bold bright_cyan")
        signal_reason_str = trading_signals.get("reason", Text("No signal info available", style="dim"))
        signal_text_style = "dim" # Default style for signal reason
        if trading_signals.get("long"): signal_text_style = "bold bright_green"
        elif trading_signals.get("short"): signal_text_style = "bold bright_red"
        elif "Blocked" in signal_reason_str: signal_text_style = "yellow"
        elif "No Signal:" not in signal_reason_str and "Initializing" not in signal_reason_str:
            signal_text_style = "white" # If reason is not a standard "no signal" or "blocked"

        # Wrap long signal reasons for better display
        wrapped_signal_reason = "\n        ".join(textwrap.wrap(signal_reason_str, width=100)) # Indent wrapped lines
        panel_content.append(Text(wrapped_signal_reason, style=signal_text_style))

        # Print the assembled panel to the console
        console.print(
            Panel(
                panel_content,
                title=f"[bold bright_magenta]{panel_title_str}[/]",
                border_style="bright_blue", # Neon blue border
                expand=False, # Panel should not expand to full width
                padding=(1, 2) # Padding inside the panel (vertical, horizontal)
            )
        )


# --- Trading Bot Class ---
class TradingBot:
    """Main orchestrator class for the Pyrmethus trading bot."""

    def __init__(self):
        logger.info(
            f"{Style.BRIGHT}{Fore.MAGENTA}--- Initializing Pyrmethus v4.5.7 (Neon Nexus Edition) ---{Style.RESET_ALL}"
        )
        self.config = TradingConfig() # Load configuration first
        # Initialize core components, handling potential critical failures
        try:
            self.exchange_manager = ExchangeManager(self.config)
            if not self.exchange_manager.exchange or not self.exchange_manager.market_info:
                # This check is somewhat redundant if ExchangeManager's init sys.exits, but good safeguard
                logger.critical(f"{Style.BRIGHT}{Fore.RED}TradingBot initialization failed: ExchangeManager did not initialize correctly or failed to load market info. Halting.{Style.RESET_ALL}")
                sys.exit(1)

            self.indicator_calculator = IndicatorCalculator(self.config)
            self.signal_generator = SignalGenerator(self.config)
            # OrderManager depends on a fully initialized ExchangeManager
            self.order_manager = OrderManager(self.config, self.exchange_manager)
        except ValueError as ve: # Catch specific ValueError from OrderManager if deps are missing
            logger.critical(f"{Style.BRIGHT}{Fore.RED}TradingBot initialization failed due to OrderManager error: {ve}. Halting.{Style.RESET_ALL}")
            sys.exit(1)
        except Exception as e: # Catch any other unexpected error during component init
            logger.critical(f"{Style.BRIGHT}{Fore.RED}Unexpected critical error during TradingBot component initialization: {e}. Halting.{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)

        self.status_display = StatusDisplay(self.config)
        self.shutdown_requested = False # Flag for graceful shutdown
        self._setup_signal_handlers() # Setup SIGINT (Ctrl+C) and SIGTERM handlers
        logger.info(f"{Style.BRIGHT}{Fore.GREEN}Pyrmethus components initialized successfully. Ready to conjure trades.{Style.RESET_ALL}")

    def _setup_signal_handlers(self):
        """Sets up OS signal handlers for graceful shutdown on SIGINT and SIGTERM."""
        try:
            signal.signal(signal.SIGINT, self._signal_handler_callback) # Handle Ctrl+C
            signal.signal(signal.SIGTERM, self._signal_handler_callback) # Handle kill/system shutdown signals
            logger.debug("Signal handlers for SIGINT and SIGTERM successfully set up.")
        except (ValueError, OSError, AttributeError, Exception) as e: # Catch potential errors (e.g., in threads, or unsupported OS calls)
             logger.warning(f"{Fore.YELLOW}Could not set all OS signal handlers (OS/threading limitation?): {e}{Style.RESET_ALL}")

    def _signal_handler_callback(self, sig_num: int, frame: Optional[Any]):
        """Internal callback for OS signals to initiate graceful shutdown."""
        if not self.shutdown_requested: # Process only the first shutdown signal
            # Get signal name if possible, otherwise use number
            sig_name = signal.Signals(sig_num).name if isinstance(sig_num, int) and sig_num in signal.Signals else f"Signal {sig_num}"
            console.print(f"\n[bold yellow]Signal {sig_name} received. Initiating graceful shutdown sequence... Please wait.[/]")
            logger.warning(f"Signal {sig_name} received. Initiating graceful shutdown...")
            self.shutdown_requested = True
        else: # Shutdown already in progress
            logger.warning("Shutdown sequence already in progress. Ignoring additional signal.")

    def run(self):
        """Starts the main trading loop and handles overall execution flow."""
        self._display_startup_info() # Print initial configuration summary
        termux_notify(f"Pyrmethus Started", f"Trading {self.config.symbol} on {self.config.interval} interval.")
        cycle_count = 0

        while not self.shutdown_requested:
            cycle_count += 1
            cycle_start_timestamp = time.monotonic()
            logger.debug(f"{Fore.BLUE}--- Starting Trading Cycle {cycle_count} ---{Style.RESET_ALL}")

            try:
                self.trading_spell_cycle(cycle_count) # Execute one full trading cycle
            except KeyboardInterrupt: # Should be caught by signal handler, but as safety
                logger.warning("\nKeyboardInterrupt detected directly in main loop. Initiating shutdown.")
                self.shutdown_requested = True; break # Exit loop
            except ccxt.AuthenticationError as auth_err: # Critical auth error during cycle
                logger.critical(f"{Style.BRIGHT}{Fore.RED}CRITICAL AUTHENTICATION ERROR in cycle {cycle_count}: {auth_err}. Halting Pyrmethus immediately.{Style.RESET_ALL}", exc_info=False)
                termux_notify("Pyrmethus CRITICAL ERROR", f"Authentication failed: {str(auth_err)[:100]}")
                self.shutdown_requested = True; break # Exit loop
            except SystemExit as se: # If a component calls sys.exit
                 logger.warning(f"SystemExit called with code {se.code} during trading cycle. Terminating Pyrmethus.")
                 self.shutdown_requested = True; break # Exit loop
            except Exception as cycle_err: # Catch any other unhandled exception from trading_spell_cycle
                logger.error(f"{Style.BRIGHT}{Fore.RED}Unhandled exception in main trading cycle {cycle_count}: {cycle_err}{Style.RESET_ALL}", exc_info=True)
                termux_notify("Pyrmethus Cycle Error", f"Exception in cycle {cycle_count}. Check logs for details.")
                # After an unhandled error, might sleep longer before retrying cycle
                sleep_duration_after_error = self.config.loop_sleep_seconds * 2
                logger.info(f"Sleeping for an extended period ({sleep_duration_after_error}s) after cycle error.")
                time.sleep(sleep_duration_after_error) # Simpler sleep here, signal handler will catch interrupt
                continue # Continue to next cycle iteration

            # Calculate sleep duration for this cycle
            cycle_duration_seconds = time.monotonic() - cycle_start_timestamp
            sleep_needed_seconds = max(0, self.config.loop_sleep_seconds - cycle_duration_seconds)
            logger.debug(f"Cycle {cycle_count} completed in {cycle_duration_seconds:.2f}s.")

            # Interruptible Sleep: Allows shutdown requests to be processed during sleep
            if not self.shutdown_requested and sleep_needed_seconds > 0:
                logger.debug(f"Sleeping for {sleep_needed_seconds:.2f} seconds before next cycle...")
                sleep_end_time = time.monotonic() + sleep_needed_seconds
                try:
                    while time.monotonic() < sleep_end_time and not self.shutdown_requested:
                        # Sleep in small chunks to allow periodic check of shutdown_requested flag
                        time.sleep(min(0.5, sleep_needed_seconds)) # Check every 0.5s or less
                except KeyboardInterrupt: # Catch Ctrl+C during sleep
                    logger.warning("\nKeyboardInterrupt detected during sleep period. Initiating shutdown.")
                    self.shutdown_requested = True # Set flag to ensure loop terminates

            if self.shutdown_requested:
                logger.info("Shutdown requested. Exiting main trading loop.")
                break # Break from while loop

        # --- End of Main Loop ---
        self.graceful_shutdown() # Perform cleanup actions
        console.print(f"\n[bold bright_cyan]Pyrmethus ({self.config.symbol}) has completed its session and returned to the ether.[/]")
        sys.exit(0) # Explicitly exit with success code

    def trading_spell_cycle(self, cycle_num: int) -> None:
        """
        Executes one complete cycle of the trading logic:
        Fetch data -> Calculate Indicators -> Get Balances/Positions -> Manage TSL ->
        Generate Signals -> Check Exits -> Execute Entries -> Display Status.
        """
        cycle_start_time_monotonic = time.monotonic()
        current_cycle_status_msg = "OK" # Default status for the cycle

        # 1. Fetch Market Data (OHLCV)
        logger.debug("Fetching latest market data (OHLCV)...")
        ohlcv_df = self.exchange_manager.fetch_ohlcv()
        if ohlcv_df is None or ohlcv_df.empty:
            logger.error(f"{Fore.RED}Cycle {cycle_num} Aborted: Failed to fetch or process valid OHLCV market data.{Style.RESET_ALL}")
            current_cycle_status_msg = "FAIL:FETCH_OHLCV_DATA"
            # Display status even on failure, showing the error
            self.status_display.print_status_panel(cycle_num, None, None, None, None, None, {"reason": current_cycle_status_msg}, {}, self.exchange_manager.market_info)
            return # End this cycle attempt

        # Extract latest price and timestamp from fetched data
        try:
            latest_candle_data = ohlcv_df.iloc[-1]
            current_market_price = safe_decimal(latest_candle_data["close"])
            last_candle_timestamp = ohlcv_df.index[-1].to_pydatetime() # Convert Pandas Timestamp to Python datetime
            if current_market_price.is_nan() or current_market_price <= 0:
                raise ValueError(f"Invalid latest close price from OHLCV: {current_market_price.normalize() if not current_market_price.is_nan() else 'NaN'}")
            logger.debug(f"Latest Candle Data: Timestamp={last_candle_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}, ClosePrice={current_market_price.normalize()}")
        except (IndexError, KeyError, ValueError, TypeError) as e:
            logger.error(f"{Fore.RED}Cycle {cycle_num} Aborted: Error processing latest candle data: {e}{Style.RESET_ALL}", exc_info=False)
            current_cycle_status_msg = "FAIL:PROCESS_LATEST_CANDLE"
            self.status_display.print_status_panel(cycle_num, None, None, None, None, None, {"reason": current_cycle_status_msg}, {}, self.exchange_
