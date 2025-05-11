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

Conjures market insights and executes trades on Bybit Futures (V5 Unified Account API)
via CCXT. Structured with classes for improved organization and leverages Bybit's
V5 position-based protection features (Stop-Loss, Take-Profit, Trailing Stop).
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

    # List of common packages for consolidated installation instructions
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
    # Initialize colorama for error message formatting if its import failed
    if e.name == "colorama": # Special handling if colorama itself is missing
        print("Missing essential package: colorama. Cannot display colored output.")
        print("Attempting basic error message...")
        print(f"Missing essential dependency: {e.name}")
        print(f"To install it, run: pip install {e.name}")
        print("\nOr, to ensure all dependencies are present, run:")
        print(f"pip install {' '.join(COMMON_PACKAGES)}")
        sys.exit(1)
    else: # Colorama is available, use it for a nicer error message
        colorama_init(autoreset=True)
        missing_pkg = e.name
        print(
            f"{Fore.RED}{Style.BRIGHT}Missing essential dependency: {Style.BRIGHT}{missing_pkg}{Style.NORMAL}"
        )
        print(
            f"{Fore.YELLOW}To install it, run: {Style.BRIGHT}pip install {missing_pkg}{Style.RESET_ALL}"
        )
        print(f"\n{Fore.CYAN}Or, to ensure all dependencies are present:")

        # Termux-specific installation advice for numpy and pandas
        is_termux = os.getenv("TERMUX_VERSION") or "com.termux" in os.environ.get("PREFIX", "")
        if is_termux:
            termux_native_pkgs = {"pandas": "python-pandas", "numpy": "python-numpy"}
            pkg_install_cmds = []
            pip_install_pkgs = list(COMMON_PACKAGES) # Start with all packages for pip

            termux_specific_installs = []
            for common_pkg, termux_pkg_name in termux_native_pkgs.items():
                if common_pkg in pip_install_pkgs:
                    termux_specific_installs.append(termux_pkg_name)
                    pip_install_pkgs.remove(common_pkg) # Remove from pip list if handled by pkg

            if termux_specific_installs:
                pkg_install_cmds.append(f"pkg install python {' '.join(termux_specific_installs)}")
            if pip_install_pkgs:
                 pkg_install_cmds.append(f"pip install {' '.join(pip_install_pkgs)}")

            install_cmd = " && ".join(pkg_install_cmds) if pkg_install_cmds else f"pip install {' '.join(COMMON_PACKAGES)}"
            print(f"{Style.BRIGHT}{install_cmd}{Style.RESET_ALL}")
            print(
                f"{Fore.YELLOW}Note: In Termux, pandas and numpy are often best installed via 'pkg' for compatibility.{Style.RESET_ALL}"
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
V5_UNIFIED_ACCOUNT_TYPE = "UNIFIED" # Bybit V5 Unified Account identifier
V5_HEDGE_MODE_POSITION_IDX = 0 # Default index for position mode (0=One-Way, 1=Buy Hedge, 2=Sell Hedge)
V5_TPSL_MODE_FULL = "Full" # Apply SL/TP to the entire position for Bybit V5
V5_SUCCESS_RETCODE = 0     # Standard success return code for Bybit V5 API
TERMUX_NOTIFY_TIMEOUT = 10 # Seconds for termux-toast command timeout

# Initialize Colorama & Rich Console
colorama_init(autoreset=True)
console = Console(log_path=False) # Disable Rich's own log file handling

# Set Decimal precision context globally
getcontext().prec = DECIMAL_PRECISION

# --- Logging Setup ---
TRADE_LEVEL_NUM = 25  # Custom logging level between INFO (20) and WARNING (30)
if not hasattr(logging.Logger, "trade"): # Ensure 'trade' method isn't already defined
    logging.addLevelName(TRADE_LEVEL_NUM, "TRADE")
    def trade_log(self, message, *args, **kws):
        """Logs a message with custom level TRADE."""
        if self.isEnabledFor(TRADE_LEVEL_NUM):
            # pylint: disable=protected-access
            self._log(TRADE_LEVEL_NUM, message, args, **kws)
    logging.Logger.trade = trade_log # type: ignore[attr-defined]

logger = logging.getLogger(__name__) # Logger for this module
log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)-8s] (%(filename)s:%(lineno)d) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
valid_log_levels = ["DEBUG", "INFO", "TRADE", "WARNING", "ERROR", "CRITICAL"]
log_level_to_set = logging.INFO # Default

if log_level_str.isdigit() and int(log_level_str) == TRADE_LEVEL_NUM:
    log_level_to_set = TRADE_LEVEL_NUM
elif log_level_str in valid_log_levels:
    log_level_to_set = getattr(logging, log_level_str)
else:
    # Early print as logger isn't fully set up for this warning
    print(f"Warning: Invalid LOG_LEVEL '{log_level_str}'. Defaulting to INFO.")
    log_level_str = "INFO" # For display later if needed

logger.setLevel(log_level_to_set)

if not logger.hasHandlers(): # Add handler only once
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
logger.propagate = False # Prevent logs from reaching the root logger

# --- Utility Functions ---
def safe_decimal(
    value: Any, default: Decimal = Decimal("NaN")
) -> Decimal:
    """Safely converts a value to Decimal, handling None, empty strings, and common invalid formats."""
    if value is None:
        return default
    try:
        str_value = str(value).strip()
        if not str_value:
            return default
        if str_value.lower() in ["nan", "none", "null"]: # Common non-numeric representations
            return default
        return Decimal(str_value)
    except (InvalidOperation, ValueError, TypeError):
        # logger.debug(f"Could not convert '{value}' (type: {type(value).__name__}) to Decimal, using default {default}")
        return default

def termux_notify(title: str, content: str) -> None:
    """Sends a notification via Termux API (toast), if available. Title is ignored by termux-toast."""
    if "com.termux" in os.environ.get("PREFIX", ""): # Check for Termux environment
        try:
            # termux-toast only uses the content argument; title is effectively ignored.
            result = subprocess.run(
                ["termux-toast", content],
                check=False, # Manually handle non-zero exit codes
                timeout=TERMUX_NOTIFY_TIMEOUT,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                error_output = result.stderr.strip() if result.stderr else result.stdout.strip()
                logger.warning(
                    f"Termux toast command failed (code {result.returncode}): {error_output}"
                )
            # logger.debug(f"Termux toast sent: '{content}' (Title '{title}' ignored by toast)")
        except FileNotFoundError:
            logger.warning(
                "Termux notify failed: 'termux-toast' command not found. Ensure Termux:API is installed and setup."
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"Termux notify failed: command timed out after {TERMUX_NOTIFY_TIMEOUT} seconds.")
        except Exception as e:
            logger.warning(f"Termux notify failed unexpectedly: {e}")
    # else: logger.debug("Not in Termux environment, skipping notification.")

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
        ccxt.AuthenticationError, ccxt.PermissionDenied, # Halt immediately
    ),
    fail_fast_exceptions: Tuple[Type[Exception], ...] = (
         ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.OrderNotFound, # No point retrying
    ),
    **kwargs: Any,
) -> Any:
    """Wraps a function call with enhanced retry logic and specific error handling."""
    last_exception: Optional[Exception] = None
    func_name = getattr(fetch_function, "__name__", "Unnamed function")

    for attempt in range(max_retries + 1): # Total attempts = initial + max_retries
        try:
            result = fetch_function(*args, **kwargs)
            if attempt > 0: # Log success only if it was a successful retry
                logger.info(f"{Style.BRIGHT}{Fore.GREEN}Successfully executed {func_name} on attempt {attempt + 1}/{max_retries + 1}.{Style.RESET_ALL}")
            return result
        except fatal_exceptions as e:
            logger.critical(f"{Style.BRIGHT}{Fore.RED}Fatal error ({type(e).__name__}) executing {func_name}: {e}. Halting immediately.{Style.RESET_ALL}", exc_info=False)
            raise # Re-raise critical error
        except fail_fast_exceptions as e:
            logger.error(f"{Fore.RED}Fail-fast error ({type(e).__name__}) executing {func_name}: {e}. Not retrying.{Style.RESET_ALL}")
            last_exception = e
            break # Break loop, don't retry
        except retry_on_exceptions as e:
            last_exception = e
            error_summary = str(e)[:150] + "..." if len(str(e)) > 150 else str(e)
            retry_msg_base = f"{Fore.YELLOW}Retryable error ({type(e).__name__}) on attempt {attempt + 1}/{max_retries + 1} for {func_name}: {error_summary}.{Style.RESET_ALL}"
            if attempt < max_retries:
                logger.warning(f"{retry_msg_base} Retrying in {delay_seconds}s...")
                time.sleep(delay_seconds)
            else:
                logger.error(f"{Fore.RED}Max retries ({max_retries + 1}) reached for {func_name}. Last error: {e}{Style.RESET_ALL}")
        except ccxt.ExchangeError as e: # Catch other generic (but potentially retryable) exchange errors
            last_exception = e
            logger.error(f"{Fore.RED}Unhandled ExchangeError during {func_name}: {e}{Style.RESET_ALL}")
            if attempt < max_retries:
                logger.warning(f"Retrying generic exchange error in {delay_seconds}s...")
                time.sleep(delay_seconds)
            else:
                logger.error(f"Max retries reached after generic exchange error for {func_name}.")
                break
        except Exception as e: # Catch truly unexpected errors
            last_exception = e
            logger.error(f"{Fore.RED}Unexpected error during {func_name}: {e}{Style.RESET_ALL}", exc_info=True) # Include stack trace
            break # Don't retry unknown errors

    if last_exception:
        raise last_exception
    # This path should ideally not be reached if logic is sound (e.g. max_retries=0 and first attempt fails without matching exception)
    raise RuntimeError(f"Function {func_name} failed after {max_retries + 1} attempts without raising a recognized or captured exception.")

# --- Configuration Class ---
class TradingConfig:
    """Loads, validates, and stores trading configuration parameters."""
    # pylint: disable=too-many-statements
    def __init__(self, env_file: str = ".env"):
        logger.debug(f"Loading configuration from environment variables / '{env_file}'...")
        env_path = Path(env_file)
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path, override=True)
            logger.info(f"Loaded configuration from {env_path}")
        else:
            logger.warning(f"Environment file '{env_path}' not found. Relying on system environment variables.")

        # Core Trading Parameters
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT", Style.DIM)
        self.market_type: str = self._get_env(
            "MARKET_TYPE", "linear", Style.DIM, allowed_values=["linear", "inverse", "swap"]
        ).lower()
        self.bybit_v5_category: str = self._determine_v5_category() # Depends on symbol and market_type
        self.interval: str = self._get_env("INTERVAL", "1m", Style.DIM)

        # Financial Parameters (Decimal for precision)
        self.risk_percentage: Decimal = self._get_env(
            "RISK_PERCENTAGE", DEFAULT_RISK_PERCENT, Fore.YELLOW, cast_type=Decimal,
            min_val=Decimal("0.00001"), max_val=Decimal("0.5") # 0.001% to 50% risk
        )
        self.sl_atr_multiplier: Decimal = self._get_env(
            "SL_ATR_MULTIPLIER", DEFAULT_SL_MULT, Fore.YELLOW, cast_type=Decimal,
            min_val=Decimal("0.1"), max_val=Decimal("20.0")
        )
        self.tp_atr_multiplier: Decimal = self._get_env(
            "TP_ATR_MULTIPLIER", DEFAULT_TP_MULT, Fore.YELLOW, cast_type=Decimal,
            min_val=Decimal("0.0"), max_val=Decimal("50.0") # TP=0 disables ATR-based TP
        )
        self.tsl_activation_atr_multiplier: Decimal = self._get_env(
            "TSL_ACTIVATION_ATR_MULTIPLIER", DEFAULT_TSL_ACT_MULT, Fore.YELLOW, cast_type=Decimal,
            min_val=Decimal("0.1"), max_val=Decimal("20.0")
        )
        self.trailing_stop_percent: Decimal = self._get_env(
            "TRAILING_STOP_PERCENT", DEFAULT_TSL_PERCENT, Fore.YELLOW, cast_type=Decimal,
            min_val=Decimal("0.001"), max_val=Decimal("10.0") # 0.1% to 10% TSL
        )

        # V5 Position Stop Parameters
        self.sl_trigger_by: str = self._get_env(
            "SL_TRIGGER_BY", "LastPrice", Style.DIM, allowed_values=["LastPrice", "MarkPrice", "IndexPrice"]
        )
        self.tsl_trigger_by: str = self._get_env(
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
            min_val=Decimal("0"), max_val=Decimal("5") # Buffer as % of trend EMA
        )
        self.atr_move_filter_multiplier: Decimal = self._get_env(
            "ATR_MOVE_FILTER_MULTIPLIER", Decimal("0.5"), Fore.CYAN, cast_type=Decimal,
            min_val=Decimal("0"), max_val=Decimal("5") # Multiplier for ATR; 0 disables
        )
        self.min_adx_level: Decimal = self._get_env(
            "MIN_ADX_LEVEL", DEFAULT_MIN_ADX, Fore.CYAN, cast_type=Decimal,
            min_val=Decimal("0"), max_val=Decimal("90")
        )

        # API Keys (Secrets)
        self.api_key: str = self._get_env("BYBIT_API_KEY", None, Fore.RED, is_secret=True)
        self.api_secret: str = self._get_env("BYBIT_API_SECRET", None, Fore.RED, is_secret=True)

        # Operational Parameters
        self.ohlcv_limit: int = self._get_env("OHLCV_LIMIT", DEFAULT_OHLCV_LIMIT, Style.DIM, cast_type=int, min_val=50, max_val=1000)
        self.loop_sleep_seconds: int = self._get_env("LOOP_SLEEP_SECONDS", DEFAULT_LOOP_SLEEP, Style.DIM, cast_type=int, min_val=1)
        self.order_check_delay_seconds: int = self._get_env("ORDER_CHECK_DELAY_SECONDS", 2, Style.DIM, cast_type=int, min_val=1)
        self.order_fill_timeout_seconds: int = self._get_env(
            "ORDER_FILL_TIMEOUT_SECONDS", 20, Style.DIM, cast_type=int, min_val=5
        ) # Implicitly used by number of verification attempts
        self.max_fetch_retries: int = self._get_env("MAX_FETCH_RETRIES", DEFAULT_MAX_RETRIES, Style.DIM, cast_type=int, min_val=0, max_val=10)
        self.retry_delay_seconds: int = self._get_env("RETRY_DELAY_SECONDS", DEFAULT_RETRY_DELAY, Style.DIM, cast_type=int, min_val=1)
        self.trade_only_with_trend: bool = self._get_env("TRADE_ONLY_WITH_TREND", True, Style.DIM, cast_type=bool)

        # Journaling
        self.journal_file_path: str = self._get_env("JOURNAL_FILE_PATH", DEFAULT_JOURNAL_FILE, Style.DIM)
        self.enable_journaling: bool = self._get_env("ENABLE_JOURNALING", True, Style.DIM, cast_type=bool)

        self._validate_config() # Cross-parameter validations
        logger.debug("Configuration loaded and validated successfully.")

    def _determine_v5_category(self) -> str:
        """Determines the Bybit V5 API category based on symbol and market type."""
        category: str
        try:
            # CCXT symbol format: BASE/QUOTE:SETTLE (e.g., BTC/USDT:USDT for USDT linear)
            # or BASE/QUOTE (e.g., BTC/USD for BTC inverse, settlement implied or via market_type)
            if ":" not in self.symbol:
                logger.warning(
                    f"Symbol '{self.symbol}' does not explicitly state settle currency. "
                    f"Inferring category from MARKET_TYPE ('{self.market_type}')."
                )
                if self.market_type == "inverse": category = "inverse" # e.g., BTC/USD (settled in BTC)
                elif self.market_type in ["linear", "swap"]: category = "linear" # e.g., BTC/USDT (settled in USDT)
                else: raise ValueError(f"Unsupported MARKET_TYPE '{self.market_type}' for category determination.")
            else: # Symbol includes settle currency (e.g., BTC/USDT:USDT)
                if self.market_type == "inverse": category = "inverse" # e.g., BTC/USD:BTC
                elif self.market_type in ["linear", "swap"]: category = "linear" # e.g., BTC/USDT:USDT
                else: raise ValueError(f"Unsupported MARKET_TYPE '{self.market_type}' for category determination.")

            logger.info(
                f"Determined Bybit V5 API category: '{category}' for symbol '{self.symbol}', type '{self.market_type}'"
            )
            return category
        except ValueError as e:
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Could not determine V5 category: {e}. Halting.{Style.RESET_ALL}",
                exc_info=True,
            )
            sys.exit(1)

    def _validate_config(self):
        """Performs post-load validation of related configuration parameters."""
        if self.fast_ema_period >= self.slow_ema_period:
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Validation Error: FAST_EMA_PERIOD ({self.fast_ema_period}) "
                f"must be less than SLOW_EMA_PERIOD ({self.slow_ema_period}). Halting.{Style.RESET_ALL}"
            )
            sys.exit(1)
        if self.trend_ema_period <= self.slow_ema_period:
            logger.warning(
                f"{Fore.YELLOW}Config Warning: TREND_EMA_PERIOD ({self.trend_ema_period}) is not greater than "
                f"SLOW_EMA_PERIOD ({self.slow_ema_period}). Trend filter might lag short-term signals.{Style.RESET_ALL}"
            )
        if self.stoch_oversold_threshold >= self.stoch_overbought_threshold:
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Validation Error: STOCH_OVERSOLD_THRESHOLD ({self.stoch_oversold_threshold.normalize()}) "
                f"must be less than STOCH_OVERBOUGHT_THRESHOLD ({self.stoch_overbought_threshold.normalize()}). Halting.{Style.RESET_ALL}"
            )
            sys.exit(1)
        if self.tsl_activation_atr_multiplier < self.sl_atr_multiplier:
            logger.warning(
                f"{Fore.YELLOW}Config Warning: TSL_ACTIVATION_ATR_MULTIPLIER ({self.tsl_activation_atr_multiplier.normalize()}) "
                f"is less than SL_ATR_MULTIPLIER ({self.sl_atr_multiplier.normalize()}). "
                f"TSL may activate before initial SL distance is fully established.{Style.RESET_ALL}"
            )
        if self.tp_atr_multiplier > Decimal("0") and self.tp_atr_multiplier <= self.sl_atr_multiplier:
            logger.warning(
                f"{Fore.YELLOW}Config Warning: TP_ATR_MULTIPLIER ({self.tp_atr_multiplier.normalize()}) "
                f"is less than or equal to SL_ATR_MULTIPLIER ({self.sl_atr_multiplier.normalize()}). "
                f"This implies a Risk:Reward ratio of 1:1 or less.{Style.RESET_ALL}"
            )

    def _cast_value(self, key: str, value_str: str, cast_type: Type, default: Any) -> Any:
        """Helper to cast string value to target type, returning default on failure."""
        val_to_cast = value_str.strip()
        if not val_to_cast: # Handle empty string after strip
            logger.warning(f"Empty value for '{key}' after stripping. Using default '{default}'.")
            return default
        try:
            if cast_type == bool:
                return val_to_cast.lower() in ["true", "1", "yes", "y", "on"]
            if cast_type == Decimal:
                if val_to_cast.lower() in ["nan", "none", "null"]:
                    raise ValueError("Non-numeric string cannot be cast to Decimal")
                return Decimal(val_to_cast)
            if cast_type == int:
                dec_val = Decimal(val_to_cast) # Use Decimal for robust int conversion
                if dec_val.as_tuple().exponent < 0: # Check for fractional part
                    raise ValueError("Decimal value with fractional part cannot be cast to int.")
                return int(dec_val)
            return cast_type(val_to_cast) # Includes str
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.error(
                f"{Fore.RED}Cast failed for '{key}' (value: '{value_str}', type: {cast_type.__name__}): {e}. "
                f"Using default '{default}'.{Style.RESET_ALL}"
            )
            return default

    def _validate_value(
        self, key: str, value: Any,
        min_val: Optional[Union[int, float, Decimal]],
        max_val: Optional[Union[int, float, Decimal]],
        allowed_values: Optional[List[Any]]
    ) -> bool:
        """Validates a value against constraints. Critical failures (min/max) exit."""
        is_numeric = isinstance(value, (int, float, Decimal))
        if (min_val is not None or max_val is not None) and not is_numeric:
            logger.error(f"Validation Error for '{key}': Non-numeric value '{value}' (type: {type(value).__name__}) cannot be compared with min/max.")
            return False # Cannot perform min/max validation

        if min_val is not None and is_numeric and value < min_val: # type: ignore
            logger.critical(f"{Style.BRIGHT}{Fore.RED}Validation Error for '{key}': Value '{value}' < min '{min_val}'. Halting.{Style.RESET_ALL}")
            sys.exit(1)
        if max_val is not None and is_numeric and value > max_val: # type: ignore
            logger.critical(f"{Style.BRIGHT}{Fore.RED}Validation Error for '{key}': Value '{value}' > max '{max_val}'. Halting.{Style.RESET_ALL}")
            sys.exit(1)

        if allowed_values:
            comp_value = str(value).lower() if isinstance(value, str) else value
            lower_allowed = [str(v).lower() if isinstance(v, str) else v for v in allowed_values]
            if comp_value not in lower_allowed:
                logger.error(f"{Fore.RED}Validation Error for '{key}': Invalid value '{value}'. Allowed: {allowed_values}.{Style.RESET_ALL}")
                return False
        return True

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
        """Fetches, casts, validates, and defaults environment variables."""
        value_str = os.getenv(key)
        source_desc: str
        use_default = False
        value_to_process: str

        if value_str is None or value_str.strip() == "": # Env var not set or is empty
            if default is None: # Required config with no default
                 severity_logger = logger.critical
                 error_msg = f"Required {'secret ' if is_secret else ''}configuration '{key}' not found and no default provided. Halting."
                 severity_logger(f"{Style.BRIGHT}{Fore.RED}{error_msg}{Style.RESET_ALL}")
                 sys.exit(1)

            use_default = True
            value_to_process = str(default) # Use string representation of default for casting
            source_desc = f"default ('{default}')"
            log_value_display = "****" if is_secret else default
        else:
            value_to_process = value_str
            source_desc = "environment variable"
            log_value_display = "****" if is_secret else value_to_process

        log_method = logger.warning if use_default and default is not None else logger.info
        log_method(f"Using {color}{key}: {log_value_display}{Style.RESET_ALL} (from {source_desc})")

        casted_value = self._cast_value(key, value_to_process, cast_type, default)

        if not self._validate_value(key, casted_value, min_val, max_val, allowed_values):
            # Validation failed (likely allowed_values or type error pre-min/max, as min/max exits)
            # Revert to the original default value.
            logger.warning(
                f"{color}Reverting '{key}' to its original default '{default}' due to validation failure of processed value '{casted_value}'.{Style.RESET_ALL}"
            )
            casted_value = default # Use the original default

            # Critical: Re-validate the original default value itself.
            if not self._validate_value(key, casted_value, min_val, max_val, allowed_values):
                logger.critical(
                    f"{Style.BRIGHT}{Fore.RED}FATAL: The hardcoded default value '{default}' for '{key}' "
                    f"failed validation. Halting.{Style.RESET_ALL}"
                )
                sys.exit(1)
        return casted_value

# --- Exchange Manager Class ---
class ExchangeManager:
    """Handles CCXT exchange interactions, data fetching, and market information."""
    def __init__(self, config: TradingConfig):
        self.config = config
        self.exchange: Optional[ccxt.Exchange] = None
        self.market_info: Optional[Dict[str, Any]] = None
        self._initialize_exchange()
        if self.exchange:
             self.market_info = self._load_market_info()
        # Critical errors during init would have called sys.exit.

    def _initialize_exchange(self):
        """Initializes the CCXT exchange instance for Bybit V5."""
        logger.info(f"Initializing Bybit V5 exchange (Market Type: {self.config.market_type})...")
        try:
            exchange_params: Dict[str, Any] = {
                "apiKey": self.config.api_key,
                "secret": self.config.api_secret,
                "options": {
                    "defaultType": self.config.market_type,
                    "adjustForTimeDifference": True,
                    "recvWindow": 10000, # Optional: Increased receive window
                    "brokerId": "TermuxNeonV5", # Custom broker ID for Bybit
                    "defaultTimeInForce": "GTC", # Good-Till-Cancelled
                },
            }
            if os.getenv("USE_BYBIT_TESTNET", "false").lower() == "true":
                logger.warning(f"{Fore.YELLOW}Using Bybit Testnet endpoint.{Style.RESET_ALL}")
                exchange_params['urls'] = {'api': 'https://api-testnet.bybit.com'}

            self.exchange = ccxt.bybit(exchange_params) # type: ignore
            logger.debug("Testing exchange connection by fetching server time...")
            self.exchange.fetch_time() # type: ignore
            logger.info(
                f"{Style.BRIGHT}{Fore.GREEN}Bybit V5 interface initialized and connection tested.{Style.RESET_ALL}"
            )
        except ccxt.AuthenticationError as e:
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Authentication failed: {e}. Check API keys/permissions. Halting.{Style.RESET_ALL}",
                exc_info=False,
            )
            sys.exit(1)
        except (ccxt.NetworkError, requests.exceptions.RequestException) as e:
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Network error initializing exchange: {e}. Check connection/endpoint. Halting.{Style.RESET_ALL}",
                exc_info=True,
            )
            sys.exit(1)
        except Exception as e:
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Unexpected error initializing exchange: {e}. Halting.{Style.RESET_ALL}",
                exc_info=True,
            )
            sys.exit(1)

    def _load_market_info(self) -> Optional[Dict[str, Any]]:
        """Loads and caches market information for the configured symbol."""
        if not self.exchange:
            logger.error("Exchange not initialized, cannot load market info.")
            return None
        try:
            logger.info(f"Loading market info for symbol: {self.config.symbol}...")
            self.exchange.load_markets(reload=True)
            market = self.exchange.market(self.config.symbol)
            if not market:
                raise ccxt.ExchangeError(f"Market {self.config.symbol} not found after loading markets.")

            # Helper to derive decimal places from CCXT precision (step size)
            def get_dp_from_precision(precision_val: Optional[Union[str, float, int]], default_dp: int) -> int:
                if precision_val is None: return default_dp
                prec_dec = safe_decimal(precision_val)
                if prec_dec.is_nan() or prec_dec <= 0: return default_dp
                # If precision is like 0.001, exponent gives negative of dp
                if prec_dec < 1: return abs(prec_dec.as_tuple().exponent)
                # If precision is like 1, 2 (meaning number of dps directly)
                if prec_dec.is_finite() and prec_dec == prec_dec.to_integral_value(rounding=ROUND_DOWN):
                    return int(prec_dec)
                return default_dp # Fallback

            amount_prec_raw = market.get("precision", {}).get("amount")
            price_prec_raw = market.get("precision", {}).get("price")
            amount_dp = get_dp_from_precision(amount_prec_raw, DEFAULT_AMOUNT_DP)
            price_dp = get_dp_from_precision(price_prec_raw, DEFAULT_PRICE_DP)

            market["precision_dp"] = {"amount": amount_dp, "price": price_dp}
            # Tick size is 10^(-price_dp) if price_prec_raw was like 0.01, or directly price_prec_raw
            market["tick_size"] = safe_decimal(price_prec_raw, Decimal("1e-" + str(DEFAULT_PRICE_DP)))
            if market["tick_size"].is_nan() or market["tick_size"] <= 0: # Ensure valid tick_size
                 market["tick_size"] = Decimal("1e-" + str(price_dp))


            min_amount_raw = market.get("limits", {}).get("amount", {}).get("min")
            market["min_order_size"] = safe_decimal(min_amount_raw, default=Decimal("NaN"))
            market["contract_size"] = safe_decimal(market.get("contractSize", "1"), default=Decimal("1"))

            min_amt_str = market["min_order_size"].normalize() if not market["min_order_size"].is_nan() else "N/A"
            logger.info(
                f"Market info for {self.config.symbol} (ID: {market.get('id')}): "
                f"Precision(AmountDP={amount_dp}, PriceDP={price_dp}), TickSize={market['tick_size'].normalize()}, "
                f"Limits(MinAmount={min_amt_str}), ContractSize={market['contract_size'].normalize()}"
            )
            return market
        except (ccxt.ExchangeError, KeyError, ValueError, TypeError, Exception) as e:
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Failed to load/parse market info for {self.config.symbol}: {e}. Halting.{Style.RESET_ALL}",
                exc_info=True,
            )
            sys.exit(1)
        return None # Should not be reached if sys.exit is called

    def format_price(self, price: Union[Decimal, str, float, int]) -> str:
        """Formats a price value to string using market precision (ROUND_HALF_EVEN)."""
        price_decimal = safe_decimal(price)
        if price_decimal.is_nan(): return "NaN"
        dp = self.market_info["precision_dp"]["price"] if self.market_info and "precision_dp" in self.market_info else DEFAULT_PRICE_DP
        try:
            quantizer = Decimal("1e-" + str(dp))
            return f"{price_decimal.quantize(quantizer, rounding=ROUND_HALF_EVEN):.{dp}f}"
        except (InvalidOperation, ValueError):
             logger.error(f"Error formatting price {price_decimal} to {dp}dp.")
             return "ERR"

    def format_amount(self, amount: Union[Decimal, str, float, int], rounding_mode=ROUND_DOWN) -> str:
        """Formats an amount (quantity) to string using market precision (default ROUND_DOWN)."""
        amount_decimal = safe_decimal(amount)
        if amount_decimal.is_nan(): return "NaN"
        dp = self.market_info["precision_dp"]["amount"] if self.market_info and "precision_dp" in self.market_info else DEFAULT_AMOUNT_DP
        try:
            quantizer = Decimal("1e-" + str(dp))
            return f"{amount_decimal.quantize(quantizer, rounding=rounding_mode):.{dp}f}"
        except (InvalidOperation, ValueError):
             logger.error(f"Error formatting amount {amount_decimal} to {dp}dp.")
             return "ERR"

    def _format_v5_param(
            self, value: Optional[Union[Decimal, str, float, int]],
            param_type: str = "price", # 'price', 'amount', or 'distance'
            allow_zero: bool = False
        ) -> Optional[str]:
        """Formats a numeric value as a string for Bybit V5 API parameters."""
        if value is None: return None
        decimal_value = safe_decimal(value, default=Decimal("NaN"))
        if decimal_value.is_nan():
            logger.warning(f"V5 Param Format: Input '{value}' (type: {type(value).__name__}) is NaN. Cannot format.")
            return None

        if decimal_value.is_zero():
            if allow_zero:
                formatter = self.format_price if param_type in ["price", "distance"] else self.format_amount
                formatted_zero = formatter(Decimal("0"))
                return formatted_zero if formatted_zero not in ["ERR", "NaN"] else None
            return None # Zero not allowed or formatting failed
        if decimal_value < 0:
            logger.warning(f"V5 Param Format: Input '{value}' is negative ({decimal_value}), invalid for API params.")
            return None

        if param_type in ["price", "distance"]: formatted_str = self.format_price(decimal_value)
        elif param_type == "amount": formatted_str = self.format_amount(decimal_value, rounding_mode=ROUND_DOWN)
        else:
            logger.error(f"V5 Param Format: Unknown param_type '{param_type}' for '{value}'.")
            return None

        if formatted_str in ["ERR", "NaN"]:
            logger.error(f"V5 Param Format: Failed to format '{value}' (type: {param_type}). Formatter returned: {formatted_str}")
            return None
        return formatted_str

    def fetch_ohlcv(self) -> Optional[pd.DataFrame]:
        """Fetches OHLCV data, converts to DataFrame, and processes numeric columns."""
        if not self.exchange:
            logger.error("Exchange not initialized, cannot fetch OHLCV.")
            return None
        logger.debug(
            f"Fetching up to {self.config.ohlcv_limit} OHLCV candles for {self.config.symbol} ({self.config.interval})..."
        )
        try:
            ohlcv_data = fetch_with_retries(
                self.exchange.fetch_ohlcv, # type: ignore
                symbol=self.config.symbol, timeframe=self.config.interval, limit=self.config.ohlcv_limit,
                max_retries=self.config.max_fetch_retries, delay_seconds=self.config.retry_delay_seconds,
            )
            if not ohlcv_data:
                logger.error(f"fetch_ohlcv for {self.config.symbol} returned no data.")
                return None
            if len(ohlcv_data) < 20: # Warn if insufficient for some indicators
                 logger.warning(f"Fetched only {len(ohlcv_data)} candles. May be insufficient for some indicators.")

            df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].map(safe_decimal)
                if df[col].apply(lambda x: isinstance(x, Decimal) and x.is_nan()).any():
                     logger.warning(f"Column '{col}' in OHLCV data contains NaNs after Decimal conversion.")

            initial_len = len(df)
            df.dropna(subset=["open", "high", "low", "close"], inplace=True, how="any")
            if len(df) < initial_len:
                logger.warning(f"Dropped {initial_len - len(df)} OHLCV rows due to NaNs in O/H/L/C columns.")

            if df.empty:
                 logger.error("OHLCV DataFrame is empty after processing. Cannot proceed.")
                 return None

            logger.debug(f"Fetched and processed {len(df)} OHLCV candles. Last timestamp: {df.index[-1]}")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch/process OHLCV data for {self.config.symbol}: {e}", exc_info=True)
            return None

    def get_balance(self) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Fetches total equity and available balance for the settlement currency (V5 API)."""
        if not self.exchange or not self.market_info:
            logger.error("Exchange or market info not available, cannot fetch balance.")
            return None, None

        settle_currency = self.market_info.get("settle")
        if not settle_currency:
            logger.error("Settle currency not found in market info. Cannot fetch balance.")
            return None, None

        logger.debug(
            f"Fetching balance for {settle_currency} (Account: {V5_UNIFIED_ACCOUNT_TYPE}, Category: {self.config.bybit_v5_category})..."
        )
        try:
            params = {"accountType": V5_UNIFIED_ACCOUNT_TYPE, "coin": settle_currency}
            balance_data = fetch_with_retries(
                self.exchange.fetch_balance, params=params, # type: ignore
                max_retries=self.config.max_fetch_retries, delay_seconds=self.config.retry_delay_seconds,
            )

            total_equity = safe_decimal(balance_data.get("total", {}).get(settle_currency))
            available_balance = safe_decimal(balance_data.get("free", {}).get(settle_currency))

            # Fallback parsing from 'info' field for Bybit V5 structure if standard fields are NaN
            if (total_equity.is_nan() or available_balance.is_nan()) and "info" in balance_data:
                logger.debug("Parsing balance from 'info' field (V5 structure fallback)...")
                info_list = balance_data.get("info", {}).get("result", {}).get("list", [])
                if info_list and isinstance(info_list, list):
                    acc_info = next((item for item in info_list if item.get("accountType") == V5_UNIFIED_ACCOUNT_TYPE), None)
                    if acc_info:
                        if total_equity.is_nan(): total_equity = safe_decimal(acc_info.get("totalEquity"))
                        if available_balance.is_nan(): available_balance = safe_decimal(acc_info.get("totalAvailableBalance"))
                        # Deeper fallback for available_balance if still NaN
                        if available_balance.is_nan() and "coin" in acc_info:
                             coin_details_list = acc_info.get("coin", [])
                             if coin_details_list and isinstance(coin_details_list, list):
                                 settle_coin_info = next((c for c in coin_details_list if c.get("coin") == settle_currency), None)
                                 if settle_coin_info:
                                      available_balance = safe_decimal(settle_coin_info.get("availableToWithdraw"))
                                      if total_equity.is_nan(): total_equity = safe_decimal(settle_coin_info.get("equity"))


            if total_equity.is_nan():
                logger.error(f"Could not extract total equity for {settle_currency}. Raw: {str(balance_data)[:500]}")
                return None, available_balance if not available_balance.is_nan() else Decimal("0")
            if available_balance.is_nan():
                logger.warning(f"Could not extract available balance for {settle_currency}. Defaulting to 0.")
                available_balance = Decimal("0")

            logger.debug(f"Balance ({settle_currency}): Equity={total_equity.normalize()}, Available={available_balance.normalize()}")
            return total_equity, available_balance
        except Exception as e:
            logger.error(f"Failed to fetch/parse balance: {e}", exc_info=True)
            return None, None

    def get_current_position(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Fetches current position details for the symbol using V5 API.
           Returns {'long': {details}, 'short': {details}} or None on error.
           Empty dicts if no position for a side.
        """
        if not self.exchange or not self.market_info:
            logger.error("Exchange or market info not available, cannot fetch position.")
            return None
        market_id = self.market_info.get("id")
        if not market_id:
            logger.error("Market ID not found in market info. Cannot fetch position.")
            return None

        logger.debug(
            f"Fetching position for {self.config.symbol} (ID: {market_id}, Cat: {self.config.bybit_v5_category}, Idx: {self.config.position_idx})..."
        )
        positions_summary: Dict[str, Dict[str, Any]] = {"long": {}, "short": {}}
        try:
            params = {"category": self.config.bybit_v5_category, "symbol": market_id}
            # Note: CCXT fetch_positions for Bybit V5 might require 'symbol' in params for filtering,
            # or it might return all positions for the category if symbols=[self.config.symbol] is not effective.
            # The current implementation relies on symbols=[self.config.symbol] and iterates the result.
            fetched_positions_list = fetch_with_retries(
                self.exchange.fetch_positions, symbols=[self.config.symbol], params=params, # type: ignore
                max_retries=self.config.max_fetch_retries, delay_seconds=self.config.retry_delay_seconds,
            )

            if not fetched_positions_list:
                logger.debug("No position data returned from fetch_positions. Assuming flat.")
                return positions_summary

            target_pos_info = None # This will hold the raw 'info' dict from CCXT
            for pos_data_ccxt in fetched_positions_list:
                raw_info = pos_data_ccxt.get("info", {})
                pos_symbol_api = raw_info.get("symbol")
                try: # positionIdx from API can be string "0", "1", "2"
                    pos_idx_api = int(raw_info.get("positionIdx", -1))
                except ValueError:
                    logger.warning(f"Could not parse positionIdx '{raw_info.get('positionIdx')}' from API. Skipping.")
                    continue

                if pos_symbol_api == market_id and pos_idx_api == self.config.position_idx:
                    target_pos_info = raw_info
                    logger.debug(f"Found matching position info: Symbol={pos_symbol_api}, Idx={pos_idx_api}")
                    break

            if not target_pos_info:
                logger.debug(f"No position found for {market_id} with Idx {self.config.position_idx}. Assuming flat.")
                return positions_summary

            qty = safe_decimal(target_pos_info.get("size", "0"))
            if qty.is_nan() or qty.copy_abs() < POSITION_QTY_EPSILON:
                logger.debug(f"Position size {qty.normalize()} negligible or zero. Considered flat.")
                return positions_summary

            api_side_str = target_pos_info.get("side", "None").lower() # 'Buy', 'Sell', or 'None' from API
            position_side_key: Optional[str] = None
            if self.config.position_idx == 0: # One-Way mode
                if api_side_str == "buy": position_side_key = "long"
                elif api_side_str == "sell": position_side_key = "short"
            elif self.config.position_idx == 1: position_side_key = "long"  # Hedge mode Buy side
            elif self.config.position_idx == 2: position_side_key = "short" # Hedge mode Sell side

            if position_side_key:
                entry_price = safe_decimal(target_pos_info.get("avgPrice", "0"))
                positions_summary[position_side_key] = {
                    "qty": qty.copy_abs(), # Always positive for summary
                    "entry_price": entry_price if not entry_price.is_nan() and entry_price > 0 else Decimal("NaN"),
                    "liq_price": safe_decimal(target_pos_info.get("liqPrice", "0"), default=Decimal("NaN")),
                    "unrealized_pnl": safe_decimal(target_pos_info.get("unrealisedPnl", "0")),
                    "api_side": api_side_str, # Original 'side' from API
                    "stop_loss_price": safe_decimal(target_pos_info.get("stopLoss", "0"), default=None),
                    "take_profit_price": safe_decimal(target_pos_info.get("takeProfit", "0"), default=None),
                    "is_tsl_active": safe_decimal(target_pos_info.get("trailingStop", "0")) > 0,
                    "tsl_trigger_price": safe_decimal(target_pos_info.get("trailingStop", "0"), default=None), # This is TSL distance for Bybit, not trigger price
                    # Note: Bybit's 'trailingStop' field in position data is the trail distance if TSL is active,
                    # 'tpslMode' and 'activePrice' for TSL activation are part of setTradingStop, not directly in position list.
                    # The 'is_tsl_active' here is a simplification. A more robust check might involve querying active orders.
                    # However, if 'trailingStop' (distance) is > 0, it implies TSL mechanisms are engaged.
                    "info": target_pos_info,
                }
                # Adjust 'tsl_trigger_price' based on 'activePrice' if available and TSL is active
                if positions_summary[position_side_key]["is_tsl_active"]:
                    active_price_tsl = safe_decimal(target_pos_info.get("activePrice", "0"), default=None)
                    if active_price_tsl and active_price_tsl > 0 :
                         positions_summary[position_side_key]["tsl_trigger_price"] = active_price_tsl
                    # else: 'trailingStop' field is the distance, not trigger for V5. 'activePrice' is the trigger.

                entry_str = positions_summary[position_side_key]["entry_price"].normalize() if not positions_summary[position_side_key]["entry_price"].is_nan() else "N/A"
                logger.debug(f"Identified {position_side_key.upper()} position: Qty={qty.copy_abs().normalize()}, Entry={entry_str}")
            else:
                 logger.warning(f"Position found (Qty: {qty.normalize()}) but couldn't map to long/short (API Side: '{api_side_str}', Idx: {self.config.position_idx}). Treating as flat.")
            return positions_summary
        except Exception as e:
            logger.error(f"Failed to fetch/parse positions for {self.config.symbol}: {e}", exc_info=True)
            return None

# --- Indicator Calculator Class ---
class IndicatorCalculator:
    """Calculates technical indicators (EMAs, Stochastic, ATR, ADX)."""
    def __init__(self, config: TradingConfig):
        self.config = config

    def calculate_indicators(self, df: pd.DataFrame) -> Optional[Dict[str, Union[Decimal, bool, int]]]:
        """Calculates indicators from OHLCV DataFrame. Returns dict or None on failure."""
        logger.info(f"{Fore.CYAN}# Calculating indicators (EMA, Stoch, ATR, ADX)...{Style.RESET_ALL}")
        if df is None or df.empty:
            logger.error(f"{Fore.RED}No DataFrame for indicator calculation.{Style.RESET_ALL}")
            return None

        required_cols = ["open", "high", "low", "close"]
        if not all(c in df.columns for c in required_cols):
            logger.error(f"{Fore.RED}DataFrame missing required columns: {set(required_cols) - set(df.columns)}{Style.RESET_ALL}")
            return None

        try:
            df_calc = df[required_cols].copy()
            # Convert Decimal columns to float for TA library compatibility / performance
            def safe_to_float(x: Any) -> float:
                if isinstance(x, (float, int)): return float(x)
                if isinstance(x, Decimal): return float('nan') if x.is_nan() else float(x)
                if isinstance(x, str): # Attempt conversion for string representations
                    try: return float(x.strip())
                    except ValueError:
                        if x.strip().lower() in ["nan", "none", "null", ""]: return float('nan')
                        logger.debug(f"Could not convert string '{x}' to float for TA.")
                        return float('nan')
                if x is None: return float('nan')
                logger.warning(f"Unexpected type {type(x)} ('{x}') for TA, using NaN.")
                return float('nan')

            for col in required_cols:
                df_calc[col] = df_calc[col].map(safe_to_float).astype(float)

            df_calc.dropna(subset=required_cols, inplace=True, how='any') # Drop rows with NaN in essential OHL C
            if df_calc.empty:
                logger.error(f"{Fore.RED}DataFrame empty after NaN drop for TA.{Style.RESET_ALL}")
                return None

            # Ensure sufficient data for lookback periods
            max_lookback = max(
                self.config.slow_ema_period, self.config.trend_ema_period,
                self.config.stoch_period + self.config.stoch_smooth_k + self.config.stoch_smooth_d, # Approx for Stoch
                self.config.atr_period, self.config.adx_period * 2 # ADX needs more data due to smoothing
            )
            min_rows_needed = max_lookback + 20 # Add buffer
            if len(df_calc) < min_rows_needed:
                logger.error(f"{Fore.RED}Insufficient data ({len(df_calc)} rows) for indicators (needs ~{min_rows_needed}).{Style.RESET_ALL}")
                return None

            # Calculate EMAs
            close_s = df_calc["close"]
            fast_ema_s = close_s.ewm(span=self.config.fast_ema_period, adjust=False).mean()
            slow_ema_s = close_s.ewm(span=self.config.slow_ema_period, adjust=False).mean()
            trend_ema_s = close_s.ewm(span=self.config.trend_ema_period, adjust=False).mean()

            # Calculate Stochastic
            high_s, low_s = df_calc["high"], df_calc["low"]
            low_min_stoch = low_s.rolling(window=self.config.stoch_period).min()
            high_max_stoch = high_s.rolling(window=self.config.stoch_period).max()
            stoch_range = high_max_stoch - low_min_stoch
            # Avoid division by zero for %K; default to 50 if range is zero
            stoch_k_raw_vals = np.where(stoch_range > 1e-12, 100 * (close_s - low_min_stoch) / stoch_range, 50.0)
            stoch_k_s = pd.Series(stoch_k_raw_vals, index=df_calc.index).fillna(50).rolling(window=self.config.stoch_smooth_k).mean().fillna(50)
            stoch_d_s = stoch_k_s.rolling(window=self.config.stoch_smooth_d).mean().fillna(50)

            # Calculate ATR
            prev_close_s = close_s.shift(1)
            tr1 = high_s - low_s
            tr2 = (high_s - prev_close_s).abs()
            tr3 = (low_s - prev_close_s).abs()
            true_range_s = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(0)
            atr_s = true_range_s.ewm(alpha=1/self.config.atr_period, adjust=False).mean() # Wilder's ATR

            # Calculate ADX, +DI, -DI
            adx_s, pdi_s, mdi_s = self._calculate_adx(high_s, low_s, close_s, atr_s, self.config.adx_period)

            # Helper to get latest valid Decimal value from a Series
            def get_latest_decimal(series: pd.Series, name: str) -> Decimal:
                valid_series = series.dropna()
                if valid_series.empty: return Decimal("NaN")
                try: return Decimal(str(valid_series.iloc[-1]))
                except (InvalidOperation, TypeError, ValueError):
                    logger.error(f"Failed to convert latest {name} value '{valid_series.iloc[-1]}' to Decimal.")
                    return Decimal("NaN")

            indicators_out: Dict[str, Union[Decimal, bool, int]] = {
                "fast_ema": get_latest_decimal(fast_ema_s, "fast_ema"),
                "slow_ema": get_latest_decimal(slow_ema_s, "slow_ema"),
                "trend_ema": get_latest_decimal(trend_ema_s, "trend_ema"),
                "stoch_k": get_latest_decimal(stoch_k_s, "stoch_k"),
                "stoch_d": get_latest_decimal(stoch_d_s, "stoch_d"),
                "atr": get_latest_decimal(atr_s, "atr"),
                "atr_period": self.config.atr_period,
                "adx": get_latest_decimal(adx_s, "adx"),
                "pdi": get_latest_decimal(pdi_s, "pdi"),
                "mdi": get_latest_decimal(mdi_s, "mdi"),
            }
            stoch_k_valid = stoch_k_s.dropna()
            indicators_out["stoch_k_prev"] = get_latest_decimal(stoch_k_valid.shift(1), "stoch_k_prev") if len(stoch_k_valid) >=1 else Decimal("NaN")


            # Stochastic K/D cross detection (using current K/D and previous K/D)
            k_now, d_now = indicators_out["stoch_k"], indicators_out["stoch_d"]
            stoch_d_valid = stoch_d_s.dropna()
            k_prev, d_prev = indicators_out["stoch_k_prev"], get_latest_decimal(stoch_d_valid.shift(1),"stoch_d_prev") if len(stoch_d_valid) >=1 else Decimal("NaN")

            indicators_out["stoch_kd_bullish"] = False
            indicators_out["stoch_kd_bearish"] = False
            if not any(v.is_nan() for v in [k_now, d_now, k_prev, d_prev]):
                if (k_prev <= d_prev) and (k_now > d_now): indicators_out["stoch_kd_bullish"] = True
                if (k_prev >= d_prev) and (k_now < d_now): indicators_out["stoch_kd_bearish"] = True

            # Check for critical NaN indicators
            critical_keys = ["fast_ema", "slow_ema", "trend_ema", "atr", "stoch_k", "stoch_d", "stoch_k_prev", "adx", "pdi", "mdi"]
            nan_indicators = [k for k in critical_keys if indicators_out.get(k, Decimal("NaN")).is_nan()]
            if nan_indicators:
                logger.error(f"{Fore.RED}Critical indicators are NaN: {', '.join(nan_indicators)}.{Style.RESET_ALL}")
                if indicators_out.get("atr", Decimal("NaN")).is_nan():
                     logger.error(f"{Fore.RED}ATR is NaN. Risk calculations will fail. Aborting indicators.{Style.RESET_ALL}")
                     return None # ATR is essential

            logger.info(f"{Style.BRIGHT}{Fore.GREEN}Indicators calculated successfully.{Style.RESET_ALL}")
            return indicators_out

        except Exception as e:
            logger.error(f"{Fore.RED}Error calculating indicators: {e}{Style.RESET_ALL}", exc_info=True)
            return None

    def _calculate_adx(
        self, high_s: pd.Series, low_s: pd.Series, close_s: pd.Series,
        atr_s: pd.Series, period: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Helper to calculate ADX, +DI, -DI using Wilder's smoothing (EMA)."""
        if period <= 0: raise ValueError("ADX period must be positive.")
        if atr_s.empty or atr_s.isnull().all():
             logger.error("ATR series empty/all NaN; cannot calculate ADX components.")
             nan_s = pd.Series(np.nan, index=high_s.index)
             return nan_s, nan_s, nan_s

        move_up, move_down = high_s.diff(), -low_s.diff()
        plus_dm_vals = np.where((move_up > move_down) & (move_up > 0), move_up, 0.0)
        minus_dm_vals = np.where((move_down > move_up) & (move_down > 0), move_down, 0.0)
        plus_dm_s, minus_dm_s = pd.Series(plus_dm_vals, index=high_s.index), pd.Series(minus_dm_vals, index=high_s.index)

        alpha = 1.0 / period # For Wilder's EMA
        smooth_plus_dm = plus_dm_s.ewm(alpha=alpha, adjust=False).mean().fillna(0)
        smooth_minus_dm = minus_dm_s.ewm(alpha=alpha, adjust=False).mean().fillna(0)

        # Avoid division by zero if ATR is zero or NaN
        pdi_vals = np.where((atr_s > 1e-12) & (~atr_s.isnull()), 100 * smooth_plus_dm / atr_s, 0.0)
        mdi_vals = np.where((atr_s > 1e-12) & (~atr_s.isnull()), 100 * smooth_minus_dm / atr_s, 0.0)
        pdi_s_out, mdi_s_out = pd.Series(pdi_vals, index=high_s.index).fillna(0), pd.Series(mdi_vals, index=high_s.index).fillna(0)

        di_sum = pdi_s_out + mdi_s_out
        dx_vals = np.where(di_sum > 1e-12, 100 * (pdi_s_out - mdi_s_out).abs() / di_sum, 0.0)
        dx_s = pd.Series(dx_vals, index=high_s.index).fillna(0)
        adx_s_out = dx_s.ewm(alpha=alpha, adjust=False).mean().fillna(0)
        return adx_s_out, pdi_s_out, mdi_s_out

# --- Signal Generator Class ---
class SignalGenerator:
    """Generates trading entry and exit signals based on indicator conditions."""
    def __init__(self, config: TradingConfig):
        self.config = config

    def generate_signals(
        self, df_last_candles: pd.DataFrame, indicators: Dict[str, Union[Decimal, bool, int]]
    ) -> Dict[str, Union[bool, str]]:
        """Generates 'long'/'short' entry signals and provides a detailed reason string."""
        result: Dict[str, Union[bool, str]] = {"long": False, "short": False, "reason": "Initializing signal check"}
        if not indicators:
            result["reason"] = "No Signal: Indicators data missing."
            logger.debug(result["reason"]); return result
        if df_last_candles is None or len(df_last_candles) < 2: # Need prev candle for ATR move filter
            result["reason"] = f"No Signal: Insufficient candle data (needs >=2, got {len(df_last_candles) if df_last_candles is not None else 0})."
            logger.debug(result["reason"]); return result

        try:
            current_price = safe_decimal(df_last_candles.iloc[-1]["close"])
            prev_close = safe_decimal(df_last_candles.iloc[-2]["close"])
            if current_price.is_nan() or current_price <= 0:
                result["reason"] = f"No Signal: Invalid current price ({current_price.normalize() if not current_price.is_nan() else 'NaN'})."
                logger.warning(result["reason"]); return result

            # Extract and validate required Decimal indicators
            req_dec_keys = ["stoch_k", "fast_ema", "slow_ema", "trend_ema", "atr", "adx", "pdi", "mdi"]
            ind: Dict[str, Decimal] = {}
            nan_keys = [k for k in req_dec_keys if not isinstance(indicators.get(k), Decimal) or indicators[k].is_nan()] # type: ignore
            if nan_keys:
                result["reason"] = f"No Signal: Required indicator(s) NaN/Missing: {', '.join(nan_keys)}."
                logger.warning(result["reason"]); return result
            for k in req_dec_keys: ind[k] = indicators[k] # type: ignore

            stoch_kd_bull_cross = indicators.get("stoch_kd_bullish", False)
            stoch_kd_bear_cross = indicators.get("stoch_kd_bearish", False)

            # EMA Cross
            ema_bullish_cross = ind["fast_ema"] > ind["slow_ema"]
            ema_bearish_cross = ind["fast_ema"] < ind["slow_ema"]
            ema_state = "Bullish" if ema_bullish_cross else "Bearish" if ema_bearish_cross else "Neutral"

            # Trend Filter
            trend_buffer = ind["trend_ema"].copy_abs() * (self.config.trend_filter_buffer_percent / 100)
            trend_allows_long = (current_price > (ind["trend_ema"] - trend_buffer)) if self.config.trade_only_with_trend else True
            trend_allows_short = (current_price < (ind["trend_ema"] + trend_buffer)) if self.config.trade_only_with_trend else True
            trend_suffix = f"(P:{current_price:.{DEFAULT_PRICE_DP}f} vs TrendEMA:{ind['trend_ema']:.{DEFAULT_PRICE_DP}f} ±{trend_buffer:.{DEFAULT_PRICE_DP}f})" if self.config.trade_only_with_trend else "(TrendFilter OFF)"

            # Stochastic Condition
            stoch_long_cond = (ind["stoch_k"] < self.config.stoch_oversold_threshold) or stoch_kd_bull_cross
            stoch_short_cond = (ind["stoch_k"] > self.config.stoch_overbought_threshold) or stoch_kd_bear_cross
            stoch_suffix = f"K:{ind['stoch_k']:.1f} (OS:{self.config.stoch_oversold_threshold.normalize()}/OB:{self.config.stoch_overbought_threshold.normalize()}) KD_Cross(B:{stoch_kd_bull_cross}/S:{stoch_kd_bear_cross})"

            # ATR Move Filter
            atr_move_ok = True; atr_suffix = "(ATR Filter OFF)"
            if self.config.atr_move_filter_multiplier > 0:
                if ind["atr"].is_nan() or ind["atr"] <= 0: atr_suffix = f"(ATR Filter Skipped: Invalid ATR {ind['atr'].normalize()})"; atr_move_ok = False
                elif prev_close.is_nan(): atr_suffix = "(ATR Filter Skipped: Prev close NaN)"; atr_move_ok = False
                else:
                    move_thresh = ind["atr"] * self.config.atr_move_filter_multiplier
                    price_move = (current_price - prev_close).copy_abs()
                    atr_move_ok = price_move > move_thresh
                    atr_suffix = f"(Move:{price_move:.{DEFAULT_PRICE_DP}f} {'OK' if atr_move_ok else 'LOW'} vs Thr:{move_thresh:.{DEFAULT_PRICE_DP}f})"

            # ADX Filter
            adx_trending = ind["adx"] > self.config.min_adx_level
            adx_long_favored = ind["pdi"] > ind["mdi"]
            adx_short_favored = ind["mdi"] > ind["pdi"]
            adx_allows_long = adx_trending and adx_long_favored
            adx_allows_short = adx_trending and adx_short_favored
            adx_suffix = f"(ADX:{ind['adx']:.1f} {'Trend' if adx_trending else 'Weak'} vs Min:{self.config.min_adx_level.normalize()} | Dir: {'+DI>-DI' if adx_long_favored else '-DI>+DI' if adx_short_favored else 'Neutral'})"

            # Combine signals
            base_long = ema_bullish_cross and stoch_long_cond
            base_short = ema_bearish_cross and stoch_short_cond
            final_long = base_long and trend_allows_long and atr_move_ok and adx_allows_long
            final_short = base_short and trend_allows_short and atr_move_ok and adx_allows_short

            if final_long:
                result["long"] = True
                result["reason"] = f"Long Signal: EMA {ema_state} & Stoch OK {stoch_suffix} & Trend OK {trend_suffix} & ATR OK {atr_suffix} & ADX OK {adx_suffix}"
            elif final_short:
                result["short"] = True
                result["reason"] = f"Short Signal: EMA {ema_state} & Stoch OK {stoch_suffix} & Trend OK {trend_suffix} & ATR OK {atr_suffix} & ADX OK {adx_suffix}"
            else: # No final signal, construct detailed "no signal" reason
                reason_parts = ["No Signal:"]
                if not base_long and not base_short: reason_parts.append(f"Base (EMA {ema_state} or Stoch {stoch_suffix}) not met.")
                elif base_long: # Potential long failed by filters
                    if not trend_allows_long: reason_parts.append(f"Long Blocked: Trend {trend_suffix}.")
                    elif not atr_move_ok: reason_parts.append(f"Long Blocked: ATR {atr_suffix}.")
                    elif not adx_allows_long: reason_parts.append(f"Long Blocked: ADX {adx_suffix}.")
                    else: reason_parts.append("Long filters met but logic error.") # Should not happen
                elif base_short: # Potential short failed by filters
                    if not trend_allows_short: reason_parts.append(f"Short Blocked: Trend {trend_suffix}.")
                    elif not atr_move_ok: reason_parts.append(f"Short Blocked: ATR {atr_suffix}.")
                    elif not adx_allows_short: reason_parts.append(f"Short Blocked: ADX {adx_suffix}.")
                    else: reason_parts.append("Short filters met but logic error.") # Should not happen
                else: reason_parts.append(f"Conditions unmet (EMA:{ema_state},Stoch:{stoch_suffix},Trend:{trend_suffix},ATR:{atr_suffix},ADX:{adx_suffix})")
                result["reason"] = " ".join(reason_parts)

            log_level = logging.INFO if result["long"] or result["short"] or "Blocked" in result["reason"] else logging.DEBUG
            logger.log(log_level, f"Signal Check: {result['reason']}")

        except Exception as e:
            logger.error(f"{Fore.RED}Error generating entry signals: {e}{Style.RESET_ALL}", exc_info=True)
            result.update({"reason": f"No Signal: Exception ({type(e).__name__})", "long": False, "short": False})
        return result

    def check_exit_signals(self, position_side: str, indicators: Dict[str, Union[Decimal, bool, int]]) -> Optional[str]:
        """Checks for signal-based exits (EMA cross, Stoch reversal). Returns exit reason or None."""
        if not indicators:
            logger.warning("Cannot check exit signals: indicators missing."); return None

        # Ensure all required Decimal indicators are valid
        dec_keys = ["fast_ema", "slow_ema", "stoch_k", "stoch_k_prev"]
        ind: Dict[str, Decimal] = {}
        for k in dec_keys:
            val = indicators.get(k)
            if not isinstance(val, Decimal) or val.is_nan():
                logger.warning(f"Cannot check exit signals: Indicator '{k}' missing/invalid (val: {val}).")
                return None
            ind[k] = val

        ema_bullish_cross = ind["fast_ema"] > ind["slow_ema"]
        ema_bearish_cross = ind["fast_ema"] < ind["slow_ema"]
        exit_reason: Optional[str] = None
        os_lvl, ob_lvl = self.config.stoch_oversold_threshold, self.config.stoch_overbought_threshold

        if position_side == "long":
            if ema_bearish_cross:
                exit_reason = f"Exit Signal (Long): EMA Bearish Cross (F {ind['fast_ema'].normalize()} < S {ind['slow_ema'].normalize()})"
            elif ind["stoch_k_prev"] >= ob_lvl and ind["stoch_k"] < ob_lvl:
                exit_reason = f"Exit Signal (Long): Stoch Reversal from OB (PrevK {ind['stoch_k_prev'].normalize():.1f} -> CurrK {ind['stoch_k'].normalize():.1f} < {ob_lvl.normalize()})"
            elif ind["stoch_k"] >= ob_lvl:
                logger.debug(f"Exit Check (Long): Stoch K ({ind['stoch_k'].normalize():.1f}) >= OB ({ob_lvl.normalize()}), awaiting bearish cross.")
        elif position_side == "short":
            if ema_bullish_cross:
                exit_reason = f"Exit Signal (Short): EMA Bullish Cross (F {ind['fast_ema'].normalize()} > S {ind['slow_ema'].normalize()})"
            elif ind["stoch_k_prev"] <= os_lvl and ind["stoch_k"] > os_lvl:
                exit_reason = f"Exit Signal (Short): Stoch Reversal from OS (PrevK {ind['stoch_k_prev'].normalize():.1f} -> CurrK {ind['stoch_k'].normalize():.1f} > {os_lvl.normalize()})"
            elif ind["stoch_k"] <= os_lvl:
                logger.debug(f"Exit Check (Short): Stoch K ({ind['stoch_k'].normalize():.1f}) <= OS ({os_lvl.normalize()}), awaiting bullish cross.")

        if exit_reason: logger.trade(f"{Fore.YELLOW}{exit_reason}{Style.RESET_ALL}")
        return exit_reason

# --- Order Manager Class ---
class OrderManager:
    """Handles order placement, position protection (SL/TP/TSL via V5 API), and closing."""
    def __init__(self, config: TradingConfig, exchange_manager: ExchangeManager):
        self.config = config
        self.exchange_manager = exchange_manager
        if not exchange_manager or not exchange_manager.exchange or not exchange_manager.market_info:
            raise ValueError("OrderManager requires ExchangeManager with initialized exchange and market_info.")
        self.exchange = exchange_manager.exchange
        self.market_info = exchange_manager.market_info
        # Tracks protection type locally: None, "ACTIVE_SLTP", "ACTIVE_TSL"
        self.protection_tracker: Dict[str, Optional[str]] = {"long": None, "short": None}

    def _calculate_trade_parameters(
        self, side: str, atr: Decimal, total_equity: Decimal, current_price: Decimal
    ) -> Optional[Dict[str, Optional[Decimal]]]:
        """Calculates SL/TP prices, order quantity, and TSL distance."""
        # Validate inputs
        if atr.is_nan() or atr <= 0: logger.error(f"Invalid ATR ({atr.normalize() if not atr.is_nan() else 'NaN'}) for param calc."); return None
        if total_equity.is_nan() or total_equity <= 0: logger.error(f"Invalid equity ({total_equity.normalize() if not total_equity.is_nan() else 'NaN'}) for param calc."); return None
        if current_price.is_nan() or current_price <= 0: logger.error(f"Invalid price ({current_price.normalize() if not current_price.is_nan() else 'NaN'}) for param calc."); return None
        if not all(k in self.market_info for k in ['tick_size', 'contract_size', 'min_order_size']):
             logger.error("Market info (tick_size/contract_size/min_order_size) incomplete for param calc."); return None
        if side not in ["buy", "sell"]: logger.error(f"Invalid side '{side}' for param calc."); return None

        try:
            risk_amount = total_equity * self.config.risk_percentage
            sl_atr_dist = atr * self.config.sl_atr_multiplier
            sl_price = current_price - sl_atr_dist if side == "buy" else current_price + sl_atr_dist
            if sl_price <= 0: logger.error(f"Calculated SL price ({sl_price:.{DEFAULT_PRICE_DP}f}) invalid (<=0)."); return None

            sl_dist_current = (current_price - sl_price).copy_abs()
            min_tick = self.market_info['tick_size']
            if sl_dist_current < min_tick: # Ensure SL distance is at least one tick
                logger.warning(f"SL distance ({sl_dist_current.normalize()}) < min tick ({min_tick.normalize()}). Adjusting.")
                sl_dist_current = min_tick
                sl_price = current_price - sl_dist_current if side == "buy" else current_price + sl_dist_current
                if sl_price <= 0: logger.error(f"Adjusted SL price ({sl_price:.{DEFAULT_PRICE_DP}f}) still invalid (<=0)."); return None
            if sl_dist_current <= 0: logger.error(f"Calculated SL distance ({sl_dist_current.normalize()}) invalid (<=0)."); return None

            # Quantity calculation
            qty_calc: Decimal
            contract_size = self.market_info['contract_size'] # Typically 1 for Bybit futures (1 unit of base or 1 USD for inverse)
            if self.config.market_type == "inverse": # Risk in base, price in quote/base, SL in quote
                # Qty_base = (Risk_base * Price_quote/base) / SL_dist_quote
                # Assumes contract_size = 1 (USD value of contract, but order amount is in base)
                qty_calc = (risk_amount * current_price) / (sl_dist_current * contract_size) # contract_size should be 1 here conceptually for qty in base
            else: # Linear (Risk in quote, SL in quote)
                # Qty_base = Risk_quote / SL_dist_quote
                # Assumes contract_size = 1 (unit of base per contract)
                qty_calc = risk_amount / (sl_dist_current * contract_size) # contract_size should be 1 here conceptually for qty in base

            qty_final_str = self.exchange_manager.format_amount(qty_calc, rounding_mode=ROUND_DOWN)
            qty_final = safe_decimal(qty_final_str)
            if qty_final.is_nan() or qty_final <= 0:
                 logger.error(f"Calculated qty ({qty_final_str}) invalid/zero. Original: {qty_calc.normalize()}"); return None
            min_order_sz = self.market_info.get('min_order_size', Decimal('NaN'))
            if not min_order_sz.is_nan() and qty_final < min_order_sz:
                logger.error(f"Calculated qty {qty_final.normalize()} < min market size {min_order_sz.normalize()}."); return None

            # TP Price
            tp_price: Optional[Decimal] = None
            if self.config.tp_atr_multiplier > 0:
                tp_atr_dist = atr * self.config.tp_atr_multiplier
                tp_price_calc = current_price + tp_atr_dist if side == "buy" else current_price - tp_atr_dist
                if tp_price_calc <= 0: logger.warning(f"Calculated TP price ({tp_price_calc:.{DEFAULT_PRICE_DP}f}) invalid (<=0). Disabling TP.")
                else: tp_price = tp_price_calc

            # TSL Distance
            tsl_dist_raw = current_price * (self.config.trailing_stop_percent / 100)
            tsl_dist = tsl_dist_raw if tsl_dist_raw >= min_tick else min_tick
            tsl_dist_final_str = self.exchange_manager.format_price(tsl_dist) # TSL distance formatted like price
            tsl_dist_final = safe_decimal(tsl_dist_final_str)
            if tsl_dist_final.is_nan() or tsl_dist_final <= 0:
                 logger.warning(f"Calculated invalid TSL distance ({tsl_dist_final_str}). TSL may fail. Original: {tsl_dist.normalize()}")
                 tsl_dist_final = Decimal('NaN') # Mark as invalid for use

            # Final formatting for SL/TP prices
            sl_price_final_str = self.exchange_manager.format_price(sl_price)
            sl_price_final = safe_decimal(sl_price_final_str)
            if sl_price_final.is_nan() or sl_price_final <= 0: logger.error(f"Formatted SL price ({sl_price_final_str}) invalid."); return None

            tp_price_final: Optional[Decimal] = None
            if tp_price is not None:
                 tp_price_final_str = self.exchange_manager.format_price(tp_price)
                 tp_price_final = safe_decimal(tp_price_final_str)
                 if tp_price_final.is_nan() or tp_price_final <= 0:
                      logger.warning(f"Formatted TP price ({tp_price_final_str}) invalid. Disabling TP.")
                      tp_price_final = None # Disable if invalid

            params_out = {"qty": qty_final, "sl_price": sl_price_final, "tp_price": tp_price_final, "tsl_distance": tsl_dist_final}
            settle_curr = self.market_info.get('settle', '')
            logger.info(
                f"Trade Params ({side.upper()}): Qty={params_out['qty'].normalize()}, "
                f"EntryPx (approx.)={current_price.normalize()}, SL={params_out['sl_price'].normalize()}, "
                f"TP={params_out['tp_price'].normalize() if params_out['tp_price'] else 'Disabled'}, "
                f"TSLDistance=~{params_out['tsl_distance'].normalize() if params_out['tsl_distance'] and not params_out['tsl_distance'].is_nan() else 'N/A'}, "
                f"RiskAmt={risk_amount.normalize():.{DEFAULT_PRICE_DP}f} {settle_curr}, ATR={atr.normalize():.{DEFAULT_PRICE_DP+1}f}"
            )
            return params_out
        except (InvalidOperation, DivisionByZero, TypeError, Exception) as e:
            logger.error(f"Error calculating trade parameters for {side.upper()}: {e}", exc_info=True)
            return None

    def _execute_market_order(self, side: str, qty_decimal: Decimal) -> Optional[Dict]:
        """Executes a market order with retries and confirmation logging."""
        qty_str = self.exchange_manager.format_amount(qty_decimal, rounding_mode=ROUND_DOWN)
        final_qty = safe_decimal(qty_str)
        if final_qty.is_nan() or final_qty <= 0:
            logger.error(f"Market order with zero/invalid formatted qty: '{qty_str}' (Original: {qty_decimal.normalize()}). Aborted.")
            return None

        base_asset = self.market_info.get('base', '')
        logger.trade(
            f"{Fore.CYAN}Attempting MARKET {side.upper()} order: {final_qty.normalize()} {base_asset} for {self.config.symbol}...{Style.RESET_ALL}"
        )
        try:
            params_v5 = {
                "category": self.config.bybit_v5_category,
                "positionIdx": self.config.position_idx,
                "timeInForce": "ImmediateOrCancel", # Or "GTC" if preferred for market
            }
            order = fetch_with_retries(
                self.exchange.create_market_order, # type: ignore
                symbol=self.config.symbol, side=side, amount=float(final_qty), params=params_v5,
                max_retries=self.config.max_fetch_retries, delay_seconds=self.config.retry_delay_seconds,
            )
            if order is None: logger.error(f"{Fore.RED}Market order submission failed (returned None).{Style.RESET_ALL}"); return None

            order_id, status = order.get("id", "[N/A]"), order.get("status", "[unknown]")
            filled_qty = safe_decimal(order.get("filled", "0"))
            avg_fill_px = safe_decimal(order.get("average", "0"))
            avg_px_log = avg_fill_px.normalize() if not avg_fill_px.is_nan() and avg_fill_px > 0 else "[N/A]"

            logger.trade(
                f"{Style.BRIGHT}{Fore.GREEN}Market order submitted: ID {order_id}, Side {side.upper()}, "
                f"Ordered {final_qty.normalize()}, Status: {status}, Filled: {filled_qty.normalize()}, AvgPx: {avg_px_log}{Style.RESET_ALL}"
            )
            termux_notify(f"{self.config.symbol} Order Submitted", f"Market {side.upper()} {final_qty.normalize()} ID:{order_id}, Status:{status}")

            if status in ["rejected", "canceled", "expired"]:
                 reason = order.get("info", {}).get("rejectReason", "No reason")
                 logger.error(f"{Fore.RED}Market order {order_id} {status}. Reason: '{reason}'. Info: {order.get('info')}{Style.RESET_ALL}")
                 return None
            # Short delay for order processing on exchange side
            logger.debug(f"Delaying {self.config.order_check_delay_seconds}s after market order {order_id}...")
            time.sleep(self.config.order_check_delay_seconds)
            return order
        except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e:
            logger.error(f"{Fore.RED}Order placement failed ({type(e).__name__}): {e}{Style.RESET_ALL}")
            termux_notify(f"{self.config.symbol} Order FAILED", f"Market {side.upper()} failed: {str(e)[:50]}")
            return None
        except Exception as e:
            logger.error(f"{Fore.RED}Unexpected error placing market order: {e}{Style.RESET_ALL}", exc_info=True)
            termux_notify(f"{self.config.symbol} Order ERROR", f"Market {side.upper()} unexpected error.")
            return None

    def _set_position_protection(
        self, position_side: str, sl_price: Optional[Decimal] = None, tp_price: Optional[Decimal] = None,
        is_tsl: bool = False, tsl_distance: Optional[Decimal] = None, tsl_activation_price: Optional[Decimal] = None,
    ) -> bool:
        """Sets SL, TP, or TSL for a position using Bybit V5 setTradingStop."""
        market_id = self.market_info.get("id")
        if not market_id: logger.error("Cannot set protection: Market ID missing."); return False

        tracker_key = position_side.lower()
        sl_str = self.exchange_manager._format_v5_param(sl_price, "price", allow_zero=True)
        tp_str = self.exchange_manager._format_v5_param(tp_price, "price", allow_zero=True)
        tsl_dist_str = self.exchange_manager._format_v5_param(tsl_distance, "distance", allow_zero=False)
        tsl_act_px_str = self.exchange_manager._format_v5_param(tsl_activation_price, "price", allow_zero=False)

        # Base params for privatePostPositionTradingStop
        params: Dict[str, Any] = {
            "category": self.config.bybit_v5_category, "symbol": market_id,
            "positionIdx": self.config.position_idx, "tpslMode": V5_TPSL_MODE_FULL,
            "slTriggerBy": self.config.sl_trigger_by, "tpTriggerBy": self.config.sl_trigger_by, # TP trigger usually same as SL
            "triggerBy": self.config.tsl_trigger_by, # For TSL, V5 uses 'triggerBy' for trail trigger type
            "stopLoss": "0", "takeProfit": "0", "trailingStop": "0", "activePrice": "0", # Defaults to clear
        }
        action_desc = ""; new_tracker_state: Optional[str] = None

        if is_tsl:
            if tsl_dist_str and tsl_act_px_str:
                params.update({"trailingStop": tsl_dist_str, "activePrice": tsl_act_px_str})
                action_desc = f"ACTIVATE/MODIFY TSL (Dist: {tsl_dist_str}, ActPx: {tsl_act_px_str})"
                new_tracker_state = "ACTIVE_TSL"
            else:
                logger.error(f"Cannot activate TSL for {position_side.upper()}: Invalid TSL distance ('{tsl_dist_str}') or activation price ('{tsl_act_px_str}').")
                return False
        elif sl_str or tp_str: # Setting SL/TP
            if sl_str: params["stopLoss"] = sl_str
            if tp_str: params["takeProfit"] = tp_str
            action_desc = f"SET SL={params['stopLoss']} TP={params['takeProfit']}"
            new_tracker_state = "ACTIVE_SLTP"
        else: # Clearing all stops
            action_desc = "CLEAR ALL SL/TP/TSL"
            new_tracker_state = None # No protection active

        logger.trade(f"{Fore.CYAN}Attempting to {action_desc} for {position_side.upper()} {self.config.symbol}...{Style.RESET_ALL}")
        # CCXT method for Bybit V5 POST /v5/position/trading-stop
        method_name = "privatePostPositionTradingStop"
        if not hasattr(self.exchange, method_name):
            logger.error(f"{Style.BRIGHT}{Fore.RED}Fatal Error: CCXT method '{method_name}' not found. Cannot manage protection.{Style.RESET_ALL}")
            return False
        method_to_call = getattr(self.exchange, method_name)
        logger.debug(f"Calling CCXT '{method_name}' with params: {params}")

        try:
            response = fetch_with_retries(
                method_to_call, params=params,
                max_retries=self.config.max_fetch_retries, delay_seconds=self.config.retry_delay_seconds,
            )
            if response and response.get("retCode") == V5_SUCCESS_RETCODE:
                logger.trade(f"{Style.BRIGHT}{Fore.GREEN}{action_desc} successful for {position_side.upper()}.{Style.RESET_ALL}")
                termux_notify(f"{self.config.symbol} Protection Set", f"{action_desc} for {position_side.upper()}")
                self.protection_tracker[tracker_key] = new_tracker_state
                return True
            ret_code = response.get("retCode", "N/A") if response else "NoResp"
            ret_msg = response.get("retMsg", "N/A") if response else "NoResp"
            logger.error(f"{Fore.RED}{action_desc} failed for {position_side.upper()}. API: Code={ret_code}, Msg='{ret_msg}'.{Style.RESET_ALL}")
            logger.debug(f"Full response from failed {method_name}: {response}")
            termux_notify(f"{self.config.symbol} Protection FAILED", f"{action_desc[:30]}... failed: {ret_msg[:50]}")
            return False
        except Exception as e:
            logger.error(f"{Fore.RED}Unexpected error during '{action_desc}' for {position_side.upper()}: {e}{Style.RESET_ALL}", exc_info=True)
            termux_notify(f"{self.config.symbol} Protection ERROR", f"{action_desc[:30]}... error.")
            return False

    def _verify_position_state(
            self, expected_side: Optional[str], expected_qty_min: Decimal = POSITION_QTY_EPSILON,
            max_attempts: int = 4, delay_seconds: float = 1.5, action_context: str = "Position Verification"
        ) -> Tuple[bool, Optional[Dict[str, Dict[str, Any]]]]:
        """Fetches current position state repeatedly to verify if it matches the expected state."""
        logger.debug(f"{action_context}: Verifying. Expect Side: '{expected_side}', MinQty: {expected_qty_min.normalize()}. Attempts: {max_attempts}.")
        last_pos_summary: Optional[Dict[str, Dict[str, Any]]] = None

        for attempt in range(max_attempts):
            logger.debug(f"{action_context}: Attempt {attempt + 1}/{max_attempts}...")
            current_summary = self.exchange_manager.get_current_position()
            last_pos_summary = current_summary # Keep track of the last fetched state

            if current_summary is None:
                logger.warning(f"{action_context}: Failed to fetch position state on attempt {attempt + 1}.")
                if attempt < max_attempts - 1: time.sleep(delay_seconds); continue
                logger.error(f"{Fore.RED}{action_context} FAILED: Could not fetch position after {max_attempts} attempts.{Style.RESET_ALL}")
                return False, last_pos_summary

            actual_side: Optional[str] = None; actual_qty = Decimal("0")
            long_pos, short_pos = current_summary.get("long", {}), current_summary.get("short", {})
            if long_pos and safe_decimal(long_pos.get("qty", "0")).copy_abs() >= POSITION_QTY_EPSILON:
                actual_side, actual_qty = "long", safe_decimal(long_pos.get("qty", "0"))
            elif short_pos and safe_decimal(short_pos.get("qty", "0")).copy_abs() >= POSITION_QTY_EPSILON:
                actual_side, actual_qty = "short", safe_decimal(short_pos.get("qty", "0"))

            is_flat = actual_side is None
            verified = False; log_suffix = ""

            if expected_side is None: # Expecting flat
                verified = is_flat
                log_suffix = f"Expected FLAT, Actual: {'FLAT' if is_flat else f'{actual_side.upper()} Qty={actual_qty.normalize()}'}"
            elif actual_side == expected_side: # Expected side matches
                qty_ok = actual_qty.copy_abs() >= expected_qty_min
                verified = qty_ok
                log_suffix = f"Expected {expected_side.upper()} (MinQty~{expected_qty_min.normalize()}), Actual: {actual_side.upper()} Qty={actual_qty.normalize()} ({'QTY OK' if qty_ok else 'QTY MISMATCH'})"
            else: # Side mismatch or unexpected state
                log_suffix = f"Expected {expected_side.upper() if expected_side else 'FLAT'}, Actual: {'FLAT' if is_flat else (actual_side.upper() + ' Qty=' + actual_qty.normalize()) if actual_side else 'UNKNOWN'} (SIDE MISMATCH)"

            logger.debug(f"{action_context} Check {attempt + 1}: {log_suffix}")
            if verified:
                logger.info(f"{Style.BRIGHT}{Fore.GREEN}{action_context} SUCCEEDED on attempt {attempt + 1}. State confirmed: {log_suffix}{Style.RESET_ALL}")
                return True, current_summary
            if attempt < max_attempts - 1: time.sleep(delay_seconds)
            else: logger.error(f"{Fore.RED}{action_context} FAILED after {max_attempts} attempts. Final state: {log_suffix}{Style.RESET_ALL}")
        return False, last_pos_summary # Should be caught by loop end

    def place_risked_market_order(
        self, side: str, atr: Decimal, total_equity: Decimal, current_price: Decimal
    ) -> bool:
        """Orchestrates a risked market order entry: calc params, order, verify, set stops."""
        if side not in ["buy", "sell"]: logger.error(f"Invalid side '{side}' for risked order."); return False
        if any(v is None or (isinstance(v, Decimal) and (v.is_nan() or v <= 0)) for v in [atr, total_equity, current_price]):
            logger.error("Entry Aborted: Invalid ATR, Equity, or Price."); return False

        logical_pos_side = "long" if side == "buy" else "short"
        logger.info(f"{Style.BRIGHT}{Fore.MAGENTA}--- Initiating Entry: {logical_pos_side.upper()} ---{Style.RESET_ALL}")

        params = self._calculate_trade_parameters(side, atr, total_equity, current_price)
        if not params or not params.get("qty") or params["qty"] <= 0 or not params.get("sl_price") or params["sl_price"] <=0: # type: ignore
            logger.error("Entry Aborted: Failed to calculate valid trade/SL parameters."); return False
        qty_order, sl_initial, tp_initial = params["qty"], params["sl_price"], params.get("tp_price") # type: ignore

        order_info = self._execute_market_order(side, qty_order) # type: ignore
        if not order_info:
            logger.error(f"Entry Aborted: Market order execution failed for {side.upper()} {qty_order.normalize()}.") # type: ignore
            self._handle_entry_failure(side, qty_order) # type: ignore
            return False
        order_id = order_info.get("id", "[N/A_ORDER_ID]")

        # Verify position opened correctly (allow some slippage in qty)
        min_qty_verify = qty_order * Decimal("0.90") # type: ignore
        verified_ok, final_pos_state = self._verify_position_state(
            expected_side=logical_pos_side, expected_qty_min=min_qty_verify,
            max_attempts=6, delay_seconds=max(self.config.order_check_delay_seconds, 1.0),
            action_context=f"Post-{logical_pos_side.upper()}-Entry Verification"
        )
        if not verified_ok:
            logger.error(f"{Fore.RED}Entry FAILED: Position verification failed after market order {order_id}. Manual check needed! Attempting cleanup...{Style.RESET_ALL}")
            self._handle_entry_failure(side, qty_order) # type: ignore
            return False

        active_pos = final_pos_state.get(logical_pos_side) if final_pos_state else {} # type: ignore
        if not active_pos:
            logger.error(f"{Fore.RED}Internal Error: Position {logical_pos_side} verified OK, but details missing. Aborting.{Style.RESET_ALL}")
            self._handle_entry_failure(side, qty_order) # type: ignore
            return False
        actual_qty, actual_entry_px = safe_decimal(active_pos.get("qty", "0")), safe_decimal(active_pos.get("entry_price", "NaN"))
        logger.info(f"{Style.BRIGHT}{Fore.GREEN}Position {logical_pos_side.upper()} confirmed: Qty={actual_qty.normalize()}, AvgEntryPx={actual_entry_px.normalize() if not actual_entry_px.is_nan() else '[N/A]'}{Style.RESET_ALL}")
        if actual_qty < qty_order * Decimal("0.99"): # type: ignore
             logger.warning(f"Filled qty {actual_qty.normalize()} notably less than ordered {qty_order.normalize()}.") # type: ignore

        stops_set = self._set_position_protection(logical_pos_side, sl_price=sl_initial, tp_price=tp_initial) # type: ignore
        if not stops_set:
            logger.error(f"{Fore.RED}Entry Alert: Failed to set initial SL/TP for {logical_pos_side.upper()}. Attempting emergency close!{Style.RESET_ALL}")
            self.close_position(logical_pos_side, actual_qty, reason="EmergencyClose:FailedInitialStopSet")
            return False

        if self.config.enable_journaling:
            if actual_entry_px.is_nan(): logger.warning("Logging trade entry to journal with N/A average entry price.")
            self.log_trade_entry_to_journal(side, actual_qty, actual_entry_px, order_id)

        logger.info(f"{Style.BRIGHT}{Fore.GREEN}--- Entry Sequence for {logical_pos_side.upper()} Completed Successfully ---{Style.RESET_ALL}")
        return True

    def manage_trailing_stop(
        self, position_side: str, entry_price: Decimal, current_market_price: Decimal, current_atr: Decimal
    ) -> None:
        """Checks TSL activation conditions and attempts to activate TSL."""
        tracker_key = position_side.lower()
        if self.protection_tracker.get(tracker_key) != "ACTIVE_SLTP":
            status = self.protection_tracker.get(tracker_key)
            log_msg = f"TSL already active/in transition (Tracker: {status})." if status == "ACTIVE_TSL" else f"No SL/TP tracked (Tracker: {status}). Cannot activate TSL."
            logger.debug(f"TSL Mgmt ({position_side.upper()}): {log_msg}"); return

        if any(v.is_nan() or v <= 0 for v in [current_atr, entry_price, current_market_price]):
            logger.debug(f"TSL Check ({position_side.upper()}): Invalid ATR/EntryPx/CurrentPx. Skipping."); return

        try:
            act_dist_pts = current_atr * self.config.tsl_activation_atr_multiplier
            tsl_act_target_px = entry_price + act_dist_pts if position_side == "long" else entry_price - act_dist_pts
            if tsl_act_target_px.is_nan() or tsl_act_target_px <= 0:
                logger.warning(f"Invalid TSL activation price ({tsl_act_target_px.normalize()}). Skipping TSL."); return

            tsl_trail_dist_pts = current_market_price * (self.config.trailing_stop_percent / 100)
            min_tick = self.market_info.get('tick_size', Decimal('1e-8'))
            tsl_trail_dist_pts = max(tsl_trail_dist_pts, min_tick) # Ensure at least min_tick
            if tsl_trail_dist_pts <= 0: logger.warning(f"Invalid TSL trail distance ({tsl_trail_dist_pts.normalize()}). Skipping TSL."); return

            activate_tsl = (position_side == "long" and current_market_price >= tsl_act_target_px) or \
                           (position_side == "short" and current_market_price <= tsl_act_target_px)

            if activate_tsl:
                logger.trade(f"{Fore.MAGENTA}TSL activation condition MET for {position_side.upper()}!{Style.RESET_ALL}")
                logger.trade(f"  Details: Entry={entry_price.normalize()}, CurrentPx={current_market_price.normalize()}, TSLTargetPx~={tsl_act_target_px:.{DEFAULT_PRICE_DP}f}")
                if self._set_position_protection(position_side, is_tsl=True, tsl_distance=tsl_trail_dist_pts, tsl_activation_price=tsl_act_target_px):
                    logger.trade(f"{Style.BRIGHT}{Fore.GREEN}TSL activated successfully for {position_side.upper()}.{Style.RESET_ALL}")
                else:
                    logger.error(f"{Fore.RED}Failed to activate TSL for {position_side.upper()} via API.{Style.RESET_ALL}")
            else:
                logger.debug(f"TSL Check ({position_side.upper()}): Activation NOT MET. (CurrentPx: {current_market_price.normalize()}, TargetActivationPx: ~{tsl_act_target_px:.{DEFAULT_PRICE_DP}f})")
        except Exception as e:
            logger.error(f"Error managing TSL for {position_side.upper()}: {e}", exc_info=True)

    def close_position(self, position_side: str, qty_to_close: Decimal, reason: str = "Strategy Exit Signal") -> bool:
        """Orchestrates position closure: clear stops, market close order, verify flat."""
        if position_side not in ["long", "short"]: logger.error(f"Invalid side '{position_side}' for close_position."); return False
        if qty_to_close.is_nan() or qty_to_close.copy_abs() < POSITION_QTY_EPSILON:
            logger.warning(f"Close requested for zero/negligible qty ({qty_to_close.normalize()}). Skipping close for {position_side.upper()}.")
            self.protection_tracker[position_side.lower()] = None; return True

        closing_order_side = "sell" if position_side == "long" else "buy"
        base_asset = self.market_info.get('base', '')
        logger.trade(
            f"{Fore.YELLOW}Attempting to CLOSE {position_side.upper()} position (Qty: {qty_to_close.normalize()} {base_asset}) "
            f"for {self.config.symbol} | Reason: {reason}...{Style.RESET_ALL}"
        )

        # Clear any existing SL/TP/TSL first
        if not self._set_position_protection(position_side, sl_price=None, tp_price=None, is_tsl=False):
            logger.warning(f"{Fore.YELLOW}Failed to confirm protection clear for {position_side.upper()}. Proceeding with close cautiously...{Style.RESET_ALL}")
        else: logger.info(f"Protection cleared (or was clear) for {position_side.upper()}.")

        close_order_info = self._execute_market_order(closing_order_side, qty_to_close)
        if not close_order_info:
            logger.error(f"{Fore.RED}Failed to submit closing market order for {position_side.upper()}. MANUAL INTERVENTION REQUIRED!{Style.RESET_ALL}")
            termux_notify(f"{self.config.symbol} CLOSE ORDER FAILED", f"Market {closing_order_side.upper()} order failed!")
            return False

        close_order_id = close_order_info.get("id", "[N/A_CLOSE_ID]")
        avg_close_px = safe_decimal(close_order_info.get("average"), default=Decimal("NaN"))
        logger.trade(f"{Fore.YELLOW}Closing market order ({close_order_id}) submitted. Reported AvgClosePx: {avg_close_px.normalize() if not avg_close_px.is_nan() else '[Pending/N/A]'}{Style.RESET_ALL}")
        termux_notify(f"{self.config.symbol} Position Closing", f"{position_side.upper()} close order {close_order_id} submitted.")

        # Verify position is flat
        verified_flat, _ = self._verify_position_state(
            expected_side=None, max_attempts=6, delay_seconds=max(self.config.order_check_delay_seconds + 0.5, 1.5),
            action_context=f"Post-{position_side.upper()}-Close Verification"
        )

        if self.config.enable_journaling:
            self.log_trade_exit_to_journal(position_side, qty_to_close, avg_close_px, close_order_id, reason)

        if not verified_flat:
            logger.error(f"{Fore.RED}Position {position_side.upper()} closure verification FAILED. Position may still be open. MANUAL INTERVENTION REQUIRED!{Style.RESET_ALL}")
            termux_notify(f"{self.config.symbol} CLOSE VERIFY FAILED", f"{position_side.upper()} position may still be open!")
            return False

        logger.trade(f"{Style.BRIGHT}{Fore.GREEN}Position {position_side.upper()} confirmed closed (flat).{Style.RESET_ALL}")
        self.protection_tracker[position_side.lower()] = None
        return True

    def _handle_entry_failure(self, failed_entry_order_side: str, attempted_qty: Decimal):
        """Handles cleanup after a failed entry sequence step, checking for lingering positions."""
        logical_pos_side = "long" if failed_entry_order_side == "buy" else "short"
        logger.warning(
            f"{Fore.YELLOW}Handling entry failure for {failed_entry_order_side.upper()} (intended qty: {attempted_qty.normalize()}). Checking for lingering position...{Style.RESET_ALL}"
        )
        time.sleep(max(self.config.order_check_delay_seconds, 1.0) + 1) # Wait for exchange state to settle

        _, current_pos_summary = self._verify_position_state(
            expected_side=None, max_attempts=2, delay_seconds=1.0, # Just fetch current state
            action_context=f"Entry-Failure-Cleanup-Check-{logical_pos_side.upper()}"
        )
        if current_pos_summary is None:
            logger.error(f"{Fore.RED}Could not fetch positions during entry failure handling for {logical_pos_side.upper()}. MANUAL CHECK URGENT!{Style.RESET_ALL}")
            termux_notify(f"{self.config.symbol} URGENT CHECK", "Failed to get position state during entry failure cleanup!"); return

        lingering_pos = current_pos_summary.get(logical_pos_side, {})
        lingering_qty = safe_decimal(lingering_pos.get("qty", "0"))

        if lingering_qty.copy_abs() >= POSITION_QTY_EPSILON:
            logger.error(f"{Fore.RED}Lingering {logical_pos_side.upper()} position (Qty: {lingering_qty.normalize()}) found after failed entry. Attempting emergency close...{Style.RESET_ALL}")
            termux_notify(f"{self.config.symbol} Emergency Close", f"Lingering {logical_pos_side.upper()} pos found.")
            if self.close_position(logical_pos_side, lingering_qty, reason="EmergencyClose:LingeringAfterEntryFail"):
                logger.info(f"Emergency close for lingering {logical_pos_side.upper()} position submitted/confirmed.")
            else:
                logger.critical(f"{Style.BRIGHT}{Fore.RED}EMERGENCY CLOSE FAILED for lingering {logical_pos_side.upper()}. MANUAL INTERVENTION URGENT!{Style.RESET_ALL}")
                termux_notify(f"{self.config.symbol} URGENT CHECK", f"Emergency close of lingering {logical_pos_side.upper()} FAILED!")
        else:
            logger.info(f"No significant lingering {logical_pos_side.upper()} position detected (Current qty: {lingering_qty.normalize()}).")
            self.protection_tracker[logical_pos_side] = None # Ensure tracker is clear

    def _write_journal_row(self, trade_data: Dict[str, Any]):
        """Helper to write a single row to the CSV trading journal."""
        if not self.config.enable_journaling: return
        journal_file = Path(self.config.journal_file_path)
        file_exists = journal_file.is_file() and journal_file.stat().st_size > 0
        try:
            journal_file.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            with journal_file.open("a", newline="", encoding="utf-8") as csvfile:
                fieldnames = ["TimestampUTC", "Symbol", "Action", "Side", "Quantity", "AvgPrice", "OrderID", "Reason", "Notes"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
                if not file_exists: writer.writeheader() # Write header if new file

                row = {f: trade_data.get(f) for f in fieldnames} # Prepare row
                for k, v in row.items(): # Format values for CSV
                    if isinstance(v, Decimal): row[k] = 'NaN' if v.is_nan() else f"{v.normalize()}"
                    elif v is None: row[k] = 'N/A'
                    else: row[k] = str(v)
                row['Notes'] = trade_data.get('Notes', '') # Ensure Notes field is present
                writer.writerow(row)
            logger.debug(f"Trade action '{trade_data.get('Action', 'Unknown')}' logged to journal: {journal_file}")
        except IOError as e:
            logger.error(f"I/O error writing to journal '{journal_file}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error writing to journal: {e}", exc_info=True)

    def log_trade_entry_to_journal(self, order_side: str, filled_qty: Decimal, avg_fill_price: Decimal, order_id: Optional[str]):
        """Logs trade entry details to the CSV journal."""
        entry_data = {
            "TimestampUTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "Symbol": self.config.symbol, "Action": "ENTRY",
            "Side": ("long" if order_side == "buy" else "short").upper(),
            "Quantity": filled_qty, "AvgPrice": avg_fill_price, "OrderID": order_id,
            "Reason": "Strategy Entry Signal",
        }
        self._write_journal_row(entry_data)

    def log_trade_exit_to_journal(self, position_side_closed: str, closed_qty: Decimal, avg_close_price: Decimal, order_id: Optional[str], exit_reason: str):
        """Logs trade exit details to the CSV journal."""
        exit_data = {
            "TimestampUTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "Symbol": self.config.symbol, "Action": "EXIT",
            "Side": position_side_closed.upper(),
            "Quantity": closed_qty, "AvgPrice": avg_close_price, "OrderID": order_id,
            "Reason": exit_reason,
        }
        self._write_journal_row(exit_data)

# --- Status Display Class ---
class StatusDisplay:
    """Handles displaying the bot's status and key information using Rich."""
    def __init__(self, config: TradingConfig):
        self.config = config
        self._price_dp = DEFAULT_PRICE_DP # Default, updated from market_info
        self._amount_dp = DEFAULT_AMOUNT_DP

    def _format_dec(self, val: Optional[Decimal], prec: Optional[int] = None, default_prec: int = 2,
                      commas: bool = False, neg_hl: bool = False, style: str = "white", style_ovr: Optional[str] = None) -> Text:
        """Formats Decimal for Rich Text display with styling."""
        if val is None or (isinstance(val, Decimal) and val.is_nan()): return Text("N/A", style="dim")
        dp = prec if prec is not None else default_prec
        try:
            fmt_val = val.quantize(Decimal("1e-" + str(dp)), rounding=ROUND_HALF_EVEN)
            fmt_str = f"{{:{',' if commas else ''}.{dp}f}}"
            disp_str = fmt_str.format(fmt_val)
            curr_style = style_ovr if style_ovr else style
            if neg_hl and not style_ovr:
                if fmt_val < 0: curr_style = "bright_red"
                elif fmt_val > 0: curr_style = "bright_green"
            return Text(disp_str, style=curr_style)
        except (ValueError, TypeError, InvalidOperation):
            logger.error(f"Error formatting decimal '{val}' for Rich display.")
            return Text("ERR", style="bold bright_red")

    def print_status_panel(
        self, cycle: int, timestamp: Optional[datetime], price: Optional[Decimal], indicators: Optional[Dict],
        positions: Optional[Dict], equity: Optional[Decimal], status_msg: Dict,
        protection_tracker: Dict, market_info: Optional[Dict]
    ):
        """Prints the main status panel using Rich Panel and Text."""
        self._price_dp = market_info["precision_dp"]["price"] if market_info and "precision_dp" in market_info else DEFAULT_PRICE_DP
        self._amount_dp = market_info["precision_dp"]["amount"] if market_info and "precision_dp" in market_info else DEFAULT_AMOUNT_DP

        ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S %Z") if timestamp else Text("Timestamp N/A", style="dim")
        title = f" Cycle {cycle} | {self.config.symbol} ({self.config.interval}) | {ts_str} "
        content = Text()

        price_fmt = self._format_dec(price, prec=self._price_dp, style_ovr="bright_white")
        settle_curr = self.config.symbol.split(":")[-1] if ":" in self.config.symbol else market_info.get("settle", "QUOTE") if market_info else "QUOTE"
        equity_fmt = self._format_dec(equity, prec=2, commas=True, style_ovr="bright_yellow")
        content.append("Price: ", style="bold bright_cyan"); content.append(price_fmt)
        content.append(" | ", style="dim")
        content.append("Equity: ", style="bold bright_yellow"); content.append(equity_fmt)
        content.append(f" {settle_curr}\n", style="bright_yellow"); content.append("---\n", style="dim")

        content.append("Indicators: ", style="bold bright_cyan")
        if indicators:
            parts = []
            def ind_val(k: str, p: int = 1, s: str = "white"): return self._format_dec(indicators.get(k), prec=p, default_style=s) # type: ignore

            parts.append(Text("EMA(F/S/T): ").append(ind_val('fast_ema', self._price_dp, "cyan")).append("/")
                         .append(ind_val('slow_ema', self._price_dp, "magenta")).append("/")
                         .append(ind_val('trend_ema', self._price_dp, "yellow")))
            stoch_txt = Text("Stoch(K/D/PrevK): ").append(ind_val('stoch_k', 1, "bright_blue")).append("/") \
                .append(ind_val('stoch_d', 1, "blue")).append("/").append(ind_val('stoch_k_prev', 1, "dim blue"))
            if indicators.get('stoch_kd_bullish'): stoch_txt.append(" [b green]▲ BullX[/]", style="green")
            elif indicators.get('stoch_kd_bearish'): stoch_txt.append(" [b red]▼ BearX[/]", style="red")
            parts.append(stoch_txt)
            parts.append(Text(f"ATR({indicators.get('atr_period', self.config.atr_period)}): ").append(ind_val('atr', self._price_dp + 1, "bright_magenta")))
            adx_val = indicators.get('adx') # type: ignore
            adx_style = "yellow" if isinstance(adx_val, Decimal) and not adx_val.is_nan() and adx_val > self.config.min_adx_level else "dim yellow"
            parts.append(Text(f"ADX({self.config.adx_period}): ").append(self._format_dec(adx_val, 1, default_style=adx_style)) # type: ignore
                         .append(" [+DI:", style="dim").append(ind_val('pdi', 1, "bright_green"))
                         .append(" -DI:", style="dim").append(ind_val('mdi', 1, "bright_red")).append("]", style="dim"))
            for i, p_txt in enumerate(parts): content.append(p_txt); content.append(" | " if i < len(parts) - 1 else "\n", style="dim")
        else: content.append(Text("Calculating or data unavailable...\n", style="dim"))
        content.append("---\n", style="dim")

        content.append("Position: ", style="bold bright_cyan")
        pos_disp = Text("FLAT", style="bold bright_green")
        active_side: Optional[str] = None; active_details: Optional[Dict] = None
        if positions:
            lp, sp = positions.get("long", {}), positions.get("short", {})
            if lp and safe_decimal(lp.get('qty', Decimal(0))).copy_abs() >= POSITION_QTY_EPSILON: active_details, active_side = lp, "long"
            elif sp and safe_decimal(sp.get('qty', Decimal(0))).copy_abs() >= POSITION_QTY_EPSILON: active_details, active_side = sp, "short"

        if active_details and active_side:
            style = "bold bright_green" if active_side == "long" else "bold bright_red"
            pos_disp = Text(f"{active_side.upper()}: ", style=style)
            pos_disp.append("Qty=", style=style).append(self._format_dec(active_details.get("qty"), self._amount_dp))
            pos_disp.append(" | EntryPx=", style="dim").append(self._format_dec(active_details.get("entry_price"), self._price_dp))
            pos_disp.append(" | PnL=", style="dim").append(self._format_dec(active_details.get("unrealized_pnl"), 4, neg_hl=True))
            pos_disp.append(" | Protection: ", style="dim")

            prot_status_txt = Text("None", style="dim"); prot_details_txt = Text("")
            tracker_stat = protection_tracker.get(active_side)
            sl_api, tp_api = active_details.get("stop_loss_price"), active_details.get("take_profit_price")
            tsl_api_active, tsl_api_trig = active_details.get("is_tsl_active", False), active_details.get("tsl_trigger_price")

            if tsl_api_active:
                 prot_status_txt = Text("TSL Active", style="bright_magenta")
                 tsl_trig_fmt = self._format_dec(tsl_api_trig, self._price_dp)
                 prot_details_txt = Text(" (TrigPx:", style="dim").append(tsl_trig_fmt).append(")", style="dim")
                 if tracker_stat != "ACTIVE_TSL": prot_status_txt.append(" [TrackerMismatch?]", style="bright_yellow")
            elif sl_api or tp_api:
                 prot_status_txt = Text("SL/TP Active", style="bright_yellow")
                 sl_fmt = self._format_dec(sl_api, self._price_dp) if sl_api else Text("N/A", style="dim")
                 tp_fmt = self._format_dec(tp_api, self._price_dp) if tp_api else Text("N/A", style="dim")
                 prot_details_txt = Text(" (S:", style="dim").append(sl_fmt).append(" T:", style="dim").append(tp_fmt).append(")", style="dim")
                 if tracker_stat != "ACTIVE_SLTP": prot_status_txt.append(" [TrackerMismatch?]", style="bright_yellow")
            elif tracker_stat: # Local tracker has state but API shows none
                 prot_status_txt = Text(f"Tracked:{tracker_stat}", style="yellow"); prot_details_txt = Text(" (Exchange:None?)", style="dim")
            pos_disp.append(prot_status_txt).append(prot_details_txt)
        content.append(pos_disp); content.append("\n---\n", style="dim")

        content.append("Signal/Status: ", style="bold bright_cyan")
        reason_str = status_msg.get("reason", "No signal/status info available")
        style = "dim" # Default
        if status_msg.get("long") or "Long Signal" in reason_str or "ENTERED_BUY" in reason_str: style = "bold bright_green"
        elif status_msg.get("short") or "Short Signal" in reason_str or "ENTERED_SELL" in reason_str: style = "bold bright_red"
        elif "Blocked" in reason_str or "FAIL:" in reason_str: style = "yellow"
        elif "CLOSED_" in reason_str or "HOLDING_" in reason_str: style = "bright_blue"
        elif "No Signal:" not in reason_str and "Initializing" not in reason_str: style = "white"
        content.append(Text("\n             ".join(textwrap.wrap(reason_str, width=100)), style=style)) # Indent wrapped

        console.print( Panel(content, title=f"[bold bright_magenta]{title}[/]", border_style="bright_blue", expand=False, padding=(1, 2)))

# --- Trading Bot Class ---
class TradingBot:
    """Main orchestrator class for the Pyrmethus trading bot."""
    def __init__(self):
        logger.info(f"{Style.BRIGHT}{Fore.MAGENTA}--- Initializing Pyrmethus v4.5.7 (Neon Nexus Edition) ---{Style.RESET_ALL}")
        self.config = TradingConfig()
        try:
            self.exchange_manager = ExchangeManager(self.config)
            self.indicator_calculator = IndicatorCalculator(self.config)
            self.signal_generator = SignalGenerator(self.config)
            self.order_manager = OrderManager(self.config, self.exchange_manager)
        except (ValueError, Exception) as e: # Catch init errors from components
            logger.critical(f"{Style.BRIGHT}{Fore.RED}TradingBot component initialization failed: {e}. Halting.{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)

        self.status_display = StatusDisplay(self.config)
        self.shutdown_requested = False
        self._setup_signal_handlers()
        logger.info(f"{Style.BRIGHT}{Fore.GREEN}Pyrmethus components initialized. Ready to trade.{Style.RESET_ALL}")

    def _setup_signal_handlers(self):
        """Sets up OS signal handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGINT, self._signal_handler_callback)
            signal.signal(signal.SIGTERM, self._signal_handler_callback)
            logger.debug("Signal handlers for SIGINT and SIGTERM set up.")
        except (ValueError, OSError, AttributeError) as e: # More specific exceptions
             logger.warning(f"{Fore.YELLOW}Could not set all OS signal handlers: {e}{Style.RESET_ALL}")

    def _signal_handler_callback(self, sig_num: int, frame: Optional[Any]):
        """Internal callback for OS signals to initiate shutdown."""
        if not self.shutdown_requested:
            sig_name = signal.Signals(sig_num).name if sig_num in signal.Signals else f"Signal {sig_num}"
            console.print(f"\n[bold yellow]Signal {sig_name} received. Initiating graceful shutdown...[/]")
            logger.warning(f"Signal {sig_name} received. Initiating graceful shutdown...")
            self.shutdown_requested = True
        else:
            logger.warning("Shutdown already in progress. Ignoring additional signal.")

    def _display_startup_info(self):
        """Displays key configuration parameters at startup using Rich Panel."""
        text_content = (
            f"Symbol: {self.config.symbol}\n"
            f"Interval: {self.config.interval}\n"
            f"Market Type: {self.config.market_type} (Category: {self.config.bybit_v5_category})\n"
            f"Position Index: {self.config.position_idx}\n"
            f"Risk Per Trade: {self.config.risk_percentage * 100:.3f}%\n"
            f"SL/TP Multipliers (ATR): SL={self.config.sl_atr_multiplier.normalize()}, TP={self.config.tp_atr_multiplier.normalize()}\n"
            f"TSL Activation (ATR Mult): {self.config.tsl_activation_atr_multiplier.normalize()}, TSL Percent: {self.config.trailing_stop_percent.normalize()}%\n"
            f"Trade Only With Trend: {self.config.trade_only_with_trend}\n"
            f"Journaling: {'Enabled' if self.config.enable_journaling else 'Disabled'} (File: '{self.config.journal_file_path}')\n"
            f"Log Level: {log_level_str}" # Use the validated or defaulted string
        )
        console.print(Panel(Text(text_content, style="bright_white"),
            title="[bold cyan]Pyrmethus Configuration Summary[/]", border_style="cyan", expand=False
        ))

    def run(self):
        """Starts the main trading loop."""
        self._display_startup_info()
        termux_notify("Pyrmethus Started", f"Trading {self.config.symbol} on {self.config.interval}.")
        cycle_count = 0

        while not self.shutdown_requested:
            cycle_count += 1
            cycle_start_time = time.monotonic()
            logger.debug(f"{Fore.BLUE}--- Trading Cycle {cycle_count} Start ---{Style.RESET_ALL}")

            try:
                self.trading_spell_cycle(cycle_count)
            except KeyboardInterrupt: # Explicitly handle KI here if not caught by signal handler first
                logger.warning("\nKeyboardInterrupt in main loop. Initiating shutdown.")
                self.shutdown_requested = True; break
            except ccxt.AuthenticationError as e:
                logger.critical(f"{Style.BRIGHT}{Fore.RED}CRITICAL AUTH ERROR (Cycle {cycle_count}): {e}. Halting.{Style.RESET_ALL}", exc_info=False)
                termux_notify("Pyrmethus CRITICAL ERROR", f"Auth failed: {str(e)[:100]}")
                self.shutdown_requested = True; break
            except SystemExit as e: # Catch sys.exit if raised internally
                 logger.warning(f"SystemExit (code {e.code}) in trading cycle. Terminating.")
                 self.shutdown_requested = True; break
            except Exception as e: # Catch-all for other cycle errors
                logger.error(f"{Style.BRIGHT}{Fore.RED}Unhandled exception in cycle {cycle_count}: {e}{Style.RESET_ALL}", exc_info=True)
                termux_notify("Pyrmethus Cycle Error", f"Exception in cycle {cycle_count}. Check logs.")
                # Extended sleep after unhandled error before retrying cycle
                time.sleep(self.config.loop_sleep_seconds * 2)
                continue # Continue to next cycle iteration

            cycle_duration = time.monotonic() - cycle_start_time
            sleep_duration = max(0, self.config.loop_sleep_seconds - cycle_duration)
            logger.debug(f"Cycle {cycle_count} completed in {cycle_duration:.2f}s. Sleeping for {sleep_duration:.2f}s.")

            if not self.shutdown_requested and sleep_duration > 0:
                # Interruptible sleep
                sleep_end = time.monotonic() + sleep_duration
                try:
                    while time.monotonic() < sleep_end and not self.shutdown_requested:
                        time.sleep(min(0.5, sleep_duration)) # Sleep in small chunks
                except KeyboardInterrupt: # Allow interruption of sleep
                    logger.warning("\nKeyboardInterrupt during sleep. Initiating shutdown.")
                    self.shutdown_requested = True

            if self.shutdown_requested:
                logger.info("Shutdown requested. Exiting main trading loop.")
                break
        self.graceful_shutdown()
        console.print(f"\n[bold bright_cyan]Pyrmethus ({self.config.symbol}) has concluded its session.[/]")
        sys.exit(0) # Ensure clean exit code 0 on normal shutdown

    def trading_spell_cycle(self, cycle_num: int) -> None:
        """Executes one complete cycle of the trading logic."""
        status_update = {"reason": "Cycle Processing..."} # For status panel

        # 1. Fetch Market Data (OHLCV)
        logger.debug("Fetching latest OHLCV data...")
        ohlcv_df = self.exchange_manager.fetch_ohlcv()
        if ohlcv_df is None or ohlcv_df.empty:
            status_update = {"reason": "FAIL:FETCH_OHLCV"}
            self.status_display.print_status_panel(cycle_num, None, None, None, None, None, status_update, self.order_manager.protection_tracker, self.exchange_manager.market_info)
            logger.error(f"{Fore.RED}Cycle {cycle_num} Aborted: Failed to fetch OHLCV.{Style.RESET_ALL}"); return

        try: # Extract latest price and timestamp
            latest_close = safe_decimal(ohlcv_df.iloc[-1]["close"])
            latest_ts = ohlcv_df.index[-1].to_pydatetime()
            if latest_close.is_nan() or latest_close <= 0: raise ValueError(f"Invalid latest close: {latest_close.normalize() if not latest_close.is_nan() else 'NaN'}")
            logger.debug(f"Latest Candle: Ts={latest_ts.strftime('%Y-%m-%d %H:%M:%S %Z')}, Px={latest_close.normalize()}")
        except (IndexError, KeyError, ValueError, TypeError) as e:
            status_update = {"reason": f"FAIL:PROCESS_CANDLE ({e})"}
            self.status_display.print_status_panel(cycle_num, None, None, None, None, None, status_update, self.order_manager.protection_tracker, self.exchange_manager.market_info)
            logger.error(f"{Fore.RED}Cycle {cycle_num} Aborted: Error processing candle data: {e}{Style.RESET_ALL}"); return

        # 2. Calculate Indicators
        indicators = self.indicator_calculator.calculate_indicators(ohlcv_df)
        if not indicators:
            status_update = {"reason": "FAIL:CALC_INDICATORS"}
            self.status_display.print_status_panel(cycle_num, latest_ts, latest_close, None, None, None, status_update, self.order_manager.protection_tracker, self.exchange_manager.market_info)
            logger.error(f"{Fore.RED}Cycle {cycle_num} Aborted: Failed to calculate indicators.{Style.RESET_ALL}"); return

        # 3. Fetch Account Balance and Current Position
        equity, _ = self.exchange_manager.get_balance()
        if equity is None or equity.is_nan() or equity <= 0:
            status_update = {"reason": "FAIL:FETCH_EQUITY_INVALID"}
            self.status_display.print_status_panel(cycle_num, latest_ts, latest_close, indicators, None, equity, status_update, self.order_manager.protection_tracker, self.exchange_manager.market_info)
            logger.error(f"{Fore.RED}Cycle {cycle_num} Aborted: Invalid/zero equity.{Style.RESET_ALL}"); return

        positions = self.exchange_manager.get_current_position()
        if positions is None:
            status_update = {"reason": "FAIL:FETCH_POSITION"}
            self.status_display.print_status_panel(cycle_num, latest_ts, latest_close, indicators, None, equity, status_update, self.order_manager.protection_tracker, self.exchange_manager.market_info)
            logger.error(f"{Fore.RED}Cycle {cycle_num} Aborted: Failed to fetch position state.{Style.RESET_ALL}"); return

        # Determine active position
        active_side: Optional[str] = None; active_details: Optional[Dict] = None
        lp, sp = positions.get("long", {}), positions.get("short", {})
        if lp and safe_decimal(lp.get("qty", "0")).copy_abs() >= POSITION_QTY_EPSILON: active_side, active_details = "long", lp
        elif sp and safe_decimal(sp.get("qty", "0")).copy_abs() >= POSITION_QTY_EPSILON: active_side, active_details = "short", sp

        # 4. If in Active Position: Manage TSL and Exits
        if active_side and active_details:
            pos_qty, entry_px = safe_decimal(active_details.get("qty")), safe_decimal(active_details.get("entry_price"))
            current_atr = indicators.get("atr") # type: ignore

            # 4a. Manage Trailing Stop (only if fixed SL/TP is active and data valid)
            if (self.order_manager.protection_tracker.get(active_side) == "ACTIVE_SLTP" and
                not any(v is None or (isinstance(v, Decimal) and (v.is_nan() or v <=0)) for v in [entry_px, latest_close, current_atr])):
                self.order_manager.manage_trailing_stop(active_side, entry_px, latest_close, current_atr) # type: ignore
                if self.order_manager.protection_tracker.get(active_side) == "ACTIVE_TSL": # TSL activated
                    logger.debug("Re-fetching position summary after TSL management.")
                    positions = self.exchange_manager.get_current_position() # Update summary
                    # Re-evaluate active_details from new summary
                    lp_new, sp_new = (positions.get("long", {}), positions.get("short", {})) if positions else ({}, {})
                    active_details = lp_new if active_side == "long" else sp_new

            # 4b. Check for Signal-Based Exits (if TSL not primary exit manager yet)
            if self.order_manager.protection_tracker.get(active_side) != "ACTIVE_TSL":
                exit_reason = self.signal_generator.check_exit_signals(active_side, indicators)
                if exit_reason:
                    logger.trade(f"Attempting to close {active_side.upper()} due to: {exit_reason}")
                    if pos_qty and not pos_qty.is_nan() and pos_qty > 0:
                        closed = self.order_manager.close_position(active_side, pos_qty, reason=exit_reason)
                        status_update = {"reason": f"CLOSED_{active_side.upper()}_SIGNAL" if closed else f"FAIL:CLOSE_SIGNAL_{active_side.upper()}"}
                        positions = self.exchange_manager.get_current_position() # Refresh after close attempt
                        self.status_display.print_status_panel(cycle_num, latest_ts, latest_close, indicators, positions, equity, status_update, self.order_manager.protection_tracker, self.exchange_manager.market_info)
                        return # Cycle action complete
                    else: logger.warning(f"Exit signal for {active_side.upper()} but invalid position qty ({pos_qty}).")

            # Re-check if position became flat (e.g., SL/TP hit on exchange)
            logger.debug(f"Re-fetching position state for {active_side.upper()} after TSL/exit checks.")
            positions = self.exchange_manager.get_current_position()
            lp_final, sp_final = (positions.get("long", {}), positions.get("short", {})) if positions else ({}, {})
            if not ((lp_final and safe_decimal(lp_final.get("qty", "0")).copy_abs() >= POSITION_QTY_EPSILON and active_side == "long") or \
                    (sp_final and safe_decimal(sp_final.get("qty", "0")).copy_abs() >= POSITION_QTY_EPSILON and active_side == "short")):
                active_side, active_details = None, None # Became flat
                logger.info("Position closed by exchange (SL/TP/TSL hit or manual) during cycle checks.")
                status_update = {"reason": "INFO:POS_CLOSED_BY_EXCHANGE_STOP"}

        # 5. If Flat: Check for New Entry Signals
        if not active_side: # If now flat (or was flat initially)
            logger.debug("Currently flat. Checking for new entry signals...")
            entry_signals = self.signal_generator.generate_signals(ohlcv_df, indicators)
            status_update = entry_signals # Use signal reason as status for display

            entry_order_side: Optional[str] = "buy" if entry_signals.get("long") else "sell" if entry_signals.get("short") else None
            if entry_order_side:
                current_atr = indicators.get("atr") # type: ignore
                if not any(v is None or (isinstance(v, Decimal) and (v.is_nan() or v <=0)) for v in [equity, current_atr, latest_close]):
                    entered = self.order_manager.place_risked_market_order(entry_order_side, current_atr, equity, latest_close) # type: ignore
                    logical_side = "long" if entry_order_side == "buy" else "short"
                    status_update = {"reason": f"ENTERED_{logical_side.upper()}" if entered else f"FAIL:ENTRY_{logical_side.upper()}"}
                    positions = self.exchange_manager.get_current_position() # Refresh after entry attempt
                    self.status_display.print_status_panel(cycle_num, latest_ts, latest_close, indicators, positions, equity, status_update, self.order_manager.protection_tracker, self.exchange_manager.market_info)
                    return # Cycle action complete
                else:
                    logger.warning(f"Cannot attempt {entry_order_side} entry: Missing critical data (Equity/ATR/Price).")
                    status_update = {"reason": f"FAIL:ENTRY_DATA_MISSING_{entry_order_side.upper()}"}
        else: # Still in position, no exit triggered this cycle
            status_update = {"reason": f"HOLDING_{active_side.upper()}"}

        # 6. Display Current Status (if no early return from an action)
        self.status_display.print_status_panel(
            cycle_num, latest_ts, latest_close, indicators, positions, equity, status_update,
            self.order_manager.protection_tracker, self.exchange_manager.market_info
        )

    def graceful_shutdown(self):
        """Performs cleanup actions before the bot exits."""
        logger.info(f"{Style.BRIGHT}{Fore.YELLOW}--- Pyrmethus Graceful Shutdown Initiated ---{Style.RESET_ALL}")
        # For V5 position-based stops, positions are typically left managed by the exchange.
        # If there were client-side managed limit orders, they would be cancelled here.
        logger.info("Graceful shutdown: No specific position/order cleanup actions implemented for V5 (relies on exchange-side stops).")
        termux_notify("Pyrmethus Shutdown", f"Bot for {self.config.symbol} is shutting down.")
        logger.info(f"{Style.BRIGHT}{Fore.YELLOW}--- Pyrmethus Shutdown Complete ---{Style.RESET_ALL}")

if __name__ == "__main__":
    try:
        bot = TradingBot()
        bot.run()
    except SystemExit as e:
        level = logging.INFO if e.code == 0 else logging.WARNING
        logger.log(level, f"Pyrmethus terminated with exit code: {e.code}")
        sys.exit(e.code) # Propagate exit code
    except Exception as e:
        logger.critical(f"{Style.BRIGHT}{Fore.RED}CRITICAL UNHANDLED EXCEPTION in main execution: {e}{Style.RESET_ALL}", exc_info=True)
        termux_notify("Pyrmethus CRASHED", "Critical unhandled exception. Check logs!")
        sys.exit(1)

