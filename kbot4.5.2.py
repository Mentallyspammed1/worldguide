#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=logging-fstring-interpolation, too-many-instance-attributes, too-many-arguments, too-many-locals, too-many-public-methods, invalid-name, unused-argument, too-many-lines
# fmt: off
#   ____        _       _   _                  _            _         _
#  |  _ \\ _   _| |_ ___| | | | __ ___   ____ _| |_ ___  ___| |_ _ __ | |__
#  | |_) | | | | __/ _ \\ | | |/ _` \\ \\ / / _` | __/ _ \\/ __| __| '_ \\| '_ \\
#  |  __/| |_| | ||  __/ | | | (_| |\\ V / (_| | ||  __/\\__ \\ |_| |_) | | | |
#  |_|    \\__, |\\__\\___|_|_|_|\\__,_| \\_/ \\__,_|\\__\\___||___/\\__| .__/|_| |_|
#         |___/                                                |_|
# Pyrmethus v2.4.1 - Enhanced Code Structure & Robustness (v2.4.1+ Enhanced)
# fmt: on
"""
Pyrmethus - Termux Trading Spell (v2.4.1 - Enhanced+)

Conjures market insights and executes trades on Bybit Futures using the
V5 Unified Account API via CCXT. Refactored into classes for better structure
and utilizing V5 position-based stop-loss/take-profit/trailing-stop features.

Enhancements in this version (beyond v2.4.1):
- Improved code structure, readability, and maintainability.
- Refactored configuration loading (`_get_env`) for clarity and reduced complexity.
- Enhanced type hinting and docstrings throughout the codebase.
- Improved error handling and logging detail, especially for API interactions.
- Addressed several pylint warnings through code improvements (e.g., reduced complexity).
- Consistent use of f-strings and modern Python practices (e.g., `pathlib`).
- Simplified complex conditional logic where possible (e.g., in signal generation).
- Ensured robust handling of Decimal types and API responses (incl. V5 structure).
- Added more checks for data validity (e.g., price > 0, ATR > 0) before calculations.
- Refined main trading loop state management with multiple position checks.
- Improved graceful shutdown sequence (order cancellation, position closure).
- Enhanced status display with more details and better formatting.
- Minor performance considerations (e.g., fewer redundant calculations).

Original v2.4.1 Enhancements:
- Fixed Termux notification command (`termux-toast`).
- Fixed Decimal conversion errors from API strings in balance/position fetching.
- Implemented robust `safe_decimal` conversion utility.
- Corrected V5 order cancellation logic in graceful shutdown.
- Ensured numeric parameters for V5 `setTradingStop` are passed as strings.
- Improved handling and display of potential NaN values.
- Added trade exit journaling.
- Minor logging and display refinements.
- Corrected Decimal('NaN') comparison error in position fetching.
- Replaced deprecated pandas `applymap` with `map`.
- Simplified previous indicator value fetching.
- Removed unused StatusDisplay method.

Features:
- Class-based architecture (Config, Exchange, Indicators, Signals, Orders, Display, Bot).
- Robust configuration loading and validation from .env file.
- Multi-condition signal generation (EMAs, Stochastic, ATR Move Filter, ADX).
- Position-based Stop Loss, Take Profit, and Trailing Stop Loss management using V5 API.
- Signal-based exit mechanism (EMA crossover, Stochastic reversal).
- Enhanced error handling with retries and specific exception management.
- High-precision calculations using the Decimal type.
- Trade journaling to CSV file.
- Termux notifications for key events.
- Graceful shutdown handling SIGINT/SIGTERM.
- Rich library integration for enhanced terminal output.
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
    colorama_init(autoreset=True)
    missing_pkg = e.name
    print(
        f"{Fore.RED}{Style.BRIGHT}Missing essential spell component: {Style.BRIGHT}{missing_pkg}{Style.NORMAL}"
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
            f"{Fore.YELLOW}Note: pandas and numpy often installed via pkg in Termux."
        )
    else:
        print(
            f"{Style.BRIGHT}pip install {' '.join(COMMON_PACKAGES)}{Style.RESET_ALL}"
        )
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

# Initialize Colorama & Rich Console
colorama_init(autoreset=True)
console = Console()

# Set Decimal precision context
getcontext().prec = DECIMAL_PRECISION

# --- Logging Setup ---
logger = logging.getLogger(__name__)
TRADE_LEVEL_NUM = logging.INFO + 5
if not hasattr(logging.Logger, "trade"):
    logging.addLevelName(TRADE_LEVEL_NUM, "TRADE")

    def trade_log(self, message, *args, **kws):
        if self.isEnabledFor(TRADE_LEVEL_NUM):
            # pylint: disable=protected-access
            self._log(TRADE_LEVEL_NUM, message, args, **kws)

    logging.Logger.trade = trade_log  # type: ignore[attr-defined]

log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)-8s] (%(filename)s:%(lineno)d) %(message)s"
)
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logger.setLevel(log_level)

# Ensure handler is added only once to prevent duplicate logs
if not any(
    isinstance(h, logging.StreamHandler)
    and getattr(h, "stream", None) == sys.stdout
    for h in logger.handlers
):
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
logger.propagate = False


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
                timeout=5,
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
            logger.warning("Termux notify failed: command timed out.")
        except Exception as e:
            logger.warning(f"Termux notify failed unexpectedly: {e}")
    # else: logger.debug("Not in Termux, skipping notification.") # Optional debug


def fetch_with_retries(
    fetch_function: Callable[..., Any],
    *args: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
    delay_seconds: int = DEFAULT_RETRY_DELAY,
    **kwargs: Any,
) -> Any:
    """Wraps a function call with retry logic for specific CCXT/network errors."""
    last_exception: Optional[Exception] = None
    func_name = getattr(fetch_function, "__name__", "Unnamed function")

    for attempt in range(max_retries + 1):
        try:
            return fetch_function(*args, **kwargs)
        except (
            ccxt.DDoSProtection,
            ccxt.RequestTimeout,
            ccxt.ExchangeNotAvailable,
            ccxt.NetworkError,
            requests.exceptions.RequestException, # Catch underlying requests errors
            ccxt.RateLimitExceeded, # Explicitly catch rate limit
        ) as e:
            last_exception = e
            retry_msg = f"{Fore.YELLOW}Retryable error ({type(e).__name__}) on attempt {attempt + 1}/{max_retries + 1} for {func_name}: {str(e)[:150]}. Retrying in {delay_seconds}s...{Style.RESET_ALL}"
            if attempt < max_retries:
                logger.warning(retry_msg)
                time.sleep(delay_seconds)
            else:
                logger.error(
                    f"{Fore.RED}Max retries ({max_retries + 1}) reached for {func_name}. Last error: {e}"
                )
                break # Stop retrying
        except ccxt.AuthenticationError as e:
            logger.critical(
                f"{Fore.RED+Style.BRIGHT}Authentication error: {e}. Check API keys. Halting.",
                exc_info=False, # Less verbose for auth failure
            )
            sys.exit(1) # Critical failure, exit immediately
        except ccxt.InsufficientFunds as e:
            logger.error(f"{Fore.RED}Insufficient funds: {e}")
            last_exception = e
            break # Don't retry InsufficientFunds
        except ccxt.InvalidOrder as e:
            logger.error(f"{Fore.RED}Invalid order parameters: {e}")
            last_exception = e
            break # Don't retry InvalidOrder
        except ccxt.OrderNotFound as e:
            logger.warning(f"{Fore.YELLOW}Order not found: {e}")
            last_exception = e
            break # Often not retryable in context
        except ccxt.PermissionDenied as e:
            logger.error(
                f"{Fore.RED}Permission denied: {e}. Check API permissions/IP whitelisting."
            )
            last_exception = e
            break # Not retryable
        except ccxt.ExchangeError as e: # Catch other specific exchange errors
            logger.error(
                f"{Fore.RED}Exchange error during {func_name}: {e}"
            )
            last_exception = e
            # Decide if specific ExchangeErrors are retryable - for now, retry generic ones
            if attempt < max_retries:
                logger.warning(f"Retrying exchange error in {delay_seconds}s...")
                time.sleep(delay_seconds)
            else:
                logger.error(
                    f"Max retries reached after exchange error for {func_name}."
                )
                break
        except Exception as e: # Catch unexpected errors during fetch
            logger.error(
                f"{Fore.RED}Unexpected error during {func_name}: {e}",
                exc_info=True, # Log traceback for unexpected errors
            )
            last_exception = e
            break # Don't retry unknown errors

    # If loop finished without success, raise the last captured exception
    if last_exception:
        raise last_exception
    else:
        # This case should ideally not be reached if the loop breaks correctly
        # or the first attempt succeeds. If it is reached, something is wrong.
        raise RuntimeError(
            f"Function {func_name} failed after unexpected issue without specific exception."
        )


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
        self.order_fill_timeout_seconds: int = self._get_env("ORDER_FILL_TIMEOUT_SECONDS", 20, Style.DIM, cast_type=int, min_val=5) # Timeout waiting for fill confirmation (not currently used directly, but good to have)
        self.max_fetch_retries: int = self._get_env("MAX_FETCH_RETRIES", DEFAULT_MAX_RETRIES, Style.DIM, cast_type=int, min_val=0, max_val=10) # Allow 0 retries
        self.retry_delay_seconds: int = self._get_env("RETRY_DELAY_SECONDS", DEFAULT_RETRY_DELAY, Style.DIM, cast_type=int, min_val=1)
        self.trade_only_with_trend: bool = self._get_env("TRADE_ONLY_WITH_TREND", True, Style.DIM, cast_type=bool)

        # Journaling
        self.journal_file_path: str = self._get_env("JOURNAL_FILE_PATH", DEFAULT_JOURNAL_FILE, Style.DIM)
        self.enable_journaling: bool = self._get_env("ENABLE_JOURNALING", True, Style.DIM, cast_type=bool)

        # Final Checks
        if not self.api_key or not self.api_secret:
            logger.critical(
                f"{Fore.RED+Style.BRIGHT}BYBIT_API_KEY or BYBIT_API_SECRET not found in environment. Halting."
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
                f"Could not determine V5 category: {e}. Halting.",
                exc_info=True,
            )
            sys.exit(1)

    def _validate_config(self):
        """Performs post-load validation of configuration parameters."""
        if self.fast_ema_period >= self.slow_ema_period:
            logger.critical(
                f"{Fore.RED+Style.BRIGHT}Validation failed: FAST_EMA ({self.fast_ema_period}) must be < SLOW_EMA ({self.slow_ema_period}). Halting."
            )
            sys.exit(1)
        if self.trend_ema_period <= self.slow_ema_period:
            logger.warning(
                f"{Fore.YELLOW}Config Warning: TREND_EMA ({self.trend_ema_period}) <= SLOW_EMA ({self.slow_ema_period}). Trend filter might lag short-term EMA signals."
            )
        if self.stoch_oversold_threshold >= self.stoch_overbought_threshold:
            logger.critical(
                f"{Fore.RED+Style.BRIGHT}Validation failed: STOCH_OVERSOLD ({self.stoch_oversold_threshold.normalize()}) must be < STOCH_OVERBOUGHT ({self.stoch_overbought_threshold.normalize()}). Halting."
            )
            sys.exit(1)
        if self.tsl_activation_atr_multiplier < self.sl_atr_multiplier:
            logger.warning(
                f"{Fore.YELLOW}Config Warning: TSL_ACT_MULT ({self.tsl_activation_atr_multiplier.normalize()}) < SL_MULT ({self.sl_atr_multiplier.normalize()}). TSL may activate before initial SL distance is reached."
            )
        # Check TP vs SL only if TP is enabled (multiplier > 0)
        if self.tp_atr_multiplier > Decimal("0") and self.tp_atr_multiplier <= self.sl_atr_multiplier:
            logger.warning(
                f"{Fore.YELLOW}Config Warning: TP_MULT ({self.tp_atr_multiplier.normalize()}) <= SL_MULT ({self.sl_atr_multiplier.normalize()}). This implies a poor Risk:Reward setup (< 1:1)."
            )

    def _cast_value(
        self, key: str, value_str: str, cast_type: Type, default: Any
    ) -> Any:
        """Helper to cast string value to target type, returning default on failure."""
        try:
            val_to_cast = value_str.strip()
            if cast_type == bool:
                # Robust boolean casting
                return val_to_cast.lower() in ["true", "1", "yes", "y", "on"]
            elif cast_type == Decimal:
                return Decimal(val_to_cast)
            elif cast_type == int:
                # Use Decimal intermediary for potentially float-like strings ("10.0")
                dec_val = Decimal(val_to_cast)
                if dec_val % 1 != 0: # Check if it has fractional part
                    raise ValueError("Cannot cast non-integer Decimal to int")
                return int(dec_val)
            elif cast_type == str:
                 return val_to_cast # Already stripped
            else:
                # Default case: use the type constructor directly
                return cast_type(val_to_cast)
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.error(
                f"{Fore.RED}Cast failed for {key} ('{value_str}' -> {cast_type.__name__}): {e}. Using default '{default}'."
            )
            # On failure, return the provided default value directly
            # Assuming the default value itself is already of the correct type or structure
            return default

    def _validate_value(
        self,
        key: str,
        value: Any,
        min_val: Optional[Union[int, float, Decimal]],
        max_val: Optional[Union[int, float, Decimal]],
        allowed_values: Optional[List[Any]],
    ) -> bool:
        """Helper to validate value against constraints. Logs critical errors and exits if bounds violated."""
        # Type check before comparison
        value_type = type(value)
        if min_val is not None and not isinstance(value, (int, float, Decimal)):
             logger.error(f"Type error for {key}: Cannot compare min_val {min_val} with value '{value}' of type {value_type.__name__}.")
             return False # Treat as validation failure, might revert to default
        if max_val is not None and not isinstance(value, (int, float, Decimal)):
             logger.error(f"Type error for {key}: Cannot compare max_val {max_val} with value '{value}' of type {value_type.__name__}.")
             return False

        # Min/Max checks (Critical - Halt on failure)
        if min_val is not None and value < min_val:
            logger.critical(
                f"{Fore.RED+Style.BRIGHT}Validation failed for {key}: Value '{value}' < minimum '{min_val}'. Halting."
            )
            sys.exit(1)
        if max_val is not None and value > max_val:
            logger.critical(
                f"{Fore.RED+Style.BRIGHT}Validation failed for {key}: Value '{value}' > maximum '{max_val}'. Halting."
            )
            sys.exit(1)

        # Allowed values check (Non-critical - Allows fallback to default)
        if allowed_values:
            # Handle case-insensitive comparison for strings
            comp_value = str(value).lower() if isinstance(value, str) else value
            lower_allowed = [str(v).lower() if isinstance(v, str) else v for v in allowed_values]
            if comp_value not in lower_allowed:
                logger.error(
                    f"{Fore.RED}Validation failed for {key}: Invalid value '{value}'. Allowed: {allowed_values}."
                )
                return False # Signal validation failure

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
        is_secret: bool = False,
    ) -> Any:
        """Gets value from environment, casts, validates, logs, and handles defaults."""
        value_str = os.getenv(key)
        log_value = "****" if is_secret and value_str else value_str
        source = "environment"

        final_value_str: Optional[str] = None

        if value_str is None or value_str.strip() == "":
            if default is None and not is_secret: # Non-secret required config
                 logger.critical(
                    f"{Fore.RED+Style.BRIGHT}Required configuration '{key}' not found in environment and no default provided. Halting."
                 )
                 sys.exit(1)
            elif default is None and is_secret: # Secret required config
                 logger.critical(
                    f"{Fore.RED+Style.BRIGHT}Required secret configuration '{key}' not found in environment. Halting."
                 )
                 sys.exit(1)
            else:
                # Use default value
                final_value_str = str(default) # Cast default to string for processing
                source = f"default ({default})"
                logger.warning(f"{color}Using default for {key}: {default}")
        else:
            final_value_str = value_str
            log_display = "****" if is_secret else final_value_str
            logger.info(f"{color}Found {key}: {log_display} (from {source})")

        # --- Casting ---
        # We should always have a non-None string to cast at this point
        if final_value_str is None:
             # This case should be prevented by the checks above, but as a safeguard:
            logger.critical(f"{Fore.RED+Style.BRIGHT}Internal error: Could not determine value string for {key}. Halting.")
            sys.exit(1)

        # Attempt to cast the determined string value (from env or default)
        casted_value = self._cast_value(key, final_value_str, cast_type, default)

        # --- Validation ---
        # Validate the casted value
        if not self._validate_value(key, casted_value, min_val, max_val, allowed_values):
            # Validation failed for the value derived from env or the initial default string.
            # This occurs for allowed_values failures or type errors during validation.
            # Critical min/max failures would have already exited.
            # Revert to the original default value provided to the function.
            logger.warning(
                f"{color}Reverting {key} to original default '{default}' due to validation failure of value '{casted_value}'."
            )
            casted_value = default # Use the original default value

            # Re-validate the original default value itself (important!)
            if not self._validate_value(key, casted_value, min_val, max_val, allowed_values):
                # If the original default value *also* fails validation, it's a critical setup error.
                logger.critical(
                    f"{Fore.RED+Style.BRIGHT}FATAL: Default value '{default}' for {key} failed validation. Halting."
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
        else:
             # Critical failure already logged in _initialize_exchange, exiting handled there
             pass

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
                    "brokerId": "TermuxPyrmV5", # Custom ID for tracking via Bybit referrals
                    "createMarketBuyOrderRequiresPrice": False, # V5 specific option
                    "defaultMarginMode": "isolated", # Or 'cross', relevant for some calls
                    "v5": True, # Explicitly enable V5 API usage in options
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
                f"{Fore.GREEN}Bybit V5 interface initialized and connection tested successfully."
            )

        except ccxt.AuthenticationError as e:
            logger.critical(
                f"{Fore.RED+Style.BRIGHT}Authentication failed: {e}. Check API keys/permissions. Halting.",
                exc_info=False,
            )
            sys.exit(1)
        except (ccxt.NetworkError, requests.exceptions.RequestException) as e:
            logger.critical(
                f"{Fore.RED+Style.BRIGHT}Network error initializing exchange: {e}. Check connection. Halting.",
                exc_info=True,
            )
            sys.exit(1)
        except Exception as e:
            logger.critical(
                f"{Fore.RED+Style.BRIGHT}Unexpected error initializing exchange: {e}. Halting.",
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
                if prec_dec < 1: # Likely a step size (e.g., 0.01)
                     # Calculate dp from step size exponent
                    return abs(prec_dec.as_tuple().exponent)
                else: # Likely number of decimal places directly
                    return int(prec_dec)

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
                else "[dim]N/A[/]"
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
                f"{Fore.RED+Style.BRIGHT}Failed to load or parse market info for {self.config.symbol}: {e}. Halting.",
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

        if (
            self.market_info
            and "precision_dp" in self.market_info
            and "price" in self.market_info["precision_dp"]
        ):
            precision = self.market_info["precision_dp"]["price"]
            # Quantizer is 10^-precision (e.g., 0.01 for 2 dp)
            quantizer = Decimal("1e-" + str(precision))
            # ROUND_HALF_EVEN is standard for financial rounding
            formatted_price = price_decimal.quantize(quantizer, rounding=ROUND_HALF_EVEN)
            # Ensure the string representation matches the precision exactly
            return f"{formatted_price:.{precision}f}"
        else:
            logger.warning(
                "Market info/price precision unavailable, using default formatting."
            )
            # Fallback to a reasonable default precision if market info missing
            precision = DEFAULT_PRICE_DP
            quantizer = Decimal("1e-" + str(precision))
            formatted_price = price_decimal.quantize(quantizer, rounding=ROUND_HALF_EVEN)
            return f"{formatted_price:.{precision}f}"

    def format_amount(
        self,
        amount: Union[Decimal, str, float, int],
        rounding_mode=ROUND_DOWN, # Default to ROUND_DOWN for order quantities
    ) -> str:
        """Formats amount (quantity) according to market precision using specified rounding."""
        amount_decimal = safe_decimal(amount)
        if amount_decimal.is_nan():
            return "NaN" # Return NaN string if input was bad

        if (
            self.market_info
            and "precision_dp" in self.market_info
            and "amount" in self.market_info["precision_dp"]
        ):
            precision = self.market_info["precision_dp"]["amount"]
            quantizer = Decimal("1e-" + str(precision))
            formatted_amount = amount_decimal.quantize(quantizer, rounding=rounding_mode)
            # Ensure the string representation matches the precision exactly
            return f"{formatted_amount:.{precision}f}"
        else:
            logger.warning(
                "Market info/amount precision unavailable, using default formatting."
            )
            # Fallback to a reasonable default precision
            precision = DEFAULT_AMOUNT_DP
            quantizer = Decimal("1e-" + str(precision))
            formatted_amount = amount_decimal.quantize(quantizer, rounding=rounding_mode)
            return f"{formatted_amount:.{precision}f}"

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
                # Note: No explicit 'params' needed for V5 fetch_ohlcv usually
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
                # "category": self.config.bybit_v5_category, # May not be needed if defaultType is set
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
            # This structure can vary slightly. Common path: info.result.list[account].totalEquity etc.
            if (total_equity.is_nan() or available_balance.is_nan()) and "info" in balance_data:
                info_result = balance_data["info"].get("result", {})
                account_list = info_result.get("list", [])
                if account_list and isinstance(account_list, list):
                    # Find the Unified account details
                    unified_acc_info = next((item for item in account_list if item.get("accountType") == V5_UNIFIED_ACCOUNT_TYPE), None)
                    if unified_acc_info:
                        if total_equity.is_nan():
                            total_equity = safe_decimal(unified_acc_info.get("totalEquity"))
                        if available_balance.is_nan():
                            # V5 uses different keys, e.g., 'totalAvailableBalance' for the whole account
                            # or specific coin balances within 'coin' list
                            available_balance = safe_decimal(unified_acc_info.get("totalAvailableBalance"))

                        # If still NaN, check the specific coin details within the account
                        if available_balance.is_nan() and "coin" in unified_acc_info:
                             coin_list = unified_acc_info.get("coin", [])
                             if coin_list and isinstance(coin_list, list):
                                 settle_coin_info = next((c for c in coin_list if c.get("coin") == settle_currency), None)
                                 if settle_coin_info:
                                      # Keys like 'availableToBorrow', 'availableToWithdraw' might be relevant
                                      available_balance = safe_decimal(settle_coin_info.get("availableToWithdraw"))
                                      # Use coin equity if total equity was missing
                                      if total_equity.is_nan():
                                          total_equity = safe_decimal(settle_coin_info.get("equity"))

            # Final Validation and Logging
            if total_equity.is_nan():
                logger.error(
                    f"Could not extract valid total equity for {settle_currency}. Balance data might be incomplete or unexpected format. Raw snippet: {str(balance_data)[:500]}"
                )
                # Return None if critical info is missing
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
            # V5 fetch_positions requires category. Symbol and positionIdx filtering might happen client-side by CCXT
            # or server-side if params are passed correctly.
            params = {
                "category": self.config.bybit_v5_category,
                "symbol": market_id, # Filter by symbol server-side if possible
                # "positionIdx": self.config.position_idx, # Pass index if API uses it for filtering
                # "settleCoin": self.market_info.get("settle") # Optional filter by settle coin
            }
            # V5 fetch_positions often returns a list for the category, even if symbol is specified
            # CCXT might handle filtering, but we should verify.
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
                # Ensure matching symbol AND position index from the raw 'info' dict
                pos_symbol = info.get("symbol")
                # Position index from API is usually string, convert to int for comparison
                pos_idx_str = info.get("positionIdx")
                try:
                    pos_idx = int(pos_idx_str) if pos_idx_str is not None else -1
                except ValueError:
                    pos_idx = -1 # Handle non-integer index string

                if pos_symbol == market_id and pos_idx == self.config.position_idx:
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
            # Liq price might be '' or '0' when no position, handle this
            liq_price_raw = safe_decimal(target_pos_info.get("liqPrice", "0"))
            unrealized_pnl = safe_decimal(target_pos_info.get("unrealisedPnl", "0"))
            # SL/TP/TSL prices from position info
            sl_price_raw = safe_decimal(target_pos_info.get("stopLoss", "0"))
            tp_price_raw = safe_decimal(target_pos_info.get("takeProfit", "0"))
            # V5: 'trailingStop' field usually shows the active TSL *trigger price*, not the distance. '0' means inactive.
            tsl_trigger_price_raw = safe_decimal(target_pos_info.get("trailingStop", "0"))

            # --- Validate and clean up parsed values ---
            # Ensure quantity is non-negative (API sometimes returns negative for shorts, CCXT might standardize)
            qty_abs = qty.copy_abs() if not qty.is_nan() else Decimal("0")

            # Entry price should be > 0 if position exists
            entry_price = entry_price if not entry_price.is_nan() and entry_price > 0 else Decimal("NaN")

            # Liq price should be > 0 if applicable
            liq_price = liq_price_raw if not liq_price_raw.is_nan() and liq_price_raw > 0 else Decimal("NaN")

            # SL/TP: None if zero or NaN
            sl_price = sl_price_raw if not sl_price_raw.is_nan() and sl_price_raw > 0 else None
            tp_price = tp_price_raw if not tp_price_raw.is_nan() and tp_price_raw > 0 else None

            # Determine if TSL is active based on the trigger price being > 0
            is_tsl_active = not tsl_trigger_price_raw.is_nan() and tsl_trigger_price_raw > 0
            tsl_trigger_price = tsl_trigger_price_raw if is_tsl_active else None

            # Check if quantity is significant enough to consider the position open
            is_position_open = qty_abs >= POSITION_QTY_EPSILON

            if not is_position_open:
                logger.debug(f"Position size {qty} is negligible or zero. Considered flat.")
                return positions_dict # Return empty dict

            # Determine position side based on API 'side' field ('buy' or 'sell')
            position_side_key = "long" if side == "buy" else "short" if side == "sell" else None

            if position_side_key:
                position_details = {
                    "qty": qty_abs, # Store absolute quantity
                    "entry_price": entry_price,
                    "liq_price": liq_price,
                    "unrealized_pnl": unrealized_pnl if not unrealized_pnl.is_nan() else Decimal("0"),
                    "side": side, # Store original API side ('buy'/'sell')
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
            elif side != "none":
                # This case might occur if size is non-zero but side is unexpected
                 logger.warning(f"Position found with size {qty} but side is '{side}'. Ignoring.")
            else: # side == "none"
                 logger.debug("Position side is 'None' but size is non-zero? Treating as flat for safety.")
                 return positions_dict # Return empty

            return positions_dict

        except Exception as e:
            logger.error(
                f"Failed to fetch or parse positions for {self.config.symbol}: {e}", exc_info=True
            )
            # Return None to indicate failure, distinct from an empty dict (flat)
            return None


# --- Indicator Calculator Class ---
class IndicatorCalculator:
    """Calculates technical indicators needed for the strategy."""

    def __init__(self, config: TradingConfig):
        self.config = config

    def calculate_indicators(
        self, df: pd.DataFrame
    ) -> Optional[Dict[str, Union[Decimal, bool, int]]]:
        """Calculates EMAs, Stochastic, ATR, ADX from OHLCV data."""
        logger.info(
            f"{Fore.CYAN}# Weaving indicator patterns (EMA, Stoch, ATR, ADX)..."
        )
        if df is None or df.empty:
            logger.error(f"{Fore.RED}No DataFrame provided for indicators.")
            return None

        required_cols = ["open", "high", "low", "close"]
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            logger.error(
                f"{Fore.RED}DataFrame missing required columns for indicators: {missing}"
            )
            return None

        try:
            # Work with a copy, ensure numeric types (handle potential Decimals from fetch)
            # Select only necessary columns to reduce memory usage
            df_calc = df[required_cols].copy()

            # Convert Decimal columns to float for TA-Lib/Pandas, map NaN Decimals to np.nan
            for col in required_cols:
                if isinstance(df_calc[col].iloc[0], Decimal): # Check type of first element
                    df_calc[col] = df_calc[col].map(
                        lambda x: float(x) if isinstance(x, Decimal) and not x.is_nan() else np.nan
                    )
                elif pd.api.types.is_numeric_dtype(df_calc[col]):
                     # Already numeric (float/int), ensure NaNs are handled if any
                     df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')
                else:
                     # Attempt conversion if it's object/string type
                     logger.warning(f"Column {col} has unexpected type {df_calc[col].dtype}, attempting numeric conversion.")
                     df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')


            # Drop rows with NaN in essential OHLC columns *after* conversion
            initial_len = len(df_calc)
            df_calc.dropna(subset=required_cols, inplace=True)
            if len(df_calc) < initial_len:
                 logger.debug(f"Dropped {initial_len - len(df_calc)} rows with NaN OHLC after float conversion.")

            if df_calc.empty:
                logger.error(
                    f"{Fore.RED}DataFrame empty after NaN drop during indicator calculation."
                )
                return None

            # --- Check Data Length ---
            # Determine the maximum period needed by any indicator + buffer
            # ADX needs roughly 2x period for proper calculation/smoothing
            # Stochastic needs period + smooth_k + smooth_d lookback
            max_period = max(
                self.config.slow_ema_period,
                self.config.trend_ema_period,
                self.config.stoch_period + self.config.stoch_smooth_k + self.config.stoch_smooth_d,
                self.config.atr_period,
                self.config.adx_period * 2, # ADX smoothing requirement
            )
            min_required_len = max_period + 10 # Add a buffer for stability
            if len(df_calc) < min_required_len:
                logger.error(
                    f"{Fore.RED}Insufficient data ({len(df_calc)} rows < required ~{min_required_len}) for reliable indicator calculation with current periods."
                )
                return None

            # --- Access Series for Calculation ---
            close_s = df_calc["close"]
            high_s = df_calc["high"]
            low_s = df_calc["low"]

            # --- Calculate Indicators using Pandas/NumPy ---
            # Exponential Moving Averages (EMAs)
            # adjust=False mimics common TA software behavior
            fast_ema_s = close_s.ewm(span=self.config.fast_ema_period, adjust=False).mean()
            slow_ema_s = close_s.ewm(span=self.config.slow_ema_period, adjust=False).mean()
            trend_ema_s = close_s.ewm(span=self.config.trend_ema_period, adjust=False).mean()

            # Stochastic Oscillator (%K, %D)
            low_min = low_s.rolling(window=self.config.stoch_period).min()
            high_max = high_s.rolling(window=self.config.stoch_period).max()
            stoch_range = high_max - low_min
            # Use np.where for safe division, default to 50 if range is near zero
            stoch_k_raw = np.where(
                stoch_range > 1e-12, # Avoid division by zero/tiny numbers
                100 * (close_s - low_min) / stoch_range,
                50.0, # Default value if range is zero
            )
            stoch_k_raw_s = pd.Series(stoch_k_raw, index=df_calc.index).fillna(50) # Fill NaNs from initial rolling window
            # Smooth %K (becomes the final %K)
            stoch_k_s = stoch_k_raw_s.rolling(window=self.config.stoch_smooth_k).mean().fillna(50)
            # Smooth %K to get %D
            stoch_d_s = stoch_k_s.rolling(window=self.config.stoch_smooth_d).mean().fillna(50)

            # Average True Range (ATR - Using Wilder's smoothing via EMA)
            prev_close = close_s.shift(1)
            tr1 = high_s - low_s
            tr2 = (high_s - prev_close).abs()
            tr3 = (low_s - prev_close).abs()
            tr_s = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(0) # Calculate True Range
            # Use EMA for Wilder's smoothing (alpha = 1 / period)
            atr_s = tr_s.ewm(alpha=1 / self.config.atr_period, adjust=False).mean()

            # ADX, +DI, -DI (using helper for clarity)
            adx_s, pdi_s, mdi_s = self._calculate_adx(
                high_s, low_s, close_s, atr_s, self.config.adx_period
            )

            # --- Extract Latest Values & Convert back to Decimal ---
            def get_latest_decimal(series: pd.Series, name: str) -> Decimal:
                """Safely get the last valid value from a Series and convert to Decimal."""
                # Get the last non-NaN value
                last_valid = series.dropna().iloc[-1] if not series.dropna().empty else None
                if last_valid is None:
                    # logger.warning(f"Indicator '{name}' calculation resulted in all NaNs or empty series.")
                    return Decimal("NaN")
                try:
                    # Convert float to string first for precise Decimal conversion
                    return Decimal(str(last_valid))
                except (InvalidOperation, TypeError):
                    logger.error(
                        f"Failed converting latest {name} value '{last_valid}' (type: {type(last_valid).__name__}) to Decimal."
                    )
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

            # --- Calculate Stochastic Cross Signals using latest Decimal values ---
            # Requires comparing latest k/d with previous k/d
            k_last = indicators_out["stoch_k"]
            d_last = indicators_out["stoch_d"]

            # Get previous values safely
            stoch_k_valid = stoch_k_s.dropna()
            stoch_d_valid = stoch_d_s.dropna()
            k_prev = Decimal("NaN")
            d_prev = Decimal("NaN")
            if len(stoch_k_valid) >= 2:
                 k_prev = safe_decimal(str(stoch_k_valid.iloc[-2]))
            if len(stoch_d_valid) >= 2:
                 d_prev = safe_decimal(str(stoch_d_valid.iloc[-2]))


            stoch_kd_bullish = False
            stoch_kd_bearish = False
            if not any(v.is_nan() for v in [k_last, d_last, k_prev, d_prev]):
                # Check for cross
                crossed_above = (k_last > d_last) and (k_prev <= d_prev)
                crossed_below = (k_last < d_last) and (k_prev >= d_prev)

                # Check if the cross occurred from the respective zones
                # Check if previous K or D was in the zone for confirmation
                prev_in_oversold = (k_prev <= self.config.stoch_oversold_threshold) or \
                                   (d_prev <= self.config.stoch_oversold_threshold)
                prev_in_overbought = (k_prev >= self.config.stoch_overbought_threshold) or \
                                     (d_prev >= self.config.stoch_overbought_threshold)

                if crossed_above and prev_in_oversold:
                    stoch_kd_bullish = True
                if crossed_below and prev_in_overbought:
                    stoch_kd_bearish = True

            indicators_out["stoch_kd_bullish"] = stoch_kd_bullish
            indicators_out["stoch_kd_bearish"] = stoch_kd_bearish

            # Final check for critical NaNs in output dict
            critical_keys = [
                "fast_ema", "slow_ema", "trend_ema", "atr",
                "stoch_k", "stoch_d", "adx", "pdi", "mdi",
            ]
            failed_indicators = [
                k for k in critical_keys if indicators_out.get(k, Decimal("NaN")).is_nan()
            ]
            if failed_indicators:
                logger.error(
                    f"{Fore.RED}Critical indicators calculated as NaN: {', '.join(failed_indicators)}. This may prevent signal generation. Check data source or indicator periods."
                )
                # Return None only if ATR is NaN, as it's crucial for risk management
                if indicators_out.get("atr", Decimal("NaN")).is_nan():
                     logger.error(f"{Fore.RED}ATR is NaN, cannot proceed with risk calculations. Aborting cycle.")
                     return None
                # Allow returning partial indicators if only less critical ones failed

            logger.info(f"{Fore.GREEN}Indicator patterns woven successfully.")
            return indicators_out

        except Exception as e:
            logger.error(
                f"{Fore.RED}Error weaving indicator patterns: {e}", exc_info=True
            )
            return None

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
        if atr_s.empty:
             raise ValueError("ATR series is empty, cannot calculate ADX")

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
        # Handle division by zero if ATR is zero or near zero
        pdi_s_raw = np.where(atr_s > 1e-12, 100 * plus_dm_s / atr_s, 0.0)
        mdi_s_raw = np.where(atr_s > 1e-12, 100 * minus_dm_s / atr_s, 0.0)
        pdi_s = pd.Series(pdi_s_raw, index=high_s.index).fillna(0)
        mdi_s = pd.Series(mdi_s_raw, index=high_s.index).fillna(0)

        # Calculate Directional Movement Index (DX)
        di_diff_abs = (pdi_s - mdi_s).abs()
        di_sum = pdi_s + mdi_s
        # Handle division by zero if sum is zero or near zero
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

            # Validate current price
            if current_price.is_nan() or current_price <= 0:
                result["reason"] = f"No Signal: Invalid current price ({current_price})"
                logger.warning(result["reason"])
                return result

            # Get required indicator values, checking for NaN
            required_indicator_keys = [
                "stoch_k", "fast_ema", "slow_ema", "trend_ema", "atr", "adx", "pdi", "mdi"
            ]
            ind_values = {key: indicators.get(key) for key in required_indicator_keys}
            nan_keys = [name for name, val in ind_values.items() if isinstance(val, Decimal) and val.is_nan()]

            if nan_keys:
                result["reason"] = f"No Signal: Required indicator(s) NaN: {', '.join(nan_keys)}"
                logger.warning(result["reason"])
                return result

            # Assign validated indicators to local variables for readability
            k, fast_ema, slow_ema, trend_ema, atr, adx, pdi, mdi = (
                ind_values["stoch_k"], ind_values["fast_ema"], ind_values["slow_ema"],
                ind_values["trend_ema"], ind_values["atr"], ind_values["adx"],
                ind_values["pdi"], ind_values["mdi"]
            )
            # Get boolean stochastic cross signals
            kd_bull = indicators.get("stoch_kd_bullish", False)
            kd_bear = indicators.get("stoch_kd_bearish", False)

            # --- Define Conditions ---

            # 1. EMA Cross
            ema_bullish_cross = fast_ema > slow_ema
            ema_bearish_cross = fast_ema < slow_ema

            # 2. Trend Filter (Price relative to Trend EMA with buffer)
            trend_buffer = trend_ema.copy_abs() * (self.config.trend_filter_buffer_percent / 100)
            price_above_trend_ema = current_price > (trend_ema - trend_buffer)
            price_below_trend_ema = current_price < (trend_ema + trend_buffer)
            # Apply filter only if enabled in config
            trend_allows_long = price_above_trend_ema if self.config.trade_only_with_trend else True
            trend_allows_short = price_below_trend_ema if self.config.trade_only_with_trend else True
            trend_reason = f"Trend(P:{current_price:.{DEFAULT_PRICE_DP}f} vs EMA:{trend_ema:.{DEFAULT_PRICE_DP}f})" if self.config.trade_only_with_trend else "TrendFilter OFF"

            # 3. Stochastic Condition (K level or KD Cross from zone)
            stoch_long_cond = (k < self.config.stoch_oversold_threshold) or kd_bull
            stoch_short_cond = (k > self.config.stoch_overbought_threshold) or kd_bear
            stoch_reason = f"Stoch(K:{k:.1f} {'BullX' if kd_bull else ''}{'BearX' if kd_bear else ''})"

            # 4. ATR Move Filter (Price moved more than X * ATR since previous close)
            atr_move_threshold = atr * self.config.atr_move_filter_multiplier
            significant_move = False
            atr_reason = "ATR Filter OFF"
            if atr_move_threshold > 0: # Filter is enabled
                if atr.is_nan() or atr <= 0:
                    atr_reason = f"ATR Filter Skipped (Invalid ATR: {atr})"
                    significant_move = False # Cannot evaluate
                elif prev_close.is_nan():
                    atr_reason = "ATR Filter Skipped (Prev Close NaN)"
                    significant_move = False # Cannot evaluate
                else:
                    price_move = (current_price - prev_close).copy_abs()
                    significant_move = price_move > atr_move_threshold
                    atr_reason = f"ATR Move({price_move:.{DEFAULT_PRICE_DP}f}) {'OK' if significant_move else 'LOW'} vs Thr({atr_move_threshold:.{DEFAULT_PRICE_DP}f})"
            else: # Filter is disabled (multiplier is 0)
                significant_move = True # Always pass if filter is off

            # 5. ADX Filter (Market is trending and direction matches)
            adx_is_trending = adx > self.config.min_adx_level
            adx_long_direction = pdi > mdi
            adx_short_direction = mdi > pdi
            adx_allows_long = adx_is_trending and adx_long_direction
            adx_allows_short = adx_is_trending and adx_short_direction
            adx_reason = f"ADX({adx:.1f}) {'OK' if adx_is_trending else 'LOW'} vs {self.config.min_adx_level:.1f} | Dir({'+DI' if adx_long_direction else '-DI' if adx_short_direction else 'NONE'})"

            # --- Combine Logic for Entry Signals ---
            # Base conditions combine EMA cross and Stochastic confirmation
            base_long_signal = ema_bullish_cross and stoch_long_cond
            base_short_signal = ema_bearish_cross and stoch_short_cond

            # Final signal requires base + trend filter + ATR filter + ADX filter
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
                # Provide detailed reason why no signal was generated
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
                else: # Catch-all if logic missed a specific blocking condition
                    reason_parts.append(f"Conditions unmet (EMA:{ema_bullish_cross}/{ema_bearish_cross}, Stoch:{stoch_long_cond}/{stoch_short_cond}, Trend:{trend_allows_long}/{trend_allows_short}, ATR:{significant_move}, ADX:{adx_allows_long}/{adx_allows_short})")
                result["reason"] = " | ".join(reason_parts)

            # Log signal check result (INFO for signals/blocks, DEBUG otherwise)
            log_level_sig = logging.INFO if result["long"] or result["short"] or "Blocked" in result["reason"] else logging.DEBUG
            logger.log(log_level_sig, f"Signal Check: {result['reason']}")

        except Exception as e:
            logger.error(f"{Fore.RED}Error generating signals: {e}", exc_info=True)
            result["reason"] = f"No Signal: Exception ({e})"
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

        # Extract necessary indicators safely
        fast_ema = indicators.get("fast_ema", Decimal("NaN"))
        slow_ema = indicators.get("slow_ema", Decimal("NaN"))
        stoch_k = indicators.get("stoch_k", Decimal("NaN"))

        # Check for NaN values in critical indicators
        if fast_ema.is_nan() or slow_ema.is_nan() or stoch_k.is_nan():
            logger.warning(
                "Cannot check exit signals due to NaN indicators (EMA/Stoch)."
            )
            return None

        exit_reason: Optional[str] = None

        # --- Define Exit Conditions ---
        # EMA Crosses against the position
        ema_bearish_cross = fast_ema < slow_ema
        ema_bullish_cross = fast_ema > slow_ema

        # Stochastic level indicating potential reversal against the position
        stoch_reached_overbought = stoch_k > self.config.stoch_overbought_threshold
        stoch_reached_oversold = stoch_k < self.config.stoch_oversold_threshold

        # --- Check Exit Logic based on Position Side ---
        if position_side == "long":
            # Exit long if EMA crosses bearish OR Stochastic reaches overbought
            if ema_bearish_cross:
                exit_reason = "Exit Signal: EMA Bearish Cross"
            elif stoch_reached_overbought:
                exit_reason = f"Exit Signal: Stoch Overbought ({stoch_k:.1f} > {self.config.stoch_overbought_threshold})"

        elif position_side == "short":
            # Exit short if EMA crosses bullish OR Stochastic reaches oversold
            if ema_bullish_cross:
                exit_reason = "Exit Signal: EMA Bullish Cross"
            elif stoch_reached_oversold:
                exit_reason = f"Exit Signal: Stoch Oversold ({stoch_k:.1f} < {self.config.stoch_oversold_threshold})"

        if exit_reason:
            logger.trade( # Use TRADE level for exit signals
                f"{Fore.YELLOW}{exit_reason} detected against {position_side.upper()} position."
            )

        return exit_reason


# --- Order Manager Class ---
class OrderManager:
    """Handles order placement, position protection (SL/TP/TSL), and closing using V5 API."""

    def __init__(
        self, config: TradingConfig, exchange_manager: ExchangeManager
    ):
        self.config = config
        self.exchange_manager = exchange_manager
        self.exchange = exchange_manager.exchange # Convenience accessor
        self.market_info = exchange_manager.market_info # Convenience accessor
        # Tracks active protection STATUS (not order IDs) for V5 position stops
        # Key: 'long' or 'short', Value: 'ACTIVE_SLTP', 'ACTIVE_TSL', None
        self.protection_tracker: Dict[str, Optional[str]] = {
            "long": None,
            "short": None,
        }

    def _calculate_trade_parameters(
        self,
        side: str, # 'buy' or 'sell'
        atr: Decimal,
        total_equity: Decimal,
        current_price: Decimal,
    ) -> Optional[Dict[str, Decimal]]:
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
            # --- Risk Amount ---
            risk_amount_per_trade = total_equity * self.config.risk_percentage

            # --- Stop Loss Calculation ---
            sl_distance_atr = atr * self.config.sl_atr_multiplier
            sl_price: Decimal
            if side == "buy":
                sl_price = current_price - sl_distance_atr
            else: # side == "sell"
                sl_price = current_price + sl_distance_atr

            # Validate SL price > 0
            if sl_price <= 0:
                logger.error(f"Calculated SL price ({sl_price:.{DEFAULT_PRICE_DP}f}) is invalid (<=0). Cannot calculate trade size.")
                return None

            # Ensure SL distance is meaningful (at least one tick size)
            sl_distance_price = (current_price - sl_price).copy_abs()
            min_tick_size = self.market_info['tick_size']
            if sl_distance_price < min_tick_size:
                sl_distance_price = min_tick_size # Adjust distance to minimum tick
                # Recalculate SL price based on minimum distance
                if side == "buy":
                    sl_price = current_price - sl_distance_price
                else: # side == "sell"
                    sl_price = current_price + sl_distance_price
                logger.warning(f"Initial SL distance was less than tick size. Adjusted SL price to {sl_price:.{DEFAULT_PRICE_DP}f} (distance {sl_distance_price}).")
                if sl_price <= 0: # Re-check after adjustment
                     logger.error(f"Adjusted SL price ({sl_price:.{DEFAULT_PRICE_DP}f}) is still invalid (<=0). Cannot calculate trade size.")
                     return None

            # --- Quantity Calculation ---
            # Quantity = Risk Amount / (Stop Loss Price Distance * Contract Value per Point)
            # For Linear contracts (USDT settled): Contract Value = Contract Size (usually 1 USDT)
            # For Inverse contracts (BTC settled): Contract Value = Contract Size / Price (e.g., 1 USD / BTC Price)
            contract_size = self.market_info['contract_size']
            value_per_point: Decimal
            if self.config.market_type == "inverse":
                 # Inverse: Value per point changes with price
                 value_per_point = contract_size / current_price # Approx using current price
            else: # Linear/Swap
                 # Linear: Value per point is fixed by contract size (usually 1 USDT)
                 value_per_point = contract_size

            if value_per_point <= 0:
                 logger.error(f"Calculated invalid value_per_point ({value_per_point}). Cannot size trade.")
                 return None

            # Risk per contract = SL Distance * Value Per Point
            risk_per_contract = sl_distance_price * value_per_point
            if risk_per_contract <= 0:
                logger.error(f"Calculated zero or negative risk per contract ({risk_per_contract}). Cannot size trade.")
                return None

            quantity = risk_amount_per_trade / risk_per_contract

            # --- Format and Validate Quantity ---
            # Format quantity based on market amount precision, rounding down
            quantity_str = self.exchange_manager.format_amount(quantity, rounding_mode=ROUND_DOWN)
            quantity_decimal = safe_decimal(quantity_str)

            if quantity_decimal.is_nan() or quantity_decimal <= 0:
                 logger.error(f"Calculated quantity ({quantity_str}) is invalid or zero.")
                 return None

            # Check against minimum order size
            min_order_size = self.market_info.get('min_order_size', Decimal('NaN'))
            if not min_order_size.is_nan() and quantity_decimal < min_order_size:
                logger.error(
                    f"Calculated quantity {quantity_decimal.normalize()} is below exchange minimum {min_order_size.normalize()}. Cannot place order."
                )
                return None

            # --- Take Profit Calculation ---
            tp_price: Optional[Decimal] = None
            if self.config.tp_atr_multiplier > 0:
                tp_distance_atr = atr * self.config.tp_atr_multiplier
                if side == "buy":
                    tp_price = current_price + tp_distance_atr
                else: # side == "sell"
                    tp_price = current_price - tp_distance_atr
                # Validate TP price > 0
                if tp_price <= 0:
                    logger.warning(f"Calculated TP price ({tp_price:.{DEFAULT_PRICE_DP}f}) is invalid (<=0). Disabling TP.")
                    tp_price = None

            # --- Trailing Stop Calculation (Distance) ---
            # TSL distance is a price difference, needs formatting like price
            tsl_distance_price = current_price * (self.config.trailing_stop_percent / 100)
            # Ensure TSL distance is at least one tick
            if tsl_distance_price < min_tick_size:
                 tsl_distance_price = min_tick_size
            # Format the distance using price precision
            tsl_distance_str = self.exchange_manager.format_price(tsl_distance_price)
            tsl_distance_decimal = safe_decimal(tsl_distance_str)
            if tsl_distance_decimal.is_nan() or tsl_distance_decimal <= 0:
                 logger.warning(f"Calculated invalid TSL distance ({tsl_distance_str}). TSL might not function correctly.")
                 # Still return params, but TSL might fail later
                 tsl_distance_decimal = Decimal('NaN') # Mark as invalid


            # --- Format Final SL/TP Prices ---
            sl_price_str = self.exchange_manager.format_price(sl_price)
            sl_price_decimal = safe_decimal(sl_price_str)
            # Format TP only if it's valid
            tp_price_decimal = None
            if tp_price is not None:
                 tp_price_str = self.exchange_manager.format_price(tp_price)
                 tp_price_decimal = safe_decimal(tp_price_str)
                 # If formatting/parsing failed, disable TP
                 if tp_price_decimal.is_nan() or tp_price_decimal <= 0:
                      logger.warning(f"Failed to format valid TP price ({tp_price_str}). Disabling TP.")
                      tp_price_decimal = None

            # --- Return Calculated Parameters ---
            params_out = {
                "qty": quantity_decimal,
                "sl_price": sl_price_decimal,
                "tp_price": tp_price_decimal, # Will be None if disabled or invalid
                "tsl_distance": tsl_distance_decimal, # Price difference for TSL
            }

            # Log the calculated parameters clearly
            log_tp_str = f"{params_out['tp_price'].normalize()}" if params_out['tp_price'] else "None"
            log_tsl_str = f"{params_out['tsl_distance'].normalize()}" if not params_out['tsl_distance'].is_nan() else "Invalid"
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
            logger.error(
                f"Error calculating trade parameters for {side} side: {e}", exc_info=True
            )
            return None

    def _execute_market_order(
        self, side: str, qty_decimal: Decimal
    ) -> Optional[Dict]:
        """Executes a market order with retries and basic confirmation."""
        if not self.exchange or not self.market_info:
            logger.error("Cannot execute market order: Exchange/Market info missing.")
            return None

        symbol = self.config.symbol
        # Format quantity strictly using market precision and rounding down
        qty_str = self.exchange_manager.format_amount(qty_decimal, rounding_mode=ROUND_DOWN)
        final_qty_decimal = safe_decimal(qty_str)

        if final_qty_decimal.is_nan() or final_qty_decimal <= 0:
            logger.error(f"Attempted market order with zero/invalid formatted quantity: {qty_str} (Original: {qty_decimal})")
            return None

        logger.trade( # Use TRADE level for order execution attempts
            f"{Fore.CYAN}Attempting MARKET {side.upper()} order: {final_qty_decimal.normalize()} {symbol}..."
        )
        try:
            # V5 create_order requires category. Other params depend on hedge mode etc.
            # CCXT `create_market_order` should handle mapping.
            params = {
                "category": self.config.bybit_v5_category,
                "positionIdx": self.config.position_idx, # Specify hedge mode index
                # 'reduceOnly': False, # Ensure it's not a reduce-only order for entries
                # 'timeInForce': 'ImmediateOrCancel' or 'FillOrKill' could be options, but market usually fills
            }
            # CCXT often prefers float for amount in create_order methods
            # Convert our precise Decimal to float for the call
            amount_float = float(final_qty_decimal)

            order = fetch_with_retries(
                self.exchange.create_market_order, # Pass function
                symbol=symbol,
                side=side,
                amount=amount_float,
                # price=None, # Not needed for market order
                params=params,
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
            )

            # Process the order response
            order_id = order.get("id", "[N/A]")
            order_status = order.get("status", "[unknown]") # e.g., 'open', 'closed', 'canceled'
            filled_qty_str = order.get("filled", "0") # Amount filled
            avg_fill_price_str = order.get("average", "0") # Average fill price

            filled_qty = safe_decimal(filled_qty_str)
            avg_fill_price = safe_decimal(avg_fill_price_str)

            logger.trade(
                f"{Fore.GREEN}Market order submitted: ID {order_id}, Side {side.upper()}, Qty {final_qty_decimal.normalize()}, Status: {order_status}, Filled: {filled_qty.normalize()}, AvgPx: {avg_fill_price.normalize() if avg_fill_price > 0 else '[N/A]'}"
            )
            termux_notify(
                f"{symbol} Order Submitted", f"Market {side.upper()} {final_qty_decimal.normalize()} ID:{order_id}"
            )

            # Basic check: If status is immediately 'rejected' or 'canceled'
            if order_status in ["rejected", "canceled"]:
                 logger.error(f"{Fore.RED}Market order {order_id} was {order_status}. Check exchange reason if available in order info: {order.get('info')}")
                 return None # Treat as failure

            # Simple delay for propagation - More robust checks happen in the calling function (place_risked_market_order)
            logger.debug(
                f"Waiting {self.config.order_check_delay_seconds}s for order {order_id} propagation..."
            )
            time.sleep(self.config.order_check_delay_seconds)

            return order # Return submitted order info (may not be fully filled yet)

        except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e:
            # Specific, often non-retryable errors
            logger.error(
                f"{Fore.RED}Order placement failed ({type(e).__name__}): {e}"
            )
            termux_notify(
                f"{symbol} Order FAILED",
                f"Market {side.upper()} failed: {str(e)[:50]}",
            )
            return None
        except Exception as e:
            # Catch exceptions from fetch_with_retries or other issues
            logger.error(
                f"{Fore.RED}Unexpected error placing market order: {e}",
                exc_info=True,
            )
            termux_notify(
                f"{symbol} Order ERROR", f"Market {side.upper()} error."
            )
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
        """Sets SL, TP, or TSL for a position using V5 setTradingStop.
        Ensures numeric parameters are passed as formatted strings. Clears other stops when activating TSL.
        Updates internal protection_tracker on success.
        """
        if not self.exchange or not self.market_info:
            logger.error("Cannot set protection: Exchange/Market info missing.")
            return False

        symbol = self.config.symbol
        market_id = self.market_info.get("id")
        if not market_id:
             logger.error("Cannot set protection: Market ID missing.")
             return False

        # Determine the tracker key ('long' or 'short')
        tracker_key = position_side.lower()
        if tracker_key not in self.protection_tracker:
             logger.error(f"Invalid position_side '{position_side}' for protection tracker.")
             return False

        # --- Prepare Base Parameters for V5 privatePostPositionSetTradingStop ---
        # Note: CCXT might map set_trading_stop or similar; using the direct V5 method name for clarity
        # Ensure all numeric price/distance values are formatted as strings using market precision.
        params: Dict[str, Any] = {
            "category": self.config.bybit_v5_category,
            "symbol": market_id,
            "positionIdx": self.config.position_idx,
            "tpslMode": V5_TPSL_MODE_FULL, # Apply to the entire position
            # Default to clearing stops unless specific values are provided below
            "stopLoss": "0",
            "takeProfit": "0",
            "trailingStop": "0", # This is the TSL *distance* string (price difference)
            "activePrice": "0", # TSL activation price string
            "slTriggerBy": self.config.sl_trigger_by,
            "tpTriggerBy": self.config.sl_trigger_by, # Use same trigger for TP as SL
            "tslTriggerBy": self.config.tsl_trigger_by,
            # slOrderType, tpOrderType default to Market
        }

        action_desc = ""
        new_tracker_state: Optional[str] = None

        # --- Logic Branch: Activate TSL ---
        if (is_tsl and tsl_distance is not None and not tsl_distance.is_nan() and tsl_distance > 0
            and tsl_activation_price is not None and not tsl_activation_price.is_nan() and tsl_activation_price > 0):

            # Format distance and activation price as strings using PRICE precision
            tsl_distance_str = self.exchange_manager.format_price(tsl_distance)
            tsl_activation_price_str = self.exchange_manager.format_price(tsl_activation_price)

            # Check if formatting resulted in valid positive numbers
            if safe_decimal(tsl_distance_str) <= 0 or safe_decimal(tsl_activation_price_str) <= 0:
                 logger.error(f"Failed to format valid positive TSL distance ({tsl_distance_str}) or activation price ({tsl_activation_price_str}). Cannot activate TSL.")
                 return False

            params["trailingStop"] = tsl_distance_str
            params["activePrice"] = tsl_activation_price_str
            # Crucially, clear fixed SL/TP when activating TSL for V5 API behavior
            params["stopLoss"] = "0"
            params["takeProfit"] = "0"

            action_desc = f"ACTIVATE TSL (Dist: {tsl_distance_str}, ActPx: {tsl_activation_price_str})"
            new_tracker_state = "ACTIVE_TSL"
            logger.debug(f"TSL Params Prepared: trailingStop={params['trailingStop']}, activePrice={params['activePrice']}, stopLoss=0, takeProfit=0")

        # --- Logic Branch: Set Fixed SL/TP ---
        # Only if not activating TSL and at least one valid SL or TP price is given
        elif not is_tsl and (
             (sl_price is not None and not sl_price.is_nan() and sl_price > 0) or
             (tp_price is not None and not tp_price.is_nan() and tp_price > 0)
            ):
            # Format valid SL/TP prices as strings, use '0' if invalid or None
            sl_str = "0"
            if sl_price is not None and not sl_price.is_nan() and sl_price > 0:
                sl_str = self.exchange_manager.format_price(sl_price)
                if safe_decimal(sl_str) <= 0: # Double check formatted value
                     logger.warning(f"Formatted SL price ({sl_str}) is invalid. Setting SL to 0.")
                     sl_str = "0"

            tp_str = "0"
            if tp_price is not None and not tp_price.is_nan() and tp_price > 0:
                tp_str = self.exchange_manager.format_price(tp_price)
                if safe_decimal(tp_str) <= 0: # Double check formatted value
                     logger.warning(f"Formatted TP price ({tp_str}) is invalid. Setting TP to 0.")
                     tp_str = "0"

            params["stopLoss"] = sl_str
            params["takeProfit"] = tp_str
            # Ensure TSL fields are cleared when setting fixed stops
            params["trailingStop"] = "0"
            params["activePrice"] = "0"

            if params["stopLoss"] != "0" or params["takeProfit"] != "0":
                action_desc = f"SET SL={params['stopLoss']} TP={params['takeProfit']}"
                new_tracker_state = "ACTIVE_SLTP"
            else:
                # This case means both sl_price and tp_price were invalid/None/zero
                action_desc = "CLEAR SL/TP (Invalid inputs provided)"
                new_tracker_state = None # Effectively clearing stops

        # --- Logic Branch: Clear All Stops ---
        else:
            # This branch is hit if:
            # - is_tsl=False and both sl_price and tp_price are None/invalid/zero
            # - Or if called explicitly with no stop info (e.g., during position close)
            action_desc = "CLEAR SL/TP/TSL"
            # Ensure all stop parameters are explicitly set to "0"
            params["stopLoss"] = "0"
            params["takeProfit"] = "0"
            params["trailingStop"] = "0"
            params["activePrice"] = "0"
            new_tracker_state = None
            logger.debug(f"Clearing all stops for {position_side.upper()} position.")

        # --- Execute API Call ---
        logger.trade( # Use TRADE level for protection setting attempts
            f"{Fore.CYAN}Attempting to {action_desc} for {position_side.upper()} {symbol}..."
        )

        try:
            # Use CCXT's implicit method mapping or explicit private method call
            # Common CCXT method: set_trading_stop (might exist)
            # Direct V5 method via private call: privatePostPositionSetTradingStop
            method_to_call: Optional[Callable] = None
            if hasattr(self.exchange, 'set_trading_stop'):
                # Prefer CCXT unified method if available and tested
                # Note: Need to verify if set_trading_stop correctly handles all V5 params
                # For now, using private call for guaranteed parameter mapping
                # method_to_call = self.exchange.set_trading_stop # Needs testing
                pass # Skip for now

            # Fallback to private method call (more reliable for specific V5 features)
            if method_to_call is None:
                 v5_method_name = "privatePostPositionSetTradingStop"
                 if not hasattr(self.exchange, v5_method_name):
                     raise NotImplementedError(
                         f"Required V5 method '{v5_method_name}' not found in CCXT Bybit instance."
                     )
                 method_to_call = getattr(self.exchange, v5_method_name)

            if method_to_call is None: # Should not happen if private method exists
                 raise RuntimeError("Could not find suitable method to set trading stop.")

            logger.debug(f"Calling exchange method '{getattr(method_to_call, '__name__', 'N/A')}' with params: {params}")

            # Use fetch_with_retries for the API call
            response = fetch_with_retries(
                method_to_call, # Pass the bound method
                params=params, # Pass params dict
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
            )
            # logger.debug(f"SetTradingStop Response: {response}") # Verbose

            # --- Process Response ---
            # V5 response check (retCode 0 indicates success)
            if response and response.get("retCode") == V5_SUCCESS_RETCODE:
                logger.trade( # Use TRADE level for success
                    f"{Fore.GREEN}{action_desc} successful for {position_side.upper()} {symbol}."
                )
                termux_notify(
                    f"{symbol} Protection Set",
                    f"{action_desc} {position_side.upper()}",
                )
                # Update protection tracker ONLY on success
                self.protection_tracker[tracker_key] = new_tracker_state
                return True
            else:
                # Handle API error response
                ret_code = response.get("retCode", "[N/A]")
                ret_msg = response.get("retMsg", "[No error message]")
                logger.error(
                    f"{Fore.RED}{action_desc} failed for {position_side.upper()} {symbol}. API Response: Code={ret_code}, Msg='{ret_msg}'"
                )
                # Log full response at debug level if helpful
                logger.debug(f"Failed SetTradingStop Full Response: {response}")
                # Do NOT update tracker on failure, keep previous state
                termux_notify(
                    f"{symbol} Protection FAILED",
                    f"{action_desc[:30]} {position_side.upper()} failed: {ret_msg[:50]}",
                )
                return False
        except Exception as e:
            # Catch exceptions from fetch_with_retries or other issues
            logger.error(
                f"{Fore.RED}Unexpected error during {action_desc} for {position_side.upper()} {symbol}: {e}",
                exc_info=True,
            )
            # Do NOT update tracker on exception
            termux_notify(
                f"{symbol} Protection ERROR",
                f"{action_desc[:30]} {position_side.upper()} error.",
            )
            return False

    def place_risked_market_order(
        self,
        side: str, # 'buy' or 'sell'
        atr: Decimal,
        total_equity: Decimal,
        current_price: Decimal,
    ) -> bool:
        """Calculates parameters, places market order, verifies position, and sets initial SL/TP."""
        if not self.exchange or not self.market_info:
            logger.error("Cannot place order: Exchange/Market info missing.")
            return False
        if side not in ["buy", "sell"]:
            logger.error(f"Invalid side '{side}' for placing order.")
            return False

        position_side = "long" if side == "buy" else "short"
        logger.info(f"--- Initiating Entry Sequence for {position_side.upper()} ---")

        # --- 1. Calculate Trade Parameters ---
        logger.debug("Calculating trade parameters...")
        trade_params = self._calculate_trade_parameters(
            side, atr, total_equity, current_price
        )
        if not trade_params:
            logger.error("Entry Aborted: Failed to calculate valid trade parameters.")
            return False

        qty_to_order = trade_params["qty"]
        initial_sl_price = trade_params["sl_price"]
        initial_tp_price = trade_params.get("tp_price") # Can be None

        # --- 2. Execute Market Order ---
        logger.debug(f"Executing market {side} order for {qty_to_order.normalize()}...")
        order_info = self._execute_market_order(side, qty_to_order)
        if not order_info:
            logger.error("Entry Aborted: Market order execution failed or rejected.")
            # Attempt cleanup in case of partial fill or unknown state
            self._handle_entry_failure(side, qty_to_order)
            return False

        order_id = order_info.get("id", "[N/A]") # Keep for journaling

        # --- 3. Wait & Verify Position Establishment ---
        # Wait slightly longer to allow position data to update on the exchange
        verify_wait_time = self.config.order_check_delay_seconds + 2
        logger.info(f"Waiting {verify_wait_time}s after market order {order_id} to verify position...")
        time.sleep(verify_wait_time)

        logger.debug("Fetching position state post-entry attempt...")
        positions_after_entry = self.exchange_manager.get_current_position()

        if positions_after_entry is None:
            logger.error(
                f"{Fore.RED}Entry Failed: Position check FAILED after market order {order_id}. Manual check required! Attempting cleanup..."
            )
            self._handle_entry_failure(side, qty_to_order) # Attempt cleanup
            return False

        active_pos_data = positions_after_entry.get(position_side)
        if not active_pos_data:
            logger.error(
                f"{Fore.RED}Entry Failed: Position {position_side.upper()} not found after market order {order_id}. Potential fill delay or order issue. Manual check required! Attempting cleanup..."
            )
            self._handle_entry_failure(side, qty_to_order)
            return False

        # Verify the details of the established position
        filled_qty = safe_decimal(active_pos_data.get("qty", "0"))
        avg_entry_price = safe_decimal(active_pos_data.get("entry_price", "NaN"))

        if filled_qty.copy_abs() < POSITION_QTY_EPSILON:
             logger.error(
                f"{Fore.RED}Entry Failed: Position {position_side.upper()} found but quantity is zero/negligible ({filled_qty}). Manual check required! Assuming entry failed."
            )
             self._handle_entry_failure(side, qty_to_order) # Treat as failure
             return False

        logger.info(
            f"{Fore.GREEN}Position {position_side.upper()} confirmed: Qty={filled_qty.normalize()}, AvgEntry={avg_entry_price.normalize() if not avg_entry_price.is_nan() else '[N/A]'}"
        )

        # Check for significant difference between intended and filled quantity (partial fill handling)
        qty_diff = (qty_to_order - filled_qty).copy_abs()
        # Define a tolerance for partial fill warning (e.g., 1% of intended quantity)
        partial_fill_tolerance = qty_to_order * Decimal("0.01")
        if qty_diff > max(POSITION_QTY_EPSILON * 10, partial_fill_tolerance): # Use a meaningful tolerance
            logger.warning(
                f"{Fore.YELLOW}Partial fill detected? Intended: {qty_to_order.normalize()}, Filled: {filled_qty.normalize()}. Proceeding with filled amount."
            )
            # For simplicity, we still use initial SL/TP calculated based on intended qty risk.
            # More complex logic could recalculate stops based on filled_qty risk, but this changes the original plan.

        # --- 4. Set Initial Position SL/TP ---
        logger.info(f"Setting initial SL/TP for {position_side.upper()} position...")
        set_stops_ok = self._set_position_protection(
            position_side, sl_price=initial_sl_price, tp_price=initial_tp_price
        )

        if not set_stops_ok:
            logger.error(
                f"{Fore.RED}Entry Failed: Failed to set initial SL/TP after establishing position {position_side.upper()}. Attempting to close position for safety!"
            )
            # Use the actual filled quantity for closing
            self.close_position(
                position_side, filled_qty, reason="EmergencyClose:FailedStopSet"
            )
            return False # Entry sequence failed

        # --- 5. Log Entry to Journal ---
        # Log using the actual average entry price and filled quantity
        if self.config.enable_journaling and not avg_entry_price.is_nan():
            self.log_trade_entry_to_journal(
                side, filled_qty, avg_entry_price, order_id
            )

        logger.info(f"{Fore.GREEN}--- Entry Sequence for {position_side.upper()} Completed Successfully ---")
        return True

    def manage_trailing_stop(
        self,
        position_side: str, # 'long' or 'short'
        entry_price: Decimal,
        current_price: Decimal,
        atr: Decimal,
    ) -> None:
        """Checks if TSL activation conditions are met and activates it using V5 position TSL."""
        if not self.exchange or not self.market_info: return # Basic check
        tracker_key = position_side.lower()

        # --- Pre-checks ---
        # Only proceed if fixed SL/TP is currently tracked as active (state 'ACTIVE_SLTP')
        # Or if no protection is set yet (state None) - allow direct TSL activation if conditions met? (Decision: No, require initial SLTP first)
        current_protection_state = self.protection_tracker.get(tracker_key)
        if current_protection_state != "ACTIVE_SLTP":
            if current_protection_state == "ACTIVE_TSL":
                 logger.debug(f"TSL already marked as active for {position_side.upper()}.")
            elif current_protection_state is None:
                 logger.debug(f"No active SL/TP tracked for {position_side.upper()}, cannot activate TSL yet.")
            else: # Should not happen
                 logger.warning(f"Unknown protection state '{current_protection_state}' for {position_side.upper()}. Skipping TSL check.")
            return

        # Validate inputs needed for calculation
        if atr.is_nan() or atr <= 0:
            logger.debug("Invalid ATR, cannot manage TSL.")
            return
        if entry_price.is_nan() or entry_price <= 0:
            logger.debug("Invalid entry price, cannot manage TSL.")
            return
        if current_price.is_nan() or current_price <= 0:
            logger.debug("Invalid current price, cannot manage TSL.")
            return

        try:
            # --- Calculate TSL Activation Price ---
            activation_distance_atr = atr * self.config.tsl_activation_atr_multiplier
            activation_price: Decimal
            if position_side == "long":
                activation_price = entry_price + activation_distance_atr
            else: # position_side == "short"
                activation_price = entry_price - activation_distance_atr

            # Validate calculated activation price
            if activation_price.is_nan() or activation_price <= 0:
                logger.warning(f"Calculated invalid TSL activation price ({activation_price}). Skipping TSL check.")
                return

            # --- Calculate TSL Distance (as price difference) ---
            tsl_distance_price = current_price * (self.config.trailing_stop_percent / 100)
            min_tick_size = self.market_info.get('tick_size', Decimal('1e-8')) # Safe default if missing
            if tsl_distance_price < min_tick_size:
                logger.debug(
                    f"Calculated TSL distance ({tsl_distance_price}) below min tick size {min_tick_size}. Adjusting to minimum."
                )
                tsl_distance_price = min_tick_size

            # --- Check TSL Activation Condition ---
            should_activate_tsl = False
            if position_side == "long" and current_price >= activation_price:
                should_activate_tsl = True
            elif position_side == "short" and current_price <= activation_price:
                should_activate_tsl = True

            # --- Activate TSL if Condition Met ---
            if should_activate_tsl:
                logger.trade( # Use TRADE level for activation attempt
                    f"{Fore.MAGENTA}TSL Activation condition met for {position_side.upper()}!"
                )
                logger.trade(
                    f"  Entry={entry_price.normalize()}, Current={current_price.normalize()}, Activation Target~={activation_price.normalize():.{DEFAULT_PRICE_DP}f}"
                )
                logger.trade(
                    f"Attempting to activate TSL (Dist ~{tsl_distance_price.normalize()}, ActPx {activation_price.normalize():.{DEFAULT_PRICE_DP}f})..."
                )

                # Call _set_position_protection to activate the TSL
                # Pass calculated distance and activation price
                activation_success = self._set_position_protection(
                    position_side,
                    is_tsl=True, # Signal TSL activation mode
                    tsl_distance=tsl_distance_price,
                    tsl_activation_price=activation_price,
                    # sl_price and tp_price are ignored when is_tsl=True
                )

                if activation_success:
                    # Tracker state is updated inside _set_position_protection on success
                    logger.trade(
                        f"{Fore.GREEN}Trailing Stop Loss activated successfully for {position_side.upper()}."
                    )
                else:
                    logger.error(
                        f"{Fore.RED}Failed to activate Trailing Stop Loss for {position_side.upper()} via API."
                    )
                    # Tracker state remains 'ACTIVE_SLTP' if activation failed, allowing retry next cycle

            else:
                # Log why TSL didn't activate (at debug level)
                logger.debug(
                    f"TSL activation condition not met for {position_side.upper()} (Current: {current_price.normalize()}, Target: ~{activation_price.normalize():.{DEFAULT_PRICE_DP}f})"
                )

        except Exception as e:
            logger.error(
                f"Error managing trailing stop for {position_side.upper()}: {e}",
                exc_info=True,
            )

    def close_position(
        self, position_side: str, qty_to_close: Decimal, reason: str = "Signal"
    ) -> bool:
        """Closes the specified position: clears stops first, then places a closing market order."""
        if not self.exchange or not self.market_info:
            logger.error("Cannot close position: Exchange/Market info missing.")
            return False
        if position_side not in ["long", "short"]:
             logger.error(f"Invalid position_side '{position_side}' for closing.")
             return False

        # Validate quantity
        if qty_to_close.is_nan() or qty_to_close.copy_abs() < POSITION_QTY_EPSILON:
            logger.warning(
                f"Close requested for zero/negligible qty ({qty_to_close}). Skipping close for {position_side}."
            )
            # If qty is zero, consider it "closed" successfully in terms of action needed.
            # Ensure tracker is clear for this side.
            self.protection_tracker[position_side] = None
            return True

        symbol = self.config.symbol
        # Determine the side of the closing order (opposite of position side)
        closing_order_side = "sell" if position_side == "long" else "buy"
        tracker_key = position_side.lower() # 'long' or 'short'

        logger.trade( # Use TRADE level for closure attempts
            f"{Fore.YELLOW}Attempting to CLOSE {position_side.upper()} position ({qty_to_close.normalize()} {symbol}) | Reason: {reason}..."
        )

        # --- 1. Clear Existing Position Protection (SL/TP/TSL) ---
        # This is crucial before sending the closing market order, especially for V5 position stops.
        logger.info(
            f"Clearing any existing position protection (SL/TP/TSL) for {position_side.upper()} before closing..."
        )
        # Call _set_position_protection with parameters that result in clearing all stops
        clear_stops_ok = self._set_position_protection(
            position_side,
            sl_price=None, tp_price=None, is_tsl=False # Ensure all stop inputs are effectively None/zero
        )

        if not clear_stops_ok:
            # Log a warning but proceed with the close order attempt anyway.
            # The position might still close, but there's a small risk of the stop order executing simultaneously.
            logger.warning(
                f"{Fore.YELLOW}Failed to confirm clearing of position protection via API for {position_side.upper()}. Proceeding with close order cautiously..."
            )
        else:
            logger.info(f"Position protection cleared successfully via API for {position_side.upper()}.")

        # Ensure local tracker is cleared regardless of API success for safety, as we are attempting to close.
        self.protection_tracker[tracker_key] = None

        # --- 2. Place Closing Market Order ---
        logger.info(
            f"Submitting MARKET {closing_order_side.upper()} order to close {position_side.upper()} position..."
        )
        close_order_info = self._execute_market_order(
            closing_order_side, qty_to_close # Use the quantity passed to the function
        )

        if not close_order_info:
            # If the closing order itself failed submission
            logger.error(
                f"{Fore.RED}Failed to submit closing market order for {position_side.upper()}. MANUAL INTERVENTION REQUIRED!"
            )
            termux_notify(
                f"{symbol} CLOSE FAILED",
                f"Market {closing_order_side.upper()} order failed!",
            )
            # Return False as the close command failed
            return False

        # --- 3. Process Close Order Info & Log ---
        close_order_id = close_order_info.get("id", "[N/A]")
        # Try to get avg price from order response, might be None/NaN initially for market orders
        avg_close_price_str = close_order_info.get("average") # CCXT unified field
        avg_close_price = safe_decimal(avg_close_price_str, default=Decimal("NaN")) # Default to NaN if missing/invalid

        logger.trade(
            f"{Fore.GREEN}Closing market order ({close_order_id}) submitted for {position_side.upper()}. AvgClosePrice: {avg_close_price.normalize() if not avg_close_price.is_nan() else '[Pending/N/A]'}"
        )
        termux_notify(
            f"{symbol} Position Closing",
            f"{position_side.upper()} close order {close_order_id} submitted.",
        )

        # --- 4. Verify Position Closed (Optional but Recommended) ---
        # Wait a bit longer after close order to allow exchange state update
        verify_close_wait = self.config.order_check_delay_seconds + 3
        logger.info(f"Waiting {verify_close_wait}s and verifying position closure...")
        time.sleep(verify_close_wait)

        is_confirmed_closed = False
        try:
            final_positions = self.exchange_manager.get_current_position()
            if final_positions is not None:
                final_pos_data = final_positions.get(position_side)
                # Check if position data is missing or quantity is negligible
                if not final_pos_data or safe_decimal(final_pos_data.get('qty', '0')).copy_abs() < POSITION_QTY_EPSILON:
                    logger.trade( # Use TRADE level for confirmation
                        f"{Fore.GREEN}Position {position_side.upper()} confirmed closed via API check."
                    )
                    is_confirmed_closed = True
                else:
                    # Position still exists after close attempt
                    lingering_qty = safe_decimal(final_pos_data.get('qty', 'NaN'))
                    logger.error(
                        f"{Fore.RED}Position {position_side.upper()} still shows qty {lingering_qty.normalize()} after close attempt (Order {close_order_id}). MANUAL CHECK REQUIRED!"
                    )
                    termux_notify(
                        f"{symbol} CLOSE VERIFY FAILED",
                        f"{position_side.upper()} ({lingering_qty.normalize()}) may still be open!",
                    )
                    # Return False because verification failed, even if order was sent
                    is_confirmed_closed = False
            else:
                # Failed to fetch final positions for verification
                logger.warning(
                    f"{Fore.YELLOW}Could not verify position closure for {position_side.upper()} (failed position fetch after close order {close_order_id}). Assuming closed based on order submission, but MANUAL CHECK ADVISED."
                )
                # Treat as success *for the bot's logic flow* if order submitted but verification failed, but log appropriately.
                # The calling function should be aware verification might have failed.
                is_confirmed_closed = True # Tentative success

        except Exception as verify_exc:
             logger.error(f"Error during position closure verification: {verify_exc}", exc_info=True)
             logger.warning("Assuming closure based on order submission due to verification error. MANUAL CHECK ADVISED.")
             is_confirmed_closed = True # Tentative success

        # --- 5. Log Exit to Journal ---
        # Log the exit attempt even if verification is uncertain, using available data.
        if self.config.enable_journaling:
            # Use the average price from the close order if available, otherwise log as N/A
            self.log_trade_exit_to_journal(
                position_side,
                qty_to_close, # Log the intended close quantity
                avg_close_price, # Might be NaN if not available from order response
                close_order_id,
                reason,
            )

        # Return the status based on order submission and verification
        return is_confirmed_closed

    def _handle_entry_failure(
        self, failed_entry_side: str, attempted_qty: Decimal
    ):
        """Attempts to close any potentially opened position after a failed entry sequence step."""
        logger.warning(
            f"{Fore.YELLOW}Handling potential entry failure for {failed_entry_side.upper()} (intended qty: {attempted_qty.normalize()}). Checking for lingering position..."
        )
        position_side_to_check = "long" if failed_entry_side == "buy" else "short"

        # Wait briefly before checking, allow state to potentially update
        time.sleep(self.config.order_check_delay_seconds + 1)
        logger.debug(f"Checking position status after {failed_entry_side} entry failure...")
        current_positions = self.exchange_manager.get_current_position()

        if current_positions is None:
            # If we can't even check the position, manual intervention is needed.
            logger.error(
                f"{Fore.RED}Could not fetch positions during entry failure handling for {failed_entry_side}. MANUAL CHECK REQUIRED for {self.config.symbol}!"
            )
            termux_notify(f"{self.config.symbol} Check Needed", "Failed pos check after entry fail")
            return # Cannot proceed with automated cleanup

        lingering_pos_data = current_positions.get(position_side_to_check)
        if lingering_pos_data:
            current_qty = safe_decimal(lingering_pos_data.get("qty", "0"))
            if current_qty.copy_abs() >= POSITION_QTY_EPSILON:
                # A position (potentially partial) exists after the failure
                logger.error(
                    f"{Fore.RED}Detected lingering {position_side_to_check.upper()} position (Qty: {current_qty.normalize()}) after failed entry sequence. Attempting emergency close."
                )
                termux_notify(f"{self.config.symbol} Emergency Close", f"Lingering {position_side_to_check} pos")
                # Attempt to close the detected quantity
                close_success = self.close_position(
                    position_side_to_check, current_qty, reason="EmergencyClose:EntryFail"
                )
                if close_success:
                    logger.info(f"Emergency close order submitted/confirmed for lingering {position_side_to_check} position.")
                else:
                    # If emergency close also fails, it's critical
                    logger.critical(
                        f"{Fore.RED+Style.BRIGHT}EMERGENCY CLOSE FAILED for {position_side_to_check.upper()}. MANUAL INTERVENTION URGENT!"
                    )
                    termux_notify(f"{self.config.symbol} URGENT CHECK", f"Emergency close FAILED!")
            else:
                # Position data exists but quantity is negligible
                logger.info(
                    f"Lingering position ({position_side_to_check}) found but quantity negligible ({current_qty}). No emergency close needed."
                )
                # Ensure tracker is clear if somehow it wasn't
                self.protection_tracker[position_side_to_check] = None
        else:
            # No position data found for the side we tried to enter
            logger.info(
                f"No lingering {position_side_to_check} position detected after entry failure."
            )
            # Ensure tracker is clear
            self.protection_tracker[position_side_to_check] = None

    def _write_journal_row(self, data: Dict[str, Any]):
        """Helper function to write a row to the CSV journal."""
        if not self.config.enable_journaling:
            return
        file_path = Path(self.config.journal_file_path)
        file_exists = file_path.is_file()
        try:
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with file_path.open("a", newline="", encoding="utf-8") as csvfile:
                # Define fieldnames in a consistent order
                fieldnames = [
                    "TimestampUTC", "Symbol", "Action", "Side",
                    "Quantity", "AvgPrice", "OrderID", "Reason", "Notes",
                ]
                writer = csv.DictWriter(
                    csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL
                )
                # Write header only if file is new or empty
                if not file_exists or file_path.stat().st_size == 0:
                    writer.writeheader()

                # Prepare row data, ensuring all fields exist and formatting Decimals
                row_to_write = {}
                for field in fieldnames:
                    value = data.get(field)
                    if isinstance(value, Decimal):
                        # Normalize removes trailing zeros, use appropriate precision if needed
                        row_to_write[field] = value.normalize()
                    elif value is None:
                        row_to_write[field] = 'N/A'
                    else:
                        row_to_write[field] = str(value) # Ensure string conversion

                # Default 'Notes' to empty string if missing
                row_to_write['Notes'] = data.get('Notes', '')

                writer.writerow(row_to_write)
            logger.debug(f"Trade {data.get('Action', '').lower()} logged to {file_path}")

        except IOError as e:
            logger.error(
                f"I/O error writing {data.get('Action', '').lower()} to journal {file_path}: {e}"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error writing {data.get('Action', '').lower()} to journal: {e}",
                exc_info=True,
            )

    def log_trade_entry_to_journal(
        self,
        side: str, # 'buy' or 'sell' (order side)
        qty: Decimal, # Filled quantity
        avg_price: Decimal, # Average fill price
        order_id: Optional[str],
    ):
        """Logs trade entry details to the CSV journal."""
        position_side = "long" if side == "buy" else "short"
        data = {
            "TimestampUTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "Symbol": self.config.symbol,
            "Action": "ENTRY",
            "Side": position_side.upper(), # Log position side (LONG/SHORT)
            "Quantity": qty,
            "AvgPrice": avg_price if not avg_price.is_nan() else None, # Log None if NaN
            "OrderID": order_id,
            "Reason": "Strategy Signal", # Default reason for entry
        }
        self._write_journal_row(data)

    def log_trade_exit_to_journal(
        self,
        position_side: str, # 'long' or 'short' (side of position being closed)
        qty: Decimal, # Quantity closed
        avg_price: Decimal, # Average close price (can be NaN)
        order_id: Optional[str], # Closing order ID
        reason: str, # Reason for closing
    ):
        """Logs trade exit details to the CSV journal."""
        data = {
            "TimestampUTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "Symbol": self.config.symbol,
            "Action": "EXIT",
            "Side": position_side.upper(), # Side of the position exited
            "Quantity": qty,
            "AvgPrice": avg_price if not avg_price.is_nan() else None, # Log None if NaN
            "OrderID": order_id,
            "Reason": reason, # Specific exit reason (e.g., SL, TP, Signal, Manual)
        }
        self._write_journal_row(data)


# --- Status Display Class ---
class StatusDisplay:
    """Handles displaying the bot status using the Rich library."""

    def __init__(self, config: TradingConfig):
        self.config = config
        # Store default precision for fallback
        self._default_price_dp = DEFAULT_PRICE_DP
        self._default_amount_dp = DEFAULT_AMOUNT_DP

    def _format_decimal(
        self,
        value: Optional[Decimal],
        precision: Optional[int] = None, # Use specific precision if provided
        default_precision: int = 2, # Fallback precision
        add_commas: bool = False, # Commas for large numbers? (Usually no for price/qty)
        highlight_negative: bool = False, # Add color for negative PnL?
    ) -> Text:
        """Formats Decimal values for Rich display, handling None, NaN, and styling."""
        if value is None or (isinstance(value, Decimal) and value.is_nan()):
            return Text("N/A", style="dim")

        # Determine precision to use
        dp = precision if precision is not None else default_precision

        try:
            # Normalize removes trailing zeros but keeps exponent for quantize
            # Quantize ensures the desired number of decimal places
            quantizer = Decimal("1e-" + str(dp))
            formatted_value = value.quantize(quantizer, rounding=ROUND_HALF_EVEN)

            # Format string with optional comma and fixed precision
            format_spec = f"{{:{',' if add_commas else ''}.{dp}f}}"
            display_str = format_spec.format(formatted_value)

            # Apply styling
            style = "white"
            if highlight_negative and formatted_value < 0:
                style = "bold red"
            elif highlight_negative and formatted_value > 0:
                 style = "bold green" # Optional: highlight positive PnL too

            return Text(display_str, style=style)

        except (ValueError, TypeError, InvalidOperation) as e:
            logger.error(f"Error formatting decimal {value}: {e}")
            return Text("ERR", style="bold red")

    def print_status_panel(
        self,
        cycle: int,
        timestamp: Optional[datetime],
        price: Optional[Decimal],
        indicators: Optional[Dict],
        positions: Optional[Dict], # The dict with 'long'/'short' keys
        equity: Optional[Decimal],
        signals: Dict, # Signal generator result dict
        protection_tracker: Dict, # Order manager's tracker {'long': state, 'short': state}
        market_info: Optional[Dict],
    ):
        """Prints the status panel to the console using Rich Panel and Text."""
        # Get precision details safely from market_info or use defaults
        price_dp = self._default_price_dp
        amount_dp = self._default_amount_dp
        if market_info and "precision_dp" in market_info:
             price_dp = market_info["precision_dp"].get("price", self._default_price_dp)
             amount_dp = market_info["precision_dp"].get("amount", self._default_amount_dp)

        # --- Build Panel Content ---
        panel_content = Text()
        ts_str = (
            timestamp.strftime("%Y-%m-%d %H:%M:%S %Z") if timestamp else "[dim]Timestamp N/A[/]"
        )
        title_text = f" Cycle {cycle} | {self.config.symbol} ({self.config.interval}) | {ts_str} "

        # --- 1. Price & Equity ---
        price_text = self._format_decimal(price, precision=price_dp, default_precision=self._default_price_dp)
        settle_curr = self.config.symbol.split(":")[-1] if ":" in self.config.symbol else "QUOTE" # Get settle currency
        equity_text = self._format_decimal(equity, precision=2, default_precision=2, add_commas=True)

        panel_content.append("Price: ", style="bold cyan")
        panel_content.append(price_text)
        panel_content.append(" | ", style="dim")
        panel_content.append("Equity: ", style="bold cyan")
        panel_content.append(equity_text)
        panel_content.append(f" {settle_curr}\n", style="white")
        panel_content.append("---\n", style="dim")

        # --- 2. Indicators ---
        panel_content.append("Indicators: ", style="bold cyan")
        if indicators:
            ind_parts = []
            # Helper to format indicator values with specific precision
            def fmt_ind(key: str, prec: int = 1, default_prec: int = 1, commas: bool = False) -> Text:
                 return self._format_decimal(
                    indicators.get(key), precision=prec, default_precision=default_prec, add_commas=commas
                 )

            # EMAs
            ema_text = Text()
            ema_text.append("EMA(F/S/T): ")
            ema_text.append(fmt_ind('fast_ema', prec=price_dp, default_prec=self._default_price_dp))
            ema_text.append("/")
            ema_text.append(fmt_ind('slow_ema', prec=price_dp, default_prec=self._default_price_dp))
            ema_text.append("/")
            ema_text.append(fmt_ind('trend_ema', prec=price_dp, default_prec=self._default_price_dp))
            ind_parts.append(ema_text)

            # Stochastic
            stoch_text = Text("Stoch(K/D): ")
            stoch_text.append(fmt_ind('stoch_k', prec=1))
            stoch_text.append("/")
            stoch_text.append(fmt_ind('stoch_d', prec=1))
            if indicators.get('stoch_kd_bullish'): stoch_text.append(" [bold green]▲[/]", style="green") # Bull cross
            elif indicators.get('stoch_kd_bearish'): stoch_text.append(" [bold red]▼[/]", style="red") # Bear cross
            ind_parts.append(stoch_text)

            # ATR
            atr_text = Text(f"ATR({indicators.get('atr_period', '?')}): ")
            atr_text.append(fmt_ind('atr', prec=price_dp, default_prec=self._default_price_dp)) # Use price precision for ATR
            ind_parts.append(atr_text)

            # ADX
            adx_text = Text(f"ADX({self.config.adx_period}): ")
            adx_text.append(fmt_ind('adx', prec=1))
            adx_text.append(f" [+DI:")
            adx_text.append(fmt_ind('pdi', prec=1))
            adx_text.append(f" -DI:")
            adx_text.append(fmt_ind('mdi', prec=1))
            adx_text.append("]")
            ind_parts.append(adx_text)

            # Join indicator parts with separators
            separator = Text(" | ", style="dim")
            for i, part in enumerate(ind_parts):
                panel_content.append(part)
                if i < len(ind_parts) - 1:
                    panel_content.append(separator)
            panel_content.append("\n")
        else:
            panel_content.append("[dim]Calculating...[/]\n", style="dim")
        panel_content.append("---\n", style="dim")

        # --- 3. Position ---
        panel_content.append("Position: ", style="bold cyan")
        pos_text = Text("FLAT", style="bold green") # Default: Flat
        active_position_data = None
        position_side_str = None

        # Check for active long or short position based on fetched data
        if positions:
             long_pos = positions.get("long")
             short_pos = positions.get("short")
             # Check if quantity exists and is significant
             if long_pos and long_pos.get('qty', Decimal(0)).copy_abs() >= POSITION_QTY_EPSILON:
                 active_position_data = long_pos
                 position_side_str = "long"
             elif short_pos and short_pos.get('qty', Decimal(0)).copy_abs() >= POSITION_QTY_EPSILON:
                 active_position_data = short_pos
                 position_side_str = "short"

        # If a position is active, format its details
        if active_position_data and position_side_str:
            pos_style = "bold green" if position_side_str == "long" else "bold red"
            pos_text = Text(f"{position_side_str.upper()}: ", style=pos_style)

            # Qty
            qty_text = self._format_decimal(active_position_data.get("qty"), precision=amount_dp, default_precision=self._default_amount_dp)
            pos_text.append(f"Qty=")
            pos_text.append(qty_text)

            # Entry Price
            entry_text = self._format_decimal(active_position_data.get("entry_price"), precision=price_dp, default_precision=self._default_price_dp)
            pos_text.append(" | Entry=", style="dim")
            pos_text.append(entry_text)

            # Unrealized PnL
            pnl_text = self._format_decimal(active_position_data.get("unrealized_pnl"), precision=4, default_precision=4, highlight_negative=True)
            pos_text.append(" | PnL=", style="dim")
            pos_text.append(pnl_text)

            # Protection Status (from tracker and position data)
            tracked_protection = protection_tracker.get(position_side_str) # 'ACTIVE_SLTP', 'ACTIVE_TSL', None
            sl_from_pos = active_position_data.get("stop_loss_price")
            tp_from_pos = active_position_data.get("take_profit_price")
            tsl_active_from_pos = active_position_data.get("is_tsl_active", False)
            tsl_trigger_from_pos = active_position_data.get("tsl_trigger_price")

            pos_text.append(" | Prot: ", style="dim")
            if tracked_protection == "ACTIVE_TSL" or tsl_active_from_pos:
                 prot_status_text = Text("TSL", style="magenta")
                 tsl_trigger_text = self._format_decimal(tsl_trigger_from_pos, precision=price_dp, default_precision=self._default_price_dp)
                 prot_details = Text(" (Trig:", style="dim").append(tsl_trigger_text).append(")", style="dim")
            elif tracked_protection == "ACTIVE_SLTP" or sl_from_pos or tp_from_pos:
                 prot_status_text = Text("SL/TP", style="yellow")
                 sl_text = self._format_decimal(sl_from_pos, precision=price_dp, default_precision=self._default_price_dp)
                 tp_text = self._format_decimal(tp_from_pos, precision=price_dp, default_precision=self._default_price_dp)
                 prot_details = Text(" (S:", style="dim").append(sl_text).append(" T:", style="dim").append(tp_text).append(")", style="dim")
            else:
                 prot_status_text = Text("None", style="dim")
                 prot_details = Text("")

            pos_text.append(prot_status_text)
            pos_text.append(prot_details)

        panel_content.append(pos_text)
        panel_content.append("\n")
        panel_content.append("---\n", style="dim")

        # --- 4. Signals ---
        panel_content.append("Signal: ", style="bold cyan")
        sig_reason = signals.get("reason", "[dim]No signal info[/]")
        sig_style = "dim" # Default style for neutral/no signal
        if signals.get("long"):
            sig_style = "bold green"
        elif signals.get("short"):
            sig_style = "bold red"
        elif "Blocked" in sig_reason:
            sig_style = "yellow"
        elif "No Signal:" not in sig_reason and "Initializing" not in sig_reason: # Explicit non-blocking message
            sig_style = "white"

        # Wrap long signal reasons for better readability
        wrapped_reason = "\n        ".join(textwrap.wrap(sig_reason, width=100)) # Adjust width as needed
        panel_content.append(Text(wrapped_reason, style=sig_style))

        # --- Print Panel ---
        console.print(
            Panel(
                panel_content,
                title=f"[bold bright_magenta]{title_text}[/]",
                border_style="bright_blue",
                expand=False, # Prevent panel from taking full width
                padding=(1, 2) # Add some padding inside the panel
            )
        )


# --- Trading Bot Class ---
class TradingBot:
    """Main orchestrator for the trading bot."""

    def __init__(self):
        logger.info(
            f"{Fore.MAGENTA+Style.BRIGHT}--- Initializing Pyrmethus v2.4.1 (Enhanced+) ---"
        )
        self.config = TradingConfig()
        self.exchange_manager = ExchangeManager(self.config)
        # Critical check: Ensure exchange and market_info loaded successfully
        if not self.exchange_manager.exchange or not self.exchange_manager.market_info:
            # Error logged in ExchangeManager, just exit
            logger.critical("TradingBot initialization failed due to ExchangeManager issues. Halting.")
            sys.exit(1)

        self.indicator_calculator = IndicatorCalculator(self.config)
        self.signal_generator = SignalGenerator(self.config)
        # Pass initialized exchange_manager to OrderManager
        self.order_manager = OrderManager(self.config, self.exchange_manager)
        self.status_display = StatusDisplay(self.config)
        self.shutdown_requested = False
        self._setup_signal_handlers()
        logger.info(f"{Fore.GREEN}Pyrmethus components initialized successfully.")

    def _setup_signal_handlers(self):
        """Sets up OS signal handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGINT, self._signal_handler) # Handle Ctrl+C
            signal.signal(signal.SIGTERM, self._signal_handler) # Handle kill/system shutdown
            logger.debug("Signal handlers for SIGINT and SIGTERM set up.")
        except ValueError as e:
             # Can happen if run in non-main thread on some OSes
             logger.warning(f"Could not set signal handlers (maybe not in main thread?): {e}")
        except Exception as e:
             logger.error(f"Unexpected error setting up signal handlers: {e}", exc_info=True)


    def _signal_handler(self, sig: int, frame: Optional[Any]):
        """Internal signal handler to initiate graceful shutdown."""
        # Check if shutdown is already requested to avoid redundant actions
        if not self.shutdown_requested:
            sig_name = signal.Signals(sig).name if isinstance(sig, int) else str(sig)
            # Use console.print for visibility even if logging is redirected
            console.print(f"\n[bold yellow]Signal {sig_name} received. Initiating graceful shutdown...[/]")
            logger.warning(f"Signal {sig_name} received. Initiating graceful shutdown...")
            self.shutdown_requested = True
        else:
            # Avoid multiple shutdown attempts if signal received again quickly
            logger.warning("Shutdown already in progress. Ignoring additional signal.")

    def run(self):
        """Starts the main trading loop."""
        self._display_startup_info()
        termux_notify(
            "Pyrmethus Started",
            f"{self.config.symbol} @ {self.config.interval}",
        )
        cycle_count = 0
        while not self.shutdown_requested:
            cycle_count += 1
            cycle_start_time = time.monotonic()
            logger.debug(f"--- Starting Cycle {cycle_count} ---")

            try:
                # Execute one full trading cycle
                self.trading_spell_cycle(cycle_count)

            except KeyboardInterrupt:
                # This handles Ctrl+C if it occurs *during* the cycle execution
                # The signal handler should ideally catch it first, but this is a backup.
                logger.warning("\nCtrl+C detected during main loop execution. Initiating shutdown.")
                self.shutdown_requested = True
                break # Exit loop immediately

            except ccxt.AuthenticationError as e:
                # Handle critical auth errors immediately
                logger.critical(
                    f"{Fore.RED+Style.BRIGHT}CRITICAL AUTH ERROR in cycle {cycle_count}: {e}. Halting immediately.", exc_info=False
                )
                termux_notify("Pyrmethus CRITICAL ERROR", f"Auth failed: {e}")
                self.shutdown_requested = True # Ensure shutdown sequence runs
                break # Exit loop

            except SystemExit as e:
                 # Catch explicit exits from lower levels (e.g., config validation)
                 logger.warning(f"SystemExit called with code {e.code}. Terminating.")
                 self.shutdown_requested = True
                 break

            except Exception as e:
                # Catch-all for unexpected errors within a cycle
                logger.error(
                    f"{Fore.RED+Style.BRIGHT}Unhandled exception in main trading cycle {cycle_count}: {e}",
                    exc_info=True, # Log full traceback for unexpected errors
                )
                logger.error(
                    f"{Fore.RED}Continuing loop, but caution advised. Check logs for details."
                )
                termux_notify(
                    "Pyrmethus Cycle Error",
                    f"Unhandled exception cycle {cycle_count}. Check logs.",
                )
                # Optional: Implement a counter for consecutive errors to halt if needed
                # Sleep longer after an unexpected error to avoid spamming
                sleep_duration = self.config.loop_sleep_seconds * 2
            else:
                # Calculate remaining sleep time based on cycle duration if no exception
                cycle_duration = time.monotonic() - cycle_start_time
                sleep_duration = max(0, self.config.loop_sleep_seconds - cycle_duration)
                logger.debug(f"Cycle {cycle_count} completed in {cycle_duration:.2f}s.")


            # --- Interruptible Sleep ---
            if not self.shutdown_requested and sleep_duration > 0:
                logger.debug(f"Sleeping for {sleep_duration:.2f} seconds...")
                sleep_end_time = time.monotonic() + sleep_duration
                try:
                    while time.monotonic() < sleep_end_time and not self.shutdown_requested:
                        # Sleep in short intervals to check shutdown flag frequently
                        time.sleep(min(0.5, sleep_duration)) # Check every 0.5s or less
                except KeyboardInterrupt:
                     # Catch Ctrl+C during sleep
                    logger.warning("\nCtrl+C detected during sleep.")
                    self.shutdown_requested = True
                    # No need to break here, the outer loop condition will handle it

            # Check shutdown flag again after sleep/cycle completion
            if self.shutdown_requested:
                logger.info("Shutdown requested, exiting main loop.")
                break

        # --- Graceful Shutdown ---
        # This block executes after the loop breaks (normally via signal or critical error)
        self.graceful_shutdown()
        console.print(
            f"\n[bold bright_cyan]Pyrmethus has returned to the ether.[/]"
        )
        # Explicitly exit with code 0 for normal shutdown
        sys.exit(0)

    def trading_spell_cycle(self, cycle_count: int) -> None:
        """Executes one cycle of the trading logic: fetch, analyze, act, display."""
        start_time = time.monotonic()
        cycle_status = "OK" # Track cycle status for logging/display

        # --- 1. Fetch Data & Basic Info ---
        logger.debug("Fetching market data (OHLCV)...")
        df = self.exchange_manager.fetch_ohlcv()
        if df is None or df.empty:
            logger.error(f"{Fore.RED}Cycle Aborted (Cycle {cycle_count}): Market data fetch failed.")
            cycle_status = "FAIL:FETCH_OHLCV"
            # Display minimal status if possible (no price/indicators)
            self.status_display.print_status_panel(cycle_count, None, None, None, None, None, {"reason": cycle_status}, {}, self.exchange_manager.market_info)
            return # Skip rest of cycle

        # Extract latest price and timestamp safely
        try:
            last_candle = df.iloc[-1]
            current_price = safe_decimal(last_candle["close"])
            last_timestamp = df.index[-1].to_pydatetime() # Convert pd.Timestamp to standard datetime
            if current_price.is_nan() or current_price <= 0:
                raise ValueError(f"Invalid latest close price: {current_price}")
            logger.debug(
                f"Latest Candle: {last_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}, Close={current_price.normalize()}"
            )
        except (IndexError, KeyError, ValueError, TypeError) as e:
            logger.error(
                f"{Fore.RED}Cycle Aborted (Cycle {cycle_count}): Error processing latest candle data: {e}",
                exc_info=False, # Less verbose for this common error
            )
            cycle_status = "FAIL:PROCESS_CANDLE"
            self.status_display.print_status_panel(cycle_count, None, None, None, None, None, {"reason": cycle_status}, {}, self.exchange_manager.market_info)
            return

        # --- 2. Calculate Indicators ---
        logger.debug("Calculating indicators...")
        indicators = self.indicator_calculator.calculate_indicators(df)
        if indicators is None:
            logger.warning(
                f"{Fore.YELLOW}Indicator calculation failed or returned None (Cycle {cycle_count}). Trading logic may be impacted."
            )
            cycle_status = "WARN:INDICATORS_FAILED"
            # Continue cycle if possible, but signal generation might fail or be unreliable
            # ATR might be missing, handle this below
        current_atr = indicators.get("atr", Decimal("NaN")) if indicators else Decimal("NaN")

        # --- 3. Get Current State (Balance & Positions) ---
        logger.debug("Fetching balance and position state...")
        total_equity, _ = self.exchange_manager.get_balance() # Available balance not directly used in logic currently
        current_positions = self.exchange_manager.get_current_position() # Returns {'long': {}, 'short': {}} or None on failure

        # --- Validate State for Trading Logic ---
        can_run_trade_logic = True
        if total_equity is None or total_equity.is_nan() or total_equity <= 0:
            logger.error(
                f"{Fore.RED}Failed fetching valid equity ({total_equity}). Trading logic skipped for Cycle {cycle_count}."
            )
            cycle_status = "FAIL:FETCH_EQUITY"
            can_run_trade_logic = False
        if current_positions is None:
            # Error already logged by get_current_position if fetch failed
            logger.error(
                f"{Fore.RED}Failed fetching positions. Trading logic skipped for Cycle {cycle_count}."
            )
            cycle_status = "FAIL:FETCH_POSITIONS"
            can_run_trade_logic = False
        if current_atr.is_nan() or current_atr <= 0:
            # ATR is essential for risk sizing and TSL
            logger.error(f"{Fore.RED}Invalid ATR ({current_atr}). Trading logic skipped for Cycle {cycle_count}.")
            cycle_status = "FAIL:INVALID_ATR"
            can_run_trade_logic = False

        # --- Prepare Data Snapshots for Logic & Display ---
        # Use snapshots to ensure consistency within the cycle's logic and display
        protection_tracker_snapshot = copy.deepcopy(self.order_manager.protection_tracker)
        # Use a safe default (empty dict) if positions fetch failed, logic will skip anyway
        live_positions_state = current_positions if current_positions is not None else {"long": {}, "short": {}}
        # This variable will hold the position state used for the *final* display panel, updated after actions
        final_positions_for_panel = copy.deepcopy(live_positions_state)
        # Default signal state if logic is skipped
        signals: Dict[str, Union[bool, str]] = {
            "long": False, "short": False, "reason": f"Skipped ({cycle_status})"
        }

        # --- 4. Execute Core Trading Logic (if state is valid) ---
        if can_run_trade_logic:
            logger.debug("Executing core trading logic...")
            # --- 4a. Extract Current Position Details ---
            active_long_pos = live_positions_state.get("long", {})
            active_short_pos = live_positions_state.get("short", {})
            long_qty = safe_decimal(active_long_pos.get("qty", "0"))
            short_qty = safe_decimal(active_short_pos.get("qty", "0"))
            long_entry = safe_decimal(active_long_pos.get("entry_price", "NaN"))
            short_entry = safe_decimal(active_short_pos.get("entry_price", "NaN"))

            has_long_pos = long_qty.copy_abs() >= POSITION_QTY_EPSILON
            has_short_pos = short_qty.copy_abs() >= POSITION_QTY_EPSILON
            is_flat = not has_long_pos and not has_short_pos
            current_pos_side = "long" if has_long_pos else "short" if has_short_pos else None

            logger.debug(f"Position State Before Actions: Flat={is_flat}, Side={current_pos_side}, LongQty={long_qty.normalize()}, ShortQty={short_qty.normalize()}")

            # --- 4b. Manage Trailing Stops (if in position) ---
            if current_pos_side and indicators: # Ensure indicators are available
                entry_price = long_entry if current_pos_side == "long" else short_entry
                if not entry_price.is_nan() and entry_price > 0:
                     logger.debug(f"Managing TSL for {current_pos_side} position...")
                     self.order_manager.manage_trailing_stop(
                         current_pos_side, entry_price, current_price, current_atr
                     )
                     # Update snapshot *after* TSL management attempt (tracker might change)
                     protection_tracker_snapshot = copy.deepcopy(self.order_manager.protection_tracker)
                else:
                     logger.warning(f"Cannot manage TSL for {current_pos_side}: Invalid entry price ({entry_price})")

            # --- 4c. Re-fetch position state AFTER TSL check/management ---
            # Important: TSL activation or a stop being hit could change the position state async.
            logger.debug("Re-fetching position state after TSL management...")
            positions_after_tsl = self.exchange_manager.get_current_position()
            if positions_after_tsl is None:
                logger.error(
                    f"{Fore.RED}Failed re-fetching positions after TSL check (Cycle {cycle_count}). Using previous state, but it might be stale. Caution advised."
                )
                cycle_status = "WARN:POS_REFETCH_TSL_FAIL"
                # Keep using the potentially stale 'live_positions_state' for subsequent logic this cycle
            else:
                # Update live state variables based on the re-fetched data
                live_positions_state = positions_after_tsl
                final_positions_for_panel = copy.deepcopy(live_positions_state) # Update panel data source
                active_long_pos = live_positions_state.get("long", {})
                active_short_pos = live_positions_state.get("short", {})
                long_qty = safe_decimal(active_long_pos.get("qty", "0"))
                short_qty = safe_decimal(active_short_pos.get("qty", "0"))
                has_long_pos = long_qty.copy_abs() >= POSITION_QTY_EPSILON
                has_short_pos = short_qty.copy_abs() >= POSITION_QTY_EPSILON
                is_flat = not has_long_pos and not has_short_pos
                current_pos_side = "long" if has_long_pos else "short" if has_short_pos else None
                logger.debug(f"Position State After TSL Check: Flat={is_flat}, Side={current_pos_side}, LongQty={long_qty.normalize()}, ShortQty={short_qty.normalize()}")

                # If position became flat unexpectedly (e.g., SL/TSL hit between checks), ensure tracker is cleared.
                if is_flat and any(self.order_manager.protection_tracker.values()):
                    logger.warning(f"{Fore.YELLOW}Position became flat after TSL logic/check, clearing protection tracker state.")
                    self.order_manager.protection_tracker = {"long": None, "short": None}
                    protection_tracker_snapshot = copy.deepcopy(self.order_manager.protection_tracker) # Update snapshot


            # --- 4d. Generate Trading Signals (Entry) ---
            can_gen_signals = indicators is not None and not current_price.is_nan() and len(df) >= 2
            if can_gen_signals:
                logger.debug("Generating entry signals...")
                signals = self.signal_generator.generate_signals(df.iloc[-2:], indicators)
            else:
                reason = "Skipped Signal Gen: " + ("Indicators missing" if indicators is None else f"Need >=2 candles ({len(df)} found)")
                signals = {"long": False, "short": False, "reason": reason}
                logger.warning(reason)


            # --- 4e. Check for Signal-Based Exits (if in position) ---
            exit_triggered_by_signal = False
            exit_side = None
            qty_to_close_on_exit = Decimal(0)

            if can_gen_signals and current_pos_side: # Only check exit if signals generated and in position
                logger.debug(f"Checking exit signals for {current_pos_side} position...")
                exit_reason = self.signal_generator.check_exit_signals(current_pos_side, indicators)
                if exit_reason:
                    exit_side = current_pos_side
                    qty_to_close_on_exit = long_qty if exit_side == "long" else short_qty
                    logger.trade( # Use TRADE level for exit initiation
                        f"{Fore.YELLOW}Attempting signal-based exit for {exit_side.upper()} (Qty: {qty_to_close_on_exit.normalize()}) | Reason: {exit_reason}"
                    )
                    # Attempt to close the position based on the signal
                    close_success = self.order_manager.close_position(
                        exit_side, qty_to_close_on_exit, reason=exit_reason
                    )
                    exit_triggered_by_signal = close_success # Track if close command was successful/verified

                    if not exit_triggered_by_signal:
                        cycle_status = "FAIL:EXIT_ORDER_FAILED" # Mark failure if close command failed
                        logger.error(f"{Fore.RED}Failed to execute closing order for {exit_side} based on exit signal. Position may still be open.")
                        # Bot will retry closing on next cycle if position persists
                    else:
                         logger.info(f"Signal-based exit initiated/confirmed for {exit_side}.")
                         # State will be re-fetched next


            # --- 4f. Re-fetch state AGAIN if an exit was triggered/attempted ---
            # This ensures entry logic uses the most up-to-date state after a potential close.
            if exit_triggered_by_signal and exit_side:
                logger.debug(f"Re-fetching state after signal exit attempt ({exit_side})...")
                positions_after_exit = self.exchange_manager.get_current_position()
                if positions_after_exit is None:
                    logger.error(f"{Fore.RED}Failed re-fetching positions after signal exit (Cycle {cycle_count}). State may be inaccurate for entry decision.")
                    cycle_status = "WARN:POS_REFETCH_EXIT_FAIL"
                    # Keep using previous state for panel, though potentially inaccurate for logic
                else:
                    # Update live state variables for subsequent entry logic
                    live_positions_state = positions_after_exit
                    final_positions_for_panel = copy.deepcopy(live_positions_state) # Update panel data source again
                    active_long_pos = live_positions_state.get("long", {})
                    active_short_pos = live_positions_state.get("short", {})
                    long_qty = safe_decimal(active_long_pos.get("qty", "0"))
                    short_qty = safe_decimal(active_short_pos.get("qty", "0"))
                    has_long_pos = long_qty.copy_abs() >= POSITION_QTY_EPSILON
                    has_short_pos = short_qty.copy_abs() >= POSITION_QTY_EPSILON
                    is_flat = not has_long_pos and not has_short_pos
                    current_pos_side = "long" if has_long_pos else "short" if has_short_pos else None
                    logger.debug(f"Position State After Signal Exit Attempt: Flat={is_flat}, Side={current_pos_side}")
                    # Ensure tracker is clear if now flat (it should be cleared by close_position, but double-check)
                    if is_flat:
                        logger.debug("Position became flat after signal exit, ensuring protection tracker is clear.")
                        self.order_manager.protection_tracker = {"long": None, "short": None}
                        protection_tracker_snapshot = copy.deepcopy(self.order_manager.protection_tracker) # Update snapshot


            # --- 4g. Execute Entry Trades (Only if currently flat and valid entry signal exists) ---
            # Use the state determined *after* potential TSL management and signal exits
            if is_flat and can_gen_signals and (signals.get("long") or signals.get("short")):
                entry_side = "buy" if signals.get("long") else "sell"
                signal_reason = signals.get('reason', '')
                log_color = Fore.GREEN if entry_side == 'buy' else Fore.RED
                logger.trade( # Use TRADE level for entry attempts
                    f"{log_color+Style.BRIGHT}Entry Signal Detected: {entry_side.upper()}! {signal_reason}. Attempting entry sequence..."
                )
                # Attempt the full entry sequence (calculates params, places order, verifies, sets stops)
                entry_successful = self.order_manager.place_risked_market_order(
                    entry_side, current_atr, total_equity, current_price
                )

                if entry_successful:
                    logger.info(f"{Fore.GREEN}Entry sequence completed successfully for {entry_side}.")
                    # Re-fetch state one last time for the most accurate panel display after entry
                    logger.debug("Re-fetching state after successful entry...")
                    positions_after_entry = self.exchange_manager.get_current_position()
                    if positions_after_entry is not None:
                        final_positions_for_panel = copy.deepcopy(positions_after_entry)
                        protection_tracker_snapshot = copy.deepcopy(self.order_manager.protection_tracker)
                    else:
                        logger.warning("Failed re-fetching positions after entry. Panel display might be slightly stale.")
                        cycle_status = "WARN:POS_REFETCH_ENTRY_FAIL"
                else:
                    # Entry failed (error logged within place_risked_market_order or _handle_entry_failure)
                    logger.error(f"{Fore.RED}Entry sequence failed for {entry_side}.")
                    cycle_status = "FAIL:ENTRY_SEQUENCE"
                    # State should ideally be flat again after failed entry handling,
                    # but re-fetch just in case for panel accuracy.
                    logger.debug("Re-fetching state after failed entry attempt...")
                    positions_after_failed_entry = self.exchange_manager.get_current_position()
                    if positions_after_failed_entry is not None:
                        final_positions_for_panel = copy.deepcopy(positions_after_failed_entry)
                        # Ensure tracker is clear if entry failed and we are flat
                        if not (final_positions_for_panel.get("long") or final_positions_for_panel.get("short")):
                             self.order_manager.protection_tracker = {"long": None, "short": None}
                        protection_tracker_snapshot = copy.deepcopy(self.order_manager.protection_tracker)
                    else:
                         logger.warning("Failed re-fetching positions after failed entry.")
                         cycle_status += "|POS_REFETCH_FAIL"


            elif is_flat:
                logger.debug("Position flat, no entry signal generated or conditions not met.")
            elif current_pos_side:
                logger.debug(f"Position ({current_pos_side.upper()}) remains open, skipping entry logic.")
            # End of core trading logic block
        else:
             logger.warning(f"Core trading logic skipped for Cycle {cycle_count} due to earlier failure/invalid state ({cycle_status}).")


        # --- 5. Display Status Panel ---
        # Use the final state variables gathered throughout the cycle for display
        logger.debug("Displaying status panel...")
        self.status_display.print_status_panel(
            cycle_count,
            last_timestamp, # Timestamp of the last candle used
            current_price, # Price used for decisions this cycle
            indicators, # Indicators calculated this cycle
            final_positions_for_panel, # Use the most up-to-date position info after actions
            total_equity, # Equity fetched this cycle
            signals, # Signals generated this cycle
            protection_tracker_snapshot, # Use the snapshot reflecting state after actions
            self.exchange_manager.market_info, # Market info for formatting
        )

        end_time = time.monotonic()
        logger.info(
            f"{Fore.MAGENTA}--- Cycle {cycle_count} Status: {cycle_status} (Duration: {end_time - start_time:.2f}s) ---"
        )

    def graceful_shutdown(self) -> None:
        """Handles cleaning up: cancelling orders and closing positions before exiting."""
        # Use console print as well for visibility during shutdown
        console.print(f"\n[bold yellow]Initiating Graceful Shutdown Sequence...[/]")
        logger.warning(f"{Fore.YELLOW+Style.BRIGHT}Initiating Graceful Shutdown Sequence...")
        termux_notify("Pyrmethus Shutdown", f"Closing {self.config.symbol}...")

        # Check if exchange manager is available for cleanup
        if not self.exchange_manager or not self.exchange_manager.exchange or not self.exchange_manager.market_info:
            logger.error(
                f"{Fore.RED}Cannot perform graceful shutdown: Exchange Manager, Exchange instance, or Market Info missing."
            )
            termux_notify("Shutdown Warning!", f"{self.config.symbol}: Cannot shutdown cleanly!")
            return

        exchange = self.exchange_manager.exchange
        symbol = self.config.symbol
        market_id = self.exchange_manager.market_info.get("id")

        # --- 1. Cancel All Open Non-Positional Orders ---
        # Note: V5 Position SL/TP are attached to the position and managed via setTradingStop (handled by close_position).
        # This step aims to cancel any pending limit/stop entry/exit orders if they were used.
        # Currently, this bot uses market orders, so this might not cancel much, but good practice.
        logger.info(
            f"{Fore.CYAN}Attempting to cancel all active non-positional orders for {symbol} (Category: {self.config.bybit_v5_category})..."
        )
        try:
            # Use cancel_all_orders with V5 category parameter
            params = {"category": self.config.bybit_v5_category}
            # Set a short timeout/retry for cancellation during shutdown
            cancel_resp = fetch_with_retries(
                exchange.cancel_all_orders,
                symbol=symbol, # Specify symbol if possible
                params=params,
                max_retries=1, # Low retries during shutdown
                delay_seconds=1,
            )
            logger.info(f"Cancel all active orders response: {str(cancel_resp)[:200]}...") # Log snippet
            # Check V5 response structure if possible (often contains a list)
            if isinstance(cancel_resp, dict) and cancel_resp.get('retCode') == V5_SUCCESS_RETCODE:
                 cancelled_list = cancel_resp.get('result', {}).get('list', [])
                 logger.info(f"Cancel all orders command successful. Found {len(cancelled_list)} items in response list.")
            elif isinstance(cancel_resp, list): # Some CCXT versions might return list directly
                 logger.info(f"Cancelled {len(cancel_resp)} active orders (based on direct list response).")
            else:
                 logger.warning("Cancel all orders response format unexpected or indicated failure. Check logs.")

        except ccxt.NotSupported:
             logger.warning(f"Exchange {exchange.id} does not support cancel_all_orders with current params/category. Skipping cancellation.")
        except Exception as e:
            logger.error(
                f"{Fore.RED}Error cancelling active orders during shutdown: {e}", exc_info=False # Less verbose during shutdown
            )

        # Clear local protection tracker state regardless of cancellation success
        logger.info("Clearing local protection tracker state...")
        self.order_manager.protection_tracker = {"long": None, "short": None}

        logger.info("Waiting briefly after order cancellation attempt...")
        time.sleep(max(self.config.order_check_delay_seconds, 2)) # Wait 2-3 seconds

        # --- 2. Check and Close Any Lingering Positions ---
        logger.info(f"{Fore.CYAN}Checking for lingering positions for {symbol} to close...")
        closed_count = 0
        positions_to_close: List[Tuple[str, Decimal]] = [] # List of (side, qty)

        try:
            # Fetch final position state
            final_positions = self.exchange_manager.get_current_position()
            if final_positions is not None:
                # Check both long and short sides
                for side in ["long", "short"]:
                    pos_data = final_positions.get(side)
                    if pos_data:
                        qty = safe_decimal(pos_data.get('qty', '0'))
                        if qty.copy_abs() >= POSITION_QTY_EPSILON:
                            logger.warning(f"{Fore.YELLOW}Found lingering {side.upper()} position (Qty: {qty.normalize()}) requiring closure.")
                            positions_to_close.append((side, qty))

                if not positions_to_close:
                    logger.info(f"{Fore.GREEN}No significant positions found requiring closure.")
                else:
                    logger.warning(f"Attempting to close {len(positions_to_close)} detected position(s)...")
                    for side, qty in positions_to_close:
                        logger.info(f"Closing {side.upper()} position (Qty: {qty.normalize()})...")
                        # Use the OrderManager's close_position method
                        # This handles clearing V5 stops first, then market closing.
                        close_success = self.order_manager.close_position(
                            side, qty, reason="GracefulShutdown"
                        )
                        if close_success:
                            closed_count += 1
                            logger.info(f"{Fore.GREEN}Closure initiated/confirmed for {side.upper()}.")
                        else:
                            logger.error(f"{Fore.RED}Closure failed for {side.upper()}. MANUAL INTERVENTION REQUIRED.")

                    # Final summary of closure attempts
                    if closed_count == len(positions_to_close):
                        logger.info(f"{Fore.GREEN}All detected positions ({closed_count}) closed successfully or closure initiated.")
                    else:
                        logger.warning(
                            f"{Fore.YELLOW}Attempted {len(positions_to_close)} closures, {closed_count} succeeded/initiated. MANUAL VERIFICATION REQUIRED for failed closures."
                        )
            else:
                # If fetching positions failed during shutdown
                logger.error(
                    f"{Fore.RED}Failed fetching positions during shutdown check. MANUAL CHECK REQUIRED for symbol {symbol}!"
                )
                termux_notify(f"{symbol} Shutdown Issue", "Failed pos check! Manual verify needed.")

        except Exception as e:
            logger.error(
                f"{Fore.RED+Style.BRIGHT}Error during position closure phase of shutdown: {e}. MANUAL CHECK REQUIRED.",
                exc_info=True,
            )
            termux_notify(f"{symbol} Shutdown Issue", f"Error closing pos: {e}")

        console.print(f"[bold yellow]Graceful Shutdown Sequence Complete.[/]")
        logger.warning(f"{Fore.YELLOW+Style.BRIGHT}Graceful Shutdown Sequence Complete. Pyrmethus rests.")
        termux_notify("Shutdown Complete", f"{self.config.symbol} shutdown finished.")

    def _display_startup_info(self):
        """Prints initial configuration details using Rich."""
        console.print(
            f"[bold bright_cyan] Summoning Pyrmethus [magenta]v2.4.1 (Enhanced+)[/]..."
        )
        # Use a Rich Table for better alignment
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="yellow")
        table.add_column(style="white")

        table.add_row("Trading Symbol:", f"{self.config.symbol}")
        table.add_row("Interval:", f"{self.config.interval}")
        table.add_row("Market Type:", f"{self.config.market_type}")
        table.add_row("V5 Category:", f"{self.config.bybit_v5_category}")
        table.add_row("---", "---")
        table.add_row("Risk %:", f"{self.config.risk_percentage:.3%}")
        table.add_row("SL ATR Mult:", f"{self.config.sl_atr_multiplier.normalize()}x")
        table.add_row("TP ATR Mult:", f"{self.config.tp_atr_multiplier.normalize()}x" if self.config.tp_atr_multiplier > 0 else "[dim]Disabled[/]")
        table.add_row("TSL Act Mult:", f"{self.config.tsl_activation_atr_multiplier.normalize()}x")
        table.add_row("TSL Trail %:", f"{self.config.trailing_stop_percent.normalize()}%")
        table.add_row("---", "---")
        table.add_row("Trend Filter:", f"{'ON' if self.config.trade_only_with_trend else 'OFF'}")
        table.add_row("ATR Move Filter:", f"{self.config.atr_move_filter_multiplier.normalize()}x" if self.config.atr_move_filter_multiplier > 0 else "[dim]Disabled[/]")
        table.add_row("ADX Filter Lvl:", f">{self.config.min_adx_level.normalize()}")
        table.add_row("---", "---")
        journal_status = (
            f"Enabled ([dim]{self.config.journal_file_path}[/])"
            if self.config.enable_journaling
            else "Disabled"
        )
        table.add_row("Journaling:", journal_status)
        v5_stops_info = (f"SLTrig:[cyan]{self.config.sl_trigger_by}[/], "
                         f"TSLTrig:[cyan]{self.config.tsl_trigger_by}[/], "
                         f"PosIdx:[cyan]{self.config.position_idx}[/]")
        table.add_row("[dim]V5 Pos Stops:[/]", f"[dim]{v5_stops_info}[/]")

        console.print(table)
        console.print("-" * 60, style="dim")


# --- Main Execution Block ---
if __name__ == "__main__":
    # Configuration is loaded within TradingConfig using dotenv

    try:
        # Initialize and run the bot
        bot = TradingBot()
        bot.run() # Contains the main loop and graceful shutdown call

    except SystemExit as e:
        # Catch sys.exit() calls for clean termination (e.g., from config validation or normal shutdown)
        # Log level depends on exit code
        if e.code == 0:
            logger.info(f"Pyrmethus process terminated normally (Exit Code: {e.code}).")
        else:
            logger.warning(f"Pyrmethus process terminated with Exit Code: {e.code}.")
        # sys.exit(e.code) # Propagate the exit code

    except Exception as main_exception:
        # Catch critical errors during bot initialization or final shutdown phases
        logger.critical(
            f"{Fore.RED+Style.BRIGHT}CRITICAL UNHANDLED ERROR during bot execution: {main_exception}",
            exc_info=True, # Log traceback for critical failures
        )
        termux_notify(
            "Pyrmethus CRITICAL ERROR",
            f"Bot failed: {str(main_exception)[:100]}",
        )
        sys.exit(1) # Ensure non-zero exit code on critical failure

