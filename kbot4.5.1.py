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
# Pyrmethus v2.4.1 - Enhanced Code Structure & Robustness
# fmt: on
"""
Pyrmethus - Termux Trading Spell (v2.4.1 - Enhanced)

Conjures market insights and executes trades on Bybit Futures using the
V5 Unified Account API via CCXT. Refactored into classes for better structure
and utilizing V5 position-based stop-loss/take-profit/trailing-stop features.

Enhancements in this version (beyond v2.4.1):
- Improved code structure, readability, and maintainability.
- Refactored configuration loading (`_get_env`) for clarity and reduced complexity.
- Enhanced type hinting and docstrings.
- Minor performance considerations and logging refinements.
- Addressed several pylint warnings through code improvements.
- Consistent use of f-strings and modern Python practices.
- Simplified complex conditional logic where possible.
- Ensured robust handling of Decimal types and API responses.

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
        print(f"{Fore.YELLOW}Note: pandas and numpy often installed via pkg in Termux.")
    else:
        print(f"{Style.BRIGHT}pip install {' '.join(COMMON_PACKAGES)}{Style.RESET_ALL}")
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
V5_HEDGE_MODE_POSITION_IDX = 0
V5_TPSL_MODE_FULL = "Full"
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
    isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) == sys.stdout
    for h in logger.handlers
):
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
logger.propagate = False


# --- Utility Functions ---
def safe_decimal(value: Any, default: Decimal = Decimal("NaN")) -> Decimal:
    """Safely converts a value to Decimal, handling None, empty strings, and invalid formats."""
    if value is None:
        return default
    try:
        # Convert potential floats or numeric types to string first for precise Decimal
        str_value = str(value).strip()
        if not str_value:  # Handle empty string after stripping
            return default
        return Decimal(str_value)
    except (InvalidOperation, ValueError, TypeError):
        # logger.warning(f"Could not convert '{value}' (type: {type(value).__name__}) to Decimal, using default {default}") # Can be noisy
        return default


def termux_notify(title: str, content: str) -> None:
    """Sends a notification via Termux API (toast), if available."""
    if platform.system() == "Linux" and "com.termux" in os.environ.get("PREFIX", ""):
        try:
            # Pass only content to termux-toast
            result = subprocess.run(
                ["termux-toast", content],
                check=False,
                timeout=5,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.warning(
                    f"Termux toast command failed with code {result.returncode}: {result.stderr or result.stdout}"
                )
            # logger.debug(f"Termux toast sent: '{content}' (Title '{title}' ignored by toast)")
        except FileNotFoundError:
            logger.warning("Termux notify failed: 'termux-toast' command not found.")
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
            requests.exceptions.RequestException,
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
                break
        except ccxt.AuthenticationError as e:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}Authentication error: {e}. Check API keys. Halting.",
                exc_info=False,
            )
            sys.exit(1)  # Less verbose exc_info for auth
        except ccxt.InsufficientFunds as e:
            logger.error(f"{Fore.RED}Insufficient funds: {e}")
            last_exception = e
            break  # Don't retry InsufficientFunds
        except ccxt.InvalidOrder as e:
            logger.error(f"{Fore.RED}Invalid order parameters: {e}")
            last_exception = e
            break  # Don't retry InvalidOrder
        except ccxt.OrderNotFound as e:
            logger.warning(f"{Fore.YELLOW}Order not found: {e}")
            last_exception = e
            break  # Often not retryable in context
        except ccxt.PermissionDenied as e:
            logger.error(
                f"{Fore.RED}Permission denied: {e}. Check API permissions/IP whitelisting."
            )
            last_exception = e
            break  # Not retryable
        except ccxt.ExchangeError as e:  # Catch other specific exchange errors
            logger.error(f"{Fore.RED}Exchange error during {func_name}: {e}")
            last_exception = e
            if attempt < max_retries:
                logger.warning(f"Retrying in {delay_seconds}s...")
                time.sleep(delay_seconds)
            else:
                logger.error(
                    f"Max retries reached after exchange error for {func_name}."
                )
                break
        except Exception as e:  # Catch unexpected errors during fetch
            logger.error(
                f"{Fore.RED}Unexpected error during {func_name}: {e}",
                exc_info=True,
            )
            last_exception = e
            break  # Don't retry unknown errors

    # If loop finished without success, raise the last captured exception or a generic one
    if last_exception:
        raise last_exception
    else:
        # This case should ideally not be reached if the loop breaks correctly
        raise RuntimeError(
            f"Function {func_name} failed after unexpected issue without specific exception."
        )


# --- Configuration Class ---
class TradingConfig:
    """Loads, validates, and holds trading configuration parameters from .env."""

    def __init__(self):
        logger.debug("Loading configuration from environment variables...")
        load_dotenv()  # Ensure .env is loaded

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
            min_val=Decimal("0.00001"),
            max_val=Decimal("0.5"),
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
            min_val=Decimal("0.0"),
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
            min_val=Decimal("0.01"),
            max_val=Decimal("10.0"),
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
        self.position_idx: int = self._get_env(
            "POSITION_IDX",
            V5_HEDGE_MODE_POSITION_IDX,
            Style.DIM,
            cast_type=int,
            allowed_values=[0, 1, 2],
        )

        # Indicator Periods (int)
        self.trend_ema_period: int = self._get_env(
            "TREND_EMA_PERIOD", 12, Style.DIM, cast_type=int, min_val=5, max_val=500
        )
        self.fast_ema_period: int = self._get_env(
            "FAST_EMA_PERIOD", 9, Style.DIM, cast_type=int, min_val=1, max_val=200
        )
        self.slow_ema_period: int = self._get_env(
            "SLOW_EMA_PERIOD", 21, Style.DIM, cast_type=int, min_val=2, max_val=500
        )
        self.stoch_period: int = self._get_env(
            "STOCH_PERIOD", 7, Style.DIM, cast_type=int, min_val=1, max_val=100
        )
        self.stoch_smooth_k: int = self._get_env(
            "STOCH_SMOOTH_K", 3, Style.DIM, cast_type=int, min_val=1, max_val=10
        )
        self.stoch_smooth_d: int = self._get_env(
            "STOCH_SMOOTH_D", 3, Style.DIM, cast_type=int, min_val=1, max_val=10
        )
        self.atr_period: int = self._get_env(
            "ATR_PERIOD", 5, Style.DIM, cast_type=int, min_val=1, max_val=100
        )
        self.adx_period: int = self._get_env(
            "ADX_PERIOD", 14, Style.DIM, cast_type=int, min_val=2, max_val=100
        )

        # Signal Logic Thresholds (Decimal)
        self.stoch_oversold_threshold: Decimal = self._get_env(
            "STOCH_OVERSOLD_THRESHOLD",
            DEFAULT_STOCH_OVERSOLD,
            Fore.CYAN,
            cast_type=Decimal,
            min_val=Decimal("0"),
            max_val=Decimal("45"),
        )
        self.stoch_overbought_threshold: Decimal = self._get_env(
            "STOCH_OVERBOUGHT_THRESHOLD",
            DEFAULT_STOCH_OVERBOUGHT,
            Fore.CYAN,
            cast_type=Decimal,
            min_val=Decimal("55"),
            max_val=Decimal("100"),
        )
        self.trend_filter_buffer_percent: Decimal = self._get_env(
            "TREND_FILTER_BUFFER_PERCENT",
            Decimal("0.5"),
            Fore.CYAN,
            cast_type=Decimal,
            min_val=Decimal("0"),
            max_val=Decimal("5"),
        )
        self.atr_move_filter_multiplier: Decimal = self._get_env(
            "ATR_MOVE_FILTER_MULTIPLIER",
            Decimal("0.5"),
            Fore.CYAN,
            cast_type=Decimal,
            min_val=Decimal("0"),
            max_val=Decimal("5"),
        )
        self.min_adx_level: Decimal = self._get_env(
            "MIN_ADX_LEVEL",
            DEFAULT_MIN_ADX,
            Fore.CYAN,
            cast_type=Decimal,
            min_val=Decimal("0"),
            max_val=Decimal("90"),
        )

        # API Keys
        self.api_key: str = self._get_env("BYBIT_API_KEY", None, Fore.RED)
        self.api_secret: str = self._get_env("BYBIT_API_SECRET", None, Fore.RED)

        # Operational Parameters
        self.ohlcv_limit: int = self._get_env(
            "OHLCV_LIMIT",
            DEFAULT_OHLCV_LIMIT,
            Style.DIM,
            cast_type=int,
            min_val=50,
            max_val=1000,
        )
        self.loop_sleep_seconds: int = self._get_env(
            "LOOP_SLEEP_SECONDS",
            DEFAULT_LOOP_SLEEP,
            Style.DIM,
            cast_type=int,
            min_val=5,
        )
        self.order_check_delay_seconds: int = self._get_env(
            "ORDER_CHECK_DELAY_SECONDS",
            2,
            Style.DIM,
            cast_type=int,
            min_val=1,
        )
        self.order_fill_timeout_seconds: int = self._get_env(
            "ORDER_FILL_TIMEOUT_SECONDS",
            20,
            Style.DIM,
            cast_type=int,
            min_val=5,
        )
        self.max_fetch_retries: int = self._get_env(
            "MAX_FETCH_RETRIES",
            DEFAULT_MAX_RETRIES,
            Style.DIM,
            cast_type=int,
            min_val=1,
            max_val=10,
        )
        self.retry_delay_seconds: int = self._get_env(
            "RETRY_DELAY_SECONDS",
            DEFAULT_RETRY_DELAY,
            Style.DIM,
            cast_type=int,
            min_val=1,
        )
        self.trade_only_with_trend: bool = self._get_env(
            "TRADE_ONLY_WITH_TREND", True, Style.DIM, cast_type=bool
        )

        # Journaling
        self.journal_file_path: str = self._get_env(
            "JOURNAL_FILE_PATH", DEFAULT_JOURNAL_FILE, Style.DIM
        )
        self.enable_journaling: bool = self._get_env(
            "ENABLE_JOURNALING", True, Style.DIM, cast_type=bool
        )

        # Final Checks
        if not self.api_key or not self.api_secret:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}BYBIT_API_KEY or BYBIT_API_SECRET not found in environment. Halting."
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
                    "Symbol format must include settle currency (e.g., BTC/USDT:USDT)"
                )
            # settle_curr = self.symbol.split(':')[-1].upper() # Not directly needed for category

            if self.market_type == "inverse":
                category = "inverse"
            elif self.market_type in ["linear", "swap"]:
                category = "linear"  # Linear includes USDT/USDC perpetuals
            else:
                raise ValueError(f"Unsupported MARKET_TYPE '{self.market_type}'")
            logger.info(
                f"Determined Bybit V5 API category: '{category}' for symbol '{self.symbol}' and type '{self.market_type}'"
            )
            return category
        except (ValueError, IndexError) as e:
            logger.critical(
                f"Could not determine V5 category from symbol '{self.symbol}' and type '{self.market_type}': {e}. Halting.",
                exc_info=True,
            )
            sys.exit(1)

    def _validate_config(self):
        """Performs post-load validation of configuration parameters."""
        if self.fast_ema_period >= self.slow_ema_period:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}FAST_EMA ({self.fast_ema_period}) must be < SLOW_EMA ({self.slow_ema_period}). Halting."
            )
            sys.exit(1)
        if self.trend_ema_period <= self.slow_ema_period:
            logger.warning(
                f"{Fore.YELLOW}TREND_EMA ({self.trend_ema_period}) <= SLOW_EMA ({self.slow_ema_period}). Consider increasing trend EMA period."
            )
        if self.stoch_oversold_threshold >= self.stoch_overbought_threshold:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}STOCH_OVERSOLD ({self.stoch_oversold_threshold.normalize()}) must be < STOCH_OVERBOUGHT ({self.stoch_overbought_threshold.normalize()}). Halting."
            )
            sys.exit(1)
        if self.tsl_activation_atr_multiplier < self.sl_atr_multiplier:
            logger.warning(
                f"{Fore.YELLOW}TSL_ACT_MULT ({self.tsl_activation_atr_multiplier.normalize()}) < SL_MULT ({self.sl_atr_multiplier.normalize()}). TSL may activate before initial SL distance is reached."
            )
        if (
            self.tp_atr_multiplier > Decimal("0")
            and self.tp_atr_multiplier <= self.sl_atr_multiplier
        ):
            logger.warning(
                f"{Fore.YELLOW}TP_MULT ({self.tp_atr_multiplier.normalize()}) <= SL_MULT ({self.sl_atr_multiplier.normalize()}). This implies a poor Risk:Reward setup."
            )

    def _cast_value(
        self, key: str, value_str: str, cast_type: Type, default: Any
    ) -> Any:
        """Helper to cast string value to target type."""
        try:
            val_to_cast = value_str.strip()
            if cast_type == bool:
                return val_to_cast.lower() in ["true", "1", "yes", "y", "on"]
            if cast_type == Decimal:
                return Decimal(val_to_cast)
            if cast_type == int:
                # Use Decimal intermediary for float strings like "10.0"
                return int(Decimal(val_to_cast))
            # Default case: use the type constructor directly
            return cast_type(val_to_cast)
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.error(
                f"{Fore.RED}Cast failed for {key} ('{value_str}' -> {cast_type.__name__}): {e}. Using default '{default}'."
            )
            # Attempt to cast the default value itself
            try:
                default_str = str(default).strip()
                if cast_type == bool:
                    return default_str.lower() in [
                        "true",
                        "1",
                        "yes",
                        "y",
                        "on",
                    ]
                if cast_type == Decimal:
                    return Decimal(default_str)
                if cast_type == int:
                    return int(Decimal(default_str))
                return cast_type(default_str)
            except (ValueError, TypeError, InvalidOperation) as cast_default_err:
                logger.critical(
                    f"{Fore.RED + Style.BRIGHT}Default value '{default}' for {key} is invalid for type {cast_type.__name__}: {cast_default_err}. Halting."
                )
                sys.exit(1)

    def _validate_value(
        self,
        key: str,
        value: Any,
        min_val: Optional[Union[int, Decimal]],
        max_val: Optional[Union[int, Decimal]],
        allowed_values: Optional[List[Any]],
    ) -> bool:
        """Helper to validate value against constraints."""
        if allowed_values:
            comp_value = str(value).lower() if isinstance(value, str) else value
            lower_allowed = [
                str(v).lower() for v in allowed_values
            ]  # Case-insensitive check for strings
            if comp_value not in lower_allowed:
                logger.error(
                    f"{Fore.RED}Validation failed for {key}: Invalid value '{value}'. Allowed: {allowed_values}."
                )
                return False
        if min_val is not None and value < min_val:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}Validation failed for {key}: {value} < minimum {min_val}. Halting."
            )
            sys.exit(1)
        if max_val is not None and value > max_val:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}Validation failed for {key}: {value} > maximum {max_val}. Halting."
            )
            sys.exit(1)
        return True

    def _get_env(
        self,
        key: str,
        default: Any,
        color: str,
        cast_type: Type = str,
        min_val: Optional[Union[int, Decimal]] = None,
        max_val: Optional[Union[int, Decimal]] = None,
        allowed_values: Optional[List[Any]] = None,
    ) -> Any:
        """Gets value from environment, casts, validates, logs, and handles defaults."""
        value_str = os.getenv(key)
        is_secret = "SECRET" in key.upper() or "KEY" in key.upper()
        log_value = "****" if is_secret and value_str else value_str

        final_value_str: Optional[str] = None

        if value_str is None or value_str.strip() == "":
            if default is None:
                # Secrets or required non-secret configs must be present
                logger.critical(
                    f"{Fore.RED + Style.BRIGHT}Required configuration '{key}' not found in environment and no default provided. Halting."
                )
                sys.exit(1)
            else:
                final_value_str = str(default)
                logger.warning(f"{color}Using default for {key}: {default}")
        else:
            final_value_str = value_str
            logger.info(f"{color}Found {key}: {log_value}")

        # We should always have a string to cast at this point
        if final_value_str is None:
            # This case should be prevented by the checks above, but as a safeguard:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}Internal error: Could not determine value string for {key}. Halting."
            )
            sys.exit(1)

        # Cast the value (handles casting default on error)
        casted_value = self._cast_value(key, final_value_str, cast_type, default)

        # Validate the casted value
        if not self._validate_value(
            key, casted_value, min_val, max_val, allowed_values
        ):
            # Validation failed, and it wasn't caught by min/max critical exits
            # This happens for `allowed_values` failure. Revert to default.
            logger.warning(
                f"Reverting {key} to default '{default}' due to validation failure."
            )
            # Recast the default value (safe because _cast_value handles default casting errors)
            casted_value = self._cast_value(key, str(default), cast_type, default)
            # Re-validate the default value itself (important!)
            if not self._validate_value(
                key, casted_value, min_val, max_val, allowed_values
            ):
                logger.critical(
                    f"{Fore.RED + Style.BRIGHT}Default value '{default}' for {key} failed validation. Halting."
                )
                sys.exit(1)

        return casted_value


# --- Exchange Manager Class ---
class ExchangeManager:
    """Handles CCXT exchange interactions, data fetching, and formatting."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.exchange: Optional[ccxt.Exchange] = None
        self.market_info: Optional[Dict] = None
        self._initialize_exchange()

    def _initialize_exchange(self):
        """Initializes the CCXT exchange instance."""
        logger.info("Initializing Bybit exchange interface (V5)...")
        try:
            exchange_params = {
                "apiKey": self.config.api_key,
                "secret": self.config.api_secret,
                "options": {
                    "defaultType": self.config.market_type,
                    "adjustForTimeDifference": True,
                    "recvWindow": 10000,
                    "brokerId": "TermuxPyrmV5",  # Custom ID for tracking
                    "createMarketBuyOrderRequiresPrice": False,
                    "v5": True,  # Ensure V5 API is used
                    # 'enableRateLimit': True, # Default is usually True
                },
            }
            # Example: Handle sandbox endpoint if needed via environment variable
            # if os.getenv("USE_SANDBOX", "false").lower() == "true":
            #     exchange_params['urls'] = {'api': 'https://api-testnet.bybit.com'}

            self.exchange = ccxt.bybit(exchange_params)
            logger.info(f"{Fore.GREEN}Bybit V5 interface initialized successfully.")
            self.market_info = self._load_market_info()

        except ccxt.AuthenticationError as e:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}Authentication failed: {e}. Check API keys/permissions.",
                exc_info=False,
            )
            sys.exit(1)
        except ccxt.NetworkError as e:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}Network error initializing exchange: {e}. Check connection.",
                exc_info=True,
            )
            sys.exit(1)
        except Exception as e:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}Unexpected error initializing exchange: {e}",
                exc_info=True,
            )
            sys.exit(1)

    def _load_market_info(self) -> Optional[Dict]:
        """Loads and caches market information for the configured symbol."""
        if not self.exchange:
            logger.error("Exchange not initialized, cannot load market info.")
            return None
        try:
            logger.debug(f"Loading market info for {self.config.symbol}...")
            self.exchange.load_markets(True)  # Force reload
            market = self.exchange.market(self.config.symbol)
            if not market:
                raise ccxt.ExchangeError(
                    f"Market {self.config.symbol} not found on exchange."
                )

            # Extract precision safely and convert to integer decimal places
            amount_precision_dec = safe_decimal(
                market.get("precision", {}).get("amount"),
                default=Decimal(f"1e-{DEFAULT_AMOUNT_DP}"),
            )
            price_precision_dec = safe_decimal(
                market.get("precision", {}).get("price"),
                default=Decimal(f"1e-{DEFAULT_PRICE_DP}"),
            )

            amount_dp = (
                abs(amount_precision_dec.as_tuple().exponent)
                if not amount_precision_dec.is_nan()
                else DEFAULT_AMOUNT_DP
            )
            price_dp = (
                abs(price_precision_dec.as_tuple().exponent)
                if not price_precision_dec.is_nan()
                else DEFAULT_PRICE_DP
            )

            market["precision_dp"] = {"amount": amount_dp, "price": price_dp}
            # Store minimum tick size as Decimal for calculations
            market["tick_size"] = Decimal("1e-" + str(price_dp))
            # Store minimum order size as Decimal
            market["min_order_size"] = safe_decimal(
                market.get("limits", {}).get("amount", {}).get("min"),
                default=Decimal("NaN"),
            )

            min_amt_str = (
                market["min_order_size"].normalize()
                if not market["min_order_size"].is_nan()
                else "N/A"
            )
            logger.info(
                f"Market info loaded: ID={market.get('id')}, "
                f"Precision(AmtDP={amount_dp}, PriceDP={price_dp}), "
                f"Limits(MinAmt={min_amt_str})"
            )
            return market
        except (ccxt.ExchangeError, KeyError, Exception) as e:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}Failed to load market info for {self.config.symbol}: {e}. Halting.",
                exc_info=True,
            )
            sys.exit(1)

    def format_price(self, price: Union[Decimal, str, float, int]) -> str:
        """Formats price according to market precision using ROUND_HALF_EVEN."""
        price_decimal = safe_decimal(price)
        if price_decimal.is_nan():
            return "NaN"  # Return NaN string if input was bad

        if (
            self.market_info
            and "precision_dp" in self.market_info
            and "price" in self.market_info["precision_dp"]
        ):
            precision = self.market_info["precision_dp"]["price"]
            quantizer = Decimal("1e-" + str(precision))
            return str(price_decimal.quantize(quantizer, rounding=ROUND_HALF_EVEN))
        else:
            logger.warning(
                "Market info/price precision unavailable, using default formatting."
            )
            # Fallback to a reasonable default precision if market info missing
            quantizer = Decimal("1e-" + str(DEFAULT_PRICE_DP))
            return str(price_decimal.quantize(quantizer, rounding=ROUND_HALF_EVEN))

    def format_amount(
        self,
        amount: Union[Decimal, str, float, int],
        rounding_mode=ROUND_DOWN,
    ) -> str:
        """Formats amount according to market precision using specified rounding."""
        amount_decimal = safe_decimal(amount)
        if amount_decimal.is_nan():
            return "NaN"  # Return NaN string if input was bad

        if (
            self.market_info
            and "precision_dp" in self.market_info
            and "amount" in self.market_info["precision_dp"]
        ):
            precision = self.market_info["precision_dp"]["amount"]
            quantizer = Decimal("1e-" + str(precision))
            return str(amount_decimal.quantize(quantizer, rounding=rounding_mode))
        else:
            logger.warning(
                "Market info/amount precision unavailable, using default formatting."
            )
            # Fallback to a reasonable default precision
            quantizer = Decimal("1e-" + str(DEFAULT_AMOUNT_DP))
            return str(amount_decimal.quantize(quantizer, rounding=rounding_mode))

    def fetch_ohlcv(self) -> Optional[pd.DataFrame]:
        """Fetches OHLCV data with retries and converts to DataFrame."""
        if not self.exchange:
            logger.error("Exchange not initialized, cannot fetch OHLCV.")
            return None
        logger.debug(
            f"Fetching {self.config.ohlcv_limit} candles for {self.config.symbol} ({self.config.interval})..."
        )
        try:
            ohlcv = fetch_with_retries(
                self.exchange.fetch_ohlcv,  # Pass the function itself
                self.config.symbol,
                self.config.interval,
                limit=self.config.ohlcv_limit,
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
            )
            if not ohlcv:
                logger.error("fetch_ohlcv returned empty list.")
                return None

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            # Convert timestamp to UTC datetime and set as index
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)

            # Convert OHLCV columns to Decimal robustly
            for col in ["open", "high", "low", "close", "volume"]:
                # Use safe_decimal utility which handles various input types
                df[col] = df[col].apply(safe_decimal)

            logger.debug(f"Fetched {len(df)} candles. Last timestamp: {df.index[-1]}")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV data: {e}", exc_info=True)
            return None

    def get_balance(self) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Fetches total equity and available balance for the settlement currency using V5 API."""
        if not self.exchange or not self.market_info:
            logger.error("Exchange or market info not available, cannot fetch balance.")
            return None, None

        settle_currency = self.market_info.get("settle")
        if not settle_currency:
            logger.error(
                "Settle currency not found in market info. Cannot determine balance currency."
            )
            return None, None

        logger.debug(
            f"Fetching balance for {settle_currency} (Category: {self.config.bybit_v5_category})..."
        )
        try:
            # V5 balance requires category and accountType
            params = {
                "category": self.config.bybit_v5_category,
                "accountType": V5_UNIFIED_ACCOUNT_TYPE,
            }
            balance_data = fetch_with_retries(
                self.exchange.fetch_balance,  # Pass function
                params=params,
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
            )
            # logger.debug(f"Raw balance data: {balance_data}") # Verbose

            total_equity = Decimal("NaN")
            available_balance = Decimal("NaN")

            # Attempt to parse V5 response structure (unified format)
            # CCXT unified structure might place details under `total` or `free` keys
            # Or sometimes within `info` for specific account types
            if (
                "info" in balance_data
                and "result" in balance_data["info"]
                and "list" in balance_data["info"]["result"]
            ):
                acc_list = balance_data["info"]["result"]["list"]
                if acc_list:
                    unified_acc = next(
                        (
                            item
                            for item in acc_list
                            if item.get("accountType") == V5_UNIFIED_ACCOUNT_TYPE
                        ),
                        None,
                    )
                    if unified_acc:
                        # V5 Unified account equity is often at the account level
                        total_equity = safe_decimal(unified_acc.get("totalEquity"))
                        # Available balance might be totalAvailableBalance or similar
                        available_balance = safe_decimal(
                            unified_acc.get("totalAvailableBalance")
                        )
                        # If not found at account level, check coin list within the account
                        if (
                            total_equity.is_nan() or available_balance.is_nan()
                        ) and "coin" in unified_acc:
                            for coin_info in unified_acc["coin"]:
                                if coin_info.get("coin") == settle_currency:
                                    # Use safe_decimal for robust conversion
                                    if total_equity.is_nan():
                                        total_equity = safe_decimal(
                                            coin_info.get("equity")
                                        )
                                    if available_balance.is_nan():
                                        # 'availableToWithdraw' or 'availableBalance' might be used
                                        available_balance = safe_decimal(
                                            coin_info.get("availableToWithdraw")
                                        ) or safe_decimal(
                                            coin_info.get("availableBalance")
                                        )
                                    break

            # Fallback to top-level CCXT unified structure if V5 parsing failed
            if total_equity.is_nan() and settle_currency in balance_data.get(
                "total", {}
            ):
                total_equity = safe_decimal(balance_data[settle_currency].get("total"))
            if available_balance.is_nan() and settle_currency in balance_data.get(
                "free", {}
            ):
                available_balance = safe_decimal(
                    balance_data[settle_currency].get("free")
                )

            # Final check and default for available balance
            if total_equity.is_nan():
                logger.error(
                    f"Could not extract valid total equity for {settle_currency}. Raw balance response snippet: {str(balance_data)[:500]}"
                )
                return None, None
            if available_balance.is_nan():
                logger.warning(
                    f"Could not extract valid available balance for {settle_currency}. Defaulting to 0."
                )
                available_balance = Decimal("0")

            logger.debug(
                f"Balance Fetched: Equity={total_equity.normalize()}, Available={available_balance.normalize()} {settle_currency}"
            )
            return total_equity, available_balance
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}", exc_info=True)
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
            logger.error("Market ID not found in market info.")
            return None

        logger.debug(
            f"Fetching position for {self.config.symbol} (ID: {market_id}, Category: {self.config.bybit_v5_category}, PosIdx: {self.config.position_idx})..."
        )
        positions_dict: Dict[str, Dict[str, Any]] = {"long": {}, "short": {}}

        try:
            params = {
                "category": self.config.bybit_v5_category,
                "symbol": market_id,
                "positionIdx": self.config.position_idx,  # Filter by index if API supports it directly
            }
            # V5 fetch_positions often prefers a list of symbols, even if just one
            position_data = fetch_with_retries(
                self.exchange.fetch_positions,  # Pass function
                symbols=[self.config.symbol],
                params=params,
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
            )
            # logger.debug(f"Raw position data: {position_data}") # Verbose

            if not position_data:
                logger.debug("No position data returned from fetch_positions.")
                return positions_dict

            # V5 returns a list, find the specific position matching symbol and index
            pos_info = None
            for p in position_data:
                info = p.get("info", {})
                # Ensure matching symbol AND position index
                if (
                    info.get("symbol") == market_id
                    and int(info.get("positionIdx", -1)) == self.config.position_idx
                ):
                    pos_info = info
                    break

            if not pos_info:
                logger.debug(
                    f"No position found matching symbol {market_id} and positionIdx {self.config.position_idx} in the returned data."
                )
                return positions_dict

            # Use safe_decimal for robust conversion from API strings
            qty = safe_decimal(pos_info.get("size", "0"))
            side = pos_info.get("side", "None").lower()  # 'Buy'/'Sell'/'None'
            entry_price = safe_decimal(pos_info.get("avgPrice", "0"))
            liq_price_raw = safe_decimal(pos_info.get("liqPrice", "0"))
            unrealized_pnl = safe_decimal(pos_info.get("unrealisedPnl", "0"))
            sl_price_raw = safe_decimal(pos_info.get("stopLoss", "0"))
            tp_price_raw = safe_decimal(pos_info.get("takeProfit", "0"))
            # Non-zero value often indicates active TSL trigger price
            tsl_active_price_raw = safe_decimal(pos_info.get("trailingStop", "0"))

            # Validate and clean up values
            liq_price = (
                liq_price_raw
                if not liq_price_raw.is_nan() and liq_price_raw > 0
                else Decimal("NaN")
            )
            sl_price = (
                sl_price_raw if not sl_price_raw.is_nan() and sl_price_raw > 0 else None
            )
            tp_price = (
                tp_price_raw if not tp_price_raw.is_nan() and tp_price_raw > 0 else None
            )
            is_tsl_active = (
                not tsl_active_price_raw.is_nan() and tsl_active_price_raw > 0
            )

            position_details = {
                "qty": qty if not qty.is_nan() else Decimal("0"),
                "entry_price": entry_price
                if not entry_price.is_nan() and entry_price > 0
                else Decimal("NaN"),
                "liq_price": liq_price,
                "unrealized_pnl": unrealized_pnl
                if not unrealized_pnl.is_nan()
                else Decimal("0"),
                "side": side,
                "info": pos_info,  # Store raw info for potential debugging
                "stop_loss_price": sl_price,
                "take_profit_price": tp_price,
                "is_tsl_active": is_tsl_active,
            }

            # Populate the correct side in the dictionary if quantity is significant
            if (
                side == "buy"
                and position_details["qty"].copy_abs() >= POSITION_QTY_EPSILON
            ):
                positions_dict["long"] = position_details
                entry_str = (
                    position_details["entry_price"].normalize()
                    if not position_details["entry_price"].is_nan()
                    else "N/A"
                )
                logger.debug(
                    f"Found LONG position: Qty={position_details['qty'].normalize()}, Entry={entry_str}"
                )
            elif (
                side == "sell"
                and position_details["qty"].copy_abs() >= POSITION_QTY_EPSILON
            ):
                positions_dict["short"] = position_details
                entry_str = (
                    position_details["entry_price"].normalize()
                    if not position_details["entry_price"].is_nan()
                    else "N/A"
                )
                logger.debug(
                    f"Found SHORT position: Qty={position_details['qty'].normalize()}, Entry={entry_str}"
                )
            elif side != "none":
                logger.debug(
                    f"Position found but size negligible or NaN. Side: {side}, Qty: {position_details['qty']}"
                )
            else:
                logger.debug("Position side is 'None'. Considered flat.")

            return positions_dict

        except Exception as e:
            logger.error(f"Failed to fetch/parse positions: {e}", exc_info=True)
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
            logger.error(f"{Fore.RED}DataFrame missing required columns: {missing}")
            return None

        try:
            # Work with a copy, ensure numeric types (handle potential Decimals from fetch)
            df_calc = df[required_cols].copy()
            for col in required_cols:
                # Convert Decimal to float for calculations, map NaN Decimals to np.nan
                df_calc[col] = df_calc[col].map(
                    lambda x: float(x)
                    if isinstance(x, Decimal) and not x.is_nan()
                    else np.nan
                )

            # Drop rows with NaN in essential OHLC columns after conversion
            df_calc.dropna(subset=required_cols, inplace=True)

            if df_calc.empty:
                logger.error(
                    f"{Fore.RED}DataFrame empty after NaN drop and conversion."
                )
                return None

            # Check Data Length against the maximum period needed by indicators
            min_required_len = (
                max(
                    self.config.slow_ema_period,
                    self.config.trend_ema_period,
                    self.config.stoch_period
                    + self.config.stoch_smooth_k
                    + self.config.stoch_smooth_d
                    - 2,  # Stoch needs lookback + smoothing
                    self.config.atr_period,
                    self.config.adx_period * 2,  # ADX needs ~2x period for smoothing
                )
                + 5  # Add a buffer
            )
            if len(df_calc) < min_required_len:
                logger.error(
                    f"{Fore.RED}Insufficient data ({len(df_calc)} rows < required {min_required_len}) for reliable indicator calculation."
                )
                return None

            close_s = df_calc["close"]
            high_s = df_calc["high"]
            low_s = df_calc["low"]

            # --- Calculate Indicators using Pandas/NumPy ---
            fast_ema_s = close_s.ewm(
                span=self.config.fast_ema_period, adjust=False
            ).mean()
            slow_ema_s = close_s.ewm(
                span=self.config.slow_ema_period, adjust=False
            ).mean()
            trend_ema_s = close_s.ewm(
                span=self.config.trend_ema_period, adjust=False
            ).mean()

            # Stochastic Oscillator (%K, %D)
            low_min = low_s.rolling(window=self.config.stoch_period).min()
            high_max = high_s.rolling(window=self.config.stoch_period).max()
            stoch_range = high_max - low_min
            # Avoid division by zero; default to 50 if range is zero
            stoch_k_raw = np.where(
                stoch_range > 1e-12,
                100 * (close_s - low_min) / stoch_range,
                50.0,
            )
            stoch_k_raw_s = pd.Series(stoch_k_raw, index=df_calc.index).fillna(
                50
            )  # Fill NaNs from rolling window
            stoch_k_s = (
                stoch_k_raw_s.rolling(window=self.config.stoch_smooth_k)
                .mean()
                .fillna(50)
            )
            stoch_d_s = (
                stoch_k_s.rolling(window=self.config.stoch_smooth_d).mean().fillna(50)
            )

            # Average True Range (ATR - Wilder's smoothing)
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
                """Safely get the last value from a Series and convert to Decimal."""
                if series.empty or pd.isna(series.iloc[-1]):
                    # logger.warning(f"Indicator '{name}' latest value is NaN.")
                    return Decimal("NaN")
                try:
                    # Convert float string representation for precision
                    return Decimal(str(series.iloc[-1]))
                except (InvalidOperation, TypeError):
                    logger.error(
                        f"Failed converting latest {name} value '{series.iloc[-1]}' to Decimal."
                    )
                    return Decimal("NaN")

            indicators_out = {
                "fast_ema": get_latest_decimal(fast_ema_s, "fast_ema"),
                "slow_ema": get_latest_decimal(slow_ema_s, "slow_ema"),
                "trend_ema": get_latest_decimal(trend_ema_s, "trend_ema"),
                "stoch_k": get_latest_decimal(stoch_k_s, "stoch_k"),
                "stoch_d": get_latest_decimal(stoch_d_s, "stoch_d"),
                "atr": get_latest_decimal(atr_s, "atr"),
                "atr_period": self.config.atr_period,  # Store period used
                "adx": get_latest_decimal(adx_s, "adx"),
                "pdi": get_latest_decimal(pdi_s, "pdi"),
                "mdi": get_latest_decimal(mdi_s, "mdi"),
            }

            # --- Calculate Stochastic Cross Signals using latest Decimal values ---
            k_last = indicators_out["stoch_k"]
            d_last = indicators_out["stoch_d"]
            k_prev = (
                get_latest_decimal(stoch_k_s.iloc[:-1], "stoch_k_prev")
                if len(stoch_k_s) >= 2
                else Decimal("NaN")
            )
            d_prev = (
                get_latest_decimal(stoch_d_s.iloc[:-1], "stoch_d_prev")
                if len(stoch_d_s) >= 2
                else Decimal("NaN")
            )

            stoch_kd_bullish = False
            stoch_kd_bearish = False
            if not any(v.is_nan() for v in [k_last, d_last, k_prev, d_prev]):
                crossed_above = (k_last > d_last) and (k_prev <= d_prev)
                crossed_below = (k_last < d_last) and (k_prev >= d_prev)
                # Check if previous K or D was in the zone
                prev_oversold = (k_prev <= self.config.stoch_oversold_threshold) or (
                    d_prev <= self.config.stoch_oversold_threshold
                )
                prev_overbought = (
                    k_prev >= self.config.stoch_overbought_threshold
                ) or (d_prev >= self.config.stoch_overbought_threshold)

                if crossed_above and prev_oversold:
                    stoch_kd_bullish = True
                if crossed_below and prev_overbought:
                    stoch_kd_bearish = True

            indicators_out["stoch_kd_bullish"] = stoch_kd_bullish
            indicators_out["stoch_kd_bearish"] = stoch_kd_bearish

            # Final check for critical NaNs in output
            critical_keys = [
                "fast_ema",
                "slow_ema",
                "trend_ema",
                "atr",
                "stoch_k",
                "stoch_d",
                "adx",
                "pdi",
                "mdi",
            ]
            failed_indicators = [
                k
                for k in critical_keys
                if indicators_out.get(k, Decimal("NaN")).is_nan()
            ]
            if failed_indicators:
                logger.error(
                    f"{Fore.RED}Critical indicators calculated as NaN: {', '.join(failed_indicators)}. Check data source or periods."
                )
                return None  # Return None if essential indicators failed

            logger.info(f"{Fore.GREEN}Indicator patterns woven successfully.")
            return indicators_out

        except Exception as e:
            logger.error(
                f"{Fore.RED}Failed weaving indicator patterns: {e}",
                exc_info=True,
            )
            return None

    def _calculate_adx(
        self,
        high_s: pd.Series,
        low_s: pd.Series,
        close_s: pd.Series,  # close_s not directly used but standard for ADX context
        atr_s: pd.Series,
        period: int,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Helper to calculate ADX, +DI, -DI using Wilder's smoothing."""
        # Calculate Directional Movement (+DM, -DM)
        move_up = high_s.diff()
        move_down = -low_s.diff()  # diff is current - previous

        plus_dm = np.where((move_up > move_down) & (move_up > 0), move_up, 0.0)
        minus_dm = np.where((move_down > move_up) & (move_down > 0), move_down, 0.0)

        # Smoothed +DM, -DM using Wilder's method (equivalent to EMA with alpha=1/period)
        plus_dm_s = (
            pd.Series(plus_dm, index=high_s.index)
            .ewm(alpha=1 / period, adjust=False)
            .mean()
        )
        minus_dm_s = (
            pd.Series(minus_dm, index=high_s.index)
            .ewm(alpha=1 / period, adjust=False)
            .mean()
        )

        # Calculate Directional Indicators (+DI, -DI)
        # Handle division by zero if ATR is zero
        pdi_s = np.where(atr_s > 1e-12, 100 * plus_dm_s / atr_s, 0.0)
        mdi_s = np.where(atr_s > 1e-12, 100 * minus_dm_s / atr_s, 0.0)
        pdi_s = pd.Series(pdi_s, index=high_s.index).fillna(0)
        mdi_s = pd.Series(mdi_s, index=high_s.index).fillna(0)

        # Calculate Directional Movement Index (DX)
        di_diff_abs = (pdi_s - mdi_s).abs()
        di_sum = pdi_s + mdi_s
        # Handle division by zero if sum is zero
        dx_s = np.where(di_sum > 1e-12, 100 * di_diff_abs / di_sum, 0.0)
        dx_s = pd.Series(dx_s, index=high_s.index).fillna(0)

        # Calculate Average Directional Index (ADX - Smoothed DX)
        adx_s = dx_s.ewm(alpha=1 / period, adjust=False).mean().fillna(0)

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
            "reason": "Initialization",
        }

        if not indicators:
            result["reason"] = "Indicators missing"
            return result
        if df_last_candles is None or len(df_last_candles) < 2:
            result["reason"] = (
                f"Insufficient candle data (<2, got {len(df_last_candles) if df_last_candles is not None else 0})"
            )
            return result

        try:
            # --- Extract Data ---
            latest_candle = df_last_candles.iloc[-1]
            prev_candle = df_last_candles.iloc[-2]
            current_price = safe_decimal(latest_candle["close"])
            prev_close = safe_decimal(prev_candle["close"])

            if current_price.is_nan() or current_price <= 0:
                result["reason"] = f"Invalid current price ({current_price})"
                return result

            # Get indicator values safely, checking for NaN
            required_indicators = {
                "k": indicators.get("stoch_k"),
                "fast_ema": indicators.get("fast_ema"),
                "slow_ema": indicators.get("slow_ema"),
                "trend_ema": indicators.get("trend_ema"),
                "atr": indicators.get("atr"),
                "adx": indicators.get("adx"),
                "pdi": indicators.get("pdi"),
                "mdi": indicators.get("mdi"),
            }
            nan_keys = [
                name
                for name, val in required_indicators.items()
                if isinstance(val, Decimal) and val.is_nan()
            ]
            if nan_keys:
                result["reason"] = f"Required indicator(s) NaN: {', '.join(nan_keys)}"
                return result

            # Assign validated indicators to local variables for readability
            k, fast_ema, slow_ema, trend_ema, atr, adx, pdi, mdi = (
                required_indicators["k"],
                required_indicators["fast_ema"],
                required_indicators["slow_ema"],
                required_indicators["trend_ema"],
                required_indicators["atr"],
                required_indicators["adx"],
                required_indicators["pdi"],
                required_indicators["mdi"],
            )
            kd_bull = indicators.get("stoch_kd_bullish", False)
            kd_bear = indicators.get("stoch_kd_bearish", False)

            # --- Define Conditions ---
            # EMA Cross
            ema_bullish_cross = fast_ema > slow_ema
            ema_bearish_cross = fast_ema < slow_ema

            # Trend Filter (Price relative to Trend EMA with buffer)
            trend_buffer = trend_ema.copy_abs() * (
                self.config.trend_filter_buffer_percent / 100
            )
            price_above_trend = current_price > (trend_ema - trend_buffer)
            price_below_trend = current_price < (trend_ema + trend_buffer)
            trend_long_ok = (
                price_above_trend if self.config.trade_only_with_trend else True
            )
            trend_short_ok = (
                price_below_trend if self.config.trade_only_with_trend else True
            )

            # Stochastic Condition (Oversold/Overbought or KD Cross)
            stoch_long_cond = (k < self.config.stoch_oversold_threshold) or kd_bull
            stoch_short_cond = (k > self.config.stoch_overbought_threshold) or kd_bear

            # ATR Move Filter
            atr_move_threshold = atr * self.config.atr_move_filter_multiplier
            significant_move = False
            atr_reason = "Move Filter OFF"
            if atr_move_threshold > 0:
                if atr <= 0:
                    atr_reason = f"Move Filter Skipped (Invalid ATR: {atr.normalize()})"
                elif prev_close.is_nan():
                    atr_reason = "Move Filter Skipped (Prev Close NaN)"
                else:
                    move = (current_price - prev_close).copy_abs()
                    significant_move = move > atr_move_threshold
                    atr_reason = f"Move({move.normalize():.4f}) {' > ' if significant_move else ' <= '} Thresh({atr_move_threshold.normalize():.4f})"
            else:  # Filter is off
                significant_move = True  # Always pass if filter is off

            # ADX Filter (Trending and Direction Confirmation)
            adx_trending = adx > self.config.min_adx_level
            adx_long_confirm = adx_trending and pdi > mdi
            adx_short_confirm = adx_trending and mdi > pdi
            adx_base_reason = f"ADX({adx:.1f})"
            if adx_trending:
                adx_dir_reason = (
                    f"+DI({pdi:.1f}) > -DI({mdi:.1f})"
                    if pdi > mdi
                    else f"-DI({mdi:.1f}) > +DI({pdi:.1f})"
                )
                adx_reason = f"{adx_base_reason} > {self.config.min_adx_level.normalize()} & {adx_dir_reason}"
            else:
                adx_reason = (
                    f"{adx_base_reason} <= {self.config.min_adx_level.normalize()}"
                )

            # --- Combine Logic for Potential Signals ---
            base_long_cond = ema_bullish_cross and stoch_long_cond
            base_short_cond = ema_bearish_cross and stoch_short_cond

            long_signal = (
                base_long_cond
                and trend_long_ok
                and significant_move
                and adx_long_confirm
            )
            short_signal = (
                base_short_cond
                and trend_short_ok
                and significant_move
                and adx_short_confirm
            )

            # --- Build Reason String ---
            if long_signal:
                reason_parts = [
                    "EMA Bull",
                    f"Stoch Long ({k:.1f})",
                    adx_reason,
                    atr_reason,
                ]
                if self.config.trade_only_with_trend:
                    reason_parts.insert(2, "Trend OK")
                result["long"] = True
                result["reason"] = "Long Signal: " + " | ".join(
                    p for p in reason_parts if p
                )
            elif short_signal:
                reason_parts = [
                    "EMA Bear",
                    f"Stoch Short ({k:.1f})",
                    adx_reason,
                    atr_reason,
                ]
                if self.config.trade_only_with_trend:
                    reason_parts.insert(2, "Trend OK")
                result["short"] = True
                result["reason"] = "Short Signal: " + " | ".join(
                    p for p in reason_parts if p
                )
            else:
                # Explain why no signal or why blocked
                reason_parts = []
                if not base_long_cond and not base_short_cond:
                    reason_parts.append(
                        f"EMA Cross ({'Bull' if ema_bullish_cross else 'Bear' if ema_bearish_cross else 'None'})"
                    )
                    reason_parts.append(
                        f"Stoch Cond ({'Long OK' if stoch_long_cond else 'Long !OK'} / {'Short OK' if stoch_short_cond else 'Short !OK'})"
                    )
                elif base_long_cond and not trend_long_ok:
                    reason_parts.append(
                        f"Long Blocked (Trend: P {current_price.normalize()} !> T {trend_ema.normalize():.2f} - Buf)"
                    )
                elif base_short_cond and not trend_short_ok:
                    reason_parts.append(
                        f"Short Blocked (Trend: P {current_price.normalize()} !< T {trend_ema.normalize():.2f} + Buf)"
                    )
                elif (base_long_cond or base_short_cond) and not significant_move:
                    reason_parts.append(f"Blocked ({atr_reason})")
                elif base_long_cond and significant_move and not adx_long_confirm:
                    reason_parts.append(f"Long Blocked ({adx_reason})")
                elif base_short_cond and significant_move and not adx_short_confirm:
                    reason_parts.append(f"Short Blocked ({adx_reason})")
                else:  # Default case if logic missed something
                    reason_parts.append("Conditions unmet")

                result["reason"] = "No Signal: " + " | ".join(
                    p for p in reason_parts if p
                )

            log_level_sig = (
                logging.INFO
                if result["long"] or result["short"] or "Blocked" in result["reason"]
                else logging.DEBUG
            )
            logger.log(log_level_sig, f"Signal Check: {result['reason']}")

        except Exception as e:
            logger.error(f"{Fore.RED}Error generating signals: {e}", exc_info=True)
            result["reason"] = f"Exception: {e}"
            result["long"] = False
            result["short"] = False

        return result

    def check_exit_signals(
        self,
        position_side: str,
        indicators: Dict[str, Union[Decimal, bool, int]],
    ) -> Optional[str]:
        """Checks for signal-based exits (EMA cross, Stoch reversal). Returns reason string or None."""
        if not indicators:
            logger.warning("Cannot check exit signals: indicators missing.")
            return None

        fast_ema = indicators.get("fast_ema", Decimal("NaN"))
        slow_ema = indicators.get("slow_ema", Decimal("NaN"))
        stoch_k = indicators.get("stoch_k", Decimal("NaN"))

        if fast_ema.is_nan() or slow_ema.is_nan() or stoch_k.is_nan():
            logger.warning(
                "Cannot check exit signals due to NaN indicators (EMA/Stoch)."
            )
            return None

        exit_reason: Optional[str] = None
        ema_bear_cross = fast_ema < slow_ema
        ema_bull_cross = fast_ema > slow_ema
        stoch_exit_long_cond = stoch_k > self.config.stoch_overbought_threshold
        stoch_exit_short_cond = stoch_k < self.config.stoch_oversold_threshold

        if position_side == "long":
            if ema_bear_cross:
                exit_reason = "EMA Bearish Cross"
            elif stoch_exit_long_cond:
                exit_reason = f"Stoch Overbought ({stoch_k:.1f} > {self.config.stoch_overbought_threshold})"
        elif position_side == "short":
            if ema_bull_cross:
                exit_reason = "EMA Bullish Cross"
            elif stoch_exit_short_cond:
                exit_reason = f"Stoch Oversold ({stoch_k:.1f} < {self.config.stoch_oversold_threshold})"

        if exit_reason:
            logger.trade(
                f"{Fore.YELLOW}Exit Signal vs {position_side.upper()}: {exit_reason}."
            )
        return exit_reason


# --- Order Manager Class ---
class OrderManager:
    """Handles order placement, position protection (SL/TP/TSL), and closing using V5 API."""

    def __init__(self, config: TradingConfig, exchange_manager: ExchangeManager):
        self.config = config
        self.exchange_manager = exchange_manager
        self.exchange = exchange_manager.exchange
        self.market_info = exchange_manager.market_info
        # Tracks active protection STATUS (not order IDs) for V5 position stops
        # Values: 'ACTIVE_SLTP', 'ACTIVE_TSL', None
        self.protection_tracker: Dict[str, Optional[str]] = {
            "long": None,
            "short": None,
        }

    def _calculate_trade_parameters(
        self,
        side: str,
        atr: Decimal,
        total_equity: Decimal,
        current_price: Decimal,
    ) -> Optional[Dict[str, Decimal]]:
        """Calculates SL price, TP price, quantity, and TSL distance based on risk and ATR."""
        if atr.is_nan() or atr <= 0:
            logger.error(f"Invalid ATR ({atr}) for parameter calculation.")
            return None
        if total_equity.is_nan() or total_equity <= 0:
            logger.error(f"Invalid equity ({total_equity}) for parameter calculation.")
            return None
        if current_price.is_nan() or current_price <= 0:
            logger.error(f"Invalid price ({current_price}) for parameter calculation.")
            return None
        if not self.market_info or "tick_size" not in self.market_info:
            logger.error("Market info or tick size missing for parameter calculation.")
            return None

        try:
            risk_amount = total_equity * self.config.risk_percentage
            sl_distance_atr = atr * self.config.sl_atr_multiplier
            tp_distance_atr = (
                atr * self.config.tp_atr_multiplier
                if self.config.tp_atr_multiplier > 0
                else Decimal("0")
            )

            if side == "buy":
                sl_price = current_price - sl_distance_atr
                tp_price = (
                    current_price + tp_distance_atr if tp_distance_atr > 0 else None
                )
            elif side == "sell":
                sl_price = current_price + sl_distance_atr
                tp_price = (
                    current_price - tp_distance_atr if tp_distance_atr > 0 else None
                )
            else:
                logger.error(f"Invalid side '{side}' for calculation.")
                return None

            # Validate SL/TP prices > 0
            if sl_price <= 0:
                logger.error(
                    f"Calculated SL price ({sl_price}) is invalid (<=0). Cannot calculate quantity."
                )
                return None
            if tp_price is not None and tp_price <= 0:
                logger.warning(
                    f"Calculated TP price ({tp_price}) is invalid (<=0). Setting TP to None."
                )
                tp_price = None

            # Ensure SL distance is meaningful (at least one tick)
            sl_distance_price = (current_price - sl_price).copy_abs()
            min_tick_size = self.market_info["tick_size"]
            if sl_distance_price < min_tick_size:
                logger.error(
                    f"SL distance ({sl_distance_price}) is smaller than minimum tick size ({min_tick_size}). Cannot calculate quantity."
                )
                return None

            # Calculate Quantity = Risk Amount / Stop Loss Distance (per unit)
            quantity = risk_amount / sl_distance_price

            # Format quantity and check against minimum order size
            quantity_str = self.exchange_manager.format_amount(
                quantity, rounding_mode=ROUND_DOWN
            )
            quantity_decimal = safe_decimal(quantity_str)
            min_order_size = self.market_info.get("min_order_size", Decimal("NaN"))

            if quantity_decimal.is_nan() or quantity_decimal <= 0:
                logger.error(
                    f"Calculated quantity ({quantity_str}) is invalid or zero."
                )
                return None

            if not min_order_size.is_nan() and quantity_decimal < min_order_size:
                logger.error(
                    f"Calculated quantity {quantity_decimal.normalize()} is below exchange minimum {min_order_size.normalize()}. Cannot place order."
                )
                return None

            # Format SL/TP prices
            sl_price_str = self.exchange_manager.format_price(sl_price)
            tp_price_str = (
                self.exchange_manager.format_price(tp_price)
                if tp_price is not None
                else "0"  # Use '0' string for API if no TP
            )

            # Calculate TSL distance (price difference based on percentage of current price)
            tsl_distance_price = current_price * (
                self.config.trailing_stop_percent / 100
            )
            # Ensure TSL distance is at least one tick
            if tsl_distance_price < min_tick_size:
                tsl_distance_price = min_tick_size
            tsl_distance_str = self.exchange_manager.format_price(tsl_distance_price)

            params = {
                "qty": quantity_decimal,
                "sl_price": safe_decimal(sl_price_str),
                "tp_price": safe_decimal(tp_price_str) if tp_price else None,
                "tsl_distance": safe_decimal(tsl_distance_str),
            }

            logger.info(
                f"Trade Params Calculated: Side={side.upper()}, "
                f"Qty={params['qty'].normalize()}, "
                f"Entry~={current_price.normalize()}, "
                f"SL={params['sl_price'].normalize()}, "
                f"TP={params['tp_price'].normalize() if params['tp_price'] else 'None'}, "
                f"TSLDist~={params['tsl_distance'].normalize()}, "
                f"RiskAmt={risk_amount.normalize():.4f}, ATR={atr.normalize():.4f}"
            )
            return params

        except (InvalidOperation, DivisionByZero, TypeError, Exception) as e:
            logger.error(f"Error calculating trade parameters: {e}", exc_info=True)
            return None

    def _execute_market_order(self, side: str, qty_decimal: Decimal) -> Optional[Dict]:
        """Executes a market order with retries and basic confirmation."""
        if not self.exchange or not self.market_info:
            logger.error("Cannot execute market order: Exchange/Market info missing.")
            return None

        symbol = self.config.symbol
        qty_str = self.exchange_manager.format_amount(qty_decimal)
        if safe_decimal(qty_str) <= 0:
            logger.error(
                f"Attempted market order with zero/invalid quantity: {qty_str}"
            )
            return None

        logger.trade(
            f"{Fore.CYAN}Attempting MARKET {side.upper()} order: {qty_str} {symbol}..."
        )
        try:
            # V5 requires category, positionIdx might be needed depending on mode
            params = {
                "category": self.config.bybit_v5_category,
                # 'positionIdx': self.config.position_idx # Usually not needed for market order itself
            }
            order = fetch_with_retries(
                self.exchange.create_market_order,  # Pass function
                symbol=symbol,
                side=side,
                amount=float(qty_str),  # CCXT often prefers float for amount
                params=params,
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
            )
            order_id = order.get("id", "N/A")
            logger.trade(
                f"{Fore.GREEN}Market order submitted: ID {order_id}, Side {side.upper()}, Qty {qty_str}"
            )
            termux_notify(
                f"{symbol} Order Submitted", f"Market {side.upper()} {qty_str}"
            )

            # Simple delay - more robust check would involve fetching order status
            logger.info(
                f"Waiting {self.config.order_check_delay_seconds}s for order {order_id} to potentially fill/propagate..."
            )
            time.sleep(self.config.order_check_delay_seconds)
            # Consider adding check_order_status logic here if market order fills are slow/unreliable

            return (
                order  # Return submitted order info (may not include fill details yet)
            )
        except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e:
            logger.error(f"{Fore.RED}Order placement failed ({type(e).__name__}): {e}")
            termux_notify(
                f"{symbol} Order Failed",
                f"Market {side.upper()} failed: {str(e)[:50]}",
            )
            return None
        except Exception as e:
            logger.error(
                f"{Fore.RED}Unexpected error placing market order: {e}",
                exc_info=True,
            )
            termux_notify(f"{symbol} Order Error", f"Market {side.upper()} error.")
            return None

    def _set_position_protection(
        self,
        position_side: str,
        sl_price: Optional[Decimal] = None,
        tp_price: Optional[Decimal] = None,
        is_tsl: bool = False,
        tsl_distance: Optional[Decimal] = None,
        tsl_activation_price: Optional[Decimal] = None,
    ) -> bool:
        """Sets SL, TP, or TSL for a position using V5 setTradingStop.
        Ensures numeric parameters are passed as formatted strings.
        """
        if not self.exchange or not self.market_info:
            logger.error("Cannot set protection: Exchange/Market info missing.")
            return False

        symbol = self.config.symbol
        market_id = self.market_info.get("id")
        if not market_id:
            logger.error("Cannot set protection: Market ID missing.")
            return False
        tracker_key = position_side.lower()  # "long" or "short"

        # --- Prepare Base Parameters for privatePostPositionSetTradingStop ---
        params = {
            "category": self.config.bybit_v5_category,
            "symbol": market_id,
            "positionIdx": self.config.position_idx,
            "tpslMode": V5_TPSL_MODE_FULL,  # Apply to the entire position
            # Default to clearing stops unless specific values are provided
            "stopLoss": "0",
            "takeProfit": "0",
            "trailingStop": "0",  # This is the TSL *distance* string
            "activePrice": "0",  # TSL activation price string
            "slTriggerBy": self.config.sl_trigger_by,
            "tpTriggerBy": self.config.sl_trigger_by,  # Usually same trigger for TP
            "tslTriggerBy": self.config.tsl_trigger_by,
        }

        action_desc = ""
        new_tracker_state: Optional[str] = None

        if (
            is_tsl
            and tsl_distance is not None
            and not tsl_distance.is_nan()
            and tsl_distance > 0
            and tsl_activation_price is not None
            and not tsl_activation_price.is_nan()
            and tsl_activation_price > 0
        ):
            # --- Activate Trailing Stop ---
            # Format distance and activation price as strings
            tsl_distance_str = self.exchange_manager.format_price(tsl_distance)
            tsl_activation_price_str = self.exchange_manager.format_price(
                tsl_activation_price
            )

            params["trailingStop"] = tsl_distance_str
            params["activePrice"] = tsl_activation_price_str
            # Clear fixed SL/TP when activating TSL for V5
            params["stopLoss"] = "0"
            params["takeProfit"] = "0"

            action_desc = f"ACTIVATE TSL (Dist: {tsl_distance_str}, ActPx: {tsl_activation_price_str})"
            new_tracker_state = "ACTIVE_TSL"
            logger.debug(
                f"TSL Params: trailingStop={params['trailingStop']}, activePrice={params['activePrice']}"
            )

        elif (sl_price is not None and not sl_price.is_nan()) or (
            tp_price is not None and not tp_price.is_nan()
        ):
            # --- Set Fixed SL/TP ---
            # Format valid SL/TP prices as strings, use '0' if invalid or None
            sl_str = (
                self.exchange_manager.format_price(sl_price)
                if sl_price and not sl_price.is_nan() and sl_price > 0
                else "0"
            )
            tp_str = (
                self.exchange_manager.format_price(tp_price)
                if tp_price and not tp_price.is_nan() and tp_price > 0
                else "0"
            )
            params["stopLoss"] = sl_str
            params["takeProfit"] = tp_str
            # Ensure TSL is cleared if setting fixed stops
            params["trailingStop"] = "0"
            params["activePrice"] = "0"

            if params["stopLoss"] != "0" or params["takeProfit"] != "0":
                action_desc = f"SET SL={params['stopLoss']} TP={params['takeProfit']}"
                new_tracker_state = "ACTIVE_SLTP"
            else:
                # This case means both sl_price and tp_price were invalid/zero
                action_desc = "CLEAR SL/TP (Invalid inputs)"
                new_tracker_state = None  # Clearing stops

        else:
            # --- Clear All Stops ---
            # Explicitly called with no SL/TP/TSL info, or as part of close_position
            action_desc = "CLEAR SL/TP/TSL"
            params["stopLoss"] = "0"
            params["takeProfit"] = "0"
            params["trailingStop"] = "0"
            params["activePrice"] = "0"
            new_tracker_state = None
            logger.debug(f"Clearing all stops for {position_side.upper()}")

        logger.trade(
            f"{Fore.CYAN}Attempting to {action_desc} for {position_side.upper()} {symbol}..."
        )

        try:
            # Use CCXT's private method mapping for V5 setTradingStop
            # Common path: private_post_position_set_trading_stop
            method_name = "private_post_position_set_trading_stop"
            if not hasattr(self.exchange, method_name):
                # Try alternative common naming convention
                method_name = "privatePostPositionSetTradingStop"
                if not hasattr(self.exchange, method_name):
                    raise NotImplementedError(
                        f"CCXT method for V5 SetTradingStop ('{method_name}' or similar) not found in exchange instance."
                    )

            method_to_call = getattr(self.exchange, method_name)
            logger.debug(f"Calling {method_name} with params: {params}")

            response = fetch_with_retries(
                method_to_call,  # Pass the bound method
                params=params,  # Pass params dict
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
            )
            # logger.debug(f"SetTradingStop Response: {response}") # Verbose

            # V5 response check (retCode 0 indicates success)
            if response and response.get("retCode") == V5_SUCCESS_RETCODE:
                logger.trade(
                    f"{Fore.GREEN}{action_desc} successful for {position_side.upper()} {symbol}."
                )
                termux_notify(
                    f"{symbol} Protection Set",
                    f"{action_desc} {position_side.upper()}",
                )
                # Update tracker only on success
                self.protection_tracker[tracker_key] = new_tracker_state
                return True
            else:
                ret_code = response.get("retCode", "N/A")
                ret_msg = response.get("retMsg", "Unknown Error")
                logger.error(
                    f"{Fore.RED}{action_desc} failed for {position_side.upper()} {symbol}. Code: {ret_code}, Msg: {ret_msg}"
                )
                # Do NOT update tracker on failure, keep previous state
                termux_notify(
                    f"{symbol} Protection FAILED",
                    f"{action_desc[:30]} {position_side.upper()} failed: {ret_msg[:50]}",
                )
                return False
        except Exception as e:
            logger.error(
                f"{Fore.RED}Error during {action_desc} for {position_side.upper()} {symbol}: {e}",
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
        side: str,
        atr: Decimal,
        total_equity: Decimal,
        current_price: Decimal,
    ) -> bool:
        """Calculates parameters, places market order, verifies position, and sets initial SL/TP."""
        if not self.exchange or not self.market_info:
            logger.error("Cannot place order: Exchange/Market info missing.")
            return False

        # --- 1. Calculate Trade Parameters ---
        params = self._calculate_trade_parameters(
            side, atr, total_equity, current_price
        )
        if not params:
            logger.error("Failed to calculate trade parameters. Cannot place order.")
            return False

        qty = params["qty"]
        sl_price = params["sl_price"]
        tp_price = params["tp_price"]  # Can be None
        position_side = "long" if side == "buy" else "short"

        # --- 2. Execute Market Order ---
        order_info = self._execute_market_order(side, qty)
        if not order_info:
            logger.error("Market order execution failed. Aborting entry sequence.")
            # No need to handle failure here, as no position should exist yet
            return False
        order_id = order_info.get("id")  # Keep for journaling

        # --- 3. Wait & Verify Position ---
        logger.info(
            f"Waiting {self.config.order_check_delay_seconds + 1}s after market order submission to verify position..."
        )
        time.sleep(self.config.order_check_delay_seconds + 1)

        positions_after_entry = self.exchange_manager.get_current_position()
        if positions_after_entry is None:
            logger.error(
                f"{Fore.RED}Position check FAILED after market order. Manual check required! Attempting cleanup..."
            )
            self._handle_entry_failure(side, qty)  # Attempt cleanup
            return False

        active_pos_data = positions_after_entry.get(position_side)
        if not active_pos_data:
            logger.error(
                f"{Fore.RED}Position {position_side.upper()} not found after market order. Manual check required! Potential partial fill or delay. Attempting cleanup..."
            )
            self._handle_entry_failure(side, qty)
            return False

        filled_qty = safe_decimal(active_pos_data.get("qty", "0"))
        avg_entry_price = safe_decimal(active_pos_data.get("entry_price", "NaN"))

        if filled_qty.copy_abs() < POSITION_QTY_EPSILON:
            logger.error(
                f"{Fore.RED}Position {position_side.upper()} found but quantity is zero/negligible ({filled_qty}). Manual check required! Assuming entry failed."
            )
            self._handle_entry_failure(side, qty)  # Treat as failure
            return False

        logger.info(
            f"Position {position_side.upper()} confirmed: Qty={filled_qty.normalize()}, AvgEntry={avg_entry_price.normalize() if not avg_entry_price.is_nan() else 'N/A'}"
        )

        # Check for significant partial fill (optional, adjust logic if needed)
        if (filled_qty - qty).copy_abs() > POSITION_QTY_EPSILON * Decimal("10"):
            logger.warning(
                f"{Fore.YELLOW}Partial fill detected? Intended: {qty.normalize()}, Filled: {filled_qty.normalize()}. Proceeding with filled amount."
            )
            # For simplicity, we still use initial SL/TP calc based on intended qty.
            # More complex logic could recalculate stops based on filled_qty, but risk profile changes.

        # --- 4. Set Position SL/TP ---
        logger.info(f"Setting initial SL/TP for {position_side.upper()} position...")
        set_stops_ok = self._set_position_protection(
            position_side, sl_price=sl_price, tp_price=tp_price
        )
        if not set_stops_ok:
            logger.error(
                f"{Fore.RED}Failed to set initial SL/TP after entry. Attempting to close position for safety!"
            )
            # Use the filled quantity for closing
            self.close_position(
                position_side, filled_qty, reason="CloseDueToFailedStopSet"
            )
            return False  # Entry sequence failed

        # --- 5. Log to Journal ---
        if self.config.enable_journaling and not avg_entry_price.is_nan():
            # Log using the actual average entry price and filled quantity
            self.log_trade_entry_to_journal(side, filled_qty, avg_entry_price, order_id)

        logger.trade(
            f"{Fore.GREEN}Entry sequence for {position_side.upper()} completed successfully."
        )
        return True

    def manage_trailing_stop(
        self,
        position_side: str,
        entry_price: Decimal,
        current_price: Decimal,
        atr: Decimal,
    ) -> None:
        """Checks if TSL activation conditions are met and activates it using V5 position TSL."""
        if not self.exchange or not self.market_info:
            return
        if atr.is_nan() or atr <= 0:
            logger.debug("Invalid ATR, cannot manage TSL.")
            return
        if entry_price.is_nan() or entry_price <= 0:
            logger.debug("Invalid entry price, cannot manage TSL.")
            return
        if current_price.is_nan() or current_price <= 0:
            logger.debug("Invalid current price, cannot manage TSL.")
            return

        tracker_key = position_side.lower()

        # Only proceed if fixed SL/TP is currently tracked as active
        if self.protection_tracker.get(tracker_key) != "ACTIVE_SLTP":
            if self.protection_tracker.get(tracker_key) == "ACTIVE_TSL":
                logger.debug(
                    f"TSL already marked as active for {position_side.upper()}."
                )
            else:  # Protection is None (cleared or never set)
                logger.debug(
                    f"No active SL/TP tracked for {position_side.upper()}, skipping TSL activation check."
                )
            return

        try:
            # --- Calculate Activation Price ---
            activation_distance_atr = atr * self.config.tsl_activation_atr_multiplier
            activation_price: Decimal = Decimal("NaN")
            if position_side == "long":
                activation_price = entry_price + activation_distance_atr
            elif position_side == "short":
                activation_price = entry_price - activation_distance_atr

            if activation_price.is_nan() or activation_price <= 0:
                logger.warning(
                    f"Calculated invalid TSL activation price ({activation_price}). Skipping TSL check."
                )
                return

            # --- Calculate TSL Distance ---
            tsl_distance_price = current_price * (
                self.config.trailing_stop_percent / 100
            )
            min_tick_size = self.market_info.get(
                "tick_size", Decimal("1e-8")
            )  # Safe default
            if tsl_distance_price < min_tick_size:
                logger.warning(
                    f"Calculated TSL distance ({tsl_distance_price}) too small, using min tick size {min_tick_size}."
                )
                tsl_distance_price = min_tick_size

            # --- Check Activation Condition ---
            should_activate_tsl = False
            if position_side == "long" and current_price >= activation_price:
                should_activate_tsl = True
            elif position_side == "short" and current_price <= activation_price:
                should_activate_tsl = True

            if should_activate_tsl:
                logger.trade(
                    f"{Fore.MAGENTA}TSL Activation condition met for {position_side.upper()}!"
                )
                logger.trade(
                    f"  Entry={entry_price.normalize()}, Current={current_price.normalize()}, Activation Target~={activation_price.normalize()}"
                )
                logger.trade(
                    f"Attempting to activate TSL (Dist ~{tsl_distance_price.normalize()}, ActPx {activation_price.normalize()})..."
                )

                # Activate TSL using _set_position_protection
                # Pass calculated distance and activation price
                if self._set_position_protection(
                    position_side,
                    is_tsl=True,
                    tsl_distance=tsl_distance_price,
                    tsl_activation_price=activation_price,
                ):
                    # Tracker state is updated inside _set_position_protection on success
                    logger.trade(
                        f"{Fore.GREEN}Trailing Stop Loss activated successfully for {position_side.upper()}."
                    )
                else:
                    logger.error(
                        f"{Fore.RED}Failed to activate Trailing Stop Loss for {position_side.upper()}."
                    )
                    # Tracker state remains 'ACTIVE_SLTP' if activation failed

            else:
                logger.debug(
                    f"TSL activation condition not met for {position_side.upper()} (Current: {current_price.normalize()}, Target: ~{activation_price.normalize()})"
                )

        except Exception as e:
            logger.error(
                f"Error managing trailing stop for {position_side.upper()}: {e}",
                exc_info=True,
            )

    def close_position(
        self, position_side: str, qty_to_close: Decimal, reason: str = "Signal"
    ) -> bool:
        """Closes the specified position: clears stops first, then places a market order."""
        if not self.exchange or not self.market_info:
            logger.error("Cannot close position: Exchange/Market info missing.")
            return False

        if qty_to_close.is_nan() or qty_to_close.copy_abs() < POSITION_QTY_EPSILON:
            logger.warning(
                f"Close requested for zero/negligible qty ({qty_to_close}). Skipping."
            )
            return True  # Consider it "closed" as there's nothing to close

        symbol = self.config.symbol
        side_to_close = "sell" if position_side == "long" else "buy"
        tracker_key = position_side.lower()

        logger.trade(
            f"{Fore.YELLOW}Attempting to close {position_side.upper()} position ({qty_to_close.normalize()} {symbol}) Reason: {reason}..."
        )

        # --- 1. Clear existing SL/TP/TSL from the position ---
        logger.info(
            f"Clearing any existing position protection (SL/TP/TSL) for {position_side.upper()} before closing..."
        )
        # Call _set_position_protection with parameters that result in clearing stops
        clear_stops_ok = self._set_position_protection(
            position_side, sl_price=None, tp_price=None, is_tsl=False
        )
        if not clear_stops_ok:
            logger.warning(
                f"{Fore.YELLOW}Failed to confirm clearing of position protection via API. Proceeding with close order anyway..."
            )
        else:
            logger.info("Position protection cleared successfully via API.")
        # Ensure tracker is cleared locally regardless of API success for safety
        self.protection_tracker[tracker_key] = None

        # --- 2. Place Closing Market Order ---
        logger.info(
            f"Submitting MARKET {side_to_close.upper()} order to close {position_side.upper()} position..."
        )
        close_order_info = self._execute_market_order(side_to_close, qty_to_close)

        if not close_order_info:
            logger.error(
                f"{Fore.RED}Failed to submit closing market order for {position_side.upper()}. MANUAL INTERVENTION REQUIRED!"
            )
            termux_notify(
                f"{symbol} CLOSE FAILED",
                f"Market {side_to_close.upper()} order failed!",
            )
            return False  # Close command failed

        close_order_id = close_order_info.get("id", "N/A")
        # Try to get avg price from order response, might be None/NaN initially
        avg_close_price = safe_decimal(close_order_info.get("average", "NaN"))
        logger.trade(
            f"{Fore.GREEN}Closing market order ({close_order_id}) submitted for {position_side.upper()}."
        )
        termux_notify(
            f"{symbol} Position Closing",
            f"{position_side.upper()} close order submitted.",
        )

        # --- 3. Verify Position Closed (Optional but recommended) ---
        logger.info("Waiting briefly and verifying position closure...")
        time.sleep(self.config.order_check_delay_seconds + 2)  # Extra delay

        final_positions = self.exchange_manager.get_current_position()
        is_confirmed_closed = False
        if final_positions is not None:
            final_pos_data = final_positions.get(position_side)
            if (
                not final_pos_data
                or safe_decimal(final_pos_data.get("qty", "0")).copy_abs()
                < POSITION_QTY_EPSILON
            ):
                logger.trade(
                    f"{Fore.GREEN}Position {position_side.upper()} confirmed closed via API check."
                )
                is_confirmed_closed = True
            else:
                lingering_qty = safe_decimal(final_pos_data.get("qty", "NaN"))
                logger.error(
                    f"{Fore.RED}Position {position_side.upper()} still shows qty {lingering_qty.normalize()} after close attempt. MANUAL CHECK REQUIRED!"
                )
                termux_notify(
                    f"{symbol} CLOSE VERIFY FAILED",
                    f"{position_side.upper()} still has position!",
                )
                # Return False because verification failed, even if order was sent
                is_confirmed_closed = False
        else:
            # Failed to fetch final positions for verification
            logger.warning(
                f"{Fore.YELLOW}Could not verify position closure for {position_side.upper()} (failed position fetch). Assuming closed based on order submission."
            )
            # Treat as success if order submitted but verification failed, log appropriately
            is_confirmed_closed = True

        # --- 4. Log Exit to Journal ---
        if is_confirmed_closed and self.config.enable_journaling:
            # Use the average price from the order if available, otherwise log as N/A
            self.log_trade_exit_to_journal(
                position_side,
                qty_to_close,
                avg_close_price,  # Might be NaN
                close_order_id,
                reason,
            )

        return is_confirmed_closed

    def _handle_entry_failure(self, failed_entry_side: str, attempted_qty: Decimal):
        """Attempts to close any potentially opened position after a failed entry sequence step."""
        logger.warning(
            f"{Fore.YELLOW}Handling potential entry failure for {failed_entry_side.upper()} (intended qty: {attempted_qty.normalize()}). Checking for lingering position..."
        )
        position_side = "long" if failed_entry_side == "buy" else "short"

        time.sleep(2)  # Short delay before checking
        positions = self.exchange_manager.get_current_position()

        if positions and positions.get(position_side):
            current_qty = safe_decimal(positions[position_side].get("qty", "0"))
            if current_qty.copy_abs() >= POSITION_QTY_EPSILON:
                logger.error(
                    f"{Fore.RED}Detected lingering {position_side.upper()} position (Qty: {current_qty.normalize()}) after entry failure. Attempting emergency close."
                )
                close_success = self.close_position(
                    position_side, current_qty, reason="EmergencyCloseEntryFail"
                )
                if close_success:
                    logger.info("Emergency close order submitted/confirmed.")
                else:
                    logger.critical(
                        f"{Fore.RED + Style.BRIGHT}EMERGENCY CLOSE FAILED for {position_side.upper()}. MANUAL INTERVENTION URGENT!"
                    )
            else:
                logger.info(
                    "Lingering position found but quantity negligible. No emergency close needed."
                )
        elif positions is None:
            logger.error(
                "Could not fetch positions during entry failure handling. Manual check advised."
            )
        else:
            logger.info("No lingering position detected after entry failure.")

    def _write_journal_row(self, data: Dict[str, Any]):
        """Helper function to write a row to the CSV journal."""
        if not self.config.enable_journaling:
            return
        file_path = self.config.journal_file_path
        file_exists = os.path.isfile(file_path)
        try:
            with open(file_path, "a", newline="", encoding="utf-8") as csvfile:
                # Define fieldnames in order
                fieldnames = [
                    "TimestampUTC",
                    "Symbol",
                    "Action",
                    "Side",
                    "Quantity",
                    "AvgPrice",
                    "OrderID",
                    "Reason",
                    "Notes",
                ]
                writer = csv.DictWriter(
                    csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL
                )
                # Write header only if file is new or empty
                if not file_exists or os.path.getsize(file_path) == 0:
                    writer.writeheader()
                # Ensure all fields are present, defaulting to 'N/A' or empty string
                row_to_write = {
                    field: data.get(field, "" if field == "Notes" else "N/A")
                    for field in fieldnames
                }
                writer.writerow(row_to_write)
            logger.debug(
                f"Trade {data.get('Action', '').lower()} logged to {file_path}"
            )
        except IOError as e:
            logger.error(
                f"Failed to write {data.get('Action', '').lower()} to journal {file_path}: {e}"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error writing {data.get('Action', '').lower()} to journal: {e}",
                exc_info=True,
            )

    def log_trade_entry_to_journal(
        self,
        side: str,
        qty: Decimal,
        avg_price: Decimal,
        order_id: Optional[str],
    ):
        """Logs trade entry details to the CSV journal."""
        data = {
            "TimestampUTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "Symbol": self.config.symbol,
            "Action": "ENTRY",
            "Side": side.upper(),
            "Quantity": qty.normalize(),
            "AvgPrice": avg_price.normalize() if not avg_price.is_nan() else "N/A",
            "OrderID": order_id or "N/A",
            "Reason": "Strategy Signal",  # Can be enhanced later
        }
        self._write_journal_row(data)

    def log_trade_exit_to_journal(
        self,
        side: str,
        qty: Decimal,
        avg_price: Decimal,
        order_id: Optional[str],
        reason: str,
    ):
        """Logs trade exit details to the CSV journal."""
        data = {
            "TimestampUTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "Symbol": self.config.symbol,
            "Action": "EXIT",
            "Side": side.upper(),  # Side of the position being exited
            "Quantity": qty.normalize(),
            "AvgPrice": avg_price.normalize() if not avg_price.is_nan() else "N/A",
            "OrderID": order_id or "N/A",
            "Reason": reason,
        }
        self._write_journal_row(data)


# --- Status Display Class ---
class StatusDisplay:
    """Handles displaying the bot status using the Rich library."""

    def __init__(self, config: TradingConfig):
        self.config = config

    def _format_decimal(
        self,
        value: Optional[Decimal],
        precision: int = 2,
        add_commas: bool = True,
    ) -> str:
        """Formats Decimal values for display, handling None and NaN."""
        if value is None or value.is_nan():
            return "[dim]N/A[/]"
        # Use normalize() to remove trailing zeros before formatting
        normalized_value = value.normalize()
        format_str = f"{{:{',' if add_commas else ''}.{precision}f}}"
        try:
            # Format the normalized value
            return format_str.format(normalized_value)
        except (ValueError, TypeError):
            # Fallback for unusual Decimal states
            return "[dim]ERR[/]"

    def print_status_panel(
        self,
        cycle: int,
        timestamp: Optional[datetime],
        price: Optional[Decimal],
        indicators: Optional[Dict],
        positions: Optional[Dict],
        equity: Optional[Decimal],
        signals: Dict,
        protection_tracker: Dict,
        market_info: Optional[Dict],
    ):
        """Prints the status panel to the console using Rich Panel."""
        # Get precision details safely from market_info or use defaults
        price_dp = (
            market_info.get("precision_dp", {}).get("price", DEFAULT_PRICE_DP)
            if market_info
            else DEFAULT_PRICE_DP
        )
        amount_dp = (
            market_info.get("precision_dp", {}).get("amount", DEFAULT_AMOUNT_DP)
            if market_info
            else DEFAULT_AMOUNT_DP
        )

        panel_content = Text()
        ts_str = (
            timestamp.strftime("%Y-%m-%d %H:%M:%S %Z") if timestamp else "Timestamp N/A"
        )
        title = f" Cycle {cycle} | {self.config.symbol} ({self.config.interval}) | {ts_str} "

        # --- Price & Equity ---
        price_str = self._format_decimal(price, precision=price_dp)
        equity_str = self._format_decimal(equity, precision=2)
        # Get settle currency from symbol (e.g., USDT from BTC/USDT:USDT)
        settle_curr = self.config.symbol.split(":")[-1]
        panel_content.append("Price: ", style="bold cyan")
        panel_content.append(f"{price_str} | ", style="white")
        panel_content.append("Equity: ", style="bold cyan")
        panel_content.append(f"{equity_str} {settle_curr}\n", style="white")
        panel_content.append("---\n", style="dim")

        # --- Indicators ---
        panel_content.append("Indicators: ", style="bold cyan")
        if indicators:
            ind_parts = []

            def fmt_ind(key, prec=1, commas=False):
                return self._format_decimal(
                    indicators.get(key), precision=prec, add_commas=commas
                )

            ind_parts.append(
                f"EMA(F/S/T): {fmt_ind('fast_ema')}/{fmt_ind('slow_ema')}/{fmt_ind('trend_ema')}"
            )
            stoch_k_str = fmt_ind("stoch_k")
            stoch_d_str = fmt_ind("stoch_d")
            stoch_sig = ""
            if indicators.get("stoch_kd_bullish"):
                stoch_sig = "[bold green] BULL[/]"
            elif indicators.get("stoch_kd_bearish"):
                stoch_sig = "[bold red] BEAR[/]"
            ind_parts.append(f"Stoch(K/D): {stoch_k_str}/{stoch_d_str}{stoch_sig}")
            ind_parts.append(
                f"ATR({indicators.get('atr_period', '?')}): {fmt_ind('atr', 4)}"
            )
            adx_str = fmt_ind("adx")
            pdi_str = fmt_ind("pdi")
            mdi_str = fmt_ind("mdi")
            ind_parts.append(
                f"ADX({self.config.adx_period}): {adx_str} [+DI:{pdi_str} -DI:{mdi_str}]"
            )
            panel_content.append(" | ".join(ind_parts) + "\n", style="white")
        else:
            panel_content.append("Calculating...\n", style="dim")
        panel_content.append("---\n", style="dim")

        # --- Positions ---
        panel_content.append("Position: ", style="bold cyan")
        pos_text = Text("FLAT", style="bold green")
        long_pos = positions.get("long") if positions else None
        short_pos = positions.get("short") if positions else None

        position_data = None
        position_side_str = ""
        pos_style = "bold green"  # Default for FLAT

        if (
            long_pos
            and safe_decimal(long_pos.get("qty", "0")).copy_abs()
            >= POSITION_QTY_EPSILON
        ):
            position_data = long_pos
            position_side_str = "long"
            pos_style = "bold green"
        elif (
            short_pos
            and safe_decimal(short_pos.get("qty", "0")).copy_abs()
            >= POSITION_QTY_EPSILON
        ):
            position_data = short_pos
            position_side_str = "short"
            pos_style = "bold red"

        if position_data:
            qty = self._format_decimal(
                position_data.get("qty"), precision=amount_dp, add_commas=False
            )
            entry = self._format_decimal(
                position_data.get("entry_price"), precision=price_dp
            )
            pnl = self._format_decimal(position_data.get("unrealized_pnl"), precision=4)
            sl = self._format_decimal(
                position_data.get("stop_loss_price"), precision=price_dp
            )
            tp = self._format_decimal(
                position_data.get("take_profit_price"), precision=price_dp
            )
            active_protection = protection_tracker.get(position_side_str)
            prot_str = (
                Text(active_protection, style="magenta")
                if active_protection
                else Text("None", style="dim")
            )

            pos_text = Text()
            pos_text.append(f"{position_side_str.upper()}: ", style=pos_style)
            pos_text.append(
                f"Qty={qty} | Entry={entry} | PnL={pnl} | Prot: ", style="white"
            )
            pos_text.append(prot_str)
            pos_text.append(f" (SL:{sl} TP:{tp})", style="dim")

        panel_content.append(pos_text)
        panel_content.append("\n")
        panel_content.append("---\n", style="dim")

        # --- Signals ---
        panel_content.append("Signal: ", style="bold cyan")
        sig_reason = signals.get("reason", "No signal info")
        sig_style = "dim"  # Default style
        if signals.get("long"):
            sig_style = "bold green"
        elif signals.get("short"):
            sig_style = "bold red"
        elif "Blocked" in sig_reason:
            sig_style = "yellow"
        elif "Signal:" in sig_reason:  # Explicit signal messages
            sig_style = "white"

        panel_content.append(Text(sig_reason, style=sig_style))

        # Print Panel
        console.print(
            Panel(
                panel_content,
                title=f"[bold bright_magenta]{title}[/]",
                border_style="bright_blue",
                expand=False,  # Prevent panel from expanding unnecessarily
            )
        )


# --- Trading Bot Class ---
class TradingBot:
    """Main orchestrator for the trading bot."""

    def __init__(self):
        logger.info(
            f"{Fore.MAGENTA + Style.BRIGHT}--- Initializing Pyrmethus v2.4.1 (Enhanced) ---"
        )
        self.config = TradingConfig()
        self.exchange_manager = ExchangeManager(self.config)
        # Critical check: Ensure exchange init succeeded and market_info is loaded
        if not self.exchange_manager.exchange or not self.exchange_manager.market_info:
            logger.critical(
                "Exchange Manager failed to initialize properly (exchange or market_info missing). Halting."
            )
            sys.exit(1)
        self.indicator_calculator = IndicatorCalculator(self.config)
        self.signal_generator = SignalGenerator(self.config)
        self.order_manager = OrderManager(self.config, self.exchange_manager)
        self.status_display = StatusDisplay(self.config)
        self.shutdown_requested = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Sets up OS signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)  # Handle Ctrl+C
        signal.signal(
            signal.SIGTERM, self._signal_handler
        )  # Handle termination signals

    def _signal_handler(self, sig, frame):
        """Internal signal handler to initiate graceful shutdown."""
        if not self.shutdown_requested:
            sig_name = signal.Signals(sig).name if isinstance(sig, int) else str(sig)
            logger.warning(
                f"{Fore.YELLOW + Style.BRIGHT}\nSignal {sig_name} received. Initiating graceful shutdown..."
            )
            self.shutdown_requested = True
        else:
            # Avoid multiple shutdown attempts if signal received again
            logger.warning("Shutdown already in progress.")

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
            try:
                self.trading_spell_cycle(cycle_count)

            except KeyboardInterrupt:
                logger.warning(
                    "\nCtrl+C detected during main loop execution. Initiating shutdown."
                )
                self.shutdown_requested = True
                break  # Exit loop immediately
            except ccxt.AuthenticationError as e:
                logger.critical(
                    f"{Fore.RED + Style.BRIGHT}CRITICAL AUTH ERROR in cycle {cycle_count}: {e}. Halting immediately.",
                    exc_info=False,
                )
                self.shutdown_requested = True  # Ensure shutdown sequence runs
                break
            except Exception as e:
                # Catch-all for unexpected errors within a cycle
                logger.error(
                    f"{Fore.RED + Style.BRIGHT}Unhandled exception in main loop (Cycle {cycle_count}): {e}",
                    exc_info=True,
                )
                logger.error(
                    f"{Fore.RED}Continuing loop, but caution advised. Check logs."
                )
                termux_notify(
                    "Pyrmethus Error",
                    f"Unhandled exception cycle {cycle_count}.",
                )
                # Sleep longer after an unexpected error
                sleep_time = self.config.loop_sleep_seconds * 2
            else:
                # Calculate remaining sleep time based on cycle duration
                cycle_duration = time.monotonic() - cycle_start_time
                sleep_time = max(0, self.config.loop_sleep_seconds - cycle_duration)

            if self.shutdown_requested:
                logger.info("Shutdown requested, breaking main loop.")
                break

            # Interruptible Sleep
            if sleep_time > 0:
                logger.debug(
                    f"Cycle {cycle_count} finished. Sleeping for {sleep_time:.2f} seconds..."
                )
                sleep_end_time = time.monotonic() + sleep_time
                try:
                    while (
                        time.monotonic() < sleep_end_time
                        and not self.shutdown_requested
                    ):
                        # Sleep in short intervals to check shutdown flag frequently
                        time.sleep(0.2)
                except KeyboardInterrupt:
                    logger.warning("\nCtrl+C detected during sleep.")
                    self.shutdown_requested = True
                    break  # Exit sleep and loop

        # --- Graceful Shutdown ---
        # This block executes after the loop breaks (normally or via signal/error)
        self.graceful_shutdown()
        console.print("\n[bold bright_cyan]Pyrmethus has returned to the ether.[/]")
        sys.exit(0)

    def trading_spell_cycle(self, cycle_count: int) -> None:
        """Executes one cycle of the trading logic: fetch, analyze, act, display."""
        logger.info(
            f"{Fore.MAGENTA + Style.BRIGHT}\n--- Starting Cycle {cycle_count} ---"
        )
        start_time = time.monotonic()
        cycle_status = "OK"  # Track cycle status for logging

        # --- 1. Fetch Data & Basic Info ---
        df = self.exchange_manager.fetch_ohlcv()
        if df is None or df.empty:
            logger.error(f"{Fore.RED}Cycle Failed: Market data fetch failed.")
            cycle_status = "FAIL_FETCH_DATA"
            # Display minimal status if possible
            self.status_display.print_status_panel(
                cycle_count,
                None,
                None,
                None,
                None,
                None,
                {"reason": cycle_status},
                {},
                self.exchange_manager.market_info,
            )
            end_time = time.monotonic()
            logger.info(
                f"{Fore.MAGENTA}--- Cycle {cycle_count} Status: {cycle_status} (Duration: {end_time - start_time:.2f}s) ---"
            )
            return  # Skip rest of cycle

        try:
            last_candle = df.iloc[-1]
            current_price = safe_decimal(last_candle["close"])
            last_timestamp = df.index[
                -1
            ].to_pydatetime()  # Convert to standard datetime
            if current_price.is_nan() or current_price <= 0:
                raise ValueError(f"Invalid latest close price: {current_price}")
            logger.debug(
                f"Latest Candle: {last_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}, Close={current_price.normalize()}"
            )
        except (IndexError, KeyError, ValueError, InvalidOperation, TypeError) as e:
            logger.error(
                f"{Fore.RED}Cycle Failed: Error processing latest candle data: {e}",
                exc_info=False,
            )
            cycle_status = "FAIL_PRICE_PROC"
            self.status_display.print_status_panel(
                cycle_count,
                None,
                None,
                None,
                None,
                None,
                {"reason": cycle_status},
                {},
                self.exchange_manager.market_info,
            )
            end_time = time.monotonic()
            logger.info(
                f"{Fore.MAGENTA}--- Cycle {cycle_count} Status: {cycle_status} (Duration: {end_time - start_time:.2f}s) ---"
            )
            return

        # --- 2. Calculate Indicators ---
        indicators = self.indicator_calculator.calculate_indicators(df)
        if indicators is None:
            logger.warning(
                f"{Fore.YELLOW}Indicator calculation failed. Trading logic may be impacted."
            )
            cycle_status = "WARN_INDICATORS"
            # Continue cycle if possible, but signals might be unreliable
        current_atr = (
            indicators.get("atr", Decimal("NaN")) if indicators else Decimal("NaN")
        )

        # --- 3. Get Current State (Balance & Positions) ---
        total_equity, _ = self.exchange_manager.get_balance()
        positions = self.exchange_manager.get_current_position()

        can_run_trade_logic = True
        if total_equity is None or total_equity.is_nan() or total_equity <= 0:
            logger.error(
                f"{Fore.RED}Failed fetching valid equity ({total_equity}). Trading logic skipped."
            )
            cycle_status = "FAIL_EQUITY"
            can_run_trade_logic = False
        if positions is None:
            logger.error(f"{Fore.RED}Failed fetching positions. Trading logic skipped.")
            cycle_status = "FAIL_POSITIONS"
            can_run_trade_logic = False

        # --- Prepare Data for Logic & Display ---
        # Use snapshots for display to reflect state at display time
        protection_tracker_snapshot = copy.deepcopy(
            self.order_manager.protection_tracker
        )
        # Use a safe default if positions fetch failed
        current_positions = (
            positions if positions is not None else {"long": {}, "short": {}}
        )
        # This variable will hold the positions used for the final display panel
        final_positions_for_panel = copy.deepcopy(current_positions)
        # Default signal state if logic is skipped
        signals: Dict[str, Union[bool, str]] = {
            "long": False,
            "short": False,
            "reason": f"Skipped: {cycle_status}",
        }

        # --- 4. Execute Core Trading Logic (if possible) ---
        if can_run_trade_logic:
            # Extract current position state
            active_long_pos = current_positions.get("long", {})
            active_short_pos = current_positions.get("short", {})
            long_qty = safe_decimal(active_long_pos.get("qty", "0"))
            short_qty = safe_decimal(active_short_pos.get("qty", "0"))
            long_entry = safe_decimal(active_long_pos.get("entry_price", "NaN"))
            short_entry = safe_decimal(active_short_pos.get("entry_price", "NaN"))

            has_long_pos = long_qty.copy_abs() >= POSITION_QTY_EPSILON
            has_short_pos = short_qty.copy_abs() >= POSITION_QTY_EPSILON
            is_flat = not has_long_pos and not has_short_pos
            current_pos_side = (
                "long" if has_long_pos else "short" if has_short_pos else None
            )

            logger.debug(
                f"Current State: Flat={is_flat}, LongQty={long_qty.normalize()}, ShortQty={short_qty.normalize()}"
            )

            # --- 4a. Manage Trailing Stops (if in position) ---
            if current_pos_side and indicators:
                entry_price = long_entry if current_pos_side == "long" else short_entry
                self.order_manager.manage_trailing_stop(
                    current_pos_side, entry_price, current_price, current_atr
                )
                # Update snapshot after potential TSL activation attempt
                protection_tracker_snapshot = copy.deepcopy(
                    self.order_manager.protection_tracker
                )

            # --- 4b. Re-fetch position state AFTER TSL check ---
            # Important: TSL activation or a stop being hit could change the position state
            logger.debug("Re-fetching position state after TSL management...")
            positions_after_tsl = self.exchange_manager.get_current_position()
            if positions_after_tsl is None:
                logger.error(
                    f"{Fore.RED}Failed re-fetching positions after TSL check. Using previous state, but it might be stale."
                )
                cycle_status = "WARN_POS_REFETCH_TSL"
                # Keep using the potentially stale 'current_positions'
            else:
                # Update live state variables based on the re-fetched data
                current_positions = positions_after_tsl
                final_positions_for_panel = copy.deepcopy(
                    current_positions
                )  # Update panel data
                active_long_pos = current_positions.get("long", {})
                active_short_pos = current_positions.get("short", {})
                long_qty = safe_decimal(active_long_pos.get("qty", "0"))
                short_qty = safe_decimal(active_short_pos.get("qty", "0"))
                has_long_pos = long_qty.copy_abs() >= POSITION_QTY_EPSILON
                has_short_pos = short_qty.copy_abs() >= POSITION_QTY_EPSILON
                is_flat = not has_long_pos and not has_short_pos
                current_pos_side = (
                    "long" if has_long_pos else "short" if has_short_pos else None
                )
                logger.debug(
                    f"State After TSL Check: Flat={is_flat}, Long={long_qty.normalize()}, Short={short_qty.normalize()}"
                )

                # If position became flat (e.g., SL/TSL hit), ensure tracker is cleared
                if is_flat and any(self.order_manager.protection_tracker.values()):
                    logger.debug(
                        "Position became flat after TSL logic/check, clearing protection tracker."
                    )
                    self.order_manager.protection_tracker = {
                        "long": None,
                        "short": None,
                    }
                    protection_tracker_snapshot = copy.deepcopy(
                        self.order_manager.protection_tracker
                    )

            # --- 4c. Generate Trading Signals (Entry) ---
            can_gen_signals = (
                indicators is not None and not current_price.is_nan() and len(df) >= 2
            )
            if can_gen_signals:
                signals = self.signal_generator.generate_signals(
                    df.iloc[-2:], indicators
                )
            else:
                reason = "Skipped Signal Gen: " + (
                    "Indicators missing"
                    if indicators is None
                    else f"Need >=2 candles ({len(df)} found)"
                )
                signals["reason"] = reason
                logger.warning(reason)

            # --- 4d. Check for Signal-Based Exits (if in position) ---
            exit_triggered = False
            exit_side = None
            if can_gen_signals and current_pos_side:
                exit_reason = self.signal_generator.check_exit_signals(
                    current_pos_side, indicators
                )
                if exit_reason:
                    exit_side = current_pos_side
                    qty_to_close = long_qty if exit_side == "long" else short_qty
                    # Attempt to close the position
                    close_success = self.order_manager.close_position(
                        exit_side, qty_to_close, reason=exit_reason
                    )
                    exit_triggered = (
                        close_success  # Track if close command was successful
                    )
                    if not exit_triggered:
                        cycle_status = (
                            "FAIL_EXIT_CLOSE"  # Mark failure if close command failed
                        )
                        logger.error(
                            f"Failed to execute closing order for {exit_side} based on exit signal."
                        )
                    else:
                        logger.info(f"Signal-based exit for {exit_side} initiated.")

            # --- 4e. Re-fetch state AGAIN if an exit was triggered/attempted ---
            if exit_triggered:
                logger.debug(
                    f"Re-fetching state after signal exit attempt ({exit_side})..."
                )
                positions_after_exit = self.exchange_manager.get_current_position()
                if positions_after_exit is None:
                    logger.error(
                        f"{Fore.RED}Failed re-fetching positions after signal exit. State may be inaccurate."
                    )
                    cycle_status = "WARN_POS_REFETCH_EXIT"
                    # Keep using previous state for panel, though potentially inaccurate
                else:
                    # Update live state variables
                    current_positions = positions_after_exit
                    final_positions_for_panel = copy.deepcopy(
                        current_positions
                    )  # Update panel data
                    active_long_pos = current_positions.get("long", {})
                    active_short_pos = current_positions.get("short", {})
                    long_qty = safe_decimal(active_long_pos.get("qty", "0"))
                    short_qty = safe_decimal(active_short_pos.get("qty", "0"))
                    has_long_pos = long_qty.copy_abs() >= POSITION_QTY_EPSILON
                    has_short_pos = short_qty.copy_abs() >= POSITION_QTY_EPSILON
                    is_flat = not has_long_pos and not has_short_pos
                    current_pos_side = (
                        "long" if has_long_pos else "short" if has_short_pos else None
                    )
                    logger.debug(
                        f"State After Signal Exit: Flat={is_flat}, Long={long_qty.normalize()}, Short={short_qty.normalize()}"
                    )
                    # Ensure tracker is clear if now flat
                    if is_flat:
                        logger.debug(
                            "Position became flat after signal exit, ensuring protection tracker is clear."
                        )
                        self.order_manager.protection_tracker = {
                            "long": None,
                            "short": None,
                        }
                        protection_tracker_snapshot = copy.deepcopy(
                            self.order_manager.protection_tracker
                        )

            # --- 4f. Execute Entry Trades (Only if currently flat and entry signal exists) ---
            if (
                is_flat
                and can_gen_signals
                and (signals.get("long") or signals.get("short"))
            ):
                if current_atr.is_nan() or current_atr <= 0:
                    logger.warning(f"Cannot enter trade: Invalid ATR ({current_atr})")
                elif total_equity <= 0:  # Re-check equity just before entry
                    logger.warning(
                        f"Cannot enter trade: Invalid Equity ({total_equity})"
                    )
                else:
                    entry_side = "buy" if signals.get("long") else "sell"
                    signal_reason = signals.get("reason", "")
                    log_color = Fore.GREEN if entry_side == "buy" else Fore.RED
                    logger.info(
                        f"{log_color + Style.BRIGHT}{entry_side.upper()} signal detected! {signal_reason}. Attempting entry..."
                    )
                    # Attempt the full entry sequence (order + stops)
                    entry_successful = self.order_manager.place_risked_market_order(
                        entry_side, current_atr, total_equity, current_price
                    )
                    if entry_successful:
                        logger.info(
                            f"{Fore.GREEN}Entry process completed successfully."
                        )
                        # Re-fetch state one last time for accurate panel display after entry
                        logger.debug(
                            "Re-fetching state after successful entry attempt..."
                        )
                        positions_after_entry = (
                            self.exchange_manager.get_current_position()
                        )
                        if positions_after_entry is not None:
                            final_positions_for_panel = copy.deepcopy(
                                positions_after_entry
                            )
                            protection_tracker_snapshot = copy.deepcopy(
                                self.order_manager.protection_tracker
                            )
                        else:
                            logger.warning(
                                "Failed re-fetching positions after entry. Panel may be slightly stale."
                            )
                            cycle_status = "WARN_POS_REFETCH_ENTRY"
                    else:
                        logger.error(f"{Fore.RED}Entry process failed.")
                        cycle_status = "FAIL_ENTRY"
                        # State should ideally be flat again after failed entry handling,
                        # but re-fetch just in case for panel accuracy.
                        logger.debug("Re-fetching state after failed entry attempt...")
                        positions_after_failed_entry = (
                            self.exchange_manager.get_current_position()
                        )
                        if positions_after_failed_entry is not None:
                            final_positions_for_panel = copy.deepcopy(
                                positions_after_failed_entry
                            )
                            # Ensure tracker is clear if entry failed and we are flat
                            if not (
                                final_positions_for_panel.get("long")
                                or final_positions_for_panel.get("short")
                            ):
                                self.order_manager.protection_tracker = {
                                    "long": None,
                                    "short": None,
                                }
                            protection_tracker_snapshot = copy.deepcopy(
                                self.order_manager.protection_tracker
                            )
                        else:
                            logger.warning(
                                "Failed re-fetching positions after failed entry."
                            )

            elif is_flat:
                logger.debug("Position flat, no entry signal generated.")
            elif current_pos_side:
                logger.debug(
                    f"Position ({current_pos_side.upper()}) remains open, skipping entry logic."
                )

        # --- 5. Display Status Panel ---
        # Use the final state variables gathered throughout the cycle
        self.status_display.print_status_panel(
            cycle_count,
            last_timestamp,
            current_price,
            indicators,
            final_positions_for_panel,  # Use the most up-to-date position info
            total_equity,
            signals,
            protection_tracker_snapshot,  # Use the snapshot reflecting state after actions
            self.exchange_manager.market_info,
        )

        end_time = time.monotonic()
        logger.info(
            f"{Fore.MAGENTA}--- Cycle {cycle_count} Status: {cycle_status} (Duration: {end_time - start_time:.2f}s) ---"
        )

    def graceful_shutdown(self) -> None:
        """Handles cleaning up orders and positions before exiting."""
        logger.warning(
            f"{Fore.YELLOW + Style.BRIGHT}\nInitiating Graceful Shutdown Sequence..."
        )
        termux_notify("Pyrmethus Shutdown", f"Closing {self.config.symbol}...")

        if (
            not self.exchange_manager
            or not self.exchange_manager.exchange
            or not self.exchange_manager.market_info
        ):
            logger.error(
                f"{Fore.RED}Cannot perform graceful shutdown: Exchange Manager or Market Info missing."
            )
            termux_notify(
                "Shutdown Warning!",
                f"{self.config.symbol} Cannot shutdown cleanly.",
            )
            return

        exchange = self.exchange_manager.exchange
        symbol = self.config.symbol

        # --- 1. Cancel All Open Orders (Active Limit/Stop Orders) ---
        # Note: V5 Position SL/TP are not typically cancelled this way.
        # They are managed via setTradingStop (handled by close_position).
        logger.info(
            f"{Fore.CYAN}Attempting to cancel all active non-positional orders for {symbol} using V5 params..."
        )
        try:
            # Use generic cancel_all_orders but provide V5 category
            params = {"category": self.config.bybit_v5_category}
            # Set a short timeout/retry for cancellation during shutdown
            cancel_resp = fetch_with_retries(
                exchange.cancel_all_orders,
                symbol=symbol,
                params=params,
                max_retries=1,
                delay_seconds=1,
            )
            logger.info(f"Cancel all active orders response: {cancel_resp}")
            # Response format varies; log if it looks like a list of cancelled orders
            if isinstance(cancel_resp, list):
                logger.info(f"Cancelled {len(cancel_resp)} active orders.")
            elif (
                isinstance(cancel_resp, dict)
                and cancel_resp.get("retCode") == V5_SUCCESS_RETCODE
            ):
                logger.info(
                    "Cancel all orders command successful (check 'list' in response if needed)."
                )
            else:
                logger.warning(
                    "Cancel all orders response format unexpected or indicated failure."
                )

        except NotImplementedError:
            logger.warning(
                "cancel_all_orders might not be fully implemented for V5 category, skipping."
            )
        except Exception as e:
            logger.error(
                f"{Fore.RED}Error cancelling active orders: {e}",
                exc_info=False,  # Less verbose during shutdown
            )

        # Clear local protection tracker regardless of cancellation success
        logger.info("Clearing local protection tracker state.")
        self.order_manager.protection_tracker = {"long": None, "short": None}

        logger.info("Waiting briefly after order cancellation attempt...")
        time.sleep(max(self.config.order_check_delay_seconds, 2))

        # --- 2. Close Any Lingering Positions ---
        logger.info(f"{Fore.CYAN}Checking for lingering positions to close...")
        closed_count = 0
        positions_to_close: Dict[str, Dict] = {}
        try:
            positions = self.exchange_manager.get_current_position()
            if positions is not None:
                for side, pos_data in positions.items():
                    if (
                        pos_data
                        and safe_decimal(pos_data.get("qty", "0")).copy_abs()
                        >= POSITION_QTY_EPSILON
                    ):
                        positions_to_close[side] = pos_data

                if not positions_to_close:
                    logger.info(
                        f"{Fore.GREEN}No significant positions found requiring closure."
                    )
                else:
                    logger.warning(
                        f"{Fore.YELLOW}Found {len(positions_to_close)} positions requiring closure: {list(positions_to_close.keys())}"
                    )
                    for side, pos_data in positions_to_close.items():
                        qty = safe_decimal(pos_data.get("qty", "0.0"))
                        logger.warning(
                            f"Attempting to close {side.upper()} position (Qty: {qty.normalize()})..."
                        )
                        # Use the OrderManager's close_position method
                        # This handles clearing V5 stops first, then market closing.
                        if self.order_manager.close_position(
                            side, qty, reason="GracefulShutdown"
                        ):
                            closed_count += 1
                            logger.info(
                                f"{Fore.GREEN}Closure initiated/confirmed for {side.upper()}."
                            )
                        else:
                            logger.error(
                                f"{Fore.RED}Closure failed for {side.upper()}. MANUAL INTERVENTION REQUIRED."
                            )

                    if closed_count == len(positions_to_close):
                        logger.info(
                            f"{Fore.GREEN}All detected positions closed successfully or closure initiated."
                        )
                    else:
                        logger.warning(
                            f"{Fore.YELLOW}Attempted {len(positions_to_close)} closures, {closed_count} closure orders submitted/confirmed. MANUAL VERIFICATION REQUIRED."
                        )
            else:
                logger.error(
                    f"{Fore.RED}Failed fetching positions during shutdown check. MANUAL CHECK REQUIRED."
                )
        except Exception as e:
            logger.error(
                f"{Fore.RED + Style.BRIGHT}Error during position closure phase: {e}. MANUAL CHECK REQUIRED.",
                exc_info=True,
            )

        logger.warning(
            f"{Fore.YELLOW + Style.BRIGHT}Graceful Shutdown Sequence Complete. Pyrmethus rests."
        )
        termux_notify("Shutdown Complete", f"{self.config.symbol} shutdown finished.")

    def _display_startup_info(self):
        """Prints initial configuration details using Rich."""
        console.print(
            "[bold bright_cyan] Summoning Pyrmethus [magenta]v2.4.1 (Enhanced)[/]..."
        )
        console.print(
            f"[yellow]Trading Symbol:[/] [white]{self.config.symbol}[/] | "
            f"Interval: [white]{self.config.interval}[/] | "
            f"Category: [white]{self.config.bybit_v5_category}[/]"
        )
        console.print(
            f"[yellow]Risk:[/] [white]{self.config.risk_percentage:.3%}[/] | "
            f"SL Mult: [white]{self.config.sl_atr_multiplier.normalize()}x[/] | "
            f"TP Mult: [white]{self.config.tp_atr_multiplier.normalize()}x[/]"
        )
        console.print(
            f"[yellow]TSL Act Mult:[/] [white]{self.config.tsl_activation_atr_multiplier.normalize()}x[/] | "
            f"TSL %: [white]{self.config.trailing_stop_percent.normalize()}%[/]"
        )
        console.print(
            f"[yellow]Trend Filter:[/] [white]{'ON' if self.config.trade_only_with_trend else 'OFF'}[/] | "
            f"ATR Move Filter: [white]{self.config.atr_move_filter_multiplier.normalize()}x[/] | "
            f"ADX Filter: [white]>{self.config.min_adx_level.normalize()}[/]"
        )
        journal_status = (
            f"Enabled ([dim]{self.config.journal_file_path}[/])"
            if self.config.enable_journaling
            else "Disabled"
        )
        console.print(f"[yellow]Journaling:[/] [white]{journal_status}[/]")
        console.print(
            f"[dim]Using V5 Position Stops (SLTrig:{self.config.sl_trigger_by}, "
            f"TSLTrig:{self.config.tsl_trigger_by}, PosIdx:{self.config.position_idx})[/]"
        )


# --- Main Execution Block ---
if __name__ == "__main__":
    # Load .env file automatically if TradingConfig doesn't do it early enough
    # load_dotenv() # TradingConfig now handles this internally

    try:
        bot = TradingBot()
        bot.run()
    except SystemExit as e:
        # Catch sys.exit() calls for clean termination (e.g., from config validation)
        logger.info(f"SystemExit called (code: {e.code}), terminating process.")
    except Exception as main_exception:
        # Catch errors during critical phases like initialization or final shutdown
        logger.critical(
            f"{Fore.RED + Style.BRIGHT}Critical error during bot execution: {main_exception}",
            exc_info=True,
        )
        termux_notify(
            "Pyrmethus CRITICAL ERROR",
            f"Bot failed: {str(main_exception)[:100]}",
        )
        sys.exit(1)  # Ensure non-zero exit code on critical failure
