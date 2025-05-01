```python
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
        self.min_adx_level: Decimal = self