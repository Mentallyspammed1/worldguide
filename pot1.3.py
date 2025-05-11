# --*- coding: utf-8 -*-
# pylint: disable=logging-fstring-interpolation, too-many-instance-attributes, too-many-arguments, too-many-locals, too-many-public-methods, invalid-name, unused-argument, too-many-lines, unnecessary-pass
# fmt: off
#   ____        _       _   _                  _            _         _
#  |  _ \\ _   _| |_ ___| | | | __ ___   ____ _| |_ ___  ___| |_ _ __ | |__   ___ _ __ ___  _ __
#  | |_) | | | | __/ _ \\ | | |/ _` \\ \\ / / _` | __/ _ \\/ __| __| '_ \\| '_ \\ / _ \\ '_ ` _ \\| '_ \\
#  |  __/| |_| | ||  __/ | | | (_| |\\ V / (_| | ||  __/\\__ \\ |_| |_) | | | |  __/ | | | | | |_) |
#  |_|    \\__, |\\__\\___|_|_|_|\\__,_| \\_/ \\__,_|\\__\\___||___/\\__| .__/|_| |_|\\___|_| |_| |_| .__/
#         |___/                                                |_|                      |_|
# Pyrmethus v4.5.8 - Neon Nexus Edition (VolumaticTrend Integration)
# fmt: on
"""
Pyrmethus - Termux Trading Spell (v4.5.8 - Neon Nexus Edition)

Conjures market insights and executes trades on Bybit Futures.
This version integrates the VolumaticTrend strategy and uses the V5 Unified Account API
via CCXT, employing classes for better structure and leveraging V5 position-based
stop-loss, take-profit, and trailing-stop features.
"""

# Standard Library Imports
import copy
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
    ROUND_UP,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    getcontext,
)
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, Literal

# Third-Party Imports
# Define COMMON_PACKAGES before the try-except block for imports
COMMON_PACKAGES = [
    "ccxt",
    "python-dotenv",
    "pandas",
    "numpy",
    "pandas-ta",  # For VolumaticTrend indicators like VWMA
    "rich",
    "colorama",
    "requests",
]

# Attempt to import colorama first for styled error messages
_COLORAMA_SUCCESSFULLY_IMPORTED = False
try:
    from colorama import Fore, Style, init as colorama_init
    _COLORAMA_SUCCESSFULLY_IMPORTED = True
except ImportError:
    # colorama itself is missing, this will be handled in the main try-except block
    # Define dummy Fore, Style, colorama_init so the rest of the script doesn't break if it tries to use them before exit
    class _DummyColor:
        # Return empty string for any attribute
        def __getattr__(self, name: str) -> str: return ""
    Fore, Style = _DummyColor(), _DummyColor()  # type: ignore
    # Dummy init function
    def colorama_init(*_args: Any, **_kwargs: Any) -> None: pass

try:
    import ccxt
    import numpy as np
    import pandas as pd
    import pandas_ta as ta  # For VWMA and other TA functions
    import requests
    from dotenv import load_dotenv
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table  # Keep for potential future use
    from rich.text import Text

except ImportError as e:
    missing_pkg = e.name
    can_use_colorama_for_error = _COLORAMA_SUCCESSFULLY_IMPORTED and missing_pkg != "colorama"

    if missing_pkg == "colorama" or not can_use_colorama_for_error:
        print(f"Missing essential spell component: {missing_pkg}")
        if missing_pkg == "colorama":
            print("Missing essential package: colorama. Cannot display colored output.")
        print(f"To conjure it, cast: pip install {missing_pkg}")
        print("\nOr, to ensure all scrolls are present, cast:")
        print(f"pip install {' '.join(COMMON_PACKAGES)}")
    else:
        colorama_init(autoreset=True)
        print(
            f"{Fore.RED}{Style.BRIGHT}Missing essential spell component: {Style.BRIGHT}{missing_pkg}{Style.NORMAL}"
        )
        print(
            f"{Fore.YELLOW}To conjure it, cast: {Style.BRIGHT}pip install {missing_pkg}{Style.RESET_ALL}"
        )
        print(f"\n{Fore.CYAN}Or, to ensure all scrolls are present, cast:")
        is_termux = "com.termux" in os.environ.get("PREFIX", "")

        if is_termux:
            termux_pkgs_to_install = []
            pip_pkgs_to_install = list(COMMON_PACKAGES)

            if "pandas" in COMMON_PACKAGES:
                termux_pkgs_to_install.append("python-pandas")
                if 'pandas' in pip_pkgs_to_install:
                    pip_pkgs_to_install.remove('pandas')
            if "numpy" in COMMON_PACKAGES:
                termux_pkgs_to_install.append("python-numpy")
                if 'numpy' in pip_pkgs_to_install:
                    pip_pkgs_to_install.remove('numpy')

            install_cmd_parts = []
            if termux_pkgs_to_install:
                install_cmd_parts.append(
                    f"pkg install {' '.join(termux_pkgs_to_install)}")
            if pip_pkgs_to_install:
                install_cmd_parts.append(
                    f"pip install {' '.join(pip_pkgs_to_install)}")

            install_cmd = " && ".join(
                install_cmd_parts) if install_cmd_parts else f"pip install {' '.join(COMMON_PACKAGES)}"
            print(f"{Style.BRIGHT}{install_cmd}{Style.RESET_ALL}")

            if termux_pkgs_to_install:
                termux_base_names = [pkg.replace(
                    'python-', '') for pkg in termux_pkgs_to_install]
                print(
                    f"{Fore.YELLOW}Note: In Termux, {' and '.join(termux_base_names)} are often best installed via 'pkg' for compatibility.{Style.RESET_ALL}"
                )
        else:
            print(
                f"{Style.BRIGHT}pip install {' '.join(COMMON_PACKAGES)}{Style.RESET_ALL}"
            )
    sys.exit(1)

# --- Constants ---
DECIMAL_PRECISION = 50
# Minimum quantity to be considered an active position
POSITION_QTY_EPSILON = Decimal("1E-12")
DEFAULT_PRICE_DP = 4  # Default decimal places for prices if not available from market info
# Default decimal places for amounts if not available from market info
DEFAULT_AMOUNT_DP = 6
DEFAULT_OHLCV_LIMIT = 200  # Default number of OHLCV candles to fetch
DEFAULT_LOOP_SLEEP = 15  # Default sleep time in seconds between trading cycles
DEFAULT_RETRY_DELAY = 3  # Default delay in seconds between retries for API calls
DEFAULT_MAX_RETRIES = 3  # Default maximum number of retries for API calls
# Default risk percentage per trade (1%)
DEFAULT_RISK_PERCENT = Decimal("0.01")
DEFAULT_SL_MULT = Decimal("1.5")  # Default ATR multiplier for Stop Loss
DEFAULT_TP_MULT = Decimal("3.0")  # Default ATR multiplier for Take Profit
# Default ATR multiplier for Trailing Stop Loss activation
DEFAULT_TSL_ACT_MULT = Decimal("1.0")
# Default Trailing Stop Loss distance as percentage of current price
DEFAULT_TSL_PERCENT = Decimal("0.5")
DEFAULT_STOCH_OVERSOLD = Decimal("30")  # Default Stochastic oversold threshold
# Default Stochastic overbought threshold
DEFAULT_STOCH_OVERBOUGHT = Decimal("70")
DEFAULT_MIN_ADX = Decimal("20")  # Default minimum ADX level for trend strength
# Default trading journal filename
DEFAULT_JOURNAL_FILE = "pyrmethus_trading_journal.csv"
V5_UNIFIED_ACCOUNT_TYPE = "UNIFIED"  # Bybit V5 Unified Account type
# Default position index: 0 for One-Way mode; Hedge mode uses 1 (Long) or 2 (Short)
DEFAULT_POSITION_IDX = 0
V5_TPSL_MODE_FULL = "Full"  # Bybit V5 TPSL mode for entire position
V5_SUCCESS_RETCODE = 0  # Bybit V5 API success return code
TERMUX_NOTIFY_TIMEOUT = 10  # Timeout in seconds for Termux notification commands

# OrderManager internal states for protection_tracker
PROTECTION_STATE_SLTP: Literal["ACTIVE_SLTP"] = "ACTIVE_SLTP"
PROTECTION_STATE_TSL: Literal["ACTIVE_TSL"] = "ACTIVE_TSL"


colorama_init(autoreset=True)
console = Console(log_path=False)  # Disable Rich's own log file handling
getcontext().prec = DECIMAL_PRECISION

# --- Logging Setup ---
TRADE_LEVEL_NUM = 25  # Custom log level for trade actions
log_level_display_name: str  # To be set during log level configuration

if not hasattr(logging, "TRADE"):
    logging.addLevelName(TRADE_LEVEL_NUM, "TRADE")

# Ensure the 'trade' method is only added if it doesn't exist
if not hasattr(logging.Logger, "trade"):
    def trade_log(self: logging.Logger, message: str, *args: Any, **kws: Any) -> None:
        """Logs a message with custom level TRADE."""
        if self.isEnabledFor(TRADE_LEVEL_NUM):
            # pylint: disable=protected-access
            self._log(TRADE_LEVEL_NUM, message, args,
                      **kws)  # type: ignore[arg-type]
    logging.Logger.trade = trade_log  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)
log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)-8s] (%(filename)s:%(lineno)d) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log_level_str_env = os.getenv("LOG_LEVEL", "INFO").upper()
valid_log_levels = ["DEBUG", "INFO", "TRADE", "WARNING", "ERROR", "CRITICAL"]
log_level_to_set: int

if log_level_str_env.isdigit() and int(log_level_str_env) == TRADE_LEVEL_NUM:
    log_level_to_set = TRADE_LEVEL_NUM
    log_level_display_name = "TRADE"
elif log_level_str_env in valid_log_levels:
    log_level_to_set = getattr(logging, log_level_str_env)
    log_level_display_name = log_level_str_env
else:
    print(f"{Fore.YELLOW}Warning: Invalid LOG_LEVEL '{log_level_str_env}'. Defaulting to INFO.{Style.RESET_ALL}")
    log_level_to_set = logging.INFO
    log_level_display_name = "INFO"

logger.setLevel(log_level_to_set)
if not logger.hasHandlers():  # Add handler only if no handlers are already configured
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
logger.propagate = False  # Prevent duplicate logs if root logger is also configured

# --- Utility Functions ---


def safe_decimal(value: Any, default: Decimal = Decimal("NaN")) -> Decimal:
    """
    Safely converts a value to a Decimal.
    Returns `default` (Decimal("NaN") by default) if conversion fails or input is None/empty/NaN-like string.
    """
    if value is None:
        return default
    try:
        str_value = str(value).strip()
        # Check for common string representations of NaN or null values
        if not str_value or str_value.lower() in ("nan", "none", "null"):
            return default
        return Decimal(str_value)
    except (InvalidOperation, ValueError, TypeError):
        return default


def termux_notify(title: str, content: str) -> None:
    """Sends a toast notification in Termux if the 'termux-toast' command is available."""
    if "com.termux" not in os.environ.get("PREFIX", ""):
        logger.debug("Not in Termux environment, skipping notification.")
        return
    try:
        # Termux-toast primarily uses content; title is effectively prepended or ignored by the toast.
        # A short timeout is used for the notification command itself.
        result = subprocess.run(
            ["termux-toast", content],
            check=False,  # Don't raise CalledProcessError, handle return code manually
            timeout=TERMUX_NOTIFY_TIMEOUT,
            capture_output=True, text=True  # Capture stdout/stderr for logging
        )
        if result.returncode != 0:
            # Consolidate error output checking for clearer logging
            error_output = (
                result.stderr or result.stdout or "[No output from termux-toast command]").strip()
            logger.warning(
                f"Termux toast command failed (code {result.returncode}): {error_output}")
    except FileNotFoundError:
        logger.warning(
            "Termux notify failed: 'termux-toast' command not found. Is Termux:API installed and configured?")
    except subprocess.TimeoutExpired:
        logger.warning(
            f"Termux notify failed: 'termux-toast' command timed out after {TERMUX_NOTIFY_TIMEOUT} seconds.")
    except Exception as e:
        # Log generic exceptions with exc_info=False to avoid large tracebacks for simple notification failures
        logger.warning(
            f"Termux notify failed unexpectedly: {e}", exc_info=False)


def fetch_with_retries(
    fetch_function: Callable[..., Any], *args: Any,
    max_retries: int = DEFAULT_MAX_RETRIES, delay_seconds: int = DEFAULT_RETRY_DELAY,
    retry_on_exceptions: Tuple[Type[Exception], ...] = (
        ccxt.DDoSProtection, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable,
        ccxt.NetworkError, ccxt.RateLimitExceeded, requests.exceptions.ConnectionError,
        requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError,
        requests.exceptions.ReadTimeout),
    fatal_exceptions: Tuple[Type[Exception], ...] = (
        ccxt.AuthenticationError, ccxt.PermissionDenied),
    fail_fast_exceptions: Tuple[Type[Exception], ...] = (
        ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.OrderNotFound),
    **kwargs: Any
) -> Any:
    """
    Executes a given function with a retry mechanism for specified exceptions.
    Logs errors and retries, and raises critical/fatal errors immediately.
    """
    last_exception: Optional[Exception] = None
    func_name = getattr(fetch_function, "__name__", "Unnamed function")

    for attempt in range(max_retries + 1):
        try:
            result = fetch_function(*args, **kwargs)
            if attempt > 0:  # Log success only if it was a retry
                logger.info(
                    f"{Style.BRIGHT}{Fore.GREEN}Successfully executed {func_name} on attempt {attempt + 1}/{max_retries + 1}.{Style.RESET_ALL}")
            return result
        except fatal_exceptions as e:
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Fatal error ({type(e).__name__}) executing {func_name}: {e}. Halting.{Style.RESET_ALL}", exc_info=False)
            raise
        except fail_fast_exceptions as e:
            logger.error(
                f"{Fore.RED}Fail-fast error ({type(e).__name__}) executing {func_name}: {e}. Not retrying this call.{Style.RESET_ALL}", exc_info=False)
            last_exception = e
            break  # Break from retry loop, do not proceed further
        except retry_on_exceptions as e:
            last_exception = e
            # Truncate long error messages
            err_summary = str(e)[:150] + ("..." if len(str(e)) > 150 else "")
            msg_base = f"Retryable error ({type(e).__name__}) on attempt {attempt + 1}/{max_retries + 1} for {func_name}: {err_summary}."
            if attempt < max_retries:
                # exc_info=False for brevity on retries
                logger.warning(
                    f"{Fore.YELLOW}{msg_base} Retrying in {delay_seconds}s...{Style.RESET_ALL}", exc_info=False)
                time.sleep(delay_seconds)
            else:
                logger.error(
                    f"{Fore.RED}Max retries ({max_retries + 1}) reached for {func_name}. Last error: {e}{Style.RESET_ALL}", exc_info=False)
        except ccxt.ExchangeError as e:  # Catch other exchange errors not specified above
            last_exception = e
            err_summary = str(e)[:150] + ("..." if len(str(e)) > 150 else "")
            logger.error(
                f"{Fore.RED}Unhandled ExchangeError during {func_name}: {err_summary}{Style.RESET_ALL}", exc_info=False)
            if attempt < max_retries:
                logger.warning(
                    f"Retrying generic exchange error in {delay_seconds}s...")
                time.sleep(delay_seconds)
            else:
                logger.error(
                    f"Max retries reached after generic exchange error for {func_name}.")
                break
        except Exception as e:  # Catch any other Python exceptions
            last_exception = e
            # exc_info=True for full traceback on unexpected Python errors
            logger.error(
                f"{Fore.RED}Unexpected Python error during {func_name}: {e}{Style.RESET_ALL}", exc_info=True)
            break  # Do not retry unexpected Python errors by default

    if last_exception:
        raise last_exception
    # This line should ideally be unreachable if the loop always raises or returns.
    # Added as a safeguard.
    raise RuntimeError(
        f"Function {func_name} failed after {max_retries + 1} attempts without returning or raising a recognized exception.")

# --- Configuration Class ---


class TradingConfig:
    """Handles loading, validation, and storage of trading configuration parameters."""
    # pylint: disable=too-many-statements

    def __init__(self, env_file: str = ".env"):
        logger.debug(
            f"Loading configuration from environment variables / '{env_file}'...")
        env_path = Path(env_file)
        if env_path.is_file():
            # `override=True` ensures .env takes precedence
            load_dotenv(dotenv_path=env_path, override=True)
            logger.info(f"Loaded configuration from {env_path}")
        else:
            logger.warning(
                f"Environment file '{env_path}' not found. Relying solely on system environment variables.")

        # Initialize attributes used by _determine_v5_category first
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT", Style.DIM,
                                         help_text="Trading symbol, e.g., BTC/USDT:USDT for Bybit V5 linear futures.")
        self.market_type: str = self._get_env("MARKET_TYPE", "linear", Style.DIM, allowed_values=[
                                              "linear", "inverse", "swap"], help_text="Market type (linear, inverse, swap). Effects V5 category.").lower()

        # Now determine category as dependent attributes are set
        self.bybit_v5_category: str = self._determine_v5_category()

        # General Trading Parameters
        self.interval: str = self._get_env(
            "INTERVAL", "1m", Style.DIM, help_text="Candle interval, e.g., '1m', '5m', '1h'.")
        self.risk_percentage: Decimal = self._get_env("RISK_PERCENTAGE", DEFAULT_RISK_PERCENT, Fore.YELLOW, cast_type=Decimal, min_val=Decimal(
            "0.00001"), max_val=Decimal("0.5"), help_text="Account equity percentage to risk per trade.")
        self.position_idx: int = self._get_env("POSITION_IDX", DEFAULT_POSITION_IDX, Style.DIM, cast_type=int, allowed_values=[
                                               0, 1, 2], help_text="Bybit V5 position index (0: One-Way, 1: Hedge Long, 2: Hedge Short).")

        # Stop-Loss, Take-Profit, Trailing-Stop Parameters
        self.sl_atr_multiplier: Decimal = self._get_env("SL_ATR_MULTIPLIER", DEFAULT_SL_MULT, Fore.YELLOW, cast_type=Decimal, min_val=Decimal(
            "0.1"), max_val=Decimal("20.0"), help_text="ATR multiplier for initial Stop Loss distance.")
        self.tp_atr_multiplier: Decimal = self._get_env("TP_ATR_MULTIPLIER", DEFAULT_TP_MULT, Fore.YELLOW, cast_type=Decimal, min_val=Decimal(
            "0.0"), max_val=Decimal("50.0"), help_text="ATR multiplier for Take Profit distance (0 to disable TP).")
        self.tsl_activation_atr_multiplier: Decimal = self._get_env("TSL_ACTIVATION_ATR_MULTIPLIER", DEFAULT_TSL_ACT_MULT, Fore.YELLOW, cast_type=Decimal, min_val=Decimal(
            "0.1"), max_val=Decimal("20.0"), help_text="ATR multiplier from entry price to activate Trailing Stop Loss.")
        self.trailing_stop_percent: Decimal = self._get_env("TRAILING_STOP_PERCENT", DEFAULT_TSL_PERCENT, Fore.YELLOW, cast_type=Decimal, min_val=Decimal(
            "0.001"), max_val=Decimal("10.0"), help_text="Trailing Stop Loss distance as a percentage of current price once activated.")
        self.sl_trigger_by: str = self._get_env("SL_TRIGGER_BY", "LastPrice", Style.DIM, allowed_values=[
                                                "LastPrice", "MarkPrice", "IndexPrice"], help_text="Price type to trigger SL.")
        self.tsl_trigger_by: str = self._get_env("TSL_TRIGGER_BY", "LastPrice", Style.DIM, allowed_values=[
                                                 # Note: Bybit TSL distance trails LastPrice. This is for activation.
                                                 "LastPrice", "MarkPrice", "IndexPrice"], help_text="Price type to trigger TSL activation price.")

        # Original Strategy Indicator Parameters
        self.trend_ema_period: int = self._get_env(
            "TREND_EMA_PERIOD", 12, Style.DIM, cast_type=int, min_val=5, max_val=500)
        self.fast_ema_period: int = self._get_env(
            "FAST_EMA_PERIOD", 9, Style.DIM, cast_type=int, min_val=1, max_val=200)
        self.slow_ema_period: int = self._get_env(
            "SLOW_EMA_PERIOD", 21, Style.DIM, cast_type=int, min_val=2, max_val=500)
        self.stoch_period: int = self._get_env(
            "STOCH_PERIOD", 7, Style.DIM, cast_type=int, min_val=1, max_val=100)
        self.stoch_smooth_k: int = self._get_env(
            "STOCH_SMOOTH_K", 3, Style.DIM, cast_type=int, min_val=1, max_val=10)
        self.stoch_smooth_d: int = self._get_env(
            "STOCH_SMOOTH_D", 3, Style.DIM, cast_type=int, min_val=1, max_val=10)
        self.atr_period: int = self._get_env(
            "ATR_PERIOD", 5, Style.DIM, cast_type=int, min_val=1, max_val=100)
        self.adx_period: int = self._get_env(
            "ADX_PERIOD", 14, Style.DIM, cast_type=int, min_val=2, max_val=100)

        # Original Strategy Thresholds & Filters
        self.stoch_oversold_threshold: Decimal = self._get_env(
            "STOCH_OVERSOLD_THRESHOLD", DEFAULT_STOCH_OVERSOLD, Fore.CYAN, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("45"))
        self.stoch_overbought_threshold: Decimal = self._get_env(
            "STOCH_OVERBOUGHT_THRESHOLD", DEFAULT_STOCH_OVERBOUGHT, Fore.CYAN, cast_type=Decimal, min_val=Decimal("55"), max_val=Decimal("100"))
        self.trend_filter_buffer_percent: Decimal = self._get_env("TREND_FILTER_BUFFER_PERCENT", Decimal(
            "0.5"), Fore.CYAN, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("5"), help_text="Buffer percent around trend EMA for trend filter.")
        self.atr_move_filter_multiplier: Decimal = self._get_env("ATR_MOVE_FILTER_MULTIPLIER", Decimal("0.5"), Fore.CYAN, cast_type=Decimal, min_val=Decimal(
            "0"), max_val=Decimal("5"), help_text="Multiplier for ATR to define significant price move (0 to disable).")
        self.min_adx_level: Decimal = self._get_env("MIN_ADX_LEVEL", DEFAULT_MIN_ADX, Fore.CYAN, cast_type=Decimal, min_val=Decimal(
            "0"), max_val=Decimal("90"), help_text="Minimum ADX value to consider trend strong enough.")
        self.trade_only_with_trend: bool = self._get_env(
            "TRADE_ONLY_WITH_TREND", True, Style.DIM, cast_type=bool, help_text="If true, original strategy entries must align with trend EMA.")

        # VolumaticTrend (VT) Strategy Parameters
        self.vt_enable: bool = self._get_env(
            "VT_ENABLE", True, Style.DIM, cast_type=bool, help_text="Enable/Disable the VolumaticTrend strategy component.")
        self.vt_trend_ema_period: int = self._get_env(
            "VT_TREND_EMA_PERIOD", 200, Style.DIM, cast_type=int, min_val=50, max_val=500, help_text="VT: Long-term EMA for trend direction.")
        self.vt_vwma_period: int = self._get_env(
            "VT_VWMA_PERIOD", 20, Style.DIM, cast_type=int, min_val=5, max_val=100, help_text="VT: VWMA period for short-term momentum.")
        self.vt_volume_spike_lookback: int = self._get_env(
            "VT_VOLUME_SPIKE_LOOKBACK", 20, Style.DIM, cast_type=int, min_val=5, max_val=100, help_text="VT: Lookback period for average volume calculation.")
        self.vt_volume_spike_multiplier: Decimal = self._get_env("VT_VOLUME_SPIKE_MULTIPLIER", Decimal("2.0"), Fore.CYAN, cast_type=Decimal, min_val=Decimal(
            "1.1"), max_val=Decimal("5.0"), help_text="VT: Multiplier for average volume to detect a spike.")

        # API and System Parameters
        self.api_key: str = self._get_env(
            "BYBIT_API_KEY", None, Fore.RED, is_secret=True, help_text="Bybit API Key.")
        self.api_secret: str = self._get_env(
            "BYBIT_API_SECRET", None, Fore.RED, is_secret=True, help_text="Bybit API Secret.")
        self.ohlcv_limit: int = self._get_env("OHLCV_LIMIT", DEFAULT_OHLCV_LIMIT, Style.DIM, cast_type=int,
                                              min_val=50, max_val=1000, help_text="Number of OHLCV candles to fetch for indicators.")
        self.loop_sleep_seconds: int = self._get_env(
            "LOOP_SLEEP_SECONDS", DEFAULT_LOOP_SLEEP, Style.DIM, cast_type=int, min_val=1, help_text="Seconds to sleep between trading cycles.")
        self.order_check_delay_seconds: int = self._get_env(
            "ORDER_CHECK_DELAY_SECONDS", 2, Style.DIM, cast_type=int, min_val=1, help_text="Delay after order submission before checking status.")
        self.order_fill_timeout_seconds: int = self._get_env("ORDER_FILL_TIMEOUT_SECONDS", 20, Style.DIM, cast_type=int, min_val=5,
                                                             # Note: Market orders usually fill quickly.
                                                             help_text="Timeout for market order to be considered filled (used contextually).")
        self.max_fetch_retries: int = self._get_env("MAX_FETCH_RETRIES", DEFAULT_MAX_RETRIES, Style.DIM,
                                                    cast_type=int, min_val=0, max_val=10, help_text="Max retries for fetch operations.")
        self.retry_delay_seconds: int = self._get_env(
            "RETRY_DELAY_SECONDS", DEFAULT_RETRY_DELAY, Style.DIM, cast_type=int, min_val=1, help_text="Delay between fetch retries.")
        self.journal_file_path: str = self._get_env(
            "JOURNAL_FILE_PATH", DEFAULT_JOURNAL_FILE, Style.DIM, help_text="Path to the trading journal CSV file.")
        self.enable_journaling: bool = self._get_env(
            "ENABLE_JOURNALING", True, Style.DIM, cast_type=bool, help_text="Enable/Disable trade journaling to CSV.")
        self.close_positions_on_shutdown: bool = self._get_env(
            "CLOSE_POSITIONS_ON_SHUTDOWN", True, Style.DIM, cast_type=bool, help_text="Automatically close open positions on graceful shutdown.")

        self._validate_config()
        logger.debug("Configuration loaded and validated successfully.")

    def _determine_v5_category(self) -> str:
        """Determines the Bybit V5 API category based on market type and symbol."""
        try:
            category: str
            if self.market_type == "inverse":
                category = "inverse"
            elif self.market_type in ["linear", "swap"]:
                category = "linear"  # 'swap' usually implies linear for Bybit V5
            else:
                # This case should ideally be caught by allowed_values in _get_env for MARKET_TYPE
                raise ValueError(
                    f"Unsupported MARKET_TYPE '{self.market_type}' for V5 category determination.")

            # Check if symbol includes settle currency (e.g., :USDT)
            if ":" not in self.symbol and self.market_type in ["linear", "inverse"]:
                logger.warning(
                    f"Symbol '{self.symbol}' does not explicitly include the settle currency (e.g., :USDT). For Bybit V5, the format BASE/QUOTE:SETTLE (e.g., BTC/USDT:USDT) is recommended for clarity, especially for linear/inverse futures.")

            logger.info(
                f"Determined Bybit V5 API category: '{category}' for symbol '{self.symbol}' and market type '{self.market_type}'")
            return category
        except ValueError as e:  # Catch specific error from this function
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Could not determine Bybit V5 category: {e}. Halting.{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)
        return ""  # Should be unreachable due to sys.exit

    def _validate_config(self) -> None:
        """Performs cross-parameter validation for the loaded configuration."""
        if self.fast_ema_period >= self.slow_ema_period:
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Validation failed: FAST_EMA_PERIOD ({self.fast_ema_period}) must be less than SLOW_EMA_PERIOD ({self.slow_ema_period}). Halting.{Style.RESET_ALL}")
            sys.exit(1)
        if self.trend_ema_period <= self.slow_ema_period:  # Trend EMA usually longer or equal to slow EMA
            logger.warning(f"{Fore.YELLOW}Config Warning: TREND_EMA_PERIOD ({self.trend_ema_period}) is not greater than SLOW_EMA_PERIOD ({self.slow_ema_period}). This might not be a typical configuration.{Style.RESET_ALL}")
        if self.stoch_oversold_threshold >= self.stoch_overbought_threshold:
            logger.critical(f"{Style.BRIGHT}{Fore.RED}Validation failed: STOCH_OVERSOLD_THRESHOLD ({self.stoch_oversold_threshold.normalize()}) must be less than STOCH_OVERBOUGHT_THRESHOLD ({self.stoch_overbought_threshold.normalize()}). Halting.{Style.RESET_ALL}")
            sys.exit(1)

        # TSL vs SL relationship
        if self.tsl_activation_atr_multiplier < self.sl_atr_multiplier:
            logger.warning(f"{Fore.YELLOW}Config Warning: TSL_ACTIVATION_ATR_MULTIPLIER ({self.tsl_activation_atr_multiplier.normalize()}) is less than SL_ATR_MULTIPLIER ({self.sl_atr_multiplier.normalize()}). Trailing stop might activate before the initial stop loss distance is fully established in terms of ATR multiples.{Style.RESET_ALL}")
        # TP vs SL relationship (Risk:Reward)
        if self.tp_atr_multiplier > Decimal("0") and self.tp_atr_multiplier <= self.sl_atr_multiplier:
            logger.warning(f"{Fore.YELLOW}Config Warning: TP_ATR_MULTIPLIER ({self.tp_atr_multiplier.normalize()}) is less than or equal to SL_ATR_MULTIPLIER ({self.sl_atr_multiplier.normalize()}). This implies a Risk:Reward ratio of 1:1 or less.{Style.RESET_ALL}")

        # VolumaticTrend specific validations
        if self.vt_enable:
            if self.vt_trend_ema_period < self.vt_vwma_period:  # Typically, trend EMA is longer term than VWMA
                logger.warning(f"{Fore.YELLOW}Config Warning (VT): VT_TREND_EMA_PERIOD ({self.vt_trend_ema_period}) is less than VT_VWMA_PERIOD ({self.vt_vwma_period}). This is unusual for typical VWMA usage patterns; ensure this is intended.{Style.RESET_ALL}")

            # Check OHLCV limit for VT indicator periods
            min_ohlcv_for_vt = max(self.vt_trend_ema_period, self.vt_vwma_period,
                                   self.vt_volume_spike_lookback) + 20  # Add a buffer
            if self.ohlcv_limit < min_ohlcv_for_vt:
                logger.warning(f"{Fore.YELLOW}Config Warning: OHLCV_LIMIT ({self.ohlcv_limit}) may be too small for VolumaticTrend strategy. Longest VT indicator period is {max(self.vt_trend_ema_period, self.vt_vwma_period, self.vt_volume_spike_lookback)}, requiring approximately {min_ohlcv_for_vt} candles for robust calculation.{Style.RESET_ALL}")

    def _cast_value(self, key: str, value_str: str, cast_type: Type, default: Any) -> Any:
        """Casts a string value to the specified type, with error handling."""
        val_to_cast = value_str.strip()
        if not val_to_cast:  # Handle empty string after stripping
            # If default is None or an empty string itself, allow it.
            if default is None or (isinstance(default, str) and not default):
                return default
            logger.warning(
                f"Empty value for '{key}' after stripping. Using default '{default}'.")
            return default
        try:
            if cast_type == bool:
                return val_to_cast.lower() in ("true", "1", "yes", "y", "on")
            if cast_type == Decimal:
                if val_to_cast.lower() in ("nan", "none", "null"):
                    return Decimal("NaN")  # Explicit NaN handling
                return Decimal(val_to_cast)
            if cast_type == int:
                # Attempt to convert via Decimal to catch non-integer floats
                dec_val = Decimal(val_to_cast)
                if dec_val.to_integral_value(rounding=ROUND_DOWN) != dec_val:
                    raise ValueError(
                        f"Decimal value '{val_to_cast}' with a fractional part cannot be cast to int without loss.")
                return int(dec_val)
            return cast_type(val_to_cast)
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.error(
                f"{Fore.RED}Cast failed for '{key}' (value: '{value_str}', target type: {cast_type.__name__}): {e}. Using default '{default}'.{Style.RESET_ALL}")
            return default

    def _validate_value(self, key: str, value: Any, min_val: Optional[Union[int, float, Decimal]], max_val: Optional[Union[int, float, Decimal]], allowed_values: Optional[List[Any]]) -> bool:
        """Validates a value against min/max bounds and allowed values list."""
        # Check for NaN before numeric comparisons if value is Decimal
        is_numeric_comparable = isinstance(value, (int, float, Decimal)) and not (
            isinstance(value, Decimal) and value.is_nan())

        if min_val is not None:
            if not is_numeric_comparable:
                logger.critical(
                    f"{Style.BRIGHT}{Fore.RED}Validation failed for '{key}': Non-numeric value '{value}' (type: {type(value).__name__}) cannot be compared with min_val '{min_val}'. Halting.{Style.RESET_ALL}")
                sys.exit(1)
            if value < min_val:  # type: ignore[operator]
                logger.critical(
                    f"{Style.BRIGHT}{Fore.RED}Validation failed for '{key}': Value '{value}' is less than minimum allowed '{min_val}'. Halting.{Style.RESET_ALL}")
                sys.exit(1)
        if max_val is not None:
            if not is_numeric_comparable:
                logger.critical(
                    f"{Style.BRIGHT}{Fore.RED}Validation failed for '{key}': Non-numeric value '{value}' (type: {type(value).__name__}) cannot be compared with max_val '{max_val}'. Halting.{Style.RESET_ALL}")
                sys.exit(1)
            if value > max_val:  # type: ignore[operator]
                logger.critical(
                    f"{Style.BRIGHT}{Fore.RED}Validation failed for '{key}': Value '{value}' is greater than maximum allowed '{max_val}'. Halting.{Style.RESET_ALL}")
                sys.exit(1)

        if allowed_values:
            # Normalize string comparison to lowercase if value and allowed_values are strings
            comp_value = str(value).lower() if isinstance(
                value, str) else value
            lower_allowed = [str(v).lower() if isinstance(
                v, str) else v for v in allowed_values]
            if comp_value not in lower_allowed:
                logger.error(
                    f"{Fore.RED}Validation failed for '{key}': Invalid value '{value}'. Allowed values are: {allowed_values}. Reverting to default.{Style.RESET_ALL}")
                return False
        return True

    def _get_env(self, key: str, default: Any, color: str, cast_type: Type = str, min_val: Optional[Union[int, float, Decimal]] = None, max_val: Optional[Union[int, float, Decimal]] = None, allowed_values: Optional[List[Any]] = None, is_secret: bool = False, help_text: Optional[str] = None) -> Any:
        """Retrieves, casts, and validates an environment variable."""
        value_str = os.getenv(key)
        source_info: str
        use_default_flag = False
        value_to_process_str: str

        if value_str is None or value_str.strip() == "":  # If env var not set or is empty string
            if default is None and not is_secret:  # Required non-secret config missing
                logger.critical(
                    f"{Style.BRIGHT}{Fore.RED}Required configuration '{key}' not found and no default value is set. {help_text or ''} Halting.{Style.RESET_ALL}")
                sys.exit(1)
            if default is None and is_secret:  # Required secret config missing
                logger.critical(
                    f"{Style.BRIGHT}{Fore.RED}Required secret configuration '{key}' not found. {help_text or ''} Halting.{Style.RESET_ALL}")
                sys.exit(1)

            use_default_flag = True
            value_to_process_str = str(default)  # Process the default value
            log_value_display = "****" if is_secret else str(default)
            source_info = f"default value ('{log_value_display}')"
        else:
            value_to_process_str = value_str  # Process the value from environment
            log_value_display = "****" if is_secret else value_str
            source_info = "environment variable"

        # Log with appropriate level: warning if default is used (and default is not None), info otherwise.
        log_method = logger.warning if use_default_flag and default is not None else logger.info
        log_method(
            f"Using {color}{key}: {log_value_display}{Style.RESET_ALL} (from {source_info})")

        casted_value = self._cast_value(
            key, value_to_process_str, cast_type, default)

        # Validate the casted value. If validation fails, revert to original default and re-validate.
        if not self._validate_value(key, casted_value, min_val, max_val, allowed_values):
            logger.warning(
                f"{color}Reverting '{key}' to its original default '{'****' if is_secret else default}' due to non-critical validation failure of processed value '{casted_value}'.{Style.RESET_ALL}")
            casted_value = default  # Revert to the original default passed to _get_env
            # Critical check: if the hardcoded default itself fails validation, it's a programming error.
            if not self._validate_value(key, casted_value, min_val, max_val, allowed_values):
                logger.critical(f"{Style.BRIGHT}{Fore.RED}FATAL: The hardcoded default value '{'****' if is_secret else default}' for '{key}' itself failed validation. This indicates a programming error in default values or validation rules. Halting.{Style.RESET_ALL}")
                sys.exit(1)
        return casted_value

# --- Exchange Manager Class ---


class ExchangeManager:
    """Manages interaction with the CCXT exchange object, market data, and balance."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.exchange: Optional[ccxt.Exchange] = None
        self.market_info: Optional[Dict[str, Any]] = None
        self._initialize_exchange()
        if self.exchange:  # Ensure exchange was initialized before loading market info
            self.market_info = self._load_market_info()
        else:  # Should not happen if _initialize_exchange sys.exits on failure
            logger.critical(
                "Exchange initialization failed. Cannot proceed with ExchangeManager setup.")
            sys.exit(1)

    def _initialize_exchange(self) -> None:
        """Initializes the CCXT exchange object and tests the connection."""
        logger.info(
            f"Initializing Bybit exchange interface (V5 API, Market Type: {self.config.market_type}, Category: {self.config.bybit_v5_category})...")
        try:
            exchange_params: Dict[str, Any] = {
                "apiKey": self.config.api_key,
                "secret": self.config.api_secret,
                "options": {
                    # Sets default for order creation (e.g. linear/inverse)
                    "defaultType": self.config.market_type,
                    "adjustForTimeDifference": True,  # CCXT handles time sync
                    "recvWindow": 10000,  # Bybit specific, increase if timestamp errors occur
                    "brokerId": "PyrmV5NEXUS",  # Custom broker ID for Bybit
                    # Default TIF for orders (can be overridden)
                    "defaultTimeInForce": "GTC"
                }
            }
            if os.getenv("USE_BYBIT_TESTNET", "false").lower() == "true":
                logger.warning(
                    f"{Fore.YELLOW}Using Bybit Testnet endpoint.{Style.RESET_ALL}")
                # Standard CCXT way to set testnet URLs
                exchange_params['urls'] = {
                    'api': 'https://api-testnet.bybit.com'}
                # self.exchange.set_sandbox_mode(True) # Alternative if CCXT supports it universally

            self.exchange = ccxt.bybit(exchange_params)
            self.exchange.fetch_time()  # Test connection and sync time
            logger.info(
                f"{Style.BRIGHT}{Fore.GREEN}Bybit V5 interface initialized and connection tested successfully.{Style.RESET_ALL}")

        except ccxt.AuthenticationError as e:
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Authentication failed with Bybit: {e}. Please check your API keys and permissions. Halting.{Style.RESET_ALL}", exc_info=False)
            sys.exit(1)
        # Catch broader network issues
        except (ccxt.NetworkError, requests.exceptions.RequestException) as e:
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Network error initializing exchange: {e}. Check internet connection and Bybit endpoint. Halting.{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)
        except Exception as e:  # Catch any other unexpected errors
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Unexpected error initializing exchange: {e}. Halting.{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)

    def _load_market_info(self) -> Optional[Dict[str, Any]]:
        """Loads and processes market information for the configured symbol."""
        if not self.exchange:
            logger.critical(
                "Exchange not initialized. Cannot load market info. Halting.")
            sys.exit(1)
        try:
            logger.info(
                f"Loading market information for symbol: {self.config.symbol}...")
            # Force reload to get latest info
            self.exchange.load_markets(reload=True)
            market = self.exchange.market(
                self.config.symbol)  # Get specific market data

            if not market:
                # Provide more specific guidance for Bybit V5 symbols
                raise ccxt.ExchangeError(
                    f"Market {self.config.symbol} not found. Ensure symbol format is correct (e.g., BTC/USDT:USDT for Bybit V5 linear, BTC/USD:BTC for inverse). Check available markets on Bybit.")

            def get_dp_from_precision_step(precision_val: Optional[Union[str, float, int]], default_dp: int) -> int:
                """Helper to determine decimal places from a step-size precision value (e.g., '0.01')."""
                if precision_val is None:
                    return default_dp
                prec_dec = safe_decimal(precision_val)
                if prec_dec.is_nan() or prec_dec.is_zero():
                    return default_dp

                # Normalize to remove trailing zeros, e.g., Decimal('0.0100') -> Decimal('0.01')
                normalized_prec_dec = prec_dec.normalize()
                # Gets the exponent (e.g., -2 for 0.01)
                exponent = normalized_prec_dec.as_tuple().exponent

                if isinstance(exponent, int) and exponent < 0:
                    # e.g., 0.01 (exponent -2) -> 2 DP; 0.123 (exponent -3) -> 3 DP
                    return -exponent
                else:
                    # e.g., 1 (exponent 0) -> 0 DP; 10 (exponent 1, for 1E1) -> 0 DP (meaning integer steps)
                    return 0

            # Extract and store precision details
            market["precision_dp"] = {
                "amount": get_dp_from_precision_step(market.get("precision", {}).get("amount"), DEFAULT_AMOUNT_DP),
                "price": get_dp_from_precision_step(market.get("precision", {}).get("price"), DEFAULT_PRICE_DP)
            }
            market["tick_size"] = safe_decimal(market.get("precision", {}).get(
                "price"), default=Decimal('NaN'))  # Min price increment
            market["amount_step"] = safe_decimal(market.get("precision", {}).get(
                "amount"), default=Decimal('NaN'))  # Min amount increment
            market["min_order_size"] = safe_decimal(market.get("limits", {}).get(
                "amount", {}).get("min"), default=Decimal("NaN"))
            market["contract_size"] = safe_decimal(market.get("contractSize"), default=Decimal(
                "1"))  # Default to 1 (e.g. for spot or if not specified)

            if market.get("contractSize") is None:
                logger.warning(
                    f"Contract size not explicitly found for {self.config.symbol}. Defaulting to 1. This is normal for Spot markets; verify if using derivatives and contract size is other than 1 unit of base.")

            # Log extracted market details for verification
            min_amt_str = market["min_order_size"].normalize(
            ) if not market["min_order_size"].is_nan() else "N/A"
            tick_size_str = market["tick_size"].normalize(
            ) if not market["tick_size"].is_nan() else "N/A"
            amount_step_str = market["amount_step"].normalize(
            ) if not market["amount_step"].is_nan() else "N/A"
            logger.info(
                f"Market {self.config.symbol} (API ID: {market.get('id', 'N/A')}): "
                f"Decimals(Amount={market['precision_dp']['amount']}, Price={market['precision_dp']['price']}), "
                f"Steps(TickSize={tick_size_str}, AmountStep={amount_step_str}), "
                f"Limits(MinOrderAmount={min_amt_str}), ContractSize={market['contract_size'].normalize()}, "
                f"SettleCurrency: {market.get('settle', 'N/A')}, Base: {market.get('base', 'N/A')}, Quote: {market.get('quote', 'N/A')}"
            )
            return market
        # Catch a broad range of potential issues
        except (ccxt.ExchangeError, KeyError, ValueError, TypeError, Exception) as e:
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Failed to load or parse market info for {self.config.symbol}: {e}. Halting.{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)
        return None  # Should be unreachable

    def format_price(self, price: Union[Decimal, str, float, int]) -> str:
        """Formats a price to the market's price precision (decimal places)."""
        price_decimal = safe_decimal(price)
        if price_decimal.is_nan():
            return "NaN"  # Return "NaN" for invalid inputs

        # Determine precision from market_info, fallback to default
        precision_dp = DEFAULT_PRICE_DP
        if self.market_info and "precision_dp" in self.market_info and "price" in self.market_info["precision_dp"]:
            precision_dp = self.market_info["precision_dp"]["price"]

        try:
            # Quantize to the required number of decimal places
            # e.g., Decimal('0.01') for 2 DP
            quantizer = Decimal('1').scaleb(-precision_dp)
            formatted_price_decimal = price_decimal.quantize(
                quantizer, rounding=ROUND_HALF_EVEN)
            # Format as string with fixed DP
            return f"{formatted_price_decimal:.{precision_dp}f}"
        except (InvalidOperation, ValueError):
            logger.error(
                f"Error formatting price '{price_decimal}' with {precision_dp} DP.", exc_info=False)
            return "ERR"  # Error string

    def format_amount(self, amount: Union[Decimal, str, float, int], rounding_mode=ROUND_DOWN) -> str:
        """Formats an amount to the market's amount precision (decimal places)."""
        amount_decimal = safe_decimal(amount)
        if amount_decimal.is_nan():
            return "NaN"

        precision_dp = DEFAULT_AMOUNT_DP
        if self.market_info and "precision_dp" in self.market_info and "amount" in self.market_info["precision_dp"]:
            precision_dp = self.market_info["precision_dp"]["amount"]

        try:
            quantizer = Decimal('1').scaleb(-precision_dp)
            formatted_amount_decimal = amount_decimal.quantize(
                quantizer, rounding=rounding_mode)
            return f"{formatted_amount_decimal:.{precision_dp}f}"
        except (InvalidOperation, ValueError):
            logger.error(
                f"Error formatting amount '{amount_decimal}' with {precision_dp} DP.", exc_info=False)
            return "ERR"

    def _format_v5_param(self, value: Optional[Union[Decimal, str, float, int]], param_type: Literal["price", "amount", "distance"] = "price", allow_zero: bool = False) -> Optional[str]:
        """
        Formats a numeric value into a string suitable for Bybit V5 API parameters.
        Uses CCXT precision methods if available, otherwise custom formatting.
        Returns None if formatting fails or value is invalid for the context.
        """
        if value is None:
            return None
        decimal_value = safe_decimal(value, default=Decimal("NaN"))

        if decimal_value.is_nan():
            logger.warning(
                f"V5 Param Format: Input '{value}' (type {param_type}) is NaN. Cannot format.")
            return None

        if decimal_value.is_zero():
            if allow_zero:
                return "0"  # "0" is used to cancel SL/TP/TSL in Bybit V5
            logger.debug(
                f"V5 Param Format: Input '{value}' (type {param_type}) is zero, but zero is not allowed here (or implies cancellation, handled by allow_zero=True).")
            return None

        # Negative values are generally invalid for prices, amounts, distances
        if decimal_value < Decimal("0"):
            logger.warning(
                f"V5 Param Format: Input '{value}' (type {param_type}) is negative ({decimal_value}), which is invalid for API parameters.")
            return None

        formatted_str: str
        # Prefer CCXT's precision methods if exchange and market are loaded
        if self.exchange and self.config.symbol and self.exchange.market(self.config.symbol):
            try:
                # Distances are often formatted like prices
                if param_type in ["price", "distance"]:
                    formatted_str = self.exchange.price_to_precision(
                        self.config.symbol, float(decimal_value))
                else:  # "amount"
                    formatted_str = self.exchange.amount_to_precision(
                        self.config.symbol, float(decimal_value))
                # Validate that CCXT formatting didn't result in an issue (e.g. NaN string)
                if safe_decimal(formatted_str).is_nan():
                    raise ValueError("CCXT formatting resulted in NaN string")
            except Exception as e_ccxt_format:
                logger.warning(
                    f"CCXT's {param_type}_to_precision for value '{decimal_value}' failed ({e_ccxt_format}). Falling back to custom formatting for V5 parameter.")
                # Fallback to custom formatters
                formatted_str = self.format_price(decimal_value) if param_type in [
                    "price", "distance"] else self.format_amount(decimal_value, ROUND_DOWN)
        else:
            logger.warning(
                "V5 Param Format: Exchange/symbol not fully available for CCXT precision methods. Using custom fallback formatters.")
            formatted_str = self.format_price(decimal_value) if param_type in [
                "price", "distance"] else self.format_amount(decimal_value, ROUND_DOWN)

        if formatted_str in ("ERR", "NaN") or safe_decimal(formatted_str).is_nan():
            logger.error(
                f"V5 Param Format: Failed to produce a valid string for value '{value}' (type: {param_type}). Formatter returned: '{formatted_str}'.")
            return None
        return formatted_str

    def fetch_ohlcv(self) -> Optional[pd.DataFrame]:
        """Fetches OHLCV data and returns it as a Pandas DataFrame."""
        if not self.exchange:
            logger.error("Exchange not initialized, cannot fetch OHLCV.")
            return None

        logger.debug(
            f"Fetching up to {self.config.ohlcv_limit} OHLCV candles for {self.config.symbol} ({self.config.interval})...")
        try:
            ohlcv_data = fetch_with_retries(
                self.exchange.fetch_ohlcv,
                symbol=self.config.symbol,
                timeframe=self.config.interval,
                limit=self.config.ohlcv_limit,
                max_retries=self.config.max_fetch_retries,  # Use configured retries
                delay_seconds=self.config.retry_delay_seconds
            )
            if not ohlcv_data:  # Empty list or None
                logger.error(
                    f"fetch_ohlcv for {self.config.symbol} returned no data.")
                return None

            # Increased minimum candles for more robust indicator calculation start
            min_candles_for_robust_ta = 20
            if len(ohlcv_data) < min_candles_for_robust_ta:
                logger.warning(
                    f"Fetched only {len(ohlcv_data)} candles for {self.config.symbol}. This may be insufficient for some indicator lookback periods. Proceeding with caution.")

            df = pd.DataFrame(ohlcv_data, columns=[
                              "timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], unit="ms", utc=True)  # Ensure UTC timezone
            df.set_index("timestamp", inplace=True)

            # Convert OHLCV columns to Decimal for precision, handling potential errors
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].apply(safe_decimal)

            initial_len = len(df)
            # Drop rows where any of O/H/L/C/V is NaN after conversion
            df.dropna(subset=["open", "high", "low", "close",
                      "volume"], inplace=True, how='any')
            if len(df) < initial_len:
                logger.warning(
                    f"Dropped {initial_len - len(df)} rows from OHLCV data due to NaN values in critical O/H/L/C/V columns.")

            if df.empty:
                logger.error(
                    "OHLCV DataFrame is empty after processing (e.g., all rows had NaNs).")
                return None

            last_ts_str = df.index[-1].strftime(
                '%Y-%m-%d %H:%M:%S %Z') if not df.empty else 'N/A'
            logger.debug(
                f"Fetched and processed {len(df)} OHLCV candles. Last timestamp: {last_ts_str}")
            return df
        except Exception as e:
            logger.error(
                f"Failed to fetch or process OHLCV data for {self.config.symbol}: {e}", exc_info=True)
            return None

    def get_balance(self) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """
        Fetches total equity and available balance for the settle currency of the configured symbol.
        Returns (total_equity, available_balance), or (None, None) on failure.
        """
        if not self.exchange or not self.market_info:
            logger.error(
                "Exchange or market info not available for balance fetch.")
            return None, None

        settle_currency = self.market_info.get("settle")
        if not settle_currency:
            logger.error(
                f"Settle currency for symbol {self.config.symbol} not found in market info. Cannot get specific balance.")
            return None, None

        logger.debug(
            f"Fetching balance for {settle_currency} (AccountType: {V5_UNIFIED_ACCOUNT_TYPE}, Category: {self.config.bybit_v5_category})...")
        try:
            # V5 API requires accountType for fetch_balance if not default
            # For Unified accounts, coin can also be specified to get details for that coin.
            balance_data = fetch_with_retries(
                self.exchange.fetch_balance,
                params={"accountType": V5_UNIFIED_ACCOUNT_TYPE,
                        "coin": settle_currency},
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds
            )

            total_equity, available_balance = Decimal("NaN"), Decimal("NaN")

            # Parse Bybit V5 specific structure from `balance_data['info']`
            if "info" in balance_data and "result" in balance_data["info"] and "list" in balance_data["info"]["result"]:
                account_list = balance_data["info"]["result"]["list"]
                if account_list and isinstance(account_list, list):
                    # Find the UNIFIED account details
                    unified_acc_info = next((item for item in account_list if item.get(
                        "accountType") == V5_UNIFIED_ACCOUNT_TYPE), None)
                    if unified_acc_info:
                        total_equity = safe_decimal(
                            unified_acc_info.get("totalEquity"))
                        # Try to get available balance specific to the settle currency
                        coin_details_list = unified_acc_info.get("coin", [])
                        if coin_details_list and isinstance(coin_details_list, list):
                            settle_coin_info = next(
                                (c for c in coin_details_list if c.get("coin") == settle_currency), None)
                            if settle_coin_info:
                                available_balance = safe_decimal(
                                    settle_coin_info.get("availableToWithdraw"))
                                # Fallback for total_equity if not found at account level but present at coin level
                                if total_equity.is_nan() and settle_coin_info.get("equity") is not None:
                                    total_equity = safe_decimal(
                                        settle_coin_info.get("equity"))

                        # Fallback for available_balance if not found at coin level
                        if available_balance.is_nan() and unified_acc_info.get("totalAvailableBalance") is not None:
                            available_balance = safe_decimal(
                                unified_acc_info.get("totalAvailableBalance"))
                            logger.debug(
                                f"Used 'totalAvailableBalance' for {settle_currency} as coin-specific 'availableToWithdraw' was not found or parsed.")

            # Fallback to CCXT standardized fields if V5 parsing failed or values are still NaN
            if total_equity.is_nan() and balance_data.get("total", {}).get(settle_currency) is not None:
                total_equity = safe_decimal(
                    balance_data["total"][settle_currency])
                logger.debug(
                    f"Used CCXT standardized 'total.{settle_currency}' balance field as fallback for total equity.")
            if available_balance.is_nan() and balance_data.get("free", {}).get(settle_currency) is not None:
                available_balance = safe_decimal(
                    balance_data["free"][settle_currency])
                logger.debug(
                    f"Used CCXT standardized 'free.{settle_currency}' balance field as fallback for available balance.")

            # Final checks and logging
            if total_equity.is_nan():
                logger.error(
                    f"Could not extract valid total equity for {settle_currency}. Raw 'info.result.list[0]' (truncated): {str(balance_data.get('info', {}).get('result', {}).get('list', [{}])[0])[:300]}")
                # Return available if parsed, else 0
                return None, available_balance if not available_balance.is_nan() else Decimal("0")
            if available_balance.is_nan():
                logger.warning(
                    f"Could not extract valid available balance for {settle_currency}. Defaulting to 0 for available balance.")
                # Default to 0 if parsing fails
                available_balance = Decimal("0")

            logger.debug(
                f"Balance ({settle_currency}): Total Equity = {total_equity.normalize()}, Available Balance = {available_balance.normalize()}")
            return total_equity, available_balance
        except Exception as e:
            logger.error(
                f"Failed to fetch or parse balance for {settle_currency}: {e}", exc_info=True)
            return None, None

    def get_current_position(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Fetches the current position for the configured symbol and V5 category/positionIdx.
        Returns a dictionary summarizing the position: {"long": {...}, "short": {...}},
        where only one side will be populated if a position exists for the configured positionIdx.
        Returns None on failure to fetch.
        """
        if not self.exchange or not self.market_info:
            logger.error(
                "Exchange or market info not available for position fetch.")
            return None

        market_id = self.market_info.get("id")  # API-specific symbol ID
        if not market_id:
            logger.error(
                f"Market ID for {self.config.symbol} not found in market info. Cannot fetch position.")
            return None

        # Initialize with empty dicts for long/short sides
        positions_summary: Dict[str, Dict[str, Any]] = {
            "long": {}, "short": {}}

        logger.debug(
            f"Fetching position for {self.config.symbol} (API ID: {market_id}, Category: {self.config.bybit_v5_category}, PositionIdx: {self.config.position_idx})...")
        try:
            # CCXT's fetch_positions can return a list. We need to find the one matching our symbol and positionIdx.
            fetched_positions_list_ccxt = fetch_with_retries(
                self.exchange.fetch_positions,
                # Specify symbol to filter (though Bybit V5 might require it in params)
                symbols=[self.config.symbol],
                params={"category": self.config.bybit_v5_category,
                        "symbol": market_id},  # Pass category and symbol for V5
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds
            )

            if not fetched_positions_list_ccxt:  # Empty list means no positions for this symbol/category
                logger.debug(
                    "No position data returned from fetch_positions (empty list). Assuming flat for configured positionIdx.")
                return positions_summary

            target_pos_info_raw_api = None
            # Iterate through returned positions to find the one matching our configured positionIdx
            for pos_data_ccxt_unified in fetched_positions_list_ccxt:
                raw_api_entry = pos_data_ccxt_unified.get(
                    "info", {})  # Bybit V5 raw response
                pos_idx_from_api_str = raw_api_entry.get("positionIdx")
                try:
                    # Convert API's positionIdx (string) to int for comparison
                    pos_idx_from_api_int = int(
                        pos_idx_from_api_str) if pos_idx_from_api_str is not None else -1  # Default if missing
                except ValueError:
                    logger.warning(
                        f"Could not parse positionIdx '{pos_idx_from_api_str}' from API response. Skipping entry: {str(raw_api_entry)[:200]}")
                    continue

                if pos_idx_from_api_int == self.config.position_idx:
                    target_pos_info_raw_api = raw_api_entry
                    logger.debug(
                        f"Found position entry from API for symbol {market_id} matching configured positionIdx={self.config.position_idx}")
                    break  # Found the relevant position entry

            if not target_pos_info_raw_api:
                logger.debug(
                    f"No position entry found from API for symbol {market_id} with positionIdx={self.config.position_idx}. Assuming flat.")
                return positions_summary

            # Process the found position entry
            qty_abs = safe_decimal(
                target_pos_info_raw_api.get("size", "0")).copy_abs()
            if qty_abs < POSITION_QTY_EPSILON:  # Treat negligible size as flat
                logger.debug(
                    f"Position size {qty_abs.normalize()} for Idx {self.config.position_idx} is negligible. Considered flat.")
                return positions_summary

            api_side_str = target_pos_info_raw_api.get(
                "side", "None").lower()  # "Buy", "Sell", or "None"
            # Will be "long" or "short"
            position_side_key: Optional[str] = None

            # Determine logical side ("long" or "short") based on positionIdx and API side
            if self.config.position_idx == 0:  # One-Way Mode
                if api_side_str == "buy":
                    position_side_key = "long"
                elif api_side_str == "sell":
                    position_side_key = "short"
                elif api_side_str == "none" and qty_abs > POSITION_QTY_EPSILON:  # Should not happen if size > 0
                    logger.warning(
                        f"Inconsistent state for One-Way (Idx 0): API side is 'None' but size is {qty_abs.normalize()}. Treating as flat for safety.")
                    return positions_summary
            elif self.config.position_idx == 1:  # Hedge Mode - Long leg
                position_side_key = "long"
                if api_side_str != "buy" and qty_abs > POSITION_QTY_EPSILON:  # API side should be "Buy" or "None" if flat
                    logger.warning(
                        f"Hedge Mode Buy (Idx 1) has API side '{api_side_str}' (expected 'Buy' or 'None') with size {qty_abs.normalize()}. Assuming 'long' based on Idx.")
            elif self.config.position_idx == 2:  # Hedge Mode - Short leg
                position_side_key = "short"
                if api_side_str != "sell" and qty_abs > POSITION_QTY_EPSILON:  # API side should be "Sell" or "None" if flat
                    logger.warning(
                        f"Hedge Mode Sell (Idx 2) has API side '{api_side_str}' (expected 'Sell' or 'None') with size {qty_abs.normalize()}. Assuming 'short' based on Idx.")

            if position_side_key:
                entry_price = safe_decimal(
                    target_pos_info_raw_api.get("avgPrice", "0"))
                # SL/TP values from API: "0" means not set.
                sl_price_api = safe_decimal(
                    target_pos_info_raw_api.get("stopLoss", "0"))
                sl_price_valid = sl_price_api if not sl_price_api.is_nan(
                ) and sl_price_api > Decimal("0") else None

                tp_price_api = safe_decimal(
                    target_pos_info_raw_api.get("takeProfit", "0"))
                tp_price_valid = tp_price_api if not tp_price_api.is_nan(
                ) and tp_price_api > Decimal("0") else None

                # Bybit V5 TSL fields:
                # 'trailingStop': The trailing distance value (e.g., "50" for 50 price points). "0" if not active.
                # 'activePrice': The price at which the TSL was activated/will activate. "0" if not active or not set.
                tsl_distance_val_api = safe_decimal(
                    target_pos_info_raw_api.get("trailingStop", "0"))
                tsl_activation_px_api = safe_decimal(
                    target_pos_info_raw_api.get("activePrice", "0"))

                # TSL is considered active if either distance or activation price is set (greater than 0)
                is_tsl_active = (not tsl_distance_val_api.is_nan() and tsl_distance_val_api > Decimal("0")) or \
                                (not tsl_activation_px_api.is_nan()
                                 and tsl_activation_px_api > Decimal("0"))

                positions_summary[position_side_key] = {
                    "qty": qty_abs,
                    # Store NaN if invalid
                    "entry_price": entry_price if not entry_price.is_nan() and entry_price > Decimal("0") else Decimal("NaN"),
                    # Liquidation price
                    "liq_price": safe_decimal(target_pos_info_raw_api.get("liqPrice", "0")),
                    # Note: Bybit spelling
                    "unrealized_pnl": safe_decimal(target_pos_info_raw_api.get("unrealisedPnl", "0")),
                    "api_side": api_side_str,  # Store the raw API side for debugging
                    # Store the full raw API response for this position
                    "info": target_pos_info_raw_api,
                    "stop_loss_price": sl_price_valid,
                    "take_profit_price": tp_price_valid,
                    "is_tsl_active": is_tsl_active,
                    "tsl_distance_val": tsl_distance_val_api if is_tsl_active and not tsl_distance_val_api.is_nan() and tsl_distance_val_api > Decimal("0") else None,
                    "tsl_trigger_price": tsl_activation_px_api if is_tsl_active and not tsl_activation_px_api.is_nan() and tsl_activation_px_api > Decimal("0") else None,
                }
                entry_str = positions_summary[position_side_key]["entry_price"].normalize(
                    # type: ignore
                ) if positions_summary[position_side_key]["entry_price"] and not positions_summary[position_side_key]["entry_price"].is_nan() else "N/A"
                logger.debug(
                    f"Identified {position_side_key.upper()} position (Idx {self.config.position_idx}): Qty={qty_abs.normalize()}, EntryPx={entry_str}")
            else:
                # This case should be rare if qty_abs > POSITION_QTY_EPSILON and positionIdx mapping logic is complete
                logger.warning(
                    f"Position size {qty_abs.normalize()} found for Idx {self.config.position_idx}, but could not map to a logical 'long'/'short' side (API side: '{api_side_str}'). Assuming flat.")
                return positions_summary

            return positions_summary
        except Exception as e:
            logger.error(
                f"Failed to fetch or parse positions for {self.config.symbol}: {e}", exc_info=True)
            return None

# --- Indicator Calculator Class ---


class IndicatorCalculator:
    """Calculates technical indicators required for the trading strategies."""

    def __init__(self, config: TradingConfig):
        self.config = config

    def calculate_indicators(self, df: pd.DataFrame) -> Optional[Dict[str, Union[Decimal, bool, int]]]:
        """
        Calculates all required technical indicators from the OHLCV DataFrame.
        Returns a dictionary of indicator values or None if calculation fails.
        """
        logger.info(
            f"{Fore.CYAN}# Weaving indicator patterns (Original & VolumaticTrend)...{Style.RESET_ALL}")
        if df is None or df.empty:
            logger.error(
                f"{Fore.RED}No DataFrame provided for indicator calculation.{Style.RESET_ALL}")
            return None

        required_ohlc_cols = ["open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in required_ohlc_cols):
            missing_cols = [
                c for c in required_ohlc_cols if c not in df.columns]
            logger.error(
                f"{Fore.RED}Input DataFrame is missing required columns for indicator calculation: {missing_cols}{Style.RESET_ALL}")
            return None

        try:
            # Work on a copy to avoid modifying the original DataFrame
            df_calc = df[required_ohlc_cols].copy()

            # Convert Decimal columns to float for TA library compatibility
            def safe_to_float(x: Any) -> float:
                if isinstance(x, (float, int)):
                    return float(x)
                if isinstance(x, Decimal):
                    return float('nan') if x.is_nan() else float(x)
                try:  # Attempt conversion for string-like values
                    val_stripped = str(x).strip().lower()
                    return float('nan') if val_stripped in ("nan", "none", "null", "") else float(val_stripped)
                except (ValueError, TypeError):
                    return float('nan')

            for col in required_ohlc_cols:
                if df_calc[col].empty:  # Handle empty series case if it occurs
                    df_calc[col] = pd.Series(dtype=float)
                    continue
                df_calc[col] = df_calc[col].apply(safe_to_float).astype(float)

            initial_len = len(df_calc)
            # Drop rows with NaNs in essential columns
            df_calc.dropna(subset=required_ohlc_cols, inplace=True, how='any')
            if len(df_calc) < initial_len:
                logger.debug(
                    f"Dropped {initial_len - len(df_calc)} rows with NaN in OHLCV columns after float conversion for TA calculations.")

            if df_calc.empty:
                logger.error(
                    f"{Fore.RED}DataFrame became empty after NaN drop during indicator pre-processing. Cannot calculate indicators.{Style.RESET_ALL}")
                return None

            # Determine minimum data length required for all indicators
            max_period_orig = max(self.config.slow_ema_period, self.config.trend_ema_period,
                                  self.config.stoch_period + self.config.stoch_smooth_k +
                                  self.config.stoch_smooth_d,  # Sum for full Stoch history
                                  # ADX often needs more data (e.g., 2x period)
                                  self.config.atr_period, self.config.adx_period * 2)
            max_period_vt = 0
            if self.config.vt_enable:
                max_period_vt = max(self.config.vt_trend_ema_period,
                                    self.config.vt_vwma_period, self.config.vt_volume_spike_lookback)

            # Add a buffer of 20 candles
            min_required_data_length = max(max_period_orig, max_period_vt) + 20
            if len(df_calc) < min_required_data_length:
                logger.error(f"{Fore.RED}Insufficient data ({len(df_calc)} rows) for robust indicator calculation. Needs approximately {min_required_data_length} rows based on configured periods. Indicators may be unreliable or NaN.{Style.RESET_ALL}")
                return None  # Or proceed with warning

            # Prepare Series for calculations
            close_s, high_s, low_s, volume_s, open_s = df_calc["close"], df_calc[
                "high"], df_calc["low"], df_calc["volume"], df_calc["open"]

            # Original Strategy Indicators
            fast_ema_s = close_s.ewm(
                span=self.config.fast_ema_period, adjust=False).mean()
            slow_ema_s = close_s.ewm(
                span=self.config.slow_ema_period, adjust=False).mean()
            trend_ema_s = close_s.ewm(
                span=self.config.trend_ema_period, adjust=False).mean()

            # Stochastic Oscillator (%K and %D)
            low_min_stoch = low_s.rolling(window=self.config.stoch_period, min_periods=max(
                1, self.config.stoch_period // 2)).min()
            high_max_stoch = high_s.rolling(window=self.config.stoch_period, min_periods=max(
                1, self.config.stoch_period // 2)).max()
            stoch_range = high_max_stoch - low_min_stoch
            # Avoid division by zero if range is zero (e.g. flat price action); default to 50 (mid-range)
            stoch_k_raw_values = np.where(
                stoch_range > 1e-12, 100 * (close_s - low_min_stoch) / stoch_range, 50.0)
            stoch_k_raw_s = pd.Series(stoch_k_raw_values, index=df_calc.index).fillna(
                50)  # Fill NaNs from initial rolling min/max with 50
            stoch_k_s = stoch_k_raw_s.rolling(
                # Smooth %K
                window=self.config.stoch_smooth_k, min_periods=1).mean().fillna(50)
            stoch_d_s = stoch_k_s.rolling(window=self.config.stoch_smooth_d, min_periods=1).mean(
            ).fillna(50)  # Smooth %D (signal line)

            # ATR (Average True Range)
            # True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_range_s = pd.concat([
                high_s - low_s,
                (high_s - close_s.shift(1)).abs(),
                (low_s - close_s.shift(1)).abs()
            ], axis=1).max(axis=1).fillna(0)  # Fill initial NaN TR with 0
            # Using Exponential Moving Average for ATR (common)
            atr_s = true_range_s.ewm(
                span=self.config.atr_period, adjust=False).mean()

            # ADX, PDI (+DI), MDI (-DI)
            adx_s, pdi_s, mdi_s = self._calculate_adx(
                high_s, low_s, close_s, atr_s, self.config.adx_period)

            # VolumaticTrend (VT) Indicators
            vt_trend_ema_s_series = pd.Series(
                np.nan, index=df_calc.index)  # Initialize with NaNs
            vt_vwma_s_series = pd.Series(np.nan, index=df_calc.index)
            vt_volume_avg_s_series = pd.Series(np.nan, index=df_calc.index)
            vt_is_volume_spike_s_series = pd.Series(
                False, index=df_calc.index, dtype=bool)  # Default to False

            if self.config.vt_enable:
                vt_trend_ema_s_series = close_s.ewm(
                    span=self.config.vt_trend_ema_period, adjust=False).mean()

                # VWMA using pandas_ta
                if hasattr(df_calc.ta, 'vwma'):  # Check if vwma method exists
                    # Use fillna for consistency
                    vwma_result = df_calc.ta.vwma(
                        length=self.config.vt_vwma_period, fillna=np.nan)
                    if isinstance(vwma_result, pd.Series):
                        vt_vwma_s_series = vwma_result
                    else:
                        logger.warning(
                            "pandas_ta.vwma did not return a Series. VWMA indicator will be NaN.")
                else:
                    logger.error(
                        "pandas_ta.vwma not found (is pandas_ta installed and up-to-date?). VWMA calculation skipped (will be NaN).")

                # Volume Spike
                vt_volume_avg_s_series = volume_s.rolling(window=self.config.vt_volume_spike_lookback, min_periods=max(
                    1, self.config.vt_volume_spike_lookback // 2)).mean()
                # Ensure vt_volume_avg_s_series is not zero to avoid division by zero or misleading spikes on very low volume
                # A small epsilon (1e-9) could be used for average volume if it can be zero for valid assets.
                vt_is_volume_spike_s_series = (volume_s > vt_volume_avg_s_series * float(
                    # Check avg_vol > small_val
                    self.config.vt_volume_spike_multiplier)) & (vt_volume_avg_s_series > 1e-9)

            # Helper to get the latest valid (non-NaN) value from a Series as Decimal
            def get_latest_decimal(series: pd.Series, name: str) -> Decimal:
                valid_series = series.dropna()
                if not valid_series.empty:
                    return safe_decimal(str(valid_series.iloc[-1]))
                logger.warning(
                    f"Indicator series '{name}' is empty or all NaNs after dropna. Returning Decimal('NaN').")
                return Decimal("NaN")

            # Helper to get the latest valid (non-NaN) boolean value
            def get_latest_bool(series: pd.Series, name: str) -> bool:
                valid_series = series.dropna()
                if not valid_series.empty:
                    return bool(valid_series.iloc[-1])  # Ensure Python bool
                logger.warning(
                    f"Boolean indicator series '{name}' is empty or all NaNs after dropna. Returning False.")
                return False

            # Consolidate indicator results into a dictionary
            # pylint: disable=line-too-long
            indicators_out: Dict[str, Union[Decimal, bool, int]] = {
                "fast_ema": get_latest_decimal(fast_ema_s, "fast_ema"),
                "slow_ema": get_latest_decimal(slow_ema_s, "slow_ema"),
                "trend_ema": get_latest_decimal(trend_ema_s, "trend_ema"),
                "stoch_k": get_latest_decimal(stoch_k_s, "stoch_k"),
                "stoch_d": get_latest_decimal(stoch_d_s, "stoch_d"),
                "atr": get_latest_decimal(atr_s, "atr"),
                # Store period for reference, useful for display or debugging
                "atr_period": self.config.atr_period,
                "adx": get_latest_decimal(adx_s, "adx"),
                "pdi": get_latest_decimal(pdi_s, "pdi"),
                "mdi": get_latest_decimal(mdi_s, "mdi"),
                # VT Indicators
                "vt_trend_ema": get_latest_decimal(vt_trend_ema_s_series, "vt_trend_ema"),
                "vt_vwma": get_latest_decimal(vt_vwma_s_series, "vt_vwma"),
                "vt_volume_avg": get_latest_decimal(vt_volume_avg_s_series, "vt_volume_avg"),
                "vt_is_volume_spike": get_latest_bool(vt_is_volume_spike_s_series, "vt_is_volume_spike"),
                # VT Candle Color (based on latest candle O/C)
                "vt_candle_is_green": bool(close_s.iloc[-1] > open_s.iloc[-1]) if not close_s.empty and not open_s.empty and not pd.isna(close_s.iloc[-1]) and not pd.isna(open_s.iloc[-1]) else False,
                "vt_candle_is_red": bool(close_s.iloc[-1] < open_s.iloc[-1]) if not close_s.empty and not open_s.empty and not pd.isna(close_s.iloc[-1]) and not pd.isna(open_s.iloc[-1]) else False,
            }
            # pylint: enable=line-too-long

            # Previous Stochastic values for crossover detection
            stoch_k_valid_series = stoch_k_s.dropna()
            indicators_out["stoch_k_prev"] = get_latest_decimal(stoch_k_valid_series.iloc[:-1] if len(
                stoch_k_valid_series) >= 2 else pd.Series(dtype=float), "stoch_k_prev")
            stoch_d_valid_series = stoch_d_s.dropna()
            stoch_d_prev_val = get_latest_decimal(stoch_d_valid_series.iloc[:-1] if len(
                # Name it distinctly
                stoch_d_valid_series) >= 2 else pd.Series(dtype=float), "stoch_d_prev")

            # Stochastic K/D Crossover Logic
            # type: ignore[assignment]
            k_now, d_now, k_prev = indicators_out["stoch_k"], indicators_out["stoch_d"], indicators_out["stoch_k_prev"]
            indicators_out["stoch_kd_bullish"], indicators_out["stoch_kd_bearish"] = False, False
            # Ensure all four values are valid Decimals
            if not any(v.is_nan() for v in [k_now, d_now, k_prev, stoch_d_prev_val]):
                if (k_prev <= stoch_d_prev_val) and (k_now > d_now):
                    indicators_out["stoch_kd_bullish"] = True
                if (k_prev >= stoch_d_prev_val) and (k_now < d_now):
                    indicators_out["stoch_kd_bearish"] = True

            # Check for NaN in critical indicators that would break strategy logic
            critical_keys = ["fast_ema", "slow_ema", "trend_ema",
                             "atr", "stoch_k", "stoch_d", "adx", "pdi", "mdi"]
            if self.config.vt_enable:
                # Add VT criticals if enabled
                critical_keys.extend(["vt_trend_ema", "vt_vwma"])

            # Identify any critical indicators that are NaN
            failed_indicators = [
                k for k in critical_keys
                if not isinstance(indicators_out.get(k), (Decimal, bool, int)) or
                (isinstance(indicators_out.get(k), Decimal) and indicators_out.get(
                    k, Decimal("NaN")).is_nan())  # type: ignore[union-attr]
            ]
            if failed_indicators:
                # ATR is particularly critical for risk calculation
                # type: ignore[union-attr]
                if indicators_out.get("atr", Decimal("NaN")).is_nan():
                    logger.error(
                        f"{Fore.RED}CRITICAL INDICATOR FAILURE: ATR is NaN. Risk calculations will fail. Aborting indicator calculation cycle.{Style.RESET_ALL}")
                    return None
                logger.warning(
                    f"{Fore.YELLOW}Warning: Some critical indicators are NaN: {', '.join(failed_indicators)}. This may impair signal generation or risk calculation.{Style.RESET_ALL}")

            logger.info(
                f"{Style.BRIGHT}{Fore.GREEN}Indicator patterns woven successfully.{Style.RESET_ALL}")
            return indicators_out
        except Exception as e:
            logger.error(
                f"{Fore.RED}Error weaving indicator patterns: {e}{Style.RESET_ALL}", exc_info=True)
            return None

    def _calculate_adx(self, high_s: pd.Series, low_s: pd.Series, close_s: pd.Series, atr_s: pd.Series, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculates ADX (Average Directional Index), +DI (Positive Directional Indicator),
        and -DI (Negative Directional Indicator). Uses EMA-like smoothing (Wilder's RMA).
        """
        if period <= 0:
            logger.error("ADX period must be positive.")
            nan_s = pd.Series(np.nan, index=high_s.index)
            return nan_s, nan_s, nan_s
        if atr_s.empty or atr_s.isnull().all():  # ADX calculation relies on ATR
            logger.error(
                "ATR series is empty or all NaN; cannot calculate ADX.")
            nan_s = pd.Series(np.nan, index=high_s.index)
            return nan_s, nan_s, nan_s

        # Calculate +DM (Positive Directional Movement) and -DM (Negative Directional Movement)
        move_up = high_s.diff()  # Current high - previous high
        # Previous low - current low (negative of diff for positive move_down value)
        move_down = -low_s.diff()

        plus_dm_values = np.where(
            (move_up > move_down) & (move_up > 0), move_up, 0.0)
        minus_dm_values = np.where(
            (move_down > move_up) & (move_down > 0), move_down, 0.0)

        plus_dm_s = pd.Series(plus_dm_values, index=high_s.index).fillna(
            0)  # Fill initial NaN with 0
        minus_dm_s = pd.Series(minus_dm_values, index=high_s.index).fillna(0)

        # Smoothed DMs using EMA (Wilder's uses RMA, which is EMA with alpha=1/N)
        # For EMA, span relates to alpha as alpha = 2 / (span + 1). For Wilder's, alpha = 1 / N.
        alpha = 1.0 / period
        # Using ewm with alpha directly for Wilder's-like smoothing.
        # min_periods=period ensures enough data for a stable initial Wilder's MA.
        smoothed_plus_dm_s = plus_dm_s.ewm(
            alpha=alpha, adjust=False, min_periods=max(1, period)).mean().fillna(0)
        smoothed_minus_dm_s = minus_dm_s.ewm(
            alpha=alpha, adjust=False, min_periods=max(1, period)).mean().fillna(0)

        # Calculate +DI and -DI
        # Ensure ATR is not zero to avoid division by zero; default DI to 0 in such cases.
        pdi_values = np.where(
            (atr_s > 1e-12) & (~atr_s.isnull()), 100 * smoothed_plus_dm_s / atr_s, 0.0)
        mdi_values = np.where(
            (atr_s > 1e-12) & (~atr_s.isnull()), 100 * smoothed_minus_dm_s / atr_s, 0.0)
        pdi_s_out = pd.Series(pdi_values, index=high_s.index).fillna(0)
        mdi_s_out = pd.Series(mdi_values, index=high_s.index).fillna(0)

        # Calculate DX (Directional Movement Index)
        di_sum = pdi_s_out + mdi_s_out
        # Avoid division by zero if sum of DIs is zero; default DX to 0.
        dx_values = np.where(di_sum > 1e-12, 100 *
                             (pdi_s_out - mdi_s_out).abs() / di_sum, 0.0)
        dx_s = pd.Series(dx_values, index=high_s.index).fillna(0)

        # Calculate ADX (Smoothed DX)
        adx_s_out = dx_s.ewm(alpha=alpha, adjust=False,
                             min_periods=max(1, period)).mean().fillna(0)
        return adx_s_out, pdi_s_out, mdi_s_out

# --- Signal Generator Class ---


class SignalGenerator:
    """Generates trading signals based on indicators and strategy rules."""

    def __init__(self, config: TradingConfig):
        self.config = config

    def _generate_original_signals(self, df_last_candles: pd.DataFrame, indicators: Dict[str, Union[Decimal, bool, int]], current_price: Decimal) -> Tuple[bool, bool, str]:
        """Generates long/short signals based on the original EMA/Stoch/ADX/ATR strategy."""
        orig_long_sig, orig_short_sig = False, False
        reason = "Initializing original signal check"

        # Previous close for ATR move filter
        prev_close = safe_decimal(
            df_last_candles.iloc[-2]["close"]) if len(df_last_candles) >= 2 else Decimal("NaN")

        # Extract required indicators, ensuring they are valid Decimals
        required_keys = ["stoch_k", "fast_ema", "slow_ema",
                         "trend_ema", "atr", "adx", "pdi", "mdi"]
        ind_values: Dict[str, Decimal] = {}
        nan_keys = [k for k in required_keys if not isinstance(indicators.get(
            # type: ignore[union-attr]
            k), Decimal) or indicators.get(k, Decimal("NaN")).is_nan()]
        if nan_keys:
            return False, False, f"Original Signal Fail: Critical indicators NaN/Missing: {', '.join(nan_keys)}."

        for key in required_keys:
            ind_values[key] = indicators[key]  # type: ignore[assignment]
        # Unpack validated decimal indicators
        k, fast_ema, slow_ema, trend_ema, atr, adx, pdi, mdi = (
            ind_values[key] for key in required_keys)
        # Get boolean stochastic cross signals
        stoch_kd_bull_cross, stoch_kd_bear_cross = bool(indicators.get(
            "stoch_kd_bullish", False)), bool(indicators.get("stoch_kd_bearish", False))

        # EMA Crossover
        ema_bullish_cross, ema_bearish_cross = fast_ema > slow_ema, fast_ema < slow_ema
        ema_cross_state_str = "Bullish" if ema_bullish_cross else "Bearish" if ema_bearish_cross else "Neutral"

        # Trend Filter (based on `trend_ema`)
        trend_buffer_abs = trend_ema.copy_abs(
        ) * (self.config.trend_filter_buffer_percent / Decimal(100))
        price_above_trend_ema_buffered = current_price > (
            trend_ema - trend_buffer_abs)  # Price must be above lower band of trend EMA
        price_below_trend_ema_buffered = current_price < (
            trend_ema + trend_buffer_abs)  # Price must be below upper band of trend EMA
        trend_allows_long = price_above_trend_ema_buffered if self.config.trade_only_with_trend else True
        trend_allows_short = price_below_trend_ema_buffered if self.config.trade_only_with_trend else True
        trend_filter_reason_suffix = f"(Price:{current_price:.{DEFAULT_PRICE_DP}f} vs TrendEMA:{trend_ema:.{DEFAULT_PRICE_DP}f} ± {trend_buffer_abs:.{DEFAULT_PRICE_DP}f})" if self.config.trade_only_with_trend else "(TrendFilter OFF)"

        # Stochastic Condition (Oversold/Overbought or K/D Crossover)
        stoch_long_cond = (
            k < self.config.stoch_oversold_threshold) or stoch_kd_bull_cross
        stoch_short_cond = (
            k > self.config.stoch_overbought_threshold) or stoch_kd_bear_cross
        stoch_reason_suffix = f"StochK:{k:.1f} (OS:{self.config.stoch_oversold_threshold.normalize()}/OB:{self.config.stoch_overbought_threshold.normalize()}), KD_Cross(Bull:{stoch_kd_bull_cross}/Bear:{stoch_kd_bear_cross})"

        # ATR Move Filter (Price must move more than a fraction of ATR)
        significant_move_filter_passed, atr_filter_reason_suffix = True, "(ATR MoveFilter OFF)"
        if self.config.atr_move_filter_multiplier > Decimal("0"):
            if atr.is_nan() or atr <= Decimal("0"):
                atr_filter_reason_suffix, significant_move_filter_passed = f"(ATR Invalid:{atr.normalize() if not atr.is_nan() else 'NaN'})", False
            elif prev_close.is_nan() or prev_close <= Decimal("0"):  # Requires valid previous close
                atr_filter_reason_suffix, significant_move_filter_passed = f"(PrevClose Invalid:{prev_close.normalize() if not prev_close.is_nan() else 'NaN'})", False
            else:
                atr_move_threshold = atr * self.config.atr_move_filter_multiplier
                price_move_abs = (current_price - prev_close).copy_abs()
                significant_move_filter_passed = price_move_abs > atr_move_threshold
                atr_filter_reason_suffix = f"(PriceMove:{price_move_abs:.{DEFAULT_PRICE_DP}f} {'OK' if significant_move_filter_passed else 'LOW'} vs Threshold:{atr_move_threshold:.{DEFAULT_PRICE_DP}f})"

        # ADX Filter (Trend strength and direction)
        adx_trend_is_strong = adx > self.config.min_adx_level
        # +DI > -DI for long, -DI > +DI for short
        adx_long_direction_favored, adx_short_direction_favored = pdi > mdi, mdi > pdi
        adx_allows_long = adx_trend_is_strong and adx_long_direction_favored
        adx_allows_short = adx_trend_is_strong and adx_short_direction_favored
        adx_filter_reason_suffix = f"(ADX:{adx:.1f} {'Strong' if adx_trend_is_strong else 'Weak'}/{self.config.min_adx_level.normalize()} | Direction:{'+DI>-DI' if adx_long_direction_favored else '-DI>+DI' if adx_short_direction_favored else 'Neutral'})"

        # Combine conditions for final original strategy signals
        base_long_conditions_met = ema_bullish_cross and stoch_long_cond
        base_short_conditions_met = ema_bearish_cross and stoch_short_cond

        orig_long_sig = base_long_conditions_met and trend_allows_long and significant_move_filter_passed and adx_allows_long
        orig_short_sig = base_short_conditions_met and trend_allows_short and significant_move_filter_passed and adx_allows_short

        # Construct detailed reason string
        if orig_long_sig:
            reason = f"Orig Long: EMA_X {ema_cross_state_str} & {stoch_reason_suffix} & TrendOK {trend_filter_reason_suffix} & ATRMoveOK {atr_filter_reason_suffix} & ADX_OK {adx_filter_reason_suffix}"
        elif orig_short_sig:
            reason = f"Orig Short: EMA_X {ema_cross_state_str} & {stoch_reason_suffix} & TrendOK {trend_filter_reason_suffix} & ATRMoveOK {atr_filter_reason_suffix} & ADX_OK {adx_filter_reason_suffix}"
        else:  # No signal, provide reasons for failure
            parts = [
                f"No Orig Signal: BaseCond(EMA_X:{ema_cross_state_str}, {stoch_reason_suffix}) -> LongBase:{base_long_conditions_met}, ShortBase:{base_short_conditions_met}."]
            if base_long_conditions_met and not trend_allows_long:
                parts.append(
                    f"LongBlocked:TrendFail {trend_filter_reason_suffix}.")
            if base_short_conditions_met and not trend_allows_short:
                parts.append(
                    f"ShortBlocked:TrendFail {trend_filter_reason_suffix}.")
            if (base_long_conditions_met or base_short_conditions_met) and not significant_move_filter_passed:
                parts.append(
                    f"Blocked:ATRMoveFail {atr_filter_reason_suffix}.")
            if base_long_conditions_met and not adx_allows_long:
                parts.append(
                    f"LongBlocked:ADXFail {adx_filter_reason_suffix}.")
            if base_short_conditions_met and not adx_allows_short:
                parts.append(
                    f"ShortBlocked:ADXFail {adx_filter_reason_suffix}.")
            reason = " ".join(parts)
        return orig_long_sig, orig_short_sig, reason

    def _generate_vt_signals(self, indicators: Dict[str, Union[Decimal, bool, int]], current_price: Decimal) -> Tuple[bool, bool, str]:
        """Generates long/short signals based on the VolumaticTrend (VT) strategy component."""
        if not self.config.vt_enable:
            return False, False, "VT Strategy Disabled by Configuration"

        reason = "Initializing VT signal check"
        # VT specific required decimal indicators
        required_vt_decimal_keys = ["vt_trend_ema", "vt_vwma"]
        # VT specific boolean indicators
        required_vt_bool_keys = ["vt_is_volume_spike",
                                 "vt_candle_is_green", "vt_candle_is_red"]

        ind_values_vt_decimal: Dict[str, Decimal] = {}
        # Validate decimal indicators
        nan_decimal_keys = [k for k in required_vt_decimal_keys if not isinstance(indicators.get(
            # type: ignore[union-attr]
            k), Decimal) or indicators.get(k, Decimal("NaN")).is_nan()]
        # Validate boolean indicators (check type, as value itself is bool)
        # Handle cases where indicator might be numpy.bool_ instead of Python's native bool
        invalid_bool_keys = [k for k in required_vt_bool_keys if not isinstance(
            indicators.get(k), (bool, np.bool_))]

        if nan_decimal_keys or invalid_bool_keys:
            missing_details = []
            if nan_decimal_keys:
                missing_details.append(
                    f"VT NaN/Missing Decimal Indicators: {', '.join(nan_decimal_keys)}")
            if invalid_bool_keys:
                missing_details.append(
                    f"VT Invalid Boolean Indicator Types: {', '.join(invalid_bool_keys)}")
            return False, False, f"VT Signal Fail: {'; '.join(missing_details)}."

        for key_vt in required_vt_decimal_keys:
            # type: ignore[assignment]
            ind_values_vt_decimal[key_vt] = indicators[key_vt]
        vt_trend_ema, vt_vwma = ind_values_vt_decimal["vt_trend_ema"], ind_values_vt_decimal["vt_vwma"]
        # Ensure Python bool type for logic
        is_vol_spike, is_green_candle, is_red_candle = bool(indicators["vt_is_volume_spike"]), bool(
            indicators["vt_candle_is_green"]), bool(indicators["vt_candle_is_red"])

        # VT Long Signal Conditions
        price_above_vt_trend_ema = current_price > vt_trend_ema
        price_above_vwma = current_price > vt_vwma
        vt_long_sig = price_above_vt_trend_ema and price_above_vwma and is_vol_spike and is_green_candle

        # VT Short Signal Conditions
        price_below_vt_trend_ema = current_price < vt_trend_ema
        price_below_vwma = current_price < vt_vwma
        vt_short_sig = price_below_vt_trend_ema and price_below_vwma and is_vol_spike and is_red_candle

        # Construct detailed reason string for VT signals
        if vt_long_sig:
            reason = (
                f"VT Long: Price > {vt_trend_ema:.{DEFAULT_PRICE_DP}f} (TrendEMA) & Price > {vt_vwma:.{DEFAULT_PRICE_DP}f} (VWMA) & VolumeSpike & GreenCandle")
        elif vt_short_sig:
            reason = (
                f"VT Short: Price < {vt_trend_ema:.{DEFAULT_PRICE_DP}f} (TrendEMA) & Price < {vt_vwma:.{DEFAULT_PRICE_DP}f} (VWMA) & VolumeSpike & RedCandle")
        else:  # No VT signal, provide context
            trend_status_str = "Above" if price_above_vt_trend_ema else "Below" if price_below_vt_trend_ema else "Neutral"
            vwma_status_str = "Above" if price_above_vwma else "Below" if price_below_vwma else "Neutral"
            candle_color_str = "Green" if is_green_candle else "Red" if is_red_candle else "Neutral"
            reason = (f"No VT Signal: Trend(Price:{current_price:.{DEFAULT_PRICE_DP}f} vs EMA:{vt_trend_ema:.{DEFAULT_PRICE_DP}f} -> {trend_status_str}), "
                      f"VWMA(Price:{current_price:.{DEFAULT_PRICE_DP}f} vs VWMA:{vt_vwma:.{DEFAULT_PRICE_DP}f} -> {vwma_status_str}), "
                      f"VolumeSpike:{is_vol_spike}, CandleColor:{candle_color_str}")
        return vt_long_sig, vt_short_sig, reason

    def generate_signals(self, df_last_candles: pd.DataFrame, indicators: Dict[str, Union[Decimal, bool, int]]) -> Dict[str, Any]:
        """
        Generates final trading signals by combining Original and VolumaticTrend (VT) strategies.
        Returns a dictionary with 'long', 'short' booleans and 'summary', 'orig_detail', 'vt_detail' strings.
        """
        result: Dict[str, Any] = {
            "long": False, "short": False,
            "summary": "Initializing signal generation...",
            "orig_detail": "N/A", "vt_detail": "N/A"
        }
        if not indicators:
            result.update(
                {"summary": "No Signal: Indicators data is missing."})
            logger.debug(result["summary"])
            return result

        # Need at least 2 candles for some filters (e.g., ATR move filter using previous close)
        if df_last_candles is None or len(df_last_candles) < 2:
            reason_no_candle = f"No Signal: Insufficient candle data (requires >=2, received {len(df_last_candles) if df_last_candles is not None else 0}). Some filters may not operate."
            result.update({"summary": reason_no_candle})
            logger.debug(reason_no_candle)
            return result

        try:
            current_price = safe_decimal(
                df_last_candles.iloc[-1]["close"])  # Latest close price
            if current_price.is_nan() or current_price <= Decimal("0"):
                reason_invalid_price = f"No Signal: Invalid current price ({current_price.normalize() if not current_price.is_nan() else 'NaN'}). Cannot generate signals."
                result.update({"summary": reason_invalid_price})
                logger.warning(reason_invalid_price)
                return result

            # Generate signals from the original strategy
            orig_long, orig_short, orig_reason = self._generate_original_signals(
                df_last_candles, indicators, current_price)
            result["orig_detail"] = orig_reason

            # Generate signals from the VolumaticTrend strategy (if enabled)
            vt_long, vt_short, vt_reason = self._generate_vt_signals(
                indicators, current_price)
            result["vt_detail"] = vt_reason if self.config.vt_enable else "VT Disabled"

            # Combine signals
            final_long, final_short = False, False
            summary_status = "No Signal"  # Default status

            # Orig Long, and (VT disabled or VT Long agrees)
            if orig_long and (not self.config.vt_enable or vt_long):
                final_long = True
                summary_status = "Long Signal (Orig Confirmed" + (
                    ", VT Agreed" if self.config.vt_enable and vt_long else "" if not self.config.vt_enable else ", VT Neutral/NoSignal") + ")"
            # Orig Short, and (VT disabled or VT Short agrees)
            elif orig_short and (not self.config.vt_enable or vt_short):
                final_short = True
                summary_status = "Short Signal (Orig Confirmed" + (
                    ", VT Agreed" if self.config.vt_enable and vt_short else "" if not self.config.vt_enable else ", VT Neutral/NoSignal") + ")"
            elif self.config.vt_enable and vt_long and not orig_long and not orig_short:  # VT Long, Orig Neutral
                final_long = True
                summary_status = "Long Signal (VT Only, Orig Neutral)"
            elif self.config.vt_enable and vt_short and not orig_long and not orig_short:  # VT Short, Orig Neutral
                final_short = True
                summary_status = "Short Signal (VT Only, Orig Neutral)"
            # Conflict
            elif self.config.vt_enable and ((orig_long and vt_short) or (orig_short and vt_long)):
                summary_status = f"Signal Conflict: Orig({'L' if orig_long else 'S' if orig_short else 'N'}) vs VT({'L' if vt_long else 'S' if vt_short else 'N'}). Holding."
            else:  # No definitive signal or agreement leading to a trade
                summary_status = "No Combined Signal (Orig & VT Neutral or Non-Confirming)"

            result["long"] = final_long
            result["short"] = final_short
            result["summary"] = summary_status

            # Log signal check results: INFO for actual signals/conflicts, DEBUG for no signal/neutral.
            log_level = logging.INFO if final_long or final_short or "Conflict" in summary_status or "Block" in orig_reason or "Block" in vt_reason else logging.DEBUG
            logger.log(
                log_level, f"Signal Check: {summary_status} | Orig: {orig_reason} | VT: {vt_reason}")

        except Exception as e:
            logger.error(
                f"{Fore.RED}Error during entry signal generation: {e}{Style.RESET_ALL}", exc_info=True)
            result.update({"summary": f"No Signal: Exception during generation ({type(e).__name__})",
                          "long": False, "short": False, "orig_detail": str(e), "vt_detail": "N/A"})
        return result

    def check_exit_signals(self, position_side: str, indicators: Dict[str, Union[Decimal, bool, int]]) -> Optional[str]:
        """
        Checks for exit signals for an active position based on Original and VT strategies.
        Returns a reason string if an exit signal is found, otherwise None.
        """
        if not indicators:
            logger.warning(
                "Cannot check exit signals: indicators data missing.")
            return None

        # Original Strategy Exit Signals
        orig_exit_reason: Optional[str] = None
        fast_ema_val, slow_ema_val = indicators.get(
            "fast_ema"), indicators.get("slow_ema")
        stoch_k_curr, stoch_k_prev = indicators.get("stoch_k"), indicators.get(
            "stoch_k_prev")  # Current and previous %K

        # Ensure all required Decimal indicators for original exit are valid
        orig_req_decs = {"fast_ema": fast_ema_val, "slow_ema": slow_ema_val,
                         "stoch_k": stoch_k_curr, "stoch_k_prev": stoch_k_prev}
        orig_inds_valid = all(isinstance(v, Decimal) and not v.is_nan()
                              for v in orig_req_decs.values())

        if orig_inds_valid:
            # type: ignore[assignment]
            fast_ema, slow_ema, k_curr, k_prev = fast_ema_val, slow_ema_val, stoch_k_curr, stoch_k_prev
            ema_bullish_cross, ema_bearish_cross = fast_ema > slow_ema, fast_ema < slow_ema
            os_lvl, ob_lvl = self.config.stoch_oversold_threshold, self.config.stoch_overbought_threshold

            if position_side == "long":
                if ema_bearish_cross:  # EMA bearish crossover
                    orig_exit_reason = f"Orig Exit (Long): EMA Bearish Cross (FastEMA {fast_ema.normalize()} < SlowEMA {slow_ema.normalize()})"
                elif k_prev >= ob_lvl and k_curr < ob_lvl:  # Stochastic %K crosses down from overbought
                    orig_exit_reason = f"Orig Exit (Long): Stoch Reversal from Overbought (PrevK {k_prev.normalize():.1f} >= OB -> CurrK {k_curr.normalize():.1f} < OB)"
            elif position_side == "short":
                if ema_bullish_cross:  # EMA bullish crossover
                    orig_exit_reason = f"Orig Exit (Short): EMA Bullish Cross (FastEMA {fast_ema.normalize()} > SlowEMA {slow_ema.normalize()})"
                elif k_prev <= os_lvl and k_curr > os_lvl:  # Stochastic %K crosses up from oversold
                    orig_exit_reason = f"Orig Exit (Short): Stoch Reversal from Oversold (PrevK {k_prev.normalize():.1f} <= OS -> CurrK {k_curr.normalize():.1f} > OS)"

        if orig_exit_reason:
            logger.trade(f"{Fore.YELLOW}{orig_exit_reason}{Style.RESET_ALL}")
            return orig_exit_reason  # Prioritize original exit

        # VolumaticTrend (VT) Strategy Exit Signals (if enabled and no original exit yet)
        if self.config.vt_enable:
            vt_exit_reason: Optional[str] = None
            # VT exit signals require current price, VT trend EMA, VT VWMA, and candle/volume states
            # Current close price added to indicators dict
            price_val = indicators.get("close_price")
            vt_trend_ema_val, vt_vwma_val = indicators.get(
                "vt_trend_ema"), indicators.get("vt_vwma")
            # Boolean indicators for VT exit
            is_vol_spike_val = indicators.get(
                "vt_is_volume_spike", False)  # Default to False if missing
            is_red_cdl_val = indicators.get("vt_candle_is_red", False)
            is_green_cdl_val = indicators.get("vt_candle_is_green", False)

            vt_req_decs = {
                "price": price_val, "vt_trend_ema": vt_trend_ema_val, "vt_vwma": vt_vwma_val}
            vt_decs_valid = all(isinstance(v, Decimal) and not v.is_nan()
                                for v in vt_req_decs.values())
            # Ensure boolean types are correct (can be np.bool_ from pandas)
            vt_bools_valid = all(isinstance(v, (bool, np.bool_)) for v in [
                                 is_vol_spike_val, is_red_cdl_val, is_green_cdl_val])

            if vt_decs_valid and vt_bools_valid:
                # type: ignore[assignment]
                price, vt_trend_ema, vt_vwma = price_val, vt_trend_ema_val, vt_vwma_val
                is_vol_spike, is_red_cdl, is_green_cdl = bool(
                    is_vol_spike_val), bool(is_red_cdl_val), bool(is_green_cdl_val)

                if position_side == "long":
                    # Exit long if price crosses below VT Trend EMA
                    if price < vt_trend_ema:
                        # type: ignore[union-attr]
                        vt_exit_reason = f"VT Exit (Long): Price < VT Trend EMA ({vt_trend_ema.normalize()})"
                    # Exit long if price crosses below VT VWMA on a volume spike with a red candle
                    elif price < vt_vwma and is_vol_spike and is_red_cdl:
                        # type: ignore[union-attr]
                        vt_exit_reason = f"VT Exit (Long): Price < VT VWMA ({vt_vwma.normalize()}) + VolumeSpike on RedCandle"
                elif position_side == "short":
                    # Exit short if price crosses above VT Trend EMA
                    if price > vt_trend_ema:
                        # type: ignore[union-attr]
                        vt_exit_reason = f"VT Exit (Short): Price > VT Trend EMA ({vt_trend_ema.normalize()})"
                    # Exit short if price crosses above VT VWMA on a volume spike with a green candle
                    elif price > vt_vwma and is_vol_spike and is_green_cdl:
                        # type: ignore[union-attr]
                        vt_exit_reason = f"VT Exit (Short): Price > VT VWMA ({vt_vwma.normalize()}) + VolumeSpike on GreenCandle"

            if vt_exit_reason:
                logger.trade(f"{Fore.YELLOW}{vt_exit_reason}{Style.RESET_ALL}")
                return vt_exit_reason

        return None  # No exit signal generated by either strategy

# --- Order Manager Class ---


class OrderManager:
    """Manages order creation, modification (SL/TP/TSL), and position verification."""

    def __init__(self, config: TradingConfig, exchange_manager: ExchangeManager):
        self.config = config
        self.exchange_manager = exchange_manager
        # Ensure critical components are available from ExchangeManager
        if not exchange_manager or not exchange_manager.exchange or not exchange_manager.market_info:
            err_msg = "OrderManager initialization failed: A valid ExchangeManager with an initialized exchange and loaded market_info is required."
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}{err_msg}{Style.RESET_ALL}")
            raise ValueError(err_msg)

        self.exchange = exchange_manager.exchange  # Convenience access
        self.market_info = exchange_manager.market_info  # Convenience access

        # Tracks the current protection type (SL/TP or TSL) active for long/short positions
        # Values can be PROTECTION_STATE_SLTP, PROTECTION_STATE_TSL, or None
        self.protection_tracker: Dict[str, Optional[Literal[PROTECTION_STATE_SLTP, PROTECTION_STATE_TSL]]] = {
            "long": None, "short": None}

    def _calculate_trade_parameters(self, side: str, atr: Decimal, total_equity: Decimal, current_price: Decimal) -> Optional[Dict[str, Optional[Decimal]]]:
        """
        Calculates trade parameters: quantity, stop-loss, take-profit, and TSL distance.
        Returns a dictionary of parameters or None if calculation fails.
        `side` is "buy" or "sell".
        """
        # Validate inputs
        if atr.is_nan() or atr <= Decimal("0"):
            logger.error(
                f"Invalid ATR value ({atr.normalize() if not atr.is_nan() else 'NaN'}) for trade parameter calculation.")
            return None
        if total_equity.is_nan() or total_equity <= Decimal("0"):
            logger.error(
                f"Invalid total equity ({total_equity.normalize() if not total_equity.is_nan() else 'NaN'}) for parameter calculation.")
            return None
        if current_price.is_nan() or current_price <= Decimal("0"):
            logger.error(
                f"Invalid current price ({current_price.normalize() if not current_price.is_nan() else 'NaN'}) for parameter calculation.")
            return None

        # Market-specific info needed for calculations
        mkt_tick_size = self.market_info.get('tick_size', Decimal('NaN'))
        mkt_contract_size = self.market_info.get(
            # In base currency units per contract
            'contract_size', Decimal('NaN'))
        # In contracts or base currency units depending on market
        mkt_min_order_size = self.market_info.get(
            'min_order_size', Decimal('NaN'))

        if not self.market_info or any(v.is_nan() for v in [mkt_tick_size, mkt_contract_size, mkt_min_order_size]):
            logger.error(
                "Market information (tick_size, contract_size, min_order_size) is incomplete or invalid. Cannot calculate trade parameters.")
            return None
        if side not in ("buy", "sell"):
            logger.error(
                f"Invalid trade side '{side}' for parameter calculation.")
            return None

        try:
            # 1. Calculate Risk Amount in Settle Currency
            risk_amt_settle_ccy = total_equity * self.config.risk_percentage

            # 2. Calculate Stop Loss Price
            # SL distance in price points (based on ATR)
            sl_dist_atr_pts = atr * self.config.sl_atr_multiplier
            sl_price_calc = current_price - \
                sl_dist_atr_pts if side == "buy" else current_price + sl_dist_atr_pts
            # SL price cannot be zero or negative
            if sl_price_calc <= Decimal("0"):
                logger.error(
                    f"Calculated Stop Loss price ({sl_price_calc:.{DEFAULT_PRICE_DP}f}) is invalid (zero or negative).")
                return None

            # Ensure SL distance is at least one tick size
            sl_dist_abs = (current_price - sl_price_calc).copy_abs()
            if mkt_tick_size <= Decimal("0"):
                logger.error("Market tick_size is invalid (zero or negative).")
                return None  # Should be caught by earlier check
            if sl_dist_abs < mkt_tick_size:
                logger.warning(
                    f"Calculated SL distance ({sl_dist_abs.normalize()}) is less than market tick size ({mkt_tick_size.normalize()}). Adjusting SL distance to one tick size.")
                sl_dist_abs = mkt_tick_size
                # Recalculate SL price based on adjusted distance
                sl_price_calc = current_price - \
                    sl_dist_abs if side == "buy" else current_price + sl_dist_abs
                if sl_price_calc <= Decimal("0"):
                    logger.error(
                        f"Adjusted SL price ({sl_price_calc:.{DEFAULT_PRICE_DP}f}) is still invalid.")
                    return None
            if sl_dist_abs <= Decimal("0"):
                logger.error(
                    f"SL distance ({sl_dist_abs.normalize()}) is invalid (zero or negative).")
                return None

            # 3. Calculate Position Quantity
            # Qty calculation differs for linear vs. inverse contracts
            qty_calc_raw: Decimal  # Raw calculated quantity before formatting
            if self.config.market_type == "inverse":
                # For inverse contracts (e.g., BTC/USD settled in BTC), quantity is typically in base currency.
                # Risk (Quote) / SL_distance_per_BaseUnit (Quote) = Quantity (Base)
                if current_price <= Decimal("0"):
                    logger.error(
                        "Invalid current_price for inverse quantity calculation.")
                    return None
                qty_calc_raw = risk_amt_settle_ccy / sl_dist_abs
            else:  # Linear contracts (e.g., BTC/USDT settled in USDT)
                # Quantity is typically in contracts. Contract size is in base currency units per contract.
                # Risk (Quote) / (SL_distance_per_BaseUnit (Quote) * ContractSize (Base/Contract)) = Quantity (Contracts)
                # Or, Risk (Quote) / Risk_per_Contract (Quote) = Quantity (Contracts)
                # Risk_per_Contract = SL_distance_abs (Quote/BaseUnit) * ContractSize (Base/Contract)
                risk_per_contract_settle_ccy = sl_dist_abs * mkt_contract_size
                if risk_per_contract_settle_ccy <= Decimal("0"):
                    logger.error(
                        f"Calculated risk per contract ({risk_per_contract_settle_ccy.normalize()}) is zero or negative. Cannot determine quantity.")
                    return None
                qty_calc_raw = risk_amt_settle_ccy / risk_per_contract_settle_ccy

            # Format quantity to market precision and check against min order size
            qty_str_fmt = self.exchange_manager.format_amount(
                qty_calc_raw, ROUND_DOWN)  # Always round down quantity
            qty_final_dec = safe_decimal(qty_str_fmt)
            if qty_final_dec.is_nan() or qty_final_dec <= Decimal("0"):
                logger.error(
                    f"Calculated quantity ('{qty_str_fmt}') is invalid or zero after formatting. Original raw: {qty_calc_raw.normalize()}")
                return None
            if qty_final_dec < mkt_min_order_size:
                logger.error(
                    f"Calculated quantity {qty_final_dec.normalize()} is less than minimum order size {mkt_min_order_size.normalize()}. Cannot place trade.")
                return None

            # 4. Calculate Take Profit Price (if enabled)
            tp_price_calc: Optional[Decimal] = None
            if self.config.tp_atr_multiplier > Decimal("0"):
                tp_dist_atr_pts = atr * self.config.tp_atr_multiplier
                tp_price_calc = current_price + \
                    tp_dist_atr_pts if side == "buy" else current_price - tp_dist_atr_pts
                if tp_price_calc <= Decimal("0"):
                    logger.warning(
                        f"Calculated Take Profit price ({tp_price_calc:.{DEFAULT_PRICE_DP}f}) is invalid (zero or negative). Disabling TP for this trade.")
                    tp_price_calc = None

            # 5. Calculate Trailing Stop Loss (TSL) Distance (not activation price yet)
            # TSL distance is based on percentage of current price
            tsl_dist_pts_calc = current_price * \
                (self.config.trailing_stop_percent / Decimal(100))
            if tsl_dist_pts_calc < mkt_tick_size:  # Ensure TSL distance is at least one tick
                logger.debug(
                    f"Calculated TSL distance ({tsl_dist_pts_calc.normalize()}) is less than market tick size ({mkt_tick_size.normalize()}). Adjusting to tick size.")
                tsl_dist_pts_calc = mkt_tick_size

            # Format TSL distance (like a price difference)
            tsl_dist_str_fmt = self.exchange_manager.format_price(
                tsl_dist_pts_calc)  # TSL distance is formatted like a price for Bybit
            tsl_dist_final_dec = safe_decimal(tsl_dist_str_fmt)
            if tsl_dist_final_dec.is_nan() or tsl_dist_final_dec <= Decimal("0"):
                logger.warning(
                    f"Formatted TSL distance ('{tsl_dist_str_fmt}') is invalid. TSL might fail or be disabled. Original raw: {tsl_dist_pts_calc.normalize()}")
                tsl_dist_final_dec = Decimal('NaN')  # Store as NaN if invalid

            # Format SL and TP prices
            sl_price_str_fmt = self.exchange_manager.format_price(
                sl_price_calc)
            sl_price_final_dec = safe_decimal(sl_price_str_fmt)
            if sl_price_final_dec.is_nan() or sl_price_final_dec <= Decimal("0"):
                logger.error(
                    f"Formatted Stop Loss price ('{sl_price_str_fmt}') is invalid.")
                return None

            tp_price_final_dec: Optional[Decimal] = None
            if tp_price_calc is not None:  # Only format if TP was calculated and valid
                tp_price_str_fmt = self.exchange_manager.format_price(
                    tp_price_calc)
                tp_price_final_dec = safe_decimal(tp_price_str_fmt)
                if tp_price_final_dec.is_nan() or tp_price_final_dec <= Decimal("0"):
                    logger.warning(
                        f"Formatted Take Profit price ('{tp_price_str_fmt}') is invalid. Disabling TP.")
                    tp_price_final_dec = None

            # Consolidate parameters
            params_out: Dict[str, Optional[Decimal]] = {
                "qty": qty_final_dec,
                "sl_price": sl_price_final_dec,
                "tp_price": tp_price_final_dec,  # Can be None
                # Store None if NaN
                "tsl_distance": tsl_dist_final_dec if not tsl_dist_final_dec.is_nan() else None
            }

            settle_ccy = self.market_info.get('settle', 'SETTLE_CURRENCY')
            base_ccy = self.market_info.get('base', 'BASE_CURRENCY')
            logger.info(
                f"Calculated Trade Parameters ({side.upper()}): "
                # type: ignore
                f"Qty={params_out['qty'].normalize() if params_out['qty'] else 'N/A'} {base_ccy}, "
                # type: ignore
                f"EntryPx (approx)={current_price.normalize():.{DEFAULT_PRICE_DP}f}, SL={params_out['sl_price'].normalize() if params_out['sl_price'] else 'N/A'}, "
                # type: ignore
                f"TP={'Disabled' if not params_out['tp_price'] else params_out['tp_price'].normalize()}, "
                # type: ignore
                f"TSL_Dist={'N/A' if not params_out['tsl_distance'] else params_out['tsl_distance'].normalize()}, "
                f"RiskAmt (approx)={risk_amt_settle_ccy.normalize():.{DEFAULT_PRICE_DP}f} {settle_ccy}, ATR={atr.normalize():.{DEFAULT_PRICE_DP+1}f}"
            )
            return params_out
        except (InvalidOperation, DivisionByZero, TypeError, Exception) as e:
            logger.error(
                f"Error calculating trade parameters for {side.upper()}: {e}", exc_info=True)
            return None

    def _execute_market_order(self, side: str, qty_decimal: Decimal) -> Optional[Dict[str, Any]]:
        """Executes a market order and returns the order response from the exchange."""
        if not self.exchange or not self.market_info:  # Should be caught by calling methods
            logger.error(
                "Market order execution failed: Exchange or Market info missing.")
            return None

        # Format quantity to API string, ensuring it's valid
        qty_str_api = self.exchange_manager.format_amount(
            qty_decimal, rounding_mode=ROUND_DOWN)  # Round down for safety
        final_qty_dec_for_log = safe_decimal(qty_str_api)
        if final_qty_dec_for_log.is_nan() or final_qty_dec_for_log <= Decimal("0"):
            logger.error(
                f"Market order aborted: Formatted quantity '{qty_str_api}' is zero or invalid (Original Decimal: {qty_decimal.normalize()}).")
            return None

        try:  # CCXT expects amount as float for create_market_order
            amount_float_for_ccxt = float(qty_str_api)
        except ValueError:
            logger.error(
                f"Could not convert formatted quantity '{qty_str_api}' to float for API call. Aborting market order.")
            return None

        logger.trade(f"{Fore.CYAN}Attempting MARKET {side.upper()} order: {final_qty_dec_for_log.normalize()} {self.market_info.get('base', 'BASE_CCY')} for {self.config.symbol}...{Style.RESET_ALL}")
        try:
            # Bybit V5 specific parameters for market order
            params_v5 = {
                "category": self.config.bybit_v5_category,
                "positionIdx": self.config.position_idx,
                # Ensures it doesn't sit on book if not fillable (good for market orders)
                "timeInForce": "ImmediateOrCancel"
            }
            order_resp = fetch_with_retries(
                self.exchange.create_market_order,
                symbol=self.config.symbol, side=side, amount=amount_float_for_ccxt, params=params_v5,
                max_retries=self.config.max_fetch_retries,  # Use configured retries
                delay_seconds=self.config.retry_delay_seconds
            )
            if order_resp is None:  # fetch_with_retries might return None if all retries fail without specific exceptions
                logger.error(
                    f"{Fore.RED}Market order submission failed (API call returned None after retries).{Style.RESET_ALL}")
                return None

            # Extract key details from response
            order_id = order_resp.get("id", "[ORDER_ID_N/A]")
            status = order_resp.get("status", "[STATUS_UNKNOWN]")
            filled_qty = safe_decimal(
                order_resp.get("filled", "0"))  # Amount filled
            avg_px = safe_decimal(order_resp.get(
                "average", "0"))  # Average fill price
            avg_px_log_str = avg_px.normalize() if not avg_px.is_nan(
            ) and avg_px > Decimal("0") else "[AVG_PRICE_N/A]"

            logger.trade(f"{Style.BRIGHT}{Fore.GREEN}Market order submitted: ID {order_id}, Side {side.upper()}, Ordered {final_qty_dec_for_log.normalize()}, API_Status: {status}, FilledQty: {filled_qty.normalize()}, AvgFillPx: {avg_px_log_str}{Style.RESET_ALL}")
            termux_notify(f"{self.config.symbol} Order Submitted",
                          f"Market {side.upper()} {final_qty_dec_for_log.normalize()} ID:{order_id}, Status:{status}")

            # Handle specific non-successful statuses
            if status == "rejected":
                # Try to get more specific rejection reason from 'info'
                reject_reason = order_resp.get('info', {}).get('rejectReason', order_resp.get(
                    'info', {}).get('retMsg', 'No specific rejection reason provided'))
                logger.error(
                    f"{Fore.RED}Market order {order_id} REJECTED by exchange. Reason: '{reject_reason}'. Full info (truncated): {str(order_resp.get('info'))[:200]}{Style.RESET_ALL}")
                return None
            if status == "canceled" and filled_qty == Decimal("0") and params_v5.get("timeInForce") == "ImmediateOrCancel":
                # This is expected if an IOC market order cannot be filled immediately (e.g., no liquidity)
                logger.error(
                    f"{Fore.RED}Market order {order_id} (IOC) was CANCELED by exchange with 0 filled. No execution occurred.{Style.RESET_ALL}")
                return None
            if status == "expired":  # Unusual for market orders but possible
                logger.error(
                    f"{Fore.RED}Market order {order_id} EXPIRED. This is unexpected for a market order.{Style.RESET_ALL}")
                return None

            # Brief delay for order propagation on the exchange side before further checks
            logger.debug(
                f"Delaying for {self.config.order_check_delay_seconds}s after market order {order_id} submission for exchange processing...")
            time.sleep(self.config.order_check_delay_seconds)
            return order_resp

        except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e:  # Fail-fast exceptions
            logger.error(
                f"{Fore.RED}Order placement failed due to exchange error ({type(e).__name__}): {e}{Style.RESET_ALL}", exc_info=False)
            termux_notify(f"{self.config.symbol} Order FAILED",
                          f"Market {side.upper()} failed: {str(e)[:50]}")
            return None
        except Exception as e:  # Other unexpected errors
            logger.error(
                f"{Fore.RED}Unexpected error during market order placement: {e}{Style.RESET_ALL}", exc_info=True)
            termux_notify(f"{self.config.symbol} Order ERROR",
                          f"Market {side.upper()} encountered an unexpected error.")
            return None

    def _set_position_protection(self, position_side: str, sl_price: Optional[Decimal] = None, tp_price: Optional[Decimal] = None, is_tsl: bool = False, tsl_distance: Optional[Decimal] = None, tsl_activation_price: Optional[Decimal] = None) -> bool:
        """
        Sets Stop Loss (SL), Take Profit (TP), or Trailing Stop Loss (TSL) for a position using Bybit V5 API.
        `position_side` is "long" or "short".
        To clear a specific protection, pass its price/distance as None or rely on "0" formatting.
        To clear all, ensure all price/distance params are effectively None or result in "0".
        """
        if not self.exchange or not self.market_info:
            logger.error(
                "Cannot set position protection: Exchange or Market info missing.")
            return False
        market_id = self.market_info.get("id")
        if not market_id:
            logger.error("Cannot set position protection: Market ID missing.")
            return False

        tracker_key = position_side.lower()  # "long" or "short"
        if tracker_key not in self.protection_tracker:
            logger.error(
                f"Invalid position_side '{position_side}' for protection tracker keying.")
            return False

        # Format parameters for Bybit V5 API. "0" means cancel that specific item.
        # `_format_v5_param` returns None if input is None or invalid (unless allow_zero is True for zero input).
        # If None is returned, we default to "0" to signify cancellation or no change to that parameter.
        sl_api_str = self.exchange_manager._format_v5_param(
            sl_price, "price", allow_zero=True) or "0"
        tp_api_str = self.exchange_manager._format_v5_param(
            tp_price, "price", allow_zero=True) or "0"
        # TSL distance and activation price: "0" also means cancel.
        tsl_dist_api_str = self.exchange_manager._format_v5_param(
            tsl_distance, "distance", allow_zero=True) or "0"
        tsl_act_px_api_str = self.exchange_manager._format_v5_param(
            tsl_activation_price, "price", allow_zero=True) or "0"

        api_params: Dict[str, Any] = {
            "category": self.config.bybit_v5_category,
            "symbol": market_id,
            "positionIdx": self.config.position_idx,
            # Ensures SL/TP apply to the whole position for the given positionIdx
            "tpslMode": V5_TPSL_MODE_FULL
        }
        action_desc = ""
        new_tracker_state: Optional[Literal[PROTECTION_STATE_SLTP,
                                            PROTECTION_STATE_TSL]] = None

        if is_tsl:
            # Activating or modifying TSL
            # Both distance and activation price needed for TSL
            if tsl_dist_api_str != "0" and tsl_act_px_api_str != "0":
                api_params.update({
                    "trailingStop": tsl_dist_api_str,
                    "activePrice": tsl_act_px_api_str,  # Price at which TSL becomes active
                    "triggerBy": self.config.tsl_trigger_by,  # Price type for TSL activation
                    "stopLoss": "0",  # Clear fixed SL when TSL is active
                    "takeProfit": "0"  # Clear fixed TP when TSL is active
                })
                action_desc = f"ACTIVATE/MODIFY TSL (Distance: {tsl_dist_api_str}, ActivationPx: {tsl_act_px_api_str})"
                new_tracker_state = PROTECTION_STATE_TSL
            else:
                logger.error(
                    f"Cannot activate TSL for {position_side.upper()}: TSL distance ('{tsl_dist_api_str}') or activation price ('{tsl_act_px_api_str}') is invalid or zero.")
                return False
        elif sl_api_str != "0" or tp_api_str != "0":  # Setting fixed SL and/or TP
            if sl_api_str != "0":
                api_params["stopLoss"] = sl_api_str
                api_params["slTriggerBy"] = self.config.sl_trigger_by
            if tp_api_str != "0":
                api_params["takeProfit"] = tp_api_str
                # Bybit uses tpTriggerBy, often same as SL or LastPrice
                api_params["tpTriggerBy"] = self.config.sl_trigger_by
            # Ensure TSL is cleared if setting fixed SL/TP
            api_params.update({"trailingStop": "0", "activePrice": "0"})
            action_desc = f"SET SL={api_params.get('stopLoss', 'NotSet')} TP={api_params.get('takeProfit', 'NotSet')}"
            new_tracker_state = PROTECTION_STATE_SLTP
        else:  # Clearing all protection (SL, TP, TSL)
            api_params.update(
                {"stopLoss": "0", "takeProfit": "0", "trailingStop": "0", "activePrice": "0"})
            action_desc = "CLEAR ALL SL/TP/TSL protection"
            new_tracker_state = None  # No protection active

        logger.trade(
            f"{Fore.CYAN}Attempting to {action_desc} for {position_side.upper()} position on {self.config.symbol}...{Style.RESET_ALL}")
        logger.debug(
            f"Calling Bybit V5 setTradingStop with parameters: {api_params}")


def set_trading_stop(self, symbol, position_side, stop_loss, take_profit, trailing_stop, order_id):
    """
    Set SL/TP/TSL for a position using Bybit V5 API with CCXT.
    """
    # Standardized and fallback method names for Bybit V5 trading stop
    private_method_name = "private_post_v5_position_trading_stop"
    private_method_name_fallback = "privatePostPositionTradingStop"

    # Check for method availability
    if not hasattr(self.ccxt_exchange, private_method_name):
        if hasattr(self.ccxt_exchange, private_method_name_fallback):
            logger.warning(
                f"CCXT method '{private_method_name}' not found, using fallback "
                f"'{private_method_name_fallback}'. Consider updating CCXT."
            )
            private_method_name = private_method_name_fallback
        else:
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Fatal Error: CCXT method for setting trading stops "
                f"('{private_method_name}' or '{private_method_name_fallback}') not found. "
                f"Cannot manage position protection. Update CCXT or check method name.{Style.RESET_ALL}"
            )
            # Cancel order to prevent unprotected position
            try:
                self.ccxt_exchange.cancel_order(order_id, symbol)
                logger.info(
                    f"Order {order_id} canceled due to missing CCXT method.")
            except Exception as cancel_error:
                logger.error(
                    f"Failed to cancel order {order_id}: {cancel_error}")
            return False

    # Prepare API parameters
    api_params = {
        'category': 'linear',
        'symbol': symbol,
        'stopLoss': str(stop_loss) if stop_loss else '',
        'takeProfit': str(take_profit) if take_profit else '',
        'trailingStop': str(trailing_stop) if trailing_stop else '',
        'slTriggerBy': 'MarkPrice',
        'tpTriggerBy': 'MarkPrice'
    }

    # Action description for logging
    action_desc = f"Set SL={stop_loss}, TP={take_profit}, TSL={trailing_stop}"
    tracker_key = f"{symbol}_{position_side.lower()}"
    new_tracker_state = {'sl': stop_loss,
                         'tp': take_profit, 'tsl': trailing_stop}

    # Execute API call with retries
    method_to_call = getattr(self.ccxt_exchange, private_method_name)
    try:
        response = fetch_with_retries(
            method_to_call,
            params=api_params,
            max_retries=self.config.max_fetch_retries,
            delay_seconds=self.config.retry_delay_seconds
        )
        if response and response.get("retCode") == V5_SUCCESS_RETCODE:
            logger.info(
                f"{Style.BRIGHT}{Fore.GREEN}Protection ({action_desc}) set successfully for "
                f"{position_side.upper()}.{Style.RESET_ALL}"
            )
            # Optional Termux notification (if enabled)
            try:
                termux_notify(
                    f"{self.config.symbol} Protection Update",
                    f"{action_desc[:30]}... for {position_side.upper()} successful."
                )
            except NameError:
                pass  # termux_notify not defined
            self.protection_tracker[tracker_key] = new_tracker_state
            return True
        else:
            ret_code = response.get(
                "retCode", "[No RetCode]") if response else "[No Response]"
            ret_msg = response.get(
                "retMsg", "[No RetMsg]") if response else "[No Response]"
            logger.error(
                f"{Fore.RED}Failed to {action_desc} for {position_side.upper()}. "
                f"API Response: Code={ret_code}, Msg='{ret_msg}'.{Style.RESET_ALL}"
            )
            logger.debug(
                f"Full API response from failed {private_method_name}: {str(response)[:500]}")
            # Optional Termux notification
            try:
                termux_notify(
                    f"{self.config.symbol} Protection FAILED",
                    f"{action_desc[:30]}... failed: {ret_msg[:50]}"
                )
            except NameError:
                pass
            # Cancel order to prevent unprotected position
            try:
                self.ccxt_exchange.cancel_order(order_id, symbol)
                logger.info(
                    f"Order {order_id} canceled due to failed protection setting.")
            except Exception as cancel_error:
                logger.error(
                    f"Failed to cancel order {order_id}: {cancel_error}")
            return False
    except Exception as e:
        logger.error(
            f"{Fore.RED}Unexpected error during '{action_desc}' for {position_side.upper()}: "
            f"{e}{Style.RESET_ALL}", exc_info=True
        )
        # Optional Termux notification
        try:
            termux_notify(
                f"{self.config.symbol} Protection ERROR",
                f"{action_desc[:30]}... encountered an error."
            )
        except NameError:
            pass
        # Cancel order to prevent unprotected position
        try:
            self.ccxt_exchange.cancel_order(order_id, symbol)
            logger.info(f"Order {order_id} canceled due to unexpected error.")
        except Exception as cancel_error:
            logger.error(f"Failed to cancel order {order_id}: {cancel_error}")
        return False

    def _verify_position_state(self, expected_side_logical: Optional[str], expected_qty_min_abs: Decimal = POSITION_QTY_EPSILON, max_attempts: int = 4, delay_seconds: float = 1.5, action_context: str = "Position State Verification") -> Tuple[bool, Optional[Dict[str, Dict[str, Any]]]]:
        """
        Verifies the current position state against an expected state (side and quantity).
        `expected_side_logical`: "long", "short", or None (for expecting flat).
        Returns (verification_success_bool, last_fetched_position_summary_dict).
        """
        logger.debug(f"{action_context}: Verifying position. Expected side: '{expected_side_logical if expected_side_logical else 'FLAT'}', MinQty (if side expected) ≈ {expected_qty_min_abs.normalize()}. Max attempts: {max_attempts}.")
        last_pos_summary: Optional[Dict[str, Dict[str, Any]]] = None

        for attempt in range(max_attempts):
            logger.debug(
                f"{action_context}: Fetch attempt {attempt + 1}/{max_attempts}...")
            current_pos_summary = self.exchange_manager.get_current_position()
            # Keep track of the last fetched summary
            last_pos_summary = current_pos_summary

            if current_pos_summary is None:  # Failed to fetch position data
                logger.warning(
                    f"{action_context} Warning: Failed to fetch position state on attempt {attempt + 1}.")
                if attempt < max_attempts - 1:
                    time.sleep(delay_seconds)
                    continue
                # Max attempts reached for fetching
                logger.error(
                    f"{Fore.RED}{action_context} FAILED: Could not fetch current position after {max_attempts} attempts.{Style.RESET_ALL}")
                return False, last_pos_summary

            # Determine actual current state from fetched summary
            actual_is_flat, actual_open_side, actual_qty_abs = True, None, Decimal(
                "0")
            long_pos_data = current_pos_summary.get("long", {})
            short_pos_data = current_pos_summary.get("short", {})

            if long_pos_data and safe_decimal(long_pos_data.get("qty", "0")).copy_abs() >= POSITION_QTY_EPSILON:
                actual_is_flat, actual_open_side, actual_qty_abs = False, "long", safe_decimal(
                    long_pos_data.get("qty", "0")).copy_abs()
            elif short_pos_data and safe_decimal(short_pos_data.get("qty", "0")).copy_abs() >= POSITION_QTY_EPSILON:
                actual_is_flat, actual_open_side, actual_qty_abs = False, "short", safe_decimal(
                    short_pos_data.get("qty", "0")).copy_abs()

            # Compare actual state with expected state
            verified_ok, log_message_suffix = False, ""
            if expected_side_logical is None:  # Expecting to be flat
                verified_ok = actual_is_flat
                log_message_suffix = f"Expected FLAT, Actual: {'FLAT' if actual_is_flat else f'{str(actual_open_side).upper()} Qty={actual_qty_abs.normalize()}'}"
            elif actual_open_side == expected_side_logical:  # Expecting a specific side, and it matches
                qty_matches_expectation = actual_qty_abs >= expected_qty_min_abs
                verified_ok = qty_matches_expectation
                log_message_suffix = (f"Expected {expected_side_logical.upper()} (MinQty≈{expected_qty_min_abs.normalize()}), "
                                      f"Actual: {actual_open_side.upper()} Qty={actual_qty_abs.normalize()} ({'QTY OK' if qty_matches_expectation else 'QTY MISMATCH/TOO LOW'})")
            else:  # Side mismatch (e.g., expected long, but actual is flat or short)
                verified_ok = False  # Verification failed due to side mismatch
                actual_state_str = 'FLAT' if actual_is_flat else (
                    f"{str(actual_open_side).upper()} Qty={actual_qty_abs.normalize()}" if actual_open_side else 'UNKNOWN_STATE')
                log_message_suffix = (f"Expected {str(expected_side_logical).upper()}, "
                                      f"Actual: {actual_state_str} (SIDE MISMATCH)")

            logger.debug(
                f"{action_context} Check (Attempt {attempt + 1}): {log_message_suffix}")
            if verified_ok:
                logger.info(
                    f"{Style.BRIGHT}{Fore.GREEN}{action_context} SUCCEEDED (Attempt {attempt + 1}). State verified: {log_message_suffix}{Style.RESET_ALL}")
                return True, current_pos_summary

            # If not verified and not last attempt, wait and retry
            if attempt < max_attempts - 1:
                logger.debug(
                    f"Position state not as expected. Retrying in {delay_seconds}s...")
                time.sleep(delay_seconds)
            else:  # Max attempts reached, verification failed
                logger.error(
                    f"{Fore.RED}{action_context} FAILED after {max_attempts} attempts. Final observed state: {log_message_suffix}{Style.RESET_ALL}")
                return False, current_pos_summary

        # Should be unreachable if loop logic is correct, but as a fallback
        return False, last_pos_summary

    def place_risked_market_order(self, side: str, atr: Decimal, total_equity: Decimal, current_price: Decimal) -> bool:
        """
        Places a market order with calculated risk, quantity, SL, and TP.
        `side` is "buy" or "sell".
        Returns True if order placed, position verified, and protection set successfully. False otherwise.
        """
        if not self.exchange or not self.market_info:  # Should be caught by TradingBot init
            logger.critical(
                "OrderManager cannot place order: Critical components (exchange/market_info) missing.")
            return False
        if side not in ("buy", "sell"):
            logger.error(
                f"Invalid side '{side}' provided for risked market order.")
            return False
        # ATR, equity, price validated by _calculate_trade_parameters, but quick check here too
        if atr.is_nan() or atr <= Decimal("0"):
            logger.error("Entry Aborted: Invalid ATR for risk calculation.")
            return False
        if total_equity is None or total_equity.is_nan() or total_equity <= Decimal("0"):
            logger.error("Entry Aborted: Invalid Equity for risk calculation.")
            return False
        if current_price.is_nan() or current_price <= Decimal("0"):
            logger.error("Entry Aborted: Invalid Current Price for entry.")
            return False

        logical_pos_side = "long" if side == "buy" else "short"
        logger.info(
            f"{Style.BRIGHT}{Fore.MAGENTA}--- Initiating Market Entry Sequence: {logical_pos_side.upper()} ---{Style.RESET_ALL}")

        # 1. Calculate Trade Parameters
        trade_params = self._calculate_trade_parameters(
            side, atr, total_equity, current_price)
        # type: ignore[union-attr]
        if not trade_params or not trade_params.get("qty") or trade_params["qty"] <= Decimal("0"):
            logger.error(
                "Entry Aborted: Failed to calculate valid trade parameters (e.g., quantity is zero or invalid).")
            return False

        qty_to_order, initial_sl_price, initial_tp_price = trade_params["qty"], trade_params.get(
            # type: ignore[assignment]
            "sl_price"), trade_params.get("tp_price")
        if initial_sl_price is None or initial_sl_price.is_nan() or initial_sl_price <= Decimal("0"):
            logger.error(
                f"Entry Aborted: Invalid Stop Loss price ({initial_sl_price}) derived from trade parameters.")
            return False

        # 2. Execute Market Order
        market_order_response = self._execute_market_order(
            side, qty_to_order)  # type: ignore[arg-type]
        if not market_order_response:
            # type: ignore[union-attr]
            logger.error(
                f"Entry Aborted: Market order execution failed for {side.upper()} {qty_to_order.normalize()}.")
            self._handle_entry_failure(side, qty_to_order)
            return False  # type: ignore[arg-type]

        order_id = market_order_response.get("id", "[N/A_ORDER_ID]")
        avg_entry_px_from_order_resp = safe_decimal(market_order_response.get(
            "average", "NaN"))  # Avg fill price from order creation

        # 3. Verify Position State
        # Expect at least e.g. 90% fill for verification to pass, to account for minor discrepancies
        min_expected_fill_qty = qty_to_order * \
            Decimal("0.90")  # type: ignore[operator]
        verification_successful, final_pos_state_summary = self._verify_position_state(
            expected_side_logical=logical_pos_side, expected_qty_min_abs=min_expected_fill_qty,
            # Use slightly longer delay for verification
            max_attempts=6, delay_seconds=max(self.config.order_check_delay_seconds, 1.0),
            action_context=f"Post-{logical_pos_side.upper()}-Entry Verification (Order {order_id})"
        )
        if not verification_successful:
            logger.error(f"{Fore.RED}Entry FAILED: Position verification failed after market order {order_id}. Manual check of position status required! Attempting cleanup of potential lingering position...{Style.RESET_ALL}")
            self._handle_entry_failure(side, qty_to_order)
            return False  # type: ignore[arg-type]

        # If verification successful, extract details from the verified position state
        active_position_details = final_pos_state_summary.get(
            # type: ignore[union-attr]
            logical_pos_side) if final_pos_state_summary else {}
        if not active_position_details:  # Should not happen if verification_successful is True
            logger.error(
                f"{Fore.RED}Internal Error: Position {logical_pos_side} verified OK, but its details are missing from summary. Aborting entry sequence.{Style.RESET_ALL}")
            self._handle_entry_failure(side, qty_to_order)
            return False  # type: ignore[arg-type]

        actual_filled_qty = safe_decimal(
            active_position_details.get("qty", "0")).copy_abs()
        actual_avg_entry_px = safe_decimal(
            active_position_details.get("entry_price", "NaN"))
        # If position entry price isn't immediately available, use from order response as fallback
        if actual_avg_entry_px.is_nan() and not avg_entry_px_from_order_resp.is_nan():
            actual_avg_entry_px = avg_entry_px_from_order_resp
            logger.debug(
                f"Used average entry price from order response ({avg_entry_px_from_order_resp.normalize()}) as position data was initially NaN for entry price.")

        avg_entry_px_display = actual_avg_entry_px.normalize(
        ) if not actual_avg_entry_px.is_nan() else '[EntryPx N/A]'
        logger.info(f"{Style.BRIGHT}{Fore.GREEN}Position {logical_pos_side.upper()} CONFIRMED: Qty={actual_filled_qty.normalize()}, AvgEntryPx={avg_entry_px_display}{Style.RESET_ALL}")

        # Warn if filled quantity is significantly less than ordered (e.g. >1% difference or a fixed small amount)
        # type: ignore[operator]
        if actual_filled_qty < qty_to_order * Decimal("0.99"):
            # type: ignore[union-attr]
            logger.warning(
                f"Filled quantity {actual_filled_qty.normalize()} is notably less than ordered quantity {qty_to_order.normalize()}. This may indicate slippage or partial fill.{Style.RESET_ALL}")

        # 4. Set Initial Stop Loss and Take Profit
        protection_set_ok = self._set_position_protection(
            logical_pos_side, sl_price=initial_sl_price, tp_price=initial_tp_price)
        if not protection_set_ok:
            logger.error(
                f"{Fore.RED}Entry Alert: FAILED to set initial SL/TP for {logical_pos_side.upper()} position. Attempting EMERGENCY CLOSE of the position!{Style.RESET_ALL}")
            self.close_position(logical_pos_side, actual_filled_qty,
                                reason="EmergencyClose:FailedInitialStopSet")
            return False

        # 5. Log trade entry to journal
        if self.config.enable_journaling:
            self.log_trade_entry_to_journal(
                side, actual_filled_qty, actual_avg_entry_px, order_id)

        logger.info(
            f"{Style.BRIGHT}{Fore.GREEN}--- Market Entry Sequence for {logical_pos_side.upper()} Completed Successfully ---{Style.RESET_ALL}")
        return True

    def manage_trailing_stop(self, position_side: str, entry_price: Decimal, current_market_price: Decimal, current_atr: Decimal) -> None:
        """
        Manages the Trailing Stop Loss (TSL). Activates TSL if conditions are met
        and the position is currently protected by a fixed SL/TP.
        """
        if not self.exchange or not self.market_info:  # Should be caught by init
            logger.error("TSL Management: Exchange or Market info missing.")
            return

        tracker_key = position_side.lower()
        # Only manage TSL if currently in fixed SL/TP mode (PROTECTION_STATE_SLTP)
        if self.protection_tracker.get(tracker_key) != PROTECTION_STATE_SLTP:
            logger.debug(
                f"TSL Check ({position_side.upper()}): Not in {PROTECTION_STATE_SLTP} state (Current state: {self.protection_tracker.get(tracker_key)}). Skipping TSL activation check.")
            return

        # Validate inputs for TSL calculation
        # type: ignore[attr-defined]
        if any(val.is_nan() or val <= Decimal("0") for val in [entry_price, current_market_price, current_atr]):
            logger.debug(
                f"TSL Check ({position_side.upper()}): Invalid ATR, entry_price, or current_market_price. Skipping TSL management. ATR:{current_atr}, EntryPx:{entry_price}, CurrPx:{current_market_price}")
            return

        try:
            # Calculate TSL activation target price (price moves by X ATRs from entry)
            activation_distance_atr_pts = current_atr * \
                self.config.tsl_activation_atr_multiplier  # type: ignore[operator]
            tsl_activation_target_price = entry_price + activation_distance_atr_pts if position_side == "long" else entry_price - \
                activation_distance_atr_pts  # type: ignore[operator]

            # type: ignore[union-attr]
            if tsl_activation_target_price.is_nan() or tsl_activation_target_price <= Decimal("0"):
                logger.warning(
                    f"Invalid TSL activation target price calculated ({tsl_activation_target_price.normalize() if not tsl_activation_target_price.is_nan() else 'NaN'}). Skipping TSL check for {position_side.upper()}.")
                return  # type: ignore[union-attr]

            # Calculate actual TSL distance to set (percentage of current price)
            tsl_distance_to_set_pts = current_market_price * \
                (self.config.trailing_stop_percent /
                 Decimal(100))  # type: ignore[operator]
            min_tick_size = self.market_info.get('tick_size', Decimal(
                '1e-8'))  # Default to small value if not found
            # type: ignore[union-attr]
            if not min_tick_size.is_nan() and min_tick_size > Decimal("0") and tsl_distance_to_set_pts < min_tick_size:
                # type: ignore[union-attr]
                logger.debug(
                    f"Calculated TSL distance ({tsl_distance_to_set_pts.normalize()}) is less than min tick size ({min_tick_size.normalize()}). Adjusting TSL distance to one tick size.")
                # type: ignore[union-attr]
                tsl_distance_to_set_pts = min_tick_size

            # type: ignore[union-attr]
            if tsl_distance_to_set_pts <= Decimal("0"):
                logger.warning(
                    f"Invalid TSL distance calculated ({tsl_distance_to_set_pts.normalize()}). Skipping TSL activation for {position_side.upper()}.")
                return  # type: ignore[union-attr]

            # Check if TSL activation condition is met
            should_activate_tsl = \
                (position_side == "long" and current_market_price >= tsl_activation_target_price) or \
                (position_side == "short" and current_market_price <=
                 tsl_activation_target_price)

            if should_activate_tsl:
                logger.trade(
                    f"{Fore.MAGENTA}Trailing Stop Loss (TSL) activation condition MET for {position_side.upper()} position!{Style.RESET_ALL}")
                logger.trade(
                    # type: ignore[union-attr]
                    f"  Details: EntryPx={entry_price.normalize():.{DEFAULT_PRICE_DP}f}, CurrPx={current_market_price.normalize():.{DEFAULT_PRICE_DP}f}, "
                    # type: ignore[union-attr]
                    f"TSL_ActivationTargetPx≈{tsl_activation_target_price.normalize():.{DEFAULT_PRICE_DP}f}, TSL_DistanceToSet≈{tsl_distance_to_set_pts.normalize():.{DEFAULT_PRICE_DP}f}"
                )

                # Activate TSL. For Bybit V5, `activePrice` is the price at which TSL becomes armed.
                # We use current_market_price as the activation price when the condition is met.
                # The TSL will then trail from this `activePrice` by `tsl_distance_to_set_pts`.
                activation_successful = self._set_position_protection(
                    position_side, is_tsl=True,
                    tsl_distance=tsl_distance_to_set_pts,  # The trailing distance
                    tsl_activation_price=current_market_price  # The price that arms the TSL
                )
                if activation_successful:
                    logger.trade(
                        f"{Style.BRIGHT}{Fore.GREEN}TSL activated successfully for {position_side.upper()}. Position now managed by exchange TSL.{Style.RESET_ALL}")
                else:
                    logger.error(
                        f"{Fore.RED}Failed to activate TSL for {position_side.upper()} via API call. Position remains on fixed SL/TP.{Style.RESET_ALL}")
            else:
                # type: ignore[union-attr]
                logger.debug(
                    f"TSL Check ({position_side.upper()}): Activation condition NOT MET. (CurrentPrice: {current_market_price.normalize():.{DEFAULT_PRICE_DP}f}, TargetActivationPrice: ~{tsl_activation_target_price.normalize():.{DEFAULT_PRICE_DP}f})")
        except Exception as e:
            logger.error(
                f"Error during TSL management for {position_side.upper()}: {e}", exc_info=True)

    def close_position(self, position_side_to_close: str, qty_abs_to_close: Decimal, reason: str = "Strategy Exit Signal") -> bool:
        """
        Closes an existing position by placing a market order for the opposite side.
        `position_side_to_close` is "long" or "short".
        Returns True if closure initiated and verified, False otherwise.
        """
        if not self.exchange or not self.market_info:
            logger.critical(
                "OrderManager cannot close position: Critical components missing.")
            return False
        if position_side_to_close not in ("long", "short"):
            logger.error(
                f"Invalid side '{position_side_to_close}' specified for closing position.")
            return False

        if qty_abs_to_close.is_nan() or qty_abs_to_close.copy_abs() < POSITION_QTY_EPSILON:
            logger.warning(
                f"Close position requested for zero or negligible quantity ({qty_abs_to_close.normalize()}). Skipping close action for {position_side_to_close.upper()}.")
            # If position was considered open but qty is negligible, reset tracker
            self.protection_tracker[position_side_to_close.lower()] = None
            return True

        closing_order_side = "sell" if position_side_to_close == "long" else "buy"
        tracker_key = position_side_to_close.lower()  # "long" or "short"

        logger.trade(f"{Fore.YELLOW}Attempting to CLOSE {position_side_to_close.upper()} position (Qty: {qty_abs_to_close.normalize()} {self.market_info.get('base', 'BASE_CCY')}) | Reason: {reason}...{Style.RESET_ALL}")

        # 1. Clear any existing SL/TP/TSL protection before sending close order
        logger.debug(
            f"Clearing any existing SL/TP/TSL protection for {position_side_to_close.upper()} before attempting to close...")
        # Clears all
        if self._set_position_protection(position_side_to_close, sl_price=None, tp_price=None, is_tsl=False):
            logger.info(
                f"Protection (SL/TP/TSL) cleared (or was already clear) for {position_side_to_close.upper()}.")
            # Update tracker to reflect no protection
            self.protection_tracker[tracker_key] = None
        else:
            # If clearing protection fails, it's a warning, but proceed with close attempt as it's critical.
            logger.warning(
                f"{Fore.YELLOW}Failed to confirm protection clear for {position_side_to_close.upper()}. Proceeding with close attempt regardless...{Style.RESET_ALL}")

        # 2. Execute Market Order to Close Position
        close_order_response = self._execute_market_order(
            closing_order_side, qty_abs_to_close)
        if not close_order_response:
            logger.error(
                f"{Fore.RED}Failed to submit closing market order for {position_side_to_close.upper()}. MANUAL INTERVENTION REQUIRED! Position may still be open.{Style.RESET_ALL}")
            termux_notify(f"{self.config.symbol} CLOSE ORDER FAILED",
                          f"Market {closing_order_side.upper()} order to close {position_side_to_close.upper()} failed!")
            return False

        close_order_id = close_order_response.get("id", "[N/A_CLOSE_ORDER_ID]")
        avg_close_px_from_order = safe_decimal(
            close_order_response.get("average"), default=Decimal("NaN"))
        avg_close_px_display = avg_close_px_from_order.normalize(
        ) if not avg_close_px_from_order.is_nan() else '[Pending/N/A]'
        logger.trade(f"{Fore.YELLOW}Closing market order ({close_order_id}) submitted for {position_side_to_close.upper()}. Reported AvgClosePx: {avg_close_px_display}{Style.RESET_ALL}")
        termux_notify(f"{self.config.symbol} Position Closing",
                      f"{position_side_to_close.upper()} position close order {close_order_id} submitted.")

        # 3. Verify Position is Closed (Flat)
        # Add a bit more delay here to ensure exchange processes the closure
        verification_delay = max(
            self.config.order_check_delay_seconds + 0.5, 1.5)
        is_verified_closed, _ = self._verify_position_state(
            expected_side_logical=None,  # Expecting flat
            max_attempts=6, delay_seconds=verification_delay,
            action_context=f"Post-{position_side_to_close.upper()}-Close Verification (Order {close_order_id})"
        )

        # 4. Log trade exit to journal (regardless of verification, log the attempt)
        if self.config.enable_journaling:
            self.log_trade_exit_to_journal(
                position_side_to_close, qty_abs_to_close, avg_close_px_from_order, close_order_id, reason)

        if not is_verified_closed:
            logger.error(f"{Fore.RED}Position {position_side_to_close.upper()} closure verification FAILED. Position may still be open or partially closed. MANUAL INTERVENTION REQUIRED!{Style.RESET_ALL}")
            termux_notify(f"{self.config.symbol} CLOSE VERIFY FAILED",
                          f"{position_side_to_close.upper()} position may still be open after close attempt!")
            return False

        logger.trade(
            f"{Style.BRIGHT}{Fore.GREEN}Position {position_side_to_close.upper()} confirmed closed (flat).{Style.RESET_ALL}")
        # Ensure tracker is cleared
        self.protection_tracker[tracker_key] = None
        return True

    def _handle_entry_failure(self, failed_entry_order_side: str, attempted_qty_abs: Decimal) -> None:
        """
        Handles the situation after a market entry order fails or verification fails.
        Checks for any lingering position that might have been unintentionally opened and attempts to close it.
        `failed_entry_order_side` is "buy" or "sell".
        """
        logger.warning(f"{Fore.YELLOW}Handling entry failure for {failed_entry_order_side.upper()} order (intended qty: {attempted_qty_abs.normalize()}). Checking for any lingering unintended position...{Style.RESET_ALL}")

        logical_pos_side_to_check = "long" if failed_entry_order_side == "buy" else "short"
        # Allow extra time for exchange state to settle before checking for lingering position
        time.sleep(max(self.config.order_check_delay_seconds, 1.0) + 1.0)

        # Check current position state. We don't have a strong expectation, just observing.
        # If a position matching the failed entry exists, it's a problem.
        _, current_pos_summary = self._verify_position_state(
            # Primarily checking if flat, or if the unintended position exists
            expected_side_logical=None,
            max_attempts=2, delay_seconds=1.0,  # Quick check
            action_context=f"Entry-Failure-Cleanup-Check for {logical_pos_side_to_check.upper()}"
        )

        if current_pos_summary is None:  # If fetching position fails, manual check is paramount
            logger.error(f"{Fore.RED}Could not fetch current positions during entry failure handling for {logical_pos_side_to_check.upper()}. MANUAL CHECK OF POSITION STATUS URGENTLY REQUIRED!{Style.RESET_ALL}")
            termux_notify(f"{self.config.symbol} URGENT CHECK",
                          "Position fetch failed during entry failure cleanup!")
            return

        lingering_position_data = current_pos_summary.get(
            logical_pos_side_to_check, {})
        current_lingering_qty = safe_decimal(
            lingering_position_data.get("qty", "0")).copy_abs()

        if current_lingering_qty >= POSITION_QTY_EPSILON:  # A significant lingering position exists
            logger.error(f"{Fore.RED}Lingering {logical_pos_side_to_check.upper()} position (Qty: {current_lingering_qty.normalize()}) found after failed entry attempt. Attempting emergency close...{Style.RESET_ALL}")
            termux_notify(f"{self.config.symbol} Emergency Close",
                          f"Lingering {logical_pos_side_to_check.upper()} position found after failed entry.")

            if self.close_position(logical_pos_side_to_check, current_lingering_qty, reason="EmergencyClose:LingeringAfterEntryFail"):
                logger.info(
                    f"Emergency close for lingering {logical_pos_side_to_check.upper()} position submitted and confirmed closed.")
            else:
                logger.critical(
                    f"{Style.BRIGHT}{Fore.RED}EMERGENCY CLOSE FAILED for lingering {logical_pos_side_to_check.upper()} position. MANUAL INTERVENTION URGENTLY REQUIRED!{Style.RESET_ALL}")
                termux_notify(f"{self.config.symbol} URGENT CHECK",
                              f"Emergency close of lingering {logical_pos_side_to_check.upper()} FAILED!")
        else:  # No significant lingering position found
            logger.info(
                f"No significant lingering {logical_pos_side_to_check.upper()} position detected (Current qty: {current_lingering_qty.normalize()}). Cleanup check complete.")
            # Ensure tracker is clear
            self.protection_tracker[logical_pos_side_to_check] = None

    def _write_journal_row(self, trade_data: Dict[str, Any]) -> None:
        """Writes a single row of trade data to the CSV journal file."""
        if not self.config.enable_journaling:
            return  # Skip if journaling is disabled

        journal_file = Path(self.config.journal_file_path)
        # Check if file exists and has content (to determine if header is needed)
        file_exists_and_has_content = journal_file.is_file(
        ) and journal_file.stat().st_size > 0

        try:
            # Ensure directory exists
            journal_file.parent.mkdir(parents=True, exist_ok=True)
            with journal_file.open("a", newline="", encoding="utf-8") as csvfile:  # Append mode
                # Define fieldnames for CSV header and row writing
                fieldnames = ["TimestampUTC", "Symbol", "Action", "Side",
                              "Quantity", "AvgPrice", "OrderID", "Reason", "Notes"]
                writer = csv.DictWriter(
                    csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)

                if not file_exists_and_has_content:  # Write header only if file is new or empty
                    writer.writeheader()

                # Prepare row data, formatting Decimals and handling None
                row_to_write = {}
                for field in fieldnames:
                    val = trade_data.get(field)
                    if isinstance(val, Decimal):
                        row_to_write[field] = 'NaN' if val.is_nan() else str(
                            val.normalize())  # Store NaN as string
                    else:
                        # Default to 'N/A' string for None
                        row_to_write[field] = str(
                            val if val is not None else 'N/A')

                # Ensure 'Notes' field is always a string, even if empty or not provided
                row_to_write['Notes'] = str(trade_data.get('Notes', ''))

                writer.writerow(row_to_write)
            logger.debug(
                f"Trade action '{trade_data.get('Action', 'Unknown')}' logged to journal: {journal_file.name}")
        except IOError as e:  # Catch file I/O errors specifically
            logger.error(
                f"I/O error writing to journal file '{journal_file.name}': {e}")
        except Exception as e:  # Catch other unexpected errors during journaling
            logger.error(
                f"Unexpected error writing to trading journal: {e}", exc_info=True)

    def log_trade_entry_to_journal(self, order_side: str, filled_qty_abs: Decimal, avg_fill_price: Decimal, order_id: Optional[str]) -> None:
        """Logs a trade entry event to the journal."""
        logical_side = ("long" if order_side == "buy" else "short").upper()
        self._write_journal_row({
            "TimestampUTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "Symbol": self.config.symbol,
            "Action": "ENTRY",
            "Side": logical_side,
            "Quantity": filled_qty_abs,
            "AvgPrice": avg_fill_price,
            "OrderID": order_id,
            "Reason": "Strategy Entry Signal"  # Default reason for entries
        })

    def log_trade_exit_to_journal(self, position_side_closed: str, closed_qty_abs: Decimal, avg_close_price: Decimal, order_id: Optional[str], exit_reason: str) -> None:
        """Logs a trade exit event to the journal."""
        self._write_journal_row({
            "TimestampUTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "Symbol": self.config.symbol,
            "Action": "EXIT",
            "Side": position_side_closed.upper(),  # "LONG" or "SHORT"
            "Quantity": closed_qty_abs,
            "AvgPrice": avg_close_price,
            "OrderID": order_id,
            "Reason": exit_reason
        })

# --- Status Display Class ---


class StatusDisplay:
    """Handles the formatting and printing of the bot's status to the console using Rich."""

    def __init__(self, config: TradingConfig):
        self.config = config
        # Default display precisions, can be overridden by market info
        self._default_price_dp_display = DEFAULT_PRICE_DP
        self._default_amount_dp_display = DEFAULT_AMOUNT_DP

    def _format_decimal_for_rich(self, value: Optional[Decimal], precision: Optional[int] = None, default_precision_fallback: int = 2, add_commas: bool = False, highlight_positive_negative: bool = False, default_style: str = "white", style_override: Optional[str] = None) -> Text:
        """Formats a Decimal value for Rich Text display, handling None, NaN, precision, and styling."""
        if value is None or (isinstance(value, Decimal) and value.is_nan()):
            # Display "N/A" for None or NaN values
            return Text("N/A", style="dim")

        dp_to_use = precision if precision is not None else default_precision_fallback
        try:
            # Quantize to ensure correct number of decimal places for display
            quantizer = Decimal('1').scaleb(-dp_to_use)
            # ROUND_HALF_EVEN is common for display
            fmt_dec_val = value.quantize(quantizer, rounding=ROUND_HALF_EVEN)

            # Format with commas if requested
            format_spec = f"{{:{',' if add_commas else ''}.{dp_to_use}f}}"
            display_str = format_spec.format(fmt_dec_val)

            # Determine style
            current_style = style_override if style_override else default_style
            if highlight_positive_negative and not style_override:  # Only apply if no specific override
                if fmt_dec_val < Decimal("0"):
                    current_style = "bright_red"
                elif fmt_dec_val > Decimal("0"):
                    current_style = "bright_green"  # Positive PnL, etc.

            return Text(display_str, style=current_style)
        except (ValueError, TypeError, InvalidOperation) as e:  # Catch formatting errors
            logger.error(
                f"Error formatting decimal '{value}' for Rich display: {e}", exc_info=False)
            return Text("ERR", style="bold bright_red")  # Error display

    def print_status_panel(self, cycle_num: int, current_timestamp: Optional[datetime], current_market_price: Optional[Decimal], indicators_data: Optional[Dict[str, Any]], current_positions_summary: Optional[Dict[str, Dict[str, Any]]], account_equity: Optional[Decimal], signal_check_result: Dict[str, Any], protection_status_tracker: Dict[str, Optional[str]], market_specific_info: Optional[Dict[str, Any]]) -> None:
        """Prints a comprehensive status panel to the console."""
        # Determine price and amount decimal places for display from market_info or defaults
        price_dp = self._default_price_dp_display
        amount_dp = self._default_amount_dp_display
        if market_specific_info and "precision_dp" in market_specific_info:
            price_dp = market_specific_info["precision_dp"].get(
                "price", self._default_price_dp_display)
            amount_dp = market_specific_info["precision_dp"].get(
                "amount", self._default_amount_dp_display)

        panel_content = Text()  # Main container for panel text
        ts_str = current_timestamp.strftime(
            "%Y-%m-%d %H:%M:%S %Z") if current_timestamp else Text("Timestamp N/A", style="dim").plain
        panel_title = f" Cycle {cycle_num} | {self.config.symbol} ({self.config.interval}) | {ts_str} "
        settle_ccy = market_specific_info.get(
            "settle", "SETTLE_CCY") if market_specific_info else "SETTLE_CCY"

        # Header: Price and Equity
        panel_content.append("Price: ", style="bold bright_cyan")
        panel_content.append(self._format_decimal_for_rich(
            current_market_price, price_dp, style_override="bright_white"))
        panel_content.append(" | Equity: ", style="bold bright_yellow")
        panel_content.append(self._format_decimal_for_rich(
            account_equity, 2, add_commas=True, style_override="bright_yellow"))
        panel_content.append(f" {settle_ccy}\n", style="bright_yellow")

        # Indicators Section
        panel_content.append("Indicators:\n", style="bold bright_cyan")
        if indicators_data:
            # Helper to format individual indicator values for display
            def fmt_ind(key: str, prec: int = 1, style: str = "white", is_bool: bool = False, true_style: Optional[str] = None, false_style: Optional[str] = None) -> Text:
                val = indicators_data.get(key)
                if is_bool:
                    # Ensure correct styling for boolean true/false
                    true_s = true_style or style  # Default true_style to base style
                    # Default false_style to dimmed base style
                    false_s = false_style or f"dim {style}"
                    # Ensure Python bool for consistent styling logic, handles np.bool_
                    bool_val = bool(val) if isinstance(
                        val, (bool, np.bool_)) else False
                    return Text(str(bool_val), style=(true_s if bool_val else false_s))

                # For numeric indicators, convert to Decimal if not already
                dec_val = val if isinstance(val, Decimal) else safe_decimal(
                    str(val) if val is not None else "NaN")
                return self._format_decimal_for_rich(dec_val, precision=prec, default_style=style)

            # Original Strategy Indicators Display
            orig_parts = []
            orig_parts.append(Text("EMA(F/S/T):").append(fmt_ind('fast_ema', price_dp, "cyan")).append("/").append(
                fmt_ind('slow_ema', price_dp, "magenta")).append("/").append(fmt_ind('trend_ema', price_dp, "yellow")))

            stoch_text_display = Text("Stoch(K/D/PrevK):").append(fmt_ind('stoch_k', 1, "bright_blue")).append(
                "/").append(fmt_ind('stoch_d', 1, "blue")).append("/").append(fmt_ind('stoch_k_prev', 1, "dim blue"))
            if indicators_data.get('stoch_kd_bullish'):
                # Rich bbcode for style
                stoch_text_display.append(" [b green]▲BullishCross[/b green]")
            elif indicators_data.get('stoch_kd_bearish'):
                stoch_text_display.append(" [b red]▼BearishCross[/b red]")
            orig_parts.append(stoch_text_display)

            orig_parts.append(Text(f"ATR({indicators_data.get('atr_period', self.config.atr_period)}):").append(
                # ATR often benefits from more precision
                fmt_ind('atr', price_dp+1, "bright_magenta")))

            adx_val = indicators_data.get('adx', Decimal("NaN"))
            adx_val = adx_val if isinstance(adx_val, Decimal) else safe_decimal(
                str(adx_val))  # type: ignore[assignment]
            adx_strength_style = "yellow" if not adx_val.is_nan(
            ) and adx_val > self.config.min_adx_level else "dim yellow"
            orig_parts.append(Text(f"ADX({self.config.adx_period}):").append(self._format_decimal_for_rich(adx_val, 1, default_style=adx_strength_style)).append(
                " [+DI:", style="dim").append(fmt_ind('pdi', 1, "bright_green")).append(" -DI:", style="dim").append(fmt_ind('mdi', 1, "bright_red")).append("]", style="dim"))
            panel_content.append("  Orig: ", style="yellow")
            panel_content.append(Text(" | ", style="dim").join(orig_parts))
            panel_content.append("\n")

            # VolumaticTrend (VT) Strategy Indicators Display (if enabled and data exists)
            # Check a key VT indicator
            if self.config.vt_enable and indicators_data.get('vt_trend_ema') is not None:
                vt_parts = []
                vt_parts.append(Text("TrendEMA:").append(
                    fmt_ind('vt_trend_ema', price_dp, "green")))
                vt_parts.append(Text("VWMA:").append(
                    fmt_ind('vt_vwma', price_dp, "green")))
                vt_parts.append(Text("VolSpike:").append(fmt_ind('vt_is_volume_spike', style="green",
                                # Highlight true spike
                                                                 is_bool=True, true_style="bright_green", false_style="dim green")))

                candle_color_str = "Green" if indicators_data.get(
                    'vt_candle_is_green') else "Red" if indicators_data.get('vt_candle_is_red') else "Neutral"
                candle_color_style = "green" if candle_color_str == "Green" else "red" if candle_color_str == "Red" else "dim"
                vt_parts.append(
                    Text(f"Candle:{candle_color_str}", style=candle_color_style))
                panel_content.append("  VT:   ", style="green")
                panel_content.append(Text(" | ", style="dim").join(vt_parts))
                panel_content.append("\n")
        else:  # Indicators not available
            panel_content.append(Text(
                "  Indicator data is currently being calculated or is unavailable...\n", style="dim"))

        # Position Status Section
        panel_content.append("Position: ", style="bold bright_cyan")
        pos_display_text = Text(
            "FLAT", style="bold bright_green")  # Default to FLAT
        active_pos_side: Optional[str] = None
        active_pos_data_dict: Optional[Dict[str, Any]] = None

        if current_positions_summary:  # Check if summary is available
            long_data, short_data = current_positions_summary.get(
                "long", {}), current_positions_summary.get("short", {})
            # Determine if a long or short position is active based on quantity
            if long_data and safe_decimal(long_data.get("qty")).copy_abs() >= POSITION_QTY_EPSILON:
                active_pos_side, active_pos_data_dict = "long", long_data
            elif short_data and safe_decimal(short_data.get("qty")).copy_abs() >= POSITION_QTY_EPSILON:
                active_pos_side, active_pos_data_dict = "short", short_data

        if active_pos_side and active_pos_data_dict:
            style_for_pos = "bold bright_green" if active_pos_side == "long" else "bold bright_red"
            pos_display_text = Text(
                f"{active_pos_side.upper()}: ", style=style_for_pos)
            pos_display_text.append("Qty=", style=style_for_pos).append(self._format_decimal_for_rich(
                active_pos_data_dict.get("qty"), amount_dp, style_override=style_for_pos))
            pos_display_text.append(" | EntryPx=", style="dim").append(self._format_decimal_for_rich(
                active_pos_data_dict.get("entry_price"), price_dp, style_override=style_for_pos))
            pos_display_text.append(" | PnL=", style="dim").append(self._format_decimal_for_rich(active_pos_data_dict.get(
                # PnL style handled by formatter
                "unrealized_pnl"), 4, add_commas=True, highlight_positive_negative=True))

            # Display Protection Status (Exchange vs Local Tracker)
            protection_text_display = Text(" | Protection: ", style="dim")
            exchange_protection_parts = []
            sl_val, tp_val = active_pos_data_dict.get(
                "stop_loss_price"), active_pos_data_dict.get("take_profit_price")
            is_tsl_active_on_exchange = active_pos_data_dict.get(
                "is_tsl_active", False)
            tsl_dist_val_exch = active_pos_data_dict.get(
                "tsl_distance_val")  # Bybit's 'trailingStop' is distance
            tsl_trigger_px_exch = active_pos_data_dict.get(
                "tsl_trigger_price")  # Bybit's 'activePrice' for TSL

            if is_tsl_active_on_exchange:
                exchange_protection_parts.append(
                    Text("TSL", style="bright_magenta"))
                if tsl_dist_val_exch:
                    exchange_protection_parts.append(Text(
                        f"(Dist:{self._format_decimal_for_rich(tsl_dist_val_exch, price_dp).plain})", style="magenta"))
                if tsl_trigger_px_exch:
                    exchange_protection_parts.append(Text(
                        f"(ActPx:{self._format_decimal_for_rich(tsl_trigger_px_exch, price_dp).plain})", style="magenta"))
            elif sl_val or tp_val:  # Fixed SL/TP
                if sl_val:
                    exchange_protection_parts.append(Text(
                        f"SL:{self._format_decimal_for_rich(sl_val, price_dp).plain}", style="bright_yellow"))
                if tp_val:
                    exchange_protection_parts.append(Text(
                        f"TP:{self._format_decimal_for_rich(tp_val, price_dp).plain}", style="bright_yellow"))

            if not exchange_protection_parts:
                # No protection set on exchange
                exchange_protection_parts.append(Text("None", style="dim"))

            protection_text_display.append("Exch:").append(
                # Join parts with space
                Text(" ").join(exchange_protection_parts))

            # Local tracker status
            local_tracker_status_str = protection_status_tracker.get(
                active_pos_side)
            protection_text_display.append(" LocalTrk:").append(Text(str(
                local_tracker_status_str) if local_tracker_status_str else "None", style="blue" if local_tracker_status_str else "dim"))

            # Check for mismatch between exchange and local tracker
            is_protection_mismatch = \
                (is_tsl_active_on_exchange and local_tracker_status_str != PROTECTION_STATE_TSL) or \
                ((sl_val or tp_val) and not is_tsl_active_on_exchange and local_tracker_status_str != PROTECTION_STATE_SLTP) or \
                (not is_tsl_active_on_exchange and not sl_val and not tp_val and local_tracker_status_str is not None)  # Exchange has no protection, but tracker thinks it does
            if is_protection_mismatch:
                protection_text_display.append(
                    Text(" [TrackerMismatch?]", style="bold bright_yellow"))

            pos_display_text.append(protection_text_display)
        panel_content.append(pos_display_text)
        panel_content.append("\n")

        # Signal/Status Section
        panel_content.append("Signal/Status:\n", style="bold bright_cyan")
        summary_reason_str = str(signal_check_result.get(
            "summary", "No status information available"))
        # Determine style for summary message based on keywords
        status_style_str = "dim"  # Default style
        if signal_check_result.get("long", False) or "Long Signal" in summary_reason_str or "ENTERED_long" in summary_reason_str:
            status_style_str = "bold bright_green"
        elif signal_check_result.get("short", False) or "Short Signal" in summary_reason_str or "ENTERED_short" in summary_reason_str:
            status_style_str = "bold bright_red"
        elif "Block" in summary_reason_str or "FAIL:" in summary_reason_str.upper() or "EmergencyClose" in summary_reason_str or "Conflict" in summary_reason_str:
            status_style_str = "yellow"
        elif "CLOSED_" in summary_reason_str or "HOLDING_" in summary_reason_str or "INFO:" in summary_reason_str:
            status_style_str = "bright_blue"
        elif not any(s in summary_reason_str for s in ["No Signal", "Initializing", "Processing..."]):
            status_style_str = "white"  # Default for other messages

        panel_content.append(
            Text(f"  Overall: {summary_reason_str}", style=status_style_str))

        # Display detailed reasons for original and VT signals, wrapped for readability
        orig_detail_str = signal_check_result.get("orig_detail")
        if orig_detail_str and orig_detail_str != "N/A":
            orig_wrapped_str = "\n".join([f"    {line}" for line in textwrap.wrap(f"Orig: {orig_detail_str}", width=max(
                # Adjust width based on console
                20, console.width - 25), subsequent_indent="      ")])
            # lstrip to remove leading spaces from first line of wrapped
            panel_content.append(
                Text(f"\n  └─ {orig_wrapped_str.lstrip()}", style="dim"))

        vt_detail_str = signal_check_result.get("vt_detail")
        if vt_detail_str and vt_detail_str != "N/A" and self.config.vt_enable:  # Only show if VT enabled
            vt_wrapped_str = "\n".join([f"    {line}" for line in textwrap.wrap(
                f"VT: {vt_detail_str}", width=max(20, console.width - 25), subsequent_indent="      ")])
            panel_content.append(
                Text(f"\n  └─ {vt_wrapped_str.lstrip()}", style="dim"))

        # Print the assembled panel
        console.print(Panel(
            panel_content, title=f"[bold bright_magenta]{panel_title}[/]", border_style="bright_blue", expand=False, padding=(1, 2)))

# --- Trading Bot Class ---


class TradingBot:
    """Main class for the Pyrmethus trading bot."""

    def __init__(self):
        logger.info(
            f"{Style.BRIGHT}{Fore.MAGENTA}--- Initializing Pyrmethus v4.5.8 (Neon Nexus - VolumaticTrend Edition) ---{Style.RESET_ALL}")
        self.config = TradingConfig()  # Load configuration first
        try:
            # Initialize core components, passing config and other dependencies
            self.exchange_manager = ExchangeManager(self.config)
            self.indicator_calculator = IndicatorCalculator(self.config)
            self.signal_generator = SignalGenerator(self.config)
            self.order_manager = OrderManager(
                self.config, self.exchange_manager)
        # Catch init errors from components (e.g., OrderManager needing valid ExchangeManager)
        except ValueError as ve:
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}TradingBot component initialization failed: {ve}. Halting.{Style.RESET_ALL}")
            sys.exit(1)
        except Exception as e:  # Catch any other unexpected critical errors during init
            logger.critical(
                f"{Style.BRIGHT}{Fore.RED}Unexpected critical error during TradingBot component initialization: {e}. Halting.{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)

        self.status_display = StatusDisplay(self.config)  # For console output
        self.shutdown_requested = False  # Flag for graceful shutdown
        self._setup_signal_handlers()  # Setup SIGINT/SIGTERM handling
        logger.info(
            f"{Style.BRIGHT}{Fore.GREEN}Pyrmethus components initialized. Ready to conjure trades.{Style.RESET_ALL}")

    def _setup_signal_handlers(self) -> None:
        """Sets up OS signal handlers for graceful shutdown (Ctrl+C, kill)."""
        signals_to_handle = [signal.SIGINT, signal.SIGTERM]
        for sig in signals_to_handle:
            try:
                signal.signal(sig, self._signal_handler_callback)
                # Use signal.Signals(sig).name for Python 3.5+ for readable name
                sig_name_str = signal.Signals(sig).name if hasattr(
                    signal, 'Signals') and sig in signal.Signals else str(sig)
                logger.debug(
                    f"Signal handler for {sig_name_str} set up successfully.")
            # AttributeError for Windows if signal not available
            except (ValueError, OSError, AttributeError, Exception) as e:
                sig_name_str = signal.Signals(sig).name if hasattr(
                    signal, 'Signals') and sig in signal.Signals else str(sig)
                logger.warning(
                    f"{Fore.YELLOW}Could not set OS signal handler for {sig_name_str} (OS/environment limitation?): {e}{Style.RESET_ALL}")

    def _signal_handler_callback(self, sig_num: int, frame: Optional[Any]) -> None:
        """Callback function for OS signals to initiate graceful shutdown."""
        # `frame` is unused but required by the signal.signal signature
        _ = frame
        # Get signal name for logging
        sig_name = signal.Signals(sig_num).name if hasattr(signal, 'Signals') and isinstance(
            sig_num, int) and sig_num in signal.Signals else f"Signal {sig_num}"

        if not self.shutdown_requested:  # Process shutdown only once
            # Use Rich console for immediate user feedback if possible
            console.print(
                f"\n[bold yellow]Signal {sig_name} received. Initiating graceful shutdown... Please wait.[/bold yellow]")
            logger.warning(
                f"Signal {sig_name} received. Initiating graceful shutdown sequence...")
            self.shutdown_requested = True  # Set flag to stop main loop
        else:
            logger.warning(
                f"Shutdown sequence already in progress. Ignoring additional signal ({sig_name}).")

    def _display_startup_info(self) -> None:
        """Displays a summary of the bot's configuration at startup."""
        vt_status_str = f"VT Strategy Enabled: {Style.BRIGHT}{self.config.vt_enable}{Style.NORMAL}"
        if self.config.vt_enable:
            vt_status_str += (f" (TrendEMA:{self.config.vt_trend_ema_period}, VWMA:{self.config.vt_vwma_period}, "
                              f"VolLookback:{self.config.vt_volume_spike_lookback}, VolMultiplier:{self.config.vt_volume_spike_multiplier.normalize()})")

        startup_text_content = Text(
            f"Symbol: {self.config.symbol}\nInterval: {self.config.interval}\n"
            f"Market Type: {self.config.market_type} (V5 Category: {self.config.bybit_v5_category})\n"
            f"Position Index (positionIdx): {self.config.position_idx} (0=One-Way, 1=HedgeMode Long, 2=HedgeMode Short)\n"
            f"Risk Per Trade: {self.config.risk_percentage * 100:.3f}%\n"
            f"SL/TP Multipliers (ATR-based): SL={self.config.sl_atr_multiplier.normalize()}, TP={self.config.tp_atr_multiplier.normalize()}\n"
            f"TSL Activation (ATR Multiplier): {self.config.tsl_activation_atr_multiplier.normalize()}, TSL Distance Percent: {self.config.trailing_stop_percent.normalize()}%\n"
            f"Trade Only With Trend (Original Strategy): {Style.BRIGHT}{self.config.trade_only_with_trend}{Style.NORMAL}\n{vt_status_str}\n"
            f"Journaling Enabled: {Style.BRIGHT}{self.config.enable_journaling}{Style.NORMAL} (File: '{self.config.journal_file_path}')\n"
            # Use the determined display name
            f"Current Log Level: {log_level_display_name}\n"
            f"Close Positions on Shutdown: {Style.BRIGHT}{self.config.close_positions_on_shutdown}{Style.NORMAL}",
            style="bright_white"  # Default style for the text block
        )
        console.print(Panel(startup_text_content,
                      title="[bold cyan]Pyrmethus Configuration Summary[/]", border_style="cyan", expand=False))

    def run(self) -> None:
        """Main execution loop of the trading bot."""
        self._display_startup_info()  # Show config summary
        termux_notify(f"Pyrmethus Started",
                      f"Trading {self.config.symbol} on {self.config.interval} interval.")
        cycle_count = 0

        while not self.shutdown_requested:
            cycle_count += 1
            cycle_start_time_monotonic = time.monotonic()  # For measuring cycle duration
            logger.debug(
                f"{Fore.BLUE}--- Trading Cycle {cycle_count} Started ---{Style.RESET_ALL}")

            try:
                # Execute one trading cycle
                self.trading_spell_cycle(cycle_count)
            except KeyboardInterrupt:  # Specifically handle Ctrl+C during a cycle
                logger.warning(
                    "\nKeyboardInterrupt detected during trading cycle. Initiating graceful shutdown.")
                self.shutdown_requested = True
                break  # Exit main loop
            except ccxt.AuthenticationError as auth_err:  # Critical auth errors stop the bot
                logger.critical(
                    f"{Style.BRIGHT}{Fore.RED}CRITICAL AUTHENTICATION ERROR (Cycle {cycle_count}): {auth_err}. Halting Pyrmethus.{Style.RESET_ALL}", exc_info=False)
                termux_notify("Pyrmethus CRITICAL ERROR",
                              f"Authentication failed: {str(auth_err)[:100]}")
                self.shutdown_requested = True
                break
            # Handle sys.exit calls from within the cycle (e.g., config issues)
            except SystemExit as se:
                logger.warning(
                    f"SystemExit called with code {se.code} during trading cycle. Terminating Pyrmethus.")
                self.shutdown_requested = True
                break
            except Exception as cycle_err:  # Catch any other unhandled errors in the cycle
                logger.error(
                    f"{Style.BRIGHT}{Fore.RED}Unhandled exception in trading cycle {cycle_count}: {cycle_err}{Style.RESET_ALL}", exc_info=True)
                termux_notify(
                    "Pyrmethus Cycle Error", f"Exception in cycle {cycle_count}. Check logs for details.")
                # Sleep longer after an unhandled error before retrying the cycle
                sleep_duration_after_error = self.config.loop_sleep_seconds * 2
                logger.info(
                    f"Sleeping for {sleep_duration_after_error}s after cycle error before attempting next cycle.")
                time.sleep(sleep_duration_after_error)
                continue  # Continue to next cycle iteration

            cycle_duration_seconds = time.monotonic() - cycle_start_time_monotonic
            logger.debug(
                f"Trading Cycle {cycle_count} completed in {cycle_duration_seconds:.2f} seconds.")

            # Sleep until next cycle, respecting shutdown requests
            if not self.shutdown_requested:
                sleep_needed_seconds = max(
                    0, self.config.loop_sleep_seconds - cycle_duration_seconds)
                if sleep_needed_seconds > 0:
                    logger.debug(
                        f"Sleeping for {sleep_needed_seconds:.2f}s until the next trading cycle...")
                    # Sleep in small chunks to allow faster response to shutdown signals
                    sleep_end_time_monotonic = time.monotonic() + sleep_needed_seconds
                    try:
                        while time.monotonic() < sleep_end_time_monotonic and not self.shutdown_requested:
                            # Sleep for max 0.5s or remaining time
                            time.sleep(min(0.5, sleep_needed_seconds))
                    except KeyboardInterrupt:  # Handle Ctrl+C during sleep
                        logger.warning(
                            "\nKeyboardInterrupt detected during sleep. Initiating graceful shutdown.")
                        self.shutdown_requested = True

            if self.shutdown_requested:  # Check flag again after sleep or if loop broke
                logger.info("Shutdown requested. Exiting main trading loop.")
                break

        self.graceful_shutdown()  # Perform cleanup
        console.print(
            f"\n[bold bright_cyan]Pyrmethus ({self.config.symbol}) has completed its session and returned to the ether.[/bold bright_cyan]")

    def trading_spell_cycle(self, cycle_num: int) -> None:
        """
        Executes a single trading cycle: fetch data, calculate indicators,
        check signals, manage positions, and display status.
        """
        # Initialize status dictionary for this cycle's display
        current_status_dict: Dict[str, Any] = {
            "summary": f"Cycle {cycle_num}: Processing...", "orig_detail": "N/A", "vt_detail": "N/A"}

        # 1. Fetch OHLCV Data
        ohlcv_df = self.exchange_manager.fetch_ohlcv()
        if ohlcv_df is None or ohlcv_df.empty:
            logger.error(
                f"{Fore.RED}Cycle {cycle_num} Aborted: Failed to fetch or process valid OHLCV data.{Style.RESET_ALL}")
            current_status_dict.update(
                {"summary": f"FAIL_CYCLE_{cycle_num}:FETCH_OHLCV_ERROR"})
            # Print status even on failure, with available info
            self.status_display.print_status_panel(cycle_num, None, None, None, None, None, current_status_dict,
                                                   self.order_manager.protection_tracker, self.exchange_manager.market_info)
            return

        # Extract current price and timestamp from latest candle
        try:
            latest_candle_data = ohlcv_df.iloc[-1]
            current_price = safe_decimal(latest_candle_data["close"])
            # Convert Pandas Timestamp to Python datetime
            last_candle_timestamp = ohlcv_df.index[-1].to_pydatetime()
            if current_price.is_nan() or current_price <= Decimal("0"):  # Validate price
                raise ValueError(
                    f"Invalid latest close price from OHLCV data: {current_price.normalize() if not current_price.is_nan() else 'NaN'}")
            logger.debug(
                f"Latest Candle Data: Timestamp={last_candle_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}, Price={current_price.normalize()}")
        except (IndexError, KeyError, ValueError, TypeError) as e:
            logger.error(
                f"{Fore.RED}Cycle {cycle_num} Aborted: Error processing latest candle data: {e}{Style.RESET_ALL}")
            current_status_dict.update(
                {"summary": f"FAIL_CYCLE_{cycle_num}:PROCESS_CANDLE_ERROR ({type(e).__name__})"})
            self.status_display.print_status_panel(cycle_num, None, None, None, None, None, current_status_dict,
                                                   self.order_manager.protection_tracker, self.exchange_manager.market_info)
            return

        # 2. Calculate Indicators
        indicators = self.indicator_calculator.calculate_indicators(ohlcv_df)
        # Failed to calculate indicators (e.g., due to insufficient data or error)
        if not indicators:
            logger.error(
                f"{Fore.RED}Cycle {cycle_num} Aborted: Failed to calculate technical indicators.{Style.RESET_ALL}")
            current_status_dict.update(
                {"summary": f"FAIL_CYCLE_{cycle_num}:CALCULATE_INDICATORS_ERROR"})
            self.status_display.print_status_panel(cycle_num, last_candle_timestamp, current_price, None, None,
                                                   None, current_status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info)
            return

        # Add current price to indicators dict for convenience in signal checks (e.g., VT exit signals)
        indicators["close_price"] = current_price

        # 3. Get Account Balance
        # We primarily use total_equity for risk calc
        total_equity, _ = self.exchange_manager.get_balance()
        if total_equity is None or total_equity.is_nan() or total_equity <= Decimal("0"):
            equity_val_str = str(total_equity.normalize(
            ) if total_equity and not total_equity.is_nan() else total_equity)
            logger.error(
                f"{Fore.RED}Cycle {cycle_num} Aborted: Invalid or zero/negative total equity ({equity_val_str}) fetched.{Style.RESET_ALL}")
            current_status_dict.update(
                {"summary": f"FAIL_CYCLE_{cycle_num}:FETCH_EQUITY_INVALID_ERROR"})
            self.status_display.print_status_panel(cycle_num, last_candle_timestamp, current_price, indicators, None,
                                                   total_equity, current_status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info)
            return

        # 4. Get Current Position
        current_positions_summary = self.exchange_manager.get_current_position()
        if current_positions_summary is None:  # Failed to fetch position
            logger.error(
                f"{Fore.RED}Cycle {cycle_num} Aborted: Failed to fetch current position status.{Style.RESET_ALL}")
            current_status_dict.update(
                {"summary": f"FAIL_CYCLE_{cycle_num}:FETCH_POSITION_ERROR"})
            self.status_display.print_status_panel(cycle_num, last_candle_timestamp, current_price, indicators, None,
                                                   total_equity, current_status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info)
            return

        # Determine active position side and details
        active_pos_side: Optional[str] = None
        active_pos_details_dict: Optional[Dict[str, Any]] = None
        if current_positions_summary.get("long", {}):
            active_pos_side, active_pos_details_dict = "long", current_positions_summary[
                "long"]
        elif current_positions_summary.get("short", {}):
            active_pos_side, active_pos_details_dict = "short", current_positions_summary[
                "short"]

        # 5. Position Management (if in a position)
        if active_pos_side and active_pos_details_dict:
            pos_qty = safe_decimal(active_pos_details_dict.get("qty"))
            pos_entry_px = safe_decimal(
                active_pos_details_dict.get("entry_price"))
            current_atr_val = indicators.get(
                "atr", Decimal("NaN"))  # type: ignore[assignment]

            # Manage Trailing Stop Loss (TSL) if applicable
            # Only manage if TSL is not yet active (i.e., on fixed SL/TP) and inputs are valid
            if (self.order_manager.protection_tracker.get(active_pos_side) == PROTECTION_STATE_SLTP and
                    # type: ignore[attr-defined]
                    not any(v.is_nan() or v <= Decimal("0") for v in [pos_entry_px, current_price, current_atr_val])):
                self.order_manager.manage_trailing_stop(
                    # type: ignore[arg-type]
                    active_pos_side, pos_entry_px, current_price, current_atr_val)
                # If TSL was activated, re-fetch position summary for accurate display
                if self.order_manager.protection_tracker.get(active_pos_side) == PROTECTION_STATE_TSL:
                    logger.debug(
                        "Re-fetching position summary after TSL management for status display update.")
                    # Update position details
                    current_positions_summary = self.exchange_manager.get_current_position()
                    if current_positions_summary:
                        active_pos_details_dict = current_positions_summary.get(
                            active_pos_side, {})
                    else:
                        # Could not refetch, status might be slightly stale for display
                        active_pos_details_dict = None

            # Check for Strategy Exit Signals (only if TSL is not active, as TSL is managed by exchange)
            if self.order_manager.protection_tracker.get(active_pos_side) != PROTECTION_STATE_TSL:
                exit_reason_str = self.signal_generator.check_exit_signals(
                    active_pos_side, indicators)
                if exit_reason_str:
                    logger.trade(
                        f"Strategy exit signal detected for {active_pos_side.upper()} position. Reason: {exit_reason_str}")
                    # type: ignore[union-attr]
                    if not pos_qty.is_nan() and pos_qty > Decimal("0"):
                        close_successful = self.order_manager.close_position(
                            # type: ignore[arg-type]
                            active_pos_side, pos_qty, reason=exit_reason_str)
                        # Update status and re-fetch position after close attempt
                        current_status_dict.update(
                            {"summary": f"CLOSED_{active_pos_side.upper()}_BY_SIGNAL: {exit_reason_str.split(':')[0]}" if close_successful else f"FAIL_CLOSE_SIGNAL_{active_pos_side.upper()}"})
                        # Refresh position state
                        current_positions_summary = self.exchange_manager.get_current_position()
                        # Display status and end cycle after attempting closure
                        self.status_display.print_status_panel(cycle_num, last_candle_timestamp, current_price, indicators, current_positions_summary,
                                                               total_equity, current_status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info)
                        return
                    else:  # Should not happen if in a valid position
                        logger.warning(
                            f"Exit signal for {active_pos_side.upper()} but position quantity is invalid ({pos_qty}). Cannot attempt close.")

            # Re-check position state after TSL management or exit signal checks, in case position was closed by exchange SL/TP
            current_positions_after_actions = self.exchange_manager.get_current_position()
            if current_positions_after_actions is None:
                logger.warning(
                    f"Failed to re-fetch position state for {active_pos_side} after TSL/exit checks. Status panel may be slightly stale for this cycle.")
            else:  # Successfully re-fetched, update active position status
                current_positions_summary = current_positions_after_actions  # Use the latest summary
                new_long_pos_data, new_short_pos_data = current_positions_summary.get(
                    "long", {}), current_positions_summary.get("short", {})
                if new_long_pos_data and safe_decimal(new_long_pos_data.get("qty", "0")).copy_abs() >= POSITION_QTY_EPSILON:
                    active_pos_side, active_pos_details_dict = "long", new_long_pos_data
                elif new_short_pos_data and safe_decimal(new_short_pos_data.get("qty", "0")).copy_abs() >= POSITION_QTY_EPSILON:
                    active_pos_side, active_pos_details_dict = "short", new_short_pos_data
                # Position is now flat (e.g., closed by exchange SL/TP or earlier signal)
                else:
                    if active_pos_side:  # If it was previously open this cycle
                        logger.info(
                            f"Position {active_pos_side.upper()} appears to have been closed by exchange stop or an earlier action during this cycle.")
                        current_status_dict.update(
                            {"summary": f"INFO:POS_{active_pos_side.upper()}_CLOSED_BY_EXCHANGE_OR_PREV_SIGNAL"})
                        # Reset tracker
                        self.order_manager.protection_tracker[active_pos_side.lower(
                        )] = None
                    active_pos_side, active_pos_details_dict = None, None  # Now flat

        # 6. Entry Logic (if currently flat)
        # If flat (either initially, or after an exit this cycle)
        if not active_pos_side:
            logger.debug("Currently flat. Checking for new entry signals...")
            entry_signals_result_dict = self.signal_generator.generate_signals(
                ohlcv_df, indicators)
            # Update main status dict with signal generation outcome
            current_status_dict = entry_signals_result_dict

            entry_order_side_str: Optional[str] = "buy" if entry_signals_result_dict.get(
                "long") else "sell" if entry_signals_result_dict.get("short") else None
            if entry_order_side_str:  # An entry signal was generated
                current_atr_val_for_entry = indicators.get(
                    "atr", Decimal("NaN"))  # type: ignore[assignment]
                # type: ignore[union-attr]
                if not current_atr_val_for_entry.is_nan() and current_atr_val_for_entry > Decimal("0"):
                    entry_successful = False
#                    entry_successful = self.order_manager.place_risked_market_order(
#                        # type: ignore[arg-type]
#                        entry_order_side_str, current_atr_val_for_entry, total_equity, current_price)

                    entered_side_logical_str = "long" if entry_order_side_str == "buy" else "short"
                    current_status_dict.update(
                        {"summary": f"ENTERED_{entered_side_logical_str.upper()}_POSITION" if entry_successful else f"FAIL_ENTRY_{entered_side_logical_str.upper()}"})
                    # Refresh position state after entry attempt
                    current_positions_summary = self.exchange_manager.get_current_position()
                    # Display status and end cycle after attempting entry
                    self.status_display.print_status_panel(cycle_num, last_candle_timestamp, current_price, indicators, current_positions_summary,
                                                           total_equity, current_status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info)
                    return
                else:  # ATR invalid, cannot calculate risk for entry
                    logger.warning(
                        f"Cannot attempt {entry_order_side_str.upper()} entry: Missing or invalid ATR ({current_atr_val_for_entry}). Risk calculation not possible.")
                    current_status_dict.update(
                        {"summary": f"FAIL_ENTRY_DATA_MISSING_ATR_{entry_order_side_str.upper()}"})
        else:  # Still in an active position, and no exit signal triggered this cycle
            current_status_dict.update(
                {"summary": f"HOLDING_{active_pos_side.upper()}_POSITION"})

        # 7. Display Final Status for the Cycle
        self.status_display.print_status_panel(cycle_num, last_candle_timestamp, current_price, indicators, current_positions_summary,
                                               total_equity, current_status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info)

    def graceful_shutdown(self) -> None:
        """Performs cleanup actions on bot shutdown, like cancelling orders and closing positions."""
        console.print(
            "\n[bold yellow]Initiating Graceful Shutdown Sequence...[/bold yellow]")
        logger.warning(
            f"{Fore.YELLOW}{Style.BRIGHT}Initiating Graceful Shutdown Sequence for Pyrmethus...{Style.RESET_ALL}")
        termux_notify("Pyrmethus Shutting Down",
                      f"Graceful shutdown initiated for {self.config.symbol}...")

        # Ensure exchange components are available for shutdown actions
        if not self.exchange_manager or not self.exchange_manager.exchange or not self.exchange_manager.market_info:
            logger.error(
                f"{Fore.RED}Cannot perform full graceful shutdown: ExchangeManager or its components (exchange, market_info) are not fully initialized.{Style.RESET_ALL}")
            termux_notify("Pyrmethus Shutdown Warning!",
                          f"{self.config.symbol}: Cannot perform full clean shutdown due to initialization issues!")
            return

        exchange = self.exchange_manager.exchange
        symbol_to_manage = self.config.symbol
        settle_coin_for_api = self.exchange_manager.market_info.get(
            "settle")  # Needed for some V5 calls

        # 1. Cancel all active non-positional orders (e.g., limit orders if any were used)
        logger.info(f"{Fore.CYAN}Attempting to cancel all active non-positional orders for {symbol_to_manage} (Category: {self.config.bybit_v5_category})...{Style.RESET_ALL}")
        try:
            # Parameters for Bybit V5 cancelAllOrders. Category is key.
            # Symbol or settleCoin can act as filters.
            cancel_params_v5 = {"category": self.config.bybit_v5_category}
            if self.config.bybit_v5_category in ["linear", "inverse"]:
                # Symbol is usually a good filter for derivatives
                cancel_params_v5["symbol"] = symbol_to_manage
            elif self.config.bybit_v5_category == "spot":
                cancel_params_v5["symbol"] = symbol_to_manage
            # For "option", baseCoin might be more relevant.
            # If settleCoin is available and useful as a filter:
            # if settle_coin_for_api: cancel_params_v5["settleCoin"] = settle_coin_for_api

            # Use fetch_with_retries for robustness, but with fewer retries for shutdown
            cancel_response = fetch_with_retries(
                exchange.cancel_all_orders,  # CCXT unified method
                # Pass symbol, CCXT might use it or ignore if params are more specific
                symbol=symbol_to_manage,
                params=cancel_params_v5,
                max_retries=1, delay_seconds=1  # Quick attempt
            )
            logger.info(
                f"Cancel all active orders API response (first 200 chars): {str(cancel_response)[:200]}...")
        except ccxt.NotSupported:  # If exchange/category doesn't support cancel_all_orders this way
            logger.warning(
                f"Exchange {exchange.id} does not support cancel_all_orders with the current parameters or for category '{self.config.bybit_v5_category}'. Skipping cancellation of non-positional orders.")
        except Exception as e:
            # exc_info=False for brevity
            logger.error(
                f"{Fore.RED}Error attempting to cancel active orders during shutdown: {e}{Style.RESET_ALL}", exc_info=False)

        # Reset local protection tracker as we are shutting down
        self.order_manager.protection_tracker = {"long": None, "short": None}
        logger.info(
            "Brief pause after order cancellation attempt to allow exchange processing...")
        # Wait a bit
        time.sleep(max(self.config.order_check_delay_seconds, 2.0))

        # 2. Close any lingering positions if configured to do so
        if self.config.close_positions_on_shutdown:
            logger.info(
                f"{Fore.CYAN}Checking for and closing any lingering positions for {symbol_to_manage} as per configuration...{Style.RESET_ALL}")
            closed_positions_count = 0
            # List of (side_str, qty_decimal)
            positions_to_attempt_close: List[Tuple[str, Decimal]] = []

            try:
                final_positions_summary = self.exchange_manager.get_current_position()
                if final_positions_summary:
                    for side_key_str in ["long", "short"]:
                        pos_data_dict = final_positions_summary.get(
                            side_key_str)
                        if pos_data_dict and safe_decimal(pos_data_dict.get("qty", "0")).copy_abs() >= POSITION_QTY_EPSILON:
                            qty_to_close_decimal = safe_decimal(
                                pos_data_dict.get("qty", "0")).copy_abs()
                            logger.warning(
                                f"{Fore.YELLOW}Found lingering {side_key_str.upper()} position (Qty: {qty_to_close_decimal.normalize()}) that will be closed due to shutdown configuration.{Style.RESET_ALL}")
                            positions_to_attempt_close.append(
                                (side_key_str, qty_to_close_decimal))

                    if not positions_to_attempt_close:
                        logger.info(
                            f"{Fore.GREEN}No significant open positions found for {symbol_to_manage} requiring closure.{Style.RESET_ALL}")
                    else:
                        for side_str, qty_decimal in positions_to_attempt_close:
                            logger.info(
                                f"Attempting to close {side_str.upper()} position (Qty: {qty_decimal.normalize()})...")
                            close_successful = self.order_manager.close_position(
                                side_str, qty_decimal, reason="GracefulShutdownClose")
                            if close_successful:
                                closed_positions_count += 1
                                logger.info(
                                    f"{Fore.GREEN}Closure initiated and/or confirmed for {side_str.upper()} position.{Style.RESET_ALL}")
                            else:
                                logger.error(
                                    f"{Fore.RED}Closure FAILED for {side_str.upper()} position. MANUAL INTERVENTION REQUIRED for this position.{Style.RESET_ALL}")

                        if closed_positions_count == len(positions_to_attempt_close):
                            logger.info(
                                f"{Fore.GREEN}All detected positions ({closed_positions_count}) were closed successfully or closure was initiated.{Style.RESET_ALL}")
                        else:
                            logger.warning(
                                f"{Fore.YELLOW}Attempted to close {len(positions_to_attempt_close)} positions, but only {closed_positions_count} succeeded or initiated. MANUAL VERIFICATION AND INTERVENTION REQUIRED for any remaining positions.{Style.RESET_ALL}")
                else:  # Failed to fetch positions during shutdown
                    logger.error(
                        f"{Fore.RED}Failed to fetch current positions during shutdown sequence for {symbol_to_manage}. MANUAL CHECK OF POSITION STATUS URGENTLY REQUIRED!{Style.RESET_ALL}")
                    termux_notify(f"{symbol_to_manage} Shutdown Issue",
                                  "Failed to fetch positions during shutdown! Manual verification needed.")
            except Exception as e:  # Catch any error during position closure phase
                logger.error(
                    f"{Fore.RED}{Style.BRIGHT}Error occurred during position closure phase of shutdown: {e}. MANUAL CHECK OF POSITIONS URGENTLY REQUIRED.{Style.RESET_ALL}", exc_info=True)
                termux_notify(f"{symbol_to_manage} Shutdown Issue",
                              f"Error closing positions: {str(e)[:50]}")
        else:  # Configured not to close positions
            logger.info(
                "Configuration set to NOT close positions on shutdown. Any open positions will remain active.")

        console.print(
            "[bold yellow]Graceful Shutdown Sequence Complete.[/bold yellow]")
        logger.warning(
            f"{Fore.YELLOW}{Style.BRIGHT}Graceful Shutdown Sequence Complete. Pyrmethus rests its spell.{Style.RESET_ALL}")
        termux_notify("Pyrmethus Shutdown Complete",
                      f"{self.config.symbol} trading session has ended.")


if __name__ == "__main__":
    try:
        bot = TradingBot()
        bot.run()
    except SystemExit as e:  # Catch sys.exit calls
        # SystemExit with code 0 is often a normal, albeit early, termination (e.g., config validation failure handled by sys.exit)
        if e.code == 0:
            logger.info(
                "Pyrmethus terminated normally (e.g., due to pre-run validation or planned exit).")
        else:
            logger.warning(f"Pyrmethus terminated with exit code: {e.code}.")
        sys.exit(e.code)  # Propagate the original exit code
    except Exception as main_execution_exception:
        # Ensure logger is available; if not (e.g., error before logger setup), fallback to print
        log_func = logger.critical if 'logger' in globals() and hasattr(logger,
                                                                        'critical') else print

        error_message_prefix = "CRITICAL UNHANDLED EXCEPTION in Pyrmethus main execution block:"
        # Check if colorama was successfully imported for colored output
        if _COLORAMA_SUCCESSFULLY_IMPORTED:
            formatted_error_message = f"{Style.BRIGHT}{Fore.RED}{error_message_prefix} {main_execution_exception}{Style.RESET_ALL}"
        else:
            formatted_error_message = f"{error_message_prefix} {main_execution_exception}"

        # Log with full traceback for debugging
        log_func(formatted_error_message, exc_info=True)

        # Send Termux notification about the crash if termux_notify is available
        if 'termux_notify' in globals() and callable(termux_notify):
            termux_notify(
                "Pyrmethus CRASHED!", "A critical unhandled exception occurred. Please check logs immediately!")

        sys.exit(1)  # Exit with a non-zero code to indicate an error
