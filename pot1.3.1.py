```python
# --*- coding: utf-8 -*-
# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals, too-many-public-methods, invalid-name, unused-argument, too-many-lines, unnecessary-pass
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

            if "pandas" in COMMON_PACKAGES: # type: ignore[operator]
                termux_pkgs_to_install.append("python-pandas")
                if 'pandas' in pip_pkgs_to_install:
                    pip_pkgs_to_install.remove('pandas')
            if "numpy" in COMMON_PACKAGES: # type: ignore[operator]
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
# Default Trailing Stop Loss distance as percentage of current price (e.g., 0.005 is 0.5%)
DEFAULT_TSL_PERCENT = Decimal("0.005")
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
        if not str_value or str_value.lower() in ("nan", "none", "null", ""): # Added empty string check
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
        # Termux-toast primarily uses content; title can be prepended if desired.
        full_content = f"{title}: {content}" if title else content
        result = subprocess.run(
            ["termux-toast", full_content],
            check=False,
            timeout=TERMUX_NOTIFY_TIMEOUT,
            capture_output=True, text=True
        )
        if result.returncode != 0:
            error_output_parts = []
            if result.stdout and result.stdout.strip(): error_output_parts.append(f"STDOUT: {result.stdout.strip()}")
            if result.stderr and result.stderr.strip(): error_output_parts.append(f"STDERR: {result.stderr.strip()}")
            error_details = " | ".join(error_output_parts) if error_output_parts else "[No output from termux-toast command]"
            logger.warning(
                "Termux toast command failed (code %s): %s", result.returncode, error_details)
    except FileNotFoundError:
        logger.warning(
            "Termux notify failed: 'termux-toast' command not found. Is Termux:API installed and configured?")
    except subprocess.TimeoutExpired:
        logger.warning(
            "Termux notify failed: 'termux-toast' command timed out after %s seconds.", TERMUX_NOTIFY_TIMEOUT)
    except Exception as e:
        logger.warning(
            "Termux notify failed unexpectedly: %s (%s)", e, type(e).__name__, exc_info=False)


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
            if attempt > 0:
                logger.info(
                    "%s%sSuccessfully executed %s on attempt %s/%s.%s", Style.BRIGHT, Fore.GREEN, func_name, attempt + 1, max_retries + 1, Style.RESET_ALL)
            return result
        except fatal_exceptions as e:
            logger.critical(
                "%s%sFatal error (%s) executing %s: %s. Halting.%s", Style.BRIGHT, Fore.RED, type(e).__name__, func_name, e, Style.RESET_ALL, exc_info=False)
            raise
        except fail_fast_exceptions as e:
            logger.error(
                "%sFail-fast error (%s) executing %s: %s. Not retrying this call.%s", Fore.RED, type(e).__name__, func_name, e, Style.RESET_ALL, exc_info=False)
            last_exception = e
            break
        except retry_on_exceptions as e:
            last_exception = e
            err_summary = str(e)[:150] + ("..." if len(str(e)) > 150 else "")
            msg_base = f"Retryable error ({type(e).__name__}) on attempt {attempt + 1}/{max_retries + 1} for {func_name}: {err_summary}."
            if attempt < max_retries:
                logger.warning(
                    "%s%s Retrying in %ss...%s", Fore.YELLOW, msg_base, delay_seconds, Style.RESET_ALL, exc_info=False)
                time.sleep(delay_seconds)
            else:
                logger.error(
                    "%sMax retries (%s) reached for %s. Last error: %s%s", Fore.RED, max_retries + 1, func_name, e, Style.RESET_ALL, exc_info=False)
        except ccxt.ExchangeError as e:  # Catch other exchange errors
            last_exception = e
            err_summary = str(e)[:150] + ("..." if len(str(e)) > 150 else "")
            logger.error(
                "%sUnhandled ExchangeError during %s: %s%s", Fore.RED, func_name, err_summary, Style.RESET_ALL, exc_info=False)
            if attempt < max_retries: # Retry generic exchange errors too
                logger.warning(
                    "Retrying generic exchange error in %ss...", delay_seconds)
                time.sleep(delay_seconds)
            else:
                logger.error(
                    "Max retries reached after generic exchange error for %s.", func_name)
                break
        except Exception as e:  # Catch any other Python exceptions
            last_exception = e
            logger.error(
                "%sUnexpected Python error during %s: %s%s", Fore.RED, func_name, e, Style.RESET_ALL, exc_info=True)
            break # Do not retry unexpected Python errors

    if last_exception:
        raise last_exception
    raise RuntimeError(
        f"Function {func_name} failed after {max_retries + 1} attempts without returning or raising a recognized exception. This indicates an issue with the retry logic or unhandled exception types.")

# --- Configuration Class ---


class TradingConfig:
    """Handles loading, validation, and storage of trading configuration parameters."""
    # pylint: disable=too-many-statements

    def __init__(self, env_file: str = ".env"):
        logger.debug(
            "Loading configuration from environment variables / '%s'...", env_file)
        env_path = Path(env_file)
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path, override=True)
            logger.info("Loaded configuration from %s", env_path)
        else:
            logger.warning(
                "Environment file '%s' not found. Relying solely on system environment variables.", env_path)

        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT", Style.DIM,
                                         help_text="Trading symbol, e.g., BTC/USDT:USDT for Bybit V5 linear futures.")
        self.market_type: str = self._get_env("MARKET_TYPE", "linear", Style.DIM, allowed_values=[
                                              "linear", "inverse", "swap"], help_text="Market type (linear, inverse, swap). Effects V5 category.").lower()
        self.bybit_v5_category: str = self._determine_v5_category()

        self.interval: str = self._get_env(
            "INTERVAL", "1m", Style.DIM, help_text="Candle interval, e.g., '1m', '5m', '1h'.")
        self.risk_percentage: Decimal = self._get_env("RISK_PERCENTAGE", DEFAULT_RISK_PERCENT, Fore.YELLOW, cast_type=Decimal, min_val=Decimal(
            "0.00001"), max_val=Decimal("0.5"), help_text="Account equity percentage to risk per trade.")
        self.position_idx: int = self._get_env("POSITION_IDX", DEFAULT_POSITION_IDX, Style.DIM, cast_type=int, allowed_values=[
                                               0, 1, 2], help_text="Bybit V5 position index (0: One-Way, 1: Hedge Long, 2: Hedge Short).")

        self.sl_atr_multiplier: Decimal = self._get_env("SL_ATR_MULTIPLIER", DEFAULT_SL_MULT, Fore.YELLOW, cast_type=Decimal, min_val=Decimal(
            "0.1"), max_val=Decimal("20.0"), help_text="ATR multiplier for initial Stop Loss distance.")
        self.tp_atr_multiplier: Decimal = self._get_env("TP_ATR_MULTIPLIER", DEFAULT_TP_MULT, Fore.YELLOW, cast_type=Decimal, min_val=Decimal(
            "0.0"), max_val=Decimal("50.0"), help_text="ATR multiplier for Take Profit distance (0 to disable TP).")
        self.tsl_activation_atr_multiplier: Decimal = self._get_env("TSL_ACTIVATION_ATR_MULTIPLIER", DEFAULT_TSL_ACT_MULT, Fore.YELLOW, cast_type=Decimal, min_val=Decimal(
            "0.1"), max_val=Decimal("20.0"), help_text="ATR multiplier from entry price to activate Trailing Stop Loss.")
        self.trailing_stop_percent: Decimal = self._get_env("TRAILING_STOP_PERCENT", DEFAULT_TSL_PERCENT, Fore.YELLOW, cast_type=Decimal, min_val=Decimal(
            "0.0001"), max_val=Decimal("0.1"), help_text="Trailing Stop Loss distance as a percentage of current price (e.g., 0.005 for 0.5%, max 10%) once activated.")
        self.sl_trigger_by: str = self._get_env("SL_TRIGGER_BY", "LastPrice", Style.DIM, allowed_values=[
                                                "LastPrice", "MarkPrice", "IndexPrice"], help_text="Price type to trigger SL.")
        self.tsl_trigger_by: str = self._get_env("TSL_TRIGGER_BY", "LastPrice", Style.DIM, allowed_values=[
                                                 "LastPrice", "MarkPrice", "IndexPrice"], help_text="Price type for TSL trigger price (Bybit V5 TSL often trails LastPrice, this is for activation check).")

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

        self.stoch_oversold_threshold: Decimal = self._get_env(
            "STOCH_OVERSOLD_THRESHOLD", DEFAULT_STOCH_OVERSOLD, Fore.CYAN, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("45"))
        self.stoch_overbought_threshold: Decimal = self._get_env(
            "STOCH_OVERBOUGHT_THRESHOLD", DEFAULT_STOCH_OVERBOUGHT, Fore.CYAN, cast_type=Decimal, min_val=Decimal("55"), max_val=Decimal("100"))
        self.trend_filter_buffer_percent: Decimal = self._get_env("TREND_FILTER_BUFFER_PERCENT", Decimal(
            "0.005"), Fore.CYAN, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("0.05"), help_text="Buffer percent (e.g. 0.005 for 0.5%) around trend EMA for trend filter.")
        self.atr_move_filter_multiplier: Decimal = self._get_env("ATR_MOVE_FILTER_MULTIPLIER", Decimal("0.5"), Fore.CYAN, cast_type=Decimal, min_val=Decimal(
            "0"), max_val=Decimal("5"), help_text="Multiplier for ATR to define significant price move (0 to disable).")
        self.min_adx_level: Decimal = self._get_env("MIN_ADX_LEVEL", DEFAULT_MIN_ADX, Fore.CYAN, cast_type=Decimal, min_val=Decimal(
            "0"), max_val=Decimal("90"), help_text="Minimum ADX value to consider trend strong enough.")
        self.trade_only_with_trend: bool = self._get_env(
            "TRADE_ONLY_WITH_TREND", True, Style.DIM, cast_type=bool, help_text="If true, original strategy entries must align with trend EMA.")

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
        category: str
        if self.market_type == "inverse": category = "inverse"
        elif self.market_type in ["linear", "swap"]: category = "linear"
        else:
            err_msg = f"Unsupported MARKET_TYPE '{self.market_type}' for V5 category. Supported: linear, inverse, swap."
            logger.critical("%s%s%s Halting.%s", Style.BRIGHT, Fore.RED, err_msg, Style.RESET_ALL); sys.exit(1)

        if category in ["linear", "inverse"] and ":" not in self.symbol:
            logger.warning(
                "Symbol '%s' for %s futures does not explicitly include settle currency (e.g., :USDT). Format BASE/QUOTE:SETTLE (e.g., BTC/USDT:USDT) is recommended for Bybit V5 clarity.", self.symbol, category)
        # Further optional checks for symbol format vs category
        if ":" in self.symbol:
            base_quote, settle = self.symbol.split(':', 1)
            base_ccy = base_quote.split('/',1)[0]
            if category == "linear" and settle == base_ccy:
                 logger.warning("Symbol '%s' (linear) has settle currency '%s' matching base. Unusual (expected e.g. :USDT). Verify.", self.symbol, settle)
            if category == "inverse" and settle != base_ccy:
                 logger.warning("Symbol '%s' (inverse) has settle currency '%s' not matching base '%s'. Unusual (expected e.g. :%s). Verify.", self.symbol, settle, base_ccy, base_ccy)

        logger.info("Determined Bybit V5 API category: '%s' for symbol '%s', market type '%s'", category, self.symbol, self.market_type)
        return category

    def _validate_config(self) -> None:
        """Performs cross-parameter validation for the loaded configuration."""
        if self.fast_ema_period >= self.slow_ema_period:
            logger.critical("%s%sValidation failed: FAST_EMA_PERIOD (%s) must be < SLOW_EMA_PERIOD (%s). Halting.%s", Style.BRIGHT, Fore.RED, self.fast_ema_period, self.slow_ema_period, Style.RESET_ALL); sys.exit(1)
        if self.trend_ema_period <= self.slow_ema_period:
            logger.warning("%sConfig Warning: TREND_EMA_PERIOD (%s) not > SLOW_EMA_PERIOD (%s). Unusual configuration.%s", Fore.YELLOW, self.trend_ema_period, self.slow_ema_period, Style.RESET_ALL)
        if self.stoch_oversold_threshold >= self.stoch_overbought_threshold:
            logger.critical("%s%sValidation failed: STOCH_OVERSOLD (%s) must be < STOCH_OVERBOUGHT (%s). Halting.%s", Style.BRIGHT, Fore.RED, self.stoch_oversold_threshold.normalize(), self.stoch_overbought_threshold.normalize(), Style.RESET_ALL); sys.exit(1)

        if self.tsl_activation_atr_multiplier < self.sl_atr_multiplier:
            logger.warning("%sConfig Warning: TSL_ACTIVATION_ATR_MULT (%s) < SL_ATR_MULT (%s). TSL might activate before initial SL distance (ATR terms) reached.%s", Fore.YELLOW, self.tsl_activation_atr_multiplier.normalize(), self.sl_atr_multiplier.normalize(), Style.RESET_ALL)
        if self.tp_atr_multiplier > Decimal("0") and self.tp_atr_multiplier <= self.sl_atr_multiplier:
            logger.warning("%sConfig Warning: TP_ATR_MULT (%s) <= SL_ATR_MULT (%s). Risk:Reward <= 1:1.%s", Fore.YELLOW, self.tp_atr_multiplier.normalize(), self.sl_atr_multiplier.normalize(), Style.RESET_ALL)
        if self.trailing_stop_percent >= Decimal("0.2"): # 20% TSL distance is very large
            logger.warning("%sConfig Warning: TRAILING_STOP_PERCENT (%s%%) is very large. Ensure intended.%s", Fore.YELLOW, (self.trailing_stop_percent * Decimal(100)).normalize(), Style.RESET_ALL)

        if self.vt_enable:
            if self.vt_trend_ema_period < self.vt_vwma_period:
                logger.warning("%sConfig Warning (VT): VT_TREND_EMA_PERIOD (%s) < VT_VWMA_PERIOD (%s). Unusual; ensure intended
