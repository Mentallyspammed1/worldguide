# -*- coding: utf-8 -*-
# pylint: disable=logging-fstring-interpolation, too-many-instance-attributes, too-many-arguments, too-many-locals, too-many-public-methods, invalid-name, unused-argument, too-many-lines, unnecessary-pass, unnecessary-lambda-assignment, line-too-long, wrong-import-order, wrong-import-position, bad-option-value
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

Enhancements in this version (4.5.7+):
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
# import platform # Not used in the provided code, can be removed if not needed later
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
import types # For signal frame type hint

# Third-Party Imports
# Define COMMON_PACKAGES before the try-except block for imports
COMMON_PACKAGES = [
    "ccxt",
    "python-dotenv",
    "pandas",
    "numpy",
    "rich",
    "colorama",
    "requests",
]
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
        is_termux = "com.termux" in os.environ.get("PREFIX", "")

        if is_termux:
            termux_pkgs_to_install = []
            pip_pkgs_to_install = list(COMMON_PACKAGES) # Start with all packages for pip

            if "pandas" in COMMON_PACKAGES:
                termux_pkgs_to_install.append("python-pandas")
                if 'pandas' in pip_pkgs_to_install: pip_pkgs_to_install.remove('pandas')
            if "numpy" in COMMON_PACKAGES:
                termux_pkgs_to_install.append("python-numpy")
                if 'numpy' in pip_pkgs_to_install: pip_pkgs_to_install.remove('numpy')

            install_cmd_parts = []
            if termux_pkgs_to_install:
                install_cmd_parts.append(f"pkg install python {' '.join(termux_pkgs_to_install)}")
            if pip_pkgs_to_install:
                install_cmd_parts.append(f"pip install {' '.join(pip_pkgs_to_install)}")

            install_cmd = " && ".join(install_cmd_parts) if install_cmd_parts else f"pip install {' '.join(COMMON_PACKAGES)}"
            print(f"{Style.BRIGHT}{install_cmd}{Style.RESET_ALL}")

            if termux_pkgs_to_install:
                termux_base_names = [pkg.replace('python-','') for pkg in termux_pkgs_to_install]
                print(
                    f"{Fore.YELLOW}Note: In Termux, {' and '.join(termux_base_names)} are often best installed via 'pkg' for compatibility.{Style.RESET_ALL}"
                )
        else: # Standard pip install for other systems
            print(
                f"{Style.BRIGHT}pip install {' '.join(COMMON_PACKAGES)}{Style.RESET_ALL}"
            )
        sys.exit(1)

# --- Constants ---
DECIMAL_PRECISION = 50
POSITION_QTY_EPSILON = Decimal("1E-12")
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
TERMUX_NOTIFY_TIMEOUT = 10

colorama_init(autoreset=True)
console = Console(log_path=False)
getcontext().prec = DECIMAL_PRECISION

# --- Logging Setup ---
TRADE_LEVEL_NUM = 25
log_level_display_name: str # Global for startup info
if not hasattr(logging, "TRADE"):
    logging.addLevelName(TRADE_LEVEL_NUM, "TRADE")
if not hasattr(logging.Logger, "trade"):
    def trade_log(self: logging.Logger, message: str, *args: Any, **kws: Any) -> None:
        if self.isEnabledFor(TRADE_LEVEL_NUM):
            self._log(TRADE_LEVEL_NUM, message, args, **kws) # type: ignore[arg-type] # pylint: disable=protected-access
    logging.Logger.trade = trade_log # type: ignore[attr-defined]

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
if not logger.hasHandlers():
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
logger.propagate = False

# --- Utility Functions ---
def safe_decimal(value: Any, default: Decimal = Decimal("NaN")) -> Decimal:
    if value is None: return default
    try:
        str_value = str(value).strip()
        if not str_value: return default
        if str_value.lower() in ["nan", "none", "null"]: return default
        return Decimal(str_value)
    except (InvalidOperation, ValueError, TypeError):
        # logger.debug(f"safe_decimal: Could not convert '{value}' to Decimal, using default {default}")
        return default

def termux_notify(title: str, content: str) -> None:
    if "com.termux" in os.environ.get("PREFIX", ""):
        try:
            result = subprocess.run(
                ["termux-toast", content], check=False, timeout=TERMUX_NOTIFY_TIMEOUT,
                capture_output=True, text=True
            )
            if result.returncode != 0:
                error_output = result.stderr.strip() if result.stderr else result.stdout.strip()
                logger.warning(f"Termux toast command failed (code {result.returncode}): {error_output if error_output else '[No output]'}")
        except FileNotFoundError:
            logger.warning("Termux notify failed: 'termux-toast' not found.")
        except subprocess.TimeoutExpired:
            logger.warning(f"Termux notify failed: command timed out after {TERMUX_NOTIFY_TIMEOUT}s.")
        except Exception as e:
            logger.warning(f"Termux notify failed unexpectedly: {e}", exc_info=False)

def fetch_with_retries(
    fetch_function: Callable[..., Any], *args: Any,
    max_retries: int = DEFAULT_MAX_RETRIES, delay_seconds: int = DEFAULT_RETRY_DELAY,
    retry_on_exceptions: Tuple[Type[Exception], ...] = (
        ccxt.DDoSProtection, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable,
        ccxt.NetworkError, ccxt.RateLimitExceeded, requests.exceptions.ConnectionError,
        requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError,
        requests.exceptions.ReadTimeout),
    fatal_exceptions: Tuple[Type[Exception], ...] = (ccxt.AuthenticationError, ccxt.PermissionDenied),
    fail_fast_exceptions: Tuple[Type[Exception], ...] = (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.OrderNotFound),
    **kwargs: Any
) -> Any:
    last_exception: Optional[Exception] = None
    func_name = getattr(fetch_function, "__name__", "Unnamed function")
    for attempt in range(max_retries + 1):
        try:
            result = fetch_function(*args, **kwargs)
            if attempt > 0:
                logger.info(f"{Style.BRIGHT}{Fore.GREEN}Successfully executed {func_name} on attempt {attempt + 1}/{max_retries + 1}.{Style.RESET_ALL}")
            return result
        except fatal_exceptions as e:
            logger.critical(f"{Style.BRIGHT}{Fore.RED}Fatal error ({type(e).__name__}) executing {func_name}: {e}. Halting.{Style.RESET_ALL}", exc_info=False)
            raise e
        except fail_fast_exceptions as e:
            logger.error(f"{Fore.RED}Fail-fast error ({type(e).__name__}) executing {func_name}: {e}. Not retrying.{Style.RESET_ALL}", exc_info=False)
            last_exception = e; break
        except retry_on_exceptions as e:
            last_exception = e
            err_summary = str(e)[:150] + ("..." if len(str(e)) > 150 else "")
            msg_base = f"Retryable error ({type(e).__name__}) on attempt {attempt + 1}/{max_retries + 1} for {func_name}: {err_summary}."
            if attempt < max_retries:
                logger.warning(f"{Fore.YELLOW}{msg_base} Retrying in {delay_seconds}s...{Style.RESET_ALL}")
                time.sleep(delay_seconds)
            else:
                logger.error(f"{Fore.RED}Max retries ({max_retries + 1}) reached for {func_name}. Last error: {e}{Style.RESET_ALL}", exc_info=False)
        except ccxt.ExchangeError as e:
            last_exception = e
            err_summary = str(e)[:150] + ("..." if len(str(e)) > 150 else "")
            logger.error(f"{Fore.RED}Unhandled ExchangeError during {func_name}: {err_summary}{Style.RESET_ALL}", exc_info=False)
            if attempt < max_retries:
                logger.warning(f"Retrying generic exchange error in {delay_seconds}s...")
                time.sleep(delay_seconds)
            else:
                logger.error(f"Max retries reached after generic exchange error for {func_name}."); break
        except Exception as e:
            last_exception = e
            logger.error(f"{Fore.RED}Unexpected error during {func_name}: {e}{Style.RESET_ALL}", exc_info=True)
            break
    if last_exception: raise last_exception
    raise RuntimeError(f"Function {func_name} failed after {max_retries + 1} attempts without specific captured exception or valid return.")

# --- Configuration Class ---
class TradingConfig:
    # pylint: disable=too-many-statements
    def __init__(self, env_file: str = ".env"):
        logger.debug(f"Loading configuration from environment variables / '{env_file}'...")
        env_path = Path(env_file)
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path, override=True)
            logger.info(f"Loaded configuration from {env_path}")
        else:
            logger.warning(f"Environment file '{env_path}' not found. Relying on system environment variables.")

        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT", Style.DIM)
        self.market_type: str = self._get_env("MARKET_TYPE", "linear", Style.DIM, allowed_values=["linear", "inverse", "swap"]).lower()
        self.bybit_v5_category: str = self._determine_v5_category()
        self.interval: str = self._get_env("INTERVAL", "1m", Style.DIM)
        self.risk_percentage: Decimal = self._get_env("RISK_PERCENTAGE", DEFAULT_RISK_PERCENT, Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.00001"), max_val=Decimal("0.5"))
        self.sl_atr_multiplier: Decimal = self._get_env("SL_ATR_MULTIPLIER", DEFAULT_SL_MULT, Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.1"), max_val=Decimal("20.0"))
        self.tp_atr_multiplier: Decimal = self._get_env("TP_ATR_MULTIPLIER", DEFAULT_TP_MULT, Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.0"), max_val=Decimal("50.0"))
        self.tsl_activation_atr_multiplier: Decimal = self._get_env("TSL_ACTIVATION_ATR_MULTIPLIER", DEFAULT_TSL_ACT_MULT, Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.1"), max_val=Decimal("20.0"))
        self.trailing_stop_percent: Decimal = self._get_env("TRAILING_STOP_PERCENT", DEFAULT_TSL_PERCENT, Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.001"), max_val=Decimal("10.0"))
        self.sl_trigger_by: str = self._get_env("SL_TRIGGER_BY", "LastPrice", Style.DIM, allowed_values=["LastPrice", "MarkPrice", "IndexPrice"])
        self.tsl_trigger_by: str = self._get_env("TSL_TRIGGER_BY", "LastPrice", Style.DIM, allowed_values=["LastPrice", "MarkPrice", "IndexPrice"])
        self.position_idx: int = self._get_env("POSITION_IDX", V5_HEDGE_MODE_POSITION_IDX, Style.DIM, cast_type=int, allowed_values=[0, 1, 2])
        self.trend_ema_period: int = self._get_env("TREND_EMA_PERIOD", 12, Style.DIM, cast_type=int, min_val=5, max_val=500)
        self.fast_ema_period: int = self._get_env("FAST_EMA_PERIOD", 9, Style.DIM, cast_type=int, min_val=1, max_val=200)
        self.slow_ema_period: int = self._get_env("SLOW_EMA_PERIOD", 21, Style.DIM, cast_type=int, min_val=2, max_val=500)
        self.stoch_period: int = self._get_env("STOCH_PERIOD", 7, Style.DIM, cast_type=int, min_val=1, max_val=100)
        self.stoch_smooth_k: int = self._get_env("STOCH_SMOOTH_K", 3, Style.DIM, cast_type=int, min_val=1, max_val=10)
        self.stoch_smooth_d: int = self._get_env("STOCH_SMOOTH_D", 3, Style.DIM, cast_type=int, min_val=1, max_val=10)
        self.atr_period: int = self._get_env("ATR_PERIOD", 5, Style.DIM, cast_type=int, min_val=1, max_val=100)
        self.adx_period: int = self._get_env("ADX_PERIOD", 14, Style.DIM, cast_type=int, min_val=2, max_val=100)
        self.stoch_oversold_threshold: Decimal = self._get_env("STOCH_OVERSOLD_THRESHOLD", DEFAULT_STOCH_OVERSOLD, Fore.CYAN, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("45"))
        self.stoch_overbought_threshold: Decimal = self._get_env("STOCH_OVERBOUGHT_THRESHOLD", DEFAULT_STOCH_OVERBOUGHT, Fore.CYAN, cast_type=Decimal, min_val=Decimal("55"), max_val=Decimal("100"))
        self.trend_filter_buffer_percent: Decimal = self._get_env("TREND_FILTER_BUFFER_PERCENT", Decimal("0.5"), Fore.CYAN, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("5"))
        self.atr_move_filter_multiplier: Decimal = self._get_env("ATR_MOVE_FILTER_MULTIPLIER", Decimal("0.5"), Fore.CYAN, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("5"))
        self.min_adx_level: Decimal = self._get_env("MIN_ADX_LEVEL", DEFAULT_MIN_ADX, Fore.CYAN, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("90"))
        self.api_key: str = self._get_env("BYBIT_API_KEY", None, Fore.RED, is_secret=True)
        self.api_secret: str = self._get_env("BYBIT_API_SECRET", None, Fore.RED, is_secret=True)
        self.ohlcv_limit: int = self._get_env("OHLCV_LIMIT", DEFAULT_OHLCV_LIMIT, Style.DIM, cast_type=int, min_val=50, max_val=1000)
        self.loop_sleep_seconds: int = self._get_env("LOOP_SLEEP_SECONDS", DEFAULT_LOOP_SLEEP, Style.DIM, cast_type=int, min_val=1)
        self.order_check_delay_seconds: int = self._get_env("ORDER_CHECK_DELAY_SECONDS", 2, Style.DIM, cast_type=int, min_val=1)
        self.order_fill_timeout_seconds: int = self._get_env("ORDER_FILL_TIMEOUT_SECONDS", 20, Style.DIM, cast_type=int, min_val=5)
        self.max_fetch_retries: int = self._get_env("MAX_FETCH_RETRIES", DEFAULT_MAX_RETRIES, Style.DIM, cast_type=int, min_val=0, max_val=10)
        self.retry_delay_seconds: int = self._get_env("RETRY_DELAY_SECONDS", DEFAULT_RETRY_DELAY, Style.DIM, cast_type=int, min_val=1)
        self.trade_only_with_trend: bool = self._get_env("TRADE_ONLY_WITH_TREND", True, Style.DIM, cast_type=bool)
        self.journal_file_path: str = self._get_env("JOURNAL_FILE_PATH", DEFAULT_JOURNAL_FILE, Style.DIM)
        self.enable_journaling: bool = self._get_env("ENABLE_JOURNALING", True, Style.DIM, cast_type=bool)
        self._validate_config()
        logger.debug("Configuration loaded and validated successfully.")

    def _determine_v5_category(self) -> str:
        try:
            category: str
            if self.market_type == "inverse": category = "inverse"
            elif self.market_type in ["linear", "swap"]: category = "linear"
            else: raise ValueError(f"Unsupported MARKET_TYPE '{self.market_type}' for category determination.")
            if ":" not in self.symbol:
                 logger.warning(f"Symbol '{self.symbol}' does not explicitly include settle currency (e.g., :USDT). Explicit format (BASE/QUOTE:SETTLE) recommended for V5.")
            logger.info(f"Determined Bybit V5 API category: '{category}' for symbol '{self.symbol}', market type '{self.market_type}'")
            return category
        except ValueError as e:
            logger.critical(f"{Style.BRIGHT}{Fore.RED}Could not determine V5 category: {e}. Halting.{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)
        return "" # Should be unreachable

    def _validate_config(self):
        if self.fast_ema_period >= self.slow_ema_period:
            logger.critical(f"{Style.BRIGHT}{Fore.RED}Validation failed: FAST_EMA_PERIOD ({self.fast_ema_period}) must be < SLOW_EMA_PERIOD ({self.slow_ema_period}). Halting.{Style.RESET_ALL}"); sys.exit(1)
        if self.trend_ema_period <= self.slow_ema_period:
            logger.warning(f"{Fore.YELLOW}Config Warning: TREND_EMA_PERIOD ({self.trend_ema_period}) <= SLOW_EMA_PERIOD ({self.slow_ema_period}). Trend filter may lag.{Style.RESET_ALL}")
        if self.stoch_oversold_threshold >= self.stoch_overbought_threshold:
            logger.critical(f"{Style.BRIGHT}{Fore.RED}Validation failed: STOCH_OVERSOLD ({self.stoch_oversold_threshold.normalize()}) must be < STOCH_OVERBOUGHT ({self.stoch_overbought_threshold.normalize()}). Halting.{Style.RESET_ALL}"); sys.exit(1)
        if self.tsl_activation_atr_multiplier < self.sl_atr_multiplier:
            logger.warning(f"{Fore.YELLOW}Config Warning: TSL_ACTIVATION_ATR_MULTIPLIER ({self.tsl_activation_atr_multiplier.normalize()}) < SL_ATR_MULTIPLIER ({self.sl_atr_multiplier.normalize()}). TSL may activate early.{Style.RESET_ALL}")
        if self.tp_atr_multiplier > Decimal("0") and self.tp_atr_multiplier <= self.sl_atr_multiplier:
            logger.warning(f"{Fore.YELLOW}Config Warning: TP_ATR_MULTIPLIER ({self.tp_atr_multiplier.normalize()}) <= SL_ATR_MULTIPLIER ({self.sl_atr_multiplier.normalize()}). R:R <= 1.{Style.RESET_ALL}")

    def _cast_value(self, key: str, value_str: str, cast_type: Type, default: Any) -> Any:
        val_to_cast = value_str.strip()
        if not val_to_cast:
            if default is None or (isinstance(default, str) and not default): return default
            logger.warning(f"Empty value for '{key}'. Using default '{default}'."); return default
        try:
            if cast_type == bool: return val_to_cast.lower() in ["true", "1", "yes", "y", "on"]
            elif cast_type == Decimal:
                if val_to_cast.lower() in ["nan", "none", "null"]: return Decimal("NaN")
                return Decimal(val_to_cast)
            elif cast_type == int:
                dec_val = Decimal(val_to_cast)
                if dec_val.to_integral_value(rounding=ROUND_DOWN) != dec_val:
                    raise ValueError(f"Value '{val_to_cast}' with fractional part for int type.")
                return int(dec_val)
            return cast_type(val_to_cast)
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.error(f"{Fore.RED}Cast failed for '{key}' (value: '{value_str}', type: {cast_type.__name__}): {e}. Using default '{default}'.{Style.RESET_ALL}"); return default

    def _validate_value(self, key: str, value: Any, min_val: Optional[Union[int, float, Decimal]], max_val: Optional[Union[int, float, Decimal]], allowed_values: Optional[List[Any]]) -> bool:
        is_num = isinstance(value, (int, float, Decimal))
        if min_val is not None:
            if not is_num: logger.critical(f"{Style.BRIGHT}{Fore.RED}Validation '{key}': Non-numeric '{value}' vs min_val '{min_val}'. Halting.{Style.RESET_ALL}"); sys.exit(1)
            if value < min_val: logger.critical(f"{Style.BRIGHT}{Fore.RED}Validation '{key}': Value '{value}' < min '{min_val}'. Halting.{Style.RESET_ALL}"); sys.exit(1) # type: ignore[operator]
        if max_val is not None:
            if not is_num: logger.critical(f"{Style.BRIGHT}{Fore.RED}Validation '{key}': Non-numeric '{value}' vs max_val '{max_val}'. Halting.{Style.RESET_ALL}"); sys.exit(1)
            if value > max_val: logger.critical(f"{Style.BRIGHT}{Fore.RED}Validation '{key}': Value '{value}' > max '{max_val}'. Halting.{Style.RESET_ALL}"); sys.exit(1) # type: ignore[operator]
        if allowed_values:
            comp_val = str(value).lower() if isinstance(value, str) else value
            lower_allowed = [str(v).lower() if isinstance(v, str) else v for v in allowed_values]
            if comp_val not in lower_allowed:
                logger.error(f"{Fore.RED}Validation '{key}': Invalid value '{value}'. Allowed: {allowed_values}. Using default.{Style.RESET_ALL}"); return False
        return True

    def _get_env(self, key: str, default: Any, color: str, cast_type: Type = str, min_val: Optional[Union[int, float, Decimal]] = None, max_val: Optional[Union[int, float, Decimal]] = None, allowed_values: Optional[List[Any]] = None, is_secret: bool = False) -> Any:
        value_str, source_info, use_default_flag, value_to_process_str = os.getenv(key), "environment variable", False, ""
        if value_str is None or value_str.strip() == "":
            if default is None: logger.critical(f"{Style.BRIGHT}{Fore.RED}Required {'secret ' if is_secret else ''}config '{key}' not found. Halting.{Style.RESET_ALL}"); sys.exit(1)
            use_default_flag, value_to_process_str, source_info = True, str(default), f"default value ('{default if not is_secret else '****'}')"
            log_value_display = default if not is_secret else "****"
        else: value_to_process_str, log_value_display = value_str, "****" if is_secret else value_str
        (logger.warning if use_default_flag and default is not None else logger.info)(f"Using {color}{key}: {log_value_display}{Style.RESET_ALL} (from {source_info})")
        casted_value = self._cast_value(key, value_to_process_str, cast_type, default)
        if not self._validate_value(key, casted_value, min_val, max_val, allowed_values):
            logger.warning(f"{color}Reverting '{key}' to default '{default if not is_secret else '****'}' due to validation failure of value '{casted_value}'.{Style.RESET_ALL}")
            casted_value = default
            if not self._validate_value(key, casted_value, min_val, max_val, allowed_values):
                logger.critical(f"{Style.BRIGHT}{Fore.RED}FATAL: Default value '{default if not is_secret else '****'}' for '{key}' failed validation. Halting.{Style.RESET_ALL}"); sys.exit(1)
        return casted_value

# --- Exchange Manager Class ---
class ExchangeManager:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.exchange: Optional[ccxt.Exchange] = None
        self.market_info: Optional[Dict[str, Any]] = None
        self._initialize_exchange()
        if self.exchange: self.market_info = self._load_market_info()

    def _initialize_exchange(self):
        logger.info(f"Initializing Bybit exchange (V5 API, Market: {self.config.market_type})...")
        try:
            params = {"apiKey": self.config.api_key, "secret": self.config.api_secret, "options": {"defaultType": self.config.market_type, "adjustForTimeDifference": True, "recvWindow": 10000, "brokerId": "PyrmV5NEXUS", "defaultTimeInForce": "GTC"}}
            if os.getenv("USE_BYBIT_TESTNET", "false").lower() == "true":
                logger.warning(f"{Fore.YELLOW}Using Bybit Testnet.{Style.RESET_ALL}"); params['urls'] = {'api': 'https://api-testnet.bybit.com'}
            self.exchange = ccxt.bybit(params)
            self.exchange.fetch_time(); logger.info(f"{Style.BRIGHT}{Fore.GREEN}Bybit V5 interface initialized & tested.{Style.RESET_ALL}")
        except ccxt.AuthenticationError as e: logger.critical(f"{Style.BRIGHT}{Fore.RED}Auth failed: {e}. Check API keys. Halting.{Style.RESET_ALL}", exc_info=False); sys.exit(1)
        except (ccxt.NetworkError, requests.exceptions.RequestException) as e: logger.critical(f"{Style.BRIGHT}{Fore.RED}Network error initializing exchange: {e}. Check internet connection and endpoint. Halting.{Style.RESET_ALL}", exc_info=True); sys.exit(1)
        except Exception as e: logger.critical(f"{Style.BRIGHT}{Fore.RED}Unexpected error initializing exchange: {e}. Halting.{Style.RESET_ALL}", exc_info=True); sys.exit(1)


    def _load_market_info(self) -> Optional[Dict[str, Any]]:
        if not self.exchange: logger.critical("Exchange not initialized. Cannot load market info. Halting."); sys.exit(1)
        try:
            logger.info(f"Loading market info for {self.config.symbol}..."); self.exchange.load_markets(reload=True)
            market = self.exchange.market(self.config.symbol)
            if not market: raise ccxt.ExchangeError(f"Market {self.config.symbol} not found. Check format (e.g., BTC/USDT:USDT).")
            def get_dp(val: Any, default_dp: int) -> int:
                if val is None: return default_dp; dec = safe_decimal(val)
                if dec.is_nan() or dec.is_zero(): return 0 if dec.is_zero() else default_dp
                if dec > 0 and dec < 1: return abs(dec.as_tuple().exponent)
                elif dec >= 1: return 0 if dec.to_integral_value() == dec else (abs(dec.as_tuple().exponent) if dec.as_tuple().exponent < 0 else default_dp)
                return default_dp
            market["precision_dp"] = {"amount": get_dp(market.get("precision", {}).get("amount"), DEFAULT_AMOUNT_DP), "price": get_dp(market.get("precision", {}).get("price"), DEFAULT_PRICE_DP)}
            market["tick_size"] = safe_decimal(market.get("precision", {}).get("price"), Decimal('NaN'))
            market["amount_step"] = safe_decimal(market.get("precision", {}).get("amount"), Decimal('NaN'))
            market["min_order_size"] = safe_decimal(market.get("limits", {}).get("amount", {}).get("min"), Decimal("NaN"))
            market["contract_size"] = safe_decimal(market.get("contractSize"), Decimal("1"))
            if market.get("contractSize") is None: logger.warning(f"Contract size not found for {self.config.symbol}. Defaulting to 1.")
            logger.info(f"Market info {self.config.symbol} (ID: {market.get('id', 'N/A')}): DP(Amt={market['precision_dp']['amount']},Px={market['precision_dp']['price']}), Steps(Tick={market['tick_size'].normalize() if not market['tick_size'].is_nan() else 'N/A'},AmtStep={market['amount_step'].normalize() if not market['amount_step'].is_nan() else 'N/A'}), Limits(MinAmt={market['min_order_size'].normalize() if not market['min_order_size'].is_nan() else 'N/A'}), ContractSz={market['contract_size'].normalize()}, Settle:{market.get('settle', 'N/A')}")
            return market
        except (ccxt.ExchangeError, KeyError, ValueError, TypeError, Exception) as e: logger.critical(f"{Style.BRIGHT}{Fore.RED}Failed to load market info for {self.config.symbol}: {e}. Halting.{Style.RESET_ALL}", exc_info=True); sys.exit(1)
        return None

    def format_price(self, price: Union[Decimal, str, float, int]) -> str:
        dec_val = safe_decimal(price)
        if dec_val.is_nan(): return "NaN"
        dp = self.market_info["precision_dp"]["price"] if self.market_info and "precision_dp" in self.market_info else DEFAULT_PRICE_DP
        try: return f"{dec_val.quantize(Decimal('1e-' + str(dp)), rounding=ROUND_HALF_EVEN):.{dp}f}"
        except: return "ERR"

    def format_amount(self, amount: Union[Decimal, str, float, int], rounding_mode=ROUND_DOWN) -> str:
        dec_val = safe_decimal(amount)
        if dec_val.is_nan(): return "NaN"
        dp = self.market_info["precision_dp"]["amount"] if self.market_info and "precision_dp" in self.market_info else DEFAULT_AMOUNT_DP
        try: return f"{dec_val.quantize(Decimal('1e-' + str(dp)), rounding=rounding_mode):.{dp}f}"
        except: return "ERR"

    def _format_v5_param(self, value: Optional[Union[Decimal, str, float, int]], param_type: Literal["price", "amount", "distance"] = "price", allow_zero: bool = False) -> Optional[str]:
        if value is None: return None
        dec_val = safe_decimal(value, Decimal("NaN"))
        if dec_val.is_nan(): logger.warning(f"V5 Param Format: Input '{value}' NaN."); return None
        if dec_val.is_zero():
            if allow_zero: return "0"
            logger.debug(f"V5 Param Format: Zero value '{value}' not allowed for '{param_type}'."); return None
        if dec_val < 0: logger.warning(f"V5 Param Format: Negative value '{value}' invalid."); return None

        fmt_str: str
        if not self.exchange or not self.config.symbol: # Fallback if CCXT methods can't be used
            logger.warning("V5 Param Format: Exchange/symbol not ready, using custom fallback formatters.")
            fmt_str = self.format_price(dec_val) if param_type in ["price", "distance"] else self.format_amount(dec_val, ROUND_DOWN)
        else:
            try:
                if param_type in ["price", "distance"]: fmt_str = self.exchange.price_to_precision(self.config.symbol, float(dec_val))
                else: fmt_str = self.exchange.amount_to_precision(self.config.symbol, float(dec_val)) # param_type == "amount"
                if safe_decimal(fmt_str).is_nan(): raise ValueError("CCXT formatting resulted in NaN")
            except Exception as e:
                logger.warning(f"CCXT {param_type}_to_precision failed ({e}), fallback to custom format.")
                fmt_str = self.format_price(dec_val) if param_type in ["price", "distance"] else self.format_amount(dec_val, ROUND_DOWN)

        if fmt_str in ["ERR", "NaN"] or safe_decimal(fmt_str).is_nan():
            logger.error(f"V5 Param Format: Failed for '{value}' ({param_type}). Result: {fmt_str}"); return None
        return fmt_str

    def fetch_ohlcv(self) -> Optional[pd.DataFrame]:
        if not self.exchange: logger.error("Exchange not init for fetch_ohlcv."); return None
        logger.debug(f"Fetching {self.config.ohlcv_limit} OHLCV for {self.config.symbol} ({self.config.interval})...")
        try:
            data = fetch_with_retries(self.exchange.fetch_ohlcv, symbol=self.config.symbol, timeframe=self.config.interval, limit=self.config.ohlcv_limit, max_retries=self.config.max_fetch_retries, delay_seconds=self.config.retry_delay_seconds)
            if not data: logger.error(f"fetch_ohlcv for {self.config.symbol} no data."); return None
            if len(data) < 20: logger.warning(f"Fetched only {len(data)} candles.")
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True); df.set_index("timestamp", inplace=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].apply(safe_decimal)
                # if df[col].apply(lambda x: isinstance(x, Decimal) and x.is_nan()).any(): logger.debug(f"OHLCV col '{col}' has NaNs.")
            init_len = len(df)
            for col in ["open", "high", "low", "close"]: df = df[~df[col].apply(lambda x: isinstance(x, Decimal) and x.is_nan())]
            if len(df) < init_len: logger.warning(f"Dropped {init_len - len(df)} OHLCV rows due to NaN in O/H/L/C.")
            if df.empty: logger.error("OHLCV DataFrame empty after processing."); return None
            logger.debug(f"Fetched {len(df)} OHLCV. Last: {df.index[-1] if not df.empty else 'N/A'}")
            return df
        except Exception as e: logger.error(f"Failed to fetch/process OHLCV for {self.config.symbol}: {e}", exc_info=True); return None

    def get_balance(self) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        if not self.exchange or not self.market_info: logger.error("Exch/market info needed for get_balance."); return None, None
        settle_ccy = self.market_info.get("settle")
        if not settle_ccy: logger.error("Settle currency not found for balance."); return None, None
        logger.debug(f"Fetching balance for {settle_ccy} (Acct: {V5_UNIFIED_ACCOUNT_TYPE}, Cat: {self.config.bybit_v5_category})...")
        try:
            bal_data = fetch_with_retries(self.exchange.fetch_balance, params={"accountType": V5_UNIFIED_ACCOUNT_TYPE, "coin": settle_ccy}, max_retries=self.config.max_fetch_retries, delay_seconds=self.config.retry_delay_seconds)
            equity, avail = Decimal("NaN"), Decimal("NaN")
            if "info" in bal_data and "result" in bal_data["info"] and "list" in bal_data["info"]["result"]:
                acc_list = bal_data["info"]["result"]["list"]
                if acc_list and isinstance(acc_list, list):
                    uni_acc = next((i for i in acc_list if i.get("accountType") == V5_UNIFIED_ACCOUNT_TYPE), None)
                    if uni_acc:
                        equity = safe_decimal(uni_acc.get("totalEquity"))
                        coin_list = uni_acc.get("coin", [])
                        if coin_list and isinstance(coin_list, list):
                            settle_coin = next((c for c in coin_list if c.get("coin") == settle_ccy), None)
                            if settle_coin:
                                avail = safe_decimal(settle_coin.get("availableToWithdraw"))
                                if equity.is_nan() and settle_coin.get("equity"): equity = safe_decimal(settle_coin.get("equity"))
                        if avail.is_nan() and uni_acc.get("totalAvailableBalance"):
                            avail = safe_decimal(uni_acc.get("totalAvailableBalance")); logger.debug(f"Used 'totalAvailableBalance' for {settle_ccy}.")
            if equity.is_nan() and bal_data.get("total", {}).get(settle_ccy): equity = safe_decimal(bal_data["total"][settle_ccy]); logger.debug("Used CCXT 'total' for equity.")
            if avail.is_nan() and bal_data.get("free", {}).get(settle_ccy): avail = safe_decimal(bal_data["free"][settle_ccy]); logger.debug("Used CCXT 'free' for available.")
            if equity.is_nan(): logger.error(f"Could not get total equity for {settle_ccy}. Raw: {str(bal_data.get('info',{}).get('result',{}).get('list',[{}])[0])[:200]}"); return None, avail if not avail.is_nan() else Decimal("0")
            if avail.is_nan(): logger.warning(f"Could not get available balance for {settle_ccy}. Defaulting to 0."); avail = Decimal("0")
            logger.debug(f"Balance ({settle_ccy}): Equity={equity.normalize()}, Avail={avail.normalize()}")
            return equity, avail
        except Exception as e: logger.error(f"Failed to fetch/parse balance: {e}", exc_info=True); return None, None

    def get_current_position(self) -> Optional[Dict[str, Dict[str, Any]]]:
        if not self.exchange or not self.market_info: logger.error("Exch/market info needed for get_current_position."); return None
        market_id = self.market_info.get("id"); summary: Dict[str, Dict[str, Any]] = {"long": {}, "short": {}}
        if not market_id: logger.error("Market ID not found for position fetch."); return None
        logger.debug(f"Fetching position for {self.config.symbol} (ID:{market_id}, Cat:{self.config.bybit_v5_category}, Idx:{self.config.position_idx})...")
        try:
            positions_list = fetch_with_retries(self.exchange.fetch_positions, symbols=[self.config.symbol], params={"category": self.config.bybit_v5_category, "symbol": market_id}, max_retries=self.config.max_fetch_retries, delay_seconds=self.config.retry_delay_seconds)
            if not positions_list: logger.debug("No position data. Assuming flat."); return summary
            raw_pos_info = None
            for pos_data in positions_list:
                entry_info = pos_data.get("info", {})
                idx_str = entry_info.get("positionIdx")
                try: idx_int = int(idx_str) if idx_str is not None else -1
                except ValueError: logger.warning(f"Could not parse positionIdx '{idx_str}'. Skipping."); continue
                if idx_int == self.config.position_idx: raw_pos_info = entry_info; logger.debug(f"Found entry for Idx {self.config.position_idx}: {raw_pos_info}"); break
            if not raw_pos_info: logger.debug(f"No entry for Idx {self.config.position_idx}. Assuming flat."); return summary

            qty_abs = safe_decimal(raw_pos_info.get("size", "0")).copy_abs()
            if qty_abs < POSITION_QTY_EPSILON: logger.debug(f"Pos size {qty_abs.normalize()} for Idx {self.config.position_idx} negligible. Flat."); return summary

            api_side = raw_pos_info.get("side", "None").lower()
            pos_key: Optional[str] = None
            if self.config.position_idx == 0: # One-Way
                if api_side == "buy": pos_key = "long"
                elif api_side == "sell": pos_key = "short"
                elif api_side == "none" and qty_abs > 0: logger.warning(f"One-Way (Idx 0) API side 'None' but size {qty_abs.normalize()}. Inconsistent."); return summary
            elif self.config.position_idx == 1: # Hedge Buy
                pos_key = "long";
                if api_side != "buy" and qty_abs > 0: logger.warning(f"Hedge Buy (Idx 1) API side '{api_side}' (not 'Buy') size {qty_abs.normalize()}. Assuming long.")
            elif self.config.position_idx == 2: # Hedge Sell
                pos_key = "short";
                if api_side != "sell" and qty_abs > 0: logger.warning(f"Hedge Sell (Idx 2) API side '{api_side}' (not 'Sell') size {qty_abs.normalize()}. Assuming short.")

            if pos_key:
                entry_px = safe_decimal(raw_pos_info.get("avgPrice", "0"))
                sl_px = safe_decimal(raw_pos_info.get("stopLoss", "0"))
                tp_px = safe_decimal(raw_pos_info.get("takeProfit", "0"))
                tsl_trig_px = safe_decimal(raw_pos_info.get("trailingStop", "0"))
                summary[pos_key] = {
                    "qty": qty_abs, "entry_price": entry_px if not entry_px.is_nan() and entry_px > 0 else Decimal("NaN"),
                    "liq_price": safe_decimal(raw_pos_info.get("liqPrice", "0")), "unrealized_pnl": safe_decimal(raw_pos_info.get("unrealisedPnl", "0")),
                    "api_side": api_side, "info": raw_pos_info,
                    "stop_loss_price": sl_px if not sl_px.is_nan() and sl_px > 0 else None,
                    "take_profit_price": tp_px if not tp_px.is_nan() and tp_px > 0 else None,
                    "is_tsl_active": not tsl_trig_px.is_nan() and tsl_trig_px > 0,
                    "tsl_trigger_price": tsl_trig_px if not tsl_trig_px.is_nan() and tsl_trig_px > 0 else None,
                }
                entry_str = summary[pos_key]['entry_price'].normalize() if summary[pos_key]['entry_price'] and not summary[pos_key]['entry_price'].is_nan() else "N/A"
                logger.debug(f"Identified {pos_key.upper()} pos (Idx {self.config.position_idx}): Qty={qty_abs.normalize()}, Entry={entry_str}")
            else: logger.warning(f"Pos size {qty_abs.normalize()} for Idx {self.config.position_idx} but no long/short map (api_side: '{api_side}'). Flat."); return summary
            return summary
        except Exception as e: logger.error(f"Failed to fetch/parse positions for {self.config.symbol}: {e}", exc_info=True); return None

# --- Indicator Calculator Class ---
class IndicatorCalculator:
    def __init__(self, config: TradingConfig): self.config = config
    def calculate_indicators(self, df: pd.DataFrame) -> Optional[Dict[str, Union[Decimal, bool, int]]]:
        logger.info(f"{Fore.CYAN}# Weaving indicator patterns...{Style.RESET_ALL}")
        if df is None or df.empty: logger.error(f"{Fore.RED}No DataFrame for indicators.{Style.RESET_ALL}"); return None
        req_cols = ["open", "high", "low", "close"]
        if not all(c in df.columns for c in req_cols): logger.error(f"{Fore.RED}DataFrame missing cols: {[c for c in req_cols if c not in df.columns]}{Style.RESET_ALL}"); return None
        try:
            df_calc = df[req_cols].copy()
            def to_float(x: Any) -> float:
                if isinstance(x, (float, int)): return float(x)
                if isinstance(x, Decimal): return float('nan') if x.is_nan() else float(x)
                if isinstance(x, str):
                    try: s = x.strip().lower(); return float('nan') if s in ["nan", "none", "null", ""] else float(s)
                    except ValueError: return float('nan')
                return float('nan') if x is None else float('nan') # Unknown types to NaN
            for col in req_cols: df_calc[col] = df_calc[col].apply(to_float).astype(float)

            init_len = len(df_calc); df_calc.dropna(subset=req_cols, inplace=True, how='any')
            if init_len > len(df_calc): logger.debug(f"Dropped {init_len - len(df_calc)} rows with NaN in OHLC for TA.")
            if df_calc.empty: logger.error(f"{Fore.RED}DataFrame empty after NaN drop for TA.{Style.RESET_ALL}"); return None

            max_p = max(self.config.slow_ema_period, self.config.trend_ema_period, self.config.stoch_period + self.config.stoch_smooth_k + self.config.stoch_smooth_d, self.config.atr_period, self.config.adx_period * 2)
            if len(df_calc) < max_p + 20: logger.error(f"{Fore.RED}Insufficient data ({len(df_calc)} rows) for indicators (needs ~{max_p + 20}).{Style.RESET_ALL}"); return None

            cl, hi, lo = df_calc["close"], df_calc["high"], df_calc["low"]
            f_ema = cl.ewm(span=self.config.fast_ema_period, adjust=False).mean()
            s_ema = cl.ewm(span=self.config.slow_ema_period, adjust=False).mean()
            t_ema = cl.ewm(span=self.config.trend_ema_period, adjust=False).mean()

            l_min_s = lo.rolling(window=self.config.stoch_period, min_periods=max(1, self.config.stoch_period // 2)).min()
            h_max_s = hi.rolling(window=self.config.stoch_period, min_periods=max(1, self.config.stoch_period // 2)).max()
            s_range = h_max_s - l_min_s
            k_raw_v = np.where(s_range > 1e-12, 100 * (cl - l_min_s) / s_range, 50.0)
            k_raw_s = pd.Series(k_raw_v, index=df_calc.index).fillna(50)
            k_s = k_raw_s.rolling(window=self.config.stoch_smooth_k, min_periods=1).mean().fillna(50)
            d_s = k_s.rolling(window=self.config.stoch_smooth_d, min_periods=1).mean().fillna(50)

            tr = pd.concat([hi - lo, (hi - cl.shift(1)).abs(), (lo - cl.shift(1)).abs()], axis=1).max(axis=1).fillna(0)
            atr_s = tr.ewm(span=self.config.atr_period, adjust=False).mean()
            adx_s, pdi_s, mdi_s = self._calculate_adx(hi, lo, cl, atr_s, self.config.adx_period)

            def latest_dec(s: pd.Series, name: str) -> Decimal:
                valid_s = s.dropna()
                if valid_s.empty: logger.warning(f"Indicator '{name}' series empty/all NaN."); return Decimal("NaN")
                return safe_decimal(str(valid_s.iloc[-1]))

            indicators_out: Dict[str, Union[Decimal, bool, int]] = {
                "fast_ema": latest_dec(f_ema, "fast_ema"), "slow_ema": latest_dec(s_ema, "slow_ema"),
                "trend_ema": latest_dec(t_ema, "trend_ema"), "stoch_k": latest_dec(k_s, "stoch_k"),
                "stoch_d": latest_dec(d_s, "stoch_d"), "atr": latest_dec(atr_s, "atr"),
                "atr_period": self.config.atr_period, "adx": latest_dec(adx_s, "adx"),
                "pdi": latest_dec(pdi_s, "pdi"), "mdi": latest_dec(mdi_s, "mdi")}

            k_valid = k_s.dropna(); indicators_out["stoch_k_prev"] = latest_dec(k_valid.iloc[:-1] if len(k_valid) >=2 else pd.Series(dtype=float), "stoch_k_prev")
            d_valid = d_s.dropna(); d_prev = latest_dec(d_valid.iloc[:-1] if len(d_valid) >=2 else pd.Series(dtype=float), "stoch_d_prev")

            k_now, d_now, k_prev = indicators_out["stoch_k"], indicators_out["stoch_d"], indicators_out["stoch_k_prev"]
            indicators_out["stoch_kd_bullish"], indicators_out["stoch_kd_bearish"] = False, False
            if not any(v.is_nan() for v in [k_now, d_now, k_prev, d_prev]): # type: ignore[has-type]
                if (k_prev <= d_prev) and (k_now > d_now): indicators_out["stoch_kd_bullish"] = True # type: ignore[operator]
                if (k_prev >= d_prev) and (k_now < d_now): indicators_out["stoch_kd_bearish"] = True # type: ignore[operator]

            crit_keys = ["fast_ema", "slow_ema", "trend_ema", "atr", "stoch_k", "stoch_d", "adx", "pdi", "mdi"]
            failed = [k for k in crit_keys if indicators_out.get(k, Decimal("NaN")).is_nan()] # type: ignore[union-attr]
            if failed:
                if indicators_out.get("atr", Decimal("NaN")).is_nan(): logger.error(f"{Fore.RED}CRITICAL: ATR is NaN. Risk calc will fail.{Style.RESET_ALL}"); return None # type: ignore[union-attr]
                logger.warning(f"{Fore.YELLOW}Warning: Indicators NaN: {', '.join(failed)}.{Style.RESET_ALL}")
            logger.info(f"{Style.BRIGHT}{Fore.GREEN}Indicator patterns woven.{Style.RESET_ALL}"); return indicators_out
        except Exception as e: logger.error(f"{Fore.RED}Error weaving indicators: {e}{Style.RESET_ALL}", exc_info=True); return None

    def _calculate_adx(self, hi: pd.Series, lo: pd.Series, cl: pd.Series, atr_s: pd.Series, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        if period <= 0: logger.error("ADX period must be > 0."); nan_s = pd.Series(np.nan, index=hi.index); return nan_s, nan_s, nan_s
        if atr_s.empty or atr_s.isnull().all(): logger.error("ATR series empty/NaN for ADX."); nan_s = pd.Series(np.nan, index=hi.index); return nan_s, nan_s, nan_s

        m_up, m_down = hi.diff(), -lo.diff()
        pdm_v = np.where((m_up > m_down) & (m_up > 0), m_up, 0.0)
        mdm_v = np.where((m_down > m_up) & (m_down > 0), m_down, 0.0)
        pdm_s, mdm_s = pd.Series(pdm_v, index=hi.index).fillna(0), pd.Series(mdm_v, index=hi.index).fillna(0)

        alpha = 1.0 / period
        s_pdm = pdm_s.ewm(alpha=alpha, adjust=False, min_periods=period).mean().fillna(0)
        s_mdm = mdm_s.ewm(alpha=alpha, adjust=False, min_periods=period).mean().fillna(0)

        pdi_v = np.where((atr_s > 1e-12) & (~atr_s.isnull()), 100 * s_pdm / atr_s, 0.0)
        mdi_v = np.where((atr_s > 1e-12) & (~atr_s.isnull()), 100 * s_mdm / atr_s, 0.0)
        pdi_s_out, mdi_s_out = pd.Series(pdi_v, index=hi.index).fillna(0), pd.Series(mdi_v, index=hi.index).fillna(0)

        di_sum = pdi_s_out + mdi_s_out
        dx_v = np.where(di_sum > 1e-12, 100 * (pdi_s_out - mdi_s_out).abs() / di_sum, 0.0)
        dx_s = pd.Series(dx_v, index=hi.index).fillna(0)
        adx_s_out = dx_s.ewm(alpha=alpha, adjust=False, min_periods=period).mean().fillna(0)
        return adx_s_out, pdi_s_out, mdi_s_out

# --- Signal Generator Class ---
class SignalGenerator:
    def __init__(self, config: TradingConfig): self.config = config
    def generate_signals(self, df_last_candles: pd.DataFrame, indicators: Dict[str, Union[Decimal, bool, int]]) -> Dict[str, Union[bool, str]]:
        res: Dict[str, Union[bool, str]] = {"long": False, "short": False, "reason": "Initializing signal check"}
        if not indicators: res["reason"] = "No Signal: Indicators missing."; logger.debug(res["reason"]); return res
        if df_last_candles is None or len(df_last_candles) < 2:
            res["reason"] = f"No Signal: Insufficient candles (need >=2, got {len(df_last_candles) if df_last_candles is not None else 0})."
            logger.debug(res["reason"]); return res
        try:
            curr_px = safe_decimal(df_last_candles.iloc[-1]["close"])
            prev_cl = safe_decimal(df_last_candles.iloc[-2]["close"])
            if curr_px.is_nan() or curr_px <= 0: res["reason"] = f"No Signal: Invalid curr_px ({curr_px.normalize() if not curr_px.is_nan() else 'NaN'})."; logger.warning(res["reason"]); return res

            req_keys = ["stoch_k", "fast_ema", "slow_ema", "trend_ema", "atr", "adx", "pdi", "mdi"]
            ind_vals: Dict[str, Decimal] = {}
            nan_keys = [k for k in req_keys if not isinstance(indicators.get(k), Decimal) or indicators.get(k, Decimal("NaN")).is_nan()] # type: ignore[union-attr]
            if nan_keys: res["reason"] = f"No Signal: Indicators NaN/Missing: {', '.join(nan_keys)}."; logger.warning(res["reason"]); return res
            for k in req_keys: ind_vals[k] = indicators[k] # type: ignore[assignment]

            k, f_ema, s_ema, t_ema, atr, adx, pdi, mdi = (ind_vals[key] for key in req_keys)
            sk_bull_x, sk_bear_x = bool(indicators.get("stoch_kd_bullish", False)), bool(indicators.get("stoch_kd_bearish", False))

            ema_bull_x, ema_bear_x = f_ema > s_ema, f_ema < s_ema
            ema_x_state = "Bullish" if ema_bull_x else "Bearish" if ema_bear_x else "Neutral"

            t_buf_abs = t_ema.copy_abs() * (self.config.trend_filter_buffer_percent / 100)
            px_abv_t = curr_px > (t_ema - t_buf_abs); px_blw_t = curr_px < (t_ema + t_buf_abs)
            t_allows_L = px_abv_t if self.config.trade_only_with_trend else True
            t_allows_S = px_blw_t if self.config.trade_only_with_trend else True
            t_reason = f"(P:{curr_px:.{DEFAULT_PRICE_DP}f} vs T_EMA:{t_ema:.{DEFAULT_PRICE_DP}f} ±{t_buf_abs:.{DEFAULT_PRICE_DP}f})" if self.config.trade_only_with_trend else "(TrendFilter OFF)"

            st_L_cond = (k < self.config.stoch_oversold_threshold) or sk_bull_x
            st_S_cond = (k > self.config.stoch_overbought_threshold) or sk_bear_x
            st_reason = f"K:{k:.1f} (OS:{self.config.stoch_oversold_threshold.normalize()}/OB:{self.config.stoch_overbought_threshold.normalize()}) KD_X(B:{sk_bull_x}/Br:{sk_bear_x})"

            sig_move, atr_f_reason = True, "(ATR MoveFilter OFF)"
            if self.config.atr_move_filter_multiplier > 0:
                if atr.is_nan() or atr <= 0: atr_f_reason, sig_move = f"(ATR Filter Skip: Invalid ATR {atr.normalize() if not atr.is_nan() else 'NaN'})", False
                elif prev_cl.is_nan() or prev_cl <= 0: atr_f_reason, sig_move = f"(ATR Filter Skip: Invalid PrevCl {prev_cl.normalize() if not prev_cl.is_nan() else 'NaN'})", False
                else:
                    atr_move_thr = atr * self.config.atr_move_filter_multiplier
                    px_move_abs = (curr_px - prev_cl).copy_abs()
                    sig_move = px_move_abs > atr_move_thr
                    atr_f_reason = f"(Move:{px_move_abs:.{DEFAULT_PRICE_DP}f} {'OK' if sig_move else 'LOW'} vs Thr:{atr_move_thr:.{DEFAULT_PRICE_DP}f})"

            adx_strong = adx > self.config.min_adx_level
            adx_L_fav, adx_S_fav = pdi > mdi, mdi > pdi
            adx_allows_L, adx_allows_S = adx_strong and adx_L_fav, adx_strong and adx_S_fav
            adx_f_reason = f"(ADX:{adx:.1f} {'STR' if adx_strong else 'WEAK'} vs Min:{self.config.min_adx_level.normalize()} | Dir: {'P>M' if adx_L_fav else 'M>P' if adx_S_fav else 'N'})"

            base_L_sig, base_S_sig = ema_bull_x and st_L_cond, ema_bear_x and st_S_cond
            final_L_sig, final_S_sig = base_L_sig and t_allows_L and sig_move and adx_allows_L, base_S_sig and t_allows_S and sig_move and adx_allows_S

            if final_L_sig: res["long"], res["reason"] = True, f"Long: EMA_X {ema_x_state} & StochOK {st_reason} & TrendOK {t_reason} & ATRMoveOK {atr_f_reason} & ADX_OK {adx_f_reason}"
            elif final_S_sig: res["short"], res["reason"] = True, f"Short: EMA_X {ema_x_state} & StochOK {st_reason} & TrendOK {t_reason} & ATRMoveOK {atr_f_reason} & ADX_OK {adx_f_reason}"
            else:
                parts = ["No Signal:"]
                parts.append(f"Base(EMA_X:{ema_x_state},Stoch:{st_reason}) -> LBase:{base_L_sig}, SBase:{base_S_sig}.")
                if base_L_sig or base_S_sig:
                    if not t_allows_L and base_L_sig: parts.append(f"L Blocked: TrendFail {t_reason}.")
                    if not t_allows_S and base_S_sig: parts.append(f"S Blocked: TrendFail {t_reason}.")
                    if not sig_move and (base_L_sig or base_S_sig): parts.append(f"Blocked: ATRMoveFail {atr_f_reason}.")
                    if not adx_allows_L and base_L_sig: parts.append(f"L Blocked: ADXFail {adx_f_reason}.")
                    if not adx_allows_S and base_S_sig: parts.append(f"S Blocked: ADXFail {adx_f_reason}.")
                res["reason"] = " ".join(parts)
            log_lvl = logging.INFO if res["long"] or res["short"] or "Blocked" in res["reason"] else logging.DEBUG
            logger.log(log_lvl, f"Signal Check: {res['reason']}")
        except Exception as e: logger.error(f"{Fore.RED}Error generating entry signals: {e}{Style.RESET_ALL}", exc_info=True); res.update({"reason": f"No Signal: Exception ({type(e).__name__})", "long": False, "short": False})
        return res

    def check_exit_signals(self, position_side: str, indicators: Dict[str, Union[Decimal, bool, int]]) -> Optional[str]:
        if not indicators: logger.warning("No indicators for exit signal check."); return None
        f_ema_v, s_ema_v, k_curr_v, k_prev_v = indicators.get("fast_ema"), indicators.get("slow_ema"), indicators.get("stoch_k"), indicators.get("stoch_k_prev")
        req = {"fast_ema": f_ema_v, "slow_ema": s_ema_v, "stoch_k_current": k_curr_v, "stoch_k_previous": k_prev_v}
        for name, val in req.items():
            if not isinstance(val, Decimal) or val.is_nan(): logger.warning(f"Exit check: Indicator '{name}' invalid ({val})."); return None
        f_ema, s_ema, k_curr, k_prev = f_ema_v, s_ema_v, k_curr_v, k_prev_v # type: ignore[assignment]

        ema_bull, ema_bear = f_ema > s_ema, f_ema < s_ema
        exit_reason: Optional[str] = None
        os_lvl, ob_lvl = self.config.stoch_oversold_threshold, self.config.stoch_overbought_threshold

        if position_side == "long":
            if ema_bear: exit_reason = f"Exit (L): EMA Bearish Cross (F {f_ema.normalize()} < S {s_ema.normalize()})"
            elif k_prev >= ob_lvl and k_curr < ob_lvl: exit_reason = f"Exit (L): Stoch Reversal OB (PrevK {k_prev.normalize():.1f} >= OB {ob_lvl.normalize()} -> CurrK {k_curr.normalize():.1f} < OB)"
            elif k_curr >= ob_lvl: logger.debug(f"Exit Check (L): Stoch K ({k_curr.normalize():.1f}) >= OB ({ob_lvl.normalize()}), awaiting cross down.")
        elif position_side == "short":
            if ema_bull: exit_reason = f"Exit (S): EMA Bullish Cross (F {f_ema.normalize()} > S {s_ema.normalize()})"
            elif k_prev <= os_lvl and k_curr > os_lvl: exit_reason = f"Exit (S): Stoch Reversal OS (PrevK {k_prev.normalize():.1f} <= OS {os_lvl.normalize()} -> CurrK {k_curr.normalize():.1f} > OS)"
            elif k_curr <= os_lvl: logger.debug(f"Exit Check (S): Stoch K ({k_curr.normalize():.1f}) <= OS ({os_lvl.normalize()}), awaiting cross up.")

        if exit_reason: logger.trade(f"{Fore.YELLOW}{exit_reason}{Style.RESET_ALL}")
        return exit_reason

# --- Order Manager Class ---
class OrderManager:
    def __init__(self, config: TradingConfig, exchange_manager: ExchangeManager):
        self.config, self.exchange_manager = config, exchange_manager
        if not exchange_manager or not exchange_manager.exchange or not exchange_manager.market_info:
            err = "OrderManager init: Valid ExchangeManager with exchange & market_info required."
            logger.critical(f"{Style.BRIGHT}{Fore.RED}{err}{Style.RESET_ALL}"); raise ValueError(err)
        self.exchange, self.market_info = exchange_manager.exchange, exchange_manager.market_info
        self.protection_tracker: Dict[str, Optional[str]] = {"long": None, "short": None}

    def _calculate_trade_parameters(self, side: str, atr: Decimal, total_equity: Decimal, current_price: Decimal) -> Optional[Dict[str, Optional[Decimal]]]:
        # Input validation
        if atr.is_nan() or atr <= 0: logger.error(f"Invalid ATR ({atr.normalize() if not atr.is_nan() else 'NaN'}) for trade params."); return None
        if total_equity.is_nan() or total_equity <= 0: logger.error(f"Invalid equity ({total_equity.normalize() if not total_equity.is_nan() else 'NaN'}) for trade params."); return None
        if current_price.is_nan() or current_price <= 0: logger.error(f"Invalid price ({current_price.normalize() if not current_price.is_nan() else 'NaN'}) for trade params."); return None
        if not self.market_info or any(self.market_info.get(k, Decimal('NaN')).is_nan() for k in ['tick_size', 'contract_size', 'min_order_size']):
             logger.error("Market info (tick_size, contract_size, min_order_size) missing/NaN for trade params."); return None
        if side not in ["buy", "sell"]: logger.error(f"Invalid side '{side}' for trade params."); return None

        try:
            risk_amt_settle = total_equity * self.config.risk_percentage
            sl_dist_atr_pts = atr * self.config.sl_atr_multiplier
            sl_px_calc = current_price - sl_dist_atr_pts if side == "buy" else current_price + sl_dist_atr_pts
            if sl_px_calc <= 0: logger.error(f"Calc SL price ({sl_px_calc:.{DEFAULT_PRICE_DP}f}) invalid (<=0)."); return None

            sl_dist_curr_abs = (current_price - sl_px_calc).copy_abs()
            min_tick = self.market_info['tick_size']
            if min_tick.is_nan() or min_tick <= 0: logger.error("Market tick_size invalid."); return None
            if sl_dist_curr_abs < min_tick:
                logger.warning(f"Initial SL dist ({sl_dist_curr_abs.normalize()}) < min tick ({min_tick.normalize()}). Adjusting."); sl_dist_curr_abs = min_tick
                sl_px_calc = current_price - sl_dist_curr_abs if side == "buy" else current_price + sl_dist_curr_abs
                if sl_px_calc <= 0: logger.error(f"Adjusted SL price ({sl_px_calc:.{DEFAULT_PRICE_DP}f}) still invalid (<=0)."); return None
            if sl_dist_curr_abs <= 0: logger.error(f"Calc SL dist ({sl_dist_curr_abs.normalize()}) invalid (<=0)."); return None

            contract_sz = self.market_info['contract_size']; qty_calc_base: Decimal
            if self.config.market_type == "inverse":
                if current_price <= 0: logger.error("Invalid current_price for inverse qty calc."); return None
                risk_amt_quote = risk_amt_settle * current_price; qty_calc_base = risk_amt_quote / sl_dist_curr_abs
            else: # Linear
                val_chg_per_pt_base = contract_sz # Assumes contract_sz is PnL multiplier (1 for USDT perps)
                if val_chg_per_pt_base <= 0: logger.error("Invalid contract size for linear qty."); return None
                risk_per_unit_base = sl_dist_curr_abs * val_chg_per_pt_base
                if risk_per_unit_base <= 0: logger.error(f"Calc zero/neg risk per unit base ({risk_per_unit_base.normalize()})."); return None
                qty_calc_base = risk_amt_settle / risk_per_unit_base

            qty_fmt_str = self.exchange_manager.format_amount(qty_calc_base, ROUND_DOWN)
            qty_final_dec = safe_decimal(qty_fmt_str)
            if qty_final_dec.is_nan() or qty_final_dec <= 0: logger.error(f"Calc qty ({qty_fmt_str}) invalid/zero. Orig: {qty_calc_base.normalize()}"); return None
            min_ord_sz = self.market_info.get('min_order_size', Decimal('NaN'))
            if min_ord_sz.is_nan(): logger.error("Min order size NaN."); return None
            if qty_final_dec < min_ord_sz: logger.error(f"Calc qty {qty_final_dec.normalize()} < min market size {min_ord_sz.normalize()}."); return None

            tp_px_calc: Optional[Decimal] = None
            if self.config.tp_atr_multiplier > 0:
                tp_dist_atr_pts = atr * self.config.tp_atr_multiplier
                tp_px_calc = current_price + tp_dist_atr_pts if side == "buy" else current_price - tp_dist_atr_pts
                if tp_px_calc <= 0: logger.warning(f"Calc TP price ({tp_px_calc:.{DEFAULT_PRICE_DP}f}) invalid (<=0). Disabling TP."); tp_px_calc = None

            tsl_dist_pts = current_price * (self.config.trailing_stop_percent / 100)
            if tsl_dist_pts < min_tick: logger.debug(f"TSL dist ({tsl_dist_pts.normalize()}) < min tick. Adjusting."); tsl_dist_pts = min_tick
            tsl_dist_fmt_str = self.exchange_manager.format_price(tsl_dist_pts)
            tsl_dist_final_dec = safe_decimal(tsl_dist_fmt_str)
            if tsl_dist_final_dec.is_nan() or tsl_dist_final_dec <= 0:
                logger.warning(f"Invalid TSL dist ('{tsl_dist_fmt_str}'). TSL might fail. Orig: {tsl_dist_pts.normalize()}"); tsl_dist_final_dec = Decimal('NaN')

            sl_px_fmt_str = self.exchange_manager.format_price(sl_px_calc)
            sl_px_final_dec = safe_decimal(sl_px_fmt_str)
            if sl_px_final_dec.is_nan() or sl_px_final_dec <= 0: logger.error(f"Formatted SL price ('{sl_px_fmt_str}') invalid."); return None

            tp_px_final_dec: Optional[Decimal] = None
            if tp_px_calc:
                 tp_px_fmt_str = self.exchange_manager.format_price(tp_px_calc)
                 tp_px_final_dec = safe_decimal(tp_px_fmt_str)
                 if tp_px_final_dec.is_nan() or tp_px_final_dec <= 0: logger.warning(f"Failed to format TP price ('{tp_px_fmt_str}'). Disabling TP."); tp_px_final_dec = None

            params: Dict[str, Optional[Decimal]] = {"qty": qty_final_dec, "sl_price": sl_px_final_dec, "tp_price": tp_px_final_dec, "tsl_distance": tsl_dist_final_dec if not tsl_dist_final_dec.is_nan() else None}
            tp_log = f"{params['tp_price'].normalize()}" if params['tp_price'] else "Disabled"
            tsl_log = f"{params['tsl_distance'].normalize()}" if params['tsl_distance'] else "Invalid/Not Set"
            settle_disp = self.market_info.get('settle', self.config.symbol.split(':')[-1] if ':' in self.config.symbol else 'SETTLE')
            logger.info(f"Trade Params ({side.upper()}): Qty={params['qty'].normalize()} {self.market_info.get('base','BASE')}, Entry~={current_price.normalize():.{DEFAULT_PRICE_DP}f}, SL={params['sl_price'].normalize()}, TP={tp_log}, TSLDist~={tsl_log}, RiskAmtSettle={risk_amt_settle.normalize():.{DEFAULT_PRICE_DP}f} {settle_disp}, ATR={atr.normalize():.{DEFAULT_PRICE_DP+1}f}")
            return params
        except (InvalidOperation, DivisionByZero, TypeError, Exception) as e: logger.error(f"Error calc trade params for {side.upper()}: {e}", exc_info=True); return None

    def _execute_market_order(self, side: str, qty_decimal: Decimal) -> Optional[Dict[str, Any]]:
        if not self.exchange or not self.market_info: logger.error("Cannot exec market order: Exch/Market info missing."); return None
        qty_str_api = self.exchange_manager.format_amount(qty_decimal, ROUND_DOWN)
        qty_final_log = safe_decimal(qty_str_api)
        if qty_final_log.is_nan() or qty_final_log <= 0: logger.error(f"Market order with zero/invalid formatted qty: '{qty_str_api}'. Aborted."); return None
        try: amt_float_ccxt = float(qty_str_api)
        except ValueError: logger.error(f"Could not convert qty string '{qty_str_api}' to float for API. Aborted."); return None

        logger.trade(f"{Fore.CYAN}Attempting MARKET {side.upper()} order: {qty_final_log.normalize()} {self.market_info.get('base', '')} for {self.config.symbol}...{Style.RESET_ALL}")
        try:
            params_v5 = {"category": self.config.bybit_v5_category, "positionIdx": self.config.position_idx, "timeInForce": "ImmediateOrCancel"}
            order_resp = fetch_with_retries(self.exchange.create_market_order, symbol=self.config.symbol, side=side, amount=amt_float_ccxt, params=params_v5, max_retries=self.config.max_fetch_retries, delay_seconds=self.config.retry_delay_seconds)
            if order_resp is None: logger.error(f"{Fore.RED}Market order submission failed (returned None).{Style.RESET_ALL}"); return None

            oid, ostatus, filled_s, avg_px_s = order_resp.get("id","[N/A]"), order_resp.get("status","[unk]"), order_resp.get("filled","0"), order_resp.get("average","0")
            filled_d, avg_px_d = safe_decimal(filled_s), safe_decimal(avg_px_s)
            avg_px_log = avg_px_d.normalize() if not avg_px_d.is_nan() and avg_px_d > 0 else "[N/A]"
            logger.trade(f"{Style.BRIGHT}{Fore.GREEN}Market order submitted: ID {oid}, Side {side.upper()}, Ordered {qty_final_log.normalize()}, Status: {ostatus}, Filled: {filled_d.normalize()}, AvgFillPx: {avg_px_log}{Style.RESET_ALL}")
            termux_notify(f"{self.config.symbol} Order Submitted", f"Market {side.upper()} {qty_final_log.normalize()} ID:{oid}, Status:{ostatus}")

            if ostatus == "rejected": logger.error(f"{Fore.RED}Market order {oid} REJECTED. Reason: '{order_resp.get('info', {}).get('rejectReason', 'N/A')}'. Info: {order_resp.get('info')}{Style.RESET_ALL}"); return None
            elif ostatus == "canceled" and filled_d == 0 and params_v5.get("timeInForce") == "ImmediateOrCancel": logger.error(f"{Fore.RED}Market order {oid} (IOC) CANCELED with 0 filled.{Style.RESET_ALL}"); return None
            elif ostatus == "expired": logger.error(f"{Fore.RED}Market order {oid} EXPIRED (unexpected).{Style.RESET_ALL}"); return None

            logger.debug(f"Short delay ({self.config.order_check_delay_seconds}s) after market order {oid} for propagation..."); time.sleep(self.config.order_check_delay_seconds)
            return order_resp
        except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e: logger.error(f"{Fore.RED}Order placement failed ({type(e).__name__}): {e}{Style.RESET_ALL}", exc_info=False); termux_notify(f"{self.config.symbol} Order FAILED", f"Market {side.upper()} failed: {str(e)[:50]}"); return None
        except Exception as e: logger.error(f"{Fore.RED}Unexpected error placing market order: {e}{Style.RESET_ALL}", exc_info=True); termux_notify(f"{self.config.symbol} Order ERROR", f"Market {side.upper()} error."); return None

    def _set_position_protection(self, position_side: str, sl_price: Optional[Decimal] = None, tp_price: Optional[Decimal] = None, is_tsl: bool = False, tsl_distance: Optional[Decimal] = None, tsl_activation_price: Optional[Decimal] = None) -> bool:
        if not self.exchange or not self.market_info: logger.error("Cannot set protection: Exch/Market info missing."); return False
        market_id = self.market_info.get("id"); tracker_key = position_side.lower()
        if not market_id: logger.error("Cannot set protection: Market ID missing."); return False
        if tracker_key not in self.protection_tracker: logger.error(f"Invalid side '{position_side}' for protection tracker."); return False

        sl_str = self.exchange_manager._format_v5_param(sl_price, "price", True) if sl_price else "0"
        tp_str = self.exchange_manager._format_v5_param(tp_price, "price", True) if tp_price else "0"
        tsl_dist_str = self.exchange_manager._format_v5_param(tsl_distance, "distance", False) if tsl_distance else "0"
        tsl_act_px_str = self.exchange_manager._format_v5_param(tsl_activation_price, "price", False) if tsl_activation_price else "0"

        api_params: Dict[str, Any] = {"category": self.config.bybit_v5_category, "symbol": market_id, "positionIdx": self.config.position_idx, "tpslMode": V5_TPSL_MODE_FULL}
        action_desc, new_tracker_state = "", None

        if is_tsl:
            if tsl_dist_str and tsl_dist_str != "0" and tsl_act_px_str and tsl_act_px_str != "0":
                api_params.update({"trailingStop": tsl_dist_str, "activePrice": tsl_act_px_str, "triggerBy": self.config.tsl_trigger_by, "stopLoss": "0", "takeProfit": "0"})
                action_desc, new_tracker_state = f"ACTIVATE/MOD TSL (Dist:{tsl_dist_str}, ActPx:{tsl_act_px_str})", "ACTIVE_TSL"
            else: logger.error(f"Cannot activate TSL for {position_side.upper()}: Invalid TSL dist ('{tsl_dist_str}') or act px ('{tsl_act_px_str}')."); return False
        elif sl_str != "0" or tp_str != "0":
            if sl_str != "0": api_params["stopLoss"] = sl_str
            if tp_str != "0": api_params["takeProfit"] = tp_str
            api_params.update({"slTriggerBy": self.config.sl_trigger_by, "tpTriggerBy": self.config.sl_trigger_by, "trailingStop": "0", "activePrice": "0"}) # tpTriggerBy often same as sl
            action_desc, new_tracker_state = f"SET SL={api_params.get('stopLoss','0')} TP={api_params.get('takeProfit','0')}", "ACTIVE_SLTP"
        else: # Clearing all
            api_params.update({"stopLoss": "0", "takeProfit": "0", "trailingStop": "0", "activePrice": "0"})
            action_desc, new_tracker_state = "CLEAR ALL SL/TP/TSL", None

        logger.trade(f"{Fore.CYAN}Attempting to {action_desc} for {position_side.upper()} {self.config.symbol}...{Style.RESET_ALL}"); logger.debug(f"V5 setTradingStop params: {api_params}")
        # Corrected private method name based on user's note
        private_method_name = "private_post_position_set_trading_stop"
        if not hasattr(self.exchange, private_method_name):
            # Fallback to common CCXT camelCase if snake_case not found
            private_method_name_fallback = "privatePostPositionSetTradingStop"
            if hasattr(self.exchange, private_method_name_fallback):
                logger.warning(f"'{private_method_name}' not found, using fallback '{private_method_name_fallback}'.")
                private_method_name = private_method_name_fallback
            else:
                logger.critical(f"{Style.BRIGHT}{Fore.RED}CCXT private method for setting trading stops ('{private_method_name}' or '{private_method_name_fallback}') not found. Cannot manage protection.{Style.RESET_ALL}")
                return False
        try:
            resp = fetch_with_retries(getattr(self.exchange, private_method_name), params=api_params, max_retries=self.config.max_fetch_retries, delay_seconds=self.config.retry_delay_seconds)
            if resp and resp.get("retCode") == V5_SUCCESS_RETCODE:
                logger.trade(f"{Style.BRIGHT}{Fore.GREEN}{action_desc} successful for {position_side.upper()}.{Style.RESET_ALL}")
                termux_notify(f"{self.config.symbol} Protection {('Set' if new_tracker_state else 'Cleared')}", f"{action_desc} for {position_side.upper()}")
                self.protection_tracker[tracker_key] = new_tracker_state; return True
            else:
                ret_code, ret_msg = (resp.get("retCode","N/A"), resp.get("retMsg","N/A")) if resp else ("N/A","N/A")
                logger.error(f"{Fore.RED}{action_desc} failed for {position_side.upper()}. API: Code={ret_code}, Msg='{ret_msg}'.{Style.RESET_ALL}"); logger.debug(f"Full response: {resp}")
                termux_notify(f"{self.config.symbol} Protection FAILED", f"{action_desc[:30]}... failed: {ret_msg[:50]}"); return False
        except Exception as e: logger.error(f"{Fore.RED}Unexpected error during '{action_desc}' for {position_side.upper()}: {e}{Style.RESET_ALL}", exc_info=True); termux_notify(f"{self.config.symbol} Protection ERROR", f"{action_desc[:30]}... error."); return False

    def _verify_position_state(self, expected_side_logical: Optional[str], expected_qty_min_abs: Decimal = POSITION_QTY_EPSILON, max_attempts: int = 4, delay_seconds: float = 1.5, action_context: str = "Position Verification") -> Tuple[bool, Optional[Dict[str, Dict[str, Any]]]]:
        logger.debug(f"{action_context}: Verifying. Expect side: '{expected_side_logical}', MinAbsQty: {expected_qty_min_abs.normalize()}. Max attempts: {max_attempts}.")
        last_pos_summary: Optional[Dict[str, Dict[str, Any]]] = None
        for attempt in range(max_attempts):
            logger.debug(f"{action_context}: Attempt {attempt + 1}/{max_attempts}...")
            curr_pos_summary = self.exchange_manager.get_current_position(); last_pos_summary = curr_pos_summary
            if curr_pos_summary is None:
                logger.warning(f"{action_context} Warn: Failed to fetch position state on attempt {attempt + 1}.")
                if attempt < max_attempts - 1: time.sleep(delay_seconds); continue
                logger.error(f"{Fore.RED}{action_context} FAILED: Could not fetch position state after {max_attempts} attempts.{Style.RESET_ALL}"); return False, last_pos_summary

            actual_flat, actual_side, actual_qty = True, None, Decimal("0")
            long_pos, short_pos = curr_pos_summary.get("long",{}), curr_pos_summary.get("short",{})
            if long_pos and safe_decimal(long_pos.get("qty","0")).copy_abs() >= POSITION_QTY_EPSILON: actual_flat, actual_side, actual_qty = False, "long", safe_decimal(long_pos.get("qty","0")).copy_abs()
            elif short_pos and safe_decimal(short_pos.get("qty","0")).copy_abs() >= POSITION_QTY_EPSILON: actual_flat, actual_side, actual_qty = False, "short", safe_decimal(short_pos.get("qty","0")).copy_abs()

            verified, log_suffix = False, ""
            if expected_side_logical is None: # Expecting flat
                verified = actual_flat
                log_suffix = f"Expected FLAT, Actual: {'FLAT' if actual_flat else f'{str(actual_side).upper()} Qty={actual_qty.normalize()}'}"
            elif actual_side == expected_side_logical: # Side matches
                qty_ok = actual_qty >= expected_qty_min_abs; verified = qty_ok
                log_suffix = f"Expected {expected_side_logical.upper()} (MinQty~{expected_qty_min_abs.normalize()}), Actual: {actual_side.upper()} Qty={actual_qty.normalize()} ({'QTY OK' if qty_ok else 'QTY MISMATCH'})"
            else: # Side mismatch
                log_suffix = f"Expected {str(expected_side_logical).upper() if expected_side_logical else 'FLAT'}, Actual: {'FLAT' if actual_flat else (str(actual_side).upper() + ' Qty=' + actual_qty.normalize()) if actual_side else 'UNKNOWN'} (SIDE MISMATCH)"

            logger.debug(f"{action_context} Check {attempt + 1}: {log_suffix}")
            if verified: logger.info(f"{Style.BRIGHT}{Fore.GREEN}{action_context} SUCCEEDED (Attempt {attempt+1}). State: {log_suffix}{Style.RESET_ALL}"); return True, curr_pos_summary
            if attempt < max_attempts - 1: logger.debug(f"State not as expected. Waiting {delay_seconds}s..."); time.sleep(delay_seconds)
            else: logger.error(f"{Fore.RED}{action_context} FAILED after {max_attempts} attempts. Final: {log_suffix}{Style.RESET_ALL}"); return False, curr_pos_summary
        return False, last_pos_summary # Should be covered by loop logic

    def place_risked_market_order(self, side: str, atr: Decimal, total_equity: Decimal, current_price: Decimal) -> bool:
        if not self.exchange or not self.market_info: logger.critical("OrderManager not init for place_risked_market_order."); return False
        if side not in ["buy", "sell"]: logger.error(f"Invalid side '{side}'."); return False
        if atr.is_nan() or atr <= 0: logger.error("Entry Aborted: Invalid ATR."); return False
        if total_equity is None or total_equity.is_nan() or total_equity <= 0: logger.error("Entry Aborted: Invalid Equity."); return False
        if current_price.is_nan() or current_price <= 0: logger.error("Entry Aborted: Invalid Current Price."); return False

        log_pos_side = "long" if side == "buy" else "short"
        logger.info(f"{Style.BRIGHT}{Fore.MAGENTA}--- Initiating Entry: {log_pos_side.upper()} ---{Style.RESET_ALL}")

        params = self._calculate_trade_parameters(side, atr, total_equity, current_price)
        if not params or not params.get("qty") or params["qty"] <= 0: logger.error("Entry Aborted: Failed to calc valid trade params."); return False
        qty_ord, sl_px, tp_px = params["qty"], params.get("sl_price"), params.get("tp_price") # qty_ord is Decimal
        if sl_px is None or sl_px.is_nan() or sl_px <= 0: logger.error(f"Entry Aborted: Invalid SL price ({sl_px})."); return False

        order_info = self._execute_market_order(side, qty_ord) # type: ignore[arg-type]
        if not order_info: logger.error(f"Entry Aborted: Market order failed for {side.upper()} {qty_ord.normalize()}."); self._handle_entry_failure(side, qty_ord); return False # type: ignore[arg-type]
        order_id = order_info.get("id", "[N/A_ID]"); avg_entry_px_order = safe_decimal(order_info.get("average", "NaN"))

        min_fill_qty = qty_ord * Decimal("0.90") # type: ignore[operator]
        verified_ok, final_pos_state = self._verify_position_state(expected_side_logical=log_pos_side, expected_qty_min_abs=min_fill_qty, max_attempts=6, delay_seconds=max(self.config.order_check_delay_seconds, 1.0), action_context=f"Post-{log_pos_side.upper()}-Entry Verify")
        if not verified_ok: logger.error(f"{Fore.RED}Entry FAILED: Position verify failed after order {order_id}. Manual check! Cleanup attempt...{Style.RESET_ALL}"); self._handle_entry_failure(side, qty_ord); return False # type: ignore[arg-type]

        active_pos = final_pos_state.get(log_pos_side) if final_pos_state else {}
        if not active_pos: logger.error(f"{Fore.RED}Internal Error: Pos {log_pos_side} verified OK, but details missing. Aborting.{Style.RESET_ALL}"); self._handle_entry_failure(side, qty_ord); return False # type: ignore[arg-type]

        fill_qty_actual = safe_decimal(active_pos.get("qty", "0")).copy_abs()
        avg_entry_px_actual = safe_decimal(active_pos.get("entry_price", "NaN"))
        if avg_entry_px_actual.is_nan() and not avg_entry_px_order.is_nan(): avg_entry_px_actual = avg_entry_px_order; logger.debug(f"Used avg entry px from order resp ({avg_entry_px_order.normalize()}).")
        logger.info(f"{Style.BRIGHT}{Fore.GREEN}Position {log_pos_side.upper()} confirmed: ActualQty={fill_qty_actual.normalize()}, AvgEntryPx={avg_entry_px_actual.normalize() if not avg_entry_px_actual.is_nan() else '[N/A]'}{Style.RESET_ALL}")
        if fill_qty_actual < qty_ord * Decimal("0.99"): logger.warning(f"Filled qty {fill_qty_actual.normalize()} < ordered {qty_ord.normalize()}. Slippage/partial fill?.") # type: ignore[operator]

        stops_ok = self._set_position_protection(log_pos_side, sl_price=sl_px, tp_price=tp_px)
        if not stops_ok: logger.error(f"{Fore.RED}Entry Alert: Failed to set initial SL/TP for {log_pos_side.upper()}. Emergency close!{Style.RESET_ALL}"); self.close_position(log_pos_side, fill_qty_actual, reason="EmergencyClose:FailedInitialStopSet"); return False

        if self.config.enable_journaling: self.log_trade_entry_to_journal(side, fill_qty_actual, avg_entry_px_actual, order_id)
        logger.info(f"{Style.BRIGHT}{Fore.GREEN}--- Entry Sequence for {log_pos_side.upper()} Completed Successfully ---{Style.RESET_ALL}"); return True

    def manage_trailing_stop(self, position_side: str, entry_price: Decimal, current_market_price: Decimal, current_atr: Decimal):
        if not self.exchange or not self.market_info: logger.error("Cannot manage TSL: Exch/Market info missing."); return
        tracker_key = position_side.lower()
        if self.protection_tracker.get(tracker_key) != "ACTIVE_SLTP":
            logger.debug(f"TSL Mgmt ({position_side.upper()}): Not ACTIVE_SLTP (Tracker: {self.protection_tracker.get(tracker_key)}). Skipping."); return
        if any(val.is_nan() or val <= 0 for val in [current_atr, entry_price, current_market_price]):
            logger.debug(f"TSL Check ({position_side.upper()}): Invalid ATR/entry_px/market_px. Skipping."); return
        try:
            act_dist_pts = current_atr * self.config.tsl_activation_atr_multiplier
            tsl_act_target_px = entry_price + act_dist_pts if position_side == "long" else entry_price - act_dist_pts
            if tsl_act_target_px.is_nan() or tsl_act_target_px <= 0: logger.warning(f"Invalid TSL act px ({tsl_act_target_px.normalize()}). Skipping TSL."); return

            tsl_trail_dist_pts = current_market_price * (self.config.trailing_stop_percent / 100)
            min_tick = self.market_info.get('tick_size', Decimal('1e-8'))
            if not min_tick.is_nan() and min_tick > 0 and tsl_trail_dist_pts < min_tick:
                logger.debug(f"TSL trail dist ({tsl_trail_dist_pts.normalize()}) < min tick. Adjusting."); tsl_trail_dist_pts = min_tick
            if tsl_trail_dist_pts <= 0: logger.warning(f"Invalid TSL trail dist ({tsl_trail_dist_pts.normalize()}). Skipping TSL."); return

            activate = (position_side == "long" and current_market_price >= tsl_act_target_px) or \
                       (position_side == "short" and current_market_price <= tsl_act_target_px)
            if activate:
                logger.trade(f"{Fore.MAGENTA}TSL activation MET for {position_side.upper()}! (Entry:{entry_price.normalize():.{DEFAULT_PRICE_DP}f}, CurrPx:{current_market_price.normalize():.{DEFAULT_PRICE_DP}f}, TSLActTargetPx~:{tsl_act_target_px.normalize():.{DEFAULT_PRICE_DP}f}, TSLDistSet~:{tsl_trail_dist_pts.normalize():.{DEFAULT_PRICE_DP}f}){Style.RESET_ALL}")
                if self._set_position_protection(position_side, is_tsl=True, tsl_distance=tsl_trail_dist_pts, tsl_activation_price=tsl_act_target_px):
                    logger.trade(f"{Style.BRIGHT}{Fore.GREEN}TSL activated successfully for {position_side.upper()}.{Style.RESET_ALL}")
                else: logger.error(f"{Fore.RED}Failed to activate TSL for {position_side.upper()} via API.{Style.RESET_ALL}")
            else: logger.debug(f"TSL Check ({position_side.upper()}): Activation NOT MET. (CurrPx:{current_market_price.normalize():.{DEFAULT_PRICE_DP}f}, TargetActPx:~{tsl_act_target_px.normalize():.{DEFAULT_PRICE_DP}f})")
        except Exception as e: logger.error(f"Error managing TSL for {position_side.upper()}: {e}", exc_info=True)

    def close_position(self, position_side_to_close: str, qty_abs_to_close: Decimal, reason: str = "Strategy Exit Signal") -> bool:
        if not self.exchange or not self.market_info: logger.critical("OrderManager not init for close_position."); return False
        if position_side_to_close not in ["long", "short"]: logger.error(f"Invalid side '{position_side_to_close}'."); return False
        if qty_abs_to_close.is_nan() or qty_abs_to_close.copy_abs() < POSITION_QTY_EPSILON:
            logger.warning(f"Close for zero/negligible qty ({qty_abs_to_close.normalize()}). Skipping for {position_side_to_close.upper()}."); self.protection_tracker[position_side_to_close.lower()] = None; return True

        closing_order_side = "sell" if position_side_to_close == "long" else "buy"
        logger.trade(f"{Fore.YELLOW}Attempting to CLOSE {position_side_to_close.upper()} (Qty: {qty_abs_to_close.normalize()} {self.market_info.get('base','')}) for {self.config.symbol} | Reason: {reason}...{Style.RESET_ALL}")

        logger.debug(f"Clearing protection for {position_side_to_close.upper()} before closing...")
        if self._set_position_protection(position_side_to_close, sl_price=None, tp_price=None, is_tsl=False):
            logger.info(f"Protection cleared for {position_side_to_close.upper()}."); self.protection_tracker[position_side_to_close.lower()] = None
        else: logger.warning(f"{Fore.YELLOW}Failed to confirm protection clear for {position_side_to_close.upper()}. Proceeding cautiously...{Style.RESET_ALL}")

        close_order_info = self._execute_market_order(closing_order_side, qty_abs_to_close)
        if not close_order_info:
            logger.error(f"{Fore.RED}Failed to submit closing market order for {position_side_to_close.upper()}. MANUAL INTERVENTION!{Style.RESET_ALL}"); termux_notify(f"{self.config.symbol} CLOSE ORDER FAILED", f"Market {closing_order_side.upper()} failed!"); return False

        close_oid = close_order_info.get("id","[N/A_CLOSE_ID]"); avg_close_px = safe_decimal(close_order_info.get("average"), Decimal("NaN"))
        logger.trade(f"{Fore.YELLOW}Closing market order ({close_oid}) submitted for {position_side_to_close.upper()}. Reported AvgClosePx: {avg_close_px.normalize() if not avg_close_px.is_nan() else '[N/A]'}{Style.RESET_ALL}")
        termux_notify(f"{self.config.symbol} Pos Closing", f"{position_side_to_close.upper()} close order {close_oid} submitted.")

        verified_flat, _ = self._verify_position_state(expected_side_logical=None, max_attempts=6, delay_seconds=max(self.config.order_check_delay_seconds + 0.5, 1.5), action_context=f"Post-{position_side_to_close.upper()}-Close Verify")
        if self.config.enable_journaling: self.log_trade_exit_to_journal(position_side_to_close, qty_abs_to_close, avg_close_px, close_oid, reason)

        if not verified_flat:
            logger.error(f"{Fore.RED}Position {position_side_to_close.upper()} closure verification FAILED. May still be open. MANUAL INTERVENTION!{Style.RESET_ALL}"); termux_notify(f"{self.config.symbol} CLOSE VERIFY FAILED", f"{position_side_to_close.upper()} pos may be open!"); return False

        logger.trade(f"{Style.BRIGHT}{Fore.GREEN}Position {position_side_to_close.upper()} confirmed closed (flat).{Style.RESET_ALL}"); self.protection_tracker[position_side_to_close.lower()] = None; return True

    def _handle_entry_failure(self, failed_entry_order_side: str, attempted_qty_abs: Decimal):
        logger.warning(f"{Fore.YELLOW}Handling entry failure for {failed_entry_order_side.upper()} (intended qty: {attempted_qty_abs.normalize()}). Checking for lingering position...{Style.RESET_ALL}")
        log_pos_side_check = "long" if failed_entry_order_side == "buy" else "short"
        time.sleep(max(self.config.order_check_delay_seconds, 1.0) + 1.0)

        _, curr_pos_summary = self._verify_position_state(expected_side_logical=None, max_attempts=2, delay_seconds=1.0, action_context=f"Entry-Fail-Cleanup-Check-{log_pos_side_check.upper()}")
        if curr_pos_summary is None:
            logger.error(f"{Fore.RED}Could not fetch positions during entry failure handling for {log_pos_side_check.upper()}. MANUAL CHECK URGENT!{Style.RESET_ALL}"); termux_notify(f"{self.config.symbol} URGENT CHECK", "Failed pos state fetch in entry fail cleanup!"); return

        lingering_pos = curr_pos_summary.get(log_pos_side_check, {}); lingering_qty = safe_decimal(lingering_pos.get("qty","0")).copy_abs()
        if lingering_qty >= POSITION_QTY_EPSILON:
            logger.error(f"{Fore.RED}Lingering {log_pos_side_check.upper()} position (Qty: {lingering_qty.normalize()}) found after failed entry. Emergency close...{Style.RESET_ALL}"); termux_notify(f"{self.config.symbol} Emergency Close", f"Lingering {log_pos_side_check.upper()} pos.")
            if self.close_position(log_pos_side_check, lingering_qty, reason="EmergencyClose:LingeringAfterEntryFail"): logger.info(f"Emergency close for lingering {log_pos_side_check.upper()} submitted/confirmed.")
            else: logger.critical(f"{Style.BRIGHT}{Fore.RED}EMERGENCY CLOSE FAILED for lingering {log_pos_side_check.upper()}. MANUAL INTERVENTION URGENT!{Style.RESET_ALL}"); termux_notify(f"{self.config.symbol} URGENT CHECK", f"Emergency close of lingering {log_pos_side_check.upper()} FAILED!")
        else: logger.info(f"No significant lingering {log_pos_side_check.upper()} position. Qty: {lingering_qty.normalize()}."); self.protection_tracker[log_pos_side_check] = None

    def _write_journal_row(self, trade_data: Dict[str, Any]):
        if not self.config.enable_journaling: return
        journal_file = Path(self.config.journal_file_path)
        file_exists_has_content = journal_file.is_file() and journal_file.stat().st_size > 0
        try:
            journal_file.parent.mkdir(parents=True, exist_ok=True)
            with journal_file.open("a", newline="", encoding="utf-8") as csvfile:
                fields = ["TimestampUTC", "Symbol", "Action", "Side", "Quantity", "AvgPrice", "OrderID", "Reason", "Notes"]
                writer = csv.DictWriter(csvfile, fieldnames=fields, quoting=csv.QUOTE_MINIMAL)
                if not file_exists_has_content: writer.writeheader()
                row = {f: ('NaN' if isinstance(trade_data.get(f), Decimal) and trade_data.get(f).is_nan() else (f"{trade_data.get(f).normalize()}" if isinstance(trade_data.get(f), Decimal) else str(trade_data.get(f, 'N/A')))) for f in fields}
                row['Notes'] = str(trade_data.get('Notes', '')) # Ensure Notes is string
                writer.writerow(row)
            logger.debug(f"Trade action '{trade_data.get('Action', 'Unk')}' logged to journal: {journal_file}")
        except IOError as e: logger.error(f"I/O error writing to journal '{journal_file}': {e}")
        except Exception as e: logger.error(f"Unexpected error writing to journal: {e}", exc_info=True)

    def log_trade_entry_to_journal(self, order_side: str, filled_qty_abs: Decimal, avg_fill_price: Decimal, order_id: Optional[str]):
        self._write_journal_row({"TimestampUTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), "Symbol": self.config.symbol, "Action": "ENTRY", "Side": ("long" if order_side == "buy" else "short").upper(), "Quantity": filled_qty_abs, "AvgPrice": avg_fill_price, "OrderID": order_id, "Reason": "Strategy Entry Signal"})

    def log_trade_exit_to_journal(self, position_side_closed: str, closed_qty_abs: Decimal, avg_close_price: Decimal, order_id: Optional[str], exit_reason: str):
        self._write_journal_row({"TimestampUTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), "Symbol": self.config.symbol, "Action": "EXIT", "Side": position_side_closed.upper(), "Quantity": closed_qty_abs, "AvgPrice": avg_close_price, "OrderID": order_id, "Reason": exit_reason})

# --- Status Display Class ---
class StatusDisplay:
    def __init__(self, config: TradingConfig):
        self.config = config
        self._default_price_dp = DEFAULT_PRICE_DP
        self._default_amount_dp = DEFAULT_AMOUNT_DP

    def _format_decimal_for_rich(self, value: Optional[Decimal], precision: Optional[int] = None, default_precision_fallback: int = 2, add_commas: bool = False, highlight_negative: bool = False, default_style: str = "white", style_override: Optional[str] = None) -> Text:
        if value is None or (isinstance(value, Decimal) and value.is_nan()): return Text("N/A", style="dim")
        dp = precision if precision is not None else default_precision_fallback
        try:
            fmt_val = value.quantize(Decimal(f"1e-{dp}"), rounding=ROUND_HALF_EVEN)
            disp_str = f"{{:{',' if add_commas else ''}.{dp}f}}".format(fmt_val)
            style = style_override if style_override else default_style
            if highlight_negative and not style_override:
                if fmt_val < 0: style = "bright_red"
                elif fmt_val > 0: style = "bright_green"
            return Text(disp_str, style=style)
        except: return Text("ERR", style="bold bright_red")

    def print_status_panel(self, cycle_num: int, current_timestamp: Optional[datetime], current_market_price: Optional[Decimal], indicators_data: Optional[Dict[str, Any]], current_positions_summary: Optional[Dict[str, Dict[str, Any]]], account_equity: Optional[Decimal], signal_check_result_or_status: Dict[str, Any], protection_status_tracker: Dict[str, Optional[str]], market_specific_info: Optional[Dict[str, Any]]):
        price_dp = self._default_price_dp
        amount_display_dp = self._default_amount_dp # Corrected: Use internal default
        if market_specific_info and "precision_dp" in market_specific_info:
             price_dp = market_specific_info["precision_dp"].get("price", self._default_price_dp)
             amount_display_dp = market_specific_info["precision_dp"].get("amount", self._default_amount_dp)

        panel = Text()
        ts_str = current_timestamp.strftime("%Y-%m-%d %H:%M:%S %Z") if current_timestamp else Text("Timestamp N/A", style="dim").plain
        title = f" Cycle {cycle_num} | {self.config.symbol} ({self.config.interval}) | {ts_str} "
        settle_ccy = market_specific_info.get("settle", "SETTLE") if market_specific_info else "SETTLE"
        panel.append("Price: ", style="bold bright_cyan"); panel.append(self._format_decimal_for_rich(current_market_price, price_dp, style_override="bright_white"))
        panel.append(" | Equity: ", style="bold bright_yellow"); panel.append(self._format_decimal_for_rich(account_equity, 2, add_commas=True, style_override="bright_yellow")); panel.append(f" {settle_ccy}\n", style="bright_yellow"); panel.append("---\n", style="dim")

        panel.append("Indicators: ", style="bold bright_cyan")
        if indicators_data:
            parts = []
            def fmt_ind(k: str, p: int = 1, s: str = "white") -> Text:
                 v = indicators_data.get(k)
                 if isinstance(v, bool): return Text(str(v), style=s)
                 d_v = v if isinstance(v, Decimal) else safe_decimal(str(v) if v is not None else "NaN")
                 return self._format_decimal_for_rich(d_v, p, default_style=s)
            parts.append(Text("EMA(F/S/T): ").append(fmt_ind('fast_ema', price_dp, "cyan")).append("/").append(fmt_ind('slow_ema', price_dp, "magenta")).append("/").append(fmt_ind('trend_ema', price_dp, "yellow")))
            st_txt = Text("Stoch(K/D/PrevK): ").append(fmt_ind('stoch_k',1,"bright_blue")).append("/").append(fmt_ind('stoch_d',1,"blue")).append("/").append(fmt_ind('stoch_k_prev',1,"dim blue"))
            if indicators_data.get('stoch_kd_bullish'): st_txt.append(" [b green]▲BullX[/]", style="green")
            elif indicators_data.get('stoch_kd_bearish'): st_txt.append(" [b red]▼BearX[/]", style="red")
            parts.append(st_txt)
            parts.append(Text(f"ATR({indicators_data.get('atr_period',self.config.atr_period)}): ").append(fmt_ind('atr', price_dp+1, "bright_magenta"))) # type: ignore[operator]
            adx_v = indicators_data.get('adx'); adx_d = adx_v if isinstance(adx_v, Decimal) else safe_decimal(str(adx_v) if adx_v is not None else "NaN")
            adx_s = "yellow" if not adx_d.is_nan() and adx_d > self.config.min_adx_level else "dim yellow"
            parts.append(Text(f"ADX({self.config.adx_period}): ").append(self._format_decimal_for_rich(adx_d,1,default_style=adx_s)).append(" [+DI:",style="dim").append(fmt_ind('pdi',1,"bright_green")).append(" -DI:",style="dim").append(fmt_ind('mdi',1,"bright_red")).append("]",style="dim"))
            panel.append(Text(" | ",style="dim").join(parts)); panel.append("\n")
        else: panel.append(Text("Calculating or data unavailable...", style="dim")); panel.append("\n")
        panel.append("---\n", style="dim")

        panel.append("Position: ", style="bold bright_cyan"); pos_disp = Text("FLAT", style="bold bright_green")
        active_pos_side, active_pos_data = None, None
        if current_positions_summary:
            long_d, short_d = current_positions_summary.get("long",{}), current_positions_summary.get("short",{})
            if long_d and long_d.get("qty") and safe_decimal(long_d["qty"]).copy_abs() >= POSITION_QTY_EPSILON: active_pos_side, active_pos_data = "long", long_d
            elif short_d and short_d.get("qty") and safe_decimal(short_d["qty"]).copy_abs() >= POSITION_QTY_EPSILON: active_pos_side, active_pos_data = "short", short_d
        if active_pos_side and active_pos_data:
            style = "bold bright_green" if active_pos_side == "long" else "bold bright_red"
            pos_disp = Text(f"{active_pos_side.upper()}: ",style=style); pos_disp.append("Qty=",style=style).append(self._format_decimal_for_rich(active_pos_data.get("qty"),amount_display_dp))
            pos_disp.append(" | EntryPx=",style="dim").append(self._format_decimal_for_rich(active_pos_data.get("entry_price"),price_dp))
            pos_disp.append(" | PnL=",style="dim").append(self._format_decimal_for_rich(active_pos_data.get("unrealized_pnl"),4,add_commas=True,highlight_negative=True))

            prot_txt = Text(" | Protection: ",style="dim"); exch_prot_parts = []
            sl, tp, tsl_act, tsl_trig = active_pos_data.get("stop_loss_price"), active_pos_data.get("take_profit_price"), active_pos_data.get("is_tsl_active",False), active_pos_data.get("tsl_trigger_price")
            if tsl_act: exch_prot_parts.append(Text("TSL",style="bright_magenta"));_ = exch_prot_parts.append(Text(f"(ActPx:{self._format_decimal_for_rich(tsl_trig,price_dp).plain})",style="magenta")) if tsl_trig else None
            elif sl or tp:
                if sl: exch_prot_parts.append(Text(f"SL:{self._format_decimal_for_rich(sl,price_dp).plain}",style="bright_yellow"))
                if tp: exch_prot_parts.append(Text(f"TP:{self._format_decimal_for_rich(tp,price_dp).plain}",style="bright_yellow"))
            if not exch_prot_parts: exch_prot_parts.append(Text("None",style="dim"))
            prot_txt.append("Exch:").append(Text(" ").join(exch_prot_parts)); prot_txt.append(" LocalTrk:").append(Text(str(protection_status_tracker.get(active_pos_side)) if protection_status_tracker.get(active_pos_side) else "None",style="blue" if protection_status_tracker.get(active_pos_side) else "dim"))
            mismatch = (tsl_act and protection_status_tracker.get(active_pos_side)!="ACTIVE_TSL") or ((sl or tp) and not tsl_act and protection_status_tracker.get(active_pos_side)!="ACTIVE_SLTP") or (not tsl_act and not sl and not tp and protection_status_tracker.get(active_pos_side) is not None)
            if mismatch: prot_txt.append(Text(" [TrackerMismatch?]",style="bold bright_yellow"))
            pos_disp.append(prot_txt)
        panel.append(pos_disp); panel.append("\n"); panel.append("---\n", style="dim")

        panel.append("Signal/Status: ", style="bold bright_cyan"); reason = str(signal_check_result_or_status.get("reason","No status info")); style_key="dim"
        if signal_check_result_or_status.get("long",False) or "Long Signal" in reason or "ENTERED_long" in reason: style_key="bold bright_green"
        elif signal_check_result_or_status.get("short",False) or "Short Signal" in reason or "ENTERED_short" in reason: style_key="bold bright_red"
        elif "Blocked" in reason or "FAIL" in reason.upper() or "EmergencyClose" in reason: style_key="yellow"
        elif "CLOSED_" in reason or "HOLDING_" in reason or "INFO:" in reason: style_key="bright_blue"
        elif not any(s in reason for s in ["No Signal:", "Initializing", "Processing..."]): style_key="white"
        panel.append(Text("\n             ".join(textwrap.wrap(reason, width=max(20, console.width - 20), subsequent_indent="")), style=style_key))
        console.print(Panel(panel, title=f"[bold bright_magenta]{title}[/]", border_style="bright_blue", expand=False, padding=(1,2)))

# --- Trading Bot Class ---
class TradingBot:
    def __init__(self):
        logger.info(f"{Style.BRIGHT}{Fore.MAGENTA}--- Initializing Pyrmethus v4.5.7 (Neon Nexus Edition) ---{Style.RESET_ALL}")
        self.config = TradingConfig()
        try:
            self.exchange_manager = ExchangeManager(self.config)
            self.indicator_calculator = IndicatorCalculator(self.config)
            self.signal_generator = SignalGenerator(self.config)
            self.order_manager = OrderManager(self.config, self.exchange_manager)
        except ValueError as ve: logger.critical(f"{Style.BRIGHT}{Fore.RED}Bot Init Failed (Component Init Error): {ve}. Halting.{Style.RESET_ALL}"); sys.exit(1)
        except Exception as e: logger.critical(f"{Style.BRIGHT}{Fore.RED}Unexpected critical error during Bot component init: {e}. Halting.{Style.RESET_ALL}", exc_info=True); sys.exit(1)
        self.status_display = StatusDisplay(self.config)
        self.shutdown_requested = False
        self._setup_signal_handlers()
        logger.info(f"{Style.BRIGHT}{Fore.GREEN}Pyrmethus components initialized. Ready to conjure trades.{Style.RESET_ALL}")

    def _setup_signal_handlers(self):
        for sig in [signal.SIGINT, signal.SIGTERM]:
            try: signal.signal(sig, self._signal_handler_callback); logger.debug(f"Signal handler for {signal.Signals(sig).name} set.")
            except (ValueError, OSError, AttributeError, Exception) as e: logger.warning(f"{Fore.YELLOW}Could not set OS signal handler for {sig}: {e}{Style.RESET_ALL}")

    def _signal_handler_callback(self, sig_num: int, frame: Optional[types.FrameType]):
        sig_name = signal.Signals(sig_num).name if hasattr(signal, "Signals") and isinstance(sig_num, int) and sig_num in signal.Signals else f"Signal {sig_num}"
        if not self.shutdown_requested:
            console.print(f"\n[bold yellow]Signal {sig_name} received. Graceful shutdown initiated...[/]"); logger.warning(f"Signal {sig_name} received. Initiating graceful shutdown...")
            self.shutdown_requested = True
        else: logger.warning("Shutdown already in progress. Ignoring additional signal.")

    def _display_startup_info(self):
        console.print(Panel(Text(
            f"Symbol: {self.config.symbol}\nInterval: {self.config.interval}\nMarket Type: {self.config.market_type} (Category: {self.config.bybit_v5_category})\n"
            f"Position Index: {self.config.position_idx} (0=One-Way, 1=HedgeBuy, 2=HedgeSell)\nRisk Per Trade: {self.config.risk_percentage * 100:.3f}%\n"
            f"SL/TP Multipliers (ATR): SL={self.config.sl_atr_multiplier.normalize()}, TP={self.config.tp_atr_multiplier.normalize()}\n"
            f"TSL Activation (ATR Mult): {self.config.tsl_activation_atr_multiplier.normalize()}, TSL Percent: {self.config.trailing_stop_percent.normalize()}%\n"
            f"Trade Only With Trend: {self.config.trade_only_with_trend}\nJournaling: {self.config.enable_journaling} ('{self.config.journal_file_path}')\n"
            f"Log Level: {log_level_display_name}", style="bright_white"), title="[bold cyan]Pyrmethus Configuration Summary[/]", border_style="cyan", expand=False))

    def run(self):
        self._display_startup_info()
        termux_notify(f"Pyrmethus Started", f"Trading {self.config.symbol} on {self.config.interval}.")
        cycle_count = 0
        while not self.shutdown_requested:
            cycle_count += 1; cycle_start_monotonic = time.monotonic()
            logger.debug(f"{Fore.BLUE}--- Cycle {cycle_count} ---{Style.RESET_ALL}")
            try: self.trading_spell_cycle(cycle_count)
            except KeyboardInterrupt: logger.warning("\nKeyboardInterrupt in main loop. Shutting down."); self.shutdown_requested = True; break
            except ccxt.AuthenticationError as e: logger.critical(f"{Style.BRIGHT}{Fore.RED}CRITICAL AUTH ERROR (Cycle {cycle_count}): {e}. Halting.{Style.RESET_ALL}",exc_info=False); termux_notify("Pyrmethus CRITICAL ERROR",f"Auth fail: {str(e)[:100]}"); self.shutdown_requested=True; break
            except SystemExit as e: logger.warning(f"SystemExit (code {e.code}) in cycle. Terminating."); self.shutdown_requested=True; break
            except Exception as e:
                logger.error(f"{Style.BRIGHT}{Fore.RED}Unhandled exception in cycle {cycle_count}: {e}{Style.RESET_ALL}", exc_info=True); termux_notify("Pyrmethus Cycle Error", f"Exception in cycle {cycle_count}.")
                sleep_err = self.config.loop_sleep_seconds * 2; logger.info(f"Sleeping {sleep_err}s after error."); time.sleep(sleep_err); continue

            duration = time.monotonic() - cycle_start_monotonic; sleep_needed = max(0, self.config.loop_sleep_seconds - duration)
            logger.debug(f"Cycle {cycle_count} took {duration:.2f}s.")
            if not self.shutdown_requested and sleep_needed > 0:
                logger.debug(f"Sleeping {sleep_needed:.2f}s..."); sleep_end = time.monotonic() + sleep_needed
                try:
                    while time.monotonic() < sleep_end and not self.shutdown_requested: time.sleep(min(0.5, sleep_needed))
                except KeyboardInterrupt: logger.warning("\nKeyboardInterrupt during sleep. Shutting down."); self.shutdown_requested = True
            if self.shutdown_requested: logger.info("Shutdown requested. Exiting main loop."); break
        self.graceful_shutdown()
        console.print(f"\n[bold bright_cyan]Pyrmethus ({self.config.symbol}) session ended.[/]")

    def trading_spell_cycle(self, cycle_num: int):
        status_dict: Dict[str, Any] = {"reason": f"Cycle {cycle_num} Processing..."}
        ohlcv_df = self.exchange_manager.fetch_ohlcv()
        if ohlcv_df is None or ohlcv_df.empty:
            logger.error(f"{Fore.RED}Cycle {cycle_num} Aborted: No OHLCV.{Style.RESET_ALL}"); status_dict = {"reason": f"FAIL_CYCLE_{cycle_num}:FETCH_OHLCV"}
            self.status_display.print_status_panel(cycle_num, None, None, None, None, None, status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info); return
        try:
            curr_px = safe_decimal(ohlcv_df.iloc[-1]["close"]); last_ts = ohlcv_df.index[-1].to_pydatetime()
            if curr_px.is_nan() or curr_px <= 0: raise ValueError(f"Invalid latest close: {curr_px.normalize() if not curr_px.is_nan() else 'NaN'}")
            logger.debug(f"Latest Candle: Ts={last_ts.strftime('%Y-%m-%d %H:%M:%S %Z')}, Price={curr_px.normalize()}")
        except (IndexError, KeyError, ValueError, TypeError) as e:
            logger.error(f"{Fore.RED}Cycle {cycle_num} Aborted: Error processing latest candle: {e}{Style.RESET_ALL}"); status_dict = {"reason": f"FAIL_CYCLE_{cycle_num}:PROCESS_CANDLE ({e})"}
            self.status_display.print_status_panel(cycle_num, None, None, None, None, None, status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info); return

        indicators = self.indicator_calculator.calculate_indicators(ohlcv_df)
        if not indicators:
            logger.error(f"{Fore.RED}Cycle {cycle_num} Aborted: No indicators.{Style.RESET_ALL}"); status_dict = {"reason": f"FAIL_CYCLE_{cycle_num}:CALC_INDICATORS"}
            self.status_display.print_status_panel(cycle_num, last_ts, curr_px, None, None, None, status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info); return

        equity, _ = self.exchange_manager.get_balance()
        if equity is None or equity.is_nan() or equity <= 0:
            logger.error(f"{Fore.RED}Cycle {cycle_num} Aborted: Invalid equity ({equity}).{Style.RESET_ALL}"); status_dict = {"reason": f"FAIL_CYCLE_{cycle_num}:FETCH_EQUITY_INVALID"}
            self.status_display.print_status_panel(cycle_num, last_ts, curr_px, indicators, None, equity, status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info); return

        pos_summary = self.exchange_manager.get_current_position()
        if pos_summary is None:
            logger.error(f"{Fore.RED}Cycle {cycle_num} Aborted: No position state.{Style.RESET_ALL}"); status_dict = {"reason": f"FAIL_CYCLE_{cycle_num}:FETCH_POSITION"}
            self.status_display.print_status_panel(cycle_num, last_ts, curr_px, indicators, None, equity, status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info); return

        active_side: Optional[str] = None; active_pos: Optional[Dict[str,Any]] = None
        long_pos_data = pos_summary.get("long", {})
        short_pos_data = pos_summary.get("short", {})

        if long_pos_data and safe_decimal(long_pos_data.get("qty", Decimal(0))).copy_abs() >= POSITION_QTY_EPSILON:
            active_side, active_pos = "long", long_pos_data
        elif short_pos_data and safe_decimal(short_pos_data.get("qty", Decimal(0))).copy_abs() >= POSITION_QTY_EPSILON:
            active_side, active_pos = "short", short_pos_data


        if active_side and active_pos:
            qty, entry_px, atr_val = safe_decimal(active_pos.get("qty")), safe_decimal(active_pos.get("entry_price")), indicators.get("atr", Decimal("NaN")) # type: ignore[union-attr]
            if self.order_manager.protection_tracker.get(active_side) == "ACTIVE_SLTP" and not any(v.is_nan() or v <= 0 for v in [entry_px, curr_px, atr_val]): # type: ignore[operator]
                self.order_manager.manage_trailing_stop(active_side, entry_px, curr_px, atr_val) # type: ignore[arg-type]
                if self.order_manager.protection_tracker.get(active_side) == "ACTIVE_TSL": # TSL activated
                    logger.debug("Re-fetching position after TSL mgmt for display."); pos_summary = self.exchange_manager.get_current_position()
                    if pos_summary: active_pos = pos_summary.get(active_side, {})
                    else: active_pos = None # Fetch failed

            if self.order_manager.protection_tracker.get(active_side) != "ACTIVE_TSL":
                exit_sig = self.signal_generator.check_exit_signals(active_side, indicators)
                if exit_sig:
                    logger.trade(f"Attempting close {active_side.upper()} due to: {exit_sig}")
                    if not qty.is_nan() and qty > 0:
                        closed = self.order_manager.close_position(active_side, qty, reason=exit_sig)
                        status_dict = {"reason": f"CLOSED_{active_side.upper()}_BY_SIGNAL" if closed else f"FAIL:CLOSE_SIGNAL_{active_side.upper()}"}
                        pos_summary = self.exchange_manager.get_current_position() # Refresh for display
                        self.status_display.print_status_panel(cycle_num, last_ts, curr_px, indicators, pos_summary, equity, status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info)
                        return # Action taken, end cycle.
                    else: logger.warning(f"Exit signal for {active_side.upper()} but qty invalid ({qty}). Cannot close.")

            # Re-check position after potential actions (TSL activation, signal exit)
            logger.debug(f"Re-fetching position state for {active_side} after management/exit checks.")
            pos_summary_after_actions = self.exchange_manager.get_current_position()
            if pos_summary_after_actions is None:
                logger.warning(f"Failed to re-fetch position state for {active_side} after actions. Status may be stale.")
            else:
                pos_summary = pos_summary_after_actions # Update with latest
                long_data_updated = pos_summary.get("long", {})
                short_data_updated = pos_summary.get("short", {})
                if not (long_data_updated and safe_decimal(long_data_updated.get("qty", Decimal(0))).copy_abs() >= POSITION_QTY_EPSILON) and \
                   not (short_data_updated and safe_decimal(short_data_updated.get("qty", Decimal(0))).copy_abs() >= POSITION_QTY_EPSILON):
                    logger.info(f"Position {active_side.upper()} appears closed by exchange (e.g. SL/TP hit).")
                    status_dict = {"reason": f"INFO:POS_{active_side.upper()}_CLOSED_BY_EXCH"}
                    self.order_manager.protection_tracker[active_side] = None
                    active_side = None # Now flat
                else: # Still in position, update active_pos
                    active_pos = pos_summary.get(active_side, {})


        if not active_side: # If flat (either initially or after exit)
            logger.debug("Currently flat. Checking for entry signals...")
            entry_signals = self.signal_generator.generate_signals(ohlcv_df, indicators)
            status_dict = entry_signals # Use signal reason for display

            entry_order_side: Optional[str] = None
            if entry_signals.get("long"): entry_order_side = "buy"
            elif entry_signals.get("short"): entry_order_side = "sell"

            if entry_order_side:
                atr_val_for_entry = indicators.get("atr", Decimal("NaN")) # type: ignore[union-attr]
                if not equity.is_nan() and equity > 0 and not atr_val_for_entry.is_nan() and atr_val_for_entry > 0 and not curr_px.is_nan() and curr_px > 0:
                    entry_success = self.order_manager.place_risked_market_order(entry_order_side, atr_val_for_entry, equity, curr_px)
                    entered_side_log = "long" if entry_order_side == "buy" else "short"
                    status_dict = {"reason": f"ENTERED_{entered_side_log.upper()}" if entry_success else f"FAIL:ENTRY_{entered_side_log.upper()}"}
                    pos_summary = self.exchange_manager.get_current_position() # Refresh for display
                    self.status_display.print_status_panel(cycle_num, last_ts, curr_px, indicators, pos_summary, equity, status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info)
                    return # Action taken, end cycle
                else: logger.warning(f"Cannot attempt {entry_order_side.upper()} entry: Missing data (Equity/ATR/Price)."); status_dict = {"reason": f"FAIL:ENTRY_DATA_MISSING_{entry_order_side.upper()}"}
        else: # Still in position, no exit signal triggered
            status_dict = {"reason": f"HOLDING_{active_side.upper()}"}

        self.status_display.print_status_panel(cycle_num, last_ts, curr_px, indicators, pos_summary, equity, status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info)

    def graceful_shutdown(self):
        logger.info(f"{Style.BRIGHT}{Fore.YELLOW}--- Pyrmethus Graceful Shutdown Sequence ---{Style.RESET_ALL}")
        logger.info("Positions with exchange-side SL/TP/TSL will remain active. No specific order cleanup configured for this strategy.")
        termux_notify("Pyrmethus Shutdown", f"Bot for {self.config.symbol} is shutting down.")
        logger.info(f"{Style.BRIGHT}{Fore.YELLOW}--- Pyrmethus Shutdown Complete ---{Style.RESET_ALL}")


if __name__ == "__main__":
    try:
        bot = TradingBot()
        bot.run()
    except SystemExit as e:
        if e.code == 0: logger.info("Pyrmethus terminated normally.")
        else: logger.warning(f"Pyrmethus terminated with exit code: {e.code}")
        sys.exit(e.code)
    except Exception as main_exception:
        log_func = logger.critical if 'logger' in globals() and hasattr(logger, 'critical') else print
        # Corrected color usage for the final error message
        err_msg_plain = f"CRITICAL UNHANDLED EXCEPTION in Pyrmethus main execution: {main_exception}"
        if 'colorama_init' in globals() and 'Fore' in globals() and 'Style' in globals():
             colorama_init(autoreset=True)
             err_msg_colored = f"{Style.BRIGHT}{Fore.RED}{err_msg_plain}{Style.RESET_ALL}"
             log_func(err_msg_colored, exc_info=True)
        else:
             log_func(err_msg_plain, exc_info=True if 'logger' in globals() else False)
             if 'logger' not in globals(): # If logger isn't available, print traceback manually
                import traceback
                traceback.print_exc()

        if 'termux_notify' in globals():
            termux_notify("Pyrmethus CRASHED", f"Critical unhandled exception: {str(main_exception)[:100]}")
        sys.exit(1)

Okay, I've reviewed the Python script and will enhance it for clarity, conciseness, and flow. My main focuses will be:

1.  **Improving `StatusDisplay` Output**:
    *   Removing the redundant `---` separators within the panel.
    *   Restructuring the "Indicators" section for better readability, likely using multiple lines.
    *   Overhauling the "Signal/Status" section by modifying `SignalGenerator` to produce more structured reasons, which `StatusDisplay` will then format into a clearer, multi-line output.

2.  **Refining `SignalGenerator` Reasons**:
    *   Making the detailed reasons generated by `_generate_original_signals` and `_generate_vt_signals` slightly more concise.
    *   Ensuring these internal methods return reasons *without* strategy prefixes (like "Original:" or "VT:").
    *   The main `generate_signals` method will then intelligently combine these reasons and add appropriate prefixes for the structured output, avoiding redundancies like "Orig: Original NoSignal...".

3.  **General Code Review**:
    *   **Clarity**: Improving comments where needed, ensuring variable names are intuitive.
    *   **Conciseness**: Removing unnecessary verbosity in logging (especially for INFO level) and code logic where possible without sacrificing readability or functionality. For example, slightly more concise logging messages or helper function calls if applicable.
    *   **Flow**: Ensuring logical grouping within classes and methods. Major architectural changes (like splitting into multiple files) will be avoided to keep the enhancement focused on the existing structure.
    *   **Error Handling and Logging**: Reviewing for consistency and user-friendliness of error messages.

Let's apply these enhancements.

```python
