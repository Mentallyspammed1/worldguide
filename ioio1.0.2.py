
terns (Original & VT)...
2025-05-10 12:48:28 [INFO    ] (ioio1.0.2.py:756) Indicator patterns woven successfully.
╭───────────  Cycle 33 | FARTCOIN/USDT:USDT (3m) | 2025-05-10 17:48:00 UTC  ───────────╮
│                                                                                      │
│  Price: 1.3272 | Equity: 4.72 USDT                                                   ││  ---                                                                                 │
│  Indicators: EMA(F/S/T): 1.3351/1.3344/1.3288 | Stoch(K/D/PrevK): 28.3/37.0/42.4 |   ││  ATR(10): 0.00771 | ADX(10): 39.0 [+DI:19.2 -DI:23.9] | VT:  | TrendEMA:1.3372 |     │
│  VWMA:1.3342 | VolSpike:False | Candle:Red                                           ││  ---                                                                                 │
│  Position: FLAT                                                                      │
│  ---                                                                                 ││  Signal/Status: No Signal. Orig: Original NoSignal: Base(EMA_X:Bullish,Stoch:K:28.3  │
│               (OS:25/OB:75) KD_Cross(Bull:False/Bear:False)) -> LongBase:False,      ││               ShortBase:False. | VT: VT: Indicators NaN/Missing or Invalid Type:     │
│               vt_boolean_indicators_invalid_type_or_missing.                         ││                                                                                      │╰──────────────────────────────────────────────────────────────────────────────────────╯
2025-05-10 12:48:30 [INFO    ] (ioio1.0.2.py:649) # Weaving indicator patterns (Original & VT)...                                                                               2025-05-10 12:48:30 [INFO    ] (ioio1.0.2.py:756) Indicator patterns woven successfully.╭───────────  Cycle 34 | FARTCOIN/USDT:USDT (3m) | 2025-05-10 17:48:00 UTC  ───────────╮│                                                                                      ││  Price: 1.3284 | Equity: 4.70 USDT                                                   │
│  ---                                                                                 ││  Indicators: EMA(F/S/T): 1.3354/1.3345/1.3289 | Stoch(K/D/PrevK): 30.2/37.6/42.4 |   ││  ATR(10): 0.00771 | ADX(10): 39.0 [+DI:19.2 -DI:23.9] | VT:  | TrendEMA:1.3372 |     │
│  VWMA:1.3342 | VolSpike:False | Candle:Red                                           │
│  ---                                                                                 ││  Position: FLAT                                                                      ││  ---                                                                                 ││  Signal/Status: No Signal. Orig: Original NoSignal: Base(EMA_X:Bullish,Stoch:K:30.2  ││               (OS:25/OB:75) KD_Cross(Bull:False/Bear:False)) -> LongBase:False,      ││               ShortBase:False. | VT: VT: Indicators NaN/Missing or Invalid Type:     │
│               vt_boolean_indicators_invalid_type_or_missing.                         ││                                                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────╯
2025-05-10 12:48:36 [INFO    ] (ioio1.0.2.py:649) # Weaving

# -*- coding: utf-8 -*-
# pylint: disable=logging-fstring-interpolation, too-many-instance-attributes, too-many-arguments, too-many-locals, too-many-public-methods, invalid-name, unused-argument, too-many-lines, unnecessary-pass, unnecessary-lambda-assignment, line-too-long
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
    "pandas-ta", # Added for VolumaticTrend indicators like VWMA
    "rich",
    "colorama",
    "requests",
]

# Attempt to import colorama first for styled error messages
# These will be properly assigned for global use after the main import block
_TEMP_FORE = None
_TEMP_STYLE = None
_TEMP_COLORAMA_INIT = None
_COLORAMA_SUCCESSFULLY_IMPORTED = False

try:
    from colorama import Fore as _F, Style as _S, init as _CI
    _TEMP_FORE = _F
    _TEMP_STYLE = _S
    _TEMP_COLORAMA_INIT = _CI
    _COLORAMA_SUCCESSFULLY_IMPORTED = True
except ImportError:
    # colorama itself is missing, this will be handled in the main try-except block
    pass

try:
    if not _COLORAMA_SUCCESSFULLY_IMPORTED: # If colorama failed above, try again to catch its specific error name
        from colorama import Fore, Style, init as colorama_init
    else: # colorama was imported, assign its components for use
        Fore, Style, colorama_init = _TEMP_FORE, _TEMP_STYLE, _TEMP_COLORAMA_INIT

    import ccxt
    import numpy as np
    import pandas as pd
    import pandas_ta as ta # For VWMA and other TA functions
    import requests
    from dotenv import load_dotenv
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table # Keep for potential future use, though not directly used in StatusDisplay
    from rich.text import Text

except ImportError as e:
    missing_pkg = e.name
    # Determine if colorama is available for styling the error message
    can_use_colorama_for_error = _COLORAMA_SUCCESSFULLY_IMPORTED and missing_pkg != "colorama"

    if missing_pkg == "colorama" or not can_use_colorama_for_error:
        print(f"Missing essential spell component: {missing_pkg}")
        if missing_pkg == "colorama":
            print("Missing essential package: colorama. Cannot display colored output.")
        print(f"To conjure it, cast: pip install {missing_pkg}")
        print("\nOr, to ensure all scrolls are present, cast:")
        print(f"pip install {' '.join(COMMON_PACKAGES)}")
        sys.exit(1)
    else:
        # Colorama is available, use it for a nicer error message
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
            pip_pkgs_to_install = list(COMMON_PACKAGES) # Start with all packages for pip

            if "pandas" in COMMON_PACKAGES:
                termux_pkgs_to_install.append("python-pandas")
                if 'pandas' in pip_pkgs_to_install: pip_pkgs_to_install.remove('pandas')
            if "numpy" in COMMON_PACKAGES:
                termux_pkgs_to_install.append("python-numpy")
                if 'numpy' in pip_pkgs_to_install: pip_pkgs_to_install.remove('numpy')
            # pandas-ta is usually pip-installed
            
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
V5_HEDGE_MODE_POSITION_IDX = 0 # Default for one-way mode or specific hedge leg
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
        """Logs a message with custom level TRADE."""
        if self.isEnabledFor(TRADE_LEVEL_NUM):
            # pylint: disable=protected-access
            self._log(TRADE_LEVEL_NUM, message, args, **kws) # type: ignore[arg-type]
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
    # Early print, logger not fully set up, but colorama is
    print(f"{Fore.YELLOW}Warning: Invalid LOG_LEVEL '{log_level_str_env}'. Defaulting to INFO.{Style.RESET_ALL}")
    log_level_to_set = logging.INFO
    log_level_display_name = "INFO" # For display in startup info

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
            logger.warning("Termux notify failed: 'termux-toast' command not found. Is Termux:API installed?")
        except subprocess.TimeoutExpired:
            logger.warning(f"Termux notify failed: command timed out after {TERMUX_NOTIFY_TIMEOUT} seconds.")
        except Exception as e: # Catch any other unexpected errors
            logger.warning(f"Termux notify failed unexpectedly: {e}", exc_info=False) # Keep exc_info concise

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
                logger.info(f"{Style.BRIGHT}{Fore.GREEN}Successfully executed {func_name} on attempt {attempt + 1}/{max_retries + 1} after previous failures.{Style.RESET_ALL}")
            return result
        except fatal_exceptions as e:
            logger.critical(f"{Style.BRIGHT}{Fore.RED}Fatal error ({type(e).__name__}) executing {func_name}: {e}. Halting immediately.{Style.RESET_ALL}", exc_info=False)
            raise e
        except fail_fast_exceptions as e:
            logger.error(f"{Fore.RED}Fail-fast error ({type(e).__name__}) executing {func_name}: {e}. Not retrying this call.{Style.RESET_ALL}", exc_info=False)
            last_exception = e; break
        except retry_on_exceptions as e:
            last_exception = e
            err_summary = str(e)[:150] + ("..." if len(str(e)) > 150 else "")
            msg_base = f"Retryable error ({type(e).__name__}) on attempt {attempt + 1}/{max_retries + 1} for {func_name}: {err_summary}."
            if attempt < max_retries:
                logger.warning(f"{Fore.YELLOW}{msg_base} Retrying in {delay_seconds}s...{Style.RESET_ALL}")
                time.sleep(delay_seconds)
            else:
                logger.error(f"{Fore.RED}Max retries ({max_retries + 1}) reached for {func_name} after retryable error. Last error: {e}{Style.RESET_ALL}", exc_info=False)
        except ccxt.ExchangeError as e:
            last_exception = e
            err_summary = str(e)[:150] + ("..." if len(str(e)) > 150 else "")
            logger.error(f"{Fore.RED}Unhandled ExchangeError during {func_name}: {err_summary}{Style.RESET_ALL}", exc_info=False) # Set exc_info=False for less verbose logs on generic exchange errors
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
    # This path should ideally not be hit if logic is correct
    raise RuntimeError(f"Function {func_name} failed after {max_retries + 1} attempts without raising a recognized or captured exception, or returning a value.")

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
            logger.warning(f"Environment file '{env_path}' not found. Relying solely on system environment variables.")

        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT", Style.DIM)
        self.market_type: str = self._get_env("MARKET_TYPE", "linear", Style.DIM, allowed_values=["linear", "inverse", "swap"]).lower()
        # Determine bybit_v5_category after symbol and market_type are loaded
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
        
        # Original Strategy Indicator Periods
        self.trend_ema_period: int = self._get_env("TREND_EMA_PERIOD", 12, Style.DIM, cast_type=int, min_val=5, max_val=500)
        self.fast_ema_period: int = self._get_env("FAST_EMA_PERIOD", 9, Style.DIM, cast_type=int, min_val=1, max_val=200)
        self.slow_ema_period: int = self._get_env("SLOW_EMA_PERIOD", 21, Style.DIM, cast_type=int, min_val=2, max_val=500)
        self.stoch_period: int = self._get_env("STOCH_PERIOD", 7, Style.DIM, cast_type=int, min_val=1, max_val=100)
        self.stoch_smooth_k: int = self._get_env("STOCH_SMOOTH_K", 3, Style.DIM, cast_type=int, min_val=1, max_val=10)
        self.stoch_smooth_d: int = self._get_env("STOCH_SMOOTH_D", 3, Style.DIM, cast_type=int, min_val=1, max_val=10)
        self.atr_period: int = self._get_env("ATR_PERIOD", 5, Style.DIM, cast_type=int, min_val=1, max_val=100)
        self.adx_period: int = self._get_env("ADX_PERIOD", 14, Style.DIM, cast_type=int, min_val=2, max_val=100)
        
        # Original Strategy Signal Logic Thresholds
        self.stoch_oversold_threshold: Decimal = self._get_env("STOCH_OVERSOLD_THRESHOLD", DEFAULT_STOCH_OVERSOLD, Fore.CYAN, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("45"))
        self.stoch_overbought_threshold: Decimal = self._get_env("STOCH_OVERBOUGHT_THRESHOLD", DEFAULT_STOCH_OVERBOUGHT, Fore.CYAN, cast_type=Decimal, min_val=Decimal("55"), max_val=Decimal("100"))
        self.trend_filter_buffer_percent: Decimal = self._get_env("TREND_FILTER_BUFFER_PERCENT", Decimal("0.5"), Fore.CYAN, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("5"))
        self.atr_move_filter_multiplier: Decimal = self._get_env("ATR_MOVE_FILTER_MULTIPLIER", Decimal("0.5"), Fore.CYAN, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("5"))
        self.min_adx_level: Decimal = self._get_env("MIN_ADX_LEVEL", DEFAULT_MIN_ADX, Fore.CYAN, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("90"))

        # VolumaticTrend Strategy Configuration
        self.vt_enable: bool = self._get_env("VT_ENABLE", True, Style.DIM, cast_type=bool) # Default to True, can be overridden by .env
        self.vt_trend_ema_period: int = self._get_env("VT_TREND_EMA_PERIOD", 200, Style.DIM, cast_type=int, min_val=50, max_val=500)
        self.vt_vwma_period: int = self._get_env("VT_VWMA_PERIOD", 20, Style.DIM, cast_type=int, min_val=5, max_val=100)
        self.vt_volume_spike_lookback: int = self._get_env("VT_VOLUME_SPIKE_LOOKBACK", 20, Style.DIM, cast_type=int, min_val=5, max_val=100)
        self.vt_volume_spike_multiplier: Decimal = self._get_env("VT_VOLUME_SPIKE_MULTIPLIER", Decimal("2.0"), Fore.CYAN, cast_type=Decimal, min_val=Decimal("1.1"), max_val=Decimal("5.0"))

        self.api_key: str = self._get_env("BYBIT_API_KEY", None, Fore.RED, is_secret=True)
        self.api_secret: str = self._get_env("BYBIT_API_SECRET", None, Fore.RED, is_secret=True)
        self.ohlcv_limit: int = self._get_env("OHLCV_LIMIT", DEFAULT_OHLCV_LIMIT, Style.DIM, cast_type=int, min_val=50, max_val=1000)
        self.loop_sleep_seconds: int = self._get_env("LOOP_SLEEP_SECONDS", DEFAULT_LOOP_SLEEP, Style.DIM, cast_type=int, min_val=1)
        self.order_check_delay_seconds: int = self._get_env("ORDER_CHECK_DELAY_SECONDS", 2, Style.DIM, cast_type=int, min_val=1)
        self.order_fill_timeout_seconds: int = self._get_env("ORDER_FILL_TIMEOUT_SECONDS", 20, Style.DIM, cast_type=int, min_val=5)
        self.max_fetch_retries: int = self._get_env("MAX_FETCH_RETRIES", DEFAULT_MAX_RETRIES, Style.DIM, cast_type=int, min_val=0, max_val=10)
        self.retry_delay_seconds: int = self._get_env("RETRY_DELAY_SECONDS", DEFAULT_RETRY_DELAY, Style.DIM, cast_type=int, min_val=1)
        self.trade_only_with_trend: bool = self._get_env("TRADE_ONLY_WITH_TREND", True, Style.DIM, cast_type=bool) # For original strategy's trend filter
        self.journal_file_path: str = self._get_env("JOURNAL_FILE_PATH", DEFAULT_JOURNAL_FILE, Style.DIM)
        self.enable_journaling: bool = self._get_env("ENABLE_JOURNALING", True, Style.DIM, cast_type=bool)
        
        self._validate_config()
        logger.debug("Configuration loaded and validated successfully.")

    def _determine_v5_category(self) -> str:
        # This method is called after self.symbol and self.market_type are initialized
        try:
            category: str
            if self.market_type == "inverse": category = "inverse"
            elif self.market_type in ["linear", "swap"]: category = "linear"
            else: raise ValueError(f"Unsupported MARKET_TYPE '{self.market_type}' for category determination.")
            if ":" not in self.symbol:
                 logger.warning(f"Symbol '{self.symbol}' does not explicitly include the settle currency (e.g., :USDT or :BTC). Explicit format (BASE/QUOTE:SETTLE) is recommended for V5 API clarity.")
            logger.info(f"Determined Bybit V5 API category: '{category}' for symbol '{self.symbol}' and market type '{self.market_type}'")
            return category
        except ValueError as e:
            logger.critical(f"{Style.BRIGHT}{Fore.RED}Could not determine V5 category: {e}. Halting.{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)
        return "" # Should be unreachable

    def _validate_config(self):
        if self.fast_ema_period >= self.slow_ema_period:
            logger.critical(f"{Style.BRIGHT}{Fore.RED}Validation failed: FAST_EMA_PERIOD ({self.fast_ema_period}) must be less than SLOW_EMA_PERIOD ({self.slow_ema_period}). Halting.{Style.RESET_ALL}"); sys.exit(1)
        if self.trend_ema_period <= self.slow_ema_period:
            logger.warning(f"{Fore.YELLOW}Config Warning: TREND_EMA_PERIOD ({self.trend_ema_period}) is less than or equal to SLOW_EMA_PERIOD ({self.slow_ema_period}). Trend filter might lag short-term EMA signals.{Style.RESET_ALL}")
        if self.stoch_oversold_threshold >= self.stoch_overbought_threshold:
            logger.critical(f"{Style.BRIGHT}{Fore.RED}Validation failed: STOCH_OVERSOLD_THRESHOLD ({self.stoch_oversold_threshold.normalize()}) must be less than STOCH_OVERBOUGHT_THRESHOLD ({self.stoch_overbought_threshold.normalize()}). Halting.{Style.RESET_ALL}"); sys.exit(1)
        if self.tsl_activation_atr_multiplier < self.sl_atr_multiplier:
            logger.warning(f"{Fore.YELLOW}Config Warning: TSL_ACTIVATION_ATR_MULTIPLIER ({self.tsl_activation_atr_multiplier.normalize()}) is less than SL_ATR_MULTIPLIER ({self.sl_atr_multiplier.normalize()}). TSL may activate before initial SL distance is fully established by price movement.{Style.RESET_ALL}")
        if self.tp_atr_multiplier > Decimal("0") and self.tp_atr_multiplier <= self.sl_atr_multiplier:
            logger.warning(f"{Fore.YELLOW}Config Warning: TP_ATR_MULTIPLIER ({self.tp_atr_multiplier.normalize()}) is less than or equal to SL_ATR_MULTIPLIER ({self.sl_atr_multiplier.normalize()}). This implies a Risk:Reward ratio of 1:1 or less.{Style.RESET_ALL}")
        
        # Validate VolumaticTrend specific EMA periods and OHLCV_LIMIT
        if self.vt_enable:
            if self.vt_trend_ema_period < self.vt_vwma_period: # VWMA period is usually shorter than long-term trend EMA
                 logger.warning(f"{Fore.YELLOW}Config Warning (VT): VT_TREND_EMA_PERIOD ({self.vt_trend_ema_period}) is less than VT_VWMA_PERIOD ({self.vt_vwma_period}). This is unusual for typical VWMA usage, ensure it's intended for your strategy.{Style.RESET_ALL}")
            
            # Check OHLCV_LIMIT against the maximum period needed by VT indicators
            min_ohlcv_for_vt = max(self.vt_trend_ema_period, self.vt_vwma_period, self.vt_volume_spike_lookback) + 20 # Add buffer
            if self.ohlcv_limit < min_ohlcv_for_vt:
                 logger.warning(f"{Fore.YELLOW}Config Warning: OHLCV_LIMIT ({self.ohlcv_limit}) may be too small for VolumaticTrend indicator periods (longest is {max(self.vt_trend_ema_period, self.vt_vwma_period, self.vt_volume_spike_lookback)}, needs ~{min_ohlcv_for_vt} candles).{Style.RESET_ALL}")


    def _cast_value(self, key: str, value_str: str, cast_type: Type, default: Any) -> Any:
        val_to_cast = value_str.strip()
        if not val_to_cast:
            if default is None or (isinstance(default, str) and not default): return default
            logger.warning(f"Empty value string for '{key}' after stripping. Using default '{default}'."); return default
        try:
            if cast_type == bool: return val_to_cast.lower() in ["true", "1", "yes", "y", "on"]
            elif cast_type == Decimal:
                if val_to_cast.lower() in ["nan", "none", "null"]: return Decimal("NaN")
                return Decimal(val_to_cast)
            elif cast_type == int:
                dec_val = Decimal(val_to_cast)
                if dec_val.to_integral_value(rounding=ROUND_DOWN) != dec_val:
                    raise ValueError(f"Decimal value '{val_to_cast}' with fractional part cannot be cast to int without loss.")
                return int(dec_val)
            return cast_type(val_to_cast)
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.error(f"{Fore.RED}Cast failed for '{key}' (value: '{value_str}', target_type: {cast_type.__name__}): {e}. Using default '{default}'.{Style.RESET_ALL}"); return default

    def _validate_value(self, key: str, value: Any, min_val: Optional[Union[int, float, Decimal]], max_val: Optional[Union[int, float, Decimal]], allowed_values: Optional[List[Any]]) -> bool:
        is_numeric_comparable = isinstance(value, (int, float, Decimal))
        if min_val is not None:
            if not is_numeric_comparable: logger.critical(f"{Style.BRIGHT}{Fore.RED}Validation failed for '{key}': Non-numeric value '{value}' (type: {type(value).__name__}) cannot be compared with min_val '{min_val}'. Halting.{Style.RESET_ALL}"); sys.exit(1)
            if value < min_val: logger.critical(f"{Style.BRIGHT}{Fore.RED}Validation failed for '{key}': Value '{value}' is less than minimum '{min_val}'. Halting.{Style.RESET_ALL}"); sys.exit(1) # type: ignore[operator]
        if max_val is not None:
            if not is_numeric_comparable: logger.critical(f"{Style.BRIGHT}{Fore.RED}Validation failed for '{key}': Non-numeric value '{value}' (type: {type(value).__name__}) cannot be compared with max_val '{max_val}'. Halting.{Style.RESET_ALL}"); sys.exit(1)
            if value > max_val: logger.critical(f"{Style.BRIGHT}{Fore.RED}Validation failed for '{key}': Value '{value}' is greater than maximum '{max_val}'. Halting.{Style.RESET_ALL}"); sys.exit(1) # type: ignore[operator]
        if allowed_values:
            comp_value = str(value).lower() if isinstance(value, str) else value
            lower_allowed = [str(v).lower() if isinstance(v, str) else v for v in allowed_values]
            if comp_value not in lower_allowed:
                logger.error(f"{Fore.RED}Validation failed for '{key}': Invalid value '{value}'. Allowed values are: {allowed_values}. Reverting to default.{Style.RESET_ALL}"); return False
        return True

    def _get_env(self, key: str, default: Any, color: str, cast_type: Type = str, min_val: Optional[Union[int, float, Decimal]] = None, max_val: Optional[Union[int, float, Decimal]] = None, allowed_values: Optional[List[Any]] = None, is_secret: bool = False) -> Any:
        value_str, source_info, use_default_flag, value_to_process_str = os.getenv(key), "environment variable", False, ""
        if value_str is None or value_str.strip() == "":
            if default is None: logger.critical(f"{Style.BRIGHT}{Fore.RED}Required {'secret ' if is_secret else ''}configuration '{key}' not found in environment and no default provided. Halting.{Style.RESET_ALL}"); sys.exit(1)
            use_default_flag, value_to_process_str, source_info = True, str(default), f"default value ('{default if not is_secret else '****'}')"
            log_value_display = default if not is_secret else "****"
        else: value_to_process_str, log_value_display = value_str, "****" if is_secret else value_str
        (logger.warning if use_default_flag and default is not None else logger.info)(f"Using {color}{key}: {log_value_display}{Style.RESET_ALL} (from {source_info})")
        casted_value = self._cast_value(key, value_to_process_str, cast_type, default)
        if not self._validate_value(key, casted_value, min_val, max_val, allowed_values):
            logger.warning(f"{color}Reverting '{key}' to its original default '{default if not is_secret else '****'}' due to non-critical validation failure of processed value '{casted_value}'.{Style.RESET_ALL}")
            casted_value = default
            if not self._validate_value(key, casted_value, min_val, max_val, allowed_values): # Validate the default itself
                logger.critical(f"{Style.BRIGHT}{Fore.RED}FATAL: The hardcoded default value '{default if not is_secret else '****'}' for '{key}' itself failed validation. Halting.{Style.RESET_ALL}"); sys.exit(1)
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
        logger.info(f"Initializing Bybit exchange interface (V5 API, Market Type: {self.config.market_type})...")
        try:
            exchange_params: Dict[str, Any] = {"apiKey": self.config.api_key, "secret": self.config.api_secret, "options": {"defaultType": self.config.market_type, "adjustForTimeDifference": True, "recvWindow": 10000, "brokerId": "PyrmV5NEXUS", "defaultTimeInForce": "GTC"}}
            if os.getenv("USE_BYBIT_TESTNET", "false").lower() == "true":
                logger.warning(f"{Fore.YELLOW}Using Bybit Testnet endpoint.{Style.RESET_ALL}"); exchange_params['urls'] = {'api': 'https://api-testnet.bybit.com'}
            self.exchange = ccxt.bybit(exchange_params)
            self.exchange.fetch_time(); logger.info(f"{Style.BRIGHT}{Fore.GREEN}Bybit V5 interface initialized and connection tested successfully.{Style.RESET_ALL}")
        except ccxt.AuthenticationError as e: logger.critical(f"{Style.BRIGHT}{Fore.RED}Authentication failed: {e}. Check API keys and permissions. Halting.{Style.RESET_ALL}", exc_info=False); sys.exit(1)
        except (ccxt.NetworkError, requests.exceptions.RequestException) as e: logger.critical(f"{Style.BRIGHT}{Fore.RED}Network error initializing exchange: {e}. Check internet connection and endpoint. Halting.{Style.RESET_ALL}", exc_info=True); sys.exit(1)
        except Exception as e: logger.critical(f"{Style.BRIGHT}{Fore.RED}Unexpected error initializing exchange: {e}. Halting.{Style.RESET_ALL}", exc_info=True); sys.exit(1)

    def _load_market_info(self) -> Optional[Dict[str, Any]]:
        if not self.exchange: logger.critical("Exchange not initialized. Cannot load market info. Halting."); sys.exit(1) # Should not be reached
        try:
            logger.info(f"Loading market info for symbol: {self.config.symbol}..."); self.exchange.load_markets(reload=True)
            market = self.exchange.market(self.config.symbol)
            if not market: raise ccxt.ExchangeError(f"Market {self.config.symbol} not found on exchange. Ensure symbol format is correct (e.g., BTC/USDT:USDT).")
            
            def get_dp_from_precision_step(precision_val: Optional[Union[str, float, int]], default_dp: int) -> int:
                if precision_val is None: return default_dp
                prec_dec = safe_decimal(precision_val) 
                if prec_dec.is_nan(): return default_dp
                if prec_dec.is_zero(): return 0
                if prec_dec > Decimal("0") and prec_dec < Decimal("1"): return abs(prec_dec.as_tuple().exponent)
                elif prec_dec >= Decimal("1"):
                    if prec_dec.to_integral_value() == prec_dec: return 0
                    else: exponent = prec_dec.as_tuple().exponent; return abs(exponent) if exponent < 0 else default_dp
                else: return default_dp # Handles negative precision_val, though unlikely

            market["precision_dp"] = {"amount": get_dp_from_precision_step(market.get("precision", {}).get("amount"), DEFAULT_AMOUNT_DP), "price": get_dp_from_precision_step(market.get("precision", {}).get("price"), DEFAULT_PRICE_DP)}
            market["tick_size"] = safe_decimal(market.get("precision", {}).get("price"), default=Decimal('NaN'))
            market["amount_step"] = safe_decimal(market.get("precision", {}).get("amount"), default=Decimal('NaN'))
            market["min_order_size"] = safe_decimal(market.get("limits", {}).get("amount", {}).get("min"), default=Decimal("NaN"))
            market["contract_size"] = safe_decimal(market.get("contractSize"), default=Decimal("1"))
            if market.get("contractSize") is None: logger.warning(f"Contract size not found in market info for {self.config.symbol}. Defaulting to 1.")
            
            min_amt_str = market["min_order_size"].normalize() if not market["min_order_size"].is_nan() else "N/A"
            tick_size_str = market["tick_size"].normalize() if not market["tick_size"].is_nan() else "N/A"
            amount_step_str = market["amount_step"].normalize() if not market["amount_step"].is_nan() else "N/A"
            logger.info(f"Market info for {self.config.symbol} (ID: {market.get('id', 'N/A')}): FormattingDP(Amount={market['precision_dp']['amount']}, Price={market['precision_dp']['price']}), ActualSteps(TickSize={tick_size_str}, AmountStep={amount_step_str}), Limits(MinAmount={min_amt_str}), ContractSize={market['contract_size'].normalize()}, SettleCurrency: {market.get('settle', 'N/A')}")
            return market
        except (ccxt.ExchangeError, KeyError, ValueError, TypeError, Exception) as e: logger.critical(f"{Style.BRIGHT}{Fore.RED}Failed to load or parse market info for {self.config.symbol}: {e}. Halting.{Style.RESET_ALL}", exc_info=True); sys.exit(1)
        return None

    def format_price(self, price: Union[Decimal, str, float, int]) -> str:
        price_decimal = safe_decimal(price)
        if price_decimal.is_nan(): return "NaN"
        precision_dp = self.market_info["precision_dp"]["price"] if self.market_info and "precision_dp" in self.market_info and "price" in self.market_info["precision_dp"] else DEFAULT_PRICE_DP
        try: quantizer = Decimal("1e-" + str(precision_dp)); formatted_price_decimal = price_decimal.quantize(quantizer, rounding=ROUND_HALF_EVEN); return f"{formatted_price_decimal:.{precision_dp}f}"
        except (InvalidOperation, ValueError): return "ERR"

    def format_amount(self, amount: Union[Decimal, str, float, int], rounding_mode=ROUND_DOWN) -> str:
        amount_decimal = safe_decimal(amount)
        if amount_decimal.is_nan(): return "NaN"
        precision_dp = self.market_info["precision_dp"]["amount"] if self.market_info and "precision_dp" in self.market_info and "amount" in self.market_info["precision_dp"] else DEFAULT_AMOUNT_DP
        try: quantizer = Decimal("1e-" + str(precision_dp)); formatted_amount_decimal = amount_decimal.quantize(quantizer, rounding=rounding_mode); return f"{formatted_amount_decimal:.{precision_dp}f}"
        except (InvalidOperation, ValueError): return "ERR"

    def _format_v5_param(self, value: Optional[Union[Decimal, str, float, int]], param_type: Literal["price", "amount", "distance"] = "price", allow_zero: bool = False) -> Optional[str]:
        if value is None: return None
        decimal_value = safe_decimal(value, default=Decimal("NaN"))
        if decimal_value.is_nan(): logger.warning(f"V5 Param Formatting: Input '{value}' converted to NaN. Cannot format."); return None
        if decimal_value.is_zero():
            if allow_zero: return "0" # Bybit API often uses "0" to clear stops
            logger.debug(f"V5 Param Formatting: Input value '{value}' is zero, but zero not allowed for '{param_type}'."); return None
        if decimal_value < Decimal("0"): logger.warning(f"V5 Param Formatting: Input value '{value}' is negative ({decimal_value}), invalid for API params."); return None
        
        formatted_str: str
        if not self.exchange or not self.config.symbol: # Fallback if CCXT methods can't be used (e.g. exchange not fully ready)
            logger.warning("V5 Param Formatting: Exchange/symbol not fully available, using custom fallback formatters.")
            formatted_str = self.format_price(decimal_value) if param_type in ["price", "distance"] else self.format_amount(decimal_value, ROUND_DOWN)
        else:
            try:
                if param_type in ["price", "distance"]: formatted_str = self.exchange.price_to_precision(self.config.symbol, float(decimal_value))
                else: formatted_str = self.exchange.amount_to_precision(self.config.symbol, float(decimal_value)) # param_type == "amount"
                if safe_decimal(formatted_str).is_nan(): raise ValueError("CCXT formatting resulted in NaN")
            except Exception as e_ccxt_format:
                logger.warning(f"CCXT {param_type}_to_precision failed ({e_ccxt_format}), falling back to custom format for V5 param.")
                formatted_str = self.format_price(decimal_value) if param_type in ["price", "distance"] else self.format_amount(decimal_value, ROUND_DOWN)
        
        if formatted_str in ["ERR", "NaN"] or safe_decimal(formatted_str).is_nan():
            logger.error(f"V5 Param Formatting: Failed to produce valid string for '{value}' (type: {param_type}). Formatter returned: {formatted_str}"); return None
        return formatted_str

    def fetch_ohlcv(self) -> Optional[pd.DataFrame]:
        if not self.exchange: logger.error("Exchange not initialized, cannot fetch OHLCV."); return None
        logger.debug(f"Fetching up to {self.config.ohlcv_limit} OHLCV candles for {self.config.symbol} (Timeframe: {self.config.interval})...")
        try:
            ohlcv_data = fetch_with_retries(self.exchange.fetch_ohlcv, symbol=self.config.symbol, timeframe=self.config.interval, limit=self.config.ohlcv_limit, max_retries=self.config.max_fetch_retries, delay_seconds=self.config.retry_delay_seconds)
            if not ohlcv_data: logger.error(f"fetch_ohlcv for {self.config.symbol} returned no data (empty list)."); return None
            if len(ohlcv_data) < 20: logger.warning(f"Fetched only {len(ohlcv_data)} candles. May be insufficient for longer lookbacks.")
            df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True); df.set_index("timestamp", inplace=True)
            for col in ["open", "high", "low", "close", "volume"]: # volume is crucial for VWMA
                df[col] = df[col].apply(safe_decimal)
            initial_len = len(df)
            for col in ["open", "high", "low", "close", "volume"]: df = df[~df[col].apply(lambda x: isinstance(x, Decimal) and x.is_nan())] # Drop rows if any of OHLCV is NaN
            if len(df) < initial_len: logger.warning(f"Dropped {initial_len - len(df)} rows from OHLCV data due to NaN values in O/H/L/C/V columns.")
            if df.empty: logger.error("OHLCV DataFrame is empty after processing (NaN drop or initial empty)."); return None
            logger.debug(f"Fetched and processed {len(df)} OHLCV candles. Last timestamp: {df.index[-1] if not df.empty else 'N/A'}")
            return df
        except Exception as e: logger.error(f"Failed to fetch or process OHLCV data for {self.config.symbol}: {e}", exc_info=True); return None

    def get_balance(self) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        if not self.exchange or not self.market_info: logger.error("Exchange or market info not available, cannot fetch balance."); return None, None
        settle_currency = self.market_info.get("settle")
        if not settle_currency: logger.error("Settle currency not found in market info. Cannot determine balance currency."); return None, None
        logger.debug(f"Fetching balance for {settle_currency} (Account: {V5_UNIFIED_ACCOUNT_TYPE}, Category: {self.config.bybit_v5_category})...")
        try:
            balance_data = fetch_with_retries(self.exchange.fetch_balance, params={"accountType": V5_UNIFIED_ACCOUNT_TYPE, "coin": settle_currency}, max_retries=self.config.max_fetch_retries, delay_seconds=self.config.retry_delay_seconds)
            total_equity, available_balance = Decimal("NaN"), Decimal("NaN")
            if "info" in balance_data and "result" in balance_data["info"] and "list" in balance_data["info"]["result"]:
                account_list = balance_data["info"]["result"]["list"]
                if account_list and isinstance(account_list, list):
                    unified_acc_info = next((item for item in account_list if item.get("accountType") == V5_UNIFIED_ACCOUNT_TYPE), None)
                    if unified_acc_info:
                        total_equity = safe_decimal(unified_acc_info.get("totalEquity"))
                        coin_details_list = unified_acc_info.get("coin", [])
                        if coin_details_list and isinstance(coin_details_list, list):
                            settle_coin_info = next((c for c in coin_details_list if c.get("coin") == settle_currency), None)
                            if settle_coin_info:
                                available_balance = safe_decimal(settle_coin_info.get("availableToWithdraw"))
                                if total_equity.is_nan() and settle_coin_info.get("equity") is not None: total_equity = safe_decimal(settle_coin_info.get("equity"))
                        if available_balance.is_nan() and unified_acc_info.get("totalAvailableBalance") is not None:
                            available_balance = safe_decimal(unified_acc_info.get("totalAvailableBalance")); logger.debug(f"Used 'totalAvailableBalance' for {settle_currency} as coin-specific 'availableToWithdraw' was not found/parsed.")
            if total_equity.is_nan() and balance_data.get("total", {}).get(settle_currency) is not None: total_equity = safe_decimal(balance_data["total"][settle_currency]); logger.debug("Used CCXT standardized 'total' balance field as fallback for equity.")
            if available_balance.is_nan() and balance_data.get("free", {}).get(settle_currency) is not None: available_balance = safe_decimal(balance_data["free"][settle_currency]); logger.debug("Used CCXT standardized 'free' balance field as fallback for available.")
            if total_equity.is_nan(): logger.error(f"Could not extract valid total equity for {settle_currency}. Raw 'info.result.list' snippet: {str(balance_data.get('info',{}).get('result',{}).get('list',[{}])[0])[:300]}"); return None, available_balance if not available_balance.is_nan() else Decimal("0")
            if available_balance.is_nan(): logger.warning(f"Could not extract valid available balance for {settle_currency}. Defaulting to 0."); available_balance = Decimal("0")
            logger.debug(f"Balance Fetched ({settle_currency}): Total Equity = {total_equity.normalize()}, Available Balance = {available_balance.normalize()}")
            return total_equity, available_balance
        except Exception as e: logger.error(f"Failed to fetch or parse balance: {e}", exc_info=True); return None, None

    def get_current_position(self) -> Optional[Dict[str, Dict[str, Any]]]:
        if not self.exchange or not self.market_info: logger.error("Exchange or market info not available, cannot fetch position."); return None
        market_id = self.market_info.get("id"); positions_summary: Dict[str, Dict[str, Any]] = {"long": {}, "short": {}}
        if not market_id: logger.error("Market ID not found in market info. Cannot fetch position."); return None
        logger.debug(f"Fetching position for {self.config.symbol} (API ID: {market_id}, Category: {self.config.bybit_v5_category}, Target PositionIdx: {self.config.position_idx})...")
        try:
            fetched_positions_list_ccxt = fetch_with_retries(self.exchange.fetch_positions, symbols=[self.config.symbol], params={"category": self.config.bybit_v5_category, "symbol": market_id}, max_retries=self.config.max_fetch_retries, delay_seconds=self.config.retry_delay_seconds)
            if not fetched_positions_list_ccxt: logger.debug("No position data returned from fetch_positions. Assuming flat."); return positions_summary
            target_pos_info_raw_api = None
            for pos_data_ccxt_unified in fetched_positions_list_ccxt:
                raw_api_entry = pos_data_ccxt_unified.get("info", {}); pos_idx_from_api_str = raw_api_entry.get("positionIdx")
                try: pos_idx_from_api_int = int(pos_idx_from_api_str) if pos_idx_from_api_str is not None else -1
                except ValueError: logger.warning(f"Could not parse positionIdx '{pos_idx_from_api_str}'. Skipping."); continue
                if pos_idx_from_api_int == self.config.position_idx: target_pos_info_raw_api = raw_api_entry; logger.debug(f"Found position entry matching configured positionIdx={self.config.position_idx}: {target_pos_info_raw_api}"); break
            if not target_pos_info_raw_api: logger.debug(f"No position entry found for Idx={self.config.position_idx}, symbol {market_id}. Assuming flat."); return positions_summary
            
            qty_abs = safe_decimal(target_pos_info_raw_api.get("size", "0")).copy_abs()
            if qty_abs < POSITION_QTY_EPSILON: logger.debug(f"Position size {qty_abs.normalize()} for Idx {self.config.position_idx} negligible. Considered flat."); return positions_summary
            
            api_side_str = target_pos_info_raw_api.get("side", "None").lower(); position_side_key: Optional[str] = None
            if self.config.position_idx == 0: # One-Way Mode
                if api_side_str == "buy": position_side_key = "long"
                elif api_side_str == "sell": position_side_key = "short"
                elif api_side_str == "none" and qty_abs > Decimal("0"): logger.warning(f"Inconsistent state for One-Way (Idx 0): API side 'None' but size {qty_abs.normalize()}."); return positions_summary
            elif self.config.position_idx == 1: # Hedge Mode - Buy
                position_side_key = "long"; 
                if api_side_str != "buy" and qty_abs > Decimal("0"): logger.warning(f"Hedge Buy (Idx 1) API side '{api_side_str}' (not 'Buy') with size {qty_abs.normalize()}. Assuming long.")
            elif self.config.position_idx == 2: # Hedge Mode - Sell
                position_side_key = "short";
                if api_side_str != "sell" and qty_abs > Decimal("0"): logger.warning(f"Hedge Sell (Idx 2) API side '{api_side_str}' (not 'Sell') with size {qty_abs.normalize()}. Assuming short.")

            if position_side_key:
                entry_price = safe_decimal(target_pos_info_raw_api.get("avgPrice", "0"))
                sl_price_api = safe_decimal(target_pos_info_raw_api.get("stopLoss", "0"))
                tp_price_api = safe_decimal(target_pos_info_raw_api.get("takeProfit", "0"))
                tsl_trigger_price_api = safe_decimal(target_pos_info_raw_api.get("trailingStop", "0"))
                positions_summary[position_side_key] = {
                    "qty": qty_abs, "entry_price": entry_price if not entry_price.is_nan() and entry_price > Decimal("0") else Decimal("NaN"),
                    "liq_price": safe_decimal(target_pos_info_raw_api.get("liqPrice", "0")), "unrealized_pnl": safe_decimal(target_pos_info_raw_api.get("unrealisedPnl", "0")),
                    "api_side": api_side_str, "info": target_pos_info_raw_api,
                    "stop_loss_price": sl_price_api if not sl_price_api.is_nan() and sl_price_api > Decimal("0") else None,
                    "take_profit_price": tp_price_api if not tp_price_api.is_nan() and tp_price_api > Decimal("0") else None,
                    "is_tsl_active": not tsl_trigger_price_api.is_nan() and tsl_trigger_price_api > Decimal("0"),
                    "tsl_trigger_price": tsl_trigger_price_api if not tsl_trigger_price_api.is_nan() and tsl_trigger_price_api > Decimal("0") else None,
                }
                entry_str = positions_summary[position_side_key]["entry_price"].normalize() if positions_summary[position_side_key]["entry_price"] and not positions_summary[position_side_key]["entry_price"].is_nan() else "N/A"
                logger.debug(f"Identified {position_side_key.upper()} position (Idx {self.config.position_idx}): Qty={qty_abs.normalize()}, Entry={entry_str}")
            else: logger.warning(f"Position size {qty_abs.normalize()} for Idx {self.config.position_idx} but no long/short map (api_side: '{api_side_str}'). Flat."); return positions_summary
            return positions_summary
        except Exception as e: logger.error(f"Failed to fetch or parse positions for {self.config.symbol}: {e}", exc_info=True); return None

# --- Indicator Calculator Class ---
class IndicatorCalculator:
    def __init__(self, config: TradingConfig): self.config = config
    def calculate_indicators(self, df: pd.DataFrame) -> Optional[Dict[str, Union[Decimal, bool, int]]]:
        logger.info(f"{Fore.CYAN}# Weaving indicator patterns (Original & VT)...{Style.RESET_ALL}")
        if df is None or df.empty: logger.error(f"{Fore.RED}No DataFrame provided for indicator calculation.{Style.RESET_ALL}"); return None
        required_ohlc_cols = ["open", "high", "low", "close", "volume"] # volume is needed for VWMA
        if not all(c in df.columns for c in required_ohlc_cols):
            missing_cols = [c for c in required_ohlc_cols if c not in df.columns]
            logger.error(f"{Fore.RED}DataFrame missing required columns for indicators: {missing_cols}{Style.RESET_ALL}"); return None
        try:
            df_calc = df[required_ohlc_cols].copy()
            def safe_to_float(x: Any) -> float:
                if isinstance(x, (float, int)): return float(x)
                if isinstance(x, Decimal): return float('nan') if x.is_nan() else float(x)
                if isinstance(x, str):
                    try: val_stripped = x.strip().lower(); return float('nan') if val_stripped in ["nan", "none", "null", ""] else float(val_stripped)
                    except ValueError: return float('nan')
                return float('nan') if x is None else float('nan') # Default for None or other types
            for col in required_ohlc_cols:
                if df_calc[col].empty: df_calc[col] = pd.Series(dtype=float); continue
                df_calc[col] = df_calc[col].apply(safe_to_float); df_calc[col] = df_calc[col].astype(float)
            
            initial_len = len(df_calc); df_calc.dropna(subset=required_ohlc_cols, inplace=True, how='any')
            if len(df_calc) < initial_len: logger.debug(f"Dropped {initial_len - len(df_calc)} rows with NaN in OHLCV columns after float conversion for TA.")
            if df_calc.empty: logger.error(f"{Fore.RED}DataFrame became empty after NaN drop during indicator pre-processing.{Style.RESET_ALL}"); return None

            # Determine max period needed for all indicators (Original + VT if enabled)
            max_period_orig = max(self.config.slow_ema_period, self.config.trend_ema_period, self.config.stoch_period + self.config.stoch_smooth_k + self.config.stoch_smooth_d, self.config.atr_period, self.config.adx_period * 2)
            max_period_vt = 0
            if self.config.vt_enable:
                max_period_vt = max(self.config.vt_trend_ema_period, self.config.vt_vwma_period, self.config.vt_volume_spike_lookback)
            max_period_needed = max(max_period_orig, max_period_vt)
            min_required_data_length = max_period_needed + 20 # Buffer
            if len(df_calc) < min_required_data_length: logger.error(f"{Fore.RED}Insufficient data ({len(df_calc)} rows) for robust indicator calculation (requires ~{min_required_data_length} rows).{Style.RESET_ALL}"); return None

            close_s, high_s, low_s, volume_s, open_s = df_calc["close"], df_calc["high"], df_calc["low"], df_calc["volume"], df_calc["open"]

            # --- Original Strategy Indicators ---
            fast_ema_s = close_s.ewm(span=self.config.fast_ema_period, adjust=False).mean()
            slow_ema_s = close_s.ewm(span=self.config.slow_ema_period, adjust=False).mean()
            trend_ema_s = close_s.ewm(span=self.config.trend_ema_period, adjust=False).mean()
            low_min_stoch = low_s.rolling(window=self.config.stoch_period, min_periods=max(1, self.config.stoch_period // 2)).min()
            high_max_stoch = high_s.rolling(window=self.config.stoch_period, min_periods=max(1, self.config.stoch_period // 2)).max()
            stoch_range = high_max_stoch - low_min_stoch
            stoch_k_raw_values = np.where(stoch_range > 1e-12, 100 * (close_s - low_min_stoch) / stoch_range, 50.0)
            stoch_k_raw_s = pd.Series(stoch_k_raw_values, index=df_calc.index).fillna(50)
            stoch_k_s = stoch_k_raw_s.rolling(window=self.config.stoch_smooth_k, min_periods=1).mean().fillna(50)
            stoch_d_s = stoch_k_s.rolling(window=self.config.stoch_smooth_d, min_periods=1).mean().fillna(50)
            true_range_s = pd.concat([high_s - low_s, (high_s - close_s.shift(1)).abs(), (low_s - close_s.shift(1)).abs()], axis=1).max(axis=1).fillna(0)
            atr_s = true_range_s.ewm(span=self.config.atr_period, adjust=False).mean()
            adx_s, pdi_s, mdi_s = self._calculate_adx(high_s, low_s, close_s, atr_s, self.config.adx_period)

            # --- VolumaticTrend Strategy Indicators (if enabled) ---
            vt_trend_ema_s_series = pd.Series(np.nan, index=df_calc.index) # Initialize with NaNs
            vt_vwma_s_series = pd.Series(np.nan, index=df_calc.index)
            vt_volume_avg_s_series = pd.Series(np.nan, index=df_calc.index)
            vt_is_volume_spike_s_series = pd.Series(False, index=df_calc.index) # Default to False

            if self.config.vt_enable:
                vt_trend_ema_s_series = close_s.ewm(span=self.config.vt_trend_ema_period, adjust=False).mean()
                # Use pandas_ta for VWMA
                if hasattr(df_calc.ta, 'vwma'): # Check if DataFrame has 'ta' accessor and 'vwma' method
                    vwma_result = df_calc.ta.vwma(length=self.config.vt_vwma_period, fillna=np.nan) # Specify fillna for clarity
                    if isinstance(vwma_result, pd.Series):
                         vt_vwma_s_series = vwma_result
                    else: # Should not happen if pandas_ta is working
                         logger.warning("pandas_ta.vwma did not return a Series. VWMA will be NaN.")
                else: # Fallback if pandas_ta.vwma is not available (very unlikely if import succeeded)
                    logger.error("pandas_ta.vwma method not found on DataFrame. VWMA calculation skipped (will be NaN). Install/reinstall pandas_ta if issues persist.")
                
                vt_volume_avg_s_series = volume_s.rolling(window=self.config.vt_volume_spike_lookback, min_periods=max(1, self.config.vt_volume_spike_lookback // 2)).mean()
                # Volume spike condition: current volume > average volume * multiplier, and average volume is positive
                # volume_s and vt_volume_avg_s_series are float Series. self.config.vt_volume_spike_multiplier is Decimal. Convert to float for comparison.
                vt_is_volume_spike_s_series = (volume_s > vt_volume_avg_s_series * float(self.config.vt_volume_spike_multiplier)) & (vt_volume_avg_s_series > 0)

            def get_latest_decimal_from_series(series: pd.Series, indicator_name: str) -> Decimal:
                valid_series = series.dropna(); return safe_decimal(str(valid_series.iloc[-1])) if not valid_series.empty else (logger.warning(f"Indicator series '{indicator_name}' empty/all NaNs."), Decimal("NaN"))[1]
            def get_latest_bool_from_series(series: pd.Series, indicator_name: str) -> bool:
                valid_series = series.dropna(); return bool(valid_series.iloc[-1]) if not valid_series.empty else (logger.warning(f"Boolean indicator series '{indicator_name}' empty/all NaNs."), False)[1]

            indicators_out: Dict[str, Union[Decimal, bool, int]] = {
                "fast_ema": get_latest_decimal_from_series(fast_ema_s, "fast_ema"), "slow_ema": get_latest_decimal_from_series(slow_ema_s, "slow_ema"),
                "trend_ema": get_latest_decimal_from_series(trend_ema_s, "trend_ema"), "stoch_k": get_latest_decimal_from_series(stoch_k_s, "stoch_k"),
                "stoch_d": get_latest_decimal_from_series(stoch_d_s, "stoch_d"), "atr": get_latest_decimal_from_series(atr_s, "atr"),
                "atr_period": self.config.atr_period, "adx": get_latest_decimal_from_series(adx_s, "adx"),
                "pdi": get_latest_decimal_from_series(pdi_s, "pdi"), "mdi": get_latest_decimal_from_series(mdi_s, "mdi"),
                # VT Indicators
                "vt_trend_ema": get_latest_decimal_from_series(vt_trend_ema_s_series, "vt_trend_ema"),
                "vt_vwma": get_latest_decimal_from_series(vt_vwma_s_series, "vt_vwma"),
                "vt_volume_avg": get_latest_decimal_from_series(vt_volume_avg_s_series, "vt_volume_avg"), # For display/debug
                "vt_is_volume_spike": get_latest_bool_from_series(vt_is_volume_spike_s_series, "vt_is_volume_spike"),
                "vt_candle_is_green": (close_s.iloc[-1] > open_s.iloc[-1]) if not close_s.empty and not open_s.empty else False,
                "vt_candle_is_red": (close_s.iloc[-1] < open_s.iloc[-1]) if not close_s.empty and not open_s.empty else False,
            }
            
            stoch_k_valid_series = stoch_k_s.dropna(); indicators_out["stoch_k_prev"] = get_latest_decimal_from_series(stoch_k_valid_series.iloc[:-1] if len(stoch_k_valid_series) >= 2 else pd.Series(dtype=float), "stoch_k_prev")
            stoch_d_valid_series = stoch_d_s.dropna(); d_prev_val = get_latest_decimal_from_series(stoch_d_valid_series.iloc[:-1] if len(stoch_d_valid_series) >= 2 else pd.Series(dtype=float), "stoch_d_prev")
            k_now, d_now, k_prev = indicators_out["stoch_k"], indicators_out["stoch_d"], indicators_out["stoch_k_prev"] # type: ignore
            indicators_out["stoch_kd_bullish"], indicators_out["stoch_kd_bearish"] = False, False
            if not any(v.is_nan() for v in [k_now, d_now, k_prev, d_prev_val]): # type: ignore
                if (k_prev <= d_prev_val) and (k_now > d_now): indicators_out["stoch_kd_bullish"] = True # type: ignore
                if (k_prev >= d_prev_val) and (k_now < d_now): indicators_out["stoch_kd_bearish"] = True # type: ignore

            critical_keys = ["fast_ema", "slow_ema", "trend_ema", "atr", "stoch_k", "stoch_d", "adx", "pdi", "mdi"]
            if self.config.vt_enable: critical_keys.extend(["vt_trend_ema", "vt_vwma"]) # vt_volume_avg and vt_is_volume_spike are derived and might be NaN if volume is zero, less critical for core logic failure.
            failed_indicators = [k for k in critical_keys if indicators_out.get(k, Decimal("NaN")).is_nan()] # type: ignore
            if failed_indicators:
                if indicators_out.get("atr", Decimal("NaN")).is_nan(): logger.error(f"{Fore.RED}CRITICAL: ATR calculated as NaN. Risk calculations will fail. Aborting.{Style.RESET_ALL}"); return None # type: ignore
                logger.warning(f"{Fore.YELLOW}Warning: Some critical indicators NaN: {', '.join(failed_indicators)}. May impair signals.{Style.RESET_ALL}")
            
            logger.info(f"{Style.BRIGHT}{Fore.GREEN}Indicator patterns woven successfully.{Style.RESET_ALL}"); return indicators_out
        except Exception as e: logger.error(f"{Fore.RED}Error weaving indicator patterns: {e}{Style.RESET_ALL}", exc_info=True); return None

    def _calculate_adx(self, high_s: pd.Series, low_s: pd.Series, close_s: pd.Series, atr_s: pd.Series, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        if period <= 0: logger.error("ADX period must be a positive integer."); nan_series = pd.Series(np.nan, index=high_s.index); return nan_series, nan_series, nan_series
        if atr_s.empty or atr_s.isnull().all(): logger.error("ATR series is empty or all NaN. Cannot calculate ADX components."); nan_series = pd.Series(np.nan, index=high_s.index); return nan_series, nan_series, nan_series
        move_up, move_down = high_s.diff(), -low_s.diff()
        plus_dm_values = np.where((move_up > move_down) & (move_up > 0), move_up, 0.0)
        minus_dm_values = np.where((move_down > move_up) & (move_down > 0), move_down, 0.0)
        plus_dm_s, minus_dm_s = pd.Series(plus_dm_values, index=high_s.index).fillna(0), pd.Series(minus_dm_values, index=high_s.index).fillna(0)
        alpha = 1.0 / period
        smoothed_plus_dm_s = plus_dm_s.ewm(alpha=alpha, adjust=False, min_periods=period).mean().fillna(0)
        smoothed_minus_dm_s = minus_dm_s.ewm(alpha=alpha, adjust=False, min_periods=period).mean().fillna(0)
        pdi_values = np.where((atr_s > 1e-12) & (~atr_s.isnull()), 100 * smoothed_plus_dm_s / atr_s, 0.0)
        mdi_values = np.where((atr_s > 1e-12) & (~atr_s.isnull()), 100 * smoothed_minus_dm_s / atr_s, 0.0)
        pdi_s_out, mdi_s_out = pd.Series(pdi_values, index=high_s.index).fillna(0), pd.Series(mdi_values, index=high_s.index).fillna(0)
        di_sum = pdi_s_out + mdi_s_out
        dx_values = np.where(di_sum > 1e-12, 100 * (pdi_s_out - mdi_s_out).abs() / di_sum, 0.0)
        dx_s = pd.Series(dx_values, index=high_s.index).fillna(0)
        adx_s_out = dx_s.ewm(alpha=alpha, adjust=False, min_periods=period).mean().fillna(0)
        return adx_s_out, pdi_s_out, mdi_s_out

# --- Signal Generator Class ---
class SignalGenerator:
    def __init__(self, config: TradingConfig): self.config = config

    def _generate_original_signals(self, df_last_candles: pd.DataFrame, indicators: Dict[str, Union[Decimal, bool, int]], current_price: Decimal) -> Tuple[bool, bool, str]:
        """Calculates signals based on the original Pyrmethus strategy logic."""
        orig_long_sig, orig_short_sig = False, False
        reason = "Original: Initializing"
        
        prev_close = safe_decimal(df_last_candles.iloc[-2]["close"]) # From 2nd to last candle
            
        required_indicator_keys = ["stoch_k", "fast_ema", "slow_ema", "trend_ema", "atr", "adx", "pdi", "mdi"]
        ind_values: Dict[str, Decimal] = {}
        nan_keys = [k for k in required_indicator_keys if not isinstance(indicators.get(k), Decimal) or indicators.get(k, Decimal("NaN")).is_nan()] # type: ignore[union-attr]
        if nan_keys:
            reason = f"Original: Indicators NaN/Missing: {', '.join(nan_keys)}."
            return orig_long_sig, orig_short_sig, reason
        for key in required_indicator_keys: ind_values[key] = indicators[key] # type: ignore[assignment]

        k, fast_ema, slow_ema, trend_ema, atr, adx, pdi, mdi = (ind_values[key] for key in required_indicator_keys)
        stoch_kd_bull_cross, stoch_kd_bear_cross = bool(indicators.get("stoch_kd_bullish", False)), bool(indicators.get("stoch_kd_bearish", False))

        ema_bullish_cross, ema_bearish_cross = fast_ema > slow_ema, fast_ema < slow_ema
        ema_cross_state = "Bullish" if ema_bullish_cross else "Bearish" if ema_bearish_cross else "Neutral"
        
        trend_buffer_abs = trend_ema.copy_abs() * (self.config.trend_filter_buffer_percent / 100)
        price_above_trend_ema_for_long = current_price > (trend_ema - trend_buffer_abs)
        price_below_trend_ema_for_short = current_price < (trend_ema + trend_buffer_abs)
        trend_allows_long = price_above_trend_ema_for_long if self.config.trade_only_with_trend else True
        trend_allows_short = price_below_trend_ema_for_short if self.config.trade_only_with_trend else True
        trend_reason_suffix = f"(P:{current_price:.{DEFAULT_PRICE_DP}f} vs TrendEMA:{trend_ema:.{DEFAULT_PRICE_DP}f} ±{trend_buffer_abs:.{DEFAULT_PRICE_DP}f})" if self.config.trade_only_with_trend else "(TrendFilter OFF)"

        stoch_long_entry_cond = (k < self.config.stoch_oversold_threshold) or stoch_kd_bull_cross
        stoch_short_entry_cond = (k > self.config.stoch_overbought_threshold) or stoch_kd_bear_cross
        stoch_state_reason = f"K:{k:.1f} (OS:{self.config.stoch_oversold_threshold.normalize()}/OB:{self.config.stoch_overbought_threshold.normalize()}) KD_Cross(Bull:{stoch_kd_bull_cross}/Bear:{stoch_kd_bear_cross})"

        significant_price_move, atr_filter_reason_suffix = True, "(ATR MoveFilter OFF)"
        if self.config.atr_move_filter_multiplier > Decimal("0"):
            if atr.is_nan() or atr <= Decimal("0"): atr_filter_reason_suffix, significant_price_move = f"(ATR Filter Skipped: Invalid ATR {atr.normalize() if not atr.is_nan() else 'NaN'})", False
            elif prev_close.is_nan() or prev_close <= Decimal("0"): atr_filter_reason_suffix, significant_price_move = f"(ATR Filter Skipped: Invalid Previous Close {prev_close.normalize() if not prev_close.is_nan() else 'NaN'})", False
            else:
                atr_move_threshold_abs = atr * self.config.atr_move_filter_multiplier
                price_move_abs = (current_price - prev_close).copy_abs()
                significant_price_move = price_move_abs > atr_move_threshold_abs
                atr_filter_reason_suffix = f"(Move:{price_move_abs:.{DEFAULT_PRICE_DP}f} {'OK' if significant_price_move else 'LOW'} vs Thr:{atr_move_threshold_abs:.{DEFAULT_PRICE_DP}f})"
        
        adx_is_trending_strong = adx > self.config.min_adx_level
        adx_long_direction_favored, adx_short_direction_favored = pdi > mdi, mdi > pdi
        adx_allows_long, adx_allows_short = adx_is_trending_strong and adx_long_direction_favored, adx_is_trending_strong and adx_short_direction_favored
        adx_filter_reason_suffix = f"(ADX:{adx:.1f} {'STRONG' if adx_is_trending_strong else 'WEAK'} vs Min:{self.config.min_adx_level.normalize()} | Dir: {'PDI>MDI' if adx_long_direction_favored else 'MDI>PDI' if adx_short_direction_favored else 'Neutral'})"

        base_long_signal_met, base_short_signal_met = ema_bullish_cross and stoch_long_entry_cond, ema_bearish_cross and stoch_short_entry_cond
        orig_long_sig, orig_short_sig = base_long_signal_met and trend_allows_long and significant_price_move and adx_allows_long, base_short_signal_met and trend_allows_short and significant_price_move and adx_allows_short

        if orig_long_sig: reason = f"Original Long: EMA_X {ema_cross_state} & StochOK {stoch_state_reason} & TrendOK {trend_reason_suffix} & ATRMoveOK {atr_filter_reason_suffix} & ADX_OK {adx_filter_reason_suffix}"
        elif orig_short_sig: reason = f"Original Short: EMA_X {ema_cross_state} & StochOK {stoch_state_reason} & TrendOK {trend_reason_suffix} & ATRMoveOK {atr_filter_reason_suffix} & ADX_OK {adx_filter_reason_suffix}"
        else:
            reason_parts = ["Original NoSignal:"]
            reason_parts.append(f"Base(EMA_X:{ema_cross_state},Stoch:{stoch_state_reason}) -> LongBase:{base_long_signal_met}, ShortBase:{base_short_signal_met}.")
            if base_long_signal_met or base_short_signal_met:
                if not trend_allows_long and base_long_signal_met: reason_parts.append(f"Long Blocked: TrendFail {trend_reason_suffix}.")
                if not trend_allows_short and base_short_signal_met: reason_parts.append(f"Short Blocked: TrendFail {trend_reason_suffix}.")
                if not significant_price_move and (base_long_signal_met or base_short_signal_met) : reason_parts.append(f"Blocked: ATRMoveFail {atr_filter_reason_suffix}.")
                if not adx_allows_long and base_long_signal_met: reason_parts.append(f"Long Blocked: ADXFail {adx_filter_reason_suffix}.")
                if not adx_allows_short and base_short_signal_met: reason_parts.append(f"Short Blocked: ADXFail {adx_filter_reason_suffix}.")
            reason = " ".join(reason_parts)
        return orig_long_sig, orig_short_sig, reason

    def _generate_vt_signals(self, indicators: Dict[str, Union[Decimal, bool, int]], current_price: Decimal) -> Tuple[bool, bool, str]:
        """Calculates signals based on the VolumaticTrend strategy logic."""
        vt_long_sig, vt_short_sig = False, False
        reason = "VT: Initializing"

        if not self.config.vt_enable:
            return False, False, "VT: Disabled by config"

        required_vt_keys = ["vt_trend_ema", "vt_vwma"] # vt_is_volume_spike, vt_candle_is_green/red are bools
        ind_values_vt: Dict[str, Decimal] = {}
        nan_keys_vt = [k for k in required_vt_keys if not isinstance(indicators.get(k), Decimal) or indicators.get(k, Decimal("NaN")).is_nan()] # type: ignore[union-attr]
        
        # Boolean indicators from the main indicators dict
        is_volume_spike = indicators.get("vt_is_volume_spike", False)
        is_green_candle = indicators.get("vt_candle_is_green", False)
        is_red_candle = indicators.get("vt_candle_is_red", False)

        if not isinstance(is_volume_spike, bool) or \
           not isinstance(is_green_candle, bool) or \
           not isinstance(is_red_candle, bool):
             nan_keys_vt.append("vt_boolean_indicators_invalid_type_or_missing")


        if nan_keys_vt:
            reason = f"VT: Indicators NaN/Missing or Invalid Type: {', '.join(nan_keys_vt)}."
            return vt_long_sig, vt_short_sig, reason
        
        for key_vt in required_vt_keys: ind_values_vt[key_vt] = indicators[key_vt] # type: ignore[assignment]
        vt_trend_ema, vt_vwma = ind_values_vt["vt_trend_ema"], ind_values_vt["vt_vwma"]
        
        # VT Long Conditions
        price_above_vt_trend_ema = current_price > vt_trend_ema
        price_above_vwma = current_price > vt_vwma
        if price_above_vt_trend_ema and price_above_vwma and is_volume_spike and is_green_candle:
            vt_long_sig = True
            reason = (f"VT Long: Price>{vt_trend_ema:.{DEFAULT_PRICE_DP}f}(TrendEMA) & "
                      f"Price>{vt_vwma:.{DEFAULT_PRICE_DP}f}(VWMA) & VolSpike & GreenCandle")
        
        # VT Short Conditions (only if no long signal)
        price_below_vt_trend_ema = current_price < vt_trend_ema
        price_below_vwma = current_price < vt_vwma
        if not vt_long_sig and price_below_vt_trend_ema and price_below_vwma and is_volume_spike and is_red_candle:
            vt_short_sig = True
            reason = (f"VT Short: Price<{vt_trend_ema:.{DEFAULT_PRICE_DP}f}(TrendEMA) & "
                      f"Price<{vt_vwma:.{DEFAULT_PRICE_DP}f}(VWMA) & VolSpike & RedCandle")

        if not vt_long_sig and not vt_short_sig: # No VT signal, provide context
            trend_state_vt = "UP" if price_above_vt_trend_ema else "DOWN" if price_below_vt_trend_ema else "NEUTRAL"
            vwma_state_vt = "ABOVE" if price_above_vwma else "BELOW" if price_below_vwma else "NEUTRAL"
            candle_color_vt = "Green" if is_green_candle else "Red" if is_red_candle else "Neutral"
            reason = (f"VT NoSignal: Trend(P:{current_price:.{DEFAULT_PRICE_DP}f} vs EMA:{vt_trend_ema:.{DEFAULT_PRICE_DP}f} -> {trend_state_vt}), "
                      f"VWMA(P:{current_price:.{DEFAULT_PRICE_DP}f} vs VWMA:{vt_vwma:.{DEFAULT_PRICE_DP}f} -> {vwma_state_vt}), "
                      f"VolSpike:{is_volume_spike}, Candle:{candle_color_vt}")
        
        return vt_long_sig, vt_short_sig, reason


    def generate_signals(self, df_last_candles: pd.DataFrame, indicators: Dict[str, Union[Decimal, bool, int]]) -> Dict[str, Union[bool, str]]:
        result: Dict[str, Union[bool, str]] = {"long": False, "short": False, "reason": "Initializing signal check"}
        if not indicators:
            result["reason"] = "No Signal: Indicators data missing."; logger.debug(result["reason"]); return result
        if df_last_candles is None or len(df_last_candles) < 2: 
            reason_no_candle = f"No Signal: Insufficient candle data (requires >=2 for ATR move filter, got {len(df_last_candles) if df_last_candles is not None else 0})."
            result["reason"] = reason_no_candle; logger.debug(reason_no_candle); return result
        try:
            current_price = safe_decimal(df_last_candles.iloc[-1]["close"])
            if current_price.is_nan() or current_price <= Decimal("0"):
                result["reason"] = f"No Signal: Invalid current price ({current_price.normalize() if not current_price.is_nan() else 'NaN'})."; logger.warning(result["reason"]); return result
            
            # Generate signals from original strategy
            orig_long, orig_short, orig_reason = self._generate_original_signals(df_last_candles, indicators, current_price)

            # Generate signals from VolumaticTrend strategy (if enabled)
            vt_long, vt_short, vt_reason = False, False, "VT: Strategy Disabled"
            if self.config.vt_enable:
                vt_long, vt_short, vt_reason = self._generate_vt_signals(indicators, current_price)

            # Combine signals: Prioritize agreement, hold on conflict, take single signal.
            final_long = False
            final_short = False
            combined_reason_parts = []

            if orig_long and vt_long: # Both signal long
                final_long = True; combined_reason_parts.append(f"Orig&VT Long: ({orig_reason}) AND ({vt_reason})")
            elif orig_short and vt_short: # Both signal short
                final_short = True; combined_reason_parts.append(f"Orig&VT Short: ({orig_reason}) AND ({vt_reason})")
            elif (orig_long and vt_short) or (orig_short and vt_long): # Conflict
                combined_reason_parts.append(f"Signal Conflict: Orig ({'L' if orig_long else 'S' if orig_short else 'N'}) vs VT ({'L' if vt_long else 'S' if vt_short else 'N'}). Holding.")
                combined_reason_parts.append(f"OrigReason: {orig_reason}"); combined_reason_parts.append(f"VTReason: {vt_reason}")
            elif orig_long: # Only Original signals long
                final_long = True; combined_reason_parts.append(f"Orig Long: {orig_reason} (VT: {vt_reason})")
            elif orig_short: # Only Original signals short
                final_short = True; combined_reason_parts.append(f"Orig Short: {orig_reason} (VT: {vt_reason})")
            elif vt_long: # Only VT signals long
                final_long = True; combined_reason_parts.append(f"VT Long: {vt_reason} (Orig: {orig_reason})")
            elif vt_short: # Only VT signals short
                final_short = True; combined_reason_parts.append(f"VT Short: {vt_reason} (Orig: {orig_reason})")
            else: # No signal from either
                combined_reason_parts.append(f"No Signal. Orig: {orig_reason} | VT: {vt_reason}")
            
            result["long"] = final_long
            result["short"] = final_short
            result["reason"] = " | ".join(combined_reason_parts)


            log_level_for_signal = logging.INFO if result["long"] or result["short"] or "Blocked" in result["reason"] or "Conflict" in result["reason"] else logging.DEBUG
            logger.log(log_level_for_signal, f"Signal Check Result: {result['reason']}")

        except Exception as e: 
            logger.error(f"{Fore.RED}Error generating entry signals: {e}{Style.RESET_ALL}", exc_info=True)
            result.update({"reason": f"No Signal: Exception during generation ({type(e).__name__})", "long": False, "short": False})
        return result

    def check_exit_signals(self, position_side: str, indicators: Dict[str, Union[Decimal, bool, int]]) -> Optional[str]:
        if not indicators: logger.warning("Cannot check exit signals: indicators data missing."); return None
        
        # Original Strategy Exit Logic
        fast_ema_val, slow_ema_val = indicators.get("fast_ema"), indicators.get("slow_ema")
        stoch_k_current_val, stoch_k_previous_val = indicators.get("stoch_k"), indicators.get("stoch_k_prev")
        orig_exit_reason: Optional[str] = None
        
        required_for_orig_exit = {"fast_ema": fast_ema_val, "slow_ema": slow_ema_val, "stoch_k_current": stoch_k_current_val, "stoch_k_previous": stoch_k_previous_val}
        orig_inds_valid = all(isinstance(val, Decimal) and not val.is_nan() for val in required_for_orig_exit.values())
        
        if orig_inds_valid:
            fast_ema, slow_ema, stoch_k_current, stoch_k_previous = fast_ema_val, slow_ema_val, stoch_k_current_val, stoch_k_previous_val # type: ignore
            ema_is_bullish_crossed, ema_is_bearish_crossed = fast_ema > slow_ema, fast_ema < slow_ema
            oversold_level, overbought_level = self.config.stoch_oversold_threshold, self.config.stoch_overbought_threshold

            if position_side == "long":
                if ema_is_bearish_crossed: orig_exit_reason = f"Exit Signal (Long Orig): EMA Bearish Cross (F {fast_ema.normalize()} < S {slow_ema.normalize()})"
                elif stoch_k_previous >= overbought_level and stoch_k_current < overbought_level: orig_exit_reason = f"Exit Signal (L Orig): Stoch Reversal OB (PrK {stoch_k_previous.normalize():.1f}>=OB -> CurK {stoch_k_current.normalize():.1f}<OB)"
            elif position_side == "short":
                if ema_is_bullish_crossed: orig_exit_reason = f"Exit Signal (Short Orig): EMA Bullish Cross (F {fast_ema.normalize()} > S {slow_ema.normalize()})"
                elif stoch_k_previous <= oversold_level and stoch_k_current > oversold_level: orig_exit_reason = f"Exit Signal (S Orig): Stoch Reversal OS (PrK {stoch_k_previous.normalize():.1f}<=OS -> CurK {stoch_k_current.normalize():.1f}>OS)"
        
        if orig_exit_reason:
            logger.trade(f"{Fore.YELLOW}{orig_exit_reason}{Style.RESET_ALL}"); return orig_exit_reason

        # VolumaticTrend Exit Logic (if enabled and original didn't fire)
        if self.config.vt_enable:
            vt_exit_reason: Optional[str] = None
            current_price = indicators.get("close_price") # This should be the current market price
            vt_trend_ema_val = indicators.get("vt_trend_ema")
            vt_vwma_val = indicators.get("vt_vwma")
            is_volume_spike = indicators.get("vt_is_volume_spike", False) # Bool
            is_red_candle = indicators.get("vt_candle_is_red", False)     # Bool
            is_green_candle = indicators.get("vt_candle_is_green", False) # Bool

            required_for_vt_exit = {"current_price": current_price, "vt_trend_ema": vt_trend_ema_val, "vt_vwma": vt_vwma_val}
            vt_inds_valid = all(isinstance(val, Decimal) and not val.is_nan() for val in required_for_vt_exit.values())
            vt_inds_valid = vt_inds_valid and isinstance(is_volume_spike, bool) and isinstance(is_red_candle, bool) and isinstance(is_green_candle, bool)

            if vt_inds_valid:
                price, vt_trend_ema, vt_vwma = current_price, vt_trend_ema_val, vt_vwma_val # type: ignore
                if position_side == "long":
                    if price < vt_trend_ema: vt_exit_reason = f"Exit Signal (Long VT): Price < VT Trend EMA ({vt_trend_ema.normalize()})" # type: ignore
                    elif price < vt_vwma and is_volume_spike and is_red_candle: vt_exit_reason = f"Exit Signal (Long VT): Price < VT VWMA ({vt_vwma.normalize()}) + VolSpike on RedCandle" # type: ignore
                elif position_side == "short":
                    if price > vt_trend_ema: vt_exit_reason = f"Exit Signal (Short VT): Price > VT Trend EMA ({vt_trend_ema.normalize()})" # type: ignore
                    elif price > vt_vwma and is_volume_spike and is_green_candle: vt_exit_reason = f"Exit Signal (Short VT): Price > VT VWMA ({vt_vwma.normalize()}) + VolSpike on GreenCandle" # type: ignore
            
            if vt_exit_reason:
                logger.trade(f"{Fore.YELLOW}{vt_exit_reason}{Style.RESET_ALL}"); return vt_exit_reason
        
        return None # No exit signal from any strategy

# --- Order Manager Class (Structurally Unchanged, assumed robust from original Pyrmethus) ---
class OrderManager:
    def __init__(self, config: TradingConfig, exchange_manager: ExchangeManager):
        self.config = config
        self.exchange_manager = exchange_manager
        if not exchange_manager or not exchange_manager.exchange or not exchange_manager.market_info:
            err_msg = "OrderManager cannot initialize: Valid ExchangeManager instance with initialized exchange and loaded market_info is required."
            logger.critical(f"{Style.BRIGHT}{Fore.RED}{err_msg}{Style.RESET_ALL}"); raise ValueError(err_msg)
        self.exchange = exchange_manager.exchange
        self.market_info = exchange_manager.market_info
        self.protection_tracker: Dict[str, Optional[str]] = {"long": None, "short": None}

    def _calculate_trade_parameters(self, side: str, atr: Decimal, total_equity: Decimal, current_price: Decimal) -> Optional[Dict[str, Optional[Decimal]]]:
        # Input validation
        if atr.is_nan() or atr <= Decimal("0"): logger.error(f"Invalid ATR ({atr.normalize() if not atr.is_nan() else 'NaN'}) for trade parameter calculation."); return None
        if total_equity.is_nan() or total_equity <= Decimal("0"): logger.error(f"Invalid total equity ({total_equity.normalize() if not total_equity.is_nan() else 'NaN'}) for parameter calculation."); return None
        if current_price.is_nan() or current_price <= Decimal("0"): logger.error(f"Invalid current price ({current_price.normalize() if not current_price.is_nan() else 'NaN'}) for parameter calculation."); return None
        
        market_tick_size = self.market_info.get('tick_size', Decimal('NaN')) if self.market_info else Decimal('NaN')
        market_contract_size = self.market_info.get('contract_size', Decimal('NaN')) if self.market_info else Decimal('NaN')
        market_min_order_size = self.market_info.get('min_order_size', Decimal('NaN')) if self.market_info else Decimal('NaN')

        if not self.market_info or market_tick_size.is_nan() or market_contract_size.is_nan() or market_min_order_size.is_nan():
             logger.error("Market info (tick_size, contract_size, min_order_size) missing, NaN, or incomplete for parameter calculation."); return None
        if side not in ["buy", "sell"]: logger.error(f"Invalid side '{side}' specified for trade parameter calculation."); return None

        try:
            risk_amount_per_trade_settle_ccy = total_equity * self.config.risk_percentage
            sl_distance_atr_points = atr * self.config.sl_atr_multiplier
            sl_price_calculated = current_price - sl_distance_atr_points if side == "buy" else current_price + sl_distance_atr_points
            if sl_price_calculated <= Decimal("0"): logger.error(f"Calculated SL price ({sl_price_calculated:.{DEFAULT_PRICE_DP}f}) is invalid (<=0). Cannot proceed."); return None

            sl_distance_from_current_abs = (current_price - sl_price_calculated).copy_abs()
            
            if market_tick_size <= Decimal("0"): logger.error("Market tick_size is invalid or zero. Cannot validate SL distance."); return None
            if sl_distance_from_current_abs < market_tick_size: 
                logger.warning(f"Initial SL distance ({sl_distance_from_current_abs.normalize()}) < min tick size ({market_tick_size.normalize()}). Adjusting SL distance to min tick size."); sl_distance_from_current_abs = market_tick_size
                sl_price_calculated = current_price - sl_distance_from_current_abs if side == "buy" else current_price + sl_distance_from_current_abs
                if sl_price_calculated <= Decimal("0"): logger.error(f"Adjusted SL price ({sl_price_calculated:.{DEFAULT_PRICE_DP}f}) is still invalid (<=0)."); return None
            if sl_distance_from_current_abs <= Decimal("0"): logger.error(f"Calculated SL distance ({sl_distance_from_current_abs.normalize()}) is invalid (<=0)."); return None

            quantity_calculated_base_asset: Decimal
            if self.config.market_type == "inverse":
                if current_price <= Decimal("0"): logger.error("Invalid current_price for inverse quantity calc."); return None
                risk_amount_in_quote_ccy = risk_amount_per_trade_settle_ccy * current_price; quantity_calculated_base_asset = risk_amount_in_quote_ccy / sl_distance_from_current_abs
            else: # Linear
                value_change_per_point_per_base_unit = market_contract_size
                if value_change_per_point_per_base_unit <= Decimal("0"): logger.error("Invalid contract size for linear quantity."); return None
                risk_per_unit_of_base = sl_distance_from_current_abs * value_change_per_point_per_base_unit
                if risk_per_unit_of_base <= Decimal("0"): logger.error(f"Calculated zero or negative risk per unit of base ({risk_per_unit_of_base.normalize()}). Cannot determine quantity."); return None
                quantity_calculated_base_asset = risk_amount_per_trade_settle_ccy / risk_per_unit_of_base
            
            quantity_str_formatted = self.exchange_manager.format_amount(quantity_calculated_base_asset, ROUND_DOWN)
            quantity_decimal_final = safe_decimal(quantity_str_formatted)
            if quantity_decimal_final.is_nan() or quantity_decimal_final <= Decimal("0"): logger.error(f"Calculated quantity ({quantity_str_formatted}) is invalid or zero after formatting. Original calc: {quantity_calculated_base_asset.normalize()}"); return None
            
            if quantity_decimal_final < market_min_order_size: logger.error(f"Calculated quantity {quantity_decimal_final.normalize()} is less than market minimum order size {market_min_order_size.normalize()}."); return None

            tp_price_calculated: Optional[Decimal] = None
            if self.config.tp_atr_multiplier > Decimal("0"):
                tp_distance_atr_points = atr * self.config.tp_atr_multiplier
                tp_price_calculated = current_price + tp_distance_atr_points if side == "buy" else current_price - tp_distance_atr_points
                if tp_price_calculated <= Decimal("0"): logger.warning(f"Calculated TP price ({tp_price_calculated:.{DEFAULT_PRICE_DP}f}) is invalid (<=0). Disabling TP for this trade."); tp_price_calculated = None
            
            tsl_distance_price_points = current_price * (self.config.trailing_stop_percent / 100)
            if tsl_distance_price_points < market_tick_size: logger.debug(f"TSL distance ({tsl_distance_price_points.normalize()}) < min tick size ({market_tick_size.normalize()}). Adjusting TSL distance to min tick size."); tsl_distance_price_points = market_tick_size
            tsl_distance_str_formatted = self.exchange_manager.format_price(tsl_distance_price_points)
            tsl_distance_decimal_final = safe_decimal(tsl_distance_str_formatted)
            if tsl_distance_decimal_final.is_nan() or tsl_distance_decimal_final <= Decimal("0"):
                logger.warning(f"Calculated invalid TSL distance ('{tsl_distance_str_formatted}'). TSL might fail. Original calc: {tsl_distance_price_points.normalize()}"); tsl_distance_decimal_final = Decimal('NaN')

            sl_price_str_formatted = self.exchange_manager.format_price(sl_price_calculated)
            sl_price_decimal_final = safe_decimal(sl_price_str_formatted)
            if sl_price_decimal_final.is_nan() or sl_price_decimal_final <= Decimal("0"): logger.error(f"Formatted SL price ('{sl_price_str_formatted}') is invalid. Aborting parameter calculation."); return None
            
            tp_price_decimal_final: Optional[Decimal] = None
            if tp_price_calculated is not None:
                 tp_price_str_formatted = self.exchange_manager.format_price(tp_price_calculated)
                 tp_price_decimal_final = safe_decimal(tp_price_str_formatted)
                 if tp_price_decimal_final.is_nan() or tp_price_decimal_final <= Decimal("0"): logger.warning(f"Failed to format a valid TP price ('{tp_price_str_formatted}'). Disabling TP for this trade."); tp_price_decimal_final = None

            params_out: Dict[str, Optional[Decimal]] = {"qty": quantity_decimal_final, "sl_price": sl_price_decimal_final, "tp_price": tp_price_decimal_final, "tsl_distance": tsl_distance_decimal_final if not tsl_distance_decimal_final.is_nan() else None}
            log_tp_str = f"{params_out['tp_price'].normalize()}" if params_out['tp_price'] else "Disabled"
            log_tsl_str = f"{params_out['tsl_distance'].normalize()}" if params_out['tsl_distance'] else "Invalid/Not Set"
            settle_ccy_display = self.market_info.get('settle', self.config.symbol.split(':')[-1] if ':' in self.config.symbol else 'SETTLE')
            logger.info(f"Trade Parameters Calculated for {side.upper()} entry: Qty={params_out['qty'].normalize()} {self.market_info.get('base', 'BASE')}, EntryPrice (approx.)={current_price.normalize():.{DEFAULT_PRICE_DP}f}, SLPrice={params_out['sl_price'].normalize()}, TPPrice={log_tp_str}, TSLDistance (for future TSL activation)={log_tsl_str}, RiskAmountSettle={risk_amount_per_trade_settle_ccy.normalize():.{DEFAULT_PRICE_DP}f} {settle_ccy_display}, ATR={atr.normalize():.{DEFAULT_PRICE_DP+1}f}")
            return params_out
        except (InvalidOperation, DivisionByZero, TypeError, Exception) as e: logger.error(f"Error calculating trade parameters for {side.upper()} side: {e}", exc_info=True); return None

    def _execute_market_order(self, side: str, qty_decimal: Decimal) -> Optional[Dict[str, Any]]:
        if not self.exchange or not self.market_info: logger.error("Cannot execute market order: Exchange or Market info missing."); return None
        qty_str_for_api = self.exchange_manager.format_amount(qty_decimal, rounding_mode=ROUND_DOWN)
        final_qty_decimal_for_log = safe_decimal(qty_str_for_api)
        if final_qty_decimal_for_log.is_nan() or final_qty_decimal_for_log <= Decimal("0"): logger.error(f"Attempted market order with zero/invalid formatted quantity: '{qty_str_for_api}' (Original Decimal: {qty_decimal.normalize()}). Order aborted."); return None
        try: amount_float_for_ccxt = float(qty_str_for_api)
        except ValueError: logger.error(f"Could not convert formatted quantity string '{qty_str_for_api}' to float for API. Order aborted."); return None

        logger.trade(f"{Fore.CYAN}Attempting MARKET {side.upper()} order: {final_qty_decimal_for_log.normalize()} {self.market_info.get('base', '')} for {self.config.symbol}...{Style.RESET_ALL}")
        try:
            params_v5 = {"category": self.config.bybit_v5_category, "positionIdx": self.config.position_idx, "timeInForce": "ImmediateOrCancel"}
            order_response = fetch_with_retries(self.exchange.create_market_order, symbol=self.config.symbol, side=side, amount=amount_float_for_ccxt, params=params_v5, max_retries=self.config.max_fetch_retries, delay_seconds=self.config.retry_delay_seconds)
            if order_response is None: logger.error(f"{Fore.RED}Market order submission failed after retries (returned None unexpectedly).{Style.RESET_ALL}"); return None

            order_id, order_status, filled_qty_str, avg_fill_price_str = order_response.get("id","[N/A]"), order_response.get("status","[unknown]"), order_response.get("filled","0"), order_response.get("average","0")
            filled_qty_decimal, avg_fill_price_decimal = safe_decimal(filled_qty_str), safe_decimal(avg_fill_price_str)
            avg_price_log_str = avg_fill_price_decimal.normalize() if not avg_fill_price_decimal.is_nan() and avg_fill_price_decimal > Decimal("0") else "[N/A]"
            logger.trade(f"{Style.BRIGHT}{Fore.GREEN}Market order submitted: ID {order_id}, Side {side.upper()}, Ordered Qty {final_qty_decimal_for_log.normalize()}, Status: {order_status}, Filled Qty: {filled_qty_decimal.normalize()}, AvgFillPx: {avg_price_log_str}{Style.RESET_ALL}")
            termux_notify(f"{self.config.symbol} Order Submitted", f"Market {side.upper()} {final_qty_decimal_for_log.normalize()} ID:{order_id}, Status:{order_status}")

            if order_status == "rejected": logger.error(f"{Fore.RED}Market order {order_id} was REJECTED. Reason: '{order_response.get('info', {}).get('rejectReason', 'No reason provided by exchange')}'. Full info: {order_response.get('info')}{Style.RESET_ALL}"); return None
            elif order_status == "canceled" and filled_qty_decimal == Decimal("0") and params_v5.get("timeInForce") == "ImmediateOrCancel": logger.error(f"{Fore.RED}Market order {order_id} (IOC) was CANCELED with 0 filled. Order did not execute.{Style.RESET_ALL}"); return None
            elif order_status == "expired": logger.error(f"{Fore.RED}Market order {order_id} EXPIRED. This is unexpected for market orders.{Style.RESET_ALL}"); return None
            
            logger.debug(f"Short delay ({self.config.order_check_delay_seconds}s) after market order {order_id} submission for propagation..."); time.sleep(self.config.order_check_delay_seconds)
            return order_response
        except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e: logger.error(f"{Fore.RED}Order placement failed ({type(e).__name__}): {e}{Style.RESET_ALL}", exc_info=False); termux_notify(f"{self.config.symbol} Order FAILED", f"Market {side.upper()} failed: {str(e)[:50]}"); return None
        except Exception as e: logger.error(f"{Fore.RED}Unexpected error placing market order: {e}{Style.RESET_ALL}", exc_info=True); termux_notify(f"{self.config.symbol} Order ERROR", f"Market {side.upper()} unexpected error."); return None

    def _set_position_protection(self, position_side: str, sl_price: Optional[Decimal] = None, tp_price: Optional[Decimal] = None, is_tsl: bool = False, tsl_distance: Optional[Decimal] = None, tsl_activation_price: Optional[Decimal] = None) -> bool:
        if not self.exchange or not self.market_info: logger.error("Cannot set position protection: Exchange or Market info missing."); return False
        market_id = self.market_info.get("id"); tracker_key = position_side.lower()
        if not market_id: logger.error("Cannot set position protection: Market ID missing."); return False
        if tracker_key not in self.protection_tracker: logger.error(f"Invalid position_side '{position_side}' for protection tracker update."); return False

        sl_price_api_str = self.exchange_manager._format_v5_param(sl_price, "price", allow_zero=True) if sl_price else "0"
        tp_price_api_str = self.exchange_manager._format_v5_param(tp_price, "price", allow_zero=True) if tp_price else "0"
        tsl_distance_api_str = self.exchange_manager._format_v5_param(tsl_distance, "distance", allow_zero=False) if tsl_distance else "0"
        tsl_activation_price_api_str = self.exchange_manager._format_v5_param(tsl_activation_price, "price", allow_zero=False) if tsl_activation_price else "0"
        
        api_params: Dict[str, Any] = {"category": self.config.bybit_v5_category, "symbol": market_id, "positionIdx": self.config.position_idx, "tpslMode": V5_TPSL_MODE_FULL}
        action_description, new_tracker_state = "", None

        if is_tsl:
            if tsl_distance_api_str and tsl_distance_api_str != "0" and tsl_activation_price_api_str and tsl_activation_price_api_str != "0":
                api_params.update({"trailingStop": tsl_distance_api_str, "activePrice": tsl_activation_price_api_str, "triggerBy": self.config.tsl_trigger_by, "stopLoss": "0", "takeProfit": "0"})
                action_description, new_tracker_state = f"ACTIVATE/MODIFY TSL (Dist: {tsl_distance_api_str}, ActPx: {tsl_activation_price_api_str})", "ACTIVE_TSL"
            else: logger.error(f"Cannot activate TSL for {position_side.upper()}: Invalid TSL distance ('{tsl_distance_api_str}') or activation price ('{tsl_activation_price_api_str}'). Must be positive values."); return False
        elif sl_price_api_str != "0" or tp_price_api_str != "0":
            if sl_price_api_str != "0": api_params["stopLoss"] = sl_price_api_str
            if tp_price_api_str != "0": api_params["takeProfit"] = tp_price_api_str
            api_params.update({"slTriggerBy": self.config.sl_trigger_by, "tpTriggerBy": self.config.sl_trigger_by, "trailingStop": "0", "activePrice": "0"})
            action_description, new_tracker_state = f"SET SL={api_params.get('stopLoss','0')} TP={api_params.get('takeProfit','0')}", "ACTIVE_SLTP"
        else: # Clearing all stops
            api_params.update({"stopLoss": "0", "takeProfit": "0", "trailingStop": "0", "activePrice": "0"})
            action_description, new_tracker_state = "CLEAR ALL SL/TP/TSL", None
        
        logger.trade(f"{Fore.CYAN}Attempting to {action_description} for {position_side.upper()} {self.config.symbol}...{Style.RESET_ALL}"); logger.debug(f"Calling V5 setTradingStop with parameters: {api_params}")
        private_method_name = "privatePostPositionTradingStop"
        if not hasattr(self.exchange, private_method_name): logger.critical(f"{Style.BRIGHT}{Fore.RED}Fatal Error: CCXT private method '{private_method_name}' not found. Cannot manage position protection.{Style.RESET_ALL}"); return False
        method_to_call = getattr(self.exchange, private_method_name)
        try:
            response = fetch_with_retries(method_to_call, params=api_params, max_retries=self.config.max_fetch_retries, delay_seconds=self.config.retry_delay_seconds)
            if response and response.get("retCode") == V5_SUCCESS_RETCODE:
                logger.trade(f"{Style.BRIGHT}{Fore.GREEN}{action_description} successful for {position_side.upper()} {self.config.symbol}.{Style.RESET_ALL}")
                termux_notify(f"{self.config.symbol} Protection {('Set' if new_tracker_state else 'Cleared')}", f"{action_description} for {position_side.upper()}")
                self.protection_tracker[tracker_key] = new_tracker_state; return True
            else:
                ret_code = response.get("retCode", "[N/A]") if response else "[No Response]"; ret_msg = response.get("retMsg", "[No error message]") if response else "[No Response]"
                logger.error(f"{Fore.RED}{action_description} failed for {position_side.upper()} {self.config.symbol}. API Response: Code={ret_code}, Msg='{ret_msg}'.{Style.RESET_ALL}"); logger.debug(f"Full response from failed {private_method_name}: {response}")
                termux_notify(f"{self.config.symbol} Protection FAILED", f"{action_description[:30]}... failed: {ret_msg[:50]}"); return False
        except Exception as e: logger.error(f"{Fore.RED}Unexpected error during '{action_description}' for {position_side.upper()} {self.config.symbol}: {e}{Style.RESET_ALL}", exc_info=True); termux_notify(f"{self.config.symbol} Protection ERROR", f"{action_description[:30]}... error."); return False

    def _verify_position_state(self, expected_side_logical: Optional[str], expected_qty_min_abs: Decimal = POSITION_QTY_EPSILON, max_attempts: int = 4, delay_seconds: float = 1.5, action_context: str = "Position Verification") -> Tuple[bool, Optional[Dict[str, Dict[str, Any]]]]:
        logger.debug(f"{action_context}: Verifying position state. Expecting side: '{expected_side_logical}', MinAbsQty (if open): {expected_qty_min_abs.normalize()}. Max attempts: {max_attempts}.")
        last_known_position_summary: Optional[Dict[str, Dict[str, Any]]] = None
        for attempt in range(max_attempts):
            logger.debug(f"{action_context}: Verification attempt {attempt + 1}/{max_attempts}...")
            current_positions_summary_fetched = self.exchange_manager.get_current_position(); last_known_position_summary = current_positions_summary_fetched
            if current_positions_summary_fetched is None:
                logger.warning(f"{action_context} Warning: Failed to fetch position state on attempt {attempt + 1}.")
                if attempt < max_attempts - 1: time.sleep(delay_seconds); continue
                else: logger.error(f"{Fore.RED}{action_context} FAILED: Could not fetch position state after {max_attempts} attempts.{Style.RESET_ALL}"); return False, last_known_position_summary

            actual_is_flat, actual_open_side_logical, actual_open_qty_abs = True, None, Decimal("0")
            long_pos_data, short_pos_data = current_positions_summary_fetched.get("long", {}), current_positions_summary_fetched.get("short", {})
            if long_pos_data and safe_decimal(long_pos_data.get("qty", "0")).copy_abs() >= POSITION_QTY_EPSILON: actual_is_flat, actual_open_side_logical, actual_open_qty_abs = False, "long", safe_decimal(long_pos_data.get("qty", "0")).copy_abs()
            elif short_pos_data and safe_decimal(short_pos_data.get("qty", "0")).copy_abs() >= POSITION_QTY_EPSILON: actual_is_flat, actual_open_side_logical, actual_open_qty_abs = False, "short", safe_decimal(short_pos_data.get("qty", "0")).copy_abs()
            
            verification_succeeded, log_message_suffix = False, ""
            if expected_side_logical is None:
                verification_succeeded = actual_is_flat
                log_message_suffix = f"Expected FLAT, Actual: {'FLAT' if actual_is_flat else f'{str(actual_open_side_logical).upper()} Qty={actual_open_qty_abs.normalize()}'}"
            elif actual_open_side_logical == expected_side_logical:
                quantity_matches_expectation = actual_open_qty_abs >= expected_qty_min_abs; verification_succeeded = quantity_matches_expectation
                log_message_suffix = (f"Expected {expected_side_logical.upper()} (MinAbsQty~{expected_qty_min_abs.normalize()}), Actual: {actual_open_side_logical.upper()} Qty={actual_open_qty_abs.normalize()} ({'QTY OK' if quantity_matches_expectation else 'QTY MISMATCH'})")
            else: log_message_suffix = (f"Expected {str(expected_side_logical).upper() if expected_side_logical else 'FLAT'}, Actual: {'FLAT' if actual_is_flat else (str(actual_open_side_logical).upper() + ' Qty=' + actual_open_qty_abs.normalize()) if actual_open_side_logical else 'UNKNOWN/ERROR'} (SIDE MISMATCH)")
            
            logger.debug(f"{action_context} Check {attempt + 1}: {log_message_suffix}")
            if verification_succeeded: logger.info(f"{Style.BRIGHT}{Fore.GREEN}{action_context} SUCCEEDED on attempt {attempt + 1}. State confirmed: {log_message_suffix}{Style.RESET_ALL}"); return True, current_positions_summary_fetched
            if attempt < max_attempts - 1: logger.debug(f"State not as expected. Waiting {delay_seconds}s for next attempt..."); time.sleep(delay_seconds)
            else: logger.error(f"{Fore.RED}{action_context} FAILED after {max_attempts} attempts. Final state check: {log_message_suffix}{Style.RESET_ALL}"); return False, current_positions_summary_fetched
        return False, last_known_position_summary # Fallback

    def place_risked_market_order(self, side: str, atr: Decimal, total_equity: Decimal, current_price: Decimal) -> bool:
        if not self.exchange or not self.market_info: logger.critical("OrderManager not fully initialized for placing order."); return False
        if side not in ["buy", "sell"]: logger.error(f"Invalid side '{side}' for place_risked_market_order."); return False
        if atr.is_nan() or atr <= Decimal("0"): logger.error("Entry Aborted: Invalid ATR value for risk calculation."); return False
        if total_equity is None or total_equity.is_nan() or total_equity <= Decimal("0"): logger.error("Entry Aborted: Invalid Equity value for risk calculation."); return False
        if current_price.is_nan() or current_price <= Decimal("0"): logger.error("Entry Aborted: Invalid Current Price for calculations."); return False

        logical_position_side = "long" if side == "buy" else "short"
        logger.info(f"{Style.BRIGHT}{Fore.MAGENTA}--- Initiating Entry Sequence for {logical_position_side.upper()} Position ---{Style.RESET_ALL}")
        
        trade_params = self._calculate_trade_parameters(side, atr, total_equity, current_price)
        if not trade_params or not trade_params.get("qty") or trade_params["qty"] <= Decimal("0"): logger.error("Entry Aborted: Failed to calculate valid trade parameters (qty, SL, etc.)."); return False # type: ignore
        qty_to_order, initial_sl_price, initial_tp_price = trade_params["qty"], trade_params.get("sl_price"), trade_params.get("tp_price") # type: ignore
        if initial_sl_price is None or initial_sl_price.is_nan() or initial_sl_price <= Decimal("0"): logger.error(f"Entry Aborted: Invalid Stop Loss price ({initial_sl_price}) calculated."); return False

        market_order_info = self._execute_market_order(side, qty_to_order) # type: ignore
        if not market_order_info: logger.error(f"Entry Aborted: Market order execution failed for {side.upper()} {qty_to_order.normalize()}."); self._handle_entry_failure(side, qty_to_order); return False # type: ignore
        entry_order_id = market_order_info.get("id", "[N/A_ORDER_ID]"); avg_entry_price_from_order_resp = safe_decimal(market_order_info.get("average", "NaN"))

        min_expected_filled_qty_abs = qty_to_order * Decimal("0.90") # type: ignore
        verification_ok, final_verified_pos_state = self._verify_position_state(expected_side_logical=logical_position_side, expected_qty_min_abs=min_expected_filled_qty_abs, max_attempts=6, delay_seconds=max(self.config.order_check_delay_seconds, 1.0), action_context=f"Post-{logical_position_side.upper()}-Entry Verification")
        if not verification_ok: logger.error(f"{Fore.RED}Entry FAILED: Position verification failed after market order {entry_order_id}. Manual check required! Attempting cleanup...{Style.RESET_ALL}"); self._handle_entry_failure(side, qty_to_order); return False # type: ignore

        active_pos_details = final_verified_pos_state.get(logical_position_side) if final_verified_pos_state else {} # type: ignore
        if not active_pos_details: logger.error(f"{Fore.RED}Internal Error: Position {logical_position_side} verified OK, but details missing. Aborting entry sequence.{Style.RESET_ALL}"); self._handle_entry_failure(side, qty_to_order); return False # type: ignore
        
        actual_filled_qty_abs = safe_decimal(active_pos_details.get("qty", "0")).copy_abs(); actual_avg_entry_price = safe_decimal(active_pos_details.get("entry_price", "NaN"))
        if actual_avg_entry_price.is_nan() and not avg_entry_price_from_order_resp.is_nan(): actual_avg_entry_price = avg_entry_price_from_order_resp; logger.debug(f"Used avg entry price from order response ({avg_entry_price_from_order_resp.normalize()}) as position data was NaN.")
        logger.info(f"{Style.BRIGHT}{Fore.GREEN}Position {logical_position_side.upper()} confirmed: Actual Qty={actual_filled_qty_abs.normalize()}, AvgEntryPx={actual_avg_entry_price.normalize() if not actual_avg_entry_price.is_nan() else '[N/A]'}{Style.RESET_ALL}")
        if actual_filled_qty_abs < qty_to_order * Decimal("0.99"): logger.warning(f"Filled quantity {actual_filled_qty_abs.normalize()} is notably less than ordered {qty_to_order.normalize()}. This might be due to slippage or partial fill.") # type: ignore

        set_stops_successful = self._set_position_protection(logical_position_side, sl_price=initial_sl_price, tp_price=initial_tp_price)
        if not set_stops_successful: logger.error(f"{Fore.RED}Entry Alert: Failed to set initial SL/TP for {logical_position_side.upper()} position. Attempting emergency close!{Style.RESET_ALL}"); self.close_position(logical_position_side, actual_filled_qty_abs, reason="EmergencyClose:FailedInitialStopSet"); return False
        
        if self.config.enable_journaling: self.log_trade_entry_to_journal(side, actual_filled_qty_abs, actual_avg_entry_price, entry_order_id)
        logger.info(f"{Style.BRIGHT}{Fore.GREEN}--- Entry Sequence for {logical_position_side.upper()} Completed Successfully ---{Style.RESET_ALL}"); return True

    def manage_trailing_stop(self, position_side: str, entry_price: Decimal, current_market_price: Decimal, current_atr: Decimal):
        if not self.exchange or not self.market_info: logger.error("Cannot manage TSL: Exchange/Market info missing."); return
        tracker_key = position_side.lower()
        if self.protection_tracker.get(tracker_key) != "ACTIVE_SLTP": logger.debug(f"TSL Management Check ({position_side.upper()}): Not ACTIVE_SLTP (Tracker: {self.protection_tracker.get(tracker_key)}). Skipping."); return
        if any(val.is_nan() or val <= Decimal("0") for val in [current_atr, entry_price, current_market_price]): # type: ignore
            logger.debug(f"TSL Check ({position_side.upper()}): Invalid ATR/entry_px/market_px. Skipping."); return
        try:
            activation_distance_points = current_atr * self.config.tsl_activation_atr_multiplier # type: ignore
            tsl_activation_target_price = entry_price + activation_distance_points if position_side == "long" else entry_price - activation_distance_points # type: ignore
            if tsl_activation_target_price.is_nan() or tsl_activation_target_price <= Decimal("0"): logger.warning(f"Invalid TSL activation price ({tsl_activation_target_price.normalize()}). Skipping TSL for {position_side.upper()}."); return

            tsl_actual_distance_points = current_market_price * (self.config.trailing_stop_percent / 100) # type: ignore
            min_tick_size = self.market_info.get('tick_size', Decimal('1e-8')) # type: ignore
            if not min_tick_size.is_nan() and min_tick_size > Decimal("0") and tsl_actual_distance_points < min_tick_size: # type: ignore
                logger.debug(f"TSL distance ({tsl_actual_distance_points.normalize()}) < min tick ({min_tick_size.normalize()}). Adjusting to min tick."); tsl_actual_distance_points = min_tick_size # type: ignore
            if tsl_actual_distance_points <= Decimal("0"): logger.warning(f"Invalid TSL distance ({tsl_actual_distance_points.normalize()}). Skipping TSL for {position_side.upper()}."); return

            should_activate_tsl = (position_side == "long" and current_market_price >= tsl_activation_target_price) or \
                                  (position_side == "short" and current_market_price <= tsl_activation_target_price)
            if should_activate_tsl:
                logger.trade(f"{Fore.MAGENTA}Trailing Stop Loss (TSL) activation condition MET for {position_side.upper()}!{Style.RESET_ALL}")
                logger.trade(f"  Details: EntryPx={entry_price.normalize():.{DEFAULT_PRICE_DP}f}, CurrentPx={current_market_price.normalize():.{DEFAULT_PRICE_DP}f}, TSLActivationTargetPx~={tsl_activation_target_price.normalize():.{DEFAULT_PRICE_DP}f}, TSLDistanceToSet~={tsl_actual_distance_points.normalize():.{DEFAULT_PRICE_DP}f}")
                activation_successful = self._set_position_protection(position_side, is_tsl=True, tsl_distance=tsl_actual_distance_points, tsl_activation_price=tsl_activation_target_price)
                if activation_successful: logger.trade(f"{Style.BRIGHT}{Fore.GREEN}TSL activated successfully for {position_side.upper()} position.{Style.RESET_ALL}")
                else: logger.error(f"{Fore.RED}Failed to activate TSL for {position_side.upper()} position via API.{Style.RESET_ALL}")
            else: logger.debug(f"TSL Check ({position_side.upper()}): Activation NOT MET. (CurrentPx: {current_market_price.normalize():.{DEFAULT_PRICE_DP}f}, TargetActivationPx: ~{tsl_activation_target_price.normalize():.{DEFAULT_PRICE_DP}f})")
        except Exception as e: logger.error(f"Error managing TSL for {position_side.upper()} position: {e}", exc_info=True)

    def close_position(self, position_side_to_close: str, qty_abs_to_close: Decimal, reason: str = "Strategy Exit Signal") -> bool:
        if not self.exchange or not self.market_info: logger.critical("OrderManager not fully initialized for closing position."); return False
        if position_side_to_close not in ["long", "short"]: logger.error(f"Invalid side '{position_side_to_close}' for close_position."); return False
        if qty_abs_to_close.is_nan() or qty_abs_to_close.copy_abs() < POSITION_QTY_EPSILON:
            logger.warning(f"Close requested for zero/negligible quantity ({qty_abs_to_close.normalize()}). Skipping close for {position_side_to_close.upper()}."); self.protection_tracker[position_side_to_close.lower()] = None; return True
        
        closing_order_side = "sell" if position_side_to_close == "long" else "buy"; tracker_key = position_side_to_close.lower()
        logger.trade(f"{Fore.YELLOW}Attempting to CLOSE {position_side_to_close.upper()} position (Qty: {qty_abs_to_close.normalize()} {self.market_info.get('base', '')}) for {self.config.symbol} | Reason: {reason}...{Style.RESET_ALL}")
        
        logger.debug(f"Clearing any existing protection for {position_side_to_close.upper()} before closing...")
        if self._set_position_protection(position_side_to_close, sl_price=None, tp_price=None, is_tsl=False):
            logger.info(f"Protection cleared (or was already clear) for {position_side_to_close.upper()} position."); self.protection_tracker[tracker_key] = None
        else: logger.warning(f"{Fore.YELLOW}Failed to explicitly confirm protection clear for {position_side_to_close.upper()}. Proceeding with close cautiously...{Style.RESET_ALL}")

        close_market_order_info = self._execute_market_order(closing_order_side, qty_abs_to_close)
        if not close_market_order_info:
            logger.error(f"{Fore.RED}Failed to submit closing market order for {position_side_to_close.upper()}. MANUAL INTERVENTION REQUIRED!{Style.RESET_ALL}"); termux_notify(f"{self.config.symbol} CLOSE ORDER FAILED", f"Market {closing_order_side.upper()} order failed!"); return False
        
        close_order_id = close_market_order_info.get("id", "[N/A_CLOSE_ORDER_ID]"); avg_close_price_decimal = safe_decimal(close_market_order_info.get("average"), default=Decimal("NaN"))
        logger.trade(f"{Fore.YELLOW}Closing market order ({close_order_id}) submitted for {position_side_to_close.upper()}. Reported AvgClosePrice: {avg_close_price_decimal.normalize() if not avg_close_price_decimal.is_nan() else '[Pending/N/A]'}{Style.RESET_ALL}")
        termux_notify(f"{self.config.symbol} Position Closing", f"{position_side_to_close.upper()} close order {close_order_id} submitted.")

        verification_ok, _ = self._verify_position_state(expected_side_logical=None, max_attempts=6, delay_seconds=max(self.config.order_check_delay_seconds + 0.5, 1.5), action_context=f"Post-{position_side_to_close.upper()}-Close Verification")
        if self.config.enable_journaling: self.log_trade_exit_to_journal(position_side_to_close, qty_abs_to_close, avg_close_price_decimal, close_order_id, reason)
        
        if not verification_ok: logger.error(f"{Fore.RED}Position {position_side_to_close.upper()} closure verification FAILED. Position may still be open. MANUAL INTERVENTION REQUIRED!{Style.RESET_ALL}"); termux_notify(f"{self.config.symbol} CLOSE VERIFY FAILED", f"{position_side_to_close.upper()} position may still be open!"); return False
        
        logger.trade(f"{Style.BRIGHT}{Fore.GREEN}Position {position_side_to_close.upper()} confirmed closed (flat) via verification.{Style.RESET_ALL}"); self.protection_tracker[tracker_key] = None; return True

    def _handle_entry_failure(self, failed_entry_order_side: str, attempted_qty_abs: Decimal):
        logger.warning(f"{Fore.YELLOW}Handling potential entry failure for {failed_entry_order_side.upper()} order (intended qty: {attempted_qty_abs.normalize()}). Checking for lingering position...{Style.RESET_ALL}")
        logical_pos_side_to_check = "long" if failed_entry_order_side == "buy" else "short"
        time.sleep(max(self.config.order_check_delay_seconds, 1.0) + 1.0)
        
        _, current_positions_summary = self._verify_position_state(expected_side_logical=None, max_attempts=2, delay_seconds=1.0, action_context=f"Entry-Failure-Cleanup-Check-{logical_pos_side_to_check.upper()}")
        if current_positions_summary is None:
            logger.error(f"{Fore.RED}Could not fetch positions during entry failure handling for {logical_pos_side_to_check.upper()}. MANUAL CHECK URGENTLY REQUIRED!{Style.RESET_ALL}"); termux_notify(f"{self.config.symbol} URGENT CHECK", "Failed to get position state during entry failure cleanup!"); return

        lingering_pos_details = current_positions_summary.get(logical_pos_side_to_check, {}); current_lingering_qty_abs = safe_decimal(lingering_pos_details.get("qty", "0")).copy_abs()
        if current_lingering_qty_abs >= POSITION_QTY_EPSILON:
            logger.error(f"{Fore.RED}Lingering {logical_pos_side_to_check.upper()} position (Qty: {current_lingering_qty_abs.normalize()}) found after failed entry. Attempting emergency close...{Style.RESET_ALL}"); termux_notify(f"{self.config.symbol} Emergency Close", f"Lingering {logical_pos_side_to_check.upper()} pos found.")
            if self.close_position(logical_pos_side_to_check, current_lingering_qty_abs, reason="EmergencyClose:LingeringAfterEntryFail"): logger.info(f"Emergency close for lingering {logical_pos_side_to_check.upper()} position submitted/confirmed.")
            else: logger.critical(f"{Style.BRIGHT}{Fore.RED}EMERGENCY CLOSE FAILED for lingering {logical_pos_side_to_check.upper()}. MANUAL INTERVENTION URGENTLY REQUIRED!{Style.RESET_ALL}"); termux_notify(f"{self.config.symbol} URGENT CHECK", f"Emergency close of lingering {logical_pos_side_to_check.upper()} FAILED!")
        else: logger.info(f"No significant lingering {logical_pos_side_to_check.upper()} position detected. Current qty: {current_lingering_qty_abs.normalize()}."); self.protection_tracker[logical_pos_side_to_check] = None

    def _write_journal_row(self, trade_data: Dict[str, Any]):
        if not self.config.enable_journaling: return
        journal_file = Path(self.config.journal_file_path)
        file_already_exists_and_has_content = journal_file.is_file() and journal_file.stat().st_size > 0
        try:
            journal_file.parent.mkdir(parents=True, exist_ok=True)
            with journal_file.open("a", newline="", encoding="utf-8") as csvfile:
                fieldnames = ["TimestampUTC", "Symbol", "Action", "Side", "Quantity", "AvgPrice", "OrderID", "Reason", "Notes"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
                if not file_already_exists_and_has_content: writer.writeheader()
                row_to_write = {field: ('NaN' if isinstance(trade_data.get(field), Decimal) and trade_data.get(field).is_nan() else (f"{trade_data.get(field).normalize()}" if isinstance(trade_data.get(field), Decimal) else str(trade_data.get(field, 'N/A')))) for field in fieldnames} # type: ignore
                row_to_write['Notes'] = str(trade_data.get('Notes', '')) # Ensure Notes is string
                writer.writerow(row_to_write)
            logger.debug(f"Trade action '{trade_data.get('Action', 'Unknown')}' logged to journal: {journal_file}")
        except IOError as e: logger.error(f"I/O error writing trade action '{trade_data.get('Action', '')}' to journal '{journal_file}': {e}")
        except Exception as e: logger.error(f"Unexpected error writing trade action '{trade_data.get('Action', '')}' to journal: {e}", exc_info=True)

    def log_trade_entry_to_journal(self, order_side: str, filled_qty_abs: Decimal, avg_fill_price: Decimal, order_id: Optional[str]):
        self._write_journal_row({"TimestampUTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), "Symbol": self.config.symbol, "Action": "ENTRY", "Side": ("long" if order_side == "buy" else "short").upper(), "Quantity": filled_qty_abs, "AvgPrice": avg_fill_price, "OrderID": order_id, "Reason": "Strategy Entry Signal"})

    def log_trade_exit_to_journal(self, position_side_closed: str, closed_qty_abs: Decimal, avg_close_price: Decimal, order_id: Optional[str], exit_reason: str):
        self._write_journal_row({"TimestampUTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), "Symbol": self.config.symbol, "Action": "EXIT", "Side": position_side_closed.upper(), "Quantity": closed_qty_abs, "AvgPrice": avg_close_price, "OrderID": order_id, "Reason": exit_reason})

# --- Status Display Class ---
class StatusDisplay:
    def __init__(self, config: TradingConfig):
        self.config = config
        self._default_price_dp_display = DEFAULT_PRICE_DP
        self._default_amount_dp_display = DEFAULT_AMOUNT_DP

    def _format_decimal_for_rich(self, value: Optional[Decimal], precision: Optional[int] = None, default_precision_fallback: int = 2, add_commas: bool = False, highlight_negative: bool = False, default_style: str = "white", style_override: Optional[str] = None) -> Text:
        if value is None or (isinstance(value, Decimal) and value.is_nan()): return Text("N/A", style="dim")
        dp_to_use = precision if precision is not None else default_precision_fallback
        try:
            formatted_decimal_val = value.quantize(Decimal(f"1e-{dp_to_use}"), rounding=ROUND_HALF_EVEN)
            format_spec = f"{{:{',' if add_commas else ''}.{dp_to_use}f}}"; display_string = format_spec.format(formatted_decimal_val)
            current_style = style_override if style_override else default_style
            if highlight_negative and not style_override:
                if formatted_decimal_val < Decimal("0"): current_style = "bright_red"
                elif formatted_decimal_val > Decimal("0"): current_style = "bright_green"
            return Text(display_string, style=current_style)
        except (ValueError, TypeError, InvalidOperation) as e: logger.error(f"Error formatting decimal '{value}' for Rich display: {e}"); return Text("ERR", style="bold bright_red")

    def print_status_panel(self, cycle_num: int, current_timestamp: Optional[datetime], current_market_price: Optional[Decimal], indicators_data: Optional[Dict[str, Any]], current_positions_summary: Optional[Dict[str, Dict[str, Any]]], account_equity: Optional[Decimal], signal_check_result_or_status: Dict[str, Any], protection_status_tracker: Dict[str, Optional[str]], market_specific_info: Optional[Dict[str, Any]]):
        price_display_dp = self._default_price_dp_display; amount_display_dp = self._default_amount_dp_display
        if market_specific_info and "precision_dp" in market_specific_info:
             price_display_dp = market_specific_info["precision_dp"].get("price", self._default_price_dp_display)
             amount_display_dp = market_specific_info["precision_dp"].get("amount", self._default_amount_dp_display)

        panel_content = Text()
        timestamp_str = current_timestamp.strftime("%Y-%m-%d %H:%M:%S %Z") if current_timestamp else Text("Timestamp N/A", style="dim").plain
        panel_title_str = f" Cycle {cycle_num} | {self.config.symbol} ({self.config.interval}) | {timestamp_str} "
        settle_ccy = market_specific_info.get("settle", "SETTLE") if market_specific_info else "SETTLE"
        panel_content.append("Price: ", style="bold bright_cyan"); panel_content.append(self._format_decimal_for_rich(current_market_price, price_display_dp, style_override="bright_white"))
        panel_content.append(" | Equity: ", style="bold bright_yellow"); panel_content.append(self._format_decimal_for_rich(account_equity, 2, add_commas=True, style_override="bright_yellow")); panel_content.append(f" {settle_ccy}\n", style="bright_yellow"); panel_content.append("---\n", style="dim")
        
        panel_content.append("Indicators: ", style="bold bright_cyan")
        if indicators_data:
            parts = []
            def fmt_ind(key: str, prec: int = 1, style: str = "white", is_bool: bool = False) -> Text:
                 val = indicators_data.get(key)
                 if is_bool: return Text(str(val), style=(style if val else "dim " + style))
                 if isinstance(val, bool): return Text(str(val), style=style)
                 dec_val = val if isinstance(val, Decimal) else safe_decimal(str(val) if val is not None else "NaN")
                 return self._format_decimal_for_rich(dec_val, precision=prec, default_style=style)

            # Original Indicators
            parts.append(Text("EMA(F/S/T): ").append(fmt_ind('fast_ema', price_display_dp, "cyan")).append("/").append(fmt_ind('slow_ema', price_display_dp, "magenta")).append("/").append(fmt_ind('trend_ema', price_display_dp, "yellow")))
            stoch_text = Text("Stoch(K/D/PrevK): ").append(fmt_ind('stoch_k',1,"bright_blue")).append("/").append(fmt_ind('stoch_d',1,"blue")).append("/").append(fmt_ind('stoch_k_prev',1,"dim blue"))
            if indicators_data.get('stoch_kd_bullish'): stoch_text.append(" [b green]▲BullX[/]", style="green")
            elif indicators_data.get('stoch_kd_bearish'): stoch_text.append(" [b red]▼BearX[/]", style="red")
            parts.append(stoch_text)
            parts.append(Text(f"ATR({indicators_data.get('atr_period',self.config.atr_period)}): ").append(fmt_ind('atr', price_display_dp+1, "bright_magenta")))
            adx_val_dec = indicators_data.get('adx'); adx_val_dec = adx_val_dec if isinstance(adx_val_dec, Decimal) else safe_decimal(str(adx_val_dec) if adx_val_dec is not None else "NaN")
            adx_style = "yellow" if not adx_val_dec.is_nan() and adx_val_dec > self.config.min_adx_level else "dim yellow"
            parts.append(Text(f"ADX({self.config.adx_period}): ").append(self._format_decimal_for_rich(adx_val_dec,1,default_style=adx_style)).append(" [+DI:",style="dim").append(fmt_ind('pdi',1,"bright_green")).append(" -DI:",style="dim").append(fmt_ind('mdi',1,"bright_red")).append("]",style="dim"))
            
            # VolumaticTrend Indicators (if enabled and present)
            if self.config.vt_enable and indicators_data.get('vt_trend_ema') is not None:
                vt_display_parts = [Text("VT: ", style="bold green")]
                vt_display_parts.append(Text("TrendEMA:").append(fmt_ind('vt_trend_ema', price_display_dp, "green")))
                vt_display_parts.append(Text("VWMA:").append(fmt_ind('vt_vwma', price_display_dp, "green")))
                vt_display_parts.append(Text("VolSpike:").append(fmt_ind('vt_is_volume_spike', style="green" if indicators_data.get('vt_is_volume_spike') else "dim green", is_bool=True)))
                candle_color_str = "Green" if indicators_data.get('vt_candle_is_green') else "Red" if indicators_data.get('vt_candle_is_red') else "Neutral"
                vt_display_parts.append(Text(f"Candle:{candle_color_str}", style="green" if candle_color_str=="Green" else "red" if candle_color_str=="Red" else "dim"))
                parts.append(Text(" | ", style="dim").join(vt_display_parts))

            panel_content.append(Text(" | ",style="dim").join(parts)); panel_content.append("\n")
        else: panel_content.append(Text("Calculating or data unavailable...", style="dim")); panel_content.append("\n")
        panel_content.append("---\n", style="dim")

        panel_content.append("Position: ", style="bold bright_cyan"); pos_display_text = Text("FLAT", style="bold bright_green")
        active_pos_side: Optional[str] = None; active_pos_data: Optional[Dict[str,Any]] = None
        if current_positions_summary:
            long_data, short_data = current_positions_summary.get("long",{}), current_positions_summary.get("short",{})
            if long_data and long_data.get("qty") and safe_decimal(long_data["qty"]).copy_abs() >= POSITION_QTY_EPSILON: active_pos_side, active_pos_data = "long", long_data
            elif short_data and short_data.get("qty") and safe_decimal(short_data["qty"]).copy_abs() >= POSITION_QTY_EPSILON: active_pos_side, active_pos_data = "short", short_data
        if active_pos_side and active_pos_data:
            style = "bold bright_green" if active_pos_side == "long" else "bold bright_red"
            pos_display_text = Text(f"{active_pos_side.upper()}: ",style=style); pos_display_text.append("Qty=",style=style).append(self._format_decimal_for_rich(active_pos_data.get("qty"),amount_display_dp))
            pos_display_text.append(" | EntryPx=",style="dim").append(self._format_decimal_for_rich(active_pos_data.get("entry_price"),price_display_dp))
            pos_display_text.append(" | PnL=",style="dim").append(self._format_decimal_for_rich(active_pos_data.get("unrealized_pnl"),4,add_commas=True,highlight_negative=True))
            
            prot_status_text = Text(" | Protection: ",style="dim"); exchange_prot_desc_parts = []
            exch_sl, exch_tp, exch_tsl_active, exch_tsl_trigger_px = active_pos_data.get("stop_loss_price"), active_pos_data.get("take_profit_price"), active_pos_data.get("is_tsl_active",False), active_pos_data.get("tsl_trigger_price")
            if exch_tsl_active:
                exchange_prot_desc_parts.append(Text("TSL",style="bright_magenta"))
                if exch_tsl_trigger_px:
                    exchange_prot_desc_parts.append(Text(f"(ActPx:{self._format_decimal_for_rich(exch_tsl_trigger_px,price_display_dp).plain})",style="magenta"))
            elif exch_sl or exch_tp:
                if exch_sl: exchange_prot_desc_parts.append(Text(f"SL:{self._format_decimal_for_rich(exch_sl,price_display_dp).plain}",style="bright_yellow"))
                if exch_tp: exchange_prot_desc_parts.append(Text(f"TP:{self._format_decimal_for_rich(exch_tp,price_display_dp).plain}",style="bright_yellow"))
            if not exchange_prot_desc_parts: exchange_prot_desc_parts.append(Text("None",style="dim"))
            prot_status_text.append("Exch:").append(Text(" ").join(exchange_prot_desc_parts)); prot_status_text.append(" LocalTrk:").append(Text(str(protection_status_tracker.get(active_pos_side)) if protection_status_tracker.get(active_pos_side) else "None",style="blue" if protection_status_tracker.get(active_pos_side) else "dim"))
            mismatch = (exch_tsl_active and protection_status_tracker.get(active_pos_side)!="ACTIVE_TSL") or ((exch_sl or exch_tp) and not exch_tsl_active and protection_status_tracker.get(active_pos_side)!="ACTIVE_SLTP") or (not exch_tsl_active and not exch_sl and not exch_tp and protection_status_tracker.get(active_pos_side) is not None)
            if mismatch: prot_status_text.append(Text(" [TrackerMismatch?]",style="bold bright_yellow"))
            pos_display_text.append(prot_status_text)
        panel_content.append(pos_display_text); panel_content.append("\n"); panel_content.append("---\n", style="dim")

        panel_content.append("Signal/Status: ", style="bold bright_cyan"); status_reason = str(signal_check_result_or_status.get("reason","No status info")); status_style_key="dim"
        if signal_check_result_or_status.get("long",False) or "Long Signal" in status_reason or "ENTERED_long" in status_reason or "Orig&VT Long" in status_reason or "VT Long" in status_reason: status_style_key="bold bright_green"
        elif signal_check_result_or_status.get("short",False) or "Short Signal" in status_reason or "ENTERED_short" in status_reason or "Orig&VT Short" in status_reason or "VT Short" in status_reason: status_style_key="bold bright_red"
        elif "Blocked" in status_reason or "FAIL:" in status_reason.upper() or "EmergencyClose" in status_reason or "Conflict" in status_reason: status_style_key="yellow"
        elif "CLOSED_" in status_reason or "HOLDING_" in status_reason or "INFO:" in status_reason: status_style_key="bright_blue"
        elif not any(s in status_reason for s in ["No Signal:", "Initializing", "Processing..."]): status_style_key="white"
        
        wrapped_status_reason = "\n             ".join(textwrap.wrap(status_reason, width=max(20, console.width - 20), subsequent_indent="")) # Adjust width dynamically
        panel_content.append(Text(wrapped_status_reason, style=status_style_key))
        console.print(Panel(panel_content, title=f"[bold bright_magenta]{panel_title_str}[/]", border_style="bright_blue", expand=False, padding=(1,2)))

# --- Trading Bot Class ---
class TradingBot:
    def __init__(self):
        logger.info(f"{Style.BRIGHT}{Fore.MAGENTA}--- Initializing Pyrmethus v4.5.8 (Neon Nexus - VT Edition) ---{Style.RESET_ALL}")
        self.config = TradingConfig()
        try:
            self.exchange_manager = ExchangeManager(self.config)
            self.indicator_calculator = IndicatorCalculator(self.config)
            self.signal_generator = SignalGenerator(self.config)
            self.order_manager = OrderManager(self.config, self.exchange_manager)
        except ValueError as ve: logger.critical(f"{Style.BRIGHT}{Fore.RED}TradingBot initialization failed (Component Init Error): {ve}. Halting.{Style.RESET_ALL}"); sys.exit(1)
        except Exception as e: logger.critical(f"{Style.BRIGHT}{Fore.RED}Unexpected critical error during TradingBot component initialization: {e}. Halting.{Style.RESET_ALL}", exc_info=True); sys.exit(1)
        self.status_display = StatusDisplay(self.config)
        self.shutdown_requested = False
        self._setup_signal_handlers()
        logger.info(f"{Style.BRIGHT}{Fore.GREEN}Pyrmethus components initialized successfully. Ready to conjure trades.{Style.RESET_ALL}")

    def _setup_signal_handlers(self):
        signals_to_handle = [signal.SIGINT, signal.SIGTERM]
        for sig in signals_to_handle:
            try: signal.signal(sig, self._signal_handler_callback); logger.debug(f"Signal handler for {signal.Signals(sig).name} set up.")
            except (ValueError, OSError, AttributeError, Exception) as e: logger.warning(f"{Fore.YELLOW}Could not set OS signal handler for {sig} (e.g., running on Windows or restricted environment): {e}{Style.RESET_ALL}")

    def _signal_handler_callback(self, sig_num: int, frame: Optional[Any]):
        sig_name = signal.Signals(sig_num).name if hasattr(signal, "Signals") else f"Signal {sig_num}" # More robust name fetching
        if not self.shutdown_requested:
            console.print(f"\n[bold yellow]Signal {sig_name} received. Initiating graceful shutdown... Please wait.[/]"); logger.warning(f"Signal {sig_name} received. Initiating graceful shutdown...")
            self.shutdown_requested = True
        else: logger.warning("Shutdown sequence already in progress. Ignoring additional signal.")

    def _display_startup_info(self):
        # Use global log_level_display_name which is set during logging setup
        vt_strategy_status = f"VT Strategy Enabled: {self.config.vt_enable}"
        if self.config.vt_enable:
            vt_strategy_status += (f" (TrendEMA:{self.config.vt_trend_ema_period}, VWMA:{self.config.vt_vwma_period}, "
                                   f"VolLookback:{self.config.vt_volume_spike_lookback}, VolMult:{self.config.vt_volume_spike_multiplier.normalize()})")
        
        console.print(Panel(Text(
            f"Symbol: {self.config.symbol}\nInterval: {self.config.interval}\nMarket Type: {self.config.market_type} (Category: {self.config.bybit_v5_category})\n"
            f"Position Index: {self.config.position_idx} (0=One-Way, 1=HedgeBuy, 2=HedgeSell)\nRisk Per Trade: {self.config.risk_percentage * 100:.3f}%\n"
            f"SL/TP Multipliers (ATR): SL={self.config.sl_atr_multiplier.normalize()}, TP={self.config.tp_atr_multiplier.normalize()}\n"
            f"TSL Activation (ATR Mult): {self.config.tsl_activation_atr_multiplier.normalize()}, TSL Percent: {self.config.trailing_stop_percent.normalize()}%\n"
            f"Trade Only With Trend (Original Strategy): {self.config.trade_only_with_trend}\n{vt_strategy_status}\n"
            f"Journaling Enabled: {self.config.enable_journaling} (File: '{self.config.journal_file_path}')\n"
            f"Log Level: {log_level_display_name}", style="bright_white"), title="[bold cyan]Pyrmethus Configuration Summary[/]", border_style="cyan", expand=False))

    def run(self):
        self._display_startup_info()
        termux_notify(f"Pyrmethus Started", f"Trading {self.config.symbol} on {self.config.interval} interval.")
        cycle_count = 0
        while not self.shutdown_requested:
            cycle_count += 1; cycle_start_time_monotonic = time.monotonic()
            logger.debug(f"{Fore.BLUE}--- Starting Trading Cycle {cycle_count} ---{Style.RESET_ALL}")
            try: self.trading_spell_cycle(cycle_count)
            except KeyboardInterrupt: logger.warning("\nKeyboardInterrupt detected in main loop. Initiating shutdown."); self.shutdown_requested = True; break
            except ccxt.AuthenticationError as auth_err: logger.critical(f"{Style.BRIGHT}{Fore.RED}CRITICAL AUTH ERROR in cycle {cycle_count}: {auth_err}. Halting.{Style.RESET_ALL}",exc_info=False); termux_notify("Pyrmethus CRITICAL ERROR",f"Auth failed: {str(auth_err)[:100]}"); self.shutdown_requested=True; break
            except SystemExit as se: logger.warning(f"SystemExit (code {se.code}) encountered in trading cycle. Terminating."); self.shutdown_requested=True; break
            except Exception as cycle_err:
                logger.error(f"{Style.BRIGHT}{Fore.RED}Unhandled exception in trading cycle {cycle_count}: {cycle_err}{Style.RESET_ALL}", exc_info=True); termux_notify("Pyrmethus Cycle Error", f"Exception in cycle {cycle_count}. Check logs.")
                sleep_duration_after_error = self.config.loop_sleep_seconds * 2; logger.info(f"Sleeping for {sleep_duration_after_error}s after cycle error before retrying cycle logic."); time.sleep(sleep_duration_after_error); continue
            
            cycle_duration_seconds = time.monotonic() - cycle_start_time_monotonic; sleep_needed_seconds = max(0, self.config.loop_sleep_seconds - cycle_duration_seconds)
            logger.debug(f"Cycle {cycle_count} completed in {cycle_duration_seconds:.2f}s.")
            if not self.shutdown_requested and sleep_needed_seconds > 0:
                logger.debug(f"Sleeping for {sleep_needed_seconds:.2f} seconds until next cycle..."); sleep_end_time = time.monotonic() + sleep_needed_seconds
                try:
                    while time.monotonic() < sleep_end_time and not self.shutdown_requested: time.sleep(min(0.5, sleep_needed_seconds))
                except KeyboardInterrupt: logger.warning("\nKeyboardInterrupt during sleep. Initiating shutdown."); self.shutdown_requested = True
            if self.shutdown_requested: logger.info("Shutdown requested. Exiting main trading loop."); break
        self.graceful_shutdown()
        console.print(f"\n[bold bright_cyan]Pyrmethus ({self.config.symbol}) has completed its session and returned to the ether.[/]")

    def trading_spell_cycle(self, cycle_num: int):
        current_cycle_status_dict: Dict[str, Any] = {"reason": f"Cycle {cycle_num} Processing..."}
        ohlcv_df = self.exchange_manager.fetch_ohlcv()
        if ohlcv_df is None or ohlcv_df.empty:
            logger.error(f"{Fore.RED}Cycle {cycle_num} Aborted: Failed to fetch OHLCV data.{Style.RESET_ALL}"); current_cycle_status_dict = {"reason": f"FAIL_CYCLE_{cycle_num}:FETCH_OHLCV_DATA"}
            self.status_display.print_status_panel(cycle_num, None, None, None, None, None, current_cycle_status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info); return
        try:
            latest_candle_data = ohlcv_df.iloc[-1]; current_market_price = safe_decimal(latest_candle_data["close"]); last_candle_timestamp = ohlcv_df.index[-1].to_pydatetime()
            if current_market_price.is_nan() or current_market_price <= Decimal("0"): raise ValueError(f"Invalid latest close price from OHLCV: {current_market_price.normalize() if not current_market_price.is_nan() else 'NaN'}")
            logger.debug(f"Latest Candle: Ts={last_candle_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}, Price={current_market_price.normalize()}")
        except (IndexError, KeyError, ValueError, TypeError) as e:
            logger.error(f"{Fore.RED}Cycle {cycle_num} Aborted: Error processing latest candle data: {e}{Style.RESET_ALL}"); current_cycle_status_dict = {"reason": f"FAIL_CYCLE_{cycle_num}:PROCESS_LATEST_CANDLE ({e})"}
            self.status_display.print_status_panel(cycle_num, None, None, None, None, None, current_cycle_status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info); return

        indicators = self.indicator_calculator.calculate_indicators(ohlcv_df)
        if not indicators:
            logger.error(f"{Fore.RED}Cycle {cycle_num} Aborted: Failed to calculate indicators.{Style.RESET_ALL}"); current_cycle_status_dict = {"reason": f"FAIL_CYCLE_{cycle_num}:CALCULATE_INDICATORS"}
            self.status_display.print_status_panel(cycle_num, last_candle_timestamp, current_market_price, None, None, None, current_cycle_status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info); return
        
        indicators["close_price"] = current_market_price # Add current price to indicators for convenience in signal checks

        total_equity, _ = self.exchange_manager.get_balance()
        if total_equity is None or total_equity.is_nan() or total_equity <= Decimal("0"):
            logger.error(f"{Fore.RED}Cycle {cycle_num} Aborted: Failed to fetch valid total equity (value: {total_equity}) or equity is zero/negative.{Style.RESET_ALL}"); current_cycle_status_dict = {"reason": f"FAIL_CYCLE_{cycle_num}:FETCH_EQUITY_INVALID"}
            self.status_display.print_status_panel(cycle_num, last_candle_timestamp, current_market_price, indicators, None, total_equity, current_cycle_status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info); return

        current_positions_summary = self.exchange_manager.get_current_position()
        if current_positions_summary is None:
            logger.error(f"{Fore.RED}Cycle {cycle_num} Aborted: Failed to fetch current position state.{Style.RESET_ALL}"); current_cycle_status_dict = {"reason": f"FAIL_CYCLE_{cycle_num}:FETCH_POSITION"}
            self.status_display.print_status_panel(cycle_num, last_candle_timestamp, current_market_price, indicators, None, total_equity, current_cycle_status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info); return

        active_pos_side_logical: Optional[str] = None; active_pos_details: Optional[Dict[str,Any]] = None
        if current_positions_summary.get("long",{}): active_pos_side_logical, active_pos_details = "long", current_positions_summary["long"]
        elif current_positions_summary.get("short",{}): active_pos_side_logical, active_pos_details = "short", current_positions_summary["short"]

        if active_pos_side_logical and active_pos_details:
            pos_qty_abs = safe_decimal(active_pos_details.get("qty")); pos_entry_price = safe_decimal(active_pos_details.get("entry_price")); current_atr = indicators.get("atr", Decimal("NaN")) # type: ignore
            if (self.order_manager.protection_tracker.get(active_pos_side_logical) == "ACTIVE_SLTP" and not any(v.is_nan() or v <= Decimal("0") for v in [pos_entry_price, current_market_price, current_atr])): # type: ignore
                self.order_manager.manage_trailing_stop(active_pos_side_logical, pos_entry_price, current_market_price, current_atr) # type: ignore
                if self.order_manager.protection_tracker.get(active_pos_side_logical) == "ACTIVE_TSL":
                    logger.debug("Re-fetching position summary after TSL management for display."); current_positions_summary = self.exchange_manager.get_current_position()
                    if current_positions_summary: active_pos_details = current_positions_summary.get(active_pos_side_logical, {}) # type: ignore
                    else: active_pos_details = None 
            
            # Check for strategy exit signals (Original or VT) if TSL is not yet the primary manager
            # If TSL is active, it's assumed to be managing the exit. This behavior can be customized.
            if self.order_manager.protection_tracker.get(active_pos_side_logical) != "ACTIVE_TSL":
                exit_reason_signal = self.signal_generator.check_exit_signals(active_pos_side_logical, indicators)
                if exit_reason_signal:
                    logger.trade(f"Attempting to close {active_pos_side_logical.upper()} position due to: {exit_reason_signal}")
                    if not pos_qty_abs.is_nan() and pos_qty_abs > Decimal("0"): # type: ignore
                        close_success = self.order_manager.close_position(active_pos_side_logical, pos_qty_abs, reason=exit_reason_signal) # type: ignore
                        current_cycle_status_dict = {"reason": f"CLOSED_{active_pos_side_logical.upper()}_BY_SIGNAL: {exit_reason_signal}" if close_success else f"FAIL:CLOSE_SIGNAL_{active_pos_side_logical.upper()}"}
                        current_positions_summary = self.exchange_manager.get_current_position(); self.status_display.print_status_panel(cycle_num, last_candle_timestamp, current_market_price, indicators, current_positions_summary, total_equity, current_cycle_status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info); return
                    else: logger.warning(f"Exit signal for {active_pos_side_logical.upper()} but position quantity invalid ({pos_qty_abs}). Cannot close.")
            
            current_positions_summary_after_actions = self.exchange_manager.get_current_position()
            if current_positions_summary_after_actions is None: logger.warning(f"Failed to re-fetch position state for {active_pos_side_logical} after TSL/exit checks. Status may be stale.")
            else:
                current_positions_summary = current_positions_summary_after_actions
                new_long_pos_data, new_short_pos_data = current_positions_summary.get("long",{}), current_positions_summary.get("short",{})
                if new_long_pos_data and safe_decimal(new_long_pos_data.get("qty","0")).copy_abs() >= POSITION_QTY_EPSILON: active_pos_side_logical, active_pos_details = "long", new_long_pos_data
                elif new_short_pos_data and safe_decimal(new_short_pos_data.get("qty","0")).copy_abs() >= POSITION_QTY_EPSILON: active_pos_side_logical, active_pos_details = "short", new_short_pos_data
                else:
                    if active_pos_side_logical: logger.info(f"Position {active_pos_side_logical.upper()} appears closed by exchange stop or earlier signal during cycle."); current_cycle_status_dict = {"reason": f"INFO:POS_{active_pos_side_logical.upper()}_CLOSED_EXCH_OR_SIGNAL"}; self.order_manager.protection_tracker[active_pos_side_logical.lower()] = None
                    active_pos_side_logical, active_pos_details = None, None

        if not active_pos_side_logical:
            logger.debug("Currently flat. Checking for new entry signals...")
            entry_signals = self.signal_generator.generate_signals(ohlcv_df, indicators)
            current_cycle_status_dict = entry_signals
            target_entry_order_side: Optional[str] = "buy" if entry_signals.get("long") else "sell" if entry_signals.get("short") else None
            if target_entry_order_side:
                current_atr = indicators.get("atr", Decimal("NaN")) # type: ignore
                if not current_atr.is_nan() and current_atr > Decimal("0"): # type: ignore
                    entry_success = self.order_manager.place_risked_market_order(target_entry_order_side, current_atr, total_equity, current_market_price) # type: ignore
                    entered_logical_side = "long" if target_entry_order_side == "buy" else "short"
                    current_cycle_status_dict = {"reason": f"ENTERED_{entered_logical_side.upper()}" if entry_success else f"FAIL:ENTRY_{entered_logical_side.upper()}"}
                    current_positions_summary = self.exchange_manager.get_current_position(); self.status_display.print_status_panel(cycle_num, last_candle_timestamp, current_market_price, indicators, current_positions_summary, total_equity, current_cycle_status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info); return
                else: logger.warning(f"Cannot attempt {target_entry_order_side.upper()} entry: Missing critical data (ATR: {current_atr})."); current_cycle_status_dict = {"reason": f"FAIL:ENTRY_DATA_MISSING_ATR_{target_entry_order_side.upper()}"}
        else: current_cycle_status_dict = {"reason": f"HOLDING_{active_pos_side_logical.upper()}"}
        
        self.status_display.print_status_panel(cycle_num, last_candle_timestamp, current_market_price, indicators, current_positions_summary, total_equity, current_cycle_status_dict, self.order_manager.protection_tracker, self.exchange_manager.market_info)

    def graceful_shutdown(self):
        logger.info(f"{Style.BRIGHT}{Fore.YELLOW}--- Pyrmethus Graceful Shutdown Sequence Initiated ---{Style.RESET_ALL}")
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
        sys.exit(e.code)
    except Exception as main_exception:
        log_func = logger.critical if 'logger' in globals() and hasattr(logger, 'critical') else print
        
        err_msg_prefix = "CRITICAL UNHANDLED EXCEPTION in Pyrmethus main execution:"
        if 'Fore' in globals() and 'Style' in globals() and _COLORAMA_SUCCESSFULLY_IMPORTED:
            err_msg = f"{Style.BRIGHT}{Fore.RED}{err_msg_prefix} {main_exception}{Style.RESET_ALL}"
        else:
            err_msg = f"{err_msg_prefix} {main_exception}"
        
        log_func(err_msg, exc_info=True)

        if 'termux_notify' in globals() and callable(termux_notify):
            termux_notify("Pyrmethus CRASHED", "Critical unhandled exception. Check logs!")
        sys.exit(1)
