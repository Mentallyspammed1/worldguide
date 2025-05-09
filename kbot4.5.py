#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=line-too-long, too-many-lines, logging-fstring-interpolation, too-many-instance-attributes, too-many-arguments, too-many-locals, too-many-public-methods, invalid-name, unused-argument
# fmt: off
#   ____        _       _   _                  _            _         _
#  |  _ \ _   _| |_ ___| | | | __ ___   ____ _| |_ ___  ___| |_ _ __ | |__
#  | |_) | | | | __/ _ \ | | |/ _` \ \ / / _` | __/ _ \/ __| __| '_ \| '_ \
#  |  __/| |_| | ||  __/ | | | (_| |\ V / (_| | ||  __/\__ \ |_| |_) | | | |
#  |_|    \__, |\__\___|_|_|_|\__,_| \_/ \__,_|\__\___||___/\__| .__/|_| |_|
#         |___/                                                |_|
# Pyrmethus v2.4.1 - Error Correction, Robustness Enhancements
# fmt: on
# pylint: enable=line-too-long
"""
Pyrmethus - Termux Trading Spell (v2.4.1)

Conjures market insights and executes trades on Bybit Futures using the
V5 Unified Account API via CCXT. Refactored into classes for better structure
and utilizing V5 position-based stop-loss/take-profit/trailing-stop features.

Enhancements in v2.4.1:
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
from datetime import datetime
from decimal import (
    ROUND_DOWN,
    ROUND_HALF_EVEN,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    getcontext,
)
from typing import Any, Dict, Optional, Tuple, Union

# Third-Party Imports
try:
    import ccxt
    import numpy as np
    import pandas as pd
    import requests
    from colorama import (
        Fore,
        Style,
        init as colorama_init,
    )  # Keep for basic init/fallback
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

# Initialize Colorama & Rich Console
colorama_init(autoreset=True)
console = Console()

# Set Decimal precision context
getcontext().prec = 50  # Increased precision

# --- Logging Setup ---
logger = logging.getLogger(__name__)
TRADE_LEVEL_NUM = logging.INFO + 5
if not hasattr(logging.Logger, "trade"):
    logging.addLevelName(TRADE_LEVEL_NUM, "TRADE")

    def trade_log(self, message, *args, **kws):
        if self.isEnabledFor(TRADE_LEVEL_NUM):
            self._log(TRADE_LEVEL_NUM, message, args, **kws)  # pylint: disable=protected-access

    logging.Logger.trade = trade_log  # type: ignore[attr-defined]

log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)-8s] (%(filename)s:%(lineno)d) %(message)s"
)
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logger.setLevel(log_level)
# Ensure handler is added only once
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
    # Termux `termux-toast` typically only takes the content. Title is ignored here.
    if platform.system() == "Linux" and "com.termux" in os.environ.get("PREFIX", ""):
        try:
            # Pass only content to termux-toast
            subprocess.run(
                ["termux-toast", content],
                check=False,
                timeout=5,
                capture_output=True,
                text=True,
            )
            # logger.debug(f"Termux toast sent: '{content}' (Title '{title}' ignored by toast)")
        except FileNotFoundError:
            logger.warning("Termux notify failed: 'termux-toast' not found.")
        except subprocess.TimeoutExpired:
            logger.warning("Termux notify failed: command timed out.")
        except Exception as e:
            logger.warning(f"Termux notify failed: {e}")
    # else: logger.debug("Not in Termux, skipping notification.") # Optional debug


def fetch_with_retries(
    fetch_function, *args, max_retries=3, delay_seconds=3, **kwargs
) -> Any:
    """Wraps a function call with retry logic for specific CCXT/network errors."""
    last_exception = None
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
            retry_msg = f"{Fore.YELLOW}Retryable error ({type(e).__name__}) on attempt {attempt + 1}/{max_retries + 1} for {fetch_function.__name__}: {str(e)[:150]}. Retrying in {delay_seconds}s...{Style.RESET_ALL}"
            if attempt < max_retries:
                logger.warning(retry_msg)
                time.sleep(delay_seconds)
            else:
                logger.error(
                    f"{Fore.RED}Max retries ({max_retries + 1}) reached for {fetch_function.__name__}. Last error: {e}"
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
            break  # Don't retry InsufficientFunds
        except ccxt.InvalidOrder as e:
            logger.error(f"{Fore.RED}Invalid order parameters: {e}")
            break  # Don't retry InvalidOrder
        except ccxt.OrderNotFound as e:
            logger.warning(f"{Fore.YELLOW}Order not found: {e}")
            break  # Often not retryable in context
        except ccxt.PermissionDenied as e:
            logger.error(
                f"{Fore.RED}Permission denied: {e}. Check API permissions/IP whitelisting."
            )
            break  # Not retryable
        except ccxt.ExchangeError as e:  # Catch other specific exchange errors
            logger.error(
                f"{Fore.RED}Exchange error during {fetch_function.__name__}: {e}"
            )
            last_exception = e
            if attempt < max_retries:
                logger.warning(f"Retrying in {delay_seconds}s...")
                time.sleep(delay_seconds)
            else:
                logger.error(
                    f"Max retries reached after exchange error for {fetch_function.__name__}."
                )
                break
        except Exception as e:  # Catch unexpected errors during fetch
            logger.error(
                f"{Fore.RED}Unexpected error during {fetch_function.__name__}: {e}",
                exc_info=True,
            )
            last_exception = e
            break  # Don't retry unknown errors
    # If loop finished without success, raise the last captured exception or a generic one
    if last_exception:
        raise last_exception
    else:
        raise RuntimeError(
            f"Function {fetch_function.__name__} failed after unexpected issue."
        )


# --- Configuration Class ---
class TradingConfig:
    """Loads, validates, and holds trading configuration parameters from .env."""

    def __init__(self):
        logger.debug("Loading configuration from environment variables...")
        self.symbol = self._get_env("SYMBOL", "BTC/USDT:USDT", Style.DIM)
        self.market_type = self._get_env(
            "MARKET_TYPE",
            "linear",
            Style.DIM,
            allowed_values=["linear", "inverse", "swap"],
        ).lower()
        self.bybit_v5_category = self._determine_v5_category()
        self.interval = self._get_env("INTERVAL", "1m", Style.DIM)
        # Financial parameters (Decimal)
        self.risk_percentage = self._get_env(
            "RISK_PERCENTAGE",
            "0.01",
            Fore.YELLOW,
            cast_type=Decimal,
            min_val=Decimal("0.00001"),
            max_val=Decimal("0.5"),
        )
        self.sl_atr_multiplier = self._get_env(
            "SL_ATR_MULTIPLIER",
            "1.5",
            Fore.YELLOW,
            cast_type=Decimal,
            min_val=Decimal("0.1"),
            max_val=Decimal("20.0"),
        )
        self.tp_atr_multiplier = self._get_env(
            "TP_ATR_MULTIPLIER",
            "3.0",
            Fore.YELLOW,
            cast_type=Decimal,
            min_val=Decimal("0.0"),
            max_val=Decimal("50.0"),
        )
        self.tsl_activation_atr_multiplier = self._get_env(
            "TSL_ACTIVATION_ATR_MULTIPLIER",
            "1.0",
            Fore.YELLOW,
            cast_type=Decimal,
            min_val=Decimal("0.1"),
            max_val=Decimal("20.0"),
        )
        # Trailing Stop: Distance based (percentage of entry price)
        self.trailing_stop_percent = self._get_env(
            "TRAILING_STOP_PERCENT",
            "0.5",
            Fore.YELLOW,
            cast_type=Decimal,
            min_val=Decimal("0.01"),
            max_val=Decimal("10.0"),
        )
        # Trigger types for V5 position stops
        self.sl_trigger_by = self._get_env(
            "SL_TRIGGER_BY",
            "LastPrice",
            Style.DIM,
            allowed_values=["LastPrice", "MarkPrice", "IndexPrice"],
        )
        self.tsl_trigger_by = self._get_env(
            "TSL_TRIGGER_BY",
            "LastPrice",
            Style.DIM,
            allowed_values=["LastPrice", "MarkPrice", "IndexPrice"],
        )  # TSL trigger usually same as SL
        # Position Index: 0 for hedge mode (default/recommended), 1=Buy side one-way, 2=Sell side one-way
        self.position_idx = self._get_env(
            "POSITION_IDX", "0", Style.DIM, cast_type=int, allowed_values=[0, 1, 2]
        )

        # Indicator Periods (int)
        self.trend_ema_period = self._get_env(
            "TREND_EMA_PERIOD", "12", Style.DIM, cast_type=int, min_val=5, max_val=500
        )
        self.fast_ema_period = self._get_env(
            "FAST_EMA_PERIOD", "9", Style.DIM, cast_type=int, min_val=1, max_val=200
        )
        self.slow_ema_period = self._get_env(
            "SLOW_EMA_PERIOD", "21", Style.DIM, cast_type=int, min_val=2, max_val=500
        )
        self.stoch_period = self._get_env(
            "STOCH_PERIOD", "7", Style.DIM, cast_type=int, min_val=1, max_val=100
        )
        self.stoch_smooth_k = self._get_env(
            "STOCH_SMOOTH_K", "3", Style.DIM, cast_type=int, min_val=1, max_val=10
        )
        self.stoch_smooth_d = self._get_env(
            "STOCH_SMOOTH_D", "3", Style.DIM, cast_type=int, min_val=1, max_val=10
        )
        self.atr_period = self._get_env(
            "ATR_PERIOD", "5", Style.DIM, cast_type=int, min_val=1, max_val=100
        )
        self.adx_period = self._get_env(
            "ADX_PERIOD", "14", Style.DIM, cast_type=int, min_val=2, max_val=100
        )

        # Signal Logic Thresholds (Decimal)
        self.stoch_oversold_threshold = self._get_env(
            "STOCH_OVERSOLD_THRESHOLD",
            "30",
            Fore.CYAN,
            cast_type=Decimal,
            min_val=Decimal("0"),
            max_val=Decimal("45"),
        )
        self.stoch_overbought_threshold = self._get_env(
            "STOCH_OVERBOUGHT_THRESHOLD",
            "70",
            Fore.CYAN,
            cast_type=Decimal,
            min_val=Decimal("55"),
            max_val=Decimal("100"),
        )
        self.trend_filter_buffer_percent = self._get_env(
            "TREND_FILTER_BUFFER_PERCENT",
            "0.5",
            Fore.CYAN,
            cast_type=Decimal,
            min_val=Decimal("0"),
            max_val=Decimal("5"),
        )
        self.atr_move_filter_multiplier = self._get_env(
            "ATR_MOVE_FILTER_MULTIPLIER",
            "0.5",
            Fore.CYAN,
            cast_type=Decimal,
            min_val=Decimal("0"),
            max_val=Decimal("5"),
        )
        self.min_adx_level = self._get_env(
            "MIN_ADX_LEVEL",
            "20",
            Fore.CYAN,
            cast_type=Decimal,
            min_val=Decimal("0"),
            max_val=Decimal("90"),
        )

        self.position_qty_epsilon = Decimal(
            "1E-12"
        )  # Threshold for considering a position 'flat'
        self.api_key = self._get_env("BYBIT_API_KEY", None, Fore.RED)
        self.api_secret = self._get_env("BYBIT_API_SECRET", None, Fore.RED)

        # Operational Parameters
        self.ohlcv_limit = self._get_env(
            "OHLCV_LIMIT", "200", Style.DIM, cast_type=int, min_val=50, max_val=1000
        )
        self.loop_sleep_seconds = self._get_env(
            "LOOP_SLEEP_SECONDS", "15", Style.DIM, cast_type=int, min_val=5
        )
        self.order_check_delay_seconds = self._get_env(
            "ORDER_CHECK_DELAY_SECONDS", "2", Style.DIM, cast_type=int, min_val=1
        )
        self.order_fill_timeout_seconds = self._get_env(
            "ORDER_FILL_TIMEOUT_SECONDS", "20", Style.DIM, cast_type=int, min_val=5
        )  # Longer for fills
        self.max_fetch_retries = self._get_env(
            "MAX_FETCH_RETRIES", "3", Style.DIM, cast_type=int, min_val=1, max_val=10
        )
        self.retry_delay_seconds = self._get_env(
            "RETRY_DELAY_SECONDS", "3", Style.DIM, cast_type=int, min_val=1
        )
        self.trade_only_with_trend = self._get_env(
            "TRADE_ONLY_WITH_TREND", "True", Style.DIM, cast_type=bool
        )

        # Journaling
        self.journal_file_path = self._get_env(
            "JOURNAL_FILE_PATH", "pyrmethus_trading_journal.csv", Style.DIM
        )
        self.enable_journaling = self._get_env(
            "ENABLE_JOURNALING", "True", Style.DIM, cast_type=bool
        )

        if not self.api_key or not self.api_secret:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}BYBIT_API_KEY or BYBIT_API_SECRET not found. Halting."
            )
            sys.exit(1)
        self._validate_config()
        logger.debug("Configuration loaded and validated successfully.")

    def _determine_v5_category(self) -> str:
        """Determines the Bybit V5 API category."""
        try:
            parts = self.symbol.replace(":", "/").split("/")
            if len(parts) < 2:
                raise ValueError("Symbol format must be BASE/QUOTE[:SETTLE]")
            # Settle currency is usually the last part for V5 perp symbols like BTC/USDT:USDT
            parts[-1].upper()
            category = ""
            if self.market_type == "inverse":
                category = "inverse"
            elif self.market_type in ["linear", "swap"]:
                category = "linear"  # Linear typically USDT or USDC settle
            else:
                raise ValueError(f"Unsupported MARKET_TYPE '{self.market_type}'")
            logger.info(
                f"Determined Bybit V5 API category: '{category}' for symbol '{self.symbol}' and type '{self.market_type}'"
            )
            return category
        except (ValueError, IndexError) as e:
            logger.critical(
                f"Could not parse symbol '{self.symbol}' for category: {e}. Halting.",
                exc_info=True,
            )
            sys.exit(1)

    def _validate_config(self):
        """Performs post-load validation."""
        if self.fast_ema_period >= self.slow_ema_period:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}FAST_EMA ({self.fast_ema_period}) must be < SLOW_EMA ({self.slow_ema_period}). Halting."
            )
            sys.exit(1)
        if self.trend_ema_period <= self.slow_ema_period:
            logger.warning(
                f"{Fore.YELLOW}TREND_EMA ({self.trend_ema_period}) <= SLOW_EMA ({self.slow_ema_period}). Consider increasing."
            )
        if self.stoch_oversold_threshold >= self.stoch_overbought_threshold:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}STOCH_OVERSOLD ({self.stoch_oversold_threshold.normalize()}) must be < STOCH_OVERBOUGHT ({self.stoch_overbought_threshold.normalize()}). Halting."
            )
            sys.exit(1)
        if self.tsl_activation_atr_multiplier < self.sl_atr_multiplier:
            logger.warning(
                f"{Fore.YELLOW}TSL_ACT_MULT ({self.tsl_activation_atr_multiplier.normalize()}) < SL_MULT ({self.sl_atr_multiplier.normalize()}). TSL may activate early."
            )
        if (
            self.tp_atr_multiplier > Decimal("0")
            and self.tp_atr_multiplier <= self.sl_atr_multiplier
        ):
            logger.warning(
                f"{Fore.YELLOW}TP_MULT ({self.tp_atr_multiplier.normalize()}) <= SL_MULT ({self.sl_atr_multiplier.normalize()}). Poor R:R setup."
            )

    def _get_env(
        self,
        key: str,
        default: Any,
        color: str,
        cast_type: type = str,
        min_val=None,
        max_val=None,
        allowed_values=None,
    ) -> Any:
        """Gets value from environment, casts, validates, logs."""
        value_str = os.getenv(key)
        is_secret = "SECRET" in key.upper() or "KEY" in key.upper()
        log_value = "****" if is_secret else value_str

        if value_str is None or value_str.strip() == "":
            if default is None and not is_secret:
                logger.critical(
                    f"{Fore.RED + Style.BRIGHT}Required config '{key}' not found. Halting."
                )
                sys.exit(1)
            if is_secret and default is None:
                logger.critical(
                    f"{Fore.RED + Style.BRIGHT}Required secret '{key}' not found. Halting."
                )
                sys.exit(1)
            value_str_for_cast = str(default) if default is not None else None
            if default is not None:
                logger.warning(f"{color}Using default for {key}: {default}")
        else:
            logger.info(f"{color}Found {key}: {log_value}")
            value_str_for_cast = value_str

        if value_str_for_cast is None:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}Config '{key}' has no value or default. Halting."
            )
            sys.exit(1)

        casted_value = None
        try:
            val_to_cast = str(
                value_str_for_cast
            ).strip()  # Ensure string and strip whitespace
            if cast_type == bool:
                casted_value = val_to_cast.lower() in ["true", "1", "yes", "y", "on"]
            elif cast_type == Decimal:
                casted_value = Decimal(val_to_cast)
            elif cast_type == int:
                casted_value = int(
                    Decimal(val_to_cast)
                )  # Use Decimal intermediary for float strings
            else:
                casted_value = cast_type(val_to_cast)
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.error(
                f"{Fore.RED}Cast failed for {key} ('{value_str_for_cast}' -> {cast_type.__name__}): {e}. Using default '{default}'."
            )
            try:  # Re-cast default
                default_str = str(default).strip() if default is not None else None
                if default_str is None:
                    casted_value = None
                elif cast_type == bool:
                    casted_value = default_str.lower() in [
                        "true",
                        "1",
                        "yes",
                        "y",
                        "on",
                    ]
                elif cast_type == Decimal:
                    casted_value = Decimal(default_str)
                elif cast_type == int:
                    casted_value = int(Decimal(default_str))
                else:
                    casted_value = cast_type(default_str)
            except (ValueError, TypeError, InvalidOperation) as cast_default_err:
                logger.critical(
                    f"{Fore.RED + Style.BRIGHT}Default '{default}' for {key} invalid ({cast_type.__name__}): {cast_default_err}. Halting."
                )
                sys.exit(1)

        # --- Validation ---
        if allowed_values:
            comp_value = (
                str(casted_value).lower()
                if isinstance(casted_value, str)
                else casted_value
            )
            lower_allowed = (
                [str(v).lower() for v in allowed_values]
                if isinstance(allowed_values[0], str)
                else allowed_values
            )
            if comp_value not in lower_allowed:
                logger.error(
                    f"{Fore.RED}Validation failed for {key}: Invalid value '{casted_value}'. Allowed: {allowed_values}. Using default: {default}"
                )
                # Recast default on validation failure (duplicate code, consider refactoring)
                try:
                    default_str = str(default).strip() if default is not None else None
                    if default_str is None:
                        return None
                    if cast_type == bool:
                        return default_str.lower() in ["true", "1", "yes", "y", "on"]
                    elif cast_type == Decimal:
                        return Decimal(default_str)
                    elif cast_type == int:
                        return int(Decimal(default_str))
                    else:
                        return cast_type(default_str)
                except (ValueError, TypeError, InvalidOperation) as cast_default_err:
                    logger.critical(
                        f"{Fore.RED + Style.BRIGHT}Default '{default}' for {key} invalid on fallback: {cast_default_err}. Halting."
                    )
                    sys.exit(1)

        if min_val is not None and casted_value < min_val:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}Validation failed for {key}: {casted_value} < min {min_val}. Halting."
            )
            sys.exit(1)
        if max_val is not None and casted_value > max_val:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}Validation failed for {key}: {casted_value} > max {max_val}. Halting."
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
                    "defaultType": self.config.market_type,  # 'linear' or 'inverse'
                    "adjustForTimeDifference": True,
                    "recvWindow": 10000,
                    "brokerId": "TermuxPyrmV5",  # Custom ID
                    "createMarketBuyOrderRequiresPrice": False,  # Allow market buys without price arg
                    "v5": True,  # Explicitly enable V5 if needed by CCXT version
                    # 'enableRateLimit': True, # Default True usually
                },
            }
            # Handle sandbox endpoint if needed
            # if os.getenv("USE_SANDBOX", "false").lower() == "true": exchange_params['urls'] = {'api': 'https://api-testnet.bybit.com'}

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
        """Loads and caches market information for the symbol."""
        if not self.exchange:
            return None
        try:
            logger.debug(f"Loading market info for {self.config.symbol}...")
            self.exchange.load_markets(True)  # Force reload
            market = self.exchange.market(self.config.symbol)
            if not market:
                raise ccxt.ExchangeError(f"Market {self.config.symbol} not found.")
            # Extract precision safely
            amount_precision = safe_decimal(
                market.get("precision", {}).get("amount"), default=Decimal("8")
            )  # Base precision
            price_precision = safe_decimal(
                market.get("precision", {}).get("price"), default=Decimal("8")
            )  # Price precision
            # Convert precision decimals to number of decimal places (integer)
            amount_dp = (
                abs(amount_precision.as_tuple().exponent)
                if not amount_precision.is_nan()
                else 8
            )
            price_dp = (
                abs(price_precision.as_tuple().exponent)
                if not price_precision.is_nan()
                else 8
            )

            # Store precision as integer decimal places
            market["precision_dp"] = {"amount": amount_dp, "price": price_dp}
            # Store minimum tick size as Decimal
            market["tick_size"] = Decimal("1e-" + str(price_dp))

            min_amt = safe_decimal(
                market.get("limits", {}).get("amount", {}).get("min"),
                default=Decimal("NaN"),
            )
            logger.info(
                f"Market info loaded: ID={market.get('id')}, Precision(AmtDP={amount_dp}, PriceDP={price_dp}), Limits(MinAmt={min_amt.normalize() if not min_amt.is_nan() else 'N/A'})"
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
        if (
            self.market_info
            and "precision_dp" in self.market_info
            and "price" in self.market_info["precision_dp"]
        ):
            precision = self.market_info["precision_dp"]["price"]
            price_decimal = safe_decimal(price)
            if price_decimal.is_nan():
                return "NaN"  # Return NaN string if input was bad
            quantizer = Decimal("1e-" + str(precision))
            return str(price_decimal.quantize(quantizer, rounding=ROUND_HALF_EVEN))
        else:
            logger.warning(
                "Market info/price precision unavailable, using default formatting."
            )
            return str(safe_decimal(price))

    def format_amount(
        self, amount: Union[Decimal, str, float, int], rounding_mode=ROUND_DOWN
    ) -> str:
        """Formats amount according to market precision using specified rounding."""
        if (
            self.market_info
            and "precision_dp" in self.market_info
            and "amount" in self.market_info["precision_dp"]
        ):
            precision = self.market_info["precision_dp"]["amount"]
            amount_decimal = safe_decimal(amount)
            if amount_decimal.is_nan():
                return "NaN"  # Return NaN string if input was bad
            quantizer = Decimal("1e-" + str(precision))
            return str(amount_decimal.quantize(quantizer, rounding=rounding_mode))
        else:
            logger.warning(
                "Market info/amount precision unavailable, using default formatting."
            )
            return str(safe_decimal(amount))

    def fetch_ohlcv(self) -> Optional[pd.DataFrame]:
        """Fetches OHLCV data with retries."""
        if not self.exchange:
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
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            # Convert OHLCV to Decimal robustly
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].apply(safe_decimal)  # Use safe_decimal utility
            logger.debug(f"Fetched {len(df)} candles. Last timestamp: {df.index[-1]}")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV data: {e}", exc_info=True)
            return None

    def get_balance(self) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Fetches total and available balance for the settlement currency."""
        if not self.exchange or not self.market_info:
            return None, None
        settle_currency = self.market_info.get("settle")
        if not settle_currency:
            logger.error("Settle currency not found in market info.")
            return None, None
        logger.debug(
            f"Fetching balance for {settle_currency} (Cat: {self.config.bybit_v5_category})..."
        )
        try:
            # V5 balance needs category and accountType
            params = {
                "category": self.config.bybit_v5_category,
                "accountType": "UNIFIED",
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
            # Parse V5 response structure (check CCXT unified structure first)
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
                            if item.get("accountType") == "UNIFIED"
                        ),
                        None,
                    )
                    if unified_acc and "coin" in unified_acc:
                        for coin_info in unified_acc["coin"]:
                            if coin_info.get("coin") == settle_currency:
                                # Use safe_decimal for robust conversion
                                total_equity = safe_decimal(coin_info.get("equity"))
                                available_balance = safe_decimal(
                                    coin_info.get("availableToWithdraw")
                                )  # Use 'availableToWithdraw' for free balance
                                break
            # Fallback to top-level structure if unified parsing failed or API changed
            if total_equity.is_nan() and settle_currency in balance_data["total"]:
                total_equity = safe_decimal(balance_data[settle_currency].get("total"))
                available_balance = safe_decimal(
                    balance_data[settle_currency].get("free")
                )

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

    def get_current_position(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Fetches current position details for the symbol using V5 API."""
        if not self.exchange or not self.market_info:
            return None
        market_id = self.market_info.get("id")
        logger.debug(
            f"Fetching position for {self.config.symbol} (ID: {market_id}, Cat: {self.config.bybit_v5_category})..."
        )
        positions_dict = {"long": {}, "short": {}}  # Default empty
        try:
            params = {"category": self.config.bybit_v5_category, "symbol": market_id}
            position_data = fetch_with_retries(
                self.exchange.fetch_positions,  # Pass function
                symbols=[self.config.symbol],  # V5 fetch_positions often prefers list
                params=params,
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
            )
            # logger.debug(f"Raw position data: {position_data}") # Verbose

            if not position_data:
                logger.debug("No position data returned.")
                return positions_dict

            # V5 returns a list, find the matching symbol and position index
            pos_info = None
            for p in position_data:
                info = p.get("info", {})
                # Match symbol AND position index (0=Hedge, 1=Buy One-Way, 2=Sell One-Way)
                if (
                    info.get("symbol") == market_id
                    and int(info.get("positionIdx", -1)) == self.config.position_idx
                ):
                    pos_info = info
                    break

            if not pos_info:
                logger.debug(
                    f"No position found for symbol {market_id} and posIdx {self.config.position_idx}."
                )
                return positions_dict

            # Use safe_decimal for robust conversion from API strings
            qty = safe_decimal(pos_info.get("size", "0"))
            side = pos_info.get("side", "None").lower()  # 'Buy'/'Sell'/'None'
            entry_price = safe_decimal(pos_info.get("avgPrice", "0"))
            liq_price = safe_decimal(pos_info.get("liqPrice", "0"))
            unrealized_pnl = safe_decimal(pos_info.get("unrealisedPnl", "0"))
            sl_price = safe_decimal(pos_info.get("stopLoss", "0"))
            tp_price = safe_decimal(pos_info.get("takeProfit", "0"))
            tsl_active_price = safe_decimal(
                pos_info.get("trailingStop", "0")
            )  # Non-zero value often indicates active TSL trigger price

            # CORRECTION: Check if liq_price is NaN before comparing with 0
            liq_price_val = Decimal("NaN")  # Default to NaN
            if not liq_price.is_nan():  # Only compare if it's a valid number
                if liq_price > 0:
                    liq_price_val = liq_price

            position_details = {
                "qty": qty if not qty.is_nan() else Decimal("0"),  # Default 0 if NaN
                "entry_price": entry_price
                if not entry_price.is_nan() and entry_price > 0
                else Decimal("NaN"),  # Use NaN if entry is 0 or NaN
                "liq_price": liq_price_val,  # Use the validated liq_price_val
                "unrealized_pnl": unrealized_pnl
                if not unrealized_pnl.is_nan()
                else Decimal("0"),
                "side": side,
                "info": pos_info,  # Store raw info
                "stop_loss_price": sl_price
                if not sl_price.is_nan() and sl_price > 0
                else None,
                "take_profit_price": tp_price
                if not tp_price.is_nan() and tp_price > 0
                else None,
                "is_tsl_active": not tsl_active_price.is_nan()
                and tsl_active_price > 0,  # Check if trigger price is set and valid
            }

            # Populate the correct side in the dictionary
            if (
                side == "buy"
                and position_details["qty"].copy_abs()
                >= self.config.position_qty_epsilon
            ):
                positions_dict["long"] = position_details
                logger.debug(
                    f"Found LONG position: Qty={position_details['qty'].normalize()}, Entry={position_details['entry_price'].normalize() if not position_details['entry_price'].is_nan() else 'N/A'}"
                )
            elif (
                side == "sell"
                and position_details["qty"].copy_abs()
                >= self.config.position_qty_epsilon
            ):
                positions_dict["short"] = position_details
                logger.debug(
                    f"Found SHORT position: Qty={position_details['qty'].normalize()}, Entry={position_details['entry_price'].normalize() if not position_details['entry_price'].is_nan() else 'N/A'}"
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
            logger.error(f"{Fore.RED}No DataFrame for indicators.")
            return None
        try:
            req_cols = [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]  # Volume needed? No, but keep standard columns
            if not all(c in df.columns for c in req_cols):
                logger.error(
                    f"{Fore.RED}DataFrame missing columns: {[c for c in req_cols if c not in df.columns]}"
                )
                return None

            # Work with float copies for performance, convert back to Decimal at the end
            df_calc = df[req_cols].copy()
            # Convert Decimal NaNs back to np.nan for calculations if present
            # CORRECTION: Use map instead of deprecated applymap
            df_calc = df_calc.map(
                lambda x: np.nan if isinstance(x, Decimal) and x.is_nan() else x
            )
            df_calc.dropna(
                subset=["open", "high", "low", "close"], inplace=True
            )  # Drop rows with NaN in OHLC

            if df_calc.empty:
                logger.error(f"{Fore.RED}DataFrame empty after NaN drop.")
                return None

            close = df_calc["close"].astype(float).values
            high = df_calc["high"].astype(float).values
            low = df_calc["low"].astype(float).values
            index = df_calc.index

            # Check Data Length
            min_required_len = (
                max(
                    self.config.slow_ema_period,
                    self.config.trend_ema_period,
                    self.config.stoch_period
                    + self.config.stoch_smooth_k
                    + self.config.stoch_smooth_d
                    - 2,
                    self.config.atr_period,
                    self.config.adx_period * 2,
                )
                + 1
            )  # ADX needs more data + buffer
            if len(df_calc) < min_required_len:
                logger.error(
                    f"{Fore.RED}Insufficient data ({len(df_calc)} < {min_required_len}) for indicators."
                )
                return None

            # Calculations (float using pandas/numpy)
            close_s = pd.Series(close, index=index)
            high_s = pd.Series(high, index=index)
            low_s = pd.Series(low, index=index)
            fast_ema_s = close_s.ewm(
                span=self.config.fast_ema_period, adjust=False
            ).mean()
            slow_ema_s = close_s.ewm(
                span=self.config.slow_ema_period, adjust=False
            ).mean()
            trend_ema_s = close_s.ewm(
                span=self.config.trend_ema_period, adjust=False
            ).mean()

            # Stochastic
            low_min = low_s.rolling(window=self.config.stoch_period).min()
            high_max = high_s.rolling(window=self.config.stoch_period).max()
            stoch_range = high_max - low_min
            stoch_k_raw = np.where(
                stoch_range > 1e-12, 100 * (close_s - low_min) / stoch_range, 50.0
            )  # Avoid NaN, default 50
            stoch_k_raw_s = pd.Series(stoch_k_raw, index=index).fillna(
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

            # ATR (Wilder's)
            prev_close = close_s.shift(1)
            tr1 = high_s - low_s
            tr2 = (high_s - prev_close).abs()
            tr3 = (low_s - prev_close).abs()
            tr_s = (
                pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(0)
            )  # Fill initial NaN
            atr_s = tr_s.ewm(alpha=1 / self.config.atr_period, adjust=False).mean()

            # ADX, +DI, -DI
            adx_s, pdi_s, mdi_s = self._calculate_adx(
                high_s, low_s, close_s, atr_s, self.config.adx_period
            )

            # Extract Latest Values & Convert to Decimal
            def get_latest_decimal(series: pd.Series, name: str) -> Decimal:
                if series.empty or series.iloc[-1] is None or pd.isna(series.iloc[-1]):
                    return Decimal("NaN")
                try:
                    return Decimal(str(series.iloc[-1]))
                except (InvalidOperation, TypeError):
                    logger.error(
                        f"Failed converting {name} value {series.iloc[-1]} to Decimal."
                    )
                    return Decimal("NaN")

            indicators_out = {
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

            # Calculate Stochastic Cross Signals using latest Decimal values
            k_last = indicators_out["stoch_k"]
            d_last = indicators_out["stoch_d"]
            # CORRECTION: Simplify getting previous values
            k_prev_val = stoch_k_s.iloc[-2] if len(stoch_k_s) >= 2 else None
            d_prev_val = stoch_d_s.iloc[-2] if len(stoch_d_s) >= 2 else None
            k_prev = safe_decimal(k_prev_val, Decimal("NaN"))
            d_prev = safe_decimal(d_prev_val, Decimal("NaN"))

            stoch_kd_bullish = False
            stoch_kd_bearish = False
            if not any(v.is_nan() for v in [k_last, d_last, k_prev, d_prev]):
                crossed_above = (k_last > d_last) and (k_prev <= d_prev)
                crossed_below = (k_last < d_last) and (k_prev >= d_prev)
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

            # Final check for critical NaNs
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
            failed = [
                k
                for k in critical_keys
                if indicators_out.get(k, Decimal("NaN")).is_nan()
            ]
            if failed:
                logger.error(
                    f"{Fore.RED}Critical indicators failed (NaN): {', '.join(failed)}."
                )
                return None

            logger.info(f"{Fore.GREEN}Indicator patterns woven successfully.")
            return indicators_out

        except Exception as e:
            logger.error(
                f"{Fore.RED}Failed weaving indicator patterns: {e}", exc_info=True
            )
            return None

    def _calculate_adx(
        self,
        high_s: pd.Series,
        low_s: pd.Series,
        close_s: pd.Series,
        atr_s: pd.Series,
        period: int,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Helper to calculate ADX, +DI, -DI."""
        # Calculate +DM, -DM
        move_up = high_s.diff()
        move_down = low_s.diff().mul(-1)  # diff is current - previous
        plus_dm = np.where((move_up > move_down) & (move_up > 0), move_up, 0)
        minus_dm = np.where((move_down > move_up) & (move_down > 0), move_down, 0)
        # Smoothed DM and ATR
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
        # Calculate +DI, -DI (handle division by zero if ATR is zero)
        pdi_s = np.where(atr_s > 1e-12, 100 * plus_dm_s / atr_s, 0)
        mdi_s = np.where(atr_s > 1e-12, 100 * minus_dm_s / atr_s, 0)
        pdi_s = pd.Series(pdi_s, index=high_s.index).fillna(0)
        mdi_s = pd.Series(mdi_s, index=high_s.index).fillna(0)
        # Calculate DX
        di_diff = (pdi_s - mdi_s).abs()
        di_sum = pdi_s + mdi_s
        dx_s = np.where(di_sum > 1e-12, 100 * di_diff / di_sum, 0)
        dx_s = pd.Series(dx_s, index=high_s.index).fillna(0)
        # Calculate ADX (Smoothed DX)
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
        """Generates 'long'/'short' entry signals and reason."""
        long_signal, short_signal = False, False
        signal_reason = "No signal - Conditions not met"
        if not indicators:
            return {"long": False, "short": False, "reason": "Indicators missing"}
        if df_last_candles is None or len(df_last_candles) < 2:
            return {
                "long": False,
                "short": False,
                "reason": "Insufficient candle data (<2)",
            }

        try:
            latest = df_last_candles.iloc[-1]
            current_price = safe_decimal(latest["close"])
            prev_close = (
                safe_decimal(df_last_candles.iloc[-2]["close"])
                if not pd.isna(df_last_candles.iloc[-2]["close"])
                else Decimal("NaN")
            )
            if current_price.is_nan() or current_price <= 0:
                return {
                    "long": False,
                    "short": False,
                    "reason": "Invalid price (NaN or <= 0)",
                }

            # Get indicator values safely
            k = indicators.get("stoch_k", Decimal("NaN"))
            fast_ema = indicators.get("fast_ema", Decimal("NaN"))
            slow_ema = indicators.get("slow_ema", Decimal("NaN"))
            trend_ema = indicators.get("trend_ema", Decimal("NaN"))
            atr = indicators.get("atr", Decimal("NaN"))
            kd_bull = indicators.get("stoch_kd_bullish", False)
            kd_bear = indicators.get("stoch_kd_bearish", False)
            adx = indicators.get("adx", Decimal("NaN"))
            pdi = indicators.get("pdi", Decimal("NaN"))
            mdi = indicators.get("mdi", Decimal("NaN"))

            req_vals = {
                "k": k,
                "fast_ema": fast_ema,
                "slow_ema": slow_ema,
                "trend_ema": trend_ema,
                "atr": atr,
                "adx": adx,
                "pdi": pdi,
                "mdi": mdi,
            }
            nan_keys = [
                n for n, v in req_vals.items() if isinstance(v, Decimal) and v.is_nan()
            ]
            if nan_keys:
                return {
                    "long": False,
                    "short": False,
                    "reason": f"Required indicator(s) NaN: {', '.join(nan_keys)}",
                }

            # --- Define Conditions ---
            ema_bullish_cross = fast_ema > slow_ema
            ema_bearish_cross = fast_ema < slow_ema
            trend_buffer = trend_ema.copy_abs() * (
                self.config.trend_filter_buffer_percent / 100
            )
            price_above_trend = current_price > trend_ema - trend_buffer
            price_below_trend = current_price < trend_ema + trend_buffer
            stoch_long_cond = (k < self.config.stoch_oversold_threshold) or kd_bull
            stoch_short_cond = (k > self.config.stoch_overbought_threshold) or kd_bear
            # ATR Move Filter
            sig_move, atr_reason = True, "Move Filter OFF"
            if self.config.atr_move_filter_multiplier > 0:
                if atr <= 0:
                    sig_move, atr_reason = (
                        False,
                        f"Move Filter Skipped (Invalid ATR: {atr.normalize()})",
                    )
                elif prev_close.is_nan():
                    sig_move, atr_reason = False, "Move Filter Skipped (Prev Close NaN)"
                else:
                    move = (current_price - prev_close).copy_abs()
                    thresh = atr * self.config.atr_move_filter_multiplier
                    sig_move = move > thresh
                    atr_reason = f"Move({move.normalize()}) {' > ' if sig_move else ' <= '} {self.config.atr_move_filter_multiplier.normalize()}xATR({thresh.normalize()})"
            # ADX Filter
            adx_trending = adx > self.config.min_adx_level
            adx_long_confirm = adx_trending and pdi > mdi
            adx_short_confirm = adx_trending and mdi > pdi
            adx_reason = (
                f"ADX({adx:.1f})>{self.config.min_adx_level.normalize()} & "
                + (
                    f"+DI({pdi:.1f})>-DI({mdi:.1f})"
                    if adx_long_confirm
                    else f"-DI({mdi:.1f})>+DI({pdi:.1f})"
                    if adx_short_confirm
                    else "DI Invalid"
                )
                if adx_trending
                else f"ADX({adx:.1f})<={self.config.min_adx_level.normalize()}"
            )

            # --- Combine Logic ---
            potential_long = (
                ema_bullish_cross and stoch_long_cond and sig_move and adx_long_confirm
            )
            potential_short = (
                ema_bearish_cross
                and stoch_short_cond
                and sig_move
                and adx_short_confirm
            )

            reason_parts = []
            if potential_long:
                trend_check = (
                    price_above_trend if self.config.trade_only_with_trend else True
                )
                if trend_check:
                    long_signal = True
                    reason_parts = [
                        "EMA Bull",
                        f"Stoch Long ({k:.1f})",
                        adx_reason,
                        atr_reason,
                    ]
                    if self.config.trade_only_with_trend:
                        reason_parts.insert(2, "Trend OK")
                    signal_reason = "Long Signal: " + " | ".join(
                        p for p in reason_parts if p
                    )
                elif self.config.trade_only_with_trend:
                    signal_reason = f"Long Blocked (Trend): P({current_price.normalize()}) !> T({trend_ema.normalize()})-{self.config.trend_filter_buffer_percent.normalize()}%"
            elif potential_short:
                trend_check = (
                    price_below_trend if self.config.trade_only_with_trend else True
                )
                if trend_check:
                    short_signal = True
                    reason_parts = [
                        "EMA Bear",
                        f"Stoch Short ({k:.1f})",
                        adx_reason,
                        atr_reason,
                    ]
                    if self.config.trade_only_with_trend:
                        reason_parts.insert(2, "Trend OK")
                    signal_reason = "Short Signal: " + " | ".join(reason_parts)
                elif self.config.trade_only_with_trend:
                    signal_reason = f"Short Blocked (Trend): P({current_price.normalize()}) !< T({trend_ema.normalize()})+{self.config.trend_filter_buffer_percent.normalize()}%"
            else:  # Build reason for no signal
                if "Blocked" not in signal_reason:
                    parts = []
                    base_cond = (
                        ema_bullish_cross
                        and stoch_long_cond
                        or ema_bearish_cross
                        and stoch_short_cond
                    )
                    if not ema_bullish_cross and not ema_bearish_cross:
                        parts.append(f"EMA ({fast_ema:.1f}/{slow_ema:.1f})")
                    elif ema_bullish_cross and not stoch_long_cond:
                        parts.append(f"Stoch !Long ({k:.1f})")
                    elif ema_bearish_cross and not stoch_short_cond:
                        parts.append(f"Stoch !Short ({k:.1f})")
                    if (
                        base_cond
                        and not sig_move
                        and self.config.atr_move_filter_multiplier > 0
                    ):
                        parts.append(atr_reason)
                    if base_cond and sig_move and not adx_trending:
                        parts.append(f"ADX !Trend ({adx:.1f})")
                    elif (
                        base_cond
                        and sig_move
                        and adx_trending
                        and potential_long
                        and not adx_long_confirm
                    ):
                        parts.append(f"ADX !Conf Long ({pdi:.1f}/{mdi:.1f})")
                    elif (
                        base_cond
                        and sig_move
                        and adx_trending
                        and potential_short
                        and not adx_short_confirm
                    ):
                        parts.append(f"ADX !Conf Short ({pdi:.1f}/{mdi:.1f})")
                    signal_reason = "No Signal: " + (
                        " | ".join(p for p in parts if p)
                        if parts
                        else "Conditions unmet"
                    )

            log_level_sig = (
                logging.INFO
                if long_signal or short_signal or "Blocked" in signal_reason
                else logging.DEBUG
            )
            logger.log(log_level_sig, f"Signal Check: {signal_reason}")
        except Exception as e:
            logger.error(f"{Fore.RED}Error generating signals: {e}", exc_info=True)
            return {"long": False, "short": False, "reason": f"Exception: {e}"}
        return {"long": long_signal, "short": short_signal, "reason": signal_reason}

    def check_exit_signals(
        self, position_side: str, indicators: Dict[str, Union[Decimal, bool, int]]
    ) -> Optional[str]:
        """Checks for signal-based exits (EMA cross, Stoch reversal). Returns reason string or None."""
        if not indicators:
            return None
        fast_ema = indicators.get("fast_ema", Decimal("NaN"))
        slow_ema = indicators.get("slow_ema", Decimal("NaN"))
        stoch_k = indicators.get("stoch_k", Decimal("NaN"))
        if fast_ema.is_nan() or slow_ema.is_nan() or stoch_k.is_nan():
            logger.warning(
                "Cannot check exit signals due to NaN indicators (EMA/Stoch)."
            )
            return None

        ema_bear_cross = fast_ema < slow_ema
        ema_bull_cross = fast_ema > slow_ema
        stoch_exit_long = stoch_k > self.config.stoch_overbought_threshold
        stoch_exit_short = stoch_k < self.config.stoch_oversold_threshold

        exit_reason = None
        if position_side == "long":
            if ema_bear_cross:
                exit_reason = "EMA Bearish Cross"
            elif stoch_exit_long:
                exit_reason = f"Stoch Overbought ({stoch_k:.1f} > {self.config.stoch_overbought_threshold})"
        elif position_side == "short":
            if ema_bull_cross:
                exit_reason = "EMA Bullish Cross"
            elif stoch_exit_short:
                exit_reason = f"Stoch Oversold ({stoch_k:.1f} < {self.config.stoch_oversold_threshold})"

        if exit_reason:
            logger.trade(
                f"{Fore.YELLOW}Exit Signal vs {position_side.upper()}: {exit_reason}."
            )
        return exit_reason


# --- Order Manager Class ---
class OrderManager:
    """Handles order placement, position protection (SL/TP/TSL), and closing."""

    def __init__(self, config: TradingConfig, exchange_manager: ExchangeManager):
        self.config = config
        self.exchange_manager = exchange_manager
        self.exchange = exchange_manager.exchange
        self.market_info = exchange_manager.market_info
        # Tracks active protection STATUS (not order IDs) for V5 position stops
        self.protection_tracker = {
            "long": None,
            "short": None,
        }  # Values: 'ACTIVE_SLTP', 'ACTIVE_TSL', None

    def _calculate_trade_parameters(
        self, side: str, atr: Decimal, total_equity: Decimal, current_price: Decimal
    ) -> Optional[Dict]:
        """Calculates SL price, TP price, and quantity based on risk and ATR."""
        if atr.is_nan() or atr <= 0:
            logger.error("Invalid ATR for parameter calculation.")
            return None
        if total_equity.is_nan() or total_equity <= 0:
            logger.error("Invalid equity for parameter calculation.")
            return None
        if current_price.is_nan() or current_price <= 0:
            logger.error("Invalid price for parameter calculation.")
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

            # Validate SL price > 0 and TP price > 0 if set
            if sl_price <= 0:
                logger.error(f"Calculated SL price ({sl_price}) is invalid (<=0).")
                return None
            if tp_price is not None and tp_price <= 0:
                logger.warning(
                    f"Calculated TP price ({tp_price}) is invalid (<=0). Setting TP to None."
                )
                tp_price = None

            sl_distance_price = (current_price - sl_price).copy_abs()
            if (
                sl_distance_price <= self.market_info["tick_size"]
            ):  # Use minimum tick size for check
                logger.error(
                    f"SL distance ({sl_distance_price}) too small or zero. Cannot calculate quantity."
                )
                return None

            # Quantity = Risk Amount / Stop Loss Distance (per unit)
            quantity = risk_amount / sl_distance_price

            # Format using exchange precision
            sl_price_str = self.exchange_manager.format_price(sl_price)
            tp_price_str = (
                self.exchange_manager.format_price(tp_price)
                if tp_price is not None
                else "0"
            )  # Use '0' string for API if no TP
            quantity_str = self.exchange_manager.format_amount(
                quantity, rounding_mode=ROUND_DOWN
            )

            # Minimum order size check
            min_order_size = safe_decimal(
                self.market_info.get("limits", {}).get("amount", {}).get("min"),
                default=Decimal("0"),
            )
            if (
                not min_order_size.is_nan()
                and safe_decimal(quantity_str) < min_order_size
            ):
                logger.error(
                    f"Calculated quantity {quantity_str} is below minimum {min_order_size.normalize()}. Cannot place order."
                )
                return None

            # TSL distance calculation (based on percentage of entry price)
            tsl_distance_price = current_price * (
                self.config.trailing_stop_percent / 100
            )
            tsl_distance_str = self.exchange_manager.format_price(
                tsl_distance_price
            )  # Format as price difference

            logger.info(
                f"Trade Params: Side={side.upper()}, Qty={quantity_str}, Entry~={current_price.normalize()}, SL={sl_price_str}, TP={tp_price_str if tp_price else 'None'}, TSLDist~={tsl_distance_str}, RiskAmt={risk_amount.normalize()}, ATR={atr.normalize()}"
            )
            return {
                "qty": safe_decimal(quantity_str),
                "sl_price": safe_decimal(sl_price_str),
                "tp_price": safe_decimal(tp_price_str) if tp_price else None,
                "tsl_distance": safe_decimal(
                    tsl_distance_str
                ),  # Store calculated TSL distance
            }

        except (InvalidOperation, DivisionByZero, TypeError, Exception) as e:
            logger.error(f"Error calculating trade parameters: {e}", exc_info=True)
            return None

    def _execute_market_order(self, side: str, qty_decimal: Decimal) -> Optional[Dict]:
        """Executes a market order with retries and waits for potential fill."""
        if not self.exchange or not self.market_info:
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
            params = {"category": self.config.bybit_v5_category}  # V5 needs category
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

            # --- Wait for potential fill ---
            # Simple delay - more robust check would involve fetching order status repeatedly
            logger.info(
                f"Waiting {self.config.order_check_delay_seconds}s for order {order_id} to potentially propagate..."
            )
            time.sleep(self.config.order_check_delay_seconds)
            # Optional: Add check_order_status logic here if needed

            return order  # Return submitted order info
        except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e:
            logger.error(f"{Fore.RED}Order placement failed ({type(e).__name__}): {e}")
            termux_notify(
                f"{symbol} Order Failed", f"Market {side.upper()} failed: {str(e)[:50]}"
            )
            return None
        except Exception as e:
            logger.error(
                f"{Fore.RED}Unexpected error placing market order: {e}", exc_info=True
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
        Passes numeric parameters as formatted strings required by Bybit API.
        """
        if not self.exchange or not self.market_info:
            return False
        symbol = self.config.symbol
        market_id = self.market_info.get("id")
        tracker_key = position_side.lower()

        # --- Prepare Base Parameters ---
        params = {
            "category": self.config.bybit_v5_category,
            "symbol": market_id,
            "positionIdx": self.config.position_idx,  # 0 for hedge, 1/2 for one-way
            "tpslMode": "Full",  # Apply to the entire position
            # Default to clearing stops unless specific values are provided
            "stopLoss": "0",
            "takeProfit": "0",
            "trailingStop": "0",
            "activePrice": "0",  # Required even if not activating TSL immediately
            "slTriggerBy": self.config.sl_trigger_by,
            "tpTriggerBy": self.config.sl_trigger_by,  # Often same trigger for TP
            "tslTriggerBy": self.config.tsl_trigger_by,  # Trigger for TSL activation check
        }

        action_desc = ""
        if (
            is_tsl
            and tsl_distance
            and not tsl_distance.is_nan()
            and tsl_distance > 0
            and tsl_activation_price
            and not tsl_activation_price.is_nan()
            and tsl_activation_price > 0
        ):
            # --- Activate Trailing Stop ---
            # Note: 'trailingStop' in V5 API is the distance value (price difference)
            params["trailingStop"] = self.exchange_manager.format_price(
                tsl_distance
            )  # Pass distance as formatted string
            params["activePrice"] = self.exchange_manager.format_price(
                tsl_activation_price
            )  # Price to start trailing
            # Clear fixed SL/TP when activating TSL for V5
            params["stopLoss"] = "0"
            params["takeProfit"] = "0"
            action_desc = f"ACTIVATE TSL (Dist: {params['trailingStop']}, ActPx: {params['activePrice']})"
            self.protection_tracker[tracker_key] = "ACTIVE_TSL"
            logger.debug(
                f"TSL Params: trailingStop={params['trailingStop']}, activePrice={params['activePrice']}"
            )

        elif (sl_price is not None and not sl_price.is_nan()) or (
            tp_price is not None and not tp_price.is_nan()
        ):
            # --- Set Fixed SL/TP ---
            # Ensure SL/TP prices are valid Decimals > 0 before formatting
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
            params["activePrice"] = "0"  # Also clear TSL activation price
            action_desc = f"SET SL={params['stopLoss']} TP={params['takeProfit']}"
            if params["stopLoss"] != "0" or params["takeProfit"] != "0":
                self.protection_tracker[tracker_key] = "ACTIVE_SLTP"
            else:  # Clearing stops
                action_desc = "CLEAR SL/TP"
                self.protection_tracker[tracker_key] = None
        else:
            # Clear stops explicitly if no SL/TP/TSL info provided
            action_desc = "CLEAR SL/TP/TSL (No values provided)"
            params["stopLoss"] = "0"
            params["takeProfit"] = "0"
            params["trailingStop"] = "0"
            params["activePrice"] = "0"
            self.protection_tracker[tracker_key] = None
            logger.debug(f"Clearing all stops for {position_side.upper()}")

        logger.trade(
            f"{Fore.CYAN}Attempting to {action_desc} for {position_side.upper()} {symbol}..."
        )

        try:
            # Use CCXT's private method mapping if available, else define path manually
            # Common path: privatePostPositionSetTradingStop
            path = "private_post_position_set_trading_stop"  # Check CCXT Bybit source if unsure
            if not hasattr(self.exchange, path):
                path = "privatePostPositionSetTradingStop"  # Alternative common naming
                if not hasattr(self.exchange, path):
                    raise NotImplementedError(
                        "CCXT method for V5 SetTradingStop not found."
                    )

            method = getattr(self.exchange, path)
            logger.debug(f"Calling {path} with params: {params}")

            response = fetch_with_retries(
                method,  # Pass the bound method
                params=params,  # Pass params dict
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
            )
            # logger.debug(f"SetTradingStop Response: {response}") # Verbose

            # V5 response check (retCode 0 indicates success)
            if response and response.get("retCode") == 0:
                logger.trade(
                    f"{Fore.GREEN}{action_desc} successful for {position_side.upper()} {symbol}."
                )
                termux_notify(
                    f"{symbol} Protection Set", f"{action_desc} {position_side.upper()}"
                )
                return True
            else:
                ret_code = response.get("retCode", "N/A")
                ret_msg = response.get("retMsg", "Unknown Error")
                logger.error(
                    f"{Fore.RED}{action_desc} failed for {position_side.upper()} {symbol}. Code: {ret_code}, Msg: {ret_msg}"
                )
                # Clear tracker on failure to avoid inconsistent state only if it wasn't a clearing action
                if (
                    action_desc != "CLEAR SL/TP/TSL (No values provided)"
                    and action_desc != "CLEAR SL/TP"
                ):
                    self.protection_tracker[tracker_key] = None
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
            # Clear tracker on exception only if it wasn't a clearing action
            if (
                action_desc != "CLEAR SL/TP/TSL (No values provided)"
                and action_desc != "CLEAR SL/TP"
            ):
                self.protection_tracker[tracker_key] = None
            termux_notify(
                f"{symbol} Protection ERROR",
                f"{action_desc[:30]} {position_side.upper()} error.",
            )
            return False

    def place_risked_market_order(
        self, side: str, atr: Decimal, total_equity: Decimal, current_price: Decimal
    ) -> bool:
        """Calculates parameters, places market order, and sets initial position SL/TP."""
        if not self.exchange or not self.market_info:
            return False
        params = self._calculate_trade_parameters(
            side, atr, total_equity, current_price
        )
        if not params:
            logger.error("Failed to calculate trade parameters. Cannot place order.")
            return False

        qty = params["qty"]
        sl_price = params["sl_price"]
        tp_price = params["tp_price"]
        position_side = "long" if side == "buy" else "short"

        # --- 1. Execute Market Order ---
        order_info = self._execute_market_order(side, qty)
        if not order_info:
            logger.error("Market order execution failed. Aborting entry.")
            return False
        order_id = order_info.get("id")  # Keep for journaling if needed

        # --- 2. Wait briefly & Verify Position ---
        logger.info("Waiting briefly after market order submission...")
        time.sleep(
            self.config.order_check_delay_seconds + 1
        )  # Give position time to appear
        positions_after_entry = self.exchange_manager.get_current_position()
        if positions_after_entry is None:
            logger.error(
                f"{Fore.RED}Position check FAILED after market order. Manual check required!"
            )
            self._handle_entry_failure(side, qty)  # Attempt cleanup
            return False
        if not positions_after_entry.get(position_side):
            logger.error(
                f"{Fore.RED}Position {position_side.upper()} not found after market order. Manual check required! Potential partial fill or delay."
            )
            self._handle_entry_failure(side, qty)
            return False

        filled_qty = safe_decimal(positions_after_entry[position_side].get("qty", "0"))
        avg_entry_price = safe_decimal(
            positions_after_entry[position_side].get("entry_price", "NaN")
        )
        logger.info(
            f"Position {position_side.upper()} confirmed: Qty={filled_qty.normalize()}, AvgEntry={avg_entry_price.normalize() if not avg_entry_price.is_nan() else 'N/A'}"
        )

        # Check if filled quantity matches intended quantity (optional, handle partial fills if necessary)
        if (filled_qty - qty).copy_abs() > self.config.position_qty_epsilon * Decimal(
            "10"
        ):  # Allow slight discrepancy
            logger.warning(
                f"{Fore.YELLOW}Partial fill detected? Intended: {qty.normalize()}, Filled: {filled_qty.normalize()}. Proceeding with filled amount."
            )
            # For simplicity, we still use initial SL/TP calc. More complex logic could adjust.

        # --- 3. Set Position SL/TP ---
        logger.info(f"Setting initial SL/TP for {position_side.upper()} position...")
        set_stops_ok = self._set_position_protection(
            position_side, sl_price=sl_price, tp_price=tp_price
        )
        if not set_stops_ok:
            logger.error(
                f"{Fore.RED}Failed to set initial SL/TP after entry. Attempting to close position for safety!"
            )
            self.close_position(
                position_side, filled_qty, reason="CloseDueToFailedStopSet"
            )  # Close immediately if stops fail
            return False

        # --- 4. Log to Journal ---
        if self.config.enable_journaling and not avg_entry_price.is_nan():
            self.log_trade_entry_to_journal(
                side, filled_qty, avg_entry_price, order_id
            )  # Use actual avg entry price

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
        """Checks and activates TSL if conditions are met using V5 position TSL."""
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

        # Check if TSL is already active according to tracker
        if self.protection_tracker.get(tracker_key) == "ACTIVE_TSL":
            logger.debug(f"TSL already marked as active for {position_side.upper()}.")
            return
        # Check if SL/TP is not active (shouldn't activate TSL if stops cleared manually or hit)
        if self.protection_tracker.get(tracker_key) is None:
            logger.debug(
                f"No active protection tracked for {position_side.upper()}, skipping TSL activation check."
            )
            return

        try:
            activation_distance_atr = atr * self.config.tsl_activation_atr_multiplier
            # Calculate TSL distance in price points based on CURRENT price and percentage
            tsl_distance_price = current_price * (
                self.config.trailing_stop_percent / 100
            )
            # Ensure TSL distance is at least the minimum tick size
            if tsl_distance_price < self.market_info["tick_size"]:
                logger.warning(
                    f"Calculated TSL distance ({tsl_distance_price}) too small, using min tick size {self.market_info['tick_size']}."
                )
                tsl_distance_price = self.market_info["tick_size"]

            should_activate_tsl = False
            activation_price = Decimal("NaN")  # Initialize
            if position_side == "long":
                activation_price = entry_price + activation_distance_atr
                if current_price >= activation_price:
                    should_activate_tsl = True
            elif position_side == "short":
                activation_price = entry_price - activation_distance_atr
                if current_price <= activation_price:
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
                    logger.trade(
                        f"{Fore.GREEN}Trailing Stop Loss activated successfully for {position_side.upper()}."
                    )
                else:
                    logger.error(
                        f"{Fore.RED}Failed to activate Trailing Stop Loss for {position_side.upper()}."
                    )
                    # State remains as ACTIVE_SLTP or None if activation failed

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
        """Closes the specified position by clearing stops and placing a market order."""
        if not self.exchange or not self.market_info:
            return False
        if (
            qty_to_close.is_nan()
            or qty_to_close.copy_abs() < self.config.position_qty_epsilon
        ):
            logger.warning(
                f"Close requested for zero/negligible qty ({qty_to_close}). Skipping."
            )
            return True
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
        # Call set_trading_stop with zeros/clearing values
        clear_stops_ok = self._set_position_protection(
            position_side, sl_price=Decimal("0"), tp_price=Decimal("0"), is_tsl=False
        )  # Explicitly clear fixed and TSL
        if not clear_stops_ok:
            logger.warning(
                f"{Fore.YELLOW}Failed to confirm clearing of position protection. Proceeding with close order anyway..."
            )
        else:
            logger.info("Position protection cleared successfully.")
        self.protection_tracker[tracker_key] = (
            None  # Update tracker regardless of API success for safety
        )

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
            return False

        close_order_id = close_order_info.get("id", "N/A")
        avg_close_price = safe_decimal(
            close_order_info.get("average", "NaN")
        )  # Try to get avg price from order
        logger.trade(
            f"{Fore.GREEN}Closing market order ({close_order_id}) submitted for {position_side.upper()}."
        )
        termux_notify(
            f"{symbol} Position Closing",
            f"{position_side.upper()} close order submitted.",
        )

        # --- 3. Verify Position Closed (Optional but recommended) ---
        logger.info("Waiting briefly and verifying position closure...")
        time.sleep(self.config.order_check_delay_seconds + 2)  # Extra delay for closure
        final_positions = self.exchange_manager.get_current_position()
        is_closed = False
        if final_positions is not None:
            if (
                not final_positions.get(position_side)
                or final_positions.get(position_side, {})
                .get("qty", Decimal("NaN"))
                .copy_abs()
                < self.config.position_qty_epsilon
            ):
                logger.trade(
                    f"{Fore.GREEN}Position {position_side.upper()} confirmed closed."
                )
                is_closed = True
            else:  # Position still exists
                lingering_qty = final_positions.get(position_side, {}).get(
                    "qty", Decimal("NaN")
                )
                logger.error(
                    f"{Fore.RED}Position {position_side.upper()} still shows qty {lingering_qty.normalize()} after close attempt. MANUAL CHECK REQUIRED!"
                )
                termux_notify(
                    f"{symbol} CLOSE VERIFY FAILED",
                    f"{position_side.upper()} still has position!",
                )
        else:  # Failed to fetch final positions
            logger.warning(
                f"{Fore.YELLOW}Could not verify position closure for {position_side.upper()} (failed fetch). Assuming closed based on order submission."
            )
            is_closed = (
                True  # Assume success if order submitted but verification failed
            )

        # --- 4. Log Exit to Journal ---
        if is_closed and self.config.enable_journaling:
            # Try to get a better close price if verification succeeded and position info is available
            if final_positions and not avg_close_price.is_nan():
                close_price_for_log = avg_close_price  # Use order avg if available
            else:
                close_price_for_log = Decimal("NaN")  # Fallback
            self.log_trade_exit_to_journal(
                position_side, qty_to_close, close_price_for_log, close_order_id, reason
            )

        return is_closed

    def _handle_entry_failure(self, failed_entry_side: str, attempted_qty: Decimal):
        """Attempts to close any potentially opened position after a failed entry sequence."""
        logger.warning(
            f"{Fore.YELLOW}Handling potential entry failure for {failed_entry_side.upper()}..."
        )
        position_side = "long" if failed_entry_side == "buy" else "short"

        time.sleep(1)  # Short delay before checking again
        positions = self.exchange_manager.get_current_position()

        if positions and positions.get(position_side):
            current_qty = safe_decimal(positions[position_side].get("qty", "0"))
            if current_qty.copy_abs() >= self.config.position_qty_epsilon:
                logger.error(
                    f"{Fore.RED}Detected lingering {position_side.upper()} position (Qty: {current_qty.normalize()}) after entry failure. Attempting emergency close."
                )
                close_success = self.close_position(
                    position_side, current_qty, reason="EmergencyCloseEntryFail"
                )
                if close_success:
                    logger.info("Emergency close order submitted.")
                else:
                    logger.critical(
                        f"{Fore.RED + Style.BRIGHT}EMERGENCY CLOSE FAILED for {position_side.upper()}. MANUAL INTERVENTION URGENT!"
                    )
            else:
                logger.info(
                    "No significant lingering position found after entry failure."
                )
        elif positions is None:
            logger.error("Could not fetch positions during entry failure handling.")
        else:
            logger.info("No lingering position detected after entry failure.")

    def log_trade_entry_to_journal(
        self, side: str, qty: Decimal, avg_price: Decimal, order_id: Optional[str]
    ):
        """Logs trade entry details to a CSV file."""
        if not self.config.enable_journaling:
            return
        file_path = self.config.journal_file_path
        file_exists = os.path.isfile(file_path)
        try:
            with open(file_path, "a", newline="", encoding="utf-8") as csvfile:
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
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists or os.path.getsize(file_path) == 0:
                    writer.writeheader()  # Write header only if file is new or empty
                writer.writerow(
                    {
                        "TimestampUTC": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        "Symbol": self.config.symbol,
                        "Action": "ENTRY",
                        "Side": side.upper(),
                        "Quantity": qty.normalize(),
                        "AvgPrice": avg_price.normalize()
                        if not avg_price.is_nan()
                        else "N/A",
                        "OrderID": order_id or "N/A",
                        "Reason": "Strategy Signal",  # Add more detail?
                        "Notes": "",
                    }
                )
            logger.debug(f"Trade entry logged to {file_path}")
        except IOError as e:
            logger.error(f"Failed to write entry to journal {file_path}: {e}")
        except Exception as e:
            logger.error(
                f"Unexpected error writing entry to journal: {e}", exc_info=True
            )

    def log_trade_exit_to_journal(
        self,
        side: str,
        qty: Decimal,
        avg_price: Decimal,
        order_id: Optional[str],
        reason: str,
    ):
        """Logs trade exit details to a CSV file."""
        if not self.config.enable_journaling:
            return
        file_path = self.config.journal_file_path
        file_exists = os.path.isfile(file_path)
        try:
            with open(file_path, "a", newline="", encoding="utf-8") as csvfile:
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
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists or os.path.getsize(file_path) == 0:
                    writer.writeheader()
                writer.writerow(
                    {
                        "TimestampUTC": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        "Symbol": self.config.symbol,
                        "Action": "EXIT",
                        "Side": side.upper(),  # Side of the position being exited
                        "Quantity": qty.normalize(),
                        "AvgPrice": avg_price.normalize()
                        if not avg_price.is_nan()
                        else "N/A",
                        "OrderID": order_id or "N/A",
                        "Reason": reason,
                        "Notes": "",  # Could add PnL calculation here later
                    }
                )
            logger.debug(f"Trade exit logged to {file_path}")
        except IOError as e:
            logger.error(f"Failed to write exit to journal {file_path}: {e}")
        except Exception as e:
            logger.error(
                f"Unexpected error writing exit to journal: {e}", exc_info=True
            )


# --- Status Display Class ---
class StatusDisplay:
    """Handles displaying the bot status using the Rich library."""

    def __init__(self, config: TradingConfig):
        self.config = config

    def _format_decimal(
        self, value: Optional[Decimal], precision: int = 2, add_commas: bool = True
    ) -> str:
        """Formats Decimal values for display, handling None and NaN."""
        if value is None or value.is_nan():
            return "[dim]N/A[/]"
        format_str = f"{{:,.{precision}f}}" if add_commas else f"{{:.{precision}f}}"
        try:
            return format_str.format(value)
        except (
            ValueError,
            TypeError,
        ):  # Handle potential formatting errors with unusual Decimal states
            return "[dim]ERR[/]"

    # CORRECTION: Removed unused print_status_panel method.
    # The correct method print_status_panel_with_market_info is used below.

    def print_status_panel_with_market_info(
        self,
        cycle: int,
        timestamp: Optional[pd.Timestamp],
        price: Optional[Decimal],
        indicators: Optional[Dict],
        positions: Optional[Dict],
        equity: Optional[Decimal],
        signals: Dict,
        protection_tracker: Dict,
        market_info: Optional[Dict],
    ):
        """Prints the status panel to the console, explicitly using market_info."""
        # Get precision details safely from market_info
        price_dp = (
            market_info.get("precision_dp", {}).get("price", 4) if market_info else 4
        )
        amount_dp = (
            market_info.get("precision_dp", {}).get("amount", 6) if market_info else 6
        )

        panel_content = ""
        title = f" Cycle {cycle} | {self.config.symbol} ({self.config.interval}) | {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z') if timestamp else 'Timestamp N/A'} "

        # --- Price & Equity ---
        price_str = self._format_decimal(price, precision=price_dp)
        equity_str = self._format_decimal(equity, precision=2)
        settle_curr = self.config.symbol.split(":")[
            -1
        ]  # Get settle currency from symbol
        panel_content += f"[bold cyan]Price:[/] {price_str} | [bold cyan]Equity:[/] {equity_str} {settle_curr}\n"
        panel_content += "---\n"

        # --- Indicators ---
        if indicators:
            ind_parts = []

            def fmt_ind(key, prec=1):
                return self._format_decimal(
                    indicators.get(key), precision=prec, add_commas=False
                )

            ind_parts.append(
                f"EMA(F/S/T): {fmt_ind('fast_ema')}/{fmt_ind('slow_ema')}/{fmt_ind('trend_ema')}"
            )
            ind_parts.append(
                f"Stoch(K/D): {fmt_ind('stoch_k')}/{fmt_ind('stoch_d')} {'[bold green]BULL[/]' if indicators.get('stoch_kd_bullish') else ''}{'[bold red]BEAR[/]' if indicators.get('stoch_kd_bearish') else ''}"
            )
            ind_parts.append(
                f"ATR({indicators.get('atr_period', '?')}): {fmt_ind('atr', 4)}"
            )
            ind_parts.append(
                f"ADX({self.config.adx_period}): {fmt_ind('adx')} [+DI:{fmt_ind('pdi')} -DI:{fmt_ind('mdi')}]"
            )
            panel_content += "[bold cyan]Indicators:[/] " + " | ".join(ind_parts) + "\n"
        else:
            panel_content += "[bold cyan]Indicators:[/] [dim]Calculating...[/]\n"
        panel_content += "---\n"

        # --- Positions ---
        pos_str = "[bold green]FLAT[/]"
        long_pos = positions.get("long") if positions else None
        short_pos = positions.get("short") if positions else None
        active_protection = None

        position_data = None
        position_side_str = ""
        if (
            long_pos
            and safe_decimal(long_pos.get("qty", "0")).copy_abs()
            >= self.config.position_qty_epsilon
        ):
            position_data = long_pos
            position_side_str = "long"
            pos_str_color = "green"
        elif (
            short_pos
            and safe_decimal(short_pos.get("qty", "0")).copy_abs()
            >= self.config.position_qty_epsilon
        ):
            position_data = short_pos
            position_side_str = "short"
            pos_str_color = "red"

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
                f"[magenta]{active_protection}[/]"
                if active_protection
                else "[dim]None[/]"
            )
            pos_str = (
                f"[bold {pos_str_color}]{position_side_str.upper()}:[/] Qty={qty} | Entry={entry} | "
                f"PnL={pnl} | Prot: {prot_str} (SL:{sl} TP:{tp})"
            )

        panel_content += f"[bold cyan]Position:[/] {pos_str}\n"
        panel_content += "---\n"

        # --- Signals ---
        sig_reason = signals.get("reason", "[dim]No signal info[/]")
        sig_style = "white"
        if signals.get("long"):
            sig_style = "bold green"
        elif signals.get("short"):
            sig_style = "bold red"
        elif "Blocked" in sig_reason:
            sig_style = "yellow"
        elif "Signal:" in sig_reason:
            sig_style = "white"  # Explicit signal messages
        else:
            sig_style = "dim"  # Default/No signal

        panel_content += f"[bold cyan]Signal:[/] [{sig_style}]{sig_reason}[/]"

        # Print Panel
        console.print(
            Panel(
                Text.from_markup(panel_content),
                title=f"[bold bright_magenta]{title}[/]",
                border_style="bright_blue",
            )
        )


# --- Trading Bot Class ---
class TradingBot:
    """Main orchestrator for the trading bot."""

    def __init__(self):
        logger.info(
            f"{Fore.MAGENTA + Style.BRIGHT}--- Initializing Pyrmethus v2.4.1 ---"
        )
        self.config = TradingConfig()
        self.exchange_manager = ExchangeManager(self.config)
        # Ensure exchange init succeeded and market_info is loaded
        if not self.exchange_manager.exchange or not self.exchange_manager.market_info:
            logger.critical("Exchange Manager failed to initialize properly. Halting.")
            sys.exit(1)
        self.indicator_calculator = IndicatorCalculator(self.config)
        self.signal_generator = SignalGenerator(self.config)
        self.order_manager = OrderManager(self.config, self.exchange_manager)
        self.status_display = StatusDisplay(
            self.config
        )  # Display now only needs config
        self.shutdown_requested = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Sets up OS signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        """Internal signal handler to set the shutdown flag."""
        if not self.shutdown_requested:
            sig_name = signal.Signals(sig).name if isinstance(sig, int) else str(sig)
            logger.warning(
                f"{Fore.YELLOW + Style.BRIGHT}\nSignal {sig_name} received. Initiating graceful shutdown..."
            )
            self.shutdown_requested = True
        else:
            logger.warning("Shutdown already in progress.")

    def run(self):
        """Starts the main trading loop."""
        self._display_startup_info()
        termux_notify(
            "Pyrmethus Started", f"{self.config.symbol} @ {self.config.interval}"
        )
        cycle_count = 0
        while not self.shutdown_requested:
            cycle_count += 1
            try:
                self.trading_spell_cycle(cycle_count)
            except KeyboardInterrupt:
                logger.warning("\nCtrl+C detected in main loop. Shutting down.")
                self.shutdown_requested = True
                break
            except ccxt.AuthenticationError as e:
                logger.critical(
                    f"{Fore.RED + Style.BRIGHT}CRITICAL AUTH ERROR: {e}. Halting."
                )
                self.shutdown_requested = True
                break  # Exit loop to shutdown
            except Exception as e:  # Catch-all for unexpected cycle errors
                logger.error(
                    f"{Fore.RED + Style.BRIGHT}Unhandled exception in main loop (Cycle {cycle_count}): {e}",
                    exc_info=True,
                )
                logger.error(
                    f"{Fore.RED}Continuing loop, but caution advised. Check logs."
                )
                termux_notify(
                    "Pyrmethus Error", f"Unhandled exception cycle {cycle_count}."
                )
                sleep_time = self.config.loop_sleep_seconds * 2  # Longer sleep
            else:
                sleep_time = self.config.loop_sleep_seconds  # Normal sleep

            if self.shutdown_requested:
                logger.info("Shutdown requested, breaking main loop.")
                break
            # Interruptible Sleep
            logger.debug(
                f"Cycle {cycle_count} finished. Sleeping for {sleep_time} seconds..."
            )
            sleep_end_time = time.time() + sleep_time
            try:
                while time.time() < sleep_end_time and not self.shutdown_requested:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                logger.warning("\nCtrl+C during sleep.")
                self.shutdown_requested = True
                break

        # Perform Graceful Shutdown outside the loop
        self.graceful_shutdown()
        console.print("[bold bright_cyan]Pyrmethus has returned to the ether.[/]")
        sys.exit(0)

    def trading_spell_cycle(self, cycle_count: int) -> None:
        """Executes one cycle of the trading logic."""
        logger.info(
            f"{Fore.MAGENTA + Style.BRIGHT}\n--- Starting Cycle {cycle_count} ---"
        )
        start_time = time.time()
        cycle_status = "OK"  # Track cycle status for logging

        # --- 1. Fetch Data & Basic Info ---
        df = self.exchange_manager.fetch_ohlcv()
        if df is None or df.empty:
            logger.error(f"{Fore.RED}Cycle Failed: Market data fetch failed.")
            cycle_status = "FAIL_FETCH_DATA"
            end_time = time.time()
            logger.info(
                f"{Fore.MAGENTA}--- Cycle {cycle_count} Status: {cycle_status} (Duration: {end_time - start_time:.2f}s) ---"
            )
            return

        try:  # Extract Price & Timestamp
            last_candle = df.iloc[-1]
            current_price = safe_decimal(last_candle["close"])
            last_timestamp = df.index[-1]
            if current_price.is_nan() or current_price <= 0:
                raise ValueError("Invalid latest close price")
            logger.debug(
                f"Latest Candle: {last_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}, Close={current_price.normalize()}"
            )
        except (IndexError, KeyError, ValueError, InvalidOperation, TypeError) as e:
            logger.error(
                f"{Fore.RED}Cycle Failed: Price/Timestamp error: {e}", exc_info=False
            )
            cycle_status = "FAIL_PRICE_PROC"
            end_time = time.time()
            logger.info(
                f"{Fore.MAGENTA}--- Cycle {cycle_count} Status: {cycle_status} (Duration: {end_time - start_time:.2f}s) ---"
            )
            return

        # --- 2. Calculate Indicators ---
        indicators = self.indicator_calculator.calculate_indicators(df)
        if indicators is None:
            logger.warning(
                f"{Fore.YELLOW}Indicator calculation failed. Logic may be impacted."
            )
            cycle_status = "WARN_INDICATORS"
        current_atr = (
            indicators.get("atr", Decimal("NaN")) if indicators else Decimal("NaN")
        )

        # --- 3. Get State (Balance & Positions) ---
        total_equity, _ = self.exchange_manager.get_balance()
        if total_equity is None or total_equity.is_nan() or total_equity <= 0:
            logger.error(
                f"{Fore.RED}Failed fetching valid equity. Trading logic skipped."
            )
            cycle_status = "FAIL_EQUITY"
            can_run_trade_logic = False
        else:
            can_run_trade_logic = True  # Equity is OK

        positions = self.exchange_manager.get_current_position()
        if positions is None:
            logger.error(f"{Fore.RED}Failed fetching positions. Trading logic skipped.")
            cycle_status = "FAIL_POSITIONS"
            can_run_trade_logic = False

        # --- State Snapshot for Panel ---
        protection_tracker_snapshot = copy.deepcopy(
            self.order_manager.protection_tracker
        )
        positions_snapshot = (
            positions if positions is not None else {"long": {}, "short": {}}
        )
        final_positions_for_panel = positions_snapshot
        signals: Dict[str, Union[bool, str]] = {
            "long": False,
            "short": False,
            "reason": "Skipped: Initial State",
        }  # Default

        # --- Main Logic Execution ---
        if can_run_trade_logic:
            active_long_pos = positions.get("long", {})
            active_short_pos = positions.get("short", {})
            long_qty = safe_decimal(active_long_pos.get("qty", "0"))
            short_qty = safe_decimal(active_short_pos.get("qty", "0"))
            long_entry = safe_decimal(active_long_pos.get("entry_price", "NaN"))
            short_entry = safe_decimal(active_short_pos.get("entry_price", "NaN"))
            has_long_pos = long_qty.copy_abs() >= self.config.position_qty_epsilon
            has_short_pos = short_qty.copy_abs() >= self.config.position_qty_epsilon
            is_flat = not has_long_pos and not has_short_pos

            # --- 4. Manage Trailing Stops ---
            if has_long_pos and indicators:
                self.order_manager.manage_trailing_stop(
                    "long", long_entry, current_price, current_atr
                )
            elif has_short_pos and indicators:
                self.order_manager.manage_trailing_stop(
                    "short", short_entry, current_price, current_atr
                )
            elif is_flat:  # Clear tracker if flat
                if (
                    self.order_manager.protection_tracker["long"]
                    or self.order_manager.protection_tracker["short"]
                ):
                    logger.debug("Position flat, clearing protection tracker.")
                    self.order_manager.protection_tracker = {
                        "long": None,
                        "short": None,
                    }
            protection_tracker_snapshot = copy.deepcopy(
                self.order_manager.protection_tracker
            )  # Update snapshot after TSL logic

            # --- Re-fetch position AFTER TSL check (important if TSL hit or stops cleared) ---
            logger.debug("Re-fetching position state after TSL/clear check...")
            positions_after_tsl = self.exchange_manager.get_current_position()
            if positions_after_tsl is None:
                logger.error(
                    f"{Fore.RED}Failed re-fetching positions after TSL check. Assuming previous state."
                )
                cycle_status = "WARN_POS_REFETCH_TSL"
                # Keep using the previously fetched 'positions' if re-fetch fails
                final_positions_for_panel = positions
            else:  # Update live state variables
                active_long_pos = positions_after_tsl.get("long", {})
                active_short_pos = positions_after_tsl.get("short", {})
                long_qty = safe_decimal(active_long_pos.get("qty", "0"))
                short_qty = safe_decimal(active_short_pos.get("qty", "0"))
                has_long_pos = long_qty.copy_abs() >= self.config.position_qty_epsilon
                has_short_pos = short_qty.copy_abs() >= self.config.position_qty_epsilon
                is_flat = not has_long_pos and not has_short_pos
                logger.debug(
                    f"State After TSL Check: Flat={is_flat}, Long={long_qty.normalize()}, Short={short_qty.normalize()}"
                )
                final_positions_for_panel = positions_after_tsl  # Update panel data
                if is_flat and (
                    self.order_manager.protection_tracker["long"]
                    or self.order_manager.protection_tracker["short"]
                ):  # Clear again if became flat
                    logger.debug(
                        "Position became flat after TSL logic, clearing protection tracker."
                    )
                    self.order_manager.protection_tracker = {
                        "long": None,
                        "short": None,
                    }
                    protection_tracker_snapshot = copy.deepcopy(
                        self.order_manager.protection_tracker
                    )

                # --- 5. Generate Trading Signals (Entry) ---
                can_gen_signals = (
                    indicators is not None
                    and not current_price.is_nan()
                    and len(df) >= 2
                )
                if can_gen_signals:
                    signals = self.signal_generator.generate_signals(
                        df.iloc[-2:], indicators
                    )
                else:
                    signals["reason"] = "Skipped Signal Gen: " + (
                        "Indicators missing"
                        if indicators is None
                        else f"Need >=2 candles ({len(df)} found)"
                    )
                    logger.warning(signals["reason"])

                # --- 6. Check for Exits (Signal-Based) ---
                exit_triggered = False
                exit_side = None
                if can_gen_signals and not is_flat:
                    pos_side_to_check = "long" if has_long_pos else "short"
                    exit_reason = self.signal_generator.check_exit_signals(
                        pos_side_to_check, indicators
                    )
                    if exit_reason:
                        exit_side = pos_side_to_check
                        qty_to_close = long_qty if exit_side == "long" else short_qty
                        exit_triggered = self.order_manager.close_position(
                            exit_side, qty_to_close, reason=exit_reason
                        )
                        if not exit_triggered:
                            cycle_status = "FAIL_EXIT_CLOSE"  # Mark failure if close command failed

                # --- Re-fetch state AGAIN if an exit was triggered ---
                if exit_triggered:
                    logger.debug(
                        f"Re-fetching state after signal exit attempt ({exit_side})..."
                    )
                    positions_after_exit = self.exchange_manager.get_current_position()
                    if positions_after_exit is None:
                        logger.error(
                            f"{Fore.RED}Failed re-fetching positions after signal exit."
                        )
                        cycle_status = "WARN_POS_REFETCH_EXIT"
                        final_positions_for_panel = (
                            positions_after_tsl  # Use previous state if re-fetch fails
                        )
                    else:  # Update live state variables
                        active_long_pos = positions_after_exit.get("long", {})
                        active_short_pos = positions_after_exit.get("short", {})
                        long_qty = safe_decimal(active_long_pos.get("qty", "0"))
                        short_qty = safe_decimal(active_short_pos.get("qty", "0"))
                        has_long_pos = (
                            long_qty.copy_abs() >= self.config.position_qty_epsilon
                        )
                        has_short_pos = (
                            short_qty.copy_abs() >= self.config.position_qty_epsilon
                        )
                        is_flat = not has_long_pos and not has_short_pos
                        logger.debug(
                            f"State After Signal Exit: Flat={is_flat}, Long={long_qty.normalize()}, Short={short_qty.normalize()}"
                        )
                        final_positions_for_panel = (
                            positions_after_exit  # Update panel data
                        )
                        if is_flat:
                            logger.debug(
                                "Position became flat after signal exit, clearing protection tracker."
                            )
                            self.order_manager.protection_tracker = {
                                "long": None,
                                "short": None,
                            }
                            protection_tracker_snapshot = copy.deepcopy(
                                self.order_manager.protection_tracker
                            )

                # --- 7. Execute Entry Trades (Only if now flat) ---
                if (
                    is_flat
                    and can_gen_signals
                    and not current_atr.is_nan()
                    and (signals.get("long") or signals.get("short"))
                ):
                    if current_atr <= 0:
                        logger.warning("Cannot enter trade: Invalid ATR (<= 0)")
                    elif total_equity <= 0:
                        logger.warning(
                            "Cannot enter trade: Invalid Equity (<= 0)"
                        )  # Should have been caught earlier but double check
                    else:
                        entry_side = "buy" if signals.get("long") else "sell"
                        logger.info(
                            f"{Fore.GREEN + Style.BRIGHT if entry_side == 'buy' else Fore.RED + Style.BRIGHT}{entry_side.upper()} signal! {signals.get('reason', '')}. Attempting entry..."
                        )
                        entry_successful = self.order_manager.place_risked_market_order(
                            entry_side, current_atr, total_equity, current_price
                        )
                        if entry_successful:
                            logger.info(f"{Fore.GREEN}Entry process completed.")
                        else:
                            logger.error(f"{Fore.RED}Entry process failed.")
                            cycle_status = "FAIL_ENTRY"
                        # Re-fetch for panel accuracy after entry attempt
                        logger.debug("Re-fetching state after entry attempt...")
                        positions_after_entry = (
                            self.exchange_manager.get_current_position()
                        )
                        if positions_after_entry is not None:
                            final_positions_for_panel = positions_after_entry
                            protection_tracker_snapshot = copy.deepcopy(
                                self.order_manager.protection_tracker
                            )
                        else:
                            logger.warning(
                                "Failed re-fetching positions after entry attempt. Panel may be stale."
                            )
                            cycle_status = "WARN_POS_REFETCH_ENTRY"
                elif is_flat:
                    logger.debug("Position flat, no entry signal.")
                elif not is_flat:
                    logger.debug(
                        f"Position ({'LONG' if has_long_pos else 'SHORT'}) remains open, skipping entry."
                    )

        else:  # Initial critical data fetch failed
            signals["reason"] = f"Skipped: {cycle_status}"

        # --- 8. Display Status Panel ---
        self.status_display.print_status_panel_with_market_info(
            cycle_count,
            last_timestamp,
            current_price,
            indicators,
            final_positions_for_panel,
            total_equity,
            signals,
            protection_tracker_snapshot,
            self.exchange_manager.market_info,
        )

        end_time = time.time()
        logger.info(
            f"{Fore.MAGENTA}--- Cycle {cycle_count} Status: {cycle_status} (Duration: {end_time - start_time:.2f}s) ---"
        )

    def graceful_shutdown(self) -> None:
        """Handles cleaning up orders and positions before exiting."""
        logger.warning(
            f"{Fore.YELLOW + Style.BRIGHT}\nInitiating Graceful Shutdown Sequence..."
        )
        termux_notify("Shutdown", f"Closing {self.config.symbol}...")
        if (
            not self.exchange_manager
            or not self.exchange_manager.exchange
            or not self.exchange_manager.market_info
        ):
            logger.error(
                f"{Fore.RED}Cannot shutdown cleanly: Exchange/Market Info missing."
            )
            termux_notify(
                "Shutdown Warning!", f"{self.config.symbol} Cannot shutdown cleanly."
            )
            return

        exchange = self.exchange_manager.exchange
        symbol = self.config.symbol

        try:  # --- 1. Cancel All Open Orders ---
            logger.info(
                f"{Fore.CYAN}Attempting to cancel all open orders for {symbol} using V5 params..."
            )
            try:
                # Use generic cancel_all_orders but provide V5 category
                params = {"category": self.config.bybit_v5_category}
                # Note: cancel_all_orders might not cancel SL/TP orders attached to positions in V5.
                # Clearing stops via _set_position_protection before closing position is more reliable for V5.
                cancel_resp = fetch_with_retries(
                    exchange.cancel_all_orders,
                    symbol=symbol,
                    params=params,
                    max_retries=1,
                    delay_seconds=1,
                )
                logger.info(f"Cancel all (active) orders response: {cancel_resp}")
                # If response is a list, check length
                if isinstance(cancel_resp, list):
                    logger.info(f"Cancelled {len(cancel_resp)} active orders.")
            except Exception as e:
                logger.error(
                    f"{Fore.RED}Error cancelling active orders: {e}", exc_info=True
                )

            logger.info("Clearing local protection tracker.")
            self.order_manager.protection_tracker = {"long": None, "short": None}
        except Exception as e:
            logger.error(
                f"{Fore.RED + Style.BRIGHT}Order cancellation phase error: {e}. MANUAL CHECK REQUIRED.",
                exc_info=True,
            )

        logger.info("Waiting after order cancellation attempt...")
        time.sleep(max(self.config.order_check_delay_seconds, 2))

        try:  # --- 2. Close Any Lingering Positions ---
            logger.info(f"{Fore.CYAN}Checking for lingering positions to close...")
            closed_count = 0
            positions = self.exchange_manager.get_current_position()
            if positions is not None:
                positions_to_close = {}
                for side, pos_data in positions.items():
                    if (
                        pos_data
                        and safe_decimal(pos_data.get("qty", "0")).copy_abs()
                        >= self.config.position_qty_epsilon
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
                        # Use the OrderManager's close_position method which clears stops first for V5
                        if self.order_manager.close_position(
                            side, qty, reason="GracefulShutdown"
                        ):
                            closed_count += 1
                            logger.info(
                                f"{Fore.GREEN}Closure initiated for {side.upper()}."
                            )
                        else:
                            logger.error(
                                f"{Fore.RED}Closure failed for {side.upper()}. MANUAL INTERVENTION REQUIRED."
                            )
                    if closed_count == len(positions_to_close):
                        logger.info(
                            f"{Fore.GREEN}All detected positions closed successfully."
                        )
                    else:
                        logger.warning(
                            f"{Fore.YELLOW}Attempted {len(positions_to_close)} closures, {closed_count} closure orders submitted. MANUAL VERIFICATION REQUIRED."
                        )
            else:
                logger.error(
                    f"{Fore.RED}Failed fetching positions during shutdown check. MANUAL CHECK REQUIRED."
                )
        except Exception as e:
            logger.error(
                f"{Fore.RED + Style.BRIGHT}Position closure phase error: {e}. MANUAL CHECK REQUIRED.",
                exc_info=True,
            )

        logger.warning(
            f"{Fore.YELLOW + Style.BRIGHT}Graceful Shutdown Sequence Complete. Pyrmethus rests."
        )
        termux_notify("Shutdown Complete", f"{self.config.symbol} shutdown finished.")

    def _display_startup_info(self):
        """Prints initial configuration details."""
        console.print("[bold bright_cyan] summoning Pyrmethus [magenta]v2.4.1[/]...")
        console.print(
            f"[yellow]Trading Symbol: [white]{self.config.symbol}[/] | Interval: [white]{self.config.interval}[/] | Category: [white]{self.config.bybit_v5_category}[/]"
        )
        console.print(
            f"[yellow]Risk: [white]{self.config.risk_percentage:.3%}[/] | SL Mult: [white]{self.config.sl_atr_multiplier.normalize()}x[/] | TP Mult: [white]{self.config.tp_atr_multiplier.normalize()}x[/]"
        )
        console.print(
            f"[yellow]TSL Act Mult: [white]{self.config.tsl_activation_atr_multiplier.normalize()}x[/] | TSL %: [white]{self.config.trailing_stop_percent.normalize()}%[/]"
        )
        console.print(
            f"[yellow]Trend Filter: [white]{'ON' if self.config.trade_only_with_trend else 'OFF'}[/] | ATR Move Filter: [white]{self.config.atr_move_filter_multiplier.normalize()}x[/] | ADX Filter: [white]>{self.config.min_adx_level.normalize()}[/]"
        )
        console.print(
            f"[yellow]Journaling: [white]{'Enabled' if self.config.enable_journaling else 'Disabled'}[/] ([dim]{self.config.journal_file_path}[/])"
        )
        console.print(
            f"[dim]Using V5 Position Stops (SLTrig:{self.config.sl_trigger_by}, TSLTrig:{self.config.tsl_trigger_by}, PosIdx:{self.config.position_idx})[/]"
        )


# --- Main Execution Block ---
if __name__ == "__main__":
    # Load .env before initializing anything that uses it
    load_dotenv()
    try:
        bot = TradingBot()
        bot.run()
    except SystemExit:  # Catch sys.exit() calls for clean termination
        logger.info("SystemExit called, terminating process.")
    except Exception as main_exception:
        # Catch errors during initialization or the final shutdown phase
        logger.critical(
            f"{Fore.RED + Style.BRIGHT}Critical error during bot execution: {main_exception}",
            exc_info=True,
        )
        termux_notify(
            "Pyrmethus CRITICAL ERROR", f"Bot failed: {str(main_exception)[:100]}"
        )
        sys.exit(1)
