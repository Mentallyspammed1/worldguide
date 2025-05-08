#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=line-too-long, too-many-lines, logging-fstring-interpolation, too-many-instance-attributes, too-many-arguments, too-many-locals, too-many-public-methods, invalid-name
# fmt: off
#   ____        _       _   _                  _            _         _
#  |  _ \ _   _| |_ ___| | | | __ ___   ____ _| |_ ___  ___| |_ _ __ | |__
#  | |_) | | | | __/ _ \ | | |/ _` \ \ / / _` | __/ _ \/ __| __| '_ \| '_ \
#  |  __/| |_| | ||  __/ | | | (_| |\ V / (_| | ||  __/\__ \ |_| |_) | | | |
#  |_|    \__, |\__\___|_|_|_|\__,_| \_/ \__,_|\__\___||___/\__| .__/|_| |_|
#         |___/                                                |_|
# Pyrmethus v2.4.0 - Class Refactor, V5 Position Stops, Enhanced Structure
# fmt: on
# pylint: enable=line-too-long
"""
Pyrmethus - Termux Trading Spell (v2.4.0)

Conjures market insights and executes trades on Bybit Futures using the
V5 Unified Account API via CCXT. Refactored into classes for better structure
and utilizing V5 position-based stop-loss/take-profit/trailing-stop features.

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
from decimal import ROUND_DOWN, ROUND_HALF_EVEN, Decimal, DivisionByZero, InvalidOperation, getcontext
from typing import Any, Dict, Optional, Tuple, Union

# Third-Party Imports
try:
    import ccxt
    import numpy as np
    import pandas as pd
    import requests
    from colorama import Fore, Style, init as colorama_init  # Keep for basic init/fallback
    from dotenv import load_dotenv
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    COMMON_PACKAGES = ["ccxt", "python-dotenv", "pandas", "numpy", "rich", "colorama", "requests"]
except ImportError as e:
    colorama_init(autoreset=True)
    missing_pkg = e.name
    print(f"{Fore.RED}{Style.BRIGHT}Missing essential spell component: {Style.BRIGHT}{missing_pkg}{Style.NORMAL}")
    print(f"{Fore.YELLOW}To conjure it, cast: {Style.BRIGHT}pip install {missing_pkg}{Style.RESET_ALL}")
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
getcontext().prec = 50

# --- Logging Setup ---
logger = logging.getLogger(__name__)
TRADE_LEVEL_NUM = logging.INFO + 5
if not hasattr(logging.Logger, "trade"):
    logging.addLevelName(TRADE_LEVEL_NUM, "TRADE")

    def trade_log(self, message, *args, **kws):
        if self.isEnabledFor(TRADE_LEVEL_NUM):
            self._log(TRADE_LEVEL_NUM, message, args, **kws)  # pylint: disable=protected-access

    logging.Logger.trade = trade_log

log_formatter = logging.Formatter("%(asctime)s [%(levelname)-8s] (%(filename)s:%(lineno)d) %(message)s")
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logger.setLevel(log_level)
if not any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout for h in logger.handlers):
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
logger.propagate = False


# --- Configuration Class ---
class TradingConfig:
    """Loads, validates, and holds trading configuration parameters from .env."""

    def __init__(self):
        logger.debug("Loading configuration from environment variables...")
        self.symbol = self._get_env("SYMBOL", "BTC/USDT:USDT", Style.DIM)
        self.market_type = self._get_env(
            "MARKET_TYPE", "linear", Style.DIM, allowed_values=["linear", "inverse", "swap"]
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
            "SL_ATR_MULTIPLIER", "1.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.1"), max_val=Decimal("20.0")
        )
        self.tp_atr_multiplier = self._get_env(
            "TP_ATR_MULTIPLIER", "3.0", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.0"), max_val=Decimal("50.0")
        )
        self.tsl_activation_atr_multiplier = self._get_env(
            "TSL_ACTIVATION_ATR_MULTIPLIER",
            "1.0",
            Fore.YELLOW,
            cast_type=Decimal,
            min_val=Decimal("0.1"),
            max_val=Decimal("20.0"),
        )
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
            "SL_TRIGGER_BY", "LastPrice", Style.DIM, allowed_values=["LastPrice", "MarkPrice", "IndexPrice"]
        )
        self.tsl_trigger_by = self._get_env(
            "TSL_TRIGGER_BY", "LastPrice", Style.DIM, allowed_values=["LastPrice", "MarkPrice", "IndexPrice"]
        )
        # Position Index: 0 for hedge mode (default/recommended), 1=Buy side one-way, 2=Sell side one-way
        # Use 0 unless you know you need specific one-way mode index handling.
        self.position_idx = self._get_env("POSITION_IDX", "0", Style.DIM, cast_type=int, allowed_values=[0, 1, 2])

        # Indicator Periods (int)
        self.trend_ema_period = self._get_env(
            "TREND_EMA_PERIOD", "12", Style.DIM, cast_type=int, min_val=5, max_val=500
        )
        self.fast_ema_period = self._get_env("FAST_EMA_PERIOD", "9", Style.DIM, cast_type=int, min_val=1, max_val=200)
        self.slow_ema_period = self._get_env("SLOW_EMA_PERIOD", "21", Style.DIM, cast_type=int, min_val=2, max_val=500)
        self.stoch_period = self._get_env("STOCH_PERIOD", "7", Style.DIM, cast_type=int, min_val=1, max_val=100)
        self.stoch_smooth_k = self._get_env("STOCH_SMOOTH_K", "3", Style.DIM, cast_type=int, min_val=1, max_val=10)
        self.stoch_smooth_d = self._get_env("STOCH_SMOOTH_D", "3", Style.DIM, cast_type=int, min_val=1, max_val=10)
        self.atr_period = self._get_env("ATR_PERIOD", "5", Style.DIM, cast_type=int, min_val=1, max_val=100)
        self.adx_period = self._get_env("ADX_PERIOD", "14", Style.DIM, cast_type=int, min_val=2, max_val=100)

        # Signal Logic Thresholds (Decimal)
        self.stoch_oversold_threshold = self._get_env(
            "STOCH_OVERSOLD_THRESHOLD", "30", Fore.CYAN, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("45")
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
            "MIN_ADX_LEVEL", "20", Fore.CYAN, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("90")
        )

        self.position_qty_epsilon = Decimal("1E-12")  # Threshold for considering a position 'flat'
        self.api_key = self._get_env("BYBIT_API_KEY", None, Fore.RED)
        self.api_secret = self._get_env("BYBIT_API_SECRET", None, Fore.RED)

        # Operational Parameters
        self.ohlcv_limit = self._get_env("OHLCV_LIMIT", "200", Style.DIM, cast_type=int, min_val=50, max_val=1000)
        self.loop_sleep_seconds = self._get_env("LOOP_SLEEP_SECONDS", "15", Style.DIM, cast_type=int, min_val=5)
        self.order_check_delay_seconds = self._get_env(
            "ORDER_CHECK_DELAY_SECONDS", "2", Style.DIM, cast_type=int, min_val=1
        )
        self.order_fill_timeout_seconds = self._get_env(
            "ORDER_FILL_TIMEOUT_SECONDS", "20", Style.DIM, cast_type=int, min_val=5
        )  # Longer for fills
        self.max_fetch_retries = self._get_env(
            "MAX_FETCH_RETRIES", "3", Style.DIM, cast_type=int, min_val=1, max_val=10
        )
        self.retry_delay_seconds = self._get_env("RETRY_DELAY_SECONDS", "3", Style.DIM, cast_type=int, min_val=1)
        self.trade_only_with_trend = self._get_env("TRADE_ONLY_WITH_TREND", "True", Style.DIM, cast_type=bool)

        # Journaling
        self.journal_file_path = self._get_env("JOURNAL_FILE_PATH", "pyrmethus_trading_journal.csv", Style.DIM)
        self.enable_journaling = self._get_env("ENABLE_JOURNALING", "True", Style.DIM, cast_type=bool)

        if not self.api_key or not self.api_secret:
            logger.critical(f"{Fore.RED + Style.BRIGHT}BYBIT_API_KEY or BYBIT_API_SECRET not found. Halting.")
            sys.exit(1)
        self._validate_config()
        logger.debug("Configuration loaded and validated successfully.")

    def _determine_v5_category(self) -> str:
        """Determines the Bybit V5 API category."""
        try:
            parts = self.symbol.replace(":", "/").split("/")
            if len(parts) < 2:
                raise ValueError("Symbol format must be BASE/QUOTE[:SETTLE]")
            parts[-1].upper()
            category = ""
            if self.market_type == "inverse":
                category = "inverse"
            elif self.market_type in ["linear", "swap"]:
                category = "linear"  # Linear uses USDT or USDC settle
            else:
                raise ValueError(f"Unsupported MARKET_TYPE '{self.market_type}'")
            logger.info(
                f"Determined Bybit V5 API category: '{category}' for symbol '{self.symbol}' and type '{self.market_type}'"
            )
            return category
        except (ValueError, IndexError) as e:
            logger.critical(f"Could not parse symbol '{self.symbol}' for category: {e}. Halting.", exc_info=True)
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
        if self.tp_atr_multiplier > Decimal("0") and self.tp_atr_multiplier <= self.sl_atr_multiplier:
            logger.warning(
                f"{Fore.YELLOW}TP_MULT ({self.tp_atr_multiplier.normalize()}) <= SL_MULT ({self.sl_atr_multiplier.normalize()}). Poor R:R setup."
            )

    def _get_env(
        self, key: str, default: Any, color: str, cast_type: type = str, min_val=None, max_val=None, allowed_values=None
    ) -> Any:
        """Gets value from environment, casts, validates, logs."""
        # (Simplified version of the robust _get_env from v2.3.3 for brevity)
        value_str = os.getenv(key)
        log_value = "****" if "SECRET" in key.upper() or "KEY" in key.upper() else value_str
        if value_str is None or value_str.strip() == "":
            if default is None and key not in ["BYBIT_API_KEY", "BYBIT_API_SECRET"]:
                logger.critical(f"{Fore.RED + Style.BRIGHT}Required config '{key}' not found. Halting.")
                sys.exit(1)
            value_str_for_cast = str(default) if default is not None else None
            if default is not None:
                logger.warning(f"{color}Using default for {key}: {default}")
        else:
            logger.info(f"{color}Summoned {key}: {log_value}")
            value_str_for_cast = value_str
        if value_str_for_cast is None and default is None:
            logger.critical(f"{Fore.RED + Style.BRIGHT}Required config '{key}' missing, no default. Halting.")
            sys.exit(1)
        casted_value = None
        try:
            if cast_type == bool:
                casted_value = str(value_str_for_cast).lower() in ["true", "1", "yes", "y", "on"]
            elif cast_type == Decimal:
                casted_value = Decimal(str(value_str_for_cast))  # Ensure str conversion
            elif cast_type == int:
                casted_value = int(Decimal(str(value_str_for_cast)))  # Use Decimal intermediary
            else:
                casted_value = cast_type(str(value_str_for_cast))  # Ensure str conversion
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.error(
                f"{Fore.RED}Cast failed for {key} ('{value_str_for_cast}' -> {cast_type.__name__}): {e}. Using default '{default}'."
            )
            try:  # Re-cast default
                if default is None:
                    casted_value = None
                elif cast_type == bool:
                    casted_value = str(default).lower() in ["true", "1", "yes", "y", "on"]
                elif cast_type == Decimal:
                    casted_value = Decimal(str(default))
                elif cast_type == int:
                    casted_value = int(Decimal(str(default)))
                else:
                    casted_value = cast_type(str(default))
            except (ValueError, TypeError, InvalidOperation) as cast_default_err:
                logger.critical(
                    f"{Fore.RED + Style.BRIGHT}Default '{default}' for {key} invalid ({cast_type.__name__}): {cast_default_err}. Halting."
                )
                sys.exit(1)
        # --- Simplified Validation (add back min/max/allowed if needed) ---
        if allowed_values:
            comp_value = str(casted_value).lower() if isinstance(casted_value, str) else casted_value
            lower_allowed = (
                [str(v).lower() for v in allowed_values] if isinstance(allowed_values[0], str) else allowed_values
            )
            if comp_value not in lower_allowed:
                logger.error(
                    f"{Fore.RED}Validation failed for {key}: Invalid value '{casted_value}'. Allowed: {allowed_values}. Using default: {default}"
                )
                # Re-cast default on validation failure
                try:
                    if default is None:
                        return None
                    if cast_type == bool:
                        return str(default).lower() in ["true", "1", "yes", "y", "on"]
                    elif cast_type == Decimal:
                        return Decimal(str(default))
                    elif cast_type == int:
                        return int(Decimal(str(default)))
                    else:
                        return cast_type(str(default))
                except (ValueError, TypeError, InvalidOperation) as cast_default_err:
                    logger.critical(
                        f"{Fore.RED + Style.BRIGHT}Default '{default}' for {key} invalid on fallback: {cast_default_err}. Halting."
                    )
                    sys.exit(1)
        # --- Add back min/max validation here if required ---
        return casted_value


# --- Utility Functions ---
def termux_notify(title: str, content: str) -> None:
    """Sends a notification via Termux API, if available."""
    if platform.system() == "Linux" and "com.termux" in os.environ.get("PREFIX", ""):
        try:
            subprocess.run(["termux-toast", "-t", title, content], check=False, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
            logger.warning(f"Termux notify failed: {e}")
    # else: logger.debug("Not in Termux, skipping notification.") # Optional debug


def fetch_with_retries(fetch_function, *args, max_retries=3, delay_seconds=3, **kwargs) -> Any:
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
            ccxt.ExchangeError,
            requests.exceptions.RequestException,
        ) as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(
                    f"{Fore.YELLOW}Retryable error ({type(e).__name__}) on attempt {attempt + 1}/{max_retries + 1} for {fetch_function.__name__}: {e}. Retrying in {delay_seconds}s..."
                )
                time.sleep(delay_seconds)
            else:
                logger.error(
                    f"{Fore.RED}Max retries ({max_retries + 1}) reached for {fetch_function.__name__}. Last error: {e}"
                )
                break  # Exit loop after max retries
        except ccxt.AuthenticationError as e:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}Authentication error: {e}. Check API keys. Halting.", exc_info=True
            )
            sys.exit(1)
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
            logger.error(f"{Fore.RED}Permission denied: {e}. Check API permissions/IP whitelisting.")
            break  # Not retryable
        except Exception as e:  # Catch unexpected errors during fetch
            logger.error(f"{Fore.RED}Unexpected error during {fetch_function.__name__}: {e}", exc_info=True)
            last_exception = e
            break  # Don't retry unknown errors
    # If loop finished without success, raise the last captured exception or a generic one
    if last_exception:
        raise last_exception
    else:
        raise RuntimeError(f"Function {fetch_function.__name__} failed after unexpected issue.")


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
                    "brokerId": "TermuxPyrmV5",  # Custom ID
                    # Explicitly set V5 usage if CCXT version needs it
                    # 'enableRateLimit': True, # Default True usually
                },
            }
            # Handle sandbox endpoint if needed (check .env or CCXT docs for exact sandbox URL)
            # if self.config.use_sandbox: exchange_params['urls'] = {'api': 'https://api-testnet.bybit.com'}

            self.exchange = ccxt.bybit(exchange_params)
            logger.info(f"{Fore.GREEN}Bybit V5 interface initialized successfully.")
            self.market_info = self._load_market_info()

        except ccxt.AuthenticationError as e:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}Authentication failed: {e}. Check API keys/permissions.", exc_info=True
            )
            sys.exit(1)
        except ccxt.NetworkError as e:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}Network error initializing exchange: {e}. Check connection.", exc_info=True
            )
            sys.exit(1)
        except Exception as e:
            logger.critical(f"{Fore.RED + Style.BRIGHT}Unexpected error initializing exchange: {e}", exc_info=True)
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
            logger.info(
                f"Market info loaded: ID={market.get('id')}, Precision(Amt={market.get('precision', {}).get('amount')}, Price={market.get('precision', {}).get('price')}), Limits(MinAmt={market.get('limits', {}).get('amount', {}).get('min')})"
            )
            # Pre-calculate Decimal precision values if needed
            market["precision"]["amount_decimal"] = Decimal(
                str(10 ** -int(market.get("precision", {}).get("amount", 8)))
            )  # Default 8 if missing
            market["precision"]["price_decimal"] = Decimal(
                str(10 ** -int(market.get("precision", {}).get("price", 8)))
            )  # Default 8 if missing
            return market
        except (ccxt.ExchangeError, KeyError, Exception) as e:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}Failed to load market info for {self.config.symbol}: {e}. Halting.",
                exc_info=True,
            )
            sys.exit(1)

    def format_price(self, price: Union[Decimal, str, float, int]) -> str:
        """Formats price according to market precision using ROUND_HALF_EVEN."""
        if self.market_info and "precision" in self.market_info and "price" in self.market_info["precision"]:
            precision = int(self.market_info["precision"]["price"])
            price_decimal = Decimal(str(price))
            quantizer = Decimal("1e-" + str(precision))
            return str(price_decimal.quantize(quantizer, rounding=ROUND_HALF_EVEN))
        else:
            logger.warning("Market info/price precision unavailable, using default formatting.")
            return str(Decimal(str(price)))  # Fallback

    def format_amount(self, amount: Union[Decimal, str, float, int], rounding_mode=ROUND_DOWN) -> str:
        """Formats amount according to market precision using specified rounding."""
        if self.market_info and "precision" in self.market_info and "amount" in self.market_info["precision"]:
            precision = int(self.market_info["precision"]["amount"])
            amount_decimal = Decimal(str(amount))
            quantizer = Decimal("1e-" + str(precision))
            return str(amount_decimal.quantize(quantizer, rounding=rounding_mode))
        else:
            logger.warning("Market info/amount precision unavailable, using default formatting.")
            return str(Decimal(str(amount)))  # Fallback

    def fetch_ohlcv(self) -> Optional[pd.DataFrame]:
        """Fetches OHLCV data with retries."""
        if not self.exchange:
            return None
        logger.debug(f"Fetching {self.config.ohlcv_limit} candles for {self.config.symbol} ({self.config.interval})...")
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

            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            # Convert OHLCV to Decimal
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else Decimal("NaN"))
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
        logger.debug(f"Fetching balance for {settle_currency}...")
        try:
            # V5 balance needs category
            params = (
                {"accountType": "UNIFIED", "coin": settle_currency}
                if self.config.bybit_v5_category in ["linear", "inverse"]
                else {}
            )
            balance_data = fetch_with_retries(
                self.exchange.fetch_balance,  # Pass function
                params=params,
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
            )
            # Parse V5 response structure (may vary slightly with CCXT version)
            total_equity = Decimal("NaN")
            available_balance = Decimal("NaN")
            # Try unified account structure first
            if "info" in balance_data and "result" in balance_data["info"] and "list" in balance_data["info"]["result"]:
                acc_list = balance_data["info"]["result"]["list"]
                if acc_list:
                    # Find the UNIFIED account data
                    unified_acc = next((item for item in acc_list if item.get("accountType") == "UNIFIED"), None)
                    if unified_acc and "totalEquity" in unified_acc and "coin" in unified_acc:
                        for coin_info in unified_acc["coin"]:
                            if coin_info.get("coin") == settle_currency:
                                total_equity = Decimal(str(coin_info.get("equity", "NaN")))
                                available_balance = Decimal(
                                    str(coin_info.get("availableToWithdraw", "NaN"))
                                )  # Or 'availableToBorrow' depending on need
                                break
            # Fallback to older structure / total structure if unified not found or empty
            if total_equity.is_nan() and settle_currency in balance_data["total"]:
                total_equity = Decimal(str(balance_data[settle_currency].get("total", "NaN")))
                available_balance = Decimal(str(balance_data[settle_currency].get("free", "NaN")))

            if total_equity.is_nan():
                logger.error(f"Could not extract total equity for {settle_currency}.")
                return None, None
            if available_balance.is_nan():
                logger.warning(f"Could not extract available balance for {settle_currency}.")
                available_balance = Decimal("0")  # Default to 0 if unavailable

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
            # logger.debug(f"Raw position data: {position_data}") # Very verbose

            if not position_data:
                logger.debug("No position data returned.")
                return positions_dict  # Return empty if no positions

            # V5 returns a list, find the matching symbol
            pos_info = None
            for p in position_data:
                if p.get("info", {}).get("symbol") == market_id:
                    pos_info = p.get("info")
                    break

            if not pos_info:
                logger.debug(f"No position found for symbol {market_id} in response.")
                return positions_dict

            size_str = pos_info.get("size", "0")
            side = pos_info.get("side", "None").lower()  # 'Buy'/'Sell'/'None'
            entry_price_str = pos_info.get("avgPrice", "0")  # V5 uses avgPrice
            qty = Decimal(str(size_str))
            entry_price = Decimal(str(entry_price_str)) if entry_price_str != "0" else Decimal("NaN")
            liq_price_str = pos_info.get("liqPrice", "0")
            liq_price = Decimal(str(liq_price_str)) if liq_price_str != "0" else Decimal("NaN")
            unrealized_pnl_str = pos_info.get("unrealisedPnl", "0")
            unrealized_pnl = Decimal(str(unrealized_pnl_str))
            # Extract stop loss and take profit from V5 response
            sl_price_str = pos_info.get("stopLoss", "0")
            tp_price_str = pos_info.get("takeProfit", "0")
            tsl_active_price_str = pos_info.get(
                "trailingStop", "0"
            )  # Trailing stop trigger price in V5 often means it's active
            # Note: 'tpslMode' ('Full' or 'Partial') might also be relevant from pos_info

            position_details = {
                "qty": qty,
                "entry_price": entry_price,
                "liq_price": liq_price,
                "unrealized_pnl": unrealized_pnl,
                "side": side,
                "info": pos_info,  # Store raw info
                "stop_loss_price": Decimal(str(sl_price_str)) if sl_price_str and sl_price_str != "0" else None,
                "take_profit_price": Decimal(str(tp_price_str)) if tp_price_str and tp_price_str != "0" else None,
                "is_tsl_active": bool(tsl_active_price_str and tsl_active_price_str != "0"),
            }

            if side == "buy" and qty.copy_abs() >= self.config.position_qty_epsilon:
                positions_dict["long"] = position_details
                logger.debug(
                    f"Found LONG position: Qty={qty.normalize()}, Entry={entry_price.normalize() if not entry_price.is_nan() else 'N/A'}"
                )
            elif side == "sell" and qty.copy_abs() >= self.config.position_qty_epsilon:
                # Store short quantity as negative for convention? No, V5 gives positive size and 'Sell' side.
                positions_dict["short"] = position_details
                logger.debug(
                    f"Found SHORT position: Qty={qty.normalize()}, Entry={entry_price.normalize() if not entry_price.is_nan() else 'N/A'}"
                )
            else:
                logger.debug("Position size is zero or side is 'None'. Considered flat.")

            return positions_dict

        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}", exc_info=True)
            return None


# --- Indicator Calculator Class ---
class IndicatorCalculator:
    """Calculates technical indicators needed for the strategy."""

    def __init__(self, config: TradingConfig):
        self.config = config

    def calculate_indicators(self, df: pd.DataFrame) -> Optional[Dict[str, Union[Decimal, bool, int]]]:
        """Calculates EMAs, Stochastic, ATR, ADX from OHLCV data."""
        logger.info(f"{Fore.CYAN}# Weaving indicator patterns (EMA, Stoch, ATR, ADX)...")
        if df is None or df.empty:
            logger.error(f"{Fore.RED}No DataFrame for indicators.")
            return None
        try:
            req_cols = ["open", "high", "low", "close"]
            if not all(c in df.columns for c in req_cols):
                logger.error(f"{Fore.RED}DataFrame missing columns: {[c for c in req_cols if c not in df.columns]}")
                return None

            # Work with float copies for performance, convert back to Decimal at the end
            df_calc = df[req_cols].copy()
            df_calc.dropna(inplace=True)
            if df_calc.empty:
                logger.error(f"{Fore.RED}DataFrame empty after NaN drop.")
                return None

            close = df_calc["close"].astype(float).values
            high = df_calc["high"].astype(float).values
            low = df_calc["low"].astype(float).values
            index = df_calc.index

            # Check Data Length
            min_required_len = max(
                self.config.slow_ema_period,
                self.config.trend_ema_period,
                self.config.stoch_period + self.config.stoch_smooth_k + self.config.stoch_smooth_d - 2,
                self.config.atr_period,
                self.config.adx_period * 2,
            )  # ADX needs more data
            if len(df_calc) < min_required_len:
                logger.error(f"{Fore.RED}Insufficient data ({len(df_calc)} < {min_required_len}) for indicators.")
                return None

            # Calculations (float using pandas/numpy)
            close_s = pd.Series(close, index=index)
            high_s = pd.Series(high, index=index)
            low_s = pd.Series(low, index=index)
            fast_ema_s = close_s.ewm(span=self.config.fast_ema_period, adjust=False).mean()
            slow_ema_s = close_s.ewm(span=self.config.slow_ema_period, adjust=False).mean()
            trend_ema_s = close_s.ewm(span=self.config.trend_ema_period, adjust=False).mean()

            # Stochastic
            low_min = low_s.rolling(window=self.config.stoch_period).min()
            high_max = high_s.rolling(window=self.config.stoch_period).max()
            stoch_range = high_max - low_min
            stoch_k_raw = np.where(
                stoch_range > 1e-12, 100 * (close_s - low_min) / stoch_range, 50.0
            )  # Avoid NaN, default 50
            stoch_k_raw_s = pd.Series(stoch_k_raw, index=index).fillna(50)  # Fill NaNs from rolling window
            stoch_k_s = stoch_k_raw_s.rolling(window=self.config.stoch_smooth_k).mean().fillna(50)
            stoch_d_s = stoch_k_s.rolling(window=self.config.stoch_smooth_d).mean().fillna(50)

            # ATR (Wilder's)
            prev_close = close_s.shift(1)
            tr1 = high_s - low_s
            tr2 = (high_s - prev_close).abs()
            tr3 = (low_s - prev_close).abs()
            tr_s = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(0)  # Fill initial NaN
            atr_s = tr_s.ewm(alpha=1 / self.config.atr_period, adjust=False).mean()

            # ADX, +DI, -DI
            adx_s, pdi_s, mdi_s = self._calculate_adx(high_s, low_s, close_s, atr_s, self.config.adx_period)

            # Extract Latest Values & Convert to Decimal
            def get_latest_decimal(series: pd.Series, name: str) -> Decimal:
                if series.empty or series.iloc[-1] is None or pd.isna(series.iloc[-1]):
                    return Decimal("NaN")
                try:
                    return Decimal(str(series.iloc[-1]))
                except (InvalidOperation, TypeError):
                    logger.error(f"Failed converting {name} value {series.iloc[-1]} to Decimal.")
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
            k_prev = get_latest_decimal(stoch_k_s.iloc[-2:-1], "stoch_k_prev")
            d_prev = get_latest_decimal(stoch_d_s.iloc[-2:-1], "stoch_d_prev")
            stoch_kd_bullish = False
            stoch_kd_bearish = False
            if not any(v.is_nan() for v in [k_last, d_last, k_prev, d_prev]):
                crossed_above = (k_last > d_last) and (k_prev <= d_prev)
                crossed_below = (k_last < d_last) and (k_prev >= d_prev)
                prev_oversold = (k_prev <= self.config.stoch_oversold_threshold) or (
                    d_prev <= self.config.stoch_oversold_threshold
                )
                prev_overbought = (k_prev >= self.config.stoch_overbought_threshold) or (
                    d_prev >= self.config.stoch_overbought_threshold
                )
                if crossed_above and prev_oversold:
                    stoch_kd_bullish = True
                if crossed_below and prev_overbought:
                    stoch_kd_bearish = True
            indicators_out["stoch_kd_bullish"] = stoch_kd_bullish
            indicators_out["stoch_kd_bearish"] = stoch_kd_bearish

            # Final check for critical NaNs
            critical_keys = ["fast_ema", "slow_ema", "trend_ema", "atr", "stoch_k", "stoch_d", "adx", "pdi", "mdi"]
            failed = [k for k in critical_keys if indicators_out.get(k, Decimal("NaN")).is_nan()]
            if failed:
                logger.error(f"{Fore.RED}Critical indicators failed (NaN): {', '.join(failed)}.")
                return None

            logger.info(f"{Fore.GREEN}Indicator patterns woven successfully.")
            return indicators_out

        except Exception as e:
            logger.error(f"{Fore.RED}Failed weaving indicator patterns: {e}", exc_info=True)
            return None

    def _calculate_adx(
        self, high_s: pd.Series, low_s: pd.Series, close_s: pd.Series, atr_s: pd.Series, period: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Helper to calculate ADX, +DI, -DI."""
        # Calculate +DM, -DM
        move_up = high_s.diff()
        move_down = low_s.diff().mul(-1)  # diff is current - previous
        plus_dm = np.where((move_up > move_down) & (move_up > 0), move_up, 0)
        minus_dm = np.where((move_down > move_up) & (move_down > 0), move_down, 0)
        # Smoothed DM and ATR
        plus_dm_s = pd.Series(plus_dm, index=high_s.index).ewm(alpha=1 / period, adjust=False).mean()
        minus_dm_s = pd.Series(minus_dm, index=high_s.index).ewm(alpha=1 / period, adjust=False).mean()
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
        self, df_last_candles: pd.DataFrame, indicators: Dict[str, Union[Decimal, bool, int]]
    ) -> Dict[str, Union[bool, str]]:
        """Generates 'long'/'short' entry signals and reason."""
        long_signal, short_signal = False, False
        signal_reason = "No signal - Conditions not met"
        if not indicators:
            return {"long": False, "short": False, "reason": "Indicators missing"}
        if df_last_candles is None or len(df_last_candles) < 2:
            return {"long": False, "short": False, "reason": "Insufficient candle data (<2)"}

        try:
            latest = df_last_candles.iloc[-1]
            current_price = Decimal(str(latest["close"]))
            prev_close = (
                Decimal(str(df_last_candles.iloc[-2]["close"]))
                if not pd.isna(df_last_candles.iloc[-2]["close"])
                else Decimal("NaN")
            )
            if current_price <= 0:
                return {"long": False, "short": False, "reason": "Invalid price (<= 0)"}

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
            nan_keys = [n for n, v in req_vals.items() if isinstance(v, Decimal) and v.is_nan()]
            if nan_keys:
                return {"long": False, "short": False, "reason": f"Required indicator(s) NaN: {', '.join(nan_keys)}"}

            # --- Define Conditions ---
            ema_bullish_cross = fast_ema > slow_ema
            ema_bearish_cross = fast_ema < slow_ema
            trend_buffer = trend_ema.copy_abs() * (self.config.trend_filter_buffer_percent / 100)
            price_above_trend = current_price > trend_ema - trend_buffer
            price_below_trend = current_price < trend_ema + trend_buffer
            stoch_long_cond = (k < self.config.stoch_oversold_threshold) or kd_bull
            stoch_short_cond = (k > self.config.stoch_overbought_threshold) or kd_bear
            # ATR Move Filter
            sig_move, atr_reason = True, "Move Filter OFF"
            if self.config.atr_move_filter_multiplier > 0:
                if atr <= 0:
                    sig_move, atr_reason = False, f"Move Filter Skipped (Invalid ATR: {atr.normalize()})"
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
            potential_long = ema_bullish_cross and stoch_long_cond and sig_move and adx_long_confirm
            potential_short = ema_bearish_cross and stoch_short_cond and sig_move and adx_short_confirm

            reason_parts = []
            if potential_long:
                trend_check = price_above_trend if self.config.trade_only_with_trend else True
                if trend_check:
                    long_signal = True
                    reason_parts = ["EMA Bull", f"Stoch Long ({k:.1f})", adx_reason, atr_reason]
                    if self.config.trade_only_with_trend:
                        reason_parts.insert(2, "Trend OK")
                    signal_reason = "Long Signal: " + " | ".join(p for p in reason_parts if p)
                elif self.config.trade_only_with_trend:
                    signal_reason = f"Long Blocked (Trend): P({current_price.normalize()}) !> T({trend_ema.normalize()})-{self.config.trend_filter_buffer_percent.normalize()}%"
            elif potential_short:
                trend_check = price_below_trend if self.config.trade_only_with_trend else True
                if trend_check:
                    short_signal = True
                    reason_parts = ["EMA Bear", f"Stoch Short ({k:.1f})", adx_reason, atr_reason]
                    if self.config.trade_only_with_trend:
                        reason_parts.insert(2, "Trend OK")
                    signal_reason = "Short Signal: " + " | ".join(reason_parts)
                elif self.config.trade_only_with_trend:
                    signal_reason = f"Short Blocked (Trend): P({current_price.normalize()}) !< T({trend_ema.normalize()})+{self.config.trend_filter_buffer_percent.normalize()}%"
            else:  # Build reason for no signal
                if "Blocked" not in signal_reason:
                    parts = []
                    base_cond = ema_bullish_cross and stoch_long_cond or ema_bearish_cross and stoch_short_cond
                    if not ema_bullish_cross and not ema_bearish_cross:
                        parts.append(f"EMA ({fast_ema:.1f}/{slow_ema:.1f})")
                    elif ema_bullish_cross and not stoch_long_cond:
                        parts.append(f"Stoch !Long ({k:.1f})")
                    elif ema_bearish_cross and not stoch_short_cond:
                        parts.append(f"Stoch !Short ({k:.1f})")
                    if base_cond and not sig_move and self.config.atr_move_filter_multiplier > 0:
                        parts.append(atr_reason)
                    if base_cond and sig_move and not adx_trending:
                        parts.append(f"ADX !Trend ({adx:.1f})")
                    elif base_cond and sig_move and adx_trending and potential_long and not adx_long_confirm:
                        parts.append(f"ADX !Conf Long ({pdi:.1f}/{mdi:.1f})")
                    elif base_cond and sig_move and adx_trending and potential_short and not adx_short_confirm:
                        parts.append(f"ADX !Conf Short ({pdi:.1f}/{mdi:.1f})")
                    signal_reason = "No Signal: " + (" | ".join(p for p in parts if p) if parts else "Conditions unmet")

            log_level_sig = logging.INFO if long_signal or short_signal or "Blocked" in signal_reason else logging.DEBUG
            logger.log(log_level_sig, f"Signal Check: {signal_reason}")
        except Exception as e:
            logger.error(f"{Fore.RED}Error generating signals: {e}", exc_info=True)
            return {"long": False, "short": False, "reason": f"Exception: {e}"}
        return {"long": long_signal, "short": short_signal, "reason": signal_reason}

    def check_exit_signals(self, position_side: str, indicators: Dict[str, Union[Decimal, bool, int]]) -> Optional[str]:
        """Checks for signal-based exits (EMA cross, Stoch reversal). Returns reason string or None."""
        if not indicators:
            return None
        fast_ema = indicators.get("fast_ema", Decimal("NaN"))
        slow_ema = indicators.get("slow_ema", Decimal("NaN"))
        stoch_k = indicators.get("stoch_k", Decimal("NaN"))
        if fast_ema.is_nan() or slow_ema.is_nan() or stoch_k.is_nan():
            logger.warning("Cannot check exit signals due to NaN indicators (EMA/Stoch).")
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
            logger.trade(f"{Fore.YELLOW}Exit Signal vs {position_side.upper()}: {exit_reason}.")
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
        self.protection_tracker = {"long": None, "short": None}  # Values: 'ACTIVE_SLTP', 'ACTIVE_TSL', None

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
            tp_distance_atr = atr * self.config.tp_atr_multiplier if self.config.tp_atr_multiplier > 0 else Decimal("0")

            if side == "buy":
                sl_price = current_price - sl_distance_atr
                tp_price = current_price + tp_distance_atr if tp_distance_atr > 0 else None
            elif side == "sell":
                sl_price = current_price + sl_distance_atr
                tp_price = current_price - tp_distance_atr if tp_distance_atr > 0 else None
            else:
                logger.error(f"Invalid side '{side}' for calculation.")
                return None

            sl_distance_price = (current_price - sl_price).copy_abs()
            if sl_distance_price <= 0:
                logger.error(f"SL distance is zero or negative ({sl_distance_price}). Cannot calculate quantity.")
                return None

            # Quantity = Risk Amount / Stop Loss Distance (per unit)
            quantity = risk_amount / sl_distance_price

            # Format using exchange precision
            sl_price_str = self.exchange_manager.format_price(sl_price)
            tp_price_str = (
                self.exchange_manager.format_price(tp_price) if tp_price is not None else "0"
            )  # Use '0' for API if no TP
            quantity_str = self.exchange_manager.format_amount(quantity, rounding_mode=ROUND_DOWN)

            # Minimum order size check
            min_order_size_str = self.market_info.get("limits", {}).get("amount", {}).get("min")
            if min_order_size_str:
                min_order_size = Decimal(str(min_order_size_str))
                if Decimal(quantity_str) < min_order_size:
                    logger.error(
                        f"Calculated quantity {quantity_str} is below minimum {min_order_size}. Cannot place order."
                    )
                    return None

            logger.info(
                f"Trade Params: Side={side.upper()}, Qty={quantity_str}, Entry~={current_price.normalize()}, SL={sl_price_str}, TP={tp_price_str if tp_price else 'None'}, RiskAmt={risk_amount.normalize()}, ATR={atr.normalize()}"
            )
            return {
                "qty": Decimal(quantity_str),
                "sl_price": Decimal(sl_price_str),
                "tp_price": Decimal(tp_price_str) if tp_price else None,
            }

        except (InvalidOperation, DivisionByZero, TypeError, Exception) as e:
            logger.error(f"Error calculating trade parameters: {e}", exc_info=True)
            return None

    def _execute_market_order(self, side: str, qty_decimal: Decimal) -> Optional[Dict]:
        """Executes a market order with retries and waits for potential fill."""
        if not self.exchange or not self.market_info:
            return None
        symbol = self.config.symbol
        self.market_info.get("id")
        qty_str = self.exchange_manager.format_amount(qty_decimal)
        logger.trade(f"{Fore.CYAN}Attempting MARKET {side.upper()} order: {qty_str} {symbol}...")
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
            logger.trade(
                f"{Fore.GREEN}Market order submitted: ID {order.get('id', 'N/A')}, Side {side.upper()}, Qty {qty_str}"
            )
            termux_notify(f"{symbol} Order Submitted", f"Market {side.upper()} {qty_str}")

            # --- Wait for potential fill ---
            # Simple delay - more robust check would involve fetching order status repeatedly
            logger.info(
                f"Waiting {self.config.order_check_delay_seconds}s for order {order.get('id')} to potentially fill..."
            )
            time.sleep(self.config.order_check_delay_seconds)
            # Optional: Add check_order_status logic here if needed

            return order  # Return submitted order info
        except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e:
            logger.error(f"{Fore.RED}Order placement failed ({type(e).__name__}): {e}")
            termux_notify(f"{symbol} Order Failed", f"Market {side.upper()} failed: {e}")
            return None
        except Exception as e:
            logger.error(f"{Fore.RED}Unexpected error placing market order: {e}", exc_info=True)
            termux_notify(f"{symbol} Order Error", f"Market {side.upper()} error.")
            return None

    def _set_position_protection(
        self,
        position_side: str,
        sl_price: Optional[Decimal] = None,
        tp_price: Optional[Decimal] = None,
        is_tsl: bool = False,
        tsl_distance: Optional[Decimal] = None,
    ) -> bool:
        """Sets SL, TP, or TSL for a position using V5 setTradingStop."""
        if not self.exchange or not self.market_info:
            return False
        symbol = self.config.symbol
        market_id = self.market_info.get("id")
        tracker_key = position_side.lower()

        params = {
            "category": self.config.bybit_v5_category,
            "symbol": market_id,
            "positionIdx": self.config.position_idx,  # 0 for hedge, 1/2 for one-way
            "tpslMode": "Full",  # Apply to the entire position
            "slTriggerBy": self.config.sl_trigger_by,
            "tpTriggerBy": self.config.sl_trigger_by,  # Often same trigger for TP
        }

        action_desc = ""
        if is_tsl and tsl_distance and tsl_distance > 0:
            # Activate Trailing Stop
            params["trailingStop"] = self.exchange_manager.format_price(tsl_distance)
            params["activePrice"] = (
                "0"  # Let Bybit activate based on current price movement (or specify activation price if needed)
            )
            # Clear fixed SL/TP when activating TSL for V5
            params["stopLoss"] = "0"
            params["takeProfit"] = "0"
            action_desc = f"ACTIVATE TSL (Dist: {params['trailingStop']})"
            self.protection_tracker[tracker_key] = "ACTIVE_TSL"
        elif sl_price is not None or tp_price is not None:
            # Set Fixed SL/TP
            params["stopLoss"] = self.exchange_manager.format_price(sl_price) if sl_price and sl_price > 0 else "0"
            params["takeProfit"] = self.exchange_manager.format_price(tp_price) if tp_price and tp_price > 0 else "0"
            # Ensure TSL is cleared if setting fixed stops
            params["trailingStop"] = "0"
            action_desc = f"SET SL={params['stopLoss']} TP={params['takeProfit']}"
            if params["stopLoss"] != "0" or params["takeProfit"] != "0":
                self.protection_tracker[tracker_key] = "ACTIVE_SLTP"
            else:  # Clearing stops
                action_desc = "CLEAR SL/TP"
                self.protection_tracker[tracker_key] = None
        else:
            logger.warning("No valid SL, TP, or TSL provided to _set_position_protection.")
            return False

        logger.trade(f"{Fore.CYAN}Attempting to {action_desc} for {position_side.upper()} {symbol}...")

        try:
            # CCXT might not have a native v5 setTradingStop, use raw call if needed
            # response = self.exchange.private_post_position_set_trading_stop(params) # Ideal if CCXT supports it
            # Fallback to raw call structure (adjust path if needed)
            path = "privatePostPositionSetTradingStop"  # Check CCXT Bybit V5 implementation details
            method = getattr(self.exchange, path, None)
            if not method:
                raise NotImplementedError(f"CCXT method {path} not found for Bybit V5.")

            response = fetch_with_retries(
                method,  # Pass the function
                params=params,  # Pass params dict
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
            )
            # logger.debug(f"SetTradingStop Response: {response}") # Verbose

            # V5 response check (retCode 0 indicates success)
            if response and response.get("retCode") == 0:
                logger.trade(f"{Fore.GREEN}{action_desc} successful for {position_side.upper()} {symbol}.")
                termux_notify(f"{symbol} Protection Set", f"{action_desc} {position_side.upper()}")
                return True
            else:
                ret_code = response.get("retCode", "N/A")
                ret_msg = response.get("retMsg", "Unknown Error")
                logger.error(
                    f"{Fore.RED}{action_desc} failed for {position_side.upper()} {symbol}. Code: {ret_code}, Msg: {ret_msg}"
                )
                # Clear tracker on failure to avoid inconsistent state
                self.protection_tracker[tracker_key] = None
                termux_notify(
                    f"{symbol} Protection FAILED", f"{action_desc} {position_side.upper()} failed: {ret_msg[:50]}"
                )
                return False
        except Exception as e:
            logger.error(
                f"{Fore.RED}Error during {action_desc} for {position_side.upper()} {symbol}: {e}", exc_info=True
            )
            self.protection_tracker[tracker_key] = None  # Clear tracker on exception
            termux_notify(f"{symbol} Protection ERROR", f"{action_desc} {position_side.upper()} error.")
            return False

    def place_risked_market_order(self, side: str, atr: Decimal, total_equity: Decimal, current_price: Decimal) -> bool:
        """Calculates parameters, places market order, and sets initial position SL/TP."""
        if not self.exchange or not self.market_info:
            return False
        params = self._calculate_trade_parameters(side, atr, total_equity, current_price)
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
        time.sleep(self.config.order_check_delay_seconds + 1)  # Give position time to appear
        positions_after_entry = self.exchange_manager.get_current_position()
        if positions_after_entry is None or not positions_after_entry.get(position_side):
            logger.error(
                f"{Fore.RED}Position {position_side.upper()} not found after market order. Manual check required! Potential partial fill or delay."
            )
            # Attempt to close any potentially partially filled position (best effort)
            self._handle_entry_failure(side, qty)
            return False

        filled_qty = positions_after_entry[position_side].get("qty", Decimal("0"))
        avg_entry_price = positions_after_entry[position_side].get("entry_price", Decimal("NaN"))
        logger.info(
            f"Position {position_side.upper()} confirmed: Qty={filled_qty.normalize()}, AvgEntry={avg_entry_price.normalize() if not avg_entry_price.is_nan() else 'N/A'}"
        )

        # Check if filled quantity matches intended quantity (optional, handle partial fills if necessary)
        if (filled_qty - qty).copy_abs() > self.config.position_qty_epsilon * Decimal("10"):  # Allow slight discrepancy
            logger.warning(
                f"{Fore.YELLOW}Partial fill detected? Intended: {qty.normalize()}, Filled: {filled_qty.normalize()}. Proceeding with filled amount."
            )
            # Adjust SL/TP calculations based on actual fill? For simplicity, we use initial calc for now.

        # --- 3. Set Position SL/TP ---
        logger.info(f"Setting initial SL/TP for {position_side.upper()} position...")
        set_stops_ok = self._set_position_protection(position_side, sl_price=sl_price, tp_price=tp_price)
        if not set_stops_ok:
            logger.error(f"{Fore.RED}Failed to set initial SL/TP after entry. Attempting to close position for safety!")
            self.close_position(position_side, filled_qty)  # Close immediately if stops fail
            return False

        # --- 4. Log to Journal ---
        if self.config.enable_journaling and not avg_entry_price.is_nan():
            self.log_trade_entry_to_journal(side, filled_qty, avg_entry_price, order_id)  # Use actual avg entry price

        logger.trade(f"{Fore.GREEN}Entry sequence for {position_side.upper()} completed successfully.")
        return True

    def manage_trailing_stop(
        self, position_side: str, entry_price: Decimal, current_price: Decimal, atr: Decimal
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
        tracker_key = position_side.lower()

        # Check if TSL is already active according to tracker
        if self.protection_tracker.get(tracker_key) == "ACTIVE_TSL":
            logger.debug(f"TSL already marked as active for {position_side.upper()}.")
            return

        try:
            activation_distance_atr = atr * self.config.tsl_activation_atr_multiplier
            tsl_distance_percent = self.config.trailing_stop_percent / 100

            should_activate_tsl = False
            if position_side == "long":
                activation_price = entry_price + activation_distance_atr
                if current_price >= activation_price:
                    should_activate_tsl = True
            elif position_side == "short":
                activation_price = entry_price - activation_distance_atr
                if current_price <= activation_price:
                    should_activate_tsl = True

            if should_activate_tsl:
                logger.trade(f"{Fore.MAGENTA}TSL Activation condition met for {position_side.upper()}!")
                logger.trade(
                    f"  Entry={entry_price.normalize()}, Current={current_price.normalize()}, Activation Target~={activation_price.normalize()}"
                )
                # Calculate TSL distance in price points based on current price and percentage
                tsl_distance_price = current_price * tsl_distance_percent
                if (
                    tsl_distance_price <= self.market_info["precision"]["price_decimal"]
                ):  # Avoid zero/too small distance
                    logger.warning(
                        f"Calculated TSL distance ({tsl_distance_price}) too small, using minimum tick size."
                    )
                    tsl_distance_price = Decimal(
                        str(self.exchange.markets[self.config.symbol]["precision"]["price"])
                    )  # Use market min tick

                logger.trade(
                    f"Attempting to activate {self.config.trailing_stop_percent}% TSL (Distance ~{tsl_distance_price.normalize()})..."
                )

                # Activate TSL using _set_position_protection
                if self._set_position_protection(position_side, is_tsl=True, tsl_distance=tsl_distance_price):
                    logger.trade(f"{Fore.GREEN}Trailing Stop Loss activated successfully for {position_side.upper()}.")
                else:
                    logger.error(f"{Fore.RED}Failed to activate Trailing Stop Loss for {position_side.upper()}.")
                    # State remains as ACTIVE_SLTP or None if activation failed

            else:
                logger.debug(
                    f"TSL activation condition not met for {position_side.upper()} (Current: {current_price.normalize()}, Target: ~{activation_price.normalize()})"
                )

        except Exception as e:
            logger.error(f"Error managing trailing stop for {position_side.upper()}: {e}", exc_info=True)

    def close_position(self, position_side: str, qty_to_close: Decimal) -> bool:
        """Closes the specified position by clearing stops and placing a market order."""
        if not self.exchange or not self.market_info:
            return False
        if qty_to_close.copy_abs() < self.config.position_qty_epsilon:
            logger.warning(f"Close requested for zero/negligible qty ({qty_to_close}). Skipping.")
            return True
        symbol = self.config.symbol
        self.market_info.get("id")
        side_to_close = "sell" if position_side == "long" else "buy"
        tracker_key = position_side.lower()

        logger.trade(
            f"{Fore.YELLOW}Attempting to close {position_side.upper()} position ({qty_to_close.normalize()} {symbol})..."
        )

        # --- 1. Clear existing SL/TP/TSL from the position ---
        logger.info(f"Clearing any existing position protection (SL/TP/TSL) for {position_side.upper()}...")
        # Call set_trading_stop with zeros
        clear_stops_ok = self._set_position_protection(position_side, sl_price=Decimal("0"), tp_price=Decimal("0"))
        if not clear_stops_ok:
            logger.warning(
                f"{Fore.YELLOW}Failed to confirm clearing of position protection. Proceeding with close order anyway..."
            )
        else:
            logger.info("Position protection cleared successfully.")
        self.protection_tracker[tracker_key] = None  # Update tracker regardless of API success for safety

        # --- 2. Place Closing Market Order ---
        logger.info(f"Submitting MARKET {side_to_close.upper()} order to close {position_side.upper()} position...")
        close_order_info = self._execute_market_order(side_to_close, qty_to_close)

        if not close_order_info:
            logger.error(
                f"{Fore.RED}Failed to submit closing market order for {position_side.upper()}. MANUAL INTERVENTION REQUIRED!"
            )
            termux_notify(f"{symbol} CLOSE FAILED", f"Market {side_to_close.upper()} order failed!")
            return False

        close_order_id = close_order_info.get("id", "N/A")
        logger.trade(f"{Fore.GREEN}Closing market order ({close_order_id}) submitted for {position_side.upper()}.")
        termux_notify(f"{symbol} Position Closing", f"{position_side.upper()} close order submitted.")

        # --- 3. Verify Position Closed (Optional but recommended) ---
        logger.info("Waiting briefly and verifying position closure...")
        time.sleep(self.config.order_check_delay_seconds + 2)  # Extra delay for closure
        final_positions = self.exchange_manager.get_current_position()
        if final_positions is not None and not final_positions.get(position_side):
            logger.trade(f"{Fore.GREEN}Position {position_side.upper()} confirmed closed.")
            return True
        elif final_positions is not None and final_positions.get(position_side):
            lingering_qty = final_positions[position_side].get("qty", Decimal("NaN"))
            logger.error(
                f"{Fore.RED}Position {position_side.upper()} still shows qty {lingering_qty.normalize()} after close attempt. MANUAL CHECK REQUIRED!"
            )
            termux_notify(f"{symbol} CLOSE VERIFY FAILED", f"{position_side.upper()} still has position!")
            return False
        else:  # Failed to fetch final positions
            logger.warning(
                f"{Fore.YELLOW}Could not verify position closure for {position_side.upper()} (failed fetch). Assuming closed based on order submission."
            )
            return True  # Assume success if order submitted but verification failed

    def _handle_entry_failure(self, failed_entry_side: str, attempted_qty: Decimal):
        """Attempts to close any potentially opened position after a failed entry sequence."""
        logger.warning(f"{Fore.YELLOW}Handling potential entry failure for {failed_entry_side.upper()}...")
        position_side = "long" if failed_entry_side == "buy" else "short"

        time.sleep(1)  # Short delay before checking again
        positions = self.exchange_manager.get_current_position()

        if positions and positions.get(position_side):
            current_qty = positions[position_side].get("qty", Decimal("0"))
            if current_qty.copy_abs() >= self.config.position_qty_epsilon:
                logger.error(
                    f"{Fore.RED}Detected lingering {position_side.upper()} position (Qty: {current_qty.normalize()}) after entry failure. Attempting emergency close."
                )
                close_success = self.close_position(position_side, current_qty)
                if close_success:
                    logger.info("Emergency close order submitted.")
                else:
                    logger.critical(
                        f"{Fore.RED + Style.BRIGHT}EMERGENCY CLOSE FAILED for {position_side.upper()}. MANUAL INTERVENTION URGENT!"
                    )
            else:
                logger.info("No significant lingering position found after entry failure.")
        elif positions is None:
            logger.error("Could not fetch positions during entry failure handling.")
        else:
            logger.info("No lingering position detected after entry failure.")

    def log_trade_entry_to_journal(self, side: str, qty: Decimal, avg_price: Decimal, order_id: Optional[str]):
        """Logs trade entry details to a CSV file."""
        if not self.config.enable_journaling:
            return
        file_path = self.config.journal_file_path
        file_exists = os.path.isfile(file_path)
        try:
            with open(file_path, "a", newline="", encoding="utf-8") as csvfile:
                fieldnames = ["TimestampUTC", "Symbol", "Action", "Side", "Quantity", "AvgPrice", "OrderID", "Notes"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()  # Write header only if file is new
                writer.writerow(
                    {
                        "TimestampUTC": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        "Symbol": self.config.symbol,
                        "Action": "ENTRY",
                        "Side": side.upper(),
                        "Quantity": qty.normalize(),
                        "AvgPrice": avg_price.normalize(),
                        "OrderID": order_id or "N/A",
                        "Notes": "Initial position entry",
                    }
                )
            logger.debug(f"Trade entry logged to {file_path}")
        except IOError as e:
            logger.error(f"Failed to write entry to journal {file_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error writing entry to journal: {e}", exc_info=True)

    # Add log_trade_exit_to_journal if needed


# --- Status Display Class ---
class StatusDisplay:
    """Handles displaying the bot status using the Rich library."""

    def __init__(self, config: TradingConfig):
        self.config = config

    def print_status_panel(
        self,
        cycle: int,
        timestamp: Optional[pd.Timestamp],
        price: Optional[Decimal],
        indicators: Optional[Dict],
        positions: Optional[Dict],
        equity: Optional[Decimal],
        signals: Dict,
        protection_tracker: Dict,
    ):
        """Prints the status panel to the console."""
        panel_content = ""
        title = f" Cycle {cycle} | {self.config.symbol} ({self.config.interval}) | {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z') if timestamp else 'Timestamp N/A'} "

        # --- Price & Equity ---
        price_str = f"{price.normalize():,}" if price else "[dim]N/A[/]"
        equity_str = f"{equity.normalize():,.2f}" if equity and not equity.is_nan() else "[dim]N/A[/]"
        panel_content += f"[bold cyan]Price:[/] {price_str} | [bold cyan]Equity:[/] {equity_str} {self.config.symbol.split('/')[-1].split(':')[-1]}\n"
        panel_content += "---\n"

        # --- Indicators ---
        if indicators:
            ind_parts = []

            def fmt_ind(val, prec=1):
                return f"{val:.{prec}f}" if isinstance(val, Decimal) and not val.is_nan() else "[dim]N/A[/]"

            ind_parts.append(
                f"EMA(F/S/T): {fmt_ind(indicators.get('fast_ema'))}/{fmt_ind(indicators.get('slow_ema'))}/{fmt_ind(indicators.get('trend_ema'))}"
            )
            ind_parts.append(
                f"Stoch(K/D): {fmt_ind(indicators.get('stoch_k'))}/{fmt_ind(indicators.get('stoch_d'))} {'[bold green]BULL[/]' if indicators.get('stoch_kd_bullish') else ''}{'[bold red]BEAR[/]' if indicators.get('stoch_kd_bearish') else ''}"
            )
            ind_parts.append(f"ATR({indicators.get('atr_period', '?')}): {fmt_ind(indicators.get('atr'), 4)}")
            ind_parts.append(
                f"ADX({self.config.adx_period}): {fmt_ind(indicators.get('adx'))} [+DI:{fmt_ind(indicators.get('pdi'))} -DI:{fmt_ind(indicators.get('mdi'))}]"
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

        if long_pos and long_pos.get("qty", Decimal(0)).copy_abs() >= self.config.position_qty_epsilon:
            entry = long_pos.get("entry_price")
            qty = long_pos.get("qty")
            pnl = long_pos.get("unrealized_pnl", Decimal("NaN"))
            sl = long_pos.get("stop_loss_price")
            tp = long_pos.get("take_profit_price")
            active_protection = protection_tracker.get("long")
            pos_str = (
                f"[bold green]LONG:[/] Qty={qty.normalize()} | Entry={entry.normalize() if entry else '[dim]N/A[/]'} | "
                f"PnL={pnl.normalize() if not pnl.is_nan() else '[dim]N/A[/]'} | "
                f"Prot: {'[magenta]' + active_protection + '[/]' if active_protection else '[dim]None[/]'} "
                f"(SL:{sl.normalize() if sl else 'N/A'} TP:{tp.normalize() if tp else 'N/A'})"
            )
        elif short_pos and short_pos.get("qty", Decimal(0)).copy_abs() >= self.config.position_qty_epsilon:
            entry = short_pos.get("entry_price")
            qty = short_pos.get("qty")
            pnl = short_pos.get("unrealized_pnl", Decimal("NaN"))
            sl = short_pos.get("stop_loss_price")
            tp = short_pos.get("take_profit_price")
            active_protection = protection_tracker.get("short")
            pos_str = (
                f"[bold red]SHORT:[/] Qty={qty.normalize()} | Entry={entry.normalize() if entry else '[dim]N/A[/]'} | "
                f"PnL={pnl.normalize() if not pnl.is_nan() else '[dim]N/A[/]'} | "
                f"Prot: {'[magenta]' + active_protection + '[/]' if active_protection else '[dim]None[/]'} "
                f"(SL:{sl.normalize() if sl else 'N/A'} TP:{tp.normalize() if tp else 'N/A'})"
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
            Panel(Text.from_markup(panel_content), title=f"[bold bright_magenta]{title}[/]", border_style="bright_blue")
        )


# --- Trading Bot Class ---
class TradingBot:
    """Main orchestrator for the trading bot."""

    def __init__(self):
        logger.info(f"{Fore.MAGENTA + Style.BRIGHT}--- Initializing Pyrmethus v2.4.0 ---")
        self.config = TradingConfig()
        self.exchange_manager = ExchangeManager(self.config)
        self.indicator_calculator = IndicatorCalculator(self.config)
        self.signal_generator = SignalGenerator(self.config)
        self.order_manager = OrderManager(self.config, self.exchange_manager)
        self.status_display = StatusDisplay(self.config)
        self.shutdown_requested = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Sets up OS signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        """Internal signal handler to set the shutdown flag."""
        # pylint: disable=unused-argument
        if not self.shutdown_requested:
            sig_name = signal.Signals(sig).name if isinstance(sig, int) else str(sig)
            logger.warning(f"{Fore.YELLOW + Style.BRIGHT}\nSignal {sig_name} received. Initiating graceful shutdown...")
            self.shutdown_requested = True
        else:
            logger.warning("Shutdown already in progress.")

    def run(self):
        """Starts the main trading loop."""
        self._display_startup_info()
        termux_notify("Pyrmethus Started", f"{self.config.symbol} @ {self.config.interval}")
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
                logger.critical(f"{Fore.RED + Style.BRIGHT}CRITICAL AUTH ERROR: {e}. Halting.")
                break  # Exit loop to shutdown
            except Exception as e:  # Catch-all for unexpected cycle errors
                logger.error(
                    f"{Fore.RED + Style.BRIGHT}Unhandled exception in main loop (Cycle {cycle_count}): {e}",
                    exc_info=True,
                )
                logger.error(f"{Fore.RED}Continuing loop, but caution advised. Check logs.")
                termux_notify("Pyrmethus Error", f"Unhandled exception cycle {cycle_count}.")
                sleep_time = self.config.loop_sleep_seconds * 2  # Longer sleep
            else:
                sleep_time = self.config.loop_sleep_seconds  # Normal sleep

            if self.shutdown_requested:
                logger.info("Shutdown requested, breaking main loop.")
                break
            # Interruptible Sleep
            logger.debug(f"Cycle {cycle_count} finished. Sleeping for {sleep_time} seconds...")
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
        logger.info(f"{Fore.MAGENTA + Style.BRIGHT}\n--- Starting Cycle {cycle_count} ---")
        start_time = time.time()
        cycle_success = True

        # --- 1. Fetch Data & Basic Info ---
        df = self.exchange_manager.fetch_ohlcv()
        if df is None or df.empty:
            logger.error(f"{Fore.RED}Cycle Aborted: Market data fetch failed.")
            end_time = time.time()
            logger.info(
                f"{Fore.MAGENTA}--- Cycle {cycle_count} ABORTED (Data Fetch) ({end_time - start_time:.2f}s) ---"
            )
            return

        try:  # Extract Price & Timestamp
            last_candle = df.iloc[-1]
            current_price = Decimal(str(last_candle["close"]))
            last_timestamp = df.index[-1]
            if pd.isna(last_candle["close"]) or current_price <= 0:
                raise ValueError("Invalid latest close price")
            logger.debug(
                f"Latest Candle: {last_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}, Close={current_price.normalize()}"
            )
            # Optional: Add stale data check here
        except (IndexError, KeyError, ValueError, InvalidOperation, TypeError) as e:
            logger.error(f"{Fore.RED}Cycle Aborted: Price/Timestamp error: {e}", exc_info=True)
            end_time = time.time()
            logger.info(
                f"{Fore.MAGENTA}--- Cycle {cycle_count} ABORTED (Price Proc) ({end_time - start_time:.2f}s) ---"
            )
            return

        # --- 2. Calculate Indicators ---
        indicators = self.indicator_calculator.calculate_indicators(df)
        if indicators is None:
            logger.error(f"{Fore.RED}Indicator calculation failed. Logic may be skipped.")
            cycle_success = False
        current_atr = indicators.get("atr", Decimal("NaN")) if indicators else Decimal("NaN")

        # --- 3. Get State (Balance & Positions) ---
        total_equity, _ = self.exchange_manager.get_balance()
        if total_equity is None or total_equity.is_nan() or total_equity <= 0:
            logger.error(f"{Fore.RED}Failed fetching valid equity. Logic may be skipped.")
            cycle_success = False
        positions = self.exchange_manager.get_current_position()
        if positions is None:
            logger.error(f"{Fore.RED}Failed fetching positions. Logic may be skipped.")
            cycle_success = False

        # --- State Snapshot for Panel ---
        protection_tracker_snapshot = copy.deepcopy(self.order_manager.protection_tracker)
        positions_snapshot = positions if positions is not None else {"long": {}, "short": {}}
        final_positions_for_panel = positions_snapshot
        signals: Dict[str, Union[bool, str]] = {
            "long": False,
            "short": False,
            "reason": "Skipped: Initial State",
        }  # Default

        # --- Main Logic Execution ---
        can_run_trade_logic = cycle_success  # Proceed only if initial fetches were ok

        if can_run_trade_logic:
            active_long_pos = positions.get("long", {})
            active_short_pos = positions.get("short", {})
            long_qty = active_long_pos.get("qty", Decimal("0.0"))
            short_qty = active_short_pos.get("qty", Decimal("0.0"))
            long_entry = active_long_pos.get("entry_price", Decimal("NaN"))
            short_entry = active_short_pos.get("entry_price", Decimal("NaN"))
            has_long_pos = long_qty.copy_abs() >= self.config.position_qty_epsilon
            has_short_pos = short_qty.copy_abs() >= self.config.position_qty_epsilon
            is_flat = not has_long_pos and not has_short_pos

            # --- 4. Manage Trailing Stops ---
            if has_long_pos and indicators:
                self.order_manager.manage_trailing_stop("long", long_entry, current_price, current_atr)
            elif has_short_pos and indicators:
                self.order_manager.manage_trailing_stop("short", short_entry, current_price, current_atr)
            elif is_flat:  # Clear tracker if flat
                if self.order_manager.protection_tracker["long"] or self.order_manager.protection_tracker["short"]:
                    logger.debug("Position flat, clearing protection tracker.")
                    self.order_manager.protection_tracker = {"long": None, "short": None}
            protection_tracker_snapshot = copy.deepcopy(
                self.order_manager.protection_tracker
            )  # Update snapshot after TSL logic

            # --- Re-fetch position AFTER TSL check (important if TSL hit) ---
            logger.debug("Re-fetching position state after TSL check...")
            positions_after_tsl = self.exchange_manager.get_current_position()
            if positions_after_tsl is None:
                logger.error(f"{Fore.RED}Failed re-fetching positions after TSL check.")
                cycle_success = False
                signals["reason"] = "Skipped: Pos re-fetch failed"
            else:  # Update live state variables
                active_long_pos = positions_after_tsl.get("long", {})
                active_short_pos = positions_after_tsl.get("short", {})
                long_qty = active_long_pos.get("qty", Decimal("0.0"))
                short_qty = active_short_pos.get("qty", Decimal("0.0"))
                has_long_pos = long_qty.copy_abs() >= self.config.position_qty_epsilon
                has_short_pos = short_qty.copy_abs() >= self.config.position_qty_epsilon
                is_flat = not has_long_pos and not has_short_pos
                logger.debug(
                    f"State After TSL Check: Flat={is_flat}, Long={long_qty.normalize()}, Short={short_qty.normalize()}"
                )
                final_positions_for_panel = positions_after_tsl  # Update panel data
                if is_flat and (
                    self.order_manager.protection_tracker["long"] or self.order_manager.protection_tracker["short"]
                ):  # Clear again if became flat
                    logger.debug("Position became flat after TSL logic, clearing protection tracker.")
                    self.order_manager.protection_tracker = {"long": None, "short": None}
                    protection_tracker_snapshot = copy.deepcopy(self.order_manager.protection_tracker)

                # --- 5. Generate Trading Signals (Entry) ---
                can_gen_signals = indicators is not None and not current_price.is_nan() and len(df) >= 2
                if can_gen_signals:
                    signals = self.signal_generator.generate_signals(df.iloc[-2:], indicators)
                else:
                    signals["reason"] = "Skipped Signal Gen: " + (
                        "Indicators missing" if indicators is None else f"Need >=2 candles ({len(df)} found)"
                    )
                    logger.warning(signals["reason"])

                # --- 6. Check for Exits (Signal-Based) ---
                exit_triggered = False
                exit_side = None
                if can_gen_signals and not is_flat:
                    pos_side_to_check = "long" if has_long_pos else "short"
                    exit_reason = self.signal_generator.check_exit_signals(pos_side_to_check, indicators)
                    if exit_reason:
                        exit_side = pos_side_to_check
                        qty_to_close = long_qty if exit_side == "long" else short_qty
                        exit_triggered = self.order_manager.close_position(exit_side, qty_to_close)

                # --- Re-fetch state AGAIN if an exit was triggered ---
                if exit_triggered:
                    logger.debug(f"Re-fetching state after signal exit attempt ({exit_side})...")
                    positions_after_exit = self.exchange_manager.get_current_position()
                    if positions_after_exit is None:
                        logger.error(f"{Fore.RED}Failed re-fetching positions after signal exit.")
                        cycle_success = False  # Mark failure
                    else:  # Update live state variables
                        active_long_pos = positions_after_exit.get("long", {})
                        active_short_pos = positions_after_exit.get("short", {})
                        long_qty = active_long_pos.get("qty", Decimal("0.0"))
                        short_qty = active_short_pos.get("qty", Decimal("0.0"))
                        has_long_pos = long_qty.copy_abs() >= self.config.position_qty_epsilon
                        has_short_pos = short_qty.copy_abs() >= self.config.position_qty_epsilon
                        is_flat = not has_long_pos and not has_short_pos
                        logger.debug(
                            f"State After Signal Exit: Flat={is_flat}, Long={long_qty.normalize()}, Short={short_qty.normalize()}"
                        )
                        final_positions_for_panel = positions_after_exit  # Update panel data
                        if is_flat:
                            logger.debug("Position became flat after signal exit, clearing protection tracker.")
                            self.order_manager.protection_tracker = {"long": None, "short": None}
                            protection_tracker_snapshot = copy.deepcopy(self.order_manager.protection_tracker)

                # --- 7. Execute Entry Trades (Only if now flat) ---
                if (
                    is_flat
                    and can_gen_signals
                    and not current_atr.is_nan()
                    and (signals.get("long") or signals.get("short"))
                ):
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
                        cycle_success = False
                    # Re-fetch for panel accuracy after entry attempt
                    logger.debug("Re-fetching state after entry attempt...")
                    positions_after_entry = self.exchange_manager.get_current_position()
                    if positions_after_entry is not None:
                        final_positions_for_panel = positions_after_entry
                        protection_tracker_snapshot = copy.deepcopy(self.order_manager.protection_tracker)
                    else:
                        logger.warning("Failed re-fetching positions after entry attempt. Panel may be stale.")
                elif is_flat:
                    logger.debug("Position flat, no entry signal.")
                elif not is_flat:
                    logger.debug(f"Position ({'LONG' if has_long_pos else 'SHORT'}) remains open, skipping entry.")

        else:  # Initial critical data fetch failed
            signals["reason"] = "Skipped: Critical data missing (Equity/Position fetch failed)"
            logger.warning(signals["reason"])

        # --- 8. Display Status Panel ---
        self.status_display.print_status_panel(
            cycle_count,
            last_timestamp,
            current_price,
            indicators,
            final_positions_for_panel,
            total_equity,
            signals,
            protection_tracker_snapshot,
        )

        end_time = time.time()
        status_log = "Complete" if cycle_success else "Completed with WARNINGS/ERRORS"
        logger.info(f"{Fore.MAGENTA}--- Cycle {cycle_count} {status_log} (Duration: {end_time - start_time:.2f}s) ---")

    def graceful_shutdown(self) -> None:
        """Handles cleaning up orders and positions before exiting."""
        logger.warning(f"{Fore.YELLOW + Style.BRIGHT}\nInitiating Graceful Shutdown Sequence...")
        termux_notify("Shutdown", f"Closing {self.config.symbol}...")
        if not self.exchange_manager or not self.exchange_manager.exchange or not self.exchange_manager.market_info:
            logger.error(f"{Fore.RED}Cannot shutdown cleanly: Exchange/Market Info missing.")
            termux_notify("Shutdown Warning!", f"{self.config.symbol} Cannot shutdown cleanly.")
            return

        exchange = self.exchange_manager.exchange
        symbol = self.config.symbol
        market_id = self.exchange_manager.market_info.get("id")

        try:  # --- 1. Cancel All Open Orders ---
            logger.info(f"{Fore.CYAN}Attempting to cancel all open orders for {symbol}...")
            cancelled = False
            try:  # Try V5 cancel all first
                logger.debug("Using V5 cancel_all...")
                params = {
                    "category": self.config.bybit_v5_category,
                    "symbol": market_id,
                    "orderFilter": "Order",
                }  # Cancel active orders
                path = "privatePostOrderCancelAll"
                method = getattr(exchange, path, None)
                if not method:
                    raise NotImplementedError(f"CCXT method {path} not found.")
                response = fetch_with_retries(
                    method, params=params, max_retries=1, delay_seconds=1
                )  # Less retry on shutdown
                if response and response.get("retCode") == 0:
                    logger.info(f"{Fore.GREEN}V5 Cancel all orders successful.")
                    cancelled = True
                    list_cancelled = response.get("result", {}).get("list", [])
                    if list_cancelled:
                        logger.debug(f"Cancelled {len(list_cancelled)} orders.")
                    else:
                        logger.debug("No active orders found to cancel via V5.")
                else:
                    logger.warning(f"V5 Cancel all orders response unclear/failed: {response}")
            except (NotImplementedError, Exception) as e:
                logger.warning(f"V5 cancel_all exception: {e}. Trying generic cancel...")

            if not cancelled:  # Fallback to generic cancel (might not work well for V5 position stops)
                try:
                    logger.info("Attempting generic cancelAllOrders (fallback)...")
                    cancel_resp = fetch_with_retries(
                        exchange.cancel_all_orders, symbol=symbol, max_retries=1, delay_seconds=1
                    )
                    logger.info(f"Generic cancelAllOrders response: {cancel_resp}")
                except Exception as e:
                    logger.error(f"{Fore.RED}Generic cancelAllOrders fallback error: {e}", exc_info=True)
            logger.info("Clearing local protection tracker.")
            self.order_manager.protection_tracker = {"long": None, "short": None}
        except Exception as e:
            logger.error(
                f"{Fore.RED + Style.BRIGHT}Order cancellation phase error: {e}. MANUAL CHECK REQUIRED.", exc_info=True
            )

        logger.info("Waiting after order cancellation...")
        time.sleep(max(self.config.order_check_delay_seconds, 2))

        try:  # --- 2. Close Any Lingering Positions ---
            logger.info(f"{Fore.CYAN}Checking for lingering positions to close...")
            closed_count = 0
            positions = self.exchange_manager.get_current_position()
            if positions is not None:
                positions_to_close = {}
                for side, pos_data in positions.items():
                    if pos_data and pos_data.get("qty", Decimal("0")).copy_abs() >= self.config.position_qty_epsilon:
                        positions_to_close[side] = pos_data
                if not positions_to_close:
                    logger.info(f"{Fore.GREEN}No significant positions found requiring closure.")
                else:
                    logger.warning(
                        f"{Fore.YELLOW}Found {len(positions_to_close)} positions requiring closure: {list(positions_to_close.keys())}"
                    )
                    for side, pos_data in positions_to_close.items():
                        qty = pos_data.get("qty", Decimal("0.0"))
                        logger.warning(f"Attempting to close {side.upper()} position (Qty: {qty.normalize()})...")
                        # Use the OrderManager's close_position method
                        if self.order_manager.close_position(side, qty):
                            closed_count += 1
                            logger.info(f"{Fore.GREEN}Closure initiated for {side.upper()}.")
                        else:
                            logger.error(f"{Fore.RED}Closure failed for {side.upper()}. MANUAL INTERVENTION REQUIRED.")
                    if closed_count == len(positions_to_close):
                        logger.info(f"{Fore.GREEN}All detected positions closed successfully.")
                    else:
                        logger.warning(
                            f"{Fore.YELLOW}Attempted {len(positions_to_close)} closures, {closed_count} closure orders submitted. MANUAL VERIFICATION REQUIRED."
                        )
            else:
                logger.error(f"{Fore.RED}Failed fetching positions during shutdown check. MANUAL CHECK REQUIRED.")
        except Exception as e:
            logger.error(
                f"{Fore.RED + Style.BRIGHT}Position closure phase error: {e}. MANUAL CHECK REQUIRED.", exc_info=True
            )

        logger.warning(f"{Fore.YELLOW + Style.BRIGHT}Graceful Shutdown Sequence Complete. Pyrmethus rests.")
        termux_notify("Shutdown Complete", f"{self.config.symbol} shutdown finished.")

    def _display_startup_info(self):
        """Prints initial configuration details."""
        console.print("[bold bright_cyan] summoning Pyrmethus [magenta]v2.4.0[/]...")
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
    except Exception as main_exception:
        # Catch errors during initialization or the final shutdown phase
        logger.critical(
            f"{Fore.RED + Style.BRIGHT}Critical error during bot initialization or final shutdown: {main_exception}",
            exc_info=True,
        )
        termux_notify("Pyrmethus CRITICAL ERROR", f"Bot failed: {main_exception}")
        sys.exit(1)
