# -*- coding: utf-8 -*-
# pylint: disable=logging-fstring-interpolation, too-many-instance-attributes, too-many-arguments, too-many-locals, too-many-public-methods, invalid-name, unused-argument, too-many-lines, wrong-import-order, wrong-import-position, unnecessary-pass, unnecessary-lambda-assignment, bad-option-value, line-too-long
# fmt: off
#   ____        _       _   _                  _            _         _
#  |  _ \ _   _| |_ ___| | | | __ ___   ____ _| |_ ___  ___| |_ _ __ | |__   ___ _ __ ___  _ __
#  | |_) | | | | __/ _ \ | | |/ _` |\ V / _` | __/ _ \/ __| __| '_ \| '_ \ / _ \ '_ ` _ \| '_ \
#  |  __/| |_| | ||  __/ | | | (_| |\ V / (_| | ||  __/\__ \ |_| |_) | | | |  __/ | | | | | |_) |
#  |_|    \__,_|\__\___|_|_|_|\__,_| \_/ \__,_|\__\___||___/\__| .__/|_| |_| \___|_| |_| |_| .__/
#                                                              |_|                       |_| 
# fmt: on
"""
Pyrmethus - Termux Trading Spell (v4.6.0 - Quantum Nexus Edition)

An advanced trading bot for Bybit Futures using CCXT's async API. Upgrades from v4.5.8 include:
- Async CCXT for non-blocking API calls.
- Multi-timeframe analysis (1m signals, 15m trend confirmation).
- Volatility filter using Bollinger Bands.
- Dynamic risk sizing based on equity and ATR.
- Enhanced SL/TP with retry logic and partial fill adjustments.
- Robust symbol validation and market info caching.
- Telemetry with JSON metrics and Telegram notifications.
- Real-time dashboard with rich.
- Backtesting support for strategy evaluation.
- ML-based signal confidence scoring (logistic regression).
- Multi-symbol trading support.
- Configuration validation and hot-reloading.
"""

# Standard Library Imports
import asyncio
import csv
import json
import logging
import os
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Type
import types

# Third-Party Imports
import ccxt.async_support as ccxt_async
import numpy as np
import pandas as pd
import requests
from colorama import Fore, Style, init as colorama_init
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from sklearn.linear_model import LogisticRegression
from telegram.ext import Application
from telegram import Bot

# Constants
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
DEFAULT_MIN_ADX = Decimal("25")
DEFAULT_BB_WIDTH_MIN = Decimal("0.02")  # Min Bollinger Band width (% of price)
DEFAULT_JOURNAL_FILE = "pyrmethus_trading_journal.csv"
DEFAULT_TELEMETRY_FILE = "pyrmethus_telemetry.json"
V5_UNIFIED_ACCOUNT_TYPE = "UNIFIED"
V5_HEDGE_MODE_POSITION_IDX = 0
V5_TPSL_MODE_FULL = "Full"
V5_SUCCESS_RETCODE = 0
TERMUX_NOTIFY_TIMEOUT = 10
TELEGRAM_MAX_MESSAGE_LEN = 4096
MAX_DRAWDOWN_PERCENT = Decimal("0.20")  # 20% max drawdown
CONFIG_RELOAD_INTERVAL = 300  # Reload config every 5 minutes

# Initialize Colorama & Rich Console
colorama_init(autoreset=True)
console = Console(log_path=False)
getcontext().prec = DECIMAL_PRECISION

# Logging Setup
TRADE_LEVEL_NUM = 25
if not hasattr(logging.Logger, "trade"):
    logging.addLevelName(TRADE_LEVEL_NUM, "TRADE")
    def trade_log(self, message, *args, **kws):
        if self.isEnabledFor(TRADE_LEVEL_NUM):
            self._log(TRADE_LEVEL_NUM, message, args, **kws)
    logging.Logger.trade = trade_log

logger = logging.getLogger(__name__)
log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)-8s] (%(filename)s:%(lineno)d) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logger.setLevel(log_level)
if not logger.hasHandlers():
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
logger.propagate = False

# Utility Functions
def safe_decimal(value: Any, default: Decimal = Decimal("NaN")) -> Decimal:
    if value is None:
        return default
    try:
        str_value = str(value).strip()
        if not str_value or str_value.lower() in ["nan", "none", "null", ""]:
            return default
        return Decimal(str_value)
    except (InvalidOperation, ValueError, TypeError):
        return default

def termux_notify(title: str, content: str) -> None:
    if "com.termux" in os.environ.get("PREFIX", ""):
        try:
            result = subprocess.run(
                ["termux-toast", content],
                check=False,
                timeout=TERMUX_NOTIFY_TIMEOUT,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                error_output = result.stderr or result.stdout
                logger.warning(f"Termux toast failed (code {result.returncode}): {error_output.strip()}")
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
            logger.warning(f"Termux notify failed: {e}")

async def telegram_notify(bot: Bot, chat_id: str, message: str) -> None:
    try:
        if len(message) > TELEGRAM_MAX_MESSAGE_LEN:
            message = message[: TELEGRAM_MAX_MESSAGE_LEN - 3] + "..."
        await bot.send_message(chat_id=chat_id, text=message, parse_mode="HTML")
        logger.debug(f"Telegram notification sent: {message[:50]}...")
    except Exception as e:
        logger.error(f"Telegram notify failed: {e}")

# Configuration Class
class TradingConfig:
    def __init__(self, env_file: str = ".env"):
        logger.debug(f"Loading configuration from {env_file}...")
        self.env_file = env_file
        self.reload_config()
        self._validate_config()

    def reload_config(self):
        env_path = Path(self.env_file)
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path, override=True)
            logger.info(f"Reloaded configuration from {env_path}")
        else:
            logger.warning(f"Environment file '{env_path}' not found.")

        # Symbols (support multiple)
        self.symbols: List[str] = self._get_env(
            "SYMBOLS", "FARTCOINUSDT", Style.DIM, cast_type=lambda x: [s.strip() for s in x.split(",")]
        )
        self.market_type: str = self._get_env(
            "MARKET_TYPE", "linear", Style.DIM, allowed_values=["linear", "inverse", "swap"]
        ).lower()
        self.bybit_v5_category: str = self._determine_v5_category()
        self.interval: str = self._get_env("INTERVAL", "1m", Style.DIM)
        self.confirmation_interval: str = self._get_env("CONFIRMATION_INTERVAL", "15m", Style.DIM)
        self.risk_percentage: Decimal = self._get_env(
            "RISK_PERCENTAGE", DEFAULT_RISK_PERCENT, Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.00001"), max_val=Decimal("0.5")
        )
        self.sl_atr_multiplier: Decimal = self._get_env(
            "SL_ATR_MULTIPLIER", DEFAULT_SL_MULT, Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.1"), max_val=Decimal("20.0")
        )
        self.tp_atr_multiplier: Decimal = self._get_env(
            "TP_ATR_MULTIPLIER", DEFAULT_TP_MULT, Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.0"), max_val=Decimal("50.0")
        )
        self.tsl_activation_atr_multiplier: Decimal = self._get_env(
            "TSL_ACTIVATION_ATR_MULTIPLIER", DEFAULT_TSL_ACT_MULT, Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.1"), max_val=Decimal("20.0")
        )
        self.trailing_stop_percent: Decimal = self._get_env(
            "TRAILING_STOP_PERCENT", DEFAULT_TSL_PERCENT, Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.01"), max_val=Decimal("10.0")
        )
        self.sl_trigger_by: str = self._get_env(
            "SL_TRIGGER_BY", "LastPrice", Style.DIM, allowed_values=["LastPrice", "MarkPrice", "IndexPrice"]
        )
        self.tsl_trigger_by: str = self._get_env(
            "TSL_TRIGGER_BY", "LastPrice", Style.DIM, allowed_values=["LastPrice", "MarkPrice", "IndexPrice"]
        )
        self.position_idx: int = self._get_env(
            "POSITION_IDX", V5_HEDGE_MODE_POSITION_IDX, Style.DIM, cast_type=int, allowed_values=[0, 1, 2]
        )
        self.trend_ema_period: int = self._get_env("TREND_EMA_PERIOD", 12, Style.DIM, cast_type=int, min_val=5, max_val=500)
        self.fast_ema_period: int = self._get_env("FAST_EMA_PERIOD", 9, Style.DIM, cast_type=int, min_val=1, max_val=200)
        self.slow_ema_period: int = self._get_env("SLOW_EMA_PERIOD", 21, Style.DIM, cast_type=int, min_val=2, max_val=500)
        self.stoch_period: int = self._get_env("STOCH_PERIOD", 7, Style.DIM, cast_type=int, min_val=1, max_val=100)
        self.stoch_smooth_k: int = self._get_env("STOCH_SMOOTH_K", 3, Style.DIM, cast_type=int, min_val=1, max_val=10)
        self.stoch_smooth_d: int = self._get_env("STOCH_SMOOTH_D", 3, Style.DIM, cast_type=int, min_val=1, max_val=10)
        self.atr_period: int = self._get_env("ATR_PERIOD", 5, Style.DIM, cast_type=int, min_val=1, max_val=100)
        self.adx_period: int = self._get_env("ADX_PERIOD", 14, Style.DIM, cast_type=int, min_val=2, max_val=100)
        self.bb_period: int = self._get_env("BB_PERIOD", 20, Style.DIM, cast_type=int, min_val=5, max_val=100)
        self.bb_std_dev: Decimal = self._get_env("BB_STD_DEV", Decimal("2.0"), Style.DIM, cast_type=Decimal, min_val=Decimal("1.0"), max_val=Decimal("5.0"))
        self.min_bb_width: Decimal = self._get_env("MIN_BB_WIDTH", DEFAULT_BB_WIDTH_MIN, Style.DIM, cast_type=Decimal, min_val=Decimal("0.0"), max_val=Decimal("0.1"))
        self.stoch_oversold_threshold: Decimal = self._get_env(
            "STOCH_OVERSOLD_THRESHOLD", DEFAULT_STOCH_OVERSOLD, Fore.CYAN, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("45")
        )
        self.stoch_overbought_threshold: Decimal = self._get_env(
            "STOCH_OVERBOUGHT_THRESHOLD", DEFAULT_STOCH_OVERBOUGHT, Fore.CYAN, cast_type=Decimal, min_val=Decimal("55"), max_val=Decimal("100")
        )
        self.trend_filter_buffer_percent: Decimal = self._get_env(
            "TREND_FILTER_BUFFER_PERCENT", Decimal("0.5"), Fore.CYAN, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("5")
        )
        self.atr_move_filter_multiplier: Decimal = self._get_env(
            "ATR_MOVE_FILTER_MULTIPLIER", Decimal("0.5"), Fore.CYAN, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("5")
        )
        self.min_adx_level: Decimal = self._get_env(
            "MIN_ADX_LEVEL", DEFAULT_MIN_ADX, Fore.CYAN, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("90")
        )
        self.api_key: str = self._get_env("BYBIT_API_KEY", None, Fore.RED, is_secret=True)
        self.api_secret: str = self._get_env("BYBIT_API_SECRET", None, Fore.RED, is_secret=True)
        self.telegram_token: str = self._get_env("TELEGRAM_TOKEN", None, Fore.RED, is_secret=True)
        self.telegram_chat_id: str = self._get_env("TELEGRAM_CHAT_ID", None, Fore.RED, is_secret=True)
        self.ohlcv_limit: int = self._get_env("OHLCV_LIMIT", DEFAULT_OHLCV_LIMIT, Style.DIM, cast_type=int, min_val=50, max_val=1000)
        self.loop_sleep_seconds: int = self._get_env("LOOP_SLEEP_SECONDS", DEFAULT_LOOP_SLEEP, Style.DIM, cast_type=int, min_val=1)
        self.order_check_delay_seconds: int = self._get_env("ORDER_CHECK_DELAY_SECONDS", 2, Style.DIM, cast_type=int, min_val=1)
        self.order_fill_timeout_seconds: int = self._get_env("ORDER_FILL_TIMEOUT_SECONDS", 20, Style.DIM, cast_type=int, min_val=5)
        self.max_fetch_retries: int = self._get_env("MAX_FETCH_RETRIES", DEFAULT_MAX_RETRIES, Style.DIM, cast_type=int, min_val=0, max_val=10)
        self.retry_delay_seconds: int = self._get_env("RETRY_DELAY_SECONDS", DEFAULT_RETRY_DELAY, Style.DIM, cast_type=int, min_val=1)
        self.trade_only_with_trend: bool = self._get_env("TRADE_ONLY_WITH_TREND", True, Style.DIM, cast_type=bool)
        self.journal_file_path: str = self._get_env("JOURNAL_FILE_PATH", DEFAULT_JOURNAL_FILE, Style.DIM)
        self.telemetry_file_path: str = self._get_env("TELEMETRY_FILE_PATH", DEFAULT_TELEMETRY_FILE, Style.DIM)
        self.enable_journaling: bool = self._get_env("ENABLE_JOURNALING", True, Style.DIM, cast_type=bool)
        self.enable_telemetry: bool = self._get_env("ENABLE_TELEMETRY", True, Style.DIM, cast_type=bool)

        if not self.api_key or not self.api_secret:
            logger.critical(f"{Fore.RED}Missing BYBIT_API_KEY or BYBIT_API_SECRET. Halting.{Style.RESET_ALL}")
            sys.exit(1)
        if not self.telegram_token or not self.telegram_chat_id:
            logger.warning(f"{Fore.YELLOW}Telegram notifications disabled: Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID.{Style.RESET_ALL}")
            self.telegram_token = None
            self.telegram_chat_id = None

    def _determine_v5_category(self) -> str:
        if self.market_type == "inverse":
            return "inverse"
        if self.market_type in ["linear", "swap"]:
            return "linear"
        raise ValueError(f"Unsupported MARKET_TYPE '{self.market_type}'")

    def _validate_config(self):
        if self.fast_ema_period >= self.slow_ema_period:
            logger.critical(f"{Fore.RED}FAST_EMA ({self.fast_ema_period}) must be < SLOW_EMA ({self.slow_ema_period}). Halting.{Style.RESET_ALL}")
            sys.exit(1)
        if self.trend_ema_period <= self.slow_ema_period:
            logger.warning(f"{Fore.YELLOW}TREND_EMA ({self.trend_ema_period}) <= SLOW_EMA ({self.slow_ema_period}). Consider adjusting periods.{Style.RESET_ALL}")
class ExchangeManager:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.exchange: Optional[ccxt_async.Exchange] = None
        self.markets_cache: Dict[str, Dict] = {}
        self._initialize_exchange()
        self._load_markets()

    async def _initialize_exchange(self):
        logger.info(f"Initializing Bybit async exchange (V5 {self.config.market_type})...")
        try:
            exchange_params = {
                "apiKey": self.config.api_key,
                "secret": self.config.api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": self.config.market_type,
                    "adjustForTimeDifference": True,
                    "recvWindow": 10000,
                    "brokerId": "TermuxQuantumV5",
                },
            }
            self.exchange = ccxt_async.bybit(exchange_params)
            await self.exchange.load_markets()
            logger.info(f"{Fore.GREEN}Bybit V5 async interface initialized.{Style.RESET_ALL}")
        except Exception as e:
            logger.critical(f"{Fore.RED}Exchange initialization failed: {e}. Halting.{Style.RESET_ALL}")
            sys.exit(1)

    async def _load_markets(self):
        try:
            markets = await self.exchange.load_markets(True)
            for symbol in self.config.symbols:
                if symbol not in markets:
                    logger.critical(f"{Fore.RED}Symbol {symbol} not found in markets. Halting.{Style.RESET_ALL}")
                    sys.exit(1)
                market = markets[symbol]
                self.markets_cache[symbol] = {
                    "precision": market.get("precision", {}),
                    "limits": market.get("limits", {}),
                    "contract_size": safe_decimal(market.get("contractSize", "1")),
                    "settle": market.get("settle"),
                    "id": market.get("id"),
                }
                self.markets_cache[symbol]["precision_dp"] = {
                    "amount": self._get_dp(market["precision"].get("amount"), DEFAULT_AMOUNT_DP),
                    "price": self._get_dp(market["precision"].get("price"), DEFAULT_PRICE_DP),
                }
                logger.info(f"Loaded market info for {symbol}: {self.markets_cache[symbol]}")
        except Exception as e:
            logger.critical(f"{Fore.RED}Failed to load markets: {e}. Halting.{Style.RESET_ALL}")
            sys.exit(1)

    def _get_dp(self, precision_val: Optional[Union[str, float, int]], default_dp: int) -> int:
        prec_dec = safe_decimal(precision_val)
        if prec_dec.is_nan() or prec_dec == 0:
            return default_dp
        if prec_dec < 1:
            exponent = prec_dec.as_tuple().exponent
            return abs(int(exponent)) if isinstance(exponent, int) else default_dp
        return int(prec_dec)

    async def validate_symbol(self, symbol: str) -> bool:
        return symbol in self.markets_cache

    async def fetch_ohlcv(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        if not await self.validate_symbol(symbol):
            logger.error(f"Invalid symbol {symbol}")
            return None
        logger.debug(f"Fetching {self.config.ohlcv_limit} {timeframe} candles for {symbol}...")
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=self.config.ohlcv_limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df = df.applymap(safe_decimal)
            df.dropna(subset=["open", "high", "low", "close"], inplace=True)
            if df.empty:
                logger.error(f"No valid OHLCV data for {symbol}")
                return None
            logger.debug(f"Fetched {len(df)} candles for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            return None

    async def get_balance(self) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        settle_currency = self.markets_cache[self.config.symbols[0]].get("settle")
        try:
            params = {"accountType": V5_UNIFIED_ACCOUNT_TYPE, "coin": settle_currency}
            balance_data = await self.exchange.fetch_balance(params=params)
            total_equity = safe_decimal(balance_data["total"].get(settle_currency, "NaN"))
            available_balance = safe_decimal(balance_data["free"].get(settle_currency, "NaN"))
            logger.debug(f"Balance: Equity={total_equity}, Available={available_balance} {settle_currency}")
            return total_equity, available_balance
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return None, None

    async def get_current_position(self, symbol: str) -> Optional[Dict]:
        try:
            params = {"category": self.config.bybit_v5_category, "symbol": symbol}
            positions = await self.exchange.fetch_positions([symbol], params=params)
            pos_data = {"long": {}, "short": {}}
            for pos in positions:
                side = pos.get("side", "").lower()
                if side in ["long", "short"]:
                    pos_data[side] = {
                        "qty": safe_decimal(pos.get("contracts", "0")),
                        "entry_price": safe_decimal(pos.get("entryPrice", "NaN")),
                        "unrealized_pnl": safe_decimal(pos.get("unrealisedPnl", "NaN")),
                        "stop_loss_price": safe_decimal(pos.get("stopLoss", "NaN")),
                        "take_profit_price": safe_decimal(pos.get("takeProfit", "NaN")),
                    }
            logger.debug(f"Position for {symbol}: {pos_data}")
            return pos_data
        except Exception as e:
            logger.error(f"Failed to fetch positions for {symbol}: {e}")
            return None

    def format_price(self, symbol: str, price: Union[Decimal, str, float, int]) -> str:
        price_decimal = safe_decimal(price)
        if price_decimal.is_nan():
            return "NaN"
        precision = self.markets_cache[symbol]["precision_dp"]["price"]
        try:
            quantizer = Decimal(f"1e-{precision}")
            formatted_price = price_decimal.quantize(quantizer, rounding=ROUND_HALF_EVEN)
            return f"{formatted_price:.{precision}f}"
        except Exception as e:
            logger.error(f"Error formatting price {price_decimal} for {symbol}: {e}")
            return "ERR"

    def format_amount(self, symbol: str, amount: Union[Decimal, str, float, int], rounding_mode=ROUND_DOWN) -> str:
        amount_decimal = safe_decimal(amount)
        if amount_decimal.is_nan():
            return "NaN"
        precision = self.markets_cache[symbol]["precision_dp"]["amount"]
        try:
            quantizer = Decimal(f"1e-{precision}")
            formatted_amount = amount_decimal.quantize(quantizer, rounding=rounding_mode)
            return f"{formatted_amount:.{precision}f}"
        except Exception as e:
            logger.error(f"Error formatting amount {amount_decimal} for {symbol}: {e}")
            return "ERR"

    async def close(self):
        if self.exchange:
            await self.exchange.close()
            logger.info("Exchange connection closed.")

class OrderManager:
    def __init__(self, config: TradingConfig, exchange_manager: ExchangeManager):
        self.config = config
        self.exchange_manager = exchange_manager
        self.protection_tracker: Dict[str, Dict[str, Optional[str]]] = {s: {"long": None, "short": None} for s in config.symbols}
        self.exchange = exchange_manager.exchange
        self.markets_cache = exchange_manager.markets_cache

    async def _verify_position_state(self, symbol: str, expected_side: Optional[str], max_attempts: int, delay_seconds: float) -> Tuple[bool, Optional[Dict]]:
        for attempt in range(max_attempts):
            positions = await self.exchange_manager.get_current_position(symbol)
            if positions is None:
                logger.warning(f"Attempt {attempt + 1}/{max_attempts}: Failed to fetch positions for {symbol}")
                await asyncio.sleep(delay_seconds)
                continue
            long_qty = safe_decimal(positions.get("long", {}).get("qty", "0"))
            short_qty = safe_decimal(positions.get("short", {}).get("qty", "0"))
            is_flat = long_qty.copy_abs() < POSITION_QTY_EPSILON and short_qty.copy_abs() < POSITION_QTY_EPSILON
            if expected_side is None:
                return is_flat, positions
            current_side = "long" if long_qty.copy_abs() >= POSITION_QTY_EPSILON else "short" if short_qty.copy_abs() >= POSITION_QTY_EPSILON else None
            if current_side == expected_side:
                return True, positions
            await asyncio.sleep(delay_seconds)
        logger.error(f"Verification failed for {symbol}")
        return False, await self.exchange_manager.get_current_position(symbol)

    async def _set_position_stops(self, symbol: str, position_side: str, qty: Decimal, entry_price: Decimal, sl_price: Decimal, tp_price: Optional[Decimal]) -> bool:
        logger.debug(f"Setting SL/TP for {symbol} {position_side.upper()} (Qty: {qty}, Entry: {entry_price})")
        sl_str = self.exchange_manager.format_price(symbol, sl_price)
        tp_str = self.exchange_manager.format_price(symbol, tp_price) if tp_price else ""
        if not sl_str:
            logger.error(f"Invalid SL price for {symbol}: {sl_price}")
            return False
        params = {
            "category": self.config.bybit_v5_category,
            "symbol": symbol,
            "stopLoss": sl_str,
            "takeProfit": tp_str,
            "slTriggerBy": self.config.sl_trigger_by,
            "tpTriggerBy": self.config.tsl_trigger_by,
            "positionIdx": self.config.position_idx,
            "tpslMode": V5_TPSL_MODE_FULL,
        }
        for attempt in range(self.config.max_fetch_retries):
            try:
                response = await self.exchange.private_post_v5_position_trading_stop(params=params)
                if response.get("retCode") == V5_SUCCESS_RETCODE:
                    logger.info(f"{Fore.GREEN}SL/TP set for {symbol} {position_side.upper()}: SL={sl_str}, TP={tp_str or 'N/A'}{Style.RESET_ALL}")
                    self.protection_tracker[symbol][position_side] = "ACTIVE_SLTP"
                    return True
                logger.warning(f"SL/TP attempt {attempt + 1} failed: {response.get('retMsg')}")
                await asyncio.sleep(self.config.retry_delay_seconds)
            except Exception as e:
                logger.warning(f"SL/TP attempt {attempt + 1} error: {e}")
                await asyncio.sleep(self.config.retry_delay_seconds)
        logger.warning(f"Primary SL/TP failed for {symbol}. Attempting fallback")
        try:
            sl_order_side = "Sell" if position_side == "long" else "Buy"
            sl_order_params = {
                "category": self.config.bybit_v5_category,
                "symbol": symbol,
                "side": sl_order_side,
                "orderType": "Limit",
                "qty": self.exchange_manager.format_amount(symbol, qty),
                "price": sl_str,
                "triggerPrice": sl_str,
                "triggerBy": self.config.sl_trigger_by,
                "positionIdx": self.config.position_idx,
                "reduceOnly": True,
            }
            sl_response = await self.exchange.create_order(symbol, "limit", sl_order_side.lower(), qty, sl_price, sl_order_params)
            logger.info(f"{Fore.GREEN}Fallback SL order placed for {symbol} {position_side.upper()}: {sl_response.get('id')}{Style.RESET_ALL}")
            self.protection_tracker[symbol][position_side] = "ACTIVE_SLTP"
            return True
        except Exception as e:
            logger.error(f"{Fore.RED}Fallback SL failed for {symbol}: {e}{Style.RESET_ALL}")
            return False

    async def place_risked_market_order(self, symbol: str, side: str, atr: Decimal, total_equity: Decimal, current_price: Decimal, confidence: Decimal) -> bool:
        position_side = "long" if side == "buy" else "short"
        risk_factor = min(Decimal("1.5"), max(Decimal("0.5"), confidence))  # Scale risk by confidence
        risk_amount = total_equity * self.config.risk_percentage * risk_factor
        sl_distance = atr * self.config.sl_atr_multiplier
        sl_price = current_price - sl_distance if side == "buy" else current_price + sl_distance
        tp_price = current_price + atr * self.config.tp_atr_multiplier if self.config.tp_atr_multiplier > 0 else None
        try:
            qty = risk_amount / sl_distance * self.markets_cache[symbol]["contract_size"]
            qty = safe_decimal(self.exchange_manager.format_amount(symbol, qty))
            if qty < self.markets_cache[symbol]["limits"]["amount"]["min"]:
                logger.error(f"Order qty {qty} below min for {symbol}")
                return False
            order_params = {
                "category": self.config.bybit_v5_category,
                "symbol": symbol,
                "positionIdx": self.config.position_idx,
                "reduceOnly": False,
            }
            order = await self.exchange.create_order(symbol, "market", side.lower(), qty, None, order_params)
            filled_qty = safe_decimal(order.get("info", {}).get("cumExecQty", qty))
            avg_entry_price = safe_decimal(order.get("avgPrice", current_price))
            fees = safe_decimal(order.get("info", {}).get("cumExecFee", "0"))
            fill_ratio = filled_qty / qty
            if fill_ratio < 0.99:
                logger.warning(f"Partial fill for {symbol}: {filled_qty}/{qty} ({fill_ratio:.2%})")
            if filled_qty < self.markets_cache[symbol]["limits"]["amount"]["min"]:
                logger.error(f"Filled qty {filled_qty} below min for {symbol}")
                await self._handle_entry_failure(symbol, side, filled_qty)
                return False
            logger.info(f"{Fore.GREEN}Order filled for {symbol}: {side.upper()}, Qty={filled_qty}, Price={avg_entry_price}, Fees={fees}{Style.RESET_ALL}")
            if self.config.enable_journaling:
                self.log_trade_entry_to_journal(symbol, side, filled_qty, avg_entry_price, order.get("id"))
            verification_ok, _ = await self._verify_position_state(symbol, position_side, 3, self.config.order_check_delay_seconds)
            if not verification_ok:
                logger.error(f"Position verification failed for {symbol}")
                await self._handle_entry_failure(symbol, side, filled_qty)
                return False
            success = await self._set_position_stops(symbol, position_side, filled_qty, avg_entry_price, sl_price, tp_price)
            if not success:
                logger.error(f"SL/TP set failed for {symbol}")
                await self._handle_entry_failure(symbol, side, filled_qty)
                return False
            logger.trade(f"{Fore.GREEN}Entry successful for {symbol} {position_side.upper()}: Qty={filled_qty}, Entry={avg_entry_price}{Style.RESET_ALL}")
            return True
        except Exception as e:
            logger.error(f"Failed to place order for {symbol}: {e}")
            await self._handle_entry_failure(symbol, side, qty)
            return False

    async def close_position(self, symbol: str, position_side: str, qty: Decimal, reason: str) -> bool:
        logger.info(f"Closing {symbol} {position_side.upper()} (Qty: {qty}, Reason: {reason})")
        close_side = "sell" if position_side == "long" else "buy"
        try:
            await self._clear_position_stops(symbol, position_side)
            order_params = {
                "category": self.config.bybit_v5_category,
                "symbol": symbol,
                "positionIdx": self.config.position_idx,
                "reduceOnly": True,
            }
            order = await self.exchange.create_order(symbol, "market", close_side.lower(), qty, None, order_params)
            avg_close_price = safe_decimal(order.get("avgPrice", "NaN"))
            fees = safe_decimal(order.get("info", {}).get("cumExecFee", "0"))
            logger.info(f"Close order for {symbol}: Price={avg_close_price}, Fees={fees}")
            if self.config.enable_journaling:
                self.log_trade_exit_to_journal(symbol, position_side, qty, avg_close_price, order.get("id"), reason)
            verification_ok, _ = await self._verify_position_state(symbol, None, 5, self.config.order_check_delay_seconds + 1.0)
            if not verification_ok:
                logger.error(f"Closure verification failed for {symbol}")
                termux_notify(f"{symbol} CLOSE FAILED", f"{position_side.upper()} may still be open!")
                return False
            logger.trade(f"{Fore.GREEN}Position {symbol} {position_side.upper()} closed{Style.RESET_ALL}")
            return True
        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {e}")
            return False

    async def _clear_position_stops(self, symbol: str, position_side: str) -> bool:
        try:
            params = {
                "category": self.config.bybit_v5_category,
                "symbol": symbol,
                "stopLoss": "",
                "takeProfit": "",
                "positionIdx": self.config.position_idx,
                "tpslMode": V5_TPSL_MODE_FULL,
            }
            response = await self.exchange.private_post_v5_position_trading_stop(params=params)
            if response.get("retCode") == V5_SUCCESS_RETCODE:
                logger.info(f"SL/TP cleared for {symbol} {position_side.upper()}")
                self.protection_tracker[symbol][position_side] = None
                return True
            logger.warning(f"Failed to clear SL/TP for {symbol}: {response.get('retMsg')}")
            return False
        except Exception as e:
            logger.warning(f"Error clearing SL/TP for {symbol}: {e}")
            return False

    async def _handle_entry_failure(self, symbol: str, failed_entry_side: str, attempted_qty: Decimal):
        position_side = "long" if failed_entry_side == "buy" else "short"
        await asyncio.sleep(self.config.order_check_delay_seconds + 1)
        _, positions = await self._verify_position_state(symbol, None, 2, 1)
        if positions is None:
            logger.error(f"Position check failed for {symbol}")
            termux_notify(f"{symbol} Check Needed", "Failed pos check after entry fail")
            return
        qty = safe_decimal(positions.get(position_side, {}).get("qty", "0"))
        if qty.copy_abs() >= POSITION_QTY_EPSILON:
            logger.error(f"Lingering {symbol} {position_side.upper()} (Qty: {qty})")
            close_success = await self.close_position(symbol, position_side, qty, "EmergencyClose:EntryFail")
            if not close_success:
                logger.critical(f"EMERGENCY CLOSE FAILED for {symbol}")
                termux_notify(f"{symbol} URGENT", "Emergency close FAILED!")
        else:
            logger.info(f"No lingering position for {symbol}")
            self.protection_tracker[symbol][position_side] = None

    def log_trade_entry_to_journal(self, symbol: str, side: str, qty: Decimal, avg_price: Decimal, order_id: Optional[str]):
        position_side = "long" if side == "buy" else "short"
        data = {
            "TimestampUTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "Symbol": symbol,
            "Action": "ENTRY",
            "Side": position_side.upper(),
            "Quantity": qty,
            "AvgPrice": avg_price,
            "OrderID": order_id or "N/A",
            "Reason": "Strategy Signal",
        }
        self._write_journal_row(data)

    def log_trade_exit_to_journal(self, symbol: str, position_side: str, qty: Decimal, avg_price: Decimal, order_id: Optional[str], reason: str):
        data = {
            "TimestampUTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "Symbol": symbol,
            "Action": "EXIT",
            "Side": position_side.upper(),
            "Quantity": qty,
            "AvgPrice": avg_price,
            "OrderID": order_id or "N/A",
            "Reason": reason,
        }
        self._write_journal_row(data)

    def _write_journal_row(self, data: Dict[str, Any]):
        if not self.config.enable_journaling:
            return
        file_path = Path(self.config.journal_file_path)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open("a", newline="", encoding="utf-8") as csvfile:
                fieldnames = ["TimestampUTC", "Symbol", "Action", "Side", "Quantity", "AvgPrice", "OrderID", "Reason", "Notes"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
                if not file_path.exists() or file_path.stat().st_size == 0:
                    writer.writeheader()
                row = {field: str(data.get(field, "N/A")).replace('NaN', 'N/A') for field in fieldnames}
                writer.writerow(row)
            logger.debug(f"Trade {data['Action'].lower()} logged for {data['Symbol']}")
        except Exception as e:
            logger.error(f"Failed to write journal: {e}")

class IndicatorCalculator:
    def __init__(self, config: TradingConfig):
        self.config = config

    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Union[Decimal, np.ndarray]]:
        indicators = {}
        indicators["trend_ema"] = self._ema(df["close"], self.config.trend_ema_period)
        indicators["fast_ema"] = self._ema(df["close"], self.config.fast_ema_period)
        indicators["slow_ema"] = self._ema(df["close"], self.config.slow_ema_period)
        indicators["atr"] = self._atr(df["high"], df["low"], df["close"], self.config.atr_period)
        indicators["adx"] = self._adx(df["high"], df["low"], df["close"], self.config.adx_period)
        indicators["pdi"] = self._plus_di(df["high"], df["low"], df["close"], self.config.adx_period)
        indicators["mdi"] = self._minus_di(df["high"], df["low"], df["close"], self.config.adx_period)
        stoch_k, stoch_d = self._stochastic(df["high"], df["low"], df["close"], self.config.stoch_period, self.config.stoch_smooth_k, self.config.stoch_smooth_d)
        indicators["stoch_k"] = stoch_k
        indicators["stoch_d"] = stoch_d
        bb_upper, bb_lower = self._bollinger_bands(df["close"], self.config.bb_period, self.config.bb_std_dev)
        indicators["bb_width"] = (bb_upper - bb_lower) / df["close"]
        return {k: safe_decimal(v[-1]) if isinstance(v, np.ndarray) else v for k, v in indicators.items()}

    def _ema(self, series: pd.Series, period: int) -> np.ndarray:
        return series.ewm(span=period, adjust=False).mean().values

    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> np.ndarray:
        tr = np.maximum(high - low, np.abs(high - close.shift(1)), np.abs(low - close.shift(1)))
        return tr.rolling(window=period).mean().values

    def _adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> np.ndarray:
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr = self._atr(high, low, close, period)
        plus_di = 100 * plus_dm.rolling(window=period).mean() / tr
        minus_di = 100 * minus_dm.rolling(window=period).mean() / tr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return adx.values

    def _plus_di(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> np.ndarray:
        plus_dm = high.diff()
        plus_dm[plus_dm < 0] = 0
        tr = self._atr(high, low, close, period)
        return 100 * plus_dm.rolling(window=period).mean() / tr

    def _minus_di(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> np.ndarray:
        minus_dm = -low.diff()
        minus_dm[minus_dm < 0] = 0
        tr = self._atr(high, low, close, period)
        return 100 * minus_dm.rolling(window=period).mean() / tr

    def _stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int, smooth_k: int, smooth_d: int) -> Tuple[np.ndarray, np.ndarray]:
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        stoch_k = stoch_k.rolling(window=smooth_k).mean()
        stoch_d = stoch_k.rolling(window=smooth_d).mean()
        return stoch_k.values, stoch_d.values

    def _bollinger_bands(self, series: pd.Series, period: int, std_dev: Decimal) -> Tuple[np.ndarray, np.ndarray]:
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = sma + float(std_dev) * std
        lower = sma - float(std_dev) * std
        return upper.values, lower.values

class SignalGenerator:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.ml_model = LogisticRegression()
        self.is_model_trained = False

    async def train_ml_model(self, df: pd.DataFrame, indicators: Dict[str, pd.Series]):
        if len(df) < 50:
            logger.warning("Insufficient data for ML training")
            return
        X = pd.DataFrame({
            "trend_ema_diff": (df["close"] - indicators["trend_ema"]) / df["close"],
            "fast_ema_diff": (indicators["fast_ema"] - indicators["slow_ema"]) / df["close"],
            "stoch_k": indicators["stoch_k"],
            "adx": indicators["adx"],
            "bb_width": indicators["bb_width"],
        }).dropna()
        y = (df["close"].shift(-5) > df["close"]).astype(int)  # Predict price increase in 5 candles
        if len(X) < 10:
            logger.warning("Not enough valid samples for ML training")
            return
        self.ml_model.fit(X, y)
        self.is_model_trained = True
        logger.info("ML model trained for signal confidence")

    def generate_signals(self, recent_candles: pd.DataFrame, indicators: Dict, confirmation_indicators: Dict) -> Dict[str, Union[bool, str, Decimal]]:
        signals = {"long": False, "short": False, "reason": "No Signal", "confidence": Decimal("0.5")}
        if len(recent_candles) < 2:
            signals["reason"] = "Insufficient candles"
            return signals
        current_price = safe_decimal(recent_candles.iloc[-1]["close"])
        prev_price = safe_decimal(recent_candles.iloc[-2]["close"])
        trend_ema = safe_decimal(indicators.get("trend_ema", "NaN"))
        fast_ema = safe_decimal(indicators.get("fast_ema", "NaN"))
        stoch_k = safe_decimal(indicators.get("stoch_k", "NaN"))
        stoch_d = safe_decimal(indicators.get("stoch_d", "NaN"))
        atr = safe_decimal(indicators.get("atr", "NaN"))
        adx = safe_decimal(indicators.get("adx", "NaN"))
        pdi = safe_decimal(indicators.get("pdi", "NaN"))
        mdi = safe_decimal(indicators.get("mdi", "NaN"))
        bb_width = safe_decimal(indicators.get("bb_width", "NaN"))
        conf_trend_ema = safe_decimal(confirmation_indicators.get("trend_ema", "NaN"))
        if any(v.is_nan() for v in [current_price, trend_ema, fast_ema, stoch_k, stoch_d, atr, adx, bb_width, conf_trend_ema]):
            signals["reason"] = "Invalid indicator data"
            return signals
        price_move = abs(current_price - prev_price)
        atr_move_threshold = atr * self.config.atr_move_filter_multiplier
        trend_buffer = trend_ema * self.config.trend_filter_buffer_percent / Decimal("100")
        is_bullish_trend = current_price > (trend_ema + trend_buffer) and fast_ema > trend_ema and current_price > conf_trend_ema
        is_bearish_trend = current_price < (trend_ema - trend_buffer) and fast_ema < trend_ema and current_price < conf_trend_ema
        stoch_oversold = self.config.stoch_oversold_threshold + (atr / current_price) * 10  # Adaptive threshold
        stoch_overbought = self.config.stoch_overbought_threshold - (atr / current_price) * 10
        is_stoch_long_ok = stoch_k < stoch_oversold and stoch_k > stoch_d
        is_stoch_short_ok = stoch_k > stoch_overbought and stoch_k < stoch_d
        is_trend_ok = not self.config.trade_only_with_trend or (is_bullish_trend or is_bearish_trend)
        is_atr_move_ok = price_move >= atr_move_threshold
        is_adx_ok = adx >= self.config.min_adx_level
        is_volatility_ok = bb_width >= self.config.min_bb_width
        is_direction_ok = pdi > mdi if is_bullish_trend else mdi > pdi
        confidence = Decimal("0.5")
        if self.is_model_trained:
            X = np.array([[
                (current_price - trend_ema) / current_price,
                (fast_ema - indicators["slow_ema"]) / current_price,
                stoch_k,
                adx,
                bb_width,
            ]])
            confidence = Decimal(str(self.ml_model.predict_proba(X)[0][1]))
            confidence = max(Decimal("0.3"), min(Decimal("0.7"), confidence))
        if is_trend_ok and is_atr_move_ok and is_adx_ok and is_volatility_ok and is_direction_ok:
            if is_bullish_trend and is_stoch_long_ok:
                signals["long"] = True
                signals["reason"] = f"Bullish Trend | Stoch(K:{stoch_k:.1f}) | ADX:{adx:.1f} | Conf:{confidence:.2f}"
                signals["confidence"] = confidence
            elif is_bearish_trend and is_stoch_short_ok:
                signals["short"] = True
                signals["reason"] = f"Bearish Trend | Stoch(K:{stoch_k:.1f}) | ADX:{adx:.1f} | Conf:{confidence:.2f}"
                signals["confidence"] = confidence
            else:
                signals["reason"] = f"Stoch(K:{stoch_k:.1f}, L:{is_stoch_long_ok}, S:{is_stoch_short_ok})"
        else:
            signals["reason"] = (
                f"Blocked: Trend({is_trend_ok}) | ATR({is_atr_move_ok}) | "
                f"ADX({is_adx_ok}) | Volatility({is_volatility_ok}) | Dir({is_direction_ok})"
            )
        return signals

class TelemetryManager:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.metrics = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": Decimal("0"),
            "max_drawdown": Decimal("0"),
            "peak_equity": Decimal("0"),
            "current_equity": Decimal("0"),
        }
        self.trade_history: List[Dict] = []

    def update_metrics(self, trade: Dict):
        if not self.config.enable_telemetry:
            return
        self.metrics["trades"] += 1
        entry_price = safe_decimal(trade.get("entry_price", "NaN"))
        exit_price = safe_decimal(trade.get("exit_price", "NaN"))
        qty = safe_decimal(trade.get("quantity", "0"))
        side = trade.get("side", "").lower()
        fees = safe_decimal(trade.get("fees", "0"))
        if entry_price.is_nan() or exit_price.is_nan() or qty.is_nan():
            logger.warning(f"Invalid trade data for telemetry: {trade}")
            return
        pnl = (exit_price - entry_price) * qty if side == "long" else (entry_price - exit_price) * qty
        pnl -= fees
        self.metrics["total_pnl"] += pnl
        self.metrics["wins" if pnl > 0 else "losses"] += 1
        self.trade_history.append({
            "timestamp": trade.get("timestamp", datetime.now(timezone.utc)),
            "symbol": trade.get("symbol", ""),
            "side": side,
            "pnl": pnl,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "qty": qty,
            "fees": fees,
        })
        self._write_telemetry()

    def update_equity(self, equity: Decimal):
        if equity.is_nan() or equity <= 0:
            return
        self.metrics["current_equity"] = equity
        self.metrics["peak_equity"] = max(self.metrics["peak_equity"], equity)
        drawdown = (self.metrics["peak_equity"] - equity) / self.metrics["peak_equity"]
        self.metrics["max_drawdown"] = max(self.metrics["max_drawdown"], drawdown)
        self._write_telemetry()

    def _write_telemetry(self):
        if not self.config.enable_telemetry:
            return
        file_path = Path(self.config.telemetry_file_path)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(self.metrics, f, default=str)
            logger.debug("Telemetry updated")
        except Exception as e:
            logger.error(f"Failed to write telemetry: {e}")

    def get_summary(self) -> Dict:
        win_rate = self.metrics["wins"] / self.metrics["trades"] if self.metrics["trades"] > 0 else 0
        profit_factor = (
            sum(t["pnl"] for t in self.trade_history if t["pnl"] > 0) /
            abs(sum(t["pnl"] for t in self.trade_history if t["pnl"] < 0))
            if any(t["pnl"] < 0 for t in self.trade_history) else float("inf")
        )
        return {
            "trades": self.metrics["trades"],
            "win_rate": f"{win_rate:.2%}",
            "total_pnl": self.metrics["total_pnl"],
            "profit_factor": f"{profit_factor:.2f}",
            "max_drawdown": f"{self.metrics['max_drawdown']:.2%}",
            "current_equity": self.metrics["current_equity"],
        }

class BacktestEngine:
    def __init__(self, config: TradingConfig, exchange_manager: ExchangeManager, signal_generator: SignalGenerator, indicator_calculator: IndicatorCalculator):
        self.config = config
        self.exchange_manager = exchange_manager
        self.signal_generator = signal_generator
        self.indicator_calculator = indicator_calculator

    async def run_backtest(self, symbol: str, start_date: str, end_date: str) -> Dict:
        logger.info(f"Running backtest for {symbol} from {start_date} to {end_date}")
        df = await self.exchange_manager.fetch_ohlcv(symbol, self.config.interval)
        if df is None or df.empty:
            logger.error("No data for backtest")
            return {}
        df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
        if df.empty:
            logger.error("No data in specified date range")
            return {}
        initial_equity = Decimal("1000")
        equity = initial_equity
        trades = []
        positions = {"long": None, "short": None}
        for i in range(len(df) - 1):
            recent_candles = df.iloc[:i + 1]
            indicators = self.indicator_calculator.calculate_indicators(recent_candles)
            signals = self.signal_generator.generate_signals(recent_candles, indicators, indicators)
            current_price = safe_decimal(recent_candles.iloc[-1]["close"])
            atr = indicators.get("atr", Decimal("0.001"))
            if signals["long"] and not positions["long"]:
                qty = (equity * self.config.risk_percentage) / (atr * self.config.sl_atr_multiplier)
                sl_price = current_price - atr * self.config.sl_atr_multiplier
                tp_price = current_price + atr * self.config.tp_atr_multiplier if self.config.tp_atr_multiplier > 0 else None
                positions["long"] = {
                    "qty": qty,
                    "entry_price": current_price,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                }
            elif signals["short"] and not positions["short"]:
                qty = (equity * self.config.risk_percentage) / (atr * self.config.sl_atr_multiplier)
                sl_price = current_price + atr * self.config.sl_atr_multiplier
                tp_price = current_price - atr * self.config.tp_atr_multiplier if self.config.tp_atr_multiplier > 0 else None
                positions["short"] = {
                    "qty": qty,
                    "entry_price": current_price,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                }
            next_price = safe_decimal(df.iloc[i + 1]["close"])
            for side in ["long", "short"]:
                if positions[side]:
                    pos = positions[side]
                    if side == "long":
                        if next_price <= pos["sl_price"] or (pos["tp_price"] and next_price >= pos["tp_price"]):
                            pnl = (next_price - pos["entry_price"]) * pos["qty"]
                            equity += pnl
                            trades.append({
                                "timestamp": df.index[i + 1],
                                "symbol": symbol,
                                "side": side,
                                "pnl": pnl,
                                "entry_price": pos["entry_price"],
                                "exit_price": next_price,
                                "qty": pos["qty"],
                                "fees": Decimal("0"),
                            })
                            positions[side] = None
                    else:
                        if next_price >= pos["sl_price"] or (pos["tp_price"] and next_price <= pos["tp_price"]):
                            pnl = (pos["entry_price"] - next_price) * pos["qty"]
                            equity += pnl
                            trades.append({
                                "timestamp": df.index[i + 1],
                                "symbol": symbol,
                                "side": side,
                                "pnl": pnl,
                                "entry_price": pos["entry_price"],
                                "exit_price": next_price,
                                "qty": pos["qty"],
                                "fees": Decimal("0"),
                            })
                            positions[side] = None
        telemetry = TelemetryManager(self.config)
        for trade in trades:
            telemetry.update_metrics(trade)
        telemetry.update_equity(equity)
        return telemetry.get_summary()

class StatusDisplay:
    def __init__(self, config: TradingConfig, telemetry: TelemetryManager):
        self.config = config
        self.telemetry = telemetry
        self.live = Live(auto_refresh=False, console=console)

    def start(self):
        self.live.start()

    def stop(self):
        self.live.stop()

    def update(self, symbol_states: Dict[str, Dict], equity: Decimal, cycle_count: int):
        table = Table(show_header=True, header_style="bold bright_cyan", box=None)
        table.add_column("Metric", style="bright_yellow")
        table.add_column("Value", style="bright_white")
        table.add_row("Cycle", str(cycle_count))
        table.add_row("Equity", f"{equity:.2f}")
        table.add_row("Trades", str(self.telemetry.metrics["trades"]))
        table.add_row("Win Rate", self.telemetry.get_summary()["win_rate"])
        table.add_row("Total PnL", f"{self.telemetry.metrics['total_pnl']:.2f}")
        table.add_row("Max Drawdown", self.telemetry.get_summary()["max_drawdown"])
        for symbol, state in symbol_states.items():
            signals = state.get("signals", {})
            table.add_row(f"{symbol} Signal", signals.get("reason", "N/A"))
            table.add_row(f"{symbol} Confidence", f"{signals.get('confidence', Decimal('0')):.2f}")
            positions = state.get("positions", {})
            for side in ["long", "short"]:
                pos = positions.get(side, {})
                if pos.get("qty", Decimal("0")).copy_abs() >= POSITION_QTY_EPSILON:
                    table.add_row(f"{symbol} {side.capitalize()} Pos", f"Qty: {pos['qty']}, Entry: {pos['entry_price']}")
        self.live.update(Panel(table, title="Pyrmethus Dashboard", border_style="bright_magenta"))
        self.live.refresh()

class TradingBot:
    def __init__(self):
        logger.info(f"{Fore.MAGENTA}Initializing Pyrmethus v4.6.0 (Quantum Nexus Edition)...{Style.RESET_ALL}")
        self.config = TradingConfig()
        self.exchange_manager = ExchangeManager(self.config)
        self.indicator_calculator = IndicatorCalculator(self.config)
        self.signal_generator = SignalGenerator(self.config)
        self.order_manager = OrderManager(self.config, self.exchange_manager)
        self.telemetry = TelemetryManager(self.config)
        self.status_display = StatusDisplay(self.config, self.telemetry)
        self.backtest_engine = BacktestEngine(self.config, self.exchange_manager, self.signal_generator, self.indicator_calculator)
        self.shutdown_requested = False
        self.telegram_bot = Bot(self.config.telegram_token) if self.config.telegram_token else None
        self._setup_signal_handlers()
        logger.info(f"{Fore.GREEN}Pyrmethus initialized{Style.RESET_ALL}")

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        logger.debug("Signal handlers set")

    def _signal_handler(self, sig: int, frame: Optional[types.FrameType]):
        if not self.shutdown_requested:
            sig_name = signal.Signals(sig).name
            console.print(f"\n[bold yellow]Signal {sig_name} received. Shutting down...[/]")
            self.shutdown_requested = True

    async def graceful_shutdown(self):
        console.print("\n[bold yellow]Initiating Graceful Shutdown...[/]")
        termux_notify("Pyrmethus Shutdown", "Closing positions...")
        for symbol in self.config.symbols:
            try:
                params = {"category": self.config.bybit_v5_category, "symbol": symbol}
                await self.exchange_manager.exchange.cancel_all_orders(symbol, params=params)
                logger.info(f"Cancelled orders for {symbol}")
            except Exception as e:
                logger.error(f"Failed to cancel orders for {symbol}: {e}")
            positions = await self.exchange_manager.get_current_position(symbol)
            if positions:
                for side in ["long", "short"]:
                    qty = safe_decimal(positions.get(side, {}).get("qty", "0"))
                    if qty.copy_abs() >= POSITION_QTY_EPSILON:
                        await self.order_manager.close_position(symbol, side, qty, "GracefulShutdown")
        await self.exchange_manager.close()
        self.status_display.stop()
        if self.telegram_bot:
            summary = self.telemetry.get_summary()
            message = (
                f"<b>Pyrmethus Shutdown</b>\n"
                f"Trades: {summary['trades']}\n"
                f"Win Rate: {summary['win_rate']}\n"
                f"Total PnL: {summary['total_pnl']:.2f}\n"
                f"Max Drawdown: {summary['max_drawdown']}"
            )
            await telegram_notify(self.telegram_bot, self.config.telegram_chat_id, message)
        console.print("[bold yellow]Shutdown Complete.[/]")
        logger.warning(f"{Fore.YELLOW}Shutdown Complete{Style.RESET_ALL}")

    async def trading_spell_cycle(self, cycle_count: int, symbol: str) -> Dict:
        state = {"signals": {}, "positions": {}, "status": "OK"}
        df = await self.exchange_manager.fetch_ohlcv(symbol, self.config.interval)
        conf_df = await self.exchange_manager.fetch_ohlcv(symbol, self.config.confirmation_interval)
        if df is None or df.empty or conf_df is None or conf_df.empty:
            state["status"] = "FAIL:FETCH_OHLCV"
            return state
        indicators = self.indicator_calculator.calculate_indicators(df)
        conf_indicators = self.indicator_calculator.calculate_indicators(conf_df)
        if not self.signal_generator.is_model_trained:
            await self.signal_generator.train_ml_model(df, {k: pd.Series(v) for k, v in indicators.items()})
        signals = self.signal_generator.generate_signals(df, indicators, conf_indicators)
        state["signals"] = signals
        total_equity, _ = await self.exchange_manager.get_balance()
        self.telemetry.update_equity(total_equity)
        if total_equity is None or total_equity.is_nan() or total_equity <= 0:
            state["status"] = "FAIL:FETCH_EQUITY"
            return state
        drawdown = (self.telemetry.metrics["peak_equity"] - total_equity) / self.telemetry.metrics["peak_equity"]
        if drawdown > MAX_DRAWDOWN_PERCENT:
            logger.error(f"Max drawdown exceeded: {drawdown:.2%}")
            state["status"] = "FAIL:MAX_DRAWDOWN"
            if self.telegram_bot:
                await telegram_notify(self.telegram_bot, self.config.telegram_chat_id, f"<b>Max Drawdown Exceeded</b>: {drawdown:.2%}")
            self.shutdown_requested = True
            return state
        positions = await self.exchange_manager.get_current_position(symbol)
        state["positions"] = positions
        if positions is None:
            state["status"] = "FAIL:FETCH_POSITIONS"
            return state
        long_qty = safe_decimal(positions.get("long", {}).get("qty", "0"))
        short_qty = safe_decimal(positions.get("short", {}).get("qty", "0"))
        if long_qty.copy_abs() >= POSITION_QTY_EPSILON or short_qty.copy_abs() >= POSITION_QTY_EPSILON:
            logger.debug(f"Position exists for {symbol}: Long={long_qty}, Short={short_qty}")
            return state
        current_price = safe_decimal(df.iloc[-1]["close"])
        atr = indicators.get("atr", Decimal("0.001"))
        if signals["long"]:
            await self.order_manager.place_risked_market_order(symbol, "buy", atr, total_equity, current_price, signals["confidence"])
        elif signals["short"]:
            await self.order_manager.place_risked_market_order(symbol, "sell", atr, total_equity, current_price, signals["confidence"])
        return state

    async def run(self):
        self._display_startup_info()
        termux_notify("Pyrmethus Started", f"Trading {', '.join(self.config.symbols)}")
        if self.telegram_bot:
            await telegram_notify(self.telegram_bot, self.config.telegram_chat_id, f"<b>Pyrmethus Started</b>: Trading {', '.join(self.config.symbols)}")
        self.status_display.start()
        cycle_count = 0
        last_config_reload = time.monotonic()
        while not self.shutdown_requested:
            cycle_count += 1
            cycle_start = time.monotonic()
            if time.monotonic() - last_config_reload > CONFIG_RELOAD_INTERVAL:
                self.config.reload_config()
                last_config_reload = time.monotonic()
            symbol_states = {}
            total_equity, _ = await self.exchange_manager.get_balance()
            tasks = [self.trading_spell_cycle(cycle_count, symbol) for symbol in self.config.symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for symbol, result in zip(self.config.symbols, results):
                if isinstance(result, Exception):
                    logger.error(f"Cycle {cycle_count} failed for {symbol}: {result}")
                    symbol_states[symbol] = {"status": "ERROR", "signals": {}, "positions": {}}
                else:
                    symbol_states[symbol] = result
            self.status_display.update(symbol_states, total_equity or Decimal("0"), cycle_count)
            cycle_duration = time.monotonic() - cycle_start
            sleep_duration = max(0, self.config.loop_sleep_seconds - cycle_duration)
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)
        await self.graceful_shutdown()
        console.print(f"\n[bold bright_cyan]Pyrmethus terminated.[/]")
        sys.exit();