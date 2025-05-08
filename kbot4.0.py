

Loaded configuration from config.json
2025-04-29 16:54:43,031 [INFO    ] (kbot4.0.py:204) Fetched SYMBOL: FARTCOIN/USDT:USDT
2025-04-29 16:54:43,031 [INFO    ] (kbot4.0.py:204) Fetched MARKET_TYPE: linear
2025-04-29 16:54:43,031 [INFO    ] (kbot4.0.py:204) Fetched INTERVAL: 3
2025-04-29 16:54:43,031 [INFO    ] (kbot4.0.py:204) Fetched RISK_PERCENTAGE: 0.01
2025-04-29 16:54:43,031 [INFO    ] (kbot4.0.py:204) Fetched SL_ATR_MULTIPLIER: 1.5
2025-04-29 16:54:43,031 [INFO    ] (kbot4.0.py:204) Fetched TSL_ACTIVATION_ATR_MULTIPLIER: 1.0
2025-04-29 16:54:43,032 [INFO    ] (kbot4.0.py:204) Fetched TRAILING_STOP_PERCENT: 0.5
2025-04-29 16:54:43,032 [WARNING ] (kbot4.0.py:201) Using default for TAKE_PROFIT_ATR_MULTIPLIERS: 2.0,4.0
2025-04-29 16:54:43,032 [WARNING ] (kbot4.0.py:201) Using default for MAX_POSITION_PERCENTAGE: 0.5
2025-04-29 16:54:43,032 [INFO    ] (kbot4.0.py:204) Fetched SL_TRIGGER_BY: LastPrice
2025-04-29 16:54:43,032 [INFO    ] (kbot4.0.py:204) Fetched TSL_TRIGGER_BY: LastPrice
2025-04-29 16:54:43,032 [INFO    ] (kbot4.0.py:204) Fetched LEVERAGE: 25
2025-04-29 16:54:43,032 [WARNING ] (kbot4.0.py:201) Using default for DYNAMIC_LEVERAGE: False
2025-04-29 16:54:43,033 [CRITICAL] (kbot4.0.py:245) Failed to cast DYNAMIC_LEVERAGE ('False'): [<class 'decimal.ConversionSyntax'>]. Using default: False
2025-04-29 16:54:43,033 [WARNING ] (kbot4.0.py:201) Using default for MULTI_TIMEFRAME_INTERVAL: 5m
2025-04-29 16:54:43,033 [INFO    ] (kbot4.0.py:204) Fetched TREND_EMA_PERIOD: 12
2025-04-29 16:54:43,033 [INFO    ] (kbot4.0.py:204) Fetched FAST_EMA_PERIOD: 5
2025-04-29 16:54:43,033 [INFO    ] (kbot4.0.py:204) Fetched SLOW_EMA_PERIOD: 12
2025-04-29 16:54:43,033 [INFO    ] (kbot4.0.py:204) Fetched STOCH_PERIOD: 7
2025-04-29 16:54:43,033 [INFO    ] (kbot4.0.py:204) Fetched STOCH_SMOOTH_K: 3
2025-04-29 16:54:43,033 [INFO    ] (kbot4.0.py:204) Fetched STOCH_SMOOTH_D: 3
2025-04-29 16:54:43,034 [INFO    ] (kbot4.0.py:204) Fetched ATR_PERIOD: 10
2025-04-29 16:54:43,034 [INFO    ] (kbot4.0.py:204) Fetched STOCH_OVERSOLD_THRESHOLD: 31
2025-04-29 16:54:43,034 [INFO    ] (kbot4.0.py:204) Fetched STOCH_OVERBOUGHT_THRESHOLD: 69
2025-04-29 16:54:43,034 [INFO    ] (kbot4.0.py:204) Fetched TREND_FILTER_BUFFER_PERCENT: 0.5
2025-04-29 16:54:43,034 [INFO    ] (kbot4.0.py:204) Fetched ATR_MOVE_FILTER_MULTIPLIER: 0.5
2025-04-29 16:54:43,034 [WARNING ] (kbot4.0.py:201) Using default for VOLATILITY_THRESHOLD: 3.0
2025-04-29 16:54:43,034 [INFO    ] (kbot4.0.py:204) Fetched BYBIT_API_KEY: ****
2025-04-29 16:54:43,034 [INFO    ] (kbot4.0.py:204) Fetched BYBIT_API_SECRET: ****
2025-04-29 16:54:43,035 [CRITICAL] (kbot4.0.py:207) Required configuration 'TELEGRAM_TOKEN' missing.
%

 remove telegram from bot and implement termux api sms
import os
import time
import logging
import sys
import subprocess
import csv
import json
from datetime import datetime
from typing import Dict, Optional, Any, Tuple, Union, List
from decimal import Decimal, getcontext, ROUND_DOWN, ROUND_HALF_EVEN
import asyncio
import signal
import platform
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt

# Import enchantments with fallback guidance
try:
    import ccxt.async_support as ccxt
    from dotenv import load_dotenv
    import pandas as pd
    import numpy as np
    from tabulate import tabulate
    from colorama import init, Fore, Style
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich import box
    COMMON_PACKAGES = ['ccxt', 'python-dotenv', 'pandas', 'numpy', 'tabulate', 'colorama', 'rich', 'matplotlib']
except ImportError as e:
    init(autoreset=True)
    missing_pkg = e.name
    print(f"{Fore.RED}{Style.BRIGHT}Missing package: {missing_pkg}")
    print(f"{Fore.YELLOW}Install with: pip install {' '.join(COMMON_PACKAGES)}")
    sys.exit(1)

# Set Decimal precision
getcontext().prec = 50

# --- Logging Setup ---
logger = logging.getLogger(__name__)
TRADE_LEVEL_NUM = logging.INFO + 5
logging.addLevelName(TRADE_LEVEL_NUM, "TRADE")

def trade_log(self, message, *args, **kws):
    """Custom logging for trade events."""
    if self.isEnabledFor(TRADE_LEVEL_NUM):
        self._log(TRADE_LEVEL_NUM, message, args, **kws)

logging.Logger.trade = trade_log

log_formatter = logging.Formatter(
    Fore.CYAN + "%(asctime)s " + Style.BRIGHT + "[%(levelname)-8s] " +
    Fore.WHITE + "(%(filename)s:%(lineno)d) " + Style.RESET_ALL +
    Fore.WHITE + "%(message)s" + Style.RESET_ALL
)

log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
valid_log_levels = ["DEBUG", "INFO", "TRADE", "WARNING", "ERROR", "CRITICAL"]
if log_level_str not in valid_log_levels:
    print(f"{Fore.YELLOW}Invalid LOG_LEVEL '{log_level_str}'. Defaulting to INFO.")
    log_level_str = "INFO"

log_level = TRADE_LEVEL_NUM if log_level_str == "TRADE" else getattr(logging, log_level_str)
logger.setLevel(log_level)

if not any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout for h in logger.handlers):
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

logger.propagate = False

# --- Journaling ---
class JournalManager:
    """Manages trade journaling to CSV."""
    def __init__(self, file_path: str, enabled: bool):
        self.file_path = file_path
        self.enabled = enabled
        self.headers = ['timestamp', 'symbol', 'side', 'price', 'quantity', 'pnl', 'reason']
        if enabled and not os.path.exists(file_path):
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log_trade(self, trade: Dict):
        """Logs a trade to CSV."""
        if not self.enabled:
            return
        try:
            with open(self.file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    trade.get('timestamp', datetime.now().isoformat()),
                    trade.get('symbol', ''),
                    trade.get('side', ''),
                    trade.get('price', ''),
                    trade.get('quantity', ''),
                    trade.get('pnl', ''),
                    trade.get('reason', '')
                ])
            logger.debug(f"Logged trade to {self.file_path}")
        except Exception as e:
            logger.error(f"{Fore.RED}Failed to log trade: {e}")

# --- Configuration ---
class TradingConfig:
    """Manages trading bot configuration with validation."""
    def __init__(self, config_file: Optional[str] = None):
        logger.debug("Loading configuration...")
        load_dotenv()
        self.config_file = config_file
        self._load_config_file()
        self.symbol = self._get_env("SYMBOL", "BTC/USDT:USDT", Fore.YELLOW)
        self.market_type = self._get_env("MARKET_TYPE", "linear", Fore.YELLOW, allowed_values=['linear', 'inverse', 'swap']).lower()
        self.interval = self._get_env("INTERVAL", "1m", Fore.YELLOW)
        self.risk_percentage = self._get_env("RISK_PERCENTAGE", "0.01", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.00001"), max_val=Decimal("0.5"))
        self.sl_atr_multiplier = self._get_env("SL_ATR_MULTIPLIER", "1.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.1"), max_val=Decimal("20.0"))
        self.tsl_activation_atr_multiplier = self._get_env("TSL_ACTIVATION_ATR_MULTIPLIER", "1.0", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.1"), max_val=Decimal("20.0"))
        self.trailing_stop_percent = self._get_env("TRAILING_STOP_PERCENT", "0.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.001"), max_val=Decimal("10.0"))
        self.take_profit_atr_multipliers = self._get_env("TAKE_PROFIT_ATR_MULTIPLIERS", "2.0,4.0", Fore.YELLOW, cast_type=lambda x: [Decimal(v) for v in x.split(',')])
        self.max_position_percentage = self._get_env("MAX_POSITION_PERCENTAGE", "0.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.1"), max_val=Decimal("1.0"))
        self.sl_trigger_by = self._get_env("SL_TRIGGER_BY", "LastPrice", Fore.YELLOW, allowed_values=["LastPrice", "MarkPrice", "IndexPrice"])
        self.tsl_trigger_by = self._get_env("TSL_TRIGGER_BY", "LastPrice", Fore.YELLOW, allowed_values=["LastPrice", "MarkPrice", "IndexPrice"])
        self.leverage = self._get_env("LEVERAGE", "10", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("1"), max_val=Decimal("100"))
        self.dynamic_leverage = self._get_env("DYNAMIC_LEVERAGE", "False", Fore.YELLOW, cast_type=bool)
        self.multi_timeframe_interval = self._get_env("MULTI_TIMEFRAME_INTERVAL", "5m", Fore.YELLOW)
        
        # Indicator Periods
        self.trend_ema_period = self._get_env("TREND_EMA_PERIOD", "12", Fore.YELLOW, cast_type=int, min_val=5, max_val=500)
        self.fast_ema_period = self._get_env("FAST_EMA_PERIOD", "9", Fore.YELLOW, cast_type=int, min_val=1, max_val=200)
        self.slow_ema_period = self._get_env("SLOW_EMA_PERIOD", "21", Fore.YELLOW, cast_type=int, min_val=2, max_val=500)
        self.stoch_period = self._get_env("STOCH_PERIOD", "7", Fore.YELLOW, cast_type=int, min_val=1, max_val=100)
        self.stoch_smooth_k = self._get_env("STOCH_SMOOTH_K", "3", Fore.YELLOW, cast_type=int, min_val=1, max_val=10)
        self.stoch_smooth_d = self._get_env("STOCH_SMOOTH_D", "3", Fore.YELLOW, cast_type=int, min_val=1, max_val=10)
        self.atr_period = self._get_env("ATR_PERIOD", "5", Fore.YELLOW, cast_type=int, min_val=1, max_val=100)
        
        # Signal Thresholds
        self.stoch_oversold_threshold = self._get_env("STOCH_OVERSOLD_THRESHOLD", "30", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("45"))
        self.stoch_overbought_threshold = self._get_env("STOCH_OVERBOUGHT_THRESHOLD", "70", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("55"), max_val=Decimal("100"))
        self.trend_filter_buffer_percent = self._get_env("TREND_FILTER_BUFFER_PERCENT", "0.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("5"))
        self.atr_move_filter_multiplier = self._get_env("ATR_MOVE_FILTER_MULTIPLIER", "0.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("5"))
        self.volatility_threshold = self._get_env("VOLATILITY_THRESHOLD", "3.0", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("1.0"), max_val=Decimal("10.0"))
        
        # Operational Parameters
        self.position_qty_epsilon = Decimal("1E-12")
        self.api_key = self._get_env("BYBIT_API_KEY", None, Fore.RED)
        self.api_secret = self._get_env("BYBIT_API_SECRET", None, Fore.RED)
        self.telegram_token = self._get_env("TELEGRAM_TOKEN", None, Fore.YELLOW)
        self.telegram_chat_id = self._get_env("TELEGRAM_CHAT_ID", None, Fore.YELLOW)
        self.ohlcv_limit = self._get_env("OHLCV_LIMIT", "200", Fore.YELLOW, cast_type=int, min_val=50, max_val=1000)
        self.loop_sleep_seconds = self._get_env("LOOP_SLEEP_SECONDS", "15", Fore.YELLOW, cast_type=int, min_val=5)
        self.max_fetch_retries = self._get_env("MAX_FETCH_RETRIES", "3", Fore.YELLOW, cast_type=int, min_val=1, max_val=10)
        self.trade_only_with_trend = self._get_env("TRADE_ONLY_WITH_TREND", "True", Fore.YELLOW, cast_type=bool)
        self.journal_file_path = self._get_env("JOURNAL_FILE_PATH", "bybit_trading_journal.csv", Fore.YELLOW)
        self.enable_journaling = self._get_env("ENABLE_JOURNALING", "True", Fore.YELLOW, cast_type=bool)
        self.backtest_enabled = self._get_env("BACKTEST_ENABLED", "False", Fore.YELLOW, cast_type=bool)
        self.backtest_data_path = self._get_env("BACKTEST_DATA_PATH", "backtest_data.csv", Fore.YELLOW)
        self.health_check_port = self._get_env("HEALTH_CHECK_PORT", "8080", Fore.YELLOW, cast_type=int, min_val=1024, max_val=65535)

        if not self.api_key or not self.api_secret:
            logger.critical(f"{Fore.RED}Missing BYBIT_API_KEY or BYBIT_API_SECRET.")
            sys.exit(1)

        # Validations
        if self.fast_ema_period >= self.slow_ema_period:
            logger.critical(f"{Fore.RED}FAST_EMA_PERIOD ({self.fast_ema_period}) must be less than SLOW_EMA_PERIOD ({self.slow_ema_period}).")
            sys.exit(1)
        if self.stoch_oversold_threshold >= self.stoch_overbought_threshold:
            logger.critical(f"{Fore.RED}STOCH_OVERSOLD_THRESHOLD ({self.stoch_oversold_threshold}) must be less than STOCH_OVERBOUGHT_THRESHOLD ({self.stoch_overbought_threshold}).")
            sys.exit(1)

        logger.debug("Configuration loaded successfully.")

    def _load_config_file(self):
        """Loads configuration from a JSON file if provided."""
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                for key, value in config.items():
                    os.environ[key.upper()] = str(value)
                logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.error(f"{Fore.RED}Failed to load config file {self.config_file}: {e}")

    def _get_env(self, key: str, default: Any, color: str, cast_type: type = str,
                 min_val: Optional[Union[int, float, Decimal]] = None,
                 max_val: Optional[Union[int, float, Decimal]] = None,
                 allowed_values: Optional[List[str]] = None) -> Any:
        """Fetches and validates environment variables."""
        value_str = os.getenv(key)
        log_value = "****" if "SECRET" in key.upper() or "KEY" in key.upper() else value_str

        if value_str is None or value_str.strip() == "":
            value = default
            if default is not None:
                logger.warning(f"{color}Using default for {key}: {default}")
            value_str = str(default) if default is not None else None
        else:
            logger.info(f"{color}Fetched {key}: {log_value}")

        if value_str is None and default is None:
            logger.critical(f"{Fore.RED}Required configuration '{key}' missing.")
            sys.exit(1)

        try:
            if cast_type == bool:
                value_to_cast = value_str if value_str is not None else str(default)
                casted_value = value_to_cast.lower() in ['true', '1', 'yes', 'y', 'on']
            elif cast_type == Decimal:
                value_to_cast = value_str if value_str is not None else str(default)
                casted_value = Decimal(value_to_cast)
            elif cast_type == int:
                value_to_cast = value_str if value_str is not None else str(default)
                casted_value = int(float(value_to_cast))
            elif callable(cast_type):
                value_to_cast = value_str if value_str is not None else str(default)
                casted_value = cast_type(value_to_cast)
            else:
                value_to_cast = value_str if value_str is not None else str(default)
                casted_value = str(value_to_cast)

            if allowed_values and str(casted_value).lower() not in [str(v).lower() for v in allowed_values]:
                logger.error(f"{Fore.RED}Invalid value '{casted_value}' for {key}. Allowed: {allowed_values}. Using default: {default}")
                return default

            if isinstance(casted_value, (Decimal, int, float)) and not isinstance(casted_value, list):
                compare_value = Decimal(str(casted_value)) if not isinstance(casted_value, Decimal) else casted_value
                min_val_comp = Decimal(str(min_val)) if min_val is not None else None
                max_val_comp = Decimal(str(max_val)) if max_val is not None else None
                if min_val_comp is not None and compare_value < min_val_comp:
                    logger.error(f"{Fore.RED}{key} value {casted_value} below minimum {min_val}. Using default: {default}")
                    return default
                if max_val_comp is not None and compare_value > max_val_comp:
                    logger.error(f"{Fore.RED}{key} value {casted_value} above maximum {max_val}. Using default: {default}")
                    return default

            return casted_value

        except Exception as e:
            logger.critical(f"{Fore.RED}Failed to cast {key} ('{value_str}'): {e}. Using default: {default}")
            return default

# --- Exchange Interface ---
class ExchangeInterface(ABC):
    """Abstract interface for exchange interactions."""
    @abstractmethod
    async def initialize(self):
        pass

    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> List:
        pass

    @abstractmethod
    async def fetch_balance(self) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        pass

    @abstractmethod
    async def create_market_order(self, symbol: str, side: str, amount: float, params: Dict) -> Dict:
        pass

    @abstractmethod
    async def set_trading_stop(self, symbol: str, params: Dict) -> Dict:
        pass

    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> Dict:
        pass

    @abstractmethod
    def price_to_precision(self, symbol: str, price: float) -> str:
        pass

    @abstractmethod
    def amount_to_precision(self, symbol: str, amount: float, rounding_mode: str) -> str:
        pass

    @abstractmethod
    async def close(self):
        pass

class BybitExchange(ExchangeInterface):
    """Bybit exchange implementation."""
    def __init__(self, config: TradingConfig):
        self.config = config
        self.exchange = ccxt.bybit({
            'apiKey': config.api_key,
            'secret': config.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': config.market_type, 'adjustForTimeDifference': True}
        })
        self.rate_limit_remaining = 120  # Bybit default per minute
        self.rate_limit_reset = time.time() + 60

    async def initialize(self):
        await self.exchange.load_markets()
        if self.config.symbol not in self.exchange.markets:
            logger.critical(f"{Fore.RED}Symbol {self.config.symbol} not found.")
            sys.exit(1)
        await self.set_leverage()

    async def set_leverage(self):
        if self.config.dynamic_leverage:
            atr = await self._fetch_atr()
            leverage = min(self.config.leverage, Decimal("10") + atr * Decimal("2"))
        else:
            leverage = self.config.leverage
        try:
            await self.exchange.private_post_position_set_leverage({
                'category': self.config.market_type,
                'symbol': self.config.symbol,
                'buyLeverage': str(leverage),
                'sellLeverage': str(leverage)
            })
            logger.info(f"Set leverage to {leverage}x")
        except Exception as e:
            logger.error(f"{Fore.RED}Failed to set leverage: {e}")

    async def _fetch_atr(self) -> Decimal:
        df = await self.fetch_ohlcv(self.config.symbol, self.config.interval, self.config.ohlcv_limit)
        df = pd.DataFrame(df, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        high = df['high'].apply(Decimal)
        low = df['low'].apply(Decimal)
        close = df['close'].shift(1).apply(Decimal)
        tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
        atr = tr.ewm(span=self.config.atr_period, adjust=False).mean()
        return Decimal(str(atr.iloc[-1]))

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> List:
        if self.rate_limit_remaining <= 0:
            wait = self.rate_limit_reset - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
            self.rate_limit_remaining = 120
            self.rate_limit_reset = time.time() + 60
        self.rate_limit_remaining -= 1
        try:
            return await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except Exception:
            logger.warning("Falling back to Binance for OHLCV")
            binance = ccxt.binance()
            return await binance.fetch_ohlcv(symbol.replace(':USDT', '/USDT'), timeframe, limit=limit)

    async def fetch_balance(self) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        self.rate_limit_remaining -= 1
        balance = await self.exchange.fetch_balance(params={'type': 'swap'})
        quote = self.exchange.market(self.config.symbol)['quote']
        free = Decimal(str(balance.get(quote, {}).get('free', '0')))
        total = Decimal(str(balance.get(quote, {}).get('total', '0')))
        return free, total

    async def create_market_order(self, symbol: str, side: str, amount: float, params: Dict) -> Dict:
        self.rate_limit_remaining -= 1
        params['category'] = self.config.market_type
        return await self.exchange.create_market_order(symbol, side, amount, params=params)

    async def set_trading_stop(self, symbol: str, params: Dict) -> Dict:
        self.rate_limit_remaining -= 1
        params['category'] = self.config.market_type
        return await self.exchange.private_post_position_set_trading_stop(params=params)

    async def fetch_ticker(self, symbol: str) -> Dict:
        self.rate_limit_remaining -= 1
        return await self.exchange.fetch_ticker(symbol)

    def price_to_precision(self, symbol: str, price: float) -> str:
        return self.exchange.price_to_precision(symbol, price)

    def amount_to_precision(self, symbol: str, amount: float, rounding_mode: str) -> str:
        return self.exchange.amount_to_precision(symbol, amount, rounding_mode=ccxt.TRUNCATE)

    async def close(self):
        await self.exchange.close()

# --- Indicator Registry ---
class IndicatorRegistry:
    """Manages technical indicators."""
    def __init__(self):
        self.indicators = {}

    def register(self, name: str, func: callable):
        self.indicators[name] = func

    def calculate(self, name: str, df: pd.DataFrame, **kwargs) -> Any:
        if name not in self.indicators:
            raise ValueError(f"Indicator {name} not registered")
        return self.indicators[name](df, **kwargs)

# --- Indicator Calculations ---
class IndicatorCalculator:
    """Handles technical indicator calculations with Decimal precision."""
    def __init__(self, config: TradingConfig):
        self.config = config
        self.registry = IndicatorRegistry()
        self.registry.register('ema', self._calculate_ema)
        self.registry.register('atr', self._calculate_atr)
        self.registry.register('stochastic', self._calculate_stochastic)

    def calculate_indicators(self, df: pd.DataFrame) -> Optional[Dict[str, Union[Decimal, bool, int]]]:
        """Calculates registered indicators."""
        try:
            if df.empty or len(df) < max(self.config.slow_ema_period, self.config.stoch_period + self.config.stoch_smooth_d):
                logger.error(f"{Fore.RED}Insufficient data for indicator calculation.")
                return None
            if (df[['open', 'high', 'low', 'close']].isna().any().any() or
                (df['timestamp'].diff().iloc[1:] > pd.Timedelta(minutes=2 * pd.Timedelta(self.config.interval).seconds / 60)).any()):
                logger.error(f"{Fore.RED}Invalid or stale market data.")
                return None

            df = df.copy()
            df['close'] = df['close'].apply(Decimal)
            df['high'] = df['high'].apply(Decimal)
            df['low'] = df['low'].apply(Decimal)

            indicators = {}
            indicators['trend_ema'] = self.registry.calculate('ema', df, period=self.config.trend_ema_period)
            indicators['fast_ema'] = self.registry.calculate('ema', df, period=self.config.fast_ema_period)
            indicators['slow_ema'] = self.registry.calculate('ema', df, period=self.config.slow_ema_period)
            indicators['atr'] = self.registry.calculate('atr', df, period=self.config.atr_period)
            stoch_k, stoch_d = self.registry.calculate('stochastic', df,
                                                     k_period=self.config.stoch_period,
                                                     smooth_k=self.config.stoch_smooth_k,
                                                     smooth_d=self.config.stoch_smooth_d)
            indicators['stoch_k'] = stoch_k
            indicators['stoch_d'] = stoch_d
            indicators['stoch_kd_bullish'] = stoch_k > stoch_d and stoch_k[-2] <= stoch_d[-2]
            indicators['stoch_kd_bearish'] = stoch_k < stoch_d and stoch_k[-2] >= stoch_d[-2]
            indicators['atr_period'] = self.config.atr_period

            return indicators
        except Exception as e:
            logger.error(f"{Fore.RED}Indicator calculation error: {e}", exc_info=True)
            return None

    def _calculate_ema(self, df: pd.DataFrame, period: int) -> Decimal:
        ema = df['close'].ewm(span=period, adjust=False).mean()
        return Decimal(str(ema.iloc[-1]))

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> Decimal:
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        return Decimal(str(atr.iloc[-1]))

    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int, smooth_k: int, smooth_d: int) -> Tuple[Decimal, Decimal]:
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        k = 100 * (df['close'] - low_min) / (high_max - low_min)
        k_smooth = k.rolling(window=smooth_k).mean()
        d = k_smooth.rolling(window=smooth_d).mean()
        return Decimal(str(k_smooth.iloc[-1])), Decimal(str(d.iloc[-1]))

# --- Strategy Interface ---
class StrategyBase(ABC):
    """Abstract base class for trading strategies."""
    def __init__(self, config: TradingConfig):
        self.config = config

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, indicators: Dict, equity: Decimal, mtf_indicators: Optional[Dict] = None) -> Dict:
        pass

class EMAStochasticStrategy(StrategyBase):
    """EMA and Stochastic-based trading strategy."""
    def generate_signals(self, df: pd.DataFrame, indicators: Dict, equity: Decimal, mtf_indicators: Optional[Dict] = None) -> Dict:
        try:
            long_signal, short_signal = False, False
            signal_reason = "No Signal"

            current_price = Decimal(str(df['close'].iloc[-1]))
            fast_ema = indicators['fast_ema']
            slow_ema = indicators['slow_ema']
            trend_ema = indicators['trend_ema']
            k = indicators['stoch_k']
            d = indicators['stoch_d']
            atr = indicators['atr']
            stoch_kd_bullish = indicators['stoch_kd_bullish']
            stoch_kd_bearish = indicators['stoch_kd_bearish']

            ema_bullish_cross = fast_ema > slow_ema
            ema_bearish_cross = fast_ema < slow_ema
            stoch_long_condition = k < self.config.stoch_oversold_threshold or stoch_kd_bullish
            stoch_short_condition = k > self.config.stoch_overbought_threshold or stoch_kd_bearish

            buffer_points = trend_ema * (self.config.trend_filter_buffer_percent / Decimal('100'))
            price_above_trend = current_price > trend_ema - buffer_points
            price_below_trend = current_price < trend_ema + buffer_points

            is_significant_move = True
            atr_move_reason = "Move Filter OFF"
            if len(df) >= 2 and not atr.is_nan():
                prev_close = Decimal(str(df['close'].iloc[-2]))
                price_move = (current_price - prev_close).copy_abs()
                atr_threshold = atr * self.config.atr_move_filter_multiplier
                is_significant_move = price_move >= atr_threshold
                atr_move_reason = f"Move Filter {'Passed' if is_significant_move else 'Failed'} (|ΔP|={price_move.normalize()} vs {atr_threshold.normalize()})"

            volatility_ok = atr < current_price * self.config.volatility_threshold / Decimal('100')
            volatility_reason = f"Volatility {'OK' if volatility_ok else 'High'} (ATR={atr.normalize()})"

            mtf_condition = True
            mtf_reason = "MTF Disabled"
            if mtf_indicators:
                mtf_trend_ema = mtf_indicators['trend_ema']
                mtf_condition = current_price > mtf_trend_ema if ema_bullish_cross else current_price < mtf_trend_ema
                mtf_reason = f"MTF {'Passed' if mtf_condition else 'Failed'} (Price={current_price.normalize()} vs MTF EMA={mtf_trend_ema.normalize()})"

            if (ema_bullish_cross and stoch_long_condition and is_significant_move and
                volatility_ok and mtf_condition):
                if self.config.trade_only_with_trend and not price_above_trend:
                    signal_reason = f"Long Blocked: Price({current_price.normalize()}) !> Trend({trend_ema.normalize()})"
                else:
                    long_signal = True
                    signal_reason = f"Long Signal: EMA Bullish | Stoch K:{k.normalize()} | {atr_move_reason} | {volatility_reason} | {mtf_reason}"
            elif (ema_bearish_cross and stoch_short_condition and is_significant_move and
                  volatility_ok and mtf_condition):
                if self.config.trade_only_with_trend and not price_below_trend:
                    signal_reason = f"Short Blocked: Price({current_price.normalize()}) !< Trend({trend_ema.normalize()})"
                else:
                    short_signal = True
                    signal_reason = f"Short Signal: EMA Bearish | Stoch K:{k.normalize()} | {atr_move_reason} | {volatility_reason} | {mtf_reason}"
            else:
                signal_reason = f"No Signal: EMA {'Bullish' if ema_bullish_cross else 'Bearish'} | Stoch K:{k.normalize()} | {volatility_reason} | {mtf_reason}"

            logger.info(Fore.GREEN if long_signal or short_signal else Fore.YELLOW, signal_reason)
            return {"long": long_signal, "short": short_signal, "reason": signal_reason}
        except Exception as e:
            logger.error(f"{Fore.RED}Signal generation error: {e}", exc_info=True)
            return {"long": False, "short": False, "reason": f"Error: {e}"}

# --- Trade Execution ---
class TradeExecutor:
    """Handles trade execution and position management."""
    def __init__(self, config: TradingConfig, exchange: ExchangeInterface, journal: JournalManager):
        self.config = config
        self.exchange = exchange
        self.journal = journal
        self.order_tracker = {
            "long": {"sl_id": None, "tsl_id": None, "tp_ids": []},
            "short": {"sl_id": None, "tsl_id": None, "tp_ids": []}
        }
        self.performance = {'trades': 0, 'wins': 0, 'pnl': Decimal('0'), 'equity_history': []}

    async def fetch_with_retries(self, fetch_function, *args, **kwargs) -> Any:
        """Fetches data with retries and backoff."""
        for attempt in range(self.config.max_fetch_retries + 1):
            if self.config.shutdown_requested:
                logger.warning("Shutdown requested, aborting fetch.")
                return None
            try:
                kwargs['params'] = kwargs.get('params', {})
                result = await fetch_function(*args, **kwargs)
                return result
            except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                wait_time = (2 ** attempt) + (random.random() * 0.1)
                logger.warning(f"{Fore.YELLOW}Network error (Attempt {attempt + 1}): {e}. Retrying in {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)
            except ccxt.AuthenticationError as e:
                logger.critical(f"{Fore.RED}Authentication error: {e}. Halting.")
                sys.exit(1)
            except Exception as e:
                logger.error(f"{Fore.RED}Unexpected error: {e}", exc_info=True)
                return None
        logger.error(f"{Fore.RED}Fetch failed after {self.config.max_fetch_retries} attempts.")
        return None

    async def place_risked_market_order(self, symbol: str, side: str, risk_percentage: Decimal, atr: Decimal) -> bool:
        """Places a market order with risk-based sizing."""
        try:
            balance, equity = await self.exchange.fetch_balance()
            if equity is None or equity <= Decimal("0"):
                logger.error(f"{Fore.RED}Invalid equity: {equity}")
                return False

            position_size = equity * risk_percentage
            position_size = min(position_size, equity * self.config.max_position_percentage)
            current_price = Decimal(str((await self.exchange.fetch_ticker(symbol))['last']))
            qty = position_size / (atr * self.config.sl_atr_multiplier)
            qty_str = self.format_amount(symbol, qty, ROUND_DOWN)

            order = await self.fetch_with_retries(
                self.exchange.create_market_order,
                symbol=symbol,
                side=side,
                amount=float(qty_str),
                params={'reduceOnly': False, 'positionIdx': 0}
            )

            if order and order.get('id'):
                logger.trade(f"{Fore.GREEN}Placed {side.upper()} order: Qty {qty_str}")
                await self.set_stop_loss(symbol, side, qty, current_price, atr)
                await self.set_take_profits(symbol, side, qty, current_price, atr)
                self.journal.log_trade({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'side': side,
                    'price': current_price,
                    'quantity': qty_str,
                    'reason': 'Entry'
                })
                self.performance['trades'] += 1
                self.performance['equity_history'].append(equity)
                await self.send_telegram(f"📈 {side.upper()} Order: {symbol} @ {current_price} Qty: {qty_str}")
                return True
            return False
        except Exception as e:
            logger.error(f"{Fore.RED}Order placement error: {e}", exc_info=True)
            return False

    async def set_stop_loss(self, symbol: str, side: str, qty: Decimal, entry_price: Decimal, atr: Decimal):
        """Sets stop-loss for a position."""
        try:
            sl_price = entry_price - (atr * self.config.sl_atr_multiplier) if side == "buy" else entry_price + (atr * self.config.sl_atr_multiplier)
            sl_price_str = self.format_price(symbol, sl_price)
            params = {
                'symbol': symbol,
                'stopLoss': sl_price_str,
                'triggerBy': self.config.sl_trigger_by,
                'positionIdx': 0
            }
            response = await self.fetch_with_retries(self.exchange.set_trading_stop, symbol=symbol, params=params)
            if response and response.get('retCode') == 0:
                self.order_tracker[side.lower()]["sl_id"] = f"POS_SL_{side.upper()}"
                logger.trade(f"{Fore.YELLOW}Set SL for {side.upper()}: {sl_price_str}")
        except Exception as e:
            logger.error(f"{Fore.RED}Failed to set SL: {e}", exc_info=True)

    async def set_take_profits(self, symbol: str, side: str, qty: Decimal, entry_price: Decimal, atr: Decimal):
        """Sets multiple take-profit levels."""
        try:
            for i, multiplier in enumerate(self.config.take_profit_atr_multipliers):
                if multiplier <= 0:
                    continue
                tp_price = entry_price + (atr * multiplier) if side == "buy" else entry_price - (atr * multiplier)
                tp_price_str = self.format_price(symbol, tp_price)
                partial_qty = qty / Decimal(str(len(self.config.take_profit_atr_multipliers)))
                params = {
                    'symbol': symbol,
                    'takeProfit': tp_price_str,
                    'triggerBy': self.config.sl_trigger_by,
                    'positionIdx': 0,
                    'qty': self.format_amount(symbol, partial_qty, ROUND_DOWN)
                }
                response = await self.fetch_with_retries(self.exchange.set_trading_stop, symbol=symbol, params=params)
                if response and response.get('retCode') == 0:
                    self.order_tracker[side.lower()]["tp_ids"].append(f"POS_TP_{side.upper()}_{i}")
                    logger.trade(f"{Fore.GREEN}Set TP {i+1} for {side.upper()}: {tp_price_str} Qty: {partial_qty}")
        except Exception as e:
            logger.error(f"{Fore.RED}Failed to set TP: {e}", exc_info=True)

    def format_price(self, symbol: str, price: Union[Decimal, str, float, int]) -> str:
        try:
            if isinstance(price, Decimal) and price.is_nan():
                return "NaN"
            return self.exchange.price_to_precision(symbol, float(price))
        except Exception:
            return str(Decimal(str(price)).quantize(Decimal("1E-8"), rounding=ROUND_HALF_EVEN).normalize())

    def format_amount(self, symbol: str, amount: Union[Decimal, str, float, int], rounding_mode=ROUND_DOWN) -> str:
        try:
            if isinstance(amount, Decimal) and amount.is_nan():
                return "NaN"
            return self.exchange.amount_to_precision(symbol, float(amount), rounding_mode=ccxt.TRUNCATE)
        except Exception:
            return str(Decimal(str(amount)).quantize(Decimal("1E-8"), rounding=rounding_mode).normalize())

    async def get_balance(self) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        return await self.exchange.fetch_balance()

    async def send_telegram(self, message: str):
        """Sends a Telegram notification."""
        if self.config.telegram_token and self.config.telegram_chat_id:
            try:
                url = f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage"
                payload = {
                    'chat_id': self.config.telegram_chat_id,
                    'text': message,
                    'parse_mode': 'Markdown'
                }
                requests.post(url, json=payload, timeout=5)
                logger.debug("Sent Telegram notification")
            except Exception as e:
                logger.error(f"{Fore.RED}Failed to send Telegram notification: {e}")

# --- Backtesting ---
class Backtester:
    """Simulates trading strategy on historical data."""
    def __init__(self, config: TradingConfig, indicator_calculator: IndicatorCalculator, strategy: StrategyBase):
        self.config = config
        self.indicator_calculator = indicator_calculator
        self.strategy = strategy

    def run_backtest(self, data_path: str) -> Dict:
        """Runs backtest on historical data."""
        try:
            df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
            df = df[['open', 'high', 'low', 'close', 'volume']].dropna()

            equity = Decimal('10000')
            position = {'long': Decimal('0'), 'short': Decimal('0')}
            trades = []
            equity_history = [equity]
            for i in range(max(self.config.slow_ema_period, self.config.stoch_period + self.config.stoch_smooth_d), len(df)):
                window = df.iloc[:i+1]
                indicators = self.indicator_calculator.calculate_indicators(window)
                if indicators:
                    signals = self.strategy.generate_signals(window, indicators, equity)
                    current_price = Decimal(str(window['close'].iloc[-1]))
                    if signals['long'] and position['long'] == 0 and position['short'] == 0:
                        qty = (equity * self.config.risk_percentage) / (indicators['atr'] * self.config.sl_atr_multiplier)
                        position['long'] = qty
                        trades.append({'time': window.index[-1], 'side': 'long', 'price': current_price, 'qty': qty})
                    elif signals['short'] and position['short'] == 0 and position['long'] == 0:
                        qty = (equity * self.config.risk_percentage) / (indicators['atr'] * self.config.sl_atr_multiplier)
                        position['short'] = qty
                        trades.append({'time': window.index[-1], 'side': 'short', 'price': current_price, 'qty': qty})

            # Calculate metrics
            returns = pd.Series(equity_history).pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
            max_drawdown = max(1 - pd.Series(equity_history) / pd.Series(equity_history).cummax())
            win_rate = sum(1 for t in trades if t.get('pnl', 0) > 0) / len(trades) if trades else 0

            # Plot equity curve
            plt.figure(figsize=(10, 6))
            plt.plot(equity_history, label='Equity')
            plt.title('Backtest Equity Curve')
            plt.xlabel('Trade')
            plt.ylabel('Equity')
            plt.legend()
            plt.savefig('backtest_equity.png')

            return {
                'equity': equity,
                'trades': trades,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate
            }
        except Exception as e:
            logger.error(f"{Fore.RED}Backtest error: {e}", exc_info=True)
            return {'equity': Decimal('0'), 'trades': [], 'sharpe_ratio': 0, 'max_drawdown': 0, 'win_rate': 0}

# --- Dashboard ---
class Dashboard:
    """Interactive CLI dashboard using rich."""
    def __init__(self, config: TradingConfig):
        self.config = config
        self.console = Console()
        self.live = None

    def start(self):
        self.live = Live(self._generate_table(), console=self.console, refresh_per_second=1)
        self.live.start()

    def stop(self):
        if self.live:
            self.live.stop()

    def update(self, data: Dict):
        self.live.update(self._generate_table(data))

    def _generate_table(self, data: Dict = None) -> Table:
        table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
        table.add_column("Metric")
        table.add_column("Value")
        if data:
            table.add_row("Symbol", data.get('symbol', self.config.symbol))
            table.add_row("Price", str(data.get('price', '-')))
            table.add_row("Equity", str(data.get('equity', '-')))
            table.add_row("Fast EMA", str(data.get('fast_ema', '-')))
            table.add_row("Slow EMA", str(data.get('slow_ema', '-')))
            table.add_row("Stoch K", str(data.get('stoch_k', '-')))
            table.add_row("Signal", data.get('signal', 'No Signal'))
        return table

# --- Health Check ---
async def run_health_check(port: int, status: Dict):
    """Simple HTTP server for health checks."""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(status).encode())

    server = HTTPServer(('0.0.0.0', port), HealthHandler)
    await asyncio.get_event_loop().run_in_executor(None, server.serve_forever)

# --- Main Bot ---
class TradingBot:
    """Main trading bot orchestrating all components."""
    def __init__(self, config_file: Optional[str] = None):
        self.config = TradingConfig(config_file)
        self.exchange = BybitExchange(self.config)
        self.journal = JournalManager(self.config.journal_file_path, self.config.enable_journaling)
        self.indicator_calculator = IndicatorCalculator(self.config)
        self.strategy = EMAStochasticStrategy(self.config)
        self.trade_executor = TradeExecutor(self.config, self.exchange, self.journal)
        self.backtester = Backtester(self.config, self.indicator_calculator, self.strategy)
        self.dashboard = Dashboard(self.config)
        self.shutdown_requested = False
        self.status = {'running': True, 'last_cycle': None}

    async def initialize(self):
        """Initializes exchange connection."""
        try:
            await self.exchange.initialize()
            logger.info(f"{Fore.GREEN}Connected to Bybit {self.config.market_type} markets.")
            self.dashboard.start()
            asyncio.create_task(run_health_check(self.config.health_check_port, self.status))
        except Exception as e:
            logger.critical(f"{Fore.RED}Initialization error: {e}", exc_info=True)
            sys.exit(1)

    async def trading_cycle(self, cycle_count: int):
        """Executes one trading cycle."""
        logger.info(f"{Fore.MAGENTA}--- Cycle {cycle_count} ---")
        df = await self.exchange.fetch_ohlcv(self.config.symbol, self.config.interval, self.config.ohlcv_limit)
        df = pd.DataFrame(df, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        mtf_df = await self.exchange.fetch_ohlcv(self.config.symbol, self.config.multi_timeframe_interval, self.config.ohlcv_limit)
        mtf_df = pd.DataFrame(mtf_df, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        mtf_df['timestamp'] = pd.to_datetime(mtf_df['timestamp'], unit='ms')
        mtf_df.set_index('timestamp', inplace=True)

        current_price = Decimal(str(df['close'].iloc[-1]))
        indicators = self.indicator_calculator.calculate_indicators(df)
        mtf_indicators = self.indicator_calculator.calculate_indicators(mtf_df)
        _, equity = await self.trade_executor.get_balance()
        signals = self.strategy.generate_signals(df, indicators, equity, mtf_indicators) if indicators else {"long": False, "short": False, "reason": "No indicators"}

        self.status['last_cycle'] = datetime.now().isoformat()
        self.dashboard.update({
            'symbol': self.config.symbol,
            'price': current_price,
            'equity': equity,
            'fast_ema': indicators.get('fast_ema') if indicators else '-',
            'slow_ema': indicators.get('slow_ema') if indicators else '-',
            'stoch_k': indicators.get('stoch_k') if indicators else '-',
            'signal': signals.get('reason', 'No Signal')
        })

        if signals['long'] and equity:
            await self.trade_executor.place_risked_market_order(self.config.symbol, "buy", self.config.risk_percentage, indicators['atr'])
        elif signals['short'] and equity:
            await self.trade_executor.place_risked_market_order(self.config.symbol, "sell", self.config.risk_percentage, indicators['atr'])

    async def main_loop(self):
        """Main trading loop."""
        await self.initialize()
        if self.config.backtest_enabled:
            result = self.backtester.run_backtest(self.config.backtest_data_path)
            logger.info(f"Backtest Result: Equity={result['equity']}, Trades={len(result['trades'])}, "
                        f"Sharpe={result['sharpe_ratio']:.2f}, Drawdown={result['max_drawdown']:.2%}, "
                        f"Win Rate={result['win_rate']:.2%}")
            return

        cycle_count = 0
        while not self.shutdown_requested:
            cycle_count += 1
            await self.trading_cycle(cycle_count)
            await asyncio.sleep(self.config.loop_sleep_seconds)

    async def shutdown(self, signum=None, frame=None):
        """Graceful shutdown."""
        self.shutdown_requested = True
        logger.warning(f"{Fore.YELLOW}Shutting down (Signal: {signal.Signals(signum).name if signum else 'Manual'})...")
        self.status['running'] = False
        self.dashboard.stop()
        await self.exchange.close()
        with open('state.json', 'w') as f:
            json.dump({'last_equity': str(self.trade_executor.performance['equity_history'][-1])}, f)
        logger.info("Shutdown complete.")

# --- Entry Point ---
async def main():
    print(f"{Fore.GREEN}{Style.BRIGHT}Pyrmethus Trading Bot v4.0")
    bot = TradingBot(config_file='config.json')
    signal.signal(signal.SIGINT, lambda s, f: asyncio.create_task(bot.shutdown(s, f)))
    signal.signal(signal.SIGTERM, lambda s, f: asyncio.create_task(bot.shutdown(s, f)))
    try:
        await bot.main_loop()
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    import random
    asyncio.run(main())