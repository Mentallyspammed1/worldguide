#!/usr/bin/env python
"""
Bybit Trading Enchanced (v3.0)

A comprehensive Bybit API helper using pybit and ccxt for trading operations.
Supports HTTP and WebSocket interactions, market data fetching, order management,
and real-time indicator calculations. Optimized for Termux with SMS alerts and
robust logging. Integrates with indicators.py for technical analysis.

Key Features:
- pybit for HTTP (orders, positions, balances) and WebSocket (kline, order updates).
- ccxt for market data (OHLCV, ticker, order book) and precision handling.
- Configuration via .env with AppConfig (API keys, symbol, indicator settings).
- Real-time indicator updates using indicators.py (EVT, ATR, etc.).
- Termux SMS alerts for critical events.
- Retry logic and error handling for robust API calls.
- CLI for setup, diagnostics, and viewing positions/orders.
- WebSocket reconnection for continuous data streams.

Usage:
- Install dependencies: pip install ccxt pybit pydantic pydantic-settings colorama websocket-client pandas pandas_ta
- Setup: python bybit_trading_enchanced.py --setup
- Run strategy: python ehlers_volumetric_strategy.py
- View positions: python bybit_trading_enchanced.py --view-positions
- View orders: python bybit_trading_enchanced.py --view-orders
"""

import argparse
import functools
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Union

import ccxt
import pandas as pd
import websocket
from colorama import Back, Fore, Style, init as colorama_init
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict
from pybit.unified_trading import HTTP, WebSocket

try:
    from indicators import calculate_all_indicators, update_indicators_incrementally
except ImportError:
    print(f"{Fore.RED}Error: indicators.py not found in the same directory.{Style.RESET_ALL}", file=sys.stderr)
    sys.exit(1)

# --- Initialize Colorama ---
colorama_init(autoreset=True)
COLORAMA_AVAILABLE = True

# --- Custom Logging Setup (Adapted from neon_logger.py) ---
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)

if not hasattr(logging.Logger, "success"):
    logging.Logger.success = log_success  # type: ignore[attr-defined]

LOG_LEVEL_COLORS: Dict[int, str] = {
    logging.DEBUG: Fore.CYAN + Style.DIM,
    logging.INFO: Fore.BLUE + Style.BRIGHT,
    SUCCESS_LEVEL: Fore.MAGENTA + Style.BRIGHT,
    logging.WARNING: Fore.YELLOW + Style.BRIGHT,
    logging.ERROR: Fore.RED + Style.BRIGHT,
    logging.CRITICAL: Back.RED + Fore.WHITE + Style.BRIGHT,
}

class ColoredConsoleFormatter(logging.Formatter):
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_colors = COLORAMA_AVAILABLE and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        color = LOG_LEVEL_COLORS.get(record.levelno, Fore.WHITE)
        record.levelname = f"{color}{original_levelname:<8}{Style.RESET_ALL}" if self.use_colors else f"{original_levelname:<8}"
        formatted_message = super().format(record)
        record.levelname = original_levelname
        return formatted_message

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s [%(module)s:%(funcName)s:%(lineno)d] - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

console_formatter = ColoredConsoleFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
file_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

def setup_logger(
    logger_name: str = "BybitTrading",
    log_file: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_rotation_bytes: int = 5 * 1024 * 1024,
    log_backup_count: int = 5,
) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=log_rotation_bytes, backupCount=log_backup_count, encoding="utf-8"
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Suppress third-party logs
    for lib in ["ccxt", "pybit", "urllib3", "websocket"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

    return logger

# --- Configuration Models (Adapted from config_models.py) ---
class ApiConfig(BaseModel):
    api_key: str = Field(..., description="Bybit API Key")
    api_secret: str = Field(..., description="Bybit API Secret")
    symbol: str = Field("BTCUSDT", description="Primary trading symbol")
    testnet_mode: bool = Field(True, description="Use testnet")
    retry_count: PositiveInt = Field(3, description="API retry attempts")
    retry_delay_seconds: PositiveFloat = Field(1.0, description="Delay between retries")

class StrategyConfig(BaseModel):
    timeframe: str = Field("5m", description="Kline timeframe")
    risk_per_trade: Decimal = Field(Decimal("0.01"), description="Risk per trade (fraction)")
    loop_delay_seconds: PositiveFloat = Field(5.0, description="Strategy loop delay")
    indicator_settings: Dict[str, Any] = Field(
        default_factory=lambda: {
            "min_data_periods": 50,
            "evt_length": 7,
            "evt_multiplier": 2.5,
            "atr_period": 14,
            "sma_short_period": 10,
            "sma_long_period": 50,
            "ema_short_period": 12,
            "ema_long_period": 26,
            "supertrend_period": 10,
            "supertrend_multiplier": 3.0,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
        },
        description="Indicator parameters"
    )
    analysis_flags: Dict[str, bool] = Field(
        default_factory=lambda: {
            "use_sma": True,
            "use_ema": True,
            "use_supertrend": True,
            "use_macd": True,
            "use_evt": True,
            "use_atr": True,
        },
        description="Enable/disable indicators"
    )

class SmsConfig(BaseModel):
    enable_sms_alerts: bool = Field(False, description="Enable SMS alerts")
    termux_phone_number: Optional[str] = Field(None, description="Phone number for SMS alerts")
    sms_cooldown_seconds: PositiveInt = Field(60, description="Cooldown for non-critical SMS")

class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore"
    )
    api_config: ApiConfig
    strategy_config: StrategyConfig
    sms_config: SmsConfig
    log_directory: str = Field("logs", description="Log directory")
    log_level: str = Field("INFO", description="Log level")

# --- Utility Functions (Adapted from bybit_utils.py) ---
def safe_decimal_conversion(value: Any, default: Optional[Decimal] = None) -> Optional[Decimal]:
    if value is None or value == "":
        return default
    try:
        return Decimal(str(value))
    except (ValueError, InvalidOperation):
        return default

def format_price(exchange: ccxt.Exchange, symbol: str, price: Any) -> str:
    try:
        return exchange.price_to_precision(symbol, price)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Error formatting price for {symbol}: {e}")
        price_dec = safe_decimal_conversion(price)
        return f"{price_dec:.8f}" if price_dec is not None else "Invalid"

def format_amount(exchange: ccxt.Exchange, symbol: str, amount: Any) -> str:
    try:
        return exchange.amount_to_precision(symbol, amount)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Error formatting amount for {symbol}: {e}")
        amount_dec = safe_decimal_conversion(amount)
        return f"{amount_dec:.8f}" if amount_dec is not None else "Invalid"

def retry_api_call(max_retries: int = 3, initial_delay: float = 1.0):
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt <= max_retries:
                try:
                    result = func(*args, **kwargs)
                    if isinstance(result, dict) and result.get("retCode", 0) != 0:
                        ret_code = result.get("retCode")
                        ret_msg = result.get("retMsg", "Unknown error")
                        if ret_code in [10001, 10002, 130035]:  # Rate limit, timeout, etc.
                            raise ccxt.NetworkError(f"Bybit error {ret_code}: {ret_msg}")
                        return result
                    return result
                except (ccxt.NetworkError, ccxt.RateLimitExceeded, websocket.WebSocketTimeoutException) as e:
                    attempt += 1
                    if attempt > max_retries:
                        logger = logging.getLogger(__name__)
                        logger.error(f"Max retries reached for {func.__name__}: {e}")
                        return {"retCode": -1, "retMsg": str(e)}
                    delay = initial_delay * (2 ** (attempt - 1))
                    logger = logging.getLogger(__name__)
                    logger.warning(f"{func.__name__} failed: {e}. Retrying after {delay:.2f}s")
                    time.sleep(delay)
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.critical(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
                    return {"retCode": -1, "retMsg": str(e)}
            return {"retCode": -1, "retMsg": "Max retries exceeded"}
        return wrapper
    return decorator

# --- Bybit Helper Class ---
class BybitHelper:
    """Comprehensive helper for Bybit API interactions using pybit and ccxt."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = setup_logger(
            logger_name="BybitTrading",
            log_file=os.path.join(config.log_directory, "trading_bot.log"),
            console_level=getattr(logging, config.log_level.upper(), logging.INFO),
            file_level=logging.DEBUG
        )
        self.api_key = config.api_config.api_key
        self.api_secret = config.api_config.api_secret
        self.symbol = config.api_config.symbol
        self.testnet = config.api_config.testnet_mode
        self.ohlcv_cache: Dict[str, pd.DataFrame] = {}
        self.ws_connected = False
        self.last_sms_time: Dict[str, float] = {"normal": 0, "critical": 0}

        # Initialize pybit HTTP session
        try:
            self.session = HTTP(
                testnet=self.testnet,
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            self.logger.info("Initialized pybit HTTP session")
        except Exception as e:
            self.logger.critical(f"Failed to initialize pybit session: {e}", exc_info=True)
            sys.exit(1)

        # Initialize ccxt exchange
        try:
            self.exchange = ccxt.bybit({
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True
            })
            if self.testnet:
                self.exchange.set_sandbox_mode(True)
            self.exchange.load_markets()
            self.logger.info("Initialized ccxt exchange")
        except Exception as e:
            self.logger.critical(f"Failed to initialize ccxt exchange: {e}", exc_info=True)
            sys.exit(1)

        # Initialize WebSocket
        self.ws = None
        self._initialize_websocket()

    def _initialize_websocket(self):
        """Initialize WebSocket connection with reconnection logic."""
        try:
            self.ws = WebSocket(
                testnet=self.testnet,
                api_key=self.api_key,
                api_secret=self.api_secret,
                ping_interval=20,
                ping_timeout=10
            )
            self.ws_connected = True
            self.logger.info("Initialized WebSocket")
        except Exception as e:
            self.logger.error(f"Failed to initialize WebSocket: {e}", exc_info=True)
            self.ws_connected = False

    @retry_api_call(max_retries=3, initial_delay=1.0)
    def diagnose_connection(self) -> bool:
        """Check API connectivity and market availability."""
        try:
            server_time = self.session.get_server_time()
            if server_time.get("retCode") != 0:
                self.logger.error(f"Server time fetch failed: {server_time.get('retMsg')}")
                return False
            if self.symbol not in self.exchange.markets:
                self.logger.error(f"Symbol {self.symbol} not available")
                return False
            self.logger.info("Connection diagnostics passed")
            return True
        except Exception as e:
            self.logger.error(f"Connection diagnostics failed: {e}", exc_info=True)
            return False

    def send_sms_alert(self, message: str, priority: str = "normal") -> bool:
        """Send SMS alert via Termux API with cooldown."""
        if not self.config.sms_config.enable_sms_alerts or not self.config.sms_config.termux_phone_number:
            return False
        current_time = time.time()
        cooldown = self.config.sms_config.sms_cooldown_seconds
        last_sent = self.last_sms_time.get(priority, 0)
        if current_time - last_sent < cooldown:
            self.logger.debug(f"SMS alert skipped due to cooldown: {message}")
            return False
        try:
            cmd = f"termux-sms-send -n {self.config.sms_config.termux_phone_number} \"{message}\""
            result = os.system(cmd)
            if result == 0:
                self.last_sms_time[priority] = current_time
                self.logger.info(f"Sent SMS alert: {message}")
                return True
            else:
                self.logger.error(f"Failed to send SMS alert: {message}")
                return False
        except Exception as e:
            self.logger.error(f"Error sending SMS alert: {e}", exc_info=True)
            return False

    @retry_api_call()
    def fetch_balance(self) -> Dict[str, Any]:
        """Fetch account balance."""
        try:
            response = self.session.get_wallet_balance(accountType="UNIFIED")
            if response.get("retCode") == 0:
                return response["result"]
            self.logger.error(f"Failed to fetch balance: {response.get('retMsg')}")
            return {}
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}", exc_info=True)
            return {}

    @retry_api_call()
    def fetch_ticker(self) -> Dict[str, Any]:
        """Fetch latest ticker data."""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker
        except Exception as e:
            self.logger.error(f"Error fetching ticker: {e}", exc_info=True)
            return {}

    @retry_api_call()
    def fetch_ohlcv(self, timeframe: str, limit: int = 200) -> List[List[float]]:
        """Fetch OHLCV data."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV: {e}", exc_info=True)
            return []

    def calculate_indicators(self, df: pd.DataFrame, incremental: bool = False, prev_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculate indicators using indicators.py."""
        try:
            config = {
                "indicator_settings": self.config.strategy_config.indicator_settings,
                "analysis_flags": self.config.strategy_config.analysis_flags,
                "strategy_params": {
                    "ehlers_volumetric": {
                        "evt_length": self.config.strategy_config.indicator_settings.get("evt_length", 7),
                        "evt_multiplier": self.config.strategy_config.indicator_settings.get("evt_multiplier", 2.5)
                    }
                }
            }
            if incremental and prev_df is not None:
                return update_indicators_incrementally(df, config, prev_df)
            return calculate_all_indicators(df, config)
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}", exc_info=True)
            return df

    def process_kline_update(self, data: Dict) -> Optional[pd.DataFrame]:
        """Process WebSocket kline update and compute incremental indicators."""
        try:
            kline_data = data.get("data", [{}])[0]
            timeframe = kline_data.get("interval")
            if timeframe != self.config.strategy_config.timeframe:
                return None
            kline = {
                "timestamp": pd.to_datetime(kline_data["start"], unit="ms"),
                "open": float(kline_data["open"]),
                "high": float(kline_data["high"]),
                "low": float(kline_data["low"]),
                "close": float(kline_data["close"]),
                "volume": float(kline_data["volume"])
            }
            df_new = pd.DataFrame([kline])
            prev_df = self.ohlcv_cache.get(timeframe)
            df_updated = self.calculate_indicators(df_new, incremental=True, prev_df=prev_df)
            if prev_df is not None:
                self.ohlcv_cache[timeframe] = pd.concat([prev_df, df_updated]).drop_duplicates(subset="timestamp").tail(200)
            else:
                self.ohlcv_cache[timeframe] = df_updated
            return df_updated
        except Exception as e:
            self.logger.error(f"Error processing kline update: {e}", exc_info=True)
            return None

    @retry_api_call()
    def get_position_info(self) -> Dict[str, Any]:
        """Fetch current position info."""
        try:
            response = self.session.get_positions(category="linear", symbol=self.symbol)
            if response.get("retCode") == 0:
                return response["result"]
            self.logger.error(f"Failed to fetch position info: {response.get('retMsg')}")
            return {}
        except Exception as e:
            self.logger.error(f"Error fetching position info: {e}", exc_info=True)
            return {}

    @retry_api_call()
    def get_open_orders(self) -> Dict[str, Any]:
        """Fetch open orders."""
        try:
            response = self.session.get_open_orders(category="linear", symbol=self.symbol)
            if response.get("retCode") == 0:
                return response["result"]
            self.logger.error(f"Failed to fetch open orders: {response.get('retMsg')}")
            return {}
        except Exception as e:
            self.logger.error(f"Error fetching open orders: {e}", exc_info=True)
            return {}

    @retry_api_call()
    def place_market_order(self, side: str, qty: float, reduce_only: bool = False) -> Dict[str, Any]:
        """Place a market order."""
        try:
            qty_str = format_amount(self.exchange, self.symbol, qty)
            response = self.session.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Market",
                qty=qty_str,
                reduceOnly=reduce_only
            )
            if response.get("retCode") == 0:
                self.logger.info(f"Placed market order: {side} {qty_str} {self.symbol}")
                self.send_sms_alert(f"Order placed: {side} {qty_str} {self.symbol}", priority="normal")
            else:
                self.logger.error(f"Failed to place market order: {response.get('retMsg')}")
                self.send_sms_alert(f"Order failed: {side} {qty_str} {self.symbol}", priority="critical")
            return response
        except Exception as e:
            self.logger.error(f"Error placing market order: {e}", exc_info=True)
            return {"retCode": -1, "retMsg": str(e)}

    @retry_api_call()
    def place_limit_order(self, side: str, qty: float, price: float, reduce_only: bool = False) -> Dict[str, Any]:
        """Place a limit order."""
        try:
            qty_str = format_amount(self.exchange, self.symbol, qty)
            price_str = format_price(self.exchange, self.symbol, price)
            response = self.session.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Limit",
                qty=qty_str,
                price=price_str,
                reduceOnly=reduce_only
            )
            if response.get("retCode") == 0:
                self.logger.info(f"Placed limit order: {side} {qty_str} @ {price_str} {self.symbol}")
            else:
                self.logger.error(f"Failed to place limit order: {response.get('retMsg')}")
            return response
        except Exception as e:
            self.logger.error(f"Error placing limit order: {e}", exc_info=True)
            return {"retCode": -1, "retMsg": str(e)}

    def subscribe_to_stream(self, topics: List[str], callback: Callable, channel_type: str = "public"):
        """Subscribe to WebSocket stream with reconnection handling."""
        def on_error(ws, error):
            self.logger.error(f"WebSocket error: {error}")
            self.ws_connected = False
            self._reconnect_websocket()

        def on_close(ws, close_status_code, close_msg):
            self.logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
            self.ws_connected = False
            self._reconnect_websocket()

        try:
            if channel_type == "public":
                self.ws.public_stream(topics=topics, callback=callback, on_error=on_error, on_close=on_close)
            elif channel_type == "private":
                self.ws.private_stream(topics=topics, callback=callback, on_error=on_error, on_close=on_close)
            self.logger.info(f"Subscribed to {topics} ({channel_type})")
        except Exception as e:
            self.logger.error(f"Error subscribing to stream: {e}", exc_info=True)
            self._reconnect_websocket()

    def _reconnect_websocket(self):
        """Attempt to reconnect WebSocket."""
        if not self.ws_connected:
            self.logger.info("Attempting WebSocket reconnection...")
            self._initialize_websocket()
            time.sleep(1)  # Wait for connection stabilization

    def get_strategy_context(self) -> Dict[str, Any]:
        """Fetch strategy context including market data and indicators."""
        context = {
            "balance": self.fetch_balance(),
            "ticker": self.fetch_ticker(),
            "position": self.get_position_info(),
            "open_orders": self.get_open_orders(),
            "ohlcv": self.fetch_ohlcv(self.config.strategy_config.timeframe)
        }
        if context.get("ohlcv"):
            df = pd.DataFrame(
                context["ohlcv"],
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = self.calculate_indicators(df)
            context["indicators"] = df
        return context

    def stop(self):
        """Clean up resources."""
        try:
            if self.ws and self.ws_connected:
                self.ws.close()
                self.logger.info("WebSocket closed")
            self.session.close()
            self.logger.info("HTTP session closed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}", exc_info=True)

# --- CLI Setup ---
def setup_env():
    """Generate .env file with user input."""
    print(f"{Fore.CYAN}Setting up .env configuration...{Style.RESET_ALL}")
    env_content = []

    # API Configuration
    env_content.append("# Bybit API Configuration")
    api_key = input("Enter Bybit API Key: ").strip()
    api_secret = input("Enter Bybit API Secret: ").strip()
    symbol = input("Enter trading symbol (e.g., BTCUSDT) [BTCUSDT]: ").strip() or "BTCUSDT"
    testnet = input("Use testnet? (y/n) [y]: ").strip().lower() in ["", "y"]
    env_content.append(f"API_CONFIG__API_KEY={api_key}")
    env_content.append(f"API_CONFIG__API_SECRET={api_secret}")
    env_content.append(f"API_CONFIG__SYMBOL={symbol}")
    env_content.append(f"API_CONFIG__TESTNET_MODE={str(testnet).lower()}")
    env_content.append("API_CONFIG__RETRY_COUNT=3")
    env_content.append("API_CONFIG__RETRY_DELAY_SECONDS=1.0")

    # Strategy Configuration
    env_content.append("\n# Strategy Configuration")
    timeframe = input("Enter timeframe (e.g., 5m, 15m) [5m]: ").strip() or "5m"
    risk = input("Enter risk per trade (e.g., 0.01 for 1%) [0.01]: ").strip() or "0.01"
    env_content.append(f"STRATEGY_CONFIG__TIMEFRAME={timeframe}")
    env_content.append(f"STRATEGY_CONFIG__RISK_PER_TRADE={risk}")
    env_content.append("STRATEGY_CONFIG__LOOP_DELAY_SECONDS=5.0")
    env_content.append("STRATEGY_CONFIG__INDICATOR_SETTINGS__EVT_LENGTH=7")
    env_content.append("STRATEGY_CONFIG__INDICATOR_SETTINGS__EVT_MULTIPLIER=2.5")
    env_content.append("STRATEGY_CONFIG__INDICATOR_SETTINGS__ATR_PERIOD=14")
    env_content.append("STRATEGY_CONFIG__INDICATOR_SETTINGS__SMA_SHORT_PERIOD=10")
    env_content.append("STRATEGY_CONFIG__INDICATOR_SETTINGS__SMA_LONG_PERIOD=50")
    env_content.append("STRATEGY_CONFIG__INDICATOR_SETTINGS__EMA_SHORT_PERIOD=12")
    env_content.append("STRATEGY_CONFIG__INDICATOR_SETTINGS__EMA_LONG_PERIOD=26")
    env_content.append("STRATEGY_CONFIG__INDICATOR_SETTINGS__SUPERTREND_PERIOD=10")
    env_content.append("STRATEGY_CONFIG__INDICATOR_SETTINGS__SUPERTREND_MULTIPLIER=3.0")
    env_content.append("STRATEGY_CONFIG__INDICATOR_SETTINGS__MACD_FAST=12")
    env_content.append("STRATEGY_CONFIG__INDICATOR_SETTINGS__MACD_SLOW=26")
    env_content.append("STRATEGY_CONFIG__INDICATOR_SETTINGS__MACD_SIGNAL=9")

    # SMS Configuration
    env_content.append("\n# SMS Configuration")
    enable_sms = input("Enable SMS alerts? (y/n) [n]: ").strip().lower() == "y"
    phone = input("Enter phone number for SMS alerts (or leave blank): ").strip() if enable_sms else ""
    env_content.append(f"SMS_CONFIG__ENABLE_SMS_ALERTS={str(enable_sms).lower()}")
    env_content.append(f"SMS_CONFIG__TERMUX_PHONE_NUMBER={phone}")
    env_content.append("SMS_CONFIG__SMS_COOLDOWN_SECONDS=60")

    # Logging Configuration
    env_content.append("\n# Logging Configuration")
    env_content.append("LOG_DIRECTORY=logs")
    env_content.append("LOG_LEVEL=INFO")

    try:
        with open(".env", "w") as f:
            f.write("\n".join(env_content))
        print(f"{Fore.GREEN}.env file created successfully.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error writing .env file: {e}{Style.RESET_ALL}", file=sys.stderr)
        sys.exit(1)

def load_config() -> AppConfig:
    """Load configuration from .env."""
    try:
        return AppConfig()
    except Exception as e:
        print(f"{Fore.RED}Error loading configuration: {e}{Style.RESET_ALL}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Bybit Trading Bot")
    parser.add_argument("--setup", action="store_true", help="Setup .env configuration")
    parser.add_argument("--view-positions", action="store_true", help="View open positions")
    parser.add_argument("--view-orders", action="store_true", help="View open orders")
    parser.add_argument("--diagnose", action="store_true", help="Run connection diagnostics")
    args = parser.parse_args()

    config = load_config()
    helper = BybitHelper(config)

    if args.setup:
        setup_env()
        return

    if args.diagnose:
        result = helper.diagnose_connection()
        print(f"{Fore.GREEN if result else Fore.RED}Diagnostics: {'Passed' if result else 'Failed'}{Style.RESET_ALL}")
        return

    if args.view_positions:
        positions = helper.get_position_info()
        print(json.dumps(positions, indent=2))
        return

    if args.view_orders:
        orders = helper.get_open_orders()
        print(json.dumps(orders, indent=2))
        return

    helper.logger.info("BybitHelper initialized. Run strategy separately (e.g., ehlers_volumetric_strategy.py).")

if __name__ == "__main__":
    main()