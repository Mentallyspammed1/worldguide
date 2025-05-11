#!/usr/bin/env python
"""
Bybit Trading Enchanced (v2.0)

A robust trading bot framework for Bybit V5 API, optimized for Termux.
Supports real-time trading with WebSocket streams, indicator calculations,
and SMS alerts. Integrates with indicators.py for technical analysis.

Key Features:
- Bybit V5 API interaction via pybit (HTTP and WebSocket).
- Indicator calculations using indicators.py (EVT, ATR, etc.).
- Incremental indicator updates for WebSocket data.
- Termux-optimized logging and SMS alerts.
- Configuration via .env with setup CLI.
- Support for testnet and live trading.

Usage:
- Install dependencies: pip install ccxt pybit pydantic pydantic-settings colorama websocket-client pandas pandas_ta
- Setup: python bybit_trading_enchanced.py --setup
- Run strategy: python ehlers_volumetric_strategy.py
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict, Optional, List
from decimal import Decimal
from pathlib import Path
import pandas as pd
from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from colorama import Fore, Style, init
from pybit.unified_trading import HTTP, WebSocket
import ccxt

try:
    from indicators import calculate_all_indicators, update_indicators_incrementally
except ImportError:
    print("Error: indicators.py not found in the same directory.", file=sys.stderr)
    sys.exit(1)

# --- Initialize Colorama ---
init(autoreset=True)


# --- Configuration Models ---
class ApiConfig(BaseModel):
    api_key: SecretStr
    api_secret: SecretStr
    symbol: str = "BTCUSDT"
    testnet_mode: bool = True


class StrategyConfig(BaseModel):
    timeframe: str = "5m"
    risk_per_trade: Decimal = Decimal("0.01")
    loop_delay_seconds: float = 5.0
    indicator_settings: Dict[str, Any] = {
        "evt_length": 7,
        "evt_multiplier": 2.5,
        "atr_period": 14,
    }


class SmsConfig(BaseModel):
    enable_sms_alerts: bool = False
    termux_phone_number: Optional[str] = None
    priority_levels: Dict[str, int] = {"critical": 1, "normal": 2, "low": 3}


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
    )
    api_config: ApiConfig
    strategy_config: StrategyConfig
    sms_config: SmsConfig
    log_directory: str = "logs"
    log_level: str = "INFO"


# --- Logger Setup ---
def setup_logger(log_dir: str, log_level: str) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "trading_bot.log"
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


# --- Load Configuration ---
def load_config() -> AppConfig:
    try:
        config = AppConfig()
        return config
    except Exception as e:
        print(
            f"{Fore.RED}Error loading configuration: {e}{Style.RESET_ALL}",
            file=sys.stderr,
        )
        sys.exit(1)


# --- Bybit Helper Class ---
class BybitHelper:
    """Helper class for Bybit V5 API interactions and trading operations."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = setup_logger(config.log_directory, config.log_level)
        self.api_key = config.api_config.api_key.get_secret_value()
        self.api_secret = config.api_config.api_secret.get_secret_value()
        self.symbol = config.api_config.symbol
        self.testnet = config.api_config.testnet_mode
        self.ohlcv_cache: Dict[str, pd.DataFrame] = {}

        # Initialize pybit HTTP session
        try:
            self.session = HTTP(
                testnet=self.testnet, api_key=self.api_key, api_secret=self.api_secret
            )
            self.logger.info("Initialized pybit HTTP session")
        except Exception as e:
            self.logger.critical(
                f"Failed to initialize pybit session: {e}", exc_info=True
            )
            sys.exit(1)

        # Initialize ccxt exchange
        try:
            self.exchange = ccxt.bybit(
                {
                    "apiKey": self.api_key,
                    "secret": self.api_secret,
                    "enableRateLimit": True,
                }
            )
            if self.testnet:
                self.exchange.set_sandbox_mode(True)
            self.exchange.load_markets()
            self.logger.info("Initialized ccxt exchange")
        except Exception as e:
            self.logger.error(f"Failed to initialize ccxt exchange: {e}", exc_info=True)
            sys.exit(1)

        # Initialize WebSocket
        self.ws = None
        self._initialize_websocket()

    def _initialize_websocket(self):
        """Initialize WebSocket connection."""
        try:
            self.ws = WebSocket(
                api_key=self.api_key, api_secret=self.api_secret, testnet=self.testnet
            )
            self.logger.info("Initialized WebSocket")
        except Exception as e:
            self.logger.error(f"Failed to initialize WebSocket: {e}", exc_info=True)

    def diagnose_connection(self) -> bool:
        """Check API connectivity and market availability."""
        try:
            server_time = self.session.get_server_time()
            if server_time.get("retCode") != 0:
                self.logger.error(
                    f"Server time fetch failed: {server_time.get('retMsg')}"
                )
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
        """Send SMS alert via Termux API."""
        if (
            not self.config.sms_config.enable_sms_alerts
            or not self.config.sms_config.termux_phone_number
        ):
            return False
        try:
            priority_level = self.config.sms_config.priority_levels.get(priority, 3)
            if priority_level <= 2:  # Only send for normal or critical
                cmd = f'termux-sms-send -n {self.config.sms_config.termux_phone_number} "{message}"'
                result = os.system(cmd)
                if result == 0:
                    self.logger.info(f"Sent SMS alert: {message}")
                    return True
                else:
                    self.logger.error(f"Failed to send SMS alert: {message}")
                    return False
        except Exception as e:
            self.logger.error(f"Error sending SMS alert: {e}", exc_info=True)
            return False
        return False

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

    def fetch_ticker(self) -> Dict[str, Any]:
        """Fetch latest ticker data."""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker
        except Exception as e:
            self.logger.error(f"Error fetching ticker: {e}", exc_info=True)
            return {}

    def fetch_ohlcv(self, timeframe: str, limit: int = 200) -> List[List[float]]:
        """Fetch OHLCV data."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV: {e}", exc_info=True)
            return []

    def calculate_indicators(
        self,
        df: pd.DataFrame,
        incremental: bool = False,
        prev_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Calculate indicators using indicators.py."""
        try:
            config = {
                "indicator_settings": {
                    "min_data_periods": 50,
                    "atr_period": self.config.strategy_config.indicator_settings.get(
                        "atr_period", 14
                    ),
                    "evt_length": self.config.strategy_config.indicator_settings.get(
                        "evt_length", 7
                    ),
                    "evt_multiplier": self.config.strategy_config.indicator_settings.get(
                        "evt_multiplier", 2.5
                    ),
                },
                "analysis_flags": {"use_atr": True, "use_evt": True},
                "strategy_params": {
                    "ehlers_volumetric": {
                        "evt_length": self.config.strategy_config.indicator_settings.get(
                            "evt_length", 7
                        ),
                        "evt_multiplier": self.config.strategy_config.indicator_settings.get(
                            "evt_multiplier", 2.5
                        ),
                    }
                },
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
            if timeframe not in [self.config.strategy_config.timeframe, "15m"]:
                return None
            kline = {
                "timestamp": pd.to_datetime(kline_data["start"], unit="ms"),
                "open": float(kline_data["open"]),
                "high": float(kline_data["high"]),
                "low": float(kline_data["low"]),
                "close": float(kline_data["close"]),
                "volume": float(kline_data["volume"]),
            }
            df_new = pd.DataFrame([kline])
            prev_df = self.ohlcv_cache.get(timeframe)
            df_updated = self.calculate_indicators(
                df_new, incremental=True, prev_df=prev_df
            )
            if prev_df is not None:
                self.ohlcv_cache[timeframe] = (
                    pd.concat([prev_df, df_updated])
                    .drop_duplicates(subset="timestamp")
                    .tail(200)
                )
            else:
                self.ohlcv_cache[timeframe] = df_updated
            return df_updated
        except Exception as e:
            self.logger.error(f"Error processing kline update: {e}", exc_info=True)
            return None

    def get_position_info(self) -> Dict[str, Any]:
        """Fetch current position info."""
        try:
            response = self.session.get_positions(category="linear", symbol=self.symbol)
            if response.get("retCode") == 0:
                return response
            self.logger.error(
                f"Failed to fetch position info: {response.get('retMsg')}"
            )
            return {}
        except Exception as e:
            self.logger.error(f"Error fetching position info: {e}", exc_info=True)
            return {}

    def get_open_orders(self) -> Dict[str, Any]:
        """Fetch open orders."""
        try:
            response = self.session.get_open_orders(
                category="linear", symbol=self.symbol
            )
            if response.get("retCode") == 0:
                return response
            self.logger.error(f"Failed to fetch open orders: {response.get('retMsg')}")
            return {}
        except Exception as e:
            self.logger.error(f"Error fetching open orders: {e}", exc_info=True)
            return {}

    def place_market_order(self, side: str, qty: float) -> Dict[str, Any]:
        """Place a market order."""
        try:
            response = self.session.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Market",
                qty=str(qty),
                reduceOnly=False,
            )
            if response.get("retCode") == 0:
                self.logger.info(f"Placed market order: {side} {qty} {self.symbol}")
            else:
                self.logger.error(
                    f"Failed to place market order: {response.get('retMsg')}"
                )
            return response
        except Exception as e:
            self.logger.error(f"Error placing market order: {e}", exc_info=True)
            return {"retCode": -1, "retMsg": str(e)}

    def subscribe_to_stream(
        self, topics: List[str], callback: callable, channel_type: str = "public"
    ):
        """Subscribe to WebSocket stream."""
        try:
            if channel_type == "public":
                self.ws.public_stream(topics=topics, callback=callback)
            elif channel_type == "private":
                self.ws.private_stream(topics=topics, callback=callback)
            self.logger.info(f"Subscribed to {topics} ({channel_type})")
        except Exception as e:
            self.logger.error(f"Error subscribing to stream: {e}", exc_info=True)

    def get_strategy_context(self) -> Dict[str, Any]:
        """Fetch strategy context including indicators."""
        context = {
            "balance": self.fetch_balance(),
            "ticker": self.fetch_ticker(),
            "position": self.get_position_info(),
            "open_orders": self.get_open_orders(),
            "ohlcv": self.fetch_ohlcv(self.config.strategy_config.timeframe),
        }
        if context.get("ohlcv"):
            df = pd.DataFrame(
                context["ohlcv"],
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = self.calculate_indicators(df)
            context["indicators"] = df
        return context

    def stop(self):
        """Stop WebSocket and cleanup."""
        try:
            if self.ws:
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
    env_content.append("# Bybit API Configuration")
    api_key = input("Enter Bybit API Key: ").strip()
    api_secret = input("Enter Bybit API Secret: ").strip()
    symbol = (
        input("Enter trading symbol (e.g., BTCUSDT) [BTCUSDT]: ").strip() or "BTCUSDT"
    )
    testnet = input("Use testnet? (y/n) [y]: ").strip().lower() in ["", "y"]
    env_content.append(f"BOT_API_CONFIG__API_KEY={api_key}")
    env_content.append(f"BOT_API_CONFIG__API_SECRET={api_secret}")
    env_content.append(f"BOT_API_CONFIG__SYMBOL={symbol}")
    env_content.append(f"BOT_API_CONFIG__TESTNET_MODE={str(testnet).lower()}")

    env_content.append("\n# Strategy Configuration")
    timeframe = input("Enter timeframe (e.g., 5m, 15m) [5m]: ").strip() or "5m"
    risk = input("Enter risk per trade (e.g., 0.01 for 1%) [0.01]: ").strip() or "0.01"
    env_content.append(f"BOT_STRATEGY_CONFIG__TIMEFRAME={timeframe}")
    env_content.append(f"BOT_STRATEGY_CONFIG__RISK_PER_TRADE={risk}")
    env_content.append("BOT_STRATEGY_CONFIG__LOOP_DELAY_SECONDS=5.0")
    env_content.append("BOT_STRATEGY_CONFIG__INDICATOR_SETTINGS__EVT_LENGTH=7")
    env_content.append("BOT_STRATEGY_CONFIG__INDICATOR_SETTINGS__EVT_MULTIPLIER=2.5")
    env_content.append("BOT_STRATEGY_CONFIG__INDICATOR_SETTINGS__ATR_PERIOD=14")

    env_content.append("\n# SMS Configuration")
    enable_sms = input("Enable SMS alerts? (y/n) [n]: ").strip().lower() == "y"
    phone = (
        input("Enter phone number for SMS alerts (or leave blank): ").strip()
        if enable_sms
        else ""
    )
    env_content.append(f"BOT_SMS_CONFIG__ENABLE_SMS_ALERTS={str(enable_sms).lower()}")
    env_content.append(f"BOT_SMS_CONFIG__TERMUX_PHONE_NUMBER={phone}")
    env_content.append("BOT_SMS_CONFIG__PRIORITY_LEVELS__CRITICAL=1")
    env_content.append("BOT_SMS_CONFIG__PRIORITY_LEVELS__NORMAL=2")
    env_content.append("BOT_SMS_CONFIG__PRIORITY_LEVELS__LOW=3")

    env_content.append("\n# Logging Configuration")
    env_content.append("LOG_DIRECTORY=logs")
    env_content.append("LOG_LEVEL=INFO")

    try:
        with open(".env", "w") as f:
            f.write("\n".join(env_content))
        print(f"{Fore.GREEN}.env file created successfully.{Style.RESET_ALL}")
    except Exception as e:
        print(
            f"{Fore.RED}Error writing .env file: {e}{Style.RESET_ALL}", file=sys.stderr
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Bybit Trading Bot")
    parser.add_argument("--setup", action="store_true", help="Setup .env configuration")
    parser.add_argument(
        "--view-positions", action="store_true", help="View open positions"
    )
    parser.add_argument("--view-orders", action="store_true", help="View open orders")
    args = parser.parse_args()

    config = load_config()
    helper = BybitHelper(config)

    if args.setup:
        setup_env()
        return

    if args.view_positions:
        positions = helper.get_position_info()
        print(f"Positions: {positions}")
        return

    if args.view_orders:
        orders = helper.get_open_orders()
        print(f"Open Orders: {orders}")
        return

    helper.logger.info("BybitHelper ready. Run strategy separately.")


if __name__ == "__main__":
    main()
