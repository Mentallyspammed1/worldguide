#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pyrmethus Trading Bot v4.0 - Enhanced Version

A cryptocurrency trading bot using CCXT, Pandas, and Rich for display.
This version removes Telegram notifications and implements Termux SMS notifications.
"""

import os
import time
import logging
import sys
import csv
import json
from datetime import datetime
from typing import Dict, Optional, Any, Tuple, Union, List
from decimal import Decimal, getcontext, ROUND_DOWN, ConversionSyntax
import asyncio
import signal
from abc import ABC, abstractmethod
from pathlib import Path
import matplotlib.pyplot as plt  # For backtesting plot

# Import enchantments with fallback guidance
try:
    import ccxt.async_support as ccxt
    from dotenv import load_dotenv
    import pandas as pd
    import numpy as np
    from tabulate import tabulate  # Still useful for potential future reports
    from colorama import init, Fore, Style
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich import box

    COMMON_PACKAGES = [
        "ccxt",
        "python-dotenv",
        "pandas",
        "numpy",
        "tabulate",
        "colorama",
        "rich",
        "matplotlib",
    ]
except ImportError as e:
    init(autoreset=True)  # Initialize colorama here for the error message
    missing_pkg = e.name
    print(f"{Fore.RED}{Style.BRIGHT}Missing required Python package: {missing_pkg}")
    print(f"{Fore.YELLOW}Please install all required packages using pip:")
    print(f"pip install {' '.join(COMMON_PACKAGES)}")
    sys.exit(1)

# Initialize Colorama
init(autoreset=True)

# Set Decimal precision
getcontext().prec = 50

# --- Logging Setup ---
logger = logging.getLogger(__name__)
TRADE_LEVEL_NUM = logging.INFO + 5
logging.addLevelName(TRADE_LEVEL_NUM, "TRADE")


def trade_log(self, message, *args, **kws):
    """Custom logging level for trade events."""
    if self.isEnabledFor(TRADE_LEVEL_NUM):
        # Yes, logger takes its '*args' as 'args'.
        self._log(TRADE_LEVEL_NUM, message, args, **kws)


logging.Logger.trade = trade_log

# Define a more visually distinct log formatter
log_formatter = logging.Formatter(
    f"{Fore.CYAN}%(asctime)s{Style.RESET_ALL} "
    f"{Style.BRIGHT}[%(levelname)-8s]{Style.RESET_ALL} "
    f"{Fore.MAGENTA}(%(filename)s:%(lineno)d){Style.RESET_ALL} "
    f"{Fore.WHITE}%(message)s{Style.RESET_ALL}"
)

log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
valid_log_levels = ["DEBUG", "INFO", "TRADE", "WARNING", "ERROR", "CRITICAL"]
if log_level_str not in valid_log_levels:
    print(f"{Fore.YELLOW}Invalid LOG_LEVEL '{log_level_str}'. Defaulting to INFO.")
    log_level_str = "INFO"

log_level = (
    TRADE_LEVEL_NUM if log_level_str == "TRADE" else getattr(logging, log_level_str)
)
logger.setLevel(log_level)

# Ensure handler is added only once
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

logger.propagate = False  # Prevent duplicate logs if root logger is configured


# --- Journaling ---
class JournalManager:
    """Manages trade journaling to a CSV file."""

    def __init__(self, file_path: str, enabled: bool):
        self.file_path = Path(file_path)  # Use pathlib
        self.enabled = enabled
        self.headers = [
            "timestamp",
            "symbol",
            "side",
            "price",
            "quantity",
            "pnl",
            "reason",
        ]
        if enabled:
            self._initialize_file()

    def _initialize_file(self):
        """Creates the journal file with headers if it doesn't exist."""
        try:
            self.file_path.parent.mkdir(
                parents=True, exist_ok=True
            )  # Ensure directory exists
            if not self.file_path.exists():
                with open(self.file_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.headers)
                logger.info(f"Initialized journal file: {self.file_path}")
        except OSError as e:
            logger.error(
                f"{Fore.RED}Failed to initialize journal file {self.file_path}: {e}"
            )
            self.enabled = False  # Disable journaling if file cannot be created

    def log_trade(self, trade: Dict):
        """Logs a single trade entry to the CSV file."""
        if not self.enabled:
            return
        try:
            # Ensure all headers are present in the trade dict, providing defaults
            row_data = [
                trade.get("timestamp", datetime.now().isoformat()),
                trade.get("symbol", "N/A"),
                trade.get("side", "N/A"),
                trade.get("price", "N/A"),
                trade.get("quantity", "N/A"),
                trade.get("pnl", "N/A"),
                trade.get("reason", "N/A"),
            ]
            with open(self.file_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
            logger.debug(f"Logged trade to {self.file_path}")
        except Exception as e:
            logger.error(f"{Fore.RED}Failed to log trade to {self.file_path}: {e}")


# --- Configuration ---
class TradingConfig:
    """
    Manages trading bot configuration from environment variables and/or a JSON file,
    with validation and default values.
    """

    def __init__(self, config_file: Optional[str] = None):
        logger.info(f"{Style.BRIGHT}--- Loading Configuration ---{Style.RESET_ALL}")
        load_dotenv()  # Load .env file if present
        self.config_file = config_file
        self.config_data = {}  # To store loaded config for summary
        self._load_config_file()  # Load JSON config first (can be overridden by env vars)

        # Fetch and validate all configuration parameters
        self._fetch_parameters()

        # Validate cross-parameter constraints
        self._validate_constraints()

        logger.debug("Configuration loading process completed.")
        self.print_config_summary()  # Print summary after loading

    def _load_config_file(self):
        """Loads configuration from a JSON file if provided."""
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    config_from_file = json.load(f)
                # Set environment variables from file keys BEFORE fetching them
                # This allows env vars to override file settings if both exist
                for key, value in config_from_file.items():
                    env_key = key.upper()
                    if (
                        env_key not in os.environ
                    ):  # Only set if not already set by environment
                        os.environ[env_key] = str(value)
                logger.info(f"Loaded base configuration from {self.config_file}")
            except json.JSONDecodeError as e:
                logger.error(
                    f"{Fore.RED}Error decoding JSON from {self.config_file}: {e}"
                )
            except Exception as e:
                logger.error(
                    f"{Fore.RED}Failed to load config file {self.config_file}: {e}"
                )

    def _fetch_parameters(self):
        """Fetches all parameters using the _get_env helper."""
        # Market & Connection
        self.symbol = self._get_env("SYMBOL", "BTC/USDT:USDT", Fore.YELLOW)
        self.market_type = self._get_env(
            "MARKET_TYPE",
            "linear",
            Fore.YELLOW,
            allowed_values=["linear", "inverse", "swap"],
        ).lower()
        self.interval = self._get_env(
            "INTERVAL", "1m", Fore.YELLOW
        )  # e.g., 1m, 5m, 1h, 1d
        self.api_key = self._get_env("BYBIT_API_KEY", None, Fore.RED)
        self.api_secret = self._get_env("BYBIT_API_SECRET", None, Fore.RED)

        # Risk Management
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
            min_val=Decimal("0.001"),
            max_val=Decimal("10.0"),
        )
        self.take_profit_atr_multipliers = self._get_env(
            "TAKE_PROFIT_ATR_MULTIPLIERS",
            "2.0,4.0",
            Fore.YELLOW,
            cast_type=lambda x: [Decimal(v.strip()) for v in x.split(",") if v.strip()],
        )
        self.max_position_percentage = self._get_env(
            "MAX_POSITION_PERCENTAGE",
            "0.5",
            Fore.YELLOW,
            cast_type=Decimal,
            min_val=Decimal("0.01"),
            max_val=Decimal("1.0"),
        )
        self.sl_trigger_by = self._get_env(
            "SL_TRIGGER_BY",
            "LastPrice",
            Fore.YELLOW,
            allowed_values=["LastPrice", "MarkPrice", "IndexPrice"],
        )
        self.tsl_trigger_by = self._get_env(
            "TSL_TRIGGER_BY",
            "LastPrice",
            Fore.YELLOW,
            allowed_values=["LastPrice", "MarkPrice", "IndexPrice"],
        )
        self.leverage = self._get_env(
            "LEVERAGE",
            "10",
            Fore.YELLOW,
            cast_type=Decimal,
            min_val=Decimal("1"),
            max_val=Decimal("100"),
        )
        self.dynamic_leverage = self._get_env(
            "DYNAMIC_LEVERAGE", False, Fore.YELLOW, cast_type=bool
        )  # Use False default

        # Multi-Timeframe & Indicators
        self.multi_timeframe_interval = self._get_env(
            "MULTI_TIMEFRAME_INTERVAL", "5m", Fore.YELLOW
        )
        self.trend_ema_period = self._get_env(
            "TREND_EMA_PERIOD", 12, Fore.YELLOW, cast_type=int, min_val=5, max_val=500
        )
        self.fast_ema_period = self._get_env(
            "FAST_EMA_PERIOD", 9, Fore.YELLOW, cast_type=int, min_val=1, max_val=200
        )
        self.slow_ema_period = self._get_env(
            "SLOW_EMA_PERIOD", 21, Fore.YELLOW, cast_type=int, min_val=2, max_val=500
        )
        self.stoch_period = self._get_env(
            "STOCH_PERIOD", 7, Fore.YELLOW, cast_type=int, min_val=1, max_val=100
        )
        self.stoch_smooth_k = self._get_env(
            "STOCH_SMOOTH_K", 3, Fore.YELLOW, cast_type=int, min_val=1, max_val=10
        )
        self.stoch_smooth_d = self._get_env(
            "STOCH_SMOOTH_D", 3, Fore.YELLOW, cast_type=int, min_val=1, max_val=10
        )
        self.atr_period = self._get_env(
            "ATR_PERIOD", 5, Fore.YELLOW, cast_type=int, min_val=1, max_val=100
        )

        # Signal Thresholds & Filters
        self.stoch_oversold_threshold = self._get_env(
            "STOCH_OVERSOLD_THRESHOLD",
            "30",
            Fore.YELLOW,
            cast_type=Decimal,
            min_val=Decimal("0"),
            max_val=Decimal("45"),
        )
        self.stoch_overbought_threshold = self._get_env(
            "STOCH_OVERBOUGHT_THRESHOLD",
            "70",
            Fore.YELLOW,
            cast_type=Decimal,
            min_val=Decimal("55"),
            max_val=Decimal("100"),
        )
        self.trend_filter_buffer_percent = self._get_env(
            "TREND_FILTER_BUFFER_PERCENT",
            "0.5",
            Fore.YELLOW,
            cast_type=Decimal,
            min_val=Decimal("0"),
            max_val=Decimal("5"),
        )
        self.atr_move_filter_multiplier = self._get_env(
            "ATR_MOVE_FILTER_MULTIPLIER",
            "0.5",
            Fore.YELLOW,
            cast_type=Decimal,
            min_val=Decimal("0"),
            max_val=Decimal("5"),
        )
        self.volatility_threshold = self._get_env(
            "VOLATILITY_THRESHOLD",
            "3.0",
            Fore.YELLOW,
            cast_type=Decimal,
            min_val=Decimal("0.1"),
            max_val=Decimal("20.0"),
        )  # Wider range?
        self.trade_only_with_trend = self._get_env(
            "TRADE_ONLY_WITH_TREND", True, Fore.YELLOW, cast_type=bool
        )

        # Operational Parameters
        self.ohlcv_limit = self._get_env(
            "OHLCV_LIMIT", 200, Fore.YELLOW, cast_type=int, min_val=50, max_val=1000
        )
        self.loop_sleep_seconds = self._get_env(
            "LOOP_SLEEP_SECONDS", 15, Fore.YELLOW, cast_type=int, min_val=5, max_val=300
        )
        self.max_fetch_retries = self._get_env(
            "MAX_FETCH_RETRIES", 3, Fore.YELLOW, cast_type=int, min_val=1, max_val=10
        )
        self.position_qty_epsilon = Decimal("1E-12")  # For float comparisons

        # Journaling
        self.journal_file_path = self._get_env(
            "JOURNAL_FILE_PATH", "bybit_trading_journal.csv", Fore.YELLOW
        )
        self.enable_journaling = self._get_env(
            "ENABLE_JOURNALING", True, Fore.YELLOW, cast_type=bool
        )

        # Backtesting
        self.backtest_enabled = self._get_env(
            "BACKTEST_ENABLED", False, Fore.YELLOW, cast_type=bool
        )
        self.backtest_data_path = self._get_env(
            "BACKTEST_DATA_PATH", "backtest_data.csv", Fore.YELLOW
        )

        # Health Check
        self.health_check_port = self._get_env(
            "HEALTH_CHECK_PORT",
            8080,
            Fore.YELLOW,
            cast_type=int,
            min_val=1024,
            max_val=65535,
        )

        # Termux SMS Notification (Replaces Telegram)
        self.termux_sms_recipient = self._get_env(
            "TERMUX_SMS_RECIPIENT", None, Fore.YELLOW
        )  # Phone number

        # Check for required API keys after trying to load them
        if not self.api_key or not self.api_secret:
            logger.critical(
                f"{Fore.RED}{Style.BRIGHT}Missing required BYBIT_API_KEY or BYBIT_API_SECRET in environment or config file. Exiting."
            )
            sys.exit(1)

    def _validate_constraints(self):
        """Validates constraints between different configuration parameters."""
        if self.fast_ema_period >= self.slow_ema_period:
            logger.critical(
                f"{Fore.RED}Configuration Error: FAST_EMA_PERIOD ({self.fast_ema_period}) must be less than SLOW_EMA_PERIOD ({self.slow_ema_period}). Exiting."
            )
            sys.exit(1)
        if self.stoch_oversold_threshold >= self.stoch_overbought_threshold:
            logger.critical(
                f"{Fore.RED}Configuration Error: STOCH_OVERSOLD_THRESHOLD ({self.stoch_oversold_threshold}) must be less than STOCH_OVERBOUGHT_THRESHOLD ({self.stoch_overbought_threshold}). Exiting."
            )
            sys.exit(1)
        # Add more cross-parameter validations if needed

    def _get_env(
        self,
        key: str,
        default: Any,
        color: str,
        cast_type: type = str,
        min_val: Optional[Union[int, float, Decimal]] = None,
        max_val: Optional[Union[int, float, Decimal]] = None,
        allowed_values: Optional[List[str]] = None,
    ) -> Any:
        """
        Fetches an environment variable, applies casting, validation, and defaults.
        Logs the process and stores the final value for summary.
        """
        value_str = os.getenv(key)
        source = "env"
        final_value = default

        if value_str is None or value_str.strip() == "":
            # Value not found in environment, use default
            value_str = str(default) if default is not None else None
            source = "default"
            log_message = f"{color}Using default for {key}: {default}"
            if (
                default is None
                and "SECRET" not in key.upper()
                and "KEY" not in key.upper()
                and "RECIPIENT" not in key.upper()
            ):  # Only warn if default is None for non-sensitive optional vars
                logger.warning(log_message)
            elif default is not None:
                logger.info(log_message)
            # No need to log for missing optional sensitive vars like SMS recipient
        else:
            # Value found in environment
            log_value = (
                "****" if "SECRET" in key.upper() or "KEY" in key.upper() else value_str
            )
            logger.info(f"{color}Fetched {key}: {log_value} (from {source})")

        if value_str is None:
            # This happens if default was None and env var was missing
            if key in [
                "BYBIT_API_KEY",
                "BYBIT_API_SECRET",
            ]:  # Check required keys explicitly
                # Error logged later in _fetch_parameters
                self.config_data[key] = ("Missing", Fore.RED)
                return None
            else:
                # Optional value is missing
                self.config_data[key] = (str(default), Fore.YELLOW)
                return default

        # Attempt casting and validation
        try:
            if cast_type == bool:
                # Explicit boolean handling first
                casted_value = value_str.lower() in ["true", "1", "yes", "y", "on"]
            elif cast_type == Decimal:
                casted_value = Decimal(value_str)
            elif cast_type == int:
                # Cast to float first to handle potential decimals in input (e.g., "10.0")
                casted_value = int(float(value_str))
            elif (
                callable(cast_type) and cast_type != str
            ):  # Handle custom casting functions (like list parser)
                casted_value = cast_type(value_str)
            else:  # Default to string
                casted_value = str(value_str)

            final_value = casted_value  # Assign successfully casted value

            # Validation: Allowed Values
            if allowed_values:
                # Case-insensitive comparison for allowed values
                if str(casted_value).lower() not in [
                    str(v).lower() for v in allowed_values
                ]:
                    logger.warning(
                        f"{Fore.YELLOW}Invalid value '{casted_value}' for {key}. Allowed: {allowed_values}. Using default: {default}"
                    )
                    final_value = default  # Revert to default
                    source += "+validation_fallback"

            # Validation: Min/Max (only for numeric types, excluding lists)
            if isinstance(final_value, (Decimal, int, float)) and not isinstance(
                final_value, list
            ):
                # Use Decimal for comparison to handle precision correctly
                compare_value = Decimal(str(final_value))
                valid = True
                if min_val is not None:
                    min_val_comp = Decimal(str(min_val))
                    if compare_value < min_val_comp:
                        logger.warning(
                            f"{Fore.YELLOW}{key} value {casted_value} is below minimum {min_val}. Using default: {default}"
                        )
                        final_value = default
                        source += "+min_fallback"
                        valid = False
                if (
                    valid and max_val is not None
                ):  # Only check max if min was okay or not present
                    max_val_comp = Decimal(str(max_val))
                    if compare_value > max_val_comp:
                        logger.warning(
                            f"{Fore.YELLOW}{key} value {casted_value} is above maximum {max_val}. Using default: {default}"
                        )
                        final_value = default
                        source += "+max_fallback"

        except (ValueError, TypeError, ConversionSyntax) as e:
            logger.error(
                f"{Fore.RED}Failed to cast {key} ('{value_str}') to {cast_type.__name__}: {e}. Using default: {default}"
            )
            final_value = default  # Revert to default on casting error
            source += "+cast_fallback"
        except (
            Exception
        ) as e:  # Catch any other unexpected errors during casting/validation
            logger.error(
                f"{Fore.RED}Unexpected error processing {key} ('{value_str}'): {e}. Using default: {default}"
            )
            final_value = default
            source += "+error_fallback"

        # Store final value and color for summary table
        final_color = (
            Fore.GREEN
            if source == "env"
            else color
            if "fallback" not in source
            else Fore.RED
        )
        display_value = (
            "****"
            if ("SECRET" in key.upper() or "KEY" in key.upper()) and final_value
            else str(final_value)
        )
        self.config_data[key] = (display_value, final_color)

        return final_value

    def print_config_summary(self):
        """Prints a summary of the loaded configuration using Rich."""
        console = Console()
        table = Table(
            title="Configuration Summary",
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED,
        )
        table.add_column("Parameter", style="dim", width=30)
        table.add_column("Value")
        table.add_column("Source/Status", width=20)

        for key, (value, color) in self.config_data.items():
            status = "Loaded"
            if color == Fore.YELLOW:
                status = "Default Used"
            elif color == Fore.RED:
                status = "Error/Missing"

            # Format list values nicely
            if isinstance(value, list):
                value_str = ", ".join(map(str, value))
            else:
                value_str = str(value)

            table.add_row(key, f"[{color}]{value_str}[/]", status)

        console.print(table)
        logger.info(f"{Style.BRIGHT}--- Configuration Loaded ---{Style.RESET_ALL}")


# --- Exchange Interface ---
class ExchangeInterface(ABC):
    """Abstract Base Class for exchange interactions."""

    @abstractmethod
    async def initialize(self):
        """Load markets and perform initial setup."""
        pass

    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> List:
        """Fetch historical OHLCV data."""
        pass

    @abstractmethod
    async def fetch_balance(self) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Fetch free and total balance for the quote currency."""
        pass

    @abstractmethod
    async def create_market_order(
        self, symbol: str, side: str, amount: float, params: Dict
    ) -> Optional[Dict]:
        """Place a market order."""
        pass

    @abstractmethod
    async def set_trading_stop(self, symbol: str, params: Dict) -> Optional[Dict]:
        """Set stop loss or take profit for a position."""
        pass

    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        """Fetch the latest ticker information."""
        pass

    @abstractmethod
    def price_to_precision(self, symbol: str, price: Union[float, Decimal]) -> str:
        """Format price to exchange precision."""
        pass

    @abstractmethod
    def amount_to_precision(
        self, symbol: str, amount: Union[float, Decimal], rounding_mode: Any
    ) -> str:
        """Format amount to exchange precision."""
        pass

    @abstractmethod
    async def fetch_position(self, symbol: str) -> Optional[Dict]:
        """Fetch current position details."""
        pass

    @abstractmethod
    async def close(self):
        """Close the exchange connection."""
        pass


class BybitExchange(ExchangeInterface):
    """Concrete implementation for the Bybit exchange."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.exchange = ccxt.bybit(
            {
                "apiKey": config.api_key,
                "secret": config.api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": config.market_type,  # 'linear', 'inverse', 'spot'
                    "adjustForTimeDifference": True,
                    # Consider adding rate limit adjustments if needed
                    # 'rateLimit': 100, # Example: Adjust if needed (default is usually sufficient)
                },
            }
        )
        # Bybit rate limits (v5 API): 10 req/sec, 600 req/min per UID per endpoint group
        # OHLCV is typically in a less restrictive group than orders/positions
        logger.debug(f"Initialized Bybit connection for {config.market_type} markets.")

    async def initialize(self):
        """Load markets and set leverage."""
        try:
            await self.exchange.load_markets()
            logger.info(f"Loaded {len(self.exchange.markets)} markets from Bybit.")
            if self.config.symbol not in self.exchange.markets:
                logger.critical(
                    f"{Fore.RED}Symbol {self.config.symbol} not found on Bybit {self.config.market_type} market. Available symbols might differ based on market type (linear/inverse). Exiting."
                )
                # Consider listing available symbols here if helpful
                # print("Available symbols:", list(self.exchange.markets.keys()))
                await self.close()
                sys.exit(1)

            # Set leverage only for derivatives markets
            if self.config.market_type in ["linear", "inverse", "swap"]:
                await self._set_leverage()
            else:
                logger.info("Leverage setting skipped for spot market.")

        except ccxt.AuthenticationError as e:
            logger.critical(
                f"{Fore.RED}Bybit Authentication Error: {e}. Check API Key/Secret. Exiting."
            )
            await self.close()
            sys.exit(1)
        except ccxt.ExchangeNotAvailable as e:
            logger.critical(f"{Fore.RED}Bybit exchange not available: {e}. Exiting.")
            await self.close()
            sys.exit(1)
        except Exception as e:
            logger.critical(
                f"{Fore.RED}Failed to initialize Bybit exchange: {e}", exc_info=True
            )
            await self.close()
            sys.exit(1)

    async def _set_leverage(self):
        """Sets the leverage for the configured symbol."""
        symbol_leverage = self.config.leverage  # Static for now, dynamic can be complex
        try:
            # Bybit v5 API requires category for setting leverage
            await self.exchange.set_leverage(
                symbol_leverage,
                self.config.symbol,
                params={"category": self.config.market_type},
            )
            logger.info(f"Set leverage for {self.config.symbol} to {symbol_leverage}x")
        except ccxt.ExchangeError as e:
            # Handle common leverage errors specifically if possible
            if "leverage not modified" in str(e).lower():
                logger.info(
                    f"Leverage for {self.config.symbol} already set to {symbol_leverage}x."
                )
            else:
                logger.error(
                    f"{Fore.YELLOW}Could not set leverage for {self.config.symbol} to {symbol_leverage}x: {e}. Manual adjustment might be needed."
                )
        except Exception as e:
            logger.error(
                f"{Fore.RED}An unexpected error occurred while setting leverage: {e}",
                exc_info=True,
            )

    async def _fetch_with_retries(self, method: callable, *args, **kwargs) -> Any:
        """Wrapper for exchange calls with retry logic."""
        for attempt in range(self.config.max_fetch_retries + 1):
            if self.config.shutdown_requested:  # Check global shutdown flag
                logger.warning("Shutdown requested, aborting API call.")
                return None
            try:
                return await method(*args, **kwargs)
            except (
                ccxt.NetworkError,
                ccxt.RequestTimeout,
                ccxt.ExchangeNotAvailable,
            ) as e:
                if attempt < self.config.max_fetch_retries:
                    wait_time = (2**attempt) + (
                        random.random() * 0.5
                    )  # Exponential backoff with jitter
                    logger.warning(
                        f"{Fore.YELLOW}Network/Timeout Error ({type(e).__name__}) calling {method.__name__} (Attempt {attempt + 1}/{self.config.max_fetch_retries}): {e}. Retrying in {wait_time:.2f}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"{Fore.RED}Network/Timeout Error calling {method.__name__} after {self.config.max_fetch_retries} retries: {e}"
                    )
                    return None  # Failed after retries
            except ccxt.RateLimitExceeded as e:
                logger.warning(
                    f"{Fore.YELLOW}Rate Limit Exceeded calling {method.__name__}: {e}. Retrying after recommended delay..."
                )
                # ccxt might automatically handle retry-after header, but explicit sleep is safer
                await asyncio.sleep(
                    self.exchange.rateLimit / 1000 * 1.1
                )  # Sleep a bit longer than required
                # Rerun the current attempt
                continue  # Don't increment attempt count for rate limit retry
            except ccxt.AuthenticationError as e:
                logger.critical(
                    f"{Fore.RED}Authentication Error during API call ({method.__name__}): {e}. Halting bot."
                )
                self.config.shutdown_requested = True  # Trigger shutdown
                # await self.close() # Close might fail if auth is already bad
                sys.exit(1)
            except ccxt.ExchangeError as e:
                # Catch specific exchange errors if needed (e.g., Insufficient Funds)
                logger.error(f"{Fore.RED}Exchange Error calling {method.__name__}: {e}")
                return None  # Usually non-recoverable without state change
            except Exception as e:
                logger.error(
                    f"{Fore.RED}Unexpected Error calling {method.__name__}: {e}",
                    exc_info=True,
                )
                return None  # Failed due to unexpected issue
        return None  # Should not be reached if retries > 0

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> List:
        """Fetch OHLCV data with retries."""
        # Add category param for Bybit v5 API if needed (depends on ccxt version/implementation)
        params = (
            {"category": self.config.market_type}
            if self.config.market_type != "spot"
            else {}
        )
        # Limit adjustment: Bybit v5 allows up to 1000, but default might be lower.
        # Ensure limit doesn't exceed exchange max.
        exchange_limit = self.exchange.limits.get("fetchOHLCV", {}).get("max", 1000)
        actual_limit = min(limit, exchange_limit)
        if limit > exchange_limit:
            logger.debug(
                f"Requested OHLCV limit {limit} exceeds exchange max {exchange_limit}. Using {actual_limit}."
            )

        data = await self._fetch_with_retries(
            self.exchange.fetch_ohlcv,
            symbol,
            timeframe,
            limit=actual_limit,
            params=params,
        )
        return data if data else []  # Return empty list on failure

    async def fetch_balance(self) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Fetch balance with retries."""
        params = {
            "type": self.config.market_type
        }  # Use 'type' for Bybit (or 'category' depending on ccxt needs)
        # Bybit v5 uses account type: UNIFIED or CONTRACT
        account_type = (
            "CONTRACT" if self.config.market_type != "spot" else "UNIFIED"
        )  # Adjust if using Unified account
        params = {"accountType": account_type}

        balance_data = await self._fetch_with_retries(
            self.exchange.fetch_balance, params=params
        )
        if balance_data:
            try:
                quote_currency = self.exchange.market(self.config.symbol)["quote"]
                # Accessing balance might differ based on account type (Unified vs Contract)
                # This structure assumes a standard ccxt balance response format
                if quote_currency in balance_data:
                    free = Decimal(str(balance_data[quote_currency].get("free", "0")))
                    total = Decimal(str(balance_data[quote_currency].get("total", "0")))
                    # For derivatives, 'total' often represents equity more accurately than 'free' + 'used'
                    equity = (
                        Decimal(
                            str(
                                balance_data["info"]
                                .get("result", {})
                                .get("list", [{}])[0]
                                .get("equity", "0")
                            )
                        )
                        if self.config.market_type != "spot"
                        else total
                    )

                    # Use equity if available for derivatives, otherwise total
                    balance_total = (
                        equity
                        if self.config.market_type != "spot" and equity > 0
                        else total
                    )

                    logger.debug(
                        f"Fetched balance: Free={free} {quote_currency}, Total/Equity={balance_total} {quote_currency}"
                    )
                    return free, balance_total
                else:
                    logger.warning(
                        f"Quote currency '{quote_currency}' not found in balance response: {balance_data}"
                    )
                    return Decimal("0"), Decimal("0")

            except KeyError as e:
                logger.error(
                    f"{Fore.RED}Error parsing balance response structure: {e}. Response: {balance_data}"
                )
                return None, None
            except Exception as e:
                logger.error(
                    f"{Fore.RED}Error processing balance data: {e}", exc_info=True
                )
                return None, None
        return None, None  # Return None if fetch failed

    async def create_market_order(
        self, symbol: str, side: str, amount: float, params: Dict = {}
    ) -> Optional[Dict]:
        """Place market order with retries."""
        # Ensure category is included for v5 API
        params["category"] = self.config.market_type
        # Bybit may require positionIdx for hedge mode, typically 0 for one-way
        params["positionIdx"] = params.get("positionIdx", 0)
        order = await self._fetch_with_retries(
            self.exchange.create_market_order, symbol, side, amount, params=params
        )
        return order

    async def set_trading_stop(self, symbol: str, params: Dict) -> Optional[Dict]:
        """Set SL/TP with retries."""
        params["category"] = self.config.market_type
        params["symbol"] = symbol
        # Use the specific v5 endpoint: POST /v5/position/trading-stop
        response = await self._fetch_with_retries(
            self.exchange.private_post_position_trading_stop, params=params
        )
        # Check response structure for success (e.g., retCode == 0)
        if response and response.get("retCode") == 0:
            logger.debug(f"Successfully set trading stop: {params}")
            return response
        else:
            logger.error(
                f"Failed to set trading stop. Params: {params}, Response: {response}"
            )
            return None

    async def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        """Fetch ticker with retries."""
        params = {"category": self.config.market_type}
        ticker = await self._fetch_with_retries(
            self.exchange.fetch_ticker, symbol, params=params
        )
        return ticker

    async def fetch_position(self, symbol: str) -> Optional[Dict]:
        """Fetch current position for the symbol."""
        params = {"category": self.config.market_type, "symbol": symbol}
        try:
            positions = await self._fetch_with_retries(
                self.exchange.fetch_positions, symbols=[symbol], params=params
            )
            if positions and len(positions) > 0:
                # Find the position matching the symbol (fetch_positions might return others)
                for pos in positions:
                    # Position structure might vary slightly; adapt as needed
                    # Check 'contracts', 'contractSize', 'side', 'unrealizedPnl', etc.
                    # Example check based on common ccxt structure:
                    if (
                        pos.get("symbol") == symbol
                        or pos.get("info", {}).get("symbol") == symbol
                    ):
                        logger.debug(
                            f"Fetched position for {symbol}: Size={pos.get('contracts', 0)}, Side={pos.get('side')}, Entry={pos.get('entryPrice')}"
                        )
                        return pos
                logger.debug(
                    f"No specific position found for {symbol} among fetched positions: {positions}"
                )
                return None  # No position found for the specific symbol
            else:
                logger.debug(f"No open positions found for {symbol}.")
                return None
        except Exception as e:
            logger.error(
                f"{Fore.RED}Error fetching position for {symbol}: {e}", exc_info=True
            )
            return None

    def price_to_precision(self, symbol: str, price: Union[float, Decimal]) -> str:
        """Format price to exchange precision."""
        try:
            return self.exchange.price_to_precision(symbol, float(price))
        except Exception as e:
            logger.warning(
                f"Could not get precision for {symbol} from exchange: {e}. Using default formatting."
            )
            # Fallback formatting
            return f"{Decimal(str(price)):.8f}"  # Adjust decimal places as needed

    def amount_to_precision(
        self, symbol: str, amount: Union[float, Decimal], rounding_mode=ccxt.TRUNCATE
    ) -> str:
        """Format amount to exchange precision."""
        try:
            # Ensure rounding_mode is passed correctly if needed, default is TRUNCATE
            return self.exchange.amount_to_precision(
                symbol, float(amount), rounding_mode=rounding_mode
            )
        except Exception as e:
            logger.warning(
                f"Could not get amount precision for {symbol} from exchange: {e}. Using default formatting."
            )
            # Fallback formatting
            return f"{Decimal(str(amount)):.8f}"  # Adjust decimal places as needed

    async def close(self):
        """Close the exchange connection."""
        try:
            if self.exchange and hasattr(self.exchange, "close"):
                await self.exchange.close()
                logger.info("Bybit exchange connection closed.")
        except Exception as e:
            logger.error(f"{Fore.RED}Error closing Bybit connection: {e}")


# --- Indicator Registry ---
class IndicatorRegistry:
    """Simple registry for technical indicator calculation functions."""

    def __init__(self):
        self.indicators = {}

    def register(self, name: str, func: callable):
        """Register an indicator calculation function."""
        if name in self.indicators:
            logger.warning(f"Indicator '{name}' is already registered. Overwriting.")
        self.indicators[name] = func
        logger.debug(f"Registered indicator: {name}")

    def calculate(self, name: str, df: pd.DataFrame, **kwargs) -> Any:
        """Calculate a registered indicator."""
        if name not in self.indicators:
            raise ValueError(f"Indicator '{name}' not registered.")
        try:
            # Pass only the necessary arguments defined by the function
            # This avoids errors if kwargs contains extra items
            import inspect

            sig = inspect.signature(self.indicators[name])
            valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return self.indicators[name](df, **valid_kwargs)
        except Exception as e:
            logger.error(
                f"{Fore.RED}Error calculating indicator '{name}': {e}", exc_info=True
            )
            # Return a default value that signals an error, e.g., NaN or None
            # The type depends on what the calling code expects
            return None  # Or pd.NA, or (pd.NA, pd.NA) for multi-return functions


# --- Indicator Calculations ---
class IndicatorCalculator:
    """Handles technical indicator calculations using Decimal precision where appropriate."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.registry = IndicatorRegistry()
        # Register indicator calculation methods
        self.registry.register("ema", self._calculate_ema)
        self.registry.register("atr", self._calculate_atr)
        self.registry.register("stochastic", self._calculate_stochastic)
        logger.info("Indicator calculator initialized.")

    def calculate_indicators(
        self, df: pd.DataFrame
    ) -> Optional[Dict[str, Union[Decimal, bool, int, None]]]:
        """Calculates all registered indicators for the given DataFrame."""
        required_length = self._get_max_required_period()
        if df.empty or len(df) < required_length:
            logger.warning(
                f"Insufficient data for indicator calculation. Need {required_length}, have {len(df)}."
            )
            return None

        # Validate data integrity (optional but recommended)
        if df[["open", "high", "low", "close"]].isnull().values.any():
            logger.warning(
                "NaN values found in OHLC data. Indicator results may be inaccurate."
            )
            # Optionally drop rows with NaNs or fill them, depending on strategy
            # df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
            # if len(df) < required_length: return None # Recheck length after dropping

        # Check for large time gaps (more than 2x interval)
        time_diff = df.index.to_series().diff()
        expected_interval = pd.Timedelta(self.config.interval)
        if (time_diff > (2 * expected_interval)).any():
            logger.warning(
                "Gaps detected in OHLCV data timestamps. Results may be inaccurate."
            )

        try:
            df_calc = df.copy()  # Work on a copy
            # Ensure correct types for calculation (use float for pandas/numpy speed)
            df_calc["open"] = df_calc["open"].astype(float)
            df_calc["high"] = df_calc["high"].astype(float)
            df_calc["low"] = df_calc["low"].astype(float)
            df_calc["close"] = df_calc["close"].astype(float)

            indicators = {}
            # Calculate indicators using the registry
            indicators["trend_ema"] = self.registry.calculate(
                "ema", df_calc, period=self.config.trend_ema_period
            )
            indicators["fast_ema"] = self.registry.calculate(
                "ema", df_calc, period=self.config.fast_ema_period
            )
            indicators["slow_ema"] = self.registry.calculate(
                "ema", df_calc, period=self.config.slow_ema_period
            )
            indicators["atr"] = self.registry.calculate(
                "atr", df_calc, period=self.config.atr_period
            )

            stoch_result = self.registry.calculate(
                "stochastic",
                df_calc,
                k_period=self.config.stoch_period,
                smooth_k=self.config.stoch_smooth_k,
                smooth_d=self.config.stoch_smooth_d,
            )

            # Handle potential failure in stochastic calculation
            if stoch_result is None or stoch_result == (None, None):
                indicators["stoch_k"], indicators["stoch_d"] = None, None
                indicators["stoch_kd_bullish"] = False
                indicators["stoch_kd_bearish"] = False
                logger.warning("Stochastic calculation failed, returning None.")
            else:
                stoch_k_series, stoch_d_series = stoch_result
                # Get the last non-NaN value if possible
                last_k = (
                    stoch_k_series.dropna().iloc[-1]
                    if not stoch_k_series.dropna().empty
                    else None
                )
                last_d = (
                    stoch_d_series.dropna().iloc[-1]
                    if not stoch_d_series.dropna().empty
                    else None
                )
                prev_k = (
                    stoch_k_series.dropna().iloc[-2]
                    if len(stoch_k_series.dropna()) >= 2
                    else None
                )
                prev_d = (
                    stoch_d_series.dropna().iloc[-2]
                    if len(stoch_d_series.dropna()) >= 2
                    else None
                )

                indicators["stoch_k"] = (
                    Decimal(str(last_k)) if last_k is not None else None
                )
                indicators["stoch_d"] = (
                    Decimal(str(last_d)) if last_d is not None else None
                )

                # Calculate K/D crossover conditions safely
                indicators["stoch_kd_bullish"] = (
                    last_k is not None
                    and last_d is not None
                    and prev_k is not None
                    and prev_d is not None
                ) and (last_k > last_d and prev_k <= prev_d)
                indicators["stoch_kd_bearish"] = (
                    last_k is not None
                    and last_d is not None
                    and prev_k is not None
                    and prev_d is not None
                ) and (last_k < last_d and prev_k >= prev_d)

            # Check if any essential indicator failed
            if any(
                v is None
                for k, v in indicators.items()
                if k not in ["stoch_kd_bullish", "stoch_kd_bearish"]
            ):  # Exclude boolean flags
                logger.error(
                    f"{Fore.RED}One or more essential indicators failed calculation."
                )
                return None

            indicators["atr_period"] = (
                self.config.atr_period
            )  # Add period for context if needed
            logger.debug("Indicators calculated successfully.")
            return indicators

        except Exception as e:
            logger.error(
                f"{Fore.RED}Critical error during indicator calculation: {e}",
                exc_info=True,
            )
            return None

    def _calculate_ema(self, df: pd.DataFrame, period: int) -> Optional[Decimal]:
        """Calculates Exponential Moving Average."""
        if period <= 0 or len(df) < period:
            return None
        try:
            ema = df["close"].ewm(span=period, adjust=False).mean()
            last_ema = ema.dropna().iloc[-1] if not ema.dropna().empty else None
            return Decimal(str(last_ema)) if last_ema is not None else None
        except Exception as e:
            logger.error(f"Error calculating EMA(period={period}): {e}")
            return None

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> Optional[Decimal]:
        """Calculates Average True Range."""
        if period <= 0 or len(df) < period + 1:
            return None  # Need one prior close
        try:
            high = df["high"]
            low = df["low"]
            close_prev = df["close"].shift(1)
            # Calculate True Range, handling potential NaNs from shift
            tr1 = high - low
            tr2 = (high - close_prev).abs()
            tr3 = (low - close_prev).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.ewm(span=period, adjust=False).mean()
            last_atr = atr.dropna().iloc[-1] if not atr.dropna().empty else None
            return Decimal(str(last_atr)) if last_atr is not None else None
        except Exception as e:
            logger.error(f"Error calculating ATR(period={period}): {e}")
            return None

    def _calculate_stochastic(
        self, df: pd.DataFrame, k_period: int, smooth_k: int, smooth_d: int
    ) -> Optional[Tuple[pd.Series, pd.Series]]:
        """Calculates Stochastic Oscillator (%K and %D). Returns the full series."""
        required = (
            k_period + smooth_k + smooth_d - 2
        )  # Min length needed for calculation
        if k_period <= 0 or smooth_k <= 0 or smooth_d <= 0 or len(df) < required:
            return None, None
        try:
            low_min = df["low"].rolling(window=k_period).min()
            high_max = df["high"].rolling(window=k_period).max()
            # Calculate %K, handle division by zero
            delta = high_max - low_min
            k = pd.Series(
                np.where(delta == 0, 0, 100 * (df["close"] - low_min) / delta),
                index=df.index,
            )
            k_smooth = k.rolling(window=smooth_k).mean()  # %K (smoothed)
            d = k_smooth.rolling(window=smooth_d).mean()  # %D
            return k_smooth, d
        except Exception as e:
            logger.error(
                f"Error calculating Stochastic(k={k_period}, sk={smooth_k}, sd={smooth_d}): {e}"
            )
            return None, None

    def _get_max_required_period(self) -> int:
        """Determines the minimum DataFrame length needed for all indicators."""
        stoch_req = (
            self.config.stoch_period
            + self.config.stoch_smooth_k
            + self.config.stoch_smooth_d
            - 2
        )
        ema_req = max(
            self.config.trend_ema_period,
            self.config.fast_ema_period,
            self.config.slow_ema_period,
        )
        atr_req = self.config.atr_period + 1
        return max(stoch_req, ema_req, atr_req, 1)  # Ensure at least 1


# --- Strategy Interface ---
class StrategyBase(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, config: TradingConfig):
        self.config = config

    @abstractmethod
    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict,
        current_position: Optional[Dict],
        equity: Decimal,
        mtf_indicators: Optional[Dict] = None,
    ) -> Dict:
        """
        Generates trading signals based on indicators and current state.

        Args:
            df: DataFrame with OHLCV data for the primary interval.
            indicators: Dictionary of calculated indicators for the primary interval.
            current_position: Dictionary representing the current position details (or None).
            equity: Current account equity.
            mtf_indicators: Dictionary of indicators for the multi-timeframe (or None).

        Returns:
            A dictionary containing signals like {'entry_long': bool, 'exit_long': bool, 'entry_short': bool, 'exit_short': bool, 'reason': str}.
        """
        pass


class EMAStochasticStrategy(StrategyBase):
    """Trading strategy based on EMA crossovers and Stochastic Oscillator."""

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict,
        current_position: Optional[Dict],
        equity: Decimal,
        mtf_indicators: Optional[Dict] = None,
    ) -> Dict:
        """Generates entry/exit signals for long and short positions."""
        signals = {
            "entry_long": False,
            "exit_long": False,
            "entry_short": False,
            "exit_short": False,
            "reason": "No Signal",
        }

        # Ensure essential indicators are present and valid
        required_keys = [
            "fast_ema",
            "slow_ema",
            "trend_ema",
            "stoch_k",
            "stoch_d",
            "atr",
            "stoch_kd_bullish",
            "stoch_kd_bearish",
        ]
        if any(indicators.get(key) is None for key in required_keys):
            signals["reason"] = "Indicators Incomplete"
            logger.warning(
                f"Signal generation skipped: Incomplete indicators - {indicators}"
            )
            return signals

        try:
            current_price = Decimal(str(df["close"].iloc[-1]))
            fast_ema = indicators["fast_ema"]
            slow_ema = indicators["slow_ema"]
            trend_ema = indicators["trend_ema"]
            k = indicators["stoch_k"]
            d = indicators["stoch_d"]
            atr = indicators["atr"]
            stoch_kd_bullish = indicators[
                "stoch_kd_bullish"
            ]  # K crossed above D in the last candle
            stoch_kd_bearish = indicators[
                "stoch_kd_bearish"
            ]  # K crossed below D in the last candle

            # --- Core Conditions ---
            ema_bullish_cross = fast_ema > slow_ema
            ema_bearish_cross = fast_ema < slow_ema
            # Stochastic entry: Cross OR deep in oversold/overbought territory
            stoch_entry_long_cond = (
                stoch_kd_bullish or k < self.config.stoch_oversold_threshold
            )
            stoch_entry_short_cond = (
                stoch_kd_bearish or k > self.config.stoch_overbought_threshold
            )
            # Stochastic exit: Opposite cross (more sensitive exit)
            stoch_exit_long_cond = stoch_kd_bearish  # Exit long if K crosses below D
            stoch_exit_short_cond = stoch_kd_bullish  # Exit short if K crosses above D

            # --- Trend Filter ---
            # Price relative to the longer-term trend EMA
            buffer_points = trend_ema * (
                self.config.trend_filter_buffer_percent / Decimal("100")
            )
            price_above_trend = current_price > trend_ema  # Simple check
            price_below_trend = current_price < trend_ema  # Simple check
            price_within_trend_buffer_long = current_price >= trend_ema - buffer_points
            price_within_trend_buffer_short = current_price <= trend_ema + buffer_points

            # --- Volatility Filter ---
            # Check if ATR is too high relative to price (potential whipsaw)
            max_allowed_atr = current_price * (
                self.config.volatility_threshold / Decimal("100")
            )
            volatility_ok = atr <= max_allowed_atr
            volatility_reason = f"Volatility {'OK' if volatility_ok else 'High'} (ATR={atr:.{8}f} vs Max={max_allowed_atr:.{8}f})"

            # --- Significant Move Filter (Optional) ---
            # Check if the last candle's move was significant compared to ATR
            is_significant_move = (
                True  # Default to true if filter not used or data insufficient
            )
            atr_move_reason = "Move Filter N/A"
            if len(df) >= 2 and self.config.atr_move_filter_multiplier > 0:
                prev_close = Decimal(str(df["close"].iloc[-2]))
                price_move = abs(current_price - prev_close)
                atr_threshold = atr * self.config.atr_move_filter_multiplier
                is_significant_move = price_move >= atr_threshold
                atr_move_reason = f"Move Filter {'Passed' if is_significant_move else 'Failed'} (|ΔP|={price_move:.{8}f} vs Thr={atr_threshold:.{8}f})"

            # --- Multi-Timeframe (MTF) Filter ---
            mtf_confirm_long = True  # Default to true if MTF disabled or data missing
            mtf_confirm_short = True
            mtf_reason = "MTF N/A"
            if (
                mtf_indicators
                and "trend_ema" in mtf_indicators
                and mtf_indicators["trend_ema"] is not None
            ):
                mtf_trend_ema = mtf_indicators["trend_ema"]
                # Long entries confirmed if primary price is above MTF trend
                mtf_confirm_long = current_price > mtf_trend_ema
                # Short entries confirmed if primary price is below MTF trend
                mtf_confirm_short = current_price < mtf_trend_ema
                mtf_reason = f"MTF Long {'OK' if mtf_confirm_long else 'BLK'} / Short {'OK' if mtf_confirm_short else 'BLK'} (Price={current_price:.{8}f} vs MTF EMA={mtf_trend_ema:.{8}f})"
            elif mtf_indicators:
                mtf_reason = "MTF Incomplete"

            # --- Position State ---
            position_size = (
                Decimal(str(current_position["contracts"]))
                if current_position and "contracts" in current_position
                else Decimal("0")
            )
            is_long = (
                position_size > self.config.position_qty_epsilon
                and current_position.get("side") == "long"
            )
            is_short = (
                position_size > self.config.position_qty_epsilon
                and current_position.get("side") == "short"
            )
            in_position = is_long or is_short

            # --- Signal Logic ---
            reason_parts = []

            # == Entry Signals ==
            if not in_position:
                # Long Entry Conditions
                can_enter_long = (
                    ema_bullish_cross
                    and stoch_entry_long_cond
                    and volatility_ok
                    and is_significant_move
                    and mtf_confirm_long
                )
                if self.config.trade_only_with_trend:
                    can_enter_long = (
                        can_enter_long and price_within_trend_buffer_long
                    )  # Must be near or above trend

                if can_enter_long:
                    signals["entry_long"] = True
                    reason_parts.append(f"{Fore.GREEN}Long Entry:")
                    reason_parts.append(
                        f"EMA Bullish ({fast_ema:.{6}f} > {slow_ema:.{6}f})"
                    )
                    reason_parts.append(
                        f"Stoch OK (K={k:.2f} < {self.config.stoch_oversold_threshold} or K/D Cross)"
                    )
                    if self.config.trade_only_with_trend:
                        reason_parts.append(
                            f"Trend OK (P={current_price:.{6}f} >= TrendEMA={trend_ema:.{6}f})"
                        )
                    reason_parts.append(volatility_reason)
                    reason_parts.append(atr_move_reason)
                    reason_parts.append(mtf_reason)
                else:
                    # Short Entry Conditions (check only if no long entry)
                    can_enter_short = (
                        ema_bearish_cross
                        and stoch_entry_short_cond
                        and volatility_ok
                        and is_significant_move
                        and mtf_confirm_short
                    )
                    if self.config.trade_only_with_trend:
                        can_enter_short = (
                            can_enter_short and price_within_trend_buffer_short
                        )  # Must be near or below trend

                    if can_enter_short:
                        signals["entry_short"] = True
                        reason_parts.append(f"{Fore.RED}Short Entry:")
                        reason_parts.append(
                            f"EMA Bearish ({fast_ema:.{6}f} < {slow_ema:.{6}f})"
                        )
                        reason_parts.append(
                            f"Stoch OK (K={k:.2f} > {self.config.stoch_overbought_threshold} or K/D Cross)"
                        )
                        if self.config.trade_only_with_trend:
                            reason_parts.append(
                                f"Trend OK (P={current_price:.{6}f} <= TrendEMA={trend_ema:.{6}f})"
                            )
                        reason_parts.append(volatility_reason)
                        reason_parts.append(atr_move_reason)
                        reason_parts.append(mtf_reason)

            # == Exit Signals ==
            elif is_long:
                # Exit Long Conditions: EMA bearish cross OR Stochastic bearish cross
                if ema_bearish_cross or stoch_exit_long_cond:
                    signals["exit_long"] = True
                    reason_parts.append(f"{Fore.YELLOW}Exit Long:")
                    if ema_bearish_cross:
                        reason_parts.append(
                            f"EMA Bearish Cross ({fast_ema:.{6}f} < {slow_ema:.{6}f})"
                        )
                    if stoch_exit_long_cond:
                        reason_parts.append(
                            f"Stoch Bearish Cross (K={k:.2f}, D={d:.2f})"
                        )
            elif is_short:
                # Exit Short Conditions: EMA bullish cross OR Stochastic bullish cross
                if ema_bullish_cross or stoch_exit_short_cond:
                    signals["exit_short"] = True
                    reason_parts.append(f"{Fore.YELLOW}Exit Short:")
                    if ema_bullish_cross:
                        reason_parts.append(
                            f"EMA Bullish Cross ({fast_ema:.{6}f} > {slow_ema:.{6}f})"
                        )
                    if stoch_exit_short_cond:
                        reason_parts.append(
                            f"Stoch Bullish Cross (K={k:.2f}, D={d:.2f})"
                        )

            # --- Final Reason Construction ---
            if not reason_parts:
                # Provide basic status if no entry/exit signal generated
                trend_status = (
                    "Above"
                    if price_above_trend
                    else "Below"
                    if price_below_trend
                    else "Neutral"
                )
                ema_status = "Bullish" if ema_bullish_cross else "Bearish"
                stoch_status = f"K={k:.2f} D={d:.2f}"
                pos_status = (
                    f"Pos: {'Long' if is_long else 'Short' if is_short else 'None'}"
                )
                signals["reason"] = (
                    f"No Signal | {pos_status} | EMA: {ema_status} | Trend: {trend_status} | Stoch: {stoch_status} | {volatility_reason}"
                )
            else:
                signals["reason"] = " | ".join(reason_parts)

            # Log the detailed reason
            log_color = (
                Fore.GREEN
                if signals["entry_long"]
                else Fore.RED
                if signals["entry_short"]
                else Fore.YELLOW
                if signals["exit_long"] or signals["exit_short"]
                else Fore.WHITE
            )
            logger.info(f"{log_color}{signals['reason']}{Style.RESET_ALL}")

            return signals

        except Exception as e:
            logger.error(f"{Fore.RED}Signal generation error: {e}", exc_info=True)
            # Return a safe default state
            return {
                "entry_long": False,
                "exit_long": False,
                "entry_short": False,
                "exit_short": False,
                "reason": f"Error: {e}",
            }


# --- Trade Execution ---
class TradeExecutor:
    """Handles placing orders, managing positions, and calculating PnL."""

    def __init__(
        self,
        config: TradingConfig,
        exchange: ExchangeInterface,
        journal: JournalManager,
    ):
        self.config = config
        self.exchange = exchange
        self.journal = journal
        # Simple state tracking (can be enhanced)
        self.current_position = None  # Store position details fetched from exchange
        self.performance = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "pnl": Decimal("0"),
            "equity_history": [],
        }
        logger.info("Trade executor initialized.")

    async def update_position_state(self):
        """Fetches and updates the current position state from the exchange."""
        self.current_position = await self.exchange.fetch_position(self.config.symbol)
        # Log position details if found
        if self.current_position:
            size = self.current_position.get("contracts", 0)
            side = self.current_position.get("side", "none")
            entry = self.current_position.get("entryPrice", 0)
            pnl = self.current_position.get("unrealizedPnl", 0)
            logger.debug(
                f"Current Position: Side={side}, Size={size}, Entry={entry}, uPnL={pnl}"
            )
        else:
            logger.debug("No active position found.")

    async def execute_entry(self, side: str, indicators: Dict):
        """Executes an entry order based on calculated risk."""
        logger.info(f"Attempting {side.upper()} entry...")
        if (
            "atr" not in indicators
            or indicators["atr"] is None
            or indicators["atr"] <= 0
        ):
            logger.error(
                f"{Fore.RED}Cannot calculate position size: ATR is invalid ({indicators.get('atr')})."
            )
            return False

        try:
            balance, equity = await self.exchange.fetch_balance()
            if equity is None or equity <= Decimal("0"):
                logger.error(
                    f"{Fore.RED}Cannot place order: Invalid equity ({equity})."
                )
                return False

            atr = indicators["atr"]
            sl_distance_atr = atr * self.config.sl_atr_multiplier
            if sl_distance_atr <= 0:
                logger.error(
                    f"{Fore.RED}Cannot calculate position size: Stop loss distance is zero or negative (ATR={atr}, SL Mult={self.config.sl_atr_multiplier})."
                )
                return False

            # --- Risk Calculation ---
            risk_amount_per_trade = equity * self.config.risk_percentage
            # Calculate quantity based on risk amount and SL distance in price terms
            # This assumes 1 contract = 1 unit of base currency (e.g., BTC for BTC/USDT)
            # For inverse contracts, this needs adjustment based on contract value
            # TODO: Add handling for inverse contracts if market_type is inverse
            qty_decimal = risk_amount_per_trade / sl_distance_atr
            # Apply max position size constraint relative to equity
            ticker = await self.exchange.fetch_ticker(self.config.symbol)
            if not ticker or "last" not in ticker:
                logger.error(
                    f"{Fore.RED}Cannot place order: Failed to fetch current price."
                )
                return False
            current_price = Decimal(str(ticker["last"]))
            max_position_value = equity * self.config.max_position_percentage
            max_qty_allowed = (
                max_position_value / current_price
            )  # Max quantity based on equity %
            qty_decimal = min(qty_decimal, max_qty_allowed)
            qty_str = self.format_amount(
                self.config.symbol, qty_decimal, rounding_mode=ROUND_DOWN
            )  # Use ROUND_DOWN for orders
            qty_final = Decimal(qty_str)  # Use the precision-adjusted value

            if qty_final <= Decimal(
                self.exchange.exchange.markets[self.config.symbol]["limits"]["amount"][
                    "min"
                ]
            ):
                logger.warning(
                    f"{Fore.YELLOW}Calculated order quantity ({qty_final}) is below minimum allowed. Skipping entry."
                )
                return False

            logger.info(
                f"Calculated Entry: Side={side.upper()}, Qty={qty_final}, Risk={risk_amount_per_trade:.4f}, SL Dist (ATR)={sl_distance_atr:.{8}f}"
            )

            # --- Place Market Order ---
            # positionIdx 0 for one-way mode, 1 for buy hedge, 2 for sell hedge
            position_idx = 0  # Assume one-way mode
            order_params = {"positionIdx": position_idx}
            order = await self.exchange.create_market_order(
                symbol=self.config.symbol,
                side=side,
                amount=float(qty_final),  # CCXT usually expects float
                params=order_params,
            )

            if order and order.get("id"):
                # --- Order successful, now set SL/TP ---
                entry_price = Decimal(
                    str(order.get("average", current_price))
                )  # Use avg fill price if available, else last price
                logger.trade(
                    f"{Fore.GREEN}ENTRY EXECUTED: {side.upper()} {qty_final} {self.config.symbol} @ ~{entry_price:.{8}f}"
                )

                # Set Stop Loss
                await self.set_stop_loss(side, entry_price, atr)
                # Set Take Profits (optional, can be managed differently)
                await self.set_take_profits(side, entry_price, atr, qty_final)

                # Log to journal
                self.journal.log_trade(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "symbol": self.config.symbol,
                        "side": side,
                        "price": f"{entry_price:.{8}f}",
                        "quantity": str(qty_final),
                        "reason": "Entry Signal",
                    }
                )
                self.performance["trades"] += 1
                self.performance["equity_history"].append(
                    equity
                )  # Record equity before trade impact
                await self.send_termux_sms(
                    f"✅ ENTRY: {side.upper()} {qty_final} {self.config.symbol} @ {entry_price:.4f}"
                )
                await self.update_position_state()  # Update position state immediately
                return True
            else:
                logger.error(
                    f"{Fore.RED}Market {side.upper()} order placement failed. Response: {order}"
                )
                await self.send_termux_sms(
                    f"⚠️ ORDER FAIL: {side.upper()} {self.config.symbol}"
                )
                return False

        except ccxt.InsufficientFunds as e:
            logger.error(
                f"{Fore.RED}Insufficient funds to place {side.upper()} order: {e}"
            )
            await self.send_termux_sms(
                f"❌ INSUFFICIENT FUNDS: {side.upper()} {self.config.symbol}"
            )
            return False
        except Exception as e:
            logger.error(
                f"{Fore.RED}Error during {side.upper()} entry execution: {e}",
                exc_info=True,
            )
            return False

    async def execute_exit(self, side_to_close: str, reason: str):
        """Closes the current position with a market order."""
        if not self.current_position:
            logger.warning("Exit requested but no position found in state.")
            return False

        position_size = Decimal(str(self.current_position.get("contracts", "0")))
        position_side = self.current_position.get("side")  # 'long' or 'short'

        if position_size <= self.config.position_qty_epsilon:
            logger.warning(
                f"Exit requested for {side_to_close} but position size is zero."
            )
            return False
        if position_side != side_to_close:
            logger.warning(
                f"Exit requested for {side_to_close} but current position is {position_side}."
            )
            return False

        exit_side = "sell" if side_to_close == "long" else "buy"
        qty_to_close = float(position_size)  # Close the entire position
        logger.info(
            f"Attempting to close {side_to_close.upper()} position (Size: {qty_to_close}) via market order ({exit_side.upper()}). Reason: {reason}"
        )

        try:
            # Use reduceOnly flag to ensure it only closes the position
            order_params = {"reduceOnly": True, "positionIdx": 0}  # Add positionIdx
            order = await self.exchange.create_market_order(
                symbol=self.config.symbol,
                side=exit_side,
                amount=qty_to_close,
                params=order_params,
            )

            if order and order.get("id"):
                exit_price = Decimal(
                    str(order.get("average", order.get("price", "0")))
                )  # Get fill price
                entry_price = Decimal(str(self.current_position.get("entryPrice", "0")))
                pnl = (
                    (exit_price - entry_price) * position_size
                    if side_to_close == "long"
                    else (entry_price - exit_price) * position_size
                )
                # TODO: Account for fees in PnL calculation

                logger.trade(
                    f"{Fore.YELLOW}EXIT EXECUTED: Closed {side_to_close.upper()} {position_size} {self.config.symbol} @ ~{exit_price:.{8}f}. Reason: {reason}. PnL: {pnl:.{8}f}"
                )

                # Update performance metrics
                if pnl > 0:
                    self.performance["wins"] += 1
                else:
                    self.performance["losses"] += 1
                self.performance["pnl"] += pnl
                _, equity = await self.exchange.fetch_balance()
                if equity:
                    self.performance["equity_history"].append(equity)

                # Log to journal
                self.journal.log_trade(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "symbol": self.config.symbol,
                        "side": exit_side,
                        "price": f"{exit_price:.{8}f}",
                        "quantity": str(position_size),
                        "pnl": f"{pnl:.{8}f}",
                        "reason": f"Exit Signal ({reason})",
                    }
                )
                await self.send_termux_sms(
                    f"⛔ EXIT: {side_to_close.upper()} {self.config.symbol} @ {exit_price:.4f}. PnL: {pnl:.2f}. Reason: {reason}"
                )
                self.current_position = None  # Clear position state after closing
                # Optionally cancel remaining TP orders here if any
                return True
            else:
                logger.error(
                    f"{Fore.RED}Market {exit_side.upper()} (close) order placement failed. Response: {order}"
                )
                await self.send_termux_sms(
                    f"⚠️ CLOSE FAIL: {side_to_close.upper()} {self.config.symbol}"
                )
                # Position might still be open, re-fetch state in next cycle
                await self.update_position_state()
                return False

        except ccxt.OrderNotFound as e:
            logger.warning(
                f"{Fore.YELLOW}Order to close position might have already been filled or cancelled: {e}"
            )
            # Potentially already closed by SL/TP, re-fetch state
            await self.update_position_state()
            return True  # Assume closed successfully if order not found
        except Exception as e:
            logger.error(
                f"{Fore.RED}Error during {exit_side.upper()} (close) execution: {e}",
                exc_info=True,
            )
            return False

    async def set_stop_loss(
        self, side: str, entry_price: Decimal, atr: Decimal
    ) -> bool:
        """Sets the initial stop-loss order based on ATR."""
        if atr <= 0:
            logger.error("Cannot set SL: Invalid ATR value.")
            return False
        try:
            sl_price = (
                entry_price - (atr * self.config.sl_atr_multiplier)
                if side == "buy"
                else entry_price + (atr * self.config.sl_atr_multiplier)
            )
            sl_price_str = self.format_price(self.config.symbol, sl_price)

            params = {
                "symbol": self.config.symbol,  # Already included in method signature
                "stopLoss": sl_price_str,
                "slTriggerBy": self.config.sl_trigger_by,  # Correct param name for v5 API
                "positionIdx": 0,  # Assume one-way mode
                # 'tpslMode': 'Full' # Ensure it applies to the whole position
            }
            logger.debug(f"Setting SL with params: {params}")
            response = await self.exchange.set_trading_stop(
                symbol=self.config.symbol, params=params
            )

            if (
                response
            ):  # Check if response is not None (success checked within exchange method)
                logger.trade(
                    f"{Fore.YELLOW}STOP LOSS set for {side.upper()} position at {sl_price_str} (Trigger: {self.config.sl_trigger_by})"
                )
                return True
            else:
                logger.error(
                    f"{Fore.RED}Failed to set Stop Loss for {side.upper()} position."
                )
                return False
        except Exception as e:
            logger.error(f"{Fore.RED}Error setting Stop Loss: {e}", exc_info=True)
            return False

    async def set_take_profits(
        self, side: str, entry_price: Decimal, atr: Decimal, total_qty: Decimal
    ) -> bool:
        """Sets multiple take-profit levels based on ATR multipliers."""
        if not self.config.take_profit_atr_multipliers or atr <= 0:
            logger.info(
                "Take profit setting skipped (no multipliers configured or invalid ATR)."
            )
            return True  # Not an error if TP is disabled

        num_tps = len(self.config.take_profit_atr_multipliers)
        if num_tps == 0:
            return True  # Nothing to do

        # Bybit v5 TP/SL setting is often for the *entire* position, not partials easily via API.
        # The `set_trading_stop` endpoint sets *one* TP and *one* SL for the position.
        # Achieving multiple TPs might require:
        # 1. Setting only the first TP via API.
        # 2. Monitoring price and placing reduce-only limit orders manually when TP levels are approached.
        # 3. Using a platform feature if available (e.g., conditional orders).

        # Let's implement setting the *first* TP level using the API.
        try:
            first_tp_multiplier = self.config.take_profit_atr_multipliers[0]
            if first_tp_multiplier <= 0:
                logger.warning(
                    "First TP multiplier is zero or negative, skipping TP set."
                )
                return True

            tp_price = (
                entry_price + (atr * first_tp_multiplier)
                if side == "buy"
                else entry_price - (atr * first_tp_multiplier)
            )
            tp_price_str = self.format_price(self.config.symbol, tp_price)

            # We can only set one TP this way. It will apply to the whole position unless manually managed later.
            params = {
                "symbol": self.config.symbol,
                "takeProfit": tp_price_str,
                "tpTriggerBy": self.config.sl_trigger_by,  # Usually same trigger as SL
                "positionIdx": 0,
                # 'tpslMode': 'Full' # Or 'Partial' if supported and desired, but API might not allow qty specification here
            }
            logger.debug(f"Setting TP1 with params: {params}")
            response = await self.exchange.set_trading_stop(
                symbol=self.config.symbol, params=params
            )

            if response:
                logger.trade(
                    f"{Fore.GREEN}TAKE PROFIT 1 set for {side.upper()} position at {tp_price_str} (Trigger: {self.config.sl_trigger_by})"
                )
                # Log other intended TPs for manual reference or future implementation
                for i, mult in enumerate(
                    self.config.take_profit_atr_multipliers[1:], start=2
                ):
                    tp_level_price = (
                        entry_price + (atr * mult)
                        if side == "buy"
                        else entry_price - (atr * mult)
                    )
                    logger.info(
                        f"Intended TP{i} level: {self.format_price(self.config.symbol, tp_level_price)}"
                    )
                return True
            else:
                logger.error(
                    f"{Fore.RED}Failed to set Take Profit 1 for {side.upper()} position."
                )
                return False

        except Exception as e:
            logger.error(f"{Fore.RED}Error setting Take Profit: {e}", exc_info=True)
            return False

    # --- Formatting Helpers ---
    def format_price(self, symbol: str, price: Union[Decimal, str, float, int]) -> str:
        """Safely format price to exchange precision."""
        try:
            if isinstance(price, Decimal) and (price.is_nan() or price.is_infinite()):
                return "NaN"  # Or handle as error
            return self.exchange.price_to_precision(symbol, float(price))
        except Exception as e:
            # Fallback if precision info not available
            precision = 8  # Default precision
            if (
                self.exchange.exchange.markets
                and symbol in self.exchange.exchange.markets
            ):
                precision = (
                    self.exchange.exchange.markets[symbol]
                    .get("precision", {})
                    .get("price", 8)
                )
            logger.debug(
                f"Using fallback price formatting for {symbol} (precision {precision}): {e}"
            )
            return f"{Decimal(str(price)):.{precision}f}"

    def format_amount(
        self,
        symbol: str,
        amount: Union[Decimal, str, float, int],
        rounding_mode=ROUND_DOWN,
    ) -> str:
        """Safely format amount/quantity to exchange precision."""
        try:
            if isinstance(amount, Decimal) and (
                amount.is_nan() or amount.is_infinite()
            ):
                return "NaN"  # Or handle as error
            # Map Decimal rounding modes to CCXT rounding modes if necessary, or use default
            ccxt_rounding = (
                ccxt.TRUNCATE if rounding_mode == ROUND_DOWN else ccxt.ROUND
            )  # Example mapping
            return self.exchange.amount_to_precision(
                symbol, float(amount), rounding_mode=ccxt_rounding
            )
        except Exception as e:
            # Fallback if precision info not available
            precision = 8  # Default precision
            if (
                self.exchange.exchange.markets
                and symbol in self.exchange.exchange.markets
            ):
                precision = (
                    self.exchange.exchange.markets[symbol]
                    .get("precision", {})
                    .get("amount", 8)
                )
            logger.debug(
                f"Using fallback amount formatting for {symbol} (precision {precision}): {e}"
            )
            # Apply rounding mode manually for fallback
            quantizer = Decimal("1e-" + str(precision))
            return str(Decimal(str(amount)).quantize(quantizer, rounding=rounding_mode))

    # --- Notification ---
    async def send_termux_sms(self, message: str):
        """Sends an SMS notification using Termux API."""
        if not self.config.termux_sms_recipient:
            logger.debug("Termux SMS recipient not set. Skipping notification.")
            return

        # Check if running in Termux (basic check)
        if "com.termux" not in os.environ.get("PREFIX", ""):
            logger.warning(
                "Not running in Termux environment. Skipping SMS notification."
            )
            return

        try:
            # Limit message length if necessary (SMS limits apply)
            max_len = 150  # Conservative limit
            truncated_message = (
                message[:max_len] + "..." if len(message) > max_len else message
            )

            command = [
                "termux-sms-send",
                "-n",
                str(self.config.termux_sms_recipient),
                truncated_message,  # Send the potentially truncated message
            ]
            logger.debug(f"Executing Termux command: {' '.join(command)}")

            # Run the command asynchronously
            process = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                logger.info(
                    f"Sent Termux SMS notification to {self.config.termux_sms_recipient}."
                )
            else:
                stderr_str = stderr.decode().strip() if stderr else "No stderr"
                logger.error(
                    f"{Fore.RED}Failed to send Termux SMS. Return code: {process.returncode}. Error: {stderr_str}"
                )

        except FileNotFoundError:
            logger.error(
                f"{Fore.RED}'termux-sms-send' command not found. Is Termux:API installed and configured?"
            )
        except Exception as e:
            logger.error(
                f"{Fore.RED}Failed to send Termux SMS notification: {e}", exc_info=True
            )


# --- Backtesting ---
class Backtester:
    """Simulates trading strategy execution on historical data."""

    def __init__(
        self,
        config: TradingConfig,
        indicator_calculator: IndicatorCalculator,
        strategy: StrategyBase,
    ):
        self.config = config
        self.indicator_calculator = indicator_calculator
        self.strategy = strategy
        logger.info("Backtester initialized.")

    def run_backtest(self, data_path: str) -> Optional[Dict]:
        """Runs the backtest simulation."""
        logger.info(f"Starting backtest using data from: {data_path}")
        try:
            # Load data
            df = pd.read_csv(
                data_path, parse_dates=["timestamp"], index_col="timestamp"
            )
            # Basic data validation
            required_cols = ["open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required_cols):
                logger.critical(
                    f"{Fore.RED}Backtest data missing required columns: {required_cols}. Found: {list(df.columns)}"
                )
                return None
            df = df[required_cols].dropna()  # Ensure only needed columns and no NaNs
            if df.empty:
                logger.critical(
                    f"{Fore.RED}Backtest data is empty or contains only NaNs after cleaning."
                )
                return None
            logger.info(f"Loaded {len(df)} data points for backtesting.")

            # Simulation parameters
            initial_equity = Decimal("10000")  # Starting capital
            equity = initial_equity
            position = None  # Holds current position details: {'side': 'long'/'short', 'qty': Decimal, 'entry_price': Decimal}
            trades = []  # List to store details of closed trades
            equity_curve = [initial_equity]  # Track equity over time

            min_data_length = self.indicator_calculator._get_max_required_period()

            # --- Simulation Loop ---
            for i in range(min_data_length, len(df)):
                if self.config.shutdown_requested:
                    logger.warning("Shutdown requested during backtest.")
                    break

                window_df = df.iloc[: i + 1]  # Expanding window
                current_price = Decimal(str(window_df["close"].iloc[-1]))
                current_time = window_df.index[-1]

                # Calculate indicators for the current window
                indicators = self.indicator_calculator.calculate_indicators(window_df)
                if not indicators:
                    logger.debug(
                        f"Skipping step {i}: Insufficient data or indicator error."
                    )
                    continue  # Skip if indicators can't be calculated

                # Simulate MTF (using same data for simplicity, real backtest needs separate MTF data)
                # mtf_indicators = self.indicator_calculator.calculate_indicators(window_df, mtf=True) # Adapt calc if needed
                mtf_indicators = None  # Placeholder

                # Generate signals based on current state and indicators
                # Pass simulated position state to strategy
                sim_position_state = (
                    {
                        "contracts": position["qty"],
                        "side": position["side"],
                        "entryPrice": position["entry_price"],
                    }
                    if position
                    else None
                )
                signals = self.strategy.generate_signals(
                    window_df, indicators, sim_position_state, equity, mtf_indicators
                )

                # --- Simulate Trade Execution ---
                atr = indicators.get("atr")
                if atr is None or atr <= 0:
                    atr = current_price * Decimal("0.01")  # Fallback ATR if needed

                # Simulate Exits FIRST
                if position:
                    if (position["side"] == "long" and signals["exit_long"]) or (
                        position["side"] == "short" and signals["exit_short"]
                    ):
                        # Simulate closing the position
                        pnl = (
                            (current_price - position["entry_price"]) * position["qty"]
                            if position["side"] == "long"
                            else (position["entry_price"] - current_price)
                            * position["qty"]
                        )
                        equity += pnl
                        trades.append(
                            {
                                "entry_time": position["entry_time"],
                                "exit_time": current_time,
                                "side": position["side"],
                                "entry_price": position["entry_price"],
                                "exit_price": current_price,
                                "qty": position["qty"],
                                "pnl": pnl,
                            }
                        )
                        logger.debug(
                            f"Backtest: Closed {position['side']} @ {current_price:.4f}, PnL: {pnl:.4f}"
                        )
                        position = None  # Clear position

                # Simulate Entries (only if not in position)
                if not position:
                    qty_to_trade = Decimal("0")
                    entry_side = None
                    if signals["entry_long"]:
                        entry_side = "long"
                    elif signals["entry_short"]:
                        entry_side = "short"

                    if entry_side:
                        # Simplified risk sizing for backtest
                        risk_amount = equity * self.config.risk_percentage
                        sl_distance = atr * self.config.sl_atr_multiplier
                        if sl_distance > 0:
                            qty_to_trade = risk_amount / sl_distance
                            # Apply max position constraint (simplified)
                            max_qty = (
                                equity * self.config.max_position_percentage
                            ) / current_price
                            qty_to_trade = min(qty_to_trade, max_qty)

                            # Simulate entry
                            position = {
                                "side": entry_side,
                                "qty": qty_to_trade,
                                "entry_price": current_price,
                                "entry_time": current_time,
                            }
                            logger.debug(
                                f"Backtest: Entered {entry_side} {qty_to_trade:.4f} @ {current_price:.4f}"
                            )
                        else:
                            logger.warning(
                                "Backtest: Skipping entry due to zero SL distance."
                            )

                equity_curve.append(equity)  # Record equity at each step

            # --- Backtest Finished ---
            logger.info(f"Backtest finished. Total Trades: {len(trades)}")

            # --- Calculate Metrics ---
            results = self._calculate_backtest_metrics(
                initial_equity, equity_curve, trades
            )

            # --- Plot Equity Curve ---
            self._plot_equity_curve(equity_curve, df.index[min_data_length:])

            return results

        except FileNotFoundError:
            logger.critical(f"{Fore.RED}Backtest data file not found: {data_path}")
            return None
        except pd.errors.EmptyDataError:
            logger.critical(f"{Fore.RED}Backtest data file is empty: {data_path}")
            return None
        except Exception as e:
            logger.error(
                f"{Fore.RED}An error occurred during backtesting: {e}", exc_info=True
            )
            return None

    def _calculate_backtest_metrics(self, initial_equity, equity_curve, trades):
        """Calculates performance metrics from backtest results."""
        final_equity = equity_curve[-1] if equity_curve else initial_equity
        total_return_pct = (
            ((final_equity / initial_equity) - 1) * 100 if initial_equity > 0 else 0
        )
        num_trades = len(trades)
        wins = sum(1 for t in trades if t["pnl"] > 0)
        losses = num_trades - wins
        win_rate = (wins / num_trades) * 100 if num_trades > 0 else 0
        total_pnl = sum(t["pnl"] for t in trades)
        avg_pnl_per_trade = total_pnl / num_trades if num_trades > 0 else 0
        avg_win = (
            sum(t["pnl"] for t in trades if t["pnl"] > 0) / wins if wins > 0 else 0
        )
        avg_loss = (
            sum(t["pnl"] for t in trades if t["pnl"] < 0) / losses if losses > 0 else 0
        )
        profit_factor = (
            abs(
                sum(t["pnl"] for t in trades if t["pnl"] > 0)
                / sum(t["pnl"] for t in trades if t["pnl"] < 0)
            )
            if losses > 0 and sum(t["pnl"] for t in trades if t["pnl"] < 0) != 0
            else float("inf")
        )

        # Max Drawdown
        equity_series = pd.Series(equity_curve)
        roll_max = equity_series.cummax()
        drawdown = (equity_series / roll_max) - 1
        max_drawdown_pct = abs(drawdown.min()) * 100 if not drawdown.empty else 0

        # Sharpe Ratio (simplified - assumes daily data and risk-free rate = 0)
        # TODO: Adjust Sharpe calculation based on actual data frequency
        returns = equity_series.pct_change().dropna()
        sharpe_ratio = (
            (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        )  # Assuming daily data (252 trading days)

        results = {
            "initial_equity": initial_equity,
            "final_equity": final_equity,
            "total_return_pct": total_return_pct,
            "total_pnl": total_pnl,
            "num_trades": num_trades,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": win_rate,
            "avg_pnl_per_trade": avg_pnl_per_trade,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_drawdown_pct,
            "sharpe_ratio": sharpe_ratio,
            "trades": trades,  # Include raw trades if needed
        }

        # Log metrics
        logger.info("--- Backtest Metrics ---")
        for key, value in results.items():
            if key != "trades":
                # Format nicely
                if "pct" in key or "rate" in key:
                    logger.info(f"{key.replace('_', ' ').title()}: {value:.2f}%")
                elif isinstance(value, Decimal) or isinstance(value, float):
                    logger.info(f"{key.replace('_', ' ').title()}: {value:.4f}")
                else:
                    logger.info(f"{key.replace('_', ' ').title()}: {value}")
        logger.info("------------------------")

        return results

    def _plot_equity_curve(self, equity_curve, index):
        """Plots the equity curve using matplotlib."""
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(
                index, equity_curve[1:], label="Equity Curve", color="blue"
            )  # Skip initial equity for plotting alignment
            plt.title(f"Backtest Equity Curve - {self.config.symbol}")
            plt.xlabel("Time")
            plt.ylabel("Equity")
            plt.legend()
            plt.grid(True)
            plot_filename = f"backtest_equity_{self.config.symbol.replace('/', '_').replace(':', '_')}.png"
            plt.savefig(plot_filename)
            logger.info(f"Equity curve plot saved to {plot_filename}")
            plt.close()  # Close the plot to free memory
        except Exception as e:
            logger.error(f"{Fore.RED}Failed to generate equity curve plot: {e}")


# --- Dashboard ---
class Dashboard:
    """Displays real-time bot status and market data in the CLI using Rich."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.console = Console()
        self.live = None
        self.last_data = {}  # Store last received data

    def start(self):
        """Starts the live updating dashboard."""
        # Initial table structure
        self.live = Live(
            self._generate_table(),
            console=self.console,
            refresh_per_second=1,
            vertical_overflow="visible",
        )
        try:
            self.live.start(refresh=True)
            logger.info("CLI Dashboard started.")
        except Exception as e:
            logger.error(f"Failed to start Rich Live display: {e}")
            self.live = None  # Disable if start fails

    def stop(self):
        """Stops the live dashboard."""
        if self.live:
            try:
                self.live.stop()
                logger.info("CLI Dashboard stopped.")
            except Exception as e:
                logger.error(f"Error stopping Rich Live display: {e}")
            self.live = None

    def update(self, data: Dict):
        """Updates the dashboard with new data."""
        if self.live:
            self.last_data = data  # Store the latest data
            self.live.update(self._generate_table(data))
            logger.debug("Dashboard updated.")

    def _generate_table(self, data: Optional[Dict] = None) -> Table:
        """Generates the Rich Table object for display."""
        if data is None:
            data = self.last_data  # Use last known data if none provided

        table = Table(
            title=f"Trading Bot Status - {self.config.symbol}",
            show_header=True,
            header_style="bold cyan",
            box=box.DOUBLE_EDGE,
            expand=True,
        )

        table.add_column("Metric", style="dim", width=25)
        table.add_column("Value")

        # --- Populate Table ---
        # Market Info
        table.add_row(
            "Symbol", f"[bold white]{data.get('symbol', self.config.symbol)}[/]"
        )
        price = data.get("price")
        price_str = (
            f"{price:.{8}f}" if isinstance(price, Decimal) else str(price or "-")
        )
        table.add_row("Current Price", price_str)
        table.add_row("Interval", self.config.interval)

        # Account & Position
        equity = data.get("equity")
        equity_str = (
            f"{equity:.4f}" if isinstance(equity, Decimal) else str(equity or "-")
        )
        table.add_row("Equity (Quote)", equity_str)
        pos = data.get("position")
        if pos and isinstance(pos, dict):
            pos_side = pos.get("side", "None")
            pos_size = Decimal(str(pos.get("contracts", "0")))
            pos_entry = Decimal(str(pos.get("entryPrice", "0")))
            pos_pnl = Decimal(str(pos.get("unrealizedPnl", "0")))
            color = (
                "green"
                if pos_side == "long"
                else "red"
                if pos_side == "short"
                else "white"
            )
            table.add_row("Position Side", f"[bold {color}]{pos_side.upper()}[/]")
            table.add_row("Position Size", f"{pos_size:.{8}f}")
            table.add_row("Entry Price", f"{pos_entry:.{8}f}")
            table.add_row("Unrealized PnL", f"{pos_pnl:.{8}f}")
        else:
            table.add_row("Position", "[grey50]None[/]")

        # Indicators (Format Decimals nicely)
        def format_ind(value):
            if isinstance(value, Decimal):
                return f"{value:.{6}f}"  # Adjust precision as needed
            return str(value or "-")

        table.add_row("Fast EMA", format_ind(data.get("fast_ema")))
        table.add_row("Slow EMA", format_ind(data.get("slow_ema")))
        table.add_row("Trend EMA", format_ind(data.get("trend_ema")))
        table.add_row("Stoch %K", format_ind(data.get("stoch_k")))
        table.add_row("Stoch %D", format_ind(data.get("stoch_d")))
        table.add_row(f"ATR ({self.config.atr_period})", format_ind(data.get("atr")))

        # Strategy Signal
        signal_reason = data.get("signal_reason", "Waiting...")
        # Apply color based on signal words
        color = (
            "green"
            if "Long Entry" in signal_reason
            else "red"
            if "Short Entry" in signal_reason
            else "yellow"
            if "Exit" in signal_reason
            else "white"
        )
        table.add_row("Last Signal", f"[{color}]{signal_reason}[/]")

        # Bot Status
        table.add_row("Last Cycle Time", str(data.get("last_cycle_time", "-")))
        table.add_row("Cycle Count", str(data.get("cycle_count", "-")))

        return table


# --- Health Check ---
async def run_health_check(port: int, status_dict_ref: Dict):
    """Runs a simple HTTP server to report bot status."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading

    # Use a mutable type (like a list containing the dict)
    # or pass a function to get the status to ensure the handler
    # always gets the latest status. Let's use the shared dict directly.

    class HealthCheckHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                try:
                    # Access the shared status dictionary
                    status_code = 200 if status_dict_ref.get("running", False) else 503
                    self.send_response(status_code)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    # Add timestamp to status dynamically
                    response_data = status_dict_ref.copy()
                    response_data["server_time"] = datetime.now().isoformat()
                    self.wfile.write(json.dumps(response_data).encode("utf-8"))
                except Exception as e:
                    # Fallback error response
                    self.send_response(500)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps({"status": "error", "message": str(e)}).encode(
                            "utf-8"
                        )
                    )
            else:
                self.send_response(404)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"status": "error", "message": "Not Found"}).encode(
                        "utf-8"
                    )
                )

        # Suppress standard request logging to keep console clean
        def log_message(self, format, *args):
            return

    def start_server():
        try:
            with HTTPServer(("0.0.0.0", port), HealthCheckHandler) as server:
                logger.info(
                    f"Health check server started on http://0.0.0.0:{port}/health"
                )
                server.serve_forever()
        except OSError as e:
            logger.critical(
                f"{Fore.RED}Could not start health check server on port {port}: {e}. Port might be in use."
            )
            # Don't exit the whole bot, just disable health check
        except Exception as e:
            logger.error(f"{Fore.RED}Health check server failed: {e}", exc_info=True)

    # Run the server in a separate thread so it doesn't block asyncio loop
    health_thread = threading.Thread(target=start_server, daemon=True)
    health_thread.start()
    # No await needed here as it runs in a separate thread


# --- Main Bot ---
class TradingBot:
    """
    Orchestrates the trading bot's components: configuration, exchange interaction,
    indicator calculation, strategy execution, journaling, and status display.
    """

    _instance = None  # Singleton pattern to ensure single bot instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TradingBot, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_file: Optional[str] = None):
        # Prevent re-initialization if already initialized (Singleton)
        if self._initialized:
            return

        print(
            f"{Fore.GREEN}{Style.BRIGHT}--- Initializing Pyrmethus Trading Bot v4.0 (Enhanced) ---{Style.RESET_ALL}"
        )
        self.config = TradingConfig(config_file)  # Handles its own logging
        self.exchange = BybitExchange(self.config)
        self.journal = JournalManager(
            self.config.journal_file_path, self.config.enable_journaling
        )
        self.indicator_calculator = IndicatorCalculator(self.config)
        self.strategy = EMAStochasticStrategy(self.config)
        self.trade_executor = TradeExecutor(self.config, self.exchange, self.journal)
        self.backtester = Backtester(
            self.config, self.indicator_calculator, self.strategy
        )
        self.dashboard = Dashboard(self.config)
        self.shutdown_requested = False
        self.config.shutdown_requested = False  # Share shutdown flag
        self.status = {
            "running": False,
            "last_cycle_time": None,
            "cycle_count": 0,
            "symbol": self.config.symbol,
        }
        self.cycle_count = 0
        self._initialized = True  # Mark as initialized
        logger.info("Trading bot components initialized.")

    async def initialize(self):
        """Initializes the exchange connection and dashboard."""
        logger.info("Performing bot initialization...")
        try:
            await (
                self.exchange.initialize()
            )  # Handles its own logging and exit on failure
            logger.info(f"{Fore.GREEN}Exchange connection established.")
            self.dashboard.start()  # Start CLI dashboard
            # Start health check server (runs in background thread)
            # No await needed here because it starts a thread
            asyncio.create_task(
                run_health_check(self.config.health_check_port, self.status)
            )
            self.status["running"] = True
            logger.info("Bot initialization complete.")
        except Exception as e:
            logger.critical(f"{Fore.RED}Bot initialization failed: {e}", exc_info=True)
            await self.shutdown(graceful=False)  # Attempt shutdown even on init error
            sys.exit(1)

    async def trading_cycle(self):
        """Executes a single trading cycle."""
        self.cycle_count += 1
        start_time = time.monotonic()
        logger.info(
            f"{Fore.MAGENTA}--- Cycle {self.cycle_count} Started ---{Style.RESET_ALL}"
        )
        self.status["cycle_count"] = self.cycle_count

        try:
            # 1. Fetch Data
            logger.debug("Fetching OHLCV data...")
            df_ohlcv = await self.exchange.fetch_ohlcv(
                self.config.symbol, self.config.interval, self.config.ohlcv_limit
            )
            if not df_ohlcv:
                logger.warning("Failed to fetch primary OHLCV data, skipping cycle.")
                return  # Skip cycle if data fetch fails

            df = pd.DataFrame(
                df_ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            # Convert columns to appropriate types for calculations
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col])

            # Fetch MTF data (optional)
            mtf_df = None
            mtf_indicators = None
            if self.config.multi_timeframe_interval:
                logger.debug(
                    f"Fetching MTF OHLCV data ({self.config.multi_timeframe_interval})..."
                )
                mtf_ohlcv = await self.exchange.fetch_ohlcv(
                    self.config.symbol,
                    self.config.multi_timeframe_interval,
                    self.config.ohlcv_limit,
                )
                if mtf_ohlcv:
                    mtf_df = pd.DataFrame(
                        mtf_ohlcv,
                        columns=["timestamp", "open", "high", "low", "close", "volume"],
                    )
                    mtf_df["timestamp"] = pd.to_datetime(mtf_df["timestamp"], unit="ms")
                    mtf_df.set_index("timestamp", inplace=True)
                    for col in ["open", "high", "low", "close", "volume"]:
                        mtf_df[col] = pd.to_numeric(mtf_df[col])
                else:
                    logger.warning("Failed to fetch MTF OHLCV data.")

            # 2. Calculate Indicators
            logger.debug("Calculating indicators...")
            indicators = self.indicator_calculator.calculate_indicators(df)
            if mtf_df is not None:
                mtf_indicators = self.indicator_calculator.calculate_indicators(
                    mtf_df
                )  # Assuming same calc method

            if not indicators:
                logger.warning(
                    "Indicator calculation failed, skipping strategy execution."
                )
                return

            # 3. Update State (Balance & Position)
            logger.debug("Fetching balance and position...")
            _, equity = await self.exchange.fetch_balance()
            await (
                self.trade_executor.update_position_state()
            )  # Fetches position into trade_executor.current_position
            current_position_state = self.trade_executor.current_position

            if equity is None:
                logger.error(
                    "Failed to fetch equity, cannot proceed with signal generation."
                )
                return

            # 4. Generate Signals
            logger.debug("Generating trading signals...")
            signals = self.strategy.generate_signals(
                df, indicators, current_position_state, equity, mtf_indicators
            )

            # 5. Execute Trades
            logger.debug("Executing trades based on signals...")
            if signals["entry_long"]:
                await self.trade_executor.execute_entry("buy", indicators)
            elif signals["entry_short"]:
                await self.trade_executor.execute_entry("sell", indicators)
            elif signals["exit_long"]:
                await self.trade_executor.execute_exit(
                    "long", signals.get("reason", "Exit Signal")
                )
            elif signals["exit_short"]:
                await self.trade_executor.execute_exit(
                    "short", signals.get("reason", "Exit Signal")
                )
            else:
                logger.debug("No entry or exit signals generated.")

            # 6. Update Dashboard & Status
            logger.debug("Updating dashboard...")
            cycle_time = datetime.now()
            self.status["last_cycle_time"] = cycle_time.isoformat()
            self.status["equity"] = (
                f"{equity:.4f}" if equity else "N/A"
            )  # Update status dict for health check
            dashboard_data = {
                "symbol": self.config.symbol,
                "price": Decimal(str(df["close"].iloc[-1])) if not df.empty else None,
                "equity": equity,
                "position": current_position_state,  # Pass the fetched position object
                "fast_ema": indicators.get("fast_ema"),
                "slow_ema": indicators.get("slow_ema"),
                "trend_ema": indicators.get("trend_ema"),
                "stoch_k": indicators.get("stoch_k"),
                "stoch_d": indicators.get("stoch_d"),
                "atr": indicators.get("atr"),
                "signal_reason": signals.get("reason", "N/A"),
                "last_cycle_time": cycle_time.strftime("%Y-%m-%d %H:%M:%S"),
                "cycle_count": self.cycle_count,
            }
            self.dashboard.update(dashboard_data)

        except Exception as e:
            logger.error(
                f"{Fore.RED}!!! Unhandled exception in trading cycle {self.cycle_count}: {e}",
                exc_info=True,
            )
            # Consider adding a counter for consecutive errors to potentially halt the bot

        finally:
            end_time = time.monotonic()
            duration = end_time - start_time
            logger.info(
                f"{Fore.MAGENTA}--- Cycle {self.cycle_count} Finished ({duration:.2f}s) ---{Style.RESET_ALL}"
            )

    async def main_loop(self):
        """The main execution loop of the trading bot."""
        await self.initialize()  # Perform startup tasks

        if self.config.backtest_enabled:
            logger.info(
                f"{Style.BRIGHT}--- Starting Backtest Mode ---{Style.RESET_ALL}"
            )
            results = self.backtester.run_backtest(self.config.backtest_data_path)
            if results:
                logger.info(f"{Style.BRIGHT}--- Backtest Complete ---{Style.RESET_ALL}")
                # Results are logged within the backtester method
            else:
                logger.error(f"{Fore.RED}Backtest failed to produce results.")
            await self.shutdown(graceful=True)  # Shutdown after backtest
            return  # Exit after backtest

        logger.info(
            f"{Style.BRIGHT}--- Starting Live Trading Mode ---{Style.RESET_ALL}"
        )
        logger.info(
            f"Trading Symbol: {self.config.symbol}, Interval: {self.config.interval}"
        )
        logger.info(f"Loop Interval: {self.config.loop_sleep_seconds} seconds")
        await self.trade_executor.send_termux_sms(
            f"🚀 Bot Started: {self.config.symbol} ({self.config.interval})"
        )

        while not self.shutdown_requested:
            await self.trading_cycle()

            # Sleep until the next cycle
            logger.debug(f"Sleeping for {self.config.loop_sleep_seconds} seconds...")
            try:
                # Use asyncio.sleep for non-blocking wait
                await asyncio.sleep(self.config.loop_sleep_seconds)
            except asyncio.CancelledError:
                logger.warning("Sleep interrupted, likely due to shutdown request.")
                break  # Exit loop if sleep is cancelled

        logger.info("Main loop exited.")
        await self.shutdown(graceful=True)  # Ensure final shutdown steps run

    async def shutdown(self, signum=None, frame=None, graceful=True):
        """Handles graceful shutdown of the bot."""
        if self.shutdown_requested and graceful:  # Avoid duplicate shutdown calls
            return

        self.shutdown_requested = True
        self.config.shutdown_requested = True  # Ensure flag is set in config too
        signal_name = f" (Signal: {signal.Signals(signum).name})" if signum else ""
        logger.warning(
            f"{Fore.YELLOW}{Style.BRIGHT}--- Initiating Shutdown{signal_name}... ---{Style.RESET_ALL}"
        )
        self.status["running"] = False

        # Stop components
        self.dashboard.stop()
        # Health check thread is daemon, should exit automatically, but can add explicit stop if needed

        # Close exchange connection
        if self.exchange:
            await self.exchange.close()

        # Final state saving (optional)
        try:
            if self.trade_executor.performance["equity_history"]:
                final_equity = self.trade_executor.performance["equity_history"][-1]
                state = {"last_equity": str(final_equity)}
                with open("bot_final_state.json", "w") as f:
                    json.dump(state, f)
                logger.info(
                    f"Saved final state (Equity: {final_equity}) to bot_final_state.json"
                )
            # Log final performance summary
            logger.info("--- Final Performance ---")
            logger.info(f"Total Trades: {self.trade_executor.performance['trades']}")
            logger.info(
                f"Wins: {self.trade_executor.performance['wins']}, Losses: {self.trade_executor.performance['losses']}"
            )
            logger.info(f"Total PnL: {self.trade_executor.performance['pnl']:.8f}")
            logger.info("-----------------------")
        except Exception as e:
            logger.error(f"Error during final state saving/logging: {e}")

        await self.trade_executor.send_termux_sms(
            f"🛑 Bot Shutdown: {self.config.symbol}{signal_name}"
        )
        logger.warning(
            f"{Fore.YELLOW}{Style.BRIGHT}--- Shutdown Complete ---{Style.RESET_ALL}"
        )

        # Required to ensure asyncio loop stops cleanly after shutdown tasks
        # Gather all pending tasks and cancel them
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        if tasks:
            logger.debug(f"Cancelling {len(tasks)} pending tasks...")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.debug("Pending tasks cancelled.")


# --- Signal Handling ---
def handle_signal(signum, frame, bot_instance):
    """Signal handler to initiate graceful shutdown."""
    if bot_instance and not bot_instance.shutdown_requested:
        logger.warning(
            f"Received signal {signal.Signals(signum).name}. Initiating shutdown..."
        )
        # Schedule the shutdown coroutine to run in the event loop
        asyncio.create_task(bot_instance.shutdown(signum, frame))
    else:
        logger.warning(
            f"Received signal {signal.Signals(signum).name}, but shutdown already in progress or bot not ready."
        )


# --- Entry Point ---
async def run_bot():
    """Sets up and runs the trading bot."""
    # Use a config file if specified, otherwise defaults apply
    config_file_path = "config.json"  # Or get from command line args: sys.argv[1] if len(sys.argv) > 1 else 'config.json'
    bot = TradingBot(config_file=config_file_path)

    # Set up signal handlers to call the bot's shutdown method
    signal.signal(signal.SIGINT, lambda s, f: handle_signal(s, f, bot))
    signal.signal(signal.SIGTERM, lambda s, f: handle_signal(s, f, bot))

    try:
        await bot.main_loop()
    except asyncio.CancelledError:
        logger.info("Main execution task cancelled.")
    except Exception as e:
        logger.critical(
            f"{Fore.RED}Critical unhandled error in main execution: {e}", exc_info=True
        )
        # Ensure shutdown is attempted even on critical error
        if bot and not bot.shutdown_requested:
            await bot.shutdown(graceful=False)
    finally:
        logger.info("Bot execution finished.")


if __name__ == "__main__":
    # Add random for backoff jitter
    import random

    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received directly in main. Exiting.")
    # Any final cleanup outside the asyncio loop if needed
    print(f"{Fore.BLUE}Bot process has terminated.{Style.RESET_ALL}")
