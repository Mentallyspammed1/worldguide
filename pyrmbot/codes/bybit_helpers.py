#!/usr/bin/env python
"""
Bybit Trading Module (v2.3 - Pyrmethus Enchanted)

A comprehensive trading bot for Bybit, optimized for Termux. Supports HTTP and WebSocket APIs,
advanced order management, real-time data streams, and SMS alerts.

Key Features:
- Robust configuration with interactive setup and validation
- Full pybit and ccxt integration for trading and market data
- Advanced WebSocket management with reconnection logic
- CLI for bot control, order placement, and diagnostics
- Colored logging with time-based rotation
- Termux-optimized SMS alerts with priority levels
- Comprehensive error handling and diagnostics

Usage:
- Run directly: `python bybit_helpers.py --help`
- Import in strategy: `from bybit_helpers import load_config, BybitHelper`
"""

import argparse
import functools
import logging
import logging.handlers
import os
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Dict, List, Literal, Optional

# --- Dependency Imports ---
try:
    import ccxt
    import websocket
    from pydantic import (
        BaseModel,
        Field,
        NonNegativeInt,
        PositiveFloat,
        PositiveInt,
        ValidationError,
        field_validator,
        model_validator,
    )
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pybit.unified_trading import HTTP, WebSocket
    from getpass import getpass
except ImportError as e:
    print(f"Error: Missing dependency: {e}")
    print(
        "Install dependencies: pip install ccxt pybit pydantic pydantic-settings colorama websocket-client"
    )
    sys.exit(1)

try:
    from colorama import Back, Fore, Style
    from colorama import init as colorama_init

    colorama_init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:

    class DummyColor:
        def __getattr__(self, name: str) -> str:
            return ""

    Fore = Style = Back = DummyColor()
    COLORAMA_AVAILABLE = False
    print("Warning: Colorama unavailable. Colored logging disabled.", file=sys.stderr)

# --- Custom Logging Setup ---
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


if not hasattr(logging.Logger, "success"):
    logging.Logger.success = log_success

LOG_LEVEL_COLORS: Dict[int, str] = {
    logging.DEBUG: Fore.CYAN + Style.DIM,
    logging.INFO: Fore.BLUE + Style.BRIGHT,
    SUCCESS_LEVEL: Fore.MAGENTA + Style.BRIGHT,
    logging.WARNING: Fore.YELLOW + Style.BRIGHT,
    logging.ERROR: Fore.RED + Style.BRIGHT,
    logging.CRITICAL: Back.RED + Fore.WHITE + Style.BRIGHT,
}


class ColoredConsoleFormatter(logging.Formatter):
    """Custom formatter for colored console logs with module names."""

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        style: str = "%",
        validate: bool = True,
    ):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style, validate=validate)
        force_no_color = os.getenv("NO_COLOR") or os.getenv(
            "BOT_LOGGING_CONFIG__FORCE_NO_COLOR"
        )
        self.use_colors = (
            COLORAMA_AVAILABLE
            and hasattr(sys.stdout, "isatty")
            and sys.stdout.isatty()
            and not force_no_color
        )

    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        color = LOG_LEVEL_COLORS.get(record.levelno, Fore.WHITE)
        record.levelname = (
            f"{color}{original_levelname:<8}{Style.RESET_ALL}"
            if self.use_colors
            else f"{original_levelname:<8}"
        )
        formatted_message = super().format(record)
        record.levelname = original_levelname
        return formatted_message


LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s [%(module)s:%(funcName)s:%(lineno)d] - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

console_formatter = ColoredConsoleFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
file_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)


def setup_logger(
    logger_name: str = "TradingBot",
    log_file: str | None = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_rotation_bytes: int = 5 * 1024 * 1024,
    log_backup_count: int = 5,
    log_rotation_time: bool = True,
    log_rotation_interval: int = 1,
) -> logging.Logger:
    """Set up a logger with console and optional rotating file handlers."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    console_h = logging.StreamHandler(sys.stdout)
    console_h.setLevel(console_level)
    console_h.setFormatter(console_formatter)
    logger.addHandler(console_h)

    if log_file:
        # Resolve relative paths
        if not os.path.isabs(log_file):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            log_file = os.path.join(script_dir, log_file)
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        termux_home = "/data/data/com.termux/files/home"
        if not log_file.startswith(termux_home) and not os.access(
            os.path.dirname(log_file), os.W_OK
        ):
            print(
                f"{Fore.YELLOW}Warning: Log path '{log_file}' may be inaccessible in Termux. "
                f"Consider using '{termux_home}/logs/'.{Style.RESET_ALL}"
            )
        if log_rotation_time:
            file_h = logging.handlers.TimedRotatingFileHandler(
                log_file,
                when="midnight",
                interval=log_rotation_interval,
                backupCount=log_backup_count,
                encoding="utf-8",
            )
        else:
            file_h = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=log_rotation_bytes,
                backupCount=log_backup_count,
                encoding="utf-8",
            )
        file_h.setLevel(file_level)
        file_h.setFormatter(file_formatter)
        logger.addHandler(file_h)

    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("websocket").setLevel(logging.WARNING)
    return logger


# --- Configuration Models ---
class APIConfig(BaseModel):
    api_key: str = Field(..., description="Bybit API Key")
    api_secret: str = Field(..., description="Bybit API Secret")
    testnet_mode: bool = Field(True, description="Use Testnet")
    symbol: str = Field("BTC/USDT:USDT", description="Primary trading symbol")
    retry_count: PositiveInt = Field(3, description="API retry attempts")
    retry_delay_seconds: PositiveFloat = Field(1.0, description="Delay between retries")
    order_book_fetch_limit: PositiveInt = Field(
        50, description="Order book depth to fetch"
    )
    shallow_ob_fetch_depth: PositiveInt = Field(
        10, description="Order book depth for analysis"
    )

    @field_validator("symbol")
    def validate_symbol(cls, v: str) -> str:
        if ":" not in v or "/" not in v:
            raise ValueError(
                "Symbol must be in 'BASE/QUOTE:SETTLE' format (e.g., 'BTC/USDT:USDT')"
            )
        return v.upper()


class StrategyConfig(BaseModel):
    timeframe: str = Field("5m", description="Kline timeframe (e.g., 1m, 5m, 1h)")
    risk_per_trade: PositiveFloat = Field(
        0.01, le=1.0, description="Risk per trade as fraction of balance"
    )
    leverage: PositiveInt = Field(10, description="Default leverage")
    default_position_mode: Literal["one-way", "hedge"] = Field(
        "one-way", description="Position mode"
    )
    loop_delay_seconds: PositiveFloat = Field(60.0, description="Strategy loop delay")

    @field_validator("timeframe")
    def validate_timeframe(cls, v: str) -> str:
        valid_timeframes = [
            "1m",
            "3m",
            "5m",
            "15m",
            "30m",
            "1h",
            "2h",
            "4h",
            "6h",
            "12h",
            "1d",
        ]
        if v not in valid_timeframes:
            raise ValueError(f"Timeframe must be one of: {', '.join(valid_timeframes)}")
        return v


class SMSConfig(BaseModel):
    enable_sms_alerts: bool = Field(False, description="Enable SMS alerts")
    sms_recipient_number: str | None = Field(
        None, pattern=r"^\+?[1-9]\d{1,14}$", description="Recipient phone number"
    )
    sms_cooldown_seconds: PositiveInt = Field(
        60, ge=10, description="Cooldown for non-critical SMS"
    )
    critical_sms_cooldown_seconds: PositiveInt = Field(
        300, ge=10, description="Cooldown for critical SMS"
    )

    @model_validator(mode="after")
    def check_sms_details(self) -> "SMSConfig":
        if self.enable_sms_alerts and not self.sms_recipient_number:
            raise ValueError("SMS enabled but no recipient number provided")
        return self


class LoggingConfig(BaseModel):
    log_file: str | None = Field("logs/trading_bot.log", description="Log file path")
    console_level: str = Field("INFO", description="Console log level")
    file_level: str = Field("DEBUG", description="File log level")
    log_rotation_bytes: PositiveInt = Field(
        5 * 1024 * 1024, description="Max log file size"
    )
    log_backup_count: NonNegativeInt = Field(5, description="Number of backup logs")
    log_rotation_time: bool = Field(True, description="Use time-based rotation")
    log_rotation_interval: PositiveInt = Field(
        1, description="Rotation interval in days"
    )

    @field_validator("console_level", "file_level")
    def validate_log_level(cls, v: str) -> str:
        levels = ["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
        v = v.upper()
        if v not in levels:
            raise ValueError(f"Log level must be one of: {', '.join(levels)}")
        return v

    @field_validator("log_file")
    def validate_log_file(cls, v: str | None) -> str | None:
        if not v or v.strip() == "":
            return None
        if any(char in v for char in ["<", ">", ":", '"', "|", "?", "*"]):
            raise ValueError(f"Log file path '{v}' contains invalid characters")
        return v.strip()


class AppConfig(BaseSettings):
    CONFIG_VERSION: str = Field("2.3", description="Configuration version")
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="BOT_", case_sensitive=False
    )
    api_config: APIConfig = Field(default_factory=APIConfig)
    sms_config: SMSConfig = Field(default_factory=SMSConfig)
    strategy_config: StrategyConfig = Field(default_factory=StrategyConfig)
    logging_config: LoggingConfig = Field(default_factory=LoggingConfig)


# --- Configuration Management ---
def generate_default_env_file(env_path: str) -> bool:
    """Generate a default .env file with example configuration."""
    default_env_content = """# Bybit Trading Bot Configuration (v2.3)
# Environment variables (e.g., BOT_API_CONFIG__API_KEY) override these settings.
# Secure this file: chmod 600 .env

# --- API Configuration ---
BOT_API_CONFIG__API_KEY=YOUR_API_KEY
BOT_API_CONFIG__API_SECRET=YOUR_API_SECRET
BOT_API_CONFIG__TESTNET_MODE=True
BOT_API_CONFIG__SYMBOL=BTC/USDT:USDT
BOT_API_CONFIG__RETRY_COUNT=3
BOT_API_CONFIG__RETRY_DELAY_SECONDS=1.0
BOT_API_CONFIG__ORDER_BOOK_FETCH_LIMIT=50
BOT_API_CONFIG__SHALLOW_OB_FETCH_DEPTH=10

# --- Strategy Configuration ---
BOT_STRATEGY_CONFIG__TIMEFRAME=5m
BOT_STRATEGY_CONFIG__RISK_PER_TRADE=0.01
BOT_STRATEGY_CONFIG__LEVERAGE=10
BOT_STRATEGY_CONFIG__DEFAULT_POSITION_MODE=one-way
BOT_STRATEGY_CONFIG__LOOP_DELAY_SECONDS=60.0

# --- Termux SMS Configuration ---
BOT_SMS_CONFIG__ENABLE_SMS_ALERTS=False
BOT_SMS_CONFIG__SMS_RECIPIENT_NUMBER=+1234567890
BOT_SMS_CONFIG__SMS_COOLDOWN_SECONDS=60
BOT_SMS_CONFIG__CRITICAL_SMS_COOLDOWN_SECONDS=300

# --- Logging Configuration ---
BOT_LOGGING_CONFIG__LOG_FILE=logs/trading_bot.log
BOT_LOGGING_CONFIG__CONSOLE_LEVEL=INFO
BOT_LOGGING_CONFIG__FILE_LEVEL=DEBUG
BOT_LOGGING_CONFIG__LOG_ROTATION_BYTES=5242880
BOT_LOGGING_CONFIG__LOG_BACKUP_COUNT=5
BOT_LOGGING_CONFIG__LOG_ROTATION_TIME=True
BOT_LOGGING_CONFIG__LOG_ROTATION_INTERVAL=1
"""
    os.makedirs(os.path.dirname(env_path) or ".", exist_ok=True)
    with open(env_path, "w") as f:
        f.write(default_env_content)
    print(f"{Fore.GREEN}Default .env file created at: {env_path}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Secure the file: chmod 600 {env_path}{Style.RESET_ALL}")
    return True


def interactive_setup(env_path: str) -> bool:
    """Guide the user through interactive configuration setup."""
    print(f"\n{Fore.MAGENTA}=== Interactive Configuration Setup ==={Style.RESET_ALL}")

    def prompt_input(
        prompt: str,
        default: str = "",
        secret: bool = False,
        validator: Callable[[str], str] | None = None,
    ) -> str:
        while True:
            input_prompt = f"{Fore.BLUE}{prompt}{Style.RESET_ALL}"
            if default:
                input_prompt += f" [{Fore.CYAN}{default}{Style.RESET_ALL}]"
            input_prompt += ": "
            value = (
                getpass(input_prompt).strip() if secret else input(input_prompt).strip()
            )
            value = value or default
            if validator:
                try:
                    return validator(value)
                except ValueError as e:
                    print(f"{Fore.RED}Invalid input: {e}{Style.RESET_ALL}")
            else:
                return value

    def validate_symbol(v: str) -> str:
        if not v or ":" not in v or "/" not in v:
            raise ValueError(
                "Symbol must be in 'BASE/QUOTE:SETTLE' format (e.g., 'BTC/USDT:USDT')"
            )
        return v.upper()

    def validate_leverage(v: str) -> str:
        if not v.isdigit() or int(v) < 1:
            raise ValueError("Leverage must be a positive integer")
        return v

    def validate_timeframe(v: str) -> str:
        valid_timeframes = [
            "1m",
            "3m",
            "5m",
            "15m",
            "30m",
            "1h",
            "2h",
            "4h",
            "6h",
            "12h",
            "1d",
        ]
        if v not in valid_timeframes:
            raise ValueError(f"Timeframe must be one of: {', '.join(valid_timeframes)}")
        return v

    def validate_phone(v: str) -> str:
        if v and not v.startswith("+") or not v[1:].isdigit():
            raise ValueError("Phone number must be in E.164 format (e.g., +1234567890)")
        return v

    api_key = prompt_input("Enter Bybit API Key", secret=True)
    api_secret = prompt_input("Enter Bybit API Secret", secret=True)
    symbol = prompt_input(
        "Enter trading symbol", "BTC/USDT:USDT", validator=validate_symbol
    )
    testnet = prompt_input("Use Testnet? (yes/no)", "yes").lower()
    timeframe = prompt_input(
        "Enter strategy timeframe (e.g., 5m, 1h)", "5m", validator=validate_timeframe
    )
    leverage = prompt_input("Enter desired leverage", "10", validator=validate_leverage)
    risk_per_trade = prompt_input("Enter risk per trade (e.g., 0.01 for 1%)", "0.01")
    position_mode = prompt_input(
        "Enter position mode (one-way/hedge)", "one-way"
    ).lower()
    sms_enable = prompt_input("Enable SMS alerts? (yes/no)", "no").lower()
    sms_number = (
        prompt_input(
            "Enter SMS recipient number (e.g., +1234567890)",
            "",
            validator=validate_phone,
        )
        if sms_enable.startswith("y")
        else ""
    )
    log_file = prompt_input("Enter log file path", "logs/trading_bot.log")

    env_content = f"""# Bybit Trading Bot Configuration (v2.3)
BOT_API_CONFIG__API_KEY={api_key}
BOT_API_CONFIG__API_SECRET={api_secret}
BOT_API_CONFIG__TESTNET_MODE={"True" if testnet.startswith("y") else "False"}
BOT_API_CONFIG__SYMBOL={symbol}
BOT_API_CONFIG__RETRY_COUNT=3
BOT_API_CONFIG__RETRY_DELAY_SECONDS=1.0
BOT_API_CONFIG__ORDER_BOOK_FETCH_LIMIT=50
BOT_API_CONFIG__SHALLOW_OB_FETCH_DEPTH=10
BOT_STRATEGY_CONFIG__TIMEFRAME={timeframe}
BOT_STRATEGY_CONFIG__RISK_PER_TRADE={risk_per_trade}
BOT_STRATEGY_CONFIG__LEVERAGE={leverage}
BOT_STRATEGY_CONFIG__DEFAULT_POSITION_MODE={"one-way" if position_mode.startswith("o") else "hedge"}
BOT_STRATEGY_CONFIG__LOOP_DELAY_SECONDS=60.0
BOT_SMS_CONFIG__ENABLE_SMS_ALERTS={"True" if sms_enable.startswith("y") else "False"}
BOT_SMS_CONFIG__SMS_RECIPIENT_NUMBER={sms_number}
BOT_SMS_CONFIG__SMS_COOLDOWN_SECONDS=60
BOT_SMS_CONFIG__CRITICAL_SMS_COOLDOWN_SECONDS=300
BOT_LOGGING_CONFIG__LOG_FILE={log_file}
BOT_LOGGING_CONFIG__CONSOLE_LEVEL=INFO
BOT_LOGGING_CONFIG__FILE_LEVEL=DEBUG
BOT_LOGGING_CONFIG__LOG_ROTATION_BYTES=5242880
BOT_LOGGING_CONFIG__LOG_BACKUP_COUNT=5
BOT_LOGGING_CONFIG__LOG_ROTATION_TIME=True
BOT_LOGGING_CONFIG__LOG_ROTATION_INTERVAL=1
"""
    with open(env_path, "w") as f:
        f.write(env_content)
    print(f"{Fore.GREEN}Configuration saved to {env_path}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Secure the file: chmod 600 {env_path}{Style.RESET_ALL}")
    return True


def load_config() -> AppConfig:
    """Load and validate configuration from .env file."""
    env_file_path = ".env"
    if not os.path.exists(env_file_path):
        print(
            f"{Fore.YELLOW}No .env file found. Generating default...{Style.RESET_ALL}"
        )
        generate_default_env_file(env_file_path)
        interactive_setup(env_file_path)
    try:
        config = AppConfig(_env_file=env_file_path)
        expected_version = AppConfig.model_fields["CONFIG_VERSION"].default
        if config.CONFIG_VERSION != expected_version:
            print(
                f"{Fore.YELLOW}Warning: Configuration version ({config.CONFIG_VERSION}) differs from expected ({expected_version}). "
                f"Regenerate .env by deleting it and running the script, or compare with the template in --help.{Style.RESET_ALL}"
            )
        return config
    except ValidationError as e:
        print(f"{Fore.RED}Configuration error: {e}{Style.RESET_ALL}")
        print(
            f"{Fore.YELLOW}Run with --setup to regenerate .env or check {env_file_path}{Style.RESET_ALL}"
        )
        sys.exit(1)


# --- Bybit Helper Class ---
class BybitHelper:
    """Main class for interacting with Bybit API."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = setup_logger(
            log_file=config.logging_config.log_file,
            console_level=getattr(logging, config.logging_config.console_level),
            file_level=getattr(logging, config.logging_config.file_level),
            log_rotation_bytes=config.logging_config.log_rotation_bytes,
            log_backup_count=config.logging_config.log_backup_count,
            log_rotation_time=config.logging_config.log_rotation_time,
            log_rotation_interval=config.logging_config.log_rotation_interval,
        )
        self._initialize_ccxt()
        self._initialize_pybit()
        self._last_sms_time: Dict[str, float] = {"normal": 0, "critical": 0}
        self._market_cache: Dict[str, Any] = {}
        self._ws_callbacks: Dict[str, Callable[[Dict[str, Any]], None]] = {}
        self._running = False
        # Check Termux API
        if config.sms_config.enable_sms_alerts:
            try:
                subprocess.run(
                    ["which", "termux-sms-send"], capture_output=True, check=True
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.logger.warning(
                    f"{Fore.YELLOW}Termux SMS alerts enabled, but 'termux-sms-send' not found. "
                    f"Install 'termux-api' package and ensure Termux:API app is running.{Style.RESET_ALL}"
                )

    def _initialize_ccxt(self) -> None:
        """Initialize CCXT exchange instance."""
        self.exchange = ccxt.bybit(
            {
                "apiKey": self.config.api_config.api_key,
                "secret": self.config.api_secret,
                "enableRateLimit": True,
            }
        )
        if self.config.api_config.testnet_mode:
            self.exchange.set_sandbox_mode(True)
        self.exchange.load_markets()
        if self.config.api_config.symbol not in self.exchange.markets:
            base_quote = self.config.api_config.symbol.split(":")[0]
            raise ValueError(
                f"Symbol '{self.config.api_config.symbol}' not found in CCXT markets. "
                f"Base '{base_quote}' {'exists' if base_quote in self.exchange.markets else 'not found'}. "
                "Verify symbol format and testnet mode."
            )
        self.logger.success("CCXT conduit forged and market data loaded.")

    def _initialize_pybit(self) -> None:
        """Initialize pybit HTTP session."""
        self.session = HTTP(
            api_key=self.config.api_config.api_key,
            api_secret=self.config.api_secret,
            testnet=self.config.api_config.testnet_mode,
        )
        self.logger.success("Pybit HTTP session established.")

    def retry_api_call(
        self,
        max_retries: Optional[int] = None,
        initial_delay: Optional[float] = None,
        error_message_prefix: str = "API Call Failed",
    ):
        """Decorator for retrying API calls with exponential backoff."""

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                effective_max_retries = (
                    max_retries or self.config.api_config.retry_count
                )
                effective_base_delay = (
                    initial_delay or self.config.api_config.retry_delay_seconds
                )
                attempt = 0
                while attempt <= effective_max_retries:
                    try:
                        result = func(*args, **kwargs)
                        if isinstance(result, dict) and result.get("retCode", 0) != 0:
                            ret_code = result.get("retCode")
                            ret_msg = result.get("retMsg", "Unknown Bybit Error")
                            retryable_codes = [
                                10001,
                                10002,
                                130035,
                            ]  # Rate limit, timeout, system error
                            if ret_code in retryable_codes:
                                raise ccxt.NetworkError(
                                    f"Bybit Error {ret_code}: {ret_msg}"
                                )
                            self.logger.error(
                                f"{func.__name__} failed with Bybit error {ret_code}: {ret_msg}"
                            )
                            return result
                        if attempt > 0:
                            self.logger.success(
                                f"{Style.BRIGHT}{func.__name__}{Style.NORMAL} succeeded after {attempt} retries."
                            )
                        return result
                    except (
                        ccxt.RateLimitExceeded,
                        ccxt.NetworkError,
                        ccxt.ExchangeNotAvailable,
                        ccxt.RequestTimeout,
                        websocket.WebSocketTimeoutException,
                        ConnectionResetError,
                    ) as e:
                        attempt += 1
                        if attempt > effective_max_retries:
                            self.logger.critical(
                                f"{error_message_prefix}: {func.__name__} failed after {effective_max_retries} retries: {e}",
                                exc_info=True,
                            )
                            self.send_sms_alert(
                                f"CRITICAL: {func.__name__} failed after {effective_max_retries} retries: {e}",
                                priority="critical",
                            )
                            raise
                        delay = effective_base_delay * (2 ** (attempt - 1))
                        self.logger.warning(
                            f"{error_message_prefix}: {func.__name__} failed: {e}. Retrying in {delay:.2f}s (Attempt {attempt}/{effective_max_retries})"
                        )
                        time.sleep(delay)
                    except Exception as e:
                        self.logger.critical(
                            f"{error_message_prefix}: Unexpected error in {func.__name__}: {e}",
                            exc_info=True,
                        )
                        self.send_sms_alert(
                            f"CRITICAL: Unexpected error in {func.__name__}: {e}",
                            priority="critical",
                        )
                        raise

            return wrapper

        return decorator

    def send_sms_alert(
        self, message: str, priority: Literal["normal", "critical"] = "normal"
    ) -> bool:
        """Send an SMS alert with cooldown based on priority."""
        if (
            not self.config.sms_config.enable_sms_alerts
            or not self.config.sms_config.sms_recipient_number
        ):
            self.logger.debug("SMS alerts disabled or no recipient, skipping send.")
            return False
        cooldown = (
            self.config.sms_config.critical_sms_cooldown_seconds
            if priority == "critical"
            else self.config.sms_config.sms_cooldown_seconds
        )
        current_time = time.time()
        if current_time - self._last_sms_time[priority] < cooldown:
            self.logger.debug(
                f"SMS alert ({priority}) skipped due to cooldown (next allowed in {cooldown - (current_time - self._last_sms_time[priority]):.1f}s)"
            )
            return False
        try:
            masked_number = f"****{self.config.sms_config.sms_recipient_number[-4:]}"
            subprocess.run(
                [
                    "termux-sms-send",
                    "-n",
                    self.config.sms_config.sms_recipient_number,
                    message,
                ],
                check=True,
                timeout=30,
            )
            self._last_sms_time[priority] = current_time
            self.logger.success(
                f"{priority.capitalize()} SMS sent to {masked_number}: {message}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to send {priority} SMS: {e}")
            return False

    @retry_api_call()
    def get_position_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Fetch position information for a symbol."""
        symbol = symbol or self.config.api_config.symbol
        return self.session.get_positions(category="linear", symbol=symbol)

    @retry_api_call()
    def get_wallet_balance(self) -> Dict[str, Any]:
        """Fetch wallet balance."""
        return self.session.get_wallet_balance(accountType="UNIFIED")

    @retry_api_call()
    def get_open_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Fetch open orders for a symbol."""
        symbol = symbol or self.config.api_config.symbol
        return self.session.get_open_orders(category="linear", symbol=symbol)

    @retry_api_call()
    def place_market_order(
        self, side: Literal["Buy", "Sell"], qty: float, symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """Place a market order."""
        symbol = symbol or self.config.api_config.symbol
        qty = self.format_amount(qty, symbol)
        try:
            result = self.session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(qty),
                positionIdx=0
                if self.config.strategy_config.default_position_mode == "one-way"
                else (1 if side == "Buy" else 2),
            )
            self.logger.success(f"Market {side} order placed: {qty} {symbol}")
            self.send_sms_alert(f"Market {side} order placed: {qty} {symbol}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to place market {side} order: {e}")
            raise

    @retry_api_call()
    def set_trading_stop(
        self,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        symbol: Optional[str] = None,
        position_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Set take-profit and stop-loss for a position."""
        symbol = symbol or self.config.api_config.symbol
        position_idx = position_idx or (
            0 if self.config.strategy_config.default_position_mode == "one-way" else 1
        )
        params = {"category": "linear", "symbol": symbol, "positionIdx": position_idx}
        if take_profit:
            params["takeProfit"] = str(self.format_price(take_profit, symbol))
        if stop_loss:
            params["stopLoss"] = str(self.format_price(stop_loss, symbol))
        try:
            result = self.session.set_trading_stop(**params)
            self.logger.success(
                f"Set TP/SL for {symbol}: TP={take_profit}, SL={stop_loss}"
            )
            return result
        except Exception as e:
            self.logger.error(f"Failed to set TP/SL for {symbol}: {e}")
            raise

    @retry_api_call()
    def fetch_ohlcv(
        self,
        timeframe: Optional[str] = None,
        limit: int = 200,
        symbol: Optional[str] = None,
    ) -> List[List[float]]:
        """Fetch OHLCV data using CCXT."""
        symbol = symbol or self.config.api_config.symbol
        timeframe = timeframe or self.config.strategy_config.timeframe
        return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    @retry_api_call()
    def fetch_ticker(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Fetch ticker data using CCXT."""
        symbol = symbol or self.config.api_config.symbol
        return self.exchange.fetch_ticker(symbol)

    @retry_api_call()
    def fetch_balance(self) -> Dict[str, Any]:
        """Fetch account balance using CCXT."""
        return self.exchange.fetch_balance()

    @retry_api_call()
    def fetch_order_book(
        self, symbol: Optional[str] = None, limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Fetch order book using CCXT."""
        symbol = symbol or self.config.api_config.symbol
        limit = limit or self.config.api_config.order_book_fetch_limit
        return self.exchange.fetch_order_book(symbol, limit=limit)

    def format_price(self, price: float, symbol: Optional[str] = None) -> float:
        """Format price to market precision."""
        symbol = symbol or self.config.api_config.symbol
        market = self.exchange.markets.get(symbol, {})
        market.get("precision", {}).get("price", 8)
        return float(self.exchange.price_to_precision(symbol, price))

    def format_amount(self, amount: float, symbol: Optional[str] = None) -> float:
        """Format quantity to market precision."""
        symbol = symbol or self.config.api_config.symbol
        market = self.exchange.markets.get(symbol, {})
        market.get("precision", {}).get("amount", 8)
        return float(self.exchange.amount_to_precision(symbol, amount))

    def analyze_order_book(
        self,
        symbol: Optional[str] = None,
        depth: Optional[int] = None,
        fetch_limit: Optional[int] = None,
    ) -> Dict[str, float]:
        """Analyze order book for liquidity and spread."""
        symbol = symbol or self.config.api_config.symbol
        analysis_depth = depth or self.config.api_config.shallow_ob_fetch_depth
        effective_fetch_limit = max(
            analysis_depth, fetch_limit or self.config.api_config.order_book_fetch_limit
        )
        effective_fetch_limit = min(effective_fetch_limit, analysis_depth * 2)
        order_book = self.fetch_order_book(symbol, effective_fetch_limit)
        bids = order_book["bids"][:analysis_depth]
        asks = order_book["asks"][:analysis_depth]
        bid_volume = sum(size for _, size in bids)
        ask_volume = sum(size for _, size in asks)
        spread = asks[0][0] - bids[0][0] if bids and asks else 0
        return {
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "spread": spread,
            "mid_price": (bids[0][0] + asks[0][0]) / 2 if bids and asks else 0,
        }

    def get_strategy_context(self) -> Dict[str, Any]:
        """Fetch data for strategy decisions."""
        try:
            ohlcv = self.fetch_ohlcv(limit=50)
            ticker = self.fetch_ticker()
            balance = self.fetch_balance()
            positions = self.get_position_info()
            order_book = self.analyze_order_book()
            sma = (
                sum(c[4] for c in ohlcv[-20:]) / 20 if ohlcv else 0
            )  # Simple Moving Average
            return {
                "ohlcv": ohlcv,
                "current_price": ticker.get("last", 0),
                "balance": balance.get("total", {}).get("USDT", 0),
                "positions": positions.get("result", {}).get("list", []),
                "order_book": order_book,
                "sma_20": sma,
            }
        except Exception as e:
            self.logger.error(f"Failed to get strategy context: {e}", exc_info=True)
            return {}

    @contextmanager
    def websocket_connection(
        self, channel_type: str = "linear", trace_logging: bool = False
    ):
        """Context manager for WebSocket connections."""
        ws = WebSocket(
            testnet=self.config.api_config.testnet_mode,
            channel_type=channel_type,
            api_key=self.config.api_config.api_key,
            api_secret=self.config.api_secret,
            trace_logging=trace_logging,
        )
        try:
            yield ws
        finally:
            ws.exit()

    def subscribe_to_stream(
        self,
        topics: List[str],
        callback: Callable[[Dict[str, Any]], None],
        channel_type: str = "linear",
        trace_logging: bool = False,
    ):
        """Subscribe to WebSocket streams with reconnection logic."""
        if not topics:
            self.logger.warning("WebSocket subscription requested with no topics.")
            return
        self._ws_callbacks["|".join(topics)] = callback
        is_private = any(
            topic.startswith(("order", "position", "wallet", "execution"))
            for topic in topics
        )
        effective_channel_type = "private" if is_private else channel_type
        max_reconnect_attempts = 5
        reconnect_delay = 5
        attempt = 0
        while self._running and attempt < max_reconnect_attempts:
            try:
                with self.websocket_connection(
                    channel_type=effective_channel_type, trace_logging=trace_logging
                ) as ws:
                    self.logger.debug("WebSocket connected. Subscribing to topics...")
                    ws.subscribe(topics, callback=callback)
                    self.logger.success(
                        f"Subscribed to WebSocket topics: {', '.join(topics)}"
                    )
                    while self._running:
                        time.sleep(1)
            except (ConnectionError, websocket.WebSocketException) as e:
                attempt += 1
                if attempt >= max_reconnect_attempts:
                    self.logger.critical(
                        f"Failed to maintain WebSocket subscription after {max_reconnect_attempts} attempts: {e}"
                    )
                    self.send_sms_alert(
                        f"CRITICAL: WebSocket failed for {', '.join(topics)}",
                        priority="critical",
                    )
                    break
                self.logger.warning(
                    f"WebSocket error: {e}. Reconnecting in {reconnect_delay}s (Attempt {attempt}/{max_reconnect_attempts})"
                )
                time.sleep(reconnect_delay)
            except KeyboardInterrupt:
                self.logger.info("WebSocket subscription interrupted by user.")
                break
            except Exception as e:
                self.logger.critical(
                    f"Unexpected error in WebSocket subscription: {e}", exc_info=True
                )
                self.send_sms_alert(
                    f"CRITICAL: WebSocket error for {', '.join(topics)}: {e}",
                    priority="critical",
                )
                break

    def diagnose_connection(self) -> bool:
        """Diagnose API connectivity and permissions."""
        healthy = True
        try:
            balance = self.get_wallet_balance()
            if balance.get("retCode") != 0:
                self.logger.error(
                    f"Balance Check: FAILED - {balance.get('retMsg', 'Unknown error')}"
                )
                healthy = False
            else:
                self.logger.success("Balance Check: OK")
        except Exception as e:
            self.logger.error(f"Balance Check: Error - {e}")
            healthy = False

        try:
            permissions = self.session.get_api_key_information()
            if permissions.get("retCode") == 0 and "result" in permissions:
                perms = permissions["result"].get("permissions", {})
                required = ["Order", "Position", "Trade"]
                missing = [
                    p
                    for p in required
                    if p not in perms.get("spot", [])
                    and p not in perms.get("contract", [])
                ]
                if missing:
                    self.logger.warning(
                        f"API Key Permissions: MISSING required permissions: {missing}"
                    )
                    healthy = False
                else:
                    self.logger.success("API Key Permissions: OK")
            else:
                self.logger.error(
                    f"API Key Permissions: FAILED - {permissions.get('retMsg', 'Unknown')}"
                )
                healthy = False
        except Exception as e:
            self.logger.error(f"API Key Permissions: Error - {e}")
            healthy = False

        return healthy

    def start(self):
        """Start the bot's main strategy loop."""
        self._running = True
        self.logger.info(
            f"{Fore.MAGENTA}Starting trading bot for {self.config.api_config.symbol}{Style.RESET_ALL}"
        )
        if not self.diagnose_connection():
            self.logger.critical("Connection diagnostics failed. Aborting startup.")
            self.send_sms_alert(
                "CRITICAL: Bot startup failed due to connection issues",
                priority="critical",
            )
            return

        # Subscribe to WebSocket streams
        def ticker_callback(data: Dict[str, Any]):
            price = data.get("data", {}).get("lastPrice")
            if price:
                self.logger.debug(
                    f"Ticker update: {self.config.api_config.symbol} @ {price}"
                )

        self.subscribe_to_stream(
            [f"tickers.{self.config.api_config.symbol}"], ticker_callback
        )
        # Main strategy loop
        while self._running:
            try:
                context = self.get_strategy_context()
                if not context:
                    self.logger.warning("Empty strategy context, skipping iteration.")
                    time.sleep(self.config.strategy_config.loop_delay_seconds)
                    continue
                price = context.get("current_price", 0)
                balance = context.get("balance", 0)
                sma = context.get("sma_20", 0)
                if price and balance and sma:
                    self.logger.info(f"Price: {price}, SMA: {sma}, Balance: {balance}")
                    if price < sma * 0.99:  # Buy if price is 1% below SMA
                        qty = (
                            balance * self.config.strategy_config.risk_per_trade
                        ) / price
                        self.place_market_order("Buy", qty)
                time.sleep(self.config.strategy_config.loop_delay_seconds)
            except Exception as e:
                self.logger.error(f"Strategy loop error: {e}", exc_info=True)
                self.send_sms_alert(f"Strategy loop error: {e}", priority="critical")
                time.sleep(60)

    def stop(self):
        """Stop the bot."""
        self._running = False
        self.logger.info(f"{Fore.MAGENTA}Stopping trading bot{Style.RESET_ALL}")


# --- CLI ---
def print_help():
    """Print detailed help message with usage and examples."""
    print(
        f"""
{Fore.MAGENTA}=== Bybit Trading Bot (v2.3) - Pyrmethus Enchanted ==={Style.RESET_ALL}

{Fore.CYAN}Description:{Style.RESET_ALL}
A Termux-optimized trading bot for Bybit, supporting HTTP and WebSocket APIs, advanced order management,
SMS alerts, and a CLI.

{Fore.CYAN}Usage:{Style.RESET_ALL}
  python {os.path.basename(__file__)} [options]

{Fore.CYAN}Options:{Style.RESET_ALL}
  --help, -h            Show this help message
  --setup               Run interactive configuration setup
  --diagnose            Run connection diagnostics
  --start               Start the bot
  --stop                Stop the bot
  --view-positions      View open positions
  --place-order SIDE QTY [SYMBOL]  Place a market order (e.g., --place-order Buy 0.01 BTC/USDT:USDT)
  --view-orders         View open orders
  --cancel-order ORDER_ID  Cancel an order by ID
  --view-balance        View wallet balance

{Fore.CYAN}Setup Instructions:{Style.RESET_ALL}
1. Install dependencies:
   {Fore.GREEN}pkg install python termux-api -y && pip install ccxt pybit pydantic pydantic-settings colorama websocket-client{Style.RESET_ALL}
2. Run with --setup to configure:
   {Fore.GREEN}python {os.path.basename(__file__)} --setup{Style.RESET_ALL}
3. Secure .env file:
   {Fore.GREEN}chmod 600 .env{Style.RESET_ALL}
4. Start the bot:
   {Fore.GREEN}python {os.path.basename(__file__)} --start{Style.RESET_ALL}

{Fore.CYAN}Example Strategy:{Style.RESET_ALL}
```python
from {os.path.splitext(os.path.basename(__file__))[0]} import load_config, BybitHelper
import time

config = load_config()
helper = BybitHelper(config)
helper.start()

# Custom strategy loop
while helper._running:
    try:
        context = helper.get_strategy_context()
        price = context.get('current_price')
        balance = context.get('balance')
        if price and balance:
            helper.logger.info(f"Price: {price}, Balance: {balance}")
            if price < context['sma_20'] * 0.99:
                qty = (balance * config.strategy_config.risk_per_trade) / price
                helper.place_market_order(side="Buy", qty=qty)
        time.sleep(config.strategy_config.loop_delay_seconds)
    except Exception as e:
        helper.logger.error(f"Strategy loop error: {e}")
        time.sleep(60)
```

{Fore.CYAN}Common Errors:{Style.RESET_ALL}
- {Fore.RED}Authentication Error{Style.RESET_ALL}: Verify API Key/Secret in .env. Check Bybit API permissions.
- {Fore.RED}Symbol Not Found{Style.RESET_ALL}: Ensure BOT_API_CONFIG__SYMBOL is correct (e.g., 'BTC/USDT:USDT') and matches testnet mode.
- {Fore.RED}Termux SMS Failed{Style.RESET_ALL}: Install 'termux-api', run Termux:API app, grant SMS permissions.
- {Fore.RED}Log File Error{Style.RESET_ALL}: Use paths under '/data/data/com.termux/files/home' or run 'termux-setup-storage'.

{Fore.CYAN}Testing:{Style.RESET_ALL}
Create `test_bybit.py`:
```python
import pytest
from bybit_trading_enchanted import load_config, BybitHelper

def test_config_loading():
    config = load_config()
    assert config.api_config.symbol == "BTC/USDT:USDT"
    assert config.api_config.testnet_mode is True

def test_helper_initialization():
    config = load_config()
    helper = BybitHelper(config)
    assert helper.session is not None
    assert helper.exchange is not None
```
Run tests: {Fore.GREEN}pytest test_bybit.py -v{Style.RESET_ALL}
Install pytest: {Fore.GREEN}pip install pytest{Style.RESET_ALL}

{Fore.CYAN}Performance Tip:{Style.RESET_ALL}
Use WebSocket subscriptions (e.g., helper.subscribe_to_stream) for real-time data to avoid HTTP rate limits.
"""
    )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Bybit Trading Bot CLI", add_help=False
    )
    parser.add_argument("--help", "-h", action="store_true", help="Show help message")
    parser.add_argument(
        "--setup", action="store_true", help="Run interactive configuration setup"
    )
    parser.add_argument(
        "--diagnose", action="store_true", help="Run connection diagnostics"
    )
    parser.add_argument("--start", action="store_true", help="Start the bot")
    parser.add_argument("--stop", action="store_true", help="Stop the bot")
    parser.add_argument(
        "--view-positions", action="store_true", help="View open positions"
    )
    parser.add_argument(
        "--place-order", nargs="+", help="Place a market order: SIDE QTY [SYMBOL]"
    )
    parser.add_argument("--view-orders", action="store_true", help="View open orders")
    parser.add_argument("--cancel-order", help="Cancel an order by ID")
    parser.add_argument(
        "--view-balance", action="store_true", help="View wallet balance"
    )
    args = parser.parse_args()

    if args.help:
        print_help()
        sys.exit(0)

    temp_logger = logging.getLogger("TempMain")
    temp_logger.setLevel(logging.INFO)
    temp_handler = logging.StreamHandler(sys.stdout)
    temp_handler.setFormatter(
        ColoredConsoleFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    )
    temp_logger.addHandler(temp_handler)

    try:
        config = load_config()
    except Exception as e:
        temp_logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    helper = BybitHelper(config)
    logger = helper.logger

    if args.setup:
        interactive_setup(".env")
        sys.exit(0)

    if args.diagnose:
        logger.info("Running connection diagnostics...")
        healthy = helper.diagnose_connection()
        logger.info(f"Diagnostics: {'Healthy' if healthy else 'Unhealthy'}")
        sys.exit(0 if healthy else 1)

    if args.start:
        logger.info("Starting bot...")
        helper.start()
        sys.exit(0)

    if args.stop:
        logger.info("Stopping bot...")
        helper.stop()
        sys.exit(0)

    if args.view_positions:
        logger.info("Fetching open positions...")
        positions = helper.get_position_info()
        for pos in positions.get("result", {}).get("list", []):
            print(
                f"Symbol: {pos['symbol']}, Side: {pos['side']}, Size: {pos['size']}, Entry: {pos['avgPrice']}"
            )
        sys.exit(0)

    if args.place_order:
        if len(args.place_order) < 2:
            logger.error("Usage: --place-order SIDE QTY [SYMBOL]")
            sys.exit(1)
        side, qty = args.place_order[0], args.place_order[1]
        symbol = (
            args.place_order[2]
            if len(args.place_order) > 2
            else config.api_config.symbol
        )
        try:
            qty = float(qty)
            if side not in ["Buy", "Sell"]:
                raise ValueError("Side must be 'Buy' or 'Sell'")
            result = helper.place_market_order(side, qty, symbol)
            print(f"Order placed: {result}")
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            sys.exit(1)
        sys.exit(0)

    if args.view_orders:
        logger.info("Fetching open orders...")
        orders = helper.get_open_orders()
        for order in orders.get("result", {}).get("list", []):
            print(
                f"Order ID: {order['orderId']}, Symbol: {order['symbol']}, Side: {order['side']}, Qty: {order['qty']}"
            )
        sys.exit(0)

    if args.cancel_order:
        logger.info(f"Cancelling order {args.cancel_order}...")
        try:
            result = helper.session.cancel_order(
                category="linear", orderId=args.cancel_order
            )
            print(f"Order cancelled: {result}")
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            sys.exit(1)
        sys.exit(0)

    if args.view_balance:
        logger.info("Fetching wallet balance...")
        balance = helper.get_wallet_balance()
        for coin in balance.get("result", {}).get("list", []):
            print(f"Coin: {coin['coin']}, Balance: {coin['equity']}")
        sys.exit(0)

    print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
