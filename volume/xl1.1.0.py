import json
import logging
import os
import re
import signal
import smtplib
import sys
import time
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, ROUND_UP, Decimal, InvalidOperation, getcontext
from email.mime.text import MIMEText
from logging.handlers import RotatingFileHandler
from typing import Any, TypedDict

# --- Timezone Handling ---
try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    try:
        ZoneInfo("America/Chicago")  # Test if tzdata is available
    except ZoneInfoNotFoundError:
        pass
except ImportError:
    # Basic UTC fallback implementation if zoneinfo is not available
    class ZoneInfo:
        _key = "UTC"

        def __init__(self, key: str) -> None:
            if key.upper() != "UTC":
                pass
            self._requested_key = key

        def __call__(self, dt: datetime | None = None) -> datetime | None:
            if dt is None: return None
            if dt.tzinfo is None: return dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)  # Ensure it's UTC

        def fromutc(self, dt: datetime) -> datetime:
            if not isinstance(dt, datetime): raise TypeError("fromutc() requires a datetime argument")
            if dt.tzinfo is None: raise ValueError("fromutc: naive datetime has no timezone")
            return dt.astimezone(UTC)

        def utcoffset(self, dt: datetime | None) -> timedelta: return timedelta(0)
        def dst(self, dt: datetime | None) -> timedelta: return timedelta(0)
        def tzname(self, dt: datetime | None) -> str: return "UTC"
        def __repr__(self) -> str: return f"ZoneInfo(key='{self._requested_key}') [Fallback: Always UTC]"
        def __str__(self) -> str: return self._key
    class ZoneInfoNotFoundError(Exception): pass

# --- Third-Party Library Imports ---
import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
from colorama import Fore, Style
from colorama import init as colorama_init
from dotenv import load_dotenv

# --- Initial Setup ---
getcontext().prec = 28  # Set Decimal precision
colorama_init(autoreset=True)  # Initialize Colorama
load_dotenv()  # Load environment variables from .env file

# --- Constants ---
BOT_VERSION = "1.5.0"
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    sys.exit(1)

CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
DEFAULT_TIMEZONE_STR = "America/Chicago"
TIMEZONE_STR = os.getenv("TIMEZONE", DEFAULT_TIMEZONE_STR)
try:
    TIMEZONE = ZoneInfo(TIMEZONE_STR)
except ZoneInfoNotFoundError:
    TIMEZONE = ZoneInfo("UTC")
    TIMEZONE_STR = "UTC"

MAX_API_RETRIES = 3
RETRY_DELAY_SECONDS = 5
POSITION_CONFIRM_DELAY_SECONDS = 8
LOOP_DELAY_SECONDS = 15  # Default loop delay if not in config
BYBIT_API_KLINE_LIMIT = 1000  # Max klines per request for Bybit
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP = {  # Map config intervals to CCXT timeframes
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}
DEFAULT_FETCH_LIMIT = 750  # Default number of candles to fetch
MAX_DF_LEN = 2000  # Max length of DataFrame to keep in memory

# Strategy Defaults
DEFAULT_VT_LENGTH = 40
DEFAULT_VT_ATR_PERIOD = 200
DEFAULT_VT_VOL_EMA_LENGTH = 950
DEFAULT_VT_ATR_MULTIPLIER = 3.0
DEFAULT_OB_SOURCE = "Wicks"  # "Wicks" or "Body"
DEFAULT_PH_LEFT = 10
DEFAULT_PH_RIGHT = 10
DEFAULT_PL_LEFT = 10
DEFAULT_PL_RIGHT = 10
DEFAULT_OB_EXTEND = True
DEFAULT_OB_MAX_BOXES = 50

QUOTE_CURRENCY = "USDT"  # Default, will be updated from config

# Color constants for logging
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
RESET = Style.RESET_ALL
BRIGHT = Style.BRIGHT
DIM = Style.DIM

# --- Create Log Directory ---
try:
    os.makedirs(LOG_DIRECTORY, exist_ok=True)
except OSError:
    sys.exit(1)

# --- Global State ---
_shutdown_requested = False  # Flag for graceful shutdown


# --- Type Definitions (using TypedDict for clarity) ---
class OrderBlock(TypedDict):
    id: str
    type: str  # 'BULL' or 'BEAR'
    timestamp: pd.Timestamp
    top: Decimal
    bottom: Decimal
    active: bool
    violated: bool
    violation_ts: pd.Timestamp | None
    extended_to_ts: pd.Timestamp | None  # For potential future use


class StrategyAnalysisResults(TypedDict):
    dataframe: pd.DataFrame
    last_close: Decimal
    current_trend_up: bool | None  # True for up, False for down, None if undetermined
    trend_just_changed: bool
    active_bull_boxes: list[OrderBlock]
    active_bear_boxes: list[OrderBlock]
    vol_norm_int: int | None  # Volume ratio normalized (0-100)
    atr: Decimal | None
    upper_band: Decimal | None  # VT upper band
    lower_band: Decimal | None  # VT lower band


# Extended MarketInfo TypedDict based on ccxt structure + custom fields
class MarketInfo(TypedDict, total=False):  # Use total=False as not all fields are guaranteed
    id: str
    symbol: str
    base: str
    quote: str
    settle: str | None
    baseId: str
    quoteId: str
    settleId: str | None
    type: str  # 'spot', 'future', 'option', 'swap'
    spot: bool
    margin: bool
    swap: bool
    future: bool
    option: bool
    active: bool | None
    contract: bool
    linear: bool | None
    inverse: bool | None
    quanto: bool | None
    taker: float
    maker: float
    contractSize: Any | None
    expiry: int | None
    expiryDatetime: str | None
    strike: float | None
    optionType: str | None
    precision: dict[str, Any]  # {'amount': float, 'price': float, 'cost': float, 'base': float, 'quote': float}
    limits: dict[str, Any]  # {'amount': {'min': float, 'max': float}, 'price': {'min': float, 'max': float}, 'cost': {'min': float, 'max': float}, 'leverage': {'min': float, 'max': float}}
    info: dict[str, Any]  # Raw exchange info
    # --- Custom Added Fields ---
    is_contract: bool  # Convenience flag
    is_linear: bool  # Convenience flag
    is_inverse: bool  # Convenience flag
    contract_type_str: str  # 'Linear', 'Inverse', 'Spot'
    min_amount_decimal: Decimal | None
    max_amount_decimal: Decimal | None
    min_cost_decimal: Decimal | None
    max_cost_decimal: Decimal | None
    amount_precision_step_decimal: Decimal | None  # Smallest tradeable unit of amount
    price_precision_step_decimal: Decimal | None  # Smallest tradeable unit of price
    contract_size_decimal: Decimal  # Contract size as Decimal


# Extended PositionInfo TypedDict based on ccxt structure + custom fields
class PositionInfo(TypedDict, total=False):  # Use total=False
    id: str | None
    symbol: str
    timestamp: int | None
    datetime: str | None
    contracts: float | None  # Size in contracts (or base currency for spot)
    contractSize: Any | None
    side: str | None  # 'long' or 'short'
    notional: Any | None  # Value in quote currency
    leverage: Any | None
    unrealizedPnl: Any | None
    realizedPnl: Any | None
    collateral: Any | None
    entryPrice: Any | None
    markPrice: Any | None
    liquidationPrice: Any | None
    marginMode: str | None  # 'cross' or 'isolated'
    hedged: bool | None
    maintenanceMargin: Any | None
    maintenanceMarginPercentage: float | None
    initialMargin: Any | None
    initialMarginPercentage: float | None
    marginRatio: float | None
    lastUpdateTimestamp: int | None
    info: dict[str, Any]  # Raw exchange info
    # --- Custom Added Fields ---
    size_decimal: Decimal  # Position size as Decimal
    entryPrice_decimal: Decimal | None
    markPrice_decimal: Decimal | None
    liquidationPrice_decimal: Decimal | None
    leverage_decimal: Decimal | None
    unrealizedPnl_decimal: Decimal | None
    notional_decimal: Decimal | None
    collateral_decimal: Decimal | None
    initialMargin_decimal: Decimal | None
    maintenanceMargin_decimal: Decimal | None
    stopLossPrice: str | None  # From Bybit V5 info if available
    takeProfitPrice: str | None  # From Bybit V5 info if available
    trailingStopLoss: str | None  # TSL distance from Bybit V5 info
    tslActivationPrice: str | None  # TSL activation price from Bybit V5 info
    be_activated: bool  # Flag if break-even SL is active
    tsl_activated: bool  # Flag if TSL is active


class SignalResult(TypedDict):
    signal: str  # 'BUY', 'SELL', 'HOLD', 'EXIT_LONG', 'EXIT_SHORT'
    reason: str
    initial_sl: Decimal | None
    initial_tp: Decimal | None


class BacktestResult(TypedDict):
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_profit: Decimal
    max_drawdown: Decimal
    profit_factor: float


# --- Logging Setup ---
class SensitiveFormatter(logging.Formatter):
    """Redacts API keys and secrets from log messages."""
    _api_key_placeholder = "***BYBIT_API_KEY_REDACTED***"
    _api_secret_placeholder = "***BYBIT_API_SECRET_REDACTED***"

    def format(self, record: logging.LogRecord) -> str:
        formatted_msg = super().format(record)
        redacted_msg = formatted_msg
        key = API_KEY
        secret = API_SECRET
        try:
            # Basic redaction: replace the full key/secret if found
            if key and isinstance(key, str) and len(key) > 4:  # Avoid redacting short strings
                redacted_msg = redacted_msg.replace(key, self._api_key_placeholder)
            if secret and isinstance(secret, str) and len(secret) > 4:
                redacted_msg = redacted_msg.replace(secret, self._api_secret_placeholder)
        except Exception:
            # Failsafe: if redaction causes an error, return the original message
            return formatted_msg
        return redacted_msg


class NeonConsoleFormatter(SensitiveFormatter):
    """Applies color coding to console log levels."""
    _level_colors = {
        logging.DEBUG: DIM + NEON_BLUE,
        logging.INFO: NEON_BLUE,
        logging.WARNING: NEON_YELLOW,
        logging.ERROR: NEON_RED,
        logging.CRITICAL: BRIGHT + NEON_RED
    }
    _default_color = NEON_BLUE  # Default color for levels not in map
    _log_format = (
        f"{DIM}%(asctime)s{RESET} {NEON_PURPLE}[%(name)-15s]{RESET} "
        f"%(levelcolor)s%(levelname)-8s{RESET} %(message)s"
    )
    _date_format = '%Y-%m-%d %H:%M:%S'  # Local time format for console

    def __init__(self, **kwargs) -> None:
        super().__init__(fmt=self._log_format, datefmt=self._date_format, **kwargs)
        # Use local timezone for console output timestamps
        self.converter = lambda timestamp, _: datetime.fromtimestamp(timestamp, tz=TIMEZONE).timetuple()

    def format(self, record: logging.LogRecord) -> str:
        # Add level-specific color information to the record before formatting
        record.levelcolor = self._level_colors.get(record.levelno, self._default_color)
        return super().format(record)


def setup_logger(name: str) -> logging.Logger:
    """Sets up a logger with file and console handlers."""
    safe_filename_part = re.sub(r'[^\w\-.]', '_', name)  # Sanitize name for filename
    logger_name = f"pyrmethus.{safe_filename_part}"
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")

    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():  # Avoid adding duplicate handlers if called multiple times
        logger.debug(f"Logger '{logger_name}' already configured.")
        return logger

    logger.setLevel(logging.DEBUG)  # Capture all levels, handlers control output level

    # --- File Handler (Rotating File) ---
    try:
        fh = RotatingFileHandler(log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8')
        fh.setLevel(logging.DEBUG)  # Log everything to file
        file_formatter = SensitiveFormatter(
            "%(asctime)s.%(msecs)03d UTC [%(name)s:%(lineno)d] %(levelname)-8s %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_formatter.converter = time.gmtime  # Use UTC for file logs
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)
    except Exception:
        pass

    # --- Console Handler (Colored Output) ---
    try:
        sh = logging.StreamHandler(sys.stdout)
        # Set console level from env var, default to INFO
        console_log_level_str = os.getenv("CONSOLE_LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, console_log_level_str, logging.INFO)
        if not isinstance(log_level, int):  # Fallback if invalid level string
             log_level = logging.INFO
        sh.setLevel(log_level)
        console_formatter = NeonConsoleFormatter()
        sh.setFormatter(console_formatter)
        logger.addHandler(sh)
    except Exception:
        pass

    logger.propagate = False  # Prevent logs from propagating to the root logger
    logger.debug(f"Logger '{logger_name}' initialized. File: '{log_filename}', Console Level: {logging.getLevelName(sh.level)}")
    return logger


# --- Initial Logger Setup ---
init_logger = setup_logger("init")  # Logger for initialization phase
init_logger.info(f"{Fore.MAGENTA}{BRIGHT}===== Pyrmethus Volumatic Bot v{BOT_VERSION} Initializing ====={Style.RESET_ALL}")
init_logger.info(f"Timezone: {TIMEZONE_STR} ({TIMEZONE})")
init_logger.debug(f"Decimal Precision: {getcontext().prec}")
init_logger.debug("Required packages: pandas, pandas_ta, numpy, ccxt, requests, python-dotenv, colorama, tzdata (optional but recommended)")


# --- Notification Setup ---
def send_notification(subject: str, body: str, logger: logging.Logger) -> bool:
    """Sends an email notification if configured."""
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = os.getenv("SMTP_PORT", "587")  # Default SMTP port
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    recipient = os.getenv("NOTIFICATION_EMAIL")

    if not all([smtp_server, smtp_port, smtp_user, smtp_password, recipient]):
        logger.warning("Notification setup incomplete. Set SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, NOTIFICATION_EMAIL in .env.")
        return False

    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = smtp_user
        msg['To'] = recipient

        with smtplib.SMTP(smtp_server, int(smtp_port)) as server:
            server.starttls()  # Upgrade connection to secure
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, recipient, msg.as_string())
        logger.info(f"Sent notification: {subject}")
        return True
    except Exception as e:
        logger.error(f"Failed to send notification: {e}")
        return False


# --- Configuration Loading & Validation ---
def _ensure_config_keys(config: dict[str, Any], default_config: dict[str, Any], parent_key: str = "") -> tuple[dict[str, Any], bool]:
    """Recursively ensures all keys from default_config exist in config."""
    updated_config = config.copy()
    changed = False
    for key, default_value in default_config.items():
        full_key_path = f"{parent_key}.{key}" if parent_key else key
        if key not in updated_config:
            updated_config[key] = default_value
            changed = True
            init_logger.info(f"{NEON_YELLOW}Config Update: Added missing key '{full_key_path}' with default: {repr(default_value)}{RESET}")
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # Recursively check nested dictionaries
            nested_config, nested_changed = _ensure_config_keys(updated_config[key], default_value, full_key_path)
            if nested_changed:
                updated_config[key] = nested_config
                changed = True
    return updated_config, changed


def _validate_and_correct_numeric(
    cfg_level: dict[str, Any],
    default_level: dict[str, Any],
    leaf_key: str,
    key_path: str,
    min_val: Decimal | int | float,
    max_val: Decimal | int | float,
    is_strict_min: bool = False,  # If True, value must be > min_val, else >= min_val
    is_int: bool = False,  # If True, value must be an integer
    allow_zero: bool = False  # If True, zero is allowed even if outside min/max range
) -> bool:
    """Validates a numeric config value, corrects type/range, logs changes."""
    original_val = cfg_level.get(leaf_key)
    default_val = default_level.get(leaf_key)
    corrected = False
    final_val = original_val
    target_type_str = 'integer' if is_int else 'float'

    try:
        # Initial type check
        if isinstance(original_val, bool):  # Booleans are not numeric
            raise TypeError("Boolean type is not valid for numeric configuration.")

        # Convert to Decimal for reliable comparison
        try:
            num_val = Decimal(str(original_val).strip())
        except (InvalidOperation, TypeError, ValueError):
            raise TypeError(f"Value '{repr(original_val)}' cannot be converted to a number.")

        if not num_val.is_finite():
            raise ValueError("Non-finite value (NaN or Infinity) is not allowed.")

        # Range check
        min_dec = Decimal(str(min_val))
        max_dec = Decimal(str(max_val))
        is_zero = num_val.is_zero()

        min_check_passed = (num_val > min_dec) if is_strict_min else (num_val >= min_dec)
        max_check_passed = (num_val <= max_dec)
        range_check_passed = min_check_passed and max_check_passed

        if not range_check_passed and not (allow_zero and is_zero):
            range_str = f"{'(' if is_strict_min else '['}{min_val}, {max_val}{']'}"
            allowed_str = f"{range_str}{' or 0' if allow_zero else ''}"
            raise ValueError(f"Value {num_val.normalize()} is outside the allowed range {allowed_str}.")

        # Type and Precision Correction
        needs_type_correction = False
        if is_int:
            if num_val % 1 != 0:  # Check if it has a fractional part
                needs_type_correction = True
                final_val = int(num_val.to_integral_value(rounding=ROUND_DOWN))  # Truncate
                init_logger.info(f"{NEON_YELLOW}Config Update: Truncated '{key_path}' from {repr(original_val)} to integer {repr(final_val)}.{RESET}")
                # Re-check range after truncation
                final_dec_trunc = Decimal(final_val)
                min_check_passed_trunc = (final_dec_trunc > min_dec) if is_strict_min else (final_dec_trunc >= min_dec)
                range_check_passed_trunc = min_check_passed_trunc and (final_dec_trunc <= max_dec)
                if not range_check_passed_trunc and not (allow_zero and final_dec_trunc.is_zero()):
                     range_str = f"{'(' if is_strict_min else '['}{min_val}, {max_val}{']'}"
                     allowed_str = f"{range_str}{' or 0' if allow_zero else ''}"
                     raise ValueError(f"Value truncated to {final_val}, which is outside allowed range {allowed_str}.")
            elif not isinstance(original_val, int):  # Correct type if it's like 10.0
                needs_type_correction = True
                final_val = int(num_val)
                init_logger.info(f"{NEON_YELLOW}Config Update: Corrected '{key_path}' from {type(original_val).__name__} to int: {repr(final_val)}.{RESET}")
            else:
                final_val = int(num_val)  # Ensure it's definitely an int
        else:  # Target is float
            if not isinstance(original_val, (float, int)):  # Allow ints to be treated as floats
                needs_type_correction = True
                final_val = float(num_val)
                init_logger.info(f"{NEON_YELLOW}Config Update: Corrected '{key_path}' from {type(original_val).__name__} to float: {repr(final_val)}.{RESET}")
            elif isinstance(original_val, float):
                # Check for potential precision issues if converting back and forth
                converted_float = float(num_val)
                if abs(original_val - converted_float) > 1e-9:  # Tolerance for float comparison
                    needs_type_correction = True
                    final_val = converted_float
                    init_logger.info(f"{NEON_YELLOW}Config Update: Adjusted '{key_path}' float representation from {repr(original_val)} to {repr(final_val)}.{RESET}")
                else:
                    final_val = converted_float  # Use the float representation
            elif isinstance(original_val, int):
                 final_val = float(original_val)  # Convert int to float if needed
            else:
                 final_val = float(num_val)  # Ensure it's definitely a float

        if needs_type_correction:
            corrected = True

    except (ValueError, InvalidOperation, TypeError) as e:
        # Handle any validation error by reverting to default
        range_str = f"{'(' if is_strict_min else '['}{min_val}, {max_val}{']'}"
        if allow_zero: range_str += " or 0"
        init_logger.warning(
            f"{NEON_YELLOW}Config Warning: Invalid value for '{key_path}'.\n"
            f"  Provided: {repr(original_val)} (Type: {type(original_val).__name__})\n"
            f"  Problem: {e}\n"
            f"  Expected: {target_type_str} in range {range_str}\n"
            f"  Using default: {repr(default_val)}{RESET}"
        )
        final_val = default_val
        corrected = True

    # Update the config dictionary if a correction was made
    if corrected:
        cfg_level[leaf_key] = final_val

    return corrected


def load_config(filepath: str) -> dict[str, Any]:
    """Loads, validates, and potentially updates the configuration file."""
    global QUOTE_CURRENCY  # Allow updating the global constant
    init_logger.info(f"{Fore.CYAN}# Loading configuration from '{filepath}'...{Style.RESET_ALL}")

    # --- Default Configuration Structure ---
    # Used for comparison, adding missing keys, and validation fallbacks
    default_config = {
        "trading_pairs": ["BTC/USDT:USDT"],  # Example format, expects Perpetual contracts
        "interval": "5",  # Default timeframe
        "enable_trading": False,  # Safety default: start in dry-run mode
        "use_sandbox": True,  # Safety default: use testnet/sandbox
        "quote_currency": "USDT",  # Primary currency for balance/PNL
        "max_concurrent_positions": 1,  # Max open positions per symbol (or globally, depending on logic)
        "risk_per_trade": 0.01,  # Risk 1% of balance per trade
        "leverage": 10,  # Default leverage for contract trading
        "retry_delay": RETRY_DELAY_SECONDS,  # API retry delay base
        "loop_delay_seconds": LOOP_DELAY_SECONDS,  # Main loop iteration delay
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS,  # Delay after placing order before checking position
        "fetch_limit": DEFAULT_FETCH_LIMIT,  # Number of candles to fetch
        "orderbook_limit": 25,  # Limit for order book depth (if used)
        "strategy_params": {
            "vt_length": DEFAULT_VT_LENGTH,
            "vt_atr_period": DEFAULT_VT_ATR_PERIOD,
            "vt_vol_ema_length": DEFAULT_VT_VOL_EMA_LENGTH,
            "vt_atr_multiplier": float(DEFAULT_VT_ATR_MULTIPLIER),
            "ob_source": DEFAULT_OB_SOURCE,  # "Wicks" or "Body"
            "ph_left": DEFAULT_PH_LEFT,
            "ph_right": DEFAULT_PH_RIGHT,
            "pl_left": DEFAULT_PL_LEFT,
            "pl_right": DEFAULT_PL_RIGHT,
            "ob_extend": DEFAULT_OB_EXTEND,  # Boolean
            "ob_max_boxes": DEFAULT_OB_MAX_BOXES,  # Max OBs to track
            "ob_entry_proximity_factor": 1.005,  # Price must be within 0.5% of OB edge for entry
            "ob_exit_proximity_factor": 1.001,  # Price must be within 0.1% of OB edge for exit/invalidation
        },
        "protection": {
            "initial_stop_loss_atr_multiple": 1.8,  # SL distance = 1.8 * ATR
            "initial_take_profit_atr_multiple": 0.7,  # TP distance = 0.7 * ATR (0 to disable)
            "enable_break_even": True,
            "break_even_trigger_atr_multiple": 1.0,  # Move SL to BE when price moves 1.0 * ATR in profit
            "break_even_offset_ticks": 2,  # Offset BE SL by N price ticks
            "enable_trailing_stop": True,
            "trailing_stop_callback_rate": 0.005,  # TSL callback rate (e.g., 0.5%) - Bybit specific? Check API docs
            "trailing_stop_activation_percentage": 0.003,  # Activate TSL when price moves 0.3% beyond entry - Bybit specific?
        },
        "notifications": {
            "enable_notifications": True,  # Master switch for email notifications
        },
        "backtesting": {
            "enabled": False,
            "start_date": "2023-01-01",  # Format YYYY-MM-DD
            "end_date": "2023-12-31",   # Format YYYY-MM-DD
        }
    }

    config_needs_saving = False
    loaded_config = {}

    # --- Load or Create Config File ---
    if not os.path.exists(filepath):
        init_logger.warning(f"{NEON_YELLOW}Config file '{filepath}' not found. Creating with defaults.{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Created default config: {filepath}{RESET}")
            loaded_config = default_config  # Start with defaults
        except OSError as e:
            init_logger.critical(f"{NEON_RED}FATAL ERROR: Could not create config file '{filepath}': {e}. Using hardcoded defaults.{RESET}")
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")  # Set global from default
            return default_config  # Return defaults as a fallback
    else:
        try:
            with open(filepath, encoding="utf-8") as f:
                loaded_config = json.load(f)
            if not isinstance(loaded_config, dict):
                raise TypeError("Config file root must be a JSON object.")
            init_logger.info(f"Loaded config from '{filepath}'.")
        except json.JSONDecodeError as e:
            init_logger.error(f"{NEON_RED}JSON decode error in '{filepath}': {e}. Recreating with defaults.{RESET}")
            try:
                backup_path = f"{filepath}.corrupted_{int(time.time())}.bak"
                os.replace(filepath, backup_path)  # Backup corrupted file
                init_logger.info(f"Backed up corrupted config to: {backup_path}")
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(default_config, f, indent=4, ensure_ascii=False)
                init_logger.info(f"{NEON_GREEN}Recreated default config: {filepath}{RESET}")
                loaded_config = default_config
            except Exception as e_create:
                init_logger.critical(f"{NEON_RED}FATAL ERROR: Error recreating config after JSON error: {e_create}. Using hardcoded defaults.{RESET}")
                QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
                return default_config
        except TypeError as e:
             init_logger.error(f"{NEON_RED}Type error reading config '{filepath}': {e}. Using hardcoded defaults.{RESET}")
             QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
             return default_config
        except Exception as e:
             init_logger.error(f"{NEON_RED}Unexpected error loading config '{filepath}': {e}. Using hardcoded defaults.{RESET}")
             QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
             return default_config

    # --- Ensure all keys exist ---
    loaded_config, keys_added = _ensure_config_keys(loaded_config, default_config)
    if keys_added:
        config_needs_saving = True
        init_logger.info("Configuration updated with missing default keys.")

    # --- Validate and Correct Specific Values ---
    cfg = loaded_config  # Alias for brevity
    def_cfg = default_config  # Alias for brevity
    changes = []

    # Top Level
    changes.append(_validate_and_correct_numeric(cfg, def_cfg, "max_concurrent_positions", "max_concurrent_positions", 1, 100, is_int=True))
    changes.append(_validate_and_correct_numeric(cfg, def_cfg, "risk_per_trade", "risk_per_trade", 0.0001, 0.2, is_strict_min=True))  # Risk 0.01% to 20%
    changes.append(_validate_and_correct_numeric(cfg, def_cfg, "leverage", "leverage", 1, 125, is_int=True))  # Leverage 1 to 125
    changes.append(_validate_and_correct_numeric(cfg, def_cfg, "retry_delay", "retry_delay", 1, 60, is_int=True))
    changes.append(_validate_and_correct_numeric(cfg, def_cfg, "loop_delay_seconds", "loop_delay_seconds", 5, 300, is_int=True))
    changes.append(_validate_and_correct_numeric(cfg, def_cfg, "position_confirm_delay_seconds", "position_confirm_delay_seconds", 1, 60, is_int=True))
    changes.append(_validate_and_correct_numeric(cfg, def_cfg, "fetch_limit", "fetch_limit", 100, BYBIT_API_KLINE_LIMIT, is_int=True))
    changes.append(_validate_and_correct_numeric(cfg, def_cfg, "orderbook_limit", "orderbook_limit", 1, 200, is_int=True))

    # Trading Pairs Validation (Basic)
    if not isinstance(cfg.get("trading_pairs"), list) or not all(isinstance(p, str) and '/' in p for p in cfg["trading_pairs"]):
        init_logger.warning(f"{NEON_YELLOW}Config Warning: 'trading_pairs' is invalid. Expected list of strings like 'BTC/USDT:USDT'. Using default: {def_cfg['trading_pairs']}{RESET}")
        cfg["trading_pairs"] = def_cfg["trading_pairs"]
        changes.append(True)

    # Interval Validation
    if str(cfg.get("interval")) not in VALID_INTERVALS:
        init_logger.warning(f"{NEON_YELLOW}Config Warning: 'interval' ('{cfg.get('interval')}') is invalid. Must be one of {VALID_INTERVALS}. Using default: '{def_cfg['interval']}'{RESET}")
        cfg["interval"] = def_cfg["interval"]
        changes.append(True)

    # Boolean Validation
    for bool_key in ["enable_trading", "use_sandbox"]:
        if not isinstance(cfg.get(bool_key), bool):
            init_logger.warning(f"{NEON_YELLOW}Config Warning: '{bool_key}' must be true or false. Using default: {def_cfg[bool_key]}{RESET}")
            cfg[bool_key] = def_cfg[bool_key]
            changes.append(True)

    # Strategy Params
    sp = cfg["strategy_params"]
    def_sp = def_cfg["strategy_params"]
    changes.append(_validate_and_correct_numeric(sp, def_sp, "vt_length", "strategy_params.vt_length", 5, 500, is_int=True))
    changes.append(_validate_and_correct_numeric(sp, def_sp, "vt_atr_period", "strategy_params.vt_atr_period", 5, 1000, is_int=True))
    changes.append(_validate_and_correct_numeric(sp, def_sp, "vt_vol_ema_length", "strategy_params.vt_vol_ema_length", 10, 2000, is_int=True))
    changes.append(_validate_and_correct_numeric(sp, def_sp, "vt_atr_multiplier", "strategy_params.vt_atr_multiplier", 0.1, 10.0))
    if sp.get("ob_source") not in ["Wicks", "Body"]:
        init_logger.warning(f"{NEON_YELLOW}Config Warning: 'strategy_params.ob_source' must be 'Wicks' or 'Body'. Using default: '{def_sp['ob_source']}'{RESET}")
        sp["ob_source"] = def_sp["ob_source"]; changes.append(True)
    changes.append(_validate_and_correct_numeric(sp, def_sp, "ph_left", "strategy_params.ph_left", 1, 50, is_int=True))
    changes.append(_validate_and_correct_numeric(sp, def_sp, "ph_right", "strategy_params.ph_right", 1, 50, is_int=True))
    changes.append(_validate_and_correct_numeric(sp, def_sp, "pl_left", "strategy_params.pl_left", 1, 50, is_int=True))
    changes.append(_validate_and_correct_numeric(sp, def_sp, "pl_right", "strategy_params.pl_right", 1, 50, is_int=True))
    if not isinstance(sp.get("ob_extend"), bool):
        init_logger.warning(f"{NEON_YELLOW}Config Warning: 'strategy_params.ob_extend' must be true or false. Using default: {def_sp['ob_extend']}{RESET}")
        sp["ob_extend"] = def_sp["ob_extend"]; changes.append(True)
    changes.append(_validate_and_correct_numeric(sp, def_sp, "ob_max_boxes", "strategy_params.ob_max_boxes", 1, 200, is_int=True))
    changes.append(_validate_and_correct_numeric(sp, def_sp, "ob_entry_proximity_factor", "strategy_params.ob_entry_proximity_factor", 1.0, 1.1))
    changes.append(_validate_and_correct_numeric(sp, def_sp, "ob_exit_proximity_factor", "strategy_params.ob_exit_proximity_factor", 1.0, 1.1))

    # Protection Params
    pp = cfg["protection"]
    def_pp = def_cfg["protection"]
    changes.append(_validate_and_correct_numeric(pp, def_pp, "initial_stop_loss_atr_multiple", "protection.initial_stop_loss_atr_multiple", 0.1, 10.0))
    changes.append(_validate_and_correct_numeric(pp, def_pp, "initial_take_profit_atr_multiple", "protection.initial_take_profit_atr_multiple", 0.0, 10.0, allow_zero=True))  # Allow 0 to disable TP
    if not isinstance(pp.get("enable_break_even"), bool):
        init_logger.warning(f"{NEON_YELLOW}Config Warning: 'protection.enable_break_even' must be true or false. Using default: {def_pp['enable_break_even']}{RESET}")
        pp["enable_break_even"] = def_pp["enable_break_even"]; changes.append(True)
    changes.append(_validate_and_correct_numeric(pp, def_pp, "break_even_trigger_atr_multiple", "protection.break_even_trigger_atr_multiple", 0.1, 10.0))
    changes.append(_validate_and_correct_numeric(pp, def_pp, "break_even_offset_ticks", "protection.break_even_offset_ticks", 0, 100, is_int=True, allow_zero=True))
    if not isinstance(pp.get("enable_trailing_stop"), bool):
        init_logger.warning(f"{NEON_YELLOW}Config Warning: 'protection.enable_trailing_stop' must be true or false. Using default: {def_pp['enable_trailing_stop']}{RESET}")
        pp["enable_trailing_stop"] = def_pp["enable_trailing_stop"]; changes.append(True)
    changes.append(_validate_and_correct_numeric(pp, def_pp, "trailing_stop_callback_rate", "protection.trailing_stop_callback_rate", 0.0001, 0.1))  # e.g., 0.01% to 10%
    changes.append(_validate_and_correct_numeric(pp, def_pp, "trailing_stop_activation_percentage", "protection.trailing_stop_activation_percentage", 0.0001, 0.1))

    # Notifications
    if not isinstance(cfg["notifications"].get("enable_notifications"), bool):
        init_logger.warning(f"{NEON_YELLOW}Config Warning: 'notifications.enable_notifications' must be true or false. Using default: {def_cfg['notifications']['enable_notifications']}{RESET}")
        cfg["notifications"]["enable_notifications"] = def_cfg['notifications']['enable_notifications']; changes.append(True)

    # Backtesting
    bp = cfg["backtesting"]
    def_bp = def_cfg["backtesting"]
    if not isinstance(bp.get("enabled"), bool):
        init_logger.warning(f"{NEON_YELLOW}Config Warning: 'backtesting.enabled' must be true or false. Using default: {def_bp['enabled']}{RESET}")
        bp["enabled"] = def_bp["enabled"]; changes.append(True)
    # Basic date format check
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for date_key in ["start_date", "end_date"]:
        if not isinstance(bp.get(date_key), str) or not date_pattern.match(bp[date_key]):
             init_logger.warning(f"{NEON_YELLOW}Config Warning: 'backtesting.{date_key}' ('{bp.get(date_key)}') is invalid. Expected YYYY-MM-DD format. Using default: '{def_bp[date_key]}'{RESET}")
             bp[date_key] = def_bp[date_key]; changes.append(True)

    # --- Update Global Quote Currency ---
    new_quote = cfg.get("quote_currency", "USDT")
    if isinstance(new_quote, str) and len(new_quote) > 1:
        QUOTE_CURRENCY = new_quote.upper()
    else:
        init_logger.warning(f"{NEON_YELLOW}Config Warning: Invalid 'quote_currency'. Using default: '{def_cfg['quote_currency']}'{RESET}")
        cfg["quote_currency"] = def_cfg["quote_currency"]
        QUOTE_CURRENCY = def_cfg["quote_currency"]
        changes.append(True)
    init_logger.info(f"Quote currency set to: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")

    # --- Save if any changes were made ---
    if any(changes) or config_needs_saving:
        init_logger.info("Configuration has been updated or corrected.")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Saved updated config to '{filepath}'.{RESET}")
        except OSError as e:
            init_logger.error(f"{NEON_RED}Failed to save updated config: {e}{RESET}")

    return cfg


# --- Utility Functions ---
def _safe_market_decimal(
    value: Any,
    field_name: str,
    allow_zero: bool = False,
    allow_negative: bool = False,
    default: Decimal | None = None
) -> Decimal | None:
    """Safely convert market data (limits, precision) to Decimal."""
    if value is None:
        return default
    try:
        # Handle potential string representations like 'inf' or scientific notation
        str_val = str(value).strip().lower()
        if str_val in ['inf', '+inf', '-inf', 'nan']:
             init_logger.warning(f"Non-finite value '{value}' for {field_name}. Using default: {default}")
             return default

        dec_val = Decimal(str_val)

        if not dec_val.is_finite():  # Double check after conversion
            init_logger.warning(f"Converted value for {field_name} is non-finite: {value}. Using default: {default}")
            return default
        if not allow_zero and dec_val.is_zero():
            # Log as debug because '0' might be valid (e.g. min cost can be 0)
            init_logger.debug(f"Zero value encountered for {field_name}: {value}. Treated as {default if default is not None else 'None'}")
            return default if not allow_zero else Decimal('0')  # Return 0 if allowed
        if not allow_negative and dec_val < 0:
            init_logger.warning(f"Negative value not allowed for {field_name}: {value}. Using default: {default}")
            return default

        # Quantize to a reasonable precision to avoid floating point issues from source
        return dec_val.quantize(Decimal('1e-15'), rounding=ROUND_DOWN)

    except (ValueError, TypeError, InvalidOperation) as e:
        init_logger.warning(f"Invalid value for {field_name}: '{value}' ({type(value).__name__}) - Error: {e}. Using default: {default}")
        return default


def _format_price(exchange: ccxt.Exchange, symbol: str, price: Decimal) -> str | None:
    """Formats a price according to the market's precision rules."""
    try:
        market = exchange.market(symbol)
        price_precision_val = market.get('precision', {}).get('price')
        if price_precision_val is None:
            init_logger.warning(f"No price precision found for {symbol}. Using raw price string.")
            return str(price.normalize())  # Return normalized decimal string

        # CCXT precision can be integer (decimal places) or float (tick size)
        if isinstance(price_precision_val, int):  # Decimal places
             tick_size = Decimal('1e-' + str(price_precision_val))
        elif isinstance(price_precision_val, float):  # Tick size
             tick_size = Decimal(str(price_precision_val))
        else:  # Assume tick size if string
             tick_size = Decimal(str(price_precision_val))

        if not tick_size.is_finite() or tick_size <= 0:
             init_logger.error(f"Invalid price tick size {tick_size} for {symbol}. Cannot format price.")
             return None

        # Use quantize with the tick size
        formatted_price = price.quantize(tick_size, rounding=ROUND_DOWN)  # Round down to be conservative
        return str(formatted_price.normalize())

    except (ccxt.ExchangeError, KeyError, ValueError, InvalidOperation, TypeError) as e:
        init_logger.error(f"Error formatting price for {symbol} ({price}): {e}")
        return None  # Indicate failure


# --- Market and Position Data Enhancement ---
def enhance_market_info(market: dict[str, Any]) -> MarketInfo:
    """Adds custom fields and Decimal types to ccxt market dict."""
    enhanced: MarketInfo = market.copy()  # Start with original dict
    limits = market.get('limits', {})
    amount_limits = limits.get('amount', {})
    cost_limits = limits.get('cost', {})
    precision = market.get('precision', {})

    # Basic contract type detection
    is_contract = market.get('contract', False) or market.get('swap', False) or market.get('future', False)
    is_linear = market.get('linear', False) and is_contract
    is_inverse = market.get('inverse', False) and is_contract
    contract_type = "Linear" if is_linear else "Inverse" if is_inverse else "Spot"

    # --- Convert precision/limits to Decimal ---
    amount_prec = precision.get('amount')
    price_prec = precision.get('price')
    contract_size = market.get('contractSize', 1.0)  # Default to 1 for spot/non-specified

    # Determine step size (tick size) from precision
    # CCXT precision: int = decimal places, float = tick size
    amount_step = None
    if isinstance(amount_prec, int): amount_step = Decimal('1e-' + str(amount_prec))
    elif isinstance(amount_prec, (float, str)): amount_step = _safe_market_decimal(amount_prec, "precision.amount", allow_zero=False)
    price_step = None
    if isinstance(price_prec, int): price_step = Decimal('1e-' + str(price_prec))
    elif isinstance(price_prec, (float, str)): price_step = _safe_market_decimal(price_prec, "precision.price", allow_zero=False)

    # Assign enhanced fields
    enhanced['is_contract'] = is_contract
    enhanced['is_linear'] = is_linear
    enhanced['is_inverse'] = is_inverse
    enhanced['contract_type_str'] = contract_type
    enhanced['min_amount_decimal'] = _safe_market_decimal(amount_limits.get('min'), 'limits.amount.min', allow_zero=True)
    enhanced['max_amount_decimal'] = _safe_market_decimal(amount_limits.get('max'), 'limits.amount.max')
    enhanced['min_cost_decimal'] = _safe_market_decimal(cost_limits.get('min'), 'limits.cost.min', allow_zero=True)
    enhanced['max_cost_decimal'] = _safe_market_decimal(cost_limits.get('max'), 'limits.cost.max')
    enhanced['amount_precision_step_decimal'] = amount_step
    enhanced['price_precision_step_decimal'] = price_step
    enhanced['contract_size_decimal'] = _safe_market_decimal(contract_size, 'contractSize', default=Decimal('1'))

    return enhanced


def enhance_position_info(position: dict[str, Any], market_info: MarketInfo) -> PositionInfo:
    """Adds custom fields and Decimal types to ccxt position dict."""
    enhanced: PositionInfo = position.copy()

    # Convert key numeric fields to Decimal
    enhanced['size_decimal'] = _safe_market_decimal(position.get('contracts'), 'position.contracts', allow_zero=True, allow_negative=True, default=Decimal('0'))
    enhanced['entryPrice_decimal'] = _safe_market_decimal(position.get('entryPrice'), 'position.entryPrice')
    enhanced['markPrice_decimal'] = _safe_market_decimal(position.get('markPrice'), 'position.markPrice')
    enhanced['liquidationPrice_decimal'] = _safe_market_decimal(position.get('liquidationPrice'), 'position.liquidationPrice')
    enhanced['leverage_decimal'] = _safe_market_decimal(position.get('leverage'), 'position.leverage')
    enhanced['unrealizedPnl_decimal'] = _safe_market_decimal(position.get('unrealizedPnl'), 'position.unrealizedPnl', allow_zero=True, allow_negative=True)
    enhanced['notional_decimal'] = _safe_market_decimal(position.get('notional'), 'position.notional', allow_zero=True, allow_negative=True)
    enhanced['collateral_decimal'] = _safe_market_decimal(position.get('collateral'), 'position.collateral', allow_zero=True, allow_negative=True)
    enhanced['initialMargin_decimal'] = _safe_market_decimal(position.get('initialMargin'), 'position.initialMargin', allow_zero=True)
    enhanced['maintenanceMargin_decimal'] = _safe_market_decimal(position.get('maintenanceMargin'), 'position.maintenanceMargin', allow_zero=True)

    # Extract Bybit V5 specific protection info if available
    info = position.get('info', {})
    enhanced['stopLossPrice'] = info.get('stopLoss')
    enhanced['takeProfitPrice'] = info.get('takeProfit')
    enhanced['trailingStopLoss'] = info.get('trailingStop')  # TSL distance value
    enhanced['tslActivationPrice'] = info.get('activePrice')  # TSL activation price

    # Initialize state flags (these would be managed by the bot's state machine)
    enhanced['be_activated'] = False
    enhanced['tsl_activated'] = False

    return enhanced


# --- Strategy Implementation ---
def analyze_market(
    df: pd.DataFrame,
    strategy_params: dict[str, Any],
    symbol: str,
    logger: logging.Logger
) -> StrategyAnalysisResults:
    """Analyzes market data using Volumatic Trend and Order Blocks."""
    lg = logger
    lg.debug(f"Analyzing market data for {symbol} (Rows: {len(df)})...")
    if len(df) < max(strategy_params.get('vt_length', DEFAULT_VT_LENGTH),
                     strategy_params.get('vt_atr_period', DEFAULT_VT_ATR_PERIOD),
                     strategy_params.get('vt_vol_ema_length', DEFAULT_VT_VOL_EMA_LENGTH),
                     strategy_params.get('ph_left', DEFAULT_PH_LEFT) + strategy_params.get('ph_right', DEFAULT_PH_RIGHT) + 1):
        lg.warning(f"Insufficient data for analysis ({len(df)} rows). Need more for lookback periods.")
        # Return default/empty results gracefully
        return StrategyAnalysisResults(
            dataframe=df, last_close=Decimal('0'), current_trend_up=None, trend_just_changed=False,
            active_bull_boxes=[], active_bear_boxes=[], vol_norm_int=None, atr=None, upper_band=None, lower_band=None
        )

    df = df.copy()  # Avoid modifying the original DataFrame
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
         df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True, drop=False)  # Use timestamp index for TA Lib

    # --- Volumatic Trend Calculation ---
    vt_length = strategy_params.get('vt_length', DEFAULT_VT_LENGTH)
    vt_atr_period = strategy_params.get('vt_atr_period', DEFAULT_VT_ATR_PERIOD)
    vt_vol_ema_length = strategy_params.get('vt_vol_ema_length', DEFAULT_VT_VOL_EMA_LENGTH)
    vt_atr_multiplier = strategy_params.get('vt_atr_multiplier', DEFAULT_VT_ATR_MULTIPLIER)

    try:
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=vt_atr_period)
        # Calculate volume EMA, handle potential division by zero if EMA is 0
        df['vol_ema'] = ta.ema(df['volume'], length=vt_vol_ema_length)
        df['vol_ratio'] = (df['volume'] / df['vol_ema'].replace(0, np.nan)).fillna(1.0)  # Avoid NaN/inf, default ratio 1 if EMA is 0
        df['trend_line'] = ta.ema(df['close'], length=vt_length)
        df['upper_band'] = df['trend_line'] + (df['atr'] * vt_atr_multiplier)
        df['lower_band'] = df['trend_line'] - (df['atr'] * vt_atr_multiplier)
    except Exception as e:
        lg.error(f"Error calculating VT indicators for {symbol}: {e}", exc_info=True)
        # Return empty results on calculation failure
        return StrategyAnalysisResults(
            dataframe=df, last_close=Decimal(str(df['close'].iloc[-1])) if not df.empty else Decimal('0'),
            current_trend_up=None, trend_just_changed=False, active_bull_boxes=[], active_bear_boxes=[],
            vol_norm_int=None, atr=None, upper_band=None, lower_band=None
        )

    # Determine trend based on last full candle
    if len(df) < 2:
        lg.warning("Not enough data (need >= 2 rows) to determine trend change.")
        current_trend_up = None
        trend_just_changed = False
    else:
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        # Define trend condition: Price > Trend Line AND Volume Ratio > 1 (indicating conviction)
        current_trend_up = (last_row['close'] > last_row['trend_line']) and (last_row['vol_ratio'] > 1.0)
        prev_trend_up = (prev_row['close'] > prev_row['trend_line']) and (prev_row['vol_ratio'] > 1.0)
        trend_just_changed = current_trend_up != prev_trend_up

    # --- Order Blocks Detection ---
    ob_source = strategy_params.get('ob_source', DEFAULT_OB_SOURCE)
    ph_left = strategy_params.get('ph_left', DEFAULT_PH_LEFT)
    ph_right = strategy_params.get('ph_right', DEFAULT_PH_RIGHT)
    pl_left = strategy_params.get('pl_left', DEFAULT_PL_LEFT)
    pl_right = strategy_params.get('pl_right', DEFAULT_PL_RIGHT)
    ob_max_boxes = strategy_params.get('ob_max_boxes', DEFAULT_OB_MAX_BOXES)

    # Define high/low source for pivots based on config
    high_col = 'high' if ob_source == "Wicks" else 'close'
    low_col = 'low' if ob_source == "Wicks" else 'open'  # Use open for 'Body' low

    # Calculate Pivot Highs/Lows using rolling window
    # Note: `center=True` makes the pivot point the middle of the window
    ph_window = ph_left + ph_right + 1
    pl_window = pl_left + pl_right + 1

    # Apply rolling function to find pivots - This can be slow on large DFs
    # Ensure enough data points for the rolling window
    if len(df) >= ph_window:
        df['pivot_high'] = df[high_col].rolling(window=ph_window, center=True, min_periods=ph_window).apply(
            lambda x: x[ph_left] if (x[ph_left] == x.max()) else np.nan, raw=True  # Simpler check: middle element is the max
        )
    else: df['pivot_high'] = np.nan
    if len(df) >= pl_window:
        df['pivot_low'] = df[low_col].rolling(window=pl_window, center=True, min_periods=pl_window).apply(
            lambda x: x[pl_left] if (x[pl_left] == x.min()) else np.nan, raw=True  # Simpler check: middle element is the min
        )
    else: df['pivot_low'] = np.nan

    # --- Identify and Filter Order Blocks ---
    all_bull_boxes: list[OrderBlock] = []
    all_bear_boxes: list[OrderBlock] = []

    for _idx, row in df.iterrows():
        ts = row['timestamp']  # Already a Timestamp object due to set_index
        # Bearish OB (formed by Pivot High)
        if not pd.isna(row['pivot_high']):
            # Define OB based on the candle *before* the pivot high candle (potentially)
            # Simplified: Use the pivot candle itself for OB boundaries
            ob_id = f"BEAR_{int(ts.timestamp() * 1000)}"
            top = Decimal(str(row['high']))  # Always use high for top
            bottom = Decimal(str(row['low']))  # Always use low for bottom (simplification)
            # More complex OB definition might look at prior candle body etc.
            ob = OrderBlock(
                id=ob_id, type="BEAR", timestamp=ts, top=top, bottom=bottom,
                active=True, violated=False, violation_ts=None, extended_to_ts=None
            )
            all_bear_boxes.append(ob)

        # Bullish OB (formed by Pivot Low)
        if not pd.isna(row['pivot_low']):
            ob_id = f"BULL_{int(ts.timestamp() * 1000)}"
            top = Decimal(str(row['high']))  # Always use high for top
            bottom = Decimal(str(row['low']))  # Always use low for bottom
            ob = OrderBlock(
                id=ob_id, type="BULL", timestamp=ts, top=top, bottom=bottom,
                active=True, violated=False, violation_ts=None, extended_to_ts=None
            )
            all_bull_boxes.append(ob)

    # Limit number of OBs, keeping the most recent ones
    all_bull_boxes = sorted(all_bull_boxes, key=lambda x: x['timestamp'], reverse=True)[:ob_max_boxes]
    all_bear_boxes = sorted(all_bear_boxes, key=lambda x: x['timestamp'], reverse=True)[:ob_max_boxes]

    # --- Check OB Violations & Activity ---
    active_bull_boxes = []
    active_bear_boxes = []
    last_close_dec = Decimal(str(df['close'].iloc[-1])) if not df.empty else Decimal('0')
    last_high_dec = Decimal(str(df['high'].iloc[-1])) if not df.empty else Decimal('0')
    last_low_dec = Decimal(str(df['low'].iloc[-1])) if not df.empty else Decimal('0')
    last_ts = df['timestamp'].iloc[-1] if not df.empty else pd.Timestamp.now(tz='UTC')

    # Simple violation check: If price closes beyond the OB range
    for ob in all_bull_boxes:
        if ob['active']:
            # Bull OB violated if low goes below OB bottom
            if last_low_dec < ob['bottom']:
                ob['violated'] = True
                ob['violation_ts'] = last_ts
                ob['active'] = False
            else:
                active_bull_boxes.append(ob)  # Still active

    for ob in all_bear_boxes:
        if ob['active']:
            # Bear OB violated if high goes above OB top
            if last_high_dec > ob['top']:
                ob['violated'] = True
                ob['violation_ts'] = last_ts
                ob['active'] = False
            else:
                active_bear_boxes.append(ob)  # Still active

    # --- Prepare Results ---
    last_row = df.iloc[-1] if not df.empty else None
    vol_norm_int = None
    atr_dec = None
    upper_band_dec = None
    lower_band_dec = None

    if last_row is not None:
        # Normalize volume ratio (0-100, clipped)
        vol_ratio = last_row.get('vol_ratio', np.nan)
        if not pd.isna(vol_ratio):
            # Scale ratio: e.g., ratio of 2 becomes 100, 1 becomes 50, 0.5 becomes 25
            vol_norm_int = int(min(max(vol_ratio * 50, 0), 100))

        atr = last_row.get('atr', np.nan)
        if not pd.isna(atr): atr_dec = Decimal(str(atr))

        upper_band = last_row.get('upper_band', np.nan)
        if not pd.isna(upper_band): upper_band_dec = Decimal(str(upper_band))

        lower_band = last_row.get('lower_band', np.nan)
        if not pd.isna(lower_band): lower_band_dec = Decimal(str(lower_band))

    # Return dataframe without index for consistency elsewhere if needed
    df_reset = df.reset_index(drop=False)

    return StrategyAnalysisResults(
        dataframe=df_reset,
        last_close=last_close_dec,
        current_trend_up=current_trend_up,
        trend_just_changed=trend_just_changed,
        active_bull_boxes=active_bull_boxes,
        active_bear_boxes=active_bear_boxes,
        vol_norm_int=vol_norm_int,
        atr=atr_dec,
        upper_band=upper_band_dec,
        lower_band=lower_band_dec
    )


def generate_signal(
    analysis: StrategyAnalysisResults,
    strategy_params: dict[str, Any],
    protection_params: dict[str, Any],
    symbol: str,
    logger: logging.Logger
) -> SignalResult:
    """Generates trading signals based on the market analysis."""
    lg = logger
    last_close = analysis['last_close']
    current_trend_up = analysis['current_trend_up']
    active_bull_boxes = analysis['active_bull_boxes']
    active_bear_boxes = analysis['active_bear_boxes']
    atr = analysis['atr']

    signal = "HOLD"  # Default signal
    reason = "No entry conditions met."
    initial_sl = None
    initial_tp = None

    if current_trend_up is None or atr is None or atr <= 0:
        reason = "Insufficient data or invalid ATR for signal generation."
        lg.debug(f"Signal ({symbol}): HOLD - {reason}")
        return SignalResult(signal=signal, reason=reason, initial_sl=None, initial_tp=None)

    # Configurable parameters
    entry_proximity = Decimal(str(strategy_params.get('ob_entry_proximity_factor', 1.005)))  # e.g., 1.005 means within 0.5%
    sl_atr_multiple = Decimal(str(protection_params.get('initial_stop_loss_atr_multiple', 1.8)))
    tp_atr_multiple = Decimal(str(protection_params.get('initial_take_profit_atr_multiple', 0.7)))

    # --- Entry Logic ---
    # Long Entry: Uptrend + Price near active Bull OB
    if current_trend_up:
        for ob in active_bull_boxes:
            if ob['active']:
                # Check if price is 'near' the OB (below top edge * proximity, but above bottom edge)
                entry_zone_top = ob['top'] * entry_proximity
                if last_close <= entry_zone_top and last_close >= ob['bottom']:
                    signal = "BUY"
                    reason = f"Price ({last_close.normalize()}) near active Bull OB [{ob['bottom'].normalize()} - {ob['top'].normalize()}] in uptrend."
                    # Calculate SL: Below OB bottom or based on ATR from entry
                    # Simple ATR based SL from last close:
                    initial_sl = last_close - (atr * sl_atr_multiple)
                    # Ensure SL is below the OB bottom for robustness
                    initial_sl = min(initial_sl, ob['bottom'] * Decimal('0.999'))  # Place slightly below OB bottom
                    if tp_atr_multiple > 0:
                        initial_tp = last_close + (atr * tp_atr_multiple)
                    break  # Take the first valid signal

    # Short Entry: Downtrend + Price near active Bear OB
    elif not current_trend_up:  # Explicitly check for downtrend (trend is not None and not True)
        for ob in active_bear_boxes:
             if ob['active']:
                # Check if price is 'near' the OB (above bottom edge / proximity, but below top edge)
                entry_zone_bottom = ob['bottom'] / entry_proximity
                if last_close >= entry_zone_bottom and last_close <= ob['top']:
                    signal = "SELL"
                    reason = f"Price ({last_close.normalize()}) near active Bear OB [{ob['bottom'].normalize()} - {ob['top'].normalize()}] in downtrend."
                    # Calculate SL: Above OB top or based on ATR from entry
                    initial_sl = last_close + (atr * sl_atr_multiple)
                    # Ensure SL is above the OB top
                    initial_sl = max(initial_sl, ob['top'] * Decimal('1.001'))  # Place slightly above OB top
                    if tp_atr_multiple > 0:
                        initial_tp = last_close - (atr * tp_atr_multiple)
                    break  # Take the first valid signal

    # --- Exit Logic (Placeholder - could be expanded) ---
    # Example: Exit if trend reverses strongly? Or if price hits opposite OB?
    # Simple exit: Not implemented here, relies on SL/TP/TSL or manual intervention.
    # Could add signals like "EXIT_LONG", "EXIT_SHORT" based on other conditions.

    # Log the generated signal
    log_level = logging.INFO if signal != "HOLD" else logging.DEBUG
    lg.log(log_level, f"Signal for {symbol}: {BRIGHT}{signal}{RESET} ({reason})")
    if initial_sl: lg.debug(f"  Proposed SL: {initial_sl.normalize()}")
    if initial_tp: lg.debug(f"  Proposed TP: {initial_tp.normalize()}")

    return SignalResult(
        signal=signal,
        reason=reason,
        initial_sl=initial_sl,
        initial_tp=initial_tp
    )


# --- Trading Functions ---
def calculate_position_size(
    exchange: ccxt.Exchange,
    symbol: str,
    balance: Decimal,
    risk_decimal: Decimal,  # Risk per trade (e.g., 0.01 for 1%)
    entry_price: Decimal,
    initial_stop_loss_price: Decimal,
    market_info: MarketInfo,
    logger: logging.Logger
) -> Decimal | None:
    """Calculates the position size based on risk percentage, stop loss distance,
    and account balance, respecting market limits and precision.
    Returns size in base currency (spot) or contracts (futures).
    """
    lg = logger
    lg.info(f"{BRIGHT}--- Calculating Position Size ({symbol}) ---{RESET}")

    # --- Input Validation ---
    quote_currency = market_info.get('quote', QUOTE_CURRENCY)
    size_unit = "Contracts" if market_info['is_contract'] else market_info.get('base', 'BASE')

    if not (isinstance(balance, Decimal) and balance.is_finite() and balance > 0):
        lg.error(f"Sizing failed ({symbol}): Invalid or non-positive balance ({balance}).")
        return None
    if not (isinstance(risk_decimal, Decimal) and risk_decimal.is_finite() and 0 < risk_decimal <= 1):
        lg.error(f"Sizing failed ({symbol}): Invalid risk fraction ({risk_decimal}). Must be > 0 and <= 1.")
        return None
    if not (isinstance(entry_price, Decimal) and entry_price.is_finite() and entry_price > 0):
        lg.error(f"Sizing failed ({symbol}): Invalid Entry price ({entry_price}).")
        return None
    if not (isinstance(initial_stop_loss_price, Decimal) and initial_stop_loss_price.is_finite() and initial_stop_loss_price > 0):
        lg.error(f"Sizing failed ({symbol}): Invalid Stop Loss price ({initial_stop_loss_price}).")
        return None
    if initial_stop_loss_price == entry_price:
        lg.error(f"Sizing failed ({symbol}): Entry ({entry_price}) and Stop Loss ({initial_stop_loss_price}) cannot be equal.")
        return None

    # --- Get Market Constraints ---
    try:
        amount_step = market_info['amount_precision_step_decimal']
        price_step = market_info['price_precision_step_decimal']
        min_amount = market_info['min_amount_decimal']
        max_amount = market_info['max_amount_decimal']
        min_cost = market_info['min_cost_decimal']
        max_cost = market_info['max_cost_decimal']
        contract_size = market_info['contract_size_decimal']
        is_inverse = market_info['is_inverse']

        if amount_step is None or not (amount_step.is_finite() and amount_step > 0):
            raise ValueError("Amount precision step (tick size) is invalid or missing.")
        if price_step is None or not (price_step.is_finite() and price_step > 0):
             raise ValueError("Price precision step (tick size) is invalid or missing.")
        if contract_size is None or not (contract_size.is_finite() and contract_size > 0):
            raise ValueError("Contract size is invalid or missing.")

        # Use effective limits (handle None)
        min_amount_eff = min_amount if min_amount is not None and min_amount >= 0 else Decimal('0')
        max_amount_eff = max_amount if max_amount is not None and max_amount > 0 else Decimal('inf')
        min_cost_eff = min_cost if min_cost is not None and min_cost >= 0 else Decimal('0')
        max_cost_eff = max_cost if max_cost is not None and max_cost > 0 else Decimal('inf')

        lg.debug(f"  Market Constraints ({symbol}):")
        lg.debug(f"    Amount Step: {amount_step}, Min Amt: {min_amount_eff}, Max Amt: {max_amount_eff}")
        lg.debug(f"    Price Step: {price_step}")
        lg.debug(f"    Cost Min: {min_cost_eff}, Cost Max: {max_cost_eff}")
        lg.debug(f"    Contract Size: {contract_size}, Type: {market_info['contract_type_str']}")

    except (KeyError, ValueError, TypeError) as e:
        lg.error(f"Sizing failed ({symbol}): Error accessing required market details: {e}")
        return None

    # --- Calculation ---
    risk_amount_quote = (balance * risk_decimal).quantize(Decimal('1e-8'), ROUND_DOWN)  # Risk amount in quote currency
    stop_loss_distance = abs(entry_price - initial_stop_loss_price)

    if stop_loss_distance <= Decimal('0'):
        lg.error(f"Sizing failed ({symbol}): Stop loss distance is zero or negative ({stop_loss_distance}). Check entry/SL prices.")
        return None

    lg.info("  Inputs:")
    lg.info(f"    Balance: {balance.normalize()} {quote_currency}")
    lg.info(f"    Risk %: {risk_decimal:.2%}")
    lg.info(f"    Risk Amt: {risk_amount_quote.normalize()} {quote_currency}")
    lg.info(f"    Entry Price: {entry_price.normalize()}")
    lg.info(f"    Stop Loss Price: {initial_stop_loss_price.normalize()}")
    lg.info(f"    SL Distance (Price): {stop_loss_distance.normalize()}")

    calculated_size = Decimal('NaN')
    try:
        if not is_inverse:  # Spot or Linear Contract
            # Risk per unit/contract = SL distance * Contract Size (value change per unit/contract for 1 price point move)
            risk_per_unit = stop_loss_distance * contract_size
            if risk_per_unit <= Decimal('1e-18'):  # Avoid division by zero or near-zero
                 raise ZeroDivisionError(f"Risk per unit is too small ({risk_per_unit}). Stop distance might be too small relative to contract size.")
            calculated_size = risk_amount_quote / risk_per_unit
            lg.debug("  Linear/Spot Calc: Size = RiskAmt / (SL_Dist * ContractSize)")
            lg.debug(f"    = {risk_amount_quote.normalize()} / ({stop_loss_distance.normalize()} * {contract_size.normalize()}) = {calculated_size}")
        else:  # Inverse Contract
            # Risk per contract (in quote) = Contract Size / Entry Price - Contract Size / SL Price (approx = Contract Size * SL Distance / (Entry * SL))
            # Simplified: Value of 1 contract = ContractSize / Price. Risk per contract = abs(Value@Entry - Value@SL)
            # More direct: How many contracts such that (ContractSize * NumContracts * SL_Distance / (Entry * SL)) = RiskAmtQuote
            # Or: RiskAmtQuote = NumContracts * ContractSize * abs(1/SL - 1/Entry)
            # NumContracts = RiskAmtQuote / (ContractSize * abs(1/SL - 1/Entry))
            value_change_per_contract = contract_size * abs(Decimal('1') / initial_stop_loss_price - Decimal('1') / entry_price)
            if value_change_per_contract <= Decimal('1e-18'):
                 raise ZeroDivisionError(f"Inverse value change per contract is too small ({value_change_per_contract}).")
            calculated_size = risk_amount_quote / value_change_per_contract
            lg.debug("  Inverse Calc: Size = RiskAmt / (ContractSize * abs(1/SL - 1/Entry))")
            lg.debug(f"    = {risk_amount_quote.normalize()} / ({contract_size.normalize()} * abs(1/{initial_stop_loss_price.normalize()} - 1/{entry_price.normalize()})) = {calculated_size}")

    except (InvalidOperation, OverflowError, ZeroDivisionError) as e:
        lg.error(f"Sizing failed ({symbol}): Calculation error: {e}.")
        return None

    if not calculated_size.is_finite() or calculated_size <= Decimal('0'):
        lg.error(f"Sizing failed ({symbol}): Calculated size is invalid or non-positive ({calculated_size}).")
        return None

    lg.info(f"  Initial Calculated Size ({symbol}) = {calculated_size.normalize()} {size_unit}")

    # --- Apply Constraints and Precision ---
    adjusted_size = calculated_size
    adjustment_reason = []

    # Function to estimate cost (can be complex for inverse)
    def estimate_cost(size: Decimal, price: Decimal) -> Decimal | None:
        if not (isinstance(size, Decimal) and size.is_finite() and size > 0): return None
        if not (isinstance(price, Decimal) and price.is_finite() and price > 0): return None
        try:
            if not is_inverse:  # Spot or Linear
                cost = size * contract_size * price
            else:  # Inverse
                cost = size * contract_size / price
            return cost.quantize(Decimal('1e-8'), ROUND_UP)  # Cost usually in quote currency
        except (InvalidOperation, OverflowError, ZeroDivisionError) as cost_err:
            lg.error(f"Cost estimation failed: {cost_err}")
            return None

    # 1. Min/Max Amount Limits
    if adjusted_size < min_amount_eff:
        adjustment_reason.append(f"Adjusted UP to Min Amount {min_amount_eff.normalize()}")
        adjusted_size = min_amount_eff
    if adjusted_size > max_amount_eff:
        adjustment_reason.append(f"Adjusted DOWN to Max Amount {max_amount_eff.normalize()}")
        adjusted_size = max_amount_eff

    if adjusted_size != calculated_size:
        lg.debug(f"  Size after Amount Limits: {adjusted_size.normalize()} {size_unit} ({'; '.join(adjustment_reason)})")

    if not adjusted_size.is_finite() or adjusted_size <= Decimal('0'):
        lg.error(f"Sizing failed ({symbol}): Size became invalid after Amount limits ({adjusted_size}). Min amount might be too high.")
        return None

    # 2. Min/Max Cost Limits (Estimate cost based on adjusted size)
    cost_adjustment_reason = []
    size_before_cost_limits = adjusted_size
    estimated_cost = estimate_cost(adjusted_size, entry_price)

    if estimated_cost is not None:
        lg.debug(f"  Estimated Cost for size {adjusted_size.normalize()}: {estimated_cost.normalize()} {quote_currency}")
        if estimated_cost < min_cost_eff:
            # This is tricky. If min cost is required, we can't simply place the trade.
            # We could try increasing size to meet min cost, but that violates the risk %
            lg.error(f"Sizing failed ({symbol}): Estimated cost {estimated_cost.normalize()} is below Min Cost {min_cost_eff.normalize()}. Cannot meet Min Cost without exceeding risk. Increase risk % or balance.")
            return None  # Cannot proceed safely

        if estimated_cost > max_cost_eff:
            cost_adjustment_reason.append(f"Cost {estimated_cost.normalize()} > Max Cost {max_cost_eff.normalize()}")
            # Reduce size to meet max cost limit
            try:
                if not is_inverse:
                     denominator = entry_price * contract_size
                     if denominator <= 0: raise ZeroDivisionError("Invalid price or contract size for max cost calc.")
                     max_size_for_max_cost = max_cost_eff / denominator
                else:  # Inverse
                     if entry_price <= 0: raise ZeroDivisionError("Invalid entry price for inverse max cost calc.")
                     max_size_for_max_cost = max_cost_eff * entry_price / contract_size

                if not max_size_for_max_cost.is_finite() or max_size_for_max_cost <= 0:
                    raise ValueError(f"Invalid max size calculated for max cost ({max_size_for_max_cost}).")

                new_adjusted_size = min(adjusted_size, max_size_for_max_cost)

                # Re-check against min amount after adjusting for max cost
                if new_adjusted_size < min_amount_eff:
                    lg.error(f"Sizing failed ({symbol}): Size reduced to {new_adjusted_size.normalize()} to meet Max Cost, but this is below Min Amount {min_amount_eff.normalize()}.")
                    return None

                adjusted_size = new_adjusted_size
                cost_adjustment_reason.append(f"Adjusted DOWN to {adjusted_size.normalize()} to meet Max Cost")
            except (InvalidOperation, OverflowError, ZeroDivisionError, ValueError) as e:
                lg.error(f"Sizing failed ({symbol}): Error applying Max Cost limit: {e}.")
                return None

    if adjusted_size != size_before_cost_limits:
        lg.debug(f"  Size after Cost Limits: {adjusted_size.normalize()} {size_unit} ({'; '.join(cost_adjustment_reason)})")

    if not adjusted_size.is_finite() or adjusted_size <= Decimal('0'):
        lg.error(f"Sizing failed ({symbol}): Size became invalid after Cost limits ({adjusted_size}).")
        return None

    # 3. Apply Amount Precision (Tick Size) - Round DOWN
    final_size = adjusted_size
    try:
        if amount_step <= 0: raise ValueError("Amount step size must be positive.")
        # Calculate number of steps and multiply back to ensure conformity
        num_steps = (adjusted_size / amount_step).to_integral_value(rounding=ROUND_DOWN)
        final_size = num_steps * amount_step

        if final_size != adjusted_size:
             lg.info(f"  Applied Amount Precision: Rounded size from {adjusted_size.normalize()} down to {final_size.normalize()} {size_unit} (Step: {amount_step})")

    except (InvalidOperation, ValueError, ZeroDivisionError) as e:
        lg.error(f"Sizing failed ({symbol}): Error applying amount precision (step={amount_step}): {e}.")
        return None

    # --- Final Validation ---
    if not final_size.is_finite() or final_size <= Decimal('0'):
        lg.error(f"Sizing failed ({symbol}): Final size is invalid or zero after precision ({final_size}).")
        return None
    if final_size < min_amount_eff:
        # This check is crucial after rounding down
        lg.error(f"Sizing failed ({symbol}): Final size {final_size.normalize()} is below Min Amount {min_amount_eff.normalize()} after precision adjustment.")
        return None
    if final_size > max_amount_eff:
        # Should not happen if max amount was applied correctly before, but good failsafe
        lg.error(f"Sizing failed ({symbol}): Final size {final_size.normalize()} exceeds Max Amount {max_amount_eff.normalize()}.")
        return None

    # Final cost check (optional but recommended)
    final_cost = estimate_cost(final_size, entry_price)
    if final_cost is not None:
        lg.debug(f"  Final Estimated Cost: {final_cost.normalize()} {quote_currency}")
        if final_cost < min_cost_eff:
            # This could happen if rounding down size made cost dip below min cost
            lg.error(f"Sizing failed ({symbol}): Final cost {final_cost.normalize()} is below Min Cost {min_cost_eff.normalize()} after precision adjustment.")
            return None
        if final_cost > max_cost_eff:
            lg.error(f"Sizing failed ({symbol}): Final cost {final_cost.normalize()} exceeds Max Cost {max_cost_eff.normalize()}.")
            return None

    # --- Success ---
    lg.info(f"{NEON_GREEN}{BRIGHT}>>> Final Calculated Size ({symbol}): {final_size.normalize()} {size_unit} <<< {RESET}")
    if final_cost:
        lg.info(f"    Estimated Cost: {final_cost.normalize()} {quote_currency}")
    all_adjustments = '; '.join(filter(None, [', '.join(adjustment_reason), ', '.join(cost_adjustment_reason)]))
    if all_adjustments:
        lg.info(f"    Adjustments Applied: {all_adjustments}")
    lg.info(f"{BRIGHT}--- End Position Sizing ({symbol}) ---{RESET}")
    return final_size


def cancel_order(exchange: ccxt.Exchange, order_id: str, symbol: str, logger: logging.Logger) -> bool:
    """Cancels a specific order with retries."""
    lg = logger
    attempts = 0
    last_exception: Exception | None = None
    lg.info(f"Attempting to cancel order ID '{order_id}' for {symbol}...")

    # Prepare params, especially for Bybit V5 which requires category/symbol
    params = {}
    is_bybit = 'bybit' in exchange.id.lower()
    if is_bybit:
        try:
            # Attempt to get market details to determine category
            market = exchange.market(symbol)
            market_id = market['id']
            # Determine category based on market type
            category = 'spot' if market.get('spot') else \
                       'linear' if market.get('linear') else \
                       'inverse' if market.get('inverse') else \
                       'option' if market.get('option') else None
            if category:
                params = {'category': category, 'symbol': market_id}
                lg.debug(f"Using Bybit V5 params for cancel: {params}")
            else:
                 lg.warning(f"Could not determine Bybit category for {symbol}. Attempting cancel without category param.")
        except (ccxt.ExchangeError, KeyError) as e:
            lg.warning(f"Could not get market details for {symbol} to set Bybit params: {e}. Proceeding without specific params.")

    order_id_str = str(order_id)  # Ensure it's a string

    while attempts <= MAX_API_RETRIES:
        try:
            exchange.cancel_order(order_id_str, symbol, params=params)
            lg.info(f"{NEON_GREEN}Successfully cancelled order {order_id_str} ({symbol}).{RESET}")
            return True
        except ccxt.OrderNotFound:
            lg.warning(f"{NEON_YELLOW}Order '{order_id_str}' ({symbol}) not found on exchange. Assuming already cancelled or filled.{RESET}")
            return True  # Treat as success if not found
        except ccxt.InvalidOrder as e:
            # E.g., order already filled/cancelled status prevents cancellation
            lg.warning(f"{NEON_YELLOW}Cannot cancel order '{order_id_str}' ({symbol}): {e}. Assuming completed or already cancelled.{RESET}")
            return True  # Treat as success if invalid state (likely filled/cancelled)
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error cancelling order {order_id_str} ({symbol}): {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            # Use exponential backoff for rate limits
            wait_time = RETRY_DELAY_SECONDS * (2 ** attempts + 1)
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded cancelling order {order_id_str} ({symbol}): {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            # Don't increment attempts here, rate limit wait replaces standard delay
            continue
        except ccxt.AuthenticationError as e:
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error cancelling order {order_id_str} ({symbol}): {e}. Cannot continue.{RESET}")
            # Potentially trigger shutdown or alert
            return False  # Fatal error for this operation
        except ccxt.ExchangeError as e:
             last_exception = e
             lg.error(f"{NEON_RED}Exchange error cancelling order {order_id_str} ({symbol}): {e}{RESET}")
             # Decide if retryable based on error type if possible
             # For now, retry generic exchange errors
        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error cancelling order {order_id_str} ({symbol}): {e}{RESET}", exc_info=True)
            # Stop retrying on unexpected errors
            return False

        # Standard retry delay (exponential backoff)
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            delay = RETRY_DELAY_SECONDS * (2 ** (attempts - 1))
            lg.info(f"Waiting {delay}s before retry...")
            time.sleep(delay)

    # If loop finishes without success
    lg.error(f"{NEON_RED}Failed to cancel order {order_id_str} ({symbol}) after {MAX_API_RETRIES + 1} attempts. Last error: {last_exception}{RESET}")
    return False


def place_trade(
    exchange: ccxt.Exchange,
    symbol: str,
    trade_signal: str,  # 'BUY', 'SELL', 'EXIT_LONG', 'EXIT_SHORT'
    position_size: Decimal,  # Size in base currency or contracts
    market_info: MarketInfo,
    logger: logging.Logger,
    reduce_only: bool = False,
    params: dict[str, Any] | None = None  # Extra params for create_order
) -> dict | None:
    """Places a market order with appropriate parameters, handling retries.
    Returns the order result dict from ccxt if successful, else None.
    """
    lg = logger

    # Map signal to order side
    side_map = {
        "BUY": "buy",
        "SELL": "sell",
        "EXIT_LONG": "sell",  # Selling to close a long
        "EXIT_SHORT": "buy"  # Buying to close a short
    }
    side = side_map.get(trade_signal.upper())

    # --- Input Validation ---
    if side is None:
        lg.error(f"Invalid trade signal '{trade_signal}' provided for {symbol}.")
        return None
    if not isinstance(position_size, Decimal) or not position_size.is_finite() or position_size <= 0:
        lg.error(f"Invalid position size '{position_size}' provided for {symbol}. Must be positive Decimal.")
        return None

    order_type = 'market'  # Using market orders for simplicity
    is_contract = market_info['is_contract']
    base_currency = market_info.get('base', 'BASE')
    size_unit = "Contracts" if is_contract else base_currency
    action_desc = f"{trade_signal} ({'Reduce-Only' if reduce_only else 'Open/Increase'})"
    market_id = market_info.get('id')

    if not market_id:
        lg.error(f"Missing market ID for {symbol} in market_info.")
        return None

    # --- Prepare Order Arguments ---
    # Ensure size conforms to amount precision *before* converting to float
    final_size_decimal = position_size
    try:
        amount_step = market_info['amount_precision_step_decimal']
        if amount_step is None or amount_step <= 0:
            raise ValueError("Invalid amount precision step.")

        num_steps = (final_size_decimal / amount_step).to_integral_value(rounding=ROUND_DOWN)
        rounded_size_decimal = num_steps * amount_step

        if rounded_size_decimal <= 0:
             # This might happen if the calculated size is smaller than the step size
             lg.error(f"Position size {position_size.normalize()} is smaller than the minimum amount step {amount_step.normalize()} for {symbol}.")
             # Check if original size was >= min_amount, if so, maybe round up? Risky.
             # Safest is to fail here.
             min_amount = market_info.get('min_amount_decimal', Decimal('0'))
             if position_size >= min_amount:
                 lg.warning(f"Calculated size {position_size.normalize()} meets min amount {min_amount.normalize()} but rounds down to zero due to step size {amount_step.normalize()}. Trying to use min amount instead.")
                 # Attempt to use the minimum amount if possible
                 num_min_steps = (min_amount / amount_step).to_integral_value(rounding=ROUND_UP)  # Round up for min amount
                 rounded_size_decimal = num_min_steps * amount_step
                 if rounded_size_decimal < min_amount:  # Failsafe
                      lg.error("Failed to adjust size to minimum amount correctly.")
                      return None
                 lg.info(f"Adjusted size to minimum amount: {rounded_size_decimal.normalize()} {size_unit}")
             else:
                  lg.error(f"Size {position_size.normalize()} rounded down to zero or less by amount step {amount_step.normalize()}. Cannot place trade.")
                  return None

        if rounded_size_decimal != final_size_decimal:
            lg.warning(f"Adjusted order size from {final_size_decimal.normalize()} to {rounded_size_decimal.normalize()} {size_unit} due to amount precision step {amount_step.normalize()}.")
            final_size_decimal = rounded_size_decimal

        # Convert final Decimal size to float for CCXT API call
        amount_float = float(final_size_decimal)
        # Check for potential issues with float conversion (very small numbers)
        if abs(amount_float) < 1e-15:
            raise ValueError(f"Final size {final_size_decimal.normalize()} converts to near-zero float ({amount_float}).")

    except (ValueError, TypeError, InvalidOperation) as e:
        lg.error(f"Failed to apply precision or convert size {position_size.normalize()} for {symbol}: {e}")
        return None

    # Base arguments for create_order
    order_args: dict[str, Any] = {
        'symbol': symbol,
        'type': order_type,
        'side': side,
        'amount': amount_float,  # Use the float amount
    }

    # --- Exchange-Specific Parameters (Bybit V5 Example) ---
    order_params: dict[str, Any] = {}
    is_bybit = 'bybit' in exchange.id.lower()
    if is_bybit and is_contract:
        try:
            category = market_info['contract_type_str'].lower()  # 'linear' or 'inverse'
            order_params = {
                'category': category,
                'positionIdx': 0,  # Assume one-way mode (0 for hedge mode main position)
                # Add other necessary Bybit V5 params if needed
            }
            if reduce_only:
                order_params['reduceOnly'] = True
                # Bybit often requires Time In Force for reduceOnly orders (e.g., IOC)
                order_params['timeInForce'] = 'IOC'  # ImmediateOrCancel is common
                lg.debug("Bybit V5 params: reduceOnly=True, timeInForce='IOC'.")
            # Add SL/TP directly to order if API supports it (Bybit V5 market orders usually don't, set separately)
            # if initial_sl: order_params['stopLoss'] = _format_price(exchange, symbol, initial_sl) ... etc

        except Exception as e:
            lg.error(f"Failed to set Bybit V5 params for {symbol} order: {e}. Proceeding without them.")
            order_params = {}  # Reset params on error

    # Merge external params if provided (external params take precedence)
    if params and isinstance(params, dict):
        lg.debug(f"Merging external params into order: {params}")
        order_params.update(params)

    if order_params:
        order_args['params'] = order_params

    # --- Log Intent and Execute ---
    lg.warning(f"{BRIGHT}===> Placing Trade ({action_desc}) <==={RESET}")
    lg.warning(f"  Symbol: {symbol} (Market ID: {market_id})")
    lg.warning(f"  Type: {order_type.upper()}")
    lg.warning(f"  Side: {side.upper()} ({trade_signal})")
    lg.warning(f"  Size: {final_size_decimal.normalize()} {size_unit} (Float: {amount_float})")
    if order_args.get('params'):
        lg.warning(f"  Params: {order_args['params']}")

    attempts = 0
    last_exception: Exception | None = None
    order_result: dict | None = None

    while attempts <= MAX_API_RETRIES:
        try:
            order_result = exchange.create_order(**order_args)

            # --- Process Result ---
            if order_result:
                 order_id = order_result.get('id', 'N/A')
                 status = order_result.get('status', 'unknown')
                 avg_price_raw = order_result.get('average')
                 filled_raw = order_result.get('filled')

                 avg_price_dec = _safe_market_decimal(avg_price_raw, 'order.average', allow_zero=True)
                 filled_dec = _safe_market_decimal(filled_raw, 'order.filled', allow_zero=True)

                 log_msg_parts = [
                     f"{NEON_GREEN}{action_desc} Order Placed!{RESET}",
                     f"ID: {order_id}",
                     f"Status: {status}"
                 ]
                 if avg_price_dec is not None and avg_price_dec > 0:
                      log_msg_parts.append(f"Avg Fill Price: ~{avg_price_dec.normalize()}")
                 if filled_dec is not None:
                      log_msg_parts.append(f"Filled: {filled_dec.normalize()} {size_unit}")

                 lg.info(" ".join(log_msg_parts))
                 lg.debug(f"Raw Order Result ({symbol}): {json.dumps(order_result, indent=2)}")

                 # Send notification on success
                 send_notification(
                     f"Trade Executed: {symbol} {action_desc}",
                     f"Order ID: {order_id}\nStatus: {status}\nSize: {final_size_decimal.normalize()} {size_unit}"
                     f"{f'\nAvg Price: ~{avg_price_dec.normalize()}' if avg_price_dec else ''}",
                     lg
                 )
                 break  # Exit retry loop on success
            else:
                 # Should not happen if create_order doesn't raise exception, but handle defensively
                 lg.error(f"Order placement for {symbol} returned empty result without exception.")
                 last_exception = ValueError("Empty order result received.")
                 # Continue retrying

        # --- Specific Error Handling ---
        except ccxt.InsufficientFunds as e:
            last_exception = e
            lg.error(f"{NEON_RED}Order Failed ({symbol}): Insufficient Funds. Check balance and leverage. {e}{RESET}")
            return None  # Non-retryable error
        except ccxt.InvalidOrder as e:
            last_exception = e
            lg.error(f"{NEON_RED}Order Failed ({symbol}): Invalid Order parameters. Check size, price, limits. {e}{RESET}")
            lg.error(f"  Order Arguments: {order_args}")
            return None  # Non-retryable error (usually config/logic issue)
        except ccxt.ExchangeNotAvailable as e:
             last_exception = e
             lg.warning(f"{NEON_YELLOW}Exchange not available placing order for {symbol}: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error placing order for {symbol}: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * (2 ** attempts + 2)  # Longer backoff for rate limits
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded placing order for {symbol}: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue  # Continue without incrementing attempts immediately
        except ccxt.AuthenticationError as e:
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication error placing order for {symbol}: {e}. Cannot continue.{RESET}")
            return None  # Fatal error
        except ccxt.ExchangeError as e:
             last_exception = e
             lg.error(f"{NEON_RED}Generic exchange error placing order for {symbol}: {e}{RESET}")
             # Potentially inspect error message for retry decision, for now retry
        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error placing order for {symbol}: {e}{RESET}", exc_info=True)
            return None  # Stop on unexpected errors

        # Standard retry delay
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            delay = RETRY_DELAY_SECONDS * (2 ** (attempts - 1))
            lg.info(f"Waiting {delay}s before retry...")
            time.sleep(delay)

    # --- Check final outcome ---
    if order_result is None:
        lg.error(f"{NEON_RED}Failed to place order for {symbol} after {MAX_API_RETRIES + 1} attempts. Last error: {last_exception}{RESET}")
        return None

    return order_result


def set_position_protection(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: MarketInfo,
    # position_info: PositionInfo, # Position info might not be needed if setting via API call directly
    logger: logging.Logger,
    stop_loss_price: Decimal | None = None,
    take_profit_price: Decimal | None = None,
    trailing_stop_distance: Decimal | None = None,  # Distance value (price difference)
    tsl_activation_price: Decimal | None = None  # Price at which TSL becomes active
) -> bool:
    """Sets Stop Loss, Take Profit, and/or Trailing Stop Loss for an existing position.
    Currently implemented primarily for Bybit V5 API.
    Returns True if successful or no action needed, False on failure.
    Note: Pass price=0 or distance=0 to *remove* existing protection.
    """
    lg = logger
    lg.debug(f"Attempting to set protection for position {symbol}...")

    log_parts = []
    has_changes = False
    if stop_loss_price is not None:
        log_parts.append(f"SL={stop_loss_price.normalize() if stop_loss_price > 0 else 'REMOVE'}")
        has_changes = True
    if take_profit_price is not None:
        log_parts.append(f"TP={take_profit_price.normalize() if take_profit_price > 0 else 'REMOVE'}")
        has_changes = True
    if trailing_stop_distance is not None:
        log_parts.append(f"TSL Dist={trailing_stop_distance.normalize() if trailing_stop_distance > 0 else 'REMOVE'}")
        has_changes = True
    if tsl_activation_price is not None:
        log_parts.append(f"TSL Act={tsl_activation_price.normalize() if tsl_activation_price > 0 else 'REMOVE'}")
        has_changes = True

    if not has_changes:
        lg.debug(f"No protection parameters provided for {symbol}. No action taken.")
        return True

    lg.info(f"  Target protections for {symbol}: {', '.join(log_parts)}")

    is_bybit = 'bybit' in exchange.id.lower()

    # --- Bybit V5 Implementation ---
    # Uses the `POST /v5/position/set-trading-stop` endpoint
    if is_bybit:
        try:
            market_id = market_info['id']
            category = market_info['contract_type_str'].lower()  # 'linear' or 'inverse'
            if category not in ['linear', 'inverse']:
                 lg.error(f"Protection setting only supported for Bybit linear/inverse contracts, not {category} for {symbol}.")
                 return False

            # --- Prepare Parameters ---
            params = {
                'category': category,
                'symbol': market_id,
                'positionIdx': 0,  # Assume one-way mode
                'tpslMode': 'Full',  # Or 'Partial' - affects how TP/SL applies to position size
                # Trigger prices based on Mark Price are generally recommended to avoid manipulation issues
                'slTriggerBy': 'MarkPrice',
                'tpTriggerBy': 'MarkPrice',
                # Order types when triggered (Market is usually most reliable)
                'slOrderType': 'Market',
                'tpOrderType': 'Market',
            }
            param_added = False  # Track if any protection values are actually being set/changed

            # Stop Loss
            if stop_loss_price is not None:
                if stop_loss_price <= 0:  # Request to remove SL
                    params['stopLoss'] = '0'
                else:
                    sl_str = _format_price(exchange, symbol, stop_loss_price)
                    if sl_str: params['stopLoss'] = sl_str
                    else: raise ValueError(f"Invalid SL price format: {stop_loss_price}")
                param_added = True

            # Take Profit
            if take_profit_price is not None:
                if take_profit_price <= 0:  # Request to remove TP
                    params['takeProfit'] = '0'
                else:
                    tp_str = _format_price(exchange, symbol, take_profit_price)
                    if tp_str: params['takeProfit'] = tp_str
                    else: raise ValueError(f"Invalid TP price format: {take_profit_price}")
                param_added = True

            # Trailing Stop Loss Distance
            if trailing_stop_distance is not None:
                if trailing_stop_distance <= 0:  # Request to remove TSL
                    params['trailingStop'] = '0'
                    # Also remove activation price if removing TSL
                    if 'activePrice' not in params: params['activePrice'] = '0'
                else:
                    # Bybit TSL distance is a price *value*, not percentage. Needs formatting.
                    # Use the price tick size for formatting the distance.
                    price_tick = market_info.get('price_precision_step_decimal')
                    if not price_tick or price_tick <= 0:
                        raise ValueError("Invalid price tick size needed for TSL distance formatting.")
                    # Format distance like a price difference
                    ts_dist_str = str(trailing_stop_distance.quantize(price_tick, rounding=ROUND_UP))  # Round up distance slightly? Or match tick?
                    # Validate the formatted string represents a positive number
                    if _safe_market_decimal(ts_dist_str, "tsl_dist_str", allow_zero=False, allow_negative=False):
                         params['trailingStop'] = ts_dist_str
                    else:
                         raise ValueError(f"Invalid TSL distance format after quantization: {ts_dist_str}")
                param_added = True

            # Trailing Stop Activation Price
            if tsl_activation_price is not None:
                if tsl_activation_price <= 0:  # Request to remove activation price (or set TSL inactive)
                     params['activePrice'] = '0'
                else:
                    act_str = _format_price(exchange, symbol, tsl_activation_price)
                    if act_str: params['activePrice'] = act_str
                    else: raise ValueError(f"Invalid TSL activation price format: {tsl_activation_price}")
                param_added = True

            # --- Make API Call ---
            if param_added:
                lg.debug(f"Calling Bybit V5 set_trading_stop with params: {params}")
                # Use the unified private_post method
                response = exchange.private_post_position_set_trading_stop(params)
                lg.debug(f"Bybit set_trading_stop response: {response}")

                # Check Bybit V5 response structure
                if isinstance(response, dict) and response.get('retCode') == 0:
                    lg.info(f"{NEON_GREEN}Successfully set protection for {symbol}.{RESET}")
                    send_notification(
                        f"Protection Updated: {symbol}",
                        f"Applied protections: {', '.join(log_parts)}",
                        lg
                    )
                    return True
                else:
                    error_msg = response.get('retMsg', 'Unknown error') if isinstance(response, dict) else 'Invalid response format'
                    lg.error(f"Failed to set protection for {symbol} via Bybit API: {error_msg} (Code: {response.get('retCode', 'N/A')})")
                    lg.debug(f"Request Params: {params}")
                    return False
            else:
                # This case should have been caught earlier, but included for safety
                lg.debug("No actual protection parameters were provided or formatted correctly.")
                return True  # No action needed is considered success

        except (ccxt.ExchangeError, ccxt.NetworkError, ValueError, TypeError, InvalidOperation) as e:
            lg.error(f"Error setting protection for {symbol} using Bybit V5 API: {e}", exc_info=True)
            return False
        except Exception as e:
             lg.error(f"Unexpected error during Bybit V5 protection setting for {symbol}: {e}", exc_info=True)
             return False

    # --- Fallback for other exchanges ---
    else:
        lg.warning(f"Automated protection setting (SL/TP/TSL) via API is not explicitly implemented for exchange '{exchange.id}'. Manual setting or exchange UI might be required.")
        # You might try placing separate SL/TP limit/stop orders here as a fallback,
        # but managing those is complex (cancelling old ones, ensuring only one triggers, etc.)
        # For now, return False indicating it wasn't handled automatically.
        return False


# --- Backtesting Framework ---
def run_backtest(
    exchange: ccxt.Exchange,
    symbol: str,
    start_date_str: str,
    end_date_str: str,
    config: dict[str, Any],
    logger: logging.Logger
) -> BacktestResult | None:
    """Runs a basic backtest of the strategy on historical data."""
    lg = logger
    lg.info(f"{BRIGHT}--- Starting Backtest for {symbol} ({start_date_str} to {end_date_str}) ---{RESET}")

    # --- Parameters ---
    strategy_params = config['strategy_params']
    protection_params = config['protection']
    risk_per_trade = Decimal(str(config['risk_per_trade']))
    Decimal(str(config['leverage']))  # For potential margin calculation if needed
    initial_balance = Decimal('10000')  # Starting virtual balance
    timeframe_str = config['interval']
    ccxt_timeframe = CCXT_INTERVAL_MAP[timeframe_str]
    fetch_limit = config['fetch_limit']

    # --- Fetch Historical Data ---
    try:
        start_dt = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end_date_str, "%Y-%m-%d").replace(tzinfo=UTC) + timedelta(days=1)  # Include end date
        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)

        all_ohlcv = []
        current_ts = start_ts
        lg.info(f"Fetching historical data for {symbol} ({ccxt_timeframe})...")
        while current_ts < end_ts:
             # Fetch in chunks, respecting exchange limits
             limit = min(fetch_limit, BYBIT_API_KLINE_LIMIT)  # Adhere to Bybit limit if applicable
             ohlcv = exchange.fetch_ohlcv(symbol, ccxt_timeframe, since=current_ts, limit=limit)
             if not ohlcv:
                 lg.info("No more data found or end date reached.")
                 break
             # Filter out data beyond the requested end date
             ohlcv = [candle for candle in ohlcv if candle[0] < end_ts]
             if not ohlcv: break

             all_ohlcv.extend(ohlcv)
             last_fetched_ts = ohlcv[-1][0]
             current_ts = last_fetched_ts + exchange.parse_timeframe(ccxt_timeframe) * 1000  # Move to next candle start
             lg.debug(f"Fetched {len(ohlcv)} candles up to {datetime.fromtimestamp(last_fetched_ts / 1000, tz=UTC)}")
             time.sleep(exchange.rateLimit / 1000)  # Respect rate limit

        if not all_ohlcv:
            lg.error(f"No historical data fetched for {symbol} in the specified range.")
            return None

        # Remove duplicates and sort
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.drop_duplicates(subset=['timestamp'], inplace=True)
        df.sort_values(by='timestamp', inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)  # Convert to datetime
        lg.info(f"Successfully fetched and prepared {len(df)} unique candles for {symbol}.")

    except (ccxt.ExchangeError, ValueError, TypeError) as e:
        lg.error(f"Failed to fetch or process historical data for {symbol}: {e}", exc_info=True)
        return None
    except Exception as e:
        lg.error(f"Unexpected error during data fetching for {symbol}: {e}", exc_info=True)
        return None

    # --- Backtest Simulation ---
    balance = initial_balance
    equity = initial_balance  # Tracks peak balance for drawdown calculation
    max_drawdown_pct = Decimal('0')
    trades = []  # List to store closed trade details
    position = None  # Current open position {'side': 'long'/'short', 'size': Decimal, 'entry_price': Decimal, 'sl': Decimal, 'tp': Decimal, 'entry_ts': Timestamp}
    market_info = enhance_market_info(exchange.market(symbol))  # Get market details for sizing/precision

    lg.info(f"Simulating strategy on {len(df)} candles...")
    for i in range(1, len(df)):  # Start from 1 to have previous candle data if needed
        current_candle = df.iloc[i]
        current_ts = current_candle['timestamp']
        current_close = Decimal(str(current_candle['close']))
        current_high = Decimal(str(current_candle['high']))
        current_low = Decimal(str(current_candle['low']))

        # --- Check for SL/TP Hit within the current candle ---
        if position:
            pnl = Decimal('0')
            exit_price = None
            exit_reason = None

            if position['side'] == 'long':
                # Check SL first (most important)
                if current_low <= position['sl']:
                    exit_price = position['sl']  # Assume SL filled at SL price
                    exit_reason = "Stop Loss"
                # Check TP
                elif position['tp'] and current_high >= position['tp']:
                    exit_price = position['tp']  # Assume TP filled at TP price
                    exit_reason = "Take Profit"

            elif position['side'] == 'short':
                # Check SL
                if current_high >= position['sl']:
                    exit_price = position['sl']
                    exit_reason = "Stop Loss"
                # Check TP
                elif position['tp'] and current_low <= position['tp']:
                    exit_price = position['tp']
                    exit_reason = "Take Profit"

            if exit_price and exit_reason:
                # --- Close Position ---
                if position['side'] == 'long':
                    pnl = (exit_price - position['entry_price']) * position['size'] * market_info['contract_size_decimal']
                else:  # Short
                    pnl = (position['entry_price'] - exit_price) * position['size'] * market_info['contract_size_decimal']

                # Simple commission estimate (e.g., 0.06% taker fee)
                commission = abs(exit_price * position['size'] * market_info['contract_size_decimal'] * Decimal('0.0006'))
                pnl -= commission
                balance += pnl

                trades.append({
                    'entry_ts': position['entry_ts'],
                    'exit_ts': current_ts,
                    'symbol': symbol,
                    'side': position['side'],
                    'size': position['size'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'sl': position['sl'],
                    'tp': position['tp'],
                    'pnl': pnl,
                    'commission': commission,
                    'exit_reason': exit_reason,
                    'balance_after': balance
                })
                lg.debug(f"[{current_ts}] Closed {position['side']} @ {exit_price.normalize()} ({exit_reason}). PnL: {pnl.normalize():.2f}, Balance: {balance.normalize():.2f}")
                position = None  # Reset position

                # Update Equity and Drawdown
                equity = max(equity, balance)
                drawdown = (equity - balance) / equity if equity > 0 else Decimal('0')
                max_drawdown_pct = max(max_drawdown_pct, drawdown)

        # --- Generate Signal based on data up to the *previous* candle's close ---
        # Use data up to index `i` (exclusive of current candle `i`) for signal generation
        # This prevents lookahead bias (using current candle's close/high/low for entry decisions)
        analysis_df = df.iloc[:i].copy()  # Data up to the end of the previous candle
        if len(analysis_df) < 50: continue  # Ensure enough data for indicators

        analysis = analyze_market(analysis_df, strategy_params, symbol, lg)
        signal_result = generate_signal(analysis, strategy_params, protection_params, symbol, lg)
        signal = signal_result['signal']
        entry_price = current_close  # Assume entry at the close of the current candle for simplicity
        sl_price = signal_result['initial_sl']
        tp_price = signal_result['initial_tp']

        # --- Check for New Entry ---
        if not position and signal in ["BUY", "SELL"] and sl_price:
            # Calculate size based on current balance and proposed SL
            pos_size = calculate_position_size(
                exchange, symbol, balance, risk_per_trade,
                entry_price, sl_price, market_info, lg
            )

            if pos_size and pos_size > 0:
                # --- Open Position ---
                position = {
                    'side': 'long' if signal == "BUY" else 'short',
                    'size': pos_size,
                    'entry_price': entry_price,
                    'sl': sl_price,
                    'tp': tp_price,
                    'entry_ts': current_ts
                }
                # Estimate entry commission
                commission = abs(entry_price * pos_size * market_info['contract_size_decimal'] * Decimal('0.0006'))
                balance -= commission  # Deduct entry commission immediately

                lg.info(f"[{current_ts}] {BRIGHT}Opened {position['side']} {symbol}{RESET} @ {entry_price.normalize()}, Size: {pos_size.normalize()}, SL: {sl_price.normalize()}, TP: {tp_price.normalize() if tp_price else 'None'}, Comm: {commission.normalize():.2f}, Balance: {balance.normalize():.2f}")
            else:
                lg.debug(f"[{current_ts}] {signal} signal generated but position size calculation failed or resulted in zero size.")

    # --- Backtest Finished - Calculate Results ---
    lg.info("Backtest simulation finished. Calculating results...")
    total_trades = len(trades)
    if total_trades == 0:
        lg.warning("No trades were executed during the backtest.")
        return BacktestResult(total_trades=0, winning_trades=0, losing_trades=0, total_profit=Decimal('0'), max_drawdown=Decimal('0'), profit_factor=0.0)

    winning_trades = sum(1 for t in trades if t['pnl'] > 0)
    losing_trades = total_trades - winning_trades
    total_profit = sum(t['pnl'] for t in trades)
    total_fees = sum(t['commission'] for t in trades) * 2  # Entry + Exit fees approx

    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))

    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float('inf')  # Handle division by zero
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

    # Final Results
    result = BacktestResult(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        total_profit=total_profit,
        max_drawdown=max_drawdown_pct,
        profit_factor=profit_factor
    )

    # --- Log Summary ---
    summary = (
        f"\n{BRIGHT}--- Backtest Summary: {symbol} ({start_date_str} to {end_date_str}) ---\n"
        f"  Initial Balance: {initial_balance:.2f} {market_info.get('quote', 'USD')}\n"
        f"  Final Balance:   {balance:.2f} {market_info.get('quote', 'USD')}\n"
        f"  Total Net Profit:{NEON_GREEN if total_profit > 0 else NEON_RED} {total_profit:.2f}{RESET} ({((balance / initial_balance) - 1) * 100:.2f}%)\n"
        f"  Max Drawdown:    {max_drawdown_pct:.2%}\n"
        f"  Total Trades:    {total_trades}\n"
        f"  Winning Trades:  {winning_trades} ({win_rate:.2f}% Win Rate)\n"
        f"  Losing Trades:   {losing_trades}\n"
        f"  Profit Factor:   {profit_factor:.2f}\n"
        f"  Total Fees Est.: {total_fees:.2f}\n"
        f"{BRIGHT}--------------------------------------------------{RESET}"
    )
    lg.info(summary)

    # Optional: Save trades to CSV
    try:
         trades_df = pd.DataFrame(trades)
         trades_filename = os.path.join(LOG_DIRECTORY, f"backtest_trades_{symbol.replace('/', '_')}_{start_date_str}_{end_date_str}.csv")
         trades_df.to_csv(trades_filename, index=False)
         lg.info(f"Saved backtest trades log to: {trades_filename}")
    except Exception as e:
         lg.error(f"Failed to save backtest trades to CSV: {e}")

    # Send notification
    if config['notifications']['enable_notifications']:
        send_notification(
            f"Backtest Completed: {symbol}",
            f"Period: {start_date_str} to {end_date_str}\n"
            f"Trades: {total_trades}, Win Rate: {win_rate:.2f}%\n"
            f"Total Profit: {total_profit:.2f}\n"
            f"Max Drawdown: {max_drawdown_pct:.2%}\n"
            f"Profit Factor: {profit_factor:.2f}",
            lg
        )

    return result


# --- Main Execution Loop ---
def main_loop(exchange: ccxt.Exchange, config: dict[str, Any], logger: logging.Logger) -> None:
    """The main operational loop of the trading bot."""
    lg = logger
    lg.info(f"{Fore.MAGENTA}{BRIGHT}===== Starting Main Bot Loop (Pyrmethus v{BOT_VERSION}) ====={Style.RESET_ALL}")

    # --- Initialization ---
    trading_pairs_config = config.get('trading_pairs', [])
    if not trading_pairs_config:
        lg.critical(f"{NEON_RED}FATAL: No 'trading_pairs' defined in config.json. Exiting.{RESET}")
        return

    timeframe_str = config['interval']
    ccxt_timeframe = CCXT_INTERVAL_MAP.get(timeframe_str)
    if not ccxt_timeframe:
        lg.critical(f"{NEON_RED}FATAL: Invalid 'interval' ('{timeframe_str}') in config.json. Use one of {VALID_INTERVALS}. Exiting.{RESET}")
        return

    max_positions = config['max_concurrent_positions']
    enable_trading = config['enable_trading']
    loop_delay = config['loop_delay_seconds']
    fetch_limit = config['fetch_limit']
    strategy_params = config['strategy_params']
    protection_params = config['protection']
    risk_per_trade = Decimal(str(config['risk_per_trade']))
    quote_ccy = config['quote_currency']
    confirm_delay = config['position_confirm_delay_seconds']

    # Validate exchange connection and fetch markets
    try:
        exchange.load_markets()
        lg.info(f"Connected to {exchange.name} (Sandbox: {config['use_sandbox']}). Markets loaded.")
    except (ccxt.AuthenticationError, ccxt.ExchangeError, ccxt.NetworkError) as e:
        lg.critical(f"{NEON_RED}FATAL: Failed to connect to exchange or load markets: {e}. Check API keys and connection.{RESET}")
        return
    except Exception as e:
        lg.critical(f"{NEON_RED}FATAL: Unexpected error during exchange initialization: {e}{RESET}", exc_info=True)
        return

    # Pre-process trading pairs and get market info
    active_symbols = []
    market_infos: dict[str, MarketInfo] = {}
    for pair_config in trading_pairs_config:
        try:
            # Expect format like "BTC/USDT:USDT" for perpetuals or "BTC/USDT" for spot
            symbol = pair_config.split(':')[0]
            if symbol not in exchange.markets:
                 lg.error(f"Symbol '{symbol}' (from config: '{pair_config}') not found on {exchange.name}. Skipping.")
                 continue
            market_raw = exchange.market(symbol)
            market_info = enhance_market_info(market_raw)
            market_infos[symbol] = market_info
            active_symbols.append(symbol)
            lg.info(f"Validated symbol: {symbol} ({market_info['contract_type_str']})")
        except (KeyError, IndexError, ValueError) as e:
             lg.error(f"Error processing trading pair '{pair_config}': {e}. Ensure format is correct (e.g., 'BTC/USDT:USDT' or 'ETH/USDT'). Skipping.")
        except Exception as e:
            lg.error(f"Unexpected error processing symbol '{pair_config}': {e}", exc_info=True)

    if not active_symbols:
        lg.critical(f"{NEON_RED}FATAL: No valid trading symbols found after validation. Check config and exchange availability. Exiting.{RESET}")
        return

    if not enable_trading:
        lg.warning(f"{NEON_YELLOW}{BRIGHT}### TRADING DISABLED ### (enable_trading=False in config.json). Running in analysis-only (dry-run) mode.{RESET}")
    else:
         lg.warning(f"{NEON_GREEN}{BRIGHT}### TRADING ENABLED ### Live orders will be placed.{RESET}")

    # --- Main Loop Start ---
    while not _shutdown_requested:
        loop_start_time = time.time()
        try:
            # --- Fetch Balances (once per loop) ---
            balance = Decimal('0')
            try:
                balance_data = exchange.fetch_balance()
                free_balance = balance_data.get(quote_ccy, {}).get('free', 0)
                if free_balance is not None:
                    balance = _safe_market_decimal(free_balance, f"balance.{quote_ccy}.free", default=Decimal('0'))
                    lg.info(f"Current Balance ({quote_ccy}): {balance.normalize()}")
                else:
                    lg.warning(f"Could not find free balance for {quote_ccy}.")
            except (ccxt.ExchangeError, ccxt.NetworkError) as e:
                lg.error(f"Failed to fetch balance: {e}. Using last known or zero.")
                # Consider if bot should pause if balance fetch fails repeatedly

            # --- Fetch All Positions (once per loop if possible) ---
            # Use fetch_positions for perpetuals/futures if supported and needed
            # For Bybit V5, fetch_positions requires symbols
            all_positions_raw = []
            try:
                 if exchange.has.get('fetchPositions'):
                     # Bybit V5 needs symbols specified
                     if 'bybit' in exchange.id.lower():
                         all_positions_raw = exchange.fetch_positions(symbols=active_symbols)
                     else:
                         all_positions_raw = exchange.fetch_positions()  # Fetch all if possible
                 else: lg.debug("Exchange does not support fetch_positions.")
            except (ccxt.ExchangeError, ccxt.NetworkError) as e:
                lg.error(f"Failed to fetch positions: {e}. Position checks might be incomplete.")
            except Exception as e:
                lg.error(f"Unexpected error fetching positions: {e}", exc_info=True)

            # Process raw positions into enhanced format, filtering for relevant symbols and non-zero size
            active_positions_map: dict[str, PositionInfo] = {}
            current_total_positions = 0
            for pos_raw in all_positions_raw:
                symbol = pos_raw.get('symbol')
                if symbol in active_symbols:
                     market_info = market_infos[symbol]
                     pos_info = enhance_position_info(pos_raw, market_info)
                     # Check if position has a non-negligible size
                     if pos_info['size_decimal'] and not pos_info['size_decimal'].is_zero():
                         active_positions_map[symbol] = pos_info
                         current_total_positions += 1
                         lg.info(f"  Active Position: {symbol}, Side: {pos_info.get('side')}, Size: {pos_info['size_decimal'].normalize()}, Entry: {pos_info.get('entryPrice_decimal', 'N/A')}")

            lg.info(f"Total active positions across monitored symbols: {current_total_positions} (Max Allowed: {max_positions})")

            # --- Process Each Symbol ---
            for symbol in active_symbols:
                if _shutdown_requested: break  # Check shutdown flag between symbols
                lg.info(f"{NEON_BLUE}--- Processing Symbol: {symbol} ---{RESET}")
                market_info = market_infos[symbol]
                current_position = active_positions_map.get(symbol)  # Get position info if exists

                # --- Fetch OHLCV Data ---
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, ccxt_timeframe, limit=fetch_limit)
                    if not ohlcv or len(ohlcv) < 50:  # Need enough data for indicators
                         lg.warning(f"Insufficient OHLCV data received for {symbol} ({len(ohlcv) if ohlcv else 0} candles). Skipping analysis.")
                         continue
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    # Basic data validation
                    if df.isnull().values.any():
                         lg.warning(f"NaN values found in OHLCV data for {symbol}. Trying to proceed cautiously.")
                         df.ffill(inplace=True)  # Forward fill NaNs as a basic fix
                except (ccxt.ExchangeError, ccxt.NetworkError) as e:
                    lg.error(f"Failed to fetch OHLCV for {symbol}: {e}. Skipping symbol this cycle.")
                    continue
                except Exception as e:
                     lg.error(f"Unexpected error fetching or processing OHLCV for {symbol}: {e}", exc_info=True)
                     continue

                # --- Analyze Market and Generate Signal ---
                analysis = analyze_market(df, strategy_params, symbol, lg)
                signal_result = generate_signal(analysis, strategy_params, protection_params, symbol, lg)
                signal = signal_result['signal']
                entry_price_signal = analysis['last_close']  # Use last close as potential entry price
                sl_price_signal = signal_result['initial_sl']
                tp_price_signal = signal_result['initial_tp']

                # --- Position Management Logic ---

                # 1. Check if we should exit an existing position (based on SL/TP - handled by exchange, or signal?)
                #    Simple strategy: Rely on exchange SL/TP. Could add signal-based exits here.
                #    Example: if current_position and signal == "EXIT_" + current_position['side'].upper(): ... close position ...

                # 2. Check if we should enter a new position
                if not current_position and signal in ["BUY", "SELL"]:
                    if current_total_positions >= max_positions:
                        lg.info(f"Max positions ({max_positions}) reached. Skipping potential new {signal} trade for {symbol}.")
                        continue

                    if not sl_price_signal:
                        lg.warning(f"Cannot enter {signal} for {symbol}: Stop loss price not generated by strategy.")
                        continue

                    # Calculate position size
                    pos_size = calculate_position_size(
                        exchange, symbol, balance, risk_per_trade,
                        entry_price_signal, sl_price_signal, market_info, lg
                    )

                    if pos_size and pos_size > 0:
                        if not enable_trading:
                            lg.warning(f"[DRY RUN] Would place {signal} order for {symbol}, Size: {pos_size.normalize()}, SL: {sl_price_signal.normalize()}, TP: {tp_price_signal.normalize() if tp_price_signal else 'None'}")
                        else:
                            # --- Place Entry Order ---
                            order_result = place_trade(
                                exchange, symbol, signal, pos_size, market_info, lg, reduce_only=False
                            )

                            if order_result and order_result.get('id'):
                                # --- Set SL/TP after confirming entry ---
                                lg.info(f"Waiting {confirm_delay}s to confirm position and set protection for {symbol}...")
                                time.sleep(confirm_delay)
                                # Re-fetch position to confirm entry before setting SL/TP
                                try:
                                    # Fetch only the specific symbol's position
                                    refetched_positions = exchange.fetch_positions(symbols=[symbol])
                                    new_position_raw = next((p for p in refetched_positions if p.get('symbol') == symbol), None)

                                    if new_position_raw:
                                        new_position = enhance_position_info(new_position_raw, market_info)
                                        if new_position['size_decimal'] and not new_position['size_decimal'].is_zero():
                                            lg.info(f"Position confirmed for {symbol}. Setting initial protection...")
                                            set_protection_success = set_position_protection(
                                                exchange, symbol, market_info, lg,
                                                stop_loss_price=sl_price_signal,
                                                take_profit_price=tp_price_signal
                                                # Add TSL params here if needed for initial setting
                                            )
                                            if not set_protection_success:
                                                 lg.error(f"Failed to set initial protection for {symbol} after entry. Position might be unprotected!")
                                                 # Consider trying to close the position if protection fails critically?
                                        else:
                                             lg.error(f"Order {order_result.get('id')} placed but position size is zero for {symbol}. Check fills or exchange status.")
                                    else:
                                         lg.error(f"Order {order_result.get('id')} placed but could not fetch/confirm position for {symbol} afterwards.")

                                except (ccxt.ExchangeError, ccxt.NetworkError) as e:
                                     lg.error(f"Failed to re-fetch position for {symbol} after order placement: {e}. Cannot set SL/TP.")
                                except Exception as e:
                                    lg.error(f"Unexpected error confirming position or setting protection for {symbol}: {e}", exc_info=True)
                            else:
                                lg.error(f"Failed to place {signal} order for {symbol}. See previous logs.")
                    else:
                         lg.info(f"Signal {signal} for {symbol} generated, but position size calculation failed or resulted in zero size.")

                # 3. Add logic for managing existing positions (e.g., updating TSL, BE) if needed
                elif current_position:
                     lg.debug(f"Holding existing {current_position.get('side')} position for {symbol}.")
                     # Add logic here to check if BE or TSL should be activated/updated based on current price and protection_params
                     # Example: check_and_update_protection(exchange, symbol, market_info, current_position, analysis, protection_params, lg)

                # Brief pause between symbols if many pairs are processed
                if len(active_symbols) > 1:
                    time.sleep(max(0.1, loop_delay / (len(active_symbols) * 2)))  # Small delay

        except KeyboardInterrupt:  # Allow manual interruption via Ctrl+C
             lg.warning("KeyboardInterrupt received during main loop.")
             _shutdown_requested = True  # Trigger graceful shutdown
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            lg.warning(f"Network issue during main loop: {e}. Retrying after delay...")
            # No sleep here, handled by main loop delay calculation
        except ccxt.AuthenticationError as e:
             lg.critical(f"{NEON_RED}Authentication Error in main loop: {e}. Bot cannot continue.{RESET}")
             _shutdown_requested = True  # Stop the bot
        except ccxt.ExchangeError as e:
             lg.error(f"Exchange Error in main loop: {e}. Check exchange status and API limits.")
             # Consider longer sleep or pausing based on error type
        except Exception as e:
            lg.error(f"!!! UNEXPECTED ERROR in main loop: {e}", exc_info=True)
            # Log traceback for debugging unexpected issues
            # Consider a longer pause after unexpected errors

        # --- Loop End ---
        if _shutdown_requested:
            lg.info("Shutdown requested, exiting main loop.")
            break

        # Calculate time elapsed and sleep for the remaining duration of the loop cycle
        loop_end_time = time.time()
        elapsed_time = loop_end_time - loop_start_time
        sleep_time = max(0, loop_delay - elapsed_time)
        lg.debug(f"Loop cycle finished in {elapsed_time:.2f}s. Sleeping for {sleep_time:.2f}s...")
        if sleep_time > 0:
             time.sleep(sleep_time)

    lg.info(f"{Fore.MAGENTA}{BRIGHT}===== Pyrmethus Volumatic Bot Main Loop Finished =====")


# --- Signal Handling for Graceful Shutdown ---
def signal_handler(sig: int, frame: Any) -> None:
    """Handles termination signals (SIGINT, SIGTERM) for graceful shutdown."""
    global _shutdown_requested
    signal_name = signal.Signals(sig).name
    init_logger.warning(f"\n{NEON_YELLOW}{BRIGHT}--- Received Signal: {signal_name} ({sig}) ---{RESET}")
    init_logger.warning("Initiating graceful shutdown. Please wait...")
    _shutdown_requested = True
    # Optionally: Add logic here to cancel open orders before exiting if desired


# --- Main Entry Point ---
if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle `kill` command

    # Load configuration
    config = load_config(CONFIG_FILE)

    # Initialize CCXT Exchange instance
    try:
        exchange = ccxt.bybit({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,  # Let ccxt handle basic rate limiting
            'options': {
                'defaultType': 'swap',  # Assume swap/perpetual by default if not specified in symbol
                'adjustForTimeDifference': True,  # Adjust nonce for clock skew
            },
            # Set sandbox mode based on config
            'urls': {'api': {'public': 'https://api-testnet.bybit.com',
                             'private': 'https://api-testnet.bybit.com'}} if config.get('use_sandbox', True) else {},
        })
        # Explicitly set sandbox mode if supported by the exchange class attribute
        if hasattr(exchange, 'set_sandbox_mode'):
             exchange.set_sandbox_mode(config.get('use_sandbox', True))

        init_logger.info(f"CCXT {exchange.id} instance created. Sandbox mode: {config.get('use_sandbox', True)}")

    except ccxt.AuthenticationError:
        init_logger.critical(f"{NEON_RED}FATAL: CCXT Authentication Failed. Check API Key/Secret.{RESET}")
        sys.exit(1)
    except Exception as e:
        init_logger.critical(f"{NEON_RED}FATAL: Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True)
        sys.exit(1)

    # --- Select Mode: Backtesting or Live/Dry Run ---
    if config.get('backtesting', {}).get('enabled', False):
        init_logger.info(f"{BRIGHT}--- Starting Backtesting Mode ---{RESET}")
        backtest_config = config['backtesting']
        start_date = backtest_config['start_date']
        end_date = backtest_config['end_date']
        symbols_to_backtest = config.get('trading_pairs', [])
        if not symbols_to_backtest:
             init_logger.error("No trading pairs defined for backtesting.")
        else:
            all_results = {}
            for pair_config in symbols_to_backtest:
                 symbol = pair_config.split(':')[0]  # Extract symbol part
                 if symbol not in exchange.markets:
                     init_logger.error(f"Symbol '{symbol}' not available on {exchange.name} for backtesting. Skipping.")
                     continue
                 init_logger.info(f"Running backtest for: {symbol}")
                 result = run_backtest(
                     exchange, symbol, start_date, end_date, config, init_logger  # Use init_logger for backtest output
                 )
                 if result: all_results[symbol] = result
                 if _shutdown_requested:  # Allow interruption during backtests
                      init_logger.warning("Shutdown requested during backtesting.")
                      break
            # Optional: Print consolidated backtest results here
            init_logger.info(f"{BRIGHT}--- Backtesting Mode Finished ---{RESET}")

    else:
        # Start the main trading loop
        main_loop(exchange, config, setup_logger("main"))  # Use a dedicated logger for the main loop

    # --- Shutdown ---
    init_logger.info("Bot shutdown sequence complete.")
    sys.exit(0)
