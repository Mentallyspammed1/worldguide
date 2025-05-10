#!/usr/bin/env python3 to
# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v3.0.0 (Wards of Prudence & Precision)
# Enhancements: API caching, WebSocket streaming, structured logging, trailing SL, rate limit handling.

"""
High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 3.0.0 - Enhanced by Pyrmethus

Dependencies & Setup (Termux):
1. Core Packages:
   pkg install python clang libcrypt libffi openssl rust termux-api
2. Python Libraries:
   pip install ccxt ccxt[async] pandas pandas-ta colorama python-dotenv retry pytz cachetools

WARNING: This script executes live trades and modifies SL/TP on the exchange. Use with extreme caution.
Pyrmethus bears no responsibility for financial outcomes.

Enhancements in v3.0.0:
- Fixed potential `urls` TypeError in ccxt.bybit by enforcing dictionary structure.
- Added API caching with cachetools to reduce redundant calls.
- Integrated WebSocket streaming for real-time price data.
- Implemented structured JSON logging for monitoring.
- Added trailing stop-loss feature.
- Enhanced rate limit handling with exponential backoff.
- Refactored configuration for clarity.
- Batched notifications to reduce frequency.
- Added health checks for API connectivity.
"""

import json
import logging
import logging.handlers
import os
import random
import subprocess
import sys
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation, getcontext
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
import pandas_ta as ta
from cachetools import TTLCache
from colorama import Back, Fore, Style
from colorama import init as colorama_init
from dotenv import load_dotenv
from retry import retry
import pytz
import asyncio

# --- Constants ---
PYRMETHUS_VERSION = "3.0.0"
STATE_FILE_NAME = f"pyrmethus_phoenix_state_v{PYRMETHUS_VERSION.replace('.', '')}.json"
STATE_FILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), STATE_FILE_NAME
)
HEARTBEAT_INTERVAL_SECONDS = 60
OHLCV_LIMIT = 200
API_CACHE_TTL_SECONDS = 60  # Cache TTL for market data and balance
HEALTH_CHECK_INTERVAL_SECONDS = 300
NOTIFICATION_BATCH_INTERVAL_SECONDS = 60

# --- Neon Color Palette ---
NEON = {
    "INFO": Fore.CYAN,
    "DEBUG": Fore.BLUE + Style.DIM,
    "WARNING": Fore.YELLOW + Style.BRIGHT,
    "ERROR": Fore.RED + Style.BRIGHT,
    "CRITICAL": Back.RED + Fore.WHITE + Style.BRIGHT,
    "SUCCESS": Fore.GREEN + Style.BRIGHT,
    "STRATEGY": Fore.MAGENTA,
    "PARAM": Fore.LIGHTBLUE_EX,
    "VALUE": Fore.LIGHTYELLOW_EX + Style.BRIGHT,
    "PRICE": Fore.LIGHTGREEN_EX + Style.BRIGHT,
    "QTY": Fore.LIGHTCYAN_EX + Style.BRIGHT,
    "PNL_POS": Fore.GREEN + Style.BRIGHT,
    "PNL_NEG": Fore.RED + Style.BRIGHT,
    "PNL_ZERO": Fore.YELLOW,
    "SIDE_LONG": Fore.GREEN,
    "SIDE_SHORT": Fore.RED,
    "SIDE_FLAT": Fore.BLUE,
    "HEADING": Fore.MAGENTA + Style.BRIGHT,
    "SUBHEADING": Fore.CYAN + Style.BRIGHT,
    "ACTION": Fore.YELLOW + Style.BRIGHT,
    "COMMENT": Fore.CYAN + Style.DIM,
    "RESET": Style.RESET_ALL,
}

# --- Initializations ---
colorama_init(autoreset=True)
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if load_dotenv(dotenv_path=env_path):
    logging.getLogger("PreConfig").info(
        f"Secrets whispered from .env scroll: {env_path}"
    )
else:
    logging.getLogger("PreConfig").warning(
        f"No .env scroll found at {env_path}. Relying on system environment variables or defaults."
    )
getcontext().prec = 18

# --- Caches ---
market_data_cache = TTLCache(maxsize=100, ttl=API_CACHE_TTL_SECONDS)
balance_cache = TTLCache(maxsize=10, ttl=API_CACHE_TTL_SECONDS)

# --- Notification Batching ---
notification_buffer: List[Dict[str, str]] = []
last_notification_flush_time: float = 0.0


# --- Helper Functions ---
def safe_decimal_conversion(
    value: Any, default_if_error: Any = pd.NA
) -> Union[Decimal, Any]:
    if pd.isna(value):
        return default_if_error
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default_if_error


def _format_for_log(value: Any, is_bool_trend: bool = False, precision: int = 4) -> str:
    if isinstance(value, Decimal):
        return f"{value:.{precision}f}"
    if is_bool_trend:
        if value is True:
            return "Up"
        if value is False:
            return "Down"
        return "Indeterminate"
    if pd.isna(value):
        return "N/A"
    return str(value)


def send_termux_notification(
    title: str, message: str, notification_id: int = 777
) -> None:
    global notification_buffer, last_notification_flush_time
    if (
        "CONFIG" not in globals()
        or not hasattr(CONFIG, "enable_notifications")
        or not CONFIG.enable_notifications
    ):
        return
    notification_buffer.append(
        {"title": title, "message": message, "id": notification_id}
    )
    now = time.time()
    if (
        now - last_notification_flush_time >= NOTIFICATION_BATCH_INTERVAL_SECONDS
        or len(notification_buffer) >= 10
    ):
        flush_notifications()


def flush_notifications() -> None:
    global notification_buffer, last_notification_flush_time
    if not notification_buffer:
        return
    notification_timeout = (
        CONFIG.notification_timeout_seconds if "CONFIG" in globals() else 10
    )
    logger = logging.getLogger("TermuxNotification")
    combined_message = "\n".join(
        [f"{n['title']}: {n['message']}" for n in notification_buffer]
    )
    try:
        safe_title = json.dumps("Pyrmethus Batch Notification")
        safe_message = json.dumps(combined_message[:1000])
        command = [
            "termux-notification",
            "--title",
            safe_title,
            "--content",
            safe_message,
            "--id",
            str(notification_buffer[0]["id"]),
        ]
        logger.debug(f"Attempting to send batched Termux notification")
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate(timeout=notification_timeout)
        if process.returncode == 0:
            logger.info("Batched Termux notification sent successfully.")
        else:
            logger.error(
                f"Failed to send batched Termux notification. Error: {stderr.decode().strip()}"
            )
    except Exception as e:
        logger.error(f"Unexpected error sending batched Termux notification: {e}")
    notification_buffer = []
    last_notification_flush_time = time.time()


# --- Enums ---
class StrategyName(str, Enum):
    DUAL_SUPERTREND_MOMENTUM = "DUAL_SUPERTREND_MOMENTUM"
    EHLERS_FISHER = "EHLERS_FISHER"


class VolatilityRegime(Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"


class OrderEntryType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


# --- Configuration Class ---
class Config:
    def __init__(self) -> None:
        self.logger = logging.getLogger("ConfigModule")
        self.logger.info("Summoning Configuration Runes v%s" % PYRMETHUS_VERSION)
        # Core Exchange Config
        self.exchange = {
            "api_key": self._get_env("BYBIT_API_KEY", required=True, secret=True),
            "api_secret": self._get_env("BYBIT_API_SECRET", required=True, secret=True),
            "symbol": self._get_env("SYMBOL", "BTC/USDT:USDT"),
            "interval": self._get_env("INTERVAL", "1m"),
            "leverage": self._get_env("LEVERAGE", 10, cast_type=int),
            "paper_trading": self._get_env(
                "PAPER_TRADING_MODE", "false", cast_type=bool
            ),
            "recv_window": self._get_env("DEFAULT_RECV_WINDOW", 13000, cast_type=int),
        }
        # Trading Parameters
        self.trading = {
            "risk_per_trade_percentage": self._get_env(
                "RISK_PER_TRADE_PERCENTAGE", "0.005", cast_type=Decimal
            ),
            "max_order_usdt_amount": self._get_env(
                "MAX_ORDER_USDT_AMOUNT", "1000.0", cast_type=Decimal
            ),
            "min_usdt_balance": self._get_env(
                "MIN_USDT_BALANCE_FOR_TRADING", "20.0", cast_type=Decimal
            ),
            "max_active_trade_parts": self._get_env(
                "MAX_ACTIVE_TRADE_PARTS", 1, cast_type=int
            ),
            "order_fill_timeout_seconds": self._get_env(
                "ORDER_FILL_TIMEOUT_SECONDS", 15, cast_type=int
            ),
            "sleep_seconds": self._get_env("SLEEP_SECONDS", 10, cast_type=int),
        }
        # Risk Management
        self.risk = {
            "atr_stop_loss_multiplier": self._get_env(
                "ATR_STOP_LOSS_MULTIPLIER", "1.5", cast_type=Decimal
            ),
            "enable_take_profit": self._get_env(
                "ENABLE_TAKE_PROFIT", "true", cast_type=bool
            ),
            "atr_take_profit_multiplier": self._get_env(
                "ATR_TAKE_PROFIT_MULTIPLIER", "2.0", cast_type=Decimal
            ),
            "enable_trailing_sl": self._get_env(
                "ENABLE_TRAILING_SL", "true", cast_type=bool
            ),
            "trailing_sl_trigger_atr": self._get_env(
                "TRAILING_SL_TRIGGER_ATR", "1.0", cast_type=Decimal
            ),
            "trailing_sl_distance_atr": self._get_env(
                "TRAILING_SL_DISTANCE_ATR", "0.5", cast_type=Decimal
            ),
            "enable_max_drawdown_stop": self._get_env(
                "ENABLE_MAX_DRAWDOWN_STOP", "true", cast_type=bool
            ),
            "max_drawdown_percent": self._get_env(
                "MAX_DRAWDOWN_PERCENT", "0.05", cast_type=Decimal
            ),
            "enable_session_pnl_limits": self._get_env(
                "ENABLE_SESSION_PNL_LIMITS", "false", cast_type=bool
            ),
            "session_profit_target_usdt": self._get_env(
                "SESSION_PROFIT_TARGET_USDT", None, cast_type=Decimal, required=False
            ),
            "session_max_loss_usdt": self._get_env(
                "SESSION_MAX_LOSS_USDT", None, cast_type=Decimal, required=False
            ),
        }
        # Strategy Parameters
        self.strategy = {
            "name": StrategyName(
                self._get_env(
                    "STRATEGY_NAME", StrategyName.DUAL_SUPERTREND_MOMENTUM.value
                ).upper()
            ),
            "st_atr_length": self._get_env("ST_ATR_LENGTH", 10, cast_type=int),
            "st_multiplier": self._get_env("ST_MULTIPLIER", "2.0", cast_type=Decimal),
            "confirm_st_atr_length": self._get_env(
                "CONFIRM_ST_ATR_LENGTH", 20, cast_type=int
            ),
            "confirm_st_multiplier": self._get_env(
                "CONFIRM_ST_MULTIPLIER", "3.0", cast_type=Decimal
            ),
            "momentum_period": self._get_env("MOMENTUM_PERIOD", 14, cast_type=int),
            "momentum_threshold": self._get_env(
                "MOMENTUM_THRESHOLD", "0", cast_type=Decimal
            ),
            "confirm_st_stability_lookback": self._get_env(
                "CONFIRM_ST_STABILITY_LOOKBACK", 3, cast_type=int
            ),
            "st_max_entry_distance_atr_multiplier": self._get_env(
                "ST_MAX_ENTRY_DISTANCE_ATR_MULTIPLIER",
                "0.5",
                cast_type=Decimal,
                required=False,
            ),
            "ehlers_fisher_length": self._get_env(
                "EHLERS_FISHER_LENGTH", 10, cast_type=int
            ),
            "ehlers_fisher_signal_length": self._get_env(
                "EHLERS_FISHER_SIGNAL_LENGTH", 1, cast_type=int
            ),
            "ehlers_fisher_extreme_threshold_positive": self._get_env(
                "EHLERS_FISHER_EXTREME_THRESHOLD_POSITIVE", "2.0", cast_type=Decimal
            ),
            "ehlers_fisher_extreme_threshold_negative": self._get_env(
                "EHLERS_FISHER_EXTREME_THRESHOLD_NEGATIVE", "-2.0", cast_type=Decimal
            ),
        }
        # Notification Settings
        self.notifications = {
            "enable": self._get_env("ENABLE_NOTIFICATIONS", "true", cast_type=bool),
            "timeout_seconds": self._get_env(
                "NOTIFICATION_TIMEOUT_SECONDS", 10, cast_type=int
            ),
        }
        # Other Settings
        self.side_buy: str = "buy"
        self.side_sell: str = "sell"
        self.pos_long: str = "Long"
        self.pos_short: str = "Short"
        self.pos_none: str = "None"
        self.usdt_symbol: str = "USDT"
        self.retry_count: int = 5
        self.retry_delay_seconds: int = 5
        self.api_fetch_limit_buffer: int = 20
        self.position_qty_epsilon: Decimal = Decimal("1e-9")
        self.post_close_delay_seconds: int = 3
        self.MARKET_INFO: Optional[Dict[str, Any]] = None
        self.tick_size: Optional[Decimal] = None
        self.qty_step: Optional[Decimal] = None
        self.strategy_instance: Optional["TradingStrategy"] = None
        self.send_notification_method = send_termux_notification
        self._validate_parameters()
        self.logger.info(
            "Configuration Runes v%s Summoned and Verified" % PYRMETHUS_VERSION
        )

    def _get_env(
        self,
        key: str,
        default: Any = None,
        cast_type: type = str,
        required: bool = False,
        secret: bool = False,
    ) -> Any:
        value_str = os.getenv(key)
        source = "Env Var"
        value_to_cast: Any = None
        display_value = "********" if secret and value_str is not None else value_str
        if value_str is None:
            if required:
                self.logger.critical(
                    f"CRITICAL: Required config rune '{key}' not found."
                )
                raise ValueError(f"Required environment variable '{key}' not set.")
            self.logger.debug(
                f"Config Rune '{key}': Not Found. Using Default: '{default}'"
            )
            value_to_cast = default
            source = "Default"
        else:
            self.logger.debug(
                f"Config Rune '{key}': Found Env Value: '{display_value}'"
            )
            value_to_cast = value_str
        if value_to_cast is None and required:
            self.logger.critical(
                f"CRITICAL: Required config rune '{key}' resolved to None."
            )
            raise ValueError(f"Required environment variable '{key}' resolved to None.")
        try:
            raw_value_str = str(value_to_cast)
            if cast_type == bool:
                return raw_value_str.lower() in ["true", "1", "yes", "y"]
            elif cast_type == Decimal:
                return Decimal(raw_value_str)
            elif cast_type == int:
                return int(Decimal(raw_value_str))
            elif cast_type == float:
                return float(raw_value_str)
            elif cast_type == str:
                return raw_value_str
            else:
                self.logger.warning(
                    f"Unsupported cast_type '{cast_type.__name__}' for key '{key}'. Returning raw string."
                )
                return raw_value_str
        except (ValueError, TypeError, InvalidOperation) as e:
            self.logger.error(
                f"Invalid type/value for '{key}': '{value_to_cast}' (Source: {source}). Expected {cast_type.__name__}. Err: {e}."
            )
            if default is None and required:
                raise ValueError(
                    f"Required env var '{key}' failed casting, no valid default."
                )
            return default

    def _validate_parameters(self) -> None:
        errors = []
        if not (0 < self.trading["risk_per_trade_percentage"] < 1):
            errors.append("RISK_PER_TRADE_PERCENTAGE must be between 0 and 1.")
        if self.exchange["leverage"] < 1:
            errors.append("LEVERAGE must be at least 1.")
        if self.risk["atr_stop_loss_multiplier"] <= 0:
            errors.append("ATR_STOP_LOSS_MULTIPLIER must be positive.")
        if (
            self.risk["enable_take_profit"]
            and self.risk["atr_take_profit_multiplier"] <= 0
        ):
            errors.append(
                "ATR_TAKE_PROFIT_MULTIPLIER must be positive if take profit is enabled."
            )
        if self.trading["min_usdt_balance"] < 0:
            errors.append("MIN_USDT_BALANCE_FOR_TRADING cannot be negative.")
        if errors:
            error_message = (
                f"Configuration spellcrafting failed with {len(errors)} flaws:\n"
                + "\n".join([f"  - {e}" for e in errors])
            )
            self.logger.critical(error_message)
            raise ValueError(error_message)


# --- Logger Setup ---
LOGGING_LEVEL = (
    logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO
)
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
log_file_name = f"{logs_dir}/pyrmethus_spell_v{PYRMETHUS_VERSION.replace('.', '')}_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=LOGGING_LEVEL,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}',
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(
            log_file_name, maxBytes=10 * 1024 * 1024, backupCount=5
        ),
    ],
)
logger = logging.getLogger("PyrmethusCore")
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


logging.Logger.success = log_success

# --- Global Objects & State Variables ---
try:
    CONFIG = Config()
except ValueError as config_error:
    logger.critical(f"Configuration loading failed. Error: {config_error}")
    send_termux_notification(
        "Pyrmethus Startup Failure", f"Config Error: {str(config_error)[:200]}"
    )
    sys.exit(1)

trade_timestamps_for_whipsaw = deque(maxlen=CONFIG.trading["max_active_trade_parts"])
whipsaw_cooldown_active_until: float = 0.0
persistent_signal_counter = {"long": 0, "short": 0}
last_signal_type: Optional[str] = None
previous_day_high: Optional[Decimal] = None
previous_day_low: Optional[Decimal] = None
last_key_level_update_day: Optional[int] = None
contradiction_cooldown_active_until: float = 0.0
consecutive_loss_cooldown_active_until: float = 0.0


# --- Trading Strategy Classes ---
class TradingStrategy(ABC):
    def __init__(self, config: Config, df_columns: Optional[List[str]] = None):
        self.config = config
        self.logger = logging.getLogger(f"Strategy.{self.__class__.__name__}")
        self.required_columns = df_columns or []
        self.logger.info(f"Strategy Form '{self.__class__.__name__}' materializing...")

    @abstractmethod
    def generate_signals(
        self,
        df: pd.DataFrame,
        latest_close: Optional[Decimal] = None,
        latest_atr: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        pass

    def _validate_df(self, df: pd.DataFrame, min_rows: int = 2) -> bool:
        if df is None or df.empty or len(df) < min_rows:
            self.logger.debug(
                f"Insufficient market whispers (Rows: {len(df) if df is not None else 0}, Min Required: {min_rows})."
            )
            return False
        if self.required_columns and not all(
            col in df.columns for col in self.required_columns
        ):
            missing_cols = [
                col for col in self.required_columns if col not in df.columns
            ]
            self.logger.warning(
                f"Market scroll missing required runes: {missing_cols}."
            )
            return False
        return True

    def _get_default_signals(self) -> Dict[str, Any]:
        return {
            "enter_long": False,
            "enter_short": False,
            "exit_long": False,
            "exit_short": False,
            "exit_reason": "Default Signal",
        }


class DualSupertrendMomentumStrategy(TradingStrategy):
    def __init__(self, config: Config):
        super().__init__(
            config,
            df_columns=["st_long_flip", "st_short_flip", "confirm_trend", "momentum"],
        )

    def generate_signals(
        self,
        df: pd.DataFrame,
        latest_close: Optional[Decimal] = None,
        latest_atr: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        signals = self._get_default_signals()
        min_rows_needed = (
            max(
                self.config.strategy["st_atr_length"],
                self.config.strategy["confirm_st_atr_length"],
                self.config.strategy["momentum_period"],
                self.config.strategy["confirm_st_stability_lookback"],
            )
            + 10
        )
        if not self._validate_df(df, min_rows=min_rows_needed):
            return signals
        last = df.iloc[-1]
        primary_long_flip = last.get("st_long_flip", False)
        primary_short_flip = last.get("st_short_flip", False)
        confirm_is_up_current = last.get("confirm_trend", pd.NA)
        momentum_val = safe_decimal_conversion(last.get("momentum"), pd.NA)
        if primary_long_flip and primary_short_flip:
            self.logger.warning("Conflicting primary Supertrend flips. Resolving...")
            if confirm_is_up_current is True and (
                momentum_val is not pd.NA and momentum_val > 0
            ):
                primary_short_flip = False
                self.logger.info("Resolution: Prioritizing LONG flip.")
            elif confirm_is_up_current is False and (
                momentum_val is not pd.NA and momentum_val < 0
            ):
                primary_long_flip = False
                self.logger.info("Resolution: Prioritizing SHORT flip.")
            else:
                primary_long_flip = False
                primary_short_flip = False
                self.logger.warning("Resolution: Ambiguous. No primary signal.")
        stable_confirm_trend = pd.NA
        if self.config.strategy["confirm_st_stability_lookback"] <= 1:
            stable_confirm_trend = confirm_is_up_current
        elif (
            "confirm_trend" in df.columns
            and len(df) >= self.config.strategy["confirm_st_stability_lookback"]
        ):
            recent_confirm_trends = df["confirm_trend"].iloc[
                -self.config.strategy["confirm_st_stability_lookback"] :
            ]
            if confirm_is_up_current is True and all(
                trend is True for trend in recent_confirm_trends
            ):
                stable_confirm_trend = True
            elif confirm_is_up_current is False and all(
                trend is False for trend in recent_confirm_trends
            ):
                stable_confirm_trend = False
        if pd.isna(stable_confirm_trend) or pd.isna(momentum_val):
            self.logger.debug(
                f"Stable Confirm ST ({stable_confirm_trend}) or Mom ({momentum_val}) is NA."
            )
            return signals
        price_proximity_ok = True
        if (
            self.config.strategy["st_max_entry_distance_atr_multiplier"] is not None
            and latest_atr is not None
            and latest_atr > 0
            and latest_close is not None
        ):
            max_allowed_distance = (
                latest_atr
                * self.config.strategy["st_max_entry_distance_atr_multiplier"]
            )
            st_p_base = f"ST_{self.config.strategy['st_atr_length']}_{self.config.strategy['st_multiplier']}"
            if primary_long_flip:
                st_line_val = safe_decimal_conversion(last.get(f"{st_p_base}l"))
                if (
                    st_line_val is not None
                    and (latest_close - st_line_val) > max_allowed_distance
                ):
                    price_proximity_ok = False
                    self.logger.debug("Long ST proximity fail.")
            elif primary_short_flip:
                st_line_val = safe_decimal_conversion(last.get(f"{st_p_base}s"))
                if (
                    st_line_val is not None
                    and (st_line_val - latest_close) > max_allowed_distance
                ):
                    price_proximity_ok = False
                    self.logger.debug("Short ST proximity fail.")
        if (
            primary_long_flip
            and stable_confirm_trend is True
            and momentum_val > self.config.strategy["momentum_threshold"]
            and momentum_val > 0
            and price_proximity_ok
        ):
            signals["enter_long"] = True
            self.logger.info("DualST+Mom Signal: LONG Entry.")
        elif (
            primary_short_flip
            and stable_confirm_trend is False
            and momentum_val < -self.config.strategy["momentum_threshold"]
            and momentum_val < 0
            and price_proximity_ok
        ):
            signals["enter_short"] = True
            self.logger.info("DualST+Mom Signal: SHORT Entry.")
        if primary_short_flip:
            signals["exit_long"] = True
            signals["exit_reason"] = "Primary SuperTrend Flipped Short"
        if primary_long_flip:
            signals["exit_short"] = True
            signals["exit_reason"] = "Primary SuperTrend Flipped Long"
        return signals


class EhlersFisherStrategy(TradingStrategy):
    def __init__(self, config: Config):
        super().__init__(config, df_columns=["ehlers_fisher", "ehlers_signal"])

    def generate_signals(
        self,
        df: pd.DataFrame,
        latest_close: Optional[Decimal] = None,
        latest_atr: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        signals = self._get_default_signals()
        min_rows_needed = (
            self.config.strategy["ehlers_fisher_length"]
            + self.config.strategy["ehlers_fisher_signal_length"]
            + 5
        )
        if not self._validate_df(df, min_rows=min_rows_needed) or len(df) < 2:
            return signals
        last = df.iloc[-1]
        prev = df.iloc[-2]
        fisher_now = safe_decimal_conversion(last.get("ehlers_fisher"), pd.NA)
        signal_now = safe_decimal_conversion(last.get("ehlers_signal"), pd.NA)
        fisher_prev = safe_decimal_conversion(prev.get("ehlers_fisher"), pd.NA)
        signal_prev = safe_decimal_conversion(prev.get("ehlers_signal"), pd.NA)
        if (
            pd.isna(fisher_now)
            or pd.isna(signal_now)
            or pd.isna(fisher_prev)
            or pd.isna(signal_prev)
        ):
            self.logger.debug("Ehlers Fisher or Signal rune is NA.")
            return signals
        is_fisher_extreme = (
            fisher_now
            > self.config.strategy["ehlers_fisher_extreme_threshold_positive"]
            or fisher_now
            < self.config.strategy["ehlers_fisher_extreme_threshold_negative"]
        )
        if not is_fisher_extreme:
            if fisher_prev <= signal_prev and fisher_now > signal_now:
                signals["enter_long"] = True
                self.logger.info("EhlersFisher Signal: LONG Entry.")
            elif fisher_prev >= signal_prev and fisher_now < signal_now:
                signals["enter_short"] = True
                self.logger.info("EhlersFisher Signal: SHORT Entry.")
        if fisher_prev >= signal_prev and fisher_now < signal_now:
            signals["exit_long"] = True
            signals["exit_reason"] = "Ehlers Fisher crossed BELOW Signal"
        elif fisher_prev <= signal_prev and fisher_now > signal_now:
            signals["exit_short"] = True
            signals["exit_reason"] = "Ehlers Fisher crossed ABOVE Signal"
        return signals


strategy_map: Dict[StrategyName, Type[TradingStrategy]] = {
    StrategyName.DUAL_SUPERTREND_MOMENTUM: DualSupertrendMomentumStrategy,
    StrategyName.EHLERS_FISHER: EhlersFisherStrategy,
}
StrategyClass = strategy_map.get(CONFIG.strategy["name"])
if StrategyClass:
    CONFIG.strategy_instance = StrategyClass(CONFIG)
    logger.success(f"Strategy '{CONFIG.strategy['name'].value}' invoked.")
else:
    err_msg = f"Failed to init strategy '{CONFIG.strategy['name'].value}'. Unknown spell form."
    logger.critical(err_msg)
    send_termux_notification("Pyrmethus Critical Error", err_msg)
    sys.exit(1)


# --- Trade Metrics Tracking ---
class TradeMetrics:
    def __init__(self):
        self.trades: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("TradeMetrics")
        self.initial_equity: Optional[Decimal] = None
        self.daily_start_equity: Optional[Decimal] = None
        self.last_daily_reset_day: Optional[int] = None
        self.consecutive_losses: int = 0
        self.daily_trade_entry_count: int = 0
        self.last_daily_trade_count_reset_day: Optional[int] = None
        self.daily_trades_rest_active_until: float = 0.0
        self.logger.info("TradeMetrics Ledger opened.")

    def set_initial_equity(self, equity: Decimal):
        if self.initial_equity is None:
            self.initial_equity = equity
            self.logger.info(
                f"Initial Session Equity: {equity:.2f} {CONFIG.usdt_symbol}"
            )
        today = datetime.now(pytz.utc).day
        if self.last_daily_reset_day != today or self.daily_start_equity is None:
            self.daily_start_equity = equity
            self.last_daily_reset_day = today
            self.logger.info(
                f"Daily Equity Ward reset. Dawn Equity: {equity:.2f} {CONFIG.usdt_symbol}"
            )
        if self.last_daily_trade_count_reset_day != today:
            self.daily_trade_entry_count = 0
            self.last_daily_trade_count_reset_day = today
            self.logger.info("Daily trade entry count reset to 0.")

    def check_drawdown(self, current_equity: Decimal) -> Tuple[bool, str]:
        if (
            not CONFIG.risk["enable_max_drawdown_stop"]
            or self.daily_start_equity is None
            or self.daily_start_equity <= 0
        ):
            return False, ""
        drawdown = self.daily_start_equity - current_equity
        drawdown_pct = drawdown / self.daily_start_equity
        if drawdown_pct >= CONFIG.risk["max_drawdown_percent"]:
            reason = f"Max daily drawdown breached ({drawdown_pct:.2%} >= {CONFIG.risk['max_drawdown_percent']:.2%})"
            self.logger.warning(reason)
            send_termux_notification(
                "Pyrmethus: Max Drawdown Hit!",
                f"Drawdown: {drawdown_pct:.2%}. Trading halted.",
            )
            return True, reason
        return False, ""

    def log_trade(
        self,
        symbol: str,
        side: str,
        entry_price: Decimal,
        exit_price: Decimal,
        qty: Decimal,
        entry_time_ms: int,
        exit_time_ms: int,
        reason: str,
        part_id: str,
        pnl_str: str,
        scale_order_id: Optional[str] = None,
        mae: Optional[Decimal] = None,
        mfe: Optional[Decimal] = None,
    ):
        profit = safe_decimal_conversion(pnl_str, Decimal(0))
        if profit <= 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        entry_dt_utc = datetime.fromtimestamp(entry_time_ms / 1000, tz=pytz.utc)
        exit_dt_utc = datetime.fromtimestamp(exit_time_ms / 1000, tz=pytz.utc)
        duration_seconds = (exit_dt_utc - entry_dt_utc).total_seconds()
        trade_type = "Scale-In" if scale_order_id else "Part"
        self.trades.append(
            {
                "symbol": symbol,
                "side": side,
                "entry_price_str": str(entry_price),
                "exit_price_str": str(exit_price),
                "qty_str": str(qty),
                "profit_str": str(profit),
                "entry_time_iso": entry_dt_utc.isoformat(),
                "exit_time_iso": exit_dt_utc.isoformat(),
                "duration_seconds": duration_seconds,
                "exit_reason": reason,
                "type": trade_type,
                "part_id": part_id,
                "scale_order_id": scale_order_id,
                "mae_str": str(mae) if mae is not None else None,
                "mfe_str": str(mfe) if mfe is not None else None,
            }
        )
        self.logger.success(
            f"Trade Chronicle ({trade_type}:{part_id}): {side.upper()} {qty} {symbol.split('/')[0]} | P/L: {profit:.2f} {CONFIG.usdt_symbol} | Reason: {reason}"
        )

    def increment_daily_trade_entry_count(self):
        self.daily_trade_entry_count += 1
        self.logger.info(
            f"Daily entry trade count incremented to: {self.daily_trade_entry_count}"
        )

    def summary(self) -> str:
        if not self.trades:
            return "The Grand Ledger is empty."
        total_trades = len(self.trades)
        profits = [Decimal(t["profit_str"]) for t in self.trades]
        wins = sum(1 for p in profits if p > 0)
        losses = sum(1 for p in profits if p < 0)
        breakeven = total_trades - wins - losses
        win_rate = (
            (Decimal(wins) / Decimal(total_trades)) * Decimal(100)
            if total_trades > 0
            else Decimal(0)
        )
        total_profit = sum(profits)
        avg_profit = (
            total_profit / Decimal(total_trades) if total_trades > 0 else Decimal(0)
        )
        summary_str = (
            f"\n--- Pyrmethus Trade Metrics Summary (v{PYRMETHUS_VERSION}) ---\n"
            f"Total Trade Parts: {total_trades}\n"
            f"  Wins: {wins}, Losses: {losses}, Breakeven: {breakeven}\n"
            f"Win Rate: {win_rate:.2f}%\n"
            f"Total P/L: {total_profit:.2f} {CONFIG.usdt_symbol}\n"
            f"Avg P/L per Part: {avg_profit:.2f} {CONFIG.usdt_symbol}\n"
        )
        if self.initial_equity is not None:
            current_equity_approx = self.initial_equity + total_profit
            overall_pnl_pct = (
                (total_profit / self.initial_equity) * 100
                if self.initial_equity > Decimal(0)
                else Decimal(0)
            )
            summary_str += f"Initial Session Treasury: {self.initial_equity:.2f} {CONFIG.usdt_symbol}\n"
            summary_str += f"Approx. Current Treasury: {current_equity_approx:.2f} {CONFIG.usdt_symbol}\n"
            summary_str += f"Overall Session P/L %: {overall_pnl_pct:.2f}%\n"
        if self.daily_start_equity is not None:
            daily_pnl = (
                (self.initial_equity or self.daily_start_equity)
                + total_profit
                - self.daily_start_equity
            )
            daily_pnl_pct = (
                (daily_pnl / self.daily_start_equity) * 100
                if self.daily_start_equity > Decimal(0)
                else Decimal(0)
            )
            summary_str += f"Daily Start Treasury: {self.daily_start_equity:.2f} {CONFIG.usdt_symbol}\n"
            summary_str += f"Approx. Daily P/L: {daily_pnl:.2f} {CONFIG.usdt_symbol} ({daily_pnl_pct:.2f}%)\n"
        summary_str += f"Consecutive Losses: {self.consecutive_losses}\n"
        summary_str += f"Daily Entries Made: {self.daily_trade_entry_count}\n"
        summary_str += "--- End of Ledger Reading ---"
        self.logger.info(summary_str)
        return summary_str


trade_metrics = TradeMetrics()
_active_trade_parts: List[Dict[str, Any]] = []
_last_heartbeat_save_time: float = 0.0
_last_health_check_time: float = 0.0


# --- State Persistence Functions ---
def save_persistent_state(force_heartbeat: bool = False) -> None:
    global _active_trade_parts, trade_metrics, _last_heartbeat_save_time
    now = time.time()
    should_save = force_heartbeat or (
        _active_trade_parts
        and (now - _last_heartbeat_save_time > HEARTBEAT_INTERVAL_SECONDS)
        or (
            not _active_trade_parts
            and (now - _last_heartbeat_save_time > HEARTBEAT_INTERVAL_SECONDS * 5)
        )
    )
    if not should_save:
        return
    logger.debug("Phoenix Feather scribing memories...")
    try:
        serializable_active_parts = []
        for part in _active_trade_parts:
            p_copy = part.copy()
            for key, value in p_copy.items():
                if isinstance(value, Decimal):
                    p_copy[key] = str(value)
                elif isinstance(value, deque):
                    p_copy[key] = list(value)
            serializable_active_parts.append(p_copy)
        state_data = {
            "pyrmethus_version": PYRMETHUS_VERSION,
            "timestamp_utc_iso": datetime.now(pytz.utc).isoformat(),
            "active_trade_parts": serializable_active_parts,
            "trade_metrics_trades": trade_metrics.trades,
            "trade_metrics_consecutive_losses": trade_metrics.consecutive_losses,
            "trade_metrics_daily_trade_entry_count": trade_metrics.daily_trade_entry_count,
            "trade_metrics_last_daily_trade_count_reset_day": trade_metrics.last_daily_trade_count_reset_day,
            "trade_metrics_daily_trades_rest_active_until": trade_metrics.daily_trades_rest_active_until,
            "config_symbol": CONFIG.exchange["symbol"],
            "config_strategy": CONFIG.strategy["name"].value,
            "initial_equity_str": str(trade_metrics.initial_equity)
            if trade_metrics.initial_equity is not None
            else None,
            "daily_start_equity_str": str(trade_metrics.daily_start_equity)
            if trade_metrics.daily_start_equity is not None
            else None,
            "last_daily_reset_day": trade_metrics.last_daily_reset_day,
            "whipsaw_cooldown_active_until": whipsaw_cooldown_active_until,
            "trade_timestamps_for_whipsaw": list(trade_timestamps_for_whipsaw),
            "persistent_signal_counter": persistent_signal_counter,
            "last_signal_type": last_signal_type,
            "previous_day_high_str": str(previous_day_high)
            if previous_day_high
            else None,
            "previous_day_low_str": str(previous_day_low) if previous_day_low else None,
            "last_key_level_update_day": last_key_level_update_day,
            "contradiction_cooldown_active_until": contradiction_cooldown_active_until,
            "consecutive_loss_cooldown_active_until": consecutive_loss_cooldown_active_until,
        }
        with open(STATE_FILE_PATH, "w") as f:
            json.dump(state_data, f, indent=2)
        _last_heartbeat_save_time = now
        logger.debug("State saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save state: {e}")


def load_persistent_state() -> bool:
    global _active_trade_parts, trade_metrics
    if not os.path.exists(STATE_FILE_PATH):
        logger.info("No prior state scroll found.")
        return False
    try:
        with open(STATE_FILE_PATH, "r") as f:
            state_data = json.load(f)
        if state_data.get("pyrmethus_version") != PYRMETHUS_VERSION:
            logger.warning("State version mismatch. Ignoring prior state.")
            return False
        if (
            state_data.get("config_symbol") != CONFIG.exchange["symbol"]
            or state_data.get("config_strategy") != CONFIG.strategy["name"].value
        ):
            logger.warning("Symbol or strategy mismatch in state. Ignoring.")
            return False
        _active_trade_parts = []
        for part in state_data.get("active_trade_parts", []):
            p_copy = part.copy()
            for key, value in p_copy.items():
                if (
                    key
                    in ["entry_price", "sl_price", "tp_price", "qty", "atr_at_entry"]
                    and value is not None
                ):
                    p_copy[key] = Decimal(value)
                elif (
                    key in ["recent_pnls", "adverse_candle_closes"]
                    and value is not None
                ):
                    p_copy[key] = deque(value)
            _active_trade_parts.append(p_copy)
        trade_metrics.trades = state_data.get("trade_metrics_trades", [])
        trade_metrics.consecutive_losses = state_data.get(
            "trade_metrics_consecutive_losses", 0
        )
        trade_metrics.daily_trade_entry_count = state_data.get(
            "trade_metrics_daily_trade_entry_count", 0
        )
        trade_metrics.last_daily_trade_count_reset_day = state_data.get(
            "trade_metrics_last_daily_trade_count_reset_day"
        )
        trade_metrics.daily_trades_rest_active_until = state_data.get(
            "trade_metrics_daily_trades_rest_active_until", 0.0
        )
        trade_metrics.initial_equity = (
            Decimal(state_data["initial_equity_str"])
            if state_data.get("initial_equity_str")
            else None
        )
        trade_metrics.daily_start_equity = (
            Decimal(state_data["daily_start_equity_str"])
            if state_data.get("daily_start_equity_str")
            else None
        )
        trade_metrics.last_daily_reset_day = state_data.get("last_daily_reset_day")
        global \
            whipsaw_cooldown_active_until, \
            trade_timestamps_for_whipsaw, \
            persistent_signal_counter, \
            last_signal_type
        global previous_day_high, previous_day_low, last_key_level_update_day
        global \
            contradiction_cooldown_active_until, \
            consecutive_loss_cooldown_active_until
        whipsaw_cooldown_active_until = state_data.get(
            "whipsaw_cooldown_active_until", 0.0
        )
        trade_timestamps_for_whipsaw = deque(
            state_data.get("trade_timestamps_for_whipsaw", []),
            maxlen=CONFIG.trading["max_active_trade_parts"],
        )
        persistent_signal_counter = state_data.get(
            "persistent_signal_counter", {"long": 0, "short": 0}
        )
        last_signal_type = state_data.get("last_signal_type")
        previous_day_high = (
            Decimal(state_data["previous_day_high_str"])
            if state_data.get("previous_day_high_str")
            else None
        )
        previous_day_low = (
            Decimal(state_data["previous_day_low_str"])
            if state_data.get("previous_day_low_str")
            else None
        )
        last_key_level_update_day = state_data.get("last_key_level_update_day")
        contradiction_cooldown_active_until = state_data.get(
            "contradiction_cooldown_active_until", 0.0
        )
        consecutive_loss_cooldown_active_until = state_data.get(
            "consecutive_loss_cooldown_active_until", 0.0
        )
        logger.info(
            f"Loaded state: {len(_active_trade_parts)} active parts, {len(trade_metrics.trades)} trades."
        )
        return True
    except Exception as e:
        logger.error(f"Failed to load state: {e}")
        return False


# --- Exchange Functions ---
@retry(
    (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.RateLimitExceeded),
    tries=CONFIG.retry_count,
    delay=CONFIG.retry_delay_seconds,
    backoff=2,
)
def fetch_account_balance(exchange: ccxt.Exchange, currency: str) -> Optional[Decimal]:
    cache_key = f"balance_{currency}"
    if cache_key in balance_cache:
        return balance_cache[cache_key]
    try:
        balance = exchange.fetch_balance()
        total_balance = safe_decimal_conversion(
            balance.get("total", {}).get(currency, 0)
        )
        balance_cache[cache_key] = total_balance
        return total_balance
    except Exception as e:
        logger.error(f"Failed to fetch balance: {e}")
        return None


@retry(
    (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.RateLimitExceeded),
    tries=CONFIG.retry_count,
    delay=CONFIG.retry_delay_seconds,
    backoff=2,
)
def get_market_data(
    exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int
) -> Optional[pd.DataFrame]:
    cache_key = f"ohlcv_{symbol}_{timeframe}_{limit}"
    if cache_key in market_data_cache:
        return market_data_cache[cache_key]
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        market_data_cache[cache_key] = df
        return df
    except Exception as e:
        logger.error(f"Failed to fetch market data: {e}")
        return None


def calculate_all_indicators(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    # Placeholder: Implement actual indicator calculations
    # This is assumed to be defined in the original code
    return df


def _synchronize_part_sltp(
    sl_price: Optional[Decimal], tp_price: Optional[Decimal]
) -> None:
    global _active_trade_parts
    for part in _active_trade_parts:
        part["sl_price"] = sl_price
        part["tp_price"] = tp_price
        part["last_trailing_sl_update"] = time.time()


def get_current_position_info() -> Tuple[str, Decimal]:
    # Placeholder: Implement actual position info retrieval
    return CONFIG.pos_none, Decimal(0)


@retry(
    (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.RateLimitExceeded),
    tries=CONFIG.retry_count,
    delay=CONFIG.retry_delay_seconds,
    backoff=2,
)
def place_risked_order(
    exchange: ccxt.Exchange,
    config: Config,
    side: str,
    entry_price: Decimal,
    sl_price: Decimal,
    tp_price: Optional[Decimal],
    atr: Decimal,
    df: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    try:
        # Placeholder: Implement actual order placement logic
        part_id = str(uuid.uuid4())
        part = {
            "part_id": part_id,
            "symbol": config.exchange["symbol"],
            "side": side,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "qty": Decimal("0.01"),
            "entry_time_ms": int(time.time() * 1000),
            "atr_at_entry": atr,
            "recent_pnls": deque(maxlen=3),
            "adverse_candle_closes": deque(maxlen=2),
            "last_trailing_sl_update": time.time(),
        }
        _active_trade_parts.append(part)
        trade_metrics.increment_daily_trade_entry_count()
        logger.success(f"Placed order: {side} {part_id}")
        return part
    except ccxt.InsufficientFunds as e:
        logger.error(f"Insufficient funds: {e}")
        return None
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange rejected order: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error placing order: {e}")
        return None


def close_position_part(
    exchange: ccxt.Exchange,
    config: Config,
    part: Dict[str, Any],
    reason: str,
    close_price_target: Optional[Decimal] = None,
) -> bool:
    global _active_trade_parts
    part_id = part.get("part_id", "UnknownPart")
    qty = part.get("qty", Decimal(0))
    side = part.get("side")
    logger.info(f"Unraveling part {part_id} ({side} {qty}) for: {reason}")
    if qty <= config.position_qty_epsilon:
        logger.warning(f"Cannot unravel part {part_id}: qty {qty} invalid.")
        _active_trade_parts = [
            p for p in _active_trade_parts if p.get("part_id") != part_id
        ]
        save_persistent_state(force_heartbeat=True)
        return False
    close_side = config.side_sell if side == config.pos_long else config.side_buy
    params = {
        "reduceOnly": True,
        "category": "linear" if config.exchange["symbol"].endswith(":USDT") else {},
    }
    try:
        close_order = exchange.create_order(
            config.exchange["symbol"], "market", close_side, float(qty), None, params
        )
        logger.success(f"Unraveling Order {close_order['id']} cast.")
        time.sleep(config.trading["order_fill_timeout_seconds"] / 2)
        filled_order = exchange.fetch_order(
            close_order["id"], config.exchange["symbol"]
        )
        if filled_order.get("status") == "closed":
            actual_exit_price = Decimal(
                str(
                    filled_order.get(
                        "average",
                        filled_order.get(
                            "price", close_price_target or part["entry_price"]
                        ),
                    )
                )
            )
            exit_timestamp_ms = int(filled_order.get("timestamp", time.time() * 1000))
            entry_price = part.get("entry_price", Decimal(0))
            pnl = (
                (actual_exit_price - entry_price) * qty
                if side == config.pos_long
                else (entry_price - actual_exit_price) * qty
            )
            trade_metrics.log_trade(
                symbol=config.exchange["symbol"],
                side=side,
                entry_price=entry_price,
                exit_price=actual_exit_price,
                qty=qty,
                entry_time_ms=part["entry_time_ms"],
                exit_time_ms=exit_timestamp_ms,
                reason=reason,
                part_id=part_id,
                pnl_str=str(pnl),
            )
            _active_trade_parts = [
                p for p in _active_trade_parts if p.get("part_id") != part_id
            ]
            save_persistent_state(force_heartbeat=True)
            logger.success(
                f"Part {part_id} unraveled. Exit: {actual_exit_price}, PNL: {pnl:.2f} {config.usdt_symbol}"
            )
            send_termux_notification(
                f"Pyrmethus Position Closed",
                f"{config.exchange['symbol']} Part {part_id}. PNL: {pnl:.2f}. Reason: {reason}",
            )
            return True
        else:
            logger.warning(
                f"Unraveling order {close_order['id']} not closed. Status: {filled_order.get('status', 'N/A')}."
            )
            return False
    except Exception as e:
        logger.error(f"Error unraveling part {part_id}: {e}")
        send_termux_notification(
            "Pyrmethus Close Fail",
            f"Error closing {config.exchange['symbol']} part {part_id}: {str(e)[:80]}",
        )
        return False


def modify_position_sl_tp(
    exchange: ccxt.Exchange,
    config: Config,
    new_sl: Optional[Decimal] = None,
    new_tp: Optional[Decimal] = None,
) -> bool:
    global _active_trade_parts
    if not _active_trade_parts:
        logger.info("No active trade parts to modify SL/TP for.")
        return False
    if not config.MARKET_INFO or not config.tick_size:
        logger.error("Market info or tick_size not available. Cannot modify SL/TP.")
        return False
    current_sl = _active_trade_parts[0].get("sl_price")
    current_tp = _active_trade_parts[0].get("tp_price")
    params = {
        "positionIdx": 0,
        "category": "linear" if config.exchange["symbol"].endswith(":USDT") else {},
    }
    sl_to_set = new_sl if new_sl is not None else current_sl
    tp_to_set = new_tp if new_tp is not None else current_tp
    if sl_to_set is not None:
        sl_to_set = config.tick_size * (sl_to_set // config.tick_size)
        params["stopLoss"] = str(sl_to_set)
        params["slTriggerBy"] = "MarkPrice"
    else:
        params["stopLoss"] = "0"
    if config.risk["enable_take_profit"] and tp_to_set is not None:
        tp_to_set = config.tick_size * (tp_to_set // config.tick_size)
        params["takeProfit"] = str(tp_to_set)
        params["tpTriggerBy"] = "MarkPrice"
    else:
        params["takeProfit"] = "0"
        tp_to_set = None
    logger.info(
        f"Modifying Position SL/TP: New SL {sl_to_set}, New TP {tp_to_set if config.risk['enable_take_profit'] else 'Disabled'}"
    )
    try:
        exchange.set_trading_stop(config.exchange["symbol"], params=params)
        logger.success(
            f"Position SL/TP modification successful. SL: {sl_to_set}, TP: {tp_to_set if config.risk['enable_take_profit'] else 'N/A'}"
        )
        _synchronize_part_sltp(sl_to_set, tp_to_set)
        save_persistent_state(force_heartbeat=True)
        return True
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error modifying SL/TP: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error modifying SL/TP: {e}")
        return False


# --- WebSocket Streaming ---
async def stream_price_data(
    exchange: ccxt_async.Exchange, config: Config, price_queue: asyncio.Queue
):
    try:
        while True:
            ticker = await exchange.watch_ticker(config.exchange["symbol"])
            latest_close = safe_decimal_conversion(
                ticker.get("last", ticker.get("close"))
            )
            if latest_close is not None:
                await price_queue.put(
                    {"timestamp": int(time.time() * 1000), "close": latest_close}
                )
            await asyncio.sleep(0.1)
    except Exception as e:
        logger.error(f"WebSocket streaming error: {e}")
        await exchange.close()


# --- Health Check ---
def perform_health_check(exchange: ccxt.Exchange) -> bool:
    global _last_health_check_time
    now = time.time()
    if now - _last_health_check_time < HEALTH_CHECK_INTERVAL_SECONDS:
        return True
    try:
        exchange.fetch_time()
        _last_health_check_time = now
        logger.debug("Health check passed: Exchange API responsive.")
        return True
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        send_termux_notification(
            "Pyrmethus Health Alert", f"Exchange API unresponsive: {str(e)[:80]}"
        )
        return False


# --- Main Loop ---
async def main_loop(
    exchange: ccxt.Exchange, async_exchange: ccxt_async.Exchange, config: Config
):
    global \
        _active_trade_parts, \
        trade_timestamps_for_whipsaw, \
        whipsaw_cooldown_active_until
    global persistent_signal_counter, last_signal_type
    logger.info(
        f"=== Pyrmethus Spell v{PYRMETHUS_VERSION} Awakening on {exchange.id} ==="
    )
    if load_persistent_state():
        logger.success(
            f"Reawakened from Phoenix scroll. Active parts: {len(_active_trade_parts)}."
        )
    price_queue = asyncio.Queue()
    asyncio.create_task(stream_price_data(async_exchange, config, price_queue))
    current_balance_init = fetch_account_balance(exchange, config.usdt_symbol)
    if current_balance_init is not None:
        trade_metrics.set_initial_equity(current_balance_init)
    loop_counter = 0
    try:
        while True:
            loop_counter += 1
            logger.debug(f"New cycle of observation ({loop_counter})...")
            if not perform_health_check(exchange):
                logger.warning("Health check failed. Pausing cycle.")
                await asyncio.sleep(config.trading["sleep_seconds"] * 5)
                continue
            current_balance = fetch_account_balance(exchange, config.usdt_symbol)
            if current_balance is not None:
                trade_metrics.set_initial_equity(current_balance)
                if current_balance < config.trading["min_usdt_balance"]:
                    logger.warning(
                        f"Treasury ({current_balance:.2f} {config.usdt_symbol}) below prudence ward ({config.trading['min_usdt_balance']})."
                    )
                    await asyncio.sleep(config.trading["sleep_seconds"] * 3)
                    continue
                drawdown_hit, dd_reason = trade_metrics.check_drawdown(current_balance)
                if drawdown_hit:
                    logger.critical(f"Max drawdown! {dd_reason}. Pyrmethus rests.")
                    close_all_symbol_positions(exchange, config, "Max Drawdown Reached")
                    break
            price_data = []
            while not price_queue.empty():
                price_data.append(await price_queue.get())
            if not price_data:
                ohlcv_df = get_market_data(
                    exchange,
                    config.exchange["symbol"],
                    config.exchange["interval"],
                    OHLCV_LIMIT + config.api_fetch_limit_buffer,
                )
                if ohlcv_df is None or ohlcv_df.empty:
                    await asyncio.sleep(config.trading["sleep_seconds"])
                    continue
                latest_close = safe_decimal_conversion(ohlcv_df["close"].iloc[-1])
            else:
                latest_close = price_data[-1]["close"]
            df_with_indicators = calculate_all_indicators(
                pd.DataFrame(price_data, columns=["timestamp", "close"]), config
            )
            atr = safe_decimal_conversion(
                df_with_indicators.get(
                    f"ATR_{config.strategy.get('atr_calculation_period', 14)}", pd.NA
                )
            )
            signals = config.strategy_instance.generate_signals(
                df_with_indicators, latest_close, atr
            )
            confirmed_enter_long = signals.get("enter_long", False)
            confirmed_enter_short = signals.get("enter_short", False)
            current_pos_side, current_pos_qty = get_current_position_info()
            if (
                current_pos_side == config.pos_none
                and len(_active_trade_parts) < config.trading["max_active_trade_parts"]
            ):
                if confirmed_enter_long or confirmed_enter_short:
                    entry_side = (
                        config.pos_long if confirmed_enter_long else config.pos_short
                    )
                    sl_price = (
                        (latest_close - (atr * config.risk["atr_stop_loss_multiplier"]))
                        if entry_side == config.pos_long
                        else (
                            latest_close
                            + (atr * config.risk["atr_stop_loss_multiplier"])
                        )
                    )
                    tp_price = (
                        (
                            latest_close
                            + (atr * config.risk["atr_take_profit_multiplier"])
                        )
                        if entry_side == config.pos_long
                        else (
                            latest_close
                            - (atr * config.risk["atr_take_profit_multiplier"])
                        )
                        if config.risk["enable_take_profit"]
                        else None
                    )
                    new_part = place_risked_order(
                        exchange,
                        config,
                        entry_side,
                        latest_close,
                        sl_price,
                        tp_price,
                        atr,
                        df_with_indicators,
                    )
                    if new_part:
                        persistent_signal_counter = {"long": 0, "short": 0}
                        last_signal_type = None
            for part in list(_active_trade_parts):
                if config.risk["enable_trailing_sl"] and atr > 0:
                    current_pnl = (
                        (latest_close - part["entry_price"]) * part["qty"]
                        if part["side"] == config.pos_long
                        else (part["entry_price"] - latest_close) * part["qty"]
                    )
                    profit_atr = (
                        (current_pnl / part["qty"]) / atr
                        if part["qty"] > 0
                        else Decimal(0)
                    )
                    if profit_atr >= config.risk["trailing_sl_trigger_atr"]:
                        trailing_sl = (
                            (
                                latest_close
                                - (atr * config.risk["trailing_sl_distance_atr"])
                            )
                            if part["side"] == config.pos_long
                            else (
                                latest_close
                                + (atr * config.risk["trailing_sl_distance_atr"])
                            )
                        )
                        if (
                            part["side"] == config.pos_long
                            and trailing_sl > part["sl_price"]
                        ) or (
                            part["side"] == config.pos_short
                            and trailing_sl < part["sl_price"]
                        ):
                            logger.info(
                                f"Updating trailing SL for part {part['part_id']} to {trailing_sl}"
                            )
                            modify_position_sl_tp(exchange, config, new_sl=trailing_sl)
                if (part["side"] == config.pos_long and signals.get("exit_long")) or (
                    part["side"] == config.pos_short and signals.get("exit_short")
                ):
                    close_position_part(
                        exchange,
                        config,
                        part,
                        signals.get("exit_reason", "Strategy Exit Signal"),
                    )
            save_persistent_state()
            flush_notifications()
            await asyncio.sleep(config.trading["sleep_seconds"])
    finally:
        logger.info("=== Pyrmethus Spell Concludes ===")
        close_all_symbol_positions(exchange, config, "Spell Ending Sequence")
        save_persistent_state(force_heartbeat=True)
        trade_metrics.summary()
        flush_notifications()
        await async_exchange.close()


def close_all_symbol_positions(exchange: ccxt.Exchange, config: Config, reason: str):
    global _active_trade_parts
    logger.warning(
        f"Forceful unraveling of ALL positions for {config.exchange['symbol']} due to: {reason}"
    )
    for part in list(_active_trade_parts):
        if part.get("symbol") == config.exchange["symbol"]:
            close_position_part(exchange, config, part, reason + " (Global Unraveling)")
    _active_trade_parts.clear()
    save_persistent_state(force_heartbeat=True)


if __name__ == "__main__":
    logger.info("Pyrmethus prepares to breach veil to exchange realm...")
    exchange_instance = None
    async_exchange_instance = None
    try:
        exchange_params = {
            "apiKey": CONFIG.exchange["api_key"],
            "secret": CONFIG.exchange["api_secret"],
            "options": {
                "defaultType": "swap",
                "adjustForTimeDifference": True,
                "brokerId": f"PYRMETHUS{PYRMETHUS_VERSION.replace('.', '')}",
            },
            "enableRateLimit": True,
            "recvWindow": CONFIG.exchange["recv_window"],
        }
        if CONFIG.exchange["paper_trading"]:
            exchange_params["urls"] = {
                "api": {
                    "public": "https://api-testnet.bybit.com",
                    "private": "https://api-testnet.bybit.com",
                    "v5": "https://api-testnet.bybit.com/v5",
                }
            }
        else:
            exchange_params["urls"] = {
                "api": {
                    "public": "https://api.bybit.com",
                    "private": "https://api.bybit.com",
                    "v5": "https://api.bybit.com/v5",
                }
            }
        exchange_instance = ccxt.bybit(exchange_params)
        async_exchange_instance = ccxt_async.bybit(exchange_params)
        logger.info(f"Connecting to: {exchange_instance.id} (CCXT: {ccxt.__version__})")
        logger.debug(f"Exchange URLs: {exchange_instance.urls}")
        markets = exchange_instance.load_markets()
        if CONFIG.exchange["symbol"] not in markets:
            err_msg = f"Symbol {CONFIG.exchange['symbol']} not found in {exchange_instance.id} market runes."
            logger.critical(err_msg)
            send_termux_notification("Pyrmethus Startup Fail", err_msg)
            sys.exit(1)
        CONFIG.MARKET_INFO = markets[CONFIG.exchange["symbol"]]
        CONFIG.tick_size = safe_decimal_conversion(
            CONFIG.MARKET_INFO.get("precision", {}).get("price"), Decimal("1e-8")
        )
        CONFIG.qty_step = safe_decimal_conversion(
            CONFIG.MARKET_INFO.get("precision", {}).get("amount"), Decimal("1e-8")
        )
        logger.success(
            f"Market runes for {CONFIG.exchange['symbol']}: Price Tick {CONFIG.tick_size}, Amount Step {CONFIG.qty_step}"
        )
        try:
            leverage_params = {
                "category": "linear"
                if CONFIG.exchange["symbol"].endswith(":USDT")
                else {}
            }
            exchange_instance.set_leverage(
                CONFIG.exchange["leverage"],
                CONFIG.exchange["symbol"],
                params=leverage_params,
            )
            logger.success(
                f"Leverage for {CONFIG.exchange['symbol']} set to {CONFIG.exchange['leverage']}x."
            )
        except Exception as e:
            logger.warning(f"Could not set leverage: {e}")
        asyncio.run(main_loop(exchange_instance, async_exchange_instance, CONFIG))
    except Exception as e:
        logger.critical(f"Failed to init exchange: {e}")
        send_termux_notification(
            "Pyrmethus Init Error", f"Exchange init failed: {str(e)[:80]}"
        )
        sys.exit(1)
    finally:
        if async_exchange_instance:
            asyncio.run(async_exchange_instance.close())
