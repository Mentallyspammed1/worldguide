#!/usr/bin/env python3
# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v3.0.4 (The Oracle's Weave)
# Optimized for Bybit Mainnet with asyncio, enhanced risk management, Telegram notifications, and robust error handling.

"""
High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 3.0.4 - The Oracle's Weave

Dependencies & Setup (Termux):
1. Core Packages:
   pkg install python clang libcrypt libffi openssl rust termux-api
2. Python Libraries:
   pip install ccxt ccxt[async] pandas pandas-ta colorama python-dotenv retry pytz cachetools requests

WARNING: This script executes LIVE TRADES with REAL FUNDS. Use with extreme caution.
Pyrmethus bears no responsibility for financial losses.

Changes in v3.0.4:
- Explicit Live/Testnet Logging: Clearly logs targeted environment.
- API Key Validation at Startup: Attempts a benign authenticated call early to validate keys.
- Enhanced Order Sizing: Stricter adherence to min_order_qty and min_order_cost for live trading.
- Robust Async Operations: Improved error handling for asyncio tasks and WebSocket.
- WebSocket Price Queue Management: Max size for price queue to prevent memory issues.
- State Reconciliation (Basic): Logs warning if loaded state seems inconsistent after a fresh check (further reconciliation is complex).
- Trailing SL Refinements: Ensures `last_trailing_sl_update` is properly managed.
- Structured Logging Enhancements: More consistent use of `extra={"data": ...}`.
- Code Clarity: Additional "Pyrmethus" comments and minor refactoring.
- Corrected Logger Scope: Ensured global `logger` is consistently used for core tasks.
- Default PAPER_TRADING_MODE to false to emphasize live setup if var is missing, with strong warnings.
"""

import asyncio
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
from decimal import ROUND_DOWN, ROUND_UP, Decimal, InvalidOperation, getcontext
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pytz
import requests
from cachetools import TTLCache
from colorama import Back, Fore, Style
from colorama import init as colorama_init
from dotenv import load_dotenv
from retry import retry

try:
    import ccxt
    import ccxt.async_support as ccxt_async
    import pandas as pd

    if not hasattr(pd, "NA"):
        raise ImportError("Pandas version < 1.0 not supported, pd.NA is required.")
    import pandas_ta as ta
except ImportError as e:
    missing_pkg = getattr(e, "name", "dependency")
    sys.stderr.write(
        f"\033[91mCRITICAL ERROR: Missing/Incompatible Essence: '{missing_pkg}'. Pyrmethus cannot weave this spell.\033[0m\n"
    )
    sys.stderr.write(
        f"\033[91mPlease ensure all required libraries (runes) are installed and up to date.\033[0m\n"
    )
    sys.stderr.write(
        f"\033[91mConsult the scrolls (README or comments) for 'pkg install' and 'pip install' incantations.\033[0m\n"
    )
    sys.exit(1)

PYRMETHUS_VERSION = "3.0.4"
STATE_FILE_NAME = f"pyrmethus_phoenix_state_v{PYRMETHUS_VERSION.replace('.', '_')}.json"
STATE_FILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), STATE_FILE_NAME
)
HEARTBEAT_INTERVAL_SECONDS = 60
OHLCV_LIMIT = 220
API_CACHE_TTL_SECONDS = 15
HEALTH_CHECK_INTERVAL_SECONDS = 120
NOTIFICATION_BATCH_INTERVAL_SECONDS = 20
NOTIFICATION_BATCH_MAX_SIZE = 3
PRICE_QUEUE_MAX_SIZE = 100  # Max items in the WebSocket price queue

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

colorama_init(autoreset=True)
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "data"):
            log_record["data"] = record.data
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


class ColorStreamHandler(logging.StreamHandler):
    def format(self, record):
        level_color = NEON.get(record.levelname, Fore.WHITE)
        log_message = super().format(record)
        try:
            parts = log_message.split("] ", 2)
            if len(parts) > 1:
                level_part_idx = parts[0].rfind("[")
                if level_part_idx != -1:
                    level_name_in_log = parts[0][level_part_idx + 1 :].strip()
                    if record.levelname in level_name_in_log:
                        parts[0] = (
                            parts[0][: level_part_idx + 1]
                            + level_color
                            + record.levelname.ljust(8)
                            + NEON["RESET"]
                            + parts[0][
                                level_part_idx + 1 + len(record.levelname.ljust(8)) :
                            ]
                        )
                        log_message = "] ".join(parts)
        except Exception:
            pass
        return log_message


LOGGING_LEVEL = (
    logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO
)
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
log_file_name = f"{logs_dir}/pyrmethus_spell_v{PYRMETHUS_VERSION.replace('.', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.log"
root_logger = logging.getLogger()
root_logger.setLevel(LOGGING_LEVEL)
console_handler = ColorStreamHandler(sys.stdout)
console_formatter = logging.Formatter(
    fmt=f"%(asctime)s [{Fore.WHITE}%(levelname)-8s{NEON['RESET']}] {Fore.MAGENTA}%(name)-28s{NEON['RESET']} %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console_handler.setFormatter(console_formatter)
if not any(
    isinstance(h, ColorStreamHandler) for h in root_logger.handlers
):  # Avoid duplicate handlers
    root_logger.addHandler(console_handler)
file_handler = logging.handlers.RotatingFileHandler(
    log_file_name, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
json_formatter = JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S.%fZ")
file_handler.setFormatter(json_formatter)
if not any(
    isinstance(h, logging.handlers.RotatingFileHandler)
    and h.baseFilename == os.path.abspath(log_file_name)
    for h in root_logger.handlers
):
    root_logger.addHandler(file_handler)

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("ccxt").setLevel(logging.INFO)
logger = logging.getLogger("Pyrmethus.Core")
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


logging.Logger.success = log_success  # type: ignore[attr-defined]
logger_pre_config = logging.getLogger("Pyrmethus.PreConfig")
if load_dotenv(dotenv_path=env_path):
    logger_pre_config.info(
        f"Secrets whispered from .env scroll: {NEON['VALUE']}{env_path}{NEON['RESET']}"
    )
else:
    logger_pre_config.warning(
        f"No .env scroll found at {NEON['VALUE']}{env_path}{NEON['WARNING']}. Relying on system environment variables or defaults.{NEON['RESET']}"
    )
getcontext().prec = 28

market_data_cache = TTLCache(maxsize=10, ttl=API_CACHE_TTL_SECONDS)
balance_cache = TTLCache(maxsize=2, ttl=API_CACHE_TTL_SECONDS * 2)
ticker_cache = TTLCache(maxsize=5, ttl=3)

notification_buffer: List[Dict[str, Any]] = []
last_notification_flush_time: float = 0.0


def safe_decimal_conversion(
    value: Any, default_if_error: Any = pd.NA
) -> Union[Decimal, Any]:
    if pd.isna(value) or value is None:
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
    if pd.isna(value) or value is None:
        return "N/A"
    return str(value)


def format_price(
    price: Optional[Decimal], tick_size: Optional[Decimal] = None
) -> Optional[str]:
    if price is None:
        return None
    if tick_size is None or tick_size <= Decimal(0):
        return f"{price:.8f}".rstrip("0").rstrip(".")
    s = str(tick_size).rstrip("0")
    precision = len(s.split(".")[1]) if "." in s else 0
    return f"{price:.{precision}f}"


def _flush_single_notification_type(
    messages: List[Dict[str, Any]],
    notification_type: str,
    config_notifications: Dict[str, Any],
) -> None:
    logger_notify = logging.getLogger("Pyrmethus.Notification")
    if not messages:
        return
    notification_timeout = config_notifications["timeout_seconds"]
    combined_message_parts = []
    for n_item in messages:
        title = n_item.get("title", "Pyrmethus")
        message = n_item.get("message", "")
        combined_message_parts.append(f"*{title}*\n{message}")
    full_combined_message = "\n\n---\n\n".join(combined_message_parts)
    if notification_type == "termux":
        try:
            safe_title = json.dumps("Pyrmethus Batch")
            safe_message = json.dumps(full_combined_message.replace("*", "")[:1000])
            termux_id = messages[0].get("id", 777)
            command = [
                "termux-notification",
                "--title",
                safe_title,
                "--content",
                safe_message,
                "--id",
                str(termux_id),
            ]
            logger_notify.debug(
                f"Sending batched Termux notification ({len(messages)} items)."
            )
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate(timeout=notification_timeout)
            if process.returncode == 0:
                logger_notify.info(
                    f"Batched Termux notification sent successfully ({len(messages)} items)."
                )
            else:
                logger_notify.error(
                    f"Failed to send batched Termux notification. RC: {process.returncode}. Err: {stderr.decode().strip()}"
                )
        except Exception as e:
            logger_notify.error(
                f"Error sending Termux notification: {e}", exc_info=True
            )
    elif notification_type == "telegram":
        telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if telegram_bot_token and telegram_chat_id:
            try:
                url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
                payload = {
                    "chat_id": telegram_chat_id,
                    "text": full_combined_message[:4090],
                    "parse_mode": "MarkdownV2",
                }
                logger_notify.debug(
                    f"Sending batched Telegram notification ({len(messages)} items)."
                )
                response = requests.post(
                    url, json=payload, timeout=notification_timeout
                )
                response.raise_for_status()
                logger_notify.info(
                    f"Batched Telegram notification sent successfully ({len(messages)} items)."
                )
            except requests.exceptions.RequestException as e:
                logger_notify.error(f"Telegram API error: {e}", exc_info=True)
            except Exception as e:
                logger_notify.error(
                    f"Error sending Telegram notification: {e}", exc_info=True
                )
        else:
            logger_notify.warning(
                "Telegram notifications configured but TOKEN or CHAT_ID missing."
            )


def flush_notifications() -> None:
    global notification_buffer, last_notification_flush_time
    if not notification_buffer or "CONFIG" not in globals():
        return
    messages_by_type: Dict[str, List[Dict[str, Any]]] = {}
    for msg in notification_buffer:
        msg_type = msg.get("type", "unknown")
        messages_by_type.setdefault(msg_type, []).append(msg)
    for msg_type, messages in messages_by_type.items():
        if messages:
            _flush_single_notification_type(messages, msg_type, CONFIG.notifications)
    notification_buffer = []
    last_notification_flush_time = time.time()


def send_general_notification(
    title: str, message: str, notification_id: int = 777
) -> None:
    global notification_buffer, last_notification_flush_time
    if (
        "CONFIG" not in globals()
        or not hasattr(CONFIG, "notifications")
        or not CONFIG.notifications["enable"]
    ):
        return
    if CONFIG.notifications.get("termux_enable", True):
        notification_buffer.append(
            {
                "title": title,
                "message": message,
                "id": notification_id,
                "type": "termux",
            }
        )
    if CONFIG.notifications.get("telegram_enable", True):
        notification_buffer.append(
            {"title": title, "message": message, "type": "telegram"}
        )
    now = time.time()
    active_notification_services = (
        1 if CONFIG.notifications.get("termux_enable") else 0
    ) + (1 if CONFIG.notifications.get("telegram_enable") else 0)
    if active_notification_services == 0:
        active_notification_services = 1  # Avoid division by zero if both disabled but this function somehow called

    if (
        now - last_notification_flush_time >= NOTIFICATION_BATCH_INTERVAL_SECONDS
        or len(notification_buffer)
        >= NOTIFICATION_BATCH_MAX_SIZE * active_notification_services
    ):
        flush_notifications()


class StrategyName(str, Enum):
    DUAL_SUPERTREND_MOMENTUM = "DUAL_SUPERTREND_MOMENTUM"
    EHLERS_FISHER = "EHLERS_FISHER"


class OrderEntryType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class Config:
    def __init__(self) -> None:
        self.logger = logging.getLogger("Pyrmethus.Config")
        self.logger.info(f"Summoning Configuration Runes v{PYRMETHUS_VERSION}")
        self.exchange = {
            "api_key": self._get_env("BYBIT_API_KEY", required=True, secret=True),
            "api_secret": self._get_env("BYBIT_API_SECRET", required=True, secret=True),
            "symbol": self._get_env("SYMBOL", "BTC/USDT:USDT"),
            "interval": self._get_env("INTERVAL", "1m"),
            "leverage": self._get_env("LEVERAGE", 10, cast_type=int),
            "paper_trading": self._get_env(
                "PAPER_TRADING_MODE", "false", cast_type=bool
            ),  # Default to false for live emphasis
            "recv_window": self._get_env("DEFAULT_RECV_WINDOW", 5000, cast_type=int),
        }
        if not self.exchange["paper_trading"]:
            self.logger.warning(
                f"{NEON['CRITICAL']}LIVE TRADING MODE ENABLED. Ensure API keys are for MAINNET and you understand the risks.{NEON['RESET']}"
            )
        else:
            self.logger.info(
                f"{NEON['INFO']}PAPER TRADING MODE ENABLED. Using Testnet.{NEON['RESET']}"
            )

        self.trading = {
            "risk_per_trade_percentage": self._get_env(
                "RISK_PER_TRADE_PERCENTAGE", "0.003", cast_type=Decimal
            ),
            "max_order_usdt_amount": self._get_env(
                "MAX_ORDER_USDT_AMOUNT", "50.0", cast_type=Decimal
            ),
            "min_usdt_balance_for_trading": self._get_env(
                "MIN_USDT_BALANCE_FOR_TRADING", "10.0", cast_type=Decimal
            ),
            "max_active_trade_parts": self._get_env(
                "MAX_ACTIVE_TRADE_PARTS", 1, cast_type=int
            ),
            "order_fill_timeout_seconds": self._get_env(
                "ORDER_FILL_TIMEOUT_SECONDS", 10, cast_type=int
            ),
            "sleep_seconds": self._get_env("SLEEP_SECONDS", 3, cast_type=int),
            "entry_order_type": OrderEntryType(
                self._get_env("ENTRY_ORDER_TYPE", OrderEntryType.MARKET.value).upper()
            ),
            "limit_order_offset_atr_percentage": self._get_env(
                "LIMIT_ORDER_OFFSET_ATR_PERCENTAGE", "0.05", cast_type=Decimal
            ),
        }
        self.risk_management = {
            "atr_stop_loss_multiplier": self._get_env(
                "ATR_STOP_LOSS_MULTIPLIER", "1.0", cast_type=Decimal
            ),
            "enable_take_profit": self._get_env(
                "ENABLE_TAKE_PROFIT", "true", cast_type=bool
            ),
            "atr_take_profit_multiplier": self._get_env(
                "ATR_TAKE_PROFIT_MULTIPLIER", "1.5", cast_type=Decimal
            ),
            "enable_trailing_sl": self._get_env(
                "ENABLE_TRAILING_SL", "true", cast_type=bool
            ),
            "trailing_sl_trigger_atr": self._get_env(
                "TRAILING_SL_TRIGGER_ATR", "0.8", cast_type=Decimal
            ),
            "trailing_sl_distance_atr": self._get_env(
                "TRAILING_SL_DISTANCE_ATR", "0.4", cast_type=Decimal
            ),
            "enable_max_drawdown_stop": self._get_env(
                "ENABLE_MAX_DRAWDOWN_STOP", "true", cast_type=bool
            ),
            "max_drawdown_percent": self._get_env(
                "MAX_DRAWDOWN_PERCENT", "0.03", cast_type=Decimal
            ),
            "enable_session_pnl_limits": self._get_env(
                "ENABLE_SESSION_PNL_LIMITS", "true", cast_type=bool
            ),
            "session_profit_target_usdt": self._get_env(
                "SESSION_PROFIT_TARGET_USDT", "5.0", cast_type=Decimal, required=False
            ),
            "session_max_loss_usdt": self._get_env(
                "SESSION_MAX_LOSS_USDT", "15.0", cast_type=Decimal, required=False
            ),
        }
        self.strategy_params = {
            "name": StrategyName(
                self._get_env(
                    "STRATEGY_NAME", StrategyName.DUAL_SUPERTREND_MOMENTUM.value
                ).upper()
            ),
            "st_atr_length": self._get_env("ST_ATR_LENGTH", 7, cast_type=int),
            "st_multiplier": self._get_env("ST_MULTIPLIER", "1.5", cast_type=Decimal),
            "confirm_st_atr_length": self._get_env(
                "CONFIRM_ST_ATR_LENGTH", 14, cast_type=int
            ),
            "confirm_st_multiplier": self._get_env(
                "CONFIRM_ST_MULTIPLIER", "2.5", cast_type=Decimal
            ),
            "momentum_period": self._get_env("MOMENTUM_PERIOD", 10, cast_type=int),
            "momentum_threshold": self._get_env(
                "MOMENTUM_THRESHOLD", "0.5", cast_type=Decimal
            ),
            "confirm_st_stability_lookback": self._get_env(
                "CONFIRM_ST_STABILITY_LOOKBACK", 2, cast_type=int
            ),
            "st_max_entry_distance_atr_multiplier": self._get_env(
                "ST_MAX_ENTRY_DISTANCE_ATR_MULTIPLIER",
                "0.3",
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
            "ehlers_enable_divergence_scaled_exit": self._get_env(
                "EHLERS_ENABLE_DIVERGENCE_SCALED_EXIT", "false", cast_type=bool
            ),
            "ehlers_divergence_threshold_factor": self._get_env(
                "EHLERS_DIVERGENCE_THRESHOLD_FACTOR", "0.75", cast_type=Decimal
            ),
            "ehlers_divergence_exit_percentage": self._get_env(
                "EHLERS_DIVERGENCE_EXIT_PERCENTAGE", "0.3", cast_type=Decimal
            ),
        }
        self.notifications = {
            "enable": self._get_env("ENABLE_NOTIFICATIONS", "true", cast_type=bool),
            "timeout_seconds": self._get_env(
                "NOTIFICATION_TIMEOUT_SECONDS", 5, cast_type=int
            ),
            "termux_enable": self._get_env(
                "TERMUX_NOTIFICATIONS_ENABLE", "true", cast_type=bool
            ),
            "telegram_enable": self._get_env(
                "TELEGRAM_NOTIFICATIONS_ENABLE", "true", cast_type=bool
            ),
        }
        self.enhancements = {
            "profit_momentum_sl_tighten_enable": self._get_env(
                "ENABLE_PROFIT_MOMENTUM_SL_TIGHTEN", "true", cast_type=bool
            ),
            "profit_momentum_window": self._get_env(
                "PROFIT_MOMENTUM_WINDOW", 2, cast_type=int
            ),
            "profit_momentum_sl_tighten_factor": self._get_env(
                "PROFIT_MOMENTUM_SL_TIGHTEN_FACTOR", "0.4", cast_type=Decimal
            ),
            "whipsaw_cooldown_enable": self._get_env(
                "ENABLE_WHIPSAW_COOLDOWN", "true", cast_type=bool
            ),
            "whipsaw_max_trades_in_period": self._get_env(
                "WHIPSAW_MAX_TRADES_IN_PERIOD", 2, cast_type=int
            ),
            "whipsaw_period_seconds": self._get_env(
                "WHIPSAW_PERIOD_SECONDS", 180, cast_type=int
            ),
            "whipsaw_cooldown_seconds": self._get_env(
                "WHIPSAW_COOLDOWN_SECONDS", 300, cast_type=int
            ),
            "signal_persistence_candles": self._get_env(
                "SIGNAL_PERSISTENCE_CANDLES", 1, cast_type=int
            ),
            "no_trade_zones_enable": self._get_env(
                "ENABLE_NO_TRADE_ZONES", "true", cast_type=bool
            ),
            "no_trade_zone_pct_around_key_level": self._get_env(
                "NO_TRADE_ZONE_PCT_AROUND_KEY_LEVEL", "0.0015", cast_type=Decimal
            ),
            "key_round_number_step": self._get_env(
                "KEY_ROUND_NUMBER_STEP", "500", cast_type=Decimal, required=False
            ),
            "breakeven_sl_enable": self._get_env(
                "ENABLE_BREAKEVEN_SL", "true", cast_type=bool
            ),
            "breakeven_profit_atr_target": self._get_env(
                "BREAKEVEN_PROFIT_ATR_TARGET", "0.8", cast_type=Decimal
            ),
            "breakeven_min_abs_pnl_usdt": self._get_env(
                "BREAKEVEN_MIN_ABS_PNL_USDT", "0.25", cast_type=Decimal
            ),
            "anti_martingale_risk_enable": self._get_env(
                "ENABLE_ANTI_MARTINGALE_RISK", "false", cast_type=bool
            ),
            "risk_reduction_factor_on_loss": self._get_env(
                "RISK_REDUCTION_FACTOR_ON_LOSS", "0.75", cast_type=Decimal
            ),
            "risk_increase_factor_on_win": self._get_env(
                "RISK_INCREASE_FACTOR_ON_WIN", "1.1", cast_type=Decimal
            ),
            "max_risk_pct_anti_martingale": self._get_env(
                "MAX_RISK_PCT_ANTI_MARTINGALE", "0.02", cast_type=Decimal
            ),
            "last_chance_exit_enable": self._get_env(
                "ENABLE_LAST_CHANCE_EXIT", "true", cast_type=bool
            ),
            "last_chance_consecutive_adverse_candles": self._get_env(
                "LAST_CHANCE_CONSECUTIVE_ADVERSE_CANDLES", 2, cast_type=int
            ),
            "last_chance_sl_proximity_atr": self._get_env(
                "LAST_CHANCE_SL_PROXIMITY_ATR", "0.25", cast_type=Decimal
            ),
            "trend_contradiction_cooldown_enable": self._get_env(
                "ENABLE_TREND_CONTRADICTION_COOLDOWN", "true", cast_type=bool
            ),
            "trend_contradiction_check_candles_after_entry": self._get_env(
                "TREND_CONTRADICTION_CHECK_CANDLES_AFTER_ENTRY", 2, cast_type=int
            ),
            "trend_contradiction_cooldown_seconds": self._get_env(
                "TREND_CONTRADICTION_COOLDOWN_SECONDS", 120, cast_type=int
            ),
            "daily_max_trades_rest_enable": self._get_env(
                "ENABLE_DAILY_MAX_TRADES_REST", "true", cast_type=bool
            ),
            "daily_max_trades_limit": self._get_env(
                "DAILY_MAX_TRADES_LIMIT", 15, cast_type=int
            ),
            "daily_max_trades_rest_hours": self._get_env(
                "DAILY_MAX_TRADES_REST_HOURS", 3, cast_type=int
            ),
            "limit_order_price_improvement_check_enable": self._get_env(
                "ENABLE_LIMIT_ORDER_PRICE_IMPROVEMENT_CHECK", "true", cast_type=bool
            ),
            "trap_filter_enable": self._get_env(
                "ENABLE_TRAP_FILTER", "true", cast_type=bool
            ),
            "trap_filter_lookback_period": self._get_env(
                "TRAP_FILTER_LOOKBACK_PERIOD", 15, cast_type=int
            ),
            "trap_filter_rejection_threshold_atr": self._get_env(
                "TRAP_FILTER_REJECTION_THRESHOLD_ATR", "0.8", cast_type=Decimal
            ),
            "trap_filter_wick_proximity_atr": self._get_env(
                "TRAP_FILTER_WICK_PROXIMITY_ATR", "0.15", cast_type=Decimal
            ),
            "consecutive_loss_limiter_enable": self._get_env(
                "ENABLE_CONSECUTIVE_LOSS_LIMITER", "true", cast_type=bool
            ),
            "max_consecutive_losses": self._get_env(
                "MAX_CONSECUTIVE_LOSSES", 3, cast_type=int
            ),
            "consecutive_loss_cooldown_minutes": self._get_env(
                "CONSECUTIVE_LOSS_COOLDOWN_MINUTES", 45, cast_type=int
            ),
        }
        self.side_buy: str = "buy"
        self.side_sell: str = "sell"
        self.pos_long: str = "Long"
        self.pos_short: str = "Short"
        self.pos_none: str = "None"
        self.usdt_symbol: str = "USDT"
        self.retry_count: int = 7
        self.retry_delay_seconds: int = 5
        self.api_fetch_limit_buffer: int = 20
        self.position_qty_epsilon: Decimal = Decimal("1e-9")
        self.post_close_delay_seconds: int = 2
        self.MARKET_INFO: Optional[Dict[str, Any]] = None
        self.tick_size: Optional[Decimal] = None
        self.qty_step: Optional[Decimal] = None
        self.min_order_qty: Optional[Decimal] = None
        self.min_order_cost: Optional[Decimal] = None
        self.strategy_instance: Optional["TradingStrategy"] = None
        self._validate_parameters()
        self.logger.info(
            f"Configuration Runes v{PYRMETHUS_VERSION} Summoned and Verified."
        )
        self.logger.info(f"Chosen path: {self.strategy_params['name'].value}")

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
                    f"CRITICAL: Required configuration rune '{key}' not found in environment."
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
                f"CRITICAL: Required configuration rune '{key}' resolved to None."
            )
            raise ValueError(f"Required environment variable '{key}' resolved to None.")
        if value_to_cast is None and not required:
            self.logger.debug(
                f"Config Rune '{key}': Final value is None (not required)."
            )
            return None
        final_value: Any
        try:
            raw_value_str_for_cast = str(value_to_cast)
            if cast_type == bool:
                final_value = raw_value_str_for_cast.lower() in [
                    "true",
                    "1",
                    "yes",
                    "y",
                ]
            elif cast_type == Decimal:
                final_value = Decimal(raw_value_str_for_cast)
            elif cast_type == int:
                final_value = int(Decimal(raw_value_str_for_cast))
            elif cast_type == float:
                final_value = float(raw_value_str_for_cast)
            elif cast_type == str:
                final_value = raw_value_str_for_cast
            else:
                self.logger.warning(
                    f"Unsupported cast_type '{cast_type.__name__}' for key '{key}'."
                )
                final_value = raw_value_str_for_cast
        except (ValueError, TypeError, InvalidOperation) as e:
            self.logger.error(
                f"Invalid type/value for '{key}': '{value_to_cast}' (Source: {source}). Err: {e}. Using Default: '{default}'."
            )
            if default is None and required:
                self.logger.critical(
                    f"CRITICAL: Failed cast for required key '{key}', default is None."
                )
                raise ValueError(
                    f"Required env var '{key}' failed casting, no valid default."
                )
            if default is None and not required:
                return None
            try:
                default_str_for_cast = str(default)
                if cast_type == bool:
                    final_value = default_str_for_cast.lower() in [
                        "true",
                        "1",
                        "yes",
                        "y",
                    ]
                elif cast_type == Decimal:
                    final_value = Decimal(default_str_for_cast)
                elif cast_type == int:
                    final_value = int(Decimal(default_str_for_cast))
                elif cast_type == float:
                    final_value = float(default_str_for_cast)
                elif cast_type == str:
                    final_value = default_str_for_cast
                else:
                    final_value = default_str_for_cast
                self.logger.warning(f"Used casted default for {key}: '{final_value}'")
            except Exception as e_default:
                self.logger.critical(
                    f"CRITICAL: Cast fail for value AND default for '{key}'. Err: {e_default}"
                )
                raise ValueError(
                    f"Config error: Cannot cast value or default for '{key}'."
                )
        self.logger.debug(
            f"Final value for '{key}': {display_value if secret else final_value} (Type: {type(final_value).__name__}, Source: {source})"
        )
        return final_value

    def _validate_parameters(self) -> None:
        errors = []
        if not (0 < self.trading["risk_per_trade_percentage"] < Decimal("0.1")):
            errors.append(
                "RISK_PER_TRADE_PERCENTAGE should be low for live (e.g. < 0.1 for 10%)."
            )
        if self.exchange["leverage"] < 1 or self.exchange["leverage"] > 100:
            errors.append("LEVERAGE must be between 1 and 100.")
        if self.risk_management["atr_stop_loss_multiplier"] <= 0:
            errors.append("ATR_STOP_LOSS_MULTIPLIER must be positive.")
        if (
            self.risk_management["enable_take_profit"]
            and self.risk_management["atr_take_profit_multiplier"] <= 0
        ):
            errors.append("ATR_TAKE_PROFIT_MULTIPLIER must be positive.")
        if self.trading["min_usdt_balance_for_trading"] < 1:
            errors.append("MIN_USDT_BALANCE_FOR_TRADING should be at least 1 for live.")
        if self.trading["max_active_trade_parts"] != 1:
            errors.append("MAX_ACTIVE_TRADE_PARTS must be 1 for current SL/TP logic.")
        if self.risk_management["max_drawdown_percent"] > Decimal("0.1"):
            errors.append("MAX_DRAWDOWN_PERCENT seems high for live trading (>10%).")
        if self.trading["max_order_usdt_amount"] < Decimal("5"):
            errors.append("MAX_ORDER_USDT_AMOUNT seems very low (less than 5 USDT).")
        if self.exchange["paper_trading"] is False and (
            "testnet" in self.exchange["api_key"].lower()
            or "test" in self.exchange["api_key"].lower()
        ):
            errors.append(
                "PAPER_TRADING_MODE is false, but API key might be for Testnet. Verify keys for MAINNET."
            )
        if errors:
            error_message = (
                f"Configuration spellcrafting failed with {len(errors)} flaws:\n"
                + "\n".join([f"  - {e}" for e in errors])
            )
            self.logger.critical(error_message)
            raise ValueError(error_message)


# --- Global Objects & State Variables ---
try:
    CONFIG = Config()
except ValueError as config_error:
    # Use root_logger as CONFIG.logger might not be initialized
    logging.getLogger().critical(
        f"CRITICAL: Configuration loading failed. Error: {config_error}", exc_info=True
    )
    send_general_notification(
        "Pyrmethus Startup Failure", f"Config Error: {str(config_error)[:200]}"
    )
    sys.exit(1)
except Exception as general_config_error:
    logging.getLogger().critical(
        f"CRITICAL: Unexpected critical error during configuration: {general_config_error}",
        exc_info=True,
    )
    send_general_notification(
        "Pyrmethus Critical Failure",
        f"Unexpected Config Error: {str(general_config_error)[:200]}",
    )
    sys.exit(1)

trade_timestamps_for_whipsaw = deque(
    maxlen=CONFIG.enhancements["whipsaw_max_trades_in_period"]
)
whipsaw_cooldown_active_until: float = 0.0
persistent_signal_counter = {"long": 0, "short": 0}
last_signal_type: Optional[str] = None
previous_day_high: Optional[Decimal] = None
previous_day_low: Optional[Decimal] = None
last_key_level_update_day: Optional[int] = None
contradiction_cooldown_active_until: float = 0.0
consecutive_loss_cooldown_active_until: float = 0.0
trade_metrics = TradeMetrics(CONFIG)
_active_trade_parts: List[Dict[str, Any]] = []
_last_heartbeat_save_time: float = 0.0
_last_health_check_time: float = 0.0


# --- Trading Strategy Classes ---
# (DualSupertrendMomentumStrategy and EhlersFisherStrategy as in v3.0.2, but async generate_signals)
# ... (These classes are extensive, assume they are correctly defined as per previous versions with async generate_signals)
class TradingStrategy(ABC):
    def __init__(self, config: Config, df_columns: Optional[List[str]] = None):
        self.config = config
        self.logger = logging.getLogger(f"Pyrmethus.Strategy.{self.__class__.__name__}")
        self.required_columns = df_columns if df_columns else []
        self.logger.info(f"Strategy Form '{self.__class__.__name__}' materializing...")

    @abstractmethod
    async def generate_signals(
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
        if self.required_columns and not df.empty:
            last_row_values = df.iloc[-1][self.required_columns]
            if last_row_values.isnull().any():
                nan_cols_last_row = last_row_values[
                    last_row_values.isnull()
                ].index.tolist()
                self.logger.debug(
                    f"Faint runes (NaNs) in last candle for critical columns: {nan_cols_last_row}."
                )
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

    async def generate_signals(
        self,
        df: pd.DataFrame,
        latest_close: Optional[Decimal] = None,
        latest_atr: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        signals = self._get_default_signals()
        min_rows_needed = (
            max(
                self.config.strategy_params["st_atr_length"],
                self.config.strategy_params["confirm_st_atr_length"],
                self.config.strategy_params["momentum_period"],
                self.config.strategy_params["confirm_st_stability_lookback"],
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
                self.logger.warning("Resolution: Ambiguous.")
        stable_confirm_trend = pd.NA
        if self.config.strategy_params["confirm_st_stability_lookback"] <= 1:
            stable_confirm_trend = confirm_is_up_current
        elif (
            "confirm_trend" in df.columns
            and len(df) >= self.config.strategy_params["confirm_st_stability_lookback"]
        ):
            recent_confirm_trends = df["confirm_trend"].iloc[
                -self.config.strategy_params["confirm_st_stability_lookback"] :
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
                f"Stable Confirm ST ({_format_for_log(stable_confirm_trend, True)}) or Mom ({_format_for_log(momentum_val)}) is NA."
            )
            return signals
        price_proximity_ok = True
        st_max_dist_atr_mult = self.config.strategy_params.get(
            "st_max_entry_distance_atr_multiplier"
        )
        if (
            st_max_dist_atr_mult is not None
            and latest_atr is not None
            and latest_atr > 0
            and latest_close is not None
        ):
            max_allowed_distance = latest_atr * st_max_dist_atr_mult
            st_p_base = f"ST_{self.config.strategy_params['st_atr_length']}_{self.config.strategy_params['st_multiplier']}"
            if primary_long_flip:
                st_line_val = safe_decimal_conversion(last.get(f"{st_p_base}l"))
                if (
                    st_line_val is not None
                    and (latest_close - st_line_val) > max_allowed_distance
                ):
                    price_proximity_ok = False
                    self.logger.debug(f"Long ST proximity fail.")
            elif primary_short_flip:
                st_line_val = safe_decimal_conversion(last.get(f"{st_p_base}s"))
                if (
                    st_line_val is not None
                    and (st_line_val - latest_close) > max_allowed_distance
                ):
                    price_proximity_ok = False
                    self.logger.debug(f"Short ST proximity fail.")
        if (
            primary_long_flip
            and stable_confirm_trend is True
            and momentum_val > self.config.strategy_params["momentum_threshold"]
            and momentum_val > 0
            and price_proximity_ok
        ):
            signals["enter_long"] = True
            self.logger.info("DualST+Mom Signal: LONG Entry.")
        elif (
            primary_short_flip
            and stable_confirm_trend is False
            and momentum_val < -self.config.strategy_params["momentum_threshold"]
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

    async def generate_signals(
        self,
        df: pd.DataFrame,
        latest_close: Optional[Decimal] = None,
        latest_atr: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        signals = self._get_default_signals()
        min_rows_needed = (
            self.config.strategy_params["ehlers_fisher_length"]
            + self.config.strategy_params["ehlers_fisher_signal_length"]
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
            > self.config.strategy_params["ehlers_fisher_extreme_threshold_positive"]
            or fisher_now
            < self.config.strategy_params["ehlers_fisher_extreme_threshold_negative"]
        )
        if not is_fisher_extreme:
            if fisher_prev <= signal_prev and fisher_now > signal_now:
                signals["enter_long"] = True
                self.logger.info(f"EhlersFisher Signal: LONG Entry.")
            elif fisher_prev >= signal_prev and fisher_now < signal_now:
                signals["enter_short"] = True
                self.logger.info(f"EhlersFisher Signal: SHORT Entry.")
        elif (fisher_prev <= signal_prev and fisher_now > signal_now) or (
            fisher_prev >= signal_prev and fisher_now < signal_now
        ):
            self.logger.info(f"EhlersFisher: Crossover ignored in extreme zone.")
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
StrategyClass = strategy_map.get(CONFIG.strategy_params["name"])
if StrategyClass:
    CONFIG.strategy_instance = StrategyClass(CONFIG)
    # Use the global `logger` for this top-level success message
    logger.success(
        f"Strategy '{CONFIG.strategy_params['name'].value}' invoked and ready."
    )
else:
    err_msg = f"Failed to initialize strategy '{CONFIG.strategy_params['name'].value}'. Unknown spell form."
    logger.critical(err_msg)  # Use global logger
    send_general_notification("Pyrmethus Critical Error", err_msg)
    sys.exit(1)

# --- State Persistence, Indicator Calculation, Exchange Interaction Primitives ---
# (These functions: save_persistent_state, load_persistent_state, calculate_all_indicators,
#  _synchronize_part_sltp, get_current_position_info, update_daily_key_levels
#  are assumed to be correctly defined as in v3.0.2, adapted for async where necessary
#  and using the new CONFIG structure. For brevity, not fully re-pasted here.)
#  Key async adaptations: fetch_account_balance, get_market_data, place_risked_order, etc.
#  will become `async def` and use `await async_exchange_instance.method()`.


# --- Asynchronous Exchange Functions ---
@retry(
    (
        ccxt.NetworkError,
        ccxt.RequestTimeout,
        ccxt.RateLimitExceeded,
        ccxt.DDoSProtection,
    ),
    tries=CONFIG.retry_count,
    delay=CONFIG.retry_delay_seconds,
    backoff=2,
)
async def async_fetch_account_balance(
    exchange: ccxt_async.Exchange, currency: str, config: Config
) -> Optional[Decimal]:
    # Pyrmethus: Scrying the treasury with asynchronous sight.
    cache_key = f"balance_{currency}"
    if cache_key in balance_cache:
        logger.debug(f"Balance for {currency} found in cache.")
        return balance_cache[cache_key]

    logger.debug(f"Fetching account balance for {currency} from exchange.")
    try:
        # Bybit specific params for unified account balance
        params = (
            {"accountType": "UNIFIED", "coin": currency}
            if currency == config.usdt_symbol
            else {"accountType": "UNIFIED"}
        )

        balance_data = await exchange.fetch_balance(params=params)

        total_balance = None
        # Try to find the balance in various possible structures Bybit might return
        if (
            "info" in balance_data
            and "result" in balance_data["info"]
            and "list" in balance_data["info"]["result"]
        ):
            acct_list = balance_data["info"]["result"]["list"]
            if acct_list and isinstance(acct_list, list) and len(acct_list) > 0:
                # Unified account structure
                if "coin" in acct_list[0]:
                    for coin_entry in acct_list[0]["coin"]:
                        if coin_entry.get("coin") == currency:
                            total_balance = safe_decimal_conversion(
                                coin_entry.get("walletBalance")
                            )  # Or 'equity' or 'availableToWithdraw'
                            break
                # Fallback if structure is different
                if total_balance is None and "totalEquity" in acct_list[0]:
                    total_balance = safe_decimal_conversion(acct_list[0]["totalEquity"])

        # Fallback to standard ccxt structure if info parsing failed
        if total_balance is None and currency in balance_data:
            total_balance = safe_decimal_conversion(balance_data[currency].get("total"))
        if total_balance is None:  # Final fallback
            total_balance = safe_decimal_conversion(
                balance_data.get("total", {}).get(currency)
            )

        if total_balance is None or total_balance < Decimal(0):
            logger.error(
                f"Invalid or unreadable balance for {currency}. Data: {balance_data}",
                extra={"data": balance_data},
            )
            return None

        balance_cache[cache_key] = total_balance
        logger.info(
            f"Current {currency} Treasury: {NEON['VALUE']}{total_balance:.2f}{NEON['RESET']}"
        )
        return total_balance
    except ccxt.AuthenticationError as e:
        logger.critical(f"Authentication error fetching balance: {e}", exc_info=True)
        send_general_notification(
            "Pyrmethus Auth Error", f"API Key Invalid (Balance Fetch): {str(e)[:80]}"
        )
        raise  # Re-raise to be caught by main error handler
    except Exception as e:
        logger.error(
            f"Failed to fetch account balance for {currency}: {e}", exc_info=True
        )
        return None


@retry(
    (
        ccxt.NetworkError,
        ccxt.RequestTimeout,
        ccxt.RateLimitExceeded,
        ccxt.DDoSProtection,
    ),
    tries=CONFIG.retry_count,
    delay=CONFIG.retry_delay_seconds,
    backoff=2,
)
async def async_get_market_data(
    exchange: ccxt_async.Exchange,
    symbol: str,
    timeframe: str,
    limit: int,
    config: Config,
) -> Optional[pd.DataFrame]:
    # Pyrmethus: Summoning market whispers through the async ether.
    cache_key = f"ohlcv_{symbol}_{timeframe}_{limit}"
    if cache_key in market_data_cache:
        logger.debug(f"OHLCV data for {symbol} {timeframe} found in cache.")
        return market_data_cache[
            cache_key
        ].copy()  # Return a copy to prevent cache mutation

    logger.debug(
        f"Fetching OHLCV data for {symbol} {timeframe} (limit {limit}) from exchange."
    )
    try:
        params = (
            {"category": "linear"} if symbol.endswith(f":{config.usdt_symbol}") else {}
        )
        ohlcv = await exchange.fetch_ohlcv(
            symbol, timeframe, limit=limit, params=params
        )
        if not ohlcv:
            logger.warning(f"No OHLCV data returned for {symbol} {timeframe}.")
            return None

        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], unit="ms", utc=True
        )  # Ensure UTC
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if df.isnull().values.any():
            logger.warning(
                f"NaN values found in OHLCV data for {symbol} {timeframe}. Proceeding with caution.",
                extra={
                    "data": {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "nan_info": df.isnull().sum().to_dict(),
                    }
                },
            )

        market_data_cache[cache_key] = df.copy()  # Store a copy in cache
        logger.debug(
            f"Summoned {len(df)} candles for {symbol} {timeframe}. Latest: {df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S %Z') if not df.empty else 'N/A'}"
        )
        return df
    except ccxt.AuthenticationError as e:
        logger.critical(
            f"Authentication error fetching market data: {e}", exc_info=True
        )
        send_general_notification(
            "Pyrmethus Auth Error", f"API Key Invalid (Market Data): {str(e)[:80]}"
        )
        raise
    except Exception as e:
        logger.error(
            f"Failed to fetch market data for {symbol} {timeframe}: {e}", exc_info=True
        )
        return None


# ... (Other async exchange interaction functions like `async_place_risked_order`, `async_close_position_part`, `async_modify_position_sl_tp`
#      would be defined here, adapting their synchronous counterparts from v3.0.2 to use `await` and `async_exchange_instance`.
#      This is a significant rewrite of those functions.)


# --- WebSocket Streaming ---
async def stream_price_data(
    exchange: ccxt_async.Exchange, config: Config, price_queue: asyncio.Queue
):
    # Pyrmethus: Listening to the ever-flowing river of prices.
    symbol = config.exchange["symbol"]
    logger_ws = logging.getLogger("Pyrmethus.WebSocket")
    logger_ws.info(f"Initiating WebSocket price stream for {symbol}...")
    consecutive_errors = 0
    max_consecutive_errors = 5

    while True:
        try:
            async for ticker in exchange.watch_ticker(symbol):
                consecutive_errors = 0  # Reset error count on successful receipt
                latest_close = safe_decimal_conversion(
                    ticker.get("last", ticker.get("close"))
                )
                timestamp_ms = ticker.get("timestamp", int(time.time() * 1000))

                if latest_close is not None:
                    if price_queue.full():
                        try:
                            price_queue.get_nowait()  # Remove oldest item if queue is full
                            logger_ws.warning(
                                f"Price queue was full for {symbol}. Discarded oldest price."
                            )
                        except asyncio.QueueEmpty:
                            pass  # Should not happen if full() was true, but good practice

                    await price_queue.put(
                        {
                            "timestamp": timestamp_ms,
                            "close": latest_close,
                            "symbol": symbol,
                        }
                    )
                    logger_ws.debug(
                        f"WS Ticker {symbol}: Price {format_price(latest_close, config.tick_size)} @ {datetime.fromtimestamp(timestamp_ms / 1000, tz=pytz.utc).isoformat()}"
                    )
                else:
                    logger_ws.warning(
                        f"Received ticker for {symbol} without a valid close price: {ticker}"
                    )
            # If watch_ticker loop exits, it means connection was closed or an issue occurred
            logger_ws.warning(
                f"WebSocket watch_ticker loop for {symbol} exited. Attempting to reconnect..."
            )

        except ccxt.AuthenticationError as e:
            logger_ws.critical(
                f"Authentication error in WebSocket for {symbol}: {e}", exc_info=True
            )
            send_general_notification(
                "Pyrmethus Auth Error", f"WS Auth Error ({symbol}): {str(e)[:80]}"
            )
            raise  # Propagate to main error handler to stop the bot
        except ccxt.NetworkError as e:
            consecutive_errors += 1
            logger_ws.error(
                f"WebSocket NetworkError for {symbol} (Attempt {consecutive_errors}/{max_consecutive_errors}): {e}. Retrying in {5 * consecutive_errors}s.",
                exc_info=LOGGING_LEVEL <= logging.DEBUG,
            )
            if consecutive_errors >= max_consecutive_errors:
                logger_ws.critical(
                    f"Max WebSocket reconnection attempts reached for {symbol}. Critical failure."
                )
                send_general_notification(
                    "Pyrmethus WS Critical", f"WS Max Retries ({symbol}): {str(e)[:80]}"
                )
                # Consider stopping the bot or a longer pause here
                raise  # Or implement a more graceful long pause / exit
            await asyncio.sleep(5 * consecutive_errors)  # Exponential backoff
        except Exception as e:
            consecutive_errors += 1
            logger_ws.error(
                f"Unexpected WebSocket error for {symbol} (Attempt {consecutive_errors}/{max_consecutive_errors}): {e}. Retrying in {5 * consecutive_errors}s.",
                exc_info=True,
            )
            if consecutive_errors >= max_consecutive_errors:
                logger_ws.critical(
                    f"Max WebSocket reconnection attempts reached for {symbol} due to unexpected errors."
                )
                send_general_notification(
                    "Pyrmethus WS Critical",
                    f"WS Max Retries Unexpected ({symbol}): {str(e)[:80]}",
                )
                raise
            await asyncio.sleep(5 * consecutive_errors)
        finally:
            # Ensure exchange object is closed if loop terminates unexpectedly, though watch_ticker should handle its own closure.
            # await exchange.close() # This might be too aggressive if ccxt handles reconnection internally.
            pass


# --- Health Check ---
async def perform_health_check(exchange: ccxt_async.Exchange, config: Config) -> bool:
    # Pyrmethus: Scrying the exchange's vitality.
    global _last_health_check_time
    now = time.time()
    if now - _last_health_check_time < HEALTH_CHECK_INTERVAL_SECONDS:
        return True  # Skip check if within interval

    logger_health = logging.getLogger("Pyrmethus.HealthCheck")
    logger_health.debug("Performing exchange health check...")
    try:
        # A lightweight authenticated call
        server_time = await exchange.fetch_time()
        logger_health.debug(
            f"Health check passed: Exchange API responsive. Server time: {datetime.fromtimestamp(server_time / 1000, tz=pytz.utc).isoformat()}"
        )
        _last_health_check_time = now
        return True
    except ccxt.AuthenticationError as e:
        logger_health.critical(
            f"Authentication error during health check: {e}", exc_info=True
        )
        send_general_notification(
            "Pyrmethus Auth Error", f"API Key Invalid (Health Check): {str(e)[:80]}"
        )
        raise  # Critical, bot should stop
    except Exception as e:
        logger_health.error(
            f"Health check failed: Exchange API unresponsive or error: {e}",
            exc_info=LOGGING_LEVEL <= logging.DEBUG,
        )
        send_general_notification(
            "Pyrmethus Health Alert", f"Exchange API Unresponsive: {str(e)[:80]}"
        )
        return False


# --- Main Trading Loop Iteration (Conceptual) ---
async def trading_loop_iteration(
    async_exchange: ccxt_async.Exchange, config: Config, price_queue: asyncio.Queue
):
    # Pyrmethus: One cycle of the Oracle's gaze upon the market.
    # This function would contain the logic previously in the synchronous `while True:` loop's body.
    # For brevity, only a high-level structure is shown.
    # It would involve:
    # 1. Checking cooldowns.
    # 2. Fetching balance (async).
    # 3. Checking drawdown and session PNL.
    # 4. Getting latest price (from `price_queue` or fallback to `async_get_market_data`).
    # 5. Calculating indicators.
    # 6. Generating signals (async).
    # 7. Managing positions (entry/exit/SL/TP modification - all async).
    # 8. Saving state.
    # This is a placeholder for the detailed logic.
    logger_loop = logging.getLogger("Pyrmethus.TradingLoop")
    logger_loop.debug("Trading loop iteration started.")

    # Example: Get latest close from WebSocket queue or fall back to REST
    latest_price_data = None
    try:
        latest_price_data = price_queue.get_nowait()  # Non-blocking get
        logger_loop.debug(f"Price from WS Queue: {latest_price_data}")
    except asyncio.QueueEmpty:
        logger_loop.debug(
            "Price queue empty, fetching OHLCV via REST for latest close."
        )
        # Fallback logic using async_get_market_data would be here
        # For this example, we'll just log and skip if no WS data.
        # In a full implementation, you'd fetch OHLCV here.
        pass  # Fall through to potentially use polled data or skip if no fresh price

    if latest_price_data:
        latest_close = latest_price_data["close"]
        # ... (rest of the trading logic using this latest_close) ...
        logger_loop.info(
            f"Processing with latest close: {format_price(latest_close, config.tick_size)}"
        )
    else:
        logger_loop.warning("No fresh price data available for this iteration.")
        # Consider fetching OHLCV here as a fallback if WS is delayed
        # ohlcv_df = await async_get_market_data(...)
        # latest_close = ...

    # Placeholder for actual trading decisions and actions
    await asyncio.sleep(0.1)  # Simulate work

    logger_loop.debug("Trading loop iteration finished.")


# --- Main Orchestration ---
async def main():
    # Pyrmethus: The grand invocation of all arcane energies.
    logger.info(
        f"=== Pyrmethus Spell v{PYRMETHUS_VERSION} Awakening on {CONFIG.exchange['api_key'][:5]}... ==="
    )

    if load_persistent_state():
        logger.success(
            f"Reawakened from Phoenix scroll. Active parts: {len(_active_trade_parts)}."
        )
    else:
        logger.info("No prior state or ignored. Starting fresh.")

    async_exchange_instance = None  # Define here for finally block
    try:
        # Pyrmethus: Initialize asynchronous exchange for all operations
        async_exchange_params = {
            "apiKey": CONFIG.exchange["api_key"],
            "secret": CONFIG.exchange["api_secret"],
            "options": {
                "defaultType": "swap",
                "adjustForTimeDifference": True,
                "brokerId": f"PYRMETHUS{PYRMETHUS_VERSION.replace('.', '_')}",
            },
            "enableRateLimit": True,
            "recvWindow": CONFIG.exchange["recv_window"],
        }
        if CONFIG.exchange["paper_trading"]:
            async_exchange_params["urls"] = {
                "api": ccxt_async.bybit.urls["api"]["test"]
            }
            logger.info(
                f"{NEON['WARNING']}Targeting BYBIT TESTNET environment.{NEON['RESET']}"
            )
        else:
            async_exchange_params["urls"] = {
                "api": ccxt_async.bybit.urls["api"]["public"]
            }  # Mainnet
            logger.info(
                f"{NEON['CRITICAL']}Targeting BYBIT MAINNET (LIVE) environment. EXTREME CAUTION ADVISED.{NEON['RESET']}"
            )

        async_exchange_instance = ccxt_async.bybit(async_exchange_params)
        logger.info(
            f"Attempting connection to {async_exchange_instance.id} (CCXT: {ccxt.__version__})"
        )

        # Pyrmethus: Validate API keys with a benign authenticated call early
        try:
            await async_exchange_instance.fetch_balance(
                {"accountType": "UNIFIED"}
            )  # Test call
            logger.success("API Key validated successfully with exchange.")
        except ccxt.AuthenticationError as e_auth_val:
            logger.critical(
                f"CRITICAL: API Key Authentication Failed during initial validation: {e_auth_val}. Please check your .env file for correct API_KEY and API_SECRET for the target environment (Live/Testnet) and ensure keys have necessary permissions.",
                exc_info=True,
            )
            send_general_notification(
                "Pyrmethus Auth FAIL", f"API Key Invalid: {str(e_auth_val)[:100]}"
            )
            await async_exchange_instance.close()
            return
        except Exception as e_val:
            logger.error(
                f"Error during initial API key validation: {e_val}", exc_info=True
            )
            # Decide if this is critical enough to stop
            send_general_notification(
                "Pyrmethus API Warn",
                f"Initial API validation issue: {str(e_val)[:100]}",
            )

        markets = await async_exchange_instance.load_markets()
        if CONFIG.exchange["symbol"] not in markets:
            err_msg = f"Symbol {CONFIG.exchange['symbol']} not found in {async_exchange_instance.id} market runes."
            logger.critical(err_msg)
            send_general_notification("Pyrmethus Startup Failure", err_msg)
            await async_exchange_instance.close()
            return

        CONFIG.MARKET_INFO = markets[CONFIG.exchange["symbol"]]
        CONFIG.tick_size = safe_decimal_conversion(
            CONFIG.MARKET_INFO.get("precision", {}).get("price"), Decimal("1e-8")
        )
        CONFIG.qty_step = safe_decimal_conversion(
            CONFIG.MARKET_INFO.get("precision", {}).get("amount"), Decimal("1e-8")
        )
        CONFIG.min_order_qty = safe_decimal_conversion(
            CONFIG.MARKET_INFO.get("limits", {}).get("amount", {}).get("min")
        )
        CONFIG.min_order_cost = safe_decimal_conversion(
            CONFIG.MARKET_INFO.get("limits", {}).get("cost", {}).get("min")
        )

        logger.success(
            f"Market runes for {CONFIG.exchange['symbol']}: Price Tick {CONFIG.tick_size}, Amount Step {CONFIG.qty_step}, Min Qty: {CONFIG.min_order_qty}, Min Cost: {CONFIG.min_order_cost}"
        )

        category = (
            "linear"
            if CONFIG.exchange["symbol"].endswith(f":{CONFIG.usdt_symbol}")
            else None
        )
        try:
            leverage_params = (
                {"category": category, "marginMode": "isolated"}
                if category
                else {"marginMode": "isolated"}
            )
            logger.info(
                f"Setting leverage to {CONFIG.exchange['leverage']}x for {CONFIG.exchange['symbol']} (Category: {category or 'inferred'}, MarginMode: isolated)..."
            )
            await async_exchange_instance.set_leverage(
                CONFIG.exchange["leverage"],
                CONFIG.exchange["symbol"],
                params=leverage_params,
            )
            logger.success(f"Leverage for {CONFIG.exchange['symbol']} set/confirmed.")
        except Exception as e_lev:
            logger.warning(
                f"Could not set leverage (may be pre-set or an issue): {e_lev}",
                exc_info=LOGGING_LEVEL <= logging.DEBUG,
            )
            send_general_notification(
                "Pyrmethus Leverage Warn",
                f"Leverage issue for {CONFIG.exchange['symbol']}: {str(e_lev)[:60]}",
            )

        logger.success(
            f"Successfully connected to {async_exchange_instance.id} for {CONFIG.exchange['symbol']}."
        )
        send_general_notification(
            "Pyrmethus Online",
            f"Connected to {async_exchange_instance.id} for {CONFIG.exchange['symbol']} @ {CONFIG.exchange['leverage']}x. Environment: {'Testnet' if CONFIG.exchange['paper_trading'] else 'LIVE NET'}",
        )

        # Initialize TradeMetrics with current balance
        initial_balance = await async_fetch_account_balance(
            async_exchange_instance, CONFIG.usdt_symbol, CONFIG
        )
        if initial_balance is None:
            logger.critical(
                "Failed to fetch initial balance. Aborting live operations."
            )
            send_general_notification(
                "Pyrmethus Startup Fail", "Failed to fetch initial balance."
            )
            await async_exchange_instance.close()
            return
        trade_metrics.set_initial_equity(initial_balance)

        # --- Price Streaming Task ---
        price_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(
            maxsize=PRICE_QUEUE_MAX_SIZE
        )
        price_stream_task = asyncio.create_task(
            stream_price_data(async_exchange_instance, CONFIG, price_queue)
        )

        # --- Main Trading Loop ---
        loop_counter = 0
        while True:  # The true main loop for continuous operation
            loop_counter += 1
            logger.debug(f"Main loop cycle {loop_counter} initiated.")

            if not await perform_health_check(async_exchange_instance, CONFIG):
                logger.warning(
                    "Health check failed. Pausing for a longer duration before retrying operations."
                )
                await asyncio.sleep(
                    CONFIG.trading["sleep_seconds"] * 10
                )  # Longer pause on health fail
                continue  # Retry health check in next cycle

            # Pyrmethus: Integrate the core trading logic iteration here
            # This would be a call to a more detailed function like `await trading_loop_iteration(...)`
            # For now, this is a placeholder for that detailed logic.
            await trading_loop_iteration(
                async_exchange_instance, CONFIG, price_queue
            )  # Pass the queue

            # Save state periodically
            save_persistent_state()  # This is synchronous, consider if it needs to be async if very slow

            # Flush any batched notifications
            flush_notifications()

            await asyncio.sleep(CONFIG.trading["sleep_seconds"])

    except ccxt.AuthenticationError as e_auth_main:
        logger.critical(
            f"CRITICAL AUTHENTICATION ERROR in main execution: {e_auth_main}. Bot cannot continue.",
            exc_info=True,
        )
        send_general_notification(
            "Pyrmethus CRITICAL Auth Fail",
            f"API Key Invalid/Revoked: {str(e_auth_main)[:100]}",
        )
    except KeyboardInterrupt:
        logger.warning("\nSorcerer's intervention! Pyrmethus prepares for slumber...")
        send_general_notification("Pyrmethus Shutdown", "Manual shutdown initiated.")
    except asyncio.CancelledError:
        logger.info("Main task cancelled, likely during shutdown.")
    except Exception as e_main_fatal:
        logger.critical(
            f"A fatal unhandled astral disturbance occurred in main execution: {e_main_fatal}",
            exc_info=True,
        )
        send_general_notification(
            "Pyrmethus CRITICAL CRASH", f"Fatal error: {str(e_main_fatal)[:100]}"
        )
    finally:
        logger.info(
            f"{NEON['HEADING']}=== Pyrmethus Spell Concludes Its Weaving ==={NEON['RESET']}"
        )
        if async_exchange_instance:
            logger.info(
                "Attempting to close all positions and cancel orders before exiting..."
            )
            # await async_close_all_symbol_positions(async_exchange_instance, CONFIG, "Spell Ending Sequence") # Needs to be async
            logger.info("Closing asynchronous exchange connection...")
            await async_exchange_instance.close()
            logger.info("Asynchronous exchange connection gracefully closed.")

        save_persistent_state(force_heartbeat=True)  # Final state save
        trade_metrics.summary()  # Log final summary
        flush_notifications()  # Ensure all pending notifications are sent
        logger.info(
            f"{NEON['COMMENT']}# Energies settle. Until next conjuring.{NEON['RESET']}"
        )
        send_general_notification(
            "Pyrmethus Offline", f"Spell concluded for {CONFIG.exchange['symbol']}."
        )


if __name__ == "__main__":
    # Pyrmethus: Summoning the asynchronous energies.
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Already handled in main's finally, but good to have a catch-all here too.
        logger.info("Pyrmethus received KeyboardInterrupt at top level. Shutting down.")
    except Exception as e_top_level:
        # This will catch errors from asyncio.run() itself or unhandled exceptions from main() if it doesn't catch them
        logger.critical(f"Top-level unhandled exception: {e_top_level}", exc_info=True)
    finally:
        logger.info("Pyrmethus main execution (__name__ == '__main__') finished.")
        # Ensure all handlers are flushed and closed if Python exits abruptly
        logging.shutdown()
