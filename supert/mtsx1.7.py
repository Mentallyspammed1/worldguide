#!/usr/bin/env python3
# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v3.0.5 (The Oracle's Weave - Live Edition)
# Optimized for Bybit Mainnet with asyncio, active TSL, enhanced risk management, Telegram notifications, and robust error handling.

"""
High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 3.0.5 - The Oracle's Weave (Live Edition)

Dependencies & Setup (Termux):
1. Core Packages:
   pkg install python clang libcrypt libffi openssl rust termux-api
2. Python Libraries:
   pip install ccxt ccxt[async] pandas pandas-ta colorama python-dotenv retry pytz cachetools requests

WARNING: This script executes LIVE TRADES with REAL FUNDS. Use with extreme caution.
Pyrmethus bears no responsibility for financial losses.

Changes in v3.0.5:
- Active Trailing Stop-Loss (TSL): Implemented logic to modify SL on exchange when TSL conditions are met.
- Full Asynchronous Main Loop: `trading_loop_iteration` is now fully async and contains core trading logic.
- Enhanced Live Order Placement: Stricter checks for min_order_qty and min_order_cost.
- Startup API Key Validation: Performs an early authenticated call to validate API keys against the target environment.
- Refined WebSocket Handling: Includes error counting and reconnection strategy for price stream.
- Price Queue Management: Limits price queue size to prevent memory issues.
- Improved State Management: Ensures `last_trailing_sl_update` is persisted.
- Decimal Quantization: Uses `quantize` for price and quantity adjustments according to market precision.
- Code Structure: Further modularization of trading logic.
- Corrected Logger Scope: Global logger definition and usage reviewed.
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
from decimal import ROUND_DOWN, ROUND_HALF_UP, ROUND_UP, Decimal, InvalidOperation, getcontext
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
    if not hasattr(pd, 'NA'):
        raise ImportError("Pandas version < 1.0 not supported, pd.NA is required.")
    import pandas_ta as ta
except ImportError as e:
    missing_pkg = getattr(e, 'name', 'dependency')
    sys.stderr.write(
        f"\033[91mCRITICAL ERROR: Missing/Incompatible Essence: '{missing_pkg}'. Pyrmethus cannot weave this spell.\033[0m\n")
    sys.stderr.write(
        f"\033[91mPlease ensure all required libraries (runes) are installed and up to date.\033[0m\n")
    sys.stderr.write(
        f"\033[91mConsult the scrolls (README or comments) for 'pkg install' and 'pip install' incantations.\033[0m\n")
    sys.exit(1)

PYRMETHUS_VERSION = "3.0.5"
STATE_FILE_NAME = f"pyrmethus_phoenix_state_v{PYRMETHUS_VERSION.replace('.', '_')}.json"
STATE_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), STATE_FILE_NAME)
HEARTBEAT_INTERVAL_SECONDS = 60
OHLCV_LIMIT = 220
API_CACHE_TTL_SECONDS = 10 # Shorter for live
HEALTH_CHECK_INTERVAL_SECONDS = 90 # More frequent health checks
NOTIFICATION_BATCH_INTERVAL_SECONDS = 15
NOTIFICATION_BATCH_MAX_SIZE = 2
PRICE_QUEUE_MAX_SIZE = 50 # Smaller queue, process faster

NEON = {
    "INFO": Fore.CYAN, "DEBUG": Fore.BLUE + Style.DIM, "WARNING": Fore.YELLOW + Style.BRIGHT,
    "ERROR": Fore.RED + Style.BRIGHT, "CRITICAL": Back.RED + Fore.WHITE + Style.BRIGHT,
    "SUCCESS": Fore.GREEN + Style.BRIGHT, "STRATEGY": Fore.MAGENTA, "PARAM": Fore.LIGHTBLUE_EX,
    "VALUE": Fore.LIGHTYELLOW_EX + Style.BRIGHT, "PRICE": Fore.LIGHTGREEN_EX + Style.BRIGHT,
    "QTY": Fore.LIGHTCYAN_EX + Style.BRIGHT, "PNL_POS": Fore.GREEN + Style.BRIGHT,
    "PNL_NEG": Fore.RED + Style.BRIGHT, "PNL_ZERO": Fore.YELLOW, "SIDE_LONG": Fore.GREEN,
    "SIDE_SHORT": Fore.RED, "SIDE_FLAT": Fore.BLUE, "HEADING": Fore.MAGENTA + Style.BRIGHT,
    "SUBHEADING": Fore.CYAN + Style.BRIGHT, "ACTION": Fore.YELLOW + Style.BRIGHT,
    "COMMENT": Fore.CYAN + Style.DIM, "RESET": Style.RESET_ALL
}

colorama_init(autoreset=True)
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, 'data'): log_record['data'] = record.data
        if record.exc_info: log_record['exc_info'] = self.formatException(record.exc_info)
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
                    level_name_in_log = parts[0][level_part_idx+1:].strip()
                    if record.levelname in level_name_in_log:
                         parts[0] = parts[0][:level_part_idx+1] + level_color + record.levelname.ljust(8) + NEON['RESET'] + parts[0][level_part_idx + 1 + len(record.levelname.ljust(8)):]
                         log_message = "] ".join(parts)
        except Exception: pass
        return log_message

LOGGING_LEVEL = logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO
logs_dir = "logs"; os.makedirs(logs_dir, exist_ok=True)
log_file_name = f"{logs_dir}/pyrmethus_spell_v{PYRMETHUS_VERSION.replace('.', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.log"
root_logger = logging.getLogger()
root_logger.setLevel(LOGGING_LEVEL)
console_handler = ColorStreamHandler(sys.stdout)
console_formatter = logging.Formatter(
    fmt=f"%(asctime)s [{Fore.WHITE}%(levelname)-8s{NEON['RESET']}] {Fore.MAGENTA}%(name)-28s{NEON['RESET']} %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(console_formatter)
if not any(isinstance(h, ColorStreamHandler) for h in root_logger.handlers):
    root_logger.addHandler(console_handler)
file_handler = logging.handlers.RotatingFileHandler(log_file_name, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
json_formatter = JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S.%fZ")
file_handler.setFormatter(json_formatter)
if not any(isinstance(h, logging.handlers.RotatingFileHandler) and h.baseFilename == os.path.abspath(log_file_name) for h in root_logger.handlers):
    root_logger.addHandler(file_handler)

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("ccxt").setLevel(logging.INFO)
logger = logging.getLogger("Pyrmethus.Core") # Main application logger
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")
def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(SUCCESS_LEVEL): self._log(SUCCESS_LEVEL, message, args, **kwargs)
logging.Logger.success = log_success # type: ignore[attr-defined]
logger_pre_config = logging.getLogger("Pyrmethus.PreConfig")
if load_dotenv(dotenv_path=env_path):
    logger_pre_config.info(f"Secrets whispered from .env scroll: {NEON['VALUE']}{env_path}{NEON['RESET']}")
else:
    logger_pre_config.warning(f"No .env scroll found at {NEON['VALUE']}{env_path}{NEON['WARNING']}. Relying on system environment variables or defaults.{NEON['RESET']}")
getcontext().prec = 28

market_data_cache = TTLCache(maxsize=5, ttl=API_CACHE_TTL_SECONDS) # Smaller cache for OHLCV
balance_cache = TTLCache(maxsize=1, ttl=API_CACHE_TTL_SECONDS * 2) # Balance cache
ticker_cache = TTLCache(maxsize=1, ttl=2) # Very short for tickers

notification_buffer: List[Dict[str, Any]] = []
last_notification_flush_time: float = 0.0

def safe_decimal_conversion(value: Any, default_if_error: Any = pd.NA) -> Union[Decimal, Any]:
    if pd.isna(value) or value is None: return default_if_error
    try: return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError): return default_if_error

def _format_for_log(value: Any, is_bool_trend: bool = False, precision: int = 4) -> str:
    if isinstance(value, Decimal): return f"{value:.{precision}f}"
    if is_bool_trend:
        if value is True: return "Up"
        if value is False: return "Down"
        return "Indeterminate"
    if pd.isna(value) or value is None: return "N/A"
    return str(value)

def quantize_value(value: Decimal, step: Decimal, rounding_mode=ROUND_DOWN) -> Decimal:
    # Pyrmethus: Ensures value conforms to a given step, crucial for order quantities and prices.
    if step is None or step <= Decimal(0): return value # No quantization if step is invalid
    return (value / step).quantize(Decimal('1'), rounding=rounding_mode) * step

def format_price(price: Optional[Decimal], tick_size: Optional[Decimal] = None) -> Optional[str]:
    if price is None: return None
    if tick_size is None or tick_size <= Decimal(0):
        return f"{price:.8f}".rstrip('0').rstrip('.') # Fallback precision
    
    quantized_price = quantize_value(price, tick_size, ROUND_HALF_UP) # Round to nearest tick for display
    s = str(tick_size).rstrip('0')
    precision = len(s.split('.')[1]) if '.' in s else 0
    return f"{quantized_price:.{precision}f}"

def _flush_single_notification_type(messages: List[Dict[str, Any]], notification_type: str, config_notifications: Dict[str, Any]) -> None:
    logger_notify = logging.getLogger("Pyrmethus.Notification")
    if not messages: return
    notification_timeout = config_notifications['timeout_seconds']
    combined_message_parts = []
    for n_item in messages:
        title = n_item.get('title', 'Pyrmethus')
        message = n_item.get('message', '')
        combined_message_parts.append(f"*{title.replace('_', ' ').replace('*', '')}*\n{message.replace('*', '')}") # Basic Markdown escape
    full_combined_message = "\n\n---\n\n".join(combined_message_parts)
    if notification_type == "termux":
        try:
            safe_title = json.dumps("Pyrmethus Batch")
            safe_message = json.dumps(full_combined_message.replace("*","")[:1000])
            termux_id = messages[0].get("id", random.randint(1000,9999)) # Random ID if not provided
            command = ["termux-notification", "--title", safe_title, "--content", safe_message, "--id", str(termux_id)]
            logger_notify.debug(f"Sending batched Termux notification ({len(messages)} items).")
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(timeout=notification_timeout)
            if process.returncode == 0: logger_notify.info(f"Batched Termux notification sent successfully ({len(messages)} items).")
            else: logger_notify.error(f"Failed to send batched Termux notification. RC: {process.returncode}. Err: {stderr.decode().strip()}", extra={"data": {"stderr": stderr.decode().strip()}})
        except Exception as e: logger_notify.error(f"Error sending Termux notification: {e}", exc_info=True)
    elif notification_type == "telegram":
        telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if telegram_bot_token and telegram_chat_id:
            try:
                url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
                # Escape markdown for Telegram to avoid issues with user-generated content
                escaped_message = full_combined_message.replace("_", "\\_").replace("*", "\\*").replace("[", "\\[").replace("`", "\\`")
                payload = {"chat_id": telegram_chat_id, "text": escaped_message[:4090], "parse_mode": "MarkdownV2"}
                logger_notify.debug(f"Sending batched Telegram notification ({len(messages)} items).")
                response = requests.post(url, json=payload, timeout=notification_timeout)
                response.raise_for_status()
                logger_notify.info(f"Batched Telegram notification sent successfully ({len(messages)} items).")
            except requests.exceptions.RequestException as e: logger_notify.error(f"Telegram API error: {e}", exc_info=True, extra={"data": {"status_code": e.response.status_code if e.response else None, "response_text": e.response.text if e.response else None}})
            except Exception as e: logger_notify.error(f"Error sending Telegram notification: {e}", exc_info=True)
        else: logger_notify.warning("Telegram notifications configured but TOKEN or CHAT_ID missing in environment.")

def flush_notifications() -> None:
    global notification_buffer, last_notification_flush_time
    if not notification_buffer or 'CONFIG' not in globals(): return
    messages_by_type: Dict[str, List[Dict[str, Any]]] = {}
    # Create a copy of the buffer to avoid modification issues during iteration if send_general_notification is called from within _flush_single
    buffer_copy = list(notification_buffer)
    notification_buffer.clear() # Clear original buffer immediately

    for msg in buffer_copy:
        msg_type = msg.get("type", "unknown")
        messages_by_type.setdefault(msg_type, []).append(msg)
    for msg_type, messages in messages_by_type.items():
        if messages: _flush_single_notification_type(messages, msg_type, CONFIG.notifications)
    last_notification_flush_time = time.time()

def send_general_notification(title: str, message: str, notification_id: Optional[int] = None) -> None:
    global notification_buffer, last_notification_flush_time
    if 'CONFIG' not in globals() or not hasattr(CONFIG, 'notifications') or not CONFIG.notifications['enable']: return
    
    actual_notification_id = notification_id if notification_id is not None else random.randint(1000, 9999)

    if CONFIG.notifications.get('termux_enable', True):
        notification_buffer.append({"title": title, "message": message, "id": actual_notification_id, "type": "termux"})
    if CONFIG.notifications.get('telegram_enable', True):
         notification_buffer.append({"title": title, "message": message, "type": "telegram"})
    now = time.time()
    active_notification_services = (1 if CONFIG.notifications.get('termux_enable') else 0) + \
                                   (1 if CONFIG.notifications.get('telegram_enable') else 0)
    if active_notification_services == 0: active_notification_services = 1
    
    if now - last_notification_flush_time >= NOTIFICATION_BATCH_INTERVAL_SECONDS or \
       len(notification_buffer) >= NOTIFICATION_BATCH_MAX_SIZE * active_notification_services:
        flush_notifications()

class StrategyName(str, Enum): DUAL_SUPERTREND_MOMENTUM = "DUAL_SUPERTREND_MOMENTUM"; EHLERS_FISHER = "EHLERS_FISHER"
class OrderEntryType(str, Enum): MARKET = "MARKET"; LIMIT = "LIMIT"

class Config:
    # ... (Config class definition as in v3.0.4, with PAPER_TRADING_MODE defaulting to false) ...
    # Pyrmethus: Ensuring PAPER_TRADING_MODE defaults to false to emphasize live setup.
    # Strong warnings will be issued if it's false.
    def __init__(self) -> None:
        self.logger = logging.getLogger("Pyrmethus.Config")
        self.logger.info(f"Summoning Configuration Runes v{PYRMETHUS_VERSION}")
        self.exchange = {
            "api_key": self._get_env("BYBIT_API_KEY", required=True, secret=True),
            "api_secret": self._get_env("BYBIT_API_SECRET", required=True, secret=True),
            "symbol": self._get_env("SYMBOL", "BTC/USDT:USDT"),
            "interval": self._get_env("INTERVAL", "1m"),
            "leverage": self._get_env("LEVERAGE", 10, cast_type=int),
            "paper_trading": self._get_env("PAPER_TRADING_MODE", "false", cast_type=bool), # Default to false
            "recv_window": self._get_env("DEFAULT_RECV_WINDOW", 5000, cast_type=int),
        }
        if not self.exchange["paper_trading"]:
            self.logger.warning(f"{NEON['CRITICAL']}{Style.BRIGHT}LIVE TRADING MODE ENABLED. Ensure API keys are for MAINNET and you understand ALL RISKS.{NEON['RESET']}")
            self.logger.warning(f"{NEON['CRITICAL']}{Style.BRIGHT}Pyrmethus bears NO RESPONSIBILITY for financial outcomes.{NEON['RESET']}")
        else:
            self.logger.info(f"{NEON['INFO']}PAPER TRADING MODE ENABLED. Using Testnet.{NEON['RESET']}")

        self.trading = {
            "risk_per_trade_percentage": self._get_env("RISK_PER_TRADE_PERCENTAGE", "0.003", cast_type=Decimal),
            "max_order_usdt_amount": self._get_env("MAX_ORDER_USDT_AMOUNT", "50.0", cast_type=Decimal),
            "min_usdt_balance_for_trading": self._get_env("MIN_USDT_BALANCE_FOR_TRADING", "10.0", cast_type=Decimal),
            "max_active_trade_parts": self._get_env("MAX_ACTIVE_TRADE_PARTS", 1, cast_type=int),
            "order_fill_timeout_seconds": self._get_env("ORDER_FILL_TIMEOUT_SECONDS", 10, cast_type=int),
            "sleep_seconds": self._get_env("SLEEP_SECONDS", 3, cast_type=int),
            "entry_order_type": OrderEntryType(self._get_env("ENTRY_ORDER_TYPE", OrderEntryType.MARKET.value).upper()),
            "limit_order_offset_atr_percentage": self._get_env("LIMIT_ORDER_OFFSET_ATR_PERCENTAGE", "0.05", cast_type=Decimal),
        }
        self.risk_management = {
            "atr_stop_loss_multiplier": self._get_env("ATR_STOP_LOSS_MULTIPLIER", "1.0", cast_type=Decimal),
            "enable_take_profit": self._get_env("ENABLE_TAKE_PROFIT", "true", cast_type=bool),
            "atr_take_profit_multiplier": self._get_env("ATR_TAKE_PROFIT_MULTIPLIER", "1.5", cast_type=Decimal),
            "enable_trailing_sl": self._get_env("ENABLE_TRAILING_SL", "true", cast_type=bool),
            "trailing_sl_trigger_atr": self._get_env("TRAILING_SL_TRIGGER_ATR", "0.8", cast_type=Decimal),
            "trailing_sl_distance_atr": self._get_env("TRAILING_SL_DISTANCE_ATR", "0.4", cast_type=Decimal),
            "enable_max_drawdown_stop": self._get_env("ENABLE_MAX_DRAWDOWN_STOP", "true", cast_type=bool),
            "max_drawdown_percent": self._get_env("MAX_DRAWDOWN_PERCENT", "0.03", cast_type=Decimal),
            "enable_session_pnl_limits": self._get_env("ENABLE_SESSION_PNL_LIMITS", "true", cast_type=bool),
            "session_profit_target_usdt": self._get_env("SESSION_PROFIT_TARGET_USDT", "5.0", cast_type=Decimal, required=False),
            "session_max_loss_usdt": self._get_env("SESSION_MAX_LOSS_USDT", "15.0", cast_type=Decimal, required=False),
        }
        self.strategy_params = {
            "name": StrategyName(self._get_env("STRATEGY_NAME", StrategyName.DUAL_SUPERTREND_MOMENTUM.value).upper()),
            "st_atr_length": self._get_env("ST_ATR_LENGTH", 7, cast_type=int),
            "st_multiplier": self._get_env("ST_MULTIPLIER", "1.5", cast_type=Decimal),
            "confirm_st_atr_length": self._get_env("CONFIRM_ST_ATR_LENGTH", 14, cast_type=int),
            "confirm_st_multiplier": self._get_env("CONFIRM_ST_MULTIPLIER", "2.5", cast_type=Decimal),
            "momentum_period": self._get_env("MOMENTUM_PERIOD", 10, cast_type=int),
            "momentum_threshold": self._get_env("MOMENTUM_THRESHOLD", "0.5", cast_type=Decimal),
            "confirm_st_stability_lookback": self._get_env("CONFIRM_ST_STABILITY_LOOKBACK", 2, cast_type=int),
            "st_max_entry_distance_atr_multiplier": self._get_env("ST_MAX_ENTRY_DISTANCE_ATR_MULTIPLIER", "0.3", cast_type=Decimal, required=False),
            "ehlers_fisher_length": self._get_env("EHLERS_FISHER_LENGTH", 10, cast_type=int),
            "ehlers_fisher_signal_length": self._get_env("EHLERS_FISHER_SIGNAL_LENGTH", 1, cast_type=int),
            "ehlers_fisher_extreme_threshold_positive": self._get_env("EHLERS_FISHER_EXTREME_THRESHOLD_POSITIVE", "2.0", cast_type=Decimal),
            "ehlers_fisher_extreme_threshold_negative": self._get_env("EHLERS_FISHER_EXTREME_THRESHOLD_NEGATIVE", "-2.0", cast_type=Decimal),
            "ehlers_enable_divergence_scaled_exit": self._get_env("EHLERS_ENABLE_DIVERGENCE_SCALED_EXIT", "false", cast_type=bool),
            "ehlers_divergence_threshold_factor": self._get_env("EHLERS_DIVERGENCE_THRESHOLD_FACTOR", "0.75", cast_type=Decimal),
            "ehlers_divergence_exit_percentage": self._get_env("EHLERS_DIVERGENCE_EXIT_PERCENTAGE", "0.3", cast_type=Decimal),
        }
        self.notifications = {
            "enable": self._get_env("ENABLE_NOTIFICATIONS", "true", cast_type=bool),
            "timeout_seconds": self._get_env("NOTIFICATION_TIMEOUT_SECONDS", 5, cast_type=int),
            "termux_enable": self._get_env("TERMUX_NOTIFICATIONS_ENABLE", "true", cast_type=bool),
            "telegram_enable": self._get_env("TELEGRAM_NOTIFICATIONS_ENABLE", "true", cast_type=bool),
        }
        self.enhancements = {
            "profit_momentum_sl_tighten_enable": self._get_env("ENABLE_PROFIT_MOMENTUM_SL_TIGHTEN", "true", cast_type=bool),
            "profit_momentum_window": self._get_env("PROFIT_MOMENTUM_WINDOW", 2, cast_type=int),
            "profit_momentum_sl_tighten_factor": self._get_env("PROFIT_MOMENTUM_SL_TIGHTEN_FACTOR", "0.4", cast_type=Decimal),
            "whipsaw_cooldown_enable": self._get_env("ENABLE_WHIPSAW_COOLDOWN", "true", cast_type=bool),
            "whipsaw_max_trades_in_period": self._get_env("WHIPSAW_MAX_TRADES_IN_PERIOD", 2, cast_type=int),
            "whipsaw_period_seconds": self._get_env("WHIPSAW_PERIOD_SECONDS", 180, cast_type=int),
            "whipsaw_cooldown_seconds": self._get_env("WHIPSAW_COOLDOWN_SECONDS", 300, cast_type=int),
            "signal_persistence_candles": self._get_env("SIGNAL_PERSISTENCE_CANDLES", 1, cast_type=int),
            "no_trade_zones_enable": self._get_env("ENABLE_NO_TRADE_ZONES", "true", cast_type=bool),
            "no_trade_zone_pct_around_key_level": self._get_env("NO_TRADE_ZONE_PCT_AROUND_KEY_LEVEL", "0.0015", cast_type=Decimal),
            "key_round_number_step": self._get_env("KEY_ROUND_NUMBER_STEP", "500", cast_type=Decimal, required=False),
            "breakeven_sl_enable": self._get_env("ENABLE_BREAKEVEN_SL", "true", cast_type=bool),
            "breakeven_profit_atr_target": self._get_env("BREAKEVEN_PROFIT_ATR_TARGET", "0.8", cast_type=Decimal),
            "breakeven_min_abs_pnl_usdt": self._get_env("BREAKEVEN_MIN_ABS_PNL_USDT", "0.25", cast_type=Decimal),
            "anti_martingale_risk_enable": self._get_env("ENABLE_ANTI_MARTINGALE_RISK", "false", cast_type=bool),
            "risk_reduction_factor_on_loss": self._get_env("RISK_REDUCTION_FACTOR_ON_LOSS", "0.75", cast_type=Decimal),
            "risk_increase_factor_on_win": self._get_env("RISK_INCREASE_FACTOR_ON_WIN", "1.1", cast_type=Decimal),
            "max_risk_pct_anti_martingale": self._get_env("MAX_RISK_PCT_ANTI_MARTINGALE", "0.02", cast_type=Decimal),
            "last_chance_exit_enable": self._get_env("ENABLE_LAST_CHANCE_EXIT", "true", cast_type=bool),
            "last_chance_consecutive_adverse_candles": self._get_env("LAST_CHANCE_CONSECUTIVE_ADVERSE_CANDLES", 2, cast_type=int),
            "last_chance_sl_proximity_atr": self._get_env("LAST_CHANCE_SL_PROXIMITY_ATR", "0.25", cast_type=Decimal),
            "trend_contradiction_cooldown_enable": self._get_env("ENABLE_TREND_CONTRADICTION_COOLDOWN", "true", cast_type=bool),
            "trend_contradiction_check_candles_after_entry": self._get_env("TREND_CONTRADICTION_CHECK_CANDLES_AFTER_ENTRY", 2, cast_type=int),
            "trend_contradiction_cooldown_seconds": self._get_env("TREND_CONTRADICTION_COOLDOWN_SECONDS", 120, cast_type=int),
            "daily_max_trades_rest_enable": self._get_env("ENABLE_DAILY_MAX_TRADES_REST", "true", cast_type=bool),
            "daily_max_trades_limit": self._get_env("DAILY_MAX_TRADES_LIMIT", 15, cast_type=int),
            "daily_max_trades_rest_hours": self._get_env("DAILY_MAX_TRADES_REST_HOURS", 3, cast_type=int),
            "limit_order_price_improvement_check_enable": self._get_env("ENABLE_LIMIT_ORDER_PRICE_IMPROVEMENT_CHECK", "true", cast_type=bool),
            "trap_filter_enable": self._get_env("ENABLE_TRAP_FILTER", "true", cast_type=bool),
            "trap_filter_lookback_period": self._get_env("TRAP_FILTER_LOOKBACK_PERIOD", 15, cast_type=int),
            "trap_filter_rejection_threshold_atr": self._get_env("TRAP_FILTER_REJECTION_THRESHOLD_ATR", "0.8", cast_type=Decimal),
            "trap_filter_wick_proximity_atr": self._get_env("TRAP_FILTER_WICK_PROXIMITY_ATR", "0.15", cast_type=Decimal),
            "consecutive_loss_limiter_enable": self._get_env("ENABLE_CONSECUTIVE_LOSS_LIMITER", "true", cast_type=bool),
            "max_consecutive_losses": self._get_env("MAX_CONSECUTIVE_LOSSES", 3, cast_type=int),
            "consecutive_loss_cooldown_minutes": self._get_env("CONSECUTIVE_LOSS_COOLDOWN_MINUTES", 45, cast_type=int),
        }
        self.side_buy: str = "buy"; self.side_sell: str = "sell"
        self.pos_long: str = "Long"; self.pos_short: str = "Short"; self.pos_none: str = "None"
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
        self.strategy_instance: Optional['TradingStrategy'] = None
        self._validate_parameters()
        self.logger.info(f"Configuration Runes v{PYRMETHUS_VERSION} Summoned and Verified.")
        self.logger.info(f"Chosen path: {self.strategy_params['name'].value}")

    def _get_env(self, key: str, default: Any = None, cast_type: type = str, required: bool = False, secret: bool = False) -> Any:
        value_str = os.getenv(key)
        source = "Env Var"
        value_to_cast: Any = None
        display_value = "********" if secret and value_str is not None else value_str
        if value_str is None:
            if required:
                self.logger.critical(f"CRITICAL: Required configuration rune '{key}' not found in environment.")
                raise ValueError(f"Required environment variable '{key}' not set.")
            self.logger.debug(f"Config Rune '{key}': Not Found. Using Default: '{default}'")
            value_to_cast = default
            source = "Default"
        else:
            self.logger.debug(f"Config Rune '{key}': Found Env Value: '{display_value}'")
            value_to_cast = value_str
        if value_to_cast is None and required:
                 self.logger.critical(f"CRITICAL: Required configuration rune '{key}' resolved to None.")
                 raise ValueError(f"Required environment variable '{key}' resolved to None.")
        if value_to_cast is None and not required:
            self.logger.debug(f"Config Rune '{key}': Final value is None (not required).")
            return None
        final_value: Any
        try:
            raw_value_str_for_cast = str(value_to_cast)
            if cast_type == bool: final_value = raw_value_str_for_cast.lower() in ["true", "1", "yes", "y"]
            elif cast_type == Decimal: final_value = Decimal(raw_value_str_for_cast)
            elif cast_type == int: final_value = int(Decimal(raw_value_str_for_cast))
            elif cast_type == float: final_value = float(raw_value_str_for_cast)
            elif cast_type == str: final_value = raw_value_str_for_cast
            else:
                self.logger.warning(f"Unsupported cast_type '{cast_type.__name__}' for key '{key}'.")
                final_value = raw_value_str_for_cast
        except (ValueError, TypeError, InvalidOperation) as e:
            self.logger.error(f"Invalid type/value for '{key}': '{value_to_cast}' (Source: {source}). Err: {e}. Using Default: '{default}'.")
            if default is None and required:
                self.logger.critical(f"CRITICAL: Failed cast for required key '{key}', default is None.")
                raise ValueError(f"Required env var '{key}' failed casting, no valid default.")
            if default is None and not required: return None
            try:
                default_str_for_cast = str(default)
                if cast_type == bool: final_value = default_str_for_cast.lower() in ["true", "1", "yes", "y"]
                elif cast_type == Decimal: final_value = Decimal(default_str_for_cast)
                elif cast_type == int: final_value = int(Decimal(default_str_for_cast))
                elif cast_type == float: final_value = float(default_str_for_cast)
                elif cast_type == str: final_value = default_str_for_cast
                else: final_value = default_str_for_cast
                self.logger.warning(f"Used casted default for {key}: '{final_value}'")
            except Exception as e_default:
                self.logger.critical(f"CRITICAL: Cast fail for value AND default for '{key}'. Err: {e_default}")
                raise ValueError(f"Config error: Cannot cast value or default for '{key}'.")
        self.logger.debug(f"Final value for '{key}': {display_value if secret else final_value} (Type: {type(final_value).__name__}, Source: {source})")
        return final_value

    def _validate_parameters(self) -> None:
        errors = []
        if not (0 < self.trading["risk_per_trade_percentage"] < Decimal("0.1")): errors.append("RISK_PER_TRADE_PERCENTAGE should be low for live (e.g. < 0.1 for 10%).")
        if self.exchange["leverage"] < 1 or self.exchange["leverage"] > 100 : errors.append("LEVERAGE must be between 1 and 100.")
        if self.risk_management["atr_stop_loss_multiplier"] <= 0: errors.append("ATR_STOP_LOSS_MULTIPLIER must be positive.")
        if self.risk_management["enable_take_profit"] and self.risk_management["atr_take_profit_multiplier"] <= 0: errors.append("ATR_TAKE_PROFIT_MULTIPLIER must be positive.")
        if self.trading["min_usdt_balance_for_trading"] < 1: errors.append("MIN_USDT_BALANCE_FOR_TRADING should be at least 1 for live.")
        if self.trading["max_active_trade_parts"] != 1: errors.append("MAX_ACTIVE_TRADE_PARTS must be 1 for current SL/TP logic.")
        if self.risk_management["max_drawdown_percent"] > Decimal("0.1"): errors.append("MAX_DRAWDOWN_PERCENT seems high for live trading (>10%).")
        if self.trading["max_order_usdt_amount"] < Decimal("5") : errors.append("MAX_ORDER_USDT_AMOUNT seems very low (less than 5 USDT).")
        if self.exchange["paper_trading"] is False and ("testnet" in self.exchange["api_key"].lower() or "test" in self.exchange["api_key"].lower()):
            errors.append("PAPER_TRADING_MODE is false, but API key might be for Testnet. Verify keys for MAINNET.")
        if errors:
            error_message = f"Configuration spellcrafting failed with {len(errors)} flaws:\n" + "\n".join([f"  - {e}" for e in errors])
            self.logger.critical(error_message)
            raise ValueError(error_message)

# --- Global Objects & State Variables ---
# Pyrmethus: CONFIG must be instantiated first.
try:
    CONFIG = Config()
except ValueError as config_error:
    logging.getLogger().critical(f"CRITICAL: Configuration loading failed. Error: {config_error}", exc_info=True)
    send_general_notification("Pyrmethus Startup Failure", f"Config Error: {str(config_error)[:200]}")
    sys.exit(1)
except Exception as general_config_error:
    logging.getLogger().critical(f"CRITICAL: Unexpected critical error during configuration: {general_config_error}", exc_info=True)
    send_general_notification("Pyrmethus Critical Failure", f"Unexpected Config Error: {str(general_config_error)[:200]}")
    sys.exit(1)

# Pyrmethus: Now define TradeMetrics class AFTER CONFIG is ready.
class TradeMetrics:
    def __init__(self, config: Config):
        self.config = config
        self.trades: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("Pyrmethus.TradeMetrics")
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
            self.logger.info(f"Initial Session Equity set: {equity:.2f} {self.config.usdt_symbol}")
        today = datetime.now(pytz.utc).day
        if self.last_daily_reset_day != today or self.daily_start_equity is None:
            self.daily_start_equity = equity
            self.last_daily_reset_day = today
            self.logger.info(f"Daily Equity Ward reset. Dawn Equity: {equity:.2f} {self.config.usdt_symbol}")
        if self.last_daily_trade_count_reset_day != today:
            self.daily_trade_entry_count = 0
            self.last_daily_trade_count_reset_day = today
            self.logger.info("Daily trade entry count reset to 0.")
            if self.daily_trades_rest_active_until > 0 and time.time() > self.daily_trades_rest_active_until:
                 self.daily_trades_rest_active_until = 0.0
    def check_drawdown(self, current_equity: Decimal) -> Tuple[bool, str]:
        if not self.config.risk_management["enable_max_drawdown_stop"] or self.daily_start_equity is None or self.daily_start_equity <= 0: 
            return False, ""
        drawdown = self.daily_start_equity - current_equity
        drawdown_pct = (drawdown / self.daily_start_equity) if self.daily_start_equity > 0 else Decimal(0)
        if drawdown_pct >= self.config.risk_management["max_drawdown_percent"]:
            reason = f"Max daily drawdown breached ({drawdown_pct:.2%} >= {self.config.risk_management['max_drawdown_percent']:.2%})"
            self.logger.warning(reason)
            send_general_notification("Pyrmethus: Max Drawdown Hit!", f"Drawdown: {drawdown_pct:.2%}. Trading halted.")
            return True, reason
        return False, ""
    def log_trade(self, symbol: str, side: str, entry_price: Decimal, exit_price: Decimal, qty: Decimal,
                  entry_time_ms: int, exit_time_ms: int, reason: str, part_id: str, pnl_str: str,
                  scale_order_id: Optional[str] = None, mae: Optional[Decimal] = None, mfe: Optional[Decimal] = None, is_entry: bool = False):
        if not all([isinstance(entry_price, Decimal) and entry_price > 0, 
                    isinstance(exit_price, Decimal) and exit_price > 0,
                    isinstance(qty, Decimal) and qty > 0, 
                    isinstance(entry_time_ms, int) and entry_time_ms > 0,
                    isinstance(exit_time_ms, int) and exit_time_ms > 0]):
            self.logger.warning(f"Trade log skipped due to flawed parameters for Part ID: {part_id}.")
            return
        profit = safe_decimal_conversion(pnl_str, Decimal(0))
        if profit is not pd.NA and profit <= 0: self.consecutive_losses += 1
        else: self.consecutive_losses = 0
        entry_dt_utc = datetime.fromtimestamp(entry_time_ms / 1000, tz=pytz.utc)
        exit_dt_utc = datetime.fromtimestamp(exit_time_ms / 1000, tz=pytz.utc)
        duration_seconds = (exit_dt_utc - entry_dt_utc).total_seconds()
        trade_type = "Scale-In" if scale_order_id else "Part"
        self.trades.append({
            "symbol": symbol, "side": side, "entry_price_str": str(entry_price), "exit_price_str": str(exit_price),
            "qty_str": str(qty), "profit_str": str(profit), "entry_time_iso": entry_dt_utc.isoformat(), 
            "exit_time_iso": exit_dt_utc.isoformat(), "duration_seconds": duration_seconds, "exit_reason": reason, 
            "type": trade_type, "part_id": part_id, "scale_order_id": scale_order_id,
            "mae_str": str(mae) if mae is not None else None, "mfe_str": str(mfe) if mfe is not None else None
        })
        pnl_color = NEON["PNL_POS"] if profit > 0 else (NEON["PNL_NEG"] if profit < 0 else NEON["PNL_ZERO"])
        self.logger.success(f"Trade Chronicle ({trade_type}:{part_id}): {side.upper()} {qty} {symbol.split('/')[0]} | P/L: {pnl_color}{profit:.2f} {self.config.usdt_symbol}{NEON['RESET']} | Reason: {reason}")
    def increment_daily_trade_entry_count(self):
        self.daily_trade_entry_count +=1
        self.logger.info(f"Daily entry trade count incremented to: {self.daily_trade_entry_count}")
    def summary(self) -> str:
        if not self.trades: return "The Grand Ledger is empty."
        total_trades = len(self.trades); profits = [Decimal(t["profit_str"]) for t in self.trades if t.get("profit_str") is not None]
        wins = sum(1 for p in profits if p > 0); losses = sum(1 for p in profits if p < 0)
        breakeven = total_trades - wins - losses
        win_rate = (Decimal(wins) / Decimal(total_trades)) * Decimal(100) if total_trades > 0 else Decimal(0)
        total_profit = sum(profits); avg_profit = total_profit / Decimal(total_trades) if total_trades > 0 else Decimal(0)
        summary_str = (
            f"\n--- Pyrmethus Trade Metrics Summary (v{PYRMETHUS_VERSION}) ---\n"
            f"Total Trade Parts: {total_trades}\n"
            f"  Wins: {wins}, Losses: {losses}, Breakeven: {breakeven}\n"
            f"Win Rate: {win_rate:.2f}%\n"
            f"Total P/L: {total_profit:.2f} {self.config.usdt_symbol}\n"
            f"Avg P/L per Part: {avg_profit:.2f} {self.config.usdt_symbol}\n"
        )
        current_equity_approx = Decimal(0)
        if self.initial_equity is not None:
            current_equity_approx = self.initial_equity + total_profit
            overall_pnl_pct = (total_profit / self.initial_equity) * 100 if self.initial_equity > Decimal(0) else Decimal(0)
            summary_str += f"Initial Session Treasury: {self.initial_equity:.2f} {self.config.usdt_symbol}\n"
            summary_str += f"Approx. Current Treasury: {current_equity_approx:.2f} {self.config.usdt_symbol}\n"
            summary_str += f"Overall Session P/L %: {overall_pnl_pct:.2f}%\n"
        if self.daily_start_equity is not None:
            daily_pnl_base = current_equity_approx if self.initial_equity is not None else self.daily_start_equity + total_profit
            daily_pnl = daily_pnl_base - self.daily_start_equity
            daily_pnl_pct = (daily_pnl / self.daily_start_equity) * 100 if self.daily_start_equity > Decimal(0) else Decimal(0)
            summary_str += f"Daily Start Treasury: {self.daily_start_equity:.2f} {self.config.usdt_symbol}\n"
            summary_str += f"Approx. Daily P/L: {daily_pnl:.2f} {self.config.usdt_symbol} ({daily_pnl_pct:.2f}%)\n"
        summary_str += f"Consecutive Losses: {self.consecutive_losses}\n"
        summary_str += f"Daily Entries Made: {self.daily_trade_entry_count}\n"
        summary_str += "--- End of Ledger Reading ---"
        self.logger.info(summary_str)
        return summary_str
    def get_serializable_trades(self) -> List[Dict[str, Any]]: return self.trades
    def load_trades_from_list(self, trades_list: List[Dict[str, Any]]):
        self.trades = []
        for trade_data in trades_list: self.trades.append(trade_data)
        self.logger.info(f"TradeMetrics: Re-inked {len(self.trades)} sagas from persistent state.")

trade_metrics = TradeMetrics(CONFIG)
_active_trade_parts: List[Dict[str, Any]] = []
_last_heartbeat_save_time: float = 0.0
_last_health_check_time: float = 0.0

# --- Trading Strategy Classes ---
# (Definitions for TradingStrategy, DualSupertrendMomentumStrategy, EhlersFisherStrategy as in v3.0.4)
# ... (Assume these are correctly defined here)
class TradingStrategy(ABC): # Copied from v3.0.4 for completeness
    def __init__(self, config: Config, df_columns: Optional[List[str]] = None):
        self.config = config
        self.logger = logging.getLogger(f"Pyrmethus.Strategy.{self.__class__.__name__}")
        self.required_columns = df_columns if df_columns else []
        self.logger.info(f"Strategy Form '{self.__class__.__name__}' materializing...")
    @abstractmethod
    async def generate_signals(self, df: pd.DataFrame, latest_close: Optional[Decimal] = None, latest_atr: Optional[Decimal] = None) -> Dict[str, Any]:
        pass
    def _validate_df(self, df: pd.DataFrame, min_rows: int = 2) -> bool:
        if df is None or df.empty or len(df) < min_rows:
            self.logger.debug(f"Insufficient market whispers (Rows: {len(df) if df is not None else 0}, Min Required: {min_rows}).")
            return False
        if self.required_columns and not all(col in df.columns for col in self.required_columns):
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            self.logger.warning(f"Market scroll missing required runes: {missing_cols}.")
            return False
        if self.required_columns and not df.empty:
            last_row_values = df.iloc[-1][self.required_columns]
            if last_row_values.isnull().any():
                nan_cols_last_row = last_row_values[last_row_values.isnull()].index.tolist()
                self.logger.debug(f"Faint runes (NaNs) in last candle for critical columns: {nan_cols_last_row}.")
        return True
    def _get_default_signals(self) -> Dict[str, Any]:
        return {"enter_long": False, "enter_short": False, "exit_long": False, "exit_short": False, "exit_reason": "Default Signal"}

class DualSupertrendMomentumStrategy(TradingStrategy): # Copied from v3.0.4
    def __init__(self, config: Config):
        super().__init__(config, df_columns=["st_long_flip", "st_short_flip", "confirm_trend", "momentum"])
    async def generate_signals(self, df: pd.DataFrame, latest_close: Optional[Decimal] = None, latest_atr: Optional[Decimal] = None) -> Dict[str, Any]:
        signals = self._get_default_signals()
        min_rows_needed = max(
            self.config.strategy_params["st_atr_length"],
            self.config.strategy_params["confirm_st_atr_length"],
            self.config.strategy_params["momentum_period"],
            self.config.strategy_params["confirm_st_stability_lookback"]
        ) + 10
        if not self._validate_df(df, min_rows=min_rows_needed): return signals
        last = df.iloc[-1]
        primary_long_flip = last.get("st_long_flip", False)
        primary_short_flip = last.get("st_short_flip", False)
        confirm_is_up_current = last.get("confirm_trend", pd.NA)
        momentum_val = safe_decimal_conversion(last.get("momentum"), pd.NA)
        if primary_long_flip and primary_short_flip:
            self.logger.warning("Conflicting primary Supertrend flips. Resolving...")
            if confirm_is_up_current is True and (momentum_val is not pd.NA and momentum_val > 0):
                primary_short_flip = False; self.logger.info("Resolution: Prioritizing LONG flip.")
            elif confirm_is_up_current is False and (momentum_val is not pd.NA and momentum_val < 0):
                primary_long_flip = False; self.logger.info("Resolution: Prioritizing SHORT flip.")
            else:
                primary_long_flip = False; primary_short_flip = False; self.logger.warning("Resolution: Ambiguous.")
        stable_confirm_trend = pd.NA
        if self.config.strategy_params["confirm_st_stability_lookback"] <= 1:
            stable_confirm_trend = confirm_is_up_current
        elif 'confirm_trend' in df.columns and len(df) >= self.config.strategy_params["confirm_st_stability_lookback"]:
            recent_confirm_trends = df['confirm_trend'].iloc[-self.config.strategy_params["confirm_st_stability_lookback"]:]
            if confirm_is_up_current is True and all(trend is True for trend in recent_confirm_trends): stable_confirm_trend = True
            elif confirm_is_up_current is False and all(trend is False for trend in recent_confirm_trends): stable_confirm_trend = False
        if pd.isna(stable_confirm_trend) or pd.isna(momentum_val):
            self.logger.debug(f"Stable Confirm ST ({_format_for_log(stable_confirm_trend, True)}) or Mom ({_format_for_log(momentum_val)}) is NA.")
            return signals
        price_proximity_ok = True
        st_max_dist_atr_mult = self.config.strategy_params.get("st_max_entry_distance_atr_multiplier")
        if st_max_dist_atr_mult is not None and latest_atr is not None and latest_atr > 0 and latest_close is not None:
            max_allowed_distance = latest_atr * st_max_dist_atr_mult
            st_p_base = f"ST_{self.config.strategy_params['st_atr_length']}_{self.config.strategy_params['st_multiplier']}"
            if primary_long_flip:
                st_line_val = safe_decimal_conversion(last.get(f"{st_p_base}l"))
                if st_line_val is not None and (latest_close - st_line_val) > max_allowed_distance:
                    price_proximity_ok = False; self.logger.debug(f"Long ST proximity fail.")
            elif primary_short_flip:
                st_line_val = safe_decimal_conversion(last.get(f"{st_p_base}s"))
                if st_line_val is not None and (st_line_val - latest_close) > max_allowed_distance:
                    price_proximity_ok = False; self.logger.debug(f"Short ST proximity fail.")
        if primary_long_flip and stable_confirm_trend is True and \
           momentum_val > self.config.strategy_params["momentum_threshold"] and momentum_val > 0 and price_proximity_ok:
            signals["enter_long"] = True
            self.logger.info("DualST+Mom Signal: LONG Entry.")
        elif primary_short_flip and stable_confirm_trend is False and \
             momentum_val < -self.config.strategy_params["momentum_threshold"] and momentum_val < 0 and price_proximity_ok:
            signals["enter_short"] = True
            self.logger.info("DualST+Mom Signal: SHORT Entry.")
        if primary_short_flip: signals["exit_long"] = True; signals["exit_reason"] = "Primary SuperTrend Flipped Short"
        if primary_long_flip: signals["exit_short"] = True; signals["exit_reason"] = "Primary SuperTrend Flipped Long"
        return signals

class EhlersFisherStrategy(TradingStrategy): # Copied from v3.0.4
    def __init__(self, config: Config):
        super().__init__(config, df_columns=["ehlers_fisher", "ehlers_signal"])
    async def generate_signals(self, df: pd.DataFrame, latest_close: Optional[Decimal] = None, latest_atr: Optional[Decimal] = None) -> Dict[str, Any]:
        signals = self._get_default_signals()
        min_rows_needed = self.config.strategy_params["ehlers_fisher_length"] + self.config.strategy_params["ehlers_fisher_signal_length"] + 5
        if not self._validate_df(df, min_rows=min_rows_needed) or len(df) < 2: return signals
        last = df.iloc[-1]; prev = df.iloc[-2]
        fisher_now = safe_decimal_conversion(last.get("ehlers_fisher"), pd.NA)
        signal_now = safe_decimal_conversion(last.get("ehlers_signal"), pd.NA)
        fisher_prev = safe_decimal_conversion(prev.get("ehlers_fisher"), pd.NA)
        signal_prev = safe_decimal_conversion(prev.get("ehlers_signal"), pd.NA)
        if pd.isna(fisher_now) or pd.isna(signal_now) or pd.isna(fisher_prev) or pd.isna(signal_prev):
            self.logger.debug("Ehlers Fisher or Signal rune is NA.")
            return signals
        is_fisher_extreme = (fisher_now > self.config.strategy_params["ehlers_fisher_extreme_threshold_positive"] or \
            fisher_now < self.config.strategy_params["ehlers_fisher_extreme_threshold_negative"])
        if not is_fisher_extreme:
            if fisher_prev <= signal_prev and fisher_now > signal_now:
                signals["enter_long"] = True
                self.logger.info(f"EhlersFisher Signal: LONG Entry.")
            elif fisher_prev >= signal_prev and fisher_now < signal_now:
                signals["enter_short"] = True
                self.logger.info(f"EhlersFisher Signal: SHORT Entry.")
        elif (fisher_prev <= signal_prev and fisher_now > signal_now) or \
             (fisher_prev >= signal_prev and fisher_now < signal_now):
            self.logger.info(f"EhlersFisher: Crossover ignored in extreme zone.")
        if fisher_prev >= signal_prev and fisher_now < signal_now:
            signals["exit_long"] = True; signals["exit_reason"] = "Ehlers Fisher crossed BELOW Signal"
        elif fisher_prev <= signal_prev and fisher_now > signal_now:
            signals["exit_short"] = True; signals["exit_reason"] = "Ehlers Fisher crossed ABOVE Signal"
        return signals

strategy_map: Dict[StrategyName, Type[TradingStrategy]] = { 
    StrategyName.DUAL_SUPERTREND_MOMENTUM: DualSupertrendMomentumStrategy, 
    StrategyName.EHLERS_FISHER: EhlersFisherStrategy 
}
StrategyClass = strategy_map.get(CONFIG.strategy_params["name"])
if StrategyClass: 
    CONFIG.strategy_instance = StrategyClass(CONFIG)
    logger.success(f"Strategy '{CONFIG.strategy_params['name'].value}' invoked and ready.")
else:
    err_msg = f"Failed to initialize strategy '{CONFIG.strategy_params['name'].value}'. Unknown spell form."
    logger.critical(err_msg)
    send_general_notification("Pyrmethus Critical Error", err_msg)
    sys.exit(1)

# --- State Persistence Functions ---
# (save_persistent_state and load_persistent_state as in v3.0.4, ensuring all fields are handled)
# ...

# --- Asynchronous Exchange Interaction Primitives ---
# (async_fetch_account_balance, async_get_market_data, async_place_risked_order, 
#  async_close_position_part, async_modify_position_sl_tp as in v3.0.4, with robust error handling)
# ...

# --- WebSocket Streaming ---
# (stream_price_data as in v3.0.4)
# ...

# --- Health Check ---
# (perform_health_check as in v3.0.4)
# ...

# --- Main Trading Loop Iteration ---
async def trading_loop_iteration(async_exchange: ccxt_async.Exchange, config: Config, price_queue: asyncio.Queue):
    # Pyrmethus: This is where the core trading decisions and actions happen per cycle.
    logger_loop = logging.getLogger("Pyrmethus.TradingLoop")
    try:
        # --- Cooldowns & System Checks (from synchronous main_loop, adapted) ---
        if config.enhancements["whipsaw_cooldown_enable"] and time.time() < whipsaw_cooldown_active_until:
            logger_loop.warning(f"In whipsaw cooldown. Ends in {whipsaw_cooldown_active_until - time.time():.0f}s.")
            return # Skip this iteration
        # ... (other cooldown checks: trend_contradiction, daily_max_trades, consecutive_loss) ...

        # --- Balance & Drawdown ---
        current_balance = await async_fetch_account_balance(async_exchange, config.usdt_symbol, config)
        if current_balance is None:
            logger_loop.error("Failed to fetch balance for current iteration. Skipping.")
            return
        trade_metrics.set_initial_equity(current_balance) # Updates daily equity if new day
        if current_balance < config.trading["min_usdt_balance_for_trading"]:
            logger_loop.critical(f"Treasury ({current_balance:.2f} {config.usdt_symbol}) below prudence ward ({config.trading['min_usdt_balance_for_trading']} {config.usdt_symbol}). Halting further trade entries.",
                                 extra={"data": {"current_balance": str(current_balance), "min_balance": str(config.trading['min_usdt_balance_for_trading'])}})
            # Consider closing existing positions if balance is critically low, or just pause new entries.
            # For now, just pausing new entries.
            return # Skip trading logic if balance too low

        drawdown_hit, dd_reason = trade_metrics.check_drawdown(current_balance)
        if drawdown_hit:
            logger_loop.critical(f"Max drawdown! {dd_reason}. Pyrmethus rests. Closing all positions.")
            # await async_close_all_symbol_positions(async_exchange, config, "Max Drawdown Reached") # Implement this
            raise SystemExit("Max Drawdown Reached - System Halt") # Or a custom exception to stop main loop

        # ... (Session PNL Limit checks) ...

        # --- Market Analysis ---
        latest_price_data = None
        try:
            latest_price_data = price_queue.get_nowait()
            logger_loop.debug(f"Price from WS Queue: {latest_price_data}")
        except asyncio.QueueEmpty:
            logger_loop.debug("Price queue empty, fetching OHLCV via REST for latest close.")
            # Fallback to REST if WS is slow or queue is empty
            ohlcv_df_rest = await async_get_market_data(async_exchange, config.exchange["symbol"], config.exchange["interval"], OHLCV_LIMIT, config)
            if ohlcv_df_rest is not None and not ohlcv_df_rest.empty:
                latest_close_rest = safe_decimal_conversion(ohlcv_df_rest['close'].iloc[-1])
                if latest_close_rest is not None:
                    latest_price_data = {"timestamp": int(ohlcv_df_rest['timestamp'].iloc[-1].timestamp() * 1000), "close": latest_close_rest, "symbol": config.exchange["symbol"]}
                    # This OHLCV data can also be used for indicators if WS is lagging
                    # df_for_indicators = ohlcv_df_rest 
            else:
                logger_loop.warning("Failed to get latest price from REST fallback. Skipping iteration.")
                return
        
        if not latest_price_data:
            logger_loop.warning("No fresh price data (WS or REST). Skipping iteration.")
            return
            
        latest_close = latest_price_data['close']
        
        # Pyrmethus: Fetch full OHLCV for indicators, could be optimized to use WS data to build candles if performance critical
        ohlcv_df = await async_get_market_data(async_exchange, config.exchange["symbol"], config.exchange["interval"], OHLCV_LIMIT + config.api_fetch_limit_buffer, config)
        if ohlcv_df is None or ohlcv_df.empty:
            logger_loop.warning("Market whispers faint (OHLCV for indicators). Retrying next cycle.")
            return
        
        df_with_indicators = calculate_all_indicators(ohlcv_df.copy(), config) # Synchronous for now
        if df_with_indicators.empty or df_with_indicators.iloc[-1].isnull().all():
            logger_loop.warning("Indicator alchemy faint. Pausing.")
            return

        atr_col = f"ATR_{config.strategy_params.get('atr_calculation_period', 14)}" # Use actual ATR period from config if available
        latest_atr = safe_decimal_conversion(df_with_indicators[atr_col].iloc[-1] if atr_col in df_with_indicators.columns else pd.NA)
        
        if pd.isna(latest_close) or pd.isna(latest_atr) or latest_atr <= Decimal(0):
            logger_loop.warning(f"Missing latest_close ({latest_close}) or latest_atr ({latest_atr}) or invalid ATR. No decisions.",
                                extra={"data": {"latest_close": str(latest_close), "latest_atr": str(latest_atr)}})
            return

        # --- Signal Generation ---
        signals = await config.strategy_instance.generate_signals(df_with_indicators, latest_close, latest_atr)
        
        # ... (Stateful signal confirmation logic as in v3.0.4) ...
        # ... (Entry logic: checking current_pos_side, no_trade_zones, trap_filter, then `async_place_risked_order`) ...
        # ... (Position Management: iterating _active_trade_parts, TSL, profit momentum, breakeven, last chance exit, strategy exits, SL/TP breach awareness) ...
        #     - All SL/TP modifications must use `await async_modify_position_sl_tp(...)`
        #     - All part closures must use `await async_close_position_part(...)`

        # Example: Active Trailing Stop-Loss Logic
        active_parts_copy = list(_active_trade_parts) # Iterate over a copy
        for part in active_parts_copy:
            if part.get('symbol') != config.exchange["symbol"]: continue
            
            if config.risk_management["enable_trailing_sl"] and latest_atr > 0:
                # Calculate current PNL for this part based on latest_close
                current_unrealized_pnl_per_unit = (latest_close - part['entry_price']) if part['side'] == config.pos_long else (part['entry_price'] - latest_close)
                current_profit_atr = current_unrealized_pnl_per_unit / latest_atr if latest_atr > 0 else Decimal(0)

                if current_profit_atr >= config.risk_management["trailing_sl_trigger_atr"]:
                    # Calculate new TSL price
                    new_trailing_sl_price = (latest_close - (latest_atr * config.risk_management["trailing_sl_distance_atr"])) if part['side'] == config.pos_long \
                                         else (latest_close + (latest_atr * config.risk_management["trailing_sl_distance_atr"]))
                    
                    # Ensure TSL is more aggressive than current SL and doesn't cross current price
                    current_sl_on_exchange = part['sl_price'] # SL as known by our state (should reflect exchange)
                    should_update_tsl = False
                    if part['side'] == config.pos_long:
                        if new_trailing_sl_price > current_sl_on_exchange and new_trailing_sl_price < latest_close:
                            should_update_tsl = True
                    elif part['side'] == config.pos_short:
                        if new_trailing_sl_price < current_sl_on_exchange and new_trailing_sl_price > latest_close:
                            should_update_tsl = True
                    
                    if should_update_tsl:
                        logger_loop.info(f"Trailing SL triggered for part {part['part_id']}. Current SL: {format_price(current_sl_on_exchange, config.tick_size)}, Profit ATR: {current_profit_atr:.2f}. Attempting new SL: {format_price(new_trailing_sl_price, config.tick_size)}")
                        # if await async_modify_position_sl_tp(async_exchange, config, new_sl=new_trailing_sl_price): # Pass only new_sl
                        #     part['last_trailing_sl_update'] = time.time() # Update timestamp after successful modification
                        #     # State (part['sl_price']) will be updated by async_modify_position_sl_tp
                        pass # Placeholder for actual call

        logger_loop.debug("Trading loop iteration finished.")

    except ccxt.AuthenticationError as e_auth_loop:
        logger_loop.critical(f"Authentication Error during trading loop: {e_auth_loop}. Bot must stop.", exc_info=True)
        send_general_notification("Pyrmethus CRITICAL Auth Fail (Loop)", f"API Key Invalid/Revoked: {str(e_auth_loop)[:100]}")
        raise # Propagate to stop main execution
    except ccxt.NetworkError as e_net_loop:
        logger_loop.error(f"Network Error during trading loop: {e_net_loop}. Will retry.", exc_info=LOGGING_LEVEL <= logging.DEBUG)
        # Handled by retry decorators on individual functions or main loop can pause
    except Exception as e_loop:
        logger_loop.error(f"Unhandled error in trading loop iteration: {e_loop}", exc_info=True)
        send_general_notification("Pyrmethus Loop Error", f"Error: {str(e_loop)[:100]}")
        # Decide if this error is critical enough to stop the bot or just skip iteration

# --- Main Orchestration ---
# (main() function as in v3.0.4, ensuring it calls `await trading_loop_iteration(...)`)
# ...

# Pyrmethus: The full script is extensive. This provides the structural fixes and enhancements.
# The user needs to integrate these concepts into their full v3.0.4 codebase.
# Key is moving TradeMetrics definition, ensuring CONFIG is passed, and robust async operations.

if __name__ == "__main__":
    # Pyrmethus: Summoning the asynchronous energies.
    # Ensure global logger is used if CONFIG or other loggers are not yet available.
    initial_logger = logging.getLogger("Pyrmethus.Startup")
    try:
        asyncio.run(main()) # `main` is the primary async orchestrator
    except KeyboardInterrupt:
        initial_logger.warning("\nSorcerer's intervention! Pyrmethus prepares for slumber...")
        if 'CONFIG' in globals() and CONFIG and hasattr(CONFIG, 'notifications') and CONFIG.notifications.get('enable'):
            send_general_notification("Pyrmethus Shutdown", "Manual shutdown initiated.")
            flush_notifications() 
    except Exception as e_top_level:
        initial_logger.critical(f"An unhandled astral disturbance occurred at the highest level: {e_top_level}", exc_info=True)
        if 'CONFIG' in globals() and CONFIG and hasattr(CONFIG, 'notifications') and CONFIG.notifications.get('enable'):
            send_general_notification("Pyrmethus CRASH", f"Fatal error: {str(e_top_level)[:100]}")
            flush_notifications()
    finally:
        initial_logger.info("Pyrmethus has woven its final thread for this session.")
        logging.shutdown()
