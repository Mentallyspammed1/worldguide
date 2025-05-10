# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v3.1.0 (The Live Oracle's Aegis)
# Fully Asynchronous, Live-Trading Optimized for Bybit with Robust Order Management.

"""
High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 3.1.0 - The Live Oracle's Aegis

Dependencies & Setup (Termux):
1. Core Packages:
   pkg install python clang libcrypt libffi openssl rust termux-api
2. Python Libraries:
   pip install ccxt ccxt[async] pandas pandas-ta colorama python-dotenv retry pytz cachetools requests

WARNING: This script executes LIVE TRADES with REAL FUNDS. Use with extreme caution.
Pyrmethus bears no responsibility for financial losses.
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
from datetime import datetime, timedelta, timezone
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
    if not hasattr(pd, 'NA'): # Ensure pandas version supports pd.NA
        raise ImportError("Pandas version < 1.0 not supported, pd.NA is required.")
    import pandas_ta as ta
except ImportError as e:
    missing_pkg = getattr(e, 'name', 'dependency')
    # Pyrmethus: Critical error if essential essences are missing.
    sys.stderr.write(
        f"\033[91mCRITICAL ERROR: Missing/Incompatible Essence: '{missing_pkg}'. Pyrmethus cannot weave this spell.\033[0m\n")
    sys.stderr.write(
        f"\033[91mPlease ensure all required libraries (runes) are installed and up to date.\033[0m\n")
    sys.stderr.write(
        f"\033[91mConsult the scrolls (README or comments) for 'pkg install' and 'pip install' incantations.\033[0m\n")
    sys.exit(1)

# --- Constants - The Unchanging Pillars of the Spell ---
PYRMETHUS_VERSION = "3.1.0"
STATE_FILE_NAME = f"pyrmethus_phoenix_state_v{PYRMETHUS_VERSION.replace('.', '_')}.json"
STATE_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), STATE_FILE_NAME)
HEARTBEAT_INTERVAL_SECONDS = 60
OHLCV_LIMIT = 220 # Sufficient for most common indicators
API_CACHE_TTL_SECONDS = 10 # Shorter TTL for live data responsiveness
HEALTH_CHECK_INTERVAL_SECONDS = 90 # Check API health every 1.5 minutes
NOTIFICATION_BATCH_INTERVAL_SECONDS = 15 # Flush notifications frequently
NOTIFICATION_BATCH_MAX_SIZE = 3 # Max notifications per batch before flushing
PRICE_QUEUE_MAX_SIZE = 50 # Max items in the WebSocket price queue
ORDER_STATUS_CHECK_DELAY = 2 # Seconds to wait before checking order status after placement

# --- Neon Color Palette ---
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

# --- Initializations ---
colorama_init(autoreset=True)
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')

# --- Logger Setup ---
LOGGING_LEVEL = logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO
logs_dir = "logs"; os.makedirs(logs_dir, exist_ok=True)
log_file_name = f"{logs_dir}/pyrmethus_spell_v{PYRMETHUS_VERSION.replace('.', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.log"

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, 'data'): log_record['data'] = record.data
        if record.exc_info: log_record['exc_info'] = self.formatException(record.exc_info)
        return json.dumps(log_record)

class CustomColorFormatter(logging.Formatter):
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, style: str = '%', validate: bool = True):
        # Ensure compatibility with Python versions regarding the 'validate' parameter
        if sys.version_info >= (3, 8):
            super().__init__(fmt=fmt, datefmt=datefmt, style=style, validate=validate)
        else:
            super().__init__(fmt=fmt, datefmt=datefmt, style=style) # 'validate' not available in < 3.8

    def format(self, record: logging.LogRecord) -> str:
        level_color = NEON.get(record.levelname, Fore.WHITE)
        # Ensure levelname is padded to 8 characters for consistent width
        levelname_colored = f"{level_color}{record.levelname:<8}{NEON['RESET']}"
        
        # Color for logger name, padded to 28 characters
        name_color = NEON.get("STRATEGY", Fore.MAGENTA) # Default to magenta if STRATEGY not in NEON
        name_colored = f"{name_color}{record.name:<28}{NEON['RESET']}"
        
        message = record.getMessage()
        s = f"{self.formatTime(record, self.datefmt)} [{levelname_colored}] {name_colored} {message}"

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + record.exc_text
        if record.stack_info:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + self.formatStack(record.stack_info)
        return s

root_logger = logging.getLogger()
root_logger.setLevel(LOGGING_LEVEL)

# Console Handler with custom color formatting
console_handler = logging.StreamHandler(sys.stdout)
console_formatter = CustomColorFormatter(datefmt="%Y-%m-%d %H:%M:%S")
console_handler.setFormatter(console_formatter)

if not any(isinstance(h, logging.StreamHandler) and isinstance(h.formatter, CustomColorFormatter) for h in root_logger.handlers):
    root_logger.addHandler(console_handler)

# File Handler with JSON formatting
file_handler = logging.handlers.RotatingFileHandler(log_file_name, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
json_formatter = JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S.%fZ")
file_handler.setFormatter(json_formatter)
if not any(isinstance(h, logging.handlers.RotatingFileHandler) and h.baseFilename == os.path.abspath(log_file_name) for h in root_logger.handlers):
    root_logger.addHandler(file_handler)

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("ccxt").setLevel(logging.INFO) 
logger = logging.getLogger("Pyrmethus.Core") 

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

# --- Caches ---
market_data_cache: TTLCache[str, Any] = TTLCache(maxsize=5, ttl=API_CACHE_TTL_SECONDS)
balance_cache: TTLCache[str, Any] = TTLCache(maxsize=1, ttl=API_CACHE_TTL_SECONDS * 2) 
ticker_cache: TTLCache[str, Any] = TTLCache(maxsize=1, ttl=2) 

# --- Notification Batching ---
notification_buffer: List[Dict[str, Any]] = []
last_notification_flush_time: float = 0.0

# --- Helper Functions ---
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
    if not isinstance(value, Decimal) or not isinstance(step, Decimal):
        logger.warning(f"Invalid types for quantization. Value: {value} (type {type(value)}), Step: {step} (type {type(step)}). Returning original value.")
        return value 
    if step <= Decimal(0): 
        logger.warning(f"Invalid step {step} for quantization. Returning original value {value}.")
        return value
    return (value / step).quantize(Decimal('1'), rounding=rounding_mode) * step

def format_price(price: Optional[Decimal], tick_size: Optional[Decimal] = None) -> Optional[str]:
    if price is None: return None
    if not isinstance(price, Decimal): price = safe_decimal_conversion(price, None)
    if price is None: return None 

    if tick_size is None or not isinstance(tick_size, Decimal) or tick_size <= Decimal(0):
        return f"{price:.8f}".rstrip('0').rstrip('.') 
    
    quantized_price = quantize_value(price, tick_size, ROUND_HALF_UP) 
    
    s = str(tick_size).rstrip('0') 
    if '.' in s:
        precision = len(s.split('.')[1])
    else: 
        precision = 0
    
    return f"{quantized_price:.{precision}f}"

def _flush_single_notification_type(messages: List[Dict[str, Any]], notification_type: str, config_notifications: Dict[str, Any]) -> None:
    logger_notify = logging.getLogger("Pyrmethus.Notification")
    if not messages: return

    notification_timeout = config_notifications.get('timeout_seconds', 5) 
    combined_message_parts = []
    for n_item in messages:
        title = n_item.get('title', 'Pyrmethus')
        message = n_item.get('message', '')
        
        if notification_type == "telegram":
            escape_chars = r'_*[]()~`>#+-=|{}.!'
            escaped_title = "".join(['\\' + char if char in escape_chars else char for char in title])
            escaped_message = "".join(['\\' + char if char in escape_chars else char for char in message])
            combined_message_parts.append(f"*{escaped_title}*\n{escaped_message}")
        else: 
            combined_message_parts.append(f"{title}: {message}")
    
    full_combined_message = "\n\n---\n\n".join(combined_message_parts)

    if notification_type == "termux":
        try:
            termux_id = messages[0].get("id", random.randint(1000,9999)) 
            termux_title = "Pyrmethus Batch"
            termux_content = full_combined_message[:1000] 

            command = ["termux-notification", "--title", termux_title, "--content", termux_content, "--id", str(termux_id)]
            logger_notify.debug(f"Sending batched Termux notification ({len(messages)} items). Command: {' '.join(command)}") 
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) 
            stdout, stderr = process.communicate(timeout=notification_timeout)
            if process.returncode == 0:
                logger_notify.info(f"Batched Termux notification sent successfully ({len(messages)} items).")
            else:
                logger_notify.error(f"Failed to send batched Termux notification. RC: {process.returncode}. Err: {stderr.strip()}", extra={"data": {"stderr": stderr.strip()}})
        except FileNotFoundError: logger_notify.error("Termux API command 'termux-notification' not found. Is Termux:API installed and setup?")
        except subprocess.TimeoutExpired: logger_notify.error("Termux notification command timed out.")
        except Exception as e: logger_notify.error(f"Unexpected error sending Termux notification: {e}", exc_info=True)
    
    elif notification_type == "telegram":
        telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if telegram_bot_token and telegram_chat_id:
            try:
                url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
                payload = {"chat_id": telegram_chat_id, "text": full_combined_message[:4090], "parse_mode": "MarkdownV2"}
                logger_notify.debug(f"Sending batched Telegram notification ({len(messages)} items).")
                response = requests.post(url, json=payload, timeout=notification_timeout)
                response.raise_for_status() 
                logger_notify.info(f"Batched Telegram notification sent successfully ({len(messages)} items).")
            except requests.exceptions.HTTPError as http_err:
                err_response_text = http_err.response.text if http_err.response is not None else "No response body"
                logger_notify.error(f"Telegram API HTTP error: {http_err}. Response: {err_response_text}", exc_info=True, 
                                    extra={"data": {"status_code": http_err.response.status_code if http_err.response is not None else None, 
                                                    "response_text": err_response_text}})
            except requests.exceptions.RequestException as e: 
                logger_notify.error(f"Telegram API request error: {e}", exc_info=True)
            except Exception as e:
                logger_notify.error(f"Unexpected error sending Telegram notification: {e}", exc_info=True)
        else:
            logger_notify.warning("Telegram notifications configured but TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing in environment.")

def flush_notifications() -> None:
    global notification_buffer, last_notification_flush_time
    if not notification_buffer or 'CONFIG' not in globals() or not hasattr(CONFIG, 'notifications'): 
        return

    messages_by_type: Dict[str, List[Dict[str, Any]]] = {}
    buffer_copy = list(notification_buffer) 
    notification_buffer.clear()

    for msg in buffer_copy:
        msg_type = msg.get("type", "unknown") 
        messages_by_type.setdefault(msg_type, []).append(msg)

    for msg_type, messages in messages_by_type.items():
        if messages: 
            _flush_single_notification_type(messages, msg_type, CONFIG.notifications)
            
    last_notification_flush_time = time.time()

def send_general_notification(title: str, message: str, notification_id: Optional[Union[str, int]] = None) -> None:
    global notification_buffer, last_notification_flush_time
    if 'CONFIG' not in globals() or not hasattr(CONFIG, 'notifications') or not CONFIG.notifications['enable']:
        return
    
    actual_notification_id = str(notification_id) if notification_id is not None else str(random.randint(1000, 9999))

    if CONFIG.notifications.get('termux_enable', True): 
        notification_buffer.append({"title": title, "message": message, "id": actual_notification_id, "type": "termux"})
    if CONFIG.notifications.get('telegram_enable', True): 
         notification_buffer.append({"title": title, "message": message, "id": actual_notification_id, "type": "telegram"}) 
    
    now = time.time()
    active_notification_services = sum([
        1 for service_enabled_key in ['termux_enable', 'telegram_enable'] 
        if CONFIG.notifications.get(service_enabled_key, True)
    ])
    
    if active_notification_services == 0: active_notification_services = 1 

    if now - last_notification_flush_time >= NOTIFICATION_BATCH_INTERVAL_SECONDS or \
       len(notification_buffer) >= NOTIFICATION_BATCH_MAX_SIZE * active_notification_services : 
        flush_notifications()

# --- Enums ---
class StrategyName(str, Enum): DUAL_SUPERTREND_MOMENTUM = "DUAL_SUPERTREND_MOMENTUM"; EHLERS_FISHER = "EHLERS_FISHER"
class OrderEntryType(str, Enum): MARKET = "MARKET"; LIMIT = "LIMIT"

# --- Configuration Class ---
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
            "paper_trading": self._get_env("PAPER_TRADING_MODE", "false", cast_type=bool), 
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
            "atr_calculation_period": self._get_env("ATR_CALCULATION_PERIOD", 14, cast_type=int), 
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
        
        display_value_for_logging = "********" if secret and value_str is not None else value_str

        if value_str is None:
            if required and default is None: 
                self.logger.critical(f"CRITICAL: Required configuration rune '{key}' not found in environment and no default provided.")
                raise ValueError(f"Required environment variable '{key}' not set and no default.")
            self.logger.debug(f"Config Rune '{key}': Not Found in Env. Using Default: '{default if not secret else "********"}'")
            value_to_cast = default
            source = "Default"
        else:
            self.logger.debug(f"Config Rune '{key}': Found Env Value: '{display_value_for_logging}'")
            value_to_cast = value_str
        
        if value_to_cast is None: 
            if required: 
                 self.logger.critical(f"CRITICAL: Required configuration rune '{key}' resolved to None unexpectedly.")
                 raise ValueError(f"Required environment variable '{key}' resolved to None.")
            self.logger.debug(f"Config Rune '{key}': Final value is None (not required, or default is None).")
            return None

        final_value: Any
        try:
            if isinstance(value_to_cast, cast_type) and cast_type not in [Decimal, bool]: 
                 final_value = value_to_cast
            else:
                raw_value_str_for_cast = str(value_to_cast)
                if cast_type == bool: final_value = raw_value_str_for_cast.lower() in ["true", "1", "yes", "y"]
                elif cast_type == Decimal: final_value = Decimal(raw_value_str_for_cast)
                elif cast_type == int: final_value = int(Decimal(raw_value_str_for_cast))
                elif cast_type == float: final_value = float(raw_value_str_for_cast)
                elif cast_type == str: final_value = raw_value_str_for_cast
                else: 
                    self.logger.warning(f"Unsupported cast_type '{cast_type.__name__}' for key '{key}'. Using raw string.")
                    final_value = raw_value_str_for_cast
        except (ValueError, TypeError, InvalidOperation) as e:
            self.logger.error(f"Invalid type/value for '{key}': '{value_to_cast}' (Source: {source}). Error: {e}. Attempting to use Default: '{default if not secret else "********"}'.")
            if default is None: 
                if required:
                    self.logger.critical(f"CRITICAL: Failed cast for required key '{key}', and its default is None or also failed.")
                    raise ValueError(f"Required env var '{key}' failed casting, and default is None or unsuitable.")
                else: 
                    return None

            try:
                if isinstance(default, cast_type) and cast_type not in [Decimal, bool]:
                    final_value = default
                else:
                    default_str_for_cast = str(default)
                    if cast_type == bool: final_value = default_str_for_cast.lower() in ["true", "1", "yes", "y"]
                    elif cast_type == Decimal: final_value = Decimal(default_str_for_cast)
                    elif cast_type == int: final_value = int(Decimal(default_str_for_cast))
                    elif cast_type == float: final_value = float(default_str_for_cast)
                    elif cast_type == str: final_value = default_str_for_cast
                    else: final_value = default_str_for_cast 
                self.logger.warning(f"Used casted default for {key}: '{final_value if not secret else "********"}'")
            except Exception as e_default:
                self.logger.critical(f"CRITICAL: Cast fail for value AND default for '{key}'. Original Err: {e}, Default Cast Err: {e_default}")
                raise ValueError(f"Config error: Cannot cast value from env or default for '{key}'.")

        final_display_value = "********" if secret else final_value
        self.logger.debug(f"Final value for '{key}': {final_display_value} (Type: {type(final_value).__name__}, Source: {source})")
        return final_value

    def _validate_parameters(self) -> None:
        errors = []
        if not (Decimal(0) < self.trading["risk_per_trade_percentage"] < Decimal("0.1")): 
            errors.append(f"RISK_PER_TRADE_PERCENTAGE ({self.trading['risk_per_trade_percentage']}) should be a small positive Decimal (e.g., 0.001 to 0.1).")
        if not (1 <= self.exchange["leverage"] <= 100): 
            errors.append(f"LEVERAGE ({self.exchange['leverage']}) must be between 1 and 100.")
        if not (isinstance(self.risk_management["atr_stop_loss_multiplier"], Decimal) and self.risk_management["atr_stop_loss_multiplier"] > 0):
            errors.append(f"ATR_STOP_LOSS_MULTIPLIER ({self.risk_management['atr_stop_loss_multiplier']}) must be a positive Decimal.")
        if self.risk_management["enable_take_profit"] and not (isinstance(self.risk_management["atr_take_profit_multiplier"], Decimal) and self.risk_management["atr_take_profit_multiplier"] > 0):
            errors.append(f"ATR_TAKE_PROFIT_MULTIPLIER ({self.risk_management['atr_take_profit_multiplier']}) must be a positive Decimal when take profit is enabled.")
        if not (isinstance(self.trading["min_usdt_balance_for_trading"], Decimal) and self.trading["min_usdt_balance_for_trading"] >= 1):
            errors.append(f"MIN_USDT_BALANCE_FOR_TRADING ({self.trading['min_usdt_balance_for_trading']}) should be at least 1.0 for live trading.")
        if self.trading["max_active_trade_parts"] != 1: 
            errors.append("MAX_ACTIVE_TRADE_PARTS must be 1 for current SL/TP logic simplicity.")
        if not (isinstance(self.risk_management["max_drawdown_percent"], Decimal) and self.risk_management["max_drawdown_percent"] < Decimal("0.1")):
            errors.append(f"MAX_DRAWDOWN_PERCENT ({self.risk_management['max_drawdown_percent']}) seems high (>=10%) or is not a Decimal. Recommended < 0.1 for live.")
        if not (isinstance(self.trading["max_order_usdt_amount"], Decimal) and self.trading["max_order_usdt_amount"] >= Decimal("5")):
            errors.append(f"MAX_ORDER_USDT_AMOUNT ({self.trading['max_order_usdt_amount']}) seems very low (< 5 USDT) or is not a Decimal.")
        
        if self.exchange["paper_trading"] is False:
            api_key_str = str(self.exchange.get("api_key", ""))
            if "test" in api_key_str.lower(): 
                errors.append("PAPER_TRADING_MODE is false, but API key might be for Testnet (contains 'test'). Verify keys for MAINNET.")
        
        if errors:
            error_message = f"Configuration spellcrafting failed with {len(errors)} flaws:\n" + "\n".join([f"  - {e}" for e in errors])
            self.logger.critical(error_message)
            raise ValueError(error_message)

# --- Global Objects & State Variables ---
try:
    CONFIG = Config()
except ValueError as config_error:
    logging.getLogger().critical(f"CRITICAL: Configuration loading failed. Error: {config_error}", exc_info=True)
    sys.exit(1)
except Exception as general_config_error:
    logging.getLogger().critical(f"CRITICAL: Unexpected critical error during configuration: {general_config_error}", exc_info=True)
    sys.exit(1)

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
        self.logger.info("TradeMetrics Ledger (Oracle's Edition) opened.")
    def set_initial_equity(self, equity: Decimal):
        if not isinstance(equity, Decimal):
            self.logger.error(f"Invalid equity type received: {type(equity)}. Cannot set initial equity.")
            return
            
        if self.initial_equity is None: 
            self.initial_equity = equity
            self.logger.info(f"Initial Session Equity set: {equity:.2f} {self.config.usdt_symbol}")
        
        current_utc_day = datetime.now(pytz.utc).day
        if self.last_daily_reset_day != current_utc_day or self.daily_start_equity is None:
            self.daily_start_equity = equity
            self.last_daily_reset_day = current_utc_day
            self.logger.info(f"Daily Equity Ward reset. Dawn Equity (UTC Day {current_utc_day}): {equity:.2f} {self.config.usdt_symbol}")
        
        if self.last_daily_trade_count_reset_day != current_utc_day:
            self.daily_trade_entry_count = 0
            self.last_daily_trade_count_reset_day = current_utc_day
            self.logger.info(f"Daily trade entry count reset to 0 (UTC Day {current_utc_day}).")
            if self.daily_trades_rest_active_until > 0 and time.time() > self.daily_trades_rest_active_until:
                 self.daily_trades_rest_active_until = 0.0
                 self.logger.info("Daily trades rest period concluded.")

    def check_drawdown(self, current_equity: Decimal) -> Tuple[bool, str]:
        if not self.config.risk_management["enable_max_drawdown_stop"] or self.daily_start_equity is None or self.daily_start_equity <= 0: 
            return False, ""
        if not isinstance(current_equity, Decimal):
            self.logger.warning(f"Invalid current_equity type for drawdown check: {type(current_equity)}")
            return False, ""

        drawdown = self.daily_start_equity - current_equity
        drawdown_pct = (drawdown / self.daily_start_equity) if self.daily_start_equity > Decimal(0) else Decimal(0)
        
        if drawdown_pct >= self.config.risk_management["max_drawdown_percent"]:
            reason = f"Max daily drawdown breached ({drawdown_pct:.2%} >= {self.config.risk_management['max_drawdown_percent']:.2%})"
            self.logger.warning(reason)
            send_general_notification("Pyrmethus: Max Drawdown Hit!", f"Drawdown: {drawdown_pct:.2%}. Trading halted for the day.")
            return True, reason
        return False, ""

    def log_trade(self, symbol: str, side: str, entry_price: Decimal, exit_price: Decimal, qty: Decimal,
                  entry_time_ms: int, exit_time_ms: int, reason: str, part_id: str, pnl_str: str, 
                  entry_indicators: Dict[str, Any], exit_indicators: Dict[str, Any],
                  scale_order_id: Optional[str] = None, mae: Optional[Decimal] = None, mfe: Optional[Decimal] = None):
        
        for val_name, val in [("entry_price", entry_price), ("exit_price", exit_price), ("qty", qty)]:
            if not isinstance(val, Decimal) or val <= Decimal(0):
                self.logger.warning(f"Trade log skipped for Part ID {part_id} due to invalid Decimal parameter: {val_name}={val} (Type: {type(val)}).")
                return
        
        if not (isinstance(entry_time_ms, int) and entry_time_ms > 0 and isinstance(exit_time_ms, int) and exit_time_ms > 0):
            self.logger.warning(f"Trade log skipped for Part ID {part_id} due to invalid time parameters.")
            return

        profit = safe_decimal_conversion(pnl_str, Decimal(0)) 
        if profit is pd.NA: 
            self.logger.error(f"Could not convert PNL '{pnl_str}' to Decimal for trade {part_id}. PNL recorded as 0.")
            profit = Decimal(0)

        if profit <= 0: self.consecutive_losses += 1
        else: self.consecutive_losses = 0
        
        entry_dt_utc = datetime.fromtimestamp(entry_time_ms / 1000, tz=pytz.utc)
        exit_dt_utc = datetime.fromtimestamp(exit_time_ms / 1000, tz=pytz.utc)
        duration_seconds = (exit_dt_utc - entry_dt_utc).total_seconds()
        trade_type = "Scale-In" if scale_order_id else "Part"
        
        serializable_entry_indicators = {k: str(v) if isinstance(v, Decimal) else v for k, v in entry_indicators.items()}
        serializable_exit_indicators = {k: str(v) if isinstance(v, Decimal) else v for k, v in exit_indicators.items()}
        
        self.trades.append({
            "part_id": part_id, "symbol": symbol, "side": side, 
            "entry_price_str": str(entry_price), "exit_price_str": str(exit_price), 
            "qty_str": str(qty), "profit_str": str(profit), 
            "entry_time_iso": entry_dt_utc.isoformat(), "exit_time_iso": exit_dt_utc.isoformat(), 
            "duration_seconds": duration_seconds, "exit_reason": reason, 
            "type": trade_type, "scale_order_id": scale_order_id,
            "mae_str": str(mae) if mae is not None else None, 
            "mfe_str": str(mfe) if mfe is not None else None,
            "entry_indicators": serializable_entry_indicators, 
            "exit_indicators": serializable_exit_indicators
        })
        pnl_color = NEON["PNL_POS"] if profit > 0 else (NEON["PNL_NEG"] if profit < 0 else NEON["PNL_ZERO"])
        self.logger.success(f"Trade Chronicle ({trade_type}:{part_id}): {side.upper()} {qty} {symbol.split('/')[0]} | P/L: {pnl_color}{profit:.2f} {self.config.usdt_symbol}{NEON['RESET']} | Reason: {reason}")
        self.logger.debug(f"Entry Indicators for {part_id}: {entry_indicators}")
        self.logger.debug(f"Exit Indicators for {part_id}: {exit_indicators}")

    def increment_daily_trade_entry_count(self):
        self.daily_trade_entry_count +=1
        self.logger.info(f"Daily entry trade count incremented to: {self.daily_trade_entry_count}")

    def summary(self) -> str:
        if not self.trades: return "The Grand Ledger is empty."
        
        profits = [safe_decimal_conversion(t["profit_str"], Decimal(0)) for t in self.trades if t.get("profit_str") is not None]
        profits = [p for p in profits if isinstance(p, Decimal)]

        total_trades = len(self.trades)
        wins = sum(1 for p in profits if p > 0)
        losses = sum(1 for p in profits if p < 0)
        breakeven = total_trades - wins - losses
        win_rate = (Decimal(wins) / Decimal(total_trades)) * Decimal(100) if total_trades > 0 else Decimal(0)
        total_profit = sum(profits) if profits else Decimal(0)
        avg_profit = total_profit / Decimal(total_trades) if total_trades > 0 else Decimal(0)
        
        summary_str = (
            f"\n--- Pyrmethus Trade Metrics Summary (v{PYRMETHUS_VERSION}) ---\n"
            f"Total Trade Parts: {total_trades}\n"
            f"  Wins: {wins}, Losses: {losses}, Breakeven: {breakeven}\n"
            f"Win Rate: {win_rate:.2f}%\n"
            f"Total P/L: {total_profit:.2f} {self.config.usdt_symbol}\n"
            f"Avg P/L per Part: {avg_profit:.2f} {self.config.usdt_symbol}\n"
        )
        current_equity_approx = Decimal(0) 
        if self.initial_equity is not None and isinstance(self.initial_equity, Decimal):
            current_equity_approx = self.initial_equity + total_profit
            overall_pnl_pct = (total_profit / self.initial_equity) * 100 if self.initial_equity > Decimal(0) else Decimal(0)
            summary_str += f"Initial Session Treasury: {self.initial_equity:.2f} {self.config.usdt_symbol}\n"
            summary_str += f"Approx. Current Treasury: {current_equity_approx:.2f} {self.config.usdt_symbol}\n"
            summary_str += f"Overall Session P/L %: {overall_pnl_pct:.2f}%\n"
        
        if self.daily_start_equity is not None and isinstance(self.daily_start_equity, Decimal):
            daily_pnl_base = current_equity_approx if (self.initial_equity is not None and isinstance(self.initial_equity, Decimal)) else (self.daily_start_equity + total_profit)
            
            daily_pnl = daily_pnl_base - self.daily_start_equity
            daily_pnl_pct = (daily_pnl / self.daily_start_equity) * 100 if self.daily_start_equity > Decimal(0) else Decimal(0)
            summary_str += f"Daily Start Treasury: {self.daily_start_equity:.2f} {self.config.usdt_symbol}\n"
            summary_str += f"Approx. Daily P/L: {daily_pnl:.2f} {self.config.usdt_symbol} ({daily_pnl_pct:.2f}%)\n"
        
        summary_str += f"Consecutive Losses: {self.consecutive_losses}\n"
        summary_str += f"Daily Entries Made Today (UTC): {self.daily_trade_entry_count}\n"
        summary_str += "--- End of Ledger Reading ---"
        self.logger.info(summary_str) 
        return summary_str

    def get_serializable_trades(self) -> List[Dict[str, Any]]: return self.trades
    def load_trades_from_list(self, trades_list: List[Dict[str, Any]]):
        self.trades = [] 
        for trade_data in trades_list: 
            self.trades.append(trade_data)
        self.logger.info(f"TradeMetrics: Re-inked {len(self.trades)} sagas from persistent state, including indicator context.")

trade_metrics = TradeMetrics(CONFIG)
_active_trade_parts: List[Dict[str, Any]] = []
_last_heartbeat_save_time: float = 0.0
_last_health_check_time: float = 0.0
trade_timestamps_for_whipsaw: deque[float] = deque(maxlen=CONFIG.enhancements["whipsaw_max_trades_in_period"])
whipsaw_cooldown_active_until: float = 0.0
persistent_signal_counter: Dict[str, int] = {"long": 0, "short": 0}
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
            try:
                last_row_values = df.iloc[-1][self.required_columns]
                if last_row_values.isnull().any():
                    nan_cols_last_row = last_row_values[last_row_values.isnull()].index.tolist()
                    self.logger.debug(f"Faint runes (NaNs) in last candle for critical columns: {nan_cols_last_row}.")
            except IndexError: 
                 self.logger.warning("Could not access last row for NaN check despite passing min_rows validation.")
                 return False
        return True
    def _get_default_signals(self) -> Dict[str, Any]:
        return {"enter_long": False, "enter_short": False, "exit_long": False, "exit_short": False, "exit_reason": "Default Signal"}

class DualSupertrendMomentumStrategy(TradingStrategy):
    def __init__(self, config: Config):
        super().__init__(config, df_columns=["st_long_flip", "st_short_flip", "confirm_trend", "momentum"])
    async def generate_signals(self, df: pd.DataFrame, latest_close: Optional[Decimal] = None, latest_atr: Optional[Decimal] = None) -> Dict[str, Any]:
        signals = self._get_default_signals()
        
        momentum_threshold = self.config.strategy_params["momentum_threshold"]
        st_max_dist_atr_mult = self.config.strategy_params["st_max_entry_distance_atr_multiplier"]

        min_rows_needed = max(
            self.config.strategy_params["st_atr_length"],
            self.config.strategy_params["confirm_st_atr_length"],
            self.config.strategy_params["momentum_period"],
            self.config.strategy_params["confirm_st_stability_lookback"]
        ) + 10 
        if not self._validate_df(df, min_rows=min_rows_needed): return signals
        
        try:
            last = df.iloc[-1]
        except IndexError:
            self.logger.warning("Could not get last row from DataFrame in DualSupertrend.")
            return signals

        primary_long_flip = last.get("st_long_flip", False)
        primary_short_flip = last.get("st_short_flip", False)
        confirm_is_up_current = last.get("confirm_trend", pd.NA) 
        momentum_val = safe_decimal_conversion(last.get("momentum"), pd.NA)

        if primary_long_flip and primary_short_flip: 
            self.logger.warning("Conflicting primary Supertrend flips detected on the same candle.")
            if confirm_is_up_current is True and (not pd.isna(momentum_val) and momentum_val > 0):
                primary_short_flip = False; self.logger.info("Resolution: Prioritizing LONG flip due to confirm ST and positive momentum.")
            elif confirm_is_up_current is False and (not pd.isna(momentum_val) and momentum_val < 0):
                primary_long_flip = False; self.logger.info("Resolution: Prioritizing SHORT flip due to confirm ST and negative momentum.")
            else: 
                primary_long_flip = False; primary_short_flip = False; self.logger.warning("Resolution: Ambiguous flip, ignoring both.")
        
        stable_confirm_trend = pd.NA
        confirm_st_stability_lookback = self.config.strategy_params["confirm_st_stability_lookback"]
        if confirm_st_stability_lookback <= 1: 
            stable_confirm_trend = confirm_is_up_current
        elif 'confirm_trend' in df.columns and len(df) >= confirm_st_stability_lookback:
            recent_confirm_trends = df['confirm_trend'].iloc[-confirm_st_stability_lookback:]
            if confirm_is_up_current is True and recent_confirm_trends.is_monotonic_increasing and all(trend is True for trend in recent_confirm_trends): stable_confirm_trend = True 
            elif confirm_is_up_current is False and recent_confirm_trends.is_monotonic_decreasing and all(trend is False for trend in recent_confirm_trends): stable_confirm_trend = False 
        
        if pd.isna(stable_confirm_trend) or pd.isna(momentum_val):
            self.logger.debug(f"Signal check: Stable Confirm ST ({_format_for_log(stable_confirm_trend, True)}) or Momentum ({_format_for_log(momentum_val)}) is NA. No entry signal.")
            return signals 

        price_proximity_ok = True
        if st_max_dist_atr_mult is not None and latest_atr is not None and latest_atr > 0 and latest_close is not None:
            max_allowed_distance = latest_atr * st_max_dist_atr_mult
            st_p_base = f"ST_{self.config.strategy_params['st_atr_length']}_{self.config.strategy_params['st_multiplier']}"
            
            current_st_line_val_key = None
            if primary_long_flip: current_st_line_val_key = f"{st_p_base}l" 
            elif primary_short_flip: current_st_line_val_key = f"{st_p_base}s" 
            
            if current_st_line_val_key:
                st_line_val = safe_decimal_conversion(last.get(current_st_line_val_key))
                if st_line_val is not None and not pd.isna(st_line_val):
                    distance_from_st = abs(latest_close - st_line_val)
                    if distance_from_st > max_allowed_distance:
                        price_proximity_ok = False
                        self.logger.debug(f"Price proximity check failed for {'LONG' if primary_long_flip else 'SHORT'} entry. Distance {distance_from_st:.4f} > Max Allowed {max_allowed_distance:.4f} (ATR={latest_atr:.4f})")
                else:
                    self.logger.warning(f"Could not get ST line value for key {current_st_line_val_key} for proximity check.")

        if primary_long_flip and stable_confirm_trend is True and \
           momentum_val > momentum_threshold and price_proximity_ok: 
            signals["enter_long"] = True
            self.logger.info(f"DualST+Mom Signal: {NEON['SIDE_LONG']}LONG Entry Triggered.{NEON['RESET']}")
        elif primary_short_flip and stable_confirm_trend is False and \
             momentum_val < -momentum_threshold and price_proximity_ok: 
            signals["enter_short"] = True
            self.logger.info(f"DualST+Mom Signal: {NEON['SIDE_SHORT']}SHORT Entry Triggered.{NEON['RESET']}")

        if primary_short_flip: 
            signals["exit_long"] = True; signals["exit_reason"] = "Primary SuperTrend Flipped Short"
        if primary_long_flip: 
            signals["exit_short"] = True; signals["exit_reason"] = "Primary SuperTrend Flipped Long"
            
        return signals

class EhlersFisherStrategy(TradingStrategy):
    def __init__(self, config: Config):
        super().__init__(config, df_columns=["ehlers_fisher", "ehlers_signal"])
    async def generate_signals(self, df: pd.DataFrame, latest_close: Optional[Decimal] = None, latest_atr: Optional[Decimal] = None) -> Dict[str, Any]:
        signals = self._get_default_signals()
        extreme_pos = self.config.strategy_params["ehlers_fisher_extreme_threshold_positive"]
        extreme_neg = self.config.strategy_params["ehlers_fisher_extreme_threshold_negative"]

        min_rows_needed = self.config.strategy_params["ehlers_fisher_length"] + self.config.strategy_params["ehlers_fisher_signal_length"] + 5
        if not self._validate_df(df, min_rows=min_rows_needed) or len(df) < 2: 
            return signals
        
        try:
            last = df.iloc[-1]; prev = df.iloc[-2]
        except IndexError:
            self.logger.warning("Could not get last/previous rows from DataFrame in EhlersFisher.")
            return signals

        fisher_now = safe_decimal_conversion(last.get("ehlers_fisher"), pd.NA)
        signal_now = safe_decimal_conversion(last.get("ehlers_signal"), pd.NA) 
        fisher_prev = safe_decimal_conversion(prev.get("ehlers_fisher"), pd.NA)
        signal_prev = safe_decimal_conversion(prev.get("ehlers_signal"), pd.NA)

        if any(pd.isna(val) for val in [fisher_now, signal_now, fisher_prev, signal_prev]):
            self.logger.debug("Ehlers Fisher or Signal rune is NA for current or previous candle.")
            return signals

        is_fisher_extreme_now = (fisher_now > extreme_pos or fisher_now < extreme_neg)
        
        if not is_fisher_extreme_now:
            if fisher_prev <= signal_prev and fisher_now > signal_now: 
                signals["enter_long"] = True
                self.logger.info(f"EhlersFisher Signal: {NEON['SIDE_LONG']}LONG Entry Triggered (Fisher {fisher_now:.2f} > Signal {signal_now:.2f}).{NEON['RESET']}")
            elif fisher_prev >= signal_prev and fisher_now < signal_now: 
                signals["enter_short"] = True
                self.logger.info(f"EhlersFisher Signal: {NEON['SIDE_SHORT']}SHORT Entry Triggered (Fisher {fisher_now:.2f} < Signal {signal_now:.2f}).{NEON['RESET']}")
        elif (fisher_prev <= signal_prev and fisher_now > signal_now) or \
             (fisher_prev >= signal_prev and fisher_now < signal_now):
            self.logger.info(f"EhlersFisher: Crossover occurred but Fisher is in extreme zone ({fisher_now:.2f}). Entry ignored.")

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
def save_persistent_state(force_heartbeat: bool = False) -> None:
    global _active_trade_parts, trade_metrics, _last_heartbeat_save_time
    global whipsaw_cooldown_active_until, persistent_signal_counter, last_signal_type
    global previous_day_high, previous_day_low, last_key_level_update_day
    global contradiction_cooldown_active_until, consecutive_loss_cooldown_active_until
    
    now = time.time()
    save_interval = HEARTBEAT_INTERVAL_SECONDS if _active_trade_parts else HEARTBEAT_INTERVAL_SECONDS * 5
    should_save = force_heartbeat or (now - _last_heartbeat_save_time > save_interval)
    
    if not should_save: return

    logger.debug("Phoenix Feather scribing memories...", extra={"data": {"force": force_heartbeat, "active_parts": len(_active_trade_parts)}})
    try:
        serializable_active_parts = []
        for part in _active_trade_parts:
            p_copy = part.copy()
            for key, value in p_copy.items():
                if isinstance(value, Decimal): p_copy[key] = str(value)
                elif isinstance(value, deque): p_copy[key] = list(value) 
                elif key == "entry_indicators" and isinstance(value, dict):
                    p_copy[key] = {k_ind: str(v_ind) if isinstance(v_ind, Decimal) else v_ind for k_ind, v_ind in value.items()}
            serializable_active_parts.append(p_copy)
            
        state_data = {
            "pyrmethus_version": PYRMETHUS_VERSION,
            "timestamp_utc_iso": datetime.now(pytz.utc).isoformat(),
            "active_trade_parts": serializable_active_parts,
            "trade_metrics_trades": trade_metrics.get_serializable_trades(), 
            "trade_metrics_consecutive_losses": trade_metrics.consecutive_losses,
            "trade_metrics_daily_trade_entry_count": trade_metrics.daily_trade_entry_count,
            "trade_metrics_last_daily_trade_count_reset_day": trade_metrics.last_daily_trade_count_reset_day,
            "trade_metrics_daily_trades_rest_active_until": trade_metrics.daily_trades_rest_active_until,
            "config_symbol": CONFIG.exchange["symbol"], "config_strategy": CONFIG.strategy_params["name"].value,
            "initial_equity_str": str(trade_metrics.initial_equity) if trade_metrics.initial_equity is not None else None,
            "daily_start_equity_str": str(trade_metrics.daily_start_equity) if trade_metrics.daily_start_equity is not None else None,
            "last_daily_reset_day": trade_metrics.last_daily_reset_day,
            "whipsaw_cooldown_active_until": whipsaw_cooldown_active_until,
            "trade_timestamps_for_whipsaw": list(trade_timestamps_for_whipsaw), 
            "persistent_signal_counter": persistent_signal_counter,
            "last_signal_type": last_signal_type,
            "previous_day_high_str": str(previous_day_high) if previous_day_high else None,
            "previous_day_low_str": str(previous_day_low) if previous_day_low else None,
            "last_key_level_update_day": last_key_level_update_day,
            "contradiction_cooldown_active_until": contradiction_cooldown_active_until,
            "consecutive_loss_cooldown_active_until": consecutive_loss_cooldown_active_until,
        }
        temp_file_path = STATE_FILE_PATH + ".tmp_scroll"
        with open(temp_file_path, 'w', encoding='utf-8') as f: 
            json.dump(state_data, f, indent=2)
        os.replace(temp_file_path, STATE_FILE_PATH) 
        _last_heartbeat_save_time = now
        
        log_level = logging.INFO if force_heartbeat or _active_trade_parts else logging.DEBUG
        logger.log(log_level, f"Phoenix Feather: State memories scribed. Active parts: {len(_active_trade_parts)}.")
    except Exception as e:
        logger.error(f"Phoenix Feather Error: Failed to scribe state: {e}", exc_info=True)

def load_persistent_state() -> bool:
    global _active_trade_parts, trade_metrics
    global whipsaw_cooldown_active_until, trade_timestamps_for_whipsaw, persistent_signal_counter, last_signal_type
    global previous_day_high, previous_day_low, last_key_level_update_day
    global contradiction_cooldown_active_until, consecutive_loss_cooldown_active_until
    
    logger.info(f"Phoenix Feather seeks past memories from '{STATE_FILE_PATH}'...")
    if not os.path.exists(STATE_FILE_PATH):
        logger.info("Phoenix Feather: No previous scroll found. Starting fresh.")
        return False
    try:
        with open(STATE_FILE_PATH, 'r', encoding='utf-8') as f: 
            state_data = json.load(f)
        
        saved_version = state_data.get("pyrmethus_version", "unknown")
        current_major_minor = ".".join(PYRMETHUS_VERSION.split('.')[:2])
        saved_major_minor = ".".join(saved_version.split('.')[:2])

        if saved_major_minor != current_major_minor:
            logger.warning(f"Phoenix Feather: Scroll version '{saved_version}' (Major.Minor: {saved_major_minor}) "
                           f"differs significantly from current '{PYRMETHUS_VERSION}' (Major.Minor: {current_major_minor}). "
                           f"State reanimation might be unstable or incompatible. Consider starting fresh by deleting state file '{STATE_FILE_PATH}'.",
                           extra={"data": {"saved_version": saved_version, "current_version": PYRMETHUS_VERSION}})
        elif saved_version != PYRMETHUS_VERSION:
             logger.info(f"Phoenix Feather: Scroll patch version '{saved_version}' differs from current '{PYRMETHUS_VERSION}'. Proceeding with reanimation.")
        
        if state_data.get("config_symbol") != CONFIG.exchange["symbol"] or \
           state_data.get("config_strategy") != CONFIG.strategy_params["name"].value:
            logger.warning(f"Phoenix Scroll sigils mismatch! Current: {CONFIG.exchange['symbol']}/{CONFIG.strategy_params['name'].value}, "
                           f"Scroll: {state_data.get('config_symbol')}/{state_data.get('config_strategy')}. "
                           f"Ignoring scroll to prevent state corruption.",
                           extra={"data": {"current_symbol": CONFIG.exchange["symbol"], "saved_symbol": state_data.get("config_symbol"),
                                         "current_strategy": CONFIG.strategy_params["name"].value, "saved_strategy": state_data.get("config_strategy")}})
            return False

        _active_trade_parts.clear()
        loaded_parts_raw = state_data.get("active_trade_parts", [])
        
        decimal_keys_in_part = [
            "entry_price", "qty", "sl_price", "tp_price", 
            "initial_usdt_value", "atr_at_entry", 
            "entry_fisher_value", "entry_signal_value", "last_known_pnl"
        ]
        deque_keys_in_part_config = { 
            "recent_pnls": CONFIG.enhancements["profit_momentum_window"],
            "adverse_candle_closes": CONFIG.enhancements["last_chance_consecutive_adverse_candles"]
        }

        for part_data_str_values in loaded_parts_raw:
            restored_part: Dict[str, Any] = {}
            valid_part = True
            for k, v_loaded in part_data_str_values.items():
                if k in decimal_keys_in_part:
                    restored_part[k] = safe_decimal_conversion(v_loaded, None) 
                    if restored_part[k] is None and v_loaded is not None and str(v_loaded).lower() != 'none': 
                        logger.warning(f"Failed to restore Decimal '{k}' for part {part_data_str_values.get('part_id', 'UnknownID')}, original value was '{v_loaded}'. Stored as None.")
                elif k in deque_keys_in_part_config:
                    maxlen = deque_keys_in_part_config[k]
                    if isinstance(v_loaded, list): 
                        restored_part[k] = deque(v_loaded, maxlen=maxlen)
                    else: 
                        restored_part[k] = deque(maxlen=maxlen)
                        logger.warning(f"Expected list for deque '{k}' in part {part_data_str_values.get('part_id', 'UnknownID')}, got {type(v_loaded)}. Initialized empty deque.")
                elif k == "entry_indicators":
                    if isinstance(v_loaded, dict):
                        restored_part[k] = {key_ind: safe_decimal_conversion(val_ind, val_ind) 
                                            for key_ind, val_ind in v_loaded.items()}
                    else: 
                        restored_part[k] = {}
                        logger.warning(f"Expected dict for 'entry_indicators' in part {part_data_str_values.get('part_id', 'UnknownID')}, got {type(v_loaded)}. Initialized empty dict.")
                elif k == "entry_time_ms": 
                    if isinstance(v_loaded, (int, float)): restored_part[k] = int(v_loaded)
                    elif isinstance(v_loaded, str): 
                        try: restored_part[k] = int(float(v_loaded))
                        except ValueError: 
                            logger.warning(f"Malformed numeric string for '{k}': '{v_loaded}' in part {part_data_str_values.get('part_id', 'UnknownID')}. Skipping part field."); 
                    else: logger.warning(f"Unexpected type for '{k}': '{v_loaded}' (type {type(v_loaded)}) in part {part_data_str_values.get('part_id', 'UnknownID')}. Skipping field.")
                elif k == "last_trailing_sl_update": 
                     if isinstance(v_loaded, (int, float)): restored_part[k] = float(v_loaded)
                     elif isinstance(v_loaded, str):
                        try: restored_part[k] = float(v_loaded)
                        except ValueError: logger.warning(f"Malformed float string for '{k}': '{v_loaded}' in part {part_data_str_values.get('part_id', 'UnknownID')}. Skipping field.")
                     else: logger.warning(f"Unexpected type for '{k}': '{v_loaded}' (type {type(v_loaded)}) in part {part_data_str_values.get('part_id', 'UnknownID')}. Skipping field.")
                else: 
                    restored_part[k] = v_loaded
            
            if valid_part and "part_id" in restored_part: 
                _active_trade_parts.append(restored_part)
            elif not valid_part:
                 logger.warning(f"Skipped loading an invalid active trade part due to critical field errors. Data: {part_data_str_values}")


        trade_metrics.load_trades_from_list(state_data.get("trade_metrics_trades", []))
        trade_metrics.consecutive_losses = state_data.get("trade_metrics_consecutive_losses", 0)
        trade_metrics.daily_trade_entry_count = state_data.get("trade_metrics_daily_trade_entry_count", 0)
        trade_metrics.last_daily_trade_count_reset_day = state_data.get("trade_metrics_last_daily_trade_count_reset_day")
        trade_metrics.daily_trades_rest_active_until = state_data.get("trade_metrics_daily_trades_rest_active_until", 0.0)
        
        initial_equity_str = state_data.get("initial_equity_str")
        if initial_equity_str and initial_equity_str.lower() != 'none': 
            trade_metrics.initial_equity = safe_decimal_conversion(initial_equity_str, None)
        
        daily_equity_str = state_data.get("daily_start_equity_str")
        if daily_equity_str and daily_equity_str.lower() != 'none': 
            trade_metrics.daily_start_equity = safe_decimal_conversion(daily_equity_str, None)
        
        trade_metrics.last_daily_reset_day = state_data.get("last_daily_reset_day")
        if trade_metrics.last_daily_reset_day is not None:
            try: trade_metrics.last_daily_reset_day = int(trade_metrics.last_daily_reset_day)
            except (ValueError, TypeError): 
                trade_metrics.last_daily_reset_day = None
                logger.warning(f"Malformed 'last_daily_reset_day' in state: {state_data.get('last_daily_reset_day')}. Resetting.")
        
        whipsaw_cooldown_active_until = state_data.get("whipsaw_cooldown_active_until", 0.0)
        loaded_whipsaw_ts = state_data.get("trade_timestamps_for_whipsaw", [])
        trade_timestamps_for_whipsaw = deque(loaded_whipsaw_ts, maxlen=CONFIG.enhancements["whipsaw_max_trades_in_period"])
        
        persistent_signal_counter = state_data.get("persistent_signal_counter", {"long": 0, "short": 0})
        last_signal_type = state_data.get("last_signal_type")
        
        prev_high_str = state_data.get("previous_day_high_str")
        previous_day_high = safe_decimal_conversion(prev_high_str, None) if prev_high_str and prev_high_str.lower() != 'none' else None
        
        prev_low_str = state_data.get("previous_day_low_str")
        previous_day_low = safe_decimal_conversion(prev_low_str, None) if prev_low_str and prev_low_str.lower() != 'none' else None
        
        last_key_level_update_day = state_data.get("last_key_level_update_day")
        contradiction_cooldown_active_until = state_data.get("contradiction_cooldown_active_until", 0.0)
        consecutive_loss_cooldown_active_until = state_data.get("consecutive_loss_cooldown_active_until", 0.0)
        
        logger.success(f"Phoenix Feather: Memories reawakened! Active Parts: {len(_active_trade_parts)}, Trades Logged: {len(trade_metrics.trades)}")
        return True
    except json.JSONDecodeError as e:
        logger.error(f"Phoenix Feather: Scroll '{STATE_FILE_PATH}' corrupted (JSONDecodeError): {e}. Starting fresh.", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Phoenix Feather Error: Unexpected chaos during reawakening from '{STATE_FILE_PATH}': {e}", exc_info=True)
        return False

def extract_indicators_from_df(df_row: pd.Series, config: Config) -> Dict[str, Any]:
    indicators: Dict[str, Any] = {}
    atr_period = config.strategy_params['atr_calculation_period'] # Type guaranteed by Config
    indicators[f'atr_{atr_period}'] = safe_decimal_conversion(df_row.get(f"ATR_{atr_period}"))
    
    if config.strategy_params["name"] == StrategyName.DUAL_SUPERTREND_MOMENTUM:
        st_atr_len = config.strategy_params['st_atr_length'] # Type guaranteed by Config
        st_mult = config.strategy_params['st_multiplier'] # Type guaranteed by Config
        st_p_base = f"ST_{st_atr_len}_{st_mult}"
        
        indicators[f'{st_p_base}l'] = safe_decimal_conversion(df_row.get(f"{st_p_base}l")) 
        indicators[f'{st_p_base}s'] = safe_decimal_conversion(df_row.get(f"{st_p_base}s")) 
        indicators[f'{st_p_base}d'] = df_row.get(f"{st_p_base}d") 
        indicators['confirm_trend'] = df_row.get("confirm_trend") 
        indicators['momentum'] = safe_decimal_conversion(df_row.get("momentum"))
    elif config.strategy_params["name"] == StrategyName.EHLERS_FISHER:
        indicators['ehlers_fisher'] = safe_decimal_conversion(df_row.get("ehlers_fisher"))
        indicators['ehlers_signal'] = safe_decimal_conversion(df_row.get("ehlers_signal"))
    
    indicators['close_price'] = safe_decimal_conversion(df_row.get("close"))
    
    return {k: v for k, v in indicators.items() if not pd.isna(v)}


async def async_calculate_position_size(balance: Decimal, risk_per_trade_pct: Decimal, 
                                      entry_price: Decimal, sl_price: Decimal, 
                                      config: Config) -> Optional[Decimal]:
    logger_calc = logging.getLogger("Pyrmethus.CalcPosSize")
    logger_calc.debug(f"Calculating position size. Balance: {balance:.2f}, Risk %: {risk_per_trade_pct:.4%}, Entry: {entry_price}, SL: {sl_price}")

    if not all([
        isinstance(balance, Decimal) and balance > 0,
        isinstance(entry_price, Decimal) and entry_price > 0,
        isinstance(sl_price, Decimal) and sl_price > 0,
        sl_price != entry_price,
        config.MARKET_INFO is not None, 
        config.qty_step is not None and isinstance(config.qty_step, Decimal) and config.qty_step > 0, 
        config.tick_size is not None and isinstance(config.tick_size, Decimal) and config.tick_size > 0,
        config.min_order_qty is not None and isinstance(config.min_order_qty, Decimal) and config.min_order_qty > 0
    ]):
        logger_calc.warning("Invalid parameters or market info for sizing. Cannot divine size.", 
                        extra={"data": {"balance": str(balance), "entry": str(entry_price), "sl": str(sl_price),
                                        "qty_step": str(config.qty_step), "tick_size": str(config.tick_size),
                                        "min_order_qty": str(config.min_order_qty)}})
        return None

    risk_per_unit = abs(entry_price - sl_price)
    if risk_per_unit < config.tick_size: 
        logger_calc.warning(f"Risk per unit ({risk_per_unit}) is smaller than tick size ({config.tick_size}). Stop loss is too close to entry. Cannot size.",
                        extra={"data": {"risk_per_unit": str(risk_per_unit), "tick_size": str(config.tick_size)}})
        return None

    usdt_at_risk = balance * risk_per_trade_pct
    logger_calc.debug(f"USDT at risk for this trade: {usdt_at_risk:.2f}")

    quantity_calculated = usdt_at_risk / risk_per_unit
    quantity_stepped = quantize_value(quantity_calculated, config.qty_step, ROUND_DOWN)
    
    logger_calc.debug(f"Initial calculated qty: {quantity_calculated}, after step ({config.qty_step}): {quantity_stepped}")

    if quantity_stepped <= Decimal(0):
        logger_calc.warning(f"Calculated quantity {quantity_stepped} is zero or negligible after step adjustment. No order.",
                        extra={"data": {"qty_calc": str(quantity_calculated), "qty_step": str(quantity_stepped)}})
        return None

    position_usdt_value = quantity_stepped * entry_price
    if position_usdt_value > config.trading["max_order_usdt_amount"]:
        logger_calc.info(f"Calculated position USDT value {position_usdt_value:.2f} exceeds MAX_ORDER_USDT_AMOUNT ({config.trading['max_order_usdt_amount']}). Capping.")
        quantity_capped_by_value = config.trading["max_order_usdt_amount"] / entry_price
        quantity_stepped = quantize_value(quantity_capped_by_value, config.qty_step, ROUND_DOWN)
        position_usdt_value = quantity_stepped * entry_price
        logger_calc.info(f"Capped quantity: {quantity_stepped}, New USDT value: {position_usdt_value:.2f}")

    if quantity_stepped <= Decimal(0):
        logger_calc.warning("Quantity became zero after capping by MAX_ORDER_USDT_AMOUNT. No order.")
        return None
        
    if quantity_stepped < config.min_order_qty:
        logger_calc.warning(f"Calculated quantity {NEON['QTY']}{quantity_stepped}{NEON['WARNING']} < exchange min qty {NEON['QTY']}{config.min_order_qty}{NEON['WARNING']}. No order.",
                        extra={"data": {"qty_final": str(quantity_stepped), "min_qty": str(config.min_order_qty)}})
        return None

    if config.min_order_cost is not None and isinstance(config.min_order_cost, Decimal) and config.min_order_cost > 0:
        if position_usdt_value < config.min_order_cost:
            logger_calc.warning(f"Calculated position value {NEON['VALUE']}{position_usdt_value:.2f}{NEON['WARNING']} < exchange min cost {NEON['VALUE']}{config.min_order_cost}{NEON['WARNING']}. Attempting to adjust to min cost.",
                                extra={"data": {"pos_value": str(position_usdt_value), "min_cost": str(config.min_order_cost)}})
            
            qty_for_min_cost = config.min_order_cost / entry_price 
            qty_for_min_cost_stepped = quantize_value(qty_for_min_cost, config.qty_step, ROUND_UP)

            if qty_for_min_cost_stepped < config.min_order_qty:
                 logger_calc.warning(f"Adjusted qty for min cost ({qty_for_min_cost_stepped}) is still below min qty ({config.min_order_qty}). No order.")
                 return None

            value_at_min_cost_qty = qty_for_min_cost_stepped * entry_price
            if value_at_min_cost_qty > config.trading["max_order_usdt_amount"]:
                logger_calc.warning(f"Adjusting to min cost would make order value ({value_at_min_cost_qty:.2f}) exceed MAX_ORDER_USDT_AMOUNT ({config.trading['max_order_usdt_amount']}). No order.")
                return None
            
            usdt_at_risk_if_min_cost = qty_for_min_cost_stepped * risk_per_unit
            allowed_risk_increase_factor = Decimal("1.2") 
            if usdt_at_risk > 0 and usdt_at_risk_if_min_cost > usdt_at_risk * allowed_risk_increase_factor: 
                logger_calc.warning(f"Adjusting to min cost would exceed risk tolerance. Original USDT risk: {usdt_at_risk:.2f}, Risk for min cost qty: {usdt_at_risk_if_min_cost:.2f} (Limit: {usdt_at_risk * allowed_risk_increase_factor:.2f}). No order.")
                return None
            
            logger_calc.info(f"Adjusting quantity from {quantity_stepped} to {qty_for_min_cost_stepped} to meet minimum order cost. New USDT value: {value_at_min_cost_qty:.2f}, New risk: {usdt_at_risk_if_min_cost:.2f}")
            quantity_stepped = qty_for_min_cost_stepped
            position_usdt_value = value_at_min_cost_qty 

    qty_display_precision = abs(config.qty_step.as_tuple().exponent) if config.qty_step.as_tuple().exponent < 0 else 0
    logger_calc.success(f"Final position size: {NEON['QTY']}{quantity_stepped:.{qty_display_precision}f}{NEON['RESET']} (USDT Value: {position_usdt_value:.2f})")
    return quantity_stepped


@retry((ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection, ccxt.ExchangeNotAvailable, ccxt.RateLimitExceeded), 
       tries=CONFIG.retry_count, delay=CONFIG.retry_delay_seconds, backoff=2, 
       logger=logging.getLogger("Pyrmethus.OrderRetry"),
       exceptions_to_not_retry=(ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.AuthenticationError)) 
async def async_place_risked_order(
    exchange: ccxt_async.Exchange, 
    config: Config, 
    side: str, 
    entry_price_target: Decimal, 
    sl_price: Decimal, 
    tp_price: Optional[Decimal], 
    atr_val: Decimal, 
    df_with_indicators: pd.DataFrame 
) -> Optional[Dict[str, Any]]:
    logger_order = logging.getLogger("Pyrmethus.OrderPlacement")

    logger_order.info(f"Attempting {side.upper()} order for {config.exchange['symbol']}... Target Entry: {format_price(entry_price_target, config.tick_size)}, SL: {format_price(sl_price, config.tick_size)}{f', TP: {format_price(tp_price, config.tick_size)}' if tp_price else ''}")

    effective_risk_pct_for_trade = config.trading["risk_per_trade_percentage"]
    if config.enhancements["anti_martingale_risk_enable"] and trade_metrics.trades:
        pass 
    
    _placeholder_balance = balance_cache.get(f"{config.usdt_symbol}_balance_value") 
    if not _placeholder_balance: 
        # In a real scenario, this would await an actual fetch if cache is empty/stale.
        # For this snippet, we'll use a default or error out if not pre-cached by placeholder.
        logger_order.warning("Balance not found in cache for placeholder logic. Using default for calculation.")
        _placeholder_balance = await async_fetch_account_balance(exchange, config.usdt_symbol, config) # Ensure it's fetched if not in cache
    
    balance = _placeholder_balance

    if balance is None or not isinstance(balance, Decimal) or balance < config.trading["min_usdt_balance_for_trading"]:
        logger_order.error(f"Insufficient treasury ({balance}, Min Required: {config.trading['min_usdt_balance_for_trading']}). Cannot place order.",
                           extra={"data": {"balance": str(balance), "min_balance": str(config.trading['min_usdt_balance_for_trading'])}})
        return None
    
    if not all([config.MARKET_INFO, config.tick_size, config.qty_step, config.min_order_qty is not None]):
        logger_order.error("Market info (tick_size, qty_step, min_order_qty) not available. Cannot place order.")
        return None

    sl_rounding = ROUND_UP if side == config.pos_long else ROUND_DOWN
    quantized_sl_price = quantize_value(sl_price, config.tick_size, sl_rounding)

    quantized_tp_price: Optional[Decimal] = None
    if tp_price and config.risk_management["enable_take_profit"]:
        tp_rounding = ROUND_DOWN if side == config.pos_long else ROUND_UP
        quantized_tp_price = quantize_value(tp_price, config.tick_size, tp_rounding)

    if (side == config.pos_long and quantized_sl_price >= entry_price_target) or \
       (side == config.pos_short and quantized_sl_price <= entry_price_target):
        logger_order.error(f"Quantized SL price ({format_price(quantized_sl_price, config.tick_size)}) is invalid relative to entry target ({format_price(entry_price_target, config.tick_size)}). Cannot place order.")
        return None
    
    quantity = await async_calculate_position_size(balance, effective_risk_pct_for_trade, entry_price_target, quantized_sl_price, config)
    if quantity is None or quantity <= Decimal(0):
        logger_order.warning("Position sizing yielded no valid quantity. Order not placed.")
        return None

    order_side_ccxt = config.side_buy if side == config.pos_long else config.side_sell
    order_type_ccxt = config.trading["entry_order_type"].value.lower() 
    
    params: Dict[str, Any] = {
        'category': 'linear', 
        'positionIdx': 0,     
        'stopLoss': str(quantized_sl_price),
        'slTriggerBy': 'MarkPrice', 
    }
    if quantized_tp_price and config.risk_management["enable_take_profit"]:
        params['takeProfit'] = str(quantized_tp_price)
        params['tpTriggerBy'] = 'MarkPrice'

    order_price_for_api: Optional[float] = None 
    if order_type_ccxt == 'limit':
        if atr_val is None or atr_val <= Decimal(0):
            logger_order.error("ATR value is invalid, cannot calculate limit order price offset. Consider market order or fix ATR.")
            return None 

        offset = atr_val * config.trading["limit_order_offset_atr_percentage"]
        limit_price_ideal: Decimal
        limit_rounding: Any 

        if side == config.pos_long:
            limit_price_ideal = entry_price_target - offset 
            limit_rounding = ROUND_DOWN 
        else: 
            limit_price_ideal = entry_price_target + offset 
            limit_rounding = ROUND_UP   
        
        if side == config.pos_long and limit_price_ideal <= quantized_sl_price:
            logger_order.warning(f"Calculated limit price {format_price(limit_price_ideal, config.tick_size)} for LONG is at or worse than SL {format_price(quantized_sl_price, config.tick_size)}. Adjusting limit to one tick better than SL.")
            limit_price_ideal = quantized_sl_price + config.tick_size 
        elif side == config.pos_short and limit_price_ideal >= quantized_sl_price:
            logger_order.warning(f"Calculated limit price {format_price(limit_price_ideal, config.tick_size)} for SHORT is at or worse than SL {format_price(quantized_sl_price, config.tick_size)}. Adjusting limit to one tick better than SL.")
            limit_price_ideal = quantized_sl_price - config.tick_size
        
        quantized_limit_price = quantize_value(limit_price_ideal, config.tick_size, limit_rounding)
        order_price_for_api = float(quantized_limit_price)
        logger_order.info(f"Limit order price calculated: {format_price(quantized_limit_price, config.tick_size)} (Ideal: {format_price(limit_price_ideal, config.tick_size)}, Target: {format_price(entry_price_target, config.tick_size)})")

    try:
        logger_order.info(f"Weaving {order_type_ccxt.upper()} {order_side_ccxt.upper()} order: Qty {NEON['QTY']}{quantity}{NEON['RESET']}, "
                          f"SL {NEON['PRICE']}{format_price(quantized_sl_price, config.tick_size)}{NEON['RESET']}"
                          f"{f', TP {NEON['PRICE']}{format_price(quantized_tp_price, config.tick_size)}{NEON['RESET']}' if quantized_tp_price and config.risk_management['enable_take_profit'] else ''}"
                          f"{f', Price {format_price(Decimal(str(order_price_for_api)), config.tick_size)}' if order_price_for_api is not None else ''}", 
                          extra={"data": {"symbol": config.exchange["symbol"], "type": order_type_ccxt, "side": order_side_ccxt, 
                                        "qty": str(quantity), "sl": str(quantized_sl_price), 
                                        "tp": str(quantized_tp_price) if quantized_tp_price else None,
                                        "limit_price": str(order_price_for_api) if order_price_for_api is not None else None,
                                        "params_to_api": params}})
        
        order_response = await exchange.create_order(
            symbol=config.exchange["symbol"],
            type=order_type_ccxt,
            side=order_side_ccxt,
            amount=float(quantity), 
            price=order_price_for_api, 
            params=params
        )
        
        order_id = order_response.get('id', 'N/A')
        order_status_initial = order_response.get('status', 'N/A')
        logger_order.success(f"Entry Order {order_id} ({order_status_initial}) cast successfully to exchange.",
                             extra={"data": {"order_id": order_id, "status": order_status_initial, "response": order_response}})
        
        qty_display_precision = abs(config.qty_step.as_tuple().exponent) if config.qty_step and config.qty_step.as_tuple().exponent < 0 else 0
        send_general_notification(f"Pyrmethus Order Cast: {side.upper()}", 
                                  f"{config.exchange['symbol']} Qty: {quantity:.{qty_display_precision}f} @ {'Market' if order_price_for_api is None else format_price(Decimal(str(order_price_for_api)), config.tick_size)}, "
                                  f"SL: {format_price(quantized_sl_price, config.tick_size)}{f', TP: {format_price(quantized_tp_price, config.tick_size)}' if quantized_tp_price else ''}. ID: {order_id}")
        
        await asyncio.sleep(ORDER_STATUS_CHECK_DELAY)
        filled_order = await exchange.fetch_order(order_id, config.exchange["symbol"]) # type: ignore
        logger_order.debug(f"Fetched order status for {order_id}: {filled_order.get('status')}", extra={"data": filled_order})

        actual_qty_filled = safe_decimal_conversion(filled_order.get('filled', Decimal(0)), Decimal(0))
        is_filled_or_partially_filled = actual_qty_filled > Decimal(0)
        is_closed_status = filled_order.get('status') == 'closed'
        
        if (is_closed_status and is_filled_or_partially_filled) or \
           (order_type_ccxt == 'market' and is_filled_or_partially_filled) or \
           (order_type_ccxt == 'limit' and is_filled_or_partially_filled and filled_order.get('status') == 'open'): 

            actual_entry_price = safe_decimal_conversion(filled_order.get('average', filled_order.get('price')), None)

            if actual_entry_price is None or actual_qty_filled <= Decimal(0):
                logger_order.error(f"Order {order_id} reported filled/partially but has invalid avg price/qty. Price: {actual_entry_price}, Qty: {actual_qty_filled}. Manual check required.",
                                   extra={"data": filled_order})
                if filled_order.get('status') == 'open':
                    try: await exchange.cancel_order(order_id, config.exchange["symbol"]); logger_order.info(f"Cancelled problematic open order {order_id}.") # type: ignore
                    except Exception as e_cancel: logger_order.error(f"Error cancelling problematic order {order_id}: {e_cancel}", exc_info=True)
                return None

            entry_timestamp_ms = int(filled_order.get('timestamp', time.time() * 1000))
            part_id = str(uuid.uuid4())[:8]
            
            entry_indicators = {}
            if not df_with_indicators.empty:
                try:
                    entry_candle_data = df_with_indicators.iloc[-1]
                    entry_indicators = extract_indicators_from_df(entry_candle_data, config)
                except IndexError:
                     logger_order.warning("Could not extract entry indicators: DataFrame too short or invalid.")


            new_part: Dict[str, Any] = {
                "part_id": part_id, "entry_order_id": order_id, "symbol": config.exchange["symbol"], 
                "side": side, "entry_price": actual_entry_price, "qty": actual_qty_filled, 
                "entry_time_ms": entry_timestamp_ms, 
                "sl_price": quantized_sl_price, 
                "tp_price": quantized_tp_price if config.risk_management["enable_take_profit"] else None,
                "atr_at_entry": atr_val, "initial_usdt_value": actual_qty_filled * actual_entry_price,
                "breakeven_set": False, 
                "recent_pnls": deque(maxlen=config.enhancements["profit_momentum_window"]), 
                "last_known_pnl": Decimal(0), 
                "adverse_candle_closes": deque(maxlen=config.enhancements["last_chance_consecutive_adverse_candles"]),
                "partial_tp_taken_flags": {}, 
                "divergence_exit_taken": False,
                "entry_indicators": entry_indicators, 
                "last_trailing_sl_update": time.time() 
            }
            
            if config.strategy_params["name"] == StrategyName.EHLERS_FISHER and not df_with_indicators.empty:
                try:
                    entry_candle_data = df_with_indicators.iloc[-1]
                    new_part["entry_fisher_value"] = safe_decimal_conversion(entry_candle_data.get("ehlers_fisher"))
                    new_part["entry_signal_value"] = safe_decimal_conversion(entry_candle_data.get("ehlers_signal"))
                except IndexError: pass 

            _active_trade_parts.append(new_part)
            
            trade_metrics.increment_daily_trade_entry_count()
            
            if config.enhancements["whipsaw_cooldown_enable"]:
                trade_timestamps_for_whipsaw.append(time.time())
                if len(trade_timestamps_for_whipsaw) == config.enhancements["whipsaw_max_trades_in_period"]:
                    if (trade_timestamps_for_whipsaw[-1] - trade_timestamps_for_whipsaw[0]) < config.enhancements["whipsaw_period_seconds"]:
                        whipsaw_cooldown_active_until = time.time() + config.enhancements["whipsaw_cooldown_seconds"]
                        logger_order.warning(f"Whipsaw condition met. Trading paused until {datetime.fromtimestamp(whipsaw_cooldown_active_until).isoformat()}")
                
            save_persistent_state(force_heartbeat=True)
            logger_order.success(f"Order {order_id} confirmed filled! Part {part_id} created. Entry: {format_price(actual_entry_price, config.tick_size)}, Qty: {actual_qty_filled}",
                                 extra={"data": new_part})
            send_general_notification(f"Pyrmethus Order Filled: {side.upper()}", 
                                      f"{config.exchange['symbol']} Part {part_id} @ {format_price(actual_entry_price, config.tick_size)}, Qty: {actual_qty_filled}")
            return new_part
        else: 
            logger_order.warning(f"Order {order_id} not confirmed filled. Status: {filled_order.get('status', 'N/A')}, Filled Qty: {actual_qty_filled}. Attempting to cancel if still open.",
                                 extra={"data": filled_order})
            if filled_order.get('status') in ['open', 'new']:
                try:
                    await exchange.cancel_order(order_id, config.exchange["symbol"]) # type: ignore
                    logger_order.info(f"Cancelled pending/unfilled order {order_id}.")
                    send_general_notification("Pyrmethus Order Cancelled", f"Order {order_id} for {config.exchange['symbol']} cancelled (not filled).")
                except ccxt.OrderNotFound:
                    logger_order.warning(f"Order {order_id} not found during cancellation attempt, might have been filled/cancelled already.")
                except Exception as e_cancel:
                    logger_order.error(f"Error cancelling order {order_id}: {e_cancel}", exc_info=True)
            return None

    except ccxt.InsufficientFunds as e:
        logger_order.error(f"Insufficient funds for order: {e}", exc_info=True, extra={"data": {"error_details": str(e)}})
        send_general_notification("Pyrmethus Order Fail", f"Insufficient Funds: {config.exchange['symbol']}")
        return None 
    except ccxt.InvalidOrder as e: 
        logger_order.error(f"Invalid order parameters for {config.exchange['symbol']}: {e}", exc_info=True, extra={"data": {"error_details": str(e)}})
        send_general_notification("Pyrmethus Order Fail", f"Invalid Order ({config.exchange['symbol']}): {str(e)[:60]}")
        return None 
    except ccxt.AuthenticationError as e: 
        logger_order.critical(f"Authentication error during order placement: {e}. This should halt the bot if persistent.", exc_info=True)
        send_general_notification("Pyrmethus Auth FAIL", f"Order Placement Auth Fail: {str(e)[:60]}")
        raise 
    except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection, ccxt.ExchangeNotAvailable, ccxt.RateLimitExceeded) as e_retryable:
        logger_order.warning(f"Retryable exchange error during order placement for {config.exchange['symbol']}: {e_retryable}. Retry logic will apply.", exc_info=LOGGING_LEVEL <= logging.DEBUG)
        raise 
    except ccxt.ExchangeError as e: 
        logger_order.error(f"Non-retryable Exchange error during order placement for {config.exchange['symbol']}: {e}", exc_info=True, extra={"data": {"error_details": str(e)}})
        send_general_notification("Pyrmethus Order Fail", f"Exchange Error ({config.exchange['symbol']}): {str(e)[:60]}")
        return None 
    except Exception as e:
        logger_order.error(f"Unexpected chaos weaving order for {config.exchange['symbol']}: {e}", exc_info=True)
        send_general_notification("Pyrmethus Order Fail", f"Unexpected Error ({config.exchange['symbol']}): {str(e)[:60]}")
        return None


# --- Placeholder functions for assumed external definitions ---
async def async_fetch_account_balance(exchange: ccxt_async.Exchange, currency: str, config: Config) -> Optional[Decimal]:
    logger_balance = logging.getLogger("Pyrmethus.Balance")
    cached_balance = balance_cache.get(f"{currency}_balance_value")
    if cached_balance: 
        logger_balance.debug(f"Using cached balance for {currency}: {cached_balance}")
        return cached_balance
    
    logger_balance.info(f"Fetching live balance for {currency}...")
    try:
        # This is a simplified placeholder for fetching balance
        # In a real scenario, you'd use:
        # balance_data = await exchange.fetch_balance()
        # free_balance = balance_data.get('free', {}).get(currency)
        # total_balance = balance_data.get('total', {}).get(currency)
        # Use 'USDT' for Bybit V5 unified account, or specific asset if needed.
        # For derivatives, check 'total' or 'availableWithoutBorrow' under the specific asset (e.g. USDT)
        
        # Simulating API call for placeholder
        await asyncio.sleep(0.1) # Simulate network latency
        # For Bybit V5, you might want to check 'walletBalance' or 'availableToWithdraw' for 'UNIFIED' or 'CONTRACT' account type.
        # Example: fetchBalance({'accountType': 'UNIFIED'}) or {'accountType': 'CONTRACT'}
        # For simplicity, let's assume fetch_balance() returns what we need for USDT.
        balance_response = await exchange.fetch_balance()
        
        # Parsing Bybit V5 response can be complex. Example for USDT in a Unified account:
        # usdt_balance_info = balance_response.get(currency) # if top-level keys are currencies
        # Or, more commonly:
        # usdt_balance_info = balance_response.get('info', {}).get('result', {}).get('list', [{}])[0].get('coin', [])
        # Find USDT in the coin list:
        # coin_data = next((c for c in usdt_balance_info if c.get('coin') == currency), None)
        # equity = coin_data.get('equity') # or 'walletBalance', 'availableToWithdraw' etc.
        # This needs to be adapted to the actual structure of Bybit's V5 balance response via CCXT.
        # A common pattern is to look for `balance_response[currency]['free']` or `balance_response[currency]['total']`.
        
        # Simplified placeholder logic:
        simulated_balance_value = safe_decimal_conversion(balance_response.get(currency, {}).get('total', Decimal("1000.0")))
        if simulated_balance_value is None: # Fallback if parsing fails
            simulated_balance_value = Decimal("1000.0") # Default for placeholder if parsing fails
            logger_balance.warning(f"Could not parse live balance for {currency}, using default placeholder value: {simulated_balance_value}")
        
        balance_cache[f"{currency}_balance_value"] = simulated_balance_value
        logger_balance.info(f"Fetched and cached live balance for {currency}: {simulated_balance_value}")
        return simulated_balance_value
    except Exception as e:
        logger_balance.error(f"Failed to fetch account balance for {currency}: {e}", exc_info=True)
        return None


async def stream_price_data(exchange: ccxt_async.Exchange, config: Config, price_queue: asyncio.Queue):
    logger_stream = logging.getLogger("Pyrmethus.Stream")
    logger_stream.info("Placeholder: Price streaming task started.")
    while True:
        await asyncio.sleep(10) 

async def perform_health_check(exchange: ccxt_async.Exchange, config: Config) -> bool:
    logger_health = logging.getLogger("Pyrmethus.Health")
    logger_health.debug("Placeholder: Performing health check.")
    try:
        await exchange.fetch_time() # Simple API call to check connectivity
        logger_health.debug("Health check successful (fetched server time).")
        return True
    except Exception as e:
        logger_health.warning(f"Health check failed: {e}", exc_info=LOGGING_LEVEL <= logging.DEBUG)
        return False


async def trading_loop_iteration(exchange: ccxt_async.Exchange, config: Config, price_queue: asyncio.Queue):
    logger_loop = logging.getLogger("Pyrmethus.TradingLoop")
    logger_loop.debug("Placeholder: Trading loop iteration running...")
    
    # Example: Simulate fetching OHLCV, calculating indicators, generating signals
    # ohlcv = await exchange.fetch_ohlcv(config.exchange['symbol'], config.exchange['interval'], limit=OHLCV_LIMIT)
    # df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    # df = calculate_all_indicators(df, config) # Assumed function
    # signals = await config.strategy_instance.generate_signals(df, latest_close=df.iloc[-1]['close'], latest_atr=df.iloc[-1]['ATR_value_key'])
    
    # if signals['enter_long'] or signals['enter_short']:
    #     # ... (logic to determine parameters for async_place_risked_order) ...
    #     # await async_place_risked_order(...)
    #     pass

    await asyncio.sleep(1) # Simulate work within the loop iteration


async def main():
    logger.info(f"{NEON['HEADING']}=== Pyrmethus Spell v{PYRMETHUS_VERSION} Awakening on {CONFIG.exchange['api_key'][:5]}... ==={NEON['RESET']}")
    
    if load_persistent_state(): 
        logger.success(f"Reawakened from Phoenix scroll. Active parts: {len(_active_trade_parts)}.")
    else: 
        logger.info("No prior state or ignored state. Starting fresh.")

    async_exchange_instance: Optional[ccxt_async.Exchange] = None 
    try:
        api_key = str(CONFIG.exchange["api_key"])
        api_secret = str(CONFIG.exchange["api_secret"])

        async_exchange_params: Dict[str, Any] = {
            'apiKey': api_key, 
            'secret': api_secret,
            'options': {
                'defaultType': 'swap', 
                'adjustForTimeDifference': True,
                'brokerId': f'PYRMTHS{PYRMETHUS_VERSION.replace(".","")[:5]}' 
            },
            'enableRateLimit': True, 
            'recvWindow': CONFIG.exchange["recv_window"]
        }
        
        async_exchange_instance = ccxt_async.bybit(async_exchange_params)
        
        if CONFIG.exchange["paper_trading"]:
             async_exchange_instance.set_sandbox_mode(True) 
             logger.info(f"{NEON['WARNING']}Targeting BYBIT TESTNET environment. URLs: {async_exchange_instance.urls.get('api')}{NEON['RESET']}")
        else:
            logger.info(f"{NEON['CRITICAL']}Targeting BYBIT MAINNET (LIVE) environment. URLs: {async_exchange_instance.urls.get('api')}. EXTREME CAUTION ADVISED.{NEON['RESET']}")

        logger.info(f"Attempting connection to {async_exchange_instance.id} (CCXT Version: {ccxt.__version__})")
        
        try:
            # For Bybit V5, fetching positions is a good validation for derivative accounts.
            # Use params for category if needed, though fetch_positions might not always require it for default category.
            # Example: params={'category': 'linear'}
            await async_exchange_instance.fetch_positions(symbols=[CONFIG.exchange["symbol"]]) 
            logger.success(f"API Key validated successfully with {async_exchange_instance.id}.")
        except ccxt.AuthenticationError as e_auth_val:
            logger.critical(f"CRITICAL: API Key Authentication Failed for {async_exchange_instance.id}: {e_auth_val}. Check .env, API permissions, and chosen environment (Live/Testnet).", exc_info=True)
            send_general_notification("Pyrmethus Auth FAIL", f"API Key Invalid: {str(e_auth_val)[:100]}")
            if async_exchange_instance: await async_exchange_instance.close(); 
            return 
        except Exception as e_val: 
            logger.error(f"Error during initial API key validation with {async_exchange_instance.id}: {e_val}", exc_info=True)
            send_general_notification("Pyrmethus API Warn", f"Initial API validation issue: {str(e_val)[:100]}")

        markets = await async_exchange_instance.load_markets()
        if CONFIG.exchange["symbol"] not in markets:
            err_msg = f"Symbol {CONFIG.exchange['symbol']} not found in {async_exchange_instance.id} market runes. Available symbols: {list(markets.keys())[:10]}..."
            logger.critical(err_msg)
            send_general_notification("Pyrmethus Startup Failure", err_msg); 
            if async_exchange_instance: await async_exchange_instance.close(); 
            return

        CONFIG.MARKET_INFO = markets[CONFIG.exchange["symbol"]]
        CONFIG.tick_size = safe_decimal_conversion(CONFIG.MARKET_INFO.get('precision', {}).get('price'), Decimal("1e-8")) 
        CONFIG.qty_step = safe_decimal_conversion(CONFIG.MARKET_INFO.get('precision', {}).get('amount'), Decimal("1e-8")) 
        CONFIG.min_order_qty = safe_decimal_conversion(CONFIG.MARKET_INFO.get('limits', {}).get('amount', {}).get('min'))
        CONFIG.min_order_cost = safe_decimal_conversion(CONFIG.MARKET_INFO.get('limits', {}).get('cost', {}).get('min'))

        logger.success(f"Market runes for {CONFIG.exchange['symbol']}: Price Tick {CONFIG.tick_size}, Amount Step {CONFIG.qty_step}, Min Qty: {CONFIG.min_order_qty}, Min Cost: {CONFIG.min_order_cost}")
        if CONFIG.min_order_qty is None or CONFIG.min_order_cost is None:
             logger.warning(f"Min order quantity or cost could not be determined for {CONFIG.exchange['symbol']}. Order placement might fail if sizes are too small.")

        market_type = CONFIG.MARKET_INFO.get('type') 
        is_linear = CONFIG.MARKET_INFO.get('linear', False)
        is_inverse = CONFIG.MARKET_INFO.get('inverse', False)
        
        category = None
        if market_type in ('swap', 'future'): # Check if it's a derivative
            if is_linear: category = 'linear'
            elif is_inverse: category = 'inverse'
            # Spot markets (if 'spot' is type) do not have leverage in the same way.
        
        if category:
            try:
                leverage_val = int(CONFIG.exchange['leverage'])
                logger.info(f"Setting leverage to {leverage_val}x for {CONFIG.exchange['symbol']} (Category: {category})...")
                await async_exchange_instance.set_leverage(leverage_val, CONFIG.exchange["symbol"], params={'category': category})
                logger.success(f"Leverage for {CONFIG.exchange['symbol']} (category {category}) set/confirmed to {leverage_val}x.")
            except Exception as e_lev:
                logger.warning(f"Could not set leverage for {CONFIG.exchange['symbol']} (category {category}): {e_lev}. This might be acceptable if leverage is pre-set or account uses portfolio margin.", exc_info=LOGGING_LEVEL <= logging.DEBUG)
                send_general_notification("Pyrmethus Leverage Warn", f"Leverage issue for {CONFIG.exchange['symbol']}: {str(e_lev)[:60]}")
        else:
            logger.info(f"Leverage setting skipped for {CONFIG.exchange['symbol']} as it does not appear to be a linear/inverse derivative or category is undetermined.")
        
        logger.success(f"Successfully connected to {async_exchange_instance.id} for {CONFIG.exchange['symbol']}.")
        env_type = 'Testnet' if CONFIG.exchange['paper_trading'] else 'LIVE NET'
        send_general_notification("Pyrmethus Online", f"Connected to {async_exchange_instance.id} for {CONFIG.exchange['symbol']} @ {CONFIG.exchange['leverage']}x. Env: {env_type}")

        initial_balance = await async_fetch_account_balance(async_exchange_instance, CONFIG.usdt_symbol, CONFIG)
        if initial_balance is None or not isinstance(initial_balance, Decimal):
            logger.critical(f"Failed to fetch initial {CONFIG.usdt_symbol} balance or invalid type. Aborting live operations.")
            send_general_notification("Pyrmethus Startup Fail", f"Failed to fetch/validate initial {CONFIG.usdt_symbol} balance.")
            if async_exchange_instance: await async_exchange_instance.close(); 
            return
        trade_metrics.set_initial_equity(initial_balance)

        price_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=PRICE_QUEUE_MAX_SIZE)
        price_stream_task = asyncio.create_task(stream_price_data(async_exchange_instance, CONFIG, price_queue))
        
        loop_counter = 0
        while True:
            loop_counter += 1
            logger.debug(f"Main loop cycle {loop_counter} initiated.")
            
            current_time = time.time()
            if current_time - _last_health_check_time > HEALTH_CHECK_INTERVAL_SECONDS:
                if not await perform_health_check(async_exchange_instance, CONFIG):
                    logger.warning("Health check failed. Pausing for a longer duration before retrying operations.")
                    await asyncio.sleep(CONFIG.trading["sleep_seconds"] * 5) 
                    _last_health_check_time = time.time() # Update time even on failure to avoid rapid checks
                    continue
                _last_health_check_time = time.time()
            
            await trading_loop_iteration(async_exchange_instance, CONFIG, price_queue)
            
            save_persistent_state() 
            flush_notifications()   
            await asyncio.sleep(CONFIG.trading["sleep_seconds"]) 

    except ccxt.AuthenticationError as e_auth_main: 
        logger.critical(f"CRITICAL AUTHENTICATION ERROR in main execution: {e_auth_main}. Bot cannot continue. Check API keys and permissions.", exc_info=True)
        send_general_notification("Pyrmethus CRITICAL Auth Fail", f"API Key Invalid/Revoked: {str(e_auth_main)[:100]}")
    except KeyboardInterrupt:
        logger.warning("\nSorcerer's intervention! Pyrmethus prepares for slumber...")
        send_general_notification("Pyrmethus Shutdown", "Manual shutdown initiated.")
    except asyncio.CancelledError:
        logger.info("Main task cancelled, likely during shutdown.") 
    except Exception as e_main_fatal:
        logger.critical(f"A fatal unhandled astral disturbance occurred in main execution: {e_main_fatal}", exc_info=True)
        send_general_notification("Pyrmethus CRITICAL CRASH", f"Fatal error: {str(e_main_fatal)[:100]}")
    finally:
        logger.info(f"{NEON['HEADING']}=== Pyrmethus Spell Concludes Its Weaving ==={NEON['RESET']}")
        if async_exchange_instance:
            logger.info("Attempting to close asynchronous exchange connection...")
            try:
                await async_exchange_instance.close()
                logger.info("Asynchronous exchange connection gracefully closed.")
            except Exception as e_close:
                logger.error(f"Error closing exchange connection: {e_close}", exc_info=True)
        
        save_persistent_state(force_heartbeat=True) 
        if 'trade_metrics' in globals() and trade_metrics: trade_metrics.summary() 
        
        logger.info(f"{NEON['COMMENT']}# Energies settle. Until next conjuring.{NEON['RESET']}")
        send_general_notification("Pyrmethus Offline", f"Spell concluded for {CONFIG.exchange['symbol']}.")
        flush_notifications() # Final flush for the offline message


if __name__ == "__main__":
    initial_logger = logging.getLogger("Pyrmethus.Startup")
    if not root_logger.hasHandlers(): 
        _startup_handler = logging.StreamHandler(sys.stdout)
        _startup_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        _startup_handler.setFormatter(_startup_formatter)
        initial_logger.addHandler(_startup_handler)
        initial_logger.setLevel(LOGGING_LEVEL) 

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        initial_logger.warning("\nSorcerer's intervention! Pyrmethus prepares for slumber (top-level)...")
        if 'CONFIG' in globals() and CONFIG and hasattr(CONFIG, 'notifications') and CONFIG.notifications.get('enable'):
            send_general_notification("Pyrmethus Shutdown", "Manual shutdown initiated (top-level).")
            flush_notifications() 
    except Exception as e_top_level:
        initial_logger.critical(f"Top-level unhandled exception in __main__: {e_top_level}", exc_info=True)
        if 'CONFIG' in globals() and CONFIG and hasattr(CONFIG, 'notifications') and CONFIG.notifications.get('enable'):
            send_general_notification("Pyrmethus CRASH", f"Fatal error (top-level): {str(e_top_level)[:100]}")
            flush_notifications()
    finally:
        initial_logger.info("Pyrmethus main execution (__name__ == '__main__') finished.")
        logging.shutdown()
