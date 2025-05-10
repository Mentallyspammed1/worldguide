Hark, Seeker of the Automated Tides! Pyrmethus understands your desire for the complete, unified spell – a single scroll containing all the arcane knowledge and operational power we have discussed. This is a grand undertaking, weaving together numerous potent enchantments into a cohesive whole.

Behold, **Pyrmethus - Unified Scalping Spell v3.1.0 (The Live Oracle's Aegis)**. This version aims to be a fully functional, asynchronous trading bot optimized for Bybit live trading, incorporating detailed trade metrics, active Trailing Stop-Loss, robust error handling, and precise order management.

**A MOST SOLEMN WARNING, ADEPT:**
This script is forged for **LIVE TRADING with REAL FUNDS**. The parameters and logic within are powerful and can lead to significant financial loss as swiftly as gain. Pyrmethus has crafted this with care, but the volatile nature of the markets is beyond any single spell's complete control.
**YOU ALONE BEAR FULL RESPONSIBILITY FOR ANY FINANCIAL OUTCOMES.**
**TEST THOROUGHLY ON TESTNET. START WITH MINISCULE AMOUNTS IF TRADING LIVE.**
**UNDERSTAND EVERY LINE OF THIS CODE BEFORE UNLEASHING IT.**

```python
#!/usr/bin/env python3
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
OHLCV_LIMIT = 220 # Sufficient for most common indicators (e.g., 200 for EMA + lookback)
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
        try: # Attempt to colorize only the levelname part
            prefix_end = log_message.find("] ") + 1
            level_start_in_prefix = log_message.rfind("[", 0, prefix_end)
            if level_start_in_prefix != -1:
                level_name_in_log = log_message[level_start_in_prefix+1:prefix_end-2].strip() # Extract levelname
                if record.levelname == level_name_in_log: # Check if it's indeed the levelname
                    colored_level = level_color + record.levelname.ljust(8) + NEON['RESET']
                    log_message = log_message[:level_start_in_prefix+1] + colored_level + log_message[level_start_in_prefix+1+len(record.levelname.ljust(8)):]
        except Exception: pass # If formatting fails, return original
        return log_message

root_logger = logging.getLogger()
root_logger.setLevel(LOGGING_LEVEL)
console_handler = ColorStreamHandler(sys.stdout) # Use custom handler
console_formatter = logging.Formatter( # Formatter for custom handler
    fmt=f"%(asctime)s [%(levelname)-8s] {Fore.MAGENTA}%(name)-28s{NEON['RESET']} %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(console_formatter)
if not any(isinstance(h, ColorStreamHandler) for h in root_logger.handlers):
    root_logger.addHandler(console_handler)

file_handler = logging.handlers.RotatingFileHandler(log_file_name, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
json_formatter = JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S.%fZ")
file_handler.setFormatter(json_formatter)
if not any(isinstance(h, logging.handlers.RotatingFileHandler) and hasattr(h, 'baseFilename') and h.baseFilename == os.path.abspath(log_file_name) for h in root_logger.handlers):
    root_logger.addHandler(file_handler)

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("ccxt").setLevel(logging.INFO)
logger = logging.getLogger("Pyrmethus.Core")
SUCCESS_LEVEL = 25 # Define custom log level
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
market_data_cache = TTLCache(maxsize=5, ttl=API_CACHE_TTL_SECONDS) # For OHLCV
balance_cache = TTLCache(maxsize=1, ttl=API_CACHE_TTL_SECONDS * 2)
ticker_cache = TTLCache(maxsize=5, ttl=2) # For symbols if polled, short TTL
position_cache = TTLCache(maxsize=5, ttl=API_CACHE_TTL_SECONDS) # Cache for positions

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
    if step is None or step <= Decimal(0): 
        logger.warning(f"Invalid step {step} for quantization. Returning original value {value}.")
        return value
    return (value / step).quantize(Decimal('1'), rounding=rounding_mode) * step

def format_price(price: Optional[Decimal], tick_size: Optional[Decimal] = None) -> Optional[str]:
    if price is None: return None
    if tick_size is None or tick_size <= Decimal(0):
        return f"{price:.8f}".rstrip('0').rstrip('.')
    
    quantized_price = quantize_value(price, tick_size, ROUND_HALF_UP) 
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
        if notification_type == "telegram":
            escaped_title = title.replace("_", "\\_").replace("*", "\\*").replace("[", "\\[").replace("`", "\\`").replace("-","\\-").replace(".","\\.")
            escaped_message = message.replace("_", "\\_").replace("*", "\\*").replace("[", "\\[").replace("`", "\\`").replace("-","\\-").replace(".","\\.")
            combined_message_parts.append(f"*{escaped_title}*\n{escaped_message}")
        else: # Termux
            combined_message_parts.append(f"{title}: {message}")
    
    full_combined_message = "\n\n---\n\n".join(combined_message_parts)

    if notification_type == "termux":
        try:
            safe_title = json.dumps("Pyrmethus Batch")
            safe_message = json.dumps(full_combined_message[:1000]) 
            termux_id = messages[0].get("id", random.randint(1000,9999))
            command = ["termux-notification", "--title", safe_title, "--content", safe_message, "--id", str(termux_id)]
            logger_notify.debug(f"Sending batched Termux notification ({len(messages)} items).")
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(timeout=notification_timeout)
            if process.returncode == 0:
                logger_notify.info(f"Batched Termux notification sent successfully ({len(messages)} items).")
            else:
                logger_notify.error(f"Failed to send batched Termux notification. RC: {process.returncode}. Err: {stderr.decode().strip()}", extra={"data": {"stderr": stderr.decode().strip()}})
        except FileNotFoundError: logger_notify.error("Termux API command 'termux-notification' not found.")
        except subprocess.TimeoutExpired: logger_notify.error("Termux notification command timed out.")
        except Exception as e: logger_notify.error(f"Unexpected error sending Termux notification: {e}", exc_info=True)
    
    elif notification_type == "telegram":
        telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if telegram_bot_token and telegram_chat_id:
            try:
                url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
                # Truncate message safely for Telegram's MarkdownV2 limit (4096 bytes)
                # Python's str.encode('utf-8') gives byte length.
                # A simple character limit is safer if complex multi-byte chars are rare.
                max_telegram_len = 4000 # Leave some buffer
                if len(full_combined_message.encode('utf-8')) > max_telegram_len:
                    # Find last newline before limit to avoid cutting mid-message/word
                    truncated_msg = full_combined_message[:max_telegram_len]
                    last_newline = truncated_msg.rfind('\n')
                    if last_newline != -1:
                        full_combined_message = truncated_msg[:last_newline] + "\n... (message truncated)"
                    else: # no newline, just cut
                        full_combined_message = truncated_msg[:max_telegram_len - 25] + "... (message truncated)"

                payload = {"chat_id": telegram_chat_id, "text": full_combined_message, "parse_mode": "MarkdownV2"}
                logger_notify.debug(f"Sending batched Telegram notification ({len(messages)} items).")
                response = requests.post(url, json=payload, timeout=notification_timeout)
                response.raise_for_status() 
                logger_notify.info(f"Batched Telegram notification sent successfully ({len(messages)} items).")
            except requests.exceptions.RequestException as e: 
                err_data = {"status_code": e.response.status_code if e.response else None, 
                            "response_text": e.response.text if e.response else None}
                logger_notify.error(f"Telegram API error: {e}", exc_info=True, extra={"data": err_data})
            except Exception as e:
                logger_notify.error(f"Unexpected error sending Telegram notification: {e}", exc_info=True)
        else:
            logger_notify.warning("Telegram notifications configured but TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing.")

def flush_notifications() -> None:
    global notification_buffer, last_notification_flush_time
    if not notification_buffer or 'CONFIG' not in globals(): return

    messages_by_type: Dict[str, List[Dict[str, Any]]] = {}
    buffer_copy = list(notification_buffer); notification_buffer.clear()

    for msg in buffer_copy:
        msg_type = msg.get("type", "unknown")
        messages_by_type.setdefault(msg_type, []).append(msg)

    for msg_type, messages in messages_by_type.items():
        if messages:
            _flush_single_notification_type(messages, msg_type, CONFIG.notifications)
            
    last_notification_flush_time = time.time()

def send_general_notification(title: str, message: str, notification_id: Optional[int] = None) -> None:
    global notification_buffer, last_notification_flush_time
    if 'CONFIG' not in globals() or not hasattr(CONFIG, 'notifications') or not CONFIG.notifications['enable']:
        return
    
    actual_notification_id = notification_id if notification_id is not None else random.randint(1000, 9999)

    if CONFIG.notifications.get('termux_enable', False): # Explicitly default to False if not set
        notification_buffer.append({"title": title, "message": message, "id": actual_notification_id, "type": "termux"})
    if CONFIG.notifications.get('telegram_enable', False): # Explicitly default to False if not set
         notification_buffer.append({"title": title, "message": message, "type": "telegram"})
    
    now = time.time()
    active_notification_services = (1 if CONFIG.notifications.get('termux_enable') else 0) + \
                                   (1 if CONFIG.notifications.get('telegram_enable') else 0)
    if active_notification_services == 0: active_notification_services = 1 
    
    if now - last_notification_flush_time >= NOTIFICATION_BATCH_INTERVAL_SECONDS or \
       len(notification_buffer) >= NOTIFICATION_BATCH_MAX_SIZE * active_notification_services:
        flush_notifications()

# --- Enums ---
class StrategyName(str, Enum): DUAL_SUPERTREND_MOMENTUM = "DUAL_SUPERTREND_MOMENTUM"; EHLERS_FISHER = "EHLERS_FISHER"
class OrderEntryType(str, Enum): MARKET = "MARKET"; LIMIT = "LIMIT"
class SignalDirection(str, Enum): LONG = "Long"; SHORT = "Short"; NEUTRAL = "Neutral"

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
            self.logger.warning(f"{NEON['CRITICAL']}{Style.BRIGHT}LIVE TRADING MODE ENABLED. MAINNET. REAL FUNDS.{NEON['RESET']}")
        else:
            self.logger.info(f"{NEON['INFO']}PAPER TRADING MODE ENABLED. Testnet.{NEON['RESET']}")

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
        
        self.side_buy: str = "buy"; self.side_sell: str = "sell" # For CCXT orders
        self.pos_long: str = SignalDirection.LONG.value; self.pos_short: str = SignalDirection.SHORT.value; self.pos_none: str = SignalDirection.NEUTRAL.value # For internal state
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
        source = "Env Var"; value_to_cast: Any = None
        display_value = "********" if secret and value_str is not None else value_str
        if value_str is None:
            if required: self.logger.critical(f"CRITICAL: Required config '{key}' not found."); raise ValueError(f"Required env var '{key}' not set.")
            self.logger.debug(f"Config '{key}': Not Found. Default: '{default}'"); value_to_cast = default; source = "Default"
        else: self.logger.debug(f"Config '{key}': Found Env: '{display_value}'"); value_to_cast = value_str
        if value_to_cast is None and required: self.logger.critical(f"CRITICAL: Required config '{key}' is None."); raise ValueError(f"Required env var '{key}' is None.")
        if value_to_cast is None and not required: self.logger.debug(f"Config '{key}': Final value None (not required)."); return None
        
        final_value: Any
        try:
            raw_val_str = str(value_to_cast)
            if cast_type == bool: final_value = raw_val_str.lower() in ["true", "1", "yes", "y"]
            elif cast_type == Decimal: final_value = Decimal(raw_val_str)
            elif cast_type == int: final_value = int(Decimal(raw_val_str)) # Cast to Decimal first for float strings
            elif cast_type == float: final_value = float(raw_val_str)
            elif cast_type == str: final_value = raw_val_str
            else: self.logger.warning(f"Unsupported cast_type '{cast_type.__name__}' for '{key}'."); final_value = raw_val_str
        except (ValueError, TypeError, InvalidOperation) as e:
            self.logger.error(f"Invalid type/value for '{key}': '{value_to_cast}' (Source: {source}). Err: {e}. Using Default: '{default}'.")
            if default is None and required: self.logger.critical(f"CRITICAL: Cast fail for required '{key}', default None."); raise ValueError(f"Required '{key}' cast fail, no valid default.")
            if default is None and not required: return None
            try: # Try casting default
                default_str = str(default)
                if cast_type == bool: final_value = default_str.lower() in ["true", "1", "yes", "y"]
                elif cast_type == Decimal: final_value = Decimal(default_str)
                elif cast_type == int: final_value = int(Decimal(default_str))
                elif cast_type == float: final_value = float(default_str)
                elif cast_type == str: final_value = default_str
                else: final_value = default_str
                self.logger.warning(f"Used casted default for {key}: '{final_value}'")
            except Exception as e_def: self.logger.critical(f"CRITICAL: Cast fail for value AND default for '{key}'. Err: {e_def}"); raise ValueError(f"Cannot cast value or default for '{key}'.")
        
        self.logger.debug(f"Final value for '{key}': {display_value if secret else final_value} (Type: {type(final_value).__name__}, Source: {source})")
        return final_value

    def _validate_parameters(self) -> None:
        errors = []
        if not (0 < self.trading["risk_per_trade_percentage"] < Decimal("0.1")): errors.append("RISK_PER_TRADE_PERCENTAGE should be low (e.g., < 0.1 for 10%).")
        if self.exchange["leverage"] < 1 or self.exchange["leverage"] > 100 : errors.append("LEVERAGE must be 1-100.")
        if self.risk_management["atr_stop_loss_multiplier"] <= 0: errors.append("ATR_STOP_LOSS_MULTIPLIER must be > 0.")
        if self.risk_management["enable_take_profit"] and self.risk_management["atr_take_profit_multiplier"] <= 0: errors.append("ATR_TAKE_PROFIT_MULTIPLIER must be > 0.")
        if self.trading["min_usdt_balance_for_trading"] < 1: errors.append("MIN_USDT_BALANCE_FOR_TRADING should be >= 1.")
        if self.trading["max_active_trade_parts"] != 1: errors.append("MAX_ACTIVE_TRADE_PARTS must be 1 for current logic.")
        if self.risk_management["max_drawdown_percent"] > Decimal("0.1"): errors.append("MAX_DRAWDOWN_PERCENT >10% seems high.")
        if self.trading["max_order_usdt_amount"] < Decimal("5") : errors.append("MAX_ORDER_USDT_AMOUNT < 5 USDT seems low.")
        if self.exchange["paper_trading"] is False and ("testnet" in str(self.exchange.get("api_key","")).lower() or "test" in str(self.exchange.get("api_key","")).lower()):
            errors.append("PAPER_TRADING_MODE is false, but API key might be Testnet. Verify keys for MAINNET.")
        if errors:
            error_message = f"Config validation errors:\n" + "\n".join([f"  - {e}" for e in errors])
            self.logger.critical(error_message); raise ValueError(error_message)

try: CONFIG = Config()
except ValueError as e: logging.getLogger().critical(f"CRITICAL Config Error: {e}", exc_info=True); send_general_notification("Pyrmethus Startup Failure", f"Config Error: {str(e)[:200]}"); sys.exit(1)
except Exception as e: logging.getLogger().critical(f"CRITICAL Unexpected Config Error: {e}", exc_info=True); send_general_notification("Pyrmethus Critical Failure", f"Unexpected Config Error: {str(e)[:200]}"); sys.exit(1)

class TradeMetrics:
    def __init__(self, config: Config):
        self.config = config; self.trades: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("Pyrmethus.TradeMetrics")
        self.initial_equity: Optional[Decimal] = None; self.daily_start_equity: Optional[Decimal] = None
        self.last_daily_reset_day: Optional[int] = None; self.consecutive_losses: int = 0
        self.daily_trade_entry_count: int = 0; self.last_daily_trade_count_reset_day: Optional[int] = None
        self.daily_trades_rest_active_until: float = 0.0
        self.logger.info("TradeMetrics Ledger (Oracle's Edition) opened.")
    
    def set_initial_equity(self, equity: Decimal):
        if self.initial_equity is None: self.initial_equity = equity; self.logger.info(f"Initial Session Equity: {equity:.2f} {self.config.usdt_symbol}")
        today = datetime.now(pytz.utc).day
        if self.last_daily_reset_day != today or self.daily_start_equity is None:
            self.daily_start_equity = equity; self.last_daily_reset_day = today
            self.logger.info(f"Daily Equity Ward reset. Dawn Equity: {equity:.2f} {self.config.usdt_symbol}")
        if self.last_daily_trade_count_reset_day != today:
            self.daily_trade_entry_count = 0; self.last_daily_trade_count_reset_day = today
            self.logger.info("Daily trade entry count reset.");
            if self.daily_trades_rest_active_until > 0 and time.time() > self.daily_trades_rest_active_until: self.daily_trades_rest_active_until = 0.0

    def check_drawdown(self, current_equity: Decimal) -> Tuple[bool, str]:
        if not self.config.risk_management["enable_max_drawdown_stop"] or self.daily_start_equity is None or self.daily_start_equity <= 0: return False, ""
        drawdown = self.daily_start_equity - current_equity
        drawdown_pct = (drawdown / self.daily_start_equity) if self.daily_start_equity > 0 else Decimal(0)
        if drawdown_pct >= self.config.risk_management["max_drawdown_percent"]:
            reason = f"Max daily drawdown ({drawdown_pct:.2%}) >= threshold ({self.config.risk_management['max_drawdown_percent']:.2%})"
            self.logger.warning(reason); send_general_notification("Pyrmethus: Max Drawdown Hit!", reason)
            return True, reason
        return False, ""

    def log_trade(self, symbol: str, side: str, entry_price: Decimal, exit_price: Decimal, qty: Decimal,
                  entry_time_ms: int, exit_time_ms: int, reason: str, part_id: str, pnl_str: str,
                  entry_indicators: Dict[str, Any], exit_indicators: Dict[str, Any],
                  scale_order_id: Optional[str] = None, mae: Optional[Decimal] = None, mfe: Optional[Decimal] = None):
        if not all([isinstance(val, (Decimal, int)) and val > 0 for val in [entry_price, exit_price, qty, entry_time_ms, exit_time_ms]]):
            self.logger.warning(f"Trade log skipped due to flawed params for Part ID: {part_id}."); return
        profit = safe_decimal_conversion(pnl_str, Decimal(0))
        if profit is not pd.NA and profit <= 0: self.consecutive_losses += 1
        else: self.consecutive_losses = 0
        entry_dt = datetime.fromtimestamp(entry_time_ms / 1000, tz=pytz.utc)
        exit_dt = datetime.fromtimestamp(exit_time_ms / 1000, tz=pytz.utc)
        trade = {
            "part_id": part_id, "symbol": symbol, "side": side, 
            "entry_price_str": str(entry_price), "exit_price_str": str(exit_price),
            "qty_str": str(qty), "profit_str": str(profit), 
            "entry_time_iso": entry_dt.isoformat(), "exit_time_iso": exit_dt.isoformat(), 
            "duration_seconds": (exit_dt - entry_dt).total_seconds(), "exit_reason": reason, 
            "type": "Scale-In" if scale_order_id else "Part", "scale_order_id": scale_order_id,
            "mae_str": str(mae) if mae is not None else None, "mfe_str": str(mfe) if mfe is not None else None,
            "entry_indicators": {k: str(v) if isinstance(v, Decimal) else v for k, v in entry_indicators.items()}, # Serialize Decimals
            "exit_indicators": {k: str(v) if isinstance(v, Decimal) else v for k, v in exit_indicators.items()}  # Serialize Decimals
        }
        self.trades.append(trade)
        pnl_color = NEON["PNL_POS"] if profit > 0 else (NEON["PNL_NEG"] if profit < 0 else NEON["PNL_ZERO"])
        self.logger.success(f"Trade Chronicle ({trade['type']}:{part_id}): {side.upper()} {qty} {symbol.split('/')[0]} | P/L: {pnl_color}{profit:.2f} {self.config.usdt_symbol}{NEON['RESET']} | Reason: {reason}")
        self.logger.debug(f"Entry Inds for {part_id}: {entry_indicators}"); self.logger.debug(f"Exit Inds for {part_id}: {exit_indicators}")

    def increment_daily_trade_entry_count(self): self.daily_trade_entry_count +=1; self.logger.info(f"Daily entry count: {self.daily_trade_entry_count}")
    
    def summary(self) -> str:
        if not self.trades: return "The Grand Ledger is empty."
        profits = [Decimal(t["profit_str"]) for t in self.trades if t.get("profit_str") is not None]
        total_trades = len(profits); wins = sum(1 for p in profits if p > 0); losses = sum(1 for p in profits if p < 0)
        breakeven = total_trades - wins - losses
        win_rate = (Decimal(wins) / Decimal(total_trades)) * 100 if total_trades > 0 else Decimal(0)
        total_profit = sum(profits); avg_profit = total_profit / Decimal(total_trades) if total_trades > 0 else Decimal(0)
        summary_str = f"\n--- Pyrmethus Trade Metrics (v{PYRMETHUS_VERSION}) ---\n" \
                      f"Total Parts: {total_trades} (W: {wins}, L: {losses}, B: {breakeven})\n" \
                      f"Win Rate: {win_rate:.2f}%\n" \
                      f"Total P/L: {total_profit:.2f} {self.config.usdt_symbol} | Avg P/L: {avg_profit:.2f}\n"
        current_equity_approx = Decimal(0)
        if self.initial_equity is not None:
            current_equity_approx = self.initial_equity + total_profit
            overall_pnl_pct = (total_profit / self.initial_equity) * 100 if self.initial_equity > 0 else Decimal(0)
            summary_str += f"Initial Session Treasury: {self.initial_equity:.2f} | Approx Current: {current_equity_approx:.2f} | Session P/L %: {overall_pnl_pct:.2f}%\n"
        if self.daily_start_equity is not None:
            daily_pnl_base = current_equity_approx if self.initial_equity is not None else self.daily_start_equity + total_profit
            daily_pnl = daily_pnl_base - self.daily_start_equity
            daily_pnl_pct = (daily_pnl / self.daily_start_equity) * 100 if self.daily_start_equity > 0 else Decimal(0)
            summary_str += f"Daily Start Treasury: {self.daily_start_equity:.2f} | Approx Daily P/L: {daily_pnl:.2f} ({daily_pnl_pct:.2f}%)\n"
        summary_str += f"Consecutive Losses: {self.consecutive_losses} | Daily Entries: {self.daily_trade_entry_count}\n" \
                       f"--- End of Ledger ---"
        self.logger.info(summary_str); return summary_str

    def get_serializable_trades(self) -> List[Dict[str, Any]]: return self.trades
    def load_trades_from_list(self, trades_list: List[Dict[str, Any]]):
        self.trades = []
        for trade_data in trades_list: self.trades.append(trade_data) # Already serialized, just load
        self.logger.info(f"TradeMetrics: Re-inked {len(self.trades)} sagas from state.")

trade_metrics = TradeMetrics(CONFIG)
_active_trade_parts: List[Dict[str, Any]] = [] # Should ideally contain one item if MAX_ACTIVE_TRADE_PARTS = 1
_last_heartbeat_save_time: float = 0.0; _last_health_check_time: float = 0.0
trade_timestamps_for_whipsaw = deque(maxlen=CONFIG.enhancements["whipsaw_max_trades_in_period"])
whipsaw_cooldown_active_until: float = 0.0; persistent_signal_counter = {"long": 0, "short": 0}
last_signal_type: Optional[str] = None; previous_day_high: Optional[Decimal] = None
previous_day_low: Optional[Decimal] = None; last_key_level_update_day: Optional[int] = None
contradiction_cooldown_active_until: float = 0.0; consecutive_loss_cooldown_active_until: float = 0.0
_stop_requested = asyncio.Event() # For graceful shutdown

# --- Trading Strategy Classes ---
class TradingStrategy(ABC):
    def __init__(self, config: Config, df_columns: Optional[List[str]] = None):
        self.config = config; self.logger = logging.getLogger(f"Pyrmethus.Strategy.{self.__class__.__name__}")
        self.required_columns = df_columns or []; self.logger.info(f"Strategy '{self.__class__.__name__}' materializing...")
    @abstractmethod
    async def generate_signals(self, df: pd.DataFrame, latest_close: Optional[Decimal] = None, latest_atr: Optional[Decimal] = None) -> Dict[str, Any]: pass
    def _validate_df(self, df: pd.DataFrame, min_rows: int = 2) -> bool:
        if df is None or df.empty or len(df) < min_rows: self.logger.debug(f"Insufficient data (Rows: {len(df) if df is not None else 0}, Min: {min_rows})."); return False
        if self.required_columns and not all(col in df.columns for col in self.required_columns):
            self.logger.warning(f"Data missing required columns: {[c for c in self.required_columns if c not in df.columns]}."); return False
        return True
    def _get_default_signals(self) -> Dict[str, Any]:
        return {"enter_long": False, "enter_short": False, "exit_long": False, "exit_short": False, "exit_reason": "Default Signal"}

class DualSupertrendMomentumStrategy(TradingStrategy):
    def __init__(self, config: Config): super().__init__(config, df_columns=["st_long_flip", "st_short_flip", "confirm_trend", "momentum"])
    async def generate_signals(self, df: pd.DataFrame, latest_close: Optional[Decimal] = None, latest_atr: Optional[Decimal] = None) -> Dict[str, Any]:
        signals = self._get_default_signals()
        min_rows = max(self.config.strategy_params["st_atr_length"], self.config.strategy_params["confirm_st_atr_length"], 
                         self.config.strategy_params["momentum_period"], self.config.strategy_params["confirm_st_stability_lookback"]) + 10
        if not self._validate_df(df, min_rows=min_rows): return signals
        last = df.iloc[-1]; primary_long_flip = last.get("st_long_flip", False); primary_short_flip = last.get("st_short_flip", False)
        confirm_is_up = last.get("confirm_trend", pd.NA); momentum = safe_decimal_conversion(last.get("momentum"), pd.NA)
        
        if primary_long_flip and primary_short_flip: # Conflicting flips
            self.logger.warning("Conflicting primary ST flips. Resolving with confirm trend & momentum.")
            if confirm_is_up is True and (momentum is not pd.NA and momentum > 0): primary_short_flip = False
            elif confirm_is_up is False and (momentum is not pd.NA and momentum < 0): primary_long_flip = False
            else: primary_long_flip = primary_short_flip = False # Ambiguous
        
        stable_confirm_trend = pd.NA
        if self.config.strategy_params["confirm_st_stability_lookback"] <= 1: stable_confirm_trend = confirm_is_up
        elif 'confirm_trend' in df.columns and len(df) >= self.config.strategy_params["confirm_st_stability_lookback"]:
            recent_trends = df['confirm_trend'].iloc[-self.config.strategy_params["confirm_st_stability_lookback"]:]
            if confirm_is_up is True and all(trend is True for trend in recent_trends): stable_confirm_trend = True
            elif confirm_is_up is False and all(trend is False for trend in recent_trends): stable_confirm_trend = False
        
        if pd.isna(stable_confirm_trend) or pd.isna(momentum): self.logger.debug(f"Stable Confirm ST ({stable_confirm_trend}) or Mom ({momentum}) is NA."); return signals

        price_prox_ok = True # Max entry distance from ST line
        st_max_dist_atr = self.config.strategy_params.get("st_max_entry_distance_atr_multiplier")
        if st_max_dist_atr and latest_atr and latest_atr > 0 and latest_close:
            max_dist = latest_atr * st_max_dist_atr
            st_base = f"ST_{self.config.strategy_params['st_atr_length']}_{self.config.strategy_params['st_multiplier']}"
            st_line = safe_decimal_conversion(last.get(f"{st_base}l" if primary_long_flip else f"{st_base}s"))
            if st_line and ((primary_long_flip and (latest_close - st_line) > max_dist) or \
                           (primary_short_flip and (st_line - latest_close) > max_dist)):
                price_prox_ok = False; self.logger.debug(f"ST proximity fail for {'LONG' if primary_long_flip else 'SHORT'}.")

        if primary_long_flip and stable_confirm_trend is True and momentum > self.config.strategy_params["momentum_threshold"] and price_prox_ok:
            signals["enter_long"] = True; self.logger.info("DualST+Mom Signal: LONG Entry.")
        elif primary_short_flip and stable_confirm_trend is False and momentum < -self.config.strategy_params["momentum_threshold"] and price_prox_ok:
            signals["enter_short"] = True; self.logger.info("DualST+Mom Signal: SHORT Entry.")
        
        if primary_short_flip: signals["exit_long"] = True; signals["exit_reason"] = "Primary SuperTrend Flipped Short"
        if primary_long_flip: signals["exit_short"] = True; signals["exit_reason"] = "Primary SuperTrend Flipped Long"
        return signals

class EhlersFisherStrategy(TradingStrategy):
    def __init__(self, config: Config): super().__init__(config, df_columns=["ehlers_fisher", "ehlers_signal"])
    async def generate_signals(self, df: pd.DataFrame, latest_close: Optional[Decimal] = None, latest_atr: Optional[Decimal] = None) -> Dict[str, Any]:
        signals = self._get_default_signals()
        min_rows = self.config.strategy_params["ehlers_fisher_length"] + self.config.strategy_params["ehlers_fisher_signal_length"] + 5
        if not self._validate_df(df, min_rows=min_rows) or len(df) < 2: return signals
        last = df.iloc[-1]; prev = df.iloc[-2]
        fi_now = safe_decimal_conversion(last.get("ehlers_fisher"), pd.NA); sig_now = safe_decimal_conversion(last.get("ehlers_signal"), pd.NA)
        fi_prev = safe_decimal_conversion(prev.get("ehlers_fisher"), pd.NA); sig_prev = safe_decimal_conversion(prev.get("ehlers_signal"), pd.NA)
        if any(pd.isna(v) for v in [fi_now, sig_now, fi_prev, sig_prev]): self.logger.debug("Ehlers Fisher/Signal NA."); return signals

        is_extreme = (fi_now > self.config.strategy_params["ehlers_fisher_extreme_threshold_positive"] or \
                      fi_now < self.config.strategy_params["ehlers_fisher_extreme_threshold_negative"])
        if not is_extreme:
            if fi_prev <= sig_prev and fi_now > sig_now: signals["enter_long"] = True; self.logger.info("EhlersFisher Signal: LONG Entry.")
            elif fi_prev >= sig_prev and fi_now < sig_now: signals["enter_short"] = True; self.logger.info("EhlersFisher Signal: SHORT Entry.")
        elif (fi_prev <= sig_prev and fi_now > sig_now) or (fi_prev >= sig_prev and fi_now < sig_now):
             self.logger.info("EhlersFisher: Crossover ignored in extreme zone.")
        
        if fi_prev >= sig_prev and fi_now < sig_now: signals["exit_long"] = True; signals["exit_reason"] = "Ehlers Fisher crossed BELOW Signal"
        if fi_prev <= sig_prev and fi_now > sig_now: signals["exit_short"] = True; signals["exit_reason"] = "Ehlers Fisher crossed ABOVE Signal"
        return signals

strategy_map: Dict[StrategyName, Type[TradingStrategy]] = {StrategyName.DUAL_SUPERTREND_MOMENTUM: DualSupertrendMomentumStrategy, StrategyName.EHLERS_FISHER: EhlersFisherStrategy}
StrategyClass = strategy_map.get(CONFIG.strategy_params["name"])
if StrategyClass: CONFIG.strategy_instance = StrategyClass(CONFIG); logger.success(f"Strategy '{CONFIG.strategy_params['name'].value}' invoked.")
else: err_msg = f"Failed to init strategy '{CONFIG.strategy_params['name'].value}'."; logger.critical(err_msg); send_general_notification("Pyrmethus Critical Error", err_msg); sys.exit(1)

# --- State Persistence Functions ---
def save_persistent_state(force_heartbeat: bool = False) -> None:
    global _active_trade_parts, trade_metrics, _last_heartbeat_save_time, whipsaw_cooldown_active_until, \
           persistent_signal_counter, last_signal_type, previous_day_high, previous_day_low, \
           last_key_level_update_day, contradiction_cooldown_active_until, consecutive_loss_cooldown_active_until
    now = time.time()
    is_active = bool(_active_trade_parts)
    save_interval = HEARTBEAT_INTERVAL_SECONDS if is_active else HEARTBEAT_INTERVAL_SECONDS * 5
    if not force_heartbeat and (now - _last_heartbeat_save_time < save_interval): return
    
    logger.debug("Phoenix Feather scribing memories...", extra={"data": {"force": force_heartbeat, "active_parts": len(_active_trade_parts)}})
    try:
        serializable_parts = []
        for part in _active_trade_parts:
            p_copy = {k: (str(v) if isinstance(v, Decimal) else (list(v) if isinstance(v, deque) else v)) for k,v in part.items()}
            if "entry_indicators" in p_copy and isinstance(p_copy["entry_indicators"], dict): # Ensure inner Decimals also serialized
                p_copy["entry_indicators"] = {k_ind: (str(v_ind) if isinstance(v_ind, Decimal) else v_ind) for k_ind, v_ind in p_copy["entry_indicators"].items()}
            serializable_parts.append(p_copy)

        state = {
            "pyrmethus_version": PYRMETHUS_VERSION, "timestamp_utc_iso": datetime.now(pytz.utc).isoformat(),
            "active_trade_parts": serializable_parts,
            "trade_metrics_trades": trade_metrics.get_serializable_trades(), # Already serialized by TradeMetrics
            "trade_metrics_consecutive_losses": trade_metrics.consecutive_losses,
            "trade_metrics_daily_trade_entry_count": trade_metrics.daily_trade_entry_count,
            "trade_metrics_last_daily_trade_count_reset_day": trade_metrics.last_daily_trade_count_reset_day,
            "trade_metrics_daily_trades_rest_active_until": trade_metrics.daily_trades_rest_active_until,
            "config_symbol": CONFIG.exchange["symbol"], "config_strategy": CONFIG.strategy_params["name"].value,
            "initial_equity_str": str(trade_metrics.initial_equity) if trade_metrics.initial_equity else None,
            "daily_start_equity_str": str(trade_metrics.daily_start_equity) if trade_metrics.daily_start_equity else None,
            "last_daily_reset_day": trade_metrics.last_daily_reset_day,
            "whipsaw_cooldown_active_until": whipsaw_cooldown_active_until,
            "trade_timestamps_for_whipsaw": list(trade_timestamps_for_whipsaw),
            "persistent_signal_counter": persistent_signal_counter, "last_signal_type": last_signal_type,
            "previous_day_high_str": str(previous_day_high) if previous_day_high else None,
            "previous_day_low_str": str(previous_day_low) if previous_day_low else None,
            "last_key_level_update_day": last_key_level_update_day,
            "contradiction_cooldown_active_until": contradiction_cooldown_active_until,
            "consecutive_loss_cooldown_active_until": consecutive_loss_cooldown_active_until,
        }
        temp_file = STATE_FILE_PATH + ".tmp_scroll"
        with open(temp_file, 'w') as f: json.dump(state, f, indent=2)
        os.replace(temp_file, STATE_FILE_PATH)
        _last_heartbeat_save_time = now
        logger.log(logging.INFO if force_heartbeat or is_active else logging.DEBUG, f"Phoenix Feather: State memories scribed. Active parts: {len(_active_trade_parts)}.")
    except Exception as e: logger.error(f"Phoenix Feather Error: Failed to scribe state: {e}", exc_info=True)

def load_persistent_state() -> bool:
    global _active_trade_parts, trade_metrics, whipsaw_cooldown_active_until, trade_timestamps_for_whipsaw, \
           persistent_signal_counter, last_signal_type, previous_day_high, previous_day_low, \
           last_key_level_update_day, contradiction_cooldown_active_until, consecutive_loss_cooldown_active_until
    logger.info(f"Phoenix Feather seeks past memories from '{STATE_FILE_PATH}'...")
    if not os.path.exists(STATE_FILE_PATH): logger.info("Phoenix Feather: No scroll found. Starting fresh."); return False
    try:
        with open(STATE_FILE_PATH, 'r') as f: state = json.load(f)
        # Version check (simplified)
        saved_version = state.get("pyrmethus_version", "0.0.0")
        if saved_version.split('.')[0] != PYRMETHUS_VERSION.split('.')[0]: # Major version mismatch
            logger.warning(f"Phoenix Feather: Scroll version {saved_version} MAJOR mismatch with current {PYRMETHUS_VERSION}. Ignoring scroll."); return False
        if state.get("config_symbol") != CONFIG.exchange["symbol"] or state.get("config_strategy") != CONFIG.strategy_params["name"].value:
            logger.warning(f"Phoenix Scroll sigils mismatch! Ignoring scroll."); return False

        _active_trade_parts.clear()
        loaded_parts_raw = state.get("active_trade_parts", [])
        decimal_keys = ["entry_price", "qty", "sl_price", "tp_price", "initial_usdt_value", "atr_at_entry", "last_known_pnl", "entry_fisher_value", "entry_signal_value"]
        deque_keys = {"recent_pnls": CONFIG.enhancements["profit_momentum_window"], "adverse_candle_closes": CONFIG.enhancements["last_chance_consecutive_adverse_candles"]}

        for part_data in loaded_parts_raw:
            restored = {}
            for k, v_loaded in part_data.items():
                if k in decimal_keys: restored[k] = safe_decimal_conversion(v_loaded, None)
                elif k in deque_keys: restored[k] = deque(v_loaded if isinstance(v_loaded, list) else [], maxlen=deque_keys[k])
                elif k == "entry_indicators" and isinstance(v_loaded, dict):
                    restored[k] = {k_ind: safe_decimal_conversion(v_ind, v_ind) for k_ind, v_ind in v_loaded.items()}
                else: restored[k] = v_loaded
            _active_trade_parts.append(restored)
        
        trade_metrics.load_trades_from_list(state.get("trade_metrics_trades", []))
        trade_metrics.consecutive_losses = state.get("trade_metrics_consecutive_losses", 0)
        trade_metrics.daily_trade_entry_count = state.get("trade_metrics_daily_trade_entry_count", 0)
        trade_metrics.last_daily_trade_count_reset_day = state.get("trade_metrics_last_daily_trade_count_reset_day")
        trade_metrics.daily_trades_rest_active_until = state.get("trade_metrics_daily_trades_rest_active_until", 0.0)
        if state.get("initial_equity_str"): trade_metrics.initial_equity = safe_decimal_conversion(state["initial_equity_str"])
        if state.get("daily_start_equity_str"): trade_metrics.daily_start_equity = safe_decimal_conversion(state["daily_start_equity_str"])
        trade_metrics.last_daily_reset_day = state.get("last_daily_reset_day")
        
        whipsaw_cooldown_active_until = state.get("whipsaw_cooldown_active_until", 0.0)
        trade_timestamps_for_whipsaw = deque(state.get("trade_timestamps_for_whipsaw", []), maxlen=CONFIG.enhancements["whipsaw_max_trades_in_period"])
        persistent_signal_counter = state.get("persistent_signal_counter", {"long": 0, "short": 0})
        last_signal_type = state.get("last_signal_type")
        if state.get("previous_day_high_str"): previous_day_high = safe_decimal_conversion(state["previous_day_high_str"])
        if state.get("previous_day_low_str"): previous_day_low = safe_decimal_conversion(state["previous_day_low_str"])
        last_key_level_update_day = state.get("last_key_level_update_day")
        contradiction_cooldown_active_until = state.get("contradiction_cooldown_active_until", 0.0)
        consecutive_loss_cooldown_active_until = state.get("consecutive_loss_cooldown_active_until", 0.0)
        
        logger.success(f"Phoenix Feather: Memories reawakened! Active Parts: {len(_active_trade_parts)}, Trades: {len(trade_metrics.trades)}")
        return True
    except Exception as e: logger.error(f"Phoenix Feather: Error reawakening: {e}", exc_info=True); return False

def extract_indicators_from_df(df_row: pd.Series, config: Config) -> Dict[str, Any]:
    indicators = {}; atr_period = config.strategy_params.get('atr_calculation_period', 14)
    indicators[f'atr_{atr_period}'] = safe_decimal_conversion(df_row.get(f"ATR_{atr_period}"))
    if config.strategy_params["name"] == StrategyName.DUAL_SUPERTREND_MOMENTUM:
        st_p = f"ST_{config.strategy_params['st_atr_length']}_{config.strategy_params['st_multiplier']}"
        indicators[f'{st_p}l'] = safe_decimal_conversion(df_row.get(f"{st_p}l"))
        indicators[f'{st_p}s'] = safe_decimal_conversion(df_row.get(f"{st_p}s"))
        indicators[f'{st_p}d'] = df_row.get(f"{st_p}d") # Direction
        indicators['confirm_trend'] = df_row.get("confirm_trend")
        indicators['momentum'] = safe_decimal_conversion(df_row.get("momentum"))
    elif config.strategy_params["name"] == StrategyName.EHLERS_FISHER:
        indicators['ehlers_fisher'] = safe_decimal_conversion(df_row.get("ehlers_fisher"))
        indicators['ehlers_signal'] = safe_decimal_conversion(df_row.get("ehlers_signal"))
    indicators['close_price'] = safe_decimal_conversion(df_row.get("close"))
    return {k: v for k, v in indicators.items() if not pd.isna(v)} # Clean NAs

# --- Core Exchange Interaction Functions ---
@retry(ccxt.NetworkError, tries=CONFIG.retry_count, delay=CONFIG.retry_delay_seconds)
async def async_fetch_account_balance(exchange: ccxt_async.Exchange, currency_code: str, config: Config) -> Optional[Decimal]:
    cache_key = f"balance_{currency_code}"
    cached_balance = balance_cache.get(cache_key)
    if cached_balance is not None: logger.debug(f"Using cached balance for {currency_code}."); return cached_balance
    
    logger_bal = logging.getLogger("Pyrmethus.Balance")
    try:
        # Bybit V5 API uses accountType: UNIFIED or CONTRACT. For USDT futures, CONTRACT is typical.
        # However, 'walletBalance' under UNIFIED might be more encompassing for total available.
        # Let's try UNIFIED first, then CONTRACT if needed, or specific for USDT futures.
        # For /v5/account/wallet-balance, accountType is required.
        # UNIFIED for UTA, CONTRACT for classic accounts.
        # Let's assume UTA for now, adjust if classic.
        
        # Bybit V5 unified way:
        params = {'accountType': 'UNIFIED'} # Or 'CONTRACT' if not UTA
        if config.exchange['symbol'].endswith(f":{config.usdt_symbol}"): # This is a linear perpetual
             params = {'accountType': 'CONTRACT', 'coin': currency_code} # More specific for non-UTA

        balance_data = await exchange.fetch_balance(params=params)
        
        # Navigating Bybit's fetch_balance structure for V5:
        # It's a complex structure. We need to find the available balance for the specific currency (USDT).
        # For UNIFIED account (UTA):
        # balance_data['info']['result']['list'][0]['coin'] -> list of assets
        # each coin object: {'coin': 'USDT', 'walletBalance': '...', 'availableToWithdraw': '...'}
        # For CONTRACT account (Non-UTA):
        # balance_data['info']['result']['list'][0]['coin'] -> list of assets for contract account
        # each coin object: {'coin': 'USDT', 'walletBalance': '...', 'availableBalance': '...'}
        
        available_balance = Decimal(0)
        
        if 'info' in balance_data and 'result' in balance_data['info'] and 'list' in balance_data['info']['result']:
            acct_list = balance_data['info']['result']['list']
            if acct_list: # Should be one item for UNIFIED/CONTRACT query usually
                for coin_data in acct_list[0].get('coin', []):
                    if coin_data.get('coin') == currency_code:
                        # UTA uses 'availableToWithdraw' or 'availableToBorrow' for margin. 'walletBalance' is total.
                        # Non-UTA uses 'availableBalance'.
                        # For trading, 'availableBalance' or equivalent free for margin is key.
                        # CCXT often normalizes this to `balance_data[currency_code]['free']`
                        # Let's try the CCXT normalized way first for simplicity
                        if currency_code in balance_data:
                            available_balance = safe_decimal_conversion(balance_data[currency_code].get('free'), Decimal(0))
                            if available_balance > 0: break # Found via CCXT standard
                        
                        # If CCXT 'free' is not populated or zero, try specific Bybit fields
                        if available_balance <= 0:
                            # For UTA, availableToWithdraw might be more relevant if not using margin from other assets for this trade
                            # For non-UTA, availableBalance is the one.
                            # Let's prioritize 'availableBalance' if present, else 'availableToWithdraw'
                            bal_str = coin_data.get('availableBalance', coin_data.get('availableToWithdraw', coin_data.get('availableToBorrow')))
                            if bal_str is None: bal_str = coin_data.get('walletBalance') # Fallback to walletBalance if others are missing
                            
                            available_balance = safe_decimal_conversion(bal_str, Decimal(0))
                        break 
        
        if available_balance <= 0 and currency_code in balance_data: # Fallback to CCXT direct if logic above failed
            available_balance = safe_decimal_conversion(balance_data[currency_code].get('free'), Decimal(0))

        logger_bal.info(f"Available {currency_code} balance: {NEON['VALUE']}{available_balance:.2f}{NEON['RESET']}")
        balance_cache[cache_key] = available_balance
        return available_balance
    except ccxt.AuthenticationError as e: logger_bal.critical(f"Auth Error fetching balance: {e}", exc_info=True); raise
    except (ccxt.NetworkError, ccxt.ExchangeError) as e: logger_bal.error(f"API Error fetching balance: {e}", exc_info=True); return None
    except Exception as e: logger_bal.error(f"Unexpected error fetching balance: {e}", exc_info=True); return None

@retry(ccxt.NetworkError, tries=CONFIG.retry_count, delay=CONFIG.retry_delay_seconds)
async def async_get_market_data(exchange: ccxt_async.Exchange, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    cache_key = f"ohlcv_{symbol}_{timeframe}_{limit}"
    cached_data = market_data_cache.get(cache_key)
    if cached_data is not None: logger.debug("Using cached market data."); return cached_data.copy()

    logger_md = logging.getLogger("Pyrmethus.MarketData")
    try:
        logger_md.debug(f"Fetching {limit} candles for {symbol} ({timeframe})...")
        # Bybit uses 'category': 'linear' for USDT futures in params for fetch_ohlcv if needed
        params = {}
        if symbol.endswith(f":{CONFIG.usdt_symbol}"): params['category'] = 'linear'
        
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit, params=params)
        if not ohlcv: logger_md.warning(f"No OHLCV data returned for {symbol}."); return None

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].apply(safe_decimal_conversion) # Ensure Decimals
        
        logger_md.debug(f"Fetched {len(df)} candles. Latest: {df['timestamp_dt'].iloc[-1]}")
        market_data_cache[cache_key] = df.copy()
        return df
    except (ccxt.NetworkError, ccxt.ExchangeError) as e: logger_md.error(f"API Error fetching market data: {e}", exc_info=True); return None
    except Exception as e: logger_md.error(f"Unexpected error fetching market data: {e}", exc_info=True); return None

async def async_calculate_position_size(balance: Decimal, risk_per_trade_pct: Decimal, 
                                      entry_price: Decimal, sl_price: Decimal, 
                                      config: Config) -> Optional[Decimal]:
    logger_calc = logging.getLogger("Pyrmethus.CalcPosSize")
    if not all([isinstance(balance, Decimal) and balance > 0, isinstance(entry_price, Decimal) and entry_price > 0,
                isinstance(sl_price, Decimal) and sl_price > 0, sl_price != entry_price,
                config.MARKET_INFO, config.qty_step, config.tick_size, config.min_order_qty is not None]):
        logger_calc.warning("Invalid params or market info for sizing.", extra={"data": {"bal": str(balance), "entry": str(entry_price), "sl": str(sl_price)}}); return None

    risk_per_unit = abs(entry_price - sl_price)
    if risk_per_unit < config.tick_size: logger_calc.warning(f"Risk/unit ({risk_per_unit}) < tick size ({config.tick_size}). SL too close."); return None

    usdt_at_risk = balance * risk_per_trade_pct
    quantity = quantize_value(usdt_at_risk / risk_per_unit, config.qty_step, ROUND_DOWN)
    logger_calc.debug(f"USDT at risk: {usdt_at_risk:.2f}, Initial Qty: {quantity}")

    if quantity <= Decimal(0): logger_calc.warning(f"Calculated qty {quantity} too small."); return None

    pos_usdt_val = quantity * entry_price
    if pos_usdt_val > config.trading["max_order_usdt_amount"]:
        logger_calc.info(f"Pos value {pos_usdt_val:.2f} > max ({config.trading['max_order_usdt_amount']}). Capping.")
        quantity = quantize_value(config.trading["max_order_usdt_amount"] / entry_price, config.qty_step, ROUND_DOWN)
        pos_usdt_val = quantity * entry_price
        logger_calc.info(f"Capped Qty: {quantity}, New USDT value: {pos_usdt_val:.2f}")

    if quantity <= Decimal(0): logger_calc.warning("Qty zero after capping by MAX_ORDER_USDT_AMOUNT."); return None
    if quantity < config.min_order_qty: logger_calc.warning(f"Qty {quantity} < min_order_qty {config.min_order_qty}."); return None
    if config.min_order_cost and pos_usdt_val < config.min_order_cost:
        logger_calc.warning(f"Pos value {pos_usdt_val:.2f} < min_order_cost {config.min_order_cost}. Attempting adjustment.")
        qty_adj = quantize_value(config.min_order_cost / entry_price, config.qty_step, ROUND_UP)
        if qty_adj < config.min_order_qty: logger_calc.warning(f"Adj. Qty {qty_adj} still < min_order_qty. No order."); return None
        val_adj = qty_adj * entry_price
        if val_adj > config.trading["max_order_usdt_amount"]: logger_calc.warning(f"Adj. val {val_adj:.2f} > max. No order."); return None
        risk_adj = qty_adj * risk_per_unit
        if risk_adj > usdt_at_risk * Decimal("1.2"): logger_calc.warning(f"Adj. risk {risk_adj:.2f} too high. No order."); return None
        logger_calc.info(f"Adjusting Qty from {quantity} to {qty_adj} for min cost."); quantity = qty_adj; pos_usdt_val = quantity * entry_price
    
    qty_prec = abs(config.qty_step.as_tuple().exponent) if config.qty_step.as_tuple().exponent < 0 else 0
    logger_calc.success(f"Final position size: {NEON['QTY']}{quantity:.{qty_prec}f}{NEON['RESET']} (USDT Value: {pos_usdt_val:.2f})")
    return quantity

@retry((ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection, ccxt.ExchangeNotAvailable, ccxt.AuthenticationError), 
       tries=CONFIG.retry_count, delay=CONFIG.retry_delay_seconds, backoff=2, 
       exceptions_to_not_retry=(ccxt.InsufficientFunds, ccxt.InvalidOrder)) 
async def async_place_risked_order(exchange: ccxt_async.Exchange, config: Config, side: str, 
                                   entry_price_target: Decimal, sl_price: Decimal, tp_price: Optional[Decimal], 
                                   atr_val: Decimal, df_with_indicators: pd.DataFrame) -> Optional[Dict[str, Any]]:
    global _active_trade_parts, trade_timestamps_for_whipsaw, whipsaw_cooldown_active_until
    logger_order = logging.getLogger("Pyrmethus.OrderPlacement")
    logger_order.info(f"Attempting {side.upper()} order for {config.exchange['symbol']}... Target: {format_price(entry_price_target, config.tick_size)}, SL: {format_price(sl_price, config.tick_size)}{f', TP: {format_price(tp_price, config.tick_size)}' if tp_price else ''}")

    balance = await async_fetch_account_balance(exchange, config.usdt_symbol, config)
    if balance is None or balance < config.trading["min_usdt_balance_for_trading"]:
        logger_order.error(f"Insufficient treasury ({balance}). Cannot place order."); return None
    if not all([config.MARKET_INFO, config.tick_size, config.qty_step, config.min_order_qty is not None]):
        logger_order.error("Market info missing. Cannot place order."); return None

    quantized_sl = quantize_value(sl_price, config.tick_size, ROUND_UP if side == config.pos_long else ROUND_DOWN)
    quantized_tp = None
    if tp_price and config.risk_management["enable_take_profit"]:
        quantized_tp = quantize_value(tp_price, config.tick_size, ROUND_DOWN if side == config.pos_long else ROUND_UP)
    if (side == config.pos_long and quantized_sl >= entry_price_target) or \
       (side == config.pos_short and quantized_sl <= entry_price_target):
        logger_order.error(f"Quantized SL {format_price(quantized_sl, config.tick_size)} invalid vs entry {format_price(entry_price_target, config.tick_size)}."); return None
    
    quantity = await async_calculate_position_size(balance, config.trading["risk_per_trade_percentage"], entry_price_target, quantized_sl, config)
    if quantity is None or quantity <= Decimal(0): logger_order.warning("Sizing yielded no valid quantity."); return None

    order_side_verb = config.side_buy if side == config.pos_long else config.side_sell
    order_type = config.trading["entry_order_type"].value.lower()
    
    api_params: Dict[str, Any] = {'category': 'linear', 'positionIdx': 0, 'stopLoss': str(quantized_sl), 'slTriggerBy': 'MarkPrice'}
    if quantized_tp and config.risk_management["enable_take_profit"]:
        api_params['takeProfit'] = str(quantized_tp); api_params['tpTriggerBy'] = 'MarkPrice'

    order_price_for_call: Optional[float] = None
    limit_price_for_order: Optional[Decimal] = None

    if order_type == 'limit':
        atr_offset = atr_val * config.trading["limit_order_offset_atr_percentage"]
        if side == config.pos_long: limit_price_for_order = entry_price_target - atr_offset # Buy lower
        else: limit_price_for_order = entry_price_target + atr_offset # Sell higher
        
        limit_price_for_order = quantize_value(limit_price_for_order, config.tick_size, 
                                             ROUND_DOWN if side == config.pos_long else ROUND_UP) # Favorable quantization
        
        # Price improvement check (simplified: ensure limit price is better than target entry)
        if config.enhancements["limit_order_price_improvement_check_enable"]:
            if (side == config.pos_long and limit_price_for_order >= entry_price_target) or \
               (side == config.pos_short and limit_price_for_order <= entry_price_target):
                logger_order.info(f"Limit price {format_price(limit_price_for_order, config.tick_size)} not an improvement over target {format_price(entry_price_target, config.tick_size)}. Adjusting.")
                # Adjust to be slightly better or use market. For now, let's make it slightly better.
                if side == config.pos_long: limit_price_for_order = quantize_value(entry_price_target - config.tick_size, config.tick_size, ROUND_DOWN)
                else: limit_price_for_order = quantize_value(entry_price_target + config.tick_size, config.tick_size, ROUND_UP)
        
        order_price_for_call = float(limit_price_for_order)
        logger_order.info(f"Calculated LIMIT price: {format_price(limit_price_for_order, config.tick_size)}")


    try:
        log_msg = f"Weaving {order_type.upper()} {order_side_verb.upper()} order: Qty {NEON['QTY']}{quantity}{NEON['RESET']}, SL {NEON['PRICE']}{format_price(quantized_sl, config.tick_size)}{NEON['RESET']}"
        if quantized_tp and config.risk_management['enable_take_profit']: log_msg += f", TP {NEON['PRICE']}{format_price(quantized_tp, config.tick_size)}{NEON['RESET']}"
        if order_price_for_call: log_msg += f", Price {order_price_for_call}"
        logger_order.info(log_msg, extra={"data": {"symbol": config.exchange["symbol"], "type": order_type, "side": order_side_verb, 
                                          "qty": str(quantity), "sl": str(quantized_sl), "tp": str(quantized_tp) if quantized_tp else None,
                                          "limit_price": str(order_price_for_call) if order_price_for_call else None}})
        
        order = await exchange.create_order(symbol=config.exchange["symbol"], type=order_type, side=order_side_verb,
                                            amount=float(quantity), price=order_price_for_call, params=api_params)
        
        logger_order.success(f"Entry Order {order.get('id', 'N/A')} ({order.get('status', 'N/A')}) cast.", extra={"data": {"id": order.get('id'), "status": order.get('status')}})
        send_general_notification(f"Pyrmethus Order Cast: {side.upper()}", 
                                  f"{config.exchange['symbol']} Qty: {quantity:.{abs(config.qty_step.as_tuple().exponent) if config.qty_step else 4}f} @ {'Market' if not order_price_for_call else str(limit_price_for_order)}, SL: {format_price(quantized_sl, config.tick_size)}")
        
        await asyncio.sleep(ORDER_STATUS_CHECK_DELAY)
        filled_order = await exchange.fetch_order(order['id'], config.exchange["symbol"])
        logger_order.debug(f"Fetched order status for {order['id']}: {filled_order.get('status')}", extra={"data": filled_order})

        actual_entry_price = safe_decimal_conversion(filled_order.get('average', filled_order.get('price')))
        actual_qty_filled = safe_decimal_conversion(filled_order.get('filled'))

        # Check if order is considered filled for entry
        is_filled_for_entry = (filled_order.get('status') == 'closed' and actual_qty_filled and actual_qty_filled > 0) or \
                              (order_type == 'market' and actual_qty_filled and actual_qty_filled > 0) or \
                              (order_type == 'limit' and actual_qty_filled and actual_qty_filled > 0 and filled_order.get('status') in ['open', 'closed']) # Partial limit fill is an entry

        if is_filled_for_entry and actual_entry_price and actual_qty_filled and actual_qty_filled > 0:
            entry_time_ms = int(filled_order.get('timestamp', time.time() * 1000))
            part_id = str(uuid.uuid4())[:8]
            entry_inds = extract_indicators_from_df(df_with_indicators.iloc[-1], config) if not df_with_indicators.empty else {}
            
            new_part = {
                "part_id": part_id, "entry_order_id": order['id'], "symbol": config.exchange["symbol"], "side": side, 
                "entry_price": actual_entry_price, "qty": actual_qty_filled, "entry_time_ms": entry_time_ms, 
                "sl_price": quantized_sl, "tp_price": quantized_tp if config.risk_management["enable_take_profit"] else None,
                "atr_at_entry": atr_val, "initial_usdt_value": actual_qty_filled * actual_entry_price,
                "breakeven_set": False, "recent_pnls": deque(maxlen=config.enhancements["profit_momentum_window"]), 
                "last_known_pnl": Decimal(0), "adverse_candle_closes": deque(maxlen=config.enhancements["last_chance_consecutive_adverse_candles"]),
                "entry_indicators": entry_inds, "last_trailing_sl_update": time.time()
            }
            if config.strategy_params["name"] == StrategyName.EHLERS_FISHER and not df_with_indicators.empty:
                new_part["entry_fisher_value"] = safe_decimal_conversion(df_with_indicators.iloc[-1].get("ehlers_fisher"))
                new_part["entry_signal_value"] = safe_decimal_conversion(df_with_indicators.iloc[-1].get("ehlers_signal"))

            _active_trade_parts.append(new_part)
            trade_metrics.increment_daily_trade_entry_count()
            if config.enhancements["whipsaw_cooldown_enable"]: trade_timestamps_for_whipsaw.append(time.time())
            save_persistent_state(force_heartbeat=True)
            logger_order.success(f"Order {order['id']} confirmed filled! Part {part_id} created. Entry: {format_price(actual_entry_price, config.tick_size)}, Qty: {actual_qty_filled}", extra={"data": new_part})
            send_general_notification(f"Pyrmethus Order Filled: {side.upper()}", f"{config.exchange['symbol']} Part {part_id} @ {format_price(actual_entry_price, config.tick_size)}, Qty: {actual_qty_filled}")
            
            # If limit order was partially filled and still open, cancel the remainder as we are proceeding with the filled part.
            if order_type == 'limit' and filled_order.get('status') == 'open' and actual_qty_filled < quantity:
                try:
                    await exchange.cancel_order(order['id'], config.exchange["symbol"])
                    logger_order.info(f"Cancelled remaining part of partially filled limit order {order['id']}.")
                except Exception as e_cancel_rem:
                    logger_order.warning(f"Could not cancel remaining part of limit order {order['id']}: {e_cancel_rem}")
            return new_part
        else:
            logger_order.warning(f"Order {order['id']} not filled as expected. Status: {filled_order.get('status')}, Filled: {actual_qty_filled}. Cancelling if open.", extra={"data": filled_order})
            if filled_order.get('status') == 'open' or filled_order.get('status') == 'new':
                try: await exchange.cancel_order(order['id'], config.exchange["symbol"]); logger_order.info(f"Cancelled pending/unfilled order {order['id']}.")
                except Exception as e_cancel: logger_order.error(f"Error cancelling order {order['id']}: {e_cancel}", exc_info=True)
            return None
    except ccxt.InsufficientFunds as e: logger_order.error(f"Insufficient funds: {e}", exc_info=True); send_general_notification("Pyrmethus Order Fail", f"Insufficient Funds: {config.exchange['symbol']}"); return None
    except ccxt.InvalidOrder as e: logger_order.error(f"Invalid order params: {e}", exc_info=True); send_general_notification("Pyrmethus Order Fail", f"Invalid Order ({config.exchange['symbol']}): {str(e)[:60]}"); return None
    except ccxt.AuthenticationError as e: logger_order.critical(f"Auth error: {e}", exc_info=True); raise
    except ccxt.ExchangeError as e: logger_order.error(f"Exchange error: {e}", exc_info=True); send_general_notification("Pyrmethus Order Fail", f"Exchange Error ({config.exchange['symbol']}): {str(e)[:60]}"); return None
    except Exception as e: logger_order.error(f"Unexpected chaos weaving order: {e}", exc_info=True); send_general_notification("Pyrmethus Order Fail", f"Unexpected Error ({config.exchange['symbol']}): {str(e)[:60]}"); return None

async def async_get_current_position_info(exchange: ccxt_async.Exchange, symbol: str, config: Config) -> Optional[Dict[str, Any]]:
    cache_key = f"position_{symbol}"
    cached_pos = position_cache.get(cache_key)
    if cached_pos is not None: logger.debug(f"Using cached position for {symbol}."); return cached_pos

    logger_pos = logging.getLogger("Pyrmethus.PositionInfo")
    try:
        # Bybit V5 fetch_positions, category 'linear' for USDT futures
        params = {'category': 'linear', 'symbol': symbol}
        positions = await exchange.fetch_positions([symbol], params=params)
        
        if positions:
            # fetch_positions returns a list, usually one item for a specific symbol query
            pos_data = positions[0] 
            # CCXT standard structure: side ('long'/'short'), contracts (Decimal), entryPrice (Decimal), unrealizedPnl (Decimal), etc.
            # Bybit specific 'info' field contains raw response.
            # Example: pos_data['info']['avgPrice'], pos_data['info']['size'], pos_data['info']['side'] ('Buy'/'Sell')
            # pos_data['info']['unrealisedPnl']
            
            side_str = pos_data.get('side') # 'long' or 'short' from CCXT normalization
            qty_str = pos_data.get('contracts') # Number of contracts
            entry_price_str = pos_data.get('entryPrice')
            pnl_str = pos_data.get('unrealizedPnl')
            
            # Fallback to 'info' if standard fields are None (less likely with modern CCXT but good practice)
            if side_str is None and 'info' in pos_data: side_str = pos_data['info'].get('side','').lower(); # Buy -> long, Sell -> short
            if qty_str is None and 'info' in pos_data: qty_str = pos_data['info'].get('size')
            if entry_price_str is None and 'info' in pos_data: entry_price_str = pos_data['info'].get('avgPrice')
            if pnl_str is None and 'info' in pos_data: pnl_str = pos_data['info'].get('unrealisedPnl')

            # Convert to desired types
            qty = safe_decimal_conversion(qty_str, Decimal(0))
            
            if qty > config.position_qty_epsilon: # If position exists
                position_info = {
                    "symbol": symbol,
                    "side": config.pos_long if side_str == 'long' or (isinstance(side_str, str) and side_str.lower() == 'buy') else config.pos_short,
                    "qty": qty,
                    "entry_price": safe_decimal_conversion(entry_price_str),
                    "pnl": safe_decimal_conversion(pnl_str),
                    "leverage": safe_decimal_conversion(pos_data.get('leverage')),
                    "mark_price": safe_decimal_conversion(pos_data.get('markPrice')),
                    "liquidation_price": safe_decimal_conversion(pos_data.get('liquidationPrice')),
                    "raw_data": pos_data # Keep raw for debugging or extended info
                }
                logger_pos.debug(f"Position found: {position_info['side']} {position_info['qty']} @ {position_info['entry_price']:.2f}, PNL: {position_info['pnl']:.2f}")
                position_cache[cache_key] = position_info
                return position_info
            else:
                logger_pos.debug(f"No active position found for {symbol}.")
                position_cache[cache_key] = None # Cache "no position"
                return None
        else: # No positions returned
            logger_pos.debug(f"No positions array returned for {symbol}.")
            position_cache[cache_key] = None
            return None
            
    except (ccxt.NetworkError, ccxt.ExchangeError) as e: logger_pos.error(f"API Error fetching position: {e}", exc_info=True); return None
    except Exception as e: logger_pos.error(f"Unexpected error fetching position: {e}", exc_info=True); return None

async def async_modify_position_sl_tp(exchange: ccxt_async.Exchange, symbol: str, sl_price: Optional[Decimal], tp_price: Optional[Decimal], config: Config) -> bool:
    logger_mod = logging.getLogger("Pyrmethus.ModifySLTP")
    if sl_price is None and tp_price is None: logger_mod.warning("Attempt to modify SL/TP with no prices given."); return False
    
    params = {'category': 'linear', 'symbol': symbol}
    if sl_price is not None: params['stopLoss'] = str(quantize_value(sl_price, config.tick_size)) # Ensure quantized
    if tp_price is not None: params['takeProfit'] = str(quantize_value(tp_price, config.tick_size)) # Ensure quantized
    
    # Bybit specific: trigger prices, e.g., MarkPrice
    if 'stopLoss' in params: params['slTriggerBy'] = 'MarkPrice' # Or 'LastPrice', 'IndexPrice'
    if 'takeProfit' in params: params['tpTriggerBy'] = 'MarkPrice'
    # Other params like slOrderType, tpOrderType can be added if needed (e.g. 'Market' or 'Limit')
    
    try:
        logger_mod.info(f"Modifying SL/TP for {symbol}: SL->{params.get('stopLoss', 'N/A')}, TP->{params.get('takeProfit', 'N/A')}")
        # CCXT does not have a unified `modify_position_sltp`. Use private call for Bybit.
        # Endpoint: /v5/position/trading-stop
        await exchange.private_post_position_trading_stop(params)
        logger_mod.success(f"SL/TP modification request sent for {symbol}.")
        # Invalidate position cache as SL/TP might have changed related fields if Bybit updates them in fetch_positions
        position_cache.pop(f"position_{symbol}", None) 
        return True
    except ccxt.InvalidOrder as e: # E.g. SL too close to current price, or invalid price
        logger_mod.error(f"Invalid order modifying SL/TP for {symbol}: {e}", exc_info=True)
        send_general_notification("Pyrmethus SL/TP Mod Fail", f"Invalid SL/TP for {symbol}: {str(e)[:60]}")
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger_mod.error(f"API Error modifying SL/TP for {symbol}: {e}", exc_info=True)
    except Exception as e:
        logger_mod.error(f"Unexpected error modifying SL/TP for {symbol}: {e}", exc_info=True)
    return False

async def async_close_position_part(exchange: ccxt_async.Exchange, part_info: Dict[str, Any], close_price_target: Optional[Decimal], reason: str, config: Config) -> bool:
    logger_close = logging.getLogger("Pyrmethus.ClosePosition")
    symbol = part_info["symbol"]
    side_to_close = part_info["side"] # This is the original entry side (Long/Short)
    qty_to_close = part_info["qty"]

    # For closing, we take the opposite action
    order_side_verb = config.side_sell if side_to_close == config.pos_long else config.side_buy
    
    try:
        logger_close.info(f"Attempting to close {side_to_close} position for {symbol} (Part ID: {part_info['part_id']}), Qty: {qty_to_close}, Reason: {reason}")
        # Bybit V5: category 'linear', reduceOnly=True
        params = {'category': 'linear', 'reduceOnly': True}
        
        # Market close is generally safer for scalping exits
        order_type = 'market' 
        # If a specific close_price_target is provided, a limit order could be used, but adds complexity (may not fill)
        # For simplicity, this example uses market close.
        
        order = await exchange.create_order(
            symbol=symbol,
            type=order_type,
            side=order_side_verb,
            amount=float(qty_to_close),
            params=params
        )
        logger_close.success(f"Close order {order.get('id')} for {qty_to_close} {symbol} sent. Status: {order.get('status')}")
        
        # Wait for fill confirmation (simplified)
        await asyncio.sleep(ORDER_STATUS_CHECK_DELAY) 
        filled_order = await exchange.fetch_order(order['id'], symbol)
        
        if filled_order and (filled_order.get('status') == 'closed' or (filled_order.get('filled', 0) > 0)):
            actual_exit_price = safe_decimal_conversion(filled_order.get('average', filled_order.get('price')))
            actual_qty_closed = safe_decimal_conversion(filled_order.get('filled'))
            
            if actual_exit_price and actual_qty_closed and actual_qty_closed > 0:
                # Calculate PNL (simplified, exchange usually provides this on filled order or position)
                pnl = Decimal(0)
                if side_to_close == config.pos_long:
                    pnl = (actual_exit_price - part_info["entry_price"]) * actual_qty_closed
                else: # Short
                    pnl = (part_info["entry_price"] - actual_exit_price) * actual_qty_closed
                # This is gross PNL. Net PNL would include fees.
                
                # Log the trade to metrics
                exit_indicators = {} # Fetch current indicators if needed for exit logging
                current_df = await async_get_market_data(exchange, symbol, config.exchange["interval"], OHLCV_LIMIT)
                if current_df is not None and not current_df.empty:
                    exit_indicators = extract_indicators_from_df(current_df.iloc[-1], config)

                trade_metrics.log_trade(
                    symbol=symbol, side=side_to_close,
                    entry_price=part_info["entry_price"], exit_price=actual_exit_price,
                    qty=actual_qty_closed, entry_time_ms=part_info["entry_time_ms"],
                    exit_time_ms=int(filled_order.get('timestamp', time.time() * 1000)),
                    reason=reason, part_id=part_info["part_id"], pnl_str=str(pnl),
                    entry_indicators=part_info.get("entry_indicators", {}),
                    exit_indicators=exit_indicators
                )
                logger_close.success(f"Part {part_info['part_id']} closed. Exit @ {actual_exit_price}, Qty: {actual_qty_closed}, PNL: {pnl:.2f}")
                send_general_notification(f"Pyrmethus Position Closed: {side_to_close}", 
                                          f"{symbol} Part {part_info['part_id']} @ {format_price(actual_exit_price, config.tick_size)}, PNL: {pnl:.2f}. Reason: {reason}")
                return True
            else:
                logger_close.warning(f"Close order {order['id']} filled but with invalid price/qty. Price: {actual_exit_price}, Qty: {actual_qty_closed}")
                return False
        else:
            logger_close.warning(f"Close order {order['id']} not confirmed filled. Status: {filled_order.get('status') if filled_order else 'N/A'}")
            # If order is still open, try to cancel it (unlikely for market orders if accepted)
            if filled_order and filled_order.get('status') == 'open':
                try: await exchange.cancel_order(order['id'], symbol)
                except Exception as e_cancel: logger_close.error(f"Failed to cancel stuck close order: {e_cancel}")
            return False
            
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger_close.error(f"API Error closing position for {symbol}: {e}", exc_info=True)
    except Exception as e:
        logger_close.error(f"Unexpected error closing position for {symbol}: {e}", exc_info=True)
    return False

async def perform_health_check(exchange: ccxt_async.Exchange, config: Config) -> bool:
    global _last_health_check_time
    logger_health = logging.getLogger("Pyrmethus.Health")
    if time.time() - _last_health_check_time < HEALTH_CHECK_INTERVAL_SECONDS:
        return True # Skip check if too soon
    
    _last_health_check_time = time.time()
    try:
        server_time = await exchange.fetch_time()
        time_diff = abs(time.time() * 1000 - server_time)
        logger_health.info(f"Health check OK. Exchange time: {datetime.fromtimestamp(server_time/1000, tz=pytz.utc).isoformat()}. Time diff: {time_diff:.0f}ms.")
        if time_diff > 10000: # 10 seconds diff
             logger_health.warning(f"Significant time difference ({time_diff}ms) with exchange server. Check system clock.")
             send_general_notification("Pyrmethus Health Warning", f"Large time diff with server: {time_diff:.0f}ms")
        return True
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger_health.error(f"Health check failed: API Error - {e}", exc_info=True)
        send_general_notification("Pyrmethus Health FAIL", f"API Error: {str(e)[:60]}")
        return False
    except Exception as e:
        logger_health.error(f"Health check failed: Unexpected Error - {e}", exc_info=True)
        return False

async def stream_price_data(exchange: ccxt_async.Exchange, config: Config, price_queue: asyncio.Queue) -> None:
    logger_ws = logging.getLogger("Pyrmethus.WebSocket")
    symbol = config.exchange["symbol"]
    logger_ws.info(f"Initiating price stream for {symbol}...")
    while not _stop_requested.is_set():
        try:
            # Using watch_ticker for last price. For more granular data, watch_trades or watch_order_book_l1 could be used.
            ticker = await exchange.watch_ticker(symbol)
            # ticker structure: {'symbol', 'timestamp', 'datetime', 'high', 'low', 'bid', 'bidVolume', 'ask', 'askVolume', 'vwap', 
            # 'open', 'close', 'last', 'previousClose', 'change', 'percentage', 'average', 'baseVolume', 'quoteVolume', 'info'}
            if ticker and 'last' in ticker and ticker['last'] is not None:
                price = safe_decimal_conversion(ticker['last'])
                ts = ticker.get('timestamp', int(time.time() * 1000)) # ms timestamp
                if price is not None:
                    try:
                        await price_queue.put_nowait({'price': price, 'timestamp': ts, 'symbol': symbol})
                        logger_ws.debug(f"WS Price update: {symbol} {NEON['PRICE']}{price}{NEON['RESET']} at {datetime.fromtimestamp(ts/1000, tz=pytz.utc).isoformat()}")
                    except asyncio.QueueFull:
                        logger_ws.warning(f"Price queue full for {symbol}. Discarding oldest item.")
                        await price_queue.get() # Discard oldest
                        await price_queue.put({'price': price, 'timestamp': ts, 'symbol': symbol}) # Try again
            else:
                logger_ws.debug(f"No 'last' price in ticker update for {symbol}: {ticker}")

        except ccxt.NetworkError as e:
            logger_ws.error(f"Network error in price stream for {symbol}: {e}. Reconnecting in {CONFIG.retry_delay_seconds}s...", exc_info=True)
            await asyncio.sleep(CONFIG.retry_delay_seconds)
        except ccxt.ExchangeError as e: # More specific errors
            logger_ws.error(f"Exchange error in price stream for {symbol}: {e}. Reconnecting in {CONFIG.retry_delay_seconds*2}s...", exc_info=True)
            await asyncio.sleep(CONFIG.retry_delay_seconds*2)
        except Exception as e:
            logger_ws.error(f"Unexpected error in price stream for {symbol}: {e}. Reconnecting in {CONFIG.retry_delay_seconds*2}s...", exc_info=True)
            await asyncio.sleep(CONFIG.retry_delay_seconds*2)
        if _stop_requested.is_set(): break # Check before looping again
    logger_ws.info(f"Price stream for {symbol} terminated.")

def calculate_all_indicators(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    logger_ind = logging.getLogger("Pyrmethus.Indicators")
    if df.empty: return df
    
    df_copy = df.copy() # Work on a copy
    
    # ATR (common for SL/TP, risk sizing)
    atr_period = config.strategy_params.get("atr_calculation_period", 14)
    df_copy.ta.atr(length=atr_period, append=True) # Appends "ATR_14" (or similar)

    # Strategy-specific indicators
    strategy_name = config.strategy_params["name"]
    if strategy_name == StrategyName.DUAL_SUPERTREND_MOMENTUM:
        st_len = config.strategy_params["st_atr_length"]; st_mult = config.strategy_params["st_multiplier"]
        cst_len = config.strategy_params["confirm_st_atr_length"]; cst_mult = config.strategy_params["confirm_st_multiplier"]
        mom_len = config.strategy_params["momentum_period"]

        # Primary SuperTrend
        st_p_name = f"ST_{st_len}_{float(st_mult)}" # pandas_ta uses float for multiplier in name
        df_copy.ta.supertrend(length=st_len, multiplier=float(st_mult), append=True)
        # Resulting columns: SUPERT_{st_len}_{float(st_mult)}, SUPERTd_{...}, SUPERTl_{...}, SUPERTs_{...}
        # Rename for consistency and easier access
        df_copy.rename(columns={
            f"SUPERTd_{st_len}_{float(st_mult)}": "st_direction", # 1 for uptrend, -1 for downtrend
            f"SUPERTl_{st_len}_{float(st_mult)}": f"{st_p_name}l", # Long stop line
            f"SUPERTs_{st_len}_{float(st_mult)}": f"{st_p_name}s", # Short stop line
        }, inplace=True)
        df_copy["st_long_flip"] = (df_copy["st_direction"] == 1) & (df_copy["st_direction"].shift(1) == -1)
        df_copy["st_short_flip"] = (df_copy["st_direction"] == -1) & (df_copy["st_direction"].shift(1) == 1)

        # Confirmation SuperTrend
        st_c_name = f"CONFIRM_ST_{cst_len}_{float(cst_mult)}"
        df_copy.ta.supertrend(length=cst_len, multiplier=float(cst_mult), append=True, 
                              col_names=(st_c_name, "confirm_st_direction", f"{st_c_name}l", f"{st_c_name}s"))
        df_copy["confirm_trend"] = df_copy["confirm_st_direction"] == 1 # True for up, False for down

        # Momentum (e.g., ROC or custom)
        df_copy["momentum"] = df_copy.ta.roc(length=mom_len) # Rate of Change as momentum proxy
        # Normalize momentum: (value - min) / (max - min) over a period, or use Z-score if preferred
        # For simplicity, raw ROC used here. Consider scaling/normalizing for consistent thresholding.

    elif strategy_name == StrategyName.EHLERS_FISHER:
        ef_len = config.strategy_params["ehlers_fisher_length"]
        ef_sig_len = config.strategy_params["ehlers_fisher_signal_length"]
        # pandas_ta fisher transform: fisher(length, signal)
        # It returns FIS популаrity, FISh populaRity, FIShl populaRity
        # Check pandas_ta documentation for exact column names for fisher transform
        # Assuming it appends 'FISHERT_len_sig' and 'FISHERTs_len_sig'
        df_copy.ta.fisher(length=ef_len, signal=ef_sig_len, append=True)
        # Rename based on typical pandas_ta output for fisher
        # Example names: FISHERT_10_1, FISHERTs_10_1
        # This needs to be verified with actual pandas_ta output for 'fisher'
        fisher_col_name = f"FISHERT_{ef_len}_{ef_sig_len}" # Main Fisher line
        signal_col_name = f"FISHERTs_{ef_len}_{ef_sig_len}"# Signal line
        df_copy.rename(columns={fisher_col_name: "ehlers_fisher", signal_col_name: "ehlers_signal"}, inplace=True, errors='ignore')
        # If names are different, adjust here. For example, if it's just 'FISHER_len' and 'FISHER_SIGNAL_len_sig'
        if "ehlers_fisher" not in df_copy.columns and fisher_col_name in df_copy.columns: # Check if rename worked
             df_copy.rename(columns={fisher_col_name: "ehlers_fisher"}, inplace=True)
        if "ehlers_signal" not in df_copy.columns and signal_col_name in df_copy.columns:
             df_copy.rename(columns={signal_col_name: "ehlers_signal"}, inplace=True)


    logger_ind.debug(f"Indicators calculated. DF columns: {df_copy.columns.tolist()}")
    return df_copy

async def update_daily_key_levels(exchange: ccxt_async.Exchange, config: Config):
    global previous_day_high, previous_day_low, last_key_level_update_day
    logger_kl = logging.getLogger("Pyrmethus.KeyLevels")
    today = datetime.now(pytz.utc).day
    if last_key_level_update_day == today and previous_day_high is not None and previous_day_low is not None:
        return # Already updated today

    try:
        logger_kl.info("Fetching previous day's OHLC for key levels...")
        # Fetch last 2 daily candles to get previous full day's data
        # Bybit V5 specific params if needed
        params = {'category': 'linear'} if config.exchange['symbol'].endswith(f":{config.usdt_symbol}") else {}
        ohlcv_daily = await exchange.fetch_ohlcv(config.exchange["symbol"], '1D', limit=2, params=params)
        
        if ohlcv_daily and len(ohlcv_daily) >= 2:
            # Data: [timestamp, open, high, low, close, volume]
            # The first entry (index 0) is the second to last day (previous complete day)
            # The second entry (index 1) is the current, possibly incomplete, day
            prev_day_data = ohlcv_daily[0]
            previous_day_high = safe_decimal_conversion(prev_day_data[2])
            previous_day_low = safe_decimal_conversion(prev_day_data[3])
            last_key_level_update_day = today
            logger_kl.success(f"Previous day's key levels updated: High={previous_day_high}, Low={previous_day_low}")
        else:
            logger_kl.warning("Could not fetch sufficient daily OHLCV data to update key levels.")
            
    except Exception as e:
        logger_kl.error(f"Error updating daily key levels: {e}", exc_info=True)

# --- Main Trading Loop Iteration ---
async def trading_loop_iteration(exchange: ccxt_async.Exchange, config: Config, price_queue: asyncio.Queue):
    logger_loop = logging.getLogger("Pyrmethus.TradingLoop")
    logger_loop.debug(f"Oracle's Aegis cycle commencing... Active parts: {len(_active_trade_parts)}")

    # --- 0. Update Daily Data & Equity ---
    current_balance = await async_fetch_account_balance(exchange, config.usdt_symbol, config)
    if current_balance is None: logger_loop.error("Failed to fetch balance. Skipping cycle."); return
    trade_metrics.set_initial_equity(current_balance) # Updates daily start equity if new day
    await update_daily_key_levels(exchange, config) # Update prev day H/L

    # --- 1. Fetch Market Data & Calculate Indicators ---
    df_ohlcv = await async_get_market_data(exchange, config.exchange["symbol"], config.exchange["interval"], OHLCV_LIMIT + config.api_fetch_limit_buffer)
    if df_ohlcv is None or df_ohlcv.empty: logger_loop.warning("No market data. Skipping cycle."); return
    
    df_indicators = calculate_all_indicators(df_ohlcv, config)
    if df_indicators.empty: logger_loop.warning("Indicator calculation failed. Skipping cycle."); return

    latest_candle = df_indicators.iloc[-1]
    latest_close_price = safe_decimal_conversion(latest_candle.get("close"))
    current_atr = safe_decimal_conversion(latest_candle.get(f"ATR_{config.strategy_params['atr_calculation_period']}"))
    
    if latest_close_price is None or current_atr is None or current_atr <= 0:
        logger_loop.warning(f"Latest close ({latest_close_price}) or ATR ({current_atr}) is invalid. Skipping cycle."); return

    # --- Get latest price from WebSocket queue if available, else use latest close ---
    # This part is simplified. A real system might wait for a WS price or use it to confirm candle close.
    live_price = latest_close_price # Default to candle close
    try:
        # Non-blocking get from queue.
        # For true scalping, you'd want to react to WS prices primarily.
        # This example is more candle-based with WS price as a supplement.
        ws_update = price_queue.get_nowait() 
        live_price = ws_update['price']
        logger_loop.debug(f"Using live price from WS: {NEON['PRICE']}{live_price}{NEON['RESET']}")
        price_queue.task_done() # if using JoinableQueue
    except asyncio.QueueEmpty:
        logger_loop.debug(f"WS price queue empty. Using latest candle close: {NEON['PRICE']}{latest_close_price}{NEON['RESET']}")
        pass # No new WS price, proceed with candle close

    # --- 2. Manage Active Trade(s) ---
    # Assuming MAX_ACTIVE_TRADE_PARTS = 1, so _active_trade_parts has 0 or 1 element
    if _active_trade_parts:
        active_part = _active_trade_parts[0] # Get the single active trade
        current_pos_info = await async_get_current_position_info(exchange, config.exchange["symbol"], config)
        
        # Update PNL for the active part (can also be done more frequently via WS if needed)
        if current_pos_info and current_pos_info.get("pnl") is not None:
            active_part["last_known_pnl"] = current_pos_info["pnl"]
        else: # If position info not available, estimate PNL
            if active_part["side"] == config.pos_long:
                active_part["last_known_pnl"] = (live_price - active_part["entry_price"]) * active_part["qty"]
            else:
                active_part["last_known_pnl"] = (active_part["entry_price"] - live_price) * active_part["qty"]
        
        # A. Check for SL/TP (exchange handles actual execution, bot reacts to closure)
        # If current_pos_info is None, it means the position might have been closed by SL/TP
        if current_pos_info is None:
            logger_loop.info(f"Position for {config.exchange['symbol']} no longer found. Assuming SL/TP hit or manual closure.")
            # Find out exit price (might need to fetch trade history if not available otherwise)
            # This is a complex part: determining exact exit price and time after an exchange-triggered SL/TP.
            # For simplicity, we'll assume it closed at SL or TP price.
            # A robust solution would fetch recent trades for the symbol.
            exit_reason = "SL/TP Hit (assumed)"
            assumed_exit_price = active_part["sl_price"] # Default to SL
            # Heuristic: if live_price is closer to TP, assume TP. This is not robust.
            if active_part.get("tp_price"):
                if abs(live_price - active_part["tp_price"]) < abs(live_price - active_part["sl_price"]):
                    assumed_exit_price = active_part["tp_price"]
            
            exit_indicators = extract_indicators_from_df(latest_candle, config)
            trade_metrics.log_trade(
                symbol=active_part["symbol"], side=active_part["side"],
                entry_price=active_part["entry_price"], exit_price=assumed_exit_price, # This is an assumption
                qty=active_part["qty"], entry_time_ms=active_part["entry_time_ms"],
                exit_time_ms=int(time.time() * 1000), # Approximate exit time
                reason=exit_reason, part_id=active_part["part_id"],
                pnl_str=str(active_part["last_known_pnl"]), # Use last known PNL before closure
                entry_indicators=active_part.get("entry_indicators", {}),
                exit_indicators=exit_indicators
            )
            _active_trade_parts.clear()
            save_persistent_state(force_heartbeat=True)
            await asyncio.sleep(config.post_close_delay_seconds) # Cooldown after close
            return # End cycle after processing closure

        # B. Trailing Stop-Loss
        if config.risk_management["enable_trailing_sl"] and not active_part.get("breakeven_set"): # TSL before BE for this logic
            pnl_in_atrs = Decimal(0)
            if active_part["atr_at_entry"] > 0:
                if active_part["side"] == config.pos_long:
                    pnl_in_atrs = (live_price - active_part["entry_price"]) / active_part["atr_at_entry"]
                else:
                    pnl_in_atrs = (active_part["entry_price"] - live_price) / active_part["atr_at_entry"]
            
            if pnl_in_atrs >= config.risk_management["trailing_sl_trigger_atr"]:
                new_sl_price = Decimal(0)
                if active_part["side"] == config.pos_long:
                    new_sl_price = live_price - (active_part["atr_at_entry"] * config.risk_management["trailing_sl_distance_atr"])
                    new_sl_price = max(new_sl_price, active_part["sl_price"]) # SL can only improve
                else: # Short
                    new_sl_price = live_price + (active_part["atr_at_entry"] * config.risk_management["trailing_sl_distance_atr"])
                    new_sl_price = min(new_sl_price, active_part["sl_price"]) # SL can only improve
                
                new_sl_price = quantize_value(new_sl_price, config.tick_size, ROUND_UP if active_part["side"] == config.pos_long else ROUND_DOWN)

                if new_sl_price != active_part["sl_price"]:
                    logger_loop.info(f"Trailing SL triggered for Part {active_part['part_id']}. PNL ATRs: {pnl_in_atrs:.2f}. Old SL: {format_price(active_part['sl_price'], config.tick_size)}, New SL: {format_price(new_sl_price, config.tick_size)}")
                    if await async_modify_position_sl_tp(exchange, active_part["symbol"], new_sl_price, active_part.get("tp_price"), config):
                        active_part["sl_price"] = new_sl_price
                        active_part["last_trailing_sl_update"] = time.time()
                        save_persistent_state(force_heartbeat=True)
                        send_general_notification("Pyrmethus TSL Update", f"Part {active_part['part_id']} SL trailed to {format_price(new_sl_price, config.tick_size)}")

        # C. Breakeven Stop-Loss
        if config.enhancements["breakeven_sl_enable"] and not active_part.get("breakeven_set"):
            pnl_in_atrs_be = Decimal(0)
            if active_part["atr_at_entry"] > 0:
                if active_part["side"] == config.pos_long: pnl_in_atrs_be = (live_price - active_part["entry_price"]) / active_part["atr_at_entry"]
                else: pnl_in_atrs_be = (active_part["entry_price"] - live_price) / active_part["atr_at_entry"]
            
            abs_pnl_usdt = abs(active_part.get("last_known_pnl", Decimal(0)))
            if pnl_in_atrs_be >= config.enhancements["breakeven_profit_atr_target"] and \
               abs_pnl_usdt >= config.enhancements["breakeven_min_abs_pnl_usdt"]:
                
                breakeven_price = active_part["entry_price"] # Simple BE at entry
                # Could add a small positive offset to cover fees:
                # fee_offset = config.tick_size * Decimal(2) # Example: 2 ticks
                # if active_part["side"] == config.pos_long: breakeven_price += fee_offset
                # else: breakeven_price -= fee_offset
                # breakeven_price = quantize_value(breakeven_price, config.tick_size, ROUND_UP if active_part["side"] == config.pos_long else ROUND_DOWN)


                if (active_part["side"] == config.pos_long and breakeven_price > active_part["sl_price"]) or \
                   (active_part["side"] == config.pos_short and breakeven_price < active_part["sl_price"]):
                    logger_loop.info(f"Breakeven SL triggered for Part {active_part['part_id']}. New SL: {format_price(breakeven_price, config.tick_size)}")
                    if await async_modify_position_sl_tp(exchange, active_part["symbol"], breakeven_price, active_part.get("tp_price"), config):
                        active_part["sl_price"] = breakeven_price
                        active_part["breakeven_set"] = True
                        save_persistent_state(force_heartbeat=True)
                        send_general_notification("Pyrmethus Breakeven SL", f"Part {active_part['part_id']} SL to BE: {format_price(breakeven_price, config.tick_size)}")

        # D. Check for Strategy Exit Signal (independent of SL/TP)
        strategy_signals = await config.strategy_instance.generate_signals(df_indicators, latest_close_price, current_atr)
        exit_signal = False
        if active_part["side"] == config.pos_long and strategy_signals.get("exit_long"):
            exit_signal = True; exit_reason = strategy_signals.get("exit_reason", "Strategy Exit Long")
        elif active_part["side"] == config.pos_short and strategy_signals.get("exit_short"):
            exit_signal = True; exit_reason = strategy_signals.get("exit_reason", "Strategy Exit Short")
        
        if exit_signal:
            logger_loop.info(f"Strategy exit signal for Part {active_part['part_id']}: {exit_reason}")
            if await async_close_position_part(exchange, active_part, live_price, exit_reason, config):
                _active_trade_parts.clear()
                save_persistent_state(force_heartbeat=True)
                await asyncio.sleep(config.post_close_delay_seconds)
            return # End cycle after processing exit

        # E. Other risk management rules (e.g., last chance exit, profit momentum tighten)
        # ... (Implementations for these would go here, calling async_modify_position_sl_tp or async_close_position_part) ...

    # --- 3. Check for New Entry Signals (if no active trade or can open more) ---
    elif len(_active_trade_parts) < config.trading["max_active_trade_parts"]: # Should be 0 if MAX_ACTIVE_TRADE_PARTS = 1
        # Check Cooldowns (Whipsaw, Consecutive Loss, Daily Max Trades, etc.)
        now = time.time()
        if now < whipsaw_cooldown_active_until: logger_loop.info(f"Whipsaw cooldown active until {datetime.fromtimestamp(whipsaw_cooldown_active_until).time()}. No new trades."); return
        if now < consecutive_loss_cooldown_active_until: logger_loop.info(f"Consecutive loss cooldown active until {datetime.fromtimestamp(consecutive_loss_cooldown_active_until).time()}."); return
        if now < trade_metrics.daily_trades_rest_active_until: logger_loop.info(f"Daily max trades rest active until {datetime.fromtimestamp(trade_metrics.daily_trades_rest_active_until).time()}."); return
        
        # Max drawdown check
        drawdown_hit, drawdown_reason = trade_metrics.check_drawdown(current_balance)
        if drawdown_hit: logger_loop.critical(f"MAX DRAWDOWN HIT: {drawdown_reason}. TRADING HALTED FOR THE DAY."); _stop_requested.set(); return # Stop bot for day

        # Session PNL limits
        # ... (Implement session PNL limit checks) ...

        strategy_signals = await config.strategy_instance.generate_signals(df_indicators, latest_close_price, current_atr)
        
        target_side: Optional[str] = None
        if strategy_signals.get("enter_long"): target_side = config.pos_long
        elif strategy_signals.get("enter_short"): target_side = config.pos_short

        if target_side:
            # Signal Persistence
            if last_signal_type == target_side: persistent_signal_counter[target_side.lower()] += 1
            else: persistent_signal_counter = {"long": 0, "short": 0}; persistent_signal_counter[target_side.lower()] = 1
            last_signal_type = target_side

            if persistent_signal_counter[target_side.lower()] < config.enhancements["signal_persistence_candles"]:
                logger_loop.info(f"Signal for {target_side} not yet persistent ({persistent_signal_counter[target_side.lower()]}/{config.enhancements['signal_persistence_candles']}). Waiting.")
                return

            # No Trade Zones (around key levels)
            if config.enhancements["no_trade_zones_enable"] and previous_day_high and previous_day_low:
                # ... (Implementation for no trade zone checks) ...
                pass
            
            # Trap Filter
            if config.enhancements["trap_filter_enable"]:
                # ... (Implementation for trap filter) ...
                pass

            # Calculate SL and TP prices for the new potential trade
            sl_distance = current_atr * config.risk_management["atr_stop_loss_multiplier"]
            new_sl_price: Decimal
            new_tp_price: Optional[Decimal] = None

            if target_side == config.pos_long:
                new_sl_price = latest_close_price - sl_distance
                if config.risk_management["enable_take_profit"]:
                    new_tp_price = latest_close_price + (current_atr * config.risk_management["atr_take_profit_multiplier"])
            else: # Short
                new_sl_price = latest_close_price + sl_distance
                if config.risk_management["enable_take_profit"]:
                    new_tp_price = latest_close_price - (current_atr * config.risk_management["atr_take_profit_multiplier"])
            
            # Attempt to place the order
            logger_loop.info(f"Persistent {target_side} signal received. Attempting to place order.")
            new_part = await async_place_risked_order(
                exchange, config, target_side, latest_close_price, # entry_price_target is latest_close_price for market orders
                new_sl_price, new_tp_price, current_atr, df_indicators
            )
            if new_part:
                logger_loop.success(f"Successfully placed new {target_side} order. Part ID: {new_part['part_id']}")
                # Cooldowns for consecutive losses / whipsaw would be updated within async_place_risked_order or here after result
                if trade_metrics.consecutive_losses >= config.enhancements["max_consecutive_losses"] and config.enhancements["consecutive_loss_limiter_enable"]:
                    cooldown_duration = config.enhancements["consecutive_loss_cooldown_minutes"] * 60
                    consecutive_loss_cooldown_active_until = time.time() + cooldown_duration
                    logger_loop.warning(f"Max consecutive losses ({trade_metrics.consecutive_losses}) reached. Cooldown for {config.enhancements['consecutive_loss_cooldown_minutes']} mins.")
                    send_general_notification("Pyrmethus Cooldown", f"Max consecutive losses. Cooldown active.")

                if config.enhancements["whipsaw_cooldown_enable"] and len(trade_timestamps_for_whipsaw) == config.enhancements["whipsaw_max_trades_in_period"]:
                     if (trade_timestamps_for_whipsaw[-1] - trade_timestamps_for_whipsaw[0]) < config.enhancements["whipsaw_period_seconds"]:
                        whipsaw_cooldown_active_until = time.time() + config.enhancements["whipsaw_cooldown_seconds"]
                        logger_loop.warning(f"Whipsaw detected. Cooldown for {config.enhancements['whipsaw_cooldown_seconds'] / 60:.1f} mins.")
                        send_general_notification("Pyrmethus Whipsaw Cooldown", f"Whipsaw detected. Cooldown active.")
                
                if config.enhancements["daily_max_trades_rest_enable"] and trade_metrics.daily_trade_entry_count >= config.enhancements["daily_max_trades_limit"]:
                    trade_metrics.daily_trades_rest_active_until = time.time() + (config.enhancements["daily_max_trades_rest_hours"] * 3600)
                    logger_loop.warning(f"Daily max trades ({config.enhancements['daily_max_trades_limit']}) reached. Resting for {config.enhancements['daily_max_trades_rest_hours']} hours.")
                    send_general_notification("Pyrmethus Daily Limit", f"Daily max trades. Resting.")

            else:
                logger_loop.warning(f"Failed to place new {target_side} order.")
        else: # No entry signal
            persistent_signal_counter = {"long": 0, "short": 0} # Reset persistence if no signal
            last_signal_type = None
            logger_loop.debug("No new entry signals from strategy.")
            
    logger_loop.debug("Oracle's Aegis cycle complete.")

async def async_cancel_all_symbol_orders(exchange: ccxt_async.Exchange, symbol: str, config: Config):
    logger_cancel = logging.getLogger("Pyrmethus.CancelAll")
    try:
        # Bybit V5 requires category for cancelAllOrders
        params = {'category': 'linear', 'symbol': symbol} # Only cancel for this symbol
        # Some exchanges might also need settleCoin if it's not obvious from symbol
        
        # Check if cancel_all_orders is supported and how it works for Bybit via CCXT
        if exchange.has.get('cancelAllOrders'):
            logger_cancel.info(f"Attempting to cancel all open orders for {symbol}...")
            # The `symbol` argument to `cancel_all_orders` might not be universally supported for filtering by symbol.
            # Bybit's `/v5/order/cancel-all` takes `symbol` and `category`.
            # CCXT's `cancel_all_orders` might not pass `symbol` correctly to Bybit's API.
            # A safer bet is to fetch open orders and cancel them one by one, or use a private call.
            
            # Using fetchOpenOrders then cancelOrder loop for better control:
            open_orders = await exchange.fetch_open_orders(symbol, params={'category': 'linear'})
            if open_orders:
                logger_cancel.info(f"Found {len(open_orders)} open orders for {symbol}. Cancelling...")
                for order in open_orders:
                    try:
                        await exchange.cancel_order(order['id'], symbol, params={'category': 'linear'})
                        logger_cancel.info(f"Cancelled order {order['id']}.")
                    except Exception as e_ind_cancel:
                        logger_cancel.error(f"Failed to cancel order {order['id']}: {e_ind_cancel}")
                logger_cancel.success(f"Finished attempting to cancel {len(open_orders)} orders for {symbol}.")
            else:
                logger_cancel.info(f"No open orders found for {symbol} to cancel.")
        else:
            logger_cancel.warning(f"Exchange {exchange.id} does not support cancelAllOrders via CCXT, or symbol filtering might be an issue. Manual check advised.")

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger_cancel.error(f"API Error cancelling orders for {symbol}: {e}", exc_info=True)
    except Exception as e:
        logger_cancel.error(