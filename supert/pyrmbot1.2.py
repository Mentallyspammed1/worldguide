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

import pytz # type: ignore[import-untyped]
import requests
from cachetools import TTLCache # type: ignore[import-untyped]
from colorama import Back, Fore, Style # type: ignore[import-untyped]
from colorama import init as colorama_init # type: ignore[import-untyped]
from dotenv import load_dotenv
from retry import retry # type: ignore[import-untyped]

try:
    import ccxt
    import ccxt.async_support as ccxt_async
    import pandas as pd
    if not hasattr(pd, 'NA'): # Ensure pandas version supports pd.NA (Pandas >= 1.0)
        raise ImportError("Pandas version < 1.0 not supported, pd.NA is required.")
    import pandas_ta as ta # type: ignore[import-untyped]
except ImportError as e:
    missing_pkg = getattr(e, 'name', str(e)) # Use str(e) as fallback for name
    sys.stderr.write(
        f"\033[91mCRITICAL ERROR: Missing/Incompatible Essence: '{missing_pkg}'. Pyrmethus cannot weave this spell.\033[0m\n")
    sys.stderr.write(
        f"\033[91mPlease ensure all required libraries (runes) are installed and up to date.\033[0m\n")
    sys.stderr.write(
        f"\033[91mConsult the scrolls (README or comments) for 'pkg install' and 'pip install' incantations.\033[0m\n")
    sys.exit(1)

# --- Constants - The Unchanging Pillars of the Spell ---
PYRMETHUS_VERSION = "3.1.0"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE_NAME_TEMPLATE = "pyrmethus_phoenix_state_v{version}.json"
STATE_FILE_NAME = STATE_FILE_NAME_TEMPLATE.format(version=PYRMETHUS_VERSION.replace('.', '_'))
STATE_FILE_PATH = os.path.join(_SCRIPT_DIR, STATE_FILE_NAME)
HEARTBEAT_INTERVAL_SECONDS = 60  # How often to save state if active
OHLCV_LIMIT = 220  # Sufficient for most common indicators (e.g., 200 for EMA + lookback)
API_CACHE_TTL_SECONDS = 10  # Shorter TTL for live data responsiveness
HEALTH_CHECK_INTERVAL_SECONDS = 90  # Check API health every 1.5 minutes
NOTIFICATION_BATCH_INTERVAL_SECONDS = 15  # Flush notifications frequently
NOTIFICATION_BATCH_MAX_SIZE = 3  # Max notifications per batch before flushing
PRICE_QUEUE_MAX_SIZE = 50  # Max items in the WebSocket price queue
ORDER_STATUS_CHECK_DELAY = 2  # Seconds to wait before checking order status after placement

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
ENV_FILE_PATH = os.path.join(_SCRIPT_DIR, '.env')
getcontext().prec = 28 # Set precision for Decimal operations

# --- Logger Setup ---
LOGGING_LEVEL_STR = os.getenv("DEBUG", "false").lower()
LOGGING_LEVEL = logging.DEBUG if LOGGING_LEVEL_STR == "true" else logging.INFO

LOGS_DIR = os.path.join(_SCRIPT_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE_NAME_TEMPLATE = "pyrmethus_spell_v{version}_{timestamp}.log"
LOG_FILE_NAME = LOG_FILE_NAME_TEMPLATE.format(
    version=PYRMETHUS_VERSION.replace('.', '_'),
    timestamp=time.strftime('%Y%m%d_%H%M%S')
)
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE_NAME)

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, 'data') and record.data: # type: ignore[attr-defined]
             log_record['data'] = record.data # type: ignore[attr-defined]
        if record.exc_info:
            log_record['exc_info'] = self.formatException(record.exc_info)
        return json.dumps(log_record)

class ColorStreamHandler(logging.StreamHandler):
    def format(self, record: logging.LogRecord) -> str:
        log_message = super().format(record)
        level_color = NEON.get(record.levelname, Fore.WHITE)
        # The default formatter structure is: "%(asctime)s [%(levelname)-8s] %(name)-28s %(message)s"
        # We want to color the "[%(levelname)-8s]" part.
        # Example: "2023-01-01 12:00:00 [INFO    ] Pyrmethus.Core Some message"
        # becomes "2023-01-01 12:00:00 [CYANINFO    RESET] Pyrmethus.Core Some message"
        
        # Find the start and end of the levelname placeholder in the formatted string
        # This assumes levelname is enclosed in '[' and ']' and padded.
        try:
            # Locate "[LEVELNAME_PADDED]"
            level_part_start = log_message.find("[")
            level_part_end = log_message.find("] ", level_part_start) # Find "] " after "["
            
            if level_part_start != -1 and level_part_end != -1 and (level_part_end > level_part_start):
                # Extract the original level part including brackets and padding
                original_level_part = log_message[level_part_start : level_part_end+1] # e.g., "[INFO    ]"
                
                # Create the colored level part
                # record.levelname is the raw level name e.g. "INFO"
                # The formatter uses `%(levelname)-8s` so it's left-justified to 8 chars
                colored_level_name = level_color + record.levelname.ljust(8) + NEON['RESET']
                colored_level_part = f"[{colored_level_name}]"

                # Replace only the levelname part if it matches what's expected
                # This check helps prevent accidental coloring if log format changes drastically
                # Check if the content inside brackets (stripped) matches record.levelname
                content_in_brackets = original_level_part[1:-1].strip() # Remove brackets, strip spaces
                if content_in_brackets == record.levelname:
                    log_message = log_message.replace(original_level_part, colored_level_part, 1) # Replace first occurrence
        except Exception:
            pass # If any error in complex coloring, fall back to uncolored but formatted message
        return log_message

root_logger = logging.getLogger()
root_logger.setLevel(LOGGING_LEVEL)

# Console Handler
console_formatter = logging.Formatter(
    fmt=f"%(asctime)s [%(levelname)-8s] {Fore.MAGENTA}%(name)-28s{NEON['RESET']} %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler = ColorStreamHandler(sys.stdout)
console_handler.setFormatter(console_formatter)
if not any(isinstance(h, ColorStreamHandler) for h in root_logger.handlers):
    root_logger.addHandler(console_handler)

# File Handler (JSON)
json_formatter = JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S.%fZ") # ISO 8601 format for JSON logs
file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE_PATH, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
)
file_handler.setFormatter(json_formatter)
if not any(isinstance(h, logging.handlers.RotatingFileHandler) and \
           hasattr(h, 'baseFilename') and \
           os.path.abspath(h.baseFilename) == os.path.abspath(LOG_FILE_PATH) for h in root_logger.handlers):
    root_logger.addHandler(file_handler)

# Silence overly verbose libraries
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("ccxt").setLevel(logging.INFO) # CCXT can be noisy on DEBUG

# Custom SUCCESS log level
SUCCESS_LEVEL = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")
def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)
logging.Logger.success = log_success # type: ignore[attr-defined]

logger = logging.getLogger("Pyrmethus.Core") # Main application logger

# --- Environment Loading ---
logger_pre_config = logging.getLogger("Pyrmethus.PreConfig")
if load_dotenv(dotenv_path=ENV_FILE_PATH):
    logger_pre_config.info(f"Secrets whispered from .env scroll: {NEON['VALUE']}{ENV_FILE_PATH}{NEON['RESET']}")
else:
    logger_pre_config.warning(f"No .env scroll found at {NEON['VALUE']}{ENV_FILE_PATH}{NEON['WARNING']}. Relying on system environment variables or defaults.{NEON['RESET']}")

# --- Caches ---
market_data_cache: TTLCache[str, pd.DataFrame] = TTLCache(maxsize=5, ttl=API_CACHE_TTL_SECONDS)
balance_cache: TTLCache[str, Decimal] = TTLCache(maxsize=1, ttl=API_CACHE_TTL_SECONDS * 2)
ticker_cache: TTLCache[str, Any] = TTLCache(maxsize=5, ttl=2) # Short TTL for ticker data if polled
position_cache: TTLCache[str, Optional[Dict[str, Any]]] = TTLCache(maxsize=5, ttl=API_CACHE_TTL_SECONDS)

# --- Notification Batching ---
notification_buffer: List[Dict[str, Any]] = []
last_notification_flush_time: float = 0.0

# --- Helper Functions ---
def safe_decimal_conversion(value: Any, default_if_error: Any = pd.NA) -> Union[Decimal, Any]:
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
        if value is True: return "Up"
        if value is False: return "Down"
        return "Indeterminate"
    if pd.isna(value) or value is None:
        return "N/A"
    return str(value)

def quantize_value(value: Decimal, step: Decimal, rounding_mode: str = ROUND_DOWN) -> Decimal:
    if not isinstance(step, Decimal) or step <= Decimal(0): # Added type check for step
        logger.warning(f"Invalid step {step} (type: {type(step)}) for quantization. Returning original value {value}.")
        return value
    return (value / step).quantize(Decimal('1'), rounding=rounding_mode) * step

def format_price(price: Optional[Decimal], tick_size: Optional[Decimal] = None) -> Optional[str]:
    if price is None:
        return None
    if tick_size is None or not isinstance(tick_size, Decimal) or tick_size <= Decimal(0): # Added type check for tick_size
        # Default formatting if tick_size is invalid
        return f"{price:.8f}".rstrip('0').rstrip('.')
    
    quantized_price = quantize_value(price, tick_size, ROUND_HALF_UP)
    # Determine precision from tick_size (e.g., 0.01 -> 2 decimal places)
    s = str(tick_size).rstrip('0')
    precision = len(s.split('.')[1]) if '.' in s and len(s.split('.')) > 1 else 0
    return f"{quantized_price:.{precision}f}"

def _flush_single_notification_type(messages: List[Dict[str, Any]], notification_type: str, config_notifications: Dict[str, Any]) -> None:
    logger_notify = logging.getLogger("Pyrmethus.Notification")
    if not messages:
        return

    notification_timeout = config_notifications.get('timeout_seconds', 5) # Default timeout
    combined_message_parts = []
    for n_item in messages:
        title = n_item.get('title', 'Pyrmethus')
        message = n_item.get('message', '')
        if notification_type == "telegram":
            # MarkdownV2 requires escaping _*[]()~`>#+-.={}!|
            # Simplified escaping for common characters:
            escape_chars = r"_*[]()~`>#+-.={}!|" # Corrected: added more chars
            escaped_title = "".join(['\\' + char if char in escape_chars else char for char in title])
            escaped_message = "".join(['\\' + char if char in escape_chars else char for char in message])
            combined_message_parts.append(f"*{escaped_title}*\n{escaped_message}")
        else:  # Termux
            combined_message_parts.append(f"{title}: {message}")
    
    full_combined_message = "\n\n---\n\n".join(combined_message_parts)

    if notification_type == "termux":
        try:
            # Ensure title and message are valid JSON strings for command line
            safe_title = json.dumps("Pyrmethus Batch") 
            safe_message = json.dumps(full_combined_message[:1000]) # Truncate for safety
            termux_id = messages[0].get("id", random.randint(1000, 9999))
            command = ["termux-notification", "--title", safe_title, "--content", safe_message, "--id", str(termux_id)]
            logger_notify.debug(f"Sending batched Termux notification ({len(messages)} items). Command: {' '.join(command)}")
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(timeout=notification_timeout)
            if process.returncode == 0:
                logger_notify.info(f"Batched Termux notification sent successfully ({len(messages)} items).")
            else:
                logger_notify.error(f"Failed to send batched Termux notification. RC: {process.returncode}. Err: {stderr.decode().strip()}", 
                                    extra={"data": {"stderr": stderr.decode(errors='replace').strip(), "stdout": stdout.decode(errors='replace').strip()}})
        except FileNotFoundError:
            logger_notify.error("Termux API command 'termux-notification' not found. Is Termux:API installed and configured?")
        except subprocess.TimeoutExpired:
            logger_notify.error(f"Termux notification command timed out after {notification_timeout} seconds.")
        except Exception as e:
            logger_notify.error(f"Unexpected error sending Termux notification: {e}", exc_info=True)
    
    elif notification_type == "telegram":
        telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if telegram_bot_token and telegram_chat_id:
            try:
                url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
                # Telegram MarkdownV2 message length limit is 4096 bytes.
                # Truncate based on byte length for safety with multi-byte characters.
                max_telegram_bytes = 4000 # Leave some buffer
                
                encoded_message = full_combined_message.encode('utf-8')
                if len(encoded_message) > max_telegram_bytes:
                    # Truncate carefully, trying to preserve whole characters and words
                    truncated_bytes = encoded_message[:max_telegram_bytes]
                    try:
                        truncated_message_str = truncated_bytes.decode('utf-8')
                    except UnicodeDecodeError: # If cut in middle of char, step back
                        truncated_message_str = truncated_bytes[:-1].decode('utf-8', 'ignore')
                    
                    # Try to find last newline to cut cleanly
                    last_newline = truncated_message_str.rfind('\n')
                    if last_newline != -1:
                        full_combined_message = truncated_message_str[:last_newline] + "\n... (message truncated)"
                    else: # No newline, just cut and add suffix
                        full_combined_message = truncated_message_str[:max_telegram_bytes - 30] + "... (message truncated)" # Ensure suffix fits

                payload = {"chat_id": telegram_chat_id, "text": full_combined_message, "parse_mode": "MarkdownV2"}
                logger_notify.debug(f"Sending batched Telegram notification ({len(messages)} items). Length: {len(full_combined_message.encode('utf-8'))} bytes.")
                response = requests.post(url, json=payload, timeout=notification_timeout)
                response.raise_for_status() 
                logger_notify.info(f"Batched Telegram notification sent successfully ({len(messages)} items).")
            except requests.exceptions.HTTPError as e_http:
                err_data = {"status_code": e_http.response.status_code, "response_text": e_http.response.text}
                logger_notify.error(f"Telegram API HTTP error: {e_http}", exc_info=True, extra={"data": err_data})
            except requests.exceptions.RequestException as e_req: 
                logger_notify.error(f"Telegram API Request error: {e_req}", exc_info=True)
            except Exception as e:
                logger_notify.error(f"Unexpected error sending Telegram notification: {e}", exc_info=True)
        else:
            logger_notify.warning("Telegram notifications configured but TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing in environment.")

def flush_notifications() -> None:
    global notification_buffer, last_notification_flush_time
    # Check if CONFIG is defined and has notifications attribute
    if 'CONFIG' not in globals() or not hasattr(CONFIG, 'notifications'):
        if notification_buffer: # Log if buffer has items but no config to process them
            logging.getLogger("Pyrmethus
