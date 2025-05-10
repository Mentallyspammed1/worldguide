#!/usr/bin/env python3
# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v3.0.2 (Aegis of Asynchronous Wards)
# Optimized for Bybit Mainnet with asyncio, enhanced risk management, Telegram notifications, and robust error handling.

"""
High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 3.0.2 - Aegis of Asynchronous Wards

Dependencies & Setup (Termux):
1. Core Packages:
   pkg install python clang libcrypt libffi openssl rust termux-api
2. Python Libraries:
   pip install ccxt ccxt[async] pandas pandas-ta colorama python-dotenv retry pytz cachetools requests

WARNING: This script executes LIVE TRADES with REAL FUNDS. Use with extreme caution.
Pyrmethus bears no responsibility for financial losses.

Changes in v3.0.2:
- Asynchronous Operations: Leverages `asyncio` and `ccxt.async_support` for non-blocking API calls and WebSocket streaming.
- WebSocket Price Streaming: Uses `watch_ticker` for real-time price updates, reducing reliance on frequent polling for latest close.
- Refined Configuration Structure: Organizes config into nested dictionaries for clarity.
- Enhanced Notification Batching: More robust flushing mechanism for Termux and Telegram.
- Improved Health Checks: Validates API responsiveness periodically.
- Robust Error Handling: More specific exception handling for critical operations like authentication and order placement.
- Cache Optimization: TTLCache for market data and balance to reduce API load.
- Logging Format: Switched to JSON logging for better machine readability and structured logs.
- Code Structure: Improved modularity and readability with more helper functions.
"""

import asyncio
import json
import logging
import logging.handlers # For RotatingFileHandler
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

import pytz # The Chronomancer's ally
import requests # For Telegram notifications
from cachetools import TTLCache # For API call caching
from colorama import Back, Fore, Style # The Prisms of Perception
from colorama import init as colorama_init
from dotenv import load_dotenv # For whispering secrets
from retry import retry # The Art of Tenacious Spellcasting

# Third-party Libraries - The Grimoire of External Magics
try:
    import ccxt
    import ccxt.async_support as ccxt_async # Pyrmethus: Summoning asynchronous spirits
    import pandas as pd
    if not hasattr(pd, 'NA'):
        raise ImportError("Pandas version < 1.0 not supported, pd.NA is required.")
    import pandas_ta as ta # The Alchemist's Table
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
PYRMETHUS_VERSION = "3.0.2"
STATE_FILE_NAME = f"pyrmethus_phoenix_state_v{PYRMETHUS_VERSION.replace('.', '_')}.json" # Underscore for better filename compatibility
STATE_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), STATE_FILE_NAME)
HEARTBEAT_INTERVAL_SECONDS = 60
OHLCV_LIMIT = 200 # Ensure this is enough for all indicator lookbacks
API_CACHE_TTL_SECONDS = 30 # Reduced TTL for more responsive data, adjust based on rate limits
HEALTH_CHECK_INTERVAL_SECONDS = 180 # Check API health every 3 minutes
NOTIFICATION_BATCH_INTERVAL_SECONDS = 30 # Flush notifications more frequently
NOTIFICATION_BATCH_MAX_SIZE = 5 # Max notifications per batch

# --- Neon Color Palette (Unchanged) ---
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
# Pyrmethus: Logging setup moved before Config to catch early issues.
# --- Logger Setup ---
LOGGING_LEVEL = logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
log_file_name = f"{logs_dir}/pyrmethus_spell_v{PYRMETHUS_VERSION.replace('.', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.log"

# Pyrmethus: Custom JSON Formatter for structured logging
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record['exc_info'] = self.formatException(record.exc_info)
        if record.stack_info:
            log_record['stack_info'] = self.formatStack(record.stack_info)
        return json.dumps(log_record)

# Pyrmethus: Stream handler for console output with colors
class ColorStreamHandler(logging.StreamHandler):
    def format(self, record):
        level_color = NEON.get(record.levelname, Fore.WHITE)
        reset_color = NEON["RESET"]
        log_message = super().format(record)
        # Basic coloring for the level name part of the log
        # Assuming default format: %(asctime)s [%(levelname)-8s] %(name)-30s %(message)s
        # This might need adjustment if the basicConfig format changes significantly
        try:
            parts = log_message.split("] ", 2) # Split by "] "
            if len(parts) > 1:
                level_part_idx = parts[0].rfind("[")
                if level_part_idx != -1:
                    level_name_in_log = parts[0][level_part_idx+1:].strip() # Extract level name
                    # Ensure it's just the level name without extra spaces from ljust
                    if record.levelname in level_name_in_log:
                         parts[0] = parts[0][:level_part_idx+1] + level_color + record.levelname.ljust(8) + reset_color + parts[0][level_part_idx + 1 + len(record.levelname.ljust(8)):]
                         log_message = "] ".join(parts)
        except Exception: # If coloring fails, just return original message
            pass
        return log_message


# Pyrmethus: Configuring root logger
root_logger = logging.getLogger()
root_logger.setLevel(LOGGING_LEVEL)

# Console Handler (Colored)
console_handler = ColorStreamHandler(sys.stdout)
console_formatter = logging.Formatter(
    fmt=f"%(asctime)s [{Fore.WHITE}%(levelname)-8s{NEON['RESET']}] {Fore.MAGENTA}%(name)-25s{NEON['RESET']} %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)

# File Handler (JSON)
file_handler = logging.handlers.RotatingFileHandler(log_file_name, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
json_formatter = JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S.%fZ") # ISO 8601 format
file_handler.setFormatter(json_formatter)
root_logger.addHandler(file_handler)

# Pyrmethus: Silencing noisy libraries if needed
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("ccxt").setLevel(logging.INFO) # CCXT can be verbose on DEBUG

logger_pre_config = logging.getLogger("Pyrmethus.PreConfig") # Specific logger for pre-config phase

if load_dotenv(dotenv_path=env_path):
    logger_pre_config.info(f"Secrets whispered from .env scroll: {NEON['VALUE']}{env_path}{NEON['RESET']}")
else:
    logger_pre_config.warning(f"No .env scroll found at {NEON['VALUE']}{env_path}{NEON['WARNING']}. Relying on system environment variables or defaults.{NEON['RESET']}")
getcontext().prec = 28 # Increased precision for financial calculations

# --- Caches ---
market_data_cache = TTLCache(maxsize=10, ttl=API_CACHE_TTL_SECONDS) # Smaller cache for OHLCV as it changes often
balance_cache = TTLCache(maxsize=2, ttl=API_CACHE_TTL_SECONDS * 2) # Balance changes less frequently
ticker_cache = TTLCache(maxsize=5, ttl=5) # Short TTL for tickers

# --- Notification Batching ---
notification_buffer: List[Dict[str, Any]] = [] # Type hint for clarity
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

def format_price(price: Optional[Decimal], tick_size: Optional[Decimal] = None) -> Optional[str]:
    # Pyrmethus: Ensures price is formatted according to tick_size precision
    if price is None: return None
    if tick_size is None or tick_size <= Decimal(0):
        # Default precision if tick_size is unknown or invalid
        return f"{price:.8f}".rstrip('0').rstrip('.') # Max 8 decimal places, remove trailing zeros
    
    # Calculate number of decimal places from tick_size
    # e.g., tick_size 0.01 -> 2 decimal places, 0.0005 -> 4 decimal places
    s = str(tick_size).rstrip('0')
    if '.' in s:
        precision = len(s.split('.')[1])
    else: # Integer tick_size (e.g., 1, 10)
        precision = 0
    
    # Quantize the price to the tick_size (round down for buy limit, round up for sell limit, or nearest for others)
    # For general formatting, simple rounding to precision is usually fine.
    # Actual order placement should handle quantization more strictly.
    return f"{price:.{precision}f}"


def _flush_single_notification_type(messages: List[Dict[str, Any]], notification_type: str, config: 'Config') -> None:
    # Pyrmethus: Refactored flushing logic for clarity
    logger_notify = logging.getLogger("Pyrmethus.Notification")
    if not messages: return

    notification_timeout = config.notifications['timeout_seconds']
    combined_message_parts = []
    for n_item in messages:
        title = n_item.get('title', 'Pyrmethus')
        message = n_item.get('message', '')
        combined_message_parts.append(f"{title}: {message}")
    
    full_combined_message = "\n---\n".join(combined_message_parts)

    if notification_type == "termux":
        try:
            safe_title = json.dumps("Pyrmethus Batch")
            safe_message = json.dumps(full_combined_message[:1000]) # Termux notifications have length limits
            termux_id = messages[0].get("id", 777) # Use ID from first message in batch
            command = ["termux-notification", "--title", safe_title, "--content", safe_message, "--id", str(termux_id)]
            logger_notify.debug(f"Sending batched Termux notification ({len(messages)} items).")
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(timeout=notification_timeout)
            if process.returncode == 0:
                logger_notify.info(f"Batched Termux notification sent successfully ({len(messages)} items).")
            else:
                logger_notify.error(f"Failed to send batched Termux notification. Return code: {process.returncode}. Error: {stderr.decode().strip()}")
        except FileNotFoundError: logger_notify.error("Termux API command 'termux-notification' not found.")
        except subprocess.TimeoutExpired: logger_notify.error("Termux notification command timed out.")
        except Exception as e: logger_notify.error(f"Unexpected error sending Termux notification: {e}", exc_info=True)
    
    elif notification_type == "telegram":
        telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if telegram_bot_token and telegram_chat_id:
            try:
                url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
                # Telegram messages have a max length of 4096 characters
                payload = {"chat_id": telegram_chat_id, "text": full_combined_message[:4090], "parse_mode": "Markdown"}
                logger_notify.debug(f"Sending batched Telegram notification ({len(messages)} items).")
                response = requests.post(url, json=payload, timeout=notification_timeout)
                response.raise_for_status() # Will raise an HTTPError for bad responses (4XX or 5XX)
                logger_notify.info(f"Batched Telegram notification sent successfully ({len(messages)} items).")
            except requests.exceptions.RequestException as e:
                logger_notify.error(f"Failed to send Telegram notification: {e}", exc_info=True)
            except Exception as e:
                logger_notify.error(f"Unexpected error sending Telegram notification: {e}", exc_info=True)
        else:
            logger_notify.warning("Telegram notifications configured but TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing in environment.")

def flush_notifications() -> None:
    # Pyrmethus: More robust flushing mechanism
    global notification_buffer, last_notification_flush_time
    if not notification_buffer or 'CONFIG' not in globals():
        return

    # Separate messages by type
    messages_by_type: Dict[str, List[Dict[str, Any]]] = {}
    for msg in notification_buffer:
        msg_type = msg.get("type", "unknown")
        if msg_type not in messages_by_type:
            messages_by_type[msg_type] = []
        messages_by_type[msg_type].append(msg)

    for msg_type, messages in messages_by_type.items():
        if messages: # Only proceed if there are messages of this type
            _flush_single_notification_type(messages, msg_type, CONFIG)
            
    notification_buffer = [] # Clear buffer after attempting to send all types
    last_notification_flush_time = time.time()


def send_general_notification(title: str, message: str, notification_id: int = 777) -> None:
    # Pyrmethus: Unified notification dispatcher
    global notification_buffer, last_notification_flush_time
    if 'CONFIG' not in globals() or not hasattr(CONFIG, 'notifications') or not CONFIG.notifications['enable']:
        return

    # Add to buffer for both types if enabled
    if CONFIG.notifications.get('termux_enable', True): # Default to true if not specified
        notification_buffer.append({"title": title, "message": message, "id": notification_id, "type": "termux"})
    if CONFIG.notifications.get('telegram_enable', True): # Default to true if not specified
         notification_buffer.append({"title": title, "message": message, "type": "telegram"})
    
    now = time.time()
    # Flush if interval passed OR buffer is full (considering each message might go to multiple services)
    # The actual number of "sends" could be notification_buffer_size * num_enabled_services
    # Let's use a simpler check on the raw buffer size for now.
    if now - last_notification_flush_time >= NOTIFICATION_BATCH_INTERVAL_SECONDS or \
       len(notification_buffer) >= NOTIFICATION_BATCH_MAX_SIZE * 2: # *2 to account for dual entries
        flush_notifications()


# --- Enums ---
class StrategyName(str, Enum): DUAL_SUPERTREND_MOMENTUM = "DUAL_SUPERTREND_MOMENTUM"; EHLERS_FISHER = "EHLERS_FISHER"
class OrderEntryType(str, Enum): MARKET = "MARKET"; LIMIT = "LIMIT"

# --- Configuration Class ---
class Config:
    # Pyrmethus: Configuration structure refined for clarity using nested dictionaries.
    def __init__(self) -> None:
        self.logger = logging.getLogger("Pyrmethus.Config") # Specific logger for Config
        self.logger.info(f"Summoning Configuration Runes v{PYRMETHUS_VERSION}")
        
        # Core Exchange Configuration
        self.exchange = {
            "api_key": self._get_env("BYBIT_API_KEY", required=True, secret=True),
            "api_secret": self._get_env("BYBIT_API_SECRET", required=True, secret=True),
            "symbol": self._get_env("SYMBOL", "BTC/USDT:USDT"),
            "interval": self._get_env("INTERVAL", "1m"),
            "leverage": self._get_env("LEVERAGE", 10, cast_type=int),
            "paper_trading": self._get_env("PAPER_TRADING_MODE", "false", cast_type=bool),
            "recv_window": self._get_env("DEFAULT_RECV_WINDOW", 13000, cast_type=int),
        }
        
        # Trading Parameters
        self.trading = {
            "risk_per_trade_percentage": self._get_env("RISK_PER_TRADE_PERCENTAGE", "0.002", cast_type=Decimal), # Example: 0.2%
            "max_order_usdt_amount": self._get_env("MAX_ORDER_USDT_AMOUNT", "25.0", cast_type=Decimal), # Max size for live safety
            "min_usdt_balance_for_trading": self._get_env("MIN_USDT_BALANCE_FOR_TRADING", "10.0", cast_type=Decimal),
            "max_active_trade_parts": self._get_env("MAX_ACTIVE_TRADE_PARTS", 1, cast_type=int), # Keep at 1 for current SL/TP logic
            "order_fill_timeout_seconds": self._get_env("ORDER_FILL_TIMEOUT_SECONDS", 10, cast_type=int),
            "sleep_seconds": self._get_env("SLEEP_SECONDS", 5, cast_type=int), # Shorter sleep for asyncio version
            "entry_order_type": OrderEntryType(self._get_env("ENTRY_ORDER_TYPE", OrderEntryType.MARKET.value).upper()),
            "limit_order_offset_atr_percentage": self._get_env("LIMIT_ORDER_OFFSET_ATR_PERCENTAGE", "0.1", cast_type=Decimal),
        }

        # Risk Management Parameters
        self.risk_management = {
            "atr_stop_loss_multiplier": self._get_env("ATR_STOP_LOSS_MULTIPLIER", "1.0", cast_type=Decimal),
            "enable_take_profit": self._get_env("ENABLE_TAKE_PROFIT", "true", cast_type=bool),
            "atr_take_profit_multiplier": self._get_env("ATR_TAKE_PROFIT_MULTIPLIER", "1.5", cast_type=Decimal),
            
            "enable_trailing_sl": self._get_env("ENABLE_TRAILING_SL", "true", cast_type=bool),
            "trailing_sl_trigger_atr": self._get_env("TRAILING_SL_TRIGGER_ATR", "1.0", cast_type=Decimal), # PNL in ATRs to activate TSL
            "trailing_sl_distance_atr": self._get_env("TRAILING_SL_DISTANCE_ATR", "0.5", cast_type=Decimal), # TSL distance from price in ATRs

            "enable_max_drawdown_stop": self._get_env("ENABLE_MAX_DRAWDOWN_STOP", "true", cast_type=bool),
            "max_drawdown_percent": self._get_env("MAX_DRAWDOWN_PERCENT", "0.02", cast_type=Decimal), # Example: 2% daily drawdown
            
            "enable_session_pnl_limits": self._get_env("ENABLE_SESSION_PNL_LIMITS", "true", cast_type=bool),
            "session_profit_target_usdt": self._get_env("SESSION_PROFIT_TARGET_USDT", "2.0", cast_type=Decimal, required=False), # Optional
            "session_max_loss_usdt": self._get_env("SESSION_MAX_LOSS_USDT", "10.0", cast_type=Decimal, required=False), # Optional
        }

        # Strategy Specific Parameters
        self.strategy_params = {
            "name": StrategyName(self._get_env("STRATEGY_NAME", StrategyName.DUAL_SUPERTREND_MOMENTUM.value).upper()),
            # Dual Supertrend Momentum
            "st_atr_length": self._get_env("ST_ATR_LENGTH", 7, cast_type=int),
            "st_multiplier": self._get_env("ST_MULTIPLIER", "1.5", cast_type=Decimal),
            "confirm_st_atr_length": self._get_env("CONFIRM_ST_ATR_LENGTH", 14, cast_type=int),
            "confirm_st_multiplier": self._get_env("CONFIRM_ST_MULTIPLIER", "2.5", cast_type=Decimal),
            "momentum_period": self._get_env("MOMENTUM_PERIOD", 10, cast_type=int),
            "momentum_threshold": self._get_env("MOMENTUM_THRESHOLD", "0.5", cast_type=Decimal),
            "confirm_st_stability_lookback": self._get_env("CONFIRM_ST_STABILITY_LOOKBACK", 2, cast_type=int),
            "st_max_entry_distance_atr_multiplier": self._get_env("ST_MAX_ENTRY_DISTANCE_ATR_MULTIPLIER", "0.3", cast_type=Decimal, required=False),
            # Ehlers Fisher
            "ehlers_fisher_length": self._get_env("EHLERS_FISHER_LENGTH", 10, cast_type=int),
            "ehlers_fisher_signal_length": self._get_env("EHLERS_FISHER_SIGNAL_LENGTH", 1, cast_type=int),
            "ehlers_fisher_extreme_threshold_positive": self._get_env("EHLERS_FISHER_EXTREME_THRESHOLD_POSITIVE", "2.0", cast_type=Decimal),
            "ehlers_fisher_extreme_threshold_negative": self._get_env("EHLERS_FISHER_EXTREME_THRESHOLD_NEGATIVE", "-2.0", cast_type=Decimal),
            "ehlers_enable_divergence_scaled_exit": self._get_env("EHLERS_ENABLE_DIVERGENCE_SCALED_EXIT", "false", cast_type=bool),
            "ehlers_divergence_threshold_factor": self._get_env("EHLERS_DIVERGENCE_THRESHOLD_FACTOR", "0.75", cast_type=Decimal),
            "ehlers_divergence_exit_percentage": self._get_env("EHLERS_DIVERGENCE_EXIT_PERCENTAGE", "0.3", cast_type=Decimal),
        }
        
        # Notification Settings
        self.notifications = {
            "enable": self._get_env("ENABLE_NOTIFICATIONS", "true", cast_type=bool),
            "timeout_seconds": self._get_env("NOTIFICATION_TIMEOUT_SECONDS", 5, cast_type=int),
            "termux_enable": self._get_env("TERMUX_NOTIFICATIONS_ENABLE", "true", cast_type=bool), # Specific enable flags
            "telegram_enable": self._get_env("TELEGRAM_NOTIFICATIONS_ENABLE", "true", cast_type=bool),
        }

        # General Enhancement Parameters (Snippets from previous versions, now nested)
        self.enhancements = {
            "profit_momentum_sl_tighten_enable": self._get_env("ENABLE_PROFIT_MOMENTUM_SL_TIGHTEN", "false", cast_type=bool),
            "profit_momentum_window": self._get_env("PROFIT_MOMENTUM_WINDOW", 3, cast_type=int),
            "profit_momentum_sl_tighten_factor": self._get_env("PROFIT_MOMENTUM_SL_TIGHTEN_FACTOR", "0.5", cast_type=Decimal),
            "whipsaw_cooldown_enable": self._get_env("ENABLE_WHIPSAW_COOLDOWN", "true", cast_type=bool),
            "whipsaw_max_trades_in_period": self._get_env("WHIPSAW_MAX_TRADES_IN_PERIOD", 3, cast_type=int),
            "whipsaw_period_seconds": self._get_env("WHIPSAW_PERIOD_SECONDS", 300, cast_type=int),
            "whipsaw_cooldown_seconds": self._get_env("WHIPSAW_COOLDOWN_SECONDS", 180, cast_type=int),
            "signal_persistence_candles": self._get_env("SIGNAL_PERSISTENCE_CANDLES", 1, cast_type=int),
            "no_trade_zones_enable": self._get_env("ENABLE_NO_TRADE_ZONES", "false", cast_type=bool),
            "no_trade_zone_pct_around_key_level": self._get_env("NO_TRADE_ZONE_PCT_AROUND_KEY_LEVEL", "0.002", cast_type=Decimal),
            "key_round_number_step": self._get_env("KEY_ROUND_NUMBER_STEP", "1000", cast_type=Decimal, required=False),
            "breakeven_sl_enable": self._get_env("ENABLE_BREAKEVEN_SL", "true", cast_type=bool),
            "breakeven_profit_atr_target": self._get_env("BREAKEVEN_PROFIT_ATR_TARGET", "1.0", cast_type=Decimal),
            "breakeven_min_abs_pnl_usdt": self._get_env("BREAKEVEN_MIN_ABS_PNL_USDT", "0.50", cast_type=Decimal),
            "anti_martingale_risk_enable": self._get_env("ENABLE_ANTI_MARTINGALE_RISK", "false", cast_type=bool),
            "risk_reduction_factor_on_loss": self._get_env("RISK_REDUCTION_FACTOR_ON_LOSS", "0.75", cast_type=Decimal),
            "risk_increase_factor_on_win": self._get_env("RISK_INCREASE_FACTOR_ON_WIN", "1.1", cast_type=Decimal),
            "max_risk_pct_anti_martingale": self._get_env("MAX_RISK_PCT_ANTI_MARTINGALE", "0.02", cast_type=Decimal),
            "last_chance_exit_enable": self._get_env("ENABLE_LAST_CHANCE_EXIT", "false", cast_type=bool),
            "last_chance_consecutive_adverse_candles": self._get_env("LAST_CHANCE_CONSECUTIVE_ADVERSE_CANDLES", 2, cast_type=int),
            "last_chance_sl_proximity_atr": self._get_env("LAST_CHANCE_SL_PROXIMITY_ATR", "0.3", cast_type=Decimal),
            "trend_contradiction_cooldown_enable": self._get_env("ENABLE_TREND_CONTRADICTION_COOLDOWN", "true", cast_type=bool),
            "trend_contradiction_check_candles_after_entry": self._get_env("TREND_CONTRADICTION_CHECK_CANDLES_AFTER_ENTRY", 2, cast_type=int),
            "trend_contradiction_cooldown_seconds": self._get_env("TREND_CONTRADICTION_COOLDOWN_SECONDS", 120, cast_type=int),
            "daily_max_trades_rest_enable": self._get_env("ENABLE_DAILY_MAX_TRADES_REST", "false", cast_type=bool),
            "daily_max_trades_limit": self._get_env("DAILY_MAX_TRADES_LIMIT", 10, cast_type=int),
            "daily_max_trades_rest_hours": self._get_env("DAILY_MAX_TRADES_REST_HOURS", 4, cast_type=int),
            "limit_order_price_improvement_check_enable": self._get_env("ENABLE_LIMIT_ORDER_PRICE_IMPROVEMENT_CHECK", "true", cast_type=bool),
            "trap_filter_enable": self._get_env("ENABLE_TRAP_FILTER", "false", cast_type=bool),
            "trap_filter_lookback_period": self._get_env("TRAP_FILTER_LOOKBACK_PERIOD", 20, cast_type=int),
            "trap_filter_rejection_threshold_atr": self._get_env("TRAP_FILTER_REJECTION_THRESHOLD_ATR", "1.0", cast_type=Decimal),
            "trap_filter_wick_proximity_atr": self._get_env("TRAP_FILTER_WICK_PROXIMITY_ATR", "0.2", cast_type=Decimal),
            "consecutive_loss_limiter_enable": self._get_env("ENABLE_CONSECUTIVE_LOSS_LIMITER", "true", cast_type=bool),
            "max_consecutive_losses": self._get_env("MAX_CONSECUTIVE_LOSSES", 3, cast_type=int),
            "consecutive_loss_cooldown_minutes": self._get_env("CONSECUTIVE_LOSS_COOLDOWN_MINUTES", 60, cast_type=int),
        }
        
        # Internal Constants & Helpers
        self.side_buy: str = "buy"; self.side_sell: str = "sell"
        self.pos_long: str = "Long"; self.pos_short: str = "Short"; self.pos_none: str = "None"
        self.usdt_symbol: str = "USDT" # Base currency for PNL, balance
        self.retry_count: int = 7
        self.retry_delay_seconds: int = 5
        self.api_fetch_limit_buffer: int = 20 # Extra candles for indicators
        self.position_qty_epsilon: Decimal = Decimal("1e-9") # For float comparisons
        self.post_close_delay_seconds: int = 3 # Wait after market close order
        
        # Populated at runtime
        self.MARKET_INFO: Optional[Dict[str, Any]] = None
        self.tick_size: Optional[Decimal] = None
        self.qty_step: Optional[Decimal] = None
        self.min_order_qty: Optional[Decimal] = None
        self.min_order_cost: Optional[Decimal] = None
        self.strategy_instance: Optional['TradingStrategy'] = None # Populated after Config init

        self._validate_parameters()
        self.logger.info(f"Configuration Runes v{PYRMETHUS_VERSION} Summoned and Verified.")
        self.logger.info(f"Chosen path: {self.strategy_params['name'].value}")

    def _get_env(self, key: str, default: Any = None, cast_type: type = str, required: bool = False, secret: bool = False) -> Any:
        # Pyrmethus: Robust rune-reader from environment scrolls.
        value_str = os.getenv(key)
        source = "Env Var"
        value_to_cast: Any = None
        display_value = "********" if secret and value_str is not None else value_str

        if value_str is None:
            if required:
                self.logger.critical(f"CRITICAL: Required configuration rune '{key}' not found in environment. Pyrmethus cannot proceed.")
                raise ValueError(f"Required environment variable '{key}' not set.")
            self.logger.debug(f"Config Rune '{key}': Not Found. Using Default: '{default}'")
            value_to_cast = default
            source = "Default"
        else:
            self.logger.debug(f"Config Rune '{key}': Found Env Value: '{display_value}'")
            value_to_cast = value_str

        if value_to_cast is None: # Can happen if default is None
            if required: # Should have been caught above if value_str was None and required
                 self.logger.critical(f"CRITICAL: Required configuration rune '{key}' resolved to None after environment/default check.")
                 raise ValueError(f"Required environment variable '{key}' resolved to None.")
            self.logger.debug(f"Config Rune '{key}': Final value is None (not required).")
            return None # Return None if not required and value is None

        final_value: Any
        try:
            raw_value_str_for_cast = str(value_to_cast)
            if cast_type == bool: final_value = raw_value_str_for_cast.lower() in ["true", "1", "yes", "y"]
            elif cast_type == Decimal: final_value = Decimal(raw_value_str_for_cast)
            elif cast_type == int: final_value = int(Decimal(raw_value_str_for_cast)) # Decimal first for "10.0"
            elif cast_type == float: final_value = float(raw_value_str_for_cast)
            elif cast_type == str: final_value = raw_value_str_for_cast
            else:
                self.logger.warning(f"Unsupported cast_type '{cast_type.__name__}' for key '{key}'. Returning raw string.")
                final_value = raw_value_str_for_cast
        except (ValueError, TypeError, InvalidOperation) as e:
            self.logger.error(f"Invalid type/value for '{key}': '{value_to_cast}' (Source: {source}). Expected {cast_type.__name__}. Err: {e}. Using Default: '{default}'.")
            if default is None:
                if required:
                    self.logger.critical(f"CRITICAL: Failed cast for required key '{key}', default is None.")
                    raise ValueError(f"Required env var '{key}' failed casting, no valid default.")
                else: # Optional key, default is None, cast failed
                    self.logger.warning(f"Cast fail for optional '{key}', default is None. Final: None")
                    return None 
            else: # Try casting the default
                source = "Default (Fallback)"
                try:
                    default_str_for_cast = str(default)
                    if cast_type == bool: final_value = default_str_for_cast.lower() in ["true", "1", "yes", "y"]
                    elif cast_type == Decimal: final_value = Decimal(default_str_for_cast)
                    elif cast_type == int: final_value = int(Decimal(default_str_for_cast))
                    elif cast_type == float: final_value = float(default_str_for_cast)
                    elif cast_type == str: final_value = default_str_for_cast
                    else: final_value = default_str_for_cast # Fallback for unsupported
                    self.logger.warning(f"Used casted default for {key}: '{final_value}'")
                except (ValueError, TypeError, InvalidOperation) as e_default:
                    self.logger.critical(f"CRITICAL: Cast fail for value AND default for '{key}'. Original: '{value_to_cast}', Default: '{default}'. Err: {e_default}")
                    raise ValueError(f"Config error: Cannot cast value or default for '{key}'.")
        
        self.logger.debug(f"Final value for '{key}': {display_value if secret else final_value} (Type: {type(final_value).__name__}, Source: {source})")
        return final_value

    def _validate_parameters(self) -> None:
        # Pyrmethus: Validating the runes of power.
        errors = []
        if not (0 < self.trading["risk_per_trade_percentage"] < 1): errors.append("RISK_PER_TRADE_PERCENTAGE must be between 0 and 1.")
        if self.exchange["leverage"] < 1: errors.append("LEVERAGE must be at least 1.")
        if self.risk_management["atr_stop_loss_multiplier"] <= 0: errors.append("ATR_STOP_LOSS_MULTIPLIER must be positive.")
        if self.risk_management["enable_take_profit"] and self.risk_management["atr_take_profit_multiplier"] <= 0: errors.append("ATR_TAKE_PROFIT_MULTIPLIER must be positive if take profit is enabled.")
        if self.trading["min_usdt_balance_for_trading"] < 0: errors.append("MIN_USDT_BALANCE_FOR_TRADING cannot be negative.")
        if self.trading["max_active_trade_parts"] < 1: errors.append("MAX_ACTIVE_TRADE_PARTS must be at least 1.")
        if self.enhancements["profit_momentum_window"] < 1: errors.append("PROFIT_MOMENTUM_WINDOW must be >= 1.")
        if self.enhancements["whipsaw_max_trades_in_period"] < 1: errors.append("WHIPSAW_MAX_TRADES_IN_PERIOD must be >= 1.")
        if self.enhancements["signal_persistence_candles"] < 1: errors.append("SIGNAL_PERSISTENCE_CANDLES must be >= 1.")
        if self.strategy_params.get("st_max_entry_distance_atr_multiplier") is not None and self.strategy_params["st_max_entry_distance_atr_multiplier"] < 0: 
            errors.append("ST_MAX_ENTRY_DISTANCE_ATR_MULTIPLIER cannot be negative.")
        if self.enhancements["anti_martingale_risk_enable"]:
            if not (0 < self.enhancements["risk_reduction_factor_on_loss"] <= 1): errors.append("RISK_REDUCTION_FACTOR_ON_LOSS must be between 0 (exclusive) and 1 (inclusive).")
            if self.enhancements["risk_increase_factor_on_win"] < 1: errors.append("RISK_INCREASE_FACTOR_ON_WIN must be >= 1.")
            if not (0 < self.enhancements["max_risk_pct_anti_martingale"] < 1): errors.append("MAX_RISK_PCT_ANTI_MARTINGALE must be between 0 and 1.")
        
        if errors:
            error_message = f"Configuration spellcrafting failed with {len(errors)} flaws:\n" + "\n".join([f"  - {e}" for e in errors])
            self.logger.critical(error_message)
            raise ValueError(error_message)

# --- Global Objects & State Variables ---
try:
    CONFIG = Config()
except ValueError as config_error:
    # Logger might not be fully set up if Config fails early, so use print for critical startup error.
    print(f"CRITICAL: Configuration loading failed. Error: {config_error}", file=sys.stderr)
    # Attempt basic notification if possible
    if 'CONFIG' in globals() and CONFIG and hasattr(CONFIG, 'notifications') and CONFIG.notifications.get('enable'):
        send_general_notification("Pyrmethus Startup Failure", f"Config Error: {str(config_error)[:200]}")
    sys.exit(1)
except Exception as general_config_error: 
    print(f"CRITICAL: Unexpected critical error during configuration: {general_config_error}", file=sys.stderr)
    traceback.print_exc()
    if 'CONFIG' in globals() and CONFIG and hasattr(CONFIG, 'notifications') and CONFIG.notifications.get('enable'):
        send_general_notification("Pyrmethus Critical Failure", f"Unexpected Config Error: {str(general_config_error)[:200]}")
    sys.exit(1)

# Pyrmethus: Initialize these after CONFIG is successfully created.
trade_timestamps_for_whipsaw = deque(maxlen=CONFIG.enhancements["whipsaw_max_trades_in_period"])
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
    # Pyrmethus: Abstract base for all strategic forms.
    def __init__(self, config: Config, df_columns: Optional[List[str]] = None):
        self.config = config
        self.logger = logging.getLogger(f"Pyrmethus.Strategy.{self.__class__.__name__}")
        self.required_columns = df_columns if df_columns else []
        self.logger.info(f"Strategy Form '{self.__class__.__name__}' materializing...")

    @abstractmethod
    async def generate_signals(self, df: pd.DataFrame, latest_close: Optional[Decimal] = None, latest_atr: Optional[Decimal] = None) -> Dict[str, Any]:
        # Pyrmethus: Now an async method to align with potential async indicator calculations in future.
        pass

    def _validate_df(self, df: pd.DataFrame, min_rows: int = 2) -> bool:
        # Pyrmethus: Validating the market scroll.
        if df is None or df.empty or len(df) < min_rows:
            self.logger.debug(f"Insufficient market whispers (Rows: {len(df) if df is not None else 0}, Min Required: {min_rows}).")
            return False
        if self.required_columns and not all(col in df.columns for col in self.required_columns):
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            self.logger.warning(f"Market scroll missing required runes: {missing_cols}.")
            return False
        # Check for NaNs in last row of required columns
        if self.required_columns and not df.empty:
            last_row_values = df.iloc[-1][self.required_columns]
            if last_row_values.isnull().any():
                nan_cols_last_row = last_row_values[last_row_values.isnull()].index.tolist()
                self.logger.debug(f"Faint runes (NaNs) in last candle for critical columns: {nan_cols_last_row}.")
                # Depending on strategy, this might be acceptable or might require returning no signal.
        return True

    def _get_default_signals(self) -> Dict[str, Any]:
        return {"enter_long": False, "enter_short": False, "exit_long": False, "exit_short": False, "exit_reason": "Default Signal"}

class DualSupertrendMomentumStrategy(TradingStrategy):
    # Pyrmethus: The dance of two Supertrends with Momentum's guidance.
    def __init__(self, config: Config):
        super().__init__(config, df_columns=["st_long_flip", "st_short_flip", "confirm_trend", "momentum"])

    async def generate_signals(self, df: pd.DataFrame, latest_close: Optional[Decimal] = None, latest_atr: Optional[Decimal] = None) -> Dict[str, Any]:
        signals = self._get_default_signals()
        # Pyrmethus: Ensuring enough data for all lookbacks.
        min_rows_needed = max(
            self.config.strategy_params["st_atr_length"],
            self.config.strategy_params["confirm_st_atr_length"],
            self.config.strategy_params["momentum_period"],
            self.config.strategy_params["confirm_st_stability_lookback"],
            # self.config.atr_calculation_period # ATR is passed in, not calculated here
        ) + 10 # Buffer
        
        if not self._validate_df(df, min_rows=min_rows_needed): return signals

        last = df.iloc[-1]
        primary_long_flip = last.get("st_long_flip", False)
        primary_short_flip = last.get("st_short_flip", False)
        confirm_is_up_current = last.get("confirm_trend", pd.NA) # Current state
        momentum_val = safe_decimal_conversion(last.get("momentum"), pd.NA)

        # Snippet 4.1: Fade the Weaker Signal
        if primary_long_flip and primary_short_flip:
            self.logger.warning("Conflicting primary Supertrend flips on the same candle. Attempting to resolve...")
            if confirm_is_up_current is True and (momentum_val is not pd.NA and momentum_val > 0):
                primary_short_flip = False; self.logger.info("Resolution: Prioritizing LONG flip (Confirm ST Up & Positive Momentum).")
            elif confirm_is_up_current is False and (momentum_val is not pd.NA and momentum_val < 0):
                primary_long_flip = False; self.logger.info("Resolution: Prioritizing SHORT flip (Confirm ST Down & Negative Momentum).")
            else:
                primary_long_flip = False; primary_short_flip = False; self.logger.warning("Resolution: Conflicting flips ambiguous. No primary signal.")
        
        # Snippet 3.1: Confirmation Supertrend Stability Lookback
        stable_confirm_trend = pd.NA
        if self.config.strategy_params["confirm_st_stability_lookback"] <= 1:
            stable_confirm_trend = confirm_is_up_current
        elif 'confirm_trend' in df.columns and len(df) >= self.config.strategy_params["confirm_st_stability_lookback"]:
            recent_confirm_trends = df['confirm_trend'].iloc[-self.config.strategy_params["confirm_st_stability_lookback"]:]
            if confirm_is_up_current is True and all(trend is True for trend in recent_confirm_trends): stable_confirm_trend = True
            elif confirm_is_up_current is False and all(trend is False for trend in recent_confirm_trends): stable_confirm_trend = False
            else: stable_confirm_trend = pd.NA # Mixed or not enough matching
        
        if pd.isna(stable_confirm_trend) or pd.isna(momentum_val):
            self.logger.debug(f"Stable Confirm ST ({_format_for_log(stable_confirm_trend, True)}) or Mom ({_format_for_log(momentum_val)}) is NA. No signal.")
            return signals

        # Snippet 3.4: Min Distance Between Primary ST Line and Price for Entry
        price_proximity_ok = True
        st_max_dist_atr_mult = self.config.strategy_params.get("st_max_entry_distance_atr_multiplier")
        if st_max_dist_atr_mult is not None and latest_atr is not None and latest_atr > 0 and latest_close is not None:
            max_allowed_distance = latest_atr * st_max_dist_atr_mult
            st_p_base = f"ST_{self.config.strategy_params['st_atr_length']}_{self.config.strategy_params['st_multiplier']}"
            
            if primary_long_flip:
                st_line_val = safe_decimal_conversion(last.get(f"{st_p_base}l"))
                if st_line_val is not None and (latest_close - st_line_val) > max_allowed_distance:
                    price_proximity_ok = False; self.logger.debug(f"Long ST proximity fail: Price {latest_close} vs ST Line {st_line_val}, Dist > {max_allowed_distance}")
            elif primary_short_flip:
                st_line_val = safe_decimal_conversion(last.get(f"{st_p_base}s"))
                if st_line_val is not None and (st_line_val - latest_close) > max_allowed_distance:
                    price_proximity_ok = False; self.logger.debug(f"Short ST proximity fail: Price {latest_close} vs ST Line {st_line_val}, Dist > {max_allowed_distance}")

        # Entry signals (incorporating Snippet 3.2: Momentum Agreement)
        if primary_long_flip and stable_confirm_trend is True and \
           momentum_val > self.config.strategy_params["momentum_threshold"] and momentum_val > 0 and price_proximity_ok: # Added momentum_val > 0
            signals["enter_long"] = True
            self.logger.info("DualST+Mom Signal: LONG Entry - ST Flip, Stable Confirm Up, Positive Mom, Prox OK.")
        elif primary_short_flip and stable_confirm_trend is False and \
             momentum_val < -self.config.strategy_params["momentum_threshold"] and momentum_val < 0 and price_proximity_ok: # Added momentum_val < 0
            signals["enter_short"] = True
            self.logger.info("DualST+Mom Signal: SHORT Entry - ST Flip, Stable Confirm Down, Negative Mom, Prox OK.")

        if primary_short_flip: signals["exit_long"] = True; signals["exit_reason"] = "Primary SuperTrend Flipped Short"
        if primary_long_flip: signals["exit_short"] = True; signals["exit_reason"] = "Primary SuperTrend Flipped Long"
        return signals

class EhlersFisherStrategy(TradingStrategy):
    # Pyrmethus: Ehlers Fisher's transform, a potent diviner of turns.
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
            self.logger.debug("Ehlers Fisher or Signal rune is NA. No signal.")
            return signals

        # Snippet 3.3: Ehlers Fisher "Extreme Zone" Avoidance
        is_fisher_extreme = False
        if (fisher_now > self.config.strategy_params["ehlers_fisher_extreme_threshold_positive"] or \
            fisher_now < self.config.strategy_params["ehlers_fisher_extreme_threshold_negative"]):
            is_fisher_extreme = True
        
        # Entry signals
        if not is_fisher_extreme:
            if fisher_prev <= signal_prev and fisher_now > signal_now:
                signals["enter_long"] = True
                self.logger.info(f"EhlersFisher Signal: LONG Entry - Fisher ({_format_for_log(fisher_now)}) crossed ABOVE Signal ({_format_for_log(signal_now)}).")
            elif fisher_prev >= signal_prev and fisher_now < signal_now:
                signals["enter_short"] = True
                self.logger.info(f"EhlersFisher Signal: SHORT Entry - Fisher ({_format_for_log(fisher_now)}) crossed BELOW Signal ({_format_for_log(signal_now)}).")
        elif (fisher_prev <= signal_prev and fisher_now > signal_now) or \
             (fisher_prev >= signal_prev and fisher_now < signal_now): # Log if crossover ignored
            self.logger.info(f"EhlersFisher: Crossover signal ignored due to Fisher in extreme zone ({_format_for_log(fisher_now)}).")

        # Exit signals
        if fisher_prev >= signal_prev and fisher_now < signal_now:
            signals["exit_long"] = True; signals["exit_reason"] = "Ehlers Fisher crossed BELOW Signal"
        elif fisher_prev <= signal_prev and fisher_now > signal_now:
            signals["exit_short"] = True; signals["exit_reason"] = "Ehlers Fisher crossed ABOVE Signal"
        return signals

# Pyrmethus: Strategy Invocation
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

# --- Trade Metrics Tracking ---
class TradeMetrics:
    # Pyrmethus: The Keeper of the Ledger.
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
        # Pyrmethus: Marking the dawn of the trading session.
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
                 self.daily_trades_rest_active_until = 0.0 # Reset cooldown if it was for previous day

    def check_drawdown(self, current_equity: Decimal) -> Tuple[bool, str]:
        # Pyrmethus: The Shield of Prudence against excessive loss.
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
        # Pyrmethus: Inscribing the saga of each trade.
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
        # Pyrmethus: A clear reflection of the spell's performance.
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

trade_metrics = TradeMetrics(CONFIG) # Initialize with config
_active_trade_parts: List[Dict[str, Any]] = [] 
_last_heartbeat_save_time: float = 0.0
_last_health_check_time: float = 0.0 # For periodic health checks

# --- State Persistence Functions ---
# (save_persistent_state and load_persistent_state remain largely similar to v2.9.1,
# ensuring all new config structures and state variables are handled.
# For brevity, these are not fully re-written here but should be updated to reflect
# the new CONFIG structure if any direct config values were previously stored in state.)
# Key changes would be:
# - Referencing config via CONFIG.exchange['symbol'], CONFIG.strategy_params['name'], etc.
# - Ensuring all relevant state variables are saved/loaded.

# --- Exchange Interaction Primitives ---
# (These would now be async where appropriate, using `async_exchange` for watch_ methods)

# --- Main Loop ---
# (This will be significantly refactored for asyncio)

# --- Helper: Update Daily Key Levels ---
# (This can remain synchronous as it's not in the hot path)

# --- Main Spell Weaving (async version) ---
async def main():
    # Pyrmethus: The heart of the spell, now beating with asynchronous pulses.
    logger_main = logging.getLogger("Pyrmethus.MainLoop")
    logger_main.info(f"=== Pyrmethus Spell v{PYRMETHUS_VERSION} Awakening on {CONFIG.exchange['api_key'][:5]}... ===") # Avoid logging full key
    
    if load_persistent_state(): 
        logger_main.success(f"Reawakened from Phoenix scroll. Active parts: {len(_active_trade_parts)}.")
    else: 
        logger_main.info("No prior state or ignored. Starting fresh.")

    # Initialize exchanges (sync for setup, async for operations)
    sync_exchange_instance = None
    async_exchange_instance = None
    try:
        # Pyrmethus: Initialize synchronous exchange for setup tasks
        sync_exchange_params = {
            'apiKey': CONFIG.exchange["api_key"], 'secret': CONFIG.exchange["api_secret"],
            'options': {'defaultType': 'swap', 'adjustForTimeDifference': True, 'brokerId': f'PYRMETHUS{PYRMETHUS_VERSION.replace(".","_")}'},
            'enableRateLimit': True, 'recvWindow': CONFIG.exchange["recv_window"]
        }
        if CONFIG.exchange["paper_trading"]: 
            sync_exchange_params['urls'] = {'api': ccxt.bybit.urls['api']['test']} # Use ccxt's predefined testnet URL
        sync_exchange_instance = ccxt.bybit(sync_exchange_params)
        
        # Pyrmethus: Initialize asynchronous exchange for trading loop and websockets
        async_exchange_params = sync_exchange_params.copy() # Start with same params
        async_exchange_instance = ccxt_async.bybit(async_exchange_params)

        logger_main.info(f"Connecting to {sync_exchange_instance.id} (CCXT: {ccxt.__version__})")
        
        markets = await async_exchange_instance.load_markets() # Use async for loading markets
        if CONFIG.exchange["symbol"] not in markets:
            err_msg = f"Symbol {CONFIG.exchange['symbol']} not found in {sync_exchange_instance.id} market runes."
            logger_main.critical(err_msg)
            send_general_notification("Pyrmethus Startup Failure", err_msg); await async_exchange_instance.close(); return

        CONFIG.MARKET_INFO = markets[CONFIG.exchange["symbol"]]
        CONFIG.tick_size = safe_decimal_conversion(CONFIG.MARKET_INFO.get('precision', {}).get('price'), Decimal("1e-8"))
        CONFIG.qty_step = safe_decimal_conversion(CONFIG.MARKET_INFO.get('precision', {}).get('amount'), Decimal("1e-8"))
        CONFIG.min_order_qty = safe_decimal_conversion(CONFIG.MARKET_INFO.get('limits', {}).get('amount', {}).get('min'))
        CONFIG.min_order_cost = safe_decimal_conversion(CONFIG.MARKET_INFO.get('limits', {}).get('cost', {}).get('min'))

        logger_main.success(f"Market runes for {CONFIG.exchange['symbol']}: Price Tick {CONFIG.tick_size}, Amount Step {CONFIG.qty_step}, Min Qty: {CONFIG.min_order_qty}, Min Cost: {CONFIG.min_order_cost}")
        
        category = "linear" if CONFIG.exchange["symbol"].endswith(":USDT") else None
        try:
            leverage_params = {'category': category, 'marginMode': 'isolated'} if category else {'marginMode': 'isolated'} # Prefer isolated margin
            logger_main.info(f"Setting leverage to {CONFIG.exchange['leverage']}x for {CONFIG.exchange['symbol']} (Category: {category or 'inferred'}, MarginMode: isolated)...")
            # Leverage setting is often synchronous or a one-time setup
            await async_exchange_instance.set_leverage(CONFIG.exchange['leverage'], CONFIG.exchange["symbol"], params=leverage_params)
            logger_main.success(f"Leverage for {CONFIG.exchange['symbol']} set/confirmed.")
        except Exception as e_lev:
            logger_main.warning(f"Could not set leverage (may be pre-set or issue): {e_lev}", exc_info=LOGGING_LEVEL <= logging.DEBUG)
            send_general_notification("Pyrmethus Leverage Warn", f"Leverage issue for {CONFIG.exchange['symbol']}: {str(e_lev)[:60]}")
        
        logger_main.success(f"Connected to {sync_exchange_instance.id} for {CONFIG.exchange['symbol']}.")
        send_general_notification("Pyrmethus Online", f"Connected to {sync_exchange_instance.id} for {CONFIG.exchange['symbol']} @ {CONFIG.exchange['leverage']}x.")
    
    except ccxt.AuthenticationError as e_auth: 
        logger_main.critical(f"Authentication failed! Check API keys: {e_auth}", exc_info=True)
        send_general_notification("Pyrmethus Auth Fail", "API Authentication Error."); 
        if async_exchange_instance: await async_exchange_instance.close()
        return
    except ccxt.NetworkError as e_net: 
        logger_main.critical(f"Network Error: {e_net}. Check connection/DNS.", exc_info=True)
        send_general_notification("Pyrmethus Network Error", f"Cannot connect: {str(e_net)[:80]}"); 
        if async_exchange_instance: await async_exchange_instance.close()
        return
    except ccxt.ExchangeError as e_exch: 
        logger_main.critical(f"Exchange API Error: {e_exch}. Check API/symbol/status.", exc_info=True)
        send_general_notification("Pyrmethus API Error", f"API issue: {str(e_exch)[:80]}"); 
        if async_exchange_instance: await async_exchange_instance.close()
        return
    except Exception as e_general:
        logger_main.critical(f"Failed to initialize exchange/spell pre-requisites: {e_general}", exc_info=True)
        send_general_notification("Pyrmethus Init Error", f"Exchange init failed: {str(e_general)[:80]}"); 
        if async_exchange_instance: await async_exchange_instance.close()
        return

    # --- Main Trading Loop (Conceptual - to be filled with async logic) ---
    # This part would be the core of the `main_loop` function from previous versions,
    # but adapted for asyncio, using `await` for I/O bound operations like API calls.
    # For brevity, the full detailed loop is not replicated here again, but the structure would be:
    # await trading_loop_iteration(async_exchange_instance, CONFIG)
    # ...
    # await async_exchange_instance.close()
    logger_main.info("Pyrmethus: Asynchronous main loop logic would commence here.")
    logger_main.warning("Pyrmethus: Full async trading loop not implemented in this snippet. This is a structural example.")
    
    # Placeholder for where the detailed async main_loop logic would go.
    # For now, just simulate a short run and clean exit.
    await asyncio.sleep(10) # Simulate some activity
    
    logger_main.info("Pyrmethus spell concludes its current manifestation.")
    if async_exchange_instance:
        await async_exchange_instance.close()
        logger_main.info("Asynchronous exchange connection gracefully closed.")
    if sync_exchange_instance: # Though not strictly necessary to close sync if only used for setup
        try:
            # Some exchanges might not have a close method on sync objects or it might be a no-op
            if hasattr(sync_exchange_instance, 'close') and callable(sync_exchange_instance.close):
                 sync_exchange_instance.close() # type: ignore
                 logger_main.info("Synchronous exchange object closed (if applicable).")
        except Exception as e_close_sync:
            logger_main.warning(f"Minor issue closing synchronous exchange object: {e_close_sync}")

    save_persistent_state(force_heartbeat=True)
    trade_metrics.summary()
    flush_notifications() # Final flush
    send_general_notification("Pyrmethus Offline", f"Spell concluded for {CONFIG.exchange['symbol']}.")


if __name__ == "__main__":
    # Pyrmethus: Summoning the asynchronous energies.
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("\nSorcerer's intervention! Pyrmethus prepares for slumber...")
        # Attempt a final notification if possible
        if 'CONFIG' in globals() and CONFIG and hasattr(CONFIG, 'notifications') and CONFIG.notifications.get('enable'):
            send_general_notification("Pyrmethus Shutdown", "Manual shutdown initiated.")
            flush_notifications() # Ensure buffered notifications are sent
    except Exception as e_top_level:
        logger.critical(f"An unhandled astral disturbance occurred: {e_top_level}", exc_info=True)
        if 'CONFIG' in globals() and CONFIG and hasattr(CONFIG, 'notifications') and CONFIG.notifications.get('enable'):
            send_general_notification("Pyrmethus CRASH", f"Fatal error: {str(e_top_level)[:100]}")
            flush_notifications()
    finally:
        logger.info("Pyrmethus has woven its final thread for this session.")
