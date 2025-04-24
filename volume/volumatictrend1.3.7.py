# pyrmethus_volumatic_bot.py
# Enhanced trading bot incorporating the Volumatic Trend and Pivot Order Block strategy
# with advanced position management (SL/TP, BE, TSL) for Bybit V5 (Linear/Inverse).
# Version 1.3.0: Implemented core strategy logic (VT, Pivots, OBs, Signals),
#               position management (BE, TSL activation), multi-request kline fetching,
#               order cancellation helper, SIGTERM handling, and general refinements.

"""
Pyrmethus Volumatic Bot: A Python Trading Bot for Bybit V5 (v1.3.0)

Implements Volumatic Trend + Pivot Order Block strategy with advanced management.

Key Features:
- Bybit V5 API (Linear/Inverse, Sandbox/Live)
- Volumatic Trend Calculation (EMA/SWMA, ATR Bands, Volume Norm)
- Pivot High/Low Detection (Wicks/Body, Lookbacks)
- Order Block Identification & Management (Active, Violated, Extend, Max Boxes)
- Signal Generation (Trend + OB Proximity/Violation)
- Risk-based Position Sizing (Decimal precision, Market Limits)
- Leverage Setting
- Market Order Execution
- Advanced Position Management:
    - Initial SL/TP (ATR-based)
    - Trailing Stop Loss (TSL) Activation (Percentage-based trigger, Callback rate distance)
    - Break-Even (BE) Stop Adjustment (ATR-based profit target, Tick offset)
- Robust API Interaction (Retries, Error Handling, Bybit Codes)
- Multi-Request Kline Fetching (Handles limits > API max)
- Secure Credentials (.env)
- Flexible Configuration (config.json, Validation, Defaults, Auto-Update)
- Detailed Logging (Neon Console, Rotating File Logs, Redaction)
- Graceful Shutdown (Ctrl+C, SIGTERM)
- Sequential Multi-Pair Processing
"""

# --- Core Libraries ---
import hashlib
import hmac
import json
import logging
import math
import os
import re # Needed for error code parsing
import signal # For SIGTERM handling
import sys
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext, InvalidOperation
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union
# Use zoneinfo for modern timezone handling (requires tzdata package)
try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    print(f"{Fore.YELLOW}Warning: 'zoneinfo' module not found. Falling back to UTC. "
          f"For timezone support, ensure Python 3.9+ and install 'tzdata' (`pip install tzdata`).{Style.RESET_ALL}")
    class ZoneInfo: # type: ignore
        def __init__(self, key: str):
            if key != "UTC":
                 print(f"{Fore.YELLOW}Requested timezone '{key}' unavailable, using UTC.{Style.RESET_ALL}")
            self._key = "UTC"
        def __call__(self, dt: Optional[datetime] = None) -> Optional[datetime]:
            if dt: return dt.replace(tzinfo=timezone.utc)
            return None
        def fromutc(self, dt: datetime) -> datetime:
            return dt.replace(tzinfo=timezone.utc)
        def utcoffset(self, dt: Optional[datetime]) -> Optional[timedelta]:
            return timedelta(0)
        def dst(self, dt: Optional[datetime]) -> Optional[timedelta]:
            return timedelta(0)
        def tzname(self, dt: Optional[datetime]) -> Optional[str]:
            return "UTC"
    class ZoneInfoNotFoundError(Exception): pass


# --- Dependencies (Install via pip) ---
import numpy as np # Requires numpy (pip install numpy)
import pandas as pd # Requires pandas (pip install pandas)
import pandas_ta as ta # Requires pandas_ta (pip install pandas_ta)
import requests # Requires requests (pip install requests)
import ccxt # Requires ccxt (pip install ccxt)
from colorama import Fore, Style, init as colorama_init # Requires colorama (pip install colorama)
from dotenv import load_dotenv # Requires python-dotenv (pip install python-dotenv)

# --- Initialize Environment and Settings ---
getcontext().prec = 28 # Set Decimal precision globally
colorama_init(autoreset=True) # Initialize Colorama
load_dotenv() # Load .env file

# --- Constants ---
BOT_VERSION = "1.3.0" # <<<< Version Updated >>>>

# API Credentials
API_KEY: Optional[str] = os.getenv("BYBIT_API_KEY")
API_SECRET: Optional[str] = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    print(f"{Fore.RED}{Style.BRIGHT}FATAL: BYBIT_API_KEY and BYBIT_API_SECRET must be set in the .env file. The arcane seals are incomplete! Exiting.{Style.RESET_ALL}")
    sys.exit(1)

# Configuration and Logging
CONFIG_FILE: str = "config.json"
LOG_DIRECTORY: str = "bot_logs"
DEFAULT_TIMEZONE_STR: str = "America/Chicago"
TIMEZONE_STR: str = os.getenv("TIMEZONE", DEFAULT_TIMEZONE_STR)
try:
    TIMEZONE = ZoneInfo(TIMEZONE_STR)
except ZoneInfoNotFoundError:
    print(f"{Fore.RED}Timezone '{TIMEZONE_STR}' not found. Install 'tzdata' (`pip install tzdata`) or check name. Using UTC fallback.{Style.RESET_ALL}")
    TIMEZONE = ZoneInfo("UTC")
    TIMEZONE_STR = "UTC"
except Exception as tz_err:
    print(f"{Fore.RED}Failed to initialize timezone '{TIMEZONE_STR}'. Error: {tz_err}. Using UTC fallback.{Style.RESET_ALL}")
    TIMEZONE = ZoneInfo("UTC")
    TIMEZONE_STR = "UTC"

# API Interaction Settings
MAX_API_RETRIES: int = 3
RETRY_DELAY_SECONDS: int = 5
POSITION_CONFIRM_DELAY_SECONDS: int = 8
LOOP_DELAY_SECONDS: int = 15
BYBIT_API_KLINE_LIMIT: int = 1000 # Max klines per Bybit V5 request

# Timeframes Mapping
VALID_INTERVALS: List[str] = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP: Dict[str, str] = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}

# Data Handling Limits
DEFAULT_FETCH_LIMIT: int = 750
MAX_DF_LEN: int = 2000 # Max DataFrame length to keep in memory

# Strategy Defaults
DEFAULT_VT_LENGTH: int = 40
DEFAULT_VT_ATR_PERIOD: int = 200
DEFAULT_VT_VOL_EMA_LENGTH: int = 950
DEFAULT_VT_ATR_MULTIPLIER: float = 3.0
DEFAULT_VT_STEP_ATR_MULTIPLIER: float = 4.0 # Unused
DEFAULT_OB_SOURCE: str = "Wicks" # "Wicks" or "Body"
DEFAULT_PH_LEFT: int = 10
DEFAULT_PH_RIGHT: int = 10
DEFAULT_PL_LEFT: int = 10
DEFAULT_PL_RIGHT: int = 10
DEFAULT_OB_EXTEND: bool = True
DEFAULT_OB_MAX_BOXES: int = 50

# Global Quote Currency (updated by load_config)
QUOTE_CURRENCY: str = "USDT"

# Logging Colors
NEON_GREEN: str = Fore.LIGHTGREEN_EX
NEON_BLUE: str = Fore.CYAN
NEON_PURPLE: str = Fore.MAGENTA
NEON_YELLOW: str = Fore.YELLOW
NEON_RED: str = Fore.LIGHTRED_EX
NEON_CYAN: str = Fore.CYAN
RESET: str = Style.RESET_ALL
BRIGHT: str = Style.BRIGHT
DIM: str = Style.DIM

# Ensure log directory exists
try:
    os.makedirs(LOG_DIRECTORY, exist_ok=True)
except OSError as e:
     print(f"{NEON_RED}{BRIGHT}FATAL: Could not create log directory '{LOG_DIRECTORY}': {e}. Ensure permissions are correct.{RESET}")
     sys.exit(1)

# Global flag for shutdown signal
_shutdown_requested = False

# --- Type Definitions ---
class OrderBlock(TypedDict):
    """Represents a bullish or bearish Order Block identified on the chart."""
    id: str                 # Unique identifier (e.g., "B_1678886400000")
    type: str               # 'bull' or 'bear'
    timestamp: pd.Timestamp # Timestamp of the candle that formed the OB (pivot candle)
    top: Decimal            # Top price level of the OB
    bottom: Decimal         # Bottom price level of the OB
    active: bool            # True if the OB is currently considered valid
    violated: bool          # True if the price has closed beyond the OB boundary
    violation_ts: Optional[pd.Timestamp] # Timestamp when violation occurred
    extended_to_ts: Optional[pd.Timestamp] # Timestamp the OB box currently extends to

class StrategyAnalysisResults(TypedDict):
    """Structured results from the strategy analysis process."""
    dataframe: pd.DataFrame             # The DataFrame with all calculated indicators (Decimal values)
    last_close: Decimal                 # The closing price of the most recent candle
    current_trend_up: Optional[bool]    # True if Volumatic Trend is up, False if down, None if undetermined
    trend_just_changed: bool            # True if the trend flipped on the last candle
    active_bull_boxes: List[OrderBlock] # List of currently active bullish OBs
    active_bear_boxes: List[OrderBlock] # List of currently active bearish OBs
    vol_norm_int: Optional[int]         # Normalized volume indicator (0-100+, integer) for the last candle
    atr: Optional[Decimal]              # ATR value for the last candle (must be positive)
    upper_band: Optional[Decimal]       # Volumatic Trend upper band value for the last candle
    lower_band: Optional[Decimal]       # Volumatic Trend lower band value for the last candle
    signal: str                         # "BUY", "SELL", "EXIT_LONG", "EXIT_SHORT", or "NONE"

class MarketInfo(TypedDict):
    """Standardized market information dictionary derived from ccxt.market."""
    id: str; symbol: str; base: str; quote: str; settle: Optional[str]
    baseId: str; quoteId: str; settleId: Optional[str]
    type: str; spot: bool; margin: bool; swap: bool; future: bool; option: bool; active: bool
    contract: bool; linear: Optional[bool]; inverse: Optional[bool]; quanto: Optional[bool]
    taker: float; maker: float; contractSize: Optional[Any]
    expiry: Optional[int]; expiryDatetime: Optional[str]; strike: Optional[float]; optionType: Optional[str]
    precision: Dict[str, Any]; limits: Dict[str, Any]; info: Dict[str, Any]
    # Custom added fields
    is_contract: bool; is_linear: bool; is_inverse: bool; contract_type_str: str
    min_amount_decimal: Optional[Decimal]; max_amount_decimal: Optional[Decimal]
    min_cost_decimal: Optional[Decimal]; max_cost_decimal: Optional[Decimal]
    amount_precision_step_decimal: Optional[Decimal]; price_precision_step_decimal: Optional[Decimal]
    contract_size_decimal: Decimal

class PositionInfo(TypedDict):
    """Standardized position information dictionary derived from ccxt.position."""
    id: Optional[str]; symbol: str; timestamp: Optional[int]; datetime: Optional[str]
    contracts: Optional[float]; contractSize: Optional[Any]; side: Optional[str]
    notional: Optional[Any]; leverage: Optional[Any]; unrealizedPnl: Optional[Any]; realizedPnl: Optional[Any]
    collateral: Optional[Any]; entryPrice: Optional[Any]; markPrice: Optional[Any]; liquidationPrice: Optional[Any]
    marginMode: Optional[str]; hedged: Optional[bool]; maintenanceMargin: Optional[Any]; maintenanceMarginPercentage: Optional[float]
    initialMargin: Optional[Any]; initialMarginPercentage: Optional[float]; marginRatio: Optional[float]
    lastUpdateTimestamp: Optional[int]; info: Dict[str, Any]
    # Custom added/parsed fields
    size_decimal: Decimal
    stopLossPrice: Optional[str]; takeProfitPrice: Optional[str]
    trailingStopLoss: Optional[str]; tslActivationPrice: Optional[str]
    # Custom flags for bot state tracking
    be_activated: bool # True if Break-Even has been set for this position instance
    tsl_activated: bool # True if Trailing Stop Loss has been set for this position instance


# --- Configuration Loading & Validation ---
class SensitiveFormatter(logging.Formatter):
    """Redacts sensitive API keys/secrets from log messages."""
    _api_key_placeholder = "***API_KEY***"
    _api_secret_placeholder = "***API_SECRET***"
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        key = API_KEY; secret = API_SECRET
        try:
            if key and isinstance(key, str) and key in msg: msg = msg.replace(key, self._api_key_placeholder)
            if secret and isinstance(secret, str) and secret in msg: msg = msg.replace(secret, self._api_secret_placeholder)
        except Exception: pass
        return msg

def setup_logger(name: str) -> logging.Logger:
    """Sets up a dedicated logger instance with console and file handlers."""
    safe_name = name.replace('/', '_').replace(':', '-')
    logger_name = f"pyrmethus_bot_{safe_name}"
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers(): return logger
    logger.setLevel(logging.DEBUG)

    # File Handler (DEBUG, Rotating, Redaction, UTC)
    try:
        fh = RotatingFileHandler(log_filename, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
        ff = SensitiveFormatter("%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s:%(lineno)d] %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
        ff.converter = time.gmtime # type: ignore
        fh.setFormatter(ff); fh.setLevel(logging.DEBUG); logger.addHandler(fh)
    except Exception as e: print(f"{NEON_RED}Error setting up file logger '{log_filename}': {e}{RESET}")

    # Console Handler (Configurable Level, Neon Colors, Local Timezone)
    try:
        sh = logging.StreamHandler(sys.stdout)
        level_colors = { logging.DEBUG: NEON_CYAN + DIM, logging.INFO: NEON_BLUE, logging.WARNING: NEON_YELLOW, logging.ERROR: NEON_RED, logging.CRITICAL: NEON_RED + BRIGHT }
        class NeonConsoleFormatter(SensitiveFormatter):
            _level_colors = level_colors; _tz = TIMEZONE
            def format(self, record: logging.LogRecord) -> str:
                level_color = self._level_colors.get(record.levelno, NEON_BLUE)
                log_fmt = f"{NEON_BLUE}%(asctime)s{RESET} - {level_color}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s"
                formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
                formatter.converter = lambda *args: datetime.now(self._tz).timetuple() # type: ignore
                return super(NeonConsoleFormatter, self).format(record)
        sh.setFormatter(NeonConsoleFormatter())
        log_level_str = os.getenv("CONSOLE_LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        sh.setLevel(log_level); logger.addHandler(sh)
    except Exception as e: print(f"{NEON_RED}Error setting up console logger: {e}{RESET}")

    logger.propagate = False
    return logger

init_logger = setup_logger("init")
init_logger.info(f"{Fore.MAGENTA}Pyrmethus Volumatic Bot v{BOT_VERSION} awakening...{Style.RESET_ALL}")
init_logger.info(f"Using Timezone: {TIMEZONE_STR}")

def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any], parent_key: str = "") -> Tuple[Dict[str, Any], bool]:
    """Recursively ensures config keys exist, adding defaults if missing."""
    updated_config = config.copy(); changed = False
    for key, default_value in default_config.items():
        full_key_path = f"{parent_key}.{key}" if parent_key else key
        if key not in updated_config:
            updated_config[key] = default_value; changed = True
            init_logger.info(f"{NEON_YELLOW}Config Spell: Added missing parameter '{full_key_path}' with default enchantment: {repr(default_value)}{RESET}")
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            nested_config, nested_changed = _ensure_config_keys(updated_config[key], default_value, full_key_path)
            if nested_changed: updated_config[key] = nested_config; changed = True
    return updated_config, changed

def load_config(filepath: str) -> Dict[str, Any]:
    """Loads, validates, and potentially updates configuration from a JSON file."""
    init_logger.info(f"{Fore.CYAN}# Conjuring configuration from '{filepath}'...{Style.RESET_ALL}")
    default_config = {
        "trading_pairs": ["BTC/USDT"], "interval": "5", "retry_delay": RETRY_DELAY_SECONDS,
        "fetch_limit": DEFAULT_FETCH_LIMIT, "orderbook_limit": 25, "enable_trading": False,
        "use_sandbox": True, "risk_per_trade": 0.01, "leverage": 20, "max_concurrent_positions": 1,
        "quote_currency": "USDT", "loop_delay_seconds": LOOP_DELAY_SECONDS,
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS,
        "strategy_params": {
            "vt_length": DEFAULT_VT_LENGTH, "vt_atr_period": DEFAULT_VT_ATR_PERIOD,
            "vt_vol_ema_length": DEFAULT_VT_VOL_EMA_LENGTH, "vt_atr_multiplier": float(DEFAULT_VT_ATR_MULTIPLIER),
            "vt_step_atr_multiplier": float(DEFAULT_VT_STEP_ATR_MULTIPLIER), # Unused
            "ob_source": DEFAULT_OB_SOURCE, "ph_left": DEFAULT_PH_LEFT, "ph_right": DEFAULT_PH_RIGHT,
            "pl_left": DEFAULT_PL_LEFT, "pl_right": DEFAULT_PL_RIGHT, "ob_extend": DEFAULT_OB_EXTEND,
            "ob_max_boxes": DEFAULT_OB_MAX_BOXES, "ob_entry_proximity_factor": 1.005,
            "ob_exit_proximity_factor": 1.001
        },
        "protection": {
             "enable_trailing_stop": True, "trailing_stop_callback_rate": 0.005,
             "trailing_stop_activation_percentage": 0.003, "enable_break_even": True,
             "break_even_trigger_atr_multiple": 1.0, "break_even_offset_ticks": 2,
             "initial_stop_loss_atr_multiple": 1.8, "initial_take_profit_atr_multiple": 0.7
        }
    }
    config_needs_saving: bool = False; loaded_config: Dict[str, Any] = {}

    if not os.path.exists(filepath):
        init_logger.warning(f"{NEON_YELLOW}Config scroll '{filepath}' not found. Crafting a default scroll.{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f: json.dump(default_config, f, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Crafted default config scroll: {filepath}{RESET}")
            global QUOTE_CURRENCY; QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            return default_config
        except IOError as e:
            init_logger.critical(f"{NEON_RED}FATAL: Error crafting default config scroll '{filepath}': {e}. The weave is broken!{RESET}")
            init_logger.warning("Using internal default configuration runes. Bot may falter.")
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT"); return default_config

    try:
        with open(filepath, "r", encoding="utf-8") as f: loaded_config = json.load(f)
        if not isinstance(loaded_config, dict): raise TypeError("Configuration scroll does not contain a valid arcane map (JSON object).")
    except json.JSONDecodeError as e:
        init_logger.error(f"{NEON_RED}Error deciphering JSON from '{filepath}': {e}. Recrafting default scroll.{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f_create: json.dump(default_config, f_create, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Recrafted default config scroll due to corruption: {filepath}{RESET}")
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT"); return default_config
        except IOError as e_create:
            init_logger.critical(f"{NEON_RED}FATAL: Error recrafting default config scroll after corruption: {e_create}. Using internal defaults.{RESET}")
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT"); return default_config
    except Exception as e:
        init_logger.critical(f"{NEON_RED}FATAL: Unexpected rift loading config scroll '{filepath}': {e}{RESET}", exc_info=True)
        init_logger.warning("Using internal default configuration runes. Bot may falter.")
        QUOTE_CURRENCY = default_config.get("quote_currency", "USDT"); return default_config

    try:
        updated_config, added_keys = _ensure_config_keys(loaded_config, default_config)
        if added_keys: config_needs_saving = True

        def validate_numeric(cfg: Dict, key_path: str, min_val: Union[int, float, Decimal], max_val: Union[int, float, Decimal], is_strict_min: bool = False, is_int: bool = False, allow_zero: bool = False) -> bool:
            nonlocal config_needs_saving; keys = key_path.split('.'); current_level = cfg; default_level = default_config
            try:
                for key in keys[:-1]: current_level = current_level[key]; default_level = default_level[key]
                leaf_key = keys[-1]; original_val = current_level.get(leaf_key); default_val = default_level.get(leaf_key)
            except (KeyError, TypeError): init_logger.error(f"Config validation error: Invalid path '{key_path}'."); return False
            if original_val is None: init_logger.warning(f"Config validation: Rune missing at '{key_path}'. Using default: {repr(default_val)}"); current_level[leaf_key] = default_val; config_needs_saving = True; return True
            corrected = False; final_val = original_val
            try:
                num_val = Decimal(str(original_val)); min_dec = Decimal(str(min_val)); max_dec = Decimal(str(max_val))
                min_check = num_val > min_dec if is_strict_min else num_val >= min_dec
                range_check = min_check and num_val <= max_dec; zero_ok = allow_zero and num_val == Decimal(0)
                if not range_check and not zero_ok: raise ValueError("Value outside allowed arcane boundaries.")
                target_type = int if is_int else float; converted_val = target_type(num_val)
                needs_correction = False
                if isinstance(original_val, bool): raise TypeError("Boolean found where numeric essence expected.")
                elif is_int and not isinstance(original_val, int): needs_correction = True
                elif not is_int and not isinstance(original_val, float):
                    if isinstance(original_val, int): converted_val = float(original_val); needs_correction = True
                    else: needs_correction = True
                elif isinstance(original_val, float) and abs(original_val - converted_val) > 1e-9: needs_correction = True
                elif isinstance(original_val, int) and original_val != converted_val: needs_correction = True
                if needs_correction: init_logger.info(f"{NEON_YELLOW}Config Spell: Corrected essence/value for '{key_path}' from {repr(original_val)} to {repr(converted_val)}.{RESET}"); final_val = converted_val; corrected = True
            except (ValueError, InvalidOperation, TypeError) as e:
                range_str = f"{'(' if is_strict_min else '['}{min_val}, {max_val}{']'}" + (" or 0" if allow_zero else "")
                init_logger.warning(f"{NEON_YELLOW}Config rune '{key_path}': Invalid value '{repr(original_val)}'. Using default: {repr(default_val)}. Error: {e}. Expected: {'integer' if is_int else 'float'}, Boundaries: {range_str}{RESET}"); final_val = default_val; corrected = True
            if corrected: current_level[leaf_key] = final_val; config_needs_saving = True
            return corrected

        init_logger.debug("# Scrutinizing configuration runes...")
        # General Validations
        if not isinstance(updated_config.get("trading_pairs"), list) or not all(isinstance(s, str) and s for s in updated_config.get("trading_pairs", [])):
            init_logger.warning(f"{NEON_YELLOW}Invalid 'trading_pairs'. Must be list of non-empty strings. Using default {default_config['trading_pairs']}.{RESET}"); updated_config["trading_pairs"] = default_config["trading_pairs"]; config_needs_saving = True
        if updated_config.get("interval") not in VALID_INTERVALS:
            init_logger.warning(f"{NEON_YELLOW}Invalid 'interval' '{updated_config.get('interval')}'. Valid: {VALID_INTERVALS}. Using default '{default_config['interval']}'.{RESET}"); updated_config["interval"] = default_config["interval"]; config_needs_saving = True
        validate_numeric(updated_config, "retry_delay", 1, 60, is_int=True)
        validate_numeric(updated_config, "fetch_limit", 50, MAX_DF_LEN, is_int=True)
        validate_numeric(updated_config, "risk_per_trade", Decimal('0'), Decimal('1'), is_strict_min=True)
        validate_numeric(updated_config, "leverage", 0, 200, is_int=True, allow_zero=True)
        validate_numeric(updated_config, "loop_delay_seconds", 1, 3600, is_int=True)
        validate_numeric(updated_config, "position_confirm_delay_seconds", 1, 60, is_int=True)
        if not isinstance(updated_config.get("quote_currency"), str) or not updated_config.get("quote_currency"):
             init_logger.warning(f"Invalid 'quote_currency'. Must be non-empty string. Using default '{default_config['quote_currency']}'."); updated_config["quote_currency"] = default_config["quote_currency"]; config_needs_saving = True
        if not isinstance(updated_config.get("enable_trading"), bool):
             init_logger.warning(f"Invalid 'enable_trading'. Must be true/false. Using default '{default_config['enable_trading']}'."); updated_config["enable_trading"] = default_config["enable_trading"]; config_needs_saving = True
        if not isinstance(updated_config.get("use_sandbox"), bool):
             init_logger.warning(f"Invalid 'use_sandbox'. Must be true/false. Using default '{default_config['use_sandbox']}'."); updated_config["use_sandbox"] = default_config["use_sandbox"]; config_needs_saving = True
        # Strategy Param Validations
        validate_numeric(updated_config, "strategy_params.vt_length", 1, 500, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_atr_period", 1, MAX_DF_LEN, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_vol_ema_length", 1, MAX_DF_LEN, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_atr_multiplier", 0.1, 20.0)
        validate_numeric(updated_config, "strategy_params.ph_left", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.ph_right", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.pl_left", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.pl_right", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.ob_max_boxes", 1, 200, is_int=True)
        validate_numeric(updated_config, "strategy_params.ob_entry_proximity_factor", 1.0, 1.1)
        validate_numeric(updated_config, "strategy_params.ob_exit_proximity_factor", 1.0, 1.1)
        if updated_config["strategy_params"].get("ob_source") not in ["Wicks", "Body"]:
             init_logger.warning(f"Invalid strategy_params.ob_source. Must be 'Wicks' or 'Body'. Using default '{DEFAULT_OB_SOURCE}'."); updated_config["strategy_params"]["ob_source"] = DEFAULT_OB_SOURCE; config_needs_saving = True
        if not isinstance(updated_config["strategy_params"].get("ob_extend"), bool):
             init_logger.warning(f"Invalid strategy_params.ob_extend. Must be true/false. Using default '{DEFAULT_OB_EXTEND}'."); updated_config["strategy_params"]["ob_extend"] = DEFAULT_OB_EXTEND; config_needs_saving = True
        # Protection Param Validations
        if not isinstance(updated_config["protection"].get("enable_trailing_stop"), bool):
             init_logger.warning(f"Invalid protection.enable_trailing_stop. Must be true/false. Using default '{default_config['protection']['enable_trailing_stop']}'."); updated_config["protection"]["enable_trailing_stop"] = default_config["protection"]["enable_trailing_stop"]; config_needs_saving = True
        if not isinstance(updated_config["protection"].get("enable_break_even"), bool):
             init_logger.warning(f"Invalid protection.enable_break_even. Must be true/false. Using default '{default_config['protection']['enable_break_even']}'."); updated_config["protection"]["enable_break_even"] = default_config["protection"]["enable_break_even"]; config_needs_saving = True
        validate_numeric(updated_config, "protection.trailing_stop_callback_rate", Decimal('0.0001'), Decimal('0.5'), is_strict_min=True)
        validate_numeric(updated_config, "protection.trailing_stop_activation_percentage", Decimal('0'), Decimal('0.5'), allow_zero=True)
        validate_numeric(updated_config, "protection.break_even_trigger_atr_multiple", Decimal('0.1'), Decimal('10.0'))
        validate_numeric(updated_config, "protection.break_even_offset_ticks", 0, 1000, is_int=True, allow_zero=True)
        validate_numeric(updated_config, "protection.initial_stop_loss_atr_multiple", Decimal('0.1'), Decimal('100.0'), is_strict_min=True)
        validate_numeric(updated_config, "protection.initial_take_profit_atr_multiple", Decimal('0'), Decimal('100.0'), allow_zero=True)

        if config_needs_saving:
             try:
                 with open(filepath, "w", encoding="utf-8") as f_write: json.dump(updated_config, f_write, indent=4, ensure_ascii=False)
                 init_logger.info(f"{NEON_GREEN}Inscribed updated configuration runes to scroll: {filepath}{RESET}")
             except Exception as save_err:
                 init_logger.error(f"{NEON_RED}Error inscribing updated configuration to '{filepath}': {save_err}{RESET}", exc_info=True)
                 init_logger.warning("Proceeding with corrected runes in memory, but scroll update failed.")

        global QUOTE_CURRENCY; QUOTE_CURRENCY = updated_config.get("quote_currency", "USDT")
        init_logger.info(f"Quote currency focus set to: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
        init_logger.info(f"{Fore.CYAN}# Configuration conjuration complete.{Style.RESET_ALL}")
        return updated_config

    except Exception as e:
        init_logger.critical(f"{NEON_RED}FATAL: Unexpected vortex during configuration processing: {e}. Using internal defaults.{RESET}", exc_info=True)
        QUOTE_CURRENCY = default_config.get("quote_currency", "USDT"); return default_config

CONFIG = load_config(CONFIG_FILE)

# --- CCXT Exchange Setup ---
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """Initializes and validates the CCXT Bybit exchange object."""
    lg = logger; lg.info(f"{Fore.CYAN}# Binding the arcane energies to the Bybit exchange...{Style.RESET_ALL}")
    try:
        exchange_options = { 'apiKey': API_KEY, 'secret': API_SECRET, 'enableRateLimit': True,
            'options': { 'defaultType': 'linear', 'adjustForTimeDifference': True, 'fetchTickerTimeout': 15000,
                         'fetchBalanceTimeout': 20000, 'createOrderTimeout': 30000, 'cancelOrderTimeout': 20000,
                         'fetchPositionsTimeout': 20000, 'fetchOHLCVTimeout': 60000 } }
        exchange = ccxt.bybit(exchange_options)
        is_sandbox = CONFIG.get('use_sandbox', True); exchange.set_sandbox_mode(is_sandbox)
        if is_sandbox: lg.warning(f"{NEON_YELLOW}<<< OPERATING IN SANDBOX REALM (Testnet Environment) >>>{RESET}")
        else: lg.warning(f"{NEON_RED}{BRIGHT}!!! <<< OPERATING IN LIVE REALM - REAL ASSETS AT STAKE >>> !!!{RESET}")

        lg.info(f"Summoning market knowledge for {exchange.id}...")
        markets_loaded = False; last_market_error = None
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                lg.debug(f"Market summon attempt {attempt + 1}/{MAX_API_RETRIES + 1}...")
                exchange.load_markets(reload=(attempt > 0))
                if exchange.markets and len(exchange.markets) > 0:
                    lg.info(f"{NEON_GREEN}Market knowledge summoned successfully ({len(exchange.markets)} symbols charted).{RESET}"); markets_loaded = True; break
                else: last_market_error = ValueError("Market summoning returned an empty void"); lg.warning(f"Market summoning returned empty void (Attempt {attempt + 1}). Retrying...")
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                last_market_error = e; lg.warning(f"Aetheric disturbance (Network Error) summoning markets (Attempt {attempt + 1}): {e}.")
                if attempt >= MAX_API_RETRIES: lg.critical(f"{NEON_RED}Max retries exceeded summoning markets. Last echo: {last_market_error}. Binding failed.{RESET}"); return None
            except ccxt.AuthenticationError as e: last_market_error = e; lg.critical(f"{NEON_RED}Authentication ritual failed: {e}. Check API seals. Binding failed.{RESET}"); return None
            except Exception as e: last_market_error = e; lg.critical(f"{NEON_RED}Unexpected rift summoning markets: {e}. Binding failed.{RESET}", exc_info=True); return None
            if not markets_loaded and attempt < MAX_API_RETRIES: delay = RETRY_DELAY_SECONDS * (attempt + 1); lg.warning(f"Retrying market summon in {delay}s..."); time.sleep(delay)
        if not markets_loaded: lg.critical(f"{NEON_RED}Failed to summon markets after all attempts. Last echo: {last_market_error}. Binding failed.{RESET}"); return None

        lg.info(f"Exchange binding established: {exchange.id} | Sandbox Realm: {is_sandbox}")
        lg.info(f"Scrying initial balance for quote currency ({QUOTE_CURRENCY})...")
        initial_balance: Optional[Decimal] = None
        try: initial_balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
        except ccxt.AuthenticationError as auth_err: lg.critical(f"{NEON_RED}Authentication Ritual Failed during balance scrying: {auth_err}. Binding failed.{RESET}"); return None
        except Exception as balance_err: lg.warning(f"{NEON_YELLOW}Initial balance scrying encountered a flicker: {balance_err}.{RESET}", exc_info=False)

        if initial_balance is not None:
            lg.info(f"{NEON_GREEN}Initial available essence: {initial_balance.normalize()} {QUOTE_CURRENCY}{RESET}")
            lg.info(f"{Fore.CYAN}# Exchange binding complete and validated.{Style.RESET_ALL}"); return exchange
        else:
            lg.error(f"{NEON_RED}Initial balance scrying FAILED for {QUOTE_CURRENCY}.{RESET}")
            if CONFIG.get('enable_trading', False): lg.critical(f"{NEON_RED}Trading rituals enabled, but balance scrying failed. Cannot proceed safely. Binding failed.{RESET}"); return None
            else: lg.warning(f"{NEON_YELLOW}Trading rituals disabled. Proceeding without confirmed balance, but spells may falter.{RESET}"); lg.info(f"{Fore.CYAN}# Exchange binding complete (balance unconfirmed).{Style.RESET_ALL}"); return exchange
    except Exception as e: lg.critical(f"{NEON_RED}Failed to bind to CCXT exchange: {e}{RESET}", exc_info=True); return None

# --- CCXT Data Fetching Helpers ---
def _safe_market_decimal(value: Optional[Any], field_name: str, allow_zero: bool = True) -> Optional[Decimal]:
    """Safely converts market info value to Decimal."""
    if value is None: return None
    try:
        s_val = str(value).strip();
        if not s_val: return None
        d_val = Decimal(s_val)
        if not allow_zero and d_val <= Decimal('0'): return None
        if allow_zero and d_val < Decimal('0'): return None
        return d_val
    except (InvalidOperation, TypeError, ValueError): return None

def _format_price(exchange: ccxt.Exchange, symbol: str, price: Union[Decimal, float, str]) -> Optional[str]:
    """Formats a price to the exchange's required precision string."""
    try:
        price_decimal = Decimal(str(price))
        if price_decimal <= 0: return None # Price must be positive
        # Use CCXT's helper for correct rounding/truncating
        formatted_str = exchange.price_to_precision(symbol, float(price_decimal))
        # Final check: ensure formatted price is still positive
        if Decimal(formatted_str) > 0:
            return formatted_str
        else:
            return None
    except (InvalidOperation, ValueError, TypeError, KeyError, AttributeError):
        return None

def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches the current market price using fetch_ticker with fallbacks."""
    lg = logger; attempts = 0; last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching current price pulse for {symbol} (Attempt {attempts + 1})")
            ticker = exchange.fetch_ticker(symbol); price: Optional[Decimal] = None; source = "N/A"
            def safe_decimal_from_ticker(value: Optional[Any], field_name: str) -> Optional[Decimal]:
                if value is None: return None
                try: s_val = str(value).strip(); return Decimal(s_val) if s_val and Decimal(s_val) > 0 else None
                except (ValueError, InvalidOperation, TypeError): return None

            price = safe_decimal_from_ticker(ticker.get('last'), 'last'); source = "'last' price"
            if price is None:
                bid = safe_decimal_from_ticker(ticker.get('bid'), 'bid'); ask = safe_decimal_from_ticker(ticker.get('ask'), 'ask')
                if bid and ask: price = (bid + ask) / Decimal('2'); source = f"mid-price (B:{bid.normalize()}, A:{ask.normalize()})"
                elif ask: price = ask; source = f"'ask' price ({ask.normalize()})"
                elif bid: price = bid; source = f"'bid' price ({bid.normalize()})"
            if price: lg.debug(f"Price pulse captured ({symbol}) via {source}: {price.normalize()}"); return price.normalize()
            else: last_exception = ValueError("No valid price found in ticker"); lg.warning(f"No valid price pulse ({symbol}, Attempt {attempts + 1}). Retrying...")
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e: last_exception = e; lg.warning(f"{NEON_YELLOW}Aetheric disturbance fetching price ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e: last_exception = e; wait_time = RETRY_DELAY_SECONDS * 3; lg.warning(f"{NEON_YELLOW}Rate limit fetching price ({symbol}): {e}. Pausing {wait_time}s...{RESET}"); time.sleep(wait_time); continue
        except ccxt.AuthenticationError as e: last_exception = e; lg.critical(f"{NEON_RED}Auth ritual failed fetching price: {e}. Stopping.{RESET}"); return None
        except ccxt.ExchangeError as e: last_exception = e; lg.error(f"{NEON_RED}Exchange rift fetching price ({symbol}): {e}{RESET}")
        except Exception as e: last_exception = e; lg.error(f"{NEON_RED}Unexpected vortex fetching price ({symbol}): {e}{RESET}", exc_info=True); return None
        attempts += 1
        if attempts <= MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS * attempts)
    lg.error(f"{NEON_RED}Failed to capture price pulse ({symbol}) after {MAX_API_RETRIES + 1} attempts. Last echo: {last_exception}{RESET}"); return None

def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger) -> pd.DataFrame:
    """Fetches OHLCV klines, handling multi-request fetching if needed."""
    lg = logger
    lg.info(f"{Fore.CYAN}# Gathering historical echoes (Klines) for {symbol} | TF: {timeframe} | Limit: {limit}...{Style.RESET_ALL}")
    if not hasattr(exchange, 'fetch_ohlcv') or not exchange.has.get('fetchOHLCV'):
        lg.error(f"Exchange {exchange.id} does not support fetchOHLCV."); return pd.DataFrame()

    # Calculate minimum required candles for strategy (rough estimate)
    min_required = 0
    try:
        sp = CONFIG.get('strategy_params', {})
        min_required = max(sp.get('vt_length', 0), sp.get('vt_atr_period', 0), sp.get('vt_vol_ema_length', 0),
                           sp.get('ph_left', 0) + sp.get('ph_right', 0) + 1,
                           sp.get('pl_left', 0) + sp.get('pl_right', 0) + 1) + 50 # Add buffer
        lg.debug(f"Estimated minimum candles required by strategy: {min_required}")
        if limit < min_required:
            lg.warning(f"{NEON_YELLOW}Requested limit ({limit}) is less than estimated strategy requirement ({min_required}). Indicator accuracy may be affected.{RESET}")
    except Exception as e: lg.warning(f"Could not estimate minimum required candles: {e}")

    # Determine category for Bybit V5
    category = 'spot' # Default
    market_id = symbol # Default
    try:
        market = exchange.market(symbol)
        market_id = market['id']
        category = 'linear' if market.get('linear') else 'inverse' if market.get('inverse') else 'spot'
        lg.debug(f"Using Bybit category: {category} for kline fetch.")
    except Exception as e: lg.warning(f"Could not determine market category for {symbol}: {e}. Using default.")

    all_ohlcv_data: List[List[Union[int, float, str]]] = []
    remaining_limit = limit
    end_timestamp_ms: Optional[int] = None # Fetch going backwards from current time

    while remaining_limit > 0:
        fetch_size = min(remaining_limit, BYBIT_API_KLINE_LIMIT)
        lg.debug(f"Fetching chunk of {fetch_size} klines for {symbol} (End TS: {end_timestamp_ms})...")
        attempts = 0
        last_exception = None
        chunk_data: Optional[List[List[Union[int, float, str]]]] = None

        while attempts <= MAX_API_RETRIES:
            try:
                params = {'category': category}
                # CCXT handles the 'until' parameter based on end_timestamp_ms
                fetch_args = {'symbol': symbol, 'timeframe': timeframe, 'limit': fetch_size, 'params': params}
                if end_timestamp_ms: fetch_args['until'] = end_timestamp_ms

                chunk_data = exchange.fetch_ohlcv(**fetch_args) # type: ignore
                fetched_count = len(chunk_data) if chunk_data else 0
                lg.debug(f"API returned {fetched_count} candles for this chunk.")

                if chunk_data:
                    # Basic validation (e.g., timestamp check on first fetch)
                    if not all_ohlcv_data: # Only check lag on the most recent chunk
                        try:
                            last_candle_timestamp_ms = chunk_data[-1][0]
                            last_ts = pd.to_datetime(last_candle_timestamp_ms, unit='ms', utc=True)
                            now_utc = pd.Timestamp.utcnow()
                            interval_seconds = exchange.parse_timeframe(timeframe)
                            if interval_seconds:
                                max_allowed_lag = interval_seconds * 2.5
                                actual_lag = (now_utc - last_ts).total_seconds()
                                if actual_lag > max_allowed_lag:
                                    last_exception = ValueError(f"Kline data potentially stale (Lag: {actual_lag:.1f}s > Max: {max_allowed_lag:.1f}s).")
                                    lg.warning(f"{NEON_YELLOW}Timestamp lag detected ({symbol}): {last_exception}. Retrying fetch...{RESET}")
                                    chunk_data = None # Discard and retry
                                    # No break here, let retry logic handle it
                            else: lg.warning("Could not parse timeframe for lag check.")
                        except Exception as ts_err: lg.warning(f"Could not validate timestamp lag ({symbol}): {ts_err}. Proceeding cautiously.")
                    # If validation passed or wasn't needed, break retry loop for this chunk
                    if chunk_data: break
                else:
                    # If API returns empty list, it might mean no more data available going back
                    lg.debug(f"API returned no data for chunk (End TS: {end_timestamp_ms}). Assuming end of history.")
                    remaining_limit = 0 # Stop fetching further chunks
                    break # Exit retry loop for this chunk

            # Error Handling (same as before, applied per chunk)
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e: last_exception = e; lg.warning(f"{NEON_YELLOW}Network error fetching kline chunk ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
            except ccxt.RateLimitExceeded as e: last_exception = e; wait_time = RETRY_DELAY_SECONDS * 3; lg.warning(f"{NEON_YELLOW}Rate limit fetching kline chunk ({symbol}): {e}. Pausing {wait_time}s...{RESET}"); time.sleep(wait_time); continue
            except ccxt.AuthenticationError as e: last_exception = e; lg.critical(f"{NEON_RED}Auth ritual failed fetching klines: {e}. Stopping.{RESET}"); return pd.DataFrame()
            except ccxt.ExchangeError as e: last_exception = e; lg.error(f"{NEON_RED}Exchange rift fetching klines ({symbol}): {e}{RESET}"); # Check non-retryable?
            except Exception as e: last_exception = e; lg.error(f"{NEON_RED}Unexpected vortex fetching klines ({symbol}): {e}{RESET}", exc_info=True); return pd.DataFrame()

            attempts += 1
            if attempts <= MAX_API_RETRIES and chunk_data is None: time.sleep(RETRY_DELAY_SECONDS * attempts)

        # --- After chunk retry loop ---
        if chunk_data:
            # Prepend older data to the main list
            all_ohlcv_data = chunk_data + all_ohlcv_data
            remaining_limit -= len(chunk_data)
            # Set the end timestamp for the *next* fetch request (oldest timestamp - 1ms)
            end_timestamp_ms = chunk_data[0][0] - 1
            # Check if we received fewer candles than requested, implies end of history
            if len(chunk_data) < fetch_size:
                 lg.debug(f"Received fewer candles ({len(chunk_data)}) than requested ({fetch_size}). Assuming end of available history.")
                 remaining_limit = 0 # Stop fetching
        else:
            # Fetching chunk failed after retries
            lg.error(f"{NEON_RED}Failed to fetch kline chunk for {symbol} after {MAX_API_RETRIES + 1} attempts. Last echo: {last_exception}{RESET}")
            # Decide whether to proceed with partial data or fail entirely
            if not all_ohlcv_data: # Failed on the very first chunk
                 return pd.DataFrame()
            else:
                 lg.warning(f"Proceeding with {len(all_ohlcv_data)} candles fetched before error.")
                 break # Exit the main fetching loop

        # Small delay between fetches to be kind to the API
        if remaining_limit > 0: time.sleep(0.5)

    # --- Process Combined Data ---
    if not all_ohlcv_data:
        lg.error(f"No kline data could be fetched for {symbol} {timeframe}.")
        return pd.DataFrame()

    lg.info(f"Total klines fetched across all requests: {len(all_ohlcv_data)}")
    # Deduplicate based on timestamp (just in case of overlap, keep first occurrence)
    seen_timestamps = set()
    unique_data = []
    for candle in all_ohlcv_data:
        ts = candle[0]
        if ts not in seen_timestamps:
            unique_data.append(candle)
            seen_timestamps.add(ts)
    if len(unique_data) != len(all_ohlcv_data):
        lg.warning(f"Removed {len(all_ohlcv_data) - len(unique_data)} duplicate candle timestamps.")
    all_ohlcv_data = unique_data

    # Sort by timestamp just to be absolutely sure
    all_ohlcv_data.sort(key=lambda x: x[0])

    # Limit to the originally requested number of candles (most recent)
    if len(all_ohlcv_data) > limit:
        lg.debug(f"Fetched {len(all_ohlcv_data)} candles, trimming to requested limit {limit}.")
        all_ohlcv_data = all_ohlcv_data[-limit:]

    # Process into DataFrame (same logic as before)
    try:
        lg.debug(f"Processing {len(all_ohlcv_data)} final candles into DataFrame ({symbol})...")
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(all_ohlcv_data, columns=cols[:len(all_ohlcv_data[0])])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        if df.empty: lg.error(f"DataFrame empty after timestamp conversion ({symbol})."); return pd.DataFrame()
        df.set_index('timestamp', inplace=True)

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                df[col] = numeric_series.apply(lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN'))
            else: lg.warning(f"Expected column '{col}' not found ({symbol}).")

        initial_len = len(df)
        essential_price_cols = ['open', 'high', 'low', 'close']
        df.dropna(subset=essential_price_cols, inplace=True)
        df = df[df['close'] > Decimal('0')]
        if 'volume' in df.columns: df.dropna(subset=['volume'], inplace=True); df = df[df['volume'] >= Decimal('0')]
        rows_dropped = initial_len - len(df)
        if rows_dropped > 0: lg.debug(f"Purged {rows_dropped} rows ({symbol}) during cleaning.")
        if df.empty: lg.warning(f"Kline DataFrame empty after cleaning ({symbol})."); return pd.DataFrame()
        if not df.index.is_monotonic_increasing: lg.warning(f"Kline index not monotonic ({symbol}), sorting..."); df.sort_index(inplace=True)
        if len(df) > MAX_DF_LEN: lg.debug(f"DataFrame length ({len(df)}) > max ({MAX_DF_LEN}). Trimming."); df = df.iloc[-MAX_DF_LEN:].copy()

        lg.info(f"{NEON_GREEN}Successfully gathered and processed {len(df)} kline echoes for {symbol} {timeframe}{RESET}")
        return df
    except Exception as e: lg.error(f"{NEON_RED}Error processing kline echoes ({symbol}): {e}{RESET}", exc_info=True); return pd.DataFrame()

def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[MarketInfo]:
    """Retrieves, validates, and standardizes market information."""
    lg = logger; lg.debug(f"Seeking market details for symbol: {symbol}...")
    attempts = 0; last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            market: Optional[Dict] = None
            if not exchange.markets or symbol not in exchange.markets:
                lg.info(f"Market details for '{symbol}' not found. Refreshing market map...");
                try: exchange.load_markets(reload=True); lg.info("Market map refreshed.")
                except Exception as reload_err: last_exception = reload_err; lg.error(f"Failed to refresh market map: {reload_err}")
            try: market = exchange.market(symbol)
            except ccxt.BadSymbol: market = None
            except Exception as fetch_err: last_exception = fetch_err; lg.warning(f"Error fetching market dict for '{symbol}': {fetch_err}. Retrying...") ; market = None

            if market is None:
                if attempts < MAX_API_RETRIES: lg.warning(f"Symbol '{symbol}' not found or fetch failed (Attempt {attempts + 1}). Retrying check...")
                else: lg.error(f"{NEON_RED}Market '{symbol}' not found on {exchange.id} after retries. Last echo: {last_exception}{RESET}"); return None
            else:
                lg.debug(f"Market found for '{symbol}'. Standardizing details...")
                std_market = market.copy()
                is_spot = std_market.get('spot', False); is_swap = std_market.get('swap', False); is_future = std_market.get('future', False)
                is_linear = std_market.get('linear'); is_inverse = std_market.get('inverse')
                std_market['is_contract'] = is_swap or is_future or std_market.get('contract', False)
                std_market['is_linear'] = is_linear is True and std_market['is_contract']
                std_market['is_inverse'] = is_inverse is True and std_market['is_contract']
                std_market['contract_type_str'] = "Linear" if std_market['is_linear'] else "Inverse" if std_market['is_inverse'] else "Spot" if is_spot else "Unknown"

                precision = std_market.get('precision', {}); limits = std_market.get('limits', {})
                amount_limits = limits.get('amount', {}); cost_limits = limits.get('cost', {})
                std_market['amount_precision_step_decimal'] = _safe_market_decimal(precision.get('amount'), 'precision.amount', allow_zero=False)
                std_market['price_precision_step_decimal'] = _safe_market_decimal(precision.get('price'), 'precision.price', allow_zero=False)
                std_market['min_amount_decimal'] = _safe_market_decimal(amount_limits.get('min'), 'limits.amount.min')
                std_market['max_amount_decimal'] = _safe_market_decimal(amount_limits.get('max'), 'limits.amount.max', allow_zero=False)
                std_market['min_cost_decimal'] = _safe_market_decimal(cost_limits.get('min'), 'limits.cost.min')
                std_market['max_cost_decimal'] = _safe_market_decimal(cost_limits.get('max'), 'limits.cost.max', allow_zero=False)
                contract_size_val = std_market.get('contractSize', '1')
                std_market['contract_size_decimal'] = _safe_market_decimal(contract_size_val, 'contractSize', allow_zero=False) or Decimal('1')

                if std_market['amount_precision_step_decimal'] is None or std_market['price_precision_step_decimal'] is None:
                    lg.error(f"{NEON_RED}CRITICAL VALIDATION FAILED:{RESET} Market '{symbol}' missing essential precision runes."); lg.error(f"  Amount Step: {std_market['amount_precision_step_decimal']}, Price Step: {std_market['price_precision_step_decimal']}"); return None

                log_msg = ( f"Market Details ({symbol}): Type={std_market['contract_type_str']}, Active={std_market.get('active')}\n"
                            f"  Precision (Amt/Price): {std_market['amount_precision_step_decimal'].normalize()} / {std_market['price_precision_step_decimal'].normalize()}\n"
                            f"  Limits (Amt Min/Max): {std_market['min_amount_decimal'].normalize() if std_market['min_amount_decimal'] is not None else 'N/A'} / {std_market['max_amount_decimal'].normalize() if std_market['max_amount_decimal'] is not None else 'N/A'}\n"
                            f"  Limits (Cost Min/Max): {std_market['min_cost_decimal'].normalize() if std_market['min_cost_decimal'] is not None else 'N/A'} / {std_market['max_cost_decimal'].normalize() if std_market['max_cost_decimal'] is not None else 'N/A'}\n"
                            f"  Contract Size: {std_market['contract_size_decimal'].normalize()}" )
                lg.debug(log_msg)
                try: final_market_info: MarketInfo = std_market; return final_market_info # type: ignore
                except Exception as cast_err: lg.error(f"Error casting market dict to TypedDict: {cast_err}"); return std_market # type: ignore

        except ccxt.BadSymbol as e: lg.error(f"Symbol '{symbol}' is invalid on {exchange.id}: {e}"); return None
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e: last_exception = e; lg.warning(f"{NEON_YELLOW}Aetheric disturbance retrieving market info ({symbol}): {e}. Retry {attempts + 1}...{RESET}"); if attempts >= MAX_API_RETRIES: lg.error(f"{NEON_RED}Max retries for NetworkError market info ({symbol}).{RESET}"); return None
        except ccxt.AuthenticationError as e: last_exception = e; lg.critical(f"{NEON_RED}Auth ritual failed getting market info: {e}. Stopping.{RESET}"); return None
        except ccxt.ExchangeError as e: last_exception = e; lg.error(f"{NEON_RED}Exchange rift retrieving market info ({symbol}): {e}{RESET}"); if attempts >= MAX_API_RETRIES: lg.error(f"{NEON_RED}Max retries for ExchangeError market info ({symbol}).{RESET}"); return None
        except Exception as e: last_exception = e; lg.error(f"{NEON_RED}Unexpected vortex retrieving market info ({symbol}): {e}{RESET}", exc_info=True); return None
        attempts += 1; time.sleep(RETRY_DELAY_SECONDS * attempts)
    lg.error(f"{NEON_RED}Failed to retrieve market info ({symbol}) after all attempts. Last echo: {last_exception}{RESET}"); return None

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches the available trading balance for a specific currency."""
    lg = logger; lg.debug(f"Scrying balance for currency: {currency}...")
    attempts = 0; last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            balance_str: Optional[str] = None; found: bool = False; balance_info: Optional[Dict] = None
            account_types_to_check = ['UNIFIED', 'CONTRACT', ''] if 'bybit' in exchange.id.lower() else ['']
            for acc_type in account_types_to_check:
                try:
                    params = {'accountType': acc_type} if acc_type else {}; type_desc = f"Type: {acc_type}" if acc_type else "Default"
                    lg.debug(f"Fetching balance ({currency}, {type_desc}, Attempt {attempts + 1})...")
                    balance_info = exchange.fetch_balance(params=params)
                    if currency in balance_info and balance_info[currency].get('free') is not None: balance_str = str(balance_info[currency]['free']); lg.debug(f"Found balance in 'free' field ({type_desc}): {balance_str}"); found = True; break
                    elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                        for acc_details in balance_info['info']['result']['list']:
                             if (not acc_type or acc_details.get('accountType') == acc_type) and isinstance(acc_details.get('coin'), list):
                                for coin_data in acc_details['coin']:
                                    if coin_data.get('coin') == currency:
                                        balance_val = coin_data.get('availableToWithdraw') or coin_data.get('availableBalance') or coin_data.get('walletBalance')
                                        if balance_val is not None: balance_str = str(balance_val); source = 'availableToWithdraw' if coin_data.get('availableToWithdraw') else 'availableBalance' if coin_data.get('availableBalance') else 'walletBalance'; lg.debug(f"Found balance in Bybit V5 (Acc: {acc_details.get('accountType')}, Field: {source}): {balance_str}"); found = True; break
                                if found: break
                        if found: break
                except ccxt.ExchangeError as e:
                    if acc_type and ("account type does not exist" in str(e).lower() or "invalid account type" in str(e).lower()): lg.debug(f"Account type '{acc_type}' not found. Trying next...")
                    elif acc_type: lg.debug(f"Minor exchange rift fetching balance ({acc_type}): {e}. Trying next...")
                    else: raise e
                    continue
                except Exception as e: lg.warning(f"Unexpected flicker fetching balance ({acc_type or 'Default'}): {e}. Trying next..."); last_exception = e; continue
            if found and balance_str is not None:
                try: balance_decimal = Decimal(balance_str); final_balance = max(balance_decimal, Decimal('0')); lg.debug(f"Parsed balance ({currency}): {final_balance.normalize()}"); return final_balance
                except (ValueError, InvalidOperation, TypeError) as e: raise ccxt.ExchangeError(f"Failed to convert balance string '{balance_str}' ({currency}): {e}")
            elif not found and balance_info is not None: raise ccxt.ExchangeError(f"Could not find balance for '{currency}'. Last response info: {balance_info.get('info')}")
            elif not found and balance_info is None: raise ccxt.ExchangeError(f"Could not find balance for '{currency}'. Fetch failed.")
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e: last_exception = e; lg.warning(f"{NEON_YELLOW}Aetheric disturbance fetching balance ({currency}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e: last_exception = e; wait_time = RETRY_DELAY_SECONDS * 3; lg.warning(f"{NEON_YELLOW}Rate limit fetching balance ({currency}): {e}. Pausing {wait_time}s...{RESET}"); time.sleep(wait_time); continue
        except ccxt.AuthenticationError as e: last_exception = e; lg.critical(f"{NEON_RED}Auth ritual failed fetching balance: {e}. Stopping.{RESET}"); raise e
        except ccxt.ExchangeError as e: last_exception = e; lg.warning(f"{NEON_YELLOW}Exchange rift fetching balance ({currency}): {e}. Retry {attempts + 1}...{RESET}")
        except Exception as e: last_exception = e; lg.error(f"{NEON_RED}Unexpected vortex fetching balance ({currency}): {e}{RESET}", exc_info=True); return None
        attempts += 1
        if attempts <= MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS * attempts)
    lg.error(f"{NEON_RED}Failed to scry balance ({currency}) after {MAX_API_RETRIES + 1} attempts. Last echo: {last_exception}{RESET}"); return None

# --- Position & Order Management ---
def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[PositionInfo]:
    """Checks for an existing open position for the symbol."""
    lg = logger; lg.debug(f"Seeking open position for symbol: {symbol}...")
    attempts = 0; last_exception = None; market_id: Optional[str] = None; category: Optional[str] = None
    try:
        market = exchange.market(symbol); market_id = market['id']
        category = 'linear' if market.get('linear') else 'inverse' if market.get('inverse') else 'spot'
        if category == 'spot': lg.info(f"Position check skipped for {symbol}: Spot market."); return None
        lg.debug(f"Using Market ID: {market_id}, Category: {category} for position check.")
    except KeyError: lg.error(f"Market '{symbol}' not found. Cannot check position."); return None
    except Exception as e: lg.error(f"Error determining market details for position check ({symbol}): {e}"); return None

    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching positions for {symbol} (Attempt {attempts + 1})...")
            positions: List[Dict] = []
            try:
                params = {'category': category, 'symbol': market_id}
                lg.debug(f"Fetching positions with params: {params}")
                if exchange.has.get('fetchPositions'):
                     all_positions = exchange.fetch_positions(params=params)
                     positions = [p for p in all_positions if p.get('symbol') == symbol or p.get('info', {}).get('symbol') == market_id]
                     lg.debug(f"Fetched {len(all_positions)} total positions ({category}), filtered to {len(positions)} for {symbol}.")
                else: raise ccxt.NotSupported("Exchange does not support fetchPositions.")
            except ccxt.ExchangeError as e:
                 no_pos_codes = [110025]; no_pos_messages = ["position not found", "no position", "position does not exist"]
                 err_str = str(e).lower(); code_str = ""; match = re.search(r'(retCode|ret_code)=(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE);
                 if match: code_str = match.group(2)
                 if not code_str: code_str = str(getattr(e, 'code', '') or getattr(e, 'retCode', ''))
                 code_match = any(str(c) in code_str for c in no_pos_codes) if code_str else False
                 if code_match or any(msg in err_str for msg in no_pos_messages): lg.info(f"No open position found for {symbol} (Exchange message: {e})."); return None
                 else: raise e

            active_position_raw: Optional[Dict] = None
            size_threshold = Decimal('1e-9');
            try: amt_step = Decimal(str(exchange.market(symbol)['precision']['amount'])); size_threshold = amt_step * Decimal('0.01') if amt_step > 0 else size_threshold
            except Exception: pass # Ignore precision errors for threshold
            lg.debug(f"Using position size threshold: {size_threshold.normalize()}")

            for pos in positions:
                size_str_info = str(pos.get('info', {}).get('size', '')).strip(); size_str_std = str(pos.get('contracts', '')).strip()
                size_str = size_str_info if size_str_info else size_str_std
                if not size_str: continue
                try:
                    size_decimal = Decimal(size_str)
                    if abs(size_decimal) > size_threshold: active_position_raw = pos; active_position_raw['size_decimal'] = size_decimal; lg.debug(f"Found active position entry ({symbol}): Size={size_decimal.normalize()}"); break
                    else: lg.debug(f"Skipping position entry near-zero size ({size_decimal.normalize()}): {pos.get('info', {})}")
                except (ValueError, InvalidOperation, TypeError) as parse_err: lg.warning(f"Could not parse position size '{size_str}' ({symbol}): {parse_err}. Skipping."); continue

            if active_position_raw:
                std_pos = active_position_raw.copy(); info = std_pos.get('info', {})
                side = std_pos.get('side'); size = std_pos['size_decimal']
                if side not in ['long', 'short']:
                    side_v5 = str(info.get('side', '')).lower()
                    if side_v5 == 'buy': side = 'long'
                    elif side_v5 == 'sell': side = 'short'
                    elif size > size_threshold: side = 'long'
                    elif size < -size_threshold: side = 'short'
                    else: side = None
                if not side: lg.error(f"Could not determine side for active position {symbol}. Size: {size}. Data: {info}"); return None
                std_pos['side'] = side

                std_pos['entryPrice'] = _safe_market_decimal(std_pos.get('entryPrice') or info.get('avgPrice') or info.get('entryPrice'), 'entryPrice')
                std_pos['leverage'] = _safe_market_decimal(std_pos.get('leverage') or info.get('leverage'), 'leverage', allow_zero=False)
                std_pos['liquidationPrice'] = _safe_market_decimal(std_pos.get('liquidationPrice') or info.get('liqPrice'), 'liquidationPrice', allow_zero=False)
                std_pos['unrealizedPnl'] = _safe_market_decimal(std_pos.get('unrealizedPnl') or info.get('unrealisedPnl') or info.get('unrealizedPnl'), 'unrealizedPnl', allow_zero=True)

                def get_protection_field(field_name: str) -> Optional[str]:
                    value = info.get(field_name); s_value = str(value).strip() if value is not None else None
                    try: return s_value if s_value and abs(Decimal(s_value)) > Decimal('1e-12') else None
                    except (InvalidOperation, ValueError, TypeError): return None
                std_pos['stopLossPrice'] = get_protection_field('stopLoss')
                std_pos['takeProfitPrice'] = get_protection_field('takeProfit')
                std_pos['trailingStopLoss'] = get_protection_field('trailingStop')
                std_pos['tslActivationPrice'] = get_protection_field('activePrice')
                # Initialize bot state flags
                std_pos['be_activated'] = False # Will be set by management logic if BE applied
                std_pos['tsl_activated'] = bool(std_pos['trailingStopLoss']) # True if TSL distance is already set

                def format_decimal_log(value: Optional[Any]) -> str: dec_val = _safe_market_decimal(value, 'log', True); return dec_val.normalize() if dec_val is not None else 'N/A'
                ep_str = format_decimal_log(std_pos.get('entryPrice')); size_str = std_pos['size_decimal'].normalize()
                sl_str = format_decimal_log(std_pos.get('stopLossPrice')); tp_str = format_decimal_log(std_pos.get('takeProfitPrice'))
                tsl_dist = format_decimal_log(std_pos.get('trailingStopLoss')); tsl_act = format_decimal_log(std_pos.get('tslActivationPrice'))
                tsl_log = f"Dist={tsl_dist}/Act={tsl_act}" if tsl_dist != 'N/A' or tsl_act != 'N/A' else "N/A"
                pnl_str = format_decimal_log(std_pos.get('unrealizedPnl')); liq_str = format_decimal_log(std_pos.get('liquidationPrice'))

                lg.info(f"{NEON_GREEN}{BRIGHT}Active {side.upper()} Position Found ({symbol}):{RESET} Size={size_str}, Entry={ep_str}, Liq={liq_str}, PnL={pnl_str}, SL={sl_str}, TP={tp_str}, TSL={tsl_log}")
                try: final_position_info: PositionInfo = std_pos; return final_position_info # type: ignore
                except Exception as cast_err: lg.error(f"Error casting position to TypedDict ({symbol}): {cast_err}"); return std_pos # type: ignore
            else: lg.info(f"No active position found for {symbol}."); return None

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e: last_exception = e; lg.warning(f"{NEON_YELLOW}Aetheric disturbance fetching positions ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e: last_exception = e; wait_time = RETRY_DELAY_SECONDS * 3; lg.warning(f"{NEON_YELLOW}Rate limit fetching positions ({symbol}): {e}. Pausing {wait_time}s...{RESET}"); time.sleep(wait_time); continue
        except ccxt.AuthenticationError as e: last_exception = e; lg.critical(f"{NEON_RED}Auth ritual failed fetching positions: {e}. Stopping.{RESET}"); return None
        except ccxt.ExchangeError as e: last_exception = e; lg.warning(f"{NEON_YELLOW}Exchange rift fetching positions ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except Exception as e: last_exception = e; lg.error(f"{NEON_RED}Unexpected vortex fetching positions ({symbol}): {e}{RESET}", exc_info=True); return None
        attempts += 1
        if attempts <= MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS * attempts)
    lg.error(f"{NEON_RED}Failed to get position info ({symbol}) after {MAX_API_RETRIES + 1} attempts. Last echo: {last_exception}{RESET}"); return None

def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: MarketInfo, logger: logging.Logger) -> bool:
    """Sets the leverage for a derivatives symbol."""
    lg = logger
    if not market_info.get('is_contract', False): lg.info(f"Leverage setting skipped ({symbol}): Not contract."); return True
    if not isinstance(leverage, int) or leverage <= 0: lg.warning(f"Leverage setting skipped ({symbol}): Invalid leverage {leverage}."); return False
    if not hasattr(exchange, 'set_leverage') or not exchange.has.get('setLeverage'): lg.error(f"Exchange {exchange.id} does not support setLeverage."); return False
    market_id = market_info['id']; attempts = 0; last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.info(f"Attempting leverage set ({market_id} to {leverage}x, Attempt {attempts + 1})...")
            params = {}; category = market_info.get('contract_type_str', 'Linear').lower()
            if 'bybit' in exchange.id.lower():
                 if category not in ['linear', 'inverse']: lg.warning(f"Leverage skipped: Invalid category '{category}' ({symbol})."); return False
                 params = {'category': category, 'buyLeverage': str(leverage), 'sellLeverage': str(leverage)}; lg.debug(f"Using Bybit V5 leverage params: {params}")
            response = exchange.set_leverage(leverage=leverage, symbol=market_id, params=params); lg.debug(f"set_leverage raw response ({symbol}): {response}")
            ret_code_str = None; ret_msg = "N/A"
            if isinstance(response, dict):
                 info_dict = response.get('info', {}); ret_code_info = info_dict.get('retCode'); ret_code_top = response.get('retCode')
                 if ret_code_info is not None and ret_code_info != 0: ret_code_str = str(ret_code_info)
                 elif ret_code_top is not None: ret_code_str = str(ret_code_top)
                 else: ret_code_str = str(ret_code_info) if ret_code_info is not None else str(ret_code_top)
                 ret_msg = info_dict.get('retMsg', response.get('retMsg', 'Unknown Bybit msg'))
            if ret_code_str == '0': lg.info(f"{NEON_GREEN}Leverage set ({market_id} to {leverage}x, Code: 0).{RESET}"); return True
            elif ret_code_str == '110045': lg.info(f"{NEON_YELLOW}Leverage already {leverage}x ({market_id}, Code: 110045).{RESET}"); return True
            elif ret_code_str is not None and ret_code_str not in ['None', '0']: raise ccxt.ExchangeError(f"Bybit API error setting leverage ({symbol}): {ret_msg} (Code: {ret_code_str})")
            else: lg.info(f"{NEON_GREEN}Leverage set/confirmed ({market_id} to {leverage}x, No specific error code).{RESET}"); return True
        except ccxt.ExchangeError as e:
            last_exception = e; err_code_str = ""; match = re.search(r'(retCode|ret_code)=(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE);
            if match: err_code_str = match.group(2)
            if not err_code_str: err_code_str = str(getattr(e, 'code', '') or getattr(e, 'retCode', ''))
            err_str = str(e).lower(); lg.error(f"{NEON_RED}Exchange rift setting leverage ({market_id}): {e} (Code: {err_code_str}){RESET}")
            if err_code_str == '110045' or "leverage not modified" in err_str: lg.info(f"{NEON_YELLOW}Leverage already set (via error). Success.{RESET}"); return True
            fatal_codes = ['10001', '10004', '110009', '110013', '110028', '110043', '110044', '110055', '3400045']
            fatal_messages = ["margin mode", "position exists", "risk limit", "parameter error", "insufficient balance", "invalid leverage"]
            if err_code_str in fatal_codes or any(msg in err_str for msg in fatal_messages): lg.error(f"{NEON_RED} >> Hint: NON-RETRYABLE leverage error ({symbol}). Aborting.{RESET}"); return False
            if attempts >= MAX_API_RETRIES: lg.error(f"{NEON_RED}Max retries for ExchangeError setting leverage ({symbol}).{RESET}"); return False
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e: last_exception = e; lg.warning(f"{NEON_YELLOW}Aetheric disturbance setting leverage ({market_id}): {e}. Retry {attempts + 1}...{RESET}"); if attempts >= MAX_API_RETRIES: lg.error(f"{NEON_RED}Max retries for NetworkError setting leverage ({symbol}).{RESET}"); return False
        except ccxt.AuthenticationError as e: last_exception = e; lg.critical(f"{NEON_RED}Auth ritual failed setting leverage ({symbol}): {e}. Stopping.{RESET}"); return False
        except Exception as e: last_exception = e; lg.error(f"{NEON_RED}Unexpected vortex setting leverage ({market_id}): {e}{RESET}", exc_info=True); return False
        attempts += 1; time.sleep(RETRY_DELAY_SECONDS * attempts)
    lg.error(f"{NEON_RED}Failed leverage set ({market_id} to {leverage}x) after {MAX_API_RETRIES + 1} attempts. Last echo: {last_exception}{RESET}"); return False

def calculate_position_size(balance: Decimal, risk_per_trade: float, initial_stop_loss_price: Decimal, entry_price: Decimal, market_info: MarketInfo, exchange: ccxt.Exchange, logger: logging.Logger) -> Optional[Decimal]:
    """Calculates position size based on risk, SL, and market constraints."""
    lg = logger; symbol = market_info['symbol']; quote_currency = market_info.get('quote', 'QUOTE'); base_currency = market_info.get('base', 'BASE')
    is_inverse = market_info.get('is_inverse', False); size_unit = base_currency if market_info.get('spot', False) else "Contracts"
    lg.info(f"{BRIGHT}--- Position Sizing Calculation ({symbol}) ---{RESET}")
    if balance <= Decimal('0'): lg.error(f"Sizing failed ({symbol}): Invalid balance {balance.normalize()}."); return None
    try: risk_decimal = Decimal(str(risk_per_trade)); assert Decimal('0') < risk_decimal <= Decimal('1')
    except Exception as e: lg.error(f"Sizing failed ({symbol}): Invalid risk '{risk_per_trade}': {e}"); return None
    if initial_stop_loss_price <= Decimal('0') or entry_price <= Decimal('0'): lg.error(f"Sizing failed ({symbol}): Entry ({entry_price.normalize()}) / SL ({initial_stop_loss_price.normalize()}) must be positive."); return None
    if initial_stop_loss_price == entry_price: lg.error(f"Sizing failed ({symbol}): SL price equals Entry price."); return None
    try:
        amount_step = market_info['amount_precision_step_decimal']; price_step = market_info['price_precision_step_decimal']
        min_amount = market_info['min_amount_decimal'] or Decimal('0'); max_amount = market_info['max_amount_decimal'] or Decimal('inf')
        min_cost = market_info['min_cost_decimal'] or Decimal('0'); max_cost = market_info['max_cost_decimal'] or Decimal('inf')
        contract_size = market_info['contract_size_decimal']
        assert amount_step and amount_step > 0; assert price_step and price_step > 0; assert contract_size > 0
        lg.debug(f"  Market Constraints ({symbol}): AmtStep={amount_step.normalize()}, Min/Max Amt={min_amount.normalize()}/{max_amount.normalize()}, Min/Max Cost={min_cost.normalize()}/{max_cost.normalize()}, ContrSize={contract_size.normalize()}")
    except (KeyError, ValueError, TypeError, AssertionError) as e: lg.error(f"Sizing failed ({symbol}): Error validating market details: {e}"); lg.debug(f" MarketInfo: {market_info}"); return None

    risk_amount_quote = (balance * risk_decimal).quantize(Decimal('1e-8'), ROUND_DOWN)
    stop_loss_distance = abs(entry_price - initial_stop_loss_price)
    if stop_loss_distance <= Decimal('0'): lg.error(f"Sizing failed ({symbol}): SL distance zero."); return None
    lg.info(f"  Balance: {balance.normalize()} {quote_currency}, Risk: {risk_decimal:.2%} ({risk_amount_quote.normalize()} {quote_currency})")
    lg.info(f"  Entry: {entry_price.normalize()}, SL: {initial_stop_loss_price.normalize()}, SL Dist: {stop_loss_distance.normalize()}")
    lg.info(f"  Contract Type: {market_info['contract_type_str']}")

    calculated_size = Decimal('0')
    try:
        if not is_inverse: # Linear / Spot
            value_change_per_unit = stop_loss_distance * contract_size
            if value_change_per_unit <= Decimal('1e-18'): lg.error(f"Sizing failed ({symbol}, Lin/Spot): Value change per unit near zero."); return None
            calculated_size = risk_amount_quote / value_change_per_unit
            lg.debug(f"  Linear/Spot Calc: {risk_amount_quote} / {value_change_per_unit} = {calculated_size}")
        else: # Inverse
            if entry_price <= 0 or initial_stop_loss_price <= 0: lg.error(f"Sizing failed ({symbol}, Inv): Entry/SL zero/negative."); return None
            inverse_factor = abs( (Decimal('1') / entry_price) - (Decimal('1') / initial_stop_loss_price) )
            if inverse_factor <= Decimal('1e-18'): lg.error(f"Sizing failed ({symbol}, Inv): Inverse factor near zero."); return None
            risk_per_contract = contract_size * inverse_factor
            if risk_per_contract <= Decimal('1e-18'): lg.error(f"Sizing failed ({symbol}, Inv): Risk per contract near zero."); return None
            calculated_size = risk_amount_quote / risk_per_contract
            lg.debug(f"  Inverse Calc: {risk_amount_quote} / {risk_per_contract} = {calculated_size}")
    except (InvalidOperation, OverflowError, ZeroDivisionError) as calc_err: lg.error(f"Sizing failed ({symbol}): Calc error: {calc_err}."); return None
    if calculated_size <= Decimal('0'): lg.error(f"Sizing failed ({symbol}): Initial size zero/negative ({calculated_size.normalize()})."); return None
    lg.info(f"  Initial Calculated Size ({symbol}) = {calculated_size.normalize()} {size_unit}")

    adjusted_size = calculated_size
    def estimate_cost(size: Decimal, price: Decimal) -> Optional[Decimal]:
        if price <= 0 or size <= 0: return None
        try: return (size * price * contract_size) if not is_inverse else (size * contract_size) / price
        except Exception: return None

    if min_amount > 0 and adjusted_size < min_amount: lg.warning(f"{NEON_YELLOW}Sizing ({symbol}): Calc size {adjusted_size.normalize()} < min {min_amount.normalize()}. Adjusting UP.{RESET}"); adjusted_size = min_amount
    if max_amount < Decimal('inf') and adjusted_size > max_amount: lg.warning(f"{NEON_YELLOW}Sizing ({symbol}): Calc size {adjusted_size.normalize()} > max {max_amount.normalize()}. Adjusting DOWN.{RESET}"); adjusted_size = max_amount
    lg.debug(f"  Size after Amount Limits ({symbol}): {adjusted_size.normalize()} {size_unit}")

    cost_adj_applied = False
    est_cost = estimate_cost(adjusted_size, entry_price)
    if est_cost is not None:
        lg.debug(f"  Estimated Cost (after amount limits, {symbol}): {est_cost.normalize()} {quote_currency}")
        if min_cost > 0 and est_cost < min_cost:
            lg.warning(f"{NEON_YELLOW}Sizing ({symbol}): Est cost {est_cost.normalize()} < min cost {min_cost.normalize()}. Increasing size.{RESET}")
            try:
                req_size = (min_cost / (entry_price * contract_size)) if not is_inverse else (min_cost * entry_price / contract_size)
                if req_size <= 0: raise ValueError("Invalid required size for min cost")
                lg.info(f"  Size required for min cost ({symbol}): {req_size.normalize()} {size_unit}")
                if max_amount < Decimal('inf') and req_size > max_amount: lg.error(f"{NEON_RED}Sizing failed ({symbol}): Cannot meet min cost ({min_cost.normalize()}) without exceeding max amount ({max_amount.normalize()}).{RESET}"); return None
                adjusted_size = max(min_amount, req_size); cost_adj_applied = True
            except Exception as cost_calc_err: lg.error(f"{NEON_RED}Sizing failed ({symbol}): Failed min cost size calc: {cost_calc_err}.{RESET}"); return None
        elif max_cost < Decimal('inf') and est_cost > max_cost:
            lg.warning(f"{NEON_YELLOW}Sizing ({symbol}): Est cost {est_cost.normalize()} > max cost {max_cost.normalize()}. Reducing size.{RESET}")
            try:
                max_size = (max_cost / (entry_price * contract_size)) if not is_inverse else (max_cost * entry_price / contract_size)
                if max_size <= 0: raise ValueError("Invalid max size for max cost")
                lg.info(f"  Max size allowed by max cost ({symbol}): {max_size.normalize()} {size_unit}")
                adjusted_size = max(min_amount, min(adjusted_size, max_size)); cost_adj_applied = True
            except Exception as cost_calc_err: lg.error(f"{NEON_RED}Sizing failed ({symbol}): Failed max cost size calc: {cost_calc_err}.{RESET}"); return None
    elif min_cost > 0 or max_cost < Decimal('inf'): lg.warning(f"Could not estimate cost ({symbol}) for limit check.")
    if cost_adj_applied: lg.info(f"  Size after Cost Limits ({symbol}): {adjusted_size.normalize()} {size_unit}")

    final_size = adjusted_size
    try:
        if amount_step <= 0: raise ValueError("Amount step zero/negative.")
        final_size = (adjusted_size / amount_step).quantize(Decimal('1'), ROUND_DOWN) * amount_step
        if final_size != adjusted_size: lg.info(f"Applied amount precision ({symbol}, Rounded DOWN to {amount_step.normalize()}): {adjusted_size.normalize()} -> {final_size.normalize()} {size_unit}")
    except Exception as fmt_err: lg.error(f"{NEON_RED}Error applying amount precision ({symbol}): {fmt_err}. Using unrounded: {final_size.normalize()}{RESET}")

    if final_size <= Decimal('0'): lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final size zero/negative ({final_size.normalize()}).{RESET}"); return None
    if min_amount > 0 and final_size < min_amount: lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final size {final_size.normalize()} < min amount {min_amount.normalize()} after precision.{RESET}"); return None
    if max_amount < Decimal('inf') and final_size > max_amount: lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final size {final_size.normalize()} > max amount {max_amount.normalize()} after precision.{RESET}"); return None

    final_cost = estimate_cost(final_size, entry_price)
    if final_cost is not None:
        lg.debug(f"  Final Estimated Cost ({symbol}): {final_cost.normalize()} {quote_currency}")
        if min_cost > 0 and final_cost < min_cost:
             lg.warning(f"{NEON_YELLOW}Sizing ({symbol}): Final cost {final_cost.normalize()} < min cost {min_cost.normalize()} after rounding.{RESET}")
             try:
                 next_size = final_size + amount_step; next_cost = estimate_cost(next_size, entry_price)
                 if next_cost is not None:
                     can_bump = (next_cost >= min_cost) and (max_amount == Decimal('inf') or next_size <= max_amount) and (max_cost == Decimal('inf') or next_cost <= max_cost)
                     if can_bump: lg.info(f"{NEON_YELLOW}Bumping final size ({symbol}) up one step to {next_size.normalize()} for min cost.{RESET}"); final_size = next_size; final_cost = estimate_cost(final_size, entry_price); lg.debug(f"  Final Cost after bump: {final_cost.normalize() if final_cost else 'N/A'}")
                     else: lg.error(f"{NEON_RED}Sizing failed ({symbol}): Cannot meet min cost even bumping size due to other limits.{RESET}"); return None
                 else: lg.error(f"{NEON_RED}Sizing failed ({symbol}): Could not estimate cost for bumped size.{RESET}"); return None
             except Exception as bump_err: lg.error(f"{NEON_RED}Sizing failed ({symbol}): Error bumping size: {bump_err}.{RESET}"); return None
        elif max_cost < Decimal('inf') and final_cost > max_cost: lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final cost {final_cost.normalize()} > max cost {max_cost.normalize()} after precision.{RESET}"); return None
    elif min_cost > 0: lg.warning(f"Could not perform final cost check ({symbol}) after precision.")

    lg.info(f"{NEON_GREEN}{BRIGHT}>>> Final Calculated Position Size ({symbol}): {final_size.normalize()} {size_unit} <<< {RESET}")
    lg.info(f"{BRIGHT}--- End Position Sizing ({symbol}) ---{RESET}")
    return final_size

def cancel_order(exchange: ccxt.Exchange, order_id: str, symbol: str, logger: logging.Logger) -> bool:
    """Cancels an order by ID with retries."""
    lg = logger
    attempts = 0
    last_exception = None
    lg.info(f"Attempting to cancel order ID: {order_id} for {symbol}...")
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Cancel order attempt {attempts + 1} for ID {order_id} ({symbol})...")
            # Bybit V5 might need symbol even for cancel by ID
            params = {}
            if 'bybit' in exchange.id.lower():
                try:
                    market = exchange.market(symbol)
                    params['category'] = 'linear' if market.get('linear') else 'inverse' if market.get('inverse') else 'spot'
                    params['symbol'] = market['id']
                except Exception as e:
                    lg.warning(f"Could not determine category/market_id for cancel order {order_id} ({symbol}): {e}")

            exchange.cancel_order(order_id, symbol, params=params)
            lg.info(f"{NEON_GREEN}Successfully cancelled order ID: {order_id} for {symbol}.{RESET}")
            return True
        except ccxt.OrderNotFound:
            lg.warning(f"{NEON_YELLOW}Order ID {order_id} ({symbol}) not found. Already cancelled or filled? Treating as success.{RESET}")
            return True # Order doesn't exist, cancellation goal achieved
        except ccxt.NetworkError as e: last_exception = e; lg.warning(f"{NEON_YELLOW}Network error cancelling order {order_id} ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e: last_exception = e; wait_time = RETRY_DELAY_SECONDS * 2; lg.warning(f"{NEON_YELLOW}Rate limit cancelling order {order_id} ({symbol}): {e}. Pausing {wait_time}s...{RESET}"); time.sleep(wait_time); continue
        except ccxt.ExchangeError as e: last_exception = e; lg.error(f"{NEON_RED}Exchange error cancelling order {order_id} ({symbol}): {e}{RESET}") # Assume retryable for now
        except Exception as e: last_exception = e; lg.error(f"{NEON_RED}Unexpected error cancelling order {order_id} ({symbol}): {e}{RESET}", exc_info=True); return False # Non-retryable
        attempts += 1
        if attempts <= MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS * attempts)
    lg.error(f"{NEON_RED}Failed to cancel order ID {order_id} ({symbol}) after {MAX_API_RETRIES + 1} attempts. Last echo: {last_exception}{RESET}"); return False

def place_trade(exchange: ccxt.Exchange, symbol: str, trade_signal: str, position_size: Decimal, market_info: MarketInfo, logger: logging.Logger, reduce_only: bool = False, params: Optional[Dict] = None) -> Optional[Dict]:
    """Places a market order (buy or sell)."""
    lg = logger; side_map = {"BUY": "buy", "SELL": "sell", "EXIT_SHORT": "buy", "EXIT_LONG": "sell"}; side = side_map.get(trade_signal.upper())
    if side is None: lg.error(f"Invalid trade signal '{trade_signal}' ({symbol})."); return None
    if not isinstance(position_size, Decimal) or position_size <= Decimal('0'): lg.error(f"Invalid position size '{position_size}' ({symbol})."); return None
    order_type = 'market'; is_contract = market_info.get('is_contract', False); base_currency = market_info.get('base', 'BASE')
    size_unit = base_currency if market_info.get('spot', False) else "Contracts"; action_desc = "Close/Reduce" if reduce_only else "Open/Increase"; market_id = market_info['id']
    try: amount_float = float(position_size); assert amount_float > 1e-15
    except Exception as float_err: lg.error(f"Failed to convert size {position_size.normalize()} ({symbol}) to float: {float_err}"); return None

    order_args = {'symbol': market_id, 'type': order_type, 'side': side, 'amount': amount_float}; order_params = {}
    if 'bybit' in exchange.id.lower() and is_contract:
        try:
            category = market_info.get('contract_type_str', 'Linear').lower(); assert category in ['linear', 'inverse']
            order_params = {'category': category, 'positionIdx': 0}
            if reduce_only: order_params['reduceOnly'] = True; order_params['timeInForce'] = 'IOC'
            lg.debug(f"Using Bybit V5 order params ({symbol}): {order_params}")
        except Exception as e: lg.error(f"Failed to set Bybit V5 params ({symbol}): {e}. Order might fail.")
    if params: order_params.update(params)
    if order_params: order_args['params'] = order_params

    lg.info(f"{BRIGHT}===> Attempting {action_desc} | {side.upper()} {order_type.upper()} Order | {symbol} | Size: {position_size.normalize()} {size_unit} <==={RESET}")
    if order_params: lg.debug(f"  with Params ({symbol}): {order_params}")

    attempts = 0; last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Executing exchange.create_order ({symbol}, Attempt {attempts + 1})...")
            order_result = exchange.create_order(**order_args)
            order_id = order_result.get('id', 'N/A'); status = order_result.get('status', 'N/A')
            avg_price = _safe_market_decimal(order_result.get('average'), 'order.avg', True)
            filled = _safe_market_decimal(order_result.get('filled'), 'order.filled', True)
            log_msg = f"{NEON_GREEN}{action_desc} Order Placed!{RESET} ID: {order_id}, Status: {status}"
            if avg_price: log_msg += f", AvgFill: ~{avg_price.normalize()}"
            if filled: log_msg += f", Filled: {filled.normalize()}"
            lg.info(log_msg); lg.debug(f"Full order result ({symbol}): {order_result}"); return order_result
        except ccxt.InsufficientFunds as e: last_exception = e; lg.error(f"{NEON_RED}Order Failed ({symbol} {action_desc}): Insufficient funds. {e}{RESET}"); return None
        except ccxt.InvalidOrder as e:
            last_exception = e; lg.error(f"{NEON_RED}Order Failed ({symbol} {action_desc}): Invalid order params. {e}{RESET}"); lg.error(f"  Args: {order_args}")
            err_lower = str(e).lower()
            if "minimum" in err_lower or "too small" in err_lower: lg.error(f"  >> Hint: Check size/cost vs market mins (MinAmt: {market_info.get('min_amount_decimal')}, MinCost: {market_info.get('min_cost_decimal')}).")
            elif "precision" in err_lower or "lot size" in err_lower: lg.error(f"  >> Hint: Check size vs amount step ({market_info.get('amount_precision_step_decimal')}).")
            elif "exceed" in err_lower or "too large" in err_lower: lg.error(f"  >> Hint: Check size/cost vs market maxs (MaxAmt: {market_info.get('max_amount_decimal')}, MaxCost: {market_info.get('max_cost_decimal')}).")
            elif "reduce only" in err_lower: lg.error(f"  >> Hint: Reduce-only failed. Check position size/direction.")
            return None
        except ccxt.ExchangeError as e:
            last_exception = e; err_code_str = ""; match = re.search(r'(retCode|ret_code)=(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE);
            if match: err_code_str = match.group(2)
            if not err_code_str: err_code_str = str(getattr(e, 'code', '') or getattr(e, 'retCode', ''))
            lg.error(f"{NEON_RED}Order Failed ({symbol} {action_desc}): Exchange rift. {e} (Code: {err_code_str}){RESET}")
            fatal_codes = ['10001','10004','110007','110013','110014','110017','110025','110040','30086','3303001','3303005','3400060','3400088']
            fatal_msgs = ["invalid parameter", "precision", "exceed limit", "risk limit", "invalid symbol", "reduce only check failed", "lot size"]
            if err_code_str in fatal_codes or any(msg in str(e).lower() for msg in fatal_msgs): lg.error(f"{NEON_RED} >> Hint: NON-RETRYABLE order error ({symbol}).{RESET}"); return None
            if attempts >= MAX_API_RETRIES: lg.error(f"{NEON_RED}Max retries for ExchangeError placing order ({symbol}).{RESET}"); return None
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e: last_exception = e; lg.warning(f"{NEON_YELLOW}Aetheric disturbance placing order ({symbol}): {e}. Retry {attempts + 1}...{RESET}"); if attempts >= MAX_API_RETRIES: lg.error(f"{NEON_RED}Max retries for NetworkError placing order ({symbol}).{RESET}"); return None
        except ccxt.RateLimitExceeded as e: last_exception = e; wait_time = RETRY_DELAY_SECONDS * 3; lg.warning(f"{NEON_YELLOW}Rate limit placing order ({symbol}): {e}. Pausing {wait_time}s...{RESET}"); time.sleep(wait_time); continue
        except ccxt.AuthenticationError as e: last_exception = e; lg.critical(f"{NEON_RED}Auth ritual failed placing order ({symbol}): {e}. Stopping.{RESET}"); return None
        except Exception as e: last_exception = e; lg.error(f"{NEON_RED}Unexpected vortex placing order ({symbol}): {e}{RESET}", exc_info=True); return None
        attempts += 1
        if attempts <= MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS * attempts)
    lg.error(f"{NEON_RED}Failed to place {action_desc} order ({symbol}) after {MAX_API_RETRIES + 1} attempts. Last echo: {last_exception}{RESET}"); return None

def _set_position_protection(exchange: ccxt.Exchange, symbol: str, market_info: MarketInfo, position_info: PositionInfo, logger: logging.Logger, stop_loss_price: Optional[Decimal] = None, take_profit_price: Optional[Decimal] = None, trailing_stop_distance: Optional[Decimal] = None, tsl_activation_price: Optional[Decimal] = None) -> bool:
    """Internal helper: Sets SL/TP/TSL using Bybit's V5 private API."""
    lg = logger; endpoint = '/v5/position/set-trading-stop'
    if not market_info.get('is_contract', False): lg.warning(f"Protection skipped ({symbol}): Not contract."); return False
    if not position_info: lg.error(f"Protection failed ({symbol}): Missing position info."); return False
    pos_side = position_info.get('side'); entry_price_any = position_info.get('entryPrice')
    if pos_side not in ['long', 'short']: lg.error(f"Protection failed ({symbol}): Invalid side '{pos_side}'."); return False
    try: assert entry_price_any is not None; entry_price = Decimal(str(entry_price_any)); assert entry_price > 0
    except Exception as e: lg.error(f"Protection failed ({symbol}): Invalid entry price '{entry_price_any}': {e}"); return False
    try: price_tick = market_info['price_precision_step_decimal']; assert price_tick and price_tick > 0
    except Exception as e: lg.error(f"Protection failed ({symbol}): Invalid price precision: {e}"); return False

    params_to_set: Dict[str, Any] = {}; log_parts: List[str] = [f"{BRIGHT}Attempting protection set ({symbol} {pos_side.upper()} @ {entry_price.normalize()}):{RESET}"]
    any_requested = False; set_tsl_active = False

    try:
        def format_param(price_decimal: Optional[Decimal], param_name: str) -> Optional[str]:
            if price_decimal is None: return None
            if price_decimal == 0: return "0" # Allow clearing
            if price_decimal < 0: lg.warning(f"Invalid negative price {price_decimal.normalize()} for {param_name} ({symbol}). Ignoring."); return None
            fmt_str = _format_price(exchange, market_info['symbol'], price_decimal)
            if fmt_str: return fmt_str
            else: lg.error(f"Failed to format {param_name} ({symbol}) value {price_decimal.normalize()}."); return None

        # TSL
        if isinstance(trailing_stop_distance, Decimal):
            any_requested = True
            if trailing_stop_distance > 0:
                min_dist = max(trailing_stop_distance, price_tick)
                if not isinstance(tsl_activation_price, Decimal) or tsl_activation_price <= 0: lg.error(f"TSL failed ({symbol}): Valid activation price required for TSL distance > 0.")
                else:
                    is_valid_act = (pos_side == 'long' and tsl_activation_price > entry_price) or (pos_side == 'short' and tsl_activation_price < entry_price)
                    if not is_valid_act: lg.error(f"TSL failed ({symbol}): Activation {tsl_activation_price.normalize()} invalid vs entry {entry_price.normalize()} for {pos_side}.")
                    else:
                        fmt_dist = format_param(min_dist, "TSL Distance"); fmt_act = format_param(tsl_activation_price, "TSL Activation")
                        if fmt_dist and fmt_act: params_to_set['trailingStop'] = fmt_dist; params_to_set['activePrice'] = fmt_act; log_parts.append(f"  - Setting TSL: Dist={fmt_dist}, Act={fmt_act}"); set_tsl_active = True
                        else: lg.error(f"TSL failed ({symbol}): Could not format params (Dist: {fmt_dist}, Act: {fmt_act}).")
            elif trailing_stop_distance == 0: params_to_set['trailingStop'] = "0"; log_parts.append("  - Clearing TSL")
            else: lg.warning(f"Invalid negative TSL distance ({trailing_stop_distance.normalize()}) for {symbol}. Ignoring.")

        # SL (ignored if TSL active)
        if not set_tsl_active and isinstance(stop_loss_price, Decimal):
            any_requested = True
            if stop_loss_price > 0:
                is_valid_sl = (pos_side == 'long' and stop_loss_price < entry_price) or (pos_side == 'short' and stop_loss_price > entry_price)
                if not is_valid_sl: lg.error(f"SL failed ({symbol}): SL price {stop_loss_price.normalize()} invalid vs entry {entry_price.normalize()} for {pos_side}.")
                else:
                    fmt_sl = format_param(stop_loss_price, "Stop Loss")
                    if fmt_sl: params_to_set['stopLoss'] = fmt_sl; log_parts.append(f"  - Setting SL: {fmt_sl}")
                    else: lg.error(f"SL failed ({symbol}): Could not format SL price {stop_loss_price.normalize()}.")
            elif stop_loss_price == 0: params_to_set['stopLoss'] = "0"; log_parts.append("  - Clearing SL")

        # TP
        if isinstance(take_profit_price, Decimal):
            any_requested = True
            if take_profit_price > 0:
                is_valid_tp = (pos_side == 'long' and take_profit_price > entry_price) or (pos_side == 'short' and take_profit_price < entry_price)
                if not is_valid_tp: lg.error(f"TP failed ({symbol}): TP price {take_profit_price.normalize()} invalid vs entry {entry_price.normalize()} for {pos_side}.")
                else:
                    fmt_tp = format_param(take_profit_price, "Take Profit")
                    if fmt_tp: params_to_set['takeProfit'] = fmt_tp; log_parts.append(f"  - Setting TP: {fmt_tp}")
                    else: lg.error(f"TP failed ({symbol}): Could not format TP price {take_profit_price.normalize()}.")
            elif take_profit_price == 0: params_to_set['takeProfit'] = "0"; log_parts.append("  - Clearing TP")

    except Exception as validation_err: lg.error(f"Unexpected error during protection validation ({symbol}): {validation_err}", exc_info=True); return False

    if not params_to_set:
        if any_requested: lg.warning(f"Protection skipped ({symbol}): No valid parameters after validation."); return False
        else: lg.debug(f"No protection changes requested ({symbol}). Skipping API."); return True

    params_to_set['symbol'] = market_info['id']; params_to_set['category'] = market_info.get('contract_type_str', 'Linear').lower(); params_to_set['positionIdx'] = 0
    lg.info("\n".join(log_parts)); lg.debug(f"  Final API params for {endpoint} ({symbol}): {params_to_set}")

    attempts = 0; last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Executing private_post {endpoint} ({symbol}, Attempt {attempts + 1})...")
            response = exchange.private_post(endpoint, params=params_to_set); lg.debug(f"Raw response from {endpoint} ({symbol}): {response}")
            ret_code = response.get('retCode'); ret_msg = response.get('retMsg', 'Unknown msg')
            if ret_code == 0: lg.info(f"{NEON_GREEN}Protection set/updated successfully ({symbol}, Code: 0).{RESET}"); return True
            else: raise ccxt.ExchangeError(f"Bybit API error setting protection ({symbol}): {ret_msg} (Code: {ret_code})")
        except ccxt.ExchangeError as e:
            last_exception = e; err_code_str = ""; match = re.search(r'(retCode|ret_code)=(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE);
            if match: err_code_str = match.group(2)
            if not err_code_str: err_code_str = str(getattr(e, 'code', '') or getattr(e, 'retCode', ''))
            lg.error(f"{NEON_RED}Protection setting failed ({symbol}): Exchange rift. {e} (Code: {err_code_str}){RESET}")
            fatal_codes = ['10001','110013','110025','110043','3400048','3400051','3400052','3400070','3400071','3400072','3400073']
            fatal_msgs = ["parameter error", "invalid price", "position status", "cannot be the same", "activation price", "distance invalid"]
            if err_code_str in fatal_codes or any(msg in str(e).lower() for msg in fatal_msgs): lg.error(f"{NEON_RED} >> Hint: NON-RETRYABLE protection error ({symbol}).{RESET}"); return False
            if attempts >= MAX_API_RETRIES: lg.error(f"{NEON_RED}Max retries for ExchangeError setting protection ({symbol}).{RESET}"); return False
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e: last_exception = e; lg.warning(f"{NEON_YELLOW}Aetheric disturbance setting protection ({symbol}): {e}. Retry {attempts + 1}...{RESET}"); if attempts >= MAX_API_RETRIES: lg.error(f"{NEON_RED}Max retries for NetworkError setting protection ({symbol}).{RESET}"); return False
        except ccxt.RateLimitExceeded as e: last_exception = e; wait_time = RETRY_DELAY_SECONDS * 3; lg.warning(f"{NEON_YELLOW}Rate limit setting protection ({symbol}): {e}. Pausing {wait_time}s...{RESET}"); time.sleep(wait_time); continue
        except ccxt.AuthenticationError as e: last_exception = e; lg.critical(f"{NEON_RED}Auth ritual failed setting protection ({symbol}): {e}. Stopping.{RESET}"); return False
        except Exception as e: last_exception = e; lg.error(f"{NEON_RED}Unexpected vortex setting protection ({symbol}): {e}{RESET}", exc_info=True); return False
        attempts += 1
        if attempts <= MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS * attempts)
    lg.error(f"{NEON_RED}Failed to set protection ({symbol}) after {MAX_API_RETRIES + 1} attempts. Last echo: {last_exception}{RESET}"); return False


# --- Strategy Implementation (Volumatic Trend + Pivot Order Blocks) ---
def find_pivots(df: pd.DataFrame, left: int, right: int, use_wicks: bool) -> Tuple[pd.Series, pd.Series]:
    """Identifies Pivot Highs and Lows based on lookback periods."""
    high_col = 'high' if use_wicks else 'close' # Use high/low for wicks, close/open for body pivots? Let's stick to high/low for simplicity now.
    low_col = 'low' if use_wicks else 'close'   # Or use 'open' if close is lower? Wick logic is simpler.

    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex for pivot calculation.")

    # Calculate rolling max/min over the lookback window (left + current + right)
    window_size = left + right + 1
    # Shift required to align window correctly for pivot identification
    # We need to compare candle 'i' with 'left' candles before and 'right' candles after.
    # rolling(center=False) looks back, so we need to shift the result forward by 'right' periods.
    rolling_max = df[high_col].rolling(window=window_size, closed='both').max().shift(-right)
    rolling_min = df[low_col].rolling(window=window_size, closed='both').min().shift(-right)

    # A pivot high occurs if the current high is the maximum in the window
    pivot_highs = df[high_col] == rolling_max
    # A pivot low occurs if the current low is the minimum in the window
    pivot_lows = df[low_col] == rolling_min

    # Filter out consecutive pivots (optional, but often desired)
    # Keep only the first pivot in a consecutive sequence
    # ph_filtered = pivot_highs & (~pivot_highs.shift(1).fillna(False))
    # pl_filtered = pivot_lows & (~pivot_lows.shift(1).fillna(False))
    # Re-evaluate: Simple consecutive filtering might remove valid pivots if price consolidates.
    # Let's return the raw pivots for now, OB logic can handle overlaps if needed.

    return pivot_highs, pivot_lows

def calculate_strategy_signals(df: pd.DataFrame, config: Dict[str, Any], logger: logging.Logger) -> StrategyAnalysisResults:
    """Calculates Volumatic Trend, Pivots, Order Blocks, and generates signals."""
    lg = logger
    lg.debug(f"Calculating strategy signals for DataFrame with {len(df)} candles...")
    # --- Parameter Extraction ---
    try:
        sp = config['strategy_params']
        vt_len = sp['vt_length']; vt_atr_period = sp['vt_atr_period']
        vt_vol_ema = sp['vt_vol_ema_length']; vt_atr_mult = Decimal(str(sp['vt_atr_multiplier']))
        ob_source = sp['ob_source']; ph_left, ph_right = sp['ph_left'], sp['ph_right']
        pl_left, pl_right = sp['pl_left'], sp['pl_right']; ob_extend = sp['ob_extend']
        ob_max_boxes = sp['ob_max_boxes']; ob_entry_prox_factor = Decimal(str(sp['ob_entry_proximity_factor']))
        ob_exit_prox_factor = Decimal(str(sp['ob_exit_proximity_factor']))
        use_wicks = ob_source.lower() == "wicks"
    except (KeyError, ValueError, InvalidOperation, TypeError) as e:
        lg.error(f"Strategy calc failed: Error accessing params: {e}"); return StrategyAnalysisResults(dataframe=df, last_close=Decimal('NaN'), current_trend_up=None, trend_just_changed=False, active_bull_boxes=[], active_bear_boxes=[], vol_norm_int=None, atr=None, upper_band=None, lower_band=None, signal="NONE")

    # --- Indicator Calculations ---
    df_calc = df.copy() # Work on a copy
    try:
        # Convert relevant columns to float for pandas_ta, store original Decimals
        float_cols = ['open', 'high', 'low', 'close', 'volume']
        df_float = pd.DataFrame(index=df_calc.index)
        for col in float_cols:
            if col in df_calc.columns:
                # Handle potential Decimal('NaN') before converting
                df_float[col] = pd.to_numeric(df_calc[col].apply(lambda x: x if x.is_finite() else np.nan), errors='coerce')
            else: df_float[col] = np.nan # Ensure column exists even if missing in input

        # 1. Volumatic Trend
        ema_col = f'EMA_{vt_len}'
        df_float[ema_col] = ta.ema(df_float['close'], length=vt_len)
        df_float['ATR'] = ta.atr(df_float['high'], df_float['low'], df_float['close'], length=vt_atr_period)
        df_float['VT_UpperBand'] = df_float[ema_col] + (df_float['ATR'] * float(vt_atr_mult))
        df_float['VT_LowerBand'] = df_float[ema_col] - (df_float['ATR'] * float(vt_atr_mult))
        if 'volume' in df_float.columns and df_float['volume'].sum() > 0:
            vol_ema_series = ta.ema(df_float['volume'], length=vt_vol_ema)
            df_float['VolNorm'] = (df_float['volume'] / vol_ema_series.replace(0, np.nan)) * 100
            df_float['VolNorm'].fillna(0, inplace=True)
            df_float['VolNormInt'] = df_float['VolNorm'].clip(0, 500).astype(int) # Example cap
        else: df_float['VolNormInt'] = 0
        df_float['TrendUp'] = df_float['close'] > df_float[ema_col]
        # Ensure TrendUp has valid booleans before diff(), handle initial NaNs
        df_float['TrendUp'] = df_float['TrendUp'].astype('boolean').fillna(method='bfill') # Forward fill first, then backfill? Or just backfill?
        df_float['TrendChanged'] = df_float['TrendUp'].diff() != False # Compare diff to False to handle NaN->True/False transitions

        # 2. Pivots
        df_calc['PivotHigh'], df_calc['PivotLow'] = find_pivots(df_calc, ph_left, ph_right, use_wicks)

        # --- Convert calculated float indicators back to Decimal in df_calc ---
        calculated_float_cols = [ema_col, 'ATR', 'VT_UpperBand', 'VT_LowerBand']
        for col in calculated_float_cols:
            if col in df_float.columns:
                df_calc[col] = df_float[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN'))
        # Add boolean/int columns
        if 'TrendUp' in df_float.columns: df_calc['TrendUp'] = df_float['TrendUp'].astype('boolean') # Keep as nullable boolean
        if 'TrendChanged' in df_float.columns: df_calc['TrendChanged'] = df_float['TrendChanged']
        if 'VolNormInt' in df_float.columns: df_calc['VolNormInt'] = df_float['VolNormInt']

    except Exception as e:
        lg.error(f"Error calculating indicators: {e}", exc_info=True)
        return StrategyAnalysisResults(dataframe=df, last_close=Decimal('NaN'), current_trend_up=None, trend_just_changed=False, active_bull_boxes=[], active_bear_boxes=[], vol_norm_int=None, atr=None, upper_band=None, lower_band=None, signal="NONE")

    # 3. Order Block Identification & Management
    active_bull_boxes: List[OrderBlock] = []
    active_bear_boxes: List[OrderBlock] = []
    try:
        pivot_high_indices = df_calc.index[df_calc['PivotHigh']]
        pivot_low_indices = df_calc.index[df_calc['PivotLow']]
        latest_ts = df_calc.index[-1]

        # Process Bearish OBs from Pivot Highs
        for ph_ts in pivot_high_indices:
            if len(active_bear_boxes) >= ob_max_boxes: break # Limit boxes
            pivot_candle = df_calc.loc[ph_ts]
            # Define OB based on source (Wicks or Body)
            ob_top = pivot_candle['high']
            ob_bottom = min(pivot_candle['open'], pivot_candle['close']) if not use_wicks else pivot_candle['low']
            # Check for valid OB (top > bottom)
            if ob_top <= ob_bottom: continue
            # Create OB
            ob_id = f"B_{ph_ts.value // 10**9}" # Unique ID based on timestamp
            new_ob = OrderBlock(id=ob_id, type='bear', timestamp=ph_ts, top=ob_top, bottom=ob_bottom, active=True, violated=False, violation_ts=None, extended_to_ts=ph_ts)
            active_bear_boxes.append(new_ob)
        # Sort by timestamp descending to easily manage max boxes (remove oldest if needed)
        active_bear_boxes.sort(key=lambda x: x['timestamp'], reverse=True)
        active_bear_boxes = active_bear_boxes[:ob_max_boxes]

        # Process Bullish OBs from Pivot Lows
        for pl_ts in pivot_low_indices:
            if len(active_bull_boxes) >= ob_max_boxes: break
            pivot_candle = df_calc.loc[pl_ts]
            ob_bottom = pivot_candle['low']
            ob_top = max(pivot_candle['open'], pivot_candle['close']) if not use_wicks else pivot_candle['high']
            if ob_top <= ob_bottom: continue
            ob_id = f"L_{pl_ts.value // 10**9}"
            new_ob = OrderBlock(id=ob_id, type='bull', timestamp=pl_ts, top=ob_top, bottom=ob_bottom, active=True, violated=False, violation_ts=None, extended_to_ts=pl_ts)
            active_bull_boxes.append(new_ob)
        active_bull_boxes.sort(key=lambda x: x['timestamp'], reverse=True)
        active_bull_boxes = active_bull_boxes[:ob_max_boxes]

        # Check Violations and Extend OBs
        for idx, candle in df_calc.iterrows():
            candle_close = candle['close']
            # Check Bear OB violations
            for ob in active_bear_boxes:
                if ob['active'] and idx > ob['timestamp']: # Only check candles after OB formation
                    if candle_close > ob['top']: # Violation condition
                        ob['active'] = False
                        ob['violated'] = True
                        ob['violation_ts'] = idx
                    elif ob_extend and not ob['violated']:
                         ob['extended_to_ts'] = idx # Extend active box
            # Check Bull OB violations
            for ob in active_bull_boxes:
                 if ob['active'] and idx > ob['timestamp']:
                     if candle_close < ob['bottom']: # Violation condition
                         ob['active'] = False
                         ob['violated'] = True
                         ob['violation_ts'] = idx
                     elif ob_extend and not ob['violated']:
                          ob['extended_to_ts'] = idx

        # Filter out inactive boxes for the final result
        active_bull_boxes = [ob for ob in active_bull_boxes if ob['active']]
        active_bear_boxes = [ob for ob in active_bear_boxes if ob['active']]
        lg.debug(f"OB Analysis: Found {len(active_bull_boxes)} active Bull OBs, {len(active_bear_boxes)} active Bear OBs.")

    except Exception as e:
        lg.error(f"Error during Pivot/OB processing: {e}", exc_info=True)
        # Continue without OBs if processing fails

    # --- Extract Last Values ---
    last_row = df_calc.iloc[-1] if not df_calc.empty else None
    last_close = last_row['close'] if last_row is not None and pd.notna(last_row['close']) else Decimal('NaN')
    current_trend_up = pd.NA if last_row is None or 'TrendUp' not in last_row or pd.isna(last_row['TrendUp']) else bool(last_row['TrendUp'])
    trend_just_changed = False # Default
    if last_row is not None and 'TrendChanged' in last_row and pd.notna(last_row['TrendChanged']):
         # Check the second to last row as well to confirm the change happened *on* the last candle
         if len(df_calc) > 1:
             second_last_row = df_calc.iloc[-2]
             if 'TrendUp' in second_last_row and pd.notna(second_last_row['TrendUp']):
                 trend_just_changed = bool(last_row['TrendUp']) != bool(second_last_row['TrendUp'])

    vol_norm_int = int(last_row['VolNormInt']) if last_row is not None and 'VolNormInt' in last_row and pd.notna(last_row['VolNormInt']) else None
    atr = last_row['ATR'] if last_row is not None and 'ATR' in last_row and pd.notna(last_row['ATR']) and last_row['ATR'] > 0 else None
    upper_band = last_row['VT_UpperBand'] if last_row is not None and 'VT_UpperBand' in last_row and pd.notna(last_row['VT_UpperBand']) else None
    lower_band = last_row['VT_LowerBand'] if last_row is not None and 'VT_LowerBand' in last_row and pd.notna(last_row['VT_LowerBand']) else None

    # --- Signal Generation ---
    signal = "NONE"
    if pd.notna(current_trend_up) and pd.notna(last_close):
        # Entry Signals
        if current_trend_up is True:
            for ob in active_bull_boxes: # Look for nearby Bull OB in uptrend
                # Price needs to be close to or inside the OB
                entry_threshold = ob['top'] * ob_entry_prox_factor
                if last_close <= entry_threshold and last_close >= ob['bottom']:
                    lg.debug(f"Potential BUY signal: Uptrend, price {last_close} near Bull OB {ob['id']} ({ob['bottom']}-{ob['top']})")
                    signal = "BUY"; break # Take first valid signal
        elif current_trend_up is False:
            for ob in active_bear_boxes: # Look for nearby Bear OB in downtrend
                entry_threshold = ob['bottom'] / ob_entry_prox_factor
                if last_close >= entry_threshold and last_close <= ob['top']:
                    lg.debug(f"Potential SELL signal: Downtrend, price {last_close} near Bear OB {ob['id']} ({ob['bottom']}-{ob['top']})")
                    signal = "SELL"; break

        # Exit Signals (Check only if no entry signal generated)
        if signal == "NONE":
            if trend_just_changed: # Exit on trend reversal
                if current_trend_up is False: signal = "EXIT_LONG" # Trend changed to down
                elif current_trend_up is True: signal = "EXIT_SHORT" # Trend changed to up
                lg.debug(f"Exit signal due to trend reversal: {signal}")
            else: # Check for OB violation as exit trigger
                if current_trend_up is True: # Currently in potential Long
                    for ob in active_bull_boxes: # Check if price broke below a supporting Bull OB
                        exit_threshold = ob['bottom'] * ob_exit_prox_factor
                        if last_close <= exit_threshold:
                             lg.debug(f"Potential EXIT_LONG signal: Price {last_close} violated Bull OB {ob['id']} ({ob['bottom']})")
                             signal = "EXIT_LONG"; break
                elif current_trend_up is False: # Currently in potential Short
                     for ob in active_bear_boxes: # Check if price broke above a supporting Bear OB
                         exit_threshold = ob['top'] / ob_exit_prox_factor
                         if last_close >= exit_threshold:
                              lg.debug(f"Potential EXIT_SHORT signal: Price {last_close} violated Bear OB {ob['id']} ({ob['top']})")
                              signal = "EXIT_SHORT"; break

    lg.debug(f"Strategy Calc Complete. Last Close: {last_close.normalize() if pd.notna(last_close) else 'N/A'}, TrendUp: {current_trend_up}, TrendChanged: {trend_just_changed}, VolNorm: {vol_norm_int}, ATR: {atr.normalize() if atr else 'N/A'}, Signal: {signal}")

    results = StrategyAnalysisResults(
        dataframe=df_calc, last_close=last_close, current_trend_up=current_trend_up,
        trend_just_changed=trend_just_changed, active_bull_boxes=active_bull_boxes,
        active_bear_boxes=active_bear_boxes, vol_norm_int=vol_norm_int, atr=atr,
        upper_band=upper_band, lower_band=lower_band, signal=signal
    )
    return results


# --- Trading Logic ---
def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: Dict[str, Any], logger: logging.Logger):
    """Performs a full trading cycle for a single symbol."""
    lg = logger
    lg.info(f"{Fore.MAGENTA}=== Starting Analysis Cycle for {symbol} ===")

    # --- 1. Fetch Market Info ---
    market_info = get_market_info(exchange, symbol, lg)
    if not market_info: lg.error(f"Cannot proceed ({symbol}): Failed market info."); return
    price_tick = market_info['price_precision_step_decimal']
    if not price_tick or price_tick <= 0: lg.error(f"Cannot proceed ({symbol}): Invalid price tick size."); return

    # --- 2. Fetch Kline Data ---
    timeframe_key = config.get("interval", "5"); ccxt_timeframe = CCXT_INTERVAL_MAP.get(timeframe_key)
    if not ccxt_timeframe: lg.error(f"Invalid interval '{timeframe_key}' ({symbol})."); return
    fetch_limit = config.get("fetch_limit", DEFAULT_FETCH_LIMIT)
    df_raw = fetch_klines_ccxt(exchange, symbol, ccxt_timeframe, fetch_limit, lg)
    if df_raw.empty: lg.error(f"Cannot proceed ({symbol}): Failed kline fetch."); return

    # --- 3. Calculate Strategy Signals ---
    strategy_results = calculate_strategy_signals(df_raw, config, lg)
    if strategy_results['last_close'] is None or pd.isna(strategy_results['last_close']): lg.error(f"Cannot proceed ({symbol}): Strategy calc failed."); return
    df_analyzed = strategy_results['dataframe']; last_close = strategy_results['last_close']
    current_trend_up = strategy_results['current_trend_up']; atr = strategy_results['atr']
    trade_signal = strategy_results['signal'] # Now comes from strategy
    lg.info(f"Analysis Results ({symbol}): Last Close={last_close.normalize()}, TrendUp={current_trend_up}, ATR={atr.normalize() if atr else 'N/A'}, Signal='{trade_signal}'")

    # --- 4. Check Existing Position ---
    position_info = get_open_position(exchange, symbol, lg)

    # --- 5. Manage Existing Position ---
    if position_info:
        pos_side = position_info['side']; pos_size = position_info['size_decimal']
        entry_price = _safe_market_decimal(position_info['entryPrice'], 'pos.entry', False)
        be_activated = position_info.get('be_activated', False) # Get bot's state flag
        tsl_activated = position_info.get('tsl_activated', False) # Get bot's state flag

        if not entry_price: lg.error(f"Cannot manage position ({symbol}): Invalid entry price."); return # Should not happen if position exists

        lg.info(f"{Fore.CYAN}# Managing existing {pos_side} position ({symbol})... BE Active: {be_activated}, TSL Active: {tsl_activated}{Style.RESET_ALL}")

        # Check for Exit Signal FIRST
        should_exit = (pos_side == 'long' and trade_signal == "EXIT_LONG") or \
                      (pos_side == 'short' and trade_signal == "EXIT_SHORT")

        if should_exit:
            lg.warning(f"{BRIGHT}>>> Strategy Exit Signal '{trade_signal}' detected for {pos_side} position on {symbol} <<<")
            if config.get("enable_trading", False):
                # Attempt to cancel existing SL/TP orders before market exit
                sl_order_id = position_info.get('info', {}).get('stopLossOrderId') # Example, actual field may vary
                tp_order_id = position_info.get('info', {}).get('takeProfitOrderId')
                if sl_order_id: cancel_order(exchange, sl_order_id, symbol, lg)
                if tp_order_id: cancel_order(exchange, tp_order_id, symbol, lg)
                time.sleep(1) # Brief pause after cancellation attempt

                # Place market exit order
                close_size = abs(pos_size)
                order_result = place_trade(exchange, symbol, trade_signal, close_size, market_info, lg, reduce_only=True)
                if order_result: lg.info(f"{NEON_GREEN}Position exit order placed successfully for {symbol}.{RESET}")
                else: lg.error(f"{NEON_RED}Failed to place position exit order for {symbol}. Position remains open.{RESET}")
            else: lg.warning(f"Trading disabled: Would place {pos_side} exit order for {symbol}.")
            return # Stop management cycle if exit signal triggered

        # --- Protection Management (BE, TSL) ---
        protection_conf = config.get('protection', {})
        enable_be = protection_conf.get('enable_break_even', False) and not be_activated # Only run if not already active
        enable_tsl = protection_conf.get('enable_trailing_stop', False) and not tsl_activated # Only run if not already active

        if (enable_be or enable_tsl) and atr is not None and atr > 0:
             current_price = fetch_current_price_ccxt(exchange, symbol, lg)
             if current_price is None: lg.warning(f"Could not fetch current price ({symbol}). Skipping BE/TSL checks.")
             else:
                 # --- Break-Even Logic ---
                 if enable_be and not tsl_activated: # Don't set BE if TSL is already active
                     be_trigger_mult = Decimal(str(protection_conf.get('break_even_trigger_atr_multiple', 1.0)))
                     be_offset_ticks = int(protection_conf.get('break_even_offset_ticks', 2))
                     profit_target_price: Optional[Decimal] = None
                     be_stop_price: Optional[Decimal] = None

                     if pos_side == 'long':
                         profit_target_price = entry_price + (atr * be_trigger_mult)
                         if current_price >= profit_target_price:
                             be_stop_price = entry_price + (price_tick * be_offset_ticks)
                             lg.info(f"BE Triggered (Long, {symbol}): Current={current_price}, Target={profit_target_price}")
                     elif pos_side == 'short':
                         profit_target_price = entry_price - (atr * be_trigger_mult)
                         if current_price <= profit_target_price:
                             be_stop_price = entry_price - (price_tick * be_offset_ticks)
                             lg.info(f"BE Triggered (Short, {symbol}): Current={current_price}, Target={profit_target_price}")

                     if be_stop_price is not None and be_stop_price > 0:
                         # Check if current SL is worse than BE price
                         current_sl_str = position_info.get('stopLossPrice')
                         current_sl = _safe_market_decimal(current_sl_str, 'current_sl', False) if current_sl_str else None
                         needs_update = True
                         if current_sl:
                              if pos_side == 'long' and current_sl >= be_stop_price: needs_update = False # Current SL already at or better than BE
                              if pos_side == 'short' and current_sl <= be_stop_price: needs_update = False # Current SL already at or better than BE
                         if needs_update:
                              lg.warning(f"{BRIGHT}>>> Moving SL to Break-Even for {symbol} at {be_stop_price.normalize()} <<<")
                              if config.get("enable_trading", False):
                                   protect_success = _set_position_protection(exchange, symbol, market_info, position_info, lg, stop_loss_price=be_stop_price)
                                   if protect_success: position_info['be_activated'] = True # Mark BE as done for this position instance
                                   else: lg.error(f"{NEON_RED}Failed to set Break-Even SL for {symbol}!{RESET}")
                              else: lg.warning(f"Trading disabled: Would set BE SL to {be_stop_price.normalize()} for {symbol}.")
                         else: lg.info(f"BE ({symbol}): Current SL ({current_sl.normalize() if current_sl else 'N/A'}) already at or better than calculated BE ({be_stop_price.normalize()}). No update needed.")
                     elif profit_target_price: lg.debug(f"BE not triggered ({symbol}): Price {current_price} hasn't reached target {profit_target_price.normalize()}.")

                 # --- Trailing Stop Loss Activation Logic ---
                 # We only *activate* it here. Bybit handles the trailing.
                 if enable_tsl and not position_info.get('be_activated'): # Check BE flag again, maybe it was just set
                     tsl_activation_perc = Decimal(str(protection_conf.get('trailing_stop_activation_percentage', 0.003)))
                     tsl_callback_rate = Decimal(str(protection_conf.get('trailing_stop_callback_rate', 0.005)))
                     activation_trigger_price: Optional[Decimal] = None
                     tsl_distance: Optional[Decimal] = None

                     if tsl_activation_perc >= 0 and tsl_callback_rate > 0:
                         if pos_side == 'long':
                             activation_trigger_price = entry_price * (Decimal('1') + tsl_activation_perc)
                             if current_price >= activation_trigger_price:
                                 tsl_distance = activation_trigger_price * tsl_callback_rate # Distance based on activation price
                                 lg.info(f"TSL Activation Triggered (Long, {symbol}): Current={current_price}, Target={activation_trigger_price}")
                         elif pos_side == 'short':
                             activation_trigger_price = entry_price * (Decimal('1') - tsl_activation_perc)
                             if current_price <= activation_trigger_price:
                                 tsl_distance = activation_trigger_price * tsl_callback_rate # Distance based on activation price
                                 lg.info(f"TSL Activation Triggered (Short, {symbol}): Current={current_price}, Target={activation_trigger_price}")

                         if tsl_distance is not None and tsl_distance > 0 and activation_trigger_price is not None:
                             lg.warning(f"{BRIGHT}>>> Activating Trailing Stop Loss for {symbol} | Distance: {tsl_distance.normalize()}, Activation: {activation_trigger_price.normalize()} <<<")
                             if config.get("enable_trading", False):
                                 # Set TSL using the dedicated function
                                 protect_success = _set_position_protection(
                                     exchange, symbol, market_info, position_info, lg,
                                     trailing_stop_distance=tsl_distance,
                                     tsl_activation_price=activation_trigger_price
                                 )
                                 if protect_success: position_info['tsl_activated'] = True # Mark TSL as active
                                 else: lg.error(f"{NEON_RED}Failed to activate Trailing Stop Loss for {symbol}!{RESET}")
                             else: lg.warning(f"Trading disabled: Would activate TSL for {symbol} (Dist: {tsl_distance.normalize()}, Act: {activation_trigger_price.normalize()}).")
                         elif activation_trigger_price: lg.debug(f"TSL not activated ({symbol}): Price {current_price} hasn't reached activation {activation_trigger_price.normalize()}.")
                     else: lg.warning(f"TSL skipped ({symbol}): Invalid activation percentage ({tsl_activation_perc}) or callback rate ({tsl_callback_rate}).")

        else: lg.debug(f"Skipping BE/TSL checks ({symbol}): Disabled, or ATR/Price unavailable.")

    # --- 6. Enter New Position ---
    elif trade_signal in ["BUY", "SELL"]:
        lg.info(f"{Fore.CYAN}# Evaluating potential {trade_signal} entry for {symbol}...{Style.RESET_ALL}")
        if not config.get("enable_trading", False): lg.warning(f"Trading disabled: Would evaluate {trade_signal} entry for {symbol}."); return
        if atr is None or atr <= 0: lg.error(f"Cannot enter ({symbol}): Invalid ATR ({atr})."); return

        protection_conf = config.get('protection', {}); sl_atr_mult = Decimal(str(protection_conf.get('initial_stop_loss_atr_multiple', 1.8))); tp_atr_mult = Decimal(str(protection_conf.get('initial_take_profit_atr_multiple', 0.7)))
        initial_sl_price: Optional[Decimal] = None; initial_tp_price: Optional[Decimal] = Decimal('0') # Default TP disabled
        if trade_signal == "BUY": initial_sl_price = last_close - (atr * sl_atr_mult); initial_tp_price = last_close + (atr * tp_atr_mult) if tp_atr_mult > 0 else Decimal('0')
        elif trade_signal == "SELL": initial_sl_price = last_close + (atr * sl_atr_mult); initial_tp_price = last_close - (atr * tp_atr_mult) if tp_atr_mult > 0 else Decimal('0')

        if initial_sl_price is None or initial_sl_price <= 0: lg.error(f"Cannot enter ({symbol}): Invalid SL price ({initial_sl_price})."); return
        if initial_tp_price < 0: lg.warning(f"Calculated TP ({initial_tp_price}) negative ({symbol}). Disabling TP."); initial_tp_price = Decimal('0')
        lg.info(f"Calculated Entry Protections ({symbol}): SL={initial_sl_price.normalize()}, TP={initial_tp_price.normalize() if initial_tp_price != 0 else 'Disabled'}")

        balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
        if balance is None or balance <= 0: lg.error(f"Cannot enter ({symbol}): Invalid balance."); return
        risk_per_trade = config.get("risk_per_trade", 0.01)
        position_size = calculate_position_size(balance, risk_per_trade, initial_sl_price, last_close, market_info, exchange, lg)
        if position_size is None or position_size <= 0: lg.error(f"Cannot enter ({symbol}): Position sizing failed."); return

        leverage = config.get("leverage", 0)
        if market_info.get('is_contract') and leverage > 0:
            if not set_leverage_ccxt(exchange, symbol, leverage, market_info, lg): lg.error(f"Cannot enter ({symbol}): Failed leverage set."); return

        lg.warning(f"{BRIGHT}>>> Initiating {trade_signal} entry for {symbol} | Size: {position_size.normalize()} <<<")
        order_result = place_trade(exchange, symbol, trade_signal, position_size, market_info, lg, reduce_only=False)
        if not order_result: lg.error(f"Entry order failed ({symbol}). No position opened."); return

        lg.info(f"Waiting {config.get('position_confirm_delay_seconds')}s to confirm position opening ({symbol})...")
        time.sleep(config.get('position_confirm_delay_seconds', POSITION_CONFIRM_DELAY_SECONDS))
        confirmed_position = None
        for confirm_attempt in range(MAX_API_RETRIES + 1):
             temp_pos = get_open_position(exchange, symbol, lg)
             if temp_pos:
                 expected_side = 'long' if trade_signal == "BUY" else 'short'
                 if temp_pos.get('side') == expected_side: lg.info(f"{NEON_GREEN}Position opening confirmed ({symbol} {expected_side}).{RESET}"); confirmed_position = temp_pos; break
                 else: lg.warning(f"Position found ({symbol}), but side ({temp_pos.get('side')}) != expected ({expected_side}). Retrying confirm..."); temp_pos = None
             if confirm_attempt < MAX_API_RETRIES: lg.warning(f"Position not confirmed ({symbol}, Attempt {confirm_attempt + 1}). Retrying in {RETRY_DELAY_SECONDS}s..."); time.sleep(RETRY_DELAY_SECONDS)
             else: lg.error(f"{NEON_RED}Failed to confirm position opening ({symbol}) after entry.{RESET}")

        if confirmed_position:
            lg.info(f"Setting initial protection (SL/TP) for new {symbol} position...")
            protect_success = _set_position_protection(exchange, symbol, market_info, confirmed_position, lg, stop_loss_price=initial_sl_price, take_profit_price=initial_tp_price)
            if protect_success: lg.info(f"Initial SL/TP set successfully for {symbol}.")
            else: lg.error(f"{NEON_RED}Failed to set initial SL/TP for {symbol}! Position unprotected.{RESET}")
        elif order_result: # Order placed but position not confirmed
             lg.error(f"{NEON_RED}CRITICAL: Entry order placed ({symbol}, ID: {order_result.get('id')}), but position confirmation failed. Manual check required!{RESET}")

    # --- 7. No Action ---
    elif not position_info: # Only log holding if no position exists
        lg.info(f"No open position and no entry signal for {symbol}. Holding pattern.")

    lg.info(f"{Fore.MAGENTA}=== Completed Analysis Cycle for {symbol} ===")


# --- Main Execution Loop ---
def signal_handler(sig, frame):
    """Handles shutdown signals."""
    global _shutdown_requested
    if not _shutdown_requested:
        print(f"\n{NEON_YELLOW}{Style.BRIGHT}Shutdown signal ({signal.Signals(sig).name}) received! Initiating graceful exit...{RESET}")
        _shutdown_requested = True
    else:
        print(f"{NEON_RED}Second shutdown signal received. Forcing exit.{RESET}")
        sys.exit(1)

def main():
    """Main execution function."""
    global CONFIG, _shutdown_requested
    main_logger = setup_logger("main")
    main_logger.info(f"{Fore.MAGENTA}{Style.BRIGHT}--- Pyrmethus Volumatic Bot v{BOT_VERSION} Initializing ---{Style.RESET_ALL}")

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler) # Termination signal

    exchange = initialize_exchange(main_logger)
    if not exchange: main_logger.critical("Exchange init failed. Shutting down."); sys.exit(1)

    trading_pairs = CONFIG.get("trading_pairs", [])
    valid_pairs: List[str] = []; all_symbols_valid = True
    main_logger.info(f"Validating configured trading pairs: {trading_pairs}")
    for pair_symbol in trading_pairs:
         market_info = get_market_info(exchange, pair_symbol, main_logger)
         if market_info and market_info.get('active'): valid_pairs.append(pair_symbol); main_logger.info(f" -> {NEON_GREEN}{pair_symbol} is valid.{RESET}")
         else: main_logger.error(f" -> {NEON_RED}{pair_symbol} invalid/inactive. Skipping.{RESET}"); all_symbols_valid = False
    if not valid_pairs: main_logger.critical("No valid trading pairs. Shutting down."); sys.exit(1)
    if not all_symbols_valid: main_logger.warning(f"Proceeding with valid pairs only: {valid_pairs}")

    main_logger.info(f"{Fore.CYAN}# Entering main trading cycle loop... Press Ctrl+C or send SIGTERM to gracefully exit.{Style.RESET_ALL}")
    while not _shutdown_requested:
        try:
            start_time = time.monotonic()
            main_logger.info(f"{Fore.YELLOW}--- New Trading Cycle ---{RESET}")
            # Optional: Reload config dynamically here if needed
            for symbol in valid_pairs:
                if _shutdown_requested: break # Check before processing next symbol
                symbol_logger = setup_logger(symbol)
                symbol_logger.info(f"--- Processing Symbol: {symbol} ---")
                try: analyze_and_trade_symbol(exchange, symbol, CONFIG, symbol_logger)
                except Exception as symbol_err: symbol_logger.error(f"{NEON_RED}!! Unhandled error during analysis ({symbol}): {symbol_err} !!{RESET}", exc_info=True); symbol_logger.error(f"Skipping cycle for {symbol}.")
                finally: symbol_logger.info(f"--- Finished Processing Symbol: {symbol} ---")
            if _shutdown_requested: break # Check after processing all symbols

            end_time = time.monotonic(); cycle_duration = end_time - start_time
            loop_delay = CONFIG.get("loop_delay_seconds", LOOP_DELAY_SECONDS)
            wait_time = max(0, loop_delay - cycle_duration)
            main_logger.info(f"Cycle duration: {cycle_duration:.2f}s. Waiting {wait_time:.2f}s...")
            # Sleep in smaller intervals to check shutdown flag more frequently
            for _ in range(int(wait_time)):
                 if _shutdown_requested: break
                 time.sleep(1)
            if not _shutdown_requested and wait_time % 1 > 0: time.sleep(wait_time % 1) # Sleep remainder

        except KeyboardInterrupt: # Should be caught by signal handler now, but keep as fallback
            if not _shutdown_requested: main_logger.warning(f"{NEON_YELLOW}{Style.BRIGHT}KeyboardInterrupt! Initiating graceful shutdown...{RESET}"); _shutdown_requested = True
        except Exception as loop_err:
            main_logger.critical(f"{NEON_RED}!! Unhandled critical error in main loop: {loop_err} !!{RESET}", exc_info=True)
            if _shutdown_requested: break # Don't pause if already shutting down
            main_logger.warning("Pausing loop for 60s due to critical error...")
            time.sleep(60)

    main_logger.info(f"{Fore.MAGENTA}{Style.BRIGHT}--- Pyrmethus Volumatic Bot Shutting Down ---{Style.RESET_ALL}")
    # Optional: Add cleanup logic here (e.g., close open orders if configured)
    main_logger.info("Shutdown complete. The ether settles.")

if __name__ == "__main__":
    main()

