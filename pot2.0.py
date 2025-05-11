# -*- coding: utf-8 -*-
# pylint: disable=logging-fstring-interpolation, too-many-instance-attributes, too-many-arguments, too-many-locals, too-many-public-methods, invalid-name, unused-argument, too-many-lines, unnecessary-pass, unnecessary-lambda-assignment, line-too-long, wrong-import-order, wrong-import-position
# fmt: off
#   ____        _       _   _                  _            _         _
#  |  _ \\ _   _| |_ ___| | | | __ ___   ____ _| |_ ___  ___| |_ _ __ | |__   ___ _ __ ___  _ __
#  | |_) | | | | __/ _ \\ | | |/ _` \\ \\ / / _` | __/ _ \\/ __| __| '_ \\| '_ \\ / _ \\ '_ ` _ \\| '_ \\
#  |  __/| |_| | ||  __/ | | | (_| |\\ V / (_| | ||  __/\\__ \\ |_| |_) | | | |  __/ | | | | | |_) |
#  |_|    \\__, |\\__\\___|_|_|_|\\__,_| \\_/ \\__,_|\\__\\___||___/\\__| .__/|_| |_|\\___|_| |_| |_| .__/
#         |___/                                                |_|                      |_|
# Pyrmethus - Grand Unified Scalping Spell v3.0.1 (Synergistic Weave Enhanced)
# fmt: on
"""
Pyrmethus - Termux Trading Spell (v3.0.1 Synergistic Weave Enhanced)

High-Frequency Trading Bot (Scalping) for Bybit USDT Futures.
Combines elements from ehlpyrm.py, mts3.4.3.py, plus helper snippets for multiple strategies.

Features:
- Multiple strategies, sophisticated risk management, order management.
- Position scaling.
- Notification via Termux, state persistence, and signal-based and time-based exits.
- WARNING: For educational and experimental purposes ONLY. EXTREME RISK. Use TESTNET only!
"""

# Standard Library Imports
import copy
import csv
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timedelta, timezone
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal, DivisionByZero, InvalidOperation, getcontext
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable

import pytz

# Third-party Library Imports
try:
    import ccxt
    import numpy as np
    import pandas as pd
    if not hasattr(pd, 'NA'): raise ImportError("Pandas version < 1.0 not supported for pd.NA.") # type: ignore
    import pandas_ta as ta # type: ignore[import]
    from colorama import Back, Fore, Style, init as colorama_init
    from dotenv import load_dotenv
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from retry import retry
except ImportError as e:
    missing_pkg = getattr(e, 'name', 'dependency')
    sys.stderr.write(f"\033[91mCRITICAL ERROR: Missing/Incompatible Essence: '{missing_pkg}'.\033[0m\n")
    sys.stderr.write(f"\033[91mPlease install it by running: pip install {missing_pkg}\033[0m\n")
    sys.exit(1)

# --- Constants ---
VERSION = "3.0.1" # Hot-Fix update number
STATE_FILE_NAME = f"pyrmethus_grand_unified_state_v{VERSION.split(' ')[0]}.json" # Robust filename
STATE_FILE_PATH = str(Path(__file__).resolve().parent / STATE_FILE_NAME) # Use Path for robustness
HEARTBEAT_INTERVAL_SECONDS = 60
OHLCV_LIMIT_DEFAULT = 300
LOGS_DIR = "logs"
EMERGENCY_STOP_FILE_DEFAULT = "PYRMETHUS_EMERGENCY_STOP.txt"

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
env_path = Path(__file__).resolve().parent / '.env'
if load_dotenv(dotenv_path=env_path): logging.getLogger("PreConfig").info(f"{NEON['INFO']}Secrets whispered from .env scroll: {NEON['VALUE']}{env_path}{NEON['RESET']}")
else: logging.getLogger("PreConfig").warning(f"{NEON['WARNING']}No .env scroll found. Relying on system environment variables or defaults.{NEON['RESET']}")
getcontext().prec = 28

# --- Helper Functions ---
PandasNAType = type(pd.NA)
def safe_decimal(value: Any, default_if_error: Union[Decimal, PandasNAType, None] = pd.NA) -> Union[Decimal, PandasNAType, None]:
    if value is None or pd.isna(value): return default_if_error
    try: return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError): return default_if_error

def log_format(value: Any, p: int = 4, trend: bool = False, color: Optional[str] = NEON["VALUE"]) -> str:
    c = color or NEON["VALUE"]
    if pd.isna(value) or value is None: return f"{NEON['COMMENT']}N/A{NEON['RESET']}"
    if trend: return f"{NEON['SIDE_LONG']}Upward{NEON['RESET']}" if value else (f"{NEON['SIDE_SHORT']}Downward{NEON['RESET']}" if value is False else f"{NEON['COMMENT']}Flat{NEON['RESET']}")
    if isinstance(value, Decimal): return f"{c}{value:.{p}f}{NEON['RESET']}"
    if isinstance(value, (float, int)): return f"{c}{float(value):.{p}f}{NEON['RESET']}"
    return f"{c}{str(value)}{NEON['RESET']}"

def short_oid(oid: Union[str, int, None]) -> str: return str(oid)[-6:] if oid else "N/A"

_termux_cmd_cache: Dict[str, Optional[bool]] = {}
def _cmd_exists(cmd: str) -> bool:
    if cmd not in _termux_cmd_cache:
        try: _termux_cmd_cache[cmd] = bool(subprocess.run(["which", cmd], capture_output=True, check=False, text=True).stdout)
        except FileNotFoundError: _termux_cmd_cache[cmd] = False
    if _termux_cmd_cache.get(cmd) is False: logging.getLogger("TermuxCmdCheck").warning(f"{NEON['WARNING']}Termux command '{cmd}' not found.{NEON['RESET']}") # Use a specific logger
    return _termux_cmd_cache.get(cmd, False)

def notify_termux(title: str, msg: str, nid: int = 777) -> None:
    if not CONFIG.enable_notifications or not _cmd_exists("termux-notification"): return
    try:
        subprocess.Popen(["termux-notification", "--title", json.dumps(title), "--content", json.dumps(msg), "--id", str(nid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logging.getLogger("TermuxNotify").info(f"{NEON['SUCCESS']}Termux notification '{title}' dispatched.{NEON['RESET']}") # Use a specific logger
    except Exception as e: logging.getLogger("TermuxNotify").error(f"{NEON['ERROR']}Termux notify error for '{title}': {e}{NEON['RESET']}")

# --- Enums ---
class StrategyName(str, Enum):
    DUAL_SUPERTREND_MOMENTUM = "DUAL_SUPERTREND_MOMENTUM"
    EHLERS_FISHER = "EHLERS_FISHER"
    STOCHRSI_MOMENTUM = "STOCHRSI_MOMENTUM"
    EHLERS_MA_CROSS = "EHLERS_MA_CROSS"

class VolatilityRegime(Enum): LOW = "LOW"; NORMAL = "NORMAL"; HIGH = "HIGH"
class OrderEntryType(str, Enum): MARKET = "MARKET"; LIMIT = "LIMIT"
class TSLType(str, Enum): NONE = "NONE"; EXCHANGE_NATIVE_PERCENTAGE = "EXCHANGE_NATIVE_PERCENTAGE"; CLIENT_SIDE_ATR = "CLIENT_SIDE_ATR"

# --- Configuration Class ---
class Config:
    def __init__(self) -> None:
        logger_cfg = logging.getLogger("ConfigModule")
        logger_cfg.info(f"{NEON['HEADING']}--- Summoning Configuration Runes v{VERSION} ---{NEON['RESET']}")
        # API
        self.api_key: str = self._get_env("BYBIT_API_KEY", "", required=True, secret=True)
        self.api_secret: str = self._get_env("BYBIT_API_SECRET", "", required=True, secret=True)
        # Trading Core
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT")
        self.market_type: str = self._get_env("MARKET_TYPE", "linear")
        self.position_idx: int = self._get_env("POSITION_IDX", 0, cast_type=int)
        self.interval: str = self._get_env("INTERVAL", "1m")
        self.leverage: int = self._get_env("LEVERAGE", 10, cast_type=int)
        self.sleep_seconds: int = self._get_env("SLEEP_SECONDS", 10, cast_type=int)
        self.strategy_name: StrategyName = StrategyName(self._get_env("STRATEGY_NAME", StrategyName.DUAL_SUPERTREND_MOMENTUM.value).upper())
        # Risk Management
        self.risk_per_trade_percentage: Decimal = self._get_env("RISK_PER_TRADE_PERCENTAGE", "0.005", cast_type=Decimal)
        self.max_order_usdt_amount: Decimal = self._get_env("MAX_ORDER_USDT_AMOUNT", "1000.0", cast_type=Decimal)
        self.required_margin_buffer: Decimal = self._get_env("REQUIRED_MARGIN_BUFFER", "1.05", cast_type=Decimal)
        self.max_account_margin_ratio: Decimal = self._get_env("MAX_ACCOUNT_MARGIN_RATIO", "0.5", cast_type=Decimal)
        self.enable_max_drawdown_stop: bool = self._get_env("ENABLE_MAX_DRAWDOWN_STOP", "true", cast_type=bool)
        self.max_drawdown_percent: Decimal = self._get_env("MAX_DRAWDOWN_PERCENT", "0.05", cast_type=Decimal)
        # Stop Loss
        self.atr_stop_loss_multiplier: Decimal = self._get_env("ATR_STOP_LOSS_MULTIPLIER", "1.5", cast_type=Decimal)
        # Trailing Stop Loss
        self.tsl_type: TSLType = TSLType(self._get_env("TSL_TYPE", TSLType.EXCHANGE_NATIVE_PERCENTAGE.value).upper())
        self.trailing_stop_percentage: Decimal = self._get_env("TRAILING_STOP_PERCENTAGE", "0.005", cast_type=Decimal)
        self.trailing_stop_activation_offset_percent: Decimal = self._get_env("TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT", "0.001", cast_type=Decimal)
        self.atr_tsl_activation_atr_profit: Decimal = self._get_env("ATR_TSL_ACTIVATION_ATR_PROFIT", "1.5", cast_type=Decimal)
        self.atr_tsl_trail_atr_distance: Decimal = self._get_env("ATR_TSL_TRAIL_ATR_DISTANCE", "1.0", cast_type=Decimal)
        # Dynamic Risk/SL Adjustments
        self.enable_dynamic_risk: bool = self._get_env("ENABLE_DYNAMIC_RISK", "false", cast_type=bool)
        self.dynamic_risk_min_pct: Decimal = self._get_env("DYNAMIC_RISK_MIN_PCT", "0.0025", cast_type=Decimal)
        self.dynamic_risk_max_pct: Decimal = self._get_env("DYNAMIC_RISK_MAX_PCT", "0.01", cast_type=Decimal)
        self.dynamic_risk_perf_window: int = self._get_env("DYNAMIC_RISK_PERF_WINDOW", 10, cast_type=int)
        self.enable_dynamic_atr_sl: bool = self._get_env("ENABLE_DYNAMIC_ATR_SL", "true", cast_type=bool)
        self.atr_short_term_period: int = self._get_env("ATR_SHORT_TERM_PERIOD", 7, cast_type=int)
        self.atr_long_term_period: int = self._get_env("ATR_LONG_TERM_PERIOD", 50, cast_type=int)
        self.volatility_ratio_low_threshold: Decimal = self._get_env("VOLATILITY_RATIO_LOW_THRESHOLD", "0.7", cast_type=Decimal)
        self.volatility_ratio_high_threshold: Decimal = self._get_env("VOLATILITY_RATIO_HIGH_THRESHOLD", "1.5", cast_type=Decimal)
        self.atr_sl_multiplier_low_vol: Decimal = self._get_env("ATR_SL_MULTIPLIER_LOW_VOL", "1.0", cast_type=Decimal)
        self.atr_sl_multiplier_normal_vol: Decimal = self._get_env("ATR_SL_MULTIPLIER_NORMAL_VOL", "1.3", cast_type=Decimal)
        self.atr_sl_multiplier_high_vol: Decimal = self._get_env("ATR_SL_MULTIPLIER_HIGH_VOL", "1.8", cast_type=Decimal)
        # Position Management Helpers
        self.enable_breakeven_sl: bool = self._get_env("ENABLE_BREAKEVEN_SL", "true", cast_type=bool)
        self.breakeven_profit_atr_target: Decimal = self._get_env("BREAKEVEN_PROFIT_ATR_TARGET", "1.0", cast_type=Decimal)
        self.breakeven_min_abs_pnl_usdt: Decimal = self._get_env("BREAKEVEN_MIN_ABS_PNL_USDT", "0.50", cast_type=Decimal)
        self.enable_partial_tp: bool = self._get_env("ENABLE_PARTIAL_TP", "false", cast_type=bool)
        self.partial_tp_atr_target: Decimal = self._get_env("PARTIAL_TP_ATR_TARGET", "2.0", cast_type=Decimal)
        self.partial_tp_close_percentage: Decimal = self._get_env("PARTIAL_TP_CLOSE_PERCENTAGE", "0.5", cast_type=Decimal)
        self.enable_time_based_stop: bool = self._get_env("ENABLE_TIME_BASED_STOP", "false", cast_type=bool)
        self.max_trade_duration_seconds: int = self._get_env("MAX_TRADE_DURATION_SECONDS", 3600, cast_type=int)
        self.emergency_stop_file_path: str = self._get_env("EMERGENCY_STOP_FILE_PATH", str(Path(__file__).resolve().parent / EMERGENCY_STOP_FILE_DEFAULT))
        # Pyramiding
        self.enable_position_scaling: bool = self._get_env("ENABLE_POSITION_SCALING", "false", cast_type=bool)
        self.max_scale_ins: int = self._get_env("MAX_SCALE_INS", 1, cast_type=int)
        self.scale_in_risk_percentage: Decimal = self._get_env("SCALE_IN_RISK_PERCENTAGE", "0.0025", cast_type=Decimal)
        self.min_profit_for_scale_in_atr: Decimal = self._get_env("MIN_PROFIT_FOR_SCALE_IN_ATR", "1.0", cast_type=Decimal)
        self.max_active_trade_parts: int = self._get_env("MAX_ACTIVE_TRADE_PARTS", 3, cast_type=int)
        # Execution
        self.entry_order_type: OrderEntryType = OrderEntryType(self._get_env("ENTRY_ORDER_TYPE", OrderEntryType.MARKET.value).upper())
        self.limit_order_offset_atr_percentage: Decimal = self._get_env("LIMIT_ORDER_OFFSET_ATR_PERCENTAGE", "0.1", cast_type=Decimal)
        self.limit_order_chase_timeout_seconds: int = self._get_env("LIMIT_ORDER_CHASE_TIMEOUT_SECONDS", 10, cast_type=int)
        self.use_post_only_orders: bool = self._get_env("USE_POST_ONLY_ORDERS", "false", cast_type=bool)
        self.order_fill_timeout_seconds: int = self._get_env("ORDER_FILL_TIMEOUT_SECONDS", 15, cast_type=int)
        # Strategy Specific Params
        self.st_atr_length: int = self._get_env("ST_ATR_LENGTH", 10, cast_type=int)
        self.st_multiplier: Decimal = self._get_env("ST_MULTIPLIER", "2.0", cast_type=Decimal)
        self.confirm_st_atr_length: int = self._get_env("CONFIRM_ST_ATR_LENGTH", 20, cast_type=int)
        self.confirm_st_multiplier: Decimal = self._get_env("CONFIRM_ST_MULTIPLIER", "3.0", cast_type=Decimal)
        self.momentum_period: int = self._get_env("MOMENTUM_PERIOD", 14, cast_type=int)
        self.momentum_threshold: Decimal = self._get_env("MOMENTUM_THRESHOLD", "0", cast_type=Decimal)
        self.confirm_st_stability_lookback: int = self._get_env("CONFIRM_ST_STABILITY_LOOKBACK", 3, cast_type=int)
        self.st_max_entry_distance_atr_multiplier: Optional[Decimal] = self._get_env("ST_MAX_ENTRY_DISTANCE_ATR_MULTIPLIER", "0.5", cast_type=Decimal, required=False)
        self.ehlers_fisher_length: int = self._get_env("EHLERS_FISHER_LENGTH", 10, cast_type=int)
        self.ehlers_fisher_signal_length: int = self._get_env("EHLERS_FISHER_SIGNAL_LENGTH", 1, cast_type=int)
        self.ehlers_fisher_extreme_threshold_positive: Decimal = self._get_env("EHLERS_FISHER_EXTREME_THRESHOLD_POSITIVE", "2.0", cast_type=Decimal)
        self.ehlers_fisher_extreme_threshold_negative: Decimal = self._get_env("EHLERS_FISHER_EXTREME_THRESHOLD_NEGATIVE", "-2.0", cast_type=Decimal)
        self.stochrsi_rsi_length: int = self._get_env("STOCHRSI_RSI_LENGTH", 14, cast_type=int)
        self.stochrsi_stoch_length: int = self._get_env("STOCHRSI_STOCH_LENGTH", 14, cast_type=int)
        self.stochrsi_k_period: int = self._get_env("STOCHRSI_K_PERIOD", 3, cast_type=int)
        self.stochrsi_d_period: int = self._get_env("STOCHRSI_D_PERIOD", 3, cast_type=int)
        self.stochrsi_overbought: Decimal = self._get_env("STOCHRSI_OVERBOUGHT", "80.0", cast_type=Decimal)
        self.stochrsi_oversold: Decimal = self._get_env("STOCHRSI_OVERSOLD", "20.0", cast_type=Decimal)
        self.ehlers_fast_period: int = self._get_env("EHLERS_FAST_PERIOD", 10, cast_type=int)
        self.ehlers_slow_period: int = self._get_env("EHLERS_SLOW_PERIOD", 30, cast_type=int)
        self.ehlers_ssf_poles: int = self._get_env("EHLERS_SSF_POLES", 2, cast_type=int)
        # Confirmation Filters
        self.enable_volume_confirmation: bool = self._get_env("ENABLE_VOLUME_CONFIRMATION", "false", cast_type=bool)
        self.volume_spike_multiplier: Decimal = self._get_env("VOLUME_SPIKE_MULTIPLIER", "1.5", cast_type=Decimal)
        self.volume_ma_period: int = self._get_env("VOLUME_MA_PERIOD", 20, cast_type=int)
        self.enable_adx_filter: bool = self._get_env("ENABLE_ADX_FILTER", "false", cast_type=bool)
        self.adx_min_strength_threshold: int = self._get_env("ADX_MIN_STRENGTH_THRESHOLD", 25, cast_type=int)
        self.adx_period: int = self._get_env("ADX_PERIOD", 14, cast_type=int)
        self.enable_roc_filter: bool = self._get_env("ENABLE_ROC_FILTER", "false", cast_type=bool)
        self.roc_period: int = self._get_env("ROC_PERIOD", 5, cast_type=int)
        self.roc_min_value_long: Decimal = self._get_env("ROC_MIN_VALUE_LONG", "0.01", cast_type=Decimal)
        self.roc_max_value_short: Decimal = self._get_env("ROC_MAX_VALUE_SHORT", "-0.01", cast_type=Decimal)
        self.enable_no_trade_zones: bool = self._get_env("ENABLE_NO_TRADE_ZONES", "false", cast_type=bool)
        self.no_trade_zone_pct_around_key_level: Decimal = self._get_env("NO_TRADE_ZONE_PCT_AROUND_KEY_LEVEL", "0.002", cast_type=Decimal)
        self.key_round_number_step: Optional[Decimal] = self._get_env("KEY_ROUND_NUMBER_STEP", "1000", cast_type=Decimal, required=False)
        self.enable_trap_filter: bool = self._get_env("ENABLE_TRAP_FILTER", "false", cast_type=bool)
        self.trap_filter_lookback_period: int = self._get_env("TRAP_FILTER_LOOKBACK_PERIOD", 20, cast_type=int)
        self.trap_filter_rejection_threshold_atr: Decimal = self._get_env("TRAP_FILTER_REJECTION_THRESHOLD_ATR", "1.0", cast_type=Decimal)
        # General Trading Behavior
        self.max_consecutive_losses: int = self._get_env("MAX_CONSECUTIVE_LOSSES", 5, cast_type=int)
        self.consecutive_loss_cooldown_minutes: int = self._get_env("CONSECUTIVE_LOSS_COOLDOWN_MINUTES", 60, cast_type=int)
        self.enable_anti_martingale_risk: bool = self._get_env("ENABLE_ANTI_MARTINGALE_RISK", "false", cast_type=bool)
        self.risk_reduction_factor_on_loss: Decimal = self._get_env("RISK_REDUCTION_FACTOR_ON_LOSS", "0.75", cast_type=Decimal)
        self.risk_increase_factor_on_win: Decimal = self._get_env("RISK_INCREASE_FACTOR_ON_WIN", "1.1", cast_type=Decimal)
        self.max_risk_pct_anti_martingale: Decimal = self._get_env("MAX_RISK_PCT_ANTI_MARTINGALE", "0.02", cast_type=Decimal)
        self.trading_allowed_hours_utc: Optional[str] = self._get_env("TRADING_ALLOWED_HOURS_UTC", None, cast_type=str, required=False)
        self.signal_persistence_candles: int = self._get_env("SIGNAL_PERSISTENCE_CANDLES", 1, cast_type=int)
        self.enable_whipsaw_cooldown: bool = self._get_env("ENABLE_WHIPSAW_COOLDOWN", "true", cast_type=bool)
        self.whipsaw_max_trades_in_period: int = self._get_env("WHIPSAW_MAX_TRADES_IN_PERIOD", 3, cast_type=int)
        self.whipsaw_period_seconds: int = self._get_env("WHIPSAW_PERIOD_SECONDS", 300, cast_type=int)
        self.whipsaw_cooldown_seconds: int = self._get_env("WHIPSAW_COOLDOWN_SECONDS", 180, cast_type=int)
        self.enable_session_pnl_limits: bool = self._get_env("ENABLE_SESSION_PNL_LIMITS", "false", cast_type=bool)
        self.session_profit_target_usdt: Optional[Decimal] = self._get_env("SESSION_PROFIT_TARGET_USDT", None, cast_type=Decimal, required=False)
        self.session_max_loss_usdt: Optional[Decimal] = self._get_env("SESSION_MAX_LOSS_USDT", None, cast_type=Decimal, required=False)
        self.cooldown_after_sl_minutes: int = self._get_env("COOLDOWN_AFTER_SL_MINUTES", 15, cast_type=int)
        self.enable_daily_max_trades_rest: bool = self._get_env("ENABLE_DAILY_MAX_TRADES_REST", "false", cast_type=bool)
        self.daily_max_trades_limit: int = self._get_env("DAILY_MAX_TRADES_LIMIT", 10, cast_type=int)
        self.daily_max_trades_rest_hours: int = self._get_env("DAILY_MAX_TRADES_REST_HOURS", 4, cast_type=int)
        # Misc / Internal
        self.atr_calculation_period: int = self.atr_short_term_period if self.enable_dynamic_atr_sl else self._get_env("ATR_CALCULATION_PERIOD", 14, cast_type=int)
        self.enable_notifications: bool = self._get_env("ENABLE_NOTIFICATIONS", "true", cast_type=bool)
        self.notification_timeout_seconds: int = self._get_env("NOTIFICATION_TIMEOUT_SECONDS", 10, cast_type=int)
        self.default_recv_window: int = self._get_env("DEFAULT_RECV_WINDOW", 13000, cast_type=int)
        self.order_book_depth: int = self._get_env("ORDER_BOOK_DEPTH", 20, cast_type=int)
        self.order_book_fetch_limit: int = max(25, self.order_book_depth)
        self.shallow_ob_fetch_depth: int = 5
        self.side_buy: str = "buy"; self.side_sell: str = "sell"; self.pos_long: str = "Long"; self.pos_short: str = "Short"; self.pos_none: str = "None"
        self.usdt_symbol: str = "USDT"; self.retry_count: int = 3; self.retry_delay_seconds: int = 3; self.api_fetch_limit_buffer: int = 30
        self.position_qty_epsilon: Decimal = Decimal("1e-9"); self.post_close_delay_seconds: int = 3; self.cache_candle_duration_multiplier: Decimal = Decimal("0.95")
        self.MARKET_INFO: Optional[Dict[str, Any]] = None # Populated by ExchangeManager
        self.PAPER_TRADING_MODE: bool = self._get_env("PAPER_TRADING_MODE", "false", cast_type=bool)
        self.send_notification_method = notify_termux
        self.strategy_instance: 'TradingStrategy' # Type hint, initialized later

        self._validate_parameters()
        logger_cfg.info(f"{NEON['HEADING']}--- Configuration Runes v{VERSION} Summoned and Verified ---{NEON['RESET']}")

    def _get_env(self, key: str, default: Any = None, cast_type: type = str, required: bool = False, color: str = NEON["PARAM"], secret: bool = False) -> Any:
        _logger = logging.getLogger("ConfigModule._get_env")
        value_str = os.getenv(key); source = "Env Var"; value_to_cast: Any = None
        display_value = "********" if secret and value_str is not None else value_str

        if value_str is None:
            if required and default is None : _logger.critical(f"{NEON['CRITICAL']}CRITICAL: Required config rune '{key}' not found and no default.{NEON['RESET']}"); raise ValueError(f"Required env var '{key}' not set, no default.")
            _logger.debug(f"{color}Config Rune {NEON['VALUE']}'{key}'{color}: Not Found. Default: '{NEON['VALUE']}{default}{color}'{NEON['RESET']}"); value_to_cast = default; source = "Default"
        else: _logger.debug(f"{color}Config Rune {NEON['VALUE']}'{key}'{color}: Env Value: '{NEON['VALUE']}{display_value}{color}'{NEON['RESET']}"); value_to_cast = value_str

        if value_to_cast is None:
            if required: _logger.critical(f"{NEON['CRITICAL']}CRITICAL: Required config rune '{key}' resolved to None.{NEON['RESET']}"); raise ValueError(f"Required env var '{key}' resolved to None.")
            return None

        final_value: Any
        try:
            raw_value_str_for_cast = str(value_to_cast)
            if cast_type == bool: final_value = raw_value_str_for_cast.lower() in ["true", "1", "yes", "y"]
            elif cast_type == Decimal: final_value = Decimal(raw_value_str_for_cast)
            elif cast_type == int: final_value = int(Decimal(raw_value_str_for_cast))
            elif cast_type == float: final_value = float(raw_value_str_for_cast)
            elif cast_type == str: final_value = raw_value_str_for_cast
            else: _logger.warning(f"{NEON['WARNING']}Unsupported cast_type '{cast_type.__name__}' for key '{key}'. Raw: '{raw_value_str_for_cast}'.{NEON['RESET']}"); final_value = raw_value_str_for_cast
        except (ValueError, TypeError, InvalidOperation) as e:
            _logger.error(f"{NEON['ERROR']}Cast error for '{key}': '{value_to_cast}' (Source: {source}). Expected {cast_type.__name__}. Err: {e}. Default: '{default}'.{NEON['RESET']}")
            if default is None:
                if required: _logger.critical(f"{NEON['CRITICAL']}CRITICAL: Cast fail for required '{key}', default is None.{NEON['RESET']}"); raise ValueError(f"Required env var '{key}' failed casting, no valid default.")
                else: _logger.warning(f"{NEON['WARNING']}Cast fail for optional '{key}', default is None. Final: None{NEON['RESET']}"); return None
            else:
                source = "Default (Fallback)"; _logger.debug(f"Casting fallback default '{default}' for '{key}' to {cast_type.__name__}")
                try:
                    default_str_for_cast = str(default)
                    if cast_type == bool: final_value = default_str_for_cast.lower() in ["true", "1", "yes", "y"]
                    elif cast_type == Decimal: final_value = Decimal(default_str_for_cast)
                    elif cast_type == int: final_value = int(Decimal(default_str_for_cast))
                    elif cast_type == float: final_value = float(default_str_for_cast)
                    elif cast_type == str: final_value = default_str_for_cast
                    else: final_value = default_str_for_cast
                    _logger.warning(f"{NEON['WARNING']}Used casted default for {key}: '{NEON['VALUE']}{final_value}{NEON['WARNING']}'{NEON['RESET']}")
                except (ValueError, TypeError, InvalidOperation) as e_default: _logger.critical(f"{NEON['CRITICAL']}CRITICAL: Cast fail for value AND default for '{key}'. Err: {e_default}{NEON['RESET']}"); raise ValueError(f"Config error: Cannot cast value or default for '{key}'.")
        display_final_value = "********" if secret else final_value
        _logger.debug(f"{color}Final value for {NEON['VALUE']}'{key}'{color}: {NEON['VALUE']}{display_final_value}{color} (Type: {type(final_value).__name__}, Source: {source}){NEON['RESET']}")
        return final_value

    def _validate_parameters(self) -> None:
        _logger = logging.getLogger("ConfigModule._validate")
        errors = []
        if not (0 < self.risk_per_trade_percentage < 1): errors.append(f"RISK_PER_TRADE_PERCENTAGE ({self.risk_per_trade_percentage}) must be between 0 and 1.")
        if self.leverage < 1: errors.append(f"LEVERAGE ({self.leverage}) must be at least 1.")
        if self.max_scale_ins < 0: errors.append(f"MAX_SCALE_INS ({self.max_scale_ins}) cannot be negative.")
        if self.limit_order_offset_atr_percentage < 0: errors.append("LIMIT_ORDER_OFFSET_ATR_PERCENTAGE cannot be negative.")
        if not (0 <= self.adx_min_strength_threshold <= 100): errors.append("ADX_MIN_STRENGTH_THRESHOLD must be between 0 and 100.")
        if self.max_consecutive_losses < 0: errors.append("MAX_CONSECUTIVE_LOSSES cannot be negative.")
        if self.trading_allowed_hours_utc:
            try:
                for r_str in self.trading_allowed_hours_utc.split(','):
                    s, e = map(int, r_str.split('-'))
                    if not (0 <= s < 24 and 0 < e <= 24 and s < e): errors.append(f"Invalid TRADING_ALLOWED_HOURS_UTC range: {r_str} (end hour is exclusive, up to 24)")
            except Exception: errors.append(f"Invalid TRADING_ALLOWED_HOURS_UTC format: {self.trading_allowed_hours_utc}. Expected 'HH-HH,HH-HH'.")
        if self.tsl_type == TSLType.NONE and (self.trailing_stop_percentage > 0 or self.atr_tsl_trail_atr_distance > 0):
            _logger.warning(f"{NEON['WARNING']}TSL_TYPE is NONE, but TSL parameters are set. TSL will be disabled.{NEON['RESET']}")
        if errors:
            error_message = f"Configuration validation failed:\n" + "\n".join([f"  - {e}" for e in errors])
            _logger.critical(f"{NEON['CRITICAL']}{error_message}{NEON['RESET']}")
            raise ValueError(error_message)

# --- Logger Setup ---
os.makedirs(LOGS_DIR, exist_ok=True)
log_file_name = f"{LOGS_DIR}/pyrmethus_spell_v{VERSION.split(' ')[0]}_{time.strftime('%Y%m%d_%H%M%S')}.log"
LOGGING_LEVEL_ENV = os.getenv("LOGGING_LEVEL", "INFO").upper()
LOGGING_LEVEL = getattr(logging, LOGGING_LEVEL_ENV, logging.INFO)

logging.basicConfig(level=LOGGING_LEVEL,
                    format="%(asctime)s.%(msecs)03d [%(levelname)-8s] %(name)-38s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file_name, mode='a')])
logger: logging.Logger = logging.getLogger("PyrmethusCore")
SUCCESS_LEVEL: int = 25; logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")
def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(SUCCESS_LEVEL): self._log(SUCCESS_LEVEL, message, args, **kwargs) # type: ignore[attr-defined]
logging.Logger.success = log_success # type: ignore[attr-defined]

if sys.stdout.isatty():
    level_colors = {
        logging.DEBUG: NEON['DEBUG'], logging.INFO: NEON['INFO'], SUCCESS_LEVEL: NEON['SUCCESS'],
        logging.WARNING: NEON['WARNING'], logging.ERROR: NEON['ERROR'], logging.CRITICAL: NEON['CRITICAL']
    }
    for level, color_code in level_colors.items():
        level_name = logging.getLevelName(level)
        plain_level_name = re.sub(r'\x1b\[[0-9;]*m', '', level_name)
        logging.addLevelName(level, f"{color_code}{plain_level_name}{NEON['RESET']}")

# --- Global Objects & State Variables ---
try: CONFIG = Config()
except ValueError as e: logging.getLogger("Main").critical(f"{NEON['CRITICAL']}Config Error: {e}{NEON['RESET']}"); notify_termux("Pyrmethus FATAL", f"Config Error: {e}"); sys.exit(1)
except Exception as e: logging.getLogger("Main").critical(f"{NEON['CRITICAL']}Unexpected Config Init Error: {e}{NEON['RESET']}"); logging.getLogger("Main").debug(traceback.format_exc()); notify_termux("Pyrmethus FATAL", f"Unexpected Config Init: {e}"); sys.exit(1)

trade_metrics: 'TradeMetrics'
_active_trade_parts: List[Dict[str, Any]] = []
_last_heartbeat_save_time: float = 0.0
_last_sl_hit_timestamp: float = 0.0
_persistent_signal_counter: Dict[str, int] = {"long": 0, "short": 0}
_last_signal_type_for_persistence: Optional[str] = None
_previous_day_high: Optional[Decimal] = None
_previous_day_low: Optional[Decimal] = None
_last_key_level_update_day: Optional[int] = None
_whipsaw_cooldown_active_until: float = 0.0
_trade_timestamps_for_whipsaw: deque = deque(maxlen=CONFIG.whipsaw_max_trades_in_period)
_stop_trading_flag: bool = False
_last_drawdown_check_time: float = 0.0
_vol_atr_analysis_results_cache: Dict[str, Any] = {}
_market_data_cache: Dict[str, pd.DataFrame] = {}
_last_market_data_fetch_ts: Dict[str, float] = {}

# --- Retry Decorator ---
def api_retry_logic(func, *args, **kwargs):
    return retry(exceptions=(ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection),
                 tries=CONFIG.retry_count, delay=CONFIG.retry_delay_seconds, backoff=2, logger=logger)(func)(*args, **kwargs)

# --- TradingStrategy Abstract Base Class & Implementations ---
class TradingStrategy(ABC):
    def __init__(self, config: Config, df_columns: Optional[List[str]] = None):
        self.config = config
        self.logger = logging.getLogger(f"Strategy.{self.__class__.__name__}")
        self.required_columns = df_columns if df_columns else []
        self.logger.info(f"{NEON['STRATEGY']}Strategy Form '{self.__class__.__name__}' materializing...{NEON['RESET']}")

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, latest_close: Decimal, latest_atr: Optional[Decimal]) -> Dict[str, Any]:
        pass

    def _validate_df(self, df: pd.DataFrame, min_rows: int = 2) -> bool:
        if df is None or df.empty or len(df) < min_rows:
            self.logger.debug(f"Insufficient data (Rows: {len(df) if df is not None else 0}, Min: {min_rows}).")
            return False
        if self.required_columns:
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                self.logger.warning(f"{NEON['WARNING']}DataFrame missing required columns: {missing_cols}.{NEON['RESET']}")
                return False
            # Check only the latest row for NaNs in required columns
            latest_row_values = df.iloc[-1][self.required_columns]
            if latest_row_values.isnull().any():
                nan_cols = latest_row_values[latest_row_values.isnull()].index.tolist()
                self.logger.debug(f"NaNs in last candle for critical columns: {nan_cols}.")
                # Optionally return False if any critical column is NaN on the latest candle
                # For now, this is a debug log, actual handling might depend on strategy
        return True


    def _get_default_signals(self) -> Dict[str, Any]:
        return {"enter_long": False, "enter_short": False, "exit_long": False, "exit_short": False, "exit_reason": "Default Signal"}

class DualSupertrendMomentumStrategy(TradingStrategy):
    def __init__(self, config: Config):
        primary_st_l_col = f"st_ST_{config.st_atr_length}_{float(config.st_multiplier)}l"
        primary_st_s_col = f"st_ST_{config.st_atr_length}_{float(config.st_multiplier)}s"
        super().__init__(config, df_columns=[
            "st_trend", "st_st_long_flip", "st_st_short_flip",
            "confirm_trend", "momentum",
            primary_st_l_col, primary_st_s_col
        ])

    def generate_signals(self, df: pd.DataFrame, latest_close: Decimal, latest_atr: Optional[Decimal]) -> Dict[str, Any]:
        signals = self._get_default_signals()
        min_rows_needed = max(self.config.st_atr_length, self.config.confirm_st_atr_length, self.config.momentum_period, self.config.confirm_st_stability_lookback) + 15
        if not self._validate_df(df, min_rows=min_rows_needed): return signals

        last = df.iloc[-1]
        primary_long_flip = last.get("st_st_long_flip", False)
        primary_short_flip = last.get("st_st_short_flip", False)
        current_confirm_trend = last.get("confirm_trend", pd.NA)
        momentum_val = safe_decimal(last.get("momentum", pd.NA))

        if primary_long_flip and primary_short_flip:
            self.logger.warning(f"{NEON['WARNING']}Conflicting primary ST flips. Resolving...{NEON['RESET']}")
            if current_confirm_trend is True and (not pd.isna(momentum_val) and momentum_val > 0): primary_short_flip = False
            elif current_confirm_trend is False and (not pd.isna(momentum_val) and momentum_val < 0): primary_long_flip = False
            else: primary_long_flip, primary_short_flip = False, False

        stable_confirm_trend: Union[bool, PandasNAType] = pd.NA
        if self.config.confirm_st_stability_lookback <= 1: stable_confirm_trend = current_confirm_trend
        elif 'confirm_trend' in df.columns and len(df) >= self.config.confirm_st_stability_lookback:
            recent_confirm_trends = df['confirm_trend'].iloc[-self.config.confirm_st_stability_lookback:]
            if current_confirm_trend is True and recent_confirm_trends.is_monotonic_increasing and recent_confirm_trends.all(): stable_confirm_trend = True
            elif current_confirm_trend is False and recent_confirm_trends.is_monotonic_decreasing and not recent_confirm_trends.any(): stable_confirm_trend = False

        if pd.isna(stable_confirm_trend) or pd.isna(momentum_val):
            self.logger.debug(f"Stable Confirm ST Trend ({log_format(stable_confirm_trend, trend=True)}) or Momentum ({log_format(momentum_val)}) is NA.")
            return signals

        price_proximity_ok = True
        if self.config.st_max_entry_distance_atr_multiplier is not None and latest_atr is not None and latest_atr > 0 and not pd.isna(latest_close):
            st_long_line_col = f"st_ST_{self.config.st_atr_length}_{float(self.config.st_multiplier)}l"
            st_short_line_col = f"st_ST_{self.config.st_atr_length}_{float(self.config.st_multiplier)}s"
            max_allowed_distance = latest_atr * self.config.st_max_entry_distance_atr_multiplier

            if primary_long_flip and st_long_line_col in last.index:
                st_line_value = safe_decimal(last.get(st_long_line_col))
                if st_line_value is not None and not pd.isna(st_line_value) and (latest_close - st_line_value) > max_allowed_distance: price_proximity_ok = False # type: ignore
            elif primary_short_flip and st_short_line_col in last.index:
                st_line_value = safe_decimal(last.get(st_short_line_col))
                if st_line_value is not None and not pd.isna(st_line_value) and (st_line_value - latest_close) > max_allowed_distance: price_proximity_ok = False # type: ignore
            if not price_proximity_ok: self.logger.debug(f"Entry suppressed: Price too far from ST line.")

        if primary_long_flip and stable_confirm_trend is True and momentum_val > self.config.momentum_threshold and momentum_val > 0 and price_proximity_ok: # type: ignore
            signals["enter_long"] = True; self.logger.info(f"{NEON['SIDE_LONG']}DualST+Mom: LONG Entry - Primary ST Long Flip, Stable Confirm Up, Positive Momentum ({log_format(momentum_val)}) > Threshold{NEON['RESET']}")
        elif primary_short_flip and stable_confirm_trend is False and momentum_val < -self.config.momentum_threshold and momentum_val < 0 and price_proximity_ok: # type: ignore
            signals["enter_short"] = True; self.logger.info(f"{NEON['SIDE_SHORT']}DualST+Mom: SHORT Entry - Primary ST Short Flip, Stable Confirm Down, Negative Momentum ({log_format(momentum_val)}) < -Threshold{NEON['RESET']}")

        if primary_short_flip: signals["exit_long"] = True; signals["exit_reason"] = "Primary ST Flipped Short"
        if primary_long_flip: signals["exit_short"] = True; signals["exit_reason"] = "Primary ST Flipped Long"
        return signals

class EhlersFisherStrategy(TradingStrategy):
    def __init__(self, config: Config):
        super().__init__(config, df_columns=["ehlers_fisher", "ehlers_signal"])

    def generate_signals(self, df: pd.DataFrame, latest_close: Decimal, latest_atr: Optional[Decimal]) -> Dict[str, Any]:
        signals = self._get_default_signals()
        min_rows_needed = self.config.ehlers_fisher_length + self.config.ehlers_fisher_signal_length + 10
        if not self._validate_df(df, min_rows=min_rows_needed) or len(df) < 2 : return signals

        last = df.iloc[-1]; prev = df.iloc[-2]
        fisher_now = safe_decimal(last.get("ehlers_fisher"), pd.NA); signal_now = safe_decimal(last.get("ehlers_signal"), pd.NA)
        fisher_prev = safe_decimal(prev.get("ehlers_fisher"), pd.NA); signal_prev = safe_decimal(prev.get("ehlers_signal"), pd.NA)

        if any(pd.isna(v) for v in [fisher_now, signal_now, fisher_prev, signal_prev]):
            self.logger.debug(f"Ehlers Fisher/Signal NA. No signal."); return signals

        is_fisher_extreme = False
        if (fisher_now > self.config.ehlers_fisher_extreme_threshold_positive or # type: ignore
            fisher_now < self.config.ehlers_fisher_extreme_threshold_negative): # type: ignore
            is_fisher_extreme = True

        if not is_fisher_extreme:
            if fisher_prev <= signal_prev and fisher_now > signal_now: # type: ignore
                signals["enter_long"] = True; self.logger.info(f"{NEON['SIDE_LONG']}EhlersFisher: LONG Entry - Fisher ({log_format(fisher_now)}) crossed ABOVE Signal ({log_format(signal_now)}){NEON['RESET']}")
            elif fisher_prev >= signal_prev and fisher_now < signal_now: # type: ignore
                signals["enter_short"] = True; self.logger.info(f"{NEON['SIDE_SHORT']}EhlersFisher: SHORT Entry - Fisher ({log_format(fisher_now)}) crossed BELOW Signal ({log_format(signal_now)}){NEON['RESET']}")
        elif (fisher_prev <= signal_prev and fisher_now > signal_now) or (fisher_prev >= signal_prev and fisher_now < signal_now): # type: ignore
             self.logger.info(f"EhlersFisher: Crossover signal ignored, Fisher in extreme zone ({log_format(fisher_now)}).")

        if fisher_prev >= signal_prev and fisher_now < signal_now: signals["exit_long"] = True; signals["exit_reason"] = "Ehlers Fisher crossed BELOW Signal" # type: ignore
        elif fisher_prev <= signal_prev and fisher_now > signal_now: signals["exit_short"] = True; signals["exit_reason"] = "Ehlers Fisher crossed ABOVE Signal" # type: ignore
        return signals

class StochRsiMomentumStrategy(TradingStrategy):
    def __init__(self, config: Config):
        super().__init__(config, df_columns=["stochrsi_k", "stochrsi_d", "momentum"])

    def generate_signals(self, df: pd.DataFrame, latest_close: Decimal, latest_atr: Optional[Decimal]) -> Dict[str, Any]:
        signals = self._get_default_signals()
        min_len = max(self.config.stochrsi_rsi_length + self.config.stochrsi_stoch_length + self.config.stochrsi_d_period, self.config.momentum_period) + 10
        if not self._validate_df(df, min_rows=min_len) or len(df) < 2: return signals

        last, prev = df.iloc[-1], df.iloc[-2]
        k_now, d_now, mom_now = safe_decimal(last.get("stochrsi_k")), safe_decimal(last.get("stochrsi_d")), safe_decimal(last.get("momentum"))
        k_prev, d_prev = safe_decimal(prev.get("stochrsi_k")), safe_decimal(prev.get("stochrsi_d"))

        if any(pd.isna(v) for v in [k_now, d_now, mom_now, k_prev, d_prev]):
            self.logger.debug("StochRSI/Mom NA. No signal."); return signals

        if k_prev <= d_prev and k_now > d_now and k_now < self.config.stochrsi_oversold and mom_now > 0: # type: ignore
            signals["enter_long"] = True; self.logger.info(f"{NEON['SIDE_LONG']}StochRSI+Mom: LONG Entry - K ({log_format(k_now)}) > D ({log_format(d_now)}), Oversold, Mom Positive{NEON['RESET']}")
        elif k_prev >= d_prev and k_now < d_now and k_now > self.config.stochrsi_overbought and mom_now < 0: # type: ignore
            signals["enter_short"] = True; self.logger.info(f"{NEON['SIDE_SHORT']}StochRSI+Mom: SHORT Entry - K ({log_format(k_now)}) < D ({log_format(d_now)}), Overbought, Mom Negative{NEON['RESET']}")

        if k_prev >= d_prev and k_now < d_now: signals["exit_long"] = True; signals["exit_reason"] = "StochRSI K crossed D Down" # type: ignore
        elif k_prev <= d_prev and k_now > d_now: signals["exit_short"] = True; signals["exit_reason"] = "StochRSI K crossed D Up" # type: ignore
        return signals

class EhlersMaCrossStrategy(TradingStrategy):
    def __init__(self, config: Config):
        super().__init__(config, df_columns=["ehlers_ssf_fast", "ehlers_ssf_slow"])

    def generate_signals(self, df: pd.DataFrame, latest_close: Decimal, latest_atr: Optional[Decimal]) -> Dict[str, Any]:
        signals = self._get_default_signals()
        min_len = max(self.config.ehlers_fast_period, self.config.ehlers_slow_period) + self.config.ehlers_ssf_poles + 10
        if not self._validate_df(df, min_rows=min_len) or len(df) < 2: return signals

        last, prev = df.iloc[-1], df.iloc[-2]
        fast_ma_now, slow_ma_now = safe_decimal(last.get("ehlers_ssf_fast")), safe_decimal(last.get("ehlers_ssf_slow"))
        fast_ma_prev, slow_ma_prev = safe_decimal(prev.get("ehlers_ssf_fast")), safe_decimal(prev.get("ehlers_ssf_slow"))

        if any(pd.isna(v) for v in [fast_ma_now, slow_ma_now, fast_ma_prev, slow_ma_prev]):
            self.logger.debug("Ehlers SSF MA NA. No signal."); return signals

        if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now: # type: ignore
            signals["enter_long"] = True; self.logger.info(f"{NEON['SIDE_LONG']}EhlersMACross: LONG Entry - Fast ({log_format(fast_ma_now)}) > Slow ({log_format(slow_ma_now)}){NEON['RESET']}")
        elif fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now: # type: ignore
            signals["enter_short"] = True; self.logger.info(f"{NEON['SIDE_SHORT']}EhlersMACross: SHORT Entry - Fast ({log_format(fast_ma_now)}) < Slow ({log_format(slow_ma_now)}){NEON['RESET']}")

        if fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now: signals["exit_long"] = True; signals["exit_reason"] = "Ehlers Fast SSF MA crossed Slow Down" # type: ignore
        elif fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now: signals["exit_short"] = True; signals["exit_reason"] = "Ehlers Fast SSF MA crossed Slow Up" # type: ignore
        return signals

# Initialize strategy instance after CONFIG is fully loaded
strategy_map_instance: Dict[StrategyName, Type[TradingStrategy]] = {
    StrategyName.DUAL_SUPERTREND_MOMENTUM: DualSupertrendMomentumStrategy,
    StrategyName.EHLERS_FISHER: EhlersFisherStrategy,
    StrategyName.STOCHRSI_MOMENTUM: StochRsiMomentumStrategy,
    StrategyName.EHLERS_MA_CROSS: EhlersMaCrossStrategy,
}
StrategyClassInstance = strategy_map_instance.get(CONFIG.strategy_name)
if StrategyClassInstance: CONFIG.strategy_instance = StrategyClassInstance(CONFIG)
else: err_msg = f"Failed to init strategy '{CONFIG.strategy_name.value}'."; logging.getLogger("Main").critical(f"{NEON['CRITICAL']}{err_msg}{NEON['RESET']}"); notify_termux("Pyrmethus FATAL", err_msg); sys.exit(1)

# --- Data Tracking ---
class TradeMetrics:
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.wins: int = 0
        self.losses: int = 0
        self.consecutive_wins: int = 0
        self.consecutive_losses: int = 0
        self.max_consecutive_wins: int = 0
        self.max_consecutive_losses: int = 0
        self.total_trades: int = 0
        self.total_profit: Decimal = Decimal(0)
        self.total_loss: Decimal = Decimal(0)
        self.win_rate: Decimal = Decimal(0)
        self.loss_rate: Decimal = Decimal(0)
        self.avg_profit: Decimal = Decimal(0)
        self.avg_loss: Decimal = Decimal(0)
        self.max_drawdown: Decimal = Decimal(0)
        self.current_drawdown: Decimal = Decimal(0)
        self.max_equity: Decimal = Decimal(0)
        self.session_start_time: datetime = datetime.now(timezone.utc)
        self.profit_target_hit: bool = False
        self.max_drawdown_hit: bool = False
        self.trades_this_session: int = 0
        self.trade_timestamps_this_session: deque = deque(maxlen=CONFIG.daily_max_trades_limit * 2) # For daily trade count and whipsaw
        self.atr_at_entry_for_sl: Decimal = Decimal('NaN') # Track ATR for current trade

    def update(self, pnl: Decimal):
        self.total_trades += 1
        self.trades_this_session += 1
        self.trade_timestamps_this_session.append(datetime.now(timezone.utc))
        if pnl > 0:
            self.wins += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.total_profit += pnl
            self.max_consecutive_wins = max(self.max_consecutive_wins, self.consecutive_wins)
            self.avg_profit = self.total_profit / self.wins if self.wins else Decimal(0)
        elif pnl < 0:
            self.losses += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.total_loss += pnl
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
            self.avg_loss = self.total_loss / self.losses if self.losses else Decimal(0)
        # self.current_drawdown = Decimal(0) # Reset drawdown after any trade for this simple version
        self.win_rate = Decimal(self.wins / self.total_trades * 100) if self.total_trades else Decimal(0)
        self.loss_rate = Decimal(self.losses / self.total_trades * 100) if self.total_trades else Decimal(0)

    def set_drawdown(self, equity: Decimal):
        self.max_equity = max(self.max_equity, equity)
        self.current_drawdown = self.max_equity - equity
        if self.current_drawdown > self.max_drawdown:
            self.max_drawdown = self.current_drawdown

    def log_performance(self, log_method: Callable[[str], None] = logger.info):
        log_method(f"{NEON['HEADING']}Trade Metrics (v{VERSION}):{NEON['RESET']}")
        log_method(f"  Trades: {NEON['VALUE']}{self.total_trades}{NEON['RESET']}, Wins: {NEON['SIDE_LONG']}{self.wins}{NEON['RESET']} ({NEON['VALUE']}{self.win_rate:.2f}%{NEON['RESET']}), Losses: {NEON['SIDE_SHORT']}{self.losses}{NEON['RESET']} ({NEON['VALUE']}{self.loss_rate:.2f}%{NEON['RESET']})")
        log_method(f"  Avg PnL (win): {NEON['PNL_POS']}{self.avg_profit:.4f}{NEON['RESET']}, Avg PnL (loss): {NEON['PNL_NEG']}{self.avg_loss:.4f}{NEON['RESET']}")
        log_method(f"  Max Drawdown: {NEON['PNL_NEG']}{self.max_drawdown:.4f}{NEON['RESET']}, Peak Equity: {NEON['VALUE']}{self.max_equity:.4f}{NEON['RESET']}")
        log_method(f"  Consec Wins/Losses: {NEON['SIDE_LONG']}{self.max_consecutive_wins}{NEON['RESET']}/{NEON['SIDE_SHORT']}{self.max_consecutive_losses}{NEON['RESET']}")
trade_metrics = TradeMetrics()

# --- State Persistence Class ---
class StateManager:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.state: Dict[str, Any] = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                state = json.load(f, parse_float=Decimal, parse_int=Decimal) # Load numbers as Decimal
                self.log_numbers_in_state(state)
                logger.info(f"{NEON['INFO']}State runes read from journal: {NEON['VALUE']}{self.file_path}{NEON['RESET']}")
                return state
        except FileNotFoundError:
            logger.info(f"{NEON['WARNING']}No state journal found, starting anew.{NEON['RESET']}")
            return {}
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"{NEON['ERROR']}State reading incantation failed, starting anew: {e}{NEON['RESET']}")
            return {}

    def _save_state(self) -> None:
        state_copy = copy.deepcopy(self.state)
        self.clean_numbers_in_state(state_copy)
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(state_copy, f, indent=4, default=lambda o: str(o) if isinstance(o, Decimal) else o)
            logger.debug(f"State incantation woven to journal: {self.file_path}")
        except Exception as e: logger.error(f"State saving incantation failed: {e}", exc_info=True)

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)
    def set(self, key: str, value: Any) -> None:
        self.state[key] = value
        global _last_heartbeat_save_time
        if time.monotonic() - _last_heartbeat_save_time > HEARTBEAT_INTERVAL_SECONDS: self.save()
    def save(self) -> None:
        global _last_heartbeat_save_time
        self._save_state(); _last_heartbeat_save_time = time.monotonic()
    def force_save(self):
        self._save_state()
    def log_numbers_in_state(self, state: Dict[str, Any], indent: int = 2, prefix: str = "") -> None:
        for key, value in state.items():
            name = f"{prefix}{key}"
            if isinstance(value, dict): self.log_numbers_in_state(value, indent + 2, name + "."); continue
            if isinstance(value, (Decimal, int, float)): logger.debug(f"{NEON['COMMENT']}{' ' * indent}{name}: {value} ({type(value).__name__}){NEON['RESET']}")
    def clean_numbers_in_state(self, state: Dict[str, Any]) -> None:
        for key, value in state.items():
            if isinstance(value, dict): self.clean_numbers_in_state(value); continue
            if isinstance(value, Decimal): state[key] = str(value.quantize(Decimal("1E-8"), rounding=ROUND_HALF_UP)) # Store as string
            elif isinstance(value, float):
                try: state[key] = str(Decimal(str(value)).quantize(Decimal("1E-8"), rounding=ROUND_HALF_UP))
                except Exception: logger.warning(f"State: Error converting float value for key '{key}'"); state[key] = "NaN"
            elif isinstance(value, int): state[key] = int(value) # Ensure JSON serializable int
state_manager = StateManager(STATE_FILE_PATH)

# --- Indicator Calculation ---
def calculate_supertrend_pta(df: pd.DataFrame, multiplier: float, atr_period: int) -> pd.DataFrame:
    st = ta.supertrend(high=df["high"], low=df["low"], close=df["close"], length=atr_period, multiplier=multiplier)
    st_df = pd.DataFrame(index=df.index) # Ensure same index
    st_df["st_trend"] = st[f"SUPERTd_{atr_period}_{multiplier}"]
    st_df[f"st_ST_{atr_period}_{multiplier}l"] = st[f"SUPERTl_{atr_period}_{multiplier}"]
    st_df[f"st_ST_{atr_period}_{multiplier}s"] = st[f"SUPERTs_{atr_period}_{multiplier}"]
    st_df["st_st_long_flip"] = (st_df["st_trend"].diff() > 1).fillna(False)
    st_df["st_st_short_flip"] = (st_df["st_trend"].diff() < -1).fillna(False)
    return st_df

def calculate_ehlers_fisher_transform(df: pd.DataFrame, length: int = 10) -> pd.DataFrame:
    fisher = ta.fisher(high=df["high"], low=df["low"], length=length)
    if isinstance(fisher, tuple) and len(fisher) == 2: fisher_s, fisher_signal_s = fisher[0], fisher[1]
    elif isinstance(fisher, pd.Series): fisher_s, fisher_signal_s = fisher, fisher.shift(1) # type: ignore
    else:
        logging.getLogger("Indicators").error(f"Ehlers Fisher invalid return type: {type(fisher)}")
        nan_s = pd.Series(np.nan, index=df.index)
        return pd.DataFrame({"ehlers_fisher": nan_s, "ehlers_signal": nan_s})
    return pd.DataFrame({"ehlers_fisher": fisher_s, "ehlers_signal": fisher_signal_s}, index=df.index)

def calculate_stochrsi(df: pd.DataFrame, rsi_length: int = 14, stoch_length: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
    stochrsi_df = ta.stochrsi(df["close"], length=rsi_length, rsi_length=rsi_length, k=stoch_length, smooth_k=smooth_k, smooth_d=smooth_d) # pandas-ta uses 'k' for stoch_length
    if not isinstance(stochrsi_df, pd.DataFrame) or "STOCHRSIk" not in stochrsi_df.columns or "STOCHRSId" not in stochrsi_df.columns:
        logging.getLogger("Indicators").error(f"StochRSI invalid return type or columns: {type(stochrsi_df)}")
        return pd.DataFrame({"stochrsi_k": np.nan, "stochrsi_d": np.nan}, index=df.index)
    return stochrsi_df.rename(columns={"STOCHRSIk": "stochrsi_k", "STOCHRSId": "stochrsi_d"})[["stochrsi_k", "stochrsi_d"]]

def calculate_ehlers_ma_ssf(df: pd.DataFrame, fast_period: int = 10, slow_period: int = 30, poles: int = 2) -> pd.DataFrame:
    ssf_fast = ta.ssf(df["close"], length=fast_period, poles=poles)
    ssf_slow = ta.ssf(df["close"], length=slow_period, poles=poles)
    if not isinstance(ssf_fast, pd.Series) or not isinstance(ssf_slow, pd.Series):
        logging.getLogger("Indicators").error(f"Ehlers SSF calculation invalid return: {type(ssf_fast)} , {type(ssf_slow)}")
        return pd.DataFrame({"ehlers_ssf_fast": np.nan, "ehlers_ssf_slow": np.nan}, index=df.index)
    return pd.DataFrame({"ehlers_ssf_fast": ssf_fast, "ehlers_ssf_slow": ssf_slow}, index=df.index)

def calculate_momentum_indicator(df: pd.DataFrame, period: int = 10) -> pd.Series:
    momentum = df["close"].diff(period).fillna(0)
    return momentum.rename("momentum")

def get_recent_ohlcv(exchange: ccxt.Exchange, symbol: str, interval: str, limit: int, custom_since: Optional[datetime] = None, log_prefix: str = "fetchOHLCV") -> Optional[pd.DataFrame]:
    tf_secs = exchange.parse_timeframe(interval)
    if not tf_secs: logging.getLogger("OHLCV").error(f"TimeFrame parse failed for {interval}"); return None
    ms_per_candle = tf_secs * 1000
    since_ms = int(custom_since.timestamp() * 1000) if custom_since else None
    now_ms = exchange.milliseconds()

    all_candles: List[list[Union[int, float]]] = []
    current_since_ms = since_ms
    attempt_num = 0 # For logging

    while True:
        attempt_num +=1
        try:
            fetch_args = [symbol, interval]
            fetch_kwargs: Dict[str, Any] = {"limit": limit}
            if current_since_ms: fetch_kwargs["since"] = current_since_ms

            candles_part = api_retry_logic(exchange.fetch_ohlcv, *fetch_args, **fetch_kwargs) # type: ignore
            if not candles_part:
                logger.warning(f"{log_prefix}: Empty Candles Part returned. Attempt:{attempt_num}/{CONFIG.retry_count}.");
                if attempt_num >= CONFIG.retry_count: return None # Exhausted retries
                continue # Retry

            all_candles.extend(candles_part)

            if len(candles_part) < limit:
                if current_since_ms is None:
                    logger.debug(f"{log_prefix}: Early end, candles part len {len(candles_part)} < limit {limit}.")
                break
            elif current_since_ms is None:
                break

            max_ts_in_part = max(c[0] for c in candles_part) if candles_part else 0
            next_since_ms = max_ts_in_part + ms_per_candle
            if next_since_ms >= now_ms:
                logger.debug(f"{log_prefix}: Date range complete ({len(all_candles)} total).")
                break
            current_since_ms = next_since_ms

        except Exception as e:
            logger.error(f"{log_prefix} OHLCV fetching failed for {symbol} - {interval} : {e}", exc_info=True)
            return None

    if not all_candles: return None
    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]: df[col] = df[col].map(safe_decimal)

    df = df[~df["close"].apply(lambda x: pd.isna(x))]

    if df.index.duplicated(keep='last').any():
        logger.warning(f"Time-series duplicates found. Keeping last...")
        df = df[~df.index.duplicated(keep='last')]
    return df.sort_index() # Ensure sorted by timestamp

# --- Exchange Class ---
class Exchange:
    def __init__(self, config: Config):
        self.config = config
        self.bybit: ccxt.Exchange = self._init_exchange()
        self.time_zone = pytz.timezone('UTC')
        self._now_time: datetime = datetime.now(self.time_zone)
        self.candles_cache: Dict[str, pd.DataFrame] = {}
        if not self.config.MARKET_INFO: raise AttributeError("market_info must be populated after init_exchange")
        self.min_order_size_base: Decimal = self._get_market_decimal('min_order_size', what="min order size")
        self.min_amount_step: Decimal = self._get_market_decimal('amount_step', what="amount step")
        self.price_tick_size: Decimal = self._get_market_decimal('tick_size', what="tick size")
        self.contract_size: Decimal = self._get_market_decimal('contract_size', what="contract size")
        if any(v.is_nan() for v in [self.min_order_size_base, self.min_amount_step, self.price_tick_size, self.contract_size]): raise ValueError(f"Some required MARKET_INFO value for {self.config.symbol} invalid post validation. Halting.")
        self.settle_currency: str = self.config.symbol.split(":")[-1]

    def _init_exchange(self) -> ccxt.Exchange:
        logger.info(f"{NEON['HEADING']}Summoning Exchange Jinn...{NEON['RESET']}")
        exchange_init_start = time.monotonic()
        try:
            params = {
                "apiKey": self.config.api_key, "secret": self.config.api_secret,
                "options": {"defaultType": self.config.market_type, "adjustForTimeDifference": True, "recvWindow": self.config.default_recv_window, "brokerId": "PyrmethusV3", "defaultTimeInForce": "GTC"}
            }
            bybit = ccxt.bybit(params)
            markets = api_retry_logic(bybit.load_markets, True)()
            if not markets or not (self.config.symbol in bybit.markets): raise ccxt.ExchangeError(f"Could not load markets or symbol {self.config.symbol} not found.")
            self.config.MARKET_INFO = bybit.market(self.config.symbol)
            if self.config.MARKET_INFO.get("contractSize") is None: logger.warning(f"{NEON['WARNING']}Symbol {self.config.symbol} contract size is None, defaulting to 1. May be incorrect for inverse.{NEON['RESET']}")
            exchange_init_elapsed = time.monotonic() - exchange_init_start
            logger.info(f"{NEON['SUCCESS']}Exchange Jinn materialized in {exchange_init_elapsed:.3f} seconds.{NEON['RESET']}")
            return bybit
        except ccxt.AuthenticationError as e: logger.critical(f"{NEON['CRITICAL']}Authentication failed: {e}. Check API keys/permissions.{NEON['RESET']}", exc_info=False); notify_termux("Pyrmethus FATAL", "Authentication failed: Check keys/perms."); sys.exit(1)
        except Exception as e: logger.critical(f"{NEON['CRITICAL']}Failed to initialize Bybit exchange: {e}{NEON['RESET']}", exc_info=True); notify_termux("Pyrmethus FATAL", f"Exchange init: {str(e)[:100]}"); sys.exit(1)

    def _get_market_decimal(self, key: str, what: str) -> Decimal:
        val = safe_decimal(self.config.MARKET_INFO.get(key), Decimal('NaN')) # type: ignore
        if val.is_nan() or val <= 0: logger.critical(f"{NEON['CRITICAL']}CRITICAL: Bad market info: '{key}' ({val.normalize() if not val.is_nan() else 'NaN'}) for {what}.{NEON['RESET']}"); return Decimal('NaN')
        return val

    def update_time(self):
        try:
            self._now_time = datetime.fromtimestamp(api_retry_logic(self.bybit.fetch_time)() / 1000, tz=self.time_zone)
            self.log_heartbeat()
        except Exception as e: logger.warning(f"{NEON['WARNING']}Failed to update time: {e}{NEON['RESET']}", exc_info=True)

    def log_heartbeat(self, log_method: Callable[[str], None] = logger.info):
        log_method(f"{Fore.WHITE}Thump-Thump...{Style.RESET_ALL} ({datetime.now(self.time_zone).strftime('%Y-%m-%d %H:%M:%S %Z')})") # Use current system time for heartbeat log

    def fetch_ohlcv(self, since: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        cache_key = f"{self.config.symbol}-{self.config.interval}"
        df_cache = self.candles_cache.get(cache_key)
        current_time_monotonic = time.monotonic()

        cache_valid = False
        if df_cache is not None and not df_cache.empty:
            elapsed_since_last_fetch = current_time_monotonic - _last_market_data_fetch_ts.get(cache_key, 0)
            time_since_last_candle = (datetime.now(self.time_zone) - df_cache.index[-1].to_pydatetime()).total_seconds()
            # Cache is valid if fetched recently AND last candle is not too old relative to interval
            interval_seconds = self.bybit.parse_timeframe(self.config.interval)
            if elapsed_since_last_fetch < self.config.sleep_seconds * 0.9 and time_since_last_candle < interval_seconds * 1.5 :
                cache_valid = True

        if cache_valid and df_cache is not None: # df_cache type guard
            logger.debug(f"FromCache: Candles from {df_cache.index[0]} to {df_cache.index[-1]} ({len(df_cache)} candles).")
            return df_cache.copy()

        df_recent = get_recent_ohlcv(self.bybit, self.config.symbol, self.config.interval, self.config.ohlcv_limit, custom_since=since)
        if df_recent is not None and not df_recent.empty:
            self.candles_cache[cache_key] = df_recent.copy() # Store a copy in cache
            _last_market_data_fetch_ts[cache_key] = current_time_monotonic
            logger.info(f"{NEON['SUCCESS']}Fetched and cached new candles for {self.config.symbol}.{NEON['RESET']}")
            return df_recent.copy() # Return another copy
        else:
            return None

    def get_orderbook(self) -> Optional[Dict[str, List[List[float]]]]:
        try:
            orderbook = api_retry_logic(self.bybit.fetch_order_book, self.config.symbol)(limit=self.config.order_book_fetch_limit)
            if not orderbook or not (orderbook.get('bids') or orderbook.get('asks')): logger.warning("Empty OB Data. Network/Exchange issue?"); return None
            return orderbook
        except Exception as e: logger.error(f"Orderbook fetch error: {e}", exc_info=True); return None

    def set_leverage(self, leverage: int) -> bool:
        try:
            api_retry_logic(self.bybit.set_leverage, self.config.symbol, leverage)() # Pass leverage as positional arg
            logger.info(f"Leverage set to {log_format(leverage,0)}x.")
            return True
        except Exception as e: logger.error(f"Leverage setting failed: {e}", exc_info=True); return False

    def adjust_balance_to_min_trade_size(self, balance: Decimal, current_price: Decimal) -> Decimal:
        if balance <= 0: return Decimal("0")
        if self.contract_size <= 0 or self.price_tick_size <= 0 or current_price <= 0: return Decimal("0")

        # Estimate a reasonable quantity based on a fraction of balance to avoid over-sizing
        # This is a heuristic; actual sizing depends on risk % and SL distance
        # Let's assume a hypothetical SL distance for this check (e.g., 1% of price)
        hypothetical_sl_distance = current_price * Decimal("0.01")
        value_per_point = self.contract_size if self.config.market_type == "linear" else self.contract_size / current_price
        risk_per_contract_hypothetical = hypothetical_sl_distance * value_per_point
        
        if risk_per_contract_hypothetical <= 0: return Decimal("0")

        # Max affordable quantity based on full balance and leverage (ignoring risk % for this check)
        max_affordable_qty = (balance * self.config.leverage) / (current_price * self.contract_size)
        
        target_qty_base = min(max_affordable_qty, CONFIG.max_order_usdt_amount / current_price) # Cap by USDT limit
        target_qty_base_fmt = self.format_amount(target_qty_base,  rounding_mode=ROUND_DOWN)
        target_qty = safe_decimal(target_qty_base_fmt)

        if target_qty is not None and not target_qty.is_nan() and target_qty >= self.min_order_size_base:
            return balance # Current balance allows trades above min size
        else:
            # If calc qty is below min, determine what balance would be needed for min size
            # Min order value in quote = min_order_size_base * current_price * contract_size
            # Required balance = (Min order value / leverage) * margin_buffer
            min_order_value_quote = self.min_order_size_base * current_price * self.contract_size
            required_balance_for_min = (min_order_value_quote / self.config.leverage) * self.config.required_margin_buffer
            logger.warning(f"Adjusting balance for min trade size check: Min order requires ~{required_balance_for_min.normalize()} {self.settle_currency}. Current available: {balance.normalize()}")
            return required_balance_for_min # Return the balance needed for min size

    def create_order(self, side: str, qty: Decimal, order_type: str = "market", price: Optional[Decimal]=None, reduce_only: bool = False, params: Dict[str, Any] = {}) -> Dict[str, Any]:
        qty_str_api = self.format_amount(qty, rounding_mode=ROUND_DOWN)
        final_qty_decimal = safe_decimal(qty_str_api)

        if final_qty_decimal is None or final_qty_decimal.is_nan() or final_qty_decimal <= 0:
            logger.error(f"Invalid quantity for order: {qty_str_api} (Original: {qty})"); return {}
        if final_qty_decimal < self.min_order_size_base:
            logger.warning(f"Order size {final_qty_decimal.normalize()} less than min {self.min_order_size_base.normalize()}. Order Aborted."); return {}
        qty_float_api = float(final_qty_decimal)

        params_final = copy.deepcopy(params)
        if reduce_only: params_final["reduceOnly"] = True
        if self.config.use_post_only_orders and order_type.lower() == "limit": params_final["postOnly"] = True

        try:
            log_order_params = f"side: {side.upper()}, amount: {qty_float_api}, type: {order_type}"
            if price: log_order_params += f", price: {self.format_price(price)}"
            log_order_params += f", reduceOnly: {reduce_only}, params: {params_final}"
            logger.info(f"{Fore.CYAN}Attempting order: {log_order_params}{Style.RESET_ALL}")

            created_order = api_retry_logic(self.bybit.create_order, symbol=self.config.symbol, type=order_type, side=side, amount=qty_float_api, price=float(price) if price else None, params=params_final)()
            if not created_order: logger.error("Order creation returned None."); return {}
            logger.info(f"{Fore.GREEN}Order created: {self.format_order(created_order, qty_deci=final_qty_decimal)}{Style.RESET_ALL}")
            return created_order
        except Exception as e: logger.error(f"{Fore.RED}Order creation failed: {e}{Style.RESET_ALL}", exc_info=True); return {}

    def format_order(self, order: Dict[str, Any], qty_deci: Decimal, p_decimal_places: Optional[int] = None) -> str:
         qty_str = self.format_amount(qty_deci)
         final_qty = f"Qty({qty_str})"; p_disp = p_decimal_places or CONFIG.MARKET_INFO['precision']['price'] if CONFIG.MARKET_INFO else 4 # type: ignore
         return f"ID({short_oid(order.get('id'))}) {order['side'].upper()} {self.config.symbol} {final_qty}, Filled {order.get('filled',0)}, AvgFill={safe_decimal(order.get('average', 0)):.{p_disp}f}, Status {order.get('status')}"

    def set_position_protection_v5(self, side_string: str, sl_price: Optional[Decimal] = None, tp_price: Optional[Decimal] = None, tsl_distance: Optional[Decimal] = None, trigger_price_tsl_activation: Optional[Decimal] = None) -> bool:
        if not self.config.MARKET_INFO: logger.error("set_position_protection_v5: MarketInfo missing, cannot set protection."); return False
        cat = self.config.market_type # V5 category often matches market_type for futures
        pos_idx = self.config.position_idx
        market_id = self.config.MARKET_INFO.get("id")
        sl_str = self.format_price(sl_price) if sl_price and sl_price > 0 else "0"
        tp_str = self.format_price(tp_price) if tp_price and tp_price > 0 else "0"

        tsl_dist_str = self.format_price(tsl_distance) if tsl_distance and tsl_distance > 0 else "0"
        tsl_act_price_str = self.format_price(trigger_price_tsl_activation) if trigger_price_tsl_activation and trigger_price_tsl_activation > 0 else "0"

        params = {
            "category": cat, "symbol": market_id, "positionIdx": pos_idx,
            "stopLoss": sl_str, "takeProfit": tp_str,
            "trailingStop": tsl_dist_str, "activePrice": tsl_act_price_str,
            "slTriggerBy": self.config.sl_trigger_by, "tpTriggerBy": self.config.sl_trigger_by,
            "tpslMode": "Full" # or "Partial"
            # "triggerBy": self.config.tsl_trigger_by # 'triggerBy' is often general, sl/tp/tslTriggerBy are more specific
        }
        # Clean up params: remove keys with "0" value if API expects them to be absent to clear
        params_cleaned = {k: v for k, v in params.items() if v != "0" or k in ["positionIdx", "category", "symbol", "tpslMode"]}
        if sl_str != "0": params_cleaned["stopLoss"] = sl_str # Ensure it's present if non-zero
        if tp_str != "0": params_cleaned["takeProfit"] = tp_str
        if tsl_dist_str != "0": params_cleaned["trailingStop"] = tsl_dist_str
        if tsl_act_price_str != "0": params_cleaned["activePrice"] = tsl_act_price_str


        try:
            logger.debug(f"Set V5 position protection: private_post_position_set_trading_stop {params_cleaned=}")
            protected = api_retry_logic(self.bybit.private_post_position_set_trading_stop, params=params_cleaned)()
            if not (protected and protected.get('retCode') == 0):
                logger.error(f"Set V5 pos protection API fail. Resp: {protected}")
                return False
            logger.info(f"{Fore.GREEN}Set V5 protection success {side_string.upper()}:  SL={sl_str} TP={tp_str} TSL={f'{tsl_dist_str} @ {tsl_act_price_str}' if tsl_dist_str != '0' else 'N/A'} {Style.RESET_ALL}")
            return True
        except Exception as e: logger.error(f"Set V5 pos protection error: {e}", exc_info=True); return False

    def close_position(self, side_to_close: str, qty_to_close: Decimal, reason: str = "Strategy Close") -> Dict[str, Any]:
        params: Dict[str, Any] = {"reduceOnly": True}
        # Determine opposite side for closing order
        closing_side = CONFIG.side_sell if side_to_close.lower() == CONFIG.pos_long.lower() else CONFIG.side_buy

        order_info = self.create_order(
            side=closing_side,
            qty=qty_to_close,
            order_type="market",
            params=params
        )
        # Log the closure attempt with reason
        if order_info and order_info.get("id"):
            logger.trade(f"{NEON['ACTION']}Attempted CLOSE of {side_to_close.upper()} position (Qty: {qty_to_close.normalize()}) for reason: {reason}. Order ID: {short_oid(order_info['id'])}{NEON['RESET']}")
        else:
            logger.error(f"{NEON['ERROR']}Failed to submit CLOSE order for {side_to_close.upper()} position (Qty: {qty_to_close.normalize()}). Reason: {reason}{NEON['RESET']}")
        return order_info

    def format_price(self, price: Optional[Decimal]) -> str:
        if price is None or price.is_nan(): return "0" # Default for API if not set
        price_dp = CONFIG.MARKET_INFO['precision']['price'] if CONFIG.MARKET_INFO else 4 # type: ignore
        return f"{price:.{price_dp}f}"

    def format_amount(self, amount: Decimal, rounding_mode = ROUND_DOWN) -> str:
        amount_dp = CONFIG.MARKET_INFO['precision']['amount'] if CONFIG.MARKET_INFO else 6 # type: ignore
        quantizer = Decimal('1e-' + str(amount_dp))
        return str(amount.quantize(quantizer, rounding=rounding_mode))

# Initialize ExchangeManager after CONFIG is loaded
try:
    exchange_manager = Exchange(CONFIG)
except Exception as e:
    logging.getLogger("Main").critical(f"{NEON['CRITICAL']}ExchangeManager Init Failed: {e}{NEON['RESET']}", exc_info=True)
    notify_termux("Pyrmethus FATAL", f"ExchangeManager Init: {e}")
    sys.exit(1)

# --- Trading Bot Class (Continued from previous response) ---
# (Ensure TradingBot class definition is complete with all methods as shown before)
# ... (Previous TradingBot methods: _setup_signal_handlers, _signal_handler, _display_startup_info, _get_position_summary, _get_bybit_balance_for_sizing, _calculate_atr_level, _time_allowed_to_trade, _emergency_stop_requested, _log_and_display_status) ...

class TradingBot:
    """Main orchestration class for the Pyrmethus trading bot."""

    def __init__(self):
        logger_core = logging.getLogger("PyrmethusCore")
        logger_core.info(f"{NEON['HEADING']}--- Initializing Pyrmethus v{VERSION} Grand Unified Scalping Spell ---{NEON['RESET']}")
        try:
            # CONFIG is already a global instance, initialized earlier
            self.config = CONFIG
            self.exchange_manager = exchange_manager # Use the globally initialized ExchangeManager
            self.order_manager = OrderManager(self.config, self.exchange_manager)
            self.signal_generator = CONFIG.strategy_instance # Strategy instance from CONFIG
        except Exception as e: logger_core.critical(f"{NEON['CRITICAL']}Failed to cast spell components: {e}{NEON['RESET']}", exc_info=True); notify_termux("Pyrmethus FATAL", f"Init fail: {e}"); sys.exit(1)

        self.state_manager = state_manager
        self.trade_metrics: TradeMetrics = trade_metrics
        self.status_display = StatusDisplay(self.config) # StatusDisplay now part of TradingBot

        self.consecutive_losses_count: int = self.state_manager.get("consecutive_losses_count", 0)
        self.last_loss_timestamp: float = self.state_manager.get("last_loss_timestamp", 0.0)
        self.last_trade_side: str = self.state_manager.get("last_trade_side", "NONE")
        self.global_qty_scale_multiplier: Decimal = safe_decimal(self.state_manager.get("global_qty_scale_multiplier", "1.0"), Decimal("1.0")) # type: ignore

        global _whipsaw_cooldown_active_until, _trade_timestamps_for_whipsaw
        _whipsaw_cooldown_active_until = self.state_manager.get("_whipsaw_cooldown_active_until", 0.0)
        _trade_timestamps_for_whipsaw = deque(self.state_manager.get("_trade_timestamps_for_whipsaw", []), maxlen=CONFIG.whipsaw_max_trades_in_period)

        self._active_trade_start_time: Optional[float] = None # Track start time of current trade part
        self._currentCycleStrLogAction: str = "" # For display panel status line
        self._is_order_processing: bool = False # Simple mutex for order actions
        self._curr_order_details: Dict[str, Any] = {} # Store details of last active order

        self.shutdown_requested: bool = False; self._setup_signal_handlers()
        logger_core.info(f"{NEON['SUCCESS']}Core systems nominal. Trading spell is hot.{NEON['RESET']}")

    def _setup_signal_handlers(self) -> None:
        for sig in [signal.SIGINT, signal.SIGTERM]:
            try: signal.signal(sig, self._signal_handler); logger.debug(f"Summoned signal handler for {signal.Signals(sig).name} guardian.")
            except Exception as e: logger.warning(f"{Fore.YELLOW}Could not conjure handler for {sig}: {e}{Style.RESET_ALL}")

    def _signal_handler(self, sig_num: int, _frame: Optional[Any]) -> None:
        if self.shutdown_requested: logger.warning("Already shutting down, sir."); return
        self.shutdown_requested = True
        sig_name = signal.Signals(sig_num).name if hasattr(signal, 'Signals') else f"Signal {sig_num}"
        console.print(f"\n[bold yellow]Signal {sig_name} received. Engaging graceful shutdown...[/]")
        logger.warning(f"Signal {sig_name} received. Commencing graceful shutdown.")

    def _display_startup_info(self) -> None:
        msg = f"Symbol:{self.config.symbol}, Interval:{self.config.interval}, " \
              f"Strategy:{self.config.strategy_name.value}, Leverage:{self.config.leverage}x, " \
              f"Risk Pct:{self.config.risk_per_trade_percentage:.3%}, PosIdx:{self.config.position_idx}  " \
              f"RunID:{os.environ.get('RUN_ID', 'N/A')}"
        logger.info(msg)
        console.print(f"{NEON['HEADING']}Pyrmethus Scalping Spell ({VERSION}){NEON['RESET']}: {msg}")

    def _get_position_summary(self) -> Tuple[Optional[str], Decimal, Decimal]:
        pos_data = api_retry_logic(self.exchange_manager.get_current_position)()
        if pos_data is None: logger.error("get_position_summary: Could not fetch or parse position."); return None, Decimal("0"), Decimal("NaN")
        long_pos, short_pos = pos_data.get("long", {}), pos_data.get("short", {})
        long_qty_d, short_qty_d = safe_decimal(long_pos.get("qty","0"), Decimal(0)), safe_decimal(short_pos.get("qty","0"), Decimal(0)) # type: ignore

        pos_qty = Decimal("0")
        pos_side: Optional[str] = None
        entry_px = Decimal("NaN")

        if long_qty_d.copy_abs() >= CONFIG.position_qty_epsilon: # type: ignore
            pos_qty = long_qty_d
            pos_side = CONFIG.pos_long
            entry_px = safe_decimal(long_pos.get("entry_price","NaN"))
            if short_qty_d.copy_abs() >= CONFIG.position_qty_epsilon: # type: ignore
                 logger.warning(f"Inconsistent pos state: Both long ({long_qty_d}) and short ({short_qty_d}) qty. Prioritizing {pos_side}.")
        elif short_qty_d.copy_abs() >= CONFIG.position_qty_epsilon: # type: ignore
            pos_qty = short_qty_d
            pos_side = CONFIG.pos_short
            entry_px = safe_decimal(short_pos.get("entry_price","NaN"))

        if pos_side and entry_px.is_nan(): logger.warning(f"Entry price is NaN for existing {pos_side.upper()} position. Inconsistent state?")
        return pos_side, pos_qty, entry_px

    def _get_bybit_balance_for_sizing(self) -> Tuple[Decimal, Decimal]:
        total_equity, available_balance = self.exchange_manager.get_balance()
        if total_equity is None or total_equity.is_nan() or total_equity <= 0: raise ValueError(f"Equity invalid {total_equity}")
        current_price_for_adj = self.exchange_manager.bybit.fetch_ticker(self.config.symbol)['last'] if self.exchange_manager.bybit else Decimal(0)
        adj_balance = self.exchange_manager.adjust_balance_to_min_trade_size(available_balance if available_balance else Decimal(0), safe_decimal(current_price_for_adj))
        return total_equity, adj_balance

    def _calculate_atr_level(self, df: pd.DataFrame, period: int) -> Decimal:
        try:
            tr_s = ta.true_range(high=df["high"], low=df["low"], close=df["close"])
            atr_s = tr_s.rolling(window=period).mean()
            atr = safe_decimal(atr_s.iloc[-1])
            if atr is None or atr.is_nan() or atr <= 0: raise ValueError(f"Invalid ATR {atr} for SL distance.") # type: ignore
            return atr # type: ignore
        except Exception as atr_error: logger.error(f"SL Level Calc skipped (bad data): {atr_error}"); return Decimal("NaN")

    def _time_allowed_to_trade(self, now_dt: datetime) -> bool:
        if not self.config.trading_allowed_hours_utc: return True
        try:
            allowed_ranges_str = self.config.trading_allowed_hours_utc
            for range_str in allowed_ranges_str.split(','):
                start_hour, end_hour = map(int, range_str.split('-'))
                if start_hour <= now_dt.hour < end_hour: return True
            return False
        except Exception as e:
            logger.error(f"Error evaluating trade time restrictions: {e}", exc_info=True)
            return True

    def _emergency_stop_requested(self) -> bool:
        global _stop_trading_flag
        if _stop_trading_flag: return True
        if Path(self.config.emergency_stop_file_path).exists():
            _stop_trading_flag = True
            logger.critical(f"{Style.BRIGHT}{Fore.RED}Emergency stop requested via file: {self.config.emergency_stop_file_path}. Halting trading.{Style.RESET_ALL}")
            notify_termux("Pyrmethus HALT", "Emergency stop requested via file")
            return True
        return False

    def _log_and_display_status(self, cycle: int, current_price: Decimal, cycle_status: str,
                                indicators: Optional[Dict[str, Any]] = None,
                                positions_summary: Optional[Dict[str, Dict[str, Any]]] = None,
                                equity: Optional[Decimal] = None,
                                signals: Optional[Dict[str, Any]] = None):
        """Consolidated status logging and display."""
        # Fetch latest position summary if not provided (e.g., after an action)
        if positions_summary is None:
            positions_summary = self.exchange_manager.get_current_position()

        # Fetch latest equity if not provided
        if equity is None:
            equity, _ = self._get_bybit_balance_for_sizing()


        # Default signals if none provided (e.g., during error states)
        if signals is None:
            signals = {"summary": "Processing...", "long": False, "short": False, "orig_detail": "N/A", "vt_detail": "N/A"}

        self.status_display.print_status_panel(
            cycle_num=cycle,
            current_timestamp=self.exchange_manager._now_time,
            current_market_price=current_price,
            indicators_data=indicators,
            current_positions_summary=positions_summary,
            account_equity=equity,
            signal_check_result=signals,
            protection_status_tracker=self.order_manager.protection_tracker,
            market_specific_info=self.exchange_manager.market_info
        )
        logger.info(f"Cycle {cycle}: Status - {cycle_status} | Price: {log_format(current_price)} | Equity: {log_format(equity, p=2)}")

    def _determine_order_qty(self, price: Decimal, equity: Decimal, atr: Decimal, adjust_for_existing: bool = False, new_risk_override: Optional[Decimal] = None) -> Decimal:
        min_qty_base = self.exchange_manager.min_order_size_base
        if not self.config.MARKET_INFO or any(v is None or (isinstance(v, Decimal) and v.is_nan()) for v in [atr, price, equity, min_qty_base]):
            logger.error(f"Cannot determine order size: Missing data or invalid state."); return Decimal("0")

        effective_risk_pct = new_risk_override if new_risk_override is not None else self.config.risk_per_trade_percentage
        risk_per_trade_quote = equity * effective_risk_pct

        # Determine ATR multiplier based on volatility regime if dynamic SL is enabled
        atr_sl_mult = self.config.atr_stop_loss_multiplier
        if self.config.enable_dynamic_atr_sl:
            vol_regime_str = _vol_atr_analysis_results_cache.get('volatility_regime', VolatilityRegime.NORMAL.value) # Default to normal
            vol_regime = VolatilityRegime(vol_regime_str) # Convert string from cache to Enum member
            if vol_regime == VolatilityRegime.LOW: atr_sl_mult = self.config.atr_sl_multiplier_low_vol
            elif vol_regime == VolatilityRegime.HIGH: atr_sl_mult = self.config.atr_sl_multiplier_high_vol
            else: atr_sl_mult = self.config.atr_sl_multiplier_normal_vol # Normal or fallback
            logger.debug(f"Dynamic SL ATR multiplier: {atr_sl_mult} for regime: {vol_regime.value}")


        sl_dist_quote = atr * atr_sl_mult
        if sl_dist_quote <= self.exchange_manager.price_tick_size: # Ensure SL distance is at least one tick
            logger.warning(f"Calculated SL distance ({sl_dist_quote}) too small, adjusting to min tick ({self.exchange_manager.price_tick_size}).")
            sl_dist_quote = self.exchange_manager.price_tick_size
        if sl_dist_quote <= Decimal("0"): logger.error("SL distance is zero or negative."); return Decimal("0")

        # Calculate quantity in base asset
        # For linear contracts: Qty_Base = Risk_Quote / SL_Distance_Quote (assuming contract size 1 for quote value per point)
        # For inverse contracts: Qty_Base = Risk_Quote / (SL_Distance_Quote * Price_Quote_per_Base)
        # Simplified: Qty_Base = Risk_Quote / (SL_Distance_Quote * Value_of_One_Base_Unit_in_Quote_Terms_at_SL)
        # More general: Risk_Quote / (SL_Distance_Quote_per_Base_Unit * ContractSize_Base_per_Contract)
        # Bybit linear: ContractSize is usually 1 (e.g., 1 BTC for BTC/USDT), value is in quote.
        # Bybit inverse: ContractSize is in quote (e.g., 1 USD for BTC/USD), value is in base.

        qty_base: Decimal
        if self.config.market_type == "linear":
            # Risk is in Quote (USDT). SL distance is in Quote. Contract size is in Base (BTC).
            # Value of 1 point move for 1 contract = ContractSize_Base * 1 (if price is quote/base)
            # Here, we want Qty_Base.
            # Risk_Quote = Qty_Base * SL_Distance_Quote
            if sl_dist_quote == 0: return Decimal("0")
            qty_base = risk_per_trade_quote / sl_dist_quote
        elif self.config.market_type == "inverse":
            # Risk is in Quote (USD). SL distance is in Quote. Contract size is in Quote (USD).
            # Qty is in Base (BTC).
            # Risk_Quote = Qty_Base * Price_Base_Quote * SL_Distance_Quote / Price_Base_Quote
            # Risk_Quote = Qty_Base * SL_Distance_Quote
            if sl_dist_quote == 0: return Decimal("0")
            qty_base = risk_per_trade_quote / sl_dist_quote
        else:
            logger.error(f"Unsupported market type '{self.config.market_type}' for quantity calculation."); return Decimal("0")


        # Format and validate quantity against exchange limits
        qty_base_formatted_str = self.exchange_manager.format_amount(qty_base, rounding_mode=ROUND_DOWN)
        final_qty_base = safe_decimal(qty_base_formatted_str)

        if final_qty_base is None or final_qty_base.is_nan() or final_qty_base <= Decimal("0"):
            logger.warning(f"Calculated quantity is invalid or zero after formatting: {qty_base_formatted_str} (Raw: {qty_base})"); return Decimal("0")

        if final_qty_base < min_qty_base:
            logger.warning(f"Calculated quantity {final_qty_base.normalize()} is below minimum {min_qty_base.normalize()}. No trade."); return Decimal("0")

        # Cap by max order USDT amount
        if self.config.max_order_usdt_amount > 0:
            max_qty_by_usdt_limit = self.config.max_order_usdt_amount / price # Max base qty allowed by USDT limit
            if final_qty_base > max_qty_by_usdt_limit:
                logger.info(f"Quantity capped by MAX_ORDER_USDT_AMOUNT from {final_qty_base.normalize()} to {max_qty_by_usdt_limit.normalize()}")
                final_qty_base = safe_decimal(self.exchange_manager.format_amount(max_qty_by_usdt_limit, ROUND_DOWN)) # type: ignore
                if final_qty_base is None or final_qty_base.is_nan() or final_qty_base < min_qty_base: # Re-check min after capping
                    logger.warning(f"Quantity after USDT cap ({final_qty_base}) is below minimum. No trade."); return Decimal("0")
        
        logger.info(f"Determined Order Qty: {log_format(final_qty_base, p=self.exchange_manager.config.MARKET_INFO['precision']['amount'])} Base Asset. Risk: {log_format(effective_risk_pct*100, p=3)}%, SL Dist: {log_format(sl_dist_quote)}") # type: ignore
        return final_qty_base # type: ignore

    def _check_enough_margin(self, orderQty: Decimal, current_price: Decimal, available_balance: Decimal) -> bool:
        if orderQty <= 0: return True # No margin needed for zero quantity
        if not self.config.MARKET_INFO or not self.config.MARKET_INFO.get('contractSize') or self.config.leverage <= 0:
            logger.error("Margin Check: Market info, contract size, or leverage missing/invalid."); return False

        # Calculate order value in quote currency
        contract_size_dec = self.exchange_manager.contract_size
        order_value_quote = orderQty * current_price * contract_size_dec # If contractSize is in base, this is fine for linear.
        if self.config.market_type == "inverse": # For inverse, contractSize is in quote, qty is in base. Value = Qty_Base * Price_Quote_per_Base * ContractSize_Quote_per_Contract_Unit (usually 1)
             order_value_quote = orderQty * current_price # Assuming contractSize is 1 USD for inverse contracts

        required_margin = (order_value_quote / self.config.leverage) * self.config.required_margin_buffer # Add buffer

        if available_balance < required_margin:
            logger.warning(f"{NEON['WARNING']}Insufficient available margin for trade. Need: {log_format(required_margin, p=2)} {self.exchange_manager.settle_currency}, Have: {log_format(available_balance, p=2)} {self.exchange_manager.settle_currency}.{NEON['RESET']}")
            return False
        
        logger.debug(f"Margin check OK. Need: {log_format(required_margin, p=2)}, Have: {log_format(available_balance, p=2)}")
        return True

    def run(self): # type: ignore[override]
        self._display_startup_info()
        termux_notify("Pyrmethus Engaged", f"Trading v{VERSION} initiated.")

        if "RUN_ID" not in os.environ: os.environ["RUN_ID"] = time.strftime('%Y%m%d_%H%M%S')

        cycle_counter = 0
        main_loop_error_count = 0

        try:
            global trade_metrics
            old_session_start_time_st = self.state_manager.get("metrics.session_start_time")
            if old_session_start_time_st: trade_metrics.session_start_time = datetime.fromisoformat(old_session_start_time_st)
            equity_old = self.state_manager.get("metrics.max_equity")
            trade_metrics.max_equity = safe_decimal(equity_old, Decimal(0)) # type: ignore
        except Exception as e: logger.error(f"{Fore.RED}Could not restore trade metrics from state: {e}{Style.RESET_ALL}", exc_info=False)

        try:
            while not self.shutdown_requested:
                cycle_counter += 1
                cycle_start_monotonic = time.monotonic()
                logger.debug(f"{Fore.BLUE}--- Trading Cycle {cycle_counter} Start ---{Style.RESET_ALL}")

                try:
                    self._trading_cycle(cycle_counter)
                except KeyboardInterrupt:
                    logger.warning(f"KeyboardInterrupt in main loop. Initiating shutdown sequence."); self.shutdown_requested = True; break
                except Exception as e:
                    main_loop_error_count += 1
                    logger.error(f"{NEON['ERROR']}Trading cycle failed ({cycle_counter}): {e} {Style.RESET_ALL}", exc_info=True)
                    termux_notify("Pyrmethus Cycle Fail", f"C{cycle_counter} cycle exception. See Logs.")
                    if main_loop_error_count > 3: logger.critical(f"{NEON['CRITICAL']}Multiple main loop errors, suspected unrecoverable. Shutdown.{Style.RESET_ALL}"); self.shutdown_requested = True; break
                    sleep_after_fail = max(60, self.config.sleep_seconds * 5)
                    logger.warning(f"{Fore.YELLOW}Sleeping {sleep_after_fail}s after unhandled cycle error...{Style.RESET_ALL}"); time.sleep(sleep_after_fail); continue
                else:
                    main_loop_error_count = 0 # Reset on successful cycle

                cycle_time = time.monotonic() - cycle_start_monotonic
                sleep_time = max(0, self.config.sleep_seconds - cycle_time)
                if sleep_time > 0 and not self.shutdown_requested:
                    logger.debug(f"Sleeping for {sleep_time:.2f} seconds...")
                    try:
                        # Interruptible sleep
                        sleep_end_time = time.monotonic() + sleep_time
                        while time.monotonic() < sleep_end_time and not self.shutdown_requested:
                            time.sleep(min(0.2, sleep_time)) # Check frequently
                    except KeyboardInterrupt:
                        logger.warning("KeyboardInterrupt during sleep, shutting down."); self.shutdown_requested = True
        finally:
            self.graceful_shutdown()
            sys.exit(0) # Ensure a clean exit code for normal shutdown

# --- Main Execution ---
if __name__ == "__main__":
    try:
        bot = TradingBot()
        bot.run()
    except SystemExit as e:
        log_level = logging.INFO if e.code == 0 else logging.WARNING
        logging.getLogger("Main").log(log_level, f"Pyrmethus terminated (Exit Code: {e.code}).")
        # sys.exit(e.code) # Already exiting
    except Exception as main_exception:
        logging.getLogger("Main").critical(f"{NEON['CRITICAL']}CRITICAL UNHANDLED ERROR during bot execution: {main_exception}{NEON['RESET']}", exc_info=True)
        notify_termux("Pyrmethus CRITICAL ERROR", f"Bot failed: {str(main_exception)[:100]}")
        sys.exit(1)
