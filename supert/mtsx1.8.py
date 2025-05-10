#!/usr/bin/env python3
# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v2.9.0 (Aegis of Dynamic Wards)
# Enhancements: True dynamic SL/TP modification on exchange, ATR-based Take-Profit.
# Previous Weave: v2.8.3 (Enhanced Weave of Wisdom)

"""
High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 2.9.0 (Aegis of Dynamic Wards) - Enhanced by Pyrmethus

Dependencies & Setup (Termux):
1. Core Packages:
   pkg install python clang libcrypt libffi openssl rust termux-api
2. Python Libraries:
   pip install ccxt pandas pandas-ta colorama python-dotenv retry pytz

WARNING: This script executes live trades and modifies SL/TP on the exchange. Use with extreme caution.
Pyrmethus bears no responsibility for financial outcomes.

Enhancements in v2.9.0 (by Pyrmethus):
- Dynamic SL/TP Management:
    - Implemented `exchange.set_trading_stop()` for modifying SL and TP of open positions on Bybit.
    - Profit Momentum SL Tightening now attempts to modify SL on the exchange.
    - Breakeven SL now attempts to modify SL on the exchange.
- ATR-Based Take-Profit:
    - Added configuration for enabling and setting an ATR-based Take-Profit.
    - Initial TP is set when an order is placed.
- State Persistence:
    - Active SL and TP prices (as set on exchange) are now stored in the state file for each trade part.
- Configuration:
    - New parameters: `ENABLE_TAKE_PROFIT`, `ATR_TAKE_PROFIT_MULTIPLIER`.
- Robustness:
    - Added more checks for market info availability before critical operations.
    - Minor refinements to logging and error handling.
"""

# Standard Library Imports
import json
import logging
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
from decimal import ROUND_HALF_UP, Decimal, DivisionByZero, InvalidOperation, getcontext
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pytz  # The Chronomancer's ally

# Third-party Libraries - The Grimoire of External Magics
try:
    import ccxt
    import pandas as pd

    if not hasattr(
        pd, "NA"
    ):  # Pyrmethus: Ensure pandas version compatibility for pd.NA
        raise ImportError("Pandas version < 1.0 not supported, pd.NA is required.")
    import pandas_ta as ta  # The Alchemist's Table
    from colorama import Back, Fore, Style  # The Prisms of Perception
    from colorama import init as colorama_init
    from dotenv import load_dotenv  # For whispering secrets
    from retry import retry  # The Art of Tenacious Spellcasting
except ImportError as e:
    missing_pkg = getattr(
        e, "name", "a required dependency"
    )  # Slightly more descriptive default
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

# --- Constants - The Unchanging Pillars of the Spell ---
PYRMETHUS_VERSION = "2.9.0"
STATE_FILE_NAME = f"pyrmethus_phoenix_state_v{PYRMETHUS_VERSION.replace('.', '')}.json"
STATE_FILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), STATE_FILE_NAME
)
HEARTBEAT_INTERVAL_SECONDS = 60
OHLCV_LIMIT = 200  # Ensure this is enough for all indicator lookbacks

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
        f"{NEON['INFO']}Secrets whispered from .env scroll: {NEON['VALUE']}{env_path}{NEON['RESET']}"
    )
else:
    logging.getLogger("PreConfig").warning(
        f"{NEON['WARNING']}No .env scroll found at {NEON['VALUE']}{env_path}{NEON['WARNING']}. Relying on system environment variables or defaults.{NEON['RESET']}"
    )
getcontext().prec = 18  # Precision for Decimal arithmetic


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
    _logger = logging.getLogger("TermuxNotification")
    # Pyrmethus: Ensuring CONFIG is accessible or using a safe default for notification_timeout_seconds
    notification_timeout = (
        CONFIG.notification_timeout_seconds
        if "CONFIG" in globals() and hasattr(CONFIG, "notification_timeout_seconds")
        else 10
    )

    if (
        "CONFIG" not in globals()
        or not hasattr(CONFIG, "enable_notifications")
        or not CONFIG.enable_notifications
    ):
        if "CONFIG" in globals() and hasattr(CONFIG, "enable_notifications"):
            _logger.debug("Termux notifications are disabled by configuration.")
        else:
            _logger.warning(
                "Attempted Termux notification before CONFIG fully loaded or with notifications disabled."
            )
        return
    try:
        safe_title = json.dumps(title)
        safe_message = json.dumps(message)
        command = [
            "termux-notification",
            "--title",
            safe_title,
            "--content",
            safe_message,
            "--id",
            str(notification_id),
        ]
        _logger.debug(
            f"{NEON['ACTION']}Attempting to send Termux notification: Title='{title}'{NEON['RESET']}"
        )
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate(
            timeout=notification_timeout
        )  # Use determined timeout
        if process.returncode == 0:
            _logger.info(
                f"{NEON['SUCCESS']}Termux notification '{title}' sent successfully.{NEON['RESET']}"
            )
        else:
            err_msg = stderr.decode().strip() if stderr else "Unknown error"
            _logger.error(
                f"{NEON['ERROR']}Failed to send Termux notification '{title}'. Return code: {process.returncode}. Error: {err_msg}{NEON['RESET']}"
            )
    except FileNotFoundError:
        _logger.error(
            f"{NEON['ERROR']}Termux API command 'termux-notification' not found.{NEON['RESET']}"
        )
    except subprocess.TimeoutExpired:
        _logger.error(
            f"{NEON['ERROR']}Termux notification command timed out for '{title}'.{NEON['RESET']}"
        )
    except Exception as e:
        _logger.error(
            f"{NEON['ERROR']}Unexpected error sending Termux notification '{title}': {e}{NEON['RESET']}"
        )
        _logger.debug(traceback.format_exc())


# --- Enums ---
class StrategyName(str, Enum):
    DUAL_SUPERTREND_MOMENTUM = "DUAL_SUPERTREND_MOMENTUM"
    EHLERS_FISHER = "EHLERS_FISHER"


class VolatilityRegime(Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"  # Note: VolatilityRegime is defined but not actively used in the provided script.


class OrderEntryType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


# --- Configuration Class ---
class Config:
    def __init__(self) -> None:
        _pre_logger = logging.getLogger("ConfigModule")
        _pre_logger.info(
            f"{NEON['HEADING']}--- Summoning Configuration Runes v{PYRMETHUS_VERSION} ---{NEON['RESET']}"
        )
        # Core Exchange & Symbol
        self.api_key: str = self._get_env("BYBIT_API_KEY", required=True, secret=True)
        self.api_secret: str = self._get_env(
            "BYBIT_API_SECRET", required=True, secret=True
        )
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT")
        self.interval: str = self._get_env("INTERVAL", "1m")
        self.leverage: int = self._get_env("LEVERAGE", 10, cast_type=int)

        # Trading Behavior
        self.sleep_seconds: int = self._get_env("SLEEP_SECONDS", 10, cast_type=int)
        self.strategy_name: StrategyName = StrategyName(
            self._get_env(
                "STRATEGY_NAME", StrategyName.DUAL_SUPERTREND_MOMENTUM.value
            ).upper()
        )
        self.strategy_instance: "TradingStrategy"  # Populated after Config init
        self.risk_per_trade_percentage: Decimal = self._get_env(
            "RISK_PER_TRADE_PERCENTAGE", "0.005", cast_type=Decimal
        )  # 0.5%
        self.max_order_usdt_amount: Decimal = self._get_env(
            "MAX_ORDER_USDT_AMOUNT", "1000.0", cast_type=Decimal
        )

        # Stop-Loss & Take-Profit
        self.atr_stop_loss_multiplier: Decimal = self._get_env(
            "ATR_STOP_LOSS_MULTIPLIER", "1.5", cast_type=Decimal
        )
        self.enable_take_profit: bool = self._get_env(
            "ENABLE_TAKE_PROFIT", "true", cast_type=bool
        )  # New
        self.atr_take_profit_multiplier: Decimal = self._get_env(
            "ATR_TAKE_PROFIT_MULTIPLIER", "2.0", cast_type=Decimal
        )  # New

        # Order Management
        self.entry_order_type: OrderEntryType = OrderEntryType(
            self._get_env("ENTRY_ORDER_TYPE", OrderEntryType.MARKET.value).upper()
        )
        self.limit_order_offset_atr_percentage: Decimal = self._get_env(
            "LIMIT_ORDER_OFFSET_ATR_PERCENTAGE", "0.1", cast_type=Decimal
        )
        self.order_fill_timeout_seconds: int = self._get_env(
            "ORDER_FILL_TIMEOUT_SECONDS", 15, cast_type=int
        )

        # Risk & Session Management
        self.enable_max_drawdown_stop: bool = self._get_env(
            "ENABLE_MAX_DRAWDOWN_STOP", "true", cast_type=bool
        )
        self.max_drawdown_percent: Decimal = self._get_env(
            "MAX_DRAWDOWN_PERCENT", "0.05", cast_type=Decimal
        )  # 5%
        self.enable_session_pnl_limits: bool = self._get_env(
            "ENABLE_SESSION_PNL_LIMITS", "false", cast_type=bool
        )
        self.session_profit_target_usdt: Optional[Decimal] = self._get_env(
            "SESSION_PROFIT_TARGET_USDT", None, cast_type=Decimal, required=False
        )
        self.session_max_loss_usdt: Optional[Decimal] = self._get_env(
            "SESSION_MAX_LOSS_USDT", None, cast_type=Decimal, required=False
        )

        # Notifications & API
        self.enable_notifications: bool = self._get_env(
            "ENABLE_NOTIFICATIONS", "true", cast_type=bool
        )
        self.notification_timeout_seconds: int = self._get_env(
            "NOTIFICATION_TIMEOUT_SECONDS", 10, cast_type=int
        )
        self.default_recv_window: int = self._get_env(
            "DEFAULT_RECV_WINDOW", 13000, cast_type=int
        )  # Increased default

        # Internal Constants & Helpers
        self.side_buy: str = "buy"
        self.side_sell: str = "sell"
        self.pos_long: str = "Long"
        self.pos_short: str = "Short"
        self.pos_none: str = "None"
        self.usdt_symbol: str = "USDT"
        self.retry_count: int = 5
        self.retry_delay_seconds: int = 5
        self.api_fetch_limit_buffer: int = (
            20  # Extra candles to fetch beyond strict need
        )
        self.position_qty_epsilon: Decimal = Decimal("1e-9")  # For float comparisons
        self.post_close_delay_seconds: int = 3  # Wait after market close order
        self.MARKET_INFO: Optional[Dict[str, Any]] = None  # Populated at startup
        self.PAPER_TRADING_MODE: bool = self._get_env(
            "PAPER_TRADING_MODE", "false", cast_type=bool
        )
        self.send_notification_method = send_termux_notification
        self.tick_size: Optional[Decimal] = None  # Populated from MARKET_INFO
        self.qty_step: Optional[Decimal] = None  # Populated from MARKET_INFO

        # Generic Indicator Periods
        self.atr_calculation_period: int = self._get_env(
            "ATR_CALCULATION_PERIOD", 14, cast_type=int
        )

        # --- Dual Supertrend Momentum Strategy Params ---
        self.st_atr_length: int = self._get_env("ST_ATR_LENGTH", 10, cast_type=int)
        self.st_multiplier: Decimal = self._get_env(
            "ST_MULTIPLIER", "2.0", cast_type=Decimal
        )
        self.confirm_st_atr_length: int = self._get_env(
            "CONFIRM_ST_ATR_LENGTH", 20, cast_type=int
        )
        self.confirm_st_multiplier: Decimal = self._get_env(
            "CONFIRM_ST_MULTIPLIER", "3.0", cast_type=Decimal
        )
        self.momentum_period: int = self._get_env("MOMENTUM_PERIOD", 14, cast_type=int)
        self.momentum_threshold: Decimal = self._get_env(
            "MOMENTUM_THRESHOLD", "0", cast_type=Decimal
        )
        self.confirm_st_stability_lookback: int = self._get_env(
            "CONFIRM_ST_STABILITY_LOOKBACK", 3, cast_type=int
        )
        self.st_max_entry_distance_atr_multiplier: Optional[Decimal] = self._get_env(
            "ST_MAX_ENTRY_DISTANCE_ATR_MULTIPLIER",
            "0.5",
            cast_type=Decimal,
            required=False,
        )

        # --- Ehlers Fisher Strategy Params ---
        self.ehlers_fisher_length: int = self._get_env(
            "EHLERS_FISHER_LENGTH", 10, cast_type=int
        )
        self.ehlers_fisher_signal_length: int = self._get_env(
            "EHLERS_FISHER_SIGNAL_LENGTH", 1, cast_type=int
        )
        self.ehlers_fisher_extreme_threshold_positive: Decimal = self._get_env(
            "EHLERS_FISHER_EXTREME_THRESHOLD_POSITIVE", "2.0", cast_type=Decimal
        )
        self.ehlers_fisher_extreme_threshold_negative: Decimal = self._get_env(
            "EHLERS_FISHER_EXTREME_THRESHOLD_NEGATIVE", "-2.0", cast_type=Decimal
        )
        self.ehlers_enable_divergence_scaled_exit: bool = self._get_env(
            "EHLERS_ENABLE_DIVERGENCE_SCALED_EXIT", "false", cast_type=bool
        )
        self.ehlers_divergence_threshold_factor: Decimal = self._get_env(
            "EHLERS_DIVERGENCE_THRESHOLD_FACTOR", "0.75", cast_type=Decimal
        )
        self.ehlers_divergence_exit_percentage: Decimal = self._get_env(
            "EHLERS_DIVERGENCE_EXIT_PERCENTAGE", "0.3", cast_type=Decimal
        )

        # --- General Enhancements (Snippets) ---
        self.enable_profit_momentum_sl_tighten: bool = self._get_env(
            "ENABLE_PROFIT_MOMENTUM_SL_TIGHTEN", "false", cast_type=bool
        )
        self.profit_momentum_window: int = self._get_env(
            "PROFIT_MOMENTUM_WINDOW", 3, cast_type=int
        )
        self.profit_momentum_sl_tighten_factor: Decimal = self._get_env(
            "PROFIT_MOMENTUM_SL_TIGHTEN_FACTOR", "0.5", cast_type=Decimal
        )

        self.enable_whipsaw_cooldown: bool = self._get_env(
            "ENABLE_WHIPSAW_COOLDOWN", "true", cast_type=bool
        )
        self.whipsaw_max_trades_in_period: int = self._get_env(
            "WHIPSAW_MAX_TRADES_IN_PERIOD", 3, cast_type=int
        )
        self.whipsaw_period_seconds: int = self._get_env(
            "WHIPSAW_PERIOD_SECONDS", 300, cast_type=int
        )
        self.whipsaw_cooldown_seconds: int = self._get_env(
            "WHIPSAW_COOLDOWN_SECONDS", 180, cast_type=int
        )

        self.max_active_trade_parts: int = self._get_env(
            "MAX_ACTIVE_TRADE_PARTS", 1, cast_type=int
        )  # Defaulting to 1 for simpler SL/TP logic initially
        self.signal_persistence_candles: int = self._get_env(
            "SIGNAL_PERSISTENCE_CANDLES", 1, cast_type=int
        )

        self.enable_no_trade_zones: bool = self._get_env(
            "ENABLE_NO_TRADE_ZONES", "false", cast_type=bool
        )
        self.no_trade_zone_pct_around_key_level: Decimal = self._get_env(
            "NO_TRADE_ZONE_PCT_AROUND_KEY_LEVEL", "0.002", cast_type=Decimal
        )
        self.key_round_number_step: Optional[Decimal] = self._get_env(
            "KEY_ROUND_NUMBER_STEP", "1000", cast_type=Decimal, required=False
        )

        self.enable_breakeven_sl: bool = self._get_env(
            "ENABLE_BREAKEVEN_SL", "true", cast_type=bool
        )
        self.breakeven_profit_atr_target: Decimal = self._get_env(
            "BREAKEVEN_PROFIT_ATR_TARGET", "1.0", cast_type=Decimal
        )
        self.breakeven_min_abs_pnl_usdt: Decimal = self._get_env(
            "BREAKEVEN_MIN_ABS_PNL_USDT", "0.50", cast_type=Decimal
        )

        self.enable_anti_martingale_risk: bool = self._get_env(
            "ENABLE_ANTI_MARTINGALE_RISK", "false", cast_type=bool
        )
        self.risk_reduction_factor_on_loss: Decimal = self._get_env(
            "RISK_REDUCTION_FACTOR_ON_LOSS", "0.75", cast_type=Decimal
        )
        self.risk_increase_factor_on_win: Decimal = self._get_env(
            "RISK_INCREASE_FACTOR_ON_WIN", "1.1", cast_type=Decimal
        )
        self.max_risk_pct_anti_martingale: Decimal = self._get_env(
            "MAX_RISK_PCT_ANTI_MARTINGALE", "0.02", cast_type=Decimal
        )

        self.enable_last_chance_exit: bool = self._get_env(
            "ENABLE_LAST_CHANCE_EXIT", "false", cast_type=bool
        )
        self.last_chance_consecutive_adverse_candles: int = self._get_env(
            "LAST_CHANCE_CONSECUTIVE_ADVERSE_CANDLES", 2, cast_type=int
        )
        self.last_chance_sl_proximity_atr: Decimal = self._get_env(
            "LAST_CHANCE_SL_PROXIMITY_ATR", "0.3", cast_type=Decimal
        )

        self.enable_trend_contradiction_cooldown: bool = self._get_env(
            "ENABLE_TREND_CONTRADICTION_COOLDOWN", "true", cast_type=bool
        )
        self.trend_contradiction_check_candles_after_entry: int = self._get_env(
            "TREND_CONTRADICTION_CHECK_CANDLES_AFTER_ENTRY", 2, cast_type=int
        )
        self.trend_contradiction_cooldown_seconds: int = self._get_env(
            "TREND_CONTRADICTION_COOLDOWN_SECONDS", 120, cast_type=int
        )

        self.enable_daily_max_trades_rest: bool = self._get_env(
            "ENABLE_DAILY_MAX_TRADES_REST", "false", cast_type=bool
        )
        self.daily_max_trades_limit: int = self._get_env(
            "DAILY_MAX_TRADES_LIMIT", 10, cast_type=int
        )
        self.daily_max_trades_rest_hours: int = self._get_env(
            "DAILY_MAX_TRADES_REST_HOURS", 4, cast_type=int
        )

        self.enable_limit_order_price_improvement_check: bool = self._get_env(
            "ENABLE_LIMIT_ORDER_PRICE_IMPROVEMENT_CHECK", "true", cast_type=bool
        )

        self.enable_trap_filter: bool = self._get_env(
            "ENABLE_TRAP_FILTER", "false", cast_type=bool
        )
        self.trap_filter_lookback_period: int = self._get_env(
            "TRAP_FILTER_LOOKBACK_PERIOD", 20, cast_type=int
        )
        self.trap_filter_rejection_threshold_atr: Decimal = self._get_env(
            "TRAP_FILTER_REJECTION_THRESHOLD_ATR", "1.0", cast_type=Decimal
        )
        self.trap_filter_wick_proximity_atr: Decimal = self._get_env(
            "TRAP_FILTER_WICK_PROXIMITY_ATR", "0.2", cast_type=Decimal
        )

        self.enable_consecutive_loss_limiter: bool = self._get_env(
            "ENABLE_CONSECUTIVE_LOSS_LIMITER", "true", cast_type=bool
        )
        self.max_consecutive_losses: int = self._get_env(
            "MAX_CONSECUTIVE_LOSSES", 3, cast_type=int
        )
        self.consecutive_loss_cooldown_minutes: int = self._get_env(
            "CONSECUTIVE_LOSS_COOLDOWN_MINUTES", 60, cast_type=int
        )

        self._validate_parameters()
        _pre_logger.info(
            f"{NEON['HEADING']}--- Configuration Runes v{PYRMETHUS_VERSION} Summoned and Verified ---{NEON['RESET']}"
        )
        _pre_logger.info(
            f"{NEON['COMMENT']}# The chosen path: {NEON['STRATEGY']}{self.strategy_name.value}{NEON['RESET']}"
        )

    def _get_env(
        self,
        key: str,
        default: Any = None,
        cast_type: type = str,
        required: bool = False,
        color: str = NEON["PARAM"],
        secret: bool = False,
    ) -> Any:
        # Pyrmethus: This function remains a robust rune-reader.
        _logger = logging.getLogger("ConfigModule._get_env")
        value_str = os.getenv(key)
        source = "Env Var"
        value_to_cast: Any = None
        display_value = "********" if secret and value_str is not None else value_str

        if value_str is None:
            if required:
                _logger.critical(
                    f"{NEON['CRITICAL']}CRITICAL: Required config rune '{key}' not found. Pyrmethus cannot proceed.{NEON['RESET']}"
                )
                raise ValueError(f"Required environment variable '{key}' not set.")
            _logger.debug(
                f"{color}Config Rune {NEON['VALUE']}'{key}'{color}: Not Found. Using Default: '{NEON['VALUE']}{default}{color}'{NEON['RESET']}"
            )
            value_to_cast = default
            source = "Default"
        else:
            _logger.debug(
                f"{color}Config Rune {NEON['VALUE']}'{key}'{color}: Found Env Value: '{NEON['VALUE']}{display_value}{color}'{NEON['RESET']}"
            )
            value_to_cast = value_str

        if (
            value_to_cast is None
        ):  # Handles cases where default is None and env var is not set
            if required:  # This should have been caught earlier if value_str was None & required. This is for default=None.
                _logger.critical(
                    f"{NEON['CRITICAL']}CRITICAL: Required config rune '{key}' resolved to None after env/default check.{NEON['RESET']}"
                )
                raise ValueError(
                    f"Required environment variable '{key}' resolved to None."
                )
            _logger.debug(
                f"{color}Config Rune {NEON['VALUE']}'{key}'{color}: Final value is None (not required).{NEON['RESET']}"
            )
            return None

        final_value: Any
        try:
            raw_value_str_for_cast = str(
                value_to_cast
            )  # Ensures we are casting from a string representation
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
                final_value = int(
                    Decimal(raw_value_str_for_cast)
                )  # Cast to Decimal first for float strings
            elif cast_type == float:
                final_value = float(raw_value_str_for_cast)
            elif cast_type == str:
                final_value = raw_value_str_for_cast
            else:  # Should not happen if cast_type is from a controlled set (bool, Decimal, int, float, str)
                _logger.warning(
                    f"{NEON['WARNING']}Unsupported cast_type '{cast_type.__name__}' for key '{key}'. Returning raw string.{NEON['RESET']}"
                )
                final_value = raw_value_str_for_cast
        except (ValueError, TypeError, InvalidOperation) as e:
            _logger.error(
                f"{NEON['ERROR']}Invalid type/value for '{key}': '{value_to_cast}' (Source: {source}). Expected {cast_type.__name__}. Err: {e}. Using Default: '{default}'.{NEON['RESET']}"
            )
            if default is None:  # If casting initial value fails AND default is None
                if required:
                    _logger.critical(
                        f"{NEON['CRITICAL']}CRITICAL: Failed cast for required key '{key}', default is None.{NEON['RESET']}"
                    )
                    raise ValueError(
                        f"Required env var '{key}' failed casting, no valid default."
                    )
                else:  # Optional, failed cast, default is None -> final value is None
                    _logger.warning(
                        f"{NEON['WARNING']}Cast fail for optional '{key}', default is None. Final: None{NEON['RESET']}"
                    )
                    return None
            else:  # Try casting the default value
                source = "Default (Fallback)"
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
                        final_value = default_str_for_cast  # Fallback for unsupported cast_type on default
                    _logger.warning(
                        f"{NEON['WARNING']}Used casted default for {key}: '{NEON['VALUE']}{final_value}{NEON['WARNING']}'{NEON['RESET']}"
                    )
                except (ValueError, TypeError, InvalidOperation) as e_default:
                    _logger.critical(
                        f"{NEON['CRITICAL']}CRITICAL: Cast fail for value AND default for '{key}'. Original: '{value_to_cast}', Default: '{default}'. Err: {e_default}{NEON['RESET']}"
                    )
                    raise ValueError(
                        f"Config error: Cannot cast value or default for '{key}'."
                    )

        display_final_value = "********" if secret else final_value
        _logger.debug(
            f"{color}Final value for {NEON['VALUE']}'{key}'{color}: {NEON['VALUE']}{display_final_value}{color} (Type: {type(final_value).__name__}, Source: {source}){NEON['RESET']}"
        )
        return final_value

    def _validate_parameters(self) -> None:
        _logger = logging.getLogger("ConfigModule._validate")
        errors = []
        if not (0 < self.risk_per_trade_percentage < 1):
            errors.append(
                "RISK_PER_TRADE_PERCENTAGE must be between 0 and 1 (exclusive)."
            )
        if self.leverage < 1:
            errors.append("LEVERAGE must be at least 1.")
        if self.atr_stop_loss_multiplier <= 0:
            errors.append("ATR_STOP_LOSS_MULTIPLIER must be positive.")
        if self.enable_take_profit and self.atr_take_profit_multiplier <= 0:
            errors.append(
                "ATR_TAKE_PROFIT_MULTIPLIER must be positive if take profit is enabled."
            )
        if self.profit_momentum_window < 1:
            errors.append("PROFIT_MOMENTUM_WINDOW must be >= 1.")
        if self.whipsaw_max_trades_in_period < 1:
            errors.append("WHIPSAW_MAX_TRADES_IN_PERIOD must be >= 1.")
        if self.signal_persistence_candles < 1:
            errors.append("SIGNAL_PERSISTENCE_CANDLES must be >= 1.")
        if (
            self.st_max_entry_distance_atr_multiplier is not None
            and self.st_max_entry_distance_atr_multiplier < 0
        ):
            errors.append("ST_MAX_ENTRY_DISTANCE_ATR_MULTIPLIER cannot be negative.")
        if self.max_active_trade_parts < 1:
            errors.append("MAX_ACTIVE_TRADE_PARTS must be at least 1.")
        # Validation for Anti-Martingale parameters
        if self.enable_anti_martingale_risk:
            if not (0 < self.risk_reduction_factor_on_loss <= 1):
                errors.append(
                    "RISK_REDUCTION_FACTOR_ON_LOSS must be between 0 (exclusive) and 1 (inclusive)."
                )
            if self.risk_increase_factor_on_win < 1:
                errors.append("RISK_INCREASE_FACTOR_ON_WIN must be >= 1.")
            if not (0 < self.max_risk_pct_anti_martingale < 1):
                errors.append(
                    "MAX_RISK_PCT_ANTI_MARTINGALE must be between 0 and 1 (exclusive)."
                )
        if errors:
            error_message = (
                f"Configuration spellcrafting failed with {len(errors)} flaws:\n"
                + "\n".join([f"  - {e}" for e in errors])
            )
            _logger.critical(f"{NEON['CRITICAL']}{error_message}{NEON['RESET']}")
            raise ValueError(error_message)


# --- Logger Setup ---
LOGGING_LEVEL: int = (
    logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO
)
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
log_file_name = f"{logs_dir}/pyrmethus_spell_v{PYRMETHUS_VERSION.replace('.', '')}_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)-8s] %(name)-30s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file_name)],
)
logger: logging.Logger = logging.getLogger("PyrmethusCore")
SUCCESS_LEVEL: int = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)  # type: ignore[attr-defined]


logging.Logger.success = log_success  # type: ignore[attr-defined]

if sys.stdout.isatty():  # Apply colors only if output is a TTY
    stream_handler = next(
        (
            h
            for h in logging.getLogger().handlers
            if isinstance(h, logging.StreamHandler)
        ),
        None,
    )
    if stream_handler:  # Ensure there's a stream handler to format
        # The default formatter from basicConfig already includes levelname. We're effectively just re-setting it here.
        # This is fine if we want to ensure our specific format string.
        colored_formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)-8s] %(name)-30s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler.setFormatter(colored_formatter)

    # Colorize level names
    # This logic correctly handles potential re-coloring by stripping previous ANSI codes.
    for level, color_code in [
        (logging.DEBUG, NEON["DEBUG"]),
        (logging.INFO, NEON["INFO"]),
        (SUCCESS_LEVEL, NEON["SUCCESS"]),
        (logging.WARNING, NEON["WARNING"]),
        (logging.ERROR, NEON["ERROR"]),
        (logging.CRITICAL, NEON["CRITICAL"]),
    ]:
        level_name_str = logging.getLevelName(level)
        # Attempt to get the base level name if it's already colorized (e.g., from multiple initializations)
        if "\033" in level_name_str:
            # Simple approach: find the last 'm' and take text after it, or first part before '\033'
            parts = level_name_str.split("\033")
            base_name = parts[0]  # Usually the original name if prepended
            if not base_name.strip():  # If first part is empty or just spaces
                # Try to find a non-ANSI part
                non_ansi_part = next(
                    (
                        part.split("m", 1)[-1]
                        for part in parts
                        if "m" in part and part.split("m", 1)[-1].strip()
                    ),
                    None,
                )
                base_name = (
                    non_ansi_part if non_ansi_part else level_name_str
                )  # Fallback to original if complex
            level_name_str = base_name.strip()

        logging.addLevelName(
            level, f"{color_code}{level_name_str.ljust(8)}{NEON['RESET']}"
        )

# --- Global Objects & State Variables ---
try:
    CONFIG = Config()
except ValueError as config_error:  # Specific error from Config validation
    logging.getLogger("PyrmethusCore").critical(
        f"{NEON['CRITICAL']}Configuration loading failed. Error: {config_error}{NEON['RESET']}"
    )
    # Attempt notification even with partial/no config
    if "CONFIG" in globals() and CONFIG and hasattr(CONFIG, "send_notification_method"):
        CONFIG.send_notification_method(
            "Pyrmethus Startup Failure", f"Config Error: {str(config_error)[:200]}"
        )
    elif "termux-api" in os.getenv("PATH", ""):
        send_termux_notification(
            "Pyrmethus Startup Failure", f"Config Error: {str(config_error)[:200]}"
        )  # Basic Termux check
    sys.exit(1)
except (
    Exception
) as general_config_error:  # Catch any other exception during Config init
    logging.getLogger("PyrmethusCore").critical(
        f"{NEON['CRITICAL']}Unexpected critical error during configuration: {general_config_error}{NEON['RESET']}"
    )
    logging.getLogger("PyrmethusCore").debug(traceback.format_exc())
    if "CONFIG" in globals() and CONFIG and hasattr(CONFIG, "send_notification_method"):
        CONFIG.send_notification_method(
            "Pyrmethus Critical Failure",
            f"Unexpected Config Error: {str(general_config_error)[:200]}",
        )
    elif "termux-api" in os.getenv("PATH", ""):
        send_termux_notification(
            "Pyrmethus Critical Failure",
            f"Unexpected Config Error: {str(general_config_error)[:200]}",
        )
    sys.exit(1)

trade_timestamps_for_whipsaw = deque(maxlen=CONFIG.whipsaw_max_trades_in_period)
whipsaw_cooldown_active_until: float = 0.0
persistent_signal_counter = {"long": 0, "short": 0}  # For signal persistence feature
last_signal_type: Optional[str] = None  # Tracks last signal type for persistence logic
previous_day_high: Optional[Decimal] = None  # For no-trade zones
previous_day_low: Optional[Decimal] = None  # For no-trade zones
last_key_level_update_day: Optional[int] = (
    None  # Tracks when key levels were last updated
)
contradiction_cooldown_active_until: float = 0.0  # Cooldown for trend contradiction
consecutive_loss_cooldown_active_until: float = (
    0.0  # Cooldown after max consecutive losses
)


# --- Trading Strategy Classes ---
class TradingStrategy(ABC):
    def __init__(self, config: Config, df_columns: Optional[List[str]] = None):
        self.config = config
        self.logger = logging.getLogger(f"Strategy.{self.__class__.__name__}")
        self.required_columns = df_columns if df_columns else []
        self.logger.info(
            f"{NEON['STRATEGY']}Strategy Form '{self.__class__.__name__}' materializing...{NEON['RESET']}"
        )

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

        if self.required_columns:
            missing_cols = [
                col for col in self.required_columns if col not in df.columns
            ]
            if missing_cols:
                self.logger.warning(
                    f"{NEON['WARNING']}Market scroll missing required runes: {NEON['VALUE']}{missing_cols}{NEON['WARNING']}.{NEON['RESET']}"
                )
                return False

            # Check for NaNs in required columns of the latest row
            if not df.empty:  # Redundant due to len(df) < min_rows check, but safe
                last_row_values = df.iloc[-1][self.required_columns]
                if last_row_values.isnull().any():
                    nan_cols_last_row = last_row_values[
                        last_row_values.isnull()
                    ].index.tolist()
                    self.logger.debug(
                        f"Faint runes (NaNs) in last candle for critical columns: {nan_cols_last_row}."
                    )
                    # Depending on strategy, this might still be okay or might be an issue.
                    # For now, just a debug log. Strategy's signal generation should handle NaNs.
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
        )  # Example columns

    def generate_signals(
        self,
        df: pd.DataFrame,
        latest_close: Optional[Decimal] = None,
        latest_atr: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        signals = self._get_default_signals()
        # Ensure minimum rows for all lookbacks used by this strategy's indicators
        min_rows_needed = (
            max(
                self.config.st_atr_length,
                self.config.confirm_st_atr_length,
                self.config.momentum_period,
                self.config.confirm_st_stability_lookback,
                self.config.atr_calculation_period,  # If ATR is used directly for logic here (e.g. proximity)
            )
            + 10
        )  # Safety buffer

        if not self._validate_df(df, min_rows=min_rows_needed):
            return signals

        last = df.iloc[-1]
        # Primary Supertrend signals
        primary_long_flip = last.get("st_long_flip", False)
        primary_short_flip = last.get("st_short_flip", False)
        # Confirmation Supertrend trend
        confirm_is_up_current = last.get(
            "confirm_trend", pd.NA
        )  # True, False, or pd.NA
        # Momentum value
        momentum_val = safe_decimal_conversion(last.get("momentum"), pd.NA)

        # Sanity check: if both primary STs flip simultaneously, it's an anomaly. Try to resolve or ignore.
        if primary_long_flip and primary_short_flip:
            self.logger.warning(
                f"{NEON['WARNING']}Conflicting primary Supertrend flips on the same candle. Attempting to resolve...{NEON['RESET']}"
            )
            # Resolve based on confirmation trend and momentum, if possible
            if confirm_is_up_current is True and (
                momentum_val is not pd.NA and momentum_val > 0
            ):
                primary_short_flip = False  # Prioritize long
                self.logger.info(
                    "Resolution: Prioritizing LONG flip due to confirm trend and momentum."
                )
            elif confirm_is_up_current is False and (
                momentum_val is not pd.NA and momentum_val < 0
            ):
                primary_long_flip = False  # Prioritize short
                self.logger.info(
                    "Resolution: Prioritizing SHORT flip due to confirm trend and momentum."
                )
            else:  # Ambiguous, ignore both flips for this candle
                primary_long_flip = False
                primary_short_flip = False
                self.logger.warning(
                    "Resolution: Ambiguous. No primary signal generated from conflicting flips."
                )

        # Confirm ST Stability Check
        stable_confirm_trend = pd.NA
        if (
            self.config.confirm_st_stability_lookback <= 1
        ):  # No stability check or 1-candle stability
            stable_confirm_trend = confirm_is_up_current
        elif (
            "confirm_trend" in df.columns
            and len(df) >= self.config.confirm_st_stability_lookback
        ):
            recent_confirm_trends = df["confirm_trend"].iloc[
                -self.config.confirm_st_stability_lookback :
            ]
            if confirm_is_up_current is True and all(
                trend is True for trend in recent_confirm_trends
            ):
                stable_confirm_trend = True
            elif confirm_is_up_current is False and all(
                trend is False for trend in recent_confirm_trends
            ):
                stable_confirm_trend = False
            else:  # Trend was not stable
                stable_confirm_trend = pd.NA

        if pd.isna(stable_confirm_trend) or pd.isna(momentum_val):
            self.logger.debug(
                f"Signal check: Stable Confirm ST ({_format_for_log(stable_confirm_trend, True)}) or Momentum ({_format_for_log(momentum_val)}) is NA. No entry signal."
            )
            return signals  # Exit/Hold signals might still be generated below

        # Price Proximity Check (optional)
        price_proximity_ok = True
        if (
            self.config.st_max_entry_distance_atr_multiplier is not None
            and latest_atr is not None
            and latest_atr > 0
            and latest_close is not None
        ):
            max_allowed_distance = (
                latest_atr * self.config.st_max_entry_distance_atr_multiplier
            )
            st_p_base = f"ST_{self.config.st_atr_length}_{self.config.st_multiplier}"  # Primary ST base name

            if primary_long_flip:
                st_line_val = safe_decimal_conversion(
                    last.get(f"{st_p_base}l")
                )  # SuperTrend Long line
                if (
                    st_line_val is not None
                    and (latest_close - st_line_val) > max_allowed_distance
                ):
                    price_proximity_ok = False
                    self.logger.debug(
                        f"Long entry filtered: Price too far from ST line. Dist: {(latest_close - st_line_val):.2f}, MaxAllowed: {max_allowed_distance:.2f}"
                    )
            elif primary_short_flip:
                st_line_val = safe_decimal_conversion(
                    last.get(f"{st_p_base}s")
                )  # SuperTrend Short line
                if (
                    st_line_val is not None
                    and (st_line_val - latest_close) > max_allowed_distance
                ):
                    price_proximity_ok = False
                    self.logger.debug(
                        f"Short entry filtered: Price too far from ST line. Dist: {(st_line_val - latest_close):.2f}, MaxAllowed: {max_allowed_distance:.2f}"
                    )

        # Entry Signals
        if (
            primary_long_flip
            and stable_confirm_trend is True
            and momentum_val > self.config.momentum_threshold
            and momentum_val > 0
            and price_proximity_ok
        ):  # Ensure momentum is positive for long
            signals["enter_long"] = True
            self.logger.info(
                f"{NEON['SIDE_LONG']}DualST+Mom Signal: LONG Entry triggered.{NEON['RESET']}"
            )
        elif (
            primary_short_flip
            and stable_confirm_trend is False
            and momentum_val < -self.config.momentum_threshold
            and momentum_val < 0
            and price_proximity_ok
        ):  # Ensure momentum is negative for short
            signals["enter_short"] = True
            self.logger.info(
                f"{NEON['SIDE_SHORT']}DualST+Mom Signal: SHORT Entry triggered.{NEON['RESET']}"
            )

        # Exit Signals (based on primary Supertrend flip)
        if primary_short_flip:  # If primary ST flips to short, exit any long position
            signals["exit_long"] = True
            signals["exit_reason"] = "Primary SuperTrend Flipped Short"
        if primary_long_flip:  # If primary ST flips to long, exit any short position
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
            self.config.ehlers_fisher_length
            + self.config.ehlers_fisher_signal_length
            + 5
        )  # Buffer for calculation
        if (
            not self._validate_df(df, min_rows=min_rows_needed) or len(df) < 2
        ):  # Need at least 2 rows for prev/current
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
            self.logger.debug(
                f"Ehlers Fisher or Signal rune is NA. Fisher: {fisher_now}, Signal: {signal_now} (Prev Fisher: {fisher_prev}, Prev Signal: {signal_prev})"
            )
            return signals

        # Extreme Zone Check (Filter entries if Fisher is in an extreme zone)
        is_fisher_extreme = False
        if (
            fisher_now > self.config.ehlers_fisher_extreme_threshold_positive
            or fisher_now < self.config.ehlers_fisher_extreme_threshold_negative
        ):
            is_fisher_extreme = True

        # Entry Signals (Crossover logic)
        if not is_fisher_extreme:  # Only consider entries if NOT in an extreme zone
            if (
                fisher_prev <= signal_prev and fisher_now > signal_now
            ):  # Fisher crosses above Signal
                signals["enter_long"] = True
                self.logger.info(
                    f"{NEON['SIDE_LONG']}EhlersFisher Signal: LONG Entry (Fisher crossed UP).{NEON['RESET']}"
                )
            elif (
                fisher_prev >= signal_prev and fisher_now < signal_now
            ):  # Fisher crosses below Signal
                signals["enter_short"] = True
                self.logger.info(
                    f"{NEON['SIDE_SHORT']}EhlersFisher Signal: SHORT Entry (Fisher crossed DOWN).{NEON['RESET']}"
                )
        elif (fisher_prev <= signal_prev and fisher_now > signal_now) or (
            fisher_prev >= signal_prev and fisher_now < signal_now
        ):  # Log if crossover happened in extreme zone
            self.logger.info(
                f"EhlersFisher: Crossover IGNORED due to Fisher in extreme zone (Value: {fisher_now:.2f})."
            )

        # Exit Signals (Crossover logic, can happen regardless of extreme zone)
        if (
            fisher_prev >= signal_prev and fisher_now < signal_now
        ):  # Fisher crossed below Signal
            signals["exit_long"] = True
            signals["exit_reason"] = "Ehlers Fisher crossed BELOW Signal"
        elif (
            fisher_prev <= signal_prev and fisher_now > signal_now
        ):  # Fisher crossed above Signal
            signals["exit_short"] = True
            signals["exit_reason"] = "Ehlers Fisher crossed ABOVE Signal"

        return signals


strategy_map: Dict[StrategyName, Type[TradingStrategy]] = {
    StrategyName.DUAL_SUPERTREND_MOMENTUM: DualSupertrendMomentumStrategy,
    StrategyName.EHLERS_FISHER: EhlersFisherStrategy,
}
StrategyClass = strategy_map.get(CONFIG.strategy_name)
if StrategyClass:
    CONFIG.strategy_instance = StrategyClass(CONFIG)
    logger.success(
        f"{NEON['SUCCESS']}Strategy '{NEON['STRATEGY']}{CONFIG.strategy_name.value}{NEON['SUCCESS']}' invoked.{NEON['RESET']}"
    )  # type: ignore [attr-defined]
else:
    err_msg = (
        f"Failed to init strategy '{CONFIG.strategy_name.value}'. Unknown spell form."
    )
    logger.critical(f"{NEON['CRITICAL']}{err_msg}{NEON['RESET']}")
    if hasattr(CONFIG, "send_notification_method"):
        CONFIG.send_notification_method("Pyrmethus Critical Error", err_msg)
    sys.exit(1)


# --- Trade Metrics Tracking ---
class TradeMetrics:
    def __init__(self):
        self.trades: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("TradeMetrics")
        self.initial_equity: Optional[Decimal] = None
        self.daily_start_equity: Optional[Decimal] = None
        self.last_daily_reset_day: Optional[int] = (
            None  # Tracks day of month for daily reset
        )
        self.consecutive_losses: int = 0
        self.daily_trade_entry_count: int = (
            0  # Tracks number of new trade entries per day
        )
        self.last_daily_trade_count_reset_day: Optional[int] = None
        self.daily_trades_rest_active_until: float = (
            0.0  # Timestamp until daily trade limit rest is active
        )
        self.logger.info(f"{NEON['INFO']}TradeMetrics Ledger opened.{NEON['RESET']}")

    def set_initial_equity(self, equity: Decimal):
        if self.initial_equity is None:  # Set overall session initial equity once
            self.initial_equity = equity
            self.logger.info(
                f"{NEON['INFO']}Initial Session Equity: {NEON['VALUE']}{equity:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}"
            )

        today = datetime.now(pytz.utc).day
        # Set/reset daily start equity
        if self.last_daily_reset_day != today or self.daily_start_equity is None:
            self.daily_start_equity = equity
            self.last_daily_reset_day = today
            self.logger.info(
                f"{NEON['INFO']}Daily Equity Ward reset. Dawn Equity: {NEON['VALUE']}{equity:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}"
            )

        # Reset daily trade entry count if new day
        if self.last_daily_trade_count_reset_day != today:
            self.daily_trade_entry_count = 0
            self.last_daily_trade_count_reset_day = today
            self.logger.info(
                f"{NEON['INFO']}Daily trade entry count reset to 0.{NEON['RESET']}"
            )
            # If a new day starts, and a rest period was active, check if it should end
            if (
                self.daily_trades_rest_active_until > 0
                and time.time() > self.daily_trades_rest_active_until
            ):
                self.daily_trades_rest_active_until = (
                    0.0  # Reset rest period if it has passed
                )

    def check_drawdown(self, current_equity: Decimal) -> Tuple[bool, str]:
        if (
            not CONFIG.enable_max_drawdown_stop
            or self.daily_start_equity is None
            or self.daily_start_equity <= 0
        ):
            return (
                False,
                "",
            )  # Drawdown check disabled or daily start equity not set/invalid

        drawdown = self.daily_start_equity - current_equity
        drawdown_pct = (
            (drawdown / self.daily_start_equity)
            if self.daily_start_equity > 0
            else Decimal(0)
        )

        if drawdown_pct >= CONFIG.max_drawdown_percent:
            reason = f"Max daily drawdown breached ({NEON['PNL_NEG']}{drawdown_pct:.2%}{NEON['RESET']} >= {NEON['VALUE']}{CONFIG.max_drawdown_percent:.2%}{NEON['RESET']})"
            self.logger.warning(f"{NEON['WARNING']}{reason}.{NEON['RESET']}")
            CONFIG.send_notification_method(
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
        pnl_str: str,  # pnl_str for consistency with state
        scale_order_id: Optional[str] = None,
        mae: Optional[Decimal] = None,
        mfe: Optional[Decimal] = None,
        is_entry: bool = False,
    ):  # is_entry currently unused here
        # Validate crucial decimal inputs
        if not all(
            [
                isinstance(entry_price, Decimal) and entry_price > 0,
                isinstance(exit_price, Decimal) and exit_price > 0,
                isinstance(qty, Decimal) and qty > 0,
                isinstance(entry_time_ms, int)
                and entry_time_ms > 0,  # Timestamps should be positive
                isinstance(exit_time_ms, int) and exit_time_ms > 0,
            ]
        ):
            self.logger.warning(
                f"{NEON['WARNING']}Trade log skipped due to flawed parameters for Part ID: {part_id}. Entry: {entry_price}, Exit: {exit_price}, Qty: {qty}{NEON['RESET']}"
            )
            return

        profit = safe_decimal_conversion(
            pnl_str, Decimal(0)
        )  # Convert pnl_str to Decimal for logic
        if profit <= 0:  # Count non-profitable trades (loss or breakeven)
            self.consecutive_losses += 1
        else:  # Profitable trade resets counter
            self.consecutive_losses = 0

        entry_dt_utc = datetime.fromtimestamp(entry_time_ms / 1000, tz=pytz.utc)
        exit_dt_utc = datetime.fromtimestamp(exit_time_ms / 1000, tz=pytz.utc)
        duration_seconds = (exit_dt_utc - entry_dt_utc).total_seconds()
        trade_type = (
            "Scale-In" if scale_order_id else "Part"
        )  # Differentiate if it's a scale-in order part

        self.trades.append(
            {
                "symbol": symbol,
                "side": side,
                "entry_price_str": str(entry_price),
                "exit_price_str": str(exit_price),  # Store as strings
                "qty_str": str(qty),
                "profit_str": str(profit),  # Store PNL as string too
                "entry_time_iso": entry_dt_utc.isoformat(),
                "exit_time_iso": exit_dt_utc.isoformat(),
                "duration_seconds": duration_seconds,
                "exit_reason": reason,
                "type": trade_type,
                "part_id": part_id,
                "scale_order_id": scale_order_id,
                "mae_str": str(mae)
                if mae is not None
                else None,  # Max Adverse Excursion
                "mfe_str": str(mfe)
                if mfe is not None
                else None,  # Max Favorable Excursion
            }
        )
        pnl_color = (
            NEON["PNL_POS"]
            if profit > 0
            else (NEON["PNL_NEG"] if profit < 0 else NEON["PNL_ZERO"])
        )
        self.logger.success(
            f"{NEON['HEADING']}Trade Chronicle ({trade_type}:{part_id}): {side.upper()} {NEON['QTY']}{qty}{NEON['RESET']} {symbol.split('/')[0]} | P/L: {pnl_color}{profit:.2f} {CONFIG.usdt_symbol}{NEON['RESET']} | Reason: {reason}"
        )  # type: ignore [attr-defined]

    def increment_daily_trade_entry_count(self):
        # Ensure daily reset logic has run for today
        today = datetime.now(pytz.utc).day
        if self.last_daily_trade_count_reset_day != today:
            self.daily_trade_entry_count = 0
            self.last_daily_trade_count_reset_day = today
            self.logger.info(
                f"{NEON['INFO']}Daily trade entry count reset to 0 before incrementing.{NEON['RESET']}"
            )
            # Also reset rest period if it was active and day changed
            if (
                self.daily_trades_rest_active_until > 0
                and time.time() > self.daily_trades_rest_active_until
            ):
                self.daily_trades_rest_active_until = 0.0

        self.daily_trade_entry_count += 1
        self.logger.info(
            f"Daily entry trade count incremented to: {self.daily_trade_entry_count}"
        )

    def summary(self) -> str:
        # Pyrmethus: The ledger's summary remains a clear reflection.
        if not self.trades:
            return f"{NEON['INFO']}The Grand Ledger is empty.{NEON['RESET']}"

        total_trades = len(self.trades)
        profits = [
            safe_decimal_conversion(t["profit_str"], Decimal(0)) for t in self.trades
        ]  # Convert from string for calculations

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
            f"\n{NEON['HEADING']}--- Pyrmethus Trade Metrics Summary (v{PYRMETHUS_VERSION}) ---{NEON['RESET']}\n"
            f"Total Trade Parts Logged: {NEON['VALUE']}{total_trades}{NEON['RESET']}\n"
            f"  Wins: {NEON['PNL_POS']}{wins}{NEON['RESET']}, Losses: {NEON['PNL_NEG']}{losses}{NEON['RESET']}, Breakeven: {NEON['PNL_ZERO']}{breakeven}{NEON['RESET']}\n"
            f"Win Rate: {NEON['VALUE']}{win_rate:.2f}%{NEON['RESET']}\n"
            f"Total P/L: {(NEON['PNL_POS'] if total_profit > 0 else (NEON['PNL_NEG'] if total_profit < 0 else NEON['PNL_ZERO']))}{total_profit:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}\n"
            f"Avg P/L per Part: {(NEON['PNL_POS'] if avg_profit > 0 else (NEON['PNL_NEG'] if avg_profit < 0 else NEON['PNL_ZERO']))}{avg_profit:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}\n"
        )

        current_equity_approx = Decimal(0)  # Default to 0 if initial_equity is unknown
        if self.initial_equity is not None:
            current_equity_approx = self.initial_equity + total_profit
            overall_pnl_pct = (
                (total_profit / self.initial_equity) * 100
                if self.initial_equity > Decimal(0)
                else Decimal(0)
            )
            summary_str += f"Initial Session Treasury: {NEON['VALUE']}{self.initial_equity:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}\n"
            summary_str += f"Approx. Current Treasury: {NEON['VALUE']}{current_equity_approx:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}\n"
            summary_str += f"Overall Session P/L %: {(NEON['PNL_POS'] if overall_pnl_pct > 0 else (NEON['PNL_NEG'] if overall_pnl_pct < 0 else NEON['PNL_ZERO']))}{overall_pnl_pct:.2f}%{NEON['RESET']}\n"

        if self.daily_start_equity is not None:
            # Use current_equity_approx if available, otherwise calculate based on daily_start_equity and trades since daily start
            # This needs careful thought if trades span multiple days and initial_equity is from a much earlier session.
            # For simplicity, assume total_profit is relevant to daily_start_equity if initial_equity isn't set,
            # or use the more accurate current_equity_approx if available.

            # If initial_equity is set, current_equity_approx is the best estimate of current total balance.
            # If initial_equity is NOT set, but daily_start_equity is, then PNL is relative to daily_start_equity.
            # This can be complex if the bot restarts mid-day after some trades.
            # The current calculation assumes total_profit is the PNL since daily_start_equity if initial_equity is None.

            daily_pnl_base = (
                current_equity_approx
                if self.initial_equity is not None
                else (self.daily_start_equity + total_profit)
            )
            daily_pnl = daily_pnl_base - self.daily_start_equity
            daily_pnl_pct = (
                (daily_pnl / self.daily_start_equity) * 100
                if self.daily_start_equity > Decimal(0)
                else Decimal(0)
            )
            summary_str += f"Daily Start Treasury: {NEON['VALUE']}{self.daily_start_equity:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}\n"
            summary_str += f"Approx. Daily P/L: {(NEON['PNL_POS'] if daily_pnl > 0 else (NEON['PNL_NEG'] if daily_pnl < 0 else NEON['PNL_ZERO']))}{daily_pnl:.2f} {CONFIG.usdt_symbol} ({daily_pnl_pct:.2f}%){NEON['RESET']}\n"

        summary_str += f"Current Consecutive Losses: {NEON['VALUE']}{self.consecutive_losses}{NEON['RESET']}\n"
        summary_str += f"Daily Entries Made Today: {NEON['VALUE']}{self.daily_trade_entry_count}{NEON['RESET']}\n"
        summary_str += f"{NEON['HEADING']}--- End of Ledger Reading ---{NEON['RESET']}"

        self.logger.info(summary_str)  # Log the summary to the file/console
        return summary_str

    def get_serializable_trades(self) -> List[Dict[str, Any]]:
        return self.trades  # Already contains stringified Decimals

    def load_trades_from_list(self, trades_list: List[Dict[str, Any]]):
        self.trades = []  # Clear existing before loading
        for trade_data in trades_list:
            # Basic validation or transformation can be done here if needed
            self.trades.append(trade_data)
        self.logger.info(
            f"{NEON['INFO']}TradeMetrics: Re-inked {len(self.trades)} sagas from persistent scroll.{NEON['RESET']}"
        )


trade_metrics = TradeMetrics()
_active_trade_parts: List[Dict[str, Any]] = []
_last_heartbeat_save_time: float = 0.0


# --- State Persistence Functions ---
def save_persistent_state(force_heartbeat: bool = False) -> None:
    # Pyrmethus: The Phoenix Feather now scribes active SL/TP prices.
    global _active_trade_parts, trade_metrics, _last_heartbeat_save_time
    global \
        whipsaw_cooldown_active_until, \
        trade_timestamps_for_whipsaw, \
        persistent_signal_counter, \
        last_signal_type
    global previous_day_high, previous_day_low, last_key_level_update_day
    global contradiction_cooldown_active_until, consecutive_loss_cooldown_active_until

    now = time.time()
    # Save if forced, or if active trades and heartbeat interval passed, or no active trades and longer interval passed
    should_save = (
        force_heartbeat
        or (
            _active_trade_parts
            and (now - _last_heartbeat_save_time > HEARTBEAT_INTERVAL_SECONDS)
        )
        or (
            not _active_trade_parts
            and (now - _last_heartbeat_save_time > HEARTBEAT_INTERVAL_SECONDS * 5)
        )
    )  # Less frequent saves if idle

    if not should_save:
        return

    logger.debug(
        f"{NEON['COMMENT']}# Phoenix Feather scribing memories... (Force: {force_heartbeat}){NEON['RESET']}"
    )
    try:
        serializable_active_parts = []
        for part in _active_trade_parts:
            p_copy = part.copy()
            for key, value in p_copy.items():
                if isinstance(value, Decimal):
                    p_copy[key] = str(value)  # Convert Decimals to strings
                elif isinstance(value, deque):
                    p_copy[key] = list(value)  # Convert deques to lists
            serializable_active_parts.append(p_copy)

        state_data = {
            "pyrmethus_version": PYRMETHUS_VERSION,
            "timestamp_utc_iso": datetime.now(pytz.utc).isoformat(),
            "active_trade_parts": serializable_active_parts,  # Includes sl_price, tp_price as strings
            "trade_metrics_trades": trade_metrics.get_serializable_trades(),  # Already stringified
            "trade_metrics_consecutive_losses": trade_metrics.consecutive_losses,
            "trade_metrics_daily_trade_entry_count": trade_metrics.daily_trade_entry_count,
            "trade_metrics_last_daily_trade_count_reset_day": trade_metrics.last_daily_trade_count_reset_day,
            "trade_metrics_daily_trades_rest_active_until": trade_metrics.daily_trades_rest_active_until,
            "config_symbol": CONFIG.symbol,
            "config_strategy": CONFIG.strategy_name.value,
            "initial_equity_str": str(trade_metrics.initial_equity)
            if trade_metrics.initial_equity is not None
            else None,
            "daily_start_equity_str": str(trade_metrics.daily_start_equity)
            if trade_metrics.daily_start_equity is not None
            else None,
            "last_daily_reset_day": trade_metrics.last_daily_reset_day,
            "whipsaw_cooldown_active_until": whipsaw_cooldown_active_until,
            "trade_timestamps_for_whipsaw": list(
                trade_timestamps_for_whipsaw
            ),  # Convert deque
            "persistent_signal_counter": persistent_signal_counter,
            "last_signal_type": last_signal_type,
            "previous_day_high_str": str(previous_day_high)
            if previous_day_high is not None
            else None,
            "previous_day_low_str": str(previous_day_low)
            if previous_day_low is not None
            else None,
            "last_key_level_update_day": last_key_level_update_day,
            "contradiction_cooldown_active_until": contradiction_cooldown_active_until,
            "consecutive_loss_cooldown_active_until": consecutive_loss_cooldown_active_until,
        }

        temp_file_path = (
            STATE_FILE_PATH + ".tmp_scroll"
        )  # Atomic write: write to temp then rename
        with open(temp_file_path, "w") as f:
            json.dump(state_data, f, indent=4)
        os.replace(temp_file_path, STATE_FILE_PATH)  # Atomic rename

        _last_heartbeat_save_time = now
        log_level = (
            logging.INFO if force_heartbeat or _active_trade_parts else logging.DEBUG
        )
        logger.log(
            log_level,
            f"{NEON['SUCCESS']}Phoenix Feather: State memories scribed. Active parts: {len(_active_trade_parts)}.{NEON['RESET']}",
        )
    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}Phoenix Feather Error: Failed to scribe state: {e}{NEON['RESET']}"
        )
        logger.debug(traceback.format_exc())


def load_persistent_state() -> bool:
    # Pyrmethus: The Feather now reawakens SL/TP prices.
    global _active_trade_parts, trade_metrics, _last_heartbeat_save_time  # _last_heartbeat_save_time not loaded, reset on start
    global \
        whipsaw_cooldown_active_until, \
        trade_timestamps_for_whipsaw, \
        persistent_signal_counter, \
        last_signal_type
    global previous_day_high, previous_day_low, last_key_level_update_day
    global contradiction_cooldown_active_until, consecutive_loss_cooldown_active_until

    logger.info(
        f"{NEON['COMMENT']}# Phoenix Feather seeks past memories from '{STATE_FILE_PATH}'...{NEON['RESET']}"
    )
    if not os.path.exists(STATE_FILE_PATH):
        logger.info(
            f"{NEON['INFO']}Phoenix Feather: No previous scroll found. Starting with a clean slate.{NEON['RESET']}"
        )
        return False

    try:
        with open(STATE_FILE_PATH, "r") as f:
            state_data = json.load(f)

        saved_version = state_data.get("pyrmethus_version", "unknown")
        if saved_version != PYRMETHUS_VERSION:
            logger.warning(
                f"{NEON['WARNING']}Phoenix Feather: Scroll version '{saved_version}' differs from current Pyrmethus version '{PYRMETHUS_VERSION}'. Caution during reanimation.{NEON['RESET']}"
            )
            # Potentially add more sophisticated migration logic here if versions are incompatible.

        # Configuration compatibility check
        if (
            state_data.get("config_symbol") != CONFIG.symbol
            or state_data.get("config_strategy") != CONFIG.strategy_name.value
        ):
            logger.warning(
                f"{NEON['WARNING']}Phoenix Scroll sigils (symbol/strategy) mismatch! "
                f"Current: {CONFIG.symbol}/{CONFIG.strategy_name.value}, "
                f"Scroll: {state_data.get('config_symbol')}/{state_data.get('config_strategy')}. "
                f"Ignoring scroll to prevent anomalies.{NEON['RESET']}"
            )
            return False

        _active_trade_parts.clear()  # Ensure list is empty before loading
        loaded_parts_raw = state_data.get("active_trade_parts", [])

        # Define keys that should be Decimals or deques in a trade part
        decimal_keys_in_part = [
            "entry_price",
            "qty",
            "sl_price",
            "tp_price",  # Core pricing/qty
            "initial_usdt_value",
            "atr_at_entry",
            "entry_fisher_value",
            "entry_signal_value",  # Strategy-specific
            "last_known_pnl",  # For stateful checks
        ]
        deque_keys_in_part_config = {  # Maps key to its configured maxlen
            "recent_pnls": CONFIG.profit_momentum_window,
            "adverse_candle_closes": CONFIG.last_chance_consecutive_adverse_candles,
        }

        for part_data_str_values in loaded_parts_raw:
            restored_part: Dict[str, Any] = {}
            valid_part_data = True
            for key, loaded_value in part_data_str_values.items():
                if key in decimal_keys_in_part:
                    # Allow None for optional fields like tp_price if loaded_value is None or "None"
                    if loaded_value is None or (
                        isinstance(loaded_value, str) and loaded_value.lower() == "none"
                    ):
                        restored_part[key] = None
                    else:
                        restored_part[key] = safe_decimal_conversion(loaded_value, None)
                        if (
                            restored_part[key] is None
                        ):  # Conversion failed for a non-"None" value
                            logger.warning(
                                f"Failed to restore Decimal for key '{key}' in part {part_data_str_values.get('part_id', 'N/A')}, value was '{loaded_value}'. Setting to None."
                            )
                elif key in deque_keys_in_part_config:
                    maxlen = deque_keys_in_part_config[key]
                    if isinstance(loaded_value, list):
                        restored_part[key] = deque(loaded_value, maxlen=maxlen)
                    else:  # Corrupted or unexpected type
                        restored_part[key] = deque(
                            maxlen=maxlen
                        )  # Initialize empty deque
                        logger.warning(
                            f"Expected list for deque key '{key}' in part, got {type(loaded_value)}. Initialized empty deque."
                        )
                elif key == "entry_time_ms":  # Handle timestamp carefully
                    if isinstance(loaded_value, (int, float)):
                        restored_part[key] = int(loaded_value)
                    elif isinstance(
                        loaded_value, str
                    ):  # Could be ISO string or numeric string
                        try:  # Attempt ISO format first (more robust for timezone-aware timestamps)
                            dt_obj = datetime.fromisoformat(
                                loaded_value.replace("Z", "+00:00")
                            )
                            restored_part[key] = int(dt_obj.timestamp() * 1000)
                        except ValueError:  # Fallback to direct int conversion
                            try:
                                restored_part[key] = int(loaded_value)
                            except ValueError:
                                logger.error(
                                    f"Malformed entry_time_ms '{loaded_value}' for part. Skipping part."
                                )
                                valid_part_data = False
                                break
                    else:
                        logger.error(
                            f"Unexpected type for entry_time_ms '{type(loaded_value)}'. Skipping part."
                        )
                        valid_part_data = False
                        break
                else:  # Other keys are loaded as is
                    restored_part[key] = loaded_value

            if valid_part_data:
                _active_trade_parts.append(restored_part)

        trade_metrics.load_trades_from_list(state_data.get("trade_metrics_trades", []))
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

        initial_equity_str = state_data.get("initial_equity_str")
        if initial_equity_str and initial_equity_str.lower() != "none":
            trade_metrics.initial_equity = safe_decimal_conversion(
                initial_equity_str, None
            )

        daily_equity_str = state_data.get("daily_start_equity_str")
        if daily_equity_str and daily_equity_str.lower() != "none":
            trade_metrics.daily_start_equity = safe_decimal_conversion(
                daily_equity_str, None
            )

        trade_metrics.last_daily_reset_day = state_data.get("last_daily_reset_day")
        if trade_metrics.last_daily_reset_day is not None:  # Ensure it's an int
            try:
                trade_metrics.last_daily_reset_day = int(
                    trade_metrics.last_daily_reset_day
                )
            except ValueError:
                trade_metrics.last_daily_reset_day = None
                logger.warning(
                    "Malformed last_daily_reset_day from state, set to None."
                )

        whipsaw_cooldown_active_until = state_data.get(
            "whipsaw_cooldown_active_until", 0.0
        )
        loaded_whipsaw_ts = state_data.get("trade_timestamps_for_whipsaw", [])
        trade_timestamps_for_whipsaw = deque(
            loaded_whipsaw_ts, maxlen=CONFIG.whipsaw_max_trades_in_period
        )

        persistent_signal_counter = state_data.get(
            "persistent_signal_counter", {"long": 0, "short": 0}
        )
        last_signal_type = state_data.get("last_signal_type")

        prev_high_str = state_data.get("previous_day_high_str")
        previous_day_high = (
            safe_decimal_conversion(prev_high_str, None)
            if prev_high_str and prev_high_str.lower() != "none"
            else None
        )
        prev_low_str = state_data.get("previous_day_low_str")
        previous_day_low = (
            safe_decimal_conversion(prev_low_str, None)
            if prev_low_str and prev_low_str.lower() != "none"
            else None
        )
        last_key_level_update_day = state_data.get("last_key_level_update_day")

        contradiction_cooldown_active_until = state_data.get(
            "contradiction_cooldown_active_until", 0.0
        )
        consecutive_loss_cooldown_active_until = state_data.get(
            "consecutive_loss_cooldown_active_until", 0.0
        )

        logger.success(
            f"{NEON['SUCCESS']}Phoenix Feather: Memories reawakened! Active Parts: {len(_active_trade_parts)}, Logged Trades: {len(trade_metrics.trades)}{NEON['RESET']}"
        )  # type: ignore [attr-defined]
        return True
    except json.JSONDecodeError as e:
        logger.error(
            f"{NEON['ERROR']}Phoenix Feather: Scroll '{STATE_FILE_PATH}' corrupted: {e}. Starting with a clean slate.{NEON['RESET']}"
        )
        return False
    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}Phoenix Feather Error: Unexpected chaos during reawakening: {e}{NEON['RESET']}"
        )
        logger.debug(traceback.format_exc())
        return False


# --- Indicator Calculation ---
def calculate_all_indicators(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    logger.debug(
        f"{NEON['COMMENT']}# Transmuting market data with indicator alchemy...{NEON['RESET']}"
    )
    if df.empty:
        logger.warning(
            f"{NEON['WARNING']}Market data scroll empty. No alchemy possible.{NEON['RESET']}"
        )
        return df.copy()  # Return an empty copy

    # Ensure essential columns are numeric
    for col in ["close", "high", "low", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            logger.error(
                f"{NEON['ERROR']}Essential market rune '{col}' missing from data. Indicator alchemy may fail.{NEON['RESET']}"
            )
            # Depending on strictness, could return df here or raise error
            return df.copy()  # Or handle more gracefully

    # Calculate indicators based on selected strategy
    if config.strategy_name == StrategyName.DUAL_SUPERTREND_MOMENTUM:
        st_p_base = f"ST_{config.st_atr_length}_{config.st_multiplier}"  # Primary ST
        st_c_base = f"CONFIRM_ST_{config.confirm_st_atr_length}_{config.confirm_st_multiplier}"  # Confirmation ST

        # Calculate Supertrends
        df.ta.supertrend(
            length=config.st_atr_length,
            multiplier=float(config.st_multiplier),
            append=True,
            col_names=(st_p_base, f"{st_p_base}d", f"{st_p_base}l", f"{st_p_base}s"),
        )
        df.ta.supertrend(
            length=config.confirm_st_atr_length,
            multiplier=float(config.confirm_st_multiplier),
            append=True,
            col_names=(st_c_base, f"{st_c_base}d", f"{st_c_base}l", f"{st_c_base}s"),
        )

        # Process primary Supertrend signals
        if f"{st_p_base}d" in df.columns:  # Direction column
            df["st_trend_up"] = df[f"{st_p_base}d"] == 1  # True if uptrend
            df["st_long_flip"] = (df["st_trend_up"]) & (
                df["st_trend_up"].shift(1) == False
            )
            df["st_short_flip"] = (df["st_trend_up"] == False) & (
                df["st_trend_up"].shift(1) == True
            )
        else:  # If ST calculation failed or column missing
            df["st_trend_up"], df["st_long_flip"], df["st_short_flip"] = (
                pd.NA,
                False,
                False,
            )
            logger.warning(
                f"Primary Supertrend direction column '{st_p_base}d' not found post-calculation."
            )

        # Process confirmation Supertrend trend
        if f"{st_c_base}d" in df.columns:
            df["confirm_trend"] = df[f"{st_c_base}d"].apply(
                lambda x: True if x == 1 else (False if x == -1 else pd.NA)
            )
        else:
            df["confirm_trend"] = pd.NA
            logger.warning(
                f"Confirmation Supertrend direction column '{st_c_base}d' not found post-calculation."
            )

        # Calculate Momentum
        if "close" in df.columns and not df["close"].isnull().all():
            df.ta.mom(
                length=config.momentum_period, append=True, col_names=("momentum",)
            )
        else:
            df["momentum"] = pd.NA
            logger.warning(
                "Momentum calculation skipped due to missing or all-NaN 'close' column."
            )

    elif config.strategy_name == StrategyName.EHLERS_FISHER:
        if (
            "high" in df.columns
            and "low" in df.columns
            and not df["high"].isnull().all()
            and not df["low"].isnull().all()
        ):
            df.ta.fisher(
                length=config.ehlers_fisher_length,
                signal=config.ehlers_fisher_signal_length,
                append=True,
                col_names=("ehlers_fisher", "ehlers_signal"),
            )
        else:
            df["ehlers_fisher"], df["ehlers_signal"] = pd.NA, pd.NA
            logger.warning(
                "Ehlers Fisher calculation skipped due to missing or all-NaN 'high'/'low' columns."
            )

    # Calculate ATR (common for SL/TP, other filters)
    atr_col_name = f"ATR_{config.atr_calculation_period}"
    if (
        "high" in df.columns
        and "low" in df.columns
        and "close" in df.columns
        and not df["high"].isnull().all()
        and not df["low"].isnull().all()
        and not df["close"].isnull().all()
    ):
        df.ta.atr(
            length=config.atr_calculation_period, append=True, col_names=(atr_col_name,)
        )
    else:
        df[atr_col_name] = pd.NA
        logger.warning(
            f"ATR ({atr_col_name}) calculation skipped due to missing or all-NaN 'high'/'low'/'close' columns."
        )

    # Trap Filter related rolling highs/lows
    if config.enable_trap_filter:
        lookback = config.trap_filter_lookback_period
        if "high" in df.columns and not df["high"].isnull().all():
            df[f"rolling_high_{lookback}"] = (
                df["high"].rolling(window=lookback, min_periods=1).max()
            )
        else:
            df[f"rolling_high_{lookback}"] = pd.NA
        if "low" in df.columns and not df["low"].isnull().all():
            df[f"rolling_low_{lookback}"] = (
                df["low"].rolling(window=lookback, min_periods=1).min()
            )
        else:
            df[f"rolling_low_{lookback}"] = pd.NA

    return df


# --- Exchange Interaction Primitives ---
@retry(
    (
        ccxt.NetworkError,
        ccxt.RequestTimeout,
        ccxt.ExchangeNotAvailable,
        ccxt.DDoSProtection,
    ),
    tries=CONFIG.retry_count,
    delay=CONFIG.retry_delay_seconds,
    backoff=2,
    logger=logger,
)
def get_market_data(
    exchange: ccxt.Exchange, symbol: str, interval: str, limit: int = OHLCV_LIMIT
) -> Optional[pd.DataFrame]:
    # Pyrmethus: Market whispers remain a vital source.
    logger.debug(
        f"{NEON['COMMENT']}# Summoning {limit} market whispers for {symbol} ({interval})...{NEON['RESET']}"
    )
    try:
        params = (
            {"category": "linear"} if symbol.endswith(":USDT") else {}
        )  # Bybit specific for USDT perpetuals
        ohlcv = exchange.fetch_ohlcv(symbol, interval, limit=limit, params=params)
        if not ohlcv:
            logger.warning(
                f"{NEON['WARNING']}No OHLCV runes received for {symbol} ({interval}).{NEON['RESET']}"
            )
            return None

        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], unit="ms", utc=True
        )  # Ensure UTC for consistency
        for col in ["open", "high", "low", "close", "volume"]:  # Ensure numeric types
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Log first and last timestamp for verification
        if not df.empty:
            logger.debug(
                f"Summoned {len(df)} candles. First: {df['timestamp'].iloc[0]}, Last: {df['timestamp'].iloc[-1]}"
            )
        return df
    except ccxt.BadSymbol as e_bs:  # Symbol not recognized by exchange
        logger.critical(
            f"{NEON['CRITICAL']}Market realm rejects symbol '{symbol}': {e_bs}. Pyrmethus cannot proceed with this symbol.{NEON['RESET']}"
        )
        # This is critical enough that it might warrant stopping the bot or specific handling.
        # For now, returns None, main loop might retry or halt based on config.
        return None
    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}Error summoning market whispers for {symbol}: {e}{NEON['RESET']}"
        )
        logger.debug(traceback.format_exc())
        return None


@retry(
    (ccxt.NetworkError, ccxt.RequestTimeout),
    tries=CONFIG.retry_count,
    delay=CONFIG.retry_delay_seconds,
)
def fetch_account_balance(
    exchange: ccxt.Exchange, currency_code: str = CONFIG.usdt_symbol
) -> Optional[Decimal]:
    # Pyrmethus: Scrying the treasury's depth.
    logger.debug(
        f"{NEON['COMMENT']}# Scrying treasury for {currency_code}...{NEON['RESET']}"
    )
    try:
        params = {"accountType": "UNIFIED"}  # Bybit V5 Unified Trading Account
        if (
            currency_code == "USDT"
        ):  # Specify coin for Bybit unified to get specific coin balance
            params["coin"] = currency_code

        balance_data = exchange.fetch_balance(params=params)

        total_balance_val = None
        # Try common paths for balance in ccxt structure
        if currency_code in balance_data:  # e.g. balance_data['USDT']['total']
            total_balance_val = balance_data[currency_code].get("total")

        if total_balance_val is None:  # e.g. balance_data['total']['USDT']
            total_balance_val = balance_data.get("total", {}).get(currency_code)

        # Bybit V5 specific fallback for Unified account 'totalEquity' if 'total' for coin not found
        if (
            total_balance_val is None
            and exchange.id == "bybit"
            and "info" in balance_data
            and "result" in balance_data["info"]
            and "list" in balance_data["info"]["result"]
            and balance_data["info"]["result"]["list"]
        ):
            # The 'list' usually contains one item for UNIFIED account type
            account_info = balance_data["info"]["result"]["list"][0]
            if (
                "totalEquity" in account_info and currency_code == CONFIG.usdt_symbol
            ):  # Assuming totalEquity is in USDT
                total_balance_val = account_info["totalEquity"]
                logger.info(
                    f"Using 'totalEquity' ({total_balance_val}) from Bybit Unified Account for {currency_code} balance."
                )
            elif "coin" in account_info:  # If multiple coins are listed
                for coin_balance_info in account_info["coin"]:
                    if coin_balance_info.get("coin") == currency_code:
                        total_balance_val = coin_balance_info.get(
                            "walletBalance"
                        )  # or equity, check Bybit docs
                        logger.info(
                            f"Using '{coin_balance_info.get('walletBalance')}' from Bybit Unified Account coin list for {currency_code}."
                        )
                        break

        total_balance = safe_decimal_conversion(total_balance_val)
        if total_balance is None:
            logger.warning(
                f"{NEON['WARNING']}{currency_code} balance rune unreadable or not found in response.{NEON['RESET']}"
            )
            logger.debug(f"Full balance response for debug: {balance_data}")
            return None

        logger.info(
            f"{NEON['INFO']}Current {currency_code} Treasury: {NEON['VALUE']}{total_balance:.2f}{NEON['RESET']}"
        )
        return total_balance
    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}Error scrying treasury for {currency_code}: {e}{NEON['RESET']}"
        )
        logger.debug(traceback.format_exc())
        return None


def get_current_position_info() -> Tuple[str, Decimal]:
    # Pyrmethus: Divining the current stance from internal state.
    global _active_trade_parts
    if not _active_trade_parts:
        return CONFIG.pos_none, Decimal(0)

    # Assuming all active parts belong to one logical position (same side).
    # This is consistent with MAX_ACTIVE_TRADE_PARTS = 1 for v2.9.0 SL/TP logic.
    # If MAX_ACTIVE_TRADE_PARTS > 1 were for different positions, this logic would need change.
    # For scaling into one position, this is fine.

    # Use the first part to determine the side of the overall position.
    # All parts of a single logical position should have the same side.
    # If parts have different sides, it's a state error.

    determined_side = _active_trade_parts[0].get("side")
    if determined_side not in [CONFIG.pos_long, CONFIG.pos_short]:
        logger.error(
            f"{NEON['ERROR']}Incoherent side '{determined_side}' in first active part. State corrupted! Clearing parts.{NEON['RESET']}"
        )
        _active_trade_parts.clear()
        save_persistent_state(force_heartbeat=True)
        return CONFIG.pos_none, Decimal(0)

    total_qty = Decimal(0)
    valid_parts_exist = False
    for part in _active_trade_parts:
        part_qty = part.get("qty", Decimal(0))
        part_side = part.get("side")
        if (
            not isinstance(part_qty, Decimal) or part_qty < 0
        ):  # Qty should be positive Decimal
            logger.warning(
                f"Invalid qty '{part_qty}' for part {part.get('part_id')}. Ignoring this part for aggregation."
            )
            continue
        if part_side != determined_side:
            logger.error(
                f"{NEON['ERROR']}Position parts have conflicting sides ('{part_side}' vs '{determined_side}'). State corrupted! Clearing parts.{NEON['RESET']}"
            )
            _active_trade_parts.clear()
            save_persistent_state(force_heartbeat=True)
            return CONFIG.pos_none, Decimal(0)
        total_qty += part_qty
        valid_parts_exist = True

    if (
        not valid_parts_exist or total_qty <= CONFIG.position_qty_epsilon
    ):  # If all parts were invalid or total qty is negligible
        if total_qty < Decimal(0):  # Should not happen if part_qty is validated as >= 0
            logger.error(
                f"{NEON['ERROR']}Negative total quantity {total_qty} after aggregation. State corrupted! Clearing parts.{NEON['RESET']}"
            )
        # If total_qty is effectively zero, means no actual position.
        _active_trade_parts.clear()
        save_persistent_state(force_heartbeat=True)
        return CONFIG.pos_none, Decimal(0)

    return determined_side, total_qty


def calculate_position_size(
    balance: Decimal,
    risk_per_trade_pct: Decimal,
    entry_price: Decimal,
    sl_price: Decimal,
    market_info: Dict,
) -> Optional[Decimal]:
    # Pyrmethus: Alchemical proportions for the venture.
    logger.debug(
        f"{NEON['COMMENT']}# Calculating alchemical proportions... Bal: {balance:.2f}, Risk %: {risk_per_trade_pct:.4f}, Entry: {entry_price}, SL: {sl_price}{NEON['RESET']}"
    )

    if not all(
        [
            isinstance(balance, Decimal) and balance > 0,
            isinstance(entry_price, Decimal) and entry_price > 0,
            isinstance(sl_price, Decimal) and sl_price > 0,  # SL price must be positive
            sl_price != entry_price,  # SL must be different from entry
            market_info,  # market_info must exist
            CONFIG.qty_step is not None
            and CONFIG.qty_step > 0,  # Qty step must be valid
        ]
    ):
        logger.warning(
            f"{NEON['WARNING']}Invalid runes for sizing. Bal: {balance}, Entry: {entry_price}, SL: {sl_price}, MarketInfo: {'Present' if market_info else 'Absent'}, QtyStep: {CONFIG.qty_step}{NEON['RESET']}"
        )
        return None

    risk_per_unit = abs(entry_price - sl_price)
    if risk_per_unit <= Decimal(
        0
    ):  # Should be caught by sl_price != entry_price, but extra check
        logger.warning(
            f"{NEON['WARNING']}Risk per unit zero or negative (Entry: {entry_price}, SL: {sl_price}). Cannot divine size.{NEON['RESET']}"
        )
        return None

    usdt_at_risk = balance * risk_per_trade_pct
    logger.debug(f"USDT at risk for this trade: {usdt_at_risk:.2f}")

    # quantity_base = USDT at risk / (price_difference_per_unit_of_asset)
    quantity_base = usdt_at_risk / risk_per_unit
    position_usdt_value = quantity_base * entry_price

    # Cap position size by max_order_usdt_amount
    if position_usdt_value > CONFIG.max_order_usdt_amount:
        logger.info(
            f"Calculated position USDT value {position_usdt_value:.2f} exceeds MAX_ORDER_USDT_AMOUNT ({CONFIG.max_order_usdt_amount}). Capping."
        )
        position_usdt_value = CONFIG.max_order_usdt_amount
        quantity_base = (
            position_usdt_value / entry_price
        )  # Recalculate quantity based on capped USDT value

    # Adjust quantity to meet exchange's quantity step (lot size)
    # CONFIG.qty_step should be populated from market_info at startup
    quantity_adjusted = (quantity_base // CONFIG.qty_step) * CONFIG.qty_step

    if (
        quantity_adjusted <= CONFIG.position_qty_epsilon
    ):  # If adjusted quantity is too small
        logger.warning(
            f"{NEON['WARNING']}Calculated qty {quantity_adjusted} is zero/negligible after step adjustment. Original base: {quantity_base}. No order.{NEON['RESET']}"
        )
        return None

    # Check against minimum order quantity from market_info
    min_qty_limit_str = market_info.get("limits", {}).get("amount", {}).get("min")
    min_qty_limit = safe_decimal_conversion(min_qty_limit_str, None)

    if min_qty_limit is not None and quantity_adjusted < min_qty_limit:
        logger.warning(
            f"{NEON['WARNING']}Calculated qty {quantity_adjusted} is less than exchange minimum {min_qty_limit}. No order.{NEON['RESET']}"
        )
        return None

    qty_display_precision = (
        abs(CONFIG.qty_step.as_tuple().exponent) if CONFIG.qty_step else 8
    )  # Default precision for display
    logger.info(
        f"Calculated position size: {NEON['QTY']}{quantity_adjusted:.{qty_display_precision}f}{NEON['RESET']} (USDT Value: ~{quantity_adjusted * entry_price:.2f})"
    )
    return quantity_adjusted


@retry(
    (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.InsufficientFunds),
    tries=2,
    delay=3,
    logger=logger,
)  # Retry on InsufficientFunds once might help with temp margin issues
def place_risked_order(
    exchange: ccxt.Exchange,
    config: Config,
    side: str,
    entry_price_target: Decimal,
    sl_price: Decimal,
    tp_price: Optional[Decimal],
    atr_val: Optional[Decimal],
    df_for_entry_context: Optional[pd.DataFrame] = None,
) -> Optional[Dict]:
    # Pyrmethus: Weaving the entry spell, now with initial SL/TP wards.
    global _active_trade_parts, trade_timestamps_for_whipsaw, whipsaw_cooldown_active_until  # Access globals

    logger.info(
        f"{NEON['ACTION']}Attempting {side.upper()} order for {config.symbol}... Target Entry: {NEON['PRICE']}{entry_price_target}{NEON['RESET']}, "
        f"SL: {NEON['PRICE']}{sl_price}{NEON['RESET']}"
        f"{f', Initial TP: {NEON["PRICE"]}{tp_price}{NEON["RESET"]}' if tp_price and config.enable_take_profit else ', TP Disabled/Not Set'}"
    )

    # Determine effective risk percentage for this trade (e.g., anti-martingale)
    effective_risk_pct_for_trade = config.risk_per_trade_percentage
    if (
        config.enable_anti_martingale_risk and trade_metrics.trades
    ):  # Check if there are past trades
        # Ensure last trade has profit_str, convert to Decimal
        last_trade_pnl_str = trade_metrics.trades[-1].get("profit_str")
        if last_trade_pnl_str is not None:
            last_trade_pnl = safe_decimal_conversion(last_trade_pnl_str, Decimal(0))
            if last_trade_pnl < 0:  # Previous trade was a loss
                effective_risk_pct_for_trade *= config.risk_reduction_factor_on_loss
                logger.info(
                    f"{NEON['INFO']}Anti-Martingale: Risk reduced to {effective_risk_pct_for_trade * 100:.3f}% (due to previous loss).{NEON['RESET']}"
                )
            elif last_trade_pnl > 0:  # Previous trade was a win
                effective_risk_pct_for_trade *= config.risk_increase_factor_on_win
                effective_risk_pct_for_trade = min(
                    effective_risk_pct_for_trade, config.max_risk_pct_anti_martingale
                )  # Cap risk increase
                logger.info(
                    f"{NEON['INFO']}Anti-Martingale: Risk adjusted to {effective_risk_pct_for_trade * 100:.3f}% (due to previous win).{NEON['RESET']}"
                )

    balance = fetch_account_balance(exchange, config.usdt_symbol)
    if balance is None or balance <= Decimal(10):  # Arbitrary small threshold
        logger.error(
            f"{NEON['ERROR']}Insufficient treasury ({balance if balance is not None else 'Error fetching'}). Cannot cast order spell.{NEON['RESET']}"
        )
        return None

    if (
        config.MARKET_INFO is None
        or config.tick_size is None
        or config.qty_step is None
    ):
        logger.error(
            f"{NEON['ERROR']}Market runes (info, tick_size, qty_step) not divined. Cannot cast order spell.{NEON['RESET']}"
        )
        return None

    quantity = calculate_position_size(
        balance,
        effective_risk_pct_for_trade,
        entry_price_target,
        sl_price,
        config.MARKET_INFO,
    )
    if quantity is None or quantity <= CONFIG.position_qty_epsilon:
        logger.warning(
            f"{NEON['WARNING']}Position sizing yielded no valid quantity. Order spell aborted.{NEON['RESET']}"
        )
        return None

    order_side_verb = config.side_buy if side == config.pos_long else config.side_sell
    order_type_str = config.entry_order_type.value.lower()  # 'market' or 'limit'
    order_price_param_for_call: Optional[float] = None  # Price for limit order

    if order_type_str == "limit":
        offset_val = (
            (atr_val * config.limit_order_offset_atr_percentage)
            if atr_val and atr_val > 0
            else Decimal(0)
        )
        # For limit buy, place slightly below target; for limit sell, slightly above
        calculated_limit_price = (
            (entry_price_target - offset_val)
            if side == config.pos_long
            else (entry_price_target + offset_val)
        )

        # Optional: Check current order book for price improvement for limit orders
        if config.enable_limit_order_price_improvement_check:
            try:
                ticker = exchange.fetch_ticker(config.symbol)
                current_best_ask = safe_decimal_conversion(ticker.get("ask"), None)
                current_best_bid = safe_decimal_conversion(ticker.get("bid"), None)

                if (
                    side == config.pos_long
                    and current_best_ask is not None
                    and current_best_ask <= calculated_limit_price
                ):
                    logger.info(
                        f"{NEON['INFO']}Price Improve (Long): Ask ({current_best_ask}) is better than/at calculated limit ({calculated_limit_price}). Adjusting limit price slightly above ask.{NEON['RESET']}"
                    )
                    calculated_limit_price = (
                        current_best_ask + config.tick_size
                    )  # Adjust to be slightly less aggressive or match ask
                elif (
                    side == config.pos_short
                    and current_best_bid is not None
                    and current_best_bid >= calculated_limit_price
                ):
                    logger.info(
                        f"{NEON['INFO']}Price Improve (Short): Bid ({current_best_bid}) is better than/at calculated limit ({calculated_limit_price}). Adjusting limit price slightly below bid.{NEON['RESET']}"
                    )
                    calculated_limit_price = current_best_bid - config.tick_size
            except Exception as e_ticker:
                logger.warning(
                    f"{NEON['WARNING']}Ticker fetch failed for limit order price improvement: {e_ticker}{NEON['RESET']}"
                )

        # Ensure limit price conforms to tick size
        order_price_param_for_call = float(
            config.tick_size * (calculated_limit_price // config.tick_size)
        )
        logger.info(
            f"Calculated Limit Price for {side} order: {order_price_param_for_call}"
        )

    # Prepare order parameters for Bybit
    # positionIdx=0 for one-way mode. SL/TP set directly on order.
    params: Dict[str, Any] = {
        "timeInForce": "GTC"
        if order_type_str == "limit"
        else "IOC",  # GoodTillCancel for Limit, ImmediateOrCancel for Market
        "positionIdx": 0,
        "stopLoss": str(
            sl_price
        ),  # Bybit API often prefers string representation for precision
        "slTriggerBy": "MarkPrice",  # Or 'LastPrice', 'IndexPrice'
    }
    if tp_price and config.enable_take_profit:
        params["takeProfit"] = str(tp_price)
        params["tpTriggerBy"] = "MarkPrice"

    if config.symbol.endswith(":USDT"):  # Bybit specific category for USDT perpetuals
        params["category"] = "linear"

    try:
        logger.info(
            f"Weaving {order_type_str.upper()} {order_side_verb.upper()} order: Qty {NEON['QTY']}{quantity}{NEON['RESET']}, SL {NEON['PRICE']}{sl_price}{NEON['RESET']}"
            f"{f', TP {NEON["PRICE"]}{tp_price}{NEON["RESET"]}' if tp_price and config.enable_take_profit else ''}"
            f"{f', Price {order_price_param_for_call}' if order_price_param_for_call is not None else ''} "
            f"Params: {params}"
        )

        order = exchange.create_order(
            config.symbol,
            order_type_str,
            order_side_verb,
            float(quantity),
            order_price_param_for_call,
            params,
        )
        logger.success(
            f"{NEON['SUCCESS']}Entry Order {order.get('id', 'N/A')} ({order.get('status', 'N/A')}) cast.{NEON['RESET']}"
        )  # type: ignore [attr-defined]
        config.send_notification_method(
            f"Pyrmethus Order: {side.upper()}",
            f"{config.symbol} Qty: {quantity:.{abs(config.qty_step.as_tuple().exponent) if config.qty_step else 4}f} "
            f"@ {'Market' if order_type_str == 'market' else order_price_param_for_call}, "
            f"SL: {sl_price}{f', TP: {tp_price}' if tp_price and config.enable_take_profit else ''}",
        )

        # Wait briefly and then check order status, especially for limit orders
        time.sleep(config.order_fill_timeout_seconds / 3)  # Partial wait
        filled_order = exchange.fetch_order(order["id"], config.symbol)

        # Check if order is filled (or partially for limit, though scalping usually wants full fill or cancel)
        is_filled_market = (
            order_type_str == "market"
            and filled_order.get("filled", 0) > 0
            and filled_order.get("status") != "canceled"
        )
        is_filled_limit = (
            order_type_str == "limit" and filled_order.get("status") == "closed"
        )  # 'closed' implies fully filled for limit

        if is_filled_market or is_filled_limit:
            actual_entry_price = safe_decimal_conversion(
                filled_order.get(
                    "average", filled_order.get("price", entry_price_target)
                ),
                entry_price_target,
            )
            actual_qty = safe_decimal_conversion(
                filled_order.get("filled", quantity), quantity
            )
            entry_timestamp_ms = int(
                filled_order.get("timestamp", time.time() * 1000)
            )  # Use order timestamp, fallback to now
            part_id = str(uuid.uuid4())[:8]  # Unique ID for this trade part

            new_part = {
                "part_id": part_id,
                "entry_order_id": order["id"],
                "symbol": config.symbol,
                "side": side,
                "entry_price": actual_entry_price,
                "qty": actual_qty,
                "entry_time_ms": entry_timestamp_ms,
                "sl_price": sl_price,  # This is the SL set on the exchange via the order
                "tp_price": tp_price
                if config.enable_take_profit
                else None,  # TP set on exchange or None
                "atr_at_entry": atr_val,
                "initial_usdt_value": actual_qty * actual_entry_price,
                "breakeven_set": False,  # Flag for breakeven SL adjustment
                "recent_pnls": deque(
                    maxlen=config.profit_momentum_window
                ),  # For profit momentum SL
                "last_known_pnl": Decimal(0),  # For stateful PNL checks
                "adverse_candle_closes": deque(
                    maxlen=config.last_chance_consecutive_adverse_candles
                ),  # For last chance exit
                "partial_tp_taken_flags": {},  # For multi-level partial TP if implemented later
                "divergence_exit_taken": False,  # For Ehlers divergence scaled exit
            }
            # Add strategy-specific context if available (e.g., Ehlers Fisher values at entry)
            if (
                config.strategy_name == StrategyName.EHLERS_FISHER
                and df_for_entry_context is not None
                and not df_for_entry_context.empty
            ):
                entry_candle_data = df_for_entry_context.iloc[-1]
                new_part["entry_fisher_value"] = safe_decimal_conversion(
                    entry_candle_data.get("ehlers_fisher")
                )
                new_part["entry_signal_value"] = safe_decimal_conversion(
                    entry_candle_data.get("ehlers_signal")
                )

            _active_trade_parts.append(new_part)
            trade_metrics.increment_daily_trade_entry_count()  # Log this entry for daily limits

            # Whipsaw detection logic
            if config.enable_whipsaw_cooldown:
                now_ts = time.time()
                trade_timestamps_for_whipsaw.append(now_ts)
                if (
                    len(trade_timestamps_for_whipsaw)
                    == config.whipsaw_max_trades_in_period
                    and (now_ts - trade_timestamps_for_whipsaw[0])
                    <= config.whipsaw_period_seconds
                ):
                    logger.critical(
                        f"{NEON['CRITICAL']}Whipsaw pattern detected! Pausing trading for {config.whipsaw_cooldown_seconds}s.{NEON['RESET']}"
                    )
                    config.send_notification_method(
                        "Pyrmethus Whipsaw Alert",
                        f"Whipsaw detected. Pausing for {config.whipsaw_cooldown_seconds}s.",
                    )
                    whipsaw_cooldown_active_until = (
                        now_ts + config.whipsaw_cooldown_seconds
                    )
                    # Optionally, clear the timestamps deque after cooldown activation to reset detection
                    # trade_timestamps_for_whipsaw.clear() # Or let it naturally cycle

            save_persistent_state(
                force_heartbeat=True
            )  # Save state immediately after opening a trade
            logger.success(
                f"{NEON['SUCCESS']}Order {order['id']} filled! Part {part_id} created. Entry: {NEON['PRICE']}{actual_entry_price}{NEON['RESET']}, Qty: {NEON['QTY']}{actual_qty}{NEON['RESET']}"
            )  # type: ignore [attr-defined]
            config.send_notification_method(
                f"Pyrmethus Order Filled: {side.upper()}",
                f"{config.symbol} Part {part_id} @ {actual_entry_price:.{abs(config.tick_size.as_tuple().exponent) if config.tick_size else 2}f}",
            )
            return new_part
        else:  # Order not filled as expected (e.g., limit order too far, market order rejected/failed)
            logger.warning(
                f"{NEON['WARNING']}Order {order['id']} not fully 'closed' or filled as expected. Status: {filled_order.get('status', 'N/A')}, Filled Qty: {filled_order.get('filled', 0)}. Attempting to cancel if still open.{NEON['RESET']}"
            )
            if (
                filled_order.get("status") == "open"
                or filled_order.get("status") == "new"
            ):  # If order is still open
                try:
                    exchange.cancel_order(
                        order["id"],
                        config.symbol,
                        params={"category": "linear"}
                        if config.symbol.endswith(":USDT")
                        else {},
                    )
                    logger.info(
                        f"Successfully cancelled unfilled/partially-filled order {order['id']}."
                    )
                except Exception as e_cancel:
                    logger.error(f"Error cancelling order {order['id']}: {e_cancel}")
            return None

    except ccxt.InsufficientFunds as e:
        logger.error(
            f"{NEON['ERROR']}Insufficient funds to cast order spell: {e}{NEON['RESET']}"
        )
        return None
    except (
        ccxt.ExchangeError
    ) as e:  # Catch specific exchange errors (e.g., rate limits, invalid params)
        logger.error(
            f"{NEON['ERROR']}Exchange rejected order spell: {e}{NEON['RESET']}"
        )
        # Potentially parse e.message for specific Bybit error codes if needed
        return None
    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}Unexpected chaos while weaving order spell: {e}{NEON['RESET']}"
        )
        logger.debug(traceback.format_exc())
        return None


def close_position_part(
    exchange: ccxt.Exchange,
    config: Config,
    part_to_close: Dict,
    reason: str,
    close_price_target: Optional[Decimal] = None,
) -> bool:
    # Pyrmethus: Unraveling a specific part of the position.
    global _active_trade_parts  # To remove the part from the list upon successful closure

    part_id = part_to_close.get("part_id", "UnknownPartID")
    part_qty = part_to_close.get("qty")
    part_side = part_to_close.get("side")

    if (
        not isinstance(part_qty, Decimal)
        or part_qty <= CONFIG.position_qty_epsilon
        or part_side not in [config.pos_long, config.pos_short]
    ):
        logger.warning(
            f"{NEON['WARNING']}Cannot unravel part {part_id}: invalid qty '{part_qty}' or side '{part_side}'. Removing from active parts if it exists.{NEON['RESET']}"
        )
        _active_trade_parts = [
            p for p in _active_trade_parts if p.get("part_id") != part_id
        ]  # Clean up bad part
        save_persistent_state(force_heartbeat=True)
        return False  # Indicate failure to close as intended

    logger.info(
        f"{NEON['ACTION']}Unraveling part {part_id} (Side: {part_side}, Qty: {NEON['QTY']}{part_qty}{NEON['RESET']}) due to: {reason}{NEON['RESET']}"
    )

    close_side_verb = (
        config.side_sell if part_side == config.pos_long else config.side_buy
    )
    params: Dict[str, Any] = {
        "reduceOnly": True
    }  # Ensure this order only reduces position
    if config.symbol.endswith(":USDT"):
        params["category"] = "linear"

    try:
        logger.info(
            f"Casting MARKET {close_side_verb.upper()} order for part {part_id}: Qty {NEON['QTY']}{part_qty}{NEON['RESET']}"
        )
        # Note: When closing a position (or part of it) with a market order,
        # the exchange should automatically handle/cancel any associated SL/TP orders for that position quantity.
        # If this is a partial close, the SL/TP on the exchange for the *remaining* position should persist.

        close_order = exchange.create_order(
            config.symbol, "market", close_side_verb, float(part_qty), None, params
        )
        logger.success(
            f"{NEON['SUCCESS']}Unraveling Order {close_order.get('id', 'N/A')} ({close_order.get('status', 'N/A')}) cast.{NEON['RESET']}"
        )  # type: ignore [attr-defined]

        # Wait for the order to likely fill and fetch its details
        time.sleep(config.order_fill_timeout_seconds / 2)  # Adjust delay as needed
        filled_close_order = exchange.fetch_order(close_order["id"], config.symbol)

        if (
            filled_close_order.get("status") == "closed"
            and filled_close_order.get("filled", 0) > 0
        ):
            actual_exit_price = safe_decimal_conversion(
                filled_close_order.get("average", filled_close_order.get("price")),
                close_price_target or part_to_close.get("entry_price"),
            )  # Fallback for price
            exit_timestamp_ms = int(
                filled_close_order.get("timestamp", time.time() * 1000)
            )

            entry_price = part_to_close.get("entry_price")  # Should be a Decimal
            if not isinstance(
                entry_price, Decimal
            ):  # If loaded from old state or error
                entry_price = safe_decimal_conversion(
                    entry_price, actual_exit_price
                )  # Fallback to avoid error

            pnl_per_unit = (
                (actual_exit_price - entry_price)
                if part_side == config.pos_long
                else (entry_price - actual_exit_price)
            )
            pnl = pnl_per_unit * part_qty  # Total PNL for this part

            trade_metrics.log_trade(
                symbol=config.symbol,
                side=part_side,
                entry_price=entry_price,
                exit_price=actual_exit_price,
                qty=part_qty,
                entry_time_ms=part_to_close.get(
                    "entry_time_ms", 0
                ),  # Ensure entry_time_ms exists
                exit_time_ms=exit_timestamp_ms,
                reason=reason,
                part_id=part_id,
                pnl_str=str(pnl),
            )  # Log PNL as string

            _active_trade_parts = [
                p for p in _active_trade_parts if p.get("part_id") != part_id
            ]  # Remove closed part
            save_persistent_state(force_heartbeat=True)

            pnl_color_key = (
                "PNL_POS" if pnl > 0 else ("PNL_NEG" if pnl < 0 else "PNL_ZERO")
            )
            logger.success(
                f"{NEON['SUCCESS']}Part {part_id} successfully unraveled. Exit Price: {NEON['PRICE']}{actual_exit_price}{NEON['RESET']}, PNL: {NEON[pnl_color_key]}{pnl:.2f} {config.usdt_symbol}{NEON['RESET']}"
            )  # type: ignore [attr-defined]
            config.send_notification_method(
                f"Pyrmethus Position Closed: {part_side}",
                f"{config.symbol} Part {part_id}. PNL: {pnl:.2f}. Reason: {reason[:50]}",
            )
            return True
        else:
            logger.warning(
                f"{NEON['WARNING']}Unraveling order {close_order.get('id', 'N/A')} for part {part_id} not 'closed' or not filled. "
                f"Status: {filled_close_order.get('status', 'N/A')}, Filled: {filled_close_order.get('filled', 0)}. Manual check advised.{NEON['RESET']}"
            )
            # If order failed, part remains in _active_trade_parts for now. Position might still exist.
            return False

    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}Error during unraveling spell for part {part_id}: {e}{NEON['RESET']}"
        )
        logger.debug(traceback.format_exc())
        config.send_notification_method(
            "Pyrmethus Close Error",
            f"Error closing {config.symbol} part {part_id}: {str(e)[:80]}",
        )
        return False


@retry(
    (ccxt.NetworkError, ccxt.RequestTimeout),
    tries=CONFIG.retry_count,
    delay=CONFIG.retry_delay_seconds,
)
def cancel_all_symbol_orders(exchange: ccxt.Exchange, symbol: str):
    # Pyrmethus: Dispelling lingering entry enchantments (e.g., open limit orders).
    # This should NOT cancel position-based SL/TP orders if they are managed by position attributes.
    logger.info(
        f"{NEON['ACTION']}Dispelling all non-positional open orders for {symbol}...{NEON['RESET']}"
    )
    try:
        params = {"category": "linear"} if symbol.endswith(":USDT") else {}
        open_orders = exchange.fetch_open_orders(symbol, params=params)
        cancelled_count = 0
        if not open_orders:
            logger.info(f"No open orders found for {symbol} to dispel.")
            return

        for order in open_orders:
            order_id = order.get("id")
            order_type = order.get("type", "").lower()
            order_status = order.get("status", "").lower()

            # Heuristic: Cancel 'limit' orders that are 'open' or 'new'.
            # Avoid cancelling 'stop', 'takeProfit', 'stopLimit', 'takeProfitLimit' if they represent SL/TP.
            # Bybit's SL/TP via `set_trading_stop` or on order creation are attributes of the position,
            # not separate 'Stop' orders in `fetch_open_orders` unless they are conditional orders.
            # This cancel is mainly for leftover entry limit orders.
            if order_type == "limit" and (
                order_status == "open" or order_status == "new"
            ):
                try:
                    exchange.cancel_order(order_id, symbol, params=params)
                    logger.info(
                        f"Dispelled limit order {order_id} ({order_type}, Status: {order_status})."
                    )
                    cancelled_count += 1
                except Exception as e_cancel:
                    logger.error(f"Failed to dispel order {order_id}: {e_cancel}")
            else:
                logger.debug(
                    f"Skipping order {order_id} (Type: {order_type}, Status: {order_status}) from dispel action."
                )

        if cancelled_count > 0:
            logger.success(
                f"{NEON['SUCCESS']}{cancelled_count} open order(s) for {symbol} dispelled.{NEON['RESET']}"
            )  # type: ignore [attr-defined]
        else:
            logger.info(
                f"No suitable open orders (e.g., pending limit entries) found to dispel for {symbol}."
            )

    except ccxt.FeatureNotSupported:
        logger.warning(
            f"{NEON['WARNING']}Exchange {exchange.id} may not fully support `fetch_open_orders` or `cancel_order` as expected.{NEON['RESET']}"
        )
    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}Error dispelling open order enchantments: {e}{NEON['RESET']}"
        )
        logger.debug(traceback.format_exc())


def close_all_symbol_positions(exchange: ccxt.Exchange, config: Config, reason: str):
    # Pyrmethus: A full unraveling of all market presence for the symbol.
    global _active_trade_parts
    logger.warning(
        f"{NEON['WARNING']}Initiating forceful unraveling of ALL positions for {config.symbol} due to: {reason}{NEON['RESET']}"
    )

    # 1. Close internally tracked parts first using market orders
    # Create a copy for iteration as close_position_part modifies _active_trade_parts
    parts_to_close_copy = list(_active_trade_parts)
    if parts_to_close_copy:
        logger.info(
            f"Unraveling {len(parts_to_close_copy)} internally tracked part(s) via market orders..."
        )
        for part in parts_to_close_copy:
            if (
                part.get("symbol") == config.symbol
            ):  # Ensure it's for the current symbol
                close_position_part(
                    exchange, config, part, reason + " (Global Unraveling)"
                )
    else:
        logger.info(
            "No internally tracked trade parts to unravel via individual market orders."
        )

    # 2. Safeguard: Fetch current positions from exchange and close any residual
    # This handles cases where _active_trade_parts might be out of sync or if positions exist outside bot's tracking.
    logger.info(
        f"Performing safeguard scrying on exchange for any residual {config.symbol} positions..."
    )
    try:
        params_fetch_pos = (
            {"category": "linear"} if config.symbol.endswith(":USDT") else {}
        )
        # Fetch positions for the specific symbol. Some exchanges return all, so filter.
        all_positions = exchange.fetch_positions(
            None, params=params_fetch_pos
        )  # Fetch all, then filter
        symbol_positions = [
            p for p in all_positions if p.get("symbol") == config.symbol
        ]

        residual_closed_count = 0
        for pos_data in symbol_positions:
            pos_qty_str = pos_data.get(
                "contracts", pos_data.get("size", "0")
            )  # 'contracts' for derivatives
            pos_qty = safe_decimal_conversion(pos_qty_str, Decimal(0))
            pos_side_ccxt = pos_data.get(
                "side"
            )  # 'long' or 'short' (lowercase from ccxt)

            if (
                pos_qty is not None and pos_qty > CONFIG.position_qty_epsilon
            ):  # If a position exists
                side_to_close_market = (
                    config.side_sell if pos_side_ccxt == "long" else config.side_buy
                )
                logger.warning(
                    f"Found residual exchange position: {pos_side_ccxt.upper()} {pos_qty} {config.symbol}. Force market unraveling."
                )

                params_close_residual = {"reduceOnly": True}
                if config.symbol.endswith(":USDT"):
                    params_close_residual["category"] = "linear"

                exchange.create_order(
                    config.symbol,
                    "market",
                    side_to_close_market,
                    float(pos_qty),
                    params=params_close_residual,
                )
                residual_closed_count += 1
                time.sleep(
                    config.post_close_delay_seconds
                )  # Allow time for order processing

        if residual_closed_count > 0:
            logger.info(
                f"Safeguard unraveling completed for {residual_closed_count} residual position(s)."
            )
        else:
            logger.info(
                "No residual positions found on exchange for safeguard unraveling."
            )
    except Exception as e_fetch_close:
        logger.error(
            f"{NEON['ERROR']}Error during final scrying/unraveling of exchange positions: {e_fetch_close}{NEON['RESET']}"
        )
        logger.debug(traceback.format_exc())

    # 3. Cancel any remaining open orders for the symbol (e.g., pending limit entries)
    cancel_all_symbol_orders(exchange, config.symbol)

    # 4. Attempt to clear SL/TP for the position by setting them to "0" (or empty string for some exchanges)
    # This is a good practice as some exchanges might keep SL/TP settings even if position is zero.
    try:
        logger.info(
            f"Attempting to clear any remaining SL/TP settings for position {config.symbol} on exchange."
        )
        # Bybit V5: setting stopLoss/takeProfit to "0" cancels them.
        params_clear_sltp = {"positionIdx": 0, "stopLoss": "0", "takeProfit": "0"}
        if config.symbol.endswith(":USDT"):
            params_clear_sltp["category"] = "linear"

        exchange.set_trading_stop(config.symbol, params=params_clear_sltp)
        logger.info(
            f"SL/TP clearing command sent for {config.symbol}. Position should be fully flat with no pending SL/TP."
        )
    except Exception as e_clear_sl_tp:
        logger.warning(
            f"{NEON['WARNING']}Could not explicitly clear SL/TP on exchange for {config.symbol}: {e_clear_sl_tp}. "
            f"This might be okay if position closure already handled it, or if not supported directly.{NEON['RESET']}"
        )

    _active_trade_parts.clear()  # Clear internal tracking after all actions
    save_persistent_state(force_heartbeat=True)  # Save the cleared state
    logger.info(
        f"All positions/orders for {config.symbol} should now be unraveled/dispelled. Internal state cleared."
    )


def modify_position_sl_tp(
    exchange: ccxt.Exchange,
    config: Config,
    part: Dict,
    new_sl: Optional[Decimal] = None,
    new_tp: Optional[Decimal] = None,
) -> bool:
    # Pyrmethus: Adjusting the protective wards (SL/TP) on the exchange.
    if not config.MARKET_INFO or not config.tick_size:
        logger.error(
            f"{NEON['ERROR']}Market info or tick_size not divined. Cannot modify SL/TP wards.{NEON['RESET']}"
        )
        return False

    part_id = part.get("part_id", "UnknownPart")
    current_sl_from_part = part.get(
        "sl_price"
    )  # This is what the bot thinks is the current SL
    current_tp_from_part = part.get(
        "tp_price"
    )  # This is what the bot thinks is the current TP

    params_sl_tp: Dict[str, Any] = {"positionIdx": 0}  # For Bybit one-way mode
    if config.symbol.endswith(":USDT"):
        params_sl_tp["category"] = "linear"

    # Determine the SL price to set. If new_sl is None, we intend to keep the current SL (or remove if current is None).
    sl_to_set_on_exchange = new_sl if new_sl is not None else current_sl_from_part
    # Determine the TP price to set.
    tp_to_set_on_exchange = new_tp if new_tp is not None else current_tp_from_part

    # Apply tick size to SL price if it's being set/modified
    if sl_to_set_on_exchange is not None:
        sl_to_set_on_exchange = config.tick_size * (
            sl_to_set_on_exchange // config.tick_size
        )
        params_sl_tp["stopLoss"] = str(sl_to_set_on_exchange)
        params_sl_tp["slTriggerBy"] = (
            "MarkPrice"  # Or LastPrice, IndexPrice as per config/strategy
        )
    else:  # If sl_to_set_on_exchange resolved to None, it means cancel SL
        params_sl_tp["stopLoss"] = "0"

    # Apply tick size to TP price if it's being set/modified AND TP is enabled
    if config.enable_take_profit:
        if tp_to_set_on_exchange is not None:
            tp_to_set_on_exchange = config.tick_size * (
                tp_to_set_on_exchange // config.tick_size
            )
            params_sl_tp["takeProfit"] = str(tp_to_set_on_exchange)
            params_sl_tp["tpTriggerBy"] = "MarkPrice"
        else:  # If tp_to_set_on_exchange resolved to None, it means cancel TP
            params_sl_tp["takeProfit"] = "0"
    else:  # If TP is globally disabled, ensure it's cancelled on exchange
        params_sl_tp["takeProfit"] = "0"
        tp_to_set_on_exchange = None  # Reflect disabled TP internally too

    # Avoid API call if effectively nothing changes or only "0" for both
    if (
        params_sl_tp.get("stopLoss") == "0"
        and params_sl_tp.get("takeProfit") == "0"
        and (current_sl_from_part is None or current_sl_from_part == 0)
        and (
            current_tp_from_part is None
            or current_tp_from_part == 0
            or not config.enable_take_profit
        )
    ):
        logger.debug(
            f"No change in SL/TP needed for part {part_id}. Both target '0' and current state is effectively no SL/TP."
        )
        return True  # No change needed, considered successful

    logger.info(
        f"{NEON['ACTION']}Modifying SL/TP for part {part_id}: Target Exchange SL {params_sl_tp.get('stopLoss')}, Target Exchange TP {params_sl_tp.get('takeProfit')}{NEON['RESET']}"
    )
    try:
        exchange.set_trading_stop(config.symbol, params=params_sl_tp)
        # If successful, update the part's state to reflect what was set on the exchange
        logger.success(
            f"{NEON['SUCCESS']}SL/TP modification successful for part {part_id}. "
            f"Exchange SL set to: {params_sl_tp.get('stopLoss')}, TP set to: {params_sl_tp.get('takeProfit')}{NEON['RESET']}"
        )  # type: ignore [attr-defined]

        part["sl_price"] = (
            sl_to_set_on_exchange  # Update part with the (tick-adjusted) price or None
        )
        part["tp_price"] = (
            tp_to_set_on_exchange  # Update part with the (tick-adjusted) price or None (if TP disabled)
        )

        save_persistent_state(force_heartbeat=True)  # Save updated state
        return True
    except ccxt.ExchangeError as e:
        logger.error(
            f"{NEON['ERROR']}Exchange error modifying SL/TP for part {part_id}: {e}{NEON['RESET']}"
        )
        # Example: Bybit error if SL is too close to market price:
        # {"retCode":110049,"retMsg":"Stop loss price is not valid",...}
        if (
            "not valid" in str(e).lower() or "too close" in str(e).lower()
        ):  # Common phrases for invalid SL/TP
            logger.warning(
                f"SL/TP for part {part_id} rejected by exchange (likely too close to market or invalid value). Original SL/TP may remain active."
            )
        # The part's sl_price/tp_price are NOT updated here, so they still reflect the pre-attempt state.
    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}Unexpected error modifying SL/TP for part {part_id}: {e}{NEON['RESET']}"
        )
        logger.debug(traceback.format_exc())
    return False


# --- Helper: Update Daily Key Levels ---
def update_daily_key_levels(exchange: ccxt.Exchange, config: Config):
    global previous_day_high, previous_day_low, last_key_level_update_day

    now_utc = datetime.now(pytz.utc)
    today_utc_day = now_utc.day  # Day of the month

    if last_key_level_update_day == today_utc_day:  # Already updated today
        return

    logger.info(
        f"{NEON['INFO']}Attempting to update daily key levels (Previous Day High/Low)...{NEON['RESET']}"
    )
    try:
        # Fetch 1-day candles. We need yesterday's candle.
        # Fetching last 2-3 daily candles should be enough.
        # `since` should be a few days ago to ensure we get full candles.
        three_days_ago_utc_start = now_utc.replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=3)
        since_timestamp_ms = int(three_days_ago_utc_start.timestamp() * 1000)

        params = {"category": "linear"} if config.symbol.endswith(":USDT") else {}
        daily_candles = exchange.fetch_ohlcv(
            config.symbol, "1d", since=since_timestamp_ms, limit=3, params=params
        )  # Fetch last 3 daily candles

        if not daily_candles or len(daily_candles) < 1:
            logger.warning(
                f"{NEON['WARNING']}Could not fetch sufficient daily candle data to update key levels.{NEON['RESET']}"
            )
            return

        # Find yesterday's candle data
        yesterday_target_date = (now_utc - timedelta(days=1)).date()
        found_yesterday_candle = False
        for candle_data in reversed(daily_candles):  # Iterate from most recent
            candle_timestamp_ms = candle_data[0]
            candle_dt_utc = datetime.fromtimestamp(
                candle_timestamp_ms / 1000, tz=pytz.utc
            )

            if candle_dt_utc.date() == yesterday_target_date:
                previous_day_high = Decimal(str(candle_data[2]))  # High price
                previous_day_low = Decimal(str(candle_data[3]))  # Low price
                last_key_level_update_day = today_utc_day  # Mark as updated for today
                found_yesterday_candle = True
                logger.info(
                    f"{NEON['INFO']}Updated key levels for {yesterday_target_date}: Prev Day High {NEON['PRICE']}{previous_day_high}{NEON['RESET']}, Prev Day Low {NEON['PRICE']}{previous_day_low}{NEON['RESET']}"
                )
                break

        if not found_yesterday_candle:
            logger.warning(
                f"{NEON['WARNING']}Could not definitively find YESTERDAY's ({yesterday_target_date}) candle data among fetched daily candles to update key levels.{NEON['RESET']}"
            )
            logger.debug(
                f"Fetched daily candles timestamps: {[datetime.fromtimestamp(c[0] / 1000, tz=pytz.utc).date() for c in daily_candles]}"
            )

    except Exception as e_kl:
        logger.error(
            f"{NEON['ERROR']}Error updating daily key levels: {e_kl}{NEON['RESET']}"
        )
        logger.debug(traceback.format_exc())


# --- Main Spell Weaving ---
def main_loop(exchange: ccxt.Exchange, config: Config) -> None:
    # Pyrmethus: The heart of the spell, now with more intricate wardings.
    global \
        _active_trade_parts, \
        trade_timestamps_for_whipsaw, \
        whipsaw_cooldown_active_until
    global persistent_signal_counter, last_signal_type
    global previous_day_high, previous_day_low, last_key_level_update_day  # Managed by update_daily_key_levels
    global contradiction_cooldown_active_until, consecutive_loss_cooldown_active_until

    logger.info(
        f"{NEON['HEADING']}=== Pyrmethus Spell v{PYRMETHUS_VERSION} Awakening on {exchange.id} for {config.symbol} ==={NEON['RESET']}"
    )

    if load_persistent_state():
        logger.success(
            f"{NEON['SUCCESS']}Reawakened from Phoenix scroll. Active parts: {len(_active_trade_parts)}.{NEON['RESET']}"
        )  # type: ignore [attr-defined]
    else:
        logger.info(
            f"{NEON['INFO']}No prior state found or state was incompatible. Starting fresh spellcraft.{NEON['RESET']}"
        )

    # Initialize/update equity and daily metrics at start of main loop
    current_balance_init = fetch_account_balance(exchange, config.usdt_symbol)
    if current_balance_init is not None:
        trade_metrics.set_initial_equity(
            current_balance_init
        )  # Sets session and daily start equity
    else:
        logger.warning(
            f"{NEON['WARNING']}Could not set initial treasury at startup. Drawdown checks might be affected until next successful balance fetch.{NEON['RESET']}"
        )

    loop_counter = 0
    try:  # Outer try for the main while loop, associated with the finally block for cleanup
        while True:
            loop_counter += 1
            try:  # Inner try for a single iteration of the trading-decision loop
                logger.debug(
                    f"{NEON['COMMENT']}# New cycle of observation ({loop_counter}). Current time: {datetime.now(pytz.utc).isoformat()}{NEON['RESET']}"
                )

                # --- Cooldowns & System Checks ---
                now = time.time()  # Consistent timestamp for checks in this iteration
                if (
                    config.enable_whipsaw_cooldown
                    and now < whipsaw_cooldown_active_until
                ):
                    remaining_cooldown = whipsaw_cooldown_active_until - now
                    logger.warning(
                        f"{NEON['WARNING']}In whipsaw cooldown. Ends in {remaining_cooldown:.0f}s.{NEON['RESET']}"
                    )
                    time.sleep(min(remaining_cooldown, config.sleep_seconds))
                    continue

                if (
                    config.enable_trend_contradiction_cooldown
                    and now < contradiction_cooldown_active_until
                ):
                    remaining_cooldown = contradiction_cooldown_active_until - now
                    logger.warning(
                        f"{NEON['WARNING']}In Trend Contradiction cooldown. Ends in {remaining_cooldown:.0f}s.{NEON['RESET']}"
                    )
                    time.sleep(min(remaining_cooldown, config.sleep_seconds))
                    continue

                if (
                    config.enable_daily_max_trades_rest
                    and now < trade_metrics.daily_trades_rest_active_until
                ):
                    remaining_rest = trade_metrics.daily_trades_rest_active_until - now
                    logger.warning(
                        f"{NEON['WARNING']}Resting (daily max trades limit reached). Resumes in {remaining_rest / 3600:.1f} hrs ({remaining_rest:.0f}s).{NEON['RESET']}"
                    )
                    time.sleep(min(remaining_rest, config.sleep_seconds * 10))
                    continue  # Longer sleep during rest

                if (
                    config.enable_consecutive_loss_limiter
                    and now < consecutive_loss_cooldown_active_until
                ):
                    remaining_cooldown = consecutive_loss_cooldown_active_until - now
                    logger.warning(
                        f"{NEON['WARNING']}In consecutive loss cooldown. Ends in {remaining_cooldown / 60:.1f} mins ({remaining_cooldown:.0f}s).{NEON['RESET']}"
                    )
                    time.sleep(min(remaining_cooldown, config.sleep_seconds))
                    continue

                # Fetch current balance and perform checks
                current_balance_iter = fetch_account_balance(
                    exchange, config.usdt_symbol
                )
                if current_balance_iter is not None:
                    trade_metrics.set_initial_equity(
                        current_balance_iter
                    )  # Refreshes daily equity if new day
                    drawdown_hit, dd_reason = trade_metrics.check_drawdown(
                        current_balance_iter
                    )
                    if drawdown_hit:
                        logger.critical(
                            f"{NEON['CRITICAL']}Max daily drawdown! {dd_reason}. Pyrmethus must rest to preserve essence.{NEON['RESET']}"
                        )
                        close_all_symbol_positions(
                            exchange, config, "Max Daily Drawdown Reached"
                        )
                        break  # Exit main loop
                else:
                    logger.warning(
                        f"{NEON['WARNING']}Failed treasury scry this cycle. Drawdown check might be on stale data.{NEON['RESET']}"
                    )

                # Session P&L Limits Check
                if (
                    config.enable_session_pnl_limits
                    and trade_metrics.initial_equity is not None
                    and current_balance_iter is not None
                ):
                    current_session_pnl = (
                        current_balance_iter - trade_metrics.initial_equity
                    )
                    if (
                        config.session_profit_target_usdt is not None
                        and current_session_pnl >= config.session_profit_target_usdt
                    ):
                        logger.critical(
                            f"{NEON['SUCCESS']}SESSION PROFIT TARGET ({config.session_profit_target_usdt} {config.usdt_symbol}) REACHED! Current Session PNL: {current_session_pnl:.2f}. Pyrmethus concludes this session.{NEON['RESET']}"
                        )
                        config.send_notification_method(
                            "Pyrmethus Session Goal!",
                            f"Profit target {config.session_profit_target_usdt} hit. PNL: {current_session_pnl:.2f}",
                        )
                        close_all_symbol_positions(
                            exchange, config, "Session Profit Target Reached"
                        )
                        break
                    if (
                        config.session_max_loss_usdt is not None
                        and current_session_pnl <= -abs(config.session_max_loss_usdt)
                    ):  # Max loss is positive in config
                        logger.critical(
                            f"{NEON['CRITICAL']}SESSION MAX LOSS ({config.session_max_loss_usdt} {config.usdt_symbol}) BREACHED! Current Session PNL: {current_session_pnl:.2f}. Pyrmethus concludes this session.{NEON['RESET']}"
                        )
                        config.send_notification_method(
                            "Pyrmethus Session Loss!",
                            f"Max session loss {config.session_max_loss_usdt} hit. PNL: {current_session_pnl:.2f}",
                        )
                        close_all_symbol_positions(
                            exchange, config, "Session Max Loss Reached"
                        )
                        break

                # Daily Max Trades Rest Check (activate if limit hit)
                if (
                    config.enable_daily_max_trades_rest
                    and trade_metrics.daily_trade_entry_count
                    >= config.daily_max_trades_limit
                    and now >= trade_metrics.daily_trades_rest_active_until
                ):  # Ensure not already resting
                    logger.critical(
                        f"{NEON['CRITICAL']}Daily max trades ({config.daily_max_trades_limit}) reached. Initiating rest for {config.daily_max_trades_rest_hours} hrs.{NEON['RESET']}"
                    )
                    config.send_notification_method(
                        "Pyrmethus Daily Trades Rest",
                        f"Max {config.daily_max_trades_limit} trades. Resting for {config.daily_max_trades_rest_hours}h.",
                    )
                    trade_metrics.daily_trades_rest_active_until = (
                        now + config.daily_max_trades_rest_hours * 3600
                    )
                    close_all_symbol_positions(
                        exchange,
                        config,
                        "Daily Max Trades Limit Reached - Initiating Rest",
                    )
                    continue  # Skip to next iteration for rest sleep

                # Consecutive Loss Limiter Check (activate if limit hit)
                if (
                    config.enable_consecutive_loss_limiter
                    and trade_metrics.consecutive_losses
                    >= config.max_consecutive_losses
                    and now >= consecutive_loss_cooldown_active_until
                ):  # Ensure not already in cooldown
                    logger.critical(
                        f"{NEON['CRITICAL']}{config.max_consecutive_losses} consecutive losses recorded! Initiating cooldown for {config.consecutive_loss_cooldown_minutes} mins.{NEON['RESET']}"
                    )
                    config.send_notification_method(
                        "Pyrmethus Consecutive Loss Cooldown",
                        f"{config.max_consecutive_losses} losses. Pausing for {config.consecutive_loss_cooldown_minutes}m.",
                    )
                    consecutive_loss_cooldown_active_until = (
                        now + config.consecutive_loss_cooldown_minutes * 60
                    )
                    trade_metrics.consecutive_losses = (
                        0  # Reset counter after activating cooldown
                    )
                    close_all_symbol_positions(
                        exchange,
                        config,
                        "Consecutive Loss Limit Reached - Initiating Cooldown",
                    )
                    continue

                # --- Market Analysis ---
                ohlcv_df = get_market_data(
                    exchange,
                    config.symbol,
                    config.interval,
                    limit=OHLCV_LIMIT + config.api_fetch_limit_buffer,
                )
                if ohlcv_df is None or ohlcv_df.empty:
                    logger.warning(
                        f"{NEON['WARNING']}Market whispers faint (no OHLCV data). Retrying after pause.{NEON['RESET']}"
                    )
                    time.sleep(config.sleep_seconds)
                    continue

                df_with_indicators = calculate_all_indicators(
                    ohlcv_df.copy(), config
                )  # Use a copy
                if (
                    df_with_indicators.empty
                    or df_with_indicators.iloc[-1].isnull().all()
                ):  # Check if last row is all NaNs
                    logger.warning(
                        f"{NEON['WARNING']}Indicator alchemy faint (empty or all-NaN last row). Pausing.{NEON['RESET']}"
                    )
                    time.sleep(config.sleep_seconds)
                    continue

                latest_close = safe_decimal_conversion(
                    df_with_indicators["close"].iloc[-1]
                )
                atr_col = f"ATR_{config.atr_calculation_period}"
                latest_atr = safe_decimal_conversion(
                    df_with_indicators[atr_col].iloc[-1]
                    if atr_col in df_with_indicators.columns
                    else pd.NA
                )

                if (
                    pd.isna(latest_close)
                    or pd.isna(latest_atr)
                    or latest_atr <= Decimal(0)
                ):
                    logger.warning(
                        f"{NEON['WARNING']}Essential market readings unclear: Close ({_format_for_log(latest_close)}) "
                        f"or ATR ({_format_for_log(latest_atr)}) is NA/invalid. No decisions possible this cycle.{NEON['RESET']}"
                    )
                    time.sleep(config.sleep_seconds)
                    continue

                # Generate trading signals from strategy
                signals = config.strategy_instance.generate_signals(
                    df_with_indicators, latest_close, latest_atr
                )

                # Signal Persistence Logic
                confirmed_enter_long, confirmed_enter_short = False, False
                if (
                    config.signal_persistence_candles <= 1
                ):  # No persistence or 1-candle persistence
                    confirmed_enter_long = signals.get("enter_long", False)
                    confirmed_enter_short = signals.get("enter_short", False)
                else:  # Stateful signal confirmation
                    current_long_sig = signals.get("enter_long", False)
                    current_short_sig = signals.get("enter_short", False)
                    if current_long_sig:  # Current signal is long
                        if (
                            last_signal_type == "long" or last_signal_type is None
                        ):  # Previous was long or no signal
                            persistent_signal_counter["long"] += 1
                        else:  # Previous was short, reset long counter
                            persistent_signal_counter["long"] = 1
                        persistent_signal_counter["short"] = 0  # Reset short counter
                        last_signal_type = "long"
                        if (
                            persistent_signal_counter["long"]
                            >= config.signal_persistence_candles
                        ):
                            confirmed_enter_long = True
                    elif current_short_sig:  # Current signal is short
                        if (
                            last_signal_type == "short" or last_signal_type is None
                        ):  # Previous was short or no signal
                            persistent_signal_counter["short"] += 1
                        else:  # Previous was long, reset short counter
                            persistent_signal_counter["short"] = 1
                        persistent_signal_counter["long"] = 0  # Reset long counter
                        last_signal_type = "short"
                        if (
                            persistent_signal_counter["short"]
                            >= config.signal_persistence_candles
                        ):
                            confirmed_enter_short = True
                    else:  # No signal this candle, reset both counters and last signal type
                        persistent_signal_counter["long"] = 0
                        persistent_signal_counter["short"] = 0
                        last_signal_type = None  # Or keep last_signal_type to detect change from signal to no-signal
                    logger.debug(
                        f"Signal persistence: Longs seen: {persistent_signal_counter['long']}, Shorts seen: {persistent_signal_counter['short']}. "
                        f"Confirmed L/S: {confirmed_enter_long}/{confirmed_enter_short}"
                    )

                current_pos_side, current_pos_qty = get_current_position_info()

                # --- Entry Logic ---
                if (
                    current_pos_side == config.pos_none
                    and len(_active_trade_parts) < config.max_active_trade_parts
                ):
                    if confirmed_enter_long or confirmed_enter_short:
                        # Update daily key levels before entry decision if needed for filters
                        update_daily_key_levels(exchange, config)

                        # No-Trade Zone Check
                        trade_allowed_by_zone = True
                        if config.enable_no_trade_zones and latest_close is not None:
                            key_levels_to_check: List[Decimal] = []
                            if previous_day_high:
                                key_levels_to_check.append(previous_day_high)
                            if previous_day_low:
                                key_levels_to_check.append(previous_day_low)
                            if (
                                config.key_round_number_step
                                and config.key_round_number_step > 0
                            ):
                                lower_round = (
                                    latest_close // config.key_round_number_step
                                ) * config.key_round_number_step
                                key_levels_to_check.extend(
                                    [
                                        lower_round,
                                        lower_round + config.key_round_number_step,
                                    ]
                                )

                            for level in key_levels_to_check:
                                zone_half_width = (
                                    level * config.no_trade_zone_pct_around_key_level
                                )
                                if (
                                    (level - zone_half_width)
                                    <= latest_close
                                    <= (level + zone_half_width)
                                ):
                                    trade_allowed_by_zone = False
                                    logger.info(
                                        f"{NEON['INFO']}Entry suppressed: Price {latest_close} is within no-trade zone around key level {level} "
                                        f"({level - zone_half_width} - {level + zone_half_width}).{NEON['RESET']}"
                                    )
                                    break

                        # Trap Filter Check
                        trap_detected = False
                        if (
                            config.enable_trap_filter
                            and latest_atr > 0
                            and len(df_with_indicators)
                            > config.trap_filter_lookback_period
                        ):
                            # Check against -2 index (completed previous candle relative to current forming candle at -1)
                            prev_candle_idx = -2
                            if len(df_with_indicators) > abs(
                                prev_candle_idx
                            ):  # Ensure -2 index is valid
                                prev_candle_high = safe_decimal_conversion(
                                    df_with_indicators["high"].iloc[prev_candle_idx]
                                )
                                prev_candle_low = safe_decimal_conversion(
                                    df_with_indicators["low"].iloc[prev_candle_idx]
                                )
                                # Use rolling high/low from the same previous candle's perspective
                                rolling_high_col = (
                                    f"rolling_high_{config.trap_filter_lookback_period}"
                                )
                                rolling_low_col = (
                                    f"rolling_low_{config.trap_filter_lookback_period}"
                                )
                                recent_high_val = safe_decimal_conversion(
                                    df_with_indicators[rolling_high_col].iloc[
                                        prev_candle_idx
                                    ]
                                )
                                recent_low_val = safe_decimal_conversion(
                                    df_with_indicators[rolling_low_col].iloc[
                                        prev_candle_idx
                                    ]
                                )

                                rejection_needed = (
                                    latest_atr
                                    * config.trap_filter_rejection_threshold_atr
                                )
                                wick_prox_val = (
                                    latest_atr * config.trap_filter_wick_proximity_atr
                                )

                                if (
                                    confirmed_enter_long
                                    and recent_high_val
                                    and prev_candle_high
                                    and latest_close
                                ):
                                    # Condition: Prev candle high near recent lookback high, and current price rejected significantly from that high
                                    if (
                                        abs(prev_candle_high - recent_high_val)
                                        < wick_prox_val
                                        and (recent_high_val - latest_close)
                                        >= rejection_needed
                                    ):
                                        trap_detected = True
                                        logger.info(
                                            f"{NEON['INFO']}Bull Trap Filter: Long entry suppressed. Prev candle high {prev_candle_high} near recent high {recent_high_val}, current close {latest_close} shows rejection.{NEON['RESET']}"
                                        )
                                elif (
                                    confirmed_enter_short
                                    and recent_low_val
                                    and prev_candle_low
                                    and latest_close
                                ):
                                    # Condition: Prev candle low near recent lookback low, and current price rejected significantly from that low
                                    if (
                                        abs(prev_candle_low - recent_low_val)
                                        < wick_prox_val
                                        and (latest_close - recent_low_val)
                                        >= rejection_needed
                                    ):
                                        trap_detected = True
                                        logger.info(
                                            f"{NEON['INFO']}Bear Trap Filter: Short entry suppressed. Prev candle low {prev_candle_low} near recent low {recent_low_val}, current close {latest_close} shows rejection.{NEON['RESET']}"
                                        )

                        if trade_allowed_by_zone and not trap_detected:
                            entry_side = config.pos_none
                            if confirmed_enter_long:
                                entry_side = config.pos_long
                            elif confirmed_enter_short:
                                entry_side = config.pos_short

                            if entry_side != config.pos_none:
                                # Calculate SL and initial TP for the new order
                                sl_dist = latest_atr * config.atr_stop_loss_multiplier
                                sl_price = (
                                    (latest_close - sl_dist)
                                    if entry_side == config.pos_long
                                    else (latest_close + sl_dist)
                                )

                                tp_price = None
                                if config.enable_take_profit:
                                    tp_dist = (
                                        latest_atr * config.atr_take_profit_multiplier
                                    )
                                    tp_price = (
                                        (latest_close + tp_dist)
                                        if entry_side == config.pos_long
                                        else (latest_close - tp_dist)
                                    )

                                # Validate SL/TP relative to entry price
                                valid_sl_tp = True
                                if entry_side == config.pos_long:
                                    if sl_price >= latest_close:
                                        valid_sl_tp = False
                                        logger.warning(
                                            f"Invalid SL for LONG: SL {sl_price} >= Entry {latest_close}"
                                        )
                                    if tp_price and tp_price <= latest_close:
                                        valid_sl_tp = False
                                        logger.warning(
                                            f"Invalid TP for LONG: TP {tp_price} <= Entry {latest_close}"
                                        )
                                elif entry_side == config.pos_short:
                                    if sl_price <= latest_close:
                                        valid_sl_tp = False
                                        logger.warning(
                                            f"Invalid SL for SHORT: SL {sl_price} <= Entry {latest_close}"
                                        )
                                    if tp_price and tp_price >= latest_close:
                                        valid_sl_tp = False
                                        logger.warning(
                                            f"Invalid TP for SHORT: TP {tp_price} >= Entry {latest_close}"
                                        )

                                if valid_sl_tp:
                                    new_part_created = place_risked_order(
                                        exchange,
                                        config,
                                        entry_side,
                                        latest_close,
                                        sl_price,
                                        tp_price,
                                        latest_atr,
                                        df_with_indicators,
                                    )
                                    if new_part_created:
                                        # Reset signal persistence counters after successful entry
                                        persistent_signal_counter = {
                                            "long": 0,
                                            "short": 0,
                                        }
                                        last_signal_type = None
                                else:
                                    logger.warning(
                                        f"Entry for {entry_side} skipped due to invalid SL/TP calculation relative to entry price {latest_close}."
                                    )
                elif current_pos_side != config.pos_none:  # Position exists
                    logger.debug(
                        f"{NEON['INFO']}Position {current_pos_side} exists (Qty: {current_pos_qty}). No new entries.{NEON['RESET']}"
                    )
                elif (
                    len(_active_trade_parts) >= config.max_active_trade_parts
                ):  # Max parts reached
                    logger.debug(
                        f"{NEON['INFO']}Max active trade parts ({config.max_active_trade_parts}) reached. No new entries.{NEON['RESET']}"
                    )

                # --- Position Management (applied to each active part) ---
                active_parts_copy = list(
                    _active_trade_parts
                )  # Iterate over a copy for safe modification
                for part in active_parts_copy:
                    if part.get("symbol") != config.symbol:
                        continue  # Should not happen if bot manages one symbol

                    part_id = part.get("part_id", "N/A")
                    part_side = part.get("side")
                    part_entry_price = part.get("entry_price")
                    part_qty = part.get("qty")
                    part_sl_price = part.get(
                        "sl_price"
                    )  # SL price as known by bot (should match exchange)
                    part_tp_price = part.get("tp_price")  # TP price as known by bot

                    if not all(
                        [
                            part_id,
                            part_side,
                            isinstance(part_entry_price, Decimal),
                            isinstance(part_qty, Decimal),
                            isinstance(part_sl_price, Decimal),
                        ]
                    ):
                        logger.warning(
                            f"Skipping management for part {part_id}: missing critical data."
                        )
                        continue
                    if (
                        part_qty <= CONFIG.position_qty_epsilon
                    ):  # Part is effectively closed
                        logger.info(
                            f"Part {part_id} has zero/negligible quantity. Removing from active list."
                        )
                        _active_trade_parts = [
                            p
                            for p in _active_trade_parts
                            if p.get("part_id") != part_id
                        ]
                        continue

                    # Trend Contradiction Check (early exit if trend quickly reverses after entry)
                    if (
                        config.enable_trend_contradiction_cooldown
                        and part.get("side")
                        and not part.get("contradiction_checked", False)
                    ):  # Check only once per part
                        time_since_entry_ms = (now * 1000) - part.get(
                            "entry_time_ms", 0
                        )
                        # Check within a window of N candles after entry (N * interval_seconds)
                        # Assuming interval is in minutes, convert to ms. Example: 1m interval = 60000ms
                        # This needs interval_duration_ms. For simplicity, use sleep_seconds as proxy for candle formation time.
                        # A more robust way is to get interval duration from config.interval string.
                        # For now, use a fixed number of main loop cycles (sleep_seconds) as proxy.
                        check_window_ms = (
                            config.trend_contradiction_check_candles_after_entry
                            * config.sleep_seconds
                            * 1000
                        )

                        if 0 < time_since_entry_ms < check_window_ms:
                            # Get current confirmation trend (example for DualSupertrend)
                            confirm_trend_now = (
                                df_with_indicators["confirm_trend"].iloc[-1]
                                if config.strategy_name
                                == StrategyName.DUAL_SUPERTREND_MOMENTUM
                                else pd.NA
                            )

                            contradiction_found = False
                            if (
                                part_side == config.pos_long
                                and confirm_trend_now is False
                            ):  # Was long, trend flipped short
                                contradiction_found = True
                            elif (
                                part_side == config.pos_short
                                and confirm_trend_now is True
                            ):  # Was short, trend flipped long
                                contradiction_found = True

                            if contradiction_found:
                                logger.warning(
                                    f"{NEON['WARNING']}Trend contradiction for part {part_id} shortly after entry! "
                                    f"Side: {part_side}, Confirm Trend Now: {confirm_trend_now}. Activating cooldown.{NEON['RESET']}"
                                )
                                contradiction_cooldown_active_until = (
                                    now + config.trend_contradiction_cooldown_seconds
                                )
                                # Optionally, could also close the position here if desired.
                        part["contradiction_checked"] = (
                            True  # Mark as checked to avoid re-triggering cooldown
                        )

                    # Calculate current unrealized PNL for this part
                    current_unrealized_pnl = Decimal(0)
                    if part_side == config.pos_long:
                        current_unrealized_pnl = (
                            latest_close - part_entry_price
                        ) * part_qty
                    elif part_side == config.pos_short:
                        current_unrealized_pnl = (
                            part_entry_price - latest_close
                        ) * part_qty

                    part.get("recent_pnls", deque()).append(
                        current_unrealized_pnl
                    )  # Update PNL history for momentum
                    part["last_known_pnl"] = (
                        current_unrealized_pnl  # Store for other checks
                    )

                    # Profit Momentum SL Tightening (modifies SL on exchange)
                    if (
                        config.enable_profit_momentum_sl_tighten
                        and len(part.get("recent_pnls", []))
                        == config.profit_momentum_window
                        and latest_atr > 0
                    ):
                        recent_pnls_list = list(part["recent_pnls"])
                        # Check if PNL has been consistently increasing and is currently positive
                        is_profit_increasing = all(
                            recent_pnls_list[i] > recent_pnls_list[i - 1]
                            for i in range(1, len(recent_pnls_list))
                        )
                        is_currently_profitable = recent_pnls_list[-1] > 0

                        if is_currently_profitable and is_profit_increasing:
                            original_atr_at_entry = part.get(
                                "atr_at_entry", latest_atr
                            )  # Use ATR at entry for consistent SL distance
                            if original_atr_at_entry <= 0:
                                original_atr_at_entry = (
                                    latest_atr  # Fallback if invalid
                                )

                            tightened_sl_dist = (
                                original_atr_at_entry * config.atr_stop_loss_multiplier
                            ) * config.profit_momentum_sl_tighten_factor
                            new_sl_price_candidate = (
                                (latest_close - tightened_sl_dist)
                                if part_side == config.pos_long
                                else (latest_close + tightened_sl_dist)
                            )

                            # Check if new SL is more aggressive (better for profit protection) and valid (not crossing current price)
                            sl_improved_and_valid = False
                            if (
                                part_side == config.pos_long
                                and new_sl_price_candidate > part_sl_price
                                and new_sl_price_candidate < latest_close
                            ):
                                sl_improved_and_valid = True
                            elif (
                                part_side == config.pos_short
                                and new_sl_price_candidate < part_sl_price
                                and new_sl_price_candidate > latest_close
                            ):
                                sl_improved_and_valid = True

                            if sl_improved_and_valid:
                                logger.info(
                                    f"{NEON['ACTION']}Profit Momentum: Attempting SL tighten for part {part_id} from {part_sl_price} to {new_sl_price_candidate:.{config.MARKET_INFO['precision']['price'] if config.MARKET_INFO else 2}}{NEON['RESET']}"
                                )
                                modify_position_sl_tp(
                                    exchange,
                                    config,
                                    part,
                                    new_sl=new_sl_price_candidate,
                                )  # TP remains unchanged by this logic

                    # Breakeven SL (modifies SL on exchange)
                    if (
                        config.enable_breakeven_sl
                        and not part.get("breakeven_set", False)
                        and latest_atr > 0
                    ):
                        # Calculate profit in terms of ATRs
                        profit_per_unit = (
                            (latest_close - part_entry_price)
                            if part_side == config.pos_long
                            else (part_entry_price - latest_close)
                        )
                        current_profit_atr = (
                            profit_per_unit / latest_atr
                            if latest_atr > 0
                            else Decimal(0)
                        )

                        atr_target_met = (
                            current_profit_atr >= config.breakeven_profit_atr_target
                        )
                        abs_pnl_target_met = (
                            current_unrealized_pnl >= config.breakeven_min_abs_pnl_usdt
                        )  # Absolute PNL check

                        if atr_target_met and abs_pnl_target_met:
                            new_sl_price_candidate = (
                                part_entry_price  # Move SL to entry price
                            )

                            # Check if moving to breakeven is an improvement and valid
                            sl_improved_and_valid_be = False
                            if (
                                part_side == config.pos_long
                                and new_sl_price_candidate > part_sl_price
                            ):  # SL moves up
                                sl_improved_and_valid_be = True
                            elif (
                                part_side == config.pos_short
                                and new_sl_price_candidate < part_sl_price
                            ):  # SL moves down
                                sl_improved_and_valid_be = True

                            if sl_improved_and_valid_be:
                                logger.info(
                                    f"{NEON['ACTION']}Breakeven SL Triggered: Attempting to move SL for part {part_id} to entry price {new_sl_price_candidate}{NEON['RESET']}"
                                )
                                if modify_position_sl_tp(
                                    exchange,
                                    config,
                                    part,
                                    new_sl=new_sl_price_candidate,
                                ):  # TP remains unchanged
                                    part["breakeven_set"] = (
                                        True  # Mark as set to avoid re-triggering
                                    )
                                    save_persistent_state(
                                        force_heartbeat=True
                                    )  # Save state after successful BE set

                    # Last Chance Exit (preemptive exit if price moves adversely near SL)
                    part.get("adverse_candle_closes", deque()).append(
                        latest_close
                    )  # Track recent closes
                    if (
                        config.enable_last_chance_exit
                        and latest_atr > 0
                        and len(part.get("adverse_candle_closes", []))
                        >= config.last_chance_consecutive_adverse_candles
                    ):
                        adverse_closes_list = list(part["adverse_candle_closes"])
                        # Check for consecutive adverse movements
                        is_cons_adverse = False
                        if (
                            part_side == config.pos_long
                        ):  # For long, adverse is price going down
                            is_cons_adverse = all(
                                adverse_closes_list[i] < adverse_closes_list[i - 1]
                                for i in range(1, len(adverse_closes_list))
                            )
                        else:  # For short, adverse is price going up
                            is_cons_adverse = all(
                                adverse_closes_list[i] > adverse_closes_list[i - 1]
                                for i in range(1, len(adverse_closes_list))
                            )

                        if is_cons_adverse:
                            dist_to_sl = abs(latest_close - part_sl_price)
                            sl_prox_thresh_price = (
                                latest_atr * config.last_chance_sl_proximity_atr
                            )  # Proximity threshold in price terms
                            if dist_to_sl <= sl_prox_thresh_price:
                                logger.warning(
                                    f"{NEON['WARNING']}Last Chance Exit Triggered for part {part_id}: "
                                    f"{config.last_chance_consecutive_adverse_candles} adverse candles and price ({latest_close}) "
                                    f"is near SL ({part_sl_price}). Distance: {dist_to_sl:.{config.MARKET_INFO['precision']['price'] if config.MARKET_INFO else 2}}, "
                                    f"Threshold: {sl_prox_thresh_price:.{config.MARKET_INFO['precision']['price'] if config.MARKET_INFO else 2}}).{NEON['RESET']}"
                                )
                                if close_position_part(
                                    exchange,
                                    config,
                                    part,
                                    "Last Chance Preemptive Exit",
                                ):
                                    # If part closed, it's removed from _active_trade_parts, so loop should continue to next part or end
                                    continue  # Move to next part in active_parts_copy if any

                    # Ehlers Fisher Scaled Exit (Strategy-specific partial close)
                    if (
                        config.strategy_name == StrategyName.EHLERS_FISHER
                        and config.ehlers_enable_divergence_scaled_exit
                        and not part.get("divergence_exit_taken", False)
                        and "entry_fisher_value" in part
                        and part["entry_fisher_value"] is not None
                        and config.qty_step is not None
                    ):  # Ensure qty_step is available
                        fisher_now = safe_decimal_conversion(
                            df_with_indicators["ehlers_fisher"].iloc[-1]
                        )
                        signal_now = safe_decimal_conversion(
                            df_with_indicators["ehlers_signal"].iloc[-1]
                        )

                        if not pd.isna(fisher_now) and not pd.isna(signal_now):
                            entry_fisher = part["entry_fisher_value"]
                            entry_signal = part.get(
                                "entry_signal_value", pd.NA
                            )  # May not always be set if old state
                            if not pd.isna(entry_fisher) and not pd.isna(entry_signal):
                                initial_spread = abs(entry_fisher - entry_signal)
                                current_spread = abs(fisher_now - signal_now)
                                divergence_met = False
                                # Check for divergence: Fisher moving back towards signal from an extreme, spread reducing
                                if (
                                    part_side == config.pos_long
                                    and fisher_now > signal_now
                                    and fisher_now < entry_fisher
                                    and current_spread
                                    < (
                                        initial_spread
                                        * config.ehlers_divergence_threshold_factor
                                    )
                                ):
                                    divergence_met = True
                                elif (
                                    part_side == config.pos_short
                                    and fisher_now < signal_now
                                    and fisher_now > entry_fisher
                                    and current_spread
                                    < (
                                        initial_spread
                                        * config.ehlers_divergence_threshold_factor
                                    )
                                ):
                                    divergence_met = True

                                if divergence_met:
                                    qty_to_close_f = (
                                        part_qty
                                        * config.ehlers_divergence_exit_percentage
                                    )
                                    # Adjust to lot size
                                    qty_to_close = (
                                        (qty_to_close_f // config.qty_step)
                                        * config.qty_step
                                        if config.qty_step > 0
                                        else qty_to_close_f
                                    )

                                    if qty_to_close > CONFIG.position_qty_epsilon:
                                        logger.info(
                                            f"{NEON['ACTION']}Ehlers Divergence Scaled Exit ({config.ehlers_divergence_exit_percentage * 100}%): "
                                            f"Part {part_id}, attempting to close Qty {qty_to_close}{NEON['RESET']}"
                                        )

                                        # Create a temporary part dict for closing this specific quantity
                                        # This does not affect the main SL/TP on the exchange for the overall position,
                                        # as market close of a portion just reduces the position size.
                                        temp_part_for_partial_close = part.copy()
                                        temp_part_for_partial_close["qty"] = (
                                            qty_to_close
                                        )

                                        if close_position_part(
                                            exchange,
                                            config,
                                            temp_part_for_partial_close,
                                            "Ehlers Divergence Partial Exit",
                                        ):
                                            part["qty"] -= (
                                                qty_to_close  # Reduce original part quantity
                                            )
                                            part["divergence_exit_taken"] = (
                                                True  # Mark this type of exit as taken
                                            )

                                            if (
                                                part["qty"]
                                                <= CONFIG.position_qty_epsilon
                                            ):  # If remaining qty is negligible
                                                logger.info(
                                                    f"Part {part_id} fully closed due to negligible remainder after Ehlers divergence partial exit."
                                                )
                                                _active_trade_parts = [
                                                    p
                                                    for p in _active_trade_parts
                                                    if p.get("part_id") != part_id
                                                ]
                                            else:  # Update the initial_usdt_value if part still active
                                                part["initial_usdt_value"] = (
                                                    part["qty"] * part_entry_price
                                                )

                                            save_persistent_state(force_heartbeat=True)
                                            # If part fully closed, continue might be needed if the original part was removed from _active_trade_parts
                                            # However, we are iterating a copy. The original 'part' dict is modified here.
                                            if part_id not in [
                                                p.get("part_id")
                                                for p in _active_trade_parts
                                            ]:  # Check if original part was removed
                                                continue  # Move to next part in active_parts_copy
                                    else:
                                        logger.debug(
                                            f"Ehlers divergence scaled exit: calculated qty_to_close ({qty_to_close}) too small or close failed."
                                        )

                    # Standard Exit Signals from Strategy (e.g., Supertrend flip)
                    # This should be one of the last checks, as other exits might be more preemptive.
                    strategy_exit_triggered = False
                    exit_reason = "Strategy Exit Signal"
                    if part_side == config.pos_long and signals.get("exit_long"):
                        strategy_exit_triggered = True
                        exit_reason = signals.get("exit_reason", "Strategy Exit Long")
                    elif part_side == config.pos_short and signals.get("exit_short"):
                        strategy_exit_triggered = True
                        exit_reason = signals.get("exit_reason", "Strategy Exit Short")

                    if strategy_exit_triggered:
                        logger.info(
                            f"{NEON['ACTION']}Strategy omen to EXIT {part_side} for part {part_id} ({config.symbol}): {exit_reason}{NEON['RESET']}"
                        )
                        if close_position_part(exchange, config, part, exit_reason):
                            continue  # Part closed, move to next

                    # SL/TP Breach Logging (actual execution is by exchange)
                    # This is for bot's awareness and logging if price touches SL/TP levels.
                    # Assumes exchange handles the actual fill. Bot will see position closed on next cycle.
                    if part_sl_price is not None:  # Check if SL is set for the part
                        if (
                            part_side == config.pos_long
                            and latest_close <= part_sl_price
                        ) or (
                            part_side == config.pos_short
                            and latest_close >= part_sl_price
                        ):
                            logger.warning(
                                f"{NEON['WARNING']}Part {part_id} SL price {part_sl_price} potentially breached by current close {latest_close}. "
                                f"Exchange should handle closure. Bot will detect on next cycle.{NEON['RESET']}"
                            )
                            # Bot does not close it here; exchange does. Loop will pick up closed position later.

                    if (
                        config.enable_take_profit and part_tp_price is not None
                    ):  # Check if TP is set and enabled
                        if (
                            part_side == config.pos_long
                            and latest_close >= part_tp_price
                        ) or (
                            part_side == config.pos_short
                            and latest_close <= part_tp_price
                        ):
                            logger.info(
                                f"{NEON['INFO']}Part {part_id} TP price {part_tp_price} potentially reached by current close {latest_close}. "
                                f"Exchange should handle closure. Bot will detect on next cycle.{NEON['RESET']}"
                            )
                            # Similar to SL, exchange handles this.

                save_persistent_state()  # Save state at end of each successful iteration's management phase
                logger.debug(
                    f"{NEON['COMMENT']}# Cycle {loop_counter} complete. Resting {config.sleep_seconds}s...{NEON['RESET']}"
                )
                time.sleep(config.sleep_seconds)

            except KeyboardInterrupt:  # User interruption
                logger.warning(
                    f"\n{NEON['WARNING']}Sorcerer's intervention (KeyboardInterrupt)! Pyrmethus prepares for slumber...{NEON['RESET']}"
                )
                if hasattr(CONFIG, "send_notification_method"):
                    CONFIG.send_notification_method(
                        "Pyrmethus Shutdown", "Manual shutdown (Ctrl+C) initiated."
                    )
                break  # Exit main while loop
            except (
                Exception
            ) as e_iter:  # Catch-all for errors within a single loop iteration
                logger.critical(
                    f"{NEON['CRITICAL']}Critical error in main loop iteration {loop_counter}! Error: {e_iter}{NEON['RESET']}"
                )
                logger.debug(traceback.format_exc())
                if hasattr(CONFIG, "send_notification_method"):
                    CONFIG.send_notification_method(
                        "Pyrmethus Critical Error",
                        f"Loop iteration {loop_counter} crashed: {str(e_iter)[:150]}",
                    )
                logger.info(
                    f"Resting longer ({config.sleep_seconds * 5}s) after critical error in iteration to allow recovery or manual intervention."
                )
                time.sleep(config.sleep_seconds * 5)  # Longer pause after an error
    finally:  # Cleanup actions when main loop exits (normally or via break/exception)
        logger.info(
            f"{NEON['HEADING']}=== Pyrmethus Spell Concludes ({datetime.now(pytz.utc).isoformat()}) ==={NEON['RESET']}"
        )
        if exchange:  # Ensure exchange object exists
            close_all_symbol_positions(
                exchange, config, "Spell Ending Sequence / Shutdown"
            )

        save_persistent_state(force_heartbeat=True)  # Final save of state
        trade_metrics.summary()  # Log final trade summary

        logger.info(
            f"{NEON['COMMENT']}# Energies settle. Until next conjuring.{NEON['RESET']}"
        )
        if hasattr(CONFIG, "send_notification_method"):
            CONFIG.send_notification_method(
                "Pyrmethus Offline",
                f"Spell concluded for {config.symbol}. Review logs.",
            )


if __name__ == "__main__":
    logger.info(
        f"{NEON['COMMENT']}# Pyrmethus prepares to breach veil to the exchange realm...{NEON['RESET']}"
    )
    exchange_instance = None  # Initialize to None
    try:
        exchange_params: Dict[str, Any] = {
            "apiKey": CONFIG.api_key,
            "secret": CONFIG.api_secret,
            "options": {
                "defaultType": "swap",  # For futures/swaps
                "adjustForTimeDifference": True,  # CCXT handles time sync issues
                "brokerId": f"PYRMETHUS{PYRMETHUS_VERSION.replace('.', '')}",  # Bybit specific for partner tracking
            },
            "enableRateLimit": True,  # Use CCXT's built-in rate limiter
            "recvWindow": CONFIG.default_recv_window,  # For Bybit, might need adjustment
        }
        if CONFIG.PAPER_TRADING_MODE:
            logger.warning(
                f"{NEON['WARNING']}PAPER TRADING MODE IS ACTIVE. Connecting to Bybit Testnet.{NEON['RESET']}"
            )
            exchange_params["urls"] = {"api": "https://api-testnet.bybit.com"}

        exchange_instance = ccxt.bybit(exchange_params)
        logger.info(
            f"Attempting to connect to: {exchange_instance.id} (CCXT Version: {ccxt.__version__})"
        )

        # Test connection and load markets
        markets = exchange_instance.load_markets()
        if CONFIG.symbol not in markets:
            err_msg = f"Symbol {CONFIG.symbol} not found in {exchange_instance.id} market runes. Please check SYMBOL in config."
            logger.critical(f"{NEON['CRITICAL']}{err_msg}{NEON['RESET']}")
            if hasattr(CONFIG, "send_notification_method"):
                CONFIG.send_notification_method("Pyrmethus Startup Fail", err_msg)
            sys.exit(1)

        CONFIG.MARKET_INFO = markets[CONFIG.symbol]

        # Extract tick size (price precision)
        price_prec_raw = CONFIG.MARKET_INFO.get("precision", {}).get("price")
        CONFIG.tick_size = safe_decimal_conversion(price_prec_raw)
        if CONFIG.tick_size is None or CONFIG.tick_size <= Decimal(0):
            logger.warning(
                f"{NEON['WARNING']}Could not determine valid tick size from market info for {CONFIG.symbol}. Got: {price_prec_raw}. Using a small default (1e-8). This may cause issues.{NEON['RESET']}"
            )
            CONFIG.tick_size = Decimal(
                "1e-8"
            )  # Fallback, but orders might fail if this is wrong

        # Extract quantity step (amount precision)
        amount_prec_raw = CONFIG.MARKET_INFO.get("precision", {}).get("amount")
        CONFIG.qty_step = safe_decimal_conversion(amount_prec_raw)
        if CONFIG.qty_step is None or CONFIG.qty_step <= Decimal(0):
            logger.warning(
                f"{NEON['WARNING']}Could not determine valid quantity step from market info for {CONFIG.symbol}. Got: {amount_prec_raw}. Using a small default (1e-8). This may cause issues.{NEON['RESET']}"
            )
            CONFIG.qty_step = Decimal("1e-8")  # Fallback

        logger.success(
            f"{NEON['SUCCESS']}Market runes for {CONFIG.symbol}: Price Tick {CONFIG.tick_size}, Amount Step {CONFIG.qty_step}{NEON['RESET']}"
        )  # type: ignore [attr-defined]

        # Set leverage (important for futures)
        category = (
            "linear" if CONFIG.symbol.endswith(":USDT") else "inverse"
        )  # Determine category for Bybit
        # Note: Bybit might also have "spot" category, but this bot is for futures.
        try:
            leverage_params = {"category": category}
            # Bybit set_leverage usually applies to both long and short for one-way mode.
            # For hedge mode, it might need buyLeverage and sellLeverage. Assume one-way.
            logger.info(
                f"Setting leverage to {CONFIG.leverage}x for {CONFIG.symbol} (Category: {category})..."
            )
            response = exchange_instance.set_leverage(
                CONFIG.leverage, CONFIG.symbol, params=leverage_params
            )
            # Parse response carefully, Bybit's response can vary.
            logger.success(
                f"{NEON['SUCCESS']}Leverage for {CONFIG.symbol} set/confirmed. Response (partial): {json.dumps(response, default=str)[:200]}...{NEON['RESET']}"
            )  # type: ignore [attr-defined]
        except Exception as e_lev:
            logger.warning(
                f"{NEON['WARNING']}Could not set leverage for {CONFIG.symbol} (may be pre-set, or an issue with API/permissions): {e_lev}{NEON['RESET']}"
            )
            if hasattr(CONFIG, "send_notification_method"):
                CONFIG.send_notification_method(
                    "Pyrmethus Leverage Warn",
                    f"Leverage set issue for {CONFIG.symbol}: {str(e_lev)[:60]}",
                )

        logger.success(
            f"{NEON['SUCCESS']}Successfully connected to {exchange_instance.id} for trading {CONFIG.symbol}.{NEON['RESET']}"
        )  # type: ignore [attr-defined]
        if hasattr(CONFIG, "send_notification_method"):
            CONFIG.send_notification_method(
                "Pyrmethus Online",
                f"Connected to {exchange_instance.id} for {CONFIG.symbol} @ {CONFIG.leverage}x.",
            )

    except AttributeError as e_attr:  # e.g. ccxt.bybit misspelled or ccxt version issue
        logger.critical(
            f"{NEON['CRITICAL']}Exchange attribute error during initialization: {e_attr}. CCXT library issue or invalid exchange ID?{NEON['RESET']}"
        )
        sys.exit(1)
    except ccxt.AuthenticationError as e_auth:
        logger.critical(
            f"{NEON['CRITICAL']}Authentication failed! Please check API keys and permissions: {e_auth}{NEON['RESET']}"
        )
        if hasattr(CONFIG, "send_notification_method"):
            CONFIG.send_notification_method(
                "Pyrmethus Auth Fail", f"API Authentication Error. Check keys."
            )
        sys.exit(1)
    except ccxt.NetworkError as e_net:  # Connectivity issues
        logger.critical(
            f"{NEON['CRITICAL']}Network Error during exchange connection: {e_net}. Check internet connection and DNS.{NEON['RESET']}"
        )
        if hasattr(CONFIG, "send_notification_method"):
            CONFIG.send_notification_method(
                "Pyrmethus Network Error",
                f"Cannot connect to exchange: {str(e_net)[:80]}",
            )
        sys.exit(1)
    except (
        ccxt.ExchangeError
    ) as e_exch:  # General exchange API errors not caught by specific types
        logger.critical(
            f"{NEON['CRITICAL']}Exchange API Error during initialization: {e_exch}. Check API status, symbol, or account permissions.{NEON['RESET']}"
        )
        if hasattr(CONFIG, "send_notification_method"):
            CONFIG.send_notification_method(
                "Pyrmethus API Error", f"Exchange API issue: {str(e_exch)[:80]}"
            )
        sys.exit(1)
    except Exception as e_general_init:  # Catch-all for any other startup errors
        logger.critical(
            f"{NEON['CRITICAL']}A critical error occurred during Pyrmethus's awakening (exchange init or pre-requisites): {e_general_init}{NEON['RESET']}"
        )
        logger.debug(traceback.format_exc())
        if hasattr(CONFIG, "send_notification_method"):
            CONFIG.send_notification_method(
                "Pyrmethus Init Error",
                f"General exchange init failed: {str(e_general_init)[:80]}",
            )
        sys.exit(1)

    # If exchange_instance is successfully created, start the main trading loop
    if exchange_instance:
        main_loop(exchange_instance, CONFIG)
    else:  # Should have been caught by specific exceptions, but as a final safeguard
        logger.critical(
            f"{NEON['CRITICAL']}Exchange instance not initialized. Pyrmethus cannot cast spells. Aborting.{NEON['RESET']}"
        )
        sys.exit(1)
