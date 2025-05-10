#!/usr/bin/env python3
# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v2.8.3 (Enhanced Weave of Wisdom)
# Features implemented core trading logic, live market interaction,
# enhanced state management, robust error handling, and numerous strategic refinements.
# Previous Weave: v2.8.2 (Operational Infusion)

"""
High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 2.8.3 (Enhanced Weave of Wisdom) - Enhanced by Pyrmethus

Dependencies & Setup (Termux):
1. Core Packages:
   pkg install python clang libcrypt libffi openssl rust termux-api
   (Rust and build essentials are often needed for cryptography dependencies)
2. Python Libraries:
   pip install ccxt pandas pandas-ta colorama python-dotenv retry pytz

WARNING: This script can execute live trades. Use with extreme caution.
Thoroughly test in a simulated environment or with minimal capital.
Pyrmethus bears no responsibility for financial outcomes.

Enhancements in v2.8.3 (by Pyrmethus - Integration of User Snippets):
- General Enhancements:
    - Profit Momentum SL Tightening: Dynamically tightens SL if PNL consistently increases.
    - Whipsaw Cooldown: Pauses trading if too many trades occur in a short period.
    - Max Active Trade Parts: Limits concurrent open positions/parts.
    - Stateful Signal Confirmation: Requires entry signals to persist for N candles.
    - No-Trade Zones: Avoids entries around key daily H/L or round numbers.
    - Enhanced Breakeven SL: Requires min absolute PNL alongside ATR target for BE.
    - Session PNL Goal/Max Loss: Stops trading if session PNL targets/limits are hit.
    - Anti-Martingale Risk Sizing: Adjusts risk based on previous trade outcome.
    - Last Chance Exit: Preemptive exit if price moves strongly adverse near SL.
    - Consecutive Loss Limiter & Cooldown: Pauses after N consecutive losses.
    - Daily Max Trades Rest: Pauses after N trades in a day.
    - Limit Order Price Improvement Check: Adjusts limit orders if market offers better price.
- Dual Supertrend Momentum Strategy Enhancements:
    - Confirmation ST Stability Lookback: Ensures confirmation trend is stable.
    - Momentum Agreement with Primary ST: Requires momentum to be explicitly positive/negative.
    - Min Distance from ST Line for Entry: Ensures entry is not too far from ST line.
    - Fade Weaker Signal: Resolves rare conflicting primary ST flips.
    - Trap Filter (Bull/Bear): Avoids entries into potential trap patterns using rolling H/L.
- Ehlers Fisher Strategy Enhancements:
    - Extreme Zone Avoidance: Avoids new entries if Fisher is in overbought/oversold zones.
    - Scaled Exit on Divergence: Partial exit if Fisher diverges from signal line, indicating weakening momentum.
- State Persistence: Updated to handle new state variables including deques.
- Configuration: Numerous new parameters added to control these features.
- TODO: Actual exchange calls for dynamic SL modifications (e.g., `exchange.set_trading_stop`).
"""

# Standard Library Imports
import json
import logging
import os
import random  # For MockExchange if used
import subprocess  # For Termux notifications
import sys
import time
import traceback
import uuid  # For unique part IDs
from abc import ABC, abstractmethod
from collections import deque # For rolling windows / history
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, DivisionByZero, InvalidOperation, getcontext
from enum import Enum
from typing import Any, Union, Dict, List, Tuple, Optional, Type

import pytz  # For timezone-aware datetimes, the Chronomancer's ally

# Third-party Libraries - The Grimoire of External Magics
try:
    import ccxt
    import pandas as pd
    if not hasattr(pd, 'NA'):  # Ensure pandas version supports pd.NA, a subtle but crucial rune
        raise ImportError(
            "Pandas version < 1.0 not supported, pd.NA is required.")
    # type: ignore[import] # The Alchemist's Table for Technical Analysis
    import pandas_ta as ta
    from colorama import Back, Fore, Style  # The Prisms of Perception
    from colorama import init as colorama_init
    from dotenv import load_dotenv  # For whispering secrets from .env scrolls
    from retry import retry  # The Art of Tenacious Spellcasting
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
PYRMETHUS_VERSION = "2.8.3"
STATE_FILE_NAME = f"pyrmethus_phoenix_state_v{PYRMETHUS_VERSION.replace('.', '')}.json"
STATE_FILE_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), STATE_FILE_NAME)
HEARTBEAT_INTERVAL_SECONDS = 60
OHLCV_LIMIT = 200

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
if load_dotenv(dotenv_path=env_path):
    logging.getLogger("PreConfig").info(
        f"{NEON['INFO']}Secrets whispered from .env scroll: {NEON['VALUE']}{env_path}{NEON['RESET']}")
else:
    logging.getLogger("PreConfig").warning(
        f"{NEON['WARNING']}No .env scroll found at {NEON['VALUE']}{env_path}{NEON['WARNING']}. Relying on system environment variables or defaults.{NEON['RESET']}")
getcontext().prec = 18

# --- Helper Functions ---
def safe_decimal_conversion(value: Any, default_if_error: Any = pd.NA) -> Union[Decimal, Any]:
    if pd.isna(value): return default_if_error
    try: return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError): return default_if_error

def _format_for_log(value: Any, is_bool_trend: bool = False, precision: int = 4) -> str:
    if isinstance(value, Decimal): return f"{value:.{precision}f}"
    if is_bool_trend:
        if value is True: return "Up"
        if value is False: return "Down"
        return "Indeterminate"
    if pd.isna(value): return "N/A"
    return str(value)

def send_termux_notification(title: str, message: str, notification_id: int = 777) -> None:
    if 'CONFIG' not in globals() or not CONFIG.enable_notifications:
        if 'CONFIG' in globals(): logging.getLogger("TermuxNotification").debug("Termux notifications are disabled by configuration.")
        else: logging.getLogger("TermuxNotification").warning("Attempted Termux notification before CONFIG loaded or with notifications disabled.")
        return
    try:
        safe_title = json.dumps(title); safe_message = json.dumps(message)
        command = ["termux-notification", "--title", safe_title, "--content", safe_message, "--id", str(notification_id)]
        logging.getLogger("TermuxNotification").debug(f"{NEON['ACTION']}Attempting to send Termux notification: Title='{title}'{NEON['RESET']}")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=CONFIG.notification_timeout_seconds)
        if process.returncode == 0: logging.getLogger("TermuxNotification").info(f"{NEON['SUCCESS']}Termux notification '{title}' sent successfully.{NEON['RESET']}")
        else:
            err_msg = stderr.decode().strip() if stderr else "Unknown error"
            logging.getLogger("TermuxNotification").error(f"{NEON['ERROR']}Failed to send Termux notification '{title}'. Return code: {process.returncode}. Error: {err_msg}{NEON['RESET']}")
    except FileNotFoundError: logging.getLogger("TermuxNotification").error(f"{NEON['ERROR']}Termux API command 'termux-notification' not found.{NEON['RESET']}")
    except subprocess.TimeoutExpired: logging.getLogger("TermuxNotification").error(f"{NEON['ERROR']}Termux notification command timed out for '{title}'.{NEON['RESET']}")
    except Exception as e:
        logging.getLogger("TermuxNotification").error(f"{NEON['ERROR']}Unexpected error sending Termux notification '{title}': {e}{NEON['RESET']}")
        logging.getLogger("TermuxNotification").debug(traceback.format_exc())

# --- Enums ---
class StrategyName(str, Enum): DUAL_SUPERTREND_MOMENTUM = "DUAL_SUPERTREND_MOMENTUM"; EHLERS_FISHER = "EHLERS_FISHER"
class VolatilityRegime(Enum): LOW = "LOW"; NORMAL = "NORMAL"; HIGH = "HIGH"
class OrderEntryType(str, Enum): MARKET = "MARKET"; LIMIT = "LIMIT"

# --- Configuration Class ---
class Config:
    def __init__(self) -> None:
        _pre_logger = logging.getLogger("ConfigModule")
        _pre_logger.info(f"{NEON['HEADING']}--- Summoning Configuration Runes v{PYRMETHUS_VERSION} ---{NEON['RESET']}")
        self.api_key: str = self._get_env("BYBIT_API_KEY", required=True, secret=True)
        self.api_secret: str = self._get_env("BYBIT_API_SECRET", required=True, secret=True)
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT")
        self.interval: str = self._get_env("INTERVAL", "1m")
        self.leverage: int = self._get_env("LEVERAGE", 10, cast_type=int)
        self.sleep_seconds: int = self._get_env("SLEEP_SECONDS", 10, cast_type=int)
        self.strategy_name: StrategyName = StrategyName(self._get_env("STRATEGY_NAME", StrategyName.DUAL_SUPERTREND_MOMENTUM.value).upper())
        self.strategy_instance: 'TradingStrategy'
        self.risk_per_trade_percentage: Decimal = self._get_env("RISK_PER_TRADE_PERCENTAGE", "0.005", cast_type=Decimal)
        self.max_order_usdt_amount: Decimal = self._get_env("MAX_ORDER_USDT_AMOUNT", "1000.0", cast_type=Decimal)
        self.required_margin_buffer: Decimal = self._get_env("REQUIRED_MARGIN_BUFFER", "1.05", cast_type=Decimal) # Not actively used, informational
        self.max_account_margin_ratio: Decimal = self._get_env("MAX_ACCOUNT_MARGIN_RATIO", "0.5", cast_type=Decimal) # Not actively used, informational
        self.enable_max_drawdown_stop: bool = self._get_env("ENABLE_MAX_DRAWDOWN_STOP", "true", cast_type=bool)
        self.max_drawdown_percent: Decimal = self._get_env("MAX_DRAWDOWN_PERCENT", "0.05", cast_type=Decimal)
        self.atr_stop_loss_multiplier: Decimal = self._get_env("ATR_STOP_LOSS_MULTIPLIER", "1.5", cast_type=Decimal)
        self.atr_calculation_period: int = self._get_env("ATR_CALCULATION_PERIOD", 14, cast_type=int) # Generic ATR period
        self.entry_order_type: OrderEntryType = OrderEntryType(self._get_env("ENTRY_ORDER_TYPE", OrderEntryType.MARKET.value).upper())
        self.limit_order_offset_atr_percentage: Decimal = self._get_env("LIMIT_ORDER_OFFSET_ATR_PERCENTAGE", "0.1", cast_type=Decimal) # For ATR based limit offset
        self.order_fill_timeout_seconds: int = self._get_env("ORDER_FILL_TIMEOUT_SECONDS", 15, cast_type=int)
        self.enable_notifications: bool = self._get_env("ENABLE_NOTIFICATIONS", "true", cast_type=bool)
        self.notification_timeout_seconds: int = self._get_env("NOTIFICATION_TIMEOUT_SECONDS", 10, cast_type=int)
        self.default_recv_window: int = self._get_env("DEFAULT_RECV_WINDOW", 13000, cast_type=int)
        self.side_buy: str = "buy"; self.side_sell: str = "sell"
        self.pos_long: str = "Long"; self.pos_short: str = "Short"; self.pos_none: str = "None"
        self.usdt_symbol: str = "USDT"
        self.retry_count: int = 5; self.retry_delay_seconds: int = 5
        self.api_fetch_limit_buffer: int = 20
        self.position_qty_epsilon: Decimal = Decimal("1e-9")
        self.post_close_delay_seconds: int = 3
        self.MARKET_INFO: Optional[Dict[str, Any]] = None
        self.PAPER_TRADING_MODE: bool = self._get_env("PAPER_TRADING_MODE", "false", cast_type=bool)
        self.send_notification_method = send_termux_notification
        self.tick_size: Optional[Decimal] = None # Will be populated from MARKET_INFO

        # --- Dual Supertrend Momentum Strategy Params ---
        self.st_atr_length: int = self._get_env("ST_ATR_LENGTH", 10, cast_type=int)
        self.st_multiplier: Decimal = self._get_env("ST_MULTIPLIER", "2.0", cast_type=Decimal)
        self.confirm_st_atr_length: int = self._get_env("CONFIRM_ST_ATR_LENGTH", 20, cast_type=int)
        self.confirm_st_multiplier: Decimal = self._get_env("CONFIRM_ST_MULTIPLIER", "3.0", cast_type=Decimal)
        self.momentum_period: int = self._get_env("MOMENTUM_PERIOD", 14, cast_type=int)
        self.momentum_threshold: Decimal = self._get_env("MOMENTUM_THRESHOLD", "0", cast_type=Decimal)
        self.confirm_st_stability_lookback: int = self._get_env("CONFIRM_ST_STABILITY_LOOKBACK", 3, cast_type=int) # Snippet 3.1
        self.st_max_entry_distance_atr_multiplier: Optional[Decimal] = self._get_env("ST_MAX_ENTRY_DISTANCE_ATR_MULTIPLIER", "0.5", cast_type=Decimal, required=False) # Snippet 3.4

        # --- Ehlers Fisher Strategy Params ---
        self.ehlers_fisher_length: int = self._get_env("EHLERS_FISHER_LENGTH", 10, cast_type=int)
        self.ehlers_fisher_signal_length: int = self._get_env("EHLERS_FISHER_SIGNAL_LENGTH", 1, cast_type=int)
        self.ehlers_fisher_extreme_threshold_positive: Decimal = self._get_env("EHLERS_FISHER_EXTREME_THRESHOLD_POSITIVE", "2.0", cast_type=Decimal) # Snippet 3.3
        self.ehlers_fisher_extreme_threshold_negative: Decimal = self._get_env("EHLERS_FISHER_EXTREME_THRESHOLD_NEGATIVE", "-2.0", cast_type=Decimal) # Snippet 3.3
        self.ehlers_enable_divergence_scaled_exit: bool = self._get_env("EHLERS_ENABLE_DIVERGENCE_SCALED_EXIT", "false", cast_type=bool) # Snippet 4.2
        self.ehlers_divergence_threshold_factor: Decimal = self._get_env("EHLERS_DIVERGENCE_THRESHOLD_FACTOR", "0.75", cast_type=Decimal) # Snippet 4.2
        self.ehlers_divergence_exit_percentage: Decimal = self._get_env("EHLERS_DIVERGENCE_EXIT_PERCENTAGE", "0.3", cast_type=Decimal) # Snippet 4.2

        # --- General Enhancements from Snippets ---
        # Snippet 1.2: Profit Momentum SL Tightening
        self.enable_profit_momentum_sl_tighten: bool = self._get_env("ENABLE_PROFIT_MOMENTUM_SL_TIGHTEN", "false", cast_type=bool)
        self.profit_momentum_window: int = self._get_env("PROFIT_MOMENTUM_WINDOW", 3, cast_type=int)
        self.profit_momentum_sl_tighten_factor: Decimal = self._get_env("PROFIT_MOMENTUM_SL_TIGHTEN_FACTOR", "0.5", cast_type=Decimal)
        # Snippet 1.3: Whipsaw Cooldown
        self.enable_whipsaw_cooldown: bool = self._get_env("ENABLE_WHIPSAW_COOLDOWN", "true", cast_type=bool)
        self.whipsaw_max_trades_in_period: int = self._get_env("WHIPSAW_MAX_TRADES_IN_PERIOD", 3, cast_type=int)
        self.whipsaw_period_seconds: int = self._get_env("WHIPSAW_PERIOD_SECONDS", 300, cast_type=int)
        self.whipsaw_cooldown_seconds: int = self._get_env("WHIPSAW_COOLDOWN_SECONDS", 180, cast_type=int)
        # Snippet 1.4: Max Active Trade Parts
        self.max_active_trade_parts: int = self._get_env("MAX_ACTIVE_TRADE_PARTS", 3, cast_type=int)
        # Snippet 1.5: Stateful Signal Confirmation
        self.signal_persistence_candles: int = self._get_env("SIGNAL_PERSISTENCE_CANDLES", 1, cast_type=int)
        # Snippet 2.1: No-Trade Zones
        self.enable_no_trade_zones: bool = self._get_env("ENABLE_NO_TRADE_ZONES", "false", cast_type=bool)
        self.no_trade_zone_pct_around_key_level: Decimal = self._get_env("NO_TRADE_ZONE_PCT_AROUND_KEY_LEVEL", "0.002", cast_type=Decimal)
        self.key_round_number_step: Optional[Decimal] = self._get_env("KEY_ROUND_NUMBER_STEP", "1000", cast_type=Decimal, required=False)
        # Snippet 2.2: Breakeven SL Enhancements (Base BE logic needs to be added)
        self.enable_breakeven_sl: bool = self._get_env("ENABLE_BREAKEVEN_SL", "true", cast_type=bool)
        self.breakeven_profit_atr_target: Decimal = self._get_env("BREAKEVEN_PROFIT_ATR_TARGET", "1.0", cast_type=Decimal)
        self.breakeven_min_abs_pnl_usdt: Decimal = self._get_env("BREAKEVEN_MIN_ABS_PNL_USDT", "0.50", cast_type=Decimal)
        # Snippet 2.3: Session PNL Limits
        self.enable_session_pnl_limits: bool = self._get_env("ENABLE_SESSION_PNL_LIMITS", "false", cast_type=bool)
        self.session_profit_target_usdt: Optional[Decimal] = self._get_env("SESSION_PROFIT_TARGET_USDT", None, cast_type=Decimal, required=False)
        self.session_max_loss_usdt: Optional[Decimal] = self._get_env("SESSION_MAX_LOSS_USDT", None, cast_type=Decimal, required=False)
        # Snippet 2.4: Anti-Martingale Risk Sizing
        self.enable_anti_martingale_risk: bool = self._get_env("ENABLE_ANTI_MARTINGALE_RISK", "false", cast_type=bool)
        self.risk_reduction_factor_on_loss: Decimal = self._get_env("RISK_REDUCTION_FACTOR_ON_LOSS", "0.75", cast_type=Decimal)
        self.risk_increase_factor_on_win: Decimal = self._get_env("RISK_INCREASE_FACTOR_ON_WIN", "1.1", cast_type=Decimal)
        self.max_risk_pct_anti_martingale: Decimal = self._get_env("MAX_RISK_PCT_ANTI_MARTINGALE", "0.02", cast_type=Decimal) # Max 2%
        # Snippet 2.5: Last Chance Exit
        self.enable_last_chance_exit: bool = self._get_env("ENABLE_LAST_CHANCE_EXIT", "false", cast_type=bool)
        self.last_chance_consecutive_adverse_candles: int = self._get_env("LAST_CHANCE_CONSECUTIVE_ADVERSE_CANDLES", 2, cast_type=int)
        self.last_chance_sl_proximity_atr: Decimal = self._get_env("LAST_CHANCE_SL_PROXIMITY_ATR", "0.3", cast_type=Decimal)
        # Snippet 3.5: Trend Contradiction Cooldown
        self.enable_trend_contradiction_cooldown: bool = self._get_env("ENABLE_TREND_CONTRADICTION_COOLDOWN", "true", cast_type=bool)
        self.trend_contradiction_check_candles_after_entry: int = self._get_env("TREND_CONTRADICTION_CHECK_CANDLES_AFTER_ENTRY", 2, cast_type=int)
        self.trend_contradiction_cooldown_seconds: int = self._get_env("TREND_CONTRADICTION_COOLDOWN_SECONDS", 120, cast_type=int)
        # Snippet 4.3: Daily Max Trades Rest (Moved to TradeMetrics for state)
        self.enable_daily_max_trades_rest: bool = self._get_env("ENABLE_DAILY_MAX_TRADES_REST", "false", cast_type=bool)
        self.daily_max_trades_limit: int = self._get_env("DAILY_MAX_TRADES_LIMIT", 10, cast_type=int)
        self.daily_max_trades_rest_hours: int = self._get_env("DAILY_MAX_TRADES_REST_HOURS", 4, cast_type=int)
        # Snippet 4.4: Limit Order Price Improvement Check
        self.enable_limit_order_price_improvement_check: bool = self._get_env("ENABLE_LIMIT_ORDER_PRICE_IMPROVEMENT_CHECK", "true", cast_type=bool)
        # Snippet 4.5: Trap Filter
        self.enable_trap_filter: bool = self._get_env("ENABLE_TRAP_FILTER", "false", cast_type=bool)
        self.trap_filter_lookback_period: int = self._get_env("TRAP_FILTER_LOOKBACK_PERIOD", 20, cast_type=int)
        self.trap_filter_rejection_threshold_atr: Decimal = self._get_env("TRAP_FILTER_REJECTION_THRESHOLD_ATR", "1.0", cast_type=Decimal)
        self.trap_filter_wick_proximity_atr: Decimal = self._get_env("TRAP_FILTER_WICK_PROXIMITY_ATR", "0.2", cast_type=Decimal) # For checking if high/low was near rolling H/L
        # Consecutive Loss Limiter (from general earlier snippets, fits theme)
        self.enable_consecutive_loss_limiter: bool = self._get_env("ENABLE_CONSECUTIVE_LOSS_LIMITER", "true", cast_type=bool)
        self.max_consecutive_losses: int = self._get_env("MAX_CONSECUTIVE_LOSSES", 3, cast_type=int)
        self.consecutive_loss_cooldown_minutes: int = self._get_env("CONSECUTIVE_LOSS_COOLDOWN_MINUTES", 60, cast_type=int)

        self._validate_parameters()
        _pre_logger.info(f"{NEON['HEADING']}--- Configuration Runes v{PYRMETHUS_VERSION} Summoned and Verified ---{NEON['RESET']}")
        _pre_logger.info(f"{NEON['COMMENT']}# The chosen path: {NEON['STRATEGY']}{self.strategy_name.value}{NEON['RESET']}")

    def _get_env(self, key: str, default: Any = None, cast_type: type = str, required: bool = False, color: str = NEON["PARAM"], secret: bool = False) -> Any:
        # ... (existing _get_env logic, no changes needed for this integration)
        _logger = logging.getLogger("ConfigModule._get_env")
        value_str = os.getenv(key)
        source = "Env Var"
        value_to_cast: Any = None
        display_value = "********" if secret and value_str is not None else value_str

        if value_str is None:
            if required:
                _logger.critical(f"{NEON['CRITICAL']}CRITICAL: Required config rune '{key}' not found. Pyrmethus cannot proceed.{NEON['RESET']}")
                raise ValueError(f"Required environment variable '{key}' not set.")
            _logger.debug(f"{color}Config Rune {NEON['VALUE']}'{key}'{color}: Not Found. Using Default: '{NEON['VALUE']}{default}{color}'{NEON['RESET']}")
            value_to_cast = default
            source = "Default"
        else:
            _logger.debug(f"{color}Config Rune {NEON['VALUE']}'{key}'{color}: Found Env Value: '{NEON['VALUE']}{display_value}{color}'{NEON['RESET']}")
            value_to_cast = value_str

        if value_to_cast is None: # Can happen if default is None
            if required: # Should have been caught above if value_str was None and required
                 _logger.critical(f"{NEON['CRITICAL']}CRITICAL: Required config rune '{key}' resolved to None after env/default check.{NEON['RESET']}")
                 raise ValueError(f"Required environment variable '{key}' resolved to None.")
            _logger.debug(f"{color}Config Rune {NEON['VALUE']}'{key}'{color}: Final value is None (not required).{NEON['RESET']}")
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
                _logger.warning(f"{NEON['WARNING']}Unsupported cast_type '{cast_type.__name__}' for key '{key}'. Returning raw string.{NEON['RESET']}")
                final_value = raw_value_str_for_cast
        except (ValueError, TypeError, InvalidOperation) as e:
            _logger.error(f"{NEON['ERROR']}Invalid type/value for '{key}': '{value_to_cast}' (Source: {source}). Expected {cast_type.__name__}. Err: {e}. Using Default: '{default}'.{NEON['RESET']}")
            if default is None:
                if required:
                    _logger.critical(f"{NEON['CRITICAL']}CRITICAL: Failed cast for required key '{key}', default is None.{NEON['RESET']}")
                    raise ValueError(f"Required env var '{key}' failed casting, no valid default.")
                else: # Optional key, default is None, cast failed
                    _logger.warning(f"{NEON['WARNING']}Cast fail for optional '{key}', default is None. Final: None{NEON['RESET']}")
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
                    _logger.warning(f"{NEON['WARNING']}Used casted default for {key}: '{NEON['VALUE']}{final_value}{NEON['WARNING']}'{NEON['RESET']}")
                except (ValueError, TypeError, InvalidOperation) as e_default:
                    _logger.critical(f"{NEON['CRITICAL']}CRITICAL: Cast fail for value AND default for '{key}'. Original: '{value_to_cast}', Default: '{default}'. Err: {e_default}{NEON['RESET']}")
                    raise ValueError(f"Config error: Cannot cast value or default for '{key}'.")
        
        display_final_value = "********" if secret else final_value
        _logger.debug(f"{color}Final value for {NEON['VALUE']}'{key}'{color}: {NEON['VALUE']}{display_final_value}{color} (Type: {type(final_value).__name__}, Source: {source}){NEON['RESET']}")
        return final_value

    def _validate_parameters(self) -> None:
        _logger = logging.getLogger("ConfigModule._validate")
        errors = []
        if not (0 < self.risk_per_trade_percentage < 1): errors.append("RISK_PER_TRADE_PERCENTAGE must be between 0 and 1.")
        if self.leverage < 1: errors.append("LEVERAGE must be at least 1.")
        # Add more validations for new params as needed
        if self.profit_momentum_window < 1: errors.append("PROFIT_MOMENTUM_WINDOW must be >= 1.")
        if self.whipsaw_max_trades_in_period < 1: errors.append("WHIPSAW_MAX_TRADES_IN_PERIOD must be >= 1.")
        if self.signal_persistence_candles < 1: errors.append("SIGNAL_PERSISTENCE_CANDLES must be >= 1.")
        if self.st_max_entry_distance_atr_multiplier is not None and self.st_max_entry_distance_atr_multiplier < 0: errors.append("ST_MAX_ENTRY_DISTANCE_ATR_MULTIPLIER cannot be negative.")
        if errors:
            error_message = f"Configuration spellcrafting failed with {len(errors)} flaws:\n" + "\n".join([f"  - {e}" for e in errors])
            _logger.critical(f"{NEON['CRITICAL']}{error_message}{NEON['RESET']}")
            raise ValueError(error_message)

# --- Logger Setup ---
# ... (existing logger setup, no changes needed for this integration)
LOGGING_LEVEL: int = (logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO)
logs_dir = "logs"; os.makedirs(logs_dir, exist_ok=True)
log_file_name = f"{logs_dir}/pyrmethus_spell_v{PYRMETHUS_VERSION.replace('.', '')}_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=LOGGING_LEVEL, format="%(asctime)s [%(levelname)-8s] %(name)-30s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file_name)])
logger: logging.Logger = logging.getLogger("PyrmethusCore")
SUCCESS_LEVEL: int = 25; logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")
def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(SUCCESS_LEVEL): self._log(SUCCESS_LEVEL, message, args, **kwargs) # type: ignore[attr-defined]
logging.Logger.success = log_success # type: ignore[attr-defined]
if sys.stdout.isatty():
    stream_handler = next((h for h in logging.getLogger().handlers if isinstance(h, logging.StreamHandler)), None)
    if stream_handler:
        colored_formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)-8s] %(name)-30s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        stream_handler.setFormatter(colored_formatter)
    for level, color_code in [(logging.DEBUG, NEON['DEBUG']), (logging.INFO, NEON['INFO']), (SUCCESS_LEVEL, NEON['SUCCESS']),
                              (logging.WARNING, NEON['WARNING']), (logging.ERROR, NEON['ERROR']), (logging.CRITICAL, NEON['CRITICAL'])]:
        level_name = logging.getLevelName(level)
        if '\033' in level_name: # Remove existing color codes if re-running
            base_name_parts = level_name.split('\033'); original_name = ""
            for part in base_name_parts:
                if not (part.startswith('[') and part.endswith('m')): original_name = part; break
            if not original_name: original_name = level_name.split('m')[-1] if 'm' in level_name else level_name
            level_name = original_name.strip()
        logging.addLevelName(level, f"{color_code}{level_name.ljust(8)}{NEON['RESET']}")

# --- Global Objects & State Variables ---
try: CONFIG = Config()
except ValueError as config_error:
    logging.getLogger("PyrmethusCore").critical(f"{NEON['CRITICAL']}Configuration loading failed. Error: {config_error}{NEON['RESET']}")
    if CONFIG and hasattr(CONFIG, 'send_notification_method'): CONFIG.send_notification_method("Pyrmethus Startup Failure", f"Config Error: {str(config_error)[:200]}")
    elif 'termux-api' in os.getenv('PATH', ''): send_termux_notification("Pyrmethus Startup Failure", f"Config Error: {str(config_error)[:200]}")
    sys.exit(1)
except Exception as general_config_error: # Catch any other exception during Config init
    logging.getLogger("PyrmethusCore").critical(f"{NEON['CRITICAL']}Unexpected critical error during configuration: {general_config_error}{NEON['RESET']}")
    logging.getLogger("PyrmethusCore").debug(traceback.format_exc())
    if CONFIG and hasattr(CONFIG, 'send_notification_method'): CONFIG.send_notification_method("Pyrmethus Critical Failure", f"Unexpected Config Error: {str(general_config_error)[:200]}")
    elif 'termux-api' in os.getenv('PATH', ''): send_termux_notification("Pyrmethus Critical Failure", f"Unexpected Config Error: {str(general_config_error)[:200]}")
    sys.exit(1)

# Snippet 1.3: Whipsaw Cooldown state
trade_timestamps_for_whipsaw = deque(maxlen=CONFIG.whipsaw_max_trades_in_period)
whipsaw_cooldown_active_until: float = 0.0
# Snippet 1.5: Stateful Signal Confirmation state
persistent_signal_counter = {"long": 0, "short": 0}
last_signal_type: Optional[str] = None
# Snippet 2.1: No-Trade Zones state
previous_day_high: Optional[Decimal] = None
previous_day_low: Optional[Decimal] = None
last_key_level_update_day: Optional[int] = None
# Snippet 3.5: Trend Contradiction Cooldown state
contradiction_cooldown_active_until: float = 0.0
# Consecutive Loss Cooldown (general snippet)
consecutive_loss_cooldown_active_until: float = 0.0

# --- Trading Strategy Abstract Base Class & Implementations ---
class TradingStrategy(ABC):
    def __init__(self, config: Config, df_columns: Optional[List[str]] = None):
        self.config = config; self.logger = logging.getLogger(f"Strategy.{self.__class__.__name__}")
        self.required_columns = df_columns if df_columns else []
        self.logger.info(f"{NEON['STRATEGY']}Strategy Form '{self.__class__.__name__}' materializing...{NEON['RESET']}")
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, latest_close: Optional[Decimal] = None, latest_atr: Optional[Decimal] = None) -> Dict[str, Any]: pass # Added latest_close, latest_atr
    def _validate_df(self, df: pd.DataFrame, min_rows: int = 2) -> bool: # ... (existing logic)
        if df is None or df.empty or len(df) < min_rows:
            self.logger.debug(f"Insufficient market whispers (Rows: {len(df) if df is not None else 0}, Min Required: {min_rows}).")
            return False
        if self.required_columns and not all(col in df.columns for col in self.required_columns):
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            self.logger.warning(f"{NEON['WARNING']}Market scroll missing required runes: {NEON['VALUE']}{missing_cols}{NEON['WARNING']}.{NEON['RESET']}")
            return False
        if self.required_columns and not df.empty:
            last_row_values = df.iloc[-1][self.required_columns]
            if last_row_values.isnull().any():
                nan_cols_last_row = last_row_values[last_row_values.isnull()].index.tolist()
                self.logger.debug(f"Faint runes (NaNs) in last candle for critical columns: {nan_cols_last_row}.")
        return True
    def _get_default_signals(self) -> Dict[str, Any]:
        return {"enter_long": False, "enter_short": False, "exit_long": False, "exit_short": False, "exit_reason": "Default Signal"}

class DualSupertrendMomentumStrategy(TradingStrategy):
    def __init__(self, config: Config):
        super().__init__(config, df_columns=["st_long_flip", "st_short_flip", "confirm_trend", "momentum"])
    def generate_signals(self, df: pd.DataFrame, latest_close: Optional[Decimal] = None, latest_atr: Optional[Decimal] = None) -> Dict[str, Any]:
        signals = self._get_default_signals()
        min_rows_needed = max(self.config.st_atr_length, self.config.confirm_st_atr_length, self.config.momentum_period, self.config.confirm_st_stability_lookback) + 10
        if not self._validate_df(df, min_rows=min_rows_needed): return signals

        last = df.iloc[-1]
        primary_long_flip = last.get("st_long_flip", False)
        primary_short_flip = last.get("st_short_flip", False)
        confirm_is_up_current = last.get("confirm_trend", pd.NA) # Current state
        momentum_val = safe_decimal_conversion(last.get("momentum"), pd.NA)

        # Snippet 4.1: Fade the Weaker Signal
        if primary_long_flip and primary_short_flip:
            self.logger.warning(f"{NEON['WARNING']}Conflicting primary Supertrend flips on the same candle. Attempting to resolve...{NEON['RESET']}")
            if confirm_is_up_current is True and (momentum_val is not pd.NA and momentum_val > 0):
                primary_short_flip = False; self.logger.info("Resolution: Prioritizing LONG flip (Confirm ST Up & Positive Momentum).")
            elif confirm_is_up_current is False and (momentum_val is not pd.NA and momentum_val < 0):
                primary_long_flip = False; self.logger.info("Resolution: Prioritizing SHORT flip (Confirm ST Down & Negative Momentum).")
            else:
                primary_long_flip = False; primary_short_flip = False; self.logger.warning("Resolution: Conflicting flips ambiguous. No primary signal.")
        
        # Snippet 3.1: Confirmation Supertrend Stability Lookback
        stable_confirm_trend = pd.NA
        if self.config.confirm_st_stability_lookback <= 1:
            stable_confirm_trend = confirm_is_up_current
        elif 'confirm_trend' in df.columns and len(df) >= self.config.confirm_st_stability_lookback:
            recent_confirm_trends = df['confirm_trend'].iloc[-self.config.confirm_st_stability_lookback:]
            if confirm_is_up_current is True and all(trend is True for trend in recent_confirm_trends): stable_confirm_trend = True
            elif confirm_is_up_current is False and all(trend is False for trend in recent_confirm_trends): stable_confirm_trend = False
            else: stable_confirm_trend = pd.NA # Mixed or not enough matching
        
        if pd.isna(stable_confirm_trend) or pd.isna(momentum_val):
            self.logger.debug(f"Stable Confirm ST ({_format_for_log(stable_confirm_trend, True)}) or Mom ({_format_for_log(momentum_val)}) is NA. No signal.")
            return signals

        # Snippet 3.4: Min Distance Between Primary ST Line and Price for Entry
        price_proximity_ok = True
        if self.config.st_max_entry_distance_atr_multiplier is not None and latest_atr is not None and latest_atr > 0 and latest_close is not None:
            max_allowed_distance = latest_atr * self.config.st_max_entry_distance_atr_multiplier
            st_p_base = f"ST_{self.config.st_atr_length}_{self.config.st_multiplier}"
            
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
           momentum_val > self.config.momentum_threshold and momentum_val > 0 and price_proximity_ok: # Added momentum_val > 0
            signals["enter_long"] = True
            self.logger.info(f"{NEON['SIDE_LONG']}DualST+Mom Signal: LONG Entry - ST Flip, Stable Confirm Up, Positive Mom ({_format_for_log(momentum_val)}), Prox OK.{NEON['RESET']}")
        elif primary_short_flip and stable_confirm_trend is False and \
             momentum_val < -self.config.momentum_threshold and momentum_val < 0 and price_proximity_ok: # Added momentum_val < 0
            signals["enter_short"] = True
            self.logger.info(f"{NEON['SIDE_SHORT']}DualST+Mom Signal: SHORT Entry - ST Flip, Stable Confirm Down, Negative Mom ({_format_for_log(momentum_val)}), Prox OK.{NEON['RESET']}")

        if primary_short_flip: signals["exit_long"] = True; signals["exit_reason"] = "Primary SuperTrend Flipped Short"
        if primary_long_flip: signals["exit_short"] = True; signals["exit_reason"] = "Primary SuperTrend Flipped Long"
        return signals

class EhlersFisherStrategy(TradingStrategy):
    def __init__(self, config: Config):
        super().__init__(config, df_columns=["ehlers_fisher", "ehlers_signal"])
    def generate_signals(self, df: pd.DataFrame, latest_close: Optional[Decimal] = None, latest_atr: Optional[Decimal] = None) -> Dict[str, Any]:
        signals = self._get_default_signals()
        min_rows_needed = self.config.ehlers_fisher_length + self.config.ehlers_fisher_signal_length + 5
        if not self._validate_df(df, min_rows=min_rows_needed) or len(df) < 2: return signals

        last = df.iloc[-1]; prev = df.iloc[-2]
        fisher_now = safe_decimal_conversion(last.get("ehlers_fisher"), pd.NA)
        signal_now = safe_decimal_conversion(last.get("ehlers_signal"), pd.NA)
        fisher_prev = safe_decimal_conversion(prev.get("ehlers_fisher"), pd.NA)
        signal_prev = safe_decimal_conversion(prev.get("ehlers_signal"), pd.NA)

        if pd.isna(fisher_now) or pd.isna(signal_now) or pd.isna(fisher_prev) or pd.isna(signal_prev):
            self.logger.debug(f"Ehlers Fisher or Signal rune is NA. No signal.")
            return signals

        # Snippet 3.3: Ehlers Fisher "Extreme Zone" Avoidance
        is_fisher_extreme = False
        if (fisher_now > self.config.ehlers_fisher_extreme_threshold_positive or \
            fisher_now < self.config.ehlers_fisher_extreme_threshold_negative):
            is_fisher_extreme = True
        
        # Entry signals
        if not is_fisher_extreme:
            if fisher_prev <= signal_prev and fisher_now > signal_now:
                signals["enter_long"] = True
                self.logger.info(f"{NEON['SIDE_LONG']}EhlersFisher Signal: LONG Entry - Fisher ({_format_for_log(fisher_now)}) crossed ABOVE Signal ({_format_for_log(signal_now)}).{NEON['RESET']}")
            elif fisher_prev >= signal_prev and fisher_now < signal_now:
                signals["enter_short"] = True
                self.logger.info(f"{NEON['SIDE_SHORT']}EhlersFisher Signal: SHORT Entry - Fisher ({_format_for_log(fisher_now)}) crossed BELOW Signal ({_format_for_log(signal_now)}).{NEON['RESET']}")
        elif (fisher_prev <= signal_prev and fisher_now > signal_now) or \
             (fisher_prev >= signal_prev and fisher_now < signal_now): # Log if crossover ignored
            self.logger.info(f"EhlersFisher: Crossover signal ignored due to Fisher in extreme zone ({_format_for_log(fisher_now)}).")

        # Exit signals
        if fisher_prev >= signal_prev and fisher_now < signal_now:
            signals["exit_long"] = True; signals["exit_reason"] = "Ehlers Fisher crossed BELOW Signal"
        elif fisher_prev <= signal_prev and fisher_now > signal_now:
            signals["exit_short"] = True; signals["exit_reason"] = "Ehlers Fisher crossed ABOVE Signal"
        return signals

strategy_map: Dict[StrategyName, Type[TradingStrategy]] = { StrategyName.DUAL_SUPERTREND_MOMENTUM: DualSupertrendMomentumStrategy, StrategyName.EHLERS_FISHER: EhlersFisherStrategy }
StrategyClass = strategy_map.get(CONFIG.strategy_name)
if StrategyClass: CONFIG.strategy_instance = StrategyClass(CONFIG); logger.success(f"{NEON['SUCCESS']}Strategy '{NEON['STRATEGY']}{CONFIG.strategy_name.value}{NEON['SUCCESS']}' invoked.{NEON['RESET']}") # type: ignore [attr-defined]
else:
    err_msg = f"Failed to init strategy '{CONFIG.strategy_name.value}'. Unknown spell form."
    logger.critical(f"{NEON['CRITICAL']}{err_msg}{NEON['RESET']}")
    CONFIG.send_notification_method("Pyrmethus Critical Error", err_msg); sys.exit(1)

# --- Trade Metrics Tracking ---
class TradeMetrics:
    def __init__(self):
        self.trades: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("TradeMetrics")
        self.initial_equity: Optional[Decimal] = None
        self.daily_start_equity: Optional[Decimal] = None
        self.last_daily_reset_day: Optional[int] = None
        self.consecutive_losses: int = 0 # For Snippet: Consecutive Loss Limiter
        # For Snippet 4.3: Daily Max Trades Rest
        self.daily_trade_entry_count: int = 0
        self.last_daily_trade_count_reset_day: Optional[int] = None
        self.daily_trades_rest_active_until: float = 0.0 # Timestamp
        self.logger.info(f"{NEON['INFO']}TradeMetrics Ledger opened.{NEON['RESET']}")

    def set_initial_equity(self, equity: Decimal):
        if self.initial_equity is None: self.initial_equity = equity; self.logger.info(f"{NEON['INFO']}Initial Session Equity: {NEON['VALUE']}{equity:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}")
        today = datetime.now(pytz.utc).day
        if self.last_daily_reset_day != today or self.daily_start_equity is None:
            self.daily_start_equity = equity; self.last_daily_reset_day = today
            self.logger.info(f"{NEON['INFO']}Daily Equity Ward reset. Dawn Equity: {NEON['VALUE']}{equity:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}")
        # Reset daily trade entry count for Snippet 4.3
        if self.last_daily_trade_count_reset_day != today:
            self.daily_trade_entry_count = 0
            self.last_daily_trade_count_reset_day = today
            self.logger.info(f"{NEON['INFO']}Daily trade entry count reset to 0.{NEON['RESET']}")
            if self.daily_trades_rest_active_until > 0 and time.time() > self.daily_trades_rest_active_until : # Reset cooldown if it was for previous day
                 self.daily_trades_rest_active_until = 0.0

    def check_drawdown(self, current_equity: Decimal) -> Tuple[bool, str]: # ... (existing logic)
        if not CONFIG.enable_max_drawdown_stop or self.daily_start_equity is None or self.daily_start_equity <= 0: return False, ""
        drawdown = self.daily_start_equity - current_equity
        drawdown_pct = (drawdown / self.daily_start_equity) if self.daily_start_equity > 0 else Decimal(0)
        if drawdown_pct >= CONFIG.max_drawdown_percent:
            reason = f"Max daily drawdown breached ({NEON['PNL_NEG']}{drawdown_pct:.2%}{NEON['RESET']} >= {NEON['VALUE']}{CONFIG.max_drawdown_percent:.2%}{NEON['RESET']})"
            self.logger.warning(f"{NEON['WARNING']}{reason}.{NEON['RESET']}")
            CONFIG.send_notification_method("Pyrmethus: Max Drawdown Hit!", f"Drawdown: {drawdown_pct:.2%}. Trading halted.")
            return True, reason
        return False, ""

    def log_trade(self, symbol: str, side: str, entry_price: Decimal, exit_price: Decimal, qty: Decimal,
                  entry_time_ms: int, exit_time_ms: int, reason: str, part_id: str, pnl_str: str,
                  scale_order_id: Optional[str] = None, mae: Optional[Decimal] = None, mfe: Optional[Decimal] = None, is_entry: bool = False):
        # ... (existing log_trade logic) ...
        if not all([isinstance(entry_price, Decimal) and entry_price > 0, isinstance(exit_price, Decimal) and exit_price > 0,
                    isinstance(qty, Decimal) and qty > 0, isinstance(entry_time_ms, int) and entry_time_ms > 0,
                    isinstance(exit_time_ms, int) and exit_time_ms > 0]):
            self.logger.warning(f"{NEON['WARNING']}Trade log skipped due to flawed parameters for Part ID: {part_id}.{NEON['RESET']}")
            return

        profit = safe_decimal_conversion(pnl_str, Decimal(0))
        # Update consecutive losses (Snippet: Consecutive Loss Limiter)
        if profit <= 0: self.consecutive_losses += 1
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
        self.logger.success(f"{NEON['HEADING']}Trade Chronicle ({trade_type}:{part_id}): {side.upper()} {NEON['QTY']}{qty}{NEON['RESET']} {symbol.split('/')[0]} | P/L: {pnl_color}{profit:.2f} {CONFIG.usdt_symbol}{NEON['RESET']} | Reason: {reason}") # type: ignore [attr-defined]

    def increment_daily_trade_entry_count(self): # For Snippet 4.3
        self.daily_trade_entry_count +=1
        self.logger.info(f"Daily entry trade count incremented to: {self.daily_trade_entry_count}")

    def summary(self) -> str: # ... (existing summary logic)
        if not self.trades: return f"{NEON['INFO']}The Grand Ledger is empty.{NEON['RESET']}"
        total_trades = len(self.trades); profits = [Decimal(t["profit_str"]) for t in self.trades]
        wins = sum(1 for p in profits if p > 0); losses = sum(1 for p in profits if p < 0)
        breakeven = total_trades - wins - losses
        win_rate = (Decimal(wins) / Decimal(total_trades)) * Decimal(100) if total_trades > 0 else Decimal(0)
        total_profit = sum(profits); avg_profit = total_profit / Decimal(total_trades) if total_trades > 0 else Decimal(0)
        summary_str = (
            f"\n{NEON['HEADING']}--- Pyrmethus Trade Metrics Summary (v{PYRMETHUS_VERSION}) ---{NEON['RESET']}\n"
            f"Total Trade Parts: {NEON['VALUE']}{total_trades}{NEON['RESET']}\n"
            f"  Wins: {NEON['PNL_POS']}{wins}{NEON['RESET']}, Losses: {NEON['PNL_NEG']}{losses}{NEON['RESET']}, Breakeven: {NEON['PNL_ZERO']}{breakeven}{NEON['RESET']}\n"
            f"Win Rate: {NEON['VALUE']}{win_rate:.2f}%{NEON['RESET']}\n"
            f"Total P/L: {(NEON['PNL_POS'] if total_profit > 0 else (NEON['PNL_NEG'] if total_profit < 0 else NEON['PNL_ZERO']))}{total_profit:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}\n"
            f"Avg P/L per Part: {(NEON['PNL_POS'] if avg_profit > 0 else (NEON['PNL_NEG'] if avg_profit < 0 else NEON['PNL_ZERO']))}{avg_profit:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}\n"
        )
        current_equity_approx = Decimal(0)
        if self.initial_equity is not None:
            current_equity_approx = self.initial_equity + total_profit
            overall_pnl_pct = (total_profit / self.initial_equity) * 100 if self.initial_equity > Decimal(0) else Decimal(0)
            summary_str += f"Initial Session Treasury: {NEON['VALUE']}{self.initial_equity:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}\n"
            summary_str += f"Approx. Current Treasury: {NEON['VALUE']}{current_equity_approx:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}\n"
            summary_str += f"Overall Session P/L %: {(NEON['PNL_POS'] if overall_pnl_pct > 0 else (NEON['PNL_NEG'] if overall_pnl_pct < 0 else NEON['PNL_ZERO']))}{overall_pnl_pct:.2f}%{NEON['RESET']}\n"
        if self.daily_start_equity is not None:
            daily_pnl_base = current_equity_approx if self.initial_equity is not None else self.daily_start_equity + total_profit
            daily_pnl = daily_pnl_base - self.daily_start_equity
            daily_pnl_pct = (daily_pnl / self.daily_start_equity) * 100 if self.daily_start_equity > Decimal(0) else Decimal(0)
            summary_str += f"Daily Start Treasury: {NEON['VALUE']}{self.daily_start_equity:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}\n"
            summary_str += f"Approx. Daily P/L: {(NEON['PNL_POS'] if daily_pnl > 0 else (NEON['PNL_NEG'] if daily_pnl < 0 else NEON['PNL_ZERO']))}{daily_pnl:.2f} {CONFIG.usdt_symbol} ({daily_pnl_pct:.2f}%){NEON['RESET']}\n"
        summary_str += f"Consecutive Losses: {NEON['VALUE']}{self.consecutive_losses}{NEON['RESET']}\n"
        summary_str += f"Daily Entries Made: {NEON['VALUE']}{self.daily_trade_entry_count}{NEON['RESET']}\n"
        summary_str += f"{NEON['HEADING']}--- End of Ledger Reading ---{NEON['RESET']}"
        self.logger.info(summary_str); return summary_str

    def get_serializable_trades(self) -> List[Dict[str, Any]]: return self.trades
    def load_trades_from_list(self, trades_list: List[Dict[str, Any]]):
        self.trades = []; 
        for trade_data in trades_list: self.trades.append(trade_data)
        self.logger.info(f"{NEON['INFO']}TradeMetrics: Re-inked {len(self.trades)} sagas.{NEON['RESET']}")

trade_metrics = TradeMetrics()
_active_trade_parts: List[Dict[str, Any]] = []
_last_heartbeat_save_time: float = 0.0

# --- State Persistence Functions ---
def save_persistent_state(force_heartbeat: bool = False) -> None:
    global _active_trade_parts, trade_metrics, _last_heartbeat_save_time
    global whipsaw_cooldown_active_until, persistent_signal_counter, last_signal_type
    global previous_day_high, previous_day_low, last_key_level_update_day
    global contradiction_cooldown_active_until, consecutive_loss_cooldown_active_until

    now = time.time()
    should_save = force_heartbeat or \
        (_active_trade_parts and (now - _last_heartbeat_save_time > HEARTBEAT_INTERVAL_SECONDS)) or \
        (not _active_trade_parts and (now - _last_heartbeat_save_time > HEARTBEAT_INTERVAL_SECONDS * 5))
    if not should_save: return

    logger.debug(f"{NEON['COMMENT']}# Phoenix Feather scribing memories... (Force: {force_heartbeat}){NEON['RESET']}")
    try:
        serializable_active_parts = []
        for part in _active_trade_parts:
            p_copy = part.copy()
            for key, value in p_copy.items():
                if isinstance(value, Decimal): p_copy[key] = str(value)
                elif isinstance(value, deque): p_copy[key] = list(value) # Serialize deques
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
            "config_symbol": CONFIG.symbol, "config_strategy": CONFIG.strategy_name.value,
            "initial_equity_str": str(trade_metrics.initial_equity) if trade_metrics.initial_equity is not None else None,
            "daily_start_equity_str": str(trade_metrics.daily_start_equity) if trade_metrics.daily_start_equity is not None else None,
            "last_daily_reset_day": trade_metrics.last_daily_reset_day,
            # New global states
            "whipsaw_cooldown_active_until": whipsaw_cooldown_active_until,
            "trade_timestamps_for_whipsaw": list(trade_timestamps_for_whipsaw), # Serialize deque
            "persistent_signal_counter": persistent_signal_counter,
            "last_signal_type": last_signal_type,
            "previous_day_high_str": str(previous_day_high) if previous_day_high else None,
            "previous_day_low_str": str(previous_day_low) if previous_day_low else None,
            "last_key_level_update_day": last_key_level_update_day,
            "contradiction_cooldown_active_until": contradiction_cooldown_active_until,
            "consecutive_loss_cooldown_active_until": consecutive_loss_cooldown_active_until,
        }
        temp_file_path = STATE_FILE_PATH + ".tmp_scroll"
        with open(temp_file_path, 'w') as f: json.dump(state_data, f, indent=4)
        os.replace(temp_file_path, STATE_FILE_PATH)
        _last_heartbeat_save_time = now
        log_level = logging.INFO if force_heartbeat or _active_trade_parts else logging.DEBUG
        logger.log(log_level, f"{NEON['SUCCESS']}Phoenix Feather: State memories scribed. Active parts: {len(_active_trade_parts)}.{NEON['RESET']}")
    except Exception as e:
        logger.error(f"{NEON['ERROR']}Phoenix Feather Error: Failed to scribe state: {e}{NEON['RESET']}"); logger.debug(traceback.format_exc())

def load_persistent_state() -> bool:
    global _active_trade_parts, trade_metrics, _last_heartbeat_save_time
    global whipsaw_cooldown_active_until, trade_timestamps_for_whipsaw, persistent_signal_counter, last_signal_type
    global previous_day_high, previous_day_low, last_key_level_update_day
    global contradiction_cooldown_active_until, consecutive_loss_cooldown_active_until
    
    logger.info(f"{NEON['COMMENT']}# Phoenix Feather seeks past memories from '{STATE_FILE_PATH}'...{NEON['RESET']}")
    if not os.path.exists(STATE_FILE_PATH): logger.info(f"{NEON['INFO']}Phoenix Feather: No previous scroll.{NEON['RESET']}"); return False
    try:
        with open(STATE_FILE_PATH, 'r') as f: state_data = json.load(f)
        saved_version = state_data.get("pyrmethus_version", "unknown")
        if saved_version != PYRMETHUS_VERSION:
            logger.warning(f"{NEON['WARNING']}Phoenix Feather: Scroll version '{saved_version}' differs from current '{PYRMETHUS_VERSION}'. Caution.{NEON['RESET']}")
        if state_data.get("config_symbol") != CONFIG.symbol or state_data.get("config_strategy") != CONFIG.strategy_name.value:
            logger.warning(f"{NEON['WARNING']}Phoenix Scroll sigils mismatch! Current: {CONFIG.symbol}/{CONFIG.strategy_name.value}, Scroll: {state_data.get('config_symbol')}/{state_data.get('config_strategy')}. Ignoring scroll.{NEON['RESET']}")
            # Archive mismatched state file logic could be added here
            return False

        _active_trade_parts.clear()
        loaded_parts_raw = state_data.get("active_trade_parts", [])
        # Keys that might need deque conversion if they were saved as lists
        deque_keys_in_part = { 
            "recent_pnls": CONFIG.profit_momentum_window,
            "adverse_candle_closes": CONFIG.last_chance_consecutive_adverse_candles
        }
        decimal_keys_in_part = ["entry_price", "qty", "sl_price", "initial_usdt_value", "atr_at_entry", 
                                "entry_fisher_value", "entry_signal_value"] # Added Ehlers state

        for part_data_str_values in loaded_parts_raw:
            restored_part = {}
            valid_part = True
            for k, v_loaded in part_data_str_values.items():
                if k in decimal_keys_in_part:
                    restored_part[k] = safe_decimal_conversion(v_loaded, None)
                    if restored_part[k] is None and v_loaded is not None : logger.warning(f"Failed to restore Decimal {k} for part {part_data_str_values.get('part_id')}")
                elif k in deque_keys_in_part:
                    if isinstance(v_loaded, list): restored_part[k] = deque(v_loaded, maxlen=deque_keys_in_part[k])
                    else: restored_part[k] = deque(maxlen=deque_keys_in_part[k]); logger.warning(f"Expected list for deque {k}, got {type(v_loaded)}")
                elif k == "entry_time_ms": # ... (existing robust entry_time_ms loading)
                    if isinstance(v_loaded, (int, float)): restored_part[k] = int(v_loaded)
                    elif isinstance(v_loaded, str): # Attempt ISO format then direct int
                        try: dt_obj = datetime.fromisoformat(v_loaded.replace("Z", "+00:00")); restored_part[k] = int(dt_obj.timestamp() * 1000)
                        except ValueError:
                            try: restored_part[k] = int(v_loaded)
                            except ValueError: logger.warning(f"Malformed entry_time_ms '{v_loaded}'. Skipping part."); valid_part = False; break
                    else: logger.warning(f"Unexpected type for entry_time_ms '{v_loaded}'. Skipping part."); valid_part = False; break
                else: restored_part[k] = v_loaded
            if valid_part: _active_trade_parts.append(restored_part)

        trade_metrics.load_trades_from_list(state_data.get("trade_metrics_trades", []))
        trade_metrics.consecutive_losses = state_data.get("trade_metrics_consecutive_losses", 0)
        trade_metrics.daily_trade_entry_count = state_data.get("trade_metrics_daily_trade_entry_count", 0)
        trade_metrics.last_daily_trade_count_reset_day = state_data.get("trade_metrics_last_daily_trade_count_reset_day")
        trade_metrics.daily_trades_rest_active_until = state_data.get("trade_metrics_daily_trades_rest_active_until", 0.0)

        initial_equity_str = state_data.get("initial_equity_str")
        if initial_equity_str and initial_equity_str.lower() != 'none': trade_metrics.initial_equity = safe_decimal_conversion(initial_equity_str, None)
        daily_equity_str = state_data.get("daily_start_equity_str")
        if daily_equity_str and daily_equity_str.lower() != 'none': trade_metrics.daily_start_equity = safe_decimal_conversion(daily_equity_str, None)
        trade_metrics.last_daily_reset_day = state_data.get("last_daily_reset_day")
        if trade_metrics.last_daily_reset_day is not None:
            try: trade_metrics.last_daily_reset_day = int(trade_metrics.last_daily_reset_day)
            except ValueError: trade_metrics.last_daily_reset_day = None; logger.warning("Malformed last_daily_reset_day.")

        # Load new global states
        whipsaw_cooldown_active_until = state_data.get("whipsaw_cooldown_active_until", 0.0)
        loaded_whipsaw_ts = state_data.get("trade_timestamps_for_whipsaw", [])
        trade_timestamps_for_whipsaw = deque(loaded_whipsaw_ts, maxlen=CONFIG.whipsaw_max_trades_in_period)
        persistent_signal_counter = state_data.get("persistent_signal_counter", {"long": 0, "short": 0})
        last_signal_type = state_data.get("last_signal_type")
        prev_high_str = state_data.get("previous_day_high_str"); previous_day_high = safe_decimal_conversion(prev_high_str, None) if prev_high_str else None
        prev_low_str = state_data.get("previous_day_low_str"); previous_day_low = safe_decimal_conversion(prev_low_str, None) if prev_low_str else None
        last_key_level_update_day = state_data.get("last_key_level_update_day")
        contradiction_cooldown_active_until = state_data.get("contradiction_cooldown_active_until", 0.0)
        consecutive_loss_cooldown_active_until = state_data.get("consecutive_loss_cooldown_active_until", 0.0)
        
        logger.success(f"{NEON['SUCCESS']}Phoenix Feather: Memories reawakened! Active Parts: {len(_active_trade_parts)}, Trades: {len(trade_metrics.trades)}{NEON['RESET']}") # type: ignore [attr-defined]
        return True
    except json.JSONDecodeError as e:
        logger.error(f"{NEON['ERROR']}Phoenix Feather: Scroll '{STATE_FILE_PATH}' corrupted: {e}. Starting fresh.{NEON['RESET']}")
        # Archive corrupted file logic could be added here
        return False
    except Exception as e:
        logger.error(f"{NEON['ERROR']}Phoenix Feather Error: Unexpected chaos during reawakening: {e}{NEON['RESET']}"); logger.debug(traceback.format_exc())
        return False

# --- Indicator Calculation ---
def calculate_all_indicators(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    logger.debug(f"{NEON['COMMENT']}# Transmuting market data with indicator alchemy...{NEON['RESET']}")
    if df.empty: logger.warning(f"{NEON['WARNING']}Market data scroll empty.{NEON['RESET']}"); return df.copy()
    for col in ['close', 'high', 'low', 'volume']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        else: logger.error(f"{NEON['ERROR']}Essential rune '{col}' missing.{NEON['RESET']}")

    if config.strategy_name == StrategyName.DUAL_SUPERTREND_MOMENTUM:
        st_p_base = f"ST_{config.st_atr_length}_{config.st_multiplier}"
        st_c_base = f"CONFIRM_ST_{config.confirm_st_atr_length}_{config.confirm_st_multiplier}"
        df.ta.supertrend(length=config.st_atr_length, multiplier=float(config.st_multiplier), append=True, col_names=(st_p_base, f"{st_p_base}d", f"{st_p_base}l", f"{st_p_base}s"))
        df.ta.supertrend(length=config.confirm_st_atr_length, multiplier=float(config.confirm_st_multiplier), append=True, col_names=(st_c_base, f"{st_c_base}d", f"{st_c_base}l", f"{st_c_base}s"))
        if f"{st_p_base}d" in df.columns:
            df["st_trend_up"] = (df[f"{st_p_base}d"] == 1)
            df["st_long_flip"] = (df["st_trend_up"]) & (df["st_trend_up"].shift(1) == False)
            df["st_short_flip"] = (df["st_trend_up"] == False) & (df["st_trend_up"].shift(1) == True)
        else: df["st_trend_up"], df["st_long_flip"], df["st_short_flip"] = pd.NA, False, False
        if f"{st_c_base}d" in df.columns:
            df["confirm_trend"] = df[f"{st_c_base}d"].apply(lambda x: True if x == 1 else (False if x == -1 else pd.NA))
        else: df["confirm_trend"] = pd.NA
        if 'close' in df.columns and not df['close'].isnull().all(): df.ta.mom(length=config.momentum_period, append=True, col_names=("momentum",))
        else: df["momentum"] = pd.NA
    elif config.strategy_name == StrategyName.EHLERS_FISHER:
        if 'high' in df.columns and 'low' in df.columns and not df['high'].isnull().all() and not df['low'].isnull().all():
            df.ta.fisher(length=config.ehlers_fisher_length, signal=config.ehlers_fisher_signal_length, append=True, col_names=("ehlers_fisher", "ehlers_signal"))
        else: df["ehlers_fisher"], df["ehlers_signal"] = pd.NA, pd.NA

    atr_col_name = f"ATR_{config.atr_calculation_period}"
    if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns and \
       not df['high'].isnull().all() and not df['low'].isnull().all() and not df['close'].isnull().all():
        df.ta.atr(length=config.atr_calculation_period, append=True, col_names=(atr_col_name,))
    else: df[atr_col_name] = pd.NA
    
    # Snippet 4.5: Trap Filter indicators
    if config.enable_trap_filter:
        if 'high' in df.columns and not df['high'].isnull().all():
             df[f'rolling_high_{config.trap_filter_lookback_period}'] = df['high'].rolling(window=config.trap_filter_lookback_period, min_periods=1).max()
        else: df[f'rolling_high_{config.trap_filter_lookback_period}'] = pd.NA
        if 'low' in df.columns and not df['low'].isnull().all():
            df[f'rolling_low_{config.trap_filter_lookback_period}'] = df['low'].rolling(window=config.trap_filter_lookback_period, min_periods=1).min()
        else: df[f'rolling_low_{config.trap_filter_lookback_period}'] = pd.NA
    return df

# --- Exchange Interaction Primitives ---
@retry((ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection), tries=CONFIG.retry_count, delay=CONFIG.retry_delay_seconds, backoff=2, logger=logger)
def get_market_data(exchange: ccxt.Exchange, symbol: str, interval: str, limit: int = OHLCV_LIMIT) -> Optional[pd.DataFrame]: # ... (existing logic)
    logger.debug(f"{NEON['COMMENT']}# Summoning {limit} market whispers for {symbol} ({interval})...{NEON['RESET']}")
    try:
        params = {'category': 'linear'} if symbol.endswith(":USDT") else {}
        ohlcv = exchange.fetch_ohlcv(symbol, interval, limit=limit, params=params)
        if not ohlcv: logger.warning(f"{NEON['WARNING']}No OHLCV runes for {symbol}.{NEON['RESET']}"); return None
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = pd.to_numeric(df[col], errors='coerce')
        logger.debug(f"Summoned {len(df)} candles. Latest: {df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S %Z') if not df.empty else 'N/A'}")
        return df
    except ccxt.BadSymbol as e_bs: logger.critical(f"{NEON['CRITICAL']}Market realm rejects symbol '{symbol}': {e_bs}.{NEON['RESET']}"); return None
    except Exception as e: logger.error(f"{NEON['ERROR']}Error summoning market whispers: {e}{NEON['RESET']}"); logger.debug(traceback.format_exc()); return None

@retry((ccxt.NetworkError, ccxt.RequestTimeout), tries=CONFIG.retry_count, delay=CONFIG.retry_delay_seconds)
def fetch_account_balance(exchange: ccxt.Exchange, currency_code: str = CONFIG.usdt_symbol) -> Optional[Decimal]: # ... (existing logic)
    logger.debug(f"{NEON['COMMENT']}# Scrying treasury for {currency_code}...{NEON['RESET']}")
    try:
        params = {'accountType': 'UNIFIED'}
        if currency_code == "USDT": params['coin'] = currency_code
        balance_data = exchange.fetch_balance(params=params)
        total_balance_val = None
        if currency_code in balance_data: total_balance_val = balance_data[currency_code].get('total')
        if total_balance_val is None: total_balance_val = balance_data.get('total', {}).get(currency_code)
        if total_balance_val is None and 'info' in balance_data and 'result' in balance_data['info'] and 'list' in balance_data['info']['result']:
            if balance_data['info']['result']['list'] and 'totalEquity' in balance_data['info']['result']['list'][0]:
                total_balance_val = balance_data['info']['result']['list'][0]['totalEquity']
                logger.info(f"Using 'totalEquity' from Unified Account for {currency_code} balance.")
        total_balance = safe_decimal_conversion(total_balance_val)
        if total_balance is None: logger.warning(f"{NEON['WARNING']}{currency_code} balance rune unreadable.{NEON['RESET']}"); return None
        logger.info(f"{NEON['INFO']}Current {currency_code} Treasury: {NEON['VALUE']}{total_balance:.2f}{NEON['RESET']}")
        return total_balance
    except Exception as e: logger.error(f"{NEON['ERROR']}Error scrying treasury: {e}{NEON['RESET']}"); logger.debug(traceback.format_exc()); return None

def get_current_position_info() -> Tuple[str, Decimal]: # ... (existing logic)
    global _active_trade_parts
    if not _active_trade_parts: return CONFIG.pos_none, Decimal(0)
    total_qty = sum(part.get('qty', Decimal(0)) for part in _active_trade_parts if isinstance(part.get('qty'), Decimal))
    if total_qty <= CONFIG.position_qty_epsilon:
        if total_qty < Decimal(0): logger.error(f"{NEON['ERROR']}Negative total quantity {total_qty}. State corrupted! Clearing parts.{NEON['RESET']}")
        _active_trade_parts.clear(); save_persistent_state(force_heartbeat=True); return CONFIG.pos_none, Decimal(0)
    current_side_str = _active_trade_parts[0].get('side')
    if current_side_str not in [CONFIG.pos_long, CONFIG.pos_short]:
        logger.warning(f"{NEON['WARNING']}Incoherent side '{current_side_str}'. Assuming None. Clearing parts.{NEON['RESET']}")
        _active_trade_parts.clear(); save_persistent_state(force_heartbeat=True); return CONFIG.pos_none, total_qty
    return current_side_str, total_qty

def calculate_position_size(balance: Decimal, risk_per_trade_pct: Decimal, entry_price: Decimal, sl_price: Decimal, market_info: Dict) -> Optional[Decimal]: # ... (existing logic)
    logger.debug(f"{NEON['COMMENT']}# Calculating alchemical proportions... Bal: {balance:.2f}, Risk %: {risk_per_trade_pct:.4f}, Entry: {entry_price}, SL: {sl_price}{NEON['RESET']}")
    if not (isinstance(balance, Decimal) and balance > 0 and isinstance(entry_price, Decimal) and entry_price > 0 and isinstance(sl_price, Decimal) and sl_price > 0 and sl_price != entry_price):
        logger.warning(f"{NEON['WARNING']}Invalid runes for sizing. Bal: {balance}, Entry: {entry_price}, SL: {sl_price}{NEON['RESET']}"); return None
    risk_per_unit = abs(entry_price - sl_price)
    if risk_per_unit <= Decimal(0): logger.warning(f"{NEON['WARNING']}Risk per unit zero. Cannot divine size.{NEON['RESET']}"); return None
    usdt_at_risk = balance * risk_per_trade_pct; logger.debug(f"USDT at risk: {usdt_at_risk:.2f}")
    quantity_base = usdt_at_risk / risk_per_unit
    position_usdt_value = quantity_base * entry_price
    if position_usdt_value > CONFIG.max_order_usdt_amount:
        logger.info(f"Calculated position USDT value {position_usdt_value:.2f} exceeds MAX_ORDER_USDT_AMOUNT. Capping.")
        position_usdt_value = CONFIG.max_order_usdt_amount; quantity_base = position_usdt_value / entry_price
    
    qty_step_str = market_info.get('precision', {}).get('amount'); min_qty_str = market_info.get('limits', {}).get('amount', {}).get('min')
    qty_step = safe_decimal_conversion(qty_step_str); min_qty = safe_decimal_conversion(min_qty_str)

    if qty_step is None or qty_step <= Decimal(0): logger.warning(f"{NEON['WARNING']}Qty step invalid for {market_info.get('symbol')}. Using raw quantity.{NEON['RESET']}")
    else: quantity_base = (quantity_base // qty_step) * qty_step if quantity_base >= qty_step else Decimal(0)
    
    if quantity_base <= CONFIG.position_qty_epsilon: logger.warning(f"{NEON['WARNING']}Calculated qty {quantity_base} is zero/negligible. No order.{NEON['RESET']}"); return None
    if min_qty is not None and quantity_base < min_qty: logger.warning(f"{NEON['WARNING']}Calculated qty {quantity_base} < exchange min {min_qty}. No order.{NEON['RESET']}"); return None
    
    qty_display_precision = 0
    if qty_step is not None and qty_step > 0: s = str(qty_step).rstrip('0'); qty_display_precision = len(s.split('.')[1]) if '.' in s else 0
    logger.info(f"Calculated position size: {NEON['QTY']}{quantity_base:.{qty_display_precision}f}{NEON['RESET']} (USDT Value: {position_usdt_value:.2f})")
    return quantity_base

@retry((ccxt.NetworkError, ccxt.RequestTimeout, ccxt.InsufficientFunds), tries=2, delay=3, logger=logger)
def place_risked_order(exchange: ccxt.Exchange, config: Config, side: str, entry_price_target: Decimal, sl_price: Decimal, atr_val: Optional[Decimal], df_for_entry_context: Optional[pd.DataFrame] = None) -> Optional[Dict]:
    global _active_trade_parts, trade_timestamps_for_whipsaw, whipsaw_cooldown_active_until
    logger.info(f"{NEON['ACTION']}Attempting {side.upper()} order for {config.symbol}... Target: {NEON['PRICE']}{entry_price_target}{NEON['RESET']}, SL: {NEON['PRICE']}{sl_price}{NEON['RESET']}")

    # Snippet 2.4: Anti-Martingale Risk Sizing
    effective_risk_pct_for_trade = config.risk_per_trade_percentage
    if config.enable_anti_martingale_risk and trade_metrics.trades:
        last_trade_pnl = Decimal(trade_metrics.trades[-1]['profit_str'])
        if last_trade_pnl < 0: # Loss
            effective_risk_pct_for_trade *= config.risk_reduction_factor_on_loss
            logger.info(f"{NEON['INFO']}Anti-Martingale: Risk reduced to {effective_risk_pct_for_trade*100:.3f}% (prev loss).{NEON['RESET']}")
        elif last_trade_pnl > 0: # Win
            effective_risk_pct_for_trade *= config.risk_increase_factor_on_win
            effective_risk_pct_for_trade = min(effective_risk_pct_for_trade, config.max_risk_pct_anti_martingale)
            logger.info(f"{NEON['INFO']}Anti-Martingale: Risk adjusted to {effective_risk_pct_for_trade*100:.3f}% (prev win).{NEON['RESET']}")
    
    balance = fetch_account_balance(exchange, config.usdt_symbol)
    if balance is None or balance <= Decimal(10): logger.error(f"{NEON['ERROR']}Insufficient treasury ({balance}). Cannot order.{NEON['RESET']}"); return None
    if config.MARKET_INFO is None: logger.error(f"{NEON['ERROR']}Market runes not divined. Cannot order.{NEON['RESET']}"); return None

    quantity = calculate_position_size(balance, effective_risk_pct_for_trade, entry_price_target, sl_price, config.MARKET_INFO)
    if quantity is None or quantity <= CONFIG.position_qty_epsilon: logger.warning(f"{NEON['WARNING']}Position sizing yielded no quantity.{NEON['RESET']}"); return None

    order_side_verb = config.side_buy if side == config.pos_long else config.side_sell
    order_type_str = config.entry_order_type.value
    order_price_param_for_call = None # For limit orders

    # Snippet 4.4: Entry Price "Improvement" Check for Limit Orders
    if order_type_str == 'Limit':
        # Calculate initial limit price based on offset
        offset_val = (atr_val * config.limit_order_offset_atr_percentage) if atr_val else Decimal(0) # Ensure atr_val is present
        if side == config.pos_long: calculated_limit_price = entry_price_target - offset_val
        else: calculated_limit_price = entry_price_target + offset_val
        
        if config.enable_limit_order_price_improvement_check and config.tick_size:
            try:
                ticker = exchange.fetch_ticker(config.symbol)
                current_best_ask = Decimal(str(ticker['ask'])) if ticker and 'ask' in ticker and ticker['ask'] else None
                current_best_bid = Decimal(str(ticker['bid'])) if ticker and 'bid' in ticker and ticker['bid'] else None

                if side == config.pos_long and current_best_ask is not None and current_best_ask <= calculated_limit_price:
                    logger.info(f"{NEON['INFO']}Price Improve: Ask ({current_best_ask}) <= Limit ({calculated_limit_price}). Adjusting.{NEON['RESET']}")
                    calculated_limit_price = current_best_ask + config.tick_size # One tick better than current market for safety
                elif side == config.pos_short and current_best_bid is not None and current_best_bid >= calculated_limit_price:
                    logger.info(f"{NEON['INFO']}Price Improve: Bid ({current_best_bid}) >= Limit ({calculated_limit_price}). Adjusting.{NEON['RESET']}")
                    calculated_limit_price = current_best_bid - config.tick_size
            except Exception as e_ticker: logger.warning(f"{NEON['WARNING']}Ticker fetch fail for price improve: {e_ticker}{NEON['RESET']}")
        order_price_param_for_call = float(calculated_limit_price)
        logger.info(f"Calculated Limit Price for order: {order_price_param_for_call}")

    params = {'timeInForce': 'GTC' if order_type_str == 'Limit' else 'IOC', 'stopLoss': float(sl_price), 'slTriggerBy': 'MarkPrice', 'positionIdx': 0}
    if config.symbol.endswith(":USDT"): params['category'] = 'linear'

    try:
        logger.info(f"Weaving {order_type_str.upper()} {order_side_verb.upper()} order: Qty {NEON['QTY']}{quantity}{NEON['RESET']}, SL {NEON['PRICE']}{sl_price}{NEON['RESET']}"
                    f"{f', Price {order_price_param_for_call}' if order_price_param_for_call else ''}")
        order = exchange.create_order(config.symbol, order_type_str, order_side_verb, float(quantity), order_price_param_for_call, params)
        logger.success(f"{NEON['SUCCESS']}Entry Order {order['id']} ({order.get('status', 'N/A')}) cast.{NEON['RESET']}") # type: ignore [attr-defined]
        config.send_notification_method(f"Pyrmethus Order: {side.upper()}", f"{config.symbol} Qty: {quantity:.4f} @ {order_price_param_for_call if order_type_str == 'Limit' else 'Market'}, SL: {sl_price}")
        
        time.sleep(config.order_fill_timeout_seconds / 3) # Brief wait
        filled_order = exchange.fetch_order(order['id'], config.symbol)

        if filled_order.get('status') == 'closed': # Fully filled
            actual_entry_price = Decimal(str(filled_order.get('average', filled_order.get('price', entry_price_target))))
            actual_qty = Decimal(str(filled_order.get('filled', quantity)))
            entry_timestamp_ms = int(filled_order.get('timestamp', time.time() * 1000))
            part_id = str(uuid.uuid4())[:8]
            
            new_part = {
                "part_id": part_id, "entry_order_id": order['id'], "sl_order_id": None, "symbol": config.symbol, 
                "side": side, "entry_price": actual_entry_price, "qty": actual_qty, "entry_time_ms": entry_timestamp_ms, 
                "sl_price": sl_price, "atr_at_entry": atr_val, "initial_usdt_value": actual_qty * actual_entry_price,
                "breakeven_set": False, # For Snippet 2.2
                "recent_pnls": deque(maxlen=config.profit_momentum_window), # Snippet 1.2
                "last_known_pnl": Decimal(0), # Snippet 1.2
                "adverse_candle_closes": deque(maxlen=config.last_chance_consecutive_adverse_candles), # Snippet 2.5
                "partial_tp_taken_flags": {}, # For more advanced partial TPs, e.g. {"tp1": True}
                "divergence_exit_taken": False, # For Snippet 4.2
            }
            # For Snippet 4.2 (Ehlers Fisher Divergence Exit)
            if config.strategy_name == StrategyName.EHLERS_FISHER and df_for_entry_context is not None and not df_for_entry_context.empty:
                entry_candle_data = df_for_entry_context.iloc[-1]
                new_part["entry_fisher_value"] = safe_decimal_conversion(entry_candle_data.get("ehlers_fisher"))
                new_part["entry_signal_value"] = safe_decimal_conversion(entry_candle_data.get("ehlers_signal"))

            _active_trade_parts.append(new_part)
            trade_metrics.increment_daily_trade_entry_count() # Snippet 4.3
            # Snippet 1.3: Whipsaw Cooldown update
            if config.enable_whipsaw_cooldown:
                now_ts = time.time()
                trade_timestamps_for_whipsaw.append(now_ts)
                if len(trade_timestamps_for_whipsaw) == config.whipsaw_max_trades_in_period and \
                   (now_ts - trade_timestamps_for_whipsaw[0]) <= config.whipsaw_period_seconds:
                    logger.critical(f"{NEON['CRITICAL']}Whipsaw detected! Pausing for {config.whipsaw_cooldown_seconds}s.{NEON['RESET']}")
                    config.send_notification_method("Pyrmethus Whipsaw", f"Pausing for {config.whipsaw_cooldown_seconds}s.")
                    whipsaw_cooldown_active_until = now_ts + config.whipsaw_cooldown_seconds
                    # close_all_symbol_positions(exchange, config, "Whipsaw Cooldown") # Optional: flatten

            save_persistent_state(force_heartbeat=True)
            logger.success(f"{NEON['SUCCESS']}Order {order['id']} filled! Part {part_id}. Entry: {NEON['PRICE']}{actual_entry_price}{NEON['RESET']}, Qty: {NEON['QTY']}{actual_qty}{NEON['RESET']}") # type: ignore [attr-defined]
            config.send_notification_method(f"Pyrmethus Order Filled: {side.upper()}", f"{config.symbol} Part {part_id} @ {actual_entry_price:.2f}")
            return new_part
        else:
            logger.warning(f"{NEON['WARNING']}Order {order['id']} not 'closed'. Status: {filled_order.get('status', 'N/A')}. Manual check.{NEON['RESET']}")
            return None
    except ccxt.InsufficientFunds as e: logger.error(f"{NEON['ERROR']}Insufficient funds: {e}{NEON['RESET']}"); return None
    except ccxt.ExchangeError as e: logger.error(f"{NEON['ERROR']}Exchange rejected order: {e}{NEON['RESET']}"); return None
    except Exception as e: logger.error(f"{NEON['ERROR']}Unexpected chaos weaving order: {e}{NEON['RESET']}"); logger.debug(traceback.format_exc()); return None

def close_position_part(exchange: ccxt.Exchange, config: Config, part_to_close: Dict, reason: str, close_price_target: Optional[Decimal] = None) -> bool: # ... (existing logic)
    global _active_trade_parts
    part_id = part_to_close.get('part_id', 'UnknownPart'); part_qty = part_to_close.get('qty', Decimal(0)); part_side = part_to_close.get('side')
    logger.info(f"{NEON['ACTION']}Unraveling part {part_id} ({part_side} {NEON['QTY']}{part_qty}{NEON['RESET']}) for: {reason}{NEON['RESET']}")
    if not isinstance(part_qty, Decimal) or part_qty <= CONFIG.position_qty_epsilon:
        logger.warning(f"{NEON['WARNING']}Cannot unravel part {part_id}: qty {part_qty} invalid.{NEON['RESET']}")
        _active_trade_parts = [p for p in _active_trade_parts if p.get('part_id') != part_id]; save_persistent_state(force_heartbeat=True); return False

    close_side_verb = config.side_sell if part_side == config.pos_long else config.side_buy
    params = {'reduceOnly': True}; 
    if config.symbol.endswith(":USDT"): params['category'] = 'linear'
    try:
        logger.info(f"Casting MARKET {close_side_verb.upper()} order for part {part_id}: Qty {NEON['QTY']}{part_qty}{NEON['RESET']}")
        close_order = exchange.create_order(config.symbol, 'market', close_side_verb, float(part_qty), None, params)
        logger.success(f"{NEON['SUCCESS']}Unraveling Order {close_order['id']} ({close_order.get('status', 'N/A')}) cast.{NEON['RESET']}") # type: ignore [attr-defined]
        time.sleep(config.order_fill_timeout_seconds / 2)
        filled_close_order = exchange.fetch_order(close_order['id'], config.symbol)

        if filled_close_order.get('status') == 'closed':
            actual_exit_price = Decimal(str(filled_close_order.get('average', filled_close_order.get('price', close_price_target or part_to_close['entry_price']))))
            exit_timestamp_ms = int(filled_close_order.get('timestamp', time.time() * 1000))
            entry_price = part_to_close.get('entry_price', Decimal(0))
            if not isinstance(entry_price, Decimal): entry_price = Decimal(str(entry_price))
            pnl_per_unit = (actual_exit_price - entry_price) if part_side == config.pos_long else (entry_price - actual_exit_price)
            pnl = pnl_per_unit * part_qty
            trade_metrics.log_trade(symbol=config.symbol, side=part_side, entry_price=entry_price, exit_price=actual_exit_price,
                                    qty=part_qty, entry_time_ms=part_to_close['entry_time_ms'], exit_time_ms=exit_timestamp_ms,
                                    reason=reason, part_id=part_id, pnl_str=str(pnl))
            _active_trade_parts = [p for p in _active_trade_parts if p.get('part_id') != part_id]
            save_persistent_state(force_heartbeat=True)
            pnl_color_key = 'PNL_POS' if pnl > 0 else ('PNL_NEG' if pnl < 0 else 'PNL_ZERO')
            logger.success(f"{NEON['SUCCESS']}Part {part_id} unraveled. Exit: {NEON['PRICE']}{actual_exit_price}{NEON['RESET']}, PNL: {NEON[pnl_color_key]}{pnl:.2f} {config.usdt_symbol}{NEON['RESET']}") # type: ignore [attr-defined]
            config.send_notification_method(f"Pyrmethus Position Closed", f"{config.symbol} Part {part_id}. PNL: {pnl:.2f}. Reason: {reason}")
            return True
        else:
            logger.warning(f"{NEON['WARNING']}Unraveling order {close_order['id']} for part {part_id} not 'closed'. Status: {filled_close_order.get('status', 'N/A')}. Manual check.{NEON['RESET']}")
            return False
    except Exception as e:
        logger.error(f"{NEON['ERROR']}Error unraveling part {part_id}: {e}{NEON['RESET']}"); logger.debug(traceback.format_exc())
        config.send_notification_method("Pyrmethus Close Fail", f"Error closing {config.symbol} part {part_id}: {str(e)[:80]}"); return False

@retry((ccxt.NetworkError, ccxt.RequestTimeout), tries=CONFIG.retry_count, delay=CONFIG.retry_delay_seconds)
def cancel_all_symbol_orders(exchange: ccxt.Exchange, symbol: str): # ... (existing logic)
    logger.info(f"{NEON['ACTION']}Dispelling all open enchantments for {symbol}...{NEON['RESET']}")
    try:
        params = {'category': 'linear'} if symbol.endswith(":USDT") else {}
        exchange.cancel_all_orders(symbol, params=params if params else None)
        logger.success(f"{NEON['SUCCESS']}All open enchantments for {symbol} dispelled.{NEON['RESET']}") # type: ignore [attr-defined]
    except ccxt.FeatureNotSupported:
        logger.warning(f"{NEON['WARNING']}Exchange {exchange.id} may not support cancel_all_orders for symbol. Trying individual.{NEON['RESET']}")
        try:
            open_orders = exchange.fetch_open_orders(symbol)
            if not open_orders: logger.info(f"No open orders for {symbol} to dispel."); return
            logger.info(f"Found {len(open_orders)} open orders for {symbol}. Dispelling individually...")
            for order in open_orders: exchange.cancel_order(order['id'], symbol, params=params if params else None); logger.info(f"Dispelled order {order['id']}.")
            logger.success(f"{NEON['SUCCESS']}Individually dispelled {len(open_orders)} orders for {symbol}.{NEON['RESET']}") # type: ignore [attr-defined]
        except Exception as e_ind: logger.error(f"{NEON['ERROR']}Error dispelling individual enchantments: {e_ind}{NEON['RESET']}")
    except Exception as e: logger.error(f"{NEON['ERROR']}Error dispelling enchantments: {e}{NEON['RESET']}")

def close_all_symbol_positions(exchange: ccxt.Exchange, config: Config, reason: str): # ... (existing logic)
    global _active_trade_parts
    logger.warning(f"{NEON['WARNING']}Forceful unraveling of ALL positions for {config.symbol} due to: {reason}{NEON['RESET']}")
    parts_to_close_copy = list(_active_trade_parts) # Operate on a copy
    if parts_to_close_copy:
        logger.info(f"Unraveling {len(parts_to_close_copy)} internally tracked part(s)...")
        for part in parts_to_close_copy:
            if part.get('symbol') == config.symbol: close_position_part(exchange, config, part, reason + " (Global Unraveling)")
    else: logger.info("No internally tracked trade parts to unravel.")
    logger.info(f"Safeguard scrying exchange for residual {config.symbol} positions...")
    try:
        params_fetch_pos = {'category': 'linear'} if config.symbol.endswith(":USDT") else {}
        positions = exchange.fetch_positions([config.symbol], params=params_fetch_pos)
        residual_closed_count = 0
        for pos_data in positions:
            if pos_data and pos_data.get('symbol') == config.symbol:
                pos_qty = safe_decimal_conversion(pos_data.get('contracts', pos_data.get('size', '0')), Decimal(0))
                pos_side_str = pos_data.get('side') # 'long' or 'short'
                if pos_qty > CONFIG.position_qty_epsilon:
                    side_to_close_market = config.side_sell if pos_side_str == 'long' else config.side_buy
                    logger.warning(f"Found residual exchange position: {pos_side_str} {pos_qty} {config.symbol}. Force market unravel.")
                    params_close_residual = {'reduceOnly': True, 'category': 'linear' if config.symbol.endswith(":USDT") else None}
                    exchange.create_order(config.symbol, 'market', side_to_close_market, float(pos_qty), params=params_close_residual)
                    residual_closed_count += 1; time.sleep(config.post_close_delay_seconds)
        if residual_closed_count > 0: logger.info(f"Safeguard unraveling attempted for {residual_closed_count} residual position(s).")
        else: logger.info("No residual positions found on exchange for safeguard unraveling.")
    except Exception as e: logger.error(f"{NEON['ERROR']}Error during final scrying/unraveling of exchange positions: {e}{NEON['RESET']}")
    cancel_all_symbol_orders(exchange, config.symbol)
    logger.info(f"All positions/orders for {config.symbol} should now be unraveled/dispelled.")

# --- Helper: Update Daily Key Levels (Snippet 2.1) ---
def update_daily_key_levels(exchange: ccxt.Exchange, config: Config):
    global previous_day_high, previous_day_low, last_key_level_update_day
    today_utc_day = datetime.now(pytz.utc).day
    if last_key_level_update_day == today_utc_day: return 

    logger.info("Attempting to update daily key levels (Prev Day H/L)...")
    try:
        # Fetch yesterday's OHLCV (1 day candle)
        # To get yesterday's full candle, 'since' should be timestamp of 2 days ago at 00:00 UTC
        two_days_ago_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=2)
        since_timestamp_ms = int(two_days_ago_utc.timestamp() * 1000)
        params = {'category': 'linear'} if config.symbol.endswith(":USDT") else {}
        daily_candles = exchange.fetch_ohlcv(config.symbol, '1d', since=since_timestamp_ms, limit=2, params=params)
        
        if daily_candles and len(daily_candles) >= 1:
            # Assuming the exchange returns candles in chronological order,
            # if we ask for 2 candles ending "now" (or up to the last full one),
            # daily_candles[0] should be yesterday's candle if today's hasn't fully formed or is the second one.
            # This logic can be tricky depending on exact exchange behavior and timing.
            # A robust way is to check timestamps. For simplicity, using first of last two.
            # If only one candle returned, it might be yesterday's or an incomplete today's.
            
            # Find yesterday's candle by checking timestamp
            yesterday_target_date = (datetime.now(pytz.utc) - timedelta(days=1)).date()
            found_yesterday = False
            for candle_data in reversed(daily_candles): # Check newest first
                candle_ts_utc = datetime.fromtimestamp(candle_data[0] / 1000, tz=pytz.utc)
                if candle_ts_utc.date() == yesterday_target_date:
                    previous_day_high = Decimal(str(candle_data[2])) # Index 2 is high
                    previous_day_low = Decimal(str(candle_data[3]))  # Index 3 is low
                    logger.info(f"{NEON['INFO']}Updated key levels: Prev Day High {previous_day_high}, Prev Day Low {previous_day_low}{NEON['RESET']}")
                    last_key_level_update_day = today_utc_day
                    found_yesterday = True
                    break
            if not found_yesterday:
                 logger.warning(f"{NEON['WARNING']}Could not definitively find yesterday's candle data to update key levels.{NEON['RESET']}")
        else:
            logger.warning(f"{NEON['WARNING']}Could not fetch sufficient daily candle data to update key levels.{NEON['RESET']}")
    except Exception as e_kl:
        logger.error(f"{NEON['ERROR']}Error updating daily key levels: {e_kl}{NEON['RESET']}")


# --- Main Spell Weaving ---
def main_loop(exchange: ccxt.Exchange, config: Config) -> None:
    global _active_trade_parts, trade_timestamps_for_whipsaw, whipsaw_cooldown_active_until
    global persistent_signal_counter, last_signal_type
    global previous_day_high, previous_day_low, last_key_level_update_day # For Snippet 2.1
    global contradiction_cooldown_active_until # For Snippet 3.5
    global consecutive_loss_cooldown_active_until # For consecutive loss limiter

    logger.info(f"{NEON['HEADING']}=== Pyrmethus Spell v{PYRMETHUS_VERSION} Awakening on {exchange.id} ==={NEON['RESET']}")
    if load_persistent_state(): logger.success(f"{NEON['SUCCESS']}Reawakened from Phoenix scroll. Active parts: {len(_active_trade_parts)}.{NEON['RESET']}") # type: ignore [attr-defined]
    else: logger.info(f"{NEON['INFO']}No prior state or ignored. Starting fresh.{NEON['RESET']}")

    current_balance = fetch_account_balance(exchange, config.usdt_symbol)
    if current_balance is not None: trade_metrics.set_initial_equity(current_balance)
    else: logger.warning(f"{NEON['WARNING']}Could not set initial treasury. Drawdown checks affected.{NEON['RESET']}")

    loop_counter = 0
    while True:
        loop_counter += 1
        try:
            logger.debug(f"{NEON['COMMENT']}# New cycle of observation ({loop_counter})...{NEON['RESET']}")

            # --- Cooldown Checks ---
            # Snippet 1.3: Whipsaw Cooldown
            if config.enable_whipsaw_cooldown and time.time() < whipsaw_cooldown_active_until:
                logger.warning(f"{NEON['WARNING']}In whipsaw cooldown. Ends in {whipsaw_cooldown_active_until - time.time():.0f}s.{NEON['RESET']}")
                time.sleep(config.sleep_seconds); continue
            # Snippet 3.5: Trend Contradiction Cooldown
            if config.enable_trend_contradiction_cooldown and time.time() < contradiction_cooldown_active_until:
                logger.warning(f"{NEON['WARNING']}In Trend Contradiction cooldown. Ends in {contradiction_cooldown_active_until - time.time():.0f}s.{NEON['RESET']}")
                time.sleep(config.sleep_seconds); continue
            # Snippet 4.3: Daily Max Trades Rest
            if config.enable_daily_max_trades_rest and trade_metrics.daily_trades_rest_active_until > time.time():
                logger.warning(f"{NEON['WARNING']}Resting (daily max trades). Resumes in {(trade_metrics.daily_trades_rest_active_until - time.time())/3600:.1f} hrs.{NEON['RESET']}")
                time.sleep(config.sleep_seconds * 10); continue
            # Consecutive Loss Cooldown
            if config.enable_consecutive_loss_limiter and time.time() < consecutive_loss_cooldown_active_until:
                logger.warning(f"{NEON['WARNING']}In consecutive loss cooldown. Ends in {(consecutive_loss_cooldown_active_until - time.time())/60:.1f} mins.{NEON['RESET']}")
                time.sleep(config.sleep_seconds); continue
            
            # --- Balance & Drawdown ---
            current_balance = fetch_account_balance(exchange, config.usdt_symbol)
            if current_balance is not None:
                trade_metrics.set_initial_equity(current_balance) # Updates daily equity if new day, resets daily trade count
                drawdown_hit, dd_reason = trade_metrics.check_drawdown(current_balance)
                if drawdown_hit:
                    logger.critical(f"{NEON['CRITICAL']}Max drawdown! {dd_reason}. Pyrmethus rests.{NEON['RESET']}")
                    close_all_symbol_positions(exchange, config, "Max Drawdown Reached"); break
            else: logger.warning(f"{NEON['WARNING']}Failed treasury scry. Drawdown check on stale data.{NEON['RESET']}")

            # Snippet 2.3: Session PNL Limits
            if config.enable_session_pnl_limits and trade_metrics.initial_equity is not None and current_balance is not None:
                current_session_pnl = current_balance - trade_metrics.initial_equity # Assumes initial_equity is session start
                if config.session_profit_target_usdt is not None and current_session_pnl >= config.session_profit_target_usdt:
                    logger.critical(f"{NEON['SUCCESS']}SESSION PROFIT TARGET {config.session_profit_target_usdt} USDT REACHED! PNL: {current_session_pnl:.2f}.{NEON['RESET']}")
                    config.send_notification_method("Pyrmethus Session Goal!", f"Profit target {config.session_profit_target_usdt} hit.")
                    close_all_symbol_positions(exchange, config, "Session Profit Target Reached"); break
                if config.session_max_loss_usdt is not None and current_session_pnl <= -abs(config.session_max_loss_usdt):
                    logger.critical(f"{NEON['CRITICAL']}SESSION MAX LOSS {config.session_max_loss_usdt} USDT REACHED! PNL: {current_session_pnl:.2f}.{NEON['RESET']}")
                    config.send_notification_method("Pyrmethus Session Loss!", f"Max session loss {config.session_max_loss_usdt} hit.")
                    close_all_symbol_positions(exchange, config, "Session Max Loss Reached"); break
            
            # Snippet 4.3: Check Daily Max Trades Limit (after count potentially incremented by a trade)
            if config.enable_daily_max_trades_rest and \
               trade_metrics.daily_trade_entry_count >= config.daily_max_trades_limit and \
               trade_metrics.daily_trades_rest_active_until <= time.time(): # Check if already resting
                logger.critical(f"{NEON['CRITICAL']}Daily max trades ({config.daily_max_trades_limit}) reached. Resting for {config.daily_max_trades_rest_hours} hrs.{NEON['RESET']}")
                config.send_notification_method("Pyrmethus Daily Trades Rest", f"Max {config.daily_max_trades_limit} trades. Resting.")
                trade_metrics.daily_trades_rest_active_until = time.time() + config.daily_max_trades_rest_hours * 3600
                close_all_symbol_positions(exchange, config, "Daily Max Trades Limit Reached"); continue

            # Consecutive Loss Limiter Check
            if config.enable_consecutive_loss_limiter and trade_metrics.consecutive_losses >= config.max_consecutive_losses:
                logger.critical(f"{NEON['CRITICAL']}{config.max_consecutive_losses} consecutive losses! Cooldown for {config.consecutive_loss_cooldown_minutes} mins.{NEON['RESET']}")
                config.send_notification_method("Pyrmethus Cooldown", f"{config.max_consecutive_losses} losses. Pausing.")
                consecutive_loss_cooldown_active_until = time.time() + config.consecutive_loss_cooldown_minutes * 60
                trade_metrics.consecutive_losses = 0 # Reset counter
                close_all_symbol_positions(exchange, config, "Consecutive Loss Limit"); continue

            # --- Market Data & Indicators ---
            ohlcv_df = get_market_data(exchange, config.symbol, config.interval, limit=OHLCV_LIMIT + config.api_fetch_limit_buffer)
            if ohlcv_df is None or ohlcv_df.empty: logger.warning(f"{NEON['WARNING']}Market whispers faint. Retrying.{NEON['RESET']}"); time.sleep(config.sleep_seconds); continue
            df_with_indicators = calculate_all_indicators(ohlcv_df.copy(), config)
            if df_with_indicators.empty or df_with_indicators.iloc[-1].isnull().all(): logger.warning(f"{NEON['WARNING']}Indicator alchemy faint. Pausing.{NEON['RESET']}"); time.sleep(config.sleep_seconds); continue

            latest_close = safe_decimal_conversion(df_with_indicators['close'].iloc[-1])
            atr_col = f"ATR_{config.atr_calculation_period}"
            latest_atr = safe_decimal_conversion(df_with_indicators[atr_col].iloc[-1] if atr_col in df_with_indicators.columns else pd.NA)
            if pd.isna(latest_close) or pd.isna(latest_atr) or latest_atr <= Decimal(0):
                logger.warning(f"{NEON['WARNING']}Missing Close/ATR or invalid ATR ({_format_for_log(latest_atr)}). No decisions.{NEON['RESET']}")
                time.sleep(config.sleep_seconds); continue

            signals = config.strategy_instance.generate_signals(df_with_indicators, latest_close, latest_atr)
            
            # Snippet 1.5: Stateful Signal Confirmation
            confirmed_enter_long, confirmed_enter_short = False, False
            if config.signal_persistence_candles <= 1:
                confirmed_enter_long = signals.get("enter_long", False)
                confirmed_enter_short = signals.get("enter_short", False)
            else:
                current_long_sig = signals.get("enter_long", False); current_short_sig = signals.get("enter_short", False)
                if current_long_sig:
                    if last_signal_type == "long" or last_signal_type is None: persistent_signal_counter["long"] += 1
                    else: persistent_signal_counter["long"] = 1; persistent_signal_counter["short"] = 0
                    last_signal_type = "long"
                    if persistent_signal_counter["long"] >= config.signal_persistence_candles: confirmed_enter_long = True
                elif current_short_sig:
                    if last_signal_type == "short" or last_signal_type is None: persistent_signal_counter["short"] += 1
                    else: persistent_signal_counter["short"] = 1; persistent_signal_counter["long"] = 0
                    last_signal_type = "short"
                    if persistent_signal_counter["short"] >= config.signal_persistence_candles: confirmed_enter_short = True
                else: # No entry signal this candle
                    persistent_signal_counter["long"] = 0; persistent_signal_counter["short"] = 0
                    # Don't reset last_signal_type here, only on flip or confirmed entry
                logger.debug(f"Signal persistence: Lcnt {persistent_signal_counter['long']}, Scnt {persistent_signal_counter['short']}. Confirmed L/S: {confirmed_enter_long}/{confirmed_enter_short}")

            current_pos_side, current_pos_qty = get_current_position_info()
            # --- Entry Logic ---
            if current_pos_side == config.pos_none:
                # Snippet 1.4: Max Active Trade Parts (already handled by current_pos_side == config.pos_none for single part)
                # If allowing multiple parts, this check `len(_active_trade_parts) < config.max_active_trade_parts` would be here.
                # For now, this structure implicitly handles max 1 "logical" position (can be multiple parts if scaling was added later)
                if len(_active_trade_parts) >= config.max_active_trade_parts:
                     logger.info(f"{NEON['INFO']}Max active trade parts ({config.max_active_trade_parts}) reached. No new entries.{NEON['RESET']}")
                elif confirmed_enter_long or confirmed_enter_short:
                    # Snippet 2.1: No-Trade Zone Check
                    update_daily_key_levels(exchange, config) # Update once per day
                    trade_allowed_by_zone = True
                    if config.enable_no_trade_zones and latest_close is not None:
                        key_levels_to_check = []
                        if previous_day_high: key_levels_to_check.append(previous_day_high)
                        if previous_day_low: key_levels_to_check.append(previous_day_low)
                        if config.key_round_number_step and config.key_round_number_step > 0:
                            lower_round = (latest_close // config.key_round_number_step) * config.key_round_number_step
                            key_levels_to_check.extend([lower_round, lower_round + config.key_round_number_step])
                        for level in key_levels_to_check:
                            zone_half_width = level * config.no_trade_zone_pct_around_key_level
                            if (level - zone_half_width) <= latest_close <= (level + zone_half_width):
                                trade_allowed_by_zone = False
                                logger.debug(f"No-trade zone active around {level}. Entry suppressed."); break
                    
                    # Snippet 4.5: Trap Filter
                    trap_detected = False
                    if config.enable_trap_filter and latest_atr > 0 and len(df_with_indicators) > config.trap_filter_lookback_period:
                        prev_candle_idx = -2 # Context for swing H/L is candle before signal candle
                        if len(df_with_indicators) > abs(prev_candle_idx): # Ensure df is long enough
                            prev_candle_high = safe_decimal_conversion(df_with_indicators['high'].iloc[prev_candle_idx])
                            prev_candle_low = safe_decimal_conversion(df_with_indicators['low'].iloc[prev_candle_idx])
                            rolling_high_col = f'rolling_high_{config.trap_filter_lookback_period}'
                            rolling_low_col = f'rolling_low_{config.trap_filter_lookback_period}'
                            recent_high_val = safe_decimal_conversion(df_with_indicators[rolling_high_col].iloc[prev_candle_idx]) # High up to prev candle
                            recent_low_val = safe_decimal_conversion(df_with_indicators[rolling_low_col].iloc[prev_candle_idx])   # Low up to prev candle
                            rejection_needed = latest_atr * config.trap_filter_rejection_threshold_atr
                            wick_prox_val = latest_atr * config.trap_filter_wick_proximity_atr

                            if confirmed_enter_long and recent_high_val and prev_candle_high:
                                if abs(prev_candle_high - recent_high_val) < wick_prox_val and \
                                   (recent_high_val - latest_close) >= rejection_needed:
                                    trap_detected = True; logger.debug(f"Bull Trap Filter: Rejected from recent high {recent_high_val}. Long suppressed.")
                            elif confirmed_enter_short and recent_low_val and prev_candle_low:
                                if abs(prev_candle_low - recent_low_val) < wick_prox_val and \
                                   (latest_close - recent_low_val) >= rejection_needed: # current close is above recent low
                                    trap_detected = True; logger.debug(f"Bear Trap Filter: Rejected from recent low {recent_low_val}. Short suppressed.")
                    
                    if trade_allowed_by_zone and not trap_detected:
                        entry_side = config.pos_none
                        if confirmed_enter_long: entry_side = config.pos_long
                        elif confirmed_enter_short: entry_side = config.pos_short
                        
                        if entry_side != config.pos_none:
                            sl_price = (latest_close - (latest_atr * config.atr_stop_loss_multiplier)) if entry_side == config.pos_long \
                                  else (latest_close + (latest_atr * config.atr_stop_loss_multiplier))
                            new_part = place_risked_order(exchange, config, entry_side, latest_close, sl_price, latest_atr, df_with_indicators)
                            if new_part: # Reset signal persistence on successful entry
                                persistent_signal_counter = {"long": 0, "short": 0}; last_signal_type = None
            
            # --- Position Management ---
            active_parts_copy = list(_active_trade_parts) # Iterate over a copy for safe modification
            for part in active_parts_copy:
                if part.get('symbol') != config.symbol: continue # Should not happen if state is clean

                # Snippet 3.5: Trend Contradiction Cooldown (simplified check after entry)
                if config.enable_trend_contradiction_cooldown and part.get('side') and not part.get('contradiction_checked', False):
                    time_since_entry_ms = (time.time() * 1000) - part['entry_time_ms']
                    # Check if within N candles (approx.)
                    check_window_ms = config.trend_contradiction_check_candles_after_entry * config.sleep_seconds * 1000 
                    if time_since_entry_ms < check_window_ms and time_since_entry_ms > 0: # Check after some time has passed but still recent
                        confirm_trend_now = df_with_indicators['confirm_trend'].iloc[-1] if config.strategy_name == StrategyName.DUAL_SUPERTREND_MOMENTUM else pd.NA # Adapt for other strats
                        if (part['side'] == config.pos_long and confirm_trend_now is False) or \
                           (part['side'] == config.pos_short and confirm_trend_now is True):
                            logger.warning(f"{NEON['WARNING']}Trend contradiction for part {part['part_id']} shortly after entry. Cooldown activated.{NEON['RESET']}")
                            contradiction_cooldown_active_until = time.time() + config.trend_contradiction_cooldown_seconds
                            # Optionally close this contradicted trade:
                            # close_position_part(exchange, config, part, "Immediate Trend Contradiction")
                    part['contradiction_checked'] = True # Check only once or periodically

                # Calculate current PNL for this part
                current_unrealized_pnl = Decimal(0)
                if part['side'] == config.pos_long: current_unrealized_pnl = (latest_close - part['entry_price']) * part['qty']
                elif part['side'] == config.pos_short: current_unrealized_pnl = (part['entry_price'] - latest_close) * part['qty']
                part.get('recent_pnls', deque()).append(current_unrealized_pnl) # For Snippet 1.2

                # Snippet 1.2: Profit Momentum SL Tightening
                if config.enable_profit_momentum_sl_tighten and len(part.get('recent_pnls',[])) == config.profit_momentum_window:
                    recent_pnls_list = list(part['recent_pnls'])
                    is_profit_increasing = all(recent_pnls_list[i] > recent_pnls_list[i-1] for i in range(1, len(recent_pnls_list)))
                    is_currently_profitable = recent_pnls_list[-1] > 0
                    if is_currently_profitable and is_profit_increasing and latest_atr > 0:
                        tightened_sl_dist = (part.get('atr_at_entry', latest_atr) * config.atr_stop_loss_multiplier) * config.profit_momentum_sl_tighten_factor
                        new_sl_price = (latest_close - tightened_sl_dist) if part['side'] == config.pos_long else (latest_close + tightened_sl_dist)
                        if (part['side'] == config.pos_long and new_sl_price > part['sl_price'] and new_sl_price < latest_close) or \
                           (part['side'] == config.pos_short and new_sl_price < part['sl_price'] and new_sl_price > latest_close):
                            logger.info(f"{NEON['ACTION']}Profit Momentum SL Tighten: Part {part['part_id']} to {new_sl_price:.{config.MARKET_INFO['precision']['price'] if config.MARKET_INFO else 2}}{NEON['RESET']}")
                            part['sl_price'] = new_sl_price # TODO: Actual exchange SL modification
                            save_persistent_state(force_heartbeat=True)
                
                # Snippet 2.2: Breakeven SL (with min PNL requirement)
                if config.enable_breakeven_sl and not part.get('breakeven_set', False) and latest_atr > 0:
                    current_profit_atr = (current_unrealized_pnl / part['qty']) / latest_atr if part['qty'] > 0 and latest_atr > 0 else Decimal(0)
                    atr_target_met = current_profit_atr >= config.breakeven_profit_atr_target
                    abs_pnl_target_met = current_unrealized_pnl >= config.breakeven_min_abs_pnl_usdt
                    if atr_target_met and abs_pnl_target_met:
                        new_sl_price = part['entry_price']
                        if (part['side'] == config.pos_long and new_sl_price > part['sl_price']) or \
                           (part['side'] == config.pos_short and new_sl_price < part['sl_price']):
                            logger.info(f"{NEON['ACTION']}Breakeven SL: Part {part['part_id']} to {new_sl_price}{NEON['RESET']}")
                            part['sl_price'] = new_sl_price; part['breakeven_set'] = True # TODO: Actual exchange SL modification
                            save_persistent_state(force_heartbeat=True)

                # Snippet 2.5: Last Chance Exit
                part.get('adverse_candle_closes', deque()).append(latest_close)
                if config.enable_last_chance_exit and latest_atr > 0 and len(part.get('adverse_candle_closes',[])) == config.last_chance_consecutive_adverse_candles:
                    adverse_closes_list = list(part['adverse_candle_closes'])
                    is_cons_adverse = (all(adverse_closes_list[i] < adverse_closes_list[i-1] for i in range(1, len(adverse_closes_list))) if part['side'] == config.pos_long else
                                       all(adverse_closes_list[i] > adverse_closes_list[i-1] for i in range(1, len(adverse_closes_list))))
                    if is_cons_adverse:
                        dist_to_sl = abs(latest_close - part['sl_price'])
                        sl_prox_thresh_price = latest_atr * config.last_chance_sl_proximity_atr
                        if dist_to_sl <= sl_prox_thresh_price:
                            logger.warning(f"{NEON['WARNING']}Last Chance Exit: Part {part['part_id']} ({dist_to_sl:.{config.MARKET_INFO['precision']['price'] if config.MARKET_INFO else 2}} <= {sl_prox_thresh_price:.{config.MARKET_INFO['precision']['price'] if config.MARKET_INFO else 2}}).{NEON['RESET']}")
                            if close_position_part(exchange, config, part, "Last Chance Preemptive Exit"): continue # Part closed, skip other checks for it
                
                # Snippet 4.2: Ehlers Fisher Scaled Exit on Divergence
                if config.strategy_name == StrategyName.EHLERS_FISHER and config.ehlers_enable_divergence_scaled_exit and \
                   not part.get('divergence_exit_taken', False) and 'entry_fisher_value' in part and part['entry_fisher_value'] is not None:
                    fisher_now = safe_decimal_conversion(df_with_indicators['ehlers_fisher'].iloc[-1])
                    signal_now = safe_decimal_conversion(df_with_indicators['ehlers_signal'].iloc[-1])
                    if fisher_now is not pd.NA and signal_now is not pd.NA:
                        entry_fisher = part['entry_fisher_value']; entry_signal = part.get('entry_signal_value', pd.NA) # ensure entry_signal_value is loaded
                        if entry_fisher is not pd.NA and entry_signal is not pd.NA:
                            initial_spread = abs(entry_fisher - entry_signal); current_spread = abs(fisher_now - signal_now)
                            divergence_met = False
                            if part['side'] == config.pos_long and fisher_now > signal_now and fisher_now < entry_fisher and current_spread < (initial_spread * config.ehlers_divergence_threshold_factor):
                                divergence_met = True
                            elif part['side'] == config.pos_short and fisher_now < signal_now and fisher_now > entry_fisher and current_spread < (initial_spread * config.ehlers_divergence_threshold_factor):
                                divergence_met = True
                            if divergence_met:
                                qty_to_close_f = part['qty'] * config.ehlers_divergence_exit_percentage
                                # Adjust qty_to_close_f to market precision
                                qty_step = safe_decimal_conversion(config.MARKET_INFO['precision']['amount']) if config.MARKET_INFO else Decimal("1e-8")
                                qty_to_close = (qty_to_close_f // qty_step) * qty_step if qty_step > 0 else qty_to_close_f

                                if qty_to_close > config.position_qty_epsilon:
                                    logger.info(f"{NEON['ACTION']}Ehlers Divergence Exit ({config.ehlers_divergence_exit_percentage*100}%): Part {part['part_id']}, Qty {qty_to_close}{NEON['RESET']}")
                                    # Create a temporary part dict for closing
                                    temp_part_to_close = part.copy()
                                    temp_part_to_close['qty'] = qty_to_close
                                    if close_position_part(exchange, config, temp_part_to_close, "Ehlers Divergence Partial Exit"):
                                        part['qty'] -= qty_to_close # Reduce original part quantity
                                        part['divergence_exit_taken'] = True
                                        if part['qty'] <= config.position_qty_epsilon: # If remaining is negligible, remove it from active parts
                                            _active_trade_parts = [p for p in _active_trade_parts if p.get('part_id') != part['part_id']]
                                            logger.info(f"Part {part['part_id']} fully closed due to negligible remainder after partial Ehlers div exit.")
                                        save_persistent_state(force_heartbeat=True)
                                        continue # Part modified or closed, re-evaluate next cycle or skip
                                else: logger.debug(f"Ehlers div exit: calculated qty_to_close {qty_to_close} too small.")
                
                # Standard Exit Signals
                if (part['side'] == config.pos_long and signals.get("exit_long")) or \
                   (part['side'] == config.pos_short and signals.get("exit_short")):
                    exit_reason = signals.get('exit_reason', 'Strategy Exit Signal')
                    logger.info(f"{NEON['ACTION']}Omen to EXIT {part['side']} for {config.symbol}: {exit_reason}{NEON['RESET']}")
                    if close_position_part(exchange, config, part, exit_reason): continue # Part closed, skip

            save_persistent_state()
            logger.debug(f"{NEON['COMMENT']}# Cycle complete. Resting {config.sleep_seconds}s...{NEON['RESET']}")
            time.sleep(config.sleep_seconds)

        except KeyboardInterrupt:
            logger.warning(f"\n{NEON['WARNING']}Sorcerer's intervention! Pyrmethus prepares for slumber...{NEON['RESET']}")
            config.send_notification_method("Pyrmethus Shutdown", "Manual shutdown initiated."); break
        except Exception as e:
            logger.critical(f"{NEON['CRITICAL']}Critical error in main loop! Error: {e}{NEON['RESET']}"); logger.debug(traceback.format_exc())
            config.send_notification_method("Pyrmethus Critical Error", f"Loop crashed: {str(e)[:150]}")
            logger.info(f"Resting longer ({config.sleep_seconds * 5}s) after critical error."); time.sleep(config.sleep_seconds * 5)
    finally:
        logger.info(f"{NEON['HEADING']}=== Pyrmethus Spell Concludes ==={NEON['RESET']}")
        if exchange: close_all_symbol_positions(exchange, config, "Spell Ending Sequence")
        save_persistent_state(force_heartbeat=True); trade_metrics.summary()
        logger.info(f"{NEON['COMMENT']}# Energies settle. Until next conjuring.{NEON['RESET']}")
        CONFIG.send_notification_method("Pyrmethus Offline", f"Spell concluded for {config.symbol}.")

if __name__ == "__main__":
    logger.info(f"{NEON['COMMENT']}# Pyrmethus prepares to breach veil to exchange realm...{NEON['RESET']}")
    exchange = None
    try:
        exchange_params = {
            'apiKey': CONFIG.api_key, 'secret': CONFIG.api_secret,
            'options': {'defaultType': 'swap', 'adjustForTimeDifference': True, 'brokerId': f'PYRMETHUS{PYRMETHUS_VERSION.replace(".","")}'},
            'enableRateLimit': True, 'recvWindow': CONFIG.default_recv_window
        }
        if CONFIG.PAPER_TRADING_MODE: exchange_params['urls'] = {'api': 'https://api-testnet.bybit.com'}
        exchange = ccxt.bybit(exchange_params)
        logger.info(f"Connecting to: {exchange.id} (CCXT: {ccxt.__version__})")
        markets = exchange.load_markets()
        if CONFIG.symbol not in markets:
            err_msg = f"Symbol {CONFIG.symbol} not found in {exchange.id} market runes."
            logger.critical(f"{NEON['CRITICAL']}{err_msg}{NEON['RESET']}")
            CONFIG.send_notification_method("Pyrmethus Startup Fail", err_msg); sys.exit(1)
        
        CONFIG.MARKET_INFO = markets[CONFIG.symbol]
        price_prec_raw = CONFIG.MARKET_INFO.get('precision', {}).get('price') # This is the tick size
        CONFIG.tick_size = safe_decimal_conversion(price_prec_raw) # Store for use
        if CONFIG.tick_size is None or CONFIG.tick_size <=0:
            logger.warning(f"{NEON['WARNING']}Could not determine valid tick size from market info. Using a small default if needed.{NEON['RESET']}")
            CONFIG.tick_size = Decimal("1e-8") # Fallback, adjust based on typical asset

        amount_prec_raw = CONFIG.MARKET_INFO.get('precision', {}).get('amount')
        logger.success(f"{NEON['SUCCESS']}Market runes for {CONFIG.symbol}: Price Tick {price_prec_raw}, Amount Step {amount_prec_raw}{NEON['RESET']}") # type: ignore [attr-defined]
        
        category = "linear" if CONFIG.symbol.endswith(":USDT") else None
        try:
            leverage_params = {'category': category} if category else {}
            logger.info(f"Setting leverage to {CONFIG.leverage}x for {CONFIG.symbol} (Category: {category or 'inferred'})...")
            response = exchange.set_leverage(CONFIG.leverage, CONFIG.symbol, params=leverage_params)
            logger.success(f"{NEON['SUCCESS']}Leverage for {CONFIG.symbol} set/confirmed. Response (part): {json.dumps(response, default=str)[:150]}...{NEON['RESET']}") # type: ignore [attr-defined]
        except Exception as e_lev:
            logger.warning(f"{NEON['WARNING']}Could not set leverage (may be pre-set or issue): {e_lev}{NEON['RESET']}")
            CONFIG.send_notification_method("Pyrmethus Leverage Warn", f"Leverage issue for {CONFIG.symbol}: {str(e_lev)[:60]}")
        logger.success(f"{NEON['SUCCESS']}Connected to {exchange.id} for {CONFIG.symbol}.{NEON['RESET']}") # type: ignore [attr-defined]
        CONFIG.send_notification_method("Pyrmethus Online", f"Connected to {exchange.id} for {CONFIG.symbol} @ {CONFIG.leverage}x.")
    except AttributeError as e_attr: logger.critical(f"{NEON['CRITICAL']}Exchange attribute error: {e_attr}. CCXT issue or invalid ID?{NEON['RESET']}"); sys.exit(1)
    except ccxt.AuthenticationError as e_auth: logger.critical(f"{NEON['CRITICAL']}Authentication failed! Check API keys: {e_auth}{NEON['RESET']}"); CONFIG.send_notification_method("Pyrmethus Auth Fail", f"API Auth Error."); sys.exit(1)
    except ccxt.NetworkError as e_net: logger.critical(f"{NEON['CRITICAL']}Network Error: {e_net}. Check connection/DNS.{NEON['RESET']}"); CONFIG.send_notification_method("Pyrmethus Network Error", f"Cannot connect: {str(e_net)[:80]}"); sys.exit(1)
    except ccxt.ExchangeError as e_exch: logger.critical(f"{NEON['CRITICAL']}Exchange API Error: {e_exch}. Check API/symbol/status.{NEON['RESET']}"); CONFIG.send_notification_method("Pyrmethus API Error", f"API issue: {str(e_exch)[:80]}"); sys.exit(1)
    except Exception as e_general:
        logger.critical(f"{NEON['CRITICAL']}Failed to init exchange/spell pre-requisites: {e_general}{NEON['RESET']}"); logger.debug(traceback.format_exc())
        CONFIG.send_notification_method("Pyrmethus Init Error", f"Exchange init failed: {str(e_general)[:80]}"); sys.exit(1)

    if exchange: main_loop(exchange, CONFIG)
    else: logger.critical(f"{NEON['CRITICAL']}Exchange not initialized. Spell aborted.{NEON['RESET']}"); sys.exit(1)

