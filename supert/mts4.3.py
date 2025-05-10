#!/usr/bin/env python3
# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v2.8.2 (Operational Infusion)
# Features implemented core trading logic, live market interaction,
# enhanced state management, and robust error handling.
# Previous Weave: v2.8.1 (Strategic Illumination & Termux Weave)

"""
High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 2.8.2 (Operational Infusion) - Enhanced by Pyrmethus

Dependencies & Setup (Termux):
1. Core Packages:
   pkg install python clang libcrypt libffi openssl rust termux-api
   (Rust and build essentials are often needed for cryptography dependencies)
2. Python Libraries:
   pip install ccxt pandas pandas-ta colorama python-dotenv retry pytz

WARNING: This script can execute live trades. Use with extreme caution.
Thoroughly test in a simulated environment or with minimal capital.
Pyrmethus bears no responsibility for financial outcomes.

Enhancements in v2.8.2 (by Pyrmethus):
- Integrated full implementations for Config helpers, Strategy classes, TradeMetrics,
  State Persistence, and Indicator Calculations from v2.8.1.
- Activated Termux notifications with v2.8.1 logic, adapted for v2.8.2 config.
- Ensured compatibility between TradeMetrics and new trading functions.
- Maintained core operational logic of v2.8.2: get_market_data, fetch_account_balance,
  calculate_position_size, place_risked_order, close_position_part.
- Retained activated main_loop with live trading decision-making.
- Kept Bybit V5 API focus for exchange initialization.
- Market info (precision, limits) loaded into Config.
- Enhanced _active_trade_parts structure handling in persistence.
- Maintained improved error handling with @retry and specific ccxt exceptions.
- Kept PAPER_TRADING_MODE in Config (informational).
- Preserved refined indicator calculation and column naming.
- Maintained comprehensive Termux notifications for critical events.
- Woven wizardly comments throughout for thematic consistency and clarity.
"""

# Standard Library Imports
import json
import logging
import os
import random  # For MockExchange if used, and unique IDs
import subprocess  # For Termux notifications
import sys
import time
import traceback
import uuid  # For unique part IDs
from abc import ABC, abstractmethod
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
    # Direct stderr write for pre-logging critical failure
    sys.stderr.write(
        f"\033[91mCRITICAL ERROR: Missing/Incompatible Essence: '{missing_pkg}'. Pyrmethus cannot weave this spell.\033[0m\n")
    sys.stderr.write(
        f"\033[91mPlease ensure all required libraries (runes) are installed and up to date.\033[0m\n")
    sys.stderr.write(
        f"\033[91mConsult the scrolls (README or comments) for 'pkg install' and 'pip install' incantations.\033[0m\n")
    sys.exit(1)

# --- Constants - The Unchanging Pillars of the Spell ---
# Updated version for the new weave
STATE_FILE_NAME = "pyrmethus_phoenix_state_v282.json"
STATE_FILE_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), STATE_FILE_NAME)
# Save state at least this often if the spell is active
HEARTBEAT_INTERVAL_SECONDS = 60
OHLCV_LIMIT = 200  # Number of candles to fetch for indicators

# --- Neon Color Palette (Enhanced as per request) - The Wizard's Aura ---
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
    "PNL_ZERO": Fore.YELLOW,  # For breakeven or zero PNL
    "SIDE_LONG": Fore.GREEN,
    "SIDE_SHORT": Fore.RED,
    "SIDE_FLAT": Fore.BLUE,
    "HEADING": Fore.MAGENTA + Style.BRIGHT,
    "SUBHEADING": Fore.CYAN + Style.BRIGHT,
    # For actions like "Placing order", "Closing position"
    "ACTION": Fore.YELLOW + Style.BRIGHT,
    "COMMENT": Fore.CYAN + Style.DIM,  # For wizardly comments within output
    "RESET": Style.RESET_ALL
}

# --- Initializations - Awakening the Spell ---
colorama_init(autoreset=True)  # Let the colors flow automatically
# Attempt to load .env file from the script's sacred ground
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if load_dotenv(dotenv_path=env_path):
    # Use a temporary logger for this specific message before full logger setup
    logging.getLogger("PreConfig").info(
        f"{NEON['INFO']}Secrets whispered from .env scroll: {NEON['VALUE']}{env_path}{NEON['RESET']}")
else:
    logging.getLogger("PreConfig").warning(
        f"{NEON['WARNING']}No .env scroll found at {NEON['VALUE']}{env_path}{NEON['WARNING']}. Relying on system environment variables or defaults.{NEON['RESET']}")
# Set precision for Decimal calculations, for the finest market etchings
getcontext().prec = 18

# --- Helper Functions - Minor Incantations ---


def safe_decimal_conversion(value: Any, default_if_error: Any = pd.NA) -> Union[Decimal, Any]:
    """
    Safely converts a value to Decimal. If conversion fails or value is NA-like,
    returns pd.NA or a specified default.
    This guards against the chaotic energies of malformed data.
    """
    if pd.isna(value):  # Handles None, np.nan, pd.NaT, pd.NA, and float NaN correctly
        return default_if_error
    try:
        # Using str(value) is important for floats to prevent precision issues with direct Decimal(float)
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        # Log this minor disturbance if needed, but often it's expected for missing data
        # logger.debug(f"Could not convert '{value}' (type: {type(value)}) to Decimal, returning default.")
        return default_if_error


def _format_for_log(value: Any, is_bool_trend: bool = False, precision: int = 4) -> str:
    """
    Formats values for logging, especially Decimals and trend booleans, making the Oracle's voice clearer.
    """
    if isinstance(value, Decimal):
        return f"{value:.{precision}f}"
    if is_bool_trend:
        if value is True:
            return "Up"
        if value is False:
            return "Down"
        return "Indeterminate"  # For pd.NA or None
    if pd.isna(value):  # pd.isna handles None as well
        return "N/A"
    return str(value)


def send_termux_notification(title: str, message: str, notification_id: int = 777) -> None:
    """Sends a notification using Termux API, a whisper on the digital wind."""
    # This function is now implemented, using CONFIG for enable/timeout settings.
    # It will be assigned to CONFIG.send_notification_method.
    # Note: This function relies on the CONFIG object being initialized.
    # For pre-config errors, direct subprocess calls might be needed or a simpler pre-config notification.
    if 'CONFIG' not in globals() or not CONFIG.enable_notifications:
        if 'CONFIG' in globals():  # Config is loaded but notifications disabled
            logging.getLogger("TermuxNotification").debug(
                "Termux notifications are disabled by configuration.")
        else:  # Config not yet loaded (e.g. very early startup error)
            logging.getLogger("TermuxNotification").warning(
                "Attempted Termux notification before CONFIG loaded or with notifications disabled.")
        return
    try:
        # Ensure strings are properly escaped for shell command
        safe_title = json.dumps(title)
        safe_message = json.dumps(message)

        command = [
            "termux-notification",
            "--title", safe_title,
            "--content", safe_message,
            "--id", str(notification_id),
        ]

        logging.getLogger("TermuxNotification").debug(
            f"{NEON['ACTION']}Attempting to send Termux notification: Title='{title}'{NEON['RESET']}")
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Use the specific notification timeout from CONFIG
        stdout, stderr = process.communicate(
            timeout=CONFIG.notification_timeout_seconds)

        if process.returncode == 0:
            logging.getLogger("TermuxNotification").info(
                f"{NEON['SUCCESS']}Termux notification '{title}' sent successfully.{NEON['RESET']}")
        else:
            err_msg = stderr.decode().strip() if stderr else "Unknown error"
            logging.getLogger("TermuxNotification").error(
                f"{NEON['ERROR']}Failed to send Termux notification '{title}'. Return code: {process.returncode}. Error: {err_msg}{NEON['RESET']}")
    except FileNotFoundError:
        logging.getLogger("TermuxNotification").error(
            f"{NEON['ERROR']}Termux API command 'termux-notification' not found. Is 'termux-api' package installed and accessible?{NEON['RESET']}")
    except subprocess.TimeoutExpired:
        logging.getLogger("TermuxNotification").error(
            f"{NEON['ERROR']}Termux notification command timed out for '{title}'.{NEON['RESET']}")
    except Exception as e:
        logging.getLogger("TermuxNotification").error(
            f"{NEON['ERROR']}Unexpected error sending Termux notification '{title}': {e}{NEON['RESET']}")
        logging.getLogger("TermuxNotification").debug(traceback.format_exc())


# --- Enums - The Sacred Glyphs ---
class StrategyName(str, Enum):
    DUAL_SUPERTREND_MOMENTUM = "DUAL_SUPERTREND_MOMENTUM"
    EHLERS_FISHER = "EHLERS_FISHER"


class VolatilityRegime(Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"


class OrderEntryType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"

# --- Configuration Class - The Spellbook's Core ---


class Config:
    """Loads and validates configuration parameters for Pyrmethus v2.8.2."""

    def __init__(self) -> None:
        _pre_logger = logging.getLogger("ConfigModule")
        _pre_logger.info(
            f"{NEON['HEADING']}--- Summoning Configuration Runes v2.8.2 ---{NEON['RESET']}")
        _pre_logger.info(
            f"{NEON['COMMENT']}# Pyrmethus attunes to the environment's whispers...{NEON['RESET']}")

        self.api_key: str = self._get_env(
            "BYBIT_API_KEY", required=True, secret=True)
        self.api_secret: str = self._get_env(
            "BYBIT_API_SECRET", required=True, secret=True)
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT")
        self.interval: str = self._get_env("INTERVAL", "1m")
        self.leverage: int = self._get_env("LEVERAGE", 10, cast_type=int)
        self.sleep_seconds: int = self._get_env(
            "SLEEP_SECONDS", 10, cast_type=int)
        self.strategy_name: StrategyName = StrategyName(self._get_env(
            "STRATEGY_NAME", StrategyName.DUAL_SUPERTREND_MOMENTUM.value).upper())
        # Forward declaration, the chosen spell form
        self.strategy_instance: 'TradingStrategy'
        self.risk_per_trade_percentage: Decimal = self._get_env(
            "RISK_PER_TRADE_PERCENTAGE", "0.005", cast_type=Decimal)  # 0.5% risk
        self.enable_dynamic_risk: bool = self._get_env(
            "ENABLE_DYNAMIC_RISK", "false", cast_type=bool)
        self.dynamic_risk_min_pct: Decimal = self._get_env(
            "DYNAMIC_RISK_MIN_PCT", "0.0025", cast_type=Decimal)
        self.dynamic_risk_max_pct: Decimal = self._get_env(
            "DYNAMIC_RISK_MAX_PCT", "0.01", cast_type=Decimal)
        self.dynamic_risk_perf_window: int = self._get_env(
            "DYNAMIC_RISK_PERF_WINDOW", 10, cast_type=int)
        self.max_order_usdt_amount: Decimal = self._get_env(
            "MAX_ORDER_USDT_AMOUNT", "1000.0", cast_type=Decimal)
        self.required_margin_buffer: Decimal = self._get_env(
            "REQUIRED_MARGIN_BUFFER", "1.05", cast_type=Decimal)
        self.max_account_margin_ratio: Decimal = self._get_env(
            "MAX_ACCOUNT_MARGIN_RATIO", "0.5", cast_type=Decimal)
        self.enable_max_drawdown_stop: bool = self._get_env(
            "ENABLE_MAX_DRAWDOWN_STOP", "true", cast_type=bool)
        self.max_drawdown_percent: Decimal = self._get_env(
            "MAX_DRAWDOWN_PERCENT", "0.05", cast_type=Decimal)  # 5% daily drawdown
        self.enable_time_based_stop: bool = self._get_env(
            "ENABLE_TIME_BASED_STOP", "false", cast_type=bool)
        self.max_trade_duration_seconds: int = self._get_env(
            "MAX_TRADE_DURATION_SECONDS", 3600, cast_type=int)
        self.enable_dynamic_atr_sl: bool = self._get_env(
            "ENABLE_DYNAMIC_ATR_SL", "true", cast_type=bool)
        self.atr_short_term_period: int = self._get_env(
            "ATR_SHORT_TERM_PERIOD", 7, cast_type=int)
        self.atr_long_term_period: int = self._get_env(
            "ATR_LONG_TERM_PERIOD", 50, cast_type=int)
        self.volatility_ratio_low_threshold: Decimal = self._get_env(
            "VOLATILITY_RATIO_LOW_THRESHOLD", "0.7", cast_type=Decimal)
        self.volatility_ratio_high_threshold: Decimal = self._get_env(
            "VOLATILITY_RATIO_HIGH_THRESHOLD", "1.5", cast_type=Decimal)
        self.atr_sl_multiplier_low_vol: Decimal = self._get_env(
            "ATR_SL_MULTIPLIER_LOW_VOL", "1.0", cast_type=Decimal)
        self.atr_sl_multiplier_normal_vol: Decimal = self._get_env(
            "ATR_SL_MULTIPLIER_NORMAL_VOL", "1.3", cast_type=Decimal)
        self.atr_sl_multiplier_high_vol: Decimal = self._get_env(
            "ATR_SL_MULTIPLIER_HIGH_VOL", "1.8", cast_type=Decimal)
        self.atr_stop_loss_multiplier: Decimal = self._get_env(
            "ATR_STOP_LOSS_MULTIPLIER", "1.5", cast_type=Decimal)  # Fallback
        self.enable_position_scaling: bool = self._get_env(
            "ENABLE_POSITION_SCALING", "false", cast_type=bool)
        self.max_scale_ins: int = self._get_env(
            "MAX_SCALE_INS", 0, cast_type=int)
        self.scale_in_risk_percentage: Decimal = self._get_env(
            "SCALE_IN_RISK_PERCENTAGE", "0.0025", cast_type=Decimal)
        self.min_profit_for_scale_in_atr: Decimal = self._get_env(
            "MIN_PROFIT_FOR_SCALE_IN_ATR", "1.0", cast_type=Decimal)
        self.trailing_stop_percentage: Decimal = self._get_env(
            "TRAILING_STOP_PERCENTAGE", "0.005", cast_type=Decimal)
        self.trailing_stop_activation_offset_percent: Decimal = self._get_env(
            "TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT", "0.001", cast_type=Decimal)
        self.entry_order_type: OrderEntryType = OrderEntryType(
            self._get_env("ENTRY_ORDER_TYPE", OrderEntryType.MARKET.value).upper())
        self.limit_order_offset_pips: int = self._get_env(
            "LIMIT_ORDER_OFFSET_PIPS", 2, cast_type=int)
        self.limit_order_fill_timeout_seconds: int = self._get_env(
            "LIMIT_ORDER_FILL_TIMEOUT_SECONDS", 20, cast_type=int)
        self.order_fill_timeout_seconds: int = self._get_env(
            "ORDER_FILL_TIMEOUT_SECONDS", 15, cast_type=int)
        self.st_atr_length: int = self._get_env(
            "ST_ATR_LENGTH", 10, cast_type=int)
        self.st_multiplier: Decimal = self._get_env(
            "ST_MULTIPLIER", "2.0", cast_type=Decimal)
        self.confirm_st_atr_length: int = self._get_env(
            "CONFIRM_ST_ATR_LENGTH", 20, cast_type=int)
        self.confirm_st_multiplier: Decimal = self._get_env(
            "CONFIRM_ST_MULTIPLIER", "3.0", cast_type=Decimal)
        self.momentum_period: int = self._get_env(
            "MOMENTUM_PERIOD", 14, cast_type=int)
        self.momentum_threshold: Decimal = self._get_env(
            "MOMENTUM_THRESHOLD", "0", cast_type=Decimal)
        self.ehlers_fisher_length: int = self._get_env(
            "EHLERS_FISHER_LENGTH", 10, cast_type=int)
        self.ehlers_fisher_signal_length: int = self._get_env(
            "EHLERS_FISHER_SIGNAL_LENGTH", 1, cast_type=int)
        self.atr_calculation_period: int = self.atr_short_term_period if self.enable_dynamic_atr_sl else self._get_env(
            "ATR_CALCULATION_PERIOD", 14, cast_type=int)
        self.enable_notifications: bool = self._get_env(
            "ENABLE_NOTIFICATIONS", "true", cast_type=bool)
        self.notification_timeout_seconds: int = self._get_env(
            "NOTIFICATION_TIMEOUT_SECONDS", 10, cast_type=int)
        self.default_recv_window: int = self._get_env(
            "DEFAULT_RECV_WINDOW", 13000, cast_type=int)
        self.side_buy: str = "buy"  # Exchange verb for buying
        self.side_sell: str = "sell"  # Exchange verb for selling
        self.pos_long: str = "Long"  # Internal representation of a long position
        self.pos_short: str = "Short"  # Internal representation of a short position
        self.pos_none: str = "None"  # Internal representation of no position
        self.usdt_symbol: str = "USDT"
        self.retry_count: int = 5
        self.retry_delay_seconds: int = 5
        self.api_fetch_limit_buffer: int = 20
        self.position_qty_epsilon: Decimal = Decimal(
            "1e-9")  # Smallest quantity considered non-zero
        self.post_close_delay_seconds: int = 3  # Wait after closing positions
        self.MARKET_INFO: Optional[Dict[str, Any]] = None  # Loaded at startup
        self.PAPER_TRADING_MODE: bool = self._get_env(
            "PAPER_TRADING_MODE", "false", cast_type=bool)
        # Assign the notification function
        self.send_notification_method = send_termux_notification

        self._validate_parameters()  # Scrutinizing the runes for flaws
        _pre_logger.info(
            f"{NEON['HEADING']}--- Configuration Runes v2.8.2 Summoned and Verified ---{NEON['RESET']}")
        _pre_logger.info(
            f"{NEON['COMMENT']}# The chosen path: {NEON['STRATEGY']}{self.strategy_name.value}{NEON['RESET']}")

    def _get_env(self, key: str, default: Any = None, cast_type: type = str, required: bool = False, color: str = NEON["PARAM"], secret: bool = False) -> Any:
        """Fetches a configuration rune (environment variable), applying defaults and casting as needed."""
        _logger = logging.getLogger("ConfigModule._get_env")
        value_str = os.getenv(key)
        source = "Env Var"
        value_to_cast: Any = None
        display_value = "********" if secret and value_str is not None else value_str

        if value_str is None:
            if required:
                _logger.critical(
                    f"{NEON['CRITICAL']}CRITICAL: Required config rune '{key}' not found in environment or .env scroll. Pyrmethus cannot proceed without this essence.{NEON['RESET']}")
                raise ValueError(
                    f"Required environment variable '{key}' not set.")
            _logger.debug(
                f"{color}Config Rune {NEON['VALUE']}'{key}'{color}: Not Found. Using Default: '{NEON['VALUE']}{default}{color}'{NEON['RESET']}")
            value_to_cast = default
            source = "Default"
        else:
            _logger.debug(
                f"{color}Config Rune {NEON['VALUE']}'{key}'{color}: Found Env Value: '{NEON['VALUE']}{display_value}{color}'{NEON['RESET']}")
            value_to_cast = value_str

        if value_to_cast is None:
            if required:
                _logger.critical(
                    f"{NEON['CRITICAL']}CRITICAL: Required config rune '{key}' resolved to None after checking environment and defaults.{NEON['RESET']}")
                raise ValueError(
                    f"Required environment variable '{key}' resolved to None.")
            _logger.debug(
                f"{color}Config Rune {NEON['VALUE']}'{key}'{color}: Final value is None (not required).{NEON['RESET']}")
            return None

        final_value: Any
        try:
            raw_value_str_for_cast = str(value_to_cast)
            if cast_type == bool:
                final_value = raw_value_str_for_cast.lower() in [
                    "true", "1", "yes", "y"]
            elif cast_type == Decimal:
                final_value = Decimal(raw_value_str_for_cast)
            elif cast_type == int:
                # Cast to Decimal first for "10.0"
                final_value = int(Decimal(raw_value_str_for_cast))
            elif cast_type == float:
                final_value = float(raw_value_str_for_cast)
            elif cast_type == str:
                final_value = raw_value_str_for_cast
            else:
                _logger.warning(
                    f"{NEON['WARNING']}Unsupported cast_type '{cast_type.__name__}' for key '{key}'. Returning raw string: '{raw_value_str_for_cast}'.{NEON['RESET']}")
                final_value = raw_value_str_for_cast
        except (ValueError, TypeError, InvalidOperation) as e:
            _logger.error(
                f"{NEON['ERROR']}Invalid type/value for '{key}': '{value_to_cast}' (Source: {source}). Expected {cast_type.__name__}. Err: {e}. Using Default: '{default}'.{NEON['RESET']}")
            if default is None:
                if required:
                    _logger.critical(
                        f"{NEON['CRITICAL']}CRITICAL: Failed cast for required key '{key}', default is None.{NEON['RESET']}")
                    raise ValueError(
                        f"Required env var '{key}' failed casting, no valid default.")
                else:
                    _logger.warning(
                        f"{NEON['WARNING']}Cast fail for optional '{key}', default is None. Final: None{NEON['RESET']}")
                    return None
            else:
                source = "Default (Fallback)"
                _logger.debug(
                    f"Casting fallback default '{default}' for '{key}' to {cast_type.__name__}")
                try:
                    default_str_for_cast = str(default)
                    if cast_type == bool:
                        final_value = default_str_for_cast.lower() in [
                            "true", "1", "yes", "y"]
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
                    _logger.warning(
                        f"{NEON['WARNING']}Used casted default for {key}: '{NEON['VALUE']}{final_value}{NEON['WARNING']}'{NEON['RESET']}")
                except (ValueError, TypeError, InvalidOperation) as e_default:
                    _logger.critical(
                        f"{NEON['CRITICAL']}CRITICAL: Cast fail for value AND default for '{key}'. Original val: '{value_to_cast}', Default val: '{default}'. Err: {e_default}{NEON['RESET']}")
                    raise ValueError(
                        f"Config error: Cannot cast value or default for '{key}'.")

        display_final_value = "********" if secret else final_value
        _logger.debug(
            f"{color}Final value for {NEON['VALUE']}'{key}'{color}: {NEON['VALUE']}{display_final_value}{color} (Type: {type(final_value).__name__}, Source: {source}){NEON['RESET']}")
        return final_value

    def _validate_parameters(self) -> None:
        """Performs basic validation of critical configuration parameters. A necessary rite of scrutiny."""
        _logger = logging.getLogger("ConfigModule._validate")
        errors = []
        if not (0 < self.risk_per_trade_percentage < 1):
            errors.append(
                f"RISK_PER_TRADE_PERCENTAGE ({self.risk_per_trade_percentage}) must be a potion strength between 0 and 1 (exclusive).")
        if self.leverage < 1:
            errors.append(
                f"LEVERAGE ({self.leverage}) must be at least 1, to amplify the spell's power.")
        if self.max_scale_ins < 0:
            errors.append(
                f"MAX_SCALE_INS ({self.max_scale_ins}) cannot be a negative enchantment.")
        if self.trailing_stop_percentage < 0:
            errors.append(
                f"TRAILING_STOP_PERCENTAGE ({self.trailing_stop_percentage}) cannot be negative.")
        if self.trailing_stop_activation_offset_percent < 0:
            errors.append(
                f"TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT ({self.trailing_stop_activation_offset_percent}) cannot be negative.")

        if self.st_atr_length <= 0 or self.confirm_st_atr_length <= 0 or self.momentum_period <= 0:
            errors.append(
                "Strategy periods (ST_ATR_LENGTH, CONFIRM_ST_ATR_LENGTH, MOMENTUM_PERIOD) must be positive integers.")
        if self.ehlers_fisher_length <= 0 or self.ehlers_fisher_signal_length <= 0:
            errors.append(
                "Ehlers Fisher lengths (EHLERS_FISHER_LENGTH, EHLERS_FISHER_SIGNAL_LENGTH) must be positive integers.")

        if errors:
            error_message = f"Configuration spellcrafting failed with {len(errors)} flaws:\n" + "\n".join([
                f"  - {e}" for e in errors])
            _logger.critical(
                f"{NEON['CRITICAL']}{error_message}{NEON['RESET']}")
            raise ValueError(error_message)


# --- Logger Setup - The Oracle's Voice ---
LOGGING_LEVEL: int = (logging.DEBUG if os.getenv(
    "DEBUG", "false").lower() == "true" else logging.INFO)
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
log_file_name = f"{logs_dir}/pyrmethus_spell_v282_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=LOGGING_LEVEL, format="%(asctime)s [%(levelname)-8s] %(name)-30s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file_name)])
logger: logging.Logger = logging.getLogger("PyrmethusCore")
SUCCESS_LEVEL: int = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(SUCCESS_LEVEL):
        # type: ignore[attr-defined]
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


logging.Logger.success = log_success  # type: ignore[attr-defined]

if sys.stdout.isatty():  # Only add colors if output is a TTY
    stream_handler = next((h for h in logging.getLogger(
    ).handlers if isinstance(h, logging.StreamHandler)), None)
    if stream_handler:
        # Create a new formatter for colored output, matching basicConfig's format string
        colored_formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)-8s] %(name)-30s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        stream_handler.setFormatter(colored_formatter)

    # Apply colored level names
    # Ensure level names are consistently 8 characters for alignment
    for level, color_code in [
        (logging.DEBUG, NEON['DEBUG']),
        (logging.INFO, NEON['INFO']),
        (SUCCESS_LEVEL, NEON['SUCCESS']),
        (logging.WARNING, NEON['WARNING']),
        (logging.ERROR, NEON['ERROR']),
        (logging.CRITICAL, NEON['CRITICAL'])
    ]:
        level_name = logging.getLevelName(level)
        # Remove existing color codes if re-running this block (e.g., in interactive session)
        if '\033' in level_name:
            base_name_parts = level_name.split('\033')
            # Find the part that is likely the original level name (not a color code part)
            original_name = ""
            for part in base_name_parts:
                # Simple check for ANSI code
                if not (part.startswith('[') and part.endswith('m')):
                    original_name = part
                    break
            if not original_name:  # Fallback if parsing fails
                original_name = level_name.split(
                    'm')[-1] if 'm' in level_name else level_name

            level_name = original_name.strip()

        logging.addLevelName(
            level, f"{color_code}{level_name.ljust(8)}{NEON['RESET']}")


# --- Global Objects - Instantiated Arcana ---
try:
    CONFIG = Config()
except ValueError as config_error:
    logging.getLogger("PyrmethusCore").critical(
        f"{NEON['CRITICAL']}Configuration loading failed. Error: {config_error}{NEON['RESET']}")
    if 'termux-api' in os.getenv('PATH', ''):
        try:
            send_termux_notification(
                "Pyrmethus Startup Failure", f"Config Error: {str(config_error)[:200]}")
        except Exception:
            pass
    sys.exit(1)
except Exception as general_config_error:
    logging.getLogger("PyrmethusCore").critical(
        f"{NEON['CRITICAL']}Unexpected critical error during configuration: {general_config_error}{NEON['RESET']}")
    logging.getLogger("PyrmethusCore").debug(traceback.format_exc())
    if 'termux-api' in os.getenv('PATH', ''):
        try:
            send_termux_notification("Pyrmethus Critical Failure",
                                     f"Unexpected Config Error: {str(general_config_error)[:200]}")
        except Exception:
            pass
    sys.exit(1)

# --- Trading Strategy Abstract Base Class & Implementations - The Schools of Magic ---


class TradingStrategy(ABC):
    """The archetypal form for all trading spells."""

    def __init__(self, config: Config, df_columns: Optional[List[str]] = None):
        self.config = config
        self.logger = logging.getLogger(f"Strategy.{self.__class__.__name__}")
        self.required_columns = df_columns if df_columns else []
        self.logger.info(
            f"{NEON['STRATEGY']}Strategy Form '{self.__class__.__name__}' materializing...{NEON['RESET']}")

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates trading signals based on the input DataFrame, which holds the market's omens.
        Returns a dictionary of signals: enter_long, enter_short, exit_long, exit_short, exit_reason.
        """
        pass

    def _validate_df(self, df: pd.DataFrame, min_rows: int = 2) -> bool:
        """
        Validates the input DataFrame (the scroll of market data) for common flaws.
        Returns True if the scroll is fit for divination, False otherwise.
        """
        if df is None or df.empty or len(df) < min_rows:
            self.logger.debug(
                f"Insufficient market whispers (Rows: {len(df) if df is not None else 0}, Min Required: {min_rows}). Awaiting clearer omens.")
            return False
        if self.required_columns and not all(col in df.columns for col in self.required_columns):
            missing_cols = [
                col for col in self.required_columns if col not in df.columns]
            self.logger.warning(
                f"{NEON['WARNING']}Market scroll is missing required runes for this strategy: {NEON['VALUE']}{missing_cols}{NEON['WARNING']}. Cannot divine signals.{NEON['RESET']}")
            return False

        if self.required_columns and not df.empty:
            last_row_values = df.iloc[-1][self.required_columns]
            if last_row_values.isnull().any():
                nan_cols_last_row = last_row_values[last_row_values.isnull(
                )].index.tolist()
                self.logger.debug(
                    f"Faint runes (NaNs) in last candle for critical columns: {nan_cols_last_row}. Signals may be unreliable or delayed.")
        return True

    def _get_default_signals(self) -> Dict[str, Any]:
        """Returns a default signal dictionary (all signals false), when the omens are unclear."""
        return {
            "enter_long": False, "enter_short": False,
            "exit_long": False, "exit_short": False,
            "exit_reason": "Default Signal - Awaiting True Omens"
        }


class DualSupertrendMomentumStrategy(TradingStrategy):
    def __init__(self, config: Config):
        super().__init__(config, df_columns=[
            # Ensure these are generated
            "st_long_flip", "st_short_flip", "confirm_trend", "momentum"])

    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        signals = self._get_default_signals()
        min_rows_needed = max(self.config.st_atr_length,
                              self.config.confirm_st_atr_length, self.config.momentum_period) + 10
        if not self._validate_df(df, min_rows=min_rows_needed):
            return signals

        last = df.iloc[-1]
        primary_long_flip = last.get("st_long_flip", False)
        primary_short_flip = last.get("st_short_flip", False)
        # Can be True, False, pd.NA
        confirm_is_up = last.get("confirm_trend", pd.NA)
        momentum_val_orig = last.get("momentum", pd.NA)
        momentum_val = safe_decimal_conversion(momentum_val_orig, pd.NA)

        if pd.isna(confirm_is_up) or pd.isna(momentum_val):
            self.logger.debug(
                f"Confirmation ST Trend ({_format_for_log(confirm_is_up, is_bool_trend=True)}) or Momentum ({_format_for_log(momentum_val)}) is NA. No signal.")
            return signals

        # Entry signals
        if primary_long_flip and confirm_is_up is True and momentum_val > self.config.momentum_threshold:
            signals["enter_long"] = True
            self.logger.info(
                f"{NEON['SIDE_LONG']}DualST+Mom Signal: LONG Entry - ST Flip, Confirm Up, Mom ({_format_for_log(momentum_val)}) > {_format_for_log(self.config.momentum_threshold)}{NEON['RESET']}")
        elif primary_short_flip and confirm_is_up is False and momentum_val < -self.config.momentum_threshold:
            signals["enter_short"] = True
            self.logger.info(
                f"{NEON['SIDE_SHORT']}DualST+Mom Signal: SHORT Entry - ST Flip, Confirm Down, Mom ({_format_for_log(momentum_val)}) < -{_format_for_log(self.config.momentum_threshold)}{NEON['RESET']}")

        # Exit signals (based on primary ST flip)
        if primary_short_flip:
            signals["exit_long"] = True
            signals["exit_reason"] = "Primary SuperTrend Flipped Short"
            self.logger.info(
                f"{NEON['ACTION']}DualST+Mom Signal: EXIT LONG - Primary ST Flipped Short{NEON['RESET']}")
        if primary_long_flip:
            signals["exit_short"] = True
            signals["exit_reason"] = "Primary SuperTrend Flipped Long"
            self.logger.info(
                f"{NEON['ACTION']}DualST+Mom Signal: EXIT SHORT - Primary ST Flipped Long{NEON['RESET']}")
        return signals


class EhlersFisherStrategy(TradingStrategy):
    def __init__(self, config: Config):
        super().__init__(config, df_columns=["ehlers_fisher", "ehlers_signal"])

    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        signals = self._get_default_signals()
        min_rows_needed = self.config.ehlers_fisher_length + \
            self.config.ehlers_fisher_signal_length + 5
        # Need prev and current
        if not self._validate_df(df, min_rows=min_rows_needed) or len(df) < 2:
            return signals

        last = df.iloc[-1]
        prev = df.iloc[-2]
        fisher_now = safe_decimal_conversion(last.get("ehlers_fisher"), pd.NA)
        signal_now = safe_decimal_conversion(last.get("ehlers_signal"), pd.NA)
        fisher_prev = safe_decimal_conversion(prev.get("ehlers_fisher"), pd.NA)
        signal_prev = safe_decimal_conversion(prev.get("ehlers_signal"), pd.NA)

        if pd.isna(fisher_now) or pd.isna(signal_now) or pd.isna(fisher_prev) or pd.isna(signal_prev):
            self.logger.debug(
                f"Ehlers Fisher or Signal rune is NA. Fisher: {_format_for_log(fisher_now)}, Signal: {_format_for_log(signal_now)}. No signal.")
            return signals

        # Entry Long: Fisher crosses above Signal
        if fisher_prev <= signal_prev and fisher_now > signal_now:
            signals["enter_long"] = True
            self.logger.info(
                f"{NEON['SIDE_LONG']}EhlersFisher Signal: LONG Entry - Fisher ({_format_for_log(fisher_now)}) crossed ABOVE Signal ({_format_for_log(signal_now)}){NEON['RESET']}")
        # Entry Short: Fisher crosses below Signal
        elif fisher_prev >= signal_prev and fisher_now < signal_now:
            signals["enter_short"] = True
            self.logger.info(
                f"{NEON['SIDE_SHORT']}EhlersFisher Signal: SHORT Entry - Fisher ({_format_for_log(fisher_now)}) crossed BELOW Signal ({_format_for_log(signal_now)}){NEON['RESET']}")

        # Exit Long: Fisher crosses below Signal
        if fisher_prev >= signal_prev and fisher_now < signal_now:
            signals["exit_long"] = True
            signals["exit_reason"] = "Ehlers Fisher crossed BELOW Signal"
            self.logger.info(
                f"{NEON['ACTION']}EhlersFisher Signal: EXIT LONG - Fisher ({_format_for_log(fisher_now)}) crossed BELOW Signal ({_format_for_log(signal_now)}){NEON['RESET']}")
        # Exit Short: Fisher crosses above Signal
        elif fisher_prev <= signal_prev and fisher_now > signal_now:
            signals["exit_short"] = True
            signals["exit_reason"] = "Ehlers Fisher crossed ABOVE Signal"
            self.logger.info(
                f"{NEON['ACTION']}EhlersFisher Signal: EXIT SHORT - Fisher ({_format_for_log(fisher_now)}) crossed ABOVE Signal ({_format_for_log(signal_now)}){NEON['RESET']}")
        return signals


strategy_map: Dict[StrategyName, Type[TradingStrategy]] = {
    StrategyName.DUAL_SUPERTREND_MOMENTUM: DualSupertrendMomentumStrategy, StrategyName.EHLERS_FISHER: EhlersFisherStrategy}
StrategyClass = strategy_map.get(CONFIG.strategy_name)
if StrategyClass:
    CONFIG.strategy_instance = StrategyClass(CONFIG)
    logger.success(  # type: ignore [attr-defined]
        f"{NEON['SUCCESS']}Strategy '{NEON['STRATEGY']}{CONFIG.strategy_name.value}{NEON['SUCCESS']}' invoked.{NEON['RESET']}")
else:
    err_msg = f"Failed to init strategy '{CONFIG.strategy_name.value}'. Unknown spell form."
    logger.critical(f"{NEON['CRITICAL']}{err_msg}{NEON['RESET']}")
    CONFIG.send_notification_method("Pyrmethus Critical Error", err_msg)
    sys.exit(1)

# --- Trade Metrics Tracking - The Grand Ledger of Deeds ---


class TradeMetrics:
    def __init__(self):
        self.trades: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("TradeMetrics")
        self.initial_equity: Optional[Decimal] = None
        self.daily_start_equity: Optional[Decimal] = None
        self.last_daily_reset_day: Optional[int] = None
        self.logger.info(
            f"{NEON['INFO']}TradeMetrics Ledger opened, ready to chronicle deeds.{NEON['RESET']}")

    def set_initial_equity(self, equity: Decimal):
        if self.initial_equity is None:
            self.initial_equity = equity
            self.logger.info(
                f"{NEON['INFO']}Initial Session Equity rune inscribed: {NEON['VALUE']}{equity:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}")

        today = datetime.now(pytz.utc).day
        if self.last_daily_reset_day != today or self.daily_start_equity is None:
            self.daily_start_equity = equity
            self.last_daily_reset_day = today
            self.logger.info(
                f"{NEON['INFO']}Daily Equity Ward reset. Dawn Equity: {NEON['VALUE']}{equity:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}")

    def check_drawdown(self, current_equity: Decimal) -> Tuple[bool, str]:
        if not CONFIG.enable_max_drawdown_stop or self.daily_start_equity is None or self.daily_start_equity <= 0:
            return False, ""

        drawdown = self.daily_start_equity - current_equity
        drawdown_pct = (
            drawdown / self.daily_start_equity) if self.daily_start_equity > 0 else Decimal(0)

        if drawdown_pct >= CONFIG.max_drawdown_percent:
            reason = f"Max daily drawdown breached ({NEON['PNL_NEG']}{drawdown_pct:.2%}{NEON['RESET']} >= {NEON['VALUE']}{CONFIG.max_drawdown_percent:.2%}{NEON['RESET']})"
            self.logger.warning(f"{NEON['WARNING']}{reason}.{NEON['RESET']}")
            CONFIG.send_notification_method(
                "Pyrmethus: Max Drawdown Hit!", f"Drawdown: {drawdown_pct:.2%}. Trading halted.")
            return True, reason
        return False, ""

    def log_trade(self, symbol: str, side: str, entry_price: Decimal, exit_price: Decimal, qty: Decimal,
                  entry_time_ms: int, exit_time_ms: int, reason: str, part_id: str, pnl_str: str,
                  scale_order_id: Optional[str] = None,
                  mae: Optional[Decimal] = None, mfe: Optional[Decimal] = None):
        """Scribes a completed trade part into the Grand Ledger, using provided PNL string."""
        if not all([isinstance(entry_price, Decimal) and entry_price > 0,
                    isinstance(exit_price, Decimal) and exit_price > 0,
                    isinstance(qty, Decimal) and qty > 0,
                    isinstance(entry_time_ms, int) and entry_time_ms > 0,
                    isinstance(exit_time_ms, int) and exit_time_ms > 0]):
            self.logger.warning(
                f"{NEON['WARNING']}Trade log skipped due to flawed parameters for Part ID: {part_id}. Entry: {entry_price}, Exit: {exit_price}, Qty: {qty}, EntryTime: {entry_time_ms}, ExitTime: {exit_time_ms}{NEON['RESET']}")
            return

        profit = safe_decimal_conversion(pnl_str, Decimal(0))
        entry_dt_utc = datetime.fromtimestamp(
            entry_time_ms / 1000, tz=pytz.utc)
        exit_dt_utc = datetime.fromtimestamp(exit_time_ms / 1000, tz=pytz.utc)
        duration_seconds = (exit_dt_utc - entry_dt_utc).total_seconds()
        trade_type = "Scale-In" if scale_order_id else "Part"

        self.trades.append({
            "symbol": symbol, "side": side,
            "entry_price_str": str(entry_price), "exit_price_str": str(exit_price),
            "qty_str": str(qty), "profit_str": str(profit),
            "entry_time_iso": entry_dt_utc.isoformat(), "exit_time_iso": exit_dt_utc.isoformat(),
            "duration_seconds": duration_seconds, "exit_reason": reason, "type": trade_type, "part_id": part_id,
            "scale_order_id": scale_order_id,
            "mae_str": str(mae) if mae is not None else None,
            "mfe_str": str(mfe) if mfe is not None else None
        })

        pnl_color = NEON["PNL_POS"] if profit > 0 else (
            NEON["PNL_NEG"] if profit < 0 else NEON["PNL_ZERO"])
        # type: ignore [attr-defined]
        self.logger.success(
            f"{NEON['HEADING']}Trade Chronicle ({trade_type}:{part_id}): "
            f"{side.upper()} {NEON['QTY']}{qty}{NEON['RESET']} {symbol.split('/')[0]} | "
            f"P/L: {pnl_color}{profit:.2f} {CONFIG.usdt_symbol}{NEON['RESET']} | Reason: {reason}")

    def summary(self) -> str:
        if not self.trades:
            return f"{NEON['INFO']}The Grand Ledger is empty. No deeds chronicled yet.{NEON['RESET']}"

        total_trades = len(self.trades)
        profits = [Decimal(t["profit_str"]) for t in self.trades]
        wins = sum(1 for p in profits if p > 0)
        losses = sum(1 for p in profits if p < 0)
        breakeven = total_trades - wins - losses
        win_rate = (Decimal(wins) / Decimal(total_trades)) * \
            Decimal(100) if total_trades > 0 else Decimal(0)
        total_profit = sum(profits)
        avg_profit = total_profit / \
            Decimal(total_trades) if total_trades > 0 else Decimal(0)

        summary_str = (
            f"\n{NEON['HEADING']}--- Pyrmethus Trade Metrics Summary (v2.8.2) ---{NEON['RESET']}\n"
            f"Total Trade Parts Chronicled: {NEON['VALUE']}{total_trades}{NEON['RESET']}\n"
            f"  Victories (Wins): {NEON['PNL_POS']}{wins}{NEON['RESET']}, Defeats (Losses): {NEON['PNL_NEG']}{losses}{NEON['RESET']}, Stalemates (Breakeven): {NEON['PNL_ZERO']}{breakeven}{NEON['RESET']}\n"
            f"Victory Rate: {NEON['VALUE']}{win_rate:.2f}%{NEON['RESET']}\n"
            f"Total Spoils (P/L): {(NEON['PNL_POS'] if total_profit > 0 else (NEON['PNL_NEG'] if total_profit < 0 else NEON['PNL_ZERO']))}{total_profit:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}\n"
            f"Average Spoils per Part: {(NEON['PNL_POS'] if avg_profit > 0 else (NEON['PNL_NEG'] if avg_profit < 0 else NEON['PNL_ZERO']))}{avg_profit:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}\n"
        )
        current_equity_approx = Decimal(0)  # Initialize for wider scope
        if self.initial_equity is not None:
            current_equity_approx = self.initial_equity + total_profit
            overall_pnl_pct = (total_profit / self.initial_equity) * \
                100 if self.initial_equity > Decimal(0) else Decimal(0)
            summary_str += f"Initial Session Treasury: {NEON['VALUE']}{self.initial_equity:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}\n"
            summary_str += f"Approx. Current Treasury: {NEON['VALUE']}{current_equity_approx:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}\n"
            summary_str += f"Overall Session Spoils %: {(NEON['PNL_POS'] if overall_pnl_pct > 0 else (NEON['PNL_NEG'] if overall_pnl_pct < 0 else NEON['PNL_ZERO']))}{overall_pnl_pct:.2f}%{NEON['RESET']}\n"

        if self.daily_start_equity is not None:
            # Ensure current_equity_approx is based on initial_equity if available, otherwise can't calculate daily PNL accurately from total_profit
            daily_pnl_base = current_equity_approx if self.initial_equity is not None else self.daily_start_equity + \
                total_profit  # Fallback if initial_equity was somehow None but daily_start_equity is set
            daily_pnl = daily_pnl_base - self.daily_start_equity
            daily_pnl_pct = (daily_pnl / self.daily_start_equity) * \
                100 if self.daily_start_equity > Decimal(0) else Decimal(0)
            summary_str += f"Daily Start Treasury: {NEON['VALUE']}{self.daily_start_equity:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}\n"
            summary_str += f"Approx. Daily P/L: {(NEON['PNL_POS'] if daily_pnl > 0 else (NEON['PNL_NEG'] if daily_pnl < 0 else NEON['PNL_ZERO']))}{daily_pnl:.2f} {CONFIG.usdt_symbol} ({daily_pnl_pct:.2f}%){NEON['RESET']}\n"

        summary_str += f"{NEON['HEADING']}--- End of Ledger Reading ---{NEON['RESET']}"
        self.logger.info(summary_str)
        return summary_str

    def get_serializable_trades(
        self) -> List[Dict[str, Any]]: return self.trades

    def load_trades_from_list(self, trades_list: List[Dict[str, Any]]):
        self.trades = []
        for trade_data in trades_list:
            self.trades.append(trade_data)
        self.logger.info(
            f"{NEON['INFO']}TradeMetrics: Re-inked {NEON['VALUE']}{len(self.trades)}{NEON['INFO']} trade sagas from the Phoenix scroll.{NEON['RESET']}")


trade_metrics = TradeMetrics()
_active_trade_parts: List[Dict[str, Any]] = []
_last_heartbeat_save_time: float = 0.0

# --- State Persistence Functions (Phoenix Feather) ---


def save_persistent_state(force_heartbeat: bool = False) -> None:
    """Scribes the current state of Pyrmethus's essence onto the Phoenix scroll for later reawakening."""
    global _active_trade_parts, trade_metrics, _last_heartbeat_save_time
    now = time.time()
    should_save = force_heartbeat or \
        (_active_trade_parts and (now - _last_heartbeat_save_time > HEARTBEAT_INTERVAL_SECONDS)) or \
        (not _active_trade_parts and (now - _last_heartbeat_save_time >
         HEARTBEAT_INTERVAL_SECONDS * 5))  # Less frequent if idle

    if not should_save:
        return

    logger.debug(
        f"{NEON['COMMENT']}# Phoenix Feather prepares to scribe memories... (Force: {force_heartbeat}){NEON['RESET']}")
    try:
        serializable_active_parts = []
        for part in _active_trade_parts:
            p_copy = part.copy()
            for key, value in p_copy.items():
                if isinstance(value, Decimal):
                    p_copy[key] = str(value)
            serializable_active_parts.append(p_copy)

        state_data = {
            "pyrmethus_version": "2.8.2",
            "timestamp_utc_iso": datetime.now(pytz.utc).isoformat(),
            "active_trade_parts": serializable_active_parts,
            "trade_metrics_trades": trade_metrics.get_serializable_trades(),
            "config_symbol": CONFIG.symbol,
            "config_strategy": CONFIG.strategy_name.value,
            "initial_equity_str": str(trade_metrics.initial_equity) if trade_metrics.initial_equity is not None else None,
            "daily_start_equity_str": str(trade_metrics.daily_start_equity) if trade_metrics.daily_start_equity is not None else None,
            "last_daily_reset_day": trade_metrics.last_daily_reset_day
        }
        temp_file_path = STATE_FILE_PATH + ".tmp_scroll"
        with open(temp_file_path, 'w') as f:
            json.dump(state_data, f, indent=4)
        os.replace(temp_file_path, STATE_FILE_PATH)
        _last_heartbeat_save_time = now

        log_level = logging.INFO if force_heartbeat or _active_trade_parts else logging.DEBUG
        logger.log(
            log_level, f"{NEON['SUCCESS']}Phoenix Feather: State memories scribed to '{STATE_FILE_NAME}'. Active parts: {len(_active_trade_parts)}.{NEON['RESET']}")
    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}Phoenix Feather Error: Failed to scribe state: {e}{NEON['RESET']}")
        logger.debug(traceback.format_exc())


def load_persistent_state() -> bool:
    """Attempts to reawaken Pyrmethus's essence from a previously scribed Phoenix scroll."""
    global _active_trade_parts, trade_metrics
    logger.info(
        f"{NEON['COMMENT']}# Phoenix Feather seeks past memories from '{STATE_FILE_PATH}'...{NEON['RESET']}")
    if not os.path.exists(STATE_FILE_PATH):
        logger.info(
            f"{NEON['INFO']}Phoenix Feather: No previous scroll found. Starting with a fresh essence.{NEON['RESET']}")
        return False
    try:
        with open(STATE_FILE_PATH, 'r') as f:
            state_data = json.load(f)

        saved_version = state_data.get("pyrmethus_version", "unknown")
        if saved_version != "2.8.2":  # Current version
            logger.warning(
                f"{NEON['WARNING']}Phoenix Feather: Scroll version '{saved_version}' differs from current '2.8.2'. Caution advised during reawakening.{NEON['RESET']}")

        if state_data.get("config_symbol") != CONFIG.symbol or \
           state_data.get("config_strategy") != CONFIG.strategy_name.value:
            logger.warning(f"{NEON['WARNING']}Phoenix Scroll sigils mismatch! Current spell ({CONFIG.symbol}/{CONFIG.strategy_name.value}) differs from scroll ({state_data.get('config_symbol')}/{state_data.get('config_strategy')}). Ignoring old scroll to prevent chaos.{NEON['RESET']}")
            archive_name = f"{STATE_FILE_PATH}.archived_{time.strftime('%Y%m%d%H%M%S')}"
            try:
                os.rename(STATE_FILE_PATH, archive_name)
                logger.info(f"Old scroll archived to: {archive_name}")
            except OSError as e_mv:
                logger.error(f"Error archiving mismatched scroll: {e_mv}")
            return False

        _active_trade_parts.clear()
        loaded_parts_raw = state_data.get("active_trade_parts", [])
        decimal_keys = ["entry_price", "qty", "sl_price",
                        "initial_usdt_value", "atr_at_entry"]

        for part_data_str_values in loaded_parts_raw:
            restored_part = {}
            valid_part = True
            for k, v_loaded in part_data_str_values.items():
                if k in decimal_keys:
                    if v_loaded is not None:
                        try:
                            restored_part[k] = Decimal(str(v_loaded))
                        except InvalidOperation:
                            logger.warning(
                                f"Phoenix Feather: Could not reforge Decimal rune '{v_loaded}' for '{k}' in part {part_data_str_values.get('part_id')}. Setting to None.")
                            restored_part[k] = None
                    else:
                        restored_part[k] = None
                elif k == "entry_time_ms":
                    if isinstance(v_loaded, (int, float)):
                        restored_part[k] = int(v_loaded)
                    elif isinstance(v_loaded, str):
                        try:
                            dt_obj = datetime.fromisoformat(
                                v_loaded.replace("Z", "+00:00"))
                            restored_part[k] = int(dt_obj.timestamp() * 1000)
                        except ValueError:
                            try:
                                restored_part[k] = int(v_loaded)
                            except ValueError:
                                logger.warning(
                                    f"Phoenix Feather: Malformed entry_time_ms '{v_loaded}' for part {part_data_str_values.get('part_id')}. Skipping part.")
                                valid_part = False
                                break
                    else:
                        logger.warning(
                            f"Phoenix Feather: Unexpected type for entry_time_ms '{v_loaded}' for part {part_data_str_values.get('part_id')}. Skipping part.")
                        valid_part = False
                        break
                else:
                    restored_part[k] = v_loaded

            if valid_part:
                _active_trade_parts.append(restored_part)

        trade_metrics.load_trades_from_list(
            state_data.get("trade_metrics_trades", []))

        initial_equity_str = state_data.get("initial_equity_str")
        if initial_equity_str and initial_equity_str.lower() != 'none':
            trade_metrics.initial_equity = safe_decimal_conversion(
                initial_equity_str, None)

        daily_equity_str = state_data.get("daily_start_equity_str")
        if daily_equity_str and daily_equity_str.lower() != 'none':
            trade_metrics.daily_start_equity = safe_decimal_conversion(
                daily_equity_str, None)

        trade_metrics.last_daily_reset_day = state_data.get(
            "last_daily_reset_day")
        if trade_metrics.last_daily_reset_day is not None:
            try:
                trade_metrics.last_daily_reset_day = int(
                    trade_metrics.last_daily_reset_day)
            except ValueError:
                logger.warning(
                    f"Phoenix Feather: Malformed last_daily_reset_day '{trade_metrics.last_daily_reset_day}'. Resetting.")
                trade_metrics.last_daily_reset_day = None

        logger.success(  # type: ignore [attr-defined]
            f"{NEON['SUCCESS']}Phoenix Feather: Memories reawakened! Active Parts: {len(_active_trade_parts)}, Chronicled Trades: {len(trade_metrics.trades)}{NEON['RESET']}")
        return True
    except json.JSONDecodeError as e:
        logger.error(
            f"{NEON['ERROR']}Phoenix Feather: Scroll '{STATE_FILE_PATH}' is corrupted or unreadable: {e}. Starting with fresh essence.{NEON['RESET']}")
        archive_name = f"{STATE_FILE_PATH}.corrupted_{time.strftime('%Y%m%d%H%M%S')}"
        try:
            os.rename(STATE_FILE_PATH, archive_name)
            logger.info(f"Corrupted scroll archived to: {archive_name}")
        except OSError as e_mv:
            logger.error(f"Error archiving corrupted scroll: {e_mv}")
        return False
    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}Phoenix Feather Error: Unexpected chaos during reawakening: {e}{NEON['RESET']}")
        logger.debug(traceback.format_exc())
        return False

# --- Indicator Calculation - The Alchemist's Art ---


def calculate_all_indicators(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    The grand alchemy chamber where raw market data (OHLCV) is transmuted
    by applying various technical indicators.
    """
    logger.debug(
        f"{NEON['COMMENT']}# Transmuting market data with indicator alchemy...{NEON['RESET']}")
    if df.empty:
        logger.warning(
            f"{NEON['WARNING']}Market data scroll empty. No indicators conjured.{NEON['RESET']}")
        return df.copy()

    for col in ['close', 'high', 'low', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            logger.error(
                f"{NEON['ERROR']}Essential market rune '{col}' missing from data scroll. Indicator alchemy will be flawed.{NEON['RESET']}")
            # Return early or ensure pandas_ta handles missing inputs gracefully (it usually does by returning NaNs for indicators)

    if config.strategy_name == StrategyName.DUAL_SUPERTREND_MOMENTUM:
        st_p_base = f"ST_{config.st_atr_length}_{config.st_multiplier}"
        st_c_base = f"CONFIRM_ST_{config.confirm_st_atr_length}_{config.confirm_st_multiplier}"

        df.ta.supertrend(length=config.st_atr_length, multiplier=float(config.st_multiplier),
                         append=True, col_names=(st_p_base, f"{st_p_base}d", f"{st_p_base}l", f"{st_p_base}s"))
        df.ta.supertrend(length=config.confirm_st_atr_length, multiplier=float(config.confirm_st_multiplier),
                         append=True, col_names=(st_c_base, f"{st_c_base}d", f"{st_c_base}l", f"{st_c_base}s"))

        primary_direction_col = f"{st_p_base}d"
        if primary_direction_col in df.columns:
            df["st_trend_up"] = (df[primary_direction_col] == 1)
            df["st_long_flip"] = (df["st_trend_up"]) & (
                df["st_trend_up"].shift(1) == False)
            df["st_short_flip"] = (df["st_trend_up"] == False) & (
                df["st_trend_up"].shift(1) == True)
        else:
            df["st_trend_up"], df["st_long_flip"], df["st_short_flip"] = pd.NA, False, False
            logger.error(
                f"{NEON['ERROR']}Primary SuperTrend direction column '{primary_direction_col}' missing!{NEON['RESET']}")

        confirm_direction_col = f"{st_c_base}d"
        if confirm_direction_col in df.columns:
            df["confirm_trend"] = df[confirm_direction_col].apply(
                lambda x: True if x == 1 else (False if x == -1 else pd.NA))
        else:
            df["confirm_trend"] = pd.NA
            logger.error(
                f"{NEON['ERROR']}Confirmation SuperTrend direction column '{confirm_direction_col}' missing!{NEON['RESET']}")

        if 'close' in df.columns and not df['close'].isnull().all():
            df.ta.mom(length=config.momentum_period,
                      append=True, col_names=("momentum",))
        else:
            df["momentum"] = pd.NA
            logger.warning(
                f"{NEON['WARNING']}Cannot calculate Momentum, 'close' prices missing/NaNs.{NEON['RESET']}")
        logger.debug(
            f"DualST+Mom indicators. Last Mom: {_format_for_log(df['momentum'].iloc[-1] if 'momentum' in df.columns and not df.empty else pd.NA)}, Last Confirm: {_format_for_log(df['confirm_trend'].iloc[-1] if 'confirm_trend' in df.columns and not df.empty else pd.NA, is_bool_trend=True)}")

    elif config.strategy_name == StrategyName.EHLERS_FISHER:
        if 'high' in df.columns and 'low' in df.columns and \
           not df['high'].isnull().all() and not df['low'].isnull().all():
            df.ta.fisher(length=config.ehlers_fisher_length, signal=config.ehlers_fisher_signal_length,
                         append=True, col_names=("ehlers_fisher", "ehlers_signal"))
        else:
            df["ehlers_fisher"], df["ehlers_signal"] = pd.NA, pd.NA
            logger.warning(
                f"{NEON['WARNING']}Cannot calculate Ehlers Fisher, 'high'/'low' prices missing/NaNs.{NEON['RESET']}")
        logger.debug(
            f"Ehlers Fisher. Last Fisher: {_format_for_log(df['ehlers_fisher'].iloc[-1] if 'ehlers_fisher' in df.columns and not df.empty else pd.NA)}, Signal: {_format_for_log(df['ehlers_signal'].iloc[-1] if 'ehlers_signal' in df.columns and not df.empty else pd.NA)}")

    atr_col_name = f"ATR_{config.atr_calculation_period}"
    if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns and \
       not df['high'].isnull().all() and not df['low'].isnull().all() and not df['close'].isnull().all():
        df.ta.atr(length=config.atr_calculation_period,
                  append=True, col_names=(atr_col_name,))
    else:
        df[atr_col_name] = pd.NA
        logger.warning(
            f"{NEON['WARNING']}Cannot calculate {atr_col_name}, HLC data missing/NaNs.{NEON['RESET']}")

    return df

# --- Exchange Interaction Primitives - Incantations for the Market Realm ---


@retry((ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection), tries=CONFIG.retry_count, delay=CONFIG.retry_delay_seconds, backoff=2, logger=logger)
def get_market_data(exchange: ccxt.Exchange, symbol: str, interval: str, limit: int = OHLCV_LIMIT) -> Optional[pd.DataFrame]:
    logger.debug(
        f"{NEON['COMMENT']}# Summoning {limit} market whispers (candles) for {symbol} ({interval})...{NEON['RESET']}")
    try:
        params = {}
        if symbol.endswith(":USDT"):
            params['category'] = 'linear'  # Bybit V5 may need category

        ohlcv = exchange.fetch_ohlcv(
            symbol, interval, limit=limit, params=params)
        if not ohlcv:
            logger.warning(
                f"{NEON['WARNING']}No OHLCV runes returned for {symbol}. Market silent or symbol/interval incorrect.{NEON['RESET']}")
            return None

        df = pd.DataFrame(
            ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        logger.debug(
            f"Summoned {len(df)} candles. Latest: {df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S %Z') if not df.empty else 'N/A'}")
        return df
    except ccxt.BadSymbol as e_bs:
        logger.critical(
            f"{NEON['CRITICAL']}Market realm rejects symbol '{symbol}' for OHLCV: {e_bs}. Ensure correct format & availability.{NEON['RESET']}")
        return None
    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}Error summoning market whispers for {symbol}: {e}{NEON['RESET']}")
        logger.debug(traceback.format_exc())
        return None


@retry((ccxt.NetworkError, ccxt.RequestTimeout), tries=CONFIG.retry_count, delay=CONFIG.retry_delay_seconds)
def fetch_account_balance(exchange: ccxt.Exchange, currency_code: str = CONFIG.usdt_symbol) -> Optional[Decimal]:
    logger.debug(
        f"{NEON['COMMENT']}# Scrying the treasury for {currency_code} balance...{NEON['RESET']}")
    try:
        # For Bybit V5 UTA, or 'CONTRACT' / 'SPOT' as needed
        params = {'accountType': 'UNIFIED'}
        if currency_code == "USDT":  # Common for futures margin
            # Bybit specific param for fetchBalance sometimes useful
            params['coin'] = currency_code

        balance_data = exchange.fetch_balance(params=params)

        total_balance_val = None
        # Path 1: Standard CCXT structure for a specific currency's total
        if currency_code in balance_data:
            total_balance_val = balance_data[currency_code].get('total')

        # Path 2: Overall total for a currency (less common for specific account types)
        if total_balance_val is None:
            total_balance_val = balance_data.get(
                'total', {}).get(currency_code)

        # Path 3: Bybit V5 specific paths if CCXT abstraction is not complete
        # Example: balance_data['info']['result']['list'][0]['totalEquity'] (for unified account)
        # Or balance_data['info']['result']['list'][0]['coin'][X]['walletBalance'] (for specific coin in account)
        # For USDT futures, we usually look for USDT in the relevant (e.g., UNIFIED or CONTRACT) account.
        # CCXT aims to put this in balance_data[currency_code]['total'].

        if total_balance_val is None:  # If still not found, log a warning and try a more general total if applicable
            logger.debug(
                f"Could not find specific {currency_code} total. Checking info response. Full balance: {json.dumps(balance_data.get('info', {}), indent=2, default=str)[:500]}...")
            # Attempt to find in 'info' if it's a unified account total equity.
            if 'info' in balance_data and 'result' in balance_data['info'] and 'list' in balance_data['info']['result']:
                if balance_data['info']['result']['list'] and 'totalEquity' in balance_data['info']['result']['list'][0]:
                    total_balance_val = balance_data['info']['result']['list'][0]['totalEquity']
                    logger.info(
                        f"Using 'totalEquity' from Unified Account for {currency_code} balance.")

        total_balance = safe_decimal_conversion(total_balance_val)

        if total_balance is None:
            logger.warning(
                f"{NEON['WARNING']}{currency_code} balance rune unreadable. Full scroll: {json.dumps(balance_data, indent=2, default=str)[:500]}...{NEON['RESET']}")
            return None

        logger.info(
            f"{NEON['INFO']}Current {currency_code} Treasury (Total Equity): {NEON['VALUE']}{total_balance:.2f}{NEON['RESET']}")
        return total_balance
    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}Error scrying treasury: {e}{NEON['RESET']}")
        logger.debug(traceback.format_exc())
        return None


def get_current_position_info() -> Tuple[str, Decimal]:
    """Determines current position side and quantity from Pyrmethus's internal memory (_active_trade_parts)."""
    global _active_trade_parts
    if not _active_trade_parts:
        return CONFIG.pos_none, Decimal(0)

    total_qty = sum(part.get('qty', Decimal(0))
                    for part in _active_trade_parts if isinstance(part.get('qty'), Decimal))

    if total_qty <= CONFIG.position_qty_epsilon:
        # If qty is zero after summing, clear parts as they are invalid/closed.
        # This is a self-correction mechanism.
        if total_qty < Decimal(0):  # Should not happen with proper logic
            logger.error(
                f"{NEON['ERROR']}Negative total quantity {total_qty} in _active_trade_parts. State corrupted! Clearing parts.{NEON['RESET']}")
        _active_trade_parts.clear()
        save_persistent_state(force_heartbeat=True)  # Persist the clear
        return CONFIG.pos_none, Decimal(0)

    current_side_str = _active_trade_parts[0].get('side')
    if current_side_str not in [CONFIG.pos_long, CONFIG.pos_short]:
        logger.warning(
            f"{NEON['WARNING']}Incoherent side '{current_side_str}' in active trade parts. Assuming None. Clearing parts.{NEON['RESET']}")
        _active_trade_parts.clear()  # Clear corrupted state
        save_persistent_state(force_heartbeat=True)
        return CONFIG.pos_none, total_qty  # Return qty but side as None

    return current_side_str, total_qty


def calculate_position_size(balance: Decimal, risk_per_trade_pct: Decimal, entry_price: Decimal, sl_price: Decimal, market_info: Dict) -> Optional[Decimal]:
    logger.debug(
        f"{NEON['COMMENT']}# Calculating alchemical proportions... Balance: {balance:.2f}, Risk %: {risk_per_trade_pct:.4f}, Entry: {entry_price}, SL: {sl_price}{NEON['RESET']}")
    if not (isinstance(balance, Decimal) and balance > 0 and isinstance(entry_price, Decimal) and entry_price > 0 and isinstance(sl_price, Decimal) and sl_price > 0 and sl_price != entry_price):
        logger.warning(
            f"{NEON['WARNING']}Invalid runes for sizing (balance/prices non-positive or SL invalid/equals entry). Bal: {balance}, Entry: {entry_price}, SL: {sl_price}{NEON['RESET']}")
        return None

    risk_per_unit = abs(entry_price - sl_price)
    # Should be caught by sl_price != entry_price but good check
    if risk_per_unit <= Decimal(0):
        logger.warning(
            f"{NEON['WARNING']}Risk per unit is zero ({risk_per_unit}). Entry {entry_price} matches SL {sl_price}. Cannot divine size.{NEON['RESET']}")
        return None

    usdt_at_risk = balance * risk_per_trade_pct
    logger.debug(f"USDT at risk for this venture: {usdt_at_risk:.2f}")

    quantity_base = usdt_at_risk / risk_per_unit
    position_usdt_value = quantity_base * entry_price

    if position_usdt_value > CONFIG.max_order_usdt_amount:
        logger.info(
            f"Calculated position USDT value {NEON['VALUE']}{position_usdt_value:.2f}{NEON['RESET']} exceeds MAX_ORDER_USDT_AMOUNT {NEON['VALUE']}{CONFIG.max_order_usdt_amount}{NEON['RESET']}. Capping.")
        position_usdt_value = CONFIG.max_order_usdt_amount
        quantity_base = position_usdt_value / entry_price

    qty_step_str = market_info.get('precision', {}).get('amount')
    min_qty_str = market_info.get('limits', {}).get('amount', {}).get('min')

    qty_step = safe_decimal_conversion(qty_step_str)
    min_qty = safe_decimal_conversion(min_qty_str)

    if qty_step is None or qty_step <= Decimal(0):
        logger.warning(
            f"{NEON['WARNING']}Qty step (precision.amount) not found/invalid for {market_info.get('symbol')}: '{qty_step_str}'. Using raw quantity (risky).{NEON['RESET']}")
    else:
        if quantity_base < qty_step:
            quantity_base = Decimal(0)
        else:
            quantity_base = (quantity_base // qty_step) * \
                qty_step  # Floor to nearest step

    if quantity_base <= CONFIG.position_qty_epsilon:
        logger.warning(
            f"{NEON['WARNING']}Calculated quantity {NEON['QTY']}{quantity_base}{NEON['WARNING']} is zero/negligible after precision. No order.{NEON['RESET']}")
        return None

    if min_qty is not None and quantity_base < min_qty:
        logger.warning(
            f"{NEON['WARNING']}Calculated quantity {NEON['QTY']}{quantity_base}{NEON['WARNING']} < exchange minimum {NEON['QTY']}{min_qty}{NEON['WARNING']}. No order.{NEON['RESET']}")
        return None

    display_precision = 0
    if qty_step is not None and qty_step > 0:
        s = str(qty_step).rstrip('0')
        if '.' in s:
            display_precision = len(s.split('.')[1])

    qty_display_format = f".{display_precision}f"
    logger.info(
        f"Calculated position size: {NEON['QTY']}{quantity_base:{qty_display_format}}{NEON['RESET']} {market_info.get('base', '')} (USDT Value: {position_usdt_value:.2f})")
    return quantity_base


@retry((ccxt.NetworkError, ccxt.RequestTimeout, ccxt.InsufficientFunds), tries=2, delay=3, logger=logger)
def place_risked_order(exchange: ccxt.Exchange, config: Config, side: str, entry_price_target: Decimal, sl_price: Decimal, atr_val: Optional[Decimal]) -> Optional[Dict]:
    """
    Places an order with calculated risk, SL, and records the new active trade part.
    'side' here is config.pos_long ("Long") or config.pos_short ("Short").
    """
    global _active_trade_parts
    logger.info(f"{NEON['ACTION']}Attempting to weave a {side.upper()} order for {config.symbol}... Target Entry: {NEON['PRICE']}{entry_price_target}{NEON['RESET']}, SL: {NEON['PRICE']}{sl_price}{NEON['RESET']}")

    balance = fetch_account_balance(exchange, config.usdt_symbol)
    if balance is None or balance <= Decimal(10):
        logger.error(
            f"{NEON['ERROR']}Insufficient treasury balance ({balance if balance else 'N/A'}) or scry failed. Cannot weave order.{NEON['RESET']}")
        config.send_notification_method(
            "Pyrmethus Order Fail", f"Insufficient balance for {config.symbol}")
        return None

    if config.MARKET_INFO is None:
        logger.error(
            f"{NEON['ERROR']}Market runes (MARKET_INFO) not divined. Cannot size/place order.{NEON['RESET']}")
        return None

    quantity = calculate_position_size(
        balance, config.risk_per_trade_percentage, entry_price_target, sl_price, config.MARKET_INFO)
    if quantity is None or quantity <= CONFIG.position_qty_epsilon:
        logger.warning(
            f"{NEON['WARNING']}Position sizing yielded negligible/no quantity. Order not placed.{NEON['RESET']}")
        return None

    # Map internal side ("Long", "Short") to exchange verb ("buy", "sell")
    order_side_verb = config.side_buy if side == config.pos_long else config.side_sell

    order_type_str = config.entry_order_type.value
    params = {
        'timeInForce': 'GTC' if order_type_str == 'Limit' else 'IOC',
        'stopLoss': float(sl_price),
        'slTriggerBy': 'MarkPrice',  # Or 'LastPrice', 'IndexPrice'
        # 0 for one-way mode. Bybit: 1 for Buy hedge, 2 for Sell hedge.
        'positionIdx': 0,
    }
    if config.symbol.endswith(":USDT"):
        params['category'] = 'linear'
    # Add 'takeProfit', 'tpTriggerBy' if TP is to be set with entry order.

    try:
        order_price_param = float(
            entry_price_target) if order_type_str == 'Limit' else None
        logger.info(
            f"Weaving {order_type_str.upper()} {order_side_verb.upper()} order: Qty {NEON['QTY']}{quantity}{NEON['RESET']}, Symbol {config.symbol}, SL {NEON['PRICE']}{sl_price}{NEON['RESET']}"
            f"{f', Price {order_price_param}' if order_price_param else ''}")

        order = exchange.create_order(config.symbol, order_type_str, order_side_verb, float(
            quantity), order_price_param, params)

        logger.success(  # type: ignore [attr-defined]
            f"{NEON['SUCCESS']}Entry Order {order['id']} ({order.get('status', 'N/A')}) cast for {NEON['QTY']}{quantity} {config.symbol.split('/')[0]}{NEON['RESET']}")
        config.send_notification_method(
            f"Pyrmethus Order Placed: {side.upper()}", f"{config.symbol} Qty: {quantity:.4f} @ {entry_price_target if order_type_str == 'Limit' else 'Market'}, SL: {sl_price}")

        # Brief wait for fill
        time.sleep(config.order_fill_timeout_seconds / 3)
        filled_order = exchange.fetch_order(order['id'], config.symbol)

        # 'closed' usually means fully filled for Bybit
        if filled_order.get('status') == 'closed':
            actual_entry_price = Decimal(str(filled_order.get(
                'average', filled_order.get('price', entry_price_target))))
            actual_qty = Decimal(str(filled_order.get('filled', quantity)))
            entry_timestamp_ms = int(filled_order.get(
                'timestamp', time.time() * 1000))
            part_id = str(uuid.uuid4())[:8]
            new_part = {
                # SL is part of entry
                "part_id": part_id, "entry_order_id": order['id'], "sl_order_id": None,
                "symbol": config.symbol, "side": side,  # Store internal "Long"/"Short"
                "entry_price": actual_entry_price, "qty": actual_qty,
                "entry_time_ms": entry_timestamp_ms, "sl_price": sl_price,
                "atr_at_entry": atr_val, "initial_usdt_value": actual_qty * actual_entry_price
            }
            _active_trade_parts.append(new_part)
            save_persistent_state(force_heartbeat=True)
            logger.success(  # type: ignore [attr-defined]
                f"{NEON['SUCCESS']}Order {order['id']} filled! Part {part_id} forged. Entry: {NEON['PRICE']}{actual_entry_price}{NEON['RESET']}, Qty: {NEON['QTY']}{actual_qty}{NEON['RESET']}")
            config.send_notification_method(
                f"Pyrmethus Order Filled: {side.upper()}", f"{config.symbol} Part {part_id} @ {actual_entry_price:.2f}")
            return new_part
        else:
            logger.warning(
                f"{NEON['WARNING']}Order {order['id']} not 'closed'. Status: {filled_order.get('status', 'N/A')}. Manual check advised.{NEON['RESET']}")
            # If Limit and not filled, consider cancellation logic here.
            return None

    except ccxt.InsufficientFunds as e:
        logger.error(
            f"{NEON['ERROR']}Insufficient funds for {quantity} of {config.symbol}: {e}{NEON['RESET']}")
        config.send_notification_method(
            "Pyrmethus Order Fail", f"Insufficient funds for {config.symbol}")
        return None
    except ccxt.ExchangeError as e:
        logger.error(
            f"{NEON['ERROR']}Exchange rejected order for {config.symbol}: {e}{NEON['RESET']}")
        config.send_notification_method(
            "Pyrmethus Order Fail", f"Exchange error: {str(e)[:100]}")
        if "Stop loss price is too close" in str(e) or "risk limit" in str(e).lower():
            logger.warning(
                f"{NEON['WARNING']}Specific Constraint: SL {sl_price} too close to entry {entry_price_target} or risk limits. Adjust.{NEON['RESET']}")
        return None
    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}Unexpected chaos weaving order for {config.symbol}: {e}{NEON['RESET']}")
        logger.debug(traceback.format_exc())
        return None


def close_position_part(exchange: ccxt.Exchange, config: Config, part_to_close: Dict, reason: str, close_price_target: Optional[Decimal] = None) -> bool:
    """Closes a specific part of an active position and records the outcome."""
    global _active_trade_parts
    part_id = part_to_close.get('part_id', 'UnknownPart')
    part_qty = part_to_close.get('qty', Decimal(0))
    part_side = part_to_close.get('side')  # "Long" or "Short"

    logger.info(f"{NEON['ACTION']}Attempting to unravel trade part {part_id} ({part_side} {NEON['QTY']}{part_qty}{NEON['RESET']} {config.symbol}) for: {reason}{NEON['RESET']}")

    if not isinstance(part_qty, Decimal) or part_qty <= CONFIG.position_qty_epsilon:
        logger.warning(
            f"{NEON['WARNING']}Cannot unravel part {part_id}: qty {part_qty} invalid/negligible.{NEON['RESET']}")
        _active_trade_parts = [
            p for p in _active_trade_parts if p.get('part_id') != part_id]
        save_persistent_state(force_heartbeat=True)
        return False

    close_side_verb = config.side_sell if part_side == config.pos_long else config.side_buy
    params = {'reduceOnly': True}
    if config.symbol.endswith(":USDT"):
        params['category'] = 'linear'

    try:
        logger.info(
            f"Casting MARKET {close_side_verb.upper()} order to unravel part {part_id}: Qty {NEON['QTY']}{part_qty}{NEON['RESET']}")
        close_order = exchange.create_order(
            config.symbol, 'market', close_side_verb, float(part_qty), None, params)
        logger.success(  # type: ignore [attr-defined]
            f"{NEON['SUCCESS']}Unraveling Order {close_order['id']} ({close_order.get('status', 'N/A')}) cast for part {part_id}.{NEON['RESET']}")

        time.sleep(config.order_fill_timeout_seconds / 2)
        filled_close_order = exchange.fetch_order(
            close_order['id'], config.symbol)

        if filled_close_order.get('status') == 'closed':
            actual_exit_price = Decimal(str(filled_close_order.get('average', filled_close_order.get(
                'price', close_price_target or part_to_close['entry_price']))))
            exit_timestamp_ms = int(filled_close_order.get(
                'timestamp', time.time() * 1000))
            entry_price = part_to_close.get('entry_price', Decimal(0))
            if not isinstance(entry_price, Decimal):
                entry_price = Decimal(str(entry_price))

            pnl_per_unit = (actual_exit_price - entry_price) if part_side == config.pos_long else (
                entry_price - actual_exit_price)
            pnl = pnl_per_unit * part_qty

            trade_metrics.log_trade(
                symbol=config.symbol, side=part_side, entry_price=entry_price, exit_price=actual_exit_price,
                qty=part_qty, entry_time_ms=part_to_close['entry_time_ms'], exit_time_ms=exit_timestamp_ms,
                reason=reason, part_id=part_id, pnl_str=str(pnl)
            )
            _active_trade_parts = [
                p for p in _active_trade_parts if p.get('part_id') != part_id]
            save_persistent_state(force_heartbeat=True)
            pnl_color_key = 'PNL_POS' if pnl > 0 else (
                'PNL_NEG' if pnl < 0 else 'PNL_ZERO')
            logger.success(  # type: ignore [attr-defined]
                f"{NEON['SUCCESS']}Part {part_id} unraveled. Exit: {NEON['PRICE']}{actual_exit_price}{NEON['RESET']}, PNL: {NEON[pnl_color_key]}{pnl:.2f} {config.usdt_symbol}{NEON['RESET']}")
            config.send_notification_method(
                f"Pyrmethus Position Closed", f"{config.symbol} Part {part_id}. PNL: {pnl:.2f}. Reason: {reason}")
            return True
        else:
            logger.warning(
                f"{NEON['WARNING']}Unraveling order {close_order['id']} for part {part_id} not 'closed'. Status: {filled_close_order.get('status', 'N/A')}. Manual check required.{NEON['RESET']}")
            return False
    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}Error unraveling part {part_id} for {config.symbol}: {e}{NEON['RESET']}")
        logger.debug(traceback.format_exc())
        config.send_notification_method(
            "Pyrmethus Close Fail", f"Error closing {config.symbol} part {part_id}: {str(e)[:80]}")
        return False


@retry((ccxt.NetworkError, ccxt.RequestTimeout), tries=CONFIG.retry_count, delay=CONFIG.retry_delay_seconds)
def cancel_all_symbol_orders(exchange: ccxt.Exchange, symbol: str):
    """Dispels all open orders for a given symbol on the exchange."""
    logger.info(
        f"{NEON['ACTION']}Dispelling all open enchantments (orders) for {symbol}...{NEON['RESET']}")
    try:
        params = {}
        if symbol.endswith(":USDT"):
            params['category'] = 'linear'

        # Bybit's cancelAllOrders can take a symbol and category.
        # If not, CCXT might handle it or throw FeatureNotSupported.
        exchange.cancel_all_orders(symbol, params=params if params else None)
        logger.success(  # type: ignore [attr-defined]
            f"{NEON['SUCCESS']}All open enchantments for {symbol} dispelled.{NEON['RESET']}")
    except ccxt.FeatureNotSupported:
        logger.warning(
            f"{NEON['WARNING']}Exchange {exchange.id} may not support cancel_all_orders for a specific symbol directly. Trying individual cancellation.{NEON['RESET']}")
        try:
            open_orders = exchange.fetch_open_orders(symbol)
            if not open_orders:
                logger.info(
                    f"No open orders found for {symbol} to dispel individually.")
                return
            logger.info(
                f"Found {len(open_orders)} open orders for {symbol}. Dispelling individually...")
            for order in open_orders:
                # Pass category if needed
                exchange.cancel_order(
                    order['id'], symbol, params=params if params else None)
                logger.info(f"Dispelled order {order['id']}.")
            # type: ignore [attr-defined]
            logger.success(
                f"{NEON['SUCCESS']}Individually dispelled {len(open_orders)} orders for {symbol}.{NEON['RESET']}")
        except Exception as e_ind:
            logger.error(
                f"{NEON['ERROR']}Error dispelling individual enchantments for {symbol}: {e_ind}{NEON['RESET']}")
    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}Error dispelling enchantments for {symbol}: {e}{NEON['RESET']}")


def close_all_symbol_positions(exchange: ccxt.Exchange, config: Config, reason: str):
    """Forcefully unravels all active position parts for the configured symbol and cancels orders."""
    global _active_trade_parts
    logger.warning(
        f"{NEON['WARNING']}Commencing forceful unraveling of ALL positions for {config.symbol} due to: {reason}{NEON['RESET']}")

    parts_to_close_copy = list(_active_trade_parts)
    if parts_to_close_copy:
        logger.info(
            f"Unraveling {len(parts_to_close_copy)} internally tracked part(s)...")
        for part in parts_to_close_copy:
            if part.get('symbol') == config.symbol:
                logger.info(
                    f"Unraveling internally tracked part: {part.get('part_id', 'UnknownID')}")
                close_position_part(exchange, config, part,
                                    reason + " (Global Unraveling)")
    else:
        logger.info("No internally tracked trade parts to unravel.")

    logger.info(
        f"Performing safeguard scrying of exchange for any residual {config.symbol} positions...")
    try:
        params_fetch_pos = {}
        if config.symbol.endswith(":USDT"):
            params_fetch_pos['category'] = 'linear'

        positions = exchange.fetch_positions(
            [config.symbol], params=params_fetch_pos)
        residual_closed_count = 0
        for pos_data in positions:
            if pos_data and pos_data.get('symbol') == config.symbol:
                pos_qty_val = pos_data.get('contracts', pos_data.get(
                    'size', pos_data.get('contractSize', '0')))
                pos_qty = safe_decimal_conversion(pos_qty_val, Decimal(0))
                pos_side_str = pos_data.get('side')  # 'long' or 'short'

                if pos_qty > CONFIG.position_qty_epsilon:
                    side_to_close_market = config.side_sell if pos_side_str == 'long' else config.side_buy
                    logger.warning(
                        f"Found residual exchange position: {pos_side_str} {NEON['QTY']}{pos_qty}{NEON['RESET']} {config.symbol}. Forceful market unraveling.")
                    params_close_residual = {'reduceOnly': True}
                    if config.symbol.endswith(":USDT"):
                        params_close_residual['category'] = 'linear'
                    exchange.create_order(config.symbol, 'market', side_to_close_market, float(
                        pos_qty), params=params_close_residual)
                    logger.info(
                        f"Residual position unraveling order cast for {NEON['QTY']}{pos_qty}{NEON['RESET']} {config.symbol}.")
                    residual_closed_count += 1
                    time.sleep(config.post_close_delay_seconds)

        if residual_closed_count > 0:
            logger.info(
                f"Safeguard unraveling attempted for {residual_closed_count} residual position(s).")
        else:
            logger.info(
                "No residual positions found on exchange for safeguard unraveling.")
    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}Error during final scrying/unraveling of exchange positions for {config.symbol}: {e}{NEON['RESET']}")

    cancel_all_symbol_orders(exchange, config.symbol)
    logger.info(
        f"All positions/orders for {config.symbol} should now be unraveled/dispelled.")


# --- Main Spell Weaving - The Heart of Pyrmethus ---
def main_loop(exchange: ccxt.Exchange, config: Config) -> None:
    logger.info(
        f"{NEON['HEADING']}=== Pyrmethus Spell v2.8.2 Awakening on {exchange.id} ==={NEON['RESET']}")
    logger.info(
        f"{NEON['COMMENT']}# Listening for {config.symbol} whispers on {config.interval}...{NEON['RESET']}")

    if load_persistent_state():
        logger.success(  # type: ignore [attr-defined]
            f"{NEON['SUCCESS']}Reawakened from Phoenix scroll. Active parts: {len(_active_trade_parts)}.{NEON['RESET']}")
    else:
        logger.info(
            f"{NEON['INFO']}No prior state or ignored. Starting fresh.{NEON['RESET']}")

    current_balance = fetch_account_balance(exchange, config.usdt_symbol)
    if current_balance is not None:
        trade_metrics.set_initial_equity(current_balance)
    else:
        logger.warning(
            f"{NEON['WARNING']}Could not set initial treasury. Drawdown checks affected.{NEON['RESET']}")

    while True:
        try:
            logger.debug(
                f"{NEON['COMMENT']}# New cycle of observation...{NEON['RESET']}")

            current_balance = fetch_account_balance(
                exchange, config.usdt_symbol)
            if current_balance is not None:
                # Updates daily equity if new day
                trade_metrics.set_initial_equity(current_balance)
                drawdown_hit, dd_reason = trade_metrics.check_drawdown(
                    current_balance)
                if drawdown_hit:
                    logger.critical(
                        f"{NEON['CRITICAL']}Max drawdown! {dd_reason}. Pyrmethus rests.{NEON['RESET']}")
                    close_all_symbol_positions(
                        exchange, config, "Max Drawdown Reached")
                    config.send_notification_method(
                        "Pyrmethus Max Drawdown", f"Halted for {config.symbol}. {dd_reason}")
                    break
            else:
                logger.warning(
                    f"{NEON['WARNING']}Failed treasury scry. Drawdown check on stale data.{NEON['RESET']}")

            ohlcv_df = get_market_data(
                exchange, config.symbol, config.interval, limit=OHLCV_LIMIT + config.api_fetch_limit_buffer)
            if ohlcv_df is None or ohlcv_df.empty:
                logger.warning(
                    f"{NEON['WARNING']}Market whispers faint for {config.symbol}. Retrying.{NEON['RESET']}")
                time.sleep(config.sleep_seconds)
                continue

            df_with_indicators = calculate_all_indicators(
                ohlcv_df.copy(), config)
            if df_with_indicators.empty or df_with_indicators.iloc[-1].isnull().all():
                logger.warning(
                    f"{NEON['WARNING']}Indicator alchemy yielded faint/no runes for {config.symbol}. Pausing.{NEON['RESET']}")
                time.sleep(config.sleep_seconds)
                continue

            signals = config.strategy_instance.generate_signals(
                df_with_indicators)
            logger.debug(
                f"Omens ({config.symbol}): EnterL={signals['enter_long']}, EnterS={signals['enter_short']}, ExitL={signals['exit_long']}, ExitS={signals['exit_short']}")

            current_pos_side, current_pos_qty = get_current_position_info()
            side_color_key = 'SIDE_LONG' if current_pos_side == config.pos_long else \
                             ('SIDE_SHORT' if current_pos_side ==
                              config.pos_short else 'SIDE_FLAT')
            logger.debug(
                f"Stance ({config.symbol}): Side={NEON[side_color_key]}{current_pos_side}{NEON['RESET']}, Qty={NEON['QTY']}{current_pos_qty}{NEON['RESET']}")

            atr_col = f"ATR_{config.atr_calculation_period}"
            latest_atr_val = df_with_indicators[atr_col].iloc[-1] if atr_col in df_with_indicators.columns and not df_with_indicators.empty else pd.NA
            latest_close_val = df_with_indicators['close'].iloc[-1] if 'close' in df_with_indicators.columns and not df_with_indicators.empty else pd.NA
            latest_atr = safe_decimal_conversion(latest_atr_val)
            latest_close = safe_decimal_conversion(latest_close_val)

            if pd.isna(latest_atr) or pd.isna(latest_close):
                logger.warning(
                    f"{NEON['WARNING']}Missing ATR ({_format_for_log(latest_atr)}) or Close ({_format_for_log(latest_close)}) for {config.symbol}. No decisions.{NEON['RESET']}")
                time.sleep(config.sleep_seconds)
                continue
            if latest_atr <= Decimal(0):
                logger.warning(
                    f"{NEON['WARNING']}ATR ({latest_atr}) for {config.symbol} not positive. SL flawed. Skipping.{NEON['RESET']}")
                time.sleep(config.sleep_seconds)
                continue

            # --- Position Management ---
            if current_pos_side == config.pos_none:
                if signals.get("enter_long"):
                    logger.info(
                        f"{NEON['ACTION']}Omens for LONG venture for {config.symbol}.{NEON['RESET']}")
                    sl_price = latest_close - \
                        (latest_atr * config.atr_stop_loss_multiplier)
                    place_risked_order(
                        exchange, config, config.pos_long, latest_close, sl_price, latest_atr)
                elif signals.get("enter_short"):
                    logger.info(
                        f"{NEON['ACTION']}Omens for SHORT venture for {config.symbol}.{NEON['RESET']}")
                    sl_price = latest_close + \
                        (latest_atr * config.atr_stop_loss_multiplier)
                    place_risked_order(
                        exchange, config, config.pos_short, latest_close, sl_price, latest_atr)
            elif current_pos_side == config.pos_long:
                if signals.get("exit_long"):
                    exit_reason = signals.get(
                        'exit_reason', 'Strategy Exit Signal')
                    logger.info(
                        f"{NEON['ACTION']}Omen to EXIT LONG for {config.symbol}: {exit_reason}{NEON['RESET']}")
                    for part in list(_active_trade_parts):  # Iterate copy
                        if part.get('side') == config.pos_long and part.get('symbol') == config.symbol:
                            close_position_part(
                                exchange, config, part, exit_reason)
            elif current_pos_side == config.pos_short:
                if signals.get("exit_short"):
                    exit_reason = signals.get(
                        'exit_reason', 'Strategy Exit Signal')
                    logger.info(
                        f"{NEON['ACTION']}Omen to EXIT SHORT for {config.symbol}: {exit_reason}{NEON['RESET']}")
                    for part in list(_active_trade_parts):  # Iterate copy
                        if part.get('side') == config.pos_short and part.get('symbol') == config.symbol:
                            close_position_part(
                                exchange, config, part, exit_reason)

            # SL checks for active parts (if not fully reliant on exchange-based SL orders placed at entry)
            # E.g., for custom TSL or other manual SL checks.
            # For now, relying on initial SL placed with order via Bybit V5 feature.

            save_persistent_state()  # Heartbeat save logic inside
            logger.debug(
                f"{NEON['COMMENT']}# Cycle complete for {config.symbol}. Resting {config.sleep_seconds}s...{NEON['RESET']}")
            time.sleep(config.sleep_seconds)

        except KeyboardInterrupt:
            logger.warning(
                f"\n{NEON['WARNING']}Sorcerer's intervention! Pyrmethus prepares for slumber...{NEON['RESET']}")
            config.send_notification_method(
                "Pyrmethus Shutdown", "Manual shutdown initiated.")
            break
        except Exception as e:
            logger.critical(
                f"{NEON['CRITICAL']}Critical error in main loop for {config.symbol}! Error: {e}{NEON['RESET']}")
            logger.debug(traceback.format_exc())
            config.send_notification_method(
                "Pyrmethus Critical Error", f"Loop crashed for {config.symbol}: {str(e)[:150]}")
            logger.info(
                f"Resting longer ({config.sleep_seconds * 5}s) after critical error.")
            time.sleep(config.sleep_seconds * 5)
    

if __name__ == "__main__":
    logger.info(
        f"{NEON['COMMENT']}# Pyrmethus prepares to breach veil to exchange realm...{NEON['RESET']}")
    exchange = None
    try:
        exchange_params = {
            'apiKey': CONFIG.api_key, 'secret': CONFIG.api_secret,
            'options': {'defaultType': 'swap', 'adjustForTimeDifference': True, 'brokerId': 'PYRMETHUSV282'},
            'enableRateLimit': True, 'recvWindow': CONFIG.default_recv_window
        }

        if CONFIG.PAPER_TRADING_MODE:
            logger.warning(
                f"{NEON['WARNING']}PAPER_TRADING_MODE True. Attuning to Bybit TESTNET.{NEON['RESET']}")
            exchange_params['urls'] = {'api': 'https://api-testnet.bybit.com'}

        exchange = ccxt.bybit(exchange_params)
        logger.info(f"Connecting to: {exchange.id} (CCXT: {ccxt.__version__})")

        markets = exchange.load_markets()
        if CONFIG.symbol not in markets:
            err_msg = f"Symbol {CONFIG.symbol} not found in {exchange.id} market runes."
            logger.critical(f"{NEON['CRITICAL']}{err_msg}{NEON['RESET']}")
            CONFIG.send_notification_method(
                "Pyrmethus Startup Fail", f"Symbol {CONFIG.symbol} not found on {exchange.id}.")
            sys.exit(1)

        CONFIG.MARKET_INFO = markets[CONFIG.symbol]
        price_prec_raw = CONFIG.MARKET_INFO.get('precision', {}).get('price')
        amount_prec_raw = CONFIG.MARKET_INFO.get('precision', {}).get('amount')

        def get_decimal_places_from_step(step_value_str: Optional[str]) -> int:
            if step_value_str is None:
                return 0
            s_val = str(step_value_str).lower().strip()
            if 'e-' in s_val:
                try:
                    return int(s_val.split('e-')[-1])
                except:
                    return 0
            if '.' in s_val:
                # Remove trailing zeros after decimal point before splitting
                return len(s_val.split('.')[1].rstrip('0'))
            return 0

        price_prec_display = get_decimal_places_from_step(str(price_prec_raw))
        amount_prec_display = get_decimal_places_from_step(
            str(amount_prec_raw))

        logger.success(  # type: ignore [attr-defined]
            f"{NEON['SUCCESS']}Market runes for {CONFIG.symbol}: Price Step {price_prec_raw} (~{price_prec_display} dec places), Amount Step {amount_prec_raw} (~{amount_prec_display} dec places){NEON['RESET']}")

        category = None
        if CONFIG.symbol.endswith(":USDT"):
            category = "linear"
        # Add elif for 'inverse' or 'spot' if needed based on symbol structure

        try:
            leverage_params = {}
            if category:
                leverage_params['category'] = category
            logger.info(
                f"Setting leverage to {CONFIG.leverage}x for {CONFIG.symbol} (Category: {category if category else 'inferred'})...")
            response = exchange.set_leverage(
                CONFIG.leverage, CONFIG.symbol, params=leverage_params)
            # Leverage response can be large, log only a summary or key parts if needed.
            logger.success(  # type: ignore [attr-defined]
                f"{NEON['SUCCESS']}Leverage for {CONFIG.symbol} set/confirmed. Exchange Response (part): {json.dumps(response, default=str)[:150]}...{NEON['RESET']}")
        except Exception as e_lev:
            logger.warning(
                f"{NEON['WARNING']}Could not set leverage for {CONFIG.symbol} (may be pre-set or issue): {e_lev}{NEON['RESET']}")
            CONFIG.send_notification_method(
                "Pyrmethus Leverage Warn", f"Leverage set issue for {CONFIG.symbol}: {str(e_lev)[:60]}")

        logger.success(  # type: ignore [attr-defined]
            f"{NEON['SUCCESS']}Connected to {exchange.id} for {CONFIG.symbol}.{NEON['RESET']}")
        CONFIG.send_notification_method(
            "Pyrmethus Online", f"Connected to {exchange.id} for {CONFIG.symbol} @ {CONFIG.leverage}x.")

    except AttributeError as e_attr:
        logger.critical(
            f"{NEON['CRITICAL']}Exchange attribute error: {e_attr}. CCXT issue or invalid exchange ID?{NEON['RESET']}")
        sys.exit(1)
    except ccxt.AuthenticationError as e_auth:
        logger.critical(
            f"{NEON['CRITICAL']}Authentication with {exchange.id if exchange else 'exchange'} failed! Check API keys: {e_auth}{NEON['RESET']}")
        if exchange:
            CONFIG.send_notification_method(
                "Pyrmethus Auth Fail", f"API Auth Error for {exchange.id}.")
        sys.exit(1)
    except ccxt.NetworkError as e_net:
        logger.critical(
            f"{NEON['CRITICAL']}Network Error connecting to {exchange.id if exchange else 'exchange'}: {e_net}. Check connection/DNS.{NEON['RESET']}")
        if exchange:
            CONFIG.send_notification_method(
                "Pyrmethus Network Error", f"Cannot connect to {exchange.id}: {str(e_net)[:80]}")
        sys.exit(1)
    except ccxt.ExchangeError as e_exch:
        logger.critical(
            f"{NEON['CRITICAL']}Exchange API Error with {exchange.id if exchange else 'exchange'}: {e_exch}. Check API/symbol/status.{NEON['RESET']}")
        if exchange:
            CONFIG.send_notification_method(
                "Pyrmethus API Error", f"{exchange.id} API issue: {str(e_exch)[:80]}")
        sys.exit(1)
    except Exception as e_general:
        logger.critical(
            f"{NEON['CRITICAL']}Failed to init exchange/spell pre-requisites: {e_general}{NEON['RESET']}")
        logger.debug(traceback.format_exc())
        CONFIG.send_notification_method(
            "Pyrmethus Init Error", f"Exchange init failed: {str(e_general)[:80]}")
        sys.exit(1)

    if exchange:
        main_loop(exchange, CONFIG)
    else:
        logger.critical(
            f"{NEON['CRITICAL']}Exchange not initialized. Pyrmethus cannot weave in an empty void. Spell aborted.{NEON['RESET']}")
        sys.exit(1)  # Should have exited earlier, but as a safeguard.
