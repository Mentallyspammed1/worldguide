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

    if not hasattr(
        pd, "NA"
    ):  # Ensure pandas version supports pd.NA, a subtle but crucial rune
        raise ImportError("Pandas version < 1.0 not supported, pd.NA is required.")
    import pandas_ta as ta  # type: ignore[import] # The Alchemist's Table for Technical Analysis
    from colorama import Back, Fore, Style  # The Prisms of Perception
    from colorama import init as colorama_init
    from dotenv import load_dotenv  # For whispering secrets from .env scrolls
    from retry import retry  # The Art of Tenacious Spellcasting
except ImportError as e:
    missing_pkg = getattr(e, "name", "dependency")
    # Direct stderr write for pre-logging critical failure
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
STATE_FILE_NAME = (
    "pyrmethus_phoenix_state_v282.json"  # Updated version for the new weave
)
STATE_FILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), STATE_FILE_NAME
)
HEARTBEAT_INTERVAL_SECONDS = 60  # Save state at least this often if the spell is active
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
    "ACTION": Fore.YELLOW
    + Style.BRIGHT,  # For actions like "Placing order", "Closing position"
    "COMMENT": Fore.CYAN + Style.DIM,  # For wizardly comments within output
    "RESET": Style.RESET_ALL,
}

# --- Initializations - Awakening the Spell ---
colorama_init(autoreset=True)  # Let the colors flow automatically
# Attempt to load .env file from the script's sacred ground
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if load_dotenv(dotenv_path=env_path):
    # Use a temporary logger for this specific message before full logger setup
    logging.getLogger("PreConfig").info(
        f"{NEON['INFO']}Secrets whispered from .env scroll: {NEON['VALUE']}{env_path}{NEON['RESET']}"
    )
else:
    logging.getLogger("PreConfig").warning(
        f"{NEON['WARNING']}No .env scroll found at {NEON['VALUE']}{env_path}{NEON['WARNING']}. Relying on system environment variables or defaults.{NEON['RESET']}"
    )
getcontext().prec = (
    18  # Set precision for Decimal calculations, for the finest market etchings
)


# --- Helper Functions - Minor Incantations ---
def safe_decimal_conversion(
    value: Any, default_if_error: Any = pd.NA
) -> Union[Decimal, Any]:
    """
    Safely converts a value to Decimal. If conversion fails, returns pd.NA or a specified default.
    This guards against the chaotic energies of malformed data.
    """
    if value is None or (
        isinstance(value, float) and pd.isna(value)
    ):  # Check for None or NaN float
        return default_if_error
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        # Log this minor disturbance if needed, but often it's expected for missing data
        # logger.debug(f"Could not convert '{value}' to Decimal, returning default.")
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
    if pd.isna(value) or value is None:
        return "N/A"
    return str(value)


def send_termux_notification(
    title: str, message: str, notification_id: int = 777
) -> None:
    """Sends a notification using Termux API, a whisper on the digital wind."""
    # This function is now implemented, using CONFIG for enable/timeout settings.
    # It will be assigned to CONFIG.send_notification_method.
    # Note: This function relies on the CONFIG object being initialized.
    # For pre-config errors, direct subprocess calls might be needed or a simpler pre-config notification.
    if "CONFIG" not in globals() or not CONFIG.enable_notifications:
        if "CONFIG" in globals():  # Config is loaded but notifications disabled
            logging.getLogger("TermuxNotification").debug(
                "Termux notifications are disabled by configuration."
            )
        else:  # Config not yet loaded (e.g. very early startup error)
            logging.getLogger("TermuxNotification").warning(
                "Attempted Termux notification before CONFIG loaded or with notifications disabled."
            )
        return
    try:
        # Ensure strings are properly escaped for shell command
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

        logging.getLogger("TermuxNotification").debug(
            f"{NEON['ACTION']}Attempting to send Termux notification: Title='{title}'{NEON['RESET']}"
        )
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        # Use the specific notification timeout from CONFIG
        stdout, stderr = process.communicate(
            timeout=CONFIG.notification_timeout_seconds
        )

        if process.returncode == 0:
            logging.getLogger("TermuxNotification").info(
                f"{NEON['SUCCESS']}Termux notification '{title}' sent successfully.{NEON['RESET']}"
            )
        else:
            err_msg = stderr.decode().strip() if stderr else "Unknown error"
            logging.getLogger("TermuxNotification").error(
                f"{NEON['ERROR']}Failed to send Termux notification '{title}'. Return code: {process.returncode}. Error: {err_msg}{NEON['RESET']}"
            )
    except FileNotFoundError:
        logging.getLogger("TermuxNotification").error(
            f"{NEON['ERROR']}Termux API command 'termux-notification' not found. Is 'termux-api' package installed and accessible?{NEON['RESET']}"
        )
    except subprocess.TimeoutExpired:
        logging.getLogger("TermuxNotification").error(
            f"{NEON['ERROR']}Termux notification command timed out for '{title}'.{NEON['RESET']}"
        )
    except Exception as e:
        logging.getLogger("TermuxNotification").error(
            f"{NEON['ERROR']}Unexpected error sending Termux notification '{title}': {e}{NEON['RESET']}"
        )
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
            f"{NEON['HEADING']}--- Summoning Configuration Runes v2.8.2 ---{NEON['RESET']}"
        )
        _pre_logger.info(
            f"{NEON['COMMENT']}# Pyrmethus attunes to the environment's whispers...{NEON['RESET']}"
        )

        self.api_key: str = self._get_env("BYBIT_API_KEY", required=True, secret=True)
        self.api_secret: str = self._get_env(
            "BYBIT_API_SECRET", required=True, secret=True
        )
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT")
        self.interval: str = self._get_env("INTERVAL", "1m")
        self.leverage: int = self._get_env("LEVERAGE", 10, cast_type=int)
        self.sleep_seconds: int = self._get_env("SLEEP_SECONDS", 10, cast_type=int)
        self.strategy_name: StrategyName = StrategyName(
            self._get_env(
                "STRATEGY_NAME", StrategyName.DUAL_SUPERTREND_MOMENTUM.value
            ).upper()
        )
        self.strategy_instance: (
            "TradingStrategy"  # Forward declaration, the chosen spell form
        )
        self.risk_per_trade_percentage: Decimal = self._get_env(
            "RISK_PER_TRADE_PERCENTAGE", "0.005", cast_type=Decimal
        )  # 0.5% risk
        self.enable_dynamic_risk: bool = self._get_env(
            "ENABLE_DYNAMIC_RISK", "false", cast_type=bool
        )
        self.dynamic_risk_min_pct: Decimal = self._get_env(
            "DYNAMIC_RISK_MIN_PCT", "0.0025", cast_type=Decimal
        )
        self.dynamic_risk_max_pct: Decimal = self._get_env(
            "DYNAMIC_RISK_MAX_PCT", "0.01", cast_type=Decimal
        )
        self.dynamic_risk_perf_window: int = self._get_env(
            "DYNAMIC_RISK_PERF_WINDOW", 10, cast_type=int
        )
        self.max_order_usdt_amount: Decimal = self._get_env(
            "MAX_ORDER_USDT_AMOUNT", "1000.0", cast_type=Decimal
        )
        self.required_margin_buffer: Decimal = self._get_env(
            "REQUIRED_MARGIN_BUFFER", "1.05", cast_type=Decimal
        )
        self.max_account_margin_ratio: Decimal = self._get_env(
            "MAX_ACCOUNT_MARGIN_RATIO", "0.5", cast_type=Decimal
        )
        self.enable_max_drawdown_stop: bool = self._get_env(
            "ENABLE_MAX_DRAWDOWN_STOP", "true", cast_type=bool
        )
        self.max_drawdown_percent: Decimal = self._get_env(
            "MAX_DRAWDOWN_PERCENT", "0.05", cast_type=Decimal
        )  # 5% daily drawdown
        self.enable_time_based_stop: bool = self._get_env(
            "ENABLE_TIME_BASED_STOP", "false", cast_type=bool
        )
        self.max_trade_duration_seconds: int = self._get_env(
            "MAX_TRADE_DURATION_SECONDS", 3600, cast_type=int
        )
        self.enable_dynamic_atr_sl: bool = self._get_env(
            "ENABLE_DYNAMIC_ATR_SL", "true", cast_type=bool
        )
        self.atr_short_term_period: int = self._get_env(
            "ATR_SHORT_TERM_PERIOD", 7, cast_type=int
        )
        self.atr_long_term_period: int = self._get_env(
            "ATR_LONG_TERM_PERIOD", 50, cast_type=int
        )
        self.volatility_ratio_low_threshold: Decimal = self._get_env(
            "VOLATILITY_RATIO_LOW_THRESHOLD", "0.7", cast_type=Decimal
        )
        self.volatility_ratio_high_threshold: Decimal = self._get_env(
            "VOLATILITY_RATIO_HIGH_THRESHOLD", "1.5", cast_type=Decimal
        )
        self.atr_sl_multiplier_low_vol: Decimal = self._get_env(
            "ATR_SL_MULTIPLIER_LOW_VOL", "1.0", cast_type=Decimal
        )
        self.atr_sl_multiplier_normal_vol: Decimal = self._get_env(
            "ATR_SL_MULTIPLIER_NORMAL_VOL", "1.3", cast_type=Decimal
        )
        self.atr_sl_multiplier_high_vol: Decimal = self._get_env(
            "ATR_SL_MULTIPLIER_HIGH_VOL", "1.8", cast_type=Decimal
        )
        self.atr_stop_loss_multiplier: Decimal = self._get_env(
            "ATR_STOP_LOSS_MULTIPLIER", "1.5", cast_type=Decimal
        )  # Fallback
        self.enable_position_scaling: bool = self._get_env(
            "ENABLE_POSITION_SCALING", "false", cast_type=bool
        )
        self.max_scale_ins: int = self._get_env("MAX_SCALE_INS", 0, cast_type=int)
        self.scale_in_risk_percentage: Decimal = self._get_env(
            "SCALE_IN_RISK_PERCENTAGE", "0.0025", cast_type=Decimal
        )
        self.min_profit_for_scale_in_atr: Decimal = self._get_env(
            "MIN_PROFIT_FOR_SCALE_IN_ATR", "1.0", cast_type=Decimal
        )
        self.trailing_stop_percentage: Decimal = self._get_env(
            "TRAILING_STOP_PERCENTAGE", "0.005", cast_type=Decimal
        )
        self.trailing_stop_activation_offset_percent: Decimal = self._get_env(
            "TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT", "0.001", cast_type=Decimal
        )
        self.entry_order_type: OrderEntryType = OrderEntryType(
            self._get_env("ENTRY_ORDER_TYPE", OrderEntryType.MARKET.value).upper()
        )
        self.limit_order_offset_pips: int = self._get_env(
            "LIMIT_ORDER_OFFSET_PIPS", 2, cast_type=int
        )
        self.limit_order_fill_timeout_seconds: int = self._get_env(
            "LIMIT_ORDER_FILL_TIMEOUT_SECONDS", 20, cast_type=int
        )
        self.order_fill_timeout_seconds: int = self._get_env(
            "ORDER_FILL_TIMEOUT_SECONDS", 15, cast_type=int
        )
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
        self.ehlers_fisher_length: int = self._get_env(
            "EHLERS_FISHER_LENGTH", 10, cast_type=int
        )
        self.ehlers_fisher_signal_length: int = self._get_env(
            "EHLERS_FISHER_SIGNAL_LENGTH", 1, cast_type=int
        )
        self.atr_calculation_period: int = (
            self.atr_short_term_period
            if self.enable_dynamic_atr_sl
            else self._get_env("ATR_CALCULATION_PERIOD", 14, cast_type=int)
        )
        self.enable_notifications: bool = self._get_env(
            "ENABLE_NOTIFICATIONS", "true", cast_type=bool
        )
        self.notification_timeout_seconds: int = self._get_env(
            "NOTIFICATION_TIMEOUT_SECONDS", 10, cast_type=int
        )
        self.default_recv_window: int = self._get_env(
            "DEFAULT_RECV_WINDOW", 13000, cast_type=int
        )
        self.side_buy: str = "buy"
        self.side_sell: str = "sell"
        self.pos_long: str = "Long"
        self.pos_short: str = "Short"
        self.pos_none: str = "None"
        self.usdt_symbol: str = "USDT"
        self.retry_count: int = 5
        self.retry_delay_seconds: int = 5
        self.api_fetch_limit_buffer: int = 20
        self.position_qty_epsilon: Decimal = Decimal("1e-9")
        self.post_close_delay_seconds: int = 3
        self.MARKET_INFO: Optional[Dict[str, Any]] = None
        self.PAPER_TRADING_MODE: bool = self._get_env(
            "PAPER_TRADING_MODE", "false", cast_type=bool
        )
        self.send_notification_method = (
            send_termux_notification  # Assign the notification function
        )

        self._validate_parameters()  # Scrutinizing the runes for flaws
        _pre_logger.info(
            f"{NEON['HEADING']}--- Configuration Runes v2.8.2 Summoned and Verified ---{NEON['RESET']}"
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
        """Fetches a configuration rune (environment variable), applying defaults and casting as needed."""
        _logger = logging.getLogger("ConfigModule._get_env")
        value_str = os.getenv(key)
        source = "Env Var"
        value_to_cast: Any = None
        display_value = "********" if secret and value_str is not None else value_str

        if value_str is None:
            if required:
                _logger.critical(
                    f"{NEON['CRITICAL']}CRITICAL: Required config rune '{key}' not found in environment or .env scroll. Pyrmethus cannot proceed without this essence.{NEON['RESET']}"
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

        if value_to_cast is None:
            if required:
                _logger.critical(
                    f"{NEON['CRITICAL']}CRITICAL: Required config rune '{key}' resolved to None after checking environment and defaults.{NEON['RESET']}"
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
            raw_value_str_for_cast = str(value_to_cast)
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
                final_value = int(Decimal(raw_value_str_for_cast))
            elif cast_type == float:
                final_value = float(raw_value_str_for_cast)
            elif cast_type == str:
                final_value = raw_value_str_for_cast
            else:
                _logger.warning(
                    f"{NEON['WARNING']}Unsupported cast_type '{cast_type.__name__}' for key '{key}'. Returning raw string: '{raw_value_str_for_cast}'.{NEON['RESET']}"
                )
                final_value = raw_value_str_for_cast
        except (ValueError, TypeError, InvalidOperation) as e:
            _logger.error(
                f"{NEON['ERROR']}Invalid type/value for '{key}': '{value_to_cast}' (Source: {source}). Expected {cast_type.__name__}. Err: {e}. Using Default: '{default}'.{NEON['RESET']}"
            )
            if default is None:
                if required:
                    _logger.critical(
                        f"{NEON['CRITICAL']}CRITICAL: Failed cast for required key '{key}', default is None.{NEON['RESET']}"
                    )
                    raise ValueError(
                        f"Required env var '{key}' failed casting, no valid default."
                    )
                else:
                    _logger.warning(
                        f"{NEON['WARNING']}Cast fail for optional '{key}', default is None. Final: None{NEON['RESET']}"
                    )
                    return None
            else:
                source = "Default (Fallback)"
                _logger.debug(
                    f"Casting fallback default '{default}' for '{key}' to {cast_type.__name__}"
                )
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
                        final_value = default_str_for_cast
                    _logger.warning(
                        f"{NEON['WARNING']}Used casted default for {key}: '{NEON['VALUE']}{final_value}{NEON['WARNING']}'{NEON['RESET']}"
                    )
                except (ValueError, TypeError, InvalidOperation) as e_default:
                    _logger.critical(
                        f"{NEON['CRITICAL']}CRITICAL: Cast fail for value AND default for '{key}'. Err: {e_default}{NEON['RESET']}"
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
        """Performs basic validation of critical configuration parameters. A necessary rite of scrutiny."""
        _logger = logging.getLogger("ConfigModule._validate")
        errors = []
        if not (0 < self.risk_per_trade_percentage < 1):
            errors.append(
                f"RISK_PER_TRADE_PERCENTAGE ({self.risk_per_trade_percentage}) must be a potion strength between 0 and 1 (exclusive)."
            )
        if self.leverage < 1:
            errors.append(
                f"LEVERAGE ({self.leverage}) must be at least 1, to amplify the spell's power."
            )
        if self.max_scale_ins < 0:
            errors.append(
                f"MAX_SCALE_INS ({self.max_scale_ins}) cannot be a negative enchantment."
            )
        if self.trailing_stop_percentage < 0:
            errors.append(
                f"TRAILING_STOP_PERCENTAGE ({self.trailing_stop_percentage}) cannot be negative."
            )
        if self.trailing_stop_activation_offset_percent < 0:
            errors.append(
                f"TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT ({self.trailing_stop_activation_offset_percent}) must be positive."
            )

        if (
            self.st_atr_length <= 0
            or self.confirm_st_atr_length <= 0
            or self.momentum_period <= 0
        ):
            errors.append(
                "Strategy periods (ST_ATR_LENGTH, CONFIRM_ST_ATR_LENGTH, MOMENTUM_PERIOD) must be positive integers."
            )
        if self.ehlers_fisher_length <= 0 or self.ehlers_fisher_signal_length <= 0:
            errors.append(
                "Ehlers Fisher lengths (EHLERS_FISHER_LENGTH, EHLERS_FISHER_SIGNAL_LENGTH) must be positive integers."
            )

        if errors:
            error_message = (
                f"Configuration spellcrafting failed with {len(errors)} flaws:\n"
                + "\n".join([f"  - {e}" for e in errors])
            )
            _logger.critical(f"{NEON['CRITICAL']}{error_message}{NEON['RESET']}")
            raise ValueError(error_message)


# --- Logger Setup - The Oracle's Voice ---
LOGGING_LEVEL: int = (
    logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO
)
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
log_file_name = f"{logs_dir}/pyrmethus_spell_v282_{time.strftime('%Y%m%d_%H%M%S')}.log"
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
if sys.stdout.isatty():
    for level, color_code in [
        (logging.DEBUG, NEON["DEBUG"]),
        (logging.INFO, NEON["INFO"]),
        (SUCCESS_LEVEL, NEON["SUCCESS"]),
        (logging.WARNING, NEON["WARNING"]),
        (logging.ERROR, NEON["ERROR"]),
        (logging.CRITICAL, NEON["CRITICAL"]),
    ]:
        logging.addLevelName(
            level, f"{color_code}{logging.getLevelName(level)}{NEON['RESET']}"
        )

# --- Global Objects - Instantiated Arcana ---
try:
    CONFIG = Config()
except ValueError as config_error:
    logging.getLogger("PyrmethusCore").critical(
        f"{NEON['CRITICAL']}Configuration loading failed. Error: {config_error}{NEON['RESET']}"
    )
    if "termux-api" in os.getenv("PATH", ""):
        send_termux_notification(
            "Pyrmethus Startup Failure", f"Config Error: {str(config_error)[:200]}"
        )
    sys.exit(1)
except Exception as general_config_error:
    logging.getLogger("PyrmethusCore").critical(
        f"{NEON['CRITICAL']}Unexpected critical error during configuration: {general_config_error}{NEON['RESET']}"
    )
    logging.getLogger("PyrmethusCore").debug(traceback.format_exc())
    if "termux-api" in os.getenv("PATH", ""):
        send_termux_notification(
            "Pyrmethus Critical Failure",
            f"Unexpected Config Error: {str(general_config_error)[:200]}",
        )
    sys.exit(1)


# --- Trading Strategy Abstract Base Class & Implementations - The Schools of Magic ---
class TradingStrategy(ABC):
    """The archetypal form for all trading spells."""

    def __init__(self, config: Config, df_columns: Optional[List[str]] = None):
        self.config = config
        self.logger = logging.getLogger(
            f"Strategy.{self.__class__.__name__}"
        )  # Each strategy has its own voice
        self.required_columns = (
            df_columns if df_columns else []
        )  # Runes needed by the spell
        self.logger.info(
            f"{NEON['STRATEGY']}Strategy Form '{self.__class__.__name__}' materializing...{NEON['RESET']}"
        )

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates trading signals based on the input DataFrame, which holds the market's omens.
        Returns a dictionary of signals: enter_long, enter_short, exit_long, exit_short, exit_reason.
        """
        pass  # Each spell must define its own incantation

    def _validate_df(self, df: pd.DataFrame, min_rows: int = 2) -> bool:
        """
        Validates the input DataFrame (the scroll of market data) for common flaws.
        Returns True if the scroll is fit for divination, False otherwise.
        """
        if df is None or df.empty or len(df) < min_rows:
            self.logger.debug(
                f"Insufficient market whispers (Rows: {len(df) if df is not None else 0}, Min Required: {min_rows}). Awaiting clearer omens."
            )
            return False
        if self.required_columns and not all(
            col in df.columns for col in self.required_columns
        ):
            missing_cols = [
                col for col in self.required_columns if col not in df.columns
            ]
            self.logger.warning(
                f"{NEON['WARNING']}Market scroll is missing required runes for this strategy: {NEON['VALUE']}{missing_cols}{NEON['WARNING']}. Cannot divine signals.{NEON['RESET']}"
            )
            return False

        if self.required_columns:
            last_row_values = df.iloc[-1][self.required_columns]
            if last_row_values.isnull().any():
                nan_cols_last_row = last_row_values[
                    last_row_values.isnull()
                ].index.tolist()
                self.logger.debug(
                    f"Faint runes (NaNs) in last candle for critical columns: {nan_cols_last_row}. Signals may be unreliable or delayed."
                )
        return True

    def _get_default_signals(self) -> Dict[str, Any]:
        """Returns a default signal dictionary (all signals false), when the omens are unclear."""
        return {
            "enter_long": False,
            "enter_short": False,
            "exit_long": False,
            "exit_short": False,
            "exit_reason": "Default Signal - Awaiting True Omens",
        }


class DualSupertrendMomentumStrategy(TradingStrategy):
    def __init__(self, config: Config):
        super().__init__(
            config,
            df_columns=["st_long_flip", "st_short_flip", "confirm_trend", "momentum"],
        )

    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        signals = self._get_default_signals()
        min_rows_needed = (
            max(
                self.config.st_atr_length,
                self.config.confirm_st_atr_length,
                self.config.momentum_period,
            )
            + 10
        )
        if not self._validate_df(df, min_rows=min_rows_needed):
            return signals

        last = df.iloc[-1]
        primary_long_flip = last.get("st_long_flip", False)
        primary_short_flip = last.get("st_short_flip", False)
        confirm_is_up = last.get("confirm_trend", pd.NA)
        momentum_val_orig = last.get("momentum", pd.NA)
        momentum_val = safe_decimal_conversion(momentum_val_orig, pd.NA)

        if pd.isna(confirm_is_up) or pd.isna(momentum_val):
            self.logger.debug(
                f"Confirmation ST Trend ({_format_for_log(confirm_is_up, is_bool_trend=True)}) or Momentum ({_format_for_log(momentum_val)}) is NA. No signal."
            )
            return signals

        if (
            primary_long_flip
            and confirm_is_up is True
            and momentum_val > self.config.momentum_threshold
        ):
            signals["enter_long"] = True
            self.logger.info(
                f"{NEON['SIDE_LONG']}DualST+Mom Signal: LONG Entry - ST Flip, Confirm Up, Mom ({_format_for_log(momentum_val)}) > {_format_for_log(self.config.momentum_threshold)}{NEON['RESET']}"
            )
        elif (
            primary_short_flip
            and confirm_is_up is False
            and momentum_val < -self.config.momentum_threshold
        ):
            signals["enter_short"] = True
            self.logger.info(
                f"{NEON['SIDE_SHORT']}DualST+Mom Signal: SHORT Entry - ST Flip, Confirm Down, Mom ({_format_for_log(momentum_val)}) < -{_format_for_log(self.config.momentum_threshold)}{NEON['RESET']}"
            )

        if primary_short_flip:
            signals["exit_long"] = True
            signals["exit_reason"] = "Primary SuperTrend Flipped Short"
            self.logger.info(
                f"{NEON['ACTION']}DualST+Mom Signal: EXIT LONG - Primary ST Flipped Short{NEON['RESET']}"
            )
        if primary_long_flip:
            signals["exit_short"] = True
            signals["exit_reason"] = "Primary SuperTrend Flipped Long"
            self.logger.info(
                f"{NEON['ACTION']}DualST+Mom Signal: EXIT SHORT - Primary ST Flipped Long{NEON['RESET']}"
            )
        return signals


class EhlersFisherStrategy(TradingStrategy):
    def __init__(self, config: Config):
        super().__init__(config, df_columns=["ehlers_fisher", "ehlers_signal"])

    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        signals = self._get_default_signals()
        min_rows_needed = (
            self.config.ehlers_fisher_length
            + self.config.ehlers_fisher_signal_length
            + 5
        )
        if not self._validate_df(df, min_rows=min_rows_needed) or len(df) < 2:
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
                f"Ehlers Fisher or Signal rune is NA. Fisher: {_format_for_log(fisher_now)}, Signal: {_format_for_log(signal_now)}. No signal."
            )
            return signals

        if fisher_prev <= signal_prev and fisher_now > signal_now:
            signals["enter_long"] = True
            self.logger.info(
                f"{NEON['SIDE_LONG']}EhlersFisher Signal: LONG Entry - Fisher ({_format_for_log(fisher_now)}) crossed ABOVE Signal ({_format_for_log(signal_now)}){NEON['RESET']}"
            )
        elif fisher_prev >= signal_prev and fisher_now < signal_now:
            signals["enter_short"] = True
            self.logger.info(
                f"{NEON['SIDE_SHORT']}EhlersFisher Signal: SHORT Entry - Fisher ({_format_for_log(fisher_now)}) crossed BELOW Signal ({_format_for_log(signal_now)}){NEON['RESET']}"
            )

        if fisher_prev >= signal_prev and fisher_now < signal_now:
            signals["exit_long"] = True
            signals["exit_reason"] = "Ehlers Fisher crossed BELOW Signal"
            self.logger.info(
                f"{NEON['ACTION']}EhlersFisher Signal: EXIT LONG - Fisher ({_format_for_log(fisher_now)}) crossed BELOW Signal ({_format_for_log(signal_now)}){NEON['RESET']}"
            )
        elif fisher_prev <= signal_prev and fisher_now > signal_now:
            signals["exit_short"] = True
            signals["exit_reason"] = "Ehlers Fisher crossed ABOVE Signal"
            self.logger.info(
                f"{NEON['ACTION']}EhlersFisher Signal: EXIT SHORT - Fisher ({_format_for_log(fisher_now)}) crossed ABOVE Signal ({_format_for_log(signal_now)}){NEON['RESET']}"
            )
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
    )
else:
    err_msg = f"Failed to init strategy '{CONFIG.strategy_name.value}'."
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
            f"{NEON['INFO']}TradeMetrics Ledger opened, ready to chronicle deeds.{NEON['RESET']}"
        )

    def set_initial_equity(self, equity: Decimal):
        if self.initial_equity is None:
            self.initial_equity = equity
            self.logger.info(
                f"{NEON['INFO']}Initial Session Equity rune inscribed: {NEON['VALUE']}{equity:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}"
            )
        today = datetime.now(pytz.utc).day
        if self.last_daily_reset_day != today or self.daily_start_equity is None:
            self.daily_start_equity = equity
            self.last_daily_reset_day = today
            self.logger.info(
                f"{NEON['INFO']}Daily Equity Ward reset. Dawn Equity: {NEON['VALUE']}{equity:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}"
            )

    def check_drawdown(self, current_equity: Decimal) -> Tuple[bool, str]:
        if (
            not CONFIG.enable_max_drawdown_stop
            or self.daily_start_equity is None
            or self.daily_start_equity <= 0
        ):
            return False, ""
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
        pnl_str: str,  # Accept pnl_str
        scale_order_id: Optional[str] = None,  # Keep optional params from v2.8.1
        mae: Optional[Decimal] = None,
        mfe: Optional[Decimal] = None,
    ):
        """Scribes a completed trade part into the Grand Ledger, using provided PNL string."""
        if not all(
            [
                isinstance(entry_price, Decimal) and entry_price > 0,
                isinstance(exit_price, Decimal) and exit_price > 0,
                isinstance(qty, Decimal) and qty > 0,
                isinstance(entry_time_ms, int) and entry_time_ms > 0,
                isinstance(exit_time_ms, int) and exit_time_ms > 0,
            ]
        ):
            self.logger.warning(
                f"{NEON['WARNING']}Trade log skipped due to flawed parameters. Part ID: {part_id}{NEON['RESET']}"
            )
            return

        profit = safe_decimal_conversion(pnl_str, Decimal(0))  # Use provided PNL string

        entry_dt_iso = datetime.fromtimestamp(
            entry_time_ms / 1000, tz=pytz.utc
        ).isoformat()
        exit_dt_iso = datetime.fromtimestamp(
            exit_time_ms / 1000, tz=pytz.utc
        ).isoformat()
        duration_seconds = (
            datetime.fromisoformat(exit_dt_iso.replace("Z", "+00:00"))
            - datetime.fromisoformat(entry_dt_iso.replace("Z", "+00:00"))
        ).total_seconds()

        trade_type = (
            "Scale-In" if scale_order_id else "Part"
        )  # Simplified type based on scale_order_id

        self.trades.append(
            {
                "symbol": symbol,
                "side": side,
                "entry_price_str": str(entry_price),
                "exit_price_str": str(exit_price),
                "qty_str": str(qty),
                "profit_str": str(profit),
                "entry_time_iso": entry_dt_iso,
                "exit_time_iso": exit_dt_iso,
                "duration_seconds": duration_seconds,
                "exit_reason": reason,
                "type": trade_type,
                "part_id": part_id,
                "scale_order_id": scale_order_id,
                "mae_str": str(mae) if mae is not None else None,
                "mfe_str": str(mfe) if mfe is not None else None,
            }
        )

        pnl_color = (
            NEON["PNL_POS"]
            if profit > 0
            else (NEON["PNL_NEG"] if profit < 0 else NEON["PNL_ZERO"])
        )
        self.logger.log(
            SUCCESS_LEVEL,
            f"{NEON['HEADING']}Trade Chronicle ({trade_type}:{part_id}): "
            f"{side.upper()} {NEON['QTY']}{qty}{NEON['RESET']} {symbol.split('/')[0]} | "
            f"P/L: {pnl_color}{profit:.2f} {CONFIG.usdt_symbol}{NEON['RESET']} | Reason: {reason}",
        )

    def summary(self) -> str:
        if not self.trades:
            return f"{NEON['INFO']}The Grand Ledger is empty.{NEON['RESET']}"
        total_trades = len(self.trades)
        wins = sum(1 for t in self.trades if Decimal(t["profit_str"]) > 0)
        losses = sum(1 for t in self.trades if Decimal(t["profit_str"]) < 0)
        breakeven = total_trades - wins - losses
        win_rate = (
            (Decimal(wins) / Decimal(total_trades)) * Decimal(100)
            if total_trades > 0
            else Decimal(0)
        )
        total_profit = sum(Decimal(t["profit_str"]) for t in self.trades)
        avg_profit = (
            total_profit / Decimal(total_trades) if total_trades > 0 else Decimal(0)
        )

        summary_str = (
            f"\n{NEON['HEADING']}--- Pyrmethus Trade Metrics Summary (v2.8.2) ---{NEON['RESET']}\n"
            f"Total Trade Parts: {NEON['VALUE']}{total_trades}{NEON['RESET']}\n"
            f"  Wins: {NEON['PNL_POS']}{wins}{NEON['RESET']}, Losses: {NEON['PNL_NEG']}{losses}{NEON['RESET']}, Breakeven: {NEON['PNL_ZERO']}{breakeven}{NEON['RESET']}\n"
            f"Win Rate: {NEON['VALUE']}{win_rate:.2f}%{NEON['RESET']}\n"
            f"Total P/L: {(NEON['PNL_POS'] if total_profit > 0 else (NEON['PNL_NEG'] if total_profit < 0 else NEON['PNL_ZERO']))}{total_profit:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}\n"
            f"Avg P/L per Part: {(NEON['PNL_POS'] if avg_profit > 0 else (NEON['PNL_NEG'] if avg_profit < 0 else NEON['PNL_ZERO']))}{avg_profit:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}\n"
        )
        if self.initial_equity is not None:
            current_equity_approx = self.initial_equity + total_profit
            overall_pnl_pct = (
                (total_profit / self.initial_equity) * 100
                if self.initial_equity > 0
                else Decimal(0)
            )
            summary_str += f"Initial Session Equity: {NEON['VALUE']}{self.initial_equity:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}\n"
            summary_str += f"Approx. Current Equity: {NEON['VALUE']}{current_equity_approx:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}\n"
            summary_str += f"Overall Session P/L %: {(NEON['PNL_POS'] if overall_pnl_pct > 0 else (NEON['PNL_NEG'] if overall_pnl_pct < 0 else NEON['PNL_ZERO']))}{overall_pnl_pct:.2f}%{NEON['RESET']}\n"
        summary_str += f"{NEON['HEADING']}--- End of Ledger Reading ---{NEON['RESET']}"
        self.logger.info(summary_str)
        return summary_str

    def get_serializable_trades(self) -> List[Dict[str, Any]]:
        return self.trades

    def load_trades_from_list(self, trades_list: List[Dict[str, Any]]):
        self.trades = trades_list
        self.logger.info(
            f"{NEON['INFO']}TradeMetrics: Re-inked {NEON['VALUE']}{len(self.trades)}{NEON['INFO']} trade sagas.{NEON['RESET']}"
        )


trade_metrics = TradeMetrics()
_active_trade_parts: List[Dict[str, Any]] = []
_last_heartbeat_save_time: float = 0.0


# --- State Persistence Functions (Phoenix Feather) ---
def save_persistent_state(force_heartbeat: bool = False) -> None:
    """Scribes the current state of Pyrmethus's essence onto the Phoenix scroll for later reawakening."""
    global _active_trade_parts, trade_metrics, _last_heartbeat_save_time
    now = time.time()
    if not (
        force_heartbeat
        or (now - _last_heartbeat_save_time > HEARTBEAT_INTERVAL_SECONDS)
    ):
        return

    logger.debug(
        f"{NEON['COMMENT']}# Phoenix Feather prepares to scribe memories...{NEON['RESET']}"
    )
    try:
        serializable_active_parts = []
        for part in _active_trade_parts:
            p_copy = part.copy()
            for key, value in p_copy.items():
                if isinstance(value, Decimal):
                    p_copy[key] = str(value)
                elif isinstance(value, (datetime, pd.Timestamp)):
                    p_copy[key] = value.isoformat()
                # entry_time_ms should be int
            serializable_active_parts.append(p_copy)

        state_data = {
            "pyrmethus_version": "2.8.2",  # Current spell version
            "timestamp_utc_iso": datetime.now(pytz.utc).isoformat(),
            "active_trade_parts": serializable_active_parts,
            "trade_metrics_trades": trade_metrics.get_serializable_trades(),
            "config_symbol": CONFIG.symbol,
            "config_strategy": CONFIG.strategy_name.value,
            "daily_start_equity_str": str(trade_metrics.daily_start_equity)
            if trade_metrics.daily_start_equity is not None
            else None,
            "last_daily_reset_day": trade_metrics.last_daily_reset_day,
        }
        temp_file_path = STATE_FILE_PATH + ".tmp"
        with open(temp_file_path, "w") as f:
            json.dump(state_data, f, indent=4)
        os.replace(temp_file_path, STATE_FILE_PATH)
        _last_heartbeat_save_time = now
        log_level = logging.INFO if force_heartbeat else logging.DEBUG
        logger.log(
            log_level,
            f"{NEON['SUCCESS']}Phoenix Feather: State memories scribed to '{STATE_FILE_NAME}'.{NEON['RESET']}",
        )
    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}Phoenix Feather Error: Failed to scribe state: {e}{NEON['RESET']}"
        )
        logger.debug(traceback.format_exc())


def load_persistent_state() -> bool:
    """Attempts to reawaken Pyrmethus's essence from a previously scribed Phoenix scroll."""
    global _active_trade_parts, trade_metrics
    logger.info(
        f"{NEON['COMMENT']}# Phoenix Feather seeks past memories from '{STATE_FILE_PATH}'...{NEON['RESET']}"
    )
    if not os.path.exists(STATE_FILE_PATH):
        logger.info(
            f"{NEON['INFO']}Phoenix Feather: No previous scroll found. Starting fresh.{NEON['RESET']}"
        )
        return False
    try:
        with open(STATE_FILE_PATH, "r") as f:
            state_data = json.load(f)

        saved_version = state_data.get("pyrmethus_version", "unknown")
        if saved_version != "2.8.2":  # Check against current version "2.8.2"
            logger.warning(
                f"{NEON['WARNING']}Phoenix Feather: Scroll version '{saved_version}' differs from current '2.8.2'. Caution advised.{NEON['RESET']}"
            )

        if (
            state_data.get("config_symbol") != CONFIG.symbol
            or state_data.get("config_strategy") != CONFIG.strategy_name.value
        ):
            logger.warning(
                f"{NEON['WARNING']}Phoenix Scroll sigils mismatch (symbol/strategy). Current: {CONFIG.symbol}/{CONFIG.strategy_name.value}. Saved: {state_data.get('config_symbol')}/{state_data.get('config_strategy')}. Ignoring old scroll.{NEON['RESET']}"
            )
            archive_name = f"{STATE_FILE_PATH}.archived_{time.strftime('%Y%m%d%H%M%S')}"
            try:
                os.rename(STATE_FILE_PATH, archive_name)
                logger.info(f"Archived to: {archive_name}")
            except OSError as e_mv:
                logger.error(f"Error archiving: {e_mv}")
            return False

        _active_trade_parts.clear()
        for part_data in state_data.get("active_trade_parts", []):
            restored = part_data.copy()
            # Keys specific to v2.8.2's _active_trade_parts structure that might be Decimal
            decimal_keys = [
                "entry_price",
                "qty",
                "sl_price",
                "initial_usdt_value",
                "atr_at_entry",
            ]
            for k, v_str_or_val in restored.items():
                if k in decimal_keys and isinstance(v_str_or_val, str):
                    try:
                        restored[k] = Decimal(v_str_or_val)
                    except InvalidOperation:
                        logger.warning(
                            f"Bad Decimal '{v_str_or_val}' for '{k}' in loaded part."
                        )
                elif k == "entry_time_ms":
                    if isinstance(v_str_or_val, str):
                        try:
                            restored[k] = int(
                                datetime.fromisoformat(
                                    v_str_or_val.replace("Z", "+00:00")
                                ).timestamp()
                                * 1000
                            )
                        except ValueError:
                            try:
                                restored[k] = int(
                                    v_str_or_val
                                )  # Try direct int if not ISO
                            except ValueError:
                                logger.warning(
                                    f"Bad entry_time_ms '{v_str_or_val}' in loaded part."
                                )
                    elif isinstance(v_str_or_val, (float, int)):
                        restored[k] = int(v_str_or_val)
            _active_trade_parts.append(restored)

        trade_metrics.load_trades_from_list(state_data.get("trade_metrics_trades", []))
        daily_equity_str = state_data.get("daily_start_equity_str")
        if daily_equity_str and daily_equity_str.lower() != "none":
            try:
                trade_metrics.daily_start_equity = Decimal(daily_equity_str)
            except InvalidOperation:
                logger.warning(
                    f"Bad daily_start_equity '{daily_equity_str}' in scroll."
                )
        trade_metrics.last_daily_reset_day = state_data.get("last_daily_reset_day")

        logger.success(
            f"{NEON['SUCCESS']}Phoenix Feather: Memories reawakened! Active Parts: {len(_active_trade_parts)}, Trades: {len(trade_metrics.trades)}{NEON['RESET']}"
        )
        return True
    except json.JSONDecodeError as e:
        logger.error(
            f"{NEON['ERROR']}Phoenix Feather: Scroll '{STATE_FILE_PATH}' corrupted: {e}. Starting fresh.{NEON['RESET']}"
        )
        archive_name = f"{STATE_FILE_PATH}.corrupted_{time.strftime('%Y%m%d%H%M%S')}"
        try:
            os.rename(STATE_FILE_PATH, archive_name)
            logger.info(f"Archived to: {archive_name}")
        except OSError as e_mv:
            logger.error(f"Error archiving: {e_mv}")
        return False
    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}Phoenix Feather Error reawakening: {e}{NEON['RESET']}"
        )
        logger.debug(traceback.format_exc())
        return False


# --- Indicator Calculation - The Alchemist's Art ---
def calculate_all_indicators(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    The grand alchemy chamber where raw market data (OHLCV) is transmuted
    by applying various technical indicators.
    """
    logger.debug(
        f"{NEON['COMMENT']}# Transmuting market data with indicator alchemy...{NEON['RESET']}"
    )
    if df.empty:
        logger.warning(
            f"{NEON['WARNING']}Market data scroll empty. No indicators conjured.{NEON['RESET']}"
        )
        return df

    for col in ["close", "high", "low", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if config.strategy_name == StrategyName.DUAL_SUPERTREND_MOMENTUM:
        st_p = f"ST_{config.st_atr_length}_{config.st_multiplier}"
        st_c = (
            f"CONFIRM_ST_{config.confirm_st_atr_length}_{config.confirm_st_multiplier}"
        )
        df.ta.supertrend(
            length=config.st_atr_length,
            multiplier=float(config.st_multiplier),
            append=True,
            col_names=(st_p, f"{st_p}d", f"{st_p}l", f"{st_p}s"),
        )
        df.ta.supertrend(
            length=config.confirm_st_atr_length,
            multiplier=float(config.confirm_st_multiplier),
            append=True,
            col_names=(st_c, f"{st_c}d", f"{st_c}l", f"{st_c}s"),
        )

        if f"{st_p}d" in df.columns:
            df["st_trend_up"] = df[f"{st_p}d"] == 1
            df["st_long_flip"] = (df["st_trend_up"]) & (
                df["st_trend_up"].shift(1) == False
            )
            df["st_short_flip"] = (df["st_trend_up"] == False) & (
                df["st_trend_up"].shift(1)
            )
        else:
            df["st_long_flip"], df["st_short_flip"] = False, False
            logger.error(f"Primary ST direction col '{f'{st_p}d'}' missing!")

        if f"{st_c}d" in df.columns:
            df["confirm_trend"] = df[f"{st_c}d"].apply(
                lambda x: True if x == 1 else (False if x == -1 else pd.NA)
            )
        else:
            df["confirm_trend"] = pd.NA
            logger.error(f"Confirm ST direction col '{f'{st_c}d'}' missing!")

        if "close" in df.columns and not df["close"].isnull().all():
            df.ta.mom(
                length=config.momentum_period, append=True, col_names=("momentum",)
            )
        else:
            df["momentum"] = pd.NA
            logger.warning("Cannot calculate momentum, 'close' missing/all NaNs.")
        logger.debug(
            f"Dual ST & Mom. Last Mom: {_format_for_log(df['momentum'].iloc[-1] if 'momentum' in df.columns and not df.empty else pd.NA)}"
        )

    elif config.strategy_name == StrategyName.EHLERS_FISHER:
        if (
            "high" in df.columns
            and "low" in df.columns
            and not (df["high"].isnull().all() or df["low"].isnull().all())
        ):
            df.ta.fisher(
                length=config.ehlers_fisher_length,
                signal=config.ehlers_fisher_signal_length,
                append=True,
                col_names=("ehlers_fisher", "ehlers_signal"),
            )
        else:
            df["ehlers_fisher"], df["ehlers_signal"] = pd.NA, pd.NA
            logger.warning("Cannot calc Ehlers, H/L missing/all NaNs.")
        logger.debug(
            f"Ehlers Fisher. Last Fisher: {_format_for_log(df['ehlers_fisher'].iloc[-1] if 'ehlers_fisher' in df.columns and not df.empty else pd.NA)}"
        )

    atr_col_name = f"ATR_{config.atr_calculation_period}"
    if "high" in df.columns and "low" in df.columns and "close" in df.columns:
        df.ta.atr(
            length=config.atr_calculation_period, append=True, col_names=(atr_col_name,)
        )
    else:
        df[atr_col_name] = pd.NA
        logger.warning(f"Cannot calculate {atr_col_name}, missing HLC data.")
    return df


# --- Exchange Interaction Primitives - Incantations for the Market Realm ---
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
    logger.debug(
        f"{NEON['COMMENT']}# Summoning {limit} market whispers (candles) for {symbol} ({interval})...{NEON['RESET']}"
    )
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, interval, limit=limit)
        if not ohlcv:
            logger.warning(
                f"{NEON['WARNING']}No OHLCV runes returned for {symbol}. The market is silent.{NEON['RESET']}"
            )
            return None
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        logger.debug(
            f"Summoned {len(df)} candles. Latest omen: {df['timestamp'].iloc[-1] if not df.empty else 'N/A'}"
        )
        return df
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
    logger.debug(
        f"{NEON['COMMENT']}# Scrying the treasury for {currency_code} balance...{NEON['RESET']}"
    )
    try:
        balance_data = exchange.fetch_balance()
        # Bybit V5 unified account might structure total differently, check 'total' or 'walletBalance' under USDT
        total_balance = safe_decimal_conversion(
            balance_data.get("total", {}).get(currency_code)
        )  # Try this first
        if total_balance is None:  # Fallback for different structures
            total_balance = safe_decimal_conversion(
                balance_data.get(currency_code, {}).get("total")
            )

        if total_balance is None:
            logger.warning(
                f"{NEON['WARNING']}{currency_code} balance not found or rune unreadable.{NEON['RESET']}"
            )
            return None
        logger.info(
            f"{NEON['INFO']}Current {currency_code} Treasury: {NEON['VALUE']}{total_balance:.2f}{NEON['RESET']}"
        )
        return total_balance
    except Exception as e:
        logger.error(f"{NEON['ERROR']}Error scrying treasury: {e}{NEON['RESET']}")
        return None


def get_current_position_info() -> Tuple[str, Decimal]:
    """Determines current position side and quantity from Pyrmethus's internal memory (_active_trade_parts)."""
    global _active_trade_parts
    if not _active_trade_parts:
        return CONFIG.pos_none, Decimal(0)
    # Assuming coherent state: all parts are in the same direction if any exist.
    total_qty = sum(
        part.get("qty", Decimal(0)) for part in _active_trade_parts
    )  # Sum quantities of all active parts
    current_side = (
        _active_trade_parts[0]["side"]
        if total_qty > CONFIG.position_qty_epsilon
        else CONFIG.pos_none
    )  # If total qty is effectively zero, consider flat
    return current_side, total_qty


def calculate_position_size(
    balance: Decimal,
    risk_per_trade_pct: Decimal,
    entry_price: Decimal,
    sl_price: Decimal,
    market_info: Dict,
) -> Optional[Decimal]:
    logger.debug(
        f"{NEON['COMMENT']}# Calculating alchemical proportions for position size... Balance: {balance:.2f}, Risk %: {risk_per_trade_pct:.4f}{NEON['RESET']}"
    )
    if balance <= 0 or entry_price <= 0 or sl_price <= 0:
        logger.warning(
            f"{NEON['WARNING']}Invalid runes for position sizing (balance/prices <=0).{NEON['RESET']}"
        )
        return None

    risk_per_unit = abs(
        entry_price - sl_price
    )  # The potential loss per unit of the base asset
    if risk_per_unit == 0:
        logger.warning(
            f"{NEON['WARNING']}Risk per unit is zero (entry matches SL). Cannot divine size.{NEON['RESET']}"
        )
        return None

    usdt_at_risk = (
        balance * risk_per_trade_pct
    )  # Total USDT amount Pyrmethus is willing to risk
    logger.debug(f"USDT at risk for this venture: {usdt_at_risk:.2f}")

    # Quantity in base currency (e.g., BTC for BTC/USDT)
    # This is the quantity of base asset where loss from entry to SL equals usdt_at_risk
    quantity_base = usdt_at_risk / risk_per_unit

    # The USDT value of the position (notional value before leverage application by exchange for margin)
    position_usdt_value = quantity_base * entry_price

    if position_usdt_value > CONFIG.max_order_usdt_amount:
        logger.info(
            f"Calculated position USDT value {NEON['VALUE']}{position_usdt_value:.2f}{NEON['RESET']} exceeds MAX_ORDER_USDT_AMOUNT {NEON['VALUE']}{CONFIG.max_order_usdt_amount}{NEON['RESET']}. Capping."
        )
        position_usdt_value = CONFIG.max_order_usdt_amount
        quantity_base = (
            position_usdt_value / entry_price
        )  # Recalculate base quantity based on capped USDT value

    # Adjust for market precision (stepSize) and limits (minQty)
    min_qty = safe_decimal_conversion(
        market_info.get("limits", {}).get("amount", {}).get("min")
    )
    qty_precision_digits = market_info.get("precision", {}).get(
        "amount"
    )  # Number of decimal places for quantity

    # CCXT often provides precision as number of decimal places. If it's a step size (e.g., 0.001), convert.
    # Assuming qty_precision_digits is number of decimal places as per Bybit V5 typical response.
    # If it's a direct step size (e.g. Decimal('0.001')), the logic below needs adjustment.
    # For now, assume it's number of decimal places.

    if qty_precision_digits is None:  # Fallback if precision not found
        logger.warning(
            f"{NEON['WARNING']}Quantity precision rune not found for {market_info.get('symbol')}. Using raw calculation.{NEON['RESET']}"
        )
    else:
        qty_precision_digits = int(qty_precision_digits)  # Ensure it's int
        if (
            qty_precision_digits >= 0
        ):  # If 0, means integer quantity. If >0, decimal places.
            quantizer = Decimal(
                "1e-" + str(qty_precision_digits)
            )  # e.g., 3 -> Decimal('0.001')
            quantity_base = (
                quantity_base // quantizer
            ) * quantizer  # Floor to the nearest step size
        # If qty_precision_digits < 0 (e.g. for rounding to tens, hundreds), pandas_ta might return -1 for 10, -2 for 100.
        # This case is less common for crypto quantities but good to be aware of. Current logic handles >= 0.

    if min_qty is not None and quantity_base < min_qty:  # Check against min order size
        logger.warning(
            f"{NEON['WARNING']}Calculated quantity {NEON['QTY']}{quantity_base}{NEON['WARNING']} is below min {NEON['QTY']}{min_qty}{NEON['WARNING']}. Cannot place order.{NEON['RESET']}"
        )
        return None

    if quantity_base <= CONFIG.position_qty_epsilon:  # Effectively zero
        logger.warning(
            f"{NEON['WARNING']}Final quantity is zero or negligible after adjustments. Cannot place order.{NEON['RESET']}"
        )
        return None

    qty_display_format = (
        f".{qty_precision_digits}f"
        if qty_precision_digits is not None and qty_precision_digits >= 0
        else ""
    )
    logger.info(
        f"Calculated position size: {NEON['QTY']}{quantity_base:{qty_display_format}}{NEON['RESET']} {market_info.get('base', '')}"
    )
    return quantity_base


@retry(
    (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.InsufficientFunds),
    tries=2,
    delay=3,
    logger=logger,
)
def place_risked_order(
    exchange: ccxt.Exchange,
    config: Config,
    side: str,
    entry_price_target: Decimal,
    sl_price: Decimal,
    atr_val: Optional[Decimal],
) -> Optional[Dict]:
    """Places an order with calculated risk, SL, and records the new active trade part."""
    global _active_trade_parts
    logger.info(
        f"{NEON['ACTION']}Attempting to weave a {side.upper()} order for {config.symbol}... Target Entry: {NEON['PRICE']}{entry_price_target}{NEON['RESET']}, SL: {NEON['PRICE']}{sl_price}{NEON['RESET']}"
    )

    balance = fetch_account_balance(exchange, config.usdt_symbol)
    if balance is None or balance <= Decimal(10):  # Min balance ward
        logger.error(
            f"{NEON['ERROR']}Insufficient treasury balance or failed to scry. Cannot weave new order.{NEON['RESET']}"
        )
        config.send_notification_method(
            "Pyrmethus Order Fail", "Insufficient balance for new order."
        )
        return None

    if config.MARKET_INFO is None:
        logger.error(
            f"{NEON['ERROR']}Market runes (MARKET_INFO) not loaded. Cannot size or place order.{NEON['RESET']}"
        )
        return None

    quantity = calculate_position_size(
        balance,
        config.risk_per_trade_percentage,
        entry_price_target,
        sl_price,
        config.MARKET_INFO,
    )
    if quantity is None or quantity <= 0:
        return None  # Sizing failed or resulted in zero quantity

    # Determine order type from config
    order_type_str = (
        "Market" if config.entry_order_type == OrderEntryType.MARKET else "Limit"
    )

    # Bybit V5 API parameters for create_order
    # 'category': 'linear' for USDT perps, 'inverse' for coin-margined. CCXT usually infers from symbol.
    params = {
        "timeInForce": "GTC",  # Good-Til-Cancelled for limit orders, often ignored for market.
        "stopLoss": float(
            sl_price
        ),  # Set SL price directly with the order (Bybit V5 feature)
        "slTriggerBy": "MarkPrice",  # Or 'LastPrice', 'IndexPrice'. MarkPrice is common for SL.
        # 'takeProfit': float(tp_price), # Optional TP can also be set here
        "positionIdx": 0,  # For Hedge Mode: 0 for one-way. For Bybit V5, 1=Buy, 2=Sell under hedge. Assume one-way for now.
        # If using hedge mode, this needs careful management.
    }

    try:
        order_price_param = (
            float(entry_price_target) if order_type_str == "Limit" else None
        )  # Price param only for Limit
        logger.info(
            f"Weaving {order_type_str.upper()} {side.upper()} order: Qty {NEON['QTY']}{quantity}{NEON['RESET']}, Symbol {config.symbol}, SL {NEON['PRICE']}{sl_price}{NEON['RESET']}"
        )

        # Ensure quantity is float for ccxt
        order = exchange.create_order(
            config.symbol,
            order_type_str,
            side,
            float(quantity),
            order_price_param,
            params,
        )

        logger.success(
            f"{NEON['SUCCESS']}Entry Order cast forth: ID {order['id']}, Status {order.get('status', 'N/A')}{NEON['RESET']}"
        )
        config.send_notification_method(
            f"Pyrmethus Order Placed: {side.upper()}",
            f"{config.symbol} Qty: {quantity} @ {entry_price_target if order_type_str == 'Limit' else 'Market'}",
        )

        # Simplified fill check: wait a moment then fetch order status. Robust check is more complex.
        # Market orders usually fill quickly. Limit orders might not.
        time.sleep(
            config.order_fill_timeout_seconds / 3
        )  # Partial timeout to check status
        filled_order = exchange.fetch_order(order["id"], config.symbol)

        if (
            filled_order.get("status") == "closed"
        ):  # 'closed' indicates filled for Bybit
            actual_entry_price = Decimal(
                str(
                    filled_order.get(
                        "average", filled_order.get("price", entry_price_target)
                    )
                )
            )
            actual_qty = Decimal(str(filled_order.get("filled", quantity)))
            entry_timestamp_ms = int(filled_order.get("timestamp", time.time() * 1000))

            part_id = str(uuid.uuid4())[:8]  # Unique sigil for this trade part
            new_part = {
                "part_id": part_id,
                "entry_order_id": order["id"],
                "sl_order_id": None,  # SL is part of entry order with Bybit V5, not a separate order ID here
                "symbol": config.symbol,
                "side": side,
                "entry_price": actual_entry_price,
                "qty": actual_qty,
                "entry_time_ms": entry_timestamp_ms,
                "sl_price": sl_price,
                "atr_at_entry": atr_val,  # Store ATR at entry for reference
                "initial_usdt_value": actual_qty
                * actual_entry_price,  # Notional value at entry
            }
            _active_trade_parts.append(new_part)
            save_persistent_state(force_heartbeat=True)  # Scribe this new venture
            logger.success(
                f"{NEON['SUCCESS']}Order filled! Part {part_id} forged. Entry: {NEON['PRICE']}{actual_entry_price}{NEON['RESET']}, Qty: {NEON['QTY']}{actual_qty}{NEON['RESET']}{NEON['RESET']}"
            )
            config.send_notification_method(
                f"Pyrmethus Order Filled: {side.upper()}",
                f"{config.symbol} Part {part_id} @ {actual_entry_price}",
            )
            return new_part
        else:
            logger.warning(
                f"{NEON['WARNING']}Order {order['id']} not immediately filled. Status: {filled_order.get('status')}. Manual check advised. Consider cancellation for Limit orders.{NEON['RESET']}"
            )
            # If it was a Limit order and not filled, one might cancel it here:
            # if order_type_str == 'Limit':
            #     exchange.cancel_order(order['id'], config.symbol)
            #     logger.info(f"Limit order {order['id']} cancelled due to not filling.")
            return None

    except ccxt.InsufficientFunds as e:
        logger.error(
            f"{NEON['ERROR']}Insufficient funds in treasury to weave order: {e}{NEON['RESET']}"
        )
        config.send_notification_method(
            "Pyrmethus Order Fail", f"Insufficient funds for {config.symbol}"
        )
        return None
    except ccxt.ExchangeError as e:  # Catch specific exchange errors
        logger.error(
            f"{NEON['ERROR']}Exchange rejected the spell (order): {e}{NEON['RESET']}"
        )
        config.send_notification_method(
            "Pyrmethus Order Fail", f"Exchange error: {str(e)[:100]}"
        )
        # Example: Bybit error for SL too close: "Stop loss price is too close to entry price"
        if "Stop loss price is too close" in str(e):  # More specific handling
            logger.warning(
                f"{NEON['WARNING']}SL price {sl_price} likely too close to entry {entry_price_target}. Adjust ATR multiplier or spread.{NEON['RESET']}"
            )
        return None
    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}Unexpected chaos while weaving order: {e}{NEON['RESET']}"
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
    """Closes a specific part of an active position and records the outcome."""
    global _active_trade_parts
    logger.info(
        f"{NEON['ACTION']}Attempting to unravel trade part {part_to_close['part_id']} ({part_to_close['side']} {part_to_close['qty']} {config.symbol}) for reason: {reason}{NEON['RESET']}"
    )

    # Determine the side for the closing order (opposite of the part's side)
    close_side = (
        config.side_sell
        if part_to_close["side"] == config.pos_long
        else config.side_buy
    )
    qty_to_close = part_to_close["qty"]

    try:
        # For Bybit V5, if SL/TP were set with create_order, they are part of the position.
        # A market order with 'reduceOnly' ensures we only close and don't open an opposite position.
        params = {"reduceOnly": True}
        logger.info(
            f"Casting MARKET {close_side.upper()} order to unravel part {part_to_close['part_id']}: Qty {NEON['QTY']}{qty_to_close}{NEON['RESET']}"
        )

        # Ensure quantity is float
        close_order = exchange.create_order(
            config.symbol, "market", close_side, float(qty_to_close), None, params
        )

        logger.success(
            f"{NEON['SUCCESS']}Unraveling Order cast forth: ID {close_order['id']}, Status {close_order.get('status', 'N/A')}{NEON['RESET']}"
        )

        # Simplified: Assume it fills. Robust check needed.
        time.sleep(config.order_fill_timeout_seconds / 3)  # Wait a bit for fill
        filled_close_order = exchange.fetch_order(close_order["id"], config.symbol)

        if filled_close_order.get("status") == "closed":  # Filled
            actual_exit_price = Decimal(
                str(
                    filled_close_order.get(
                        "average",
                        filled_close_order.get(
                            "price", close_price_target or part_to_close["entry_price"]
                        ),
                    )
                )
            )
            exit_timestamp_ms = int(
                filled_close_order.get("timestamp", time.time() * 1000)
            )

            # Calculate PNL for this part
            pnl_per_unit = (
                (actual_exit_price - part_to_close["entry_price"])
                if part_to_close["side"] == config.pos_long
                else (part_to_close["entry_price"] - actual_exit_price)
            )
            pnl = pnl_per_unit * part_to_close["qty"]

            # Log the trade to metrics
            trade_metrics.log_trade(
                symbol=config.symbol,
                side=part_to_close["side"],
                entry_price=part_to_close["entry_price"],
                exit_price=actual_exit_price,
                qty=part_to_close["qty"],
                entry_time_ms=part_to_close["entry_time_ms"],
                exit_time_ms=exit_timestamp_ms,
                reason=reason,
                part_id=part_to_close["part_id"],
                pnl_str=str(pnl),  # Pass calculated PNL as string
            )

            # Remove the closed part from active trades
            _active_trade_parts = [
                p
                for p in _active_trade_parts
                if p["part_id"] != part_to_close["part_id"]
            ]
            save_persistent_state(force_heartbeat=True)  # Scribe the change
            logger.success(
                f"{NEON['SUCCESS']}Part {part_to_close['part_id']} unraveled. Exit: {NEON['PRICE']}{actual_exit_price}{NEON['RESET']}, PNL: {NEON[('PNL_POS' if pnl > 0 else ('PNL_NEG' if pnl < 0 else 'PNL_ZERO'))]}{pnl:.2f}{NEON['RESET']}"
            )
            config.send_notification_method(
                f"Pyrmethus Position Closed",
                f"{config.symbol} Part {part_to_close['part_id']} closed. PNL: {pnl:.2f}",
            )
            return True
        else:
            logger.warning(
                f"{NEON['WARNING']}Unraveling order {close_order['id']} for part {part_to_close['part_id']} not filled. Status: {filled_close_order.get('status')}. Manual check required.{NEON['RESET']}"
            )
            return False

    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}Error unraveling position part {part_to_close['part_id']}: {e}{NEON['RESET']}"
        )
        logger.debug(traceback.format_exc())
        config.send_notification_method(
            "Pyrmethus Close Fail",
            f"Error closing {config.symbol} part {part_to_close['part_id']}: {str(e)[:80]}",
        )
        return False


@retry(
    (ccxt.NetworkError, ccxt.RequestTimeout),
    tries=CONFIG.retry_count,
    delay=CONFIG.retry_delay_seconds,
)
def cancel_all_symbol_orders(exchange: ccxt.Exchange, symbol: str):
    """Dispels all open orders for a given symbol on the exchange."""
    logger.info(
        f"{NEON['ACTION']}Dispelling all open enchantments (orders) for {symbol}...{NEON['RESET']}"
    )
    try:
        # Bybit V5 might need category for cancelAllOrders if it's not inferred
        # params = {'category': 'linear'} # if symbol is linear, e.g. BTC/USDT:USDT
        # exchange.cancel_all_orders(symbol, params)
        exchange.cancel_all_orders(symbol)  # Simpler call, ccxt might handle category
        logger.success(
            f"{NEON['SUCCESS']}All open enchantments for {symbol} dispelled.{NEON['RESET']}"
        )
    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}Error dispelling enchantments for {symbol}: {e}{NEON['RESET']}"
        )


def close_all_symbol_positions(exchange: ccxt.Exchange, config: Config, reason: str):
    """Forcefully unravels all active position parts for the configured symbol and cancels orders."""
    global _active_trade_parts
    logger.warning(
        f"{NEON['WARNING']}Unraveling ALL positions for {config.symbol} due to: {reason}{NEON['RESET']}"
    )

    # Close parts known to Pyrmethus
    parts_to_close_copy = list(
        _active_trade_parts
    )  # Iterate over a copy as list is modified
    for part in parts_to_close_copy:
        if part["symbol"] == config.symbol:
            logger.info(f"Unraveling internally tracked part: {part['part_id']}")
            close_position_part(exchange, config, part, reason + " (Global Close)")

    # As a safeguard, fetch current position from exchange and close if any residual amount exists
    # This is crucial if Pyrmethus's internal state desynchronized.
    try:
        # For Bybit V5, fetch_positions might need category if not inferred by symbol
        # positions = exchange.fetch_positions([config.symbol], {'category': 'linear'})
        positions = exchange.fetch_positions([config.symbol])
        for pos_data in positions:
            if pos_data and safe_decimal_conversion(
                pos_data.get("contracts", "0")
            ) != Decimal(0):  # 'contracts' or 'size'
                pos_symbol = pos_data.get("symbol")
                pos_side = pos_data.get("side")  # 'long' or 'short'
                pos_qty = safe_decimal_conversion(pos_data.get("contracts", "0"))

                if (
                    pos_symbol == config.symbol
                    and pos_qty > CONFIG.position_qty_epsilon
                ):
                    side_to_close_market = (
                        config.side_sell if pos_side == "long" else config.side_buy
                    )
                    logger.warning(
                        f"Found residual exchange position: {pos_side} {NEON['QTY']}{pos_qty}{NEON['RESET']} {config.symbol}. Attempting forceful market unraveling."
                    )
                    exchange.create_order(
                        config.symbol,
                        "market",
                        side_to_close_market,
                        float(pos_qty),
                        params={"reduceOnly": True},
                    )
                    logger.info(
                        f"Residual position unraveling order cast for {NEON['QTY']}{pos_qty}{NEON['RESET']} {config.symbol}."
                    )
                    # Give it a moment to process
                    time.sleep(config.post_close_delay_seconds)
    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}Error during final scrying/unraveling of exchange positions: {e}{NEON['RESET']}"
        )

    cancel_all_symbol_orders(
        exchange, config.symbol
    )  # Also dispel any straggling conditional orders


# --- Main Spell Weaving - The Heart of Pyrmethus ---
def main_loop(exchange: ccxt.Exchange, config: Config) -> None:
    logger.info(
        f"{NEON['HEADING']}=== Pyrmethus Spell v2.8.2 Awakening on {exchange.id} ==={NEON['RESET']}"
    )
    logger.info(
        f"{NEON['COMMENT']}# Listening to whispers for {config.symbol} on {config.interval}...{NEON['RESET']}"
    )

    if load_persistent_state():
        logger.success(
            f"{NEON['SUCCESS']}Successfully reawakened from Phoenix scroll.{NEON['RESET']}"
        )
    else:
        logger.info(
            f"{NEON['INFO']}No valid prior state. Starting with a fresh essence.{NEON['RESET']}"
        )

    current_balance = fetch_account_balance(exchange, config.usdt_symbol)
    if current_balance is not None:
        trade_metrics.set_initial_equity(current_balance)
    else:
        logger.warning(
            f"{NEON['WARNING']}Could not set initial treasury reading. Balance scrying failed.{NEON['RESET']}"
        )

    while True:
        try:
            logger.debug(
                f"{NEON['COMMENT']}# New cycle of observation... The market's currents shift...{NEON['RESET']}"
            )
            current_balance = fetch_account_balance(
                exchange, config.usdt_symbol
            )  # Update balance each cycle
            if current_balance is not None:
                trade_metrics.set_initial_equity(
                    current_balance
                )  # This also handles daily reset for drawdown
                drawdown_hit, dd_reason = trade_metrics.check_drawdown(current_balance)
                if drawdown_hit:
                    logger.critical(
                        f"{NEON['CRITICAL']}Max drawdown ward triggered! {dd_reason}. Pyrmethus must rest to conserve essence.{NEON['RESET']}"
                    )
                    close_all_symbol_positions(exchange, config, "Max Drawdown Reached")
                    break  # Exit the main loop, ending the spell for now
            else:
                logger.warning(
                    f"{NEON['WARNING']}Failed to scry treasury balance this cycle.{NEON['RESET']}"
                )

            # Fetch market data (OHLCV)
            ohlcv_df = get_market_data(
                exchange, config.symbol, config.interval, limit=OHLCV_LIMIT
            )
            if ohlcv_df is None or ohlcv_df.empty:
                logger.warning(
                    f"{NEON['WARNING']}Market whispers are faint (no data). Retrying after a brief pause.{NEON['RESET']}"
                )
                time.sleep(config.sleep_seconds)
                continue

            # Calculate all necessary indicators
            df_with_indicators = calculate_all_indicators(ohlcv_df.copy(), config)
            if (
                df_with_indicators.empty or df_with_indicators.iloc[-1].isnull().all()
            ):  # Check if last row is all NaN
                logger.warning(
                    f"{NEON['WARNING']}Indicator alchemy yielded no clear runes or latest runes are faint. Pausing.{NEON['RESET']}"
                )
                time.sleep(config.sleep_seconds)
                continue

            # Generate trading signals from the chosen strategy
            signals = config.strategy_instance.generate_signals(df_with_indicators)
            logger.debug(
                f"Divine Omens (Signals): EnterL={signals['enter_long']}, EnterS={signals['enter_short']}, ExitL={signals['exit_long']}, ExitS={signals['exit_short']}"
            )

            # Get current position state from Pyrmethus's memory
            current_pos_side, current_pos_qty = get_current_position_info()
            logger.debug(
                f"Current Stance: Side={NEON[('SIDE_LONG' if current_pos_side == config.pos_long else ('SIDE_SHORT' if current_pos_side == config.pos_short else 'SIDE_FLAT'))]}{current_pos_side}{NEON['RESET']}, Qty={NEON['QTY']}{current_pos_qty}{NEON['RESET']}"
            )

            # Extract latest ATR and Close price for SL calculation and decision making
            atr_col = f"ATR_{config.atr_calculation_period}"
            latest_atr = (
                safe_decimal_conversion(df_with_indicators[atr_col].iloc[-1])
                if atr_col in df_with_indicators.columns
                and not df_with_indicators.empty
                else None
            )
            latest_close = (
                safe_decimal_conversion(df_with_indicators["close"].iloc[-1])
                if "close" in df_with_indicators.columns
                and not df_with_indicators.empty
                else None
            )

            if latest_atr is None or latest_close is None:
                logger.warning(
                    f"{NEON['WARNING']}Missing latest ATR or Close price runes. Cannot make trade decisions.{NEON['RESET']}"
                )
                time.sleep(config.sleep_seconds)
                continue
            if latest_atr <= Decimal(0):  # ATR should be positive
                logger.warning(
                    f"{NEON['WARNING']}ATR rune ({latest_atr}) is not positive. SL calculation would be flawed. Skipping.{NEON['RESET']}"
                )
                time.sleep(config.sleep_seconds)
                continue

            # --- Position Management Decisions ---
            if (
                current_pos_side == config.pos_none
            ):  # If FLAT (no active position parts)
                if signals.get("enter_long"):
                    logger.info(
                        f"{NEON['ACTION']}Favorable omens for a LONG venture detected.{NEON['RESET']}"
                    )
                    sl_price = latest_close - (
                        latest_atr * config.atr_stop_loss_multiplier
                    )
                    place_risked_order(
                        exchange,
                        config,
                        config.pos_long,
                        latest_close,
                        sl_price,
                        latest_atr,
                    )
                elif signals.get("enter_short"):
                    logger.info(
                        f"{NEON['ACTION']}Shadowy omens for a SHORT venture detected.{NEON['RESET']}"
                    )
                    sl_price = latest_close + (
                        latest_atr * config.atr_stop_loss_multiplier
                    )
                    place_risked_order(
                        exchange,
                        config,
                        config.pos_short,
                        latest_close,
                        sl_price,
                        latest_atr,
                    )

            elif current_pos_side == config.pos_long:  # If in a LONG position
                if signals.get("exit_long"):
                    logger.info(
                        f"{NEON['ACTION']}Omen to EXIT LONG position: {signals.get('exit_reason', 'Strategy Exit Signal')}{NEON['RESET']}"
                    )
                    # Close all parts of the long position
                    for part in list(
                        _active_trade_parts
                    ):  # Iterate copy as list might change
                        if part["side"] == config.pos_long:
                            close_position_part(
                                exchange,
                                config,
                                part,
                                signals.get("exit_reason", "Strategy Exit Signal"),
                            )

            elif current_pos_side == config.pos_short:  # If in a SHORT position
                if signals.get("exit_short"):
                    logger.info(
                        f"{NEON['ACTION']}Omen to EXIT SHORT position: {signals.get('exit_reason', 'Strategy Exit Signal')}{NEON['RESET']}"
                    )
                    for part in list(_active_trade_parts):  # Iterate copy
                        if part["side"] == config.pos_short:
                            close_position_part(
                                exchange,
                                config,
                                part,
                                signals.get("exit_reason", "Strategy Exit Signal"),
                            )

            # Pyramiding/Scaling logic would be woven here if enabled and conditions met.
            # This is complex and involves checking profit levels, existing parts, max_scale_ins, etc. Omitted for current focus.

            save_persistent_state()  # Scribe memories at end of cycle (heartbeat logic within)
            logger.debug(
                f"{NEON['COMMENT']}# Cycle complete. Pyrmethus rests, observing the aether for {config.sleep_seconds}s...{NEON['RESET']}"
            )
            time.sleep(config.sleep_seconds)

        except KeyboardInterrupt:
            logger.warning(
                f"\n{NEON['WARNING']}Sorcerer's intervention (Ctrl+C)! Pyrmethus prepares for slumber...{NEON['RESET']}"
            )
            config.send_notification_method(
                "Pyrmethus Shutdown", "Manual shutdown initiated."
            )
            break
        except (
            Exception
        ) as e:  # Catch-all for unexpected errors in the main loop's weave
            logger.critical(
                f"{NEON['CRITICAL']}A critical error disrupted Pyrmethus's weave! Error: {e}{NEON['RESET']}"
            )
            logger.debug(traceback.format_exc())  # Reveal the shadowy details
            config.send_notification_method(
                "Pyrmethus Critical Error", f"Bot loop crashed: {str(e)[:150]}"
            )
            # Consider a longer sleep or break after repeated critical errors to avoid spamming or rate limits
            time.sleep(config.sleep_seconds * 5)  # Longer pause after major error


if __name__ == "__main__":
    logger.info(
        f"{NEON['COMMENT']}# Pyrmethus prepares to breach the veil to the exchange realm...{NEON['RESET']}"
    )
    exchange = None  # Initialize to ensure it's in scope for finally block
    try:
        exchange_params = {
            "apiKey": CONFIG.api_key,
            "secret": CONFIG.api_secret,
            "options": {
                "defaultType": "swap",  # Assuming futures/swap
                "adjustForTimeDifference": True,
                "brokerId": "PYRMETHUSV282",  # Custom Broker ID for Bybit tracking (if supported/useful)
            },
            "enableRateLimit": True,
            "recvWindow": CONFIG.default_recv_window,
        }

        # For Bybit V5, ensure using Unified Trading Account API keys.
        # Testnet interaction: Bybit V5 testnet uses different API endpoints.
        # CCXT might handle this with a 'testnet' option or by manually setting URLs.
        if CONFIG.PAPER_TRADING_MODE:
            logger.warning(
                f"{NEON['WARNING']}PAPER_TRADING_MODE is True. Ensure API keys are for Bybit TESTNET and endpoints are correct.{NEON['RESET']}"
            )
            # For Bybit V5 via CCXT, you might need to set the specific testnet API URL if 'testnet': True option isn't sufficient
            # e.g., exchange = ccxt.bybit({**exchange_params, 'urls': {'api': 'https://api-testnet.bybit.com'}})
            # Some versions of CCXT might use: exchange_params['options']['testnet'] = True
            # This depends on the CCXT library version and its Bybit V5 support.
            # For now, assume API keys dictate testnet/mainnet, or user configures URLs if needed.

        exchange = ccxt.bybit(
            exchange_params
        )  # Ensure 'bybit' is the correct CCXT ID for Bybit V5

        markets = exchange.load_markets()
        if CONFIG.symbol not in markets:
            err_msg = f"Symbol {CONFIG.symbol} not found in exchange's market runes. Pyrmethus cannot trade this ether."
            logger.critical(f"{NEON['CRITICAL']}{err_msg}{NEON['RESET']}")
            CONFIG.send_notification_method(
                "Pyrmethus Startup Fail", f"Symbol {CONFIG.symbol} not found."
            )
            sys.exit(1)

        CONFIG.MARKET_INFO = markets[
            CONFIG.symbol
        ]  # Store an imprint of the market's rules
        price_prec = CONFIG.MARKET_INFO.get("precision", {}).get("price", "N/A")
        amount_prec = CONFIG.MARKET_INFO.get("precision", {}).get("amount", "N/A")
        logger.success(
            f"{NEON['SUCCESS']}Market runes for {CONFIG.symbol} loaded. Precision: Price {price_prec}, Amount {amount_prec}{NEON['RESET']}"
        )

        # Leverage setting for Bybit V5 (example for USDT Perpetual - 'linear' category)
        # This is a powerful enchantment and should be wielded with understanding.
        # Often set once per symbol/margin mode on the exchange UI, but can be confirmed/set via API.
        try:
            # Category for Bybit V5: 'linear' (USDT perps), 'inverse' (coin-margined contracts)
            # This is a basic assumption based on symbol suffix. More robust detection might be needed.
            category = "linear" if CONFIG.symbol.endswith("USDT") else "inverse"

            # Bybit V5 setLeverage might need buyLeverage and sellLeverage if in hedge mode,
            # or just leverage for one-way mode. CCXT abstracts some of this.
            # The `symbol` parameter for set_leverage should be the exchange-specific ID, usually CONFIG.symbol works.
            params_leverage = {"category": category}
            if hasattr(exchange, "set_margin_mode") and CONFIG.MARKET_INFO.get(
                "margin", False
            ):  # If margin trading, check mode
                # exchange.set_margin_mode('isolated', CONFIG.symbol, {'category': category}) # or 'cross'
                pass  # Margin mode setting is complex, often done on UI.

            logger.info(
                f"Attempting to set leverage to {CONFIG.leverage}x for {CONFIG.symbol} (Category: {category})"
            )
            response = exchange.set_leverage(
                CONFIG.leverage, CONFIG.symbol, params_leverage
            )
            logger.success(
                f"{NEON['SUCCESS']}Leverage for {CONFIG.symbol} set/confirmed to {CONFIG.leverage}x. Response: {response}{NEON['RESET']}"
            )
        except Exception as e_lev:
            logger.warning(
                f"{NEON['WARNING']}Could not set leverage for {CONFIG.symbol} (may already be set, or an issue): {e_lev}{NEON['RESET']}"
            )
            CONFIG.send_notification_method(
                "Pyrmethus Leverage Warn",
                f"Leverage set issue for {CONFIG.symbol}: {str(e_lev)[:60]}",
            )

        logger.success(
            f"{NEON['SUCCESS']}Successfully connected to the exchange realm: {exchange.id}{NEON['RESET']}"
        )
        CONFIG.send_notification_method(
            "Pyrmethus Online", f"Connected to {exchange.id} for {CONFIG.symbol}"
        )

    except AttributeError as e:
        logger.critical(
            f"{NEON['CRITICAL']}Exchange attribute error: {e}. CCXT issue or wrong exchange name?{NEON['RESET']}"
        )
        sys.exit(1)
    except ccxt.AuthenticationError as e:
        logger.critical(
            f"{NEON['CRITICAL']}Authentication with exchange failed! Check API key runes: {e}{NEON['RESET']}"
        )
        CONFIG.send_notification_method("Pyrmethus Auth Fail", "API Key Auth Error.")
        sys.exit(1)
    except ccxt.NetworkError as e:
        logger.critical(
            f"{NEON['CRITICAL']}Network Error connecting to exchange: {e}. Check connectivity.{NEON['RESET']}"
        )
        CONFIG.send_notification_method(
            "Pyrmethus Network Error", f"Cannot connect: {str(e)[:80]}"
        )
        sys.exit(1)
    except ccxt.ExchangeError as e:
        logger.critical(
            f"{NEON['CRITICAL']}Exchange API Error: {e}. Check API permissions or symbol.{NEON['RESET']}"
        )
        CONFIG.send_notification_method(
            "Pyrmethus API Error", f"Exchange API issue: {str(e)[:80]}"
        )
        sys.exit(1)
    except Exception as e:
        logger.critical(
            f"{NEON['CRITICAL']}Failed to initialize exchange connection: {e}{NEON['RESET']}"
        )
        logger.debug(traceback.format_exc())
        CONFIG.send_notification_method(
            "Pyrmethus Init Error", f"Exchange init failed: {str(e)[:80]}"
        )
        sys.exit(1)

    if exchange:  # Only proceed if the connection to the exchange realm was successful
        main_loop(exchange, CONFIG)
    else:
        logger.critical(
            f"{NEON['CRITICAL']}Exchange not initialized. Pyrmethus cannot weave spells in an empty void. Spell aborted.{NEON['RESET']}"
        )
