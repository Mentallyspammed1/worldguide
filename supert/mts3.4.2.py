# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v2.8.0 (Strategic Illumination)
# Features reworked strategy logic (Dual ST + Momentum) and enhanced Neon Colorization.
# Previous Weave: v2.6.1 (Phoenix Feather Resilience & Compatibility Weave)

"""High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 2.8.0 (Strategic Illumination)

Enhancements:
- Reworked Strategy Logic: Implemented Dual Supertrend with Momentum confirmation (DUAL_SUPERTREND_MOMENTUM).
- Enhanced Neon Colorization: Comprehensive and thematic color usage for improved terminal output clarity.
- Added calculate_momentum function and integrated into indicator calculation pipeline.
- Updated versioning and state file naming.
- Refined Supertrend calculation for robust pandas_ta column name matching.
- Corrected ccxt.load_markets() parameter.
- Implemented get_market_data function for fetching OHLCV.
- Added EhlersFisherStrategy example and its indicator calculation.
- Improved TSL placement logic in place_risked_order.
- Automated SL cancellation in close_position.
- Added basic config validation.

Core Features from v2.6.1 (Persistence, Dynamic ATR SL, Pyramiding Foundation, etc.) remain.
"""

# Standard Library Imports
import json
import logging
import os
import random  # For MockExchange
import shutil
import subprocess
import sys
import time
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, DivisionByZero, InvalidOperation, getcontext
from enum import Enum
from typing import Any, Union, Dict, List, Tuple, Optional, Type

import pytz

# Third-party Libraries
try:
    import ccxt
    import pandas as pd

    if not hasattr(pd, "NA"):  # Ensure pandas version supports pd.NA
        raise ImportError("Pandas version < 1.0 not supported, pd.NA is required.")
    import pandas_ta as ta  # type: ignore[import]
    from colorama import Back, Fore, Style  # The Prisms of Perception
    from colorama import init as colorama_init
    from dotenv import load_dotenv
    from retry import retry
except ImportError as e:
    missing_pkg = getattr(e, "name", "dependency")
    sys.stderr.write(
        f"\033[91mCRITICAL ERROR: Missing/Incompatible Essence: '{missing_pkg}'.\033[0m\n"
    )
    sys.stderr.write(
        f"\033[91mPlease ensure all required libraries are installed and up to date.\033[0m\n"
    )
    sys.exit(1)

# --- Constants ---
STATE_FILE_NAME = "pyrmethus_phoenix_state_v280.json"  # Updated version
STATE_FILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), STATE_FILE_NAME
)
HEARTBEAT_INTERVAL_SECONDS = 60  # Save state at least this often if active

# --- Neon Color Palette (Enhanced as per request) ---
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
    "RESET": Style.RESET_ALL,
}

# --- Initializations ---
colorama_init(autoreset=True)
# Attempt to load .env file from the script's directory
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
getcontext().prec = 18  # Set precision for Decimal calculations


# --- Enums ---
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


# --- Configuration Class ---
class Config:
    """Loads and validates configuration parameters from environment variables for Pyrmethus v2.8.0."""

    def __init__(self) -> None:
        _pre_logger = logging.getLogger(__name__)  # Standard logger can be used here
        _pre_logger.info(
            f"{NEON['HEADING']}--- Summoning Configuration Runes v2.8.0 ---{NEON['RESET']}"
        )

        # --- API Credentials ---
        self.api_key: str = self._get_env("BYBIT_API_KEY", required=True, secret=True)
        self.api_secret: str = self._get_env(
            "BYBIT_API_SECRET", required=True, secret=True
        )

        # --- Trading Parameters ---
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT")
        self.interval: str = self._get_env("INTERVAL", "1m")
        self.leverage: int = self._get_env("LEVERAGE", 25, cast_type=int)
        self.sleep_seconds: int = self._get_env("SLEEP_SECONDS", 10, cast_type=int)

        # --- Strategy Selection ---
        self.strategy_name: StrategyName = StrategyName(
            self._get_env(
                "STRATEGY_NAME", StrategyName.DUAL_SUPERTREND_MOMENTUM.value
            ).upper()
        )
        self.strategy_instance: "TradingStrategy"  # Forward declaration

        # --- Risk Management ---
        self.risk_per_trade_percentage: Decimal = self._get_env(
            "RISK_PER_TRADE_PERCENTAGE", "0.01", cast_type=Decimal
        )
        self.enable_dynamic_risk: bool = self._get_env(
            "ENABLE_DYNAMIC_RISK", "false", cast_type=bool
        )
        self.dynamic_risk_min_pct: Decimal = self._get_env(
            "DYNAMIC_RISK_MIN_PCT", "0.005", cast_type=Decimal
        )
        self.dynamic_risk_max_pct: Decimal = self._get_env(
            "DYNAMIC_RISK_MAX_PCT", "0.015", cast_type=Decimal
        )
        self.dynamic_risk_perf_window: int = self._get_env(
            "DYNAMIC_RISK_PERF_WINDOW", 10, cast_type=int
        )  # Trades
        self.max_order_usdt_amount: Decimal = self._get_env(
            "MAX_ORDER_USDT_AMOUNT", "50.0", cast_type=Decimal
        )  # Cap per single order
        self.required_margin_buffer: Decimal = self._get_env(
            "REQUIRED_MARGIN_BUFFER", "1.05", cast_type=Decimal
        )  # e.g., 1.05 for 5% buffer
        self.max_account_margin_ratio: Decimal = self._get_env(
            "MAX_ACCOUNT_MARGIN_RATIO", "0.8", cast_type=Decimal
        )  # e.g., 0.8 for 80%
        self.enable_max_drawdown_stop: bool = self._get_env(
            "ENABLE_MAX_DRAWDOWN_STOP", "true", cast_type=bool
        )
        self.max_drawdown_percent: Decimal = self._get_env(
            "MAX_DRAWDOWN_PERCENT", "0.10", cast_type=Decimal
        )  # 10% daily drawdown
        self.enable_time_based_stop: bool = self._get_env(
            "ENABLE_TIME_BASED_STOP", "false", cast_type=bool
        )
        self.max_trade_duration_seconds: int = self._get_env(
            "MAX_TRADE_DURATION_SECONDS", 3600, cast_type=int
        )  # 1 hour

        # --- Dynamic ATR Stop Loss ---
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
            "ATR_STOP_LOSS_MULTIPLIER", "1.3", cast_type=Decimal
        )  # Fallback if dynamic is off

        # --- Position Scaling (Pyramiding) ---
        self.enable_position_scaling: bool = self._get_env(
            "ENABLE_POSITION_SCALING", "true", cast_type=bool
        )
        self.max_scale_ins: int = self._get_env(
            "MAX_SCALE_INS", 1, cast_type=int
        )  # Max additional entries
        self.scale_in_risk_percentage: Decimal = self._get_env(
            "SCALE_IN_RISK_PERCENTAGE", "0.005", cast_type=Decimal
        )
        self.min_profit_for_scale_in_atr: Decimal = self._get_env(
            "MIN_PROFIT_FOR_SCALE_IN_ATR", "1.0", cast_type=Decimal
        )  # e.g. 1 ATR in profit
        self.enable_scale_out: bool = self._get_env(
            "ENABLE_SCALE_OUT", "false", cast_type=bool
        )  # Partial profit taking
        self.scale_out_trigger_atr: Decimal = self._get_env(
            "SCALE_OUT_TRIGGER_ATR", "2.0", cast_type=Decimal
        )  # e.g. 2 ATRs in profit

        # --- Trailing Stop Loss (Exchange Native - Bybit V5) ---
        self.trailing_stop_percentage: Decimal = self._get_env(
            "TRAILING_STOP_PERCENTAGE", "0.005", cast_type=Decimal
        )  # 0.5%
        self.trailing_stop_activation_offset_percent: Decimal = self._get_env(
            "TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT", "0.001", cast_type=Decimal
        )  # 0.1%

        # --- Execution ---
        self.entry_order_type: OrderEntryType = OrderEntryType(
            self._get_env("ENTRY_ORDER_TYPE", OrderEntryType.MARKET.value).upper()
        )
        self.limit_order_offset_pips: int = self._get_env(
            "LIMIT_ORDER_OFFSET_PIPS", 2, cast_type=int
        )  # For limit entries
        self.limit_order_fill_timeout_seconds: int = self._get_env(
            "LIMIT_ORDER_FILL_TIMEOUT_SECONDS", 20, cast_type=int
        )
        self.order_fill_timeout_seconds: int = self._get_env(
            "ORDER_FILL_TIMEOUT_SECONDS", 15, cast_type=int
        )  # For market orders

        # --- Strategy-Specific Parameters ---
        # Dual Supertrend Momentum
        self.st_atr_length: int = self._get_env(
            "ST_ATR_LENGTH", 10, cast_type=int
        )  # Primary ST
        self.st_multiplier: Decimal = self._get_env(
            "ST_MULTIPLIER", "2.0", cast_type=Decimal
        )  # Primary ST
        self.confirm_st_atr_length: int = self._get_env(
            "CONFIRM_ST_ATR_LENGTH", 20, cast_type=int
        )  # Confirmation ST
        self.confirm_st_multiplier: Decimal = self._get_env(
            "CONFIRM_ST_MULTIPLIER", "3.0", cast_type=Decimal
        )  # Confirmation ST
        self.momentum_period: int = self._get_env(
            "MOMENTUM_PERIOD", 14, cast_type=int
        )  # For DUAL_SUPERTREND_MOMENTUM
        self.momentum_threshold: Decimal = self._get_env(
            "MOMENTUM_THRESHOLD", "0", cast_type=Decimal
        )  # For DUAL_SUPERTREND_MOMENTUM (e.g., Mom > 0 for long)
        # Ehlers Fisher
        self.ehlers_fisher_length: int = self._get_env(
            "EHLERS_FISHER_LENGTH", 10, cast_type=int
        )
        self.ehlers_fisher_signal_length: int = self._get_env(
            "EHLERS_FISHER_SIGNAL_LENGTH", 1, cast_type=int
        )

        # --- Misc / Internal ---
        self.volume_ma_period: int = self._get_env(
            "VOLUME_MA_PERIOD", 20, cast_type=int
        )
        # Corrected: Use atr_short_term_period if dynamic ATR SL is enabled, otherwise use a general ATR_CALCULATION_PERIOD
        self.atr_calculation_period: int = (
            self.atr_short_term_period
            if self.enable_dynamic_atr_sl
            else self._get_env("ATR_CALCULATION_PERIOD", 14, cast_type=int)
        )
        self.enable_sms_alerts: bool = self._get_env(
            "ENABLE_SMS_ALERTS", "true", cast_type=bool
        )
        self.sms_recipient_number: Optional[str] = self._get_env(
            "SMS_RECIPIENT_NUMBER", None, cast_type=str
        )
        self.sms_timeout_seconds: int = self._get_env(
            "SMS_TIMEOUT_SECONDS", 30, cast_type=int
        )
        self.default_recv_window: int = self._get_env(
            "DEFAULT_RECV_WINDOW", 13000, cast_type=int
        )  # milliseconds
        self.order_book_depth: int = self._get_env(
            "ORDER_BOOK_DEPTH", 20, cast_type=int
        )
        self.order_book_fetch_limit: int = max(
            25, self.order_book_depth
        )  # Ensure fetch limit is adequate
        self.shallow_ob_fetch_depth: int = 5  # For quick price checks

        # --- Internal Constants ---
        self.side_buy: str = "buy"
        self.side_sell: str = "sell"
        self.pos_long: str = "Long"
        self.pos_short: str = "Short"
        self.pos_none: str = "None"
        self.usdt_symbol: str = "USDT"
        self.retry_count: int = 3
        self.retry_delay_seconds: int = 2
        self.api_fetch_limit_buffer: int = 20
        self.position_qty_epsilon: Decimal = Decimal("1e-9")
        self.post_close_delay_seconds: int = 3
        self.cache_candle_duration_multiplier: Decimal = Decimal("0.95")

        self._validate_parameters()
        _pre_logger.info(
            f"{NEON['HEADING']}--- Configuration Runes v2.8.0 Summoned and Verified ---{NEON['RESET']}"
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
        _logger = logging.getLogger(__name__)  # Use standard logger
        value_str = os.getenv(key)
        source = "Env Var"
        value_to_cast: Any = None
        display_value = "********" if secret and value_str is not None else value_str

        if value_str is None:
            if required:
                _logger.critical(
                    f"{NEON['CRITICAL']}CRITICAL: Required config rune '{key}' not found in environment or .env scroll.{NEON['RESET']}"
                )
                raise ValueError(f"Required environment variable '{key}' not set.")
            _logger.debug(
                f"{color}Config Rune {NEON['VALUE']}'{key}'{color}: Not Set. Using Default: '{NEON['VALUE']}{default}{color}'{NEON['RESET']}"
            )
            value_to_cast = default
            source = "Default"
        else:
            _logger.debug(
                f"{color}Config Rune {NEON['VALUE']}'{key}'{color}: Found Env Value: '{NEON['VALUE']}{display_value}{color}'{NEON['RESET']}"
            )
            value_to_cast = value_str

        if value_to_cast is None:  # Handles if default was None and env var not set
            if required:  # Should have been caught above, but as a safeguard
                _logger.critical(
                    f"{NEON['CRITICAL']}CRITICAL: Required config rune '{key}' resolved to None.{NEON['RESET']}"
                )
                raise ValueError(
                    f"Required environment variable '{key}' resolved to None."
                )
            return None  # Not required, value is None

        final_value: Any
        try:
            raw_value_str_for_cast = str(
                value_to_cast
            )  # Ensure it's a string before type-specific casting
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
                )  # Cast to Decimal first for "1.0" -> 1
            elif cast_type == float:
                final_value = float(raw_value_str_for_cast)
            elif cast_type == str:
                final_value = raw_value_str_for_cast
            else:  # Should not happen if cast_type is one of the above
                _logger.warning(
                    f"{NEON['WARNING']}Unsupported cast_type '{cast_type.__name__}' for key '{key}'. Returning raw.{NEON['RESET']}"
                )
                final_value = value_to_cast
        except (ValueError, TypeError, InvalidOperation) as e:
            _logger.error(
                f"{NEON['ERROR']}Invalid type/value for '{key}': '{value_to_cast}' (Source: {source}). Expected {cast_type.__name__}. Error: {e}. Trying default '{default}'.{NEON['RESET']}"
            )
            if default is None:  # If default is also None, and casting failed
                if required:
                    _logger.critical(
                        f"{NEON['CRITICAL']}CRITICAL: Failed cast for required key '{key}', default is None.{NEON['RESET']}"
                    )
                    raise ValueError(
                        f"Required env var '{key}' failed casting, no valid default."
                    )
                else:  # Not required, default is None, cast failed
                    _logger.warning(
                        f"{NEON['WARNING']}Casting failed for optional '{key}', default is None. Final value: None{NEON['RESET']}"
                    )
                    return None  # Return None if default is None and casting failed
            else:  # Default is not None, try casting the default
                source = "Default (Fallback)"
                _logger.debug(
                    f"Casting fallback default '{default}' for '{key}' to {cast_type.__name__}"
                )
                try:
                    default_str_for_cast = str(
                        default
                    )  # Ensure default is string before type-specific casting
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
                        final_value = (
                            default  # Should not happen with supported cast_types
                        )
                    _logger.warning(
                        f"{NEON['WARNING']}Used casted default for {key}: '{NEON['VALUE']}{final_value}{NEON['WARNING']}'{NEON['RESET']}"
                    )
                except (ValueError, TypeError, InvalidOperation) as e_default:
                    _logger.critical(
                        f"{NEON['CRITICAL']}CRITICAL: Failed cast for BOTH value ('{value_to_cast}') AND default ('{default}') for key '{key}' to {cast_type.__name__}. Error on default: {e_default}{NEON['RESET']}"
                    )
                    raise ValueError(
                        f"Config error: Cannot cast value or default for '{key}' to {cast_type.__name__}."
                    )

        display_final_value = "********" if secret else final_value
        _logger.debug(
            f"{color}Using final value for {NEON['VALUE']}'{key}'{color}: {NEON['VALUE']}{display_final_value}{color} (Type: {type(final_value).__name__}, Source: {source}){NEON['RESET']}"
        )
        return final_value

    def _validate_parameters(self) -> None:
        """Performs basic validation of critical configuration parameters."""
        _logger = logging.getLogger(__name__)
        errors = []
        if not (0 < self.risk_per_trade_percentage < 1):
            errors.append(
                f"RISK_PER_TRADE_PERCENTAGE ({self.risk_per_trade_percentage}) must be between 0 and 1 (exclusive)."
            )
        if self.leverage < 1:
            errors.append(f"LEVERAGE ({self.leverage}) must be at least 1.")
        if self.max_scale_ins < 0:
            errors.append(f"MAX_SCALE_INS ({self.max_scale_ins}) cannot be negative.")
        if self.trailing_stop_percentage < 0:
            errors.append(
                f"TRAILING_STOP_PERCENTAGE ({self.trailing_stop_percentage}) cannot be negative."
            )
        if self.trailing_stop_activation_offset_percent < 0:
            errors.append(
                f"TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT ({self.trailing_stop_activation_offset_percent}) cannot be negative."
            )

        if errors:
            error_message = (
                f"Configuration validation failed with {len(errors)} error(s):\n"
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
log_file_name = f"{logs_dir}/pyrmethus_spell_v280_{time.strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)-8s] %(name)-30s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file_name)],
)
logger: logging.Logger = logging.getLogger("PyrmethusCore")

SUCCESS_LEVEL: int = 25  # Between INFO and WARNING
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)  # type: ignore[attr-defined]


logging.Logger.success = log_success  # type: ignore[attr-defined]

if sys.stdout.isatty():  # Apply NEON colors only if output is a TTY
    logging.addLevelName(
        logging.DEBUG,
        f"{NEON['DEBUG']}{logging.getLevelName(logging.DEBUG)}{NEON['RESET']}",
    )
    logging.addLevelName(
        logging.INFO,
        f"{NEON['INFO']}{logging.getLevelName(logging.INFO)}{NEON['RESET']}",
    )
    logging.addLevelName(
        SUCCESS_LEVEL,
        f"{NEON['SUCCESS']}{logging.getLevelName(SUCCESS_LEVEL)}{NEON['RESET']}",
    )  # SUCCESS uses its own bright green
    logging.addLevelName(
        logging.WARNING,
        f"{NEON['WARNING']}{logging.getLevelName(logging.WARNING)}{NEON['RESET']}",
    )
    logging.addLevelName(
        logging.ERROR,
        f"{NEON['ERROR']}{logging.getLevelName(logging.ERROR)}{NEON['RESET']}",
    )
    logging.addLevelName(
        logging.CRITICAL,
        f"{NEON['CRITICAL']}{logging.getLevelName(logging.CRITICAL)}{NEON['RESET']}",
    )

# --- Global Objects - Instantiated Arcana ---
try:
    CONFIG = Config()
except ValueError as config_error:  # Catch validation errors from Config
    logging.getLogger().critical(
        f"{NEON['CRITICAL']}Configuration loading failed. Error: {config_error}{NEON['RESET']}"
    )
    sys.exit(1)
except (
    Exception
) as general_config_error:  # Catch any other unexpected error during config
    logging.getLogger().critical(
        f"{NEON['CRITICAL']}Unexpected critical error during configuration: {general_config_error}{NEON['RESET']}"
    )
    logging.getLogger().debug(
        traceback.format_exc()
    )  # Log stack trace for detailed debugging
    sys.exit(1)


# --- Trading Strategy Abstract Base Class & Implementations ---
class TradingStrategy(ABC):
    def __init__(self, config: Config, df_columns: Optional[List[str]] = None):
        self.config = config
        self.logger = logging.getLogger(
            f"{NEON['STRATEGY']}Strategy.{self.__class__.__name__}{NEON['RESET']}"
        )
        self.required_columns = df_columns if df_columns else []

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generates trading signals based on the input DataFrame with indicator data."""
        pass

    def _validate_df(self, df: pd.DataFrame, min_rows: int = 2) -> bool:
        """Validates the input DataFrame for common issues before signal generation."""
        if df is None or df.empty or len(df) < min_rows:
            self.logger.debug(
                f"Insufficient data (Rows: {len(df) if df is not None else 0}, Min Required: {min_rows}). Awaiting more market whispers."
            )
            return False
        if self.required_columns and not all(
            col in df.columns for col in self.required_columns
        ):
            missing_cols = [
                col for col in self.required_columns if col not in df.columns
            ]
            self.logger.warning(
                f"{NEON['WARNING']}DataFrame is missing required columns for this strategy: {NEON['VALUE']}{missing_cols}{NEON['WARNING']}. Cannot divine signals.{NEON['RESET']}"
            )
            return False
        # Check for NaNs in the last row for required columns, which might indicate calculation issues or insufficient leading data.
        if self.required_columns:
            last_row_values = df.iloc[-1][self.required_columns]
            if last_row_values.isnull().any():
                nan_cols_last_row = last_row_values[
                    last_row_values.isnull()
                ].index.tolist()
                self.logger.debug(
                    f"NaN values in last candle for critical columns: {nan_cols_last_row}. Signals may be unreliable."
                )
                # Strategies should handle these NaNs, typically by not generating a signal.
        return True

    def _get_default_signals(self) -> Dict[str, Any]:
        """Returns a default signal dictionary (all signals false)."""
        return {
            "enter_long": False,
            "enter_short": False,
            "exit_long": False,
            "exit_short": False,
            "exit_reason": "Default Signal - Awaiting Omens",
        }


class DualSupertrendMomentumStrategy(TradingStrategy):  # REWORKED STRATEGY
    def __init__(self, config: Config):
        super().__init__(
            config,
            df_columns=["st_long_flip", "st_short_flip", "confirm_trend", "momentum"],
        )
        self.logger.info(
            f"{NEON['STRATEGY']}Dual Supertrend with Momentum Confirmation strategy initialized.{NEON['RESET']}"
        )

    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        signals = self._get_default_signals()
        # Ensure enough data for all indicators; +10 as a general buffer beyond max period
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
        confirm_is_up = last.get(
            "confirm_trend", pd.NA
        )  # pd.NA if trend is indeterminate

        momentum_val_orig = last.get("momentum", pd.NA)
        momentum_val = safe_decimal_conversion(
            momentum_val_orig, pd.NA
        )  # Convert to Decimal or pd.NA

        if pd.isna(confirm_is_up) or pd.isna(momentum_val):
            self.logger.debug(
                f"Confirmation ST Trend ({_format_for_log(confirm_is_up, is_bool_trend=True)}) or Momentum ({_format_for_log(momentum_val)}) is NA. No signal."
            )
            return signals

        # Entry Signals: Supertrend flip + Confirmation ST direction + Momentum confirmation
        if (
            primary_long_flip
            and confirm_is_up is True
            and momentum_val > self.config.momentum_threshold
        ):
            signals["enter_long"] = True
            self.logger.info(
                f"{NEON['SIDE_LONG']}DualST+Mom Signal: LONG Entry - Primary ST Long Flip, Confirm ST Up, Momentum ({_format_for_log(momentum_val)}) > {_format_for_log(self.config.momentum_threshold)}{NEON['RESET']}"
            )
        elif (
            primary_short_flip
            and confirm_is_up is False
            and momentum_val < -self.config.momentum_threshold
        ):  # Assuming symmetrical threshold for short
            signals["enter_short"] = True
            self.logger.info(
                f"{NEON['SIDE_SHORT']}DualST+Mom Signal: SHORT Entry - Primary ST Short Flip, Confirm ST Down, Momentum ({_format_for_log(momentum_val)}) < -{_format_for_log(self.config.momentum_threshold)}{NEON['RESET']}"
            )

        # Exit Signals: Based on primary SuperTrend flips (can be enhanced)
        if primary_short_flip:  # If primary ST flips short, exit any long position.
            signals["exit_long"] = True
            signals["exit_reason"] = "Primary SuperTrend Flipped Short"
            self.logger.info(
                f"{NEON['ACTION']}DualST+Mom Signal: EXIT LONG - Primary ST Flipped Short{NEON['RESET']}"
            )
        if primary_long_flip:  # If primary ST flips long, exit any short position.
            signals["exit_short"] = True
            signals["exit_reason"] = "Primary SuperTrend Flipped Long"
            self.logger.info(
                f"{NEON['ACTION']}DualST+Mom Signal: EXIT SHORT - Primary ST Flipped Long{NEON['RESET']}"
            )
        return signals


class EhlersFisherStrategy(TradingStrategy):
    def __init__(self, config: Config):
        super().__init__(config, df_columns=["ehlers_fisher", "ehlers_signal"])
        self.logger.info(
            f"{NEON['STRATEGY']}Ehlers Fisher Transform strategy initialized.{NEON['RESET']}"
        )

    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        signals = self._get_default_signals()
        min_rows_needed = (
            self.config.ehlers_fisher_length
            + self.config.ehlers_fisher_signal_length
            + 5
        )  # Approx
        if not self._validate_df(df, min_rows=min_rows_needed):
            return signals

        last = df.iloc[-1]
        prev = df.iloc[-2]  # Need previous candle for crossover

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
                f"Ehlers Fisher or Signal is NA. Fisher: {_format_for_log(fisher_now)}, Signal: {_format_for_log(signal_now)}. No signal."
            )
            return signals

        # Entry Signals: Fisher line crosses Signal line
        if fisher_prev <= signal_prev and fisher_now > signal_now:
            signals["enter_long"] = True
            self.logger.info(
                f"{NEON['SIDE_LONG']}EhlersFisher Signal: LONG Entry - Fisher crossed ABOVE Signal.{NEON['RESET']}"
            )
        elif fisher_prev >= signal_prev and fisher_now < signal_now:
            signals["enter_short"] = True
            self.logger.info(
                f"{NEON['SIDE_SHORT']}EhlersFisher Signal: SHORT Entry - Fisher crossed BELOW Signal.{NEON['RESET']}"
            )

        # Exit Signals: Opposite crossover
        if (
            fisher_prev >= signal_prev and fisher_now < signal_now
        ):  # Fisher crosses below Signal
            signals["exit_long"] = True
            signals["exit_reason"] = "Ehlers Fisher crossed BELOW Signal"
            self.logger.info(
                f"{NEON['ACTION']}EhlersFisher Signal: EXIT LONG - Fisher crossed BELOW Signal.{NEON['RESET']}"
            )
        elif (
            fisher_prev <= signal_prev and fisher_now > signal_now
        ):  # Fisher crosses above Signal
            signals["exit_short"] = True
            signals["exit_reason"] = "Ehlers Fisher crossed ABOVE Signal"
            self.logger.info(
                f"{NEON['ACTION']}EhlersFisher Signal: EXIT SHORT - Fisher crossed ABOVE Signal.{NEON['RESET']}"
            )

        return signals


strategy_map: Dict[StrategyName, Type[TradingStrategy]] = {
    StrategyName.DUAL_SUPERTREND_MOMENTUM: DualSupertrendMomentumStrategy,
    StrategyName.EHLERS_FISHER: EhlersFisherStrategy,  # Corrected
}
StrategyClass = strategy_map.get(CONFIG.strategy_name)
if StrategyClass:
    CONFIG.strategy_instance = StrategyClass(CONFIG)
else:
    logger.critical(
        f"{NEON['CRITICAL']}Failed to find and initialize strategy class for '{CONFIG.strategy_name.value}'. Pyrmethus cannot weave.{NEON['RESET']}"
    )
    sys.exit(1)


# --- Trade Metrics Tracking ---
class TradeMetrics:
    def __init__(self):
        self.trades: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("TradeMetrics")
        self.initial_equity: Optional[Decimal] = None
        self.daily_start_equity: Optional[Decimal] = None
        self.last_daily_reset_day: Optional[int] = None

    def set_initial_equity(self, equity: Decimal):
        if self.initial_equity is None:
            self.initial_equity = equity
            self.logger.info(
                f"{NEON['INFO']}Initial Equity for session set: {NEON['VALUE']}{equity:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}"
            )

        today = datetime.now(pytz.utc).day
        if self.last_daily_reset_day != today:
            self.daily_start_equity = equity
            self.last_daily_reset_day = today
            self.logger.info(
                f"{NEON['INFO']}Daily equity reset for drawdown tracking. Start Equity: {NEON['VALUE']}{equity:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}"
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
        scale_order_id: Optional[str] = None,
        part_id: Optional[str] = None,
        mae: Optional[Decimal] = None,
        mfe: Optional[Decimal] = None,
    ):
        if not all(
            [
                entry_price > 0,
                exit_price > 0,
                qty > 0,
                entry_time_ms > 0,
                exit_time_ms > 0,
            ]
        ):
            self.logger.warning(
                f"{NEON['WARNING']}Trade log skipped due to invalid parameters.{NEON['RESET']}"
            )
            return

        profit_per_unit = (
            (exit_price - entry_price)
            if (
                side.lower() == CONFIG.side_buy.lower()
                or side.lower() == CONFIG.pos_long.lower()
            )
            else (entry_price - exit_price)
        )
        profit = profit_per_unit * qty

        entry_dt_iso = datetime.fromtimestamp(
            entry_time_ms / 1000, tz=pytz.utc
        ).isoformat()
        exit_dt_iso = datetime.fromtimestamp(
            exit_time_ms / 1000, tz=pytz.utc
        ).isoformat()
        duration_seconds = (
            datetime.fromisoformat(exit_dt_iso) - datetime.fromisoformat(entry_dt_iso)
        ).total_seconds()

        trade_type = (
            "Scale-In"
            if scale_order_id
            else ("Initial" if part_id == "initial" else "Part")
        )

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
                "part_id": part_id or "unknown",
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
            f"{NEON['HEADING']}Trade Chronicle ({trade_type} Part:{part_id or 'N/A'}): "
            f"{side.upper()} {NEON['QTY']}{qty}{NEON['RESET']} {symbol.split('/')[0]} | "
            f"P/L: {pnl_color}{profit:.2f} {CONFIG.usdt_symbol}{NEON['RESET']} | Reason: {reason}",
        )

    def calculate_mae_mfe(
        self,
        part_id: str,
        entry_price: Decimal,
        exit_price: Decimal,
        side: str,
        entry_time_ms: int,
        exit_time_ms: int,
        exchange: ccxt.Exchange,
        symbol: str,
        interval: str,
    ) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """
        Calculates Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE)
        for a given trade part. This requires fetching OHLCV data for the trade's duration.
        NOTE: This is a placeholder. A full implementation requires fetching/querying OHLCV for the trade.
        """
        self.logger.debug(
            f"Attempting MAE/MFE calculation for part {part_id} (Placeholder - actual OHLCV fetch needed)."
        )

        if exit_time_ms <= entry_time_ms:
            self.logger.warning(
                f"MAE/MFE calc: Exit time ({exit_time_ms}) is not after entry time ({entry_time_ms}) for part {part_id}."
            )
            return None, None

        # --- This is a simplified placeholder and will not provide accurate MAE/MFE ---
        # --- A proper implementation would fetch OHLCV for the exact time range. ---
        self.logger.warning(
            f"MAE/MFE calculation for part {part_id} is a placeholder. "
            f"Accurate calculation requires fetching/querying OHLCV data for the trade's duration "
            f"({datetime.fromtimestamp(entry_time_ms / 1000, tz=pytz.utc)} to {datetime.fromtimestamp(exit_time_ms / 1000, tz=pytz.utc)})."
        )
        return None, None

    def get_performance_trend(self, window: int) -> float:
        if window <= 0 or not self.trades:
            return 0.5  # Neutral trend if no data or invalid window
        recent_trades = self.trades[-window:]
        if not recent_trades:
            return 0.5
        wins = sum(1 for t in recent_trades if Decimal(t["profit_str"]) > 0)
        return float(wins / len(recent_trades))

    def summary(self) -> str:
        if not self.trades:
            return "The Grand Ledger is empty. No trades chronicled yet."
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
            f"\n{NEON['HEADING']}--- Pyrmethus Trade Metrics Summary (v2.8.0) ---{NEON['RESET']}\n"
            f"Total Trade Parts Chronicled: {NEON['VALUE']}{total_trades}{NEON['RESET']}\n"
            f"  Victories (Wins): {NEON['PNL_POS']}{wins}{NEON['RESET']}\n"
            f"  Defeats (Losses): {NEON['PNL_NEG']}{losses}{NEON['RESET']}\n"
            f"  Stalemates (Breakeven): {NEON['PNL_ZERO']}{breakeven}{NEON['RESET']}\n"
            f"Victory Rate (by parts): {NEON['VALUE']}{win_rate:.2f}%{NEON['RESET']}\n"
            f"Total Spoils (P/L): {(NEON['PNL_POS'] if total_profit > 0 else (NEON['PNL_NEG'] if total_profit < 0 else NEON['PNL_ZERO']))}{total_profit:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}\n"
            f"Avg Spoils per Part: {(NEON['PNL_POS'] if avg_profit > 0 else (NEON['PNL_NEG'] if avg_profit < 0 else NEON['PNL_ZERO']))}{avg_profit:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}\n"
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

        summary_str += f"{NEON['HEADING']}--- End of Grand Ledger ---{NEON['RESET']}"
        self.logger.info(summary_str)
        return summary_str

    def get_serializable_trades(self) -> List[Dict[str, Any]]:
        return self.trades

    def load_trades_from_list(self, trades_list: List[Dict[str, Any]]) -> None:
        self.trades = trades_list
        self.logger.info(
            f"{NEON['INFO']}TradeMetrics: Re-inked {NEON['VALUE']}{len(self.trades)}{NEON['INFO']} trades from Phoenix scroll.{NEON['RESET']}"
        )


trade_metrics = TradeMetrics()
_active_trade_parts: List[Dict[str, Any]] = []  # Stores dicts of active trade parts
_last_heartbeat_save_time: float = 0.0


# --- State Persistence Functions (Phoenix Feather) ---
def save_persistent_state(force_heartbeat: bool = False) -> None:
    global _active_trade_parts, trade_metrics, _last_heartbeat_save_time
    now = time.time()
    if force_heartbeat or (
        now - _last_heartbeat_save_time > HEARTBEAT_INTERVAL_SECONDS
    ):
        try:
            serializable_active_parts = []
            for part in _active_trade_parts:
                serializable_part = part.copy()
                for key, value in serializable_part.items():
                    if isinstance(value, Decimal):
                        serializable_part[key] = str(value)
                    # entry_time_ms should already be an int, no need to serialize if so
                    elif isinstance(
                        value, (datetime, pd.Timestamp)
                    ):  # Should ideally be int (ms)
                        serializable_part[key] = value.isoformat()
                serializable_active_parts.append(serializable_part)

            state_data = {
                "timestamp_utc_iso": datetime.now(pytz.utc).isoformat(),
                "last_heartbeat_utc_iso": datetime.now(
                    pytz.utc
                ).isoformat(),  # For tracking save time
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
            os.replace(temp_file_path, STATE_FILE_PATH)  # Atomic replace
            _last_heartbeat_save_time = now
            logger.log(
                logging.DEBUG if not force_heartbeat else logging.INFO,
                f"{NEON['SUCCESS']}Phoenix Feather: State memories scribed to scroll.{NEON['RESET']}",
            )
        except Exception as e:
            logger.error(
                f"{NEON['ERROR']}Phoenix Feather Error scribing state: {e}{NEON['RESET']}"
            )
            logger.debug(traceback.format_exc())


def load_persistent_state() -> bool:
    global _active_trade_parts, trade_metrics
    if not os.path.exists(STATE_FILE_PATH):
        logger.info(
            f"{NEON['INFO']}Phoenix Feather: No previous scroll found ({NEON['VALUE']}{STATE_FILE_PATH}{NEON['INFO']}). Starting with a fresh essence.{NEON['RESET']}"
        )
        return False
    try:
        with open(STATE_FILE_PATH, "r") as f:
            state_data = json.load(f)

        # Validate if the saved state matches current critical config (symbol, strategy)
        if (
            state_data.get("config_symbol") != CONFIG.symbol
            or state_data.get("config_strategy") != CONFIG.strategy_name.value
        ):
            logger.warning(
                f"{NEON['WARNING']}Phoenix Feather: Scroll sigils (symbol/strategy) mismatch current configuration. "
                f"Saved: {state_data.get('config_symbol')}/{state_data.get('config_strategy')}, "
                f"Current: {CONFIG.symbol}/{CONFIG.strategy_name.value}. Ignoring old scroll.{NEON['RESET']}"
            )
            os.remove(STATE_FILE_PATH)  # Remove mismatched state file
            return False

        loaded_active_parts = state_data.get("active_trade_parts", [])
        _active_trade_parts.clear()
        for part_data in loaded_active_parts:
            restored_part = part_data.copy()
            # Restore Decimals and potentially entry_time_ms if saved as ISO
            for key, value_str in restored_part.items():
                if key in ["entry_price", "qty", "sl_price"] and isinstance(
                    value_str, str
                ):
                    try:
                        restored_part[key] = Decimal(value_str)
                    except InvalidOperation:
                        logger.warning(
                            f"Could not convert '{value_str}' to Decimal for key '{key}' in loaded state."
                        )
                if key == "entry_time_ms":  # entry_time_ms should be int
                    if isinstance(value_str, str):  # If saved as ISO string
                        try:
                            restored_part[key] = int(
                                datetime.fromisoformat(value_str).timestamp() * 1000
                            )
                        except ValueError:
                            pass  # If it's already int as string, int() below will handle
                    if isinstance(
                        value_str, (str, float)
                    ):  # If it's a string int "123" or float 123.0
                        try:
                            restored_part[key] = int(value_str)
                        except ValueError:
                            logger.warning(
                                f"Could not convert '{value_str}' to int for entry_time_ms."
                            )
            _active_trade_parts.append(restored_part)

        trade_metrics.load_trades_from_list(state_data.get("trade_metrics_trades", []))

        daily_start_equity_str = state_data.get("daily_start_equity_str")
        if daily_start_equity_str:
            try:
                trade_metrics.daily_start_equity = Decimal(daily_start_equity_str)
            except InvalidOperation:
                logger.warning(
                    f"Could not load daily_start_equity_str: {daily_start_equity_str}"
                )
        trade_metrics.last_daily_reset_day = state_data.get("last_daily_reset_day")

        saved_time_str = state_data.get("timestamp_utc_iso", "ancient times")
        logger.success(
            f"{NEON['SUCCESS']}Phoenix Feather: Memories from {NEON['VALUE']}{saved_time_str}{NEON['SUCCESS']} reawakened! Active parts: {len(_active_trade_parts)}, Trades: {len(trade_metrics.trades)}.{NEON['RESET']}"
        )
        return True
    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}Phoenix Feather Error reawakening state: {e}. Starting with a fresh essence.{NEON['RESET']}"
        )
        logger.debug(traceback.format_exc())
        try:
            os.remove(STATE_FILE_PATH)  # Attempt to remove corrupted state file
        except OSError:
            pass
        _active_trade_parts.clear()  # Reset state
        trade_metrics.trades.clear()
        return False


# --- Helper Functions, Retry Decorator, SMS Alert, Exchange Initialization ---
PandasNAType = type(pd.NA)  # For type hinting if needed


def safe_decimal_conversion(
    value: Any, default: Union[Decimal, PandasNAType, None] = Decimal("0.0")
) -> Union[Decimal, PandasNAType, None]:
    """Safely converts a value to Decimal, returning default on failure or NA/None."""
    if pd.isna(value) or value is None:  # pd.isna handles None, np.nan, pd.NaT, pd.NA
        return default
    try:
        return Decimal(str(value))  # Convert to string first to handle floats etc.
    except (InvalidOperation, TypeError, ValueError):
        return default


def format_order_id(order_id: Union[str, int, None]) -> str:
    """Formats an order ID for concise logging (e.g., last 6 chars)."""
    return str(order_id)[-6:] if order_id else "N/A"


def _format_for_log(
    value: Any,
    precision: int = 4,
    is_bool_trend: bool = False,
    color: Optional[str] = NEON["VALUE"],
) -> str:
    """Helper to format values for logging with NEON colors and precision."""
    reset = NEON["RESET"]
    val_color = color if color else ""  # Use provided color or default NEON["VALUE"]

    if pd.isna(value) or value is None:
        return f"{Style.DIM}N/A{reset}"
    if is_bool_trend:  # Special formatting for boolean trend values
        if value is True:
            return f"{NEON['SIDE_LONG']}Upward Flow{reset}"
        if value is False:
            return f"{NEON['SIDE_SHORT']}Downward Tide{reset}"
        return f"{Style.DIM}N/A (Trend Indeterminate){reset}"  # Should be caught by pd.isna
    if isinstance(value, Decimal):
        return f"{val_color}{value:.{precision}f}{reset}"
    if isinstance(value, (float, int)):
        return f"{val_color}{float(value):.{precision}f}{reset}"  # Ensure it's float for formatting
    if isinstance(value, bool):  # For other booleans not trends
        return f"{val_color}{str(value)}{reset}"
    return f"{val_color}{str(value)}{reset}"  # Fallback for other types


def format_price(
    exchange: ccxt.Exchange, symbol: str, price: Union[float, Decimal]
) -> str:
    """Formats a price to the exchange's required precision for the symbol."""
    try:
        return exchange.price_to_precision(symbol, float(price))
    except Exception:  # Fallback if precision info is unavailable or error occurs
        return str(
            Decimal(str(price))
            .quantize(Decimal("1e-8"), rounding=ROUND_HALF_UP)
            .normalize()
        )


def format_amount(
    exchange: ccxt.Exchange, symbol: str, amount: Union[float, Decimal]
) -> str:
    """Formats an amount (quantity) to the exchange's required precision."""
    try:
        return exchange.amount_to_precision(symbol, float(amount))
    except Exception:  # Fallback
        return str(
            Decimal(str(amount))
            .quantize(Decimal("1e-8"), rounding=ROUND_HALF_UP)
            .normalize()
        )


@retry(
    tries=CONFIG.retry_count,
    delay=CONFIG.retry_delay_seconds,
    backoff=2,
    logger=logger,
    exceptions=(
        ccxt.NetworkError,
        ccxt.RequestTimeout,
        ccxt.ExchangeNotAvailable,
        ccxt.DDoSProtection,
    ),
)
def safe_api_call(func, *args, **kwargs):
    """Wraps an API call with retry logic for common transient network/exchange errors."""
    return func(*args, **kwargs)


# Simplified SMS Alert for review context (actual Termux integration is more complex)
def send_sms_alert(message: str) -> bool:
    """Simulates sending an SMS alert by logging it."""
    logger.info(f"{NEON['STRATEGY']}[SMS SIMULATED]: {message}{NEON['RESET']}")
    # In a real Termux environment, this would use:
    # subprocess.run(["termux-sms-send", "-n", CONFIG.sms_recipient_number, message], ...)
    return True  # Assume success for simulation


def initialize_exchange() -> Optional[ccxt.Exchange]:
    """Initializes and configures the CCXT exchange object for Bybit."""
    logger.info(
        f"{NEON['INFO']}{Style.BRIGHT}Opening Portal to Bybit (Pyrmethus v2.8.0)...{NEON['RESET']}"
    )
    if not CONFIG.api_key or not CONFIG.api_secret:
        logger.warning(
            f"{NEON['WARNING']}API Key/Secret not found. Using a MOCK exchange object for simulation.{NEON['RESET']}"
        )
        return MockExchange()  # Return a mock object if keys are missing
    try:
        exchange = ccxt.bybit(
            {
                "apiKey": CONFIG.api_key,
                "secret": CONFIG.api_secret,
                "enableRateLimit": True,  # Enable CCXT's built-in rate limiter
                "options": {
                    "defaultType": "linear",  # For USDT perpetuals
                    "adjustForTimeDifference": True,  # Adjust for clock drift
                },
                "recvWindow": CONFIG.default_recv_window,  # Bybit V5 specific
            }
        )
        # exchange.set_sandbox_mode(True) # Uncomment for Bybit Testnet
        logger.debug("Loading market structures from Bybit...")
        exchange.load_markets(reload=True)  # Use reload=True for modern ccxt
        logger.debug(f"Loaded {len(exchange.markets)} market structures.")
        logger.debug("Performing initial balance check...")
        exchange.fetch_balance(params={"category": "linear"})  # V5 specific
        logger.success(
            f"{NEON['SUCCESS']}Portal to Bybit Opened & Authenticated (V5 API).{NEON['RESET']}"
        )

        if (
            hasattr(exchange, "sandbox") and exchange.sandbox
        ):  # Check if sandbox mode is active
            logger.warning(
                f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}!!! TESTNET MODE ACTIVE VIA CCXT SANDBOX FLAG !!!{NEON['RESET']}"
            )
        else:
            # Check if using Bybit's demo account (often has 'DEMO' in API key or specific URLs)
            # This is a heuristic; actual detection might need more specific checks if Bybit provides them.
            if "DEMO" in CONFIG.api_key.upper():  # Simple check
                logger.warning(
                    f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}!!! POTENTIAL DEMO/PAPER TRADING ACCOUNT DETECTED (via API Key) !!!{NEON['RESET']}"
                )
            else:
                logger.warning(
                    f"{NEON['CRITICAL']}!!! LIVE SCALPING MODE ACTIVE - EXTREME CAUTION !!!{NEON['RESET']}"
                )
        return exchange
    except Exception as e:
        logger.critical(f"{NEON['CRITICAL']}Portal opening failed: {e}{NEON['RESET']}")
        logger.debug(traceback.format_exc())
    return None


class MockExchange:  # For testing without real API keys
    def __init__(self):
        self.id = "mock_bybit"
        self.options = {"defaultType": "linear"}
        self.markets = {
            CONFIG.symbol: {
                "id": CONFIG.symbol.replace("/", "").replace(":", ""),
                "symbol": CONFIG.symbol,
                "contract": True,
                "linear": True,
                "limits": {"amount": {"min": 0.001}, "price": {"min": 0.1}},
                "precision": {"amount": 3, "price": 1},
            }
        }
        self.sandbox = True

    def market(self, s):
        return self.markets.get(s)

    def load_markets(self, reload=False):
        pass  # `reload` is the modern ccxt param

    def fetch_balance(self, params=None):
        return {
            CONFIG.usdt_symbol: {
                "free": Decimal("10000"),
                "total": Decimal("10000"),
                "used": Decimal("0"),
            }
        }

    def fetch_ticker(self, s):
        return {
            "last": Decimal("30000.0"),
            "bid": Decimal("29999.0"),
            "ask": Decimal("30001.0"),
        }

    def fetch_positions(self, symbols=None, params=None):
        global _active_trade_parts
        qty = sum(p["qty"] for p in _active_trade_parts)
        side = CONFIG.pos_none
        avg_px = Decimal("0")
        if qty > CONFIG.position_qty_epsilon:
            side = (
                _active_trade_parts[0]["side"]
                if _active_trade_parts
                else CONFIG.pos_none
            )
            total_value = sum(p["entry_price"] * p["qty"] for p in _active_trade_parts)
            avg_px = total_value / qty if qty > 0 else Decimal("0")

        # Bybit V5 `side` is "Buy" or "Sell" or "None" for no position
        exchange_side = (
            "Buy"
            if side == CONFIG.pos_long
            else ("Sell" if side == CONFIG.pos_short else "None")
        )

        return (
            [
                {
                    "info": {
                        "symbol": self.markets[CONFIG.symbol]["id"],
                        "positionIdx": 0,
                        "size": str(qty),
                        "avgPrice": str(avg_px),
                        "side": exchange_side,
                    }
                }
            ]
            if qty > CONFIG.position_qty_epsilon
            else []
        )

    def create_market_order(self, s, side, amt, params=None):
        return {
            "id": f"mock_mkt_{int(time.time() * 1000)}",
            "status": "closed",
            "average": self.fetch_ticker(s)["last"],
            "filled": float(amt),
            "timestamp": int(time.time() * 1000),
        }

    def create_limit_order(self, s, side, amt, price, params=None):
        return {
            "id": f"mock_lim_{int(time.time() * 1000)}",
            "status": "open",
            "price": float(price),
            "amount": float(amt),
        }

    def create_order(self, s, type, side, amt, price=None, params=None):  # For SL/TSL
        # Mock SL/TSL order creation
        if (
            type.lower() == "stopmarket"
            and params
            and (params.get("stopPrice") or params.get("trailingStop"))
        ):
            return {
                "id": f"mock_stop_{int(time.time() * 1000)}",
                "status": "open",
                "params": params,
                "symbol": s,
                "type": type,
                "side": side,
                "amount": float(amt),
            }
        # Fallback for other conditional orders
        return {"id": f"mock_cond_{int(time.time() * 1000)}", "status": "open"}

    def fetch_order(self, id, s, params=None):
        # Simulate limit order fill after a short delay
        if "lim_" in id:
            time.sleep(0.05)
            return {
                "id": id,
                "status": "closed",
                "average": self.fetch_ticker(s)["last"],
                "filled": 1.0,
                "timestamp": int(time.time() * 1000),
            }  # Assuming it fills with qty 1 for simplicity
        # Market and other orders fill immediately
        return {
            "id": id,
            "status": "closed",
            "average": self.fetch_ticker(s)["last"],
            "filled": 1.0,
            "timestamp": int(time.time() * 1000),
        }

    def fetch_open_orders(self, s, params=None):
        return []  # Assume no other open orders for simplicity

    def cancel_order(self, id, s, params=None):
        return {"id": id, "status": "canceled"}

    def set_leverage(self, leverage, s, params=None):
        return {"status": "ok", "info": {"leverage": str(leverage)}}  # Mock response

    def price_to_precision(self, s, p):
        return f"{float(p):.{self.markets[s]['precision']['price']}f}"

    def amount_to_precision(self, s, a):
        return f"{float(a):.{self.markets[s]['precision']['amount']}f}"

    def parse_timeframe(self, tf_str):  # Convert timeframe string to seconds
        if "m" in tf_str:
            return int(tf_str.replace("m", "")) * 60
        if "h" in tf_str:
            return int(tf_str.replace("h", "")) * 3600
        if "d" in tf_str:
            return int(tf_str.replace("d", "")) * 86400
        return 60  # Default to 1 minute if parsing fails

    has = {"fetchOHLCV": True, "fetchL2OrderBook": True}  # Advertise capabilities

    def fetch_ohlcv(
        self, s, tf, lim, since=None, params=None
    ):  # More structured mock OHLCV
        now_ms = int(time.time() * 1000)
        tf_s = self.parse_timeframe(tf)
        data_points = []
        start_ts = (
            now_ms - (lim - 1) * tf_s * 1000
        )  # Calculate start timestamp based on limit
        if (
            since is not None and since > start_ts
        ):  # If since is provided and more recent, adjust start
            start_ts = since
            # lim = (now_ms - since) // (tf_s * 1000) + 1 # Approximate new limit, though mock will still generate 'lim' for simplicity

        for i in range(lim):  # Still generate 'lim' candles for simplicity in mock
            ts = start_ts + i * tf_s * 1000  # Timestamping from calculated start
            if (
                ts > now_ms + tf_s
            ):  # Don't generate candles too far into the future (allow one partial future candle)
                continue

            price_base = 30000 + (i - lim / 2) * 10  # Base price trend
            price_noise = (random.random() - 0.5) * 50  # Random noise component

            open_p = (
                price_base + price_noise + (i % 5 - 2) * 3
            )  # Add some deterministic variation
            close_p = price_base + price_noise + (i % 3 - 1) * 2

            high_p = max(open_p, close_p) + abs(random.random() * 10)
            low_p = min(open_p, close_p) - abs(random.random() * 10)

            # Ensure OHLC order: O, H, L, C (high must be >= open/close, low <= open/close)
            if open_p > high_p:
                high_p = open_p
            if open_p < low_p:
                low_p = open_p
            if close_p > high_p:
                high_p = close_p
            if close_p < low_p:
                low_p = close_p

            volume = (
                100 + i + (ts % 50) + random.randint(0, 50)
            )  # Volume with some trend and randomness
            data_points.append([ts, open_p, high_p, low_p, close_p, volume])
        return data_points

    def fetch_l2_order_book(self, s, limit=None):
        last_price = self.fetch_ticker(s)["last"]
        book_depth = limit or 5
        bids_list = [
            [
                float(last_price) - i * 0.1 - random.random() * 0.1,
                1.0 + i * 0.1 + random.random(),
            ]
            for i in range(1, book_depth + 1)
        ]
        asks_list = [
            [
                float(last_price) + i * 0.1 + random.random() * 0.1,
                1.0 + i * 0.1 + random.random(),
            ]
            for i in range(1, book_depth + 1)
        ]
        return {
            "bids": bids_list,
            "asks": asks_list,
            "timestamp": int(time.time() * 1000),
            "datetime": datetime.now(pytz.utc).isoformat(),
        }


# --- Indicator Calculation Functions - Scrying the Market ---
vol_atr_analysis_results_cache: Dict[
    str, Any
] = {}  # Cache for ATR/Volume analysis results


def calculate_supertrend(
    df: pd.DataFrame, length: int, multiplier: Decimal, prefix: str = ""
) -> pd.DataFrame:
    val_col = f"{prefix}supertrend_val"
    trend_col = f"{prefix}trend"  # True for up, False for down, pd.NA if indeterminate
    long_flip_col = f"{prefix}st_long_flip"
    short_flip_col = f"{prefix}st_short_flip"
    output_cols = [val_col, trend_col, long_flip_col, short_flip_col]

    if (
        not all(c in df.columns for c in ["high", "low", "close"])
        or df.empty
        or len(df) < length + 1
    ):  # Min length for ATR
        logger.warning(
            f"{NEON['WARNING']}Scrying (Supertrend {prefix}): Insufficient data or missing HLC columns (Rows: {len(df)}, Min: {length + 1}). Populating NAs.{NEON['RESET']}"
        )
        for col in output_cols:
            df[col] = pd.NA
        return df

    try:
        temp_df_st = (
            df[["high", "low", "close"]].apply(pd.to_numeric, errors="coerce").copy()
        )  # Ensure numeric and use a copy

        st_df = temp_df_st.ta.supertrend(
            length=length, multiplier=float(multiplier), append=False
        )

        m_str = str(float(multiplier))
        pta_val_col_pattern = f"SUPERT_{length}_{m_str}"
        pta_dir_col_pattern = f"SUPERTd_{length}_{m_str}"

        if (
            st_df is None
            or pta_val_col_pattern not in st_df.columns
            or pta_dir_col_pattern not in st_df.columns
        ):
            logger.error(
                f"{NEON['ERROR']}Scrying (Supertrend {prefix}): pandas_ta did not return expected SuperTrend columns. Found: {st_df.columns if st_df is not None else 'None'}. Expected patterns like: {pta_val_col_pattern}, {pta_dir_col_pattern}{NEON['RESET']}"
            )
            for col in output_cols:
                df[col] = pd.NA
            return df

        df[val_col] = st_df[pta_val_col_pattern].apply(
            lambda x: safe_decimal_conversion(x, pd.NA)
        )
        df[trend_col] = st_df[pta_dir_col_pattern].apply(
            lambda x: True if x == 1 else (False if x == -1 else pd.NA)
        )

        prev_trend = df[trend_col].shift(1)
        df[long_flip_col] = (prev_trend == False) & (df[trend_col] == True)
        df[short_flip_col] = (prev_trend == True) & (df[trend_col] == False)

        df[long_flip_col] = df[long_flip_col].fillna(False)
        df[short_flip_col] = df[short_flip_col].fillna(False)

        if not df.empty and val_col in df.columns and not pd.isna(df[val_col].iloc[-1]):
            logger.debug(
                f"Scrying (Supertrend({length},{multiplier},{prefix.replace('_', '') if prefix else 'Primary'})): "
                f"Value={_format_for_log(df[val_col].iloc[-1], color=NEON['VALUE'])}, "
                f"Trend={_format_for_log(df[trend_col].iloc[-1], is_bool_trend=True)}, "
                f"LongFlip={df[long_flip_col].iloc[-1]}, ShortFlip={df[short_flip_col].iloc[-1]}"
            )
    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}Scrying (Supertrend {prefix}): Error during calculation: {e}{NEON['RESET']}"
        )
        logger.debug(traceback.format_exc())
        for col in output_cols:
            df[col] = pd.NA
    return df


def calculate_momentum(df: pd.DataFrame, length: int) -> pd.DataFrame:
    momentum_col = "momentum"
    if "close" not in df.columns or df.empty or len(df) < length:
        df[momentum_col] = pd.NA
        logger.warning(
            f"{NEON['WARNING']}Scrying (Momentum): Insufficient data for momentum (Rows: {len(df)}, Min: {length}). Populating NAs.{NEON['RESET']}"
        )
        return df

    try:
        temp_df_mom = df[["close"]].apply(pd.to_numeric, errors="coerce").copy()
        mom_series = temp_df_mom.ta.mom(length=length, append=False)
        df[momentum_col] = mom_series.apply(lambda x: safe_decimal_conversion(x, pd.NA))

        if (
            not df.empty
            and momentum_col in df.columns
            and not pd.isna(df[momentum_col].iloc[-1])
        ):
            logger.debug(
                f"Scrying (Momentum({length})): Value={_format_for_log(df[momentum_col].iloc[-1], color=NEON['VALUE'])}"
            )
    except Exception as e:
        logger.error(f"{NEON['ERROR']}Scrying (Momentum): Error: {e}{NEON['RESET']}")
        logger.debug(traceback.format_exc())
        df[momentum_col] = pd.NA
    return df


def calculate_ehlers_fisher(
    df: pd.DataFrame, length: int, signal_length: int
) -> pd.DataFrame:
    fisher_col = "ehlers_fisher"
    signal_col = "ehlers_signal"
    output_cols = [fisher_col, signal_col]

    required_ohlc_cols = ["high", "low", "close"]
    if (
        not all(c in df.columns for c in required_ohlc_cols)
        or df.empty
        or len(df) < length + signal_length + 5
    ):  # Approx min length
        logger.warning(
            f"{NEON['WARNING']}Scrying (EhlersFisher): Insufficient data or missing HLC columns (Rows: {len(df)}, Min: {length + signal_length + 5}). Populating NAs.{NEON['RESET']}"
        )
        for col in output_cols:
            df[col] = pd.NA
        return df

    try:
        temp_df_ef = df[required_ohlc_cols].apply(pd.to_numeric, errors="coerce").copy()
        fisher_df_pta = temp_df_ef.ta.fisher(
            length=length, signal=signal_length, append=False
        )

        pta_fisher_col_name = f"FISHERT_{length}_{signal_length}"
        pta_signal_col_name = f"FISHERTs_{length}_{signal_length}"

        if fisher_df_pta is not None and not fisher_df_pta.empty:
            df[fisher_col] = (
                fisher_df_pta[pta_fisher_col_name].apply(
                    lambda x: safe_decimal_conversion(x, pd.NA)
                )
                if pta_fisher_col_name in fisher_df_pta
                else pd.NA
            )
            df[signal_col] = (
                fisher_df_pta[pta_signal_col_name].apply(
                    lambda x: safe_decimal_conversion(x, pd.NA)
                )
                if pta_signal_col_name in fisher_df_pta
                else pd.NA
            )
        else:
            df[fisher_col], df[signal_col] = pd.NA, pd.NA

        if (
            not df.empty
            and fisher_col in df.columns
            and signal_col in df.columns
            and not pd.isna(df[fisher_col].iloc[-1])
            and not pd.isna(df[signal_col].iloc[-1])
        ):
            f_val, s_val = df[fisher_col].iloc[-1], df[signal_col].iloc[-1]
            logger.debug(
                f"Scrying (EhlersFisher({length},{signal_length})): Fisher={_format_for_log(f_val, color=NEON['VALUE'])}, Signal={_format_for_log(s_val, color=NEON['VALUE'])}"
            )
    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}Scrying (EhlersFisher): Error during calculation: {e}{NEON['RESET']}"
        )
        logger.debug(traceback.format_exc())
        for col in output_cols:
            df[col] = pd.NA
    return df


def analyze_volume_atr(
    df: pd.DataFrame,
    short_atr_len: int,
    long_atr_len: int,
    vol_ma_len: int,
    dynamic_sl_enabled: bool,
) -> Dict[str, Union[Decimal, PandasNAType, None]]:
    results: Dict[str, Union[Decimal, PandasNAType, None]] = {
        "atr_short": pd.NA,
        "atr_long": pd.NA,
        "volatility_regime": VolatilityRegime.NORMAL,
        "volume_ma": pd.NA,
        "last_volume": pd.NA,
        "volume_ratio": pd.NA,
    }
    if df.empty or not all(c in df.columns for c in ["high", "low", "close", "volume"]):
        logger.warning(
            f"{NEON['WARNING']}Scrying (Vol/ATR): Missing HLCV columns.{NEON['RESET']}"
        )
        return results

    try:
        temp_df = df.copy()  # Work on a copy
        # Ensure numeric types for calculations
        for col_name in ["high", "low", "close", "volume"]:
            temp_df[col_name] = pd.to_numeric(temp_df[col_name], errors="coerce")

        # Drop rows where HLC are NaN as ATR cannot be calculated
        temp_df.dropna(subset=["high", "low", "close"], inplace=True)
        if (
            temp_df.empty
            or len(temp_df)
            < max(short_atr_len, long_atr_len if dynamic_sl_enabled else 0, vol_ma_len)
            + 1
        ):  # +1 for pandas_ta
            logger.warning(
                f"{NEON['WARNING']}Scrying (Vol/ATR): Insufficient data after NaN drop for ATR/VolMA (Rows: {len(temp_df)}).{NEON['RESET']}"
            )
            return results

        # ATR Short
        if len(temp_df) >= short_atr_len + 1:
            results["atr_short"] = safe_decimal_conversion(
                temp_df.ta.atr(length=short_atr_len, append=False).iloc[-1], pd.NA
            )

        # ATR Long & Volatility Regime (if enabled)
        if dynamic_sl_enabled:
            if len(temp_df) >= long_atr_len + 1:
                results["atr_long"] = safe_decimal_conversion(
                    temp_df.ta.atr(length=long_atr_len, append=False).iloc[-1], pd.NA
                )

            atr_s, atr_l = results["atr_short"], results["atr_long"]
            if (
                not pd.isna(atr_s)
                and not pd.isna(atr_l)
                and atr_s is not None
                and atr_l is not None
                and atr_l > CONFIG.position_qty_epsilon
            ):
                vol_ratio = atr_s / atr_l
                if vol_ratio < CONFIG.volatility_ratio_low_threshold:
                    results["volatility_regime"] = VolatilityRegime.LOW
                elif vol_ratio > CONFIG.volatility_ratio_high_threshold:
                    results["volatility_regime"] = VolatilityRegime.HIGH

        # Volume Analysis (use original df for last_volume to get the very latest, temp_df for MA)
        if "volume" in df.columns and not df["volume"].empty:
            results["last_volume"] = safe_decimal_conversion(
                df["volume"].iloc[-1], pd.NA
            )

        if (
            "volume" in temp_df.columns and len(temp_df) >= vol_ma_len
        ):  # Use temp_df (cleaned) for MA
            results["volume_ma"] = safe_decimal_conversion(
                temp_df["volume"]
                .rolling(window=vol_ma_len, min_periods=1)
                .mean()
                .iloc[-1],
                pd.NA,
            )

        # Volume Ratio
        if (
            not pd.isna(results["last_volume"])
            and not pd.isna(results["volume_ma"])
            and results["volume_ma"] is not None
            and results["volume_ma"] > CONFIG.position_qty_epsilon
            and results["last_volume"] is not None
        ):
            try:
                results["volume_ratio"] = results["last_volume"] / results["volume_ma"]  # type: ignore
            except (DivisionByZero, TypeError):
                results["volume_ratio"] = pd.NA

        logger.debug(
            f"Scrying (Vol/ATR): ShortATR={_format_for_log(results['atr_short'], 5)}, Regime={results['volatility_regime'].value},VolRatio={_format_for_log(results['volume_ratio'], 2)}"
        )

    except Exception as e:
        logger.error(f"{NEON['ERROR']}Scrying (Vol/ATR): Error: {e}{NEON['RESET']}")
        logger.debug(traceback.format_exc())
        # Reset results to NA on error, keep default regime
        results = {key: pd.NA for key in results if key != "volatility_regime"}
        results["volatility_regime"] = VolatilityRegime.NORMAL
    return results


def get_current_atr_sl_multiplier() -> Decimal:
    """Determines the ATR multiplier for SL based on current volatility regime."""
    if (
        not CONFIG.enable_dynamic_atr_sl or not vol_atr_analysis_results_cache
    ):  # Check if cache has been populated
        return CONFIG.atr_stop_loss_multiplier  # Fallback

    regime = vol_atr_analysis_results_cache.get(
        "volatility_regime", VolatilityRegime.NORMAL
    )
    if regime == VolatilityRegime.LOW:
        return CONFIG.atr_sl_multiplier_low_vol
    if regime == VolatilityRegime.HIGH:
        return CONFIG.atr_sl_multiplier_high_vol
    return CONFIG.atr_sl_multiplier_normal_vol  # Default to normal


def calculate_all_indicators(
    df: pd.DataFrame, config: Config
) -> Tuple[pd.DataFrame, Dict[str, Any]]:  # Return type for dict
    """Calculates all indicators required by the active strategy and common analysis."""
    global vol_atr_analysis_results_cache

    df_copy = df.copy()  # Work on a copy to avoid modifying the original DataFrame

    # Calculate indicators based on active strategy
    if config.strategy_name == StrategyName.DUAL_SUPERTREND_MOMENTUM:
        df_copy = calculate_supertrend(
            df_copy, config.st_atr_length, config.st_multiplier
        )
        df_copy = calculate_supertrend(
            df_copy,
            config.confirm_st_atr_length,
            config.confirm_st_multiplier,
            prefix="confirm_",
        )
        df_copy = calculate_momentum(df_copy, config.momentum_period)
    elif config.strategy_name == StrategyName.EHLERS_FISHER:
        df_copy = calculate_ehlers_fisher(
            df_copy, config.ehlers_fisher_length, config.ehlers_fisher_signal_length
        )
    # Add other strategy-specific indicator calls here if new strategies are added

    # Common analysis (ATR, Volume) - always calculated
    vol_atr_analysis_results_cache = analyze_volume_atr(
        df_copy,
        config.atr_short_term_period,
        config.atr_long_term_period,
        config.volume_ma_period,
        config.enable_dynamic_atr_sl,
    )
    return df_copy, vol_atr_analysis_results_cache


# --- Position & Order Management - Manipulating Market Presence ---
def get_current_position(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    """Consolidates bot's internal state (_active_trade_parts) with raw exchange position."""
    global _active_trade_parts
    exchange_pos_data = _get_raw_exchange_position(exchange, symbol)

    if not _active_trade_parts:  # If bot has no memory of active parts, trust exchange
        return exchange_pos_data

    # Consolidate bot's understanding of the position from its parts
    consolidated_qty = sum(part.get("qty", Decimal(0)) for part in _active_trade_parts)

    if (
        consolidated_qty <= CONFIG.position_qty_epsilon
    ):  # If bot thinks position is closed
        _active_trade_parts.clear()
        save_persistent_state()  # Save cleared state
        return exchange_pos_data  # Trust exchange if bot thinks it's flat

    # If bot thinks it has a position, calculate its average entry and side
    total_value = sum(
        part.get("entry_price", Decimal(0)) * part.get("qty", Decimal(0))
        for part in _active_trade_parts
    )
    avg_entry_price = (
        total_value / consolidated_qty if consolidated_qty > 0 else Decimal("0")
    )
    current_pos_side = (
        _active_trade_parts[0]["side"] if _active_trade_parts else CONFIG.pos_none
    )  # Assuming all parts have same side

    # Log discrepancy if bot state and exchange state differ significantly
    # Use a larger epsilon for comparing consolidated qty due to potential minor fill differences over multiple parts
    if exchange_pos_data["side"] != current_pos_side or abs(
        exchange_pos_data["qty"] - consolidated_qty
    ) > CONFIG.position_qty_epsilon * Decimal("10"):  # Increased tolerance
        logger.warning(
            f"{NEON['WARNING']}Position Discrepancy Detected! "
            f"Bot State: {current_pos_side} Qty {NEON['QTY']}{consolidated_qty:.5f}{NEON['WARNING']}. "
            f"Exchange State: {exchange_pos_data['side']} Qty {NEON['QTY']}{exchange_pos_data['qty']:.5f}{NEON['WARNING']}. "
            f"Consider manual review or state reset if persistent.{NEON['RESET']}"
        )
        # Depending on reconciliation strategy, might choose to trust exchange or bot, or halt.
        # For now, return bot's view but log warning.

    return {
        "side": current_pos_side,
        "qty": consolidated_qty,
        "entry_price": avg_entry_price,
        "num_parts": len(_active_trade_parts),
    }


def _get_raw_exchange_position(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    """Fetches raw position data from the exchange for Bybit V5 API."""
    default_pos_state: Dict[str, Any] = {
        "side": CONFIG.pos_none,
        "qty": Decimal("0.0"),
        "entry_price": Decimal("0.0"),
    }
    try:
        market = exchange.market(symbol)
        market_id = market["id"]
        # Determine category based on market type (linear/inverse)
        category = (
            "inverse" if market.get("inverse") else "linear"
        )  # Default to linear if not specified

        params = {"category": category, "symbol": market_id}
        fetched_positions = safe_api_call(
            exchange.fetch_positions, symbols=[symbol], params=params
        )

        if not fetched_positions:
            return default_pos_state

        for pos_data in fetched_positions:
            pos_info = pos_data.get("info", {})
            if pos_info.get("symbol") != market_id:
                continue

            # For Bybit V5 One-Way Mode, positionIdx is 0.
            if int(pos_info.get("positionIdx", -1)) == 0:
                size_dec = safe_decimal_conversion(pos_info.get("size", "0"))
                if (
                    size_dec > CONFIG.position_qty_epsilon
                ):  # If position size is significant
                    entry_price_dec = safe_decimal_conversion(pos_info.get("avgPrice"))
                    bybit_side_str = pos_info.get(
                        "side"
                    )  # "Buy" for Long, "Sell" for Short

                    current_pos_side = (
                        CONFIG.pos_long
                        if bybit_side_str == "Buy"
                        else (
                            CONFIG.pos_short
                            if bybit_side_str == "Sell"
                            else CONFIG.pos_none
                        )
                    )

                    if current_pos_side != CONFIG.pos_none:
                        return {
                            "side": current_pos_side,
                            "qty": size_dec,
                            "entry_price": entry_price_dec,
                        }
    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}_get_raw_exchange_position: Error fetching position for {symbol}: {e}{NEON['RESET']}"
        )
        logger.debug(traceback.format_exc())
    return default_pos_state


def set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int) -> bool:
    """Sets leverage for the symbol using Bybit V5 API parameters."""
    logger.info(
        f"{NEON['INFO']}Setting leverage to {NEON['VALUE']}{leverage}x{NEON['INFO']} for {NEON['VALUE']}{symbol}{NEON['INFO']}...{NEON['RESET']}"
    )
    try:
        # Bybit V5 API requires category and separate buy/sell leverage params
        params = {
            "category": "linear",  # Assuming linear contracts
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        }
        response = safe_api_call(
            exchange.set_leverage, leverage, symbol, params=params
        )  # ccxt handles mapping leverage to buyLeverage/sellLeverage
        logger.info(
            f"{NEON['SUCCESS']}Leverage set for {NEON['VALUE']}{symbol}{NEON['SUCCESS']} to {NEON['VALUE']}{leverage}x{NEON['SUCCESS']}. Response: {response}{NEON['RESET']}"
        )
        return True
    except ccxt.ExchangeError as e:
        err_str = str(e).lower()
        # Bybit V5 "Set leverage not modified" error code is 110044.
        if "leverage not modified" in err_str or "110044" in err_str:
            logger.info(
                f"{NEON['INFO']}Leverage for {NEON['VALUE']}{symbol}{NEON['INFO']} already {NEON['VALUE']}{leverage}x{NEON['INFO']} or no change needed.{NEON['RESET']}"
            )
            return True
        logger.error(
            f"{NEON['ERROR']}Failed to set leverage for {symbol} to {leverage}x: {e}{NEON['RESET']}"
        )
    except Exception as e_unexp:  # Catch any other unexpected errors
        logger.error(
            f"{NEON['ERROR']}Unexpected error setting leverage for {symbol}: {e_unexp}{NEON['RESET']}"
        )
        logger.debug(traceback.format_exc())
    return False


def calculate_dynamic_risk() -> Decimal:
    """Calculates dynamic risk percentage based on recent performance."""
    if not CONFIG.enable_dynamic_risk:
        return CONFIG.risk_per_trade_percentage

    trend = trade_metrics.get_performance_trend(CONFIG.dynamic_risk_perf_window)
    base_risk = CONFIG.risk_per_trade_percentage
    min_risk = CONFIG.dynamic_risk_min_pct
    max_risk = CONFIG.dynamic_risk_max_pct

    # Scale risk based on performance trend (0.0 to 1.0)
    if trend >= 0.5:  # Winning more or breakeven
        scale_factor = (
            trend - 0.5
        ) / 0.5  # Scale from 0 (at 0.5 trend) to 1 (at 1.0 trend)
        dynamic_risk = base_risk + (max_risk - base_risk) * Decimal(scale_factor)
    else:  # Losing more
        scale_factor = (
            0.5 - trend
        ) / 0.5  # Scale from 0 (at 0.5 trend) to 1 (at 0.0 trend)
        dynamic_risk = base_risk - (base_risk - min_risk) * Decimal(scale_factor)

    final_risk = max(
        min_risk, min(max_risk, dynamic_risk)
    )  # Clamp within min/max bounds
    logger.info(
        f"{NEON['INFO']}Dynamic Risk Calculation: Perf Trend ({CONFIG.dynamic_risk_perf_window} trades)={NEON['VALUE']}{trend:.2f}{NEON['INFO']}, "
        f"BaseRisk={NEON['VALUE']}{base_risk:.3%}{NEON['INFO']}, AdjustedRisk={NEON['VALUE']}{final_risk:.3%}{NEON['RESET']}"
    )
    return final_risk


def calculate_position_size(
    usdt_equity: Decimal,
    risk_pct: Decimal,
    entry_px: Decimal,
    sl_px: Decimal,
    leverage: int,
    symbol: str,
    exchange: ccxt.Exchange,
) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """Calculates position size based on risk, equity, entry/SL, and leverage."""
    if not (
        entry_px > 0
        and sl_px > 0
        and 0 < risk_pct < 1
        and usdt_equity > 0
        and leverage > 0
    ):
        logger.error(
            f"{NEON['ERROR']}RiskCalc: Invalid inputs for position sizing.{NEON['RESET']}"
        )
        return None, None

    price_diff_per_unit = abs(entry_px - sl_px)
    if (
        price_diff_per_unit < CONFIG.position_qty_epsilon
    ):  # Avoid division by zero or near-zero
        logger.error(
            f"{NEON['ERROR']}RiskCalc: Entry and SL prices are too close or invalid.{NEON['RESET']}"
        )
        return None, None

    try:
        risk_amount_usdt = usdt_equity * risk_pct
        # Quantity in base currency (e.g., BTC)
        quantity_raw = risk_amount_usdt / price_diff_per_unit

        # Format quantity to exchange's precision
        quantity_prec_str = format_amount(exchange, symbol, quantity_raw)
        quantity_final = Decimal(quantity_prec_str)

        if quantity_final <= CONFIG.position_qty_epsilon:
            logger.warning(
                f"{NEON['WARNING']}RiskCalc: Calculated quantity ({_format_for_log(quantity_final, 8, color=NEON['QTY'])}) is negligible or zero.{NEON['RESET']}"
            )
            return None, None

        # Calculate required margin
        position_value_usdt = quantity_final * entry_px
        margin_required = position_value_usdt / Decimal(leverage)

        logger.debug(
            f"RiskCalc: RiskAmt={_format_for_log(risk_amount_usdt, 2)} USDT, Qty={_format_for_log(quantity_final, 8, color=NEON['QTY'])}, "
            f"MarginReq={_format_for_log(margin_required, 4)} USDT"
        )
        return quantity_final, margin_required
    except (DivisionByZero, InvalidOperation, Exception) as e:
        logger.error(
            f"{NEON['ERROR']}RiskCalc: Error during position size calculation: {e}{NEON['RESET']}"
        )
        return None, None


def wait_for_order_fill(
    exchange: ccxt.Exchange,
    order_id: str,
    symbol: str,
    timeout_seconds: int,
    order_type: str = "market",
) -> Optional[Dict[str, Any]]:
    """Waits for an order to fill or fail, with a timeout."""
    start_time = time.time()
    short_order_id = format_order_id(order_id)
    logger.info(
        f"{NEON['INFO']}Order Vigil ({order_type}): ...{short_order_id} for '{symbol}' (Timeout: {timeout_seconds}s)...{NEON['RESET']}"
    )
    params = (
        {"category": "linear"}
        if exchange.options.get("defaultType") == "linear"
        else {}
    )

    while time.time() - start_time < timeout_seconds:
        try:
            order_details = safe_api_call(
                exchange.fetch_order, order_id, symbol, params=params
            )
            status = order_details.get("status")
            if status == "closed":  # Filled
                logger.success(
                    f"{NEON['SUCCESS']}Order Vigil: ...{short_order_id} FILLED/CLOSED.{NEON['RESET']}"
                )
                return order_details
            if status in ["canceled", "rejected", "expired"]:  # Failed
                logger.error(
                    f"{NEON['ERROR']}Order Vigil: ...{short_order_id} FAILED with status: '{status}'.{NEON['RESET']}"
                )
                return order_details

            # Still open or partially filled, continue polling
            logger.debug(
                f"Order ...{short_order_id} status: {status}. Vigil continues..."
            )
            time.sleep(
                1.0 if order_type == "limit" else 0.75
            )  # Poll more frequently for market orders
        except ccxt.OrderNotFound:  # Can happen due to propagation delay
            logger.warning(
                f"{NEON['WARNING']}Order Vigil: ...{short_order_id} not found yet. Retrying...{NEON['RESET']}"
            )
            time.sleep(1.5)  # Longer sleep for propagation
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e_net:  # Network issues
            logger.warning(
                f"{NEON['WARNING']}Order Vigil: Network issue for ...{short_order_id}: {e_net}. Retrying...{NEON['RESET']}"
            )
            time.sleep(3)
        except Exception as e:  # Other errors
            logger.warning(
                f"{NEON['WARNING']}Order Vigil: Error checking ...{short_order_id}: {type(e).__name__}. Retrying...{NEON['RESET']}"
            )
            logger.debug(traceback.format_exc())
            time.sleep(2)

    logger.error(
        f"{NEON['ERROR']}Order Vigil: ...{short_order_id} fill TIMED OUT after {timeout_seconds}s.{NEON['RESET']}"
    )
    # Attempt one final fetch after timeout
    try:
        final_details = safe_api_call(
            exchange.fetch_order, order_id, symbol, params=params
        )
        logger.info(
            f"Final status for ...{short_order_id} after timeout: {final_details.get('status', 'unknown')}"
        )
        return final_details
    except Exception as e_final:
        logger.error(
            f"{NEON['ERROR']}Final check for ...{short_order_id} failed: {type(e_final).__name__}{NEON['RESET']}"
        )
        return None


def place_risked_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    risk_percentage: Decimal,
    current_short_atr: Union[Decimal, PandasNAType, None],
    leverage: int,
    max_order_cap_usdt: Decimal,
    margin_check_buffer: Decimal,
    tsl_percent: Decimal,
    tsl_activation_offset_percent: Decimal,
    entry_type: OrderEntryType,
    is_scale_in: bool = False,
    existing_position_avg_price: Optional[Decimal] = None,
) -> Optional[Dict[str, Any]]:
    """Places a market or limit order with risk calculation, SL, and TSL."""
    global _active_trade_parts
    action_type = "Scale-In" if is_scale_in else "Initial Entry"
    logger.info(
        f"{NEON['ACTION']}Ritual of {action_type} ({entry_type.value}): {side.upper()} for '{symbol}'...{NEON['RESET']}"
    )

    if (
        pd.isna(current_short_atr)
        or current_short_atr is None
        or current_short_atr <= 0
    ):
        logger.error(
            f"{NEON['ERROR']}Invalid Short ATR ({_format_for_log(current_short_atr)}) for {action_type}. Cannot proceed.{NEON['RESET']}"
        )
        return None

    v5_api_category = "linear"  # Assuming linear contracts for Bybit V5
    try:
        # --- 1. Pre-computation and Checks ---
        balance_data = safe_api_call(
            exchange.fetch_balance, params={"category": v5_api_category}
        )
        market_info = exchange.market(symbol)
        min_qty_allowed = safe_decimal_conversion(
            market_info.get("limits", {}).get("amount", {}).get("min"), Decimal("0")
        )

        usdt_equity = safe_decimal_conversion(
            balance_data.get(CONFIG.usdt_symbol, {}).get("total"), Decimal("NaN")
        )
        usdt_free_margin = safe_decimal_conversion(
            balance_data.get(CONFIG.usdt_symbol, {}).get("free"), Decimal("NaN")
        )

        if usdt_equity.is_nan() or usdt_equity <= 0:
            logger.error(
                f"{NEON['ERROR']}Invalid account equity ({_format_for_log(usdt_equity)}). Cannot place order.{NEON['RESET']}"
            )
            return None

        # --- 2. Estimate Entry Price ---
        ticker = safe_api_call(exchange.fetch_ticker, symbol)
        signal_price = safe_decimal_conversion(
            ticker.get("last"), pd.NA
        )  # Use last price as signal price
        if pd.isna(signal_price) or signal_price <= 0:
            logger.error(
                f"{NEON['ERROR']}Failed to get valid signal price ({_format_for_log(signal_price)}). Cannot place order.{NEON['RESET']}"
            )
            return None

        # --- 3. Calculate Estimated SL Price ---
        sl_atr_multiplier = get_current_atr_sl_multiplier()
        sl_distance = current_short_atr * sl_atr_multiplier
        sl_price_estimated = (
            (signal_price - sl_distance)
            if side == CONFIG.side_buy
            else (signal_price + sl_distance)
        )
        if sl_price_estimated <= 0:  # SL price must be positive
            logger.error(
                f"{NEON['ERROR']}Invalid estimated SL price ({_format_for_log(sl_price_estimated)}). Cannot place order.{NEON['RESET']}"
            )
            return None

        # --- 4. Calculate Position Size ---
        current_order_risk_pct = (
            calculate_dynamic_risk()
            if CONFIG.enable_dynamic_risk and not is_scale_in
            else (
                CONFIG.scale_in_risk_percentage
                if is_scale_in
                else CONFIG.risk_per_trade_percentage
            )
        )

        order_quantity, estimated_margin = calculate_position_size(
            usdt_equity,
            current_order_risk_pct,
            signal_price,
            sl_price_estimated,
            leverage,
            symbol,
            exchange,
        )
        if order_quantity is None or order_quantity <= CONFIG.position_qty_epsilon:
            logger.error(
                f"{NEON['ERROR']}Position size calculation failed or resulted in zero/negligible quantity for {action_type}. Qty: {_format_for_log(order_quantity, color=NEON['QTY'])}{NEON['RESET']}"
            )
            return None

        # --- 5. Perform Margin and Cap Checks ---
        order_value_usdt = order_quantity * signal_price
        if order_value_usdt > max_order_cap_usdt:
            logger.info(
                f"{NEON['INFO']}Order value ({_format_for_log(order_value_usdt, 2)}) exceeds MAX_ORDER_USDT_AMOUNT ({_format_for_log(max_order_cap_usdt, 2)}). Capping order size.{NEON['RESET']}"
            )
            capped_qty_raw = max_order_cap_usdt / signal_price
            order_quantity = Decimal(format_amount(exchange, symbol, capped_qty_raw))
            logger.info(
                f"{NEON['INFO']}Order quantity capped to {_format_for_log(order_quantity, color=NEON['QTY'])}.{NEON['RESET']}"
            )
            if order_quantity <= CONFIG.position_qty_epsilon:  # Recheck after capping
                logger.error(
                    f"{NEON['ERROR']}Capped order quantity is negligible. Cannot place order.{NEON['RESET']}"
                )
                return None
            if order_quantity is not None:  # Ensure mypy
                estimated_margin = (order_quantity * signal_price) / Decimal(
                    leverage
                )  # Recalculate margin

        if (
            min_qty_allowed > 0
            and order_quantity is not None
            and order_quantity < min_qty_allowed
        ):
            logger.error(
                f"{NEON['ERROR']}Calculated quantity {_format_for_log(order_quantity, color=NEON['QTY'])} is below minimum allowed {_format_for_log(min_qty_allowed, color=NEON['QTY'])}.{NEON['RESET']}"
            )
            return None

        if estimated_margin is not None and (
            usdt_free_margin.is_nan()
            or usdt_free_margin < estimated_margin * margin_check_buffer
        ):
            logger.error(
                f"{NEON['ERROR']}Insufficient free margin for {action_type}. "
                f"Need approx: {_format_for_log(estimated_margin * margin_check_buffer, 2)} USDT, "
                f"Available: {_format_for_log(usdt_free_margin, 2)} USDT.{NEON['RESET']}"
            )
            return None

        # --- 6. Place Entry Order ---
        entry_order_id: Optional[str] = None
        entry_order_response: Optional[Dict[str, Any]] = None
        limit_price_str_for_slippage: Optional[str] = (
            None  # For slippage calculation if limit order
        )

        if entry_type == OrderEntryType.MARKET:
            entry_order_response = safe_api_call(
                exchange.create_market_order,
                symbol,
                side,
                float(order_quantity),
                params={"category": v5_api_category, "positionIdx": 0},
            )
            entry_order_id = entry_order_response.get("id")
        elif entry_type == OrderEntryType.LIMIT:
            pip_value = Decimal("1") / (
                Decimal("10") ** market_info["precision"]["price"]
            )
            limit_offset_value = CONFIG.limit_order_offset_pips * pip_value
            limit_entry_price = (
                (signal_price - limit_offset_value)
                if side == CONFIG.side_buy
                else (signal_price + limit_offset_value)
            )
            limit_price_str_for_slippage = format_price(
                exchange, symbol, limit_entry_price
            )
            logger.info(
                f"Placing LIMIT {action_type} order: Qty={_format_for_log(order_quantity, color=NEON['QTY'])}, Price={_format_for_log(limit_price_str_for_slippage, color=NEON['PRICE'])}"
            )
            entry_order_response = safe_api_call(
                exchange.create_limit_order,
                symbol,
                side,
                float(order_quantity),
                float(limit_price_str_for_slippage),
                params={"category": v5_api_category, "positionIdx": 0},
            )
            entry_order_id = entry_order_response.get("id")

        if not entry_order_id:
            logger.critical(
                f"{NEON['CRITICAL']}{action_type} {entry_type.value} order placement FAILED to return an ID!{NEON['RESET']}"
            )
            return None

        # --- 7. Wait for Entry Order Fill ---
        fill_timeout = (
            CONFIG.limit_order_fill_timeout_seconds
            if entry_type == OrderEntryType.LIMIT
            else CONFIG.order_fill_timeout_seconds
        )
        filled_entry_details = wait_for_order_fill(
            exchange,
            entry_order_id,
            symbol,
            fill_timeout,
            order_type=entry_type.value.lower(),
        )

        if entry_type == OrderEntryType.LIMIT and (
            not filled_entry_details or filled_entry_details.get("status") != "closed"
        ):
            logger.warning(
                f"{NEON['WARNING']}Limit order ...{format_order_id(entry_order_id)} did not fill within timeout or failed. Attempting cancellation.{NEON['RESET']}"
            )
            try:
                safe_api_call(
                    exchange.cancel_order,
                    entry_order_id,
                    symbol,
                    params={"category": v5_api_category},
                )
            except Exception as e_cancel:
                logger.error(
                    f"Failed to cancel unfilled/failed limit order ...{format_order_id(entry_order_id)}: {e_cancel}"
                )
            return None

        if not filled_entry_details or filled_entry_details.get("status") != "closed":
            logger.error(
                f"{NEON['ERROR']}{action_type} order ...{format_order_id(entry_order_id)} was not successfully filled/closed. Status: {filled_entry_details.get('status') if filled_entry_details else 'timeout'}.{NEON['RESET']}"
            )
            return None

        actual_fill_price = safe_decimal_conversion(filled_entry_details.get("average"))
        actual_filled_quantity = safe_decimal_conversion(
            filled_entry_details.get("filled")
        )
        entry_timestamp_ms = filled_entry_details.get("timestamp")  # Should be int

        if (
            actual_filled_quantity <= CONFIG.position_qty_epsilon
            or actual_fill_price <= CONFIG.position_qty_epsilon
            or entry_timestamp_ms is None
        ):
            logger.critical(
                f"{NEON['CRITICAL']}Invalid fill data for {action_type} order ...{format_order_id(entry_order_id)}. "
                f"Qty: {_format_for_log(actual_filled_quantity, color=NEON['QTY'])}, Price: {_format_for_log(actual_fill_price, color=NEON['PRICE'])}.{NEON['RESET']}"
            )
            return None

        # Calculate slippage
        reference_price_for_slippage = (
            signal_price
            if entry_type == OrderEntryType.MARKET
            else Decimal(limit_price_str_for_slippage)
        )  # type: ignore
        slippage = abs(actual_fill_price - reference_price_for_slippage)
        slippage_percentage = (
            (slippage / reference_price_for_slippage * 100)
            if reference_price_for_slippage > 0
            else Decimal(0)
        )
        logger.info(
            f"{action_type} Slippage: RefPx={_format_for_log(reference_price_for_slippage, 4, color=NEON['PARAM'])}, FillPx={_format_for_log(actual_fill_price, 4, color=NEON['PRICE'])}, "
            f"Slip={_format_for_log(slippage, 4, color=NEON['WARNING'])} ({slippage_percentage:.3f}%)"
        )

        # --- Add to Active Trade Parts ---
        new_part_identifier = entry_order_id if is_scale_in else "initial"
        if new_part_identifier == "initial" and any(
            p["id"] == "initial" for p in _active_trade_parts
        ):
            logger.error(
                f"{NEON['ERROR']}Attempted to create a second 'initial' trade part. This should not happen. Aborting order logic.{NEON['RESET']}"
            )
            # Consider emergency close if a position was opened due to this logic error.
            return None

        # Recalculate SL price based on actual fill price
        actual_sl_price_raw = (
            (actual_fill_price - sl_distance)
            if side == CONFIG.side_buy
            else (actual_fill_price + sl_distance)
        )
        actual_sl_price_str = format_price(exchange, symbol, actual_sl_price_raw)

        _active_trade_parts.append(
            {
                "id": new_part_identifier,
                "entry_price": actual_fill_price,
                "entry_time_ms": int(entry_timestamp_ms),  # Ensure int
                "side": side,
                "qty": actual_filled_quantity,
                "sl_price": Decimal(actual_sl_price_str),  # Store precise SL
            }
        )

        # --- 8. Place Fixed Stop Loss (SL) Order ---
        sl_placed_successfully = False
        if Decimal(actual_sl_price_str) > 0:
            sl_order_side = (
                CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy
            )
            # Bybit V5: set SL via create_order with stopLoss param or modify_trading_stop
            # Using create_order with stopLoss for atomicity with the part, if possible
            # However, if tpslMode is "Full", it applies to whole position. "Partial" applies to this order qty.
            # For simplicity and current structure, placing a separate StopMarket order for SL.
            sl_params_v5 = {
                "category": v5_api_category,
                "stopPrice": actual_sl_price_str,  # Trigger price for SL
                "reduceOnly": True,
                "positionIdx": 0,
                # "tpslMode": "Partial", # If setting SL for this specific quantity
                # "slOrderType": "Market" # Ensure it's a market SL
            }
            try:
                logger.info(
                    f"Placing Fixed SL Ward: Side={sl_order_side}, Qty={_format_for_log(actual_filled_quantity, color=NEON['QTY'])}, TriggerAt={_format_for_log(actual_sl_price_str, color=NEON['PRICE'])}"
                )
                sl_order_response = safe_api_call(
                    exchange.create_order,
                    symbol,
                    "StopMarket",
                    sl_order_side,
                    float(actual_filled_quantity),
                    price=None,
                    params=sl_params_v5,
                )
                logger.success(
                    f"{NEON['SUCCESS']}Fixed SL Ward placed for part {new_part_identifier}. ID:...{format_order_id(sl_order_response.get('id'))}{NEON['RESET']}"
                )
                sl_placed_successfully = True
            except ccxt.InsufficientFunds as e_funds:
                logger.error(
                    f"{NEON['CRITICAL']}SL Placement Failed (Part {new_part_identifier}): Insufficient Funds! {e_funds}{NEON['RESET']}"
                )
            except ccxt.InvalidOrder as e_inv_ord:
                logger.error(
                    f"{NEON['CRITICAL']}SL Placement Failed (Part {new_part_identifier}): Invalid Order! {e_inv_ord}{NEON['RESET']}"
                )  # Often param issues
            except Exception as e_sl:
                logger.error(
                    f"{NEON['CRITICAL']}SL Placement Failed (Part {new_part_identifier}): {e_sl}{NEON['RESET']}"
                )
                logger.debug(traceback.format_exc())
        else:
            logger.error(
                f"{NEON['CRITICAL']}Invalid actual SL price ({_format_for_log(actual_sl_price_str)}) for part {new_part_identifier}! SL not placed.{NEON['RESET']}"
            )

        # --- 9. Optionally Place Trailing Stop Loss (TSL) Order ---
        if (
            not is_scale_in
            and CONFIG.trailing_stop_percentage > 0
            and actual_filled_quantity > CONFIG.position_qty_epsilon
        ):
            tsl_activation_offset_value = (
                actual_fill_price * CONFIG.trailing_stop_activation_offset_percent
            )
            tsl_activation_price_raw = (
                (actual_fill_price + tsl_activation_offset_value)
                if side == CONFIG.side_buy
                else (actual_fill_price - tsl_activation_offset_value)
            )
            tsl_activation_price_str = format_price(
                exchange, symbol, tsl_activation_price_raw
            )

            tsl_value_for_api_str = str(
                (CONFIG.trailing_stop_percentage * Decimal("100")).normalize()
            )  # e.g., "0.5" for 0.5%

            if Decimal(tsl_activation_price_str) > 0:
                tsl_params_specific_v5 = {
                    "category": v5_api_category,
                    "trailingStop": tsl_value_for_api_str,
                    "activePrice": tsl_activation_price_str,
                    # "tpslMode": "Full", # If TSL should apply to the whole position
                    "reduceOnly": True,
                    "positionIdx": 0,
                }
                try:
                    tsl_order_side = (
                        CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy
                    )
                    logger.info(
                        f"Placing Trailing SL Ward: Side={tsl_order_side}, Qty={_format_for_log(actual_filled_quantity, color=NEON['QTY'])}, Trail={NEON['VALUE']}{tsl_value_for_api_str}%{NEON['RESET']}, ActivateAt={NEON['PRICE']}{tsl_activation_price_str}{NEON['RESET']}"
                    )
                    tsl_order_response = safe_api_call(
                        exchange.create_order,
                        symbol,
                        "StopMarket",
                        tsl_order_side,
                        float(actual_filled_quantity),
                        price=None,
                        params=tsl_params_specific_v5,
                    )
                    logger.success(
                        f"{NEON['SUCCESS']}Trailing SL Ward placed. ID:...{format_order_id(tsl_order_response.get('id'))}{NEON['RESET']}"
                    )
                except Exception as e_tsl:
                    logger.warning(
                        f"{NEON['WARNING']}Failed to place Trailing SL Ward: {e_tsl}{NEON['RESET']}"
                    )
                    logger.debug(traceback.format_exc())
            else:
                logger.error(
                    f"{NEON['ERROR']}Invalid TSL activation price ({_format_for_log(tsl_activation_price_str)})! TSL not placed.{NEON['RESET']}"
                )

        # --- 10. Handle SL Placement Failure ---
        if not sl_placed_successfully:
            logger.critical(
                f"{NEON['CRITICAL']}CRITICAL: SL FAILED for {action_type} part {new_part_identifier}. Emergency close of entire position advised.{NEON['RESET']}"
            )
            # Attempt to close the position that was just opened
            close_position(
                exchange,
                symbol,
                {
                    "side": side,
                    "qty": actual_filled_quantity,
                    "entry_price": actual_fill_price,
                },
                reason=f"EMERGENCY CLOSE - SL FAIL ({action_type} part {new_part_identifier})",
            )
            # Remove the part that failed SL placement from _active_trade_parts
            _active_trade_parts = [
                p for p in _active_trade_parts if p["id"] != new_part_identifier
            ]
            save_persistent_state()
            return None

        save_persistent_state()  # Save state after successful order and SL/TSL placement
        logger.success(
            f"{NEON['SUCCESS']}{action_type} for {NEON['QTY']}{actual_filled_quantity}{NEON['SUCCESS']} {symbol} @ {NEON['PRICE']}{actual_fill_price}{NEON['SUCCESS']} successful. State saved.{NEON['RESET']}"
        )
        return filled_entry_details

    except ccxt.InsufficientFunds as e_funds:
        logger.error(
            f"{NEON['CRITICAL']}{action_type} Order Failed: Insufficient Funds! {e_funds}{NEON['RESET']}"
        )
    except ccxt.InvalidOrder as e_inv_ord:  # Often due to incorrect parameters, price/qty precision, or market state
        logger.error(
            f"{NEON['CRITICAL']}{action_type} Order Failed: Invalid Order! {e_inv_ord}{NEON['RESET']}"
        )
        logger.debug(
            f"Params sent: symbol={symbol}, side={side}, qty={order_quantity if 'order_quantity' in locals() else 'N/A'}, entry_type={entry_type.value}"
        )
    except Exception as e_ritual:
        logger.error(
            f"{NEON['CRITICAL']}{action_type} Ritual FAILED: {e_ritual}{NEON['RESET']}"
        )
        logger.debug(traceback.format_exc())
    return None


def close_partial_position(
    exchange: ccxt.Exchange,
    symbol: str,
    close_qty: Optional[Decimal] = None,
    reason: str = "Scale Out",
) -> Optional[Dict[str, Any]]:
    """Closes a part of the position, typically the oldest part for FIFO."""
    global _active_trade_parts
    if not _active_trade_parts:
        logger.info(
            f"{NEON['INFO']}No active trade parts to partially close for {symbol}.{NEON['RESET']}"
        )
        return None

    # Select the part to close (e.g., oldest for FIFO)
    oldest_part = min(_active_trade_parts, key=lambda p: p["entry_time_ms"])
    quantity_to_close = (
        close_qty if close_qty is not None and close_qty > 0 else oldest_part["qty"]
    )

    if quantity_to_close > oldest_part["qty"]:
        logger.warning(
            f"{NEON['WARNING']}Requested partial close quantity {NEON['QTY']}{close_qty}{NEON['WARNING']} > oldest part quantity {NEON['QTY']}{oldest_part['qty']}{NEON['WARNING']}. "
            f"Closing oldest part fully.{NEON['RESET']}"
        )
        quantity_to_close = oldest_part["qty"]

    position_side = oldest_part["side"]
    side_to_execute_close = (
        CONFIG.side_sell if position_side == CONFIG.pos_long else CONFIG.side_buy
    )
    amount_to_close_str = format_amount(exchange, symbol, quantity_to_close)

    logger.info(
        f"{NEON['ACTION']}Scaling Out: Closing {NEON['QTY']}{amount_to_close_str}{NEON['ACTION']} of {position_side} position "
        f"(Part ID: {oldest_part['id']}, Reason: {reason}).{NEON['RESET']}"
    )
    try:
        # Note: Managing SL for partial closes is complex.
        # Ideally, the SL order associated with this part should be cancelled or modified.
        # For simplicity here, we'll log a warning.
        logger.warning(
            f"{NEON['WARNING']}ACTION REQUIRED: Manually cancel/adjust SL for partially closed part ID {oldest_part['id']}. Automated SL adjustment for partial closes is complex.{NEON['RESET']}"
        )

        params = {"reduceOnly": True, "category": "linear", "positionIdx": 0}
        close_order_response = safe_api_call(
            exchange.create_market_order,
            symbol=symbol,
            side=side_to_execute_close,
            amount=float(amount_to_close_str),
            params=params,
        )

        if close_order_response and close_order_response.get("status") == "closed":
            exit_price = safe_decimal_conversion(close_order_response.get("average"))
            exit_time_ms = int(close_order_response.get("timestamp"))  # Ensure int

            mae, mfe = trade_metrics.calculate_mae_mfe(
                oldest_part["id"],
                oldest_part["entry_price"],
                exit_price,
                oldest_part["side"],
                oldest_part["entry_time_ms"],
                exit_time_ms,
                exchange,
                symbol,
                CONFIG.interval,
            )
            trade_metrics.log_trade(
                symbol,
                oldest_part["side"],
                oldest_part["entry_price"],
                exit_price,
                quantity_to_close,
                oldest_part["entry_time_ms"],
                exit_time_ms,
                reason,
                part_id=oldest_part["id"],
                mae=mae,
                mfe=mfe,
            )

            # Update or remove the part from _active_trade_parts
            if (
                abs(quantity_to_close - oldest_part["qty"])
                < CONFIG.position_qty_epsilon
            ):  # If entire part closed
                _active_trade_parts.remove(oldest_part)
            else:  # If part was partially closed
                oldest_part["qty"] -= quantity_to_close

            save_persistent_state()
            logger.success(
                f"{NEON['SUCCESS']}Scale Out successful for {NEON['QTY']}{amount_to_close_str}{NEON['SUCCESS']} {symbol}. State saved.{NEON['RESET']}"
            )
            return close_order_response
        else:
            logger.error(
                f"{NEON['ERROR']}Scale Out order failed for {symbol}. Response: {close_order_response}{NEON['RESET']}"
            )
    except Exception as e:
        logger.error(f"{NEON['ERROR']}Scale Out Ritual FAILED: {e}{NEON['RESET']}")
        logger.debug(traceback.format_exc())
    return None


def close_position(
    exchange: ccxt.Exchange,
    symbol: str,
    position_to_close_details: Dict[str, Any],
    reason: str = "Signal",
) -> Optional[Dict[str, Any]]:
    """Closes the entire consolidated position by closing all active parts."""
    global _active_trade_parts
    if not _active_trade_parts:
        logger.info(
            f"{NEON['INFO']}No active trade parts to close for {symbol}.{NEON['RESET']}"
        )
        return None

    total_quantity_to_close = sum(
        part.get("qty", Decimal(0)) for part in _active_trade_parts
    )
    if total_quantity_to_close <= CONFIG.position_qty_epsilon:
        logger.info(
            f"{NEON['INFO']}Total quantity of active parts is negligible. Clearing parts without exchange action.{NEON['RESET']}"
        )
        _active_trade_parts.clear()
        save_persistent_state()
        return None

    position_side_for_log = _active_trade_parts[0][
        "side"
    ]  # Assume all parts have same side
    side_to_execute_close = (
        CONFIG.side_sell
        if position_side_for_log == CONFIG.pos_long
        else CONFIG.side_buy
    )
    amount_to_close_str = format_amount(exchange, symbol, total_quantity_to_close)

    logger.info(
        f"{NEON['ACTION']}Closing ALL parts of {position_side_for_log} position for {symbol} "
        f"(Total Qty: {NEON['QTY']}{amount_to_close_str}{NEON['ACTION']}, Reason: {reason}).{NEON['RESET']}"
    )
    try:
        # Attempt to cancel open SL/TSL orders before closing the position
        cancelled_sl_count = cancel_open_orders(
            exchange, symbol, reason=f"Pre-Close Position ({reason})"
        )
        logger.info(
            f"{NEON['INFO']}Cancelled {NEON['VALUE']}{cancelled_sl_count}{NEON['INFO']} SL/TSL orders before closing position.{NEON['RESET']}"
        )
        time.sleep(0.5)  # Small delay for cancellations to process

        params = {"reduceOnly": True, "category": "linear", "positionIdx": 0}
        close_order_response = safe_api_call(
            exchange.create_market_order,
            symbol,
            side_to_execute_close,
            float(total_quantity_to_close),
            params=params,
        )

        if close_order_response and close_order_response.get("status") == "closed":
            exit_price = safe_decimal_conversion(close_order_response.get("average"))
            exit_time_ms = int(close_order_response.get("timestamp"))  # Ensure int

            # Log each part as closed
            for part in list(
                _active_trade_parts
            ):  # Iterate over a copy for safe removal
                mae, mfe = trade_metrics.calculate_mae_mfe(
                    part["id"],
                    part["entry_price"],
                    exit_price,
                    part["side"],
                    part["entry_time_ms"],
                    exit_time_ms,
                    exchange,
                    symbol,
                    CONFIG.interval,
                )
                trade_metrics.log_trade(
                    symbol,
                    part["side"],
                    part["entry_price"],
                    exit_price,
                    part["qty"],
                    part["entry_time_ms"],
                    exit_time_ms,
                    reason,
                    part_id=part["id"],
                    mae=mae,
                    mfe=mfe,
                )

            _active_trade_parts.clear()  # All parts are closed
            save_persistent_state()
            logger.success(
                f"{NEON['SUCCESS']}All parts of position for {symbol} closed successfully. State saved.{NEON['RESET']}"
            )
            # Removed manual SL cancellation warning as it's now attempted automatically
            return close_order_response
        else:
            logger.error(
                f"{NEON['ERROR']}Consolidated close order failed for {symbol}. Response: {close_order_response}{NEON['RESET']}"
            )
    except Exception as e:
        logger.error(f"{NEON['ERROR']}Close Position Ritual FAILED: {e}{NEON['RESET']}")
        logger.debug(traceback.format_exc())
    return None


def cancel_open_orders(
    exchange: ccxt.Exchange, symbol: str, reason: str = "Cleanup"
) -> int:
    """Cancels all open orders for the specified symbol."""
    logger.info(
        f"{NEON['INFO']}Attempting to cancel ALL open orders for {NEON['VALUE']}{symbol}{NEON['INFO']} (Reason: {reason})...{NEON['RESET']}"
    )
    cancelled_count = 0
    try:
        # Bybit V5 API requires category for fetching/cancelling orders
        params = {"category": "linear"}  # Assuming linear
        open_orders = safe_api_call(exchange.fetch_open_orders, symbol, params=params)

        if not open_orders:
            logger.info(f"No open orders found for {symbol} to cancel.")
            return 0

        logger.warning(
            f"{NEON['WARNING']}Found {NEON['VALUE']}{len(open_orders)}{NEON['WARNING']} open order(s) for {symbol}. Cancelling...{NEON['RESET']}"
        )
        for order in open_orders:
            order_id = order.get("id")
            if order_id:
                try:
                    safe_api_call(
                        exchange.cancel_order, order_id, symbol, params=params
                    )
                    logger.info(
                        f"Cancelled order {NEON['VALUE']}{order_id}{NEON['INFO']} for {symbol}."
                    )
                    cancelled_count += 1
                except ccxt.OrderNotFound:  # Order already filled or cancelled
                    logger.info(
                        f"Order {NEON['VALUE']}{order_id}{NEON['INFO']} already closed/cancelled."
                    )
                    cancelled_count += 1  # Count as handled
                except Exception as e_cancel:
                    logger.error(
                        f"{NEON['ERROR']}Failed to cancel order {order_id}: {e_cancel}{NEON['RESET']}"
                    )
            else:
                logger.error(
                    f"{NEON['ERROR']}Found an open order without an ID for {symbol}. Cannot cancel.{NEON['RESET']}"
                )
        logger.info(
            f"Order cancellation process for {symbol} complete. Cancelled/Handled: {NEON['VALUE']}{cancelled_count}{NEON['RESET']}."
        )
    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}Error fetching/cancelling open orders for {symbol}: {e}{NEON['RESET']}"
        )
        logger.debug(traceback.format_exc())
    return cancelled_count


# --- Strategy Signal Generation Wrapper ---
def generate_strategy_signals(
    df_with_indicators: pd.DataFrame, strategy_instance: TradingStrategy
) -> Dict[str, Any]:
    """Wrapper to call the `generate_signals` method of the active strategy."""
    if strategy_instance:
        return strategy_instance.generate_signals(df_with_indicators)
    logger.error(
        f"{NEON['ERROR']}Strategy instance not initialized! Cannot generate signals.{NEON['RESET']}"
    )

    # Create a dummy base strategy instance to get default signals
    # This requires CONFIG to be globally available for the dummy TradingStrategy constructor
    class DummyStrategy(
        TradingStrategy
    ):  # Define a minimal concrete class for instantiation
        def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
            return self._get_default_signals()

    return DummyStrategy(CONFIG)._get_default_signals()


# --- Trading Logic - The Core Spell Weaving ---
_stop_trading_flag = False  # Global flag to halt trading (e.g., due to max drawdown)
_last_drawdown_check_time = 0  # Timestamp of the last drawdown check


def trade_logic(
    exchange: ccxt.Exchange, symbol: str, market_data_df: pd.DataFrame
) -> None:
    """Main trading logic function executed each cycle."""
    global _active_trade_parts, _stop_trading_flag, _last_drawdown_check_time

    cycle_time_str = (
        market_data_df.index[-1].strftime("%Y-%m-%d %H:%M:%S %Z")
        if not market_data_df.empty
        else "N/A"
    )
    logger.info(
        f"{NEON['SUBHEADING']}========== Pyrmethus Cycle Start v2.8.0 ({CONFIG.strategy_name.value}) for '{symbol}' | Candle: {cycle_time_str} =========={NEON['RESET']}"
    )

    now_timestamp = time.time()

    if _stop_trading_flag:
        logger.critical(
            f"{NEON['CRITICAL']}STOP TRADING FLAG IS ACTIVE (likely due to max drawdown). No new trades will be initiated.{NEON['RESET']}"
        )
        return

    if market_data_df.empty:
        logger.warning(
            f"{NEON['WARNING']}Market data DataFrame is empty. Skipping trade logic for this cycle.{NEON['RESET']}"
        )
        return

    # --- Max Drawdown Check ---
    if CONFIG.enable_max_drawdown_stop and (
        now_timestamp - _last_drawdown_check_time > 300
    ):  # Check every 5 mins
        try:
            balance = safe_api_call(
                exchange.fetch_balance, params={"category": "linear"}
            )  # V5 specific
            current_equity = safe_decimal_conversion(
                balance.get(CONFIG.usdt_symbol, {}).get("total")
            )
            if not pd.isna(current_equity):  # Ensure equity is valid
                trade_metrics.set_initial_equity(
                    current_equity
                )  # Updates daily start if new day
                breached, reason = trade_metrics.check_drawdown(current_equity)
                if breached:
                    _stop_trading_flag = True
                    logger.critical(
                        f"{NEON['CRITICAL']}MAX DRAWDOWN LIMIT REACHED: {reason}. Halting all new trading activities!{NEON['RESET']}"
                    )
                    send_sms_alert(
                        f"[Pyrmethus] CRITICAL: Max Drawdown STOP Activated: {reason}"
                    )
                    return  # Stop further logic in this cycle
            _last_drawdown_check_time = now_timestamp
        except Exception as e_dd_check:
            logger.error(
                f"{NEON['ERROR']}Error during drawdown check: {e_dd_check}{NEON['RESET']}"
            )

    # --- Calculate Indicators and Get Current Market State ---
    df_with_indicators, current_vol_atr_data = calculate_all_indicators(
        market_data_df.copy(), CONFIG
    )  # Use a copy for indicators
    current_atr_short = current_vol_atr_data.get(
        "atr_short", Decimal("0")
    )  # Default to 0 if not found
    current_close_price = safe_decimal_conversion(
        df_with_indicators["close"].iloc[-1], pd.NA
    )

    if (
        pd.isna(current_atr_short)
        or current_atr_short <= 0
        or pd.isna(current_close_price)
        or current_close_price <= 0
    ):
        logger.warning(
            f"{NEON['WARNING']}Invalid ATR ({_format_for_log(current_atr_short)}) or Close Price ({_format_for_log(current_close_price)}). Skipping logic.{NEON['RESET']}"
        )
        return

    # --- Get Current Position and Strategy Signals ---
    current_position = get_current_position(exchange, symbol)
    pos_side, total_pos_qty, avg_pos_entry_price = (
        current_position["side"],
        current_position["qty"],
        current_position["entry_price"],
    )
    num_active_parts = current_position.get("num_parts", 0)

    strategy_signals = generate_strategy_signals(
        df_with_indicators, CONFIG.strategy_instance
    )

    # --- Time-Based Stop ---
    if CONFIG.enable_time_based_stop and pos_side != CONFIG.pos_none:
        now_ms = int(now_timestamp * 1000)
        for part in list(_active_trade_parts):  # Iterate over a copy if modifying list
            duration_in_ms = now_ms - part["entry_time_ms"]
            if duration_in_ms > CONFIG.max_trade_duration_seconds * 1000:
                reason = f"Time Stop Hit ({duration_in_ms / 1000:.0f}s > {CONFIG.max_trade_duration_seconds}s)"
                logger.warning(
                    f"{NEON['WARNING']}TIME STOP for part {part['id']} ({pos_side}). Closing entire position.{NEON['RESET']}"
                )
                close_position(exchange, symbol, current_position, reason=reason)
                return  # Exit after closing

    # --- Scale Out Logic ---
    if CONFIG.enable_scale_out and pos_side != CONFIG.pos_none and num_active_parts > 0:
        profit_in_atr = Decimal("0")
        if (
            current_atr_short > 0 and avg_pos_entry_price > 0
        ):  # Ensure valid ATR and entry price
            price_difference = (
                (current_close_price - avg_pos_entry_price)
                if pos_side == CONFIG.pos_long
                else (avg_pos_entry_price - current_close_price)
            )
            profit_in_atr = price_difference / current_atr_short

        if profit_in_atr >= CONFIG.scale_out_trigger_atr:
            logger.info(
                f"{NEON['ACTION']}SCALE-OUT Triggered: Position is {NEON['VALUE']}{profit_in_atr:.2f}{NEON['ACTION']} ATRs in profit. Closing oldest part.{NEON['RESET']}"
            )
            close_partial_position(
                exchange,
                symbol,
                close_qty=None,
                reason=f"Scale Out Profit Target ({profit_in_atr:.2f} ATR)",
            )
            # Re-fetch position state after partial close
            current_position = get_current_position(exchange, symbol)
            pos_side, total_pos_qty, avg_pos_entry_price = (
                current_position["side"],
                current_position["qty"],
                current_position["entry_price"],
            )
            num_active_parts = current_position.get("num_parts", 0)
            if pos_side == CONFIG.pos_none:
                return  # If position fully closed by scale-out

    # --- Strategy Exit Logic ---
    should_exit_long = pos_side == CONFIG.pos_long and strategy_signals.get(
        "exit_long", False
    )
    should_exit_short = pos_side == CONFIG.pos_short and strategy_signals.get(
        "exit_short", False
    )
    if should_exit_long or should_exit_short:
        exit_reason = strategy_signals.get("exit_reason", "Oracle Decrees Exit")
        logger.warning(
            f"{NEON['ACTION']}*** STRATEGY EXIT for remaining {pos_side} position (Reason: {exit_reason}) ***{NEON['RESET']}"
        )
        close_position(exchange, symbol, current_position, reason=exit_reason)
        return  # Exit after closing

    # --- Scale-In (Pyramiding) Logic ---
    if (
        CONFIG.enable_position_scaling
        and pos_side != CONFIG.pos_none
        and num_active_parts < (CONFIG.max_scale_ins + 1)
    ):
        profit_in_atr = Decimal("0")
        if current_atr_short > 0 and avg_pos_entry_price > 0:
            price_difference = (
                (current_close_price - avg_pos_entry_price)
                if pos_side == CONFIG.pos_long
                else (avg_pos_entry_price - current_close_price)
            )
            profit_in_atr = price_difference / current_atr_short

        can_scale_in_based_on_profit = (
            profit_in_atr >= CONFIG.min_profit_for_scale_in_atr
        )

        # Check if new strategy signal aligns with current position direction
        scale_in_long_signal = (
            strategy_signals.get("enter_long", False) and pos_side == CONFIG.pos_long
        )
        scale_in_short_signal = (
            strategy_signals.get("enter_short", False) and pos_side == CONFIG.pos_short
        )

        if can_scale_in_based_on_profit and (
            scale_in_long_signal or scale_in_short_signal
        ):
            logger.success(
                f"{NEON['ACTION']}*** PYRAMIDING OPPORTUNITY: New signal to add to existing {pos_side} position. ***{NEON['RESET']}"
            )
            scale_in_side_to_enter = (
                CONFIG.side_buy if scale_in_long_signal else CONFIG.side_sell
            )
            place_risked_order(
                exchange=exchange,
                symbol=symbol,
                side=scale_in_side_to_enter,
                risk_percentage=CONFIG.scale_in_risk_percentage,  # Use specific risk for scale-in
                current_short_atr=current_atr_short,
                leverage=CONFIG.leverage,
                max_order_cap_usdt=CONFIG.max_order_usdt_amount,
                margin_check_buffer=CONFIG.required_margin_buffer,
                tsl_percent=CONFIG.trailing_stop_percentage,  # TSL might be for whole position
                tsl_activation_offset_percent=CONFIG.trailing_stop_activation_offset_percent,
                entry_type=CONFIG.entry_order_type,
                is_scale_in=True,
                existing_position_avg_price=avg_pos_entry_price,
            )
            return  # End cycle after attempting scale-in

    # --- Initial Entry Logic ---
    if pos_side == CONFIG.pos_none:  # Only consider new entries if flat
        enter_long_signal = strategy_signals.get("enter_long", False)
        enter_short_signal = strategy_signals.get("enter_short", False)

        if enter_long_signal or enter_short_signal:
            side_to_enter = CONFIG.side_buy if enter_long_signal else CONFIG.side_sell
            entry_color = NEON["SIDE_LONG"] if enter_long_signal else NEON["SIDE_SHORT"]
            logger.success(
                f"{entry_color}*** INITIAL {side_to_enter.upper()} ENTRY SIGNAL ({CONFIG.strategy_name.value}) ***{NEON['RESET']}"
            )
            # Cancel any lingering orders before new entry (e.g., from previous failed attempts)
            cancel_open_orders(exchange, symbol, reason="Pre-Initial Entry Cleanup")
            time.sleep(0.5)  # Small delay for cancellations
            place_risked_order(
                exchange=exchange,
                symbol=symbol,
                side=side_to_enter,
                risk_percentage=calculate_dynamic_risk(),  # Use dynamic risk for initial entry
                current_short_atr=current_atr_short,
                leverage=CONFIG.leverage,
                max_order_cap_usdt=CONFIG.max_order_usdt_amount,
                margin_check_buffer=CONFIG.required_margin_buffer,
                tsl_percent=CONFIG.trailing_stop_percentage,
                tsl_activation_offset_percent=CONFIG.trailing_stop_activation_offset_percent,
                entry_type=CONFIG.entry_order_type,
                is_scale_in=False,
            )  # This is an initial entry
            return  # End cycle after attempting entry

    # --- If Holding or No Action ---
    if pos_side != CONFIG.pos_none:
        pos_color = (
            NEON["SIDE_LONG"] if pos_side == CONFIG.pos_long else NEON["SIDE_SHORT"]
        )
        logger.info(
            f"{NEON['INFO']}Holding {pos_color}{pos_side}{NEON['INFO']} position ({NEON['VALUE']}{num_active_parts}{NEON['INFO']} parts). Awaiting signals or stops.{NEON['RESET']}"
        )
    else:
        logger.info(
            f"{NEON['INFO']}Holding Cash ({NEON['SIDE_FLAT']}Flat{NEON['INFO']}). No entry signals or conditions met this cycle.{NEON['RESET']}"
        )

    save_persistent_state(
        force_heartbeat=True
    )  # Heartbeat save if no other major action triggered save
    logger.info(
        f"{NEON['SUBHEADING']}========== Pyrmethus Cycle End v2.8.0 for '{symbol}' =========={NEON['RESET']}\n"
    )


# --- Graceful Shutdown - Withdrawing the Arcane Energies ---
def graceful_shutdown(
    exchange_instance: Optional[ccxt.Exchange], trading_symbol: Optional[str]
) -> None:
    """Handles graceful shutdown: saves state, cancels orders, closes positions."""
    logger.warning(
        f"\n{NEON['WARNING']}{Style.BRIGHT}Unweaving Sequence Initiated for Pyrmethus v2.8.0...{NEON['RESET']}"
    )

    save_persistent_state(force_heartbeat=True)  # Final state save

    if exchange_instance and trading_symbol:
        try:
            logger.warning(
                f"Unweaving: Cancelling ALL open orders for '{trading_symbol}' as part of shutdown..."
            )
            cancel_open_orders(
                exchange_instance, trading_symbol, "Bot Shutdown Cleanup"
            )
            time.sleep(1.5)  # Allow time for cancellations

            # Check current position again before attempting to close
            # Use _get_raw_exchange_position to be sure, as _active_trade_parts might be stale if save failed
            current_pos_on_exchange = _get_raw_exchange_position(
                exchange_instance, trading_symbol
            )
            if (
                current_pos_on_exchange["side"] != CONFIG.pos_none
                and current_pos_on_exchange["qty"] > CONFIG.position_qty_epsilon
            ):
                logger.warning(
                    f"{NEON['WARNING']}Unweaving: Active {current_pos_on_exchange['side']} position found on exchange. Attempting final consolidated close...{NEON['RESET']}"
                )
                # Use the position details fetched directly from exchange for closing
                close_position(
                    exchange_instance,
                    trading_symbol,
                    current_pos_on_exchange,
                    "Bot Shutdown Final Close",
                )
            elif (
                _active_trade_parts
            ):  # If bot *thinks* it has parts but exchange is flat
                logger.warning(
                    f"{NEON['WARNING']}Unweaving: Bot remembers active parts, but exchange reports FLAT. Clearing bot state.{NEON['RESET']}"
                )
                _active_trade_parts.clear()
                save_persistent_state(force_heartbeat=True)
            else:
                logger.info(
                    f"{NEON['INFO']}Unweaving: No active position found on exchange or in bot state for '{trading_symbol}'.{NEON['RESET']}"
                )

        except Exception as e_cleanup:
            logger.error(
                f"{NEON['ERROR']}Error during shutdown cleanup for {trading_symbol}: {e_cleanup}{NEON['RESET']}"
            )
            logger.debug(traceback.format_exc())
    else:
        logger.warning(
            f"{NEON['WARNING']}Unweaving: Exchange instance or symbol not available. Skipping exchange cleanup.{NEON['RESET']}"
        )

    trade_metrics.summary()  # Log final trade metrics

    logger.info(
        f"{NEON['HEADING']}--- Pyrmethus Spell Unweaving v2.8.0 Complete ---{NEON['RESET']}"
    )
    send_sms_alert(
        f"[Pyrmethus/{CONFIG.strategy_name.value if 'CONFIG' in globals() else 'N/A'}] Shutdown sequence complete."
    )


# --- Data Fetching (Moved here for better organization before main) ---
_last_market_data_fetch_ts: Dict[
    str, float
] = {}  # Cache timestamp per symbol_timeframe
_market_data_cache: Dict[str, pd.DataFrame] = {}  # Cache DataFrame per symbol_timeframe


def get_market_data(
    exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 150
) -> Optional[pd.DataFrame]:
    """Fetches OHLCV market data and returns a pandas DataFrame, with caching."""
    global _market_data_cache, _last_market_data_fetch_ts
    cache_key = f"{symbol}_{timeframe}"
    current_time_seconds = time.time()

    # Check cache
    if cache_key in _market_data_cache and cache_key in _last_market_data_fetch_ts:
        try:
            candle_duration_seconds = exchange.parse_timeframe(timeframe)  # in seconds
            # If current time is within the same candle as the last fetch (with a buffer)
            if (current_time_seconds - _last_market_data_fetch_ts[cache_key]) < (
                candle_duration_seconds * float(CONFIG.cache_candle_duration_multiplier)
            ):
                # And if cached data has enough rows
                if len(_market_data_cache[cache_key]) >= limit:
                    logger.debug(
                        f"Data Fetch: Using CACHED market data for {NEON['VALUE']}{cache_key}{NEON['DEBUG']} ({len(_market_data_cache[cache_key])} candles).{NEON['RESET']}"
                    )
                    return _market_data_cache[
                        cache_key
                    ].copy()  # Return a copy to prevent modification
                else:
                    logger.debug(
                        f"Cache for {cache_key} has insufficient rows ({len(_market_data_cache[cache_key])} vs {limit}). Fetching fresh."
                    )
            else:
                logger.debug(f"Cache for {cache_key} expired. Fetching fresh.")
        except (
            Exception
        ) as e_parse_tf:  # Handle if parse_timeframe fails for some reason
            logger.warning(
                f"Could not parse timeframe '{timeframe}' for caching: {e_parse_tf}. Cache validation skipped for this call."
            )

    logger.info(
        f"{NEON['INFO']}Fetching market data for {NEON['VALUE']}{symbol}{NEON['INFO']} ({timeframe}, limit={limit})...{NEON['RESET']}"
    )
    try:
        # Bybit V5 API often requires 'category' for linear contracts
        params = {"category": "linear"}  # Assuming linear, adjust if supporting inverse
        ohlcv = safe_api_call(
            exchange.fetch_ohlcv, symbol, timeframe, limit=limit, params=params
        )
        if not ohlcv:
            logger.warning(
                f"{NEON['WARNING']}No OHLCV data returned for {symbol}. Market might be inactive or API issue.{NEON['RESET']}"
            )
            return None

        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)

        # Ensure numeric types and handle potential NaNs from API
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Fill NaNs - simple forward fill then backward fill
        df.ffill(inplace=True)
        df.bfill(inplace=True)  # For NaNs at the beginning

        if df.isnull().values.any():
            logger.error(
                f"{NEON['ERROR']}Unfillable NaNs remain in OHLCV data for {symbol} after ffill/bfill. Data quality compromised.{NEON['RESET']}"
            )
            # If critical columns like 'close' are still NaN at the end, it's problematic
            if df[["close"]].iloc[-1].isnull().any():
                logger.critical(
                    f"{NEON['CRITICAL']}Last 'close' price is NaN for {symbol}. Cannot proceed with this data.{NEON['RESET']}"
                )
                return None  # Or handle more gracefully depending on strategy needs

        _market_data_cache[cache_key] = df.copy()
        _last_market_data_fetch_ts[cache_key] = current_time_seconds
        logger.debug(
            f"{NEON['DEBUG']}Fetched and cached {len(df)} candles for {symbol}. Last candle: {df.index[-1]}{NEON['RESET']}"
        )
        return df.copy()  # Return a copy
    except Exception as e:
        logger.error(
            f"{NEON['ERROR']}Failed to fetch or process market data for {symbol}: {e}{NEON['RESET']}"
        )
        logger.debug(traceback.format_exc())
        return None


# --- Main Execution - Igniting the Spell ---
def main() -> None:
    start_time_readable = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    logger.info(
        f"{NEON['HEADING']}--- Pyrmethus Scalping Spell v2.8.0 (Strategic Illumination) Initializing ({start_time_readable}) ---{NEON['RESET']}"
    )
    logger.info(
        f"{NEON['SUBHEADING']}--- Active Strategy Path: {NEON['VALUE']}{CONFIG.strategy_name.value}{NEON['RESET']} ---"
    )
    logger.info(
        f"Symbol: {NEON['VALUE']}{CONFIG.symbol}{NEON['RESET']}, Interval: {NEON['VALUE']}{CONFIG.interval}{NEON['RESET']}, Leverage: {NEON['VALUE']}{CONFIG.leverage}x{NEON['RESET']}"
    )
    logger.info(
        f"Risk/Trade (Base): {NEON['VALUE']}{CONFIG.risk_per_trade_percentage:.2%}{NEON['RESET']}, Max Order Cap (USDT): {NEON['VALUE']}{CONFIG.max_order_usdt_amount}{NEON['RESET']}"
    )
    logger.info(
        f"Dynamic Risk: {NEON['VALUE']}{CONFIG.enable_dynamic_risk}{NEON['RESET']}, Dynamic SL: {NEON['VALUE']}{CONFIG.enable_dynamic_atr_sl}{NEON['RESET']}, Pyramiding: {NEON['VALUE']}{CONFIG.enable_position_scaling}{NEON['RESET']}"
    )

    current_exchange_instance: Optional[ccxt.Exchange] = None
    unified_trading_symbol: Optional[str] = None
    should_run_bot: bool = True
    cycle_count: int = 0

    try:
        current_exchange_instance = initialize_exchange()
        if not current_exchange_instance:
            logger.critical(
                f"{NEON['CRITICAL']}Exchange portal failed to open. Pyrmethus cannot weave magic. Exiting.{NEON['RESET']}"
            )
            return

        try:
            market_details = current_exchange_instance.market(CONFIG.symbol)
            unified_trading_symbol = market_details["symbol"]
            if not market_details.get(
                "contract"
            ):  # Check if it's a futures/contract market
                logger.critical(
                    f"{NEON['CRITICAL']}Market '{unified_trading_symbol}' is not a contract/futures market. Pyrmethus targets futures.{NEON['RESET']}"
                )
                return
        except Exception as e_market:
            logger.critical(
                f"{NEON['CRITICAL']}Symbol validation error for '{CONFIG.symbol}': {e_market}. Exiting.{NEON['RESET']}"
            )
            return

        logger.info(
            f"{NEON['SUCCESS']}Spell focused on symbol: {NEON['VALUE']}{unified_trading_symbol}{NEON['SUCCESS']} (Type: {market_details.get('type', 'N/A')}){NEON['RESET']}"
        )

        if not set_leverage(
            current_exchange_instance, unified_trading_symbol, CONFIG.leverage
        ):
            logger.warning(
                f"{NEON['WARNING']}Leverage setting for {unified_trading_symbol} may not have been applied or confirmed. Proceeding with caution.{NEON['RESET']}"
            )

        if load_persistent_state():
            logger.info(
                f"{NEON['SUCCESS']}Phoenix Feather: Previous session state successfully restored.{NEON['RESET']}"
            )
            if (
                _active_trade_parts
            ):  # If bot remembers active parts, verify with exchange
                logger.warning(
                    f"{NEON['WARNING']}State Reconciliation Check:{NEON['RESET']} Bot remembers {NEON['VALUE']}{len(_active_trade_parts)}{NEON['WARNING']} active trade part(s). Verifying with exchange...{NEON['RESET']}"
                )
                exchange_pos = _get_raw_exchange_position(
                    current_exchange_instance, unified_trading_symbol
                )
                bot_qty = sum(p["qty"] for p in _active_trade_parts)
                bot_side = (
                    _active_trade_parts[0]["side"]
                    if _active_trade_parts
                    else CONFIG.pos_none
                )

                # Reconciliation logic: if discrepancy, trust exchange and clear bot state
                if (
                    exchange_pos["side"] == CONFIG.pos_none
                    and bot_side != CONFIG.pos_none
                ):
                    logger.critical(
                        f"{NEON['CRITICAL']}RECONCILIATION FAILED:{NEON['RESET']} Bot remembers {bot_side} (Qty: {bot_qty}), but exchange is FLAT. Clearing bot state to match exchange."
                    )
                    _active_trade_parts.clear()
                    save_persistent_state()
                elif exchange_pos["side"] != bot_side or abs(
                    exchange_pos["qty"] - bot_qty
                ) > CONFIG.position_qty_epsilon * Decimal(
                    "10"
                ):  # Use larger epsilon for sum of parts
                    logger.critical(
                        f"{NEON['CRITICAL']}RECONCILIATION FAILED:{NEON['RESET']} Discrepancy found: Bot ({bot_side} Qty {bot_qty}) vs Exchange ({exchange_pos['side']} Qty {exchange_pos['qty']}). Clearing bot state to match exchange."
                    )
                    _active_trade_parts.clear()
                    save_persistent_state()
                else:
                    logger.info(
                        f"{NEON['SUCCESS']}State Reconciliation: Bot's active trade parts are consistent with exchange position.{NEON['RESET']}"
                    )
        else:
            logger.info(
                f"{NEON['INFO']}Starting with a fresh session state. No previous memories reawakened.{NEON['RESET']}"
            )

        # Set initial equity for drawdown tracking
        try:
            balance = safe_api_call(
                current_exchange_instance.fetch_balance, params={"category": "linear"}
            )  # V5 specific
            initial_equity = safe_decimal_conversion(
                balance.get(CONFIG.usdt_symbol, {}).get("total")
            )
            if not pd.isna(initial_equity):  # Ensure equity is valid
                trade_metrics.set_initial_equity(initial_equity)
            else:
                logger.error(
                    f"{NEON['ERROR']}Failed to get valid initial equity for drawdown tracking. Drawdown protection may be impaired.{NEON['RESET']}"
                )
        except Exception as e_bal_init:
            logger.error(
                f"{NEON['ERROR']}Failed to set initial equity due to API error: {e_bal_init}{NEON['RESET']}"
            )

        market_base_name = unified_trading_symbol.split("/")[0].split(":")[
            0
        ]  # e.g., BTC from BTC/USDT:USDT
        send_sms_alert(
            f"[{market_base_name}/{CONFIG.strategy_name.value}] Pyrmethus v2.8.0 Initialized & Weaving. Symbol: {unified_trading_symbol}."
        )

        # --- Main Trading Loop ---
        while should_run_bot:
            cycle_start_monotonic = time.monotonic()
            cycle_count += 1
            logger.debug(
                f"{NEON['DEBUG']}--- Cycle {cycle_count} Start ({time.strftime('%H:%M:%S')}) ---{NEON['RESET']}"
            )

            # Health check (e.g., connectivity, API status)
            try:
                # A simple health check: fetch balance again.
                if not current_exchange_instance.fetch_balance(
                    params={"category": "linear"}
                ):  # V5 specific
                    raise Exception(
                        "Exchange health check (fetch_balance) returned falsy"
                    )
            except Exception as e_health:
                logger.critical(
                    f"{NEON['CRITICAL']}Account/Exchange health check failed: {e_health}. Pausing and will retry.{NEON['RESET']}"
                )
                time.sleep(CONFIG.sleep_seconds * 10)  # Longer pause for health issues
                continue

            try:
                # Determine data limit based on longest indicator lookback
                indicator_max_lookback = max(
                    CONFIG.st_atr_length,
                    CONFIG.confirm_st_atr_length,
                    CONFIG.momentum_period,
                    CONFIG.ehlers_fisher_length,
                    CONFIG.ehlers_fisher_signal_length,  # Added Ehlers
                    CONFIG.atr_short_term_period,
                    CONFIG.atr_long_term_period,
                    CONFIG.volume_ma_period,
                    50,
                )  # General minimum
                data_limit_needed = (
                    indicator_max_lookback + CONFIG.api_fetch_limit_buffer + 20
                )  # Buffer for NaNs and stability

                df_market_candles = get_market_data(
                    current_exchange_instance,
                    unified_trading_symbol,
                    CONFIG.interval,
                    limit=data_limit_needed,
                )

                if df_market_candles is not None and not df_market_candles.empty:
                    trade_logic(
                        current_exchange_instance,
                        unified_trading_symbol,
                        df_market_candles,
                    )
                else:
                    logger.warning(
                        f"{NEON['WARNING']}Skipping trade logic this cycle: invalid or missing market data for {unified_trading_symbol}.{NEON['RESET']}"
                    )

            # Handle specific CCXT exceptions
            except ccxt.RateLimitExceeded as e_rate:
                logger.warning(
                    f"{NEON['WARNING']}Rate Limit Exceeded: {e_rate}. Sleeping longer...{NEON['RESET']}"
                )
                time.sleep(CONFIG.sleep_seconds * 6)
            except (
                ccxt.NetworkError,
                ccxt.ExchangeNotAvailable,
                ccxt.RequestTimeout,
            ) as e_net:
                logger.warning(
                    f"{NEON['WARNING']}Network/Exchange Issue: {e_net}. Sleeping and will retry.{NEON['RESET']}"
                )
                sleep_mult = 6 if isinstance(e_net, ccxt.ExchangeNotAvailable) else 3
                time.sleep(CONFIG.sleep_seconds * sleep_mult)
            except ccxt.AuthenticationError as e_auth:
                logger.critical(
                    f"{NEON['CRITICAL']}FATAL: Authentication Error: {e_auth}. Pyrmethus cannot continue. Stopping.{NEON['RESET']}"
                )
                send_sms_alert(
                    f"[{market_base_name}/{CONFIG.strategy_name.value}] CRITICAL: Auth Error! Bot stopping."
                )
                should_run_bot = False
            except Exception as e_loop:
                logger.exception(
                    f"{NEON['CRITICAL']}!!! UNEXPECTED CRITICAL ERROR in Main Loop: {e_loop} !!! Pyrmethus is stopping!{NEON['RESET']}"
                )
                send_sms_alert(
                    f"[{market_base_name}/{CONFIG.strategy_name.value}] CRITICAL UNEXPECTED ERROR: {type(e_loop).__name__}! Bot stopping."
                )
                should_run_bot = False

            # Calculate sleep duration for the end of the cycle
            if should_run_bot:
                elapsed_cycle_time = time.monotonic() - cycle_start_monotonic
                sleep_duration = max(0, CONFIG.sleep_seconds - elapsed_cycle_time)
                logger.debug(
                    f"Cycle {cycle_count} processed in {elapsed_cycle_time:.2f}s. Sleeping for {sleep_duration:.2f}s."
                )
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

    except KeyboardInterrupt:
        logger.warning(
            f"\n{NEON['WARNING']}{Style.BRIGHT}KeyboardInterrupt detected. Initiating graceful unweaving...{NEON['RESET']}"
        )
        should_run_bot = False
    except (
        Exception
    ) as startup_err:  # Catch errors during initial setup before main loop
        logger.critical(
            f"{NEON['CRITICAL']}CRITICAL STARTUP ERROR (Pyrmethus v2.8.0): {startup_err}{NEON['RESET']}"
        )
        logger.debug(traceback.format_exc())  # Log full traceback
        if (
            "CONFIG" in globals()
            and hasattr(CONFIG, "enable_sms_alerts")
            and CONFIG.enable_sms_alerts
        ):  # Check if CONFIG is available
            send_sms_alert(
                f"[Pyrmethus] CRITICAL STARTUP ERROR: {type(startup_err).__name__}."
            )
        should_run_bot = False  # Ensure bot doesn't run
    finally:
        graceful_shutdown(current_exchange_instance, unified_trading_symbol)
        logger.info(
            f"{NEON['HEADING']}--- Pyrmethus Scalping Spell v2.8.0 Deactivated ---{NEON['RESET']}"
        )


if __name__ == "__main__":
    main()
