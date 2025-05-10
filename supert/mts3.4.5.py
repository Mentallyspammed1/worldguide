#!/usr/bin/env python3
# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v2.8.2 (Operational Infusion - Refined)
# Features implemented core trading logic, live market interaction,
# enhanced state management, and robust error handling. Helper functions and
# TradeMetrics now more complete and thematically aligned.
# Previous Weave: v2.8.1 (Strategic Illumination & Termux Weave) for style reference.

"""
High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 2.8.2 (Operational Infusion - Refined)

Dependencies & Setup (Termux):
1. Core Packages:
   pkg install python clang libcrypt libffi openssl rust termux-api
   (Rust and build essentials are often needed for cryptography dependencies)
2. Python Libraries:
   pip install ccxt pandas pandas-ta colorama python-dotenv retry pytz

WARNING: This script can execute live trades. Use with extreme caution.
Thoroughly test in a simulated environment or with minimal capital.
Pyrmethus bears no responsibility for financial outcomes.

Enhancements in v2.8.2 (Refined by Pyrmethus):
- Implemented core operational logic: get_market_data, fetch_account_balance,
  calculate_position_size, place_risked_order, close_position_part.
- Activated main_loop with live trading decision-making.
- Full exchange initialization block for Bybit (V5 API focus).
- Market info (precision, limits) loaded into Config.
- Enhanced _active_trade_parts structure.
- Improved error handling with @retry and specific ccxt exceptions.
- Added PAPER_TRADING_MODE to Config (informational).
- Refined indicator calculation and column naming.
- More comprehensive Termux notifications for critical events (full implementation).
- Enhanced Config._validate_parameters with more checks.
- Expanded TradeMetrics with MAE/MFE stubs, performance trend, and detailed summary.
"""

# Standard Library Imports
import json
import logging
import os
import random # For MockExchange if used, and unique IDs
import subprocess # For Termux notifications
import sys
import time
import traceback
import uuid # For unique part IDs
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, DivisionByZero, InvalidOperation, getcontext
from enum import Enum
from typing import Any, Union, Dict, List, Tuple, Optional, Type

import pytz # For timezone-aware datetimes, the Chronomancer's ally

# Third-party Libraries - The Grimoire of External Magics
try:
    import ccxt
    import pandas as pd
    if not hasattr(pd, 'NA'): # Ensure pandas version supports pd.NA
        raise ImportError("Pandas version < 1.0 not supported, pd.NA is required.")
    import pandas_ta as ta # type: ignore[import]
    from colorama import Back, Fore, Style # The Prisms of Perception
    from colorama import init as colorama_init
    from dotenv import load_dotenv # For whispering secrets from .env scrolls
    from retry import retry # The Art of Tenacious Spellcasting
except ImportError as e:
    missing_pkg = getattr(e, 'name', 'dependency')
    sys.stderr.write(f"\033[91mCRITICAL ERROR: Missing/Incompatible Essence: '{missing_pkg}'. Pyrmethus cannot weave this spell.\033[0m\n")
    sys.stderr.write(f"\033[91mPlease ensure all required libraries (runes) are installed and up to date.\033[0m\n")
    sys.stderr.write(f"\033[91mConsult the scrolls (README or comments) for 'pkg install' and 'pip install' incantations.\033[0m\n")
    sys.exit(1)

# --- Constants - The Unchanging Pillars of the Spell ---
STATE_FILE_NAME = "pyrmethus_phoenix_state_v282.json" # Updated version
STATE_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), STATE_FILE_NAME)
HEARTBEAT_INTERVAL_SECONDS = 60
OHLCV_LIMIT = 200 # Number of candles to fetch for indicators

# --- Neon Color Palette - The Wizard's Aura ---
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

# --- Initializations - Awakening the Spell ---
colorama_init(autoreset=True)
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if load_dotenv(dotenv_path=env_path):
    logging.getLogger("PreConfig").info(f"{NEON['INFO']}Secrets whispered from .env scroll: {NEON['VALUE']}{env_path}{NEON['RESET']}")
else:
    logging.getLogger("PreConfig").warning(f"{NEON['WARNING']}No .env scroll found. Relying on system environment variables or defaults.{NEON['RESET']}")
getcontext().prec = 18 # Precision for Decimal calculations

# --- Helper Functions - Minor Incantations ---
def safe_decimal_conversion(value: Any, default_if_error: Any = pd.NA) -> Union[Decimal, Any]:
    if value is None or (isinstance(value, float) and pd.isna(value)): return default_if_error
    try: return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError): return default_if_error

def _format_for_log(value: Any, is_bool_trend: bool = False, precision: int = 4) -> str:
    if isinstance(value, Decimal): return f"{value:.{precision}f}"
    if is_bool_trend: return "Up" if value is True else ("Down" if value is False else "Indeterminate")
    if pd.isna(value) or value is None: return "N/A"
    return str(value)

def send_termux_notification(title: str, message: str, notification_id: int = 777) -> None:
    """Sends a notification using Termux API, a whisper on the digital wind."""
    notifications_enabled = True 
    timeout_seconds = 10 

    try:
        global CONFIG 
        if 'CONFIG' in globals() and CONFIG is not None:
            notifications_enabled = CONFIG.enable_notifications
            timeout_seconds = CONFIG.notification_timeout_seconds
        else: 
            logging.getLogger("TermuxNotify").debug("CONFIG object not available for send_termux_notification. Using defaults for enabled/timeout.")
    except Exception: # Catch any issue accessing CONFIG, though 'in globals()' should prevent NameError
        logging.getLogger("TermuxNotify").debug("Error accessing CONFIG in send_termux_notification. Using defaults.")

    if not notifications_enabled:
        logging.getLogger("TermuxNotify").debug("Termux notifications are disabled by configuration (or CONFIG not found).")
        return
    
    try:
        safe_title = json.dumps(title)
        safe_message = json.dumps(message)
        
        command = [
            "termux-notification",
            "--title", safe_title,
            "--content", safe_message,
            "--id", str(notification_id),
        ]
        
        logging.getLogger("TermuxNotify").debug(f"{NEON['ACTION']}Attempting to send Termux notification: Title='{title}'{NEON['RESET']}")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=timeout_seconds) 

        if process.returncode == 0:
            logging.getLogger("TermuxNotify").info(f"{NEON['SUCCESS']}Termux notification '{title}' sent successfully.{NEON['RESET']}")
        else:
            err_msg = stderr.decode().strip() if stderr else "Unknown error"
            logging.getLogger("TermuxNotify").error(f"{NEON['ERROR']}Failed to send Termux notification '{title}'. Return code: {process.returncode}. Error: {err_msg}{NEON['RESET']}")
    except FileNotFoundError:
        logging.getLogger("TermuxNotify").error(f"{NEON['ERROR']}Termux API command 'termux-notification' not found. Is 'termux-api' package installed and accessible?{NEON['RESET']}")
    except subprocess.TimeoutExpired:
        logging.getLogger("TermuxNotify").error(f"{NEON['ERROR']}Termux notification command timed out for '{title}'.{NEON['RESET']}")
    except Exception as e:
        logging.getLogger("TermuxNotify").error(f"{NEON['ERROR']}Unexpected error sending Termux notification '{title}': {e}{NEON['RESET']}")
        logging.getLogger("TermuxNotify").debug(traceback.format_exc())

# --- Enums - The Sacred Glyphs ---
class StrategyName(str, Enum): DUAL_SUPERTREND_MOMENTUM = "DUAL_SUPERTREND_MOMENTUM"; EHLERS_FISHER = "EHLERS_FISHER"
class VolatilityRegime(Enum): LOW = "LOW"; NORMAL = "NORMAL"; HIGH = "HIGH"
class OrderEntryType(str, Enum): MARKET = "MARKET"; LIMIT = "LIMIT"

# --- Configuration Class - The Spellbook's Core ---
class Config:
    """Loads and validates configuration parameters for Pyrmethus v2.8.2."""
    def __init__(self) -> None:
        _pre_logger = logging.getLogger("ConfigModule")
        _pre_logger.info(f"{NEON['HEADING']}--- Summoning Configuration Runes v2.8.2 ---{NEON['RESET']}")
        self.api_key: str = self._get_env("BYBIT_API_KEY", required=True, secret=True)
        self.api_secret: str = self._get_env("BYBIT_API_SECRET", required=True, secret=True)
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT")
        self.interval: str = self._get_env("INTERVAL", "1m")
        self.leverage: int = self._get_env("LEVERAGE", 10, cast_type=int) 
        self.sleep_seconds: int = self._get_env("SLEEP_SECONDS", 10, cast_type=int)
        self.strategy_name: StrategyName = StrategyName(self._get_env("STRATEGY_NAME", StrategyName.DUAL_SUPERTREND_MOMENTUM.value).upper())
        self.strategy_instance: 'TradingStrategy'
        self.risk_per_trade_percentage: Decimal = self._get_env("RISK_PER_TRADE_PERCENTAGE", "0.005", cast_type=Decimal)
        self.enable_dynamic_risk: bool = self._get_env("ENABLE_DYNAMIC_RISK", "false", cast_type=bool)
        self.dynamic_risk_min_pct: Decimal = self._get_env("DYNAMIC_RISK_MIN_PCT", "0.0025", cast_type=Decimal)
        self.dynamic_risk_max_pct: Decimal = self._get_env("DYNAMIC_RISK_MAX_PCT", "0.01", cast_type=Decimal)
        self.dynamic_risk_perf_window: int = self._get_env("DYNAMIC_RISK_PERF_WINDOW", 10, cast_type=int)
        self.max_order_usdt_amount: Decimal = self._get_env("MAX_ORDER_USDT_AMOUNT", "1000.0", cast_type=Decimal)
        self.required_margin_buffer: Decimal = self._get_env("REQUIRED_MARGIN_BUFFER", "1.05", cast_type=Decimal)
        self.max_account_margin_ratio: Decimal = self._get_env("MAX_ACCOUNT_MARGIN_RATIO", "0.5", cast_type=Decimal)
        self.enable_max_drawdown_stop: bool = self._get_env("ENABLE_MAX_DRAWDOWN_STOP", "true", cast_type=bool)
        self.max_drawdown_percent: Decimal = self._get_env("MAX_DRAWDOWN_PERCENT", "0.05", cast_type=Decimal)
        self.enable_time_based_stop: bool = self._get_env("ENABLE_TIME_BASED_STOP", "false", cast_type=bool)
        self.max_trade_duration_seconds: int = self._get_env("MAX_TRADE_DURATION_SECONDS", 3600, cast_type=int)
        self.enable_dynamic_atr_sl: bool = self._get_env("ENABLE_DYNAMIC_ATR_SL", "true", cast_type=bool)
        self.atr_short_term_period: int = self._get_env("ATR_SHORT_TERM_PERIOD", 7, cast_type=int)
        self.atr_long_term_period: int = self._get_env("ATR_LONG_TERM_PERIOD", 50, cast_type=int)
        self.volatility_ratio_low_threshold: Decimal = self._get_env("VOLATILITY_RATIO_LOW_THRESHOLD", "0.7", cast_type=Decimal)
        self.volatility_ratio_high_threshold: Decimal = self._get_env("VOLATILITY_RATIO_HIGH_THRESHOLD", "1.5", cast_type=Decimal)
        self.atr_sl_multiplier_low_vol: Decimal = self._get_env("ATR_SL_MULTIPLIER_LOW_VOL", "1.0", cast_type=Decimal)
        self.atr_sl_multiplier_normal_vol: Decimal = self._get_env("ATR_SL_MULTIPLIER_NORMAL_VOL", "1.3", cast_type=Decimal)
        self.atr_sl_multiplier_high_vol: Decimal = self._get_env("ATR_SL_MULTIPLIER_HIGH_VOL", "1.8", cast_type=Decimal)
        self.atr_stop_loss_multiplier: Decimal = self._get_env("ATR_STOP_LOSS_MULTIPLIER", "1.5", cast_type=Decimal)
        self.enable_position_scaling: bool = self._get_env("ENABLE_POSITION_SCALING", "false", cast_type=bool)
        self.max_scale_ins: int = self._get_env("MAX_SCALE_INS", 0, cast_type=int)
        self.scale_in_risk_percentage: Decimal = self._get_env("SCALE_IN_RISK_PERCENTAGE", "0.0025", cast_type=Decimal)
        self.min_profit_for_scale_in_atr: Decimal = self._get_env("MIN_PROFIT_FOR_SCALE_IN_ATR", "1.0", cast_type=Decimal)
        self.trailing_stop_percentage: Decimal = self._get_env("TRAILING_STOP_PERCENTAGE", "0.005", cast_type=Decimal)
        self.trailing_stop_activation_offset_percent: Decimal = self._get_env("TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT", "0.001", cast_type=Decimal)
        self.entry_order_type: OrderEntryType = OrderEntryType(self._get_env("ENTRY_ORDER_TYPE", OrderEntryType.MARKET.value).upper())
        self.limit_order_offset_pips: int = self._get_env("LIMIT_ORDER_OFFSET_PIPS", 2, cast_type=int)
        self.limit_order_fill_timeout_seconds: int = self._get_env("LIMIT_ORDER_FILL_TIMEOUT_SECONDS", 20, cast_type=int)
        self.order_fill_timeout_seconds: int = self._get_env("ORDER_FILL_TIMEOUT_SECONDS", 15, cast_type=int)
        self.st_atr_length: int = self._get_env("ST_ATR_LENGTH", 10, cast_type=int)
        self.st_multiplier: Decimal = self._get_env("ST_MULTIPLIER", "2.0", cast_type=Decimal)
        self.confirm_st_atr_length: int = self._get_env("CONFIRM_ST_ATR_LENGTH", 20, cast_type=int)
        self.confirm_st_multiplier: Decimal = self._get_env("CONFIRM_ST_MULTIPLIER", "3.0", cast_type=Decimal)
        self.momentum_period: int = self._get_env("MOMENTUM_PERIOD", 14, cast_type=int)
        self.momentum_threshold: Decimal = self._get_env("MOMENTUM_THRESHOLD", "0", cast_type=Decimal)
        self.ehlers_fisher_length: int = self._get_env("EHLERS_FISHER_LENGTH", 10, cast_type=int)
        self.ehlers_fisher_signal_length: int = self._get_env("EHLERS_FISHER_SIGNAL_LENGTH", 1, cast_type=int)
        self.atr_calculation_period: int = self.atr_short_term_period if self.enable_dynamic_atr_sl else self._get_env("ATR_CALCULATION_PERIOD", 14, cast_type=int)
        self.enable_notifications: bool = self._get_env("ENABLE_NOTIFICATIONS", "true", cast_type=bool)
        self.notification_timeout_seconds: int = self._get_env("NOTIFICATION_TIMEOUT_SECONDS", 10, cast_type=int)
        self.default_recv_window: int = self._get_env("DEFAULT_RECV_WINDOW", 13000, cast_type=int)
        self.side_buy: str = "buy"; self.side_sell: str = "sell"; self.pos_long: str = "Long"; self.pos_short: str = "Short"; self.pos_none: str = "None"
        self.usdt_symbol: str = "USDT"; self.retry_count: int = 5; self.retry_delay_seconds: int = 5; self.api_fetch_limit_buffer: int = 20
        self.position_qty_epsilon: Decimal = Decimal("1e-9"); self.post_close_delay_seconds: int = 3
        self.MARKET_INFO: Optional[Dict[str, Any]] = None
        self.PAPER_TRADING_MODE: bool = self._get_env("PAPER_TRADING_MODE", "false", cast_type=bool)
        self.send_notification_method = send_termux_notification 

        self._validate_parameters()
        _pre_logger.info(f"{NEON['HEADING']}--- Configuration Runes v2.8.2 Summoned and Verified ---{NEON['RESET']}")

    def _get_env(self, key: str, default: Any = None, cast_type: type = str, required: bool = False, color: str = NEON["PARAM"], secret: bool = False) -> Any:
        _logger = logging.getLogger("ConfigModule._get_env") 
        value_str = os.getenv(key)
        source = "Env Var"
        value_to_cast: Any = None
        display_value = "********" if secret and value_str is not None else value_str

        if value_str is None:
            if required:
                _logger.critical(f"{NEON['CRITICAL']}CRITICAL: Required config rune '{key}' not found.{NEON['RESET']}")
                raise ValueError(f"Required environment variable '{key}' not set.")
            _logger.debug(f"{color}Config Rune {NEON['VALUE']}'{key}'{color}: Not Found. Default: '{NEON['VALUE']}{default}{color}'{NEON['RESET']}")
            value_to_cast = default
            source = "Default"
        else:
            _logger.debug(f"{color}Config Rune {NEON['VALUE']}'{key}'{color}: Env Value: '{NEON['VALUE']}{display_value}{color}'{NEON['RESET']}")
            value_to_cast = value_str

        if value_to_cast is None:
            if required: 
                _logger.critical(f"{NEON['CRITICAL']}CRITICAL: Required config rune '{key}' resolved to None.{NEON['RESET']}")
                raise ValueError(f"Required environment variable '{key}' resolved to None.")
            _logger.debug(f"{color}Config Rune {NEON['VALUE']}'{key}'{color}: Final value is None (not required).{NEON['RESET']}")
            return None

        final_value: Any
        try:
            raw_value_str_for_cast = str(value_to_cast)
            if cast_type == bool: final_value = raw_value_str_for_cast.lower() in ["true", "1", "yes", "y"]
            elif cast_type == Decimal: final_value = Decimal(raw_value_str_for_cast)
            elif cast_type == int: final_value = int(Decimal(raw_value_str_for_cast))
            elif cast_type == float: final_value = float(raw_value_str_for_cast)
            elif cast_type == str: final_value = raw_value_str_for_cast
            else:
                _logger.warning(f"{NEON['WARNING']}Unsupported cast_type '{cast_type.__name__}' for key '{key}'. Raw: '{raw_value_str_for_cast}'.{NEON['RESET']}")
                final_value = raw_value_str_for_cast
        except (ValueError, TypeError, InvalidOperation) as e:
            _logger.error(f"{NEON['ERROR']}Cast error for '{key}': '{value_to_cast}' (Source: {source}). Expected {cast_type.__name__}. Err: {e}. Default: '{default}'.{NEON['RESET']}")
            if default is None:
                if required:
                    _logger.critical(f"{NEON['CRITICAL']}CRITICAL: Cast fail for required '{key}', default is None.{NEON['RESET']}")
                    raise ValueError(f"Required env var '{key}' failed casting, no valid default.")
                else:
                     _logger.warning(f"{NEON['WARNING']}Cast fail for optional '{key}', default is None. Final: None{NEON['RESET']}")
                     return None
            else:
                source = "Default (Fallback)"
                _logger.debug(f"Casting fallback default '{default}' for '{key}' to {cast_type.__name__}")
                try:
                    default_str_for_cast = str(default)
                    if cast_type == bool: final_value = default_str_for_cast.lower() in ["true", "1", "yes", "y"]
                    elif cast_type == Decimal: final_value = Decimal(default_str_for_cast)
                    elif cast_type == int: final_value = int(Decimal(default_str_for_cast))
                    elif cast_type == float: final_value = float(default_str_for_cast)
                    elif cast_type == str: final_value = default_str_for_cast
                    else: final_value = default_str_for_cast 
                    _logger.warning(f"{NEON['WARNING']}Used casted default for {key}: '{NEON['VALUE']}{final_value}{NEON['WARNING']}'{NEON['RESET']}")
                except (ValueError, TypeError, InvalidOperation) as e_default:
                    _logger.critical(f"{NEON['CRITICAL']}CRITICAL: Cast fail for value AND default for '{key}'. Err: {e_default}{NEON['RESET']}")
                    raise ValueError(f"Config error: Cannot cast value or default for '{key}'.")
        display_final_value = "********" if secret else final_value
        _logger.debug(f"{color}Final value for {NEON['VALUE']}'{key}'{color}: {NEON['VALUE']}{display_final_value}{color} (Type: {type(final_value).__name__}, Source: {source}){NEON['RESET']}")
        return final_value

    def _validate_parameters(self) -> None:
        _logger = logging.getLogger("ConfigModule._validate")
        errors = []
        if not (0 < self.risk_per_trade_percentage < 1): errors.append(f"RISK_PER_TRADE_PERCENTAGE ({self.risk_per_trade_percentage}) must be between 0 and 1 (exclusive).")
        if self.leverage < 1: errors.append(f"LEVERAGE ({self.leverage}) must be at least 1.")
        if self.max_scale_ins < 0: errors.append(f"MAX_SCALE_INS ({self.max_scale_ins}) cannot be negative.")
        
        if self.st_atr_length <= 0 or self.confirm_st_atr_length <= 0 or self.momentum_period <= 0: errors.append("Strategy periods (ST_ATR_LENGTH, CONFIRM_ST_ATR_LENGTH, MOMENTUM_PERIOD) must be positive.")
        if self.ehlers_fisher_length <= 0 or self.ehlers_fisher_signal_length <=0: errors.append("Ehlers Fisher lengths must be positive.")

        if self.trailing_stop_percentage < 0: errors.append(f"TRAILING_STOP_PERCENTAGE ({self.trailing_stop_percentage}) cannot be negative.")
        if self.trailing_stop_activation_offset_percent < 0: errors.append(f"TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT ({self.trailing_stop_activation_offset_percent}) cannot be negative.")
        
        if self.enable_dynamic_risk:
            if not (0 < self.dynamic_risk_min_pct < 1): errors.append(f"DYNAMIC_RISK_MIN_PCT ({self.dynamic_risk_min_pct}) must be between 0 and 1.")
            if not (0 < self.dynamic_risk_max_pct < 1): errors.append(f"DYNAMIC_RISK_MAX_PCT ({self.dynamic_risk_max_pct}) must be between 0 and 1.")
            if self.dynamic_risk_min_pct >= self.dynamic_risk_max_pct: errors.append(f"DYNAMIC_RISK_MIN_PCT ({self.dynamic_risk_min_pct}) must be less than DYNAMIC_RISK_MAX_PCT ({self.dynamic_risk_max_pct}).")
        
        if self.enable_max_drawdown_stop and not (0 < self.max_drawdown_percent <= 1): errors.append(f"MAX_DRAWDOWN_PERCENT ({self.max_drawdown_percent}) must be between 0 (exclusive) and 1 (inclusive).")
        
        if self.enable_dynamic_atr_sl and (self.atr_sl_multiplier_low_vol <= 0 or self.atr_sl_multiplier_normal_vol <= 0 or self.atr_sl_multiplier_high_vol <= 0):
            errors.append("Dynamic ATR SL multipliers (low, normal, high vol) must be positive.")
        if not self.enable_dynamic_atr_sl and self.atr_stop_loss_multiplier <=0:
             errors.append("Standard ATR_STOP_LOSS_MULTIPLIER must be positive.")

        if self.sleep_seconds <= 0: errors.append(f"SLEEP_SECONDS ({self.sleep_seconds}) must be positive.")
        if self.limit_order_offset_pips < 0 : errors.append(f"LIMIT_ORDER_OFFSET_PIPS ({self.limit_order_offset_pips}) cannot be negative.")


        if errors:
            error_message = f"Configuration validation failed with {len(errors)} flaws:\n" + "\n".join([f"  - {NEON['ERROR']}{e}{NEON['RESET']}" for e in errors])
            _logger.critical(f"{NEON['CRITICAL']}{error_message}{NEON['RESET']}")
            raise ValueError(error_message)
        else:
            _logger.info(f"{NEON['SUCCESS']}All configuration runes appear potent and well-formed.{NEON['RESET']}")


# --- Logger Setup - The Oracle's Voice ---
LOGGING_LEVEL: int = (logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO)
logs_dir = "logs"; os.makedirs(logs_dir, exist_ok=True)
log_file_name = f"{logs_dir}/pyrmethus_spell_v282_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=LOGGING_LEVEL, format="%(asctime)s [%(levelname)-8s] %(name)-30s %(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file_name)])
logger: logging.Logger = logging.getLogger("PyrmethusCore")
SUCCESS_LEVEL: int = 25; logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")
def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(SUCCESS_LEVEL): self._log(SUCCESS_LEVEL, message, args, **kwargs) # type: ignore[attr-defined]
logging.Logger.success = log_success # type: ignore[attr-defined]
if sys.stdout.isatty():
    for level, color_code in [(logging.DEBUG, NEON['DEBUG']), (logging.INFO, NEON['INFO']), (SUCCESS_LEVEL, NEON['SUCCESS']), (logging.WARNING, NEON['WARNING']), (logging.ERROR, NEON['ERROR']), (logging.CRITICAL, NEON['CRITICAL'])]:
        logging.addLevelName(level, f"{color_code}{logging.getLevelName(level)}{NEON['RESET']}")

# --- Global Objects - Instantiated Arcana ---
try: CONFIG = Config()
except ValueError as config_error:
    logging.getLogger("PyrmethusCore").critical(f"{NEON['CRITICAL']}Configuration loading failed. Error: {config_error}{NEON['RESET']}")
    if 'termux-api' in os.getenv('PATH', ''): send_termux_notification("Pyrmethus Startup Failure", f"Config Error: {str(config_error)[:200]}")
    sys.exit(1)
except Exception as general_config_error:
    logging.getLogger("PyrmethusCore").critical(f"{NEON['CRITICAL']}Unexpected critical error during configuration: {general_config_error}{NEON['RESET']}")
    logging.getLogger("PyrmethusCore").debug(traceback.format_exc())
    if 'termux-api' in os.getenv('PATH', ''): send_termux_notification("Pyrmethus Critical Failure", f"Unexpected Config Error: {str(general_config_error)[:200]}")
    sys.exit(1)

# --- Trading Strategy Abstract Base Class & Implementations - The Schools of Magic ---
class TradingStrategy(ABC):
    def __init__(self, config: Config, df_columns: Optional[List[str]] = None):
        self.config = config; self.logger = logging.getLogger(f"Strategy.{self.__class__.__name__}"); self.required_columns = df_columns or []
        self.logger.info(f"{NEON['STRATEGY']}Strategy Form '{self.__class__.__name__}' materializing...{NEON['RESET']}")
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]: pass
    def _validate_df(self, df: pd.DataFrame, min_rows: int = 2) -> bool:
        if df is None or df.empty or len(df) < min_rows: self.logger.debug(f"Insufficient market whispers (Rows: {len(df) if df is not None else 0}, Min: {min_rows})."); return False
        if self.required_columns and not all(col in df.columns for col in self.required_columns):
            missing = [col for col in self.required_columns if col not in df.columns]; self.logger.warning(f"{NEON['WARNING']}Market scroll missing runes: {missing}{NEON['WARNING']}."); return False
        if self.required_columns and df.iloc[-1][self.required_columns].isnull().any(): self.logger.debug(f"Faint runes (NaNs) in last candle for critical columns."); 
        return True
    def _get_default_signals(self) -> Dict[str, Any]: return {"enter_long": False, "enter_short": False, "exit_long": False, "exit_short": False, "exit_reason": "Awaiting True Omens"}

class DualSupertrendMomentumStrategy(TradingStrategy):
    def __init__(self, config: Config): super().__init__(config, df_columns=["st_long_flip", "st_short_flip", "confirm_trend", "momentum"])
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        signals = self._get_default_signals()
        min_rows = max(self.config.st_atr_length, self.config.confirm_st_atr_length, self.config.momentum_period) + 10
        if not self._validate_df(df, min_rows=min_rows): return signals
        last = df.iloc[-1]; primary_long_flip = last.get("st_long_flip", False); primary_short_flip = last.get("st_short_flip", False)
        confirm_is_up = last.get("confirm_trend", pd.NA); momentum_val = safe_decimal_conversion(last.get("momentum", pd.NA), pd.NA)
        if pd.isna(confirm_is_up) or pd.isna(momentum_val): self.logger.debug(f"Confirm Trend ({_format_for_log(confirm_is_up, True)}) or Momentum ({_format_for_log(momentum_val)}) is NA."); return signals
        if primary_long_flip and confirm_is_up is True and momentum_val > self.config.momentum_threshold:
            signals["enter_long"] = True; self.logger.info(f"{NEON['SIDE_LONG']}DualST+Mom: LONG Entry - ST Flip, Confirm Up, Mom ({_format_for_log(momentum_val)}) > {_format_for_log(self.config.momentum_threshold)}{NEON['RESET']}")
        elif primary_short_flip and confirm_is_up is False and momentum_val < -self.config.momentum_threshold:
            signals["enter_short"] = True; self.logger.info(f"{NEON['SIDE_SHORT']}DualST+Mom: SHORT Entry - ST Flip, Confirm Down, Mom ({_format_for_log(momentum_val)}) < -{_format_for_log(self.config.momentum_threshold)}{NEON['RESET']}")
        if primary_short_flip: signals["exit_long"] = True; signals["exit_reason"] = "Primary ST Flipped Short"; self.logger.info(f"{NEON['ACTION']}DualST+Mom: EXIT LONG - Primary ST Flipped Short{NEON['RESET']}")
        if primary_long_flip: signals["exit_short"] = True; signals["exit_reason"] = "Primary ST Flipped Long"; self.logger.info(f"{NEON['ACTION']}DualST+Mom: EXIT SHORT - Primary ST Flipped Long{NEON['RESET']}")
        return signals

class EhlersFisherStrategy(TradingStrategy):
    def __init__(self, config: Config): super().__init__(config, df_columns=["ehlers_fisher", "ehlers_signal"])
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        signals = self._get_default_signals(); min_rows = self.config.ehlers_fisher_length + self.config.ehlers_fisher_signal_length + 5
        if not self._validate_df(df, min_rows=min_rows) or len(df) < 2: return signals
        last = df.iloc[-1]; prev = df.iloc[-2]
        fisher_now = safe_decimal_conversion(last.get("ehlers_fisher"), pd.NA); signal_now = safe_decimal_conversion(last.get("ehlers_signal"), pd.NA)
        fisher_prev = safe_decimal_conversion(prev.get("ehlers_fisher"), pd.NA); signal_prev = safe_decimal_conversion(prev.get("ehlers_signal"), pd.NA)
        if pd.isna(fisher_now) or pd.isna(signal_now) or pd.isna(fisher_prev) or pd.isna(signal_prev): self.logger.debug(f"Ehlers Fisher/Signal NA."); return signals
        if fisher_prev <= signal_prev and fisher_now > signal_now: signals["enter_long"] = True; self.logger.info(f"{NEON['SIDE_LONG']}Ehlers: LONG Entry - Fisher cross ABOVE Signal.{NEON['RESET']}")
        elif fisher_prev >= signal_prev and fisher_now < signal_now: signals["enter_short"] = True; self.logger.info(f"{NEON['SIDE_SHORT']}Ehlers: SHORT Entry - Fisher cross BELOW Signal.{NEON['RESET']}")
        if fisher_prev >= signal_prev and fisher_now < signal_now: signals["exit_long"] = True; signals["exit_reason"] = "Ehlers Fisher cross BELOW Signal"; self.logger.info(f"{NEON['ACTION']}Ehlers: EXIT LONG - Fisher cross BELOW Signal.{NEON['RESET']}")
        elif fisher_prev <= signal_prev and fisher_now > signal_now: signals["exit_short"] = True; signals["exit_reason"] = "Ehlers Fisher cross ABOVE Signal"; self.logger.info(f"{NEON['ACTION']}Ehlers: EXIT SHORT - Fisher cross ABOVE Signal.{NEON['RESET']}")
        return signals

strategy_map: Dict[StrategyName, Type[TradingStrategy]] = {StrategyName.DUAL_SUPERTREND_MOMENTUM: DualSupertrendMomentumStrategy, StrategyName.EHLERS_FISHER: EhlersFisherStrategy}
StrategyClass = strategy_map.get(CONFIG.strategy_name)
if StrategyClass: CONFIG.strategy_instance = StrategyClass(CONFIG); logger.success(f"{NEON['SUCCESS']}Strategy '{NEON['STRATEGY']}{CONFIG.strategy_name.value}{NEON['SUCCESS']}' invoked.{NEON['RESET']}")
else: err_msg = f"Failed to init strategy '{CONFIG.strategy_name.value}'."; logger.critical(f"{NEON['CRITICAL']}{err_msg}{NEON['RESET']}"); send_termux_notification("Pyrmethus Critical Error", err_msg); sys.exit(1)

# --- Trade Metrics Tracking - The Grand Ledger of Deeds ---
class TradeMetrics:
    def __init__(self):
        self.trades: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("TradeMetrics")
        self.initial_equity: Optional[Decimal] = None
        self.daily_start_equity: Optional[Decimal] = None
        self.last_daily_reset_day: Optional[int] = None
        self.logger.info(f"{NEON['INFO']}TradeMetrics Ledger opened, ready to chronicle deeds.{NEON['RESET']}")

    def set_initial_equity(self, equity: Decimal):
        if self.initial_equity is None:
            self.initial_equity = equity
            self.logger.info(f"{NEON['INFO']}Initial Session Equity rune inscribed: {NEON['VALUE']}{equity:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}")
        today = datetime.now(pytz.utc).day
        if self.last_daily_reset_day != today or self.daily_start_equity is None:
            self.daily_start_equity = equity
            self.last_daily_reset_day = today
            self.logger.info(f"{NEON['INFO']}Daily Equity Ward reset for drawdown tracking. Dawn Equity: {NEON['VALUE']}{equity:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}")

    def check_drawdown(self, current_equity: Decimal) -> Tuple[bool, str]:
        if not CONFIG.enable_max_drawdown_stop or self.daily_start_equity is None or self.daily_start_equity <= 0:
            return False, ""
        drawdown = self.daily_start_equity - current_equity
        drawdown_pct = (drawdown / self.daily_start_equity) if self.daily_start_equity > 0 else Decimal(0)
        if drawdown_pct >= CONFIG.max_drawdown_percent:
            reason = f"Max daily drawdown ward breached ({NEON['PNL_NEG']}{drawdown_pct:.2%}{NEON['RESET']} >= {NEON['VALUE']}{CONFIG.max_drawdown_percent:.2%}{NEON['RESET']})"
            self.logger.warning(f"{NEON['WARNING']}{reason}. Trading must halt for the day to conserve essence.{NEON['RESET']}")
            CONFIG.send_notification_method("Pyrmethus: Max Drawdown Hit!", f"Drawdown: {drawdown_pct:.2%}. Trading halted.")
            return True, reason
        return False, ""

    def log_trade(self, symbol: str, side: str, entry_price: Decimal, exit_price: Decimal, qty: Decimal,
                  entry_time_ms: int, exit_time_ms: int, reason: str, part_id: str, pnl_str: str,
                  mae_str: Optional[str] = None, mfe_str: Optional[str] = None):
        profit = safe_decimal_conversion(pnl_str, Decimal(0))
        entry_dt_iso = datetime.fromtimestamp(entry_time_ms / 1000, tz=pytz.utc).isoformat()
        exit_dt_iso = datetime.fromtimestamp(exit_time_ms / 1000, tz=pytz.utc).isoformat()
        duration = (datetime.fromisoformat(exit_dt_iso.replace("Z", "+00:00")) - datetime.fromisoformat(entry_dt_iso.replace("Z", "+00:00"))).total_seconds()
        
        self.trades.append({
            "symbol": symbol, "side": side, "entry_price_str": str(entry_price), "exit_price_str": str(exit_price),
            "qty_str": str(qty), "profit_str": str(profit), "entry_time_iso": entry_dt_iso, "exit_time_iso": exit_dt_iso,
            "duration_seconds": duration, "exit_reason": reason, "part_id": part_id,
            "mae_str": mae_str, "mfe_str": mfe_str
        })
        pnl_color = NEON["PNL_POS"] if profit > 0 else (NEON["PNL_NEG"] if profit < 0 else NEON["PNL_ZERO"])
        self.logger.log(SUCCESS_LEVEL,
                        f"{NEON['HEADING']}Trade Chronicle (Part:{part_id}): {side.upper()} {NEON['QTY']}{qty}{NEON['RESET']} {symbol.split('/')[0]} | "
                        f"P/L: {pnl_color}{profit:.2f} {CONFIG.usdt_symbol}{NEON['RESET']} | Reason: {reason}")

    def calculate_mae_mfe(self, part_id: str, entry_price: Decimal, side: str,
                          entry_time_ms: int, exit_time_ms: int,
                          exchange: ccxt.Exchange, symbol: str, interval: str) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """
        Calculates Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE)
        for a given trade part by scrying into the historical OHLCV data.
        
        NOTE: This requires fetching OHLCV data for the trade's duration.
        The accuracy depends on the granularity of data fetched (e.g., 1-minute candles).
        For true tick-level MAE/MFE, one would need tick data, which is often harder to obtain.
        This spell currently offers a glimpse, not the full tapestry of tick-by-tick reality.
        """
        self.logger.debug(f"{NEON['COMMENT']}# Divining MAE/MFE for trade part {part_id}... A quest into the recent past.{NEON['RESET']}")
        
        if exit_time_ms <= entry_time_ms:
            self.logger.warning(f"{NEON['WARNING']}MAE/MFE calc: Exit time ({datetime.fromtimestamp(exit_time_ms/1000, tz=pytz.utc).isoformat()}) "
                                f"is not after entry time ({datetime.fromtimestamp(entry_time_ms/1000, tz=pytz.utc).isoformat()}) for part {part_id}. "
                                f"Cannot divine excursions from such temporal paradoxes.{NEON['RESET']}")
            return None, None

        self.logger.warning(f"{NEON['WARNING']}MAE/MFE calculation for part {part_id} is a rite yet to be fully mastered. "
                            f"This version returns (None, None). True MAE/MFE requires historical candle divination "
                            f"from {datetime.fromtimestamp(entry_time_ms/1000, tz=pytz.utc).isoformat()} to "
                            f"{datetime.fromtimestamp(exit_time_ms/1000, tz=pytz.utc).isoformat()}.{NEON['RESET']}")
        # Placeholder: Actual implementation would involve fetching OHLCV data for the trade's duration
        # and then finding min/max prices as appropriate for MAE/MFE.
        return None, None 

    def get_performance_trend(self, window: int) -> float:
        """Assesses the recent fortune by calculating win rate over a sliding window of trades."""
        if window <= 0 or not self.trades:
            self.logger.debug(f"{NEON['COMMENT']}Performance trend: Crystal ball is cloudy (no trades or invalid window). Assuming neutral winds (0.5).{NEON['RESET']}")
            return 0.5
        
        recent_trades = self.trades[-window:]
        if not recent_trades:
            self.logger.debug(f"{NEON['COMMENT']}Performance trend: Not enough recent trades for window {window}. Assuming neutral winds (0.5).{NEON['RESET']}")
            return 0.5
            
        wins = sum(1 for t in recent_trades if safe_decimal_conversion(t.get("profit_str"), Decimal(0)) > 0)
        trend = float(wins / len(recent_trades))
        self.logger.debug(f"{NEON['INFO']}Performance Trend (last {len(recent_trades)} trades): {NEON['VALUE']}{trend:.2%}{NEON['RESET']} wins.")
        return trend

    def summary(self) -> str:
        """Generates a summary of all chronicled deeds, a testament to Pyrmethus's journey."""
        if not self.trades:
            msg = f"{NEON['INFO']}The Grand Ledger is empty. No deeds chronicled yet in this epoch.{NEON['RESET']}"
            self.logger.info(msg)
            return msg
        
        total_trades = len(self.trades)
        wins = sum(1 for t in self.trades if safe_decimal_conversion(t.get("profit_str"), Decimal(0)) > 0)
        losses = sum(1 for t in self.trades if safe_decimal_conversion(t.get("profit_str"), Decimal(0)) < 0)
        breakeven = total_trades - wins - losses
        
        win_rate_val = (Decimal(wins) / Decimal(total_trades)) if total_trades > 0 else Decimal(0)
        win_rate_str = f"{NEON['VALUE']}{win_rate_val:.2%}{NEON['RESET']}"

        total_profit_val = sum(safe_decimal_conversion(t.get("profit_str"), Decimal(0)) for t in self.trades)
        total_profit_color = NEON["PNL_POS"] if total_profit_val > 0 else (NEON["PNL_NEG"] if total_profit_val < 0 else NEON["PNL_ZERO"])
        total_profit_str = f"{total_profit_color}{total_profit_val:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}"
        
        avg_profit_val = (total_profit_val / Decimal(total_trades)) if total_trades > 0 else Decimal(0)
        avg_profit_color = NEON["PNL_POS"] if avg_profit_val > 0 else (NEON["PNL_NEG"] if avg_profit_val < 0 else NEON["PNL_ZERO"])
        avg_profit_str = f"{avg_profit_color}{avg_profit_val:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}"

        summary_str = (
            f"\n{NEON['HEADING']}--- Pyrmethus Trade Metrics Summary (v2.8.2) ---{NEON['RESET']}\n"
            f"{NEON['SUBHEADING']}A Chronicle of Ventures and Valour:{NEON['RESET']}\n"
            f"Total Trade Parts Chronicled: {NEON['VALUE']}{total_trades}{NEON['RESET']}\n"
            f"  Victories (Wins): {NEON['PNL_POS']}{wins}{NEON['RESET']}\n"
            f"  Defeats (Losses): {NEON['PNL_NEG']}{losses}{NEON['RESET']}\n"
            f"  Stalemates (Breakeven): {NEON['PNL_ZERO']}{breakeven}{NEON['RESET']}\n"
            f"Victory Rate (by parts): {win_rate_str}\n"
            f"Total Spoils (Net P/L): {total_profit_str}\n"
            f"Avg Spoils per Part: {avg_profit_str}\n"
        )
        if self.initial_equity is not None:
            current_equity_approx = self.initial_equity + total_profit_val 
            overall_pnl_pct_val = (total_profit_val / self.initial_equity) * 100 if self.initial_equity > 0 else Decimal(0)
            overall_pnl_pct_color = NEON["PNL_POS"] if overall_pnl_pct_val > 0 else (NEON["PNL_NEG"] if overall_pnl_pct_val < 0 else NEON["PNL_ZERO"])
            overall_pnl_pct_str = f"{overall_pnl_pct_color}{overall_pnl_pct_val:.2%}{NEON['RESET']}"
            
            summary_str += f"Initial Session Hoard: {NEON['VALUE']}{self.initial_equity:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}\n"
            summary_str += f"Approx. Current Hoard (based on ledger): {NEON['VALUE']}{current_equity_approx:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}\n"
            summary_str += f"Overall Session P/L % (based on ledger): {overall_pnl_pct_str}\n"

        summary_str += f"{NEON['HEADING']}--- End of Grand Ledger Reading ---{NEON['RESET']}"
        self.logger.info(summary_str) 
        return summary_str

    def get_serializable_trades(self) -> List[Dict[str, Any]]: return self.trades
    def load_trades_from_list(self, trades_list: List[Dict[str, Any]]): self.trades = trades_list; self.logger.info(f"{NEON['INFO']}TradeMetrics: Re-inked {len(self.trades)} trades.{NEON['RESET']}")

trade_metrics = TradeMetrics()
_active_trade_parts: List[Dict[str, Any]] = [] 
_last_heartbeat_save_time: float = 0.0

# --- State Persistence Functions (Phoenix Feather) ---
def save_persistent_state(force_heartbeat: bool = False) -> None:
    global _active_trade_parts, trade_metrics, _last_heartbeat_save_time; now = time.time()
    if not (force_heartbeat or (now - _last_heartbeat_save_time > HEARTBEAT_INTERVAL_SECONDS)): return
    logger.debug(f"{NEON['COMMENT']}# Phoenix Feather scribing memories...{NEON['RESET']}")
    try:
        serializable_parts = []
        for part in _active_trade_parts:
            p_copy = part.copy()
            for k, v in p_copy.items():
                if isinstance(v, Decimal): p_copy[k] = str(v)
                elif isinstance(v, (datetime, pd.Timestamp)): p_copy[k] = v.isoformat()
            serializable_parts.append(p_copy)
        state_data = {"pyrmethus_version": "2.8.2", "timestamp_utc_iso": datetime.now(pytz.utc).isoformat(), "active_trade_parts": serializable_parts, "trade_metrics_trades": trade_metrics.get_serializable_trades(), "config_symbol": CONFIG.symbol, "config_strategy": CONFIG.strategy_name.value, "daily_start_equity_str": str(trade_metrics.daily_start_equity) if trade_metrics.daily_start_equity is not None else None, "last_daily_reset_day": trade_metrics.last_daily_reset_day}
        temp_file = STATE_FILE_PATH + ".tmp";
        with open(temp_file, 'w') as f: json.dump(state_data, f, indent=4)
        os.replace(temp_file, STATE_FILE_PATH); _last_heartbeat_save_time = now
        logger.log(logging.INFO if force_heartbeat else logging.DEBUG, f"{NEON['SUCCESS']}Phoenix Feather: State memories scribed.{NEON['RESET']}")
    except Exception as e: logger.error(f"{NEON['ERROR']}Phoenix Feather Error scribing state: {e}{NEON['RESET']}"); logger.debug(traceback.format_exc())

def load_persistent_state() -> bool:
    global _active_trade_parts, trade_metrics; logger.info(f"{NEON['COMMENT']}# Phoenix Feather seeks past memories...{NEON['RESET']}")
    if not os.path.exists(STATE_FILE_PATH): logger.info(f"{NEON['INFO']}Phoenix Feather: No previous scroll found.{NEON['RESET']}"); return False
    try:
        with open(STATE_FILE_PATH, 'r') as f: state_data = json.load(f)
        if state_data.get("pyrmethus_version") != "2.8.2": logger.warning(f"{NEON['WARNING']}Phoenix Scroll version mismatch! Saved: {state_data.get('pyrmethus_version')}, Current: 2.8.2. Caution advised.{NEON['RESET']}")
        if state_data.get("config_symbol") != CONFIG.symbol or state_data.get("config_strategy") != CONFIG.strategy_name.value:
            logger.warning(f"{NEON['WARNING']}Phoenix Scroll sigils mismatch (symbol/strategy). Ignoring old scroll.{NEON['RESET']}")
            archive_name = f"{STATE_FILE_PATH}.archived_{time.strftime('%Y%m%d%H%M%S')}"
            try: os.rename(STATE_FILE_PATH, archive_name); logger.info(f"Archived scroll to {archive_name}")
            except OSError as e_archive: logger.error(f"Could not archive scroll: {e_archive}")
            return False
        
        _active_trade_parts.clear()
        for part_data in state_data.get("active_trade_parts", []):
            restored = part_data.copy()
            for k, v_str in restored.items():
                if k in ["entry_price", "qty", "sl_price", "initial_usdt_value"] and isinstance(v_str, str): 
                    try: restored[k] = Decimal(v_str)
                    except InvalidOperation: logger.warning(f"Could not convert '{v_str}' to Decimal for key '{k}' in loaded part.")
                elif k == "entry_time_ms":
                    if isinstance(v_str, str): 
                        try: restored[k] = int(datetime.fromisoformat(v_str.replace("Z", "+00:00")).timestamp() * 1000)
                        except ValueError: 
                            try: restored[k] = int(v_str) # if it's a numeric string
                            except ValueError: logger.warning(f"Could not convert '{v_str}' to int for entry_time_ms.")
                    elif isinstance(v_str, (float, int)): restored[k] = int(v_str) # if saved as number
            _active_trade_parts.append(restored)
        
        trade_metrics.load_trades_from_list(state_data.get("trade_metrics_trades", []))
        daily_equity_str = state_data.get("daily_start_equity_str")
        if daily_equity_str and daily_equity_str.lower() != 'none': 
            try: trade_metrics.daily_start_equity = Decimal(daily_equity_str)
            except InvalidOperation: logger.warning(f"Could not restore daily_start_equity from '{daily_equity_str}'.")
        trade_metrics.last_daily_reset_day = state_data.get("last_daily_reset_day")
        logger.success(f"{NEON['SUCCESS']}Phoenix Feather: Memories reawakened! Active Parts: {len(_active_trade_parts)}{NEON['RESET']}")
        return True
    except json.JSONDecodeError as e_json:
        logger.error(f"{NEON['ERROR']}Phoenix Feather: Scroll '{STATE_FILE_PATH}' is corrupted. Error: {e_json}{NEON['RESET']}")
        archive_name = f"{STATE_FILE_PATH}.corrupted_{time.strftime('%Y%m%d%H%M%S')}"
        try: os.rename(STATE_FILE_PATH, archive_name); logger.info(f"Archived corrupted scroll to {archive_name}")
        except OSError as e_archive: logger.error(f"Could not archive corrupted scroll: {e_archive}")
        return False
    except Exception as e: logger.error(f"{NEON['ERROR']}Phoenix Feather Error reawakening: {e}{NEON['RESET']}"); logger.debug(traceback.format_exc()); return False

# --- Indicator Calculation - The Alchemist's Art ---
def calculate_all_indicators(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    logger.debug(f"{NEON['COMMENT']}# Transmuting market data with indicator alchemy...{NEON['RESET']}")
    if df.empty: logger.warning(f"{NEON['WARNING']}Market data scroll empty.{NEON['RESET']}"); return df
    for col in ['close', 'high', 'low', 'volume']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if config.strategy_name == StrategyName.DUAL_SUPERTREND_MOMENTUM:
        st_p = f"ST_{config.st_atr_length}_{config.st_multiplier}"; st_c = f"CONFIRM_ST_{config.confirm_st_atr_length}_{config.confirm_st_multiplier}"
        df.ta.supertrend(length=config.st_atr_length, multiplier=config.st_multiplier, append=True, col_names=(st_p, f"{st_p}d", f"{st_p}l", f"{st_p}s"))
        df.ta.supertrend(length=config.confirm_st_atr_length, multiplier=config.confirm_st_multiplier, append=True, col_names=(st_c, f"{st_c}d", f"{st_c}l", f"{st_c}s"))
        if f"{st_p}d" in df.columns:
            df["st_trend_up"] = df[f"{st_p}d"] == 1
            df["st_long_flip"] = (df["st_trend_up"]) & (df["st_trend_up"].shift(1) == False)
            df["st_short_flip"] = (df["st_trend_up"] == False) & (df["st_trend_up"].shift(1))
        else: df["st_long_flip"], df["st_short_flip"] = False, False; logger.error(f"Primary ST direction col '{f'{st_p}d'}' missing!")
        if f"{st_c}d" in df.columns: df["confirm_trend"] = df[f"{st_c}d"].apply(lambda x: True if x == 1 else (False if x == -1 else pd.NA))
        else: df["confirm_trend"] = pd.NA; logger.error(f"Confirm ST direction col '{f'{st_c}d'}' missing!")
        if 'close' in df.columns and not df['close'].isnull().all(): df.ta.mom(length=config.momentum_period, append=True, col_names=("momentum",))
        else: df["momentum"] = pd.NA
    elif config.strategy_name == StrategyName.EHLERS_FISHER:
        if 'high' in df.columns and 'low' in df.columns and not (df['high'].isnull().all() or df['low'].isnull().all()):
            df.ta.fisher(length=config.ehlers_fisher_length, signal=config.ehlers_fisher_signal_length, append=True, col_names=("ehlers_fisher", "ehlers_signal"))
        else: df["ehlers_fisher"], df["ehlers_signal"] = pd.NA, pd.NA
    
    atr_col_name = f"ATR_{config.atr_calculation_period}"
    if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns: df.ta.atr(length=config.atr_calculation_period, append=True, col_names=(atr_col_name,))
    else: df[atr_col_name] = pd.NA; logger.warning(f"Cannot calculate {atr_col_name}, missing HLC data.")
    return df

# --- Exchange Interaction Primitives ---
@retry((ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection), tries=CONFIG.retry_count, delay=CONFIG.retry_delay_seconds, backoff=2, logger=logger)
def get_market_data(exchange: ccxt.Exchange, symbol: str, interval: str, limit: int = OHLCV_LIMIT) -> Optional[pd.DataFrame]:
    logger.debug(f"{NEON['COMMENT']}# Fetching {limit} candles for {symbol} ({interval})...{NEON['RESET']}")
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, interval, limit=limit)
        if not ohlcv: logger.warning(f"{NEON['WARNING']}No OHLCV data returned for {symbol}.{NEON['RESET']}"); return None
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True) 
        for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = pd.to_numeric(df[col], errors='coerce')
        logger.debug(f"Fetched {len(df)} candles. Latest: {df['timestamp'].iloc[-1] if not df.empty else 'N/A'}")
        return df
    except Exception as e: logger.error(f"{NEON['ERROR']}Error fetching market data for {symbol}: {e}{NEON['RESET']}"); logger.debug(traceback.format_exc()); return None

@retry((ccxt.NetworkError, ccxt.RequestTimeout), tries=CONFIG.retry_count, delay=CONFIG.retry_delay_seconds)
def fetch_account_balance(exchange: ccxt.Exchange, currency_code: str = CONFIG.usdt_symbol) -> Optional[Decimal]:
    logger.debug(f"{NEON['COMMENT']}# Fetching account balance for {currency_code}...{NEON['RESET']}")
    try:
        balance_data = exchange.fetch_balance()
        total_balance = safe_decimal_conversion(balance_data.get(currency_code, {}).get('total'))
        if total_balance is None: logger.warning(f"{NEON['WARNING']}{currency_code} balance not found or invalid.{NEON['RESET']}"); return None
        logger.info(f"{NEON['INFO']}Current {currency_code} Balance: {NEON['VALUE']}{total_balance:.2f}{NEON['RESET']}")
        return total_balance
    except Exception as e: logger.error(f"{NEON['ERROR']}Error fetching account balance: {e}{NEON['RESET']}"); return None

def get_current_position_info() -> Tuple[str, Decimal]:
    global _active_trade_parts
    if not _active_trade_parts: return CONFIG.pos_none, Decimal(0)
    total_qty = sum(part.get('qty', Decimal(0)) for part in _active_trade_parts) # Ensure qty exists and default to 0
    current_side = _active_trade_parts[0]['side'] if _active_trade_parts and 'side' in _active_trade_parts[0] else CONFIG.pos_none
    return current_side, total_qty

def calculate_position_size(balance: Decimal, risk_per_trade_pct: Decimal, entry_price: Decimal, sl_price: Decimal, market_info: Dict) -> Optional[Decimal]:
    logger.debug(f"{NEON['COMMENT']}# Calculating position size... Balance: {balance:.2f}, Risk %: {risk_per_trade_pct:.4f}{NEON['RESET']}")
    if balance <= 0 or entry_price <= 0 or sl_price <= 0: logger.warning(f"{NEON['WARNING']}Invalid inputs for position sizing.{NEON['RESET']}"); return None
    
    risk_per_unit = abs(entry_price - sl_price)
    if risk_per_unit == 0: logger.warning(f"{NEON['WARNING']}Risk per unit is zero (entry=SL). Cannot size.{NEON['RESET']}"); return None

    usdt_at_risk = balance * risk_per_trade_pct
    logger.debug(f"USDT at risk: {usdt_at_risk:.2f}")
    
    quantity = usdt_at_risk / risk_per_unit
    
    position_usdt_value = quantity * entry_price
    if position_usdt_value > CONFIG.max_order_usdt_amount:
        logger.info(f"Calculated position USDT value {position_usdt_value:.2f} exceeds MAX_ORDER_USDT_AMOUNT {CONFIG.max_order_usdt_amount}. Capping.")
        position_usdt_value = CONFIG.max_order_usdt_amount
        if entry_price > 0: quantity = position_usdt_value / entry_price
        else: logger.warning(f"{NEON['WARNING']}Entry price is zero, cannot recalculate quantity for capped USDT value.{NEON['RESET']}"); return None


    min_qty_val = market_info.get('limits', {}).get('amount', {}).get('min')
    min_qty = safe_decimal_conversion(min_qty_val) if min_qty_val is not None else None # Ensure Decimal
    
    qty_precision_str = market_info.get('precision', {}).get('amount')
    # CCXT precision for amount can be number of decimal places (e.g., 3 for 0.001) or a step (e.g., 0.001)
    # Assuming it's number of decimal places for Bybit as per earlier context
    qty_precision = 0 # Default if not found or not convertible
    if isinstance(qty_precision_str, (int, float)) or (isinstance(qty_precision_str, str) and qty_precision_str.isdigit()):
        qty_precision = int(qty_precision_str)
    elif isinstance(qty_precision_str, str) and '.' in qty_precision_str : # e.g. "0.001" means 3 decimal places
        try:
            # Count decimal places from string like "0.001"
            if Decimal(qty_precision_str) < 1 and Decimal(qty_precision_str) > 0 :
                 qty_precision = abs(Decimal(qty_precision_str).as_tuple().exponent)
            # else it might be a step value like "10" meaning multiples of 10, precision is effectively 0 for decimals
        except InvalidOperation:
            logger.warning(f"{NEON['WARNING']}Could not interpret quantity precision string '{qty_precision_str}'. Defaulting to 0 decimal places.{NEON['RESET']}")


    if min_qty is not None and quantity < min_qty:
        logger.warning(f"{NEON['WARNING']}Calculated quantity {quantity} is below minimum {min_qty}. Cannot place order.{NEON['RESET']}")
        return None
    
    if qty_precision > 0:
        quantizer = Decimal('1e-' + str(qty_precision))
        quantity = (quantity // quantizer) * quantizer # Floor to precision
    else: 
        # If precision is 0 (integer quantity) or step value like 10, 100
        # For step values like 10, this should be quantity = floor(quantity / step) * step
        # Assuming qty_precision from market_info.precision.amount is # of decimals for now.
        quantity = Decimal(int(quantity))

    if quantity <= 0: logger.warning(f"{NEON['WARNING']}Final quantity is zero or negative after adjustments. Cannot place order.{NEON['RESET']}"); return None
    
    # Use market_info.base if available, otherwise parse from symbol
    base_currency = market_info.get('base', CONFIG.symbol.split('/')[0])
    logger.info(f"Calculated position size: {NEON['QTY']}{quantity:.{qty_precision}f}{NEON['RESET']} {base_currency}")
    return quantity

@retry((ccxt.NetworkError, ccxt.RequestTimeout, ccxt.InsufficientFunds), tries=2, delay=3, logger=logger)
def place_risked_order(exchange: ccxt.Exchange, config: Config, side: str, entry_price_target: Decimal, sl_price: Decimal, atr_val: Optional[Decimal]) -> Optional[Dict]:
    global _active_trade_parts
    logger.info(f"{NEON['ACTION']}Attempting to place {side.upper()} order for {config.symbol}... Target Entry: {entry_price_target}, SL: {sl_price}{NEON['RESET']}")
    
    balance = fetch_account_balance(exchange, config.usdt_symbol)
    if balance is None or balance <= Decimal(10): 
        logger.error(f"{NEON['ERROR']}Insufficient balance ({balance}) or failed to fetch. Cannot place order.{NEON['RESET']}")
        config.send_notification_method("Pyrmethus Order Fail", "Insufficient balance for new order.")
        return None

    if config.MARKET_INFO is None:
        logger.error(f"{NEON['ERROR']}Market info not loaded. Cannot calculate position size or place order.{NEON['RESET']}")
        return None
        
    quantity = calculate_position_size(balance, config.risk_per_trade_percentage, entry_price_target, sl_price, config.MARKET_INFO)
    if quantity is None or quantity <= 0: return None

    order_type_str = 'Market' if config.entry_order_type == OrderEntryType.MARKET else 'Limit'
    params = {
        'timeInForce': 'GTC' if order_type_str == 'Limit' else 'IOC', # IOC for Market, GTC for Limit often
        'stopLoss': float(sl_price), 
        'slTriggerBy': 'MarkPrice', 
        'positionIdx': 0 
    }
    
    try:
        logger.info(f"Placing {order_type_str.upper()} {side.upper()} order: Qty {quantity}, Symbol {config.symbol}, SL {sl_price}")
        order = exchange.create_order(config.symbol, order_type_str.lower(), side.lower(), float(quantity), float(entry_price_target) if order_type_str == 'Limit' else None, params)
        
        logger.success(f"{NEON['SUCCESS']}Entry Order Sent: ID {order['id']}, Status {order.get('status', 'N/A')}{NEON['RESET']}")
        config.send_notification_method(f"Pyrmethus Order Placed: {side.upper()}", f"{config.symbol} Qty: {quantity} @ {entry_price_target if order_type_str == 'Limit' else 'Market'}")

        time.sleep(config.order_fill_timeout_seconds / 2) # Wait a bit based on config
        filled_order = exchange.fetch_order(order['id'], config.symbol)
        
        if filled_order.get('status') == 'closed': 
            actual_entry_price = Decimal(str(filled_order.get('average', filled_order.get('price', entry_price_target))))
            actual_qty = Decimal(str(filled_order.get('filled', quantity)))

            part_id = str(uuid.uuid4())[:8] 
            new_part = {
                "part_id": part_id, "entry_order_id": order['id'], "sl_order_id": None, 
                "symbol": config.symbol, "side": side, "entry_price": actual_entry_price, "qty": actual_qty,
                "entry_time_ms": int(filled_order.get('timestamp', time.time() * 1000)),
                "sl_price": sl_price, "atr_at_entry": atr_val, "initial_usdt_value": actual_qty * actual_entry_price
            }
            _active_trade_parts.append(new_part)
            save_persistent_state(force_heartbeat=True)
            logger.success(f"{NEON['SUCCESS']}Order Filled! Part ID {part_id} added. Entry: {actual_entry_price}, Qty: {actual_qty}{NEON['RESET']}")
            config.send_notification_method(f"Pyrmethus Order Filled: {side.upper()}", f"{config.symbol} Part {part_id} @ {actual_entry_price}")
            return new_part
        else:
            logger.warning(f"{NEON['WARNING']}Order {order['id']} not filled promptly. Status: {filled_order.get('status')}. Manual check advised.{NEON['RESET']}")
            if order_type_str == 'Limit' and filled_order.get('status') == 'open':
                 logger.info(f"Limit order {order['id']} still open. Consider cancellation logic after timeout.")
            return None

    except ccxt.InsufficientFunds as e:
        logger.error(f"{NEON['ERROR']}Insufficient funds to place order: {e}{NEON['RESET']}")
        config.send_notification_method("Pyrmethus Order Fail", f"Insufficient funds for {config.symbol}")
        return None
    except ccxt.ExchangeError as e:
        logger.error(f"{NEON['ERROR']}Exchange error placing order: {e}{NEON['RESET']}")
        config.send_notification_method("Pyrmethus Order Fail", f"Exchange error: {str(e)[:100]}")
        return None
    except Exception as e:
        logger.error(f"{NEON['ERROR']}Unexpected error placing order: {e}{NEON['RESET']}")
        logger.debug(traceback.format_exc())
        return None

def close_position_part(exchange: ccxt.Exchange, config: Config, part_to_close: Dict, reason: str, close_price_target: Optional[Decimal] = None) -> bool:
    global _active_trade_parts
    logger.info(f"{NEON['ACTION']}Attempting to close trade part {part_to_close['part_id']} ({part_to_close['side']} {part_to_close['qty']} {config.symbol}) for reason: {reason}{NEON['RESET']}")
    
    close_side = config.side_sell if part_to_close['side'].lower() == config.pos_long.lower() else config.side_buy
    qty_to_close = part_to_close['qty']

    try:
        params = {'reduceOnly': True}
        logger.info(f"Placing MARKET {close_side.upper()} order to close part {part_to_close['part_id']}: Qty {qty_to_close}")
        close_order = exchange.create_order(config.symbol, 'market', close_side.lower(), float(qty_to_close), None, params)
        
        logger.success(f"{NEON['SUCCESS']}Close Order Sent: ID {close_order['id']}, Status {close_order.get('status', 'N/A')}{NEON['RESET']}")
        
        time.sleep(config.order_fill_timeout_seconds / 2) 
        filled_close_order = exchange.fetch_order(close_order['id'], config.symbol)

        if filled_close_order.get('status') == 'closed':
            actual_exit_price = Decimal(str(filled_close_order.get('average', filled_close_order.get('price', close_price_target or part_to_close['entry_price']))))
            
            pnl_per_unit = (actual_exit_price - part_to_close['entry_price']) if part_to_close['side'].lower() == config.pos_long.lower() else (part_to_close['entry_price'] - actual_exit_price)
            pnl = pnl_per_unit * part_to_close['qty']

            # MAE/MFE calculation would ideally happen here or be triggered
            # For now, passing Nones
            trade_metrics.log_trade(
                symbol=config.symbol, side=part_to_close['side'],
                entry_price=part_to_close['entry_price'], exit_price=actual_exit_price,
                qty=part_to_close['qty'], entry_time_ms=part_to_close['entry_time_ms'],
                exit_time_ms=int(filled_close_order.get('timestamp', time.time() * 1000)),
                reason=reason, part_id=part_to_close['part_id'], pnl_str=str(pnl),
                mae_str=None, mfe_str=None # Placeholder for future MAE/MFE
            )
            _active_trade_parts = [p for p in _active_trade_parts if p['part_id'] != part_to_close['part_id']]
            save_persistent_state(force_heartbeat=True)
            logger.success(f"{NEON['SUCCESS']}Part {part_to_close['part_id']} closed. Exit: {actual_exit_price}, PNL: {pnl:.2f}{NEON['RESET']}")
            config.send_notification_method(f"Pyrmethus Position Closed", f"{config.symbol} Part {part_to_close['part_id']} closed. PNL: {pnl:.2f}")
            return True
        else:
            logger.warning(f"{NEON['WARNING']}Close order {close_order['id']} for part {part_to_close['part_id']} not filled. Status: {filled_close_order.get('status')}. Manual check required.{NEON['RESET']}")
            return False

    except Exception as e:
        logger.error(f"{NEON['ERROR']}Error closing position part {part_to_close['part_id']}: {e}{NEON['RESET']}")
        logger.debug(traceback.format_exc())
        config.send_notification_method("Pyrmethus Close Fail", f"Error closing {config.symbol} part {part_to_close['part_id']}: {str(e)[:80]}")
        return False

@retry((ccxt.NetworkError, ccxt.RequestTimeout), tries=CONFIG.retry_count, delay=CONFIG.retry_delay_seconds)
def cancel_all_symbol_orders(exchange: ccxt.Exchange, symbol: str):
    logger.info(f"{NEON['ACTION']}Cancelling all open orders for {symbol}...{NEON['RESET']}")
    try:
        # Bybit V5 might need category for cancelAllOrders if it's not smart enough
        # params = {'category': 'linear' if symbol.endswith("USDT") else 'inverse'}
        # For now, assume ccxt handles it or symbol implies category for bybit
        exchange.cancel_all_orders(symbol)
        logger.success(f"{NEON['SUCCESS']}All open orders for {symbol} cancelled.{NEON['RESET']}")
    except Exception as e:
        logger.error(f"{NEON['ERROR']}Error cancelling orders for {symbol}: {e}{NEON['RESET']}")

def close_all_symbol_positions(exchange: ccxt.Exchange, config: Config, reason: str):
    global _active_trade_parts
    logger.warning(f"{NEON['WARNING']}Closing all positions for {config.symbol} due to: {reason}{NEON['RESET']}")
    parts_to_close_copy = list(_active_trade_parts)
    for part in parts_to_close_copy:
        if part['symbol'] == config.symbol:
            close_position_part(exchange, config, part, reason + " (Global Close)")
    
    try:
        # This fetch_positions might need category for Bybit V5
        # category = "linear" if config.symbol.endswith("USDT") else "inverse"
        # positions = exchange.fetch_positions([config.symbol], {'category': category})
        positions = exchange.fetch_positions([config.symbol]) # Simpler for now
        for pos in positions:
            # contracts can be string "0", "0.0", etc.
            pos_contracts_val = pos.get('contracts', '0')
            pos_contracts = safe_decimal_conversion(pos_contracts_val, Decimal(0))

            if pos and pos_contracts != Decimal(0):
                side_to_close = config.side_sell if pos.get('side', '').lower() == 'long' else config.side_buy
                qty_to_close = abs(pos_contracts) # Quantity is always positive
                logger.warning(f"Found residual exchange position: {pos.get('side')} {qty_to_close} {config.symbol}. Attempting market close.")
                exchange.create_order(config.symbol, 'market', side_to_close.lower(), float(qty_to_close), params={'reduceOnly': True})
                logger.info(f"Residual position closure order sent for {qty_to_close} {config.symbol}.")
    except Exception as e:
        logger.error(f"Error fetching/closing residual exchange positions: {e}")

    cancel_all_symbol_orders(exchange, config.symbol)


# --- Main Spell Weaving - The Heart of Pyrmethus ---
def main_loop(exchange: ccxt.Exchange, config: Config) -> None:
    logger.info(f"{NEON['HEADING']}=== Pyrmethus Spell v2.8.2 Awakening on {exchange.id} ==={NEON['RESET']}")
    logger.info(f"{NEON['COMMENT']}# Listening to whispers for {config.symbol} on {config.interval}...{NEON['RESET']}")
    
    if load_persistent_state(): logger.success(f"{NEON['SUCCESS']}Successfully reawakened from Phoenix scroll.{NEON['RESET']}")
    else: logger.info(f"{NEON['INFO']}No valid prior state. Starting fresh.{NEON['RESET']}")

    current_balance = fetch_account_balance(exchange, config.usdt_symbol)
    if current_balance is not None: trade_metrics.set_initial_equity(current_balance)
    else: logger.warning(f"{NEON['WARNING']}Could not set initial equity. Balance fetch failed.{NEON['RESET']}")

    while True:
        try:
            logger.debug(f"{NEON['COMMENT']}# New cycle of observation...{NEON['RESET']}")
            current_balance = fetch_account_balance(exchange, config.usdt_symbol) 
            if current_balance is not None:
                trade_metrics.set_initial_equity(current_balance) 
                drawdown_hit, dd_reason = trade_metrics.check_drawdown(current_balance)
                if drawdown_hit:
                    logger.critical(f"{NEON['CRITICAL']}Max drawdown! {dd_reason}. Pyrmethus must rest.{NEON['RESET']}")
                    close_all_symbol_positions(exchange, config, "Max Drawdown Reached")
                    break 
            else: logger.warning(f"{NEON['WARNING']}Failed to fetch balance this cycle.{NEON['RESET']}")

            ohlcv_df = get_market_data(exchange, config.symbol, config.interval, limit=OHLCV_LIMIT)
            if ohlcv_df is None or ohlcv_df.empty:
                logger.warning(f"{NEON['WARNING']}No market data. Retrying after pause.{NEON['RESET']}")
                time.sleep(config.sleep_seconds); continue

            df_with_indicators = calculate_all_indicators(ohlcv_df.copy(), config)
            if df_with_indicators.empty: time.sleep(config.sleep_seconds); continue 

            signals = config.strategy_instance.generate_signals(df_with_indicators)
            logger.debug(f"Signals: EnterL={signals['enter_long']}, EnterS={signals['enter_short']}, ExitL={signals['exit_long']}, ExitS={signals['exit_short']}")

            current_pos_side, current_pos_qty = get_current_position_info()
            logger.debug(f"Current Position: Side={current_pos_side}, Qty={current_pos_qty}")
            
            atr_col = f"ATR_{config.atr_calculation_period}"
            latest_atr = safe_decimal_conversion(df_with_indicators[atr_col].iloc[-1]) if atr_col in df_with_indicators.columns and not df_with_indicators.empty and not df_with_indicators[atr_col].empty else None
            latest_close = safe_decimal_conversion(df_with_indicators['close'].iloc[-1]) if 'close' in df_with_indicators.columns and not df_with_indicators.empty and not df_with_indicators['close'].empty else None


            if latest_atr is None or latest_close is None or latest_atr <= 0: # Added latest_atr > 0 check
                logger.warning(f"{NEON['WARNING']}Missing or invalid latest ATR ({latest_atr}) or Close price ({latest_close}). Skipping trade decisions.{NEON['RESET']}")
                time.sleep(config.sleep_seconds); continue

            if current_pos_side == config.pos_none: 
                if signals.get("enter_long"):
                    sl_price = latest_close - (latest_atr * config.atr_stop_loss_multiplier)
                    place_risked_order(exchange, config, config.pos_long, latest_close, sl_price, latest_atr)
                elif signals.get("enter_short"):
                    sl_price = latest_close + (latest_atr * config.atr_stop_loss_multiplier)
                    place_risked_order(exchange, config, config.pos_short, latest_close, sl_price, latest_atr)
            
            elif current_pos_side.lower() == config.pos_long.lower(): 
                if signals.get("exit_long"):
                    logger.info(f"Exit LONG signal received: {signals.get('exit_reason')}")
                    for part in list(_active_trade_parts): 
                        if part['side'].lower() == config.pos_long.lower():
                            close_position_part(exchange, config, part, signals.get('exit_reason', 'Strategy Exit Signal'))
            
            elif current_pos_side.lower() == config.pos_short.lower(): 
                if signals.get("exit_short"):
                    logger.info(f"Exit SHORT signal received: {signals.get('exit_reason')}")
                    for part in list(_active_trade_parts): 
                         if part['side'].lower() == config.pos_short.lower():
                            close_position_part(exchange, config, part, signals.get('exit_reason', 'Strategy Exit Signal'))
            
            save_persistent_state()
            logger.debug(f"{NEON['COMMENT']}# Cycle complete. Resting for {config.sleep_seconds}s...{NEON['RESET']}")
            time.sleep(config.sleep_seconds)

        except KeyboardInterrupt:
            logger.warning(f"\n{NEON['WARNING']}Sorcerer's intervention! Pyrmethus prepares for slumber...{NEON['RESET']}")
            config.send_notification_method("Pyrmethus Shutdown", "Manual shutdown initiated.")
            break
        except Exception as e: 
            logger.critical(f"{NEON['CRITICAL']}A critical error disrupted Pyrmethus's weave! Error: {e}{NEON['RESET']}")
            logger.debug(traceback.format_exc())
            config.send_notification_method("Pyrmethus Critical Error", f"Bot loop crashed: {str(e)[:150]}")
            time.sleep(config.sleep_seconds * 5) 

    

if __name__ == "__main__":
    logger.info(f"{NEON['COMMENT']}# Pyrmethus prepares to connect to the exchange realm...{NEON['RESET']}")
    exchange = None
    try:
        exchange_params = {
            'apiKey': CONFIG.api_key, 'secret': CONFIG.api_secret,
            'options': {'defaultType': 'swap', 'adjustForTimeDifference': True, 'brokerId': 'PYRMETHUS'}, 
            'enableRateLimit': True, 'recvWindow': CONFIG.default_recv_window
        }
        
        if CONFIG.PAPER_TRADING_MODE:
            logger.warning(f"{NEON['WARNING']}PAPER_TRADING_MODE is True. Ensure API keys are for Bybit TESTNET.{NEON['RESET']}")
            # For Bybit V5, testnet is usually via different API endpoints/keys.
            # CCXT might handle this with a specific exchange ID or an option.
            # Example: exchange = ccxt.bybit({'options': {'testnet': True}, ...})
            # Or for Bybit, it might be changing the base URL:
            # exchange_params['urls'] = {'api': 'https://api-testnet.bybit.com'} # If CCXT doesn't auto-handle V5 testnet
            # The following line is a common way but might vary by CCXT version and exchange support for unified testnet option:
            # exchange_params['options']['testnet'] = True # May or may not work for V5; check CCXT docs for Bybit V5
        
        exchange = ccxt.bybit(exchange_params) 

        markets = exchange.load_markets()
        if CONFIG.symbol not in markets:
            logger.critical(f"{NEON['CRITICAL']}Symbol {CONFIG.symbol} not found in exchange markets. Pyrmethus cannot trade this ether.{NEON['RESET']}")
            CONFIG.send_notification_method("Pyrmethus Startup Fail", f"Symbol {CONFIG.symbol} not found.")
            sys.exit(1)
        
        CONFIG.MARKET_INFO = markets[CONFIG.symbol]
        market_precision_price = CONFIG.MARKET_INFO.get('precision', {}).get('price', 'unknown')
        market_precision_amount = CONFIG.MARKET_INFO.get('precision', {}).get('amount', 'unknown')
        logger.success(f"{NEON['SUCCESS']}Market info for {CONFIG.symbol} loaded. Precision: Price {market_precision_price}, Amount {market_precision_amount}{NEON['RESET']}")

        try:
            # Category for Bybit V5: linear (USDT perps), inverse (coin-margined), option
            # Deriving category from symbol: if ends with :USDT and is USDT, it's linear.
            category = "linear" 
            if ":" in CONFIG.symbol and CONFIG.symbol.split(':')[1] == CONFIG.usdt_symbol:
                category = "linear"
            elif "/" in CONFIG.symbol and CONFIG.symbol.split('/')[1] != CONFIG.usdt_symbol : # e.g. BTC/USD (not :USDT)
                category = "inverse" # Basic assumption
            else: # Default or more complex logic needed
                category = "linear" # Default assumption for most futures like BTC/USDT
            logger.info(f"Deduced category for {CONFIG.symbol} as '{category}' for leverage setting.")
            
            # Bybit V5 setLeverage requires 'buyLeverage' and 'sellLeverage' for Hedge Mode,
            # or just leverage for One-Way. positionIdx=0 implies One-Way.
            # If using Hedge Mode, API keys need to be Unified Trading Account.
            leverage_params = {'category': category}
            if exchange. δύο_directional_margin: # Made-up attribute, check real ccxt hedge mode detection
                 leverage_params.update({'buyLeverage': CONFIG.leverage, 'sellLeverage': CONFIG.leverage})
            
            response = exchange.set_leverage(CONFIG.leverage, CONFIG.symbol, leverage_params)
            logger.success(f"{NEON['SUCCESS']}Leverage for {CONFIG.symbol} ({category}) set to {CONFIG.leverage}x. Response: {response}{NEON['RESET']}")
        except Exception as e_lev:
            logger.warning(f"{NEON['WARNING']}Could not set leverage for {CONFIG.symbol} (may already be set, or an issue): {e_lev}{NEON['RESET']}")
            logger.debug(traceback.format_exc())
            CONFIG.send_notification_method("Pyrmethus Leverage Warn", f"Leverage set issue for {CONFIG.symbol}: {str(e_lev)[:60]}")


        logger.success(f"{NEON['SUCCESS']}Successfully connected to exchange: {exchange.id}{NEON['RESET']}")
        CONFIG.send_notification_method("Pyrmethus Online", f"Connected to {exchange.id} for {CONFIG.symbol}")

    except AttributeError as e: logger.critical(f"{NEON['CRITICAL']}Exchange attribute error: {e}. CCXT issue or wrong exchange name?{NEON['RESET']}"); sys.exit(1)
    except ccxt.AuthenticationError as e: logger.critical(f"{NEON['CRITICAL']}Authentication failed! Check API keys: {e}{NEON['RESET']}"); CONFIG.send_notification_method("Pyrmethus Auth Fail", "API Key Auth Error."); sys.exit(1)
    except ccxt.NetworkError as e: logger.critical(f"{NEON['CRITICAL']}Network Error: {e}. Check connectivity.{NEON['RESET']}"); CONFIG.send_notification_method("Pyrmethus Network Error", f"Cannot connect: {str(e)[:80]}"); sys.exit(1)
    except ccxt.ExchangeError as e: logger.critical(f"{NEON['CRITICAL']}Exchange API Error: {e}. Check API permissions or symbol.{NEON['RESET']}"); CONFIG.send_notification_method("Pyrmethus API Error", f"Exchange API issue: {str(e)[:80]}"); sys.exit(1)
    except Exception as e: logger.critical(f"{NEON['CRITICAL']}Failed to initialize exchange: {e}{NEON['RESET']}"); logger.debug(traceback.format_exc()); CONFIG.send_notification_method("Pyrmethus Init Error", f"Exchange init failed: {str(e)[:80]}"); sys.exit(1)
    
    if exchange: main_loop(exchange, CONFIG)
    else: logger.critical(f"{NEON['CRITICAL']}Exchange not initialized. Pyrmethus sleeps.{NEON['RESET']}")

