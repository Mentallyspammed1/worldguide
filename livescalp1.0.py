# sxs.py
# Enhanced and Upgraded Scalping Bot Framework
# Derived from xrscalper.py, focusing on robust execution, error handling,
# advanced position management (BE, TSL), and Bybit V5 compatibility.

import json
import logging
import os
import time
from datetime import datetime, timezone
from decimal import ROUND_DOWN, ROUND_UP, Decimal, InvalidOperation, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo  # Use zoneinfo for modern timezone handling

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta  # Import pandas_ta
import requests
from colorama import Fore, Style, init
from dotenv import load_dotenv

# --- Initialization ---
init(autoreset=True)  # Ensure colorama resets styles automatically
load_dotenv()  # Load environment variables from .env file

# Set Decimal precision (high precision for financial calculations)
# Consider adjusting based on performance vs. required precision needs.
getcontext().prec = 36

# --- Neon Color Scheme ---
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.CYAN
RESET = Style.RESET_ALL

# --- Constants ---
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    # Log error and raise immediately if keys are missing
    # Use a basic handler for this critical startup error before full logging is set up
    logging.basicConfig(level=logging.CRITICAL, format="%(levelname)s: %(message)s")
    logging.critical(
        f"{NEON_RED}CRITICAL: BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env file.{RESET}"
    )
    raise ValueError(
        "BYBIT_API_KEY and BYBIT_API_SECRET environment variables are not set."
    )

CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
# Ensure the log directory exists early
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# Default timezone (IANA format), can be overridden by config
DEFAULT_TIMEZONE_STR = "America/Chicago"
TIMEZONE = ZoneInfo(DEFAULT_TIMEZONE_STR)

# Default API retry settings (can be overridden by config)
DEFAULT_MAX_API_RETRIES = 5
DEFAULT_RETRY_DELAY_SECONDS = 7
# Default loop delays (can be overridden by config)
DEFAULT_LOOP_DELAY_SECONDS = 10
DEFAULT_POSITION_CONFIRM_DELAY_SECONDS = 10

VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP = {
    "1": "1m",
    "3": "3m",
    "5": "5m",
    "15": "15m",
    "30": "30m",
    "60": "1h",
    "120": "2h",
    "240": "4h",
    "D": "1d",
    "W": "1w",
    "M": "1M",
}
# HTTP status codes considered retryable for generic network issues
RETRYABLE_HTTP_CODES = [429, 500, 502, 503, 504]

# Default indicator periods (can be overridden by config.json) - Standardized Names
DEFAULT_ATR_PERIOD = 14
DEFAULT_CCI_PERIOD = 20
DEFAULT_WILLIAMS_R_PERIOD = 14
DEFAULT_MFI_PERIOD = 14
DEFAULT_STOCH_RSI_PERIOD = 14
DEFAULT_STOCH_RSI_RSI_PERIOD = 14
DEFAULT_STOCH_RSI_K_PERIOD = 3
DEFAULT_STOCH_RSI_D_PERIOD = 3
DEFAULT_RSI_PERIOD = 14
DEFAULT_BBANDS_PERIOD = 20
DEFAULT_BBANDS_STDDEV = 2.0
DEFAULT_SMA10_PERIOD = 10
DEFAULT_EMA_SHORT_PERIOD = 9
DEFAULT_EMA_LONG_PERIOD = 21
DEFAULT_MOMENTUM_PERIOD = 7
DEFAULT_VOLUME_MA_PERIOD = 15
DEFAULT_FIB_PERIOD = 50
DEFAULT_PSAR_STEP = 0.02
DEFAULT_PSAR_MAX_STEP = 0.2

FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]  # Standard Fibonacci levels

# --- Configuration Loading & Validation ---


class SensitiveFormatter(logging.Formatter):
    """Formatter to redact sensitive information (API keys) from logs."""

    _patterns: Dict[str, str] = {}  # Cache patterns for slight performance gain

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record, redacting sensitive data."""
        msg = super().format(record)
        # Redact API Key if present and not already cached
        if API_KEY and API_KEY not in self._patterns:
            self._patterns[API_KEY] = "***API_KEY***"
        if API_KEY and API_KEY in msg:
            msg = msg.replace(API_KEY, self._patterns[API_KEY])

        # Redact API Secret if present and not already cached
        if API_SECRET and API_SECRET not in self._patterns:
            self._patterns[API_SECRET] = "***API_SECRET***"
        if API_SECRET and API_SECRET in msg:
            msg = msg.replace(API_SECRET, self._patterns[API_SECRET])
        return msg


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file, creating default if not found,
    ensuring all default keys are present with validation, and saving updates.
    Returns the validated configuration dictionary.
    """
    global TIMEZONE  # Allow updating the global timezone constant

    # Define the default configuration structure and values
    default_config = {
        # Trading pair and timeframe
        "symbol": "BTC/USDT:USDT",  # Bybit linear perpetual example
        "interval": "5",  # Default timeframe (e.g., "5" for 5 minutes)
        # API and Bot Behavior
        "retry_delay": DEFAULT_RETRY_DELAY_SECONDS,  # Delay between API retries
        "max_api_retries": DEFAULT_MAX_API_RETRIES,  # Max retries for API calls
        "enable_trading": False,  # Safety Feature: Must be explicitly set to true to trade
        "use_sandbox": True,  # Safety Feature: Use testnet by default
        "max_concurrent_positions": 1,  # Max open positions (current logic supports 1)
        "quote_currency": "USDT",  # Quote currency (ensure matches symbol quote)
        "position_confirm_delay_seconds": DEFAULT_POSITION_CONFIRM_DELAY_SECONDS,
        "loop_delay_seconds": DEFAULT_LOOP_DELAY_SECONDS,
        "timezone": DEFAULT_TIMEZONE_STR,  # IANA timezone name
        "bybit_account_type": "CONTRACT",  # V5 Account Type ('CONTRACT', 'UNIFIED', 'SPOT')
        "bybit_category": "linear",  # V5 Category ('linear', 'inverse', 'spot')
        # Risk Management
        "risk_per_trade": 0.01,  # Fraction of balance to risk (0.01 = 1%)
        "leverage": 20,  # Desired leverage
        "stop_loss_multiple": 1.8,  # ATR multiple for initial SL
        "take_profit_multiple": 0.7,  # ATR multiple for initial TP
        # Order Execution
        "entry_order_type": "market",  # "market" or "limit"
        "limit_order_offset_buy": 0.0005,  # % offset for BUY limit (0.05%)
        "limit_order_offset_sell": 0.0005,  # % offset for SELL limit
        # Advanced Position Management
        "enable_trailing_stop": True,  # Use exchange-native Trailing Stop Loss
        "trailing_stop_callback_rate": 0.005,  # % to calc trail distance from activation price
        "trailing_stop_activation_percentage": 0.003,  # % profit move to calc TSL activation price
        "enable_break_even": True,  # Move SL to break-even
        "break_even_trigger_atr_multiple": 1.0,  # Move SL when profit >= X * ATR
        "break_even_offset_ticks": 2,  # Place BE SL X ticks beyond entry
        "time_based_exit_minutes": None,  # Optional: Exit after X minutes
        # Indicator Periods & Parameters
        "atr_period": DEFAULT_ATR_PERIOD,
        "ema_short_period": DEFAULT_EMA_SHORT_PERIOD,
        "ema_long_period": DEFAULT_EMA_LONG_PERIOD,
        "rsi_period": DEFAULT_RSI_PERIOD,
        "bollinger_bands_period": DEFAULT_BBANDS_PERIOD,
        "bollinger_bands_std_dev": DEFAULT_BBANDS_STDDEV,
        "cci_period": DEFAULT_CCI_PERIOD,
        "williams_r_period": DEFAULT_WILLIAMS_R_PERIOD,
        "mfi_period": DEFAULT_MFI_PERIOD,
        "stoch_rsi_period": DEFAULT_STOCH_RSI_PERIOD,
        "stoch_rsi_rsi_period": DEFAULT_STOCH_RSI_RSI_PERIOD,
        "stoch_rsi_k_period": DEFAULT_STOCH_RSI_K_PERIOD,
        "stoch_rsi_d_period": DEFAULT_STOCH_RSI_D_PERIOD,
        "psar_step": DEFAULT_PSAR_STEP,
        "psar_max_step": DEFAULT_PSAR_MAX_STEP,
        "sma_10_period": DEFAULT_SMA10_PERIOD,
        "momentum_period": DEFAULT_MOMENTUM_PERIOD,
        "volume_ma_period": DEFAULT_VOLUME_MA_PERIOD,
        "fibonacci_period": DEFAULT_FIB_PERIOD,
        # Indicator Calculation & Scoring Control
        "orderbook_limit": 25,  # Depth of order book levels
        "signal_score_threshold": 1.5,  # Weighted score needed for signal
        "stoch_rsi_oversold_threshold": 25,
        "stoch_rsi_overbought_threshold": 75,
        "volume_confirmation_multiplier": 1.5,  # Vol > Multiplier * VolMA
        "indicators": {  # Toggle calculation and scoring contribution
            "ema_alignment": True,
            "momentum": True,
            "volume_confirmation": True,
            "stoch_rsi": True,
            "rsi": True,
            "bollinger_bands": True,
            "vwap": True,
            "cci": True,
            "wr": True,
            "psar": True,
            "sma_10": True,
            "mfi": True,
            "orderbook": True,
        },
        "weight_sets": {  # Define scoring weights for different strategies
            "scalping": {
                "ema_alignment": 0.2,
                "momentum": 0.3,
                "volume_confirmation": 0.2,
                "stoch_rsi": 0.6,
                "rsi": 0.2,
                "bollinger_bands": 0.3,
                "vwap": 0.4,
                "cci": 0.3,
                "wr": 0.3,
                "psar": 0.2,
                "sma_10": 0.1,
                "mfi": 0.2,
                "orderbook": 0.15,
            },
            "default": {
                "ema_alignment": 0.3,
                "momentum": 0.2,
                "volume_confirmation": 0.1,
                "stoch_rsi": 0.4,
                "rsi": 0.3,
                "bollinger_bands": 0.2,
                "vwap": 0.3,
                "cci": 0.2,
                "wr": 0.2,
                "psar": 0.3,
                "sma_10": 0.1,
                "mfi": 0.2,
                "orderbook": 0.1,
            },
            # Ensure keys match "indicators" and _check_ methods
        },
        "active_weight_set": "default",  # Select the active weight set
    }

    config = default_config.copy()  # Start with defaults
    needs_saving = False  # Flag to track if the file needs updating

    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                loaded_config = json.load(f)
            # Merge loaded config over defaults, ensuring all default keys exist
            config = _merge_configs(loaded_config, default_config)
            print(f"{NEON_GREEN}Loaded configuration from {filepath}{RESET}")
            # Check if merge introduced changes needing save
            if config != loaded_config:  # Simple check
                needs_saving = True
                print(
                    f"{NEON_YELLOW}Configuration merged with new defaults/structure.{RESET}"
                )

        except (json.JSONDecodeError, IOError) as e:
            print(
                f"{NEON_RED}Error loading config file {filepath}: {e}. Using default config.{RESET}"
            )
            config = default_config
            needs_saving = True
        except Exception as e:
            print(
                f"{NEON_RED}Unexpected error loading config {filepath}: {e}. Using default config.{RESET}"
            )
            config = default_config
            needs_saving = True
    else:
        print(
            f"{NEON_YELLOW}Config file not found. Creating default config at {filepath}{RESET}"
        )
        config = default_config
        needs_saving = True

    # --- Validation Section ---
    current_config = config
    original_config_before_validation = config.copy()

    # Helper for validation logging and default setting
    def validate_param(key, default_value, validation_func, error_msg_format):
        is_valid = False
        current_value = current_config.get(key)
        try:
            if key in current_config and validation_func(current_value):
                is_valid = True
            else:
                current_config[key] = default_value
                value_repr = (
                    repr(current_value) if current_value is not None else "None"
                )
                print(
                    f"{NEON_RED}{error_msg_format.format(key=key, value=value_repr, default=default_value)}{RESET}"
                )
        except Exception as validation_err:
            print(
                f"{NEON_RED}Error validating '{key}' ({repr(current_value)}): {validation_err}. Resetting to default '{default_value}'.{RESET}"
            )
            current_config[key] = default_value
        return is_valid

    # Validate symbol (non-empty string)
    validate_param(
        "symbol",
        default_config["symbol"],
        lambda v: isinstance(v, str) and v.strip(),
        "CRITICAL: Config key '{key}' is missing, empty, or invalid ({value}). Resetting to default: '{default}'.",
    )

    # Validate interval
    validate_param(
        "interval",
        default_config["interval"],
        lambda v: v in VALID_INTERVALS,
        "Invalid interval '{value}' for '{key}'. Resetting to default '{default}'. Valid: "
        + str(VALID_INTERVALS)
        + ".",
    )

    # Validate entry order type
    validate_param(
        "entry_order_type",
        default_config["entry_order_type"],
        lambda v: v in ["market", "limit"],
        "Invalid entry_order_type '{value}' for '{key}'. Resetting to '{default}'.",
    )

    # Validate Bybit Account Type and Category
    validate_param(
        "bybit_account_type",
        default_config["bybit_account_type"],
        lambda v: isinstance(v, str) and v.upper() in ["CONTRACT", "UNIFIED", "SPOT"],
        "Invalid bybit_account_type '{value}' for '{key}'. Must be 'CONTRACT', 'UNIFIED', or 'SPOT'. Resetting to '{default}'.",
    )
    current_config["bybit_account_type"] = current_config[
        "bybit_account_type"
    ].upper()  # Ensure uppercase

    validate_param(
        "bybit_category",
        default_config["bybit_category"],
        lambda v: isinstance(v, str) and v.lower() in ["linear", "inverse", "spot"],
        "Invalid bybit_category '{value}' for '{key}'. Must be 'linear', 'inverse', or 'spot'. Resetting to '{default}'.",
    )
    current_config["bybit_category"] = current_config[
        "bybit_category"
    ].lower()  # Ensure lowercase

    # Validate timezone
    try:
        tz_str = current_config.get("timezone", default_config["timezone"])
        tz_info = ZoneInfo(tz_str)
        current_config["timezone"] = tz_str  # Store the valid string
        TIMEZONE = tz_info  # Update global constant
    except Exception as tz_err:
        print(
            f"{NEON_RED}Invalid timezone '{tz_str}' in config: {tz_err}. Resetting to default '{default_config['timezone']}'.{RESET}"
        )
        current_config["timezone"] = default_config["timezone"]
        TIMEZONE = ZoneInfo(default_config["timezone"])

    # Validate active weight set exists
    validate_param(
        "active_weight_set",
        default_config["active_weight_set"],
        lambda v: isinstance(v, str) and v in current_config.get("weight_sets", {}),
        "Active weight set '{value}' for '{key}' not found in 'weight_sets'. Resetting to '{default}'.",
    )

    # Validate numeric parameters (using Decimal for robust range checks)
    numeric_params = {
        # key: (min_val, max_val, allow_min_equal, allow_max_equal, is_integer, default_val)
        "risk_per_trade": (
            "0",
            "1",
            False,
            False,
            False,
            default_config["risk_per_trade"],
        ),
        "leverage": ("1", "1000", True, True, True, default_config["leverage"]),
        "stop_loss_multiple": (
            "0",
            "inf",
            False,
            True,
            False,
            default_config["stop_loss_multiple"],
        ),
        "take_profit_multiple": (
            "0",
            "inf",
            False,
            True,
            False,
            default_config["take_profit_multiple"],
        ),
        "trailing_stop_callback_rate": (
            "0",
            "1",
            False,
            False,
            False,
            default_config["trailing_stop_callback_rate"],
        ),
        "trailing_stop_activation_percentage": (
            "0",
            "1",
            True,
            False,
            False,
            default_config["trailing_stop_activation_percentage"],
        ),
        "break_even_trigger_atr_multiple": (
            "0",
            "inf",
            False,
            True,
            False,
            default_config["break_even_trigger_atr_multiple"],
        ),
        "break_even_offset_ticks": (
            "0",
            "1000",
            True,
            True,
            True,
            default_config["break_even_offset_ticks"],
        ),
        "signal_score_threshold": (
            "0",
            "inf",
            False,
            True,
            False,
            default_config["signal_score_threshold"],
        ),
        "atr_period": ("2", "1000", True, True, True, default_config["atr_period"]),
        "ema_short_period": (
            "1",
            "1000",
            True,
            True,
            True,
            default_config["ema_short_period"],
        ),
        "ema_long_period": (
            "1",
            "1000",
            True,
            True,
            True,
            default_config["ema_long_period"],
        ),
        "rsi_period": ("2", "1000", True, True, True, default_config["rsi_period"]),
        "bollinger_bands_period": (
            "2",
            "1000",
            True,
            True,
            True,
            default_config["bollinger_bands_period"],
        ),
        "bollinger_bands_std_dev": (
            "0",
            "10",
            False,
            True,
            False,
            default_config["bollinger_bands_std_dev"],
        ),
        "cci_period": ("2", "1000", True, True, True, default_config["cci_period"]),
        "williams_r_period": (
            "2",
            "1000",
            True,
            True,
            True,
            default_config["williams_r_period"],
        ),
        "mfi_period": ("2", "1000", True, True, True, default_config["mfi_period"]),
        "stoch_rsi_period": (
            "2",
            "1000",
            True,
            True,
            True,
            default_config["stoch_rsi_period"],
        ),
        "stoch_rsi_rsi_period": (
            "2",
            "1000",
            True,
            True,
            True,
            default_config["stoch_rsi_rsi_period"],
        ),
        "stoch_rsi_k_period": (
            "1",
            "1000",
            True,
            True,
            True,
            default_config["stoch_rsi_k_period"],
        ),
        "stoch_rsi_d_period": (
            "1",
            "1000",
            True,
            True,
            True,
            default_config["stoch_rsi_d_period"],
        ),
        "psar_step": ("0", "1", False, True, False, default_config["psar_step"]),
        "psar_max_step": (
            "0",
            "1",
            False,
            True,
            False,
            default_config["psar_max_step"],
        ),
        "sma_10_period": (
            "1",
            "1000",
            True,
            True,
            True,
            default_config["sma_10_period"],
        ),
        "momentum_period": (
            "1",
            "1000",
            True,
            True,
            True,
            default_config["momentum_period"],
        ),
        "volume_ma_period": (
            "1",
            "1000",
            True,
            True,
            True,
            default_config["volume_ma_period"],
        ),
        "fibonacci_period": (
            "2",
            "1000",
            True,
            True,
            True,
            default_config["fibonacci_period"],
        ),
        "orderbook_limit": (
            "1",
            "200",
            True,
            True,
            True,
            default_config["orderbook_limit"],
        ),  # Bybit V5 linear up to 200
        "position_confirm_delay_seconds": (
            "0",
            "120",
            True,
            True,
            False,
            default_config["position_confirm_delay_seconds"],
        ),
        "loop_delay_seconds": (
            "1",
            "300",
            True,
            True,
            False,
            default_config["loop_delay_seconds"],
        ),
        "stoch_rsi_oversold_threshold": (
            "0",
            "100",
            True,
            False,
            False,
            default_config["stoch_rsi_oversold_threshold"],
        ),
        "stoch_rsi_overbought_threshold": (
            "0",
            "100",
            False,
            True,
            False,
            default_config["stoch_rsi_overbought_threshold"],
        ),
        "volume_confirmation_multiplier": (
            "0",
            "inf",
            False,
            True,
            False,
            default_config["volume_confirmation_multiplier"],
        ),
        "limit_order_offset_buy": (
            "0",
            "0.1",
            True,
            False,
            False,
            default_config["limit_order_offset_buy"],
        ),  # 10% max offset
        "limit_order_offset_sell": (
            "0",
            "0.1",
            True,
            False,
            False,
            default_config["limit_order_offset_sell"],
        ),
        "retry_delay": ("1", "120", True, True, False, default_config["retry_delay"]),
        "max_api_retries": (
            "0",
            "10",
            True,
            True,
            True,
            default_config["max_api_retries"],
        ),
        "max_concurrent_positions": (
            "1",
            "10",
            True,
            True,
            True,
            default_config["max_concurrent_positions"],
        ),
    }
    for key, (
        min_str,
        max_str,
        allow_min,
        allow_max,
        is_int,
        default_val,
    ) in numeric_params.items():
        value = current_config.get(key)
        is_valid = False
        if value is not None:
            try:
                val_dec = Decimal(str(value))
                if not val_dec.is_finite():
                    raise ValueError("Value not finite")

                min_dec = Decimal(min_str)
                max_dec = Decimal(max_str)
                lower_ok = (val_dec >= min_dec) if allow_min else (val_dec > min_dec)
                upper_ok = (val_dec <= max_dec) if allow_max else (val_dec < max_dec)

                if lower_ok and upper_ok:
                    if is_int:
                        if val_dec % 1 == 0:
                            current_config[key] = int(val_dec)
                            is_valid = True
                        else:
                            raise ValueError("Non-integer for integer param")
                    else:
                        current_config[key] = float(
                            val_dec
                        )  # Store validated non-integers as float
                        is_valid = True
            except (ValueError, TypeError, InvalidOperation):
                pass  # Handled below

        if not is_valid:
            err_msg = (
                f"Invalid value for '{{key}}' ({{value}}). Must be {'integer' if is_int else 'number'} "
                f"between {min_str} ({'inclusive' if allow_min else 'exclusive'}) and "
                f"{max_str} ({'inclusive' if allow_max else 'exclusive'}). Resetting to default '{{default}}'."
            )
            validate_param(key, default_val, lambda v: False, err_msg)  # Force reset

    # Validate time_based_exit_minutes (None or positive float/int)
    time_exit_key = "time_based_exit_minutes"
    time_exit_value = current_config.get(time_exit_key)
    time_exit_valid = False
    if time_exit_value is None:
        time_exit_valid = True
    else:
        try:
            time_exit_float = float(time_exit_value)
            if time_exit_float > 0:
                current_config[time_exit_key] = time_exit_float
                time_exit_valid = True
            else:
                raise ValueError("Must be positive")
        except (ValueError, TypeError):
            pass

    if not time_exit_valid:
        validate_param(
            time_exit_key,
            default_config[time_exit_key],
            lambda v: False,
            "Invalid value for '{{key}}' ({{value}}). Must be 'None' or positive. Resetting to default ('{{default}}').",
        )

    # Validate boolean parameters
    bool_params = [
        "enable_trading",
        "use_sandbox",
        "enable_trailing_stop",
        "enable_break_even",
    ]
    for key in bool_params:
        validate_param(
            key,
            default_config[key],
            lambda v: isinstance(v, bool),
            "Invalid value for '{{key}}' ({{value}}). Must be boolean. Resetting to default '{{default}}'.",
        )

    # Validate indicator enable flags (must be boolean)
    indicators_key = "indicators"
    if indicators_key in current_config and isinstance(
        current_config[indicators_key], dict
    ):
        indicators_dict = current_config[indicators_key]
        default_indicators = default_config[indicators_key]
        for ind_key, ind_val in list(
            indicators_dict.items()
        ):  # Use list copy for safe iteration/deletion
            if ind_key not in default_indicators:
                print(
                    f"{NEON_YELLOW}Warning: Unknown key '{ind_key}' found in '{indicators_key}'. Removing."
                )
                del current_config[indicators_key][ind_key]  # Remove unknown keys
                continue
            if not isinstance(ind_val, bool):
                default_ind_val = default_indicators.get(ind_key, False)
                print(
                    f"{NEON_RED}Invalid value for '{indicators_key}.{ind_key}' ({repr(ind_val)}). Must be boolean. Resetting to '{default_ind_val}'.{RESET}"
                )
                indicators_dict[ind_key] = default_ind_val
    else:
        print(
            f"{NEON_RED}Invalid or missing '{indicators_key}' section. Resetting to default.{RESET}"
        )
        current_config[indicators_key] = default_config[indicators_key].copy()

    # Validate weight sets structure and values
    ws_key = "weight_sets"
    if ws_key in current_config and isinstance(current_config[ws_key], dict):
        weight_sets = current_config[ws_key]
        default_indicators_keys = default_config["indicators"].keys()
        for set_name, weights in list(weight_sets.items()):  # Use list copy
            if not isinstance(weights, dict):
                print(
                    f"{NEON_RED}Invalid structure for weight set '{set_name}'. Removing."
                )
                del current_config[ws_key][set_name]
                continue
            for ind_key, weight_val in list(weights.items()):  # Use list copy
                if ind_key not in default_indicators_keys:
                    print(
                        f"{NEON_YELLOW}Warning: Weight defined for unknown indicator '{ind_key}' in set '{set_name}'. Removing.{RESET}"
                    )
                    del weights[ind_key]
                    continue
                try:
                    weight_dec = Decimal(str(weight_val))
                    if not weight_dec.is_finite() or weight_dec < 0:
                        raise ValueError("Non-negative finite")
                    weights[ind_key] = float(weight_dec)  # Store as float
                except (ValueError, TypeError, InvalidOperation):
                    default_weight = (
                        default_config[ws_key].get("default", {}).get(ind_key, 0.0)
                    )  # Fallback to default set, then 0
                    print(
                        f"{NEON_RED}Invalid weight '{repr(weight_val)}' for '{ind_key}' in set '{set_name}'. Resetting to default '{default_weight}'.{RESET}"
                    )
                    weights[ind_key] = float(default_weight)
    else:
        print(
            f"{NEON_RED}Invalid or missing '{ws_key}' section. Resetting to default.{RESET}"
        )
        current_config[ws_key] = default_config[ws_key].copy()

    # If config was updated during merge/validation or file creation, save it back
    if needs_saving or current_config != original_config_before_validation:
        try:
            with open(filepath, "w", encoding="utf-8") as f_write:
                json.dump(
                    current_config,
                    f_write,
                    indent=4,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            print(f"{NEON_YELLOW}Saved updated configuration to {filepath}{RESET}")
        except IOError as e:
            print(f"{NEON_RED}Error writing updated config file {filepath}: {e}{RESET}")
        except Exception as e:
            print(
                f"{NEON_RED}Unexpected error saving config file {filepath}: {e}{RESET}"
            )

    return current_config


def _merge_configs(loaded_config: Dict, default_config: Dict) -> Dict:
    """
    Recursively merges loaded configuration over default values.
    Ensures all default keys exist, prioritizing loaded values. Handles nested dicts.
    Adds keys present only in loaded_config.
    """
    merged = default_config.copy()

    for key, value in loaded_config.items():
        if key in merged and isinstance(value, dict) and isinstance(merged[key], dict):
            merged[key] = _merge_configs(value, merged[key])  # Recurse for nested dicts
        else:
            merged[key] = value  # Overwrite default or add new key

    # Ensure all keys from default are present (handles cases where a key was missing)
    for key, default_value in default_config.items():
        if key not in merged:
            merged[key] = default_value

    return merged


# --- Logging Setup ---
def setup_logger(
    name: str, config: Dict[str, Any], level: int = logging.INFO
) -> logging.Logger:
    """Sets up a logger with rotating file and colored console handlers based on config."""
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()  # Prevent duplicate handlers on re-setup

    logger.setLevel(logging.DEBUG)  # Capture all levels at the logger

    # --- File Handler (Rotating, UTC Timestamps) ---
    log_filename = os.path.join(LOG_DIRECTORY, f"{name}.log")
    try:
        os.makedirs(LOG_DIRECTORY, exist_ok=True)  # Ensure dir exists
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_formatter = SensitiveFormatter(
            "%(asctime)s.%(msecs)03dZ %(levelname)-8s [%(name)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",  # ISO 8601 format
        )
        file_formatter.converter = time.gmtime  # Use UTC time
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)  # Log DEBUG and above to file
        logger.addHandler(file_handler)
    except Exception as e:
        print(
            f"{NEON_RED}Error setting up file logger {log_filename}: {e}. File logging disabled.{RESET}"
        )
        # Add basic stream handler if no handlers exist at all (e.g., file failed)
        if not logger.hasHandlers():
            basic_handler = logging.StreamHandler()
            basic_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            logger.addHandler(basic_handler)

    # --- Console Handler (Colored, Local Time from Config) ---
    # Avoid duplicate console logs if file handler failed and added basic stream handler
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler()
        try:
            console_tz_str = config.get("timezone", DEFAULT_TIMEZONE_STR)
            console_tz = ZoneInfo(console_tz_str)
        except Exception:
            print(
                f"{NEON_RED}Invalid timezone '{console_tz_str}' for console logs. Using UTC.{RESET}"
            )
            console_tz = ZoneInfo("UTC")

        console_formatter = SensitiveFormatter(
            f"{NEON_BLUE}%(asctime)s{RESET} {NEON_YELLOW}%(levelname)-8s{RESET} {NEON_PURPLE}[%(name)s]{RESET} %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S %Z",  # Includes timezone abbreviation
        )

        # Custom converter for local time display
        def local_time_converter(*args):
            return datetime.now(console_tz).timetuple()

        console_formatter.converter = local_time_converter
        stream_handler.setFormatter(console_formatter)
        stream_handler.setLevel(level)  # Set console level (e.g., INFO)
        logger.addHandler(stream_handler)

    logger.propagate = False  # Prevent duplicate logs in root logger
    return logger


# --- CCXT Exchange Setup ---
def initialize_exchange(
    config: Dict[str, Any], logger: logging.Logger
) -> Optional[ccxt.Exchange]:
    """Initializes the CCXT Bybit exchange object with V5 defaults and enhanced error handling."""
    lg = logger
    try:
        exchange_options = {
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "enableRateLimit": True,
            "rateLimit": 150,  # Adjust based on V5 limits & VIP level
            "options": {
                "defaultType": config.get(
                    "bybit_category", "linear"
                ),  # Use category from config
                "adjustForTimeDifference": True,
                "fetchTickerTimeout": 15000,
                "fetchBalanceTimeout": 20000,
                "createOrderTimeout": 25000,
                "cancelOrderTimeout": 20000,
                "fetchPositionsTimeout": 25000,
                "fetchOHLCVTimeout": 20000,
                "fetchOrderBookTimeout": 15000,
                "setLeverageTimeout": 20000,
                "fetchMyTradesTimeout": 20000,
                "fetchClosedOrdersTimeout": 25000,
                "user-agent": "sxsBot/1.2 (+https://github.com/your_repo)",  # Optional: Update URL
                # Bybit V5 specific: Account type can be set globally or per-request
                # 'accountType': config.get('bybit_account_type', 'CONTRACT'), # Can set globally
            },
        }

        exchange_id = "bybit"  # Hardcoded for this specialized setup
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class(exchange_options)

        # --- Sandbox Mode Setup ---
        if config.get("use_sandbox", True):
            lg.warning(f"{NEON_YELLOW}INITIALIZING IN SANDBOX MODE (Testnet){RESET}")
            try:
                exchange.set_sandbox_mode(True)
                lg.info(
                    f"Sandbox mode enabled via set_sandbox_mode(True) for {exchange.id}."
                )
                # Verify URL (CCXT can be inconsistent)
                current_api_url = exchange.urls.get("api", "")
                if "testnet" not in current_api_url:
                    lg.warning(
                        f"set_sandbox_mode did not change API URL. Current: {current_api_url}"
                    )
                    # Attempt manual override
                    test_url = exchange.describe().get("urls", {}).get("test")
                    api_test_url = None
                    if isinstance(test_url, str):
                        api_test_url = test_url
                    elif isinstance(test_url, dict):
                        api_test_url = (
                            test_url.get("public")
                            or test_url.get("private")
                            or list(test_url.values())[0]
                        )

                    if api_test_url and isinstance(api_test_url, str):
                        exchange.urls["api"] = api_test_url
                        lg.info(
                            f"Manually set API URL to Testnet: {exchange.urls['api']}"
                        )
                    else:
                        fallback_test_url = "https://api-testnet.bybit.com"
                        lg.warning(
                            f"Could not determine testnet URL reliably. Hardcoding fallback: {fallback_test_url}"
                        )
                        exchange.urls["api"] = fallback_test_url
                else:
                    lg.info(
                        f"Confirmed API URL is set to Testnet: {exchange.urls['api']}"
                    )

            except AttributeError:
                lg.warning(
                    f"{exchange.id} ccxt version might not support set_sandbox_mode. Manually setting Testnet URL."
                )
                exchange.urls["api"] = "https://api-testnet.bybit.com"
                lg.info(
                    f"Manually set Bybit API URL to Testnet: {exchange.urls['api']}"
                )
            except Exception as e_sandbox:
                lg.error(
                    f"Error enabling sandbox mode: {e_sandbox}. Ensure API keys are for Testnet.",
                    exc_info=True,
                )
        else:
            lg.info(
                f"{NEON_GREEN}INITIALIZING IN LIVE (Real Money) Environment.{RESET}"
            )
            # Ensure API URL is production
            current_api_url = exchange.urls.get("api", "")
            if "testnet" in current_api_url:
                lg.warning(
                    f"Detected testnet URL in live mode ({current_api_url}). Resetting to production."
                )
                # Find production URL
                prod_url_info = exchange.describe().get("urls", {}).get("api")
                prod_url = None
                if isinstance(prod_url_info, str):
                    prod_url = prod_url_info
                elif isinstance(prod_url_info, dict):
                    prod_url = (
                        prod_url_info.get("public")
                        or prod_url_info.get("private")
                        or list(prod_url_info.values())[0]
                    )

                if prod_url and isinstance(prod_url, str):
                    exchange.urls["api"] = prod_url
                else:  # Fallback to 'www' or hardcoded if needed
                    www_url = exchange.describe().get("urls", {}).get("www")
                    fallback_prod_url = "https://api.bybit.com"
                    if www_url and isinstance(www_url, str):
                        exchange.urls["api"] = www_url
                        lg.info(
                            f"Reset API URL using 'www' URL: {exchange.urls['api']}"
                        )
                    else:
                        lg.warning(
                            f"Could not determine production URL. Hardcoding fallback: {fallback_prod_url}"
                        )
                        exchange.urls["api"] = fallback_prod_url
                lg.info(
                    f"Reset API URL to Production (best guess): {exchange.urls.get('api')}"
                )

        lg.info(
            f"Initializing {exchange.id} (API: {exchange.urls.get('api', 'URL Not Set')})..."
        )

        # --- Load Markets ---
        lg.info(f"Loading markets for {exchange.id}...")
        try:
            safe_api_call(exchange.load_markets, lg, reload=True)
            lg.info(
                f"Markets loaded successfully. Found {len(exchange.symbols)} symbols."
            )
            # Validate target symbol
            target_symbol = config.get("symbol")
            if target_symbol and target_symbol not in exchange.markets:
                lg.error(
                    f"{NEON_RED}FATAL: Target symbol '{target_symbol}' not found in loaded markets!{RESET}"
                )
                if "/" in target_symbol and ":" not in target_symbol:
                    base, quote = target_symbol.split("/")
                    suggested = f"{base}/{quote}:{quote}"
                    lg.warning(
                        f"{NEON_YELLOW}Hint: Bybit V5 linear format is like '{suggested}'.{RESET}"
                    )
                if 0 < len(exchange.symbols) < 50:
                    lg.debug(f"Available: {sorted(exchange.symbols)}")
                return None  # Fatal if symbol missing
            else:
                lg.info(f"Target symbol '{target_symbol}' found in loaded markets.")
                # Validate category vs market type
                market = exchange.market(target_symbol)
                market_category = (
                    "linear"
                    if market.get("linear")
                    else "inverse"
                    if market.get("inverse")
                    else "spot"
                )
                config_category = config.get("bybit_category")
                if market.get("spot") and config_category != "spot":
                    lg.warning(
                        f"Config category '{config_category}' mismatch for SPOT market '{target_symbol}'. API calls may fail."
                    )
                elif market.get("contract") and market_category != config_category:
                    lg.warning(
                        f"Config category '{config_category}' mismatch for {market_category.upper()} contract '{target_symbol}'. API calls may fail."
                    )

        except Exception as market_err:
            lg.critical(
                f"{NEON_RED}CRITICAL: Failed to load markets: {market_err}. Exiting.{RESET}",
                exc_info=True,
            )
            return None

        # --- Initial Connection Test (Fetch Balance for configured account type) ---
        account_type_to_test = config.get("bybit_account_type", "CONTRACT")
        lg.info(
            f"Performing initial connection test (Fetch Balance - Account: {account_type_to_test})..."
        )
        quote_curr = config.get("quote_currency", "USDT")
        balance_decimal = fetch_balance(
            exchange, quote_curr, lg, config
        )  # Pass config for account type

        if balance_decimal is not None:
            lg.info(
                f"{NEON_GREEN}Connected & fetched {quote_curr} balance: {balance_decimal:.4f}{RESET}"
            )
            if balance_decimal == 0:
                lg.warning(
                    f"{NEON_YELLOW}Initial {quote_curr} balance is zero in '{account_type_to_test}' account. Check funds.{RESET}"
                )
        else:
            lg.critical(
                f"{NEON_RED}CRITICAL: Initial balance fetch failed. Check API permissions, Account Type ({account_type_to_test}), network.{RESET}"
            )
            # Consider if fatal: return None

        lg.info(
            f"CCXT exchange initialized ({exchange.id}). Sandbox: {config.get('use_sandbox')}, "
            f"Default Type: {exchange.options.get('defaultType')}, Account Type: {account_type_to_test}"
        )
        return exchange

    except ccxt.AuthenticationError as e:
        lg.critical(f"{NEON_RED}CCXT Authentication Error: {e}{RESET}")
        lg.critical(
            f"{NEON_RED}>> Check API Key/Secret, permissions, IP whitelist (.env & exchange).{RESET}"
        )
    except ccxt.ExchangeError as e:
        lg.critical(f"{NEON_RED}CCXT Exchange Error: {e}{RESET}")
        lg.critical(f"{NEON_RED}>> Check exchange settings or API endpoints.{RESET}")
    except ccxt.NetworkError as e:
        lg.critical(f"{NEON_RED}CCXT Network Error: {e}{RESET}")
        lg.critical(f"{NEON_RED}>> Check internet connection/firewall.{RESET}")
    except Exception as e:
        lg.critical(
            f"{NEON_RED}Unexpected error initializing CCXT: {e}{RESET}", exc_info=True
        )

    return None


# --- API Call Wrapper with Retries ---
def safe_api_call(func, logger: logging.Logger, *args, **kwargs):
    """
    Wraps an API call with robust retry logic for network, rate limit, and specific errors.
    Uses exponential backoff with jitter.

    Args:
        func: The CCXT method or other callable to execute.
        logger: The logger instance.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Raises:
        ccxt.AuthenticationError: If authentication fails (non-retryable).
        ccxt.ExchangeError: If a non-retryable exchange error occurs.
        ccxt.RequestTimeout: If max retries are exceeded.
        Exception: For other unexpected errors during the call.

    Returns:
        The result of the wrapped function call.
    """
    lg = logger
    # Use global config if available, otherwise use defaults
    global config
    max_retries = (
        config.get("max_api_retries", DEFAULT_MAX_API_RETRIES)
        if "config" in globals()
        else DEFAULT_MAX_API_RETRIES
    )
    base_retry_delay = (
        config.get("retry_delay", DEFAULT_RETRY_DELAY_SECONDS)
        if "config" in globals()
        else DEFAULT_RETRY_DELAY_SECONDS
    )

    attempts = 0
    last_exception = None

    while attempts <= max_retries:
        wait_time = 0.0  # Initialize wait time for this attempt
        try:
            result = func(*args, **kwargs)
            lg.debug(f"API call '{func.__name__}' successful (Attempt {attempts + 1}).")
            return result

        # --- Retryable Network/Availability Errors ---
        except (
            ccxt.NetworkError,
            ccxt.RequestTimeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            ccxt.ExchangeNotAvailable,
            ccxt.DDoSProtection,
        ) as e:
            last_exception = e
            wait_time = base_retry_delay * (1.5**attempts)  # Exponential backoff
            wait_time *= 1 + (np.random.rand() - 0.5) * 0.2  # Add jitter +/- 10%
            wait_time = min(wait_time, 60)  # Cap wait time
            lg.warning(
                f"{NEON_YELLOW}Retryable network/availability error in '{func.__name__}': {type(e).__name__}. "
                f"Waiting {wait_time:.1f}s (Attempt {attempts + 1}/{max_retries + 1}). Error: {e}{RESET}"
            )

        # --- Rate Limit Errors ---
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            # Stronger exponential backoff for rate limits
            calculated_wait = base_retry_delay * (2.0**attempts)
            calculated_wait *= 1 + (np.random.rand() - 0.5) * 0.2  # Jitter

            # Check for 'Retry-After' header
            header_wait = 0
            retry_after_header = None
            if hasattr(e, "http_headers") and e.http_headers:
                retry_after_header = e.http_headers.get(
                    "Retry-After"
                ) or e.http_headers.get("retry-after")
            elif hasattr(e, "response") and hasattr(e.response, "headers"):
                retry_after_header = e.response.headers.get(
                    "Retry-After"
                ) or e.response.headers.get("retry-after")

            if retry_after_header:
                try:
                    header_wait = (
                        float(retry_after_header) + 0.5
                    )  # Assume seconds, add buffer
                    if header_wait > 1000:
                        header_wait = (header_wait / 1000.0) + 0.5  # Convert ms
                    lg.debug(
                        f"Rate limit Retry-After header: {retry_after_header} -> Parsed wait: {header_wait:.1f}s"
                    )
                except (ValueError, TypeError):
                    lg.warning(
                        f"Could not parse Retry-After header: {retry_after_header}"
                    )

            # Use the longer of calculated backoff or header wait, capped
            wait_time = max(calculated_wait, header_wait)
            wait_time = min(wait_time, 90)  # Cap rate limit wait

            lg.warning(
                f"{NEON_YELLOW}Rate limit exceeded in '{func.__name__}'. Waiting {wait_time:.1f}s "
                f"(Attempt {attempts + 1}/{max_retries + 1}). Error: {e}{RESET}"
            )

        # --- Authentication Errors (Non-Retryable) ---
        except ccxt.AuthenticationError as e:
            lg.error(
                f"{NEON_RED}Authentication Error in '{func.__name__}': {e}. Aborting call.{RESET}"
            )
            lg.error(
                f"{NEON_RED}>> Check API Key/Secret, permissions, IP whitelist, Environment (Live/Testnet).{RESET}"
            )
            raise e

        # --- Exchange Specific Errors (Check if Retryable) ---
        except ccxt.ExchangeError as e:
            last_exception = e
            err_str = str(e).lower()
            http_code = getattr(e, "http_status_code", None)
            exch_code = None  # Exchange-specific error code (e.g., Bybit retCode)

            # Try extracting Bybit's retCode
            try:
                # Example: "bybit {"retCode":10006,...}" or "[10006] message"
                if '"retcode":' in err_str:
                    code_str = err_str.split('"retcode":')[1].split(",")[0].strip()
                    if code_str.isdigit():
                        exch_code = int(code_str)
                elif err_str.startswith("[") and "]" in err_str:
                    code_str = err_str[1 : err_str.find("]")]
                    if code_str.isdigit():
                        exch_code = int(code_str)
            except Exception:
                pass

            # Define retryable conditions
            bybit_retry_codes = [
                10001,
                10002,
                10006,
                10016,
                10018,
                130021,
                130150,
                131204,
                170131,
            ]
            retryable_messages = [
                "internal server error",
                "service unavailable",
                "system busy",
                "matching engine busy",
                "please try again",
                "nonce is too small",
                "order placement optimization",
            ]  # Common transient messages

            is_retryable = (
                (exch_code in bybit_retry_codes)
                or (http_code in RETRYABLE_HTTP_CODES)
                or any(msg in err_str for msg in retryable_messages)
            )

            if is_retryable:
                wait_time = base_retry_delay * (1.5**attempts)  # Standard backoff
                wait_time *= 1 + (np.random.rand() - 0.5) * 0.2  # Jitter
                wait_time = min(wait_time, 60)  # Cap
                lg.warning(
                    f"{NEON_YELLOW}Potentially retryable exchange error in '{func.__name__}': {e} (Code: {exch_code}, HTTP: {http_code}). "
                    f"Waiting {wait_time:.1f}s (Attempt {attempts + 1}/{max_retries + 1})...{RESET}"
                )
            else:
                # Non-retryable exchange error
                lg.error(
                    f"{NEON_RED}Non-retryable Exchange Error in '{func.__name__}': {e} (Code: {exch_code}, HTTP: {http_code}){RESET}"
                )
                raise e

        # --- Catch any other unexpected error ---
        except Exception as e:
            last_exception = e
            lg.error(
                f"{NEON_RED}Unexpected error during API call '{func.__name__}': {e}{RESET}",
                exc_info=True,
            )
            raise e

        # --- Sleep and Increment Attempt ---
        if attempts < max_retries:  # Only sleep if we are going to retry
            if (
                wait_time <= 0
            ):  # Should not happen if an exception was caught, but safety check
                lg.warning(
                    f"Wait time invalid ({wait_time}) for retry attempt {attempts + 1}. Using base delay."
                )
                wait_time = base_retry_delay
            time.sleep(wait_time)
            attempts += 1
        else:  # Max retries reached, break loop (will raise error below)
            break

    # If loop completes, max retries exceeded
    lg.error(
        f"{NEON_RED}Max retries ({max_retries + 1}) exceeded for API call '{func.__name__}'. Last Error: {type(last_exception).__name__}{RESET}"
    )
    if last_exception:
        raise last_exception
    else:
        # Should not happen normally, but provide a generic error
        raise ccxt.RequestTimeout(
            f"Max retries exceeded for {func.__name__} (no specific exception captured)"
        )


# --- CCXT Data Fetching (Using safe_api_call) ---


def fetch_current_price_ccxt(
    exchange: ccxt.Exchange, symbol: str, logger: logging.Logger
) -> Optional[Decimal]:
    """Fetch current price using CCXT ticker with fallbacks, retries, and Decimal conversion."""
    lg = logger
    try:
        ticker = safe_api_call(exchange.fetch_ticker, lg, symbol)
        if not ticker:
            lg.error(f"Failed to fetch ticker for {symbol} (returned None or empty).")
            return None

        lg.debug(f"Raw Ticker data for {symbol}: {json.dumps(ticker, indent=2)}")

        def to_decimal(value, context_str: str = "price") -> Optional[Decimal]:
            """Safely converts value to a positive, finite Decimal."""
            if value is None:
                return None
            try:
                d = Decimal(str(value))
                if d.is_finite() and d > 0:
                    return d
                lg.debug(
                    f"Invalid {context_str} value (non-finite/non-positive): {value}."
                )
                return None
            except (InvalidOperation, ValueError, TypeError):
                lg.debug(f"Invalid {context_str} format: {value}.")
                return None

        # Price Extraction Logic with Priority: last > mark > close > average > mid > ask > bid
        p_last = to_decimal(ticker.get("last"), "last")
        p_mark = to_decimal(ticker.get("mark"), "mark")
        p_close = to_decimal(
            ticker.get("close", ticker.get("last")), "close/last"
        )  # Use 'close', fallback to 'last'
        p_bid = to_decimal(ticker.get("bid"), "bid")
        p_ask = to_decimal(ticker.get("ask"), "ask")
        p_avg = to_decimal(ticker.get("average"), "average")
        p_mid = (p_bid + p_ask) / 2 if p_bid and p_ask and p_bid < p_ask else None

        market_info = exchange.market(symbol) if symbol in exchange.markets else {}
        is_contract = market_info.get("contract", False) or market_info.get("type") in [
            "swap",
            "future",
        ]

        price_candidates = [
            (p_mark, "Mark Price (Contract)" if is_contract else "Mark Price"),
            (p_last, "Last Price"),
            (p_close, "Close Price"),
            (p_avg, "Average Price"),
            (p_mid, "Mid Price (Bid/Ask)"),
            (p_ask, "Ask Price (Fallback)"),
            (p_bid, "Bid Price (Last Resort)"),
        ]

        price = None
        source = "N/A"
        for p_val, p_src in price_candidates:
            if p_val is not None:
                # If using ask, check spread (simple warning)
                if p_src.startswith("Ask") and p_bid and p_ask:
                    spread_pct = (
                        ((p_ask - p_bid) / p_ask) * 100 if p_ask > 0 else Decimal("0")
                    )
                    if spread_pct > Decimal("2.0"):
                        lg.warning(
                            f"Using 'ask' ({p_ask}) as fallback, spread > 2% ({spread_pct:.2f}%)"
                        )
                price, source = p_val, p_src
                break

        if price is not None and price.is_finite() and price > 0:
            # Log with precision derived from the price itself (e.g., 4 decimal places)
            price_str = (
                f"{price:.{max(4, abs(price.normalize().as_tuple().exponent))}f}"
            )
            lg.info(f"Current price ({symbol}): {price_str} (Source: {source})")
            return price
        else:
            lg.error(
                f"{NEON_RED}Failed to extract valid price for {symbol}. Ticker: {ticker}{RESET}"
            )
            return None

    except Exception as e:
        lg.error(
            f"{NEON_RED}Error fetching/processing price for {symbol}: {e}{RESET}",
            exc_info=False,
        )
        return None


def fetch_klines_ccxt(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,  # User-provided interval (e.g., "5")
    limit: int = 250,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Fetch OHLCV kline data using CCXT with retries, validation, and Decimal conversion."""
    lg = logger or logging.getLogger(__name__)
    empty_df = pd.DataFrame(
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    ).set_index("timestamp")

    if not exchange.has["fetchOHLCV"]:
        lg.error(f"Exchange {exchange.id} does not support fetchOHLCV.")
        return empty_df

    ccxt_timeframe = CCXT_INTERVAL_MAP.get(timeframe)
    if not ccxt_timeframe:
        lg.error(
            f"Invalid timeframe '{timeframe}'. Valid: {list(VALID_INTERVALS)}. Cannot fetch klines."
        )
        return empty_df

    lg.debug(f"Fetching {limit} klines for {symbol} ({ccxt_timeframe})...")
    try:
        ohlcv_data = safe_api_call(
            exchange.fetch_ohlcv, lg, symbol, timeframe=ccxt_timeframe, limit=limit
        )

        if (
            ohlcv_data is None
            or not isinstance(ohlcv_data, list)
            or len(ohlcv_data) == 0
        ):
            if ohlcv_data is not None:  # Log only if empty list returned without error
                lg.warning(f"No kline data returned for {symbol} {ccxt_timeframe}.")
            return empty_df  # Error logged by safe_api_call if it failed

        df = pd.DataFrame(
            ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        # --- Data Cleaning and Type Conversion ---
        # 1. Timestamp to datetime (UTC), coerce errors, set index
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], unit="ms", errors="coerce", utc=True
        )
        initial_len = len(df)
        df.dropna(subset=["timestamp"], inplace=True)
        if len(df) < initial_len:
            lg.debug(f"Dropped {initial_len - len(df)} rows with invalid timestamps.")
        if df.empty:
            lg.warning("DataFrame empty after timestamp processing.")
            return empty_df
        df.set_index("timestamp", inplace=True)

        # 2. Convert OHLCV columns to Decimal, robustly handling errors/NaNs
        cols_to_convert = ["open", "high", "low", "close", "volume"]
        for col in cols_to_convert:
            if col not in df.columns:
                lg.warning(f"Column '{col}' missing.")
                continue
            try:
                # Convert to string first, then Decimal, replace errors/non-finite with NaN
                df[col] = (
                    df[col]
                    .astype(str)
                    .apply(
                        lambda x: Decimal(x)
                        if str(x).strip() and Decimal(x).is_finite()
                        else Decimal("NaN")
                    )
                )
                # Ensure prices>0, volume>=0
                is_price = col in ["open", "high", "low", "close"]
                df[col] = df[col].apply(
                    lambda d: d
                    if isinstance(d, Decimal) and (d > 0 if is_price else d >= 0)
                    else Decimal("NaN")
                )
            except (InvalidOperation, TypeError, ValueError) as conv_err:
                lg.error(
                    f"Error converting column '{col}' to Decimal: {conv_err}. Trying float fallback."
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")  # Fallback to float
                df[col] = df[col].apply(
                    lambda x: x if np.isfinite(x) else np.nan
                )  # Ensure float NaNs

        # 3. Drop rows with NaN in essential price columns (O, H, L, C)
        initial_len = len(df)
        essential_cols = ["open", "high", "low", "close"]
        df.dropna(subset=essential_cols, how="any", inplace=True)
        if len(df) < initial_len:
            lg.debug(f"Dropped {initial_len - len(df)} rows with NaN price data.")

        # 4. Check OHLC consistency (High >= Low, etc.)
        try:
            # Ensure columns are numeric-like (Decimal or float) before comparison
            invalid_ohlc = (
                (df["high"] < df["low"])
                | (df["high"] < df["open"])
                | (df["high"] < df["close"])
                | (df["low"] > df["open"])
                | (df["low"] > df["close"])
            )
            invalid_count = invalid_ohlc.sum()
            if invalid_count > 0:
                lg.warning(f"Found {invalid_count} inconsistent OHLC rows. Dropping.")
                df = df[~invalid_ohlc]
        except TypeError as cmp_err:
            lg.warning(
                f"Could not perform OHLC check (type error: {cmp_err}). Skipping."
            )
        except Exception as cmp_err:
            lg.warning(f"Error during OHLC check: {cmp_err}. Skipping.")

        if df.empty:
            lg.warning(f"Kline data for {symbol} became empty after cleaning.")
            return empty_df

        # 5. Sort by timestamp index and remove duplicates (keep last)
        df.sort_index(inplace=True)
        if df.index.has_duplicates:
            num_dupes = df.index.duplicated().sum()
            lg.debug(f"Found {num_dupes} duplicate timestamps. Keeping last.")
            df = df[~df.index.duplicated(keep="last")]

        lg.info(f"Fetched and processed {len(df)} klines for {symbol} {ccxt_timeframe}")
        if lg.isEnabledFor(logging.DEBUG) and not df.empty:
            lg.debug(f"Kline check: First:\n{df.head(1)}\nLast:\n{df.tail(1)}")
        return df

    except Exception as e:
        lg.error(
            f"{NEON_RED}Unexpected error fetching/processing klines for {symbol}: {e}{RESET}",
            exc_info=True,
        )
        return empty_df


def fetch_orderbook_ccxt(
    exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger
) -> Optional[Dict]:
    """Fetch orderbook using ccxt with retries, validation, and Decimal conversion."""
    lg = logger
    if not exchange.has["fetchOrderBook"]:
        lg.error(f"Exchange {exchange.id} does not support fetchOrderBook.")
        return None

    try:
        lg.debug(f"Fetching order book for {symbol} (limit {limit})...")
        orderbook = safe_api_call(exchange.fetch_order_book, lg, symbol, limit=limit)

        if not orderbook:  # Error logged by safe_api_call
            lg.warning(f"fetch_order_book for {symbol} returned None.")
            return None

        if (
            not isinstance(orderbook, dict)
            or "bids" not in orderbook
            or "asks" not in orderbook
            or not isinstance(orderbook["bids"], list)
            or not isinstance(orderbook["asks"], list)
        ):
            lg.warning(f"Invalid orderbook structure received: {orderbook}")
            return None

        # Convert prices/amounts to Decimal, validate format and values
        cleaned_book = {
            "bids": [],
            "asks": [],
            "timestamp": orderbook.get("timestamp"),
            "datetime": orderbook.get("datetime"),
            "nonce": orderbook.get("nonce"),
        }
        errors = 0
        for side in ["bids", "asks"]:
            for entry in orderbook[side]:
                if isinstance(entry, list) and len(entry) == 2:
                    try:
                        price = Decimal(str(entry[0]))
                        amount = Decimal(str(entry[1]))
                        if (
                            price.is_finite()
                            and price > 0
                            and amount.is_finite()
                            and amount >= 0
                        ):
                            cleaned_book[side].append([price, amount])
                        else:
                            errors += 1
                    except (InvalidOperation, ValueError, TypeError):
                        errors += 1
                else:
                    errors += 1  # Invalid format

        if errors > 0:
            lg.debug(
                f"Orderbook ({symbol}): Found {errors} invalid entries (format/value)."
            )

        # Sort bids descending, asks ascending (usually done by ccxt, but verify)
        cleaned_book["bids"].sort(key=lambda x: x[0], reverse=True)
        cleaned_book["asks"].sort(key=lambda x: x[0])

        if not cleaned_book["bids"] and not cleaned_book["asks"]:
            lg.warning(f"Orderbook for {symbol} is empty after cleaning.")
        elif not cleaned_book["bids"]:
            lg.warning(f"Orderbook ({symbol}) has no valid bids.")
        elif not cleaned_book["asks"]:
            lg.warning(f"Orderbook ({symbol}) has no valid asks.")

        lg.debug(
            f"Processed orderbook for {symbol} ({len(cleaned_book['bids'])} bids, {len(cleaned_book['asks'])} asks)."
        )
        return cleaned_book

    except Exception as e:
        lg.error(
            f"{NEON_RED}Error fetching/processing order book for {symbol}: {e}{RESET}",
            exc_info=False,
        )
        return None


# --- Trading Analyzer Class ---
class TradingAnalyzer:
    """
    Analyzes trading data using pandas_ta and generates weighted signals.
    Handles Decimal/float conversions and provides market precision helpers.
    """

    def __init__(
        self,
        df: pd.DataFrame,  # Expects OHLCV columns (Decimal), indexed by timestamp
        logger: logging.Logger,
        config: Dict[str, Any],
        market_info: Dict[str, Any],
    ) -> None:
        """
        Initializes the TradingAnalyzer.

        Args:
            df: Pandas DataFrame with OHLCV data (expects Decimals).
            logger: Logger instance.
            config: Bot configuration dictionary.
            market_info: Market details dictionary (precision, limits).
        """
        self.df = df.copy()  # Work on a copy
        self.logger = logger
        self.config = config
        self.market_info = market_info
        self.symbol = market_info.get("symbol", "UNKNOWN_SYMBOL")
        self.interval = config.get("interval", "N/A")
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval)

        # Stores latest indicator values (Decimal for price/ATR, float for others)
        self.indicator_values: Dict[str, Union[Decimal, float, None, datetime]] = {}
        # Stores generated pandas_ta column names mapped to internal keys
        self.ta_column_names: Dict[str, Optional[str]] = {}
        # Stores calculated Fibonacci levels (Decimal)
        self.fib_levels_data: Dict[str, Decimal] = {}
        # Active weight set and weights
        self.active_weight_set_name = config.get("active_weight_set", "default")
        self.weights = config.get("weight_sets", {}).get(
            self.active_weight_set_name, {}
        )
        if not self.weights:
            logger.warning(
                f"{NEON_YELLOW}Weight set '{self.active_weight_set_name}' empty for {self.symbol}. Score=0.{RESET}"
            )
            self.weights = {}

        # Cached precision values
        self._cached_price_precision: Optional[int] = None
        self._cached_min_tick_size: Optional[Decimal] = None
        self._cached_amount_precision: Optional[int] = None
        self._cached_min_amount_step: Optional[Decimal] = None

        self._initialize_analysis()

    def _initialize_analysis(self) -> None:
        """Checks DataFrame validity and runs initial calculations."""
        if self.df.empty:
            self.logger.warning(
                f"Analyzer: Empty DataFrame for {self.symbol}. No calculations."
            )
            return

        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in self.df.columns for col in required_cols):
            self.logger.error(
                f"Analyzer: DF missing columns {required_cols}. Cannot analyze."
            )
            self.df = pd.DataFrame()  # Clear invalid DF
            return
        if self.df[required_cols].isnull().all().all():
            self.logger.error("Analyzer: DF contains all NaNs. Cannot analyze.")
            self.df = pd.DataFrame()
            return

        try:
            self._calculate_all_indicators()
            self._update_latest_indicator_values()
            self.calculate_fibonacci_levels()
        except Exception as init_err:
            self.logger.error(
                f"Error during initial analysis for {self.symbol}: {init_err}",
                exc_info=True,
            )

    def _get_ta_col_name(
        self, base_name: str, result_df_columns: List[str]
    ) -> Optional[str]:
        """Helper to find the actual column name generated by pandas_ta."""
        if not result_df_columns:
            return None

        # Dynamically build expected patterns based on config
        bb_std_str = f"{float(self.config.get('bollinger_bands_std_dev', DEFAULT_BBANDS_STDDEV)):.1f}"
        psar_step = str(float(self.config.get("psar_step", DEFAULT_PSAR_STEP)))
        psar_max = str(float(self.config.get("psar_max_step", DEFAULT_PSAR_MAX_STEP)))
        param_keys = [
            ("atr_period", DEFAULT_ATR_PERIOD),
            ("ema_short_period", DEFAULT_EMA_SHORT_PERIOD),
            ("ema_long_period", DEFAULT_EMA_LONG_PERIOD),
            ("momentum_period", DEFAULT_MOMENTUM_PERIOD),
            ("cci_period", DEFAULT_CCI_PERIOD),
            ("williams_r_period", DEFAULT_WILLIAMS_R_PERIOD),
            ("mfi_period", DEFAULT_MFI_PERIOD),
            ("rsi_period", DEFAULT_RSI_PERIOD),
            ("bollinger_bands_period", DEFAULT_BBANDS_PERIOD),
            ("sma_10_period", DEFAULT_SMA10_PERIOD),
            ("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD),
            ("stoch_rsi_period", DEFAULT_STOCH_RSI_PERIOD),
            ("stoch_rsi_rsi_period", DEFAULT_STOCH_RSI_RSI_PERIOD),
            ("stoch_rsi_k_period", DEFAULT_STOCH_RSI_K_PERIOD),
            ("stoch_rsi_d_period", DEFAULT_STOCH_RSI_D_PERIOD),
        ]
        p = {
            key: self.config.get(key, default) for key, default in param_keys
        }  # Config params

        # Potential pandas_ta column names for each internal key
        patterns = {
            "ATR": [f"ATRr_{p['atr_period']}"],
            "EMA_Short": [f"EMA_{p['ema_short_period']}"],
            "EMA_Long": [f"EMA_{p['ema_long_period']}"],
            "Momentum": [f"MOM_{p['momentum_period']}"],
            "CCI": [f"CCI_{p['cci_period']}", f"CCI_{p['cci_period']}_0.015"],
            "Williams_R": [f"WILLR_{p['williams_r_period']}"],
            "MFI": [f"MFI_{p['mfi_period']}"],
            "VWAP": ["VWAP_D"],
            "PSAR_long": [f"PSARl_{psar_step}_{psar_max}"],
            "PSAR_short": [f"PSARs_{psar_step}_{psar_max}"],
            "SMA_10": [f"SMA_{p['sma_10_period']}"],
            "StochRSI_K": [
                f"STOCHRSIk_{p['stoch_rsi_period']}_{p['stoch_rsi_rsi_period']}_{p['stoch_rsi_k_period']}"
            ],
            "StochRSI_D": [
                f"STOCHRSId_{p['stoch_rsi_period']}_{p['stoch_rsi_rsi_period']}_{p['stoch_rsi_k_period']}_{p['stoch_rsi_d_period']}"
            ],
            "RSI": [f"RSI_{p['rsi_period']}"],
            "BB_Lower": [f"BBL_{p['bollinger_bands_period']}_{bb_std_str}"],
            "BB_Middle": [f"BBM_{p['bollinger_bands_period']}_{bb_std_str}"],
            "BB_Upper": [f"BBU_{p['bollinger_bands_period']}_{bb_std_str}"],
            "Volume_MA": [f"VOL_SMA_{p['volume_ma_period']}"],  # Custom name
        }

        patterns_to_check = patterns.get(base_name, [])
        if not patterns_to_check:
            return None

        # Search Strategy: Exact -> Case-Insensitive -> StartsWith -> Substring Fallback
        for pattern in patterns_to_check:  # 1. Exact Match
            if pattern in result_df_columns:
                return pattern

        cols_lower_map = {col.lower(): col for col in result_df_columns}
        for pattern in patterns_to_check:  # 2. Case-Insensitive Exact Match
            if pattern.lower() in cols_lower_map:
                return cols_lower_map[pattern.lower()]

        for pattern in patterns_to_check:  # 3. Starts With Match
            pattern_lower = pattern.lower()
            for col in result_df_columns:
                col_lower = col.lower()
                if col.startswith(pattern) or col_lower.startswith(pattern_lower):
                    suffix = col[len(pattern) :]
                    if not any(
                        c.isalpha() for c in suffix
                    ):  # Allow numbers/_/. in suffix
                        return col

        # 4. Fallback: Unique simple base name substring check
        simple_base = base_name.split("_")[0].lower()
        matches = [col for col in result_df_columns if simple_base in col.lower()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:  # Ambiguous, try to resolve using full patterns
            for pattern in patterns_to_check:
                if pattern in matches:
                    return pattern
            self.logger.warning(
                f"Ambiguous substring match for '{base_name}': {matches}. Cannot resolve."
            )
            return None

        self.logger.debug(
            f"Could not map indicator '{base_name}' (Patterns: {patterns_to_check})"
        )
        return None

    def _calculate_all_indicators(self):
        """Calculates enabled indicators using pandas_ta."""
        if self.df.empty:
            return self.logger.warning(
                f"DataFrame empty, cannot calculate indicators for {self.symbol}."
            )

        # Check minimum required data length
        required_periods = []
        indicators_cfg = self.config.get("indicators", {})
        weights = self.weights

        def add_req(ind_key, cfg_key, default):
            is_enabled = indicators_cfg.get(ind_key, False)
            weight = float(weights.get(ind_key, 0.0))
            if is_enabled and weight > 0:
                try:
                    period = int(self.config.get(cfg_key, default))
                    if period > 0:
                        required_periods.append(period)
                except (ValueError, TypeError):
                    pass  # Ignore invalid format

        add_req("atr", "atr_period", DEFAULT_ATR_PERIOD)
        add_req("momentum", "momentum_period", DEFAULT_MOMENTUM_PERIOD)
        add_req("cci", "cci_period", DEFAULT_CCI_PERIOD)
        # ... Add other period-based indicators ...
        add_req("volume_confirmation", "volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)
        if (
            indicators_cfg.get("ema_alignment", False)
            and float(weights.get("ema_alignment", 0.0)) > 0
        ):
            add_req("ema_alignment", "ema_short_period", DEFAULT_EMA_SHORT_PERIOD)
            add_req("ema_alignment", "ema_long_period", DEFAULT_EMA_LONG_PERIOD)
        # ... Add StochRSI ...

        min_required_data = max(required_periods) + 30 if required_periods else 50
        if len(self.df) < min_required_data:
            self.logger.warning(
                f"{NEON_YELLOW}Insufficient data ({len(self.df)} points) for {self.symbol} "
                f"(min recommended: {min_required_data}). Results may contain NaNs.{RESET}"
            )

        try:
            # Prepare DataFrame for pandas_ta (convert Decimal to float)
            original_types = {}
            df_calc = self.df  # Use alias
            cols_to_float = ["open", "high", "low", "close", "volume"]
            for col in cols_to_float:
                if col in df_calc.columns:
                    first_valid_idx = df_calc[col].first_valid_index()
                    if first_valid_idx is not None:
                        original_types[col] = type(df_calc.loc[first_valid_idx, col])
                        if original_types[col] == Decimal:
                            df_calc[col] = df_calc[col].apply(
                                lambda x: float(x)
                                if isinstance(x, Decimal) and x.is_finite()
                                else np.nan
                            )
                    else:
                        original_types[col] = None

            # Create pandas_ta Strategy
            ta_strategy = ta.Strategy(name="SXS_Strategy", ta=[])
            calculated_indicator_keys = set()

            # Define TA indicators based on map
            ta_map = {
                "atr": {
                    "kind": "atr",
                    "length": lambda: int(
                        self.config.get("atr_period", DEFAULT_ATR_PERIOD)
                    ),
                },
                "ema_short": {
                    "kind": "ema",
                    "length": lambda: int(
                        self.config.get("ema_short_period", DEFAULT_EMA_SHORT_PERIOD)
                    ),
                },
                "ema_long": {
                    "kind": "ema",
                    "length": lambda: int(
                        self.config.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD)
                    ),
                },
                "momentum": {
                    "kind": "mom",
                    "length": lambda: int(
                        self.config.get("momentum_period", DEFAULT_MOMENTUM_PERIOD)
                    ),
                },
                "cci": {
                    "kind": "cci",
                    "length": lambda: int(
                        self.config.get("cci_period", DEFAULT_CCI_PERIOD)
                    ),
                },
                "wr": {
                    "kind": "willr",
                    "length": lambda: int(
                        self.config.get("williams_r_period", DEFAULT_WILLIAMS_R_PERIOD)
                    ),
                },
                "mfi": {
                    "kind": "mfi",
                    "length": lambda: int(
                        self.config.get("mfi_period", DEFAULT_MFI_PERIOD)
                    ),
                },
                "sma_10": {
                    "kind": "sma",
                    "length": lambda: int(
                        self.config.get("sma_10_period", DEFAULT_SMA10_PERIOD)
                    ),
                },
                "rsi": {
                    "kind": "rsi",
                    "length": lambda: int(
                        self.config.get("rsi_period", DEFAULT_RSI_PERIOD)
                    ),
                },
                "vwap": {"kind": "vwap"},
                "psar": {
                    "kind": "psar",
                    "step": lambda: float(
                        self.config.get("psar_step", DEFAULT_PSAR_STEP)
                    ),
                    "max_step": lambda: float(
                        self.config.get("psar_max_step", DEFAULT_PSAR_MAX_STEP)
                    ),
                },
                "stoch_rsi": {
                    "kind": "stochrsi",
                    "length": lambda: int(
                        self.config.get("stoch_rsi_period", DEFAULT_STOCH_RSI_PERIOD)
                    ),
                    "rsi_length": lambda: int(
                        self.config.get(
                            "stoch_rsi_rsi_period", DEFAULT_STOCH_RSI_RSI_PERIOD
                        )
                    ),
                    "k": lambda: int(
                        self.config.get(
                            "stoch_rsi_k_period", DEFAULT_STOCH_RSI_K_PERIOD
                        )
                    ),
                    "d": lambda: int(
                        self.config.get(
                            "stoch_rsi_d_period", DEFAULT_STOCH_RSI_D_PERIOD
                        )
                    ),
                },
                "bollinger_bands": {
                    "kind": "bbands",
                    "length": lambda: int(
                        self.config.get("bollinger_bands_period", DEFAULT_BBANDS_PERIOD)
                    ),
                    "std": lambda: float(
                        self.config.get(
                            "bollinger_bands_std_dev", DEFAULT_BBANDS_STDDEV
                        )
                    ),
                },
            }

            # Add ATR always (needed for risk management)
            if "atr" in ta_map:
                try:
                    params = {k: v() for k, v in ta_map["atr"].items() if k != "kind"}
                    ta_strategy.ta.append(ta.Indicator(ta_map["atr"]["kind"], **params))
                    calculated_indicator_keys.add("atr")
                except Exception as e:
                    self.logger.error(f"Error preparing ATR: {e}")

            # Add others based on config/weights
            for key, is_enabled in indicators_cfg.items():
                if key == "atr":
                    continue
                weight = float(weights.get(key, 0.0))
                if not is_enabled or weight == 0.0:
                    continue

                if key == "ema_alignment":
                    for ema_key in ["ema_short", "ema_long"]:
                        if (
                            ema_key not in calculated_indicator_keys
                            and ema_key in ta_map
                        ):
                            try:
                                params = {
                                    k: v()
                                    for k, v in ta_map[ema_key].items()
                                    if k != "kind"
                                }
                                ta_strategy.ta.append(
                                    ta.Indicator(ta_map[ema_key]["kind"], **params)
                                )
                                calculated_indicator_keys.add(ema_key)
                            except Exception as e:
                                self.logger.error(f"Error preparing {ema_key}: {e}")
                elif key in ta_map:
                    if key not in calculated_indicator_keys:
                        try:
                            indicator_def = ta_map[key]
                            params = {
                                k: v() for k, v in indicator_def.items() if k != "kind"
                            }
                            ta_strategy.ta.append(
                                ta.Indicator(indicator_def["kind"], **params)
                            )
                            calculated_indicator_keys.add(key)
                        except Exception as e:
                            self.logger.error(f"Error preparing {key}: {e}")
                # Skip volume_confirmation, orderbook (handled separately)

            # Run the TA Strategy
            if ta_strategy.ta:
                self.logger.info(
                    f"Running pandas_ta strategy with {len(ta_strategy.ta)} indicators..."
                )
                try:
                    df_calc.ta.strategy(ta_strategy, append=True)
                except Exception as ta_err:
                    self.logger.error(
                        f"Error running TA strategy: {ta_err}", exc_info=True
                    )
            else:
                self.logger.info("No pandas_ta indicators added to strategy.")

            # Calculate Volume MA separately
            vol_key = "volume_confirmation"
            vol_ma_p = int(
                self.config.get("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)
            )
            if (
                indicators_cfg.get(vol_key, False)
                and float(weights.get(vol_key, 0.0)) > 0
                and vol_ma_p > 0
            ):
                try:
                    vol_ma_col = f"VOL_SMA_{vol_ma_p}"  # Custom name
                    if "volume" in df_calc.columns and pd.api.types.is_numeric_dtype(
                        df_calc["volume"]
                    ):
                        df_calc[vol_ma_col] = ta.sma(
                            df_calc["volume"].fillna(0), length=vol_ma_p
                        )
                        calculated_indicator_keys.add("volume_ma")
                    else:
                        self.logger.warning(
                            "Cannot calculate Volume MA (Volume column missing/invalid)."
                        )
                except Exception as vol_ma_err:
                    self.logger.error(f"Error calculating Volume MA: {vol_ma_err}")

            # Map Calculated Column Names
            final_df_columns = df_calc.columns.tolist()
            indicator_mapping = {  # Internal Name : TA Base Name for _get_ta_col_name
                "ATR": "ATR",
                "EMA_Short": "EMA_Short",
                "EMA_Long": "EMA_Long",
                "Momentum": "Momentum",
                "CCI": "CCI",
                "Williams_R": "Williams_R",
                "MFI": "MFI",
                "SMA_10": "SMA_10",
                "RSI": "RSI",
                "VWAP": "VWAP",
                "PSAR_long": "PSAR_long",
                "PSAR_short": "PSAR_short",
                "StochRSI_K": "StochRSI_K",
                "StochRSI_D": "StochRSI_D",
                "BB_Lower": "BB_Lower",
                "BB_Middle": "BB_Middle",
                "BB_Upper": "BB_Upper",
                "Volume_MA": "Volume_MA",
            }
            self.ta_column_names = {}  # Reset mapping
            for internal_name, ta_base_name in indicator_mapping.items():
                mapped_col = self._get_ta_col_name(ta_base_name, final_df_columns)
                if mapped_col:
                    self.ta_column_names[internal_name] = mapped_col

            # Convert key price-based indicator columns back to Decimal if original was Decimal
            cols_to_decimalize = [
                "ATR",
                "BB_Lower",
                "BB_Middle",
                "BB_Upper",
                "PSAR_long",
                "PSAR_short",
                "VWAP",
                "SMA_10",
                "EMA_Short",
                "EMA_Long",
            ]
            if original_types.get("close") == Decimal:
                for key in cols_to_decimalize:
                    col_name = self.ta_column_names.get(key)
                    if col_name and col_name in df_calc.columns:
                        try:
                            df_calc[col_name] = df_calc[col_name].apply(
                                lambda x: Decimal(str(x))
                                if pd.notna(x) and np.isfinite(x)
                                else Decimal("NaN")
                            )
                        except (ValueError, TypeError, InvalidOperation) as conv_err:
                            self.logger.error(
                                f"Failed converting '{col_name}' back to Decimal: {conv_err}."
                            )

            self.logger.debug(
                f"Indicator calculations complete. Mapped columns: {self.ta_column_names}"
            )

        except Exception as e:
            self.logger.error(
                f"{NEON_RED}Error during indicator calculation: {e}{RESET}",
                exc_info=True,
            )

    def _update_latest_indicator_values(self):
        """Updates indicator_values dict with latest values from DataFrame."""
        self.indicator_values = {}  # Reset
        if self.df.empty:
            return self.logger.warning("Cannot update latest values: DataFrame empty.")

        try:
            if not self.df.index.is_monotonic_increasing:
                self.logger.warning("DataFrame index not sorted, sorting.")
                self.df.sort_index(inplace=True)
            latest = self.df.iloc[-1]
            self.indicator_values["Timestamp"] = self.df.index[
                -1
            ]  # Store timestamp (datetime)

        except IndexError:
            return self.logger.error("Error accessing latest row (empty DF?).")
        except Exception as e:
            return self.logger.error(f"Error getting latest row: {e}")

        # Process Base OHLCV (expect Decimal)
        for base_col in ["open", "high", "low", "close", "volume"]:
            key_name = base_col.capitalize()
            raw_value = latest.get(base_col)
            value = (
                raw_value
                if isinstance(raw_value, Decimal) and raw_value.is_finite()
                else Decimal("NaN")
            )
            if value.is_nan() and pd.notna(
                raw_value
            ):  # Handle case where it might be float/int after fallback
                try:
                    value = Decimal(str(raw_value))
                except:
                    value = Decimal("NaN")
            # Ensure prices>0, vol>=0
            if key_name != "Volume" and isinstance(value, Decimal) and value <= 0:
                value = Decimal("NaN")
            if key_name == "Volume" and isinstance(value, Decimal) and value < 0:
                value = Decimal("NaN")
            self.indicator_values[key_name] = value

        # Process TA indicators using mapped names
        for key, col_name in self.ta_column_names.items():
            is_price_based = key in [
                "ATR",
                "BB_Lower",
                "BB_Middle",
                "BB_Upper",
                "PSAR_long",
                "PSAR_short",
                "VWAP",
                "SMA_10",
                "EMA_Short",
                "EMA_Long",
            ]
            target_type = Decimal if is_price_based else float
            value = (
                Decimal("NaN") if target_type == Decimal else np.nan
            )  # Default NaN type

            if col_name and col_name in latest.index:
                raw_value = latest[col_name]
                if pd.notna(raw_value):
                    try:
                        if target_type == Decimal:
                            d_val = Decimal(str(raw_value))
                            value = (
                                d_val
                                if d_val.is_finite() and d_val > 0
                                else Decimal("NaN")
                            )
                        else:  # Float
                            f_val = float(raw_value)
                            value = f_val if np.isfinite(f_val) else np.nan
                    except (ValueError, TypeError, InvalidOperation):
                        pass  # Keep default NaN

            self.indicator_values[key] = value

        # Log Summary (formatted)
        log_vals = {}
        price_prec = self.get_price_precision()
        amt_prec = self.get_amount_precision_places()
        price_keys = [
            "Open",
            "High",
            "Low",
            "Close",
            "ATR",
            "BB_Lower",
            "BB_Middle",
            "BB_Upper",
            "PSAR_long",
            "PSAR_short",
            "VWAP",
            "SMA_10",
            "EMA_Short",
            "EMA_Long",
        ]
        amount_keys = ["Volume", "Volume_MA"]

        for k, v in self.indicator_values.items():
            if k == "Timestamp":
                log_vals[k] = v.strftime("%Y-%m-%d %H:%M:%S %Z")
                continue
            formatted_val = "NaN"
            if isinstance(v, Decimal) and v.is_finite():
                prec = (
                    price_prec
                    if k in price_keys
                    else amt_prec
                    if k in amount_keys
                    else 8
                )
                try:
                    formatted_val = f"{v:.{prec}f}"
                except ValueError:
                    formatted_val = str(v)
            elif isinstance(v, float) and np.isfinite(v):
                prec = amt_prec if k in amount_keys else 5
                try:
                    formatted_val = f"{v:.{prec}f}"
                except ValueError:
                    formatted_val = str(v)

            if formatted_val != "NaN":
                log_vals[k] = formatted_val

        if log_vals:
            sorted_log_vals = dict(sorted(log_vals.items()))
            self.logger.debug(
                f"Latest values ({self.symbol}): {json.dumps(sorted_log_vals)}"
            )
        else:
            self.logger.warning(
                f"No valid latest indicator values found for {self.symbol}."
            )

    def calculate_fibonacci_levels(
        self, window: Optional[int] = None
    ) -> Dict[str, Decimal]:
        """Calculates Fibonacci retracement levels using Decimal precision."""
        window = window or int(self.config.get("fibonacci_period", DEFAULT_FIB_PERIOD))
        self.fib_levels_data = {}  # Clear previous

        if self.df.empty or len(self.df) < window:
            self.logger.debug(
                f"Fibonacci skipped: Not enough data ({len(self.df)}/{window})."
            )
            return {}

        df_slice = self.df.tail(window)
        try:
            high_series = df_slice["high"].dropna()
            low_series = df_slice["low"].dropna()
            if high_series.empty or low_series.empty:
                return {}  # No valid data

            high_price = high_series.max()
            low_price = low_series.min()

            if (
                not isinstance(high_price, Decimal)
                or not high_price.is_finite()
                or not isinstance(low_price, Decimal)
                or not low_price.is_finite()
                or low_price <= 0
            ):
                self.logger.warning(
                    f"Invalid high/low for Fibonacci: H={high_price}, L={low_price}"
                )
                return {}

            diff = high_price - low_price
            levels = {}
            min_tick = self.get_min_tick_size()
            quantizer = (
                min_tick
                if min_tick.is_finite() and min_tick > 0
                else Decimal("1e-" + str(self.get_price_precision()))
            )

            if diff <= 0:  # Handle flat or error case
                level_price = high_price.quantize(quantizer, rounding=ROUND_DOWN)
                levels = {
                    f"Fib_{float(lvl) * 100:.1f}%": level_price for lvl in FIB_LEVELS
                }
            else:
                for level_pct_str in map(str, FIB_LEVELS):
                    level_pct = Decimal(level_pct_str)
                    level_name = f"Fib_{level_pct * 100:.1f}%"
                    level_price = (high_price - (diff * level_pct)).quantize(
                        quantizer, rounding=ROUND_DOWN
                    )
                    levels[level_name] = level_price

            self.fib_levels_data = levels
            price_prec = self.get_price_precision()
            log_levels = {k: f"{v:.{price_prec}f}" for k, v in levels.items()}
            self.logger.debug(f"Fibonacci levels ({window} bars): {log_levels}")
            return levels

        except Exception as e:
            self.logger.error(
                f"{NEON_RED}Fibonacci calculation error: {e}{RESET}", exc_info=True
            )
            return {}

    # --- Precision and Limit Helpers (with caching) ---

    def get_price_precision(self) -> int:
        """Determines price precision (decimal places) from market info, cached."""
        if self._cached_price_precision is not None:
            return self._cached_price_precision
        precision, source = None, "Unknown"
        try:
            prec_info = self.market_info.get("precision", {})
            price_prec_val = prec_info.get("price")
            if isinstance(price_prec_val, int) and price_prec_val >= 0:
                precision, source = price_prec_val, "precision.price (int)"
            elif price_prec_val is not None:  # Try as tick size
                try:
                    tick = Decimal(str(price_prec_val))
                    if tick.is_finite() and tick > 0:
                        precision = abs(tick.normalize().as_tuple().exponent)
                        source = f"precision.price (tick: {tick})"
                except:
                    pass
            # Fallback: limits.price.min as tick size
            if precision is None:
                min_price = (
                    self.market_info.get("limits", {}).get("price", {}).get("min")
                )
                if min_price is not None:
                    try:
                        tick = Decimal(str(min_price))
                        if tick.is_finite() and 0 < tick < 1:
                            precision = abs(tick.normalize().as_tuple().exponent)
                            source = f"limits.price.min ({tick})"
                    except:
                        pass
            # Fallback: Last close price
            if precision is None:
                last_close = self.indicator_values.get("Close")
                if (
                    isinstance(last_close, Decimal)
                    and last_close.is_finite()
                    and last_close > 0
                ):
                    p = abs(last_close.normalize().as_tuple().exponent)
                    if 0 <= p <= 12:
                        precision, source = p, f"Last Close ({last_close})"
        except Exception as e:
            self.logger.warning(f"Error determining price precision: {e}")

        if precision is None:
            precision, source = 4, "Default (4)"  # Final fallback
        self._cached_price_precision = precision
        self.logger.debug(
            f"Price precision ({self.symbol}): {precision} (Source: {source})"
        )
        return precision

    def get_min_tick_size(self) -> Decimal:
        """Gets the minimum price increment (tick size) as Decimal, cached."""
        if self._cached_min_tick_size is not None:
            return self._cached_min_tick_size
        tick_size, source = None, "Unknown"
        try:
            # Try precision.price as tick size
            prec_info = self.market_info.get("precision", {})
            price_prec_val = prec_info.get("price")
            if price_prec_val is not None and not isinstance(price_prec_val, int):
                try:
                    tick = Decimal(str(price_prec_val))
                    if tick.is_finite() and tick > 0:
                        tick_size, source = tick, "precision.price (value)"
                except:
                    pass
            # Fallback: limits.price.min
            if tick_size is None:
                min_price = (
                    self.market_info.get("limits", {}).get("price", {}).get("min")
                )
                if min_price is not None:
                    try:
                        tick = Decimal(str(min_price))
                        if tick.is_finite() and tick > 0:
                            tick_size, source = tick, "limits.price.min"
                    except:
                        pass
            # Fallback: Calculate from integer precision
            if (
                tick_size is None
                and isinstance(price_prec_val, int)
                and price_prec_val >= 0
            ):
                tick_size = Decimal("1e-" + str(price_prec_val))
                source = f"precision.price (int: {price_prec_val})"
        except Exception as e:
            self.logger.warning(f"Error determining tick size: {e}")

        if tick_size is None:  # Fallback: calculate from derived places
            places = self.get_price_precision()
            tick_size = Decimal("1e-" + str(places))
            source = f"Derived Precision ({places})"
        if (
            not isinstance(tick_size, Decimal)
            or not tick_size.is_finite()
            or tick_size <= 0
        ):
            tick_size = Decimal("0.00000001")  # Emergency fallback
            source = "Emergency Fallback"
            self.logger.error(
                f"Failed to get valid tick size! Using fallback: {tick_size}"
            )

        self._cached_min_tick_size = tick_size
        self.logger.debug(
            f"Min Tick Size ({self.symbol}): {tick_size} (Source: {source})"
        )
        return tick_size

    def get_amount_precision_places(self) -> int:
        """Determines amount precision (decimal places) from market info, cached."""
        if self._cached_amount_precision is not None:
            return self._cached_amount_precision
        precision, source = None, "Unknown"
        try:
            prec_info = self.market_info.get("precision", {})
            amt_prec_val = prec_info.get("amount")
            if isinstance(amt_prec_val, int) and amt_prec_val >= 0:
                precision, source = amt_prec_val, "precision.amount (int)"
            elif amt_prec_val is not None:  # Try as step size
                try:
                    step = Decimal(str(amt_prec_val))
                    if step.is_finite() and step > 0:
                        precision = abs(step.normalize().as_tuple().exponent)
                        source = f"precision.amount (step: {step})"
                except:
                    pass
            # Fallback: limits.amount.min as step size
            if precision is None:
                min_amt = (
                    self.market_info.get("limits", {}).get("amount", {}).get("min")
                )
                if min_amt is not None:
                    try:
                        step = Decimal(str(min_amt))
                        if (
                            step.is_finite() and 0 < step <= 1
                        ):  # Treat as step if fractional
                            if step < 1 or "." in str(min_amt):
                                precision = abs(step.normalize().as_tuple().exponent)
                                source = f"limits.amount.min (step: {step})"
                        elif (
                            step.is_finite() and step >= 1 and step % 1 == 0
                        ):  # Integer min amount -> 0 places
                            precision = 0
                            source = f"limits.amount.min (int: {step})"
                    except:
                        pass
        except Exception as e:
            self.logger.warning(f"Error determining amount precision: {e}")

        if precision is None:
            precision, source = 8, "Default (8)"  # Final fallback
        self._cached_amount_precision = precision
        self.logger.debug(
            f"Amount precision ({self.symbol}): {precision} (Source: {source})"
        )
        return precision

    def get_min_amount_step(self) -> Decimal:
        """Gets the minimum amount increment (step size) as Decimal, cached."""
        if self._cached_min_amount_step is not None:
            return self._cached_min_amount_step
        step_size, source = None, "Unknown"
        try:
            # Try precision.amount as step size
            prec_info = self.market_info.get("precision", {})
            amt_prec_val = prec_info.get("amount")
            if amt_prec_val is not None and not isinstance(amt_prec_val, int):
                try:
                    step = Decimal(str(amt_prec_val))
                    if step.is_finite() and step > 0:
                        step_size, source = step, "precision.amount (value)"
                except:
                    pass
            # Fallback: limits.amount.min
            if step_size is None:
                min_amt = (
                    self.market_info.get("limits", {}).get("amount", {}).get("min")
                )
                if min_amt is not None:
                    try:
                        step = Decimal(str(min_amt))
                        if step.is_finite() and step > 0:
                            step_size, source = step, "limits.amount.min"
                    except:
                        pass
            # Fallback: Calculate from integer precision
            if (
                step_size is None
                and isinstance(amt_prec_val, int)
                and amt_prec_val >= 0
            ):
                step_size = Decimal("1e-" + str(amt_prec_val))
                source = f"precision.amount (int: {amt_prec_val})"
        except Exception as e:
            self.logger.warning(f"Error determining amount step: {e}")

        if step_size is None:  # Fallback: calculate from derived places
            places = self.get_amount_precision_places()
            step_size = Decimal("1e-" + str(places))
            source = f"Derived Precision ({places})"
        if (
            not isinstance(step_size, Decimal)
            or not step_size.is_finite()
            or step_size <= 0
        ):
            step_size = Decimal("0.00000001")  # Emergency fallback
            source = "Emergency Fallback"
            self.logger.error(
                f"Failed to get valid amount step! Using fallback: {step_size}"
            )

        self._cached_min_amount_step = step_size
        self.logger.debug(
            f"Min Amount Step ({self.symbol}): {step_size} (Source: {source})"
        )
        return step_size

    def get_nearest_fibonacci_levels(
        self, current_price: Decimal, num_levels: int = 5
    ) -> List[Tuple[str, Decimal]]:
        """Finds the N nearest Fibonacci levels to the current price."""
        if not self.fib_levels_data:
            return []
        if (
            not isinstance(current_price, Decimal)
            or not current_price.is_finite()
            or current_price <= 0
        ):
            self.logger.warning(
                f"Invalid current price ({current_price}) for Fib comparison."
            )
            return []

        try:
            distances = []
            for name, level_price in self.fib_levels_data.items():
                if (
                    isinstance(level_price, Decimal)
                    and level_price.is_finite()
                    and level_price > 0
                ):
                    distances.append(
                        {
                            "name": name,
                            "level": level_price,
                            "distance": abs(current_price - level_price),
                        }
                    )
            if not distances:
                return []
            distances.sort(key=lambda x: x["distance"])
            return [(item["name"], item["level"]) for item in distances[:num_levels]]
        except Exception as e:
            self.logger.error(
                f"{NEON_RED}Error finding nearest Fib levels: {e}{RESET}", exc_info=True
            )
            return []

    # --- Signal Generation and Indicator Checks ---

    def generate_trading_signal(
        self, current_price: Decimal, orderbook_data: Optional[Dict]
    ) -> str:
        """Generates final trading signal (BUY/SELL/HOLD) based on weighted indicator scores."""
        final_score = Decimal("0.0")
        total_weight = Decimal("0.0")
        active_count = 0
        contrib_scores = {}  # Store individual scores for debugging

        if not self.indicator_values:
            return "HOLD"  # No data
        if (
            not isinstance(current_price, Decimal)
            or not current_price.is_finite()
            or current_price <= 0
        ):
            return "HOLD"
        if not self.weights:
            return "HOLD"  # No weights

        available_checks = {
            m.replace("_check_", "") for m in dir(self) if m.startswith("_check_")
        }

        for indicator_key, is_enabled in self.config.get("indicators", {}).items():
            if not is_enabled:
                continue
            if indicator_key not in available_checks:
                continue  # Skip if no check method

            weight_val = self.weights.get(indicator_key)
            if weight_val is None:
                continue
            try:
                weight = Decimal(str(weight_val))
                if not weight.is_finite() or weight < 0:
                    raise ValueError("Invalid weight")
                if weight == 0:
                    continue
            except (ValueError, TypeError, InvalidOperation):
                continue

            check_method_name = f"_check_{indicator_key}"
            score_float = np.nan
            try:
                method = getattr(self, check_method_name)
                if indicator_key == "orderbook":
                    score_float = (
                        method(orderbook_data, current_price)
                        if orderbook_data
                        else np.nan
                    )
                else:
                    score_float = method()
            except Exception as e:
                self.logger.error(
                    f"Error in check '{check_method_name}': {e}", exc_info=True
                )

            if pd.notna(score_float) and np.isfinite(score_float):
                try:
                    score_dec = max(
                        Decimal("-1.0"), min(Decimal("1.0"), Decimal(str(score_float)))
                    )  # Clamp [-1, 1]
                    final_score += score_dec * weight
                    total_weight += weight
                    active_count += 1
                    contrib_scores[indicator_key] = f"{score_dec:.3f}"
                except (ValueError, TypeError, InvalidOperation):
                    pass  # Ignore conversion errors

        final_signal = "HOLD"
        if total_weight > Decimal("1e-9"):
            try:
                threshold = Decimal(
                    str(self.config.get("signal_score_threshold", "1.5"))
                )
                if not threshold.is_finite() or threshold <= 0:
                    raise ValueError("Invalid threshold")
            except (ValueError, TypeError, InvalidOperation):
                threshold = Decimal(str(default_config["signal_score_threshold"]))
                self.logger.warning(
                    f"Invalid signal threshold. Using default {threshold}."
                )

            if final_score >= threshold:
                final_signal = "BUY"
            elif final_score <= -threshold:
                final_signal = "SELL"

        price_prec = self.get_price_precision()
        sig_color = (
            NEON_GREEN
            if final_signal == "BUY"
            else NEON_RED
            if final_signal == "SELL"
            else NEON_YELLOW
        )
        log_msg = (
            f"Signal ({self.symbol} @ {current_price:.{price_prec}f}): "
            f"Set='{self.active_weight_set_name}', ActInd={active_count}, "
            f"TotW={total_weight:.2f}, Score={final_score:.4f} (Thr: +/-{threshold:.2f}) "
            f"==> {sig_color}{final_signal}{RESET}"
        )
        self.logger.info(log_msg)
        if self.logger.isEnabledFor(logging.DEBUG) and contrib_scores:
            self.logger.debug(
                f"  Scores ({self.symbol}): {json.dumps(dict(sorted(contrib_scores.items())))}"
            )

        return final_signal

    # --- Indicator Check Methods (return float score -1.0 to 1.0 or np.nan) ---
    # These methods rely on self.indicator_values being up-to-date.

    def _check_ema_alignment(self) -> float:
        """Checks EMA alignment and price confirmation. Score: [-1.0, 1.0] or NaN."""
        ema_s = self.indicator_values.get("EMA_Short")  # Decimal | NaN
        ema_l = self.indicator_values.get("EMA_Long")  # Decimal | NaN
        close = self.indicator_values.get("Close")  # Decimal | NaN
        if not all(
            isinstance(v, Decimal) and v.is_finite() for v in [ema_s, ema_l, close]
        ):
            return np.nan
        try:
            short_above_long = ema_s > ema_l
            price_above_short = close > ema_s
            price_above_long = close > ema_l
            if price_above_short and short_above_long:
                return 1.0  # Strong Bullish
            if not price_above_long and not short_above_long:
                return -1.0  # Strong Bearish
            if short_above_long:
                return 0.3 if price_above_short else -0.2  # Weak Bullish EMA / Disagree
            else:
                return (
                    -0.3 if not price_above_long else 0.2
                )  # Weak Bearish EMA / Disagree
        except TypeError:
            return np.nan

    def _check_momentum(self) -> float:
        """Scores Momentum indicator value. Score: [-1.0, 1.0] or NaN."""
        mom = self.indicator_values.get("Momentum")  # float | NaN
        if not isinstance(mom, (float, int)) or not np.isfinite(mom):
            return np.nan
        # Simple thresholding (tune thresholds as needed)
        if mom >= 0.5:
            return 1.0
        if mom >= 0.1:
            return 0.5
        if mom <= -0.5:
            return -1.0
        if mom <= -0.1:
            return -0.5
        return 0.0

    def _check_volume_confirmation(self) -> float:
        """Scores volume relative to its MA. Score: [-1.0, 1.0] or NaN."""
        vol = self.indicator_values.get("Volume")  # Decimal | NaN
        vol_ma = self.indicator_values.get("Volume_MA")  # float | NaN
        if not isinstance(vol, Decimal) or not vol.is_finite() or vol < 0:
            return np.nan
        if (
            not isinstance(vol_ma, (float, int))
            or not np.isfinite(vol_ma)
            or vol_ma <= 0
        ):
            return np.nan
        try:
            vol_ma_d = Decimal(str(vol_ma))
            multiplier = Decimal(
                str(self.config.get("volume_confirmation_multiplier", 1.5))
            )
            if multiplier <= 0:
                multiplier = Decimal("1.5")
            ratio = vol / vol_ma_d if vol_ma_d > Decimal("1e-12") else Decimal("1")
            if ratio >= multiplier * Decimal("1.5"):
                return 1.0  # Very high vol
            if ratio >= multiplier:
                return 0.7  # High vol
            if ratio <= (Decimal("1") / multiplier):
                return -0.4  # Low vol (weak confirmation)
            return 0.0
        except (InvalidOperation, ZeroDivisionError):
            return np.nan

    def _check_stoch_rsi(self) -> float:
        """Scores Stochastic RSI K/D values. Score: [-1.0, 1.0] or NaN."""
        k = self.indicator_values.get("StochRSI_K")  # float | NaN
        d = self.indicator_values.get("StochRSI_D")  # float | NaN
        if not all(isinstance(v, (float, int)) and np.isfinite(v) for v in [k, d]):
            return np.nan
        try:
            os = float(self.config.get("stoch_rsi_oversold_threshold", 25))
            ob = float(self.config.get("stoch_rsi_overbought_threshold", 75))
            if not (0 < os < 100 and 0 < ob < 100 and os < ob):
                raise ValueError()
        except:
            os, ob = 25.0, 75.0
        score = 0.0
        if k < os and d < os:
            score = 1.0  # Strong Oversold -> Buy
        elif k > ob and d > ob:
            score = -1.0  # Strong Overbought -> Sell
        if k > d + 1.0 and d < ob:
            score = max(score, 0.7)  # Bullish cross
        elif d > k + 1.0 and k > os:
            score = min(score, -0.7)  # Bearish cross
        if k > 50 and d > 50 and score >= 0:
            score = max(score, 0.2)  # Both > 50
        elif k < 50 and d < 50 and score <= 0:
            score = min(score, -0.2)  # Both < 50
        return max(-1.0, min(1.0, score))

    def _check_rsi(self) -> float:
        """Scores RSI relative to levels. Score: [-1.0, 1.0] or NaN."""
        rsi = self.indicator_values.get("RSI")  # float | NaN
        if not isinstance(rsi, (float, int)) or not np.isfinite(rsi):
            return np.nan
        if rsi >= 80:
            return -1.0  # Extreme OB
        if rsi >= 70:
            return -0.7  # OB
        if rsi > 60:
            return -0.3
        if rsi <= 20:
            return 1.0  # Extreme OS
        if rsi <= 30:
            return 0.7  # OS
        if rsi < 40:
            return 0.3
        return 0.0  # Neutral (40-60)

    def _check_cci(self) -> float:
        """Scores CCI relative to +/-100, +/-200. Score: [-1.0, 1.0] or NaN."""
        cci = self.indicator_values.get("CCI")  # float | NaN
        if not isinstance(cci, (float, int)) or not np.isfinite(cci):
            return np.nan
        if cci >= 200:
            return -1.0
        if cci >= 100:
            return -0.7
        if cci > 0:
            return -0.2
        if cci <= -200:
            return 1.0
        if cci <= -100:
            return 0.7
        if cci < 0:
            return 0.2
        return 0.0

    def _check_wr(self) -> float:
        """Scores Williams %R relative to -20/-80. Score: [-1.0, 1.0] or NaN."""
        wr = self.indicator_values.get("Williams_R")  # float | NaN (-100 to 0)
        if not isinstance(wr, (float, int)) or not np.isfinite(wr):
            return np.nan
        if wr >= -10:
            return -1.0  # Extreme OB
        if wr >= -20:
            return -0.7  # OB
        if wr > -50:
            return -0.2
        if wr <= -90:
            return 1.0  # Extreme OS
        if wr <= -80:
            return 0.7  # OS
        if wr < -50:
            return 0.2
        return 0.0  # Exactly -50

    def _check_psar(self) -> float:
        """Scores PSAR position relative to price. Score: [-1.0, 1.0] or NaN."""
        psar_l = self.indicator_values.get("PSAR_long")  # Decimal | NaN
        psar_s = self.indicator_values.get("PSAR_short")  # Decimal | NaN
        l_active = isinstance(psar_l, Decimal) and psar_l.is_finite()
        s_active = isinstance(psar_s, Decimal) and psar_s.is_finite()
        if l_active and not s_active:
            return 1.0  # Uptrend (PSAR below price)
        if s_active and not l_active:
            return -1.0  # Downtrend (PSAR above price)
        if not l_active and not s_active:
            return np.nan  # Indeterminate
        self.logger.warning(
            f"PSAR check: Both Long ({psar_l}) and Short ({psar_s}) seem active."
        )
        return 0.0  # Unusual state

    def _check_sma_10(self) -> float:
        """Scores price vs 10-SMA. Score: [-0.6, 0.6] or NaN."""
        sma = self.indicator_values.get("SMA_10")  # Decimal | NaN
        close = self.indicator_values.get("Close")  # Decimal | NaN
        if not all(isinstance(v, Decimal) and v.is_finite() for v in [sma, close]):
            return np.nan
        try:
            if close > sma:
                return 0.6
            if close < sma:
                return -0.6
            return 0.0
        except TypeError:
            return np.nan

    def _check_vwap(self) -> float:
        """Scores price vs VWAP. Score: [-0.7, 0.7] or NaN."""
        vwap = self.indicator_values.get("VWAP")  # Decimal | NaN
        close = self.indicator_values.get("Close")  # Decimal | NaN
        if not all(isinstance(v, Decimal) and v.is_finite() for v in [vwap, close]):
            return np.nan
        try:
            if close > vwap:
                return 0.7
            if close < vwap:
                return -0.7
            return 0.0
        except TypeError:
            return np.nan

    def _check_mfi(self) -> float:
        """Scores Money Flow Index vs 20/80. Score: [-1.0, 1.0] or NaN."""
        mfi = self.indicator_values.get("MFI")  # float | NaN
        if not isinstance(mfi, (float, int)) or not np.isfinite(mfi):
            return np.nan
        if mfi >= 90:
            return -1.0  # Extreme OB
        if mfi >= 80:
            return -0.7  # OB
        if mfi > 65:
            return -0.3
        if mfi <= 10:
            return 1.0  # Extreme OS
        if mfi <= 20:
            return 0.7  # OS
        if mfi < 35:
            return 0.3
        return 0.0  # Neutral (35-65)

    def _check_bollinger_bands(self) -> float:
        """Scores price vs Bollinger Bands. Score: [-1.0, 1.0] or NaN."""
        bbl = self.indicator_values.get("BB_Lower")  # Decimal | NaN
        bbm = self.indicator_values.get("BB_Middle")  # Decimal | NaN
        bbu = self.indicator_values.get("BB_Upper")  # Decimal | NaN
        close = self.indicator_values.get("Close")  # Decimal | NaN
        if not all(
            isinstance(v, Decimal) and v.is_finite() for v in [bbl, bbm, bbu, close]
        ):
            return np.nan
        try:
            if close <= bbl:
                return 1.0  # Touch/Below Lower -> Buy
            if close >= bbu:
                return -1.0  # Touch/Above Upper -> Sell
            band_width = bbu - bbl
            if band_width <= 0:
                return np.nan  # Avoid division by zero
            # Scale score between bands, centered on middle band
            if close > bbm:  # Above middle band -> Reversion bias (sell)
                pos_in_upper = (close - bbm) / (bbu - bbm)  # 0 to ~1
                score = float(pos_in_upper) * -0.7
                return max(-0.7, score)
            else:  # Below middle band -> Reversion bias (buy)
                pos_in_lower = (bbm - close) / (bbm - bbl)  # 0 to ~1
                score = float(pos_in_lower) * 0.7
                return min(0.7, score)
        except (TypeError, ZeroDivisionError, InvalidOperation):
            return np.nan

    def _check_orderbook(
        self, orderbook_data: Optional[Dict], current_price: Decimal
    ) -> float:
        """Analyzes Order Book Imbalance. Score: [-1.0, 1.0] or NaN."""
        if (
            not orderbook_data
            or not isinstance(orderbook_data.get("bids"), list)
            or not isinstance(orderbook_data.get("asks"), list)
        ):
            return np.nan
        bids = orderbook_data[
            "bids"
        ]  # Assumes list of [Decimal(price), Decimal(amount)]
        asks = orderbook_data["asks"]
        if not bids or not asks:
            return np.nan

        try:
            levels = min(
                len(bids), len(asks), int(self.config.get("orderbook_limit", 10))
            )
            if levels <= 0:
                return 0.0
            bid_vol = sum(b[1] for b in bids[:levels])
            ask_vol = sum(a[1] for a in asks[:levels])
            total_vol = bid_vol + ask_vol
            if total_vol <= Decimal("1e-12"):
                return 0.0
            obi_diff = (bid_vol - ask_vol) / total_vol  # [-1, 1]
            score = float(max(Decimal("-1.0"), min(Decimal("1.0"), obi_diff)))
            # self.logger.debug(f"OB Check ({levels} levels): BidVol={bid_vol:.4f}, AskVol={ask_vol:.4f}, Score={score:.4f}")
            return score
        except (
            IndexError,
            ValueError,
            TypeError,
            InvalidOperation,
            ZeroDivisionError,
        ) as e:
            self.logger.warning(f"Orderbook analysis failed: {e}", exc_info=False)
            return np.nan

    # --- TP/SL Calculation ---
    def calculate_entry_tp_sl(
        self, entry_price_estimate: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """
        Calculates potential Take Profit and initial Stop Loss based on entry, signal, and ATR.
        Returns (Validated Entry Estimate, TP Price, SL Price) using Decimal and market constraints.
        Returns None for TP/SL if calculation fails.
        """
        final_tp, final_sl = None, None
        if signal not in ["BUY", "SELL"]:
            return entry_price_estimate, None, None

        atr_val = self.indicator_values.get("ATR")
        if not isinstance(atr_val, Decimal) or not atr_val.is_finite() or atr_val <= 0:
            self.logger.warning(f"TP/SL Calc Fail ({signal}): Invalid ATR ({atr_val}).")
            return entry_price_estimate, None, None
        if (
            not isinstance(entry_price_estimate, Decimal)
            or not entry_price_estimate.is_finite()
            or entry_price_estimate <= 0
        ):
            self.logger.warning(
                f"TP/SL Calc Fail ({signal}): Invalid entry estimate ({entry_price_estimate})."
            )
            return entry_price_estimate, None, None

        try:
            tp_mult = Decimal(str(self.config.get("take_profit_multiple", "1.0")))
            sl_mult = Decimal(str(self.config.get("stop_loss_multiple", "1.5")))
            if tp_mult <= 0 or sl_mult <= 0:
                tp_mult = Decimal(str(default_config["take_profit_multiple"]))
                sl_mult = Decimal(str(default_config["stop_loss_multiple"]))

            min_tick = self.get_min_tick_size()
            quantizer = min_tick  # Use tick size for quantization

            tp_offset = atr_val * tp_mult
            sl_offset = atr_val * sl_mult

            if signal == "BUY":
                tp_raw = entry_price_estimate + tp_offset
                sl_raw = entry_price_estimate - sl_offset
                # Round TP towards neutral (down), SL away (down)
                rounding_tp, rounding_sl = ROUND_DOWN, ROUND_DOWN
            else:  # SELL
                tp_raw = entry_price_estimate - tp_offset
                sl_raw = entry_price_estimate + sl_offset
                # Round TP towards neutral (up), SL away (up)
                rounding_tp, rounding_sl = ROUND_UP, ROUND_UP

            if tp_raw.is_finite():
                final_tp = tp_raw.quantize(quantizer, rounding=rounding_tp)
            if sl_raw.is_finite():
                final_sl = sl_raw.quantize(quantizer, rounding=rounding_sl)

            # Validation: Ensure SL/TP are valid, beyond entry by min tick, and positive
            if final_sl is not None:
                if abs(final_sl - entry_price_estimate) < min_tick or final_sl <= 0:
                    # Try adjusting SL one tick further away
                    adj_sl = (
                        (entry_price_estimate - min_tick).quantize(
                            quantizer, rounding_sl
                        )
                        if signal == "BUY"
                        else (entry_price_estimate + min_tick).quantize(
                            quantizer, rounding_sl
                        )
                    )
                    if (
                        adj_sl.is_finite()
                        and adj_sl > 0
                        and abs(adj_sl - entry_price_estimate) >= min_tick
                    ):
                        self.logger.debug(
                            f"Adjusted SL ({final_sl}) to {adj_sl} (1 tick away from {entry_price_estimate})"
                        )
                        final_sl = adj_sl
                    else:
                        self.logger.warning(
                            f"Invalid/too close SL ({final_sl}) for Entry {entry_price_estimate}. Nullifying SL."
                        )
                        final_sl = None

            if final_tp is not None:
                if abs(final_tp - entry_price_estimate) < min_tick or final_tp <= 0:
                    self.logger.warning(
                        f"Invalid/too close TP ({final_tp}) for Entry {entry_price_estimate}. Nullifying TP."
                    )
                    final_tp = None

            price_prec = self.get_price_precision()
            tp_str = f"{final_tp:.{price_prec}f}" if final_tp else "None"
            sl_str = f"{final_sl:.{price_prec}f}" if final_sl else "None"
            self.logger.debug(
                f"Calc TP/SL ({signal}): Entry={entry_price_estimate:.{price_prec}f}, ATR={atr_val:.{price_prec + 2}f}, "
                f"Tick={min_tick}, TP={tp_str}, SL={sl_str}"
            )

            return entry_price_estimate, final_tp, final_sl

        except Exception as e:
            self.logger.error(f"Unexpected error calculating TP/SL: {e}", exc_info=True)
            return entry_price_estimate, None, None


# --- Trading Logic Helper Functions ---


def fetch_balance(
    exchange: ccxt.Exchange,
    currency: str,
    logger: logging.Logger,
    config: Dict[str, Any],  # Pass config to get account type
) -> Optional[Decimal]:
    """
    Fetches available balance for a currency, handling Bybit V5 account types.
    Returns available balance as Decimal, or None on failure.
    """
    lg = logger
    balance_info = None
    account_type_used = "N/A"
    params = {}

    # Bybit V5: Use accountType from config
    if exchange.id == "bybit":
        account_type = config.get("bybit_account_type", "CONTRACT").upper()
        params = {"accountType": account_type}
        account_type_used = account_type
        lg.debug(
            f"Attempting Bybit balance fetch for {currency} (Account: {account_type})..."
        )
        try:
            balance_info = safe_api_call(exchange.fetch_balance, lg, params=params)
        except ccxt.ExchangeError as e:
            # Handle account type errors gracefully, maybe fallback?
            if "account type does not exist" in str(e).lower() or "30082" in str(
                e
            ):  # Bybit code for invalid account type
                lg.warning(
                    f"Account type {account_type} invalid/not found: {e}. Trying default fetch."
                )
                params = {}  # Clear params for default fetch
                account_type_used = "Default (Fallback)"
            else:
                lg.warning(
                    f"Exchange error fetching {account_type} balance: {e}. Trying default."
                )
                params = {}
                account_type_used = "Default (Fallback)"
            balance_info = None  # Trigger fallback
        except Exception as e:
            lg.warning(f"Failed fetching {account_type} balance: {e}. Trying default.")
            params = {}
            account_type_used = "Default (Fallback)"
            balance_info = None

    # Default fetch (if not Bybit or fallback needed)
    if balance_info is None:
        if account_type_used == "N/A":
            account_type_used = "Default"  # Mark if default was first try
        lg.debug(f"Fetching balance for {currency} (Account: {account_type_used})...")
        try:
            balance_info = safe_api_call(
                exchange.fetch_balance, lg, params=params
            )  # Use potentially cleared params
        except Exception as e:
            lg.error(f"Failed to fetch balance (Account: {account_type_used}): {e}")
            return None

    # --- Parse balance_info ---
    if not balance_info:
        lg.error(f"Balance fetch returned empty (Account: {account_type_used}).")
        return None

    free_balance_str = None
    parse_source = "Unknown"

    # Try standard paths first
    if currency in balance_info and isinstance(balance_info.get(currency), dict):
        free = balance_info[currency].get("free") or balance_info[currency].get(
            "available"
        )
        if free is not None:
            free_balance_str, parse_source = (
                str(free),
                f"Standard ['{currency}']['free/avail']",
            )
    elif (
        isinstance(balance_info.get("free"), dict)
        and balance_info["free"].get(currency) is not None
    ):
        free_balance_str, parse_source = (
            str(balance_info["free"][currency]),
            f"Top-level ['free']['{currency}']",
        )
    elif (
        isinstance(balance_info.get("available"), dict)
        and balance_info["available"].get(currency) is not None
    ):
        free_balance_str, parse_source = (
            str(balance_info["available"][currency]),
            f"Top-level ['available']['{currency}']",
        )

    # Bybit V5 Specific Parsing (if standard failed or we know it's Bybit)
    if (
        free_balance_str is None
        and exchange.id == "bybit"
        and isinstance(balance_info.get("info"), dict)
    ):
        info = balance_info["info"]
        result = info.get("result", info)
        if isinstance(result.get("list"), list) and result["list"]:
            # Iterate through accounts in the list
            for account in result["list"]:
                acc_type_resp = account.get("accountType")
                # Match the config account type OR check SPOT if currency matches
                is_target_type = acc_type_resp == config.get(
                    "bybit_account_type", "CONTRACT"
                )
                is_spot_match = (
                    acc_type_resp == "SPOT"
                    and config.get("bybit_account_type") == "SPOT"
                )

                if is_target_type or is_spot_match:
                    # UNIFIED/SPOT: Check 'coin' list
                    if isinstance(account.get("coin"), list):
                        for coin_data in account["coin"]:
                            if coin_data.get("coin") == currency:
                                free = (
                                    coin_data.get("availableToWithdraw")
                                    or coin_data.get("availableBalance")
                                    or coin_data.get("walletBalance")
                                )
                                if free is not None:
                                    free_balance_str = str(free)
                                    parse_source = f"Bybit V5 info.list[coin='{currency}'] (Type: {acc_type_resp})"
                                    break  # Found in coin list
                        if free_balance_str:
                            break  # Found, exit outer loop
                    # CONTRACT: Direct fields
                    elif acc_type_resp == "CONTRACT":
                        free = account.get("availableBalance") or account.get(
                            "walletBalance"
                        )
                        if free is not None:
                            free_balance_str = str(free)
                            parse_source = "Bybit V5 info.list[CONTRACT]"
                            break  # Found in contract details
            # Fallback: Check first account in list if no specific match found
            if free_balance_str is None and result["list"]:
                first_acc = result["list"][0]
                free = (
                    first_acc.get("availableBalance")
                    or first_acc.get("availableToWithdraw")
                    or first_acc.get("walletBalance")
                )
                if free is not None:
                    free_balance_str = str(free)
                    parse_source = "Bybit V5 info.list[0] (Fallback)"

    # Use 'total' as last resort if 'free'/'available' failed
    if free_balance_str is None:
        total_str, total_src = None, "Unknown Total"
        if (
            currency in balance_info
            and isinstance(balance_info.get(currency), dict)
            and balance_info[currency].get("total") is not None
        ):
            total_str, total_src = (
                str(balance_info[currency]["total"]),
                f"Standard ['{currency}']['total']",
            )
        # Add other total paths if needed
        if total_str is not None:
            lg.warning(
                f"{NEON_YELLOW}Using 'total' balance for {currency} ({total_str}) as fallback ({total_src}).{RESET}"
            )
            free_balance_str = total_str
            parse_source = total_src + " (Fallback)"
        else:
            lg.error(
                f"{NEON_RED}Could not find any balance for {currency} (Account: {account_type_used}).{RESET}"
            )
            lg.debug(f"Full balance_info: {json.dumps(balance_info, default=str)}")
            return None

    # Convert final string to Decimal
    try:
        final_balance = Decimal(free_balance_str)
        if not final_balance.is_finite():
            final_balance = Decimal("0")
        if final_balance < 0:
            final_balance = Decimal("0")  # Treat negative available as zero
        lg.info(
            f"Available {currency} balance ({parse_source}, Acc: {account_type_used}): {final_balance:.4f}"
        )
        return final_balance
    except (ValueError, TypeError, InvalidOperation) as e:
        lg.error(
            f"Failed converting balance '{free_balance_str}' ({parse_source}) to Decimal: {e}"
        )
        return None


def get_market_info(
    exchange: ccxt.Exchange, symbol: str, logger: logging.Logger
) -> Optional[Dict]:
    """
    Gets market information dictionary from CCXT, ensuring markets are loaded.
    Adds convenience flags (is_contract, is_linear, etc.).
    """
    lg = logger
    try:
        if not exchange.markets or not exchange.markets_by_id:
            lg.info(f"Markets not loaded for {exchange.id}. Loading...")
            try:
                safe_api_call(exchange.load_markets, lg, reload=True)
            except Exception as load_err:
                lg.error(
                    f"{NEON_RED}Failed loading markets: {load_err}. Cannot get market info.{RESET}"
                )
                return None

        market = exchange.market(symbol)
        if not market or not isinstance(market, dict):
            lg.error(f"{NEON_RED}Market '{symbol}' not found in CCXT markets.{RESET}")
            # Hint for Bybit V5 format
            if "/" in symbol and ":" not in symbol and exchange.id == "bybit":
                base, quote = symbol.split("/")[:2]
                suggested = f"{base}/{quote}:{quote}"
                lg.warning(
                    f"{NEON_YELLOW}Hint: Try Bybit V5 linear format like '{suggested}'.{RESET}"
                )
            return None

        # Add Convenience Flags
        m_type = market.get("type", "").lower()
        is_spot = m_type == "spot"
        is_swap = m_type == "swap"
        is_future = m_type == "future"
        is_contract = is_swap or is_future or market.get("contract", False)
        is_linear = market.get("linear", False)
        is_inverse = market.get("inverse", False)
        # Infer linear/inverse if contract but not set explicitly
        if is_contract and not is_linear and not is_inverse:
            default_type = exchange.options.get("defaultType", "").lower()
            quote_id = market.get("quoteId", "").upper()
            if default_type == "linear" or quote_id in ["USDT", "USDC", "BUSD"]:
                is_linear = True
            elif default_type == "inverse" or quote_id == "USD":
                is_inverse = True
            else:
                is_linear = True  # Default assumption

        market.update(
            {
                "is_spot": is_spot,
                "is_contract": is_contract,
                "is_linear": is_linear,
                "is_inverse": is_inverse,
            }
        )

        lg.debug(
            f"Market Info ({symbol}): ID={market.get('id')}, Type={m_type}, "
            f"Contract={is_contract}, Linear={is_linear}, Inverse={is_inverse}, "
            f"Active={market.get('active', True)}"
        )
        lg.debug(f"  Precision: {market.get('precision')}")
        lg.debug(f"  Limits: {market.get('limits')}")

        if not market.get("active", True):
            lg.warning(f"{NEON_YELLOW}Market {symbol} is inactive.{RESET}")

        return market

    except ccxt.BadSymbol as e:
        lg.error(f"{NEON_RED}Invalid symbol '{symbol}': {e}{RESET}")
        return None
    except Exception as e:
        lg.error(
            f"{NEON_RED}Error getting market info for {symbol}: {e}{RESET}",
            exc_info=True,
        )
        return None


def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float,
    initial_stop_loss_price: Decimal,
    entry_price: Decimal,
    market_info: Dict,
    exchange: ccxt.Exchange,
    logger: logging.Logger,
    config: Dict[str, Any],  # Pass config directly
) -> Optional[Decimal]:
    """
    Calculates position size based on risk, SL, balance, and market constraints (Linear/Spot only).
    Returns size as Decimal, or None on failure.
    """
    lg = logger
    symbol = market_info.get("symbol", "UNKNOWN")
    quote_currency = market_info.get("quote", config.get("quote_currency", "USDT"))
    base_currency = market_info.get("base", "BASE")
    is_contract = market_info.get("is_contract", False)
    is_linear = market_info.get("is_linear", False)
    is_inverse = market_info.get("is_inverse", False)
    size_unit = base_currency if (is_linear or not is_contract) else "Contracts"

    # --- Input Validation ---
    if not (isinstance(balance, Decimal) and balance.is_finite() and balance > 0):
        lg.error(f"Size Calc Fail ({symbol}): Invalid balance ({balance}).")
        return None
    if not (isinstance(risk_per_trade, (float, int)) and 0 < risk_per_trade < 1):
        lg.error(
            f"Size Calc Fail ({symbol}): Invalid risk_per_trade ({risk_per_trade})."
        )
        return None
    if not (
        isinstance(sl_price := initial_stop_loss_price, Decimal)
        and sl_price.is_finite()
        and sl_price > 0
    ):
        lg.error(f"Size Calc Fail ({symbol}): Invalid SL price ({sl_price}).")
        return None
    if not (
        isinstance(entry_price, Decimal) and entry_price.is_finite() and entry_price > 0
    ):
        lg.error(f"Size Calc Fail ({symbol}): Invalid entry price ({entry_price}).")
        return None
    if sl_price == entry_price:
        lg.error(f"Size Calc Fail ({symbol}): SL price equals entry price.")
        return None
    if "limits" not in market_info or "precision" not in market_info:
        lg.error(f"Size Calc Fail ({symbol}): Market info missing limits/precision.")
        return None

    # --- Block Inverse Contracts (Requires different logic) ---
    if is_inverse:
        lg.error(
            f"{NEON_RED}Inverse contract sizing not implemented for {symbol}.{RESET}"
        )
        return None
    if is_contract and not is_linear:
        lg.warning(
            f"{NEON_YELLOW}Market {symbol} is contract but not Linear. Assuming Linear sizing.{RESET}"
        )

    try:
        # Get precision/step using a temporary Analyzer instance
        analyzer = TradingAnalyzer(pd.DataFrame(), lg, config, market_info)
        min_amount_step = analyzer.get_min_amount_step()
        amount_prec_places = analyzer.get_amount_precision_places()

        risk_amount_quote = balance * Decimal(str(risk_per_trade))
        sl_distance_quote = abs(entry_price - sl_price)
        if not sl_distance_quote.is_finite() or sl_distance_quote <= 0:
            lg.error(
                f"Size Calc Fail ({symbol}): Invalid SL distance ({sl_distance_quote})."
            )
            return None

        # Contract Size (Multiplier, usually 1 for Linear/Spot)
        contract_size = Decimal("1")
        if is_contract:  # Should be linear here
            cs_str = market_info.get("contractSize")
            if cs_str is not None:
                try:
                    cs = Decimal(str(cs_str))
                    contract_size = cs if cs.is_finite() and cs > 0 else contract_size
                except:
                    lg.warning(
                        f"Invalid contract size '{cs_str}', using {contract_size}."
                    )

        # --- Initial Size Calculation (Base Currency for Linear/Spot) ---
        # Size = Risk Amount (Quote) / (SL Distance (Quote/Unit) * Contract Size Multiplier)
        if sl_distance_quote <= 0 or contract_size <= 0:
            lg.error(
                f"Size Calc Fail ({symbol}): Invalid SL distance or contract size."
            )
            return None
        calculated_size = risk_amount_quote / (sl_distance_quote * contract_size)

        if not calculated_size.is_finite() or calculated_size <= 0:
            lg.error(f"Initial size calc resulted in {calculated_size}. Check inputs.")
            return None

        lg.info(
            f"Position Sizing ({symbol}): Balance={balance:.2f}{quote_currency}, Risk={risk_per_trade:.2%}, RiskAmt={risk_amount_quote:.4f}{quote_currency}"
        )
        lg.info(f"  Entry={entry_price}, SL={sl_price}, SL Dist={sl_distance_quote}")
        lg.info(f"  ContractSize={contract_size}, SizeUnit={size_unit}")
        lg.info(
            f"  Initial Calc Size = {calculated_size:.{amount_prec_places + 4}f} {size_unit}"
        )

        # --- Apply Market Limits and Step Size ---
        limits = market_info.get("limits", {})
        amt_limits = limits.get("amount", {})
        cost_limits = limits.get("cost", {})
        min_amt = Decimal(str(amt_limits.get("min", "0")))
        max_amt = Decimal(str(amt_limits.get("max", "inf")))
        min_cost = Decimal(str(cost_limits.get("min", "0")))
        max_cost = Decimal(str(cost_limits.get("max", "inf")))

        adjusted_size = calculated_size

        # 1. Apply Amount Step (Round DOWN)
        if min_amount_step.is_finite() and min_amount_step > 0:
            adj_size_step = (adjusted_size // min_amount_step) * min_amount_step
            if adj_size_step != adjusted_size:
                lg.info(
                    f"  Size adjusted by Step ({min_amount_step}): {adjusted_size:.{amount_prec_places + 2}f} -> {adj_size_step:.{amount_prec_places}f}"
                )
            adjusted_size = adj_size_step
        else:  # Fallback: Round to precision places
            adj_size_prec = adjusted_size.quantize(
                Decimal("1e-" + str(amount_prec_places)), rounding=ROUND_DOWN
            )
            if adj_size_prec != adjusted_size:
                lg.warning(
                    f"Invalid amount step ({min_amount_step}). Adjusting by precision ({amount_prec_places} places)."
                )
                lg.info(
                    f"  Size adjusted by Precision: {adjusted_size:.{amount_prec_places + 2}f} -> {adj_size_prec:.{amount_prec_places}f}"
                )
            adjusted_size = adj_size_prec

        # 2. Check Min Amount Limit
        if min_amt.is_finite() and min_amt > 0 and adjusted_size < min_amt:
            lg.error(
                f"{NEON_RED}Size Calc Fail: Adjusted size {adjusted_size:.{amount_prec_places}f} < Min Amount {min_amt}.{RESET}"
            )
            return None

        # 3. Clamp by Max Amount Limit (and re-apply step)
        if max_amt.is_finite() and adjusted_size > max_amt:
            orig_size = adjusted_size
            adjusted_size = max_amt
            if min_amount_step.is_finite() and min_amount_step > 0:
                adjusted_size = (adjusted_size // min_amount_step) * min_amount_step
            else:
                adjusted_size = adjusted_size.quantize(
                    Decimal("1e-" + str(amount_prec_places)), rounding=ROUND_DOWN
                )
            lg.warning(
                f"{NEON_YELLOW}Size capped by Max Amount: {orig_size:.{amount_prec_places}f} -> {adjusted_size:.{amount_prec_places}f}{RESET}"
            )
            # Re-check min amount after capping
            if min_amt.is_finite() and min_amt > 0 and adjusted_size < min_amt:
                lg.error(
                    f"{NEON_RED}Size Calc Fail: Size after max cap {adjusted_size} < Min Amount {min_amt}.{RESET}"
                )
                return None

        # 4. Check Cost Limits (Min/Max)
        est_cost = adjusted_size * entry_price * contract_size
        lg.debug(
            f"  Cost Check: Final Size={adjusted_size:.{amount_prec_places}f}, Est. Cost={est_cost:.4f} (Min:{min_cost}, Max:{max_cost})"
        )

        if min_cost.is_finite() and min_cost > 0 and est_cost < min_cost:
            lg.error(
                f"{NEON_RED}Size Calc Fail: Est. Cost {est_cost:.4f} < Min Cost {min_cost}.{RESET}"
            )
            return None

        if max_cost.is_finite() and max_cost > 0 and est_cost > max_cost:
            if entry_price > 0 and contract_size > 0:
                # Calculate max size based on max cost, round down by step
                size_from_max_cost = max_cost / (entry_price * contract_size)
                adj_size_cost_capped = (
                    (size_from_max_cost // min_amount_step) * min_amount_step
                    if min_amount_step > 0
                    else size_from_max_cost.quantize(
                        Decimal("1e-" + str(amount_prec_places)), rounding=ROUND_DOWN
                    )
                )

                if adj_size_cost_capped < adjusted_size and adj_size_cost_capped > 0:
                    lg.warning(
                        f"{NEON_YELLOW}Size capped by Max Cost: {adjusted_size:.{amount_prec_places}f} -> {adj_size_cost_capped:.{amount_prec_places}f}{RESET}"
                    )
                    adjusted_size = adj_size_cost_capped
                    # Re-check min amount after cost capping
                    if min_amt.is_finite() and min_amt > 0 and adjusted_size < min_amt:
                        lg.error(
                            f"{NEON_RED}Size Calc Fail: Size after max cost cap {adjusted_size} < Min Amount {min_amt}.{RESET}"
                        )
                        return None
                else:
                    lg.warning(
                        f"Max Cost capping ineffective. Original Size: {adjusted_size}"
                    )
            else:
                lg.error(
                    f"{NEON_RED}Size Calc Fail: Est. Cost {est_cost:.4f} > Max Cost {max_cost}, cannot recalc.{RESET}"
                )
                return None

        # --- Final Validation ---
        final_size = adjusted_size
        if not final_size.is_finite() or final_size <= 0:
            lg.error(
                f"{NEON_RED}Final size is invalid ({final_size}) after adjustments.{RESET}"
            )
            return None
        if min_amt.is_finite() and min_amt > 0 and final_size < min_amt:
            lg.error(
                f"{NEON_RED}Final size {final_size} < Min Amount {min_amt}.{RESET}"
            )
            return None

        # Format using amount_to_precision (optional, step rounding should suffice)
        try:
            final_size_str = exchange.amount_to_precision(symbol, float(final_size))
            final_size_decimal = Decimal(final_size_str)
            lg.info(
                f"{NEON_GREEN}Final Position Size ({symbol}): {final_size_decimal} {size_unit} (Formatted: {final_size_str}){RESET}"
            )
            return final_size_decimal
        except Exception as fmt_err:
            lg.warning(
                f"Final size formatting failed: {fmt_err}. Using unformatted Decimal."
            )
            lg.info(
                f"{NEON_GREEN}Final Position Size ({symbol}): {final_size} {size_unit} (Unformatted){RESET}"
            )
            return final_size

    except Exception as e:
        lg.error(
            f"{NEON_RED}Unexpected error calculating position size: {e}{RESET}",
            exc_info=True,
        )
        return None


def get_open_position(
    exchange: ccxt.Exchange,
    symbol: str,
    logger: logging.Logger,
    config: Dict[str, Any],  # Pass config for category/account type
) -> Optional[Dict]:
    """
    Checks for an open position using fetch_positions/fetch_position with Bybit V5 parsing.
    Returns a standardized dictionary for the open position, or None.
    """
    lg = logger
    has_fetch_pos = exchange.has.get("fetchPosition", False)
    has_fetch_pos_list = exchange.has.get("fetchPositions", False)

    if not has_fetch_pos and not has_fetch_pos_list:
        lg.warning(f"{exchange.id} supports neither fetchPosition nor fetchPositions.")
        return None

    market_info = get_market_info(exchange, symbol, lg)
    if not market_info:
        lg.error(f"Cannot get position: Failed getting market info for {symbol}.")
        return None
    market_id = market_info.get("id")
    if not market_id:
        lg.error(f"Cannot get position: Market ID missing for {symbol}.")
        return None

    # Params for Bybit V5 (category is crucial)
    params = {}
    if exchange.id == "bybit":
        params = {
            "symbol": market_id,
            "category": config.get("bybit_category", "linear"),
        }

    positions_data = []
    fetch_method = "None"

    try:
        # Attempt 1: fetchPosition (preferred for V5 if available)
        if has_fetch_pos:
            fetch_method = f"fetchPosition(symbol='{symbol}')"
            lg.debug(f"Attempting {fetch_method} with params: {params}...")
            try:
                pos = safe_api_call(exchange.fetch_position, lg, symbol, params=params)
                if isinstance(pos, dict) and pos:
                    positions_data = [pos]  # Wrap for consistency
                elif pos is not None:
                    lg.info(f"No active position via {fetch_method} (empty return).")
            except ccxt.PositionNotFound:
                lg.info(f"No active position via {fetch_method} (PositionNotFound).")
            except ccxt.ExchangeError as e:
                # Check for Bybit V5 "position not found" code/message
                no_pos_codes = [110025]
                no_pos_msgs = ["position not found", "no position"]
                if getattr(e, "code", None) in no_pos_codes or any(
                    m in str(e).lower() for m in no_pos_msgs
                ):
                    lg.info(f"No active position via {fetch_method} (Code/Msg: {e}).")
                else:
                    lg.warning(
                        f"Exchange error during {fetch_method}: {e}. Falling back."
                    )
            except ccxt.NotSupported:
                lg.debug(f"{fetch_method} not supported. Falling back.")
            except Exception as e:
                lg.warning(f"Error during {fetch_method}: {e}. Falling back.")

        # Attempt 2: fetchPositions filtered by symbol (if Attempt 1 failed)
        if not positions_data and has_fetch_pos_list:
            fetch_method = f"fetchPositions(symbols=['{symbol}'])"
            lg.debug(f"Attempting {fetch_method} with params: {params}...")
            try:
                fetched = safe_api_call(
                    exchange.fetch_positions, lg, symbols=[symbol], params=params
                )
                if fetched is not None:
                    positions_data = fetched
                if not positions_data:
                    lg.info(f"No active position via {fetch_method}.")
            except ccxt.NotSupported:
                lg.debug(f"{fetch_method} filtering not supported. Falling back.")
            except Exception as e:
                lg.warning(f"Error during {fetch_method}: {e}. Falling back.")

        # Attempt 3: Fetch ALL positions (last resort)
        if (
            not positions_data
            and has_fetch_pos_list
            and fetch_method.endswith("Falling back.")
        ):
            fetch_method = "fetchPositions (all, filtered locally)"
            lg.debug(
                f"Attempting {fetch_method} with params: {params}..."
            )  # Keep category param for V5
            try:
                all_pos = safe_api_call(exchange.fetch_positions, lg, params=params)
                if all_pos:
                    # Filter locally by market_id (more reliable) or symbol
                    positions_data = [
                        p
                        for p in all_pos
                        if p.get("info", {}).get("symbol") == market_id
                        or p.get("symbol") == symbol
                    ]
                    if not positions_data:
                        lg.info(f"No position matched {symbol} after fetching all.")
                else:
                    lg.info("Fallback fetch of all positions returned no data.")
            except Exception as e:
                lg.error(f"Error during {fetch_method}: {e}")

    except Exception as fetch_err:
        lg.error(
            f"{NEON_RED}Unexpected error during position fetch logic: {fetch_err}{RESET}",
            exc_info=True,
        )
        return None

    # --- Process the found position data ---
    if not positions_data:
        lg.info(f"No active position found for {symbol} (Checked via: {fetch_method}).")
        return None

    active_position = None
    # Use min amount step / 2 as zero threshold, default to small Decimal
    analyzer = TradingAnalyzer(pd.DataFrame(), lg, config, market_info)
    min_size_thresh = (
        analyzer.get_min_amount_step() / 2
        if analyzer.get_min_amount_step() > 0
        else Decimal("1e-9")
    )

    for pos_data in positions_data:
        if not isinstance(pos_data, dict):
            continue
        pos_symbol = pos_data.get("symbol")
        info_symbol = pos_data.get("info", {}).get("symbol")
        if pos_symbol != symbol and info_symbol != market_id:
            continue  # Skip if symbol mismatch

        # Extract Size (handle string/float/int, '+' sign)
        size_val = pos_data.get("contracts")  # Standard float/int
        size_src = "'contracts'"
        if size_val is None and isinstance(info := pos_data.get("info"), dict):
            size_val = info.get("size") or info.get("contracts")  # V5 string/float
            size_src = "'info.size' or 'info.contracts'"
        if size_val is None:
            continue

        try:
            size_dec = Decimal(str(size_val).lstrip("+"))
            if abs(size_dec) > min_size_thresh:
                active_position = pos_data.copy()
                active_position["contractsDecimal"] = (
                    size_dec  # Standardized Decimal size
                )
                active_position["sizeSource"] = size_src
                lg.debug(
                    f"Found potential active pos for {symbol} (Size: {size_dec} from {size_src})"
                )
                break  # Found the active position
            else:
                lg.debug(
                    f"Ignoring pos entry for {symbol} (Size {size_dec} <= Threshold {min_size_thresh})"
                )
        except (ValueError, TypeError, InvalidOperation):
            continue

    # --- Post-Process and Standardize Found Position ---
    if active_position:
        try:
            info = active_position.setdefault("info", {})
            size_dec = active_position["contractsDecimal"]

            # Standardize Side
            side = active_position.get("side")
            if side not in ["long", "short"]:
                side = (
                    "long"
                    if size_dec > min_size_thresh
                    else "short"
                    if size_dec < -min_size_thresh
                    else None
                )
                if side is None:
                    lg.warning("Could not determine side reliably.")
                    return None
                active_position["side"] = side

            # Helper for safe Decimal conversion (returns None if invalid/zero)
            def safe_decimal(val_str):
                if val_str is None or str(val_str).strip() in ["", "0", "0.0"]:
                    return None
                try:
                    d = Decimal(str(val_str))
                    return d if d.is_finite() and d > 0 else None
                except:
                    return None

            # Standardize key values as Decimal
            active_position["entryPriceDecimal"] = safe_decimal(
                active_position.get("entryPrice")
                or info.get("entryPrice")
                or info.get("avgPrice")
            )
            active_position["liquidationPriceDecimal"] = safe_decimal(
                active_position.get("liquidationPrice") or info.get("liqPrice")
            )
            active_position["leverageDecimal"] = safe_decimal(
                active_position.get("leverage") or info.get("leverage")
            )
            active_position["stopLossPriceDecimal"] = safe_decimal(info.get("stopLoss"))
            active_position["takeProfitPriceDecimal"] = safe_decimal(
                info.get("takeProfit")
            )
            active_position["trailingStopLossValueDecimal"] = safe_decimal(
                info.get("trailingStop")
            )  # V5: Price Distance
            active_position["trailingStopActivationPriceDecimal"] = safe_decimal(
                info.get("activePrice")
            )  # V5: TSL Activation Price

            # PNL can be negative
            pnl_val = active_position.get("unrealizedPnl") or info.get("unrealisedPnl")
            try:
                active_position["unrealizedPnlDecimal"] = (
                    Decimal(str(pnl_val)) if pnl_val is not None else None
                )
            except:
                active_position["unrealizedPnlDecimal"] = None

            # Timestamp (ms)
            ts_val = info.get("updatedTime") or active_position.get("timestamp")
            try:
                active_position["timestamp_ms"] = (
                    int(ts_val) if ts_val is not None else None
                )
            except:
                active_position["timestamp_ms"] = None

            # Log Formatted Info
            pp = analyzer.get_price_precision()
            ap = analyzer.get_amount_precision_places()

            def fmt(v, p):
                return (
                    f"{v:.{p}f}" if isinstance(v, Decimal) and v.is_finite() else "N/A"
                )

            ts_str = (
                datetime.fromtimestamp(
                    active_position["timestamp_ms"] / 1000, tz=timezone.utc
                ).strftime("%H:%M:%S %Z")
                if active_position["timestamp_ms"]
                else "N/A"
            )

            log_parts = [
                f"{NEON_GREEN}Active {active_position['side'].upper()} ({symbol}):{RESET}",
                f"Size={fmt(abs(size_dec), ap)}",
                f"Entry={fmt(active_position['entryPriceDecimal'], pp)}",
                f"Liq={fmt(active_position['liquidationPriceDecimal'], pp)}",
                f"Lev={fmt(active_position['leverageDecimal'], 1)}x"
                if active_position["leverageDecimal"]
                else "Lev=N/A",
                f"PnL={fmt(active_position['unrealizedPnlDecimal'], 2)}",  # PNL often shown with 2 decimals
                f"SL={fmt(active_position['stopLossPriceDecimal'], pp)}",
                f"TP={fmt(active_position['takeProfitPriceDecimal'], pp)}",
                f"TSL(Val/Act): {fmt(active_position['trailingStopLossValueDecimal'], pp)}/{fmt(active_position['trailingStopActivationPriceDecimal'], pp)}",
                f"(Updated: {ts_str})",
            ]
            lg.info(
                " ".join(p.replace("=N/A", "=None") for p in log_parts)
            )  # Clean up N/A representation
            lg.debug(
                f"Full processed position: {json.dumps(active_position, default=str, indent=2)}"
            )
            return active_position

        except Exception as proc_err:
            lg.error(
                f"Error processing active position for {symbol}: {proc_err}",
                exc_info=True,
            )
            lg.debug(f"Problematic raw data: {active_position}")
            return None
    else:
        # No position found with size above threshold
        return None


def set_leverage_ccxt(
    exchange: ccxt.Exchange,
    symbol: str,
    leverage: int,
    market_info: Dict,
    logger: logging.Logger,
    config: Dict[str, Any],  # Pass config for category
) -> bool:
    """Sets leverage using CCXT, handling Bybit V5 specifics. Returns True on success/already set."""
    lg = logger
    if not market_info.get("is_contract", False):
        lg.debug(f"Leverage setting skipped for {symbol} (not a contract).")
        return True  # Success (no action needed)
    if not isinstance(leverage, int) or leverage <= 0:
        lg.error(f"Invalid leverage {leverage} for {symbol}. Must be positive integer.")
        return False
    if not exchange.has.get("setLeverage"):
        lg.warning(
            f"Exchange {exchange.id} might not support setLeverage. Attempting anyway..."
        )

    market_id = market_info.get("id")
    if not market_id:
        lg.error(f"Cannot set leverage: Market ID missing for {symbol}.")
        return False

    try:
        # Optional: Check current leverage first to avoid redundant calls (adds latency)
        # current_pos = get_open_position(exchange, symbol, lg, config)
        # if current_pos and current_pos.get('leverageDecimal') == Decimal(str(leverage)):
        #      lg.info(f"Leverage for {symbol} already at {leverage}x.")
        #      return True

        lg.info(f"Setting leverage for {symbol} (ID: {market_id}) to {leverage}x...")
        params = {}
        # Bybit V5 requires category and matching buy/sell leverage as strings
        if exchange.id == "bybit":
            params = {
                "category": config.get("bybit_category", "linear"),
                "buyLeverage": str(leverage),
                "sellLeverage": str(leverage),
            }

        response = safe_api_call(
            exchange.set_leverage, lg, leverage, symbol, params=params
        )
        lg.info(
            f"{NEON_GREEN}Successfully set leverage for {symbol} to {leverage}x.{RESET}"
        )
        lg.debug(f"Set leverage response: {response}")
        return True

    except ccxt.ExchangeError as e:
        # Check if error indicates leverage already set (common response)
        err_str = str(e).lower()
        already_set_msgs = [
            "leverage not modified",
            "same leverage",
            "leverage is already",
        ]
        # Bybit V5 codes: 110043 (Leverage not modified)
        already_set_codes = [110043]
        if getattr(e, "code", None) in already_set_codes or any(
            msg in err_str for msg in already_set_msgs
        ):
            lg.info(
                f"Leverage for {symbol} likely already set to {leverage}x (Message: {e})."
            )
            return True  # Consider it a success
        else:
            lg.error(
                f"{NEON_RED}Failed to set leverage for {symbol} to {leverage}x: {e}{RESET}",
                exc_info=True,
            )
            return False
    except Exception as e:
        lg.error(
            f"{NEON_RED}Unexpected error setting leverage for {symbol} to {leverage}x: {e}{RESET}",
            exc_info=True,
        )
        return False


# --- Placeholder for Main Execution Logic ---
# The functions above provide the framework.
# A main loop would typically:
# 1. Load config, setup logger, initialize exchange.
# 2. Enter a loop (with sleep defined by loop_delay_seconds).
# 3. Inside loop:
#    a. Fetch latest klines, price, orderbook.
#    b. Create TradingAnalyzer instance.
#    c. Check for existing open position using get_open_position.
#    d. **If Position Exists:**
#       - Check exit conditions (SL, TP hit based on current price).
#       - Check time-based exit.
#       - Check advanced management (Break Even trigger, Trailing Stop update - requires fetching current price frequently).
#       - Place closing order if condition met.
#    e. **If No Position:**
#       - Generate trading signal using analyzer.generate_trading_signal.
#       - If BUY/SELL signal:
#          - Fetch balance.
#          - Calculate initial TP/SL using analyzer.calculate_entry_tp_sl.
#          - Calculate position size using calculate_position_size.
#          - Set leverage using set_leverage_ccxt (if needed).
#          - Place entry order (market or limit) with attached SL/TP if supported/configured.
#          - Wait (position_confirm_delay_seconds) and confirm position opened.
#    f. Handle exceptions gracefully within the loop.

if __name__ == "__main__":
    # --- Basic Setup ---
    config = load_config(CONFIG_FILE)
    logger = setup_logger(
        "sxsBot", config, level=logging.INFO
    )  # Use INFO for console by default
    exchange = initialize_exchange(config, logger)

    if not exchange:
        logger.critical(
            f"{NEON_RED}Exchange initialization failed. Bot cannot start.{RESET}"
        )
        exit(1)  # Exit if exchange setup fails critically

    logger.info(f"{NEON_CYAN}--- SXS Bot Framework Initialized ---{RESET}")
    logger.info(f"Symbol: {config.get('symbol')}, Interval: {config.get('interval')}")
    logger.info(
        f"Trading Enabled: {config.get('enable_trading')}, Sandbox: {config.get('use_sandbox')}"
    )
    logger.info(
        f"Risk/Trade: {config.get('risk_per_trade'):.2%}, Leverage: {config.get('leverage')}x"
    )
    logger.info(f"Active Weight Set: {config.get('active_weight_set')}")

    # --- Example Usage Snippet (Not a full trading loop) ---
    symbol = config.get("symbol")
    interval = config.get("interval")
    if symbol and interval:
        try:
            logger.info("-" * 30)
            logger.info("Running example data fetch and analysis...")

            # 1. Fetch Klines
            klines_df = fetch_klines_ccxt(
                exchange, symbol, interval, limit=200, logger=logger
            )

            if not klines_df.empty:
                # 2. Get Market Info
                market_info = get_market_info(exchange, symbol, logger)

                if market_info:
                    # 3. Initialize Analyzer
                    analyzer = TradingAnalyzer(klines_df, logger, config, market_info)

                    # 4. Fetch Current Price & Orderbook
                    current_price = fetch_current_price_ccxt(exchange, symbol, logger)
                    ob_limit = config.get("orderbook_limit", 25)
                    orderbook = fetch_orderbook_ccxt(exchange, symbol, ob_limit, logger)

                    if current_price:
                        # 5. Generate Signal
                        signal = analyzer.generate_trading_signal(
                            current_price, orderbook
                        )
                        logger.info(f"Example Signal Generated: {signal}")

                        # 6. Calculate Example TP/SL (if signal is BUY/SELL)
                        if signal in ["BUY", "SELL"]:
                            _, tp_price, sl_price = analyzer.calculate_entry_tp_sl(
                                current_price, signal
                            )
                            logger.info(
                                f"Example TP: {tp_price}, Example SL: {sl_price}"
                            )

                        # 7. Check Open Position
                        open_pos = get_open_position(exchange, symbol, logger, config)
                        if not open_pos:
                            logger.info("No open position found for example check.")
                            # Example Size Calculation (if no position and BUY/SELL signal)
                            if signal in ["BUY", "SELL"] and sl_price:
                                balance = fetch_balance(
                                    exchange, config["quote_currency"], logger, config
                                )
                                if balance is not None:
                                    pos_size = calculate_position_size(
                                        balance,
                                        config["risk_per_trade"],
                                        sl_price,
                                        current_price,
                                        market_info,
                                        exchange,
                                        logger,
                                        config,
                                    )
                                    if pos_size:
                                        logger.info(
                                            f"Example Position Size calculated: {pos_size}"
                                        )
                                    else:
                                        logger.warning(
                                            "Example Size Calculation failed."
                                        )
            else:
                logger.warning("Failed to fetch klines for example run.")

            logger.info("Example analysis complete.")
            logger.info("-" * 30)

        except Exception as e:
            logger.error(
                f"{NEON_RED}Error during example run: {e}{RESET}", exc_info=True
            )

    logger.info("SXS Bot Framework execution finished (example run).")
    # In a real bot, replace the example snippet with the main trading loop.
