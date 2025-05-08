```python
# File: config_loader.py
import json
import os
import sys
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Set, Tuple, Union

# Import constants and color codes from utils
# Ensure utils.py defines these constants
try:
    from utils import (
        CONFIG_FILE,
        DEFAULT_INDICATOR_PERIODS,
        NEON_RED,
        NEON_YELLOW,
        POSITION_CONFIRM_DELAY_SECONDS,  # Used if needed by default_config
        RESET_ALL_STYLE,
        RETRY_DELAY_SECONDS,
        VALID_INTERVALS,  # Should be a tuple or set for efficiency
    )
except ImportError:
    # Provide fallbacks if utils is missing - prevents crashing but functionality limited
    print(
        "Warning: Failed to import constants from utils.py. Using default fallbacks.",
        file=sys.stderr,
    )
    CONFIG_FILE = "config.json"
    DEFAULT_INDICATOR_PERIODS: Dict[str, Union[int, float]] = {}
    NEON_RED = ""
    NEON_YELLOW = ""
    RESET_ALL_STYLE = ""
    POSITION_CONFIRM_DELAY_SECONDS = 5  # Default int, config uses float
    RETRY_DELAY_SECONDS = 5
    # Define as a tuple for immutability, convert to set for efficient lookup
    VALID_INTERVALS: Tuple[str, ...] = (
        "1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M",
    )

# For validation, using a set for intervals is more efficient
VALID_INTERVALS_SET: Set[str] = set(VALID_INTERVALS)

# Standard market types (lowercase for consistent comparison)
STANDARD_MARKET_TYPES: Set[str] = {
    "spot", "margin", "future", "swap", "option", "unified",
}
# Valid entry order types
VALID_ENTRY_ORDER_TYPES: Set[str] = {"market", "limit", "conditional"}

# Indicator parameter keys that are expected to be floats (or can be)
FLOAT_INDICATOR_PARAM_KEYS: Set[str] = {
    "bollinger_bands_std_dev",
    "cci_constant",
    "psar_initial_af",
    "psar_af_step",
    "psar_max_af",
}


def _ensure_config_keys(
    current_config: Dict[str, Any], default_config_template: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Recursively ensures all keys from the default config template are present
    in the loaded config. Adds missing keys with their default values.
    Does not overwrite existing keys unless they are dictionaries that need
    recursive checking.
    """
    updated_config = current_config.copy()
    for key, default_value in default_config_template.items():
        if key not in updated_config:
            updated_config[key] = default_value
        elif isinstance(default_value, dict) and \
             isinstance(updated_config.get(key), dict):
            # Recursively check nested dictionaries
            updated_config[key] = _ensure_config_keys(
                updated_config[key], default_value
            )
    return updated_config


def _write_config_to_file(
    filepath: str, config_data: Dict[str, Any], success_message: str
) -> bool:
    """Helper to write config data to a JSON file."""
    try:
        with open(filepath, "w", encoding="utf-8") as f_write:
            json.dump(config_data, f_write, indent=4, ensure_ascii=False)
        # Success messages to stdout, warnings/errors to stderr
        print(f"{NEON_YELLOW}{success_message.format(filepath=filepath)}{RESET_ALL_STYLE}")
        return True
    except IOError as e:
        print(
            f"{NEON_RED}Error writing to config file {filepath}: {e}{RESET_ALL_STYLE}",
            file=sys.stderr,
        )
        return False


def _validate_numeric_config_value(
    key: str,
    config_dict: Dict[str, Any],
    default_value: Any,
    save_needed_flag: List[bool],  # Pass as a list to modify its boolean element
    min_value: Union[int, float, Decimal, None] = None,
    max_value: Union[int, float, Decimal, None] = None,
    is_integer: bool = False,
    allow_none: bool = False,
) -> None:
    """
    Validates a numeric configuration value.
    Updates config_dict and sets save_needed_flag[0] if correction occurs.
    """
    value = config_dict.get(key)

    if allow_none and value is None:
        return  # None is allowed and present

    # Boolean values are not valid numeric types here
    if isinstance(value, bool):
        error_message = f"Boolean value '{value}' not valid"
    else:
        try:
            # Use Decimal for precise arithmetic and comparison
            num_val = Decimal(str(value)) # str(value) is safer for Decimal
            if is_integer and num_val != Decimal(int(num_val)):
                raise ValueError("Value must be an integer.")
            if min_value is not None and num_val < Decimal(str(min_value)):
                raise ValueError(f"Value '{num_val}' is less than minimum '{min_value}'.")
            if max_value is not None and num_val > Decimal(str(max_value)):
                raise ValueError(f"Value '{num_val}' is greater than maximum '{max_value}'.")
            # If value was float, but int required, convert (e.g. 10.0 for int leverage)
            # Or if value was string "10", convert to actual number.
            config_dict[key] = int(num_val) if is_integer else float(num_val)
            return  # Valid
        except (InvalidOperation, ValueError, TypeError) as e:
            error_message = str(e)

    # If validation failed or value was boolean
    print(
        f"{NEON_RED}Validation failed for '{key}' (value: '{value}'): {error_message}. "
        f"Using default: '{default_value}'.{RESET_ALL_STYLE}",
        file=sys.stderr,
    )
    config_dict[key] = default_value
    save_needed_flag[0] = True


def load_config(filepath: str = CONFIG_FILE) -> Dict[str, Any]:
    """
    Loads configuration from a JSON file.
    - Creates a default config file if it doesn't exist.
    - Ensures all keys from a default template are present in the loaded config.
    - Performs validation on critical configuration values.
    - Updates the config file if keys were added or values were corrected.

    Args:
        filepath: The path to the configuration file.

    Returns:
        The loaded, potentially updated, and validated configuration dictionary.
    """
    # Define the default configuration structure
    # This acts as a template and provides default values
    default_config: Dict[str, Any] = {
        "exchange_id": "bybit",
        "default_market_type": "unified",  # unified, spot, linear, inverse
        "symbols_to_trade": ["BTC/USDT:USDT"],
        "interval": "5",
        "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
        # --- API & Connection ---
        "api_key": None,
        "api_secret": None,
        "use_sandbox": False,
        "max_api_retries": 3,
        "retry_delay": RETRY_DELAY_SECONDS,  # From utils or fallback
        "api_timeout_ms": 15000,
        "market_cache_duration_seconds": 3600,
        "circuit_breaker_cooldown_seconds": 300,
        # --- Trading Parameters ---
        "enable_trading": False,
        "risk_per_trade": 0.01,
        "leverage": 10,
        "max_concurrent_positions": 1,
        "quote_currency": "USDT",
        # --- Order Management ---
        "entry_order_type": "market",
        "limit_order_offset_buy": 0.0005,
        "limit_order_offset_sell": 0.0005,
        "order_confirmation_delay_seconds": 0.75,
        "position_confirm_delay_seconds": 5.0, # Using float for precision
        "position_confirm_retries": 3,
        "close_confirm_delay_seconds": 2.0,
        "protection_setup_timeout_seconds": 30,
        "limit_order_timeout_seconds": 300,
        "limit_order_poll_interval_seconds": 5,
        "limit_order_stale_timeout_seconds": 600,
        "adjust_limit_orders": False,
        "post_only": False,
        "time_in_force": "GTC",
        # --- Position Management ---
        "enable_trailing_stop": True,
        "trailing_stop_distance_percent": 0.01,
        "trailing_stop_activation_offset_percent": 0.005,
        "tsl_activate_immediately_if_profitable": True,
        "enable_break_even": True,
        "break_even_trigger_atr_multiple": 1.0,
        "break_even_offset_ticks": 2,
        "time_based_exit_minutes": None,
        "stop_loss_multiple": 1.5,
        "take_profit_multiple": 2.0,
        # --- Analysis & Indicators ---
        "signal_score_threshold": 0.7,
        "kline_limit": 500,
        "min_kline_length": 100,
        "orderbook_limit": 25,
        "min_active_indicators_for_signal": 7,
        "indicators": {
            "ema_alignment": True, "momentum": True, "volume_confirmation": True,
            "stoch_rsi": True, "rsi": True, "bollinger_bands": True, "vwap": True,
            "cci": True, "wr": True, "psar": True, "sma_10": True, "mfi": True,
            "orderbook": True,
        },
        "weight_sets": {
            "default": {
                "ema_alignment": 0.3, "momentum": 0.2, "volume_confirmation": 0.1,
                "stoch_rsi": 0.4, "rsi": 0.3, "bollinger_bands": 0.2, "vwap": 0.3,
                "cci": 0.2, "wr": 0.2, "psar": 0.3, "sma_10": 0.1, "mfi": 0.2,
                "orderbook": 0.1,
            },
        },
        "active_weight_set": "default",
        # --- Indicator Periods (can be overridden by DEFAULT_INDICATOR_PERIODS from utils) ---
        "atr_period": 14, "cci_window": 20, "cci_constant": 0.015,
        "williams_r_window": 14, "mfi_window": 14, "stoch_rsi_window": 14,
        "stoch_rsi_rsi_window": 12, "stoch_rsi_k": 3, "stoch_rsi_d": 3,
        "rsi_period": 14, "bollinger_bands_period": 20, "bollinger_bands_std_dev": 2.0,
        "sma_10_window": 10, "ema_short_period": 9, "ema_long_period": 21,
        "momentum_period": 7, "volume_ma_period": 15, "fibonacci_window": 50,
        "psar_initial_af": 0.02, "psar_af_step": 0.02, "psar_max_af": 0.2,
        # --- Exchange Specific Options ---
        "exchange_options": {"options": {}},
        # --- Optional Parameters for Specific API Calls ---
        "market_load_params": {}, "balance_fetch_params": {},
        "fetch_positions_params": {}, "create_order_params": {},
        "edit_order_params": {}, "cancel_order_params": {},
        "cancel_all_orders_params": {}, "fetch_order_params": {},
        "fetch_open_orders_params": {}, "fetch_closed_orders_params": {},
        "fetch_my_trades_params": {}, "set_leverage_params": {},
        "set_trading_stop_params": {}, "set_position_mode_params": {},
        "library_log_levels": {},
    }
    # Merge default indicator periods from utils.py into the default_config.
    # This allows utils.py to be the source of truth for these specific defaults.
    default_config.update(DEFAULT_INDICATOR_PERIODS)

    if not os.path.exists(filepath):
        if _write_config_to_file(
            filepath, default_config, "Created default config file: {filepath}"
        ):
            return default_config
        else:
            # Fallback to in-memory default if file creation fails
            print(
                f"{NEON_RED}Returning in-memory default config due to creation error.{RESET_ALL_STYLE}",
                file=sys.stderr
            )
            return default_config

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            config_from_file = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(
            f"{NEON_RED}Error loading config file {filepath}: {e}. "
            f"Attempting to recreate with defaults.{RESET_ALL_STYLE}",
            file=sys.stderr,
        )
        if _write_config_to_file(
            filepath, default_config, "Recreated default config file: {filepath}"
        ):
            return default_config
        else:
            print(
                f"{NEON_RED}Returning in-memory default config due to load/recreation error.{RESET_ALL_STYLE}",
                file=sys.stderr
            )
            return default_config # Fallback if reading and recreating fails

    # Ensure all keys from default_config are present in the loaded config
    # This adds new default options to existing user configs without overwriting values
    processed_config = _ensure_config_keys(config_from_file, default_config)

    if processed_config != config_from_file:
        _write_config_to_file(
            filepath,
            processed_config,
            "Updated config file '{filepath}' with new/missing default keys.",
        )
        # Continue with validation using processed_config

    # --- Perform Validations ---
    # Use a list for save_needed_flag so its change is visible in _validate_numeric_config_value
    save_needed_flag = [False]

    # Validate 'interval'
    current_interval = processed_config.get("interval")
    if current_interval not in VALID_INTERVALS_SET:
        print(
            f"{NEON_RED}Invalid interval '{current_interval}'. "
            f"Using default '{default_config['interval']}'.{RESET_ALL_STYLE}",
            file=sys.stderr,
        )
        processed_config["interval"] = default_config["interval"]
        save_needed_flag[0] = True

    # Validate 'exchange_id' (must be a non-empty string)
    current_exchange_id = processed_config.get("exchange_id")
    if not isinstance(current_exchange_id, str) or not current_exchange_id.strip():
        print(
            f"{NEON_RED}Invalid 'exchange_id'. Must be a non-empty string. "
            f"Using default '{default_config['exchange_id']}'.{RESET_ALL_STYLE}",
            file=sys.stderr,
        )
        processed_config["exchange_id"] = default_config["exchange_id"]
        save_needed_flag[0] = True

    # Validate 'default_market_type'
    market_type = processed_config.get("default_market_type", "").lower()
    if not market_type:
        print(
            f"{NEON_RED}Missing 'default_market_type'. "
            f"Using default '{default_config['default_market_type']}'.{RESET_ALL_STYLE}",
            file=sys.stderr,
        )
        processed_config["default_market_type"] = default_config["default_market_type"]
        save_needed_flag[0] = True
    elif market_type not in STANDARD_MARKET_TYPES:
        print(
            f"{NEON_YELLOW}Warning: 'default_market_type' ('{market_type}') "
            f"is not a standard CCXT type. Ensure it's supported by your exchange.{RESET_ALL_STYLE}",
            file=sys.stderr, # Warnings also to stderr
        )

    # Validate 'entry_order_type'
    current_entry_order_type = processed_config.get("entry_order_type")
    if current_entry_order_type not in VALID_ENTRY_ORDER_TYPES:
        print(
            f"{NEON_RED}Invalid 'entry_order_type' ('{current_entry_order_type}'). "
            f"Using default 'market'.{RESET_ALL_STYLE}",
            file=sys.stderr,
        )
        processed_config["entry_order_type"] = "market" # Hardcoded default for this rule
        save_needed_flag[0] = True

    # Validate numeric parameters
    numeric_params_to_validate = [
        ("max_api_retries", {"min_value": 0, "is_integer": True}),
        ("retry_delay", {"min_value": 0}),
        ("api_timeout_ms", {"min_value": 1000, "is_integer": True}),
        ("risk_per_trade", {"min_value": 0, "max_value": 1}),
        ("leverage", {"min_value": 1}), # Typically int, but some exchanges might allow float
        ("max_concurrent_positions", {"min_value": 1, "is_integer": True}),
        ("signal_score_threshold", {"min_value": 0, "max_value": 1}), # Assuming 0-1 range
        ("orderbook_limit", {"min_value": 1, "is_integer": True}),
        ("kline_limit", {"min_value": 10, "is_integer": True}),
        ("min_kline_length", {"min_value": 1, "is_integer": True}),
        ("position_confirm_delay_seconds", {"min_value": 0}),
        ("trailing_stop_distance_percent", {"min_value": Decimal("1e-9")}),
        ("trailing_stop_activation_offset_percent", {"min_value": 0}),
        ("break_even_trigger_atr_multiple", {"min_value": 0}),
        ("break_even_offset_ticks", {"min_value": 0, "is_integer": True}),
        ("time_based_exit_minutes", {"min_value": 1, "allow_none": True, "is_integer": True}),
    ]

    for key, kwargs in numeric_params_to_validate:
        _validate_numeric_config_value(
            key, processed_config, default_config[key], save_needed_flag, **kwargs
        )

    # Validate Indicator Periods from default_config (which includes DEFAULT_INDICATOR_PERIODS)
    # Iterate over keys that are known indicator periods defined in default_config
    indicator_period_keys = DEFAULT_INDICATOR_PERIODS.keys() | {
        k for k in default_config if "period" in k or "window" in k or "_af" in k or "constant" in k
    }

    for key in indicator_period_keys:
        if key not in default_config: continue # Skip if not a primary default_config key

        # Determine if integer and minimum value
        is_int = isinstance(default_config[key], int) and \
                   key not in FLOAT_INDICATOR_PARAM_KEYS
        min_val = 1 if is_int else Decimal("1e-9") # Integer periods >= 1, float params > 0

        _validate_numeric_config_value(
            key,
            processed_config,
            default_config[key],
            save_needed_flag,
            min_value=min_val,
            is_integer=is_int,
        )

    # Validate 'symbols_to_trade' list
    symbols = processed_config.get("symbols_to_trade")
    if not (
        isinstance(symbols, list) and
        all(isinstance(s, str) and s.strip() for s in symbols)
        # Optionally: and symbols # To ensure list is not empty, but an empty list might be valid
    ): # Current default is not empty, so let's assume it should not be empty.
        print(
            f"{NEON_RED}Invalid 'symbols_to_trade': Must be a list of non-empty strings. "
            f"Using default: {default_config['symbols_to_trade']}.{RESET_ALL_STYLE}",
            file=sys.stderr,
        )
        processed_config["symbols_to_trade"] = default_config["symbols_to_trade"]
        save_needed_flag[0] = True


    if save_needed_flag[0]:
        _write_config_to_file(
            filepath,
            processed_config,
            "Corrected invalid values and saved updated config to: {filepath}",
        )

    return processed_config

```