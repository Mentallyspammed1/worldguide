# File: config_loader.py
import json
import os
from decimal import Decimal, InvalidOperation
from typing import Any, Dict
import sys

# Import constants and color codes from utils
# Ensure utils.py defines these constants
try:
    from utils import (
        CONFIG_FILE,
        DEFAULT_INDICATOR_PERIODS,
        NEON_RED,
        NEON_YELLOW,
        POSITION_CONFIRM_DELAY_SECONDS,
        RESET_ALL_STYLE,
        RETRY_DELAY_SECONDS,
        VALID_INTERVALS,
    )
except ImportError:
    # Provide fallbacks if utils is missing - prevents crashing but functionality limited
    print(
        "Warning: Failed to import constants from utils.py. Using default fallbacks.",
        file=sys.stderr,
    )
    CONFIG_FILE = "config.json"
    DEFAULT_INDICATOR_PERIODS = {}
    NEON_RED = NEON_YELLOW = RESET_ALL_STYLE = ""
    POSITION_CONFIRM_DELAY_SECONDS = 5
    RETRY_DELAY_SECONDS = 5
    VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]


def _ensure_config_keys(
    config: Dict[str, Any], default_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Recursively ensures all keys from the default config are present in the loaded config."""
    updated_config = config.copy()
    for key, default_value in default_config.items():
        if key not in updated_config:
            updated_config[key] = default_value
        elif isinstance(default_value, dict) and isinstance(
            updated_config.get(key), dict
        ):
            # Recursively check nested dictionaries
            updated_config[key] = _ensure_config_keys(
                updated_config[key], default_value
            )
        # Optional: Add type checking here if needed
    return updated_config


def load_config(filepath: str = CONFIG_FILE) -> Dict[str, Any]:
    """
    Loads configuration from JSON file, creates default if not found,
    ensures all default keys are present, and performs basic validation.

    Args:
        filepath (str): The path to the configuration file.

    Returns:
        Dict[str, Any]: The loaded and validated configuration dictionary.
    """
    default_config = {
        "exchange_id": "bybit",
        "default_market_type": "unified",  # unified, spot, linear, inverse
        "symbols_to_trade": [
            "BTC/USDT:USDT"
        ],  # Example for Bybit USDT Linear Perpetual
        "interval": "5",
        "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
        # --- API & Connection ---
        "api_key": None,  # Loaded from env or needs to be set here
        "api_secret": None,  # Loaded from env or needs to be set here
        "use_sandbox": False,
        "max_api_retries": 3,
        "retry_delay": RETRY_DELAY_SECONDS,
        "api_timeout_ms": 15000,  # Increased default timeout
        "market_cache_duration_seconds": 3600,  # Cache market info for 1 hour
        "circuit_breaker_cooldown_seconds": 300,  # Cooldown after repeated failures
        # --- Trading Parameters ---
        "enable_trading": False,  # SAFETY FIRST: Default False
        "risk_per_trade": 0.01,
        "leverage": 10,
        "max_concurrent_positions": 1,  # Limit per symbol
        "quote_currency": "USDT",
        # --- Order Management ---
        "entry_order_type": "market",  # 'market' or 'limit'
        "limit_order_offset_buy": 0.0005,  # % offset for BUY limit orders (e.g., 0.0005 = 0.05%)
        "limit_order_offset_sell": 0.0005,  # % offset for SELL limit orders
        "order_confirmation_delay_seconds": 0.75,  # Short delay after placing order before fetching status
        "position_confirm_delay_seconds": 5.0,  # Delay after filled order before checking position & setting protection
        "position_confirm_retries": 3,  # Retries for position confirmation
        "close_confirm_delay_seconds": 2.0,  # Delay after closing before checking if position is gone
        "protection_setup_timeout_seconds": 30,  # Max time to wait for position confirm before skipping protection
        "limit_order_timeout_seconds": 300,  # Max time to wait for limit order fill before cancel
        "limit_order_poll_interval_seconds": 5,  # How often to check limit order status
        "limit_order_stale_timeout_seconds": 600,  # Cancel limit order if no update after this long
        "adjust_limit_orders": False,  # Whether to modify limit order price if market moves
        "post_only": False,  # Use post-only for limit orders
        "time_in_force": "GTC",  # Default Time-in-Force (GTC, IOC, FOK, PostOnly)
        # --- Position Management ---
        "enable_trailing_stop": True,
        "trailing_stop_distance_percent": 0.01,  # TSL distance as % of entry/activation price (e.g., 1%)
        "trailing_stop_activation_offset_percent": 0.005,  # Activate TSL after 0.5% profit move
        "tsl_activate_immediately_if_profitable": True,  # Use activePrice=0 if already profitable
        "enable_break_even": True,
        "break_even_trigger_atr_multiple": 1.0,
        "break_even_offset_ticks": 2,
        "time_based_exit_minutes": None,  # Exit after X minutes if set
        "stop_loss_multiple": 1.5,  # Initial SL sizing ATR multiple
        "take_profit_multiple": 2.0,  # Initial TP target ATR multiple
        # --- Analysis & Indicators ---
        "signal_score_threshold": 0.7,
        "kline_limit": 500,
        "min_kline_length": 100,
        "orderbook_limit": 25,
        "min_active_indicators_for_signal": 7,
        "indicators": {
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
        "weight_sets": {
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
            # Add other sets like "scalping" if needed
        },
        "active_weight_set": "default",
        # --- Indicator Periods (Merged with defaults below) ---
        "atr_period": 14,
        "cci_window": 20,
        "cci_constant": 0.015,
        "williams_r_window": 14,
        "mfi_window": 14,
        "stoch_rsi_window": 14,
        "stoch_rsi_rsi_window": 12,
        "stoch_rsi_k": 3,
        "stoch_rsi_d": 3,
        "rsi_period": 14,
        "bollinger_bands_period": 20,
        "bollinger_bands_std_dev": 2.0,
        "sma_10_window": 10,
        "ema_short_period": 9,
        "ema_long_period": 21,
        "momentum_period": 7,
        "volume_ma_period": 15,
        "fibonacci_window": 50,
        "psar_initial_af": 0.02,
        "psar_af_step": 0.02,
        "psar_max_af": 0.2,
        # --- Exchange Specific Options (Optional) ---
        "exchange_options": {  # Passed directly to CCXT constructor
            "options": {
                # Example: 'adjustForTimeDifference': True, # CCXT handles this by default
                # Example: 'brokerId': 'YOUR_BROKER_ID' # If using Bybit broker referral
                # Example: 'recvWindow': 6000 # Override default recvWindow
            }
        },
        # --- Optional Parameters for Specific API Calls ---
        "market_load_params": {},  # e.g., {'category': 'unifiedaccount'} for Bybit V5
        "balance_fetch_params": {},  # e.g., {'accountType': 'UNIFIED'} for Bybit V5
        "fetch_positions_params": {},
        "create_order_params": {},  # Default params added to every create_order call
        "edit_order_params": {},
        "cancel_order_params": {},
        "cancel_all_orders_params": {},
        "fetch_order_params": {},
        "fetch_open_orders_params": {},
        "fetch_closed_orders_params": {},
        "fetch_my_trades_params": {},
        "set_leverage_params": {},
        "set_trading_stop_params": {},  # For Bybit V5 SL/TP/TSL endpoint
        "set_position_mode_params": {},
        "library_log_levels": {  # Optional: Control verbosity of libraries
            # "ccxt": "WARNING",
            # "urllib3": "WARNING"
        },
    }
    # Merge default indicator periods from utils into the default config
    # This allows users to override only specific periods in their config.json
    default_config.update(DEFAULT_INDICATOR_PERIODS)  # Ensure this is defined in utils

    if not os.path.exists(filepath):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            print(
                f"{NEON_YELLOW}Created default config file: {filepath}{RESET_ALL_STYLE}"
            )
            return default_config
        except IOError as e:
            print(
                f"{NEON_RED}Error creating default config file {filepath}: {e}{RESET_ALL_STYLE}"
            )
            return default_config  # Return default if creation failed

    try:
        with open(filepath, encoding="utf-8") as f:
            config_from_file = json.load(f)

        # Ensure all keys from default are present, add missing ones
        updated_config = _ensure_config_keys(config_from_file, default_config)

        # If updates were made, write them back
        if updated_config != config_from_file:
            try:
                with open(filepath, "w", encoding="utf-8") as f_write:
                    json.dump(updated_config, f_write, indent=4, ensure_ascii=False)
                print(
                    f"{NEON_YELLOW}Updated config file '{filepath}' with missing default keys.{RESET_ALL_STYLE}"
                )
            except IOError as e:
                print(
                    f"{NEON_RED}Error writing updated config file {filepath}: {e}{RESET_ALL_STYLE}"
                )

        # --- Perform Basic Validation ---
        save_needed = False  # Flag to save if corrections are made

        # Validate interval
        if updated_config.get("interval") not in VALID_INTERVALS:
            print(
                f"{NEON_RED}Invalid interval '{updated_config.get('interval')}'. Using default '{default_config['interval']}'.{RESET_ALL_STYLE}"
            )
            updated_config["interval"] = default_config["interval"]
            save_needed = True

        # Validate exchange_id
        if not isinstance(
            updated_config.get("exchange_id"), str
        ) or not updated_config.get("exchange_id"):
            print(
                f"{NEON_RED}Invalid 'exchange_id'. Using default '{default_config['exchange_id']}'.{RESET_ALL_STYLE}"
            )
            updated_config["exchange_id"] = default_config["exchange_id"]
            save_needed = True

        # Validate default_market_type
        market_type = updated_config.get("default_market_type")
        if not isinstance(market_type, str) or not market_type:
            print(
                f"{NEON_RED}Invalid 'default_market_type'. Using default '{default_config['default_market_type']}'.{RESET_ALL_STYLE}"
            )
            updated_config["default_market_type"] = default_config[
                "default_market_type"
            ]
            save_needed = True
        elif market_type.lower() not in [
            "spot",
            "margin",
            "future",
            "swap",
            "option",
            "unified",
        ]:
            print(
                f"{NEON_YELLOW}Warning: 'default_market_type' '{market_type}' is not standard. Ensure correctness.{RESET_ALL_STYLE}"
            )

        # Validate entry order type
        if updated_config.get("entry_order_type") not in [
            "market",
            "limit",
            "conditional",
        ]:  # Added conditional
            print(
                f"{NEON_RED}Invalid entry_order_type '{updated_config.get('entry_order_type')}'. Using 'market'.{RESET_ALL_STYLE}"
            )
            updated_config["entry_order_type"] = "market"
            save_needed = True

        # Simplified numeric validator function
        def _validate_numeric(
            key, cfg, default, min_v=None, max_v=None, is_int=False, allow_none=False
        ):
            nonlocal save_needed
            value = cfg.get(key)
            if allow_none and value is None:
                return True
            if isinstance(value, bool):
                return False  # No bools for numeric fields
            try:
                num_val = Decimal(str(value))
                if is_int and num_val != Decimal(int(num_val)):
                    raise ValueError("Must be integer")
                if min_v is not None and num_val < Decimal(str(min_v)):
                    raise ValueError(f"Less than min {min_v}")
                if max_v is not None and num_val > Decimal(str(max_v)):
                    raise ValueError(f"Greater than max {max_v}")
                return True  # Valid
            except (InvalidOperation, ValueError, TypeError) as e:
                print(
                    f"{NEON_RED}Validation failed for '{key}' ({value}): {e}. Using default '{default}'.{RESET_ALL_STYLE}"
                )
                cfg[key] = default
                save_needed = True
                return False

        # Use the validator
        _validate_numeric(
            "max_api_retries",
            updated_config,
            default_config["max_api_retries"],
            min_v=0,
            is_int=True,
        )
        _validate_numeric(
            "retry_delay", updated_config, default_config["retry_delay"], min_v=0
        )
        _validate_numeric(
            "api_timeout_ms",
            updated_config,
            default_config["api_timeout_ms"],
            min_v=1000,
            is_int=True,
        )
        _validate_numeric(
            "risk_per_trade",
            updated_config,
            default_config["risk_per_trade"],
            min_v=0,
            max_v=1,
        )
        _validate_numeric(
            "leverage", updated_config, default_config["leverage"], min_v=1
        )  # Allow float leverage? Usually int. Check exchange. Let's keep > 0.
        _validate_numeric(
            "max_concurrent_positions",
            updated_config,
            default_config["max_concurrent_positions"],
            min_v=1,
            is_int=True,
        )
        _validate_numeric(
            "signal_score_threshold",
            updated_config,
            default_config["signal_score_threshold"],
            min_v=0,
        )
        _validate_numeric(
            "orderbook_limit",
            updated_config,
            default_config["orderbook_limit"],
            min_v=1,
            is_int=True,
        )
        _validate_numeric(
            "kline_limit",
            updated_config,
            default_config["kline_limit"],
            min_v=10,
            is_int=True,
        )  # Need reasonable min
        _validate_numeric(
            "min_kline_length",
            updated_config,
            default_config["min_kline_length"],
            min_v=1,
            is_int=True,
        )
        _validate_numeric(
            "position_confirm_delay_seconds",
            updated_config,
            default_config["position_confirm_delay_seconds"],
            min_v=0,
        )
        _validate_numeric(
            "time_based_exit_minutes",
            updated_config,
            default_config["time_based_exit_minutes"],
            min_v=1,
            allow_none=True,
        )

        # Validate TSL parameters
        _validate_numeric(
            "trailing_stop_distance_percent",
            updated_config,
            default_config["trailing_stop_distance_percent"],
            min_v=1e-9,
        )  # Must be > 0
        _validate_numeric(
            "trailing_stop_activation_offset_percent",
            updated_config,
            default_config["trailing_stop_activation_offset_percent"],
            min_v=0,
        )  # Can be 0

        # Validate BE parameters
        _validate_numeric(
            "break_even_trigger_atr_multiple",
            updated_config,
            default_config["break_even_trigger_atr_multiple"],
            min_v=0,
        )
        _validate_numeric(
            "break_even_offset_ticks",
            updated_config,
            default_config["break_even_offset_ticks"],
            min_v=0,
            is_int=True,
        )

        # Validate Indicator Periods
        for key in DEFAULT_INDICATOR_PERIODS:
            # Most periods should be > 0 integers, but some (like std dev, constants, PSAR AF) can be float/Decimal > 0.
            is_int_param = isinstance(default_config[key], int) and key not in [
                "bollinger_bands_std_dev",
                "cci_constant",
                "psar_initial_af",
                "psar_af_step",
                "psar_max_af",
            ]
            min_value = (
                1 if is_int_param else Decimal("1e-9")
            )  # Int periods >= 1, float params > 0
            _validate_numeric(
                key,
                updated_config,
                default_config[key],
                min_v=min_value,
                is_int=is_int_param,
            )

        # Validate symbols_to_trade list
        symbols = updated_config.get("symbols_to_trade")
        if (
            not isinstance(symbols, list)
            or not symbols
            or not all(isinstance(s, str) and s for s in symbols)
        ):
            print(
                f"{NEON_RED}Invalid 'symbols_to_trade': Must be a non-empty list of non-empty strings.{RESET_ALL_STYLE}"
            )
            updated_config["symbols_to_trade"] = default_config["symbols_to_trade"]
            save_needed = True

        # Save corrected config if needed
        if save_needed:
            try:
                with open(filepath, "w", encoding="utf-8") as f_write:
                    json.dump(updated_config, f_write, indent=4, ensure_ascii=False)
                print(
                    f"{NEON_YELLOW}Corrected invalid values and saved config: {filepath}{RESET_ALL_STYLE}"
                )
            except IOError as e:
                print(
                    f"{NEON_RED}Error writing corrected config file {filepath}: {e}{RESET_ALL_STYLE}"
                )

        return updated_config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(
            f"{NEON_RED}Error loading config file {filepath}: {e}. Using default config.{RESET_ALL_STYLE}"
        )
        # Attempt to recreate default if loading failed badly
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            print(
                f"{NEON_YELLOW}Created default config file: {filepath}{RESET_ALL_STYLE}"
            )
        except IOError as e_create:
            print(
                f"{NEON_RED}Error creating default config file after load error: {e_create}{RESET_ALL_STYLE}"
            )
        return default_config

    except Exception as e:  # Catch other potential errors like permission denied
        print(
            f"{NEON_RED}An unexpected error occurred during config loading: {e}{RESET_ALL_STYLE}"
        )
        print(f"{NEON_YELLOW}Returning default configuration.{RESET_ALL_STYLE}")
        return default_config
