# File: config_loader.py
import json
import os
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional, Tuple, Union

# Import constants and utilities
import constants
from utils import ZoneInfo # Use the potentially patched ZoneInfo

def load_config(filepath: str = constants.CONFIG_FILE) -> Tuple[Dict[str, Any], Any]:
    """Load configuration from JSON file, creating default if not found,
       and ensuring all default keys are present with validation.

    Returns:
        A tuple containing: (config_dictionary, timezone_object)
    """
    default_config = {
        "symbols_to_trade": ["FARTCOIN/USDT:USDT"],
        "interval": "5",
        "retry_delay": constants.RETRY_DELAY_SECONDS,
        "orderbook_limit": 25,
        "signal_score_threshold": 1.5,
        "stoch_rsi_oversold_threshold": 25,
        "stoch_rsi_overbought_threshold": 75,
        "stop_loss_multiple": 1.8,
        "take_profit_multiple": 0.7,
        "volume_confirmation_multiplier": 1.5,
        "scalping_signal_threshold": 2.5,
        "enable_trading": True,
        "use_sandbox": False,
        "risk_per_trade": 0.01,
        "leverage": 20,
        "max_concurrent_positions": 1,
        "quote_currency": "USDT",
        "entry_order_type": "market",
        "limit_order_offset_buy": 0.0005,
        "limit_order_offset_sell": 0.0005,
        "enable_trailing_stop": True,
        "trailing_stop_callback_rate": 0.005,
        "trailing_stop_activation_percentage": 0.003,
        "enable_break_even": True,
        "break_even_trigger_atr_multiple": 1.0,
        "break_even_offset_ticks": 2,
        "position_confirm_delay_seconds": constants.POSITION_CONFIRM_DELAY_SECONDS,
        "time_based_exit_minutes": None,
        "indicators": {
            "ema_alignment": True, "momentum": True, "volume_confirmation": True,
            "stoch_rsi": True, "rsi": True, "bollinger_bands": True, "vwap": True,
            "cci": True, "wr": True, "psar": True, "sma_10": True, "mfi": True,
            "orderbook": True,
        },
        "weight_sets": {
            "scalping": {
                "ema_alignment": 0.2, "momentum": 0.3, "volume_confirmation": 0.2,
                "stoch_rsi": 0.6, "rsi": 0.2, "bollinger_bands": 0.3, "vwap": 0.4,
                "cci": 0.3, "wr": 0.3, "psar": 0.2, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.15,
            },
            "default": {
                "ema_alignment": 0.3, "momentum": 0.2, "volume_confirmation": 0.1,
                "stoch_rsi": 0.4, "rsi": 0.3, "bollinger_bands": 0.2, "vwap": 0.3,
                "cci": 0.2, "wr": 0.2, "psar": 0.3, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.1,
            }
        },
        "active_weight_set": "default"
    }
    # Add default indicator periods to the default config
    default_config.update(constants.DEFAULT_INDICATOR_PERIODS)

    # --- Timezone Setup ---
    # Use environment variable if set, otherwise use default from constants
    configured_timezone_str = os.getenv("TIMEZONE", constants.DEFAULT_TIMEZONE_STR)
    try:
        timezone_obj = ZoneInfo(configured_timezone_str)
    except Exception as tz_err:
        print(f"{constants.NEON_YELLOW}Warning: Could not load timezone '{configured_timezone_str}'. Using UTC. Error: {tz_err}{constants.RESET}")
        timezone_obj = ZoneInfo("UTC")


    # --- Load or Create Config File ---
    if not os.path.exists(filepath):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            print(f"{constants.NEON_YELLOW}Created default config file: {filepath}{constants.RESET}")
            return default_config, timezone_obj # Return default config and timezone
        except IOError as e:
            print(f"{constants.NEON_RED}Error creating default config file {filepath}: {e}{constants.RESET}")
            return default_config, timezone_obj # Return default if creation failed

    # --- Load Existing Config and Validate ---
    try:
        with open(filepath, encoding="utf-8") as f:
            config_from_file = json.load(f)

        # Ensure all keys from default are present, add missing ones
        updated_config = _ensure_config_keys(config_from_file, default_config)
        # If updates were made, write them back
        if updated_config != config_from_file:
             try:
                 with open(filepath, "w", encoding="utf-8") as f_write:
                    json.dump(updated_config, f_write, indent=4)
                 print(f"{constants.NEON_YELLOW}Updated config file with missing default keys: {filepath}{constants.RESET}")
             except IOError as e:
                 print(f"{constants.NEON_RED}Error writing updated config file {filepath}: {e}{constants.RESET}")

        # --- Validate crucial values after loading/updating ---
        save_needed = False # Flag to save config if corrections are made

        # Validate interval
        if updated_config.get("interval") not in constants.VALID_INTERVALS:
            print(f"{constants.NEON_RED}Invalid interval '{updated_config.get('interval')}' found in config. Using default '{default_config['interval']}'.{constants.RESET}")
            updated_config["interval"] = default_config["interval"]
            save_needed = True

        # Validate entry order type
        if updated_config.get("entry_order_type") not in ["market", "limit"]:
             print(f"{constants.NEON_RED}Invalid entry_order_type '{updated_config.get('entry_order_type')}' in config. Using default 'market'.{constants.RESET}")
             updated_config["entry_order_type"] = "market"
             save_needed = True

        # Validate numeric ranges
        def validate_numeric(key: str, min_val: Optional[Union[int, float]] = None, max_val: Optional[Union[int, float]] = None, is_int: bool = False, allow_none: bool = False):
            nonlocal save_needed
            value = updated_config.get(key)
            default_value = default_config.get(key)
            valid = False

            if allow_none and value is None:
                valid = True
            elif isinstance(value, bool):
                 print(f"{constants.NEON_RED}Config value '{key}' ({value}) has invalid type bool. Expected numeric.")
            elif isinstance(value, (int, float)):
                if is_int and not isinstance(value, int):
                     print(f"{constants.NEON_RED}Config value '{key}' ({value}) must be an integer.")
                else:
                    try:
                        val_decimal = Decimal(str(value))
                        min_decimal = Decimal(str(min_val)) if min_val is not None else None
                        max_decimal = Decimal(str(max_val)) if max_val is not None else None

                        if (min_decimal is None or val_decimal >= min_decimal) and \
                           (max_decimal is None or val_decimal <= max_decimal):
                            valid = True
                        else:
                            range_str = ""
                            if min_val is not None: range_str += f" >= {min_val}"
                            if max_val is not None: range_str += f" <= {max_val}"
                            print(f"{constants.NEON_RED}Config value '{key}' ({value}) out of range ({range_str.strip()}).")
                    except InvalidOperation:
                         print(f"{constants.NEON_RED}Config value '{key}' ({value}) could not be converted to Decimal for validation.")
            else:
                 print(f"{constants.NEON_RED}Config value '{key}' ({value}) has invalid type {type(value)}. Expected numeric.")

            if not valid:
                print(f"{constants.NEON_YELLOW}Using default value for '{key}': {default_value}{constants.RESET}")
                updated_config[key] = default_value
                save_needed = True

        # Validate core settings
        validate_numeric("retry_delay", min_val=0)
        validate_numeric("risk_per_trade", min_val=0, max_val=1)
        validate_numeric("leverage", min_val=1, is_int=True)
        validate_numeric("max_concurrent_positions", min_val=1, is_int=True)
        validate_numeric("signal_score_threshold", min_val=0)
        validate_numeric("stop_loss_multiple", min_val=0)
        validate_numeric("take_profit_multiple", min_val=0)
        validate_numeric("trailing_stop_callback_rate", min_val=1e-9) # Must be > 0
        validate_numeric("trailing_stop_activation_percentage", min_val=0)
        validate_numeric("break_even_trigger_atr_multiple", min_val=0)
        validate_numeric("break_even_offset_ticks", min_val=0, is_int=True)
        validate_numeric("position_confirm_delay_seconds", min_val=0)
        validate_numeric("time_based_exit_minutes", min_val=1, allow_none=True)
        validate_numeric("orderbook_limit", min_val=1, is_int=True)
        validate_numeric("limit_order_offset_buy", min_val=0)
        validate_numeric("limit_order_offset_sell", min_val=0)
        validate_numeric("bollinger_bands_std_dev", min_val=0)

        # Validate indicator periods
        for key, default_val in constants.DEFAULT_INDICATOR_PERIODS.items():
             is_int_param = isinstance(default_val, int) or key in ["stoch_rsi_k", "stoch_rsi_d"]
             min_value = 1 if is_int_param else 1e-9
             validate_numeric(key, min_val=min_value, is_int=is_int_param)

        # Validate symbols_to_trade
        symbols = updated_config.get("symbols_to_trade")
        if not isinstance(symbols, list) or not symbols or not all(isinstance(s, str) for s in symbols):
             print(f"{constants.NEON_RED}Invalid 'symbols_to_trade' format in config. Must be a non-empty list of strings.{constants.RESET}")
             updated_config["symbols_to_trade"] = default_config["symbols_to_trade"]
             print(f"{constants.NEON_YELLOW}Using default value for 'symbols_to_trade': {updated_config['symbols_to_trade']}{constants.RESET}")
             save_needed = True

        # Save corrected config if needed
        if save_needed:
             try:
                 with open(filepath, "w", encoding="utf-8") as f_write:
                    json.dump(updated_config, f_write, indent=4)
                 print(f"{constants.NEON_YELLOW}Corrected invalid values and saved updated config file: {filepath}{constants.RESET}")
             except IOError as e:
                 print(f"{constants.NEON_RED}Error writing corrected config file {filepath}: {e}{constants.RESET}")

        return updated_config, timezone_obj # Return loaded/validated config and timezone

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"{constants.NEON_RED}Error loading config file {filepath}: {e}. Using default config.{constants.RESET}")
        try:
            # Attempt to recreate default if loading failed badly
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            print(f"{constants.NEON_YELLOW}Created default config file: {filepath}{constants.RESET}")
        except IOError as e_create:
             print(f"{constants.NEON_RED}Error creating default config file after load error: {e_create}{constants.RESET}")
        return default_config, timezone_obj # Return default config and timezone


def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively ensures all keys from the default config are present in the loaded config."""
    updated_config = config.copy()
    for key, default_value in default_config.items():
        if key not in updated_config:
            updated_config[key] = default_value
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # Recursively check nested dictionaries
            updated_config[key] = _ensure_config_keys(updated_config[key], default_value)
    return updated_config
```

```python
