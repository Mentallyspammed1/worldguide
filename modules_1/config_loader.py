# File: config_loader.py
import json
import os
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional, Union

# Import constants and color codes from utils
from utils import (CONFIG_FILE, DEFAULT_INDICATOR_PERIODS, NEON_RED,
                   NEON_YELLOW, POSITION_CONFIRM_DELAY_SECONDS,
                   RESET_ALL_STYLE, RETRY_DELAY_SECONDS, VALID_INTERVALS) # Corrected RESET to RESET_ALL_STYLE


def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively ensures all keys from the default config are present in the loaded config."""
    updated_config = config.copy()
    for key, default_value in default_config.items():
        if key not in updated_config:
            updated_config[key] = default_value
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # Recursively check nested dictionaries
            updated_config[key] = _ensure_config_keys(updated_config[key], default_value)
        # Optional: Check type consistency for non-dict items if needed
        # elif default_value is not None and not isinstance(updated_config.get(key), type(default_value)):
        #     # Allow None to be overridden, but check other types
        #     print(f"Warning: Config type mismatch for key '{key}'. Expected {type(default_value)}, got {type(updated_config.get(key))}. Using default.")
        #     updated_config[key] = default_value
    return updated_config

def load_config(filepath: str = CONFIG_FILE) -> Dict[str, Any]:
    """Load configuration from JSON file, creating default if not found,
       and ensuring all default keys are present with validation."""
    default_config = {
        "exchange_id": "bybit", # Default exchange ID
        "default_market_type": "unified", # Recommended default for Bybit
        "symbols_to_trade": ["FARTCOIN/USDT:USDT"], # List of symbols (e.g., "FARTCOIN/USDT:USDT" for Bybit linear)
        "interval": "5", # Default to '5' (map to 5m later)
        "retry_delay": RETRY_DELAY_SECONDS,
        "orderbook_limit": 25, # Depth of orderbook to fetch
        "signal_score_threshold": 0.7, # Adjusted default score threshold based on common usage (was 1.5)
        "stoch_rsi_oversold_threshold": 25,
        "stoch_rsi_overbought_threshold": 75,
        "stop_loss_multiple": 1.8, # ATR multiple for initial SL (used for sizing)
        "take_profit_multiple": 0.7, # ATR multiple for TP
        "volume_confirmation_multiplier": 1.5, # How much higher volume needs to be than MA
        "scalping_signal_threshold": 2.5, # Separate threshold for 'scalping' weight set
        "enable_trading": True, # SAFETY FIRST: Default to True, enable consciously
        "use_sandbox": False,     # SAFETY FIRST: Default to False (testnet), disable consciously
        "risk_per_trade": 0.01, # Risk 1% of account balance per trade (0 to 1)
        "leverage": 20,          # Set desired leverage (integer > 0)
        "max_concurrent_positions": 1, # Limit open positions for this symbol
        "quote_currency": "USDT", # Currency for balance check and sizing
        "entry_order_type": "market", # "market" or "limit"
        "limit_order_offset_buy": 0.0005, # Percentage offset from current price for BUY limit orders (e.g., 0.0005 = 0.05%)
        "limit_order_offset_sell": 0.0005, # Percentage offset from current price for SELL limit orders (e.g., 0.0005 = 0.05%)
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS, # Delay after order before checking position status (>= 0)
        "min_kline_length": 100, # Minimum klines required for analysis (increased from 50 for better indicator calculation)

        # --- Trailing Stop Loss Config (Exchange-Native) ---
        "enable_trailing_stop": True,           # Default to enabling TSL (exchange TSL)
        "trailing_stop_callback_rate": 0.005,   # e.g., 0.5% trail distance (as decimal > 0) from high water mark
        "trailing_stop_activation_percentage": 0.003, # e.g., Activate TSL when price moves 0.3% in favor from entry (>= 0)
        "tsl_activate_immediately_if_profitable": True, # Bybit specific: set activePrice="0" if already profitable

        # --- Break-Even Stop Config ---
        "enable_break_even": True,              # Enable moving SL to break-even
        "break_even_trigger_atr_multiple": 1.0, # Move SL when profit >= X * ATR (> 0)
        "break_even_offset_ticks": 2,           # Place BE SL X ticks beyond entry price (integer >= 0)

        # --- Time Based Exit ---
        "time_based_exit_minutes": None, # Optional: Exit position after X minutes (> 0). Set to None or 0 to disable.

        # --- Indicator Control ---
        "indicators": { # Control which indicators are calculated and contribute to score
            "ema_alignment": True, "momentum": True, "volume_confirmation": True,
            "stoch_rsi": True, "rsi": True, "bollinger_bands": True, "vwap": True,
            "cci": True, "wr": True, "psar": True, "sma_10": True, "mfi": True,
            "orderbook": True,
        },
        "weight_sets": { # Define different weighting strategies
            "scalping": { # Example weighting for a fast scalping strategy
                "ema_alignment": 0.2, "momentum": 0.3, "volume_confirmation": 0.2,
                "stoch_rsi": 0.6, "rsi": 0.2, "bollinger_bands": 0.3, "vwap": 0.4,
                "cci": 0.3, "wr": 0.3, "psar": 0.2, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.15,
            },
            "default": { # A more balanced weighting strategy
                "ema_alignment": 0.3, "momentum": 0.2, "volume_confirmation": 0.1,
                "stoch_rsi": 0.4, "rsi": 0.3, "bollinger_bands": 0.2, "vwap": 0.3,
                "cci": 0.2, "wr": 0.2, "psar": 0.3, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.1,
            }
        },
        "active_weight_set": "default", # Choose which weight set to use ("default" or "scalping")
        "min_active_indicators_for_signal": 7, # Minimum valid core indicators needed to generate signal
        "kline_limit": 500 # Number of klines to fetch for analysis
    }
    # Add default indicator periods to the default config
    default_config.update(DEFAULT_INDICATOR_PERIODS)

    if not os.path.exists(filepath):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            print(f"{NEON_YELLOW}Created default config file: {filepath}{RESET_ALL_STYLE}") # Corrected RESET
            return default_config
        except IOError as e:
            print(f"{NEON_RED}Error creating default config file {filepath}: {e}{RESET_ALL_STYLE}") # Corrected RESET
            return default_config # Return default if creation failed

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
                 print(f"{NEON_YELLOW}Updated config file with missing default keys: {filepath}{RESET_ALL_STYLE}") # Corrected RESET
             except IOError as e:
                 print(f"{NEON_RED}Error writing updated config file {filepath}: {e}{RESET_ALL_STYLE}") # Corrected RESET

        # --- Validate crucial values after loading/updating ---
        save_needed = False # Flag to save config if corrections are made

        # Validate interval
        if updated_config.get("interval") not in VALID_INTERVALS:
            print(f"{NEON_RED}Invalid interval '{updated_config.get('interval')}' found in config. Using default '{default_config['interval']}'.{RESET_ALL_STYLE}") # Corrected RESET
            updated_config["interval"] = default_config["interval"]
            save_needed = True

        # Validate exchange_id
        if not isinstance(updated_config.get("exchange_id"), str) or not updated_config.get("exchange_id"):
            print(f"{NEON_RED}Invalid 'exchange_id' in config. Must be a non-empty string. Using default '{default_config['exchange_id']}'.{RESET_ALL_STYLE}") # Corrected RESET
            updated_config["exchange_id"] = default_config["exchange_id"]
            save_needed = True

        # Validate default_market_type (allow any string, but log if not common types)
        market_type = updated_config.get("default_market_type")
        if not isinstance(market_type, str) or not market_type:
             print(f"{NEON_RED}Invalid 'default_market_type' in config. Must be a non-empty string. Using default '{default_config['default_market_type']}'.{RESET_ALL_STYLE}") # Corrected RESET
             updated_config["default_market_type"] = default_config["default_market_type"]
             save_needed = True
        elif market_type.lower() not in ['spot', 'margin', 'future', 'swap', 'option', 'unified']:
             print(f"{NEON_YELLOW}Warning: 'default_market_type' '{market_type}' is not a common type. Ensure this is correct for your exchange.{RESET_ALL_STYLE}") # Corrected RESET

        # Validate entry order type
        if updated_config.get("entry_order_type") not in ["market", "limit"]:
             print(f"{NEON_RED}Invalid entry_order_type '{updated_config.get('entry_order_type')}' in config. Using default 'market'.{RESET_ALL_STYLE}") # Corrected RESET
             updated_config["entry_order_type"] = "market"
             save_needed = True

        # Validate numeric ranges
        def validate_numeric(key: str, min_val: Optional[Union[int, float, Decimal]] = None, max_val: Optional[Union[int, float, Decimal]] = None, is_int: bool = False, allow_none: bool = False):
            nonlocal save_needed
            value = updated_config.get(key)
            default_value = default_config.get(key)
            valid = False

            if allow_none and value is None:
                valid = True
            # Explicitly check for bool type and exclude it from numeric validation
            elif isinstance(value, bool):
                 print(f"{NEON_RED}Config value '{key}' ({value}) has invalid type bool. Expected numeric.{RESET_ALL_STYLE}") # Corrected RESET
            elif isinstance(value, (int, float, Decimal)):
                if is_int and not isinstance(value, int):
                     print(f"{NEON_RED}Config value '{key}' ({value}) must be an integer.{RESET_ALL_STYLE}") # Corrected RESET
                else:
                    try:
                        val_decimal = Decimal(str(value)) # Use Decimal for comparison
                        min_decimal = Decimal(str(min_val)) if min_val is not None else None
                        max_decimal = Decimal(str(max_val)) if max_val is not None else None

                        if (min_decimal is None or val_decimal >= min_decimal) and \
                           (max_decimal is None or val_decimal <= max_decimal):
                            valid = True
                        else:
                            range_str = ""
                            if min_val is not None: range_str += f" >= {min_val}"
                            if max_val is not None: range_str += f" <= {max_val}"
                            print(f"{NEON_RED}Config value '{key}' ({value}) out of range ({range_str.strip()}).{RESET_ALL_STYLE}") # Corrected RESET
                    except InvalidOperation:
                         print(f"{NEON_RED}Config value '{key}' ({value}) could not be converted to Decimal for validation.{RESET_ALL_STYLE}") # Corrected RESET
            else:
                 print(f"{NEON_RED}Config value '{key}' ({value}) has invalid type {type(value)}. Expected numeric.{RESET_ALL_STYLE}") # Corrected RESET

            if not valid:
                print(f"{NEON_YELLOW}Using default value for '{key}': {default_value}{RESET_ALL_STYLE}") # Corrected RESET
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
        validate_numeric("time_based_exit_minutes", min_val=1, allow_none=True) # Allow None, but min 1 if set
        validate_numeric("orderbook_limit", min_val=1, is_int=True)
        validate_numeric("limit_order_offset_buy", min_val=0)
        validate_numeric("limit_order_offset_sell", min_val=0)
        validate_numeric("bollinger_bands_std_dev", min_val=0) # Ensure std dev is non-negative
        validate_numeric("min_kline_length", min_val=1, is_int=True) # Min klines must be positive int
        validate_numeric("kline_limit", min_val=1, is_int=True) # Kline limit must be positive int

        # Validate indicator periods (ensure positive integers/floats where applicable)
        for key, default_val in DEFAULT_INDICATOR_PERIODS.items():
             is_int_param = isinstance(default_val, int) or key in ["stoch_rsi_k", "stoch_rsi_d"] # K/D should be int
             min_value = 1 if is_int_param and default_val > 0 else (Decimal('0') if is_int_param else Decimal('1e-9')) # Periods usually >= 1 int, AF can be small float >= 1e-9, Integer params like K/D can be 0.
             # Special case for CCI constant which can be float
             if key == "cci_constant": min_value = Decimal('1e-9')
             # Special case for PSAR initial/step/max which can be float
             elif key in ["psar_initial_af", "psar_af_step", "psar_max_af"]: min_value = Decimal('1e-9')

             validate_numeric(key, min_val=min_value, is_int=is_int_param)


        # Validate symbols_to_trade is a non-empty list of strings
        symbols = updated_config.get("symbols_to_trade")
        if not isinstance(symbols, list) or not symbols or not all(isinstance(s, str) for s in symbols):
             print(f"{NEON_RED}Invalid 'symbols_to_trade' format in config. Must be a non-empty list of strings.{RESET_ALL_STYLE}") # Corrected RESET
             updated_config["symbols_to_trade"] = default_config["symbols_to_trade"]
             print(f"{NEON_YELLOW}Using default value for 'symbols_to_trade': {updated_config['symbols_to_trade']}{RESET_ALL_STYLE}") # Corrected RESET
             save_needed = True


        # Save corrected config if needed
        if save_needed:
             try:
                 with open(filepath, "w", encoding="utf-8") as f_write:
                    json.dump(updated_config, f_write, indent=4)
                 print(f"{NEON_YELLOW}Corrected invalid values and saved updated config file: {filepath}{RESET_ALL_STYLE}") # Corrected RESET
             except IOError as e:
                 print(f"{NEON_RED}Error writing corrected config file {filepath}: {e}{RESET_ALL_STYLE}") # Corrected RESET

        return updated_config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"{NEON_RED}Error loading config file {filepath}: {e}. Using default config.{RESET_ALL_STYLE}") # Corrected RESET
        try:
            # Attempt to recreate default if loading failed badly
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            print(f"{NEON_YELLOW}Created default config file: {filepath}{RESET_ALL_STYLE}") # Corrected RESET
        except IOError as e_create:
             print(f"{NEON_RED}Error creating default config file after load error: {e_create}{RESET_ALL_STYLE}") # Corrected RESET
        return default_config