# File: app_config.py
import json
import logging
from pathlib import Path
from decimal import Decimal, getcontext, ROUND_HALF_UP, InvalidOperation, ROUND_DOWN, ROUND_UP, ROUND_CEILING
from typing import Dict, Any, Optional, Union

# --- Configuration Constants ---
CONFIG_FILE = Path("config.json")
STATE_FILE = Path("bot_state.json")
LOG_FILE = Path("trading_bot.log")

# --- Decimal Precision Configuration ---
# Affects display; calculations use higher precision temporarily.
DECIMAL_DISPLAY_PRECISION = 8
# High precision is crucial for intermediate steps.
CALCULATION_PRECISION = 28

# Set Decimal context globally. This will be effective when this module is imported.
# It can be re-affirmed in main.py for absolute certainty before other imports.
try:
    getcontext().prec = CALCULATION_PRECISION
    # Default rounding, can be overridden for specific cases
    getcontext().rounding = ROUND_HALF_UP
except Exception as e:
    # Use basic print as logger might not be ready if this fails early
    print(f"CRITICAL: Failed to set Decimal context in app_config.py: {e}", file=sys.stderr)
    # Exiting here might be too early if main.py can also set it.
    # For now, just print the error.


# --- Helper Functions for JSON (De)Serialization with Decimals ---

def decimal_serializer(obj: Any) -> Union[str, Any]:
    """JSON serializer for Decimal objects, handling special values."""
    if isinstance(obj, Decimal):
        if obj.is_nan():
            return 'NaN'
        if obj.is_infinite():
            return 'Infinity' if obj > 0 else '-Infinity'
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable by this function")

def decimal_decoder(dct: Dict[str, Any]) -> Dict[str, Any]:
    """JSON decoder hook to convert numeric-like strings back to Decimal."""
    new_dct = {}
    for key, value in dct.items():
        if isinstance(value, str):
            if value == 'NaN':
                 new_dct[key] = Decimal('NaN')
            elif value == 'Infinity':
                 new_dct[key] = Decimal('Infinity')
            elif value == '-Infinity':
                 new_dct[key] = Decimal('-Infinity')
            else:
                try:
                    new_dct[key] = Decimal(value)
                except InvalidOperation:
                    new_dct[key] = value
        elif isinstance(value, list):
            new_list = []
            for item in value:
                if isinstance(item, dict):
                    new_list.append(decimal_decoder(item))
                elif isinstance(item, str):
                    if item == 'NaN': new_list.append(Decimal('NaN'))
                    elif item == 'Infinity': new_list.append(Decimal('Infinity'))
                    elif item == '-Infinity': new_list.append(Decimal('-Infinity'))
                    else:
                        try: new_list.append(Decimal(item))
                        except InvalidOperation: new_list.append(item)
                else:
                    new_list.append(item)
            new_dct[key] = new_list
        elif isinstance(value, dict):
            new_dct[key] = decimal_decoder(value)
        else:
            new_dct[key] = value
    return new_dct

# --- Configuration and State Loading/Saving Functions ---

def load_config(config_path: Path, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """Loads configuration from a JSON file with Decimal conversion and validation."""
    if not config_path.is_file():
        logger.error(f"Configuration file not found: {config_path}")
        return None
    try:
        with open(config_path, 'r') as f:
            config = json.load(f, object_hook=decimal_decoder)
        logger.info(f"Configuration loaded successfully from {config_path}")
        if not validate_config(config, logger):
             logger.error("Configuration validation failed.")
             return None
        return config
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON configuration file {config_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}", exc_info=True)
        return None

def validate_config(config: Dict[str, Any], logger: logging.Logger) -> bool:
    """Validates the loaded configuration."""
    is_valid = True
    required_sections = [
        "exchange", "api_credentials", "trading_settings", "indicator_settings",
        "risk_management", "logging"
    ]
    for section in required_sections:
        if section not in config:
            logger.error(f"Missing required configuration section: '{section}'")
            is_valid = False
    if not is_valid: return False

    exchange_cfg = config.get("exchange", {})
    if not isinstance(exchange_cfg.get("id"), str) or not exchange_cfg["id"]:
        logger.error("Config validation failed: 'exchange.id' must be a non-empty string.")
        is_valid = False
    for key in ['max_retries', 'retry_delay_seconds']:
        val = exchange_cfg.get(key)
        if val is not None and not isinstance(val, int) or (isinstance(val, int) and val < 0):
             logger.error(f"Config validation failed: 'exchange.{key}' must be a non-negative integer if provided.")
             is_valid = False

    api_creds = config.get("api_credentials", {})
    if not isinstance(api_creds.get("api_key"), str) or not api_creds.get("api_key"):
        logger.error("Config validation failed: 'api_credentials.api_key' must be a non-empty string.")
        is_valid = False
    if not isinstance(api_creds.get("api_secret"), str) or not api_creds.get("api_secret"):
        logger.error("Config validation failed: 'api_credentials.api_secret' must be a non-empty string.")
        is_valid = False

    settings = config.get("trading_settings", {})
    required_trading_keys = ["symbol", "timeframe", "leverage", "quote_asset", "category"]
    for key in required_trading_keys:
        if key not in settings:
            logger.error(f"Config validation failed: Missing required key '{key}' in 'trading_settings'.")
            is_valid = False
    if not is_valid: return False

    if not isinstance(settings.get("symbol"), str) or not settings.get("symbol"):
        logger.error("Config validation failed: 'trading_settings.symbol' must be a non-empty string.")
        is_valid = False
    if not isinstance(settings.get("timeframe"), str) or not settings.get("timeframe"):
        logger.error("Config validation failed: 'trading_settings.timeframe' must be a non-empty string.")
        is_valid = False
    leverage = settings.get("leverage")
    if not isinstance(leverage, Decimal) or not leverage.is_finite() or leverage <= 0:
        logger.error("Config validation failed: 'trading_settings.leverage' must be a positive finite number (loaded as Decimal).")
        is_valid = False
    if not isinstance(settings.get("quote_asset"), str) or not settings.get("quote_asset"):
        logger.error("Config validation failed: 'trading_settings.quote_asset' must be a non-empty string.")
        is_valid = False
    category = settings.get("category")
    if category not in ['linear', 'inverse', 'spot']:
        logger.error(f"Config validation failed: 'trading_settings.category' must be 'linear', 'inverse', or 'spot'. Found: {category}")
        is_valid = False
    if "poll_interval_seconds" in settings and (not isinstance(settings["poll_interval_seconds"], int) or settings["poll_interval_seconds"] <= 0):
         logger.error("Config validation failed: 'trading_settings.poll_interval_seconds' must be a positive integer.")
         is_valid = False
    if "hedge_mode" in settings and not isinstance(settings["hedge_mode"], bool):
         logger.error("Config validation failed: 'trading_settings.hedge_mode' must be a boolean (true/false).")
         is_valid = False

    indicators = config.get("indicator_settings", {})
    for key in ["rsi_period", "macd_fast", "macd_slow", "macd_signal", "ema_short_period", "ema_long_period", "atr_period", "ohlcv_fetch_limit"]:
         val = indicators.get(key)
         if val is not None and (not isinstance(val, int) or val <= 0):
              logger.error(f"Config validation failed: 'indicator_settings.{key}' must be a positive integer if provided.")
              is_valid = False
    ema_short = indicators.get("ema_short_period")
    ema_long = indicators.get("ema_long_period")
    if isinstance(ema_short, int) and isinstance(ema_long, int) and ema_short >= ema_long:
        logger.error("Config validation failed: 'ema_short_period' must be less than 'ema_long_period'.")
        is_valid = False
    for key in ["rsi_overbought", "rsi_oversold", "macd_hist_threshold", "strong_buy_threshold", "buy_threshold", "sell_threshold", "strong_sell_threshold"]:
         val = indicators.get(key)
         if val is not None and (not isinstance(val, Decimal) or not val.is_finite()):
              logger.error(f"Config validation failed: 'indicator_settings.{key}' must be a finite number (loaded as Decimal) if provided.")
              is_valid = False
    weights = indicators.get("signal_weights")
    if weights is not None:
         if not isinstance(weights, dict):
              logger.error("Config validation failed: 'indicator_settings.signal_weights' must be a dictionary.")
              is_valid = False
         else:
              for key_w, val_w in weights.items():
                   if not isinstance(val_w, Decimal) or not val_w.is_finite() or val_w < 0:
                        logger.error(f"Config validation failed: Signal weight '{key_w}' must be a non-negative finite number (loaded as Decimal).")
                        is_valid = False

    risk = config.get("risk_management", {})
    risk_percent = risk.get("risk_per_trade_percent")
    if not isinstance(risk_percent, Decimal) or not risk_percent.is_finite() or not (Decimal(0) < risk_percent <= Decimal(100)):
         logger.error("Config validation failed: 'risk_per_trade_percent' must be a finite number between 0 (exclusive) and 100 (inclusive).")
         is_valid = False
    sl_method = risk.get("stop_loss_method")
    if sl_method not in [None, "atr", "fixed_percent"]:
         logger.error("Config validation failed: 'risk_management.stop_loss_method' must be 'atr' or 'fixed_percent' if provided.")
         is_valid = False
    if sl_method == "atr":
         atr_mult = risk.get("atr_multiplier")
         if not isinstance(atr_mult, Decimal) or not atr_mult.is_finite() or atr_mult <= 0:
              logger.error("Config validation failed: 'atr_multiplier' must be a positive finite Decimal for ATR stop loss.")
              is_valid = False
    if sl_method == "fixed_percent":
         fixed_perc = risk.get("fixed_stop_loss_percent")
         if not isinstance(fixed_perc, Decimal) or not fixed_perc.is_finite() or not (Decimal(0) < fixed_perc < Decimal(100)):
              logger.error("Config validation failed: 'fixed_stop_loss_percent' must be a finite Decimal between 0 and 100 (exclusive).")
              is_valid = False
    if risk.get("use_break_even_sl", False):
        for key in ["break_even_trigger_atr", "break_even_offset_atr"]:
            val = risk.get(key)
            if not isinstance(val, Decimal) or not val.is_finite() or val < 0:
                logger.error(f"Config validation failed: '{key}' must be a non-negative finite Decimal for Break-Even SL.")
                is_valid = False
    if risk.get("use_trailing_sl", False):
        val = risk.get("trailing_sl_atr_multiplier")
        if not isinstance(val, Decimal) or not val.is_finite() or val <= 0:
            logger.error("Config validation failed: 'trailing_sl_atr_multiplier' must be a positive finite Decimal for Trailing SL.")
            is_valid = False

    log_cfg = config.get("logging", {})
    log_level_str = log_cfg.get("level", "INFO").upper()
    if log_level_str not in logging._nameToLevel:
        logger.error(f"Config validation failed: Invalid log level '{log_level_str}' in 'logging' section.")
        is_valid = False

    if is_valid:
        logger.info("Configuration validation successful.")
    else:
        logger.error("Configuration validation failed. Please review the errors above.")
    return is_valid

def load_state(state_path: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Loads the bot's state from a JSON file with Decimal conversion."""
    if not state_path.is_file():
        logger.warning(f"State file not found at {state_path}. Starting with empty state.")
        return {}
    try:
        with open(state_path, 'r') as f:
            state = json.load(f, object_hook=decimal_decoder)
        logger.info(f"Bot state loaded successfully from {state_path}")
        return state
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON state file {state_path}: {e}. Using empty state.", exc_info=True)
        return {}
    except Exception as e:
        logger.error(f"Error loading state from {state_path}: {e}. Using empty state.", exc_info=True)
        return {}

def save_state(state: Dict[str, Any], state_path: Path, logger: logging.Logger) -> None:
    """Saves the bot's state to a JSON file with Decimal serialization."""
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=4, default=decimal_serializer)
        logger.debug(f"Bot state saved successfully to {state_path}")
    except TypeError as e:
        logger.error(f"Error serializing state for saving (check for non-Decimal/non-standard types): {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error saving state to {state_path}: {e}", exc_info=True)

```

```python
