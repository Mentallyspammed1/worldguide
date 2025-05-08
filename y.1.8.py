# Enhanced Python Trading Bot Code
# Incorporates improvements based on the provided enhancement plan.

import ccxt
import pandas as pd
import pandas_ta as ta
import json
import logging
import logging.handlers
import time
import sys
import math  # Added for math.isfinite, math.ceil
from decimal import Decimal, getcontext, ROUND_HALF_UP, InvalidOperation, ROUND_DOWN, ROUND_UP, ROUND_CEILING
from typing import Dict, Any, Optional, Tuple, List, Union
from enum import Enum
from pathlib import Path

# --- Configuration ---

# Set Decimal precision (adjust as needed for your asset's precision)
# This affects display; calculations use higher precision temporarily.
DECIMAL_DISPLAY_PRECISION = 8
# Add buffer for intermediate calculations; final results should be rounded appropriately.
# High precision is crucial for intermediate steps.
CALCULATION_PRECISION = 28
getcontext().prec = CALCULATION_PRECISION
# Default rounding, can be overridden for specific cases like conservative SL/TP rounding.
getcontext().rounding = ROUND_HALF_UP

CONFIG_FILE = Path("config.json")
STATE_FILE = Path("bot_state.json")
LOG_FILE = Path("trading_bot.log")
LOG_LEVEL = logging.INFO  # Default log level, can be overridden by config

# --- Enums ---


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"
    NONE = "none"  # Represents no position or Bybit's 'None' side in hedge mode


class Signal(Enum):
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2

# --- Helper Functions ---


def setup_logger() -> logging.Logger:
    """Sets up the application logger with console and file handlers."""
    log_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    logger = logging.getLogger("TradingBot")
    # Set initial level, may be updated by config later
    logger.setLevel(LOG_LEVEL)

    # Prevent adding handlers multiple times if called again
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(log_formatter)
    logger.addHandler(stdout_handler)

    # File Handler (Rotating)
    try:
        log_dir = LOG_FILE.parent
        # Ensure log directory exists
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=5*1024*1024, backupCount=3  # 5 MB per file, 3 backups
        )
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # Log to console if file handler fails
        # Note: logger might not be fully configured yet if file handler fails early
        print(
            f"ERROR: Failed to set up file logging handler: {e}", file=sys.stderr)
        # Use basic logging temporarily if logger is not available
        logging.basicConfig(level=logging.ERROR)
        logging.error(
            f"Failed to set up file logging handler: {e}", exc_info=True)

    # Suppress noisy libraries if needed
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    # Adjust if more ccxt detail needed
    logging.getLogger("ccxt").setLevel(logging.WARNING)

    return logger


def decimal_serializer(obj: Any) -> Union[str, Any]:
    """JSON serializer for Decimal objects, handling special values."""
    if isinstance(obj, Decimal):
        if obj.is_nan():
            return 'NaN'
        if obj.is_infinite():
            return 'Infinity' if obj > 0 else '-Infinity'
        # Standard finite Decimal to string
        return str(obj)
    # Let the default JSON encoder handle other types or raise TypeError
    raise TypeError(
        f"Object of type {obj.__class__.__name__} is not JSON serializable by this function")


def decimal_decoder(dct: Dict[str, Any]) -> Dict[str, Any]:
    """JSON decoder hook to convert numeric-like strings back to Decimal."""
    new_dct = {}
    for key, value in dct.items():
        if isinstance(value, str):
            # Handle special values first
            if value == 'NaN':
                new_dct[key] = Decimal('NaN')
            elif value == 'Infinity':
                new_dct[key] = Decimal('Infinity')
            elif value == '-Infinity':
                new_dct[key] = Decimal('-Infinity')
            else:
                # Attempt direct Decimal conversion for potential numbers
                try:
                    new_dct[key] = Decimal(value)
                except InvalidOperation:
                    # Keep as string if it's not a valid Decimal representation
                    new_dct[key] = value
        elif isinstance(value, list):
            # Recursively decode lists: handle dicts, strings, and others
            new_list = []
            for item in value:
                if isinstance(item, dict):
                    new_list.append(decimal_decoder(item))
                elif isinstance(item, str):
                    if item == 'NaN':
                        new_list.append(Decimal('NaN'))
                    elif item == 'Infinity':
                        new_list.append(Decimal('Infinity'))
                    elif item == '-Infinity':
                        new_list.append(Decimal('-Infinity'))
                    else:
                        try:
                            new_list.append(Decimal(item))
                        except InvalidOperation:
                            new_list.append(item)  # Keep as string
                else:
                    new_list.append(item)  # Keep other types as is
            new_dct[key] = new_list
        elif isinstance(value, dict):
            # Recursively decode nested dictionaries
            new_dct[key] = decimal_decoder(value)
        else:
            # Keep other types as is
            new_dct[key] = value
    return new_dct


def load_config(config_path: Path, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """Loads configuration from a JSON file with Decimal conversion and validation."""
    if not config_path.is_file():
        logger.error(f"Configuration file not found: {config_path}")
        return None
    try:
        with open(config_path, 'r') as f:
            # Use object_hook for robust Decimal handling in nested structures
            config = json.load(f, object_hook=decimal_decoder)
        logger.info(f"Configuration loaded successfully from {config_path}")
        # --- Add specific config validation here ---
        if not validate_config(config, logger):
            logger.error("Configuration validation failed.")
            return None
        return config
    except json.JSONDecodeError as e:
        logger.error(
            f"Error decoding JSON configuration file {config_path}: {e}")
        return None
    except Exception as e:
        logger.error(
            f"Error loading configuration from {config_path}: {e}", exc_info=True)
        return None


def validate_config(config: Dict[str, Any], logger: logging.Logger) -> bool:
    """Validates the loaded configuration."""
    is_valid = True  # Assume valid initially
    required_sections = [
        "exchange", "api_credentials", "trading_settings", "indicator_settings",
        "risk_management", "logging"
    ]
    for section in required_sections:
        if section not in config:
            logger.error(
                f"Missing required configuration section: '{section}'")
            is_valid = False

    # If basic sections are missing, don't proceed with detailed checks
    if not is_valid:
        return False

    # --- Validate Exchange Section ---
    exchange_cfg = config.get("exchange", {})
    if not isinstance(exchange_cfg.get("id"), str) or not exchange_cfg["id"]:
        logger.error(
            "Config validation failed: 'exchange.id' must be a non-empty string.")
        is_valid = False
    # Validate optional numeric settings
    for key in ['max_retries', 'retry_delay_seconds']:
        val = exchange_cfg.get(key)
        if val is not None and not isinstance(val, int) or (isinstance(val, int) and val < 0):
            logger.error(
                f"Config validation failed: 'exchange.{key}' must be a non-negative integer if provided.")
            is_valid = False

    # --- Validate API Credentials ---
    api_creds = config.get("api_credentials", {})
    if not isinstance(api_creds.get("api_key"), str) or not api_creds.get("api_key"):
        logger.error(
            "Config validation failed: 'api_credentials.api_key' must be a non-empty string.")
        is_valid = False
    if not isinstance(api_creds.get("api_secret"), str) or not api_creds.get("api_secret"):
        logger.error(
            "Config validation failed: 'api_credentials.api_secret' must be a non-empty string.")
        is_valid = False

    # --- Validate Trading Settings ---
    settings = config.get("trading_settings", {})
    required_trading_keys = ["symbol", "timeframe",
                             "leverage", "quote_asset", "category"]
    for key in required_trading_keys:
        if key not in settings:
            logger.error(
                f"Config validation failed: Missing required key '{key}' in 'trading_settings'.")
            is_valid = False
    if not is_valid:
        return False  # Stop if basic trading settings are missing

    if not isinstance(settings.get("symbol"), str) or not settings.get("symbol"):
        logger.error(
            "Config validation failed: 'trading_settings.symbol' must be a non-empty string.")
        is_valid = False
    if not isinstance(settings.get("timeframe"), str) or not settings.get("timeframe"):
        logger.error(
            "Config validation failed: 'trading_settings.timeframe' must be a non-empty string.")
        is_valid = False
    leverage = settings.get("leverage")
    if not isinstance(leverage, Decimal) or not leverage.is_finite() or leverage <= 0:
        logger.error(
            "Config validation failed: 'trading_settings.leverage' must be a positive finite number (loaded as Decimal).")
        is_valid = False
    if not isinstance(settings.get("quote_asset"), str) or not settings.get("quote_asset"):
        logger.error(
            "Config validation failed: 'trading_settings.quote_asset' must be a non-empty string.")
        is_valid = False
    category = settings.get("category")
    if category not in ['linear', 'inverse', 'spot']:
        logger.error(
            f"Config validation failed: 'trading_settings.category' must be 'linear', 'inverse', or 'spot'. Found: {category}")
        is_valid = False
    if "poll_interval_seconds" in settings and (not isinstance(settings["poll_interval_seconds"], int) or settings["poll_interval_seconds"] <= 0):
        logger.error(
            "Config validation failed: 'trading_settings.poll_interval_seconds' must be a positive integer.")
        is_valid = False
    if "hedge_mode" in settings and not isinstance(settings["hedge_mode"], bool):
        logger.error(
            "Config validation failed: 'trading_settings.hedge_mode' must be a boolean (true/false).")
        is_valid = False

    # --- Validate Indicator Settings ---
    indicators = config.get("indicator_settings", {})
    # Check periods are positive integers if present
    for key in ["rsi_period", "macd_fast", "macd_slow", "macd_signal", "ema_short_period", "ema_long_period", "atr_period", "ohlcv_fetch_limit"]:
        val = indicators.get(key)
        if val is not None and (not isinstance(val, int) or val <= 0):
            logger.error(
                f"Config validation failed: 'indicator_settings.{key}' must be a positive integer if provided.")
            is_valid = False
    # Check EMA period relationship
    ema_short = indicators.get("ema_short_period")
    ema_long = indicators.get("ema_long_period")
    if isinstance(ema_short, int) and isinstance(ema_long, int) and ema_short >= ema_long:
        logger.error(
            "Config validation failed: 'ema_short_period' must be less than 'ema_long_period'.")
        is_valid = False
    # Check thresholds are Decimals if present
    for key in ["rsi_overbought", "rsi_oversold", "macd_hist_threshold", "strong_buy_threshold", "buy_threshold", "sell_threshold", "strong_sell_threshold"]:
        val = indicators.get(key)
        if val is not None and (not isinstance(val, Decimal) or not val.is_finite()):
            logger.error(
                f"Config validation failed: 'indicator_settings.{key}' must be a finite number (loaded as Decimal) if provided.")
            is_valid = False
    # Validate signal weights structure and values
    weights = indicators.get("signal_weights")
    if weights is not None:
        if not isinstance(weights, dict):
            logger.error(
                "Config validation failed: 'indicator_settings.signal_weights' must be a dictionary.")
            is_valid = False
        else:
            for key, val in weights.items():
                if not isinstance(val, Decimal) or not val.is_finite() or val < 0:
                    logger.error(
                        f"Config validation failed: Signal weight '{key}' must be a non-negative finite number (loaded as Decimal).")
                    is_valid = False

    # --- Validate Risk Management ---
    risk = config.get("risk_management", {})
    risk_percent = risk.get("risk_per_trade_percent")
    if not isinstance(risk_percent, Decimal) or not risk_percent.is_finite() or not (Decimal(0) < risk_percent <= Decimal(100)):
        logger.error(
            "Config validation failed: 'risk_per_trade_percent' must be a finite number between 0 (exclusive) and 100 (inclusive).")
        is_valid = False
    sl_method = risk.get("stop_loss_method")
    if sl_method not in [None, "atr", "fixed_percent"]:
        logger.error(
            "Config validation failed: 'risk_management.stop_loss_method' must be 'atr' or 'fixed_percent' if provided.")
        is_valid = False
    if sl_method == "atr":
        atr_mult = risk.get("atr_multiplier")
        if not isinstance(atr_mult, Decimal) or not atr_mult.is_finite() or atr_mult <= 0:
            logger.error(
                "Config validation failed: 'atr_multiplier' must be a positive finite Decimal for ATR stop loss.")
            is_valid = False
    if sl_method == "fixed_percent":
        fixed_perc = risk.get("fixed_stop_loss_percent")
        if not isinstance(fixed_perc, Decimal) or not fixed_perc.is_finite() or not (Decimal(0) < fixed_perc < Decimal(100)):
            logger.error(
                "Config validation failed: 'fixed_stop_loss_percent' must be a finite Decimal between 0 and 100 (exclusive).")
            is_valid = False
    # Validate BE/TSL parameters if features are enabled
    if risk.get("use_break_even_sl", False):
        for key in ["break_even_trigger_atr", "break_even_offset_atr"]:
            val = risk.get(key)
            if not isinstance(val, Decimal) or not val.is_finite() or val < 0:
                logger.error(
                    f"Config validation failed: '{key}' must be a non-negative finite Decimal for Break-Even SL.")
                is_valid = False
    if risk.get("use_trailing_sl", False):
        val = risk.get("trailing_sl_atr_multiplier")
        if not isinstance(val, Decimal) or not val.is_finite() or val <= 0:
            logger.error(
                "Config validation failed: 'trailing_sl_atr_multiplier' must be a positive finite Decimal for Trailing SL.")
            is_valid = False

    # --- Validate Logging ---
    log_cfg = config.get("logging", {})
    log_level_str = log_cfg.get("level", "INFO").upper()
    if log_level_str not in logging._nameToLevel:
        logger.error(
            f"Config validation failed: Invalid log level '{log_level_str}' in 'logging' section.")
        is_valid = False

    if is_valid:
        logger.info("Configuration validation successful.")
    else:
        logger.error(
            "Configuration validation failed. Please review the errors above.")
    return is_valid


def load_state(state_path: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Loads the bot's state from a JSON file with Decimal conversion."""
    if not state_path.is_file():
        logger.warning(
            f"State file not found at {state_path}. Starting with empty state.")
        return {}  # Return empty dict if no state file
    try:
        with open(state_path, 'r') as f:
            # Use object_hook for robust Decimal handling in nested structures
            state = json.load(f, object_hook=decimal_decoder)
        logger.info(f"Bot state loaded successfully from {state_path}")
        # --- Add optional state validation here ---
        # e.g., check if 'active_position' has required keys if not None
        return state
    except json.JSONDecodeError as e:
        logger.error(
            f"Error decoding JSON state file {state_path}: {e}. Using empty state.", exc_info=True)
        return {}
    except Exception as e:
        logger.error(
            f"Error loading state from {state_path}: {e}. Using empty state.", exc_info=True)
        return {}


def save_state(state: Dict[str, Any], state_path: Path, logger: logging.Logger) -> None:
    """Saves the bot's state to a JSON file with Decimal serialization."""
    try:
        # Ensure directory exists
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=4, default=decimal_serializer)
        logger.debug(f"Bot state saved successfully to {state_path}")
    except TypeError as e:
        logger.error(
            f"Error serializing state for saving (check for non-Decimal/non-standard types): {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error saving state to {state_path}: {e}", exc_info=True)


# --- Exchange Interaction Wrapper ---

class BybitV5Wrapper:
    """
    Wraps CCXT exchange interactions, focusing on Bybit V5 specifics,
    error handling, Decimal usage, and rate limiting.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.logger = logger
        self.config = config
        self.exchange_id = config['exchange'].get('id', 'bybit')
        self.category = config['trading_settings'].get(
            'category', 'linear')  # linear, inverse, spot
        self.hedge_mode = config['trading_settings'].get(
            'hedge_mode', False)  # Bybit hedge mode
        self.max_retries = config['exchange'].get('max_retries', 3)
        self.retry_delay = config['exchange'].get('retry_delay_seconds', 5)

        if self.exchange_id != 'bybit':
            self.logger.warning(
                f"This wrapper is optimized for Bybit V5, but exchange ID is set to '{self.exchange_id}'. Some features might not work as expected.")

        try:
            exchange_class = getattr(ccxt, self.exchange_id)
        except AttributeError:
            self.logger.critical(
                f"CCXT exchange class not found for ID: '{self.exchange_id}'. Exiting.")
            raise  # Re-raise critical error

        ccxt_config = {
            'apiKey': config['api_credentials']['api_key'],
            'secret': config['api_credentials']['api_secret'],
            'enableRateLimit': True,  # Let CCXT handle basic rate limiting
            'options': {
                # Default type influences which market (spot/swap) is used if symbol is ambiguous
                'defaultType': 'swap' if self.category in ['linear', 'inverse'] else 'spot',
                'adjustForTimeDifference': True,  # CCXT handles time sync issues
                # Add broker ID if present in config
            },
            # Explicitly set sandbox mode if needed via config
        }
        if config['exchange'].get('broker_id'):
            ccxt_config['options']['broker_id'] = config['exchange']['broker_id']
        if config['exchange'].get('sandbox_mode', False):
            self.logger.warning("Sandbox mode enabled. Using testnet URLs.")
            # Newer CCXT versions might prefer this
            ccxt_config['sandboxMode'] = True

        self.exchange = exchange_class(ccxt_config)

        # Set sandbox URL if sandboxMode is enabled (redundant if sandboxMode=True works)
        if config['exchange'].get('sandbox_mode', False):
            # Check if set_sandbox_mode exists and call it if sandboxMode didn't handle it
            if hasattr(self.exchange, 'set_sandbox_mode') and not self.exchange.sandbox:
                try:
                    self.exchange.set_sandbox_mode(True)
                except Exception as e:
                    self.logger.error(f"Error calling set_sandbox_mode: {e}")

        # Load markets to get precision details, contract sizes, limits etc.
        self.load_markets_with_retry()

    def load_markets_with_retry(self, reload=False):
        """Loads markets with retry logic."""
        retries = 0
        while retries <= self.max_retries:
            try:
                self.exchange.load_markets(reload=reload)
                self.logger.info(
                    f"Markets loaded successfully for {self.exchange_id}.")
                return True
            except ccxt.AuthenticationError:
                self.logger.exception(
                    "Authentication failed loading markets. Check API keys.")
                raise  # Re-raise critical error
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.OnMaintenance) as e:
                self.logger.warning(
                    f"Network/Availability error loading markets: {e}. Retrying in {self.retry_delay}s... (Attempt {retries + 1}/{self.max_retries + 1})")
            except ccxt.ExchangeError as e:
                self.logger.exception(
                    f"Exchange error loading markets: {e}. Might retry.")
                # Decide if this specific ExchangeError is retryable
                if retries < self.max_retries:
                    self.logger.info(f"Retrying in {self.retry_delay}s...")
                else:
                    self.logger.error(
                        "Exchange error loading markets after retries.")
                    raise  # Re-raise after retries
            except Exception as e:
                self.logger.exception(f"Unexpected error loading markets: {e}")
                raise  # Re-raise critical error

            retries += 1
            if retries <= self.max_retries:
                time.sleep(self.retry_delay)

        self.logger.critical(
            f"Failed to load markets for {self.exchange_id} after {self.max_retries + 1} attempts.")
        # Raise specific error
        raise ccxt.ExchangeError(
            "Failed to load markets after multiple retries.")

    def get_market(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Gets market data for a symbol, ensuring it's loaded and has precision."""
        try:
            # Ensure markets are loaded if not already
            if not self.exchange.markets or symbol not in self.exchange.markets:
                self.logger.warning(
                    f"Market data for {symbol} not loaded or missing. Attempting to load/reload markets.")
                self.load_markets_with_retry(
                    reload=True)  # Use retry mechanism

            market = self.exchange.market(symbol)
            if not market:
                self.logger.error(
                    f"Market data for symbol '{symbol}' is null or empty after fetching.")
                return None

            # Ensure precision values are loaded correctly
            precision = market.get('precision', {})
            if not precision or 'amount' not in precision or 'price' not in precision:
                self.logger.warning(
                    f"Precision info incomplete for market {symbol}. Reloading markets again.")
                self.load_markets_with_retry(reload=True)
                market = self.exchange.market(symbol)  # Try fetching again
                precision = market.get('precision', {})
                if not precision or 'amount' not in precision or 'price' not in precision:
                    self.logger.error(
                        f"Failed to load complete precision info for {symbol} even after reload.")
                    return None  # Cannot proceed reliably without precision

            # Ensure contractSize is loaded and converted to Decimal for derivatives
            if market.get('contract') and 'contractSize' in market:
                try:
                    # CCXT might return float/int, convert safely
                    cs = market['contractSize']
                    market['contractSize'] = Decimal(
                        str(cs)) if cs is not None else Decimal('NaN')
                except (InvalidOperation, TypeError):
                    self.logger.error(
                        f"Failed to parse contractSize '{market['contractSize']}' as Decimal for {symbol}.")
                    return None  # Cannot proceed reliably without contract size for derivatives

            return market
        except ccxt.BadSymbol:
            self.logger.error(
                f"Symbol '{symbol}' not found on {self.exchange_id}.")
            return None
        except ccxt.ExchangeError as e:  # Catch errors during market loading/fetching within this method
            self.logger.error(
                f"Exchange error fetching market data for {symbol}: {e}", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(
                f"Unexpected error fetching market data for {symbol}: {e}", exc_info=True)
            return None

    def safe_ccxt_call(self, method_name: str, *args, **kwargs) -> Optional[Any]:
        """
        Safely executes a CCXT method with retries and enhanced error handling.
        Distinguishes between retryable and non-retryable errors.
        Injects Bybit V5 'category' parameter where needed.
        """
        retries = 0
        while retries <= self.max_retries:
            try:
                method = getattr(self.exchange, method_name)

                # --- Inject Bybit V5 'category' parameter ---
                # Clone params to avoid modifying the original kwargs dict directly if retrying
                call_params = kwargs.get('params', {}).copy()
                # Check if it's Bybit and the method likely requires 'category' for V5 Unified/Contract accounts
                if self.exchange_id == 'bybit' and self.category in ['linear', 'inverse'] and 'category' not in call_params:
                    # List of methods known or likely to require 'category' for V5 derivatives
                    methods_requiring_category = [
                        'create_order', 'edit_order', 'cancel_order', 'cancel_all_orders',
                        'fetch_order', 'fetch_open_orders', 'fetch_closed_orders', 'fetch_my_trades',
                        'fetch_position', 'fetch_positions',  # fetch_position singular might exist
                        'fetch_balance', 'set_leverage', 'set_margin_mode',
                        'fetch_leverage_tiers', 'fetch_funding_rate', 'fetch_funding_rates',
                        'fetch_funding_rate_history',
                        'private_post_position_trading_stop',  # Specific endpoint for SL/TP
                        'private_post_position_switch_margin_mode',  # Specific endpoint
                        'private_post_position_switch_mode',  # Specific endpoint
                        # Add others if discovered: e.g., transfer, withdraw?
                    ]
                    if method_name in methods_requiring_category or method_name.startswith('private_post_position'):
                        call_params['category'] = self.category

                # Update kwargs for the current call attempt with potentially modified params
                current_kwargs = kwargs.copy()
                current_kwargs['params'] = call_params

                self.logger.debug(
                    f"Calling CCXT method: {method_name}, Args: {args}, Kwargs: {current_kwargs}")
                result = method(*args, **current_kwargs)
                self.logger.debug(
                    f"CCXT call {method_name} successful. Result snippet: {str(result)[:200]}...")
                return result

            # --- Specific CCXT/Bybit Error Handling (Non-Retryable First) ---
            except ccxt.AuthenticationError as e:
                self.logger.error(
                    f"Authentication Error calling {method_name}: {e}. Check API keys. Non-retryable.")
                return None
            except ccxt.PermissionDenied as e:
                self.logger.error(
                    f"Permission Denied calling {method_name}: {e}. Check API key permissions (IP whitelist, endpoint access). Non-retryable.")
                return None
            except ccxt.AccountSuspended as e:
                self.logger.error(
                    f"Account Suspended calling {method_name}: {e}. Non-retryable.")
                return None
            except ccxt.InvalidOrder as e:  # Includes OrderNotFound, OrderNotFillable etc.
                self.logger.error(
                    f"Invalid Order parameters or state calling {method_name}: {e}. Check order details (size, price, type, status). Non-retryable.")
                # Log args/kwargs for debugging InvalidOrder
                self.logger.debug(
                    f"Failed Call Details - Method: {method_name}, Args: {args}, Kwargs: {kwargs}")
                return None
            except ccxt.InsufficientFunds as e:
                self.logger.error(
                    f"Insufficient Funds calling {method_name}: {e}. Non-retryable.")
                return None
            except ccxt.BadSymbol as e:
                self.logger.error(
                    f"Invalid Symbol calling {method_name}: {e}. Non-retryable.")
                return None
            except ccxt.BadRequest as e:
                # Often parameter errors, potentially Bybit specific codes in message
                self.logger.error(
                    f"Bad Request calling {method_name}: {e}. Check parameters. Assuming non-retryable.")
                self.logger.debug(
                    f"Failed Call Details - Method: {method_name}, Args: {args}, Kwargs: {kwargs}")
                # Example: Parse e.args[0] for specific Bybit error codes
                if self.exchange_id == 'bybit':
                    msg = str(e).lower()
                    # Bybit specific non-retryable error codes (examples)
                    # Param error, position idx error, set leverage error, qty error
                    non_retryable_codes = [
                        '110007', '110045', '110043', '110014']
                    if any(f"ret_code={code}" in msg or f"retcode={code}" in msg or f"'{code}'" in msg for code in non_retryable_codes):
                        self.logger.error(
                            f"Detected specific non-retryable Bybit error code in message: {msg}")
                        # Already logged above, just confirming non-retryable nature
                return None
            except ccxt.MarginModeAlreadySet as e:  # Example specific error
                self.logger.warning(
                    f"Margin mode already set as requested: {e}. Considered success for {method_name}.")
                return {}  # Return empty dict to indicate success/no action needed
            except ccxt.OperationFailed as e:  # Catch specific operational failures if needed
                self.logger.error(
                    f"Operation Failed calling {method_name}: {e}. May indicate position/margin issues. Assuming non-retryable.")
                return None

            # --- Retryable Errors ---
            except ccxt.RateLimitExceeded as e:
                self.logger.warning(
                    f"Rate Limit Exceeded calling {method_name}: {e}. Retrying in {self.retry_delay}s... (Attempt {retries + 1}/{self.max_retries + 1})")
            except ccxt.NetworkError as e:  # Includes ConnectionError, Timeout, etc.
                self.logger.warning(
                    f"Network Error calling {method_name}: {e}. Retrying in {self.retry_delay}s... (Attempt {retries + 1}/{self.max_retries + 1})")
            except ccxt.ExchangeNotAvailable as e:  # Maintenance or temporary outage
                self.logger.warning(
                    f"Exchange Not Available calling {method_name}: {e}. Retrying in {self.retry_delay}s... (Attempt {retries + 1}/{self.max_retries + 1})")
            except ccxt.OnMaintenance as e:  # Specific maintenance error
                self.logger.warning(
                    f"Exchange On Maintenance calling {method_name}: {e}. Retrying in {self.retry_delay}s... (Attempt {retries + 1}/{self.max_retries + 1})")
            except ccxt.ExchangeError as e:  # General exchange error, potentially retryable
                # Check for specific Bybit codes/messages that might be non-retryable
                msg = str(e).lower()
                # Example: Bybit position idx errors, certain margin errors might be non-retryable
                non_retryable_msgs = ['position idx not match', 'insufficient available balance',
                                      'risk limit', 'order cost not available', 'cannot be modified', 'order status is incorrect']
                # Bybit specific retryable error codes (examples from docs or experience)
                # Server busy, timeout, internal error (sometimes retryable)
                retryable_codes = ['10002', '10006', '10016']

                if any(term in msg for term in non_retryable_msgs) and not any(f"ret_code={code}" in msg or f"retcode={code}" in msg for code in retryable_codes):
                    self.logger.error(
                        f"Potentially non-retryable Exchange Error calling {method_name}: {e}.")
                    return None
                elif any(f"ret_code={code}" in msg or f"retcode={code}" in msg for code in retryable_codes):
                    self.logger.warning(
                        f"Retryable Exchange Error code detected calling {method_name}: {e}. Retrying...")
                    # Fall through to retry logic
                else:
                    # Assume retryable for generic ExchangeError if not identified as non-retryable
                    self.logger.warning(
                        f"Generic Exchange Error calling {method_name}: {e}. Retrying in {self.retry_delay}s... (Attempt {retries + 1}/{self.max_retries + 1})")

            # --- Unexpected Errors ---
            except Exception as e:
                self.logger.error(
                    f"Unexpected error during CCXT call '{method_name}': {e}", exc_info=True)
                # Decide if unexpected errors should be retried or immediately fail
                # Assume non-retryable for safety unless specifically known otherwise
                return None

            # --- Retry Logic ---
            retries += 1
            if retries <= self.max_retries:
                time.sleep(self.retry_delay)
            else:
                self.logger.error(
                    f"CCXT call '{method_name}' failed after {self.max_retries + 1} attempts.")
                return None

        # Should not be reached if loop condition is correct, but satisfy linters/type checkers
        return None

    # --- Specific API Call Wrappers ---

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetches OHLCV data and returns it as a Pandas DataFrame with Decimal types."""
        self.logger.info(
            f"Fetching {limit} OHLCV candles for {symbol} ({timeframe})...")
        ohlcv = self.safe_ccxt_call(
            'fetch_ohlcv', symbol, timeframe, limit=limit)
        if ohlcv is None or not ohlcv:
            # safe_ccxt_call already logged the error
            return None

        try:
            df = pd.DataFrame(
                ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            if df.empty:
                self.logger.warning(
                    f"Fetched OHLCV data for {symbol} resulted in an empty DataFrame.")
                return df  # Return empty df

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Convert OHLCV columns to Decimal, handling potential None or non-numeric values robustly
            with getcontext() as ctx:
                ctx.prec = CALCULATION_PRECISION  # Use high precision for conversion
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].apply(lambda x: Decimal(
                        str(x)) if x is not None else Decimal('NaN'))

            # Validate data sanity (e.g., check for NaNs, zero prices)
            nan_mask = df[['open', 'high', 'low', 'close']].isnull().any(
                axis=1) | df[['open', 'high', 'low', 'close']].applymap(lambda x: x.is_nan()).any(axis=1)
            if nan_mask.any():
                self.logger.warning(
                    f"{nan_mask.sum()} rows with NaN values found in fetched OHLCV data for {symbol}.")
                # Optionally drop rows with NaNs if they interfere with indicators:
                # df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

            zero_price_mask = (
                df[['open', 'high', 'low', 'close']] <= Decimal(0)).any(axis=1)
            if zero_price_mask.any():
                self.logger.warning(
                    f"{zero_price_mask.sum()} rows with zero or negative values found in fetched OHLCV prices for {symbol}.")

            self.logger.info(
                f"Successfully fetched and processed {len(df)} OHLCV candles for {symbol}.")
            return df
        except Exception as e:
            self.logger.error(
                f"Error processing OHLCV data into DataFrame: {e}", exc_info=True)
            return None

    def fetch_balance(self) -> Optional[Dict[str, Decimal]]:
        """Fetches account balance, returning total equity amounts as Decimals."""
        self.logger.debug("Fetching account balance...")
        # Bybit V5 requires category for fetch_balance (handled by safe_ccxt_call)
        balance_data = self.safe_ccxt_call('fetch_balance')

        if balance_data is None:
            return None

        balances = {}
        try:
            if self.exchange_id == 'bybit' and self.category in ['linear', 'inverse', 'spot']:
                # --- Bybit V5 Parsing ---
                account_list = balance_data.get('info', {}).get(
                    'result', {}).get('list', [])
                if not account_list:
                    self.logger.warning(
                        "Balance response structure unexpected (no 'list'). Trying top-level parsing.")
                    # Fallback below
                else:
                    account_type_map = {'linear': 'CONTRACT',
                                        'inverse': 'CONTRACT', 'spot': 'SPOT'}
                    target_account_type = account_type_map.get(self.category)
                    unified_account = None
                    specific_account = None

                    for acc in account_list:
                        acc_type = acc.get('accountType')
                        if acc_type == 'UNIFIED':
                            unified_account = acc
                            break  # Prefer UNIFIED if found
                        elif acc_type == target_account_type:
                            specific_account = acc

                    account_to_parse = unified_account or specific_account

                    if account_to_parse:
                        self.logger.debug(
                            f"Parsing balance from account type: {account_to_parse.get('accountType')}")
                        coin_data = account_to_parse.get('coin', [])
                        with getcontext() as ctx:
                            ctx.prec = CALCULATION_PRECISION
                            for coin_info in coin_data:
                                asset = coin_info.get('coin')
                                # Use 'equity' for total value including PnL
                                equity_str = coin_info.get('equity')
                                if asset and equity_str is not None and equity_str != '':
                                    try:
                                        balances[asset] = Decimal(
                                            str(equity_str))
                                    except InvalidOperation:
                                        self.logger.warning(
                                            f"Could not convert balance for {asset} to Decimal: {equity_str}")
                    else:
                        self.logger.warning(
                            f"Could not find relevant account type ('UNIFIED' or '{target_account_type}') in Bybit V5 balance response list.")
                        found_types = [acc.get('accountType')
                                       for acc in account_list]
                        self.logger.debug(
                            f"Account types found in response: {found_types}")
                        # Fallback below

            # --- Fallback or Standard CCXT Parsing ---
            if not balances:  # If Bybit parsing failed or not Bybit
                self.logger.debug(
                    "Using standard CCXT 'total' balance parsing.")
                with getcontext() as ctx:
                    ctx.prec = CALCULATION_PRECISION
                    for asset, bal_info in balance_data.get('total', {}).items():
                        if bal_info is not None:
                            try:
                                balances[asset] = Decimal(str(bal_info))
                            except InvalidOperation:
                                self.logger.warning(
                                    f"Could not convert balance (fallback) for {asset} to Decimal: {bal_info}")

            if not balances:
                self.logger.warning("Parsed balance data is empty.")
                self.logger.debug(f"Raw balance data: {balance_data}")
                return {}  # Return empty dict if parsing found nothing
            else:
                self.logger.info(
                    f"Balance fetched successfully. Assets with total equity: {list(balances.keys())}")
                # Log specific quote asset balance for clarity
                quote_asset = self.config['trading_settings']['quote_asset']
                if quote_asset in balances:
                    self.logger.info(
                        f"{quote_asset} Equity: {balances[quote_asset]:.{DECIMAL_DISPLAY_PRECISION}f}")
                else:
                    self.logger.warning(
                        f"Configured quote asset '{quote_asset}' not found in fetched balances.")
            return balances

        except Exception as e:
            self.logger.error(
                f"Error parsing balance data: {e}", exc_info=True)
            self.logger.debug(f"Raw balance data: {balance_data}")
            return None

    def fetch_positions(self, symbol: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Fetches open positions, converting relevant fields to Decimal.
        Handles Bybit V5 specifics like 'category' and parsing 'info'.
        Filters for non-zero positions matching the symbol.
        """
        target_symbol = symbol or self.config['trading_settings']['symbol']
        self.logger.debug(f"Fetching positions for symbol: {target_symbol}...")
        params = {}
        symbols_arg = [target_symbol]  # CCXT standard way to filter by symbol

        # Bybit V5 specific: Can filter by symbol directly in params, requires category
        if self.exchange_id == 'bybit':
            params['category'] = self.category
            params['symbol'] = target_symbol
            symbols_arg = None  # Don't pass symbols list if using params filter for Bybit

        positions_data = self.safe_ccxt_call(
            'fetch_positions', symbols=symbols_arg, params=params)

        if positions_data is None:
            return None

        processed_positions = []
        try:
            with getcontext() as ctx:
                ctx.prec = CALCULATION_PRECISION  # Use high precision for conversions

                for pos in positions_data:
                    # Filter by symbol again just in case fetch_positions returned others
                    if pos.get('symbol') != target_symbol:
                        continue

                    # --- Reliable check for active position size ---
                    size_str = pos.get('info', {}).get('size')  # Bybit V5 size
                    # Standard CCXT field (might be float/int/None)
                    contracts_val = pos.get('contracts')

                    size_dec = Decimal('0')
                    is_active = False

                    if size_str is not None and size_str != '':
                        try:
                            size_dec = Decimal(str(size_str))
                            if size_dec.is_finite() and size_dec != Decimal(0):
                                is_active = True
                        except InvalidOperation:
                            self.logger.warning(
                                f"Could not parse position size '{size_str}' from info as Decimal for {target_symbol}.")
                    elif contracts_val is not None:
                        try:
                            size_dec = Decimal(str(contracts_val))
                            if size_dec.is_finite() and size_dec != Decimal(0):
                                is_active = True
                        except InvalidOperation:
                            self.logger.warning(
                                f"Could not parse position contracts '{contracts_val}' as Decimal for {target_symbol}.")

                    if not is_active:
                        self.logger.debug(
                            f"Skipping zero-size or invalid-size position for: {target_symbol}")
                        continue  # Skip zero-size or unparsable-size positions

                    processed = pos.copy()  # Work on a copy
                    # Store unified Decimal size in 'contracts'
                    processed['contracts'] = size_dec

                    # --- Convert relevant fields to Decimal ---
                    # Standard CCXT fields
                    decimal_fields_std = ['contractSize', 'entryPrice', 'leverage',
                                          'liquidationPrice', 'markPrice', 'notional',
                                          'unrealizedPnl', 'initialMargin', 'maintenanceMargin',
                                          'initialMarginPercentage', 'maintenanceMarginPercentage',
                                          'marginRatio', 'collateral']
                    # Bybit V5 specific fields often in 'info'
                    decimal_fields_info = ['avgPrice', 'cumRealisedPnl', 'liqPrice', 'markPrice',
                                           'positionValue', 'stopLoss', 'takeProfit', 'trailingStop',
                                           'unrealisedPnl', 'positionIM', 'positionMM', 'createdTime', 'updatedTime']
                    # Exclude 'size' as we handled it above

                    # Process standard fields
                    for field in decimal_fields_std:
                        if field in processed and processed[field] is not None:
                            try:
                                processed[field] = Decimal(
                                    str(processed[field]))
                            except InvalidOperation:
                                self.logger.warning(
                                    f"Could not convert standard position field '{field}' ({processed[field]}) to Decimal for {target_symbol}.")
                                processed[field] = Decimal('NaN')

                    # Process info fields
                    if 'info' in processed and isinstance(processed['info'], dict):
                        info = processed['info']
                        for field in decimal_fields_info:
                            if field in info and info[field] is not None and info[field] != '':
                                try:
                                    info[field] = Decimal(str(info[field]))
                                except InvalidOperation:
                                    self.logger.warning(
                                        f"Could not convert position info field '{field}' ({info[field]}) to Decimal for {target_symbol}.")
                                    info[field] = Decimal('NaN')

                    # --- Determine position side more reliably ---
                    side_enum = PositionSide.NONE
                    if 'info' in processed and isinstance(processed['info'], dict) and 'side' in processed['info']:
                        side_str = str(processed['info']['side']).lower()
                        if side_str == 'buy':
                            side_enum = PositionSide.LONG
                        elif side_str == 'sell':
                            side_enum = PositionSide.SHORT
                        elif side_str == 'none':
                            side_enum = PositionSide.NONE
                        else:
                            self.logger.warning(
                                f"Unrecognized side '{side_str}' in position info for {target_symbol}")

                    # Fallback to CCXT 'side' if info side is missing or ambiguous ('None')
                    if side_enum == PositionSide.NONE and 'side' in processed and processed['side']:
                        side_str_std = str(processed['side']).lower()
                        if side_str_std == 'long':
                            side_enum = PositionSide.LONG
                        elif side_str_std == 'short':
                            side_enum = PositionSide.SHORT

                    # Final check: Log warning if side is still None despite non-zero size
                    if side_enum == PositionSide.NONE and size_dec != Decimal(0):
                        self.logger.warning(
                            f"Position for {target_symbol} has size {size_dec} but side is determined as 'None'. Check exchange data consistency.")
                        # Might indicate an issue, but store 'none' as determined
                        # Logic using the position should handle 'none' side defensively

                    # Update main 'side' field with 'long'/'short'/'none' string
                    processed['side'] = side_enum.value

                    # Add positionIdx and ensure it's an integer
                    pos_idx_val = processed.get('info', {}).get('positionIdx')
                    if pos_idx_val is not None:
                        try:
                            processed['positionIdx'] = int(pos_idx_val)
                        except (ValueError, TypeError):
                            self.logger.warning(
                                f"Could not parse positionIdx '{pos_idx_val}' as integer for {target_symbol}.")
                            # Mark as None if parsing fails
                            processed['positionIdx'] = None
                    else:
                        # Default to 0 for one-way mode if not present
                        processed['positionIdx'] = 0 if not self.hedge_mode else None

                    processed_positions.append(processed)

            self.logger.info(
                f"Fetched {len(processed_positions)} active position(s) for {target_symbol}.")
            return processed_positions
        except Exception as e:
            self.logger.error(
                f"Error processing position data for {target_symbol}: {e}", exc_info=True)
            self.logger.debug(f"Raw positions data: {positions_data}")
            return None

    def format_value_for_api(self, symbol: str, value_type: str, value: Decimal,
                             rounding_mode: str = ROUND_HALF_UP) -> str:
        """
        Formats amount or price to string based on market precision/step size for API calls.
        Uses ccxt's precision formatting methods.
        Raises ValueError on invalid input or if market data is missing.
        Allows specifying rounding mode (e.g., ROUND_DOWN for amounts).
        """
        if not isinstance(value, Decimal):
            raise ValueError(
                f"Invalid input value type for formatting: {type(value)}. Expected Decimal.")
        if not value.is_finite():
            raise ValueError(
                f"Invalid Decimal value for formatting: {value}. Must be finite.")

        market = self.get_market(symbol)
        if not market:
            raise ValueError(
                f"Market data not found for {symbol}, cannot format value.")

        # CCXT formatting methods usually expect float input.
        # Be cautious about float conversion precision loss for very small numbers.
        # However, ccxt methods are designed to handle this based on market precision rules.
        value_float = float(value)

        try:
            if value_type == 'amount':
                # Use amount_to_precision for formatting quantity
                # CCXT's amount_to_precision often implies ROUND_DOWN (floor) for step size logic
                # Check CCXT docs/implementation or test for specific exchange behavior if exact rounding is critical.
                # Bybit typically truncates (ROUND_DOWN) amounts to step size.
                formatted_value = self.exchange.amount_to_precision(
                    symbol, value_float)
            elif value_type == 'price':
                # Use price_to_precision for formatting price
                # CCXT's price_to_precision typically uses ROUND_HALF_UP based on tick size.
                # Override rounding_mode if needed (e.g., conservative SL/TP) - Note: CCXT might not expose rounding mode control directly.
                # If specific rounding (UP/DOWN) is needed, manual quantization might be required instead of relying solely on price_to_precision.
                formatted_value = self.exchange.price_to_precision(
                    symbol, value_float)
            else:
                raise ValueError(
                    f"Invalid value_type: {value_type}. Use 'amount' or 'price'.")

            # CCXT formatting methods return strings. Return as string.
            return formatted_value
        except ccxt.ExchangeError as e:
            self.logger.error(
                f"CCXT error formatting {value_type} ('{value}') for {symbol}: {e}")
            raise ValueError(
                f"CCXT error formatting {value_type} for {symbol}") from e
        except Exception as e:
            self.logger.error(
                f"Unexpected error formatting {value_type} ('{value}') for {symbol}: {e}", exc_info=True)
            raise ValueError(
                f"Unexpected error formatting {value_type} for {symbol}") from e

    def quantize_value(self, value: Decimal, precision_type: str, market: Dict[str, Any],
                       rounding_mode: str = ROUND_HALF_UP) -> Optional[Decimal]:
        """
        Quantizes a Decimal value based on market precision (tick/step size).
        Uses Decimal arithmetic for accurate quantization.
        precision_type should be 'price' or 'amount'.
        """
        if not isinstance(value, Decimal) or not value.is_finite():
            self.logger.error(f"Invalid value for quantization: {value}")
            return None
        if not market or 'precision' not in market:
            self.logger.error(
                "Market data or precision missing for quantization.")
            return None

        precision_val_str = market['precision'].get(precision_type)
        if precision_val_str is None:
            self.logger.error(
                f"Precision value for type '{precision_type}' not found in market data.")
            # Fallback: return original value? Or fail? Fail is safer.
            return None

        try:
            tick_size = Decimal(str(precision_val_str))
            if not tick_size.is_finite() or tick_size <= 0:
                raise InvalidOperation("Invalid tick/step size")

            # Quantize using Decimal's quantize method with the tick size
            # value.quantize(tick_size, rounding=rounding_mode) doesn't work directly like this for step size
            # Formula: round(value / tick_size) * tick_size
            # Ensure context precision is high enough for the division
            with getcontext() as ctx:
                ctx.prec = CALCULATION_PRECISION  # Ensure high precision for intermediate division
                quantized_value = (
                    value / tick_size).quantize(Decimal('1'), rounding=rounding_mode) * tick_size
            return quantized_value

        except (InvalidOperation, ValueError, TypeError) as e:
            self.logger.error(
                f"Error quantizing value {value} with {precision_type} precision '{precision_val_str}': {e}")
            return None

    def create_order(self, symbol: str, order_type: str, side: OrderSide, amount: Decimal,
                     price: Optional[Decimal] = None, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Creates an order, ensuring amount and price are correctly formatted using market precision.
        Handles Bybit V5 specifics like 'category' and 'positionIdx'.
        Uses format_value_for_api for final string conversion.
        """
        self.logger.info(
            f"Attempting to create {side.value} {order_type} order for {amount} {symbol} @ {price or 'market'}...")

        if amount <= 0:
            self.logger.error(
                f"Cannot create order: Amount must be positive ({amount}).")
            return None
        # Allow price=0 for certain order types if needed, but generally require positive price for limit orders
        if order_type.lower() not in ['market', 'stop_market', 'take_profit_market'] and (price is None or price <= 0):
            self.logger.error(
                f"Cannot create non-market order: Price must be positive ({price}).")
            return None

        market = self.get_market(symbol)
        if not market:
            self.logger.error(
                f"Cannot create order: Market data for {symbol} not found.")
            return None

        try:
            # Format amount (typically ROUND_DOWN for exchanges)
            amount_str = self.format_value_for_api(
                symbol, 'amount', amount, rounding_mode=ROUND_DOWN)
            # Re-check if formatted amount is still positive (could become zero if input was smaller than step size)
            if Decimal(amount_str) <= 0:
                self.logger.error(
                    f"Order amount {amount} after formatting to step size became non-positive ({amount_str}). Cannot place order.")
                return None

            price_str = None
            if price is not None and order_type.lower() not in ['market', 'stop_market', 'take_profit_market']:
                price_str = self.format_value_for_api(
                    symbol, 'price', price, rounding_mode=ROUND_HALF_UP)  # Default rounding for price

            self.logger.debug(
                f"Formatted order values: Amount='{amount_str}', Price='{price_str}'")

        except ValueError as e:
            self.logger.error(f"Error formatting order values: {e}")
            return None
        except Exception as e:
            self.logger.error(
                f"Unexpected error during value formatting: {e}", exc_info=True)
            return None

        # --- Prepare Parameters ---
        order_params = params.copy() if params else {}

        # --- Bybit V5 Specific Parameters ---
        if self.exchange_id == 'bybit':
            # 'category' is handled by safe_ccxt_call

            # Determine positionIdx for hedge mode entries/exits
            if self.hedge_mode:
                if 'positionIdx' not in order_params:
                    # For opening: 1 for Buy/Long, 2 for Sell/Short (default assumption)
                    # For closing: 'reduceOnly' is key, positionIdx should match the position being closed (caller should provide if needed)
                    # Assume opening if not reduceOnly
                    if not order_params.get('reduceOnly', False):
                        order_params['positionIdx'] = 1 if side == OrderSide.BUY else 2
                        self.logger.debug(
                            f"Hedge mode: Setting positionIdx={order_params['positionIdx']} for opening {side.value} order.")
                    # If reduceOnly is true, caller *must* provide the correct positionIdx in params
                    # Handled in close_position method.
            else:
                # Ensure positionIdx is 0 for one-way mode if not specified
                order_params.setdefault('positionIdx', 0)

        # Prepare arguments for safe_ccxt_call
        call_args = [symbol, order_type, side.value, amount_str]
        if order_type.lower() in ['market', 'stop_market', 'take_profit_market']:
            call_args.append(None)  # No price argument for market orders
        else:
            if price_str is None:
                self.logger.error(
                    f"Price is required for order type '{order_type}' but was not provided or formatted correctly.")
                return None
            call_args.append(price_str)  # Price argument for limit orders

        order_result = self.safe_ccxt_call(
            'create_order', *call_args, params=order_params)

        if order_result:
            order_id = order_result.get('id')
            status = order_result.get('status', 'unknown')
            self.logger.info(
                f"Order creation request successful. Order ID: {order_id}, Initial Status: {status}")
            # Optional: Parse result back to Decimal if needed elsewhere immediately
            # e.g., if order_result['info'] contains useful Decimal fields
        else:
            # Error logged by safe_ccxt_call
            self.logger.error(
                f"Failed to create {side.value} {order_type} order for {symbol}.")

        return order_result  # Return raw CCXT result (or None on failure)

    def set_leverage(self, symbol: str, leverage: Decimal) -> bool:
        """Sets leverage for a symbol. Handles Bybit V5 specifics."""
        self.logger.info(f"Setting leverage for {symbol} to {leverage}x...")
        if not isinstance(leverage, Decimal):
            self.logger.error(
                f"Invalid leverage type: {type(leverage)}. Must be Decimal.")
            return False
        if not leverage.is_finite() or leverage <= 0:
            self.logger.error(
                f"Invalid leverage value: {leverage}. Must be a positive finite Decimal.")
            return False

        # Leverage value should typically be passed as a number (float/int) to CCXT's set_leverage
        # but Bybit V5 params require strings for buy/sell leverage.
        leverage_float = float(leverage)
        # Bybit expects integer string for leverage
        leverage_str = f"{int(leverage)}"

        params = {}
        # Bybit V5 requires setting buyLeverage and sellLeverage separately in params
        if self.exchange_id == 'bybit':
            # 'category' handled by safe_ccxt_call
            params['buyLeverage'] = leverage_str
            params['sellLeverage'] = leverage_str
            # Symbol is passed as main argument, not needed in params for Bybit V5 setLeverage via CCXT

        # CCXT's set_leverage expects: leverage (number), symbol (string), params (dict)
        result = self.safe_ccxt_call(
            'set_leverage', leverage_float, symbol, params=params)

        if result is not None:  # safe_ccxt_call returns None only on failure after retries
            self.logger.info(
                f"Leverage for {symbol} set to {leverage}x request sent successfully.")
            # Optional: Check Bybit specific retCode if available in result['info']
            if self.exchange_id == 'bybit' and isinstance(result, dict) and 'info' in result:
                ret_code = result['info'].get('retCode')
                ret_msg = result['info'].get('retMsg', 'Unknown Bybit message')
                if ret_code == 0:
                    self.logger.info(
                        "Bybit API confirmed successful leverage setting (retCode 0).")
                    return True
                else:
                    # Request sent but Bybit indicated an issue (e.g., 110043: leverage not modified)
                    # Treat "not modified" as success if the value is already correct.
                    # Need more info to confirm current leverage if error occurs.
                    self.logger.warning(
                        f"Bybit leverage setting response - Code: {ret_code}, Msg: {ret_msg}. Assuming success if no modification needed, check manually if unsure.")
                    # Let's assume success unless it's a clear failure code.
                    # Error codes indicating definite failure might be different.
                    # If ret_code implies failure (e.g. param error), safe_ccxt_call might have caught it earlier.
                    # If it's just "not modified", we can consider it okay.
                    return True  # Assume okay for now
            # If no 'info' or not Bybit, assume success based on non-None result
            return True
        else:
            # Error handled and logged by safe_ccxt_call
            return False

    def set_protection(self, symbol: str, stop_loss: Optional[Decimal] = None,
                       take_profit: Optional[Decimal] = None, trailing_stop: Optional[Dict[str, Decimal]] = None,
                       position_idx: Optional[int] = None) -> bool:
        """
        Sets stop loss, take profit, or trailing stop for a position using Bybit V5's trading stop endpoint.
        Note: This modifies existing position parameters, does not place new orders.
        Uses quantize_value and format_value_for_api for price formatting.
        """
        action_parts = []
        if stop_loss is not None:
            action_parts.append(f"SL={stop_loss}")
        if take_profit is not None:
            action_parts.append(f"TP={take_profit}")
        if trailing_stop is not None:
            action_parts.append(f"TSL={trailing_stop}")

        if not action_parts:
            self.logger.warning(
                "set_protection called with no protection levels specified.")
            return False

        self.logger.info(
            f"Attempting to set protection for {symbol} (Idx: {position_idx if position_idx is not None else 'N/A'}): {' / '.join(action_parts)}")

        market = self.get_market(symbol)
        if not market:
            self.logger.error(
                f"Cannot set protection: Market data for {symbol} not found.")
            return False

        # --- Prepare Parameters for private_post_position_trading_stop ---
        params = {
            # 'category': self.category, # safe_ccxt_call adds this
            'symbol': symbol,
        }

        try:
            # Quantize and Format SL and TP using market precision
            if stop_loss is not None:
                if stop_loss <= 0:
                    raise ValueError("Stop loss must be positive.")
                # Quantize first for accuracy, then format for API string
                quantized_sl = self.quantize_value(
                    stop_loss, 'price', market, ROUND_HALF_UP)
                if quantized_sl is None:
                    raise ValueError("Failed to quantize stop loss price.")
                params['stopLoss'] = self.format_value_for_api(
                    symbol, 'price', quantized_sl)
                # Add trigger price type if needed (defaults usually okay)
                # params['slTriggerBy'] = 'MarkPrice' # Or LastPrice, IndexPrice

            if take_profit is not None:
                if take_profit <= 0:
                    raise ValueError("Take profit must be positive.")
                quantized_tp = self.quantize_value(
                    take_profit, 'price', market, ROUND_HALF_UP)
                if quantized_tp is None:
                    raise ValueError("Failed to quantize take profit price.")
                params['takeProfit'] = self.format_value_for_api(
                    symbol, 'price', quantized_tp)
                # params['tpTriggerBy'] = 'MarkPrice'

            # --- Trailing Stop Handling (Bybit V5 specific) ---
            if trailing_stop:
                # Bybit uses 'trailingStop' for the distance/value (price points or percentage string)
                # 'activePrice' for the activation price (optional)
                ts_value = trailing_stop.get('distance') or trailing_stop.get(
                    'value')  # Use 'distance' or 'value' key
                ts_active_price = trailing_stop.get('activation_price')
                # ts_is_percentage = trailing_stop.get('is_percentage', False) # Example flag

                if ts_value is not None:
                    if not isinstance(ts_value, Decimal) or not ts_value.is_finite() or ts_value <= 0:
                        raise ValueError(
                            "Trailing stop distance/value must be a positive finite Decimal.")

                    # Bybit API expects TSL distance as a string value.
                    # If it represents price points, quantize and format it like a price difference.
                    # If it's a percentage, format as "X%". Requires careful handling based on config.
                    # Assuming price points here:
                    # Quantize the *distance* based on price precision (tick size)
                    quantized_ts_dist = self.quantize_value(
                        ts_value, 'price', market, ROUND_HALF_UP)
                    if quantized_ts_dist is None:
                        raise ValueError(
                            "Failed to quantize trailing stop distance.")
                    params['trailingStop'] = self.format_value_for_api(
                        symbol, 'price', quantized_ts_dist)
                    # If percentage: params['trailingStop'] = f"{ts_value}%" # Requires validation ts_value is suitable percentage
                    self.logger.debug(
                        f"Formatted trailing stop value: {params['trailingStop']}")

                if ts_active_price is not None:
                    if not isinstance(ts_active_price, Decimal) or not ts_active_price.is_finite() or ts_active_price <= 0:
                        raise ValueError(
                            "Trailing stop activation price must be a positive finite Decimal.")
                    quantized_ts_active = self.quantize_value(
                        ts_active_price, 'price', market, ROUND_HALF_UP)
                    if quantized_ts_active is None:
                        raise ValueError(
                            "Failed to quantize trailing stop activation price.")
                    params['activePrice'] = self.format_value_for_api(
                        symbol, 'price', quantized_ts_active)
                    self.logger.debug(
                        f"Formatted trailing stop activation price: {params['activePrice']}")

                # Add trigger price types if needed (defaults are usually okay)
                # params['tpslMode'] = 'Partial' # Or 'Full'
                # params['slTriggerBy'] = 'MarkPrice'
                # params['tpTriggerBy'] = 'MarkPrice'

        except ValueError as e:
            self.logger.error(
                f"Invalid or unformattable protection parameter value: {e}")
            return False
        except Exception as e:
            self.logger.error(
                f"Error formatting protection parameters: {e}", exc_info=True)
            return False

        # --- Position Index for Hedge Mode ---
        if self.hedge_mode:
            if position_idx is None:
                self.logger.error(
                    "Hedge mode active, but positionIdx is required for set_protection and was not provided.")
                return False
            params['positionIdx'] = position_idx
        else:
            params.setdefault('positionIdx', 0)

        # --- Execute API Call ---
        # Use the specific private method for Bybit V5 trading stop
        if not hasattr(self.exchange, 'private_post_position_trading_stop'):
            self.logger.error(
                "CCXT exchange object does not have 'private_post_position_trading_stop'. Cannot set protection this way.")
            # Alternative: Could try modifying an existing SL/TP order? More complex.
            return False

        result = self.safe_ccxt_call(
            'private_post_position_trading_stop', params=params)

        # --- Process Result ---
        if result is not None:
            ret_code = result.get('retCode')
            ret_msg = result.get('retMsg', 'No message')
            ext_info = result.get('retExtInfo', {})

            if ret_code == 0:
                self.logger.info(
                    f"Protection levels set/updated successfully via API for {symbol} (Idx: {params['positionIdx']}).")
                return True
            else:
                # Log specific Bybit error code/message
                self.logger.error(
                    f"Failed to set protection for {symbol}. Code: {ret_code}, Msg: {ret_msg}, Extra: {ext_info}")
                self.logger.debug(f"Params sent: {params}")
                # Check for common benign errors (e.g., SL/TP not modified because value is the same)
                # Bybit might use specific codes for this, adjust logic if needed. Example: 34036?
                # if ret_code == 34036: # Hypothetical code for "SL/TP not modified"
                #    self.logger.info("Protection levels were not modified (potentially already set to the desired value).")
                #    return True # Treat as success if no change was needed
                return False
        else:
            # safe_ccxt_call already logged the failure reason
            self.logger.error(
                f"API call failed for setting protection on {symbol}.")
            return False


# --- Trading Strategy Analyzer ---

class TradingAnalyzer:
    """Analyzes market data using technical indicators to generate trading signals."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.logger = logger
        self.indicator_config = config.get(
            "indicator_settings", {})  # Renamed for clarity
        self.weights = self.indicator_config.get("signal_weights", {})

        # Ensure weights are Decimals and normalize them
        self._initialize_weights()

    def _initialize_weights(self):
        """Validates and normalizes signal weights loaded from config."""
        if not self.weights:
            self.logger.warning(
                "No 'signal_weights' found in indicator_settings. Using default weights.")
            self.weights = {"rsi": Decimal("0.3"), "macd": Decimal(
                "0.4"), "ema_cross": Decimal("0.3")}
        else:
            # Ensure loaded weights are valid positive Decimals
            valid_weights = {}
            for k, v in self.weights.items():
                if isinstance(v, Decimal) and v.is_finite() and v >= 0:
                    valid_weights[k] = v
                else:
                    self.logger.warning(
                        f"Invalid signal weight for '{k}' ({v}). Removing from calculation.")
            self.weights = valid_weights

        if not self.weights:
            self.logger.error(
                "No valid signal weights configured. Cannot generate weighted signals.")
            return  # Leave weights empty

        total_weight = sum(self.weights.values())
        # Check if total weight is effectively zero
        if total_weight <= Decimal('1e-18'):
            self.logger.error(
                "Total signal weight is zero or negligible. Cannot normalize. Disabling weighted signals.")
            self.weights = {}  # Disable weights
        elif abs(total_weight - Decimal("1.0")) > Decimal("1e-9"):  # Check if normalization needed
            self.logger.warning(
                f"Signal weights sum to {total_weight}, not 1. Normalizing.")
            try:
                # Use high precision for division
                with getcontext() as ctx:
                    ctx.prec = CALCULATION_PRECISION
                    self.weights = {k: (v / total_weight).quantize(Decimal("1e-6"), rounding=ROUND_HALF_UP)
                                    # Quantize for clarity
                                    for k, v in self.weights.items()}
                self.logger.info(f"Normalized weights: {self.weights}")
                # Re-check sum after normalization (optional)
                # final_sum = sum(self.weights.values())
                # self.logger.debug(f"Sum after normalization: {final_sum}")
            except InvalidOperation as e:
                self.logger.error(
                    f"Error normalizing weights: {e}. Disabling weighted signals.")
                self.weights = {}

    def calculate_indicators(self, ohlcv_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculates all configured technical indicators using pandas_ta."""
        if ohlcv_df is None or ohlcv_df.empty:
            self.logger.error(
                "Cannot calculate indicators: OHLCV data is missing or empty.")
            return None

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in ohlcv_df.columns for col in required_cols):
            self.logger.error(
                f"Missing required OHLCV columns in DataFrame: {required_cols}")
            return None

        # Verify data types (should be Decimal from fetch_ohlcv)
        try:
            if not all(isinstance(ohlcv_df[col].iloc[0], Decimal) for col in required_cols if not ohlcv_df[col].empty):
                self.logger.error(
                    "OHLCV columns are not all Decimal type. Conversion error likely occurred earlier.")
                return None
        except IndexError:
            self.logger.error("OHLCV DataFrame seems empty or malformed.")
            return None

        self.logger.debug(
            f"Calculating indicators for {len(ohlcv_df)} candles...")
        df = ohlcv_df.copy()

        # --- Convert Decimal columns to float for pandas_ta compatibility ---
        # Handle potential non-finite Decimals (NaN, Inf) before converting
        float_cols = ['open', 'high', 'low', 'close', 'volume']
        # Create new df for float conversion
        df_float = pd.DataFrame(index=df.index)
        conversion_failed = False
        for col in float_cols:
            try:
                # Replace non-finite Decimals with pd.NA before converting to float
                # Use apply for robust handling
                df_float[col] = df[col].apply(lambda x: float(
                    x) if isinstance(x, Decimal) and x.is_finite() else pd.NA)
                # Ensure the column is numeric, coercing errors to NaN
                df_float[col] = pd.to_numeric(df_float[col], errors='coerce')
            except Exception as e:
                self.logger.error(
                    f"Error converting column {col} to float for TA calculation: {e}", exc_info=True)
                conversion_failed = True
        if conversion_failed:
            return None

        # Drop rows with NaN in essential OHLC columns after conversion, as TA libs often fail
        initial_len = len(df_float)
        df_float.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        if len(df_float) < initial_len:
            self.logger.warning(
                f"Dropped {initial_len - len(df_float)} rows with NaN in OHLC columns before TA calculation.")

        if df_float.empty:
            self.logger.error(
                "DataFrame is empty after handling NaNs in OHLC data. Cannot calculate indicators.")
            return None

        # --- Calculate Indicators using pandas_ta on the float DataFrame ---
        try:
            # RSI
            rsi_period = self.indicator_config.get("rsi_period")
            if isinstance(rsi_period, int) and rsi_period > 0:
                df_float.ta.rsi(length=rsi_period, append=True,
                                col_names=(f"RSI_{rsi_period}"))

            # MACD
            macd_fast = self.indicator_config.get("macd_fast")
            macd_slow = self.indicator_config.get("macd_slow")
            macd_signal = self.indicator_config.get("macd_signal")
            if all(isinstance(p, int) and p > 0 for p in [macd_fast, macd_slow, macd_signal]):
                # Uses default names MACD_F_S_SIG, MACDh_..., MACDs_...
                df_float.ta.macd(fast=macd_fast, slow=macd_slow,
                                 signal=macd_signal, append=True)

            # EMAs
            ema_short_period = self.indicator_config.get("ema_short_period")
            if isinstance(ema_short_period, int) and ema_short_period > 0:
                df_float.ta.ema(length=ema_short_period, append=True,
                                col_names=(f"EMA_{ema_short_period}"))
            ema_long_period = self.indicator_config.get("ema_long_period")
            if isinstance(ema_long_period, int) and ema_long_period > 0:
                df_float.ta.ema(length=ema_long_period, append=True,
                                col_names=(f"EMA_{ema_long_period}"))

            # ATR
            atr_period = self.indicator_config.get("atr_period")
            if isinstance(atr_period, int) and atr_period > 0:
                # Using RMA (similar to Wilder's) is common for ATR
                df_float.ta.atr(length=atr_period, append=True,
                                mamode='rma', col_names=(f"ATR_{atr_period}"))

            self.logger.debug(
                f"Indicators calculated (float). Columns added: {df_float.columns.difference(float_cols).tolist()}")

            # --- Merge calculated float indicators back into the original Decimal DataFrame ---
            # Identify newly added indicator columns
            indicator_cols = df_float.columns.difference(float_cols).tolist()

            # Convert float indicators back to Decimal for internal consistency
            with getcontext() as ctx:
                ctx.prec = CALCULATION_PRECISION  # Use high precision for conversion
                for col in indicator_cols:
                    # Handle potential NaN/Inf from calculations using math.isfinite
                    df[col] = df_float[col].apply(
                        lambda x: Decimal(str(x)) if pd.notna(x) and isinstance(
                            x, (float, int)) and math.isfinite(x) else Decimal('NaN')
                    )

            # Keep original Decimal OHLCV columns (already in 'df')
            # Reindex df to match the original ohlcv_df index to include rows dropped during float calculation
            # This ensures the output DataFrame has the same index and length as the input OHLCV data
            df = df.reindex(ohlcv_df.index)

            self.logger.debug(
                "Indicators converted back to Decimal and merged.")
            return df

        except Exception as e:
            self.logger.error(
                f"Error calculating technical indicators with pandas_ta: {e}", exc_info=True)
            return None

    def generate_signal(self, indicators_df: pd.DataFrame) -> Tuple[Signal, Dict[str, Any]]:
        """
        Generates a trading signal based on the latest indicator values
        and configured weighted scoring.
        Returns the final signal and contributing factors/scores.
        """
        if indicators_df is None or indicators_df.empty:
            self.logger.warning(
                "Cannot generate signal: Indicators data is missing or empty.")
            return Signal.HOLD, {}

        try:
            # Get the absolute latest row based on index (most recent candle)
            latest_data = indicators_df.iloc[-1]
        except IndexError:
            self.logger.error(
                "Cannot get latest data: Indicators DataFrame is unexpectedly empty after calculation.")
            return Signal.HOLD, {}

        # Check if latest data timestamp is recent enough (optional sanity check)
        # time_diff = pd.Timestamp.utcnow().tz_localize(None) - latest_data.name # Assuming index is timestamp
        # if time_diff > pd.Timedelta(minutes=15): # Example threshold
        #      self.logger.warning(f"Latest indicator data is stale ({latest_data.name}). Signal might be unreliable.")

        scores = {}
        contributing_factors = {}
        # Use high precision for calculations
        with getcontext() as ctx:
            ctx.prec = CALCULATION_PRECISION

            # --- Evaluate Individual Indicators ---
            # RSI
            rsi_key = "rsi"
            rsi_weight = self.weights.get(rsi_key, Decimal(0))
            rsi_period = self.indicator_config.get("rsi_period")
            rsi_col = f"RSI_{rsi_period}" if isinstance(
                rsi_period, int) else None
            if rsi_weight > 0 and rsi_col and rsi_col in latest_data:
                rsi_value = latest_data[rsi_col]
                if isinstance(rsi_value, Decimal) and rsi_value.is_finite():
                    overbought = self.indicator_config.get(
                        "rsi_overbought", Decimal("70"))
                    oversold = self.indicator_config.get(
                        "rsi_oversold", Decimal("30"))
                    rsi_score = Decimal(0)
                    if rsi_value > overbought:
                        rsi_score = Decimal("-1")  # Sell
                    elif rsi_value < oversold:
                        rsi_score = Decimal("1")  # Buy
                    scores[rsi_key] = rsi_score * rsi_weight
                    contributing_factors[rsi_key] = {
                        "value": rsi_value, "score": rsi_score, "weight": rsi_weight}
                else:
                    self.logger.debug(
                        f"Invalid RSI value ({rsi_value}). Skipping RSI score.")

            # MACD (Histogram)
            macd_key = "macd"
            macd_weight = self.weights.get(macd_key, Decimal(0))
            macd_fast = self.indicator_config.get("macd_fast", 12)
            macd_slow = self.indicator_config.get("macd_slow", 26)
            macd_signal_p = self.indicator_config.get("macd_signal", 9)
            macdh_col = f"MACDh_{macd_fast}_{macd_slow}_{macd_signal_p}" if all(
                isinstance(p, int) for p in [macd_fast, macd_slow, macd_signal_p]) else None
            if macd_weight > 0 and macdh_col and macdh_col in latest_data:
                macd_hist = latest_data[macdh_col]
                if isinstance(macd_hist, Decimal) and macd_hist.is_finite():
                    hist_threshold = self.indicator_config.get(
                        "macd_hist_threshold", Decimal("0"))
                    macd_score = Decimal(0)
                    if macd_hist > hist_threshold:
                        macd_score = Decimal("1")  # Buy
                    elif macd_hist < -hist_threshold:
                        macd_score = Decimal("-1")  # Sell
                    scores[macd_key] = macd_score * macd_weight
                    contributing_factors[macd_key] = {
                        "histogram": macd_hist, "score": macd_score, "weight": macd_weight}
                else:
                    self.logger.debug(
                        f"Invalid MACD Histogram value ({macd_hist}). Skipping MACD score.")

            # EMA Cross
            ema_key = "ema_cross"
            ema_cross_weight = self.weights.get(ema_key, Decimal(0))
            ema_short_period = self.indicator_config.get("ema_short_period")
            ema_long_period = self.indicator_config.get("ema_long_period")
            ema_short_col = f"EMA_{ema_short_period}" if isinstance(
                ema_short_period, int) else None
            ema_long_col = f"EMA_{ema_long_period}" if isinstance(
                ema_long_period, int) else None
            if ema_cross_weight > 0 and ema_short_col and ema_long_col and all(c in latest_data for c in [ema_short_col, ema_long_col]):
                ema_short = latest_data[ema_short_col]
                ema_long = latest_data[ema_long_col]
                if isinstance(ema_short, Decimal) and ema_short.is_finite() and isinstance(ema_long, Decimal) and ema_long.is_finite():
                    ema_cross_score = Decimal(0)
                    # Simple state check: short > long is bullish
                    if ema_short > ema_long:
                        ema_cross_score = Decimal("1")  # Bullish state
                    elif ema_short < ema_long:
                        ema_cross_score = Decimal("-1")  # Bearish state
                    scores[ema_key] = ema_cross_score * ema_cross_weight
                    contributing_factors[ema_key] = {
                        "short_ema": ema_short, "long_ema": ema_long, "score": ema_cross_score, "weight": ema_cross_weight}
                else:
                    self.logger.debug(
                        f"Invalid EMA values (Short={ema_short}, Long={ema_long}). Skipping EMA Cross score.")

        # --- Combine Scores ---
        if not scores:
            self.logger.warning(
                "No valid indicator scores generated. Defaulting to HOLD.")
            final_score = Decimal(0)
        else:
            final_score = sum(scores.values())

        # --- Determine Final Signal ---
        strong_buy_thresh = self.indicator_config.get(
            "strong_buy_threshold", Decimal("0.7"))
        buy_thresh = self.indicator_config.get("buy_threshold", Decimal("0.2"))
        sell_thresh = self.indicator_config.get(
            "sell_threshold", Decimal("-0.2"))
        strong_sell_thresh = self.indicator_config.get(
            "strong_sell_threshold", Decimal("-0.7"))

        # Validate thresholds order
        if not (strong_sell_thresh <= sell_thresh < buy_thresh <= strong_buy_thresh):
            self.logger.error(
                "Signal thresholds are improperly configured (overlapping or wrong order: SS <= S < B <= SB). Using default HOLD.")
            final_signal = Signal.HOLD
        else:
            if final_score >= strong_buy_thresh:
                final_signal = Signal.STRONG_BUY
            elif final_score >= buy_thresh:
                final_signal = Signal.BUY
            elif final_score <= strong_sell_thresh:
                final_signal = Signal.STRONG_SELL
            elif final_score <= sell_thresh:
                final_signal = Signal.SELL
            else:
                final_signal = Signal.HOLD

        # Quantize score for logging/state
        quantized_score = final_score.quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP)
        self.logger.info(
            f"Signal generated: {final_signal.name} (Score: {quantized_score})")
        # Factors remain high precision Decimals
        self.logger.debug(f"Contributing factors: {contributing_factors}")

        # Prepare output details (can be saved to state if needed)
        # Convert Decimal factors to string for easier JSON saving if required later
        # factors_out = {k: {k2: str(v2) for k2, v2 in v.items()} for k, v in contributing_factors.items()}
        signal_details_out = {"final_score": quantized_score,
                              "factors": contributing_factors}  # Keep Decimals for now

        return final_signal, signal_details_out


# --- Position and Risk Management ---

class PositionManager:
    """Handles position sizing, stop-loss, take-profit, and exit logic."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, exchange_wrapper: BybitV5Wrapper):
        self.logger = logger
        self.config = config
        self.risk_config = config.get("risk_management", {})
        self.trading_config = config.get("trading_settings", {})
        self.indicator_config = config.get("indicator_settings", {})
        self.exchange = exchange_wrapper
        self.symbol = self.trading_config.get("symbol")
        if not self.symbol:
            logger.critical(
                "Trading symbol is not defined in config. Cannot initialize PositionManager.")
            raise ValueError("Trading symbol is required.")
        self.hedge_mode = self.trading_config.get("hedge_mode", False)

    def get_base_quote(self, market: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """Gets base and quote assets from the provided market data."""
        if market:
            base = market.get('base')
            quote = market.get('quote')
            if base and quote:
                return base, quote
            else:
                self.logger.error(
                    f"Base or Quote asset missing in market data for {self.symbol}.")
        return None, None

    def calculate_position_size(self, entry_price: Decimal, stop_loss_price: Decimal,
                                available_equity: Decimal, quote_asset: str) -> Optional[Decimal]:
        """
        Calculates position size based on risk percentage of equity and stop distance.
        Returns size in base currency (e.g., BTC amount for BTC/USDT), adjusted for market limits/precision.
        Uses high precision Decimal calculations.
        """
        risk_percent_config = self.risk_config.get(
            "risk_per_trade_percent", Decimal("1.0"))
        # Validation already done in load_config, assume valid Decimal here
        risk_percent = risk_percent_config / Decimal("100")

        # Validate inputs
        if not isinstance(entry_price, Decimal) or not entry_price.is_finite() or entry_price <= 0:
            self.logger.error(
                f"Invalid entry_price for size calculation: {entry_price}")
            return None
        if not isinstance(stop_loss_price, Decimal) or not stop_loss_price.is_finite() or stop_loss_price <= 0:
            self.logger.error(
                f"Invalid stop_loss_price for size calculation: {stop_loss_price}")
            return None
        if not isinstance(available_equity, Decimal) or not available_equity.is_finite() or available_equity <= 0:
            self.logger.error(
                f"Invalid available_equity for size calculation: {available_equity}")
            return None
        if entry_price == stop_loss_price:
            self.logger.error(
                "Entry price and stop-loss price cannot be the same for size calculation.")
            return None

        market = self.exchange.get_market(self.symbol)
        if not market:
            self.logger.error(
                f"Cannot calculate position size: Market data for {self.symbol} not found.")
            return None
        base_asset, market_quote_asset = self.get_base_quote(market)
        if not base_asset or not market_quote_asset:
            self.logger.error(
                f"Could not determine base/quote asset for {self.symbol}. Cannot calculate size.")
            return None
        if market_quote_asset != quote_asset:
            self.logger.warning(
                f"Configured quote_asset '{quote_asset}' differs from market quote '{market_quote_asset}'. Using market quote '{market_quote_asset}' for calculations.")
            quote_asset = market_quote_asset  # Use actual market quote

        # Use high precision context for calculations
        with getcontext() as ctx:
            ctx.prec = CALCULATION_PRECISION

            # Amount of quote currency to risk
            risk_amount_quote = available_equity * risk_percent
            quantized_equity = available_equity.quantize(
                Decimal('0.01'), rounding=ROUND_DOWN)
            quantized_risk_amount = risk_amount_quote.quantize(
                Decimal('0.01'), rounding=ROUND_DOWN)
            self.logger.info(
                f"Risk per trade: {risk_percent:.2%}, Equity ({quote_asset}): {quantized_equity}, Risk Amount: {quantized_risk_amount} {quote_asset}")

            # Price distance for stop loss (absolute value)
            stop_loss_distance = abs(entry_price - stop_loss_price)
            if stop_loss_distance <= Decimal(0):
                self.logger.error(
                    f"Stop loss distance is zero or negative ({stop_loss_distance}). Check SL price relative to entry.")
                return None

            # --- Calculate position size in base currency ---
            position_size_base = None
            # spot, swap (linear/inverse determined next)
            contract_type = market.get('type', 'spot')
            is_inverse = market.get('inverse', False)
            # Assume linear if swap and not inverse
            is_linear = market.get(
                'linear', True) if contract_type == 'swap' and not is_inverse else False

            # Get contract size (should be Decimal from get_market)
            # Default to 1 for spot or if missing/invalid for derivatives (error prone, should be caught by get_market)
            contract_size = market.get('contractSize', Decimal('1.0'))
            if contract_size is None or not contract_size.is_finite() or contract_size <= 0:
                # This case should ideally be prevented by get_market ensuring contractSize exists and is valid Decimal
                self.logger.error(
                    f"Invalid or missing contract size ({contract_size}) for {self.symbol}. Cannot calculate size.")
                return None

            if is_inverse:
                # Inverse: Size_Base = Risk_Quote * Entry / (ContractSize * Stop_Distance)
                position_size_base = (
                    risk_amount_quote * entry_price) / (contract_size * stop_loss_distance)
                self.logger.debug(
                    f"Calculating size for INVERSE contract (ContractSize: {contract_size}).")
            else:  # Linear or Spot
                # Linear/Spot: Size_Base = Risk_Quote / (ContractSize * Stop_Distance)
                position_size_base = risk_amount_quote / \
                    (contract_size * stop_loss_distance)
                calc_type = 'LINEAR' if is_linear else 'SPOT'
                self.logger.debug(
                    f"Calculating size for {calc_type} contract (ContractSize: {contract_size}).")

            if position_size_base is None or not position_size_base.is_finite() or position_size_base <= 0:
                self.logger.error(
                    f"Calculated position size is invalid (zero, negative, or non-finite): {position_size_base}")
                return None

            # Log with more precision
            self.logger.debug(
                f"Calculated raw position size: {position_size_base:.{DECIMAL_DISPLAY_PRECISION+4}f} {base_asset}")

            # --- Apply Precision, Lot Size, Min/Max Limits ---
            try:
                limits = market.get('limits', {})
                min_amount = limits.get('amount', {}).get('min')
                max_amount = limits.get('amount', {}).get('max')

                min_amount_dec = Decimal(
                    str(min_amount)) if min_amount is not None else None
                max_amount_dec = Decimal(
                    str(max_amount)) if max_amount is not None else None

                # 1. Quantize the amount based on market step size (typically round down/truncate)
                # Use the helper function for Decimal-based quantization
                quantized_size = self.quantize_value(
                    position_size_base, 'amount', market, rounding_mode=ROUND_DOWN)
                if quantized_size is None:
                    self.logger.error(
                        "Failed to quantize calculated position size based on step size.")
                    return None
                if quantized_size <= 0:
                    self.logger.warning(
                        f"Calculated size {position_size_base} became zero or negative ({quantized_size}) after quantization/rounding down. Risk amount might be too small for minimum step size.")
                    return None
                self.logger.debug(
                    f"Position size after step-size quantization (ROUND_DOWN): {quantized_size} {base_asset}")

                # 2. Check against minimum amount AFTER quantization
                if min_amount_dec is not None and quantized_size < min_amount_dec:
                    self.logger.warning(
                        f"Quantized position size {quantized_size} {base_asset} is below minimum order size {min_amount_dec} {base_asset}. Cannot open position with this risk setup.")
                    return None

                # 3. Check against maximum amount
                final_size_base = quantized_size  # Start with quantized size
                if max_amount_dec is not None and final_size_base > max_amount_dec:
                    self.logger.warning(
                        f"Quantized position size {final_size_base} exceeds max limit {max_amount_dec}. Capping size to max limit.")
                    # Cap the size to the maximum allowed, then re-quantize it to ensure it adheres to step size
                    capped_size = max_amount_dec
                    final_size_base = self.quantize_value(
                        capped_size, 'amount', market, rounding_mode=ROUND_DOWN)
                    if final_size_base is None or final_size_base <= 0:
                        self.logger.error(
                            "Failed to quantize the capped maximum position size or result was non-positive. Cannot proceed.")
                        return None
                    self.logger.info(
                        f"Position size capped and re-quantized to: {final_size_base} {base_asset}")

                # 4. Final check: Ensure size is still positive
                if final_size_base <= 0:
                    self.logger.error(
                        f"Final position size is zero or negative ({final_size_base}) after adjustments. Cannot open position.")
                    return None

                self.logger.info(
                    f"Calculated Final Position Size: {final_size_base} {base_asset}")
                return final_size_base

            except ValueError as e:
                self.logger.error(
                    f"Error applying market limits/precision to position size: {e}")
                return None
            except Exception as e:
                self.logger.error(
                    f"Unexpected error during position size finalization: {e}", exc_info=True)
                return None

    def quantize_price(self, price: Decimal, side_for_conservative_rounding: Optional[PositionSide] = None) -> Optional[Decimal]:
        """
        Quantizes price according to market precision rules (tick size) using Decimal math.
        Optionally applies conservative rounding for SL/TP based on side.
        """
        if not isinstance(price, Decimal):
            self.logger.error(
                f"Invalid price type for quantization: {type(price)}. Expected Decimal.")
            return None
        if not price.is_finite():
            self.logger.error(
                f"Invalid price value for quantization: {price}. Must be finite.")
            return None

        market = self.exchange.get_market(self.symbol)
        if not market:
            self.logger.error(
                f"Cannot quantize price: Market data not found for {self.symbol}.")
            return None

        # Determine rounding mode
        rounding_mode = ROUND_HALF_UP  # Default
        if side_for_conservative_rounding is not None:
            # Conservative rounding for SL: Round down for Long, Round up for Short
            # Conservative rounding for TP: Round up for Long, Round down for Short
            # Assuming 'price' here is typically an SL based on usage context
            # Adjust if this function is used for TPs as well, maybe pass intent ('sl' or 'tp')
            if side_for_conservative_rounding == PositionSide.LONG:
                rounding_mode = ROUND_DOWN  # Round Long SL down
            elif side_for_conservative_rounding == PositionSide.SHORT:
                # Round Short SL up (ROUND_CEILING might also work)
                rounding_mode = ROUND_UP

        # Use the Decimal quantization helper
        quantized_price = self.quantize_value(
            price, 'price', market, rounding_mode=rounding_mode)

        if quantized_price is None:
            # Error logged by quantize_value
            return None

        if side_for_conservative_rounding:
            self.logger.debug(
                f"Quantized price {price} to {quantized_price} using {rounding_mode} for {side_for_conservative_rounding.name} side.")
        else:
            self.logger.debug(
                f"Quantized price {price} to {quantized_price} using default rounding.")

        return quantized_price

    def calculate_stop_loss(self, entry_price: Decimal, side: PositionSide, latest_indicators: pd.Series) -> Optional[Decimal]:
        """
        Calculates the initial stop loss price based on configured method (ATR or fixed %).
        Uses high precision Decimal math and quantizes the result.
        """
        if not isinstance(entry_price, Decimal) or not entry_price.is_finite() or entry_price <= 0:
            self.logger.error(
                f"Invalid entry price for SL calculation: {entry_price}")
            return None

        sl_method = self.risk_config.get("stop_loss_method", "atr").lower()
        stop_loss_price_raw = None

        with getcontext() as ctx:
            ctx.prec = CALCULATION_PRECISION  # High precision for calculations

            if sl_method == "atr":
                atr_multiplier = self.risk_config.get(
                    "atr_multiplier", Decimal("1.5"))
                # Validation done in config load

                atr_period = self.indicator_config.get("atr_period", 14)
                atr_col = f"ATR_{atr_period}" if isinstance(
                    atr_period, int) else None

                if not atr_col:
                    self.logger.error(
                        "ATR period not configured correctly or is not an integer.")
                    return None
                if atr_col not in latest_indicators or pd.isna(latest_indicators[atr_col]):
                    self.logger.error(
                        f"ATR column '{atr_col}' not found or is NaN in latest indicators.")
                    return None

                # Should be Decimal from indicator calculation
                atr_value = latest_indicators[atr_col]
                if not isinstance(atr_value, Decimal) or not atr_value.is_finite() or atr_value <= 0:
                    self.logger.error(
                        f"Cannot calculate ATR stop loss: Invalid or non-positive ATR value ({atr_value}).")
                    return None

                stop_distance = atr_value * atr_multiplier
                if side == PositionSide.LONG:
                    stop_loss_price_raw = entry_price - stop_distance
                elif side == PositionSide.SHORT:
                    stop_loss_price_raw = entry_price + stop_distance
                else:
                    self.logger.error(
                        "Invalid position side for SL calculation.")
                    return None
                self.logger.debug(
                    f"Calculated SL based on ATR: Entry={entry_price}, Side={side.name}, ATR={atr_value}, Multiplier={atr_multiplier}, Distance={stop_distance}, Raw SL={stop_loss_price_raw}")

            elif sl_method == "fixed_percent":
                fixed_percent_config = self.risk_config.get(
                    "fixed_stop_loss_percent", Decimal("2.0"))
                # Validation done in config load
                fixed_percent = fixed_percent_config / Decimal("100")

                if side == PositionSide.LONG:
                    stop_loss_price_raw = entry_price * \
                        (Decimal("1") - fixed_percent)
                elif side == PositionSide.SHORT:
                    stop_loss_price_raw = entry_price * \
                        (Decimal("1") + fixed_percent)
                else:
                    self.logger.error(
                        "Invalid position side for SL calculation.")
                    return None
                self.logger.debug(
                    f"Calculated SL based on Fixed Percent: Entry={entry_price}, Side={side.name}, Percent={fixed_percent*100}%, Raw SL={stop_loss_price_raw}")

            else:
                self.logger.error(
                    f"Unknown stop loss method configured: {sl_method}")
                return None

        # --- Validation and Quantization ---
        if stop_loss_price_raw is None or not stop_loss_price_raw.is_finite() or stop_loss_price_raw <= 0:
            self.logger.error(
                f"Calculated raw stop loss price ({stop_loss_price_raw}) is invalid. Cannot set SL.")
            return None

        # Quantize SL price conservatively (down for long, up for short)
        quantized_sl = self.quantize_price(
            stop_loss_price_raw, side_for_conservative_rounding=side)

        if quantized_sl is None:
            self.logger.error("Failed to quantize calculated stop loss price.")
            return None

        # Final check: Ensure quantized SL didn't cross entry price
        # Quantize entry price for fair comparison
        quantized_entry = self.quantize_price(entry_price)
        if quantized_entry is None:
            self.logger.warning(
                "Could not quantize entry price for final SL validation. Skipping check.")
        else:
            if side == PositionSide.LONG and quantized_sl >= quantized_entry:
                self.logger.error(
                    f"Quantized SL {quantized_sl} is >= quantized entry price {quantized_entry}. Invalid SL. Check ATR/percentage values or market tick size.")
                # Option: Adjust SL slightly further away by one tick? Risky. Fail for now.
                # market = self.exchange.get_market(self.symbol)
                # tick_size = self.exchange.quantize_value(Decimal('0'), 'price', market) # Get tick size? Complex.
                return None
            if side == PositionSide.SHORT and quantized_sl <= quantized_entry:
                self.logger.error(
                    f"Quantized SL {quantized_sl} is <= quantized entry price {quantized_entry}. Invalid SL. Check ATR/percentage values or market tick size.")
                return None

        self.logger.info(
            f"Calculated Initial Stop Loss Price (Quantized): {quantized_sl}")
        return quantized_sl

    def check_ma_cross_exit(self, indicators_df: pd.DataFrame, position_side: PositionSide) -> bool:
        """Checks if an MA cross exit condition is met based on config and previous candle."""
        if not self.trading_config.get("use_ma_cross_exit", False):
            return False

        ema_short_period = self.indicator_config.get("ema_short_period")
        ema_long_period = self.indicator_config.get("ema_long_period")
        ema_short_col = f"EMA_{ema_short_period}" if isinstance(
            ema_short_period, int) else None
        ema_long_col = f"EMA_{ema_long_period}" if isinstance(
            ema_long_period, int) else None

        if not ema_short_col or not ema_long_col:
            self.logger.warning(
                "MA cross exit enabled, but EMA periods not configured correctly.")
            return False
        if len(indicators_df) < 2:
            self.logger.debug(
                "Not enough data points (< 2) to check for MA cross.")
            return False

        try:
            latest_data = indicators_df.iloc[-1]
            prev_data = indicators_df.iloc[-2]
        except IndexError:
            self.logger.warning(
                "Could not get latest and previous indicator data for MA cross check.")
            return False

        if not all(c in latest_data and c in prev_data for c in [ema_short_col, ema_long_col]):
            self.logger.warning(
                f"Cannot check MA cross exit: EMA columns missing from latest/previous indicators.")
            return False

        # Get current and previous EMA values
        ema_short_now = latest_data[ema_short_col]
        ema_long_now = latest_data[ema_long_col]
        ema_short_prev = prev_data[ema_short_col]
        ema_long_prev = prev_data[ema_long_col]

        # Check if all EMA values are valid finite Decimals
        valid_emas = all(isinstance(e, Decimal) and e.is_finite() for e in [
                         ema_short_now, ema_long_now, ema_short_prev, ema_long_prev])
        if not valid_emas:
            self.logger.warning(
                f"Invalid or non-finite EMA values for MA cross exit check.")
            return False

        # Check for actual cross:
        # Bearish cross (exit long): Short was >= Long previously, and Short is < Long now.
        # Bullish cross (exit short): Short was <= Long previously, and Short is > Long now.
        exit_signal = False
        if position_side == PositionSide.LONG:
            if ema_short_prev >= ema_long_prev and ema_short_now < ema_long_now:
                self.logger.info(
                    "MA Cross Exit triggered for LONG position (Short EMA crossed below Long EMA).")
                exit_signal = True
        elif position_side == PositionSide.SHORT:
            if ema_short_prev <= ema_long_prev and ema_short_now > ema_long_now:
                self.logger.info(
                    "MA Cross Exit triggered for SHORT position (Short EMA crossed above Long EMA).")
                exit_signal = True

        return exit_signal

    def manage_stop_loss(self, position: Dict[str, Any], latest_indicators: pd.Series, current_state: Dict[str, Any]) -> Optional[Decimal]:
        """
        Manages stop loss adjustments (Break-Even, Trailing).
        Returns the new SL price (Decimal) if an update is needed and valid, otherwise None.
        Requires 'active_position' and 'stop_loss_price' in current_state.
        Uses Decimal types for calculations and comparisons.
        """
        if not position or position.get('side') == 'none' or position.get('side') is None:
            self.logger.debug(
                "No active position or side is None, cannot manage SL.")
            return None
        if not current_state.get('active_position'):
            self.logger.warning(
                "manage_stop_loss called but state has no active position. Sync issue?")
            return None

        # Extract necessary data, ensuring they are Decimals and finite
        entry_price = position.get('entryPrice')
        current_sl_state = current_state.get('stop_loss_price')
        position_side_str = position.get('side')
        mark_price = position.get('markPrice')

        if not isinstance(entry_price, Decimal) or not entry_price.is_finite():
            self.logger.warning(
                f"Invalid entry price ({entry_price}) for SL management.")
            return None
        if not isinstance(current_sl_state, Decimal) or not current_sl_state.is_finite():
            self.logger.warning(
                f"Invalid current SL price in state ({current_sl_state}) for SL management.")
            return None
        if not isinstance(mark_price, Decimal) or not mark_price.is_finite():
            self.logger.warning(
                f"Invalid mark price ({mark_price}) for SL management.")
            return None
        if entry_price <= 0 or mark_price <= 0 or current_sl_state <= 0:
            self.logger.warning(
                f"Entry ({entry_price}), mark ({mark_price}), or SL ({current_sl_state}) is non-positive. Cannot manage SL.")
            return None

        try:
            position_side = PositionSide(position_side_str)
        except ValueError:
            self.logger.error(f"Invalid side string '{position_side_str}'.")
            return None

        new_sl_price_proposal = None  # Stores the proposed new SL price before validation
        proposed_sl_type = ""  # 'BE' or 'TSL'
        state_updated_this_cycle = False  # Track if BE was proposed this cycle

        # Use high precision context
        with getcontext() as ctx:
            ctx.prec = CALCULATION_PRECISION

            # --- Get ATR value ---
            atr_value = None
            atr_period = self.indicator_config.get("atr_period", 14)
            atr_col = f"ATR_{atr_period}" if isinstance(
                atr_period, int) else None
            if atr_col and atr_col in latest_indicators and pd.notna(latest_indicators[atr_col]):
                val = latest_indicators[atr_col]
                if isinstance(val, Decimal) and val.is_finite() and val > 0:
                    atr_value = val
                else:
                    self.logger.warning(
                        f"ATR value ({val}) is invalid/non-positive.")
            else:
                self.logger.warning(f"ATR column '{atr_col}' not found/NaN.")

            # 1. Break-Even Stop Loss
            use_be = self.risk_config.get("use_break_even_sl", False)
            if use_be and atr_value is not None and not current_state.get('break_even_achieved', False):
                be_trigger_atr = self.risk_config.get(
                    "break_even_trigger_atr", Decimal("1.0"))
                be_offset_atr = self.risk_config.get(
                    # Small profit offset
                    "break_even_offset_atr", Decimal("0.1"))
                # Config validation ensures these are valid Decimals

                profit_target_distance = atr_value * be_trigger_atr
                offset_distance = atr_value * be_offset_atr
                target_be_price_raw = entry_price

                if position_side == PositionSide.LONG:
                    target_be_price_raw = entry_price + offset_distance
                elif position_side == PositionSide.SHORT:
                    target_be_price_raw = entry_price - offset_distance

                # Quantize the target BE price (conservatively: down for long, up for short target SL)
                quantized_be_price = self.quantize_price(
                    target_be_price_raw, side_for_conservative_rounding=position_side)

                if quantized_be_price is not None:
                    # Check if price reached the trigger level
                    be_triggered = False
                    if position_side == PositionSide.LONG and mark_price >= (entry_price + profit_target_distance):
                        be_triggered = True
                    elif position_side == PositionSide.SHORT and mark_price <= (entry_price - profit_target_distance):
                        be_triggered = True

                    if be_triggered:
                        # Check if the proposed BE price is better (higher for long, lower for short) than the current SL
                        is_better = False
                        if position_side == PositionSide.LONG and quantized_be_price > current_sl_state:
                            is_better = True
                        elif position_side == PositionSide.SHORT and quantized_be_price < current_sl_state:
                            is_better = True

                        if is_better:
                            self.logger.info(
                                f"Break-Even Triggered ({position_side.name}): Mark Price {mark_price} reached target. Proposing SL update from {current_sl_state} to BE price {quantized_be_price}")
                            new_sl_price_proposal = quantized_be_price
                            proposed_sl_type = "BE"
                            state_updated_this_cycle = True  # Prevent TSL override this cycle
                        else:
                            self.logger.debug(
                                f"BE triggered, but proposed BE price {quantized_be_price} not better than current SL {current_sl_state}.")
                else:
                    self.logger.warning(
                        "Failed to quantize break-even price. Skipping BE check.")

            # 2. Trailing Stop Loss (only if BE wasn't just proposed)
            use_tsl = self.risk_config.get("use_trailing_sl", False)
            if use_tsl and atr_value is not None and not state_updated_this_cycle:
                tsl_atr_multiplier = self.risk_config.get(
                    "trailing_sl_atr_multiplier", Decimal("2.0"))
                # Config validation ensures valid Decimal

                trail_distance = atr_value * tsl_atr_multiplier
                potential_tsl_price_raw = None
                if position_side == PositionSide.LONG:
                    potential_tsl_price_raw = mark_price - trail_distance
                elif position_side == PositionSide.SHORT:
                    potential_tsl_price_raw = mark_price + trail_distance

                if potential_tsl_price_raw is not None:
                    # Quantize potential TSL price (conservatively)
                    quantized_tsl_price = self.quantize_price(
                        potential_tsl_price_raw, side_for_conservative_rounding=position_side)

                    if quantized_tsl_price is not None:
                        # Check if the proposed TSL price is better than the current SL
                        is_better = False
                        if position_side == PositionSide.LONG and quantized_tsl_price > current_sl_state:
                            is_better = True
                        elif position_side == PositionSide.SHORT and quantized_tsl_price < current_sl_state:
                            is_better = True

                        if is_better:
                            self.logger.debug(
                                f"Trailing SL Update ({position_side.name}): Potential TSL {quantized_tsl_price} is better than Current SL {current_sl_state}")

                            # Prevent TSL from moving SL unfavorably after BE achieved (clamp to entry)
                            if current_state.get('break_even_achieved', False):
                                quantized_entry = self.quantize_price(
                                    entry_price)
                                if quantized_entry is not None:
                                    if position_side == PositionSide.LONG and quantized_tsl_price < quantized_entry:
                                        self.logger.warning(
                                            f"TSL {quantized_tsl_price} below entry {quantized_entry} after BE. Clamping SL to entry.")
                                        quantized_tsl_price = quantized_entry
                                    elif position_side == PositionSide.SHORT and quantized_tsl_price > quantized_entry:
                                        self.logger.warning(
                                            f"TSL {quantized_tsl_price} above entry {quantized_entry} after BE. Clamping SL to entry.")
                                        quantized_tsl_price = quantized_entry
                                else:
                                    self.logger.warning(
                                        "Could not quantize entry for TSL vs BE check.")

                            # Propose the potentially clamped TSL price
                            new_sl_price_proposal = quantized_tsl_price
                            proposed_sl_type = "TSL"
                        # else: TSL is not better, do nothing
                    else:
                        self.logger.warning(
                            f"Failed to quantize potential TSL price ({potential_tsl_price_raw}). Skipping TSL update.")

        # --- Final Validation and Return ---
        if new_sl_price_proposal is not None:
            # Validate proposal is finite positive
            if not new_sl_price_proposal.is_finite() or new_sl_price_proposal <= 0:
                self.logger.warning(
                    f"Proposed new SL ({new_sl_price_proposal}) is invalid. Ignoring update.")
                return None

            # Validate against current mark price
            # Allow SL slightly crossing mark price due to quantization/timing, but not significantly far
            # Use a tolerance (e.g., half a tick size?) or just check strict inequality? Strict is safer.
            price_tick_size = None
            market = self.exchange.get_market(self.symbol)
            if market and 'precision' in market and 'price' in market['precision']:
                tick_str = market['precision']['price']
                if tick_str:
                    try:
                        price_tick_size = Decimal(str(tick_str))
                        except:
                            pass

            tolerance = (price_tick_size / Decimal(2)
                         ) if price_tick_size else Decimal('1e-9')

            sl_invalid = False
            if position_side == PositionSide.LONG and new_sl_price_proposal > (mark_price + tolerance):
                self.logger.warning(
                    f"Proposed new SL {new_sl_price_proposal} is significantly > current mark price {mark_price}. Invalid SL.")
                sl_invalid = True
            if position_side == PositionSide.SHORT and new_sl_price_proposal < (mark_price - tolerance):
                self.logger.warning(
                    f"Proposed new SL {new_sl_price_proposal} is significantly < current mark price {mark_price}. Invalid SL.")
                sl_invalid = True

            if sl_invalid:
                return None

            # Only return if the new SL is meaningfully different from the current one (use tolerance)
            if abs(new_sl_price_proposal - current_sl_state) > tolerance:
                self.logger.info(
                    f"Proposing {proposed_sl_type} SL update from {current_sl_state} to {new_sl_price_proposal}")
                return new_sl_price_proposal
            else:
                self.logger.debug(
                    f"Calculated new {proposed_sl_type} SL {new_sl_price_proposal} not significantly different from current {current_sl_state}. No update needed.")
                return None  # No significant change

        return None  # No update needed


# --- Main Trading Bot Class ---

class TradingBot:
    """
    The main trading bot class orchestrating the fetch-analyze-execute loop.
    Manages state, interacts with exchange wrapper, analyzer, and position manager.
    """

    def __init__(self, config_path: Path, state_path: Path):
        self.config_path = config_path
        self.state_path = state_path
        # Setup logger first, potentially updating level later from config
        self.logger = setup_logger()

        self.config = load_config(config_path, self.logger)
        if not self.config:
            self.logger.critical(
                "Failed to load or validate configuration. Exiting.")
            sys.exit(1)

        # Apply log level from config
        self._apply_log_level_from_config()

        self.state = load_state(state_path, self.logger)
        self._initialize_default_state()

        try:
            self.exchange = BybitV5Wrapper(self.config, self.logger)
            self.analyzer = TradingAnalyzer(self.config, self.logger)
            self.position_manager = PositionManager(
                self.config, self.logger, self.exchange)
        except Exception as e:
            self.logger.critical(
                f"Failed to initialize core components: {e}. Exiting.", exc_info=True)
            sys.exit(1)

        # Extract frequently used config values
        self._load_trading_parameters()

        self.is_running = True  # Flag to control the main loop

    def _apply_log_level_from_config(self):
        """Sets the logger level based on the loaded configuration."""
        global LOG_LEVEL
        log_level_str = self.config.get(
            "logging", {}).get("level", "INFO").upper()
        log_level_enum = getattr(logging, log_level_str, None)
        if isinstance(log_level_enum, int):
            if self.logger.level != log_level_enum:
                LOG_LEVEL = log_level_enum
                self.logger.setLevel(LOG_LEVEL)
                for handler in self.logger.handlers:
                    handler.setLevel(LOG_LEVEL)
                self.logger.info(
                    f"Log level set to {log_level_str} ({LOG_LEVEL}) from config.")
        else:
            self.logger.warning(
                f"Invalid log level '{log_level_str}' in config. Using default {logging.getLevelName(self.logger.level)}.")

    def _initialize_default_state(self):
        """Ensures essential state keys exist with default values."""
        self.state.setdefault(
            'active_position', None)  # Stores dict of the current position
        # Stores Decimal SL price
        self.state.setdefault('stop_loss_price', None)
        # Stores Decimal TP price
        self.state.setdefault('take_profit_price', None)
        # Flag for BE SL activation
        self.state.setdefault('break_even_achieved', False)
        # ID of the last order placed
        self.state.setdefault('last_order_id', None)
        # Timestamp of last successful state sync
        self.state.setdefault('last_sync_time', None)

    def _load_trading_parameters(self):
        """Loads and validates essential trading parameters from config."""
        settings = self.config.get('trading_settings', {})
        self.symbol = settings.get('symbol')
        self.timeframe = settings.get('timeframe')
        self.leverage = settings.get('leverage')  # Should be Decimal
        self.quote_asset = settings.get('quote_asset')
        self.category = settings.get('category')  # linear/inverse/spot
        self.poll_interval = settings.get('poll_interval_seconds', 60)
        self.hedge_mode = settings.get('hedge_mode', False)
        self.exit_on_opposing_signal = settings.get(
            'exit_on_opposing_signal', True)
        self.use_ma_cross_exit = settings.get('use_ma_cross_exit', False)
        self.post_order_verify_delay = settings.get(
            'post_order_verify_delay_seconds', 5)

        # Validate essential settings (already checked in validate_config, but double-check here)
        if not all([self.symbol, self.timeframe, self.leverage, self.quote_asset, self.category]):
            self.logger.critical(
                "Essential trading settings missing after loading config. Exiting.")
            sys.exit(1)

    def run(self):
        """Starts the main trading loop."""
        self.logger.info(f"--- Starting Trading Bot ---")
        self.logger.info(
            f"Symbol: {self.symbol}, Timeframe: {self.timeframe}, Category: {self.category}, Quote: {self.quote_asset}")
        self.logger.info(
            f"Leverage: {self.leverage}x, Hedge Mode: {self.hedge_mode}, Poll Interval: {self.poll_interval}s")

        if not self.initialize_exchange_settings():
            self.logger.critical(
                "Failed to initialize exchange settings (e.g., leverage/mode). Exiting.")
            sys.exit(1)

        self.logger.info("Performing initial position sync...")
        initial_position = self.get_current_position()
        self.sync_bot_state_with_position(initial_position)
        # Save potentially updated initial state
        save_state(self.state, self.state_path, self.logger)

        while self.is_running:
            try:
                self.logger.info(
                    f"--- New Trading Cycle ({pd.Timestamp.utcnow()}) ---")
                start_time = time.time()

                # 1. Fetch Market Data (OHLCV)
                ohlcv_limit = self.config.get(
                    "indicator_settings", {}).get("ohlcv_fetch_limit", 250)
                ohlcv_data = self.exchange.fetch_ohlcv(
                    self.symbol, self.timeframe, limit=ohlcv_limit)
                if ohlcv_data is None or ohlcv_data.empty:
                    self.logger.warning(
                        "Failed to fetch OHLCV data. Skipping cycle.")
                    self._wait_for_next_cycle(start_time)
                    continue

                # 2. Calculate Indicators & Generate Signal
                indicators_df = self.analyzer.calculate_indicators(ohlcv_data)
                if indicators_df is None or indicators_df.empty:
                    self.logger.warning(
                        "Failed to calculate indicators. Skipping cycle.")
                    self._wait_for_next_cycle(start_time)
                    continue
                try:
                    # Check staleness
                    if indicators_df.iloc[-1].name < (pd.Timestamp.utcnow() - pd.Timedelta(minutes=15)):
                        self.logger.warning(
                            f"Latest indicator data timestamp ({indicators_df.iloc[-1].name}) seems stale. Skipping cycle.")
                        self._wait_for_next_cycle(start_time)
                        continue
                    # Series of latest indicators
                    latest_indicators = indicators_df.iloc[-1]
                except IndexError:
                    self.logger.warning(
                        "Indicators DataFrame empty after calculation. Skipping cycle.")
                    self._wait_for_next_cycle(start_time)
                    continue

                signal, signal_details = self.analyzer.generate_signal(
                    indicators_df)

                # 3. Fetch Current State (Position) - Fetch again for latest data
                current_position = self.get_current_position()  # Fetches live data from exchange

                # 4. Sync Bot State with Exchange State - CRITICAL STEP
                self.sync_bot_state_with_position(current_position)

                # 5. Decision Making & Execution
                # Use the *synced state* ('active_position') to decide
                if self.state.get('active_position'):
                    # Pass full indicators_df for potential multi-candle checks (like MA cross)
                    # Pass live position data (current_position) for accurate SL management
                    self.manage_existing_position(
                        indicators_df, signal, current_position)
                else:
                    # No active position according to state, check for entry signals
                    # Pass latest indicators Series
                    self.attempt_new_entry(
                        signal, latest_indicators, signal_details)

                # 6. Save State (after potential actions)
                save_state(self.state, self.state_path, self.logger)

                # 7. Wait for Next Cycle
                self._wait_for_next_cycle(start_time)

            except KeyboardInterrupt:
                self.logger.info(
                    "KeyboardInterrupt received. Stopping bot gracefully...")
                self.is_running = False
                # self.perform_shutdown_cleanup() # Optional: close positions, cancel orders
            except ccxt.AuthenticationError:
                self.logger.critical(
                    "Authentication failed during main loop. Stopping bot.", exc_info=True)
                self.is_running = False
            except ccxt.NetworkError as e:
                self.logger.error(
                    f"Network error in main loop: {e}. Bot will retry after poll interval.")
                # Use standard wait after network errors
                time.sleep(self.poll_interval)
            except Exception as e:
                self.logger.critical(
                    f"An critical unexpected error occurred in the main loop: {e}", exc_info=True)
                # Decide whether to stop or continue based on severity
                self.logger.info(
                    "Attempting to continue after unexpected error, but manual check advised.")
                # Wait longer before next cycle after unexpected error
                time.sleep(self.poll_interval * 2)

        self.logger.info("--- Trading Bot stopped ---")

    def initialize_exchange_settings(self) -> bool:
        """Set initial exchange settings like leverage and position mode."""
        self.logger.info("Initializing exchange settings...")
        success = True

        # 1. Set Leverage (required for futures/swaps)
        if self.category in ['linear', 'inverse']:
            if not self.exchange.set_leverage(self.symbol, self.leverage):
                self.logger.error(
                    f"Failed to set initial leverage to {self.leverage}x for {self.symbol}.")
                success = False  # Mark as failure but allow continuing if possible
            else:
                self.logger.info(
                    f"Leverage set request to {self.leverage}x for {self.symbol} sent.")
        else:
            self.logger.info("Skipping leverage setting (spot category).")

        # 2. Set Hedge Mode vs One-Way Mode (Bybit specific)
        if self.exchange.exchange_id == 'bybit' and self.category in ['linear', 'inverse']:
            try:
                target_mode = 3 if self.hedge_mode else 0  # 0: One-way, 3: Hedge Mode
                mode_name = 'Hedge' if self.hedge_mode else 'One-way'
                self.logger.info(
                    f"Checking/Setting position mode for {self.symbol} (Target: {mode_name})...")

                # Bybit V5 endpoint: POST /v5/position/switch-mode
                # Needs category, symbol, mode
                # category added by safe_ccxt_call
                params = {'symbol': self.symbol, 'mode': target_mode}

                # This endpoint might not be directly exposed in CCXT, use private call name if known
                # Check common CCXT naming conventions or source if needed
                # Assuming 'private_post_position_switch_mode' based on common patterns
                method_name = 'private_post_position_switch_mode'

                if hasattr(self.exchange.exchange, method_name):
                    result = self.exchange.safe_ccxt_call(
                        method_name, params=params)
                    if result and result.get('retCode') == 0:
                        self.logger.info(
                            f"Position mode successfully set to {mode_name}.")
                    # 110025: Position mode is not modified
                    elif result and result.get('retCode') == 110025:
                        self.logger.info(
                            f"Position mode is already set correctly ({mode_name}).")
                    elif result:  # Call succeeded but Bybit returned another error
                        self.logger.error(
                            f"Failed to set position mode on Bybit. Code: {result.get('retCode')}, Msg: {result.get('retMsg', 'Unknown error')}")
                        success = False
                    else:  # safe_ccxt_call failed (network, auth, etc.)
                        self.logger.error(
                            "API call failed when trying to set position mode.")
                        success = False
                else:
                    self.logger.warning(
                        f"CCXT method '{method_name}' not found. Cannot verify/set position mode automatically.")
                    # Consider this non-critical? Or set success=False? Depends on requirements.
                    # Let's assume non-critical for now, but log warning prominently.

            except Exception as e:
                self.logger.error(
                    f"Error setting position mode: {e}", exc_info=True)
                success = False
        else:
            self.logger.debug(
                "Skipping position mode setting (not Bybit derivatives).")

        # 3. Set Margin Mode (Optional, Example: ISOLATED)
        # Note: Requires no open positions/orders usually. Be careful enabling this.
        # margin_mode_config = self.config['trading_settings'].get('margin_mode', 'isolated').lower()
        # if self.category in ['linear', 'inverse'] and margin_mode_config in ['isolated', 'cross']:
        #     try:
        #         # Bybit V5 uses private_post_position_switch_margin_mode
        #         # tradeMode: 0 cross, 1 isolated
        #         trade_mode = 1 if margin_mode_config == 'isolated' else 0
        #         self.logger.info(f"Attempting to set margin mode to '{margin_mode_config}' for {self.symbol}...")
        #         params = {
        #             'symbol': self.symbol,
        #             'tradeMode': trade_mode,
        #             'buyLeverage': f"{int(self.leverage)}", # Required param for switch
        #             'sellLeverage': f"{int(self.leverage)}" # Required param for switch
        #         } # category added by safe_ccxt_call
        #         method_name = 'private_post_position_switch_margin_mode'
        #         if hasattr(self.exchange.exchange, method_name):
        #             result = self.exchange.safe_ccxt_call(method_name, params=params)
        #             # Check result['retCode'] == 0 for success, handle "not modified" etc.
        #             # ... (similar logic as hedge mode switching) ...
        #         else:
        #             self.logger.warning(f"CCXT method '{method_name}' not found. Cannot set margin mode.")
        #     except Exception as e:
        #         self.logger.error(f"Error setting margin mode: {e}", exc_info=True)
        #         success = False

        return success

    def get_current_position(self) -> Optional[Dict[str, Any]]:
        """
        Fetches and returns the single active position relevant to the bot for its symbol.
        Handles hedge mode by trying to match the position index stored in the bot's state.
        Returns the processed position dict (with Decimals) or None.
        """
        positions = self.exchange.fetch_positions(
            self.symbol)  # Uses wrapper method
        if positions is None:
            self.logger.warning(
                f"Could not fetch positions for {self.symbol}. Assuming no position.")
            return None

        # fetch_positions wrapper already filters for active size and processes fields.
        # Now, filter based on bot's logic (hedge vs one-way).
        # Already filtered by symbol in fetch_positions
        active_positions_for_symbol = positions

        if not active_positions_for_symbol:
            self.logger.info(
                f"No active position found on exchange for {self.symbol}.")
            return None

        if self.hedge_mode:
            state_pos = self.state.get('active_position')
            if state_pos:
                # Bot thinks it has a position, find the matching one by index and side
                target_idx = state_pos.get('position_idx')
                target_side_enum_val = state_pos.get(
                    'side', PositionSide.NONE).value  # Get 'long'/'short' string

                if target_idx is None or target_side_enum_val == PositionSide.NONE.value:
                    self.logger.error(
                        "Hedge mode active, but bot state is inconsistent (missing idx or side). Cannot identify position.")
                    # Clear inconsistent state? Or just return None? Return None for now.
                    # self._clear_position_state("Inconsistent hedge mode state")
                    return None

                found_match = None
                for p in active_positions_for_symbol:
                    p_idx = p.get('positionIdx')
                    # 'long'/'short' string from processed position
                    p_side_str = p.get('side')
                    if p_idx == target_idx and p_side_str == target_side_enum_val:
                        found_match = p
                        break
                    elif p_idx == target_idx:
                        self.logger.warning(
                            f"Hedge mode position found with matching Idx ({target_idx}) but mismatched Side (State: {target_side_enum_val}, Exchange: {p_side_str}). Ignoring.")

                if found_match:
                    self.logger.info(
                        f"Found active hedge position matching state: Side {found_match.get('side')}, Idx {found_match.get('positionIdx')}, Size {found_match.get('contracts')}")
                    return found_match
                else:
                    self.logger.warning(
                        f"Bot state indicates hedge position (Idx: {target_idx}, Side: {target_side_enum_val}), but no matching active position found. State may be stale.")
                    return None  # Let sync handle clearing state
            else:
                # Bot state has no position, but found active position(s) on exchange. Ignore them.
                pos_details = [
                    f"Idx:{p.get('positionIdx')}, Side:{p.get('side')}, Size:{p.get('contracts')}" for p in active_positions_for_symbol]
                self.logger.info(
                    f"Ignoring {len(active_positions_for_symbol)} active hedge position(s) found on exchange as bot state is empty: [{'; '.join(pos_details)}]")
                return None

        else:  # Non-hedge (One-way) mode
            if len(active_positions_for_symbol) > 1:
                self.logger.error(
                    f"CRITICAL: Found {len(active_positions_for_symbol)} active positions for {self.symbol} in non-hedge mode. Check exchange state manually!")
                # Decide handling: Return first? None? None is safer.
                return None
            # Return the single active position (fetch_positions ensures size > 0)
            pos = active_positions_for_symbol[0]
            # Check positionIdx is 0 as expected for one-way
            if pos.get('positionIdx') != 0:
                self.logger.warning(
                    f"One-way mode active, but found position with non-zero index ({pos.get('positionIdx')}). Check exchange mode setting.")
                # Still return the position, but log warning
            self.logger.info(
                f"Found active non-hedge position: Side {pos.get('side')}, Size {pos.get('contracts')}, Idx {pos.get('positionIdx')}")
            return pos

    def sync_bot_state_with_position(self, current_position_on_exchange: Optional[Dict[str, Any]]):
        """
        Updates the bot's internal state based on the fetched position from the exchange.
        Clears state if the position is gone or doesn't match expected state. Critical for consistency.
        """
        bot_state_position = self.state.get('active_position')
        bot_thinks_has_position = bot_state_position is not None
        state_changed = False  # Flag to check if save_state is needed

        if current_position_on_exchange:
            # --- Position exists on exchange ---
            exchange_pos_symbol = current_position_on_exchange.get('symbol')
            exchange_pos_side_str = current_position_on_exchange.get(
                'side')  # 'long'/'short'/'none'
            exchange_pos_size = current_position_on_exchange.get(
                'contracts')  # Decimal size
            exchange_pos_entry = current_position_on_exchange.get(
                'entryPrice')  # Decimal entry
            exchange_pos_idx = current_position_on_exchange.get(
                'positionIdx')  # Int index
            # Get SL from 'info' field, which should be Decimal from fetch_positions
            exchange_sl_dec = current_position_on_exchange.get(
                'info', {}).get('stopLoss')  # Decimal or NaN

            # Basic validation of essential data fetched
            if not all([exchange_pos_symbol, exchange_pos_side_str != PositionSide.NONE.value,
                        exchange_pos_size is not None, exchange_pos_entry is not None]):
                self.logger.error(
                    "Fetched exchange position data is incomplete or side is 'none'. Cannot sync state reliably.")
                if bot_thinks_has_position:
                    self.logger.warning(
                        "Clearing potentially stale bot state due to incomplete/invalid exchange data.")
                    self._clear_position_state(
                        "Incomplete/invalid exchange position data")
                    state_changed = True
                return

            # Validate SL value (should be finite positive Decimal or None/NaN)
            if exchange_sl_dec is not None and (not exchange_sl_dec.is_finite() or exchange_sl_dec <= 0):
                self.logger.debug(
                    f"Ignoring invalid SL value ({exchange_sl_dec}) from exchange position info.")
                exchange_sl_dec = None  # Treat invalid SL as no SL

            if not bot_thinks_has_position:
                # Bot thought no position, but found one. Adopt if matches mode.
                position_matches_mode = False
                if self.hedge_mode and exchange_pos_idx in [1, 2]:
                    position_matches_mode = True
                elif not self.hedge_mode and exchange_pos_idx == 0:
                    position_matches_mode = True

                if position_matches_mode:
                    self.logger.warning(
                        f"Found unexpected active position matching bot mode ({exchange_pos_side_str}, Idx:{exchange_pos_idx}). Adopting into state.")
                    try:
                        adopted_side = PositionSide(exchange_pos_side_str)
                    except ValueError:
                        self.logger.error(
                            f"Invalid side '{exchange_pos_side_str}'. Cannot adopt.")
                        return

                    self.state['active_position'] = {
                        'symbol': exchange_pos_symbol, 'side': adopted_side,
                        'entry_price': exchange_pos_entry, 'size': exchange_pos_size,
                        'position_idx': exchange_pos_idx, 'order_id': None,  # Unknown entry order
                    }
                    # Adopt SL if found
                    self.state['stop_loss_price'] = exchange_sl_dec
                    # Assume no TP initially
                    self.state['take_profit_price'] = None
                    self.state['break_even_achieved'] = False  # Reset BE
                    state_changed = True
                    if exchange_sl_dec:
                        self.logger.info(
                            f"Adopted SL {exchange_sl_dec} found on exchange.")
                    else:
                        self.logger.warning(
                            "No valid SL found on adopted exchange position.")
                else:
                    self.logger.warning(
                        f"Found active position ({exchange_pos_side_str}, Idx:{exchange_pos_idx}) NOT matching bot mode ({'Hedge' if self.hedge_mode else 'One-way'}). Ignoring.")

            else:
                # --- Bot knew about a position. Verify and update details. ---
                state_pos = self.state['active_position']
                state_side_enum = state_pos['side']
                state_idx = state_pos.get('position_idx')

                match = False
                if exchange_pos_symbol == state_pos['symbol'] and exchange_pos_side_str == state_side_enum.value:
                    if self.hedge_mode and exchange_pos_idx == state_idx:
                        match = True
                    elif not self.hedge_mode and exchange_pos_idx == 0:
                        match = True

                if match:
                    # Position matches state, update dynamic values
                    self.logger.debug(
                        f"Exchange position matches state. Syncing details...")
                    if state_pos['size'] != exchange_pos_size:
                        self.logger.info(
                            f"Position size changed: State={state_pos['size']}, Exchange={exchange_pos_size}. Updating state.")
                        state_pos['size'] = exchange_pos_size
                        state_changed = True
                    if state_pos['entry_price'] != exchange_pos_entry:
                        self.logger.info(
                            f"Position entry price changed: State={state_pos['entry_price']}, Exchange={exchange_pos_entry}. Updating state.")
                        state_pos['entry_price'] = exchange_pos_entry
                        state_changed = True

                    # --- Sync Stop Loss State ---
                    current_state_sl = self.state.get(
                        'stop_loss_price')  # Decimal or None
                    tolerance = Decimal('1e-9')  # Tolerance for comparison

                    if exchange_sl_dec is not None:  # Exchange reports an SL
                        sl_differs = current_state_sl is None or abs(
                            exchange_sl_dec - current_state_sl) > tolerance
                        if sl_differs:
                            self.logger.info(
                                f"Updating state SL from {current_state_sl} to match exchange SL {exchange_sl_dec}.")
                            self.state['stop_loss_price'] = exchange_sl_dec
                            state_changed = True
                            # Check if this update implies BE was achieved
                            self._check_and_update_be_status(
                                state_pos, exchange_sl_dec)  # May set state_changed=True
                    else:  # Exchange reports NO SL
                        if current_state_sl is not None:
                            self.logger.warning(
                                f"Bot state has SL {current_state_sl}, but no active SL found on exchange. Clearing state SL.")
                            self.state['stop_loss_price'] = None
                            state_changed = True
                            if self.state.get('break_even_achieved', False):
                                self.logger.info(
                                    "Resetting break_even_achieved flag as SL disappeared.")
                                # Already marked state_changed
                                self.state['break_even_achieved'] = False

                    # Update last sync time if state matched
                    # Don't mark state_changed for this timestamp update
                    self.state['last_sync_time'] = time.time()

                else:
                    # Position exists, but doesn't match state. Clear bot state.
                    self.logger.warning(
                        f"Found active position ({exchange_pos_symbol},{exchange_pos_side_str},Idx:{exchange_pos_idx}) NOT matching bot state ({state_pos.get('symbol')},{state_side_enum.value},Idx:{state_idx}). Clearing bot state.")
                    self._clear_position_state(
                        "Mismatch with exchange position")
                    state_changed = True

        else:
            # --- No position on exchange ---
            if bot_thinks_has_position:
                pos_details = f"({bot_state_position.get('side', PositionSide.NONE).name}, Size:{bot_state_position.get('size')})"
                self.logger.info(
                    f"Position {pos_details} no longer found on exchange. Clearing bot state.")
                self._clear_position_state(
                    "Position closed/missing on exchange")
                state_changed = True
            # else: Bot thought no position, exchange has no position. State is correct.

        # Save state if any relevant part changed
        if state_changed:
            save_state(self.state, self.state_path, self.logger)

    def _check_and_update_be_status(self, position_state: Dict[str, Any], current_sl: Decimal):
        """Checks if the current SL implies break-even is achieved and updates state. Internal helper."""
        if not self.state.get('break_even_achieved', False):  # Only update if not already achieved
            entry = position_state.get('entry_price')
            side = position_state.get('side')  # PositionSide Enum

            if entry is None or side is None or current_sl is None:
                return

            # Quantize entry for fair comparison
            quantized_entry = self.position_manager.quantize_price(entry)
            if quantized_entry is None:
                return

            be_achieved = False
            if side == PositionSide.LONG and current_sl >= quantized_entry:
                be_achieved = True
            elif side == PositionSide.SHORT and current_sl <= quantized_entry:
                be_achieved = True

            if be_achieved:
                self.logger.info(
                    f"Marking Break-Even as achieved based on SL {current_sl} vs Entry {quantized_entry}.")
                self.state['break_even_achieved'] = True
                # No need to return state_changed flag, sync_bot_state handles saving

    def _clear_position_state(self, reason: str):
        """Internal helper to safely clear all position-related state variables."""
        self.logger.info(f"Clearing position state. Reason: {reason}")
        self.state['active_position'] = None
        self.state['stop_loss_price'] = None
        self.state['take_profit_price'] = None
        self.state['break_even_achieved'] = False
        # Keep last_order_id? Useful for debugging closure. Let's keep it.
        # self.state['last_order_id'] = None

    def manage_existing_position(self, indicators_df: pd.DataFrame, signal: Signal, live_position_data: Optional[Dict[str, Any]]):
        """
        Manages SL, TP, and potential exits for the active position based on state and live data.
        Uses the already synced state. Updates state based on outcomes.
        """
        position_state = self.state.get('active_position')  # Use synced state
        if not position_state:
            self.logger.warning(
                "manage_existing_position called but state has no active position. Skipping.")
            return
        if not live_position_data:
            # Sync should ideally clear state if position disappears, but handle defensively
            self.logger.error(
                "manage_existing_position called without live position data. Sync likely failed or position closed unexpectedly. Skipping management.")
            return

        position_side = position_state['side']  # PositionSide Enum
        # Needed for hedge mode API calls
        position_idx = position_state.get('position_idx')
        self.logger.info(
            f"Managing existing {position_side.name} position (Idx: {position_idx if self.hedge_mode else 'N/A'})...")

        # --- Exit Checks ---
        # 1. MA Cross Exit
        if self.use_ma_cross_exit and self.position_manager.check_ma_cross_exit(indicators_df, position_side):
            self.logger.info("MA Cross exit condition met.")
            self.close_position("MA Cross Exit")
            return  # Exit management after closing

        # 2. Signal-Based Exit
        should_exit_on_signal = False
        if self.exit_on_opposing_signal:
            if position_side == PositionSide.LONG and signal in [Signal.SELL, Signal.STRONG_SELL]:
                should_exit_on_signal = True
            elif position_side == PositionSide.SHORT and signal in [Signal.BUY, Signal.STRONG_BUY]:
                should_exit_on_signal = True
        if should_exit_on_signal:
            self.logger.info(f"Opposing signal ({signal.name}) received.")
            self.close_position(f"Opposing Signal ({signal.name})")
            return  # Exit management after closing

        # --- Stop Loss Management (only if not exiting) ---
        current_sl_in_state = self.state.get('stop_loss_price')
        if current_sl_in_state is None:
            self.logger.warning(
                "Cannot manage SL: Stop loss price missing from state. Consider manual check or adding logic to set initial SL if missing.")
        else:
            # Pass live position data (already has Decimals) and current state to manager
            # Pass only latest row for SL management
            latest_indicators = indicators_df.iloc[-1]
            new_sl_price = self.position_manager.manage_stop_loss(
                live_position_data, latest_indicators, self.state)

            if new_sl_price is not None:
                # A new SL price (BE or TSL) was proposed. Attempt to set it.
                self.logger.info(
                    f"Attempting to update stop loss on exchange to: {new_sl_price}")
                protection_params = {'stop_loss': new_sl_price}
                if self.hedge_mode:
                    protection_params['position_idx'] = position_idx

                if self.exchange.set_protection(self.symbol, **protection_params):
                    self.logger.info(
                        f"Stop loss update request successful for {new_sl_price}.")
                    # Update state ONLY on successful request
                    self.state['stop_loss_price'] = new_sl_price
                    # Check if this update achieved break-even status (updates state internally)
                    self._check_and_update_be_status(
                        position_state, new_sl_price)
                    # Save state immediately
                    save_state(self.state, self.state_path, self.logger)
                else:
                    self.logger.error(
                        f"Failed to update stop loss on exchange to {new_sl_price}. API call failed or returned error. State SL remains {current_sl_in_state}.")
                    # Do NOT update state if API call fails

        self.logger.debug("Position management cycle finished.")

    def attempt_new_entry(self, signal: Signal, latest_indicators: pd.Series, signal_details: Dict[str, Any]):
        """
        Attempts to enter a new position based on the signal, calculates SL and size,
        places the order (potentially with SL attached), updates state, and verifies.
        """
        if self.state.get('active_position'):
            self.logger.debug(
                "Skipping entry attempt: Position already active.")
            return

        entry_signal = False
        target_side: Optional[PositionSide] = None
        order_side: Optional[OrderSide] = None

        if signal in [Signal.BUY, Signal.STRONG_BUY]:
            entry_signal = True
            target_side = PositionSide.LONG
            order_side = OrderSide.BUY
        elif signal in [Signal.SELL, Signal.STRONG_SELL]:
            entry_signal = True
            target_side = PositionSide.SHORT
            order_side = OrderSide.SELL

        if not entry_signal:
            self.logger.info(
                f"Signal is {signal.name}. No entry condition met.")
            return

        self.logger.info(
            f"Entry signal {signal.name} received. Preparing entry for {target_side.name}...")

        # --- Pre-computation and Validation ---
        # 1. Get Current Price Estimate (use last close or fetch ticker)
        entry_price_estimate = latest_indicators.get('close')
        if not isinstance(entry_price_estimate, Decimal) or not entry_price_estimate.is_finite() or entry_price_estimate <= 0:
            self.logger.warning(
                "Using ticker last price for entry estimate as close price is invalid.")
            ticker = self.exchange.safe_ccxt_call('fetch_ticker', self.symbol)
            if ticker and ticker.get('last'):
                try:
                    entry_price_estimate = Decimal(str(ticker['last']))
                except:
                    entry_price_estimate = None  # Reset if conversion fails
            if not isinstance(entry_price_estimate, Decimal) or not entry_price_estimate.is_finite() or entry_price_estimate <= 0:
                self.logger.error(
                    "Failed to get valid entry price estimate from close or ticker. Cannot enter.")
                return

        # 2. Calculate Stop Loss
        stop_loss_price = self.position_manager.calculate_stop_loss(
            entry_price_estimate, target_side, latest_indicators)
        if stop_loss_price is None:
            self.logger.error("Failed to calculate stop loss. Cannot enter.")
            return

        # 3. Calculate Position Size
        balance_info = self.exchange.fetch_balance()
        if not balance_info or self.quote_asset not in balance_info:
            self.logger.error(
                f"Failed to fetch balance or quote asset '{self.quote_asset}' missing. Cannot size position.")
            return
        available_equity = balance_info[self.quote_asset]
        if not isinstance(available_equity, Decimal) or not available_equity.is_finite() or available_equity <= 0:
            self.logger.error(
                f"Invalid available equity {available_equity}. Cannot size position.")
            return

        position_size = self.position_manager.calculate_position_size(
            entry_price_estimate, stop_loss_price, available_equity, self.quote_asset
        )
        if position_size is None or position_size <= 0:
            self.logger.warning(
                "Position size calculation failed or resulted in non-positive size. No entry.")
            return

        # --- Place Entry Order (Market Order with SL attached if possible) ---
        order_type = 'market'
        self.logger.info(
            f"Attempting {order_side.value} {order_type} order: Size={position_size}, Est.Entry={entry_price_estimate}, Calc.SL={stop_loss_price}")

        # Prepare parameters (Bybit V5 allows attaching SL/TP with market order)
        # Use quantized SL price string for the parameter
        sl_price_str = self.exchange.format_value_for_api(
            self.symbol, 'price', stop_loss_price)
        order_params = {'stopLoss': sl_price_str}
        # Optional: Add TP param if calculated: 'takeProfit': tp_price_str
        # Optional: Specify trigger price type: 'slTriggerBy': 'MarkPrice'

        position_idx = None
        if self.hedge_mode:
            position_idx = 1 if order_side == OrderSide.BUY else 2
            order_params['positionIdx'] = position_idx
        else:
            order_params['positionIdx'] = 0

        # Create the order
        order_result = self.exchange.create_order(
            symbol=self.symbol, order_type=order_type, side=order_side,
            amount=position_size, price=None, params=order_params
        )

        # --- Post-Order Handling ---
        if order_result and order_result.get('id'):
            order_id = order_result['id']
            self.logger.info(
                f"Entry order placed successfully. ID: {order_id}, Status: {order_result.get('status', 'unknown')}. Intended SL: {stop_loss_price}")
            self.state['last_order_id'] = order_id

            # --- Optimistic State Update ---
            self.state['active_position'] = {
                'symbol': self.symbol, 'side': target_side,
                'entry_price': entry_price_estimate,  # Will be updated by sync
                'size': position_size,  # Will be updated by sync
                'order_id': order_id, 'position_idx': position_idx
            }
            # Store intended SL
            self.state['stop_loss_price'] = stop_loss_price
            self.state['take_profit_price'] = None
            self.state['break_even_achieved'] = False
            save_state(self.state, self.state_path,
                       self.logger)  # Save immediately
            self.logger.info(
                f"Bot state updated optimistically for new {target_side.name} position.")

            # --- Verification Step ---
            if self.post_order_verify_delay > 0:
                self.logger.info(
                    f"Waiting {self.post_order_verify_delay}s before verifying entry...")
                time.sleep(self.post_order_verify_delay)
                self.logger.info("Verifying position status after entry...")
                final_pos = self.get_current_position()
                # Sync state again based on verification result (will update entry/size/SL if needed)
                self.sync_bot_state_with_position(final_pos)

                # Check if SL seems correct after sync
                if self.state.get('active_position'):
                    state_sl = self.state.get('stop_loss_price')
                    if state_sl is not None:
                        tolerance = Decimal('1e-9')
                        if abs(state_sl - stop_loss_price) < tolerance:
                            self.logger.info(
                                f"Stop loss {stop_loss_price} confirmed active in state after verification.")
                        else:
                            self.logger.warning(
                                f"State SL ({state_sl}) differs from intended SL ({stop_loss_price}) after sync. Using state value.")
                    else:
                        # State SL is None after sync, meaning exchange didn't apply it or it got cancelled?
                        self.logger.warning(
                            "Stop loss NOT found on exchange after placing order with SL param. Attempting to set SL again.")
                        protection_params = {'stop_loss': stop_loss_price}
                        if self.hedge_mode:
                            protection_params['position_idx'] = self.state['active_position'].get(
                                'position_idx')
                        if self.exchange.set_protection(self.symbol, **protection_params):
                            self.logger.info(
                                "Successfully set stop loss via set_protection after initial attempt failed/missing.")
                            # Update state SL after explicit set
                            self.state['stop_loss_price'] = stop_loss_price
                            save_state(self.state, self.state_path,
                                       self.logger)  # Save again
                        else:
                            self.logger.error(
                                "Retry failed to set stop loss after entry.")
                else:
                    self.logger.error(
                        "Position not found during verification after placing entry order. State cleared by sync.")
            else:
                self.logger.info("Skipping post-order verification step.")

        else:
            self.logger.error(
                "Failed to place entry order (API call failed or returned invalid result).")
            # Ensure state remains clean
            self._clear_position_state("Entry order placement failed")
            save_state(self.state, self.state_path, self.logger)

    def close_position(self, reason: str):
        """Closes the current active position with a market order using reduceOnly."""
        position_state = self.state.get('active_position')
        if not position_state:
            self.logger.warning(
                "Close position called but no active position in state.")
            return

        size_to_close = position_state.get('size')
        current_side = position_state.get('side')
        position_idx = position_state.get('position_idx')

        # Validate state data
        if not isinstance(size_to_close, Decimal) or size_to_close <= 0:
            self.logger.error(
                f"Cannot close: Invalid size {size_to_close} in state. Clearing state.")
            self._clear_position_state(f"Invalid size in state during close")
            save_state(self.state, self.state_path, self.logger)
            return
        if current_side is None or current_side == PositionSide.NONE:
            self.logger.error(
                f"Cannot close: Invalid side {current_side} in state. Clearing state.")
            self._clear_position_state(f"Invalid side in state during close")
            save_state(self.state, self.state_path, self.logger)
            return
        if self.hedge_mode and position_idx is None:
            self.logger.error(
                f"Cannot close hedge mode: positionIdx missing. Clearing state.")
            self._clear_position_state(
                "Missing hedge idx in state during close")
            save_state(self.state, self.state_path, self.logger)
            return

        close_side = OrderSide.SELL if current_side == PositionSide.LONG else OrderSide.BUY
        order_type = 'market'

        self.logger.info(
            f"Attempting to close {current_side.name} position (Idx:{position_idx if self.hedge_mode else 'N/A'}, Size:{size_to_close}) via {order_type} order. Reason: {reason}")

        # Parameters for closing order
        close_params = {'reduceOnly': True}
        if self.hedge_mode:
            close_params['positionIdx'] = position_idx
        else:
            close_params['positionIdx'] = 0

        # Optional: Cancel existing SL/TP orders first? Bybit might do this automatically.
        # self.cancel_position_related_orders(position_idx) # Needs implementation

        # Create the closing order
        order_result = self.exchange.create_order(
            symbol=self.symbol, order_type=order_type, side=close_side,
            amount=size_to_close, price=None, params=close_params
        )

        if order_result and order_result.get('id'):
            order_id = order_result['id']
            status = order_result.get('status', 'unknown')
            self.logger.info(
                f"Position close order placed successfully. ID: {order_id}, Status: {status}. Reason: {reason}")
            # Optimistic state clearing
            self._clear_position_state(
                f"Close order placed (ID: {order_id}, Reason: {reason})")
            self.state['last_order_id'] = order_id  # Store close order ID
            save_state(self.state, self.state_path,
                       self.logger)  # Save cleared state

            # Optional: Verify closure after delay?
            # time.sleep(verify_delay) ... verify ...

        else:
            self.logger.error(
                f"Failed to place position close order. Reason: {reason}. Position state remains unchanged.")
            # State remains, will retry closing on next cycle if conditions persist.

    def _wait_for_next_cycle(self, cycle_start_time: float):
        """Waits until the next polling interval, accounting for execution time."""
        cycle_end_time = time.time()
        execution_time = cycle_end_time - cycle_start_time
        wait_time = max(0, self.poll_interval - execution_time)
        self.logger.debug(
            f"Cycle execution time: {execution_time:.2f}s. Waiting for {wait_time:.2f}s...")
        if wait_time > 0:
            time.sleep(wait_time)


# --- Main Execution ---

if __name__ == "__main__":
    # Set Decimal context globally at the very beginning
    try:
        getcontext().prec = CALCULATION_PRECISION  # Set high precision for calculations
        getcontext().rounding = ROUND_HALF_UP  # Default rounding
        print(
            f"Decimal context set: Precision={getcontext().prec}, Rounding={getcontext().rounding}")
    except Exception as e:
        # Use basic print as logger might not be ready
        print(f"CRITICAL: Failed to set Decimal context: {e}", file=sys.stderr)
        sys.exit(1)

    bot_instance = None
    try:
        # Initialize logger first (using default level)
        # Logger level will be updated from config inside TradingBot.__init__
        initial_logger = setup_logger()
        initial_logger.info("Logger initialized. Starting bot setup...")

        # Create bot instance (initializes components, loads config/state)
        bot_instance = TradingBot(
            config_path=CONFIG_FILE, state_path=STATE_FILE)

        # Start the main loop
        bot_instance.run()

    except SystemExit as e:
        print(f"Bot exited with code {e.code}.", file=sys.stderr)
        # Logger might be available, try logging exit reason
        logger = logging.getLogger("TradingBot")
        if logger.hasHandlers():
            logger.info(f"Bot process terminated with exit code {e.code}.")
        sys.exit(e.code)  # Propagate exit code
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received during initialization or main execution. Exiting.", file=sys.stderr)
        logger = logging.getLogger("TradingBot")
        if logger.hasHandlers():
            logger.info("KeyboardInterrupt received. Exiting.")
        sys.exit(0)
    except Exception as e:
        # Catch any unexpected critical errors during setup or run
        print(f"CRITICAL UNHANDLED ERROR: {e}", file=sys.stderr)
        logger = logging.getLogger("TradingBot")
        if logger.hasHandlers():  # Check if logger setup was successful
            logger.critical(
                f"Unhandled exception caused bot termination: {e}", exc_info=True)
        else:  # Logger not ready, print traceback
            import traceback
            traceback.print_exc()
        sys.exit(1)  # Exit with error code
