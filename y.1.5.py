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
# import os # Unused import removed
import math # Added for math.isfinite, math.ceil
from decimal import Decimal, getcontext, ROUND_HALF_UP, InvalidOperation, ROUND_DOWN, ROUND_UP
from typing import Dict, Any, Optional, Tuple, List, Union
from enum import Enum
from pathlib import Path

# --- Configuration ---

# Set Decimal precision (adjust as needed for your asset's precision)
DECIMAL_PRECISION = 8
# Add buffer for intermediate calculations; final results should be rounded appropriately
getcontext().prec = DECIMAL_PRECISION + 6
getcontext().rounding = ROUND_HALF_UP # Default rounding, can be overridden for specific cases

CONFIG_FILE = Path("config.json")
STATE_FILE = Path("bot_state.json")
LOG_FILE = Path("trading_bot.log")
LOG_LEVEL = logging.INFO # Default log level, can be overridden by config

# --- Enums ---

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"
    NONE = "none" # Represents no position or Bybit's 'None' side in hedge mode

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
    logger.setLevel(LOG_LEVEL) # Set initial level, may be updated by config later

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
        log_dir.mkdir(parents=True, exist_ok=True) # Ensure log directory exists
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=5*1024*1024, backupCount=3 # 5 MB per file, 3 backups
        )
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # Log to console if file handler fails
        logger.error(f"Failed to set up file logging handler: {e}", exc_info=True)

    # Suppress noisy libraries if needed
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("ccxt").setLevel(logging.WARNING) # Adjust if more ccxt detail needed

    return logger

def decimal_serializer(obj: Any) -> Union[str, Any]:
    """JSON serializer for Decimal objects, handling special values."""
    if isinstance(obj, Decimal):
        if obj.is_nan():
            return 'NaN'
        if obj.is_infinite():
            # Ensure standard representation for Infinity
            return 'Infinity' if obj > 0 else '-Infinity'
        # Standard finite Decimal to string
        return str(obj)
    # Let the default JSON encoder handle other types or raise TypeError
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable by this function")

def decimal_decoder(dct: Dict[str, Any]) -> Dict[str, Any]:
    """JSON decoder hook to convert numeric-like strings back to Decimal."""
    new_dct = {}
    for key, value in dct.items():
        if isinstance(value, str):
            try:
                # Attempt direct Decimal conversion for potential numbers or special values
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
                    try:
                        new_list.append(Decimal(item))
                    except InvalidOperation:
                        new_list.append(item) # Keep as string if not Decimal
                else:
                    new_list.append(item) # Keep other types as is
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
        logger.error(f"Error decoding JSON configuration file {config_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}", exc_info=True)
        return None

def validate_config(config: Dict[str, Any], logger: logging.Logger) -> bool:
    """Validates the loaded configuration."""
    is_valid = True # Assume valid initially
    required_keys = [
        "exchange", "api_credentials", "trading_settings", "indicator_settings",
        "risk_management", "logging"
    ]
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required configuration section: '{key}'")
            is_valid = False

    # If basic sections are missing, don't proceed with detailed checks
    if not is_valid:
        return False

    # Example: Validate specific parameters within sections
    exchange_cfg = config.get("exchange", {})
    if not isinstance(exchange_cfg.get("id"), str):
        logger.error("Config validation failed: 'exchange.id' must be a string.")
        is_valid = False

    api_creds = config.get("api_credentials", {})
    if not isinstance(api_creds.get("api_key"), str) or not api_creds.get("api_key"):
        logger.error("Config validation failed: 'api_credentials.api_key' must be a non-empty string.")
        is_valid = False
    if not isinstance(api_creds.get("api_secret"), str) or not api_creds.get("api_secret"):
        logger.error("Config validation failed: 'api_credentials.api_secret' must be a non-empty string.")
        is_valid = False

    settings = config.get("trading_settings", {})
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

    indicators = config.get("indicator_settings", {})
    if "ema_short_period" in indicators and "ema_long_period" in indicators:
        ema_short = indicators["ema_short_period"]
        ema_long = indicators["ema_long_period"]
        if not (isinstance(ema_short, int) and ema_short > 0):
             logger.error("Config validation failed: 'ema_short_period' must be a positive integer.")
             is_valid = False
        if not (isinstance(ema_long, int) and ema_long > 0):
             logger.error("Config validation failed: 'ema_long_period' must be a positive integer.")
             is_valid = False
        # Check periods only if both are valid integers
        if is_valid and isinstance(ema_short, int) and isinstance(ema_long, int) and ema_short >= ema_long:
            logger.error("Config validation failed: 'ema_short_period' must be less than 'ema_long_period'.")
            is_valid = False

    risk = config.get("risk_management", {})
    risk_percent = risk.get("risk_per_trade_percent")
    # Ensure risk_percent is loaded as Decimal by the decoder
    if not isinstance(risk_percent, Decimal) or not risk_percent.is_finite() or not (Decimal(0) < risk_percent <= Decimal(100)):
         logger.error("Config validation failed: 'risk_per_trade_percent' must be a finite number between 0 (exclusive) and 100 (inclusive).")
         is_valid = False

    if is_valid:
        logger.info("Configuration validation successful.")
    return is_valid


def load_state(state_path: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Loads the bot's state from a JSON file with Decimal conversion."""
    if not state_path.is_file():
        logger.warning(f"State file not found at {state_path}. Starting with empty state.")
        return {} # Return empty dict if no state file
    try:
        with open(state_path, 'r') as f:
            # Use object_hook for robust Decimal handling in nested structures
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
        state_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=4, default=decimal_serializer)
        logger.debug(f"Bot state saved successfully to {state_path}")
    except TypeError as e:
        logger.error(f"Error serializing state for saving (check for non-Decimal/non-standard types): {e}", exc_info=True)
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
        self.category = config['trading_settings'].get('category', 'linear') # linear, inverse, spot
        self.hedge_mode = config['trading_settings'].get('hedge_mode', False) # Bybit hedge mode
        self.max_retries = config['exchange'].get('max_retries', 3)
        self.retry_delay = config['exchange'].get('retry_delay_seconds', 5)

        if self.exchange_id != 'bybit':
             self.logger.warning(f"This wrapper is optimized for Bybit V5, but exchange ID is set to '{self.exchange_id}'. Some features might not work as expected.")

        try:
            exchange_class = getattr(ccxt, self.exchange_id)
        except AttributeError:
             self.logger.critical(f"CCXT exchange class not found for ID: '{self.exchange_id}'. Exiting.")
             raise # Re-raise critical error

        self.exchange = exchange_class({
            'apiKey': config['api_credentials']['api_key'],
            'secret': config['api_credentials']['api_secret'],
            'enableRateLimit': True, # Let CCXT handle basic rate limiting
            'options': {
                # Default type influences which market (spot/swap) is used if symbol is ambiguous
                'defaultType': 'swap' if self.category in ['linear', 'inverse'] else 'spot',
                'adjustForTimeDifference': True, # CCXT handles time sync issues
                'broker_id': config['exchange'].get('broker_id', None), # Optional broker ID for Bybit
                # Add other necessary options if needed
            },
            # Explicitly set sandbox mode if needed via config
            # 'sandboxMode': config['exchange'].get('sandbox_mode', False), # Example
        })

        # Set sandbox URL if sandboxMode is enabled
        if config['exchange'].get('sandbox_mode', False):
            self.logger.warning("Sandbox mode enabled. Using testnet URLs.")
            self.exchange.set_sandbox_mode(True)

        # Load markets to get precision details, contract sizes, limits etc.
        try:
            self.exchange.load_markets()
            self.logger.info(f"Markets loaded successfully for {self.exchange_id}.")
        except ccxt.AuthenticationError:
            self.logger.exception("Authentication failed loading markets. Check API keys.")
            raise # Re-raise critical error
        except ccxt.NetworkError as e:
            self.logger.exception(f"Network error loading markets: {e}")
            raise # Re-raise critical error
        except ccxt.ExchangeError as e:
            self.logger.exception(f"Exchange error loading markets: {e}")
            raise # Re-raise critical error
        except Exception as e:
             self.logger.exception(f"Unexpected error loading markets: {e}")
             raise # Re-raise critical error

    def get_market(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Gets market data for a symbol, ensuring it's loaded."""
        try:
            # Ensure markets are loaded if not already
            if not self.exchange.markets:
                self.logger.warning("Markets not loaded. Attempting to load markets now.")
                self.exchange.load_markets()

            market = self.exchange.market(symbol)
            if market:
                # Ensure precision values are loaded correctly (sometimes might be missing initially)
                if 'precision' not in market or not market['precision'] or \
                   'amount' not in market['precision'] or 'price' not in market['precision']:
                     self.logger.warning(f"Precision info incomplete for market {symbol}. Reloading markets.")
                     self.exchange.load_markets(reload=True)
                     market = self.exchange.market(symbol) # Try fetching again
                     if 'precision' not in market or not market['precision'] or \
                        'amount' not in market['precision'] or 'price' not in market['precision']:
                          self.logger.error(f"Failed to load complete precision info for {symbol} even after reload.")
                          return None # Cannot proceed reliably without precision
            else:
                 # This case should ideally be caught by BadSymbol below, but added for clarity
                 self.logger.error(f"Market data for symbol '{symbol}' is null or empty after fetching.")
                 return None
            return market
        except ccxt.BadSymbol:
            self.logger.error(f"Symbol '{symbol}' not found on {self.exchange_id}.")
            return None
        except ccxt.ExchangeError as e: # Catch errors during market loading/fetching within this method
             self.logger.error(f"Exchange error fetching market data for {symbol}: {e}", exc_info=True)
             return None
        except Exception as e:
            self.logger.error(f"Unexpected error fetching market data for {symbol}: {e}", exc_info=True)
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
                params = kwargs.get('params', {}).copy()
                # Check if it's Bybit and the method likely requires 'category' for V5 Unified/Contract accounts
                if self.exchange_id == 'bybit' and self.category in ['linear', 'inverse'] and 'category' not in params:
                     # List of methods known or likely to require 'category' for V5 derivatives
                     methods_requiring_category = [
                         'create_order', 'edit_order', 'cancel_order', 'cancel_all_orders',
                         'fetch_order', 'fetch_open_orders', 'fetch_closed_orders', 'fetch_my_trades',
                         'fetch_position', 'fetch_positions', # fetch_position singular might exist
                         'fetch_balance', 'set_leverage', 'set_margin_mode',
                         'fetch_leverage_tiers', 'fetch_funding_rate', 'fetch_funding_rates',
                         'fetch_funding_rate_history',
                         'private_post_position_trading_stop', # Specific endpoint for SL/TP
                         # Add others if discovered: e.g., transfer, withdraw?
                     ]
                     if method_name in methods_requiring_category:
                         params['category'] = self.category
                         # Update kwargs for the current call attempt
                         kwargs['params'] = params

                self.logger.debug(f"Calling CCXT method: {method_name}, Args: {args}, Kwargs: {kwargs}")
                result = method(*args, **kwargs)
                self.logger.debug(f"CCXT call {method_name} successful. Result snippet: {str(result)[:200]}...")
                return result

            # --- Specific CCXT/Bybit Error Handling (Non-Retryable First) ---
            except ccxt.AuthenticationError as e:
                self.logger.error(f"Authentication Error: {e}. Check API keys. Non-retryable.")
                return None
            except ccxt.PermissionDenied as e:
                 self.logger.error(f"Permission Denied: {e}. Check API key permissions (IP whitelist, endpoint access). Non-retryable.")
                 return None
            except ccxt.AccountSuspended as e:
                 self.logger.error(f"Account Suspended: {e}. Non-retryable.")
                 return None
            except ccxt.InvalidOrder as e: # Includes OrderNotFound, OrderNotFillable etc.
                self.logger.error(f"Invalid Order parameters or state: {e}. Check order details (size, price, type, status). Non-retryable.")
                return None
            except ccxt.InsufficientFunds as e:
                self.logger.error(f"Insufficient Funds: {e}. Non-retryable.")
                return None
            except ccxt.BadSymbol as e:
                self.logger.error(f"Invalid Symbol: {e}. Non-retryable.")
                return None
            except ccxt.BadRequest as e:
                 # Often parameter errors, potentially Bybit specific codes in message
                 self.logger.error(f"Bad Request: {e}. Check parameters. Assuming non-retryable.")
                 # Example: Could parse e.args[0] for specific Bybit error codes like '110007' (parameter error)
                 # if '110007' in str(e): self.logger.error("... specific parameter error ...")
                 return None
            except ccxt.MarginModeAlreadySet as e: # Example specific error
                 self.logger.warning(f"Margin mode already set as requested: {e}. Considered success.")
                 return {} # Return empty dict to indicate success/no action needed
            except ccxt.OperationFailed as e: # Catch specific operational failures if needed
                 # Example: Bybit might raise this for position size issues not caught by InvalidOrder
                 self.logger.error(f"Operation Failed: {e}. May indicate position/margin issues. Assuming non-retryable.")
                 return None

            # --- Retryable Errors ---
            except ccxt.RateLimitExceeded as e:
                self.logger.warning(f"Rate Limit Exceeded: {e}. Retrying in {self.retry_delay}s... (Attempt {retries + 1}/{self.max_retries + 1})")
                # CCXT's built-in rate limiter might handle this, but explicit retry adds robustness
            except ccxt.NetworkError as e: # Includes ConnectionError, Timeout, etc.
                self.logger.warning(f"Network Error: {e}. Retrying in {self.retry_delay}s... (Attempt {retries + 1}/{self.max_retries + 1})")
            except ccxt.ExchangeNotAvailable as e: # Maintenance or temporary outage
                self.logger.warning(f"Exchange Not Available: {e}. Retrying in {self.retry_delay}s... (Attempt {retries + 1}/{self.max_retries + 1})")
            except ccxt.OnMaintenance as e: # Specific maintenance error
                 self.logger.warning(f"Exchange On Maintenance: {e}. Retrying in {self.retry_delay}s... (Attempt {retries + 1}/{self.max_retries + 1})")
            except ccxt.ExchangeError as e: # General exchange error, potentially retryable
                # Check for specific Bybit codes/messages that might be non-retryable
                msg = str(e).lower()
                # Example: Bybit position idx errors, certain margin errors might be non-retryable
                non_retryable_msgs = ['position idx not match', 'insufficient available balance', 'risk limit', 'order cost not available']
                if any(term in msg for term in non_retryable_msgs):
                     self.logger.error(f"Potentially non-retryable Exchange Error: {e}.")
                     return None
                # Otherwise, assume retryable for generic ExchangeError
                self.logger.warning(f"Generic Exchange Error: {e}. Retrying in {self.retry_delay}s... (Attempt {retries + 1}/{self.max_retries + 1})")

            # --- Unexpected Errors ---
            except Exception as e:
                 self.logger.error(f"Unexpected error during CCXT call '{method_name}': {e}", exc_info=True)
                 # Decide if unexpected errors should be retried or immediately fail
                 # Let's assume non-retryable for safety unless specifically known otherwise
                 return None


            # --- Retry Logic ---
            retries += 1
            if retries <= self.max_retries:
                time.sleep(self.retry_delay)
            else:
                self.logger.error(f"CCXT call '{method_name}' failed after {self.max_retries + 1} attempts.")
                return None

        # Should not be reached if loop condition is correct, but satisfy linters/type checkers
        return None

    # --- Specific API Call Wrappers ---

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetches OHLCV data and returns it as a Pandas DataFrame with Decimal types."""
        self.logger.info(f"Fetching {limit} OHLCV candles for {symbol} ({timeframe})...")
        # Add 'since' parameter? Fetching last 'limit' candles is typical for TA.
        ohlcv = self.safe_ccxt_call('fetch_ohlcv', symbol, timeframe, limit=limit) # Add since=None if needed
        if ohlcv is None or not ohlcv:
            # safe_ccxt_call already logged the error
            # self.logger.error(f"Failed to fetch OHLCV data for {symbol}.")
            return None

        try:
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            if df.empty:
                 self.logger.warning(f"Fetched OHLCV data for {symbol} resulted in an empty DataFrame.")
                 return df # Return empty df

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Convert OHLCV columns to Decimal, handling potential None or non-numeric values robustly
            for col in ['open', 'high', 'low', 'close', 'volume']:
                # Use apply with a lambda for safe conversion row by row
                df[col] = df[col].apply(lambda x: Decimal(str(x)) if x is not None else Decimal('NaN'))

            # Validate data sanity (e.g., check for NaNs, zero prices)
            nan_mask = df[['open', 'high', 'low', 'close']].isnull().any(axis=1)
            if nan_mask.any():
                 self.logger.warning(f"{nan_mask.sum()} rows with NaN values found in fetched OHLCV data for {symbol}.")
                 # Optionally drop rows with NaNs if they interfere with indicators: df.dropna(subset=[...], inplace=True)

            zero_price_mask = (df[['open', 'high', 'low', 'close']] <= Decimal(0)).any(axis=1)
            if zero_price_mask.any():
                 self.logger.warning(f"{zero_price_mask.sum()} rows with zero or negative values found in fetched OHLCV prices for {symbol}.")

            self.logger.info(f"Successfully fetched and processed {len(df)} OHLCV candles for {symbol}.")
            return df
        except Exception as e:
            self.logger.error(f"Error processing OHLCV data into DataFrame: {e}", exc_info=True)
            return None

    def fetch_balance(self) -> Optional[Dict[str, Decimal]]:
        """Fetches account balance, returning total equity amounts as Decimals."""
        self.logger.debug("Fetching account balance...")
        # Bybit V5 requires category for fetch_balance (handled by safe_ccxt_call)
        balance_data = self.safe_ccxt_call('fetch_balance')

        if balance_data is None:
            # Error logged by safe_ccxt_call
            # self.logger.error("Failed to fetch balance.")
            return None

        balances = {}
        try:
            # We primarily need the 'total' or 'equity' for risk calculation base.
            # CCXT unified structure: balance_data['total'][ASSET]
            # Bybit V5 structure is nested within 'info'

            if self.exchange_id == 'bybit' and self.category in ['linear', 'inverse', 'spot']:
                # --- Bybit V5 Parsing ---
                account_list = balance_data.get('info', {}).get('result', {}).get('list', [])
                if not account_list:
                     self.logger.warning("Balance response structure unexpected (no 'list'). Trying top-level parsing.")
                     # Fallback to standard CCXT structure parsing
                     for asset, bal_info in balance_data.get('total', {}).items():
                          if bal_info is not None:
                              try:
                                  balances[asset] = Decimal(str(bal_info))
                              except InvalidOperation:
                                  self.logger.warning(f"Could not convert balance (fallback) for {asset} to Decimal: {bal_info}")
                     if balances: return balances # Return if fallback worked
                     else:
                          self.logger.error("Failed to parse balance from Bybit V5 response (neither 'info.result.list' nor 'total' structure found).")
                          self.logger.debug(f"Raw balance data: {balance_data}")
                          return None

                # Determine target account type based on category
                account_type_map = {'linear': 'CONTRACT', 'inverse': 'CONTRACT', 'spot': 'SPOT'}
                target_account_type = account_type_map.get(self.category)
                unified_account = None
                specific_account = None

                for acc in account_list:
                    acc_type = acc.get('accountType')
                    if acc_type == 'UNIFIED':
                        unified_account = acc
                        break # Prefer UNIFIED if found
                    elif acc_type == target_account_type:
                        specific_account = acc

                # Choose which account data to parse
                account_to_parse = unified_account or specific_account

                if account_to_parse:
                    self.logger.debug(f"Parsing balance from account type: {account_to_parse.get('accountType')}")
                    coin_data = account_to_parse.get('coin', [])
                    for coin_info in coin_data:
                        asset = coin_info.get('coin')
                        # Use 'equity' for total value including PnL (suitable for risk %)
                        # 'walletBalance' is cash balance excluding PnL
                        # 'availableToWithdraw' / 'availableBalance' might correspond to 'free'
                        total_balance_str = coin_info.get('equity')
                        if asset and total_balance_str is not None:
                             try:
                                 balances[asset] = Decimal(str(total_balance_str))
                             except InvalidOperation:
                                 self.logger.warning(f"Could not convert balance for {asset} to Decimal: {total_balance_str}")
                        # Optionally parse 'free' balance if needed elsewhere
                        # free_balance_str = coin_info.get('availableToWithdraw') or coin_info.get('availableBalance')
                        # if asset and free_balance_str is not None: ...
                else:
                     self.logger.warning(f"Could not find relevant account type ('UNIFIED' or '{target_account_type}') in Bybit V5 balance response list.")
                     # Optional: Log the account types that were found
                     found_types = [acc.get('accountType') for acc in account_list]
                     self.logger.debug(f"Account types found in response: {found_types}")
                     # Fallback attempt (as above)
                     for asset, bal_info in balance_data.get('total', {}).items():
                          if bal_info is not None:
                              try: balances[asset] = Decimal(str(bal_info))
                              except InvalidOperation: pass # Ignore conversion errors in fallback
                     if not balances:
                          self.logger.error("Failed to parse balance from Bybit V5 response after fallback.")
                          self.logger.debug(f"Raw balance data: {balance_data}")
                          return None


            else: # --- Standard CCXT Parsing (Non-Bybit or fallback) ---
                for asset, bal_info in balance_data.get('total', {}).items():
                     if bal_info is not None:
                         try:
                            balances[asset] = Decimal(str(bal_info))
                         except InvalidOperation:
                             self.logger.warning(f"Could not convert balance for {asset} to Decimal: {bal_info}")
                # Optionally parse 'free' balance: balance_data.get('free', {})

            if not balances:
                 self.logger.warning("Parsed balance data is empty.")
                 self.logger.debug(f"Raw balance data: {balance_data}")
                 # Return empty dict instead of None if parsing finished but found nothing
                 return {}
            else:
                 self.logger.info(f"Balance fetched successfully. Assets with total equity: {list(balances.keys())}")
            return balances

        except Exception as e:
            self.logger.error(f"Error parsing balance data: {e}", exc_info=True)
            self.logger.debug(f"Raw balance data: {balance_data}")
            return None

    def fetch_positions(self, symbol: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Fetches open positions, converting relevant fields to Decimal.
        Handles Bybit V5 specifics like 'category' and parsing 'info'.
        Filters for non-zero positions.
        """
        self.logger.debug(f"Fetching positions for symbol: {symbol or 'all'}...")
        params = {}
        symbols_arg = [symbol] if symbol else None # CCXT standard way to filter by symbol

        # Bybit V5 specific: Can filter by symbol directly in params, and requires category
        if self.exchange_id == 'bybit':
            params['category'] = self.category
            if symbol:
                params['symbol'] = symbol
                symbols_arg = None # Don't pass symbols list if using params filter for Bybit

        positions_data = self.safe_ccxt_call('fetch_positions', symbols=symbols_arg, params=params)

        if positions_data is None:
            # Error logged by safe_ccxt_call
            # self.logger.error(f"Failed to fetch positions for {symbol or 'all'}.")
            return None

        processed_positions = []
        try:
            for pos in positions_data:
                # --- Reliable check for active position ---
                # Use 'size' from 'info' for Bybit V5 as primary check
                # Use 'contracts' from standard CCXT structure as fallback
                size_str = pos.get('info', {}).get('size') # Bybit V5 size
                contracts_val = pos.get('contracts') # Standard CCXT field

                size_dec = Decimal('0')
                is_active = False

                if size_str is not None and size_str != '':
                    try:
                        size_dec = Decimal(str(size_str))
                        if size_dec != Decimal(0): is_active = True
                    except InvalidOperation:
                         self.logger.warning(f"Could not parse position size '{size_str}' from info as Decimal.")
                elif contracts_val is not None:
                     try:
                          # contracts might already be float/int from CCXT, convert safely
                          size_dec = Decimal(str(contracts_val))
                          if size_dec != Decimal(0): is_active = True
                     except InvalidOperation:
                          self.logger.warning(f"Could not parse position contracts '{contracts_val}' as Decimal.")

                if not is_active:
                    self.logger.debug(f"Skipping zero-size or invalid-size position: {pos.get('symbol')}")
                    continue # Skip zero-size or unparsable-size positions

                processed = pos.copy() # Work on a copy
                processed['contracts'] = size_dec # Store unified size in 'contracts'

                # --- Convert relevant fields to Decimal ---
                # Standard CCXT fields (some might overlap with info fields)
                decimal_fields_std = ['contractSize', 'entryPrice', 'leverage',
                                      'liquidationPrice', 'markPrice', 'notional',
                                      'unrealizedPnl', 'initialMargin', 'maintenanceMargin',
                                      'initialMarginPercentage', 'maintenanceMarginPercentage',
                                      'marginRatio', 'collateral'] # Add more if needed
                # Bybit V5 specific fields often in 'info'
                decimal_fields_info = ['avgPrice', 'cumRealisedPnl', 'liqPrice', 'markPrice',
                                       'positionValue', 'stopLoss', 'takeProfit', 'trailingStop',
                                       'unrealisedPnl', 'size', 'positionIM', 'positionMM',
                                       'createdTime', 'updatedTime'] # Timestamps might be numeric strings

                # Process standard fields
                for field in decimal_fields_std:
                    if field in processed and processed[field] is not None:
                        try:
                            processed[field] = Decimal(str(processed[field]))
                        except InvalidOperation:
                             self.logger.warning(f"Could not convert standard position field '{field}' to Decimal: {processed[field]}")
                             processed[field] = Decimal('NaN') # Mark as NaN if conversion fails

                # Process info fields (handle potential non-dict 'info')
                if 'info' in processed and isinstance(processed['info'], dict):
                    info = processed['info']
                    for field in decimal_fields_info:
                        if field in info and info[field] is not None and info[field] != '':
                            try:
                                info[field] = Decimal(str(info[field]))
                            except InvalidOperation:
                                self.logger.warning(f"Could not convert position info field '{field}' to Decimal: {info[field]}")
                                info[field] = Decimal('NaN') # Mark as NaN

                # --- Determine position side more reliably ---
                # Use Bybit V5 'side' field from 'info' first
                side_enum = PositionSide.NONE # Default
                if 'info' in processed and isinstance(processed['info'], dict) and 'side' in processed['info']:
                    side_str = processed['info']['side'].lower()
                    if side_str == 'buy':
                        side_enum = PositionSide.LONG
                    elif side_str == 'sell':
                        side_enum = PositionSide.SHORT
                    elif side_str == 'none': # Explicit 'None' side from Bybit (e.g., hedge mode no position)
                         side_enum = PositionSide.NONE
                    else:
                         self.logger.warning(f"Unrecognized side '{side_str}' in position info for {processed.get('symbol')}")

                # Fallback to CCXT 'side' if info side is missing or ambiguous ('None')
                if side_enum == PositionSide.NONE and 'side' in processed and processed['side']:
                     side_str_std = str(processed['side']).lower()
                     if side_str_std == 'long':
                          side_enum = PositionSide.LONG
                     elif side_str_std == 'short':
                          side_enum = PositionSide.SHORT

                # Final check: If side is still 'None', but size > 0, log a warning.
                # This shouldn't happen if 'info.side' or 'side' is present for active positions.
                if side_enum == PositionSide.NONE and size_dec != Decimal(0):
                     self.logger.warning(f"Position for {processed.get('symbol')} has size {size_dec} but side is determined as 'None'. Check exchange data consistency.")
                     # Cannot reliably determine side, might need manual intervention or different logic. Skip?
                     # continue # Or assign a default based on PnL sign? Risky.

                processed['side'] = side_enum.value # Update the main 'side' field with 'long'/'short'/'none' string

                # Add positionIdx for hedge mode if available and store as int
                if 'info' in processed and isinstance(processed['info'], dict) and 'positionIdx' in processed['info']:
                    try:
                        processed['positionIdx'] = int(processed['info']['positionIdx'])
                    except (ValueError, TypeError):
                         self.logger.warning(f"Could not parse positionIdx '{processed['info']['positionIdx']}' as integer.")
                         processed['positionIdx'] = None # Or some default?
                else:
                     # Ensure positionIdx is present even if not hedge mode (set to 0 or None)
                     processed['positionIdx'] = 0 # Bybit uses 0 for one-way mode

                processed_positions.append(processed)

            self.logger.info(f"Fetched {len(processed_positions)} active position(s) for {symbol or 'all'}.")
            return processed_positions
        except Exception as e:
            self.logger.error(f"Error processing position data: {e}", exc_info=True)
            self.logger.debug(f"Raw positions data: {positions_data}")
            return None

    def format_value_for_api(self, symbol: str, value_type: str, value: Decimal) -> str:
        """
        Formats amount or price to string based on market precision for API calls.
        Raises ValueError on invalid input or if market data is missing.
        """
        if not isinstance(value, Decimal):
             raise ValueError(f"Invalid input value type for formatting: {type(value)}. Expected Decimal.")
        if not value.is_finite():
             raise ValueError(f"Invalid Decimal value for formatting: {value}. Must be finite.")

        market = self.get_market(symbol)
        if not market:
            # Error logged by get_market
            raise ValueError(f"Market data not found for {symbol}, cannot format value.")
        if 'precision' not in market or market['precision'] is None:
             raise ValueError(f"Market precision data missing for {symbol}, cannot format value.")

        # CCXT formatting methods usually expect float input
        value_float = float(value)

        try:
            if value_type == 'amount':
                # Use amount_to_precision for formatting quantity
                formatted_value = self.exchange.amount_to_precision(symbol, value_float)
            elif value_type == 'price':
                # Use price_to_precision for formatting price
                formatted_value = self.exchange.price_to_precision(symbol, value_float)
            else:
                raise ValueError(f"Invalid value_type: {value_type}. Use 'amount' or 'price'.")

            # CCXT formatting methods return strings. Return as string for consistency.
            return formatted_value
        except ccxt.ExchangeError as e:
             # Catch potential errors within CCXT formatting (e.g., if precision mode is weird)
             self.logger.error(f"CCXT error formatting {value_type} for {symbol}: {e}")
             raise ValueError(f"CCXT error formatting {value_type} for {symbol}") from e
        except Exception as e:
             self.logger.error(f"Unexpected error formatting {value_type} for {symbol}: {e}", exc_info=True)
             raise ValueError(f"Unexpected error formatting {value_type} for {symbol}") from e


    def create_order(self, symbol: str, order_type: str, side: OrderSide, amount: Decimal,
                     price: Optional[Decimal] = None, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Creates an order, ensuring amount and price are correctly formatted using market precision.
        Handles Bybit V5 specifics like 'category' and 'positionIdx'.
        """
        self.logger.info(f"Attempting to create {side.value} {order_type} order for {amount} {symbol} @ {price or 'market'}...")

        if amount <= 0:
             self.logger.error(f"Cannot create order: Amount must be positive ({amount}).")
             return None
        if order_type.lower() != 'market' and (price is None or price <= 0):
             self.logger.error(f"Cannot create non-market order: Price must be positive ({price}).")
             return None

        market = self.get_market(symbol)
        if not market:
            self.logger.error(f"Cannot create order: Market data for {symbol} not found.")
            return None

        try:
            # Format amount and price using market precision rules via helper function
            amount_str = self.format_value_for_api(symbol, 'amount', amount)
            price_str = self.format_value_for_api(symbol, 'price', price) if price is not None and order_type.lower() != 'market' else None

            self.logger.debug(f"Formatted order values: Amount='{amount_str}', Price='{price_str}'")

        except ValueError as e:
             self.logger.error(f"Error formatting order values: {e}")
             return None
        except Exception as e:
             self.logger.error(f"Unexpected error during value formatting: {e}", exc_info=True)
             return None


        # --- Prepare Parameters ---
        # Start with user-provided params (user params take precedence)
        order_params = params.copy() if params else {}

        # --- Bybit V5 Specific Parameters ---
        if self.exchange_id == 'bybit':
            # 'category' is handled by safe_ccxt_call

            # Determine positionIdx for hedge mode entries/exits
            if self.hedge_mode:
                # If 'positionIdx' is not already provided in params, determine based on side.
                # Assumes this create_order call is for opening a new position or potentially closing.
                # Closing logic might need to pass the specific positionIdx in params.
                if 'positionIdx' not in order_params:
                    # For opening: 1 for Buy/Long, 2 for Sell/Short
                    order_params['positionIdx'] = 1 if side == OrderSide.BUY else 2
                    self.logger.debug(f"Hedge mode: Setting positionIdx={order_params['positionIdx']} for {side.value} order.")
            else:
                 # Ensure positionIdx is 0 for one-way mode if not specified
                 if 'positionIdx' not in order_params:
                      order_params['positionIdx'] = 0


        # Prepare arguments for safe_ccxt_call
        # Pass formatted strings directly, CCXT handles conversion if needed
        call_args = [symbol, order_type, side.value, amount_str]
        # Price argument handling depends on order type
        if order_type.lower() == 'market':
             call_args.append(None) # Explicitly pass None for price in market orders
        else:
             if price_str is None:
                  self.logger.error(f"Price is required for order type '{order_type}' but was not provided or formatted correctly.")
                  return None
             call_args.append(price_str)


        order_result = self.safe_ccxt_call('create_order', *call_args, params=order_params)

        if order_result:
            order_id = order_result.get('id')
            status = order_result.get('status', 'unknown')
            self.logger.info(f"Order creation request successful. Order ID: {order_id}, Initial Status: {status}")
            # Further processing/parsing of the result can be done here if needed
            # e.g., converting fields back to Decimal, checking fill details immediately (though market orders might take milliseconds)
        else:
            # Error logged by safe_ccxt_call
            self.logger.error(f"Failed to create {side.value} {order_type} order for {symbol}.")

        return order_result # Return raw CCXT result (or None on failure)

    def set_leverage(self, symbol: str, leverage: Decimal) -> bool:
        """Sets leverage for a symbol. Handles Bybit V5 specifics."""
        self.logger.info(f"Setting leverage for {symbol} to {leverage}x...")
        if not isinstance(leverage, Decimal):
             self.logger.error(f"Invalid leverage type: {type(leverage)}. Must be Decimal.")
             return False
        if not leverage.is_finite() or leverage <= 0:
             self.logger.error(f"Invalid leverage value: {leverage}. Must be a positive finite Decimal.")
             return False

        # Leverage value should typically be passed as a number (float/int) to CCXT's set_leverage
        # but Bybit V5 params require strings for buy/sell leverage.
        leverage_float = float(leverage)
        leverage_str = str(leverage) # Use string for params

        params = {}
        # Bybit V5 requires setting buyLeverage and sellLeverage separately in params
        # And requires the symbol argument within the main call
        if self.exchange_id == 'bybit':
            # 'category' handled by safe_ccxt_call
            params['buyLeverage'] = leverage_str
            params['sellLeverage'] = leverage_str

        # CCXT's set_leverage expects: leverage (number), symbol (string), params (dict)
        result = self.safe_ccxt_call('set_leverage', leverage_float, symbol, params=params)

        # Check result: Success might be indicated by non-None result or lack of error.
        # Bybit V5 setLeverage usually returns None or empty dict {} on success via CCXT.
        if result is not None: # safe_ccxt_call returns None only on failure after retries
             self.logger.info(f"Leverage for {symbol} set to {leverage}x request sent successfully.")
             # Optional: Check Bybit specific retCode if available in result['info']
             if isinstance(result, dict) and 'info' in result:
                  ret_code = result['info'].get('retCode')
                  if ret_code == 0:
                       self.logger.info("Bybit API confirmed successful leverage setting (retCode 0).")
                       return True
                  else:
                       # Request sent but Bybit indicated an issue
                       ret_msg = result['info'].get('retMsg', 'Unknown Bybit error')
                       self.logger.error(f"Failed to set leverage on Bybit. Code: {ret_code}, Msg: {ret_msg}")
                       return False
             # If no 'info' or not Bybit, assume success based on non-None result
             return True
        else:
             # Error handled and logged by safe_ccxt_call
             # self.logger.error(f"Failed to set leverage for {symbol} to {leverage}x.")
             return False


    def set_protection(self, symbol: str, stop_loss: Optional[Decimal] = None,
                       take_profit: Optional[Decimal] = None, trailing_stop: Optional[Dict[str, Decimal]] = None,
                       position_idx: Optional[int] = None) -> bool:
        """
        Sets stop loss, take profit, or trailing stop for a position using Bybit V5's trading stop endpoint.
        Note: This modifies existing position parameters, does not place new orders.
        """
        action_parts = []
        if stop_loss is not None: action_parts.append(f"SL={stop_loss}")
        if take_profit is not None: action_parts.append(f"TP={take_profit}")
        if trailing_stop is not None: action_parts.append(f"TSL={trailing_stop}")

        if not action_parts:
            self.logger.warning("set_protection called with no protection levels specified.")
            return False

        self.logger.info(f"Attempting to set protection for {symbol} (Idx: {position_idx if position_idx is not None else 'N/A'}): {' / '.join(action_parts)}")

        market = self.get_market(symbol)
        if not market:
            self.logger.error(f"Cannot set protection: Market data for {symbol} not found.")
            return False

        # --- Prepare Parameters for private_post_position_trading_stop ---
        params = {
            # 'category': self.category, # safe_ccxt_call adds this
            'symbol': symbol,
            # Bybit expects prices as strings, formatted to tick size
        }

        try:
            # Format SL and TP using market precision
            if stop_loss is not None:
                if stop_loss <= 0: raise ValueError("Stop loss must be positive.")
                params['stopLoss'] = self.format_value_for_api(symbol, 'price', stop_loss)
            if take_profit is not None:
                if take_profit <= 0: raise ValueError("Take profit must be positive.")
                params['takeProfit'] = self.format_value_for_api(symbol, 'price', take_profit)

            # --- Trailing Stop Handling (Bybit V5 specific) ---
            if trailing_stop:
                # Bybit uses 'trailingStop' for the distance/value (price points)
                # 'activePrice' for the activation price
                ts_value = trailing_stop.get('distance') or trailing_stop.get('value') # Use 'distance' or 'value' key
                ts_active_price = trailing_stop.get('activation_price')

                if ts_value is not None:
                     if not isinstance(ts_value, Decimal) or not ts_value.is_finite() or ts_value <= 0:
                          raise ValueError("Trailing stop distance/value must be a positive finite Decimal.")
                     # Bybit API expects TSL distance as a string value representing price points.
                     # Formatting as 'price' might work if tick size is appropriate, but safer to format based on price precision rules.
                     # Let's use format_value_for_api assuming the distance is in price units.
                     # WARNING: Verify if Bybit expects TSL distance formatted differently (e.g., percentage needs different handling).
                     params['trailingStop'] = self.format_value_for_api(symbol, 'price', ts_value)
                     self.logger.debug(f"Formatted trailing stop value: {params['trailingStop']}")
                     # Add trigger price types if needed (defaults are usually okay)
                     # params['tpslMode'] = 'Partial' # If needed
                     # params['tpTriggerBy'] = 'MarkPrice' # Or LastPrice, IndexPrice
                     # params['slTriggerBy'] = 'MarkPrice' # Or LastPrice, IndexPrice

                if ts_active_price is not None:
                     if not isinstance(ts_active_price, Decimal) or not ts_active_price.is_finite() or ts_active_price <= 0:
                          raise ValueError("Trailing stop activation price must be a positive finite Decimal.")
                     params['activePrice'] = self.format_value_for_api(symbol, 'price', ts_active_price)
                     self.logger.debug(f"Formatted trailing stop activation price: {params['activePrice']}")

        except ValueError as e:
             self.logger.error(f"Invalid or unformattable protection parameter value: {e}")
             return False
        except Exception as e:
             self.logger.error(f"Error formatting protection parameters: {e}", exc_info=True)
             return False


        # --- Position Index for Hedge Mode ---
        # Required by Bybit V5 for hedge mode positions
        if self.hedge_mode:
            if position_idx is None:
                 # Try to infer from SL/TP direction relative to mark price? Risky.
                 # Best practice: Caller MUST provide position_idx for hedge mode.
                 self.logger.error("Hedge mode active, but positionIdx is required for set_protection and was not provided.")
                 return False
            params['positionIdx'] = position_idx
        else:
             # Ensure positionIdx is 0 for one-way mode if not specified (Bybit default)
             params.setdefault('positionIdx', 0)


        # --- Execute API Call ---
        # Use the specific private method for Bybit V5 trading stop
        # This method might not be universally available in CCXT; ensure your version supports it.
        # If not, you might need to use exchange-specific methods or fall back to order modification.
        if not hasattr(self.exchange, 'private_post_position_trading_stop'):
             self.logger.error("CCXT exchange object does not have 'private_post_position_trading_stop'. Cannot set protection this way.")
             # Alternative: Try modifying an existing SL/TP order? Or use create_order with modify params? More complex.
             return False

        result = self.safe_ccxt_call('private_post_position_trading_stop', params=params)

        # --- Process Result ---
        if result is not None:
            # Check Bybit V5 API docs for response structure on success/failure
            # Expecting {'retCode': 0, 'retMsg': 'OK', ...} on success
            ret_code = result.get('retCode')
            ret_msg = result.get('retMsg', 'No message')
            ext_info = result.get('retExtInfo', {}) # Contains detailed errors sometimes

            if ret_code == 0:
                 self.logger.info(f"Protection levels set/updated successfully via API for {symbol} (Idx: {params['positionIdx']}).")
                 return True
            else:
                 # Log specific Bybit error code/message
                 self.logger.error(f"Failed to set protection for {symbol}. Code: {ret_code}, Msg: {ret_msg}, Extra: {ext_info}")
                 self.logger.debug(f"Params sent: {params}")
                 return False
        else:
             # safe_ccxt_call already logged the failure reason (e.g., network error, auth error)
             self.logger.error(f"API call failed for setting protection on {symbol}.")
             return False


# --- Trading Strategy Analyzer ---

class TradingAnalyzer:
    """Analyzes market data using technical indicators to generate trading signals."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.logger = logger
        self.config = config.get("indicator_settings", {})
        self.weights = self.config.get("signal_weights", {})
        if not self.weights:
            self.logger.warning("No 'signal_weights' found in indicator_settings. Using default weights.")
            # Ensure default weights are Decimals
            self.weights = {
                "rsi": Decimal("0.3"),
                "macd": Decimal("0.4"),
                "ema_cross": Decimal("0.3"),
            }
        else:
            # Ensure loaded weights are Decimals (config loader should handle this)
            self.weights = {k: Decimal(str(v)) for k, v in self.weights.items() if isinstance(v, (str, int, float, Decimal))}

        # Normalize weights if they don't sum to 1
        total_weight = sum(self.weights.values())
        # Use a small tolerance for comparison due to potential floating point inaccuracies during loading/summing
        if abs(total_weight - Decimal("1.0")) > Decimal("1e-9"):
             self.logger.warning(f"Signal weights sum to {total_weight}, not 1. Normalizing.")
             if total_weight == Decimal(0):
                  self.logger.error("Total signal weight is zero, cannot normalize. Disabling weighted signals.")
                  # Handle this case: maybe equal weights, or disable signals?
                  # For now, keep weights as they are but log error. Weighted sum will be 0.
             else:
                  try:
                       self.weights = {k: v / total_weight for k, v in self.weights.items()}
                       self.logger.info(f"Normalized weights: {self.weights}")
                  except InvalidOperation:
                       self.logger.error("Error normalizing weights (division by zero or invalid operation). Disabling weighted signals.")
                       self.weights = {}


    def calculate_indicators(self, ohlcv_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculates all configured technical indicators using pandas_ta."""
        if ohlcv_df is None or ohlcv_df.empty:
            self.logger.error("Cannot calculate indicators: OHLCV data is missing or empty.")
            return None

        # Ensure required columns exist and have Decimal type (or can be converted)
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in ohlcv_df.columns for col in required_cols):
             self.logger.error(f"Missing required OHLCV columns in DataFrame: {required_cols}")
             return None
        # Verify first row data types (assuming consistency)
        first_row = ohlcv_df.iloc[0]
        if not all(isinstance(first_row[col], Decimal) for col in required_cols):
             # Attempt conversion if they are numeric strings or numbers
             self.logger.warning("OHLCV columns are not all Decimal type. Attempting conversion.")
             try:
                  for col in required_cols:
                       ohlcv_df[col] = ohlcv_df[col].apply(lambda x: Decimal(str(x)) if x is not None else Decimal('NaN'))
             except (InvalidOperation, TypeError, ValueError) as e:
                  self.logger.error(f"Failed to convert OHLCV columns to Decimal: {e}. Cannot calculate indicators.")
                  return None


        self.logger.debug(f"Calculating indicators for {len(ohlcv_df)} candles...")
        # Work on a copy to avoid modifying the original DataFrame passed to the function
        df = ohlcv_df.copy()

        # Convert Decimal columns to float for pandas_ta compatibility
        # Handle potential non-finite Decimals (NaN, Inf) before converting
        float_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in float_cols:
            try:
                # Replace non-finite Decimals with NaN before converting to float
                df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) and x.is_finite() else pd.NA)
                # Ensure the column is numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                 self.logger.error(f"Error converting column {col} to float for TA calculation: {e}", exc_info=True)
                 return None # Cannot proceed if conversion fails

        # Drop rows with NaN in essential OHLC columns after conversion, as TA libs often fail
        initial_len = len(df)
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        if len(df) < initial_len:
             self.logger.warning(f"Dropped {initial_len - len(df)} rows with NaN in OHLC columns before TA calculation.")

        if df.empty:
             self.logger.error("DataFrame is empty after handling NaNs in OHLC data. Cannot calculate indicators.")
             return None

        try:
            # --- Calculate Indicators using pandas_ta ---
            # Ensure periods are integers from config
            rsi_period = self.config.get("rsi_period")
            if isinstance(rsi_period, int) and rsi_period > 0:
                df.ta.rsi(length=rsi_period, append=True, col_names=(f"RSI_{rsi_period}"))

            macd_fast = self.config.get("macd_fast")
            macd_slow = self.config.get("macd_slow")
            macd_signal = self.config.get("macd_signal")
            if all(isinstance(p, int) and p > 0 for p in [macd_fast, macd_slow, macd_signal]):
                # Generate default MACD column names (e.g., MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9)
                df.ta.macd(fast=macd_fast, slow=macd_slow, signal=macd_signal, append=True)

            ema_short_period = self.config.get("ema_short_period")
            if isinstance(ema_short_period, int) and ema_short_period > 0:
                df.ta.ema(length=ema_short_period, append=True, col_names=(f"EMA_{ema_short_period}"))

            ema_long_period = self.config.get("ema_long_period")
            if isinstance(ema_long_period, int) and ema_long_period > 0:
                df.ta.ema(length=ema_long_period, append=True, col_names=(f"EMA_{ema_long_period}"))

            atr_period = self.config.get("atr_period")
            if isinstance(atr_period, int) and atr_period > 0:
                # pandas-ta typically names ATR 'ATR_X'. ATRr uses RMA smoothing. Default is SMA.
                # Using RMA (similar to Wilder's) is common for ATR in trading systems.
                df.ta.atr(length=atr_period, append=True, mamode='rma', col_names=(f"ATR_{atr_period}")) # Use RMA, name it ATR_X

            # Add other indicators as needed...
            # Example: Bollinger Bands
            # bbands_period = self.config.get("bbands_period")
            # bbands_stddev = self.config.get("bbands_stddev")
            # if isinstance(bbands_period, int) and bbands_period > 0 and isinstance(bbands_stddev, (int, float, Decimal)) and Decimal(str(bbands_stddev)) > 0:
            #     df.ta.bbands(length=bbands_period, std=float(bbands_stddev), append=True)

            self.logger.debug(f"Indicators calculated. Columns added: {df.columns.difference(ohlcv_df.columns).tolist()}")

            # --- Convert calculated indicator columns back to Decimal ---
            # Identify newly added columns by pandas_ta
            original_cols = set(ohlcv_df.columns) | set(float_cols) # Include original + float-converted cols
            new_indicator_cols = [col for col in df.columns if col not in original_cols]

            for col in new_indicator_cols:
                 if pd.api.types.is_numeric_dtype(df[col]):
                     # Convert float indicators back to Decimal for internal consistency
                     # Handle potential NaN/Inf from calculations using math.isfinite
                     df[col] = df[col].apply(
                         lambda x: Decimal(str(x)) if pd.notna(x) and isinstance(x, (float, int)) and math.isfinite(x) else Decimal('NaN')
                     )

            # Restore original Decimal types for OHLCV columns by merging back or reassigning
            # This ensures we keep the original precision and Decimal type for price/volume data
            for col in required_cols: # Use the original OHLCV columns
                 if col in ohlcv_df.columns:
                     # Assign back the original Decimal column to the calculated DataFrame
                     df[col] = ohlcv_df[col]

            # Reindex df to match the original ohlcv_df index to include rows dropped earlier
            # This ensures the output DataFrame has the same index as the input OHLCV data
            df = df.reindex(ohlcv_df.index)

            return df

        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}", exc_info=True)
            return None

    def generate_signal(self, indicators_df: pd.DataFrame) -> Tuple[Signal, Dict[str, Any]]:
        """
        Generates a trading signal based on the latest indicator values
        and configured weighted scoring.
        Returns the final signal and contributing factors/scores.
        """
        if indicators_df is None or indicators_df.empty:
            self.logger.warning("Cannot generate signal: Indicators data is missing or empty.")
            return Signal.HOLD, {}

        try:
            # Get the absolute latest row based on index
            latest_data = indicators_df.iloc[-1]
        except IndexError:
             self.logger.error("Cannot get latest data: Indicators DataFrame is unexpectedly empty.")
             return Signal.HOLD, {}


        scores = {}
        contributing_factors = {}

        # --- Evaluate Individual Indicators ---
        # Ensure values are valid finite Decimals before comparison

        # RSI Example
        rsi_weight = self.weights.get("rsi", Decimal(0))
        rsi_period = self.config.get("rsi_period")
        # Use the exact column name generated in calculate_indicators
        rsi_col = f"RSI_{rsi_period}" if isinstance(rsi_period, int) else None
        if rsi_weight > 0 and rsi_col and rsi_col in latest_data:
            rsi_value = latest_data[rsi_col]
            if isinstance(rsi_value, Decimal) and rsi_value.is_finite():
                # Ensure thresholds are Decimals (config loader should handle this)
                overbought = self.config.get("rsi_overbought", Decimal("70"))
                oversold = self.config.get("rsi_oversold", Decimal("30"))
                rsi_score = Decimal(0)
                if rsi_value > overbought:
                    rsi_score = Decimal("-1") # Sell signal
                elif rsi_value < oversold:
                    rsi_score = Decimal("1") # Buy signal
                scores["rsi"] = rsi_score * rsi_weight
                contributing_factors["rsi"] = {"value": rsi_value, "score": rsi_score, "weight": rsi_weight}
            else:
                 self.logger.warning(f"Invalid or non-finite RSI value ({rsi_value}) for signal generation. Skipping RSI score.")


        # MACD Example (Histogram sign)
        macd_weight = self.weights.get("macd", Decimal(0))
        macd_fast = self.config.get("macd_fast", 12)
        macd_slow = self.config.get("macd_slow", 26)
        macd_signal_p = self.config.get("macd_signal", 9)
        # Use the exact column name generated by pandas_ta (e.g., MACDh_12_26_9)
        macdh_col = f"MACDh_{macd_fast}_{macd_slow}_{macd_signal_p}" if all(isinstance(p, int) for p in [macd_fast, macd_slow, macd_signal_p]) else None

        if macd_weight > 0 and macdh_col and macdh_col in latest_data:
            macd_hist = latest_data[macdh_col]
            if isinstance(macd_hist, Decimal) and macd_hist.is_finite():
                # Simple histogram sign check for cross direction
                macd_score = Decimal(0)
                # Add threshold to avoid noise near zero cross? Ensure it's Decimal.
                hist_threshold = self.config.get("macd_hist_threshold", Decimal("0"))
                if macd_hist > hist_threshold: # MACD line crossed above signal line (or is positive)
                    macd_score = Decimal("1") # Buy signal
                elif macd_hist < -hist_threshold: # MACD line crossed below signal line (or is negative)
                    macd_score = Decimal("-1") # Sell signal
                scores["macd"] = macd_score * macd_weight
                contributing_factors["macd"] = {"histogram": macd_hist, "score": macd_score, "weight": macd_weight}
            else:
                 self.logger.warning(f"Invalid or non-finite MACD Histogram value ({macd_hist}) for signal generation. Skipping MACD score.")

        # EMA Cross Example
        ema_cross_weight = self.weights.get("ema_cross", Decimal(0))
        ema_short_period = self.config.get("ema_short_period")
        ema_long_period = self.config.get("ema_long_period")
        # Use exact column names
        ema_short_col = f"EMA_{ema_short_period}" if isinstance(ema_short_period, int) else None
        ema_long_col = f"EMA_{ema_long_period}" if isinstance(ema_long_period, int) else None

        if ema_cross_weight > 0 and ema_short_col and ema_long_col and all(c in latest_data for c in [ema_short_col, ema_long_col]):
            ema_short = latest_data[ema_short_col]
            ema_long = latest_data[ema_long_col]
            if isinstance(ema_short, Decimal) and ema_short.is_finite() and isinstance(ema_long, Decimal) and ema_long.is_finite():
                ema_cross_score = Decimal(0)
                # Simple check: short > long is bullish
                # More robust: check previous candle's state for actual cross confirmation
                # prev_ema_short = indicators_df[ema_short_col].iloc[-2]
                # prev_ema_long = indicators_df[ema_long_col].iloc[-2]
                # if ema_short > ema_long and prev_ema_short <= prev_ema_long: # Bullish cross
                #     ema_cross_score = Decimal("1")
                # elif ema_short < ema_long and prev_ema_short >= prev_ema_long: # Bearish cross
                #     ema_cross_score = Decimal("-1")
                # Simple state check:
                if ema_short > ema_long:
                    ema_cross_score = Decimal("1") # Bullish state
                elif ema_short < ema_long:
                    ema_cross_score = Decimal("-1") # Bearish state

                scores["ema_cross"] = ema_cross_score * ema_cross_weight
                contributing_factors["ema_cross"] = {"short_ema": ema_short, "long_ema": ema_long, "score": ema_cross_score, "weight": ema_cross_weight}
            else:
                 self.logger.warning(f"Invalid or non-finite EMA values (Short={ema_short}, Long={ema_long}) for signal generation. Skipping EMA Cross score.")

        # --- Combine Scores ---
        if not scores:
             self.logger.warning("No valid indicator scores generated. Defaulting to HOLD.")
             return Signal.HOLD, {"final_score": Decimal(0), "factors": contributing_factors}

        final_score = sum(scores.values())

        # --- Determine Final Signal ---
        # Ensure thresholds are Decimals (config loader should handle this)
        strong_buy_threshold = self.config.get("strong_buy_threshold", Decimal("0.7"))
        buy_threshold = self.config.get("buy_threshold", Decimal("0.2"))
        sell_threshold = self.config.get("sell_threshold", Decimal("-0.2"))
        strong_sell_threshold = self.config.get("strong_sell_threshold", Decimal("-0.7"))

        # Validate thresholds are comparable
        if not (strong_sell_threshold < sell_threshold < buy_threshold < strong_buy_threshold):
             self.logger.error("Signal thresholds are improperly configured (overlapping or wrong order). Using default HOLD.")
             return Signal.HOLD, {"final_score": final_score, "factors": contributing_factors}


        signal = Signal.HOLD
        if final_score >= strong_buy_threshold:
            signal = Signal.STRONG_BUY
        elif final_score >= buy_threshold:
            signal = Signal.BUY
        elif final_score <= strong_sell_threshold:
            signal = Signal.STRONG_SELL
        elif final_score <= sell_threshold:
            signal = Signal.SELL

        # Quantize score for logging clarity
        quantized_score = final_score.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        self.logger.info(f"Signal generated: {signal.name} (Score: {quantized_score})")
        self.logger.debug(f"Contributing factors: {contributing_factors}")
        # Convert factors to strings for saving state if needed, or keep as Decimal
        signal_details_out = {"final_score": quantized_score, "factors": contributing_factors}

        return signal, signal_details_out


# --- Position and Risk Management ---

class PositionManager:
    """Handles position sizing, stop-loss, take-profit, and exit logic."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, exchange_wrapper: BybitV5Wrapper):
        self.logger = logger
        self.config = config
        self.risk_config = config.get("risk_management", {})
        self.trading_config = config.get("trading_settings", {})
        self.indicator_config = config.get("indicator_settings", {}) # Cache for convenience
        self.exchange = exchange_wrapper
        self.symbol = self.trading_config.get("symbol")
        # Ensure symbol is set
        if not self.symbol:
             logger.critical("Trading symbol is not defined in config. Cannot initialize PositionManager.")
             raise ValueError("Trading symbol is required.")
        self.hedge_mode = self.trading_config.get("hedge_mode", False)

    def get_base_quote(self) -> Tuple[Optional[str], Optional[str]]:
        """Gets base and quote assets from the market data for the symbol."""
        market = self.exchange.get_market(self.symbol)
        if market:
            base = market.get('base')
            quote = market.get('quote')
            if base and quote:
                 return base, quote
            else:
                 self.logger.error(f"Base or Quote asset missing in market data for {self.symbol}.")
                 return None, None
        self.logger.error(f"Failed to get market data for {self.symbol} to determine base/quote.")
        return None, None

    def calculate_position_size(self, entry_price: Decimal, stop_loss_price: Decimal,
                                available_equity: Decimal, quote_asset: str) -> Optional[Decimal]:
        """
        Calculates position size based on risk percentage of equity and stop distance.
        Returns size in base currency (e.g., BTC amount for BTC/USDT), adjusted for market limits/precision.
        """
        # Ensure risk percent is Decimal and within valid range (config loader should handle type)
        risk_percent_config = self.risk_config.get("risk_per_trade_percent", Decimal("1.0"))
        if not isinstance(risk_percent_config, Decimal) or not (Decimal(0) < risk_percent_config <= Decimal(100)):
             self.logger.error(f"Invalid risk_per_trade_percent configuration: {risk_percent_config}. Using default 1%.")
             risk_percent = Decimal("0.01") # Default to 1% risk
        else:
             risk_percent = risk_percent_config / Decimal("100")

        # Leverage is applied by the exchange, not directly in size calculation based on risk % of equity.
        # leverage = self.trading_config.get("leverage", Decimal("1"))

        # Validate inputs
        if not isinstance(entry_price, Decimal) or not entry_price.is_finite() or entry_price <= 0:
             self.logger.error(f"Invalid entry_price for size calculation: {entry_price}")
             return None
        if not isinstance(stop_loss_price, Decimal) or not stop_loss_price.is_finite() or stop_loss_price <= 0:
             self.logger.error(f"Invalid stop_loss_price for size calculation: {stop_loss_price}")
             return None
        if not isinstance(available_equity, Decimal) or not available_equity.is_finite() or available_equity <= 0:
            self.logger.error(f"Invalid available_equity for size calculation: {available_equity}")
            return None
        if entry_price == stop_loss_price:
             self.logger.error("Entry price and stop-loss price cannot be the same for size calculation.")
             return None

        market = self.exchange.get_market(self.symbol)
        if not market:
            self.logger.error(f"Cannot calculate position size: Market data for {self.symbol} not found.")
            return None
        base_asset, market_quote_asset = self.get_base_quote()
        if not base_asset or not market_quote_asset:
             self.logger.error(f"Could not determine base/quote asset for {self.symbol}. Cannot calculate size.")
             return None
        # Verify configured quote asset matches market quote asset
        if market_quote_asset != quote_asset:
             self.logger.warning(f"Configured quote_asset '{quote_asset}' differs from market quote '{market_quote_asset}'. Using market quote '{market_quote_asset}' for calculations.")
             # Use the actual market quote asset for calculations
             quote_asset = market_quote_asset

        # Amount of quote currency to risk based on total equity
        risk_amount_quote = available_equity * risk_percent
        # Quantize risk amount for logging clarity (optional)
        quantized_equity = available_equity.quantize(Decimal('0.01'), rounding=ROUND_DOWN)
        quantized_risk_amount = risk_amount_quote.quantize(Decimal('0.01'), rounding=ROUND_DOWN)
        self.logger.info(f"Risk per trade: {risk_percent:.2%}, Total Equity ({quote_asset}): {quantized_equity}, Risk Amount: {quantized_risk_amount} {quote_asset}")

        # Price distance for stop loss (absolute value)
        stop_loss_distance = abs(entry_price - stop_loss_price)
        if stop_loss_distance <= Decimal(0):
             # This should be caught by entry_price == stop_loss_price check, but double-check
             self.logger.error(f"Stop loss distance is zero or negative ({stop_loss_distance}). Check SL price relative to entry.")
             return None

        # --- Calculate position size in base currency ---
        position_size_base = None
        contract_type = 'linear' # Default assumption
        if market.get('linear', False): contract_type = 'linear'
        elif market.get('inverse', False): contract_type = 'inverse'
        elif market.get('spot', False): contract_type = 'spot'
        else:
             # Try inferring from symbol or market info if flags are missing
             if market.get('type') == 'swap' and market.get('settle', '').upper() == market.get('base', '').upper():
                  contract_type = 'inverse'
             elif market.get('type') == 'swap':
                  contract_type = 'linear'
             elif market.get('type') == 'spot':
                  contract_type = 'spot'
             else:
                  self.logger.warning(f"Could not reliably determine contract type (linear/inverse/spot) for {self.symbol}. Assuming linear.")
                  contract_type = 'linear'


        if contract_type == 'inverse':
             # Inverse: Size_Base = (Risk Amount in Quote * Entry Price) / Stop Distance
             # PnL = Size_Base * ContractSize * (1/Exit - 1/Entry)
             # Risk_Quote = Size_Base * ContractSize * abs(1/SL - 1/Entry) * SL_Price (approx)
             # More accurately: Risk_Quote = Size_Base * ContractSize * abs(Entry - SL) / Entry
             # Size_Base = Risk_Quote * Entry / (ContractSize * Stop_Distance)
             contract_size = market.get('contractSize', Decimal('1.0')) # Default to 1 if not specified
             if contract_size is None or contract_size <= 0:
                  self.logger.error(f"Invalid contract size {contract_size} for inverse contract {self.symbol}. Cannot calculate size.")
                  return None
             position_size_base = (risk_amount_quote * entry_price) / (contract_size * stop_loss_distance)
             self.logger.debug(f"Calculating size for INVERSE contract (ContractSize: {contract_size}).")
        else: # Linear or Spot
             # Linear/Spot: Size_Base = Risk Amount in Quote / Stop Loss Distance in Quote per Base Unit
             # PnL = Size_Base * ContractSize * (Exit - Entry)
             # Risk_Quote = Size_Base * ContractSize * Stop_Distance
             # Size_Base = Risk_Quote / (ContractSize * Stop_Distance)
             contract_size = market.get('contractSize', Decimal('1.0')) # Default to 1 for spot/linear if not specified
             if contract_size is None or contract_size <= 0:
                  self.logger.error(f"Invalid contract size {contract_size} for {contract_type} contract {self.symbol}. Cannot calculate size.")
                  return None
             position_size_base = risk_amount_quote / (contract_size * stop_loss_distance)
             self.logger.debug(f"Calculating size for {contract_type.upper()} contract (ContractSize: {contract_size}).")

        if position_size_base is None or not position_size_base.is_finite() or position_size_base <= 0:
             self.logger.error(f"Calculated position size is invalid (zero, negative, or non-finite): {position_size_base}")
             return None

        self.logger.debug(f"Calculated raw position size: {position_size_base:.{DECIMAL_PRECISION}f} {base_asset}")

        # --- Apply Precision, Lot Size, Min/Max Limits ---
        try:
            # Get min/max order size limits from market data
            limits_amount = market.get('limits', {}).get('amount', {})
            min_amount = limits_amount.get('min')
            max_amount = limits_amount.get('max')

            # Convert limits to Decimal if they exist
            min_amount_dec = Decimal(str(min_amount)) if min_amount is not None else None
            max_amount_dec = Decimal(str(max_amount)) if max_amount is not None else None

            # 1. Check against minimum amount BEFORE final formatting
            if min_amount_dec is not None and position_size_base < min_amount_dec:
                self.logger.warning(f"Calculated position size {position_size_base:.{DECIMAL_PRECISION}f} {base_asset} is below minimum order size {min_amount_dec} {base_asset}. Cannot open position with this risk.")
                return None

            # 2. Apply amount precision using the exchange wrapper's formatter
            # This handles rounding/truncation according to exchange rules (step size / lot size)
            formatted_size_str = self.exchange.format_value_for_api(self.symbol, 'amount', position_size_base)
            final_size_base = Decimal(formatted_size_str)
            self.logger.debug(f"Position size after applying precision/step size: {final_size_base} {base_asset}")


            # 3. Re-check minimum amount AFTER precision adjustment
            # The formatted size might be slightly smaller or larger
            if min_amount_dec is not None and final_size_base < min_amount_dec:
                self.logger.warning(f"Position size {final_size_base} after precision adjustment is below minimum {min_amount_dec}. Cannot open position.")
                # Option: Could try increasing size to meet minimum, but this increases risk. Safer to abort.
                return None

            # 4. Check against maximum amount
            if max_amount_dec is not None and final_size_base > max_amount_dec:
                self.logger.warning(f"Calculated position size {final_size_base} exceeds max limit {max_amount_dec}. Capping size to max limit.")
                # Cap the size to the maximum allowed
                # Re-apply formatting to ensure the capped value adheres to precision/step rules
                formatted_max_size_str = self.exchange.format_value_for_api(self.symbol, 'amount', max_amount_dec)
                final_size_base = Decimal(formatted_max_size_str)
                self.logger.info(f"Position size capped and formatted to: {final_size_base} {base_asset}")

            # 5. Final check: Ensure size is still positive after all adjustments
            if final_size_base <= 0:
                 self.logger.error(f"Final position size is zero or negative ({final_size_base}) after adjustments. Cannot open position.")
                 return None

            self.logger.info(f"Calculated Final Position Size: {final_size_base} {base_asset}")
            return final_size_base

        except ValueError as e:
             # Errors from format_value_for_api or Decimal conversion of limits
             self.logger.error(f"Error applying market limits/precision to position size: {e}")
             return None
        except Exception as e:
             self.logger.error(f"Unexpected error during position size finalization: {e}", exc_info=True)
             return None


    def quantize_price(self, price: Decimal, side_for_conservative_rounding: Optional[PositionSide] = None) -> Optional[Decimal]:
         """
         Quantizes price according to market precision rules (tick size).
         Optionally applies conservative rounding for SL/TP based on side.
         """
         if not isinstance(price, Decimal):
              self.logger.error(f"Invalid price type for quantization: {type(price)}. Expected Decimal.")
              return None
         if not price.is_finite():
              self.logger.error(f"Invalid price value for quantization: {price}. Must be finite.")
              return None

         market = self.exchange.get_market(self.symbol)
         if not market:
              self.logger.error(f"Cannot quantize price: Market data not found for {self.symbol}.")
              return None
         if 'precision' not in market or market['precision'] is None or 'price' not in market['precision']:
              self.logger.error(f"Cannot quantize price: Market price precision (tick size) not found for {self.symbol}.")
              # Fallback: return unquantized price with warning? Or fail? Let's fail for safety.
              return None

         try:
              # Use the exchange's price_to_precision method first - this should handle tick size rounding
              # Pass price as float, method returns string
              price_str = self.exchange.price_to_precision(self.symbol, float(price))
              quantized_price = Decimal(price_str)

              # --- Optional: Apply explicit conservative rounding for SL/TP ---
              # This might be redundant if price_to_precision with appropriate rounding mode is used by CCXT,
              # but adds an explicit safety layer. Requires tick_size.
              # tick_size_str = market.get('precision', {}).get('price')
              # if tick_size_str is not None and side_for_conservative_rounding is not None:
              #      try:
              #           tick_size = Decimal(str(tick_size_str))
              #           if tick_size > 0: # Ensure tick_size is valid
              #                # Determine rounding direction based on SL/TP and side
              #                # Assuming 'price' is an SL here based on typical usage context
              #                if side_for_conservative_rounding == PositionSide.LONG: # Round down Long SL
              #                     quantized_price = (price // tick_size) * tick_size
              #                elif side_for_conservative_rounding == PositionSide.SHORT: # Round up Short SL
              #                     # Ceiling division: math.ceil(price / tick_size) * tick_size
              #                     # Decimal equivalent: use ROUND_UP or custom logic
              #                     quantized_price = (price / tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * tick_size
              #                # Add similar logic for TP if needed (round up Long TP, round down Short TP)
              #                self.logger.debug(f"Applied conservative rounding for {side_for_conservative_rounding.name} SL: {price} -> {quantized_price}")
              #      except (InvalidOperation, ValueError) as e:
              #           self.logger.warning(f"Could not apply conservative rounding: Invalid tick size '{tick_size_str}'? Error: {e}")
              #           # Fallback to the price_to_precision result
              #           quantized_price = Decimal(price_str)

              return quantized_price

         except (ValueError, InvalidOperation) as e:
              # Error from price_to_precision or Decimal conversion
              self.logger.error(f"Error quantizing price {price} for {self.symbol}: {e}", exc_info=True)
              return None
         except Exception as e:
              self.logger.error(f"Unexpected error quantizing price {price} for {self.symbol}: {e}", exc_info=True)
              return None


    def calculate_stop_loss(self, entry_price: Decimal, side: PositionSide, latest_indicators: pd.Series) -> Optional[Decimal]:
        """Calculates the initial stop loss price based on configured method (ATR or fixed %)."""
        if not isinstance(entry_price, Decimal) or not entry_price.is_finite() or entry_price <= 0:
             self.logger.error(f"Invalid entry price for SL calculation: {entry_price}")
             return None

        sl_method = self.risk_config.get("stop_loss_method", "atr").lower()
        stop_loss_price = None

        if sl_method == "atr":
            atr_multiplier = self.risk_config.get("atr_multiplier", Decimal("1.5"))
            if not isinstance(atr_multiplier, Decimal) or not atr_multiplier.is_finite() or atr_multiplier <= 0:
                 self.logger.error(f"Invalid atr_multiplier: {atr_multiplier}. Must be positive finite Decimal.")
                 return None

            atr_period = self.indicator_config.get("atr_period", 14)
            # Use the exact column name generated in calculate_indicators (e.g., ATR_14)
            atr_col = f"ATR_{atr_period}" if isinstance(atr_period, int) else None

            if not atr_col:
                 self.logger.error("ATR period not configured correctly or is not an integer.")
                 return None
            if atr_col not in latest_indicators:
                 self.logger.error(f"ATR column '{atr_col}' not found in latest indicators.")
                 return None

            atr_value = latest_indicators[atr_col]
            if not isinstance(atr_value, Decimal) or not atr_value.is_finite() or atr_value <= 0:
                self.logger.error(f"Cannot calculate ATR stop loss: Invalid or non-positive ATR value ({atr_value}).")
                return None

            stop_distance = atr_value * atr_multiplier
            if side == PositionSide.LONG:
                stop_loss_price = entry_price - stop_distance
            elif side == PositionSide.SHORT:
                stop_loss_price = entry_price + stop_distance
            else: # Should not happen if called correctly
                 self.logger.error("Invalid position side for SL calculation.")
                 return None
            self.logger.debug(f"Calculated SL based on ATR: Entry={entry_price}, Side={side.name}, ATR={atr_value}, Multiplier={atr_multiplier}, Distance={stop_distance}, Raw SL={stop_loss_price}")

        elif sl_method == "fixed_percent":
            fixed_percent_config = self.risk_config.get("fixed_stop_loss_percent", Decimal("2.0"))
            if not isinstance(fixed_percent_config, Decimal) or not fixed_percent_config.is_finite() or not (Decimal(0) < fixed_percent_config < Decimal(100)):
                 self.logger.error(f"Invalid fixed_stop_loss_percent: {fixed_percent_config}%. Must be between 0 and 100 (exclusive).")
                 return None
            fixed_percent = fixed_percent_config / Decimal("100")

            if side == PositionSide.LONG:
                stop_loss_price = entry_price * (Decimal("1") - fixed_percent)
            elif side == PositionSide.SHORT:
                stop_loss_price = entry_price * (Decimal("1") + fixed_percent)
            else:
                 self.logger.error("Invalid position side for SL calculation.")
                 return None
            self.logger.debug(f"Calculated SL based on Fixed Percent: Entry={entry_price}, Side={side.name}, Percent={fixed_percent*100}%, Raw SL={stop_loss_price}")

        else:
            self.logger.error(f"Unknown stop loss method configured: {sl_method}")
            return None

        # Basic validation of calculated price
        if stop_loss_price is None or not stop_loss_price.is_finite() or stop_loss_price <= 0:
             self.logger.error(f"Calculated stop loss price ({stop_loss_price}) is invalid (zero, negative, or non-finite). Cannot set SL.")
             return None

        # Quantize SL price to market precision (pass side for potential conservative rounding)
        quantized_sl = self.quantize_price(stop_loss_price, side_for_conservative_rounding=side)

        if quantized_sl is None:
             self.logger.error("Failed to quantize calculated stop loss price.")
             return None

        # Final check: Ensure quantized SL didn't cross entry price due to rounding/quantization
        if side == PositionSide.LONG and quantized_sl >= entry_price:
             self.logger.error(f"Quantized SL {quantized_sl} is >= entry price {entry_price}. Invalid SL. Check ATR/percentage values or market tick size.")
             # Option: Adjust SL slightly further away? Requires careful thought. For now, fail.
             # Example adjustment: quantized_sl = self.quantize_price(quantized_sl - tick_size, side)
             return None
        if side == PositionSide.SHORT and quantized_sl <= entry_price:
             self.logger.error(f"Quantized SL {quantized_sl} is <= entry price {entry_price}. Invalid SL. Check ATR/percentage values or market tick size.")
             # Example adjustment: quantized_sl = self.quantize_price(quantized_sl + tick_size, side)
             return None


        self.logger.info(f"Calculated Initial Stop Loss Price (Quantized): {quantized_sl}")
        return quantized_sl


    def check_ma_cross_exit(self, latest_indicators: pd.Series, position_side: PositionSide) -> bool:
        """Checks if an MA cross exit condition is met based on config."""
        if not self.trading_config.get("use_ma_cross_exit", False):
            return False # Feature not enabled

        ema_short_period = self.indicator_config.get("ema_short_period")
        ema_long_period = self.indicator_config.get("ema_long_period")
        # Use exact column names
        ema_short_col = f"EMA_{ema_short_period}" if isinstance(ema_short_period, int) else None
        ema_long_col = f"EMA_{ema_long_period}" if isinstance(ema_long_period, int) else None


        if not ema_short_col or not ema_long_col:
             self.logger.warning("MA cross exit enabled, but EMA periods not configured correctly.")
             return False

        if not all(c in latest_indicators for c in [ema_short_col, ema_long_col]):
            self.logger.warning(f"Cannot check MA cross exit: EMA columns ({ema_short_col}, {ema_long_col}) not available in latest indicators.")
            return False

        ema_short = latest_indicators[ema_short_col]
        ema_long = latest_indicators[ema_long_col]

        # Check if EMA values are valid Decimals
        if not isinstance(ema_short, Decimal) or not ema_short.is_finite() or \
           not isinstance(ema_long, Decimal) or not ema_long.is_finite():
            self.logger.warning(f"Invalid or non-finite EMA values for MA cross exit check: Short={ema_short}, Long={ema_long}")
            return False

        # Check for bearish cross for long position exit, bullish cross for short position exit
        exit_signal = False
        if position_side == PositionSide.LONG and ema_short < ema_long:
            # More robust: Check if previous state was ema_short >= ema_long for actual cross
            self.logger.info("MA Cross Exit triggered for LONG position (Short EMA crossed below Long EMA).")
            exit_signal = True
        elif position_side == PositionSide.SHORT and ema_short > ema_long:
            # More robust: Check if previous state was ema_short <= ema_long
            self.logger.info("MA Cross Exit triggered for SHORT position (Short EMA crossed above Long EMA).")
            exit_signal = True

        return exit_signal


    def manage_stop_loss(self, position: Dict[str, Any], latest_indicators: pd.Series, current_state: Dict[str, Any]) -> Optional[Decimal]:
        """
        Manages stop loss adjustments (Break-Even, Trailing).
        Returns the new SL price (Decimal) if an update is needed and valid, otherwise None.
        Requires 'active_position' and 'stop_loss_price' in current_state.
        Uses Decimal types for calculations and comparisons.
        """
        # Ensure position data and state are valid
        if not position or position.get('side') == 'none' or position.get('side') is None:
            self.logger.debug("No active position or side is None, cannot manage SL.")
            return None
        if not current_state.get('active_position'):
             self.logger.warning("manage_stop_loss called but state has no active position. Sync issue?")
             return None

        # Extract necessary data, ensuring they are Decimals
        entry_price = position.get('entryPrice') # Should be Decimal from fetch_positions processing
        current_sl_state = current_state.get('stop_loss_price') # Should be Decimal from state
        position_side_str = position.get('side') # 'long' or 'short'
        mark_price = position.get('markPrice') # Should be Decimal

        # Validate necessary data types and values
        if not isinstance(entry_price, Decimal) or not entry_price.is_finite():
            self.logger.warning(f"Invalid or missing entry price ({entry_price}) for SL management.")
            return None
        if not isinstance(current_sl_state, Decimal) or not current_sl_state.is_finite():
            self.logger.warning(f"Invalid or missing current SL price in state ({current_sl_state}) for SL management.")
            # Cannot manage SL if we don't know the current one
            return None
        if not isinstance(mark_price, Decimal) or not mark_price.is_finite():
            self.logger.warning(f"Invalid or missing mark price ({mark_price}) for SL management.")
            return None
        if entry_price <= 0 or mark_price <= 0 or current_sl_state <= 0:
             self.logger.warning(f"Entry price ({entry_price}), mark price ({mark_price}), or current SL ({current_sl_state}) is non-positive. Cannot manage SL.")
             return None

        try:
            position_side = PositionSide(position_side_str)
        except ValueError:
             self.logger.error(f"Invalid position side string '{position_side_str}' in position data.")
             return None

        new_sl_price = None # Stores the proposed new SL price
        state_updated_this_cycle = False # Track if BE was set in this cycle to prevent TSL overriding it immediately

        # --- Get ATR value ---
        atr_value = None
        use_atr_based_sl = self.risk_config.get("use_break_even_sl", False) or self.risk_config.get("use_trailing_sl", False)
        if use_atr_based_sl:
            atr_period = self.indicator_config.get("atr_period", 14)
            # Use exact column name
            atr_col = f"ATR_{atr_period}" if isinstance(atr_period, int) else None

            if atr_col and atr_col in latest_indicators:
                 val = latest_indicators[atr_col]
                 if isinstance(val, Decimal) and val.is_finite() and val > 0:
                      atr_value = val
                 else:
                      self.logger.warning(f"ATR value ({val}) is invalid or non-positive. Cannot perform ATR-based SL management.")
            else:
                 self.logger.warning(f"ATR column '{atr_col}' not found or not configured. Cannot perform ATR-based SL management.")


        # 1. Break-Even Stop Loss
        use_be = self.risk_config.get("use_break_even_sl", False)
        if use_be and atr_value is not None and not current_state.get('break_even_achieved', False):
            be_trigger_atr = self.risk_config.get("break_even_trigger_atr", Decimal("1.0"))
            be_offset_atr = self.risk_config.get("break_even_offset_atr", Decimal("0.1")) # Small profit offset

            if not all(isinstance(d, Decimal) and d.is_finite() for d in [be_trigger_atr, be_offset_atr]):
                 self.logger.error("Invalid Break-Even ATR configuration (trigger/offset). Skipping BE check.")
            else:
                 profit_target_distance = atr_value * be_trigger_atr
                 offset_distance = atr_value * be_offset_atr
                 target_be_price = entry_price # Default BE is entry price

                 # Calculate target BE price with offset
                 if position_side == PositionSide.LONG:
                      target_be_price = entry_price + offset_distance # Target SL slightly above entry
                 elif position_side == PositionSide.SHORT:
                      target_be_price = entry_price - offset_distance # Target SL slightly below entry

                 # Quantize the target BE price
                 quantized_be_price = self.quantize_price(target_be_price, side_for_conservative_rounding=position_side)

                 if quantized_be_price is not None:
                      # Check if price reached the trigger level
                      be_triggered = False
                      if position_side == PositionSide.LONG:
                           profit_target_price = entry_price + profit_target_distance
                           if mark_price >= profit_target_price: be_triggered = True
                      elif position_side == PositionSide.SHORT:
                           profit_target_price = entry_price - profit_target_distance
                           if mark_price <= profit_target_price: be_triggered = True

                      if be_triggered:
                           # Check if the quantized BE price is actually better than the current SL
                           is_better = False
                           if position_side == PositionSide.LONG and quantized_be_price > current_sl_state: is_better = True
                           elif position_side == PositionSide.SHORT and quantized_be_price < current_sl_state: is_better = True

                           if is_better:
                                self.logger.info(f"Break-Even Triggered ({position_side.name}): Mark Price {mark_price} reached target. Proposing SL update from {current_sl_state} to BE price {quantized_be_price}")
                                new_sl_price = quantized_be_price
                                # Mark BE as intended to be achieved (state updated by caller on success)
                                # current_state['break_even_achieved'] = True # Don't modify state here
                                state_updated_this_cycle = True # Prevent TSL override this cycle
                           else:
                                self.logger.debug(f"BE triggered ({position_side.name}), but proposed BE price {quantized_be_price} is not better than current SL {current_sl_state}.")
                 else:
                      self.logger.warning("Failed to quantize break-even price. Skipping BE check.")


        # 2. Trailing Stop Loss (only if BE wasn't just set in this cycle)
        use_tsl = self.risk_config.get("use_trailing_sl", False)
        if use_tsl and atr_value is not None and not state_updated_this_cycle:
             tsl_atr_multiplier = self.risk_config.get("trailing_sl_atr_multiplier", Decimal("2.0"))
             if not isinstance(tsl_atr_multiplier, Decimal) or not tsl_atr_multiplier.is_finite() or tsl_atr_multiplier <= 0:
                  self.logger.error(f"Invalid trailing_sl_atr_multiplier: {tsl_atr_multiplier}. Skipping TSL check.")
             else:
                  trail_distance = atr_value * tsl_atr_multiplier
                  potential_tsl_price = None

                  if position_side == PositionSide.LONG:
                      potential_tsl_price = mark_price - trail_distance
                  elif position_side == PositionSide.SHORT:
                      potential_tsl_price = mark_price + trail_distance

                  if potential_tsl_price is not None:
                      # Quantize potential TSL price (conservatively)
                      quantized_tsl_price = self.quantize_price(potential_tsl_price, side_for_conservative_rounding=position_side)

                      if quantized_tsl_price is not None:
                           # Check if the quantized TSL price is better than the current SL
                           is_better = False
                           if position_side == PositionSide.LONG and quantized_tsl_price > current_sl_state: is_better = True
                           elif position_side == PositionSide.SHORT and quantized_tsl_price < current_sl_state: is_better = True

                           if is_better:
                                self.logger.debug(f"Trailing SL Update ({position_side.name}): Potential TSL {quantized_tsl_price} is better than Current SL {current_sl_state}")
                                # Final check: Ensure TSL doesn't move SL unfavorably after BE achieved
                                if current_state.get('break_even_achieved', False):
                                     # Re-quantize entry for comparison, handle potential failure
                                     quantized_entry = self.quantize_price(entry_price)
                                     if quantized_entry is not None:
                                          if position_side == PositionSide.LONG and quantized_tsl_price < quantized_entry:
                                               self.logger.warning(f"TSL calculation resulted in SL ({quantized_tsl_price}) below quantized entry ({quantized_entry}) after BE. Clamping SL to entry.")
                                               quantized_tsl_price = quantized_entry
                                          elif position_side == PositionSide.SHORT and quantized_tsl_price > quantized_entry:
                                               self.logger.warning(f"TSL calculation resulted in SL ({quantized_tsl_price}) above quantized entry ({quantized_entry}) after BE. Clamping SL to entry.")
                                               quantized_tsl_price = quantized_entry
                                     else:
                                          self.logger.warning("Could not quantize entry price for TSL vs BE check. Proceeding with calculated TSL.")

                                # Assign the potentially clamped TSL price
                                new_sl_price = quantized_tsl_price
                           # else: TSL is not better than current SL, do nothing
                      else:
                           self.logger.warning(f"Failed to quantize potential trailing SL price ({potential_tsl_price}). Skipping TSL update.")


        # --- Return the final new SL price if it's valid and different from current ---
        if new_sl_price is not None:
            # Final validation of the calculated new_sl_price
            if not new_sl_price.is_finite() or new_sl_price <= 0:
                 self.logger.warning(f"Calculated new SL price ({new_sl_price}) is invalid (non-finite or non-positive). Ignoring update.")
                 return None

            # Ensure SL is still valid relative to current mark price after all adjustments
            # Allow SL to be placed exactly at mark price if needed (e.g., immediate trail)
            if position_side == PositionSide.LONG and new_sl_price > mark_price:
                 self.logger.warning(f"Calculated new SL price {new_sl_price} is > current mark price {mark_price}. Invalid SL. Ignoring update.")
                 return None
            if position_side == PositionSide.SHORT and new_sl_price < mark_price:
                 self.logger.warning(f"Calculated new SL price {new_sl_price} is < current mark price {mark_price}. Invalid SL. Ignoring update.")
                 return None

            # Only return if the new SL is meaningfully different from the current one
            # Use a small tolerance based on price precision (tick size) if available
            price_tick_size = None
            market = self.exchange.get_market(self.symbol)
            if market and 'precision' in market and 'price' in market['precision']:
                 tick_str = market['precision']['price']
                 if tick_str:
                      try: price_tick_size = Decimal(str(tick_str))
                      except: pass

            # Use half tick size as tolerance, or a small default Decimal value
            tolerance = (price_tick_size / Decimal(2)) if price_tick_size else Decimal('1e-9') # Use smaller tolerance if tick unknown

            if abs(new_sl_price - current_sl_state) > tolerance:
                 self.logger.info(f"Proposing SL update from {current_sl_state} to {new_sl_price}")
                 return new_sl_price
            else:
                 self.logger.debug(f"Calculated new SL {new_sl_price} is not significantly different from current {current_sl_state} (tolerance: {tolerance}). No update needed.")
                 return None # No significant change needed

        return None # No update needed


# --- Main Trading Bot Class ---

class TradingBot:
    """
    The main trading bot class orchestrating the fetch-analyze-execute loop.
    Manages state, interacts with exchange wrapper, analyzer, and position manager.
    """
    def __init__(self, config_path: Path, state_path: Path):
        self.config_path = config_path
        self.state_path = state_path
        # Setup logger first, as other components depend on it
        self.logger = setup_logger()

        self.config = load_config(config_path, self.logger)
        if not self.config:
            self.logger.critical("Failed to load or validate configuration. Exiting.")
            sys.exit(1) # Exit if config is invalid

        # Apply log level from config if present and valid
        global LOG_LEVEL # Allow modification of the global default
        log_level_str = self.config.get("logging", {}).get("level", "INFO").upper()
        log_level_enum = getattr(logging, log_level_str, None)
        if isinstance(log_level_enum, int):
             # Check if level actually changed to avoid unnecessary logging
             if self.logger.level != log_level_enum:
                  LOG_LEVEL = log_level_enum # Update global for potential future logger creations
                  self.logger.setLevel(LOG_LEVEL)
                  # Update handlers' levels too
                  for handler in self.logger.handlers:
                      handler.setLevel(LOG_LEVEL)
                  self.logger.info(f"Log level set to {log_level_str} ({LOG_LEVEL}) from config.")
        else:
             self.logger.warning(f"Invalid log level '{log_level_str}' in config. Using default {logging.getLevelName(self.logger.level)}.")


        self.state = load_state(state_path, self.logger)
        # Ensure essential state keys exist with default values upon loading/initialization
        self.state.setdefault('active_position', None) # Stores dict of the current position managed by the bot
        self.state.setdefault('stop_loss_price', None) # Stores Decimal SL price set by the bot
        self.state.setdefault('take_profit_price', None) # Stores Decimal TP price set by the bot
        self.state.setdefault('break_even_achieved', False) # Flag for BE SL activation
        self.state.setdefault('last_order_id', None) # ID of the last order placed by the bot
        self.state.setdefault('last_sync_time', None) # Timestamp of last successful state sync

        try:
            self.exchange = BybitV5Wrapper(self.config, self.logger)
            self.analyzer = TradingAnalyzer(self.config, self.logger)
            self.position_manager = PositionManager(self.config, self.logger, self.exchange)
        except Exception as e:
             # Catch errors during component initialization (e.g., market loading, config issues)
             self.logger.critical(f"Failed to initialize core components: {e}. Exiting.", exc_info=True)
             sys.exit(1)


        # Extract frequently used config values for convenience and clarity
        self.symbol = self.config.get('trading_settings', {}).get('symbol')
        self.timeframe = self.config.get('trading_settings', {}).get('timeframe')
        self.leverage = self.config.get('trading_settings', {}).get('leverage') # Should be Decimal from loader
        self.quote_asset = self.config.get('trading_settings', {}).get('quote_asset')
        self.poll_interval = self.config.get('trading_settings', {}).get('poll_interval_seconds', 60)
        self.hedge_mode = self.config.get('trading_settings', {}).get('hedge_mode', False)

        # Validate essential settings are present after loading
        if not all([self.symbol, self.timeframe, self.leverage, self.quote_asset]):
             self.logger.critical("Essential trading settings (symbol, timeframe, leverage, quote_asset) missing in configuration. Exiting.")
             sys.exit(1)

        self.is_running = True # Flag to control the main loop

    def run(self):
        """Starts the main trading loop."""
        self.logger.info(f"--- Starting Trading Bot ---")
        self.logger.info(f"Symbol: {self.symbol}, Timeframe: {self.timeframe}, Quote: {self.quote_asset}, Leverage: {self.leverage}x, Hedge Mode: {self.hedge_mode}")
        self.logger.info(f"Poll Interval: {self.poll_interval}s")

        # Initialize exchange settings (leverage, margin mode) before starting loop
        if not self.initialize_exchange_settings():
             self.logger.critical("Failed to initialize exchange settings (e.g., leverage). Exiting.")
             sys.exit(1)

        # Initial sync before starting loop to align state with reality
        self.logger.info("Performing initial position sync...")
        initial_position = self.get_current_position()
        self.sync_bot_state_with_position(initial_position)
        save_state(self.state, self.state_path, self.logger) # Save potentially updated initial state


        while self.is_running:
            try:
                self.logger.info(f"--- New Trading Cycle ---")
                start_time = time.time()

                # 1. Fetch Market Data (OHLCV)
                # Fetch slightly more data than strictly needed by indicators for stability
                ohlcv_limit = self.config.get("indicator_settings", {}).get("ohlcv_fetch_limit", 250)
                ohlcv_data = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=ohlcv_limit)
                if ohlcv_data is None or ohlcv_data.empty:
                    self.logger.warning("Failed to fetch OHLCV data or data was empty. Skipping cycle.")
                    self._wait_for_next_cycle(start_time)
                    continue

                # 2. Calculate Indicators & Generate Signal
                indicators_df = self.analyzer.calculate_indicators(ohlcv_data)
                if indicators_df is None or indicators_df.empty:
                    self.logger.warning("Failed to calculate indicators or result was empty. Skipping cycle.")
                    self._wait_for_next_cycle(start_time)
                    continue
                try:
                     # Ensure indicators_df is not empty before accessing iloc[-1]
                     if indicators_df.empty:
                          raise IndexError("Indicators DataFrame is empty after calculation.")
                     latest_indicators = indicators_df.iloc[-1]
                except IndexError:
                     self.logger.warning("Indicators DataFrame is empty or calculation failed to produce rows. Skipping cycle.")
                     self._wait_for_next_cycle(start_time)
                     continue

                signal, signal_details = self.analyzer.generate_signal(indicators_df)

                # 3. Fetch Current State (Position) - Fetch again for latest data before decisions
                current_position = self.get_current_position() # Fetches live data from exchange

                # 4. Sync Bot State with Exchange State
                # This crucial step reconciles internal state with live position data
                self.sync_bot_state_with_position(current_position)

                # 5. Decision Making & Execution
                # Use the synced state ('active_position') to decide the next action
                if self.state.get('active_position'):
                    # Pass the fetched live position data (current_position) for accurate management
                    self.manage_existing_position(latest_indicators, signal, current_position)
                else:
                    # No active position according to state, check for entry signals
                    self.attempt_new_entry(signal, latest_indicators, signal_details)

                # 6. Save State
                # Save at the end of each cycle to persist the latest state,
                # even if no action was taken or if actions within manage/entry failed.
                save_state(self.state, self.state_path, self.logger)

                # 7. Wait for Next Cycle
                self._wait_for_next_cycle(start_time)

            except KeyboardInterrupt:
                self.logger.info("KeyboardInterrupt received. Stopping bot gracefully...")
                self.is_running = False
                # Perform any cleanup if needed (e.g., attempt to close open positions? cancel orders?)
                # self.perform_shutdown_cleanup()
            except ccxt.AuthenticationError:
                 # This shouldn't happen here if initial load worked, but handle defensively
                 self.logger.critical("Authentication failed during main loop. Stopping bot.", exc_info=True)
                 self.is_running = False
            except ccxt.NetworkError as e:
                 # Log network errors but continue running, assuming temporary issue
                 self.logger.error(f"Network error in main loop: {e}. Bot will retry after delay.", exc_info=False) # Less verbose logging for network errors
                 # Use the standard poll interval wait after network errors
                 time.sleep(self.poll_interval)
            except Exception as e:
                # Catch-all for unexpected errors in the loop
                self.logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
                # Consider if bot should stop or continue after unexpected errors
                # For now, wait longer and continue, but might need review based on error type
                self.logger.info(f"Waiting for {self.poll_interval * 2}s after unexpected error before next cycle.")
                time.sleep(self.poll_interval * 2)

        self.logger.info("--- Trading Bot stopped ---")


    def initialize_exchange_settings(self) -> bool:
        """Set initial exchange settings like leverage and potentially margin mode."""
        self.logger.info("Initializing exchange settings...")
        success = True

        # Set Leverage (required for futures/swaps)
        if self.exchange.category in ['linear', 'inverse']:
             if not self.exchange.set_leverage(self.symbol, self.leverage):
                  self.logger.error(f"Failed to set initial leverage to {self.leverage}x for {self.symbol}. Check permissions/settings. Continuing but trading might fail.")
                  success = False # Mark as potential failure but continue if possible
             else:
                  self.logger.info(f"Leverage set to {self.leverage}x for {self.symbol}.")
        else:
             self.logger.info("Skipping leverage setting (not applicable for spot category).")


        # Set Margin Mode (Example: Set to ISOLATED if configured)
        # Note: Changing margin mode might require no open positions/orders.
        # margin_mode_config = self.config['trading_settings'].get('margin_mode', 'isolated').lower() # e.g., 'isolated' or 'cross'
        # if self.exchange.category in ['linear', 'inverse']:
        #      try:
        #           # Check current mode first (implementation varies by exchange/CCXT version)
        #           # This part is highly exchange-specific and might need direct API calls or specific CCXT methods
        #           # current_mode = self.exchange.fetch_margin_mode(self.symbol) # Hypothetical
        #           # if current_mode != margin_mode_config:
        #
        #           self.logger.info(f"Attempting to set margin mode to '{margin_mode_config}' for {self.symbol}...")
        #           # Bybit V5 uses private_post_position_switch_margin_mode
        #           # params = {'symbol': self.symbol, 'tradeMode': 0 if margin_mode_config == 'cross' else 1, 'buyLeverage': str(self.leverage), 'sellLeverage': str(self.leverage), 'category': self.exchange.category}
        #           # result = self.exchange.safe_ccxt_call('private_post_position_switch_margin_mode', params=params)
        #           # Check result for success (e.g., retCode == 0)
        #
        #           # Placeholder for generic CCXT call (might not work for all exchanges)
        #           # result = self.exchange.set_margin_mode(margin_mode_config, self.symbol, params={'category': self.exchange.category})
        #           # if result is not None: # Assuming success if no error and non-None result
        #           #      self.logger.info(f"Margin mode set to '{margin_mode_config}' successfully.")
        #           # else:
        #           #      self.logger.error(f"Failed to set margin mode to '{margin_mode_config}'.")
        #           #      success = False
        #           pass # Replace with actual implementation if needed
        #
        #      except ccxt.NotSupported:
        #           self.logger.warning(f"Exchange {self.exchange.id} does not support setting margin mode via standard CCXT method for {self.symbol}.")
        #      except Exception as e:
        #           self.logger.error(f"Failed to set margin mode to '{margin_mode_config}': {e}", exc_info=True)
        #           success = False
        # else:
        #      self.logger.info("Skipping margin mode setting (not applicable for spot category).")


        # Set Hedge Mode vs One-Way Mode (Bybit specific)
        if self.exchange_id == 'bybit' and self.exchange.category in ['linear', 'inverse']:
             try:
                  # 0: One-way, 3: Hedge Mode (Both Long & Short)
                  target_mode = 3 if self.hedge_mode else 0
                  self.logger.info(f"Checking/Setting position mode for {self.symbol} (Target: {'Hedge' if self.hedge_mode else 'One-way'})...")
                  # Bybit V5 endpoint: POST /v5/position/switch-mode
                  # Requires category, symbol, mode
                  # Note: Switching mode might require no open positions or orders.
                  params = {'category': self.exchange.category, 'symbol': self.symbol, 'mode': target_mode}
                  # This endpoint might not be directly exposed in CCXT, use private call if needed
                  # Example using a hypothetical direct call name (check CCXT implementation)
                  if hasattr(self.exchange, 'private_post_position_switch_mode'):
                       result = self.exchange.safe_ccxt_call('private_post_position_switch_mode', params=params)
                       if result and result.get('retCode') == 0:
                            self.logger.info(f"Position mode successfully set to {'Hedge' if self.hedge_mode else 'One-way'}.")
                       elif result: # Call succeeded but Bybit returned an error
                            # Common error: 110025 - Position mode is not modified
                            if result.get('retCode') == 110025:
                                 self.logger.info(f"Position mode is already set correctly ({'Hedge' if self.hedge_mode else 'One-way'}).")
                            else:
                                 self.logger.error(f"Failed to set position mode on Bybit. Code: {result.get('retCode')}, Msg: {result.get('retMsg', 'Unknown error')}")
                                 success = False
                       else: # safe_ccxt_call failed
                            self.logger.error("API call failed when trying to set position mode.")
                            success = False
                  else:
                       self.logger.warning("CCXT method 'private_post_position_switch_mode' not found. Cannot verify/set position mode automatically.")

             except Exception as e:
                  self.logger.error(f"Error setting position mode: {e}", exc_info=True)
                  success = False


        return success


    def get_current_position(self) -> Optional[Dict[str, Any]]:
        """
        Fetches and returns the single active position relevant to the bot for its symbol.
        Handles hedge mode by trying to match the position index stored in the bot's state.
        Returns None if no relevant position is found or if fetching fails.
        """
        positions = self.exchange.fetch_positions(self.symbol)
        if positions is None:
            self.logger.warning(f"Could not fetch positions for {self.symbol}. Assuming no position.")
            return None # Indicate failure to fetch

        # Filter for positions that are genuinely active (non-zero size)
        # Uses 'contracts' which should be populated with Decimal size by fetch_positions wrapper
        active_positions = [p for p in positions if p.get('contracts') is not None and p['contracts'] != Decimal(0)]

        if not active_positions:
            self.logger.info(f"No active position found on exchange for {self.symbol}.")
            return None # No position exists

        if self.hedge_mode:
            # In hedge mode, the bot logic aims to manage one side (long or short) at a time,
            # identified by positionIdx (1 for long, 2 for short).
            # We need to find the position that matches the bot's *intended* side/idx if it thinks it has one.
            state_pos = self.state.get('active_position')
            if state_pos:
                 # Bot thinks it has a position, find the matching one on the exchange
                 target_idx = state_pos.get('position_idx')
                 target_side_enum = state_pos.get('side') # PositionSide Enum from state

                 if target_idx is None or target_side_enum is None:
                      self.logger.error("Hedge mode active, but bot state is inconsistent (missing idx or side). Cannot reliably identify position.")
                      return None # Inconsistent state

                 found_match = None
                 for p in active_positions:
                      # positionIdx should be int (0 for one-way, 1 for long hedge, 2 for short hedge)
                      p_idx = p.get('positionIdx')
                      # side should be 'long' or 'short' string
                      p_side_str = p.get('side')

                      # Match based on positionIdx primarily
                      if p_idx is not None and p_idx == target_idx:
                           # Verify side matches as well for robustness
                           if p_side_str == target_side_enum.value:
                                found_match = p
                                break
                           else:
                                self.logger.warning(f"Hedge mode position found with matching Idx ({target_idx}) but mismatched Side (State: {target_side_enum.value}, Exchange: {p_side_str}). Ignoring.")
                                # Don't select this position if side mismatches index expectation

                 if found_match:
                      self.logger.info(f"Found active hedge mode position matching bot state: Side {found_match.get('side')}, Idx {found_match.get('positionIdx')}, Size {found_match.get('contracts')}")
                      return found_match
                 else:
                      # Bot state has a position, but no matching one found on exchange (by idx and side).
                      # This implies the bot's intended position was closed or doesn't exist.
                      self.logger.warning(f"Bot state indicates hedge position (Idx: {target_idx}, Side: {target_side_enum.value}), but no matching active position found on exchange. State may be stale.")
                      # Let sync_bot_state handle clearing the stale state later. Return None for now.
                      return None
            else:
                 # Bot state has no position, but found active position(s) on exchange.
                 # These are considered external positions not managed by this bot instance.
                 pos_details = [f"Idx: {p.get('positionIdx')}, Side: {p.get('side')}, Size: {p.get('contracts')}" for p in active_positions]
                 self.logger.warning(f"Found {len(active_positions)} active hedge mode position(s) on exchange, but bot state has no active position. Ignoring external positions: [{'; '.join(pos_details)}]")
                 return None # Don't return external positions if bot thinks it has none

        else: # Non-hedge (One-way) mode
            if len(active_positions) > 1:
                 # This shouldn't happen in one-way mode for a single symbol. Log error.
                 self.logger.error(f"CRITICAL: Expected only one active position for {self.symbol} in non-hedge mode, but found {len(active_positions)}. Check exchange state manually!")
                 # Decide how to handle: Use first? Use largest? Abort?
                 # Using the first one found might lead to unexpected behavior. Returning None is safer.
                 return None
            # Return the single active position
            pos = active_positions[0]
            self.logger.info(f"Found active non-hedge position: Side {pos.get('side')}, Size {pos.get('contracts')}")
            return pos


    def sync_bot_state_with_position(self, current_position_on_exchange: Optional[Dict[str, Any]]):
        """
        Updates the bot's internal state ('active_position', 'stop_loss_price', etc.)
        based on the fetched position from the exchange. Clears state if the position is gone
        or doesn't match the expected state (e.g., wrong side/idx in hedge mode).
        """
        bot_state_position = self.state.get('active_position')
        bot_thinks_has_position = bot_state_position is not None

        if current_position_on_exchange:
            # --- Position exists on exchange ---
            exchange_pos_symbol = current_position_on_exchange.get('symbol')
            exchange_pos_side_str = current_position_on_exchange.get('side') # 'long'/'short'
            exchange_pos_size = current_position_on_exchange.get('contracts') # Decimal size
            exchange_pos_entry = current_position_on_exchange.get('entryPrice') # Decimal entry
            exchange_pos_idx = current_position_on_exchange.get('positionIdx') # Int index
            exchange_sl = current_position_on_exchange.get('info', {}).get('stopLoss') # Decimal SL from info (if available)

            # Validate essential data from exchange position
            if not all([exchange_pos_symbol, exchange_pos_side_str, exchange_pos_size is not None, exchange_pos_entry is not None]):
                 self.logger.error("Fetched exchange position data is incomplete. Cannot sync state.")
                 # If bot thought it had a position, maybe clear it? Or wait for next cycle?
                 if bot_thinks_has_position:
                      self.logger.warning("Clearing potentially stale bot state due to incomplete exchange data.")
                      self._clear_position_state("Incomplete exchange position data")
                 return

            # Convert exchange SL to Decimal if present and valid
            exchange_sl_dec = None
            if isinstance(exchange_sl, Decimal) and exchange_sl.is_finite() and exchange_sl > 0:
                 exchange_sl_dec = exchange_sl
            elif isinstance(exchange_sl, (str, float, int)) and str(exchange_sl) not in ['', '0']: # Check for valid SL string/number
                 try:
                      sl_dec_temp = Decimal(str(exchange_sl))
                      if sl_dec_temp.is_finite() and sl_dec_temp > 0:
                           exchange_sl_dec = sl_dec_temp
                 except InvalidOperation:
                      self.logger.debug(f"Could not parse stop loss '{exchange_sl}' from exchange position info.")


            if not bot_thinks_has_position:
                # Bot thought no position, but found one. This implies an external position or recovery after crash.
                # Adopt the position into state ONLY if it matches the bot's expected mode (hedge/one-way).
                position_matches_mode = False
                if self.hedge_mode:
                     # Accept if idx is 1 (long) or 2 (short)
                     if exchange_pos_idx in [1, 2]: position_matches_mode = True
                else:
                     # Accept if idx is 0 (one-way)
                     if exchange_pos_idx == 0: position_matches_mode = True

                if position_matches_mode:
                     self.logger.warning(f"Found unexpected active position on exchange matching bot mode ({exchange_pos_side_str}, Idx: {exchange_pos_idx}). Adopting into state.")
                     try:
                          adopted_side = PositionSide(exchange_pos_side_str)
                     except ValueError:
                          self.logger.error(f"Invalid side '{exchange_pos_side_str}' on adopted position. Cannot sync.")
                          return

                     self.state['active_position'] = {
                         'symbol': exchange_pos_symbol,
                         'side': adopted_side,
                         'entry_price': exchange_pos_entry,
                         'size': exchange_pos_size,
                         'position_idx': exchange_pos_idx,
                         'order_id': None, # Cannot know the entry order ID
                     }
                     # Attempt to adopt the SL found on the exchange if it exists
                     if exchange_sl_dec is not None:
                          self.state['stop_loss_price'] = exchange_sl_dec
                          self.logger.info(f"Adopted stop loss {exchange_sl_dec} found on exchange position.")
                     else:
                          self.state['stop_loss_price'] = None # No SL found or invalid
                          self.logger.warning("No valid stop loss found on the adopted exchange position. SL state is None.")

                     self.state['take_profit_price'] = None # Assume no TP managed by bot initially
                     self.state['break_even_achieved'] = False # Reset BE state
                     self.logger.info(f"Bot state synced. Current position: {self.state['active_position']['side'].name} {self.state['active_position']['size']} @ {self.state['active_position']['entry_price']}. SL: {self.state['stop_loss_price']}")
                else:
                     # Found position doesn't match bot's mode (e.g., hedge position found when bot is in one-way)
                     self.logger.warning(f"Found active position on exchange ({exchange_pos_side_str}, Idx: {exchange_pos_idx}) that does NOT match bot's configured mode ({'Hedge' if self.hedge_mode else 'One-way'}). Ignoring external position.")

            else:
                # --- Bot already knew about a position. Verify and update details. ---
                state_pos = self.state['active_position']
                state_side_enum = state_pos['side']
                state_idx = state_pos.get('position_idx')

                # Check if the exchange position matches the bot's state (symbol, side, and idx for hedge)
                match = False
                if exchange_pos_symbol == state_pos['symbol'] and \
                   exchange_pos_side_str == state_side_enum.value:
                    if self.hedge_mode:
                         if exchange_pos_idx == state_idx:
                              match = True
                    else: # Non-hedge mode (expect idx 0)
                         if exchange_pos_idx == 0:
                              match = True

                if match:
                     # Position matches state, update dynamic values like size, entry, SL
                     self.logger.debug(f"Exchange position matches state. Updating details: Size={exchange_pos_size}, Entry={exchange_pos_entry}, Exchange SL={exchange_sl_dec}")
                     state_pos['size'] = exchange_pos_size
                     state_pos['entry_price'] = exchange_pos_entry # Update entry if avg price changed

                     # --- Sync Stop Loss State ---
                     current_state_sl = self.state.get('stop_loss_price') # Decimal or None

                     if exchange_sl_dec is not None:
                          # Exchange reports an SL. Update state SL if it's different or state SL was None.
                          # Use tolerance to avoid updates due to minor float/Decimal conversion differences.
                          sl_differs = current_state_sl is None or abs(exchange_sl_dec - current_state_sl) > Decimal('1e-9')
                          if sl_differs:
                               self.logger.info(f"Detected SL change on exchange. Updating state SL from {current_state_sl} to {exchange_sl_dec}.")
                               self.state['stop_loss_price'] = exchange_sl_dec
                               # Check if this SL update implies BE was achieved
                               self._check_and_update_be_status(state_pos, exchange_sl_dec)
                     else:
                          # Exchange reports NO SL (or zero). If state has an SL, it means the SL was likely hit, cancelled, or failed to set.
                          if current_state_sl is not None:
                               self.logger.warning(f"Bot state has SL {current_state_sl}, but no active SL found on exchange position. Clearing state SL.")
                               self.state['stop_loss_price'] = None
                               # Should we assume the position should be closed now? Or just clear SL state?
                               # Clearing SL state seems safer, management logic might re-apply or close later.
                               # Reset BE flag if SL disappears
                               if self.state.get('break_even_achieved', False):
                                    self.logger.info("Resetting break_even_achieved flag as SL disappeared.")
                                    self.state['break_even_achieved'] = False

                     # Update last sync time
                     self.state['last_sync_time'] = time.time()

                else:
                     # Position exists, but doesn't match state (e.g., wrong side/idx/symbol). Treat as external/mismatch.
                     self.logger.warning(f"Found active position on exchange ({exchange_pos_symbol}, {exchange_pos_side_str}, Idx: {exchange_pos_idx}) that does NOT match bot state ({state_pos.get('symbol')}, {state_side_enum.value}, Idx: {state_idx}). Clearing bot state.")
                     self._clear_position_state("Mismatch with exchange position")


        else:
            # --- No position on exchange ---
            if bot_thinks_has_position:
                # Bot thought there was a position, but it's gone. Clear state.
                state_pos_details = f"({bot_state_position.get('side', PositionSide.NONE).name}, Size: {bot_state_position.get('size')})"
                self.logger.info(f"Position {state_pos_details} no longer found on exchange. Clearing bot state.")
                self._clear_position_state("Position closed/missing on exchange")
            # else: Bot thought no position, exchange has no position. State is correct. Do nothing.


    def _check_and_update_be_status(self, position_state: Dict[str, Any], current_sl: Decimal):
        """Checks if the current SL implies break-even is achieved and updates state."""
        if not self.state.get('break_even_achieved', False): # Only update if not already achieved
            entry = position_state.get('entry_price')
            side = position_state.get('side') # PositionSide Enum

            if entry is None or side is None or current_sl is None: return # Not enough info

            be_achieved = False
            # Quantize entry for comparison to avoid precision issues
            quantized_entry = self.position_manager.quantize_price(entry)
            if quantized_entry is None:
                 self.logger.warning("Could not quantize entry price for BE status check.")
                 return

            if side == PositionSide.LONG and current_sl >= quantized_entry:
                 be_achieved = True
            elif side == PositionSide.SHORT and current_sl <= quantized_entry:
                 be_achieved = True

            if be_achieved:
                 self.logger.info(f"Marking Break-Even as achieved based on updated SL {current_sl} relative to entry {quantized_entry}.")
                 self.state['break_even_achieved'] = True


    def _clear_position_state(self, reason: str):
        """Internal helper to safely clear all position-related state variables."""
        self.logger.info(f"Clearing position state. Reason: {reason}")
        self.state['active_position'] = None
        self.state['stop_loss_price'] = None
        self.state['take_profit_price'] = None
        self.state['break_even_achieved'] = False
        # Keep last_order_id for potential reference/debugging? Or clear it too?
        # self.state['last_order_id'] = None # Optional: Clear last order ID as well


    def manage_existing_position(self, latest_indicators: pd.Series, signal: Signal, live_position_data: Optional[Dict[str, Any]]):
        """
        Manages SL, TP, and potential exits for the active position based on state and live data.
        Updates self.state directly based on outcomes of API calls.
        """
        self.logger.info("Managing existing position...")
        # Use the state that was just synced
        position_state = self.state.get('active_position')

        if not position_state:
             self.logger.warning("manage_existing_position called but state has no active position after sync. Skipping management.")
             return
        if not live_position_data:
             # This might happen if sync failed right before this call, but state wasn't cleared yet.
             self.logger.error("manage_existing_position called without live position data (sync likely failed). Cannot proceed.")
             # Consider re-fetching or skipping cycle. Skipping is safer.
             return

        position_side = position_state['side'] # PositionSide Enum
        position_idx = position_state.get('position_idx') # Needed for hedge mode API calls

        # Ensure we have a valid SL price in state to manage from (sync should handle this)
        current_sl_in_state = self.state.get('stop_loss_price')
        if current_sl_in_state is None:
             self.logger.warning("Cannot manage position: Stop loss price is missing from bot state even after sync. Manual intervention might be needed.")
             # Should we try to set an initial SL based on current state? Risky.
             # Or fetch SL from exchange again? Done in sync. If still None, maybe close?
             # For now, skip SL management if SL state is missing. Exit checks might still run.
             pass # Continue to exit checks


        # --- Exit Checks ---
        # 1. Check for MA Cross Exit (if enabled)
        if self.position_manager.check_ma_cross_exit(latest_indicators, position_side):
            self.logger.info("MA Cross exit condition met. Closing position.")
            self.close_position("MA Cross Exit")
            return # Exit management cycle after closing

        # 2. Check for Signal-Based Exit (Opposing Signal)
        exit_on_signal = self.config.get('trading_settings', {}).get('exit_on_opposing_signal', True)
        should_exit_on_signal = False
        if exit_on_signal:
            if position_side == PositionSide.LONG and signal in [Signal.SELL, Signal.STRONG_SELL]:
                self.logger.info(f"Opposing signal ({signal.name}) received for LONG position. Closing.")
                should_exit_on_signal = True
            elif position_side == PositionSide.SHORT and signal in [Signal.BUY, Signal.STRONG_BUY]:
                self.logger.info(f"Opposing signal ({signal.name}) received for SHORT position. Closing.")
                should_exit_on_signal = True

        if should_exit_on_signal:
            self.close_position(f"Opposing Signal ({signal.name})")
            return # Exit management cycle

        # --- Stop Loss Management (only if not exiting) ---
        if current_sl_in_state is not None:
             # Pass live position data and current state to position manager for SL calculation
             new_sl_price = self.position_manager.manage_stop_loss(live_position_data, latest_indicators, self.state)

             if new_sl_price is not None:
                 # A new SL price (BE or TSL) was calculated and is different from the current one.
                 # Validate the proposed SL before attempting to set it.
                 live_mark_price = live_position_data.get('markPrice')
                 is_valid_sl = True
                 if live_mark_price: # Check against mark price if available
                      if position_side == PositionSide.LONG and new_sl_price > live_mark_price:
                           self.logger.warning(f"Proposed new SL {new_sl_price} is > current mark price {live_mark_price}. Invalid SL.")
                           is_valid_sl = False
                      if position_side == PositionSide.SHORT and new_sl_price < live_mark_price:
                           self.logger.warning(f"Proposed new SL {new_sl_price} is < current mark price {live_mark_price}. Invalid SL.")
                           is_valid_sl = False
                 else:
                      self.logger.warning("Mark price not available in live data, cannot perform final SL validation against it.")

                 if is_valid_sl:
                      self.logger.info(f"Attempting to update stop loss on exchange to: {new_sl_price}")
                      # Pass only the SL price to set_protection, along with position_idx if needed
                      protection_params = {'stop_loss': new_sl_price}
                      if self.hedge_mode:
                           protection_params['position_idx'] = position_idx

                      if self.exchange.set_protection(self.symbol, **protection_params):
                          self.logger.info(f"Stop loss update request successful for {new_sl_price}.")
                          # Update state ONLY on successful exchange update request
                          self.state['stop_loss_price'] = new_sl_price
                          # Check if this update achieved break-even status
                          self._check_and_update_be_status(position_state, new_sl_price)
                          # Save state immediately after successful SL update
                          save_state(self.state, self.state_path, self.logger)
                      else:
                          self.logger.error(f"Failed to update stop loss on exchange to {new_sl_price}. API call failed or returned error.")
                          # State SL remains unchanged if exchange update fails. Will retry next cycle.
                          # Do NOT update self.state['break_even_achieved'] if API call failed.
                 else:
                      self.logger.error(f"Proposed SL update to {new_sl_price} was deemed invalid. Aborting update.")
        else:
             self.logger.debug("Skipping SL management as current SL is missing in state.")


        self.logger.debug("Position management cycle finished. No further actions taken.")


    def attempt_new_entry(self, signal: Signal, latest_indicators: pd.Series, signal_details: Dict[str, Any]):
        """
        Attempts to enter a new position based on the signal, calculates SL and size,
        places the order, and updates state. Includes post-order verification.
        """
        if self.state.get('active_position'):
             # This check should be redundant due to the main loop logic, but good for safety
             self.logger.warning("Attempted entry when a position is already active in state. Skipping entry.")
             return

        self.logger.info("Checking for new entry opportunities...")

        entry_signal = False
        target_side: Optional[PositionSide] = None # e.g., PositionSide.LONG
        order_side: Optional[OrderSide] = None   # e.g., OrderSide.BUY

        if signal in [Signal.BUY, Signal.STRONG_BUY]:
             entry_signal = True
             target_side = PositionSide.LONG
             order_side = OrderSide.BUY
        elif signal in [Signal.SELL, Signal.STRONG_SELL]:
             entry_signal = True
             target_side = PositionSide.SHORT
             order_side = OrderSide.SELL

        if not entry_signal:
            self.logger.info(f"Signal is {signal.name}. No entry condition met.")
            return

        # --- Pre-computation and Validation before placing order ---
        self.logger.info(f"Entry signal {signal.name} received. Preparing entry for {target_side.name}...")

        # 1. Get Current Price (use last close as estimate, or fetch ticker for more accuracy)
        # Using close price avoids an extra API call but might be slightly stale.
        entry_price_estimate = latest_indicators.get('close')
        if not isinstance(entry_price_estimate, Decimal) or not entry_price_estimate.is_finite() or entry_price_estimate <= 0:
             self.logger.warning("Could not get valid close price from indicators for entry estimate. Fetching ticker...")
             try:
                  ticker = self.exchange.safe_ccxt_call('fetch_ticker', self.symbol)
                  if ticker and ticker.get('last'):
                       entry_price_estimate = Decimal(str(ticker['last']))
                       if not entry_price_estimate.is_finite() or entry_price_estimate <= 0:
                            raise ValueError(f"Invalid ticker price: {entry_price_estimate}")
                       self.logger.info(f"Using ticker last price for entry estimate: {entry_price_estimate}")
                  else:
                       self.logger.error("Failed to get valid last price from ticker. Cannot proceed with entry.")
                       return
             except (ValueError, Exception) as e:
                  self.logger.error(f"Failed to fetch or parse ticker price: {e}. Cannot proceed with entry.")
                  return

        # 2. Calculate Stop Loss based on the estimated entry price
        stop_loss_price = self.position_manager.calculate_stop_loss(entry_price_estimate, target_side, latest_indicators)
        if stop_loss_price is None:
            self.logger.error("Failed to calculate a valid stop loss price. Cannot proceed with entry.")
            return

        # 3. Calculate Position Size based on risk, SL, and available equity
        balance_info = self.exchange.fetch_balance()
        if not balance_info:
             self.logger.error("Failed to fetch balance info. Cannot size position.")
             return
        if self.quote_asset not in balance_info:
             self.logger.error(f"Quote asset '{self.quote_asset}' not found in balance info: {list(balance_info.keys())}. Cannot size position.")
             return

        available_equity = balance_info[self.quote_asset] # Should be Decimal equity
        if not isinstance(available_equity, Decimal) or not available_equity.is_finite() or available_equity <= 0:
             self.logger.error(f"Invalid or non-positive available equity for {self.quote_asset}: {available_equity}. Cannot size position.")
             return

        position_size = self.position_manager.calculate_position_size(
            entry_price_estimate, stop_loss_price, available_equity, self.quote_asset
        )
        if position_size is None or position_size <= 0:
            self.logger.warning("Position size calculation failed or resulted in zero/negative size. No entry.")
            return

        # --- Place Entry Order ---
        # Consider using a limit order slightly away from market for better entry? More complex handling.
        # Using Market order for simplicity, assuming quick fill near estimate.
        order_type = 'market'
        self.logger.info(f"Attempting to place {order_side.value} {order_type} order for {position_size} {self.symbol} with calculated SL {stop_loss_price}")

        # Prepare parameters for create_order
        # Bybit V5 allows setting SL/TP directly with the order using 'stopLoss'/'takeProfit' params
        order_params = {
            # Use string representation for prices in params, quantized by calculate_stop_loss
            'stopLoss': str(stop_loss_price),
            # 'takeProfit': str(take_profit_price), # Add if TP is calculated/used
            # Trigger prices by Mark Price (common default, confirm if needed for exchange)
            # 'slTriggerBy': 'MarkPrice', # Example: 'LastPrice', 'IndexPrice'
            # 'tpTriggerBy': 'MarkPrice',
        }

        # Add positionIdx for hedge mode entry (1 for long, 2 for short)
        position_idx = None
        if self.hedge_mode:
             position_idx = 1 if order_side == OrderSide.BUY else 2
             order_params['positionIdx'] = position_idx
        else:
             order_params['positionIdx'] = 0 # Explicitly set for one-way

        # Create the order
        order_result = self.exchange.create_order(
            symbol=self.symbol,
            order_type=order_type,
            side=order_side,
            amount=position_size, # Pass Decimal size
            price=None, # Market order has no price argument
            params=order_params
        )

        # --- Post-Order Placement Handling ---
        if order_result and order_result.get('id'):
            order_id = order_result['id']
            self.logger.info(f"Entry order ({order_side.value} {order_type}) placed successfully. Order ID: {order_id}. Status: {order_result.get('status', 'unknown')}")
            self.state['last_order_id'] = order_id

            # --- Update Bot State (Optimistic Update) ---
            # Assume market order fills quickly near the estimated price.
            # A more robust system would poll the order status until filled, get the actual fill price/size,
            # and then update the state. This is a simplification for polling bots.
            self.state['active_position'] = {
                'symbol': self.symbol,
                'side': target_side,
                'entry_price': entry_price_estimate, # Use estimate; sync will update with actual later
                'size': position_size, # Use requested size; sync will update with actual later
                'order_id': order_id, # Store the entry order ID
                'position_idx': position_idx # Store idx if hedge mode
            }
            self.state['stop_loss_price'] = stop_loss_price # Store the SL we *intended* to set
            self.state['take_profit_price'] = None # Reset TP state
            self.state['break_even_achieved'] = False # Reset BE state
            save_state(self.state, self.state_path, self.logger) # Save state immediately after potential entry
            self.logger.info(f"Bot state updated optimistically for new {target_side.name} position. Intended SL: {stop_loss_price}.")

            # --- Optional: Short Delay and Verification Step ---
            verify_delay = self.config.get('trading_settings', {}).get('post_order_verify_delay_seconds', 5) # Configurable delay
            if verify_delay > 0:
                 self.logger.info(f"Waiting {verify_delay}s before verifying position and SL after entry order...")
                 time.sleep(verify_delay)
                 self.logger.info("Verifying position status after entry...")
                 final_pos = self.get_current_position()
                 # Sync state again based on verification result
                 self.sync_bot_state_with_position(final_pos)

                 # Explicitly check if the intended SL seems to be active after sync
                 if self.state.get('active_position'): # Check if position still exists after sync
                      state_sl = self.state.get('stop_loss_price')
                      if state_sl is not None:
                           # Check if state SL matches the intended SL (within tolerance)
                           if abs(state_sl - stop_loss_price) < Decimal('1e-9'):
                                self.logger.info(f"Stop loss {stop_loss_price} confirmed active in state after verification.")
                           else:
                                # State SL got updated by sync to something different
                                self.logger.warning(f"Stop loss in state ({state_sl}) differs from intended SL ({stop_loss_price}) after verification sync. Using state value.")
                      else:
                           # State SL is None after sync, meaning exchange didn't report one
                           self.logger.warning("Stop loss NOT found or is zero on exchange position after placing order with SL parameter. Attempting to set SL again via set_protection.")
                           # Retry setting SL using set_protection
                           protection_params = {'stop_loss': stop_loss_price}
                           if self.hedge_mode:
                                protection_params['position_idx'] = self.state['active_position'].get('position_idx')

                           if not self.exchange.set_protection(self.symbol, **protection_params):
                                self.logger.error("Retry failed to set stop loss after entry using set_protection.")
                           else:
                                self.logger.info("Successfully set stop loss via set_protection after initial attempt failed/was not found.")
                                # Update state SL to intended value after successful explicit set
                                self.state['stop_loss_price'] = stop_loss_price
                 else:
                      # Position disappeared between optimistic update and verification
                      self.logger.error("Position not found during verification after placing entry order. State already cleared by sync.")

                 # Save state again after verification/retry
                 save_state(self.state, self.state_path, self.logger)
            else:
                 self.logger.info("Skipping post-order verification step (delay set to 0).")

        else:
            # Order creation failed at the API call level
            self.logger.error("Failed to place entry order (API call failed or returned invalid result).")
            # Ensure state remains clean if order failed
            self._clear_position_state("Entry order placement failed")
            save_state(self.state, self.state_path, self.logger)


    def close_position(self, reason: str):
        """Closes the current active position (based on state) with a market order."""
        if not self.state.get('active_position'):
            self.logger.warning("Close position called but no active position in state.")
            return

        position_info = self.state['active_position']
        size_to_close = position_info.get('size') # Should be Decimal
        current_side = position_info.get('side') # PositionSide Enum
        position_idx = position_info.get('position_idx') # For hedge mode

        # Validate state data before attempting close
        if size_to_close is None or not isinstance(size_to_close, Decimal) or size_to_close <= 0:
             self.logger.error(f"Cannot close position: Size in state is invalid ({size_to_close}). Clearing state.")
             self._clear_position_state(f"Invalid size in state during close: {size_to_close}")
             save_state(self.state, self.state_path, self.logger)
             return
        if current_side is None or current_side == PositionSide.NONE:
             self.logger.error(f"Cannot close position: Side in state is invalid ({current_side}). Clearing state.")
             self._clear_position_state(f"Invalid side in state during close: {current_side}")
             save_state(self.state, self.state_path, self.logger)
             return
        if self.hedge_mode and position_idx is None:
             self.logger.error(f"Cannot close hedge mode position: positionIdx missing from state. Clearing state.")
             self._clear_position_state("Missing positionIdx in state for hedge mode close")
             save_state(self.state, self.state_path, self.logger)
             return


        close_side = OrderSide.SELL if current_side == PositionSide.LONG else OrderSide.BUY
        order_type = 'market' # Use market order for reliable closing

        self.logger.info(f"Attempting to close {current_side.name} position (Idx: {position_idx if self.hedge_mode else 'N/A'}) of size {size_to_close} with {order_type} order. Reason: {reason}")

        # Parameters for closing order
        close_params = {
            'reduceOnly': True # Crucial: ensure it only reduces/closes the specific position
        }
        if self.hedge_mode:
             # Need the correct positionIdx to close the specific side
             close_params['positionIdx'] = position_idx
        else:
             close_params['positionIdx'] = 0 # Explicitly set for one-way

        # Optional: Cancel existing TP/SL orders before closing?
        # Bybit might do this automatically with reduceOnly market orders, but explicit cancellation can be safer.
        # self.cancel_open_orders_for_symbol() # Implement cancellation if needed

        # Create the closing order
        order_result = self.exchange.create_order(
            symbol=self.symbol,
            order_type=order_type,
            side=close_side,
            amount=size_to_close, # Pass Decimal size
            price=None, # Market order
            params=close_params
        )

        if order_result and order_result.get('id'):
            order_id = order_result['id']
            status = order_result.get('status', 'unknown')
            self.logger.info(f"Position close order ({close_side.value} {order_type}) placed successfully. Order ID: {order_id}, Status: {status}. Reason: {reason}")
            # Clear bot state immediately after placing close order (Optimistic Update)
            # More robust: wait for fill confirmation before clearing state, but market orders usually fill fast.
            self._clear_position_state(f"Close order placed (ID: {order_id}, Reason: {reason})")
            self.state['last_order_id'] = order_id # Store the close order ID
            save_state(self.state, self.state_path, self.logger) # Save cleared state

            # Optional: Verify closure after delay?
            # time.sleep(verify_delay)
            # pos_after_close = self.get_current_position()
            # if pos_after_close is None: logger.info("Position closure verified.")
            # else: logger.error("Position still found after attempting closure!")

        else:
            # Close order placement failed
            self.logger.error(f"Failed to place position close order for reason: {reason}. Position state remains unchanged.")
            # State remains unchanged, will likely retry closing on next cycle if conditions persist.

    def _wait_for_next_cycle(self, cycle_start_time: float):
        """Waits until the next polling interval, accounting for execution time."""
        cycle_end_time = time.time()
        execution_time = cycle_end_time - cycle_start_time
        wait_time = max(0, self.poll_interval - execution_time)
        self.logger.debug(f"Cycle execution time: {execution_time:.2f}s. Waiting for {wait_time:.2f}s...")
        if wait_time > 0:
            time.sleep(wait_time)


# --- Main Execution ---

if __name__ == "__main__":
    # Ensure Decimal context is set globally at the very beginning
    try:
        getcontext().prec = DECIMAL_PRECISION + 6 # Set precision early
        getcontext().rounding = ROUND_HALF_UP
    except Exception as e:
         print(f"CRITICAL: Failed to set Decimal context: {e}", file=sys.stderr)
         sys.exit(1)

    # Create bot instance (initializes logger, loads config/state, sets up exchange components)
    bot_instance = None
    try:
        bot_instance = TradingBot(config_path=CONFIG_FILE, state_path=STATE_FILE)
        # Start the main loop
        bot_instance.run()
    except SystemExit as e:
         # Logged critical errors during init or run already caused exit
         print(f"Bot exited with code {e.code}.")
         # Logger might be available, try logging exit reason
         if bot_instance and bot_instance.logger:
              bot_instance.logger.info(f"Bot process terminated with exit code {e.code}.")
         sys.exit(e.code) # Propagate exit code
    except KeyboardInterrupt:
         # Handled within run loop, but catch here if occurs during init
         print("KeyboardInterrupt received during initialization. Exiting.")
         if bot_instance and bot_instance.logger:
              bot_instance.logger.info("KeyboardInterrupt received during initialization.")
         sys.exit(0)
    except Exception as e:
         # Catch any unexpected critical errors during setup or run that weren't handled
         print(f"CRITICAL UNHANDLED ERROR: {e}", file=sys.stderr)
         # Try logging if logger was initialized
         logger = logging.getLogger("TradingBot")
         if logger.hasHandlers(): # Check if logger setup was successful
              logger.critical(f"Unhandled exception caused bot termination: {e}", exc_info=True)
         else: # Logger not ready, print traceback
              import traceback
              traceback.print_exc()
         sys.exit(1) # Exit with error code
