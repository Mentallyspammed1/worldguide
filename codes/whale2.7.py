```python
# whale2.6.py
# Enhanced version focusing on stop-loss/take-profit mechanisms,
# including break-even logic, MA cross exit condition, correct Bybit V5 API usage,
# multi-symbol support, and improved state management.

import argparse
import hashlib
import hmac
import json
import logging
import math
import os
import time
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext, InvalidOperation
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional, Tuple, List, Union

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests # Keep requests for potential future use, though ccxt handles most HTTP
from colorama import Fore, Style, init
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter # Keep for potential future use
from urllib3.util.retry import Retry # Keep for potential future use
from zoneinfo import ZoneInfo # Use zoneinfo for modern timezone handling

# Initialize colorama and set Decimal precision
getcontext().prec = 28  # Sufficient precision for most financial calculations
init(autoreset=True)
load_dotenv()

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
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env file")

CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
# Timezone for logging and display (adjust as needed, see list: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)
TIMEZONE = ZoneInfo("America/Chicago") # Example: "America/New_York", "Europe/London", "Asia/Tokyo"
MAX_API_RETRIES = 3 # Max retries for recoverable API errors (e.g., network issues, rate limits)
RETRY_DELAY_SECONDS = 5 # Base delay between retries (may increase for rate limits)
# Intervals supported by the bot's logic. Note: Ensure your chosen interval has sufficient liquidity/data on Bybit.
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP = { # Map our simplified interval strings to ccxt's expected format
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}
# HTTP status codes considered potentially retryable for some operations
RETRY_ERROR_CODES = [429, 500, 502, 503, 504]

# --- Default Indicator Periods (can be overridden by config.json) ---
DEFAULT_ATR_PERIOD = 14
DEFAULT_CCI_WINDOW = 20
DEFAULT_WILLIAMS_R_WINDOW = 14
DEFAULT_MFI_WINDOW = 14
DEFAULT_STOCH_RSI_WINDOW = 14 # Window for Stoch RSI calculation itself
DEFAULT_STOCH_WINDOW = 14     # Window for underlying RSI in StochRSI (often same as Stoch RSI window)
DEFAULT_K_WINDOW = 3          # K period for StochRSI
DEFAULT_D_WINDOW = 3          # D period for StochRSI
DEFAULT_RSI_WINDOW = 12
DEFAULT_BOLLINGER_BANDS_PERIOD = 30
DEFAULT_BOLLINGER_BANDS_STD_DEV = 2.0 # Ensure float
DEFAULT_SMA_10_WINDOW = 10
DEFAULT_EMA_SHORT_PERIOD = 8
DEFAULT_EMA_LONG_PERIOD = 22
DEFAULT_MOMENTUM_PERIOD = 7
DEFAULT_VOLUME_MA_PERIOD = 15
DEFAULT_FIB_WINDOW = 50
DEFAULT_PSAR_AF = 0.02
DEFAULT_PSAR_MAX_AF = 0.2

FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0] # Standard Fibonacci levels
# LOOP_DELAY_SECONDS is dynamically loaded from config
POSITION_CONFIRM_DELAY = 10 # Seconds to wait after placing order before checking position status
# QUOTE_CURRENCY is dynamically loaded from config

# Ensure log directory exists
os.makedirs(LOG_DIRECTORY, exist_ok=True)


class SensitiveFormatter(logging.Formatter):
    """Custom log formatter to redact sensitive information (API keys) from logs."""
    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record, redacting API key/secret."""
        msg = super().format(record)
        if API_KEY:
            msg = msg.replace(API_KEY, "***API_KEY***")
        if API_SECRET:
            msg = msg.replace(API_SECRET, "***API_SECRET***")
        return msg


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Loads configuration from a JSON file.
    Creates a default config file if it doesn't exist.
    Ensures all keys from the default config are present in the loaded config,
    adding missing keys with default values.
    """
    default_config = {
        "symbols": ["BCH/USDT:USDT"], # List of symbols to trade (CCXT format: BASE/QUOTE:SETTLE)
        "interval": "3", # Default trading interval (string format: "1", "5", "15", "60", "D", etc.)
        "loop_delay": 15, # Default delay in seconds between full bot cycles (processing all symbols)
        # --- Indicator Periods ---
        "atr_period": DEFAULT_ATR_PERIOD,
        "ema_short_period": DEFAULT_EMA_SHORT_PERIOD,
        "ema_long_period": DEFAULT_EMA_LONG_PERIOD,
        "rsi_period": DEFAULT_RSI_WINDOW,
        "bollinger_bands_period": DEFAULT_BOLLINGER_BANDS_PERIOD,
        "bollinger_bands_std_dev": DEFAULT_BOLLINGER_BANDS_STD_DEV,
        "cci_window": DEFAULT_CCI_WINDOW,
        "williams_r_window": DEFAULT_WILLIAMS_R_WINDOW,
        "mfi_window": DEFAULT_MFI_WINDOW,
        "stoch_rsi_window": DEFAULT_STOCH_RSI_WINDOW,
        "stoch_rsi_rsi_window": DEFAULT_STOCH_WINDOW, # Underlying RSI period for StochRSI
        "stoch_rsi_k": DEFAULT_K_WINDOW,
        "stoch_rsi_d": DEFAULT_D_WINDOW,
        "psar_af": DEFAULT_PSAR_AF,
        "psar_max_af": DEFAULT_PSAR_MAX_AF,
        "sma_10_window": DEFAULT_SMA_10_WINDOW,
        "momentum_period": DEFAULT_MOMENTUM_PERIOD,
        "volume_ma_period": DEFAULT_VOLUME_MA_PERIOD,
        "fibonacci_window": DEFAULT_FIB_WINDOW,
        # --- Data Fetching ---
        "orderbook_limit": 25, # Depth of orderbook levels to fetch
        # --- Signal Generation ---
        "signal_score_threshold": 1.5, # Score needed to trigger BUY/SELL signal for 'default' weight set
        "scalping_signal_threshold": 2.5, # Separate threshold for 'scalping' weight set
        "stoch_rsi_oversold_threshold": 25,
        "stoch_rsi_overbought_threshold": 75,
        "volume_confirmation_multiplier": 1.5, # How much higher current volume needs to be than its MA
        # --- Trading Parameters ---
        "enable_trading": True, # Master switch for placing orders (True = live/testnet trading, False = analysis only)
        "use_sandbox": False,   # Set to True to use Bybit's testnet environment
        "risk_per_trade": 0.01, # Percentage of account balance to risk per trade (e.g., 0.01 = 1%)
        "leverage": 20,         # Desired leverage for contract markets (check exchange limits and risk tolerance!)
        "max_concurrent_positions_total": 1, # Global limit on total open positions across ALL symbols
        "quote_currency": "USDT", # The currency for balance checks and position sizing (e.g., USDT, USDC)
        # CRITICAL: This MUST match your setting on the Bybit exchange website/app for the traded category (e.g., Linear Perpetual).
        # The bot currently operates assuming "One-Way" mode for simplicity in position management.
        # Hedge mode requires more complex logic to track separate long/short positions per symbol.
        "position_mode": "One-Way", # "One-Way" or "Hedge"
        # --- Initial Stop Loss / Take Profit (ATR Based) ---
        # Used for initial placement and position sizing calculation.
        "stop_loss_multiple": 1.8, # Initial Stop Loss distance = X * ATR
        "take_profit_multiple": 0.7, # Initial Take Profit distance = Y * ATR
        # --- MA Cross Exit ---
        "enable_ma_cross_exit": True, # Enable closing position if short EMA crosses below long EMA (for long) or vice versa (for short)
        # --- Trailing Stop Loss (Exchange-Side TSL) ---
        "enable_trailing_stop": True, # Enable using Bybit's native Trailing Stop Loss feature
        # Trail distance as a percentage of the entry price (e.g., 0.005 = 0.5%).
        # The bot calculates the absolute price distance required by the API based on this percentage.
        "trailing_stop_callback_rate": 0.005,
        # Activate TSL only after the price moves this percentage in profit from the entry price (e.g., 0.003 = 0.3%).
        # Set to 0 (or 0.0) for immediate TSL activation upon position entry (API value '0').
        "trailing_stop_activation_percentage": 0.003,
        # --- Break-Even Stop ---
        "enable_break_even": True,              # Enable moving Stop Loss to break-even point
        # Trigger BE when profit (in price points) reaches: X * Current ATR value
        "break_even_trigger_atr_multiple": 0.6, # Example: Trigger BE when profit >= 0.6 * ATR
        # Place the BE Stop Loss this many minimum price increments (ticks) away from the entry price
        # in the direction of profit (e.g., 2 ticks to cover potential commission/slippage on exit).
        "break_even_offset_ticks": 2,
         # Set to True to always cancel any active TSL and set a fixed SL at the calculated BE price when BE is triggered.
         # Set to False to allow TSL to potentially remain active if it's already trailing at a better price than the calculated BE price.
         # Note: Bybit V5 often automatically cancels TSL when a new fixed SL is set via API, so True is generally safer and more predictable.
        "break_even_force_fixed_sl": True,
        # --- Indicator Control & Weighting ---
        "indicators": { # Control which indicators are calculated and contribute to the score
            "ema_alignment": True, "momentum": True, "volume_confirmation": True,
            "stoch_rsi": True, "rsi": True, "bollinger_bands": True, "vwap": True,
            "cci": True, "wr": True, "psar": True, "sma_10": True, "mfi": True,
            "orderbook": True, # Flag to enable fetching and scoring orderbook imbalance data
        },
        "weight_sets": { # Define different weighting strategies for indicator scores
            "scalping": { # Example: Fast, momentum-focused weighting
                "ema_alignment": 0.2, "momentum": 0.3, "volume_confirmation": 0.2,
                "stoch_rsi": 0.6, "rsi": 0.2, "bollinger_bands": 0.3, "vwap": 0.4,
                "cci": 0.3, "wr": 0.3, "psar": 0.2, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.15,
            },
            "default": { # Example: More balanced weighting
                "ema_alignment": 0.3, "momentum": 0.2, "volume_confirmation": 0.1,
                "stoch_rsi": 0.4, "rsi": 0.3, "bollinger_bands": 0.2, "vwap": 0.3,
                "cci": 0.2, "wr": 0.2, "psar": 0.3, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.1,
            }
            # Add more weight sets here if desired
        },
        "active_weight_set": "scalping" # Choose which weight set defined above to use ("default", "scalping", etc.)
    }

    if not os.path.exists(filepath):
        print(f"{NEON_YELLOW}Config file not found at {filepath}. Creating default config...{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            print(f"{NEON_GREEN}Default config file created: {filepath}{RESET}")
            return default_config
        except IOError as e:
            print(f"{NEON_RED}Error creating default config file {filepath}: {e}{RESET}")
            print(f"{NEON_YELLOW}Using internal default config values.{RESET}")
            return default_config # Return default even if creation fails

    try:
        with open(filepath, encoding="utf-8") as f:
            config_from_file = json.load(f)
            # Ensure all default keys exist in the loaded config recursively
            updated_config = _ensure_config_keys(config_from_file, default_config)
            # Save back if keys were added during the update
            if updated_config != config_from_file:
                print(f"{NEON_YELLOW}Config file updated with missing default keys: {filepath}{RESET}")
                try:
                    with open(filepath, "w", encoding="utf-8") as f_write:
                        json.dump(updated_config, f_write, indent=4)
                except IOError as e:
                    print(f"{NEON_RED}Error writing updated config file {filepath}: {e}{RESET}")
            return updated_config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"{NEON_RED}Error loading config file {filepath}: {e}. Using default config.{RESET}")
        # Attempt to create default if loading failed badly
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            print(f"{NEON_YELLOW}Created default config file: {filepath}{RESET}")
        except IOError as e_create:
            print(f"{NEON_RED}Error creating default config file after load error: {e_create}{RESET}")
        return default_config


def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively ensures that all keys present in the default_config dictionary
    are also present in the config dictionary. If a key is missing in config,
    it's added with the value from default_config.
    """
    updated_config = config.copy()
    for key, default_value in default_config.items():
        if key not in updated_config:
            updated_config[key] = default_value
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # Recursively check nested dictionaries
            updated_config[key] = _ensure_config_keys(updated_config[key], default_value)
        # Optional: Add type checking/conversion here if needed
        # elif not isinstance(updated_config.get(key), type(default_value)):
        #     print(f"Warning: Config type mismatch for key '{key}'. Expected {type(default_value)}, got {type(updated_config.get(key))}.")
        #     # Decide whether to overwrite with default, attempt conversion, or just warn
    return updated_config

# --- Load Configuration ---
CONFIG = load_config(CONFIG_FILE)
QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT") # Default to USDT if missing
LOOP_DELAY_SECONDS = int(CONFIG.get("loop_delay", 15)) # Use configured loop delay, default 15s
console_log_level = logging.INFO # Default console level, can be changed via args

# --- Logger Setup ---
# Global dictionary to manage loggers per symbol/module
loggers: Dict[str, logging.Logger] = {}

def setup_logger(name: str, is_symbol_logger: bool = False) -> logging.Logger:
    """
    Sets up a logger instance with both file and console handlers.
    If is_symbol_logger is True, creates a symbol-specific log file.
    Otherwise, uses a generic name (e.g., 'main').
    """
    if name in loggers:
        # If logger already exists, ensure its console handler level matches the current global setting.
        logger = loggers[name]
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(console_log_level)
        return logger

    # Clean the name for use in filenames if it's a symbol logger
    safe_name = name.replace('/', '_').replace(':', '-') if is_symbol_logger else name
    logger_instance_name = f"livebot_{safe_name}" # Unique internal name for the logger instance
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_instance_name}.log")
    logger = logging.getLogger(logger_instance_name)

    # Set the base logging level to DEBUG to capture all messages
    logger.setLevel(logging.DEBUG)

    # --- File Handler ---
    # Writes DEBUG level and above to a rotating file. Includes line numbers for easier debugging.
    try:
        # Ensure the log directory exists before creating the handler
        os.makedirs(LOG_DIRECTORY, exist_ok=True)
        # Rotate log files when they reach 10MB, keep 5 backup files.
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        # Use the SensitiveFormatter to redact API keys/secrets from file logs.
        # Include line number [%(lineno)d] in file logs.
        file_formatter = SensitiveFormatter(
            "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S' # Consistent timestamp format
            )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG) # Log everything (DEBUG and above) to the file
        logger.addHandler(file_handler)
    except Exception as e:
        # Print error to console if file logger setup fails
        print(f"{NEON_RED}Error setting up file logger for {log_filename}: {e}{RESET}")

    # --- Stream Handler (Console) ---
    stream_handler = logging.StreamHandler()
    # Use the SensitiveFormatter for console output as well.
    # Use color codes for better readability.
    stream_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S' # Add timestamp format to console output
    )
    stream_handler.setFormatter(stream_formatter)
    # Set console logging level based on the global variable (can be set by command-line args)
    stream_handler.setLevel(console_log_level)
    logger.addHandler(stream_handler)

    # Prevent log messages from propagating to the root logger, which avoids duplicate console outputs.
    logger.propagate = False

    loggers[name] = logger # Store the configured logger instance
    return logger

def get_logger(name: str, is_symbol_logger: bool = False) -> logging.Logger:
    """Retrieves an existing logger instance or creates a new one if needed."""
    if name not in loggers:
        return setup_logger(name, is_symbol_logger)
    else:
        # Ensure the existing logger's console level matches the current global setting
        logger = loggers[name]
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(console_log_level)
        return logger

# --- CCXT Exchange Setup ---
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """
    Initializes the CCXT Bybit exchange object with V5 API preferences,
    error handling, connection testing, and market loading.
    """
    try:
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True, # Enable CCXT's built-in rate limiter
            'options': {
                # Assume linear contracts (USDT margined) as default. Adjust if trading inverse or other types.
                'defaultType': 'linear',
                'adjustForTimeDifference': True, # Automatically sync client time with server time
                # Connection timeouts (in milliseconds)
                'fetchTickerTimeout': 10000, # 10 seconds
                'fetchBalanceTimeout': 15000, # 15 seconds
                'createOrderTimeout': 20000, # 20 seconds for placing orders
                'fetchOrderTimeout': 15000,  # 15 seconds for fetching orders
                'fetchPositionsTimeout': 15000, # 15 seconds for fetching positions
                'recvWindow': 10000, # Bybit default is 5000ms, 10000ms is safer for potential latency
                'brokerId': 'livebot71', # Optional: Identify your bot via Bybit's Broker Program (replace if applicable)

                # Explicitly request V5 API versions for relevant endpoints using 'versions' dictionary.
                # This tells CCXT which API version to prefer for these specific calls.
                'versions': {
                    'public': {
                        'GET': {
                            'market/tickers': 'v5',   # Fetch Ticker(s)
                            'market/kline': 'v5',     # Fetch OHLCV
                            'market/orderbook': 'v5', # Fetch Order Book
                        },
                    },
                    'private': {
                        'GET': {
                            'position/list': 'v5',          # Fetch Positions
                            'account/wallet-balance': 'v5', # Fetch Wallet Balance (specific coin)
                            'order/realtime': 'v5',         # Fetch Open Orders
                            'order/history': 'v5',          # Fetch Order History
                            # Unified Margin specific endpoints (uncomment and test if using Unified account type)
                            # 'unified/account/wallet-balance': 'v5',
                        },
                        'POST': {
                            'order/create': 'v5',            # Create Order
                            'order/cancel': 'v5',            # Cancel Order
                            'position/set-leverage': 'v5',   # Set Leverage
                            'position/trading-stop': 'v5',   # Set SL/TP/TSL for a position
                            # Unified Margin specific endpoints (uncomment and test if using Unified)
                            # 'unified/private/order/create': 'v5',
                            # 'unified/private/order/cancel': 'v5',
                        },
                    },
                },
                 # Fallback default options, potentially redundant with 'versions' but can help older CCXT versions
                 'default_options': {
                    'adjustForTimeDifference': True,
                    'warnOnFetchOpenOrdersWithoutSymbol': False, # Suppress a common CCXT warning
                    'recvWindow': 10000,
                    # Explicitly setting method names to use V5 (might be internally mapped by CCXT based on 'versions')
                    'fetchPositions': 'v5',
                    'fetchBalance': 'v5',
                    'createOrder': 'v5',
                    'fetchOrder': 'v5',
                    'fetchTicker': 'v5',
                    'fetchOHLCV': 'v5',
                    'fetchOrderBook': 'v5',
                    'setLeverage': 'v5',
                    # CCXT often uses privatePostPositionTradingStop internally when calling unified methods like modify_position with SL/TP
                    'private_post_v5_position_trading_stop': 'v5',
                },
                # Define V5 account types if needed by the specific CCXT version/implementation
                'accounts': {
                    'future': {'linear': 'CONTRACT', 'inverse': 'CONTRACT'}, # Future contracts map to CONTRACT account
                    'swap': {'linear': 'CONTRACT', 'inverse': 'CONTRACT'},   # Swap contracts map to CONTRACT account
                    'option': {'unified': 'OPTION'},                          # Option contracts map to OPTION account
                    'spot': {'unified': 'SPOT'},                              # Spot maps to SPOT account
                    # Unified account support for various underlying types
                    'unified': {'linear': 'UNIFIED', 'inverse': 'UNIFIED', 'spot': 'UNIFIED', 'option': 'UNIFIED'},
                },
                'bybit': { # Bybit specific options block
                     'defaultSettleCoin': QUOTE_CURRENCY, # Set default settlement coin (e.g., USDT) if needed
                }
            }
        }

        # Select the Bybit exchange class dynamically
        exchange_class = getattr(ccxt, 'bybit')
        exchange = exchange_class(exchange_options)

        # Set sandbox mode (testnet) if configured
        if CONFIG.get('use_sandbox'):
            logger.warning(f"{NEON_YELLOW}--- USING SANDBOX MODE (Testnet) ---{RESET}")
            exchange.set_sandbox_mode(True)
        else:
             logger.info(f"{NEON_GREEN}--- USING LIVE TRADING MODE ---{RESET}")


        # --- Load Markets ---
        # Essential for getting symbol details (precision, limits, etc.)
        logger.info(f"Loading markets for {exchange.id}...")
        # Explicitly load markets by category for Bybit V5 if possible, as it can be faster.
        # Load linear and spot initially, as these are common. Add others if needed.
        try:
             logger.debug("Attempting to load markets by category (linear, spot)...")
             exchange.load_markets(params={'category': 'linear'})
             exchange.load_markets(params={'category': 'spot'})
             # If trading inverse or options, load them too:
             # exchange.load_markets(params={'category': 'inverse'})
             # exchange.load_markets(params={'category': 'option'})
             logger.debug("Market loading by category attempted.")
        except Exception as market_load_err:
            logger.warning(f"Could not pre-load markets by category: {market_load_err}. Relying on default load_markets().")
            exchange.load_markets() # Fallback to default load if category loading fails

        exchange.last_load_markets_timestamp = time.time() # Store timestamp for periodic reload check
        logger.info(f"Markets loaded for {exchange.id}. Total markets: {len(exchange.markets)}")

        logger.info(f"CCXT exchange initialized ({exchange.id}). Sandbox: {CONFIG.get('use_sandbox')}")
        logger.info(f"CCXT version: {ccxt.__version__}") # Log CCXT version for debugging

        # --- Test API Credentials & Permissions ---
        # Fetch balance as a basic test. Use the configured QUOTE_CURRENCY.
        # Specify account type for Bybit V5 (e.g., CONTRACT, UNIFIED, SPOT) - crucial for correct balance.
        # Assume 'CONTRACT' for derivatives (like USDT perpetuals) unless config/logic dictates otherwise.
        # A robust implementation might determine this based on the symbols being traded.
        account_type_to_test = 'CONTRACT' # Default assumption for linear perpetuals
        # Consider 'UNIFIED' if using Unified Trading Account
        # account_type_to_test = 'UNIFIED'

        logger.info(f"Attempting initial balance fetch ({QUOTE_CURRENCY}, Account Type: {account_type_to_test})...")
        try:
            # Use our enhanced balance fetching function
            balance_decimal = fetch_balance(exchange, QUOTE_CURRENCY, logger, account_type=account_type_to_test)
            if balance_decimal is not None:
                 logger.info(f"{NEON_GREEN}Successfully connected and fetched initial balance.{RESET} ({QUOTE_CURRENCY} available: {balance_decimal:.4f})")
            else:
                 # This could happen if the currency isn't in the specified account type, or API permissions are wrong.
                 logger.warning(f"{NEON_YELLOW}Initial balance fetch for {QUOTE_CURRENCY} in {account_type_to_test} account returned None or zero. Check API key permissions (Read, Trade), ensure keys match account (Real/Testnet), verify IP whitelist if used, and confirm the account type contains {QUOTE_CURRENCY}.{RESET}")
                 # Try fetching for UNIFIED as a fallback if CONTRACT failed
                 if account_type_to_test == 'CONTRACT':
                      logger.info(f"Attempting fallback balance fetch for UNIFIED account type...")
                      balance_decimal_unified = fetch_balance(exchange, QUOTE_CURRENCY, logger, account_type='UNIFIED')
                      if balance_decimal_unified is not None:
                           logger.info(f"{NEON_GREEN}Successfully fetched balance using UNIFIED account type fallback.{RESET} ({QUOTE_CURRENCY} available: {balance_decimal_unified:.4f})")
                      else:
                           logger.warning(f"{NEON_YELLOW}Fallback balance fetch for UNIFIED account also returned None or zero.{RESET}")

        except ccxt.AuthenticationError as auth_err:
            logger.error(f"{NEON_RED}CCXT Authentication Error during initial balance fetch: {auth_err}{RESET}")
            logger.error(f"{NEON_RED}>> Critical Setup Issue: Ensure API keys are correct, have required permissions (Read, Trade), match the account type (Real/Testnet), and IP whitelist is correctly configured on Bybit if enabled.{RESET}")
            return None # Authentication failure is critical, cannot proceed.
        except ccxt.ExchangeError as balance_err:
            # Handle potential errors if account type is wrong, currency not found etc.
            bybit_code = getattr(balance_err, 'code', None) # Get Bybit specific error code if available
            logger.warning(f"{NEON_YELLOW}Exchange error during initial balance fetch (tried {account_type_to_test}): {balance_err} (Code: {bybit_code}). Continuing, but double-check API permissions, account type, and currency presence if trading fails.{RESET}")
        except Exception as balance_err:
            logger.warning(f"{NEON_YELLOW}An unexpected error occurred during initial balance fetch: {balance_err}. Continuing, but verify exchange connection and API key validity.{RESET}", exc_info=True)

        return exchange

    except ccxt.AuthenticationError as e:
        logger.error(f"{NEON_RED}CCXT Authentication Error during exchange initialization: {e}{RESET}")
        logger.error(f"{NEON_RED}>> Check API keys, permissions, IP whitelist, and Real/Testnet selection in your .env file and on Bybit.{RESET}")
    except ccxt.ExchangeError as e:
        logger.error(f"{NEON_RED}CCXT Exchange Error during initialization: {e}{RESET}")
    except ccxt.NetworkError as e:
        logger.error(f"{NEON_RED}CCXT Network Error during initialization: {e}{RESET}")
    except Exception as e:
        logger.error(f"{NEON_RED}Failed to initialize CCXT exchange due to an unexpected error: {e}{RESET}", exc_info=True)

    return None


def _determine_category(market_info: Dict) -> Optional[str]:
    """
    Helper function to determine the Bybit V5 category ('linear', 'inverse', 'spot', 'option')
    based on the CCXT market information dictionary.
    """
    if not market_info:
        return None

    # Primary check: CCXT's standardized 'type' field
    market_type = market_info.get('type') # e.g., 'spot', 'swap', 'future', 'option'

    # Secondary checks: Specific contract flags (linear/inverse)
    is_linear = market_info.get('linear', False)
    is_inverse = market_info.get('inverse', False)
    is_contract = market_info.get('contract', False) # General flag indicating it's a derivative

    if market_type in ['swap', 'future'] or is_contract:
        if is_linear: return 'linear'
        if is_inverse: return 'inverse'
        # Fallback for contracts if linear/inverse flags aren't explicitly set (less common)
        # Try inferring from settle currency
        settle_currency = market_info.get('settle')
        if settle_currency and settle_currency == market_info.get('quote'):
             return 'linear' # Settles in quote (e.g., USDT for BTC/USDT) -> Linear
        elif settle_currency and settle_currency == market_info.get('base'):
             return 'inverse' # Settles in base (e.g., BTC for BTC/USD) -> Inverse
        # Default assumption for contracts if flags/settle are unclear
        return 'linear' # Assume linear if unsure
    elif market_type == 'spot':
        return 'spot'
    elif market_type == 'option':
        return 'option'
    # Fallback if type isn't clear but contract flags are set
    elif is_linear: return 'linear'
    elif is_inverse: return 'inverse'
    else:
        # If none of the above match, return None or a default guess
        # Returning None is safer to avoid using incorrect parameters
        return None


# --- CCXT Data Fetching Functions ---
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger, market_info: Dict = None) -> Optional[Decimal]:
    """
    Fetches the current market price for a symbol using CCXT's fetch_ticker.
    Uses V5 parameters if applicable and provides fallbacks (midpoint -> last -> ask -> bid).
    Returns the price as a Decimal or None if fetching fails.
    """
    lg = logger
    try:
        lg.debug(f"Fetching ticker for {symbol}...")
        params = {}
        market_id = symbol # Default to the symbol string
        if 'bybit' in exchange.id.lower() and market_info:
             market_id = market_info.get('id', symbol) # Use exchange-specific market ID if available
             category = _determine_category(market_info)
             if category:
                 params['category'] = category
                 # CCXT usually handles mapping the standard symbol to the market_id,
                 # so explicitly setting params['symbol'] = market_id might be redundant or cause issues.
                 lg.debug(f"Using params for fetch_ticker ({symbol}): {params}")
             else:
                  lg.warning(f"Could not determine category for {symbol} to fetch ticker. Using default.")

        # Fetch the ticker data for the symbol
        ticker = exchange.fetch_ticker(symbol, params=params)

        price = None
        bid_price = ticker.get('bid')
        ask_price = ticker.get('ask')
        last_price = ticker.get('last')

        # Strategy:
        # 1. Try Bid/Ask Midpoint: Best reflects current market spread center.
        # 2. Fallback to Last Price: Price of the last executed trade.
        # 3. Fallback to Ask Price: Price sellers are asking (relevant for market buys).
        # 4. Fallback to Bid Price: Price buyers are bidding (relevant for market sells).

        # 1. Try bid/ask midpoint
        if bid_price is not None and ask_price is not None:
            try:
                bid_decimal = Decimal(str(bid_price))
                ask_decimal = Decimal(str(ask_price))
                # Ensure prices are valid and bid <= ask
                if bid_decimal > 0 and ask_decimal > 0 and bid_decimal <= ask_decimal:
                    price = (bid_decimal + ask_decimal) / 2
                    lg.debug(f"Using bid/ask midpoint for {symbol}: {price:.8f} (Bid: {bid_decimal:.8f}, Ask: {ask_decimal:.8f})")
                elif bid_decimal > ask_decimal:
                     # This state is unusual (crossed market) but can happen briefly.
                     lg.warning(f"Invalid ticker state: Bid ({bid_decimal}) > Ask ({ask_decimal}) for {symbol}. Using 'ask' price as fallback.")
                     price = ask_decimal # Use ask as a safer fallback in this unstable scenario
                else: # One or both prices were <= 0
                     lg.debug(f"Bid/Ask prices not positive ({bid_price}, {ask_price}) for {symbol}. Proceeding to next fallback.")
            except (InvalidOperation, ValueError, TypeError) as e:
                lg.warning(f"Could not parse bid/ask prices ({bid_price}, {ask_price}) for {symbol}: {e}. Proceeding to next fallback.")

        # 2. Try 'last' price if midpoint failed
        if price is None and last_price is not None:
            try:
                last_decimal = Decimal(str(last_price))
                if last_decimal > 0:
                    price = last_decimal
                    lg.debug(f"Using 'last' price as fallback for {symbol}: {price:.8f}")
                else:
                     lg.debug(f"'Last' price is not positive ({last_price}) for {symbol}. Proceeding to next fallback.")
            except (InvalidOperation, ValueError, TypeError) as e:
                lg.warning(f"Could not parse 'last' price ({last_price}) for {symbol}: {e}. Proceeding to next fallback.")

        # 3. Try 'ask' price if last failed
        if price is None and ask_price is not None:
            try:
                ask_decimal = Decimal(str(ask_price))
                if ask_decimal > 0:
                    price = ask_decimal
                    lg.warning(f"Using 'ask' price as final fallback for {symbol}: {price:.8f}")
                else:
                     lg.debug(f"'Ask' price is not positive ({ask_price}) for {symbol}. Proceeding to next fallback.")
            except (InvalidOperation, ValueError, TypeError) as e:
                lg.warning(f"Could not parse 'ask' price ({ask_price}) for {symbol}: {e}. Proceeding to next fallback.")

        # 4. Try 'bid' price as last resort
        if price is None and bid_price is not None:
            try:
                bid_decimal = Decimal(str(bid_price))
                if bid_decimal > 0:
                    price = bid_decimal
                    lg.warning(f"Using 'bid' price as final fallback for {symbol}: {price:.8f}")
                else:
                     lg.debug(f"'Bid' price is not positive ({bid_price}) for {symbol}.")
            except (InvalidOperation, ValueError, TypeError) as e:
                lg.warning(f"Could not parse 'bid' price ({bid_price}) for {symbol}: {e}.")

        # --- Final Check ---
        if price is not None and price > 0:
            return price
        else:
            lg.error(f"{NEON_RED}Failed to fetch a valid positive current price for {symbol} from ticker after all fallbacks.{RESET}")
            lg.debug(f"Ticker data received for {symbol}: {ticker}") # Log the problematic ticker
            return None

    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error fetching price for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}Exchange error fetching price for {symbol}: {e}{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error fetching price for {symbol}: {e}{RESET}", exc_info=True)
    return None


def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 250, logger: logging.Logger = None, market_info: Dict = None) -> pd.DataFrame:
    """
    Fetches OHLCV kline data using CCXT with retries for network issues,
    V5 parameters, data validation, and conversion to a pandas DataFrame.
    """
    lg = logger or logging.getLogger(__name__) # Use provided logger or default
    if not exchange.has['fetchOHLCV']:
        lg.error(f"Exchange {exchange.id} does not support fetchOHLCV method.")
        return pd.DataFrame()

    ohlcv = None
    ccxt_tf = CCXT_INTERVAL_MAP.get(timeframe, timeframe) # Use mapped timeframe if available

    for attempt in range(MAX_API_RETRIES + 1):
        try:
            lg.debug(f"Fetching klines for {symbol}, timeframe={ccxt_tf}, limit={limit} (Attempt {attempt+1}/{MAX_API_RETRIES + 1})")
            params = {}
            market_id = symbol
            if 'bybit' in exchange.id.lower() and market_info:
                 market_id = market_info.get('id', symbol)
                 category = _determine_category(market_info)
                 if category:
                     params['category'] = category
                     # params['symbol'] = market_id # CCXT handles symbol->id mapping
                     lg.debug(f"Using params for fetch_ohlcv ({symbol}): {params}")
                 else:
                      lg.warning(f"Could not determine category for {symbol} to fetch klines. Using default.")

            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=ccxt_tf, limit=limit, params=params)

            # Basic validation: check if data is returned and is a non-empty list
            if isinstance(ohlcv, list) and len(ohlcv) > 0:
                lg.debug(f"Successfully fetched {len(ohlcv)} raw kline entries for {symbol}.")
                break # Success, exit retry loop
            else:
                lg.warning(f"fetch_ohlcv returned type {type(ohlcv)} or empty list (length {len(ohlcv) if isinstance(ohlcv, list) else 'N/A'}) for {symbol} (Attempt {attempt+1}).")
                # No immediate retry here, let the loop handle standard delay if attempts remain

        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            lg.warning(f"Network error/timeout fetching klines for {symbol} (Attempt {attempt + 1}/{MAX_API_RETRIES + 1}): {e}.")
            if attempt < MAX_API_RETRIES:
                lg.info(f"Retrying in {RETRY_DELAY_SECONDS}s...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                lg.error(f"{NEON_RED}Max retries reached fetching klines for {symbol} after network errors: {e}{RESET}")
                # Do not re-raise here, return empty DataFrame below
        except ccxt.RateLimitExceeded as e:
            # Handle rate limit by waiting longer, potentially using retry-after header
            retry_after_ms = getattr(e, 'retryAfter', None) # Check if header provides wait time
            if retry_after_ms is not None:
                wait_time = (retry_after_ms / 1000.0) + 0.2 # Use header + small buffer
                lg.warning(f"Rate limit exceeded fetching klines for {symbol}. Retrying after {wait_time:.2f}s (from Retry-After header)... (Attempt {attempt+1})")
            else:
                # Fallback: Use exchange's default rateLimit property or a larger fixed delay
                wait_time = (exchange.rateLimit / 1000.0) + 1.0 # Use CCXT's rateLimit property + buffer
                lg.warning(f"Rate limit exceeded fetching klines for {symbol}. Retrying after {wait_time:.2f}s (calculated)... (Attempt {attempt+1})")
            time.sleep(wait_time)
            # Rate limit doesn't consume a standard 'retry', loop continues
        except ccxt.ExchangeError as e:
            # Non-recoverable exchange errors (e.g., invalid symbol, bad request)
            lg.error(f"{NEON_RED}Exchange error fetching klines for {symbol}: {e}{RESET}")
            # Do not retry these, return empty DataFrame below
            ohlcv = None # Ensure ohlcv is None
            break # Exit retry loop
        except Exception as e:
            # Unexpected errors during fetch
            lg.error(f"{NEON_RED}Unexpected error fetching klines for {symbol}: {e}{RESET}", exc_info=True)
            ohlcv = None # Ensure ohlcv is None
            break # Exit retry loop

        # If attempt was not successful (except for RateLimit which handles its own sleep) and retries remain:
        if attempt < MAX_API_RETRIES and ohlcv is None:
            lg.info(f"Retrying fetch in {RETRY_DELAY_SECONDS}s...")
            time.sleep(RETRY_DELAY_SECONDS)


    # --- Process Data (if fetched successfully) ---
    if not isinstance(ohlcv, list) or not ohlcv: # Check if list is empty or not a list after retries/errors
        lg.warning(f"{NEON_YELLOW}No valid kline data returned for {symbol} {ccxt_tf} after all attempts.{RESET}")
        return pd.DataFrame()

    try:
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if df.empty:
            lg.warning(f"{NEON_YELLOW}Kline data DataFrame is empty for {symbol} {ccxt_tf} immediately after creation.{RESET}")
            return df # Return empty DataFrame

        # Convert timestamp to datetime (UTC) and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce', utc=True)
        df.dropna(subset=['timestamp'], inplace=True) # Drop rows where timestamp conversion failed
        if df.empty:
            lg.warning(f"{NEON_YELLOW}Kline data DataFrame empty after timestamp conversion/dropna for {symbol} {ccxt_tf}.{RESET}")
            return df
        df.set_index('timestamp', inplace=True)

        # Ensure OHLCV columns are numeric, coerce errors to NaN
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- Data Cleaning ---
        initial_len = len(df)
        # Drop rows with NaN in critical price columns (O, H, L, C)
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        # Optional: Drop rows with NaN volume if your strategy requires it
        # df.dropna(subset=['volume'], inplace=True)
        # Drop rows with non-positive close price (often indicates invalid data)
        df = df[df['close'] > 0]
        # Optional: Drop rows with zero volume if needed
        # df = df[df['volume'] > 0]

        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
            lg.debug(f"Dropped {rows_dropped} rows with NaN/invalid price/volume data during cleaning for {symbol}.")

        if df.empty:
            lg.warning(f"{NEON_YELLOW}Kline data for {symbol} {ccxt_tf} is empty after processing and cleaning.{RESET}")
            return pd.DataFrame()

        df.sort_index(inplace=True) # Ensure data is sorted chronologically by timestamp index
        lg.info(f"Successfully fetched and processed {len(df)} valid klines for {symbol} {ccxt_tf}")
        return df

    except Exception as e: # Catch errors during DataFrame processing/cleaning
        lg.error(f"{NEON_RED}Error processing kline data into DataFrame for {symbol}: {e}{RESET}", exc_info=True)
        return pd.DataFrame()


def fetch_orderbook_ccxt(exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger, market_info: Dict = None) -> Optional[Dict]:
    """
    Fetches order book data using CCXT with retries, V5 parameters, and basic validation.
    Returns the raw order book dictionary or None on failure.
    """
    lg = logger
    if not exchange.has['fetchOrderBook']:
        lg.error(f"Exchange {exchange.id} does not support fetchOrderBook method.")
        return None

    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching order book for {symbol}, limit={limit} (Attempt {attempts+1}/{MAX_API_RETRIES + 1})")
            params = {}
            market_id = symbol
            if 'bybit' in exchange.id.lower() and market_info:
                 market_id = market_info.get('id', symbol)
                 category = _determine_category(market_info)
                 if category:
                     params['category'] = category
                     # params['symbol'] = market_id # CCXT handles mapping
                     lg.debug(f"Using params for fetch_order_book ({symbol}): {params}")
                 else:
                     lg.warning(f"Could not determine category for {symbol} to fetch order book. Using default.")


            orderbook = exchange.fetch_order_book(symbol, limit=limit, params=params)

            # --- Validation ---
            if not isinstance(orderbook, dict):
                lg.warning(f"fetch_order_book did not return a dictionary for {symbol} (Got: {type(orderbook)}). Attempt {attempts + 1}.")
            elif 'bids' not in orderbook or 'asks' not in orderbook:
                lg.warning(f"Orderbook structure missing 'bids' or 'asks' key for {symbol}. Attempt {attempts + 1}.")
            elif not isinstance(orderbook.get('bids'), list) or not isinstance(orderbook.get('asks'), list):
                lg.warning(f"Invalid orderbook structure: 'bids' or 'asks' are not lists for {symbol}. Attempt {attempts + 1}.")
            # Allow empty bids/asks lists if the structure is otherwise correct (might be thin liquidity)
            elif not orderbook.get('bids') and not orderbook.get('asks'):
                 lg.debug(f"Orderbook received for {symbol} but both bids and asks lists are empty.")
                 return orderbook # Return the empty book
            else:
                 # Basic check: ensure first bid price <= first ask price if both exist
                 valid_spread = True
                 if orderbook['bids'] and orderbook['asks']:
                      try:
                           first_bid = Decimal(str(orderbook['bids'][0][0]))
                           first_ask = Decimal(str(orderbook['asks'][0][0]))
                           if first_bid > first_ask:
                                lg.warning(f"Orderbook spread is crossed for {symbol}: Bid {first_bid} > Ask {first_ask}. Data might be stale.")
                                valid_spread = False # Potentially problematic, but proceed
                      except (IndexError, InvalidOperation, ValueError, TypeError):
                           lg.warning(f"Could not validate orderbook spread for {symbol}.")

                 lg.debug(f"Successfully fetched orderbook for {symbol} with {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks.")
                 return orderbook # Success

        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            lg.warning(f"{NEON_YELLOW}Orderbook fetch network error/timeout for {symbol}: {e}. Retrying... (Attempt {attempts + 1}/{MAX_API_RETRIES + 1}){RESET}")
        except ccxt.RateLimitExceeded as e:
            retry_after_ms = getattr(e, 'retryAfter', None)
            if retry_after_ms is not None: wait_time = (retry_after_ms / 1000.0) + 0.2
            else: wait_time = (exchange.rateLimit / 1000.0) + 1.0
            lg.warning(f"Rate limit exceeded fetching orderbook for {symbol}. Retrying in {wait_time:.2f}s... (Attempt {attempts+1})")
            time.sleep(wait_time)
            # Rate limit doesn't count as a standard attempt for the loop counter
            continue # Skip standard delay and attempt increment
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error fetching orderbook for {symbol}: {e}{RESET}")
            return None # Don't retry definitive exchange errors
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching orderbook for {symbol}: {e}{RESET}", exc_info=True)
            return None # Don't retry unexpected errors

        # Increment attempt counter and wait before retrying network/validation issues
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)

    lg.error(f"{NEON_RED}Max retries reached fetching orderbook for {symbol}.{RESET}")
    return None


# --- Trading Analyzer Class ---
class TradingAnalyzer:
    """
    Analyzes historical OHLCV data using pandas_ta to calculate technical indicators
    and generates weighted trading signals based on configured strategies.
    Manages Fibonacci levels and provides access to latest indicator values.
    Interacts with a shared state dictionary for symbol-specific states like break-even.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        logger: logging.Logger,
        config: Dict[str, Any],
        market_info: Dict[str, Any], # Pass full market info for precision, limits, etc.
        symbol_state: Dict[str, Any], # Pass the mutable state dictionary for this symbol
    ) -> None:
        """
        Initializes the TradingAnalyzer.

        Args:
            df: pandas DataFrame with OHLCV data (index='timestamp').
            logger: Logger instance for this analyzer.
            config: The main bot configuration dictionary.
            market_info: The CCXT market info dictionary for the symbol, including derived values.
            symbol_state: Mutable dictionary holding persistent state for this symbol (e.g., 'break_even_triggered').
        """
        self.df = df # Expects index 'timestamp' and columns 'open', 'high', 'low', 'close', 'volume'
        self.logger = logger
        self.config = config

        if not market_info:
            raise ValueError("TradingAnalyzer requires valid market_info.")
        self.market_info = market_info # Store the comprehensive market info
        self.symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL') # Get CCXT symbol
        self.interval = config.get("interval", "UNKNOWN_INTERVAL") # User-friendly interval string
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval, "UNKNOWN_INTERVAL") # CCXT format interval

        self.indicator_values: Dict[str, Optional[float]] = {} # Stores latest calculated indicator values as floats
        self.signals: Dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 1} # Simple signal state (1 = active)
        self.active_weight_set_name = config.get("active_weight_set", "default")
        self.weights = config.get("weight_sets",{}).get(self.active_weight_set_name, {})
        self.fib_levels_data: Dict[str, Optional[Decimal]] = {} # Stores calculated fib levels (as Decimals or None)
        # Stores the actual column names generated by pandas_ta for each indicator.
        # Keys are our internal indicator names (e.g., "EMA_Short"), values are DataFrame column names (e.g., "EMA_8").
        self.ta_column_names: Dict[str, Optional[str]] = {}

        # --- Symbol State (mutable reference) ---
        # Use the passed-in state dictionary to allow persistence across analysis cycles.
        if 'break_even_triggered' not in symbol_state:
             symbol_state['break_even_triggered'] = False # Initialize if first time for this symbol
             self.logger.debug(f"Initialized state for {self.symbol}: {symbol_state}")
        self.symbol_state = symbol_state # Keep a reference to the mutable dictionary

        if not self.weights:
            logger.error(f"{NEON_RED}Active weight set '{self.active_weight_set_name}' not found or empty in config for {self.symbol}. Indicator weighting will be disabled.{RESET}")
            self.weights = {} # Ensure it's an empty dict if invalid

        # --- Initial Calculations ---
        if not self.df.empty:
            self._calculate_all_indicators() # Calculate indicators and populate self.ta_column_names
            self._update_latest_indicator_values() # Populate self.indicator_values
            self.calculate_fibonacci_levels() # Populate self.fib_levels_data
        else:
            self.logger.warning(f"DataFrame is empty for {self.symbol}. Skipping indicator and Fibonacci calculations.")

    # --- State Management Properties ---
    @property
    def break_even_triggered(self) -> bool:
        """Gets the break-even status from the shared symbol state dictionary."""
        return self.symbol_state.get('break_even_triggered', False)

    @break_even_triggered.setter
    def break_even_triggered(self, value: bool):
        """Sets the break-even status in the shared symbol state dictionary."""
        if self.symbol_state.get('break_even_triggered') != value:
            self.symbol_state['break_even_triggered'] = value
            self.logger.info(f"Updated break_even_triggered state for {self.symbol} to: {value}")
        else:
             self.logger.debug(f"Break-even state for {self.symbol} already set to {value}.")

    # --- Internal Helper ---
    def _get_ta_col_name(self, base_name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Helper to safely retrieve the actual DataFrame column name for a given indicator
        from the mapping populated during indicator calculation.
        """
        # Note: This relies on self.ta_column_names being populated correctly.
        # A more complex version could search df.columns if the stored name is missing or None.
        return self.ta_column_names.get(base_name, default)

    # --- Indicator Calculation ---
    def _calculate_all_indicators(self) -> None:
        """Calculates all configured technical indicators using pandas_ta and stores column names."""
        self.logger.debug(f"Calculating indicators for {self.symbol} using pandas_ta...")
        if self.df.empty:
            self.logger.warning(f"DataFrame is empty for {self.symbol}, cannot calculate indicators.")
            return

        # Retrieve parameters from config, falling back to defaults
        atr_period = self.config.get("atr_period", DEFAULT_ATR_PERIOD)
        ema_short_period = self.config.get("ema_short_period", DEFAULT_EMA_SHORT_PERIOD)
        ema_long_period = self.config.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD)
        rsi_period = self.config.get("rsi_period", DEFAULT_RSI_WINDOW)
        bb_period = self.config.get("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD)
        bb_std = self.config.get("bollinger_bands_std_dev", DEFAULT_BOLLINGER_BANDS_STD_DEV)
        cci_window = self.config.get("cci_window", DEFAULT_CCI_WINDOW)
        wr_window = self.config.get("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW)
        mfi_window = self.config.get("mfi_window", DEFAULT_MFI_WINDOW)
        stoch_rsi_window = self.config.get("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW)
        stoch_rsi_rsi_window = self.config.get("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW)
        stoch_rsi_k = self.config.get("stoch_rsi_k", DEFAULT_K_WINDOW)
        stoch_rsi_d = self.config.get("stoch_rsi_d", DEFAULT_D_WINDOW)
        psar_af = self.config.get("psar_af", DEFAULT_PSAR_AF)
        psar_max_af = self.config.get("psar_max_af", DEFAULT_PSAR_MAX_AF)
        sma10_window = self.config.get("sma_10_window", DEFAULT_SMA_10_WINDOW)
        momentum_period = self.config.get("momentum_period", DEFAULT_MOMENTUM_PERIOD)
        volume_ma_period = self.config.get("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)

        # Ensure Bollinger Bands standard deviation is a float
        try:
            bb_std_float = float(bb_std)
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid Bollinger Bands std dev '{bb_std}' in config. Using default {DEFAULT_BOLLINGER_BANDS_STD_DEV}.")
            bb_std_float = DEFAULT_BOLLINGER_BANDS_STD_DEV

        # --- Define the pandas_ta Strategy ---
        # This bundles multiple indicator calculations into one call.
        MyStrategy = ta.Strategy(
            name="MultiIndicatorStrategy_Whale",
            description="Combines various TA indicators for signal generation",
            ta=[
                # Volatility / Risk Management Aid
                {"kind": "atr", "length": atr_period, "mamode": "ema"}, # Average True Range (using EMA smoothing)
                # Trend / Alignment Indicators
                {"kind": "ema", "length": ema_short_period}, # Short Exponential Moving Average
                {"kind": "ema", "length": ema_long_period},  # Long Exponential Moving Average
                {"kind": "vwap"},                            # Volume Weighted Average Price (typically daily reset)
                # Momentum / Oscillators
                {"kind": "rsi", "length": rsi_period},       # Relative Strength Index
                {"kind": "cci", "length": cci_window},       # Commodity Channel Index
                {"kind": "willr", "length": wr_window},      # Williams %R
                {"kind": "mfi", "length": mfi_window},       # Money Flow Index
                # Stochastic RSI (combines RSI and Stochastic)
                {"kind": "stochrsi", "length": stoch_rsi_window, "rsi_length": stoch_rsi_rsi_window, "k": stoch_rsi_k, "d": stoch_rsi_d},
                {"kind": "mom", "length": momentum_period}, # Momentum indicator
                # Bollinger Bands (Volatility and Mean Reversion)
                {"kind": "bbands", "length": bb_period, "std": bb_std_float},
                # Volume Confirmation
                {"kind": "sma", "close": "volume", "length": volume_ma_period, "prefix": "VOL"}, # Simple Moving Average of Volume
                # Other Trend/Signal Indicators
                {"kind": "psar", "af": psar_af, "max_af": psar_max_af}, # Parabolic SAR
                {"kind": "sma", "length": sma10_window}, # Simple Moving Average (e.g., 10 period)
            ]
        )

        try:
            # Apply the strategy to the DataFrame. This appends new columns directly to self.df.
            self.df.ta.strategy(MyStrategy)

            # --- Store Actual Generated Column Names ---
            # Construct expected names based on pandas_ta conventions and parameters.
            # This mapping allows us to reference indicators consistently later, even if column names are complex.
            self.ta_column_names = {
                "ATR": f"ATRr_{atr_period}", # ATR usually appends 'r' and length
                "EMA_Short": f"EMA_{ema_short_period}",
                "EMA_Long": f"EMA_{ema_long_period}",
                "RSI": f"RSI_{rsi_period}",
                "BBL": f"BBL_{bb_period}_{bb_std_float:.1f}", # Lower BBand
                "BBM": f"BBM_{bb_period}_{bb_std_float:.1f}", # Middle BBand (SMA)
                "BBU": f"BBU_{bb_period}_{bb_std_float:.1f}", # Upper BBand
                "CCI": f"CCI_{cci_window}_0.015", # Pandas TA adds constant suffix to CCI
                "WR": f"WILLR_{wr_window}",
                "MFI": f"MFI_{mfi_window}",
                "STOCHRSIk": f"STOCHRSIk_{stoch_rsi_window}_{stoch_rsi_rsi_window}_{stoch_rsi_k}_{stoch_rsi_d}",
                "STOCHRSId": f"STOCHRSId_{stoch_rsi_window}_{stoch_rsi_rsi_window}_{stoch_rsi_k}_{stoch_rsi_d}",
                "MOM": f"MOM_{momentum_period}",
                "VOL_MA": f"VOL_SMA_{volume_ma_period}", # Volume MA uses prefix specified
                "PSARl": f"PSARl_{psar_af}_{psar_max_af}", # Long PSAR signal line
                "PSARs": f"PSARs_{psar_af}_{psar_max_af}", # Short PSAR signal line
                "SMA10": f"SMA_{sma10_window}",
                "VWAP": "VWAP_D", # VWAP default is often daily ('D') - verify if different timeframes used
            }

            # --- Verify Column Existence ---
            # Check if the expected columns were actually created in the DataFrame.
            missing_cols = []
            for key, expected_col_name in self.ta_column_names.items():
                 # Special check for VWAP common variations
                 if key == "VWAP" and expected_col_name not in self.df.columns:
                     if "VWAP" in self.df.columns: # Check for simple 'VWAP' name
                         self.ta_column_names[key] = "VWAP" # Update mapping if found
                     else:
                         # Try other potential VWAP names if needed, e.g., based on interval
                         # vwap_interval_name = f"VWAP_{self.ccxt_interval}" # Example
                         # if vwap_interval_name in self.df.columns:
                         #     self.ta_column_names[key] = vwap_interval_name
                         # else:
                         missing_cols.append(expected_col_name) # Still missing after checks
                 elif expected_col_name not in self.df.columns:
                     missing_cols.append(expected_col_name)

            if missing_cols:
                 self.logger.warning(f"Following expected indicator columns not found in DataFrame after calculation for {self.symbol}: {missing_cols}. Check pandas_ta version or strategy definition. Related indicator values will be None.")
                 # Set missing column names to None in our mapping to prevent KeyErrors later
                 for key, col_name in list(self.ta_column_names.items()): # Iterate over copy of items
                      if col_name in missing_cols:
                           self.ta_column_names[key] = None

            self.logger.debug(f"Indicators calculated for {self.symbol}. DataFrame columns now include: {list(self.df.columns)}")

        except ImportError as ie:
             self.logger.critical(f"{NEON_RED}ImportError calculating indicators: {ie}. Is pandas_ta installed correctly? Disabling indicator calculations.{RESET}")
             self.ta_column_names = {key: None for key in self.ta_column_names} # Mark all as unavailable
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error calculating indicators via pandas_ta strategy for {self.symbol}: {e}{RESET}", exc_info=True)
            # Mark all columns as unavailable if the strategy failed catastrophically
            self.ta_column_names = {key: None for key in self.ta_column_names}


    def _update_latest_indicator_values(self) -> None:
        """Extracts the latest values of all calculated indicators from the DataFrame."""
        self.indicator_values = {} # Reset values for this cycle
        if self.df.empty or self.df.index.empty:
            self.logger.warning(f"DataFrame is empty or has no index, cannot update latest indicator values for {self.symbol}.")
            return

        try:
            # Get the last row (latest data) of the DataFrame
            latest_data = self.df.iloc[-1]
            latest_timestamp = latest_data.name # Get the timestamp index of the last row
        except IndexError:
            self.logger.error(f"Cannot access the last row of the DataFrame for {self.symbol} (IndexError). DataFrame might be malformed.")
            return

        # Iterate through our mapped indicator names and their corresponding DataFrame column names
        for key, col_name in self.ta_column_names.items():
            if col_name and col_name in latest_data.index: # Check if the expected column exists in the latest data Series
                value = latest_data[col_name]
                # Convert the value to float, handling potential NaN, None, or pandas NA types
                try:
                    if pd.isna(value): # Catches numpy NaN, None, pandas NA
                         self.indicator_values[key] = None
                    else:
                         # Ensure conversion to float handles various numeric types
                         self.indicator_values[key] = float(value)
                except (ValueError, TypeError):
                    self.logger.warning(f"Could not convert value '{value}' (type: {type(value)}) for indicator '{key}' ({col_name}) to float. Setting to None.")
                    self.indicator_values[key] = None
            else:
                # Log if the column was expected (i.e., col_name was not None) but is missing in the data
                if col_name:
                    self.logger.debug(f"Column '{col_name}' for indicator '{key}' not found in the latest data row for {self.symbol}.")
                self.indicator_values[key] = None # Mark indicator as unavailable if column is missing or mapping failed

        # Also add the latest basic OHLCV data to the indicator_values dictionary for easy access
        for col in ['close', 'high', 'low', 'open', 'volume']:
            try:
                value = latest_data[col]
                self.indicator_values[col] = float(value) if pd.notna(value) else None
            except (KeyError):
                self.logger.debug(f"Base column '{col}' not found in latest data for {self.symbol}.")
                self.indicator_values[col] = None
            except (ValueError, TypeError):
                 self.logger.warning(f"Could not convert base value '{value}' for '{col}' to float. Setting to None.")
                 self.indicator_values[col] = None

        # Log a summary of the latest extracted values at DEBUG level for verification
        log_vals = {k: f"{v:.4f}" if isinstance(v, float) else str(v) for k, v in self.indicator_values.items()}
        self.logger.debug(f"Latest Indicator Values ({self.symbol} @ {latest_timestamp}): {log_vals}")


    # --- Fibonacci Calculation ---
    def calculate_fibonacci_levels(self, window: Optional[int] = None) -> Dict[str, Optional[Decimal]]:
        """
        Calculates Fibonacci retracement levels based on the high/low over a specified window.
        Uses Decimal for precision and quantizes levels to the market's minimum tick size.

        Args:
            window: The number of recent periods (candles) to consider for high/low. Uses config default if None.

        Returns:
            A dictionary where keys are Fibonacci level names (e.g., "Fib_38.2%")
            and values are the calculated price levels as Decimals (or None if calculation fails).
            Updates self.fib_levels_data.
        """
        window = window or self.config.get("fibonacci_window", DEFAULT_FIB_WINDOW)
        self.fib_levels_data = {} # Reset previous levels

        if len(self.df) < window:
            self.logger.debug(f"Not enough data ({len(self.df)}) for Fibonacci window ({window}) on {self.symbol}. Skipping calculation.")
            return {}

        df_slice = self.df.tail(window)
        try:
            # Find the absolute highest high and lowest low in the window, dropping NaNs first
            high_price_raw = df_slice["high"].dropna().max()
            low_price_raw = df_slice["low"].dropna().min()

            # Check if valid high/low prices were found
            if pd.isna(high_price_raw) or pd.isna(low_price_raw):
                self.logger.warning(f"Could not find valid high/low prices in the last {window} periods for Fibonacci calculation on {self.symbol}.")
                return {}

            # Convert valid prices to Decimal
            high_price = Decimal(str(high_price_raw))
            low_price = Decimal(str(low_price_raw))
            price_range = high_price - low_price

            # Get minimum price increment (tick size) for quantization
            min_tick = self.get_min_tick_size()
            if min_tick is None or min_tick <= 0:
                self.logger.warning(f"Invalid min_tick_size ({min_tick}) derived for {self.symbol}. Fibonacci levels will not be quantized.")
                min_tick = None # Disable quantization if tick size is invalid

            levels: Dict[str, Optional[Decimal]] = {}
            if price_range > 0:
                for level_pct in FIB_LEVELS:
                    level_name = f"Fib_{level_pct * 100:.1f}%"
                    # Standard Retracement Calculation: High - (Range * Percentage)
                    # Assumes recent move was down from high to low; levels measure potential bounce up.
                    # For uptrend assumption (Low + Range * Pct), reverse the logic if needed based on trend context.
                    level_price_raw = high_price - (price_range * Decimal(str(level_pct)))

                    # Quantize the calculated level to the nearest valid price tick
                    if min_tick:
                        # Quantize DOWN for retracement levels (safer side)
                        level_price_quantized = (level_price_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                        # Ensure quantization didn't push below the low (can happen with large ticks)
                        level_price_quantized = max(level_price_quantized, low_price)
                    else:
                        level_price_quantized = level_price_raw # Use raw value if quantization is disabled

                    levels[level_name] = level_price_quantized
            else:
                 # If high == low (zero range), all levels are the same price.
                 self.logger.debug(f"Fibonacci range is zero (High={high_price}, Low={low_price}) for {self.symbol} in window {window}. Setting all levels to this price.")
                 level_price_quantized = high_price # Already a valid price
                 if min_tick: # Still quantize the single price level if possible
                      level_price_quantized = (high_price / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                 for level_pct in FIB_LEVELS:
                     levels[f"Fib_{level_pct * 100:.1f}%"] = level_price_quantized

            self.fib_levels_data = levels
            # Log calculated levels as strings for readability
            log_levels = {k: f"{v:.{self.get_price_precision()}f}" if v is not None else 'N/A' for k, v in levels.items()}
            self.logger.debug(f"Calculated Fibonacci levels for {self.symbol}: {log_levels}")
            return levels

        except (InvalidOperation, ValueError, TypeError, KeyError) as e:
            self.logger.error(f"{NEON_RED}Fibonacci calculation error for {self.symbol}: {e}{RESET}", exc_info=True)
            self.fib_levels_data = {} # Ensure data is cleared on error
            return {}
        except Exception as e:
             self.logger.error(f"{NEON_RED}Unexpected error during Fibonacci calculation for {self.symbol}: {e}{RESET}", exc_info=True)
             self.fib_levels_data = {}
             return {}


    # --- Market Info Helpers (using pre-derived values from self.market_info) ---
    def get_price_precision(self) -> int:
        """Gets the number of decimal places for price from the stored market info."""
        # Use the 'price_precision_digits' value pre-calculated and stored in market_info
        precision_digits = self.market_info.get('price_precision_digits')
        if precision_digits is None:
            self.logger.warning(f"Price precision digits not found in derived market info for {self.symbol}. Using default 8.")
            return 8 # Default fallback
        return int(precision_digits) # Ensure integer

    def get_min_tick_size(self) -> Optional[Decimal]:
        """Gets the minimum price increment (tick size) as a Decimal from the stored market info."""
        # Use the 'min_tick_size' value pre-calculated and stored in market_info
        min_tick = self.market_info.get('min_tick_size')
        if min_tick is None:
            self.logger.warning(f"Min tick size not found in derived market info for {self.symbol}. Returning None.")
            return None
        # Ensure it's a Decimal
        return Decimal(str(min_tick)) if not isinstance(min_tick, Decimal) else min_tick

    def get_amount_precision(self) -> int:
        """Gets the number of decimal places for amount/quantity from the stored market info."""
        precision_digits = self.market_info.get('amount_precision_digits')
        if precision_digits is None:
            self.logger.warning(f"Amount precision digits not found in derived market info for {self.symbol}. Using default 3.")
            return 3 # Default fallback
        return int(precision_digits) # Ensure integer

    def get_min_order_amount(self) -> Optional[Decimal]:
        """Gets the minimum order amount (in base currency) as a Decimal from the stored market info."""
        min_amount = self.market_info.get('min_order_amount')
        if min_amount is None:
            self.logger.warning(f"Min order amount not found in derived market info for {self.symbol}. Returning None.")
            return None
        return Decimal(str(min_amount)) if not isinstance(min_amount, Decimal) else min_amount

    def get_min_order_cost(self) -> Optional[Decimal]:
        """Gets the minimum order cost (in quote currency) as a Decimal from the stored market info."""
        min_cost = self.market_info.get('min_order_cost')
        # It's valid for min_cost to be None for some markets (e.g., derivatives often rely on min amount)
        if min_cost is None:
            return None
        return Decimal(str(min_cost)) if not isinstance(min_cost, Decimal) else min_cost


    # --- Individual Indicator Signal Checks ---
    # These functions return a float score, typically between -1.0 (strong sell) and 1.0 (strong buy).
    # 0.0 represents a neutral or unavailable signal.

    def _check_ema_alignment(self) -> float:
        """Checks if the short-term EMA is above or below the long-term EMA."""
        ema_short = self.indicator_values.get("EMA_Short")
        ema_long = self.indicator_values.get("EMA_Long")
        if ema_short is None or ema_long is None: return 0.0 # Not enough data or calculation failed
        if ema_short > ema_long: return 1.0 # Bullish alignment
        if ema_short < ema_long: return -1.0 # Bearish alignment
        return 0.0 # EMAs are exactly equal (rare)

    def _check_momentum(self) -> float:
        """Checks the Momentum indicator (MOM). Positive MOM suggests upward momentum."""
        mom = self.indicator_values.get("MOM")
        if mom is None: return 0.0
        # Simple check: positive -> bullish, negative -> bearish.
        # Could be scaled based on typical range or rate of change for stronger signals.
        if mom > 0: return 1.0
        if mom < 0: return -1.0
        return 0.0

    def _check_volume_confirmation(self) -> float:
        """Checks if the current volume is significantly above its moving average."""
        volume = self.indicator_values.get("volume") # Use the base 'volume' key
        vol_ma = self.indicator_values.get("VOL_MA") # Volume Moving Average
        multiplier = self.config.get("volume_confirmation_multiplier", 1.5)
        if volume is None or vol_ma is None or vol_ma <= 0: return 0.0 # Cannot compare if data missing or MA is zero
        # This check provides confirmation strength, not direction.
        # A high score (1.0) means volume supports the current price move (whatever direction it is).
        # A low score (0.0) means volume is not confirming the move.
        return 1.0 if volume > (vol_ma * multiplier) else 0.0

    def _check_stoch_rsi(self) -> float:
        """
        Checks Stochastic RSI K and D lines, primarily looking for crossovers
        in overbought/oversold zones.
        """
        k = self.indicator_values.get("STOCHRSIk")
        d = self.indicator_values.get("STOCHRSId")
        oversold = float(self.config.get("stoch_rsi_oversold_threshold", 25))
        overbought = float(self.config.get("stoch_rsi_overbought_threshold", 75))
        if k is None or d is None: return 0.0

        # Strongest signals: Crossovers within extreme zones
        if k > d and k < oversold and d < oversold: return 1.0  # Bullish crossover below oversold line
        if k < d and k > overbought and d > overbought: return -1.0 # Bearish crossover above overbought line

        # Weaker signals: Deep in extreme zones (potential reversal)
        if k < oversold and d < oversold: return 0.8 # Both lines deep in oversold -> potential buy soon
        if k > overbought and d > overbought: return -0.8 # Both lines deep in overbought -> potential sell soon

        # General momentum signals: K line relative to D line
        if k > d : return 0.5 # K above D generally indicates bullish momentum
        if k < d : return -0.5 # K below D generally indicates bearish momentum

        return 0.0 # Neutral / lines crossing mid-range

    def _check_rsi(self) -> float:
        """Checks the standard RSI level for overbought/oversold conditions."""
        rsi = self.indicator_values.get("RSI")
        if rsi is None: return 0.0
        # Using standard 30/70 levels for strong signals, 40/60 for weaker leanings.
        if rsi < 30: return 1.0  # Strong Oversold -> Bullish signal
        if rsi > 70: return -1.0 # Strong Overbought -> Bearish signal
        if rsi < 40: return 0.5  # Leaning oversold
        if rsi > 60: return -0.5  # Leaning overbought
        return 0.0 # Neutral range (e.g., 40-60)

    def _check_bollinger_bands(self, current_price: Decimal) -> float:
        """Checks the current price relative to the Bollinger Bands."""
        bbl = self.indicator_values.get("BBL") # Lower Band
        bbu = self.indicator_values.get("BBU") # Upper Band
        bbm = self.indicator_values.get("BBM") # Middle Band (SMA)
        if bbl is None or bbu is None or bbm is None or current_price is None: return 0.0

        try:
            price_f = float(current_price) # Convert Decimal price for comparison with float indicators
            # Strong signals: Price breaking outside the bands (potential reversal or breakout continuation)
            if price_f < bbl: return 1.0 # Price below lower band -> Strong bullish reversal potential
            if price_f > bbu: return -1.0 # Price above upper band -> Strong bearish reversal potential

            # Optional: Weaker signals based on crossing the middle band
            # if price_f > bbm: return 0.3 # Price above middle band -> Weak bullish bias
            # if price_f < bbm: return -0.3 # Price below middle band -> Weak bearish bias

            return 0.0 # Price is within the bands and not touching extremes
        except (ValueError, TypeError):
             self.logger.warning(f"Could not convert current price {current_price} to float for Bollinger Band check.")
             return 0.0

    def _check_vwap(self, current_price: Decimal) -> float:
        """Checks if the current price is above or below the Volume Weighted Average Price (VWAP)."""
        vwap = self.indicator_values.get("VWAP")
        if vwap is None or current_price is None: return 0.0
        try:
            price_f = float(current_price)
            # Simple check: Above VWAP is bullish, below is bearish.
            if price_f > vwap: return 1.0
            if price_f < vwap: return -1.0
            return 0.0 # Price is exactly on VWAP
        except (ValueError, TypeError):
             self.logger.warning(f"Could not convert current price {current_price} to float for VWAP check.")
             return 0.0

    def _check_cci(self) -> float:
        """Checks the Commodity Channel Index (CCI) for extreme overbought/oversold levels."""
        cci = self.indicator_values.get("CCI")
        if cci is None: return 0.0
        # Using +/- 150 for strong signals, +/- 80 for weaker signals. Adjust thresholds as needed.
        if cci < -150: return 1.0 # Strong Oversold -> Bullish signal
        if cci > 150: return -1.0 # Strong Overbought -> Bearish signal
        if cci < -80: return 0.5  # Oversold zone
        if cci > 80: return -0.5  # Overbought zone
        return 0.0 # Neutral zone

    def _check_wr(self) -> float:
        """Checks the Williams %R oscillator for overbought/oversold levels."""
        wr = self.indicator_values.get("WR") # Williams %R typically ranges from -100 to 0
        if wr is None: return 0.0
        # Standard levels: Below -80 is oversold, above -20 is overbought.
        if wr < -80: return 1.0 # Oversold -> Bullish signal
        if wr > -20: return -1.0 # Overbought -> Bearish signal
        # Optional: Add intermediate levels, e.g., crossing -50
        if wr < -50: return 0.3 # Approaching oversold / below midpoint
        if wr > -50: return -0.3 # Approaching overbought / above midpoint
        return 0.0 # Mid-range

    def _check_psar(self, current_price: Decimal) -> float:
        """Checks the current price relative to the Parabolic SAR (PSAR) dots."""
        # PSAR plots dots below price in an uptrend (PSARl has value) and above price in a downtrend (PSARs has value).
        psar_long_val = self.indicator_values.get("PSARl") # Value appears when SAR is below price (potential uptrend)
        psar_short_val = self.indicator_values.get("PSARs") # Value appears when SAR is above price (potential downtrend)

        # Check if either value is valid (not None and not NaN)
        is_long_signal = psar_long_val is not None and not np.isnan(psar_long_val)
        is_short_signal = psar_short_val is not None and not np.isnan(psar_short_val)

        if is_long_signal:
            # SAR dot is below the price, indicating potential uptrend.
            return 1.0 # Bullish signal
        elif is_short_signal:
            # SAR dot is above the price, indicating potential downtrend.
            return -1.0 # Bearish signal
        else:
            # Neither PSAR value is currently valid (e.g., at the very start of data or calculation issue)
            return 0.0

    def _check_sma10(self, current_price: Decimal) -> float:
        """Checks if the current price is above or below the 10-period Simple Moving Average."""
        sma10 = self.indicator_values.get("SMA10")
        if sma10 is None or current_price is None: return 0.0
        try:
            price_f = float(current_price)
            # Provides a weaker trend/momentum signal compared to EMA crossover.
            if price_f > sma10: return 0.5 # Price above short-term MA -> Weak bullish
            if price_f < sma10: return -0.5 # Price below short-term MA -> Weak bearish
            return 0.0 # Price is exactly on SMA10
        except (ValueError, TypeError):
             self.logger.warning(f"Could not convert current price {current_price} to float for SMA10 check.")
             return 0.0

    def _check_mfi(self) -> float:
        """Checks the Money Flow Index (MFI) for overbought/oversold conditions (volume-weighted RSI)."""
        mfi = self.indicator_values.get("MFI")
        if mfi is None: return 0.0
        # Standard levels: Below 20 is oversold, above 80 is overbought.
        if mfi < 20: return 1.0 # Oversold -> Bullish signal
        if mfi > 80: return -1.0 # Overbought -> Bearish signal
        if mfi < 40: return 0.4 # Leaning oversold
        if mfi > 60: return -0.4 # Leaning overbought
        return 0.0 # Neutral range

    def _check_orderbook_imbalance(self, orderbook: Optional[Dict]) -> float:
        """
        Calculates a simple order book imbalance score based on the volume
        within the top N bid and ask levels.
        Returns a score between -1.0 (strong sell pressure) and 1.0 (strong buy pressure).
        """
        if orderbook is None:
            self.logger.debug("Orderbook data not available for imbalance check.")
            return 0.0
        if not isinstance(orderbook.get('bids'), list) or not isinstance(orderbook.get('asks'), list):
             self.logger.warning("Orderbook format invalid (bids/asks not lists). Cannot calculate imbalance.")
             return 0.0
        if not orderbook['bids'] or not orderbook['asks']:
            self.logger.debug("Orderbook bids or asks list is empty. Cannot calculate imbalance.")
            return 0.0 # Cannot calculate if one side is empty

        try:
            # Consider the volume within the top N levels (e.g., 10)
            depth = 10 # Number of price levels to sum volume from
            # Sum the volume (amount) from the top 'depth' bid levels
            total_bid_volume = sum(Decimal(str(bid[1])) for bid in orderbook['bids'][:depth] if len(bid) > 1)
            # Sum the volume (amount) from the top 'depth' ask levels
            total_ask_volume = sum(Decimal(str(ask[1])) for ask in orderbook['asks'][:depth] if len(ask) > 1)

            total_volume = total_bid_volume + total_ask_volume
            if total_volume == 0:
                self.logger.debug("Total volume in top orderbook levels is zero.")
                return 0.0

            # Calculate Imbalance Ratio: (Buy Volume - Sell Volume) / Total Volume
            imbalance_ratio = (total_bid_volume - total_ask_volume) / total_volume

            # Scale the score: The ratio is already between -1 and 1.
            # We can optionally amplify it slightly if desired, but keeping it direct is fine.
            score = float(imbalance_ratio)
            # self.logger.debug(f"Orderbook imbalance: BidsVol={total_bid_volume:.4f}, AsksVol={total_ask_volume:.4f}, Ratio={imbalance_ratio:.3f}, Score={score:.2f}")
            return score

        except (TypeError, ValueError, InvalidOperation, IndexError) as e:
            self.logger.warning(f"Could not calculate orderbook imbalance for {self.symbol}: {e}")
            return 0.0
        except Exception as e:
             self.logger.error(f"Unexpected error during orderbook imbalance calculation for {self.symbol}: {e}", exc_info=True)
             return 0.0


    def generate_trading_signal(self, current_price: Decimal, orderbook: Optional[Dict] = None) -> str:
        """
        Generates a final trading signal ("BUY", "SELL", or "HOLD") based on the
        weighted sum of individual indicator scores. Uses the 'active_weight_set'
        defined in the configuration.

        Args:
            current_price: The current market price (as Decimal) for price-relative checks.
            orderbook: The fetched orderbook data (dictionary) if the orderbook indicator is enabled.

        Returns:
            The final trading signal as a string: "BUY", "SELL", or "HOLD".
        """
        if not self.weights:
            self.logger.error(f"Cannot generate signal for {self.symbol}: No weights loaded for active set '{self.active_weight_set_name}'. Defaulting to HOLD.")
            return "HOLD"
        if current_price is None or current_price.is_nan() or current_price <= 0:
             self.logger.error(f"Cannot generate signal for {self.symbol}: Invalid current price ({current_price}). Defaulting to HOLD.")
             return "HOLD"

        total_score = Decimal("0.0")
        total_weight_used = Decimal("0.0") # Sum of absolute weights for *active* and *weighted* indicators
        active_indicators_config = self.config.get("indicators", {})
        scores_log: Dict[str, str] = {} # Dictionary to store log messages for each indicator's score

        # --- Define Indicator Check Functions ---
        # This dictionary maps internal indicator names to their corresponding check functions.
        indicator_checks = {
            "ema_alignment": self._check_ema_alignment,
            "momentum": self._check_momentum,
            "volume_confirmation": self._check_volume_confirmation, # Note: Provides strength, not direction
            "stoch_rsi": self._check_stoch_rsi,
            "rsi": self._check_rsi,
            "bollinger_bands": lambda: self._check_bollinger_bands(current_price), # Use lambda for functions needing price
            "vwap": lambda: self._check_vwap(current_price),
            "cci": self._check_cci,
            "wr": self._check_wr,
            "psar": lambda: self._check_psar(current_price),
            "sma_10": lambda: self._check_sma10(current_price), # Ensure key matches config/check function
            "mfi": self._check_mfi,
            "orderbook": lambda: self._check_orderbook_imbalance(orderbook) # Use lambda for orderbook check
        }

        # --- Calculate Weighted Score ---
        for name, check_func in indicator_checks.items():
            is_enabled = active_indicators_config.get(name, False) # Check if indicator is enabled in config
            weight_val = self.weights.get(name, 0.0) # Get weight from the active weight set

            # Handle potential None or invalid weight values gracefully
            try:
                 weight = Decimal(str(weight_val)) if weight_val is not None else Decimal(0)
            except (InvalidOperation, ValueError, TypeError):
                 self.logger.warning(f"Invalid weight format '{weight_val}' for indicator '{name}' in weight set '{self.active_weight_set_name}'. Using weight 0.")
                 weight = Decimal(0)

            if is_enabled and weight != 0: # Only calculate if indicator is enabled AND has a non-zero weight
                try:
                    score_float = check_func() # Call the check function to get the raw float score (-1 to 1)

                    # Validate the returned score
                    if score_float is None or not isinstance(score_float, (float, int)):
                         self.logger.warning(f"Indicator check '{name}' returned invalid score type {type(score_float)} or None. Using score 0.")
                         score = Decimal(0)
                    else:
                         score = Decimal(str(score_float)) # Convert valid float score to Decimal

                    # Apply weight (handle volume confirmation separately if needed)
                    # For volume confirmation, its score (0 or 1) should perhaps multiply the scores of momentum indicators?
                    # Current simple approach: Add its weighted score like others.
                    weighted_score = score * weight
                    total_score += weighted_score
                    total_weight_used += abs(weight) # Sum absolute weight for potential normalization
                    scores_log[name] = f"{score:.2f} (w={weight}, ws={weighted_score:.2f})" # Log score, weight, weighted score

                except Exception as e:
                     self.logger.warning(f"Error calculating score for indicator '{name}' on {self.symbol}: {e}")
                     scores_log[name] = "Error"
            elif is_enabled and weight == 0:
                 scores_log[name] = "Enabled, Not Weighted"
            else: # Indicator disabled
                scores_log[name] = "Disabled"


        # --- Determine Signal based on Threshold ---
        # Select the appropriate threshold based on the active weight set name
        threshold_key = "signal_score_threshold" # Default threshold key
        if self.active_weight_set_name == "scalping":
             threshold_key = "scalping_signal_threshold"
        # Add more elif blocks here if you define other weight sets with specific thresholds

        try:
            # Get the threshold value from config, convert to Decimal
            threshold = Decimal(str(self.config.get(threshold_key, 1.5))) # Default threshold if key missing
        except (InvalidOperation, ValueError, TypeError):
             self.logger.warning(f"Invalid threshold format for '{threshold_key}' in config. Using default 1.5.")
             threshold = Decimal("1.5")

        # Normalize Score (Optional, but can make threshold more consistent across different weight sets)
        # normalized_score = total_score / total_weight_used if total_weight_used > 0 else Decimal("0.0")
        # Decision: Use raw total_score against threshold for now, as weights directly control influence.
        final_score_to_use = total_score

        signal = "HOLD"
        signal_color = NEON_YELLOW
        if final_score_to_use >= threshold:
            signal = "BUY"
            signal_color = NEON_GREEN
        elif final_score_to_use <= -threshold:
            signal = "SELL"
            signal_color = NEON_RED

        # Log the final decision process
        price_prec = self.get_price_precision()
        self.logger.info(
            f"Signal Calc ({self.symbol}): Price={current_price:.{price_prec}f}, "
            f"Score={final_score_to_use:.3f} (TotalWeight={total_weight_used:.2f}), "
            f"Threshold=+/={threshold:.2f}, Signal={signal_color}{signal}{RESET}"
        )
        self.logger.debug(f"Indicator Scores: {scores_log}")

        # Update internal simple signal state (optional, mainly for potential internal logic)
        self.signals = {"BUY": 0, "SELL": 0, "HOLD": 0}
        self.signals[signal] = 1

        return signal

    # --- Risk Management Calculations ---
    def calculate_entry_tp_sl(
        self, entry_price: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """
        Calculates potential Take Profit (TP) and initial Stop Loss (SL) levels
        based on the provided entry price, the latest ATR value, and configured multipliers.
        Uses Decimal precision and quantizes results to the market's tick size.

        The initial SL calculated here is primarily used for:
        1. Position sizing calculation before entry.
        2. Setting the initial protective stop loss order after entry (if not using TSL immediately).

        Args:
            entry_price: The estimated or actual entry price (as Decimal).
            signal: The trading signal ("BUY" or "SELL").

        Returns:
            A tuple containing (entry_price, take_profit, stop_loss),
            all as Decimals or None if calculation fails or is not applicable.
        """
        if signal not in ["BUY", "SELL"] or entry_price is None or entry_price.is_nan() or entry_price <= 0:
            # No TP/SL needed for HOLD signal or if entry price is invalid
            if entry_price is None or entry_price.is_nan() or entry_price <= 0:
                 self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: Invalid entry price ({entry_price}).{RESET}")
            return entry_price, None, None

        # Get the latest ATR value (float) from calculated indicators
        atr_val_float = self.indicator_values.get("ATR")
        if atr_val_float is None or pd.isna(atr_val_float) or atr_val_float <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: ATR value is invalid ({atr_val_float}). Check indicator calculation.{RESET}")
            return entry_price, None, None

        try:
            # Convert valid float ATR to Decimal for precise calculations
            atr = Decimal(str(atr_val_float))

            # Get multipliers from config, convert to Decimal
            tp_multiple = Decimal(str(self.config.get("take_profit_multiple", 1.0)))
            sl_multiple = Decimal(str(self.config.get("stop_loss_multiple", 1.5)))

            # Get market precision info for logging and quantization
            price_precision = self.get_price_precision()
            min_tick = self.get_min_tick_size()
            if min_tick is None or min_tick <= 0:
                 self.logger.error(f"{NEON_RED}Cannot calculate TP/SL for {self.symbol}: Invalid min tick size ({min_tick}) derived from market info.{RESET}")
                 return entry_price, None, None

            # Calculate raw TP/SL offsets based on ATR
            tp_offset = atr * tp_multiple
            sl_offset = atr * sl_multiple

            take_profit_raw: Optional[Decimal] = None
            stop_loss_raw: Optional[Decimal] = None
            take_profit: Optional[Decimal] = None
            stop_loss: Optional[Decimal] = None

            # Calculate raw levels based on signal direction
            if signal == "BUY":
                take_profit_raw = entry_price + tp_offset
                stop_loss_raw = entry_price - sl_offset
                # Quantize TP UP (away from entry), SL DOWN (away from entry) to the nearest valid tick
                take_profit = (take_profit_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick
                stop_loss = (stop_loss_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick

            elif signal == "SELL":
                take_profit_raw = entry_price - tp_offset
                stop_loss_raw = entry_price + sl_offset
                # Quantize TP DOWN (away from entry), SL UP (away from entry) to the nearest valid tick
                take_profit = (take_profit_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                stop_loss = (stop_loss_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick


            # --- Validation and Adjustments ---
            # 1. Ensure SL is actually beyond entry by at least one tick (quantization might place it too close)
            if stop_loss is not None:
                if signal == "BUY" and stop_loss >= entry_price:
                    adjusted_sl = entry_price - min_tick
                    self.logger.warning(f"{NEON_YELLOW}BUY signal SL calculation ({stop_loss:.{price_precision}f}) is too close to/above entry ({entry_price:.{price_precision}f}). Adjusting SL down by one tick to {adjusted_sl:.{price_precision}f}.{RESET}")
                    stop_loss = adjusted_sl
                elif signal == "SELL" and stop_loss <= entry_price:
                    adjusted_sl = entry_price + min_tick
                    self.logger.warning(f"{NEON_YELLOW}SELL signal SL calculation ({stop_loss:.{price_precision}f}) is too close to/below entry ({entry_price:.{price_precision}f}). Adjusting SL up by one tick to {adjusted_sl:.{price_precision}f}.{RESET}")
                    stop_loss = adjusted_sl

            # 2. Ensure TP provides potential profit relative to entry (at least one tick away)
            if take_profit is not None:
                if signal == "BUY" and take_profit <= entry_price:
                    adjusted_tp = entry_price + min_tick
                    self.logger.warning(f"{NEON_YELLOW}BUY signal TP calculation ({take_profit:.{price_precision}f}) resulted in non-profitable level vs entry ({entry_price:.{price_precision}f}). Adjusting TP up by one tick to {adjusted_tp:.{price_precision}f}.{RESET}")
                    take_profit = adjusted_tp
                elif signal == "SELL" and take_profit >= entry_price:
                    adjusted_tp = entry_price - min_tick
                    self.logger.warning(f"{NEON_YELLOW}SELL signal TP calculation ({take_profit:.{price_precision}f}) resulted in non-profitable level vs entry ({entry_price:.{price_precision}f}). Adjusting TP down by one tick to {adjusted_tp:.{price_precision}f}.{RESET}")
                    take_profit = adjusted_tp

            # 3. Final checks: Ensure SL/TP are still valid positive prices after adjustments
            if stop_loss is not None and stop_loss <= 0:
                self.logger.error(f"{NEON_RED}Stop loss calculation resulted in zero or negative price ({stop_loss:.{price_precision}f}) for {self.symbol} {signal}. Cannot set SL.{RESET}")
                stop_loss = None
            if take_profit is not None and take_profit <= 0:
                self.logger.error(f"{NEON_RED}Take profit calculation resulted in zero or negative price ({take_profit:.{price_precision}f}) for {self.symbol} {signal}. Cannot set TP.{RESET}")
                take_profit = None

            # Format values for logging
            tp_log = f"{take_profit:.{price_precision}f}" if take_profit else 'N/A'
            sl_log = f"{stop_loss:.{price_precision}f}" if stop_loss else 'N/A'
            atr_log = f"{atr:.{price_precision+1}f}" # Log ATR with slightly higher precision
            entry_log = f"{entry_price:.{price_precision}f}"

            self.logger.debug(f"Calculated TP/SL for {self.symbol} {signal}: Entry={entry_log}, TP={tp_log}, SL={sl_log} (based on ATR={atr_log})")
            return entry_price, take_profit, stop_loss

        except (InvalidOperation, ValueError, TypeError) as e:
            self.logger.error(f"{NEON_RED}Error during TP/SL calculation math for {self.symbol}: {e}{RESET}", exc_info=True)
            return entry_price, None, None
        except Exception as e:
             self.logger.error(f"{NEON_RED}Unexpected error calculating TP/SL for {self.symbol}: {e}{RESET}", exc_info=True)
             return entry_price, None, None


# --- Trading Logic Helper Functions ---
def fetch_balance(exchange: ccxt.Exchange, currency_code: str, logger: logging.Logger, account_type: str = 'CONTRACT') -> Optional[Decimal]:
    """
    Fetches the available balance for a specific currency code using CCXT,
    specifying the Bybit V5 account type.

    Args:
        exchange: The initialized CCXT exchange object.
        currency_code: The currency code (e.g., "USDT", "BTC").
        logger: The logger instance.
        account_type: The Bybit V5 account type ('CONTRACT', 'SPOT', 'UNIFIED').

    Returns:
        The available balance as a Decimal, or None if fetching fails or balance is zero/not found.
    """
    lg = logger
    try:
        lg.debug(f"Fetching balance for {currency_code} (Account Type: {account_type})...")
        params = {}
        # Bybit V5 fetch_balance requires accountType and optionally coin
        if 'bybit' in exchange.id.lower():
             params['accountType'] = account_type
             # Specifying the coin filters the response to just that currency
             params['coin'] = currency_code
             lg.debug(f"Using params for fetch_balance: {params}")
        else:
             lg.warning(f"Attempting balance fetch for non-Bybit exchange ({exchange.id}) without specific account type params.")

        # Retry mechanism for balance fetching (less critical than trading, but good practice)
        balance_info = None
        for attempt in range(MAX_API_RETRIES + 1):
             try:
                 balance_info = exchange.fetch_balance(params=params)
                 break # Success
             except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                 lg.warning(f"Network error fetching balance (Attempt {attempt+1}): {e}")
                 if attempt < MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS)
                 else: raise e # Re-raise after max retries
             except ccxt.RateLimitExceeded as e:
                  retry_after_ms = getattr(e, 'retryAfter', None)
                  wait_time = (retry_after_ms / 1000.0 + 0.2) if retry_after_ms else (exchange.rateLimit / 1000.0 + 1.0)
                  lg.warning(f"Rate limit fetching balance. Waiting {wait_time:.2f}s...")
                  time.sleep(wait_time)
                  # Don't increment standard attempt counter for rate limit
             except ccxt.ExchangeError as e:
                  # Handle specific errors like account type mismatch if possible
                  if "account type mismatch" in str(e).lower() or getattr(e,'code',None) == 170001:
                       lg.warning(f"Exchange error likely due to incorrect account type '{account_type}' for balance fetch: {e}")
                  else:
                       lg.error(f"Exchange error fetching balance: {e}")
                  raise e # Re-raise exchange errors

        if not balance_info:
             lg.error("Failed to fetch balance info after retries.")
             return None

        # lg.debug(f"Full balance info response: {json.dumps(balance_info, indent=2)}") # Very verbose, enable if needed

        # --- Parse Balance ---
        # Structure varies between exchanges and account types. Need robust parsing.
        # Bybit V5 structure for fetch_balance with coin specified:
        # Usually under info -> result -> list -> [0] -> coin -> [0]
        balance_decimal: Optional[Decimal] = None

        try:
            # Try parsing Bybit V5 structure first (most specific)
            if 'bybit' in exchange.id.lower() and 'info' in balance_info:
                result_list = balance_info.get('info', {}).get('result', {}).get('list', [])
                if result_list and isinstance(result_list, list) and len(result_list) > 0:
                    # The list might contain different account types, filter by requested type
                    account_data = None
                    for item in result_list:
                        if isinstance(item, dict) and item.get('accountType') == account_type:
                            account_data = item
                            break

                    if account_data and isinstance(account_data.get('coin'), list):
                        for coin_data in account_data['coin']:
                            if isinstance(coin_data, dict) and coin_data.get('coin') == currency_code:
                                # Prioritize 'availableToWithdraw' or 'availableToBorrow' as it excludes unrealized PnL.
                                # Fallback to 'walletBalance' (total equity including PnL). Use with caution for sizing.
                                balance_str = coin_data.get('availableToWithdraw', coin_data.get('availableToBorrow'))
                                if balance_str is None:
                                     balance_str = coin_data.get('walletBalance')

                                if balance_str is not None:
                                    balance_decimal = Decimal(str(balance_str))
                                    lg.debug(f"Parsed balance from info.result.list: {balance_decimal:.8f}")
                                    break # Found the coin in the correct account type

            # Fallback: Try standard CCXT balance structure ('free', 'total')
            if balance_decimal is None:
                currency_balance_dict = balance_info.get(currency_code)
                if currency_balance_dict and isinstance(currency_balance_dict, dict):
                    # Prefer 'free' (available balance)
                    available_balance_str = currency_balance_dict.get('free')
                    if available_balance_str is None:
                        # Fallback to 'total' if 'free' is missing (less ideal for sizing)
                        available_balance_str = currency_balance_dict.get('total')

                    if available_balance_str is not None:
                        balance_decimal = Decimal(str(available_balance_str))
                        lg.debug(f"Parsed balance from top-level CCXT structure ('free' or 'total'): {balance_decimal:.8f}")

            # --- Final Check ---
            if balance_decimal is not None and balance_decimal >= 0: # Allow zero balance
                lg.info(f"Available balance for {currency_code} ({account_type}): {balance_decimal:.4f}")
                return balance_decimal
            else:
                lg.warning(f"{NEON_YELLOW}Balance for currency {currency_code} ({account_type}) not found or invalid in response.{RESET}")
                lg.debug(f"Balance response keys: {balance_info.keys()}")
                # Log relevant part of response if parsing failed
                if 'bybit' in exchange.id.lower() and 'info' in balance_info:
                    lg.debug(f"Relevant info.result.list: {balance_info.get('info', {}).get('result', {}).get('list', [])}")
                elif currency_code in balance_info:
                     lg.debug(f"Relevant balance dict: {balance_info.get(currency_code)}")

                return None # Return None if not found or invalid

        except (InvalidOperation, ValueError, TypeError, KeyError) as e:
            lg.error(f"{NEON_RED}Could not parse balance for {currency_code} from response: {e}{RESET}")
            lg.debug(f"Problematic balance response snippet: {balance_info.get('info', {}).get('result', {}).get('list', 'N/A')}")
            return None

    except ccxt.AuthenticationError as e:
        lg.error(f"{NEON_RED}Authentication error fetching balance: {e}{RESET}")
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}Exchange error fetching balance: {e}{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error fetching balance: {e}{RESET}", exc_info=True)

    return None


def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """
    Fetches and prepares market information for a specific symbol using CCXT.
    Reloads markets if necessary and derives standardized precision/limit values.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The trading symbol (e.g., "BTC/USDT:USDT").
        logger: Logger instance.

    Returns:
        A dictionary containing the market information, including derived keys like
        'min_tick_size', 'price_precision_digits', 'amount_precision_digits',
        'min_order_amount', 'min_order_cost', and 'is_contract'. Returns None if
        the symbol is invalid or info cannot be fetched/processed.
    """
    lg = logger
    try:
        # --- Reload Markets Periodically or if Missing ---
        # Markets contain crucial precision and limit data. Reload if stale or symbol missing.
        reload_interval_seconds = 3600 # Reload markets every hour
        needs_reload = False
        if not hasattr(exchange, 'last_load_markets_timestamp'):
             needs_reload = True
             lg.info("Initial market load timestamp missing.")
        elif time.time() - exchange.last_load_markets_timestamp > reload_interval_seconds:
             needs_reload = True
             lg.info(f"Market data is older than {reload_interval_seconds}s.")
        if symbol not in exchange.markets:
             needs_reload = True
             lg.info(f"Symbol {symbol} not found in currently loaded markets.")

        if needs_reload:
             lg.info("Reloading markets...")
             try:
                 # Try reloading specific categories first for efficiency
                 exchange.load_markets(reload=True, params={'category': 'linear'})
                 exchange.load_markets(reload=True, params={'category': 'spot'})
                 # Add other categories if needed
                 exchange.last_load_markets_timestamp = time.time()
                 lg.info("Markets reloaded by category.")
             except Exception as reload_err_cat:
                  lg.warning(f"Market reload by category failed: {reload_err_cat}. Attempting full reload.")
                  try:
                       exchange.load_markets(reload=True)
                       exchange.last_load_markets_timestamp = time.time()
                       lg.info("Markets fully reloaded.")
                  except Exception as reload_err_full:
                       lg.error(f"{NEON_RED}Failed to reload markets: {reload_err_full}{RESET}")
                       # Proceed cautiously, existing market data might be stale or incomplete
                       # If symbol is still missing, we will fail below.

        # --- Check if Symbol Exists After Potential Reload ---
        if symbol not in exchange.markets:
            lg.error(f"{NEON_RED}Symbol '{symbol}' not found on exchange {exchange.id} even after market reload attempt. Please check the symbol format and availability.{RESET}")
            # Log available market symbols for debugging (can be very long)
            # lg.debug(f"Available market symbols: {list(exchange.markets.keys())}")
            return None

        # --- Get Market Data and Derive Values ---
        market = exchange.markets[symbol]
        # lg.debug(f"Raw market info retrieved for {symbol}: {market}") # Too verbose for regular use

        # Create a copy to avoid modifying the original CCXT market dict
        market_info_derived = market.copy()

        # Derive and add standardized info for easier access later
        market_info_derived['is_contract'] = market.get('contract', False) or market.get('type') in ['swap', 'future']
        market_info_derived['min_tick_size'] = get_min_tick_size_from_market(market, lg) # Decimal or None
        market_info_derived['price_precision_digits'] = get_price_precision_digits_from_market(market, lg) # int or None
        market_info_derived['amount_precision_digits'] = get_amount_precision_digits_from_market(market, lg) # int or None
        market_info_derived['min_order_amount'] = get_min_order_amount_from_market(market, lg) # Decimal or None
        market_info_derived['min_order_cost'] = get_min_order_cost_from_market(market, lg) # Decimal or None

        # Log key derived values for confirmation
        tick_log = f"{market_info_derived['min_tick_size']}" if market_info_derived['min_tick_size'] else 'N/A'
        price_prec_log = market_info_derived['price_precision_digits'] if market_info_derived['price_precision_digits'] is not None else 'N/A'
        amt_prec_log = market_info_derived['amount_precision_digits'] if market_info_derived['amount_precision_digits'] is not None else 'N/A'
        min_amt_log = f"{market_info_derived['min_order_amount']}" if market_info_derived['min_order_amount'] else 'N/A'
        min_cost_log = f"{market_info_derived['min_order_cost']}" if market_info_derived['min_order_cost'] else 'N/A'
        lg.debug(f"Derived info for {symbol}: TickSize={tick_log}, PricePrecDigits={price_prec_log}, AmtPrecDigits={amt_prec_log}, MinAmt={min_amt_log}, MinCost={min_cost_log}")

        # --- Validate Critical Derived Values ---
        # Ensure essential values needed for calculations and trading are present.
        critical_missing = []
        if market_info_derived['min_tick_size'] is None: critical_missing.append("min_tick_size")
        if market_info_derived['price_precision_digits'] is None: critical_missing.append("price_precision_digits")
        if market_info_derived['amount_precision_digits'] is None: critical_missing.append("amount_precision_digits")
        if market_info_derived['min_order_amount'] is None: critical_missing.append("min_order_amount")
        # min_order_cost can sometimes be None, especially for derivatives, so it's less critical for initial validation.

        if critical_missing:
             lg.error(f"{NEON_RED}Market info for {symbol} is missing critical derived data: {', '.join(critical_missing)}. Cannot safely use this symbol.{RESET}")
             return None

        return market_info_derived # Return the dictionary with added derived values

    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error fetching/processing market info for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}Exchange error fetching/processing market info for {symbol}: {e}{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error fetching/processing market info for {symbol}: {e}{RESET}", exc_info=True)
    return None

# --- Market Info Helper Functions (Called by get_market_info) ---
# These extract specific pieces of data from the raw CCXT market structure.

def get_min_tick_size_from_market(market: Dict, logger: logging.Logger) -> Optional[Decimal]:
    """Extracts minimum price increment (tick size) from CCXT market structure as Decimal."""
    lg = logger
    symbol = market.get('symbol', 'N/A')
    try:
        # 1. Check CCXT standard: market['precision']['price'] (can be tick size or digits)
        precision_info = market.get('precision', {})
        tick_size_raw = precision_info.get('price')

        if tick_size_raw is not None:
            try:
                tick_size_dec = Decimal(str(tick_size_raw))
                # If it's < 1, it's likely the tick size directly.
                if tick_size_dec < 1 and tick_size_dec > 0:
                    lg.debug(f"Tick size for {symbol} from precision.price: {tick_size_dec}")
                    return tick_size_dec
                # If it's >= 1, it might be the number of digits (handled by precision digits func).
                # If it's 0, precision might be 1 (e.g., price 1, 2, 3). Tick size is 1.
                elif tick_size_dec == 0:
                     lg.debug(f"Tick size for {symbol} inferred as 1 from precision.price=0")
                     return Decimal('1')
                # else: Fall through to other methods if ambiguous
            except (InvalidOperation, ValueError, TypeError):
                lg.debug(f"Could not parse precision.price ({tick_size_raw}) as Decimal for {symbol}. Trying other methods.")


        # 2. Check Bybit V5 specific: market['info']['priceFilter']['tickSize']
        info = market.get('info', {})
        tick_size_v5 = info.get('priceFilter', {}).get('tickSize')
        if tick_size_v5 is not None:
            try:
                tick_size_dec = Decimal(str(tick_size_v5))
                if tick_size_dec > 0:
                    lg.debug(f"Tick size for {symbol} from info.priceFilter.tickSize: {tick_size_dec}")
                    return tick_size_dec
            except (InvalidOperation, ValueError, TypeError):
                 lg.warning(f"Could not parse info.priceFilter.tickSize ({tick_size_v5}) for {symbol}.")

        # 3. Check Bybit older/other specific: market['info']['tickSize'] (less common now)
        tick_size_v3 = info.get('tickSize')
        if tick_size_v3 is not None:
             try:
                tick_size_dec = Decimal(str(tick_size_v3))
                if tick_size_dec > 0:
                    lg.debug(f"Tick size for {symbol} from info.tickSize: {tick_size_dec}")
                    return tick_size_dec
             except (InvalidOperation, ValueError, TypeError):
                 lg.warning(f"Could not parse info.tickSize ({tick_size_v3}) for {symbol}.")


        # 4. Fallback: Infer from price precision digits if tick size not found directly
        lg.debug(f"Could not find direct tick size for {symbol}. Inferring from precision digits...")
        price_digits = get_price_precision_digits_from_market(market, logger) # Call the digits helper
        if price_digits is not None and price_digits >= 0:
            inferred_tick_size = Decimal('1') / (Decimal('10') ** price_digits)
            lg.debug(f"Inferred tick size for {symbol} from {price_digits} digits: {inferred_tick_size}")
            return inferred_tick_size

        # 5. Final Failure
        lg.error(f"{NEON_RED}Could not determine minimum tick size for {symbol} from market data.{RESET}")
        lg.debug(f"Market precision info: {precision_info}")
        lg.debug(f"Market info dict: {info}")
        return None

    except Exception as e:
        lg.error(f"Unexpected error determining min tick size for {symbol}: {e}", exc_info=True)
        return None

def get_price_precision_digits_from_market(market: Dict, logger: logging.Logger) -> Optional[int]:
    """Extracts number of decimal places for price from CCXT market structure."""
    lg = logger
    symbol = market.get('symbol', 'N/A')
    try:
        # 1. Check CCXT standard: market['precision']['price']
        #    If it's an integer >= 1, it's likely the number of digits.
        precision_info = market.get('precision', {})
        price_prec_raw = precision_info.get('price')
        if price_prec_raw is not None:
            try:
                price_prec_dec = Decimal(str(price_prec_raw))
                # Heuristic: If it's >= 1 and an integer, assume it's digits.
                if price_prec_dec >= 1 and price_prec_dec.to_integral_value() == price_prec_dec:
                    digits = int(price_prec_dec)
                    lg.debug(f"Price precision digits for {symbol} from precision.price (as integer): {digits}")
                    return digits
                elif price_prec_dec == 0: # Precision 0 means integer price (0 digits)
                    lg.debug(f"Price precision digits for {symbol} from precision.price (as 0): 0")
                    return 0
                # If < 1 or non-integer, it's likely the tick size, handled below.
            except (InvalidOperation, ValueError, TypeError):
                 lg.debug(f"Could not parse precision.price ({price_prec_raw}) as digits for {symbol}. Inferring...")

        # 2. Infer from Minimum Tick Size (most reliable if direct digits aren't available)
        min_tick = get_min_tick_size_from_market(market, logger) # Use the helper
        if min_tick is not None and min_tick > 0:
            # Number of digits = negative exponent of the normalized tick size
            # E.g., tick=0.01 -> exponent=-2 -> digits=2
            # E.g., tick=1 -> exponent=0 -> digits=0
            # E.g., tick=0.00000005 -> exponent=-8 -> digits=8
            digits = abs(min_tick.normalize().as_tuple().exponent)
            lg.debug(f"Inferred price precision digits for {symbol} from tick size {min_tick}: {digits}")
            return digits

        # 3. Check Bybit V5 specific info (less common for precision digits)
        # Bybit usually defines tickSize, not explicit digits.

        # 4. Final Failure
        lg.error(f"{NEON_RED}Could not determine price precision digits for {symbol}.{RESET}")
        return None

    except Exception as e:
        lg.error(f"Unexpected error determining price precision digits for {symbol}: {e}", exc_info=True)
        return None

def get_amount_precision_digits_from_market(market: Dict, logger: logging.Logger) -> Optional[int]:
    """Extracts number of decimal places for amount/quantity from CCXT market structure."""
    lg = logger
    symbol = market.get('symbol', 'N/A')
    try:
        # 1. Check CCXT standard: market['precision']['amount']
        #    Can be digits (int >= 1) or step size (float < 1).
        precision_info = market.get('precision', {})
        amount_prec_raw = precision_info.get('amount')
        if amount_prec_raw is not None:
            try:
                amount_prec_dec = Decimal(str(amount_prec_raw))
                # Heuristic: Integer >= 1 -> digits. Float < 1 -> step size.
                if amount_prec_dec >= 1 and amount_prec_dec.to_integral_value() == amount_prec_dec:
                    digits = int(amount_prec_dec)
                    lg.debug(f"Amount precision digits for {symbol} from precision.amount (as integer): {digits}")
                    return digits
                elif amount_prec_dec < 1 and amount_prec_dec > 0: # Infer digits from step size
                    digits = abs(amount_prec_dec.normalize().as_tuple().exponent)
                    lg.debug(f"Inferred amount precision digits for {symbol} from precision.amount (as step size {amount_prec_dec}): {digits}")
                    return digits
                elif amount_prec_dec == 0: # Precision 0 means integer amount
                     lg.debug(f"Amount precision digits for {symbol} from precision.amount (as 0): 0")
                     return 0
            except (InvalidOperation, ValueError, TypeError):
                 lg.debug(f"Could not parse precision.amount ({amount_prec_raw}) for {symbol}. Trying other methods.")

        # 2. Check Bybit V5 specific: market['info']['lotSizeFilter']['qtyStep']
        info = market.get('info', {})
        step_size_v5 = info.get('lotSizeFilter', {}).get('qtyStep')
        if step_size_v5 is not None:
            try:
                 step_size = Decimal(str(step_size_v5))
                 if step_size > 0:
                     # Infer digits from step size
                     digits = abs(step_size.normalize().as_tuple().exponent)
                     lg.debug(f"Inferred amount precision digits for {symbol} from info.lotSizeFilter.qtyStep {step_size}: {digits}")
                     return digits
            except (InvalidOperation, ValueError, TypeError):
                lg.warning(f"Could not parse info.lotSizeFilter.qtyStep ({step_size_v5}) for {symbol}.")

        # 3. Check Bybit older specific: market['info']['basePrecision'] (often digits)
        base_precision_v3 = info.get('basePrecision')
        if base_precision_v3 is not None:
            try:
                 # Usually represents digits directly
                 digits = int(Decimal(str(base_precision_v3)))
                 lg.debug(f"Amount precision digits for {symbol} from info.basePrecision: {digits}")
                 return digits
            except (InvalidOperation, ValueError, TypeError):
                 lg.warning(f"Could not parse info.basePrecision ({base_precision_v3}) as digits for {symbol}.")


        # 4. Final Failure
        lg.error(f"{NEON_RED}Could not determine amount precision digits for {symbol}.{RESET}")
        return None

    except Exception as e:
        lg.error(f"Unexpected error determining amount precision digits for {symbol}: {e}", exc_info=True)
        return None

def get_min_order_amount_from_market(market: Dict, logger: logging.Logger) -> Optional[Decimal]:
    """Extracts minimum order amount (in base currency) from CCXT market structure as Decimal."""
    lg = logger
    symbol = market.get('symbol', 'N/A')
    try:
        # 1. Check CCXT standard: market['limits']['amount']['min']
        limits_info = market.get('limits', {})
        amount_info = limits_info.get('amount', {})
        min_amount_ccxt = amount_info.get('min')
        if min_amount_ccxt is not None:
            try:
                min_amount = Decimal(str(min_amount_ccxt))
                lg.debug(f"Min order amount for {symbol} from limits.amount.min: {min_amount}")
                # Return value even if zero, let sizing logic handle comparison
                return min_amount
            except (InvalidOperation, ValueError, TypeError):
                 lg.warning(f"Could not parse limits.amount.min ({min_amount_ccxt}) for {symbol}.")

        # 2. Check Bybit V5 specific: market['info']['lotSizeFilter']['minOrderQty']
        info = market.get('info', {})
        min_amount_v5 = info.get('lotSizeFilter', {}).get('minOrderQty')
        if min_amount_v5 is not None:
            try:
                min_amount = Decimal(str(min_amount_v5))
                lg.debug(f"Min order amount for {symbol} from info.lotSizeFilter.minOrderQty: {min_amount}")
                return min_amount
            except (InvalidOperation, ValueError, TypeError):
                lg.warning(f"Could not parse info.lotSizeFilter.minOrderQty ({min_amount_v5}) for {symbol}.")

        # 3. Check Bybit older/spot specific: market['info']['minTradeQuantity']
        min_amount_v3_spot = info.get('minTradeQuantity')
        if min_amount_v3_spot is not None:
            try:
                min_amount = Decimal(str(min_amount_v3_spot))
                lg.debug(f"Min order amount for {symbol} from info.minTradeQuantity: {min_amount}")
                return min_amount
            except (InvalidOperation, ValueError, TypeError):
                 lg.warning(f"Could not parse info.minTradeQuantity ({min_amount_v3_spot}) for {symbol}.")

        # 4. Final Failure - Return None, indicating it couldn't be found
        lg.error(f"{NEON_RED}Could not determine minimum order amount for {symbol}.{RESET}")
        return None

    except Exception as e:
        lg.error(f"Unexpected error determining minimum order amount for {symbol}: {e}", exc_info=True)
        return None

def get_min_order_cost_from_market(market: Dict, logger: logging.Logger) -> Optional[Decimal]:
    """
    Extracts minimum order cost (in quote currency) from CCXT market structure as Decimal.
    Returns None if not explicitly defined (common for derivatives).
    """
    lg = logger
    symbol = market.get('symbol', 'N/A')
    try:
        # 1. Check CCXT standard: market['limits']['cost']['min']
        limits_info = market.get('limits', {})
        cost_info = limits_info.get('cost', {})
        min_cost_ccxt = cost_info.get('min')
        if min_cost_ccxt is not None:
            try:
                min_cost = Decimal(str(min_cost_ccxt))
                lg.debug(f"Min order cost for {symbol} from limits.cost.min: {min_cost}")
                # Return value even if zero
                return min_cost
            except (InvalidOperation, ValueError, TypeError):
                 lg.warning(f"Could not parse limits.cost.min ({min_cost_ccxt}) for {symbol}.")

        # 2. Check Bybit V5 specific: market['info']['lotSizeFilter']['minOrderAmt'] (Note: Amt, not Qty)
        # This seems to represent min cost for SPOT markets in V5. Derivatives might not have it.
        info = market.get('info', {})
        min_cost_v5_spot = info.get('lotSizeFilter', {}).get('minOrderAmt')
        if min_cost_v5_spot is not None:
            try:
                min_cost = Decimal(str(min_cost_v5_spot))
                lg.debug(f"Min order cost for {symbol} from info.lotSizeFilter.minOrderAmt: {min_cost}")
                return min_cost
            except (InvalidOperation, ValueError, TypeError):
                lg.warning(f"Could not parse info.lotSizeFilter.minOrderAmt ({min_cost_v5_spot}) for {symbol}.")


        # 3. Check Bybit older/spot specific: market['info']['minTradeAmount'] or market['info']['minOrderAmount']
        min_cost_v3_spot = info.get('minTradeAmount', info.get('minOrderAmount')) # Check both names
        if min_cost_v3_spot is not None:
             try:
                min_cost = Decimal(str(min_cost_v3_spot))
                lg.debug(f"Min order cost for {symbol} from info.minTradeAmount/minOrderAmount: {min_cost}")
                return min_cost
             except (InvalidOperation, ValueError, TypeError):
                 lg.warning(f"Could not parse info.minTradeAmount/minOrderAmount ({min_cost_v3_spot}) for {symbol}.")


        # 4. Not Found - Return None
        # Derivatives often don't explicitly list min cost, relying on min amount * price.
        lg.debug(f"Minimum order cost not explicitly defined for {symbol}. Will rely on min amount checks.")
        return None

    except Exception as e:
        lg.error(f"Unexpected error determining minimum order cost for {symbol}: {e}", exc_info=True)
        return None


def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float,
    initial_stop_loss_price: Decimal,
    entry_price: Decimal,
    market_info: Dict,
    exchange: ccxt.Exchange, # Pass exchange for potential future use (e.g., contract value fetching)
    logger: logging.Logger
) -> Optional[Decimal]:
    """
    Calculates the position
