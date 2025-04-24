# Enhanced version focusing on stop-loss/take-profit mechanisms,
# including break-even logic, MA cross exit condition, correct Bybit V5 API usage,
# multi-symbol support, and improved state management.

import argparse
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
from colorama import Fore, Style, init
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

# Initialize colorama and set precision
getcontext().prec = 28  # Increased precision for financial calculations
init(autoreset=True)
load_dotenv()

# Neon Color Scheme
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
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env")

CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
# Timezone for logging and display (adjust as needed)
TIMEZONE = ZoneInfo("America/Chicago") # e.g., "America/New_York", "Europe/London", "Asia/Tokyo"
MAX_API_RETRIES = 3 # Max retries for recoverable API errors
RETRY_DELAY_SECONDS = 5 # Delay between retries
# Intervals supported by the bot's logic. Note: Only certain intervals might have enough liquidity for scalping/intraday.
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP = { # Map our intervals to ccxt's expected format
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}
RETRY_ERROR_CODES = [429, 500, 502, 503, 504] # HTTP status codes considered retryable

# Default periods (can be overridden by config.json)
DEFAULT_ATR_PERIOD = 14
DEFAULT_CCI_WINDOW = 20
DEFAULT_WILLIAMS_R_WINDOW = 14
DEFAULT_MFI_WINDOW = 14
DEFAULT_STOCH_RSI_WINDOW = 14 # Window for Stoch RSI calculation itself
DEFAULT_STOCH_WINDOW = 14     # Window for underlying RSI in StochRSI (often same as Stoch RSI window)
DEFAULT_K_WINDOW = 3          # K period for StochRSI
DEFAULT_D_WINDOW = 3          # D period for StochRSI
DEFAULT_RSI_WINDOW = 14
DEFAULT_BOLLINGER_BANDS_PERIOD = 20
DEFAULT_BOLLINGER_BANDS_STD_DEV = 2.0 # Ensure float
DEFAULT_SMA_10_WINDOW = 10
DEFAULT_EMA_SHORT_PERIOD = 9
DEFAULT_EMA_LONG_PERIOD = 21
DEFAULT_MOMENTUM_PERIOD = 7
DEFAULT_VOLUME_MA_PERIOD = 15
DEFAULT_FIB_WINDOW = 50
DEFAULT_PSAR_AF = 0.02
DEFAULT_PSAR_MAX_AF = 0.2

FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0] # Standard Fibonacci levels
LOOP_DELAY_SECONDS = 15 # Time between the end of one cycle and the start of the next loop for *all* symbols
POSITION_CONFIRM_DELAY = 10 # Seconds to wait after placing order before checking position status
# QUOTE_CURRENCY dynamically loaded from config

os.makedirs(LOG_DIRECTORY, exist_ok=True)


class SensitiveFormatter(logging.Formatter):
    """Formatter to redact sensitive information (API keys) from logs."""
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if API_KEY:
            msg = msg.replace(API_KEY, "***API_KEY***")
        if API_SECRET:
            msg = msg.replace(API_SECRET, "***API_SECRET***")
        return msg


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from JSON file, creating default if not found,
       and ensuring all default keys are present."""
    default_config = {
        "symbols": ["BTC/USDT:USDT"], # List of symbols to trade (CCXT format)
        "interval": "5", # Default to 5 minute interval (string format for our logic)
        "loop_delay": LOOP_DELAY_SECONDS, # Delay between full bot cycles (processing all symbols)
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
        "stoch_rsi_rsi_window": DEFAULT_STOCH_WINDOW,
        "stoch_rsi_k": DEFAULT_K_WINDOW,
        "stoch_rsi_d": DEFAULT_D_WINDOW,
        "psar_af": DEFAULT_PSAR_AF,
        "psar_max_af": DEFAULT_PSAR_MAX_AF,
        "sma_10_window": DEFAULT_SMA_10_WINDOW,
        "momentum_period": DEFAULT_MOMENTUM_PERIOD,
        "volume_ma_period": DEFAULT_VOLUME_MA_PERIOD,
        "orderbook_limit": 25, # Depth of orderbook to fetch
        "signal_score_threshold": 1.5, # Score needed to trigger BUY/SELL signal for 'default' weight set
        "scalping_signal_threshold": 2.5, # Separate threshold for 'scalping' weight set
        "stoch_rsi_oversold_threshold": 25,
        "stoch_rsi_overbought_threshold": 75,
        "stop_loss_multiple": 1.8, # ATR multiple for initial SL (used for sizing)
        "take_profit_multiple": 0.7, # ATR multiple for TP
        "volume_confirmation_multiplier": 1.5, # How much higher volume needs to be than MA
        "fibonacci_window": DEFAULT_FIB_WINDOW,
        "enable_trading": False, # SAFETY FIRST: Default to False, enable consciously
        "use_sandbox": True,     # SAFETY FIRST: Default to True (testnet), disable consciously
        "risk_per_trade": 0.01, # Risk 1% of account balance per trade
        "leverage": 20,          # Set desired leverage (check exchange limits)
        "max_concurrent_positions_total": 1, # Global limit on open positions across all symbols
        "quote_currency": "USDT", # Currency for balance check and sizing
        "position_mode": "One-Way", # "One-Way" or "Hedge" - CRITICAL: Match exchange setting. Bot assumes One-Way by default.
        # --- MA Cross Exit Config ---
        "enable_ma_cross_exit": True, # Enable closing position on adverse EMA cross
        # --- Trailing Stop Loss Config ---
        "enable_trailing_stop": True, # Default to enabling TSL (exchange TSL)
        # Trail distance as a percentage of the entry price (e.g., 0.5%)
        # Bybit API expects absolute distance; this percentage is used for calculation.
        "trailing_stop_callback_rate": 0.005, # Example: 0.5% trail distance relative to entry price
        # Activate TSL when price moves this percentage in profit from entry (e.g., 0.3%)
        # Set to 0 for immediate TSL activation upon entry (API value '0').
        "trailing_stop_activation_percentage": 0.003, # Example: Activate when 0.3% in profit
        # --- Break-Even Stop Config ---
        "enable_break_even": True,              # Enable moving SL to break-even
        # Move SL when profit (in price points) reaches X * Current ATR
        "break_even_trigger_atr_multiple": 1.0, # Example: Trigger BE when profit = 1x ATR
        # Place BE SL this many minimum price increments (ticks) beyond entry price
        # E.g., 2 ticks to cover potential commission/slippage on exit
        "break_even_offset_ticks": 2,
         # Set to True to always cancel TSL and revert to Fixed SL when BE is triggered.
         # Set to False to allow TSL to potentially remain active if it's better than BE price.
         # Note: Bybit V5 often automatically cancels TSL when a fixed SL is set, so True is generally safer.
        "break_even_force_fixed_sl": True,
        # --- End Protection Config ---
        "indicators": { # Control which indicators are calculated and contribute to score
            "ema_alignment": True, "momentum": True, "volume_confirmation": True,
            "stoch_rsi": True, "rsi": True, "bollinger_bands": True, "vwap": True,
            "cci": True, "wr": True, "psar": True, "sma_10": True, "mfi": True,
            "orderbook": True, # Flag to enable fetching and scoring orderbook data
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
        "active_weight_set": "default" # Choose which weight set to use ("default" or "scalping")
    }

    if not os.path.exists(filepath):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            print(f"{NEON_YELLOW}Created default config file: {filepath}{RESET}")
            return default_config
        except IOError as e:
            print(f"{NEON_RED}Error creating default config file {filepath}: {e}{RESET}")
            return default_config # Return default even if creation fails

    try:
        with open(filepath, encoding="utf-8") as f:
            config_from_file = json.load(f)
            # Ensure all default keys exist in the loaded config recursively
            updated_config = _ensure_config_keys(config_from_file, default_config)
            # Save back if keys were added during the update
            if updated_config != config_from_file:
                try:
                    with open(filepath, "w", encoding="utf-8") as f_write:
                        json.dump(updated_config, f_write, indent=4)
                    print(f"{NEON_YELLOW}Updated config file with missing default keys: {filepath}{RESET}")
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
    """Recursively ensures all keys from the default config are present in the loaded config."""
    updated_config = config.copy()
    for key, default_value in default_config.items():
        if key not in updated_config:
            updated_config[key] = default_value
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # Recursively check nested dictionaries
            updated_config[key] = _ensure_config_keys(updated_config[key], default_value)
        # Type mismatch handling could be added here if needed
    return updated_config

CONFIG = load_config(CONFIG_FILE)
QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT")
LOOP_DELAY_SECONDS = int(CONFIG.get("loop_delay", 15)) # Use configured loop delay, default 15
console_log_level = logging.INFO # Default console level, can be changed via args

# --- Logger Setup ---
# Global logger dictionary to manage loggers per symbol
loggers: Dict[str, logging.Logger] = {}

def setup_logger(name: str, is_symbol_logger: bool = False) -> logging.Logger:
    """Sets up a logger with file and console handlers.
       If is_symbol_logger is True, uses symbol-specific naming and file.
       Otherwise, uses a generic name (e.g., 'main')."""

    if name in loggers:
        # If logger already exists, update console level if needed
        logger = loggers[name]
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(console_log_level)
        return logger

    # Clean name for filename if it's a symbol
    safe_name = name.replace('/', '_').replace(':', '-') if is_symbol_logger else name
    logger_instance_name = f"livebot_{safe_name}" # Unique internal name for the logger
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_instance_name}.log")
    logger = logging.getLogger(logger_instance_name)

    # Set base logging level to DEBUG to capture everything
    logger.setLevel(logging.DEBUG)

    # File Handler (writes DEBUG and above, includes line numbers)
    try:
        # Ensure the log directory exists
        os.makedirs(LOG_DIRECTORY, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        # Add line number to file logs for easier debugging
        file_formatter = SensitiveFormatter("%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG) # Log everything to the file
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"{NEON_RED}Error setting up file logger for {log_filename}: {e}{RESET}")


    # Stream Handler (console)
    stream_handler = logging.StreamHandler()
    stream_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S' # Add timestamp format to console
    )
    stream_handler.setFormatter(stream_formatter)
    # Set console level based on global variable (which can be set by args)
    stream_handler.setLevel(console_log_level)
    logger.addHandler(stream_handler)

    # Prevent logs from propagating to the root logger (avoids duplicate outputs)
    logger.propagate = False

    loggers[name] = logger # Store logger instance
    return logger

def get_logger(name: str, is_symbol_logger: bool = False) -> logging.Logger:
    """Retrieves or creates a logger instance."""
    if name not in loggers:
        return setup_logger(name, is_symbol_logger)
    else:
        # Ensure existing logger's console level matches current setting
        logger = loggers[name]
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(console_log_level)
        return logger

# --- CCXT Exchange Setup ---
# (initialize_exchange function remains largely the same as whale2.0.py,
#  but ensure logger passed is the main logger, not symbol-specific initially)
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """Initializes the CCXT Bybit exchange object with error handling."""
    try:
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True, # Let ccxt handle basic rate limiting
            'options': {
                'defaultType': 'linear', # Assume linear contracts (USDT margined) - adjust if needed
                'adjustForTimeDifference': True, # Auto-sync time with server
                # Connection timeouts (milliseconds)
                'fetchTickerTimeout': 10000, # 10 seconds
                'fetchBalanceTimeout': 15000, # 15 seconds
                'createOrderTimeout': 20000, # Timeout for placing orders
                'fetchOrderTimeout': 15000,  # Timeout for fetching orders
                'fetchPositionsTimeout': 15000, # Timeout for fetching positions
                # Add any exchange-specific options if needed
                'recvWindow': 10000, # Bybit default is 5000, 10000 is safer
                'brokerId': 'livebot71', # Example: Add a broker ID for Bybit if desired
                # Explicitly request V5 API for relevant endpoints
                'default_options': {
                    'adjustForTimeDifference': True,
                    'warnOnFetchOpenOrdersWithoutSymbol': False, # Suppress warning
                    'recvWindow': 10000,
                    'fetchPositions': 'v5', # Request V5
                    'fetchBalance': 'v5',   # Request V5
                    'createOrder': 'v5',    # Request V5
                    'fetchOrder': 'v5',     # Request V5
                    'fetchTicker': 'v5',    # Request V5
                    'fetchOHLCV': 'v5',     # Request V5
                    'fetchOrderBook': 'v5', # Request V5
                    'setLeverage': 'v5',
                    'private_post_v5_position_trading_stop': 'v5', # Ensure protection uses V5
                },
                'accounts': { # Define V5 account types if needed by ccxt version/implementation
                    'future': {'linear': 'CONTRACT', 'inverse': 'CONTRACT'},
                    'swap': {'linear': 'CONTRACT', 'inverse': 'CONTRACT'},
                    'option': {'unified': 'OPTION'},
                    'spot': {'unified': 'SPOT'},
                    'unified': {'linear': 'UNIFIED', 'inverse': 'UNIFIED', 'spot': 'UNIFIED', 'option': 'UNIFIED'}, # Unified account support
                },
                'bybit': { # Bybit specific options
                     'defaultSettleCoin': QUOTE_CURRENCY, # Set default settlement coin (e.g., USDT)
                }
            }
        }

        # Select Bybit class
        exchange_class = getattr(ccxt, 'bybit') # Use getattr for robustness
        exchange = exchange_class(exchange_options)

        # Set sandbox mode if configured
        if CONFIG.get('use_sandbox'):
            logger.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet){RESET}")
            exchange.set_sandbox_mode(True)

        # Test connection by fetching markets (essential for market info)
        logger.info(f"Loading markets for {exchange.id}...")
        # Load markets explicitly, specifying category can speed it up for Bybit V5
        # Load linear and spot initially if those are the main types used
        try:
             logger.debug("Loading linear markets...")
             exchange.load_markets(params={'category': 'linear'})
             logger.debug("Loading spot markets...")
             exchange.load_markets(params={'category': 'spot'})
             # If inverse or options are used, load them too
             # exchange.load_markets(params={'category': 'inverse'})
             # exchange.load_markets(params={'category': 'option'})
        except Exception as market_load_err:
            logger.warning(f"Could not pre-load markets by category: {market_load_err}. Relying on default load.")
            exchange.load_markets() # Fallback to default load

        exchange.last_load_markets_timestamp = time.time() # Store timestamp for periodic reload check

        logger.info(f"CCXT exchange initialized ({exchange.id}). Sandbox: {CONFIG.get('use_sandbox')}")
        logger.info(f"CCXT version: {ccxt.__version__}") # Log CCXT version

        # Test API credentials and permissions by fetching balance
        # Specify account type for Bybit V5 (CONTRACT for linear/USDT, UNIFIED if using that)
        account_type_to_test = 'CONTRACT' # Start with common type
        logger.info(f"Attempting initial balance fetch (Default Account Type: {account_type_to_test})...")
        try:
            # Use our enhanced balance function
            balance_decimal = fetch_balance(exchange, QUOTE_CURRENCY, logger)
            if balance_decimal is not None:
                 logger.info(f"{NEON_GREEN}Successfully connected and fetched initial balance.{RESET} ({QUOTE_CURRENCY} available: {balance_decimal:.4f})")
            else:
                 logger.warning(f"{NEON_YELLOW}Initial balance fetch returned None. Check API permissions/account type if trading fails.{RESET}")
        except ccxt.AuthenticationError as auth_err:
            logger.error(f"{NEON_RED}CCXT Authentication Error during initial balance fetch: {auth_err}{RESET}")
            logger.error(f"{NEON_RED}>> Ensure API keys are correct, have necessary permissions (Read, Trade), match the account type (Real/Testnet), and IP whitelist is correctly set if enabled on Bybit.{RESET}")
            return None # Critical failure, cannot proceed
        except ccxt.ExchangeError as balance_err:
            # Handle potential errors if account type is wrong etc.
            bybit_code = getattr(balance_err, 'code', None)
            logger.warning(f"{NEON_YELLOW}Exchange error during initial balance fetch (tried {account_type_to_test}): {balance_err} (Code: {bybit_code}). Continuing, but check API permissions/account type if trading fails.{RESET}")
        except Exception as balance_err:
            logger.warning(f"{NEON_YELLOW}Could not perform initial balance fetch: {balance_err}. Continuing, but check API permissions/account type if trading fails.{RESET}")

        return exchange

    except ccxt.AuthenticationError as e:
        logger.error(f"{NEON_RED}CCXT Authentication Error during initialization: {e}{RESET}")
        logger.error(f"{NEON_RED}>> Check API keys, permissions, IP whitelist, and Real/Testnet selection.{RESET}")
    except ccxt.ExchangeError as e:
        logger.error(f"{NEON_RED}CCXT Exchange Error initializing: {e}{RESET}")
    except ccxt.NetworkError as e:
        logger.error(f"{NEON_RED}CCXT Network Error initializing: {e}{RESET}")
    except Exception as e:
        logger.error(f"{NEON_RED}Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True)

    return None

# --- CCXT Data Fetching Functions ---
# (fetch_current_price_ccxt, fetch_klines_ccxt, fetch_orderbook_ccxt remain mostly the same,
#  but ensure they accept and use market_info for params where needed)
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger, market_info: Dict = None) -> Optional[Decimal]:
    """Fetch the current price of a trading symbol using CCXT ticker with fallbacks."""
    lg = logger
    try:
        lg.debug(f"Fetching ticker for {symbol}...")
        params = {}
        market_id = symbol # Default
        if 'bybit' in exchange.id.lower() and market_info:
             market_id = market_info.get('id', symbol) # Use exchange ID if available
             category = None
             if market_info.get('is_contract'):
                 category = 'linear' if market_info.get('linear', True) else 'inverse'
             elif market_info.get('spot'):
                 category = 'spot'
             if category:
                 params['category'] = category
                 # Use market_id from market_info if available for Bybit V5
                 params['symbol'] = market_id
                 lg.debug(f"Using params for fetch_ticker ({symbol}): {params}")

        # Pass symbol and params
        ticker = exchange.fetch_ticker(symbol, params=params)
        # lg.debug(f"Ticker data for {symbol}: {ticker}") # Less verbose logging

        price = None
        # Prioritize bid/ask midpoint for more realistic current price
        bid_price = ticker.get('bid')
        ask_price = ticker.get('ask')
        last_price = ticker.get('last')

        # 1. Try bid/ask midpoint first
        if bid_price is not None and ask_price is not None:
            try:
                bid_decimal = Decimal(str(bid_price))
                ask_decimal = Decimal(str(ask_price))
                if bid_decimal > 0 and ask_decimal > 0 and bid_decimal <= ask_decimal:
                    price = (bid_decimal + ask_decimal) / 2
                    lg.debug(f"Using bid/ask midpoint for {symbol}: {price} (Bid: {bid_decimal}, Ask: {ask_decimal})")
                elif bid_decimal > ask_decimal:
                     lg.warning(f"Invalid ticker state: Bid ({bid_decimal}) > Ask ({ask_decimal}) for {symbol}. Using 'ask' as fallback.")
                     price = ask_decimal # Use ask as a safer fallback in this case
            except (InvalidOperation, ValueError, TypeError) as e:
                lg.warning(f"Could not parse bid/ask prices ({bid_price}, {ask_price}) for {symbol}: {e}")

        # 2. If midpoint fails, try 'last' price
        if price is None and last_price is not None:
            try:
                last_decimal = Decimal(str(last_price))
                if last_decimal > 0:
                    price = last_decimal
                    lg.debug(f"Using 'last' price as fallback for {symbol}: {price}")
            except (InvalidOperation, ValueError, TypeError) as e:
                lg.warning(f"Could not parse 'last' price ({last_price}) for {symbol}: {e}")

        # 3. If last fails, try 'ask' price
        if price is None and ask_price is not None:
            try:
                ask_decimal = Decimal(str(ask_price))
                if ask_decimal > 0:
                    price = ask_decimal
                    lg.warning(f"Using 'ask' price as final fallback for {symbol}: {price}")
            except (InvalidOperation, ValueError, TypeError) as e:
                lg.warning(f"Could not parse 'ask' price ({ask_price}) for {symbol}: {e}")

        # 4. If ask fails, try 'bid' price (less ideal, but better than nothing)
        if price is None and bid_price is not None:
            try:
                bid_decimal = Decimal(str(bid_price))
                if bid_decimal > 0:
                    price = bid_decimal
                    lg.warning(f"Using 'bid' price as final fallback for {symbol}: {price}")
            except (InvalidOperation, ValueError, TypeError) as e:
                lg.warning(f"Could not parse 'bid' price ({bid_price}) for {symbol}: {e}")

        # --- Final Check ---
        if price is not None and price > 0:
            return price
        else:
            lg.error(f"{NEON_RED}Failed to fetch a valid positive current price for {symbol} from ticker.{RESET}")
            return None

    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error fetching price for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}Exchange error fetching price for {symbol}: {e}{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error fetching price for {symbol}: {e}{RESET}", exc_info=True)
    return None

# fetch_klines_ccxt, fetch_orderbook_ccxt are largely unchanged, just ensure market_info is passed
def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 250, logger: logging.Logger = None, market_info: Dict = None) -> pd.DataFrame:
    """Fetch OHLCV kline data using CCXT with retries and basic validation."""
    lg = logger or logging.getLogger(__name__)
    try:
        if not exchange.has['fetchOHLCV']:
            lg.error(f"Exchange {exchange.id} does not support fetchOHLCV.")
            return pd.DataFrame()

        ohlcv = None
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                lg.debug(f"Fetching klines for {symbol}, {timeframe}, limit={limit} (Attempt {attempt+1}/{MAX_API_RETRIES + 1})")
                params = {}
                market_id = symbol
                if 'bybit' in exchange.id.lower() and market_info:
                     market_id = market_info.get('id', symbol)
                     category = None
                     if market_info.get('is_contract'):
                         category = 'linear' if market_info.get('linear', True) else 'inverse'
                     elif market_info.get('spot'):
                         category = 'spot'
                     if category:
                         params['category'] = category
                         # Use market_id from market_info if available
                         params['symbol'] = market_id
                         lg.debug(f"Using params for fetch_ohlcv ({symbol}): {params}")

                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params=params)
                if ohlcv is not None and len(ohlcv) > 0: # Basic check if data was returned and not empty
                    break # Success
                else:
                    lg.warning(f"fetch_ohlcv returned {type(ohlcv)} (length {len(ohlcv) if ohlcv is not None else 'N/A'}) for {symbol} (Attempt {attempt+1}). Retrying...")
                    time.sleep(1) # Short delay

            except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                if attempt < MAX_API_RETRIES:
                    lg.warning(f"Network error fetching klines for {symbol} (Attempt {attempt + 1}/{MAX_API_RETRIES + 1}): {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    lg.error(f"{NEON_RED}Max retries reached fetching klines for {symbol} after network errors.{RESET}")
                    raise e # Re-raise the last error
            except ccxt.RateLimitExceeded as e:
                wait_time = exchange.rateLimit / 1000.0 + 1 # Use CCXT's rateLimit property + buffer
                lg.warning(f"Rate limit exceeded fetching klines for {symbol}. Retrying in {wait_time:.2f}s... (Attempt {attempt+1})")
                time.sleep(wait_time)
            except ccxt.ExchangeError as e:
                lg.error(f"{NEON_RED}Exchange error fetching klines for {symbol}: {e}{RESET}")
                raise e # Re-raise non-network errors immediately
            except Exception as e:
                lg.error(f"{NEON_RED}Unexpected error fetching klines for {symbol}: {e}{RESET}", exc_info=True)
                raise e


        if not ohlcv: # Check if list is empty or still None after retries/errors
            lg.warning(f"{NEON_YELLOW}No kline data returned for {symbol} {timeframe} after retries.{RESET}")
            return pd.DataFrame()

        # --- Data Processing ---
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if df.empty:
            lg.warning(f"{NEON_YELLOW}Kline data DataFrame is empty for {symbol} {timeframe} immediately after creation.{RESET}")
            return df

        # Convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True) # Drop rows with invalid timestamps
        if df.empty:
            lg.warning(f"{NEON_YELLOW}Kline data DataFrame empty after timestamp conversion for {symbol} {timeframe}.{RESET}")
            return df
        df.set_index('timestamp', inplace=True)

        # Ensure numeric types, coerce errors to NaN
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- Data Cleaning ---
        initial_len = len(df)
        # Drop rows with any NaN in critical price columns
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        # Drop rows with non-positive close price (invalid data)
        df = df[df['close'] > 0]

        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
            lg.debug(f"Dropped {rows_dropped} rows with NaN/invalid price/volume data for {symbol}.")

        if df.empty:
            lg.warning(f"{NEON_YELLOW}Kline data for {symbol} {timeframe} was empty after processing/cleaning.{RESET}")
            return pd.DataFrame()

        df.sort_index(inplace=True) # Sort index just in case
        lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}")
        return df

    except Exception as e: # Catch any error not explicitly handled during retries
        lg.error(f"{NEON_RED}Unexpected error fetching or processing klines for {symbol}: {e}{RESET}", exc_info=True)
    return pd.DataFrame()

def fetch_orderbook_ccxt(exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger, market_info: Dict = None) -> Optional[Dict]:
    """Fetch orderbook data using ccxt with retries and basic validation."""
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            if not exchange.has['fetchOrderBook']:
                lg.error(f"Exchange {exchange.id} does not support fetchOrderBook.")
                return None

            lg.debug(f"Fetching order book for {symbol}, limit={limit} (Attempt {attempts+1}/{MAX_API_RETRIES + 1})")
            params = {}
            market_id = symbol
            if 'bybit' in exchange.id.lower() and market_info:
                 market_id = market_info.get('id', symbol)
                 category = None
                 if market_info.get('is_contract'):
                     category = 'linear' if market_info.get('linear', True) else 'inverse'
                 elif market_info.get('spot'):
                     category = 'spot'
                 if category:
                     params['category'] = category
                     # Use market_id from market_info if available
                     params['symbol'] = market_id
                     lg.debug(f"Using params for fetch_order_book ({symbol}): {params}")

            orderbook = exchange.fetch_order_book(symbol, limit=limit, params=params)

            # --- Validation ---
            if not orderbook:
                lg.warning(f"fetch_order_book returned None or empty data for {symbol} (Attempt {attempts+1}).")
            elif not isinstance(orderbook.get('bids'), list) or not isinstance(orderbook.get('asks'), list):
                lg.warning(f"{NEON_YELLOW}Invalid orderbook structure (bids/asks not lists) for {symbol}. Attempt {attempts + 1}.{RESET}")
            elif not orderbook['bids'] and not orderbook['asks']:
                 lg.warning(f"{NEON_YELLOW}Orderbook received but bids and asks lists are both empty for {symbol}. (Attempt {attempts + 1}).{RESET}")
                 return orderbook # Return the empty book
            else:
                 lg.debug(f"Successfully fetched orderbook for {symbol} with {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks.")
                 return orderbook

        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            lg.warning(f"{NEON_YELLOW}Orderbook fetch network error for {symbol}: {e}. Retrying in {RETRY_DELAY_SECONDS}s... (Attempt {attempts + 1}/{MAX_API_RETRIES + 1}){RESET}")
        except ccxt.RateLimitExceeded as e:
            wait_time = exchange.rateLimit / 1000.0 + 1
            lg.warning(f"Rate limit exceeded fetching orderbook for {symbol}. Retrying in {wait_time:.2f}s... (Attempt {attempts+1})")
            time.sleep(wait_time)
            attempts += 1 # Increment here after sleep
            continue # Skip standard delay
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error fetching orderbook for {symbol}: {e}{RESET}")
            return None # Don't retry definitive errors
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
# (Class definition remains largely the same, ensure __init__ uses symbol_state correctly)
class TradingAnalyzer:
    """Analyzes trading data using pandas_ta and generates weighted signals."""
    def __init__(
        self,
        df: pd.DataFrame,
        logger: logging.Logger,
        config: Dict[str, Any],
        market_info: Dict[str, Any], # Pass market info for precision etc.
        symbol_state: Dict[str, Any], # Pass mutable state dict for the symbol
    ) -> None:
        self.df = df # Expects index 'timestamp' and columns 'open', 'high', 'low', 'close', 'volume'
        self.logger = logger
        self.config = config
        if not market_info:
            raise ValueError("TradingAnalyzer requires valid market_info.")
        self.market_info = market_info
        self.symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
        self.interval = config.get("interval", "UNKNOWN_INTERVAL")
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval, "UNKNOWN_INTERVAL")
        self.indicator_values: Dict[str, float] = {} # Stores latest indicator float values
        self.signals: Dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 0} # Simple signal state
        self.active_weight_set_name = config.get("active_weight_set", "default")
        self.weights = config.get("weight_sets",{}).get(self.active_weight_set_name, {})
        self.fib_levels_data: Dict[str, Decimal] = {} # Stores calculated fib levels (as Decimals)
        self.ta_column_names: Dict[str, Optional[str]] = {} # Stores actual column names generated by pandas_ta

        # --- Symbol State (mutable) ---
        # Use the passed-in state dictionary
        if 'break_even_triggered' not in symbol_state:
             symbol_state['break_even_triggered'] = False # Initialize if first time
             self.logger.debug(f"Initialized state for {self.symbol}: {symbol_state}")
        self.symbol_state = symbol_state # Keep reference

        if not self.weights:
            logger.error(f"Active weight set '{self.active_weight_set_name}' not found or empty in config for {self.symbol}. Indicator weighting will not work.")

        # Calculate indicators, update latest values, calculate fib levels
        self._calculate_all_indicators()
        self._update_latest_indicator_values()
        self.calculate_fibonacci_levels()

    @property
    def break_even_triggered(self) -> bool:
        return self.symbol_state.get('break_even_triggered', False)

    @break_even_triggered.setter
    def break_even_triggered(self, value: bool):
        self.symbol_state['break_even_triggered'] = value
        self.logger.debug(f"Updated break_even_triggered state for {self.symbol} to {value}.")

    # (_get_ta_col_name, _calculate_all_indicators, _update_latest_indicator_values,
    #  calculate_fibonacci_levels, get_price_precision, get_min_tick_size, get_nearest_fibonacci_levels,
    #  calculate_ema_alignment_score, generate_trading_signal, _check_* methods,
    #  calculate_entry_tp_sl)
    # ... methods remain the same as provided in whale2.1.py ...
    # ... (Paste the corrected versions of these methods here from the previous iteration) ...
    # --- Fibonacci Calculation ---
    def calculate_fibonacci_levels(self, window: Optional[int] = None) -> Dict[str, Decimal]:
        """Calculates Fibonacci retracement levels over a specified window using Decimal."""
        window = window or self.config.get("fibonacci_window", DEFAULT_FIB_WINDOW)
        if len(self.df) < window:
            self.logger.debug(f"Not enough data ({len(self.df)}) for Fibonacci window ({window}) on {self.symbol}. Skipping.")
            self.fib_levels_data = {}
            return {}

        df_slice = self.df.tail(window)
        try:
            # Ensure high/low are valid numbers before converting to Decimal
            high_price_raw = df_slice["high"].dropna().max()
            low_price_raw = df_slice["low"].dropna().min()

            if pd.isna(high_price_raw) or pd.isna(low_price_raw):
                self.logger.warning(f"Could not find valid high/low in the last {window} periods for Fibonacci on {self.symbol}.")
                self.fib_levels_data = {}
                return {}

            high = Decimal(str(high_price_raw))
            low = Decimal(str(low_price_raw))
            diff = high - low

            levels = {}
            min_tick = self.get_min_tick_size()
            if min_tick <= 0:
                self.logger.warning(f"Invalid min_tick_size ({min_tick}) for Fibonacci quantization on {self.symbol}. Levels will not be quantized.")
                min_tick = None # Disable quantization if tick size is invalid

            if diff > 0:
                for level_pct in FIB_LEVELS:
                    level_name = f"Fib_{level_pct * 100:.1f}%"
                    # Calculate level: High - (Range * Pct) for downtrend assumption (standard)
                    # Or Low + (Range * Pct) for uptrend assumption
                    # Using High - diff * level assumes retracement from High towards Low
                    level_price = (high - (diff * Decimal(str(level_pct))))

                    # Quantize the result to the market's price precision (using tick size) if possible
                    if min_tick:
                        level_price_quantized = (level_price / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                    else:
                        level_price_quantized = level_price # Use raw value if quantization fails

                    levels[level_name] = level_price_quantized
            else:
                 # If high == low, all levels will be the same
                 self.logger.debug(f"Fibonacci range is zero (High={high}, Low={low}) for {self.symbol} in window {window}. Setting levels to high/low.")
                 if min_tick and min_tick > 0: # Check min_tick again
                     level_price_quantized = (high / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                 else:
                     level_price_quantized = high # Use raw if quantization fails
                 for level_pct in FIB_LEVELS:
                     levels[f"Fib_{level_pct * 100:.1f}%"] = level_price_quantized

            self.fib_levels_data = levels
            self.logger.debug(f"Calculated Fibonacci levels for {self.symbol}: { {k: str(v) for k,v in levels.items()} }") # Log as strings
            return levels
        except Exception as e:
            self.logger.error(f"{NEON_RED}Fibonacci calculation error for {self.symbol}: {e}{RESET}", exc_info=True)
            self.fib_levels_data = {}
            return {}

    # --- Risk Management Calculations ---
    def calculate_entry_tp_sl(
        self, entry_price: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """
        Calculates potential take profit (TP) and initial stop loss (SL) levels
        based on the provided entry price, ATR, and configured multipliers. Uses Decimal precision.
        The initial SL calculated here is primarily used for position sizing and setting the initial protection.
        Returns (entry_price, take_profit, stop_loss), all as Decimal or None.
        """
        if signal not in ["BUY", "SELL"]:
            return entry_price, None, None # No TP/SL needed for HOLD

        atr_val_float = self.indicator_values.get("ATR") # Get float ATR value
        if atr_val_float is None or pd.isna(atr_val_float) or atr_val_float <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: ATR is invalid ({atr_val_float}).{RESET}")
            return entry_price, None, None
        if entry_price is None or entry_price.is_nan() or entry_price <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: Provided entry price is invalid ({entry_price}).{RESET}")
            return entry_price, None, None

        try:
            atr = Decimal(str(atr_val_float)) # Convert valid float ATR to Decimal

            # Get multipliers from config, convert to Decimal
            tp_multiple = Decimal(str(self.config.get("take_profit_multiple", 1.0)))
            sl_multiple = Decimal(str(self.config.get("stop_loss_multiple", 1.5)))

            # Get market precision for logging
            price_precision = self.get_price_precision()
            # Get minimum price increment (tick size) for quantization
            min_tick = self.get_min_tick_size()
            if min_tick <= 0:
                 self.logger.error(f"Cannot calculate TP/SL for {self.symbol}: Invalid min tick size ({min_tick}).")
                 return entry_price, None, None


            take_profit = None
            stop_loss = None

            if signal == "BUY":
                tp_offset = atr * tp_multiple
                sl_offset = atr * sl_multiple
                take_profit_raw = entry_price + tp_offset
                stop_loss_raw = entry_price - sl_offset
                # Quantize TP UP, SL DOWN to the nearest tick size
                take_profit = (take_profit_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick
                stop_loss = (stop_loss_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick

            elif signal == "SELL":
                tp_offset = atr * tp_multiple
                sl_offset = atr * sl_multiple
                take_profit_raw = entry_price - tp_offset
                stop_loss_raw = entry_price + sl_offset
                # Quantize TP DOWN, SL UP to the nearest tick size
                take_profit = (take_profit_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                stop_loss = (stop_loss_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick


            # --- Validation ---
            # Ensure SL is actually beyond entry by at least one tick
            min_sl_distance = min_tick # Minimum distance from entry for SL
            if signal == "BUY" and stop_loss >= entry_price:
                adjusted_sl = (entry_price - min_sl_distance).quantize(min_tick, rounding=ROUND_DOWN)
                self.logger.warning(f"{NEON_YELLOW}BUY signal SL calculation ({stop_loss}) is too close to or above entry ({entry_price}). Adjusting SL down to {adjusted_sl}.{RESET}")
                stop_loss = adjusted_sl
            elif signal == "SELL" and stop_loss <= entry_price:
                adjusted_sl = (entry_price + min_sl_distance).quantize(min_tick, rounding=ROUND_UP)
                self.logger.warning(f"{NEON_YELLOW}SELL signal SL calculation ({stop_loss}) is too close to or below entry ({entry_price}). Adjusting SL up to {adjusted_sl}.{RESET}")
                stop_loss = adjusted_sl

            # Ensure TP is potentially profitable relative to entry (at least one tick away)
            min_tp_distance = min_tick # Minimum distance from entry for TP
            if signal == "BUY" and take_profit <= entry_price:
                adjusted_tp = (entry_price + min_tp_distance).quantize(min_tick, rounding=ROUND_UP)
                self.logger.warning(f"{NEON_YELLOW}BUY signal TP calculation ({take_profit}) resulted in non-profitable level. Adjusting TP up to {adjusted_tp}.{RESET}")
                take_profit = adjusted_tp
            elif signal == "SELL" and take_profit >= entry_price:
                adjusted_tp = (entry_price - min_tp_distance).quantize(min_tick, rounding=ROUND_DOWN)
                self.logger.warning(f"{NEON_YELLOW}SELL signal TP calculation ({take_profit}) resulted in non-profitable level. Adjusting TP down to {adjusted_tp}.{RESET}")
                take_profit = adjusted_tp

            # Final checks: Ensure SL/TP are still valid (positive price) after adjustments
            if stop_loss is not None and stop_loss <= 0:
                self.logger.error(f"{NEON_RED}Stop loss calculation resulted in zero or negative price ({stop_loss}) for {self.symbol}. Cannot set SL.{RESET}")
                stop_loss = None
            if take_profit is not None and take_profit <= 0:
                self.logger.error(f"{NEON_RED}Take profit calculation resulted in zero or negative price ({take_profit}) for {self.symbol}. Cannot set TP.{RESET}")
                take_profit = None

            # Format for logging
            tp_log = f"{take_profit:.{price_precision}f}" if take_profit else 'N/A'
            sl_log = f"{stop_loss:.{price_precision}f}" if stop_loss else 'N/A'
            atr_log = f"{atr:.{price_precision+1}f}" if atr else 'N/A'
            entry_log = f"{entry_price:.{price_precision}f}" if entry_price else 'N/A'

            self.logger.debug(f"Calculated TP/SL for {self.symbol} {signal}: Entry={entry_log}, TP={tp_log}, SL={sl_log}, ATR={atr_log}")
            return entry_price, take_profit, stop_loss

        except (InvalidOperation, ValueError, TypeError, Exception) as e:
            self.logger.error(f"{NEON_RED}Error calculating TP/SL for {self.symbol}: {e}{RESET}", exc_info=True)
            return entry_price, None, None


# --- Trading Logic Helper Functions ---
# (fetch_balance, get_market_info, calculate_position_size, _manual_amount_rounding,
#  get_open_position, set_leverage_ccxt, place_trade, _set_position_protection,
#  set_trailing_stop_loss)
# ... functions remain the same as provided in whale2.1.py ...
# ... (Paste the corrected versions of these functions here from the previous iteration) ...


# --- Main Analysis and Trading Logic ---

# Global dictionary to hold state per symbol (e.g., break_even_triggered)
# Allows state to persist across calls to analyze_and_trade_symbol
symbol_states: Dict[str, Dict[str, Any]] = {}

def analyze_and_trade_symbol(
    exchange: ccxt.Exchange,
    symbol: str,
    config: Dict[str, Any],
    logger: logging.Logger,
    market_info: Dict, # Pass pre-fetched market info
    symbol_state: Dict[str, Any], # Pass mutable state dict for the symbol
    current_balance: Optional[Decimal], # Pass current balance
    all_open_positions: Dict[str, Dict] # Pass dict of all open positions {symbol: position_dict}
) -> None:
    """Analyzes a single symbol and executes/manages trades based on signals and config."""
    lg = logger # Use the symbol-specific logger
    lg.info(f"---== Analyzing {symbol} ({config['interval']}) ==---")
    cycle_start_time = time.monotonic()

    # --- 1. Fetch Data (Kline, Ticker, Orderbook) ---
    ccxt_interval = CCXT_INTERVAL_MAP.get(config["interval"])
    # Determine required kline history
    required_kline_history = max([
        config.get(p, globals().get(f"DEFAULT_{p.upper()}", 14)) # Get period from config or default const
        for p in ["atr_period", "ema_long_period", "cci_window", "williams_r_window",
                  "mfi_window", "sma_10_window", "rsi_period", "bollinger_bands_period",
                  "volume_ma_period", "fibonacci_window"]
    ] + [
        config.get("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW) + \
        config.get("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW) + \
        max(config.get("stoch_rsi_k", DEFAULT_K_WINDOW), config.get("stoch_rsi_d", DEFAULT_D_WINDOW))
    ]) + 50 # Add buffer

    kline_limit = max(250, required_kline_history) # Fetch reasonable minimum or required amount
    klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=kline_limit, logger=lg, market_info=market_info)
    if klines_df.empty or len(klines_df) < max(50, required_kline_history): # Check sufficient data
        lg.error(f"{NEON_RED}Failed to fetch sufficient kline data for {symbol} (fetched {len(klines_df)}, required ~{max(50, required_kline_history)}). Skipping analysis.{RESET}")
        return

    # Fetch current price using market_info
    current_price = fetch_current_price_ccxt(exchange, symbol, lg, market_info=market_info)
    if current_price is None:
        lg.warning(f"{NEON_YELLOW}Failed to fetch current ticker price for {symbol}. Using last close from klines as fallback.{RESET}")
        try:
            last_close_val = klines_df['close'].iloc[-1]
            if pd.notna(last_close_val) and last_close_val > 0:
                current_price = Decimal(str(last_close_val))
                lg.info(f"Using last close price: {current_price}")
            else:
                lg.error(f"{NEON_RED}Last close price ({last_close_val}) is also invalid. Cannot proceed for {symbol}.{RESET}")
                return
        except (IndexError, ValueError, TypeError, InvalidOperation) as e:
            lg.error(f"{NEON_RED}Error getting last close price for {symbol}: {e}. Cannot proceed.{RESET}")
            return

    # Fetch order book if enabled and weighted
    orderbook_data = None
    active_weights = config.get("weight_sets", {}).get(config.get("active_weight_set", "default"), {})
    if config.get("indicators",{}).get("orderbook", False) and Decimal(str(active_weights.get("orderbook", 0))) != 0:
        orderbook_data = fetch_orderbook_ccxt(exchange, symbol, config.get("orderbook_limit", 25), lg, market_info=market_info)
        if orderbook_data is None:
             lg.warning(f"{NEON_YELLOW}Orderbook fetching enabled but failed for {symbol}. Score will be NaN.{RESET}")


    # --- 2. Analyze Data & Generate Signal ---
    try:
        # Pass persistent symbol_state dict
        analyzer = TradingAnalyzer(
            df=klines_df.copy(),
            logger=lg,
            config=config,
            market_info=market_info,
            symbol_state=symbol_state
        )
    except ValueError as e_analyzer:
        lg.error(f"{NEON_RED}Failed to initialize TradingAnalyzer for {symbol}: {e_analyzer}. Skipping analysis.{RESET}")
        return

    # Generate the trading signal
    signal = analyzer.generate_trading_signal(current_price, orderbook_data)

    # Get necessary data points for potential trade/management
    price_precision = analyzer.get_price_precision()
    min_tick_size = analyzer.get_min_tick_size()
    current_atr_float = analyzer.indicator_values.get("ATR")

    if min_tick_size <= 0:
         lg.error(f"{NEON_RED}Invalid minimum tick size ({min_tick_size}) for {symbol}. Cannot calculate trade parameters. Skipping cycle.{RESET}")
         return

    # Calculate potential initial SL/TP (used for sizing if no position exists)
    _, tp_potential, sl_potential = analyzer.calculate_entry_tp_sl(current_price, signal)


    # --- 3. Log Analysis Summary ---
    atr_log = f"{current_atr_float:.{price_precision+1}f}" if current_atr_float is not None and pd.notna(current_atr_float) else 'N/A'
    sl_pot_log = f"{sl_potential:.{price_precision}f}" if sl_potential else 'N/A'
    tp_pot_log = f"{tp_potential:.{price_precision}f}" if tp_potential else 'N/A'
    lg.info(f"Analysis: ATR={atr_log}, Potential SL={sl_pot_log}, Potential TP={tp_pot_log}")

    tsl_enabled = config.get('enable_trailing_stop', False)
    be_enabled = config.get('enable_break_even', False)
    ma_exit_enabled = config.get('enable_ma_cross_exit', False)
    lg.info(f"Config: TSL={tsl_enabled}, BE={be_enabled}, MA Cross Exit={ma_exit_enabled}")


    # --- 4. Check Position & Execute/Manage ---
    if not config.get("enable_trading", False):
        lg.info(f"{NEON_YELLOW}Trading is disabled in config. Analysis complete, no trade actions taken for {symbol}.{RESET}")
        cycle_end_time = time.monotonic()
        lg.debug(f"---== Analysis Cycle End for {symbol} ({cycle_end_time - cycle_start_time:.2f}s) ==---")
        return

    # Use the pre-fetched position data for this symbol
    open_position = all_open_positions.get(symbol) # Get position for this symbol, will be None if not open

    # Get current count of ALL open positions (across all symbols)
    current_total_open_positions = len(all_open_positions)

    # --- Scenario 1: No Open Position for this Symbol ---
    if open_position is None:
        if signal in ["BUY", "SELL"]:
            # --- Check Max Position Limit ---
            max_pos_total = config.get("max_concurrent_positions_total", 1)
            if current_total_open_positions >= max_pos_total:
                 lg.info(f"{NEON_YELLOW}Signal {signal} for {symbol}, but max concurrent positions ({max_pos_total}) already reached. Skipping new entry.{RESET}")
                 return

            lg.info(f"*** {signal} Signal & No Open Position: Initiating Trade Sequence for {symbol} ***")

            # --- Pre-Trade Checks & Setup ---
            # Balance check (using balance passed from main loop)
            if current_balance is None or current_balance <= 0:
                lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Invalid or zero balance ({current_balance} {QUOTE_CURRENCY}).{RESET}")
                return

            # Check potential SL (needed for sizing)
            if sl_potential is None:
                lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Cannot size trade, Potential Initial Stop Loss calculation failed (ATR invalid?).{RESET}")
                return

            # Set Leverage (only for contracts)
            if market_info.get('is_contract', False):
                leverage = int(config.get("leverage", 1))
                if leverage > 0:
                    # Check current leverage ONLY if necessary (e.g., if it varies per symbol or needs verification)
                    # For simplicity, we attempt to set it based on config every time before entry.
                    if not set_leverage_ccxt(exchange, symbol, leverage, market_info, lg):
                        lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Failed to set/confirm leverage to {leverage}x. Check logs.{RESET}")
                        return
            else:
                 lg.info(f"Leverage setting skipped for {symbol} (Spot market).")


            # Calculate Position Size using potential SL
            position_size = calculate_position_size(
                balance=current_balance, # Use passed balance
                risk_per_trade=config["risk_per_trade"],
                initial_stop_loss_price=sl_potential, # Use potential SL for sizing
                entry_price=current_price, # Use current price as estimate
                market_info=market_info,
                exchange=exchange,
                logger=lg
            )

            if position_size is None or position_size <= 0:
                lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Invalid position size calculated ({position_size}). Check balance, risk, SL, market limits.{RESET}")
                return

            # --- Place Initial Market Order ---
            lg.info(f"==> Placing {signal} market order for {symbol} | Size: {position_size} <==")
            trade_order = place_trade(
                exchange=exchange,
                symbol=symbol,
                trade_signal=signal,
                position_size=position_size, # Pass Decimal size
                market_info=market_info,
                logger=lg,
                reduce_only=False # Opening trade
            )

            # --- Post-Order: Verify Position and Set Protection ---
            if trade_order and trade_order.get('id'):
                order_id = trade_order['id']
                lg.info(f"Order {order_id} placed for {symbol}. Waiting {POSITION_CONFIRM_DELAY}s for position confirmation...")
                time.sleep(POSITION_CONFIRM_DELAY)

                # Re-fetch position state specifically for this symbol AFTER the delay
                lg.info(f"Attempting to confirm position for {symbol} after order {order_id}...")
                confirmed_position = get_open_position(exchange, symbol, lg, market_info=market_info, position_mode=config.get("position_mode", "One-Way"))

                if confirmed_position:
                    # --- Position Confirmed ---
                    try:
                        entry_price_actual = confirmed_position.get('entryPriceDecimal') # Use Decimal version
                        pos_size_actual = confirmed_position.get('contracts') # Use Decimal version
                        valid_entry = False

                        # Validate actual entry price and size
                        if entry_price_actual and pos_size_actual:
                             min_size_threshold = Decimal('1e-9') # Basic non-zero check
                             if market_info:
                                 try: min_size_threshold = max(min_size_threshold, Decimal(str(market_info['limits']['amount']['min'])) * Decimal('0.01'))
                                 except Exception: pass

                             if entry_price_actual > 0 and abs(pos_size_actual) >= min_size_threshold:
                                 valid_entry = True
                             else:
                                 lg.error(f"Confirmed position has invalid entry price ({entry_price_actual}) or size ({pos_size_actual}).")
                        else:
                            lg.error("Confirmed position missing valid entryPrice or size Decimal information.")


                        if valid_entry:
                            lg.info(f"{NEON_GREEN}Position Confirmed for {symbol}! Actual Entry: ~{entry_price_actual:.{price_precision}f}, Actual Size: {pos_size_actual}{RESET}")

                            # --- Recalculate SL/TP based on ACTUAL entry price ---
                            _, tp_actual, sl_actual = analyzer.calculate_entry_tp_sl(entry_price_actual, signal)

                            # --- Set Protection based on Config (TSL or Fixed SL/TP) ---
                            protection_set_success = False
                            if config.get("enable_trailing_stop", False):
                                lg.info(f"Setting Trailing Stop Loss for {symbol} (TP target: {tp_actual})...")
                                protection_set_success = set_trailing_stop_loss(
                                    exchange=exchange, symbol=symbol, market_info=market_info,
                                    position_info=confirmed_position, config=config, logger=lg,
                                    take_profit_price=tp_actual # Pass optional TP
                                )
                            else:
                                lg.info(f"Setting Fixed Stop Loss ({sl_actual}) and Take Profit ({tp_actual}) for {symbol}...")
                                if sl_actual or tp_actual: # Only call if at least one is valid
                                    protection_set_success = _set_position_protection(
                                        exchange=exchange, symbol=symbol, market_info=market_info,
                                        position_info=confirmed_position, logger=lg,
                                        stop_loss_price=sl_actual, # Pass Decimal or None
                                        take_profit_price=tp_actual, # Pass Decimal or None
                                        trailing_stop_distance='0', # Ensure TSL is cancelled
                                        tsl_activation_price='0'
                                    )
                                else:
                                    lg.warning(f"{NEON_YELLOW}Fixed SL/TP calculation based on actual entry failed or resulted in None for {symbol}. No fixed protection set.{RESET}")
                                    protection_set_success = True # Treat as 'success' if no protection needed

                            # --- Final Status Log ---
                            if protection_set_success:
                                lg.info(f"{NEON_GREEN}=== TRADE ENTRY & PROTECTION SETUP COMPLETE for {symbol} ({signal}) ===")
                                # Reset BE state for the new trade
                                analyzer.break_even_triggered = False
                            else:
                                lg.error(f"{NEON_RED}=== TRADE placed BUT FAILED TO SET/CONFIRM PROTECTION (SL/TP/TSL) for {symbol} ({signal}) ===")
                                lg.warning(f"{NEON_YELLOW}Position is open without automated protection. Manual monitoring/intervention may be required!{RESET}")
                        else:
                            lg.error(f"{NEON_RED}Trade placed, but confirmed position data is invalid. Cannot set protection.{RESET}")
                            lg.warning(f"{NEON_YELLOW}Position may be open without protection. Manual check needed!{RESET}")

                    except Exception as post_trade_err:
                        lg.error(f"{NEON_RED}Error during post-trade processing (protection setting) for {symbol}: {post_trade_err}{RESET}", exc_info=True)
                        lg.warning(f"{NEON_YELLOW}Position may be open without protection. Manual check needed!{RESET}")

                else:
                    # --- Position NOT Confirmed ---
                    lg.error(f"{NEON_RED}Trade order {order_id} placed, but FAILED TO CONFIRM open position for {symbol} after {POSITION_CONFIRM_DELAY}s delay!{RESET}")
                    lg.warning(f"{NEON_YELLOW}Order might have failed to fill, filled partially, rejected, or there's a significant delay/API issue.{RESET}")
                    lg.warning(f"{NEON_YELLOW}No protection will be set. Manual investigation of order {order_id} and position status required!{RESET}")
                    # Optional: Try fetching the order status itself for more info
                    try:
                        order_status_info = exchange.fetch_order(order_id, symbol)
                        lg.info(f"Status of order {order_id}: {order_status_info.get('status', 'Unknown')}, Filled: {order_status_info.get('filled', 'N/A')}")
                    except Exception as fetch_order_err:
                        lg.warning(f"Could not fetch status for order {order_id}: {fetch_order_err}")

            else:
                # place_trade function returned None (order failed)
                lg.error(f"{NEON_RED}=== TRADE EXECUTION FAILED for {symbol} ({signal}). See previous logs for details. ===")
        else:
            # No open position, and signal is HOLD
            lg.info(f"Signal is HOLD and no open position for {symbol}. No trade action taken.")


    # --- Scenario 2: Existing Open Position Found for this Symbol ---
    else: # open_position is not None
        pos_side = open_position.get('side', 'unknown')
        pos_size_dec = open_position.get('contracts') # Should be Decimal from get_open_position
        entry_price_dec = open_position.get('entryPriceDecimal') # Use Decimal entry
        # Use parsed Decimal versions for protection checks
        current_sl_price_dec = open_position.get('stopLossPriceDecimal') # Might be None or Decimal(0)
        current_tp_price_dec = open_position.get('takeProfitPriceDecimal') # Might be None or Decimal(0)
        tsl_distance_dec = open_position.get('trailingStopLossDistanceDecimal', Decimal(0))
        is_tsl_active = tsl_distance_dec > 0

        # Format for logging
        sl_log_str = f"{current_sl_price_dec:.{price_precision}f}" if current_sl_price_dec else 'N/A'
        tp_log_str = f"{current_tp_price_dec:.{price_precision}f}" if current_tp_price_dec else 'N/A'
        pos_size_log = f"{pos_size_dec:.8f}" if pos_size_dec else 'N/A'
        entry_price_log = f"{entry_price_dec:.{price_precision}f}" if entry_price_dec else 'N/A'

        lg.info(f"Existing {pos_side.upper()} position found for {symbol}. Size: {pos_size_log}, Entry: {entry_price_log}, SL: {sl_log_str}, TP: {tp_log_str}, TSL Active: {is_tsl_active}")

        # Validate essential position info for management
        if pos_side not in ['long', 'short'] or not entry_price_dec or not pos_size_dec or pos_size_dec <= 0:
             lg.error(f"{NEON_RED}Cannot manage existing position for {symbol}: Invalid side ('{pos_side}'), entry price ('{entry_price_dec}'), or size ('{pos_size_dec}') detected.{RESET}")
             return


        # --- ** Check Exit Conditions ** ---
        # Priority: 1. Main Signal, 2. MA Cross (if enabled)
        # Note: Exchange SL/TP/TSL trigger independently

        # Condition 1: Main signal opposes the current position direction
        exit_signal_triggered = False
        if (pos_side == 'long' and signal == "SELL") or \
           (pos_side == 'short' and signal == "BUY"):
            exit_signal_triggered = True
            lg.warning(f"{NEON_YELLOW}*** EXIT Signal Triggered: New signal ({signal}) opposes existing {pos_side} position for {symbol}. ***{RESET}")

        # Condition 2: MA Cross exit (if enabled and not already exiting via signal)
        ma_cross_exit = False
        if not exit_signal_triggered and config.get("enable_ma_cross_exit", False):
            ema_short = analyzer.indicator_values.get("EMA_Short")
            ema_long = analyzer.indicator_values.get("EMA_Long")

            if pd.notna(ema_short) and pd.notna(ema_long):
                # Check for adverse cross based on position side
                if pos_side == 'long' and ema_short < ema_long:
                    ma_cross_exit = True
                    lg.warning(f"{NEON_YELLOW}*** MA CROSS EXIT (Bearish): Short EMA ({ema_short:.{price_precision}f}) crossed below Long EMA ({ema_long:.{price_precision}f}). Closing LONG position. ***{RESET}")
                elif pos_side == 'short' and ema_short > ema_long:
                    ma_cross_exit = True
                    lg.warning(f"{NEON_YELLOW}*** MA CROSS EXIT (Bullish): Short EMA ({ema_short:.{price_precision}f}) crossed above Long EMA ({ema_long:.{price_precision}f}). Closing SHORT position. ***{RESET}")
            else:
                lg.warning("MA cross exit check skipped: EMA values not available.")


        # --- Execute Position Close if Exit Condition Met ---
        if exit_signal_triggered or ma_cross_exit:
            lg.info(f"Attempting to close {pos_side} position for {symbol} with a market order...")
            try:
                # Determine the side needed to close the position
                close_side_signal = "SELL" if pos_side == 'long' else "BUY"
                # Use the Decimal size from the fetched position info
                size_to_close = abs(pos_size_dec)
                if size_to_close <= 0: # Should be caught by earlier check, but defensive
                    raise ValueError(f"Cannot close: Existing position size {pos_size_dec} is zero or invalid.")

                # --- Place Closing Market Order (reduceOnly=True) ---
                close_order = place_trade(
                    exchange=exchange, symbol=symbol, trade_signal=close_side_signal,
                    position_size=size_to_close, market_info=market_info, logger=lg,
                    reduce_only=True # CRITICAL for closing
                )

                if close_order:
                     order_id = close_order.get('id', 'N/A')
                     # Handle the "Position not found" dummy response from place_trade
                     if close_order.get('info',{}).get('reason') == 'Position not found on close attempt':
                          lg.info(f"{NEON_GREEN}Position for {symbol} confirmed already closed (or closing order unnecessary).{RESET}")
                          # Reset state if position is closed
                          analyzer.break_even_triggered = False
                     else:
                          lg.info(f"{NEON_GREEN}Closing order {order_id} placed for {symbol}. Waiting {POSITION_CONFIRM_DELAY}s to verify closure...{RESET}")
                          time.sleep(POSITION_CONFIRM_DELAY)
                          # Verify position is actually closed
                          final_position = get_open_position(exchange, symbol, lg, market_info=market_info, position_mode=config.get("position_mode", "One-Way"))
                          if final_position is None:
                              lg.info(f"{NEON_GREEN}=== POSITION for {symbol} successfully closed. ===")
                              # Reset state after successful closure
                              analyzer.break_even_triggered = False
                          else:
                              lg.error(f"{NEON_RED}*** POSITION CLOSE FAILED for {symbol} after placing reduceOnly order {order_id}. Position still detected: {final_position}{RESET}")
                              lg.warning(f"{NEON_YELLOW}Manual investigation required!{RESET}")
                else:
                    lg.error(f"{NEON_RED}Failed to place closing order for {symbol}. Manual intervention required.{RESET}")

            except (ValueError, InvalidOperation, TypeError) as size_err:
                 lg.error(f"{NEON_RED}Error determining size for closing order ({symbol}): {size_err}. Manual intervention required.{RESET}")
            except Exception as close_err:
                 lg.error(f"{NEON_RED}Unexpected error during position close attempt for {symbol}: {close_err}{RESET}", exc_info=True)
                 lg.warning(f"{NEON_YELLOW}Manual intervention required!{RESET}")

        # --- Check Break-Even Condition (Only if NOT Exiting and BE not already triggered for this trade) ---
        elif config.get("enable_break_even", False) and not analyzer.break_even_triggered:
            # Use already validated entry_price_dec and current_atr_float
            if current_atr_float and pd.notna(current_atr_float) and current_atr_float > 0:
                try:
                    current_atr_dec = Decimal(str(current_atr_float))
                    trigger_multiple = Decimal(str(config.get("break_even_trigger_atr_multiple", 1.0)))
                    offset_ticks = int(config.get("break_even_offset_ticks", 2))
                    min_tick = min_tick_size # Use min_tick_size fetched earlier

                    profit_target_offset = current_atr_dec * trigger_multiple
                    be_stop_price = None
                    trigger_met = False
                    trigger_price = None

                    if pos_side == 'long':
                        trigger_price = entry_price_dec + profit_target_offset
                        if current_price >= trigger_price:
                            trigger_met = True
                            be_stop_raw = entry_price_dec + (min_tick * offset_ticks)
                            be_stop_price = (be_stop_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick
                            if be_stop_price <= entry_price_dec:
                                be_stop_price = entry_price_dec + min_tick

                    elif pos_side == 'short':
                        trigger_price = entry_price_dec - profit_target_offset
                        if current_price <= trigger_price:
                            trigger_met = True
                            be_stop_raw = entry_price_dec - (min_tick * offset_ticks)
                            be_stop_price = (be_stop_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                            if be_stop_price >= entry_price_dec:
                                be_stop_price = entry_price_dec - min_tick

                    # --- Trigger Break-Even SL Modification ---
                    if trigger_met:
                        if be_stop_price and be_stop_price > 0:
                            lg.warning(f"{NEON_PURPLE}*** BREAK-EVEN Triggered for {symbol} {pos_side.upper()} position! ***")
                            lg.info(f"  Current Price: {current_price:.{price_precision}f}, Trigger Price: {trigger_price:.{price_precision}f}")
                            lg.info(f"  Current SL: {sl_log_str}, Current TP: {tp_log_str}, TSL Active: {is_tsl_active}")
                            lg.info(f"  Moving Stop Loss to: {be_stop_price:.{price_precision}f} (Entry {entry_price_dec:.{price_precision}f} +/- {offset_ticks} ticks)")

                            # Determine if SL needs update based on current protection and BE target
                            needs_sl_update = True
                            if not config.get("break_even_force_fixed_sl", True) and is_tsl_active:
                                 # If not forcing fixed SL and TSL is active, potentially keep TSL.
                                 # This requires more complex logic to check if TSL's current stop is better than BE.
                                 # Simplified approach for now: Force Fixed SL if break_even_force_fixed_sl is True
                                 lg.warning("BE triggered with TSL active and break_even_force_fixed_sl=False - Keeping TSL. (Complex state - Verify behavior!)")
                                 needs_sl_update = False # Don't change SL if keeping TSL
                                 analyzer.break_even_triggered = True # Mark BE as logic-triggered
                            elif current_sl_price_dec: # Fixed SL exists
                                if (pos_side == 'long' and current_sl_price_dec >= be_stop_price) or \
                                   (pos_side == 'short' and current_sl_price_dec <= be_stop_price):
                                     lg.info(f"  Existing SL ({sl_log_str}) is already at or better than break-even target ({be_stop_price:.{price_precision}f}). No SL modification needed.")
                                     needs_sl_update = False
                                     analyzer.break_even_triggered = True

                            if needs_sl_update:
                                lg.info(f"  Modifying protection: Setting Fixed SL to {be_stop_price:.{price_precision}f}, keeping TP {tp_log_str}, cancelling TSL (if active).")
                                be_success = _set_position_protection(
                                    exchange=exchange, symbol=symbol, market_info=market_info,
                                    position_info=open_position, logger=lg,
                                    stop_loss_price=be_stop_price, # New BE SL
                                    take_profit_price=current_tp_price_dec, # Keep existing TP
                                    trailing_stop_distance='0', # Cancel TSL when setting fixed BE SL
                                    tsl_activation_price='0'
                                )
                                if be_success:
                                    lg.info(f"{NEON_GREEN}Break-even stop loss successfully set/updated for {symbol}.{RESET}")
                                    analyzer.break_even_triggered = True # Mark BE as actioned
                                else:
                                    lg.error(f"{NEON_RED}Failed to set break-even stop loss for {symbol}. Original protection may still be active.{RESET}")

                        else: # be_stop_price calculation failed or invalid
                            lg.error(f"{NEON_RED}Break-even triggered, but calculated BE stop price ({be_stop_price}) is invalid. Cannot modify SL.{RESET}")

                except (InvalidOperation, ValueError, TypeError, KeyError, Exception) as be_err:
                    lg.error(f"{NEON_RED}Error during break-even check/calculation for {symbol}: {be_err}{RESET}", exc_info=True)
            else:
                # Log reason if BE check skipped due to missing data
                if not entry_price_dec or entry_price_dec <= 0: lg.debug(f"BE check skipped: Invalid entry price.")
                elif not current_atr_float or not pd.notna(current_atr_float) or current_atr_float <= 0: lg.debug(f"BE check skipped: Invalid ATR.")
                elif not min_tick_size or min_tick_size <= 0: lg.debug(f"BE check skipped: Invalid min tick size.")

        # --- No Exit/BE Condition Met ---
        else:
             # If not exiting and BE not triggered (or already handled), log holding state
             lg.info(f"Signal is {signal}. Holding existing {pos_side.upper()} position for {symbol}. No management action required this cycle.")
             # Optional: Add logic here to adjust TP/SL based on new analysis if desired,
             # being careful not to conflict with TSL or BE logic.

    # --- End of Scenario 2 (Existing Open Position) ---

    cycle_end_time = time.monotonic()
    lg.info(f"---== Analysis Cycle End for {symbol} ({cycle_end_time - cycle_start_time:.2f}s) ==---")


# --- Main Function ---
def main():
    """Main function to run the bot."""
    # Determine console log level from arguments
    parser = argparse.ArgumentParser(description="Enhanced Bybit Trading Bot (Whale 2.1)")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable DEBUG level logging to console")
    parser.add_argument("-s", "--symbol", help="Trade only a specific symbol (e.g., BTC/USDT:USDT), overrides config list")
    args = parser.parse_args()

    global console_log_level
    if args.debug:
        console_log_level = logging.DEBUG
        print(f"{NEON_YELLOW}Console logging set to DEBUG level.{RESET}")
    else:
        console_log_level = logging.INFO

    # Setup main logger (used for init and overall status)
    main_logger = get_logger("main")
    main_logger.info(f"*** Live Trading Bot Whale 2.1 Initializing ***")
    main_logger.info(f"Using config file: {CONFIG_FILE}")
    main_logger.info(f"Logging to directory: {LOG_DIRECTORY}")
    main_logger.info(f"Configured Timezone: {TIMEZONE}")
    main_logger.info(f"Trading Enabled: {CONFIG.get('enable_trading', False)}")
    main_logger.info(f"Using Sandbox: {CONFIG.get('use_sandbox', True)}")
    main_logger.info(f"Quote Currency: {QUOTE_CURRENCY}")
    main_logger.info(f"Risk Per Trade: {CONFIG.get('risk_per_trade', 0.01):.2%}")
    main_logger.info(f"Leverage: {CONFIG.get('leverage', 1)}x")
    main_logger.info(f"Max Total Positions: {CONFIG.get('max_concurrent_positions_total', 1)}")
    main_logger.info(f"Position Mode (Config): {CONFIG.get('position_mode', 'One-Way')}")


    # Validate interval from config
    if CONFIG.get("interval") not in VALID_INTERVALS:
        main_logger.error(f"{NEON_RED}Invalid 'interval' ({CONFIG.get('interval')}) in {CONFIG_FILE}. Must be one of {VALID_INTERVALS}. Exiting.{RESET}")
        return

    # Initialize exchange
    exchange = initialize_exchange(main_logger)
    if not exchange:
        main_logger.critical(f"{NEON_RED}Failed to initialize exchange. Bot cannot start.{RESET}")
        return

    # Determine symbols to trade
    symbols_to_trade = []
    if args.symbol:
         symbols_to_trade = [args.symbol]
         main_logger.info(f"Trading only specified symbol: {args.symbol}")
    else:
         symbols_to_trade = CONFIG.get("symbols", [])
         if not symbols_to_trade:
              main_logger.error(f"{NEON_RED}No symbols configured in {CONFIG_FILE} and no symbol specified via argument. Exiting.{RESET}")
              return
         main_logger.info(f"Trading symbols from config: {', '.join(symbols_to_trade)}")

    # Validate symbols and fetch initial market info
    valid_symbols = []
    market_infos: Dict[str, Dict] = {}
    symbol_loggers: Dict[str, logging.Logger] = {}

    main_logger.info("Validating symbols and fetching market info...")
    for symbol in symbols_to_trade:
        logger = get_logger(symbol, is_symbol_logger=True) # Get/create logger for the symbol
        symbol_loggers[symbol] = logger # Store logger
        market_info = get_market_info(exchange, symbol, logger)
        if market_info:
            valid_symbols.append(symbol)
            market_infos[symbol] = market_info
            # Initialize state for valid symbols
            symbol_states[symbol] = {} # Empty dict to start
        else:
            logger.error(f"Symbol {symbol} is invalid or market info could not be fetched. It will be skipped.")

    if not valid_symbols:
        main_logger.error(f"{NEON_RED}No valid symbols remaining after validation. Exiting.{RESET}")
        return

    main_logger.info(f"Validated symbols: {', '.join(valid_symbols)}")

    # --- Bot Main Loop ---
    main_logger.info(f"Starting main trading loop for {len(valid_symbols)} symbols...")
    while True:
        loop_start_utc = datetime.now(ZoneInfo("UTC"))
        loop_start_local = loop_start_utc.astimezone(TIMEZONE)
        main_logger.debug(f"--- New Main Loop Cycle --- | {loop_start_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        try:
            # Fetch global data once per loop cycle (balance, all positions)
            current_balance = fetch_balance(exchange, QUOTE_CURRENCY, main_logger)
            if current_balance is None:
                 main_logger.warning(f"{NEON_YELLOW}Failed to fetch balance for {QUOTE_CURRENCY} this cycle. Size calculations may fail.{RESET}")
                 # Decide if this is critical - maybe skip trading this cycle?
                 # For now, continue and let analyze_and_trade_symbol handle missing balance.

            # Fetch all open positions once
            all_positions_raw = []
            try:
                 # Fetch positions based on category if possible (more efficient for Bybit V5)
                 # Fetch linear and inverse separately if trading both types
                 # For simplicity, fetch all contract types if any contracts are being traded
                 if any(market_infos[s].get('is_contract') for s in valid_symbols):
                      try:
                           all_positions_raw.extend(exchange.fetch_positions(params={'category': 'linear'}))
                      except Exception as e_lin:
                           main_logger.warning(f"Could not fetch linear positions: {e_lin}")
                      try:
                           all_positions_raw.extend(exchange.fetch_positions(params={'category': 'inverse'}))
                      except Exception as e_inv:
                           main_logger.warning(f"Could not fetch inverse positions: {e_inv}")
                 # Fetch spot positions if trading spot
                 if any(market_infos[s].get('type') == 'spot' for s in valid_symbols):
                     try:
                         # Spot positions are usually implied by balance, but fetch_positions might return something for margin spot
                         # For now, assume spot positions aren't fetched this way, handled by balance
                         pass
                     except Exception as e_spot:
                          main_logger.warning(f"Could not fetch spot positions: {e_spot}")

                 # If fetching by category failed or wasn't applicable, try fetching all
                 if not all_positions_raw:
                      all_positions_raw = exchange.fetch_positions()

            except Exception as e_pos:
                main_logger.error(f"{NEON_RED}Failed to fetch all open positions this cycle: {e_pos}{RESET}")
                # Need to decide how to handle this - skip management? Assume no positions?
                all_positions_raw = [] # Assume none if fetch failed

            # Process raw positions into a dictionary keyed by symbol
            all_open_positions: Dict[str, Dict] = {}
            for pos in all_positions_raw:
                 if pos.get('symbol') in valid_symbols:
                     # Perform basic validation (size > 0) before adding
                     pos_size_str = pos.get('contracts', pos.get('info',{}).get('size', '0'))
                     try:
                         pos_size = Decimal(str(pos_size_str))
                         # Use a threshold slightly larger than zero
                         if abs(pos_size) > Decimal('1e-9'):
                              # Enhance position info (add Decimal versions, etc.) - Reuse logic from get_open_position
                              enhanced_pos = pos.copy() # Work on a copy
                              info_dict = enhanced_pos.get('info', {})
                              # Add Decimal versions for consistency
                              try: enhanced_pos['entryPriceDecimal'] = Decimal(str(enhanced_pos.get('entryPrice', info_dict.get('avgPrice','0'))))
                              except: enhanced_pos['entryPriceDecimal'] = None
                              try: enhanced_pos['contractsDecimal'] = Decimal(str(enhanced_pos.get('contracts', info_dict.get('size','0'))))
                              except: enhanced_pos['contractsDecimal'] = None
                              # ... (add other decimal enhancements from get_open_position if needed directly here) ...

                              all_open_positions[pos['symbol']] = enhanced_pos # Store the enhanced position
                     except (InvalidOperation, ValueError, TypeError):
                         pass # Ignore positions with unparseable size

            main_logger.info(f"Total open positions detected: {len(all_open_positions)}")


            # --- Iterate Through Symbols ---
            for symbol in valid_symbols:
                symbol_logger = symbol_loggers[symbol]
                symbol_market_info = market_infos[symbol]
                current_symbol_state = symbol_states[symbol] # Get the persistent state dict

                # Run analysis and trading logic for the individual symbol
                analyze_and_trade_symbol(
                    exchange=exchange,
                    symbol=symbol,
                    config=CONFIG,
                    logger=symbol_logger,
                    market_info=symbol_market_info,
                    symbol_state=current_symbol_state,
                    current_balance=current_balance, # Pass overall balance
                    all_open_positions=all_open_positions # Pass all positions dict
                )
                # Short delay between processing symbols to avoid hitting rate limits too quickly if many symbols
                time.sleep(0.5) # Adjust as needed

        except KeyboardInterrupt:
            main_logger.info("KeyboardInterrupt received. Attempting graceful shutdown...")
            # Optional: Add logic here to close open positions if desired on exit
            # close_all_open_positions(exchange, all_open_positions, market_infos, main_logger)
            main_logger.info("Shutdown complete.")
            break
        except ccxt.AuthenticationError as e:
            main_logger.critical(f"{NEON_RED}CRITICAL: Authentication Error during main loop: {e}. Bot stopped. Check API keys/permissions.{RESET}")
            break # Stop on authentication errors
        except ccxt.NetworkError as e:
            main_logger.error(f"{NEON_RED}Network Error in main loop: {e}. Retrying after longer delay...{RESET}")
            time.sleep(RETRY_DELAY_SECONDS * 10) # Longer delay for network issues
        except ccxt.RateLimitExceeded as e:
             main_logger.warning(f"{NEON_YELLOW}Rate Limit Exceeded in main loop: {e}. CCXT should handle internally, but consider increasing loop delay if frequent.{RESET}")
             time.sleep(RETRY_DELAY_SECONDS * 2) # Short delay
        except ccxt.ExchangeNotAvailable as e:
            main_logger.error(f"{NEON_RED}Exchange Not Available: {e}. Retrying after significant delay...{RESET}")
            time.sleep(LOOP_DELAY_SECONDS * 5) # Long delay
        except ccxt.ExchangeError as e:
            bybit_code = getattr(e, 'code', None)
            err_str = str(e).lower()
            main_logger.error(f"{NEON_RED}Exchange Error encountered in main loop: {e} (Code: {bybit_code}){RESET}")
            if bybit_code == 10016 or "system maintenance" in err_str:
                main_logger.warning(f"{NEON_YELLOW}Exchange likely in maintenance (Code: {bybit_code}). Waiting longer...{RESET}")
                time.sleep(LOOP_DELAY_SECONDS * 10) # Wait several minutes
            else:
                # For other exchange errors, retry after a moderate delay
                time.sleep(RETRY_DELAY_SECONDS * 3)
        except Exception as e:
            main_logger.error(f"{NEON_RED}An unexpected critical error occurred in the main loop: {e}{RESET}", exc_info=True)
            main_logger.error("Bot encountered a potentially fatal error. For safety, the bot will stop.")
            # Consider sending a notification here (email, Telegram, etc.)
            break # Stop the bot on unexpected errors

        # --- Loop Delay ---
        # Use LOOP_DELAY_SECONDS loaded from config
        main_logger.debug(f"Main loop cycle finished. Sleeping for {LOOP_DELAY_SECONDS} seconds...")
        time.sleep(LOOP_DELAY_SECONDS)

    # --- End of Main Loop ---
    main_logger.info(f"*** Live Trading Bot has stopped. ***")

# --- Entry Point ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Catch any uncaught exceptions during initialization or main loop exit
        print(f"{NEON_RED}Unhandled exception reached top level: {e}{RESET}")
        # Attempt to log if possible, otherwise just print
        try:
            # Use the main logger if available
            logger = get_logger("main")
            logger.critical("Unhandled exception caused script termination.", exc_info=True)
        except Exception: # Ignore logging errors during final exception handling
             import traceback
             traceback.print_exc()
    finally:
         print(f"{NEON_CYAN}Bot execution finished.{RESET}")