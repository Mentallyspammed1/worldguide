# whale2.6.py
# Enhanced version focusing on stop-loss/take-profit mechanisms,
# including break-even logic, MA cross exit condition, correct Bybit V5 API usage,
# multi-symbol support, and improved state management.

import argparse
import json
import logging
import os
import time
from datetime import datetime
from decimal import ROUND_DOWN, ROUND_UP, Decimal, InvalidOperation, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any
from zoneinfo import ZoneInfo  # Use zoneinfo for modern timezone handling

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
from colorama import Fore, Style, init
from dotenv import load_dotenv

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
TIMEZONE = ZoneInfo("America/Chicago")  # e.g., "America/New_York", "Europe/London", "Asia/Tokyo"
MAX_API_RETRIES = 3  # Max retries for recoverable API errors
RETRY_DELAY_SECONDS = 5  # Delay between retries
# Intervals supported by the bot's logic. Note: Only certain intervals might have enough liquidity for scalping/intraday.
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP = {  # Map our intervals to ccxt's expected format
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}
RETRY_ERROR_CODES = [429, 500, 502, 503, 504]  # HTTP status codes considered retryable

# Default periods (can be overridden by config.json)
DEFAULT_ATR_PERIOD = 14
DEFAULT_CCI_WINDOW = 20
DEFAULT_WILLIAMS_R_WINDOW = 14
DEFAULT_MFI_WINDOW = 14
DEFAULT_STOCH_RSI_WINDOW = 14  # Window for Stoch RSI calculation itself
DEFAULT_STOCH_WINDOW = 14     # Window for underlying RSI in StochRSI (often same as Stoch RSI window)
DEFAULT_K_WINDOW = 3          # K period for StochRSI
DEFAULT_D_WINDOW = 3          # D period for StochRSI
DEFAULT_RSI_WINDOW = 12
DEFAULT_BOLLINGER_BANDS_PERIOD = 30
DEFAULT_BOLLINGER_BANDS_STD_DEV = 2.0  # Ensure float
DEFAULT_SMA_10_WINDOW = 10
DEFAULT_EMA_SHORT_PERIOD = 8
DEFAULT_EMA_LONG_PERIOD = 22
DEFAULT_MOMENTUM_PERIOD = 7
DEFAULT_VOLUME_MA_PERIOD = 15
DEFAULT_FIB_WINDOW = 50
DEFAULT_PSAR_AF = 0.02
DEFAULT_PSAR_MAX_AF = 0.2

FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]  # Standard Fibonacci levels
# LOOP_DELAY_SECONDS is now loaded from config
POSITION_CONFIRM_DELAY = 10  # Seconds to wait after placing order before checking position status
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


def load_config(filepath: str) -> dict[str, Any]:
    """Load configuration from JSON file, creating default if not found,
    and ensuring all default keys are present.
    """
    default_config = {
        "symbols": ["BCH/USDT:USDT"],  # List of symbols to trade (CCXT format)
        "interval": "3",  # Default to 5 minute interval (string format for our logic)
        "loop_delay": 15,  # Default delay between full bot cycles (processing all symbols)
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
        "orderbook_limit": 25,  # Depth of orderbook to fetch
        "signal_score_threshold": 1.5,  # Score needed to trigger BUY/SELL signal for 'default' weight set
        "scalping_signal_threshold": 2.5,  # Separate threshold for 'scalping' weight set
        "stoch_rsi_oversold_threshold": 25,
        "stoch_rsi_overbought_threshold": 75,
        "stop_loss_multiple": 1.8,  # ATR multiple for initial SL (used for sizing)
        "take_profit_multiple": 0.7,  # ATR multiple for TP
        "volume_confirmation_multiplier": 1.5,  # How much higher volume needs to be than MA
        "fibonacci_window": DEFAULT_FIB_WINDOW,
        "enable_trading": True,
        "use_sandbox": False,
        "risk_per_trade": 0.01,  # Risk 1% of account balance per trade
        "leverage": 20,          # Set desired leverage (check exchange limits)
        "max_concurrent_positions_total": 1,  # Global limit on open positions across all symbols
        "quote_currency": "USDT",  # Currency for balance check and sizing
        "position_mode": "One-Way",  # "One-Way" or "Hedge" - CRITICAL: Match exchange setting. Bot assumes One-Way by default.
        # --- MA Cross Exit Config ---
        "enable_ma_cross_exit": True,  # Enable closing position on adverse EMA cross
        # --- Trailing Stop Loss Config ---
        "enable_trailing_stop": True,  # Default to enabling TSL (exchange TSL)
        # Trail distance as a percentage of the entry price (e.g., 0.5%)
        # Bybit API expects absolute distance; this percentage is used for calculation.
        "trailing_stop_callback_rate": 0.005,  # Example: 0.5% trail distance relative to entry price
        # Activate TSL when price moves this percentage in profit from entry (e.g., 0.3%)
        # Set to 0 for immediate TSL activation upon entry (API value '0').
        "trailing_stop_activation_percentage": 0.003,  # Example: Activate when 0.3% in profit
        # --- Break-Even Stop Config ---
        "enable_break_even": True,              # Enable moving SL to break-even
        # Move SL when profit (in price points) reaches X * Current ATR
        "break_even_trigger_atr_multiple": 0.6,  # Example: Trigger BE when profit = 1x ATR
        # Place BE SL this many minimum price increments (ticks) beyond entry price
        # E.g., 2 ticks to cover potential commission/slippage on exit
        "break_even_offset_ticks": 2,
         # Set to True to always cancel TSL and revert to Fixed SL when BE is triggered.
         # Set to False to allow TSL to potentially remain active if it's better than BE price.
         # Note: Bybit V5 often automatically cancels TSL when a fixed SL is set, so True is generally safer.
        "break_even_force_fixed_sl": True,
        # --- End Protection Config ---
        "indicators": {  # Control which indicators are calculated and contribute to score
            "ema_alignment": True, "momentum": True, "volume_confirmation": True,
            "stoch_rsi": True, "rsi": True, "bollinger_bands": True, "vwap": True,
            "cci": True, "wr": True, "psar": True, "sma_10": True, "mfi": True,
            "orderbook": True,  # Flag to enable fetching and scoring orderbook data
        },
        "weight_sets": {  # Define different weighting strategies
            "scalping": {  # Example weighting for a fast scalping strategy
                "ema_alignment": 0.2, "momentum": 0.3, "volume_confirmation": 0.2,
                "stoch_rsi": 0.6, "rsi": 0.2, "bollinger_bands": 0.3, "vwap": 0.4,
                "cci": 0.3, "wr": 0.3, "psar": 0.2, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.15,
            },
            "default": {  # A more balanced weighting strategy
                "ema_alignment": 0.3, "momentum": 0.2, "volume_confirmation": 0.1,
                "stoch_rsi": 0.4, "rsi": 0.3, "bollinger_bands": 0.2, "vwap": 0.3,
                "cci": 0.2, "wr": 0.2, "psar": 0.3, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.1,
            }
        },
        "active_weight_set": "scalping"  # Choose which weight set to use ("default" or "scalping")
    }

    if not os.path.exists(filepath):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            return default_config
        except OSError:
            return default_config  # Return default even if creation fails

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
                except OSError:
                    pass
            return updated_config
    except (FileNotFoundError, json.JSONDecodeError):
        # Attempt to create default if loading failed badly
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
        except OSError:
            pass
        return default_config


def _ensure_config_keys(config: dict[str, Any], default_config: dict[str, Any]) -> dict[str, Any]:
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
LOOP_DELAY_SECONDS = int(CONFIG.get("loop_delay", 15))  # Use configured loop delay, default 15
console_log_level = logging.INFO  # Default console level, can be changed via args

# --- Logger Setup ---
# Global logger dictionary to manage loggers per symbol
loggers: dict[str, logging.Logger] = {}


def setup_logger(name: str, is_symbol_logger: bool = False) -> logging.Logger:
    """Sets up a logger with file and console handlers.
    If is_symbol_logger is True, uses symbol-specific naming and file.
    Otherwise, uses a generic name (e.g., 'main').
    """
    if name in loggers:
        # If logger already exists, update console level if needed
        logger = loggers[name]
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(console_log_level)
        return logger

    # Clean name for filename if it's a symbol
    safe_name = name.replace('/', '_').replace(':', '-') if is_symbol_logger else name
    logger_instance_name = f"livebot_{safe_name}"  # Unique internal name for the logger
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
        file_handler.setLevel(logging.DEBUG)  # Log everything to the file
        logger.addHandler(file_handler)
    except Exception:
        pass

    # Stream Handler (console)
    stream_handler = logging.StreamHandler()
    stream_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'  # Add timestamp format to console
    )
    stream_handler.setFormatter(stream_formatter)
    # Set console level based on global variable (which can be set by args)
    stream_handler.setLevel(console_log_level)
    logger.addHandler(stream_handler)

    # Prevent logs from propagating to the root logger (avoids duplicate outputs)
    logger.propagate = False

    loggers[name] = logger  # Store logger instance
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
def initialize_exchange(logger: logging.Logger) -> ccxt.Exchange | None:
    """Initializes the CCXT Bybit exchange object with error handling."""
    try:
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,  # Let ccxt handle basic rate limiting
            'options': {
                'defaultType': 'linear',  # Assume linear contracts (USDT margined) - adjust if needed
                'adjustForTimeDifference': True,  # Auto-sync time with server
                # Connection timeouts (milliseconds)
                'fetchTickerTimeout': 10000,  # 10 seconds
                'fetchBalanceTimeout': 15000,  # 15 seconds
                'createOrderTimeout': 20000,  # Timeout for placing orders
                'fetchOrderTimeout': 15000,  # Timeout for fetching orders
                'fetchPositionsTimeout': 15000,  # Timeout for fetching positions
                'recvWindow': 10000,  # Bybit default is 5000, 10000 is safer
                'brokerId': 'livebot71',  # Example: Add a broker ID for Bybit if desired
                # Explicitly request V5 API for relevant endpoints using 'versions'
                'versions': {
                    'public': {
                        'GET': {
                            'market/tickers': 'v5',
                            'market/kline': 'v5',
                            'market/orderbook': 'v5',
                        },
                    },
                    'private': {
                        'GET': {
                            'position/list': 'v5',
                            'account/wallet-balance': 'v5',
                            'order/realtime': 'v5',  # For fetching open orders
                            'order/history': 'v5',  # For fetching historical orders
                            # Unified Margin specific (if needed, uncomment and test)
                            'unified/account/wallet-balance': 'v5',
                            # Spot specific (if needed)
                            # 'spot/v3/private/order': 'v5',
                            # 'spot/v3/private/account': 'v5',
                        },
                        'POST': {
                            'order/create': 'v5',
                            'order/cancel': 'v5',
                            'position/set-leverage': 'v5',
                            'position/trading-stop': 'v5',  # For SL/TP/TSL
                            # Unified Margin specific (if needed)
                            'unified/private/order/create': 'v5',
                            'unified/private/order/cancel': 'v5',
                            # Spot specific (if needed)
                            # 'spot/v3/private/order': 'v5',
                            # 'spot/v3/private/cancel-order': 'v5',
                        },
                    },
                },
                 # Fallback default options if 'versions' isn't fully respected by current ccxt
                 'default_options': {
                    'adjustForTimeDifference': True,
                    'warnOnFetchOpenOrdersWithoutSymbol': False,  # Suppress warning
                    'recvWindow': 10000,
                    # Explicitly setting methods to use V5 (may be redundant with 'versions')
                    'fetchPositions': 'v5',
                    'fetchBalance': 'v5',
                    'createOrder': 'v5',
                    'fetchOrder': 'v5',
                    'fetchTicker': 'v5',
                    'fetchOHLCV': 'v5',
                    'fetchOrderBook': 'v5',
                    'setLeverage': 'v5',
                    'private_post_v5_position_trading_stop': 'v5',  # Ensure protection uses V5
                    # CCXT uses privatePostPositionTradingStop internally when calling modify_position with SL/TP
                },
                'accounts': {  # Define V5 account types if needed by ccxt version/implementation
                    'future': {'linear': 'CONTRACT', 'inverse': 'CONTRACT'},
                    'swap': {'linear': 'CONTRACT', 'inverse': 'CONTRACT'},
                    'option': {'unified': 'OPTION'},
                    'spot': {'unified': 'SPOT'},
                    'unified': {'linear': 'UNIFIED', 'inverse': 'UNIFIED', 'spot': 'UNIFIED', 'option': 'UNIFIED'},  # Unified account support
                },
                'bybit': {  # Bybit specific options
                     'defaultSettleCoin': QUOTE_CURRENCY,  # Set default settlement coin (e.g., USDT)
                }
            }
        }

        # Select Bybit class
        exchange_class = ccxt.bybit  # Use getattr for robustness
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
            exchange.load_markets()  # Fallback to default load

        exchange.last_load_markets_timestamp = time.time()  # Store timestamp for periodic reload check

        logger.info(f"CCXT exchange initialized ({exchange.id}). Sandbox: {CONFIG.get('use_sandbox')}")
        logger.info(f"CCXT version: {ccxt.__version__}")  # Log CCXT version

        # Test API credentials and permissions by fetching balance
        # Specify account type for Bybit V5 (CONTRACT for linear/USDT, UNIFIED if using that)
        account_type_to_test = 'CONTRACT'  # Start with common type
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
            return None  # Critical failure, cannot proceed
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


def _determine_category(market_info: dict) -> str | None:
    """Helper to determine Bybit V5 category from market info."""
    if not market_info:
        return None
    market_type = market_info.get('type')  # spot, swap, future, option
    is_linear = market_info.get('linear', False)
    is_inverse = market_info.get('inverse', False)
    is_contract = market_info.get('contract', False)  # General contract flag

    if market_type == 'swap' or market_type == 'future' or is_contract:
        if is_linear: return 'linear'
        if is_inverse: return 'inverse'
        # Fallback for contracts if linear/inverse isn't explicitly set (assume linear)
        return 'linear'
    elif market_type == 'spot':
        return 'spot'
    elif market_type == 'option':
        return 'option'
    # Fallback if type isn't clear but contract flags are set
    elif is_linear: return 'linear'
    elif is_inverse: return 'inverse'
    else:
        # Default guess if nothing else matches (e.g., for older CCXT versions or unusual markets)
        return 'linear'  # Or None if unsure


# --- CCXT Data Fetching Functions ---
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger, market_info: dict = None) -> Decimal | None:
    """Fetch the current price of a trading symbol using CCXT ticker with fallbacks."""
    lg = logger
    try:
        lg.debug(f"Fetching ticker for {symbol}...")
        params = {}
        if 'bybit' in exchange.id.lower() and market_info:
             market_info.get('id', symbol)  # Use exchange ID if available
             category = _determine_category(market_info)
             if category:
                 params['category'] = category
                 # Use market_id from market_info if available for Bybit V5
                 # params['symbol'] = market_id # CCXT usually handles mapping symbol to id
                 lg.debug(f"Using params for fetch_ticker ({symbol}): {params}")

        # Pass symbol and params
        ticker = exchange.fetch_ticker(symbol, params=params)

        price = None
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
                     price = ask_decimal  # Use ask as a safer fallback in this case
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


def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 250, logger: logging.Logger = None, market_info: dict = None) -> pd.DataFrame:
    """Fetch OHLCV kline data using CCXT with retries, V5 params, and basic validation."""
    lg = logger or logging.getLogger(__name__)
    try:
        if not exchange.has['fetchOHLCV']:
            lg.error(f"Exchange {exchange.id} does not support fetchOHLCV.")
            return pd.DataFrame()

        ohlcv = None
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                lg.debug(f"Fetching klines for {symbol}, {timeframe}, limit={limit} (Attempt {attempt + 1}/{MAX_API_RETRIES + 1})")
                params = {}
                if 'bybit' in exchange.id.lower() and market_info:
                     market_info.get('id', symbol)
                     category = _determine_category(market_info)
                     if category:
                         params['category'] = category
                         # params['symbol'] = market_id # CCXT handles symbol->id mapping
                         lg.debug(f"Using params for fetch_ohlcv ({symbol}): {params}")

                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params=params)
                if ohlcv is not None and len(ohlcv) > 0:  # Basic check if data was returned and not empty
                    break  # Success
                else:
                    lg.warning(f"fetch_ohlcv returned {type(ohlcv)} (length {len(ohlcv) if ohlcv is not None else 'N/A'}) for {symbol} (Attempt {attempt + 1}). Retrying...")
                    time.sleep(1)  # Short delay

            except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                if attempt < MAX_API_RETRIES:
                    lg.warning(f"Network error fetching klines for {symbol} (Attempt {attempt + 1}/{MAX_API_RETRIES + 1}): {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    lg.error(f"{NEON_RED}Max retries reached fetching klines for {symbol} after network errors.{RESET}")
                    raise e  # Re-raise the last error
            except ccxt.RateLimitExceeded as e:
                # Use retry-after header if available, otherwise calculate wait time
                retry_after = getattr(e, 'retryAfter', None)
                if retry_after is not None:
                    wait_time = retry_after / 1000.0 + 0.1  # Use header + small buffer
                else:
                    wait_time = exchange.rateLimit / 1000.0 + 1  # Use CCXT's rateLimit property + buffer
                lg.warning(f"Rate limit exceeded fetching klines for {symbol}. Retrying in {wait_time:.2f}s... (Attempt {attempt + 1})")
                time.sleep(wait_time)
            except ccxt.ExchangeError as e:
                lg.error(f"{NEON_RED}Exchange error fetching klines for {symbol}: {e}{RESET}")
                raise e  # Re-raise non-network errors immediately
            except Exception as e:
                lg.error(f"{NEON_RED}Unexpected error fetching klines for {symbol}: {e}{RESET}", exc_info=True)
                raise e

        if not ohlcv:  # Check if list is empty or still None after retries/errors
            lg.warning(f"{NEON_YELLOW}No kline data returned for {symbol} {timeframe} after retries.{RESET}")
            return pd.DataFrame()

        # --- Data Processing ---
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if df.empty:
            lg.warning(f"{NEON_YELLOW}Kline data DataFrame is empty for {symbol} {timeframe} immediately after creation.{RESET}")
            return df

        # Convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)  # Drop rows with invalid timestamps
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

        df.sort_index(inplace=True)  # Sort index just in case
        lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}")
        return df

    except Exception as e:  # Catch any error not explicitly handled during retries
        lg.error(f"{NEON_RED}Unexpected error fetching or processing klines for {symbol}: {e}{RESET}", exc_info=True)
    return pd.DataFrame()


def fetch_orderbook_ccxt(exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger, market_info: dict = None) -> dict | None:
    """Fetch orderbook data using ccxt with retries, V5 params, and basic validation."""
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            if not exchange.has['fetchOrderBook']:
                lg.error(f"Exchange {exchange.id} does not support fetchOrderBook.")
                return None

            lg.debug(f"Fetching order book for {symbol}, limit={limit} (Attempt {attempts + 1}/{MAX_API_RETRIES + 1})")
            params = {}
            if 'bybit' in exchange.id.lower() and market_info:
                 market_info.get('id', symbol)
                 category = _determine_category(market_info)
                 if category:
                     params['category'] = category
                     # params['symbol'] = market_id # CCXT handles mapping
                     lg.debug(f"Using params for fetch_order_book ({symbol}): {params}")

            orderbook = exchange.fetch_order_book(symbol, limit=limit, params=params)

            # --- Validation ---
            if not orderbook:
                lg.warning(f"fetch_order_book returned None or empty data for {symbol} (Attempt {attempts + 1}).")
            elif not isinstance(orderbook.get('bids'), list) or not isinstance(orderbook.get('asks'), list):
                lg.warning(f"{NEON_YELLOW}Invalid orderbook structure (bids/asks not lists) for {symbol}. Attempt {attempts + 1}.{RESET}")
            elif not orderbook['bids'] and not orderbook['asks']:
                 lg.warning(f"{NEON_YELLOW}Orderbook received but bids and asks lists are both empty for {symbol}. (Attempt {attempts + 1}).{RESET}")
                 return orderbook  # Return the empty book
            else:
                 lg.debug(f"Successfully fetched orderbook for {symbol} with {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks.")
                 return orderbook

        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            lg.warning(f"{NEON_YELLOW}Orderbook fetch network error for {symbol}: {e}. Retrying in {RETRY_DELAY_SECONDS}s... (Attempt {attempts + 1}/{MAX_API_RETRIES + 1}){RESET}")
        except ccxt.RateLimitExceeded as e:
            retry_after = getattr(e, 'retryAfter', None)
            if retry_after is not None: wait_time = retry_after / 1000.0 + 0.1
            else: wait_time = exchange.rateLimit / 1000.0 + 1
            lg.warning(f"Rate limit exceeded fetching orderbook for {symbol}. Retrying in {wait_time:.2f}s... (Attempt {attempts + 1})")
            time.sleep(wait_time)
            attempts += 1  # Increment here after sleep
            continue  # Skip standard delay
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error fetching orderbook for {symbol}: {e}{RESET}")
            return None  # Don't retry definitive errors
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching orderbook for {symbol}: {e}{RESET}", exc_info=True)
            return None  # Don't retry unexpected errors

        # Increment attempt counter and wait before retrying network/validation issues
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)

    lg.error(f"{NEON_RED}Max retries reached fetching orderbook for {symbol}.{RESET}")
    return None


# --- Trading Analyzer Class ---
class TradingAnalyzer:
    """Analyzes trading data using pandas_ta and generates weighted signals."""
    def __init__(
        self,
        df: pd.DataFrame,
        logger: logging.Logger,
        config: dict[str, Any],
        market_info: dict[str, Any],  # Pass market info for precision etc.
        symbol_state: dict[str, Any],  # Pass mutable state dict for the symbol
    ) -> None:
        self.df = df  # Expects index 'timestamp' and columns 'open', 'high', 'low', 'close', 'volume'
        self.logger = logger
        self.config = config
        if not market_info:
            raise ValueError("TradingAnalyzer requires valid market_info.")
        self.market_info = market_info
        self.symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
        self.interval = config.get("interval", "UNKNOWN_INTERVAL")
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval, "UNKNOWN_INTERVAL")
        self.indicator_values: dict[str, float | None] = {}  # Stores latest indicator float values, allow None
        self.signals: dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 0}  # Simple signal state
        self.active_weight_set_name = config.get("active_weight_set", "default")
        self.weights = config.get("weight_sets", {}).get(self.active_weight_set_name, {})
        self.fib_levels_data: dict[str, Decimal] = {}  # Stores calculated fib levels (as Decimals)
        self.ta_column_names: dict[str, str | None] = {}  # Stores actual column names generated by pandas_ta

        # --- Symbol State (mutable) ---
        # Use the passed-in state dictionary
        if 'break_even_triggered' not in symbol_state:
             symbol_state['break_even_triggered'] = False  # Initialize if first time
             self.logger.debug(f"Initialized state for {self.symbol}: {symbol_state}")
        self.symbol_state = symbol_state  # Keep reference

        if not self.weights:
            logger.error(f"Active weight set '{self.active_weight_set_name}' not found or empty in config for {self.symbol}. Indicator weighting will not work.")

        # Calculate indicators, update latest values, calculate fib levels
        self._calculate_all_indicators()
        self._update_latest_indicator_values()
        self.calculate_fibonacci_levels()

    @property
    def break_even_triggered(self) -> bool:
        """Gets the break-even status from the shared symbol state."""
        return self.symbol_state.get('break_even_triggered', False)

    @break_even_triggered.setter
    def break_even_triggered(self, value: bool) -> None:
        """Sets the break-even status in the shared symbol state."""
        self.symbol_state['break_even_triggered'] = value
        self.logger.debug(f"Updated break_even_triggered state for {self.symbol} to {value}.")

    def _get_ta_col_name(self, base_name: str, default: str | None = None) -> str | None:
        """Helper to get the actual column name generated by pandas_ta, handling potential parameter variations."""
        # Note: This implementation relies on the names stored during calculation.
        # A more robust version might search df.columns if the stored name is missing.
        return self.ta_column_names.get(base_name, default)

    def _calculate_all_indicators(self) -> None:
        """Calculates all configured technical indicators using pandas_ta."""
        self.logger.debug(f"Calculating indicators for {self.symbol}...")
        if self.df.empty:
            self.logger.warning(f"DataFrame is empty for {self.symbol}, cannot calculate indicators.")
            return

        # Use config values or defaults
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

        # Ensure bb_std is float
        try:
            bb_std_float = float(bb_std)
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid Bollinger Bands std dev ({bb_std}). Using default {DEFAULT_BOLLINGER_BANDS_STD_DEV}.")
            bb_std_float = DEFAULT_BOLLINGER_BANDS_STD_DEV

        # Custom TA Strategy Definition using pandas_ta.Strategy
        MyStrategy = ta.Strategy(
            name="MultiIndicatorStrategy_Whale",
            description="Combines various TA indicators for signal generation",
            ta=[
                # Volatility / Risk
                {"kind": "atr", "length": atr_period},
                # Trend / Alignment
                {"kind": "ema", "length": ema_short_period},
                {"kind": "ema", "length": ema_long_period},
                {"kind": "vwap"},  # Volume Weighted Average Price
                # Momentum / Oscillators
                {"kind": "rsi", "length": rsi_period},
                {"kind": "cci", "length": cci_window},
                {"kind": "willr", "length": wr_window},
                {"kind": "mfi", "length": mfi_window},
                {"kind": "stochrsi", "length": stoch_rsi_window, "rsi_length": stoch_rsi_rsi_window, "k": stoch_rsi_k, "d": stoch_rsi_d},
                {"kind": "mom", "length": momentum_period},  # Momentum indicator
                # Bollinger Bands
                {"kind": "bbands", "length": bb_period, "std": bb_std_float},
                # Volume Confirmation
                {"kind": "sma", "close": "volume", "length": volume_ma_period, "prefix": "VOL"},  # Volume MA
                # Other potentially useful
                {"kind": "psar", "af": psar_af, "max_af": psar_max_af},  # Parabolic SAR
                {"kind": "sma", "length": sma10_window},  # Simple Moving Average 10
            ]
        )

        try:
            # Run the strategy - this appends columns to self.df
            self.df.ta.strategy(MyStrategy)

            # Store actual generated column names (handles potential variations)
            # Construct names based on expected pandas_ta output format
            self.ta_column_names = {
                "ATR": f"ATRr_{atr_period}",
                "EMA_Short": f"EMA_{ema_short_period}",
                "EMA_Long": f"EMA_{ema_long_period}",
                "RSI": f"RSI_{rsi_period}",
                "BBL": f"BBL_{bb_period}_{bb_std_float:.1f}",  # Lower BBand
                "BBM": f"BBM_{bb_period}_{bb_std_float:.1f}",  # Middle BBand (SMA)
                "BBU": f"BBU_{bb_period}_{bb_std_float:.1f}",  # Upper BBand
                "CCI": f"CCI_{cci_window}_0.015",  # Pandas TA adds constant suffix
                "WR": f"WILLR_{wr_window}",
                "MFI": f"MFI_{mfi_window}",
                "STOCHRSIk": f"STOCHRSIk_{stoch_rsi_window}_{stoch_rsi_rsi_window}_{stoch_rsi_k}_{stoch_rsi_d}",
                "STOCHRSId": f"STOCHRSId_{stoch_rsi_window}_{stoch_rsi_rsi_window}_{stoch_rsi_k}_{stoch_rsi_d}",
                "MOM": f"MOM_{momentum_period}",
                "VOL_MA": f"VOL_SMA_{volume_ma_period}",
                "PSARl": f"PSARl_{psar_af}_{psar_max_af}",  # Long PSAR
                "PSARs": f"PSARs_{psar_af}_{psar_max_af}",  # Short PSAR
                "SMA10": f"SMA_{sma10_window}",
                "VWAP": "VWAP_D",  # VWAP usually daily ('D') by default - may vary
            }
            # Verify that the generated columns exist
            missing_cols = []
            for key, col_name in self.ta_column_names.items():
                 # Check for VWAP variations if default name not found
                 if key == "VWAP" and col_name not in self.df.columns:
                     if "VWAP" in self.df.columns:  # Check for simple 'VWAP'
                         self.ta_column_names[key] = "VWAP"
                     elif col_name not in self.df.columns:  # Still missing after check
                          missing_cols.append(col_name)
                 elif col_name not in self.df.columns:
                     missing_cols.append(col_name)

            if missing_cols:
                 self.logger.warning(f"Following expected indicator columns not found in DataFrame after calculation for {self.symbol}: {missing_cols}. Check pandas_ta version or strategy definition. Values will be None.")
                 # Set missing column names to None in our mapping
                 for key, col_name in self.ta_column_names.items():
                      if col_name in missing_cols:
                           self.ta_column_names[key] = None

            self.logger.debug(f"Indicators calculated for {self.symbol}. DataFrame columns: {self.df.columns.tolist()}")

        except Exception as e:
            self.logger.error(f"{NEON_RED}Error calculating indicators for {self.symbol}: {e}{RESET}", exc_info=True)
            # Set all mapped column names to None if calculation failed catastrophically
            self.ta_column_names = dict.fromkeys(self.ta_column_names)

    def _update_latest_indicator_values(self) -> None:
        """Extracts the latest values of calculated indicators."""
        self.indicator_values = {}  # Reset
        if self.df.empty or self.df.index.empty:
            self.logger.warning(f"DataFrame empty, cannot update latest indicator values for {self.symbol}.")
            return

        try:
            latest_index = self.df.index[-1]
            latest_data = self.df.iloc[-1]
        except IndexError:
            self.logger.warning(f"Cannot access last row of DataFrame for {self.symbol} (IndexError).")
            return

        for key, col_name in self.ta_column_names.items():
            if col_name and col_name in latest_data.index:  # Check if column exists in the Series
                value = latest_data[col_name]
                # Convert to float, handle NaN/None
                try:
                    # Check for pandas NA types as well
                    if pd.isna(value):
                         self.indicator_values[key] = None
                    else:
                         self.indicator_values[key] = float(value)
                except (ValueError, TypeError):
                    self.indicator_values[key] = None
            else:
                # Log if column was expected but missing
                if col_name:
                    self.logger.debug(f"Column '{col_name}' for indicator '{key}' not found in latest data for {self.symbol}.")
                self.indicator_values[key] = None  # Indicator wasn't calculated or column name mismatch

        # Add latest price/volume data as well
        for col in ['close', 'high', 'low', 'open', 'volume']:
            try:
                value = latest_data[col]
                self.indicator_values[col] = float(value) if pd.notna(value) else None
            except (KeyError, ValueError, TypeError):
                 self.indicator_values[col] = None

        # Log a summary of latest values at DEBUG level
        log_vals = {k: f"{v:.4f}" if isinstance(v, float) else str(v) for k, v in self.indicator_values.items()}
        self.logger.debug(f"Latest Indicator Values ({self.symbol} @ {latest_index}): {log_vals}")

    # --- Fibonacci Calculation ---
    def calculate_fibonacci_levels(self, window: int | None = None) -> dict[str, Decimal]:
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
            min_tick = self.get_min_tick_size()  # Use helper method
            if min_tick is None or min_tick <= 0:
                self.logger.warning(f"Invalid min_tick_size ({min_tick}) for Fibonacci quantization on {self.symbol}. Levels will not be quantized.")
                min_tick = None  # Disable quantization if tick size is invalid

            if diff > 0:
                for level_pct in FIB_LEVELS:
                    level_name = f"Fib_{level_pct * 100:.1f}%"
                    # Calculate level: High - (Range * Pct) for downtrend assumption (standard)
                    level_price = (high - (diff * Decimal(str(level_pct))))

                    # Quantize the result to the market's price precision (using tick size) if possible
                    if min_tick:
                        # Ensure quantization is done correctly based on tick size
                        level_price_quantized = (level_price / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                    else:
                        level_price_quantized = level_price  # Use raw value if quantization fails

                    levels[level_name] = level_price_quantized
            else:
                 # If high == low, all levels will be the same
                 self.logger.debug(f"Fibonacci range is zero (High={high}, Low={low}) for {self.symbol} in window {window}. Setting levels to high/low.")
                 if min_tick and min_tick > 0:  # Check min_tick again
                     level_price_quantized = (high / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                 else:
                     level_price_quantized = high  # Use raw if quantization fails
                 for level_pct in FIB_LEVELS:
                     levels[f"Fib_{level_pct * 100:.1f}%"] = level_price_quantized

            self.fib_levels_data = levels
            self.logger.debug(f"Calculated Fibonacci levels for {self.symbol}: { {k: str(v) for k, v in levels.items()} }")  # Log as strings
            return levels
        except Exception as e:
            self.logger.error(f"{NEON_RED}Fibonacci calculation error for {self.symbol}: {e}{RESET}", exc_info=True)
            self.fib_levels_data = {}
            return {}

    # --- Market Info Helpers (using derived values from market_info) ---
    def get_price_precision(self) -> int:
        """Gets the number of decimal places for price from derived market info."""
        # Use the pre-calculated value stored in market_info
        precision_digits = self.market_info.get('price_precision_digits')
        if precision_digits is None:
            self.logger.warning(f"Price precision digits not found in derived market info for {self.symbol}. Using default 8.")
            return 8
        return precision_digits

    def get_min_tick_size(self) -> Decimal | None:
        """Gets the minimum price increment (tick size) as a Decimal from derived market info."""
        # Use the pre-calculated value stored in market_info
        min_tick = self.market_info.get('min_tick_size')
        if min_tick is None:
            self.logger.warning(f"Min tick size not found in derived market info for {self.symbol}. Returning None.")
            return None
        return min_tick

    # --- Signal Generation Logic ---
    def _check_ema_alignment(self) -> float:
        """Checks if short EMA is above long EMA."""
        ema_short = self.indicator_values.get("EMA_Short")
        ema_long = self.indicator_values.get("EMA_Long")
        if ema_short is None or ema_long is None: return 0.0
        return 1.0 if ema_short > ema_long else (-1.0 if ema_short < ema_long else 0.0)

    def _check_momentum(self) -> float:
        """Checks if momentum is positive."""
        mom = self.indicator_values.get("MOM")
        if mom is None: return 0.0
        # Simple check > 0 or < 0. Could be scaled based on typical range.
        return 1.0 if mom > 0 else (-1.0 if mom < 0 else 0.0)

    def _check_volume_confirmation(self) -> float:
        """Checks if current volume is significantly above its MA."""
        volume = self.indicator_values.get("volume")  # Use 'volume' key
        vol_ma = self.indicator_values.get("VOL_MA")
        multiplier = self.config.get("volume_confirmation_multiplier", 1.5)
        if volume is None or vol_ma is None or vol_ma == 0: return 0.0
        # Provide score indicating volume confirmation, not direction
        return 1.0 if volume > vol_ma * multiplier else 0.0

    def _check_stoch_rsi(self) -> float:
        """Checks Stochastic RSI K/D lines and overbought/oversold levels."""
        k = self.indicator_values.get("STOCHRSIk")
        d = self.indicator_values.get("STOCHRSId")
        oversold = self.config.get("stoch_rsi_oversold_threshold", 25)
        overbought = self.config.get("stoch_rsi_overbought_threshold", 75)
        if k is None or d is None: return 0.0

        # Prioritize crossovers and extreme levels
        if k > d and k < oversold and d < oversold: return 1.0  # Bullish crossover in oversold
        if k < d and k > overbought and d > overbought: return -1.0  # Bearish crossover in overbought
        if k < oversold and d < oversold: return 0.8  # Deep oversold (potential buy)
        if k > overbought and d > overbought: return -0.8  # Deep overbought (potential sell)
        if k > d: return 0.5  # General bullish momentum (K above D)
        if k < d: return -0.5  # General bearish momentum (K below D)
        return 0.0  # Neutral / consolidating

    def _check_rsi(self) -> float:
        """Checks RSI level (simple overbought/oversold)."""
        rsi = self.indicator_values.get("RSI")
        if rsi is None: return 0.0
        if rsi < 30: return 1.0  # Strong Oversold -> Bullish
        if rsi > 70: return -1.0  # Strong Overbought -> Bearish
        if rsi < 40: return 0.5  # Leaning oversold
        if rsi > 60: return -0.5  # Leaning overbought
        return 0.0

    def _check_bollinger_bands(self, current_price: Decimal) -> float:
        """Checks price relative to Bollinger Bands."""
        bbl = self.indicator_values.get("BBL")
        bbu = self.indicator_values.get("BBU")
        bbm = self.indicator_values.get("BBM")  # Middle band
        if bbl is None or bbu is None or bbm is None or current_price is None: return 0.0
        price_f = float(current_price)
        if price_f < bbl: return 1.0  # Below lower band -> Strong bullish potential
        if price_f > bbu: return -1.0  # Above upper band -> Strong bearish potential
        # Optional: Add logic based on middle band crossing?
        # if price_f > bbm: return 0.3 # Above middle band -> Weak bullish
        # if price_f < bbm: return -0.3 # Below middle band -> Weak bearish
        return 0.0

    def _check_vwap(self, current_price: Decimal) -> float:
        """Checks if price is above or below VWAP."""
        vwap = self.indicator_values.get("VWAP")
        if vwap is None or current_price is None: return 0.0
        price_f = float(current_price)
        return 1.0 if price_f > vwap else (-1.0 if price_f < vwap else 0.0)

    def _check_cci(self) -> float:
        """Checks CCI for overbought/oversold."""
        cci = self.indicator_values.get("CCI")
        if cci is None: return 0.0
        if cci < -150: return 1.0  # Strong Oversold -> Bullish
        if cci > 150: return -1.0  # Strong Overbought -> Bearish
        if cci < -80: return 0.5  # Oversold
        if cci > 80: return -0.5  # Overbought
        return 0.0

    def _check_wr(self) -> float:
        """Checks Williams %R for overbought/oversold."""
        wr = self.indicator_values.get("WR")
        if wr is None: return 0.0
        # Williams %R is negative, -80 is oversold, -20 is overbought
        if wr < -80: return 1.0  # Oversold -> Bullish
        if wr > -20: return -1.0  # Overbought -> Bearish
        if wr < -50: return 0.3  # Approaching oversold
        if wr > -50: return -0.3  # Approaching overbought
        return 0.0

    def _check_psar(self, current_price: Decimal) -> float:
        """Checks price relative to Parabolic SAR."""
        psar_long = self.indicator_values.get("PSARl")  # Appears when trend is potentially UP (dot below price)
        psar_short = self.indicator_values.get("PSARs")  # Appears when trend is potentially DOWN (dot above price)
        if current_price is None: return 0.0
        float(current_price)

        # If PSAR Long has a value, it's below the price (potential uptrend)
        if psar_long is not None and not np.isnan(psar_long):
            return 1.0  # Price above PSAR dot -> Bullish signal

        # If PSAR Short has a value, it's above the price (potential downtrend)
        elif psar_short is not None and not np.isnan(psar_short):
            return -1.0  # Price below PSAR dot -> Bearish signal

        return 0.0  # Neither PSAR value is valid (e.g., start of data)

    def _check_sma10(self, current_price: Decimal) -> float:
        """Checks if price is above or below SMA10."""
        sma10 = self.indicator_values.get("SMA10")
        if sma10 is None or current_price is None: return 0.0
        price_f = float(current_price)
        return 0.5 if price_f > sma10 else (-0.5 if price_f < sma10 else 0.0)  # Weaker signal

    def _check_mfi(self) -> float:
        """Checks Money Flow Index for overbought/oversold."""
        mfi = self.indicator_values.get("MFI")
        if mfi is None: return 0.0
        if mfi < 20: return 1.0  # Oversold -> Bullish
        if mfi > 80: return -1.0  # Overbought -> Bearish
        if mfi < 40: return 0.4  # Leaning oversold
        if mfi > 60: return -0.4  # Leaning overbought
        return 0.0

    def _check_orderbook_imbalance(self, orderbook: dict | None) -> float:
        """Calculates a simple order book imbalance score."""
        if orderbook is None or not orderbook.get('bids') or not orderbook.get('asks'):
            return 0.0  # Cannot calculate if orderbook is missing or empty

        try:
            # Consider top N levels (e.g., 10) or total volume within a % range
            depth = 10  # Number of levels to consider
            total_bid_volume = sum(Decimal(str(bid[1])) for bid in orderbook['bids'][:depth])
            total_ask_volume = sum(Decimal(str(ask[1])) for ask in orderbook['asks'][:depth])

            if total_bid_volume + total_ask_volume == 0:
                return 0.0

            # Imbalance ratio: (Bids - Asks) / (Bids + Asks)
            imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)

            # Scale the score (e.g., clip between -1 and 1)
            score = float(max(-1, min(1, imbalance * 2)))  # Multiply to amplify effect slightly
            # self.logger.debug(f"Orderbook imbalance: Bids={total_bid_volume:.4f}, Asks={total_ask_volume:.4f}, Score={score:.2f}")
            return score

        except (TypeError, ValueError, InvalidOperation, IndexError) as e:
            self.logger.warning(f"Could not calculate orderbook imbalance for {self.symbol}: {e}")
            return 0.0

    def generate_trading_signal(self, current_price: Decimal, orderbook: dict | None = None) -> str:
        """Generates a BUY, SELL, or HOLD signal based on weighted indicator scores.
        Uses the 'active_weight_set' defined in the config.
        """
        if not self.weights:
            self.logger.error(f"Cannot generate signal for {self.symbol}: No weights loaded for active set '{self.active_weight_set_name}'.")
            return "HOLD"
        if current_price is None or current_price.is_nan():
             self.logger.error(f"Cannot generate signal for {self.symbol}: Invalid current price ({current_price}).")
             return "HOLD"

        total_score = Decimal("0.0")
        total_weight = Decimal("0.0")  # Sum of absolute weights for *active* indicators
        active_indicators = self.config.get("indicators", {})
        scores_log = {}  # For logging individual scores

        # --- Calculate scores for active indicators ---
        indicator_checks = {
            "ema_alignment": self._check_ema_alignment,
            "momentum": self._check_momentum,
            "volume_confirmation": self._check_volume_confirmation,
            "stoch_rsi": self._check_stoch_rsi,
            "rsi": self._check_rsi,
            "bollinger_bands": lambda: self._check_bollinger_bands(current_price),
            "vwap": lambda: self._check_vwap(current_price),
            "cci": self._check_cci,
            "wr": self._check_wr,
            "psar": lambda: self._check_psar(current_price),
            "sma_10": lambda: self._check_sma10(current_price),  # Changed from sma10 to sma_10
            "mfi": self._check_mfi,
            "orderbook": lambda: self._check_orderbook_imbalance(orderbook)
        }

        for name, check_func in indicator_checks.items():
            if active_indicators.get(name, False):  # Check if indicator is enabled in config
                weight_val = self.weights.get(name, 0.0)
                # Handle potential None or invalid weight values gracefully
                try:
                     weight = Decimal(str(weight_val)) if weight_val is not None else Decimal(0)
                except (InvalidOperation, ValueError, TypeError):
                     self.logger.warning(f"Invalid weight format '{weight_val}' for indicator '{name}'. Using weight 0.")
                     weight = Decimal(0)

                if weight != 0:  # Only calculate if weighted
                    try:
                        score_float = check_func()  # Get float score from check function
                        # Ensure score is float or can be converted, default to 0 on error
                        try:
                             score = Decimal(str(score_float)) if score_float is not None else Decimal(0)
                        except (InvalidOperation, ValueError, TypeError):
                             self.logger.warning(f"Invalid score type {type(score_float)} from indicator '{name}'. Using score 0.")
                             score = Decimal(0)

                        weighted_score = score * weight
                        total_score += weighted_score
                        total_weight += abs(weight)  # Use absolute weight for normalization denominator
                        scores_log[name] = f"{score:.2f} (w={weight}, ws={weighted_score:.2f})"
                    except Exception as e:
                         self.logger.warning(f"Error calculating score for indicator '{name}' on {self.symbol}: {e}")
                         scores_log[name] = "Error"
                else:
                    scores_log[name] = "Not Weighted"
            else:
                scores_log[name] = "Disabled"

        # --- Normalize Score (Optional but recommended) ---
        normalized_score = Decimal("0.0")
        if total_weight > 0:
            normalized_score = total_score / total_weight
        else:
            self.logger.debug(f"Total weight is zero for {self.symbol}. Cannot normalize score.")

        # --- Determine Signal based on Threshold ---
        threshold_key = "signal_score_threshold"  # Default key
        if self.active_weight_set_name == "scalping":
             threshold_key = "scalping_signal_threshold"

        try:
            threshold = Decimal(str(self.config.get(threshold_key, 1.5)))  # Default if key missing
        except (InvalidOperation, ValueError, TypeError):
             self.logger.warning(f"Invalid threshold format for '{threshold_key}'. Using default 1.5.")
             threshold = Decimal("1.5")

        # Apply threshold to the raw total_score
        final_score_to_use = total_score

        signal = "HOLD"
        if final_score_to_use >= threshold:
            signal = "BUY"
        elif final_score_to_use <= -threshold:
            signal = "SELL"

        # Log the decision process
        price_prec = self.get_price_precision()
        self.logger.info(f"Signal Calc ({self.symbol}): Price={current_price:.{price_prec}f}, Score={final_score_to_use:.3f} (Norm={normalized_score:.3f}), Threshold=+/={threshold}, Signal={NEON_GREEN if signal == 'BUY' else NEON_RED if signal == 'SELL' else NEON_YELLOW}{signal}{RESET}")
        self.logger.debug(f"Indicator Scores: {scores_log}")

        # Update internal simple signal state (optional)
        self.signals = {"BUY": 0, "SELL": 0, "HOLD": 0}
        self.signals[signal] = 1

        return signal

    # --- Risk Management Calculations ---
    def calculate_entry_tp_sl(
        self, entry_price: Decimal, signal: str
    ) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
        """Calculates potential take profit (TP) and initial stop loss (SL) levels
        based on the provided entry price, ATR, and configured multipliers. Uses Decimal precision.
        The initial SL calculated here is primarily used for position sizing and setting the initial protection.
        Returns (entry_price, take_profit, stop_loss), all as Decimal or None.
        """
        if signal not in ["BUY", "SELL"]:
            return entry_price, None, None  # No TP/SL needed for HOLD

        atr_val_float = self.indicator_values.get("ATR")  # Get float ATR value
        if atr_val_float is None or pd.isna(atr_val_float) or atr_val_float <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: ATR is invalid ({atr_val_float}).{RESET}")
            return entry_price, None, None
        if entry_price is None or entry_price.is_nan() or entry_price <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: Provided entry price is invalid ({entry_price}).{RESET}")
            return entry_price, None, None

        try:
            atr = Decimal(str(atr_val_float))  # Convert valid float ATR to Decimal

            # Get multipliers from config, convert to Decimal
            tp_multiple = Decimal(str(self.config.get("take_profit_multiple", 1.0)))
            sl_multiple = Decimal(str(self.config.get("stop_loss_multiple", 1.5)))

            # Get market precision for logging
            price_precision = self.get_price_precision()
            # Get minimum price increment (tick size) for quantization
            min_tick = self.get_min_tick_size()
            if min_tick is None or min_tick <= 0:
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
            min_sl_distance = min_tick  # Minimum distance from entry for SL
            if signal == "BUY" and stop_loss >= entry_price:
                adjusted_sl = (entry_price - min_sl_distance).quantize(min_tick, rounding=ROUND_DOWN)
                self.logger.warning(f"{NEON_YELLOW}BUY signal SL calculation ({stop_loss}) is too close to or above entry ({entry_price}). Adjusting SL down to {adjusted_sl}.{RESET}")
                stop_loss = adjusted_sl
            elif signal == "SELL" and stop_loss <= entry_price:
                adjusted_sl = (entry_price + min_sl_distance).quantize(min_tick, rounding=ROUND_UP)
                self.logger.warning(f"{NEON_YELLOW}SELL signal SL calculation ({stop_loss}) is too close to or below entry ({entry_price}). Adjusting SL up to {adjusted_sl}.{RESET}")
                stop_loss = adjusted_sl

            # Ensure TP is potentially profitable relative to entry (at least one tick away)
            min_tp_distance = min_tick  # Minimum distance from entry for TP
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
            atr_log = f"{atr:.{price_precision + 1}f}" if atr else 'N/A'
            entry_log = f"{entry_price:.{price_precision}f}" if entry_price else 'N/A'

            self.logger.debug(f"Calculated TP/SL for {self.symbol} {signal}: Entry={entry_log}, TP={tp_log}, SL={sl_log}, ATR={atr_log}")
            return entry_price, take_profit, stop_loss

        except (InvalidOperation, ValueError, TypeError, Exception) as e:
            self.logger.error(f"{NEON_RED}Error calculating TP/SL for {self.symbol}: {e}{RESET}", exc_info=True)
            return entry_price, None, None


# --- Trading Logic Helper Functions ---
def fetch_balance(exchange: ccxt.Exchange, currency_code: str, logger: logging.Logger) -> Decimal | None:
    """Fetch the available balance for a specific currency using CCXT with V5 params."""
    lg = logger
    try:
        lg.debug(f"Fetching balance for {currency_code}...")
        params = {}
        # Determine account type for Bybit V5 balance fetch
        # Common types: CONTRACT (for derivatives), SPOT, UNIFIED
        # Let's assume CONTRACT for USDT unless config specifies otherwise
        # A more robust approach would determine based on markets being traded
        account_type = 'UNIFIED'  # Default assumption for USDT derivatives
        # Add logic here if supporting SPOT or UNIFIED accounts based on config/market
        # Example: if any(market_infos[s]['type'] == 'spot' for s in valid_symbols): account_type = 'SPOT' ? (Complex)

        if 'bybit' in exchange.id.lower():
             params['accountType'] = account_type
             # V5 also uses 'coin' parameter for specific currency balance
             params['coin'] = currency_code
             lg.debug(f"Using params for fetch_balance: {params}")

        balance_info = exchange.fetch_balance(params=params)
        # lg.debug(f"Full balance info: {balance_info}") # Can be very verbose

        # Find the specific currency balance
        # Structure might vary slightly; check 'free', 'total', 'info'
        currency_balance = balance_info.get(currency_code)

        if currency_balance:
            # Prioritize 'free' or 'available' balance for trading
            # Bybit V5 often uses 'available' within the currency dict
            available_balance_str = currency_balance.get('free', currency_balance.get('available'))
            if available_balance_str is None:
                 # Fallback to 'total' if free/available not found
                 available_balance_str = currency_balance.get('total')

            if available_balance_str is not None:
                try:
                    balance_decimal = Decimal(str(available_balance_str))
                    lg.info(f"Available balance for {currency_code}: {balance_decimal:.4f}")
                    return balance_decimal
                except (InvalidOperation, ValueError, TypeError) as e:
                    lg.error(f"{NEON_RED}Could not parse balance for {currency_code} from string '{available_balance_str}': {e}{RESET}")
                    return None
            else:
                 lg.warning(f"Could not find 'free', 'available', or 'total' balance fields for {currency_code} in balance response.")
                 return None
        else:
            # Check if balance is nested deeper, e.g., within info['result']['list'] for Bybit V5 CONTRACT
            if 'bybit' in exchange.id.lower() and params.get('accountType') == 'CONTRACT' and 'info' in balance_info:
                balance_list = balance_info.get('info', {}).get('result', {}).get('list', [])
                for item in balance_list:
                    # Check accountType within the list item as well
                    if item.get('accountType') == account_type and isinstance(item.get('coin'), list):
                         for coin_data in item['coin']:
                              if coin_data.get('coin') == currency_code:
                                   # Use 'availableToWithdraw' or 'walletBalance' - 'availableToWithdraw' is safer as it excludes unrealized PnL
                                   balance_str = coin_data.get('availableToWithdraw', coin_data.get('walletBalance'))
                                   if balance_str:
                                       try:
                                           balance_decimal = Decimal(str(balance_str))
                                           lg.info(f"Available balance for {currency_code} (from info.list): {balance_decimal:.4f}")
                                           return balance_decimal
                                       except (InvalidOperation, ValueError, TypeError) as e:
                                           lg.error(f"{NEON_RED}Could not parse balance for {currency_code} from info.list string '{balance_str}': {e}{RESET}")
                                           return None
            # If not found in common structures or nested list
            lg.warning(f"{NEON_YELLOW}Balance data for currency {currency_code} not found in expected structure.{RESET}")
            lg.debug(f"Balance response keys: {balance_info.keys()}")
            return None

    except ccxt.AuthenticationError as e:
        lg.error(f"{NEON_RED}Authentication error fetching balance: {e}{RESET}")
    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error fetching balance: {e}{RESET}")
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}Exchange error fetching balance: {e}{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error fetching balance: {e}{RESET}", exc_info=True)

    return None


def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> dict | None:
    """Fetches and returns market information for a symbol."""
    lg = logger
    try:
        # Reload markets periodically or if info is missing
        reload_interval = 3600  # Reload every hour
        if not hasattr(exchange, 'last_load_markets_timestamp') or time.time() - exchange.last_load_markets_timestamp > reload_interval:
             lg.info(f"Reloading markets (last loaded > {reload_interval}s ago)...")
             exchange.load_markets(reload=True)
             exchange.last_load_markets_timestamp = time.time()

        if symbol not in exchange.markets:
            lg.warning(f"Symbol {symbol} not found in initially loaded markets. Attempting specific load...")
            try:
                 # Attempt to load just this market - might require category hint for Bybit
                 params = {}
                 # Basic category guess for loading (less reliable than having full market info)
                 if 'bybit' in exchange.id.lower():
                      if ':USDT' in symbol or '/USDT' in symbol: params['category'] = 'linear'
                      elif '/USD' in symbol: params['category'] = 'inverse'  # Guess inverse
                      else: params['category'] = 'spot'  # Guess spot

                 exchange.load_markets(symbols=[symbol], params=params)
                 if symbol not in exchange.markets:
                     lg.error(f"{NEON_RED}Symbol {symbol} still not found after specific load attempt.{RESET}")
                     return None
            except Exception as load_err:
                 lg.error(f"{NEON_RED}Failed to load specific market for {symbol}: {load_err}{RESET}")
                 return None

        market = exchange.markets[symbol]
        lg.debug(f"Market info retrieved for {symbol}")  #: {market}") # Too verbose

        # --- Add derived/standardized info for easier use ---
        market['is_contract'] = market.get('contract', False) or market.get('type') in ['swap', 'future']
        market['min_tick_size'] = get_min_tick_size_from_market(market, lg)  # Use helper
        market['price_precision_digits'] = get_price_precision_digits_from_market(market, lg)  # Use helper
        market['amount_precision_digits'] = get_amount_precision_digits_from_market(market, lg)  # Use helper
        market['min_order_amount'] = get_min_order_amount_from_market(market, lg)  # Use helper
        market['min_order_cost'] = get_min_order_cost_from_market(market, lg)  # Use helper

        # Log key derived values
        lg.debug(f"Derived info for {symbol}: TickSize={market['min_tick_size']}, PricePrec={market['price_precision_digits']}, AmtPrec={market['amount_precision_digits']}, MinAmt={market['min_order_amount']}, MinCost={market['min_order_cost']}")

        # Ensure critical derived values are present
        if market['min_tick_size'] is None or market['min_order_amount'] is None:
             lg.error(f"Market info for {symbol} is missing critical derived data (tick size or min amount). Cannot use symbol.")
             return None

        return market

    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error fetching market info for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}Exchange error fetching market info for {symbol}: {e}{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error fetching market info for {symbol}: {e}{RESET}", exc_info=True)
    return None


# --- Market Info Helper Functions ---
def get_min_tick_size_from_market(market: dict, logger: logging.Logger) -> Decimal | None:
    """Extracts minimum price increment (tick size) from CCXT market structure."""
    try:
        # CCXT standard is 'precision': {'price': tick_size} (sometimes as float/int, sometimes string)
        precision_info = market.get('precision', {})
        tick_size_raw = precision_info.get('price')

        if tick_size_raw is None:
             # Fallback: check Bybit specific info if available
             info = market.get('info', {})
             # V5 uses priceFilter.tickSize
             tick_size_raw = info.get('priceFilter', {}).get('tickSize')
             if tick_size_raw is None:
                  # Older V3 style?
                  tick_size_raw = info.get('tickSize')

        if tick_size_raw is not None:
            tick_size_str = str(tick_size_raw)
            # Handle scientific notation or direct decimal string
            tick_size = Decimal(tick_size_str)
            if tick_size > 0:
                return tick_size
            else:
                 logger.warning(f"Parsed tick size is not positive ({tick_size_str} -> {tick_size}) for {market.get('symbol')}. Returning None.")
                 return None
        else:
             logger.warning(f"Could not find price precision/tick size for {market.get('symbol')} in market data: {precision_info} / {market.get('info')}")
             # Attempt to infer from price precision digits if available
             price_digits = get_price_precision_digits_from_market(market, logger)
             if price_digits is not None and price_digits >= 0:  # Check digits >= 0
                 return Decimal('1') / (Decimal('10') ** price_digits)
             return None  # Cannot determine

    except (InvalidOperation, ValueError, TypeError, KeyError) as e:
        logger.error(f"Error determining min tick size for {market.get('symbol')}: {e}")
        return None


def get_price_precision_digits_from_market(market: dict, logger: logging.Logger) -> int | None:
    """Extracts number of decimal places for price from CCXT market structure."""
    try:
        # CCXT sometimes uses 'precision': {'price': tick_size} or {'price': digits}
        precision_info = market.get('precision', {})
        price_prec_raw = precision_info.get('price')

        # If precision.price is directly available, try to determine if it's digits or tick size
        if price_prec_raw is not None:
            try:
                price_prec_dec = Decimal(str(price_prec_raw))
                # Heuristic: If it's < 1, it's likely a tick size. If >= 1 and integer, likely digits.
                if price_prec_dec < 1 and price_prec_dec > 0:  # Likely tick size
                    return abs(price_prec_dec.normalize().as_tuple().exponent)
                elif price_prec_dec >= 1 and price_prec_dec.to_integral_value() == price_prec_dec:  # Likely integer digits
                    return int(price_prec_dec)
                elif price_prec_dec == 0:  # Precision 0 usually means integer price (0 digits)
                    return 0
                # else: fall through to tick size inference
            except (InvalidOperation, ValueError, TypeError):
                pass  # Fall through if conversion fails

        # If price precision wasn't digits or couldn't parse, try inferring from tick size
        min_tick = get_min_tick_size_from_market(market, logger)
        if min_tick is not None and min_tick > 0:
            return abs(min_tick.normalize().as_tuple().exponent)

        # Final fallback if nothing worked
        logger.warning(f"Could not determine price precision digits for {market.get('symbol')}. Using default 8.")
        return 8

    except Exception as e:
        logger.error(f"Error determining price precision digits for {market.get('symbol')}: {e}")
        return 8  # Default


def get_amount_precision_digits_from_market(market: dict, logger: logging.Logger) -> int | None:
    """Extracts number of decimal places for amount/quantity from CCXT market structure."""
    try:
        # CCXT standard is 'precision': {'amount': digits or step_size}
        precision_info = market.get('precision', {})
        amount_prec_raw = precision_info.get('amount')

        # If precision.amount is directly available, try to interpret it
        if amount_prec_raw is not None:
            try:
                amount_prec_dec = Decimal(str(amount_prec_raw))
                # Heuristic: If < 1, likely step size. If >= 1 and integer, likely digits.
                if amount_prec_dec < 1 and amount_prec_dec > 0:  # Likely step size
                    return abs(amount_prec_dec.normalize().as_tuple().exponent)
                elif amount_prec_dec >= 1 and amount_prec_dec.to_integral_value() == amount_prec_dec:  # Likely integer digits
                    return int(amount_prec_dec)
                elif amount_prec_dec == 0:  # Precision 0 means integer amount
                    return 0
                # else: fall through to Bybit specific / default
            except (InvalidOperation, ValueError, TypeError):
                 pass  # Fall through if conversion fails

        # If not found or ambiguous, check Bybit V5 specific in 'info' - uses lotSizeFilter.qtyStep
        info = market.get('info', {})
        step_size_str = info.get('lotSizeFilter', {}).get('qtyStep')
        if step_size_str:
            try:
                 step_size = Decimal(step_size_str)
                 if step_size > 0:
                     # If step size is 1, precision is 0
                     if step_size == 1: return 0
                     # Otherwise, digits = exponent of step size
                     return abs(step_size.normalize().as_tuple().exponent)
                 # else: fall through if step size invalid
            except (InvalidOperation, ValueError, TypeError):
                pass  # Fall through if conversion fails

        # Final fallback
        logger.warning(f"Could not determine amount precision digits for {market.get('symbol')}. Using default 3.")
        return 3

    except Exception as e:
        logger.error(f"Error determining amount precision digits for {market.get('symbol')}: {e}")
        return 3  # Default


def get_min_order_amount_from_market(market: dict, logger: logging.Logger) -> Decimal | None:
    """Extracts minimum order amount (in base currency) from CCXT market structure."""
    try:
        limits_info = market.get('limits', {})
        amount_info = limits_info.get('amount', {})
        min_amount_raw = amount_info.get('min')

        if min_amount_raw is None:
            # Bybit V5 specific in 'info' - uses lotSizeFilter.minOrderQty
            info = market.get('info', {})
            min_amount_raw = info.get('lotSizeFilter', {}).get('minOrderQty')
            if min_amount_raw is None:
                 # Spot market V3?
                 min_amount_raw = info.get('minTradeQuantity')

        if min_amount_raw is not None:
            min_amount = Decimal(str(min_amount_raw))
            # Return value even if zero, let sizing logic handle comparison
            return min_amount
        else:
            logger.warning(f"Could not find minimum order amount for {market.get('symbol')}. Returning small default.")
            return Decimal('1e-8')  # Default if not found

    except (InvalidOperation, ValueError, TypeError, KeyError) as e:
        logger.error(f"Error determining minimum order amount for {market.get('symbol')}: {e}")
        return Decimal('1e-8')  # Default


def get_min_order_cost_from_market(market: dict, logger: logging.Logger) -> Decimal | None:
    """Extracts minimum order cost (in quote currency) from CCXT market structure."""
    try:
        # CCXT standard is 'limits': {'cost': {'min': value}}
        limits_info = market.get('limits', {})
        cost_info = limits_info.get('cost', {})
        min_cost_raw = cost_info.get('min')

        if min_cost_raw is None:
             # Bybit V5 specific in 'info'
             info = market.get('info', {})
             # Bybit Spot V3 uses minOrderAmount (in quote currency)
             min_cost_raw = info.get('minOrderAmount', info.get('minTradeAmount'))  # Check both V3 names
             if min_cost_raw is None:
                  # Derivatives might not explicitly list min cost, rely on min amount * price.
                  logger.debug(f"Minimum order cost not explicitly found for {market.get('symbol')}. Will rely on min amount.")
                  return None  # Indicate not found

        if min_cost_raw is not None:
            min_cost = Decimal(str(min_cost_raw))
            # Return value even if zero
            return min_cost
        else:
            # This path shouldn't be reached if logic above is correct
            logger.warning(f"Could not find minimum order cost for {market.get('symbol')}. Returning None.")
            return None  # Indicate not found

    except (InvalidOperation, ValueError, TypeError, KeyError) as e:
        logger.error(f"Error determining minimum order cost for {market.get('symbol')}: {e}")
        return None


def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float,
    initial_stop_loss_price: Decimal,
    entry_price: Decimal,
    market_info: dict,
    exchange: ccxt.Exchange,  # Keep exchange object for potential future use (e.g., formatting)
    logger: logging.Logger
) -> Decimal | None:
    """Calculates the position size based on account balance, risk percentage,
    and distance to initial stop loss. Uses Decimal for precision.
    """
    lg = logger
    symbol = market_info.get('symbol', 'N/A')

    # --- Input Validation ---
    if balance is None or balance <= 0:
        lg.error(f"Cannot calculate position size for {symbol}: Invalid balance ({balance}).")
        return None
    if not (0 < risk_per_trade < 1):
        lg.error(f"Cannot calculate position size for {symbol}: Invalid risk_per_trade ({risk_per_trade}). Must be between 0 and 1.")
        return None
    if initial_stop_loss_price is None or initial_stop_loss_price <= 0:
        lg.error(f"Cannot calculate position size for {symbol}: Invalid initial_stop_loss_price ({initial_stop_loss_price}).")
        return None
    if entry_price is None or entry_price <= 0:
        lg.error(f"Cannot calculate position size for {symbol}: Invalid entry_price ({entry_price}).")
        return None
    if entry_price == initial_stop_loss_price:
         lg.error(f"Cannot calculate position size for {symbol}: Entry price and SL price are identical.")
         return None

    try:
        # --- Get Market Parameters (using derived values) ---
        contract_size_str = str(market_info.get('contractSize', '1'))  # Default to 1 for spot/linear
        contract_size = Decimal(contract_size_str)
        amount_precision_digits = market_info.get('amount_precision_digits')  # Use derived value
        min_order_amount = market_info.get('min_order_amount')  # Use derived value
        min_order_cost = market_info.get('min_order_cost')  # Use derived value (can be None)

        if amount_precision_digits is None or min_order_amount is None:
             lg.error(f"Cannot calculate position size for {symbol}: Missing critical market precision/limit data (amount digits or min amount).")
             return None
        min_amount_step = Decimal('1') / (Decimal('10') ** amount_precision_digits)

        # --- Calculate Risk Amount ---
        risk_amount = balance * Decimal(str(risk_per_trade))

        # --- Calculate Distance to Stop Loss ---
        stop_loss_distance = abs(entry_price - initial_stop_loss_price)
        if stop_loss_distance <= 0:  # Should be caught by earlier check, but defensive
            lg.error(f"Cannot calculate position size for {symbol}: Stop loss distance is zero or negative (Entry: {entry_price}, SL: {initial_stop_loss_price}).")
            return None

        # --- Calculate Position Size (Raw) ---
        position_size_raw = Decimal(0)
        risk_per_unit_or_contract = Decimal(0)  # For logging

        if market_info.get('inverse', False):
             # Inverse: Size (Contracts) = Risk Amount (Quote) / Risk per Contract (Quote)
             # Risk per Contract (Quote) ~= Contract Size (Base/Contract) * SL Distance (Quote/Base) * (1/EntryPrice) <- Simplified
             # A more accurate formula involves contract value at entry vs SL
             # Let's use a reasonable approximation: Risk per contract in quote = contract_size * sl_distance / entry_price
             # WARNING: This is still an approximation for inverse contracts!
             if entry_price > 0:
                 risk_per_unit_or_contract = (contract_size * stop_loss_distance) / entry_price
                 if risk_per_unit_or_contract > 0:
                      position_size_raw = risk_amount / risk_per_unit_or_contract
                 else:
                      lg.error(f"Cannot calculate position size for {symbol} (Inverse): Invalid risk per contract ({risk_per_unit_or_contract}).")
                      return None
             else:
                  lg.error(f"Cannot calculate position size for {symbol} (Inverse): Invalid entry price ({entry_price}).")
                  return None
        else:  # Linear or Spot
            # Size (Base Units) = Risk Amount (Quote) / (SL Distance (Quote/Base) * Contract Size (Base/Contract))
            risk_per_unit_or_contract = stop_loss_distance * contract_size
            if risk_per_unit_or_contract <= 0:
                lg.error(f"Cannot calculate position size for {symbol} (Linear/Spot): Invalid risk per unit ({risk_per_unit_or_contract}).")
                return None
            position_size_raw = risk_amount / risk_per_unit_or_contract

        lg.debug(f"Position Size Calc ({symbol}): Balance={balance:.2f}, Risk%={risk_per_trade:.2%}, RiskAmt={risk_amount:.2f}")
        lg.debug(f"  Entry={entry_price}, SL={initial_stop_loss_price}, SLDist={stop_loss_distance}")
        lg.debug(f"  ContractSize={contract_size}, RiskPerUnit/Contract={risk_per_unit_or_contract:.8f}")
        lg.debug(f"  Raw Position Size={position_size_raw:.8f}")

        # --- Apply Precision and Limits ---

        # 1. Quantize to Amount Step Size (Round Down - safer not to exceed risk)
        quantized_size = (position_size_raw / min_amount_step).quantize(Decimal("1"), rounding=ROUND_DOWN) * min_amount_step
        lg.debug(f"  Quantized Size (Step={min_amount_step}): {quantized_size:.{amount_precision_digits}f}")

        # 2. Check Minimum Order Amount
        if quantized_size < min_order_amount:
            lg.warning(f"{NEON_YELLOW}Calculated position size {quantized_size} is below minimum order amount {min_order_amount} for {symbol}. Cannot place trade.{RESET}")
            return None

        # 3. Check Minimum Order Cost (if applicable and calculable)
        if min_order_cost is not None and min_order_cost > 0:
            # Estimate cost more accurately based on type
            estimated_cost = Decimal(0)
            if market_info.get('inverse', False):
                 # Cost (Quote) = Size (Contracts) * Contract Size (Base/Contract) * Entry Price (Quote/Base) <- Incorrect for Inverse value
                 # Cost (Quote) = Size (Contracts) * Contract Value (Quote / Contract) -> Use contract_size as value proxy? Risky.
                 # Let's use Size * Contract Size / Entry Price (rough estimate of base value) * Entry Price = Size * Contract Size
                 # This estimates the *base* currency value. We need quote.
                 # Cost approximation for inverse: Size (Contracts) * ContractSize (Value in Quote, e.g. 100 USD for BTCUSD)
                 # If contractSize is in Base (e.g., 1 BTC), Cost = Size * ContractSize * Price
                 # Assuming Bybit `contractSize` is in BASE for inverse (e.g., 1 BTC for BTCUSD)
                 estimated_cost = quantized_size * contract_size * entry_price  # Base Value * Price -> Quote Value
            else:  # Linear / Spot
                 estimated_cost = quantized_size * entry_price  # Base Amount * Price -> Quote Value

            if estimated_cost < min_order_cost:
                 lg.warning(f"{NEON_YELLOW}Estimated cost {estimated_cost:.4f} for size {quantized_size} is below minimum order cost {min_order_cost} for {symbol}. Cannot place trade.{RESET}")
                 return None
            lg.debug(f"  Estimated Cost={estimated_cost:.4f} (Min Cost Req={min_order_cost})")

        # 4. Final Check: Ensure size is positive
        if quantized_size <= 0:
            lg.error(f"Final calculated position size is zero or negative ({quantized_size}) for {symbol} after adjustments.")
            return None

        lg.info(f"Calculated Position Size for {symbol}: {quantized_size:.{amount_precision_digits}f}")
        return quantized_size

    except (InvalidOperation, ValueError, TypeError, KeyError, ZeroDivisionError) as e:
        lg.error(f"{NEON_RED}Error calculating position size for {symbol}: {e}{RESET}", exc_info=True)
        return None


def _manual_amount_rounding(amount: Decimal, market_info: dict, lg: logging.Logger) -> Decimal:
    """Manually rounds the amount based on amount precision digits."""
    symbol = market_info.get('symbol', 'N/A')
    try:
        amount_precision_digits = market_info.get('amount_precision_digits')
        if amount_precision_digits is None or amount_precision_digits < 0:  # Ensure non-negative digits
            lg.warning(f"Amount precision digits ({amount_precision_digits}) invalid for {symbol}, cannot perform manual rounding. Returning original.")
            return amount

        # Create the quantizer, e.g., Decimal('0.001') for 3 digits
        quantizer = Decimal('1e-' + str(amount_precision_digits))
        rounded_amount = amount.quantize(quantizer, rounding=ROUND_DOWN)
        # lg.debug(f"Manual rounding for {symbol}: Original={amount}, Digits={amount_precision_digits}, Quantizer={quantizer}, Rounded={rounded_amount}")
        return rounded_amount
    except Exception as e:
        lg.error(f"Error during manual amount rounding for {symbol}: {e}. Returning original amount.")
        return amount


def get_open_position(
    exchange: ccxt.Exchange,
    symbol: str,
    logger: logging.Logger,
    market_info: dict = None,
    position_mode: str = "One-Way"  # Add position mode from config
) -> dict | None:
    """Fetches the open position for a specific symbol using CCXT, handling V5 params.
    Returns the enhanced position dictionary if found and size > 0, otherwise None.
    Handles both One-Way and Hedge mode position fetching logic for Bybit V5.
    """
    lg = logger
    try:
        lg.debug(f"Fetching open position for {symbol} (Mode: {position_mode})...")
        params = {'symbol': market_info.get('id', symbol)} if market_info else {'symbol': symbol}
        category = None
        if 'bybit' in exchange.id.lower() and market_info:
            category = _determine_category(market_info)
            if category:
                params['category'] = category
                # For Hedge Mode, Bybit V5 might need settleCoin? Test this.
                # if position_mode == "Hedge": params['settleCoin'] = market_info.get('settle')
            lg.debug(f"Using params for fetch_positions ({symbol}): {params}")

        # --- Fetch Positions ---
        # Use fetch_positions(symbols=[symbol]) for potentially better performance/filtering
        positions = exchange.fetch_positions(symbols=[symbol], params=params)
        # lg.debug(f"Raw positions fetched for {symbol}: {positions}") # Verbose

        open_position = None

        if position_mode == "One-Way":
            # In One-Way mode, there should be at most one position entry per symbol.
            if positions:
                # Find the position for the exact symbol (CCXT might return variations)
                for pos in positions:
                    # Check both 'symbol' and 'info'.'symbol' for robustness
                    pos_symbol = pos.get('symbol')
                    info_symbol = pos.get('info', {}).get('symbol')
                    market_id = market_info.get('id') if market_info else None  # Exchange market ID

                    # Match criteria:
                    # 1. CCXT symbol matches input symbol
                    # 2. OR info.symbol matches market_id (if available)
                    # 3. OR info.symbol matches input symbol (less precise)
                    is_match = (pos_symbol == symbol) or \
                               (market_id and info_symbol == market_id) or \
                               (info_symbol == symbol)

                    if is_match:
                        pos_size_str = pos.get('contracts', pos.get('info', {}).get('size', '0'))
                        try:
                            pos_size = Decimal(str(pos_size_str))
                            # Use a threshold slightly larger than zero to consider it open
                            if abs(pos_size) > Decimal('1e-9'):
                                open_position = pos
                                break  # Found the one-way position
                        except (InvalidOperation, ValueError, TypeError):
                            lg.warning(f"Could not parse position size '{pos_size_str}' for {symbol}. Skipping this entry.")
                            continue
            # else: lg.debug(f"No position entries returned for {symbol} in One-Way mode check.")

        elif position_mode == "Hedge":
            # In Hedge mode, there can be two entries per symbol (Buy and Sell side).
            # We need to find the one with a non-zero size.
            # Bybit V5 returns 'positionIdx': 1 for Buy side, 2 for Sell side in Hedge mode.
            long_pos = None
            short_pos = None
            for pos in positions:
                 pos_symbol = pos.get('symbol')
                 info_symbol = pos.get('info', {}).get('symbol')
                 market_id = market_info.get('id') if market_info else None
                 is_match = (pos_symbol == symbol) or \
                            (market_id and info_symbol == market_id) or \
                            (info_symbol == symbol)

                 if is_match:
                     pos_size_str = pos.get('contracts', pos.get('info', {}).get('size', '0'))
                     # Determine side based on 'side' field OR positionIdx
                     pos_side_raw = pos.get('side', pos.get('info', {}).get('side', 'Unknown')).lower()
                     pos_idx = int(pos.get('info', {}).get('positionIdx', 0))  # 0=One-Way, 1=Buy Hedge, 2=Sell Hedge

                     # Infer side if missing or 'none'/'unknown'
                     pos_side = 'unknown'
                     if pos_side_raw in ['long', 'buy']: pos_side = 'long'
                     elif pos_side_raw in ['short', 'sell']: pos_side = 'short'
                     elif pos_idx == 1: pos_side = 'long'
                     elif pos_idx == 2: pos_side = 'short'

                     try:
                         pos_size = Decimal(str(pos_size_str))
                         if abs(pos_size) > Decimal('1e-9'):
                             if pos_side == 'long':
                                 long_pos = pos
                             elif pos_side == 'short':
                                 short_pos = pos
                             # else: lg.debug(f"Position entry found but side unknown for {symbol} (Idx={pos_idx}, Side={pos_side_raw})")
                     except (InvalidOperation, ValueError, TypeError):
                         lg.warning(f"Could not parse position size '{pos_size_str}' for {symbol} (Hedge Mode). Skipping.")
                         continue

            # Prefer the entry that has a non-zero size.
            if long_pos and short_pos:
                 lg.warning(f"Found BOTH Long and Short active positions for {symbol} in Hedge Mode. This is unusual. Prioritizing LONG.")
                 open_position = long_pos  # Prioritize one if both somehow exist
            elif long_pos:
                 open_position = long_pos
                 lg.debug(f"Found Hedge Mode LONG position for {symbol}.")
            elif short_pos:
                 open_position = short_pos
                 lg.debug(f"Found Hedge Mode SHORT position for {symbol}.")
            # else: lg.debug(f"No active Buy or Sell side position found for {symbol} in Hedge mode check.")

        # --- Process and Enhance Found Position ---
        if open_position:
            lg.debug(f"Found potentially open position object for {symbol}. Enhancing...")
            # Enhance with Decimal types and standardized fields
            enhanced_pos = open_position.copy()
            info_dict = enhanced_pos.get('info', {})

            # --- Standardize Side ---
            side_raw = enhanced_pos.get('side', info_dict.get('side', '')).lower()
            pos_idx = int(info_dict.get('positionIdx', 0))
            if side_raw not in ['long', 'short']:
                if pos_idx == 1: side_raw = 'long'
                elif pos_idx == 2: side_raw = 'short'
                else:  # Try inferring from size sign (less reliable for V5)
                    size_str = enhanced_pos.get('contracts', info_dict.get('size'))
                    if size_str:
                        try:
                            size_dec = Decimal(str(size_str))
                            if size_dec > 0: side_raw = 'long'
                            elif size_dec < 0: side_raw = 'short'
                            else: side_raw = 'unknown'  # Size is zero
                        except: side_raw = 'unknown'
                    else: side_raw = 'unknown'
            # Ensure consistent 'long' or 'short'
            enhanced_pos['side'] = 'long' if side_raw in ['long', 'buy'] else ('short' if side_raw in ['short', 'sell'] else 'unknown')

            # --- Entry Price (Decimal) ---
            try:
                # Prefer info.avgPrice for Bybit V5 accuracy
                entry_price_str = info_dict.get('avgPrice', enhanced_pos.get('entryPrice'))
                enhanced_pos['entryPriceDecimal'] = Decimal(str(entry_price_str)) if entry_price_str and str(entry_price_str) != '0' else None
            except (InvalidOperation, ValueError, TypeError):
                lg.warning(f"Could not parse entry price for {symbol}: {entry_price_str}")
                enhanced_pos['entryPriceDecimal'] = None

            # --- Position Size / Contracts (Decimal) ---
            try:
                # Prefer info.size for Bybit V5 accuracy
                contracts_str = info_dict.get('size', enhanced_pos.get('contracts'))
                if contracts_str:
                    size_dec = Decimal(str(contracts_str))
                    # Bybit V5 'size' is positive. Ensure sign matches inferred side.
                    if enhanced_pos['side'] == 'long': enhanced_pos['contractsDecimal'] = abs(size_dec)
                    elif enhanced_pos['side'] == 'short': enhanced_pos['contractsDecimal'] = -abs(size_dec)
                    else: enhanced_pos['contractsDecimal'] = Decimal(0)  # Unknown side means size 0
                else:
                     enhanced_pos['contractsDecimal'] = Decimal(0)
            except (InvalidOperation, ValueError, TypeError):
                lg.warning(f"Could not parse contracts/size for {symbol}: {contracts_str}")
                enhanced_pos['contractsDecimal'] = None

            # --- Stop Loss Price (Decimal) ---
            try:
                # Check info.stopLoss first
                sl_price_str = info_dict.get('stopLoss', enhanced_pos.get('stopLossPrice'))
                if sl_price_str and str(sl_price_str).strip() and str(sl_price_str) != "0":
                    enhanced_pos['stopLossPriceDecimal'] = Decimal(str(sl_price_str))
                else:
                    enhanced_pos['stopLossPriceDecimal'] = None
            except (InvalidOperation, ValueError, TypeError):
                lg.warning(f"Could not parse stop loss price for {symbol}: {sl_price_str}")
                enhanced_pos['stopLossPriceDecimal'] = None

            # --- Take Profit Price (Decimal) ---
            try:
                # Check info.takeProfit first
                tp_price_str = info_dict.get('takeProfit', enhanced_pos.get('takeProfitPrice'))
                if tp_price_str and str(tp_price_str).strip() and str(tp_price_str) != "0":
                     enhanced_pos['takeProfitPriceDecimal'] = Decimal(str(tp_price_str))
                else:
                     enhanced_pos['takeProfitPriceDecimal'] = None
            except (InvalidOperation, ValueError, TypeError):
                lg.warning(f"Could not parse take profit price for {symbol}: {tp_price_str}")
                enhanced_pos['takeProfitPriceDecimal'] = None

            # --- Trailing Stop Distance (Decimal) ---
            try:
                 tsl_dist_str = info_dict.get('trailingStop')
                 if tsl_dist_str and str(tsl_dist_str).strip() and str(tsl_dist_str) != "0":
                      enhanced_pos['trailingStopLossDistanceDecimal'] = Decimal(str(tsl_dist_str))
                 else:
                      enhanced_pos['trailingStopLossDistanceDecimal'] = Decimal(0)  # Use 0 for inactive
            except (InvalidOperation, ValueError, TypeError):
                 lg.warning(f"Could not parse trailing stop distance for {symbol}: {tsl_dist_str}")
                 enhanced_pos['trailingStopLossDistanceDecimal'] = Decimal(0)

            # --- TSL Activation Price (Decimal) ---
            try:
                 tsl_act_str = info_dict.get('activePrice')  # V5 field for TSL activation
                 if tsl_act_str and str(tsl_act_str).strip() and str(tsl_act_str) != "0":
                      enhanced_pos['tslActivationPriceDecimal'] = Decimal(str(tsl_act_str))
                 else:
                      enhanced_pos['tslActivationPriceDecimal'] = Decimal(0)  # Use 0 for inactive/immediate
            except (InvalidOperation, ValueError, TypeError):
                 lg.warning(f"Could not parse TSL activation price for {symbol}: {tsl_act_str}")
                 enhanced_pos['tslActivationPriceDecimal'] = Decimal(0)

            # --- Final Validation ---
            # Ensure the enhanced position still looks valid (has side, size, entry)
            if enhanced_pos['side'] in ['long', 'short'] and \
               enhanced_pos['contractsDecimal'] is not None and \
               abs(enhanced_pos['contractsDecimal']) > Decimal('1e-9') and \
               enhanced_pos['entryPriceDecimal'] is not None and \
               enhanced_pos['entryPriceDecimal'] > 0:
                # Log using enhanced values
                side_log = enhanced_pos['side']
                size_log = enhanced_pos['contractsDecimal']
                entry_log = enhanced_pos['entryPriceDecimal']
                sl_log = enhanced_pos['stopLossPriceDecimal']
                tp_log = enhanced_pos['takeProfitPriceDecimal']
                tsl_dist_log = enhanced_pos['trailingStopLossDistanceDecimal']
                tsl_act_log = enhanced_pos['tslActivationPriceDecimal']
                price_prec = market_info.get('price_precision_digits', 6) if market_info else 6

                sl_str = f"{sl_log:.{price_prec}f}" if sl_log else 'N/A'
                tp_str = f"{tp_log:.{price_prec}f}" if tp_log else 'N/A'
                tsl_act = tsl_dist_log > 0
                tsl_dist_str = f"{tsl_dist_log:.{price_prec}f}" if tsl_act else 'N/A'
                tsl_act_str = f"{tsl_act_log:.{price_prec}f}" if tsl_act and tsl_act_log > 0 else ('Immediate' if tsl_act else 'N/A')

                lg.info(f"Confirmed open {side_log.upper()} position for {symbol}. Size: {size_log}, Entry: {entry_log:.{price_prec}f}, SL: {sl_str}, TP: {tp_str}, TSL: {tsl_act} (Dist={tsl_dist_str}, Act={tsl_act_str})")
                return enhanced_pos
            else:
                lg.debug(f"Position object found for {symbol} but failed validation after enhancement (Size={enhanced_pos.get('contractsDecimal')}, Entry={enhanced_pos.get('entryPriceDecimal')}, Side={enhanced_pos.get('side')}). Treating as closed.")
                return None

        else:
            lg.info(f"No open position found for {symbol}.")
            return None

    except ccxt.AuthenticationError as e:
        lg.error(f"{NEON_RED}Authentication error fetching position for {symbol}: {e}{RESET}")
    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error fetching position for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        # Handle specific errors like "position not found" gracefully
        err_str = str(e).lower()
        bybit_code = getattr(e, 'code', None)  # Bybit specific error code
        # Bybit V5: 110021, 110025, 110043 often mean no position or contract not found
        # Bybit V3?: 30033 might mean position not found
        no_pos_codes = [110021, 110025, 110043, 30033]
        if "position not found" in err_str or \
           (bybit_code and bybit_code in no_pos_codes) or \
           "contract name not exist" in err_str or \
           "symbol param invalid" in err_str:  # Add check for invalid symbol param
            lg.info(f"No open position found for {symbol} (Exchange confirmed: {e}).")
            return None
        else:
            lg.error(f"{NEON_RED}Exchange error fetching position for {symbol}: {e} (Code: {bybit_code}){RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error fetching position for {symbol}: {e}{RESET}", exc_info=True)

    return None


def set_leverage_ccxt(
    exchange: ccxt.Exchange,
    symbol: str,
    leverage: int,
    market_info: dict,
    logger: logging.Logger
) -> bool:
    """Sets leverage for a symbol using CCXT with V5 params."""
    lg = logger
    if not market_info.get('is_contract'):
        lg.info(f"Leverage setting skipped for {symbol} (not a contract market).")
        return True  # Not applicable, so return True

    if leverage <= 0:
        lg.warning(f"Leverage setting skipped for {symbol}: Invalid leverage value ({leverage}).")
        return False

    # Use max leverage from market info if available
    max_leverage_raw = market_info.get('limits', {}).get('leverage', {}).get('max')
    if max_leverage_raw is not None:
         try:
             max_leverage = int(float(str(max_leverage_raw)))  # Handle float/string
             if leverage > max_leverage:
                 lg.warning(f"{NEON_YELLOW}Requested leverage {leverage}x for {symbol} exceeds max allowed {max_leverage}x. Setting to max.{RESET}")
                 leverage = max_leverage  # Use integer part of max leverage
         except (ValueError, TypeError):
              lg.warning(f"Could not parse max leverage '{max_leverage_raw}'. Proceeding with requested leverage {leverage}x.")
    else:
         lg.debug(f"Max leverage limit not found in market info for {symbol}.")

    try:
        lg.info(f"Attempting to set leverage for {symbol} to {leverage}x...")
        params = {}
        if 'bybit' in exchange.id.lower() and market_info:
            category = _determine_category(market_info)
            if category:
                params['category'] = category
                # Bybit V5 requires buyLeverage and sellLeverage, must be strings
                params['buyLeverage'] = str(float(leverage))  # Ensure string format, allow float for API
                params['sellLeverage'] = str(float(leverage))
                lg.debug(f"Using params for set_leverage ({symbol}): {params}")

        response = exchange.set_leverage(leverage, symbol=symbol, params=params)
        lg.debug(f"Set leverage response for {symbol}: {response}")  # Can be verbose

        # --- Verification ---
        # Bybit V5 response might be empty on success (retCode=0 in underlying call)
        # Check if an exception occurred. If not, assume success.
        # More robust check could involve fetching position info again, but adds latency.
        lg.info(f"{NEON_GREEN}Successfully set leverage for {symbol} to {leverage}x (API call successful).{RESET}")
        return True

    except ccxt.AuthenticationError as e:
        lg.error(f"{NEON_RED}Authentication error setting leverage for {symbol}: {e}{RESET}")
    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error setting leverage for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        # Handle common errors like "leverage not modified"
        err_str = str(e).lower()
        bybit_code = getattr(e, 'code', None)
        # Bybit V5: 110045 = leverage not modified (note: previous thought had 110044, check docs/real response)
        # Bybit V3: 30036 = leverage not modified
        if "leverage not modified" in err_str or bybit_code in [110045, 30036]:
            lg.info(f"Leverage for {symbol} already set to {leverage}x.")
            return True  # Already set, consider it success
        elif "set margin mode" in err_str or "switch margin mode" in err_str or bybit_code == 110028:
             lg.error(f"{NEON_RED}Exchange error setting leverage for {symbol}: {e}. >> Possible conflict with Margin Mode (Isolated/Cross). Ensure account setting matches bot expectation.{RESET}")
        elif "risk limit" in err_str or bybit_code == 110044:
             lg.error(f"{NEON_RED}Exchange error setting leverage for {symbol}: {e}. >> Leverage may exceed risk limit tier. Check Bybit settings.{RESET}")
        else:
            lg.error(f"{NEON_RED}Exchange error setting leverage for {symbol}: {e} (Code: {bybit_code}){RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error setting leverage for {symbol}: {e}{RESET}", exc_info=True)

    return False


def place_trade(
    exchange: ccxt.Exchange,
    symbol: str,
    trade_signal: str,  # "BUY" or "SELL"
    position_size: Decimal,  # Positive Decimal value
    market_info: dict,
    logger: logging.Logger,
    reduce_only: bool = False,
    order_type: str = 'market'  # Default to market orders
) -> dict | None:
    """Places a trade order using CCXT with V5 params and robust error handling."""
    lg = logger
    side = 'buy' if trade_signal == "BUY" else 'sell'
    order_tag = "Open" if not reduce_only else "Close"

    if position_size <= 0:
        lg.error(f"Trade placement aborted for {symbol}: Invalid position size ({position_size}).")
        return None

    try:
        # --- Format Amount ---
        # Use manual rounding first based on derived precision for safety
        amount_dec = _manual_amount_rounding(position_size, market_info, lg)
        if amount_dec != position_size:
            lg.info(f"Amount adjusted by manual rounding from {position_size} to {amount_dec}")

        # Use CCXT's formatting as a final check (optional, manual might be safer)
        # try:
        #      amount_str_ccxt = exchange.amount_to_precision(symbol, float(amount_dec))
        #      amount_dec_ccxt = Decimal(amount_str_ccxt)
        #      if amount_dec_ccxt != amount_dec:
        #           lg.warning(f"CCXT amount formatting changed value from {amount_dec} to {amount_dec_ccxt}. Using CCXT value.")
        #           amount_dec = amount_dec_ccxt
        # except Exception as fmt_err:
        #      lg.warning(f"exchange.amount_to_precision failed ({fmt_err}). Using manually rounded amount for {symbol}.")

        # Final check on adjusted amount
        min_order_amount = market_info.get('min_order_amount')  # Use derived value
        if min_order_amount is None:  # Should not happen if market_info validation passed
             lg.error(f"{NEON_RED}{order_tag} Trade Aborted ({symbol} {side}): Minimum order amount not available in market info.{RESET}")
             return None

        if amount_dec < min_order_amount:
             lg.error(f"{NEON_RED}{order_tag} Trade Aborted ({symbol} {side}): Final amount {amount_dec} is less than minimum required {min_order_amount}. Original requested: {position_size}{RESET}")
             return None
        if amount_dec <= 0:
             lg.error(f"{NEON_RED}{order_tag} Trade Aborted ({symbol} {side}): Final amount {amount_dec} is zero or negative.{RESET}")
             return None

        amount_final = float(amount_dec)  # CCXT create_order usually expects float

        # --- Prepare Order Params ---
        params = {
            'timeInForce': 'ImmediateOrCancel' if order_type.lower() == 'market' else 'GoodTillCancel',  # IOC for Market, GTC for Limit
            'reduceOnly': reduce_only,
        }
        position_mode = CONFIG.get("position_mode", "One-Way")  # Get mode from global config
        if position_mode == "Hedge":
             # For Hedge mode closing orders, need to know if closing Buy or Sell side
             if reduce_only:
                  # Sell closes Buy(1), Buy closes Sell(2)
                  params['positionIdx'] = 1 if side == 'sell' else 2
             else:  # Opening hedge trade
                  # Buy opens Buy(1), Sell opens Sell(2)
                  params['positionIdx'] = 1 if side == 'buy' else 2
        else:  # One-Way mode
             params['positionIdx'] = 0  # Explicitly set for V5 One-Way

        if 'bybit' in exchange.id.lower() and market_info:
             category = _determine_category(market_info)
             if category:
                 params['category'] = category
             # Add orderLinkID for tracking?
             # params['orderLinkId'] = f'bot_{symbol.replace("/", "")}_{int(time.time() * 1000)}' # Example ID

        lg.info(f"Placing {order_tag} {side.upper()} {order_type.upper()} order for {symbol} | Amount: {amount_final} | Params: {params}")

        # --- Create Order ---
        order = None
        try:
            # Use unified create_order method
            order = exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount_final,
                price=None,  # None for market orders
                params=params
            )

            lg.info(f"{NEON_GREEN}{order_tag} Order placed successfully for {symbol}! Order ID: {order.get('id', 'N/A')}{RESET}")
            lg.debug(f"Order details: {order}")  # Verbose
            return order

        except ccxt.InsufficientFunds as e:
            lg.error(f"{NEON_RED}{order_tag} Order FAILED ({symbol} {side}): Insufficient Funds. Details: {e}{RESET}")
        except ccxt.InvalidOrder as e:
            # Handle specific invalid order reasons if possible
            err_str = str(e).lower()
            bybit_code = getattr(e, 'code', None)
            # Bybit V5 error codes are strings like "110007" or integers depending on ccxt version
            code_str = str(bybit_code) if bybit_code is not None else ""

            if code_str == '110043' or ("position idx not match position mode" in err_str):
                 lg.warning(f"{NEON_YELLOW}{order_tag} Order Warning ({symbol} {side}): Reduce-only order failed - likely no position exists or mode mismatch. Details: {e}{RESET}")
                 return {'info': {'reason': 'Position not found or mode mismatch on close attempt'}, 'id': None, 'status': 'rejected'}
            elif code_str == '110040' or "order would exceed position" in err_str:
                 lg.error(f"{NEON_RED}{order_tag} Order FAILED ({symbol} {side}): Invalid Order - Reduce-only would increase/exceed position. Check position state. Details: {e}{RESET}")
            elif "min order value" in err_str or "order cost" in err_str or code_str == '110014':
                 lg.error(f"{NEON_RED}{order_tag} Order FAILED ({symbol} {side}): Invalid Order - Minimum order cost not met ({market_info.get('min_order_cost')}). Check size/price. Details: {e}{RESET}")
            elif "order quantity" in err_str or "size" in err_str or code_str == '110007':
                 lg.error(f"{NEON_RED}{order_tag} Order FAILED ({symbol} {side}): Invalid Order - Quantity issue (Min: {min_order_amount}, Step: {market_info.get('precision', {}).get('amount')}). Check size. Details: {e}{RESET}")
            else:
                 lg.error(f"{NEON_RED}{order_tag} Order FAILED ({symbol} {side}): Invalid Order. Details: {e} (Code: {code_str}){RESET}")
        except ccxt.RateLimitExceeded as e:
            lg.error(f"{NEON_RED}{order_tag} Order FAILED ({symbol} {side}): Rate Limit Exceeded. Details: {e}{RESET}")
            # Consider waiting and retrying place_trade if this happens often
        except ccxt.NetworkError as e:
            lg.error(f"{NEON_RED}{order_tag} Order FAILED ({symbol} {side}): Network Error. Details: {e}{RESET}")
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}{order_tag} Order FAILED ({symbol} {side}): Exchange Error. Details: {e}{RESET}")
        except Exception as e:
            lg.error(f"{NEON_RED}{order_tag} Order FAILED ({symbol} {side}): Unexpected Error. Details: {e}{RESET}", exc_info=True)

        return None

    except Exception as pre_err:
        lg.error(f"{NEON_RED}Error during pre-trade preparation for {symbol}: {pre_err}{RESET}", exc_info=True)
        return None


def _set_position_protection(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: dict,
    position_info: dict,  # Pass the enhanced position dict
    logger: logging.Logger,
    stop_loss_price: Decimal | None = None,
    take_profit_price: Decimal | None = None,
    trailing_stop_distance: Decimal | str | None = None,  # Can be distance or '0' to cancel
    tsl_activation_price: Decimal | str | None = None  # Can be price or '0'
) -> bool:
    """Sets Stop Loss, Take Profit, and/or Trailing Stop Loss for an existing position using Bybit V5 API.
    Handles Decimal inputs and formatting. Use '0' strings for TSL distance/activation to cancel TSL.
    Uses the unified modify_position method if available, otherwise direct V5 call.
    """
    lg = logger
    symbol_id = market_info.get('id', symbol)
    category = _determine_category(market_info)
    # Get position index from enhanced dict (defaults to 0 if not found)
    pos_idx = int(position_info.get('info', {}).get('positionIdx', 0))

    if not category:
        lg.error(f"Cannot set protection for {symbol}: Failed to determine category.")
        return False

    try:
        params = {
            # Parameters for modify_position or direct V5 call
            'category': category,  # Needed for direct call
            'symbol': symbol_id,  # Needed for direct call
            'positionIdx': pos_idx,  # Needed for direct call & Hedge Mode
            # Unified modify_position might take sl/tp/tsl directly in top-level params
        }

        # --- Format Protection Parameters ---
        price_prec_digits = market_info.get('price_precision_digits')
        if price_prec_digits is None:
             lg.error(f"Cannot set protection for {symbol}: Failed to get price precision digits.")
             return False
        price_format_str = f'{{:.{price_prec_digits}f}}'  # e.g., {:.2f}

        log_parts = ["Setting protection"]
        protection_params_for_api = {}  # Store formatted params for the API call

        # Stop Loss
        if stop_loss_price is not None:
            if stop_loss_price > 0:
                protection_params_for_api['stopLoss'] = price_format_str.format(stop_loss_price)
                log_parts.append(f"SL={protection_params_for_api['stopLoss']}")
            else:  # Treat 0 or negative as intent to remove SL
                 protection_params_for_api['stopLoss'] = '0'
                 log_parts.append("SL=Remove")

        # Take Profit
        if take_profit_price is not None:
            if take_profit_price > 0:
                protection_params_for_api['takeProfit'] = price_format_str.format(take_profit_price)
                log_parts.append(f"TP={protection_params_for_api['takeProfit']}")
            else:  # Treat 0 or negative as intent to remove TP
                 protection_params_for_api['takeProfit'] = '0'
                 log_parts.append("TP=Remove")

        # Trailing Stop
        if trailing_stop_distance is not None:
            if isinstance(trailing_stop_distance, str) and trailing_stop_distance == '0':
                protection_params_for_api['trailingStop'] = '0'  # Explicit cancel
                protection_params_for_api['activePrice'] = '0'  # Also clear activation price
                log_parts.append("TSL=Cancel")
            elif isinstance(trailing_stop_distance, Decimal) and trailing_stop_distance > 0:
                 protection_params_for_api['trailingStop'] = price_format_str.format(trailing_stop_distance)
                 log_parts.append(f"TSL_Dist={protection_params_for_api['trailingStop']}")
                 # Activation price
                 if tsl_activation_price is not None:
                     if isinstance(tsl_activation_price, str) and tsl_activation_price == '0':
                         protection_params_for_api['activePrice'] = '0'  # Explicit immediate
                         log_parts.append("TSL_Act=Immediate")
                     elif isinstance(tsl_activation_price, Decimal) and tsl_activation_price > 0:
                         protection_params_for_api['activePrice'] = price_format_str.format(tsl_activation_price)
                         log_parts.append(f"TSL_ActAt={protection_params_for_api['activePrice']}")
                     else:  # Invalid activation price, default to immediate
                         lg.warning(f"Invalid TSL activation price ({tsl_activation_price}) for {symbol}. Using immediate.")
                         protection_params_for_api['activePrice'] = '0'
                 else:  # Activation price not provided, default to immediate
                      protection_params_for_api['activePrice'] = '0'
                      log_parts.append("TSL_Act=Immediate")
            else:  # Invalid distance type or value (e.g., Decimal(0))
                 lg.warning(f"Invalid TSL distance ({trailing_stop_distance}). Treating as cancel.")
                 protection_params_for_api['trailingStop'] = '0'
                 protection_params_for_api['activePrice'] = '0'
                 log_parts.append("TSL=Cancel")

        # --- Make API Call ---
        if not protection_params_for_api:
            lg.info(f"No valid protection parameters provided to set for {symbol}. Skipping API call.")
            return True  # Nothing to do

        lg.info(f"{' '.join(log_parts)} for {symbol} ({category}, Idx={pos_idx}). API Params: {protection_params_for_api}")

        # Combine base params and protection params
        final_api_params = {**params, **protection_params_for_api}

        # Add tpslMode for direct V5 call (if not using modify_position)
        if 'tpslMode' not in final_api_params: final_api_params['tpslMode'] = 'Full'

        response = None
        try:
            # Prefer modify_position if available and handles SL/TP/TSL via params
            if hasattr(exchange, 'modify_position'):
                lg.debug("Attempting protection via exchange.modify_position...")
                # Need to map our params to what modify_position expects
                # This requires checking the specific CCXT implementation for Bybit modify_position
                # Example mapping (CHECK CCXT DOCS/CODE FOR ACCURACY):
                unified_params = {
                    'stopLoss': {'triggerPrice': protection_params_for_api.get('stopLoss')},
                    'takeProfit': {'triggerPrice': protection_params_for_api.get('takeProfit')},
                    'trailingStop': {'trailingTrace': protection_params_for_api.get('trailingStop'),  # Check correct param name
                                     'activationPrice': protection_params_for_api.get('activePrice')}
                }
                # Filter out None values
                unified_params['stopLoss'] = {k: v for k, v in unified_params['stopLoss'].items() if v is not None}
                unified_params['takeProfit'] = {k: v for k, v in unified_params['takeProfit'].items() if v is not None}
                unified_params['trailingStop'] = {k: v for k, v in unified_params['trailingStop'].items() if v is not None}
                # Remove empty dicts
                if not unified_params['stopLoss']: del unified_params['stopLoss']
                if not unified_params['takeProfit']: del unified_params['takeProfit']
                if not unified_params['trailingStop']: del unified_params['trailingStop']

                # Merge with base params potentially needed by modify_position (like positionIdx for hedge)
                unified_params['positionIdx'] = pos_idx  # Pass index if needed

                if unified_params:  # Only call if there are parameters to send
                    # response = exchange.modify_position(symbol, params=unified_params)
                    # Sticking to direct call for now as modify_position mapping is uncertain
                    lg.debug("modify_position mapping uncertain, using direct V5 call instead.")
                    response = exchange.privatePostPositionTradingStop(params=final_api_params)
                else:
                     lg.info("No protection parameters formatted for modify_position call.")
                     return True

            else:  # Fallback to direct V5 call
                lg.debug("Using direct V5 call privatePostPositionTradingStop.")
                response = exchange.privatePostPositionTradingStop(params=final_api_params)

            lg.debug(f"Set protection raw response for {symbol}: {response}")

            # --- Check Response ---
            if response and response.get('retCode') == 0:
                lg.info(f"{NEON_GREEN}Protection successfully set/updated for {symbol}.{RESET}")
                return True
            else:
                ret_code = response.get('retCode', 'N/A')
                ret_msg = response.get('retMsg', 'Unknown Error')
                lg.error(f"{NEON_RED}Failed to set protection for {symbol}. Response Code: {ret_code}, Msg: {ret_msg}{RESET}")
                lg.debug(f"Full failed response: {response}")
                # Add specific error code hints
                code_str = str(ret_code)
                if code_str == '110026': lg.error(f"{NEON_RED}>> Hint: SL/TP price invalid (wrong side, too close?).{RESET}")
                elif code_str == '110042': lg.error(f"{NEON_RED}>> Hint: TSL parameters invalid.{RESET}")
                elif code_str == '110086': lg.error(f"{NEON_RED}>> Hint: SL/TP price cannot be the same.{RESET}")
                elif code_str == '110084': lg.error(f"{NEON_RED}>> Hint: TP price invalid (wrong side?).{RESET}")
                elif code_str == '110085': lg.error(f"{NEON_RED}>> Hint: SL price invalid (wrong side?).{RESET}")
                elif code_str == '110025': lg.error(f"{NEON_RED}>> Hint: Position not found or size is zero.{RESET}")
                elif code_str == '110055': lg.error(f"{NEON_RED}>> Hint: Position Idx ({pos_idx}) mismatch with Position Mode.{RESET}")
                return False

        except ccxt.AuthenticationError as e:
            lg.error(f"{NEON_RED}Authentication error setting protection for {symbol}: {e}{RESET}")
        except ccxt.NetworkError as e:
            lg.error(f"{NEON_RED}Network error setting protection for {symbol}: {e}{RESET}")
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error setting protection for {symbol}: {e}{RESET}")
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error setting protection for {symbol}: {e}{RESET}", exc_info=True)

    except Exception as e:
        lg.error(f"{NEON_RED}Error preparing parameters for protection setting ({symbol}): {e}{RESET}", exc_info=True)

    return False


def set_trailing_stop_loss(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: dict,
    position_info: dict,  # Pass the enhanced position dict
    config: dict[str, Any],
    logger: logging.Logger,
    take_profit_price: Decimal | None = None  # Optionally set TP at the same time
) -> bool:
    """Calculates and sets a Trailing Stop Loss using Bybit V5 API via _set_position_protection.
    Uses percentages from config relative to entry price to calculate distance/activation.
    """
    lg = logger
    if not config.get("enable_trailing_stop", False):
        lg.debug(f"Trailing Stop Loss is disabled in config for {symbol}.")
        return True  # Not enabled, considered success (did not fail)

    try:
        entry_price = position_info.get('entryPriceDecimal')
        pos_side = position_info.get('side')  # 'long' or 'short'

        if not entry_price or entry_price <= 0 or pos_side not in ['long', 'short']:
            lg.error(f"Cannot set TSL for {symbol}: Invalid position data (Entry: {entry_price}, Side: {pos_side}).")
            return False

        # --- Get TSL Config ---
        callback_rate_pct = Decimal(str(config.get("trailing_stop_callback_rate", 0.005)))  # e.g., 0.5%
        activation_pct = Decimal(str(config.get("trailing_stop_activation_percentage", 0.003)))  # e.g., 0.3%
        min_tick = market_info.get('min_tick_size')

        if callback_rate_pct <= 0:
            lg.error(f"Cannot set TSL for {symbol}: Invalid trailing_stop_callback_rate ({callback_rate_pct}). Must be positive.")
            return False
        if activation_pct < 0:
            lg.warning(f"Invalid trailing_stop_activation_percentage ({activation_pct}). Using 0 (immediate activation).")
            activation_pct = Decimal(0)
        if min_tick is None or min_tick <= 0:
             lg.error(f"Cannot set TSL for {symbol}: Invalid min tick size ({min_tick}).")
             return False

        # --- Calculate Absolute Distance ---
        # TSL distance is always positive, difference in price points
        tsl_distance_raw = entry_price * callback_rate_pct
        # Quantize distance UP to the nearest tick size (safer side)
        tsl_distance = (tsl_distance_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick
        if tsl_distance <= 0:  # Ensure distance is at least one tick
             tsl_distance = min_tick
        lg.debug(f"TSL Calc ({symbol}): Entry={entry_price}, Callback%={callback_rate_pct:.3%}, RawDist={tsl_distance_raw}, QuantizedDist={tsl_distance}")

        # --- Calculate Activation Price ---
        tsl_activation_price = Decimal('0')  # Default: immediate activation
        if activation_pct > 0:
            activation_offset = entry_price * activation_pct
            if pos_side == 'long':
                act_price_raw = entry_price + activation_offset
                # Quantize activation price UP (trigger later/safer for long)
                tsl_activation_price = (act_price_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick
                 # Ensure activation is strictly > entry
                if tsl_activation_price <= entry_price:
                     tsl_activation_price = entry_price + min_tick
            else:  # Short
                act_price_raw = entry_price - activation_offset
                # Quantize activation price DOWN (trigger later/safer for short)
                tsl_activation_price = (act_price_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                # Ensure activation is strictly < entry
                if tsl_activation_price >= entry_price:
                     tsl_activation_price = entry_price - min_tick

            # Ensure activation price is valid (positive)
            if tsl_activation_price <= 0:
                 lg.warning(f"Calculated TSL activation price ({tsl_activation_price}) is invalid. Using immediate activation.")
                 tsl_activation_price = Decimal('0')
            lg.debug(f"TSL Calc ({symbol}): Activation%={activation_pct:.3%}, Offset={activation_offset}, RawActPrice={act_price_raw}, QuantizedActPrice={tsl_activation_price}")

        # --- Call Protection Function ---
        lg.info(f"Setting Trailing Stop for {symbol}: Distance={tsl_distance}, ActivationPrice={tsl_activation_price if tsl_activation_price > 0 else 'Immediate'}")
        # CRITICAL: Pass stop_loss_price=None (or '0'?) to _set_position_protection if TSL should override fixed SL.
        # If TSL should co-exist (Bybit usually doesn't allow this easily), logic needs change.
        # Assuming TSL replaces fixed SL when activated.
        return _set_position_protection(
            exchange=exchange,
            symbol=symbol,
            market_info=market_info,
            position_info=position_info,
            logger=lg,
            stop_loss_price=None,  # Let TSL handle the stop loss
            take_profit_price=take_profit_price,  # Set TP if provided
            trailing_stop_distance=tsl_distance,  # Pass calculated distance
            tsl_activation_price=tsl_activation_price  # Pass calculated activation price ('0' for immediate)
        )

    except (InvalidOperation, ValueError, TypeError, KeyError) as e:
        lg.error(f"{NEON_RED}Error calculating TSL parameters for {symbol}: {e}{RESET}", exc_info=True)
        return False
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error setting TSL for {symbol}: {e}{RESET}", exc_info=True)
        return False


# --- Main Analysis and Trading Logic ---

# Global dictionary to hold state per symbol (e.g., break_even_triggered)
# Allows state to persist across calls to analyze_and_trade_symbol
symbol_states: dict[str, dict[str, Any]] = {}


def analyze_and_trade_symbol(
    exchange: ccxt.Exchange,
    symbol: str,
    config: dict[str, Any],
    logger: logging.Logger,
    market_info: dict,  # Pass pre-fetched market info
    symbol_state: dict[str, Any],  # Pass mutable state dict for the symbol
    current_balance: Decimal | None,  # Pass current balance
    all_open_positions: dict[str, dict]  # Pass dict of all open positions {symbol: position_dict}
) -> None:
    """Analyzes a single symbol and executes/manages trades based on signals and config."""
    lg = logger  # Use the symbol-specific logger
    lg.info(f"---== Analyzing {symbol} ({config['interval']}) ==---")
    cycle_start_time = time.monotonic()

    # --- 1. Fetch Data (Kline, Ticker, Orderbook) ---
    ccxt_interval = CCXT_INTERVAL_MAP.get(config["interval"])
    # Determine required kline history based on longest indicator period + buffer
    indicator_periods = [
        config.get(p, globals().get(f"DEFAULT_{p.upper()}", 14))  # Get period from config or default const
        for p in ["atr_period", "ema_long_period", "cci_window", "williams_r_window",
                  "mfi_window", "sma_10_window", "rsi_period", "bollinger_bands_period",
                  "volume_ma_period", "fibonacci_window"]
    ]
    stoch_rsi_lookback = config.get("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW) + \
                         config.get("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW) + \
                         max(config.get("stoch_rsi_k", DEFAULT_K_WINDOW), config.get("stoch_rsi_d", DEFAULT_D_WINDOW))
    indicator_periods.append(stoch_rsi_lookback)
    required_kline_history = max(indicator_periods) if indicator_periods else 100
    kline_buffer = 50  # Add buffer for indicator warmup
    kline_limit = max(250, required_kline_history + kline_buffer)  # Fetch reasonable minimum or required amount + buffer

    lg.debug(f"Required kline history for indicators: {required_kline_history}, Fetching limit: {kline_limit}")
    klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=kline_limit, logger=lg, market_info=market_info)

    # Check if sufficient data was fetched AFTER cleaning/processing inside fetch_klines_ccxt
    if klines_df.empty or len(klines_df) < required_kline_history:  # Check against required period
        lg.error(f"{NEON_RED}Failed to fetch sufficient kline data for {symbol} (fetched {len(klines_df)}, needed >{required_kline_history}). Skipping analysis.{RESET}")
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
    if config.get("indicators", {}).get("orderbook", False) and Decimal(str(active_weights.get("orderbook", 0))) != 0:
        orderbook_data = fetch_orderbook_ccxt(exchange, symbol, config.get("orderbook_limit", 25), lg, market_info=market_info)
        if orderbook_data is None:
             lg.warning(f"{NEON_YELLOW}Orderbook fetching enabled but failed for {symbol}. Score will be 0.{RESET}")

    # --- 2. Analyze Data & Generate Signal ---
    try:
        # Pass persistent symbol_state dict
        analyzer = TradingAnalyzer(
            df=klines_df.copy(),
            logger=lg,
            config=config,
            market_info=market_info,
            symbol_state=symbol_state  # Pass the mutable state dict here
        )
    except ValueError as e_analyzer:
        lg.error(f"{NEON_RED}Failed to initialize TradingAnalyzer for {symbol}: {e_analyzer}. Skipping analysis.{RESET}")
        return
    except Exception as e_analyzer_other:
         lg.error(f"{NEON_RED}Unexpected error initializing TradingAnalyzer for {symbol}: {e_analyzer_other}. Skipping analysis.{RESET}", exc_info=True)
         return

    # Generate the trading signal
    signal = analyzer.generate_trading_signal(current_price, orderbook_data)

    # Get necessary data points for potential trade/management
    price_precision_digits = analyzer.get_price_precision()
    min_tick_size = analyzer.get_min_tick_size()
    current_atr_float = analyzer.indicator_values.get("ATR")

    if min_tick_size is None or min_tick_size <= 0:
         lg.error(f"{NEON_RED}Invalid minimum tick size ({min_tick_size}) for {symbol}. Cannot calculate trade parameters. Skipping cycle.{RESET}")
         return

    # Calculate potential initial SL/TP (used for sizing if no position exists)
    _, tp_potential, sl_potential = analyzer.calculate_entry_tp_sl(current_price, signal)

    # --- 3. Log Analysis Summary ---
    atr_log = f"{current_atr_float:.{price_precision_digits + 1}f}" if current_atr_float is not None and pd.notna(current_atr_float) else 'N/A'
    sl_pot_log = f"{sl_potential:.{price_precision_digits}f}" if sl_potential else 'N/A'
    tp_pot_log = f"{tp_potential:.{price_precision_digits}f}" if tp_potential else 'N/A'
    lg.info(f"Analysis: ATR={atr_log}, Potential SL={sl_pot_log}, Potential TP={tp_pot_log}")

    tsl_enabled = config.get('enable_trailing_stop', False)
    be_enabled = config.get('enable_break_even', False)
    ma_exit_enabled = config.get('enable_ma_cross_exit', False)
    lg.info(f"Config: TSL={tsl_enabled}, BE={be_enabled} (Triggered: {analyzer.break_even_triggered}), MA Cross Exit={ma_exit_enabled}")

    # --- 4. Check Position & Execute/Manage ---
    if not config.get("enable_trading", False):
        lg.info(f"{NEON_YELLOW}Trading is disabled in config. Analysis complete, no trade actions taken for {symbol}.{RESET}")
        cycle_end_time = time.monotonic()
        lg.debug(f"---== Analysis Cycle End for {symbol} ({cycle_end_time - cycle_start_time:.2f}s) ==---")
        return

    # Use the pre-fetched position data for this symbol
    open_position = all_open_positions.get(symbol)  # Get enhanced position dict or None

    # Get current count of ALL *validated* open positions
    current_total_open_positions = len(all_open_positions)

    # --- Scenario 1: No Open Position for this Symbol ---
    if open_position is None:
        # Reset break-even state if no position exists for this symbol
        if analyzer.break_even_triggered:
             lg.info(f"Resetting break-even triggered state for {symbol} as no position is open.")
             analyzer.break_even_triggered = False

        if signal in ["BUY", "SELL"]:
            # --- Check Max Position Limit ---
            max_pos_total = config.get("max_concurrent_positions_total", 1)
            if current_total_open_positions >= max_pos_total:
                 lg.info(f"{NEON_YELLOW}Signal {signal} for {symbol}, but max concurrent positions ({max_pos_total}) already reached ({current_total_open_positions} open). Skipping new entry.{RESET}")
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
                    # Attempt to set leverage based on config before entry.
                    if not set_leverage_ccxt(exchange, symbol, leverage, market_info, lg):
                        lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Failed to set/confirm leverage to {leverage}x. Check logs.{RESET}")
                        return
            else:
                 lg.info(f"Leverage setting skipped for {symbol} (Spot market).")

            # Calculate Position Size using potential SL
            position_size = calculate_position_size(
                balance=current_balance,  # Use passed balance
                risk_per_trade=config["risk_per_trade"],
                initial_stop_loss_price=sl_potential,  # Use potential SL for sizing
                entry_price=current_price,  # Use current price as estimate
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
                position_size=position_size,  # Pass Decimal size
                market_info=market_info,
                logger=lg,
                reduce_only=False  # Opening trade
            )

            # --- Post-Order: Verify Position and Set Protection ---
            if trade_order and trade_order.get('id'):
                order_id = trade_order['id']
                lg.info(f"Order {order_id} placed for {symbol}. Waiting {POSITION_CONFIRM_DELAY}s for position confirmation...")
                time.sleep(POSITION_CONFIRM_DELAY)

                # Re-fetch position state specifically for this symbol AFTER the delay
                lg.info(f"Attempting to confirm position for {symbol} after order {order_id}...")
                # Use the same enhanced fetching logic
                confirmed_position = get_open_position(exchange, symbol, lg, market_info=market_info, position_mode=config.get("position_mode", "One-Way"))

                if confirmed_position:
                    # --- Position Confirmed ---
                    try:
                        entry_price_actual = confirmed_position.get('entryPriceDecimal')  # Use Decimal version
                        pos_size_actual = confirmed_position.get('contractsDecimal')  # Use Decimal version

                        if entry_price_actual and entry_price_actual > 0 and \
                           pos_size_actual is not None and abs(pos_size_actual) > Decimal('1e-9'):
                            lg.info(f"{NEON_GREEN}Position Confirmed for {symbol}! Actual Entry: ~{entry_price_actual:.{price_precision_digits}f}, Actual Size: {pos_size_actual}{RESET}")

                            # Add the newly confirmed position to the global state for the next symbol's check
                            all_open_positions[symbol] = confirmed_position

                            # --- Recalculate SL/TP based on ACTUAL entry price ---
                            _, tp_actual, sl_actual = analyzer.calculate_entry_tp_sl(entry_price_actual, signal)

                            # --- Set Protection based on Config (TSL or Fixed SL/TP) ---
                            protection_set_success = False
                            if config.get("enable_trailing_stop", False):
                                lg.info(f"Setting Trailing Stop Loss for {symbol} (TP target: {tp_actual})...")
                                protection_set_success = set_trailing_stop_loss(
                                    exchange=exchange, symbol=symbol, market_info=market_info,
                                    position_info=confirmed_position, config=config, logger=lg,
                                    take_profit_price=tp_actual  # Pass optional TP
                                )
                            else:
                                lg.info(f"Setting Fixed Stop Loss ({sl_actual}) and Take Profit ({tp_actual}) for {symbol}...")
                                if sl_actual or tp_actual:  # Only call if at least one is valid
                                    protection_set_success = _set_position_protection(
                                        exchange=exchange, symbol=symbol, market_info=market_info,
                                        position_info=confirmed_position, logger=lg,
                                        stop_loss_price=sl_actual,  # Pass Decimal or None
                                        take_profit_price=tp_actual,  # Pass Decimal or None
                                        trailing_stop_distance='0',  # Ensure TSL is cancelled
                                        tsl_activation_price='0'
                                    )
                                else:
                                    lg.warning(f"{NEON_YELLOW}Fixed SL/TP calculation based on actual entry failed or resulted in None for {symbol}. No fixed protection set.{RESET}")
                                    protection_set_success = True  # Treat as 'success' if no protection needed

                            # --- Final Status Log ---
                            if protection_set_success:
                                lg.info(f"{NEON_GREEN}=== TRADE ENTRY & PROTECTION SETUP COMPLETE for {symbol} ({signal}) ===")
                                # Reset BE state for the new trade
                                analyzer.break_even_triggered = False
                            else:
                                lg.error(f"{NEON_RED}=== TRADE placed BUT FAILED TO SET/CONFIRM PROTECTION (SL/TP/TSL) for {symbol} ({signal}) ===")
                                lg.warning(f"{NEON_YELLOW}Position is open without automated protection. Manual monitoring/intervention may be required!{RESET}")
                        else:
                            lg.error(f"{NEON_RED}Trade placed, but confirmed position data is invalid (Entry={entry_price_actual}, Size={pos_size_actual}). Cannot set protection.{RESET}")
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
                        fetch_params = {}
                        if 'bybit' in exchange.id.lower() and market_info:
                             category = _determine_category(market_info)
                             if category: fetch_params['category'] = category
                        order_status_info = exchange.fetch_order(order_id, symbol, params=fetch_params)
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
    else:  # open_position is not None (and is validated)
        pos_side = open_position.get('side')  # 'long' or 'short'
        pos_size_dec = open_position.get('contractsDecimal')  # Decimal (negative for short)
        entry_price_dec = open_position.get('entryPriceDecimal')  # Decimal
        # Use parsed Decimal versions for protection checks
        current_sl_price_dec = open_position.get('stopLossPriceDecimal')  # Might be None
        current_tp_price_dec = open_position.get('takeProfitPriceDecimal')  # Might be None
        tsl_distance_dec = open_position.get('trailingStopLossDistanceDecimal', Decimal(0))  # Decimal(0) if inactive
        is_tsl_active = tsl_distance_dec > 0

        # Format for logging
        sl_log_str = f"{current_sl_price_dec:.{price_precision_digits}f}" if current_sl_price_dec else 'N/A'
        tp_log_str = f"{current_tp_price_dec:.{price_precision_digits}f}" if current_tp_price_dec else 'N/A'
        pos_size_log = f"{pos_size_dec}"  # Already Decimal
        entry_price_log = f"{entry_price_dec:.{price_precision_digits}f}" if entry_price_dec else 'N/A'

        lg.info(f"Existing {pos_side.upper()} position found for {symbol}. Size: {pos_size_log}, Entry: {entry_price_log}, SL: {sl_log_str}, TP: {tp_log_str}, TSL Active: {is_tsl_active}")

        # --- ** Check Exit Conditions ** ---
        exit_signal_triggered = False
        if (pos_side == 'long' and signal == "SELL") or \
           (pos_side == 'short' and signal == "BUY"):
            exit_signal_triggered = True
            lg.warning(f"{NEON_YELLOW}*** EXIT Signal Triggered: New signal ({signal}) opposes existing {pos_side} position for {symbol}. ***{RESET}")

        ma_cross_exit = False
        if not exit_signal_triggered and config.get("enable_ma_cross_exit", False):
            ema_short = analyzer.indicator_values.get("EMA_Short")
            ema_long = analyzer.indicator_values.get("EMA_Long")
            if pd.notna(ema_short) and pd.notna(ema_long):
                if pos_side == 'long' and ema_short < ema_long:
                    ma_cross_exit = True
                    lg.warning(f"{NEON_YELLOW}*** MA CROSS EXIT (Bearish): Short EMA ({ema_short:.{price_precision_digits}f}) < Long EMA ({ema_long:.{price_precision_digits}f}). Closing LONG. ***{RESET}")
                elif pos_side == 'short' and ema_short > ema_long:
                    ma_cross_exit = True
                    lg.warning(f"{NEON_YELLOW}*** MA CROSS EXIT (Bullish): Short EMA ({ema_short:.{price_precision_digits}f}) > Long EMA ({ema_long:.{price_precision_digits}f}). Closing SHORT. ***{RESET}")
            else:
                lg.warning("MA cross exit check skipped: EMA values not available.")

        # --- Execute Position Close if Exit Condition Met ---
        if exit_signal_triggered or ma_cross_exit:
            lg.info(f"Attempting to close {pos_side} position for {symbol} with a market order...")
            try:
                close_side_signal = "SELL" if pos_side == 'long' else "BUY"
                size_to_close = abs(pos_size_dec)  # Use absolute size
                if size_to_close <= 0:
                    raise ValueError(f"Cannot close: Existing position size {pos_size_dec} is zero or invalid.")

                close_order = place_trade(
                    exchange=exchange, symbol=symbol, trade_signal=close_side_signal,
                    position_size=size_to_close, market_info=market_info, logger=lg,
                    reduce_only=True  # CRITICAL for closing
                )

                if close_order:
                     order_id = close_order.get('id')
                     if close_order.get('info', {}).get('reason') == 'Position not found or mode mismatch on close attempt':
                          lg.info(f"{NEON_GREEN}Position for {symbol} likely already closed (or closing order unnecessary).{RESET}")
                          analyzer.break_even_triggered = False  # Reset state
                          # Remove from global state if confirmed closed
                          if symbol in all_open_positions: del all_open_positions[symbol]
                     elif order_id:
                          lg.info(f"{NEON_GREEN}Closing order {order_id} placed for {symbol}. Waiting {POSITION_CONFIRM_DELAY}s to verify closure...{RESET}")
                          time.sleep(POSITION_CONFIRM_DELAY)
                          # Verify position is actually closed
                          final_position = get_open_position(exchange, symbol, lg, market_info=market_info, position_mode=config.get("position_mode", "One-Way"))
                          if final_position is None:
                              lg.info(f"{NEON_GREEN}=== POSITION for {symbol} successfully closed. ===")
                              analyzer.break_even_triggered = False  # Reset state
                              # Remove from global state
                              if symbol in all_open_positions: del all_open_positions[symbol]
                          else:
                              lg.error(f"{NEON_RED}*** POSITION CLOSE FAILED for {symbol} after placing reduceOnly order {order_id}. Position still detected: {final_position.get('side')} size {final_position.get('contractsDecimal')}{RESET}")
                              lg.warning(f"{NEON_YELLOW}Manual investigation required!{RESET}")
                     else:  # Order failed but didn't return the dummy response
                         lg.error(f"{NEON_RED}Failed to place closing order for {symbol} (Order ID N/A). Manual intervention required.{RESET}")
                else:  # place_trade returned None
                    lg.error(f"{NEON_RED}Failed to place closing order for {symbol}. Manual intervention required.{RESET}")

            except (ValueError, InvalidOperation, TypeError) as size_err:
                 lg.error(f"{NEON_RED}Error determining size for closing order ({symbol}): {size_err}. Manual intervention required.{RESET}")
            except Exception as close_err:
                 lg.error(f"{NEON_RED}Unexpected error during position close attempt for {symbol}: {close_err}{RESET}", exc_info=True)
                 lg.warning(f"{NEON_YELLOW}Manual intervention required!{RESET}")

        # --- Check Break-Even Condition (Only if NOT Exiting and BE enabled and BE not already triggered for this trade) ---
        elif config.get("enable_break_even", False) and not analyzer.break_even_triggered:
            # Use already validated entry_price_dec and current_atr_float
            if current_atr_float and pd.notna(current_atr_float) and current_atr_float > 0:
                try:
                    current_atr_dec = Decimal(str(current_atr_float))
                    trigger_multiple = Decimal(str(config.get("break_even_trigger_atr_multiple", 1.0)))
                    offset_ticks = int(config.get("break_even_offset_ticks", 2))
                    # min_tick already fetched and validated earlier

                    profit_target_offset = current_atr_dec * trigger_multiple
                    be_stop_price = None
                    trigger_met = False
                    trigger_price = None  # Price level that needs to be reached

                    if pos_side == 'long':
                        trigger_price = entry_price_dec + profit_target_offset
                        if current_price >= trigger_price:
                            trigger_met = True
                            be_stop_raw = entry_price_dec + (min_tick_size * offset_ticks)
                            be_stop_price = (be_stop_raw / min_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick_size
                            if be_stop_price <= entry_price_dec: be_stop_price = entry_price_dec + min_tick_size

                    elif pos_side == 'short':
                        trigger_price = entry_price_dec - profit_target_offset
                        if current_price <= trigger_price:
                            trigger_met = True
                            be_stop_raw = entry_price_dec - (min_tick_size * offset_ticks)
                            be_stop_price = (be_stop_raw / min_tick_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick_size
                            if be_stop_price >= entry_price_dec: be_stop_price = entry_price_dec - min_tick_size

                    # --- Trigger Break-Even SL Modification ---
                    if trigger_met:
                        if be_stop_price and be_stop_price > 0:
                            lg.warning(f"{NEON_PURPLE}*** BREAK-EVEN Triggered for {symbol} {pos_side.upper()} position! ***{RESET}")
                            lg.info(f"  Current Price: {current_price:.{price_precision_digits}f}, Trigger Price: {trigger_price:.{price_precision_digits}f}")
                            lg.info(f"  Current SL: {sl_log_str}, Current TP: {tp_log_str}, TSL Active: {is_tsl_active}")
                            lg.info(f"  Target Break-Even SL: {be_stop_price:.{price_precision_digits}f}")

                            needs_sl_update = True
                            force_fixed_sl = config.get("break_even_force_fixed_sl", True)

                            if is_tsl_active:
                                if force_fixed_sl:
                                     lg.info("  TSL active, replacing with fixed BE SL (break_even_force_fixed_sl=True).")
                                else:
                                     lg.warning("BE triggered, TSL active & break_even_force_fixed_sl=False. Keeping TSL (verify behavior).")
                                     needs_sl_update = False
                                     if not analyzer.break_even_triggered:
                                         analyzer.break_even_triggered = True  # Mark BE logically triggered
                                         lg.info(f"Break-even logic triggered for {symbol}, but TSL remains active.")

                            if needs_sl_update and current_sl_price_dec:  # Check if fixed SL exists and needs update
                                if (pos_side == 'long' and current_sl_price_dec >= be_stop_price) or \
                                   (pos_side == 'short' and current_sl_price_dec <= be_stop_price):
                                     lg.info(f"  Existing Fixed SL ({sl_log_str}) is already better than BE target. No SL modification needed.")
                                     needs_sl_update = False
                                     if not analyzer.break_even_triggered:
                                         analyzer.break_even_triggered = True  # Mark BE logically triggered
                                         lg.info(f"Break-even logic triggered for {symbol}, existing SL is sufficient.")

                            # --- Perform the SL Update ---
                            if needs_sl_update:
                                lg.info(f"  Modifying protection: Setting Fixed SL to {be_stop_price:.{price_precision_digits}f}, keeping TP {tp_log_str}, cancelling TSL (if active).")
                                be_success = _set_position_protection(
                                    exchange=exchange, symbol=symbol, market_info=market_info,
                                    position_info=open_position, logger=lg,
                                    stop_loss_price=be_stop_price,  # New BE SL
                                    take_profit_price=current_tp_price_dec,  # Keep existing TP
                                    trailing_stop_distance='0',  # CRITICAL: Cancel TSL
                                    tsl_activation_price='0'
                                )
                                if be_success:
                                    lg.info(f"{NEON_GREEN}Break-even stop loss successfully set/updated for {symbol}.{RESET}")
                                    analyzer.break_even_triggered = True  # Mark BE actioned
                                    # Update the global position state immediately after successful API call
                                    if symbol in all_open_positions:
                                         all_open_positions[symbol]['stopLossPriceDecimal'] = be_stop_price
                                         all_open_positions[symbol]['trailingStopLossDistanceDecimal'] = Decimal(0)  # Mark TSL as inactive
                                else:
                                    lg.error(f"{NEON_RED}Failed to set break-even stop loss for {symbol}. Original protection may still be active.{RESET}")

                        else:  # be_stop_price calculation failed or invalid
                            lg.error(f"{NEON_RED}Break-even triggered, but calculated BE stop price ({be_stop_price}) is invalid. Cannot modify SL.{RESET}")

                except (InvalidOperation, ValueError, TypeError, KeyError, Exception) as be_err:
                    lg.error(f"{NEON_RED}Error during break-even check/calculation for {symbol}: {be_err}{RESET}", exc_info=True)
            else:
                if not current_atr_float or not pd.notna(current_atr_float) or current_atr_float <= 0: lg.debug(f"BE check skipped for {symbol}: Invalid ATR.")

        # --- No Exit/BE Condition Met ---
        elif not (exit_signal_triggered or ma_cross_exit):  # Ensure we weren't trying to exit
             lg.info(f"Signal is {signal}. Holding existing {pos_side.upper()} position for {symbol}. No management action required this cycle.")
             # Optional: Add logic here to adjust TP/SL based on new analysis if desired,
             # e.g., if TSL is not active, maybe adjust fixed SL/TP based on new ATR/Fib levels?

    # --- End of Scenario 2 (Existing Open Position) ---

    cycle_end_time = time.monotonic()
    lg.info(f"---== Analysis Cycle End for {symbol} ({cycle_end_time - cycle_start_time:.2f}s) ==---")


# --- Main Function ---
def main() -> None:
    """Main function to run the bot."""
    # Determine console log level from arguments
    parser = argparse.ArgumentParser(description="Enhanced Bybit Trading Bot (Whale 2.6 - Multi/V5/BE)")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable DEBUG level logging to console")
    parser.add_argument("-s", "--symbol", help="Trade only a specific symbol (e.g., BTC/USDT:USDT), overrides config list")
    args = parser.parse_args()

    global console_log_level
    console_log_level = logging.DEBUG if args.debug else logging.INFO

    # Setup main logger (used for init and overall status)
    main_logger = get_logger("main")
    main_logger.info("*** Live Trading Bot Whale 2.6 (Multi/V5/BE) Initializing ***")
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
    market_infos: dict[str, dict] = {}
    symbol_loggers: dict[str, logging.Logger] = {}

    main_logger.info("Validating symbols and fetching market info...")
    for symbol in symbols_to_trade:
        # Clean up symbol name if needed (e.g., remove extra spaces)
        symbol = symbol.strip()
        if not symbol: continue

        logger = get_logger(symbol, is_symbol_logger=True)  # Get/create logger for the symbol
        symbol_loggers[symbol] = logger  # Store logger
        market_info = get_market_info(exchange, symbol, logger)
        if market_info:
            valid_symbols.append(symbol)
            market_infos[symbol] = market_info
            # Initialize state for valid symbols
            if symbol not in symbol_states:  # Avoid overwriting if restarting
                symbol_states[symbol] = {'break_even_triggered': False}  # Initialize state
            logger.info(f"Symbol {symbol} validated. Market Type: {market_info.get('type', 'N/A')}, Category: {_determine_category(market_info)}")
        else:
            logger.error(f"Symbol {symbol} is invalid or market info could not be fetched. It will be skipped.")

    if not valid_symbols:
        main_logger.error(f"{NEON_RED}No valid symbols remaining after validation. Exiting.{RESET}")
        return

    main_logger.info(f"Validated symbols ready for trading: {', '.join(valid_symbols)}")

    # --- Bot Main Loop ---
    main_logger.info(f"Starting main trading loop for {len(valid_symbols)} symbols... Loop Delay: {LOOP_DELAY_SECONDS}s")
    while True:
        loop_start_utc = datetime.now(ZoneInfo("UTC"))
        loop_start_local = loop_start_utc.astimezone(TIMEZONE)
        main_logger.debug(f"--- New Main Loop Cycle --- | {loop_start_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        try:
            # --- Fetch Global Data Once Per Loop ---
            # 1. Fetch Balance
            current_balance = fetch_balance(exchange, QUOTE_CURRENCY, main_logger)
            if current_balance is None:
                 main_logger.warning(f"{NEON_YELLOW}Failed to fetch balance for {QUOTE_CURRENCY} this cycle. Size calculations may fail.{RESET}")

            # 2. Fetch All Open Positions for Tracked Symbols
            all_open_positions: dict[str, dict] = {}  # Holds ENHANCED position dicts for currently open positions
            try:
                 main_logger.debug("Fetching all positions for relevant categories...")
                 raw_positions_list = []
                 categories_to_check = set()
                 for sym in valid_symbols:
                      cat = _determine_category(market_infos[sym])
                      if cat: categories_to_check.add(cat)

                 # Fetch positions per category for ALL symbols at once if possible
                 # Note: fetch_positions(symbols=None, params={'category': cat}) fetches all for that category
                 for category in categories_to_check:
                      try:
                           fetch_params = {'category': category}
                           category_positions = exchange.fetch_positions(symbols=None, params=fetch_params)
                           raw_positions_list.extend(category_positions)
                           main_logger.debug(f"Fetched {len(category_positions)} raw position entries for category '{category}'.")
                      except ccxt.ExchangeError as e_cat:
                           if "account type mismatch" in str(e_cat).lower() or getattr(e_cat, 'code', None) == 170001:
                                main_logger.debug(f"Skipping position fetch for category '{category}': Account type mismatch (Error: {e_cat}).")
                           else:
                               main_logger.warning(f"Could not fetch positions for category '{category}': {e_cat}")
                      except Exception as e_cat:
                           main_logger.warning(f"Unexpected error fetching positions for category '{category}': {e_cat}")

                 # Process raw positions: Filter for valid symbols and enhance valid ones
                 main_logger.debug(f"Processing {len(raw_positions_list)} raw position entries...")
                 position_mode = CONFIG.get("position_mode", "One-Way")
                 temp_all_positions: dict[str, dict] = {}  # Temporary dict for this cycle's fetch

                 # Group positions by symbol first for easier processing
                 raw_by_symbol: dict[str, list[dict]] = {}
                 for pos_raw in raw_positions_list:
                      symbol_raw = pos_raw.get('symbol')
                      # Map exchange ID back to CCXT symbol if possible
                      ccxt_symbol = None
                      if symbol_raw:
                           for s, mi in market_infos.items():
                                if mi.get('id') == symbol_raw:
                                     ccxt_symbol = s
                                     break
                           if not ccxt_symbol: ccxt_symbol = symbol_raw  # Use raw if no match

                      if ccxt_symbol and ccxt_symbol in valid_symbols:
                           if ccxt_symbol not in raw_by_symbol:
                                raw_by_symbol[ccxt_symbol] = []
                           raw_by_symbol[ccxt_symbol].append(pos_raw)

                 # Enhance and validate positions for each relevant symbol
                 for symbol, symbol_raw_list in raw_by_symbol.items():
                      market_info = market_infos.get(symbol)
                      if not market_info: continue

                      # Simulate get_open_position by passing the raw list for this symbol
                      original_fetch = exchange.fetch_positions
                      exchange.fetch_positions = lambda symbols=None, params=None: symbol_raw_list
                      try:
                          enhanced_pos = get_open_position(exchange, symbol, main_logger, market_info, position_mode)
                          if enhanced_pos:
                              temp_all_positions[symbol] = enhanced_pos
                      finally:
                          exchange.fetch_positions = original_fetch  # Restore

                 all_open_positions = temp_all_positions  # Update global state for this cycle
                 main_logger.info(f"Total validated open positions detected across tracked symbols: {len(all_open_positions)}")
                 if len(all_open_positions) > 0:
                      main_logger.debug(f"Open positions: {list(all_open_positions.keys())}")

            except ccxt.AuthenticationError as e_pos_auth:
                 main_logger.critical(f"{NEON_RED}CRITICAL: Authentication Error fetching positions: {e_pos_auth}. Bot cannot manage positions.{RESET}")
                 all_open_positions = {}  # Assume none
            except Exception as e_pos:
                main_logger.error(f"{NEON_RED}Failed to fetch or process open positions this cycle: {e_pos}{RESET}", exc_info=True)
                all_open_positions = {}  # Assume none

            # --- Iterate Through Symbols ---
            for symbol in valid_symbols:
                symbol_logger = symbol_loggers[symbol]
                symbol_market_info = market_infos[symbol]
                # Get the persistent state dict for this symbol
                if symbol not in symbol_states: symbol_states[symbol] = {'break_even_triggered': False}
                current_symbol_state = symbol_states[symbol]

                # Run analysis and trading logic for the individual symbol
                try:
                    analyze_and_trade_symbol(
                        exchange=exchange,
                        symbol=symbol,
                        config=CONFIG,
                        logger=symbol_logger,
                        market_info=symbol_market_info,
                        symbol_state=current_symbol_state,  # Pass the mutable state dict
                        current_balance=current_balance,  # Pass overall balance
                        all_open_positions=all_open_positions  # Pass dict of currently known open positions
                    )
                except Exception as symbol_analysis_err:
                     symbol_logger.error(f"{NEON_RED}!! Unhandled error during analysis/trading for {symbol}: {symbol_analysis_err} !!{RESET}", exc_info=True)
                     symbol_logger.error(f"{NEON_YELLOW}Skipping rest of cycle for {symbol} due to error.{RESET}")

                # Short delay between processing symbols to avoid hitting rate limits too quickly
                time.sleep(0.5)  # Adjust as needed

        except KeyboardInterrupt:
            main_logger.info("KeyboardInterrupt received. Attempting graceful shutdown...")
            main_logger.info("Shutdown complete.")
            break
        except ccxt.AuthenticationError as e:
            main_logger.critical(f"{NEON_RED}CRITICAL: Authentication Error during main loop: {e}. Bot stopped.{RESET}")
            break
        except ccxt.NetworkError as e:
            main_logger.error(f"{NEON_RED}Network Error in main loop: {e}. Retrying after longer delay...{RESET}")
            time.sleep(RETRY_DELAY_SECONDS * 10)
        except ccxt.RateLimitExceeded as e:
             retry_after = getattr(e, 'retryAfter', None)
             wait_time = RETRY_DELAY_SECONDS * 2
             if retry_after is not None: wait_time = max(wait_time, retry_after / 1000.0 + 0.5)
             main_logger.warning(f"{NEON_YELLOW}Rate Limit Exceeded in main loop: {e}. Waiting {wait_time:.1f}s.{RESET}")
             time.sleep(wait_time)
        except ccxt.ExchangeNotAvailable as e:
            main_logger.error(f"{NEON_RED}Exchange Not Available: {e}. Retrying after significant delay...{RESET}")
            time.sleep(LOOP_DELAY_SECONDS * 5)
        except ccxt.ExchangeError as e:
            bybit_code = getattr(e, 'code', None)
            err_str = str(e).lower()
            main_logger.error(f"{NEON_RED}Exchange Error encountered in main loop: {e} (Code: {bybit_code}){RESET}")
            if bybit_code == 10016 or "system maintenance" in err_str:
                main_logger.warning(f"{NEON_YELLOW}Exchange likely in maintenance (Code: {bybit_code}). Waiting longer...{RESET}")
                time.sleep(LOOP_DELAY_SECONDS * 10)
            elif bybit_code == 10006:  # Request timeout
                 main_logger.warning(f"{NEON_YELLOW}Exchange request timeout (Code: {bybit_code}). Retrying after delay...{RESET}")
                 time.sleep(RETRY_DELAY_SECONDS * 5)
            else:
                time.sleep(RETRY_DELAY_SECONDS * 3)  # Moderate delay for other errors
        except Exception as e:
            main_logger.error(f"{NEON_RED}An unexpected critical error occurred in the main loop: {e}{RESET}", exc_info=True)
            main_logger.error("Bot encountered a potentially fatal error. For safety, the bot will stop.")
            break  # Stop the bot

        # --- Loop Delay ---
        main_logger.debug(f"Main loop cycle finished. Sleeping for {LOOP_DELAY_SECONDS} seconds...")
        time.sleep(LOOP_DELAY_SECONDS)

    # --- End of Main Loop ---
    main_logger.info("*** Live Trading Bot has stopped. ***")


# --- Entry Point ---
if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            logger = get_logger("main")
            logger.critical("Unhandled exception caused script termination.", exc_info=True)
        except Exception:
             import traceback
             traceback.print_exc()
    finally:
         pass
