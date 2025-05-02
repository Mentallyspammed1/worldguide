# -*- coding: utf-8 -*-
"""
Pyrmethus Volumatic Trend + OB Trading Bot (CCXT Async Version - Enhanced)

Disclaimer: Trading involves risk. This bot is provided for educational purposes
and demonstration. Use at your own risk. Test thoroughly on paper/testnet before
using real funds. Ensure you understand the strategy and code.
"""
import asyncio
import datetime
import json
import logging
import os
import signal
import sys
import time
from decimal import ROUND_DOWN, ROUND_UP, Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TypedDict, Union

# Third-party imports
import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
from colorama import Fore, Style, init
from dotenv import load_dotenv

# --- Initialize Colorama & Decimal Precision ---
init(autoreset=True)
# Set high decimal precision globally for all Decimal operations
# Adjust as needed, but 28 is usually sufficient for crypto prices/quantities
getcontext().prec = 28

# --- Constants ---
# Define constants for signal strings to avoid typos
SIGNAL_BUY = "BUY"
SIGNAL_SELL = "SELL"
SIGNAL_EXIT_LONG = "EXIT_LONG"
SIGNAL_EXIT_SHORT = "EXIT_SHORT"
SIGNAL_HOLD = "HOLD"
SIDE_BUY = "Buy"  # Internal representation for long side
SIDE_SELL = "Sell" # Internal representation for short side
SIDE_NONE = "None" # Internal representation for flat
ORDER_SIDE_BUY = "buy"   # CCXT representation for buy order
ORDER_SIDE_SELL = "sell" # CCXT representation for sell order
ORDER_TYPE_MARKET = "market"
ORDER_TYPE_LIMIT = "limit"
MODE_PAPER = "paper"
MODE_LIVE = "live" # Assume anything not 'paper' is live/testnet for execution logic

# --- Logging Setup ---
# Define logger at module level before potential early exits
log = logging.getLogger("PyrmethusVolumaticBotCCXT")
# Default level, will be overridden by config later if setup_logging is called again
log_level = logging.INFO
log.setLevel(log_level)
# Basic console handler for early messages before full config
_initial_ch = logging.StreamHandler()
_initial_ch.setLevel(log_level)
_initial_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
_initial_ch.setFormatter(_initial_formatter)
log.addHandler(_initial_ch)
log.propagate = False # Prevent propagation to root logger


# Import strategy class and type hints
# Assuming strategy.py defines VolumaticOBStrategy and possibly AnalysisResults TypedDict
try:
    from strategy import VolumaticOBStrategy
    # Define a placeholder if not defined in strategy.py, ensuring all expected keys exist
    # Ensure this matches the actual return type of strategy.update()
    class AnalysisResults(TypedDict, total=False):
        last_signal: str # 'BUY', 'SELL', 'EXIT_LONG', 'EXIT_SHORT', 'HOLD'
        last_close: float
        last_atr: Optional[float] # Can be None or NaN initially
        current_trend: Optional[bool] # True=UP, False=DOWN, None=Undetermined
        active_bull_boxes: List[Dict]
        active_bear_boxes: List[Dict]
        # Add other fields returned by strategy.update if needed
except ImportError:
    log.critical("CRITICAL: Failed to import VolumaticOBStrategy from strategy.py. Ensure the file exists and is correct.")
    # Define dummy class and type hint to prevent further NameErrors if script somehow continues
    # (shouldn't happen due to sys.exit below, but defensive programming)
    class VolumaticOBStrategy: # type: ignore
        def __init__(self, *args, **kwargs):
            log.critical("Using dummy VolumaticOBStrategy due to import failure.")
            self.price_tick = Decimal("0.01")
            self.min_data_len = 50
            self.price_precision = 2
            self.amount_precision = 8
        def update(self, df: pd.DataFrame) -> Optional['AnalysisResults']:
            log.error("Dummy strategy update called - returning None.")
            return None
        def round_amount(self, amount: Union[float, Decimal]) -> float: return float(f"{Decimal(str(amount)):.{self.amount_precision}f}")
        def round_price(self, price: Union[float, Decimal]) -> float: return float(f"{Decimal(str(price)):.{self.price_precision}f}")
        def format_amount(self, amount: Union[float, Decimal]) -> str: return f"{Decimal(str(amount)):.{self.amount_precision}f}"
        def format_price(self, price: Union[float, Decimal]) -> str: return f"{Decimal(str(price)):.{self.price_precision}f}"
    class AnalysisResults(TypedDict, total=False): pass # type: ignore
    sys.exit(1) # Ensure exit after critical import failure


# --- Load Environment Variables ---
load_dotenv()
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
# Read testnet flag, default to False if not set or invalid
TESTNET_STR = os.getenv("BYBIT_TESTNET", "False").lower()
TESTNET = TESTNET_STR == "true" # Corrected logic: True if "true"

# --- Global Variables ---
config: Dict[str, Any] = {}
exchange: Optional[ccxt.Exchange] = None # Type hint for CCXT exchange instance
strategy_instance: Optional[VolumaticOBStrategy] = None
market: Optional[Dict[str, Any]] = None # CCXT market structure
latest_dataframe: Optional[pd.DataFrame] = None # Holds OHLCV data
# Store position info using Decimal for precision
current_position: Dict[str, Any] = {"size": Decimal(0), "side": SIDE_NONE, "entry_price": Decimal(0), "timestamp": 0.0}
last_position_check_time: float = 0.0 # Track REST API calls (monotonic time)
last_health_check_time: float = 0.0 # Track health checks (monotonic time)
last_ws_update_time: float = 0.0 # Track WebSocket health (monotonic time)
startup_time: float = 0.0 # Record bot startup time
running_tasks: Set[asyncio.Task] = set() # Store background tasks for cancellation
stop_event = asyncio.Event() # Event to signal shutdown

# --- Locks for Shared Resources ---
# Use asyncio locks to prevent race conditions in async operations
data_lock = asyncio.Lock() # Protects latest_dataframe
position_lock = asyncio.Lock() # Protects current_position and REST position fetches/updates
order_lock = asyncio.Lock() # Protects order placement/cancellation/closing

# --- Logging Setup Function ---
def setup_logging(level_str: str = "INFO"):
    """Configures logging for the application based on config."""
    global log_level, _initial_ch
    try:
        log_level = getattr(logging, level_str.upper(), logging.INFO)
    except AttributeError:
        print(f"Warning: Invalid log level '{level_str}' in config. Using INFO.")
        log_level = logging.INFO

    log.setLevel(log_level)

    # Remove initial basic handler if it exists and we haven't already removed it
    if _initial_ch and _initial_ch in log.handlers:
        log.removeHandler(_initial_ch)
        _initial_ch = None # type: ignore

    # Prevent adding multiple handlers if re-configured
    if not log.handlers:
        # Console Handler with Color
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ch.setFormatter(formatter)
        log.addHandler(ch)

        # Optional File Handler (configure path as needed)
        log_filename_template = config.get("log_filename") # Get filename template from config
        if log_filename_template:
            try:
                # Append timestamp to filename if desired (e.g., using placeholder)
                if "{timestamp}" in log_filename_template:
                    timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    log_filename = log_filename_template.format(timestamp=timestamp_str)
                else:
                    log_filename = log_filename_template

                log_path = Path(log_filename).resolve() # Use pathlib for path manipulation

                # Ensure log directory exists
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log.info(f"Ensured log directory exists: {log_path.parent}")

                fh = logging.FileHandler(log_path, encoding='utf-8') # Specify encoding
                fh.setLevel(log_level)
                fh.setFormatter(formatter)
                log.addHandler(fh)
                log.info(f"Logging to file: {log_path}")
            except Exception as e:
                log.error(f"Failed to set up file logging to '{log_filename_template}': {e}", exc_info=True)

    # Prevent log messages from propagating to the root logger
    log.propagate = False

# --- Configuration Loading ---
def load_config(path_str: str = "config.json") -> Dict[str, Any]:
    """Loads configuration from a JSON file with validation."""
    config_path = Path(path_str).resolve()
    try:
        with config_path.open('r', encoding='utf-8') as f: # Specify encoding
            conf = json.load(f)

            # --- Basic Validation ---
            required_keys = ["exchange", "symbol", "timeframe", "order", "strategy", "data", "checks", "mode", "log_level"]
            missing_keys = [key for key in required_keys if key not in conf]
            if missing_keys:
                 raise ValueError(f"Config file '{config_path}' missing required top-level keys: {', '.join(missing_keys)}")

            # --- Nested Validation (Examples) ---
            strategy_conf = conf.get('strategy', {})
            if 'params' not in strategy_conf:
                 raise ValueError("Config file missing 'strategy.params'.")
            if 'stop_loss' not in strategy_conf:
                 raise ValueError("Config file missing 'strategy.stop_loss'.")
            if 'method' not in strategy_conf.get('stop_loss', {}):
                 raise ValueError("Config file missing 'strategy.stop_loss.method'.")

            order_conf = conf.get('order', {})
            required_order_keys = ['type', 'risk_per_trade_percent', 'leverage', 'tp_ratio']
            missing_order_keys = [k for k in required_order_keys if k not in order_conf]
            if missing_order_keys:
                 raise ValueError(f"Config file missing required keys in 'order': {', '.join(missing_order_keys)}")

            checks_conf = conf.get('checks', {})
            required_check_keys = ['health_check_interval', 'position_check_interval', 'ws_timeout_factor']
            missing_check_keys = [k for k in required_check_keys if k not in checks_conf]
            if missing_check_keys:
                 raise ValueError(f"Config file missing required keys in 'checks': {', '.join(missing_check_keys)}")

            # Validate numeric types where expected and positivity
            try:
                risk_percent = float(order_conf['risk_per_trade_percent'])
                leverage = str(order_conf['leverage']) # Leverage validated later
                tp_ratio = float(order_conf['tp_ratio'])
                health_interval = int(checks_conf['health_check_interval'])
                pos_interval = int(checks_conf['position_check_interval'])
                ws_timeout_factor = float(checks_conf['ws_timeout_factor'])

                if not (risk_percent > 0): raise ValueError("'order.risk_per_trade_percent' must be positive.")
                if not (tp_ratio > 0): raise ValueError("'order.tp_ratio' must be positive.")
                if not (health_interval > 0): raise ValueError("'checks.health_check_interval' must be positive.")
                if not (pos_interval > 0): raise ValueError("'checks.position_check_interval' must be positive.")
                if not (ws_timeout_factor > 0): raise ValueError("'checks.ws_timeout_factor' must be positive.")
                # Validate leverage format roughly (numeric string or number)
                float(leverage) # Check if convertible to float

            except (ValueError, TypeError, KeyError) as e:
                raise ValueError(f"Invalid numeric, missing, or non-positive value in config: {e}")

            log.info(f"Configuration loaded and validated successfully from '{config_path}'.")
            return conf
    except FileNotFoundError:
        log.critical(f"CRITICAL: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except (json.JSONDecodeError, ValueError) as e:
        log.critical(f"CRITICAL: Error loading or validating configuration '{config_path}': {e}")
        sys.exit(1)
    except Exception as e:
        log.critical(f"CRITICAL: Unexpected error loading configuration: {e}", exc_info=True)
        sys.exit(1)

# --- CCXT Exchange Interaction (Async) ---
async def connect_ccxt() -> Optional[ccxt.Exchange]:
    """Initializes and connects to the CCXT exchange."""
    global exchange # Allow modification of the global variable
    exchange_id = config.get('exchange', 'bybit').lower()
    # Use account type from config, default to 'unified' for Bybit V5 flexibility
    account_type = config.get('account_type', 'unified').lower() # unified, contract, spot etc.

    if not API_KEY or not API_SECRET:
        log.critical("CRITICAL: API Key or Secret not found in environment variables (.env file).")
        return None

    if not hasattr(ccxt, exchange_id):
        log.critical(f"CRITICAL: Exchange '{exchange_id}' is not supported by CCXT.")
        return None

    try:
        log.info(f"Connecting to CCXT exchange '{exchange_id}' (Account: {account_type}, Testnet: {TESTNET})...")
        exchange_options = {
            'defaultType': account_type,
            'adjustForTimeDifference': True,
            # Bybit V5 specific options (check CCXT/Bybit docs for latest)
            # 'enableUnifiedMargin': account_type == 'unified', # Might be implicitly handled by defaultType='unified'
            # 'enableUnifiedAccount': account_type == 'unified', # Might be implicitly handled
            'recvWindow': config.get('ccxt_options', {}).get('recvWindow', 10000), # Increase recvWindow if timestamp errors occur (default 5000)
            # Add other exchange-specific options here from config if needed
            **(config.get('ccxt_options', {}).get('extra_options', {}))
        }
        # Filter out None values from options if any were conditional
        exchange_options = {k: v for k, v in exchange_options.items() if v is not None}

        exchange_config = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True, # Enable CCXT's built-in rate limiter
            'options': exchange_options,
            # 'asyncio_loop': asyncio.get_event_loop(), # DEPRECATED: Remove this line
        }

        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class(exchange_config)

        if TESTNET:
            if hasattr(exchange, 'set_sandbox_mode'):
                log.warning("Attempting to enable Testnet (Sandbox) mode via set_sandbox_mode(True).")
                try:
                    # Some exchanges might require this method to use testnet URLs
                    exchange.set_sandbox_mode(True)
                except Exception as e:
                    log.error(f"Failed to enable sandbox mode via set_sandbox_mode: {e}. Ensure TESTNET API keys are used if needed.")
            else:
                log.warning(f"Exchange '{exchange_id}' may not support set_sandbox_mode via CCXT. Ensure TESTNET API keys/URLs are used if applicable.")

        # Test connection by loading markets (also fetches server time)
        log.info("Loading markets to test connection...")
        await exchange.load_markets()
        log.info(f"{Fore.GREEN}Successfully connected to {exchange.name}. Loaded {len(exchange.markets)} markets.{Style.RESET_ALL}")

        # Check required capabilities after connection
        # Base requirements for most trading bots
        required_methods = ['fetchOHLCV', 'fetchPositions', 'fetchBalance', 'createOrder', 'cancelOrder', 'fetchOpenOrders']
        critical_methods = ['fetchOHLCV', 'fetchPositions', 'fetchBalance', 'createOrder'] # Bot cannot function without these

        # Conditional requirements based on config
        if config.get('websockets', {}).get('watch_klines', True): # Check config if WS is intended
             required_methods.append('watchOHLCV')
             critical_methods.append('watchOHLCV') # Kline WS is critical for this bot design
        if config.get('websockets', {}).get('watch_positions', True):
             required_methods.append('watchPositions')
        if config.get('websockets', {}).get('watch_orders', True):
             required_methods.append('watchOrders')
        if config['order']['type'].lower() == ORDER_TYPE_LIMIT:
             # Limit orders might need fetchTicker or fetchOrderBook for price reference
             required_methods.append('fetchTicker') # Assume fetchTicker needed for limit orders
        if config['order'].get('leverage'):
             required_methods.append('setLeverage') # Check if leverage setting is intended
        if exchange.has.get('cancelAllOrders') is False and exchange.has.get('cancelOrders') is False:
            # If neither cancel method is available, closing positions might be problematic
            log.warning("Exchange supports neither cancelAllOrders nor cancelOrders. Closing positions might leave SL/TP orders active.")

        missing_methods = [method for method in required_methods if not exchange.has.get(method)]
        missing_critical = [method for method in critical_methods if not exchange.has.get(method)]

        if missing_methods:
             log.warning(f"Exchange {exchange.name} might be missing required capabilities for this bot's configuration: {', '.join(missing_methods)}")

        if missing_critical:
             log.critical(f"CRITICAL: Exchange is missing essential methods: {', '.join(missing_critical)}. Cannot continue.")
             await exchange.close()
             return None

        return exchange
    except ccxt.AuthenticationError as e:
        log.critical(f"CRITICAL: CCXT Authentication Error: {e}. Check API keys, permissions, and ensure IP whitelist (if used) is correct.")
        return None
    except ccxt.NetworkError as e:
         log.critical(f"CRITICAL: CCXT Network Error connecting to exchange: {e}. Check internet connection and firewall.")
         return None
    except Exception as e:
        log.critical(f"CRITICAL: Failed to initialize CCXT exchange: {e}", exc_info=True)
        # Ensure exchange object is cleaned up if partially created
        if exchange and hasattr(exchange, 'close'):
            try: await exchange.close()
            except: pass # Ignore errors during cleanup on critical failure
        exchange = None
        return None

async def load_exchange_market(symbol: str) -> Optional[Dict[str, Any]]:
    """Loads or re-loads market data for a specific symbol."""
    global market # Allow modification of the global variable
    if not exchange:
        log.error("Cannot load market, exchange not connected.")
        return None
    try:
        await exchange.load_markets(True) # Force reload to get latest info
        if symbol in exchange.markets:
            market = exchange.markets[symbol]
            # Validate essential market data
            if not market or not market.get('precision') or not market.get('limits'):
                 log.error(f"Market data for {symbol} is incomplete or missing precision/limits.")
                 return None
            if market.get('active') is False:
                 log.warning(f"Market {symbol} is marked as inactive on the exchange.")
                 # Decide whether to proceed or exit based on configuration/strategy needs
                 # For now, we'll allow proceeding but log a warning.
                 # return None # Uncomment to exit if inactive market is critical

            log.info(f"Market data loaded/updated for {symbol}.")
            # Log key details at DEBUG level
            precision = market.get('precision', {})
            limits = market.get('limits', {})
            amount_limits = limits.get('amount', {})
            cost_limits = limits.get('cost', {})
            leverage_limits = limits.get('leverage', {})
            log.debug(f"Market Details ({symbol}):\n"
                      f"  Precision: Price={precision.get('price')}, Amount={precision.get('amount')}\n"
                      f"  Limits: Amount(min={amount_limits.get('min')}, max={amount_limits.get('max')}), "
                      f"Cost(min={cost_limits.get('min')}, max={cost_limits.get('max')}), "
                      f"Leverage(max={leverage_limits.get('max')})\n"
                      f"  Type: {market.get('type')}, Contract: {market.get('contract', False)}, Linear/Inverse: {market.get('linear', 'N/A')}/{market.get('inverse', 'N/A')}")
            return market
        else:
            log.error(f"Symbol '{symbol}' not found in loaded markets for {exchange.name}.")
            available_symbols = list(exchange.markets.keys())
            # Show a sample of available symbols, ensure list is not excessively long
            sample_size = min(10, len(available_symbols))
            log.error(f"Available symbols sample ({sample_size}/{len(available_symbols)}): {available_symbols[:sample_size]}...")
            return None
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        log.error(f"Failed to load market data for {symbol}: {e}")
        return None
    except Exception as e:
        log.error(f"Unexpected error loading market data for {symbol}: {e}", exc_info=True)
        return None

async def fetch_initial_data(symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetches historical OHLCV data using CCXT."""
    if not exchange:
        log.error("Cannot fetch data, exchange not connected.")
        return None
    if not exchange.has.get('fetchOHLCV'):
        log.error(f"Cannot fetch data: Exchange {exchange.name} does not support fetchOHLCV.")
        return None

    log.info(f"Fetching initial {limit} candles for {symbol} ({timeframe})...")
    try:
        # CCXT fetch_ohlcv returns list: [[timestamp, open, high, low, close, volume]]
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not ohlcv:
            log.warning(f"Received empty list from fetch_ohlcv for {symbol}, {timeframe}. No initial data.")
            # Return an empty DataFrame with correct columns and index type
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'], index=pd.to_datetime([], utc=True))

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # Convert timestamp to datetime and set as index (UTC is standard for CCXT timestamps)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.set_index('timestamp')
        # Ensure numeric types (use float for price/volume)
        for col in ['open', 'high', 'low', 'close', 'volume']:
             # Use errors='coerce' to turn invalid parsing into NaN
             df[col] = pd.to_numeric(df[col], errors='coerce')

        # Check for NaNs which might indicate gaps or exchange issues
        if df.isnull().values.any():
            nan_counts = df.isnull().sum()
            log.warning(f"NaN values found in fetched OHLCV data:\n{nan_counts[nan_counts > 0]}")
            # Option: Fill NaNs (e.g., forward fill) or drop rows, depending on strategy needs
            # df = df.ffill() # Example: Forward fill
            # Drop rows missing essential price data, keep volume NaNs if strategy handles them
            df_original_len = len(df)
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            if len(df) < df_original_len:
                log.warning(f"Dropped {df_original_len - len(df)} rows with NaN price data.")

        if df.empty:
             log.error(f"DataFrame became empty after handling NaNs for {symbol}, {timeframe}.")
             return None

        log.info(f"Fetched {len(df)} initial candles. From {df.index.min()} to {df.index.max()}")
        return df
    except ccxt.NetworkError as e:
        log.error(f"Network error fetching initial klines: {e}")
        return None
    except ccxt.ExchangeError as e:
         log.error(f"Exchange error fetching initial klines: {e}")
         # Check if the error is about the symbol/timeframe combination
         error_str = str(e).lower()
         if "not supported" in error_str or "invalid" in error_str or "doesn't exist" in error_str:
             log.error(f"The symbol '{symbol}' or timeframe '{timeframe}' might not be supported by {exchange.name} for OHLCV fetching.")
         return None
    except Exception as e:
        log.error(f"Unexpected error fetching initial klines: {e}", exc_info=True)
        return None

async def get_current_position(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetches and updates the current position state asynchronously via REST API using Decimal.
    Returns the updated state dict or None if a critical error occurs preventing state determination.
    Uses position_lock internally.
    """
    global current_position, last_position_check_time
    if not exchange or not market:
        log.warning("Cannot get position, exchange or market not ready.")
        return current_position # Return last known state if connection isn't ready

    # Rate limit REST checks to avoid hitting API limits
    now_mono = time.monotonic()
    check_interval = config.get('checks', {}).get('position_check_interval', 30)
    if now_mono - last_position_check_time < check_interval:
        # log.debug("Position check skipped (rate limit). Returning cached state.")
        return current_position # Return last known state

    log.debug(f"Fetching position for {symbol} via REST...")
    async with position_lock: # Lock to prevent concurrent updates from REST/WS
        # Update check time immediately inside lock before API call
        # This prevents rapid retries if the API call hangs or fails quickly
        last_position_check_time = now_mono
        try:
            # Use fetch_positions for Bybit V5 (even for single symbol) as fetch_position might be deprecated/different
            # Check if the method exists first
            if not exchange.has.get('fetchPositions'):
                 log.error(f"Exchange {exchange.name} does not support fetchPositions. Cannot determine position state.")
                 return None # Critical failure

            all_positions = await exchange.fetch_positions([symbol])

            pos_data = None
            if all_positions:
                # Find the position matching the exact symbol (e.g., 'BTC/USDT:USDT')
                for p in all_positions:
                    # CCXT usually standardizes 'symbol', but check 'info' if needed
                    if p and p.get('symbol') == symbol:
                        pos_data = p
                        break

            # --- Check if pos_data was found before accessing ---
            if pos_data:
                # Parse position data carefully - structure varies! Use Decimal.
                # Common fields: 'contracts' (size in base), 'contractSize' (value of 1 contract),
                # 'side' ('long'/'short'), 'entryPrice', 'leverage', 'unrealizedPnl', 'initialMargin', etc.
                # Use .get() with defaults and handle potential None or empty strings
                size_str = str(pos_data.get('contracts', '0') or '0') # Size in base currency (e.g., BTC)
                side = str(pos_data.get('side', 'none') or 'none').lower() # 'long', 'short', or 'none'
                entry_price_str = str(pos_data.get('entryPrice', '0') or '0')

                # Safely convert to Decimal, handling potential errors
                try:
                    # Use string conversion for robustness against float inaccuracies
                    size = Decimal(size_str)
                    entry_price = Decimal(entry_price_str)
                    # Position is considered flat if size is effectively zero (or negative, which indicates error)
                    # Use a small tolerance for floating point comparisons if size wasn't string? No, size should be precise.
                    is_flat = size.is_zero() or size < Decimal(0)
                except InvalidOperation:
                    log.error(f"Error converting position data to Decimal: size='{size_str}', entry='{entry_price_str}'. Assuming flat.")
                    size = Decimal(0)
                    entry_price = Decimal(0)
                    side = 'none'
                    is_flat = True

                # Determine side string based on parsed side and flatness
                if is_flat:
                     current_side_str = SIDE_NONE
                     size = Decimal(0) # Ensure size is zero if flat
                     entry_price = Decimal(0) # Reset entry price if flat
                elif side == 'long':
                     current_side_str = SIDE_BUY
                elif side == 'short':
                     current_side_str = SIDE_SELL
                else: # Handle unexpected 'side' values when size > 0
                     log.warning(f"Position has size {size} but side is '{side}'. Interpreting as '{SIDE_NONE}'. Exchange data: {pos_data.get('info')}")
                     current_side_str = SIDE_NONE
                     # Might need intervention or treating as flat depending on exchange behavior
                     size = Decimal(0)
                     entry_price = Decimal(0)


                # Update global state
                new_position_state = {
                    "size": size,
                    "side": current_side_str,
                    "entry_price": entry_price,
                    "timestamp": time.time() # Record time of successful check
                }

                # Log change if significant (size or side changed)
                # Compare Decimal objects directly
                if new_position_state["size"] != current_position["size"] or new_position_state["side"] != current_position["side"]:
                     log.info(f"Updated Position State: Side={new_position_state['side']}, Size={new_position_state['size']}, Entry={new_position_state['entry_price']}")
                     current_position = new_position_state
                else:
                     # Update timestamp even if state is the same, but log less verbosely
                     current_position["timestamp"] = new_position_state["timestamp"]
                     log.debug(f"Position state confirmed: Side={current_position['side']}, Size={current_position['size']}")

            else: # No position found for the symbol
                 if current_position['size'] != Decimal(0):
                     log.info(f"Position for {symbol} now reported as flat (previously {current_position['side']} {current_position['size']}).")
                     # Update global state to flat
                     current_position = {"size": Decimal(0), "side": SIDE_NONE, "entry_price": Decimal(0), "timestamp": time.time()}
                 else:
                     log.debug(f"No active position found for {symbol}.")
                     # Update timestamp even if already flat
                     current_position["timestamp"] = time.time()

            # last_position_check_time already updated at start of lock
            return current_position # Return the updated state

        except ccxt.NetworkError as e:
            log.warning(f"Network error fetching position: {e}. Returning last known state: {current_position}")
            # Return cached state on temporary network issues, signal processing should handle this
            return current_position
        except ccxt.ExchangeError as e:
            log.error(f"Exchange error fetching position: {e}. Cannot reliably determine position state.")
            # Returning None signals to callers (like process_signals) that the state is unknown
            # and they should likely skip actions based on potentially stale data.
            return None
        except Exception as e:
            # Catch other unexpected errors.
            log.error(f"Unexpected error fetching/parsing position: {e}", exc_info=True)
            log.warning(f"Returning None due to unexpected error, cannot determine position state.")
            # Returning None is safer than returning potentially incorrect cached state
            return None


async def get_wallet_balance(quote_currency: str = "USDT") -> Decimal:
    """Fetches available equity/balance in the specified quote currency using Decimal. Returns Decimal(0) on failure."""
    if not exchange: return Decimal(0)
    log.debug(f"Fetching wallet balance for {quote_currency}...")
    balance_data = None # Initialize to prevent UnboundLocalError in except block
    try:
        # fetch_balance structure depends heavily on exchange and account type
        if not exchange.has.get('fetchBalance'):
             log.error(f"Exchange {exchange.name} does not support fetchBalance. Cannot get balance.")
             return Decimal(0)

        balance_data = await exchange.fetch_balance()

        # --- Adapt parsing based on expected structure (e.g., Bybit Unified/Contract V5) ---
        total_equity = Decimal(0)
        free_balance = Decimal(0)

        # Try standard CCXT structure first
        if quote_currency in balance_data.get('total', {}):
            try:
                total_equity_val = balance_data['total'][quote_currency]
                if total_equity_val is not None: # Check for None before converting
                    total_equity = Decimal(str(total_equity_val)) # Use string conversion
                    log.debug(f"Parsed CCXT 'total' balance: {total_equity}")
            except (InvalidOperation, TypeError):
                log.warning(f"Could not parse CCXT total balance for {quote_currency}: {balance_data['total'].get(quote_currency)}")
        if quote_currency in balance_data.get('free', {}):
            try:
                free_balance_val = balance_data['free'][quote_currency]
                if free_balance_val is not None: # Check for None before converting
                    free_balance = Decimal(str(free_balance_val)) # Use string conversion
                    log.debug(f"Parsed CCXT 'free' balance: {free_balance}")
            except (InvalidOperation, TypeError):
                log.warning(f"Could not parse CCXT free balance for {quote_currency}: {balance_data['free'].get(quote_currency)}")

        # Try Bybit V5 specific 'info' structure (might vary)
        # Example path for Unified: balance_data['info']['result']['list'][0]['totalEquity']
        # Example path for Contract: balance_data['info']['result']['list'][0]['equity'] (or walletBalance)
        bybit_specific_equity = Decimal(0)
        if 'info' in balance_data and isinstance(balance_data['info'], dict):
             result = balance_data['info'].get('result', {})
             if isinstance(result, dict) and 'list' in result and isinstance(result['list'], list) and len(result['list']) > 0:
                 log.debug(f"Checking Bybit 'info.result.list' for balance (found {len(result['list'])} items)...")
                 # Find the account info matching the quote currency (e.g., USDT)
                 account_info = None
                 for account in result['list']:
                     # Unified/Spot accounts often list by 'coin'
                     if isinstance(account, dict) and account.get('coin') == quote_currency:
                         log.debug(f"Found account info matching coin '{quote_currency}': {account}")
                         account_info = account
                         break
                     # Contract accounts might not have 'coin' per item, check common equity keys
                     elif isinstance(account, dict) and 'coin' not in account:
                          equity_keys_check = ['totalEquity', 'equity', 'walletBalance']
                          if any(key in account for key in equity_keys_check):
                              log.debug(f"Found account info without specific coin, checking equity keys: {account}")
                              account_info = account
                              # Assume first relevant dict is the main one for contract
                              break

                 if isinstance(account_info, dict):
                     # Check multiple possible keys for equity/balance in Bybit V5 response
                     equity_keys = ['totalEquity', 'equity', 'walletBalance'] # Order preference? totalEquity > equity > walletBalance?
                     for key in equity_keys:
                         if key in account_info:
                             log.debug(f"Checking Bybit balance key '{key}'...")
                             try:
                                 equity_val = account_info[key]
                                 # Handle potential None or empty string values
                                 if equity_val is not None and equity_val != '':
                                     bybit_specific_equity = Decimal(str(equity_val)) # Use string conversion
                                     if bybit_specific_equity > Decimal(0): # Compare with Decimal(0)
                                         log.debug(f"Found valid Bybit balance using key '{key}': {bybit_specific_equity}")
                                         break # Use the first valid one found
                                 else:
                                     log.debug(f"Bybit balance key '{key}' found but value is None or empty.")
                             except (InvalidOperation, TypeError):
                                 log.warning(f"Could not parse Bybit balance key '{key}': {account_info.get(key)}")
                                 continue
                         else:
                             log.debug(f"Bybit balance key '{key}' not found in account info.")
                 else:
                     log.debug(f"No specific account info found in Bybit 'info.result.list' matching '{quote_currency}' or generic contract structure.")
             else:
                 log.debug("Bybit 'info.result.list' structure not found or empty.")
        else:
             log.debug("Bybit 'info' structure not found in balance data.")


        # --- Determine which balance to use ---
        # Prefer Bybit specific equity if found and positive
        if bybit_specific_equity > Decimal(0):
            log.debug(f"Using Bybit specific equity: {bybit_specific_equity} {quote_currency}")
            return bybit_specific_equity
        # Fallback to CCXT total equity
        elif total_equity > Decimal(0):
            log.debug(f"Using CCXT 'total' balance: {total_equity} {quote_currency}")
            return total_equity
        # As last resort, use free balance, but warn
        elif free_balance > Decimal(0):
             log.warning(f"Total equity not found or zero, using 'free' balance: {free_balance} {quote_currency}. This might underestimate risk capacity if margin is used.")
             return free_balance
        else:
             log.error(f"Could not determine a valid balance/equity for {quote_currency}. Found 0 or less.")
             log.debug(f"Full balance data: {balance_data}") # Log full structure for debugging
             return Decimal(0) # Return 0 if no balance found

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        log.error(f"Could not fetch wallet balance: {e}")
        return Decimal(0) # Indicate failure to fetch by returning 0
    except (InvalidOperation, TypeError) as e:
         log.error(f"Error converting balance data to Decimal: {e}")
         log.debug(f"Balance data causing conversion error: {balance_data}")
         return Decimal(0)
    except Exception as e:
        log.error(f"Unexpected error fetching balance: {e}", exc_info=True)
        return Decimal(0)

async def calculate_order_qty(entry_price: float, sl_price: float, risk_percent: float) -> Optional[float]:
    """Calculates order quantity based on risk, SL distance, and equity using Decimal for precision. Returns float."""
    if not market or not strategy_instance:
        log.error("Cannot calculate quantity: Market or strategy instance not available.")
        return None
    # Ensure strategy instance has necessary Decimal attributes/methods
    if not hasattr(strategy_instance, 'price_tick') or not isinstance(strategy_instance.price_tick, Decimal):
         log.error("Strategy instance missing valid 'price_tick' Decimal attribute.")
         return None
    if strategy_instance.price_tick <= Decimal(0):
         log.error("Strategy instance 'price_tick' must be positive.")
         return None
    if not hasattr(strategy_instance, 'round_amount') or not callable(strategy_instance.round_amount):
        log.error("Strategy instance missing callable 'round_amount' method.")
        return None
    # format_amount is used for logging, not critical for calculation
    # if not hasattr(strategy_instance, 'format_amount') or not callable(strategy_instance.format_amount):
    #     log.warning("Strategy instance missing callable 'format_amount' method. Logging may be less precise.")

    # --- Use Decimal for precise calculations ---
    try:
        # Use string conversion for floats to maintain precision
        sl_decimal = Decimal(str(sl_price))
        entry_decimal = Decimal(str(entry_price))
        risk_percent_decimal = Decimal(str(risk_percent))
        price_tick_decimal = strategy_instance.price_tick
    except (InvalidOperation, TypeError, AttributeError) as e:
         log.error(f"Error initializing Decimal values for quantity calculation: {e}")
         return None

    # Ensure SL is meaningfully different from entry, considering price tick size
    # Use a small tolerance (e.g., half a tick)
    min_sl_distance = price_tick_decimal / Decimal(2)
    if abs(sl_decimal - entry_decimal) < min_sl_distance:
        log.error(f"Stop loss price {sl_decimal} is too close to entry price {entry_decimal} (Min Distance: {min_sl_distance}). Cannot calculate quantity.")
        return None

    # Get current equity
    quote_currency = market.get('quote', 'USDT') # e.g., USDT in BTC/USDT
    balance = await get_wallet_balance(quote_currency) # Returns Decimal(0) on failure
    if balance <= Decimal(0):
        log.error(f"Cannot calculate order quantity: Invalid or zero balance ({balance}) for {quote_currency}.")
        return None

    log.debug(f"Calculating Qty: Balance={balance:.4f} {quote_currency}, Risk={risk_percent_decimal}%, Entry={entry_decimal}, SL={sl_decimal}")

    try:
        # Calculate risk amount in quote currency
        risk_amount = balance * (risk_percent_decimal / Decimal(100))
        # Calculate stop loss distance in quote currency per unit of base currency
        sl_distance_per_unit = abs(entry_decimal - sl_decimal)

        if sl_distance_per_unit <= Decimal(0):
            # Should be caught by the earlier check, but safety first
            raise ValueError("Stop loss distance is zero or negative.")

        # Calculate quantity in base asset (e.g., BTC for BTC/USDT)
        # Qty (Base) = Risk Amount (Quote) / SL Distance per Unit (Quote/Base)
        qty_base = risk_amount / sl_distance_per_unit
        log.debug(f"Calculated Raw Qty: {qty_base} (RiskAmt={risk_amount:.4f}, SLDist={sl_distance_per_unit})")

    except (InvalidOperation, ValueError, ZeroDivisionError, TypeError) as e:
        log.error(f"Error during quantity calculation math: {e}")
        log.error(f"Inputs: balance={balance}, risk%={risk_percent_decimal}, entry={entry_decimal}, sl={sl_decimal}")
        return None

    # Round the calculated quantity DOWN to the market's amount precision/tick size
    # Use the strategy's rounding method which should handle market['precision']['amount']
    try:
        # Assuming round_amount returns float as required by CCXT create_order amount
        # Pass the Decimal value to the strategy's rounding method
        # Ensure strategy's round_amount handles rounding direction (e.g., ROUND_DOWN)
        qty_rounded = strategy_instance.round_amount(qty_base) # Expect float return
    except Exception as e:
        log.error(f"Error calling strategy's round_amount method: {e}")
        return None

    log.debug(f"Quantity after rounding: {qty_rounded}")

    # Convert market limits to float for comparison
    min_qty = 0.0
    max_qty = float('inf')
    try:
        min_qty_val = market.get('limits', {}).get('amount', {}).get('min')
        max_qty_val = market.get('limits', {}).get('amount', {}).get('max')
        if min_qty_val is not None: min_qty = float(min_qty_val)
        if max_qty_val is not None: max_qty = float(max_qty_val)
    except (ValueError, TypeError):
        log.warning("Could not parse market amount limits. Using defaults (min=0, max=inf).")

    qty_final = qty_rounded

    # --- Validate against market limits ---
    if qty_final <= 0:
        log.error(f"Calculated order quantity is zero or negative ({qty_final}) after rounding. Min Qty required: {min_qty}. Aborting trade.")
        return None

    if min_qty > 0 and qty_final < min_qty:
        log.warning(f"Calculated qty {qty_final} is below market minimum ({min_qty}).")
        # Option 1: Use min_qty (increases risk) - Current choice
        # Option 2: Abort trade ( Safer: return None )
        qty_final = min_qty
        # Recalculate actual risk if using min_qty
        try:
            # Use Decimal for risk calculation
            actual_risk_amount = Decimal(str(min_qty)) * sl_distance_per_unit
            actual_risk_percent = (actual_risk_amount / balance) * 100 if balance > Decimal(0) else Decimal(0)
            log.warning(f"{Fore.YELLOW}Adjusting order quantity to minimum: {qty_final}. "
                        f"Actual Risk: {actual_risk_amount:.2f} {quote_currency} ({actual_risk_percent:.2f}%){Style.RESET_ALL}")
            # Check if adjusted risk is acceptable (e.g., not > 2x intended risk)
            max_acceptable_risk_mult = Decimal(str(config.get('order', {}).get('max_min_qty_risk_multiplier', 2.0)))
            if actual_risk_percent > (risk_percent_decimal * max_acceptable_risk_mult):
                 log.error(f"Risk after adjusting to min qty ({actual_risk_percent:.2f}%) exceeds acceptable threshold ({risk_percent_decimal * max_acceptable_risk_mult:.2f}%). Aborting trade.")
                 return None
        except (InvalidOperation, TypeError) as e:
             log.error(f"Error calculating risk after adjusting to min qty: {e}. Aborting trade.")
             return None

    elif max_qty > 0 and qty_final > max_qty:
        log.warning(f"Calculated qty {qty_final} exceeds market maximum ({max_qty}). Adjusting down to max.")
        qty_final = max_qty # Use max allowed quantity

    # Final check: Ensure quantity is not zero after adjustments
    if qty_final <= 0:
         log.error(f"Final quantity is zero or negative ({qty_final}) after limit adjustments. Aborting trade.")
         return None

    # Log the final calculated quantity using strategy formatting
    formatted_qty = safe_format('format_amount', qty_final, str(qty_final))

    log.info(f"Calculated Order Qty: {formatted_qty} {market.get('base', '')} "
             f"(Balance={balance:.2f}, Risk={risk_percent}%, TargetRiskAmt={risk_amount:.4f}, FinalQty={qty_final})") # Log exact float qty
    return qty_final # Return float

# Helper function for safe formatting (used in signal processing and order placement)
def safe_format(method_name: str, value: Any, default_str: str = "N/A") -> str:
    """Safely formats a value using a strategy instance method, with fallback."""
    if value is None: return default_str
    if strategy_instance and hasattr(strategy_instance, method_name) and callable(getattr(strategy_instance, method_name)):
        try:
            return str(getattr(strategy_instance, method_name)(value)) # Ensure result is string
        except Exception as e:
            log.warning(f"Error calling strategy formatter '{method_name}' for value '{value}': {e}")
            # Fallback to reasonably formatted string representation
            if isinstance(value, Decimal):
                 return f"{value:.8f}" # Example precision for Decimal
            elif isinstance(value, float):
                 return f"{value:.8f}" # Example precision for float
            else:
                 return str(value)
    else:
        # Log missing method only once? Or rely on earlier checks.
        # log.warning(f"Strategy instance missing '{method_name}' method.")
        # Fallback to reasonably formatted string representation
        if isinstance(value, Decimal):
             return f"{value:.8f}"
        elif isinstance(value, float):
             return f"{value:.8f}"
        else:
             return str(value)

async def place_order(symbol: str, side: str, qty: float, price: Optional[float]=None, sl_price: Optional[float]=None, tp_price: Optional[float]=None) -> Optional[Dict]:
    """Places an order using CCXT create_order with SL/TP params if supported. Returns order dict or None. Uses order_lock."""
    if not exchange or not strategy_instance or not market:
        log.error("Cannot place order: Exchange, strategy, or market not ready.")
        return None
    # Check for required strategy methods/attributes
    if not all(hasattr(strategy_instance, attr) for attr in ['round_amount', 'round_price', 'price_tick']):
        log.error("Cannot place order: Strategy instance missing required methods/attributes (round_amount, round_price, price_tick).")
        return None
    if not isinstance(strategy_instance.price_tick, Decimal) or strategy_instance.price_tick <= Decimal(0):
        log.error("Cannot place order: Strategy instance 'price_tick' must be a positive Decimal.")
        return None
    if not callable(strategy_instance.round_amount) or not callable(strategy_instance.round_price):
        log.error("Cannot place order: Strategy instance 'round_amount' or 'round_price' is not callable.")
        return None

    mode = config.get("mode", MODE_LIVE).lower()
    order_type = config['order']['type'].lower() # 'market' or 'limit'
    # Ensure side is lowercase for CCXT API call ('buy' or 'sell')
    order_side_lower = side.lower()
    if order_side_lower not in [ORDER_SIDE_BUY, ORDER_SIDE_SELL]:
        log.error(f"Invalid order side '{side}' provided to place_order. Must be '{ORDER_SIDE_BUY}' or '{ORDER_SIDE_SELL}'.")
        return None

    # Use strategy's formatting methods for logging
    qty_str = safe_format('format_amount', qty)
    price_str = f" @{safe_format('format_price', price)}" if price and order_type == ORDER_TYPE_LIMIT else ""
    sl_str = f" SL={safe_format('format_price', sl_price)}" if sl_price else ""
    tp_str = f" TP={safe_format('format_price', tp_price)}" if tp_price else ""

    if mode == MODE_PAPER:
        # Simulate order placement in paper trading mode
        log.warning(f"{Fore.YELLOW}[PAPER MODE] Simulating {order_side_lower.upper()} {order_type.upper()} order: "
                    f"{qty_str} {symbol}{price_str}{sl_str}{tp_str}{Style.RESET_ALL}")
        # Simulate immediate fill at desired price or last price
        simulated_fill_price = price if price and order_type == ORDER_TYPE_LIMIT else None
        if not simulated_fill_price:
            try:
                if exchange.has.get('fetchTicker'):
                     ticker = await exchange.fetch_ticker(symbol)
                     simulated_fill_price = ticker.get('last')
                else:
                     log.warning("[PAPER MODE] Cannot fetch ticker for simulated fill price (not supported). Using limit price or 0.")
                     simulated_fill_price = price or 0
            except Exception as e:
                log.error(f"[PAPER MODE] Could not fetch ticker for simulated fill price: {e}")
                simulated_fill_price = price or 0 # Fallback

        if simulated_fill_price is None or simulated_fill_price <= 0:
             log.error("[PAPER MODE] Could not determine a valid simulated fill price. Aborting simulation.")
             return None

        # Update paper position state (needs a dedicated paper trading state manager for accuracy)
        log.info(f"[PAPER MODE] Assuming order filled at ~{safe_format('format_price', simulated_fill_price)}. Update paper position state accordingly.") # Placeholder
        # Simulate a realistic order response
        return {
            "id": f"paper_{int(time.time())}", "timestamp": time.time() * 1000,
            "datetime": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "symbol": symbol, "type": order_type, "side": order_side_lower, # Ensure side is lowercase
            "amount": qty, "filled": qty, "remaining": 0.0,
            "price": price, "average": simulated_fill_price,
            "status": "closed", "fee": None, "cost": qty * simulated_fill_price,
            "info": {"paperTrade": True, "simulated": True, "slPrice": sl_price, "tpPrice": tp_price}
        }

    # --- Live/Testnet Order Placement ---
    async with order_lock: # Prevent concurrent order placements
        # Round amount and price using strategy methods just before placing
        # Ensure methods exist and handle potential errors
        try:
            amount_rounded = strategy_instance.round_amount(qty) # Expects float return
            limit_price_rounded = strategy_instance.round_price(price) if price and order_type == ORDER_TYPE_LIMIT else None # Expects float return
        except Exception as e:
            log.error(f"Error rounding amount/price using strategy methods: {e}. Using raw values.")
            amount_rounded = qty
            limit_price_rounded = price if order_type == ORDER_TYPE_LIMIT else None

        # --- Final Validation Before Placing Order ---
        if amount_rounded <= 0:
             log.error(f"Attempted to place order with zero/negative amount after rounding: {amount_rounded}. Aborting.")
             return None
        # Check min quantity again after final rounding
        min_qty = 0.0
        try:
            min_qty_val = market.get('limits', {}).get('amount', {}).get('min')
            if min_qty_val is not None: min_qty = float(min_qty_val)
        except (ValueError, TypeError): pass # Ignore parsing errors, keep default 0.0

        if min_qty > 0 and amount_rounded < min_qty:
             log.error(f"Final order amount {amount_rounded} is below minimum {min_qty} after rounding. Aborting order.")
             return None
        # Check min cost if applicable (amount * price)
        min_cost = 0.0
        try:
            min_cost_val = market.get('limits', {}).get('cost', {}).get('min')
            if min_cost_val is not None: min_cost = float(min_cost_val)
        except (ValueError, TypeError): pass

        if min_cost > 0:
             current_price_for_cost_check = None
             if limit_price_rounded and order_type == ORDER_TYPE_LIMIT:
                 current_price_for_cost_check = limit_price_rounded
             else: # Market order or missing limit price
                 try:
                     if exchange.has.get('fetchTicker'):
                         ticker = await exchange.fetch_ticker(symbol)
                         current_price_for_cost_check = ticker.get('last')
                     else:
                         log.warning("Cannot check min cost: fetchTicker not supported.")
                 except Exception as e:
                     log.warning(f"Could not fetch ticker to check min cost: {e}")

             if current_price_for_cost_check:
                 try:
                     # Use float for cost calculation as CCXT expects float amount/price
                     order_cost = amount_rounded * float(current_price_for_cost_check)
                     if order_cost < min_cost:
                          log.error(f"Estimated order cost {order_cost:.4f} (Qty:{amount_rounded} * Price:{current_price_for_cost_check}) is below minimum {min_cost}. Aborting order.")
                          return None
                 except (ValueError, TypeError) as e:
                      log.warning(f"Could not calculate order cost for validation: {e}")
             else:
                  log.warning("Could not determine price to check minimum cost limit.")


        # --- Prepare CCXT Params for SL/TP ---
        # Syntax varies significantly by exchange (Bybit V5 Unified/Contract example)
        ccxt_params = {
             # 'positionIdx': 0, # 0: One-Way Mode. Set later from config if needed.
             'timeInForce': config['order'].get('time_in_force', 'GTC'), # GoodTillCancel, ImmediateOrCancel, FillOrKill
             # 'orderLinkId': f'bot_{int(time.time()*1000)}' # Optional client order ID
        }
        # Add positionIdx based on config if needed (e.g., for Bybit Hedge Mode)
        position_idx = config['order'].get('positionIdx')
        if position_idx is not None:
            try:
                ccxt_params['positionIdx'] = int(position_idx)
                log.debug(f"Using positionIdx: {ccxt_params['positionIdx']}")
            except (ValueError, TypeError):
                log.error(f"Invalid positionIdx '{position_idx}' in config. Must be an integer. Ignoring.")


        sl_price_rounded = None
        if sl_price:
            try:
                sl_price_rounded = strategy_instance.round_price(sl_price) # Expects float return
            except Exception as e:
                log.error(f"Error rounding SL price: {e}. Using raw value {sl_price}.")
                sl_price_rounded = sl_price

            # Check if SL price is valid relative to side/market price (basic check)
            # Use limit price for check if available, otherwise fetch ticker (can be slow/race condition)
            check_price = limit_price_rounded if limit_price_rounded else None
            if not check_price and exchange.has.get('fetchTicker'):
                try:
                    ticker = await exchange.fetch_ticker(symbol)
                    check_price = ticker.get('last')
                except Exception: pass # Ignore errors, validation might be less strict

            if check_price and sl_price_rounded:
                # Add a small buffer based on price tick to avoid immediate trigger due to rounding/slippage
                # Ensure price_tick is positive before using
                sl_buffer = float(strategy_instance.price_tick) * 2 # Use 2 ticks buffer
                try:
                    check_price_float = float(check_price)
                    # SL for buy must be below check price, SL for sell must be above
                    if (order_side_lower == ORDER_SIDE_BUY and sl_price_rounded >= (check_price_float - sl_buffer)) or \
                       (order_side_lower == ORDER_SIDE_SELL and sl_price_rounded <= (check_price_float + sl_buffer)):
                        log.error(f"Invalid SL price {sl_price_rounded} relative to order side '{order_side_lower}' and current/limit price '{check_price_float}' (Buffer: {sl_buffer}). Aborting.")
                        return None
                except (ValueError, TypeError) as e:
                    log.warning(f"Could not validate SL price due to type error: {e}")


            if sl_price_rounded:
                # Ensure SL price is not zero or negative after rounding/validation
                if sl_price_rounded <= 0:
                    log.error(f"Invalid SL price {sl_price_rounded} (zero or negative). Aborting.")
                    return None
                ccxt_params.update({
                    'stopLoss': str(sl_price_rounded), # Use string representation for price for robustness
                    'slTriggerBy': config['order'].get('sl_trigger_type', 'LastPrice'), # MarkPrice, IndexPrice, LastPrice (check exchange support)
                    # 'tpslMode': 'Full', # Or 'Partial'. Bybit V5: Affects if SL/TP closes whole position. Default usually 'Full'.
                    # 'slOrderType': 'Market', # Bybit might require specifying SL order type (Market or Limit)
                })
                log.info(f"Prepared SL: Price={sl_price_rounded}, Trigger={ccxt_params['slTriggerBy']}")

        tp_price_rounded = None
        if tp_price:
             try:
                 tp_price_rounded = strategy_instance.round_price(tp_price) # Expects float return
             except Exception as e:
                 log.error(f"Error rounding TP price: {e}. Using raw value {tp_price}.")
                 tp_price_rounded = tp_price

             # Basic validation for TP price
             check_price = limit_price_rounded if limit_price_rounded else None
             # Fetch ticker only if needed and not already fetched for SL check
             if not check_price and exchange.has.get('fetchTicker') and not (sl_price and check_price): # Avoid double fetch
                 try:
                     ticker = await exchange.fetch_ticker(symbol)
                     check_price = ticker.get('last')
                 except Exception: pass

             if check_price and tp_price_rounded:
                 # Add a small buffer based on price tick
                 tp_buffer = float(strategy_instance.price_tick) * 2 # Use 2 ticks buffer
                 try:
                     check_price_float = float(check_price)
                     # TP for buy must be above check price, TP for sell must be below
                     if (order_side_lower == ORDER_SIDE_BUY and tp_price_rounded <= (check_price_float + tp_buffer)) or \
                        (order_side_lower == ORDER_SIDE_SELL and tp_price_rounded >= (check_price_float - tp_buffer)):
                         log.error(f"Invalid TP price {tp_price_rounded} relative to order side '{order_side_lower}' and current/limit price '{check_price_float}' (Buffer: {tp_buffer}). Aborting.")
                         return None
                 except (ValueError, TypeError) as e:
                     log.warning(f"Could not validate TP price due to type error: {e}")

             if tp_price_rounded:
                 # Ensure TP price is not zero or negative
                 if tp_price_rounded <= 0:
                     log.error(f"Invalid TP price {tp_price_rounded} (zero or negative). Setting TP to None.")
                     tp_price_rounded = None # Do not send invalid TP
                 else:
                     ccxt_params.update({
                         'takeProfit': str(tp_price_rounded), # Use string representation
                         'tpTriggerBy': config['order'].get('tp_trigger_type', 'LastPrice'),
                         # 'tpOrderType': 'Market',
                     })
                     log.info(f"Prepared TP: Price={tp_price_rounded}, Trigger={ccxt_params['tpTriggerBy']}")

        # Log the attempt
        log.warning(f"{Fore.CYAN}Attempting to place {order_side_lower.upper()} {order_type.upper()} order:{Style.RESET_ALL}\n"
                    f"  Symbol: {symbol}\n"
                    f"  Amount: {safe_format('format_amount', amount_rounded)} (Raw: {qty})\n"
                    f"  Limit Price: {safe_format('format_price', limit_price_rounded) if limit_price_rounded else 'N/A'}\n"
                    f"  SL Price: {ccxt_params.get('stopLoss', 'N/A')}\n"
                    f"  TP Price: {ccxt_params.get('takeProfit', 'N/A')}\n"
                    f"  Params: {ccxt_params}")

        try:
            # Place the order using ccxt.create_order
            # Ensure required methods exist
            if not exchange.has.get('createOrder'):
                 log.error(f"Order Failed: Exchange {exchange.name} does not support createOrder.")
                 return None

            order = await exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=order_side_lower, # Ensure side is lowercase
                amount=amount_rounded, # Use the final rounded amount (float)
                price=limit_price_rounded, # Pass None for market orders (float or None)
                params=ccxt_params # Pass exchange-specific params here
            )
            log.info(f"{Fore.GREEN}Order placed successfully! ID: {order.get('id')}{Style.RESET_ALL}")
            log.debug(f"Order details: {json.dumps(order, indent=2)}")

            # Force position check soon after placing order to confirm state change
            global last_position_check_time
            # Use lock to safely reset timer
            async with position_lock:
                last_position_check_time = 0

            return order # Return the order details dictionary

        except ccxt.InsufficientFunds as e:
            log.error(f"{Fore.RED}Order Failed: Insufficient Funds.{Style.RESET_ALL} {e}")
            # Log available balance for debugging
            await get_wallet_balance(market['quote'])
            return None
        except ccxt.InvalidOrder as e:
             log.error(f"{Fore.RED}Order Failed: Invalid Order Parameters.{Style.RESET_ALL} Check config, calculations, market limits, and SL/TP validity. {e}")
             log.error(f"Order details attempted: symbol={symbol}, type={order_type}, side={order_side_lower}, amount={amount_rounded}, price={limit_price_rounded}, params={ccxt_params}")
             # Common issues: Price/amount precision, below min size/cost, invalid SL/TP params for the exchange/market, invalid TimeInForce.
             return None
        except ccxt.ExchangeNotAvailable as e:
             log.error(f"{Fore.RED}Order Failed: Exchange Not Available (Maintenance?).{Style.RESET_ALL} {e}")
             # Consider implementing retry logic or pausing trading.
             return None
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            log.error(f"{Fore.RED}Order Failed: {type(e).__name__} - {e}{Style.RESET_ALL}")
            # Consider retrying network errors. Exchange errors might be permanent (e.g., invalid symbol, API key issue).
            error_str = str(e).lower()
            if "margin check failed" in error_str:
                 log.error("Margin check failed - potentially insufficient funds or leverage issue.")
                 await get_wallet_balance(market['quote']) # Log balance
            elif "order cost" in error_str and "too small" in error_str:
                 log.error("Order cost too small - check minimum cost limit for the market.")
            elif "size" in error_str and ("too small" in error_str or "below min" in error_str):
                 log.error("Order size too small - check minimum amount limit for the market.")
            elif "precision" in error_str:
                 log.error("Precision error - check amount or price rounding against market rules.")
            return None
        except Exception as e:
            log.error(f"{Fore.RED}Unexpected error placing order: {e}{Style.RESET_ALL}", exc_info=True)
            return None


async def close_position(symbol: str, position_data: Dict[str, Any]) -> Optional[Dict]:
    """Closes the current position using a reduce-only market order. Cancels existing orders first. Returns closing order dict or None/info dict. Uses order_lock."""
    if not exchange or not strategy_instance or not market:
        log.error("Cannot close position: Exchange, strategy, or market not ready.")
        return None

    mode = config.get("mode", MODE_LIVE).lower()
    # Use Decimal for size comparison
    current_size = position_data.get('size', Decimal(0))
    current_side = position_data.get('side', SIDE_NONE) # Expect 'Buy' or 'Sell' from get_current_position

    if mode == MODE_PAPER:
        if current_size > Decimal(0):
            closing_side = ORDER_SIDE_SELL if current_side == SIDE_BUY else ORDER_SIDE_BUY
            log.warning(f"{Fore.YELLOW}[PAPER MODE] Simulating closing {current_side} position for {symbol} (Size: {current_size}) via {closing_side.upper()} order.{Style.RESET_ALL}")
            # Update paper trading state
            log.info("[PAPER MODE] Assuming position closed. Update paper state.") # Placeholder
            # Simulate a response
            return {
                "id": f"paper_close_{int(time.time())}",
                "timestamp": time.time() * 1000,
                "datetime": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "symbol": symbol, "type": ORDER_TYPE_MARKET, "side": closing_side,
                "amount": float(current_size), "filled": float(current_size), "remaining": 0.0,
                "price": None, "average": None, # Cannot easily simulate fill price here
                "status": "closed", "fee": None, "cost": None,
                "info": {"paperTrade": True, "simulatedClose": True, "reduceOnly": True}
            }
        else:
            log.info("[PAPER MODE] Attempted to close position, but already flat.")
            return {"info": {"alreadyFlat": True, "paperTrade": True}}

    # --- Live/Testnet Position Close ---
    async with order_lock: # Ensure only one closing order attempt at a time
        # Fetch position state *again* right before closing for maximum accuracy
        log.debug("Re-fetching position state just before closing...")
        # Reset check timer to allow immediate fetch
        global last_position_check_time
        async with position_lock: # Use inner lock only to modify check time
             last_position_check_time = 0
        latest_pos_data = await get_current_position(symbol)
        # If fetch fails critically, abort the close attempt
        if latest_pos_data is None:
             log.error("Could not re-fetch position state before closing due to API error. Aborting close.")
             return None

        current_size = latest_pos_data.get('size', Decimal(0))
        current_side = latest_pos_data.get('side', SIDE_NONE)

        if current_size <= Decimal(0) or current_side == SIDE_NONE:
            log.info(f"Attempted to close position for {symbol}, but re-check shows it's already flat.")
            return {"info": {"alreadyFlat": True}}

        # Determine the side and amount for the closing order
        # Closing a BUY position requires a SELL order, closing a SELL position requires a BUY order
        side_to_close = ORDER_SIDE_SELL if current_side == SIDE_BUY else ORDER_SIDE_BUY
        # Use the exact size from the latest fetched position data, converted to float for CCXT
        try:
             amount_to_close = float(current_size)
             if amount_to_close <= 0: # Sanity check
                 raise ValueError("Position size is zero or negative.")
        except (ValueError, TypeError) as e:
             log.error(f"Invalid position size for closing: {current_size}. Error: {e}. Aborting.")
             return None

        # Format amount for logging
        formatted_amount = safe_format('format_amount', amount_to_close, str(amount_to_close))
        log.warning(f"{Fore.YELLOW}Attempting to close {current_side} position for {symbol} (Size: {formatted_amount}). Placing {side_to_close.upper()} Market order...{Style.RESET_ALL}")

        # --- Cancel Existing SL/TP Orders First (Important!) ---
        # This prevents the SL/TP from executing *after* the manual close order.
        # Note: cancel_all_orders might affect other manual orders for the same symbol.
        try:
            log.info(f"Attempting to cancel ALL existing open orders for {symbol} before closing position...")
            # Use cancel_all_orders if supported and appropriate
            if exchange.has.get('cancelAllOrders'):
                cancel_result = await exchange.cancel_all_orders(symbol)
                log.info(f"Cancel all orders result: {cancel_result}") # Log result (might be list of orders or status)
                await asyncio.sleep(0.5) # Short delay to allow cancellation processing on the exchange
            elif exchange.has.get('cancelOrders') and exchange.has.get('fetchOpenOrders'): # Check fetchOpenOrders too
                 log.warning("Exchange does not support cancel_all_orders, attempting to fetch and cancel open orders individually...")
                 open_orders = await exchange.fetch_open_orders(symbol)
                 if open_orders:
                     order_ids = [o['id'] for o in open_orders]
                     log.info(f"Found {len(order_ids)} open orders to cancel: {order_ids}")
                     cancel_results = await exchange.cancel_orders(order_ids, symbol)
                     log.info(f"Individual cancel results: {cancel_results}")
                     await asyncio.sleep(0.5)
                 else:
                     log.info("No open orders found for the symbol to cancel.")
            else:
                log.warning("Exchange does not support efficient cancellation of all/multiple orders via CCXT. Manual cancellation of SL/TP might be needed if they weren't attached to the entry order.")
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            # Log warning but proceed with close attempt - cancellation might fail if no orders exist
            log.warning(f"Could not cancel orders before closing position (might be none or API issue): {e}")
        except Exception as e:
            # Log error but proceed with close attempt
            log.error(f"Unexpected error cancelling orders: {e}", exc_info=True)


        # --- Place the Reduce-Only Market Order ---
        params = {
            'reduceOnly': True,
            # Add positionIdx if needed (from config)
            **( {'positionIdx': int(config['order']['positionIdx'])} if config['order'].get('positionIdx') is not None else {} )
            # 'closeOnTrigger': False # Ensure it's not treated as a conditional close order (usually not needed for market)
        }
        try:
            if not exchange.has.get('createOrder'):
                 log.error(f"Close Order Failed: Exchange {exchange.name} does not support createOrder.")
                 return None

            order = await exchange.create_order(
                symbol=symbol,
                type=ORDER_TYPE_MARKET, # Use market order for immediate close
                side=side_to_close.lower(), # Ensure lowercase
                amount=amount_to_close, # Use float amount
                params=params
            )
            log.info(f"{Fore.GREEN}Position close order placed successfully! ID: {order.get('id')}{Style.RESET_ALL}")
            log.debug(f"Close Order details: {json.dumps(order, indent=2)}")

            # Force position check soon after closing attempt to confirm flat state
            async with position_lock: # Use inner lock only to modify check time
                 last_position_check_time = 0
            return order

        except ccxt.InvalidOrder as e:
             # This often happens if the position was already closed (manually, by SL/TP, or race condition)
             # Or if the size changed between fetch and close attempt.
             log.warning(f"{Fore.YELLOW}Close order failed, likely position already closed or size mismatch: {e}{Style.RESET_ALL}")
             # Bybit error "Order quantity exceeds the current position size" (code 110017) is common here.
             async with position_lock: # Use inner lock only to modify check time
                 last_position_check_time = 0 # Force position check to confirm state
             return {"info": {"error": str(e), "alreadyFlatOrChanged": True}}
        except ccxt.InsufficientFunds as e:
             # Should not happen with reduceOnly market order, but log if it does
             log.error(f"{Fore.RED}Close order failed: Insufficient Funds (unexpected for reduceOnly). {e}{Style.RESET_ALL}")
             async with position_lock: # Use inner lock only to modify check time
                 last_position_check_time = 0
             return None
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            log.error(f"{Fore.RED}Close order failed: {type(e).__name__} - {e}{Style.RESET_ALL}")
            # Consider retry logic for network errors
            return None
        except Exception as e:
            log.error(f"{Fore.RED}Unexpected error closing position: {e}{Style.RESET_ALL}", exc_info=True)
            return None

async def set_leverage(symbol: str, leverage: Union[int, float, str]):
    """Sets leverage for the specified symbol using CCXT (if supported)."""
    if not exchange or not market:
        log.warning("Cannot set leverage: Exchange or market not ready.")
        return
    # Check if the exchange instance supports the setLeverage method
    if not exchange.has.get('setLeverage'):
         log.warning(f"Exchange {exchange.name} does not support setting leverage via CCXT standardized method 'setLeverage'. Manual setting might be required, or check for exchange-specific methods/params.")
         return

    try:
        # Allow float leverage for some exchanges, but usually int is expected
        leverage_val = float(leverage) # Convert string/int to float first
        if leverage_val < 1:
             log.error(f"Invalid leverage value {leverage}. Must be 1 or greater. Leverage not set.")
             return
        # Some exchanges require integer leverage
        # leverage_val_final = int(leverage_val) if leverage_val.is_integer() else leverage_val
        leverage_val_final = leverage_val # Use float for flexibility, CCXT might handle conversion
    except (ValueError, TypeError):
         log.error(f"Invalid leverage value type: {leverage}. Must be numeric. Leverage not set.")
         return

    # Validate leverage against market limits if available
    max_leverage = None
    try:
         # Limits structure can vary
         leverage_limits = market.get('limits', {}).get('leverage', {})
         if leverage_limits and 'max' in leverage_limits and leverage_limits['max'] is not None:
              max_leverage = float(leverage_limits['max'])
    except (ValueError, TypeError):
         log.warning("Could not parse maximum leverage limit from market data.")

    if max_leverage is not None and leverage_val_final > max_leverage:
         log.error(f"Requested leverage {leverage_val_final} exceeds market maximum ({max_leverage}) for {symbol}. Leverage not set.")
         return

    log.info(f"Attempting to set leverage for {symbol} to {leverage_val_final}x...")
    try:
        # Note: CCXT's set_leverage abstracts underlying calls. Behavior depends on exchange.
        # Some exchanges require setting buy/sell leverage separately or have mode requirements (e.g., hedge vs one-way).
        # Bybit V5: set_leverage usually works for the symbol directly in one-way mode.
        # Check CCXT docs for params needed for specific exchanges (e.g., {'buyLeverage': L, 'sellLeverage': L})
        # Bybit V5 might need {'buyLeverage': L, 'sellLeverage': L} for unified margin, or just L for contract
        params = {}
        # Example: Add separate buy/sell leverage if needed based on config/exchange behavior
        # if config.get('order', {}).get('set_separate_leverage', False):
        #     params = {'buyLeverage': leverage_val_final, 'sellLeverage': leverage_val_final}

        response = await exchange.set_leverage(leverage_val_final, symbol, params=params)
        log.info(f"{Fore.GREEN}Leverage for {symbol} set to {leverage_val_final}x request sent.{Style.RESET_ALL} (Confirmation may depend on exchange response)")
        log.debug(f"Set leverage response: {response}") # Log exchange response if any

        # Optional: Verify leverage by fetching position data immediately after setting.
        # verify_leverage = config.get('checks', {}).get('verify_leverage_after_set', False)
        # if verify_leverage:
        #     log.debug("Verifying leverage via position check...")
        #     await asyncio.sleep(1) # Short delay
        #     global last_position_check_time
        #     async with position_lock: last_position_check_time = 0 # Force check
        #     pos_check = await get_current_position(symbol)
        #     ... (parsing logic as before) ...

    except ccxt.ExchangeError as e:
         # Handle common errors like "leverage not modified"
         # Bybit V5 error code for "Leverage not modified": 110044
         error_str = str(e).lower()
         if "not modified" in error_str or "110044" in str(e):
              log.warning(f"Leverage for {symbol} likely already set to {leverage_val_final}x (Exchange response: Not modified).")
         elif "set leverage not supported" in error_str:
              log.warning(f"Exchange reports 'setLeverage' is not supported for {symbol} (or current account/market settings). {e}")
         else:
              log.error(f"Failed to set leverage for {symbol}: {e}")
    except Exception as e:
        log.error(f"Unexpected error setting leverage: {e}", exc_info=True)


# --- WebSocket Watcher Loops ---
async def watch_kline_loop(symbol: str, timeframe: str):
    """Watches for new OHLCV candles via WebSocket and triggers processing."""
    global last_ws_update_time
    if not exchange:
         log.error("Kline watcher cannot start: Exchange not initialized.")
         return
    if not exchange.has.get('watchOHLCV'):
        log.error(f"{Fore.RED}Kline watcher cannot start: Exchange '{exchange.name}' reports 'watchOHLCV' is not supported via CCXT for the current configuration.{Style.RESET_ALL}")
        log.error("The bot relies on WebSocket klines. Consider implementing polling with fetch_ohlcv as an alternative if WS is unavailable, or ensure your exchange/account supports WS klines.")
        # Trigger shutdown as this is critical
        asyncio.create_task(shutdown(signal_type=None), name="WSUnsupportedShutdown")
        return # Exit task if not supported

    log.info(f"Starting Kline watcher for {symbol} ({timeframe})...")
    while not stop_event.is_set():
        try:
            # watch_ohlcv returns a list of *closed* candles since the last call
            # [[timestamp, open, high, low, close, volume]]
            candles = await exchange.watch_ohlcv(symbol, timeframe)

            if stop_event.is_set(): break # Check immediately after await returns

            if not candles:
                # log.debug("watch_ohlcv returned empty list.") # Can be noisy
                # Update timestamp even on empty receive to show WS connection is alive
                last_ws_update_time = time.monotonic()
                continue

            now_mono = time.monotonic()
            log.debug(f"Kline WS received {len(candles)} candle(s) (Last update: {now_mono - last_ws_update_time:.1f}s ago)")
            last_ws_update_time = now_mono # Update health check timestamp

            # Process each received closed candle
            for candle_data in candles:
                try:
                    # Ensure candle_data is a list/tuple of expected length
                    if not isinstance(candle_data, (list, tuple)) or len(candle_data) != 6:
                         log.warning(f"Received malformed candle data via WS: {candle_data}. Skipping.")
                         continue

                    ts_ms, o, h, l, c, v = candle_data
                    # Validate data types before processing
                    if not all(isinstance(x, (int, float)) for x in [ts_ms, o, h, l, c, v]):
                        log.warning(f"Received invalid data types in WS candle: {[type(x) for x in candle_data]}. Skipping.")
                        continue
                    # Check for NaN/inf which shouldn't happen in valid candles
                    if not all(np.isfinite(x) for x in [o, h, l, c, v]):
                         log.warning(f"Received non-finite values in WS candle: {[o, h, l, c, v]}. Skipping.")
                         continue

                    # Convert timestamp to UTC datetime
                    ts = pd.to_datetime(ts_ms, unit='ms', utc=True)
                    # Convert OHLCV to float, handle potential errors gracefully
                    try:
                        new_data = {'open': float(o), 'high': float(h), 'low': float(l), 'close': float(c), 'volume': float(v)}
                    except (ValueError, TypeError) as e:
                        log.error(f"Error converting WS candle values to float: {e}. Candle: {candle_data}. Skipping.")
                        continue

                    log.debug(f"WS Kline Processed: T={ts}, O={o}, H={h}, L={l}, C={c}, V={v}")

                    # Process the confirmed candle data asynchronously
                    # Use create_task to avoid blocking WS loop if processing takes time
                    proc_task = asyncio.create_task(process_candle(ts, new_data), name=f"CandleProcessor_{ts_ms}")
                    running_tasks.add(proc_task)
                    proc_task.add_done_callback(running_tasks.discard)

                except (ValueError, TypeError) as e:
                    log.error(f"Error processing individual WS candle data {candle_data}: {e}", exc_info=True)
                    continue # Skip this candle and process the next
                except Exception as e:
                     log.error(f"Unexpected error processing WS candle {candle_data}: {e}", exc_info=True)
                     continue

        except ccxt.NetworkError as e:
            log.warning(f"Kline Watcher Network Error: {e}. Reconnecting...")
            await asyncio.sleep(5) # Wait before implicit reconnection by watch_ohlcv
        except ccxt.ExchangeError as e:
            log.warning(f"Kline Watcher Exchange Error: {e}. Retrying...")
            # Could add logic here to switch to polling if WS fails repeatedly
            await asyncio.sleep(15) # Longer wait for exchange errors
        except asyncio.CancelledError:
             log.info("Kline watcher task cancelled.")
             break # Exit loop cleanly on cancellation
        except Exception as e:
            log.error(f"Unexpected error in Kline watcher loop: {e}", exc_info=True)
            await asyncio.sleep(30) # Longer sleep on unexpected errors before retrying
    log.info("Kline watcher loop finished.")

async def process_candle(timestamp: pd.Timestamp, data: Dict[str, float]):
    """Adds candle data to the DataFrame and triggers strategy analysis. Uses data_lock briefly."""
    global latest_dataframe # Allow modification
    if not strategy_instance:
         log.warning("Strategy instance not available, skipping candle processing.")
         return

    df_copy_for_analysis: Optional[pd.DataFrame] = None # Initialize

    async with data_lock: # Ensure exclusive access to the dataframe for update
        if latest_dataframe is None:
            log.warning("DataFrame not initialized, skipping candle processing.")
            return

        # Ensure timestamp is timezone-aware (UTC) to match WS data
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize('UTC')
        # Ensure index is timezone-aware (UTC)
        if latest_dataframe.index.tzinfo is None and not latest_dataframe.empty:
             log.warning("DataFrame index was not timezone-aware. Localizing to UTC.")
             latest_dataframe.index = latest_dataframe.index.tz_localize('UTC')

        # Check if timestamp already exists (e.g., duplicate WS message or update)
        if timestamp in latest_dataframe.index:
             # Update the existing row - useful if WS sends updates for the same closed candle
             log.debug(f"Updating existing candle data for {timestamp}.")
             latest_dataframe.loc[timestamp, list(data.keys())] = list(data.values())
        else:
             # Append new candle data
             log.debug(f"Adding new candle {timestamp} to DataFrame.")
             # Create a new DataFrame row with the correct index type and timezone
             new_row = pd.DataFrame([data], index=pd.DatetimeIndex([timestamp], tz='UTC'))
             # Ensure columns match before concatenating
             if not all(col in new_row.columns for col in latest_dataframe.columns if col in data):
                  log.warning("Column mismatch between DataFrame and new row. Realigning.")
                  # Potentially reindex or handle differently based on needs

             try:
                 # Use concat instead of append (append is deprecated)
                 latest_dataframe = pd.concat([latest_dataframe, new_row])
                 # Sort index just in case messages arrive out of order (can be slow on large dfs)
                 # latest_dataframe = latest_dataframe.sort_index() # Optional: uncomment if needed
             except Exception as e:
                  log.error(f"Error concatenating new candle data: {e}", exc_info=True)
                  return # Avoid proceeding with corrupted dataframe

             # Prune old data to maintain max_df_len
             max_len = config.get('data', {}).get('max_df_len', 2000)
             if len(latest_dataframe) > max_len:
                 latest_dataframe = latest_dataframe.iloc[-(max_len):]
                 log.debug(f"DataFrame pruned to {len(latest_dataframe)} rows.")

        # Create a copy of the dataframe *inside* the lock for consistent analysis state
        df_copy_for_analysis = latest_dataframe.copy()
        # Lock released here

    # --- Trigger Strategy Analysis on the copied DataFrame (outside the lock) ---
    if df_copy_for_analysis is None:
        log.error("Failed to create DataFrame copy for analysis.")
        return

    log.info(f"Running analysis on DataFrame snapshot ending with candle: {timestamp}")
    try:
        # Run the strategy's update method
        # It's crucial that strategy.update() does NOT modify the df it receives
        # or works on its own internal copy if modifications are needed.
        analysis_results: Optional[AnalysisResults] = strategy_instance.update(df_copy_for_analysis)

        # Check if results are valid before processing
        if analysis_results is None or not isinstance(analysis_results, dict) or 'last_signal' not in analysis_results:
             # This log message suggests the error might be happening inside strategy.update()
             # when calculating indicators and trying to fill NaNs, or returning unexpected type.
             log.error(f"Strategy update returned invalid results (None, not dict, or missing 'last_signal'). Check strategy.py for errors (e.g., NaN handling, indicator calculations, return value).")
             # Log the state of df_copy's tail for debugging
             log.debug(f"DataFrame tail sent to strategy:\n{df_copy_for_analysis.tail()}")
             return

        # Process the generated signals asynchronously
        # Create a new task to handle signal processing without blocking candle updates
        signal_task = asyncio.create_task(process_signals(analysis_results), name=f"SignalProcessor_{timestamp.value // 10**6}") # Use ms timestamp in name
        running_tasks.add(signal_task)
        # Remove task from set when done to prevent memory leak
        signal_task.add_done_callback(running_tasks.discard)

    except AttributeError as e:
         # Catch the specific 'numpy.ndarray' object has no attribute 'fillna' if it bubbles up
         # Or other attribute errors potentially caused by strategy code
         if "'numpy.ndarray' object has no attribute 'fillna'" in str(e):
              log.error(f"{Fore.RED}CRITICAL ERROR during strategy analysis: {e}. "
                        f"This likely means an indicator calculation in strategy.py returned a NumPy array "
                        f"and '.fillna()' was called on it directly. FIX REQUIRED IN strategy.py.{Style.RESET_ALL}", exc_info=True)
              log.error("Please check lines in strategy.py mentioned in the traceback and ensure results are Pandas Series/DataFrame before using .fillna().")
              # Consider stopping the bot here if strategy analysis is critically broken
              asyncio.create_task(shutdown(signal_type=None), name="StrategyErrorShutdown")
         else:
              # Log other AttributeErrors or unexpected errors
              log.error(f"AttributeError during strategy analysis update: {e}", exc_info=True)
    except Exception as e:
        log.error(f"Unexpected error during strategy analysis: {e}", exc_info=True)
        log.debug(f"DataFrame tail sent to strategy:\n{df_copy_for_analysis.tail()}")


async def watch_positions_loop(symbol: str):
    """(Optional) Watches for position updates via WebSocket. Uses position_lock only to trigger REST check."""
    global last_ws_update_time, last_position_check_time
    if not exchange: return
    if not exchange.has.get('watchPositions'):
        log.info("Position watcher skipped: Exchange does not support watchPositions.")
        return

    log.info(f"Starting Position watcher for {symbol} (if supported)...")
    while not stop_event.is_set():
        try:
            # watch_positions usually returns a list of all positions for the account type
            positions_updates = await exchange.watch_positions([symbol]) # Watch specific symbol if supported

            if stop_event.is_set(): break

            now_mono = time.monotonic()
            log.debug(f"Position WS received data (Last update: {now_mono - last_ws_update_time:.1f}s ago)")
            last_ws_update_time = now_mono # Update health check timestamp

            if not positions_updates: continue

            pos_update_for_symbol = None
            for p_update in positions_updates:
                # Filter for the specific symbol being traded
                if p_update and p_update.get('symbol') == symbol:
                     pos_update_for_symbol = p_update
                     break # Found our symbol

            if pos_update_for_symbol:
                log.debug(f"WS Position Update Raw: {pos_update_for_symbol}")
                # Parse the update carefully using Decimal
                try:
                    size_str = str(pos_update_for_symbol.get('contracts', '0') or '0')
                    side = str(pos_update_for_symbol.get('side', 'none') or 'none').lower()
                    entry_price_str = str(pos_update_for_symbol.get('entryPrice', '0') or '0')

                    # Use string conversion for robustness
                    ws_size = Decimal(size_str)
                    ws_entry_price = Decimal(entry_price_str)
                    ws_side = SIDE_BUY if side == 'long' else SIDE_SELL if side == 'short' else SIDE_NONE
                    is_flat = ws_size.is_zero() or ws_size < Decimal(0) # Treat negative size as flat/error

                    if is_flat:
                         ws_side = SIDE_NONE
                         ws_size = Decimal(0)
                         ws_entry_price = Decimal(0)

                    log_msg = (f"WS Position Update Parsed: Side={ws_side}, Size={ws_size}, Entry={ws_entry_price}")

                    # Update internal state cautiously - REST check remains the source of truth for critical actions
                    # Only use lock to trigger the REST check, not to update position state from WS
                    async with position_lock:
                        # Log if WS state differs significantly from last known REST state
                        # Compare size and side
                        if ws_size != current_position['size'] or ws_side != current_position['side']:
                            log.warning(f"{Fore.MAGENTA}{log_msg} - Differs from cache ({current_position['side']} {current_position['size']}). Forcing REST check.{Style.RESET_ALL}")
                            # Force a REST check soon to confirm the change
                            last_position_check_time = 0 # Reset timer to trigger REST check soon
                        else:
                             log.debug(log_msg + " - Matches cache.")


                except (InvalidOperation, TypeError) as e:
                    log.error(f"Error parsing WS position update data {pos_update_for_symbol}: {e}")
                except Exception as e:
                     log.error(f"Unexpected error processing WS position update: {e}", exc_info=True)
            else:
                 # This might happen if the position for the symbol closes, or initial fetch shows no position
                 log.debug(f"Received position update via WS, but not for {symbol} (or position is flat).")
                 # If we previously had a position according to cache, force a REST check
                 async with position_lock:
                     if current_position['size'] != Decimal(0):
                         log.warning(f"{Fore.MAGENTA}WS position update no longer includes {symbol} or shows flat. Forcing REST check.{Style.RESET_ALL}")
                         last_position_check_time = 0


        except ccxt.NetworkError as e: log.warning(f"Position Watcher Network Error: {e}. Reconnecting..."); await asyncio.sleep(5)
        except ccxt.ExchangeError as e: log.warning(f"Position Watcher Exchange Error: {e}. Retrying..."); await asyncio.sleep(15)
        except asyncio.CancelledError: log.info("Position watcher task cancelled."); break
        except Exception as e: log.error(f"Unexpected error in Position watcher: {e}", exc_info=True); await asyncio.sleep(30)
    log.info("Position watcher loop finished.")

async def watch_orders_loop(symbol: str):
    """(Optional) Watches for order updates (fills, cancellations) via WebSocket. Uses position_lock only to trigger REST check."""
    global last_ws_update_time, last_position_check_time
    if not exchange: return
    if not exchange.has.get('watchOrders'):
        log.info("Order watcher skipped: Exchange does not support watchOrders.")
        return

    log.info(f"Starting Order watcher for {symbol}...")
    while not stop_event.is_set():
        try:
            orders = await exchange.watch_orders(symbol)

            if stop_event.is_set(): break

            now_mono = time.monotonic()
            log.debug(f"Order WS received data (Last update: {now_mono - last_ws_update_time:.1f}s ago)")
            last_ws_update_time = now_mono # Update health check timestamp

            if not orders: continue

            for order_update in orders:
                 # Process order updates - log fills, SL/TP triggers, cancellations
                 status = order_update.get('status') # 'open', 'closed', 'canceled', 'expired', 'rejected'
                 order_id = order_update.get('id')
                 client_order_id = order_update.get('clientOrderId')
                 filled_val = order_update.get('filled')
                 filled_str = safe_format('format_amount', filled_val, str(filled_val)) if filled_val is not None else 'N/A'
                 avg_price_val = order_update.get('average')
                 avg_price_str = safe_format('format_price', avg_price_val, str(avg_price_val)) if avg_price_val is not None else 'N/A'
                 order_type = order_update.get('type')
                 order_side = order_update.get('side')
                 reduce_only = order_update.get('reduceOnly', False) # Default to False if missing
                 post_only = order_update.get('postOnly', False) # Default to False if missing

                 log.info(f"{Fore.CYAN}Order Update via WS [{symbol}]: ID={order_id}, ClientID={client_order_id}, Side={order_side}, Type={order_type}, Status={status}, Filled={filled_str}, AvgPrice={avg_price_str}, ReduceOnly={reduce_only}{Style.RESET_ALL}")
                 log.debug(f"Full Order Update WS Data: {order_update}")

                 # If an order reaches a terminal state (filled, canceled, rejected),
                 # it's a strong indicator the position state might have changed.
                 # Also trigger on partial fills ('closed' status might only mean fully filled, need to check 'filled' amount?)
                 # Triggering on any 'closed' or 'canceled' status seems safest.
                 if status in ['closed', 'canceled', 'rejected', 'expired'] and order_id:
                      log.warning(f"{Fore.YELLOW}Order {order_id} ({order_side} {order_type}) reached terminal state '{status}' via WS.{Style.RESET_ALL}")
                      # Force a position check via REST API to get the definitive state
                      log.info("Forcing REST position check after order update.")
                      async with position_lock: # Use lock just to modify check time
                          last_position_check_time = 0

        except ccxt.NetworkError as e: log.warning(f"Order Watcher Network Error: {e}. Reconnecting..."); await asyncio.sleep(5)
        except ccxt.ExchangeError as e: log.warning(f"Order Watcher Exchange Error: {e}. Retrying..."); await asyncio.sleep(15)
        except asyncio.CancelledError: log.info("Order watcher task cancelled."); break
        except Exception as e: log.error(f"Unexpected error in Order watcher: {e}", exc_info=True); await asyncio.sleep(30)
    log.info("Order watcher loop finished.")

# --- Signal Processing & Execution (Async) ---
async def process_signals(results: AnalysisResults):
    """Processes strategy signals and executes trades based on position state. Uses locks implicitly via called functions."""
    if stop_event.is_set():
        log.warning("Signal processing skipped: Stop event is set.")
        return
    if not results or not strategy_instance or not market:
        log.warning("Signal processing skipped: Missing analysis results, strategy, or market data.")
        return
    # Check for essential strategy attributes needed for SL/TP calculation
    if not all(hasattr(strategy_instance, attr) for attr in ['price_tick', 'round_price']):
        log.error("Signal processing skipped: Strategy instance missing required attributes/methods (price_tick, round_price).")
        return
    if not isinstance(strategy_instance.price_tick, Decimal) or strategy_instance.price_tick <= Decimal(0):
        log.error("Signal processing skipped: Strategy instance 'price_tick' must be a positive Decimal.")
        return

    # Extract data from results (handle potential missing keys gracefully)
    signal = results.get('last_signal', SIGNAL_HOLD) # Default to HOLD if missing
    last_close = results.get('last_close')
    last_atr = results.get('last_atr') # Can be None or NaN
    active_bull_boxes = results.get('active_bull_boxes', [])
    active_bear_boxes = results.get('active_bear_boxes', [])
    symbol = config['symbol']

    # Validate essential data
    if last_close is None or pd.isna(last_close) or not np.isfinite(last_close) or last_close <= 0:
        log.warning(f"Cannot process signal '{signal}': Invalid or non-positive last close price ({last_close}).")
        return
    # ATR might be NaN initially, handle that in SL calculation
    if last_atr is not None and (pd.isna(last_atr) or not np.isfinite(last_atr)):
         log.debug("Last ATR is NaN or non-finite, will affect ATR-based SL.")
         last_atr = None # Treat as unavailable for calculations needing a valid float

    # Format price for logging using strategy method if available
    formatted_close = safe_format('format_price', last_close, str(last_close))
    log.debug(f"Processing Signal: {signal}, Last Close: {formatted_close}, Last ATR: {last_atr}")

    # --- Get Current Position State (Crucial Step - Use REST API Fetch) ---
    # Lock is acquired within get_current_position if needed
    # Force fetch by resetting timer (ensures we use latest state for decision)
    global last_position_check_time
    async with position_lock: # Use lock just to modify check time
        last_position_check_time = 0
    pos_data = await get_current_position(symbol)

    # If get_current_position returned None, it means a critical error occurred fetching state.
    # Do not proceed with actions based on potentially stale cached data.
    if pos_data is None:
        log.error("Could not get reliable position data due to API error. Skipping signal action to avoid errors.")
        return

    # Use Decimal for position size comparison
    current_pos_size = pos_data.get('size', Decimal(0))
    current_pos_side = pos_data.get('side', SIDE_NONE) # 'Buy', 'Sell', 'None'
    is_long = current_pos_side == SIDE_BUY and current_pos_size > Decimal(0)
    is_short = current_pos_side == SIDE_SELL and current_pos_size > Decimal(0)
    is_flat = not is_long and not is_short

    log.info(f"Processing signal '{signal}' | Current Position: {'Long' if is_long else 'Short' if is_short else 'Flat'} (Size: {current_pos_size})")

    # --- Execute Actions based on Signal and Current Position State ---
    try:
        # Use Decimal for TP ratio for precision
        tp_ratio = Decimal(str(config['order'].get('tp_ratio', 2.0)))
        risk_percent = float(config['order'].get('risk_per_trade_percent', 1.0)) # Keep as float for calculate_order_qty
        sl_method = config.get('strategy', {}).get('stop_loss', {}).get('method', 'ATR').upper()
        sl_atr_multiplier = float(config.get('strategy', {}).get('stop_loss', {}).get('atr_multiplier', 1.5))
        sl_ob_buffer_atr_mult = float(config.get('strategy', {}).get('stop_loss', {}).get('ob_buffer_atr_mult', 0.1))
        sl_ob_buffer_ticks = int(config.get('strategy', {}).get('stop_loss', {}).get('ob_buffer_ticks', 5))
    except (ValueError, TypeError, KeyError) as e:
         log.error(f"Error parsing configuration values for signal processing: {e}. Using defaults.")
         tp_ratio = Decimal("2.0")
         risk_percent = 1.0
         sl_method = "ATR"
         sl_atr_multiplier = 1.5
         sl_ob_buffer_atr_mult = 0.1
         sl_ob_buffer_ticks = 5

    # --- Helper function for SL/TP calculation and validation ---
    def calculate_sl_tp(is_buy_signal: bool, entry_price: float, atr: Optional[float]) -> tuple[Optional[float], Optional[float]]:
        sl_price_raw: Optional[float] = None
        tp_price_raw: Optional[float] = None
        sl_price_final: Optional[float] = None
        tp_price_final: Optional[float] = None

        # Re-check strategy instance and price_tick availability (already checked before calling)
        if not strategy_instance: return None, None # Should not happen
        price_tick_decimal = strategy_instance.price_tick

        # Validate entry price
        if entry_price is None or not np.isfinite(entry_price) or entry_price <= 0:
             log.error(f"Cannot calculate SL/TP: Invalid entry price provided ({entry_price}).")
             return None, None

        # Calculate SL
        if sl_method == "ATR":
            if atr and atr > 0 and np.isfinite(atr):
                try:
                    sl_delta = atr * sl_atr_multiplier
                    sl_price_raw = entry_price - sl_delta if is_buy_signal else entry_price + sl_delta
                    log.debug(f"Calculated ATR SL: {entry_price} {'-' if is_buy_signal else '+'} ({atr} * {sl_atr_multiplier}) = {sl_price_raw}")
                except TypeError as e:
                    log.error(f"TypeError during ATR SL calculation: {e}. Inputs: entry={entry_price}, atr={atr}, mult={sl_atr_multiplier}")
                    return None, None
            else:
                log.warning(f"ATR SL method selected, but ATR ({atr}) is invalid or zero. Cannot calculate SL.")
                return None, None
        elif sl_method == "OB":
            try:
                # Use Decimal for buffer calculation
                sl_buffer = Decimal(0)
                if atr and atr > 0 and np.isfinite(atr):
                    sl_buffer = Decimal(str(atr)) * Decimal(str(sl_ob_buffer_atr_mult))
                else:
                    sl_buffer = price_tick_decimal * Decimal(sl_ob_buffer_ticks)
                # Ensure buffer is positive and non-zero
                sl_buffer = abs(sl_buffer)
                if sl_buffer.is_zero():
                     log.warning("OB SL buffer calculated to zero. Using one price tick instead.")
                     sl_buffer = price_tick_decimal

                entry_price_dec = Decimal(str(entry_price))

                if is_buy_signal:
                    # Expect list of dicts like [{'top': float, 'bottom': float}, ...]
                    relevant_obs = [Decimal(str(b['bottom'])) for b in active_bull_boxes
                                    if isinstance(b, dict) and 'bottom' in b
                                    and isinstance(b['bottom'], (float, int)) and np.isfinite(b['bottom'])
                                    and Decimal(str(b['bottom'])) < entry_price_dec]
                    if relevant_obs:
                        ob_sl_level = max(relevant_obs) # Highest bottom below entry
                        sl_price_raw_dec = ob_sl_level - sl_buffer
                        sl_price_raw = float(sl_price_raw_dec)
                        log.debug(f"Calculated OB SL (Buy): Max Bull Bottom={ob_sl_level}, Buffer={sl_buffer}, SL={sl_price_raw}")
                    else: log.warning("OB SL method selected (Buy), but no valid Bull OB found below price. Cannot calculate SL."); return None, None
                else: # Sell signal
                    relevant_obs = [Decimal(str(b['top'])) for b in active_bear_boxes
                                    if isinstance(b, dict) and 'top' in b
                                    and isinstance(b['top'], (float, int)) and np.isfinite(b['top'])
                                    and Decimal(str(b['top'])) > entry_price_dec]
                    if relevant_obs:
                        ob_sl_level = min(relevant_obs) # Lowest top above entry
                        sl_price_raw_dec = ob_sl_level + sl_buffer
                        sl_price_raw = float(sl_price_raw_dec)
                        log.debug(f"Calculated OB SL (Sell): Min Bear Top={ob_sl_level}, Buffer={sl_buffer}, SL={sl_price_raw}")
                    else: log.warning("OB SL method selected (Sell), but no valid Bear OB found above price. Cannot calculate SL."); return None, None
            except (ValueError, TypeError, KeyError, InvalidOperation) as e:
                 log.error(f"Error calculating OB SL: {e}. OB Data: Bull={active_bull_boxes}, Bear={active_bear_boxes}")
                 return None, None
        else:
            log.error(f"Unknown SL method '{sl_method}'. Cannot calculate SL."); return None, None

        # Validate and Round SL Price
        if sl_price_raw is None or not np.isfinite(sl_price_raw) or sl_price_raw <= 0:
             log.error(f"Invalid or non-positive raw SL price calculated: {sl_price_raw}. Aborting entry.")
             return None, None
        # Check SL validity relative to entry price (SL must be worse than entry)
        if (is_buy_signal and sl_price_raw >= entry_price) or (not is_buy_signal and sl_price_raw <= entry_price):
             log.error(f"Invalid SL price calculated for {'BUY' if is_buy_signal else 'SELL'} signal (SL={sl_price_raw}, Entry={entry_price}). SL must be below entry for BUY, above for SELL. Aborting entry.")
             return None, None

        try:
            sl_price_final = strategy_instance.round_price(sl_price_raw) # Expects float return
        except Exception as e:
            log.error(f"Error rounding SL price {sl_price_raw}: {e}. Using raw value."); sl_price_final = sl_price_raw

        if sl_price_final is None or not np.isfinite(sl_price_final) or sl_price_final <= 0:
             log.error(f"Invalid or non-positive final SL price after rounding: {sl_price_final}. Aborting entry.")
             return None, None
        # Re-validate after rounding
        if (is_buy_signal and sl_price_final >= entry_price) or (not is_buy_signal and sl_price_final <= entry_price):
             log.error(f"Invalid SL price after rounding for {'BUY' if is_buy_signal else 'SELL'} signal (SL={sl_price_final}, Entry={entry_price}). Aborting entry.")
             return None, None


        log.info(f"Calculated Entry SL: {safe_format('format_price', sl_price_final)} (Raw: {sl_price_raw})")

        # Calculate TP Price based on final SL
        try:
            # Use Decimal for precision in TP calculation
            entry_price_dec = Decimal(str(entry_price))
            sl_price_final_dec = Decimal(str(sl_price_final))
            sl_distance = abs(entry_price_dec - sl_price_final_dec)

            # Check if distance is meaningful (greater than half a price tick)
            if sl_distance > price_tick_decimal / Decimal(2):
                 tp_delta = sl_distance * tp_ratio
                 tp_price_raw_dec = entry_price_dec + tp_delta if is_buy_signal else entry_price_dec - tp_delta
                 tp_price_raw = float(tp_price_raw_dec)

                 try:
                     tp_price_final = strategy_instance.round_price(tp_price_raw) # Expects float return
                 except Exception as e:
                     log.error(f"Error rounding TP price {tp_price_raw}: {e}. Using raw value."); tp_price_final = tp_price_raw

                 # Basic TP validation
                 if tp_price_final is None or not np.isfinite(tp_price_final) or tp_price_final <= 0:
                      log.warning(f"Invalid or non-positive final TP price after rounding: {tp_price_final}. Setting TP to None.")
                      tp_price_final = None
                 elif (is_buy_signal and tp_price_final <= entry_price) or (not is_buy_signal and tp_price_final >= entry_price):
                      log.warning(f"Calculated TP price {tp_price_final} is not beyond entry price {entry_price}. Setting TP to None.")
                      tp_price_final = None
                 else:
                      log.info(f"Calculated Entry TP: {safe_format('format_price', tp_price_final)} (Raw: {tp_price_raw}, Ratio: {tp_ratio})")
            else:
                 log.warning("SL distance is zero or negative after rounding. Cannot calculate TP.")
                 tp_price_final = None
        except (InvalidOperation, TypeError, AttributeError) as e:
             log.error(f"Error calculating TP distance: {e}. Setting TP to None.")
             tp_price_final = None

        return sl_price_final, tp_price_final
    # --- End Helper Function ---


    # --- BUY Signal ---
    if signal == SIGNAL_BUY and is_flat:
        log.warning(f"{Fore.GREEN}{Style.BRIGHT}BUY Signal received - Attempting to Enter Long.{Style.RESET_ALL}")
        entry_price_for_calc = last_close # Use last close as entry estimate for calcs
        sl_price, tp_price = calculate_sl_tp(is_buy_signal=True, entry_price=entry_price_for_calc, atr=last_atr)

        if sl_price is not None:
            # Calculate Quantity (Pass entry and final rounded SL)
            qty = await calculate_order_qty(entry_price_for_calc, sl_price, risk_percent)
            if qty and qty > 0:
                # Determine entry price for order placement (limit or market)
                order_entry_price = entry_price_for_calc if config['order']['type'].lower() == ORDER_TYPE_LIMIT else None
                # Place the order (lock acquired within place_order)
                await place_order(symbol=symbol, side=ORDER_SIDE_BUY, qty=qty,
                                  price=order_entry_price,
                                  sl_price=sl_price, tp_price=tp_price)
            else:
                log.error("BUY order cancelled: Quantity calculation failed or resulted in zero/None.")
        else:
            log.error("BUY order cancelled: Failed to calculate valid Stop Loss.")

    # --- SELL Signal ---
    elif signal == SIGNAL_SELL and is_flat:
        log.warning(f"{Fore.RED}{Style.BRIGHT}SELL Signal received - Attempting to Enter Short.{Style.RESET_ALL}")
        entry_price_for_calc = last_close
        sl_price, tp_price = calculate_sl_tp(is_buy_signal=False, entry_price=entry_price_for_calc, atr=last_atr)

        if sl_price is not None:
            # Calculate Quantity
            qty = await calculate_order_qty(entry_price_for_calc, sl_price, risk_percent)
            if qty and qty > 0:
                # Determine entry price for order placement (limit or market)
                order_entry_price = entry_price_for_calc if config['order']['type'].lower() == ORDER_TYPE_LIMIT else None
                # Place the order (lock acquired within place_order)
                await place_order(symbol=symbol, side=ORDER_SIDE_SELL, qty=qty,
                                  price=order_entry_price,
                                  sl_price=sl_price, tp_price=tp_price)
            else:
                log.error("SELL order cancelled: Quantity calculation failed or resulted in zero/None.")
        else:
            log.error("SELL order cancelled: Failed to calculate valid Stop Loss.")

    # --- EXIT_LONG Signal ---
    elif signal == SIGNAL_EXIT_LONG and is_long:
        log.warning(f"{Fore.YELLOW}{Style.BRIGHT}EXIT_LONG Signal received - Attempting to Close Long Position.{Style.RESET_ALL}")
        # Pass the recently fetched position data to close_position
        await close_position(symbol, pos_data)

    # --- EXIT_SHORT Signal ---
    elif signal == SIGNAL_EXIT_SHORT and is_short:
        log.warning(f"{Fore.YELLOW}{Style.BRIGHT}EXIT_SHORT Signal received - Attempting to Close Short Position.{Style.RESET_ALL}")
        # Pass the recently fetched position data to close_position
        await close_position(symbol, pos_data)

    # --- No Action / Already in State / Hold Signal ---
    elif signal == SIGNAL_HOLD:
        log.debug("HOLD Signal - No action taken.")
    elif signal == SIGNAL_BUY and is_long:
        log.debug("BUY Signal received, but already Long. No action.")
        # Future Enhancement: Add logic for pyramiding/adding to position if desired
    elif signal == SIGNAL_SELL and is_short:
        log.debug("SELL Signal received, but already Short. No action.")
        # Future Enhancement: Add logic for pyramiding/adding to position if desired
    elif signal == SIGNAL_EXIT_LONG and not is_long:
        log.debug("EXIT_LONG Signal received, but not Long. No action.")
    elif signal == SIGNAL_EXIT_SHORT and not is_short:
        log.debug("EXIT_SHORT Signal received, but not Short. No action.")
    else:
        # Catchall for unhandled combinations (should ideally not be reached with current signals)
        log.debug(f"Signal '{signal}' received but no matching action criteria met (Position: {current_pos_side} {current_pos_size}).")


# --- Periodic Health Check ---
async def periodic_check_loop():
    """Runs periodic checks for WebSocket health, stale positions, etc."""
    global last_health_check_time, last_position_check_time, last_ws_update_time
    try:
        check_interval = int(config.get('checks', {}).get('health_check_interval', 60))
        pos_check_interval = int(config.get('checks', {}).get('position_check_interval', 30))
        # Consider WS dead if no message received for longer than health check interval + buffer
        ws_timeout = float(config.get('checks', {}).get('ws_timeout_factor', 1.5)) * check_interval
        if check_interval <= 0 or pos_check_interval <= 0 or ws_timeout <= 0:
             raise ValueError("Check intervals and timeout factor must be positive.")
    except (ValueError, TypeError, KeyError) as e:
        log.error(f"Invalid interval configuration in 'checks': {e}. Using defaults (60s health, 30s pos, 90s WS timeout).")
        check_interval = 60
        pos_check_interval = 30
        ws_timeout = 90.0

    log.info(f"Starting Periodic Check loop (Health Interval: {check_interval}s, Position Interval: {pos_check_interval}s, WS Timeout: {ws_timeout}s)")

    # Initial sleep offset slightly to avoid running immediately on startup
    await asyncio.sleep(5)

    while not stop_event.is_set():
        next_check_time = time.monotonic() + check_interval
        try:
            # Sleep until the next check time (handle stop_event during sleep)
            while time.monotonic() < next_check_time:
                if stop_event.is_set(): break
                # Sleep in smaller increments to check stop_event more frequently
                await asyncio.sleep(min(1.0, next_check_time - time.monotonic()))
            if stop_event.is_set(): break # Exit if stop signal received during sleep

            now_mono = time.monotonic()
            log.debug(f"Running periodic checks (Time since last: {now_mono - last_health_check_time:.1f}s)...")
            last_health_check_time = now_mono

            force_pos_check = False
            # 1. Check WebSocket Health (based on last message time from any WS watcher)
            if last_ws_update_time > 0: # Ensure WS has sent at least one message
                time_since_last_ws = now_mono - last_ws_update_time
                if time_since_last_ws >= ws_timeout:
                     log.warning(f"{Fore.RED}WebSocket potentially stale! No updates received for {time_since_last_ws:.1f}s (Timeout: {ws_timeout}s).{Style.RESET_ALL}")
                     # Action: Could try to reconnect WS or trigger a more serious alert.
                     # The main loop already checks if essential tasks (like kline watcher) have died.
                     # Forcing a REST position check is prudent here.
                     force_pos_check = True
                else:
                     log.debug(f"WebSocket health check OK (last update {time_since_last_ws:.1f}s ago).")
            else:
                 # If WS hasn't sent anything after a reasonable startup period (e.g., > ws_timeout), force check.
                 if startup_time > 0 and now_mono - startup_time > ws_timeout: # Check startup_time is set
                     log.warning(f"WebSocket health check: No WS messages received after {now_mono - startup_time:.1f}s. Forcing position check.")
                     force_pos_check = True
                 else:
                     log.debug("WebSocket health check: No WS messages received yet (within startup grace period).")


            # 2. Force Position Check if REST data is stale OR if WS seems stale
            time_since_last_pos_check = now_mono - last_position_check_time if last_position_check_time > 0 else float('inf') # Treat 0 as infinitely long ago
            if force_pos_check or time_since_last_pos_check >= pos_check_interval:
                 if force_pos_check and not (time_since_last_pos_check >= pos_check_interval):
                      log.warning("Periodic check forcing REST position update due to potential WS timeout...")
                 else:
                      log.info("Periodic check forcing REST position update (interval or WS timeout)...")
                 # Run the check non-blockingly
                 pos_check_task = asyncio.create_task(get_current_position(config['symbol']), name="PeriodicPosCheck")
                 running_tasks.add(pos_check_task)
                 pos_check_task.add_done_callback(running_tasks.discard)
            else:
                log.debug(f"Periodic position check skipped (last check {time_since_last_pos_check:.1f}s ago).")


            # 3. Add other checks as needed:
            #    - Check available balance periodically? (Could be added to get_current_position or here)
            #    - Check exchange status endpoint? (Requires specific CCXT method/API call)
            #    - Check for excessive error logs within a time window?

            log.debug("Periodic checks complete.")

        except asyncio.CancelledError:
            log.info("Periodic check loop cancelled.")
            break # Exit loop cleanly on cancellation
        except Exception as e:
            log.error(f"Unexpected error in periodic check loop: {e}", exc_info=True)
            # Avoid tight loop on persistent error - sleep until next planned check time
            wait_time = max(1.0, next_check_time - time.monotonic()) # Ensure at least 1s sleep
            log.info(f"Sleeping for {wait_time:.1f}s after error in periodic check.")
            await asyncio.sleep(wait_time)

    log.info("Periodic check loop finished.")


# --- Graceful Shutdown ---
async def shutdown(signal_type: Optional[signal.Signals] = None):
    """Cleans up resources, cancels tasks, and exits gracefully."""
    if stop_event.is_set():
        # Avoid running shutdown multiple times if signals are received quickly
        log.warning("Shutdown already in progress.")
        return

    signal_name = f"signal {signal_type.name}" if signal_type else "internal request"
    log.warning(f"{Fore.RED}Shutdown initiated by {signal_name}...{Style.RESET_ALL}")
    stop_event.set() # Signal all loops and tasks to stop

    # Optional: Implement logic to close open positions on exit (use with extreme caution!)
    close_on_exit = config.get('shutdown', {}).get('close_open_position', False)
    if close_on_exit:
        try:
            # Ensure this function is robust and has timeouts
            log.info("Attempting to close open position on exit (timeout 30s)...")
            await asyncio.wait_for(close_open_position_on_exit(), timeout=30.0)
        except asyncio.TimeoutError:
            log.error("Timeout occurred while trying to close position on exit.")
        except Exception as e:
            log.error(f"Error during close_open_position_on_exit: {e}", exc_info=True)

    # Cancel all running asyncio tasks collected in running_tasks
    # Make a copy as the set might change during iteration
    tasks_to_cancel = list(running_tasks)
    cancelled_task_count = 0
    if tasks_to_cancel:
        log.info(f"Cancelling {len(tasks_to_cancel)} running background tasks...")
        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()
                cancelled_task_count += 1
        # Wait for tasks to finish cancelling with a timeout
        log.info(f"Waiting up to 10 seconds for {cancelled_task_count} tasks to cancel...")
        # Use return_exceptions=True to prevent gather from stopping if one task raises non-CancelledError
        # gather itself doesn't have a timeout, so wrap it
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                timeout=10.0
            )
            log.info(f"Gather results received for cancelled tasks: {len(results)}")
            # Log any exceptions that occurred during task cancellation/execution
            for i, result in enumerate(results):
                 # Get task name robustly
                 task_name = "Unknown Task"
                 try:
                     # Check if index is valid before accessing tasks_to_cancel
                     if i < len(tasks_to_cancel):
                         task_name = tasks_to_cancel[i].get_name()
                 except Exception: pass # Ignore if name cannot be retrieved

                 if isinstance(result, asyncio.CancelledError):
                      log.debug(f"Task '{task_name}' cancelled successfully.")
                 elif isinstance(result, Exception):
                      # Log exception with traceback if log level is DEBUG or lower
                      log_exc_info = log.level <= logging.DEBUG
                      log.error(f"Task '{task_name}' raised an exception during shutdown: {result}", exc_info=result if log_exc_info else False)
                 # else: log task result if needed
        except asyncio.TimeoutError:
             log.error("Timeout waiting for background tasks to cancel. Some tasks may not have terminated cleanly.")
        except Exception as e:
             log.error(f"Error gathering cancelled tasks: {e}", exc_info=True)


    # Close the CCXT exchange connection
    if exchange and hasattr(exchange, 'close'):
        # Check if already closed to avoid errors
        is_closed = getattr(exchange, 'closed', False) # Use getattr for safety
        if not is_closed:
            log.info("Closing CCXT exchange connection (timeout 10s)...")
            try:
                # Add timeout for closing connection
                await asyncio.wait_for(exchange.close(), timeout=10.0)
                log.info("Exchange connection closed.")
            except asyncio.TimeoutError:
                 log.error("Timeout closing exchange connection during shutdown.")
            except Exception as e:
                log.error(f"Error closing exchange connection during shutdown: {e}", exc_info=True)
        else:
            log.info("Exchange connection already closed.")

    log.warning("Shutdown sequence complete. Exiting.")
    # Allow logs to flush before exiting - get loop and call stop/close if needed
    await asyncio.sleep(0.5) # Simple delay for flushing

    # Explicitly exit - needed if shutdown is called not from main loop exit or signal handler
    # Use os._exit for a more forceful exit if sys.exit hangs (e.g., due to non-daemon threads or loop issues)
    # sys.exit(0) # Preferred, but might hang in some complex async scenarios
    os._exit(0) # Force exit after cleanup attempt

async def close_open_position_on_exit():
     """Closes any open position for the configured symbol during shutdown. USE WITH CAUTION."""
     log.warning("Executing close_open_position_on_exit (Configurable feature)...")
     # Ensure exchange is still usable (might fail if connection closed early)
     if not exchange or not market:
         log.error("Cannot check/close position on exit: Exchange or market not available.")
         return
     # Check if connection seems closed
     if getattr(exchange, 'closed', True):
          log.error("Cannot check/close position on exit: Exchange connection appears closed.")
          return

     try:
         # Fetch position one last time using a short timeout?
         # Need to be careful not to hang shutdown sequence
         log.info("Checking for open position to close on exit...")
         # Force fetch by resetting timer
         global last_position_check_time
         async with position_lock:
             last_position_check_time = 0
         # Use get_current_position which has internal locking and error handling
         pos_data = await get_current_position(config['symbol'])

         # Check if pos_data is valid before accessing size
         if pos_data and pos_data.get('size', Decimal(0)) > Decimal(0):
             log.warning(f"{Fore.YELLOW}Found open {pos_data['side']} position (Size: {pos_data['size']}). Attempting to close...{Style.RESET_ALL}")
             # Ensure close_position is robust and handles potential errors gracefully during shutdown
             close_order_info = await close_position(config['symbol'], pos_data)
             if close_order_info and not close_order_info.get('info', {}).get('error'):
                 log.info("Close order placed successfully on exit.")
                 await asyncio.sleep(2.0) # Give order time to process and potentially fill
                 # Final check (best effort)
                 async with position_lock:
                     last_position_check_time = 0
                 final_pos = await get_current_position(config['symbol'])
                 if final_pos and final_pos.get('size', Decimal(0)).is_zero():
                      log.info("Position confirmed closed on exit.")
                 elif final_pos is None:
                      log.error("Failed to confirm position closure on exit: Could not fetch final position state.")
                 else:
                      log.error(f"Failed to confirm position closure on exit. Final state: {final_pos}")
             elif close_order_info and close_order_info.get('info', {}).get('alreadyFlatOrChanged'):
                  log.warning("Position was already flat or changed when attempting close on exit.")
             else:
                  log.error("Failed to place close order on exit.")
         elif pos_data is None:
              log.error("Could not check position on exit due to API error.")
         else:
             log.info("No open position found to close on exit.")
     except Exception as e:
         log.error(f"Error during position close on exit: {e}", exc_info=True)


# --- Main Application Entry Point ---
async def main():
    """Main asynchronous function to initialize and run the bot."""
    global config, exchange, market, strategy_instance, latest_dataframe, running_tasks, last_ws_update_time, startup_time, current_position

    startup_time = time.monotonic() # Record startup time for WS health check grace period

    print(Fore.MAGENTA + Style.BRIGHT + "\n~~~ Pyrmethus Volumatic+OB Trading Bot Starting (CCXT Async - Enhanced) ~~~" + Style.RESET_ALL)
    print(f"Timestamp: {datetime.datetime.now(datetime.timezone.utc).isoformat()}")

    # Load configuration first
    config = load_config() # Exits on failure

    # Setup logging based on config
    setup_logging(config.get("log_level", "INFO"))
    log.info("Logging configured.")
    # Use default=str for types not serializable by default (like Decimal)
    log.debug(f"Full Config: {json.dumps(config, indent=2, default=str)}")

    # Validate API Keys early
    if not API_KEY or not API_SECRET:
        log.critical("CRITICAL: BYBIT_API_KEY or BYBIT_API_SECRET not set in .env file.")
        sys.exit(1)
    log.info(f"API Key found (ending with ...{API_KEY[-4:]})")

    # Connect to exchange
    exchange = await connect_ccxt()
    if not exchange:
        log.critical("Failed to connect to exchange. Exiting.")
        sys.exit(1)

    # Load market info for the target symbol
    market = await load_exchange_market(config['symbol'])
    if not market:
        log.critical(f"Failed to load market data for {config['symbol']}. Exiting.")
        if exchange and hasattr(exchange, 'close'): await exchange.close() # Clean up connection
        sys.exit(1)

    # Attempt to set leverage (log warning/error if fails, but continue)
    try:
        leverage_to_set = config['order'].get('leverage') # Can be int or float string
        if leverage_to_set: # Only set if leverage is specified in config
             await set_leverage(config['symbol'], leverage_to_set)
        else:
             log.info("Leverage setting skipped: 'leverage' not specified in config['order'].")
    except (ValueError, TypeError, KeyError) as e:
         log.error(f"Invalid leverage configuration: {e}. Skipping leverage setting.")
    except Exception as e:
         log.error(f"Unexpected error during initial leverage setting: {e}", exc_info=True)


    # Initialize Strategy Engine
    try:
        strategy_params = config.get('strategy', {}).get('params', {})
        # Pass market data and potentially config to strategy for precision/limits/settings access
        strategy_instance = VolumaticOBStrategy(market=market, config=config, **strategy_params)
        log.info(f"Strategy '{type(strategy_instance).__name__}' initialized.")
        # Log strategy parameters being used
        log.debug(f"Strategy Params: {strategy_params}")
        # Log min data length required by strategy
        min_data_len_strat = getattr(strategy_instance, 'min_data_len', 50) # Use strategy's min_len or default
        log.info(f"Strategy requires minimum {min_data_len_strat} data points.")

    except Exception as e:
        log.critical(f"Failed to initialize strategy: {e}. Check config.json['strategy']['params'] and strategy.py.", exc_info=True)
        if exchange and hasattr(exchange, 'close'): await exchange.close()
        sys.exit(1)

    # Fetch Initial Historical Data
    async with data_lock: # Lock dataframe during initial population
        initial_fetch_limit = config.get('data', {}).get('fetch_limit', 750)
        # Ensure fetch limit is at least what the strategy needs + buffer
        required_fetch_limit = min_data_len_strat + config.get('data', {}).get('fetch_buffer', 50)
        if initial_fetch_limit < required_fetch_limit:
            log.warning(f"Configured fetch_limit ({initial_fetch_limit}) is less than required by strategy + buffer ({required_fetch_limit}). Increasing fetch limit.")
            initial_fetch_limit = required_fetch_limit

        latest_dataframe = await fetch_initial_data(
            config['symbol'],
            config['timeframe'],
            initial_fetch_limit
        )
        if latest_dataframe is None or latest_dataframe.empty: # Check for None or empty DF
            log.critical("Failed to fetch initial market data or data was empty. Exiting.")
            if exchange and hasattr(exchange, 'close'): await exchange.close()
            sys.exit(1)
        elif len(latest_dataframe) < min_data_len_strat:
            log.warning(f"Initial data fetched ({len(latest_dataframe)}) is less than minimum required by strategy ({min_data_len_strat}). Strategy results may be unreliable initially.")
            # Still run initial analysis, strategy should handle insufficient data gracefully
        # else: # Removed else, run initial analysis even if data is slightly short

        # Run initial analysis on historical data if dataframe is available
        log.info("Running initial analysis on historical data...")
        try:
            # Make a copy for analysis
            df_copy = latest_dataframe.copy()
            initial_results = strategy_instance.update(df_copy)

            # --- Check for errors during initial analysis ---
            if initial_results is None or not isinstance(initial_results, dict):
                 log.error("Initial strategy analysis returned None or invalid type. Check strategy logic and data.")
                 # Decide whether to proceed or exit - Proceeding for now, strategy might recover
            else:
                 trend_val = initial_results.get('current_trend')
                 trend_str = 'UP' if trend_val is True else 'DOWN' if trend_val is False else 'UNDETERMINED'
                 # Get precision from strategy or default
                 price_prec_attr = getattr(strategy_instance, 'price_precision', None)
                 # Fallback to market precision if strategy doesn't provide it
                 price_prec = price_prec_attr if isinstance(price_prec_attr, int) else int(market.get('precision', {}).get('price', 2))
                 last_close_val = initial_results.get('last_close')
                 close_str = f"{last_close_val:.{price_prec}f}" if last_close_val is not None and price_prec is not None else "N/A"

                 log.info(f"Initial Analysis Complete: Trend={trend_str}, "
                          f"Last Close={close_str}, "
                          f"Initial Signal={initial_results.get('last_signal', 'N/A')}")

        except AttributeError as e:
             # Catch potential numpy/fillna error during initial analysis too
             if "'numpy.ndarray' object has no attribute 'fillna'" in str(e):
                  log.error(f"{Fore.RED}CRITICAL ERROR during initial strategy analysis: {e}. FIX REQUIRED IN strategy.py.{Style.RESET_ALL}", exc_info=True)
                  # Exit might be necessary if strategy can't initialize
                  if exchange and hasattr(exchange, 'close'): await exchange.close()
                  sys.exit(1)
             else:
                  log.error(f"AttributeError during initial strategy analysis: {e}", exc_info=True)
        except Exception as e:
             log.error(f"Unexpected error during initial strategy analysis: {e}", exc_info=True)
             # Decide whether to continue or exit based on severity - Proceeding for now


    # Fetch initial position state (REST call)
    initial_pos_data = await get_current_position(config['symbol']) # Updates global current_position
    if initial_pos_data is None:
        log.critical("Failed to fetch initial position state due to API error. Exiting.")
        if exchange and hasattr(exchange, 'close'): await exchange.close()
        sys.exit(1)
    # Use the updated global current_position for logging
    log.info(f"Initial Position State: {current_position['side']} (Size: {current_position['size']}, Entry: {current_position['entry_price']})")

    # --- Start Background Tasks ---
    log.info(f"{Fore.CYAN}Setup complete. Starting WebSocket watchers and periodic checks...{Style.RESET_ALL}")
    log.info(f"Trading Mode: {config.get('mode', MODE_LIVE)}")
    log.info(f"Symbol: {config['symbol']} | Timeframe: {config['timeframe']}")

    # Initialize WS update time to now to avoid immediate timeout warning
    last_ws_update_time = time.monotonic()

    # Create and store tasks for graceful shutdown
    # Use specific names for easier debugging/monitoring
    tasks_to_start = []
    # Kline watcher is critical
    kline_task = asyncio.create_task(watch_kline_loop(config['symbol'], config['timeframe']), name="KlineWatcher")
    tasks_to_start.append(kline_task)

    # Optional watchers based on config/needs and exchange support
    if config.get('websockets', {}).get('watch_positions', True): # Default to true if key missing
         if exchange.has.get('watchPositions'):
             tasks_to_start.append(asyncio.create_task(watch_positions_loop(config['symbol']), name="PositionWatcher"))
         else:
             log.warning("Position watching via WebSocket configured but not supported by exchange. Skipping.")
    if config.get('websockets', {}).get('watch_orders', True): # Default to true if key missing
         if exchange.has.get('watchOrders'):
             tasks_to_start.append(asyncio.create_task(watch_orders_loop(config['symbol']), name="OrderWatcher"))
         else:
             log.warning("Order watching via WebSocket configured but not supported by exchange. Skipping.")

    # Periodic health/position check
    tasks_to_start.append(asyncio.create_task(periodic_check_loop(), name="PeriodicChecker"))

    # Add tasks to the global set
    running_tasks.update(tasks_to_start)

    # Keep main running, monitor essential tasks, and handle shutdown signal
    log.info("Main loop running. Monitoring essential tasks... (Press Ctrl+C to stop)")
    while not stop_event.is_set():
         # Check if essential tasks (like kline watcher) are still running
         # Use the stored kline_task variable
         if kline_task.done():
              log.critical(f"{Fore.RED}{Style.BRIGHT}CRITICAL: Kline watcher task has terminated unexpectedly!{Style.RESET_ALL}")
              try:
                   # This will raise the exception if the task failed
                   kline_task.result()
              except asyncio.CancelledError:
                   log.warning("Kline watcher was cancelled (likely during shutdown).") # Expected during shutdown
              except Exception as e:
                   # Log the actual error that caused the task to terminate
                   log.critical(f"Kline watcher failed with error: {e}", exc_info=True) # Log traceback

              log.critical("Attempting to stop bot gracefully due to essential task failure...")
              # Trigger shutdown without waiting for OS signal
              # Use create_task to avoid blocking main loop if shutdown takes time
              # Ensure shutdown task itself is not added to running_tasks to avoid self-cancellation issues
              # Check if shutdown isn't already running
              if not stop_event.is_set():
                   # Use a local variable to avoid potential race condition with global shutdown_task_ref
                   _shutdown_task = asyncio.create_task(shutdown(signal_type=None), name="CriticalShutdown")
                   # Optionally add to running_tasks if you want shutdown itself to be cancellable? No, usually not.
              break # Exit main monitoring loop

         # Heartbeat sleep for the main loop
         await asyncio.sleep(5) # Check tasks every 5 seconds

    log.info("Main loop finished.")


if __name__ == "__main__":
    # Logging is already set up at module level now

    # Get the asyncio event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        log.debug("No running event loop found, creating new one.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Setup signal handlers for graceful shutdown (SIGINT: Ctrl+C, SIGTERM: kill)
    sig_handled = False
    shutdown_task_ref: Optional[asyncio.Task] = None # Keep track of the shutdown task to prevent duplicates

    def _handle_signal(sig: signal.Signals):
        """Internal function to handle OS signals and initiate shutdown."""
        global shutdown_task_ref
        if not stop_event.is_set() and (shutdown_task_ref is None or shutdown_task_ref.done()):
            log.warning(f"Received signal {sig.name}. Initiating shutdown...")
            # Ensure shutdown runs within the loop's context
            # Check if loop is running before creating task
            try:
                current_loop = asyncio.get_running_loop()
                if current_loop.is_running():
                    # Create shutdown task but don't await it here
                    shutdown_task_ref = current_loop.create_task(shutdown(sig), name=f"ShutdownHandler_{sig.name}")
                else:
                    # This case should be rare if signal is handled by running loop
                    log.error(f"Cannot initiate shutdown for signal {sig.name}: Event loop is not running.")
            except RuntimeError:
                 log.error(f"Cannot initiate shutdown for signal {sig.name}: No running event loop.")
        else:
            log.warning(f"Received signal {sig.name}, but shutdown already in progress or requested.")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            # Use add_signal_handler for proper async handling
            loop.add_signal_handler(sig, _handle_signal, sig)
            sig_handled = True
            log.debug(f"Asyncio signal handler for {sig.name} added.")
        except NotImplementedError:
            # Fallback for platforms where add_signal_handler might fail (e.g., some Windows setups)
            log.warning(f"Asyncio signal handler for {sig.name} not supported. Using signal.signal fallback.")
            try:
                # Wrap the handler call in loop.call_soon_threadsafe if signal might arrive from different thread
                # This is generally safer for the fallback.
                signal.signal(sig, lambda s, f: loop.call_soon_threadsafe(_handle_signal, signal.Signals(s)))
                sig_handled = True
            except (ValueError, OSError, RuntimeError, AttributeError, TypeError) as e:
                 # ValueError: signal only works in main thread
                 # OSError: [Errno 22] Invalid argument (happens on Windows sometimes)
                 # RuntimeError: Cannot schedule new futures after shutdown
                 # AttributeError: 'NoneType' object has no attribute 'call_soon_threadsafe' (if loop is None)
                 # TypeError: 'asyncio.unix_events._UnixSelectorEventLoop' object is not callable (if loop is closed?)
                 log.error(f"Failed to set fallback signal handler for {sig.name}: {e}")

    if not sig_handled:
         log.warning("No signal handlers could be set. Graceful shutdown via Ctrl+C or SIGTERM might not work reliably.")

    main_task = None
    try:
        # Create the main task here, inside the try block where the loop is available
        main_task = loop.create_task(main(), name="MainBotLoop")
        # Run the event loop until the main task completes (or is cancelled)
        loop.run_until_complete(main_task)

    except asyncio.CancelledError:
         log.info("Main task cancelled (likely during shutdown).")
    except KeyboardInterrupt: # Catch Ctrl+C if signal handler fails or isn't set
         log.warning("KeyboardInterrupt caught directly. Initiating shutdown...")
         if not stop_event.is_set() and (shutdown_task_ref is None or shutdown_task_ref.done()):
              # Manually trigger shutdown coroutine if KeyboardInterrupt bypasses signal handler
              # Ensure loop is running to schedule shutdown
              try:
                  current_loop = asyncio.get_running_loop()
                  if current_loop.is_running():
                      # Create and run the shutdown task until completion
                      shutdown_task_ref = current_loop.create_task(shutdown(signal.SIGINT), name=f"ShutdownHandler_KeyboardInterrupt")
                      try:
                          loop.run_until_complete(shutdown_task_ref)
                      except RuntimeError as e:
                          log.error(f"Error running shutdown task after KeyboardInterrupt: {e}") # e.g., loop stopped unexpectedly
                  else:
                      log.error("Loop not running, cannot initiate shutdown via KeyboardInterrupt.")
              except RuntimeError:
                   log.error("No running loop, cannot initiate shutdown via KeyboardInterrupt.")
    except Exception as e:
         log.critical(f"Unhandled exception in main execution: {e}", exc_info=True)
         # Attempt graceful shutdown even on unexpected main errors
         if not stop_event.is_set() and (shutdown_task_ref is None or shutdown_task_ref.done()):
              log.critical("Attempting shutdown after unhandled main exception...")
              try:
                  current_loop = asyncio.get_running_loop()
                  if current_loop.is_running():
                      # Create and run the shutdown task until completion
                      shutdown_task_ref = current_loop.create_task(shutdown(signal_type=None), name="ShutdownHandler_MainException")
                      try:
                          loop.run_until_complete(shutdown_task_ref)
                      except RuntimeError as e:
                          log.error(f"Error running shutdown task after main exception: {e}")
                  else:
                       log.error("Loop not running, cannot initiate shutdown after main exception.")
              except RuntimeError:
                   log.error("No running loop, cannot initiate shutdown after main exception.")
    finally:
         log.info("Entering final cleanup phase...")

         # Ensure shutdown task completes if it was started and loop is running
         if shutdown_task_ref and not shutdown_task_ref.done():
             try:
                 current_loop = asyncio.get_running_loop()
                 if current_loop.is_running():
                     log.info("Waiting for shutdown task to complete...")
                     # Wait for the already running shutdown task
                     loop.run_until_complete(shutdown_task_ref)
             except RuntimeError:
                 log.warning("Could not wait for shutdown task: No running event loop.")
             except Exception as e:
                 log.error(f"Error waiting for shutdown task: {e}")


         # Final check to ensure exchange connection is closed
         if exchange and hasattr(exchange, 'close') and not getattr(exchange, 'closed', True):
             log.warning("Exchange connection still open after main loop exit. Attempting final close.")
             try:
                 # Run close within the loop if it's still running
                 current_loop = asyncio.get_running_loop()
                 if current_loop.is_running():
                     loop.run_until_complete(exchange.close())
                 else:
                     # If loop is stopped, cannot run async close cleanly.
                     log.error("Event loop stopped. Cannot run final exchange close asynchronously.")
             except RuntimeError:
                 log.error("Event loop stopped or unavailable. Cannot run final exchange close asynchronously.")
             except Exception as e:
                 log.error(f"Error during final exchange close: {e}")

         # Cancel any remaining tasks just in case (e.g., tasks created outside running_tasks set)
         try:
              # Check if loop is available and running before accessing tasks
              current_loop = asyncio.get_running_loop()
              if current_loop and not current_loop.is_closed():
                   current_task = asyncio.current_task(loop=current_loop) # Get current task if any
                   all_tasks = asyncio.all_tasks(loop=current_loop)
                   # Exclude self, main_task, shutdown_task if they exist and are tasks
                   tasks_to_exclude = {t for t in [current_task, main_task, shutdown_task_ref] if isinstance(t, asyncio.Task)}
                   remaining_tasks = [t for t in all_tasks if t not in tasks_to_exclude and not t.done()]
                   if remaining_tasks:
                        log.warning(f"Cancelling {len(remaining_tasks)} potentially lingering tasks...")
                        for task in remaining_tasks:
                             task.cancel()
                        # Wait briefly for cancellations if loop is running
                        if current_loop.is_running():
                            # Use a timeout to prevent hanging indefinitely
                            try:
                                # Need an async context to await gather
                                async def gather_remaining():
                                    await asyncio.gather(*remaining_tasks, return_exceptions=True)
                                # Run this temporary async function within the loop
                                loop.run_until_complete(asyncio.wait_for(gather_remaining(), timeout=5.0))
                                log.info("Lingering task cancellation complete.")
                            except asyncio.TimeoutError:
                                log.error("Timeout waiting for lingering tasks to cancel.")
                            except Exception as gather_exc:
                                log.error(f"Error gathering lingering tasks: {gather_exc}")
                        else:
                            log.warning("Loop stopped, cannot wait for lingering task cancellation.")
              else:
                   log.warning("Event loop closed or unavailable, cannot perform final task cleanup.")
         except RuntimeError:
              log.warning("Event loop unavailable, cannot perform final task cleanup.")
         except Exception as e:
              log.error(f"Error during final task cleanup: {e}")


         # Close the loop if it's not already closed (may not be necessary/recommended in newer Python)
         try:
             # Use get_event_loop_policy().get_event_loop() for safer access
             current_loop = asyncio.get_event_loop_policy().get_event_loop()
             if current_loop and not current_loop.is_closed():
                 log.info("Closing asyncio event loop.")
                 current_loop.close()
         except RuntimeError:
             log.debug("No running event loop to close or already closed.")
         except Exception as e:
             log.error(f"Error closing event loop: {e}")

         log.info(f"{Fore.MAGENTA}~~~ Pyrmethus Volumatic+OB Trading Bot Finished ~~~{Style.RESET_ALL}")
         logging.shutdown() # Ensure all handlers are flushed/closed
