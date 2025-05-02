
                     % python mainx.py                                                      Loaded environment variables from .env file.                                                                                                  ~~~ Pyrmethus Volumatic+OB Trading Bot Initializing ~~~                2025-05-01 19:04:29 - PyrmethusVolumaticBot - INFO - Logging level set to: INFO                                                               2025-05-01 19:04:29 - PyrmethusVolumaticBot - INFO - Registered shutdown signal handlers.                                                     2025-05-01 19:04:29 - PyrmethusVolumaticBot - INFO - Connecting to Bybit Mainnet HTTP API...                                                  2025-05-01 19:04:31 - PyrmethusVolumaticBot - INFO - Successfully connected. Server time: 2025-05-01 19:04:30.858836                          2025-05-01 19:04:31 - PyrmethusVolumaticBot - INFO - Fetched market info for DOTUSDT.                                                         2025-05-01 19:04:31 - PyrmethusVolumaticBot - INFO - Market Info: TickSize=0.0001, QtyStep=0.1                                                2025-05-01 19:04:31 - PyrmethusVolumaticBot - INFO - Attempting to set leverage for DOTUSDT to 25x...                                         2025-05-01 19:04:32 - PyrmethusVolumaticBot - ERROR - Exception setting leverage: leverage not modified (ErrCode: 110043) (ErrTime: 00:04:32).Request → POST https://api.bybit.com/v5/position/set-leverage: {"category": "linear", "symbol": "DOTUSDT", "buyLeverage": "25.0", "sellLeverage": "25.0"}.                                                           2025-05-01 19:04:32 - PyrmethusVolumaticBot - INFO - Strategy 'VolumaticOBStrategy' initialized successfully.                                 2025-05-01 19:04:32 - PyrmethusVolumaticBot - INFO - Strategy requires minimum 1000 candles.                                                  2025-05-01 19:04:32 - PyrmethusVolumaticBot - INFO - Fetching initial 1100 klines for DOTUSDT (interval: 5)...
2025-05-01 19:04:33 - PyrmethusVolumaticBot - INFO - Fetched 1000 initial candles. Data spans from 2025-04-28 12:45:00 to 2025-05-02 00:00:00 2025-05-01 19:04:33 - PyrmethusVolumaticBot - INFO - Running initial analysis on historical data...
2025-05-01 19:04:33 - PyrmethusVolumaticBot - INFO - Initial Analysis Complete: Current Trend Estimate = Undetermined, Last Signal State = HOLD                                                                      2025-05-01 19:04:33 - PyrmethusVolumaticBot - INFO - Starting WebSocket thread...
2025-05-01 19:04:33 - PyrmethusVolumaticBot - INFO - WebSocket thread starting...                                                             2025-05-01 19:04:33 - PyrmethusVolumaticBot - INFO - Initializing WebSocket connection...                                                     2025-05-01 19:04:33 - PyrmethusVolumaticBot - INFO - Waiting up to 20s for WebSocket connection and subscription confirmation...
2025-05-01 19:04:33 - PyrmethusVolumaticBot - ERROR - WebSocket run_forever error or initialization failed: _WebSocketManager.__init__() got an unexpected keyword argument 'message_handler'                        2025-05-01 19:04:33 - PyrmethusVolumaticBot - INFO - Attempting to reconnect WebSocket after error in 15 seconds...                           2025-05-01 19:04:48 - PyrmethusVolumaticBot - INFO - Initializing WebSocket connection...
2025-05-01 19:04:48 - PyrmethusVolumaticBot - ERROR - WebSocket run_forever error or initialization failed: _WebSocketManager.__init__() got an unexpected keyword argument 'message_handler'                        2025-05-01 19:04:48 - PyrmethusVolumaticBot - INFO - Attempting to reconnect WebSocket after error in 15 seconds...
2025-05-01 19:04:53 - PyrmethusVolumaticBot - ERROR - WebSocket did not confirm subscription within 20s. Check logs, connection, credentials.
2025-05-01 19:04:53 - PyrmethusVolumaticBot - INFO - Stopping WebSocket connection...
2025-05-01 19:04:53 - PyrmethusVolumaticBot - INFO - WebSocket object not found (might be between reconnect attempts), relying on stop_event. 2025-05-01 19:04:53 - PyrmethusVolumaticBot - INFO - WebSocket thread finished.                                                               2025-05-01 19:04:53 - PyrmethusVolumaticBot - INFO - Waiting for WebSocket thread to join...                                                  2025-05-01 19:04:53 - PyrmethusVolumaticBot - INFO - WebSocket thread joined successfully.                                                    2025-05-01 19:04:53 - PyrmethusVolumaticBot - INFO - WebSocket connection stopped.
2025-05-01 19:04:53 - PyrmethusVolumaticBot - CRITICAL - Failed to start WebSocket connection. Exiting.                                       2025-05-01 19:04:53 - PyrmethusVolumaticBot - INFO - WebSocket thread already stopped or not initialized.
import os
import sys
import json
import time
import datetime
import logging
import signal
import threading
from decimal import Decimal, ROUND_DOWN, ROUND_UP, InvalidOperation, getcontext
from typing import Dict, Optional, Any, Tuple, List  # Added Tuple, List

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from colorama import init, Fore, Style
from pybit.unified_trading import HTTP, WebSocket

# Import strategy class and type hints from strategy.py
try:
    from strategy import VolumaticOBStrategy, AnalysisResults
except ImportError as e:
    print(f"ERROR: Could not import from strategy.py: {e}", file=sys.stderr)
    print("Ensure strategy.py is in the same directory.", file=sys.stderr)
    sys.exit(1)

# --- Initialize Colorama ---
# autoreset=True ensures color resets after each print
init(autoreset=True)

# --- Set Decimal Precision ---
# Set precision high enough for crypto prices and quantities
getcontext().prec = 18

# --- Load Environment Variables ---
# Load API keys and settings from .env file in the same directory
if load_dotenv():
    print("Loaded environment variables from .env file.")
else:
    print("WARN: .env file not found or empty. API keys might be missing.")

API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
TESTNET = os.getenv("BYBIT_TESTNET", "False").lower() in ("true", "1", "t")

# --- Global Variables & State ---
config: Dict[str, Any] = {}
session: Optional[HTTP] = None
ws: Optional[WebSocket] = None
ws_thread: Optional[threading.Thread] = None
ws_connected = threading.Event() # Signals WebSocket connection status
stop_event = threading.Event() # Used for graceful shutdown coordination
latest_dataframe: Optional[pd.DataFrame] = None
strategy_instance: Optional[VolumaticOBStrategy] = None
market_info: Optional[Dict[str, Any]] = None
data_lock = threading.Lock() # Protects access to latest_dataframe
# position_lock = threading.Lock() # Use if modifying position state based on WS (currently not)
order_lock = threading.Lock() # Protects order placement/closing logic to prevent race conditions
last_position_check_time: float = 0
POSITION_CHECK_INTERVAL: int = 10 # Default, overridden by config

# --- Logging Setup ---
# Use a single logger instance throughout the application
log = logging.getLogger("PyrmethusVolumaticBot")
log_level = logging.INFO # Default, will be overridden by config

def setup_logging(level_str: str = "INFO"):
    """Configures logging format, level, and handlers."""
    global log_level
    try:
        log_level = getattr(logging, level_str.upper())
    except AttributeError:
        log_level = logging.INFO
        print(f"WARN: Invalid log level '{level_str}'. Defaulting to INFO.", file=sys.stderr)

    log.setLevel(log_level)

    # Prevent adding multiple handlers if called again
    if not log.handlers:
        # Console Handler (StreamHandler)
        ch = logging.StreamHandler(sys.stdout) # Use stdout for console logs
        ch.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        log.addHandler(ch)

        # Optional: File Handler (Uncomment and configure if needed)
        # log_filename = f"bot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        # try:
        #     fh = logging.FileHandler(log_filename)
        #     fh.setLevel(log_level)
        #     fh.setFormatter(formatter)
        #     log.addHandler(fh)
        # except Exception as e:
        #     log.error(f"Failed to set up file logging to {log_filename}: {e}")

    # Prevent messages from propagating to the root logger if handlers are set
    log.propagate = False

# --- Configuration Loading ---
def load_config(path: str = "config.json") -> Dict[str, Any]:
    """Loads and validates the configuration from a JSON file."""
    global POSITION_CHECK_INTERVAL
    try:
        with open(path, 'r', encoding='utf-8') as f: # Specify encoding
            conf = json.load(f)
        log.info(f"Configuration loaded successfully from '{path}'.")

        # Basic validation (add more checks as needed)
        required_keys = ["symbol", "interval", "mode", "log_level", "order", "strategy", "data", "websocket"]
        if not all(key in conf for key in required_keys):
            missing = [key for key in required_keys if key not in conf]
            raise ValueError(f"Config file missing one or more required top-level keys: {missing}")
        if not all(key in conf["order"] for key in ["risk_per_trade_percent", "leverage", "type"]):
             raise ValueError("Config file missing required keys in 'order' section (risk_per_trade_percent, leverage, type).")
        if not all(key in conf["strategy"] for key in ["params", "stop_loss"]):
             raise ValueError("Config file missing required keys in 'strategy' section (params, stop_loss).")
        if not all(key in conf["data"] for key in ["fetch_limit", "max_df_len"]):
             raise ValueError("Config file missing required keys in 'data' section (fetch_limit, max_df_len).")
        if not all(key in conf["websocket"] for key in ["ping_interval", "connect_timeout"]):
             raise ValueError("Config file missing required keys in 'websocket' section (ping_interval, connect_timeout).")

        # Validate specific values
        if conf["order"]["type"] not in ["Market", "Limit"]:
            raise ValueError(f"Invalid order type '{conf['order']['type']}'. Must be 'Market' or 'Limit'.")
        if not isinstance(conf["order"]["risk_per_trade_percent"], (int, float)) or not 0 < conf["order"]["risk_per_trade_percent"] <= 100:
             raise ValueError("risk_per_trade_percent must be a number between 0 (exclusive) and 100 (inclusive).")
        if not isinstance(conf["order"]["leverage"], int) or not 1 <= conf["order"]["leverage"] <= 100: # Adjust max leverage based on Bybit limits
             raise ValueError("leverage must be an integer between 1 and 100 (check Bybit limits for your symbol/tier).")

        # Update global interval if set in config
        POSITION_CHECK_INTERVAL = int(conf.get("position_check_interval", 10))
        if POSITION_CHECK_INTERVAL <= 0:
             log.warning("position_check_interval must be positive. Using default 10s.")
             POSITION_CHECK_INTERVAL = 10

        return conf
    except FileNotFoundError:
        log.critical(f"CRITICAL: Configuration file '{path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        log.critical(f"CRITICAL: Configuration file '{path}' contains invalid JSON: {e}")
        sys.exit(1)
    except ValueError as e:
        log.critical(f"CRITICAL: Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        log.critical(f"CRITICAL: Unexpected error loading configuration: {e}", exc_info=True)
        sys.exit(1)

# --- Bybit API Interaction ---
def connect_bybit() -> Optional[HTTP]:
    """Establishes and tests connection to Bybit HTTP API."""
    if not API_KEY or not API_SECRET:
        log.critical("CRITICAL: Bybit API Key or Secret not found. Check .env file or environment variables.")
        sys.exit(1)
    try:
        log.info(f"Connecting to Bybit {'Testnet' if TESTNET else 'Mainnet'} HTTP API...")
        s = HTTP(testnet=TESTNET, api_key=API_KEY, api_secret=API_SECRET)
        # Test connection with a simple read-only call
        server_time_resp = s.get_server_time()
        if server_time_resp.get('retCode') == 0:
            server_ts_nano = server_time_resp.get('result', {}).get('timeNano', '0')
            server_ts = int(server_ts_nano) / 1e9 if server_ts_nano.isdigit() else time.time()
            log.info(f"Successfully connected. Server time: {datetime.datetime.fromtimestamp(server_ts)}")
            return s
        else:
            log.critical(f"CRITICAL: Failed to connect or verify connection: {server_time_resp.get('retMsg', 'Unknown Error')} (Code: {server_time_resp.get('retCode')})")
            return None
    except Exception as e:
        log.critical(f"CRITICAL: Exception during Bybit HTTP API connection: {e}", exc_info=True)
        return None

def get_market_info(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetches and validates instrument information for the specified symbol."""
    if not session:
        log.error("HTTP session not available for get_market_info.")
        return None
    try:
        log.debug(f"Fetching market info for {symbol}...")
        response = session.get_instruments_info(category="linear", symbol=symbol)
        if response.get('retCode') == 0 and response.get('result', {}).get('list'):
            info = response['result']['list'][0]
            log.info(f"Fetched market info for {symbol}.")
            log.debug(f"Market Info Details: {json.dumps(info, indent=2)}")

            # Validate presence of required filter details for precision and order limits
            price_filter = info.get('priceFilter')
            lot_filter = info.get('lotSizeFilter')
            if not price_filter or not lot_filter:
                 log.error(f"Market info for {symbol} is missing priceFilter or lotSizeFilter. Cannot proceed.")
                 return None

            required_price_keys = ['tickSize']
            required_lot_keys = ['qtyStep', 'minOrderQty', 'maxOrderQty']

            if not all(k in price_filter for k in required_price_keys) or \
               not all(k in lot_filter for k in required_lot_keys):
                log.error(f"Market info for {symbol} is missing required filter details "
                          f"(tickSize, qtyStep, minOrderQty, maxOrderQty). Cannot proceed.")
                return None

            # Convert to Decimal early for consistency and precision
            try:
                 info['priceFilter']['tickSize'] = Decimal(price_filter['tickSize'])
                 info['lotSizeFilter']['qtyStep'] = Decimal(lot_filter['qtyStep'])
                 info['lotSizeFilter']['minOrderQty'] = Decimal(lot_filter['minOrderQty'])
                 info['lotSizeFilter']['maxOrderQty'] = Decimal(lot_filter['maxOrderQty'])
            except (InvalidOperation, TypeError, KeyError) as e:
                 log.error(f"Could not convert market info filter values to Decimal for {symbol}: {e}")
                 return None

            return info
        else:
            log.error(f"Failed to get market info for {symbol}: {response.get('retMsg', 'Unknown Error')} "
                      f"(Code: {response.get('retCode', 'N/A')})")
            return None
    except Exception as e:
        # Log traceback only if debug level is enabled
        log.error(f"Exception fetching market info for {symbol}: {e}", exc_info=(log_level <= logging.DEBUG))
        return None

def fetch_initial_data(symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetches historical Klines data and prepares the initial DataFrame."""
    if not session:
         log.error("HTTP session not available for fetch_initial_data.")
         return None
    log.info(f"Fetching initial {limit} klines for {symbol} (interval: {interval})...")
    try:
        response = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            limit=min(limit, 1000) # Bybit V5 limit is 1000 per request
        )
        if response.get('retCode') == 0 and response.get('result', {}).get('list'):
            kline_list = response['result']['list']
            if not kline_list:
                 log.warning(f"Received empty kline list from Bybit for initial fetch of {symbol}/{interval}.")
                 # Return an empty DataFrame with expected columns
                 return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'turnover'])

            df = pd.DataFrame(kline_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            # Convert types immediately after creation
            df = df.astype({
                'timestamp': 'int64', 'open': 'float64', 'high': 'float64',
                'low': 'float64', 'close': 'float64', 'volume': 'float64', 'turnover': 'float64'
            })
            # Convert timestamp (milliseconds) to DatetimeIndex
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            # Bybit V5 returns oldest first, so sort chronologically (should already be sorted)
            df = df.sort_index()
            log.info(f"Fetched {len(df)} initial candles. Data spans from {df.index.min()} to {df.index.max()}")
            return df
        else:
            log.error(f"Failed to fetch initial klines for {symbol}/{interval}: {response.get('retMsg', 'Error')} "
                      f"(Code: {response.get('retCode', 'N/A')})")
            return None
    except Exception as e:
        log.error(f"Exception fetching initial klines: {e}", exc_info=(log_level <= logging.DEBUG))
        return None

def _get_flat_position_representation() -> Dict[str, Any]:
    """Returns a dictionary representing a flat position."""
    return {
        "size": Decimal(0),
        "side": "None",
        "avgPrice": Decimal(0),
        "liqPrice": Decimal(0),
        "unrealisedPnl": Decimal(0),
        "markPrice": Decimal(0),
        "leverage": Decimal(0)
    }

def get_current_position(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetches current position details for the symbol. Includes rate limiting.
    Returns a dictionary representing the position, or None if rate limited.
    On error or if flat, returns a dictionary representing a flat position.
    """
    global last_position_check_time
    if not session:
        log.error("HTTP session not available for get_current_position.")
        return _get_flat_position_representation() # Return flat on session error

    now = time.time()
    # Check if the interval has passed since the last check
    if now - last_position_check_time < POSITION_CHECK_INTERVAL:
        # log.debug("Skipping position check due to rate limit.")
        return None # Indicate check was skipped

    log.debug(f"Fetching position for {symbol}...")
    try:
        response = session.get_positions(category="linear", symbol=symbol)
        # Update check time regardless of success/failure to prevent spamming on errors
        last_position_check_time = time.time()

        if response.get('retCode') == 0 and response.get('result', {}).get('list'):
            # Bybit returns a list, even when filtering by symbol. Assume first entry is the relevant one.
            position = response['result']['list'][0]
            # Convert relevant fields to Decimal for precise calculations
            pos_data = {
                "size": Decimal(position.get('size', '0')),
                "side": position.get('side', 'None'), # 'Buy', 'Sell', or 'None'
                "avgPrice": Decimal(position.get('avgPrice', '0')),
                "liqPrice": Decimal(position.get('liqPrice', '0')) if position.get('liqPrice') else Decimal(0),
                "unrealisedPnl": Decimal(position.get('unrealisedPnl', '0')),
                "markPrice": Decimal(position.get('markPrice', '0')), # Useful for context
                "leverage": Decimal(position.get('leverage', '0')), # Confirm leverage setting
                # Add other fields if needed
            }
            # Treat negligible size as flat
            if pos_data['size'] < Decimal(market_info['lotSizeFilter']['minOrderQty']) / 10: # Arbitrary small fraction
                 log.debug(f"Position size {pos_data['size']} is negligible, treating as flat.")
                 return _get_flat_position_representation()

            log.debug(f"Position Data: Size={pos_data['size']}, Side={pos_data['side']}, AvgPrice={pos_data['avgPrice']}")
            return pos_data
        elif response.get('retCode') == 110001: # Parameter error (e.g., invalid symbol)
             log.error(f"Parameter error fetching position for {symbol}. Is symbol valid? {response.get('retMsg', '')}")
             return _get_flat_position_representation() # Assume flat on symbol error
        else:
            log.error(f"Failed to get position for {symbol}: {response.get('retMsg', 'Error')} "
                      f"(Code: {response.get('retCode', 'N/A')})")
            return _get_flat_position_representation() # Return flat representation on API error
    except Exception as e:
        last_position_check_time = time.time() # Update time even on exception
        log.error(f"Exception fetching position for {symbol}: {e}", exc_info=(log_level <= logging.DEBUG))
        return _get_flat_position_representation() # Return flat representation on exception

def get_wallet_balance(account_type: str = "UNIFIED", coin: str = "USDT") -> Optional[Decimal]:
    """Fetches account equity for risk calculation (UNIFIED account type)."""
    if not session:
        log.error("HTTP session not available for get_wallet_balance.")
        return None
    try:
        # V5 Unified Trading uses get_wallet_balance
        response = session.get_wallet_balance(accountType=account_type, coin=coin)
        if response.get('retCode') == 0 and response.get('result', {}).get('list'):
            # Unified account balance info is usually in the first item of the list
            balance_info_list = response['result']['list']
            if not balance_info_list:
                log.error(f"Empty balance list received for account type {account_type}, coin {coin}.")
                return None
            balance_info = balance_info_list[0]

            # Use 'equity' as the basis for risk calculation in Unified account
            if 'equity' in balance_info:
                equity = Decimal(balance_info['equity'])
                log.debug(f"Account Equity ({coin}): {equity}")
                if equity < 0:
                    log.warning(f"Account equity is negative: {equity}")
                return equity
            else:
                log.warning(f"Could not find 'equity' field in wallet balance response for {account_type} account.")
                # Fallback: Try 'totalAvailableBalance'? Less accurate for risk based on margin.
                if 'totalAvailableBalance' in balance_info:
                     avail_balance = Decimal(balance_info['totalAvailableBalance'])
                     log.warning(f"Falling back to totalAvailableBalance: {avail_balance}")
                     return avail_balance
                log.error("Neither 'equity' nor 'totalAvailableBalance' found in balance response.")
                return None
        else:
            log.error(f"Failed to get wallet balance: {response.get('retMsg', 'Unknown Error')} "
                      f"(Code: {response.get('retCode', 'N/A')})")
            return None
    except (InvalidOperation, TypeError) as e:
         log.error(f"Error converting wallet balance to Decimal: {e}")
         return None
    except Exception as e:
        log.error(f"Exception fetching wallet balance: {e}", exc_info=(log_level <= logging.DEBUG))
        return None

def calculate_order_qty(entry_price: float, sl_price: float, risk_percent: float) -> Optional[float]:
    """
    Calculates order quantity based on risk percentage, SL distance, and account equity.
    Rounds down to the nearest valid quantity step and checks against min/max limits.
    Uses Decimal for precision.
    """
    if not market_info or not strategy_instance:
        log.error("Market info or strategy instance not available for quantity calculation.")
        return None
    if not all(isinstance(p, (float, int)) and np.isfinite(p) for p in [entry_price, sl_price, risk_percent]):
        log.error(f"Invalid input for quantity calculation: entry={entry_price}, sl={sl_price}, risk%={risk_percent}")
        return None
    if risk_percent <= 0:
        log.error(f"Invalid risk percentage: {risk_percent}. Must be positive.")
        return None

    try:
        entry_decimal = Decimal(str(entry_price))
        sl_decimal = Decimal(str(sl_price))
        tick_size = strategy_instance.tick_size # Already Decimal
        qty_step = strategy_instance.qty_step # Already Decimal
        min_qty_decimal = market_info['lotSizeFilter']['minOrderQty'] # Already Decimal
        max_qty_decimal = market_info['lotSizeFilter']['maxOrderQty'] # Already Decimal

        # Ensure SL is meaningfully different from entry (at least one tick away)
        if abs(sl_decimal - entry_decimal) < tick_size:
            log.error(f"Stop loss price {sl_decimal} is too close to entry price {entry_decimal} "
                      f"(tick size: {tick_size}). Cannot calculate quantity.")
            return None

        balance = get_wallet_balance()
        if balance is None or balance <= 0:
            log.error(f"Cannot calculate order quantity: Invalid or zero balance ({balance}).")
            return None

        risk_amount = balance * (Decimal(str(risk_percent)) / 100)
        sl_distance_points = abs(entry_decimal - sl_decimal)

        if sl_distance_points <= 0: # Should be caught by tick_size check, but double-check
             log.error("Stop loss distance calculated as zero or negative. Cannot calculate quantity.")
             return None

        # For Linear contracts (XXX/USDT), PnL is in Quote currency (USDT).
        # Loss per contract = Qty (in Base) * SL_Distance (in Quote)
        # We want: Qty * SL_Distance <= Risk Amount
        # Qty (in Base Asset, e.g., BTC) = Risk Amount / SL_Distance
        qty_base = risk_amount / sl_distance_points

    except (InvalidOperation, ZeroDivisionError, TypeError) as e:
        log.error(f"Error during quantity calculation math: {e}")
        return None
    except Exception as e: # Catch any other unexpected errors
        log.error(f"Unexpected error during quantity calculation: {e}", exc_info=(log_level <= logging.DEBUG))
        return None

    # Round down to the minimum quantity step using Decimal
    qty_rounded_decimal = (qty_base // qty_step) * qty_step

    # Check against min/max order quantity
    if qty_rounded_decimal < min_qty_decimal:
        log.warning(f"Calculated quantity {qty_rounded_decimal} is below minimum ({min_qty_decimal}).")
        # Decision: Use min_qty (higher risk) or skip trade?
        # Current behavior: Use min_qty but warn about increased risk.
        qty_final_decimal = min_qty_decimal
        # Recalculate actual risk if using min_qty
        actual_risk_amount = min_qty_decimal * sl_distance_points
        actual_risk_percent = (actual_risk_amount / balance) * 100 if balance > 0 else Decimal(0)
        log.warning(f"Using minimum quantity {min_qty_decimal}. "
                    f"Actual Risk: {actual_risk_amount:.2f} USDT ({actual_risk_percent:.2f}%)")

    elif qty_rounded_decimal > max_qty_decimal:
        log.warning(f"Calculated quantity {qty_rounded_decimal} exceeds maximum ({max_qty_decimal}). Using maximum.")
        qty_final_decimal = max_qty_decimal
    else:
        qty_final_decimal = qty_rounded_decimal

    if qty_final_decimal <= 0:
        log.error(f"Final calculated quantity is zero or negative ({qty_final_decimal}). Cannot place order.")
        return None

    # Convert final Decimal quantity back to float for use elsewhere (API requires string anyway)
    qty_final_float = float(qty_final_decimal)

    log.info(f"Calculated Order Qty: {qty_final_float:.{strategy_instance.qty_precision}f} "
             f"(Balance={balance:.2f}, Risk={risk_percent}%, RiskAmt={risk_amount:.2f}, "
             f"SLDist={sl_distance_points:.{strategy_instance.price_precision}f})")
    return qty_final_float

def place_order(symbol: str, side: str, qty: float, price: Optional[float]=None, sl_price: Optional[float]=None, tp_price: Optional[float]=None) -> Optional[Dict]:
    """
    Places an order (Market or Limit) via Bybit API with optional SL/TP.
    Handles rounding, quantity checks, and basic error reporting.
    Uses an order_lock to prevent concurrent order placements.
    """
    if not session or not strategy_instance or not market_info:
        log.error("Cannot place order: Session, strategy instance, or market info missing.")
        return None
    if config.get("mode", "Live").lower() == "paper":
        qty_str = f"{qty:.{strategy_instance.qty_precision}f}"
        price_str = f"{price:.{strategy_instance.price_precision}f}" if price else "(Market)"
        sl_str = f"{sl_price:.{strategy_instance.price_precision}f}" if sl_price else "N/A"
        tp_str = f"{tp_price:.{strategy_instance.price_precision}f}" if tp_price else "N/A"
        log.warning(f"[PAPER MODE] Simulating {side} order placement: Qty={qty_str}, Symbol={symbol}, Price={price_str}, SL={sl_str}, TP={tp_str}")
        # Simulate a successful response for paper trading state management
        return {"retCode": 0, "retMsg": "OK", "result": {"orderId": f"paper_{int(time.time())}"}, "paperTrade": True}

    # Acquire lock to ensure only one order placement happens at a time
    with order_lock:
        order_type = config['order']['type']
        min_qty_decimal = market_info['lotSizeFilter']['minOrderQty']
        min_qty_float = float(min_qty_decimal)

        # Final quantity rounding and validation before placing order
        # Use the strategy's rounding function which should handle precision correctly
        qty_rounded = strategy_instance.round_qty(qty)
        if qty_rounded is None or qty_rounded <= 0:
             log.error(f"Attempted to place order with zero, negative, or invalid rounded quantity ({qty_rounded}). Original qty: {qty}.")
             return None
        if qty_rounded < min_qty_float:
            log.warning(f"Final quantity {qty_rounded:.{strategy_instance.qty_precision}f} is less than min qty {min_qty_float}. Adjusting to minimum.")
            qty_rounded = min_qty_float

        # Determine reference price for SL/TP validation (use provided price for limit, estimate for market)
        ref_entry_price: Optional[float] = None
        if order_type == "Limit" and price:
            ref_entry_price = price
        else: # Market order or missing limit price
            # Try getting current mark price first
            pos_data = get_current_position(symbol) # Rate limited internally
            if pos_data and pos_data.get('markPrice') > 0:
                 ref_entry_price = float(pos_data['markPrice'])
            else: # Fallback to last candle close if mark price unavailable
                 with data_lock:
                      if latest_dataframe is not None and not latest_dataframe.empty:
                           ref_entry_price = latest_dataframe['close'].iloc[-1]

        if ref_entry_price is None:
             log.error("Could not determine a reference entry price for SL/TP validation. Skipping SL/TP.")
             sl_price = None
             tp_price = None
        else:
             ref_entry_price = float(ref_entry_price) # Ensure float

        # Prepare order parameters dictionary
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side, # "Buy" or "Sell"
            "orderType": order_type,
            "qty": str(qty_rounded), # API requires quantity as a string
            "timeInForce": "GTC", # GoodTillCancel is common for entries with SL/TP
            "reduceOnly": False, # This is an entry order
            "positionIdx": 0 # Required for one-way position mode
        }

        # Add price for Limit orders
        if order_type == "Limit":
            if price and isinstance(price, (float, int)) and np.isfinite(price):
                # Round limit price according to tick size (use default rounding for limit orders)
                # Be slightly aggressive on limit entry: round UP for BUY, DOWN for SELL
                limit_rounding = ROUND_UP if side == "Buy" else ROUND_DOWN
                limit_price_rounded = strategy_instance.round_price(price, rounding_mode=limit_rounding)
                if limit_price_rounded is None:
                     log.error(f"Failed to round limit price {price}. Order cancelled.")
                     return None
                params["price"] = str(limit_price_rounded)
            else:
                log.error(f"Limit order requires a valid price. Got: {price}. Order cancelled.")
                return None

        # Add SL/TP using Bybit's parameters, with validation
        sl_price_str: Optional[str] = None
        if sl_price and isinstance(sl_price, (float, int)) and np.isfinite(sl_price):
            # Round SL price (away from entry: DOWN for Buy, UP for Sell)
            sl_rounding = ROUND_DOWN if side == "Buy" else ROUND_UP
            sl_price_rounded = strategy_instance.round_price(sl_price, rounding_mode=sl_rounding)

            if sl_price_rounded is None:
                 log.error(f"Failed to round SL price {sl_price}. SL skipped.")
            elif ref_entry_price is not None:
                # Validate SL relative to reference entry price
                is_invalid_sl = (side == "Buy" and sl_price_rounded >= ref_entry_price) or \
                                (side == "Sell" and sl_price_rounded <= ref_entry_price)
                if is_invalid_sl:
                    log.error(f"Invalid SL price {sl_price_rounded:.{strategy_instance.price_precision}f} for {side} order "
                              f"relative to reference entry {ref_entry_price:.{strategy_instance.price_precision}f}. SL skipped.")
                else:
                    sl_price_str = str(sl_price_rounded)
                    params["stopLoss"] = sl_price_str
            else: # Cannot validate if ref_entry_price is unknown
                 sl_price_str = str(sl_price_rounded)
                 params["stopLoss"] = sl_price_str
                 log.warning(f"Setting StopLoss at: {sl_price_str} (Could not validate against entry price).")

        tp_price_str: Optional[str] = None
        if tp_price and isinstance(tp_price, (float, int)) and np.isfinite(tp_price):
            # Round TP price (towards profit: UP for Buy, DOWN for Sell)
            tp_rounding = ROUND_UP if side == "Buy" else ROUND_DOWN
            tp_price_rounded = strategy_instance.round_price(tp_price, rounding_mode=tp_rounding)

            if tp_price_rounded is None:
                 log.error(f"Failed to round TP price {tp_price}. TP skipped.")
            elif ref_entry_price is not None:
                # Validate TP relative to reference entry price
                is_invalid_tp = (side == "Buy" and tp_price_rounded <= ref_entry_price) or \
                                (side == "Sell" and tp_price_rounded >= ref_entry_price)
                if is_invalid_tp:
                    log.error(f"Invalid TP price {tp_price_rounded:.{strategy_instance.price_precision}f} for {side} order "
                              f"relative to reference entry {ref_entry_price:.{strategy_instance.price_precision}f}. TP skipped.")
                else:
                    tp_price_str = str(tp_price_rounded)
                    params["takeProfit"] = tp_price_str
            else: # Cannot validate if ref_entry_price is unknown
                 tp_price_str = str(tp_price_rounded)
                 params["takeProfit"] = tp_price_str
                 log.warning(f"Setting TakeProfit at: {tp_price_str} (Could not validate against entry price).")

        log.warning(f"Placing {side} {order_type} order: Qty={params['qty']} {symbol} "
                    f"{'@'+str(params.get('price')) if 'price' in params else '(Market)'} "
                    f"SL={sl_price_str or 'N/A'} TP={tp_price_str or 'N/A'}")
        try:
            response = session.place_order(**params)
            log.debug(f"Place Order Response: {response}")

            if response.get('retCode') == 0:
                order_id = response.get('result', {}).get('orderId')
                log.info(f"{Fore.GREEN}Order placed successfully! OrderID: {order_id}{Style.RESET_ALL}")
                # Optional: Store order ID for tracking, wait for fill via WS?
                return response
            else:
                # Log specific Bybit error messages
                error_msg = response.get('retMsg', 'Unknown Error')
                error_code = response.get('retCode', 'N/A')
                log.error(f"{Fore.RED}Failed to place order: {error_msg} (Code: {error_code}){Style.RESET_ALL}")
                # Provide hints for common errors
                if error_code == 110007: log.error("Hint: Check available margin, leverage, position size limits, and potential open orders.")
                if error_code == 110045: log.error("Hint: Cannot place order with SL/TP if position size is zero or if it increases risk.")
                if "position mode not modified" in error_msg: log.error("Hint: Ensure Bybit account is set to One-Way position mode for linear perpetuals.")
                if "risk limit" in error_msg.lower(): log.error("Hint: Position size might exceed Bybit's risk limits for the current tier. Check Bybit settings.")
                if "order cost" in error_msg.lower(): log.error("Hint: Insufficient margin for order cost. Check balance and leverage.")
                return response # Return error response for potential handling upstream
        except Exception as e:
            log.error(f"Exception occurred during order placement: {e}", exc_info=(log_level <= logging.DEBUG))
            return None

def close_position(symbol: str, position_data: Dict[str, Any]) -> Optional[Dict]:
    """
    Closes an existing position using a reduce-only market order.
    Uses an order_lock to prevent concurrent closing attempts.
    """
    if not session or not strategy_instance:
        log.error("Cannot close position: Session or strategy instance missing.")
        return None
    if config.get("mode", "Live").lower() == "paper":
        log.warning(f"[PAPER MODE] Simulating closing position for {symbol} (Size: {position_data.get('size', 'N/A')}, Side: {position_data.get('side', 'N/A')})")
        # Simulate success
        return {"retCode": 0, "retMsg": "OK", "result": {"orderId": f"paper_close_{int(time.time())}"}, "paperTrade": True}

    # Acquire lock to ensure only one closing order happens at a time
    with order_lock:
        current_size = position_data.get('size', Decimal(0))
        current_side = position_data.get('side', 'None') # 'Buy' or 'Sell'

        if current_side == 'None' or current_size <= 0:
             log.warning(f"Attempting to close position for {symbol}, but position data indicates it's already flat or size is zero. No action taken.")
             return {"retCode": 0, "retMsg": "Position already flat or zero size", "result": {}, "alreadyFlat": True}

        # Determine the side needed to close the position
        side_to_close = "Sell" if current_side == "Buy" else "Buy"
        # Quantity to close is the current position size
        qty_to_close_decimal = current_size

        # Round quantity using strategy settings (usually not needed for market close, but good practice)
        # Convert Decimal to float for the rounding function
        qty_to_close_float = float(qty_to_close_decimal)
        qty_to_close_rounded = strategy_instance.round_qty(qty_to_close_float) # Round down just in case? Or use default?

        if qty_to_close_rounded is None or qty_to_close_rounded <= 0:
             log.error(f"Calculated quantity to close for {symbol} is zero, negative or invalid ({qty_to_close_rounded}). Cannot place closing order.")
             return None

        log.warning(f"Attempting to close {current_side} position for {symbol} (Size: {current_size}). Placing {side_to_close} Market order (Reduce-Only)...")
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side_to_close,
            "orderType": "Market", # Use Market order for immediate closure
            "qty": str(qty_to_close_rounded), # API requires string quantity
            "reduceOnly": True, # CRITICAL: Ensures this order only reduces/closes the position
            "positionIdx": 0 # Required for one-way mode
        }
        try:
            # Optional: Cancel existing SL/TP orders before closing market order
            # This might prevent conflicts or unexpected partial fills of SL/TP
            log.info(f"Attempting to cancel existing Stop Orders (SL/TP) for {symbol} before closing...")
            # Use orderFilter="StopOrder" to target only SL/TP
            response_cancel = session.cancel_all_orders(category="linear", symbol=symbol, orderFilter="StopOrder")
            log.debug(f"Cancel SL/TP Response: {response_cancel}")
            if response_cancel.get('retCode') != 0:
                 # Log warning but proceed with close attempt anyway
                 log.warning(f"Could not cancel stop orders before closing: {response_cancel.get('retMsg', 'Error')} (Code: {response_cancel.get('retCode')}). Proceeding with close.")
            else:
                 log.info("Successfully cancelled open stop orders.")
            time.sleep(0.5) # Brief pause after cancellation attempt

            # Place the closing market order
            response = session.place_order(**params)
            log.debug(f"Close Position Order Response: {response}")

            if response.get('retCode') == 0:
                order_id = response.get('result', {}).get('orderId')
                log.info(f"{Fore.YELLOW}Position close order placed successfully! OrderID: {order_id}{Style.RESET_ALL}")
                return response
            else:
                error_msg = response.get('retMsg', 'Unknown Error')
                error_code = response.get('retCode', 'N/A')
                log.error(f"{Fore.RED}Failed to place close order: {error_msg} (Code: {error_code}){Style.RESET_ALL}")
                # Handle common reduce-only errors (often mean position changed or closed already)
                # 110043: Reduce-only order qty exceeds open position size
                # 3400070: Position has been closed
                # 110025: Position size is zero
                if error_code in [110043, 3400070, 110025]:
                    log.warning("Reduce-only error likely means position size changed or closed between check and execution. Re-checking position soon.")
                    global last_position_check_time # Force re-check sooner
                    last_position_check_time = 0
                return response # Return error response
        except Exception as e:
            log.error(f"Exception occurred during position closing: {e}", exc_info=(log_level <= logging.DEBUG))
            return None

def set_leverage(symbol: str, leverage: int):
    """Sets the leverage for the specified symbol (requires one-way mode)."""
    if not session:
        log.error("HTTP session not available for set_leverage.")
        return
    # Validate leverage value (adjust range based on Bybit's limits if necessary)
    # Leverage is already validated during config load, but double-check here.
    if not 1 <= leverage <= 100: # Re-check just in case
         log.error(f"Invalid leverage value: {leverage}. Must be between 1 and 100 (check Bybit limits).")
         return

    log.info(f"Attempting to set leverage for {symbol} to {leverage}x...")
    try:
        # Bybit V5 requires setting buy and sell leverage equally for one-way mode
        leverage_str = str(float(leverage)) # API expects string representation of a float
        response = session.set_leverage(
            category="linear",
            symbol=symbol,
            buyLeverage=leverage_str,
            sellLeverage=leverage_str
        )
        log.debug(f"Set Leverage Response: {response}")
        if response.get('retCode') == 0:
            log.info(f"Leverage for {symbol} set to {leverage}x successfully.")
        else:
            error_code = response.get('retCode')
            error_msg = response.get('retMsg', 'Unknown Error')
            # Common error: 110044 means leverage not modified (it was already set to this value)
            if error_code == 110044:
                 log.warning(f"Leverage for {symbol} already set to {leverage}x (Code: 110044 - Not modified).")
            # Error 110045 might indicate trying to change leverage with open position/orders
            elif error_code == 110045:
                 log.error(f"Failed to set leverage for {symbol}: Cannot modify leverage with open positions or orders. (Code: 110045)")
            else:
                 log.error(f"Failed to set leverage for {symbol}: {error_msg} (Code: {error_code})")
    except Exception as e:
        log.error(f"Exception setting leverage: {e}", exc_info=(log_level <= logging.DEBUG))

# --- WebSocket Handling ---
def handle_ws_message(msg: Dict):
    """Callback function to process incoming WebSocket messages."""
    # log.debug(f"WS Recv: {msg}") # Very verbose, enable only for deep debugging
    global latest_dataframe
    if stop_event.is_set(): return # Don't process if shutdown is initiated

    topic = msg.get("topic", "")
    data = msg.get("data", [])

    # --- Handle Kline Updates ---
    # Topic format: kline.{interval}.{symbol}
    if topic.startswith("kline."):
        # Ensure it's the correct kline topic we subscribed to
        expected_kline_topic = f"kline.{config['interval']}.{config['symbol']}"
        if topic != expected_kline_topic:
            log.debug(f"Ignoring kline message from unexpected topic: {topic}")
            return

        if not data: return # Ignore empty data pushes
        # Bybit V5 Kline WS pushes one candle object per message in the 'data' list
        kline_item = data[0]
        # Process only confirmed (closed) candles
        if not kline_item.get('confirm', False):
            # log.debug("Ignoring unconfirmed kline update.")
            return

        try:
            # Extract data for the confirmed candle
            ts_ms = int(kline_item['start'])
            ts = pd.to_datetime(ts_ms, unit='ms')

            df_copy_for_analysis: Optional[pd.DataFrame] = None # Initialize here

            # --- Acquire Lock to Update DataFrame ---
            with data_lock:
                if latest_dataframe is None:
                    log.warning("DataFrame not initialized yet, skipping WS kline processing.")
                    return

                # Check if this candle timestamp already exists (can happen on reconnect)
                if ts in latest_dataframe.index:
                    # log.debug(f"Ignoring duplicate confirmed candle via WS: {ts}")
                    return

                log.debug(f"Confirmed Kline received via WS: T={ts}, O={kline_item['open']}, H={kline_item['high']}, L={kline_item['low']}, C={kline_item['close']}, V={kline_item['volume']}")

                new_data = {
                    'open': float(kline_item['open']),
                    'high': float(kline_item['high']),
                    'low': float(kline_item['low']),
                    'close': float(kline_item['close']),
                    'volume': float(kline_item['volume']),
                    'turnover': float(kline_item.get('turnover', 0.0)) # Turnover might not always be present
                }
                # Create a new DataFrame row with the timestamp as index
                new_row = pd.DataFrame([new_data], index=[ts])

                # Append the new row
                # Use pd.concat for potentially better performance than append
                latest_dataframe = pd.concat([latest_dataframe, new_row])

                # Prune the DataFrame to maintain max length
                max_len = config['data']['max_df_len']
                if len(latest_dataframe) > max_len:
                    latest_dataframe = latest_dataframe.iloc[-max_len:]
                    # log.debug(f"DataFrame pruned to {len(latest_dataframe)} rows.")

                # --- Prepare for Analysis ---
                if strategy_instance:
                    log.info(f"Running analysis on new confirmed candle: {ts}")
                    # Pass a copy to the strategy to prevent modification issues if analysis takes time
                    df_copy_for_analysis = latest_dataframe.copy()
                # else: df_copy_for_analysis remains None

            # --- Process Signals (outside data_lock) ---
            if df_copy_for_analysis is not None and strategy_instance:
                try:
                    analysis_results = strategy_instance.update(df_copy_for_analysis)
                    # Process the generated signals to potentially execute trades
                    process_signals(analysis_results)
                except Exception as e:
                    # Log the full traceback for strategy errors
                    log.error(f"Error during strategy analysis triggered by WS update: {e}", exc_info=True)

        except (KeyError, ValueError, TypeError) as e:
            log.error(f"Error parsing kline data from WS message: {e} - Data: {kline_item}")
        except Exception as e:
             log.error(f"Unexpected error handling kline WS message: {e}", exc_info=True)

    # --- Handle Position Updates ---
    # Topic format: position.{symbol} (Private topic)
    elif topic.startswith("position"):
         if data:
             for pos_update in data:
                 # Filter for the symbol we are trading
                 if pos_update.get('symbol') == config['symbol']:
                      log.info(f"{Fore.CYAN}Position update via WS: Size={pos_update.get('size')}, Side={pos_update.get('side')}, "
                               f"AvgPrice={pos_update.get('avgPrice')}, PnL={pos_update.get('unrealisedPnl')}{Style.RESET_ALL}")
                      # OPTIONAL: Directly update internal state based on WS.
                      # Requires careful locking (position_lock) and state management.
                      # Safer approach: Trigger a faster HTTP position check if needed.
                      # global last_position_check_time
                      # last_position_check_time = 0 # Force check on next main loop iteration
                      pass # Currently just logging the update

    # --- Handle Order Updates ---
    # Topic format: order (Private topic, catches all order updates for the account)
    elif topic.startswith("order"):
        if data:
             for order_update in data:
                 # Filter for the symbol we are trading
                 if order_update.get('symbol') == config['symbol']:
                     order_status = order_update.get('orderStatus')
                     order_id = order_update.get('orderId')
                     log.info(f"{Fore.CYAN}Order update via WS: ID={order_id}, Status={order_status}, "
                              f"Type={order_update.get('orderType')}, Side={order_update.get('side')}, "
                              f"Price={order_update.get('price')}, Qty={order_update.get('qty')}, "
                              f"AvgFillPrice={order_update.get('avgPrice')}, CumExecQty={order_update.get('cumExecQty')}"
                              f"{Style.RESET_ALL}")
                     # Can use this to track fills, cancellations, SL/TP triggers etc.
                     # Example: If orderStatus is 'Filled' and it matches an expected entry/exit order ID.
                     if order_status in ["Filled", "PartiallyFilled", "Cancelled", "Rejected", "Triggered"]:
                         # Potentially trigger position check or update internal state
                         pass
                     pass # Currently just logging the update

    # --- Handle Connection Status / Authentication ---
    elif msg.get("op") == "auth":
        if msg.get("success"):
            log.info(f"{Fore.GREEN}WebSocket authenticated successfully.{Style.RESET_ALL}")
        else:
            log.error(f"{Fore.RED}WebSocket authentication failed: {msg.get('ret_msg', 'No reason provided')}{Style.RESET_ALL}")
            # Consider stopping the bot or attempting reconnect if auth fails persistently
            ws_connected.clear() # Ensure connection status reflects failure
    elif msg.get("op") == "subscribe":
         if msg.get("success"):
            subscribed_topics = msg.get('ret_msg') or msg.get('args') # Location varies slightly
            log.info(f"{Fore.GREEN}WebSocket subscribed successfully to: {subscribed_topics}{Style.RESET_ALL}")
            ws_connected.set() # Signal that connection and subscription are likely successful
         else:
            log.error(f"{Fore.RED}WebSocket subscription failed: {msg.get('ret_msg', 'No reason provided')}{Style.RESET_ALL}")
            ws_connected.clear() # Signal potential connection issue
            # Consider stopping or retrying
    elif msg.get("op") == "pong":
        log.debug("WebSocket Pong received (heartbeat OK)")
    elif "success" in msg and not msg.get("success"):
        # Catch other potential operation failures
        log.error(f"{Fore.RED}WebSocket operation failed: {msg}{Style.RESET_ALL}")


def run_websocket_loop():
    """Target function for the WebSocket thread."""
    global ws, ws_connected
    log.info("WebSocket thread starting...")
    ping_interval = config['websocket']['ping_interval']

    while not stop_event.is_set():
        # Initialize WebSocket connection within the loop for reconnection logic
        log.info("Initializing WebSocket connection...")
        ws_connected.clear()
        ws = None # Ensure ws is reset before creating a new one
        try:
            ws = WebSocket(
                testnet=TESTNET,
                channel_type="private", # Use private for positions/orders
                api_key=API_KEY,
                api_secret=API_SECRET,
                message_handler=handle_ws_message # Pass handler directly
            )

            # Define required subscriptions
            kline_topic = f"kline.{config['interval']}.{config['symbol']}"
            position_topic = f"position.{config['symbol']}" # Specific symbol position updates
            order_topic = "order" # All order updates for the account
            topics_to_subscribe = [kline_topic, position_topic, order_topic]

            # Subscribe *before* starting the connection loop
            ws.subscribe(topics_to_subscribe)

            # Start the blocking run_forever loop
            log.info(f"WebSocket run_forever starting (ping interval: {ping_interval}s)...")
            ws.run_forever(ping_interval=ping_interval) # This blocks until exit() or error

        except Exception as e:
            log.error(f"WebSocket run_forever error or initialization failed: {e}", exc_info=(log_level <= logging.DEBUG))
            ws_connected.clear() # Ensure state is cleared on error
            if ws:
                try:
                    ws.exit() # Attempt to clean up the failed connection
                except Exception as exit_e:
                    log.error(f"Error trying to exit failed WebSocket: {exit_e}")
                ws = None

            if stop_event.is_set():
                log.info("WebSocket stopping due to stop_event.")
                break # Exit loop if stopping

            reconnect_delay = 15 # Seconds
            log.info(f"Attempting to reconnect WebSocket after error in {reconnect_delay} seconds...")
            stop_event.wait(timeout=reconnect_delay) # Wait or exit if stop signal received

        else: # If run_forever exits cleanly (e.g., ws.exit() called)
             log.info("WebSocket run_forever loop exited cleanly.")
             ws_connected.clear()
             if not stop_event.is_set():
                 log.warning("WebSocket exited unexpectedly without stop signal. Attempting restart...")
                 time.sleep(5) # Brief pause before restarting loop
             else:
                 break # Exit loop if stopping

    log.info("WebSocket thread finished.")
    ws_connected.clear()
    ws = None # Ensure ws is None after thread finishes


def start_websocket_connection() -> bool:
    """Initializes and starts the WebSocket connection in a separate thread."""
    global ws_thread
    if not API_KEY or not API_SECRET:
        log.error("Cannot start WebSocket: API credentials missing.")
        return False
    if ws_thread and ws_thread.is_alive():
        log.warning("WebSocket thread is already running.")
        return True # Already running

    log.info("Starting WebSocket thread...")
    stop_event.clear() # Ensure stop event is clear before starting
    ws_connected.clear() # Reset connection status event

    try:
        # The run_websocket_loop now handles initialization and subscription
        ws_thread = threading.Thread(target=run_websocket_loop, daemon=True, name="WebSocketThread")
        ws_thread.start()

        # Wait briefly for the connection and subscription confirmation (handled via ws_connected event)
        connect_timeout = config['websocket'].get('connect_timeout', 15) # Increased default timeout
        log.info(f"Waiting up to {connect_timeout}s for WebSocket connection and subscription confirmation...")
        if ws_connected.wait(timeout=connect_timeout):
            log.info(f"{Fore.GREEN}WebSocket connected and subscribed successfully.{Style.RESET_ALL}")
            return True
        else:
            log.error(f"{Fore.RED}WebSocket did not confirm subscription within {connect_timeout}s. Check logs, connection, credentials.{Style.RESET_ALL}")
            # Attempt cleanup even if connection failed
            stop_websocket_connection() # Signal the thread to stop and clean up
            return False

    except Exception as e:
        log.critical(f"CRITICAL: Failed to start WebSocket thread: {e}", exc_info=True)
        stop_websocket_connection() # Attempt cleanup
        return False

def stop_websocket_connection():
    """Stops the WebSocket connection and joins the thread gracefully."""
    global ws, ws_thread, ws_connected
    if not (ws_thread and ws_thread.is_alive()):
        log.info("WebSocket thread already stopped or not initialized.")
        if ws: # Clean up ws object if thread died but object remains
             try: ws.exit()
             except: pass
             ws = None
        ws_thread = None
        ws_connected.clear()
        return

    log.info("Stopping WebSocket connection...")
    stop_event.set() # Signal the loop in the thread to stop
    ws_connected.clear() # Signal connection is down

    if ws:
        try:
            ws.exit() # Signal run_forever to stop
            log.info("WebSocket exit() called.")
        except Exception as e:
            log.error(f"Error calling WebSocket exit(): {e}")
    else:
         log.info("WebSocket object not found (might be between reconnect attempts), relying on stop_event.")

    if ws_thread and ws_thread.is_alive():
        log.info("Waiting for WebSocket thread to join...")
        ws_thread.join(timeout=10) # Wait up to 10 seconds for the thread to finish
        if ws_thread.is_alive():
            log.warning("WebSocket thread did not stop gracefully after 10 seconds.")
        else:
            log.info("WebSocket thread joined successfully.")

    # Clean up global variables
    ws = None
    ws_thread = None
    ws_connected.clear()
    log.info("WebSocket connection stopped.")

# --- Signal Processing & Trade Execution ---

def calculate_sl_tp(side: str, entry_price: float, last_atr: Optional[float], results: AnalysisResults) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculates Stop Loss and Take Profit prices based on strategy config.
    Uses Decimal for precision in calculations. Returns floats.
    """
    if not strategy_instance or not market_info:
        log.error("Cannot calculate SL/TP: Strategy instance or market info missing.")
        return None, None
    if not isinstance(entry_price, (float, int)) or not np.isfinite(entry_price):
        log.error(f"Cannot calculate SL/TP: Invalid entry price {entry_price}.")
        return None, None

    sl_price_raw: Optional[float] = None
    tp_price_raw: Optional[float] = None
    sl_price_final: Optional[float] = None
    tp_price_final: Optional[float] = None

    try:
        entry_decimal = Decimal(str(entry_price))
        sl_method = strategy_instance.sl_method
        sl_atr_multiplier = Decimal(str(strategy_instance.sl_atr_multiplier))
        tp_ratio = Decimal(str(config['order'].get('tp_ratio', 2.0)))
        tick_size = strategy_instance.tick_size # Already Decimal
        last_atr_decimal = Decimal(str(last_atr)) if last_atr and np.isfinite(last_atr) else None

        # --- Calculate Stop Loss ---
        if sl_method == "ATR":
            if last_atr_decimal and last_atr_decimal > 0:
                sl_distance = last_atr_decimal * sl_atr_multiplier
                sl_price_decimal = entry_decimal - sl_distance if side == "Buy" else entry_decimal + sl_distance
                sl_price_raw = float(sl_price_decimal)
            else:
                log.error(f"Cannot calculate ATR Stop Loss: Invalid last_atr value ({last_atr}).")
                return None, None # Cannot proceed without valid SL

        elif sl_method == "OB":
            # Define buffers using Decimal
            sl_buffer_atr_fraction = Decimal("0.1") # Buffer as fraction of ATR
            sl_buffer_price_fraction = Decimal("0.0005") # Buffer as fraction of price (fallback)
            buffer: Decimal
            if last_atr_decimal and last_atr_decimal > 0:
                 buffer = last_atr_decimal * sl_buffer_atr_fraction
            else:
                 buffer = entry_decimal * sl_buffer_price_fraction
                 log.warning("ATR unavailable for OB SL buffer, using price fraction fallback.")
            buffer = max(buffer, tick_size) # Ensure buffer is at least one tick

            sl_target_price: Optional[Decimal] = None
            if side == "Buy":
                # Find lowest bottom of active bull OBs below entry
                relevant_obs = [b for b in results.get('active_bull_boxes', []) if Decimal(str(b['bottom'])) < entry_decimal]
                if relevant_obs:
                    lowest_bottom = min(Decimal(str(b['bottom'])) for b in relevant_obs)
                    sl_target_price = lowest_bottom - buffer
                else:
                    log.warning("OB SL method chosen for BUY, but no active Bull OB found below entry. Falling back to ATR.")
                    if last_atr_decimal and last_atr_decimal > 0:
                        sl_distance = last_atr_decimal * sl_atr_multiplier
                        sl_target_price = entry_decimal - sl_distance
                    else: log.error("Cannot set SL: Fallback ATR is unavailable."); return None, None
            else: # side == "Sell"
                # Find highest top of active bear OBs above entry
                relevant_obs = [b for b in results.get('active_bear_boxes', []) if Decimal(str(b['top'])) > entry_decimal]
                if relevant_obs:
                    highest_top = max(Decimal(str(b['top'])) for b in relevant_obs)
                    sl_target_price = highest_top + buffer
                else:
                    log.warning("OB SL method chosen for SELL, but no active Bear OB found above entry. Falling back to ATR.")
                    if last_atr_decimal and last_atr_decimal > 0:
                        sl_distance = last_atr_decimal * sl_atr_multiplier
                        sl_target_price = entry_decimal + sl_distance
                    else: log.error("Cannot set SL: Fallback ATR is unavailable."); return None, None

            if sl_target_price is not None:
                 sl_price_raw = float(sl_target_price)

        if sl_price_raw is None:
            log.error("Stop Loss price could not be calculated. Cannot determine trade parameters.")
            return None, None

        # --- Validate and Round SL ---
        # Ensure SL is on the correct side of the entry price (more than a tick away)
        sl_price_decimal_raw = Decimal(str(sl_price_raw))
        is_invalid_raw_sl = (side == "Buy" and sl_price_decimal_raw >= entry_decimal - tick_size) or \
                            (side == "Sell" and sl_price_decimal_raw <= entry_decimal + tick_size)

        if is_invalid_raw_sl:
            log.error(f"Calculated raw SL price {sl_price_raw} is not logical or too close for a {side} trade from entry {entry_price}. Attempting fallback ATR SL.")
            if last_atr_decimal and last_atr_decimal > 0:
                sl_distance = last_atr_decimal * sl_atr_multiplier
                sl_price_decimal_raw = entry_decimal - sl_distance if side == "Buy" else entry_decimal + sl_distance
                sl_price_raw = float(sl_price_decimal_raw)
                # Re-check after fallback
                is_invalid_raw_sl = (side == "Buy" and sl_price_decimal_raw >= entry_decimal - tick_size) or \
                                    (side == "Sell" and sl_price_decimal_raw <= entry_decimal + tick_size)
                if is_invalid_raw_sl:
                     log.error("Fallback ATR SL is also invalid or too close. Cannot determine SL.")
                     return None, None
            else:
                log.error("Cannot set SL: Fallback ATR is unavailable.")
                return None, None

        # Round SL price (away from entry: DOWN for Buy, UP for Sell)
        sl_rounding = ROUND_DOWN if side == "Buy" else ROUND_UP
        sl_price_final = strategy_instance.round_price(sl_price_raw, rounding_mode=sl_rounding)
        if sl_price_final is None or not np.isfinite(sl_price_final):
             log.error(f"Failed to round SL price {sl_price_raw}.")
             return None, None
        log.info(f"Calculated SL price for {side}: {sl_price_final}")

        # --- Calculate Take Profit ---
        sl_price_final_decimal = Decimal(str(sl_price_final))
        sl_distance_decimal = abs(entry_decimal - sl_price_final_decimal)

        if sl_distance_decimal > 0 and tp_ratio > 0:
            tp_distance = sl_distance_decimal * tp_ratio
            tp_price_decimal = entry_decimal + tp_distance if side == "Buy" else entry_decimal - tp_distance
            tp_price_raw = float(tp_price_decimal)

            # Round TP price (towards profit: UP for Buy, DOWN for Sell)
            tp_rounding = ROUND_UP if side == "Buy" else ROUND_DOWN
            tp_price_final = strategy_instance.round_price(tp_price_raw, rounding_mode=tp_rounding)

            if tp_price_final is None or not np.isfinite(tp_price_final):
                 log.warning(f"Failed to round TP price {tp_price_raw}. TP will not be set.")
                 tp_price_final = None
            else:
                 # Final validation for TP vs Entry
                 is_invalid_tp = (side == "Buy" and tp_price_final <= entry_price) or \
                                 (side == "Sell" and tp_price_final >= entry_price)
                 if is_invalid_tp:
                      log.warning(f"Calculated TP price {tp_price_final} is not logical for {side} from entry {entry_price}. TP will not be set.")
                      tp_price_final = None
                 else:
                      log.info(f"Calculated TP price for {side}: {tp_price_final} (Ratio: {tp_ratio})")
        else:
            log.warning("Cannot calculate TP: SL distance is zero or TP ratio is not positive.")
            tp_price_final = None

    except (InvalidOperation, TypeError) as e:
         log.error(f"Decimal or calculation error during SL/TP calculation: {e}")
         return None, None
    except Exception as e:
         log.error(f"Unexpected error during SL/TP calculation: {e}", exc_info=(log_level <= logging.DEBUG))
         return None, None

    return sl_price_final, tp_price_final


def process_signals(results: AnalysisResults):
    """
    Processes the strategy signals, calculates order parameters,
    checks current position state, and executes trades or closes positions.
    """
    if not results or not strategy_instance or not market_info:
        log.warning("Signal processing skipped: Missing results, strategy instance, or market info.")
        return
    if stop_event.is_set():
        log.warning("Signal processing skipped: Stop event is set.")
        return

    signal = results.get('last_signal')
    last_close = results.get('last_close')
    last_atr = results.get('last_atr')
    symbol = config['symbol']

    log.debug(f"Processing Signal: {signal}, Last Close: {last_close}, Last ATR: {last_atr}")

    if signal is None:
        log.debug("No signal generated. No action.")
        return
    if pd.isna(last_close):
        log.warning("Cannot process signal: Last close price is NaN.")
        return

    # --- Get Current Position State ---
    # Crucial for deciding whether to enter or exit. Rate limiting is handled internally.
    position_data = get_current_position(symbol)
    if position_data is None:
        log.warning("Position check skipped due to rate limit. Will re-evaluate on next candle.")
        return # Wait for next cycle if check was skipped

    # Use the fetched position data
    current_pos_size = position_data.get('size', Decimal(0))
    current_pos_side = position_data.get('side', 'None') # 'Buy', 'Sell', or 'None'

    # Check size against minimum order quantity for robustness
    min_qty_decimal = market_info['lotSizeFilter']['minOrderQty']
    is_long = current_pos_side == 'Buy' and current_pos_size >= min_qty_decimal
    is_short = current_pos_side == 'Sell' and current_pos_size >= min_qty_decimal
    is_flat = not is_long and not is_short

    log.info(f"Current Position: {'Long' if is_long else 'Short' if is_short else 'Flat'} (Size: {current_pos_size})")

    # --- Execute Actions Based on Signal and Position State ---

    # BUY Signal: Enter Long if Flat
    if signal == "BUY" and is_flat:
        log.warning(f"{Fore.GREEN}{Style.BRIGHT}BUY Signal Received - Attempting to Enter Long.{Style.RESET_ALL}")
        sl_price, tp_price = calculate_sl_tp("Buy", last_close, last_atr, results)
        if sl_price is None:
            log.error("Failed to calculate valid SL for BUY signal. Order cancelled.")
            return

        qty = calculate_order_qty(last_close, sl_price, config['order']['risk_per_trade_percent'])
        if qty and qty > 0:
            # Use last_close as the reference price for Limit orders if applicable
            limit_price = last_close if config['order']['type'] == "Limit" else None
            place_order(symbol, "Buy", qty, price=limit_price, sl_price=sl_price, tp_price=tp_price)
        else:
            log.error("Order quantity calculation failed or resulted in zero/negative. Cannot place BUY order.")

    # SELL Signal: Enter Short if Flat
    elif signal == "SELL" and is_flat:
        log.warning(f"{Fore.RED}{Style.BRIGHT}SELL Signal Received - Attempting to Enter Short.{Style.RESET_ALL}")
        sl_price, tp_price = calculate_sl_tp("Sell", last_close, last_atr, results)
        if sl_price is None:
            log.error("Failed to calculate valid SL for SELL signal. Order cancelled.")
            return

        qty = calculate_order_qty(last_close, sl_price, config['order']['risk_per_trade_percent'])
        if qty and qty > 0:
            limit_price = last_close if config['order']['type'] == "Limit" else None
            place_order(symbol, "Sell", qty, price=limit_price, sl_price=sl_price, tp_price=tp_price)
        else:
            log.error("Order quantity calculation failed or resulted in zero/negative. Cannot place SELL order.")

    # EXIT_LONG Signal: Close Long Position if Currently Long
    elif signal == "EXIT_LONG" and is_long:
        log.warning(f"{Fore.YELLOW}{Style.BRIGHT}EXIT_LONG Signal Received - Attempting to Close Long Position.{Style.RESET_ALL}")
        close_position(symbol, position_data) # Pass the fetched position data

    # EXIT_SHORT Signal: Close Short Position if Currently Short
    elif signal == "EXIT_SHORT" and is_short:
        log.warning(f"{Fore.YELLOW}{Style.BRIGHT}EXIT_SHORT Signal Received - Attempting to Close Short Position.{Style.RESET_ALL}")
        close_position(symbol, position_data)

    # HOLD Signal or Signal Matches Current State -> No Action Needed
    elif signal == "HOLD":
        log.debug("HOLD Signal - No trade action.")
    elif signal == "BUY" and is_long:
        log.debug("BUY Signal received, but already Long. No action.")
    elif signal == "SELL" and is_short:
        log.debug("SELL Signal received, but already Short. No action.")
    elif signal == "EXIT_LONG" and not is_long:
         log.debug("EXIT_LONG Signal received, but not Long. No action.")
    elif signal == "EXIT_SHORT" and not is_short:
         log.debug("EXIT_SHORT Signal received, but not Short. No action.")
    # Log cases where entry signal received while already in opposite position (should ideally be handled by EXIT first)
    elif signal == "BUY" and is_short:
         log.warning("BUY Signal received while Short. Strategy should have generated EXIT_SHORT first. No action.")
    elif signal == "SELL" and is_long:
         log.warning("SELL Signal received while Long. Strategy should have generated EXIT_LONG first. No action.")


# --- Graceful Shutdown ---
def handle_shutdown_signal(signum, frame):
    """Handles termination signals (SIGINT, SIGTERM) for graceful shutdown."""
    if stop_event.is_set(): # Prevent running multiple times if signal received repeatedly
         log.warning("Shutdown already in progress.")
         return
    log.warning(f"Shutdown signal {signal.Signals(signum).name} ({signum}) received. Initiating graceful shutdown...")
    stop_event.set() # Signal all loops and threads to stop

    # 1. Stop WebSocket first to prevent processing new data/signals
    stop_websocket_connection()

    # 2. Optional: Implement logic to manage open positions/orders on shutdown
    #    USE WITH EXTREME CAUTION - unexpected closure can lead to losses.
    #    Make this configurable (e.g., in config.json)
    close_on_exit = config.get("close_position_on_exit", False)
    if close_on_exit and config.get("mode", "Live").lower() != "paper":
        log.warning("Attempting to close open position on exit (close_position_on_exit=True)...")
        # Need to ensure we get the latest position data, might need retry if rate limited
        pos_data = None
        max_retries = 3
        retry_delay = max(1, POSITION_CHECK_INTERVAL / 2) # Wait at least 1 second
        for attempt in range(max_retries):
             # Force position check by resetting timer (or just call directly if interval allows)
             global last_position_check_time
             last_position_check_time = 0 # Force check on next call
             pos_data = get_current_position(config['symbol'])
             if pos_data is not None: # Got data (even if flat), break retry
                  break
             log.warning(f"Position check rate limited during shutdown. Retrying in {retry_delay}s... (Attempt {attempt+1}/{max_retries})")
             time.sleep(retry_delay)

        if pos_data and pos_data.get('size', Decimal(0)) > 0 and pos_data.get('side') != 'None':
             log.warning(f"Found open {pos_data.get('side')} position (Size: {pos_data.get('size')}). Attempting market close.")
             close_response = close_position(config['symbol'], pos_data)
             if close_response and close_response.get('retCode') == 0:
                  log.info("Position close order placed successfully during shutdown.")
                  # Optionally wait briefly to see if WS confirms closure?
                  time.sleep(2)
             elif close_response and close_response.get('alreadyFlat'):
                  log.info("Position was already closed before shutdown close attempt.")
             else:
                  log.error(f"Failed to place position close order during shutdown. Response: {close_response}")
        elif pos_data:
             log.info("No open position found to close on exit.")
        else:
             log.error("Could not determine position status during shutdown due to repeated errors/rate limit.")

    # 3. Final log message and exit
    log.info("Shutdown sequence complete. Exiting.")
    # Give logs a moment to flush before exiting
    logging.shutdown()
    time.sleep(0.5)
    sys.exit(0)

# --- Main Execution Block ---
if __name__ == "__main__":
    print(Fore.MAGENTA + Style.BRIGHT + "\n~~~ Pyrmethus Volumatic+OB Trading Bot Initializing ~~~" + Style.RESET_ALL)

    # Load configuration first, as it dictates logging level etc.
    config = load_config() # Exits on critical error

    # Setup logging based on config
    setup_logging(config.get("log_level", "INFO"))
    log.info(f"Logging level set to: {logging.getLevelName(log.level)}")
    log.debug(f"Full Config: {json.dumps(config, indent=2, default=str)}") # Use default=str for Decimal

    # Register signal handlers for graceful shutdown (Ctrl+C, kill)
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    log.info("Registered shutdown signal handlers.")

    # Connect to Bybit HTTP API
    session = connect_bybit()
    if not session:
        # connect_bybit already logs critical error
        sys.exit(1)

    # Get Market Info (Symbol, Precision, Limits)
    market_info = get_market_info(config['symbol'])
    if not market_info:
        log.critical(f"Could not retrieve valid market info for symbol '{config['symbol']}'. Check symbol and API connection. Exiting.")
        sys.exit(1)
    log.info(f"Market Info: TickSize={market_info['priceFilter']['tickSize']}, QtyStep={market_info['lotSizeFilter']['qtyStep']}")

    # Set Leverage (Important!) - Do this before initializing strategy if strategy uses it
    # Ensure leverage is set before potentially placing orders
    set_leverage(config['symbol'], config['order']['leverage'])

    # Initialize Strategy Engine
    try:
        # Pass market info and strategy-specific parameters
        strategy_instance = VolumaticOBStrategy(market_info=market_info, **config['strategy'])
        log.info(f"Strategy '{type(strategy_instance).__name__}' initialized successfully.")
        log.info(f"Strategy requires minimum {strategy_instance.min_data_len} candles.")
    except (ValueError, KeyError, TypeError) as e:
        log.critical(f"Failed to initialize strategy: {e}. Check config.json strategy params. Exiting.", exc_info=True)
        sys.exit(1)
    except Exception as e:
         log.critical(f"Unexpected error initializing strategy: {e}", exc_info=True)
         sys.exit(1)

    # Fetch Initial Historical Data
    with data_lock: # Protect dataframe initialization
        latest_dataframe = fetch_initial_data(
            config['symbol'],
            config['interval'],
            config['data']['fetch_limit']
        )

    if latest_dataframe is None: # Check for None specifically (indicates fetch error)
        log.critical("Failed to fetch initial historical data. Cannot proceed. Exiting.")
        sys.exit(1)
    if latest_dataframe.empty:
        log.warning("Fetched initial data, but the DataFrame is empty. "
                    "Check symbol/interval on Bybit. Bot will attempt to run using WebSocket data, "
                    "but strategy may need more history.")
        # Allow continuing, WS might populate data, but strategy needs min_data_len
    elif len(latest_dataframe) < strategy_instance.min_data_len:
         log.warning(f"Initial data fetched ({len(latest_dataframe)} candles) is less than minimum required by strategy "
                     f"({strategy_instance.min_data_len}). Strategy calculations may be inaccurate until more data arrives via WebSocket.")
         # Allow continuing

    # Run Initial Analysis on historical data (if enough data exists)
    # This pre-fills indicators and establishes the initial trend/state
    initial_analysis_done = False
    if latest_dataframe is not None and not latest_dataframe.empty and len(latest_dataframe) >= strategy_instance.min_data_len:
        log.info("Running initial analysis on historical data...")
        df_copy_initial: Optional[pd.DataFrame] = None
        with data_lock: # Access dataframe safely
            # Pass a copy for analysis
            df_copy_initial = latest_dataframe.copy()

        if df_copy_initial is not None:
            try:
                initial_results = strategy_instance.update(df_copy_initial)
                if initial_results:
                    trend_str = 'UP' if initial_results.get('current_trend') else 'DOWN' if initial_results.get('current_trend') is False else 'Undetermined'
                    log.info(f"Initial Analysis Complete: Current Trend Estimate = {trend_str}, "
                             f"Last Signal State = {strategy_instance.last_signal_state}")
                    initial_analysis_done = True
                else:
                     log.error("Initial analysis returned no results.")
            except Exception as e:
                 log.error(f"Error during initial strategy analysis: {e}", exc_info=True)
        else:
             log.error("Failed to copy dataframe for initial analysis.") # Should not happen if checks passed
    else:
         log.info("Skipping initial analysis due to insufficient historical data.")

    # Start WebSocket Connection
    if not start_websocket_connection():
        log.critical("Failed to start WebSocket connection. Exiting.")
        # Ensure cleanup if WS fails to start
        stop_websocket_connection()
        sys.exit(1)

    # --- Bot Running Loop ---
    log.info(f"{Fore.CYAN}{Style.BRIGHT}Bot is now running for {config['symbol']} ({config['interval']}). Mode: {config['mode']}. Waiting for signals...{Style.RESET_ALL}")
    log.info("Press Ctrl+C to stop gracefully.")

    # Main loop primarily keeps the script alive and performs periodic health checks.
    # Most logic is driven by WebSocket updates handled in handle_ws_message.
    while not stop_event.is_set():
        try:
            # 1. Check WebSocket Health (is the thread alive?)
            if ws_thread and not ws_thread.is_alive():
                log.error("WebSocket thread appears to have died unexpectedly!")
                if not stop_event.is_set():
                    log.info("Attempting to restart WebSocket connection...")
                    # Attempt full stop/start cycle
                    stop_websocket_connection() # Ensure old resources are cleaned
                    time.sleep(5) # Wait before restarting
                    if not start_websocket_connection():
                        log.critical("Failed to restart WebSocket after failure. Stopping bot.")
                        handle_shutdown_signal(signal.SIGTERM, None) # Trigger shutdown
                        break # Exit main loop
                    else:
                         log.info("WebSocket connection restarted successfully.")
                else:
                    log.info("WebSocket thread died, but shutdown already in progress.")
                    break # Exit loop if shutdown initiated

            # 2. Periodic Position Check (Fallback/Verification)
            # Although WS provides position updates, a periodic check ensures sync.
            now = time.time()
            # Check slightly more often than the rate limit interval to ensure it runs
            if now - last_position_check_time > POSITION_CHECK_INTERVAL:
                log.debug("Performing periodic position check (verification)...")
                # Fetch position data, result is handled internally by get_current_position (updates time, logs)
                get_current_position(config['symbol']) # Result not needed here, just triggers the check

            # 3. Sleep efficiently until the next check or stop signal
            # Wait for a short duration or until the stop_event is set
            # Timeout determines frequency of health checks etc.
            check_interval = 10.0 # Seconds
            stop_event.wait(timeout=check_interval)

        except KeyboardInterrupt: # Allow Ctrl+C to break the loop and trigger shutdown
             log.warning("KeyboardInterrupt detected in main loop.")
             if not stop_event.is_set():
                  handle_shutdown_signal(signal.SIGINT, None)
             break # Exit loop
        except Exception as e:
             # Catch unexpected errors in the main loop
             log.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
             log.warning("Attempting to continue after error...")
             # Wait a bit longer after an unexpected error before resuming checks
             time.sleep(15)

    # --- End of Script ---
    # Shutdown sequence is handled by handle_shutdown_signal
    log.info("Main loop terminated.")
    # Final confirmation message (might not be reached if sys.exit called in handler)
    print(Fore.MAGENTA + Style.BRIGHT + "~~~ Pyrmethus Volumatic+OB Trading Bot Stopped ~~~" + Style.RESET_ALL)

