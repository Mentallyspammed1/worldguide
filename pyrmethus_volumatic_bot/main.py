# -*- coding: utf-8 -*-
# Suggest adding a requirements.txt file with dependencies:
# pybit>=5.4.0  # Check for the latest compatible version
# pandas
# numpy
# python-dotenv
# colorama

import os
import sys
import json
import time
import datetime
import logging
import signal
import threading
from decimal import Decimal, ROUND_DOWN, ROUND_UP, InvalidOperation, getcontext, DivisionByZero
from typing import Dict, Optional, Any, Tuple, List, Union # Added Union

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from colorama import init, Fore, Style
from pybit.unified_trading import HTTP, WebSocket

# Import strategy class and type hints from strategy.py
try:
    from strategy import VolumaticOBStrategy, AnalysisResults
except ImportError as e:
    # Use print for early errors before logging might be configured
    print(f"ERROR: Could not import from strategy.py: {e}", file=sys.stderr)
    print("Ensure strategy.py is in the same directory or accessible via PYTHONPATH.", file=sys.stderr)
    sys.exit(1)

# --- Initialize Colorama ---
# autoreset=True ensures color resets after each print
init(autoreset=True)

# --- Set Decimal Precision ---
# Set precision high enough for crypto prices and quantities
# 18 places should be sufficient for most pairs on Bybit.
getcontext().prec = 18

# --- Load Environment Variables ---
# Load API keys and settings from .env file in the same directory
if load_dotenv():
    print("Loaded environment variables from .env file.")
else:
    print("WARN: .env file not found or empty. Ensure API keys and settings are set as environment variables.")

API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
TESTNET = os.getenv("BYBIT_TESTNET", "False").lower() in ("true", "1", "t")

# --- Global Variables & State ---
config: Dict[str, Any] = {}
session: Optional[HTTP] = None
ws: Optional[WebSocket] = None
ws_thread: Optional[threading.Thread] = None
ws_connected = threading.Event() # Signals WebSocket connection and subscription status
stop_event = threading.Event() # Used for graceful shutdown coordination
latest_dataframe: Optional[pd.DataFrame] = None
strategy_instance: Optional[VolumaticOBStrategy] = None
market_info: Optional[Dict[str, Any]] = None
data_lock = threading.Lock() # Protects access to latest_dataframe
order_lock = threading.Lock() # Protects order placement/closing logic to prevent race conditions
last_position_check_time: float = 0.0 # Initialize to 0.0
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
        # Use print here as logging might not be fully set up
        print(f"WARN: Invalid log level '{level_str}'. Defaulting to INFO.", file=sys.stderr)

    log.setLevel(log_level)
    log.handlers.clear() # Clear existing handlers to prevent duplicates

    # Console Handler (StreamHandler)
    ch = logging.StreamHandler(sys.stdout) # Use stdout for console logs
    ch.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    log.addHandler(ch)

    # Optional: File Handler (Uncomment and configure if needed)
    # log_filename = f"bot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    # try:
    #     fh = logging.FileHandler(log_filename, encoding='utf-8') # Specify encoding
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

        # --- Comprehensive Validation ---
        required_keys = ["symbol", "interval", "mode", "log_level", "order", "strategy", "data", "websocket"]
        missing = [key for key in required_keys if key not in conf]
        if missing:
            raise ValueError(f"Config file missing required top-level keys: {missing}")

        # Order Section
        order_conf = conf["order"]
        req_order_keys = ["risk_per_trade_percent", "leverage", "type", "tp_ratio"]
        missing_order = [k for k in req_order_keys if k not in order_conf]
        if missing_order:
            raise ValueError(f"Config 'order' section missing required keys: {missing_order}")
        if order_conf["type"] not in ["Market", "Limit"]:
            raise ValueError(f"Invalid order type '{order_conf['type']}'. Must be 'Market' or 'Limit'.")
        if not isinstance(order_conf["risk_per_trade_percent"], (int, float)) or not 0 < order_conf["risk_per_trade_percent"] <= 100:
             raise ValueError("risk_per_trade_percent must be a number > 0 and <= 100.")
        if not isinstance(order_conf["leverage"], int) or not 1 <= order_conf["leverage"] <= 100: # Adjust max based on Bybit limits/tier
             raise ValueError("leverage must be an integer between 1 and 100 (check Bybit limits).")
        if not isinstance(order_conf["tp_ratio"], (int, float)) or order_conf["tp_ratio"] <= 0:
             raise ValueError("tp_ratio must be a positive number.")

        # Strategy Section
        strategy_conf = conf["strategy"]
        req_strategy_keys = ["params", "stop_loss"]
        missing_strategy = [k for k in req_strategy_keys if k not in strategy_conf]
        if missing_strategy:
            raise ValueError(f"Config 'strategy' section missing required keys: {missing_strategy}")
        if not isinstance(strategy_conf["params"], dict):
             raise ValueError("strategy 'params' must be a dictionary.")
        if not isinstance(strategy_conf["stop_loss"], dict):
             raise ValueError("strategy 'stop_loss' must be a dictionary.")
        req_sl_keys = ["method", "atr_multiplier"]
        missing_sl = [k for k in req_sl_keys if k not in strategy_conf["stop_loss"]]
        if missing_sl:
             raise ValueError(f"Config 'strategy.stop_loss' section missing required keys: {missing_sl}")
        if strategy_conf["stop_loss"]["method"] not in ["ATR", "OB"]:
             raise ValueError("strategy stop_loss method must be 'ATR' or 'OB'.")
        if not isinstance(strategy_conf["stop_loss"]["atr_multiplier"], (int, float)) or strategy_conf["stop_loss"]["atr_multiplier"] <= 0:
             raise ValueError("strategy stop_loss atr_multiplier must be a positive number.")

        # Data Section
        data_conf = conf["data"]
        req_data_keys = ["fetch_limit", "max_df_len"]
        missing_data = [k for k in req_data_keys if k not in data_conf]
        if missing_data:
            raise ValueError(f"Config 'data' section missing required keys: {missing_data}")
        if not isinstance(data_conf["fetch_limit"], int) or data_conf["fetch_limit"] <= 0:
             raise ValueError("data fetch_limit must be a positive integer.")
        if not isinstance(data_conf["max_df_len"], int) or data_conf["max_df_len"] <= 50: # Ensure reasonable minimum length
             raise ValueError("data max_df_len must be a positive integer (recommended > 50).")

        # WebSocket Section
        ws_conf = conf["websocket"]
        req_ws_keys = ["ping_interval", "connect_timeout"]
        missing_ws = [k for k in req_ws_keys if k not in ws_conf]
        if missing_ws:
            raise ValueError(f"Config 'websocket' section missing required keys: {missing_ws}")
        if not isinstance(ws_conf["ping_interval"], int) or ws_conf["ping_interval"] <= 0:
             raise ValueError("websocket ping_interval must be a positive integer (seconds).")
        if not isinstance(ws_conf["connect_timeout"], int) or ws_conf["connect_timeout"] <= 0:
             raise ValueError("websocket connect_timeout must be a positive integer (seconds).")

        # Optional Settings
        POSITION_CHECK_INTERVAL = int(conf.get("position_check_interval", 10))
        if POSITION_CHECK_INTERVAL <= 0:
             log.warning("position_check_interval must be positive. Using default 10s.")
             POSITION_CHECK_INTERVAL = 10

        conf["close_position_on_exit"] = conf.get("close_position_on_exit", False)
        if not isinstance(conf["close_position_on_exit"], bool):
             log.warning("close_position_on_exit should be true or false. Defaulting to false.")
             conf["close_position_on_exit"] = False

        return conf
    except FileNotFoundError:
        log.critical(f"CRITICAL: Configuration file '{path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        log.critical(f"CRITICAL: Configuration file '{path}' contains invalid JSON: {e}")
        sys.exit(1)
    except (ValueError, KeyError, TypeError) as e: # Catch validation errors
        log.critical(f"CRITICAL: Configuration error in '{path}': {e}")
        sys.exit(1)
    except Exception as e:
        log.critical(f"CRITICAL: Unexpected error loading configuration from '{path}': {e}", exc_info=True)
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
            # Handle potential non-numeric value gracefully
            server_ts = int(server_ts_nano) / 1e9 if server_ts_nano.isdigit() else time.time()
            server_dt = datetime.datetime.fromtimestamp(server_ts)
            log.info(f"Successfully connected. Server time: {server_dt.strftime('%Y-%m-%d %H:%M:%S.%f')}")
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
            info_list = response['result']['list']
            if not info_list:
                 log.error(f"Market info list is empty for {symbol}. Is the symbol correct for linear perpetuals?")
                 return None
            info = info_list[0] # Assume first item is the correct one for the symbol
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

            missing_p_keys = [k for k in required_price_keys if k not in price_filter]
            missing_l_keys = [k for k in required_lot_keys if k not in lot_filter]

            if missing_p_keys or missing_l_keys:
                missing_keys = missing_p_keys + missing_l_keys
                log.error(f"Market info for {symbol} is missing required filter details: {missing_keys}. Cannot proceed.")
                return None

            # Convert to Decimal early for consistency and precision
            try:
                 info['priceFilter']['tickSize'] = Decimal(price_filter['tickSize'])
                 info['lotSizeFilter']['qtyStep'] = Decimal(lot_filter['qtyStep'])
                 info['lotSizeFilter']['minOrderQty'] = Decimal(lot_filter['minOrderQty'])
                 info['lotSizeFilter']['maxOrderQty'] = Decimal(lot_filter['maxOrderQty'])
            except (InvalidOperation, TypeError, KeyError, ValueError) as e: # Added ValueError
                 log.error(f"Could not convert market info filter values to Decimal for {symbol}: {e}")
                 return None

            # Check if values are positive as expected
            if info['priceFilter']['tickSize'] <= 0 or info['lotSizeFilter']['qtyStep'] <= 0 or \
               info['lotSizeFilter']['minOrderQty'] <= 0 or info['lotSizeFilter']['maxOrderQty'] <= 0:
                log.error(f"Market info for {symbol} contains non-positive filter values. Cannot proceed.")
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
    """
    Fetches historical Klines data and prepares the initial DataFrame.
    Note: Bybit V5 API has a limit of 1000 klines per request.
    This function currently only makes one request. If limit > 1000, only 1000 are fetched.
    Implement looping if more than 1000 initial candles are strictly required.
    """
    if not session:
         log.error("HTTP session not available for fetch_initial_data.")
         return None

    # Adjust limit based on API constraints
    actual_limit = min(limit, 1000)
    if limit > 1000:
        log.warning(f"Requested {limit} initial candles, but Bybit API limit is 1000 per request. Fetching {actual_limit}.")
        # Consider implementing pagination here if needed

    log.info(f"Fetching initial {actual_limit} klines for {symbol} (interval: {interval})...")
    try:
        response = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            limit=actual_limit
        )
        if response.get('retCode') == 0 and 'result' in response and 'list' in response['result']:
            kline_list = response['result']['list']
            if not kline_list:
                 log.warning(f"Received empty kline list from Bybit for initial fetch of {symbol}/{interval}.")
                 # Return an empty DataFrame with expected columns and index type
                 return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'turnover']).astype({'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64', 'turnover': 'float64'}).set_index(pd.to_datetime([]))

            df = pd.DataFrame(kline_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            # Convert types immediately after creation for safety
            df = df.astype({
                'timestamp': 'int64', 'open': 'float64', 'high': 'float64',
                'low': 'float64', 'close': 'float64', 'volume': 'float64', 'turnover': 'float64'
            })
            # Convert timestamp (milliseconds) to DatetimeIndex (UTC by default)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            # Bybit V5 returns oldest first, so sort chronologically (should already be sorted but good practice)
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
    """Returns a dictionary representing a flat position with Decimal types."""
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
    Fetches current position details for the symbol. Includes simple rate limiting.
    Returns a dictionary representing the position (using Decimals), or None if rate limited.
    On error or if flat, returns a dictionary representing a flat position.
    """
    global last_position_check_time
    if not session:
        log.error("HTTP session not available for get_current_position.")
        return _get_flat_position_representation() # Return flat on session error
    if not market_info: # Need market info for min qty check
        log.error("Market info not available for get_current_position.")
        return _get_flat_position_representation()

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
            position_list = response['result']['list']
            if not position_list:
                 log.warning(f"Received empty position list for {symbol}. Assuming flat.")
                 return _get_flat_position_representation()

            position = position_list[0]
            # Convert relevant fields to Decimal for precise calculations, handle missing keys gracefully
            pos_data = {
                "size": Decimal(position.get('size', '0')),
                "side": position.get('side', 'None'), # 'Buy', 'Sell', or 'None'
                "avgPrice": Decimal(position.get('avgPrice', '0')),
                "liqPrice": Decimal(position.get('liqPrice', '0')) if position.get('liqPrice') else Decimal(0),
                "unrealisedPnl": Decimal(position.get('unrealisedPnl', '0')),
                "markPrice": Decimal(position.get('markPrice', '0')), # Useful for context
                "leverage": Decimal(position.get('leverage', '0')), # Confirm leverage setting
                # Add other fields if needed (e.g., positionValue, riskId, riskLimitValue)
            }
            # Treat negligible size as flat (compare against minOrderQty)
            min_qty = market_info['lotSizeFilter']['minOrderQty']
            if pos_data['size'] < min_qty:
                 if pos_data['size'] > 0: # Log only if size is non-zero but below minimum
                     log.debug(f"Position size {pos_data['size']} is below minimum order quantity ({min_qty}), treating as flat.")
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
    except (InvalidOperation, TypeError, ValueError) as e: # Catch conversion errors
         last_position_check_time = time.time() # Update time even on exception
         log.error(f"Error converting position data to Decimal for {symbol}: {e}")
         return _get_flat_position_representation()
    except Exception as e:
        last_position_check_time = time.time() # Update time even on exception
        log.error(f"Exception fetching position for {symbol}: {e}", exc_info=(log_level <= logging.DEBUG))
        return _get_flat_position_representation() # Return flat representation on exception

def get_wallet_balance(account_type: str = "UNIFIED", coin: str = "USDT") -> Optional[Decimal]:
    """Fetches account equity for risk calculation (assumes UNIFIED account type)."""
    if not session:
        log.error("HTTP session not available for get_wallet_balance.")
        return None
    try:
        # V5 Unified Trading uses get_wallet_balance
        response = session.get_wallet_balance(accountType=account_type, coin=coin)
        if response.get('retCode') == 0 and response.get('result', {}).get('list'):
            balance_info_list = response['result']['list']
            if not balance_info_list:
                log.error(f"Empty balance list received for account type {account_type}, coin {coin}.")
                return None
            # Find the specific coin balance info within the list
            balance_info = next((item for item in balance_info_list if item.get('coin') == coin), None)

            if not balance_info:
                 log.error(f"Could not find balance info for coin {coin} in the response list.")
                 return None

            # Use 'equity' as the basis for risk calculation in Unified account
            if 'equity' in balance_info:
                equity_str = balance_info['equity']
                if equity_str is None or equity_str == "":
                     log.error(f"Equity field is null or empty for {coin}.")
                     return None
                equity = Decimal(equity_str)
                log.debug(f"Account Equity ({coin}): {equity}")
                if equity < 0:
                    log.warning(f"Account equity is negative: {equity}")
                # Consider returning 0 if equity is negative for risk calculation?
                # return max(Decimal(0), equity)
                return equity
            else:
                log.warning(f"Could not find 'equity' field in wallet balance response for {account_type}/{coin}.")
                # Fallback: Try 'walletBalance'? Less accurate for risk based on margin.
                if 'walletBalance' in balance_info:
                     wallet_balance_str = balance_info['walletBalance']
                     if wallet_balance_str is None or wallet_balance_str == "":
                          log.error(f"Fallback 'walletBalance' field is null or empty for {coin}.")
                          return None
                     wallet_balance = Decimal(wallet_balance_str)
                     log.warning(f"Falling back to walletBalance: {wallet_balance}")
                     return wallet_balance # return max(Decimal(0), wallet_balance)
                log.error("Neither 'equity' nor 'walletBalance' found in balance response for {coin}.")
                return None
        else:
            log.error(f"Failed to get wallet balance: {response.get('retMsg', 'Unknown Error')} "
                      f"(Code: {response.get('retCode', 'N/A')})")
            return None
    except (InvalidOperation, TypeError, ValueError) as e: # Added ValueError
         log.error(f"Error converting wallet balance to Decimal: {e}")
         return None
    except Exception as e:
        log.error(f"Exception fetching wallet balance: {e}", exc_info=(log_level <= logging.DEBUG))
        return None

def calculate_order_qty(entry_price: float, sl_price: float, risk_percent: float) -> Optional[float]:
    """
    Calculates order quantity based on risk percentage, SL distance, and account equity.
    Rounds down to the nearest valid quantity step and checks against min/max limits.
    Uses Decimal for precision. Returns the final quantity as a float.
    """
    if not market_info or not strategy_instance:
        log.error("Market info or strategy instance not available for quantity calculation.")
        return None
    # Validate inputs rigorously
    if not all(isinstance(p, (float, int)) and np.isfinite(p) for p in [entry_price, sl_price]):
        log.error(f"Invalid input for quantity calculation: entry={entry_price}, sl={sl_price} must be finite numbers.")
        return None
    if not isinstance(risk_percent, (float, int)) or not 0 < risk_percent <= 100:
        log.error(f"Invalid risk percentage: {risk_percent}. Must be > 0 and <= 100.")
        return None

    try:
        entry_decimal = Decimal(str(entry_price))
        sl_decimal = Decimal(str(sl_price))
        tick_size = strategy_instance.tick_size # Already Decimal
        qty_step = strategy_instance.qty_step # Already Decimal
        min_qty_decimal = market_info['lotSizeFilter']['minOrderQty'] # Already Decimal
        max_qty_decimal = market_info['lotSizeFilter']['maxOrderQty'] # Already Decimal

        # Ensure SL is meaningfully different from entry (at least one tick away)
        sl_distance_points = abs(entry_decimal - sl_decimal)
        if sl_distance_points < tick_size:
            log.error(f"Stop loss price {sl_decimal} is too close to entry price {entry_decimal} "
                      f"(distance: {sl_distance_points}, tick size: {tick_size}). Cannot calculate quantity.")
            return None

        balance = get_wallet_balance()
        if balance is None:
            log.error("Cannot calculate order quantity: Failed to retrieve wallet balance.")
            return None
        if balance <= 0:
            log.warning(f"Cannot calculate order quantity: Account balance is zero or negative ({balance}).")
            return None

        risk_amount = balance * (Decimal(str(risk_percent)) / 100)

        # For Linear contracts (XXX/USDT), PnL is in Quote currency (USDT).
        # Loss per contract = Qty (in Base) * SL_Distance (in Quote)
        # We want: Qty * SL_Distance <= Risk Amount
        # Qty (in Base Asset, e.g., BTC) = Risk Amount / SL_Distance
        # Handle potential division by zero although checked earlier
        if sl_distance_points <= 0:
             log.error("Stop loss distance calculated as zero or negative. Cannot calculate quantity.")
             return None
        qty_base = risk_amount / sl_distance_points

    except (InvalidOperation, DivisionByZero, TypeError, ValueError) as e: # Added ValueError
        log.error(f"Error during quantity calculation math: {e}")
        return None
    except Exception as e: # Catch any other unexpected errors
        log.error(f"Unexpected error during quantity calculation: {e}", exc_info=(log_level <= logging.DEBUG))
        return None

    # --- Rounding and Limit Checks ---
    # Round down to the minimum quantity step using Decimal flooring division
    if qty_step <= 0:
        log.error(f"Invalid quantity step ({qty_step}). Cannot round quantity.")
        return None
    qty_rounded_decimal = (qty_base // qty_step) * qty_step

    # Check against min/max order quantity
    if qty_rounded_decimal < min_qty_decimal:
        log.warning(f"Calculated quantity {qty_rounded_decimal} is below minimum ({min_qty_decimal}).")
        # Decision: Use min_qty (higher risk) or skip trade?
        # Current behavior: Use min_qty but warn about increased risk.
        qty_final_decimal = min_qty_decimal
        # Recalculate actual risk if using min_qty
        actual_risk_amount = min_qty_decimal * sl_distance_points
        try:
            actual_risk_percent = (actual_risk_amount / balance) * 100 if balance > 0 else Decimal(0)
            log.warning(f"Using minimum quantity {min_qty_decimal}. "
                        f"Actual Risk: {actual_risk_amount:.4f} USDT ({actual_risk_percent:.2f}%) "
                        f"(Original target: {risk_percent}%)")
        except DivisionByZero:
            log.error("Division by zero calculating actual risk percentage.") # Should not happen if balance > 0
            # Continue with min_qty anyway? Or return None? Let's continue but log error.

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

    # Use strategy's precision settings for logging clarity
    price_prec = strategy_instance.price_precision
    qty_prec = strategy_instance.qty_precision
    log.info(f"Calculated Order Qty: {qty_final_float:.{qty_prec}f} "
             f"(Balance={balance:.2f}, Risk={risk_percent}%, RiskAmt={risk_amount:.4f}, "
             f"SLDist={sl_distance_points:.{price_prec}f})")
    return qty_final_float

def place_order(symbol: str, side: str, qty: float, price: Optional[float]=None, sl_price: Optional[float]=None, tp_price: Optional[float]=None) -> Optional[Dict]:
    """
    Places an order (Market or Limit) via Bybit API with optional SL/TP.
    Handles rounding, quantity checks, and basic error reporting.
    Uses an order_lock to prevent concurrent order placements.
    Returns the API response dictionary, or None on critical failure before API call.
    """
    if not session or not strategy_instance or not market_info:
        log.error("Cannot place order: Session, strategy instance, or market info missing.")
        return None
    if side not in ["Buy", "Sell"]:
        log.error(f"Invalid order side: {side}. Must be 'Buy' or 'Sell'.")
        return None

    # --- Paper Trading Simulation ---
    if config.get("mode", "Live").lower() == "paper":
        qty_str = f"{qty:.{strategy_instance.qty_precision}f}" if qty else "N/A"
        price_str = f"{price:.{strategy_instance.price_precision}f}" if price else "(Market)"
        sl_str = f"{sl_price:.{strategy_instance.price_precision}f}" if sl_price else "N/A"
        tp_str = f"{tp_price:.{strategy_instance.price_precision}f}" if tp_price else "N/A"
        log.warning(f"[PAPER MODE] Simulating {side} order placement: Qty={qty_str}, Symbol={symbol}, Price={price_str}, SL={sl_str}, TP={tp_str}")
        # Simulate a successful response for paper trading state management
        return {"retCode": 0, "retMsg": "OK", "result": {"orderId": f"paper_{int(time.time())}"}, "paperTrade": True}

    # --- Live Trading Logic ---
    # Acquire lock to ensure only one order placement happens at a time
    with order_lock:
        order_type = config['order']['type']
        min_qty_decimal = market_info['lotSizeFilter']['minOrderQty']
        min_qty_float = float(min_qty_decimal)

        # Final quantity validation and rounding before placing order
        if not isinstance(qty, (float, int)) or not np.isfinite(qty) or qty <= 0:
            log.error(f"Invalid quantity provided for order: {qty}. Must be a positive number.")
            return None

        # Use the strategy's rounding function which should handle precision correctly
        qty_rounded = strategy_instance.round_qty(qty) # Should round down by default
        if qty_rounded is None or qty_rounded <= 0:
             log.error(f"Attempted to place order with zero, negative, or invalid rounded quantity ({qty_rounded}). Original qty: {qty}.")
             return None
        if qty_rounded < min_qty_float:
            log.warning(f"Final quantity {qty_rounded:.{strategy_instance.qty_precision}f} is less than min qty {min_qty_float}. Adjusting to minimum.")
            qty_rounded = min_qty_float

        # Determine reference price for SL/TP validation (use provided price for limit, estimate for market)
        ref_entry_price: Optional[float] = None
        if order_type == "Limit" and price and isinstance(price, (float, int)) and np.isfinite(price):
            ref_entry_price = price # Use the specified limit price as reference
        else: # Market order or missing/invalid limit price
            # Try getting current mark price first (more up-to-date than last close)
            # Note: get_current_position is rate-limited, might return None
            pos_data = get_current_position(symbol)
            if pos_data and pos_data.get('markPrice', Decimal(0)) > 0:
                 ref_entry_price = float(pos_data['markPrice'])
                 log.debug(f"Using current mark price {ref_entry_price} as reference for SL/TP validation.")
            else: # Fallback to last candle close if mark price unavailable or zero
                 with data_lock:
                      if latest_dataframe is not None and not latest_dataframe.empty:
                           last_close = latest_dataframe['close'].iloc[-1]
                           if pd.notna(last_close):
                                ref_entry_price = float(last_close)
                                log.debug(f"Using last close price {ref_entry_price} as reference for SL/TP validation.")

        if ref_entry_price is None:
             log.error("Could not determine a reference entry price for SL/TP validation. Skipping SL/TP placement with order.")
             sl_price = None # Ensure SL/TP are not sent if validation failed
             tp_price = None
        else:
             ref_entry_price = float(ref_entry_price) # Ensure float

        # --- Prepare Order Parameters ---
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side, # "Buy" or "Sell"
            "orderType": order_type,
            "qty": str(qty_rounded), # API requires quantity as a string
            "timeInForce": "GTC", # GoodTillCancel is common for entries with SL/TP
            "reduceOnly": False, # This is an entry order
            "positionIdx": 0 # Required for one-way position mode
            # "orderLinkId": f"entry_{side.lower()}_{int(time.time()*1000)}" # Optional: Custom ID
        }

        # Add price for Limit orders
        limit_price_str: Optional[str] = None
        if order_type == "Limit":
            if price and isinstance(price, (float, int)) and np.isfinite(price):
                # Round limit price according to tick size
                # Be slightly aggressive on limit entry: round UP for BUY, DOWN for SELL
                limit_rounding = ROUND_UP if side == "Buy" else ROUND_DOWN
                limit_price_rounded = strategy_instance.round_price(price, rounding_mode=limit_rounding)
                if limit_price_rounded is None:
                     log.error(f"Failed to round limit price {price}. Order cancelled.")
                     return None
                limit_price_str = str(limit_price_rounded)
                params["price"] = limit_price_str
            else:
                log.error(f"Limit order requires a valid price. Got: {price}. Order cancelled.")
                return None

        # Add SL/TP using Bybit's parameters, with validation against reference price
        sl_price_str: Optional[str] = None
        if sl_price and isinstance(sl_price, (float, int)) and np.isfinite(sl_price):
            # Round SL price (away from entry: DOWN for Buy, UP for Sell)
            sl_rounding = ROUND_DOWN if side == "Buy" else ROUND_UP
            sl_price_rounded = strategy_instance.round_price(sl_price, rounding_mode=sl_rounding)

            if sl_price_rounded is None:
                 log.error(f"Failed to round SL price {sl_price}. SL skipped.")
            elif ref_entry_price is not None:
                # Validate SL relative to reference entry price (allow SL to be equal if needed?)
                # Let's require SL to be strictly worse than entry
                is_invalid_sl = (side == "Buy" and sl_price_rounded >= ref_entry_price) or \
                                (side == "Sell" and sl_price_rounded <= ref_entry_price)
                if is_invalid_sl:
                    log.error(f"Invalid SL price {sl_price_rounded:.{strategy_instance.price_precision}f} for {side} order "
                              f"relative to reference entry {ref_entry_price:.{strategy_instance.price_precision}f}. SL skipped.")
                else:
                    sl_price_str = str(sl_price_rounded)
                    params["stopLoss"] = sl_price_str
            else: # Cannot validate if ref_entry_price is unknown, proceed with caution
                 sl_price_str = str(sl_price_rounded)
                 params["stopLoss"] = sl_price_str
                 log.warning(f"Setting StopLoss at: {sl_price_str} (Could not validate against reference entry price).")

        tp_price_str: Optional[str] = None
        if tp_price and isinstance(tp_price, (float, int)) and np.isfinite(tp_price):
            # Round TP price (towards profit: UP for Buy, DOWN for Sell)
            tp_rounding = ROUND_UP if side == "Buy" else ROUND_DOWN
            tp_price_rounded = strategy_instance.round_price(tp_price, rounding_mode=tp_rounding)

            if tp_price_rounded is None:
                 log.error(f"Failed to round TP price {tp_price}. TP skipped.")
            elif ref_entry_price is not None:
                # Validate TP relative to reference entry price (require TP > entry for Buy, TP < entry for Sell)
                is_invalid_tp = (side == "Buy" and tp_price_rounded <= ref_entry_price) or \
                                (side == "Sell" and tp_price_rounded >= ref_entry_price)
                if is_invalid_tp:
                    log.error(f"Invalid TP price {tp_price_rounded:.{strategy_instance.price_precision}f} for {side} order "
                              f"relative to reference entry {ref_entry_price:.{strategy_instance.price_precision}f}. TP skipped.")
                else:
                    tp_price_str = str(tp_price_rounded)
                    params["takeProfit"] = tp_price_str
            else: # Cannot validate if ref_entry_price is unknown, proceed with caution
                 tp_price_str = str(tp_price_rounded)
                 params["takeProfit"] = tp_price_str
                 log.warning(f"Setting TakeProfit at: {tp_price_str} (Could not validate against reference entry price).")

        # --- Place Order API Call ---
        log.warning(f"Placing {side} {order_type} order: Qty={params['qty']} {symbol} "
                    f"{'@'+limit_price_str if limit_price_str else '(Market)'} "
                    f"SL={sl_price_str or 'N/A'} TP={tp_price_str or 'N/A'}")
        try:
            response = session.place_order(**params)
            log.debug(f"Place Order Response: {json.dumps(response, indent=2)}") # Log full response in debug

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
                if "order cost" in error_msg.lower() or "insufficient available balance" in error_msg.lower():
                    log.error("Hint: Insufficient margin for order cost. Check balance and leverage.")
                return response # Return error response for potential handling upstream
        except Exception as e:
            log.error(f"Exception occurred during order placement: {e}", exc_info=(log_level <= logging.DEBUG))
            return None # Indicate failure to make API call

def close_position(symbol: str, position_data: Dict[str, Any]) -> Optional[Dict]:
    """
    Closes an existing position using a reduce-only market order.
    Uses an order_lock to prevent concurrent closing attempts.
    Returns the API response dictionary, or None on critical failure before API call.
    """
    if not session or not strategy_instance:
        log.error("Cannot close position: Session or strategy instance missing.")
        return None

    # --- Paper Trading Simulation ---
    if config.get("mode", "Live").lower() == "paper":
        log.warning(f"[PAPER MODE] Simulating closing position for {symbol} (Size: {position_data.get('size', 'N/A')}, Side: {position_data.get('side', 'N/A')})")
        # Simulate success
        return {"retCode": 0, "retMsg": "OK", "result": {"orderId": f"paper_close_{int(time.time())}"}, "paperTrade": True}

    # --- Live Trading Logic ---
    # Acquire lock to ensure only one closing order happens at a time
    with order_lock:
        current_size = position_data.get('size', Decimal(0))
        current_side = position_data.get('side', 'None') # 'Buy' or 'Sell'

        # Validate position data before attempting close
        if current_side == 'None' or current_size <= 0:
             log.warning(f"Attempting to close position for {symbol}, but position data indicates it's already flat or size is zero/negative. No action taken.")
             # Return a simulated success response indicating it was already flat
             return {"retCode": 0, "retMsg": "Position already flat or zero size", "result": {}, "alreadyFlat": True}

        # Determine the side needed to close the position
        side_to_close = "Sell" if current_side == "Buy" else "Buy"
        # Quantity to close is the current position size
        qty_to_close_decimal = current_size

        # Round quantity using strategy settings (usually not strictly needed for market close, but good practice)
        # Convert Decimal to float for the rounding function
        try:
            qty_to_close_float = float(qty_to_close_decimal)
        except (TypeError, ValueError):
             log.error(f"Invalid position size for closing: {current_size}. Cannot place closing order.")
             return None

        # Use default rounding (usually round down) for closing order
        qty_to_close_rounded = strategy_instance.round_qty(qty_to_close_float)

        if qty_to_close_rounded is None or qty_to_close_rounded <= 0:
             log.error(f"Calculated quantity to close for {symbol} is zero, negative or invalid ({qty_to_close_rounded}). Original size: {current_size}. Cannot place closing order.")
             return None

        log.warning(f"Attempting to close {current_side} position for {symbol} (Size: {current_size}). Placing {side_to_close} Market order (Reduce-Only)...")

        # --- Cancel Existing SL/TP Orders ---
        # Optional but recommended: Cancel existing SL/TP orders before closing market order
        # This might prevent conflicts or unexpected partial fills of SL/TP
        log.info(f"Attempting to cancel existing Stop Orders (SL/TP) for {symbol} before closing...")
        try:
            # Use orderFilter="StopOrder" to target only SL/TP
            response_cancel = session.cancel_all_orders(category="linear", symbol=symbol, orderFilter="StopOrder")
            log.debug(f"Cancel SL/TP Response: {json.dumps(response_cancel, indent=2)}")
            if response_cancel.get('retCode') != 0:
                 # Log warning but proceed with close attempt anyway
                 # Common reason: No active stop orders found (retCode 110001, retMsg "No order found") - this is OK.
                 if response_cancel.get('retCode') == 110001 and "No order found" in response_cancel.get('retMsg', ''):
                      log.info("No active stop orders found to cancel.")
                 else:
                      log.warning(f"Could not cancel stop orders before closing: {response_cancel.get('retMsg', 'Error')} (Code: {response_cancel.get('retCode')}). Proceeding with close.")
            else:
                 cancelled_count = len(response_cancel.get('result', {}).get('list', []))
                 log.info(f"Successfully cancelled {cancelled_count} open stop order(s).")
            time.sleep(0.5) # Brief pause after cancellation attempt before placing close order
        except Exception as cancel_e:
             log.error(f"Exception occurred during stop order cancellation: {cancel_e}", exc_info=(log_level <= logging.DEBUG))
             log.warning("Proceeding with close attempt despite cancellation error.")

        # --- Place Closing Market Order ---
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side_to_close,
            "orderType": "Market", # Use Market order for immediate closure
            "qty": str(qty_to_close_rounded), # API requires string quantity
            "reduceOnly": True, # CRITICAL: Ensures this order only reduces/closes the position
            "positionIdx": 0 # Required for one-way mode
            # "orderLinkId": f"close_{side_to_close.lower()}_{int(time.time()*1000)}" # Optional
        }
        try:
            response = session.place_order(**params)
            log.debug(f"Close Position Order Response: {json.dumps(response, indent=2)}")

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
                # 110044: Reduce only order cannot be placed as you have no position
                if error_code in [110043, 3400070, 110025, 110044]:
                    log.warning("Reduce-only error likely means position size changed or closed between check and execution. Re-checking position soon.")
                    global last_position_check_time # Force re-check sooner
                    last_position_check_time = 0
                    # Return the error response but add a flag indicating likely already flat
                    response["alreadyFlatHint"] = True
                return response # Return error response
        except Exception as e:
            log.error(f"Exception occurred during position closing: {e}", exc_info=(log_level <= logging.DEBUG))
            return None # Indicate failure to make API call

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
        leverage_str = str(float(leverage)) # API expects string representation of a float (e.g., "25.0")
        response = session.set_leverage(
            category="linear",
            symbol=symbol,
            buyLeverage=leverage_str,
            sellLeverage=leverage_str
        )
        log.debug(f"Set Leverage Response: {json.dumps(response, indent=2)}")
        if response.get('retCode') == 0:
            log.info(f"Leverage for {symbol} set to {leverage}x successfully.")
        else:
            error_code = response.get('retCode')
            error_msg = response.get('retMsg', 'Unknown Error')
            # Common error: 110043 means leverage not modified (it was already set to this value)
            # Note: Log showed 110043 in original problem description, let's use that code.
            if error_code == 110043: # Leverage not modified
                 log.warning(f"Leverage for {symbol} already set to {leverage}x (Code: {error_code} - Not modified).")
            # Error 110045 might indicate trying to change leverage with open position/orders
            elif error_code == 110045:
                 log.error(f"Failed to set leverage for {symbol}: Cannot modify leverage with open positions or orders. (Code: {error_code})")
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

    try:
        topic = msg.get("topic", "")
        msg_type = msg.get("type", "") # snapshot or delta
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
            is_confirmed = kline_item.get('confirm', False)
            if not is_confirmed:
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

            except (KeyError, ValueError, TypeError, IndexError) as e: # Added IndexError
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
                                   f"AvgPrice={pos_update.get('avgPrice')}, PnL={pos_update.get('unrealisedPnl')}, "
                                   f"LiqPrice={pos_update.get('liqPrice')}{Style.RESET_ALL}")
                          # OPTIONAL: Directly update internal state based on WS.
                          # Requires careful locking (position_lock) and state management.
                          # Safer approach: Trigger a faster HTTP position check if needed,
                          # or use this WS update to confirm state after placing/closing orders.
                          # Example: If we just closed, seeing size=0 here confirms success.
                          # global last_position_check_time
                          # last_position_check_time = 0 # Force check on next main loop iteration? Risky if updates are frequent.
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
                                  f"AvgFillPrice={order_update.get('avgPrice')}, CumExecQty={order_update.get('cumExecQty')}, "
                                  f"RejectReason={order_update.get('rejectReason', 'N/A')}" # Added reject reason
                                  f"{Style.RESET_ALL}")
                         # Can use this to track fills, cancellations, SL/TP triggers etc.
                         # Example: If orderStatus is 'Filled' and it matches an expected entry/exit order ID.
                         if order_status in ["Filled", "PartiallyFilled", "Cancelled", "Rejected", "Triggered"]:
                             # Potentially trigger position check or update internal state
                             # If an order is filled/triggered, a position check might be useful soon.
                             if order_status in ["Filled", "Triggered"]:
                                 log.info(f"Order {order_id} reached final state: {order_status}. Triggering position check.")
                                 global last_position_check_time
                                 last_position_check_time = 0 # Force check soon
                             pass
                         pass # Currently just logging the update

        # --- Handle Connection Status / Authentication / Subscription ---
        elif "op" in msg:
            op = msg.get("op")
            if op == "auth":
                if msg.get("success"):
                    log.info(f"{Fore.GREEN}WebSocket authenticated successfully.{Style.RESET_ALL}")
                    # Authentication success doesn't mean subscriptions are ready yet.
                else:
                    log.error(f"{Fore.RED}WebSocket authentication failed: {msg.get('ret_msg', 'No reason provided')}{Style.RESET_ALL}")
                    # Consider stopping the bot or attempting reconnect if auth fails persistently
                    ws_connected.clear() # Ensure connection status reflects failure
            elif op == "subscribe":
                 if msg.get("success"):
                    subscribed_topics = msg.get('ret_msg') or msg.get('args') # Location varies slightly
                    log.info(f"{Fore.GREEN}WebSocket subscribed successfully to: {subscribed_topics}{Style.RESET_ALL}")
                    # This is the point where we can consider the connection fully ready
                    ws_connected.set()
                 else:
                    log.error(f"{Fore.RED}WebSocket subscription failed: {msg.get('ret_msg', 'No reason provided')}{Style.RESET_ALL}")
                    ws_connected.clear() # Signal potential connection issue
                    # Consider stopping or retrying
            elif op == "pong":
                log.debug("WebSocket Pong received (heartbeat OK)")
            elif "success" in msg and not msg.get("success"):
                # Catch other potential operation failures (e.g., unsubscribe)
                log.error(f"{Fore.RED}WebSocket operation '{op}' failed: {msg.get('ret_msg', msg)}{Style.RESET_ALL}")
            else:
                 log.debug(f"Received WS operation response: {msg}") # Log other ops if needed

        # --- Handle other message types if necessary ---
        # E.g., execution reports, wallet updates (if subscribed)
        # elif topic.startswith("execution"): ...
        # elif topic.startswith("wallet"): ...
        else:
            log.debug(f"Unhandled WS message type/topic: {msg}")

    except Exception as e:
        log.error(f"Unhandled exception in handle_ws_message: {e}", exc_info=True)
        log.error(f"Problematic WS message: {msg}")


def run_websocket_loop():
    """Target function for the WebSocket thread. Handles connection, subscription, and reconnection."""
    global ws, ws_connected
    log.info("WebSocket thread starting...")
    ping_interval = config['websocket']['ping_interval']

    while not stop_event.is_set():
        # Initialize WebSocket connection within the loop for reconnection logic
        log.info("Initializing WebSocket connection...")
        ws_connected.clear() # Signal connection is not ready
        ws = None # Ensure ws is reset before creating a new one
        try:
            # Check pybit version compatibility if TypeError persists.
            # The 'message_handler' argument in the constructor is expected in recent pybit v5 versions.
            ws = WebSocket(
                testnet=TESTNET,
                channel_type="private", # Use private for positions/orders/auth
                api_key=API_KEY,
                api_secret=API_SECRET,
                message_handler=handle_ws_message # Pass callback function
            )
            log.info("WebSocket object created. Subscribing...")

            # Define required subscriptions
            kline_topic = f"kline.{config['interval']}.{config['symbol']}"
            position_topic = f"position.{config['symbol']}" # Specific symbol position updates
            order_topic = "order" # All order updates for the account
            # Add other topics if needed (e.g., public trades, orderbook)
            topics_to_subscribe = [kline_topic, position_topic, order_topic]

            # Subscribe *before* starting the connection loop
            ws.subscribe(topics_to_subscribe)
            log.info(f"Subscription request sent for topics: {topics_to_subscribe}")

            # Start the blocking run_forever loop
            # This handles the connection, pinging, and message dispatching
            log.info(f"WebSocket run_forever starting (ping interval: {ping_interval}s)...")
            ws.run_forever(ping_interval=ping_interval) # This blocks until exit() or error

        except TypeError as te:
             # Specifically catch TypeError which might indicate the message_handler issue
             log.error(f"WebSocket Initialization TypeError: {te}. "
                       f"This might be due to an incompatible pybit version or incorrect arguments. "
                       f"Ensure pybit is up-to-date (>=5.4.0 recommended).", exc_info=True)
             ws_connected.clear()
             # No ws object likely created, or it's in a bad state
             ws = None
        except Exception as e:
            log.error(f"WebSocket run_forever error or initialization/subscription failed: {e}", exc_info=(log_level <= logging.DEBUG))
            ws_connected.clear() # Ensure state is cleared on error
            if ws:
                try:
                    ws.exit() # Attempt to clean up the failed connection
                except Exception as exit_e:
                    log.error(f"Error trying to exit failed WebSocket: {exit_e}")
                ws = None

        # --- Reconnection Logic ---
        if stop_event.is_set():
            log.info("WebSocket stopping due to stop_event.")
            break # Exit loop if stopping

        reconnect_delay = 15 # Seconds
        log.info(f"Attempting to reconnect WebSocket after error in {reconnect_delay} seconds...")
        # Use stop_event.wait() for interruptible sleep
        interrupted = stop_event.wait(timeout=reconnect_delay)
        if interrupted:
             log.info("WebSocket reconnection interrupted by stop_event.")
             break # Exit loop if stopping during wait

    # --- Thread Exit ---
    log.info("WebSocket thread finished.")
    ws_connected.clear()
    # Ensure ws is cleaned up if loop exits while ws object exists
    if ws:
        try:
            ws.exit()
        except Exception as exit_e:
            log.debug(f"Exception during final WebSocket exit: {exit_e}")
        ws = None


def start_websocket_connection() -> bool:
    """Initializes and starts the WebSocket connection in a separate thread."""
    global ws_thread
    if not API_KEY or not API_SECRET:
        log.error("Cannot start WebSocket: API credentials missing.")
        return False
    if ws_thread and ws_thread.is_alive():
        log.warning("WebSocket thread is already running.")
        return True # Already running, assume it's okay

    log.info("Starting WebSocket thread...")
    stop_event.clear() # Ensure stop event is clear before starting
    ws_connected.clear() # Reset connection status event

    try:
        # The run_websocket_loop now handles initialization and subscription
        ws_thread = threading.Thread(target=run_websocket_loop, daemon=True, name="WebSocketThread")
        ws_thread.start()

        # Wait for the connection and subscription confirmation (signaled by ws_connected event)
        connect_timeout = config['websocket'].get('connect_timeout', 20) # Use configured timeout
        log.info(f"Waiting up to {connect_timeout}s for WebSocket connection and subscription confirmation...")
        if ws_connected.wait(timeout=connect_timeout):
            log.info(f"{Fore.GREEN}WebSocket connected and subscribed successfully.{Style.RESET_ALL}")
            return True
        else:
            # This block is reached if ws_connected.wait() times out
            log.error(f"{Fore.RED}WebSocket did not confirm subscription within {connect_timeout}s. "
                      f"Check logs (especially for auth/subscribe errors), connection, credentials, API permissions.{Style.RESET_ALL}")
            # Attempt cleanup even if connection failed to confirm
            stop_websocket_connection() # Signal the thread to stop and clean up
            return False

    except Exception as e:
        log.critical(f"CRITICAL: Failed to start WebSocket thread: {e}", exc_info=True)
        stop_websocket_connection() # Attempt cleanup
        return False

def stop_websocket_connection():
    """Stops the WebSocket connection and joins the thread gracefully."""
    global ws, ws_thread, ws_connected
    log.info("Attempting to stop WebSocket connection...")
    stop_event.set() # Signal the loop in the thread to stop
    ws_connected.clear() # Signal connection is down/stopping

    current_thread = ws_thread # Capture current thread object

    if ws:
        try:
            log.info("Calling WebSocket exit()...")
            ws.exit() # Signal run_forever to stop blocking
            log.info("WebSocket exit() called.")
        except Exception as e:
            log.error(f"Error calling WebSocket exit(): {e}")
    else:
         log.info("WebSocket object not found (might be between reconnect attempts or failed init), relying on stop_event.")

    if current_thread and current_thread.is_alive():
        log.info(f"Waiting for WebSocket thread '{current_thread.name}' to join...")
        current_thread.join(timeout=10) # Wait up to 10 seconds for the thread to finish
        if current_thread.is_alive():
            log.warning(f"WebSocket thread '{current_thread.name}' did not stop gracefully after 10 seconds.")
        else:
            log.info(f"WebSocket thread '{current_thread.name}' joined successfully.")
    elif current_thread:
         log.info(f"WebSocket thread '{current_thread.name}' was already stopped.")
    else:
         log.info("No WebSocket thread was found to stop.")

    # Clean up global variables
    ws = None
    ws_thread = None
    ws_connected.clear()
    log.info("WebSocket connection stopped.")

# --- Signal Processing & Trade Execution ---

def calculate_sl_tp(side: str, entry_price: float, last_atr: Optional[float], results: AnalysisResults) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculates Stop Loss and Take Profit prices based on strategy config and market info.
    Uses Decimal for precision in calculations. Returns final prices as floats.
    """
    if not strategy_instance or not market_info:
        log.error("Cannot calculate SL/TP: Strategy instance or market info missing.")
        return None, None
    if not isinstance(entry_price, (float, int)) or not np.isfinite(entry_price):
        log.error(f"Cannot calculate SL/TP: Invalid entry price {entry_price}.")
        return None, None
    if side not in ["Buy", "Sell"]:
         log.error(f"Cannot calculate SL/TP: Invalid side {side}.")
         return None, None

    sl_price_raw: Optional[float] = None
    tp_price_raw: Optional[float] = None
    sl_price_final: Optional[float] = None
    tp_price_final: Optional[float] = None

    try:
        entry_decimal = Decimal(str(entry_price))
        sl_method = strategy_instance.sl_method
        sl_atr_multiplier = Decimal(str(strategy_instance.sl_atr_multiplier))
        tp_ratio = Decimal(str(config['order'].get('tp_ratio', 2.0))) # Get TP ratio from config
        tick_size = strategy_instance.tick_size # Already Decimal
        last_atr_decimal = Decimal(str(last_atr)) if last_atr and np.isfinite(last_atr) and last_atr > 0 else None

        # --- Calculate Stop Loss ---
        sl_target_price_decimal: Optional[Decimal] = None

        if sl_method == "ATR":
            if last_atr_decimal:
                sl_distance = last_atr_decimal * sl_atr_multiplier
                sl_target_price_decimal = entry_decimal - sl_distance if side == "Buy" else entry_decimal + sl_distance
            else:
                log.error(f"Cannot calculate ATR Stop Loss: Invalid or missing last_atr value ({last_atr}).")
                return None, None # Cannot proceed without valid SL calculation method

        elif sl_method == "OB":
            # Define buffers using Decimal
            sl_buffer_atr_fraction = Decimal("0.1") # Buffer as fraction of ATR
            sl_buffer_price_fraction = Decimal("0.0005") # Buffer as fraction of price (fallback)
            buffer: Decimal
            if last_atr_decimal:
                 buffer = last_atr_decimal * sl_buffer_atr_fraction
            else:
                 buffer = entry_decimal * sl_buffer_price_fraction
                 log.warning("ATR unavailable for OB SL buffer, using price fraction fallback.")
            # Ensure buffer is at least one tick size
            buffer = max(buffer, tick_size)

            ob_sl_target_price: Optional[Decimal] = None
            if side == "Buy":
                # Find lowest bottom of active bull OBs below entry
                active_bull_boxes = results.get('active_bull_boxes', [])
                relevant_obs = [b for b in active_bull_boxes if Decimal(str(b['bottom'])) < entry_decimal]
                if relevant_obs:
                    lowest_bottom = min(Decimal(str(b['bottom'])) for b in relevant_obs)
                    ob_sl_target_price = lowest_bottom - buffer
                    log.debug(f"OB SL (Buy): Found relevant bull OBs. Lowest bottom={lowest_bottom}, Buffer={buffer}, Target SL={ob_sl_target_price}")
                else:
                    log.warning("OB SL method chosen for BUY, but no active Bull OB found below entry. Falling back to ATR SL.")
                    # Fallback logic moved outside the if/else block
            else: # side == "Sell"
                # Find highest top of active bear OBs above entry
                active_bear_boxes = results.get('active_bear_boxes', [])
                relevant_obs = [b for b in active_bear_boxes if Decimal(str(b['top'])) > entry_decimal]
                if relevant_obs:
                    highest_top = max(Decimal(str(b['top'])) for b in relevant_obs)
                    ob_sl_target_price = highest_top + buffer
                    log.debug(f"OB SL (Sell): Found relevant bear OBs. Highest top={highest_top}, Buffer={buffer}, Target SL={ob_sl_target_price}")
                else:
                    log.warning("OB SL method chosen for SELL, but no active Bear OB found above entry. Falling back to ATR SL.")
                    # Fallback logic moved outside the if/else block

            # Assign OB target or handle fallback
            if ob_sl_target_price is not None:
                 sl_target_price_decimal = ob_sl_target_price
            elif last_atr_decimal: # Fallback to ATR if OB not found/applicable
                 sl_distance = last_atr_decimal * sl_atr_multiplier
                 sl_target_price_decimal = entry_decimal - sl_distance if side == "Buy" else entry_decimal + sl_distance
                 log.debug(f"OB SL fallback to ATR: ATR={last_atr_decimal}, Multiplier={sl_atr_multiplier}, Target SL={sl_target_price_decimal}")
            else:
                 log.error("Cannot set SL: OB method failed and fallback ATR is unavailable.")
                 return None, None

        else: # Should not happen if config validation is correct
             log.error(f"Unknown SL method: {sl_method}")
             return None, None

        # Check if a valid SL target was determined
        if sl_target_price_decimal is None:
            log.error("Stop Loss price could not be calculated. Cannot determine trade parameters.")
            return None, None

        # --- Validate and Round SL ---
        # Ensure SL is on the correct side of the entry price and at least one tick away
        is_invalid_raw_sl = (side == "Buy" and sl_target_price_decimal >= entry_decimal - tick_size) or \
                            (side == "Sell" and sl_target_price_decimal <= entry_decimal + tick_size)

        if is_invalid_raw_sl:
            # If the calculated SL is invalid, log error. Consider if a fallback is possible/desirable here.
            # For now, treat it as a critical failure for this trade setup.
            log.error(f"Calculated raw SL price {sl_target_price_decimal} is not logical or too close for a {side} trade from entry {entry_decimal} (tick size: {tick_size}). Cannot set SL.")
            # Maybe attempt a simple ATR fallback one last time if not already done?
            if sl_method != "ATR" and last_atr_decimal:
                 log.warning("Attempting final ATR fallback for invalid SL...")
                 sl_distance = last_atr_decimal * sl_atr_multiplier
                 sl_target_price_decimal = entry_decimal - sl_distance if side == "Buy" else entry_decimal + sl_distance
                 is_invalid_raw_sl = (side == "Buy" and sl_target_price_decimal >= entry_decimal - tick_size) or \
                                     (side == "Sell" and sl_target_price_decimal <= entry_decimal + tick_size)
                 if is_invalid_raw_sl:
                      log.error("Final ATR fallback SL is also invalid. Cannot determine SL.")
                      return None, None
                 else:
                      log.info("Using final ATR fallback SL.")
            else:
                 return None, None # No valid SL found

        # Convert raw SL to float before rounding
        sl_price_raw = float(sl_target_price_decimal)

        # Round SL price (away from entry: DOWN for Buy, UP for Sell)
        sl_rounding = ROUND_DOWN if side == "Buy" else ROUND_UP
        sl_price_final = strategy_instance.round_price(sl_price_raw, rounding_mode=sl_rounding)
        if sl_price_final is None or not np.isfinite(sl_price_final):
             log.error(f"Failed to round valid SL price {sl_price_raw}.")
             return None, None # Cannot proceed without valid rounded SL
        log.info(f"Calculated SL price for {side}: {sl_price_final:.{strategy_instance.price_precision}f}")

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
                 # Final validation for TP vs Entry (TP must be better than entry)
                 is_invalid_tp = (side == "Buy" and tp_price_final <= entry_price) or \
                                 (side == "Sell" and tp_price_final >= entry_price)
                 if is_invalid_tp:
                      log.warning(f"Calculated TP price {tp_price_final:.{strategy_instance.price_precision}f} is not logical for {side} from entry {entry_price:.{strategy_instance.price_precision}f}. TP will not be set.")
                      tp_price_final = None
                 else:
                      log.info(f"Calculated TP price for {side}: {tp_price_final:.{strategy_instance.price_precision}f} (Ratio: {tp_ratio})")
        else:
            log.warning("Cannot calculate TP: SL distance is zero or TP ratio is not positive.")
            tp_price_final = None

    except (InvalidOperation, TypeError, DivisionByZero, ValueError) as e: # Added ValueError
         log.error(f"Decimal or calculation error during SL/TP calculation: {e}")
         return None, None
    except Exception as e:
         log.error(f"Unexpected error during SL/TP calculation: {e}", exc_info=(log_level <= logging.DEBUG))
         return None, None

    # Return final rounded float values
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
    last_atr = results.get('last_atr') # May be None if not calculated/available
    symbol = config['symbol']

    log.debug(f"Processing Signal: {signal}, Last Close: {last_close}, Last ATR: {last_atr}")

    # Validate essential data from results
    if signal is None:
        log.debug("No signal generated ('None'). No action.")
        return
    if signal not in ["BUY", "SELL", "EXIT_LONG", "EXIT_SHORT", "HOLD"]:
         log.warning(f"Received unknown signal type: {signal}. Ignoring.")
         return
    if pd.isna(last_close) or not isinstance(last_close, (float, int)):
        log.warning(f"Cannot process signal: Invalid last close price ({last_close}).")
        return

    # --- Get Current Position State ---
    # Crucial for deciding whether to enter or exit. Rate limiting is handled internally.
    position_data = get_current_position(symbol)
    if position_data is None:
        log.warning("Position check skipped due to rate limit. Will re-evaluate on next candle.")
        return # Wait for next cycle if check was skipped

    # Use the fetched position data (which uses Decimal)
    current_pos_size = position_data.get('size', Decimal(0))
    current_pos_side = position_data.get('side', 'None') # 'Buy', 'Sell', or 'None'

    # Check size against minimum order quantity for robustness
    min_qty_decimal = market_info['lotSizeFilter']['minOrderQty']
    is_long = current_pos_side == 'Buy' and current_pos_size >= min_qty_decimal
    is_short = current_pos_side == 'Sell' and current_pos_size >= min_qty_decimal
    is_flat = not is_long and not is_short

    log.info(f"Current Position: {'Long' if is_long else 'Short' if is_short else 'Flat'} (Size: {current_pos_size}) | Signal: {signal}")

    # --- Execute Actions Based on Signal and Position State ---

    # BUY Signal: Enter Long if Flat
    if signal == "BUY" and is_flat:
        log.warning(f"{Fore.GREEN}{Style.BRIGHT}BUY Signal Received - Attempting to Enter Long.{Style.RESET_ALL}")
        sl_price, tp_price = calculate_sl_tp("Buy", last_close, last_atr, results)
        if sl_price is None: # calculate_sl_tp returns None for SL if it cannot be determined
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
         log.warning("BUY Signal received while Short. Strategy should have generated EXIT_SHORT first. Closing Short position before considering Long.")
         close_position(symbol, position_data) # Close the short first
         # Note: This won't immediately open a long; the next candle's signal will determine that.
    elif signal == "SELL" and is_long:
         log.warning("SELL Signal received while Long. Strategy should have generated EXIT_LONG first. Closing Long position before considering Short.")
         close_position(symbol, position_data) # Close the long first
         # Note: This won't immediately open a short.


# --- Graceful Shutdown ---
def handle_shutdown_signal(signum, frame):
    """Handles termination signals (SIGINT, SIGTERM) for graceful shutdown."""
    if stop_event.is_set(): # Prevent running multiple times if signal received repeatedly
         log.warning("Shutdown already in progress.")
         return
    log.warning(f"Shutdown signal {signal.Signals(signum).name} ({signum}) received. Initiating graceful shutdown...")
    stop_event.set() # Signal all loops and threads to stop

    # 1. Stop WebSocket first to prevent processing new data/signals
    stop_websocket_connection() # This function now handles waiting for the thread

    # 2. Optional: Implement logic to manage open positions/orders on shutdown
    #    USE WITH EXTREME CAUTION - unexpected closure can lead to losses.
    close_on_exit = config.get("close_position_on_exit", False)
    if close_on_exit and config.get("mode", "Live").lower() != "paper":
        log.warning("Attempting to close open position on exit (close_position_on_exit=True)...")
        # Need to ensure we get the latest position data, might need retry if rate limited
        pos_data = None
        max_retries = 3
        # Use a shorter delay than the interval for faster shutdown checks
        retry_delay = max(1.0, POSITION_CHECK_INTERVAL / 3.0) # Wait at least 1 second

        for attempt in range(max_retries):
             if stop_event.is_set() and attempt > 0: # Check if stop was re-triggered during wait
                  log.warning("Shutdown signal received again during position check retries. Aborting close attempt.")
                  break

             # Force position check by resetting timer
             global last_position_check_time
             last_position_check_time = 0.0 # Force check on next call
             log.info(f"Checking position status for shutdown close (Attempt {attempt+1}/{max_retries})...")
             pos_data = get_current_position(config['symbol'])

             if pos_data is not None: # Got data (even if flat), break retry
                  break

             log.warning(f"Position check rate limited or failed during shutdown. Retrying in {retry_delay:.1f}s...")
             # Use interruptible sleep
             interrupted = stop_event.wait(timeout=retry_delay)
             if interrupted:
                  log.warning("Shutdown signal received again while waiting for position check retry. Aborting close attempt.")
                  break # Exit retry loop if interrupted

        # Check if position data was successfully retrieved and if position is open
        if pos_data and pos_data.get('size', Decimal(0)) > 0 and pos_data.get('side') != 'None':
             log.warning(f"Found open {pos_data.get('side')} position (Size: {pos_data.get('size')}). Attempting market close.")
             close_response = close_position(config['symbol'], pos_data) # Pass the fetched data

             # Check response from close_position
             if close_response and close_response.get('retCode') == 0:
                  if close_response.get('alreadyFlat'):
                       log.info("Position was already closed before shutdown close attempt could execute.")
                  else:
                       log.info("Position close order placed successfully during shutdown.")
                       # Optionally wait briefly to see if WS confirms closure?
                       time.sleep(2) # Short wait, WS might already be stopped
             elif close_response and close_response.get('alreadyFlatHint'):
                  log.warning("Close order failed, likely because position was already closed or size changed.")
             elif close_response: # Close order failed for other reasons
                  log.error(f"Failed to place position close order during shutdown. Response Code: {close_response.get('retCode')}, Msg: {close_response.get('retMsg')}")
             else: # close_position returned None (e.g., exception before API call)
                  log.error("Failed to execute close_position function during shutdown.")
        elif pos_data: # Position data retrieved, but position is flat
             log.info("No open position found to close on exit.")
        else: # Failed to retrieve position data after retries
             log.error("Could not determine position status during shutdown due to repeated errors/rate limit. Manual check required.")

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
    # load_config handles its own logging and exits on critical error
    config = load_config()

    # Setup logging based on config (now safe to use log)
    setup_logging(config.get("log_level", "INFO"))
    log.info(f"Logging level set to: {logging.getLevelName(log.level)}")
    # Use default=str to handle Decimal serialization in debug log
    log.debug(f"Full Config: {json.dumps(config, indent=2, default=str)}")

    # Register signal handlers for graceful shutdown (Ctrl+C, kill)
    try:
        signal.signal(signal.SIGINT, handle_shutdown_signal)
        signal.signal(signal.SIGTERM, handle_shutdown_signal)
        log.info("Registered shutdown signal handlers.")
    except Exception as e:
         log.error(f"Could not register signal handlers: {e}. Shutdown might not be graceful.")

    # Connect to Bybit HTTP API
    session = connect_bybit()
    if not session:
        # connect_bybit already logs critical error and exits
        sys.exit(1) # Redundant, but safe

    # Get Market Info (Symbol, Precision, Limits)
    market_info = get_market_info(config['symbol'])
    if not market_info:
        log.critical(f"Could not retrieve valid market info for symbol '{config['symbol']}'. Check symbol and API connection. Exiting.")
        sys.exit(1)
    # Log key market info using Decimal values directly
    log.info(f"Market Info: TickSize={market_info['priceFilter']['tickSize']}, QtyStep={market_info['lotSizeFilter']['qtyStep']}, MinQty={market_info['lotSizeFilter']['minOrderQty']}")

    # Set Leverage (Important!) - Do this before initializing strategy if strategy uses it
    # Ensure leverage is set before potentially placing orders
    set_leverage(config['symbol'], config['order']['leverage'])

    # Initialize Strategy Engine
    try:
        # Pass market info and strategy-specific parameters
        strategy_instance = VolumaticOBStrategy(market_info=market_info, **config['strategy'])
        log.info(f"Strategy '{type(strategy_instance).__name__}' initialized successfully.")
        # Log calculated precision from strategy instance
        log.info(f"Strategy using Price Precision: {strategy_instance.price_precision}, Qty Precision: {strategy_instance.qty_precision}")
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
                # Use update method for initial analysis as well
                initial_results = strategy_instance.update(df_copy_initial)
                if initial_results:
                    # Log initial state based on strategy's properties after update
                    trend_str = 'UP' if strategy_instance.current_trend else 'DOWN' if strategy_instance.current_trend is False else 'Undetermined'
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
        log.critical("Failed to start and confirm WebSocket connection. Exiting.")
        # Ensure cleanup if WS fails to start
        stop_websocket_connection() # Already called within start_websocket_connection on failure, but call again for safety
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
            # Check slightly more often than the rate limit interval to ensure it runs eventually
            if now - last_position_check_time >= POSITION_CHECK_INTERVAL:
                log.debug("Performing periodic position check (verification)...")
                # Fetch position data, result is handled internally by get_current_position (updates time, logs)
                # Result not needed here, just triggers the check and updates last_position_check_time
                get_current_position(config['symbol'])

            # 3. Sleep efficiently until the next check or stop signal
            # Wait for a short duration or until the stop_event is set
            # Timeout determines frequency of health checks etc.
            check_interval = 5.0 # Seconds - check WS health more frequently
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
             stop_event.wait(timeout=15.0) # Use interruptible wait

    # --- End of Script ---
    # Shutdown sequence is handled by handle_shutdown_signal
    # If loop exited normally (e.g. stop_event set without signal), ensure shutdown runs
    if not stop_event.is_set(): # Should not happen if loop condition is correct, but safety check
         log.warning("Main loop exited unexpectedly without stop_event set. Triggering shutdown.")
         handle_shutdown_signal(signal.SIGTERM, None) # Or SIGINT?

    log.info("Main loop terminated.")
    # Final confirmation message (might not be reached if sys.exit called in handler)
    print(Fore.MAGENTA + Style.BRIGHT + "\n~~~ Pyrmethus Volumatic+OB Trading Bot Stopped ~~~" + Style.RESET_ALL)

