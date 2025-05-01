# -*- coding: utf-8 -*-
"""
Pyrmethus Volumatic Trend + OB Trading Bot for Bybit V5 API (Enhanced)
"""
import os
import sys
import json
import time
import datetime
import logging
import signal
import threading
from decimal import Decimal, ROUND_DOWN, ROUND_UP, InvalidOperation

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from colorama import init, Fore, Style
from pybit.unified_trading import HTTP, WebSocket

# Import strategy class and type hints
from strategy import VolumaticOBStrategy, AnalysisResults

# --- Initialize Colorama ---
init(autoreset=True)

# --- Load Environment Variables ---
load_dotenv()
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
TESTNET = os.getenv("BYBIT_TESTNET", "False").lower() == "true"

# --- Global Variables ---
config = {}
session: Optional[HTTP] = None
ws: Optional[WebSocket] = None
ws_thread: Optional[threading.Thread] = None
ws_connected = threading.Event()
stop_event = threading.Event() # For graceful shutdown
latest_dataframe: Optional[pd.DataFrame] = None
strategy_instance: Optional[VolumaticOBStrategy] = None
market_info: Optional[Dict[str, Any]] = None
data_lock = threading.Lock() # Protect dataframe access
position_lock = threading.Lock() # Protect position updates/checks
order_lock = threading.Lock() # Protect order placement/closing logic
last_position_check_time = 0
POSITION_CHECK_INTERVAL = 10 # Default, overridden by config

# --- Logging Setup ---
log = logging.getLogger("PyrmethusVolumaticBot")
log_level = logging.INFO # Default, will be overridden by config

def setup_logging(level_str="INFO"):
    """Configures logging for the application."""
    global log_level
    log_level = getattr(logging, level_str.upper(), logging.INFO)
    log.setLevel(log_level)
    # Prevent duplicate handlers if re-called
    if not log.handlers:
        # Console Handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        log.addHandler(ch)
        # Optional: File Handler
        # log_filename = f"bot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        # fh = logging.FileHandler(log_filename)
        # fh.setLevel(log_level)
        # fh.setFormatter(formatter)
        # log.addHandler(fh)
    log.propagate = False # Prevent root logger from handling messages too

# --- Configuration Loading ---
def load_config(path="config.json") -> Dict:
    """Loads configuration from a JSON file."""
    try:
        with open(path, 'r') as f:
            conf = json.load(f)
            # Update global interval if set in config
            global POSITION_CHECK_INTERVAL
            POSITION_CHECK_INTERVAL = conf.get("position_check_interval", 10)
            return conf
    except FileNotFoundError:
        log.critical(f"CRITICAL: Configuration file '{path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        log.critical(f"CRITICAL: Configuration file '{path}' contains invalid JSON: {e}")
        sys.exit(1)
    except Exception as e:
        log.critical(f"CRITICAL: Error loading configuration: {e}", exc_info=True)
        sys.exit(1)

# --- Bybit Interaction ---
def connect_bybit() -> Optional[HTTP]:
    """Connects to Bybit HTTP API."""
    if not API_KEY or not API_SECRET:
        log.critical("CRITICAL: Bybit API Key or Secret not found in .env file.")
        sys.exit(1)
    try:
        log.info(f"Connecting to Bybit {'Testnet' if TESTNET else 'Mainnet'} HTTP API...")
        s = HTTP(testnet=TESTNET, api_key=API_KEY, api_secret=API_SECRET)
        # Test connection by getting server time
        server_time = s.get_server_time()
        log.info(f"Successfully connected. Server time: {datetime.datetime.fromtimestamp(int(server_time['result']['timeNano']) / 1e9)}")
        return s
    except Exception as e:
        log.critical(f"CRITICAL: Failed to connect to Bybit HTTP API: {e}", exc_info=True)
        return None

def get_market_info(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetches instrument information for the symbol."""
    if not session:
        log.error("HTTP session not available for get_market_info.")
        return None
    try:
        log.debug(f"Fetching market info for {symbol}...")
        response = session.get_instruments_info(category="linear", symbol=symbol)
        if response['retCode'] == 0 and response['result']['list']:
            info = response['result']['list'][0]
            log.info(f"Fetched market info for {symbol}.")
            log.debug(f"Market Info Details: {json.dumps(info, indent=2)}")
            # Validate required fields for precision
            if not info.get('priceFilter', {}).get('tickSize') or \
               not info.get('lotSizeFilter', {}).get('qtyStep') or \
               not info.get('lotSizeFilter', {}).get('minOrderQty') or \
               not info.get('lotSizeFilter', {}).get('maxOrderQty'):
                log.error(f"Market info for {symbol} missing required price/lot filter details (tickSize, qtyStep, min/maxOrderQty).")
                return None
            return info
        else:
            log.error(f"Failed to get market info for {symbol}: {response.get('retMsg', 'Unknown Error')} (Code: {response.get('retCode', 'N/A')})")
            return None
    except Exception as e:
        log.error(f"Exception fetching market info for {symbol}: {e}", exc_info=log_level <= logging.DEBUG)
        return None

def fetch_initial_data(symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetches historical Klines and ensures chronological order."""
    if not session:
         log.error("HTTP session not available for fetch_initial_data.")
         return None
    log.info(f"Fetching initial {limit} candles for {symbol} ({interval})...")
    try:
        response = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        if response['retCode'] == 0 and response['result']['list']:
            kline_list = response['result']['list']
            if not kline_list:
                 log.warning(f"Received empty kline list from Bybit for initial fetch.")
                 return pd.DataFrame()

            df = pd.DataFrame(kline_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            # Bybit V5 API returns klines oldest first, no need to reverse
            # df = df.iloc[::-1]
            df = df.astype({
                'timestamp': 'int64', 'open': 'float64', 'high': 'float64',
                'low': 'float64', 'close': 'float64', 'volume': 'float64', 'turnover': 'float64'
            })
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            df = df.sort_index() # Ensure sort order just in case
            log.info(f"Fetched {len(df)} initial candles. From {df.index.min()} to {df.index.max()}")
            return df
        else:
            log.error(f"Failed to fetch initial klines: {response.get('retMsg', 'Unknown Error')} (Code: {response.get('retCode', 'N/A')})")
            return None
    except Exception as e:
        log.error(f"Exception fetching initial klines: {e}", exc_info=log_level <= logging.DEBUG)
        return None

def get_current_position(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetches current position details, handling potential None session and rate limiting."""
    global last_position_check_time
    if not session:
        log.error("HTTP session not available for get_current_position.")
        return {"size": Decimal(0), "side": "None"} # Return a default "flat" state on error

    # Rate limit position checks
    now = time.time()
    if now - last_position_check_time < POSITION_CHECK_INTERVAL:
        #log.debug("Skipping position check due to rate limit.")
        return None # Indicate check was skipped, calling function should handle this

    log.debug(f"Fetching position for {symbol}...")
    # Position lock not strictly needed here as it's read-only, but good practice if state was modified
    # with position_lock:
    try:
        response = session.get_positions(category="linear", symbol=symbol)
        # Update check time regardless of success/failure to prevent spamming on errors
        last_position_check_time = time.time()

        if response['retCode'] == 0 and response['result']['list']:
            # Bybit returns a list, even when filtering by symbol. Assume first entry is correct.
            position = response['result']['list'][0]
            # Convert relevant fields to Decimal/float for calculations
            position['size'] = Decimal(position.get('size', '0'))
            position['avgPrice'] = Decimal(position.get('avgPrice', '0'))
            position['liqPrice'] = Decimal(position.get('liqPrice', '0')) if position.get('liqPrice') else Decimal(0)
            position['unrealisedPnl'] = Decimal(position.get('unrealisedPnl', '0'))
            position['curRealisedPnl'] = Decimal(position.get('curRealisedPnl', '0')) # Useful for tracking
            # IMPORTANT: Bybit V5 uses side "Buy" or "Sell", size is always positive. Side "None" means flat.
            log.debug(f"Position Data: Size={position['size']}, Side={position['side']}, AvgPrice={position['avgPrice']}")
            return position
        elif response['retCode'] == 110001: # Parameter error - likely invalid symbol if this happens late
             log.error(f"Parameter error fetching position for {symbol}. Is symbol valid? {response.get('retMsg', '')}")
             return {"size": Decimal(0), "side": "None"} # Assume flat on symbol error
        else:
            log.error(f"Failed to get position for {symbol}: {response.get('retMsg', 'Error')} (Code: {response.get('retCode', 'N/A')})")
            # On persistent errors, returning a default "flat" might be dangerous.
            # Consider stopping the bot or implementing retry logic here.
            # For now, return flat to avoid placing orders based on stale/wrong info.
            return {"size": Decimal(0), "side": "None"} # Return a default "flat" state on error
    except Exception as e:
        last_position_check_time = time.time() # Update time even on exception
        log.error(f"Exception fetching position for {symbol}: {e}", exc_info=log_level <= logging.DEBUG)
        return {"size": Decimal(0), "side": "None"} # Return a default "flat" state on error

def get_wallet_balance(account_type="UNIFIED", coin="USDT") -> Optional[Decimal]:
    """Fetches available balance (using equity as proxy for UNIFIED)."""
    if not session:
        log.error("HTTP session not available for get_wallet_balance.")
        return None
    try:
        # V5 Unified Trading uses get_wallet_balance
        response = session.get_wallet_balance(accountType=account_type, coin=coin)
        if response['retCode'] == 0 and response['result']['list']:
            balance_info = response['result']['list'][0] # Assuming only one list item for unified
            # Using account equity as the basis for risk calculation in Unified account
            if 'equity' in balance_info:
                balance = Decimal(balance_info['equity'])
                log.debug(f"Account Equity ({coin}): {balance}")
                if balance < 0: log.warning(f"Account equity is negative: {balance}")
                return balance
            else:
                log.warning(f"Could not find 'equity' field in wallet balance response for UNIFIED account.")
                # Fallback: Try totalAvailableBalance? Might not reflect full risk potential.
                if 'totalAvailableBalance' in balance_info:
                     balance = Decimal(balance_info['totalAvailableBalance'])
                     log.warning(f"Falling back to totalAvailableBalance: {balance}")
                     return balance
                return None
        else:
            log.error(f"Failed to get wallet balance: {response.get('retMsg', 'Unknown Error')} (Code: {response.get('retCode', 'N/A')})")
            return None
    except Exception as e:
        log.error(f"Exception fetching wallet balance: {e}", exc_info=log_level <= logging.DEBUG)
        return None

def calculate_order_qty(entry_price: float, sl_price: float, risk_percent: float) -> Optional[float]:
    """Calculates order quantity based on risk percentage, SL distance, and equity."""
    if not market_info or not strategy_instance:
        log.error("Market info or strategy instance not available for qty calculation.")
        return None
    if abs(Decimal(str(sl_price)) - Decimal(str(entry_price))) < strategy_instance.tick_size:
        log.error(f"Stop loss price {sl_price} is too close to entry price {entry_price}.")
        return None

    balance = get_wallet_balance()
    if balance is None or balance <= 0:
        log.error(f"Cannot calculate order quantity: Invalid balance ({balance}).")
        return None

    try:
        risk_amount = balance * (Decimal(str(risk_percent)) / 100)
        sl_distance = abs(Decimal(str(entry_price)) - Decimal(str(sl_price)))

        if sl_distance == 0:
             log.error("Stop loss distance is zero.")
             return None

        # For Linear contracts (XXX/USDT), PnL is in USDT.
        # Loss per contract = Qty (in Base) * SL_Distance (in Quote)
        # We want: Qty * SL_Distance <= Risk Amount
        # Qty (in Base Asset, e.g., BTC) = Risk Amount / SL_Distance
        qty_base = risk_amount / sl_distance

    except (InvalidOperation, ZeroDivisionError, TypeError) as e:
        log.error(f"Error during quantity calculation math: {e}")
        return None

    # Round down to the minimum quantity step
    qty_rounded = strategy_instance.round_qty(float(qty_base))

    # Check against min/max order quantity from market_info
    min_qty = float(market_info['lotSizeFilter']['minOrderQty'])
    max_qty = float(market_info['lotSizeFilter']['maxOrderQty'])

    if qty_rounded < min_qty:
        log.warning(f"Calculated quantity {qty_rounded} is below minimum ({min_qty}). Risking more than planned.")
        # Decide: Use min_qty (higher risk) or skip trade? For now, use min_qty.
        qty_final = min_qty
        # Recalculate actual risk if using min_qty
        actual_risk_amount = Decimal(str(min_qty)) * sl_distance
        actual_risk_percent = (actual_risk_amount / balance) * 100 if balance > 0 else 0
        log.warning(f"Using min qty {min_qty}. Actual Risk: {actual_risk_amount:.2f} USDT ({actual_risk_percent:.2f}%)")

    elif qty_rounded > max_qty:
        log.warning(f"Calculated quantity {qty_rounded} exceeds maximum ({max_qty}). Using maximum.")
        qty_final = max_qty
    else:
        qty_final = qty_rounded

    if qty_final <= 0:
        log.error(f"Final calculated quantity is zero or negative ({qty_final}). Cannot place order.")
        return None

    log.info(f"Calculated Order Qty: {qty_final:.{strategy_instance.qty_precision}f} "
             f"(Balance={balance:.2f}, Risk={risk_percent}%, RiskAmt={risk_amount:.2f}, SLDist={sl_distance:.{strategy_instance.price_precision}f})")
    return qty_final

def place_order(symbol: str, side: str, qty: float, price: Optional[float]=None, sl_price: Optional[float]=None, tp_price: Optional[float]=None) -> Optional[Dict]:
    """Places an order via Bybit API with SL/TP."""
    if not session or not strategy_instance: return None
    if config.get("mode", "Live").lower() == "paper":
        log.warning(f"[PAPER MODE] Would place {side} order for {qty} {symbol} at ~{price} SL={sl_price} TP={tp_price}")
        # Simulate success for paper trading state management
        return {"retCode": 0, "retMsg": "OK", "result": {"orderId": f"paper_{int(time.time())}"}, "paperTrade": True}

    with order_lock: # Ensure only one order placement happens at a time
        order_type = config['order']['type']
        # Ensure quantity is correctly rounded *before* placing order
        qty_rounded = strategy_instance.round_qty(qty)
        if qty_rounded <= 0:
             log.error(f"Attempted to place order with zero or negative rounded quantity ({qty_rounded}). Original qty: {qty}.")
             return None
        min_qty = float(market_info['lotSizeFilter']['minOrderQty'])
        if qty_rounded < min_qty:
            log.warning(f"Final quantity {qty_rounded} is less than min qty {min_qty}. Using min qty.")
            qty_rounded = min_qty

        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side, # "Buy" or "Sell"
            "orderType": order_type,
            "qty": str(qty_rounded), # API requires string quantity
            "timeInForce": "GTC", # GoodTillCancel is common
            "reduceOnly": False, # This is an entry order
            "positionIdx": 0 # Required for one-way mode
        }

        if order_type == "Limit" and price:
            limit_price_rounded = strategy_instance.round_price(price)
            params["price"] = str(limit_price_rounded)
        elif order_type == "Limit" and not price:
            log.error("Limit order requires a price.")
            return None

        # Add SL/TP using Bybit's parameters
        if sl_price:
            sl_price_rounded = strategy_instance.round_price(sl_price)
            # Check validity (SL should not cross entry for market order logic)
            if (side == "Buy" and sl_price_rounded >= (price or 9999999)) or \
               (side == "Sell" and sl_price_rounded <= (price or 0)):
                log.error(f"Invalid SL price {sl_price_rounded} for {side} order at price {price}. SL skipped.")
            else:
                params["stopLoss"] = str(sl_price_rounded)
                log.info(f"Setting StopLoss at: {sl_price_rounded}")

        if tp_price:
            tp_price_rounded = strategy_instance.round_price(tp_price)
             # Check validity (TP should be profitable relative to entry)
            if (side == "Buy" and tp_price_rounded <= (price or 0)) or \
               (side == "Sell" and tp_price_rounded >= (price or 9999999)):
                log.error(f"Invalid TP price {tp_price_rounded} for {side} order at price {price}. TP skipped.")
            else:
                params["takeProfit"] = str(tp_price_rounded)
                log.info(f"Setting TakeProfit at: {tp_price_rounded}")

        log.warning(f"Placing {side} {order_type} order: {params['qty']} {symbol} "
                    f"{'@'+str(params.get('price')) if 'price' in params else ''} "
                    f"SL={params.get('stopLoss', 'N/A')} TP={params.get('takeProfit', 'N/A')}")
        try:
            response = session.place_order(**params)
            log.debug(f"Place Order Response: {response}")
            if response['retCode'] == 0:
                order_id = response['result'].get('orderId')
                log.info(f"{Fore.GREEN}Order placed successfully! OrderID: {order_id}{Style.RESET_ALL}")
                # Optional: Store order ID for tracking
                return response
            else:
                # Log specific errors
                error_msg = response.get('retMsg', 'Unknown Error')
                error_code = response.get('retCode', 'N/A')
                log.error(f"{Fore.RED}Failed to place order: {error_msg} (Code: {error_code}){Style.RESET_ALL}")
                # Handle common errors explicitly if needed (e.g., margin, position mode)
                if error_code == 110007: log.error("Insufficient margin error. Check balance and leverage.")
                if "position mode not modified" in error_msg: log.error("Position mode mismatch? Ensure Bybit is in One-Way mode for this symbol.")
                return response # Return error response
        except Exception as e:
            log.error(f"Exception placing order: {e}", exc_info=log_level <= logging.DEBUG)
            return None

def close_position(symbol: str, position_data: Dict) -> Optional[Dict]:
    """Closes an existing position using a reduce-only market order."""
    if not session or not strategy_instance: return None
    if config.get("mode", "Live").lower() == "paper":
        log.warning(f"[PAPER MODE] Would close position for {symbol} (Size: {position_data['size']}, Side: {position_data['side']})")
        # Simulate success
        return {"retCode": 0, "retMsg": "OK", "result": {"orderId": f"paper_close_{int(time.time())}"}, "paperTrade": True}

    with order_lock: # Ensure only one closing order happens at a time
        side_to_close = "Buy" if position_data['side'] == "Sell" else "Sell"
        qty_to_close = float(position_data['size']) # Close the full size fetched

        if qty_to_close <= 0:
             log.warning(f"Attempting to close position for {symbol}, but fetched size is {qty_to_close}. Assuming already flat.")
             return {"retCode": 0, "retMsg": "OK", "result": {}, "alreadyFlat": True}


        log.warning(f"Closing {position_data['side']} position for {symbol} (Qty: {qty_to_close}). Placing {side_to_close} Market order...")
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side_to_close,
            "orderType": "Market", # Use Market order to ensure closure
            "qty": str(qty_to_close),
            "reduceOnly": True, # Ensure this only closes position
            "positionIdx": 0 # Required for one-way mode
        }
        try:
            # Cancel existing SL/TP orders before closing, if possible/necessary
            # response_cancel = session.cancel_all_orders(category="linear", symbol=symbol, orderFilter="StopOrder")
            # log.debug(f"Cancel SL/TP Response: {response_cancel}")
            # time.sleep(0.5) # Brief pause after cancellation

            response = session.place_order(**params)
            log.debug(f"Close Position Response: {response}")
            if response['retCode'] == 0:
                order_id = response['result'].get('orderId')
                log.info(f"{Fore.YELLOW}Position close order placed successfully! OrderID: {order_id}{Style.RESET_ALL}")
                return response
            else:
                error_msg = response.get('retMsg', 'Unknown Error')
                error_code = response.get('retCode', 'N/A')
                log.error(f"{Fore.RED}Failed to place close order: {error_msg} (Code: {error_code}){Style.RESET_ALL}")
                # Handle reduce-only error (means position likely changed or closed already)
                if error_code in [110043, 3400070]: # reduce-only quantity error codes
                    log.warning("Reduce-only error suggests position size changed or already zero. Re-checking position on next cycle.")
                    global last_position_check_time # Force re-check sooner
                    last_position_check_time = 0
                return response # Return error response
        except Exception as e:
            log.error(f"Exception closing position: {e}", exc_info=log_level <= logging.DEBUG)
            return None

def set_leverage(symbol: str, leverage: int):
    """Sets leverage for the specified symbol."""
    if not session:
        log.error("HTTP session not available for set_leverage.")
        return
    # Ensure leverage is within Bybit's typical limits (e.g., 1-100)
    if not 1 <= leverage <= 100:
         log.error(f"Invalid leverage value: {leverage}. Must be between 1 and 100.")
         return

    log.info(f"Setting leverage for {symbol} to {leverage}x...")
    try:
        response = session.set_leverage(
            category="linear",
            symbol=symbol,
            buyLeverage=str(float(leverage)), # API expects string representation of float/int
            sellLeverage=str(float(leverage))
        )
        log.debug(f"Set Leverage Response: {response}")
        if response['retCode'] == 0:
            log.info(f"Leverage for {symbol} set to {leverage}x successfully.")
        else:
            # Common error: 110044 means leverage not modified (already set)
            if response['retCode'] == 110044:
                 log.warning(f"Leverage for {symbol} already set to {leverage}x (Not modified).")
            else:
                 log.error(f"Failed to set leverage for {symbol}: {response.get('retMsg', 'Unknown Error')} (Code: {response.get('retCode', 'N/A')})")
    except Exception as e:
        log.error(f"Exception setting leverage: {e}", exc_info=log_level <= logging.DEBUG)

# --- WebSocket Handling ---
def handle_ws_message(msg):
    """Callback function to process incoming WebSocket messages."""
    # log.debug(f"WS Recv: {msg}") # Too verbose for normal operation
    global latest_dataframe
    if stop_event.is_set(): return # Don't process if stopping

    topic = msg.get("topic", "")
    data = msg.get("data", [])

    # Handle Kline Updates
    if topic.startswith(f"kline.{config['interval']}.{config['symbol']}"):
        if not data: return
        kline_item = data[0] # Bybit V5 Kline usually sends one item per push
        if not kline_item.get('confirm', False):
            # log.debug("Ignoring unconfirmed kline update.")
            return # Only process confirmed (closed) candles

        try:
            ts_ms = int(kline_item['start'])
            ts = pd.to_datetime(ts_ms, unit='ms')
            # Check if this candle is already processed (can happen on reconnect)
            with data_lock:
                 if latest_dataframe is not None and ts in latest_dataframe.index:
                      # log.debug(f"Ignoring already processed candle: {ts}")
                      return

            log.debug(f"Confirmed Kline received: T={ts} C={kline_item['close']}")

            new_data = {
                'open': float(kline_item['open']),
                'high': float(kline_item['high']),
                'low': float(kline_item['low']),
                'close': float(kline_item['close']),
                'volume': float(kline_item['volume']),
                'turnover': float(kline_item.get('turnover', 0.0))
            }
        except (KeyError, ValueError) as e:
            log.error(f"Error parsing kline data: {e} - Data: {kline_item}")
            return

        with data_lock:
            if latest_dataframe is None:
                log.warning("DataFrame not ready, skipping WS kline processing.")
                return

            # Create new row DataFrame
            new_row = pd.DataFrame([new_data], index=[ts])

            # Append (already checked if exists)
            latest_dataframe = pd.concat([latest_dataframe, new_row])
            # Prune
            if len(latest_dataframe) > config['data']['max_df_len']:
                latest_dataframe = latest_dataframe.iloc[-(config['data']['max_df_len']):]
                log.debug(f"DataFrame pruned to {len(latest_dataframe)} rows.")

            # --- Trigger Analysis ---
            if strategy_instance:
                log.info(f"Running analysis on new confirmed candle: {ts}")
                # Create a copy for analysis to avoid modification during processing
                df_copy = latest_dataframe.copy()
                try:
                    analysis_results = strategy_instance.update(df_copy)
                    process_signals(analysis_results)
                except Exception as e:
                    log.error(f"Error during strategy analysis: {e}", exc_info=True) # Log traceback

    # Handle Position Updates (Optional but recommended for faster state awareness)
    elif topic.startswith("position"):
         if data:
             for pos_update in data:
                 if pos_update.get('symbol') == config['symbol']:
                      log.info(f"{Fore.CYAN}Position update via WS: {pos_update}{Style.RESET_ALL}")
                      # OPTIONAL: Update internal state faster than periodic check
                      # Be very careful with locking and ensuring consistency if doing this
                      # Force a check on next cycle might be safer:
                      # global last_position_check_time
                      # last_position_check_time = 0
                      pass

    # Handle Order Updates (Optional)
    elif topic.startswith("order"):
        if data:
             for order_update in data:
                 if order_update.get('symbol') == config['symbol']:
                     log.info(f"{Fore.CYAN}Order update via WS: Status={order_update.get('orderStatus')}, ID={order_update.get('orderId')}{Style.RESET_ALL}")
                     # Can track fills, cancellations, SL/TP triggers here
                     pass

    # Handle Connection Status / Auth
    elif msg.get("op") == "auth":
        if msg.get("success"):
            log.info(f"{Fore.GREEN}WebSocket authenticated successfully.{Style.RESET_ALL}")
        else:
            log.error(f"WebSocket authentication failed: {msg}")
    elif msg.get("op") == "subscribe":
         if msg.get("success"):
            log.info(f"{Fore.GREEN}WebSocket subscribed successfully to: {msg.get('ret_msg')}{Style.RESET_ALL}")
         else:
            log.error(f"WebSocket subscription failed: {msg}")
    elif msg.get("op") == "pong":
        log.debug("WebSocket Pong received")

def start_websocket_connection():
    """Initializes and starts the WebSocket connection for private and public data."""
    global ws, ws_thread, ws_connected
    if not API_KEY or not API_SECRET:
        log.error("Cannot start WebSocket: API credentials missing.")
        return False
    if ws_thread and ws_thread.is_alive():
        log.warning("WebSocket thread already running.")
        return True

    log.info("Starting WebSocket connection...")
    ws_connected.clear()
    try:
        # Need private channel type for positions/orders
        ws = WebSocket(testnet=TESTNET, channel_type="private", api_key=API_KEY, api_secret=API_SECRET)

        # --- Define Subscriptions ---
        # Public Kline Topic
        kline_topic = f"kline.{config['interval']}.{config['symbol']}"
        # Private Topics
        position_topic = f"position.{config['symbol']}" # Specific symbol for positions
        order_topic = f"order" # Listen to all orders for the account

        topics = [kline_topic, position_topic, order_topic]

        # Register the message handler
        ws.websocket_data.add_handler(handle_ws_message)

        # Subscribe to topics
        ws.subscribe(topics)

        # Start the WebSocket connection loop in a separate thread
        ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
        ws_thread.start()

        # Give it a moment to establish connection and subscribe
        # A more robust check would involve waiting for subscription confirmation messages
        log.info("Waiting for WebSocket connection and initial messages...")
        time.sleep(5) # Simple wait, adjust as needed

        if ws_thread.is_alive():
            log.info(f"{Fore.GREEN}WebSocket thread started. Monitoring topics: {topics}{Style.RESET_ALL}")
            ws_connected.set() # Signal connection attempt is done
            return True
        else:
             log.error("WebSocket thread failed to start.")
             return False

    except Exception as e:
        log.critical(f"CRITICAL: Failed to initialize or start WebSocket: {e}", exc_info=True)
        return False

def stop_websocket_connection():
    """Stops the WebSocket connection gracefully."""
    global ws, ws_thread, ws_connected
    if ws:
        log.info("Stopping WebSocket connection...")
        try:
            # Unsubscribe first? Optional.
            # ws.unsubscribe([...])
            ws.exit()
        except Exception as e:
            log.error(f"Error closing WebSocket: {e}")
    else:
         log.info("WebSocket object not found, cannot stop.")


    if ws_thread and ws_thread.is_alive():
        log.info("Waiting for WebSocket thread to join...")
        ws_thread.join(timeout=10)
        if ws_thread.is_alive():
            log.warning("WebSocket thread did not stop gracefully after 10 seconds.")
        else:
            log.info("WebSocket thread joined.")
    elif ws_thread:
         log.info("WebSocket thread was already stopped.")

    ws = None
    ws_thread = None
    ws_connected.clear()
    log.info("WebSocket stopped.")

# --- Signal Processing & Execution ---
def process_signals(results: AnalysisResults):
    """Processes the strategy signals and decides on trade actions."""
    if not results or not strategy_instance or not market_info:
        log.warning("Signal processing skipped: Missing results, strategy instance, or market info.")
        return
    if stop_event.is_set():
        log.warning("Signal processing skipped: Stop event is set.")
        return

    signal = results['last_signal']
    last_close = results['last_close']
    last_atr = results['last_atr']
    symbol = config['symbol']

    log.debug(f"Processing Signal: {signal}, Last Close: {last_close}, Last ATR: {last_atr}")

    if pd.isna(last_close):
        log.warning("Cannot process signal: Last close price is NaN.")
        return

    # --- Get Current Position ---
    # This is crucial to avoid duplicate orders and manage exits correctly
    # Rate limiting is handled inside get_current_position
    position_data = get_current_position(symbol)
    if position_data is None:
        log.warning("Position check skipped due to rate limit. Re-evaluating on next candle.")
        return # Wait for next candle if check was skipped
    if not position_data: # Check if position fetch actually failed (returned Falsy)
        log.error("Could not get current position data due to error. Skipping signal processing cycle.")
        return

    # Ensure size is Decimal for comparison
    current_pos_size = position_data.get('size', Decimal(0))
    current_pos_side = position_data.get('side', 'None') # 'Buy', 'Sell', or 'None'

    is_long = current_pos_side == 'Buy' and current_pos_size > 0
    is_short = current_pos_side == 'Sell' and current_pos_size > 0 # Bybit V5 size is positive for short too
    is_flat = not is_long and not is_short

    log.info(f"Current Position: {'Long' if is_long else 'Short' if is_short else 'Flat'} (Size: {current_pos_size})")

    # --- Signal Actions ---
    tp_ratio = Decimal(str(config['order'].get('tp_ratio', 2.0))) # Default R:R of 2 if not set

    # BUY Signal
    if signal == "BUY" and is_flat:
        log.warning(f"{Fore.GREEN}{Style.BRIGHT}BUY Signal Received - Entering Long.{Style.RESET_ALL}")
        # Calculate Stop Loss
        sl_price_raw = None
        sl_method = config['strategy']['stop_loss']['method']
        if sl_method == "ATR" and last_atr:
            sl_multiplier = float(config['strategy']['stop_loss']['atr_multiplier'])
            sl_price_raw = last_close - (last_atr * sl_multiplier)
        elif sl_method == "OB":
             # Find the lowest point of active bull OBs below current price
             relevant_obs = [b for b in results['active_bull_boxes'] if b['bottom'] < last_close]
             if relevant_obs:
                  lowest_bottom = min(b['bottom'] for b in relevant_obs)
                  # SL slightly below the lowest relevant OB bottom
                  sl_buffer = (last_atr * 0.1) if last_atr else (last_close * 0.001) # Small buffer
                  sl_price_raw = lowest_bottom - sl_buffer
             else:
                  log.warning("OB SL method chosen but no relevant Bull OB found below price. Falling back to ATR.")
                  if last_atr: sl_price_raw = last_close - (last_atr * 2.0) # Fallback ATR SL

        if sl_price_raw is None:
             log.error("Stop Loss price could not be calculated. Cannot place BUY order.")
             return

        # Ensure SL is below entry
        if sl_price_raw >= last_close:
             log.error(f"Calculated SL price {sl_price_raw} is not below entry price {last_close}. Using fallback ATR SL.")
             if last_atr: sl_price_raw = last_close - (last_atr * 2.0) # Fallback ATR SL
             else: log.error("Cannot set SL: Fallback ATR is unavailable."); return

        sl_price = strategy_instance.round_price(sl_price_raw)
        log.info(f"Calculated SL price for BUY: {sl_price}")

        # Calculate Take Profit
        sl_distance = Decimal(str(last_close)) - Decimal(str(sl_price))
        tp_price_raw = float(Decimal(str(last_close)) + (sl_distance * tp_ratio)) if sl_distance > 0 else None
        tp_price = strategy_instance.round_price(tp_price_raw) if tp_price_raw else None
        log.info(f"Calculated TP price for BUY: {tp_price} (Ratio: {tp_ratio})")

        # Calculate Order Quantity
        qty = calculate_order_qty(last_close, sl_price, config['order']['risk_per_trade_percent'])
        if qty and qty > 0:
            place_order(symbol, "Buy", qty, price=last_close if config['order']['type'] == "Limit" else None, sl_price=sl_price, tp_price=tp_price)
        else:
            log.error("Order quantity calculation failed or resulted in zero/negative. Cannot place BUY order.")

    # SELL Signal
    elif signal == "SELL" and is_flat:
        log.warning(f"{Fore.RED}{Style.BRIGHT}SELL Signal Received - Entering Short.{Style.RESET_ALL}")
        # Calculate Stop Loss
        sl_price_raw = None
        sl_method = config['strategy']['stop_loss']['method']
        if sl_method == "ATR" and last_atr:
             sl_multiplier = float(config['strategy']['stop_loss']['atr_multiplier'])
             sl_price_raw = last_close + (last_atr * sl_multiplier)
        elif sl_method == "OB":
            # Find the highest point of active bear OBs above current price
            relevant_obs = [b for b in results['active_bear_boxes'] if b['top'] > last_close]
            if relevant_obs:
                  highest_top = max(b['top'] for b in relevant_obs)
                  # SL slightly above the highest relevant OB top
                  sl_buffer = (last_atr * 0.1) if last_atr else (last_close * 0.001) # Small buffer
                  sl_price_raw = highest_top + sl_buffer
            else:
                  log.warning("OB SL method chosen but no relevant Bear OB found above price. Falling back to ATR.")
                  if last_atr: sl_price_raw = last_close + (last_atr * 2.0) # Fallback ATR SL

        if sl_price_raw is None:
             log.error("Stop Loss price could not be calculated. Cannot place SELL order.")
             return

        # Ensure SL is above entry
        if sl_price_raw <= last_close:
             log.error(f"Calculated SL price {sl_price_raw} is not above entry price {last_close}. Using fallback ATR SL.")
             if last_atr: sl_price_raw = last_close + (last_atr * 2.0) # Fallback ATR SL
             else: log.error("Cannot set SL: Fallback ATR is unavailable."); return

        sl_price = strategy_instance.round_price(sl_price_raw)
        log.info(f"Calculated SL price for SELL: {sl_price}")

        # Calculate Take Profit
        sl_distance = Decimal(str(sl_price)) - Decimal(str(last_close))
        tp_price_raw = float(Decimal(str(last_close)) - (sl_distance * tp_ratio)) if sl_distance > 0 else None
        tp_price = strategy_instance.round_price(tp_price_raw) if tp_price_raw else None
        log.info(f"Calculated TP price for SELL: {tp_price} (Ratio: {tp_ratio})")

        # Calculate Order Quantity
        qty = calculate_order_qty(last_close, sl_price, config['order']['risk_per_trade_percent'])
        if qty and qty > 0:
            place_order(symbol, "Sell", qty, price=last_close if config['order']['type'] == "Limit" else None, sl_price=sl_price, tp_price=tp_price)
        else:
            log.error("Order quantity calculation failed or resulted in zero/negative. Cannot place SELL order.")

    # EXIT_LONG Signal
    elif signal == "EXIT_LONG" and is_long:
        log.warning(f"{Fore.YELLOW}{Style.BRIGHT}EXIT_LONG Signal Received - Closing Long Position.{Style.RESET_ALL}")
        close_position(symbol, position_data)

    # EXIT_SHORT Signal
    elif signal == "EXIT_SHORT" and is_short:
        log.warning(f"{Fore.YELLOW}{Style.BRIGHT}EXIT_SHORT Signal Received - Closing Short Position.{Style.RESET_ALL}")
        close_position(symbol, position_data)

    # HOLD Signal or signal matches current state -> Do Nothing
    elif signal == "HOLD":
        log.debug("HOLD Signal - No action.")
    elif signal == "BUY" and is_long:
        log.debug("BUY Signal - Already Long.")
    elif signal == "SELL" and is_short:
        log.debug("SELL Signal - Already Short.")
    elif signal == "EXIT_LONG" and not is_long:
         log.debug("EXIT_LONG Signal - Not Long.")
    elif signal == "EXIT_SHORT" and not is_short:
         log.debug("EXIT_SHORT Signal - Not Short.")


# --- Graceful Shutdown ---
def handle_shutdown_signal(signum, frame):
    """Handles termination signals like SIGINT (Ctrl+C) and SIGTERM."""
    if stop_event.is_set(): # Avoid running multiple times
         return
    log.warning(f"Shutdown signal {signal.Signals(signum).name} received. Initiating graceful shutdown...")
    stop_event.set() # Signal loops and threads to stop
    # Stop WS first to prevent processing more data
    stop_websocket_connection()

    # --- Optional: Add logic here to manage open positions/orders on shutdown ---
    # Example: Close open position? (Use with caution!)
    # close_on_exit = False # Make this configurable
    # if close_on_exit:
    #     log.warning("Attempting to close open position on exit...")
    #     pos_data = get_current_position(config['symbol'])
    #     # Need to handle the 'None' return from rate limited check here
    #     while pos_data is None: # Retry if rate limited
    #          log.info("Waiting briefly to re-check position for exit closure...")
    #          time.sleep(min(POSITION_CHECK_INTERVAL, 3)) # Wait a bit
    #          pos_data = get_current_position(config['symbol'])

    #     if pos_data and pos_data.get('size', Decimal(0)) > 0:
    #          close_position(config['symbol'], pos_data)
    #     else:
    #          log.info("No open position found to close on exit.")

    log.info("Shutdown complete. Exiting.")
    # Give logs a moment to flush
    time.sleep(1)
    sys.exit(0)

# --- Main Execution ---
if __name__ == "__main__":
    print(Fore.MAGENTA + Style.BRIGHT + "\n~~~ Pyrmethus Volumatic+OB Trading Bot Starting ~~~")

    # Load configuration
    config = load_config()
    setup_logging(config.get("log_level", "INFO"))
    log.info(f"Configuration loaded from config.json")
    log.debug(f"Config: {json.dumps(config, indent=2)}")

    # Register shutdown handlers
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)

    # Connect to Bybit
    session = connect_bybit()
    if not session:
        sys.exit(1) # Critical error, already logged

    # Get Market Info
    market_info = get_market_info(config['symbol'])
    if not market_info:
        log.critical("Could not retrieve market info. Check symbol and connection. Exiting.")
        sys.exit(1)

    # Set Leverage (Important!)
    set_leverage(config['symbol'], config['order']['leverage'])

    # Initialize Strategy
    try:
        strategy_instance = VolumaticOBStrategy(market_info=market_info, **config['strategy']['params'])
    except (ValueError, KeyError, TypeError) as e:
        log.critical(f"Failed to initialize strategy: {e}. Check config.json strategy params. Exiting.", exc_info=True)
        sys.exit(1)

    # Fetch Initial Data
    with data_lock:
        latest_dataframe = fetch_initial_data(
            config['symbol'],
            config['interval'],
            config['data']['fetch_limit']
        )

    if latest_dataframe is None: # Check for None specifically (indicates fetch error)
        log.critical("Failed to fetch initial data. Exiting.")
        sys.exit(1)
    if latest_dataframe.empty:
        log.warning("Fetched initial data but the DataFrame is empty. Check symbol/interval on Bybit. Will attempt to continue with WS.")
        # Allow continuing, WS might populate data, but strategy needs min_data_len
    elif len(latest_dataframe) < strategy_instance.min_data_len:
         log.warning(f"Fetched data ({len(latest_dataframe)}) is less than minimum required ({strategy_instance.min_data_len}). Strategy may need more data from WS.")
         # Allow continuing

    # Run Initial Analysis (only if enough data exists)
    if latest_dataframe is not None and len(latest_dataframe) >= strategy_instance.min_data_len:
        log.info("Running initial analysis on historical data...")
        with data_lock:
            initial_results = strategy_instance.update(latest_dataframe.copy())
            # Display initial state but don't trade yet
            log.info(f"Initial Analysis: Trend={initial_results['current_trend']}, Signal={initial_results['last_signal']}")
    else:
         log.info("Skipping initial analysis due to insufficient historical data.")


    # Start WebSocket
    if not start_websocket_connection():
        log.critical("Failed to start WebSocket connection. Exiting.")
        sys.exit(1)

    log.info(f"{Fore.CYAN}Bot is running for {config['symbol']} ({config['interval']}). Waiting for confirmed candle updates...{Style.RESET_ALL}")
    log.info(f"Trading Mode: {config['mode']}")
    log.info("Press Ctrl+C to stop.")

    # Main loop for periodic checks and keeping the script alive
    while not stop_event.is_set():
        try:
            # Periodically check WebSocket health (basic check on thread)
            if ws_thread and not ws_thread.is_alive():
                log.error("WebSocket thread appears to have died unexpectedly! Attempting restart...")
                stop_websocket_connection() # Clean up first
                time.sleep(10) # Wait before restarting
                if not stop_event.is_set():
                    if not start_websocket_connection():
                        log.critical("Failed to restart WebSocket after failure. Stopping bot.")
                        stop_event.set() # Trigger shutdown
                    else:
                         log.info("WebSocket connection restarted successfully.")

            # Periodically ensure position data is recent (in case WS position updates fail)
            now = time.time()
            # Check slightly more often than the interval itself to ensure freshness
            if now - last_position_check_time > POSITION_CHECK_INTERVAL:
                log.debug("Periodic position check triggered.")
                get_current_position(config['symbol']) # Force check (result ignored here, just updates time/logs)

            # Sleep until the next check or stop signal
            stop_event.wait(timeout=5) # Sleep efficiently, checking stop_event periodically

        except Exception as e:
             log.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
             log.warning("Attempting to continue, but consider investigating the error.")
             time.sleep(10) # Wait a bit longer after an unexpected error

    # Final cleanup (already handled by shutdown handler, but good practice)
    log.info("Main loop finished.")
    if ws_connected.is_set():
        stop_websocket_connection()
    log.info("Pyrmethus Bot has stopped.")
