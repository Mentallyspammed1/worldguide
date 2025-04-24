import os
import time
import logging
import json
import threading
import numpy as np  # Keep numpy for potential future calculations if needed
from collections import deque
from dotenv import load_dotenv
from pybit import HTTP, WebSocket

# --- Constants & Configuration ---
# Load environment variables
load_dotenv()

# API Credentials & Settings
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
USE_TESTNET = os.getenv("BYBIT_TESTNET", "True").lower() == "true"
SYMBOL = "BTCUSDT"

# Trading Parameters
ORDER_SIZE_USDT = 15      # Size of your counter-trade in USDT
MIN_ORDER_QTY = 0.001     # Check Bybit for the symbol's minimum order quantity
TICK_SIZE = 0.5           # Price precision step
ORDER_QTY_PRECISION = 3   # Number of decimal places for order quantity

# Strategy Parameters
MIN_LIQUIDATION_QTY = 0.5  # Minimum liquidation size (in base asset) to react to
ENTRY_PRICE_OFFSET_TICKS = 2  # How many ticks away from best bid/ask for limit entry
TAKE_PROFIT_TICKS = 10    # Ticks profit target for TP order
STOP_LOSS_TICKS = 8       # Ticks loss for server-side SL order
POSITION_TIMEOUT_SECONDS = 20  # Max time to hold position if TP not hit
COOLDOWN_SECONDS = 10     # Wait time after a trade closes before new entry attempt
# Choose SL trigger: 'LastPrice', 'MarkPrice', 'IndexPrice'
STOP_LOSS_TRIGGER_METHOD = "LastPrice"

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
# Suppress excessively verbose pybit websocket logging if desired
# logging.getLogger('websockets').setLevel(logging.WARNING)

# --- Global State Variables ---
# Order Book Data
latest_order_book = {'bids': [], 'asks': []}
last_best_bid = None
last_best_ask = None

# Position & Order State
in_position = False
current_position_side = None  # 'Buy' or 'Sell'
active_order_id = None      # ID of the entry order OR the TP order
entry_price = None
# stop_loss_trigger_price = None # REMOVED - Handled by server-side SL
position_entry_time = None
last_trade_exit_time = 0
# Store position size accurately after fill
position_size = 0.0

# Thread Locks
order_book_lock = threading.Lock()
position_lock = threading.Lock()  # Lock for modifying position state variables

# --- Bybit API Initialization ---
try:
    session = HTTP(
        testnet=USE_TESTNET,
        api_key=API_KEY,
        api_secret=API_SECRET,
        recv_window=10000
    )
    # Check connection
    server_time = session.get_server_time()
    logging.info(
        f"HTTP session initialized. Testnet: {USE_TESTNET}. "
        f"Server Time: {server_time.get('timeNano', 'N/A')}"
    )
except Exception as e:
    logging.exception("Fatal Error: Failed to initialize Bybit HTTP session.")
    exit()

# --- WebSocket Functions ---

def handle_public_message(msg):
    """Processes public WebSocket messages (Order Book, Liquidations)."""
    global latest_order_book, last_best_bid, last_best_ask

    topic = msg.get('topic', '')

    # Handle Order Book Updates
    if topic.startswith('orderbook.50.'):
        if msg.get('data'):
            with order_book_lock:
                # Simplified update logic (can be prone to missed deltas on reconnect)
                # Consider a more robust order book implementation if needed
                data = msg.get('data')
                type = msg.get('type')
                if type == 'snapshot':
                    latest_order_book['bids'] = [
                        [float(p), float(q)] for p, q in data.get('b', [])
                    ]
                    latest_order_book['asks'] = [
                        [float(p), float(q)] for p, q in data.get('a', [])
                    ]
                elif type == 'delta':
                     # Basic delta handling (less robust than full merge logic)
                    for side, book in [('b', 'bids'), ('a', 'asks')]:
                        for price_str, qty_str in data.get(side, []):
                            price, qty = float(price_str), float(qty_str)
                            # Remove existing
                            latest_order_book[book] = [
                                lvl for lvl in latest_order_book[book]
                                if lvl[0] != price
                            ]
                            if qty > 0:  # Add if quantity > 0
                                latest_order_book[book].append([price, qty])
                    # Re-sort after delta
                    latest_order_book['bids'].sort(key=lambda x: x[0], reverse=True)
                    latest_order_book['asks'].sort(key=lambda x: x[0])

                # Update best bid/ask
                last_best_bid = (latest_order_book['bids'][0][0]
                                 if latest_order_book['bids'] else None)
                last_best_ask = (latest_order_book['asks'][0][0]
                                 if latest_order_book['asks'] else None)
        # logging.debug(f"Order book updated. Best Bid: {last_best_bid}, Best Ask: {last_best_ask}")

    # Handle Liquidation Updates
    elif topic.startswith('liquidations.'):
        data_list = msg.get('data', [])
        for liq in data_list:
            try:
                # 'Buy' side means SHORT liq, 'Sell' side means LONG liq
                liq_side = liq.get('side')
                liq_price_str = liq.get('price')
                liq_qty_str = liq.get('qty')
                liq_time_ms = liq.get('time')

                if not all([liq_side, liq_price_str, liq_qty_str]):
                    logging.warning(f"Incomplete liquidation data: {liq}")
                    continue

                liq_price = float(liq_price_str)
                liq_qty = float(liq_qty_str)

                logging.info(
                    f"LIQUIDATION DETECTED: Side={liq_side}, Qty={liq_qty:.4f}, "
                    f"Price={liq_price:.2f}, Time={liq_time_ms}"
                )

                # Trigger the entry logic based on the liquidation event
                handle_liquidation_event(liq_side, liq_qty)

            except Exception as e:
                logging.exception(f"Error processing liquidation data: {liq}")


def handle_private_message(msg):
    """Processes private WebSocket messages (Order Updates)."""
    global in_position, active_order_id, entry_price, position_entry_time
    global last_trade_exit_time, current_position_side, position_size

    # Check if it's an order update message
    # Bybit v5 format: {'topic': 'order', 'creationTime': 1677720000000, 'data': [...] }
    if msg.get('topic') == 'order':
        order_updates = msg.get('data', [])
        for order_info in order_updates:
            order_id = order_info.get('orderId')
            status = order_info.get('orderStatus')
            side = order_info.get('side') # 'Buy' or 'Sell'
            filled_qty_str = order_info.get('cumExecQty', '0')
            avg_price_str = order_info.get('avgPrice', '0')
            reduce_only = order_info.get('reduceOnly', False)

            logging.debug(f"Private WS Order Update: {order_info}")

            with position_lock:
                # --- Handle Entry Order Fill ---
                # Check if the update is for our currently active *entry* order
                if not in_position and active_order_id == order_id:
                    if status == 'Filled':
                        entry_price = float(avg_price_str) if avg_price_str else 0.0
                        position_size = float(filled_qty_str) if filled_qty_str else 0.0
                        logging.info(
                            f"ENTRY ORDER {order_id} FILLED! Side: {side}, "
                            f"Qty: {position_size}, AvgPrice: {entry_price:.2f}"
                        )

                        in_position = True
                        position_entry_time = time.time()
                        current_position_side = side # Confirm side based on filled order
                        # Now place the Take Profit order
                        place_take_profit_order() # active_order_id will be updated inside

                    elif status in ['Cancelled', 'Rejected', 'Expired']:
                        logging.warning(f"Entry order {order_id} {status}. Resetting.")
                        reset_position_state() # Resets active_order_id too

                    # Note: Partial fills are complex. This basic logic assumes full fill or failure.
                    # For partial entry fills, you'd need to adjust TP size.

                # --- Handle Take Profit or Stop Loss Fill ---
                # If we are in position, *any* filled order that is reduceOnly means an exit
                elif in_position and status == 'Filled' and reduce_only:
                    exit_price = float(avg_price_str) if avg_price_str else 0.0
                    exit_qty = float(filled_qty_str) if filled_qty_str else 0.0
                    # Check if it's our TP order or the server-side SL that filled
                    if active_order_id == order_id:
                        logging.info(
                            f"TAKE PROFIT Order {order_id} FILLED at {exit_price:.2f}. "
                            f"Exiting position."
                        )
                    else:
                         # This means the server-side SL must have triggered and filled
                         # (Bybit creates a separate order for the SL execution)
                         logging.info(
                            f"STOP LOSS Order {order_id} FILLED at {exit_price:.2f}. "
                            f"Exiting position."
                         )
                         # We might want to try and cancel our now-redundant TP order
                         # if it hasn't been cancelled automatically by SL fill.
                         cancel_active_order("SL filled, cancelling TP")

                    reset_position_state()
                    last_trade_exit_time = time.time()

                # --- Handle Reduce-Only Order Cancellation (e.g., manual cancel of TP) ---
                elif in_position and active_order_id == order_id and reduce_only and status == 'Cancelled':
                     logging.warning(f"Active TP Order {order_id} was Cancelled externally! Position may still be open without TP.")
                     # Decide on action: close market? try placing TP again? For now, just log.
                     # For safety, maybe trigger market close here:
                     # close_position_market("TP was cancelled externally")


def setup_websocket():
    """Initializes and connects the Public and Private WebSockets."""
    ws_public_url = ("wss://stream-testnet.bybit.com/v5/public/linear" if USE_TESTNET
                     else "wss://stream.bybit.com/v5/public/linear")
    ws_private_url = ("wss://stream-testnet.bybit.com/v5/private" if USE_TESTNET
                      else "wss://stream.bybit.com/v5/private")

    # --- Public WebSocket ---
    ws_public = WebSocket(testnet=USE_TESTNET, channel_type="linear")
    logging.info(f"Connecting to Public WebSocket: {ws_public_url}")
    ws_public.subscribe(["orderbook.50." + SYMBOL, "liquidations." + SYMBOL])
    ws_public.websocket_data_handler(handle_public_message) # Register handler
    # Start public stream in a background thread (pybit handles the thread)
    ws_public.run()
    logging.info("Public WebSocket connected and subscriptions sent.")

    # --- Private WebSocket ---
    # Authentication is handled internally by pybit v5+ when api_key/secret provided
    ws_private = WebSocket(
        testnet=USE_TESTNET,
        channel_type="private", # Important: Use 'private' channel type
        api_key=API_KEY,
        api_secret=API_SECRET
    )
    logging.info(f"Connecting to Private WebSocket: {ws_private_url}")
    # Subscribe to order updates
    ws_private.subscribe(["order"]) # Topic for private order updates
    ws_private.websocket_data_handler(handle_private_message) # Register handler
    # Start private stream in a background thread
    ws_private.run()
    logging.info("Private WebSocket connected and subscriptions sent.")

# --- Trading Logic & Actions ---

def handle_liquidation_event(liquidation_side, liquidation_qty):
    """Processes a filtered liquidation event and attempts to place an entry order."""
    global active_order_id, current_position_side # Only write these under lock

    # Acquire lock to check and modify position state
    with position_lock:
        now = time.time()
        # Check Cooldown and if already attempting entry or in position
        if active_order_id or in_position:
            logging.debug("Ignoring liquidation: Already active (order placed or in position).")
            return
        if now - last_trade_exit_time < COOLDOWN_SECONDS:
            logging.debug(
                f"Ignoring liquidation: Still in cooldown "
                f"({COOLDOWN_SECONDS - (now - last_trade_exit_time):.1f}s left)."
            )
            return

        # Check Minimum Size
        if liquidation_qty < MIN_LIQUIDATION_QTY:
            logging.debug(
                f"Ignoring liquidation: Size {liquidation_qty:.4f} < minimum {MIN_LIQUIDATION_QTY:.4f}."
            )
            return

        # Determine Trade Direction (Fade the liquidation)
        # 'Sell' liq side means LONG was forced to sell -> We BUY
        # 'Buy' liq side means SHORT was forced to buy -> We SELL
        trade_side = 'Buy' if liquidation_side == 'Sell' else 'Sell'

        # Get current market prices (lock already held for order book)
        with order_book_lock:
            best_bid = last_best_bid
            best_ask = last_best_ask

        if best_bid is None or best_ask is None:
            logging.warning("Cannot place order: Order book data unavailable.")
            return

        # Calculate Entry Price Target
        entry_price_target = 0.0
        if trade_side == 'Buy':
            entry_price_target = best_bid + (ENTRY_PRICE_OFFSET_TICKS * TICK_SIZE)
            # Ensure we don't cross the spread immediately with limit order
            entry_price_target = min(entry_price_target, best_ask - TICK_SIZE)
        else: # trade_side == 'Sell'
            entry_price_target = best_ask - (ENTRY_PRICE_OFFSET_TICKS * TICK_SIZE)
            # Ensure we don't cross the spread immediately
            entry_price_target = max(entry_price_target, best_bid + TICK_SIZE)

        # Calculate Order Quantity
        mid_price = (best_bid + best_ask) / 2
        if mid_price <= 0:
            logging.warning("Cannot calculate order quantity: Mid price is zero.")
            return
        order_qty = round(ORDER_SIZE_USDT / mid_price, ORDER_QTY_PRECISION)
        if order_qty < MIN_ORDER_QTY:
             logging.warning(
                 f"Calculated order qty {order_qty} is below minimum {MIN_ORDER_QTY}. Skipping trade."
             )
             return

        # Calculate Stop Loss Price for the entry order
        sl_price = 0.0
        if trade_side == 'Buy':
            sl_price = entry_price_target - (STOP_LOSS_TICKS * TICK_SIZE)
        else: # trade_side == 'Sell'
            sl_price = entry_price_target + (STOP_LOSS_TICKS * TICK_SIZE)

        # Place the Entry Order with Server-Side Stop Loss
        logging.info(
            f"Attempting to {trade_side} {order_qty} based on {liquidation_side} liq. "
            f"Target Entry: {entry_price_target:.2f}, SL Price: {sl_price:.2f}"
        )
        placed_order_id = place_limit_order_with_sl(
            side=trade_side,
            price=entry_price_target,
            qty=order_qty,
            stop_loss_price=sl_price,
        )

        if placed_order_id:
            # Set state variables *after* successful placement attempt
            active_order_id = placed_order_id
            # Tentatively set side, will be confirmed on fill by WebSocket
            current_position_side = trade_side
            logging.info(f"Entry order {active_order_id} placed. Waiting for fill via WebSocket.")
        else:
            logging.error("Failed to place entry order.")
            # Ensure state is clean if placement failed
            reset_position_state()


def place_limit_order_with_sl(side, price, qty, stop_loss_price):
    """Places a limit order with a server-side stop loss."""
    try:
        # Round prices to tick size
        price = round(price / TICK_SIZE) * TICK_SIZE
        sl_price = round(stop_loss_price / TICK_SIZE) * TICK_SIZE
        price_str = f"{price:.{max(0, str(TICK_SIZE)[::-1].find('.'))}f}"
        sl_price_str = f"{sl_price:.{max(0, str(TICK_SIZE)[::-1].find('.'))}f}"
        qty_str = f"{qty:.{ORDER_QTY_PRECISION}f}"

        logging.info(
            f"Placing {side} Limit: Qty={qty_str}, Price={price_str}, "
            f"SL Price={sl_price_str} (Trigger: {STOP_LOSS_TRIGGER_METHOD})"
        )

        order_result = session.place_order(
            category="linear",
            symbol=SYMBOL,
            side=side,
            orderType="Limit",
            qty=qty_str,
            price=price_str,
            timeInForce="PostOnly", # Aim to be maker for entry
            stopLoss=sl_price_str,
            slTriggerBy=STOP_LOSS_TRIGGER_METHOD,
            reduceOnly=False # This is an entry order
        )

        logging.debug(f"Place entry order raw result: {order_result}")

        if order_result and order_result.get('retCode') == 0:
            order_id = order_result.get('result', {}).get('orderId')
            if order_id:
                logging.info(f"Entry order with SL placed successfully. ID: {order_id}")
                return order_id
            else:
                logging.error(f"Entry order placed but no orderId returned: {order_result}")
                return None
        else:
            err_code = order_result.get('retCode', 'N/A')
            err_msg = order_result.get('retMsg', 'No Response/Error')
            logging.error(f"Failed to place entry order: Code={err_code}, Msg='{err_msg}'")
            return None
    except Exception as e:
        logging.exception(f"Exception placing {side} entry order with SL.")
        return None


def place_take_profit_order():
    """Places the reduce-only take profit order after entry confirmation."""
    global active_order_id # Will store the TP order ID now

    if not in_position or not current_position_side or not entry_price or position_size <= 0:
        logging.error("Cannot place TP: Not in a valid position state.")
        # Consider closing market if state is inconsistent
        # close_position_market("Inconsistent state for TP placement")
        return

    # Calculate TP price
    tp_price = 0.0
    tp_side = ''
    if current_position_side == 'Buy':
        tp_price = entry_price + (TAKE_PROFIT_TICKS * TICK_SIZE)
        tp_side = 'Sell'
    else: # Sell position
        tp_price = entry_price - (TAKE_PROFIT_TICKS * TICK_SIZE)
        tp_side = 'Buy'

    # Round TP price
    tp_price = round(tp_price / TICK_SIZE) * TICK_SIZE
    tp_price_str = f"{tp_price:.{max(0, str(TICK_SIZE)[::-1].find('.'))}f}"
    qty_str = f"{position_size:.{ORDER_QTY_PRECISION}f}" # Use actual filled quantity

    logging.info(
        f"Placing {tp_side} Take Profit order: Qty={qty_str}, Price={tp_price_str}"
    )

    try:
        order_result = session.place_order(
            category="linear",
            symbol=SYMBOL,
            side=tp_side,
            orderType="Limit",
            qty=qty_str,
            price=tp_price_str,
            timeInForce="GTC", # GoodTillCancel for TP
            reduceOnly=True # CRITICAL: Ensure it only closes the position
        )

        logging.debug(f"Place TP order raw result: {order_result}")

        if order_result and order_result.get('retCode') == 0:
            order_id = order_result.get('result', {}).get('orderId')
            if order_id:
                logging.info(f"Take Profit order placed successfully. ID: {order_id}")
                # Update active_order_id to the TP order ID
                active_order_id = order_id
            else:
                logging.error(f"TP order placed but no orderId returned: {order_result}")
                # Close market if TP placement fails? Critical decision.
                close_position_market("Failed to get TP order ID")
        else:
            err_code = order_result.get('retCode', 'N/A')
            err_msg = order_result.get('retMsg', 'No Response/Error')
            logging.error(f"Failed to place TP order: Code={err_code}, Msg='{err_msg}'")
            # Close market if TP placement fails? Critical decision.
            close_position_market(f"Failed to place TP order (Code: {err_code})")

    except Exception as e:
        logging.exception(f"Exception placing {tp_side} TP order.")
        close_position_market("Exception placing TP order")


def cancel_active_order(reason=""):
    """Cancels the currently tracked active order (entry or TP)."""
    global active_order_id

    if not active_order_id:
        logging.debug("No active order ID to cancel.")
        return True # No order existed

    order_id_to_cancel = active_order_id
    active_order_id = None # Optimistically clear

    try:
        log_prefix = f"Cancelling order {order_id_to_cancel}"
        if reason: log_prefix += f" ({reason})"
        logging.info(f"{log_prefix}...")

        cancel_result = session.cancel_order(
            category="linear",
            symbol=SYMBOL,
            orderId=order_id_to_cancel
        )
        logging.debug(f"Cancel order raw result: {cancel_result}")

        if cancel_result and cancel_result.get('retCode') == 0:
             logging.info(f"Successfully cancelled order ID: {order_id_to_cancel}")
             return True
        else:
             # Log error, but ID is already cleared. Order might be filled/gone.
             err_code = cancel_result.get('retCode', 'N/A')
             err_msg = cancel_result.get('retMsg', 'Unknown Error')
             logging.error(
                 f"Failed to cancel order {order_id_to_cancel} "
                 f"(might be filled/gone): Code={err_code}, Msg='{err_msg}'"
             )
             return False # Indicate potential issue, though state is cleared

    except Exception as e:
        logging.exception(f"Exception cancelling order ID {order_id_to_cancel}.")
        return False


def close_position_market(reason=""):
    """Attempts to close the current position with a market order."""
    global position_lock # Ensure lock is acquired correctly

    with position_lock: # Acquire lock before modifying state/placing orders
        if not in_position:
            logging.info("Request to close position, but not in position.")
            return

        log_prefix = "Attempting market close of position"
        if reason: log_prefix += f" ({reason})"
        logging.info(log_prefix)

        # 1. Cancel the outstanding Take Profit order first (if it exists)
        # Must do this *before* placing market order to avoid race condition
        if active_order_id: # If TP order ID is tracked
            cancel_active_order("Closing position market")
            # Even if cancel fails, proceed to market close for safety

        # 2. Get current position size and side to place correct closing order
        current_pos_size = 0.0
        actual_pos_side = None
        try:
            position_info = session.get_positions(category="linear", symbol=SYMBOL)
            logging.debug(f"Get positions result for closing: {position_info}")
            if position_info and position_info.get('retCode') == 0:
                pos_list = position_info.get('result', {}).get('list', [])
                if pos_list and pos_list[0].get('symbol') == SYMBOL:
                    current_pos_size = float(pos_list[0].get('size', '0'))
                    actual_pos_side = pos_list[0].get('side') # 'Buy', 'Sell', or 'None'
            else:
                 err_msg = position_info.get('retMsg', 'Failed to get position info')
                 logging.error(f"Could not get position info for market close: {err_msg}")
                 # Fallback: try closing based on initially recorded side/size? Risky.
                 # For now, we won't place market order if we can't confirm size.
                 reset_position_state() # Reset state even if close fails
                 return # Exit function

        except Exception as e:
             logging.exception("Exception getting position info during market close.")
             reset_position_state() # Reset state even if close fails
             return # Exit function

        # 3. Place Market Order if position exists
        if current_pos_size > 0 and actual_pos_side in ['Buy', 'Sell']:
            close_side = 'Sell' if actual_pos_side == 'Buy' else 'Buy'
            qty_str = f"{current_pos_size:.{ORDER_QTY_PRECISION}f}"
            logging.info(
                f"Position Size: {current_pos_size}. Placing market {close_side} "
                f"order for {qty_str}."
            )
            try:
                order_result = session.place_order(
                    category="linear", symbol=SYMBOL, side=close_side,
                    orderType="Market", qty=qty_str, reduceOnly=True
                )
                logging.debug(f"Place market close order raw result: {order_result}")
                if not (order_result and order_result.get('retCode') == 0):
                     err_code = order_result.get('retCode', 'N/A')
                     err_msg = order_result.get('retMsg', 'No Response/Error')
                     logging.error(
                         f"CRITICAL: Failed to place market close order! "
                         f"Code={err_code}, Msg='{err_msg}' Manual intervention likely needed."
                     )
                # Market order assumed filled or failed, state reset happens below

            except Exception as e:
                 logging.exception("CRITICAL: Exception placing market close order!")
                 # Manual intervention likely needed if market order fails
        else:
            logging.warning("Position size reported as zero or side invalid ('None') before market close attempt.")

        # 4. Reset state regardless of market order success (to prevent reprocessing)
        # WebSocket message for the market order fill should confirm final state.
        reset_position_state()
        global last_trade_exit_time
        last_trade_exit_time = time.time()


def reset_position_state():
    """Resets all variables related to holding a position or active order."""
    global in_position, active_order_id, entry_price, position_entry_time
    global current_position_side, position_size
    # No need to acquire lock here if called from within a locked context,
    # but safer to acquire if called externally. Assume called from locked context mostly.
    # If called from KeyboardInterrupt, lock should be acquired there.
    logging.debug("Resetting position state.")
    in_position = False
    active_order_id = None
    entry_price = None
    position_entry_time = None
    current_position_side = None
    position_size = 0.0


# --- Main Loop & Position Management ---

def check_position_timeout():
    """Checks if the current position has timed out."""
    with position_lock:
        if in_position and position_entry_time:
            now = time.time()
            duration = now - position_entry_time
            if duration > POSITION_TIMEOUT_SECONDS:
                logging.warning(
                    f"Position TIMEOUT ({duration:.1f}s > {POSITION_TIMEOUT_SECONDS}s)! "
                    f"Closing market."
                )
                # Use a separate thread or non-blocking call if close_position_market
                # could block excessively, but for now, call directly.
                close_position_market("Timeout")
                # State reset happens inside close_position_market

def run_bot():
    """Main execution loop for the bot."""
    logging.info("Starting Liquidation Hunter Bot...")

    # --- Initialize WebSocket Connections ---
    # Run setup in a separate thread to allow main loop to start
    # Using daemon=True so thread exits when main program exits
    ws_thread = threading.Thread(target=setup_websocket, daemon=True)
    ws_thread.start()

    logging.info("Waiting for WebSocket connections and initial data...")
    # Wait longer to ensure private WS auth and subscriptions likely complete
    time.sleep(15)

    logging.info("Starting main monitoring loop.")
    while True:
        try:
            # Primary task in main loop is now checking for position timeout
            # Order entries and exits are handled reactively by WebSocket messages
            check_position_timeout()

            # Keep main thread alive and prevent high CPU usage
            time.sleep(1) # Check timeout every second

        except KeyboardInterrupt:
            logging.info("Shutdown signal received...")
            # Graceful shutdown: Try to close any open position
            with position_lock:
                 if in_position:
                     logging.warning("Position open during shutdown! Attempting market close.")
                     # This call handles cancelling TP order as well
                     close_position_market("Keyboard Interrupt Shutdown")
                 elif active_order_id:
                      # If an entry or TP order is active but not in position
                      logging.info(f"Cancelling active order {active_order_id} during shutdown.")
                      cancel_active_order("Keyboard Interrupt Shutdown")
            logging.info("Exiting bot.")
            break
        except Exception as e:
            # Log unexpected errors in the main loop
            logging.exception("FATAL ERROR in main loop!")
            # Consider adding emergency position closing here as well for safety
            logging.error("Attempting emergency shutdown procedure after main loop error.")
            try:
                with position_lock:
                    if in_position: close_position_market("Main Loop Unhandled Exception")
                    elif active_order_id: cancel_active_order("Main Loop Unhandled Exception")
            except Exception as shutdown_e:
                 logging.exception("Error during emergency shutdown!")
            time.sleep(5) # Pause before potentially restarting or exiting


if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        logging.error("CRITICAL: API Key or Secret not found in .env file. Exiting.")
    else:
        # Ensure clean state on start
        reset_position_state()
        run_bot()
