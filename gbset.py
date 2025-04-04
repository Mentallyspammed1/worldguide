python                                                                               import ccxt
import os
from dotenv import load_dotenv                                                          import time                                                                             
# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
SYMBOL = os.getenv("SYMBOL", "BTC/USDT")  # Trading symbol, default BTC/USDT if not
in .env
GRID_LEVELS = int(os.getenv("GRID_LEVELS", 7))  # Number of grid levels, default 7
GRID_INTERVAL = float(os.getenv("GRID_INTERVAL", 100))  # Price interval between grid
levels, default 100 USDT                                                                ORDER_SIZE = float(os.getenv("ORDER_SIZE", 0.001))  # Order quantity, default 0.001 BTC
LEVERAGE = int(os.getenv("LEVERAGE", 20))  # Leverage, default 20x
CHECK_INTERVAL_SECONDS = int(os.getenv("CHECK_INTERVAL_SECONDS", 30)) # Check interval,
default 30 seconds
PAPER_TRADING_MODE = os.getenv("PAPER_TRADING_MODE", "true").lower() == "true" # Paper
trading mode, default true                                                              MAX_OPEN_ORDERS = int(os.getenv("MAX_OPEN_ORDERS", 20)) # Max open orders per side,
default 20
GRID_TOTAL_PROFIT_TARGET_USD = float(os.getenv("GRID_TOTAL_PROFIT_TARGET_USD", 10)) #
Profit target in USD, default 10
GRID_TOTAL_LOSS_LIMIT_USD = float(os.getenv("GRID_TOTAL_LOSS_LIMIT_USD", -5)) # Loss
limit in USD, default -5
                                                                                        LOG_FILE = "gridbot_ccxt.log"

# --- Initialize Bybit exchange ---
exchange = ccxt.bybit({                                                                     'apiKey': os.getenv('BYBIT_API_KEY'),
    'secret': os.getenv('BYBIT_API_SECRET'),                                                'options': {
        'defaultType': 'spot', # or 'future', 'swap'                                        },
})

if PAPER_TRADING_MODE:
    log_message("INFO", "[PAPER TRADING MODE ENABLED]")                                     exchange.set_sandbox_mode(True) # Enable sandbox for testing
                                                                                        
# --- Global Variables ---
active_buy_orders = {}  # Stores active buy order IDs keyed by price
active_sell_orders = {} # Stores active sell order IDs keyed by price
trade_pairs = {} # Tracks open buy orders to match with sells for PNL calc (buy_order_id-> "buy_order_id,0,buy_price,0")
position_size = 0                                                                       entry_price = 0
realized_pnl_usd = 0                                                                    unrealized_pnl_usd = 0
total_pnl_usd = 0
order_fill_count = 0
order_placed_count = 0 # Counter for orders placed in one cycle
volatility_multiplier = 1
trend_bias = 0 # -1: Downtrend, 0: Neutral, 1: Uptrend

                                                                                        # --- Enhanced Logging Function ---
def log_message(level, message):                                                            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"{timestamp} [{level}] {message}"
    with open(LOG_FILE, "a") as f:                                                              f.write(log_entry + "\n")
    if level in ["INFO", "WARNING", "ERROR"]:
        print(log_entry)                                                                

# --- Helper Functions ---
def fetch_market_price(symbol):                                                             try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:                                                                      log_message("ERROR", f"Error fetching market price for {symbol}: {e}")
        return None                                                                     
def place_limit_order(side, symbol, quantity, price):
    if PAPER_TRADING_MODE:                                                                      paper_order_id = f"PAPER_{int(time.time())}_{os.urandom(4).hex()}"
        log_message("INFO", f"[PAPER TRADING] Simulating {side} order for {quantity}    {symbol} at {price}, ID: {paper_order_id}")
        return {'id': paper_order_id, 'info': {'status': 'open'}} # Simulate successful open order
    try:
        order = exchange.create_limit_order(symbol, side, quantity, price)                      log_message("INFO", f"Placed {side} order for {quantity} {symbol} at {price},
Order ID: {order['id']}")                                                                       return order
    except Exception as e:                                                                      log_message("ERROR", f"Error placing {side} order for {symbol} at {price}: {e}")
        return None                                                                     
def cancel_order(order_id, symbol):
    if PAPER_TRADING_MODE:
        log_message("INFO", f"[PAPER TRADING] Simulating cancellation of order ID:
{order_id} for {symbol}")
        return True
    try:                                                                                        exchange.cancel_order(order_id, symbol)
        log_message("INFO", f"Cancelled order ID: {order_id} for {symbol}")
        return True                                                                         except Exception as e:
        log_message("WARNING", f"Error cancelling order ID: {order_id} for {symbol}:
{e}")
        return False

def cancel_all_orders(symbol):
    if PAPER_TRADING_MODE:
        log_message("INFO", f"[PAPER TRADING] Simulating cancellation of all orders for {symbol}")
        active_buy_orders.clear()
        active_sell_orders.clear()
        trade_pairs.clear()
        return True
    try:                                                                                        orders = exchange.cancel_all_orders(symbol) # This might not work for spot,
needs to be tested for Bybit spot
        log_message("INFO", f"Cancelled all open orders for {symbol}")
        active_buy_orders.clear()
        active_sell_orders.clear()
        trade_pairs.clear()
        return True                                                                         except Exception as e:
        log_message("WARNING", f"Error cancelling all orders for {symbol}: {e}")
        return False
                                                                                        def fetch_open_orders(symbol):
    try:                                                                                        orders = exchange.fetch_open_orders(symbol)
        return orders
    except Exception as e:                                                                      log_message("WARNING", f"Error fetching open orders for {symbol}: {e}")
        return []                                                                       
def fetch_position(symbol):                                                                 global position_size, entry_price, unrealized_pnl_usd
    if PAPER_TRADING_MODE:
        log_message("DEBUG", "[PAPER TRADING] Simulating position fetch.")
        position_size = 0                                                                       entry_price = 0
        unrealized_pnl_usd = 0                                                                  return {'size': position_size, 'entryPrice': entry_price, 'unrealizedPnl':
unrealized_pnl_usd}                                                                     
    try:                                                                                        positions = exchange.fetch_positions([symbol])
        position = None                                                                         for p in positions:
            if p['symbol'] == symbol and p['side'] != 'long/short': # Check for relevantposition                                                                                                position = p
                break                                                                   
        if position and position['contracts'] != 0: # Check if position exists and is   not zero
            position_size = position['contracts'] # Or 'amount' for spot
            entry_price = position['entryPrice'] # Or 'average' for spot
            unrealized_pnl_usd = position['unrealizedPnl'] # Or calculate for spot
        else:
            position_size = 0
            entry_price = 0
            unrealized_pnl_usd = 0
                                                                                                log_message("DEBUG", f"Position: Size={position_size}, Entry={entry_price},
Unrealized PNL={unrealized_pnl_usd} USD")
        return position                                                                     except Exception as e:
        log_message("WARNING", f"Error fetching position for {symbol}: {e}")
        position_size = 0                                                                       entry_price = 0
        unrealized_pnl_usd = 0
        return None

def set_leverage(symbol, leverage):
    if PAPER_TRADING_MODE:
        log_message("INFO", "[PAPER TRADING] Skipping leverage setting.")
        return True
    try:
        exchange.set_leverage(leverage, symbol)
        log_message("INFO", f"Leverage set to {leverage}x for {symbol}")
        return True                                                                         except Exception as e:
        log_message("ERROR", f"Error setting leverage for {symbol} to {leverage}x: {e}")        return False
                                                                                        def calculate_grid_prices(current_price, grid_levels, grid_interval):
    buy_prices = []
    sell_prices = []                                                                        for i in range(1, grid_levels + 1):
        buy_price = current_price - (i * grid_interval)                                         sell_price = current_price + (i * grid_interval)
        buy_prices.append(round(buy_price, 2)) # Round prices as needed for your symbol         sell_prices.append(round(sell_price, 2))
    return buy_prices, sell_prices                                                      
def place_grid_orders(symbol, buy_prices, sell_prices, quantity):                           global order_placed_count
    order_placed_count = 0                                                              
    log_message("INFO", "Placing Sell orders...")                                           for price in sell_prices:
        if len(active_sell_orders) >= MAX_OPEN_ORDERS:                                              log_message("WARNING", f"Max open sell orders ({MAX_OPEN_ORDERS}) reached.
Skipping sell order at {price}.")
            continue
        order = place_limit_order('sell', symbol, quantity, price)
        if order and 'id' in order:
            active_sell_orders[price] = order['id']
            order_placed_count += 1
        time.sleep(0.1) # Rate limiting

    log_message("INFO", "Placing Buy orders...")
    for price in buy_prices:
        if len(active_buy_orders) >= MAX_OPEN_ORDERS:                                               log_message("WARNING", f"Max open buy orders ({MAX_OPEN_ORDERS}) reached.
Skipping buy order at {price}.")
            continue
        order = place_limit_order('buy', symbol, quantity, price)
        if order and 'id' in order:
            active_buy_orders[price] = order['id']
            order_placed_count += 1
        time.sleep(0.1) # Rate limiting

    log_message("INFO", f"Initial grid placed. Placed {order_placed_count} orders.")


def manage_grid_orders(symbol, grid_interval, quantity, max_open_orders):
    global order_placed_count, order_fill_count
    log_message("INFO", "--- Managing Grid Cycle Start ---")                                order_placed_count = 0

    current_price = fetch_market_price(symbol)                                              if current_price is None:                                                                   log_message("ERROR", "Failed to fetch current price. Skipping grid management
cycle.")                                                                                        return                                                                                                                                                                      # --- Check and Replenish Orders ---                                                    filled_buy_orders_prices = []                                                           filled_sell_orders_prices = []                                                      
    open_orders = fetch_open_orders(symbol) # Fetch all open orders to check status     
    # --- Check Buy Orders ---                                                              buy_prices_to_check = list(active_buy_orders.keys()) # Iterate over a copy to allow
modification                                                                                for price in buy_prices_to_check:
        order_id = active_buy_orders[price]                                                     order_status = None

        for order in open_orders:
            if order['id'] == order_id:
                order_status = order['status']
                break

        if order_status is None or order_status == 'closed' or order_status == 'filled':
# Assume filled if not found in open orders or status is closed/filled
            log_message("INFO", f"Buy order at {price} (ID: {order_id}) possibly filled/
closed. Replenishing.")
            filled_buy_orders_prices.append(price)
            del active_buy_orders[price]
            order_fill_count += 1

                                                                                            # --- Check Sell Orders ---
    sell_prices_to_check = list(active_sell_orders.keys()) # Iterate over a copy to
allow modification
    for price in sell_prices_to_check:
        order_id = active_sell_orders[price]
        order_status = None
                                                                                                for order in open_orders:
            if order['id'] == order_id:
                order_status = order['status']
                break                                                                   
        if order_status is None or order_status == 'closed' or order_status == 'filled':
# Assume filled if not found in open orders or status is closed/filled
            log_message("INFO", f"Sell order at {price} (ID: {order_id}) possibly       filled/closed. Replenishing.")
            filled_sell_orders_prices.append(price)                                                 del active_sell_orders[price]
            order_fill_count += 1                                                       
                                                                                            # --- Replenish Grid Levels ---
    dynamic_grid_interval = GRID_INTERVAL * volatility_multiplier # Example dynamic
interval                                                                                                                                                                            for filled_price in filled_buy_orders_prices:                                               new_sell_price = round(filled_price + dynamic_grid_interval, 2) # Round as      needed
        if new_sell_price not in active_sell_orders and len(active_sell_orders) <
max_open_orders:                                                                                    log_message("INFO", f"Replenishing Sell order at {new_sell_price} after Buy
fill at {filled_price}")                                                                            order = place_limit_order('sell', symbol, quantity, new_sell_price)                     if order and 'id' in order:                                                                 active_sell_orders[new_sell_price] = order['id']                                        order_placed_count += 1                                                 
    for filled_price in filled_sell_orders_prices:
        new_buy_price = round(filled_price - dynamic_grid_interval, 2) # Round as needed
        if new_buy_price not in active_buy_orders and len(active_buy_orders) <
max_open_orders:
            log_message("INFO", f"Replenishing Buy order at {new_buy_price} after Sell
fill at {filled_price}")
            order = place_limit_order('buy', symbol, quantity, new_buy_price)
            if order and 'id' in order:
                active_buy_orders[new_buy_price] = order['id']                                          order_placed_count += 1

    log_message("INFO", f"--- Managing Grid Cycle End (Placed {order_placed_count}      replenishment orders) ---")

                                                                                        def manage_position_and_pnl(symbol, profit_target_usd, loss_limit_usd):
    global realized_pnl_usd, unrealized_pnl_usd, total_pnl_usd                              fetch_position(symbol) # Update position_size, entry_price, unrealized_pnl_usd

    total_pnl_usd = realized_pnl_usd + unrealized_pnl_usd
                                                                                            log_message("INFO", f"PNL Status: Realized={realized_pnl_usd:.2f} USD,
Unrealized={unrealized_pnl_usd:.2f} USD, Total={total_pnl_usd:.2f} USD")
    log_message("INFO", f"Current Position: Size={position_size} {SYMBOL}, Avg
Entry={entry_price}")                                                                       log_message("INFO", f"Active Orders: {len(active_buy_orders)} Buys,
{len(active_sell_orders)} Sells. Total Fills: {order_fill_count}")

    if profit_target_usd > 0 and total_pnl_usd >= profit_target_usd:                            log_message("WARNING", f"--- PROFIT TARGET REACHED: Total PNL
{total_pnl_usd:.2f} USD >= Target {profit_target_usd:.2f} USD ---")                             log_message("WARNING", "--- Closing all positions and stopping bot ---")
        cancel_all_orders(symbol) # Ensure symbol is passed
        exit(0) # Gracefully exit

    if loss_limit_usd < 0 and total_pnl_usd <= loss_limit_usd: # Loss limit is usually
negative
        log_message("ERROR", f"--- STOP LOSS TRIGGERED: Total PNL {total_pnl_usd:.2f}   USD <= Loss Limit {loss_limit_usd:.2f} USD ---")
        log_message("ERROR", "--- Closing all positions and stopping bot ---")                  cancel_all_orders(symbol) # Ensure symbol is passed
        exit(1) # Exit with error code                                                  

# --- Main Bot Execution ---
def run_grid_bot():                                                                         log_message("INFO", "--- CCXT Bybit Grid Bot Started ---")
    log_message("INFO", f"Symbol: {SYMBOL}, Grid Levels: {GRID_LEVELS}, Interval:       {GRID_INTERVAL}, Order Size: {ORDER_SIZE}, Leverage: {LEVERAGE}")
    log_message("INFO", f"Paper Trading Mode: {PAPER_TRADING_MODE}, Check Interval:     {CHECK_INTERVAL_SECONDS} seconds")
    log_message("INFO", f"Profit Target: {GRID_TOTAL_PROFIT_TARGET_USD} USD, Loss Limit:
{GRID_TOTAL_LOSS_LIMIT_USD} USD, Max Open Orders: {MAX_OPEN_ORDERS}")                   
    if not PAPER_TRADING_MODE:
        if not exchange.check_required_credentials():
            log_message("ERROR", "API keys not properly configured. Please check .env
file and Bybit API settings.")
            exit(1)

        if not set_leverage(SYMBOL, LEVERAGE):
            log_message("ERROR", "Failed to set leverage. Bot exiting.")                            exit(1)
    else:
        log_message("INFO", "Running in Paper Trading mode.")
                                                                                        
    current_price = fetch_market_price(SYMBOL)
    if current_price is None:
        log_message("ERROR", "Failed to fetch initial market price. Bot exiting.")
        exit(1)
    log_message("INFO", f"Initial Market Price: {current_price}")                       
    buy_prices, sell_prices = calculate_grid_prices(current_price, GRID_LEVELS,
GRID_INTERVAL)
    place_grid_orders(SYMBOL, buy_prices, sell_prices, ORDER_SIZE)

    log_message("INFO", "Starting main monitoring loop...")
    while True:
        manage_grid_orders(SYMBOL, GRID_INTERVAL, ORDER_SIZE, MAX_OPEN_ORDERS)
        manage_position_and_pnl(SYMBOL, GRID_TOTAL_PROFIT_TARGET_USD,
GRID_TOTAL_LOSS_LIMIT_USD)
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_grid_bot()
```                                                                                     
**Before Running:**                                                                     
1.  **Install Libraries:**
    ```bash
    pip install ccxt python-dotenv
    ```
2.  **Create `.env` file:** In the same directory as your script, create a file named
`.env` and add your Bybit API keys and other configurations:

    ```dotenv
    BYBIT_API_KEY=YOUR_BYBIT_API_KEY                                                        BYBIT_API_SECRET=YOUR_BYBIT_API_SECRET
    SYMBOL=BTC/USDT        # Optional: Trading symbol (default BTC/USDT)
    GRID_LEVELS=7          # Optional: Number of grid levels                                GRID_INTERVAL=100      # Optional: Price interval                                       ORDER_SIZE=0.001       # Optional: Order size
    LEVERAGE=20            # Optional: Leverage
    CHECK_INTERVAL_SECONDS=30 # Optional: Check interval                                    PAPER_TRADING_MODE=true # Optional: Set to false for live trading
    MAX_OPEN_ORDERS=20      # Optional: Max open orders per side
    GRID_TOTAL_PROFIT_TARGET_USD=10 # Optional: Profit target                               GRID_TOTAL_LOSS_LIMIT_USD=-5   # Optional: Loss limit (negative value)
