#!/usr/bin/env python3
"""Bybit Futures Terminal - v1.1 - WorldGuides Edit - Arcane Edition

🔮 Arcane Enhancements: Robust error handling, input validation, decorators,
   funding rate display, dependency checks, and refined mystical UX.
"""

import hashlib
import hmac
import json
import os
import sys
import time
import urllib.parse
from collections.abc import Callable
from functools import wraps
from typing import Any

# Attempt to import dependencies and guide user if missing
try:
    import ccxt
    import pandas as pd
    import pandas_ta as ta  # Check pandas_ta specifically
    import requests
    from colorama import Fore, Style, init
    from dotenv import load_dotenv
except ImportError as e:
    missing_module = str(e).split("'")[1]
    print(f"{Fore.RED}{Style.BRIGHT}🔥 Error: Missing required library '{missing_module}'.")
    print(f"{Fore.YELLOW}   Please install it using: {Style.BRIGHT}pip install {missing_module}{Style.RESET_ALL}")
    sys.exit(1)

from indicators import (  # Assuming indicators.py uses pandas_ta
    ATR,
    FibonacciPivotPoints,
    rsi,
)

# --- Constants ---
BASE_API_URL: str = "https://api.bybit.com"
API_VERSION: str = "v5"
CATEGORY: str = "linear" # Assuming linear perpetuals/futures
RECV_WINDOW: str = "5000"
DEFAULT_TIMEOUT: int = 10 # Seconds for requests timeout

# --- Initialization ---
# 🔮 Initialize Colorama
init(autoreset=True)

# ✨ Load API keys from .env file
load_dotenv()
BYBIT_API_KEY: str | None = os.environ.get("BYBIT_API_KEY")
BYBIT_API_SECRET: str | None = os.environ.get("BYBIT_API_SECRET")

# 🌐 CCXT Exchange object (initialized later)
EXCHANGE: ccxt.Exchange | None = None

# --- Decorators for Pre-checks ---
def require_api_keys(func: Callable) -> Callable:
    """Decorator to ensure API keys are loaded before executing a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not BYBIT_API_KEY or not BYBIT_API_SECRET:
            print(Fore.RED + Style.BRIGHT + "🔥 API keys are not imbued. This action cannot proceed.")
            pause_terminal()
            return None # Indicate failure or inability to proceed
        return func(*args, **kwargs)
    return wrapper

def require_exchange(func: Callable) -> Callable:
    """Decorator to ensure the CCXT EXCHANGE object is initialized."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not EXCHANGE:
            print(Fore.RED + Style.BRIGHT + "🔥 CCXT Exchange client is offline. This action requires an active connection.")
            pause_terminal()
            return None # Indicate failure or inability to proceed
        return func(*args, **kwargs)
    return wrapper

# --- 🔑 API Key Handling ---
def initialize_exchange() -> bool:
    """Initializes the CCXT Bybit exchange client."""
    global EXCHANGE
    if BYBIT_API_KEY and BYBIT_API_SECRET:
        try:
            EXCHANGE = ccxt.bybit(
                {
                    "apiKey": BYBIT_API_KEY,
                    "secret": BYBIT_API_SECRET,
                    "options": {"defaultType": "swap", "adjustForTimeDifference": True}, # Added time sync
                    "timeout": 20000, # CCXT timeout in milliseconds
                }
            )
            # Test connection
            EXCHANGE.load_markets()
            print(Fore.CYAN + Style.BRIGHT + "🔗 CCXT Bybit Futures client connection established.")
            return True
        except (ccxt.AuthenticationError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
             print(Fore.RED + Style.BRIGHT + f"🔥 CCXT Connection Error: {e.__class__.__name__} - {e}")
        except ccxt.ExchangeError as e:
            print(Fore.RED + Style.BRIGHT + f"🔥 CCXT Exchange Initialization Error: {e}")
        except Exception as e:
            print(Fore.RED + Style.BRIGHT + f"🔥 General Error initializing CCXT: {e}")
    else:
        print(
            Fore.YELLOW # Changed to Yellow as it's a warning, not necessarily an error
            + Style.BRIGHT
            + "⚠️ API keys missing or not loaded. Trading/Account features disabled."
        )
    EXCHANGE = None
    return False

@require_api_keys
def debug_display_api_keys() -> None:
    """🔧 Displays loaded API keys (masked)."""
    os.system("clear")
    print(Fore.YELLOW + Style.BRIGHT + "🔧 DEBUG: Displaying API Keys (Masked for safety)")
    masked_key = mask_api_key(BYBIT_API_KEY)
    masked_secret = mask_api_key(BYBIT_API_SECRET)
    print(Fore.WHITE + f"API Key: {Fore.CYAN}{masked_key}")
    print(Fore.WHITE + f"API Secret: {Fore.CYAN}{masked_secret}")
    pause_terminal()

def mask_api_key(api_key: str | None) -> str:
    """Masks API key for display."""
    if not api_key or len(api_key) < 8:
        return "Invalid/Short Key"
    return api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]

# --- 📜 API Request Functions (Direct Requests) ---
@require_api_keys
def generate_signature(api_secret: str, params: dict[str, Any]) -> str:
    """🖋️ Generates API signature for Bybit requests."""
    # Ensure all values are strings for urlencode
    params_str = {k: str(v) for k, v in params.items()}
    query_string = urllib.parse.urlencode(sorted(params_str.items()))
    # print(f"DEBUG: Signature Base String: {query_string}") # Uncomment for debugging signature issues
    signature = hmac.new(
        api_secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return signature

@require_api_keys
def send_bybit_request(method: str, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Handles sending requests to Bybit API and robust error handling."""
    api_url = f"{BASE_API_URL}/{API_VERSION}/{endpoint}"
    headers = {"Content-Type": "application/json"}

    request_params: dict[str, Any] = params if params is not None else {}

    # Add authentication parameters
    timestamp = str(int(time.time() * 1000))
    auth_params = {
        "api_key": BYBIT_API_KEY,
        "timestamp": timestamp,
        "recv_window": RECV_WINDOW,
    }

    # Combine original params and auth params for signature generation
    # For GET, signature includes query params. For POST, it includes body params.
    signature_payload = {**request_params, **auth_params} if method == "POST" else {**auth_params}
    # If GET request has parameters, they should also be included in the signature base string
    if method == "GET" and params:
         signature_payload.update(params)

    auth_params["sign"] = generate_signature(BYBIT_API_SECRET, signature_payload)

    # Add auth headers/params to the actual request
    headers.update({f"X-BAPI-{k.replace('_','').upper()}": str(v) for k, v in auth_params.items()}) # Bybit V5 uses headers for auth

    try:
        # print(f"DEBUG: Sending {method} to {api_url}") # Debugging
        # print(f"DEBUG: Headers: {headers}") # Debugging
        # print(f"DEBUG: Params/Body: {request_params}") # Debugging

        if method == "POST":
            response = requests.post(api_url, headers=headers, json=request_params, timeout=DEFAULT_TIMEOUT)
        elif method == "GET":
            response = requests.get(api_url, headers=headers, params=request_params, timeout=DEFAULT_TIMEOUT) # Use request_params for GET query
        else:
            print(Fore.RED + Style.BRIGHT + f"🔥 Unsupported HTTP method: {method}")
            return None

        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()

    except requests.exceptions.Timeout:
        print(Fore.RED + Style.BRIGHT + f"🔥 API Request Timeout ({DEFAULT_TIMEOUT}s) to {endpoint}.")
        return None
    except requests.exceptions.ConnectionError as e:
        print(Fore.RED + Style.BRIGHT + f"🔥 API Connection Error: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(Fore.RED + Style.BRIGHT + f"🔥 API Request Exception: {e}")
        if e.response is not None:
            print(Fore.RED + f"   Response Status: {e.response.status_code}")
            try:
                print(Fore.RED + f"   Response Body: {e.response.text}") # Show error from Bybit if available
            except Exception:
                pass # Ignore if response body isn't readable text
        return None
    except json.JSONDecodeError:
        print(Fore.RED + Style.BRIGHT + f"🔥 Failed to parse JSON response from {endpoint}.")
        return None
    except Exception as e: # Catch any other unexpected errors
        print(Fore.RED + Style.BRIGHT + f"🔥 An unexpected error occurred during API request: {e}")
        return None

# --- 交易操作 (Trading Actions - Direct Requests) ---
@require_api_keys
def place_order_requests(symbol: str, side: str, order_type: str, qty: float, price: float | None = None) -> dict[str, Any] | None:
    """Places an order using direct requests."""
    endpoint = "order/create"
    params = {
        "category": CATEGORY,
        "symbol": symbol,
        "side": side.capitalize(),
        "orderType": order_type.capitalize(),
        "qty": str(qty),
        "timeInForce": "GTC", # Good-Til-Cancelled is common default
    }
    if order_type.lower() == "limit" and price is not None:
        params["price"] = str(price)

    order_data = send_bybit_request("POST", endpoint, params)

    if order_data and order_data["retCode"] == 0:
        # Success
        order_info = order_data.get("result", {}) # V5 uses 'result' not 'result.order' directly
        # Bybit V5 create response gives orderId directly in result
        return order_info if order_info else {} # Return result dict or empty dict
    else:
        # Failure
        error_msg = order_data.get('retMsg', 'Unknown error') if order_data else "No response data"
        print(
            Fore.RED
            + Style.BRIGHT
            + f"🔥 Bybit API Order Error ({order_data.get('retCode', 'N/A')}): {error_msg}"
        )
        return None

@require_api_keys
def place_market_order_requests() -> None:
    """🛒 Places a market order via direct requests."""
    details = get_order_details_from_user("market")
    if not details: return

    symbol, side, amount = details
    order_result = place_order_requests(symbol, side, "market", amount)
    if order_result:
        display_order_execution_message("MARKET ORDER EXECUTED", symbol, side, amount, order_id=order_result.get('orderId'))
    else:
        print(Fore.RED + Style.BRIGHT + "🔥 Market order placement failed.")
    pause_terminal()

@require_api_keys
def place_limit_order_requests() -> None:
    """🚧 Places a limit order via direct requests."""
    details = get_order_details_from_user("limit")
    if not details: return

    symbol, side, amount, price = details
    order_result = place_order_requests(symbol, side, "limit", amount, price)
    if order_result:
        display_order_execution_message("LIMIT ORDER PLACED", symbol, side, amount, price=price, order_id=order_result.get('orderId'))
    else:
        print(Fore.RED + Style.BRIGHT + "🔥 Limit order placement failed.")
    pause_terminal()

# --- 订单管理 (Order Management) ---
@require_api_keys
def cancel_single_order_requests() -> None:
    """✨ Cancels a single order using direct requests."""
    order_id, symbol = get_order_id_symbol_from_user("Cancel Single Order")
    if not order_id or not symbol: return # Symbol is mandatory for V5 cancel

    endpoint = "order/cancel"
    params = {
        "category": CATEGORY,
        "symbol": symbol,
        "orderId": order_id,
    }

    result = send_bybit_request("POST", endpoint, params)
    if result and result["retCode"] == 0:
        cancelled_order_id = result.get('result', {}).get('orderId', order_id) # Confirm ID from response
        print(Fore.GREEN + Style.BRIGHT + f"✅ Order {cancelled_order_id} cancellation successful!")
    else:
        error_msg = result.get('retMsg', 'Unknown error') if result else "No response data"
        print(Fore.RED + Style.BRIGHT + f"🔥 Cancellation failed ({result.get('retCode', 'N/A')}): {error_msg}")
    pause_terminal()

@require_api_keys
def cancel_all_orders_requests() -> None:
    """💥 Cancels ALL open orders for a symbol or all symbols using direct requests."""
    symbol = get_symbol_input("Enter symbol to cancel ALL orders for (e.g., BTCUSDT, leave blank for ALL symbols): ", allow_blank=True)
    target = f"symbol {symbol}" if symbol else "ALL symbols"

    if not get_confirmation(f"⚠️ Confirm mass cancellation for {target}?"):
        print(Fore.YELLOW + Style.BRIGHT + " Mass order cancellation aborted.")
        pause_terminal()
        return

    endpoint = "order/cancel-all"
    params = {"category": CATEGORY}
    if symbol:
        params["symbol"] = symbol

    result = send_bybit_request("POST", endpoint, params)
    if result and result["retCode"] == 0:
        print(Fore.GREEN + Style.BRIGHT + "✅ Mass order cancellation incantation successful!")
        print(Fore.WHITE + f"🪙 Orders for {target} are now vanquished.")
        cancelled_list = result.get('result', {}).get('list', [])
        cancelled_orders_count = len(cancelled_list) if isinstance(cancelled_list, list) else 0
        print(Fore.WHITE + f"🆔 Orders cancelled count: {Fore.CYAN}{cancelled_orders_count}")
    else:
        error_msg = result.get('retMsg', 'Unknown error') if result else "No response data"
        print(Fore.RED + Style.BRIGHT + f"🔥 Mass cancellation failed ({result.get('retCode', 'N/A')}): {error_msg}")
    pause_terminal()

@require_exchange
def cancel_futures_order_ccxt() -> None:
    """Cancels a Futures order using CCXT."""
    order_id, symbol = get_order_id_symbol_from_user("Cancel Order (CCXT)")
    if not order_id: return # Order ID is mandatory

    try:
        # CCXT often requires the symbol for cancellation on Bybit
        if not symbol:
             print(Fore.YELLOW + Style.BRIGHT + "⚠️ Symbol is generally required for CCXT cancellation on Bybit. Attempting without...")
        EXCHANGE.cancel_order(order_id, symbol=symbol) # Pass symbol even if None, CCXT might handle it
        print(Fore.GREEN + Style.BRIGHT + f"✅ Order ID {order_id} cancellation successful (CCXT)!")
    except ccxt.OrderNotFound:
        print(Fore.YELLOW + Style.BRIGHT + f"⚠️ Order ID {order_id} not found or already vanquished (CCXT).")
    except ccxt.InvalidOrder as e:
         print(Fore.RED + Style.BRIGHT + f"🔥 Invalid Order/Parameters (CCXT): {e}")
    except ccxt.ExchangeError as e:
        print(Fore.RED + Style.BRIGHT + f"🔥 Bybit Exchange Error during cancellation (CCXT): {e}")
    except Exception as e:
        print(Fore.RED + Style.BRIGHT + f"🔥 Error cancelling order (CCXT): {e}")
    pause_terminal()

@require_exchange
def view_open_futures_orders_ccxt() -> None:
    """Views open Futures orders using CCXT."""
    symbol = get_symbol_input("Enter symbol to view open orders (e.g., BTCUSDT, or leave blank for all): ", allow_blank=True)

    try:
        print(Fore.CYAN + f"⏳ Fetching open orders for {symbol if symbol else 'all symbols'}...")
        open_orders = EXCHANGE.fetch_open_orders(symbol=symbol if symbol else None)

        if open_orders:
            df = pd.DataFrame(open_orders)
            # Ensure necessary columns exist, adding them with None if missing
            required_cols = ['datetime', 'id', 'symbol', 'type', 'side', 'amount', 'price', 'status']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = None
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms') # Convert timestamp to datetime
            display_dataframe("OPEN FUTURES ORDERS", df,
                              required_cols,
                              formatters={'price': "{:.4f}".format, 'amount': "{:.4f}".format}) # Adjust price format
        else:
            print(Fore.YELLOW + "✅ No open orders found.")

    except ccxt.ExchangeError as e:
        print(Fore.RED + Style.BRIGHT + f"🔥 Bybit Exchange Error fetching open orders: {e}")
    except Exception as e:
        print(Fore.RED + Style.BRIGHT + f"🔥 Error fetching open orders: {e}")
    pause_terminal()

@require_exchange
def view_open_futures_positions_ccxt() -> None:
    """Views open Futures positions using CCXT."""
    try:
        print(Fore.CYAN + "⏳ Fetching open positions...")
        # Bybit needs the market type for positions
        positions = EXCHANGE.fetch_positions(params={'category': CATEGORY})
        # Filter out positions with zero contracts/size
        positions = [p for p in positions if p.get('contracts') is not None and float(p['contracts']) != 0]

        if positions:
            df = pd.DataFrame(positions)
             # Select and rename columns for better readability
            col_map = {
                'symbol': 'Symbol',
                'entryPrice': 'Entry Price',
                'markPrice': 'Mark Price', # Often more relevant than liq price alone
                'liquidationPrice': 'Liq. Price',
                'contracts': 'Contracts',
                'side': 'Side',
                'unrealizedPnl': 'Unrealized PNL',
                'percentage': 'PNL %',
                'leverage': 'Leverage',
                'initialMargin': 'Margin'
            }
            # Ensure columns exist before selecting/renaming
            display_cols_raw = [k for k in col_map if k in df.columns]
            display_cols_renamed = [col_map[k] for k in display_cols_raw]

            df_display = df[display_cols_raw].copy() # Create a copy to avoid SettingWithCopyWarning
            df_display.columns = display_cols_renamed

            display_dataframe("OPEN FUTURES POSITIONS", df_display,
                              formatters={'Entry Price': "{:.4f}".format,
                                          'Mark Price': "{:.4f}".format,
                                          'Liq. Price': "{:.4f}".format,
                                          'Contracts': "{:.4f}".format,
                                          'Unrealized PNL': "{:.4f}".format,
                                          'PNL %': "{:.2f}%".format,
                                          'Margin': "{:.4f}".format})
        else:
            print(Fore.YELLOW + "✅ No open positions found.")

    except ccxt.ExchangeError as e:
        print(Fore.RED + Style.BRIGHT + f"🔥 Bybit Exchange Error fetching positions: {e}")
    except Exception as e:
        print(Fore.RED + Style.BRIGHT + f"🔥 Error fetching positions: {e}")
    pause_terminal()

# --- 账户操作 (Account Operations) ---
@require_exchange
def view_account_balance() -> None:
    """💰 Fetches and displays Futures account balance using CCXT."""
    try:
        print(Fore.CYAN + "⏳ Fetching account balance...")
        # Fetch balance for the specific account type (Unified Trading Account - Contract)
        balance = EXCHANGE.fetch_balance(params={'accountType': 'CONTRACT'})

        if balance and balance.get('info', {}).get('result', {}).get('list'):
            account_info = balance['info']['result']['list'][0] # Assuming one contract account
            equity = account_info.get('equity', 'N/A')
            unrealized_pnl = account_info.get('unrealisedPnl', 'N/A')
            available_balance = account_info.get('availableToWithdraw', 'N/A') # Or 'availableBalance'
            total_margin = account_info.get('totalInitialMargin', 'N/A')

            balance_data = {
                "Metric": ["Equity", "Available Balance", "Unrealized PNL", "Total Initial Margin"],
                "Value": [equity, available_balance, unrealized_pnl, total_margin]
            }
            df = pd.DataFrame(balance_data)
            display_dataframe("ACCOUNT BALANCE (CONTRACT)", df,
                              formatters={"Value": lambda x: f"{float(x):.4f}" if isinstance(x, (str, int, float)) and x != 'N/A' else x})

        else:
            print(Fore.YELLOW + "⚠️ Could not parse account balance details from response.")
            # print(f"DEBUG: Full Balance Response:\n{balance}") # Uncomment for debugging

    except ccxt.ExchangeError as e:
        print(Fore.RED + Style.BRIGHT + f"🔥 Bybit Exchange Error fetching balance: {e}")
    except Exception as e:
        print(Fore.RED + Style.BRIGHT + f"🔥 Error fetching balance: {e}")
    pause_terminal()

@require_exchange
def view_order_history() -> None:
    """📜 Fetches and displays Futures order history using CCXT."""
    symbol = get_symbol_input("Enter Futures symbol (e.g., BTCUSDT): ")
    if not symbol: return

    try:
        print(Fore.CYAN + f"⏳ Fetching order history for {symbol}...")
        # Fetch both closed and canceled orders
        orders = EXCHANGE.fetch_closed_orders(symbol=symbol, limit=50) # Limit history length
        # Note: Bybit V5 might not have a separate canceled endpoint via CCXT easily, closed might include them
        # If needed, add fetch_canceled_orders if supported and distinct

        if orders:
            df = pd.DataFrame(orders)
            required_cols = ['datetime', 'id', 'symbol', 'type', 'side', 'amount', 'average', 'price', 'status', 'fee']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = None # Add missing columns

            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df_display = df[required_cols].sort_values(by='datetime', ascending=False) # Sort by time

            display_dataframe(f"ORDER HISTORY FOR {symbol}", df_display,
                              formatters={"price": "{:.4f}".format,
                                          "average": "{:.4f}".format, # Filled price
                                          "amount": "{:.4f}".format,
                                          "fee": lambda x: f"{x['cost']:.4f} {x['currency']}" if x and isinstance(x, dict) else 'N/A'})
        else:
            print(Fore.YELLOW + f"⚠️ No order history found for {symbol}.")

    except ccxt.ExchangeError as e:
        print(Fore.RED + Style.BRIGHT + f"🔥 Bybit Exchange Error fetching order history: {e}")
    except Exception as e:
        print(Fore.RED + Style.BRIGHT + f"🔥 Error fetching order history: {e}")
    pause_terminal()

# --- 市场数据 (Market Data) ---
@require_exchange
def fetch_symbol_price() -> None:
    """💰 Fetches and displays the current price of a Futures symbol."""
    symbol = get_symbol_input("Enter Futures symbol (e.g., BTCUSDT): ")
    if not symbol: return

    try:
        print(Fore.CYAN + f"⏳ Fetching ticker for {symbol}...")
        ticker = EXCHANGE.fetch_ticker(symbol)
        if ticker and "last" in ticker:
            os.system("clear")
            print(
                Fore.CYAN
                + Style.BRIGHT
                + "╔════════════"
                + Fore.GREEN
                + "SYMBOL PRICE"
                + Fore.CYAN
                + "═════════════╗"
            )
            print(
                Fore.WHITE
                + Style.BRIGHT
                + f"\nCurrent price of {Fore.GREEN}{symbol}{Fore.WHITE}: {Fore.GREEN}{ticker['last']:.4f}" # Increased precision
            )
        else:
            print(Fore.RED + Style.BRIGHT + f"🔥 Could not fetch price for {symbol}.")
    except ccxt.BadSymbol as e:
         print(Fore.RED + Style.BRIGHT + f"🔥 Invalid Symbol: {e}")
    except ccxt.ExchangeError as e:
        print(Fore.RED + Style.BRIGHT + f"🔥 Bybit Exchange Error fetching price: {e}")
    except Exception as e:
        print(Fore.RED + Style.BRIGHT + f"🔥 Error fetching price: {e}")
    pause_terminal()

@require_exchange
def get_order_book() -> None:
    """📖 Fetches and displays the order book for a Futures symbol."""
    symbol = get_symbol_input("Enter Futures symbol (e.g., BTCUSDT): ")
    if not symbol: return

    try:
        print(Fore.CYAN + f"⏳ Fetching order book for {symbol}...")
        orderbook = EXCHANGE.fetch_order_book(symbol, limit=10) # Limit depth
        if orderbook and "bids" in orderbook and "asks" in orderbook:
            bid_df = pd.DataFrame(orderbook["bids"], columns=["Price", "Amount"])
            ask_df = pd.DataFrame(orderbook["asks"], columns=["Price", "Amount"])

            # Display side-by-side if possible, otherwise sequentially
            # Simple sequential display for terminal:
            display_dataframe(f"ORDER BOOK - BIDS ({symbol})", bid_df, color=Fore.GREEN, formatters={"Price": "{:.4f}".format})
            display_dataframe(f"ORDER BOOK - ASKS ({symbol})", ask_df, color=Fore.RED, formatters={"Price": "{:.4f}".format})
        else:
            print(Fore.RED + Style.BRIGHT + f"🔥 Could not fetch order book for {symbol}.")
    except ccxt.BadSymbol as e:
         print(Fore.RED + Style.BRIGHT + f"🔥 Invalid Symbol: {e}")
    except ccxt.ExchangeError as e:
        print(Fore.RED + Style.BRIGHT + f"🔥 Bybit Exchange Error fetching order book: {e}")
    except Exception as e:
        print(Fore.RED + Style.BRIGHT + f"🔥 Error fetching order book: {e}")
    pause_terminal()

@require_exchange
def list_available_symbols() -> None:
    """📜 Lists available Futures trading symbols on Bybit."""
    try:
        print(Fore.CYAN + "⏳ Loading available markets...")
        markets = EXCHANGE.load_markets()
        # Filter for linear perpetuals and futures
        futures_symbols = [
            symbol for symbol, market in markets.items()
            if market.get('type') in ['swap', 'future'] and market.get('linear')
        ]
        futures_symbols.sort() # Sort alphabetically

        os.system("clear")
        print(
            Fore.CYAN
            + Style.BRIGHT
            + "╔══════════"
            + Fore.GREEN
            + "AVAILABLE LINEAR FUTURES/SWAPS"
            + Fore.CYAN
            + "═══════════╗"
        )
        if futures_symbols:
            # Simple comma-separated list for now
            print(Fore.GREEN + ", ".join(futures_symbols))
            # Consider pagination or columns for very long lists if needed
        else:
            print(Fore.YELLOW + "⚠️ No linear futures/swap symbols found.")
        print(
            Fore.CYAN + Style.BRIGHT + "\n---------------------------------------"
        )
    except ccxt.ExchangeError as e:
        print(Fore.RED + Style.BRIGHT + f"🔥 Bybit Exchange Error loading markets: {e}")
    except Exception as e:
        print(Fore.RED + Style.BRIGHT + f"🔥 Error fetching symbols: {e}")
    pause_terminal()

@require_exchange
def display_funding_rates() -> None:
    """💸 Fetches and displays funding rates for symbols."""
    symbol = get_symbol_input("Enter symbol for funding rate (e.g., BTCUSDT, or leave blank for multiple): ", allow_blank=True)

    try:
        print(Fore.CYAN + f"⏳ Fetching funding rates for {symbol if symbol else 'multiple symbols'}...")
        if symbol:
            rates = [EXCHANGE.fetch_funding_rate(symbol)]
        else:
            # Fetch for a few popular symbols as an example, fetching all can be slow/rate-limited
            popular_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT'] # Example list
            rates = EXCHANGE.fetch_funding_rates(symbols=popular_symbols)
            rates = list(rates.values()) # Convert dict to list

        if rates:
            df = pd.DataFrame(rates)
            required_cols = ['symbol', 'fundingRate', 'fundingTimestamp', 'fundingTime']
            for col in required_cols:
                 if col not in df.columns:
                     df[col] = None

            df['fundingTime'] = pd.to_datetime(df['fundingTimestamp'], unit='ms') # Convert timestamp
            df['fundingRate'] = df['fundingRate'] * 100 # Display as percentage

            display_dataframe("FUNDING RATES", df[['symbol', 'fundingRate', 'fundingTime']],
                              column_names=['Symbol', 'Funding Rate (%)', 'Next Funding Time'],
                              formatters={'Funding Rate (%)': "{:.4f}%".format})
        else:
            print(Fore.YELLOW + f"⚠️ No funding rate data found for {'symbol ' + symbol if symbol else 'requested symbols'}.")

    except ccxt.BadSymbol as e:
         print(Fore.RED + Style.BRIGHT + f"🔥 Invalid Symbol: {e}")
    except ccxt.ExchangeError as e:
        print(Fore.RED + Style.BRIGHT + f"🔥 Bybit Exchange Error fetching funding rates: {e}")
    except Exception as e:
        print(Fore.RED + Style.BRIGHT + f"🔥 Error fetching funding rates: {e}")
    pause_terminal()


# --- 技术指标 (Technical Indicators) ---
@require_exchange
def display_rsi_indicator() -> None:
    """📈 Calculates and displays RSI for a given symbol."""
    params = get_indicator_params_from_user("RSI")
    if not params: return

    symbol, period, timeframe = params
    try:
        ohlcv = fetch_ohlcv_data(symbol, timeframe, period + 150) # Fetch more data for stability
        if ohlcv is None: return

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = pd.to_numeric(df['close']) # Ensure numeric type

        rsi_series = rsi(df['close'], period) # Assuming rsi is from indicators.py using pandas_ta

        if rsi_series is None or rsi_series.isna().all():
             print(Fore.YELLOW + Style.BRIGHT + "⚠️ RSI calculation failed or returned all NaNs.")
             return

        last_rsi = rsi_series.iloc[-1]
        if pd.isna(last_rsi):
            print(Fore.YELLOW + Style.BRIGHT + "⚠️ RSI calculation incomplete (NaN) for the last candle.")
        else:
            os.system("clear")
            print_indicator_header("RSI INDICATOR")
            print(
                Fore.WHITE
                + Style.BRIGHT
                + f"\nRSI ({period}) for {Fore.GREEN}{symbol}{Fore.WHITE} ({timeframe}): {Fore.GREEN}{last_rsi:.2f}"
            )
            if last_rsi > 70:
                print(Fore.RED + Style.BRIGHT + "  🔥 Overbought condition!")
            elif last_rsi < 30:
                print(Fore.GREEN + Style.BRIGHT + "  ✅ Oversold condition!")
            print_indicator_footer()

    except Exception as e:
        print(Fore.RED + Style.BRIGHT + f"🔥 Error calculating RSI: {e}")
    pause_terminal()

@require_exchange
def display_atr_indicator() -> None:
    """📈 Calculates and displays ATR for a given symbol."""
    params = get_indicator_params_from_user("ATR")
    if not params: return

    symbol, period, timeframe = params
    try:
        ohlcv = fetch_ohlcv_data(symbol, timeframe, period + 150) # Fetch more data
        if ohlcv is None: return

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])

        atr_series = ATR(df['high'], df['low'], df['close'], period) # Assuming ATR from indicators.py

        if atr_series is None or atr_series.isna().all():
             print(Fore.YELLOW + Style.BRIGHT + "⚠️ ATR calculation failed or returned all NaNs.")
             return

        last_atr = atr_series.iloc[-1]
        if pd.isna(last_atr):
            print(Fore.YELLOW + Style.BRIGHT + "⚠️ ATR calculation incomplete (NaN) for the last candle.")
        else:
            os.system("clear")
            print_indicator_header("ATR INDICATOR")
            print(
                Fore.WHITE
                + Style.BRIGHT
                + f"\nATR ({period}) for {Fore.GREEN}{symbol}{Fore.WHITE} ({timeframe}): {Fore.GREEN}{last_atr:.4f}"
            )
            print_indicator_footer()

    except Exception as e:
        print(Fore.RED + Style.BRIGHT + f"🔥 Error calculating ATR: {e}")
    pause_terminal()

@require_exchange
def display_fibonacci_pivot_points_indicator() -> None:
    """📐 Displays Fibonacci Pivot Points for a given symbol."""
    params = get_indicator_params_from_user("Fibonacci Pivot Points", need_period=False)
    if not params: return

    symbol, _, timeframe = params # Period is None here
    try:
        # Need previous period's high/low/close for standard pivots
        # Fetching 2 candles allows using the *previous* candle's data
        ohlcv = fetch_ohlcv_data(symbol, timeframe, 2)
        if ohlcv is None or len(ohlcv) < 2:
            print(Fore.YELLOW + Style.BRIGHT + f"⚠️ Need at least 2 candles of data for {timeframe} pivots.")
            pause_terminal()
            return

        # Use the previous candle's data
        prev_high = float(ohlcv[0][2])
        prev_low = float(ohlcv[0][3])
        prev_close = float(ohlcv[0][4])

        pivots = FibonacciPivotPoints(prev_high, prev_low, prev_close) # Assuming function from indicators.py

        os.system("clear")
        print_indicator_header("FIBONACCI PIVOT POINTS")
        print(
            Fore.WHITE
            + Style.BRIGHT
            + f"\n📐 Fibonacci Pivots for {Fore.GREEN}{symbol}{Fore.WHITE} (based on previous {timeframe} candle):"
        )
        print(Fore.CYAN + "-" * 40)
        print_pivot_level("Resistance 3 (R3)", pivots['R3'], Fore.RED)
        print_pivot_level("Resistance 2 (R2)", pivots['R2'], Fore.RED)
        print_pivot_level("Resistance 1 (R1)", pivots['R1'], Fore.RED)
        print_pivot_level("Pivot (P)", pivots['Pivot'], Fore.YELLOW) # Pivot often yellow/white
        print_pivot_level("Support 1 (S1)", pivots['S1'], Fore.GREEN)
        print_pivot_level("Support 2 (S2)", pivots['S2'], Fore.GREEN)
        print_pivot_level("Support 3 (S3)", pivots['S3'], Fore.GREEN)
        print_indicator_footer()

    except Exception as e:
        print(Fore.RED + Style.BRIGHT + f"🔥 Error calculating Fibonacci Pivot Points: {e}")
    pause_terminal()

# --- Display Functions ---
def display_dataframe(title: str, df: pd.DataFrame, columns: list[str] | None = None,
                      column_names: list[str] | None = None, color: str = Fore.WHITE,
                      formatters: dict[str, Callable] | None = None) -> None:
    """Displays a Pandas DataFrame in a stylized format."""
    os.system("clear")
    print(
        Fore.CYAN
        + Style.BRIGHT
        + f"╔═══════ {Fore.GREEN}{title}{Fore.CYAN} ═══════╗" # Adjusted padding
    )
    if df.empty:
        print(Fore.YELLOW + "       --- No data to display ---")
    else:
        print(Fore.WHITE + Style.BRIGHT + "\n📜 Data:")
        df_to_print = df.copy()
        if columns:
            # Ensure only existing columns are selected
            existing_cols = [col for col in columns if col in df_to_print.columns]
            df_to_print = df_to_print[existing_cols]
        if column_names and len(column_names) == len(df_to_print.columns):
            df_to_print.columns = column_names

        # Apply formatting safely
        safe_formatters = {}
        if formatters:
            for col, fmt in formatters.items():
                if col in df_to_print.columns:
                    safe_formatters[col] = fmt

        # Use pandas option to control display width if needed
        # pd.set_option('display.width', 120) # Example
        print(Fore.CYAN + "-" * 100) # Adjust width as needed
        print(color + df_to_print.to_string(index=False, formatters=safe_formatters) + Fore.RESET)
        print(Fore.CYAN + "-" * 100)

def print_indicator_header(indicator_name: str) -> None:
    """Prints stylized header for indicator displays."""
    print(
        Fore.CYAN
        + Style.BRIGHT
        + f"╔═══════ {Fore.GREEN}{indicator_name}{Fore.CYAN} ═══════╗"
    )

def print_indicator_footer() -> None:
    """Prints stylized footer for indicator displays."""
    print(
        Fore.CYAN + Style.BRIGHT + "\n" + "-" * 40
    )

def print_pivot_level(level_name: str, level_value: float, color: str) -> None:
    """Prints a formatted pivot level."""
    print(f"  {Fore.WHITE}{level_name:<18}: {color}{level_value:.4f}{Style.RESET_ALL}")

def display_order_execution_message(order_type_text: str, symbol: str, side: str, amount: float, price: float | None = None, order_id: str | None = None) -> None:
    """Displays a stylized order execution/placement message."""
    os.system("clear")
    print(
        Fore.CYAN
        + Style.BRIGHT
        + f"╔═══════ {Fore.GREEN}{order_type_text}{Fore.CYAN} ═══════╗"
    )
    print(f"  {Fore.WHITE}🪙 Symbol: {Fore.GREEN}{symbol}{Style.RESET_ALL}")
    side_color = Fore.GREEN if side.lower() == 'buy' else Fore.RED
    print(f"  {Fore.WHITE}Side: {side_color}{side.upper()}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}⚖️ Amount: {Fore.GREEN}{amount}{Style.RESET_ALL}")
    if price is not None:
        print(f"  {Fore.WHITE}💰 Price: {Fore.GREEN}{price:.4f}{Style.RESET_ALL}")
    if order_id:
        print(f"  {Fore.WHITE}🆔 Order ID: {Fore.CYAN}{order_id}{Style.RESET_ALL}")
    print(Fore.CYAN + Style.BRIGHT + "╚" + "═" * (len(order_type_text) + 14) + "╝") # Footer

# --- Menu Display Functions ---
def display_main_menu() -> str:
    """📜 Displays the Main Menu."""
    os.system("clear")
    print(Fore.CYAN + Style.BRIGHT + "╔══════════════════════════════════╗")
    print(f"║   {Fore.MAGENTA}Bybit Futures Terminal v1.1{Fore.CYAN}    ║")
    print(f"║   {Fore.GREEN}  WorldGuides Edit - Arcane Edition {Fore.CYAN} ║")
    print(f"║       {Fore.YELLOW}✨ Forged by Pyrmethus ✨{Fore.CYAN}       ║")
    print(Fore.CYAN + Style.BRIGHT + "╚══════════════════════════════════╝")
    print(Fore.YELLOW + Style.BRIGHT + "\n🔮 Choose your path, seeker:")
    print(f"  {Fore.MAGENTA}1.{Fore.WHITE} Account Operations (账户操作)")
    print(f"  {Fore.MAGENTA}2.{Fore.WHITE} Market Data (市场数据)")
    print(f"  {Fore.MAGENTA}3.{Fore.WHITE} Trading Actions (交易操作)")
    print(f"  {Fore.MAGENTA}4.{Fore.WHITE} Order Management (订单管理)")
    print(f"  {Fore.MAGENTA}5.{Fore.WHITE} Display API Keys (Debug)")
    print(f"  {Fore.MAGENTA}6.{Fore.WHITE} Exit (退出)")
    return get_validated_input("🔮 Enter your choice (1-6): ", r"^[1-6]$")


def display_trading_menu() -> str:
    """⚔️ Displays the Trading Actions Submenu."""
    header = "Bybit Futures Trading Actions"
    options = {
        "1": "Place Market Order (市价单)",
        "2": "Place Limit Order (限价单)",
        "3": "Back to Main Menu (返回主菜单)"
    }
    prompt = "⚔️ Enter your choice (1-3): "
    valid_choices = r"^[1-3]$"
    subtitle = "(Using Direct Requests)"
    return display_menu_template(header, options, prompt, valid_choices, subtitle)

def display_order_management_menu() -> str:
    """🗂️ Displays the Order Management Submenu."""
    header = "Bybit Futures Order Management"
    options = {
        "1": "Cancel Single Order (Direct Requests)",
        "2": "Cancel All Orders (Mass Cancel - Requests)",
        "3": "Cancel Futures Order (CCXT)",
        "4": "View Open Futures Orders (CCXT)",
        "5": "View Open Futures Positions (CCXT)",
        "6": "Place Trailing Stop (Simulated) - WIP",
        "7": "Back to Main Menu"
    }
    prompt = "🗂️ Enter your choice (1-7): "
    valid_choices = r"^[1-7]$"
    subtitle = "(Using Both CCXT & Requests)"
    return display_menu_template(header, options, prompt, valid_choices, subtitle)

def display_account_menu() -> str:
    """🏦 Displays the Account Operations Submenu."""
    header = "Bybit Futures Account Operations"
    options = {
        "1": "View Account Balance",
        "2": "View Order History",
        "3": "Deposit Funds (Simulated) - WIP",
        "4": "Withdraw Funds (Simulated) - WIP",
        "5": "Back to Main Menu"
    }
    prompt = "🏦 Enter your choice (1-5): "
    valid_choices = r"^[1-5]$"
    subtitle = "(Using CCXT & Pandas)"
    return display_menu_template(header, options, prompt, valid_choices, subtitle)

def display_market_menu() -> str:
    """📊 Displays the Market Data Submenu."""
    header = "Bybit Futures Market Data"
    options = {
        "1": "Fetch Symbol Price",
        "2": "Get Order Book",
        "3": "List Available Symbols",
        "4": "Display RSI",
        "5": "Display ATR",
        "6": "Display Fibonacci Pivot Points",
        "7": "Display Funding Rates", # New Option
        "8": "Back to Main Menu"
    }
    prompt = "📊 Enter your choice (1-8): "
    valid_choices = r"^[1-8]$"
    subtitle = "(Using CCXT)"
    return display_menu_template(header, options, prompt, valid_choices, subtitle)

def display_menu_template(header: str, options: dict[str, str], prompt: str, valid_choices_regex: str, subtitle: str | None = None) -> str:
    """Generic function to display a formatted menu."""
    os.system("clear")
    width = 40
    print(Fore.CYAN + Style.BRIGHT + "╔" + "═" * (width-2) + "╗")
    print(f"║ {Fore.GREEN}{header:<{width-4}}{Fore.CYAN} ║")
    if subtitle:
        print(f"║ {Fore.YELLOW}{subtitle:<{width-4}}{Fore.CYAN} ║")
    print(Fore.CYAN + Style.BRIGHT + "╠" + "═" * (width-2) + "╣")
    for key, value in options.items():
        print(f"║ {Fore.MAGENTA}{key}. {Fore.WHITE}{value:<{width-6}}{Fore.CYAN} ║")
    print(Fore.CYAN + Style.BRIGHT + "╚" + "═" * (width-2) + "╝")
    return get_validated_input(prompt, valid_choices_regex)

# --- Utility Functions ---
def pause_terminal() -> None:
    """Pauses the terminal execution until user presses Enter."""
    print() # Add a newline for spacing
    input(Fore.YELLOW + Style.BRIGHT + "Press Enter to continue...")

def get_validated_input(prompt: str, validation_regex: str) -> str:
    """Gets user input and validates it against a regex, looping until valid."""
    while True:
        user_input = input(Fore.YELLOW + Style.BRIGHT + prompt + Style.RESET_ALL).strip()
        import re  # Import regex module here
        if re.match(validation_regex, user_input):
            return user_input
        else:
            print(Fore.RED + Style.BRIGHT + "🔥 Invalid input format. Please try again.")

def get_positive_float(prompt: str) -> float | None:
    """Gets positive float input from the user."""
    while True:
        try:
            value_str = input(Fore.YELLOW + Style.BRIGHT + prompt + Style.RESET_ALL).strip()
            value = float(value_str)
            if value > 0:
                return value
            else:
                print(Fore.RED + Style.BRIGHT + "🔥 Please enter a positive number.")
        except ValueError:
            print(Fore.RED + Style.BRIGHT + "🔥 Invalid number format. Please try again.")
        except EOFError: # Handle Ctrl+D
             print(Fore.RED + "\nInput cancelled.")
             return None

def get_symbol_input(prompt: str, allow_blank: bool = False) -> str | None:
    """Gets symbol input from user, converting to uppercase."""
    while True:
        symbol = input(Fore.YELLOW + Style.BRIGHT + prompt + Style.RESET_ALL).strip().upper()
        if symbol:
            # Basic validation: Check if it looks like a crypto pair (e.g., ends with USDT, BTC, etc.)
            # This is a loose check, CCXT/API will do the real validation
            if len(symbol) > 3: # Simple length check
                 # Optional: More specific regex like r"^[A-Z0-9]{2,}/?[A-Z]{2,}$"
                 return symbol
            else:
                 print(Fore.RED + Style.BRIGHT + "🔥 Symbol seems too short. Please enter a valid trading pair (e.g., BTCUSDT).")
        elif allow_blank:
            return None # Return None if blank is allowed
        else:
            print(Fore.RED + Style.BRIGHT + "🔥 Symbol cannot be empty.")

def get_order_details_from_user(order_type: str) -> tuple[str, str, float] | tuple[str, str, float, float] | None:
    """Collects and validates common order details from user."""
    symbol = get_symbol_input("🪙 Enter symbol (e.g., BTCUSDT): ")
    if not symbol: return None

    side = get_validated_input("Buy/Sell (向/卖): ", r"^(buy|sell)$")

    amount = get_positive_float("⚖️ Enter quantity: ")
    if amount is None: return None

    if order_type == "limit":
        price = get_positive_float("💰 Enter price: ")
        if price is None: return None
        return symbol, side, amount, price
    else: # Market order
        return symbol, side, amount

def get_order_id_symbol_from_user(action_name: str) -> tuple[str | None, str | None]:
    """Helper function to get order ID and optional symbol from user."""
    order_id = input(Fore.YELLOW + Style.BRIGHT + f"🆔 Enter Order ID for {action_name}: " + Style.RESET_ALL).strip()
    if not order_id:
        print(Fore.RED + Style.BRIGHT + "🔥 Order ID is required.")
        return None, None
    # V5 cancel usually needs symbol
    symbol = get_symbol_input(f"🪙 Enter symbol for Order ID {order_id} (e.g., BTCUSDT): ", allow_blank=False)
    return order_id, symbol

def get_confirmation(prompt: str) -> bool:
    """Gets a yes/no confirmation from the user."""
    while True:
        choice = input(Fore.YELLOW + Style.BRIGHT + prompt + " (yes/no): " + Style.RESET_ALL).lower().strip()
        if choice == 'yes':
            return True
        elif choice == 'no':
            return False
        else:
            print(Fore.RED + Style.BRIGHT + "🔥 Please enter 'yes' or 'no'.")

def display_invalid_choice_message() -> None:
    """Displays a generic invalid choice message."""
    print(
        Fore.RED
        + Style.BRIGHT
        + "🔥 Invalid choice. Please enter a valid option from the menu."
    )
    pause_terminal()

# --- Main Terminal Logic ---
def main() -> None:
    """🔮 Main function to run the Bybit Futures Terminal."""
    print(Fore.CYAN + "Initializing terminal...")
    if not initialize_exchange():
        print(Fore.YELLOW + Style.BRIGHT + "Proceeding with limited functionality (Market Data/Indicators may fail).")
        pause_terminal()

    # --- Main Menu Loop ---
    menu_actions = {
        "1": handle_account_menu,
        "2": handle_market_data_menu,
        "3": handle_trading_menu,
        "4": handle_order_management_menu,
        "5": debug_display_api_keys,
    }

    while True:
        choice_main = display_main_menu()
        if choice_main == "6":
            print(Fore.MAGENTA + Style.BRIGHT + "🚪 Exiting terminal... May your trades be ever in profit.")
            break

        action = menu_actions.get(choice_main)
        if action:
            action() # Call the appropriate handler function
        else:
            display_invalid_choice_message()

# --- Menu Handler Functions ---
def handle_account_menu() -> None:
    """Handles the Account Operations submenu."""
    actions = {
        "1": view_account_balance,
        "2": view_order_history,
        "3": lambda: print_wip("Simulated Deposit Feature"),
        "4": lambda: print_wip("Simulated Withdrawal Feature"),
    }
    while True:
        choice = display_account_menu()
        if choice == "5": break
        action = actions.get(choice)
        if action: action()
        else: display_invalid_choice_message()

def handle_market_data_menu() -> None:
    """Handles the Market Data submenu."""
    actions = {
        "1": fetch_symbol_price,
        "2": get_order_book,
        "3": list_available_symbols,
        "4": display_rsi_indicator,
        "5": display_atr_indicator,
        "6": display_fibonacci_pivot_points_indicator,
        "7": display_funding_rates, # New action
    }
    while True:
        choice = display_market_menu()
        if choice == "8": break # Updated exit option
        action = actions.get(choice)
        if action: action()
        else: display_invalid_choice_message()

def handle_trading_menu() -> None:
    """Handles the Trading Actions submenu."""
    actions = {
        "1": place_market_order_requests,
        "2": place_limit_order_requests,
    }
    while True:
        choice = display_trading_menu()
        if choice == "3": break
        action = actions.get(choice)
        if action: action()
        else: display_invalid_choice_message()

def handle_order_management_menu() -> None:
    """Handles the Order Management submenu."""
    actions = {
        "1": cancel_single_order_requests,
        "2": cancel_all_orders_requests,
        "3": cancel_futures_order_ccxt,
        "4": view_open_futures_orders_ccxt,
        "5": view_open_futures_positions_ccxt,
        "6": lambda: print_wip("Trailing Stop (Simulated)"),
    }
    while True:
        choice = display_order_management_menu()
        if choice == "7": break
        action = actions.get(choice)
        if action: action()
        else: display_invalid_choice_message()

def print_wip(feature_name: str) -> None:
    """Prints a standard Work-In-Progress message."""
    print(Fore.YELLOW + Style.BRIGHT + f"🚧 {feature_name} - Work In Progress")
    pause_terminal()

# --- Entry Point ---
if __name__ == "__main__":
    # 📜 Check for indicators.py (optional, as pandas_ta is now the primary check)
    if not os.path.exists("indicators.py"):
        print(Fore.YELLOW + Style.BRIGHT + "⚠️ indicators.py not found. Ensure technical indicator functions are available if needed.")
        # Optionally create a placeholder if strictly required by imports, but pandas_ta check is more critical
        # with open("indicators.py", 'w') as f: f.write("# Placeholder\nimport pandas as pd\n...")

    main()
