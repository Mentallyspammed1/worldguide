# File: main.py
# -*- coding: utf-8 -*-

"""
Main Demonstration Script for Bybit V5 API Helper Modules

This script showcases the usage of various helper functions designed to interact
with the Bybit V5 API via the CCXT library. It covers setup, account information,
market data retrieval, order management, and position management.

**WARNING:** Order placement tests should ONLY be run on TESTNET accounts.
Ensure TESTNET_MODE is True in your configuration before enabling order tests.
"""

import logging
import os
import sys
import time
from typing import Optional, Dict, Any
from decimal import Decimal
from functools import wraps # Needed for retry decorator example

# --- Standard Library Imports ---
# (Already included above)

# --- Third-party Imports ---
try:
    import ccxt
    from ccxt.base.errors import (
        NetworkError, RateLimitExceeded, ExchangeNotAvailable,
        AuthenticationError, BadSymbol, InsufficientFunds, OrderNotFound,
        InvalidOrder
    ) # Import specific exceptions for better handling
except ImportError:
    print("FATAL ERROR: CCXT library not found. Please install it: pip install ccxt")
    sys.exit(1)

try:
    # Colorama is optional for colored console output
    from colorama import init, Fore, Style, Back
    init(autoreset=True) # Initialize colorama to automatically reset colors
except ImportError:
    print("Warning: 'colorama' library not found. Logs will not be colored. Install using: pip install colorama")
    # Define dummy color constants if colorama is not available
    class DummyColor:
        def __getattr__(self, name: str) -> str: return "" # Return empty string for any attribute
    Fore = Style = Back = DummyColor()

try:
    # Dotenv is optional for loading environment variables from a .env file
    from dotenv import load_dotenv
    if load_dotenv():
        print(f"{Fore.CYAN}Successfully loaded environment variables from '.env' file.{Style.RESET_ALL}")
    else:
        print(f"{Fore.CYAN}No '.env' file found or it's empty. Relying on system environment variables.{Style.RESET_ALL}")
except ImportError:
    print(f"{Fore.YELLOW}Warning: 'python-dotenv' library not found. Relying on system environment variables only. Install using: pip install python-dotenv{Style.RESET_ALL}")

# --- Local Module Imports ---
# Import necessary components from the organized helper modules
try:
    from config import Config
    from utils import (validate_market, safe_decimal_conversion, format_price,
                       format_amount, format_order_id, send_sms_alert, retry_api_call) # Basic utilities/placeholders
    from exchange_setup import initialize_bybit
    from account_config import (set_leverage, set_position_mode_bybit_v5,
                               fetch_account_info_bybit_v5, set_isolated_margin_bybit_v5)
    from market_data import (fetch_usdt_balance, fetch_ohlcv_paginated, fetch_funding_rate,
                             fetch_l2_order_book_validated, fetch_ticker_validated, fetch_recent_trades)
    from margin_calculation import calculate_margin_requirement
    from order_management import (place_market_order_slippage_check, cancel_all_orders, place_limit_order_tif,
                                  place_native_stop_loss, fetch_open_orders_filtered,
                                  place_native_trailing_stop, update_limit_order)
    from position_management import (get_current_position_bybit_v5, close_position_reduce_only,
                                     fetch_position_risk_bybit_v5)
except ImportError as e:
    print(f"{Back.RED}{Fore.WHITE}FATAL ERROR: Failed to import local helper module: {e}{Style.RESET_ALL}")
    print("Ensure all required Python files (config.py, utils.py, etc.) are present in the same directory or accessible in PYTHONPATH.")
    sys.exit(1)


# --- Logger Setup ---
# Configure a central logger for the application
log_level = logging.DEBUG # Use DEBUG for development, INFO or WARNING for production
# Create logger instance (using __name__ is standard practice)
main_logger = logging.getLogger(__name__)
main_logger.setLevel(log_level)
# Prevent duplicate handlers if this script is reloaded or run multiple times
if not main_logger.hasHandlers():
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    # Create formatter and add it to the handler
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] {%(name)s:%(funcName)s:%(lineno)d} - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    # Add the handler to the logger
    main_logger.addHandler(handler)

main_logger.info(f"{Fore.YELLOW}{Style.BRIGHT}--- Main Script Execution Started ---{Style.RESET_ALL}")

# --- Placeholder Implementation Notes ---
# The imported helper modules might rely on placeholders (like `retry_api_call` in utils.py).
# For robust applications, replace placeholders with real implementations.
# Example: Real Retry Decorator (adapt as needed)
# def real_retry_api_call(max_retries=3, initial_delay=1.0, backoff_factor=2.0,
#                         retry_exceptions=(NetworkError, RateLimitExceeded, ExchangeNotAvailable)):
#     """Decorator for retrying API calls with exponential backoff."""
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             retries = 0
#             delay = initial_delay
#             while retries <= max_retries:
#                 try:
#                     return func(*args, **kwargs)
#                 except retry_exceptions as e:
#                     retries += 1
#                     if retries > max_retries:
#                         main_logger.error(f"API call '{func.__name__}' failed after {max_retries} retries: {e}")
#                         raise # Re-raise the last exception
#                     main_logger.warning(f"API call '{func.__name__}' failed (Retry {retries}/{max_retries}): {e}. Retrying in {delay:.2f}s...")
#                     time.sleep(delay)
#                     delay *= backoff_factor # Exponential backoff
#                 except Exception as e: # Catch other unexpected errors
#                     main_logger.error(f"API call '{func.__name__}' failed with unexpected error: {e}", exc_info=True)
#                     raise # Re-raise unexpected errors immediately
#         return wrapper
#     return decorator
# # To use the real decorator, uncomment the following line:
# # retry_api_call = real_retry_api_call() # This overrides the imported placeholder

# --- Configuration Loading ---
main_logger.info("Loading application configuration...")
try:
    config = Config()
    # --- CRITICAL OVERRIDES FOR SAFETY ---
    # Ensure Testnet mode is explicitly set for this demonstration script
    config.TESTNET_MODE = False
    main_logger.warning(f"{Back.YELLOW}{Fore.BLACK}ENSURING TESTNET MODE IS ACTIVE: {config.TESTNET_MODE}{Style.RESET_ALL}")

    # Set a default symbol for testing (can be overridden by environment variables)
    config.SYMBOL = os.getenv("BYBIT_DEMO_SYMBOL", "BTC/USDT:USDT")
    config.DEFAULT_LEVERAGE = int(os.getenv("BYBIT_DEMO_LEVERAGE", 5)) # Lower leverage for safety
    config.ENABLE_SMS_ALERTS = os.getenv("ENABLE_SMS_ALERTS", "False").lower() == "true" # Keep SMS off by default

    main_logger.info(f"Configuration Loaded: Testnet={config.TESTNET_MODE}, Symbol={config.SYMBOL}, Default Leverage={config.DEFAULT_LEVERAGE}, SMS Alerts={config.ENABLE_SMS_ALERTS}")

    # Validate essential credentials
    if not config.API_KEY or not config.API_SECRET:
        main_logger.critical(f"{Back.RED}{Fore.WHITE}FATAL: Bybit API Key or Secret not found.{Style.RESET_ALL}")
        main_logger.critical("Please set BYBIT_API_KEY and BYBIT_API_SECRET environment variables or in a '.env' file.")
        sys.exit(1)
    main_logger.debug("API credentials found.")

except Exception as e:
    main_logger.critical(f"Failed to initialize configuration: {e}", exc_info=True)
    sys.exit(1)

# --- Helper Function ---
# Required for V5 category parameter in some calls, especially order management
def _get_v5_category(market_info: Optional[Dict[str, Any]]) -> Optional[str]:
    """Determines the Bybit V5 category based on market info."""
    if not market_info:
        main_logger.error("Cannot determine V5 category: Market info is missing.")
        return None
    market_type = market_info.get('type')
    is_linear = market_info.get('linear', False)
    is_inverse = market_info.get('inverse', False)

    if market_type == 'spot':
        return 'spot'
    elif market_type in ['swap', 'future']:
        if is_linear:
            return 'linear'
        elif is_inverse:
            return 'inverse'
    elif market_type == 'option':
         return 'option'

    main_logger.warning(f"Could not determine V5 category for market type '{market_type}' (Linear: {is_linear}, Inverse: {is_inverse}).")
    return None # Default or indicate failure

# --- Main Execution Block ---
exchange_instance: Optional[ccxt.bybit] = None
market_info: Optional[Dict[str, Any]] = None
v5_category: Optional[str] = None

try:
    # === Step 1: Initialize Exchange ===
    main_logger.info(f"\n{Style.BRIGHT}--- Step 1: Initializing Bybit Exchange ---{Style.RESET_ALL}")
    exchange_instance = initialize_bybit(config)
    if not exchange_instance:
        # Error logged within initialize_bybit
        raise RuntimeError("Exchange initialization failed.")
    main_logger.info(f"{Fore.GREEN}Exchange Initialized OK: {exchange_instance.id} (Testnet: {config.TESTNET_MODE}){Style.RESET_ALL}")

    # === Step 2: Validate Market ===
    main_logger.info(f"\n{Style.BRIGHT}--- Step 2: Validating Market: {config.SYMBOL} ---{Style.RESET_ALL}")
    market_info = validate_market(exchange_instance, config.SYMBOL, config)
    if not market_info:
        raise BadSymbol(f"Market validation failed for symbol {config.SYMBOL}. Ensure it's valid and active on Bybit.")
    main_logger.info(f"{Fore.GREEN}Market Validation OK for {config.SYMBOL}. Type: {market_info.get('type')}, Contract Type: {market_info.get('contractType', 'N/A')}, Linear/Inverse: {'Linear' if market_info.get('linear') else ('Inverse' if market_info.get('inverse') else 'N/A')}{Style.RESET_ALL}")
    # Determine V5 Category early
    v5_category = _get_v5_category(market_info)
    if not v5_category:
         main_logger.warning(f"{Fore.YELLOW}Could not determine V5 API category for {config.SYMBOL}. Some API calls might require manual adjustment.{Style.RESET_ALL}")
    else:
         main_logger.info(f"Determined V5 API Category: '{v5_category}'")


    # === Step 3: Fetch Balance ===
    main_logger.info(f"\n{Style.BRIGHT}--- Step 3: Fetching USDT Balance ---{Style.RESET_ALL}")
    # Assuming USDT is the collateral currency for linear contracts
    equity, available = fetch_usdt_balance(exchange_instance, config)
    if equity is None or available is None:
        # The retry decorator (if active) should handle transient errors.
        # If it still returns None, it's a persistent issue.
        main_logger.warning(f"{Fore.YELLOW}Fetch Balance failed after potential retries (returned None). Proceeding cautiously, but balance-dependent actions might fail.{Style.RESET_ALL}")
        # Set defaults to avoid crashing later tests, but log the issue
        equity, available = Decimal("0.0"), Decimal("0.0")
    else:
        main_logger.info(f"{Fore.GREEN}Fetch Balance OK: Total Equity={format_price(equity, config.QUOTE_PRECISION)} USDT, Available Balance={format_price(available, config.QUOTE_PRECISION)} USDT{Style.RESET_ALL}")

    # === Step 4: Set Leverage (if applicable) ===
    if v5_category in ['linear', 'inverse']: # Leverage applies to derivatives
        main_logger.info(f"\n{Style.BRIGHT}--- Step 4: Setting Leverage ---{Style.RESET_ALL}")
        lev_ok = set_leverage(exchange_instance, config.SYMBOL, config.DEFAULT_LEVERAGE, config, market_info=market_info)
        if not lev_ok:
            # This might fail legitimately on some account types (e.g., UTA Isolated mode requires per-position leverage)
            # The function itself logs detailed reasons.
            main_logger.warning(f"{Fore.YELLOW}Set Leverage to {config.DEFAULT_LEVERAGE}x potentially failed or was unnecessary (e.g., UTA Isolated Mode). Check logs.{Style.RESET_ALL}")
        else:
            main_logger.info(f"{Fore.GREEN}Set Leverage request successful (or leverage already set) to {config.DEFAULT_LEVERAGE}x for {config.SYMBOL}.{Style.RESET_ALL}")
    else:
        main_logger.info(f"\n{Style.BRIGHT}--- Step 4: Skipping Leverage Setting (Not applicable for '{v5_category}' category) ---{Style.RESET_ALL}")


    # === Step 5: Fetch Ticker ===
    main_logger.info(f"\n{Style.BRIGHT}--- Step 5: Fetching Ticker Information ---{Style.RESET_ALL}")
    ticker = fetch_ticker_validated(exchange_instance, config.SYMBOL, config, market_info=market_info)
    if not ticker:
        main_logger.warning(f"{Fore.YELLOW}Fetch Ticker failed after potential retries.{Style.RESET_ALL}")
    else:
        spread = ticker.get('ask', 0) - ticker.get('bid', 0)
        spread_pct = (spread / ticker['last'] * 100) if ticker.get('last') and ticker['last'] > 0 else 0
        main_logger.info(f"{Fore.GREEN}Fetch Ticker OK: Last={ticker.get('last')}, Bid={ticker.get('bid')}, Ask={ticker.get('ask')}, Spread={spread:.{config.PRICE_PRECISION}f} ({spread_pct:.4f}%){Style.RESET_ALL}")

    # === Step 6: Fetch Current Position (if applicable) ===
    if v5_category in ['linear', 'inverse']:
        main_logger.info(f"\n{Style.BRIGHT}--- Step 6: Fetching Current Position ---{Style.RESET_ALL}")
        position = get_current_position_bybit_v5(exchange_instance, config.SYMBOL, config, market_info=market_info)
        # This function returns a default dict if no position, check the 'side'
        if position['side'] == config.POS_NONE:
            main_logger.info(f"{Fore.CYAN}Fetch Position OK: No active position found for {config.SYMBOL}.{Style.RESET_ALL}")
        else:
            main_logger.info(f"{Fore.GREEN}Fetch Position OK: Side={position['side']}, Qty={position['qty']}, Entry Price={position['entry_price']}, Liq Price={position.get('liq_price', 'N/A')}{Style.RESET_ALL}")

        # === Step 7: Fetch Position Risk (only if position exists) ===
        if position['side'] != config.POS_NONE:
            main_logger.info(f"\n{Style.BRIGHT}--- Step 7: Fetching Position Risk Details ---{Style.RESET_ALL}")
            risk = fetch_position_risk_bybit_v5(exchange_instance, config.SYMBOL, config, market_info=market_info)
            if not risk:
                main_logger.warning(f"{Fore.YELLOW}Fetch Position Risk failed or position closed between checks.{Style.RESET_ALL}")
            else:
                 # Format percentages nicely
                imr_pct = f"{risk['imr'] * 100:.2f}%" if risk.get('imr') is not None else 'N/A'
                mmr_pct = f"{risk['mmr'] * 100:.2f}%" if risk.get('mmr') is not None else 'N/A'
                main_logger.info(f"{Fore.GREEN}Fetch Position Risk OK: Liq Price={risk.get('liq_price', 'N/A')}, IMR={imr_pct}, MMR={mmr_pct}, Value={risk.get('position_value', 'N/A')}{Style.RESET_ALL}")
        else:
            main_logger.info(f"\n{Style.BRIGHT}--- Step 7: Skipping Fetch Position Risk (No position) ---{Style.RESET_ALL}")
    else:
        main_logger.info(f"\n{Style.BRIGHT}--- Steps 6 & 7: Skipping Position Fetch/Risk (Not applicable for '{v5_category}' category) ---{Style.RESET_ALL}")
        position = {'side': config.POS_NONE, 'qty': Decimal(0)} # Define dummy position for later checks

    # === Optional: More Advanced Tests ===
    run_advanced_tests = True # Set to False to skip these non-essential demos

    if run_advanced_tests:
        main_logger.info(f"\n{Style.BRIGHT}--- Running Advanced Functionality Tests ---{Style.RESET_ALL}")

        # Fetch Account Info (UTA Status, Margin Mode etc.)
        main_logger.info("\n--- Adv 1. Fetching Account Info ---")
        acc_info = fetch_account_info_bybit_v5(exchange_instance, config)
        if acc_info:
            margin_mode = acc_info.get('marginMode', 'N/A') # REGULAR / PORTFOLIO
            uta_status = acc_info.get('unifiedMarginStatus', 'N/A') # 1: Regular, 2: UTA Pro, 3/4: UTA Gradual
            main_logger.info(f"{Fore.GREEN}Fetch Account Info OK: UTA Status={uta_status}, Margin Mode={margin_mode}{Style.RESET_ALL}")
        else:
            main_logger.warning(f"{Fore.YELLOW}Fetch Account Info failed.{Style.RESET_ALL}")


        # Fetch OHLCV Data
        main_logger.info("\n--- Adv 2. Fetching OHLCV Data ---")
        # Fetch last 100 hourly candles for the symbol
        ohlcv_df = fetch_ohlcv_paginated(exchange_instance, config.SYMBOL, '1h', config, max_total_candles=100, market_info=market_info)
        if ohlcv_df is not None and not ohlcv_df.empty:
            main_logger.info(f"{Fore.GREEN}Fetch OHLCV OK: Fetched {len(ohlcv_df)} candles. Last Close: {ohlcv_df['close'].iloc[-1]} @ {ohlcv_df.index[-1]}{Style.RESET_ALL}")
        elif ohlcv_df is not None and ohlcv_df.empty:
             main_logger.info(f"{Fore.CYAN}Fetch OHLCV OK: No candles returned for the specified period/symbol.{Style.RESET_ALL}")
        else:
            main_logger.warning(f"{Fore.YELLOW}Fetch OHLCV failed.{Style.RESET_ALL}")

        # Fetch Funding Rate (if applicable)
        if v5_category in ['linear', 'inverse'] and market_info.get('type') == 'swap':
             main_logger.info("\n--- Adv 3. Fetching Funding Rate ---")
             funding_rate_info = fetch_funding_rate(exchange_instance, config.SYMBOL, config, market_info=market_info)
             if funding_rate_info:
                 rate = funding_rate_info.get('fundingRate')
                 timestamp = funding_rate_info.get('fundingRateTimestamp')
                 dt_object = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(timestamp / 1000)) if timestamp else 'N/A'
                 main_logger.info(f"{Fore.GREEN}Fetch Funding Rate OK: Rate={rate:.8f}, Timestamp={dt_object} UTC{Style.RESET_ALL}")
             else:
                 main_logger.warning(f"{Fore.YELLOW}Fetch Funding Rate failed.{Style.RESET_ALL}")
        else:
             main_logger.info("\n--- Adv 3. Skipping Funding Rate (Not applicable for this market) ---")


    # === Order Placement Tests (USE WITH EXTREME CAUTION - TESTNET ONLY) ===
    # <<< Set run_order_tests to True ONLY if you are SURE you are on TESTNET >>>
    run_order_tests = False # Default to False for safety
    # run_order_tests = True # Uncomment to enable tests (AFTER VERIFYING TESTNET)

    if run_order_tests and not config.TESTNET_MODE:
        main_logger.error(f"{Back.RED}{Fore.WHITE}CRITICAL SAFETY CHECK FAILED: Order tests enabled but TESTNET_MODE is FALSE.{Style.RESET_ALL}")
        main_logger.error("Aborting script to prevent accidental live trading.")
        sys.exit(1)
    elif run_order_tests and config.TESTNET_MODE:
        main_logger.warning(f"\n{Back.MAGENTA}{Fore.WHITE}{Style.BRIGHT}--- Running Order Placement Tests (TESTNET ACTIVE) ---{Style.RESET_ALL}")
        main_logger.warning("These tests will place and cancel orders. Ensure sufficient testnet funds.")
        time.sleep(3) # Pause for user awareness

        if not v5_category:
             main_logger.error(f"{Back.RED}Cannot run order tests: V5 category for {config.SYMBOL} could not be determined. Aborting.{Style.RESET_ALL}")
             sys.exit(1)

        # --- Pre-Test Cleanup: Ensure Flat Position ---
        main_logger.info("\n--- Pre-Test: Ensuring Flat Position ---")
        initial_pos = get_current_position_bybit_v5(exchange_instance, config.SYMBOL, config, market_info=market_info)
        if initial_pos['side'] != config.POS_NONE:
             main_logger.warning(f"Existing {initial_pos['side']} position found (Qty: {initial_pos['qty']}). Attempting market close...")
             close_result = close_position_reduce_only(exchange_instance, config.SYMBOL, config, position_to_close=initial_pos, reason="Test Prep Close", market_info=market_info)
             if close_result:
                 main_logger.info(f"Market close order placed (ID: ...{format_order_id(close_result.get('id'))}). Waiting for position update...")
                 time.sleep(5) # Wait for closure to likely process
             else:
                 main_logger.error(f"{Back.RED}Failed to place market close order for pre-existing position.{Style.RESET_ALL}")
                 # Attempt to cancel all orders as a fallback cleanup
                 cancel_all_orders(exchange_instance, config.SYMBOL, config, reason="Pre-Test Cleanup", market_info=market_info)
                 sys.exit(1) # Abort if cleanup fails initially

             # Verify closure
             current_pos_check = get_current_position_bybit_v5(exchange_instance, config.SYMBOL, config, market_info=market_info)
             if current_pos_check['side'] != config.POS_NONE:
                  main_logger.error(f"{Back.RED}Failed to close pre-existing position (still {current_pos_check['side']} {current_pos_check['qty']}). Aborting order tests.{Style.RESET_ALL}")
                  sys.exit(1)
             else:
                  main_logger.info(f"{Fore.GREEN}Pre-existing position closed successfully.{Style.RESET_ALL}")
        else:
             main_logger.info(f"{Fore.CYAN}Position confirmed flat. Proceeding with order tests.{Style.RESET_ALL}")
        # Also cancel any lingering orders before starting new tests
        main_logger.info("Cancelling any existing open orders...")
        cancel_all_orders(exchange_instance, config.SYMBOL, config, reason="Pre-Test Cleanup", market_info=market_info)
        time.sleep(1)


        # --- Test 8a: Place Limit Order (PostOnly) ---
        main_logger.info("\n--- Test 8a. Place Limit Order (GTC, PostOnly) ---")
        current_ticker = fetch_ticker_validated(exchange_instance, config.SYMBOL, config, market_info=market_info)
        if current_ticker and current_ticker.get('bid') and current_ticker['bid'] > 0:
            # Place a buy limit order 0.5% below the current bid price
            limit_buy_price = current_ticker['bid'] * Decimal("0.995")
            limit_qty = safe_decimal_conversion(config.MIN_ORDER_QTY_BTC) # Use configured minimum quantity
            if not limit_qty: limit_qty = Decimal("0.001") # Fallback if config missing
            main_logger.info(f"Attempting to place LIMIT BUY: Qty={limit_qty}, Price={format_price(limit_buy_price, config.PRICE_PRECISION)}")

            limit_order = place_limit_order_tif(exchange_instance, config.SYMBOL, 'buy', limit_qty, limit_buy_price, config,
                                                time_in_force='GTC', is_post_only=True, market_info=market_info)
            if limit_order:
                limit_order_id = limit_order.get('id')
                main_logger.info(f"{Fore.GREEN}Limit Order (PostOnly) placed OK: ID ...{format_order_id(limit_order_id)}, Status: {limit_order.get('status')}{Style.RESET_ALL}")
                time.sleep(1) # Short pause

                # --- Test 8b: Fetch Open Limit Order ---
                main_logger.info("\n--- Test 8b. Fetch Open Limit Order (Filtered) ---")
                open_limit_orders = fetch_open_orders_filtered(exchange_instance, config.SYMBOL, config, side='buy', order_type='limit', order_filter='Order', market_info=market_info)
                found_order = False
                if open_limit_orders:
                    for o in open_limit_orders:
                        if o.get('id') == limit_order_id:
                            main_logger.info(f"{Fore.GREEN}Successfully fetched the specific open limit order: ID ...{format_order_id(o.get('id'))}{Style.RESET_ALL}")
                            found_order = True
                            break
                if not found_order:
                     main_logger.warning(f"{Fore.YELLOW}Could not fetch the specific open limit order (might have filled instantly, been rejected post-placement, or filter failed).{Style.RESET_ALL}")

                # --- Test 8c: Cancel All Orders ---
                main_logger.info("\n--- Test 8c. Cancel All Orders (Should cancel the limit order) ---")
                cancel_ok = cancel_all_orders(exchange_instance, config.SYMBOL, config, reason="Test Cancel Limit", market_info=market_info)
                if cancel_ok:
                     main_logger.info(f"{Fore.GREEN}Cancel All Orders request executed successfully.{Style.RESET_ALL}")
                else:
                     main_logger.warning(f"{Fore.YELLOW}Cancel All Orders reported potential issues or no orders to cancel. Check logs.{Style.RESET_ALL}")
                time.sleep(2) # Wait for cancellation to process

                # Verify cancellation
                open_orders_after_cancel = fetch_open_orders_filtered(exchange_instance, config.SYMBOL, config, order_filter='Order', market_info=market_info)
                if not open_orders_after_cancel:
                     main_logger.info(f"{Fore.GREEN}Verified no open regular orders remain after cancellation.{Style.RESET_ALL}")
                else:
                     main_logger.warning(f"{Fore.YELLOW}Found {len(open_orders_after_cancel)} open regular orders after cancel call! Order IDs: {[o.get('id') for o in open_orders_after_cancel]}{Style.RESET_ALL}")

            else:
                 main_logger.warning(f"{Fore.YELLOW}Limit Order placement failed or was rejected (e.g., PostOnly failed due to price crossing spread). Check logs.{Style.RESET_ALL}")
        else:
            main_logger.error("Cannot get valid ticker bid price for limit order test. Skipping.")


        # --- Test 9a: Place Market Order (Entry) ---
        main_logger.info("\n--- Test 9a. Place Market Order (Entry) ---")
        entry_qty = safe_decimal_conversion(config.MIN_ORDER_QTY_BTC) # Use configured minimum
        if not entry_qty: entry_qty = Decimal("0.001") # Fallback
        main_logger.info(f"Attempting to place MARKET BUY: Qty={entry_qty}")

        market_entry_order = place_market_order_slippage_check(exchange_instance, config.SYMBOL, 'buy', entry_qty, config, market_info=market_info)
        if not market_entry_order:
            main_logger.error(f"{Back.RED}Market Entry Order Placement Failed. Aborting further dependent tests.{Style.RESET_ALL}")
        else:
            entry_order_id = market_entry_order.get('id')
            main_logger.info(f"{Fore.GREEN}Market Entry Order OK: ID ...{format_order_id(entry_order_id)}, Status: {market_entry_order.get('status')}, Avg Fill Price: {market_entry_order.get('average')}{Style.RESET_ALL}")
            main_logger.info("Waiting for position update...")
            time.sleep(5) # Wait for position to likely establish

            # Verify position opened
            pos_after_entry = get_current_position_bybit_v5(exchange_instance, config.SYMBOL, config, market_info=market_info)
            # Check if side is LONG and quantity is close to expected (allowing for fees/slippage)
            if pos_after_entry['side'] != config.POS_LONG or pos_after_entry['qty'] < entry_qty * Decimal("0.9"):
                main_logger.error(f"{Back.RED}Position check after market buy failed! Expected LONG ~{entry_qty}, Got: Side={pos_after_entry['side']}, Qty={pos_after_entry['qty']}. Aborting further tests.{Style.RESET_ALL}")
            else:
                main_logger.info(f"{Fore.GREEN}Position Confirmed: LONG {pos_after_entry['qty']} @ Entry ~{pos_after_entry['entry_price']}{Style.RESET_ALL}")
                entry_price_decimal = safe_decimal_conversion(pos_after_entry['entry_price'])

                # --- Test 9b: Place Native Stop Loss ---
                if entry_price_decimal and entry_price_decimal > 0:
                    main_logger.info("\n--- Test 9b. Place Native Stop Loss ---")
                    # Set SL 5% below entry price for test
                    sl_price = entry_price_decimal * Decimal("0.95")
                    sl_qty = pos_after_entry['qty'] # Close the full position on SL
                    main_logger.info(f"Attempting to place NATIVE STOP LOSS (SELL): Qty={sl_qty}, Trigger Price={format_price(sl_price, config.PRICE_PRECISION)}")

                    sl_order = place_native_stop_loss(exchange_instance, config.SYMBOL, 'sell', sl_qty, sl_price, config, trigger_by='MarkPrice', market_info=market_info)
                    if sl_order:
                        sl_order_id = sl_order.get('id')
                        main_logger.info(f"{Fore.GREEN}Native SL Order OK: ID ...{format_order_id(sl_order_id)}, Status: {sl_order.get('status')}, Trigger: {format_price(sl_price, config.PRICE_PRECISION)}{Style.RESET_ALL}")
                        time.sleep(1)

                        # Verify SL order exists using fetch_open_orders_filtered with 'StopOrder'
                        open_stops = fetch_open_orders_filtered(exchange_instance, config.SYMBOL, config, order_filter='StopOrder', market_info=market_info)
                        found_sl = any(o.get('id') == sl_order_id for o in open_stops) if open_stops else False
                        if found_sl:
                             main_logger.info(f"{Fore.GREEN}Verified SL order exists via fetch (StopOrder filter).{Style.RESET_ALL}")
                        else:
                             main_logger.warning(f"{Fore.YELLOW}Could not verify SL order via fetch (StopOrder filter). Might have triggered or API delay.{Style.RESET_ALL}")

                        # --- Test 9c: Cancel Specific Stop Order ---
                        # NOTE: CCXT's cancel_all_orders might not cancel stops depending on exchange/params.
                        # Using explicit cancel_order is safer for specific stops.
                        main_logger.info(f"\n--- Test 9c. Cancel Specific Stop Order (ID: ...{format_order_id(sl_order_id)}) ---")
                        try:
                             # V5 requires category for order modification/cancellation
                             cancel_params = {'category': v5_category}
                             main_logger.info(f"Attempting exchange.cancel_order with ID: {sl_order_id}, Symbol: {config.SYMBOL}, Params: {cancel_params}")
                             cancel_response = exchange_instance.cancel_order(sl_order_id, config.SYMBOL, params=cancel_params)
                             main_logger.info(f"{Fore.GREEN}Specific SL order cancellation request sent. Response: {cancel_response}{Style.RESET_ALL}")
                             # Add verification step if possible
                             time.sleep(2)
                             open_stops_after_cancel = fetch_open_orders_filtered(exchange_instance, config.SYMBOL, config, order_filter='StopOrder', market_info=market_info)
                             found_sl_after_cancel = any(o.get('id') == sl_order_id for o in open_stops_after_cancel) if open_stops_after_cancel else False
                             if not found_sl_after_cancel:
                                 main_logger.info(f"{Fore.GREEN}Verified SL order no longer exists after cancellation attempt.{Style.RESET_ALL}")
                             else:
                                 main_logger.warning(f"{Fore.YELLOW}SL order ...{format_order_id(sl_order_id)} still found after cancellation attempt!{Style.RESET_ALL}")

                        except OrderNotFound:
                             main_logger.warning(f"{Fore.YELLOW}Specific SL order ID ...{format_order_id(sl_order_id)} not found during cancellation (already triggered, cancelled, or incorrect ID?).{Style.RESET_ALL}")
                        except (InvalidOrder, ccxt.ExchangeError) as cancel_err:
                             main_logger.error(f"Failed to cancel specific SL order ID ...{format_order_id(sl_order_id)}: {cancel_err}", exc_info=True)
                        except Exception as generic_cancel_err:
                             main_logger.error(f"Unexpected error cancelling specific SL order: {generic_cancel_err}", exc_info=True)

                    else:
                        main_logger.warning(f"{Fore.YELLOW}Native Stop Loss Placement Failed. Check logs.{Style.RESET_ALL}")
                else:
                    main_logger.error("Could not place Stop Loss: Invalid entry price detected after market buy.")


                # --- Test 9d: Close Position (Market ReduceOnly) ---
                main_logger.info("\n--- Test 9d. Close Position (Market ReduceOnly) ---")
                # Fetch position again right before closing, in case SL triggered or state changed
                pos_before_close = get_current_position_bybit_v5(exchange_instance, config.SYMBOL, config, market_info=market_info)
                if pos_before_close['side'] == config.POS_NONE:
                     main_logger.info(f"{Fore.CYAN}Position appears already closed before final close attempt. Skipping.{Style.RESET_ALL}")
                else:
                    main_logger.info(f"Attempting to close {pos_before_close['side']} position of Qty {pos_before_close['qty']} via Market ReduceOnly order.")
                    close_order = close_position_reduce_only(exchange_instance, config.SYMBOL, config, position_to_close=pos_before_close, reason="Test Close", market_info=market_info)
                    if close_order:
                        main_logger.info(f"{Fore.GREEN}Close Position Order OK: ID ...{format_order_id(close_order.get('id'))}, Status: {close_order.get('status')}{Style.RESET_ALL}")
                    else:
                        # The function logs details on failure
                        main_logger.warning(f"{Fore.YELLOW}Close Position attempt failed or reported issues. Position might still be open or was already closed.{Style.RESET_ALL}")

                # Final position check after attempting closure
                main_logger.info("Waiting briefly for final position check...")
                time.sleep(3)
                final_pos_check = get_current_position_bybit_v5(exchange_instance, config.SYMBOL, config, market_info=market_info)
                if final_pos_check['side'] == config.POS_NONE:
                    main_logger.info(f"{Fore.GREEN}{Style.BRIGHT}Position Confirmed Flat After All Tests.{Style.RESET_ALL}")
                else:
                     main_logger.error(f"{Back.RED}{Fore.WHITE}POSITION IS NOT FLAT AFTER TESTS! Side: {final_pos_check['side']}, Qty: {final_pos_check['qty']}{Style.RESET_ALL}")
                     # Attempt one last forceful cleanup
                     main_logger.warning("Attempting final market close and order cancellation...")
                     close_position_reduce_only(exchange_instance, config.SYMBOL, config, position_to_close=final_pos_check, reason="Final Cleanup Close", market_info=market_info)
                     time.sleep(1)
                     cancel_all_orders(exchange_instance, config.SYMBOL, config, reason="Final Cleanup Cancel", market_info=market_info)


    elif run_order_tests and not config.TESTNET_MODE:
         # This case is already handled by the critical check above, but kept for clarity
         main_logger.error(f"{Back.RED}Order tests skipped because TESTNET_MODE is False.{Style.RESET_ALL}")
    else:
        main_logger.info(f"\n{Fore.CYAN}--- Order placement tests were disabled ---{Style.RESET_ALL}")


    main_logger.info(f"\n{Fore.GREEN}{Style.BRIGHT}--- Main Script Execution Completed Successfully ---{Style.RESET_ALL}")

# --- Exception Handling ---
except AuthenticationError as e:
     main_logger.critical(f"\n{Back.RED}{Fore.WHITE}--- CRITICAL: Authentication Error ---{Style.RESET_ALL}")
     main_logger.critical(f"Failed to authenticate with Bybit. Check API Key/Secret and permissions: {e}")
     main_logger.critical("Ensure keys are correctly set in environment variables or '.env' file and have necessary API permissions (read/trade).")
     sys.exit(1)
except (NetworkError, ExchangeNotAvailable, ccxt.RequestTimeout) as e:
     main_logger.critical(f"\n{Back.RED}{Fore.WHITE}--- CRITICAL: Network/Exchange Error ---{Style.RESET_ALL}")
     main_logger.critical(f"Could not connect to Bybit or the exchange is unavailable/timed out: {e}")
     main_logger.critical("Check internet connection and Bybit status page.")
     sys.exit(1)
except BadSymbol as e:
     main_logger.critical(f"\n{Back.RED}{Fore.WHITE}--- CRITICAL: Bad Symbol Error ---{Style.RESET_ALL}")
     main_logger.critical(f"Invalid, inactive, or incorrectly formatted symbol used: '{config.SYMBOL}'. Error: {e}")
     main_logger.critical("Verify the symbol exists on Bybit (Testnet/Mainnet) and matches the required format (e.g., BTC/USDT:USDT for linear swaps).")
     sys.exit(1)
except InsufficientFunds as e:
     main_logger.critical(f"\n{Back.RED}{Fore.WHITE}--- CRITICAL: Insufficient Funds Error ---{Style.RESET_ALL}")
     main_logger.critical(f"Operation failed due to insufficient funds: {e}")
     main_logger.critical(f"Check available balance ({available:.4f} USDT reported earlier) and order costs/margin requirements.")
     # Optionally send SMS alert here if configured
     send_sms_alert(f"[BybitApp] CRITICAL: Insufficient Funds Error for {config.SYMBOL}", config)
     sys.exit(1)
except (InvalidOrder, OrderNotFound, ccxt.ExchangeError) as e: # Catch more specific CCXT errors
    main_logger.critical(f"\n{Back.RED}{Fore.WHITE}--- CRITICAL: Exchange Logic Error ---{Style.RESET_ALL}")
    main_logger.critical(f"An error occurred related to order placement, cancellation, or position management: {type(e).__name__} - {e}", exc_info=True)
    main_logger.critical("This could be due to invalid parameters, rate limits, margin issues, or other exchange rules.")
    send_sms_alert(f"[BybitApp] CRITICAL: Exchange Logic Error - {type(e).__name__}", config)
    sys.exit(1)
except Exception as e:
    # Catch any other unexpected exceptions
    main_logger.critical(f"\n{Back.RED}{Fore.WHITE}--- CRITICAL: Unhandled Exception ---{Style.RESET_ALL}")
    main_logger.critical(f"An unexpected error occurred during main script execution: {e}", exc_info=True) # Log traceback
    # Optional: Send SMS on critical failure
    send_sms_alert(f"[BybitApp] CRITICAL FAILURE in main script: {type(e).__name__}", config)
    sys.exit(1)

finally:
    # --- Cleanup ---
    # CCXT typically handles HTTP connection pooling, so explicit closing isn't usually required.
    # However, if using WebSockets or other persistent connections, close them here.
    if exchange_instance:
         main_logger.debug("Exchange instance cleanup (if necessary - typically handled by CCXT for HTTP).")
         # Example if explicit close were needed:
         # try:
         #     if hasattr(exchange_instance, 'close'):
         #         exchange_instance.close()
         #         main_logger.info("Closed exchange connection.")
         # except Exception as close_err:
         #     main_logger.warning(f"Error during exchange instance cleanup (ignored): {close_err}")
    print("-" * 80)
    main_logger.info(f"{Fore.YELLOW}Main script execution finished.{Style.RESET_ALL}")
