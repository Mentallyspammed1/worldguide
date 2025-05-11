# File: main.py
# -*- coding: utf-8 -*-

"""
Main Script to Demonstrate Usage of Bybit V5 Helper Modules
"""

import logging
import sys
import time
from decimal import Decimal

# Third-party imports
try:
    import ccxt
except ImportError:
    print("Error: CCXT library not found. Please install it: pip install ccxt")
    sys.exit(1)
try:
    from colorama import init, Fore, Style, Back

    init(autoreset=True)  # Initialize colorama
except ImportError:
    print(
        "Warning: colorama library not found. Logs will not be colored. Install: pip install colorama"
    )

    # Define dummy color constants if colorama is not available
    class DummyColor:
        def __getattr__(self, name: str) -> str:
            return ""

    Fore = Style = Back = DummyColor()
try:
    from dotenv import load_dotenv

    load_dotenv()
    print(
        f"{Fore.CYAN}Attempted to load environment variables from .env file.{Style.RESET_ALL}"
    )
except ImportError:
    print(
        f"{Fore.YELLOW}dotenv library not found, relying on system environment variables only.{Style.RESET_ALL}"
    )

# --- Local Module Imports ---
# Import necessary components from the split modules
from config import Config
from utils import (
    validate_market,
    format_order_id,
    send_sms_alert,
)  # Import placeholders/utils
from exchange_setup import initialize_bybit
from account_config import set_leverage, fetch_account_info_bybit_v5
from market_data import (
    fetch_usdt_balance,
    fetch_ohlcv_paginated,
    fetch_ticker_validated,
)
from order_management import (
    place_market_order_slippage_check,
    cancel_all_orders,
    place_limit_order_tif,
    place_native_stop_loss,
    fetch_open_orders_filtered,
)
from position_management import (
    get_current_position_bybit_v5,
    close_position_reduce_only,
    fetch_position_risk_bybit_v5,
)

# --- Logger Setup ---
# Configure logging for the main script and potentially set level for imported modules
log_level = logging.DEBUG  # Set to INFO or WARNING for production
# Create logger
main_logger = logging.getLogger()  # Get root logger or a specific name like 'bybit_app'
main_logger.setLevel(log_level)
# Create console handler
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(log_level)
# Create formatter and add it to the handler
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] {%(name)s:%(lineno)d:%(funcName)s} - %(message)s",  # Include module name
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
# Add the handler to the logger
if not main_logger.hasHandlers():
    main_logger.addHandler(handler)

main_logger.info(
    f"{Fore.YELLOW}{Style.BRIGHT}--- Main Script Execution Started ---{Style.RESET_ALL}"
)

# --- Real Implementation of Placeholders (CRITICAL) ---
# The helper modules rely on these being correctly defined and available.
# For this example, we'll keep using the placeholders defined in utils.py,
# but in a real application, you would implement the actual logic here or import it.

# 1. retry_api_call: The placeholder in utils.py just executes the function once.
#    A real implementation would handle ccxt exceptions and retries.
#    Example structure (requires importing `ccxt.base.errors`):
#    from functools import wraps
#    from ccxt.base.errors import NetworkError, RateLimitExceeded, ExchangeNotAvailable # etc.
#    def real_retry_api_call(max_retries=3, initial_delay=1.0, ...):
#        def decorator(func):
#            @wraps(func)
#            def wrapper(*args, **kwargs):
#                retries = 0
#                while retries <= max_retries:
#                    try:
#                        return func(*args, **kwargs)
#                    except (NetworkError, RateLimitExceeded, ExchangeNotAvailable) as e:
#                        retries += 1
#                        if retries > max_retries:
#                            main_logger.error(f"API call {func.__name__} failed after {max_retries} retries: {e}")
#                            raise
#                        delay = initial_delay * (2 ** (retries - 1)) # Exponential backoff
#                        main_logger.warning(f"API call {func.__name__} failed (Retry {retries}/{max_retries}): {e}. Retrying in {delay:.2f}s...")
#                        time.sleep(delay)
#                    # Add other specific CCXT errors if needed
#                    except Exception as e: # Catch other unexpected errors
#                        main_logger.error(f"API call {func.__name__} failed with unexpected error: {e}", exc_info=True)
#                        raise # Re-raise unexpected errors immediately
#            return wrapper
#        return decorator
#    # If using real retry, assign it:
#    # retry_api_call = real_retry_api_call # This would override the placeholder import if needed

# 2. safe_decimal_conversion, format_*, send_sms_alert:
#    These are currently placeholders in utils.py. Use them as is for this demo,
#    or provide real implementations here.

# --- Configuration ---
main_logger.info("Loading configuration...")
try:
    config = Config()
    # Override specific config settings if needed for this run
    config.TESTNET_MODE = True  # Ensure testnet for this example run
    config.SYMBOL = "BTC/USDT:USDT"  # Set the primary symbol for testing
    config.DEFAULT_LEVERAGE = 5  # Use lower leverage for testing
    config.ENABLE_SMS_ALERTS = False  # Keep SMS disabled for demo

    main_logger.info(
        f"Config Loaded: Testnet={config.TESTNET_MODE}, Symbol={config.SYMBOL}, Leverage={config.DEFAULT_LEVERAGE}"
    )

    # Validate essential config like API keys
    if not config.API_KEY or not config.API_SECRET:
        main_logger.error(
            f"{Back.RED}{Fore.WHITE}FATAL: Bybit API Key or Secret not found in environment variables or .env file.{Style.RESET_ALL}"
        )
        main_logger.error("Please set BYBIT_API_KEY and BYBIT_API_SECRET.")
        sys.exit(1)

except Exception as e:
    main_logger.critical(f"Failed to initialize configuration: {e}", exc_info=True)
    sys.exit(1)

# --- Main Execution Block ---
exchange_instance: Optional[ccxt.bybit] = None
try:
    # 1. Initialize Exchange
    main_logger.info("\n--- 1. Initializing Exchange ---")
    exchange_instance = initialize_bybit(config)
    if not exchange_instance:
        raise RuntimeError("Exchange initialization failed.")
    main_logger.info(
        f"{Fore.GREEN}Exchange Initialized OK: {exchange_instance.id} (Testnet: {config.TESTNET_MODE}){Style.RESET_ALL}"
    )

    # 2. Validate Market
    main_logger.info("\n--- 2. Validating Market ---")
    market_info = validate_market(exchange_instance, config.SYMBOL, config)
    if not market_info:
        raise RuntimeError(f"Market validation failed for symbol {config.SYMBOL}")
    main_logger.info(
        f"{Fore.GREEN}Market Validation OK for {config.SYMBOL}. Type: {market_info.get('type')}, Logic: {market_info.get('linear') or market_info.get('inverse')}{Style.RESET_ALL}"
    )

    # 3. Fetch Balance
    main_logger.info("\n--- 3. Fetching USDT Balance ---")
    equity, available = fetch_usdt_balance(exchange_instance, config)
    if equity is None or available is None:
        # Retry decorator should handle retries, this means final failure
        main_logger.warning(
            f"{Fore.YELLOW}Fetch Balance failed after retries (returned None). Proceeding cautiously.{Style.RESET_ALL}"
        )
        # Set defaults to avoid crashing later tests, but log the issue
        equity, available = Decimal("0.0"), Decimal("0.0")
    else:
        main_logger.info(
            f"{Fore.GREEN}Fetch Balance OK: Equity={equity:.4f} USDT, Available={available:.4f} USDT{Style.RESET_ALL}"
        )

    # 4. Set Leverage
    main_logger.info("\n--- 4. Setting Leverage ---")
    lev_ok = set_leverage(
        exchange_instance, config.SYMBOL, config.DEFAULT_LEVERAGE, config
    )
    if not lev_ok:
        # This might fail legitimately on some account types (e.g., UTA Isolated)
        main_logger.warning(
            f"{Fore.YELLOW}Set Leverage to {config.DEFAULT_LEVERAGE}x potentially failed or was unnecessary. Check logs for details.{Style.RESET_ALL}"
        )
    else:
        main_logger.info(
            f"{Fore.GREEN}Set Leverage OK (or already set) to {config.DEFAULT_LEVERAGE}x for {config.SYMBOL}.{Style.RESET_ALL}"
        )

    # 5. Fetch Ticker
    main_logger.info("\n--- 5. Fetching Ticker ---")
    ticker = fetch_ticker_validated(exchange_instance, config.SYMBOL, config)
    if not ticker:
        main_logger.warning(
            f"{Fore.YELLOW}Fetch Ticker failed after retries.{Style.RESET_ALL}"
        )
    else:
        main_logger.info(
            f"{Fore.GREEN}Fetch Ticker OK: Last={ticker.get('last')}, Bid={ticker.get('bid')}, Ask={ticker.get('ask')}, Spread={ticker.get('spread_pct'):.4f}%{Style.RESET_ALL}"
        )

    # 6. Fetch Current Position
    main_logger.info("\n--- 6. Fetching Current Position ---")
    position = get_current_position_bybit_v5(exchange_instance, config.SYMBOL, config)
    # This function returns a default dict if no position, so check the side
    if position["side"] == config.POS_NONE:
        main_logger.info(
            f"{Fore.CYAN}Fetch Position OK: No active position found for {config.SYMBOL}.{Style.RESET_ALL}"
        )
    else:
        main_logger.info(
            f"{Fore.GREEN}Fetch Position OK: Side={position['side']}, Qty={position['qty']}, Entry={position['entry_price']}{Style.RESET_ALL}"
        )

    # 7. Fetch Position Risk (only if position exists)
    if position["side"] != config.POS_NONE:
        main_logger.info("\n--- 7. Fetching Position Risk ---")
        risk = fetch_position_risk_bybit_v5(exchange_instance, config.SYMBOL, config)
        if not risk:
            main_logger.warning(
                f"{Fore.YELLOW}Fetch Position Risk failed or position closed between checks.{Style.RESET_ALL}"
            )
        else:
            main_logger.info(
                f"{Fore.GREEN}Fetch Position Risk OK: Liq={risk['liq_price']}, IMR={risk['imr']:.4%}, MMR={risk['mmr']:.4%}, Value={risk['position_value']}{Style.RESET_ALL}"
            )
    else:
        main_logger.info("\n--- 7. Skipping Fetch Position Risk (No position) ---")

    # --- Optional: More Advanced Tests (Account Config, OHLCV, Orders) ---
    run_advanced_tests = True  # Set to False to skip these

    if run_advanced_tests:
        main_logger.info(
            f"\n{Style.BRIGHT}--- Running Advanced Tests ---{Style.RESET_ALL}"
        )

        # Fetch Account Info
        main_logger.info("\n--- Adv 1. Fetching Account Info ---")
        acc_info = fetch_account_info_bybit_v5(exchange_instance, config)
        if acc_info:
            main_logger.info(
                f"{Fore.GREEN}Fetch Account Info OK: UTA Status={acc_info.get('unifiedMarginStatus')}, MarginMode={acc_info.get('marginMode')}{Style.RESET_ALL}"
            )

        # Fetch OHLCV
        main_logger.info("\n--- Adv 2. Fetching OHLCV Data ---")
        ohlcv_df = fetch_ohlcv_paginated(
            exchange_instance, config.SYMBOL, "1h", config, max_total_candles=100
        )  # Fetch last 100 hourly candles
        if ohlcv_df is not None and not ohlcv_df.empty:
            main_logger.info(
                f"{Fore.GREEN}Fetch OHLCV OK: Fetched {len(ohlcv_df)} candles. Last Close: {ohlcv_df['close'].iloc[-1]}{Style.RESET_ALL}"
            )
        elif ohlcv_df is not None and ohlcv_df.empty:
            main_logger.info(
                f"{Fore.CYAN}Fetch OHLCV OK: No candles returned for the period.{Style.RESET_ALL}"
            )
        else:
            main_logger.warning(f"{Fore.YELLOW}Fetch OHLCV failed.{Style.RESET_ALL}")

        # --- Order Placement Tests (Use with extreme caution, ONLY on TESTNET) ---
        run_order_tests = True  # <<< Set to True to run order tests ON TESTNET
        if run_order_tests and config.TESTNET_MODE:
            main_logger.warning(
                f"\n{Back.MAGENTA}{Fore.WHITE}{Style.BRIGHT}--- Running Order Placement Tests (TESTNET) ---{Style.RESET_ALL}"
            )
            time.sleep(2)  # Pause before potentially modifying state

            # Ensure position is flat before starting tests
            main_logger.info("\n--- Pre-Test: Ensure Flat Position ---")
            initial_pos = get_current_position_bybit_v5(
                exchange_instance, config.SYMBOL, config
            )
            if initial_pos["side"] != config.POS_NONE:
                main_logger.warning(
                    f"Existing {initial_pos['side']} position found ({initial_pos['qty']}). Attempting to close..."
                )
                close_result = close_position_reduce_only(
                    exchange_instance,
                    config.SYMBOL,
                    config,
                    position_to_close=initial_pos,
                    reason="Test Prep Close",
                )
                time.sleep(3)  # Wait for closure to process
                current_pos_check = get_current_position_bybit_v5(
                    exchange_instance, config.SYMBOL, config
                )
                if current_pos_check["side"] != config.POS_NONE:
                    main_logger.error(
                        f"{Back.RED}Failed to close pre-existing position. Aborting order tests.{Style.RESET_ALL}"
                    )
                    sys.exit(1)  # Abort if cleanup fails
                else:
                    main_logger.info(
                        f"{Fore.GREEN}Pre-existing position closed successfully.{Style.RESET_ALL}"
                    )
            else:
                main_logger.info(
                    f"{Fore.CYAN}Position confirmed flat. Proceeding with order tests.{Style.RESET_ALL}"
                )

            # Test Limit Order Placement
            main_logger.info("\n--- Test 8a. Place Limit Order ---")
            current_ticker = fetch_ticker_validated(
                exchange_instance, config.SYMBOL, config
            )
            if current_ticker and current_ticker.get("bid"):
                limit_buy_price = current_ticker["bid"] * Decimal(
                    "0.995"
                )  # Place 0.5% below current bid
                limit_qty = Decimal("0.001")  # Small BTC test quantity
                limit_order = place_limit_order_tif(
                    exchange_instance,
                    config.SYMBOL,
                    "buy",
                    limit_qty,
                    limit_buy_price,
                    config,
                    time_in_force="GTC",
                    is_post_only=True,
                )
                if limit_order:
                    main_logger.info(
                        f"{Fore.GREEN}Limit Order (PostOnly) placed OK: ID ...{format_order_id(limit_order.get('id'))}, Status: {limit_order.get('status')}{Style.RESET_ALL}"
                    )
                    limit_order_id = limit_order.get("id")
                    time.sleep(1)

                    # Test Fetch Open Orders (Filtered)
                    main_logger.info("\n--- Test 8b. Fetch Open Limit Order ---")
                    open_limit_orders = fetch_open_orders_filtered(
                        exchange_instance,
                        config.SYMBOL,
                        config,
                        side="buy",
                        order_type="limit",
                        order_filter="Order",
                    )
                    found_order = False
                    if open_limit_orders:
                        for o in open_limit_orders:
                            if o.get("id") == limit_order_id:
                                main_logger.info(
                                    f"{Fore.GREEN}Successfully fetched the open limit order: ID ...{format_order_id(o.get('id'))}{Style.RESET_ALL}"
                                )
                                found_order = True
                                break
                    if not found_order:
                        main_logger.warning(
                            f"{Fore.YELLOW}Could not fetch the specific open limit order (might have filled or filter failed).{Style.RESET_ALL}"
                        )

                    # Test Cancel All Orders (should cancel the limit order)
                    main_logger.info("\n--- Test 8c. Cancel All Orders ---")
                    cancel_ok = cancel_all_orders(
                        exchange_instance,
                        config.SYMBOL,
                        config,
                        reason="Test Cancel Limit",
                    )
                    if cancel_ok:
                        main_logger.info(
                            f"{Fore.GREEN}Cancel All Orders executed successfully.{Style.RESET_ALL}"
                        )
                    else:
                        main_logger.warning(
                            f"{Fore.YELLOW}Cancel All Orders reported potential issues. Check logs.{Style.RESET_ALL}"
                        )
                    time.sleep(1)
                    # Verify cancellation
                    open_orders_after_cancel = fetch_open_orders_filtered(
                        exchange_instance, config.SYMBOL, config, order_filter="Order"
                    )
                    if not open_orders_after_cancel:
                        main_logger.info(
                            f"{Fore.GREEN}Verified no open regular orders after cancellation.{Style.RESET_ALL}"
                        )
                    else:
                        main_logger.warning(
                            f"{Fore.YELLOW}Found {len(open_orders_after_cancel)} open orders after cancel call!{Style.RESET_ALL}"
                        )

                else:
                    main_logger.warning(
                        f"{Fore.YELLOW}Limit Order placement failed or was rejected (e.g., PostOnly failed).{Style.RESET_ALL}"
                    )

            else:
                main_logger.error(
                    "Cannot get current ticker bid price for limit order test."
                )

            # Test Market Order Entry
            main_logger.info("\n--- Test 9a. Place Market Order (Entry) ---")
            entry_qty = Decimal("0.001")  # Small BTC test quantity
            market_entry_order = place_market_order_slippage_check(
                exchange_instance, config.SYMBOL, "buy", entry_qty, config
            )
            if not market_entry_order:
                main_logger.error(
                    f"{Back.RED}Market Entry Order Placement Failed. Aborting further dependent tests.{Style.RESET_ALL}"
                )
            else:
                main_logger.info(
                    f"{Fore.GREEN}Market Entry Order OK: ID ...{format_order_id(market_entry_order.get('id'))}, Status: {market_entry_order.get('status')}{Style.RESET_ALL}"
                )
                time.sleep(3)  # Wait for position to update

                # Verify position opened
                pos_after_entry = get_current_position_bybit_v5(
                    exchange_instance, config.SYMBOL, config
                )
                if pos_after_entry["side"] != config.POS_LONG or pos_after_entry[
                    "qty"
                ] < entry_qty * Decimal("0.9"):
                    main_logger.error(
                        f"{Back.RED}Position check after market buy failed! Side: {pos_after_entry['side']}, Qty: {pos_after_entry['qty']}. Aborting.{Style.RESET_ALL}"
                    )
                else:
                    main_logger.info(
                        f"{Fore.GREEN}Position Confirmed: LONG {pos_after_entry['qty']} @ Entry ~{pos_after_entry['entry_price']}{Style.RESET_ALL}"
                    )

                    # Test Native Stop Loss
                    main_logger.info("\n--- Test 9b. Place Native Stop Loss ---")
                    sl_price = pos_after_entry["entry_price"] * Decimal(
                        "0.95"
                    )  # 5% below entry for test
                    sl_order = place_native_stop_loss(
                        exchange_instance,
                        config.SYMBOL,
                        "sell",
                        pos_after_entry["qty"],
                        sl_price,
                        config,
                        trigger_by="MarkPrice",
                    )
                    if sl_order:
                        main_logger.info(
                            f"{Fore.GREEN}Native SL Order OK: ID ...{format_order_id(sl_order.get('id'))}, Status: {sl_order.get('status')}, Trigger: {sl_price}{Style.RESET_ALL}"
                        )
                        sl_order_id = sl_order.get("id")

                        # Fetch Stop Orders to verify
                        open_stops = fetch_open_orders_filtered(
                            exchange_instance,
                            config.SYMBOL,
                            config,
                            order_filter="StopOrder",
                        )
                        found_sl = (
                            any(o.get("id") == sl_order_id for o in open_stops)
                            if open_stops
                            else False
                        )
                        if found_sl:
                            main_logger.info(
                                f"{Fore.GREEN}Verified SL order exists via fetch (StopOrder filter).{Style.RESET_ALL}"
                            )
                        else:
                            main_logger.warning(
                                f"{Fore.YELLOW}Could not verify SL order via fetch (StopOrder filter).{Style.RESET_ALL}"
                            )

                        # Need a way to cancel specific stop orders or use cancel_all with StopOrder filter
                        main_logger.info(
                            "\n--- Test 9c. Cancel Specific Stop Order (Manual - Requires cancel_order) ---"
                        )
                        try:
                            cancel_params = {"category": _get_v5_category(market_info)}
                            main_logger.info(
                                f"Attempting to cancel SL order ID: {sl_order_id}..."
                            )
                            exchange_instance.cancel_order(
                                sl_order_id, config.SYMBOL, params=cancel_params
                            )
                            main_logger.info(
                                f"{Fore.GREEN}Specific SL order cancellation request sent.{Style.RESET_ALL}"
                            )
                        except Exception as cancel_err:
                            main_logger.error(
                                f"Failed to cancel specific SL order: {cancel_err}"
                            )
                        time.sleep(1)

                    else:
                        main_logger.warning(
                            f"{Fore.YELLOW}Native Stop Loss Placement Failed.{Style.RESET_ALL}"
                        )

                    # Test Close Position
                    main_logger.info(
                        "\n--- Test 9d. Close Position (Market ReduceOnly) ---"
                    )
                    # Fetch position again before closing, in case SL triggered or something changed
                    pos_before_close = get_current_position_bybit_v5(
                        exchange_instance, config.SYMBOL, config
                    )
                    if pos_before_close["side"] == config.POS_NONE:
                        main_logger.info(
                            f"{Fore.CYAN}Position already closed before final close attempt. Skipping.{Style.RESET_ALL}"
                        )
                    else:
                        close_order = close_position_reduce_only(
                            exchange_instance,
                            config.SYMBOL,
                            config,
                            position_to_close=pos_before_close,
                            reason="Test Close",
                        )
                        if close_order:
                            main_logger.info(
                                f"{Fore.GREEN}Close Position Order OK: ID ...{format_order_id(close_order.get('id'))}{Style.RESET_ALL}"
                            )
                        else:
                            main_logger.warning(
                                f"{Fore.YELLOW}Close Position attempt failed or position was already closed.{Style.RESET_ALL}"
                            )

                    # Final position check
                    time.sleep(2)
                    final_pos_check = get_current_position_bybit_v5(
                        exchange_instance, config.SYMBOL, config
                    )
                    if final_pos_check["side"] == config.POS_NONE:
                        main_logger.info(
                            f"{Fore.GREEN}{Style.BRIGHT}Position Confirmed Flat After All Tests.{Style.RESET_ALL}"
                        )
                    else:
                        main_logger.error(
                            f"{Back.RED}Position is NOT flat after tests! Side: {final_pos_check['side']}, Qty: {final_pos_check['qty']}{Style.RESET_ALL}"
                        )

        elif run_order_tests:
            main_logger.error(
                f"{Back.RED}Order tests skipped because TESTNET_MODE is False.{Style.RESET_ALL}"
            )
        else:
            main_logger.info(
                f"{Fore.CYAN}Order placement tests were disabled.{Style.RESET_ALL}"
            )

    main_logger.info(
        f"\n{Fore.GREEN}{Style.BRIGHT}--- Main Script Execution Completed Successfully ---{Style.RESET_ALL}"
    )

except ccxt.AuthenticationError as e:
    main_logger.critical(
        f"\n{Back.RED}{Fore.WHITE}--- Authentication Error ---{Style.RESET_ALL}"
    )
    main_logger.critical(
        f"Failed to authenticate with Bybit. Check API Key/Secret and permissions: {e}"
    )
    main_logger.critical(
        "Ensure keys are correctly set in environment variables or .env file."
    )
    sys.exit(1)
except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
    main_logger.critical(
        f"\n{Back.RED}{Fore.WHITE}--- Network/Exchange Error ---{Style.RESET_ALL}"
    )
    main_logger.critical(f"Could not connect to Bybit or exchange is unavailable: {e}")
    sys.exit(1)
except ccxt.BadSymbol as e:
    main_logger.critical(
        f"\n{Back.RED}{Fore.WHITE}--- Bad Symbol Error ---{Style.RESET_ALL}"
    )
    main_logger.critical(
        f"Invalid or inactive symbol used: {config.SYMBOL}. Error: {e}"
    )
    sys.exit(1)
except Exception as e:
    main_logger.critical(
        f"\n{Back.RED}{Fore.WHITE}--- Main Script Execution Failed ---{Style.RESET_ALL}"
    )
    main_logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
    # Optional: Send SMS on critical failure
    send_sms_alert(
        f"[BybitApp] CRITICAL FAILURE in main script: {type(e).__name__}", config
    )
    sys.exit(1)

finally:
    # Clean up resources if needed (though ccxt usually handles connections)
    if exchange_instance:
        try:
            # CCXT doesn't have an explicit close method usually needed for HTTP APIs
            # exchange_instance.close()
            main_logger.debug("Exchange instance cleanup (if necessary).")
        except Exception as close_err:
            main_logger.warning(
                f"Error during exchange instance cleanup (ignored): {close_err}"
            )
    print("-" * 60)
    main_logger.info(f"{Fore.YELLOW}Finished main script execution.{Style.RESET_ALL}")
