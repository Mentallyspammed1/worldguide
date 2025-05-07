# File: main.py
"""
Main execution script for the XR Scalper Trading Bot.

This script initializes the bot, loads configuration, sets up logging,
connects to the exchange, and runs the main trading loop which iterates
through specified symbols, analyzes market data, and potentially places trades
based on the defined strategy.
"""

import logging
import os
import sys
import time # Retained for time.monotonic(), though asyncio.get_event_loop().time() is an alternative for async
import signal # Import signal for graceful shutdown
import traceback # Import traceback for detailed error reporting
import asyncio # Import asyncio for asynchronous operations
from datetime import datetime
from typing import Dict, Any, Optional # Added Optional for type hinting

import ccxt
import ccxt.async_support as ccxt_async # Use async version for async methods
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv

# --- Import Custom Modules ---
try:
    import config_loader
    import exchange_api # Assumed to contain async initialize_exchange
    import logger_setup # logger_setup should provide configure_logging(config)
    import trading_strategy # Assumed to contain async analyze_and_trade_symbol
    import utils # utils should provide constants, timezone handling, sensitive data masking
except ImportError as e:
    print(f"\033[1;91mCRITICAL ERROR: Failed to import one or more custom modules. Ensure all required modules " # Using direct color codes as utils might not be loaded
          f"(config_loader.py, exchange_api.py, logger_setup.py, trading_strategy.py, utils.py) "
          f"are in the correct directory and do not have syntax errors.\033[0m", file=sys.stderr)
    print(f"ImportError details: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
except Exception as e:
     print(f"\033[1;91mCRITICAL ERROR: An unexpected error occurred during module import: {e}\033[0m", file=sys.stderr) # Using direct color codes
     traceback.print_exc(file=sys.stderr)
     sys.exit(1)

# --- Constants ---
DEFAULT_LOOP_INTERVAL_SECONDS = 60  # Default loop interval if not in config
# Use constants from utils where appropriate
NEON_GREEN = utils.NEON_GREEN
NEON_YELLOW = utils.NEON_YELLOW
NEON_RED = utils.NEON_RED
RESET = utils.RESET_ALL_STYLE # Use RESET_ALL_STYLE from utils

# --- Global State for Shutdown ---
shutdown_requested = False

def signal_handler(signum: int, frame: Any) -> None:
    """
    Signal handler to set the shutdown flag on receiving signals like SIGINT or SIGTERM.
    """
    global shutdown_requested
    shutdown_requested = True
    print(f"\n{NEON_YELLOW}Signal {signum} received ({signal.Signals(signum).name}). Requesting bot shutdown...{RESET}", file=sys.stderr)


# --- Main Execution Function ---
async def main() -> None: # Changed to async def
    """
    Main function to initialize the bot, set up resources, and run the analysis/trading loop.
    Handles setup, configuration loading, exchange connection, and the main processing loop.
    """
    global API_KEY, API_SECRET # Access globals if not passed as arguments

    CONFIG: Dict[str, Any] = {}
    try:
        CONFIG = config_loader.load_config()
        print(f"{NEON_GREEN}Configuration loaded successfully from '{utils.CONFIG_FILE}'.{RESET}", file=sys.stderr) # Basic print before logger
    except FileNotFoundError:
        print(f"{NEON_RED}CRITICAL: Configuration file '{utils.CONFIG_FILE}' not found. Exiting.{RESET}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"{NEON_RED}CRITICAL: Error loading configuration: {e}. Exiting.{RESET}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    try:
        logger_setup.configure_logging(CONFIG)
    except Exception as e:
         print(f"{NEON_RED}CRITICAL: Error configuring logging: {e}. Exiting.{RESET}", file=sys.stderr)
         traceback.print_exc(file=sys.stderr)
         sys.exit(1)

    init_logger = logging.getLogger("xrscalper_bot_init")
    init_logger.info("--- Initializing XR Scalper Bot (Async Version) ---")

    configured_timezone = os.getenv("TIMEZONE", CONFIG.get("timezone", utils.DEFAULT_TIMEZONE))
    try:
        utils.set_timezone(configured_timezone)
        init_logger.info(f"Using Timezone: {utils.get_timezone()}")
    except Exception as e:
         init_logger.warning(f"Failed to set timezone to '{configured_timezone}': {e}. Using system default.", exc_info=True)

    try:
        if hasattr(utils, 'SensitiveFormatter') and hasattr(utils.SensitiveFormatter, 'set_sensitive_data'):
             # Check if API_KEY and API_SECRET are loaded from .env (they are global in this script)
             if 'API_KEY' in globals() and API_KEY and 'API_SECRET' in globals() and API_SECRET:
                 utils.SensitiveFormatter.set_sensitive_data(API_KEY, API_SECRET)
                 init_logger.debug("Sensitive data masking configured for logging.")
             else:
                 init_logger.warning("API keys not loaded/available, sensitive data masking may not function correctly.")
        else:
             init_logger.warning("Sensitive data masking features not found in utils module.")
    except Exception as e:
        init_logger.warning(f"Error configuring sensitive data masking: {e}", exc_info=True)

    try:
        current_time_str = datetime.now(utils.get_timezone()).strftime('%Y-%m-%d %H:%M:%S %Z')
        init_logger.info(f"Startup Time: {current_time_str}")
    except Exception as e:
         init_logger.warning(f"Could not format startup time with timezone: {e}. Falling back.", exc_info=True)
         init_logger.info(f"Startup Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    quote_currency = CONFIG.get('quote_currency')
    if quote_currency:
        init_logger.info(f"Quote Currency: {quote_currency}")
    else:
        init_logger.warning("Quote currency not specified in configuration.")

    try:
        python_version = sys.version.split()[0]
        # Use the async ccxt module for version check
        ccxt_version = getattr(ccxt_async, '__version__', 'N/A')
        pandas_version = getattr(pd, '__version__', 'N/A')
        pandas_ta_version = getattr(ta, '__version__', 'N/A')
        if hasattr(ta, 'version') and callable(ta.version) and pandas_ta_version == 'N/A': # More specific check
             try:
                pandas_ta_version = ta.version()
             except Exception:
                 pandas_ta_version = 'N/A (ta.version() error)'
    except Exception as e:
        init_logger.warning(f"Error getting dependency versions: {e}", exc_info=True)
        python_version = sys.version.split()[0]
        ccxt_version = pandas_version = pandas_ta_version = 'N/A'

    init_logger.info(f"Versions: Python={python_version}, CCXT={ccxt_version}, "
                     f"Pandas={pandas_version}, PandasTA={pandas_ta_version}")

    enable_trading = CONFIG.get("enable_trading", False)
    use_sandbox = CONFIG.get("use_sandbox", True) # Default to True based on safety principles

    if enable_trading:
        init_logger.warning("!!! LIVE TRADING IS ENABLED !!!")
        if use_sandbox:
            init_logger.warning("--> Operating in SANDBOX (Testnet) Environment.")
        else:
            init_logger.warning("!!! CAUTION: OPERATING WITH REAL MONEY !!!")

        risk_per_trade_config = CONFIG.get('risk_per_trade', 0.0)
        risk_pct = risk_per_trade_config * 100
        leverage = CONFIG.get('leverage', 1)
        symbols_to_trade = CONFIG.get('symbols_to_trade', [])
        tsl_enabled = CONFIG.get('enable_trailing_stop', False)
        be_enabled = CONFIG.get('enable_break_even', False)
        time_exit_enabled = CONFIG.get('time_based_exit_minutes') is not None and CONFIG.get('time_based_exit_minutes', 0) > 0 # Check if configured and > 0
        entry_order_type = CONFIG.get('entry_order_type', 'market').lower()
        interval = CONFIG.get('interval', 'N/A') # Ensure interval is a string for CCXT_INTERVAL_MAP

        init_logger.info("--- Critical Trading Settings ---")
        init_logger.info(f"  Symbols: {symbols_to_trade}")
        init_logger.info(f"  Interval: {str(interval)}") # Ensure interval is logged as string
        init_logger.info(f"  Entry Order Type: {entry_order_type}")
        init_logger.info(f"  Risk per Trade: {risk_pct:.2f}%") # Log formatted percentage
        init_logger.info(f"  Leverage: {leverage}x")
        init_logger.info(f"  Trailing Stop Loss Enabled: {tsl_enabled}")
        init_logger.info(f"  Break Even Enabled: {be_enabled}")
        init_logger.info(f"  Time Based Exit Enabled: {time_exit_enabled}")
        init_logger.info("---------------------------------")

        if not symbols_to_trade:
             init_logger.error("Trading enabled, but 'symbols_to_trade' list is empty. Exiting.")
             return
        if risk_per_trade_config <= 0 and enable_trading: # Log warning if risk is 0 but trading enabled
            init_logger.warning(f"Risk per trade is set to {risk_pct:.2f}%. Positions might not open unless strategy ignores risk.")
        if leverage <= 0:
             init_logger.error(f"Leverage must be greater than 0. Found {leverage}. Exiting.")
             return
        if entry_order_type not in ['market', 'limit']:
             init_logger.error(f"Invalid 'entry_order_type': '{entry_order_type}'. Must be 'market' or 'limit'. Exiting.")
             return
        # Ensure interval is a string before checking in CCXT_INTERVAL_MAP keys
        if not interval or str(interval) not in utils.CCXT_INTERVAL_MAP:
             init_logger.error(f"Invalid or missing 'interval': '{interval}'. Cannot map to CCXT timeframe. Exiting.")
             return

        init_logger.info("Review settings. Starting trading loop in 5 seconds...")
        await asyncio.sleep(5) # Use asyncio.sleep
    else:
        init_logger.info("Live trading is DISABLED. Running in analysis-only mode.")
        symbols_to_process_analysis = CONFIG.get("symbols_to_trade", [])
        if not symbols_to_process_analysis:
             init_logger.error("'symbols_to_trade' is empty. Nothing to process in analysis mode. Exiting.")
             return
        init_logger.info(f"Analysis symbols: {symbols_to_process_analysis}")
        init_logger.info(f"Analysis Interval: {str(CONFIG.get('interval', 'N/A'))}")

    init_logger.info("Initializing exchange connection...")
    # Ensure API_KEY and API_SECRET are available (loaded globally)
    exchange: Optional[ccxt_async.Exchange] = None # Corrected type hint to async_support Exchange
    try:
        # Prepare exchange options from CONFIG, including defaultType for Bybit
        exchange_options = {
             'options': CONFIG.get('exchange_options', {}).get('options', {}),
        }
        # Ensure defaultType is set, fallback to 'unified' for Bybit, 'linear' otherwise
        exchange_id = CONFIG.get("exchange_id", "bybit").lower()
        if 'defaultType' not in exchange_options['options']:
             exchange_options['options']['defaultType'] = 'unified' if exchange_id == 'bybit' else 'linear'

        # Pass these combined options to initialize_exchange
        exchange = await exchange_api.initialize_exchange(API_KEY, API_SECRET, CONFIG, init_logger)
    except Exception as e:
        init_logger.error(f"Unhandled exception during exchange_api.initialize_exchange: {e}", exc_info=True)
        init_logger.error("Bot cannot continue. Exiting.")
        return

    if not exchange:
        init_logger.error("Failed to initialize exchange connection (exchange object is None). Bot cannot continue. Exiting.")
        return

    symbols_to_process = CONFIG.get("symbols_to_trade", [])

    try:
        init_logger.info(f"Loading exchange markets for {exchange.id}...") # This should now work
        # Load markets with params from config if available (e.g., for Bybit Unified category)
        market_load_params = CONFIG.get('market_load_params', {})
        if exchange.id == 'bybit' and exchange.options.get('defaultType') == 'unified' and not market_load_params:
             market_load_params = {'category': 'unifiedaccount'} # Default category for Unified load_markets

        await exchange.load_markets(reload=True, params=market_load_params) # Use await for async ccxt method
        init_logger.info(f"Exchange '{exchange.id}' initialized and markets loaded successfully.")

        if enable_trading:
            if use_sandbox:
                init_logger.info("Connected to exchange in SANDBOX mode.")
            else:
                init_logger.info("Connected to exchange in LIVE (Real Money) mode.")
        else:
             init_logger.info(f"Connected to exchange (mode: {'sandbox' if use_sandbox else 'live'}) for analysis.")

        # Use CCXT's own symbols list
        available_symbols = exchange.symbols if hasattr(exchange, 'symbols') and exchange.symbols else []
        if not available_symbols:
             init_logger.error("Could not retrieve available symbols from exchange. Cannot validate config. Exiting.")
             return

        invalid_symbols = [s for s in symbols_to_process if s not in available_symbols]
        if invalid_symbols:
            init_logger.error(f"Invalid symbols in config (not on exchange): {invalid_symbols}. Exiting.")
            return
        init_logger.info("All configured symbols validated against exchange markets.")

    except ccxt_async.NetworkError as ne: # Use async version exceptions
        init_logger.error(f"Network Error during exchange setup: {ne}. Check connection/status. Exiting.", exc_info=True)
        return
    except ccxt_async.ExchangeError as ee: # Use async version exceptions
        init_logger.error(f"Exchange Error during exchange setup: {ee}. Check API keys/permissions. Exiting.", exc_info=True)
        return
    except AttributeError as ae: # Specifically catch if exchange object is not as expected after await
        init_logger.error(f"AttributeError during exchange setup (likely API issue or response format): {ae}. Exiting.", exc_info=True)
        return
    except Exception as e:
        init_logger.error(f"An unexpected error occurred during exchange initialization/market loading: {e}. Exiting.", exc_info=True)
        return

    try:
        # Signal handling might need adjustments in asyncio context depending on implementation
        # The current signal handler sets a flag, which the loop checks. This is a common pattern.
        # For more complex shutdown, gather pending tasks and cancel them.
        signal.signal(signal.SIGINT, signal_handler)
        # signal.signal(signal.SIGTERM, signal_handler) # SIGTERM might not be reliable on all platforms or in asyncio
        init_logger.info("Signal handlers registered for graceful shutdown (SIGINT).")
    except Exception as e: # e.g. on Windows if SIGTERM is not available or other signal issues
         init_logger.warning(f"Failed to register signal handlers: {e}. Graceful shutdown via signals might be affected.", exc_info=True)

    loop_interval_seconds_cfg = CONFIG.get("loop_interval_seconds", DEFAULT_LOOP_INTERVAL_SECONDS)
    if not isinstance(loop_interval_seconds_cfg, (int, float)) or loop_interval_seconds_cfg <= 0:
         init_logger.error(f"Invalid 'loop_interval_seconds': {loop_interval_seconds_cfg}. Must be positive. Exiting.")
         return
    loop_interval_seconds = float(loop_interval_seconds_cfg)

    init_logger.info(f"Starting main analysis/trading loop for symbols: {symbols_to_process}")
    init_logger.info(f"Loop Interval: {loop_interval_seconds} seconds")

    try:
        while not shutdown_requested:
            loop_start_time = time.monotonic() # time.monotonic() is fine for measuring intervals
            current_cycle_time = datetime.now(utils.get_timezone())
            init_logger.info(f"--- Starting Loop Cycle @ {current_cycle_time.strftime('%Y-%m-%d %H:%M:%S %Z')} ---")

            # Consider using asyncio.gather for processing symbols in parallel if desired
            # tasks = [trading_strategy.analyze_and_trade_symbol(exchange, symbol, CONFIG, logging.getLogger(f"xrscalper_bot_{symbol.replace('/', '_').replace(':', '-')}"), enable_trading) for symbol in symbols_to_process]
            # await asyncio.gather(*tasks)

            for symbol in symbols_to_process:
                if shutdown_requested:
                    init_logger.info("Shutdown requested, stopping symbol processing in current cycle.")
                    break

                # Create a dedicated logger for each symbol processing
                safe_symbol_name = symbol.replace('/', '_').replace(':', '-')
                symbol_logger = logging.getLogger(f"xrscalper_bot_{safe_symbol_name}")

                try:
                    symbol_logger.info(f"Processing symbol: {symbol}")
                    # analyze_and_trade_symbol is an async function
                    await trading_strategy.analyze_and_trade_symbol(
                        exchange,
                        symbol,
                        CONFIG,
                        symbol_logger,
                        enable_trading
                    )
                except ccxt_async.NetworkError as ne: # Use async version exceptions
                     symbol_logger.error(f"Network Error for {symbol}: {ne}. Retrying next cycle.", exc_info=True)
                except ccxt_async.ExchangeError as ee: # Use async version exceptions
                     symbol_logger.error(f"Exchange Error for {symbol}: {ee}. Retrying next cycle.", exc_info=True)
                except Exception as symbol_err:
                    symbol_logger.error(
                        f"!!! Unhandled Exception during analysis for {symbol}: {symbol_err} !!!",
                        exc_info=True
                    )
                    symbol_logger.warning("Attempting to continue to next symbol/cycle.")
                finally:
                    # Optional: await asyncio.sleep(CONFIG.get("delay_between_symbols", 0.5)) # Add small delay between symbols if needed
                    symbol_logger.info(f"Finished processing symbol: {symbol}\n")


            if shutdown_requested:
                init_logger.info("Shutdown requested, exiting main loop after current cycle.")
                break

            loop_end_time = time.monotonic()
            elapsed_time = loop_end_time - loop_start_time
            sleep_duration = max(0.1, loop_interval_seconds - elapsed_time) # Ensure minimal sleep

            init_logger.debug(f"Loop cycle finished. Elapsed: {elapsed_time:.2f}s. Sleeping for: {sleep_duration:.2f}s.")
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration) # Use asyncio.sleep
            if elapsed_time > loop_interval_seconds:
                init_logger.warning(f"Loop cycle duration ({elapsed_time:.2f}s) exceeded target interval ({loop_interval_seconds}s).")

    except Exception as loop_err:
        init_logger.critical(
            f"!!! CRITICAL UNHANDLED EXCEPTION IN MAIN ASYNC LOOP: {loop_err} !!!",
            exc_info=True
        )
        init_logger.critical("The bot encountered a critical error and will exit.")
    finally:
        init_logger.info("--- XR Scalper Bot Shutting Down ---")
        # Add any async cleanup specific to the exchange if needed, e.g., await exchange.close()
        if exchange and hasattr(exchange, 'close') and callable(exchange.close):
            try:
                init_logger.info(f"Closing exchange connection for {exchange.id}...")
                await exchange.close() # Important for async ccxt exchanges
                init_logger.info("Exchange connection closed.")
            except Exception as ex_close_err:
                init_logger.error(f"Error closing exchange connection: {ex_close_err}", exc_info=True)

        # Logging shutdown should happen after all handlers are done.
        # It's called implicitly on exit, but explicit call is fine.
        logging.shutdown()


# --- Script Entry Point ---
# These globals are set before main() is called.
load_dotenv()
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")

if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        print(f"{NEON_RED}CRITICAL ERROR: BYBIT_API_KEY and BYBIT_API_SECRET must be set in the .env file.{RESET}", file=sys.stderr) # Use color codes
        sys.exit(1)

    try:
        # Run the main async function
        asyncio.run(main())
    except KeyboardInterrupt:
        # This handles Ctrl+C if it bypasses the signal handler or occurs during asyncio.run setup/teardown
        print(f"\n{NEON_YELLOW}Bot execution interrupted by user (KeyboardInterrupt). Finalizing shutdown.{RESET}", file=sys.stderr) # Use color codes
        # Ensure shutdown_requested is set if not already, for any lingering tasks (though asyncio.run will exit)
        shutdown_requested = True
    except Exception as e:
        print(f"{NEON_RED}CRITICAL UNHANDLED ERROR at script execution entry point: {e}{RESET}", file=sys.stderr) # Use color codes
        traceback.print_exc(file=sys.stderr)
        sys.exit(1) # Ensure exit on critical error
    finally:
        # This print will occur after main() has completed or been interrupted.
        # logging.shutdown() should have been called within main's finally block.
        print(f"{NEON_GREEN}XR Scalper Bot script execution has concluded.{RESET}", file=sys.stderr) # Use color codes