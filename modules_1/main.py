```python
# File: main.py
"""
Main execution script for the XR Scalper Trading Bot.

Initializes and runs the bot, managing configuration, logging,
exchange connection (via BybitAPI client), and the main trading loop.
"""

import asyncio
import importlib.metadata # For getting package versions (Python 3.8+)
import importlib.util # For checking module availability
import logging
import os
import signal
import sys
import time # For monotonic clock loop timing
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Minimal color constants for pre-logging critical errors
_PRE_LOGGING_ERROR_RED = "\033[1;91m"
_PRE_LOGGING_RESET_COLOR = "\033[0m"

# --- Import Third-Party Libraries ---
try:
    import ccxt.async_support as ccxt_async
    from dotenv import load_dotenv
    import pandas as pd
    import pandas_ta as ta
except ImportError as e:
    print(
        f"{_PRE_LOGGING_ERROR_RED}CRITICAL ERROR: Missing required library: {e}. "
        f"Please install dependencies (e.g., pip install -r requirements.txt){_PRE_LOGGING_RESET_COLOR}",
        file=sys.stderr,
    )
    sys.exit(1)

# --- Import Custom Modules ---
try:
    import config_loader
    from exchange_api import BybitAPI
    import logger_setup
    import trading_strategy
    import utils
except ImportError as e:
    print(
        f"{_PRE_LOGGING_ERROR_RED}CRITICAL ERROR: Failed to import one or more custom modules: {e}{_PRE_LOGGING_RESET_COLOR}",
        file=sys.stderr,
    )
    print(
        f"Ensure config_loader.py, exchange_api.py, logger_setup.py, "
        f"trading_strategy.py, utils.py exist.{_PRE_LOGGING_RESET_COLOR}",
        file=sys.stderr,
    )
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
except Exception as e: # Catch any other unexpected errors during custom module imports
    print(
        f"{_PRE_LOGGING_ERROR_RED}CRITICAL ERROR: Unexpected error during custom module import: {e}{_PRE_LOGGING_RESET_COLOR}",
        file=sys.stderr,
    )
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

# --- Constants ---
DEFAULT_LOOP_INTERVAL_SECONDS = 60.0 # Default interval for the main trading loop

# Color constants from utils, available after successful import
NEON_GREEN = utils.NEON_GREEN
NEON_YELLOW = utils.NEON_YELLOW
NEON_RED = utils.NEON_RED
NEON_PURPLE = utils.NEON_PURPLE
RESET = utils.RESET_ALL_STYLE

# --- Global State for Shutdown ---
shutdown_requested = False # Flag to signal graceful shutdown

def signal_handler(signum: int, _frame: Any) -> None:
    """
    Sets the global shutdown_requested flag upon receiving SIGINT or SIGTERM.
    Ensures that the shutdown process is initiated only once.
    """
    global shutdown_requested
    if not shutdown_requested:
        shutdown_requested = True
        # Use print to stderr as logging might be compromised during shutdown
        print(
            f"\n{NEON_YELLOW}Signal {signal.Signals(signum).name} received. "
            f"Requesting graceful shutdown...{RESET}",
            file=sys.stderr,
        )
    else:
        print(
            f"\n{NEON_YELLOW}Shutdown already in progress. Please wait.{RESET}",
            file=sys.stderr,
        )

def _get_dependency_versions() -> Dict[str, str]:
    """Helper function to gather versions of key dependencies."""
    versions = {
        "Python": sys.version.split()[0],
        "CCXT": "N/A",
        "Pandas": "N/A",
        "PandasTA": "N/A",
    }
    try:
        versions["CCXT"] = importlib.metadata.version("ccxt")
    except importlib.metadata.PackageNotFoundError:
        versions["CCXT"] = getattr(ccxt_async, "__version__", "N/A") # Fallback for older CCXT

    try:
        versions["Pandas"] = pd.__version__
    except AttributeError:
        pass # Keep N/A

    try:
        if hasattr(ta, "__version__"):
            versions["PandasTA"] = getattr(ta, "__version__", "N/A")
        elif hasattr(ta, "version"): # Older pandas_ta version attribute
            if callable(ta.version):
                versions["PandasTA"] = ta.version()
            elif isinstance(ta.version, str):
                versions["PandasTA"] = ta.version
    except Exception:
        pass # Keep N/A

    return versions

def _validate_trading_settings(
    config: Dict[str, Any], logger: logging.Logger
) -> Tuple[bool, List[str], str, float]:
    """
    Validates critical trading settings from the configuration.

    Args:
        config: The application configuration dictionary.
        logger: The logger instance for reporting errors or info.

    Returns:
        A tuple: (is_valid, symbols, interval_str, risk_per_trade_validated).
                 is_valid is False if any critical setting is invalid.
    """
    symbols = config.get("symbols_to_trade", [])
    interval_str = str(config.get("interval", ""))
    leverage = config.get("leverage", 0) # Default to 0 for easy type check
    entry_type = config.get("entry_order_type", "")
    risk_per_trade_config = config.get("risk_per_trade", -1.0) # Default to invalid

    valid_config = True

    if not symbols:
        logger.critical("Configuration error: 'symbols_to_trade' cannot be empty.")
        valid_config = False
    if not interval_str or interval_str not in utils.CCXT_INTERVAL_MAP:
        logger.critical(f"Configuration error: Invalid 'interval': '{interval_str}'.")
        valid_config = False
    if not isinstance(leverage, (int, float)) or leverage <= 0:
        logger.critical(f"Configuration error: Invalid 'leverage': {leverage}. Must be a positive number.")
        valid_config = False
    if entry_type not in ["market", "limit", "conditional"]:
        logger.critical(
            f"Configuration error: Invalid 'entry_order_type': '{entry_type}'. "
            "Allowed values are 'market', 'limit', 'conditional'."
        )
        valid_config = False

    risk_per_trade_validated: float = -1.0
    try:
        risk_per_trade_validated = float(risk_per_trade_config)
        if not (0.0 <= risk_per_trade_validated <= 1.0): # Range 0.0 (0%) to 1.0 (100%)
            raise ValueError("Risk per trade must be between 0.0 and 1.0 (inclusive).")
    except (ValueError, TypeError) as e:
        logger.critical(
            f"Configuration error: Invalid 'risk_per_trade': '{risk_per_trade_config}'. Error: {e}"
        )
        valid_config = False

    if valid_config:
        logger.info("--- Critical Trading Settings Validated ---")
        logger.info(f"  Symbols: {symbols}")
        logger.info(f"  Interval: {interval_str} (CCXT: {utils.CCXT_INTERVAL_MAP.get(interval_str)})")
        logger.info(f"  Entry Type: {entry_type}")
        logger.info(f"  Risk/Trade: {risk_per_trade_validated * 100:.2f}%")
        logger.info(f"  Leverage: {leverage}x")
        logger.info(f"  Trailing Stop Loss Enabled: {config.get('enable_trailing_stop', False)}")
        logger.info(f"  Break Even Stop Enabled: {config.get('enable_break_even', False)}")
        logger.info(
            f"  Time-Based Exit: {config.get('time_based_exit_minutes') or 'Disabled'} minutes"
        )
        logger.info("-----------------------------------------")
    else:
        logger.critical("Critical trading settings validation failed. Bot cannot start in trading mode.")

    return valid_config, symbols, interval_str, risk_per_trade_validated


async def _process_symbol_task(
    api_client: BybitAPI,
    symbol: str,
    config: Dict[str, Any],
    enable_trading: bool,
) -> bool:
    """
    Wrapper to process one symbol and handle its specific errors.
    Returns True if an AuthenticationError occurred, signaling a need for shutdown.
    """
    # Sanitize symbol name for logger (replace special chars common in trading pairs)
    safe_symbol_name = symbol.replace("/", "_").replace(":", "-")
    symbol_logger = logging.getLogger(f"App.Symbol.{safe_symbol_name}")
    symbol_logger.debug(f"--- Processing: {symbol} ---")

    try:
        await trading_strategy.analyze_and_trade_symbol(
            api_client, symbol, config, symbol_logger, enable_trading
        )
    except ccxt_async.AuthenticationError as e:
        symbol_logger.critical(f"Authentication Error for {symbol}: {e}! This is critical.")
        return True # Signal fatal error
    except (ccxt_async.NetworkError, ccxt_async.RequestTimeout, asyncio.TimeoutError) as e:
        symbol_logger.warning(f"Network or Timeout issue for {symbol}: {e}")
    except ccxt_async.RateLimitExceeded as e:
        symbol_logger.warning(f"Rate Limit Exceeded for {symbol}: {e}")
    except ccxt_async.ExchangeError as e:
        # Log specific exchange errors without full traceback unless in debug mode
        symbol_logger.error(f"Exchange Error for {symbol}: {e}", exc_info=symbol_logger.isEnabledFor(logging.DEBUG))
    except Exception as e:
        symbol_logger.error(
            f"!!! Unhandled error during processing of symbol {symbol}: {e} !!!",
            exc_info=True,
        )
    finally:
        symbol_logger.debug(f"--- Finished processing: {symbol} ---")
    return False # No fatal error


# --- Main Execution Function ---
async def main() -> None:
    """
    Main asynchronous function: loads configuration, sets up logging,
    initializes the BybitAPI client, validates critical settings,
    and runs the main trading or analysis loop.
    """
    global shutdown_requested

    # --- Load Configuration ---
    config: Dict[str, Any] = {}
    try:
        config = config_loader.load_config()
        # Augment config with API keys from environment variables if not already present
        config["api_key"] = config.get("api_key") or os.getenv("BYBIT_API_KEY")
        config["api_secret"] = config.get("api_secret") or os.getenv("BYBIT_API_SECRET")
        # Use print here as logger is not yet configured
        print(
            f"{NEON_GREEN}Configuration loaded from '{utils.CONFIG_FILE}'.{RESET}",
            file=sys.stderr,
        )
    except FileNotFoundError:
        print(
            f"{_PRE_LOGGING_ERROR_RED}CRITICAL: Configuration file '{utils.CONFIG_FILE}' not found. Exiting.{_PRE_LOGGING_RESET_COLOR}",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(
            f"{_PRE_LOGGING_ERROR_RED}CRITICAL: Error loading configuration: {e}. Exiting.{_PRE_LOGGING_RESET_COLOR}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # --- Configure Logging ---
    try:
        logger_setup.configure_logging(config)
    except Exception as e:
        print(
            f"{_PRE_LOGGING_ERROR_RED}CRITICAL: Error configuring logging: {e}. Exiting.{_PRE_LOGGING_RESET_COLOR}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    init_logger = logging.getLogger("App.Init")
    init_logger.info(f"--- {NEON_PURPLE}Initializing XR Scalper Bot (Async Version){RESET} ---")

    # --- Timezone Setup ---
    configured_timezone = os.getenv("TIMEZONE", config.get("timezone", utils.DEFAULT_TIMEZONE))
    try:
        utils.set_timezone(configured_timezone)
        init_logger.info(f"Using Timezone: {utils.get_timezone()}")
    except Exception as e:
        init_logger.warning(
            f"Failed to set timezone to '{configured_timezone}': {e}. Using system default or UTC.",
            exc_info=True,
        )

    # --- Sensitive Data Masking for Logs ---
    try:
        formatter_class = getattr(utils, "SensitiveFormatter", None)
        if formatter_class and hasattr(formatter_class, "set_sensitive_data"):
            if config.get("api_key") and config.get("api_secret"):
                formatter_class.set_sensitive_data(config["api_key"], config["api_secret"])
                init_logger.debug("Sensitive data masking for logs has been configured.")
            else:
                init_logger.warning("API key/secret not found in config; log masking may be incomplete.")
        else:
            init_logger.warning(
                "SensitiveFormatter or set_sensitive_data method not found in utils. Log masking unavailable."
            )
    except Exception as e:
        init_logger.warning(f"Error configuring sensitive data masking for logs: {e}", exc_info=True)

    # --- Log Startup Information ---
    try:
        startup_dt_str = datetime.now(utils.get_timezone()).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception: # Fallback if timezone info is problematic
        startup_dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    init_logger.info(f"Startup Time: {startup_dt_str}")
    init_logger.info(f"Quote Currency: {config.get('quote_currency', 'N/A')}")

    # --- Log Dependency Versions ---
    try:
        versions = _get_dependency_versions()
        version_str = ", ".join([f"{lib}={ver}" for lib, ver in versions.items()])
        init_logger.info(f"Core Dependency Versions: {version_str}")
    except Exception as e:
        init_logger.warning(f"Could not determine all dependency versions: {e}", exc_info=True)


    # --- Validate Settings & Determine Operating Mode ---
    enable_trading = config.get("enable_trading", False)
    use_sandbox = config.get("use_sandbox", True) # Default to sandbox for safety
    symbols_to_process: List[str] = []
    loop_interval_seconds = float(config.get("loop_interval_seconds", DEFAULT_LOOP_INTERVAL_SECONDS))

    init_logger.info(f"Trading Enabled: {NEON_GREEN if enable_trading else NEON_RED}{enable_trading}{RESET}")
    init_logger.info(f"Using Sandbox Environment: {NEON_YELLOW if use_sandbox else NEON_RED}{use_sandbox}{RESET}")

    if enable_trading:
        if not use_sandbox:
            init_logger.warning(
                f"{NEON_RED}!!! CAUTION: LIVE TRADING WITH REAL MONEY IS ENABLED !!!{RESET}"
            )
        else:
            init_logger.info(f"{NEON_YELLOW}Operating in SANDBOX mode (simulated trading).{RESET}")

        valid_settings, trading_symbols, _, _ = _validate_trading_settings(config, init_logger)
        if not valid_settings:
            init_logger.critical("Due to invalid critical settings, the bot cannot start in trading mode. Exiting.")
            return # Exit main function
        symbols_to_process = trading_symbols
        init_logger.info("Proceeding with trading operations. Countdown to start: 5 seconds...")
        await asyncio.sleep(5) # Brief pause before trading starts
    else: # Analysis-only mode
        init_logger.info("Live trading is DISABLED. Bot will run in analysis-only mode.")
        symbols_to_process = config.get("symbols_to_trade", [])
        analysis_interval = str(config.get("interval", ""))

        if not symbols_to_process:
            init_logger.error("Configuration error: 'symbols_to_trade' is empty. Cannot run analysis. Exiting.")
            return
        if not analysis_interval or analysis_interval not in utils.CCXT_INTERVAL_MAP:
            init_logger.critical(
                f"Configuration error: Invalid 'interval' for analysis: '{analysis_interval}'. Exiting."
            )
            return
        init_logger.info(f"Analysis Symbols: {symbols_to_process}")
        init_logger.info(f"Analysis Interval: {analysis_interval} (CCXT: {utils.CCXT_INTERVAL_MAP.get(analysis_interval)})")


    # --- Initialize API Client ---
    api_client: Optional[BybitAPI] = None
    try:
        init_logger.info("Initializing API Client...")
        api_client = BybitAPI(config, init_logger) # Pass logger for API client's internal use
        if not await api_client.initialize():
            init_logger.critical("API Client initialization failed. Please check credentials and connection. Exiting.")
            return
        init_logger.info(f"API Client for '{api_client.exchange.id}' initialized successfully.")
    except ValueError as ve: # Specific error for config issues in BybitAPI
        init_logger.critical(f"API Client configuration error: {ve}. Exiting.")
        return
    except Exception as e:
        init_logger.critical(f"Failed to initialize API Client: {e}", exc_info=True)
        return


    # --- Register Signal Handlers for Graceful Shutdown ---
    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: signal_handler(s, None))
        init_logger.info("Signal handlers for SIGINT and SIGTERM registered with the event loop.")
    except (NotImplementedError, AttributeError, ValueError) as e:
        # Fallback for environments where add_signal_handler is not available (e.g., Windows Python < 3.8)
        init_logger.warning(
            f"Could not use loop.add_signal_handler ({e}). Using signal.signal() as fallback. "
            "Note: Fallback might not be as robust for async operations."
        )
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            init_logger.info("Fallback signal handlers registered using signal.signal().")
        except Exception as e_sig:
            init_logger.error(f"Failed to register fallback signal handlers: {e_sig}. Shutdown might be abrupt.")


    # --- Main Processing Loop ---
    if not symbols_to_process: # Final check, redundant but safe
        init_logger.error("No symbols configured for processing. Exiting.")
        if api_client: await api_client.close()
        return

    if loop_interval_seconds <= 0:
        init_logger.critical(f"Invalid loop_interval_seconds: {loop_interval_seconds}. Must be > 0. Exiting.")
        if api_client: await api_client.close()
        return

    init_logger.info(
        f"Starting main processing loop for symbols: {symbols_to_process}. "
        f"Loop interval: {loop_interval_seconds:.2f} seconds."
    )

    try:
        while not shutdown_requested:
            loop_start_time = time.monotonic()
            current_dt_str = datetime.now(utils.get_timezone()).strftime("%Y-%m-%d %H:%M:%S %Z")
            init_logger.debug(f"--- Main Loop Cycle Start @ {current_dt_str} ---")

            # Process symbols sequentially
            for symbol in symbols_to_process:
                if shutdown_requested:
                    init_logger.info(f"Shutdown requested, interrupting symbol processing for {symbol}.")
                    break

                auth_error_occurred = await _process_symbol_task(
                    api_client, symbol, config, enable_trading
                )
                if auth_error_occurred:
                    init_logger.critical("Critical authentication error detected. Initiating immediate shutdown.")
                    shutdown_requested = True # Signal main loop to terminate
                    break # Exit symbol processing loop

            if shutdown_requested:
                init_logger.info("Shutdown detected, exiting main processing loop.")
                break

            # Calculate sleep duration to maintain loop interval
            elapsed_seconds = time.monotonic() - loop_start_time
            sleep_duration = max(0.1, loop_interval_seconds - elapsed_seconds) # Ensure minimum 0.1s sleep

            init_logger.debug(
                f"Loop cycle complete. Elapsed: {elapsed_seconds:.2f}s. "
                f"Sleeping for: {sleep_duration:.2f}s."
            )
            if elapsed_seconds > loop_interval_seconds:
                init_logger.warning(
                    f"Loop cycle duration ({elapsed_seconds:.1f}s) exceeded configured interval "
                    f"({loop_interval_seconds:.1f}s). Consider increasing interval or optimizing tasks."
                )
            await asyncio.sleep(sleep_duration)

    except asyncio.CancelledError:
        init_logger.info("Main processing loop was cancelled.")
    except Exception as loop_err:
        init_logger.critical(f"!!! CRITICAL UNHANDLED ERROR IN MAIN LOOP: {loop_err} !!!", exc_info=True)
    finally:
        init_logger.info(f"--- {NEON_PURPLE}Bot Shutting Down Gracefully...{RESET} ---")
        if api_client:
            init_logger.info("Closing API client connection...")
            await api_client.close()
            init_logger.info("API client connection closed.")
        init_logger.info("Flushing logs and shutting down logging system.")
        logging.shutdown() # Ensure all log handlers are closed properly


# --- Script Entry Point ---
if __name__ == "__main__":
    load_dotenv() # Load environment variables from .env file first

    # Pre-flight check for API keys before starting the async event loop.
    # This checks .env and config file (if load_config is light enough for a quick check).
    # Note: config_loader.load_config() might be called again in main(), consider implications.
    # For this check, it's about failing fast if keys are nowhere to be found.
    try:
        temp_config_for_check = config_loader.load_config()
    except Exception:
        temp_config_for_check = {} # If config file itself is an issue, rely on env vars

    api_key_present = bool(os.getenv("BYBIT_API_KEY") or temp_config_for_check.get("api_key"))
    api_secret_present = bool(os.getenv("BYBIT_API_SECRET") or temp_config_for_check.get("api_secret"))

    if not (api_key_present and api_secret_present):
        print(
            f"{_PRE_LOGGING_ERROR_RED}CRITICAL ERROR: API key and/or secret not found in environment variables "
            f"or configuration file. Please ensure they are set.{_PRE_LOGGING_RESET_COLOR}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        asyncio.run(main())
    except KeyboardInterrupt: # Handles Ctrl+C if signal handlers didn't catch it or during setup
        print(
            f"\n{NEON_YELLOW}KeyboardInterrupt detected during execution. Exiting abruptly...{RESET}",
            file=sys.stderr,
        )
    except Exception as e: # Catch-all for unexpected synchronous errors during asyncio.run or setup
        print(
            f"{_PRE_LOGGING_ERROR_RED}CRITICAL RUNTIME ERROR: An unexpected error occurred: {e}{_PRE_LOGGING_RESET_COLOR}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    finally:
        # This message prints after asyncio.run() completes or if an exception occurred
        print(
            f"{NEON_GREEN}Script execution has concluded.{RESET}",
            file=sys.stderr,
        )
```