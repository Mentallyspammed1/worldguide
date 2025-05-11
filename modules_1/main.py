# File: main.py
"""
Main execution script for the XR Scalper Trading Bot.

Initializes and runs the bot, managing configuration, logging,
exchange connection (via BybitAPI client), and the main trading loop.
"""

import logging
import os
import sys
import time  # For monotonic clock loop timing
import signal
import traceback
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
import importlib.util  # For checking module availability
import importlib.metadata  # For getting package versions (Python 3.8+)

# Import third-party libraries
try:
    import pandas as pd
    import pandas_ta as ta
    import ccxt.async_support as ccxt_async  # Keep for type hints if needed
    from dotenv import load_dotenv
except ImportError as e:
    print(
        f"\033[1;91mCRITICAL ERROR: Missing required library: {e}. Please install dependencies (e.g., pip install -r requirements.txt)\033[0m",
        file=sys.stderr,
    )
    sys.exit(1)

# --- Import Custom Modules ---
try:
    import config_loader
    from exchange_api import BybitAPI  # Import the API client class
    import logger_setup
    import trading_strategy
    import utils
except ImportError as e:
    _NEON_RED = "\033[1;91m"
    _RESET = "\033[0m"  # Fallback colors
    print(
        f"{_NEON_RED}CRITICAL ERROR: Failed to import one or more custom modules: {e}{_RESET}",
        file=sys.stderr,
    )
    print(
        f"Ensure config_loader.py, exchange_api.py, logger_setup.py, trading_strategy.py, utils.py exist.{_RESET}",
        file=sys.stderr,
    )
    if "traceback" not in sys.modules:
        import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
except Exception as e:
    _NEON_RED = "\033[1;91m"
    _RESET = "\033[0m"
    print(
        f"{_NEON_RED}CRITICAL ERROR: Unexpected error during module import: {e}{_RESET}",
        file=sys.stderr,
    )
    if "traceback" not in sys.modules:
        import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

# --- Constants ---
DEFAULT_LOOP_INTERVAL_SECONDS = 60
# Use constants from utils after ensuring it's imported
NEON_GREEN = utils.NEON_GREEN
NEON_YELLOW = utils.NEON_YELLOW
NEON_RED = utils.NEON_RED
NEON_PURPLE = utils.NEON_PURPLE
RESET = utils.RESET_ALL_STYLE

# --- Global State for Shutdown ---
shutdown_requested = False  # Defined globally


def signal_handler(signum: int, frame: Any) -> None:
    """Sets the *global* shutdown flag on receiving SIGINT or SIGTERM."""
    global shutdown_requested  # Indicate modification of the global variable
    if not shutdown_requested:
        shutdown_requested = True
        print(
            f"\n{NEON_YELLOW}Signal {signal.Signals(signum).name} received. Requesting graceful shutdown...{RESET}",
            file=sys.stderr,
        )
    else:
        print(
            f"\n{NEON_YELLOW}Shutdown already requested. Please wait.{RESET}",
            file=sys.stderr,
        )


# --- Main Execution Function ---
async def main() -> None:
    """
    Main async function: loads config, sets up logging, initializes the
    BybitAPI client, validates critical settings, and runs the trading loop.
    """
    global shutdown_requested  # <--- ADD THIS LINE TO ACCESS THE GLOBAL VARIABLE

    # --- Load Configuration ---
    CONFIG: Dict[str, Any] = {}
    try:
        CONFIG = config_loader.load_config()
        # Add API keys from env to config if not present
        if not CONFIG.get("api_key") and os.getenv("BYBIT_API_KEY"):
            CONFIG["api_key"] = os.getenv("BYBIT_API_KEY")
        if not CONFIG.get("api_secret") and os.getenv("BYBIT_API_SECRET"):
            CONFIG["api_secret"] = os.getenv("BYBIT_API_SECRET")
        print(
            f"{NEON_GREEN}Configuration loaded from '{utils.CONFIG_FILE}'.{RESET}",
            file=sys.stderr,
        )
    except FileNotFoundError:
        print(
            f"{NEON_RED}CRITICAL: Config file '{utils.CONFIG_FILE}' not found.{RESET}",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"{NEON_RED}CRITICAL: Error loading config: {e}.{RESET}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # --- Configure Logging ---
    try:
        logger_setup.configure_logging(CONFIG)
    except Exception as e:
        print(
            f"{NEON_RED}CRITICAL: Error configuring logging: {e}. Exiting.{RESET}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    init_logger = logging.getLogger("App.Init")  # Use hierarchical name
    init_logger.info(
        f"--- {NEON_PURPLE}Initializing XR Scalper Bot (Async Version){RESET} ---"
    )

    # --- Timezone Setup ---
    configured_timezone = os.getenv(
        "TIMEZONE", CONFIG.get("timezone", utils.DEFAULT_TIMEZONE)
    )
    try:
        utils.set_timezone(configured_timezone)
        init_logger.info(f"Using Timezone: {utils.get_timezone()}")
    except Exception as e:
        init_logger.warning(
            f"Failed set timezone '{configured_timezone}': {e}.", exc_info=True
        )

    # --- Sensitive Data Masking ---
    try:
        if hasattr(utils, "SensitiveFormatter") and hasattr(
            utils.SensitiveFormatter, "set_sensitive_data"
        ):
            if CONFIG.get("api_key") and CONFIG.get("api_secret"):
                utils.SensitiveFormatter.set_sensitive_data(
                    CONFIG["api_key"], CONFIG["api_secret"]
                )
                init_logger.debug("Sensitive data masking configured.")
            else:
                init_logger.warning("API keys not found in config, masking incomplete.")
        else:
            init_logger.warning("SensitiveFormatter not found.")
    except Exception as e:
        init_logger.warning(f"Error config sensitive masking: {e}", exc_info=True)

    # --- Log Startup Info ---
    try:
        init_logger.info(
            f"Startup Time: {datetime.now(utils.get_timezone()).strftime('%Y-%m-%d %H:%M:%S %Z')}"
        )
    except Exception:
        init_logger.info(
            f"Startup Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
    init_logger.info(f"Quote Currency: {CONFIG.get('quote_currency', 'N/A')}")

    # --- Log Dependency Versions ---
    try:
        python_version = sys.version.split()[0]
        ccxt_version = (
            importlib.metadata.version("ccxt")
            if importlib.util.find_spec("importlib.metadata")
            else getattr(ccxt_async, "__version__", "N/A")
        )
        pandas_version = pd.__version__
        pandas_ta_version = "N/A"
        if hasattr(ta, "version"):
            if callable(ta.version):
                try:
                    pandas_ta_version = ta.version()
                except Exception:
                    pass
            elif isinstance(ta.version, str):
                pandas_ta_version = ta.version
        elif hasattr(ta, "__version__"):
            pandas_ta_version = getattr(ta, "__version__", "N/A")
        init_logger.info(
            f"Versions: Python={python_version}, CCXT={ccxt_version}, Pandas={pandas_version}, PandasTA={pandas_ta_version}"
        )
    except Exception as e:
        init_logger.warning(f"Error getting dependency versions: {e}", exc_info=True)

    # --- Trading Mode & Critical Settings Validation ---
    enable_trading = CONFIG.get("enable_trading", False)
    use_sandbox = CONFIG.get("use_sandbox", True)
    init_logger.info(
        f"Trading Enabled: {NEON_GREEN if enable_trading else NEON_RED}{enable_trading}{RESET}"
    )
    init_logger.info(
        f"Using Sandbox: {NEON_YELLOW if use_sandbox else NEON_RED}{use_sandbox}{RESET}"
    )

    if enable_trading:
        if not use_sandbox:
            init_logger.warning(f"{NEON_RED}!!! CAUTION: REAL MONEY !!!{RESET}")
        else:
            init_logger.info(f"{NEON_YELLOW}Operating in SANDBOX.{RESET}")
        # Validate critical settings
        symbols = CONFIG.get("symbols_to_trade", [])
        interval_str = str(CONFIG.get("interval", ""))
        leverage = CONFIG.get("leverage", 0)
        entry_type = CONFIG.get("entry_order_type", "")
        risk_pct_config = CONFIG.get("risk_per_trade", -1.0)
        valid_config = True
        if not symbols:
            init_logger.critical("'symbols_to_trade' empty.")
            valid_config = False
        if not interval_str or interval_str not in utils.CCXT_INTERVAL_MAP:
            init_logger.critical(f"Invalid interval '{interval_str}'.")
            valid_config = False
        if not isinstance(leverage, (int, float)) or leverage <= 0:
            init_logger.critical(f"Invalid leverage '{leverage}'.")
            valid_config = False
        if entry_type not in ["market", "limit", "conditional"]:
            init_logger.critical(f"Invalid 'entry_order_type': '{entry_type}'.")
            valid_config = False
        try:
            risk_pct = float(risk_pct_config)
            assert 0 <= risk_pct <= 1
        except:
            init_logger.critical(f"Invalid 'risk_per_trade': {risk_pct_config}.")
            valid_config = False
        if not valid_config:
            init_logger.critical("Invalid settings. Exiting.")
            return
        # Log validated settings
        init_logger.info("--- Critical Trading Settings ---")
        init_logger.info(f"  Symbols: {symbols}")
        init_logger.info(
            f"  Interval: {interval_str} ({utils.CCXT_INTERVAL_MAP.get(interval_str)})"
        )
        init_logger.info(f"  Entry Type: {entry_type}")
        init_logger.info(f"  Risk/Trade: {risk_pct * 100:.2f}%")
        init_logger.info(f"  Leverage: {leverage}x")
        init_logger.info(f"  TSL: {CONFIG.get('enable_trailing_stop', False)}")
        init_logger.info(f"  BE: {CONFIG.get('enable_break_even', False)}")
        init_logger.info(
            f"  Time Exit: {CONFIG.get('time_based_exit_minutes') or 'Disabled'}"
        )
        init_logger.info("---------------------------------")
        init_logger.info("Starting trading loop in 5 seconds...")
        await asyncio.sleep(5)
    else:  # Analysis mode
        init_logger.info("Live trading DISABLED. Analysis-only mode.")
        symbols_to_process = CONFIG.get("symbols_to_trade", [])
        if not symbols_to_process:
            init_logger.error("'symbols_to_trade' empty.")
            return
        interval_str = str(CONFIG.get("interval", ""))
        if not interval_str or interval_str not in utils.CCXT_INTERVAL_MAP:
            init_logger.critical(f"Invalid interval '{interval_str}'.")
            return
        init_logger.info(f"Analysis Symbols: {symbols_to_process}")
        init_logger.info(
            f"Analysis Interval: {interval_str} ({utils.CCXT_INTERVAL_MAP.get(interval_str)})"
        )

    # --- Initialize API Client ---
    api_client: Optional[BybitAPI] = None
    try:
        init_logger.info("Initializing API Client...")
        api_client = BybitAPI(CONFIG, init_logger)
        initialized_ok = await api_client.initialize()
        if not initialized_ok:
            init_logger.critical("API Client initialization failed. Exiting.")
            return
        init_logger.info(f"API Client '{api_client.exchange.id}' initialized.")
    except ValueError as ve:
        init_logger.critical(f"API Client config error: {ve}.")
        return
    except Exception as e:
        init_logger.critical(f"API Client init error: {e}", exc_info=True)
        return

    # --- Signal Handling ---
    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: signal_handler(s, None))
        init_logger.info("Signal handlers registered using loop.")
    except (NotImplementedError, AttributeError, ValueError) as e:  # Fallback/Warning
        init_logger.warning(
            f"Could not register signal handlers with loop ({e}). Using signal.signal fallback."
        )
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except Exception as e_sig:
            init_logger.error(f"Failed signal.signal registration: {e_sig}.")

    # --- Main Loop Setup ---
    try:
        loop_interval_seconds = float(
            CONFIG.get("loop_interval_seconds", DEFAULT_LOOP_INTERVAL_SECONDS)
        )
        assert loop_interval_seconds > 0
    except Exception as e:
        init_logger.critical(f"Invalid loop interval: {e}. Exiting.")
        await api_client.close()
        return
    symbols_to_process = CONFIG.get("symbols_to_trade", [])
    if not symbols_to_process:
        init_logger.error("No symbols configured. Exiting.")
        await api_client.close()
        return

    init_logger.info(
        f"Starting main processing loop for: {symbols_to_process}. Interval: {loop_interval_seconds:.2f}s"
    )

    # --- Main Trading Loop ---
    try:
        while not shutdown_requested:  # Access the global flag
            loop_start_time = time.monotonic()
            current_dt = datetime.now(utils.get_timezone())
            init_logger.debug(
                f"--- Loop Cycle Start @ {current_dt.strftime('%H:%M:%S %Z')} ---"
            )

            async def process_symbol_wrapper(symbol: str):
                """Wrapper to process one symbol and handle its errors."""
                safe_symbol_name = symbol.replace("/", "_").replace(":", "-")
                symbol_logger = logging.getLogger(
                    f"App.Symbol.{safe_symbol_name}"
                )  # Hierarchical name
                symbol_logger.debug(f"--- Processing: {symbol} ---")
                try:
                    await trading_strategy.analyze_and_trade_symbol(
                        api_client, symbol, CONFIG, symbol_logger, enable_trading
                    )
                except (
                    ccxt_async.NetworkError,
                    ccxt_async.RequestTimeout,
                    asyncio.TimeoutError,
                ) as e:
                    symbol_logger.warning(f"Network/Timeout: {e}")
                except ccxt_async.RateLimitExceeded as e:
                    symbol_logger.warning(f"Rate Limit: {e}")
                except ccxt_async.AuthenticationError as e:
                    symbol_logger.critical(f"Auth Error: {e}! SHUTTING DOWN.")
                    global shutdown_requested
                    shutdown_requested = True
                except ccxt_async.ExchangeError as e:
                    symbol_logger.error(f"Exchange Error: {e}", exc_info=False)
                except Exception as e:
                    symbol_logger.error(
                        f"!!! Unhandled Symbol Error: {e} !!!", exc_info=True
                    )
                finally:
                    symbol_logger.debug(f"--- Finished: {symbol} ---")

            # Process symbols sequentially
            for symbol in symbols_to_process:
                if shutdown_requested:
                    break
                await process_symbol_wrapper(symbol)

            if shutdown_requested:
                init_logger.info("Shutdown requested during cycle.")
                break

            # Loop Delay
            elapsed = time.monotonic() - loop_start_time
            sleep_duration = max(0.1, loop_interval_seconds - elapsed)
            init_logger.debug(
                f"Cycle End. Elapsed:{elapsed:.2f}s. Sleep:{sleep_duration:.2f}s."
            )
            if elapsed > loop_interval_seconds:
                init_logger.warning(
                    f"Loop duration > interval ({elapsed:.1f}s > {loop_interval_seconds:.1f}s)"
                )
            await asyncio.sleep(sleep_duration)

    except asyncio.CancelledError:
        init_logger.info("Main task cancelled.")
    except Exception as loop_err:
        init_logger.critical(
            f"!!! CRITICAL MAIN LOOP ERROR: {loop_err} !!!", exc_info=True
        )
    finally:
        init_logger.info(f"--- {NEON_PURPLE}Bot Shutting Down...{RESET} ---")
        if api_client:
            await api_client.close()
        logging.shutdown()


# --- Script Entry Point ---
if __name__ == "__main__":
    load_dotenv()
    # Final check for API keys
    api_key_present = bool(
        os.getenv("BYBIT_API_KEY")
    ) or config_loader.load_config().get("api_key")
    api_secret_present = bool(
        os.getenv("BYBIT_API_SECRET")
    ) or config_loader.load_config().get("api_secret")
    if not (api_key_present and api_secret_present):
        print(f"{NEON_RED}CRITICAL ERROR: API keys missing.{RESET}", file=sys.stderr)
        sys.exit(1)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{NEON_YELLOW}Ctrl+C detected. Exiting...{RESET}", file=sys.stderr)
    except Exception as e:
        print(f"{NEON_RED}CRITICAL Runtime Error: {e}{RESET}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    finally:
        print(f"{NEON_GREEN}Script execution concluded.{RESET}", file=sys.stderr)
