# main.py
import asyncio
import logging
import signal
import sys

# Import local modules
try:
    import bybit_helpers as bybit  # Needed for closing exchange in finally
    import config as cfg  # Load configuration
    from neon_logger import setup_logger  # Use the logger setup
    from strategy import ExampleMACDRSIStrategy  # Import the specific strategy class

    # Colorama for initial messages if logger fails
    try:
        from colorama import Back, Fore, Style, init

        init(autoreset=True)  # Initialize Colorama
        COLORAMA_AVAILABLE = True
    except ImportError:

        class DummyColor:
            def __getattr__(self, name: str) -> str:
                return ""

        Fore = Style = Back = DummyColor()
        COLORAMA_AVAILABLE = False
except ImportError as e:
    err_back = Back.RED if COLORAMA_AVAILABLE else ""
    err_fore = Fore.WHITE if COLORAMA_AVAILABLE else ""
    reset_all = Style.RESET_ALL if COLORAMA_AVAILABLE else ""
    print(
        f"{err_back}{err_fore}FATAL: Failed to import core modules (config, neon_logger, strategy, bybit_helpers): {e}{reset_all}",
        file=sys.stderr,
    )
    print("Ensure all required .py files are in the correct location.", file=sys.stderr)
    sys.exit(1)

# --- Global Variables ---
# Use type hints for clarity
logger: logging.Logger | None = None
strategy_instance: ExampleMACDRSIStrategy | None = None
main_task: asyncio.Task | None = None
exchange_instance_ref: bybit.ccxt.bybit | None = None  # Keep reference for final cleanup


# --- Signal Handling for Graceful Shutdown ---
shutdown_requested = False


def handle_signal(sig, frame):
    global shutdown_requested, logger, main_task, strategy_instance
    if shutdown_requested:  # Prevent duplicate handling
        print("Shutdown already in progress...")
        return
    shutdown_requested = True

    signal_name = signal.Signals(sig).name
    print(
        f"\n{Fore.YELLOW}{Style.BRIGHT}>>> Signal {signal_name} ({sig}) received. Initiating graceful shutdown... <<< {Style.RESET_ALL}"
    )
    if logger:
        logger.warning(f"Signal {signal_name} ({sig}) received. Initiating graceful shutdown...")

    # Request strategy loop to stop
    if strategy_instance and strategy_instance.is_running:
        if logger:
            logger.info("Requesting strategy loop stop...")
        # Run stop in a new task to avoid blocking signal handler
        asyncio.create_task(strategy_instance.stop())
    else:
        if logger:
            logger.info("Strategy instance not found or not running.")

    # Attempt to cancel the main task (run_loop)
    if main_task and not main_task.done():
        if logger:
            logger.info("Cancelling main strategy task...")
        main_task.cancel()

    # Note: Further cleanup (like exchange close) is handled in main()'s finally block


async def main():
    """Main async function to setup and run the strategy."""
    global logger, strategy_instance, main_task, exchange_instance_ref

    # --- 1. Setup Logger ---
    try:
        # Ensure LOGGING_CONFIG exists and has necessary keys
        log_config = getattr(cfg, "LOGGING_CONFIG", {})
        log_console_level_str = log_config.get("CONSOLE_LEVEL_STR", "INFO").upper()
        log_file_level_str = log_config.get("FILE_LEVEL_STR", "DEBUG").upper()
        log_third_party_level_str = log_config.get("THIRD_PARTY_LOG_LEVEL_STR", "WARNING").upper()

        log_console_level = logging.getLevelName(log_console_level_str)
        log_file_level = logging.getLevelName(log_file_level_str)
        log_third_party_level = logging.getLevelName(log_third_party_level_str)

        logger = setup_logger(
            logger_name=log_config.get("LOGGER_NAME", "TradingBot"),
            log_file=log_config.get("LOG_FILE", "trading_bot.log"),
            console_level=log_console_level if isinstance(log_console_level, int) else logging.INFO,
            file_level=log_file_level if isinstance(log_file_level, int) else logging.DEBUG,
            log_rotation_bytes=log_config.get("LOG_ROTATION_BYTES", 5 * 1024 * 1024),
            log_backup_count=log_config.get("LOG_BACKUP_COUNT", 5),
            third_party_log_level=log_third_party_level if isinstance(log_third_party_level, int) else logging.WARNING,
        )
        logger.info("=" * 60)
        logger.info(f"=== {log_config.get('LOGGER_NAME', 'Trading Bot')} Initializing ===")
        # Load actual config values (assuming config.py uses simple dicts or a class instance)
        # This assumes cfg is the module itself or an instance holding the dicts/attrs
        api_conf = getattr(cfg, "API_CONFIG", {})
        strat_conf = getattr(cfg, "STRATEGY_CONFIG", {})
        logger.info(f"Testnet Mode: {api_conf.get('TESTNET_MODE', 'N/A')}")
        logger.info(f"Symbol: {api_conf.get('SYMBOL', 'N/A')}")
        logger.info(f"Strategy: {strat_conf.get('name', 'N/A')}")
        logger.info(f"Timeframe: {strat_conf.get('timeframe', 'N/A')}")
        logger.info("=" * 60)

    except Exception as e:
        print(f"{Back.RED}{Fore.WHITE}FATAL: Logger setup failed: {e}{Style.RESET_ALL}", file=sys.stderr)
        sys.exit(1)

    # --- 2. Load Full Configuration ---
    # Assuming config.py defines API_CONFIG, STRATEGY_CONFIG, etc.
    # Create a single dictionary to pass to the strategy
    full_config = {
        "API_CONFIG": getattr(cfg, "API_CONFIG", {}),
        "STRATEGY_CONFIG": getattr(cfg, "STRATEGY_CONFIG", {}),
        "LOGGING_CONFIG": getattr(cfg, "LOGGING_CONFIG", {}),
        "SMS_CONFIG": getattr(cfg, "SMS_CONFIG", {}),
    }

    # --- 3. Instantiate Strategy ---
    try:
        strategy_instance = ExampleMACDRSIStrategy(config=full_config, logger=logger)
        # Keep a reference to the exchange instance created inside the strategy for final cleanup
        # The instance is created during strategy_instance._initialize()
    except Exception as e:
        logger.critical(f"Failed to instantiate strategy: {e}", exc_info=True)
        # No need to close exchange here as it likely wasn't created
        sys.exit(1)

    # --- 4. Run the Strategy Loop ---
    run_success = False
    try:
        logger.info("Starting strategy execution loop...")
        # Assign the task so signal handler can cancel it
        main_task = asyncio.create_task(strategy_instance.run_loop())
        # Keep reference to exchange instance after initialization inside strategy
        # This relies on _initialize() setting self.exchange correctly
        await asyncio.sleep(0.1)  # Allow loop to start and potentially initialize exchange
        exchange_instance_ref = strategy_instance.exchange

        await main_task
        # If main_task finishes without CancelledError or other exception, it might mean
        # the loop exited normally (e.g., self.is_running became False internally)
        run_success = True
        logger.info("Strategy run_loop finished normally.")

    except asyncio.CancelledError:
        logger.warning("Main strategy task was cancelled (likely due to shutdown signal).")
        # Allow finally block to handle cleanup
        run_success = True  # Consider cancellation successful completion for cleanup purposes

    except Exception as e:
        logger.critical(f"Strategy execution loop encountered an unhandled error: {e}", exc_info=True)
        run_success = False  # Mark as failed

    finally:
        logger.info("--- Main Execution Block Finalizing ---")
        # --- Final Cleanup ---
        # Ensure strategy cleanup is called if instance exists
        if strategy_instance:
            logger.info("Running strategy internal cleanup...")
            try:
                await strategy_instance._cleanup()
            except Exception as strategy_cleanup_err:
                logger.error(f"Error during strategy internal cleanup: {strategy_cleanup_err}", exc_info=True)

        # Final attempt to close the exchange connection using the reference
        if exchange_instance_ref and hasattr(exchange_instance_ref, "close") and callable(exchange_instance_ref.close):
            if not exchange_instance_ref.closed:
                logger.info("Attempting final exchange connection close...")
                try:
                    await exchange_instance_ref.close()
                    logger.info("Final exchange connection close successful.")
                except Exception as final_close_err:
                    logger.error(f"Error during final exchange connection close: {final_close_err}", exc_info=True)
            else:
                logger.info("Exchange connection already closed.")
        else:
            logger.warning("Exchange instance reference not available for final cleanup or close method missing.")

        logger.info(f"Run success status: {run_success}")


# --- Script Entry Point ---
if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
    for s in signals:
        try:
            # Use add_signal_handler for async compatibility
            loop.add_signal_handler(s, lambda s=s: asyncio.create_task(handle_signal(s, None)))
            # loop.add_signal_handler(s, handle_signal, s, None) # Alternative if lambda causes issues
        except NotImplementedError:
            # Windows might not support all signals
            print(f"Warning: Signal {s} registration not supported on this platform.", file=sys.stderr)

    print(f"{Fore.CYAN}Starting Asynchronous Trading Bot...{Style.RESET_ALL}")
    start_time = time.monotonic()

    try:
        # Run the main asynchronous function
        asyncio.run(main())

    except KeyboardInterrupt:
        # This catches Ctrl+C if it happens *before* the signal handler is registered
        # or if the handler fails to stop the loop quickly enough.
        print(f"\n{Fore.YELLOW}KeyboardInterrupt caught at top level. Exiting immediately.{Style.RESET_ALL}")
        # Cleanup might not fully complete here

    except Exception as top_level_exc:
        # Catch errors during asyncio.run() itself if main() fails very early
        print(
            f"{Back.RED}{Fore.WHITE}FATAL UNHANDLED ERROR during asyncio.run(): {top_level_exc}{Style.RESET_ALL}",
            file=sys.stderr,
        )
        if logger:  # Try logging if logger exists
            logger.critical("Fatal unhandled error during asyncio.run()", exc_info=True)
        sys.exit(1)  # Exit with error status

    finally:
        end_time = time.monotonic()
        print(
            f"{Fore.CYAN}Application shutdown complete. Total runtime: {end_time - start_time:.2f} seconds.{Style.RESET_ALL}"
        )
        if logger:
            logger.info(f"--- Application Shutdown Complete (Runtime: {end_time - start_time:.2f}s) ---")
