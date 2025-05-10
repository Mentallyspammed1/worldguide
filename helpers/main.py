# main.py
import asyncio
import logging
import signal
import sys
import time  # For timing runtime
import traceback  # For detailed error printing

# --- Define COLORAMA_AVAILABLE and Dummies EARLY ---
# This ensures they exist even if later imports fail before colorama try block
try:
    from colorama import Back, Fore, Style
    from colorama import init as colorama_init

    colorama_init(autoreset=True)  # Initialize Colorama automatically
    COLORAMA_AVAILABLE = True
except ImportError:

    class DummyColor:  # type: ignore
        def __getattr__(self, name: str) -> str:
            return ""

    Fore = Style = Back = DummyColor()  # type: ignore
    COLORAMA_AVAILABLE = False
    print(
        "Warning: 'colorama' library not found. Run 'pip install colorama' for vibrant logs.",
        file=sys.stderr,
    )

# --- Import local modules ---
try:
    # If using MACDRSIStrategy, change the line above to:
    # from strategy import ExampleMACDRSIStrategy # REMOVE THE CIRCULAR IMPORT FROM strategy.py if using this
    # --------------------------------------------
    import bybit_helpers as bybit  # Needed for closing exchange in finally
    import config as cfg  # Load configuration

    # --- IMPORT YOUR CHOSEN STRATEGY CLASS HERE ---
    # Make sure this matches the strategy you intend to run
    from ehlers_volumetric_strategy import EhlersVolumetricStrategy
    from neon_logger import setup_logger  # Use the logger setup
except ImportError as e:
    # Use the already defined Fore, Style, Back for error message
    err_back = Back.RED if COLORAMA_AVAILABLE else ""
    err_fore = Fore.WHITE if COLORAMA_AVAILABLE else ""
    reset_all = Style.RESET_ALL if COLORAMA_AVAILABLE else ""
    print(
        f"{err_back}{err_fore}FATAL: Failed to import core modules (config, neon_logger, strategy, bybit_helpers): {e}{reset_all}",
        file=sys.stderr,
    )
    print(
        "Ensure all required .py files are in the correct location and have no syntax errors.",
        file=sys.stderr,
    )
    traceback.print_exc()  # Print detailed traceback for import errors
    sys.exit(1)
except Exception as e:  # Catch other potential errors during import phase
    err_back = Back.RED if COLORAMA_AVAILABLE else ""
    err_fore = Fore.WHITE if COLORAMA_AVAILABLE else ""
    reset_all = Style.RESET_ALL if COLORAMA_AVAILABLE else ""
    print(
        f"{err_back}{err_fore}FATAL: Unexpected error during initial imports: {e}{reset_all}",
        file=sys.stderr,
    )
    traceback.print_exc()
    sys.exit(1)


# --- Global Variables ---
logger: logging.Logger | None = None
strategy_instance: EhlersVolumetricStrategy | None = (
    None  # Change type hint if using different strategy
)
main_task: asyncio.Task | None = None
exchange_instance_ref: bybit.ccxt.bybit | None = (
    None  # Keep reference for final cleanup
)
shutdown_requested = False


# --- Signal Handling for Graceful Shutdown ---
async def handle_signal(sig: signal.Signals):
    """Asynchronously handles shutdown signals."""
    global shutdown_requested, logger, main_task, strategy_instance
    if shutdown_requested:
        print("\nShutdown already in progress...")
        if logger:
            logger.warning("Shutdown already requested, ignoring duplicate signal.")
        return
    shutdown_requested = True

    signal_name = signal.Signals(sig).name
    print(
        f"\n{Fore.YELLOW}{Style.BRIGHT}>>> Signal {signal_name} ({sig}) received. Initiating graceful shutdown... <<< {Style.RESET_ALL}"
    )
    if logger:
        logger.warning(
            f"Signal {signal_name} ({sig}) received. Initiating graceful shutdown..."
        )

    # 1. Request strategy loop to stop
    if strategy_instance and getattr(
        strategy_instance, "is_running", False
    ):  # Check if running
        if logger:
            logger.info("Requesting strategy loop stop...")
        # Run stop in a new task to avoid blocking signal handler
        asyncio.create_task(strategy_instance.stop())
        # Give it a moment to potentially finish the current cycle
        await asyncio.sleep(1)
    elif strategy_instance:
        if logger:
            logger.info("Strategy instance exists but is not running.")
    else:
        if logger:
            logger.info("Strategy instance not found.")

    # 2. Attempt to cancel the main task (run_loop)
    if main_task and not main_task.done():
        if logger:
            logger.info("Cancelling main strategy task...")
        main_task.cancel()
        try:
            # Give cancellation a chance to propagate
            await asyncio.wait_for(main_task, timeout=10.0)
            if logger:
                logger.info("Main task cancellation processed.")
        except asyncio.CancelledError:
            if logger:
                logger.info("Main task successfully cancelled.")
        except TimeoutError:
            if logger:
                logger.error(
                    "Timeout waiting for main task to cancel. Forcing exit might be needed."
                )
        except Exception as e:
            if logger:
                logger.error(f"Error during main task cancellation: {e}")

    # 3. Further cleanup (like exchange close) is handled in main()'s finally block.


async def main():
    """Main async function to setup and run the strategy."""
    global logger, strategy_instance, main_task, exchange_instance_ref

    # --- 1. Setup Logger (Must happen first) ---
    try:
        log_config = getattr(cfg, "LOGGING_CONFIG", {})
        log_console_level_str = log_config.get("CONSOLE_LEVEL_STR", "INFO").upper()
        log_file_level_str = log_config.get("FILE_LEVEL_STR", "DEBUG").upper()
        log_third_party_level_str = log_config.get(
            "THIRD_PARTY_LOG_LEVEL_STR", "WARNING"
        ).upper()

        log_console_level = logging.getLevelName(log_console_level_str)
        log_file_level = logging.getLevelName(log_file_level_str)
        log_third_party_level = logging.getLevelName(log_third_party_level_str)

        if not isinstance(log_console_level, int):
            print(
                f"Warning: Invalid CONSOLE_LEVEL '{log_console_level_str}'. Defaulting to INFO.",
                file=sys.stderr,
            )
            log_console_level = logging.INFO
        if not isinstance(log_file_level, int):
            print(
                f"Warning: Invalid FILE_LEVEL '{log_file_level_str}'. Defaulting to DEBUG.",
                file=sys.stderr,
            )
            log_file_level = logging.DEBUG
        if not isinstance(log_third_party_level, int):
            print(
                f"Warning: Invalid THIRD_PARTY_LOG_LEVEL '{log_third_party_level_str}'. Defaulting to WARNING.",
                file=sys.stderr,
            )
            log_third_party_level = logging.WARNING

        logger = setup_logger(
            logger_name=log_config.get("LOGGER_NAME", "TradingBot"),
            log_file=log_config.get("LOG_FILE", "trading_bot.log"),
            console_level=log_console_level,
            file_level=log_file_level,
            log_rotation_bytes=log_config.get("LOG_ROTATION_BYTES", 5 * 1024 * 1024),
            log_backup_count=log_config.get("LOG_BACKUP_COUNT", 5),
            third_party_log_level=log_third_party_level,
        )
        logger.info("=" * 60)
        logger.info(
            f"=== {log_config.get('LOGGER_NAME', 'Trading Bot')} Initializing ==="
        )
        api_conf = getattr(cfg, "API_CONFIG", {})
        strat_conf = getattr(cfg, "STRATEGY_CONFIG", {})
        logger.info(f"Testnet Mode: {api_conf.get('TESTNET_MODE', 'N/A')}")
        logger.info(f"Symbol: {api_conf.get('SYMBOL', 'N/A')}")
        # --- Read STRATEGY name from loaded config ---
        logger.info(
            f"Strategy: {strat_conf.get('name', 'N/A')}"
        )  # Ensure config.py has the correct name
        # ---------------------------------------------
        logger.info(f"Timeframe: {strat_conf.get('timeframe', 'N/A')}")
        logger.info("=" * 60)

    except Exception as e:
        print(
            f"{Back.RED}{Fore.WHITE}FATAL: Logger setup failed: {e}{Style.RESET_ALL}",
            file=sys.stderr,
        )
        traceback.print_exc()
        sys.exit(1)

    # --- 2. Load Full Configuration ---
    full_config = {
        "API_CONFIG": getattr(cfg, "API_CONFIG", {}),
        "STRATEGY_CONFIG": getattr(cfg, "STRATEGY_CONFIG", {}),
        "LOGGING_CONFIG": getattr(cfg, "LOGGING_CONFIG", {}),
        "SMS_CONFIG": getattr(cfg, "SMS_CONFIG", {}),
    }

    # --- 3. Instantiate Strategy ---
    try:
        # --- Ensure this matches the imported strategy class ---
        strategy_instance = EhlersVolumetricStrategy(config=full_config, logger=logger)
        # ------------------------------------------------------
    except ValueError as e:  # Catch the specific configuration error
        logger.critical(
            f"Strategy instantiation failed due to configuration error: {e}",
            exc_info=True,
        )
        # Error message already logged by strategy's __init__
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Failed to instantiate strategy: {e}", exc_info=True)
        sys.exit(1)

    # --- 4. Run the Strategy Loop ---
    run_success = False
    try:
        logger.info("Starting strategy execution loop...")
        # Assign the task so signal handler can cancel it
        main_task = asyncio.create_task(strategy_instance.run_loop())

        # Store exchange reference AFTER initialization inside strategy completes
        # run_loop calls _initialize which sets self.exchange
        # Need to wait for _initialize to potentially finish
        initialized = (
            await strategy_instance._initialize()
        )  # Run initialization explicitly
        if not initialized:
            logger.critical(
                "Strategy initialization method failed. Cannot start trading loop."
            )
            # Cleanup will be handled in finally block
            sys.exit(1)

        exchange_instance_ref = strategy_instance.exchange  # Store reference now

        # Wait for the main strategy loop task to complete
        await main_task

        if main_task.cancelled():
            logger.warning(
                "Strategy loop task was cancelled (expected during shutdown)."
            )
            run_success = True
        elif main_task.exception():
            loop_exception = main_task.exception()
            logger.critical(
                f"Strategy loop task exited with an exception: {loop_exception}",
                exc_info=loop_exception,
            )
            run_success = False
        else:
            logger.info("Strategy run_loop finished normally.")
            run_success = True

    except asyncio.CancelledError:
        logger.warning(
            "Main execution task was cancelled (likely during shutdown signal or init failure)."
        )
        run_success = True  # Treat cancellation as success for cleanup

    except Exception as e:
        logger.critical(
            f"Main execution block encountered an unhandled error: {e}", exc_info=True
        )
        run_success = False

    finally:
        logger.info(
            f"{Style.BRIGHT}--- Main Execution Block Finalizing (Run Success: {run_success}) ---{Style.RESET_ALL}"
        )

        # --- Final Cleanup ---
        if strategy_instance:
            logger.info("Running strategy internal cleanup (_cleanup)...")
            try:
                # Check if cleanup is async, await if needed
                cleanup_result = strategy_instance._cleanup()
                if asyncio.iscoroutine(cleanup_result):
                    await cleanup_result
            except Exception as strategy_cleanup_err:
                logger.error(
                    f"Error during strategy internal cleanup: {strategy_cleanup_err}",
                    exc_info=True,
                )

        # Final attempt to close the exchange connection
        if (
            exchange_instance_ref
            and hasattr(exchange_instance_ref, "close")
            and callable(exchange_instance_ref.close)
        ):
            logger.info("Attempting final exchange connection close...")
            try:
                if not exchange_instance_ref.closed:
                    close_result = exchange_instance_ref.close()
                    if asyncio.iscoroutine(close_result):  # close() is usually async
                        await close_result
                    logger.info(
                        f"{Fore.GREEN}Final exchange connection close successful.{Style.RESET_ALL}"
                    )
                else:
                    logger.info("Exchange connection was already closed.")
            except Exception as final_close_err:
                logger.error(
                    f"Error during final exchange connection close: {final_close_err}",
                    exc_info=True,
                )
        else:
            logger.warning(
                "Exchange instance reference not available for final cleanup or close method missing (may be normal if init failed)."
            )

        if not run_success:
            logger.error("Main execution block finished due to errors.")


# --- Script Entry Point ---
if __name__ == "__main__":
    start_time = time.monotonic()
    print(f"{Fore.CYAN}Starting Asynchronous Trading Bot...{Style.RESET_ALL}")
    event_loop: asyncio.AbstractEventLoop | None = None

    try:
        # Get event loop *before* running
        # Using get_event_loop is okay here before run starts
        event_loop = asyncio.get_event_loop()

        # Register signal handlers
        signals_to_handle = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
        for s in signals_to_handle:
            try:
                event_loop.add_signal_handler(
                    s, lambda s=s: asyncio.create_task(handle_signal(s))
                )
            except (NotImplementedError, RuntimeError) as e:  # Catch RuntimeError too
                print(
                    f"Warning: Signal {s.name} registration failed: {e}",
                    file=sys.stderr,
                )

        # Run the main asynchronous function
        event_loop.run_until_complete(main())

    except KeyboardInterrupt:
        print(
            f"\n{Fore.YELLOW}{Style.BRIGHT}KeyboardInterrupt caught at top level. Exiting immediately.{Style.RESET_ALL}"
        )

    except SystemExit as e:  # Catch sys.exit calls
        print(f"SystemExit called with code {e.code}. Exiting.")
        # Allow finally block to run if loop was running

    except Exception as top_level_exc:
        print(
            f"{Back.RED}{Fore.WHITE}FATAL UNHANDLED ERROR during asyncio execution: {top_level_exc}{Style.RESET_ALL}",
            file=sys.stderr,
        )
        traceback.print_exc()  # Print detailed traceback
        if logger:
            logger.critical(
                "Fatal unhandled error during asyncio execution", exc_info=True
            )
        sys.exit(1)

    finally:
        # --- Event Loop Cleanup ---
        if (
            event_loop and not event_loop.is_closed()
        ):  # Check if loop exists and isn't closed
            print("Cleaning up remaining asyncio tasks...")
            try:
                tasks = [
                    task
                    for task in asyncio.all_tasks(loop=event_loop)
                    if not task.done()
                ]
                if tasks:
                    print(f"Cancelling {len(tasks)} outstanding tasks...")
                    for task in tasks:
                        task.cancel()
                    # Wait briefly for tasks to cancel
                    event_loop.run_until_complete(
                        asyncio.gather(*tasks, return_exceptions=True)
                    )

                # Shutdown async generators
                if hasattr(event_loop, "shutdown_asyncgens"):
                    event_loop.run_until_complete(event_loop.shutdown_asyncgens())

            except RuntimeError as e:
                if "cannot schedule new futures after shutdown" in str(e):
                    print("Event loop already shut down during task cleanup.")
                else:
                    print(f"Error during asyncio task cleanup: {e}")
            except Exception as loop_cleanup_err:
                print(f"Error during asyncio task cleanup: {loop_cleanup_err}")
            finally:
                if not event_loop.is_closed():
                    print("Closing asyncio event loop.")
                    event_loop.close()

        end_time = time.monotonic()
        total_runtime = end_time - start_time
        print(
            f"{Fore.CYAN}--- Application Shutdown Complete (Total Runtime: {total_runtime:.2f}s) ---{Style.RESET_ALL}"
        )
        if logger:
            logger.info(
                f"--- Application Shutdown Complete (Runtime: {total_runtime:.2f}s) ---"
            )
