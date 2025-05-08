# File: main.py
import sys
import logging
from decimal import getcontext, ROUND_HALF_UP # For initial context setting
from pathlib import Path # For CONFIG_FILE, STATE_FILE if used directly

# Assuming other modules are in the same directory
from app_config import CONFIG_FILE, STATE_FILE, CALCULATION_PRECISION # Import constants
from logger_utils import setup_logger # For initial logger setup
from bot_core import TradingBot # The main bot class

if __name__ == "__main__":
    # Set Decimal context globally at the very beginning
    # app_config.py also attempts this, but doing it here ensures it's set
    # before any other module that might use Decimal is imported.
    try:
        getcontext().prec = CALCULATION_PRECISION
        getcontext().rounding = ROUND_HALF_UP
        # Use basic print as logger might not be fully ready for this very first message
        print(f"Decimal context set in main.py: Precision={getcontext().prec}, Rounding={getcontext().rounding}")
    except Exception as e:
         print(f"CRITICAL: Failed to set Decimal context in main.py: {e}", file=sys.stderr)
         sys.exit(1)

    bot_instance = None
    # Initialize logger first (using default level from logger_utils)
    # Logger level will be updated from config inside TradingBot.__init__
    initial_logger = setup_logger() # This returns the "TradingBot" logger instance
    initial_logger.info("Logger initialized from main.py. Starting bot setup...")

    try:
        # Create bot instance (initializes components, loads config/state)
        bot_instance = TradingBot(config_path=CONFIG_FILE, state_path=STATE_FILE)

        # Start the main loop
        bot_instance.run()

    except SystemExit as e:
         # SystemExit is often a "clean" exit, e.g., from sys.exit()
         print(f"Bot exited with code {e.code}.", file=sys.stderr)
         # Logger might be available from bot_instance or initial_logger
         logger_to_use = bot_instance.logger if bot_instance and hasattr(bot_instance, 'logger') else initial_logger
         if logger_to_use.hasHandlers():
              logger_to_use.info(f"Bot process terminated with exit code {e.code}.")
         sys.exit(e.code) # Propagate exit code
    except KeyboardInterrupt:
         print("\nKeyboardInterrupt received in main.py. Exiting.", file=sys.stderr)
         logger_to_use = bot_instance.logger if bot_instance and hasattr(bot_instance, 'logger') else initial_logger
         if logger_to_use.hasHandlers():
              logger_to_use.info("KeyboardInterrupt received. Exiting.")
         sys.exit(0)
    except Exception as e:
         # Catch any unexpected critical errors during setup or run
         print(f"CRITICAL UNHANDLED ERROR in main.py: {e}", file=sys.stderr)
         logger_to_use = bot_instance.logger if bot_instance and hasattr(bot_instance, 'logger') else initial_logger
         if logger_to_use.hasHandlers():
              logger_to_use.critical(f"Unhandled exception caused bot termination: {e}", exc_info=True)
         else: # Logger not ready or failed, print traceback to stderr
              import traceback
              traceback.print_exc()
         sys.exit(1) # Exit with error code
```
