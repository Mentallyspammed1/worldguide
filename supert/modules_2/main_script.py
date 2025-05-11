# File: main_script.py
#!/usr/bin/env python

# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v2.0.1 (Syntax Fix)
# Conjures high-frequency trades on Bybit Futures with enhanced precision and adaptable strategies.

"""High-Frequency Trading Bot (Scalping) for Bybit USDT Futures (Modular Version)
Version: 2.0.1 (Unified: Selectable Strategies + Precision + Native SL/TSL + Syntax Fix).

Features:
- Multiple strategies selectable via config: "DUAL_SUPERTREND", "STOCHRSI_MOMENTUM", "EHLERS_FISHER", "EHLERS_MA_CROSS".
- Enhanced Precision: Uses Decimal for critical financial calculations.
- Exchange-native Trailing Stop Loss (TSL) placed immediately after entry.
- Exchange-native fixed Stop Loss placed immediately after entry.
- ATR for volatility measurement and initial Stop-Loss calculation.
- Optional Volume spike and Order Book pressure confirmation.
- Risk-based position sizing with margin checks.
- Termux SMS alerts for critical events and trade actions.
- Robust error handling and logging with Neon color support.
- Graceful shutdown on KeyboardInterrupt with position/order closing attempt.
- Stricter position detection logic (Bybit V5 API).

Disclaimer:
- **EXTREME RISK**: Educational purposes ONLY. High-risk. Use at own absolute risk.
- **EXCHANGE-NATIVE SL/TSL DEPENDENCE**: Relies on exchange-native orders. Subject to exchange performance, slippage, API reliability.
- Parameter Sensitivity: Requires significant tuning and testing.
- API Rate Limits: Monitor usage.
- Slippage: Market orders are prone to slippage.
- Test Thoroughly: **DO NOT RUN LIVE WITHOUT EXTENSIVE TESTNET/DEMO TESTING.**
- Termux Dependency: Requires Termux:API.
- API Changes: Code targets Bybit V5 via CCXT, updates may be needed.
"""

# Standard Library Imports
import logging # For LOGGING_LEVEL display
import os
import sys
import time
import traceback
from decimal import getcontext, Decimal # For type hint if needed, and precision setting
from typing import Any # For type hints

# Third-party Libraries (Initial check and setup)
try:
    import ccxt
    import pandas as pd
    import pandas_ta as ta
    from colorama import Back, Fore, Style
    from colorama import init as colorama_init
    from dotenv import load_dotenv
except ImportError as e:
    missing_pkg = getattr(e, 'name', 'Unknown')
    print(f"CRITICAL: Missing required Python package: '{missing_pkg}'.")
    print("Please install it, e.g., 'pip install {missing_pkg}' or check requirements.txt.")
    sys.exit(1)

# --- Initializations (must happen before custom module imports that use them) ---
colorama_init(autoreset=True)
load_dotenv()
getcontext().prec = 18 # Set Decimal precision globally

# --- Custom Module Imports ---
# Order can matter if modules instantiate objects that depend on others (e.g., CONFIG needs logger)
from logger_setup import logger, LOGGING_LEVEL # Logger first
from config import CONFIG                     # Then Config, which uses logger
from utils import send_sms_alert              # Utils might be needed early by other modules
from exchange_handler import initialize_exchange, set_leverage
from data_fetcher import get_market_data
from trading_cycle import trade_logic
from shutdown_handler import graceful_shutdown

# --- Main Execution ---
def main() -> None:
    """Main function to initialize, set up, and run the trading loop."""
    start_time = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    logger.info(
        f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Bybit Scalping Spell v2.0.1 Initializing ({start_time}) ---{Style.RESET_ALL}"
    )
    logger.info(f"{Fore.CYAN}--- Strategy Enchantment: {CONFIG.strategy_name} ---{Style.RESET_ALL}")
    logger.info(f"{Fore.GREEN}--- Warding Rune: Initial ATR + Exchange Trailing Stop ---{Style.RESET_ALL}")
    logger.warning(
        f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}--- !!! LIVE FUTURES SCALPING MODE - EXTREME RISK INVOLVED !!! ---{Style.RESET_ALL}"
    )

    exchange_instance: ccxt.Exchange | None = None
    current_symbol: str | None = None
    run_bot: bool = True
    cycle_count: int = 0

    try:
        exchange_instance = initialize_exchange()
        if not exchange_instance:
            logger.critical("Exchange initialization failed. Exiting.")
            return

        try:
            sym_input = input(
                f"{Fore.YELLOW}Enter symbol {Style.DIM}(Default [{CONFIG.symbol}]){Style.NORMAL}: {Style.RESET_ALL}"
            ).strip()
            symbol_to_use = sym_input or CONFIG.symbol
            market = exchange_instance.market(symbol_to_use)
            current_symbol = market["symbol"]
            if not market.get("contract"):
                raise ValueError("Not a contract/futures market")
            logger.info(f"{Fore.GREEN}Using Symbol: {current_symbol} (Type: {market.get('type')}){Style.RESET_ALL}")
            if not set_leverage(exchange_instance, current_symbol, CONFIG.leverage):
                raise RuntimeError("Leverage setup failed")
        except (ccxt.BadSymbol, KeyError, ValueError, RuntimeError) as e:
            logger.critical(f"Symbol/Leverage setup failed: {e}")
            send_sms_alert(f"[ScalpBot] CRITICAL: Symbol/Leverage setup FAILED ({e}). Exiting.")
            return
        except Exception as e: # Catch any other exception during this critical setup
            logger.critical(f"Unexpected error during symbol/leverage setup: {e}")
            logger.debug(traceback.format_exc())
            send_sms_alert("[ScalpBot] CRITICAL: Unexpected symbol/leverage setup error. Exiting.")
            return


        logger.info(f"{Fore.MAGENTA}--- Configuration Summary ---{Style.RESET_ALL}")
        logger.info(f"{Fore.WHITE}Symbol: {current_symbol}, Interval: {CONFIG.interval}, Leverage: {CONFIG.leverage}x")
        # ... (logging other config details as in original) ...
        if CONFIG.strategy_name == "DUAL_SUPERTREND":
            logger.info(f"  Params: ST={CONFIG.st_atr_length}/{CONFIG.st_multiplier}, ConfirmST={CONFIG.confirm_st_atr_length}/{CONFIG.confirm_st_multiplier}")
        # Add other strategy param logging here if desired
        logger.info(f"{Fore.GREEN}Risk: {CONFIG.risk_per_trade_percentage:.3%}/trade, MaxPosValue: {CONFIG.max_order_usdt_amount:.4f} USDT")
        logger.info(f"{Fore.GREEN}Initial SL: {CONFIG.atr_stop_loss_multiplier} * ATR({CONFIG.atr_calculation_period})")
        logger.info(f"{Fore.GREEN}Trailing SL: {CONFIG.trailing_stop_percentage:.2%}, Activation Offset: {CONFIG.trailing_stop_activation_offset_percent:.2%}")
        logger.info(f"{Fore.YELLOW}Vol Confirm: {CONFIG.require_volume_spike_for_entry} (MA={CONFIG.volume_ma_period}, Thr={CONFIG.volume_spike_threshold})")
        logger.info(f"{Fore.YELLOW}OB Confirm: {CONFIG.fetch_order_book_per_cycle} (Depth={CONFIG.order_book_depth}, L>={CONFIG.order_book_ratio_threshold_long}, S<={CONFIG.order_book_ratio_threshold_short})")
        logger.info(f"{Fore.WHITE}Sleep: {CONFIG.sleep_seconds}s, Margin Buffer: {CONFIG.required_margin_buffer:.1%}, SMS: {CONFIG.enable_sms_alerts}")
        logger.info(f"{Fore.CYAN}Logging Level: {logging.getLevelName(LOGGING_LEVEL)}") # Use imported LOGGING_LEVEL
        logger.info(f"{Fore.MAGENTA}{'-' * 30}{Style.RESET_ALL}")

        market_base = current_symbol.split("/")[0]
        send_sms_alert(f"[{market_base}] Bot configured ({CONFIG.strategy_name}). SL: ATR+TSL. Starting loop.")

        while run_bot:
            cycle_start_time = time.monotonic()
            cycle_count += 1
            logger.debug(f"{Fore.CYAN}--- Cycle {cycle_count} Start ---{Style.RESET_ALL}")
            try:
                # Assert current_symbol and exchange_instance are not None for type checker
                assert current_symbol is not None, "Symbol not set"
                assert exchange_instance is not None, "Exchange not initialized"

                data_limit = max(100, CONFIG.st_atr_length * 2, CONFIG.confirm_st_atr_length * 2,
                                 CONFIG.stochrsi_rsi_length + CONFIG.stochrsi_stoch_length + 5,
                                 CONFIG.momentum_length * 2, CONFIG.ehlers_fisher_length * 2,
                                 CONFIG.ehlers_fast_period * 2, CONFIG.ehlers_slow_period * 2,
                                 CONFIG.atr_calculation_period * 2, CONFIG.volume_ma_period * 2
                                 ) + CONFIG.api_fetch_limit_buffer
                
                df = get_market_data(exchange_instance, current_symbol, CONFIG.interval, limit=data_limit)

                if df is not None and not df.empty:
                    trade_logic(exchange_instance, current_symbol, df.copy())
                else:
                    logger.warning(f"{Fore.YELLOW}No valid market data for {current_symbol}. Skipping cycle.{Style.RESET_ALL}")

            except ccxt.RateLimitExceeded as e:
                logger.warning(f"{Back.YELLOW}{Fore.BLACK}Rate Limit Exceeded: {e}. Sleeping longer...{Style.RESET_ALL}")
                time.sleep(CONFIG.sleep_seconds * 5)
                if current_symbol: send_sms_alert(f"[{current_symbol.split('/')[0]}] WARNING: Rate limit hit!")
            except ccxt.NetworkError as e:
                logger.warning(f"{Fore.YELLOW}Network error: {e}. Retrying next cycle.{Style.RESET_ALL}")
                time.sleep(CONFIG.sleep_seconds)
            except ccxt.ExchangeNotAvailable as e:
                logger.error(f"{Back.RED}{Fore.WHITE}Exchange unavailable: {e}. Sleeping much longer...{Style.RESET_ALL}")
                time.sleep(CONFIG.sleep_seconds * 10)
                if current_symbol: send_sms_alert(f"[{current_symbol.split('/')[0]}] ERROR: Exchange unavailable!")
            except ccxt.AuthenticationError as e:
                logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Authentication Error: {e}. Stopping NOW.{Style.RESET_ALL}")
                run_bot = False
                if current_symbol: send_sms_alert(f"[{current_symbol.split('/')[0]}] CRITICAL: Authentication Error! Stopping NOW.")
            except ccxt.ExchangeError as e:
                logger.error(f"{Fore.RED}Unhandled Exchange error: {e}{Style.RESET_ALL}")
                logger.debug(traceback.format_exc())
                if current_symbol: send_sms_alert(f"[{current_symbol.split('/')[0]}] ERROR: Unhandled Exchange error: {type(e).__name__}")
                time.sleep(CONFIG.sleep_seconds)
            except Exception as e:
                logger.exception(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}!!! UNEXPECTED CRITICAL ERROR: {e} !!!{Style.RESET_ALL}")
                run_bot = False
                if current_symbol: send_sms_alert(f"[{current_symbol.split('/')[0]}] CRITICAL UNEXPECTED ERROR: {type(e).__name__}! Stopping NOW.")

            if run_bot:
                elapsed = time.monotonic() - cycle_start_time
                sleep_dur = max(0, CONFIG.sleep_seconds - elapsed)
                logger.debug(f"Cycle {cycle_count} time: {elapsed:.2f}s. Sleeping: {sleep_dur:.2f}s.")
                if sleep_dur > 0: time.sleep(sleep_dur)

    except KeyboardInterrupt:
        logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}KeyboardInterrupt received. Arcane energies withdrawing...{Style.RESET_ALL}")
        run_bot = False
    except Exception as e: # Catch errors during initial setup before loop
        logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Fatal error during bot setup: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[ScalpBot] CRITICAL: Fatal setup error: {type(e).__name__}! Bot did not start.")
        run_bot = False # Ensure shutdown is called if error happens before loop
    finally:
        graceful_shutdown(exchange_instance, current_symbol)
        market_base_final = current_symbol.split("/")[0] if current_symbol else "Bot"
        send_sms_alert(f"[{market_base_final}] Bot process terminated.")
        logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Scalping Spell Deactivated ---{Style.RESET_ALL}")


if __name__ == "__main__":
    # This check ensures that if CONFIG fails to load (e.g. due to missing required env var),
    # the script exits before trying to run main().
    # config.py raises ValueError if Config instantiation fails.
    # main_script.py imports config, so if it fails, the script will exit here.
    try:
        # Test if CONFIG was loaded. If config.py failed, this import would have failed.
        # This is more of a conceptual check; the import itself is the test.
        if 'CONFIG' not in globals() or CONFIG is None : # CONFIG should be loaded from config.py
             # This case should ideally be caught by config.py itself or the import failing
            print(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}CRITICAL: Configuration failed to load. Exiting.{Style.RESET_ALL}")
            sys.exit(1)
        main()
    except NameError: # If CONFIG somehow wasn't imported due to an issue not caught by its own module.
        print(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}CRITICAL: Essential configuration (CONFIG) not found. Exiting.{Style.RESET_ALL}")
        sys.exit(1)
    except Exception as e: # Catch any truly unexpected top-level error
        print(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}UNHANDLED TOP-LEVEL EXCEPTION: {e}{Style.RESET_ALL}")
        traceback.print_exc()
        sys.exit(1)
# End of main_script.py
```
