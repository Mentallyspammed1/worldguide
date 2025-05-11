# File: exchange_setup.py
# -*- coding: utf-8 -*-

"""
Handles Bybit Exchange Initialization and Setup
"""

import logging
import sys
from typing import Optional

try:
    import ccxt
except ImportError:
    print("Error: CCXT library not found. Please install it: pip install ccxt")
    sys.exit(1)
try:
    from colorama import Fore, Style, Back
except ImportError:
    print(
        "Warning: colorama library not found. Logs will not be colored. Install: pip install colorama"
    )

    class DummyColor:
        def __getattr__(self, name: str) -> str:
            return ""

    Fore = Style = Back = DummyColor()

from config import Config
from utils import retry_api_call, _get_v5_category, send_sms_alert

logger = logging.getLogger(__name__)


# Snippet 1 / Function 1: Initialize Bybit Exchange
@retry_api_call(max_retries=3, initial_delay=2.0)
def initialize_bybit(config: Config) -> Optional[ccxt.bybit]:
    """
    Initializes and validates the Bybit CCXT exchange instance using V5 API settings.

    Sets sandbox mode, default order type (swap), loads markets, performs an initial
    balance check, and attempts to set default margin mode based on config.

    Args:
        config: The configuration object containing API keys, testnet flag, symbol, etc.

    Returns:
        A configured and validated `ccxt.bybit` instance, or `None` if initialization fails.

    Raises:
        Catches and logs CCXT exceptions during initialization. Relies on the
        `retry_api_call` decorator for retries, raising the final exception on failure.
    """
    func_name = "initialize_bybit"
    logger.info(
        f"{Fore.BLUE}[{func_name}] Initializing Bybit (V5) exchange instance...{Style.RESET_ALL}"
    )
    try:
        exchange_class = getattr(ccxt, config.EXCHANGE_ID)
        exchange = exchange_class(
            {
                "apiKey": config.API_KEY,
                "secret": config.API_SECRET,
                "enableRateLimit": True,  # Enable CCXT's built-in rate limiter
                "options": {
                    "defaultType": "swap",  # Default to swap markets for V5 futures/perps
                    "adjustForTimeDifference": True,
                    "recvWindow": config.DEFAULT_RECV_WINDOW,
                    "brokerId": "PyrmethusV2Mod",  # Optional: Identify your bot via Broker ID
                },
            }
        )

        if config.TESTNET_MODE:
            logger.info(f"[{func_name}] Enabling Bybit Sandbox (Testnet) mode.")
            exchange.set_sandbox_mode(False)

        logger.debug(f"[{func_name}] Loading markets...")
        exchange.load_markets(reload=True)  # Force reload initially
        if not exchange.markets:
            raise ccxt.ExchangeError(f"[{func_name}] Failed to load markets.")
        logger.debug(
            f"[{func_name}] Markets loaded successfully ({len(exchange.markets)} symbols)."
        )

        # Perform an initial API call to validate credentials and connectivity
        logger.debug(
            f"[{func_name}] Performing initial balance fetch for validation..."
        )
        # Use V5 UNIFIED account type for the initial check
        exchange.fetch_balance({"accountType": "UNIFIED"})
        logger.debug(f"[{func_name}] Initial balance check successful.")

        # Attempt to set default margin mode (Best effort, depends on account type)
        try:
            market = exchange.market(config.SYMBOL)
            category = _get_v5_category(market)
            if category and category in ["linear", "inverse"]:
                logger.debug(
                    f"[{func_name}] Attempting to set initial margin mode '{config.DEFAULT_MARGIN_MODE}' for {config.SYMBOL} (Category: {category})..."
                )
                params = {"category": category}
                # Use set_margin_mode for cross/isolated (might fail on UTA isolated, which is ok)
                exchange.set_margin_mode(
                    config.DEFAULT_MARGIN_MODE, config.SYMBOL, params=params
                )
                logger.info(
                    f"[{func_name}] Initial margin mode potentially set to '{config.DEFAULT_MARGIN_MODE}' for {config.SYMBOL}."
                )
            else:
                logger.warning(
                    f"[{func_name}] Cannot determine contract category for {config.SYMBOL} ({category}). Skipping initial margin mode set."
                )
        except (
            ccxt.NotSupported,
            ccxt.ExchangeError,
            ccxt.ArgumentsRequired,
            ccxt.BadSymbol,
        ) as e_margin:
            logger.warning(
                f"{Fore.YELLOW}[{func_name}] Could not set initial margin mode for {config.SYMBOL}: {e_margin}. "
                f"This might be expected (e.g., UTA Isolated accounts require per-symbol setup). Verify account settings.{Style.RESET_ALL}"
            )

        # Use logger.info for success, not logger.success if not defined everywhere
        logger.info(
            f"{Fore.GREEN}[{func_name}] Bybit exchange initialized successfully. Testnet: {config.TESTNET_MODE}.{Style.RESET_ALL}"
        )
        return exchange

    except (
        ccxt.AuthenticationError,
        ccxt.NetworkError,
        ccxt.ExchangeNotAvailable,
        ccxt.ExchangeError,
    ) as e:
        logger.error(
            f"{Fore.RED}[{func_name}] Initialization attempt failed: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        raise  # Re-raise for the decorator to handle retries/failure

    except Exception as e:
        logger.critical(
            f"{Back.RED}[{func_name}] Unexpected critical error during Bybit initialization: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        send_sms_alert(
            f"[BybitHelper] CRITICAL: Bybit init failed! Unexpected: {type(e).__name__}",
            config,
        )
        return None  # Return None on critical unexpected failure


# --- END OF FILE exchange_setup.py ---

# ---------------------------------------------------------------------------
