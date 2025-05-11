# File: shutdown_handler.py
import time
import traceback
from typing import Any

# Third-party Libraries
try:
    import ccxt # For type hinting Exchange object
    from colorama import Fore, Back, Style
except ImportError:
    class DummyCCXTExchange: pass
    ccxt = type('ccxt', (), {'Exchange': DummyCCXTExchange})() # type: ignore[call-arg]
    class DummyColor:
        def __getattr__(self, name: str) -> str:
            return ""
    Fore, Back, Style = DummyColor(), DummyColor(), DummyColor()

# Custom module imports
from logger_setup import logger
from config import CONFIG
from utils import send_sms_alert
from order_management import cancel_open_orders, get_current_position, close_position


def graceful_shutdown(exchange: ccxt.Exchange | None, symbol: str | None) -> None:
    """Attempts to close position and cancel orders before exiting."""
    logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}Shutdown requested. Initiating graceful exit...{Style.RESET_ALL}")
    market_base = symbol.split("/")[0] if symbol else "Bot"
    send_sms_alert(f"[{market_base}] Shutdown requested. Attempting cleanup...")
    if not exchange or not symbol:
        logger.warning(f"{Fore.YELLOW}Shutdown: Exchange/Symbol not available.{Style.RESET_ALL}")
        return

    try:
        cancel_open_orders(exchange, symbol, reason="Graceful Shutdown")
        time.sleep(1)

        position = get_current_position(exchange, symbol)
        if position["side"] != CONFIG.pos_none:
            logger.warning(
                f"{Fore.YELLOW}Shutdown: Active {position['side']} position found (Qty: {position['qty']:.8f}). Closing...{Style.RESET_ALL}"
            )
            if close_position(exchange, symbol, position, reason="Shutdown"):
                logger.info(f"{Fore.CYAN}Shutdown: Close order placed. Waiting {CONFIG.post_close_delay_seconds * 2}s...{Style.RESET_ALL}")
                time.sleep(CONFIG.post_close_delay_seconds * 2)
                final_pos = get_current_position(exchange, symbol)
                if final_pos["side"] == CONFIG.pos_none:
                    logger.success(f"{Fore.GREEN}{Style.BRIGHT}Shutdown: Position confirmed CLOSED.{Style.RESET_ALL}") # type: ignore[attr-defined]
                    send_sms_alert(f"[{market_base}] Position confirmed CLOSED on shutdown.")
                else:
                    logger.error(f"{Back.RED}{Fore.WHITE}Shutdown: FAILED TO CONFIRM closure. Final: {final_pos['side']} Qty={final_pos['qty']:.8f}{Style.RESET_ALL}")
                    send_sms_alert(f"[{market_base}] ERROR: Failed confirm closure! Final: {final_pos['side']} Qty={final_pos['qty']:.8f}. MANUAL CHECK!")
            else:
                logger.error(f"{Back.RED}{Fore.WHITE}Shutdown: Failed to place close order. MANUAL INTERVENTION NEEDED.{Style.RESET_ALL}")
                send_sms_alert(f"[{market_base}] ERROR: Failed PLACE close order on shutdown. MANUAL CHECK!")
        else:
            logger.info(f"{Fore.GREEN}Shutdown: No active position found.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] No active position found on shutdown.")
    except Exception as e:
        logger.error(f"{Fore.RED}Shutdown: Error during cleanup: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}] Error during shutdown sequence: {type(e).__name__}")
    logger.info(f"{Fore.YELLOW}{Style.BRIGHT}--- Scalping Bot Shutdown Complete ---{Style.RESET_ALL}")

# End of shutdown_handler.py
```

```python
