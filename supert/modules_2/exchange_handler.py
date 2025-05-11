# File: exchange_handler.py
import time
import traceback
from typing import Any

# Third-party Libraries
try:
    import ccxt
    from colorama import Fore, Style, Back # Added Back for critical messages
except ImportError:
    class DummyCCXTExchange: pass
    class DummyCCXT:
        Exchange = DummyCCXTExchange
        AuthenticationError = Exception
        NetworkError = Exception
        ExchangeError = Exception
    ccxt = DummyCCXT() # type: ignore[assignment]
    class DummyColor:
        def __getattr__(self, name: str) -> str:
            return ""
    Fore, Style, Back = DummyColor(), DummyColor(), DummyColor()


# Custom module imports
from logger_setup import logger
from config import CONFIG
from utils import send_sms_alert


def initialize_exchange() -> ccxt.Exchange | None:
    """Initializes and returns the CCXT Bybit exchange instance."""
    logger.info(f"{Fore.BLUE}Initializing CCXT Bybit connection...{Style.RESET_ALL}")
    if not CONFIG.api_key or not CONFIG.api_secret:
        logger.critical("API keys missing in .env file or config.")
        send_sms_alert("[ScalpBot] CRITICAL: API keys missing. Bot stopped.")
        return None
    try:
        exchange = ccxt.bybit(
            {
                "apiKey": CONFIG.api_key,
                "secret": CONFIG.api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "linear",
                    "recvWindow": CONFIG.default_recv_window,
                    "adjustForTimeDifference": True,
                },
            }
        )
        logger.debug("Loading markets...")
        exchange.load_markets(True)
        logger.debug("Fetching initial balance...")
        exchange.fetch_balance()
        logger.success( # type: ignore[attr-defined]
            f"{Fore.GREEN}{Style.BRIGHT}CCXT Bybit Session Initialized (LIVE SCALPING MODE - EXTREME CAUTION!).{Style.RESET_ALL}"
        )
        send_sms_alert("[ScalpBot] Initialized & authenticated successfully.")
        return exchange
    except ccxt.AuthenticationError as e:
        logger.critical(f"Authentication failed: {e}. Check keys/IP/permissions.")
        send_sms_alert(f"[ScalpBot] CRITICAL: Authentication FAILED: {e}. Bot stopped.")
    except ccxt.NetworkError as e:
        logger.critical(f"Network error on init: {e}. Check connection/Bybit status.")
        send_sms_alert(f"[ScalpBot] CRITICAL: Network Error on Init: {e}. Bot stopped.")
    except ccxt.ExchangeError as e:
        logger.critical(f"Exchange error on init: {e}. Check Bybit status/API docs.")
        send_sms_alert(f"[ScalpBot] CRITICAL: Exchange Error on Init: {e}. Bot stopped.")
    except Exception as e:
        logger.critical(f"Unexpected error during init: {e}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[ScalpBot] CRITICAL: Unexpected Init Error: {type(e).__name__}. Bot stopped.")
    return None


def set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int) -> bool:
    """Sets leverage for a futures symbol (Bybit V5 focus)."""
    logger.info(f"{Fore.CYAN}Leverage Setting: Attempting to set {leverage}x for {symbol}...{Style.RESET_ALL}")
    try:
        market = exchange.market(symbol)
        if not market.get("contract"):
            logger.error(f"{Fore.RED}Leverage Setting: Cannot set for non-contract market: {symbol}.{Style.RESET_ALL}")
            return False
    except Exception as e:
        logger.error(f"{Fore.RED}Leverage Setting: Failed to get market info for '{symbol}': {e}{Style.RESET_ALL}")
        return False

    for attempt in range(CONFIG.retry_count):
        try:
            params = {"buyLeverage": str(leverage), "sellLeverage": str(leverage)}
            response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
            logger.success( # type: ignore[attr-defined]
                f"{Fore.GREEN}Leverage Setting: Set to {leverage}x for {symbol}. Response: {response}{Style.RESET_ALL}"
            )
            return True
        except ccxt.ExchangeError as e:
            err_str = str(e).lower()
            if "leverage not modified" in err_str or "leverage is same as requested" in err_str:
                logger.info(f"{Fore.CYAN}Leverage Setting: Already set to {leverage}x for {symbol}.{Style.RESET_ALL}")
                return True
            logger.warning(
                f"{Fore.YELLOW}Leverage Setting: Exchange error (Attempt {attempt + 1}/{CONFIG.retry_count}): {e}{Style.RESET_ALL}"
            )
            if attempt < CONFIG.retry_count - 1:
                time.sleep(CONFIG.retry_delay_seconds)
            else:
                logger.error(
                    f"{Fore.RED}Leverage Setting: Failed after {CONFIG.retry_count} attempts.{Style.RESET_ALL}"
                )
        except (ccxt.NetworkError, Exception) as e:
            logger.warning(
                f"{Fore.YELLOW}Leverage Setting: Network/Other error (Attempt {attempt + 1}/{CONFIG.retry_count}): {e}{Style.RESET_ALL}"
            )
            if attempt < CONFIG.retry_count - 1:
                time.sleep(CONFIG.retry_delay_seconds)
            else:
                logger.error(
                    f"{Fore.RED}Leverage Setting: Failed after {CONFIG.retry_count} attempts.{Style.RESET_ALL}"
                )
    return False

# End of exchange_handler.py
```

```python
