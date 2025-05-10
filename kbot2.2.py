# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Termux Trading Spell (v2.1.1 - Precision Enhanced & Robust)
# Conjures market insights and executes trades on Bybit Futures with refined precision and improved robustness.

import logging
import os
import sys
import time
from decimal import ROUND_DOWN, Decimal, DivisionByZero, InvalidOperation
from typing import Any

# Attempt to import necessary enchantments
try:
    import ccxt
    import numpy as np
    import pandas as pd
    from colorama import Back, Fore, Style, init
    from dotenv import load_dotenv
    from tabulate import tabulate
except ImportError as e:
    # Provide specific guidance for Termux users
    init(autoreset=True)  # Initialize colorama for error messages
    missing_pkg = e.name
    # Offer to install all common dependencies
    sys.exit(1)

# Weave the Colorama magic into the terminal
init(autoreset=True)

# Set Decimal precision (adjust if needed, higher precision means more memory/CPU)
# The default precision (usually 28) is often sufficient, but be aware of potential needs.
# getcontext().prec = 30 # Example: Increase precision if needed

# --- Arcane Configuration ---

# Summon secrets from the .env scroll
load_dotenv()

# Configure the Ethereal Log Scribe
# Define custom log levels for trade actions
TRADE_LEVEL_NUM = logging.INFO + 5  # Between INFO and WARNING
logging.addLevelName(TRADE_LEVEL_NUM, "TRADE")


def trade(self, message, *args, **kws) -> None:
    if self.isEnabledFor(TRADE_LEVEL_NUM):
        # pylint: disable=protected-access
        self._log(TRADE_LEVEL_NUM, message, args, **kws)


logging.Logger.trade = trade

# More detailed log format, includes module and line number for debugging
log_formatter = logging.Formatter(
    Fore.CYAN
    + "%(asctime)s "
    + Style.BRIGHT
    + "[%(levelname)s] "
    + Fore.WHITE
    + "(%(filename)s:%(lineno)d) "  # Added file/line info
    + Style.RESET_ALL
    + Fore.WHITE
    + "%(message)s"
)
logger = logging.getLogger(__name__)
# Set level via environment variable or default to INFO
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logger.setLevel(log_level)

stream_handler = logging.StreamHandler(sys.stdout)  # Explicitly use stdout
stream_handler.setFormatter(log_formatter)
if not logger.hasHandlers():
    logger.addHandler(stream_handler)
logger.propagate = False  # Prevent duplicate messages if root logger is configured


class TradingConfig:
    """Holds the sacred parameters of our spell, enhanced with precision awareness."""

    def __init__(self) -> None:
        self.symbol = self._get_env(
            "SYMBOL", "FARTCOIN/USDT:USDT", Fore.YELLOW
        )  # CCXT Unified Symbol
        self.market_type = self._get_env(
            "MARKET_TYPE", "linear", Fore.YELLOW
        ).lower()  # 'linear' or 'inverse'
        self.interval = self._get_env("INTERVAL", "1m", Fore.YELLOW)
        self.risk_percentage = self._get_env(
            "RISK_PERCENTAGE", "0.01", Fore.YELLOW, cast_type=Decimal
        )  # Use Decimal for risk % (e.g., 0.01 for 1%)
        self.sl_atr_multiplier = self._get_env(
            "SL_ATR_MULTIPLIER", "1.5", Fore.YELLOW, cast_type=Decimal
        )
        self.tsl_activation_atr_multiplier = self._get_env(
            "TSL_ACTIVATION_ATR_MULTIPLIER", "1.0", Fore.YELLOW, cast_type=Decimal
        )
        # Bybit uses percentage for TSL distance (e.g., 0.5 for 0.5%)
        self.trailing_stop_percent = self._get_env(
            "TRAILING_STOP_PERCENT", "0.5", Fore.YELLOW, cast_type=Decimal
        )  # Use Decimal (e.g. 0.5 for 0.5%)
        self.sl_trigger_by = self._get_env(
            "SL_TRIGGER_BY", "LastPrice", Fore.YELLOW
        )  # Options: LastPrice, MarkPrice, IndexPrice (check Bybit docs for validity)
        self.tsl_trigger_by = self._get_env(
            "TSL_TRIGGER_BY", "LastPrice", Fore.YELLOW
        )  # Usually same as SL, check Bybit docs

        # Epsilon: Small value for comparing quantities, dynamically determined if possible.
        # Default is very small, will be overridden after market info is loaded.
        self.position_qty_epsilon = Decimal("0.000000001")  # Default tiny Decimal

        self.api_key = self._get_env("BYBIT_API_KEY", None, Fore.RED)
        self.api_secret = self._get_env("BYBIT_API_SECRET", None, Fore.RED)
        self.ohlcv_limit = self._get_env(
            "OHLCV_LIMIT", "200", Fore.YELLOW, cast_type=int
        )  # Number of candles to fetch
        self.loop_sleep_seconds = self._get_env(
            "LOOP_SLEEP_SECONDS", "15", Fore.YELLOW, cast_type=int
        )  # Pause between cycles
        self.order_check_delay_seconds = self._get_env(
            "ORDER_CHECK_DELAY_SECONDS", "2", Fore.YELLOW, cast_type=int
        )  # Wait before checking order status
        self.order_check_timeout_seconds = self._get_env(
            "ORDER_CHECK_TIMEOUT_SECONDS", "12", Fore.YELLOW, cast_type=int
        )  # Max time for order status check loop
        self.max_fetch_retries = self._get_env(
            "MAX_FETCH_RETRIES", "3", Fore.YELLOW, cast_type=int
        )  # Retries for fetching data
        self.trade_only_with_trend = self._get_env(
            "TRADE_ONLY_WITH_TREND", "True", Fore.YELLOW, cast_type=bool
        )  # Only trade in direction of trend_ema
        self.trend_ema_period = self._get_env(
            "TREND_EMA_PERIOD", "20", Fore.YELLOW, cast_type=int
        )  # EMA period for trend filter

        if not self.api_key or not self.api_secret:
            logger.critical(
                Fore.RED
                + Style.BRIGHT
                + "BYBIT_API_KEY or BYBIT_API_SECRET not found in .env scroll! Halting."
            )
            sys.exit(1)

        # Validate market type
        if self.market_type not in ["linear", "inverse"]:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}Invalid MARKET_TYPE '{self.market_type}'. Must be 'linear' or 'inverse'. Halting."
            )
            sys.exit(1)

    def _get_env(
        self, key: str, default: Any, color: str, cast_type: type = str
    ) -> Any:
        value = os.getenv(key)
        if value is None or value == "":  # Treat empty string as not set
            value = default
            if default is not None:
                logger.warning(f"{color}Using default value for {key}: {value}")
        else:
            log_value = "****" if "SECRET" in key or "KEY" in key else value
            logger.info(f"{color}Summoned {key}: {log_value}")

        try:
            if value is None:
                return None  # Handle cases where default is None
            if cast_type == bool:
                return str(value).lower() in ["true", "1", "yes", "y", "on"]
            if cast_type == Decimal:
                try:
                    # Ensure we handle potential string default correctly
                    return Decimal(str(value))
                except InvalidOperation:
                    logger.error(
                        f"{Fore.RED}Invalid numeric value for {key} ('{value}'). Using default: {default}"
                    )
                    # Attempt to cast default, handling potential None default
                    if default is None:
                        return None
                    try:
                        return Decimal(str(default))
                    except InvalidOperation:
                        logger.critical(
                            f"{Fore.RED + Style.BRIGHT}Default value '{default}' for {key} also cannot be cast to Decimal. Halting."
                        )
                        sys.exit(1)
            # For int, float, str
            casted_value = cast_type(value)
            # Add basic validation for numeric types
            if cast_type in [int, float, Decimal] and casted_value < 0:
                if key not in [
                    "SL_ATR_MULTIPLIER",
                    "TSL_ACTIVATION_ATR_MULTIPLIER",
                    "TRAILING_STOP_PERCENT",
                ]:  # Allow negative multipliers? Generally no.
                    logger.warning(
                        f"{Fore.YELLOW}Configuration value for {key} ({casted_value}) is negative. Ensure this is intended."
                    )
            return casted_value
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.error(
                f"{Fore.RED}Could not cast {key} ('{value}') to {cast_type.__name__}: {e}. Using default: {default}"
            )
            # Attempt to cast default if value failed
            try:
                if default is None:
                    return None
                if cast_type == bool:
                    return str(default).lower() in ["true", "1", "yes", "y", "on"]
                if cast_type == Decimal:
                    return Decimal(str(default))
                return cast_type(default)
            except (ValueError, TypeError, InvalidOperation):
                logger.critical(
                    f"{Fore.RED + Style.BRIGHT}Default value '{default}' for {key} also cannot be cast to {cast_type.__name__}. Halting."
                )
                sys.exit(1)


CONFIG = TradingConfig()
MARKET_INFO: dict | None = None  # Global to store market details after connection
EXCHANGE: ccxt.Exchange | None = None  # Global for the exchange instance

# --- Exchange Nexus Initialization ---
try:
    exchange_options = {
        "apiKey": CONFIG.api_key,
        "secret": CONFIG.api_secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",  # Generic futures type
            "defaultSubType": CONFIG.market_type,  # 'linear' or 'inverse'
            "adjustForTimeDifference": True,
            # Bybit V5 API requires category for some unified endpoints if not default
            "brokerId": "PyrmethusV2",  # Custom identifier for Bybit API
            # Add category if needed, though defaultSubType should handle it for many calls
            # 'v5': {'category': CONFIG.market_type} # Example if specific category needed
        },
    }
    # Log options excluding secrets for debugging
    log_options = exchange_options.copy()
    log_options["apiKey"] = "****"
    log_options["secret"] = "****"
    logger.debug(f"Initializing CCXT Bybit with options: {log_options}")

    EXCHANGE = ccxt.bybit(exchange_options)

    # Test connectivity and credentials
    EXCHANGE.check_required_credentials()
    logger.info("Credentials format check passed.")
    # Fetch time to verify connectivity and clock sync
    server_time = EXCHANGE.fetch_time()
    logger.info(f"Exchange time synchronized: {EXCHANGE.iso8601(server_time)}")

    # Load markets (force reload)
    EXCHANGE.load_markets(True)
    logger.info(
        Fore.GREEN
        + Style.BRIGHT
        + f"Successfully connected to Bybit Nexus ({CONFIG.market_type.capitalize()} Markets)."
    )

    # Verify symbol exists and get market details
    if CONFIG.symbol not in EXCHANGE.markets:
        logger.error(
            Fore.RED
            + Style.BRIGHT
            + f"Symbol {CONFIG.symbol} not found in Bybit {CONFIG.market_type} market spirits."
        )
        # Suggest available symbols more effectively
        available_symbols = []
        quote_currency = CONFIG.symbol.split("/")[-1].split(":")[
            0
        ]  # e.g., USDT from BTC/USDT:USDT
        for s, m in EXCHANGE.markets.items():
            is_correct_type = (CONFIG.market_type == "linear" and m.get("linear")) or (
                CONFIG.market_type == "inverse" and m.get("inverse")
            )
            if m.get("active") and is_correct_type and m.get("quote") == quote_currency:
                available_symbols.append(s)

        suggestion_limit = 15
        suggestions = ", ".join(sorted(available_symbols)[:suggestion_limit])
        if len(available_symbols) > suggestion_limit:
            suggestions += "..."
        logger.info(
            Fore.CYAN
            + f"Available active {CONFIG.market_type} symbols with {quote_currency} quote (sample): "
            + suggestions
        )
        sys.exit(1)
    else:
        MARKET_INFO = EXCHANGE.market(CONFIG.symbol)
        logger.info(
            Fore.CYAN
            + f"Market spirit for {CONFIG.symbol} acknowledged (ID: {MARKET_INFO.get('id')})."
        )

        # Log key precision and limits using Decimal where appropriate
        price_prec_str = str(MARKET_INFO["precision"]["price"])
        amount_prec_str = str(MARKET_INFO["precision"]["amount"])
        min_amount_str = (
            str(MARKET_INFO["limits"]["amount"]["min"])
            if MARKET_INFO["limits"]["amount"].get("min") is not None
            else None
        )
        max_amount_str = (
            str(MARKET_INFO["limits"]["amount"]["max"])
            if MARKET_INFO["limits"]["amount"].get("max") is not None
            else None
        )
        contract_size_str = str(
            MARKET_INFO.get("contractSize", "1")
        )  # Default to '1' if not present
        min_cost_str = (
            str(MARKET_INFO["limits"].get("cost", {}).get("min"))
            if MARKET_INFO["limits"].get("cost", {}).get("min") is not None
            else None
        )

        # Dynamically set epsilon based on amount precision (step size)
        try:
            # amount_step = EXCHANGE.markets[CONFIG.symbol]['precision']['amount'] # This is precision (digits), not step size
            # Need to calculate step size: 1 / (10 ** amount_precision_digits) ? Or use market info if step is provided.
            # Let's try calculating from precision digits if 'step' isn't directly available
            amount_precision_digits = MARKET_INFO["precision"].get(
                "amount"
            )  # Number of decimal places
            if amount_precision_digits is not None:
                amount_step = Decimal("1") / (
                    Decimal("10") ** int(amount_precision_digits)
                )
                CONFIG.position_qty_epsilon = amount_step / Decimal(
                    "2"
                )  # Half the smallest step
                logger.info(
                    f"Dynamically set position_qty_epsilon based on amount precision ({amount_precision_digits} digits): {CONFIG.position_qty_epsilon}"
                )
            else:
                logger.warning(
                    f"Could not determine amount precision digits. Using default epsilon: {CONFIG.position_qty_epsilon}"
                )
        except Exception as e:
            logger.warning(
                f"Could not determine amount step size for dynamic epsilon: {e}. Using default: {CONFIG.position_qty_epsilon}"
            )

        logger.debug(
            f"Market Precision: Price={price_prec_str}, Amount={amount_prec_str}"
        )
        logger.debug(
            f"Market Limits: Min Amount={min_amount_str}, Max Amount={max_amount_str}, Min Cost={min_cost_str}"
        )
        logger.debug(f"Contract Size: {contract_size_str}")

        # Validate that we can convert these critical values to Decimal
        try:
            Decimal(
                price_prec_str
            )  # Price precision is often an exponent, not a direct value
            Decimal(amount_prec_str)  # Amount precision is often decimal places
            if min_amount_str is not None:
                Decimal(min_amount_str)
            # Max amount can be None
            # if max_amount_str is not None: Decimal(max_amount_str)
            Decimal(contract_size_str)
            if min_cost_str is not None:
                Decimal(min_cost_str)
        except (InvalidOperation, TypeError, Exception) as e:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}Failed to parse critical market info (precision/limits/size/cost) as numbers: {e}. Halting.",
                exc_info=True,
            )
            sys.exit(1)


except ccxt.AuthenticationError as e:
    logger.critical(
        Fore.RED
        + Style.BRIGHT
        + f"Authentication failed! Check API Key/Secret validity and permissions. Error: {e}"
    )
    sys.exit(1)
except ccxt.NetworkError as e:
    logger.critical(
        Fore.RED
        + Style.BRIGHT
        + f"Network error connecting to Bybit: {e}. Check internet connection and Bybit status."
    )
    sys.exit(1)
except ccxt.ExchangeNotAvailable as e:
    logger.critical(
        Fore.RED
        + Style.BRIGHT
        + f"Bybit exchange is currently unavailable: {e}. Check Bybit status."
    )
    sys.exit(1)
except ccxt.ExchangeError as e:
    logger.critical(
        Fore.RED + Style.BRIGHT + f"Exchange Nexus Error during initialization: {e}",
        exc_info=True,
    )
    sys.exit(1)
except Exception as e:
    logger.critical(
        Fore.RED + Style.BRIGHT + f"Unexpected error during Nexus initialization: {e}",
        exc_info=True,
    )
    sys.exit(1)


# --- Global State Runes ---
# Tracks active SL/TSL order IDs associated with a potential long or short position.
# Reset when a position is closed or a new entry order is successfully placed.
order_tracker: dict[str, dict[str, str | None]] = {
    "long": {"sl_id": None, "tsl_id": None},
    "short": {"sl_id": None, "tsl_id": None},
}


# --- Termux Utility Spell ---
def termux_notify(title: str, content: str) -> None:
    """Sends a notification using Termux API (if available)."""
    if not os.getenv("TERMUX_VERSION"):
        logger.debug("Not running in Termux environment. Skipping notification.")
        return
    try:
        # Check if command exists using a more reliable method
        if os.system("command -v termux-toast > /dev/null 2>&1") == 0:
            # Basic sanitization to prevent trivial command injection risks
            # Replace potential shell metacharacters with safe alternatives or remove them
            safe_title = (
                title.replace('"', "'")
                .replace("`", "'")
                .replace("$", "")
                .replace("\\", "")
                .replace(";", "")
                .replace("&", "")
                .replace("|", "")
            )
            safe_content = (
                content.replace('"', "'")
                .replace("`", "'")
                .replace("$", "")
                .replace("\\", "")
                .replace(";", "")
                .replace("&", "")
                .replace("|", "")
            )
            # Limit length to avoid issues
            max_len = 100
            full_message = f"{safe_title}: {safe_content}"[:max_len]
            # Use list format for subprocess.run for better security than os.system
            import subprocess

            cmd_list = [
                "termux-toast",
                "-g",
                "middle",
                "-c",
                "green",
                "-s",
                full_message,
            ]
            result = subprocess.run(
                cmd_list, capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                logger.warning(
                    f"termux-toast command failed with code {result.returncode}: {result.stderr}"
                )
        else:
            logger.debug("termux-toast command not found. Skipping notification.")
    except FileNotFoundError:
        logger.debug(
            "termux-toast command not found (FileNotFound). Skipping notification."
        )
    except Exception as e:
        logger.warning(
            Fore.YELLOW + f"Could not conjure Termux notification: {e}", exc_info=True
        )


# --- Precision Casting Spells ---


def format_price(symbol: str, price: float | Decimal | str) -> str:
    """Formats price according to market precision rules using exchange's method."""
    global MARKET_INFO, EXCHANGE
    if MARKET_INFO is None or EXCHANGE is None:
        logger.error(
            f"{Fore.RED}Market info or Exchange not loaded for {symbol}, cannot format price."
        )
        return str(
            Decimal(str(price)).quantize(Decimal("0.00000001"))
        )  # Fallback with reasonable precision
    try:
        # CCXT's price_to_precision handles rounding/truncation based on market rules.
        # It typically expects float input. Convert Decimal carefully.
        return EXCHANGE.price_to_precision(symbol, float(price))
    except Exception as e:
        logger.error(
            f"{Fore.RED}Error formatting price {price} for {symbol}: {e}. Using fallback."
        )
        try:
            # Fallback: Convert Decimal/string to string with reasonable precision
            return f"{Decimal(str(price)):.8f}"  # Adjust precision as needed
        except Exception:
            return str(float(price))  # Last resort fallback


def format_amount(
    symbol: str, amount: float | Decimal | str, rounding_mode=ROUND_DOWN
) -> str:
    """Formats amount according to market precision rules using exchange's method."""
    global MARKET_INFO, EXCHANGE
    if MARKET_INFO is None or EXCHANGE is None:
        logger.error(
            f"{Fore.RED}Market info or Exchange not loaded for {symbol}, cannot format amount."
        )
        return str(Decimal(str(amount)).quantize(Decimal("0.00000001")))  # Fallback
    try:
        # CCXT's amount_to_precision handles rounding/truncation (often ROUND_DOWN implicitly).
        # Crucially, ensure it respects step size. It typically expects float input.
        # Map Python Decimal rounding modes to CCXT rounding modes if needed, though default usually works.
        ccxt_rounding_mode = (
            ccxt.TRUNCATE if rounding_mode == ROUND_DOWN else ccxt.ROUND
        )  # Basic mapping
        return EXCHANGE.amount_to_precision(
            symbol, float(amount), rounding_mode=ccxt_rounding_mode
        )
    except Exception as e:
        logger.error(
            f"{Fore.RED}Error formatting amount {amount} for {symbol}: {e}. Using fallback."
        )
        try:
            # Fallback: Convert Decimal/string to string with reasonable precision
            return f"{Decimal(str(amount)):.8f}"  # Adjust precision as needed
        except Exception:
            return str(float(amount))  # Last resort fallback


# --- Core Spell Functions ---


def fetch_with_retries(fetch_function, *args, **kwargs) -> Any:
    """Generic wrapper to fetch data with retries and exponential backoff."""
    global EXCHANGE
    if EXCHANGE is None:
        logger.critical("Exchange object is None, cannot fetch data.")
        return None

    last_exception = None
    for attempt in range(
        CONFIG.max_fetch_retries + 1
    ):  # +1 to allow logging final failure
        try:
            result = fetch_function(*args, **kwargs)
            # Optional: Add basic validation for common failure patterns (e.g., empty list when data expected)
            # if isinstance(result, list) and not result and fetch_function.__name__ == 'fetch_ohlcv':
            #     logger.warning(f"{fetch_function.__name__} returned empty list, might indicate issue.")
            return result
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
            last_exception = e
            wait_time = 2**attempt  # Exponential backoff
            logger.warning(
                Fore.YELLOW
                + f"{fetch_function.__name__}: Network disturbance (Attempt {attempt + 1}/{CONFIG.max_fetch_retries}). Retrying in {wait_time}s... Error: {e}"
            )
            if attempt < CONFIG.max_fetch_retries:
                time.sleep(wait_time)
            else:
                logger.error(
                    Fore.RED
                    + f"{fetch_function.__name__}: Failed after {CONFIG.max_fetch_retries} attempts due to network issues."
                )
        except ccxt.ExchangeNotAvailable as e:
            last_exception = e
            logger.error(
                Fore.RED
                + f"{fetch_function.__name__}: Exchange not available: {e}. Stopping retries."
            )
            break  # No point retrying if exchange is down
        except ccxt.AuthenticationError as e:
            last_exception = e
            logger.critical(
                Fore.RED
                + Style.BRIGHT
                + f"{fetch_function.__name__}: Authentication error: {e}. Halting script."
            )
            # Optionally trigger immediate shutdown? For now, just stop retries and let main loop handle exit.
            # graceful_shutdown() # Consider if this is safe here
            sys.exit(1)  # Critical error, stop immediately
        except ccxt.ExchangeError as e:
            # Includes rate limit errors not automatically handled, potentially invalid requests etc.
            last_exception = e
            # Check for specific retryable errors if needed (e.g., specific Bybit error codes)
            # Bybit error code example: 10006 = timeout, 10016 = internal error (maybe retry)
            should_retry = True
            if hasattr(e, "args") and len(e.args) > 0 and isinstance(e.args[0], str):
                if "Rate limit exceeded" in e.args[0]:
                    wait_time = 5 * (attempt + 1)  # Longer wait for rate limits
                else:
                    wait_time = 2 * (
                        attempt + 1
                    )  # Simple backoff for other exchange errors

                # Check for non-retryable errors based on message content (example)
                if (
                    "Invalid symbol" in e.args[0]
                    or "Order quantity not specified" in e.args[0]
                ):  # Add more non-retryable patterns
                    logger.error(
                        Fore.RED
                        + f"{fetch_function.__name__}: Non-retryable exchange error: {e}. Stopping retries."
                    )
                    should_retry = False

            if should_retry and attempt < CONFIG.max_fetch_retries:
                logger.warning(
                    Fore.YELLOW
                    + f"{fetch_function.__name__}: Exchange error (Attempt {attempt + 1}/{CONFIG.max_fetch_retries}). Retrying in {wait_time}s... Error: {e}"
                )
                time.sleep(wait_time)
            elif should_retry:  # Final attempt failed
                logger.error(
                    Fore.RED
                    + f"{fetch_function.__name__}: Failed after {CONFIG.max_fetch_retries} attempts due to exchange errors."
                )
            else:  # Non-retryable error encountered
                break  # Exit retry loop

        except Exception as e:
            last_exception = e
            logger.error(
                Fore.RED
                + f"{fetch_function.__name__}: Unexpected shadow encountered: {e}",
                exc_info=True,
            )
            break  # Stop on unexpected errors

    # If loop finished without returning, it means all retries failed or a break occurred
    logger.error(
        f"{fetch_function.__name__} ultimately failed. Last exception: {last_exception}"
    )
    return None  # Indicate failure


def fetch_market_data(symbol: str, timeframe: str, limit: int) -> pd.DataFrame | None:
    """Fetch OHLCV data using the retry wrapper and perform validation."""
    global EXCHANGE
    logger.info(
        Fore.CYAN + f"# Channeling market whispers for {symbol} ({timeframe})..."
    )

    if EXCHANGE is None or not hasattr(EXCHANGE, "fetch_ohlcv"):
        logger.error(
            Fore.RED
            + "Exchange object not properly initialized or missing fetch_ohlcv."
        )
        return None

    # Ensure limit is positive
    if limit <= 0:
        logger.error(f"Invalid OHLCV limit requested: {limit}. Using default 100.")
        limit = 100

    ohlcv_data = fetch_with_retries(
        EXCHANGE.fetch_ohlcv, symbol, timeframe, limit=limit
    )

    if ohlcv_data is None:
        logger.error(
            Fore.RED + f"Failed to fetch OHLCV data for {symbol} after retries."
        )
        return None
    if not isinstance(ohlcv_data, list) or not ohlcv_data:
        logger.error(Fore.RED + f"Received empty or invalid OHLCV data: {ohlcv_data}")
        return None

    try:
        df = pd.DataFrame(
            ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        # Convert timestamp immediately
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        df.dropna(
            subset=["timestamp"], inplace=True
        )  # Drop rows where timestamp conversion failed

        # Convert numeric columns, coercing errors to NaN
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Check for NaNs in critical price columns *after* conversion
        initial_len = len(df)
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        if len(df) < initial_len:
            logger.warning(
                f"Dropped {initial_len - len(df)} rows with missing essential price data from OHLCV."
            )

        if df.empty:
            logger.error(
                Fore.RED
                + "DataFrame is empty after processing OHLCV data (all rows dropped?)."
            )
            return None

        df = df.set_index("timestamp")
        # Ensure data is sorted chronologically
        df.sort_index(inplace=True)

        # Check for duplicate timestamps (can indicate data issues)
        if df.index.duplicated().any():
            logger.warning(
                Fore.YELLOW
                + "Duplicate timestamps found in OHLCV data. Keeping last entry."
            )
            df = df[~df.index.duplicated(keep="last")]

        # Check time difference between first and last candle vs expected interval
        if len(df) > 1:
            time_diff = df.index[-1] - df.index[-2]
            expected_interval = pd.Timedelta(
                EXCHANGE.parse_timeframe(timeframe), unit="s"
            )
            # Allow some tolerance for minor timing differences
            if abs(time_diff - expected_interval) > expected_interval * 0.1:
                logger.warning(
                    f"Unexpected time gap between last two candles: {time_diff} (expected ~{expected_interval})"
                )

        logger.info(
            Fore.GREEN
            + f"Market whispers received ({len(df)} candles). Latest: {df.index[-1]}"
        )
        return df
    except Exception as e:
        logger.error(
            Fore.RED + f"Error processing OHLCV data into DataFrame: {e}", exc_info=True
        )
        return None


def calculate_indicators(df: pd.DataFrame) -> dict[str, Decimal] | None:
    """Calculate technical indicators, returning results as Decimals for precision."""
    logger.info(Fore.CYAN + "# Weaving indicator patterns...")
    if df is None or df.empty:
        logger.error(
            Fore.RED + "Cannot calculate indicators on missing or empty DataFrame."
        )
        return None
    try:
        # Ensure data is float for calculations, convert to Decimal at the end
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        # --- Indicator Periods ---
        fast_ema_period = 8
        slow_ema_period = 12
        confirm_ema_period = 5  # Currently unused in signals, but calculated
        trend_ema_period = CONFIG.trend_ema_period
        stoch_period = 10
        smooth_k = 3
        smooth_d = 3
        atr_period = 10

        # --- Check Data Length Requirements ---
        required_len_ema = max(
            fast_ema_period, slow_ema_period, trend_ema_period, confirm_ema_period
        )
        # Need period + smooth_k + smooth_d - 2 data points for smoothed Stoch
        required_len_stoch = stoch_period + smooth_k + smooth_d - 2
        # Need atr_period + 1 for shift(), and potentially more for EMA smoothing
        required_len_atr = atr_period + 1  # Minimum for basic TR calc
        min_required_len = max(required_len_ema, required_len_stoch, required_len_atr)

        if len(df) < min_required_len:
            logger.error(
                f"{Fore.RED}Not enough data ({len(df)}) for all indicators (minimum required: {min_required_len}). Cannot calculate reliable indicators."
            )
            return None
        elif len(df) < required_len_ema:
            logger.warning(
                f"Not enough data ({len(df)}) for full EMA periods (requires {required_len_ema}). Results may be inaccurate."
            )
        elif len(df) < required_len_stoch:
            logger.warning(
                f"Not enough data ({len(df)}) for full Stochastic period (requires {required_len_stoch}). Results may be inaccurate."
            )
        elif len(df) < required_len_atr:
            logger.warning(
                f"Not enough data ({len(df)}) for full ATR period (requires {required_len_atr}). Results may be inaccurate."
            )

        # --- Calculations ---
        fast_ema_series = close.ewm(span=fast_ema_period, adjust=False).mean()
        slow_ema_series = close.ewm(span=slow_ema_period, adjust=False).mean()
        trend_ema_series = close.ewm(span=trend_ema_period, adjust=False).mean()
        confirm_ema_series = close.ewm(span=confirm_ema_period, adjust=False).mean()

        low_min = low.rolling(window=stoch_period).min()
        high_max = high.rolling(window=stoch_period).max()
        stoch_k_raw = (
            100 * (close - low_min) / (high_max - low_min + 1e-12)
        )  # Add epsilon for stability
        stoch_k = stoch_k_raw.rolling(window=smooth_k).mean()
        stoch_d = stoch_k.rolling(window=smooth_d).mean()

        tr_df = pd.DataFrame(index=df.index)
        tr_df["hl"] = high - low
        tr_df["hc"] = (high - close.shift()).abs()
        tr_df["lc"] = (low - close.shift()).abs()
        tr_df["tr"] = tr_df[["hl", "hc", "lc"]].max(axis=1)
        tr_df.dropna(
            subset=["tr"], inplace=True
        )  # Drop rows where TR couldn't be calculated

        atr_series = pd.Series(index=df.index, dtype=float)  # Initialize with NaN
        if not tr_df.empty:
            # Use Exponential Moving Average (Wilder's smoothing) for ATR
            atr_series = tr_df["tr"].ewm(alpha=1 / atr_period, adjust=False).mean()

        # --- Extract Latest Values & Convert to Decimal ---
        # Define quantizers for consistent decimal places
        # Adjust precision based on typical price/indicator scales
        price_quantizer = Decimal(
            "0.00000001"
        )  # 8 decimal places for price-like values
        percent_quantizer = Decimal("0.01")  # 2 decimal places for Stoch

        # Helper to safely get latest non-NaN value, convert to Decimal, and handle errors
        def get_latest_decimal(
            series: pd.Series, quantizer: Decimal, default_val: Decimal = Decimal("NaN")
        ) -> Decimal:
            if series.empty:
                return default_val
            latest_val = series.iloc[-1]
            if pd.isna(latest_val):
                logger.warning(
                    f"Indicator calculation for {series.name if hasattr(series, 'name') else 'series'} resulted in NaN. Returning default."
                )
                return default_val
            try:
                # Convert via string for precision
                return Decimal(str(latest_val)).quantize(quantizer)
            except (InvalidOperation, TypeError) as e:
                logger.error(
                    f"Could not convert indicator value {latest_val} to Decimal: {e}. Returning default."
                )
                return default_val

        indicators_out = {
            "fast_ema": get_latest_decimal(fast_ema_series, price_quantizer),
            "slow_ema": get_latest_decimal(slow_ema_series, price_quantizer),
            "trend_ema": get_latest_decimal(trend_ema_series, price_quantizer),
            "confirm_ema": get_latest_decimal(confirm_ema_series, price_quantizer),
            "stoch_k": get_latest_decimal(
                stoch_k, percent_quantizer, default_val=Decimal("50.00")
            ),  # Default neutral
            "stoch_d": get_latest_decimal(
                stoch_d, percent_quantizer, default_val=Decimal("50.00")
            ),  # Default neutral
            "atr": get_latest_decimal(
                atr_series, price_quantizer, default_val=Decimal("0.0")
            ),  # Default zero
        }

        # Check if any crucial indicator calculation failed (returned NaN default)
        if any(
            val.is_nan()
            for key, val in indicators_out.items()
            if key in ["fast_ema", "slow_ema", "trend_ema", "stoch_k", "stoch_d", "atr"]
        ):
            logger.error(
                f"{Fore.RED}One or more critical indicators failed to calculate (NaN)."
            )
            return None  # Signal failure if essential indicators are NaN

        logger.info(Fore.GREEN + "Indicator patterns woven successfully.")
        return indicators_out

    except Exception as e:
        logger.error(
            Fore.RED + f"Failed to weave indicator patterns: {e}", exc_info=True
        )
        return None


def get_current_position(symbol: str) -> dict[str, dict[str, Any]] | None:
    """Fetch current positions using retry wrapper, returning quantities and prices as Decimals."""
    global EXCHANGE
    logger.info(Fore.CYAN + f"# Consulting position spirits for {symbol}...")

    if EXCHANGE is None:
        logger.error("Exchange object not available for fetching positions.")
        return None

    # Initialize with Decimal zero
    pos_dict = {
        "long": {
            "qty": Decimal("0.0"),
            "entry_price": Decimal("0.0"),
            "liq_price": Decimal("0.0"),
            "pnl": Decimal("0.0"),
        },
        "short": {
            "qty": Decimal("0.0"),
            "entry_price": Decimal("0.0"),
            "liq_price": Decimal("0.0"),
            "pnl": Decimal("0.0"),
        },
    }

    # Bybit V5 might require category parameter here too
    params = {"category": CONFIG.market_type} if CONFIG.market_type else {}
    positions_data = fetch_with_retries(
        EXCHANGE.fetch_positions, symbols=[symbol], params=params
    )

    if positions_data is None:
        logger.error(
            Fore.RED + f"Failed to fetch positions for {symbol} after retries."
        )
        return None  # Indicate failure to fetch

    if not isinstance(positions_data, list):
        logger.error(
            f"Unexpected data type received from fetch_positions: {type(positions_data)}. Expected list."
        )
        return None

    if not positions_data:
        logger.info(Fore.BLUE + f"No open positions reported by exchange for {symbol}.")
        return pos_dict

    # Filter positions for the exact symbol (CCXT fetch_positions with symbol *should* do this)
    # And handle potential differences in unified vs V5 API responses
    active_positions_found = 0
    for pos in positions_data:
        pos_symbol = pos.get("symbol")
        if pos_symbol != symbol:
            logger.debug(f"Ignoring position data for different symbol: {pos_symbol}")
            continue

        # Use info dictionary for safer access to raw exchange data if needed
        pos_info = pos.get("info", {})

        # Determine side ('long' or 'short') - check unified field first, then 'info'
        side = pos.get("side")  # Unified field
        if side is None:
            # Fallback for Bybit V5 'info' field if unified 'side' is missing
            side_raw = pos_info.get("side", "").lower()  # e.g., "Buy" or "Sell"
            if side_raw == "buy":
                side = "long"
            elif side_raw == "sell":
                side = "short"
            else:
                logger.warning(
                    f"Could not determine side for position: {pos_info}. Skipping."
                )
                continue

        # Get quantity ('contracts' or 'size') - check unified field first
        contracts_str = pos.get("contracts")  # Unified field
        if contracts_str is None:
            contracts_str = pos_info.get("size")  # Common Bybit V5 field

        # Get entry price - check unified field first
        entry_price_str = pos.get("entryPrice")
        if entry_price_str is None:
            entry_price_str = pos_info.get(
                "avgPrice", pos_info.get("entryPrice")
            )  # Check avgPrice too

        # Get Liq Price and PnL (these are less standardized in CCXT)
        liq_price_str = pos.get("liquidationPrice")
        if liq_price_str is None:
            liq_price_str = pos_info.get("liqPrice")
        # PnL might be 'unrealizedPnl' or similar
        pnl_str = pos.get("unrealizedPnl")
        if pnl_str is None:
            # Check Bybit specific info fields (names might vary)
            pnl_str = pos_info.get("unrealisedPnl", pos_info.get("unrealizedPnl"))

        if side in pos_dict and contracts_str is not None:
            try:
                contracts = Decimal(str(contracts_str))
                # Use epsilon to check if effectively zero
                if contracts.copy_abs() < CONFIG.position_qty_epsilon:
                    logger.debug(
                        f"Ignoring effectively zero size {side} position for {symbol} (Qty: {contracts})."
                    )
                    continue

                entry_price = (
                    Decimal(str(entry_price_str))
                    if entry_price_str is not None
                    else Decimal("0.0")
                )
                liq_price = (
                    Decimal(str(liq_price_str))
                    if liq_price_str is not None
                    else Decimal("0.0")
                )
                pnl = Decimal(str(pnl_str)) if pnl_str is not None else Decimal("0.0")

                # Assuming single position per side (no hedge mode complexity here)
                pos_dict[side]["qty"] = contracts
                pos_dict[side]["entry_price"] = entry_price
                pos_dict[side]["liq_price"] = liq_price
                pos_dict[side]["pnl"] = pnl

                logger.info(
                    Fore.YELLOW
                    + f"Found active {side.upper()} position: Qty={contracts}, Entry={entry_price:.4f}, Liq≈{liq_price:.4f}, PnL≈{pnl:.4f}"
                )
                active_positions_found += 1
            except (InvalidOperation, TypeError) as e:
                logger.error(
                    f"Could not parse position data for {side} side: Qty='{contracts_str}', Entry='{entry_price_str}', Liq='{liq_price_str}', PnL='{pnl_str}'. Error: {e}"
                )
                continue  # Skip this position entry
        elif side not in pos_dict:
            logger.warning(f"Position data found for unknown side '{side}'. Skipping.")

    if active_positions_found == 0:
        logger.info(
            Fore.BLUE
            + f"No active non-zero positions found for {symbol} after filtering."
        )

    logger.info(Fore.GREEN + "Position spirits consulted.")
    return pos_dict


def get_balance(currency: str = "USDT") -> tuple[Decimal | None, Decimal | None]:
    """Fetches the free and total balance for a specific currency using retry wrapper, returns Decimals."""
    global EXCHANGE
    logger.info(Fore.CYAN + f"# Querying the Vault of {currency}...")

    if EXCHANGE is None:
        logger.error("Exchange object not available for fetching balance.")
        return None, None

    # Bybit V5 fetch_balance might need accountType (UNIFIED or CONTRACT) or coin
    # CCXT should handle unification, but params might be needed if issues arise.
    # params = {'accountType': 'UNIFIED'} # Example for Unified Trading Account
    params = {}  # Start without specific params, rely on CCXT defaults/config
    balance_data = fetch_with_retries(EXCHANGE.fetch_balance, params=params)

    if balance_data is None:
        logger.error(
            Fore.RED
            + "Failed to fetch balance after retries. Cannot assess risk capital."
        )
        return None, None

    try:
        # Access nested dictionary safely, handle case variations
        free_balance = Decimal("0.0")
        total_balance = Decimal("0.0")

        # Check for currency directly in top level (less common now)
        if currency in balance_data:
            free_balance_str = balance_data.get(currency, {}).get("free")
            total_balance_str = balance_data.get(currency, {}).get("total")
        # Check standard 'free' and 'total' structures
        elif "free" in balance_data and "total" in balance_data:
            free_balance_str = balance_data.get("free", {}).get(currency)
            total_balance_str = balance_data.get("total", {}).get(currency)
        # Check 'info' structure as fallback (exchange-specific)
        elif "info" in balance_data and isinstance(balance_data["info"], dict):
            # Bybit V5 structure might be different, e.g., inside 'result' -> 'list'
            info_data = balance_data["info"]
            if (
                "result" in info_data
                and isinstance(info_data["result"], dict)
                and "list" in info_data["result"]
                and isinstance(info_data["result"]["list"], list)
            ):
                for account in info_data["result"]["list"]:
                    if account.get("coin") == currency:
                        # Fields might be 'walletBalance', 'availableToWithdraw', etc. Check Bybit docs.
                        # Using 'availableToWithdraw' for free, 'walletBalance' for total as a guess
                        free_balance_str = account.get("availableToWithdraw")
                        total_balance_str = account.get("walletBalance")
                        break  # Found the currency
            else:  # Try simpler info structure
                free_balance_str = balance_data["info"].get("free", {}).get(currency)
                total_balance_str = balance_data["info"].get("total", {}).get(currency)
        else:
            free_balance_str = None
            total_balance_str = None

        if free_balance_str is not None:
            free_balance = Decimal(str(free_balance_str))
        else:
            logger.warning(
                f"Could not find free balance for {currency} in balance data."
            )

        if total_balance_str is not None:
            total_balance = Decimal(str(total_balance_str))
        else:
            logger.warning(
                f"Could not find total balance for {currency} in balance data."
            )

        # Use 'total' balance as equity for risk calculation (more stable than free)
        equity = total_balance

        logger.info(
            Fore.GREEN
            + f"Vault contains {free_balance:.4f} free {currency} (Equity/Total: {equity:.4f})."
        )
        return free_balance, equity  # Return free and total (equity)

    except (InvalidOperation, TypeError) as e:
        logger.error(
            Fore.RED
            + f"Error parsing balance data for {currency}: {e}. Raw keys: {list(balance_data.keys()) if isinstance(balance_data, dict) else 'N/A'}"
        )
        logger.debug(f"Raw balance data: {balance_data}")
        return None, None
    except Exception as e:
        logger.error(
            Fore.RED + f"Unexpected shadow encountered querying vault: {e}",
            exc_info=True,
        )
        return None, None


def check_order_status(
    order_id: str, symbol: str, timeout: int = CONFIG.order_check_timeout_seconds
) -> dict | None:
    """Checks order status with retries and timeout. Returns the order dict or None if not found/timeout."""
    global EXCHANGE
    logger.info(Fore.CYAN + f"Verifying status of order {order_id} for {symbol}...")
    if EXCHANGE is None:
        logger.error("Exchange object not available for checking order status.")
        return None

    start_time = time.time()
    last_status = "unknown"
    attempt = 0
    while time.time() - start_time < timeout:
        attempt += 1
        logger.debug(f"Checking order {order_id}, attempt {attempt}...")
        try:
            # Use fetch_with_retries for the underlying fetch_order call
            # Add params if needed (e.g., Bybit V5 might need category)
            params = {"category": CONFIG.market_type} if CONFIG.market_type else {}
            order_status = fetch_with_retries(
                EXCHANGE.fetch_order, order_id, symbol, params=params
            )

            if order_status and isinstance(order_status, dict):
                last_status = order_status.get("status", "unknown")
                logger.info(f"Order {order_id} status check: {last_status}")
                # Return the full order dict if found
                # Check for terminal states (closed, canceled, rejected, expired)
                if last_status in ["closed", "canceled", "rejected", "expired"]:
                    logger.info(
                        f"Order {order_id} reached terminal state: {last_status}."
                    )
                    return order_status
                # If open or partially filled, continue loop until timeout or terminal state
                # (Unless immediate return is desired)
                # return order_status # Uncomment to return immediately even if not terminal

            else:
                # This case means fetch_with_retries failed or returned unexpected data
                logger.warning(
                    f"fetch_order call failed or returned invalid data for {order_id} after retries. Check logs."
                )
                # Continue the loop to retry check_order_status itself

        except ccxt.OrderNotFound:
            # Order is definitively not found on the exchange. It's gone.
            logger.error(
                Fore.RED + f"Order {order_id} confirmed NOT FOUND by exchange."
            )
            return None  # Explicitly indicate not found (terminal state)
        except Exception as e:
            # Catch any other unexpected error during the check itself
            logger.error(
                f"Unexpected error during order status check loop for {order_id}: {e}",
                exc_info=True,
            )
            # Decide whether to retry or fail; retrying is part of the loop.

        # Wait before the next check_order_status attempt
        check_interval = 1.5  # seconds, slightly longer interval
        time_elapsed = time.time() - start_time
        if time_elapsed + check_interval < timeout:
            logger.debug(
                f"Order {order_id} status ({last_status}) not terminal, sleeping {check_interval}s..."
            )
            time.sleep(check_interval)
        else:
            break  # Exit loop if next sleep would exceed timeout

    logger.error(
        Fore.RED
        + f"Timed out checking status for order {order_id} after {timeout} seconds. Last known status: {last_status}."
    )
    # Optionally fetch the order one last time outside the loop?
    # final_check = fetch_with_retries(EXCHANGE.fetch_order, order_id, symbol)
    # return final_check # Return the last known state even on timeout, or None? Let's return None on timeout.
    return None  # Indicate timeout or persistent failure to get a terminal status


def place_risked_market_order(
    symbol: str, side: str, risk_percentage: Decimal, atr: Decimal
) -> bool:
    """Places a market order with calculated size and initial ATR-based stop-loss, using Decimal precision."""
    logger.trade(
        Style.BRIGHT + f"Attempting {side.upper()} market entry for {symbol}..."
    )  # Use custom trade level

    global MARKET_INFO, EXCHANGE
    if MARKET_INFO is None or EXCHANGE is None:
        logger.error(
            Fore.RED + "Market info or Exchange not available. Cannot place order."
        )
        return False

    # --- Pre-computation & Validation ---
    # Use Total Equity for risk calculation, it's more stable than Free Balance
    _, total_equity = get_balance(
        MARKET_INFO.get("quote", "USDT")
    )  # Use quote currency from market info
    if total_equity is None or total_equity <= Decimal("0"):
        logger.error(Fore.RED + "Cannot place order: Invalid or zero account equity.")
        return False

    if atr is None or atr.is_nan() or atr <= Decimal("0"):
        logger.error(
            Fore.RED
            + f"Cannot place order: Invalid ATR value ({atr}). Check indicator calculation."
        )
        return False

    try:
        # Fetch current price using fetch_ticker with retries
        ticker_data = fetch_with_retries(EXCHANGE.fetch_ticker, symbol)
        if not ticker_data or ticker_data.get("last") is None:
            logger.error(
                Fore.RED
                + "Cannot fetch current ticker price for sizing/SL calculation."
            )
            return False
        # Use 'last' price as current price estimate
        price = Decimal(str(ticker_data["last"]))
        logger.debug(f"Current ticker price: {price:.4f}")

        # --- Calculate Stop Loss Price ---
        sl_distance_points = CONFIG.sl_atr_multiplier * atr
        if sl_distance_points <= Decimal(0):
            logger.error(
                f"{Fore.RED}Stop distance calculation resulted in zero or negative value ({sl_distance_points}). Check ATR ({atr}) and multiplier ({CONFIG.sl_atr_multiplier})."
            )
            return False

        if side == "buy":
            sl_price_raw = price - sl_distance_points
        else:  # side == "sell"
            sl_price_raw = price + sl_distance_points

        # Format SL price according to market precision *before* using it in calculations
        sl_price_formatted_str = format_price(symbol, sl_price_raw)
        sl_price = Decimal(sl_price_formatted_str)  # Use the formatted price as Decimal
        logger.debug(
            f"ATR: {atr:.6f}, SL Multiplier: {CONFIG.sl_atr_multiplier}, SL Distance Points: {sl_distance_points:.6f}"
        )
        logger.debug(
            f"Raw SL Price: {sl_price_raw:.6f}, Formatted SL Price for API: {sl_price_formatted_str}"
        )

        # Sanity check SL placement relative to current price (allow for small spread/slippage)
        # Use a small tolerance based on price precision
        price_step = (
            Decimal("1") / (Decimal("10") ** int(MARKET_INFO["precision"]["price"]))
            if MARKET_INFO["precision"]["price"]
            else Decimal("0.000001")
        )
        if side == "buy" and sl_price >= price - price_step:
            logger.error(
                Fore.RED
                + f"Calculated SL price ({sl_price}) is too close to or above current price ({price}). ATR ({atr}) might be too large or price feed issue? Aborting."
            )
            return False
        if side == "sell" and sl_price <= price + price_step:
            logger.error(
                Fore.RED
                + f"Calculated SL price ({sl_price}) is too close to or below current price ({price}). ATR ({atr}) might be too large or price feed issue? Aborting."
            )
            return False

        # --- Calculate Position Size ---
        risk_amount_quote = total_equity * risk_percentage
        # Stop distance in quote currency (absolute difference between entry and SL price)
        stop_distance_quote = abs(
            price - sl_price
        )  # Use current price as estimated entry

        if stop_distance_quote <= Decimal("0"):
            logger.error(
                Fore.RED
                + f"Stop distance in quote currency is zero or negative ({stop_distance_quote}). Check ATR, multiplier, or market precision. Cannot calculate size."
            )
            return False

        # Calculate quantity based on contract size and linear/inverse type
        Decimal(str(MARKET_INFO.get("contractSize", "1")))
        qty_raw = Decimal("0")

        try:
            if CONFIG.market_type == "linear":
                # Linear (e.g., BTC/USDT:USDT): Size is in Base currency (BTC). Value = Size * Price.
                # Risk Amount (Quote) = Qty (Base) * Stop Distance (Quote)
                # Qty (Base) = Risk Amount (Quote) / Stop Distance (Quote)
                qty_raw = risk_amount_quote / stop_distance_quote
                logger.debug(
                    f"Linear Sizing: Qty (Base) = {risk_amount_quote:.4f} / {stop_distance_quote:.4f} = {qty_raw}"
                )

            elif CONFIG.market_type == "inverse":
                # Inverse (e.g., BTC/USD:BTC): Size is in Contracts (often USD value). Value = Size (Contracts) * ContractValue / Price.
                # Risk Amount (Quote = USD) = Qty (Contracts) * ContractValue (USD/Contract) * abs(1/entry_price - 1/sl_price) * Current Price (USD/BTC)
                # Simpler Approximation (Verify against Bybit contract specs!):
                # Risk Amount (Quote) = Qty (Contracts) * Stop Distance (Quote) / Entry Price (Quote) <- Check this formula!
                # Qty (Contracts) = Risk Amount (Quote) * Entry Price (Quote) / Stop Distance (Quote)
                # Assumes contract value is 1 USD. If contract value is e.g. 100 USD, need to divide by 100.
                # Let's use the formula based on change in value per contract:
                # Risk (Quote) = Qty (Contracts) * Contract Size (Base/Contract) * Stop Distance (Quote)
                # Qty (Contracts) = Risk (Quote) / (Contract Size (Base/Contract) * Stop Distance (Quote))
                # THIS NEEDS CAREFUL VALIDATION FOR BYBIT INVERSE CONTRACTS.
                # The formula below assumes 1 contract = 1 USD, which is common but not universal.
                logger.warning(
                    Fore.YELLOW
                    + "Inverse contract sizing assumes 1 Contract = 1 USD value. VERIFY THIS against Bybit contract specs."
                )
                if price <= Decimal("0"):
                    logger.error(
                        Fore.RED
                        + "Cannot calculate inverse size with zero or negative price."
                    )
                    return False
                # Qty (Contracts) = Risk Amount (USD) * Entry Price (USD) / Stop Distance (USD)
                qty_raw = (risk_amount_quote * price) / stop_distance_quote
                logger.debug(
                    f"Inverse Sizing (Approx, assumes 1 Contract = 1 USD): Qty (Contracts) = ({risk_amount_quote:.4f} * {price:.4f}) / {stop_distance_quote:.4f} = {qty_raw}"
                )

            else:
                logger.error(
                    f"Unsupported market type for sizing: {CONFIG.market_type}"
                )
                return False
        except DivisionByZero:
            logger.error(
                Fore.RED
                + "Division by zero during quantity calculation. Check prices and stop distance."
            )
            return False

        # Format quantity according to market precision (ROUND_DOWN to be conservative)
        qty_formatted_str = format_amount(symbol, qty_raw, ROUND_DOWN)
        qty = Decimal(qty_formatted_str)
        logger.debug(
            f"Risk Amount: {risk_amount_quote:.4f} {MARKET_INFO.get('quote', 'Quote')}, Stop Distance: {stop_distance_quote:.4f} {MARKET_INFO.get('quote', 'Quote')}"
        )
        logger.debug(f"Raw Qty: {qty_raw:.8f}, Formatted Qty (Rounded Down): {qty}")

        # --- Validate Quantity Against Market Limits ---
        min_qty_str = (
            str(MARKET_INFO["limits"]["amount"]["min"])
            if MARKET_INFO["limits"]["amount"].get("min") is not None
            else "0"
        )
        max_qty_str = (
            str(MARKET_INFO["limits"]["amount"]["max"])
            if MARKET_INFO["limits"]["amount"].get("max") is not None
            else None
        )
        min_qty = Decimal(min_qty_str)
        max_qty = Decimal(max_qty_str) if max_qty_str is not None else None

        if qty < min_qty or qty.is_zero() or qty < CONFIG.position_qty_epsilon:
            logger.error(
                Fore.RED
                + f"Calculated quantity ({qty}) is zero or below minimum ({min_qty}). Risk amount ({risk_amount_quote:.4f}), stop distance ({stop_distance_quote:.4f}), or ATR might be too small. Cannot place order."
            )
            return False
        if max_qty is not None and qty > max_qty:
            logger.warning(
                Fore.YELLOW
                + f"Calculated quantity {qty} exceeds maximum {max_qty}. Capping order size to {max_qty}."
            )
            qty = max_qty  # Use the Decimal max_qty
            # Re-format capped amount - crucial! Use ROUND_DOWN again.
            qty_formatted_str = format_amount(symbol, qty, ROUND_DOWN)
            qty = Decimal(qty_formatted_str)
            logger.info(f"Re-formatted capped Qty: {qty}")
            if qty < min_qty:  # Double check after re-formatting capped value
                logger.error(
                    Fore.RED
                    + f"Capped quantity ({qty}) is now below minimum ({min_qty}). Aborting."
                )
                return False

        # Validate minimum cost if available
        min_cost_str = (
            str(MARKET_INFO["limits"].get("cost", {}).get("min"))
            if MARKET_INFO["limits"].get("cost", {}).get("min") is not None
            else None
        )
        if min_cost_str is not None:
            min_cost = Decimal(min_cost_str)
            # Estimate cost based on market type
            if CONFIG.market_type == "linear":
                estimated_cost = (
                    qty * price
                )  # Cost is Qty (Base) * Price (Quote/Base) = Quote
            elif CONFIG.market_type == "inverse":
                # Cost for inverse is trickier. If qty is in contracts (USD), cost might be related to margin needed.
                # Let's use a simple check: Qty (Contracts) * Contract Size (Base/Contract) * Price (Quote/Base) ? No...
                # Use qty * price as a rough proxy for value, compare with min_cost (needs check)
                estimated_cost = qty  # If qty is in contracts (USD), maybe cost is just qty? Check Bybit docs.
                logger.warning(
                    "Min cost check for inverse markets is approximate. Assumes cost relates to number of contracts."
                )
            else:
                estimated_cost = Decimal("0")

            if estimated_cost < min_cost:
                logger.error(
                    Fore.RED
                    + f"Estimated order cost/value ({estimated_cost:.4f}) is below minimum required ({min_cost:.4f}). Increase risk or adjust strategy. Cannot place order."
                )
                return False

        logger.info(
            Fore.YELLOW
            + f"Calculated Order: Side={side.upper()}, Qty={qty}, Entry≈{price:.4f}, SL={sl_price} (ATR={atr:.4f})"
        )

        # --- Cast the Market Order Spell ---
        logger.trade(f"Submitting {side.upper()} market order for {qty} {symbol}...")
        # Bybit V5 might require category in params
        order_params = {"category": CONFIG.market_type} if CONFIG.market_type else {}
        order = fetch_with_retries(
            EXCHANGE.create_market_order,
            symbol=symbol,
            side=side,
            amount=float(qty),  # CCXT expects float amount
            params=order_params,
        )

        # Handle potential failure from fetch_with_retries itself
        if order is None:
            logger.error(
                Fore.RED
                + "Market order placement failed after retries (fetch_with_retries returned None)."
            )
            return False

        logger.debug(f"Market order raw response: {order}")
        order_id = order.get("id")
        if not order_id:
            logger.error(Fore.RED + "Market order submission failed to return an ID.")
            # Check if order info contains error details
            if isinstance(order.get("info"), dict) and order["info"].get("retMsg"):
                logger.error(
                    f"Exchange message: {order['info']['retMsg']} (Code: {order['info'].get('retCode')})"
                )
            return False
        logger.trade(f"Market order submitted: ID {order_id}")

        # --- Verify Order Fill (Crucial Step) ---
        logger.info(
            f"Waiting {CONFIG.order_check_delay_seconds}s before checking fill status for order {order_id}..."
        )
        time.sleep(CONFIG.order_check_delay_seconds)  # Allow time for potential fill
        # Use the dedicated check_order_status function which includes retries and timeout
        order_status_data = check_order_status(
            order_id, symbol, timeout=CONFIG.order_check_timeout_seconds
        )

        filled_qty = Decimal("0.0")
        average_price = price  # Fallback to estimated entry price if check fails or price unavailable
        order_final_status = "unknown"

        if order_status_data and isinstance(order_status_data, dict):
            order_final_status = order_status_data.get("status", "unknown")
            filled_str = order_status_data.get("filled")
            average_str = order_status_data.get("average")  # Average fill price

            if filled_str is not None:
                try:
                    filled_qty = Decimal(str(filled_str))
                except InvalidOperation:
                    logger.error(
                        f"Could not parse filled quantity '{filled_str}' to Decimal."
                    )
            if average_str is not None:
                try:
                    avg_price_decimal = Decimal(str(average_str))
                    if avg_price_decimal > 0:  # Use actual fill price only if valid
                        average_price = avg_price_decimal
                except InvalidOperation:
                    logger.error(
                        f"Could not parse average price '{average_str}' to Decimal."
                    )

            logger.debug(
                f"Order {order_id} status check result: Status='{order_final_status}', Filled='{filled_str}', AvgPrice='{average_str}'"
            )

            # 'closed' means fully filled for market orders on most exchanges
            if (
                order_final_status == "closed"
                and filled_qty >= CONFIG.position_qty_epsilon
            ):
                logger.trade(
                    Fore.GREEN
                    + Style.BRIGHT
                    + f"Order {order_id} confirmed FILLED: {filled_qty} @ {average_price:.4f}"
                )
            # Handle partial fills for market orders (less common, but possible)
            elif (
                order_final_status == "open"
                and filled_qty > CONFIG.position_qty_epsilon
            ):
                logger.warning(
                    Fore.YELLOW
                    + f"Market Order {order_id} status is 'open' but partially filled ({filled_qty}). This is unusual. Proceeding with filled amount."
                )
                # Potentially wait longer or try to cancel remainder? For now, proceed.
            elif (
                order_final_status in ["open", "partially_filled"]
                and filled_qty < CONFIG.position_qty_epsilon
            ):
                logger.error(
                    Fore.RED
                    + f"Order {order_id} has status '{order_final_status}' but filled quantity is effectively zero ({filled_qty}). Aborting SL placement."
                )
                # Attempt to cancel just in case it's stuck (unlikely for market, but safe)
                try:
                    logger.info(
                        f"Attempting cancellation of stuck/unfilled order {order_id}."
                    )
                    fetch_with_retries(EXCHANGE.cancel_order, order_id, symbol)
                except Exception as cancel_err:
                    logger.warning(
                        f"Failed to cancel stuck order {order_id}: {cancel_err}"
                    )
                return False
            else:  # canceled, rejected, expired, failed, unknown, or closed with zero fill
                logger.error(
                    Fore.RED
                    + Style.BRIGHT
                    + f"Order {order_id} did not fill successfully: Status '{order_final_status}', Filled Qty: {filled_qty}. Aborting SL placement."
                )
                # Attempt to cancel if not already in a terminal state (defensive)
                if order_final_status not in ["canceled", "rejected", "expired"]:
                    try:
                        logger.info(
                            f"Attempting cancellation of failed/unknown status order {order_id}."
                        )
                        fetch_with_retries(EXCHANGE.cancel_order, order_id, symbol)
                    except Exception:
                        pass
                return False
        else:
            # check_order_status already logged error (e.g., timeout or not found)
            logger.error(
                Fore.RED
                + Style.BRIGHT
                + f"Could not determine final status for order {order_id}. Assuming failure. Aborting SL placement."
            )
            # Attempt to cancel just in case it's stuck somehow
            try:
                logger.info(
                    f"Attempting cancellation of unknown status order {order_id}."
                )
                fetch_with_retries(EXCHANGE.cancel_order, order_id, symbol)
            except Exception:
                pass
            return False

        # Final check on filled quantity after status check
        if filled_qty < CONFIG.position_qty_epsilon:
            logger.error(
                Fore.RED
                + f"Order {order_id} resulted in effectively zero filled quantity ({filled_qty}) after status check. No position opened."
            )
            return False

        # --- Place Initial Stop-Loss Order ---
        position_side = "long" if side == "buy" else "short"
        sl_order_side = "sell" if side == "buy" else "buy"

        # Format SL price and filled quantity correctly for the SL order
        # Use the SL price calculated earlier based on estimated entry, already formatted string
        sl_price_str_for_order = sl_price_formatted_str
        # Use the *actual filled quantity* for the SL order size, re-format it precisely (ROUND_DOWN just in case)
        sl_qty_str_for_order = format_amount(symbol, filled_qty, ROUND_DOWN)
        sl_qty_decimal = Decimal(sl_qty_str_for_order)

        # Ensure SL quantity is still valid (e.g., not below min after formatting)
        if sl_qty_decimal < min_qty:
            logger.error(
                f"{Fore.RED}Stop loss quantity ({sl_qty_decimal}) is below minimum ({min_qty}) after formatting filled amount. Cannot place SL. Position UNPROTECTED."
            )
            # Attempt emergency close
            logger.warning(
                Fore.YELLOW
                + "Attempting emergency market closure of unprotected position..."
            )
            try:
                emergency_close_order = fetch_with_retries(
                    EXCHANGE.create_market_order,
                    symbol,
                    sl_order_side,
                    float(sl_qty_decimal),
                    params={"reduceOnly": True},
                )
                logger.trade(
                    Fore.GREEN
                    + f"Emergency closure order placed: ID {emergency_close_order.get('id') if emergency_close_order else 'N/A'}"
                )
            except Exception as close_err:
                logger.critical(
                    Fore.RED
                    + Style.BRIGHT
                    + f"EMERGENCY CLOSURE FAILED: {close_err}. MANUAL INTERVENTION REQUIRED!"
                )
            return False  # Signal overall failure

        # Define parameters for the stop-loss order (Bybit V5 specific)
        sl_params = {
            "category": CONFIG.market_type,  # Required for V5 order placement
            "stopLoss": sl_price_str_for_order,  # Trigger price for the stop loss
            "slTriggerBy": CONFIG.sl_trigger_by,  # e.g., 'LastPrice', 'MarkPrice'
            "tpslMode": "Full",  # Apply SL to the entire position
            "reduceOnly": True,  # Ensure it only closes the position
            # 'positionIdx': 0 # For Bybit unified: 0 for one-way mode (usually default, confirm if needed)
        }
        logger.trade(
            f"Placing SL order: Side={sl_order_side}, Qty={sl_qty_str_for_order}, Trigger={sl_price_str_for_order}, TriggerBy={CONFIG.sl_trigger_by}"
        )
        logger.debug(f"SL Params (for create_order): {sl_params}")

        try:
            # CCXT create_order should map these params for Bybit V5 stop orders.
            # The type 'stop_market' is common, but Bybit might handle SL via params on a 'market' or 'limit' order type,
            # or require a dedicated endpoint. Let's assume CCXT handles 'stop_market' correctly with params.
            # We pass the trigger price in 'stopLoss' param, amount, and side.
            # The order type should indicate it's triggered.
            # Note: create_stop_market_order might be more explicit if available and reliable.
            # if EXCHANGE.has.get('createStopMarketOrder'):
            #     sl_order = fetch_with_retries(
            #         EXCHANGE.create_stop_market_order,
            #         symbol=symbol, side=sl_order_side, amount=float(sl_qty_str_for_order),
            #         price=sl_price_str_for_order, # Stop price
            #         params=sl_params
            #     )
            # else:
            # Try with create_order and type 'stop_market'
            sl_order = fetch_with_retries(
                EXCHANGE.create_order,
                symbol=symbol,
                type="market",  # Bybit V5 often uses 'Market' type with SL/TP params to set them *on the position*
                side=sl_order_side,
                amount=float(sl_qty_str_for_order),  # The amount to close
                price=None,  # Not needed for market SL
                params=sl_params,  # Pass SL parameters here
            )

            # Handle potential failure from fetch_with_retries
            if sl_order is None:
                raise ccxt.ExchangeError(
                    "Stop loss order placement failed after retries (fetch_with_retries returned None). Position might be UNPROTECTED."
                )

            sl_order_id = sl_order.get("id")
            logger.debug(f"SL order raw response: {sl_order}")

            # Check if Bybit V5 returned success code (0) even without a distinct SL order ID
            # This happens when SL is set directly on the position rather than as a separate order.
            is_position_sl_set = False
            if not sl_order_id and isinstance(sl_order.get("info"), dict):
                if sl_order["info"].get("retCode") == 0:
                    logger.trade(
                        Fore.GREEN
                        + Style.BRIGHT
                        + f"Stop Loss likely set directly on the {position_side.upper()} position (Trigger: {sl_price_str_for_order}). No separate SL order ID."
                    )
                    is_position_sl_set = True
                    # Use a placeholder to indicate SL is active on the position in the tracker
                    sl_order_id = f"POS_SL_{position_side.upper()}"
                else:
                    error_msg = sl_order["info"].get("retMsg", "Unknown reason.")
                    raise ccxt.ExchangeError(
                        f"Stop loss placement failed. Exchange message: {error_msg} (Code: {sl_order['info'].get('retCode')})"
                    )

            # --- Update Global State ---
            # CRITICAL: Clear any previous tracker state for this side before setting new IDs
            order_tracker[position_side] = {
                "sl_id": sl_order_id,
                "tsl_id": None,
            }  # Store actual ID or placeholder
            logger.trade(
                Fore.GREEN
                + Style.BRIGHT
                + f"Initial SL protection activated for {position_side.upper()} position. Trigger: {sl_price_str_for_order}"
                + (f" (ID: {sl_order_id})" if not is_position_sl_set else "")
            )

            # Use actual average fill price in notification
            entry_msg = (
                f"ENTERED {side.upper()} {filled_qty} {symbol.split('/')[0]} @ {average_price:.4f}. "
                f"Initial SL @ {sl_price_str_for_order}. TSL pending profit threshold."
            )
            logger.trade(Back.BLUE + Fore.WHITE + Style.BRIGHT + entry_msg)
            termux_notify(
                "Trade Entry",
                f"{side.upper()} {symbol} @ {average_price:.4f}, SL: {sl_price_str_for_order}",
            )
            return True

        except ccxt.InsufficientFunds as e:
            # This is critical - position opened but SL failed due to funds. Emergency close needed.
            logger.critical(
                Fore.RED
                + Style.BRIGHT
                + f"Insufficient funds to place stop-loss order for {filled_qty} {symbol} after entry: {e}. Position is UNPROTECTED."
            )
            logger.warning(
                Fore.YELLOW
                + "Attempting emergency market closure of unprotected position..."
            )
            try:
                close_qty_str = format_amount(symbol, filled_qty, ROUND_DOWN)
                emergency_close_order = fetch_with_retries(
                    EXCHANGE.create_market_order,
                    symbol,
                    sl_order_side,
                    float(close_qty_str),
                    params={"reduceOnly": True, "category": CONFIG.market_type},
                )
                if emergency_close_order:
                    logger.trade(
                        Fore.GREEN
                        + f"Emergency closure order placed: ID {emergency_close_order.get('id')}"
                    )
                    # Reset tracker state as position *should* be closed (assuming success)
                    order_tracker[position_side] = {"sl_id": None, "tsl_id": None}
                else:
                    logger.critical(
                        Fore.RED
                        + Style.BRIGHT
                        + "EMERGENCY CLOSURE FAILED (order placement failed after retries). MANUAL INTERVENTION REQUIRED!"
                    )

            except Exception as close_err:
                logger.critical(
                    Fore.RED
                    + Style.BRIGHT
                    + f"EMERGENCY CLOSURE FAILED: {close_err}. MANUAL INTERVENTION REQUIRED! Position is likely open and unprotected."
                )
                # Do NOT reset tracker state here, as we don't know the position status
            return False  # Signal overall failure of the entry attempt

        except (ccxt.ExchangeError, ccxt.InvalidOrder) as e:
            # SL placement failed for other reasons (e.g., invalid params, rate limit)
            logger.error(
                Fore.RED
                + Style.BRIGHT
                + f"Failed to place initial SL order: {e}. Position might be UNPROTECTED."
            )
            logger.warning(
                Fore.YELLOW
                + "Position may be open without Stop Loss due to SL placement error. Consider emergency closure or manual intervention."
            )
            # Log specific exchange message if available
            if (
                isinstance(getattr(e, "args", None), tuple)
                and len(e.args) > 0
                and isinstance(e.args[0], str)
            ):
                logger.error(f"Exchange message excerpt: {e.args[0][:500]}")
            # Do NOT reset tracker state, SL was not placed.
            return False  # Signal failure
        except Exception as e:
            logger.error(
                Fore.RED + Style.BRIGHT + f"Unexpected error placing SL: {e}",
                exc_info=True,
            )
            logger.warning(
                Fore.YELLOW
                + "Position may be open without Stop Loss due to unexpected SL placement error."
            )
            # Do NOT reset tracker state
            return False

    except ccxt.InsufficientFunds as e:
        logger.error(
            Fore.RED
            + Style.BRIGHT
            + f"Insufficient funds to place initial {side.upper()} market order for {qty} {symbol}: {e}"
        )
        return False
    except (ccxt.ExchangeError, ccxt.InvalidOrder) as e:
        # Error placing the initial market order itself
        logger.error(
            Fore.RED + Style.BRIGHT + f"Exchange error placing market order: {e}"
        )
        # Log specific exchange message if available
        if (
            isinstance(getattr(e, "args", None), tuple)
            and len(e.args) > 0
            and isinstance(e.args[0], str)
        ):
            logger.error(f"Exchange message excerpt: {e.args[0][:500]}")
        return False
    except Exception as e:
        logger.error(
            Fore.RED
            + Style.BRIGHT
            + f"Unexpected error during market order placement: {e}",
            exc_info=True,
        )
        return False


def manage_trailing_stop(
    symbol: str,
    position_side: str,  # 'long' or 'short'
    position_qty: Decimal,
    entry_price: Decimal,
    current_price: Decimal,
    atr: Decimal,
) -> None:
    """Manages the activation and placement of a trailing stop loss, using Decimal."""
    global order_tracker, EXCHANGE, MARKET_INFO  # We need to modify the global tracker and use exchange/market

    if EXCHANGE is None or MARKET_INFO is None:
        logger.error("Exchange or Market Info not available, cannot manage TSL.")
        return

    # --- Initial Checks ---
    if position_qty < CONFIG.position_qty_epsilon or entry_price <= Decimal("0"):
        if (
            order_tracker[position_side]["sl_id"]
            or order_tracker[position_side]["tsl_id"]
        ):
            logger.debug(
                f"Position {position_side} appears closed or invalid (Qty: {position_qty}, Entry: {entry_price}). Clearing stale order trackers."
            )
            order_tracker[position_side] = {"sl_id": None, "tsl_id": None}
        return  # No position to manage TSL for

    if atr is None or atr.is_nan() or atr <= Decimal("0"):
        logger.warning(
            Fore.YELLOW + "Cannot evaluate TSL activation: Invalid ATR value."
        )
        return

    # --- Get Current Tracker State ---
    initial_sl_id = order_tracker[position_side]["sl_id"]
    active_tsl_id = order_tracker[position_side]["tsl_id"]

    # If TSL is already active (has an ID or placeholder), assume exchange handles it.
    if active_tsl_id:
        # Check if it's a placeholder or real ID
        is_placeholder_tsl = active_tsl_id.startswith("POS_TSL_")
        log_msg = f"{position_side.upper()} TSL {'(Position Based)' if is_placeholder_tsl else '(Order ID: ' + active_tsl_id + ')'} is already active."
        logger.debug(log_msg)
        # Sanity check: Ensure initial SL ID is None if TSL is active
        if initial_sl_id and not initial_sl_id.startswith(
            "POS_SL_"
        ):  # Don't clear SL placeholder if TSL is also placeholder
            logger.warning(
                f"Inconsistent state: TSL active ({active_tsl_id}) but initial SL ID ({initial_sl_id}) is also present. Clearing initial SL ID."
            )
            order_tracker[position_side]["sl_id"] = None
        return

    # If TSL is not active, check if we *should* activate it.
    # Requires an initial SL to be present (we replace SL with TSL).
    if not initial_sl_id:
        logger.debug(
            f"Cannot activate TSL for {position_side.upper()}: Initial SL protection (Order ID or Position SL) is missing from tracker."
        )
        return

    # --- Check TSL Activation Condition ---
    profit = Decimal("0.0")
    if position_side == "long":
        profit = current_price - entry_price
    else:  # short
        profit = entry_price - current_price

    # Activation threshold in price points
    activation_threshold_points = CONFIG.tsl_activation_atr_multiplier * atr
    logger.debug(
        f"{position_side.upper()} Profit: {profit:.4f}, TSL Activation Threshold (Points): {activation_threshold_points:.4f} ({CONFIG.tsl_activation_atr_multiplier} * ATR)"
    )

    # Activate TSL only if profit exceeds the threshold
    if profit > activation_threshold_points:
        logger.trade(
            Fore.GREEN
            + Style.BRIGHT
            + f"Profit threshold reached for {position_side.upper()} position (Profit {profit:.4f} > Threshold {activation_threshold_points:.4f}). Activating TSL."
        )

        # --- Cancel Initial SL before placing TSL (CRITICAL STEP) ---
        # Check if the initial SL was a separate order or set on the position
        is_placeholder_sl = initial_sl_id.startswith("POS_SL_")

        if not is_placeholder_sl:
            logger.trade(
                f"Attempting to cancel initial SL order (ID: {initial_sl_id}) before placing TSL..."
            )
            try:
                # Use fetch_with_retries for cancellation robustness
                # Add category param for Bybit V5 cancel
                params = {"category": CONFIG.market_type} if CONFIG.market_type else {}
                cancel_response = fetch_with_retries(
                    EXCHANGE.cancel_order, initial_sl_id, symbol, params=params
                )
                # Check response - successful cancellation might not return much
                logger.trade(
                    Fore.GREEN
                    + f"Successfully sent cancel request for initial SL (ID: {initial_sl_id}). Response snippet: {str(cancel_response)[:100]}"
                )
                order_tracker[position_side]["sl_id"] = (
                    None  # Mark as cancelled locally *only after success*
                )

            except ccxt.OrderNotFound:
                logger.warning(
                    Fore.YELLOW
                    + f"Initial SL order (ID: {initial_sl_id}) already gone (not found) when trying to cancel. Proceeding with TSL placement."
                )
                order_tracker[position_side]["sl_id"] = None  # Assume it's gone
            except (ccxt.ExchangeError, ccxt.NetworkError) as e:
                logger.error(
                    Fore.RED
                    + Style.BRIGHT
                    + f"Failed to cancel initial SL order (ID: {initial_sl_id}): {e}. Aborting TSL placement to avoid potential duplicate stop orders."
                )
                return  # Stop the TSL process for this cycle
            except Exception as e:
                logger.error(
                    Fore.RED
                    + Style.BRIGHT
                    + f"Unexpected error cancelling initial SL order: {e}",
                    exc_info=True,
                )
                logger.error(
                    "Aborting TSL placement due to unexpected cancellation error."
                )
                return  # Stop the TSL process
        else:
            logger.trade(
                f"Initial SL ({initial_sl_id}) was set on position, no separate order to cancel. Proceeding to set TSL."
            )
            # We will overwrite the position's SL with a TSL using the setTradingStop endpoint/params.
            order_tracker[position_side]["sl_id"] = None  # Clear the placeholder

        # --- Place/Set Trailing Stop Loss ---
        # Bybit V5 typically sets TSL directly on the position using specific parameters.
        # We might use `exchange.private_post_position_set_trading_stop` or similar via CCXT's implicit methods,
        # or pass params via `edit_position` or `create_order` if CCXT maps them.

        # Use the current position quantity, formatted precisely
        tsl_qty_str = format_amount(
            symbol, position_qty, ROUND_DOWN
        )  # Format for logging/display, not usually needed for setting TSL on position

        # TSL distance as percentage string (e.g., "0.5" for 0.5%)
        trail_percent_str = str(
            CONFIG.trailing_stop_percent.quantize(Decimal("0.01"))
        )  # Ensure 2 decimal places for percentage

        # Bybit V5 Parameters for setting TSL on position:
        tsl_params = {
            "category": CONFIG.market_type,  # Required
            "symbol": MARKET_INFO["id"],  # Use exchange-specific market ID
            "trailingStop": trail_percent_str,  # Trailing distance percentage (as string)
            "tpslMode": "Full",  # Apply to the whole position
            "slTriggerBy": CONFIG.tsl_trigger_by,  # Trigger type for the trail
            # 'activePrice': format_price(symbol, current_price), # Optional: Price to activate the trail. If omitted, usually activates based on current price vs trail %.
            # 'positionIdx': 0 # Assuming one-way mode
        }
        logger.trade(
            f"Setting TSL on position: Symbol={symbol}, Qty={tsl_qty_str}, Trail%={trail_percent_str}, TriggerBy={CONFIG.tsl_trigger_by}"
        )
        logger.debug(f"TSL Params (for setTradingStop): {tsl_params}")

        try:
            # Attempt to use a hypothetical unified method first (less likely to exist)
            # response = EXCHANGE.edit_position(symbol, params=tsl_params) # Fictional unified method

            # More likely: Use implicit API call via ccxt for Bybit's specific endpoint
            # Endpoint: POST /v5/position/set-trading-stop
            # Check ccxt/python/ccxt/bybit.py for the implicit method name or define it.
            # It might be something like: exchange.private_post_position_set_trading_stop(tsl_params)
            # Let's assume the method exists (replace with actual if known or found in CCXT source)
            if hasattr(EXCHANGE, "private_post_position_set_trading_stop"):
                response = fetch_with_retries(
                    EXCHANGE.private_post_position_set_trading_stop, params=tsl_params
                )
            else:
                # Fallback: Try setting via create_order params (might work if CCXT maps it)
                logger.warning(
                    "Trying to set TSL via create_order params as set_trading_stop method not found/verified in CCXT."
                )
                tsl_order_side = "sell" if position_side == "long" else "buy"
                create_order_params = {
                    "category": CONFIG.market_type,
                    "trailingStop": trail_percent_str,
                    "tpslMode": "Full",
                    "slTriggerBy": CONFIG.tsl_trigger_by,
                    "reduceOnly": True,  # Important for safety
                }
                # We send a dummy amount (min qty?) as create_order needs it, but the params should modify the position TSL.
                dummy_qty_str = format_amount(
                    symbol,
                    Decimal(str(MARKET_INFO["limits"]["amount"]["min"])),
                    ROUND_DOWN,
                )
                response = fetch_with_retries(
                    EXCHANGE.create_order,
                    symbol=symbol,
                    type="market",
                    side=tsl_order_side,
                    amount=float(dummy_qty_str),
                    price=None,
                    params=create_order_params,
                )

            # --- Process TSL Set Response ---
            logger.debug(f"Set TSL raw response: {response}")

            # Handle potential failure from fetch_with_retries
            if response is None:
                raise ccxt.ExchangeError(
                    "Set TSL request failed after retries (fetch_with_retries returned None)."
                )

            # Check Bybit V5 response structure for success (retCode == 0)
            if (
                isinstance(response.get("info"), dict)
                and response["info"].get("retCode") == 0
            ):
                logger.trade(
                    Fore.GREEN
                    + Style.BRIGHT
                    + f"Trailing Stop Loss successfully set for {position_side.upper()} position. Trail: {trail_percent_str}%"
                )
                # Update tracker - set TSL active marker (use a placeholder ID)
                order_tracker[position_side]["tsl_id"] = (
                    f"POS_TSL_{position_side.upper()}"  # Mark TSL as active on position
                )
                order_tracker[position_side]["sl_id"] = (
                    None  # Ensure initial SL ID/placeholder is cleared
                )
                termux_notify(
                    "TSL Activated", f"{position_side.upper()} {symbol} TSL active."
                )
                return  # Success
            else:
                # Extract error message if possible
                error_msg = "Unknown reason."
                if isinstance(response.get("info"), dict):
                    error_msg = response["info"].get("retMsg", error_msg)
                    error_code = response["info"].get("retCode")
                    error_msg += f" (Code: {error_code})"
                raise ccxt.ExchangeError(
                    f"Failed to set trailing stop loss. Exchange message: {error_msg}"
                )

        except (ccxt.ExchangeError, ccxt.InvalidOrder) as e:
            # TSL setting failed. Initial SL was already cancelled/cleared. Position might be UNPROTECTED.
            logger.error(
                Fore.RED
                + Style.BRIGHT
                + f"CRITICAL: Failed to set TSL after clearing initial SL/placeholder: {e}"
            )
            logger.warning(
                Fore.YELLOW
                + Style.BRIGHT
                + "Position may be UNPROTECTED. MANUAL INTERVENTION STRONGLY ADVISED."
            )
            # Reset local tracker as TSL failed
            order_tracker[position_side]["tsl_id"] = None
            termux_notify("TSL FAILED!", f"{symbol} POS UNPROTECTED! Check bot.")
        except Exception as e:
            logger.error(
                Fore.RED + Style.BRIGHT + f"Unexpected error setting TSL: {e}",
                exc_info=True,
            )
            logger.warning(
                Fore.YELLOW
                + Style.BRIGHT
                + "Position might be UNPROTECTED after unexpected TSL setting error. MANUAL INTERVENTION ADVISED."
            )
            order_tracker[position_side]["tsl_id"] = None
            termux_notify("TSL FAILED!", f"{symbol} POS UNPROTECTED! Check bot.")

    else:
        # Profit threshold not met
        sl_status_log = (
            f"(ID: {initial_sl_id})"
            if not initial_sl_id.startswith("POS_")
            else f"({initial_sl_id})"
        )
        logger.debug(
            f"{position_side.upper()} profit ({profit:.4f}) has not crossed TSL activation threshold ({activation_threshold_points:.4f}). Keeping initial SL {sl_status_log}."
        )


def print_status_panel(
    cycle: int,
    timestamp: pd.Timestamp,
    price: Decimal | None,
    indicators: dict[str, Decimal] | None,
    positions: dict[str, dict[str, Any]] | None,
    equity: Decimal | None,
    signals: dict[str, bool],
    order_tracker_state: dict[
        str, dict[str, str | None]
    ],  # Pass tracker state explicitly
) -> None:
    """Displays the current state using a mystical status panel with Decimal precision."""
    Fore.MAGENTA + Style.BRIGHT

    timestamp.strftime("%Y-%m-%d %H:%M:%S %Z") if timestamp else f"{Fore.YELLOW}N/A"
    f"{equity:.4f} {MARKET_INFO.get('quote', 'Quote')}" if equity is not None and equity >= 0 else f"{Fore.YELLOW}N/A"

    # --- Market & Indicators ---
    price_str = f"{price:.4f}" if price is not None else f"{Fore.YELLOW}N/A"
    atr_str = (
        f"{indicators['atr']:.6f}"
        if indicators
        and indicators.get("atr") is not None
        and not indicators["atr"].is_nan()
        else f"{Fore.YELLOW}N/A"
    )
    trend_ema = indicators.get("trend_ema") if indicators else None
    trend_ema_str = (
        f"{trend_ema:.4f}"
        if trend_ema is not None and not trend_ema.is_nan()
        else f"{Fore.YELLOW}N/A"
    )

    price_color = Fore.WHITE
    trend_desc = f"{Fore.YELLOW}Trend N/A"
    if price is not None and trend_ema is not None and not trend_ema.is_nan():
        if price > trend_ema:
            price_color = Fore.GREEN
            trend_desc = f"{price_color}(Above Trend)"
        elif price < trend_ema:
            price_color = Fore.RED
            trend_desc = f"{price_color}(Below Trend)"
        else:
            price_color = Fore.YELLOW
            trend_desc = f"{price_color}(At Trend)"

    stoch_k = indicators.get("stoch_k") if indicators else None
    stoch_d = indicators.get("stoch_d") if indicators else None
    stoch_k_str = (
        f"{stoch_k:.2f}"
        if stoch_k is not None and not stoch_k.is_nan()
        else f"{Fore.YELLOW}N/A"
    )
    stoch_d_str = (
        f"{stoch_d:.2f}"
        if stoch_d is not None and not stoch_d.is_nan()
        else f"{Fore.YELLOW}N/A"
    )
    stoch_color = Fore.YELLOW
    stoch_desc = f"{Fore.YELLOW}Stoch N/A"
    if stoch_k is not None and not stoch_k.is_nan():
        if stoch_k < Decimal(25):
            stoch_color = Fore.GREEN
            stoch_desc = f"{stoch_color}Oversold (<25)"
        elif stoch_k > Decimal(75):
            stoch_color = Fore.RED
            stoch_desc = f"{stoch_color}Overbought (>75)"
        else:
            stoch_color = Fore.YELLOW
            stoch_desc = f"{stoch_color}Neutral"

    fast_ema = indicators.get("fast_ema") if indicators else None
    slow_ema = indicators.get("slow_ema") if indicators else None
    fast_ema_str = (
        f"{fast_ema:.4f}"
        if fast_ema is not None and not fast_ema.is_nan()
        else f"{Fore.YELLOW}N/A"
    )
    slow_ema_str = (
        f"{slow_ema:.4f}"
        if slow_ema is not None and not slow_ema.is_nan()
        else f"{Fore.YELLOW}N/A"
    )
    ema_cross_color = Fore.WHITE
    ema_desc = f"{Fore.YELLOW}EMA N/A"
    if (
        fast_ema is not None
        and not fast_ema.is_nan()
        and slow_ema is not None
        and not slow_ema.is_nan()
    ):
        if fast_ema > slow_ema:
            ema_cross_color = Fore.GREEN
            ema_desc = f"{ema_cross_color}Bullish Cross"
        elif fast_ema < slow_ema:
            ema_cross_color = Fore.RED
            ema_desc = f"{ema_cross_color}Bearish Cross"
        else:
            ema_cross_color = Fore.YELLOW
            ema_desc = f"{Fore.YELLOW}EMA Aligned"

    [
        [Fore.CYAN + "Market", Fore.WHITE + CONFIG.symbol, f"{price_color}{price_str}"],
        [
            Fore.CYAN + f"Trend EMA ({CONFIG.trend_ema_period})",
            f"{Fore.WHITE}{trend_ema_str}",
            trend_desc,
        ],
        [Fore.CYAN + "ATR (10)", f"{Fore.WHITE}{atr_str}", ""],
        [
            Fore.CYAN + "EMA Fast/Slow (8/12)",
            f"{ema_cross_color}{fast_ema_str} / {slow_ema_str}",
            ema_desc,
        ],
        [
            Fore.CYAN + "Stoch %K/%D (10,3,3)",
            f"{stoch_color}{stoch_k_str} / {stoch_d_str}",
            stoch_desc,
        ],
    ]

    # --- Positions & Orders ---
    pos_avail = positions is not None
    long_pos = positions.get("long", {}) if pos_avail else {}
    short_pos = positions.get("short", {}) if pos_avail else {}

    long_qty = long_pos.get("qty", Decimal("0.0"))
    short_qty = short_pos.get("qty", Decimal("0.0"))
    long_entry = long_pos.get("entry_price", Decimal("0.0"))
    short_entry = short_pos.get("entry_price", Decimal("0.0"))
    long_pnl = long_pos.get("pnl", Decimal("0.0"))
    short_pnl = short_pos.get("pnl", Decimal("0.0"))
    long_liq = long_pos.get("liq_price", Decimal("0.0"))
    short_liq = short_pos.get("liq_price", Decimal("0.0"))

    # Use the passed tracker state snapshot
    long_sl_id = order_tracker_state["long"]["sl_id"]
    long_tsl_id = order_tracker_state["long"]["tsl_id"]
    short_sl_id = order_tracker_state["short"]["sl_id"]
    short_tsl_id = order_tracker_state["short"]["tsl_id"]

    # Determine SL/TSL status strings
    def get_stop_status(sl_id, tsl_id) -> str:
        if tsl_id:
            if tsl_id.startswith("POS_TSL_"):
                return f"{Fore.GREEN}TSL Active (Pos)"
            else:
                return f"{Fore.GREEN}TSL Active (ID: ...{tsl_id[-6:]})"
        elif sl_id:
            if sl_id.startswith("POS_SL_"):
                return f"{Fore.YELLOW}SL Active (Pos)"
            else:
                return f"{Fore.YELLOW}SL Active (ID: ...{sl_id[-6:]})"
        else:
            return f"{Fore.RED}None"

    long_stop_status = (
        get_stop_status(long_sl_id, long_tsl_id)
        if long_qty >= CONFIG.position_qty_epsilon
        else f"{Fore.WHITE}-"
    )
    short_stop_status = (
        get_stop_status(short_sl_id, short_tsl_id)
        if short_qty >= CONFIG.position_qty_epsilon
        else f"{Fore.WHITE}-"
    )

    # Format position details, handle potential None from failed fetch
    if not pos_avail:
        long_qty_str, short_qty_str = f"{Fore.YELLOW}N/A", f"{Fore.YELLOW}N/A"
        long_entry_str, short_entry_str = f"{Fore.YELLOW}N/A", f"{Fore.YELLOW}N/A"
        long_pnl_str, short_pnl_str = f"{Fore.YELLOW}N/A", f"{Fore.YELLOW}N/A"
        long_liq_str, short_liq_str = f"{Fore.YELLOW}N/A", f"{Fore.YELLOW}N/A"
    else:
        long_qty_str = (
            f"{long_qty:.8f}".rstrip("0").rstrip(".") if long_qty != Decimal(0) else "0"
        )
        short_qty_str = (
            f"{short_qty:.8f}".rstrip("0").rstrip(".")
            if short_qty != Decimal(0)
            else "0"
        )
        long_entry_str = f"{long_entry:.4f}" if long_entry != Decimal(0) else "-"
        short_entry_str = f"{short_entry:.4f}" if short_entry != Decimal(0) else "-"
        long_pnl_color = Fore.GREEN if long_pnl >= 0 else Fore.RED
        short_pnl_color = Fore.GREEN if short_pnl >= 0 else Fore.RED
        long_pnl_str = (
            f"{long_pnl_color}{long_pnl:+.4f}{Fore.WHITE}"
            if long_qty >= CONFIG.position_qty_epsilon
            else "-"
        )
        short_pnl_str = (
            f"{short_pnl_color}{short_pnl:+.4f}{Fore.WHITE}"
            if short_qty >= CONFIG.position_qty_epsilon
            else "-"
        )
        long_liq_str = f"{Fore.RED}{long_liq:.4f}{Fore.WHITE}" if long_liq > 0 else "-"
        short_liq_str = (
            f"{Fore.RED}{short_liq:.4f}{Fore.WHITE}" if short_liq > 0 else "-"
        )

    [
        [Fore.CYAN + "Status", Fore.GREEN + "LONG", Fore.RED + "SHORT"],
        [
            Fore.CYAN + "Quantity",
            f"{Fore.WHITE}{long_qty_str}",
            f"{Fore.WHITE}{short_qty_str}",
        ],
        [
            Fore.CYAN + "Entry Price",
            f"{Fore.WHITE}{long_entry_str}",
            f"{Fore.WHITE}{short_entry_str}",
        ],
        [Fore.CYAN + "Unrealized PnL", long_pnl_str, short_pnl_str],
        [Fore.CYAN + "Liq. Price", long_liq_str, short_liq_str],
        [Fore.CYAN + "Active Stop", long_stop_status, short_stop_status],
    ]

    # --- Signals ---
    Fore.GREEN + Style.BRIGHT if signals.get("long", False) else Fore.WHITE
    Fore.RED + Style.BRIGHT if signals.get("short", False) else Fore.WHITE


def generate_signals(
    indicators: dict[str, Decimal] | None, current_price: Decimal | None
) -> dict[str, bool]:
    """Generates trading signals based on indicator conditions, using Decimal."""
    long_signal = False
    short_signal = False
    signal_reason = "No signal"

    if not indicators:
        logger.warning("Cannot generate signals: indicators are missing.")
        return {"long": False, "short": False, "reason": "Indicators missing"}
    if current_price is None or current_price <= Decimal(0):
        logger.warning("Cannot generate signals: current price is missing or invalid.")
        return {"long": False, "short": False, "reason": "Invalid price"}

    try:
        # Use .get with default Decimal('NaN') to detect missing/failed indicators
        k = indicators.get("stoch_k", Decimal("NaN"))
        indicators.get(
            "stoch_d", Decimal("NaN")
        )  # Not used in current logic, but available
        fast_ema = indicators.get("fast_ema", Decimal("NaN"))
        slow_ema = indicators.get("slow_ema", Decimal("NaN"))
        trend_ema = indicators.get("trend_ema", Decimal("NaN"))
        # confirm_ema = indicators.get('confirm_ema', Decimal('NaN')) # EMA(5)

        # Check if any required indicator is NaN
        required_indicators = [k, fast_ema, slow_ema, trend_ema]
        if any(ind.is_nan() for ind in required_indicators):
            logger.warning(
                "Cannot generate signals: One or more required indicators is NaN."
            )
            return {"long": False, "short": False, "reason": "NaN indicator"}

        # Define conditions using Decimal comparisons
        ema_bullish_cross = fast_ema > slow_ema
        ema_bearish_cross = fast_ema < slow_ema
        price_above_trend = current_price > trend_ema
        price_below_trend = current_price < trend_ema

        # Simpler Stochastic check (K level only)
        stoch_oversold = k < Decimal(25)
        stoch_overbought = k > Decimal(75)

        # Confirmation EMA (e.g., EMA 5) condition - currently unused
        # price_above_confirm = current_price > confirm_ema
        # price_below_confirm = current_price < confirm_ema

        # --- Signal Logic ---
        if ema_bullish_cross and stoch_oversold:
            if CONFIG.trade_only_with_trend:
                if price_above_trend:
                    long_signal = True
                    signal_reason = (
                        "Long: Bullish EMA Cross, Stoch Oversold, Price Above Trend EMA"
                    )
                else:
                    signal_reason = (
                        "Long Blocked: Price Below Trend EMA (Trend Filter ON)"
                    )
            else:  # Trend filter off
                long_signal = True
                signal_reason = (
                    "Long: Bullish EMA Cross, Stoch Oversold (Trend Filter OFF)"
                )

        elif ema_bearish_cross and stoch_overbought:
            if CONFIG.trade_only_with_trend:
                if price_below_trend:
                    short_signal = True
                    signal_reason = "Short: Bearish EMA Cross, Stoch Overbought, Price Below Trend EMA"
                else:
                    signal_reason = (
                        "Short Blocked: Price Above Trend EMA (Trend Filter ON)"
                    )
            else:  # Trend filter off
                short_signal = True
                signal_reason = (
                    "Short: Bearish EMA Cross, Stoch Overbought (Trend Filter OFF)"
                )
        else:
            # Log reasons for no signal if helpful for debugging
            reason_parts = []
            if not ema_bullish_cross and not ema_bearish_cross:
                reason_parts.append("No EMA cross")
            if ema_bullish_cross and not stoch_oversold:
                reason_parts.append("Stoch not Oversold")
            if ema_bearish_cross and not stoch_overbought:
                reason_parts.append("Stoch not Overbought")
            signal_reason = f"No signal ({', '.join(reason_parts)})"

        if long_signal or short_signal:
            logger.info(f"Signal Generated: {signal_reason}")
        else:
            logger.debug(f"Signal Check: {signal_reason}")

    except Exception as e:
        logger.error(f"{Fore.RED}Error generating signals: {e}", exc_info=True)
        return {"long": False, "short": False, "reason": f"Exception: {e}"}

    return {"long": long_signal, "short": short_signal, "reason": signal_reason}


def trading_spell_cycle(cycle_count: int) -> None:
    """Executes one cycle of the trading spell with enhanced precision and logic."""
    logger.info(Fore.MAGENTA + Style.BRIGHT + f"\n--- Starting Cycle {cycle_count} ---")
    start_time = time.time()

    # 1. Fetch Market Data
    df = fetch_market_data(CONFIG.symbol, CONFIG.interval, CONFIG.ohlcv_limit)
    if df is None or df.empty:
        logger.error(
            Fore.RED + "Halting cycle: Market data fetch failed or returned empty."
        )
        return  # Skip cycle

    # Get current price and timestamp from the latest candle
    current_price: Decimal | None = None
    last_timestamp: pd.Timestamp | None = None
    try:
        last_candle = df.iloc[-1]
        # Use close price of the last *completed* candle
        current_price_float = last_candle["close"]
        if pd.isna(current_price_float):
            raise ValueError("Latest close price is NaN")
        current_price = Decimal(str(current_price_float))
        last_timestamp = df.index[-1]
        logger.debug(f"Latest candle: Time={last_timestamp}, Close={current_price:.4f}")

        # Check for stale data (compare last candle time to current time)
        # Ensure timezone awareness or lack thereof is consistent
        now_utc = pd.Timestamp.utcnow().tz_localize(None)  # Naive UTC timestamp
        # Make last_timestamp naive if it's timezone-aware
        if last_timestamp.tzinfo is not None:
            last_timestamp_naive = last_timestamp.tz_convert(None)
        else:
            last_timestamp_naive = last_timestamp

        time_diff = now_utc - last_timestamp_naive
        # Allow for interval duration + some buffer (e.g., 2 intervals)
        allowed_lag = pd.Timedelta(
            EXCHANGE.parse_timeframe(CONFIG.interval) * 2, unit="s"
        ) + pd.Timedelta(minutes=1)
        if time_diff > allowed_lag:
            logger.warning(
                Fore.YELLOW
                + f"Market data may be stale. Last candle: {last_timestamp} ({time_diff} ago). Allowed lag: ~{allowed_lag}"
            )

    except (IndexError, KeyError, ValueError) as e:
        logger.error(
            Fore.RED + f"Failed to get current price/timestamp from DataFrame: {e}"
        )
        return  # Skip cycle
    except (InvalidOperation, TypeError) as e:
        logger.error(
            Fore.RED + f"Error converting current price to Decimal: {e}", exc_info=True
        )
        return  # Skip cycle

    # 2. Calculate Indicators (returns Decimals)
    indicators = calculate_indicators(df)
    if indicators is None:
        logger.error(
            Fore.RED + "Halting cycle: Indicator calculation failed or returned None."
        )
        return  # Skip cycle
    current_atr = indicators.get("atr")  # Keep as Decimal

    # 3. Get Current State (Balance & Positions as Decimals)
    # Fetch balance first
    free_balance, current_equity = get_balance(
        MARKET_INFO.get("quote", "USDT")
    )  # Use quote currency
    if current_equity is None:
        logger.error(
            Fore.RED + "Halting cycle: Failed to fetch current balance/equity."
        )
        # Don't proceed without knowing equity for risk calculation
        return
        # Alternatively, use a cached value? Safer to halt.
        # current_equity = Decimal("-1.0") # Placeholder if proceeding cautiously

    # Fetch positions
    positions = get_current_position(CONFIG.symbol)
    if positions is None:
        logger.error(Fore.RED + "Halting cycle: Failed to fetch current positions.")
        return

    # Ensure positions dict has expected structure and use Decimal
    long_pos = positions.get(
        "long", {"qty": Decimal("0.0"), "entry_price": Decimal("0.0")}
    )
    short_pos = positions.get(
        "short", {"qty": Decimal("0.0"), "entry_price": Decimal("0.0")}
    )
    long_pos.get("qty", Decimal("0.0"))
    short_pos.get("qty", Decimal("0.0"))

    # --- Capture State Snapshot for Status Panel ---
    # Do this *before* potentially modifying state (like TSL management)
    order_tracker_snapshot = {
        "long": order_tracker["long"].copy(),
        "short": order_tracker["short"].copy(),
    }
    positions_snapshot = positions.copy()  # Copy the fetched positions for the panel

    # 4. Manage Trailing Stops (pass Decimals)
    # Check and manage TSL only if a position is actively held.
    # Use the *snapshot* of position state for consistency within the cycle
    active_long_qty = positions_snapshot.get("long", {}).get("qty", Decimal("0.0"))
    active_short_qty = positions_snapshot.get("short", {}).get("qty", Decimal("0.0"))

    if active_long_qty >= CONFIG.position_qty_epsilon:
        logger.debug("Managing TSL for existing LONG position...")
        manage_trailing_stop(
            CONFIG.symbol,
            "long",
            active_long_qty,
            positions_snapshot["long"]["entry_price"],
            current_price,
            current_atr,
        )
    elif active_short_qty >= CONFIG.position_qty_epsilon:
        logger.debug("Managing TSL for existing SHORT position...")
        manage_trailing_stop(
            CONFIG.symbol,
            "short",
            active_short_qty,
            positions_snapshot["short"]["entry_price"],
            current_price,
            current_atr,
        )
    else:
        # If flat, ensure trackers are clear (belt-and-suspenders check)
        if (
            order_tracker["long"]["sl_id"]
            or order_tracker["long"]["tsl_id"]
            or order_tracker["short"]["sl_id"]
            or order_tracker["short"]["tsl_id"]
        ):
            logger.info("Position is flat, ensuring order trackers are cleared.")
            order_tracker["long"] = {"sl_id": None, "tsl_id": None}
            order_tracker["short"] = {"sl_id": None, "tsl_id": None}
            # Update the snapshot if cleared here? No, snapshot should reflect state before management.

    # 5. Generate Trading Signals (pass Decimals)
    signals_data = generate_signals(indicators, current_price)
    signals = {
        "long": signals_data["long"],
        "short": signals_data["short"],
    }  # Extract bools

    # 6. Execute Trades based on Signals
    # Re-fetch positions *after* TSL management and *before* entry decision,
    # in case TSL hit and closed the position during management.
    # This prevents trying to enter a new trade if the position was just closed.
    current_positions_after_tsl = get_current_position(CONFIG.symbol)
    if current_positions_after_tsl is None:
        logger.error(
            Fore.RED
            + "Halting cycle: Failed to re-fetch positions after TSL management."
        )
        return

    long_qty_after_tsl = current_positions_after_tsl.get("long", {}).get(
        "qty", Decimal("0.0")
    )
    short_qty_after_tsl = current_positions_after_tsl.get("short", {}).get(
        "qty", Decimal("0.0")
    )
    is_flat_after_tsl = (
        long_qty_after_tsl < CONFIG.position_qty_epsilon
        and short_qty_after_tsl < CONFIG.position_qty_epsilon
    )
    logger.debug(
        f"Position Status After TSL Check: Flat = {is_flat_after_tsl} (Long Qty: {long_qty_after_tsl}, Short Qty: {short_qty_after_tsl})"
    )

    if is_flat_after_tsl:
        if signals.get("long"):
            logger.info(
                Fore.GREEN + Style.BRIGHT + "Long signal detected! Attempting entry..."
            )
            if place_risked_market_order(
                CONFIG.symbol, "buy", CONFIG.risk_percentage, current_atr
            ):
                logger.info(f"Long entry successful for cycle {cycle_count}.")
                # No need to pause here, next cycle starts after main loop sleep
            else:
                logger.error(f"Long entry attempt failed for cycle {cycle_count}.")
                # Optional: Implement cooldown logic here if needed

        elif signals.get("short"):
            logger.info(
                Fore.RED + Style.BRIGHT + "Short signal detected! Attempting entry."
            )
            if place_risked_market_order(
                CONFIG.symbol, "sell", CONFIG.risk_percentage, current_atr
            ):
                logger.info(f"Short entry successful for cycle {cycle_count}.")
            else:
                logger.error(f"Short entry attempt failed for cycle {cycle_count}.")
                # Optional: Implement cooldown logic here if needed

        # If a trade was attempted, maybe pause briefly? Usually handled by main loop sleep.
        # if trade_attempted:
        #     logger.debug("Brief pause after trade attempt...")
        #     time.sleep(1)

    elif not is_flat_after_tsl:
        logger.info("Position already open, skipping new entry signals.")
        # Future: Add exit logic based on counter-signals or other conditions if desired.
        # Example: if long_qty_after_tsl > 0 and signals.get("short"): close_position("long")
        # Example: if short_qty_after_tsl > 0 and signals.get("long"): close_position("short")

    # 7. Display Status Panel
    # Use the state captured *before* TSL management and trade execution for consistency
    print_status_panel(
        cycle_count,
        last_timestamp,
        current_price,
        indicators,
        positions_snapshot,
        current_equity,
        signals,
        order_tracker_snapshot,  # Use the snapshots
    )

    end_time = time.time()
    logger.info(
        Fore.MAGENTA
        + f"--- Cycle {cycle_count} Complete (Duration: {end_time - start_time:.2f}s) ---"
    )


def graceful_shutdown() -> None:
    """Dispels active orders and closes open positions gracefully with precision."""
    logger.warning(
        Fore.YELLOW + Style.BRIGHT + "\nInitiating Graceful Shutdown Sequence..."
    )
    termux_notify("Shutdown", f"Closing orders/positions for {CONFIG.symbol}.")

    global EXCHANGE, MARKET_INFO
    if EXCHANGE is None or MARKET_INFO is None:
        logger.error(
            Fore.RED
            + "Exchange object or Market Info not available. Cannot perform clean shutdown."
        )
        return

    symbol = CONFIG.symbol
    MARKET_INFO.get("id")  # Use market ID if needed

    # 1. Cancel All Open Orders for the Symbol
    try:
        logger.info(Fore.CYAN + f"Dispelling all open orders for {symbol}...")
        # Bybit V5 cancelAllOrders might need category
        params = {"category": CONFIG.market_type} if CONFIG.market_type else {}
        # Fetch open orders first to log IDs before cancelling (best effort)
        open_orders_list = []
        try:
            open_orders_list = fetch_with_retries(
                EXCHANGE.fetch_open_orders, symbol, params=params
            )
            if open_orders_list:
                order_ids = [o.get("id", "N/A") for o in open_orders_list]
                logger.info(
                    f"Found {len(open_orders_list)} open orders to cancel: {', '.join(order_ids)}"
                )
            else:
                logger.info("No open orders found via fetch_open_orders.")
        except Exception as fetch_err:
            logger.warning(
                Fore.YELLOW
                + f"Could not fetch open orders before cancelling: {fetch_err}. Proceeding with cancel all."
            )

        # Send cancel_all command
        try:
            response = fetch_with_retries(
                EXCHANGE.cancel_all_orders, symbol, params=params
            )
            # Check response for success indicators if possible (Bybit V5 often returns list of cancelled IDs or success code)
            success = False
            if (
                isinstance(response, dict)
                and response.get("info", {}).get("retCode") == 0
            ):
                success = True
                logger.info(Fore.GREEN + "Cancel command successful (retCode 0).")
            elif isinstance(
                response, list
            ):  # Sometimes returns list of cancelled orders
                success = True
                logger.info(
                    Fore.GREEN + "Cancel command likely successful (returned list)."
                )

            if success:
                logger.info(
                    Fore.GREEN
                    + f"Cancel all orders command sent successfully for {symbol}."
                )
            else:
                logger.warning(
                    Fore.YELLOW
                    + f"Cancel all orders command sent, but success confirmation unclear. Response: {str(response)[:200]}"
                )

        except Exception as cancel_err:
            logger.error(
                Fore.RED
                + f"Error sending cancel_all_orders command: {cancel_err}. MANUAL CHECK REQUIRED."
            )

        # Clear local tracker regardless, assuming intent was cancellation
        logger.info("Clearing local order tracker.")
        order_tracker["long"] = {"sl_id": None, "tsl_id": None}
        order_tracker["short"] = {"sl_id": None, "tsl_id": None}

    except Exception as e:
        logger.error(
            Fore.RED
            + Style.BRIGHT
            + f"Unexpected error during order cancellation phase: {e}. MANUAL CHECK REQUIRED on exchange.",
            exc_info=True,
        )

    # Add a small delay after cancelling orders before checking/closing positions
    logger.info("Waiting briefly after order cancellation...")
    time.sleep(CONFIG.order_check_delay_seconds + 1)  # Slightly longer wait

    # 2. Close Any Open Positions
    try:
        logger.info(Fore.CYAN + "Checking for lingering positions to close...")
        # Fetch final position state using the dedicated function with retries
        positions = get_current_position(symbol)  # Already uses fetch_with_retries

        closed_count = 0
        if positions:
            min_qty = (
                Decimal(str(MARKET_INFO["limits"]["amount"]["min"]))
                if MARKET_INFO["limits"]["amount"].get("min") is not None
                else Decimal("0")
            )

            for side, pos_data in positions.items():
                qty = pos_data.get("qty", Decimal("0.0"))
                entry_price = pos_data.get("entry_price", Decimal("0.0"))

                # Check if quantity is significant using epsilon
                if qty.copy_abs() >= CONFIG.position_qty_epsilon:
                    close_side = "sell" if side == "long" else "buy"
                    logger.warning(
                        Fore.YELLOW
                        + f"Closing {side} position (Qty: {qty}, Entry: {entry_price}) with market order..."
                    )
                    try:
                        # Format quantity precisely for closure order (use ROUND_DOWN)
                        close_qty_str = format_amount(
                            symbol, qty.copy_abs(), ROUND_DOWN
                        )  # Use absolute value and round down
                        close_qty_decimal = Decimal(close_qty_str)

                        if close_qty_decimal < min_qty:
                            logger.error(
                                f"{Fore.RED}Closure quantity {close_qty_decimal} is below minimum {min_qty}. Cannot place closure order automatically."
                            )
                            continue  # Skip this position closure

                        close_params = {
                            "reduceOnly": True,
                            "category": CONFIG.market_type,
                        }
                        # Use fetch_with_retries for placing the closure order
                        close_order = fetch_with_retries(
                            EXCHANGE.create_market_order,
                            symbol=symbol,
                            side=close_side,
                            amount=float(close_qty_decimal),  # CCXT needs float
                            params=close_params,
                        )
                        if close_order and close_order.get("id"):
                            logger.trade(
                                Fore.GREEN
                                + f"Position closure order placed: ID {close_order.get('id')}"
                            )
                            closed_count += 1
                            # Wait slightly longer to allow fill confirmation
                            time.sleep(CONFIG.order_check_delay_seconds + 1)
                            # Optional: Verify closure order status
                            # closure_status = check_order_status(close_order['id'], symbol)
                            # logger.info(f"Closure order {close_order['id']} final status: {closure_status.get('status') if closure_status else 'Unknown'}")
                        elif (
                            close_order
                            and close_order.get("info", {}).get("retCode") == 0
                        ):  # Check for success code even without ID
                            logger.trade(
                                Fore.GREEN
                                + "Position closure order likely successful (retCode 0)."
                            )
                            closed_count += 1
                            time.sleep(CONFIG.order_check_delay_seconds + 1)
                        else:
                            # Log critical error if closure fails
                            error_msg = (
                                close_order.get("info", {}).get(
                                    "retMsg", "No ID and no success code."
                                )
                                if isinstance(close_order, dict)
                                else str(close_order)
                            )
                            logger.critical(
                                Fore.RED
                                + Style.BRIGHT
                                + f"FAILED TO PLACE closure order for {side} position ({qty}). Response: {error_msg}. MANUAL INTERVENTION REQUIRED!"
                            )

                    except (ccxt.ExchangeError, ccxt.InvalidOrder) as e:
                        logger.critical(
                            Fore.RED
                            + Style.BRIGHT
                            + f"FAILED TO CLOSE {side} position ({qty}): {e}. MANUAL INTERVENTION REQUIRED!"
                        )
                    except Exception as e:
                        logger.critical(
                            Fore.RED
                            + Style.BRIGHT
                            + f"Unexpected error closing {side} position: {e}. MANUAL INTERVENTION REQUIRED!",
                            exc_info=True,
                        )
                else:
                    logger.debug(f"No significant {side} position found (Qty: {qty}).")

            if closed_count == 0 and any(
                p["qty"].copy_abs() >= CONFIG.position_qty_epsilon
                for p in positions.values()
            ):
                logger.warning(
                    Fore.YELLOW
                    + "Attempted shutdown but no closure orders were successfully placed for open positions. MANUAL CHECK REQUIRED."
                )
            elif closed_count > 0:
                logger.info(
                    Fore.GREEN + f"Successfully placed {closed_count} closure order(s)."
                )
            else:
                logger.info(Fore.GREEN + "No open positions found requiring closure.")

        elif positions is None:
            logger.error(
                Fore.RED
                + Style.BRIGHT
                + "Could not fetch final positions during shutdown. MANUAL CHECK REQUIRED."
            )

    except Exception as e:
        logger.error(
            Fore.RED
            + Style.BRIGHT
            + f"Error during position closure check: {e}. Manual check advised.",
            exc_info=True,
        )

    logger.warning(Fore.YELLOW + Style.BRIGHT + "Graceful Shutdown Sequence Complete.")
    termux_notify("Shutdown Complete", f"{CONFIG.symbol} bot stopped.")


# --- Main Spell Invocation ---
if __name__ == "__main__":
    logger.info(
        Back.MAGENTA
        + Fore.WHITE
        + Style.BRIGHT
        + "*** Pyrmethus Termux Trading Spell Activated (v2.1.1 Precision/Robust) ***"
    )
    logger.info(f"Log Level: {log_level_str}")
    # Log key configuration parameters
    logger.info(f"Symbol: {CONFIG.symbol} ({CONFIG.market_type.capitalize()})")
    logger.info(f"Timeframe: {CONFIG.interval}")
    logger.info(f"Risk per trade: {CONFIG.risk_percentage * 100:.2f}%")
    logger.info(f"SL Multiplier: {CONFIG.sl_atr_multiplier} * ATR")
    logger.info(f"TSL Activation: {CONFIG.tsl_activation_atr_multiplier} * ATR Profit")
    logger.info(f"TSL Trail Percent: {CONFIG.trailing_stop_percent}%")
    logger.info(
        f"Trigger Prices: SL={CONFIG.sl_trigger_by}, TSL={CONFIG.tsl_trigger_by}"
    )
    logger.info(
        f"Trend Filter EMA({CONFIG.trend_ema_period}): {CONFIG.trade_only_with_trend}"
    )
    logger.info(f"Position Quantity Epsilon: {CONFIG.position_qty_epsilon}")
    logger.info(f"Loop Interval: {CONFIG.loop_sleep_seconds}s")

    if (
        MARKET_INFO and EXCHANGE
    ):  # Check if market info and exchange loaded successfully
        termux_notify("Bot Started", f"Monitoring {CONFIG.symbol} (v2.1.1)")
        logger.info(
            Fore.GREEN
            + Style.BRIGHT
            + "Initialization complete. Awaiting market whispers..."
        )
    else:
        # Error should have been logged during init, exit was likely called, but double-check.
        logger.critical(
            Fore.RED
            + Style.BRIGHT
            + "Exchange or Market info failed to load during initialization. Cannot start trading loop."
        )
        sys.exit(1)

    cycle = 0
    try:
        while True:
            cycle += 1
            trading_spell_cycle(cycle)
            logger.info(
                Fore.BLUE
                + f"Resting for {CONFIG.loop_sleep_seconds} seconds before next cycle..."
            )
            time.sleep(CONFIG.loop_sleep_seconds)

    except KeyboardInterrupt:
        logger.warning(Fore.YELLOW + "\nCtrl+C detected! Initiating shutdown...")
        graceful_shutdown()
    except Exception as e:
        logger.critical(
            Fore.RED + Style.BRIGHT + f"\nFATAL RUNTIME ERROR in Main Loop: {e}",
            exc_info=True,
        )
        termux_notify("Bot CRASHED", f"{CONFIG.symbol} Error: Check logs!")
        logger.warning(Fore.YELLOW + "Attempting graceful shutdown after crash...")
        graceful_shutdown()  # Attempt cleanup even on unexpected crash
        sys.exit(1)
    finally:
        # Ensure logs are flushed before exit
        logging.shutdown()
