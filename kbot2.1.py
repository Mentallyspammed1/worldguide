# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Termux Trading Spell (v2.1 - Precision Enhanced & Robust)
# Conjures market insights and executes trades on Bybit Futures with refined precision and improved robustness.

import logging
import os
import sys
import time
from decimal import Decimal, InvalidOperation
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
# Standard float precision is usually sufficient for trading logic, but Decimal offers exactness.
# We will primarily use it for critical financial calculations like position sizing.
# getcontext().prec = 28 # Example: Set precision to 28 digits (default is usually sufficient)
# Let's keep the default precision unless specific issues arise.

# --- Arcane Configuration ---

# Summon secrets from the .env scroll
load_dotenv()

# Configure the Ethereal Log Scribe
# Define custom log levels for trade actions
TRADE_LEVEL_NUM = logging.INFO + 5  # Between INFO and WARNING
logging.addLevelName(TRADE_LEVEL_NUM, "TRADE")


def trade(self, message, *args, **kws) -> None:
    if self.isEnabledFor(TRADE_LEVEL_NUM):
        self._log(TRADE_LEVEL_NUM, message, args, **kws)


logging.Logger.trade = trade

log_formatter = logging.Formatter(
    Fore.CYAN
    + "%(asctime)s "
    + Style.BRIGHT
    + "[%(levelname)s] "
    + Style.RESET_ALL
    + Fore.WHITE
    + "%(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(
    logging.INFO
)  # Set to DEBUG for more verbose output (e.g., raw API responses)
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
        )  # 'linear' (USDT) or 'inverse' (Coin margined)
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
        )  # Options: LastPrice, MarkPrice, IndexPrice
        self.tsl_trigger_by = self._get_env(
            "TSL_TRIGGER_BY", "LastPrice", Fore.YELLOW
        )  # Usually same as SL, check Bybit docs

        # Epsilon: A small value to compare floating point/decimal numbers for near-equality, crucial for quantity checks.
        # Needs to be smaller than the smallest possible order size increment (step size). Fetch this dynamically if possible.
        # For now, using a reasonably small Decimal value.
        self.position_qty_epsilon = Decimal(
            "0.0000001"
        )  # Threshold for considering position effectively zero (as Decimal)

        self.api_key = self._get_env("BYBIT_API_KEY", None, Fore.RED)
        self.api_secret = self._get_env("BYBIT_API_SECRET", None, Fore.RED)
        self.ohlcv_limit = 200  # Number of candles to fetch
        self.loop_sleep_seconds = 15  # Pause between cycles
        self.order_check_delay_seconds = (
            2  # Wait before checking order status after placement
        )
        self.order_check_timeout_seconds = 10  # Max time to wait for order status check
        self.max_fetch_retries = 3  # Retries for fetching data/balance/positions
        self.trade_only_with_trend = self._get_env(
            "TRADE_ONLY_WITH_TREND", "True", Fore.YELLOW, cast_type=bool
        )  # Only trade in direction of trend_ema
        self.trend_ema_period = self._get_env(
            "TREND_EMA_PERIOD", "10", Fore.YELLOW, cast_type=int
        )  # EMA period for trend filter

        if not self.api_key or not self.api_secret:
            logger.critical(
                Fore.RED
                + Style.BRIGHT
                + "BYBIT_API_KEY or BYBIT_API_SECRET not found in .env scroll! Halting."
            )
            sys.exit(1)

    def _get_env(
        self, key: str, default: Any, color: str, cast_type: type = str
    ) -> Any:
        value = os.getenv(key)
        if value is None:
            value = default
            # Don't log warning if default is None (like for API keys where error is raised later)
            if default is not None:
                logger.warning(f"{color}Using default value for {key}: {value}")
        else:
            # Don't log secrets
            log_value = "****" if "SECRET" in key or "KEY" in key else value
            logger.info(f"{color}Summoned {key}: {log_value}")

        try:
            if value is None:
                return None
            if cast_type == bool:
                return str(value).lower() in ["true", "1", "yes", "y"]
            # Handle Decimal casting explicitly
            if cast_type == Decimal:
                try:
                    return Decimal(str(value))
                except InvalidOperation:
                    logger.error(
                        f"{Fore.RED}Invalid numeric value for {key} ('{value}'). Using default: {default}"
                    )
                    return Decimal(str(default))  # Attempt to cast default
            return cast_type(value)  # For int, float, str
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.error(
                f"{Fore.RED}Could not cast {key} ('{value}') to {cast_type.__name__}: {e}. Using default: {default}"
            )
            # Attempt to cast default if value failed
            try:
                if default is None:
                    return None
                if cast_type == bool:
                    return str(default).lower() in ["true", "1", "yes", "y"]
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

# --- Exchange Nexus Initialization ---
try:
    exchange = ccxt.bybit(
        {
            "apiKey": CONFIG.api_key,
            "secret": CONFIG.api_secret,
            "enableRateLimit": True,  # Let CCXT handle rate limiting
            "options": {
                "defaultType": "future",  # Generic futures type
                "defaultSubType": CONFIG.market_type,  # 'linear' or 'inverse'
                "adjustForTimeDifference": True,  # Helps with timestamp issues
                # 'warnOnFetchOpenOrdersWithoutSymbol': False, # Suppress warning if needed, but usually good to specify symbol
            },
        }
    )
    # Test connectivity
    exchange.check_required_credentials()  # Check if keys are valid format
    logger.info("Credentials format check passed.")
    # exchange.fetch_time() # Check clock sync and connectivity
    # logger.info(f"Exchange time synchronized: {exchange.iso8601(exchange.milliseconds())}")

    exchange.load_markets(True)  # Force reload markets
    logger.info(
        Fore.GREEN
        + Style.BRIGHT
        + f"Successfully connected to Bybit Nexus ({CONFIG.market_type.capitalize()} Markets)."
    )

    # Verify symbol exists and get market details
    if CONFIG.symbol not in exchange.markets:
        logger.error(
            Fore.RED
            + Style.BRIGHT
            + f"Symbol {CONFIG.symbol} not found in Bybit {CONFIG.market_type} market spirits."
        )
        # Try to suggest similar available symbols
        available_symbols = [
            s
            for s in exchange.markets
            if exchange.markets[s].get("active")
            and exchange.markets[s].get(
                CONFIG.market_type
            )  # Check if it's linear/inverse
            and exchange.markets[s].get("quote")
            == CONFIG.symbol.split("/")[1].split(":")[
                0
            ]  # Match quote currency (e.g., USDT)
        ]
        # Limit suggestions for clarity
        suggestion_limit = 15
        suggestions = ", ".join(available_symbols[:suggestion_limit])
        if len(available_symbols) > suggestion_limit:
            suggestions += "..."
        logger.info(
            Fore.CYAN
            + f"Available active {CONFIG.market_type} symbols with {CONFIG.symbol.split('/')[1].split(':')[0]} quote (sample): "
            + suggestions
        )
        sys.exit(1)
    else:
        MARKET_INFO = exchange.market(CONFIG.symbol)
        logger.info(
            Fore.CYAN
            + f"Market spirit for {CONFIG.symbol} acknowledged (ID: {MARKET_INFO.get('id')})."
        )
        # Log key precision and limits using Decimal where appropriate
        price_prec = MARKET_INFO["precision"]["price"]
        amount_prec = MARKET_INFO["precision"]["amount"]
        min_amount = MARKET_INFO["limits"]["amount"]["min"]
        max_amount = MARKET_INFO["limits"]["amount"]["max"]
        contract_size = MARKET_INFO.get(
            "contractSize", "1"
        )  # Default to '1' if not present
        min_cost = MARKET_INFO["limits"].get("cost", {}).get("min")

        # Dynamically set epsilon based on amount precision if possible
        try:
            amount_step = exchange.decimal_to_precision(
                1,
                ccxt.ROUND_UP,
                MARKET_INFO["precision"]["amount"],
                ccxt.PRECISION_MODE_DECIMAL_PLACES,
            )
            CONFIG.position_qty_epsilon = Decimal(amount_step) / Decimal(
                "2"
            )  # Half the smallest step
            logger.info(
                f"Dynamically set position_qty_epsilon based on amount precision: {CONFIG.position_qty_epsilon}"
            )
        except Exception as e:
            logger.warning(
                f"Could not determine amount step size for dynamic epsilon: {e}. Using default: {CONFIG.position_qty_epsilon}"
            )

        logger.debug(f"Market Precision: Price={price_prec}, Amount={amount_prec}")
        logger.debug(
            f"Market Limits: Min Amount={min_amount}, Max Amount={max_amount}, Min Cost={min_cost}"
        )
        logger.debug(f"Contract Size: {contract_size}")

        # Validate that we can convert these critical values
        try:
            Decimal(str(price_prec))
            Decimal(str(amount_prec))
            if min_amount is not None:
                Decimal(str(min_amount))
            # Max amount can be None for some markets
            # if max_amount is not None: Decimal(str(max_amount))
            Decimal(str(contract_size))
            if min_cost is not None:
                Decimal(str(min_cost))
        except (InvalidOperation, TypeError, Exception) as e:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}Failed to parse critical market info (precision/limits/size/cost) as numbers: {e}. Halting."
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
        Fore.RED + Style.BRIGHT + f"Exchange Nexus Error during initialization: {e}"
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
    # Check if running within Termux environment more reliably
    if not os.getenv("TERMUX_VERSION"):
        logger.debug("Not running in Termux environment. Skipping notification.")
        return
    try:
        # Use `command -v` for better check if command exists
        toast_cmd_check = os.system("command -v termux-toast > /dev/null 2>&1")
        if toast_cmd_check == 0:  # Command exists
            toast_cmd = "termux-toast"
            # Basic sanitization for shell command arguments (improved slightly)
            safe_title = (
                title.replace('"', "'")
                .replace("`", "'")
                .replace("$", "")
                .replace("\\", "")
            )
            safe_content = (
                content.replace('"', "'")
                .replace("`", "'")
                .replace("$", "")
                .replace("\\", "")
            )
            # Construct command safely - avoid complex shell features in strings
            # Use -s for short duration toast
            cmd = f'{toast_cmd} -g middle -c green -s "{safe_title}: {safe_content}"'
            exit_code = os.system(cmd)
            if exit_code != 0:
                logger.warning(
                    f"termux-toast command failed with exit code {exit_code}."
                )
        else:
            logger.debug("termux-toast command not found. Skipping notification.")
    except Exception as e:
        logger.warning(Fore.YELLOW + f"Could not conjure Termux notification: {e}")


# --- Precision Casting Spells ---


def format_price(symbol: str, price: float | Decimal) -> str:
    """Formats price according to market precision rules using exchange's method."""
    global MARKET_INFO
    if MARKET_INFO is None:
        logger.error(
            f"{Fore.RED}Market info not loaded for {symbol}, cannot format price."
        )
        return str(float(price))  # Fallback, potentially incorrect precision
    try:
        # CCXT's price_to_precision typically expects float input and handles rounding/truncation.
        return exchange.price_to_precision(symbol, float(price))
    except Exception as e:
        logger.error(
            f"{Fore.RED}Error formatting price {price} for {symbol}: {e}. Using fallback."
        )
        # Fallback: Convert Decimal to string with reasonable precision, hoping it's acceptable
        try:
            return f"{Decimal(price):.8f}"  # Adjust precision as needed
        except Exception:
            return str(float(price))  # Last resort fallback


def format_amount(symbol: str, amount: float | Decimal) -> str:
    """Formats amount according to market precision rules using exchange's method."""
    global MARKET_INFO
    if MARKET_INFO is None:
        logger.error(
            f"{Fore.RED}Market info not loaded for {symbol}, cannot format amount."
        )
        return str(float(amount))  # Fallback, potentially incorrect precision
    try:
        # CCXT's amount_to_precision typically expects float input and handles rounding/truncation (often ROUND_DOWN).
        # Crucially, ensure it respects step size.
        return exchange.amount_to_precision(symbol, float(amount))
    except Exception as e:
        logger.error(
            f"{Fore.RED}Error formatting amount {amount} for {symbol}: {e}. Using fallback."
        )
        # Fallback: Convert Decimal to string with reasonable precision
        try:
            return f"{Decimal(amount):.8f}"  # Adjust precision as needed
        except Exception:
            return str(float(amount))  # Last resort fallback


# --- Core Spell Functions ---


def fetch_with_retries(fetch_function, *args, **kwargs) -> Any:
    """Generic wrapper to fetch data with retries and backoff."""
    for attempt in range(CONFIG.max_fetch_retries):
        try:
            return fetch_function(*args, **kwargs)
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
            logger.warning(
                Fore.YELLOW
                + f"{fetch_function.__name__}: Network disturbance (Attempt {attempt + 1}/{CONFIG.max_fetch_retries}): {e}. Retrying..."
            )
            if attempt < CONFIG.max_fetch_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
            else:
                logger.error(
                    Fore.RED
                    + f"{fetch_function.__name__}: Failed after {CONFIG.max_fetch_retries} attempts due to network issues."
                )
                return None  # Indicate failure
        except ccxt.ExchangeNotAvailable as e:
            logger.error(
                Fore.RED
                + f"{fetch_function.__name__}: Exchange not available: {e}. Stopping retries."
            )
            return None  # No point retrying if exchange is down
        except ccxt.ExchangeError as e:
            # Some exchange errors might be temporary (e.g., rate limits not handled by enableRateLimit)
            # Others might be permanent (e.g., invalid symbol). Log and potentially retry.
            logger.warning(
                Fore.YELLOW
                + f"{fetch_function.__name__}: Exchange error (Attempt {attempt + 1}/{CONFIG.max_fetch_retries}): {e}. Retrying..."
            )
            if attempt < CONFIG.max_fetch_retries - 1:
                time.sleep(
                    1 * (attempt + 1)
                )  # Simple backoff for general exchange errors
            else:
                logger.error(
                    Fore.RED
                    + f"{fetch_function.__name__}: Failed after {CONFIG.max_fetch_retries} attempts due to exchange errors."
                )
                return None
        except Exception as e:
            logger.error(
                Fore.RED
                + f"{fetch_function.__name__}: Unexpected shadow encountered: {e}",
                exc_info=True,
            )
            return None  # Stop on unexpected errors
    return None


def fetch_market_data(symbol: str, timeframe: str, limit: int) -> pd.DataFrame | None:
    """Fetch OHLCV data using the retry wrapper."""
    logger.info(
        Fore.CYAN + f"# Channeling market whispers for {symbol} ({timeframe})..."
    )

    # Check if exchange object is valid
    if not hasattr(exchange, "fetch_ohlcv"):
        logger.error(Fore.RED + "Exchange object not properly initialized.")
        return None

    ohlcv_data = fetch_with_retries(
        exchange.fetch_ohlcv, symbol, timeframe, limit=limit
    )

    if ohlcv_data is None:
        logger.error(Fore.RED + "Failed to fetch OHLCV data.")
        return None
    if not ohlcv_data:
        logger.error(Fore.RED + "Received empty OHLCV data.")
        return None

    try:
        df = pd.DataFrame(
            ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        # Convert to numeric, coercing errors (should not happen with valid API data)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows where essential price data is missing
        initial_len = len(df)
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        if len(df) < initial_len:
            logger.warning(
                f"Dropped {initial_len - len(df)} rows with missing price data from OHLCV."
            )

        if df.empty:
            logger.error(
                Fore.RED
                + "DataFrame is empty after processing OHLCV data (all rows had NaNs?)."
            )
            return None

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        # Ensure data is sorted chronologically, although fetch_ohlcv usually guarantees this
        df.sort_index(inplace=True)

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

        # --- EMAs ---
        fast_ema_period = 8
        slow_ema_period = 12
        confirm_ema_period = 5
        # Trend EMA period from config
        trend_ema_period = CONFIG.trend_ema_period

        # Check data length requirements
        required_len_ema = max(
            fast_ema_period, slow_ema_period, trend_ema_period, confirm_ema_period
        )
        if len(df) < required_len_ema:
            logger.warning(
                f"Not enough data ({len(df)}) for all EMA periods (max required: {required_len_ema}). Results may be inaccurate."
            )

        fast_ema_series = close.ewm(span=fast_ema_period, adjust=False).mean()
        slow_ema_series = close.ewm(span=slow_ema_period, adjust=False).mean()
        trend_ema_series = close.ewm(span=trend_ema_period, adjust=False).mean()
        confirm_ema_series = close.ewm(span=confirm_ema_period, adjust=False).mean()

        # --- Stochastic Oscillator (%K, %D) ---
        stoch_period = 10
        smooth_k = 3
        smooth_d = 3
        k_now, d_now = (
            Decimal("50.0"),
            Decimal("50.0"),
        )  # Default neutral Decimal values

        # Need period + smooth_k + smooth_d - 2 data points for smoothed Stoch
        required_len_stoch = stoch_period + smooth_k + smooth_d - 2
        if len(df) < required_len_stoch:
            logger.warning(
                f"Not enough data ({len(df)}) for Stochastic (requires {required_len_stoch}). Using default neutral values."
            )
        else:
            low_min = low.rolling(window=stoch_period).min()
            high_max = high.rolling(window=stoch_period).max()
            # Add epsilon to prevent division by zero if high_max == low_min
            stoch_k_raw = 100 * (close - low_min) / (high_max - low_min + 1e-12)
            stoch_k = stoch_k_raw.rolling(window=smooth_k).mean()
            stoch_d = stoch_k.rolling(window=smooth_d).mean()

            # Get latest non-NaN values
            k_latest = stoch_k.iloc[-1]
            d_latest = stoch_d.iloc[-1]

            if pd.isna(k_latest) or pd.isna(d_latest):
                logger.warning(
                    "Stochastic calculation resulted in NaN for latest value. Using default neutral values."
                )
            else:
                k_now = Decimal(str(k_latest))
                d_now = Decimal(str(d_latest))

        # --- ATR (Average True Range) ---
        atr_period = 10
        atr = Decimal("0.0")  # Default Decimal zero

        # Need atr_period + 1 for shift(), and potentially more for EMA smoothing
        required_len_atr = atr_period + 1  # Minimum for basic TR calc
        if len(df) < required_len_atr:
            logger.warning(
                f"Not enough data ({len(df)}) for ATR (requires {required_len_atr}). Using default value 0."
            )
        else:
            tr_df = pd.DataFrame(index=df.index)
            tr_df["hl"] = high - low
            tr_df["hc"] = (high - close.shift()).abs()
            tr_df["lc"] = (low - close.shift()).abs()
            # Ensure TR is calculated only where all components are valid (especially shift)
            tr_df["tr"] = tr_df[["hl", "hc", "lc"]].max(axis=1)
            tr_df.dropna(
                subset=["tr"], inplace=True
            )  # Drop rows where TR couldn't be calculated (e.g., first row due to shift)

            if not tr_df.empty:
                # Use Exponential Moving Average for ATR for smoother results, common practice
                # Wilder's smoothing (alpha = 1/period) is common for ATR
                atr_series = tr_df["tr"].ewm(alpha=1 / atr_period, adjust=False).mean()
                atr_latest = atr_series.iloc[-1]
                if pd.isna(atr_latest):
                    logger.warning(
                        "ATR calculation resulted in NaN for latest value. Using default value 0."
                    )
                else:
                    atr = Decimal(str(atr_latest))
            else:
                logger.warning(
                    "Could not calculate any True Range values. Using default ATR 0."
                )

        logger.info(Fore.GREEN + "Indicator patterns woven successfully.")
        # Convert final indicator values to Decimal, handling potential NaN from calculations
        # Use '.quantize' for consistent decimal places if desired, but direct conversion is usually fine
        # Define a standard quantizer for prices/ATR and another for percentages like Stoch
        price_quantizer = Decimal(
            "0.00000001"
        )  # 8 decimal places for price-like values
        percent_quantizer = Decimal("0.01")  # 2 decimal places for Stoch

        # Helper to safely convert and quantize
        def safe_decimal(value, quantizer):
            if pd.isna(value):
                return Decimal(0)  # Default to 0 if NaN
            try:
                return Decimal(str(value)).quantize(quantizer)
            except (InvalidOperation, TypeError):
                logger.warning(
                    f"Could not convert indicator value {value} to Decimal. Returning 0."
                )
                return Decimal(0)

        return {
            "fast_ema": safe_decimal(fast_ema_series.iloc[-1], price_quantizer),
            "slow_ema": safe_decimal(slow_ema_series.iloc[-1], price_quantizer),
            "trend_ema": safe_decimal(trend_ema_series.iloc[-1], price_quantizer),
            "confirm_ema": safe_decimal(confirm_ema_series.iloc[-1], price_quantizer),
            "stoch_k": k_now.quantize(
                percent_quantizer
            ),  # Already Decimal from above logic
            "stoch_d": d_now.quantize(
                percent_quantizer
            ),  # Already Decimal from above logic
            "atr": atr.quantize(price_quantizer),  # Already Decimal from above logic
        }
    except Exception as e:
        logger.error(
            Fore.RED + f"Failed to weave indicator patterns: {e}", exc_info=True
        )
        return None


def get_current_position(symbol: str) -> dict[str, dict[str, Any]] | None:
    """Fetch current positions using retry wrapper, returning quantities and prices as Decimals."""
    logger.info(Fore.CYAN + f"# Consulting position spirits for {symbol}...")

    # Initialize with Decimal zero
    pos_dict = {
        "long": {"qty": Decimal("0.0"), "entry_price": Decimal("0.0")},
        "short": {"qty": Decimal("0.0"), "entry_price": Decimal("0.0")},
    }

    positions_data = fetch_with_retries(exchange.fetch_positions, symbols=[symbol])

    if positions_data is None:
        logger.error(
            Fore.RED + f"Failed to fetch positions for {symbol} after retries."
        )
        return None  # Indicate failure to fetch

    if not positions_data:
        logger.info(Fore.BLUE + f"No open positions reported by exchange for {symbol}.")
        return pos_dict

    # Filter positions for the exact symbol requested (fetch_positions with symbol should already do this, but double-check)
    symbol_positions = [p for p in positions_data if p.get("symbol") == symbol]

    if not symbol_positions:
        logger.info(
            Fore.BLUE
            + f"No matching position details found for {symbol} in fetched data (exchange might return empty list if flat)."
        )
        return pos_dict

    # Logic assumes non-hedge mode or aggregates hedge mode positions (summing might be needed if hedge mode used)
    # For Bybit unified margin, typically one entry per symbol/side exists.
    active_positions_found = 0
    for pos in symbol_positions:
        # Use info dictionary for safer access
        pos_info = pos.get("info", {})
        side = pos.get("side")  # 'long' or 'short' (unified field)
        # Contracts field name can vary ('contracts', 'contractSize', 'size', etc.) Check Bybit response structure via debug log if needed.
        # Common Bybit V5 field: 'size'
        contracts_str = pos_info.get(
            "size", pos.get("contracts")
        )  # Amount of contracts/base currency
        entry_price_str = pos_info.get("entryPrice", pos.get("entryPrice"))

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

                # If hedge mode is possible, this might overwrite. Assuming single position per side for now.
                pos_dict[side]["qty"] = contracts
                pos_dict[side]["entry_price"] = entry_price
                logger.info(
                    Fore.YELLOW
                    + f"Found active {side} position: Qty={contracts}, Entry={entry_price}"
                )
                active_positions_found += 1
            except (InvalidOperation, TypeError) as e:
                logger.error(
                    f"Could not parse position data for {side} side: Qty='{contracts_str}', Entry='{entry_price_str}'. Error: {e}"
                )
                continue  # Skip this position entry

    if active_positions_found == 0:
        logger.info(
            Fore.BLUE
            + f"No active non-zero positions found for {symbol} after filtering."
        )

    logger.info(Fore.GREEN + "Position spirits consulted.")
    return pos_dict


def get_balance(currency: str = "USDT") -> tuple[Decimal | None, Decimal | None]:
    """Fetches the free and total balance for a specific currency using retry wrapper, returns Decimals."""
    logger.info(Fore.CYAN + f"# Querying the Vault of {currency}...")

    balance_data = fetch_with_retries(exchange.fetch_balance)

    if balance_data is None:
        logger.error(
            Fore.RED
            + "Failed to fetch balance after retries. Cannot assess risk capital."
        )
        return None, None

    try:
        # Use Decimal for balances
        # Access nested dictionary safely
        free_balance_str = balance_data.get("free", {}).get(currency)
        total_balance_str = balance_data.get("total", {}).get(currency)

        free_balance = (
            Decimal(str(free_balance_str))
            if free_balance_str is not None
            else Decimal("0.0")
        )
        total_balance = (
            Decimal(str(total_balance_str))
            if total_balance_str is not None
            else Decimal("0.0")
        )

        logger.info(
            Fore.GREEN
            + f"Vault contains {free_balance:.4f} free {currency} (Total: {total_balance:.4f})."
        )
        return free_balance, total_balance
    except (InvalidOperation, TypeError) as e:
        logger.error(
            Fore.RED
            + f"Error parsing balance data for {currency}: {e}. Raw data: {balance_data.get(currency)}"
        )
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
    """Checks order status with retries and timeout. Returns the order dict or None."""
    logger.info(Fore.CYAN + f"Verifying status of order {order_id} for {symbol}...")
    start_time = time.time()
    last_status = "unknown"
    while time.time() - start_time < timeout:
        try:
            # Use fetch_with_retries for the underlying fetch_order call
            order_status = fetch_with_retries(exchange.fetch_order, order_id, symbol)

            if order_status:
                last_status = order_status.get("status", "unknown")
                logger.info(f"Order {order_id} status check: {last_status}")
                # Return the full order dict if found, regardless of status initially
                # We might want to exit early if status is terminal (closed, canceled, rejected)
                if last_status in ["closed", "canceled", "rejected", "expired"]:
                    logger.info(
                        f"Order {order_id} reached terminal state: {last_status}."
                    )
                    return order_status
                # If open or partially filled, keep checking or return current state? For now, return immediately.
                return order_status
            else:
                # This case means fetch_with_retries failed after retries (e.g., network issues)
                # or fetch_order returned None/empty structure unexpectedly.
                logger.warning(
                    f"fetch_order call failed or returned empty structure for {order_id} after retries. Check logs."
                )
                # Continue the loop to retry check_order_status itself
                pass  # Go to sleep and retry check

        except ccxt.OrderNotFound:
            # Order is definitively not found on the exchange. It's gone.
            logger.error(
                Fore.RED + f"Order {order_id} confirmed NOT FOUND by exchange."
            )
            return None  # Explicitly indicate not found (terminal state)

        except Exception as e:
            # Catch any other unexpected error during the check itself (not the underlying fetch)
            logger.error(
                f"Unexpected error during order status check loop for {order_id}: {e}",
                exc_info=True,
            )
            # Decide whether to retry or fail; retrying might be okay here.
            pass  # Go to sleep and retry check

        # Wait before the next check_order_status attempt
        check_interval = 1  # seconds
        # Ensure we don't sleep past the timeout
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
    return None  # Indicate timeout or persistent failure to get a terminal status


def place_risked_market_order(
    symbol: str, side: str, risk_percentage: Decimal, atr: Decimal
) -> bool:
    """Places a market order with calculated size and initial ATR-based stop-loss, using Decimal precision."""
    logger.trade(
        Style.BRIGHT + f"Attempting {side.upper()} market entry for {symbol}..."
    )  # Use custom trade level

    global MARKET_INFO
    if MARKET_INFO is None:
        logger.error(Fore.RED + "Market info not available. Cannot place order.")
        return False

    # --- Pre-computation & Validation ---
    free_balance, _ = get_balance(
        "USDT"
    )  # Assuming USDT is the quote currency for risk calc
    if free_balance is None or free_balance <= Decimal("0"):
        logger.error(
            Fore.RED + "Cannot place order: Invalid or zero available balance."
        )
        return False

    if atr is None or atr <= Decimal("0"):
        logger.error(
            Fore.RED
            + f"Cannot place order: Invalid ATR value ({atr}). Check indicator calculation."
        )
        return False

    try:
        # Fetch current price using fetch_ticker with retries
        ticker_data = fetch_with_retries(exchange.fetch_ticker, symbol)
        if not ticker_data or ticker_data.get("last") is None:
            logger.error(
                Fore.RED
                + "Cannot fetch current ticker price for sizing/SL calculation."
            )
            return False
        price = Decimal(str(ticker_data["last"]))

        # --- Calculate Stop Loss Price ---
        sl_distance_points = CONFIG.sl_atr_multiplier * atr
        if side == "buy":
            sl_price_raw = price - sl_distance_points
        else:  # side == "sell"
            sl_price_raw = price + sl_distance_points

        # Format SL price according to market precision *before* using it in calculations
        sl_price_formatted_str = format_price(symbol, sl_price_raw)
        sl_price = Decimal(sl_price_formatted_str)  # Use the formatted price as Decimal
        logger.debug(
            f"Current Price: {price}, ATR: {atr:.6f}, SL Multiplier: {CONFIG.sl_atr_multiplier}"
        )
        logger.debug(f"SL Distance Points: {sl_distance_points:.6f}")
        logger.debug(
            f"Raw SL Price: {sl_price_raw:.6f}, Formatted SL Price: {sl_price}"
        )

        # Sanity check SL placement relative to current price
        if side == "buy" and sl_price >= price:
            logger.error(
                Fore.RED
                + f"Calculated SL price ({sl_price}) is >= current price ({price}). ATR might be too large or price feed issue? Aborting."
            )
            return False
        if side == "sell" and sl_price <= price:
            logger.error(
                Fore.RED
                + f"Calculated SL price ({sl_price}) is <= current price ({price}). ATR might be too large or price feed issue? Aborting."
            )
            return False

        # --- Calculate Position Size ---
        risk_amount_usd = free_balance * risk_percentage
        # Stop distance in quote currency (absolute difference between entry and SL price)
        stop_distance_usd = abs(
            price - sl_price
        )  # Use current price as estimated entry

        if stop_distance_usd <= Decimal("0"):
            logger.error(
                Fore.RED
                + f"Stop distance is zero or negative ({stop_distance_usd}). Check ATR, multiplier, or market precision. Cannot calculate size."
            )
            return False

        # Calculate quantity based on contract size and linear/inverse type
        Decimal(str(MARKET_INFO.get("contractSize", "1")))
        qty_raw = Decimal("0")

        # Sizing logic needs careful checking based on Bybit's contract specs (Linear vs Inverse)
        if CONFIG.market_type == "linear":
            # For Linear (e.g., BTC/USDT:USDT): Size is in Base currency (BTC). Value = Size * Price.
            # Risk Amount (USDT) = Qty (Base) * Stop Distance (USDT)
            # Qty (Base) = Risk Amount (USDT) / Stop Distance (USDT)
            qty_raw = risk_amount_usd / stop_distance_usd
            logger.debug(
                f"Linear Sizing: Qty (Base) = {risk_amount_usd:.4f} / {stop_distance_usd:.4f} = {qty_raw}"
            )

        elif CONFIG.market_type == "inverse":
            # For Inverse (e.g., BTC/USD:BTC): Size is in Contracts (often USD value). Value = Size (Contracts) * ContractValue / Price.
            # Risk Amount (Quote = USD) = Qty (Contracts) * ContractValue (USD/Contract) * abs(1/entry_price - 1/sl_price) (Change in BTC per contract) * Current Price (USD/BTC)
            # This is complex. A common simplification (verify!):
            # Qty (Contracts) = Risk Amount (USD) / (Stop Distance (USD) * Contract Size (Base/Contract)) * Price (USD/Base) ??? No, that's more complex.
            # Let's use: Risk Amount (USD) = Qty (Contracts) * Stop Distance (USD) / Entry Price (USD)  <- Approximation? Needs validation.
            # Qty (Contracts) = Risk Amount (USD) * Entry Price (USD) / Stop Distance (USD)
            # This assumes contract size represents $1 value, which is common for Bybit inverse. CHECK CONTRACT SPECS.
            logger.warning(
                Fore.YELLOW
                + "Inverse contract sizing uses approximation: Qty = (Risk * Entry) / StopDist. VERIFY THIS against Bybit contract specs."
            )
            if price <= Decimal("0"):
                logger.error(
                    Fore.RED
                    + "Cannot calculate inverse size with zero or negative price."
                )
                return False
            qty_raw = (risk_amount_usd * price) / stop_distance_usd
            logger.debug(
                f"Inverse Sizing (Approx): Qty (Contracts) = ({risk_amount_usd:.4f} * {price:.4f}) / {stop_distance_usd:.4f} = {qty_raw}"
            )

        else:
            logger.error(f"Unsupported market type for sizing: {CONFIG.market_type}")
            return False

        # Format quantity according to market precision (ROUND_DOWN implicitly by amount_to_precision usually)
        qty_formatted_str = format_amount(symbol, qty_raw)
        qty = Decimal(qty_formatted_str)
        logger.debug(
            f"Risk Amount: {risk_amount_usd:.4f} USDT, Stop Distance: {stop_distance_usd:.4f} USDT"
        )
        logger.debug(f"Raw Qty: {qty_raw:.8f}, Formatted Qty: {qty}")

        # --- Validate Quantity Against Market Limits ---
        min_qty_str = MARKET_INFO.get("limits", {}).get("amount", {}).get("min")
        max_qty_str = MARKET_INFO.get("limits", {}).get("amount", {}).get("max")
        min_qty = Decimal(str(min_qty_str)) if min_qty_str is not None else Decimal("0")
        max_qty = (
            Decimal(str(max_qty_str)) if max_qty_str is not None else None
        )  # Max can be None

        if qty < min_qty or qty.is_zero() or qty < CONFIG.position_qty_epsilon:
            logger.error(
                Fore.RED
                + f"Calculated quantity ({qty}) is zero or below minimum ({min_qty}). Risk amount, price movement, or ATR might be too small. Cannot place order."
            )
            return False
        if max_qty is not None and qty > max_qty:
            logger.warning(
                Fore.YELLOW
                + f"Calculated quantity {qty} exceeds maximum {max_qty}. Capping order size to {max_qty}."
            )
            qty = max_qty  # Use the Decimal max_qty
            # Re-format capped amount - crucial!
            qty_formatted_str = format_amount(symbol, qty)
            qty = Decimal(qty_formatted_str)
            logger.info(f"Re-formatted capped Qty: {qty}")
            if qty < min_qty:  # Double check after re-formatting capped value
                logger.error(
                    Fore.RED
                    + f"Capped quantity ({qty}) is now below minimum ({min_qty}). Aborting."
                )
                return False

        # Validate minimum cost if available
        min_cost_str = MARKET_INFO.get("limits", {}).get("cost", {}).get("min")
        if min_cost_str is not None:
            min_cost = Decimal(str(min_cost_str))
            estimated_cost = qty * price  # Approximation, depends on linear/inverse
            if estimated_cost < min_cost:
                logger.error(
                    Fore.RED
                    + f"Estimated order cost ({estimated_cost:.4f}) is below minimum required ({min_cost:.4f}). Increase risk or adjust strategy. Cannot place order."
                )
                return False

        logger.info(
            Fore.YELLOW
            + f"Calculated Order: Side={side.upper()}, Qty={qty}, Entry≈{price:.4f}, SL={sl_price} (ATR={atr:.4f})"
        )

        # --- Cast the Market Order Spell ---
        logger.trade(f"Submitting {side.upper()} market order for {qty} {symbol}...")
        order_params = {}  # No extra params needed for basic market order
        order = exchange.create_market_order(
            symbol, side, float(qty), params=order_params
        )  # CCXT expects float amount
        order_id = order.get("id")
        logger.debug(f"Market order raw response: {order}")
        if not order_id:
            logger.error(Fore.RED + "Market order submission failed to return an ID.")
            # Check if order info contains error details
            if order.get("info", {}).get("retMsg"):
                logger.error(f"Exchange message: {order['info']['retMsg']}")
            return False
        logger.trade(f"Market order submitted: ID {order_id}")

        # --- Verify Order Fill (Crucial Step) ---
        logger.info(
            f"Waiting {CONFIG.order_check_delay_seconds}s before checking fill status for order {order_id}..."
        )
        time.sleep(CONFIG.order_check_delay_seconds)  # Allow time for potential fill
        order_status_data = check_order_status(
            order_id, symbol, timeout=CONFIG.order_check_timeout_seconds
        )

        filled_qty = Decimal("0.0")
        average_price = price  # Fallback to estimated entry price if check fails
        order_final_status = "unknown"

        if order_status_data:
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
                    average_price = Decimal(
                        str(average_str)
                    )  # Use actual fill price if available
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
            elif order_final_status in ["open", "partially_filled"]:
                # Market orders shouldn't stay 'open' long. Partial fills need handling.
                logger.warning(
                    Fore.YELLOW
                    + f"Order {order_id} status is '{order_final_status}'. Filled Qty: {filled_qty}. SL will be based on filled amount."
                )
                if filled_qty < CONFIG.position_qty_epsilon:
                    logger.error(
                        Fore.RED
                        + f"Order {order_id} has status '{order_final_status}' but filled quantity is effectively zero ({filled_qty}). Aborting SL placement."
                    )
                    # Attempt to cancel just in case it's stuck somehow (unlikely for market)
                    try:
                        exchange.cancel_order(order_id, symbol)
                        logger.info(
                            f"Attempted cancellation of stuck order {order_id}."
                        )
                    except Exception as cancel_err:
                        logger.warning(
                            f"Failed to cancel stuck order {order_id}: {cancel_err}"
                        )
                    return False
                # Continue, but use filled_qty for SL
            else:  # canceled, rejected, expired, failed, unknown
                logger.error(
                    Fore.RED
                    + Style.BRIGHT
                    + f"Order {order_id} did not fill successfully: Status '{order_final_status}'. Aborting SL placement."
                )
                # Attempt to cancel just in case it's stuck somehow
                if (
                    order_final_status != "canceled"
                ):  # Avoid cancelling already cancelled order
                    try:
                        exchange.cancel_order(order_id, symbol)
                        logger.info(
                            f"Attempted cancellation of failed order {order_id}."
                        )
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
                exchange.cancel_order(order_id, symbol)
                logger.info(
                    f"Attempted cancellation of unknown status order {order_id}."
                )
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
        # Use the SL price calculated earlier based on estimated entry, already formatted
        sl_price_str_for_order = sl_price_formatted_str  # Already formatted string
        # Use the *actual filled quantity* for the SL order size, re-format it precisely
        sl_qty_str_for_order = format_amount(symbol, filled_qty)

        # Define parameters for the stop-loss order
        sl_params = {
            "stopLossPrice": sl_price_str_for_order,  # Trigger price for the stop market order
            "reduceOnly": True,
            "triggerPrice": sl_price_str_for_order,  # CCXT often uses stopLossPrice, but some exchanges might use triggerPrice. Redundant but sometimes safe.
            "triggerBy": CONFIG.sl_trigger_by,  # e.g., 'LastPrice', 'MarkPrice'
            # Bybit specific potentially useful params (check CCXT unification/docs for v5 API)
            # 'tpslMode': 'Full', # or 'Partial' - affects if TP/SL apply to whole position
            # 'slTriggerBy': CONFIG.sl_trigger_by, # More specific param if available
            # 'positionIdx': 0 # For Bybit unified: 0 for one-way, 1 for long hedge, 2 for short hedge (assuming one-way mode)
        }
        logger.trade(
            f"Placing SL order: Side={sl_order_side}, Qty={sl_qty_str_for_order}, Trigger={sl_price_str_for_order}, TriggerBy={CONFIG.sl_trigger_by}"
        )
        logger.debug(f"SL Params: {sl_params}")

        try:
            # Use create_order with stop type. CCXT standard is often 'stop_market' or just 'stop'.
            # 'stop_market' is generally preferred for guaranteed exit at market price once triggered.
            # Check exchange.has['createStopMarketOrder'] or similar if needed.
            sl_order = exchange.create_order(
                symbol=symbol,
                type="stop_market",  # Explicitly use stop-market type
                side=sl_order_side,
                amount=float(sl_qty_str_for_order),  # CCXT expects float amount
                price=None,  # Market stop loss doesn't need a limit price
                params=sl_params,
            )
            sl_order_id = sl_order.get("id")
            logger.debug(f"SL order raw response: {sl_order}")
            if not sl_order_id:
                # Check info for error message
                error_msg = sl_order.get("info", {}).get("retMsg", "Unknown reason.")
                raise ccxt.ExchangeError(
                    f"Stop loss order placement did not return an ID. Exchange message: {error_msg}"
                )

            # --- Update Global State ---
            # CRITICAL: Clear any previous tracker state for this side before setting new IDs
            order_tracker[position_side] = {"sl_id": sl_order_id, "tsl_id": None}
            logger.trade(
                Fore.GREEN
                + Style.BRIGHT
                + f"Initial SL placed for {position_side.upper()} position: ID {sl_order_id}, Trigger: {sl_price_str_for_order}"
            )

            # Use actual average fill price in notification
            entry_msg = (
                f"ENTERED {side.upper()} {filled_qty} {symbol.split('/')[0]} @ {average_price:.4f}. "
                f"Initial SL @ {sl_price_str_for_order} (ID: {sl_order_id}). TSL pending profit threshold."
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
                # Use the same filled quantity and opposite side
                close_qty_str = format_amount(symbol, filled_qty)
                emergency_close_order = exchange.create_market_order(
                    symbol,
                    sl_order_side,
                    float(close_qty_str),
                    params={"reduceOnly": True},
                )
                logger.trade(
                    Fore.GREEN
                    + f"Emergency closure order placed: ID {emergency_close_order.get('id')}"
                )
                # Reset tracker state as position *should* be closed
                order_tracker[position_side] = {"sl_id": None, "tsl_id": None}
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
            # Optionally trigger emergency closure here as well? Safer to log and let user decide/next cycle handle?
            # Let's log clearly and return False. The position exists but is unprotected.
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
        if hasattr(e, "args") and len(e.args) > 0 and isinstance(e.args[0], str):
            if "info" in e.args[0]:  # CCXT often includes raw response here
                logger.error(
                    f"Raw exchange response excerpt: {e.args[0][:500]}"
                )  # Log first 500 chars
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
    global order_tracker  # We need to modify the global tracker

    # --- Initial Checks ---
    # Ensure position is valid and significant
    if position_qty < CONFIG.position_qty_epsilon or entry_price <= Decimal("0"):
        # If position seems closed/invalid, ensure local tracker reflects this.
        if (
            order_tracker[position_side]["sl_id"]
            or order_tracker[position_side]["tsl_id"]
        ):
            logger.debug(
                f"Position {position_side} appears closed or invalid (Qty: {position_qty}, Entry: {entry_price}). Clearing stale order trackers."
            )
            order_tracker[position_side] = {"sl_id": None, "tsl_id": None}
        return  # No position to manage TSL for

    # Check if ATR is valid for activation calculation
    if atr is None or atr <= Decimal("0"):
        logger.warning(
            Fore.YELLOW + "Cannot evaluate TSL activation: Invalid ATR value."
        )
        return

    # --- Get Current Tracker State ---
    initial_sl_id = order_tracker[position_side]["sl_id"]
    active_tsl_id = order_tracker[position_side]["tsl_id"]

    # If TSL is already active, the exchange handles the trailing.
    # Optionally: Could add a periodic check here to confirm the TSL order still exists on the exchange,
    # but this adds API calls and complexity. For now, assume it's active if we have the ID.
    if active_tsl_id:
        logger.debug(
            f"{position_side.upper()} TSL (ID: {active_tsl_id}) is already active. Exchange manages the trail."
        )
        # Sanity check: ensure initial SL ID is None if TSL is active
        if initial_sl_id:
            logger.warning(
                f"Inconsistent state: TSL active (ID: {active_tsl_id}) but initial SL ID ({initial_sl_id}) is also present. Clearing initial SL ID."
            )
            order_tracker[position_side]["sl_id"] = None
        return  # Nothing more to do here if TSL is active

    # If TSL is not active, check if we *should* activate it.
    # Requires an initial SL to be present (we replace SL with TSL).
    if not initial_sl_id:
        logger.debug(
            f"Cannot activate TSL for {position_side.upper()}: Initial SL ID is missing from tracker. Either SL was triggered, manually cancelled, or never placed correctly."
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
        logger.trade(
            f"Attempting to cancel initial SL (ID: {initial_sl_id}) before placing TSL..."
        )
        try:
            # Use fetch_with_retries for cancellation robustness
            cancel_response = fetch_with_retries(
                exchange.cancel_order, initial_sl_id, symbol
            )
            # Check response - some exchanges return order structure, others just confirmation.
            # If OrderNotFound is raised, it's also considered success here.
            logger.trade(
                Fore.GREEN
                + f"Successfully cancelled initial SL (ID: {initial_sl_id}). Response: {cancel_response}"
            )
            order_tracker[position_side]["sl_id"] = (
                None  # Mark as cancelled locally *only after success*
            )

        except ccxt.OrderNotFound:
            logger.warning(
                Fore.YELLOW
                + f"Initial SL (ID: {initial_sl_id}) already gone (not found) when trying to cancel. Proceeding with TSL placement."
            )
            order_tracker[position_side]["sl_id"] = None  # Assume it's gone
        except (ccxt.ExchangeError, ccxt.NetworkError) as e:
            logger.error(
                Fore.RED
                + Style.BRIGHT
                + f"Failed to cancel initial SL (ID: {initial_sl_id}): {e}. Aborting TSL placement to avoid potential duplicate stop orders."
            )
            # Do NOT proceed with TSL if cancellation failed, as the initial SL might still be active.
            return  # Stop the TSL process for this cycle
        except Exception as e:
            logger.error(
                Fore.RED
                + Style.BRIGHT
                + f"Unexpected error cancelling initial SL: {e}",
                exc_info=True,
            )
            logger.error("Aborting TSL placement due to unexpected cancellation error.")
            return  # Stop the TSL process

        # --- Place Trailing Stop Loss Order ---
        tsl_order_side = "sell" if position_side == "long" else "buy"
        # Use the current position quantity, formatted precisely
        tsl_qty_str = format_amount(symbol, position_qty)

        # Convert Decimal percentage (e.g., 0.5) to float for CCXT param if needed,
        # but check Bybit V5 API - it might take string percentage like "0.5".
        # CCXT standard param 'trailingPercent' usually expects the percentage value directly (e.g., 0.5 for 0.5%).
        trail_percent_value = float(
            CONFIG.trailing_stop_percent
        )  # Standard CCXT approach

        # Bybit V5 API Parameters for TSL (refer to official docs / CCXT Bybit overrides):
        # Might involve `trailingStop`, `activePrice`, `tpslMode`, `slTriggerBy` within params.
        # CCXT aims to unify this via `trailingPercent` and potentially `triggerPrice` for activation.
        tsl_params = {
            "reduceOnly": True,
            "triggerBy": CONFIG.tsl_trigger_by,  # Use configured trigger type for the trail itself
            "trailingPercent": trail_percent_value,  # CCXT standard parameter
            # 'activationPrice': format_price(symbol, current_price) # Optional: Price at which the trail *starts*.
            # Bybit V5 might use 'activePrice'. CCXT might map 'triggerPrice' to this?
            # If not provided, trail might start immediately based on current price vs trail offset.
            # Let's omit activationPrice for now, assuming immediate trail start is okay.
            # Bybit specific params if CCXT unification isn't complete:
            # 'tpslMode': 'Full',
            # 'trailingStop': str(CONFIG.trailing_stop_percent / 100), # e.g., '0.005' for 0.5%? Check Bybit docs carefully. CCXT 'trailingPercent' should handle this.
            # 'activePrice': format_price(symbol, current_price),
            # 'positionIdx': 0 # Assuming one-way mode
        }
        logger.trade(
            f"Placing TSL order: Side={tsl_order_side}, Qty={tsl_qty_str}, Trail%={trail_percent_value}, TriggerBy={CONFIG.tsl_trigger_by}"
        )
        logger.debug(f"TSL Params: {tsl_params}")

        try:
            # Use create_order with a specific trailing stop type if available and unified by CCXT for Bybit.
            # Check `exchange.has['createTrailingStopMarketOrder']`.
            # If not available or reliable, using a standard type ('stop_market' or 'market') with trailing params might work if the exchange supports it via params.
            # However, Bybit V5 likely has a dedicated TSL order type or mechanism. CCXT should ideally handle this.
            # Let's try a standard type with params first, as specific types can be less common in CCXT unification.
            # We might need 'stop_market' type with trailing params, or just 'market' type if the trail modifies an existing position attribute on Bybit.
            # Let's *assume* CCXT maps `create_order` with `trailingPercent` correctly for Bybit.
            # We might need to use `exchange.private_post_v5_order_create(params)` if CCXT unification fails.

            # Attempt 1: Using a potentially unified type if available (less likely?)
            # tsl_order_type = 'trailing_stop_market' # Check exchange.has first
            # if not exchange.has.get('createTrailingStopMarketOrder'):
            #    logger.debug("Exchange does not explicitly support 'createTrailingStopMarketOrder', trying standard order with params.")
            #    tsl_order_type = 'market' # Or 'stop_market'? Need to know how Bybit handles TSL orders via API.
            # Let's assume 'market' type with 'trailingPercent' param works for modifying position's TSL setting on Bybit V5 (needs verification).
            # Or perhaps it's better to use `exchange.edit_position` or similar if available?

            # Safer bet: Use `create_order` but expect it might fail if Bybit needs a specific endpoint/params not fully unified.
            # Bybit's `POST /v5/order/create` with `tpslMode="Partial"` and `trailingStop` might be the direct way.
            # Let's try the unified `create_order` first. Assume it might place a separate closing order with trailing logic.
            tsl_order = exchange.create_order(
                symbol=symbol,
                # Type depends heavily on how CCXT/Bybit handle this. 'market' or 'stop_market' with trailing params?
                # Let's try 'market' assuming it modifies the position or creates a trailing close order.
                # If this fails, Bybit might need a specific endpoint call or different type.
                type="market",  # GUESS - This might be wrong for Bybit TSL placement via CCXT unified method.
                # If this fails, check Bybit V5 API docs for placing TSL. It might modify the position directly.
                # A common pattern is also 'stop_market' with trailing params. Let's favor that.
                # type='stop_market', # TRY THIS INSTEAD?
                side=tsl_order_side,
                amount=float(tsl_qty_str),  # CCXT expects float amount
                price=None,  # Market based trail
                params=tsl_params,
            )

            # --- Process TSL Order Response ---
            tsl_order_id = tsl_order.get("id")
            logger.debug(f"TSL order submission raw response: {tsl_order}")
            if not tsl_order_id:
                # If no ID, check if the response indicates success via info dict (maybe it modified the position?)
                # Bybit V5 might return success without a new order ID if it just sets TSL on the position.
                # Look for clues in the 'info' dictionary.
                if (
                    tsl_order.get("info", {}).get("retCode") == 0
                ):  # Bybit V5 success code is 0
                    logger.trade(
                        Fore.GREEN
                        + Style.BRIGHT
                        + f"Trailing Stop Loss likely activated for {position_side.upper()} position (modified position). Trail: {trail_percent_value}%"
                    )
                    # Update tracker - set TSL active marker (use a placeholder ID or flag)
                    order_tracker[position_side]["tsl_id"] = (
                        f"ACTIVE_{position_side.upper()}"  # Mark TSL as active
                    )
                    order_tracker[position_side]["sl_id"] = (
                        None  # Ensure initial SL ID is cleared
                    )
                    termux_notify(
                        "TSL Activated", f"{position_side.upper()} {symbol} TSL active."
                    )
                    return  # Success
                else:
                    error_msg = tsl_order.get("info", {}).get(
                        "retMsg", "Unknown reason."
                    )
                    raise ccxt.ExchangeError(
                        f"Trailing stop order placement failed. Exchange message: {error_msg}"
                    )

            # If we got an order ID, it means a separate TSL order was created.
            order_tracker[position_side]["tsl_id"] = tsl_order_id
            order_tracker[position_side]["sl_id"] = (
                None  # Ensure initial SL ID is cleared (should be already)
            )
            logger.trade(
                Fore.GREEN
                + Style.BRIGHT
                + f"Trailing Stop Loss order placed for {position_side.upper()}: ID {tsl_order_id}, Trail: {trail_percent_value}%"
            )
            termux_notify(
                "TSL Activated",
                f"{position_side.upper()} {symbol} TSL active (ID: ...{tsl_order_id[-6:]}).",
            )

        except (ccxt.ExchangeError, ccxt.InvalidOrder) as e:
            # TSL placement failed. Initial SL was already cancelled. Position is UNPROTECTED.
            logger.error(
                Fore.RED
                + Style.BRIGHT
                + f"CRITICAL: Failed to place TSL order after cancelling initial SL: {e}"
            )
            logger.warning(
                Fore.YELLOW
                + Style.BRIGHT
                + "Position is UNPROTECTED. MANUAL INTERVENTION STRONGLY ADVISED."
            )
            # Reset local tracker as TSL failed
            order_tracker[position_side]["tsl_id"] = None
            # CRITICAL: At this point, the initial SL is cancelled, and TSL failed.
            # Consider placing a new *regular* stop loss as a fallback? This adds complexity.
            # For now, just log the critical situation.
            termux_notify("TSL FAILED!", f"{symbol} POS UNPROTECTED! Check bot.")
        except Exception as e:
            logger.error(
                Fore.RED + Style.BRIGHT + f"Unexpected error placing TSL: {e}",
                exc_info=True,
            )
            logger.warning(
                Fore.YELLOW
                + Style.BRIGHT
                + "Position might be UNPROTECTED after unexpected TSL placement error. MANUAL INTERVENTION ADVISED."
            )
            order_tracker[position_side]["tsl_id"] = None
            termux_notify("TSL FAILED!", f"{symbol} POS UNPROTECTED! Check bot.")

    else:
        # Profit threshold not met
        logger.debug(
            f"{position_side.upper()} profit ({profit:.4f}) has not crossed TSL activation threshold ({activation_threshold_points:.4f}). Keeping initial SL (ID: {initial_sl_id})."
        )


def print_status_panel(
    cycle: int,
    timestamp: pd.Timestamp,
    price: Decimal,
    indicators: dict[str, Decimal],
    positions: dict[str, dict[str, Any]],
    equity: Decimal | None,
    signals: dict[str, bool],
    order_tracker_state: dict[
        str, dict[str, str | None]
    ],  # Pass tracker state explicitly
) -> None:
    """Displays the current state using a mystical status panel with Decimal precision."""
    Fore.MAGENTA + Style.BRIGHT

    # Market & Indicators
    price_str = f"{price:.4f}" if price is not None else "N/A"
    atr_str = (
        f"{indicators.get('atr', Decimal(0)):.6f}"
        if indicators and indicators.get("atr") is not None
        else "N/A"
    )
    trend_ema = indicators.get("trend_ema", Decimal(0)) if indicators else Decimal(0)
    trend_ema_str = (
        f"{trend_ema:.4f}"
        if indicators and indicators.get("trend_ema") is not None
        else "N/A"
    )
    price_color = Fore.WHITE
    trend_desc = ""
    if price is not None and indicators and indicators.get("trend_ema") is not None:
        if price > trend_ema:
            price_color = Fore.GREEN
            trend_desc = f"{price_color}(Above Trend)"
        elif price < trend_ema:
            price_color = Fore.RED
            trend_desc = f"{price_color}(Below Trend)"
        else:
            price_color = Fore.YELLOW
            trend_desc = f"{price_color}(At Trend)"

    stoch_k = indicators.get("stoch_k", Decimal(50)) if indicators else Decimal(50)
    stoch_d = indicators.get("stoch_d", Decimal(50)) if indicators else Decimal(50)
    stoch_k_str = (
        f"{stoch_k:.2f}"
        if indicators and indicators.get("stoch_k") is not None
        else "N/A"
    )
    stoch_d_str = (
        f"{stoch_d:.2f}"
        if indicators and indicators.get("stoch_d") is not None
        else "N/A"
    )
    stoch_color = Fore.YELLOW
    stoch_desc = ""
    if indicators and indicators.get("stoch_k") is not None:
        if stoch_k < Decimal(25):
            stoch_color = Fore.GREEN
            stoch_desc = f"{stoch_color}Oversold"
        elif stoch_k > Decimal(75):
            stoch_color = Fore.RED
            stoch_desc = f"{stoch_color}Overbought"
        else:
            stoch_color = Fore.YELLOW

    fast_ema = indicators.get("fast_ema", Decimal(0)) if indicators else Decimal(0)
    slow_ema = indicators.get("slow_ema", Decimal(0)) if indicators else Decimal(0)
    fast_ema_str = (
        f"{fast_ema:.4f}"
        if indicators and indicators.get("fast_ema") is not None
        else "N/A"
    )
    slow_ema_str = (
        f"{slow_ema:.4f}"
        if indicators and indicators.get("slow_ema") is not None
        else "N/A"
    )
    ema_cross_color = Fore.WHITE
    ema_desc = ""
    if (
        indicators
        and indicators.get("fast_ema") is not None
        and indicators.get("slow_ema") is not None
    ):
        if fast_ema > slow_ema:
            ema_cross_color = Fore.GREEN
            ema_desc = f"{ema_cross_color}Bullish Cross"
        elif fast_ema < slow_ema:
            ema_cross_color = Fore.RED
            ema_desc = f"{ema_cross_color}Bearish Cross"
        else:
            ema_cross_color = Fore.YELLOW

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

    # Positions & Orders
    long_pos = (
        positions.get("long", {"qty": Decimal("0.0"), "entry_price": Decimal("0.0")})
        if positions
        else {"qty": Decimal("0.0"), "entry_price": Decimal("0.0")}
    )
    short_pos = (
        positions.get("short", {"qty": Decimal("0.0"), "entry_price": Decimal("0.0")})
        if positions
        else {"qty": Decimal("0.0"), "entry_price": Decimal("0.0")}
    )

    # Use the passed tracker state snapshot
    long_sl_id = order_tracker_state["long"]["sl_id"]
    long_tsl_id = order_tracker_state["long"]["tsl_id"]
    short_sl_id = order_tracker_state["short"]["sl_id"]
    short_tsl_id = order_tracker_state["short"]["tsl_id"]

    # Determine SL/TSL status strings
    long_stop_status = Fore.RED + "None"
    if long_tsl_id:
        # Check if it's the placeholder ID or a real one
        if "ACTIVE" in long_tsl_id:
            long_stop_status = f"{Fore.GREEN}TSL Active (Pos Mod)"
        else:
            long_stop_status = f"{Fore.GREEN}TSL Active (ID: ...{long_tsl_id[-6:]})"
    elif long_sl_id:
        long_stop_status = f"{Fore.YELLOW}SL Active (ID: ...{long_sl_id[-6:]})"

    short_stop_status = Fore.RED + "None"
    if short_tsl_id:
        if "ACTIVE" in short_tsl_id:
            short_stop_status = f"{Fore.GREEN}TSL Active (Pos Mod)"
        else:
            short_stop_status = f"{Fore.GREEN}TSL Active (ID: ...{short_tsl_id[-6:]})"
    elif short_sl_id:
        short_stop_status = f"{Fore.YELLOW}SL Active (ID: ...{short_sl_id[-6:]})"

    long_qty_str = (
        f"{long_pos['qty']:.8f}".rstrip("0").rstrip(".")
        if long_pos["qty"] != Decimal(0)
        else "0"
    )
    short_qty_str = (
        f"{short_pos['qty']:.8f}".rstrip("0").rstrip(".")
        if short_pos["qty"] != Decimal(0)
        else "0"
    )
    long_entry_str = (
        f"{long_pos['entry_price']:.4f}"
        if long_pos["entry_price"] != Decimal(0)
        else "-"
    )
    short_entry_str = (
        f"{short_pos['entry_price']:.4f}"
        if short_pos["entry_price"] != Decimal(0)
        else "-"
    )

    [
        [Fore.CYAN + "Position", Fore.GREEN + "LONG", Fore.RED + "SHORT"],
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
        [Fore.CYAN + "Active Stop", long_stop_status, short_stop_status],
    ]

    # Signals
    Fore.GREEN if signals.get("long", False) else Fore.WHITE
    Fore.RED if signals.get("short", False) else Fore.WHITE


def generate_signals(
    indicators: dict[str, Decimal], current_price: Decimal
) -> dict[str, bool]:
    """Generates trading signals based on indicator conditions, using Decimal."""
    long_signal = False
    short_signal = False

    if not indicators:
        logger.warning("Cannot generate signals: indicators are missing.")
        return {"long": False, "short": False}
    if current_price is None or current_price <= Decimal(0):
        logger.warning("Cannot generate signals: current price is missing or invalid.")
        return {"long": False, "short": False}

    try:
        # Use .get with default Decimal values for safety
        k = indicators.get("stoch_k", Decimal(50))
        indicators.get("stoch_d", Decimal(50))
        fast_ema = indicators.get("fast_ema", Decimal(0))
        slow_ema = indicators.get("slow_ema", Decimal(0))
        trend_ema = indicators.get("trend_ema", Decimal(0))
        # confirm_ema = indicators.get('confirm_ema', Decimal(0)) # EMA(5)

        # Define conditions using Decimal comparisons
        # Add small tolerance for EMA crosses to avoid flickering? Maybe not needed with Decimal.
        ema_bullish_cross = fast_ema > slow_ema
        ema_bearish_cross = fast_ema < slow_ema
        price_above_trend = current_price > trend_ema
        price_below_trend = current_price < trend_ema

        # Stochastic conditions (more robust: check crossover within zones)
        # stoch_oversold_area = k < Decimal(25) and d < Decimal(25)
        # stoch_overbought_area = k > Decimal(75) and d > Decimal(75)
        # Stoch K crossing *above* D in oversold area
        # stoch_bullish_cross = stoch_oversold_area and k > d # Check previous values needed for true cross detection
        # Stoch K crossing *below* D in overbought area
        # stoch_bearish_cross = stoch_overbought_area and k < d # Check previous values needed for true cross detection

        # Simpler Stochastic check (K level only)
        stoch_oversold = k < Decimal(25)
        stoch_overbought = k > Decimal(75)

        # Confirmation EMA (e.g., EMA 5) condition - price should be above/below it
        # price_above_confirm = current_price > confirm_ema
        # price_below_confirm = current_price < confirm_ema

        # --- Signal Logic ---
        signal_reason = "No signal"

        # Long Signal: Bullish EMA cross AND Stoch Oversold (AND Price > Confirm EMA?)
        if ema_bullish_cross and stoch_oversold:  # and price_above_confirm:
            if CONFIG.trade_only_with_trend:
                if price_above_trend:
                    long_signal = True
                    signal_reason = (
                        "Long: EMA Cross Bullish, Stoch Oversold, Price > Trend EMA"
                    )
                else:
                    signal_reason = (
                        "Long Signal Blocked: Price Below Trend EMA (Trend Filter ON)"
                    )
            else:  # Trend filter off
                long_signal = True
                signal_reason = (
                    "Long: EMA Cross Bullish, Stoch Oversold (Trend Filter OFF)"
                )

        # Short Signal: Bearish EMA cross AND Stoch Overbought (AND Price < Confirm EMA?)
        elif ema_bearish_cross and stoch_overbought:  # and price_below_confirm:
            if CONFIG.trade_only_with_trend:
                if price_below_trend:
                    short_signal = True
                    signal_reason = (
                        "Short: EMA Cross Bearish, Stoch Overbought, Price < Trend EMA"
                    )
                else:
                    signal_reason = (
                        "Short Signal Blocked: Price Above Trend EMA (Trend Filter ON)"
                    )
            else:  # Trend filter off
                short_signal = True
                signal_reason = (
                    "Short: EMA Cross Bearish, Stoch Overbought (Trend Filter OFF)"
                )

        if long_signal or short_signal:
            logger.info(f"Signal Generated: {signal_reason}")
        else:
            logger.debug(f"Signal Check: {signal_reason}")

    except Exception as e:
        logger.error(f"{Fore.RED}Error generating signals: {e}", exc_info=True)
        return {"long": False, "short": False}

    return {"long": long_signal, "short": short_signal}


def trading_spell_cycle(cycle_count: int) -> None:
    """Executes one cycle of the trading spell with enhanced precision and logic."""
    logger.info(Fore.MAGENTA + Style.BRIGHT + f"\n--- Starting Cycle {cycle_count} ---")
    start_time = time.time()

    # 1. Fetch Market Data
    df = fetch_market_data(CONFIG.symbol, CONFIG.interval, CONFIG.ohlcv_limit)
    if df is None or df.empty:
        logger.error(Fore.RED + "Halting cycle: Market data fetch failed.")
        return  # Skip cycle if data is unavailable

    # Get current price and timestamp from the latest candle
    current_price: Decimal | None = None
    last_timestamp: pd.Timestamp | None = None
    try:
        # Ensure index is sorted if not already done in fetch
        # df.sort_index(inplace=True)
        last_candle = df.iloc[-1]
        current_price_float = last_candle["close"]
        current_price = Decimal(str(current_price_float))
        last_timestamp = df.index[-1]
        logger.debug(
            f"Latest candle: Time={last_timestamp}, Close={current_price:.4f}, High={last_candle['high']:.4f}, Low={last_candle['low']:.4f}, Open={last_candle['open']:.4f}"
        )
        # Check for stale data (e.g., if timestamp is too old)
        time_diff = pd.Timestamp.utcnow().tz_localize(None) - last_timestamp
        if time_diff > pd.Timedelta(minutes=5):  # Allow some lag, adjust as needed
            logger.warning(
                Fore.YELLOW
                + f"Market data seems stale. Last candle timestamp: {last_timestamp} ({time_diff} ago)."
            )

    except (IndexError, KeyError) as e:
        logger.error(
            Fore.RED + f"Failed to get current price/timestamp from DataFrame: {e}"
        )
        return  # Skip cycle if data is incomplete
    except (InvalidOperation, TypeError) as e:
        logger.error(
            Fore.RED + f"Error converting current price to Decimal: {e}", exc_info=True
        )
        return  # Skip cycle

    # 2. Calculate Indicators (returns Decimals)
    indicators = calculate_indicators(df)
    if indicators is None:
        logger.error(Fore.RED + "Halting cycle: Indicator calculation failed.")
        return  # Skip cycle if indicators fail
    current_atr = indicators.get("atr")  # Keep as Decimal

    # 3. Get Current State (Balance & Positions as Decimals)
    # Fetch balance first to know available capital
    free_balance, current_equity = get_balance("USDT")  # Assuming USDT quote
    if current_equity is None:
        # Allow proceeding without equity if balance fetch fails, but log warning
        logger.warning(
            Fore.YELLOW
            + "Failed to fetch current balance/equity. Status panel may be incomplete. Risk calculation might use stale data if entry occurs."
        )
        # Use a placeholder that indicates data is missing
        current_equity = Decimal("-1.0")  # Placeholder

    # Fetch positions
    positions = get_current_position(CONFIG.symbol)
    if positions is None:
        logger.error(Fore.RED + "Halting cycle: Failed to fetch current positions.")
        # Decide if we should halt or proceed cautiously assuming flat? Halting is safer.
        return

    # Ensure positions dict has expected structure and use Decimal
    long_pos = positions.get(
        "long", {"qty": Decimal("0.0"), "entry_price": Decimal("0.0")}
    )
    short_pos = positions.get(
        "short", {"qty": Decimal("0.0"), "entry_price": Decimal("0.0")}
    )

    # 4. Manage Trailing Stops (pass Decimals)
    # Check and manage TSL only if a position is actively held.
    if long_pos["qty"] >= CONFIG.position_qty_epsilon:
        logger.debug("Managing TSL for existing LONG position...")
        manage_trailing_stop(
            CONFIG.symbol,
            "long",
            long_pos["qty"],
            long_pos["entry_price"],
            current_price,
            current_atr,
        )
    elif short_pos["qty"] >= CONFIG.position_qty_epsilon:
        logger.debug("Managing TSL for existing SHORT position...")
        manage_trailing_stop(
            CONFIG.symbol,
            "short",
            short_pos["qty"],
            short_pos["entry_price"],
            current_price,
            current_atr,
        )
    else:
        # If flat, ensure trackers are clear (should be handled by TSL logic too, but belt-and-suspenders)
        if (
            order_tracker["long"]["sl_id"]
            or order_tracker["long"]["tsl_id"]
            or order_tracker["short"]["sl_id"]
            or order_tracker["short"]["tsl_id"]
        ):
            logger.debug("Position is flat, ensuring order trackers are cleared.")
            order_tracker["long"] = {"sl_id": None, "tsl_id": None}
            order_tracker["short"] = {"sl_id": None, "tsl_id": None}

    # 5. Generate Trading Signals (pass Decimals)
    signals = generate_signals(indicators, current_price)

    # --- Capture State Snapshot for Status Panel ---
    # This ensures the panel reflects the state *before* any potential trade execution in this cycle.
    order_tracker_snapshot = {
        "long": order_tracker["long"].copy(),
        "short": order_tracker["short"].copy(),
    }
    positions_snapshot = positions.copy()  # Copy the fetched positions for the panel

    # 6. Execute Trades based on Signals
    # Check if flat (neither long nor short position significantly open)
    is_flat = (
        long_pos["qty"] < CONFIG.position_qty_epsilon
        and short_pos["qty"] < CONFIG.position_qty_epsilon
    )
    logger.debug(
        f"Position Status: Flat = {is_flat} (Long Qty: {long_pos['qty']}, Short Qty: {short_pos['qty']})"
    )

    if is_flat:
        if signals.get("long"):
            logger.info(
                Fore.GREEN + Style.BRIGHT + "Long signal detected! Attempting entry..."
            )
            if place_risked_market_order(
                CONFIG.symbol, "buy", CONFIG.risk_percentage, current_atr
            ):
                pass
            else:
                logger.error(f"Long entry attempt failed for cycle {cycle_count}.")
                # Optional: Add a cooldown period after failed entry?

        elif signals.get("short"):
            logger.info(
                Fore.RED + Style.BRIGHT + "Short signal detected! Attempting entry."
            )
            if place_risked_market_order(
                CONFIG.symbol, "sell", CONFIG.risk_percentage, current_atr
            ):
                pass
            else:
                logger.error(f"Short entry attempt failed for cycle {cycle_count}.")
                # Optional: Add a cooldown period after failed entry?

        # If a trade was attempted (successfully or not), pause briefly to allow
        # exchange state/local trackers to potentially update before the *next* cycle's fetch.
        # This pause is within the current cycle's logic flow.
        if signals.get("long") or signals.get("short"):
            logger.info("Pausing briefly after trade attempt...")
            time.sleep(2)  # Small pause

    elif not is_flat:
        logger.info("Position already open, skipping new entry signals.")
        # Future enhancement: Add logic here to exit positions based on counter-signals,
        # profit targets, or other exit conditions if desired.
        # Example: if (long position exists and short signal) -> close long position
        # Example: if (short position exists and long signal) -> close short position
        # Ensure any exit logic also calls graceful_shutdown components or manages orders correctly.

    # 7. Display Status Panel
    # Use the state captured *before* trade execution for consistency in the panel for this cycle
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

    # Ensure exchange object is available and MARKET_INFO is loaded
    if (
        "exchange" not in globals()
        or not hasattr(exchange, "cancel_all_orders")
        or MARKET_INFO is None
    ):
        logger.error(
            Fore.RED
            + "Exchange object or Market Info not available. Cannot perform clean shutdown."
        )
        return

    symbol = CONFIG.symbol
    MARKET_INFO.get("id")  # Use market ID if needed by exchange for specific calls

    # 1. Cancel All Open Orders for the Symbol
    try:
        logger.info(Fore.CYAN + f"Dispelling all open orders for {symbol}...")
        # Fetch open orders first to log IDs before cancelling
        open_orders = []
        try:
            # Use fetch_with_retries for robustness
            open_orders = fetch_with_retries(exchange.fetch_open_orders, symbol)
        except Exception as fetch_err:
            logger.warning(
                Fore.YELLOW
                + f"Could not fetch open orders before cancelling: {fetch_err}. Proceeding with cancel all."
            )

        if open_orders:
            order_ids = [o.get("id", "N/A") for o in open_orders]
            logger.info(
                f"Found {len(open_orders)} open orders to cancel: {', '.join(order_ids)}"
            )
            # cancel_all_orders might need specific params for Bybit V5?
            # E.g., category=linear/inverse, settleCoin=USDT? Check CCXT Bybit overrides/docs.
            # Standard call:
            # response = exchange.cancel_all_orders(symbol)
            # More specific V5 call via CCXT might look like:
            # params = {'category': CONFIG.market_type} # 'linear' or 'inverse'
            # response = exchange.cancel_all_orders(symbol, params=params)
            # Let's try the simple one first, then add params if needed.
            try:
                # Use fetch_with_retries for cancellation robustness
                response = fetch_with_retries(exchange.cancel_all_orders, symbol)
                logger.info(
                    Fore.GREEN + f"Cancel command sent. Exchange response: {response}"
                )  # Response varies
            except Exception as cancel_err:
                logger.error(
                    Fore.RED
                    + f"Error sending cancel_all_orders command: {cancel_err}. Manual check required."
                )

        else:
            logger.info(Fore.GREEN + "No open orders found for the symbol to cancel.")

        # Clear local tracker regardless of API response, assuming intent was cancellation
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
    time.sleep(3)

    # 2. Close Any Open Positions
    try:
        logger.info(Fore.CYAN + "Checking for lingering positions to close...")
        # Fetch final position state using the dedicated function with retries
        positions = get_current_position(
            symbol
        )  # Already uses fetch_with_retries internally

        closed_count = 0
        if positions:
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
                        # Format quantity precisely for closure order
                        close_qty_str = format_amount(symbol, qty)
                        close_params = {"reduceOnly": True}
                        # Use fetch_with_retries for placing the closure order
                        close_order = fetch_with_retries(
                            exchange.create_market_order,
                            symbol=symbol,
                            side=close_side,
                            amount=float(close_qty_str),  # CCXT needs float
                            params=close_params,
                        )
                        if close_order and close_order.get("id"):
                            logger.trade(
                                Fore.GREEN
                                + f"Position closure order placed: ID {close_order.get('id')}"
                            )
                            closed_count += 1
                            # Add a small delay to allow closure order to process before final log
                            time.sleep(CONFIG.order_check_delay_seconds)
                        else:
                            logger.critical(
                                Fore.RED
                                + Style.BRIGHT
                                + f"FAILED TO PLACE closure order for {side} position ({qty}). Response: {close_order}. MANUAL INTERVENTION REQUIRED!"
                            )

                    except (ccxt.ExchangeError, ccxt.InvalidOrder) as e:
                        # Log critical error if closure fails
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

            if closed_count == 0:  # Check if any closures were attempted
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
        + "*** Pyrmethus Termux Trading Spell Activated (v2.1 Precision/Robust) ***"
    )
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

    if MARKET_INFO:  # Check if market info loaded successfully
        termux_notify("Bot Started", f"Monitoring {CONFIG.symbol} (v2.1)")
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
            + "Market info failed to load during initialization. Cannot start trading loop."
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
