# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Termux Trading Spell (v2 - Precision Enhanced)
# Conjures market insights and executes trades on Bybit Futures with refined precision.

import contextlib
import logging
import os
import subprocess
import sys
import time
from decimal import ROUND_DOWN, Decimal
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional, Tuple, Union

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
    init(autoreset=True)  # Initialize colorama for error messages only if needed here
    missing_pkg = e.name
    print(
        Fore.RED
        + Style.BRIGHT
        + f"Error: Missing required enchantment '{missing_pkg}'."
    )
    print(Fore.YELLOW + "Please conjure it using pip:")
    print(Fore.CYAN + f"  pip install {missing_pkg}")
    print(Fore.YELLOW + "Or install all common dependencies:")
    print(Fore.CYAN + "  pip install ccxt numpy pandas colorama python-dotenv tabulate")
    # Specific note for numpy on Termux if issues arise
    if missing_pkg == "numpy":
        print(
            Fore.YELLOW
            + "Note: If numpy fails to install on Termux, you might need additional dependencies:"
        )
        print(Fore.CYAN + "  pkg install build-essential python-numpy")
    sys.exit(1)

# Weave the Colorama magic into the terminal
init(autoreset=True)

# Set Decimal precision if needed (default is usually sufficient, 28 digits)
# getcontext().prec = 28

# --- Arcane Configuration ---

# Summon secrets from the .env scroll
load_dotenv()

# --- Ethereal Log Scribe Configuration ---
LOG_FILENAME = "pyrmethus_spellbook.log"
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
LOG_BACKUP_COUNT = 3

# Configure formatter
log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
color_log_formatter = logging.Formatter(
    Fore.CYAN
    + "%(asctime)s "
    + Style.BRIGHT
    + "[%(levelname)s] "
    + Style.RESET_ALL
    + Fore.WHITE
    + "%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set to DEBUG for more verbose output
logger.propagate = False  # Prevent duplicate logs if root logger is configured

# Remove existing handlers if any (e.g., during reloads)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Console Handler (with color)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(color_log_formatter)
logger.addHandler(stream_handler)

# File Handler (without color, rotating)
try:
    file_handler = RotatingFileHandler(
        LOG_FILENAME,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
except PermissionError:
    logger.warning(
        Fore.YELLOW
        + f"Permission denied writing log file '{LOG_FILENAME}'. Log file disabled."
    )
except Exception as e:
    logger.warning(
        Fore.YELLOW + f"Failed to initialize log file handler: {e}. Log file disabled."
    )


class TradingConfig:
    """Holds the sacred parameters of our spell, enhanced with precision awareness."""

    def __init__(self) -> None:
        self.symbol = self._get_env(
            "SYMBOL", "BTC/USDT:USDT", Fore.YELLOW
        )  # CCXT Unified Symbol
        self.market_type = self._get_env(
            "MARKET_TYPE", "linear", Fore.YELLOW, allowed_values=["linear", "inverse"]
        )
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
        # Validate trigger price types allowed by Bybit (check their API docs)
        allowed_trigger_prices = ["LastPrice", "MarkPrice", "IndexPrice"]
        self.sl_trigger_by = self._get_env(
            "SL_TRIGGER_BY",
            "LastPrice",
            Fore.YELLOW,
            allowed_values=allowed_trigger_prices,
        )
        self.tsl_trigger_by = self._get_env(
            "TSL_TRIGGER_BY",
            "LastPrice",
            Fore.YELLOW,
            allowed_values=allowed_trigger_prices,
        )

        # Threshold for considering position closed (as Decimal)
        # Adjust based on minimum contract size or typical dust amounts
        self.position_qty_epsilon = self._get_env(
            "POSITION_QTY_EPSILON", "0.000001", Fore.YELLOW, cast_type=Decimal
        )
        self.api_key = self._get_env(
            "BYBIT_API_KEY", None, Fore.YELLOW
        )  # Use Yellow for warning maybe?
        self.api_secret = self._get_env("BYBIT_API_SECRET", None, Fore.YELLOW)
        self.ohlcv_limit = self._get_env("OHLCV_LIMIT", 200, Fore.YELLOW, cast_type=int)
        self.loop_sleep_seconds = self._get_env(
            "LOOP_SLEEP_SECONDS", 15, Fore.YELLOW, cast_type=int
        )
        self.order_check_delay_seconds = self._get_env(
            "ORDER_CHECK_DELAY_SECONDS", 2, Fore.YELLOW, cast_type=int
        )
        self.order_check_timeout_seconds = self._get_env(
            "ORDER_CHECK_TIMEOUT_SECONDS", 10, Fore.YELLOW, cast_type=int
        )  # Max time to wait for order status check
        self.max_fetch_retries = self._get_env(
            "MAX_FETCH_RETRIES", 3, Fore.YELLOW, cast_type=int
        )
        self.trade_only_with_trend = self._get_env(
            "TRADE_ONLY_WITH_TREND", "True", Fore.YELLOW, cast_type=bool
        )  # Only trade in direction of trend_ema

        if not self.api_key or not self.api_secret:
            logger.critical(
                Fore.RED
                + Style.BRIGHT
                + "BYBIT_API_KEY or BYBIT_API_SECRET not found in .env scroll! Halting."
            )
            sys.exit(1)

        if self.risk_percentage <= 0 or self.risk_percentage >= 1:
            logger.warning(
                f"{Fore.YELLOW}RISK_PERCENTAGE ({self.risk_percentage}) seems unusual. Ensure it's a decimal (e.g., 0.01 for 1%)."
            )
        if (
            self.sl_atr_multiplier <= 0
            or self.tsl_activation_atr_multiplier <= 0
            or self.trailing_stop_percent <= 0
        ):
            logger.warning(
                f"{Fore.YELLOW}ATR/TSL multipliers and percentages should be positive."
            )

    def _get_env(
        self,
        key: str,
        default: Any,
        color: str,
        cast_type: type = str,
        allowed_values: Optional[list] = None,
    ) -> Any:
        value = os.getenv(key)
        is_secret = "SECRET" in key or "API_KEY" in key
        log_value = "****" if is_secret else value

        if value is None:
            value = default
            # Don't log warning if default is None (like for API keys where error is raised later)
            if default is not None:
                logger.warning(
                    f"{color}Using default value for {key}: {'****' if is_secret else default}"
                )
        else:
            logger.info(f"{color}Summoned {key}: {log_value}")

        # Attempt casting
        casted_value = None
        try:
            if value is None:
                casted_value = None
            elif cast_type == bool:
                casted_value = str(value).lower() in ["true", "1", "yes", "y"]
            elif cast_type == Decimal:
                # Ensure string conversion before Decimal for robustness
                casted_value = Decimal(str(value))
            else:
                casted_value = cast_type(value)
        except (ValueError, TypeError) as e:
            logger.error(
                f"{Fore.RED}Could not cast {key} ('{log_value}') to {cast_type.__name__}: {e}. Using default: {'****' if is_secret else default}"
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
            except (ValueError, TypeError):
                logger.critical(
                    f"{Fore.RED + Style.BRIGHT}Default value '{default}' for {key} also cannot be cast to {cast_type.__name__}. Halting."
                )
                sys.exit(1)

        # Validate against allowed values if provided
        if allowed_values is not None and casted_value not in allowed_values:
            logger.error(
                f"{Fore.RED}Invalid value '{casted_value}' for {key}. Allowed values: {allowed_values}. Using default: {default}"
            )
            # Return the default value if validation fails
            # Ensure the default itself is valid (it should be if configured correctly)
            if default in allowed_values:
                return default
            else:
                logger.critical(
                    f"{Fore.RED + Style.BRIGHT}Default value '{default}' for {key} is also not in allowed values {allowed_values}. Halting."
                )
                sys.exit(1)

        return casted_value


CONFIG = TradingConfig()
MARKET_INFO: Optional[Dict[str, Any]] = (
    None  # Global to store market details after connection
)

# --- Exchange Nexus Initialization ---
try:
    logger.info(Fore.BLUE + "Initializing connection to Bybit Nexus...")
    exchange = ccxt.bybit(
        {
            "apiKey": CONFIG.api_key,
            "secret": CONFIG.api_secret,
            "enableRateLimit": True,
            # Consider adding options for testnet if needed
            # 'options': {'defaultType': 'future', 'defaultSubType': CONFIG.market_type, 'testnet': True},
        }
    )
    # Set market type based on config (important for subsequent calls)
    exchange.options["defaultType"] = "future"  # Generic futures type
    exchange.options["defaultSubType"] = CONFIG.market_type  # 'linear' or 'inverse'

    exchange.load_markets()
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
        available_symbols = []
        quote_currency = CONFIG.symbol.split("/")[1].split(":")[0]  # e.g., USDT
        with contextlib.suppress(Exception):  # Avoid crashing if market data is weird
            available_symbols = [
                s
                for s in exchange.markets
                if exchange.markets[s].get("active")
                and exchange.markets[s].get("type") == "future"  # Ensure it's a future
                and exchange.markets[s].get("subType")
                == CONFIG.market_type  # Match linear/inverse
                and exchange.markets[s].get("quote")
                == quote_currency  # Match quote currency
            ][:15]  # Limit suggestions
        if available_symbols:
            logger.info(
                Fore.CYAN
                + f"Available active {CONFIG.market_type} symbols with {quote_currency} quote (sample): "
                + ", ".join(available_symbols)
            )
        else:
            logger.info(
                Fore.CYAN
                + f"Could not find similar active {CONFIG.market_type} symbols with {quote_currency} quote."
            )
        sys.exit(1)
    else:
        MARKET_INFO = exchange.market(CONFIG.symbol)
        logger.info(Fore.CYAN + f"Market spirit for {CONFIG.symbol} acknowledged.")
        # Log key precision and limits using Decimal where appropriate
        # Use .get for safety in case keys are missing
        price_prec_raw = MARKET_INFO.get("precision", {}).get("price")
        amount_prec_raw = MARKET_INFO.get("precision", {}).get("amount")
        min_amount_raw = MARKET_INFO.get("limits", {}).get("amount", {}).get("min")
        max_amount_raw = MARKET_INFO.get("limits", {}).get("amount", {}).get("max")
        contract_size_raw = MARKET_INFO.get(
            "contractSize", "1"
        )  # Default to '1' if not present

        logger.debug(
            f"Raw Market Precision: Price={price_prec_raw}, Amount={amount_prec_raw}"
        )
        logger.debug(
            f"Raw Market Limits: Min Amount={min_amount_raw}, Max Amount={max_amount_raw}"
        )
        logger.debug(f"Raw Contract Size: {contract_size_raw}")

        # Validate that we can convert these critical values to Decimal
        try:
            Decimal(str(price_prec_raw))
            Decimal(str(amount_prec_raw))
            if min_amount_raw is not None:
                Decimal(str(min_amount_raw))
            # Max amount can sometimes be None/null
            if max_amount_raw is not None:
                Decimal(str(max_amount_raw))
            Decimal(str(contract_size_raw))
            logger.info(
                "Successfully parsed market precision, limits, and contract size."
            )
        except (TypeError, ValueError, Exception) as e:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}Failed to parse critical market info (precision/limits/size) as numbers: {e}. Market Info: {MARKET_INFO}. Halting."
            )
            sys.exit(1)


except ccxt.AuthenticationError:
    logger.critical(
        Fore.RED
        + Style.BRIGHT
        + "Authentication failed! Check your API keys in the .env scroll. Halting."
    )
    sys.exit(1)
except ccxt.ExchangeNotAvailable:
    logger.critical(
        Fore.RED
        + Style.BRIGHT
        + "Exchange Nexus is currently unavailable (e.g., maintenance). Halting."
    )
    sys.exit(1)
except ccxt.NetworkError as e:
    logger.critical(
        Fore.RED
        + Style.BRIGHT
        + f"Network disturbance during Nexus initialization: {e}. Check connection. Halting."
    )
    sys.exit(1)
except ccxt.ExchangeError as e:
    logger.critical(
        Fore.RED
        + Style.BRIGHT
        + f"Exchange Nexus Error during initialization: {e}. Halting."
    )
    sys.exit(1)
except Exception as e:
    logger.critical(
        Fore.RED + Style.BRIGHT + f"Unexpected error during Nexus initialization: {e}",
        exc_info=True,
    )
    sys.exit(1)


# --- Global State Runes ---
# Tracks active SL/TSL order IDs associated with a potential long or short position
# This helps manage order cancellation/replacement logic
order_tracker: Dict[str, Dict[str, Optional[str]]] = {
    "long": {"sl_id": None, "tsl_id": None},
    "short": {"sl_id": None, "tsl_id": None},
}


# --- Termux Utility Spell ---
def termux_notify(title: str, content: str) -> None:
    """Sends a notification using Termux API (if available). Uses subprocess for safety."""
    # Check if termux-toast command exists and is executable
    try:
        result = subprocess.run(
            ["command", "-v", "termux-toast"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.debug(
                "termux-toast command not found or not executable. Skipping notification."
            )
            return
    except FileNotFoundError:
        logger.debug("'command' utility not found? Skipping Termux check.")
        return
    except Exception as e:
        logger.warning(Fore.YELLOW + f"Error checking for termux-toast: {e}")
        return

    try:
        # Use subprocess.run for safer command execution than os.system
        # Arguments are passed as a list to avoid shell injection issues.
        full_content = f"{title}: {content}"
        cmd = ["termux-toast", "-g", "middle", "-c", "green", full_content]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=5)
        logger.debug(f"Termux notification sent: {full_content}")
    except FileNotFoundError:
        # Should have been caught above, but double-check
        logger.warning(
            Fore.YELLOW + "termux-toast command not found when trying to notify."
        )
    except subprocess.TimeoutExpired:
        logger.warning(Fore.YELLOW + "Termux notification command timed out.")
    except subprocess.CalledProcessError as e:
        logger.warning(
            Fore.YELLOW + f"Termux notification command failed with error: {e.stderr}"
        )
    except Exception as e:
        logger.warning(Fore.YELLOW + f"Could not conjure Termux notification: {e}")


# --- Precision Casting Spells ---


def format_price(symbol: str, price: Union[float, Decimal]) -> str:
    """Formats price according to market precision rules using exchange methods."""
    if MARKET_INFO is None:
        logger.error(f"{Fore.RED}Market info not loaded, cannot format price.")
        return str(float(price))  # Fallback to simple float string

    try:
        # CCXT methods usually expect float input
        # price_to_precision typically handles rounding/truncation as per exchange rules
        return exchange.price_to_precision(symbol, float(price))
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting price {price} for {symbol}: {e}")
        return str(float(price))  # Fallback


def format_amount(symbol: str, amount: Union[float, Decimal]) -> str:
    """Formats amount according to market precision rules using exchange methods."""
    if MARKET_INFO is None:
        logger.error(f"{Fore.RED}Market info not loaded, cannot format amount.")
        return str(float(amount))  # Fallback

    try:
        # CCXT methods usually expect float input
        # amount_to_precision typically handles rounding/truncation as per exchange rules (often ROUND_DOWN/truncate)
        return exchange.amount_to_precision(symbol, float(amount))
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting amount {amount} for {symbol}: {e}")
        return str(float(amount))  # Fallback


# --- Core Spell Functions ---


def fetch_market_data(
    symbol: str, timeframe: str, limit: int, retries: int = CONFIG.max_fetch_retries
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data, handling transient errors with retries."""
    logger.info(
        Fore.CYAN
        + f"Channeling market whispers for {symbol} ({timeframe}, limit={limit})..."
    )
    for attempt in range(retries):
        try:
            # Check if exchange object is valid and has the method
            if not hasattr(exchange, "fetch_ohlcv"):
                logger.error(
                    Fore.RED
                    + "Exchange object not properly initialized or lacks fetch_ohlcv."
                )
                return None

            ohlcv: list[list[Union[int, float]]] = exchange.fetch_ohlcv(
                symbol, timeframe, limit=limit
            )

            if not ohlcv:
                # It's possible to get an empty list if the market has no recent trades
                logger.warning(
                    Fore.YELLOW
                    + f"Received empty OHLCV data (Attempt {attempt + 1}/{retries}). Market might be inactive?"
                )
                # Continue retrying in case it was a temporary glitch
                if attempt < retries - 1:
                    time.sleep(1 * (attempt + 1))  # Simple backoff
                    continue
                else:
                    logger.error(
                        Fore.RED
                        + f"Received empty OHLCV data after {retries} attempts. Cannot proceed."
                    )
                    return None

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            # Convert OHLCV columns to numeric, coercing errors (should not happen with valid API data)
            price_cols = ["open", "high", "low", "close"]
            for col in price_cols + ["volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Drop rows where essential price data is missing (NaN)
            df.dropna(subset=price_cols, inplace=True)

            if df.empty:
                logger.error(
                    Fore.RED
                    + "DataFrame is empty after processing OHLCV data (possibly all rows had NaNs)."
                )
                return None

            # Set timestamp as index after cleaning
            df = df.set_index("timestamp")

            logger.info(Fore.GREEN + f"Market whispers received ({len(df)} candles).")
            return df

        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
            logger.warning(
                Fore.YELLOW
                + f"Network disturbance fetching data (Attempt {attempt + 1}/{retries}): {e}. Retrying..."
            )
            if attempt < retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
            else:
                logger.error(
                    Fore.RED
                    + f"Failed to fetch market data after {retries} attempts due to network issues."
                )
                return None
        except ccxt.ExchangeError as e:
            # Includes things like rate limits hit despite enableRateLimit
            logger.error(Fore.RED + f"Exchange rejected data request: {e}")
            return None
        except Exception as e:
            logger.error(
                Fore.RED + f"Unexpected shadow encountered fetching data: {e}",
                exc_info=True,
            )
            return None
    return None  # Should be unreachable if retries > 0, but acts as safeguard


def calculate_indicators(df: pd.DataFrame) -> Optional[Dict[str, Decimal]]:
    """Calculate technical indicators, returning results as Decimals for precision."""
    logger.info(Fore.CYAN + "Weaving indicator patterns...")
    if df.empty:
        logger.error(Fore.RED + "Cannot calculate indicators on empty DataFrame.")
        return None
    if len(df) < 2:  # Need at least 2 data points for most indicators
        logger.error(Fore.RED + f"Not enough data ({len(df)}) to calculate indicators.")
        return None

    try:
        # Ensure data is float for calculations, convert to Decimal at the end
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        # --- EMAs ---
        # Using common periods, adjust=False is standard for TA libraries
        fast_ema_series = close.ewm(span=8, adjust=False).mean()
        slow_ema_series = close.ewm(span=12, adjust=False).mean()
        trend_ema_series = close.ewm(
            span=50, adjust=False
        ).mean()  # Using 50 for trend is common
        # confirm_ema_series = close.ewm(span=5, adjust=False).mean() # Removed confirm_ema for simplicity, can be added back

        # --- Stochastic Oscillator (%K, %D) ---
        stoch_period = 14  # Common period
        stoch_smooth_k = 3
        stoch_smooth_d = 3
        k_now, d_now = 50.0, 50.0  # Default neutral values

        if len(df) < stoch_period:
            logger.warning(
                f"Not enough data ({len(df)}) for Stochastic period {stoch_period}. Using default values."
            )
        else:
            low_min = low.rolling(window=stoch_period).min()
            high_max = high.rolling(window=stoch_period).max()
            # Add epsilon to prevent division by zero if high_max == low_min
            stoch_k_raw = 100 * (close - low_min) / (high_max - low_min + 1e-12)
            stoch_k = stoch_k_raw.rolling(window=stoch_smooth_k).mean()
            stoch_d = stoch_k.rolling(window=stoch_smooth_d).mean()

            # Get the latest non-NaN values
            k_now = stoch_k.iloc[-1] if not pd.isna(stoch_k.iloc[-1]) else 50.0
            d_now = stoch_d.iloc[-1] if not pd.isna(stoch_d.iloc[-1]) else 50.0

        # --- ATR (Average True Range) ---
        atr_period = 14  # Common period
        atr = 0.0  # Default to 0 if cannot calculate

        if (
            len(df) < atr_period + 1
        ):  # Need at least period+1 for shift() and initial calculation
            logger.warning(
                f"Not enough data ({len(df)}) for ATR period {atr_period}. Using default value 0."
            )
        else:
            # Calculate True Range (TR)
            high_low = high - low
            high_close_prev = abs(high - close.shift(1))
            low_close_prev = abs(low - close.shift(1))
            tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(
                axis=1
            )
            # Calculate ATR using Wilder's smoothing (equivalent to RMA or specific EWMA)
            # alpha = 1 / atr_period
            # atr_series = tr.ewm(alpha=alpha, adjust=False).mean() # Standard EMA
            # Wilder's Smoothing (RMA) is often preferred for ATR:
            atr_series = tr.ewm(
                alpha=1 / atr_period, adjust=False
            ).mean()  # Pandas EWMA with alpha=1/N approximates Wilder's for large N
            # Or implement Wilder's manually if needed: (more complex)
            # atr_series = tr.rolling(atr_period).mean() # Simple MA for first value
            # ... then apply smoothing formula ...

            atr = atr_series.iloc[-1]
            if pd.isna(atr):
                atr = 0.0  # Handle potential NaN at start

        logger.info(Fore.GREEN + "Indicator patterns woven successfully.")

        # --- Convert final indicator values to Decimal ---
        # Use '.quantize' for safety and consistent precision, especially for prices/ATR
        # Define a small Decimal quantum for price/ATR, larger for % indicators
        price_quantum = Decimal("0.00000001")  # 8 decimal places
        percent_quantum = Decimal("0.01")  # 2 decimal places

        def safe_decimal(value, quantum):
            if pd.isna(value):
                return Decimal("NaN")  # Or Decimal(0)? NaN might be clearer.
            # Convert float to string first to avoid potential binary representation issues
            return Decimal(str(value)).quantize(quantum, rounding=ROUND_DOWN)

        indicators_dec = {
            "fast_ema": safe_decimal(fast_ema_series.iloc[-1], price_quantum),
            "slow_ema": safe_decimal(slow_ema_series.iloc[-1], price_quantum),
            "trend_ema": safe_decimal(trend_ema_series.iloc[-1], price_quantum),
            "stoch_k": safe_decimal(k_now, percent_quantum),
            "stoch_d": safe_decimal(d_now, percent_quantum),
            "atr": safe_decimal(atr, price_quantum),
        }

        # Check for NaN results after conversion
        if any(val.is_nan() for val in indicators_dec.values()):
            logger.warning(
                f"{Fore.YELLOW}Some indicators resulted in NaN: {indicators_dec}"
            )
            # Decide how to handle NaNs - return None, or replace NaN with 0/default?
            # Replacing with 0 might lead to bad decisions. Returning None is safer.
            # logger.error(f"{Fore.RED}Indicator calculation resulted in NaN values. Cannot proceed.")
            # return None
            # Or replace NaNs with 0 (use cautiously):
            for k, v in indicators_dec.items():
                if v.is_nan():
                    logger.warning(f"Replacing NaN for indicator '{k}' with 0.")
                    indicators_dec[k] = Decimal("0.0")

        return indicators_dec

    except Exception as e:
        logger.error(
            Fore.RED + f"Failed to weave indicator patterns: {e}", exc_info=True
        )
        return None


def get_current_position(symbol: str) -> Optional[Dict[str, Dict[str, Decimal]]]:
    """Fetch current positions, returning quantities and entry prices as Decimals."""
    logger.info(Fore.CYAN + f"Consulting position spirits for {symbol}...")
    # Initialize with Decimal zero, representing a flat state
    pos_dict: Dict[str, Dict[str, Decimal]] = {
        "long": {"qty": Decimal("0.0"), "entry_price": Decimal("0.0")},
        "short": {"qty": Decimal("0.0"), "entry_price": Decimal("0.0")},
    }
    try:
        # fetch_positions can take symbols list. Use the specific symbol.
        # This is generally preferred over fetch_position (singular) which might be deprecated or less reliable.
        positions_raw = exchange.fetch_positions(symbols=[symbol])

        if not positions_raw:
            logger.info(
                Fore.BLUE + f"No open positions reported by exchange for {symbol}."
            )
            return pos_dict

        # Filter positions for the exact symbol requested (fetch_positions might return other related symbols sometimes)
        # And ensure 'info' field exists for detailed data if needed, and 'contracts' is present.
        symbol_positions = [
            p
            for p in positions_raw
            if p.get("symbol") == symbol
            and p.get("info")
            and p.get("contracts") is not None
        ]

        if not symbol_positions:
            logger.info(
                Fore.BLUE
                + f"No detailed position data found for {symbol} in fetched list."
            )
            return pos_dict

        # Bybit Unified Margin typically returns one entry per side (long/short) in non-hedge mode.
        # In hedge mode, it might return separate entries if positions were opened at different times.
        # This logic assumes we sum up quantities if multiple entries for the same side exist (hedge mode).
        active_positions_found = 0
        for pos in symbol_positions:
            side = pos.get("side")  # 'long' or 'short'
            # Use 'contracts' for quantity (amount of base currency or contracts depending on market)
            contracts_str = pos.get("contracts")
            # Use 'entryPrice'
            entry_price_str = pos.get("entryPrice")

            if side in pos_dict and contracts_str is not None:
                try:
                    contracts = Decimal(str(contracts_str))
                    # Use epsilon to check if effectively zero before adding
                    if contracts.copy_abs() < CONFIG.position_qty_epsilon:
                        logger.debug(
                            f"Ignoring effectively zero size position entry: Side={side}, Qty={contracts}"
                        )
                        continue

                    entry_price = (
                        Decimal(str(entry_price_str))
                        if entry_price_str is not None
                        else Decimal("0.0")
                    )

                    # Aggregate quantities and calculate weighted average entry price if multiple entries exist per side
                    current_qty = pos_dict[side]["qty"]
                    current_entry = pos_dict[side]["entry_price"]

                    if current_qty.is_zero():  # First entry for this side
                        pos_dict[side]["qty"] = contracts
                        pos_dict[side]["entry_price"] = entry_price
                    else:  # Aggregate (hedge mode or multiple fills) - Calculate weighted average entry
                        total_qty = current_qty + contracts
                        # Ensure total_qty is not zero before division
                        if not total_qty.is_zero():
                            weighted_entry = (
                                (current_qty * current_entry)
                                + (contracts * entry_price)
                            ) / total_qty
                            pos_dict[side]["entry_price"] = (
                                weighted_entry  # Keep precision
                            )
                        # else: handle case where adding position results in zero (e.g., closing part of hedge) - entry price becomes irrelevant
                        #    pos_dict[side]["entry_price"] = Decimal("0.0")
                        pos_dict[side]["qty"] = total_qty

                    # Re-check aggregated quantity against epsilon
                    if pos_dict[side]["qty"].copy_abs() >= CONFIG.position_qty_epsilon:
                        logger.info(
                            Fore.YELLOW
                            + f"Detected active {side} position: Qty={pos_dict[side]['qty']}, Avg Entry={pos_dict[side]['entry_price']:.6f}"
                        )
                        active_positions_found += 1
                    else:
                        # If aggregation resulted in zero, reset entry price
                        logger.info(
                            Fore.BLUE
                            + f"{side} position is now effectively zero after aggregation."
                        )
                        pos_dict[side]["qty"] = Decimal("0.0")
                        pos_dict[side]["entry_price"] = Decimal("0.0")

                except (TypeError, ValueError) as e:
                    logger.error(
                        f"{Fore.RED}Error parsing position data for side {side}: {e}. Data: {pos}"
                    )
                    # Continue to next position entry if possible
                    continue
            elif side not in pos_dict:
                logger.warning(
                    f"{Fore.YELLOW}Received position data with unexpected side '{side}'. Ignoring."
                )

        if active_positions_found == 0:
            logger.info(
                Fore.BLUE
                + f"No active non-zero positions found for {symbol} after processing."
            )

        logger.info(Fore.GREEN + "Position spirits consulted.")
        return pos_dict

    except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
        logger.warning(
            Fore.YELLOW
            + f"Network disturbance consulting position spirits: {e}. Cannot reliably get position."
        )
        return None  # Indicate failure
    except ccxt.AuthenticationError:
        logger.error(
            Fore.RED + "Authentication error fetching positions. Check API keys."
        )
        return None  # Indicate failure
    except ccxt.ExchangeError as e:
        logger.error(Fore.RED + f"Exchange rejected position spirit consultation: {e}")
        return None  # Indicate failure
    except Exception as e:
        logger.error(
            Fore.RED
            + f"Unexpected shadow encountered consulting position spirits: {e}",
            exc_info=True,
        )
        return None  # Indicate failure


def get_balance(currency: str = "USDT") -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """Fetches the free and total balance for a specific currency as Decimals."""
    logger.info(Fore.CYAN + f"Querying the Vault of {currency}...")
    try:
        # Use fetch_balance() - parameters might be needed for specific account types (e.g., unified)
        # params = {'accountType': 'UNIFIED'} # Example for Bybit Unified Margin, check CCXT docs
        balance = exchange.fetch_balance()  # params=params

        # Balance structure can vary; use .get generously
        free_balance_data = balance.get("free", {})
        total_balance_data = balance.get("total", {})

        if not isinstance(free_balance_data, dict) or not isinstance(
            total_balance_data, dict
        ):
            logger.error(
                f"{Fore.RED}Unexpected balance structure received. Cannot parse {currency} balance."
            )
            logger.debug(f"Balance data: {balance}")
            return None, None

        free_balance_str = free_balance_data.get(currency)
        total_balance_str = total_balance_data.get(currency)

        # Convert to Decimal, defaulting to 0 if None or parsing fails
        free_balance = Decimal("0.0")
        total_balance = Decimal("0.0")
        try:
            if free_balance_str is not None:
                free_balance = Decimal(str(free_balance_str))
        except Exception as e:
            logger.warning(
                f"{Fore.YELLOW}Could not parse free balance '{free_balance_str}' to Decimal: {e}. Using 0."
            )
        try:
            if total_balance_str is not None:
                total_balance = Decimal(str(total_balance_str))
        except Exception as e:
            logger.warning(
                f"{Fore.YELLOW}Could not parse total balance '{total_balance_str}' to Decimal: {e}. Using 0."
            )

        logger.info(
            Fore.GREEN
            + f"Vault contains {free_balance:.4f} free {currency} (Total: {total_balance:.4f})."
        )
        return free_balance, total_balance

    except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
        logger.warning(
            Fore.YELLOW
            + f"Network disturbance querying vault: {e}. Cannot assess risk capital."
        )
        return None, None
    except ccxt.AuthenticationError:
        logger.error(
            Fore.RED + "Authentication error fetching balance. Check API keys."
        )
        return None, None
    except ccxt.ExchangeError as e:
        logger.error(Fore.RED + f"Exchange rejected vault query: {e}")
        return None, None
    except Exception as e:
        logger.error(
            Fore.RED + f"Unexpected shadow encountered querying vault: {e}",
            exc_info=True,
        )
        return None, None


def check_order_status(
    order_id: str, symbol: str, timeout: int = CONFIG.order_check_timeout_seconds
) -> Optional[Dict[str, Any]]:
    """
    Checks order status with retries and timeout.
    Returns the order dict if found (regardless of status), or None if not found or error/timeout occurs.
    """
    logger.info(
        Fore.CYAN
        + f"Verifying status of order {order_id} for {symbol} (timeout: {timeout}s)..."
    )
    start_time = time.time()
    attempt = 0
    while time.time() - start_time < timeout:
        attempt += 1
        try:
            # fetch_order requires the order ID and symbol
            order_status = exchange.fetch_order(order_id, symbol)

            if order_status and isinstance(order_status, dict):
                status = order_status.get("status", "unknown")
                logger.info(
                    f"Order {order_id} status check attempt {attempt}: Status '{status}'"
                )
                # Return the full order dict as soon as it's successfully fetched
                return order_status
            else:
                # This case might indicate the order *was* found but the structure is empty/unexpected
                # Or potentially an issue with the ccxt method implementation for this exchange
                logger.warning(
                    f"fetch_order returned empty or unexpected structure for {order_id} (Attempt {attempt}). Retrying..."
                )

        except ccxt.OrderNotFound:
            # Order is definitively not found on the exchange.
            # This could mean it filled and was archived quickly, was cancelled, rejected, or never existed.
            logger.warning(
                Fore.YELLOW
                + f"Order {order_id} not found by exchange (Attempt {attempt})."
            )
            # Returning None indicates it's not fetchable / doesn't exist in an open/pending state.
            return None

        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
            logger.warning(
                f"Network issue checking order {order_id} (Attempt {attempt}): {e}. Retrying..."
            )
        except ccxt.AuthenticationError:
            logger.error(
                Fore.RED
                + f"Authentication error checking order {order_id}. Check API keys."
            )
            return None  # Fatal error for this operation
        except ccxt.ExchangeError as e:
            # Exchange error likely means the request failed, not necessarily that the order failed.
            logger.error(
                f"Exchange error checking order {order_id} (Attempt {attempt}): {e}. Retrying..."
            )
            # Consider if specific exchange errors should terminate the check (e.g., invalid order ID format)
        except Exception as e:
            logger.error(
                f"Unexpected error checking order {order_id} (Attempt {attempt}): {e}",
                exc_info=True,
            )
            # Treat unexpected errors cautiously, maybe retry a few times but then fail.

        # Wait before retrying
        check_interval = 1.5  # seconds, slightly longer interval
        # Ensure we don't sleep past the timeout
        if time.time() - start_time + check_interval < timeout:
            time.sleep(check_interval)
        else:
            logger.debug(f"Approaching timeout for order {order_id}, stopping checks.")
            break  # Exit loop if next sleep would exceed timeout

    logger.error(
        Fore.RED
        + f"Timed out or failed to determine status for order {order_id} after {timeout} seconds and {attempt} attempts."
    )
    return None  # Indicate timeout or persistent failure to get status


def place_risked_market_order(
    symbol: str, side: str, risk_percentage: Decimal, atr: Decimal
) -> bool:
    """
    Places a market order with calculated size based on risk and ATR-based stop-loss.
    Uses Decimal precision for calculations. Returns True if entry and initial SL placement succeed.
    """
    logger.info(
        Fore.BLUE
        + Style.BRIGHT
        + f"=== Preparing {side.upper()} Market Incantation for {symbol} ==="
    )

    if MARKET_INFO is None:
        logger.error(Fore.RED + "Market info not available. Cannot place order.")
        return False

    # 1. Get Required Info (Balance, Price, Market Rules)
    quote_currency = MARKET_INFO.get("quote", "USDT")  # Assume USDT if not specified
    free_balance, _ = get_balance(quote_currency)
    if free_balance is None or free_balance <= Decimal("0"):
        logger.error(
            Fore.RED
            + f"Cannot place order: Invalid or zero available {quote_currency} balance."
        )
        return False

    if atr is None or atr.is_nan() or atr <= Decimal("0"):
        logger.error(Fore.RED + f"Cannot place order: Invalid ATR value ({atr}).")
        return False

    try:
        ticker = exchange.fetch_ticker(symbol)
        price_str = ticker.get("last")  # Use 'last' price for calculations
        if price_str is None:
            logger.error(Fore.RED + "Cannot fetch current price for sizing.")
            return False
        current_price = Decimal(str(price_str))
        logger.debug(f"Current Price ({symbol}): {current_price}")

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger.error(f"{Fore.RED}Failed to fetch ticker price: {e}")
        return False
    except Exception as e:
        logger.error(f"{Fore.RED}Unexpected error fetching ticker: {e}", exc_info=True)
        return False

    contract_size = Decimal(str(MARKET_INFO.get("contractSize", "1")))
    min_qty_str = MARKET_INFO.get("limits", {}).get("amount", {}).get("min")
    max_qty_str = MARKET_INFO.get("limits", {}).get("amount", {}).get("max")
    min_qty = Decimal(str(min_qty_str)) if min_qty_str is not None else None
    max_qty = Decimal(str(max_qty_str)) if max_qty_str is not None else None

    # 2. Calculate Stop Loss Price
    sl_distance_points = CONFIG.sl_atr_multiplier * atr
    if side == "buy":
        sl_price_raw = current_price - sl_distance_points
    else:  # side == "sell"
        sl_price_raw = current_price + sl_distance_points

    # Format SL price according to market precision *before* using it in calculations
    sl_price_formatted_str = format_price(symbol, sl_price_raw)
    sl_price = Decimal(sl_price_formatted_str)  # Use the formatted price as Decimal
    logger.debug(
        f"ATR: {atr:.6f}, SL Multiplier: {CONFIG.sl_atr_multiplier}, SL Distance Pts: {sl_distance_points:.6f}"
    )
    logger.debug(f"Raw SL Price: {sl_price_raw:.6f}, Formatted SL Price: {sl_price}")

    # Ensure SL is not triggered immediately (e.g., due to large spread or ATR volatility)
    # Allow for a small buffer (e.g., a fraction of a tick size) if needed, but simple check first.
    if side == "buy" and sl_price >= current_price:
        logger.error(
            Fore.RED
            + f"Calculated SL price ({sl_price}) is >= current price ({current_price}). Risk too high or ATR too large? Aborting."
        )
        return False
    if side == "sell" and sl_price <= current_price:
        logger.error(
            Fore.RED
            + f"Calculated SL price ({sl_price}) is <= current price ({current_price}). Risk too high or ATR too large? Aborting."
        )
        return False

    # 3. Calculate Position Size based on Risk
    risk_amount_quote = (
        free_balance * risk_percentage
    )  # Risk amount in quote currency (e.g., USDT)
    # Stop distance in quote currency (absolute difference between entry and SL price)
    stop_distance_quote = abs(
        current_price - sl_price
    )  # Use current price as estimated entry

    if stop_distance_quote <= Decimal("0"):
        logger.error(
            Fore.RED
            + f"Stop distance in quote currency is zero or negative ({stop_distance_quote}). Check ATR, multiplier, or market precision. Cannot calculate size."
        )
        return False

    qty_raw = Decimal("0")
    try:
        if CONFIG.market_type == "linear":
            # Qty (in contracts/base) = Risk Amount (Quote) / Stop Distance (Quote)
            # If contract size is not 1 base unit, adjust.
            # Value of 1 contract = contract_size * price (approx)
            # Risk per contract = Stop Distance (Quote) * contract_size
            # Qty (contracts) = Risk Amount (Quote) / (Stop Distance (Quote) * contract_size)
            if contract_size.is_zero():
                raise ValueError("Contract size cannot be zero.")
            qty_raw = risk_amount_quote / (stop_distance_quote * contract_size)
            logger.debug(
                f"Linear Sizing: Risk={risk_amount_quote:.4f}, StopDist={stop_distance_quote:.4f}, ContractSize={contract_size}"
            )

        elif CONFIG.market_type == "inverse":
            # Qty (in contracts) = Risk Amount (Base) / Stop Distance (Base)
            # Risk Amount (Base) = Risk Amount (Quote) / Price
            # Stop Distance (Base) = Stop Distance (Quote) / Price
            # Qty (Base) = Risk Amount (Base) / Stop Distance (Base) = Risk Amount (Quote) / Stop Distance (Quote)
            # Qty (Contracts) = Qty (Base) / Contract Size (Base)
            # For $1 inverse contracts (common): Contract Size (Base) = 1 / Price
            # Qty (Contracts) = Qty (Base) * Price = (Risk Amount (Quote) / Stop Distance (Quote)) * Price
            if current_price.is_zero():
                raise ValueError(
                    "Current price cannot be zero for inverse calculation."
                )
            if stop_distance_quote.is_zero():
                raise ValueError(
                    "Stop distance cannot be zero."
                )  # Already checked above

            # This calculation assumes the contract value is fixed in quote currency (e.g., $1 per contract)
            # Check MARKET_INFO['contractSize'] for inverse markets. If it's '1', it usually means $1.
            if contract_size == Decimal("1"):  # Assuming $1 contract size for inverse
                qty_raw = (risk_amount_quote / stop_distance_quote) * current_price
                logger.debug(
                    f"Inverse Sizing ($1 Contract): Risk={risk_amount_quote:.4f}, StopDist={stop_distance_quote:.4f}, Price={current_price}"
                )
            else:
                # If contract size for inverse is not 1 (e.g., represents base currency amount), the linear calc might be closer? Needs verification.
                logger.warning(
                    f"{Fore.YELLOW}Inverse contract size is not 1 ({contract_size}). Sizing calculation might need adjustment based on exact contract specs. Using simplified formula."
                )
                # Fallback to a simplified approach - treat as linear for now, but this is likely incorrect.
                # Qty (contracts) = Risk Amount (Quote) / (Stop Distance (Quote) * contract_size * price) ? Highly speculative.
                # Safest is to use the $1 contract formula and log warning, or require explicit handling.
                qty_raw = (
                    risk_amount_quote / stop_distance_quote
                ) * current_price  # Stick to $1 assumption with warning.

        else:
            logger.error(f"Unsupported market type for sizing: {CONFIG.market_type}")
            return False

    except (ValueError, ZeroDivisionError) as e:
        logger.error(
            f"{Fore.RED}Error during quantity calculation: {e}. Check inputs (price, stop distance, contract size)."
        )
        return False

    # 4. Format and Validate Quantity
    qty_formatted_str = format_amount(symbol, qty_raw)
    qty = Decimal(qty_formatted_str)
    logger.debug(f"Raw Qty: {qty_raw:.8f}, Formatted Qty: {qty}")

    if qty.is_zero() or qty < CONFIG.position_qty_epsilon:
        logger.error(
            Fore.RED
            + f"Calculated quantity ({qty}) is zero or too small after precision formatting. Risk amount, price movement, or ATR might be too small."
        )
        return False
    if min_qty is not None and qty < min_qty:
        logger.error(
            Fore.RED
            + f"Calculated quantity {qty} is below minimum {min_qty}. Cannot place order. Increase risk or adjust strategy."
        )
        # Optionally: Could increase size to min_qty if risk allows, but safer to abort.
        return False
    if max_qty is not None and qty > max_qty:
        logger.warning(
            Fore.YELLOW
            + f"Calculated quantity {qty} exceeds maximum {max_qty}. Capping order size to {max_qty}."
        )
        qty = max_qty  # Use the Decimal max_qty
        # Re-format capped amount potentially needed if max_qty wasn't already precise
        qty_formatted_str = format_amount(symbol, qty)
        qty = Decimal(qty_formatted_str)
        if qty < min_qty:  # Check again after capping and reformatting
            logger.error(
                Fore.RED
                + f"Capped quantity {qty} is now below minimum {min_qty}. Aborting."
            )
            return False

    logger.info(
        Fore.YELLOW
        + f"Calculated Order: Side={side.upper()}, Qty={qty}, Entry≈{current_price:.4f}, SL={sl_price:.4f}"
    )

    # 5. Place Market Order
    order_id = None
    order = None
    try:
        logger.info(
            Fore.CYAN + f"Submitting {side.upper()} market order for {qty} {symbol}..."
        )
        # CCXT expects float amount for create_market_order
        order = exchange.create_market_order(
            symbol, side, float(qty), params={}
        )  # No extra params needed for basic market order
        order_id = order.get("id")
        if not order_id:
            logger.error(
                Fore.RED
                + "Market order submission failed to return an ID. Order might have failed."
            )
            # Attempt to log response if available
            logger.debug(f"Order submission response: {order}")
            return False
        logger.info(Fore.CYAN + f"Market order submitted: ID {order_id}")

    except ccxt.InsufficientFunds as e:
        logger.error(
            Fore.RED
            + Style.BRIGHT
            + f"Insufficient funds to place {side.upper()} market order for {qty} {symbol}: {e}"
        )
        return False
    except (ccxt.ExchangeError, ccxt.InvalidOrder) as e:
        logger.error(
            Fore.RED + Style.BRIGHT + f"Exchange error placing market order: {e}"
        )
        # Log order attempt details if possible
        logger.debug(
            f"Failed order details: Symbol={symbol}, Side={side}, Qty={float(qty)}"
        )
        return False
    except Exception as e:
        logger.error(
            Fore.RED
            + Style.BRIGHT
            + f"Unexpected error during market order placement: {e}",
            exc_info=True,
        )
        return False

    # 6. Verify Order Fill
    logger.info(
        f"Waiting {CONFIG.order_check_delay_seconds}s before checking order status..."
    )
    time.sleep(CONFIG.order_check_delay_seconds)
    order_status_data = check_order_status(
        order_id, symbol, timeout=CONFIG.order_check_timeout_seconds
    )

    filled_qty = Decimal("0.0")
    average_price = current_price  # Fallback to estimated entry price
    order_final_status = "unknown"

    if order_status_data:
        order_final_status = order_status_data.get("status", "unknown")
        filled_str = order_status_data.get("filled")
        average_str = order_status_data.get("average")  # Average fill price

        try:
            if filled_str is not None:
                filled_qty = Decimal(str(filled_str))
            if average_str is not None:
                # Use actual fill price if available for logging/potential adjustments
                average_price = Decimal(str(average_str))
        except (TypeError, ValueError) as e:
            logger.warning(
                f"{Fore.YELLOW}Could not parse filled quantity or average price from order status: {e}. Using defaults."
            )
            logger.debug(f"Order Status Data: {order_status_data}")

        if (
            order_final_status == "closed"
        ):  # 'closed' usually means fully filled for market orders
            logger.info(
                Fore.GREEN
                + Style.BRIGHT
                + f"Order {order_id} confirmed filled: {filled_qty} @ {average_price:.4f}"
            )
            # Sanity check: filled quantity should match requested quantity (or be very close)
            if abs(filled_qty - qty) > CONFIG.position_qty_epsilon * Decimal(
                "10"
            ):  # Allow slightly larger tolerance for fill vs request
                logger.warning(
                    f"{Fore.YELLOW}Filled quantity {filled_qty} differs significantly from requested {qty}."
                )
                # Decide if this is critical. For now, proceed with filled_qty.

        elif order_final_status in ["open", "partially_filled"]:
            # Market orders shouldn't stay 'open' long unless partially filled due to liquidity/size
            logger.warning(
                Fore.YELLOW
                + f"Order {order_id} status is '{order_final_status}'. Filled: {filled_qty}. SL will be based on filled amount."
            )
            if filled_qty < CONFIG.position_qty_epsilon:
                logger.error(
                    Fore.RED
                    + f"Order {order_id} has status '{order_final_status}' but filled quantity is effectively zero ({filled_qty}). Aborting SL placement."
                )
                # Attempt to cancel the potentially stuck order
                with contextlib.suppress(Exception):
                    exchange.cancel_order(order_id, symbol)
                return False
            # Continue, but use filled_qty for SL
        else:  # canceled, rejected, expired, failed, unknown
            logger.error(
                Fore.RED
                + f"Order {order_id} did not fill successfully: Status '{order_final_status}'. Aborting SL placement."
            )
            # Attempt to cancel just in case it's stuck somehow (unlikely for market)
            with contextlib.suppress(Exception):
                exchange.cancel_order(order_id, symbol)
            return False
    else:
        # check_order_status already logged error (e.g., timeout or not found)
        logger.error(
            Fore.RED
            + f"Could not determine status for order {order_id}. Assuming failure. Aborting SL placement."
        )
        # Attempt to cancel just in case it's stuck somehow
        with contextlib.suppress(Exception):
            exchange.cancel_order(order_id, symbol)
        return False

    # Re-check filled quantity against epsilon after status check
    if filled_qty < CONFIG.position_qty_epsilon:
        logger.error(
            Fore.RED
            + f"Order {order_id} resulted in effectively zero filled quantity ({filled_qty}) after status check. No position opened."
        )
        return False

    # 7. Place Initial Stop-Loss Order
    position_side = "long" if side == "buy" else "short"
    sl_order_side = "sell" if side == "buy" else "buy"

    # Use the SL price calculated earlier based on estimated entry price (current_price)
    # Re-formatting here ensures it matches market rules again, though it should already be formatted.
    sl_price_str_for_order = format_price(symbol, sl_price)
    # Use the *actual filled quantity* for the SL order size
    sl_qty_str_for_order = format_amount(symbol, filled_qty)

    # Ensure SL quantity is valid after formatting
    sl_qty_decimal = Decimal(sl_qty_str_for_order)
    if sl_qty_decimal < CONFIG.position_qty_epsilon or (
        min_qty is not None and sl_qty_decimal < min_qty
    ):
        logger.error(
            Fore.RED
            + f"Stop loss quantity ({sl_qty_decimal}) is invalid (zero, too small, or below min {min_qty}) after formatting filled amount {filled_qty}. Cannot place SL."
        )
        logger.warning(
            Fore.YELLOW + "Attempting emergency closure of unprotected position..."
        )
        try:
            # Use the same filled quantity and opposite side
            close_qty_str = format_amount(
                symbol, filled_qty
            )  # Format again just in case
            exchange.create_market_order(
                symbol, sl_order_side, float(close_qty_str), params={"reduceOnly": True}
            )
            logger.info(
                Fore.GREEN + "Emergency closure order placed for unprotected position."
            )
        except Exception as close_err:
            logger.critical(
                Fore.RED
                + Style.BRIGHT
                + f"EMERGENCY CLOSURE FAILED: {close_err}. MANUAL INTERVENTION REQUIRED!"
            )
        return False

    sl_params = {
        "stopLossPrice": sl_price_str_for_order,  # The trigger price for the stop market order
        "reduceOnly": True,
        # 'triggerPrice': sl_price_str_for_order, # CCXT often uses stopLossPrice, but some exchanges might use triggerPrice. Redundant here usually.
        "triggerBy": CONFIG.sl_trigger_by,  # e.g., 'LastPrice', 'MarkPrice'
        # Bybit specific potentially useful params (check CCXT unification/docs):
        # 'tpslMode': 'Full', # Affects if TP/SL apply to whole position
        # 'slTriggerBy': CONFIG.sl_trigger_by, # More specific param if available
        # 'positionIdx': 0 # For Bybit unified: 0 for one-way, 1 for long hedge, 2 for short hedge (important in hedge mode)
    }
    logger.info(
        Fore.CYAN
        + f"Placing Initial SL order: Side={sl_order_side}, Qty={sl_qty_str_for_order}, Trigger={sl_price_str_for_order}, TriggerBy={CONFIG.sl_trigger_by}"
    )
    logger.debug(f"SL Params: {sl_params}")

    sl_order_id = None
    try:
        # Use create_order with stop type. CCXT standard is often 'stop_market' or 'stop'.
        # create_order is the unified method. Check exchange.has['createStopMarketOrder'] etc. if needed.
        # We need a Stop Market order (triggers at stopLossPrice, fills at market)
        sl_order = exchange.create_order(
            symbol=symbol,
            type="stop_market",  # Standard type for stop-loss market order
            side=sl_order_side,
            amount=float(sl_qty_str_for_order),  # CCXT expects float amount
            price=None,  # Market stop loss doesn't need a limit price
            params=sl_params,
        )
        sl_order_id = sl_order.get("id")
        if not sl_order_id:
            # If no ID, the order likely failed silently or CCXT couldn't parse response
            raise ccxt.ExchangeError(
                f"Stop loss order placement did not return an ID. Response: {sl_order}"
            )

        # Store the SL order ID
        order_tracker[position_side]["sl_id"] = sl_order_id
        order_tracker[position_side]["tsl_id"] = (
            None  # Ensure TSL is cleared on new entry
        )
        logger.info(
            Fore.GREEN
            + Style.BRIGHT
            + f"Initial SL placed for {position_side.upper()} position: ID {sl_order_id}, Trigger: {sl_price_str_for_order}"
        )

        # Use actual average fill price in notification
        entry_msg = (
            f"ENTERED {side.upper()} {filled_qty} {symbol} @ {average_price:.4f}. "
            f"Initial SL @ {sl_price_str_for_order} (ID: {sl_order_id})."
        )
        logger.info(Back.BLUE + Fore.WHITE + Style.BRIGHT + entry_msg)
        termux_notify(
            "Trade Entry",
            f"{side.upper()} {symbol} @ {average_price:.4f} | SL ID: ...{sl_order_id[-6:]}",
        )
        return True  # Success! Both entry and SL placed.

    except ccxt.InsufficientFunds as e:
        # This shouldn't happen if entry worked, but check anyway
        logger.error(
            Fore.RED
            + Style.BRIGHT
            + f"Insufficient funds to place stop-loss order: {e}. Position is UNPROTECTED."
        )
        # Trigger emergency closure
        logger.warning(
            Fore.YELLOW + "Attempting emergency closure of unprotected position..."
        )
        try:
            close_qty_str = format_amount(symbol, filled_qty)
            exchange.create_market_order(
                symbol, sl_order_side, float(close_qty_str), params={"reduceOnly": True}
            )
            logger.info(Fore.GREEN + "Emergency closure order placed.")
        except Exception as close_err:
            logger.critical(
                Fore.RED
                + Style.BRIGHT
                + f"EMERGENCY CLOSURE FAILED: {close_err}. MANUAL INTERVENTION REQUIRED!"
            )
        return False  # Signal failure

    except (ccxt.ExchangeError, ccxt.InvalidOrder) as e:
        logger.error(
            Fore.RED
            + Style.BRIGHT
            + f"Failed to place initial SL order (ID: {sl_order_id if sl_order_id else 'N/A'}): {e}. Position might be UNPROTECTED."
        )
        logger.warning(
            Fore.YELLOW
            + "Position may be open without Stop Loss due to SL placement error."
        )
        # Trigger emergency closure
        logger.warning(
            Fore.YELLOW + "Attempting emergency closure of unprotected position..."
        )
        try:
            close_qty_str = format_amount(symbol, filled_qty)
            exchange.create_market_order(
                symbol, sl_order_side, float(close_qty_str), params={"reduceOnly": True}
            )
            logger.info(Fore.GREEN + "Emergency closure order placed.")
        except Exception as close_err:
            logger.critical(
                Fore.RED
                + Style.BRIGHT
                + f"EMERGENCY CLOSURE FAILED: {close_err}. MANUAL INTERVENTION REQUIRED!"
            )
        return False  # Signal failure
    except Exception as e:
        logger.error(
            Fore.RED + Style.BRIGHT + f"Unexpected error placing SL: {e}", exc_info=True
        )
        # Trigger emergency closure
        logger.warning(
            Fore.YELLOW
            + "Attempting emergency closure of unprotected position due to unexpected SL error..."
        )
        try:
            close_qty_str = format_amount(symbol, filled_qty)
            exchange.create_market_order(
                symbol, sl_order_side, float(close_qty_str), params={"reduceOnly": True}
            )
            logger.info(Fore.GREEN + "Emergency closure order placed.")
        except Exception as close_err:
            logger.critical(
                Fore.RED
                + Style.BRIGHT
                + f"EMERGENCY CLOSURE FAILED: {close_err}. MANUAL INTERVENTION REQUIRED!"
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
    """
    Manages the activation and placement of a trailing stop loss, replacing the initial SL.
    Uses Decimal for calculations.
    """
    # Ensure we have a valid position to manage
    if position_qty < CONFIG.position_qty_epsilon or entry_price <= Decimal("0"):
        # If position is closed or invalid, ensure trackers are clear
        if (
            order_tracker[position_side]["tsl_id"]
            or order_tracker[position_side]["sl_id"]
        ):
            logger.debug(
                f"Position {position_side} closed or invalid, clearing any stale order trackers."
            )
            order_tracker[position_side] = {"sl_id": None, "tsl_id": None}
        return

    # Check current state from tracker
    initial_sl_id = order_tracker[position_side]["sl_id"]
    tsl_id = order_tracker[position_side]["tsl_id"]
    has_initial_sl = initial_sl_id is not None
    has_tsl = tsl_id is not None

    # If TSL is already active, the exchange handles the trailing.
    # Optional: Periodically verify the TSL order still exists on the exchange? (Adds complexity/API calls)
    if has_tsl:
        logger.debug(
            f"{position_side.upper()} TSL (ID: {tsl_id}) is already active. Exchange managing trail."
        )
        # Example verification (optional, use sparingly):
        # if cycle_count % 10 == 0: # Check every 10 cycles
        #    tsl_status = check_order_status(tsl_id, symbol, timeout=5)
        #    if tsl_status is None or tsl_status.get('status') not in ['open']: # Assuming TSL stays 'open' until triggered
        #        logger.warning(f"Active TSL order {tsl_id} no longer found or not open. Status: {tsl_status}. Resetting tracker.")
        #        order_tracker[position_side]["tsl_id"] = None
        #        # Potentially need to place a new SL/TSL here if position still exists? Risky.
        return

    # --- Check for TSL Activation Condition ---
    # Can only activate TSL if we have an initial SL to replace
    if not has_initial_sl:
        logger.debug(
            f"No initial SL tracked for {position_side} position. Cannot activate TSL."
        )
        return

    if atr is None or atr.is_nan() or atr <= Decimal("0"):
        logger.warning(
            Fore.YELLOW + "Cannot evaluate TSL activation: Invalid ATR value ({atr})."
        )
        return

    # Calculate profit in points (quote currency)
    profit = Decimal("0.0")
    if position_side == "long":
        profit = current_price - entry_price
    else:  # short
        profit = entry_price - current_price

    # Calculate activation threshold in points
    activation_threshold_points = CONFIG.tsl_activation_atr_multiplier * atr
    logger.debug(
        f"{position_side.upper()} Profit: {profit:.4f}, TSL Activation Threshold (Points): {activation_threshold_points:.4f} (ATR={atr:.4f})"
    )

    # Activate TSL only if profit exceeds the threshold
    if profit > activation_threshold_points:
        logger.info(
            Fore.GREEN
            + Style.BRIGHT
            + f"Profit threshold reached for {position_side.upper()} position. Activating TSL."
        )

        # --- 1. Cancel Initial SL ---
        logger.info(
            Fore.CYAN
            + f"Attempting to cancel initial SL (ID: {initial_sl_id}) before placing TSL..."
        )
        try:
            # Use cancel_order with the stored ID
            exchange.cancel_order(initial_sl_id, symbol)
            logger.info(
                Fore.GREEN + f"Successfully cancelled initial SL (ID: {initial_sl_id})."
            )
            order_tracker[position_side]["sl_id"] = (
                None  # Mark as cancelled locally immediately
            )
        except ccxt.OrderNotFound:
            logger.warning(
                Fore.YELLOW
                + f"Initial SL (ID: {initial_sl_id}) not found when trying to cancel. Might have been triggered or already cancelled/rejected."
            )
            order_tracker[position_side]["sl_id"] = None  # Assume it's gone
        except (ccxt.ExchangeError, ccxt.NetworkError) as e:
            logger.error(
                Fore.RED + f"Failed to cancel initial SL (ID: {initial_sl_id}): {e}."
            )
            logger.warning(
                Fore.YELLOW
                + "Proceeding with TSL placement attempt, but risk of duplicate orders exists if SL cancellation failed silently."
            )
            # Mark SL as None locally anyway to allow TSL placement attempt
            order_tracker[position_side]["sl_id"] = None
        except Exception as e:
            logger.error(
                Fore.RED + f"Unexpected error cancelling initial SL: {e}", exc_info=True
            )
            order_tracker[position_side]["sl_id"] = None  # Assume cancelled locally

        # --- 2. Place Trailing Stop Loss Order ---
        tsl_order_side = "sell" if position_side == "long" else "buy"
        # Use the current position quantity for the TSL order
        tsl_qty_str = format_amount(symbol, position_qty)
        tsl_qty_decimal = Decimal(tsl_qty_str)

        # Ensure TSL quantity is valid
        min_qty_str = MARKET_INFO.get("limits", {}).get("amount", {}).get("min")
        min_qty = Decimal(str(min_qty_str)) if min_qty_str is not None else None
        if tsl_qty_decimal < CONFIG.position_qty_epsilon or (
            min_qty is not None and tsl_qty_decimal < min_qty
        ):
            logger.error(
                Fore.RED
                + f"Trailing stop loss quantity ({tsl_qty_decimal}) is invalid (zero, too small, or below min {min_qty}) using position qty {position_qty}. Cannot place TSL."
            )
            logger.warning(
                Fore.YELLOW
                + "Position is now UNPROTECTED as initial SL may have been cancelled. MANUAL INTERVENTION MAY BE NEEDED."
            )
            # Consider emergency closure here?
            return  # Abort TSL placement

        # Convert Decimal percentage (e.g., 0.5) to float for CCXT param if needed, but CCXT might handle string % too.
        # Bybit API likely expects percentage value, e.g., '0.5' for 0.5%. CCXT should handle this.
        trail_percent_value_str = str(CONFIG.trailing_stop_percent)  # Pass as string

        # Check CCXT documentation and exchange capabilities for trailing stop parameters
        # Common parameters: 'trailingPercent', 'trailingAmount', 'activationPrice'
        tsl_params = {
            "reduceOnly": True,
            "triggerBy": CONFIG.tsl_trigger_by,  # Use configured trigger type
            "trailingPercent": trail_percent_value_str,  # CCXT standard parameter for percentage-based trail. Pass as string.
            # 'activationPrice': format_price(symbol, current_price) # Optional: Price at which the trail *starts*.
            # If not provided, trail might start immediately based on trigger price rules.
            # Bybit might use 'activePrice'. CCXT might map this. Test needed.
            # Bybit specific params (check if needed/mapped by CCXT):
            # 'tpslMode': 'Full',
            # 'trailingStop': trail_percent_value_str, # Bybit might use this param name directly?
            # 'activePrice': format_price(symbol, current_price), # Bybit's activation price
            # 'positionIdx': 0
        }
        logger.info(
            Fore.CYAN
            + f"Placing TSL order: Side={tsl_order_side}, Qty={tsl_qty_str}, Trail%={trail_percent_value_str}, TriggerBy={CONFIG.tsl_trigger_by}"
        )
        logger.debug(f"TSL Params: {tsl_params}")

        new_tsl_id = None
        try:
            # Use create_order with the specific type for trailing stops if available and preferred
            # Check exchange.has['createTrailingStopMarketOrder'] or similar.
            # If not available, try 'stop_market' or 'market' with trailing params if supported by unified method.
            # Let's try the specific type first if the exchange 'has' it.
            order_type = "trailing_stop_market"  # Ideal type if supported
            # if not exchange.has.get('createTrailingStopMarketOrder'):
            #     logger.debug(f"Exchange does not explicitly report support for 'trailing_stop_market' type via 'has'. Trying anyway or falling back...")
            #     # Fallback? Maybe try 'stop_market' with trailing params? Requires testing.
            #     # order_type = 'stop_market' # Example fallback

            tsl_order = exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=tsl_order_side,
                amount=float(tsl_qty_str),  # CCXT expects float amount
                price=None,  # Market based trail
                params=tsl_params,
            )
            new_tsl_id = tsl_order.get("id")
            if not new_tsl_id:
                raise ccxt.ExchangeError(
                    f"Trailing stop order placement did not return an ID. Response: {tsl_order}"
                )

            # Update tracker
            order_tracker[position_side]["tsl_id"] = new_tsl_id
            order_tracker[position_side]["sl_id"] = (
                None  # Ensure initial SL ID is cleared
            )
            logger.info(
                Fore.GREEN
                + Style.BRIGHT
                + f"Trailing Stop Loss activated for {position_side.upper()}: ID {new_tsl_id}, Trail: {trail_percent_value_str}%"
            )
            termux_notify(
                "TSL Activated",
                f"{position_side.upper()} {symbol} TSL ID: ...{new_tsl_id[-6:]}",
            )

        except (ccxt.ExchangeError, ccxt.InvalidOrder) as e:
            logger.error(
                Fore.RED
                + Style.BRIGHT
                + f"Failed to place TSL order (ID: {new_tsl_id if new_tsl_id else 'N/A'}, tried type '{order_type}'): {e}"
            )
            logger.warning(
                Fore.YELLOW
                + "Position might be UNPROTECTED as initial SL was likely cancelled and TSL placement failed."
            )
            # CRITICAL: At this point, the initial SL might be cancelled, and TSL failed.
            # Reset local tracker as TSL failed
            order_tracker[position_side]["tsl_id"] = None
            # Consider placing a new *regular* stop loss as a fallback? This adds complexity.
            # For now, log critical warning. Manual intervention might be needed.
            logger.critical(
                Fore.RED
                + Style.BRIGHT
                + "MANUAL INTERVENTION RECOMMENDED: Check position and place a stop loss manually if needed."
            )
            # place_fallback_stop_loss(symbol, position_side, position_qty, current_price) # Example function call if implemented
        except Exception as e:
            logger.error(
                Fore.RED + Style.BRIGHT + f"Unexpected error placing TSL: {e}",
                exc_info=True,
            )
            logger.warning(
                Fore.YELLOW
                + "Position might be unprotected after unexpected TSL placement error."
            )
            order_tracker[position_side]["tsl_id"] = None
            logger.critical(
                Fore.RED
                + Style.BRIGHT
                + "MANUAL INTERVENTION RECOMMENDED due to unexpected TSL error."
            )


def print_status_panel(
    cycle: int,
    timestamp: pd.Timestamp,
    price: Decimal,
    indicators: Dict[str, Decimal],
    positions: Dict[str, Dict[str, Decimal]],
    equity: Optional[Decimal],
    signals: Dict[str, bool],
    order_tracker_state: Dict[
        str, Dict[str, Optional[str]]
    ],  # Pass tracker state explicitly
) -> None:
    """Displays the current state using a mystical status panel with Decimal precision."""
    headers = ["Metric", "Value", "Detail"]
    panel_data = []

    # Time & Cycle
    panel_data.append(
        [
            Fore.MAGENTA + "Cycle",
            f"{Style.BRIGHT}{cycle}",
            f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
    )

    # Market & Indicators
    trend_ema = indicators.get("trend_ema", Decimal("NaN"))
    price_color = (
        Fore.GREEN
        if price > trend_ema
        else Fore.RED
        if price < trend_ema
        else Fore.WHITE
    )
    panel_data.append(
        [
            Fore.CYAN + "Market",
            f"{Style.BRIGHT}{CONFIG.symbol}",
            f"{price_color}{price:.4f}{Style.RESET_ALL}",
        ]
    )

    atr = indicators.get("atr", Decimal("NaN"))
    panel_data.append([Fore.CYAN + "ATR (14)", f"{Fore.WHITE}{atr:.6f}", ""])

    fast_ema = indicators.get("fast_ema", Decimal("NaN"))
    slow_ema = indicators.get("slow_ema", Decimal("NaN"))
    ema_cross_color = (
        Fore.GREEN
        if fast_ema > slow_ema
        else Fore.RED
        if fast_ema < slow_ema
        else Fore.WHITE
    )
    ema_status = (
        f"{Fore.GREEN}Bullish"
        if ema_cross_color == Fore.GREEN
        else f"{Fore.RED}Bearish"
        if ema_cross_color == Fore.RED
        else "Neutral"
    )
    panel_data.append(
        [
            Fore.CYAN + "EMA (8/12)",
            f"{ema_cross_color}{fast_ema:.4f} / {slow_ema:.4f}",
            ema_status,
        ]
    )

    trend_status = (
        f"{price_color}(Above Trend)"
        if price > trend_ema
        else f"{price_color}(Below Trend)"
        if price < trend_ema
        else "(At Trend)"
    )
    panel_data.append(
        [Fore.CYAN + "EMA Trend (50)", f"{Fore.WHITE}{trend_ema:.4f}", trend_status]
    )

    stoch_k = indicators.get("stoch_k", Decimal("NaN"))
    stoch_d = indicators.get("stoch_d", Decimal("NaN"))
    stoch_color = (
        Fore.GREEN
        if stoch_k < Decimal(25)
        else Fore.RED
        if stoch_k > Decimal(75)
        else Fore.YELLOW
    )
    stoch_status = (
        f"{Fore.GREEN}Oversold (<25)"
        if stoch_color == Fore.GREEN
        else f"{Fore.RED}Overbought (>75)"
        if stoch_color == Fore.RED
        else "Neutral"
    )
    panel_data.append(
        [
            Fore.CYAN + "Stoch %K/%D",
            f"{stoch_color}{stoch_k:.2f} / {stoch_d:.2f}",
            stoch_status,
        ]
    )

    # Account & Positions
    equity_str = (
        f"{equity:.2f} {MARKET_INFO.get('quote', 'USD')}"
        if equity is not None and not equity.is_nan()
        else f"{Fore.YELLOW}N/A"
    )
    panel_data.append([Fore.CYAN + "Equity", equity_str, ""])

    long_pos = positions.get(
        "long", {"qty": Decimal("0.0"), "entry_price": Decimal("0.0")}
    )
    short_pos = positions.get(
        "short", {"qty": Decimal("0.0"), "entry_price": Decimal("0.0")}
    )

    long_qty_str = (
        f"{Fore.GREEN}{long_pos['qty']}"
        if long_pos["qty"] > CONFIG.position_qty_epsilon
        else f"{Fore.WHITE}0.0"
    )
    short_qty_str = (
        f"{Fore.RED}{short_pos['qty']}"
        if short_pos["qty"] > CONFIG.position_qty_epsilon
        else f"{Fore.WHITE}0.0"
    )
    panel_data.append(
        [
            Fore.CYAN + "Position Qty",
            long_qty_str + " (Long)",
            short_qty_str + " (Short)",
        ]
    )

    long_entry_str = (
        f"{long_pos['entry_price']:.4f}"
        if long_pos["qty"] > CONFIG.position_qty_epsilon
        else "-"
    )
    short_entry_str = (
        f"{short_pos['entry_price']:.4f}"
        if short_pos["qty"] > CONFIG.position_qty_epsilon
        else "-"
    )
    panel_data.append(
        [
            Fore.CYAN + "Entry Price",
            f"{Fore.GREEN}{long_entry_str}",
            f"{Fore.RED}{short_entry_str}",
        ]
    )

    # Use the passed tracker state for order status
    long_sl_id = order_tracker_state["long"]["sl_id"]
    long_tsl_id = order_tracker_state["long"]["tsl_id"]
    short_sl_id = order_tracker_state["short"]["sl_id"]
    short_tsl_id = order_tracker_state["short"]["tsl_id"]

    long_stop_status = Fore.RED + Style.DIM + "None"
    if long_tsl_id:
        long_stop_status = f"{Fore.GREEN}TSL (...{long_tsl_id[-6:]})"
    elif long_sl_id:
        long_stop_status = f"{Fore.YELLOW}SL (...{long_sl_id[-6:]})"

    short_stop_status = Fore.RED + Style.DIM + "None"
    if short_tsl_id:
        short_stop_status = f"{Fore.GREEN}TSL (...{short_tsl_id[-6:]})"
    elif short_sl_id:
        short_stop_status = f"{Fore.YELLOW}SL (...{short_sl_id[-6:]})"
    panel_data.append([Fore.CYAN + "Active Stop", long_stop_status, short_stop_status])

    # Signals
    long_signal_color = (
        Fore.GREEN + Style.BRIGHT
        if signals.get("long", False)
        else Fore.WHITE + Style.DIM
    )
    short_signal_color = (
        Fore.RED + Style.BRIGHT
        if signals.get("short", False)
        else Fore.WHITE + Style.DIM
    )
    panel_data.append(
        [
            Fore.CYAN + "Signals",
            f"{long_signal_color}LONG",
            f"{short_signal_color}SHORT",
        ]
    )

    # Format with tabulate
    # Using 'fancy_grid' for better visual separation
    output = tabulate(
        panel_data,
        headers=headers,
        tablefmt="fancy_grid",
        stralign="left",
        numalign="left",
    )
    print("\n" + output)


def generate_signals(
    indicators: Dict[str, Decimal], current_price: Decimal
) -> Dict[str, bool]:
    """Generates trading signals based on indicator conditions, using Decimal."""
    signals = {"long": False, "short": False}
    logger.debug("Generating signals...")

    if not indicators:
        logger.warning("Cannot generate signals: indicators are missing.")
        return signals

    try:
        # Use .get with default Decimal values for safety, check for NaN
        k = indicators.get("stoch_k", Decimal("NaN"))
        d = indicators.get("stoch_d", Decimal("NaN"))  # Use D as well
        fast_ema = indicators.get("fast_ema", Decimal("NaN"))
        slow_ema = indicators.get("slow_ema", Decimal("NaN"))
        trend_ema = indicators.get("trend_ema", Decimal("NaN"))

        # Check if any crucial indicator is NaN
        if any(ind.is_nan() for ind in [k, d, fast_ema, slow_ema, trend_ema]):
            logger.warning(
                f"{Fore.YELLOW}Cannot generate signals: One or more indicators are NaN. Indicators: {indicators}"
            )
            return signals

        # --- Define Conditions using Decimal comparisons ---
        # EMA Cross
        ema_bullish_cross = fast_ema > slow_ema
        ema_bearish_cross = fast_ema < slow_ema
        # Price vs Trend EMA
        price_above_trend = current_price > trend_ema
        price_below_trend = current_price < trend_ema
        # Stochastic Levels & Cross (More robust than just level)
        stoch_oversold_level = k < Decimal(25)  # K is below 25
        stoch_overbought_level = k > Decimal(75)  # K is above 75
        stoch_bullish_cross = k > d  # K crossed above D
        stoch_bearish_cross = k < d  # K crossed below D

        logger.debug(
            f"Signal Inputs: Price={current_price:.4f}, FastEMA={fast_ema:.4f}, SlowEMA={slow_ema:.4f}, TrendEMA={trend_ema:.4f}, StochK={k:.2f}, StochD={d:.2f}"
        )
        logger.debug(
            f"Conditions: EMA Bullish={ema_bullish_cross}, EMA Bearish={ema_bearish_cross}, Price Above Trend={price_above_trend}, Stoch Oversold={stoch_oversold_level}, Stoch Overbought={stoch_overbought_level}, Stoch Bull Cross={stoch_bullish_cross}"
        )

        # --- Signal Logic ---

        # Long Signal:
        # 1. Bullish EMA cross (fast > slow)
        # 2. Stochastic K crossed *above* D (momentum confirmation)
        # 3. Stochastic K is in the oversold region (or recently exited it)
        # 4. Trend Filter (Optional): Price is above Trend EMA
        long_conditions_met = (
            ema_bullish_cross
            and stoch_bullish_cross
            and stoch_oversold_level  # Require K to be low when crossing D
        )
        if long_conditions_met:
            if CONFIG.trade_only_with_trend:
                if price_above_trend:
                    signals["long"] = True
                    logger.debug(
                        "Long Signal Criteria Met: EMA Cross Bullish, Stoch Bullish Cross in Oversold, Price Above Trend EMA"
                    )
                else:
                    logger.debug(
                        "Long Signal Blocked: Price Below Trend EMA (Trend Filter ON)"
                    )
            else:  # Trend filter off
                signals["long"] = True
                logger.debug(
                    "Long Signal Criteria Met: EMA Cross Bullish, Stoch Bullish Cross in Oversold (Trend Filter OFF)"
                )

        # Short Signal:
        # 1. Bearish EMA cross (fast < slow)
        # 2. Stochastic K crossed *below* D (momentum confirmation)
        # 3. Stochastic K is in the overbought region (or recently exited it)
        # 4. Trend Filter (Optional): Price is below Trend EMA
        short_conditions_met = (
            ema_bearish_cross
            and stoch_bearish_cross
            and stoch_overbought_level  # Require K to be high when crossing D
        )
        if short_conditions_met:
            if CONFIG.trade_only_with_trend:
                if price_below_trend:
                    signals["short"] = True
                    logger.debug(
                        "Short Signal Criteria Met: EMA Cross Bearish, Stoch Bearish Cross in Overbought, Price Below Trend EMA"
                    )
                else:
                    logger.debug(
                        "Short Signal Blocked: Price Above Trend EMA (Trend Filter ON)"
                    )
            else:  # Trend filter off
                signals["short"] = True
                logger.debug(
                    "Short Signal Criteria Met: EMA Cross Bearish, Stoch Bearish Cross in Overbought (Trend Filter OFF)"
                )

        # Refinement Ideas (Not Implemented):
        # - Add cooldown period after a signal.
        # - Add volume confirmation.
        # - Add divergence checks.
        # - Check if EMAs are separating strongly (momentum).
        # - Check previous candle patterns.

    except Exception as e:
        logger.error(f"{Fore.RED}Error generating signals: {e}", exc_info=True)
        return {"long": False, "short": False}  # Return default on error

    logger.debug(f"Generated Signals: Long={signals['long']}, Short={signals['short']}")
    return signals


def trading_spell_cycle(cycle_count: int) -> None:
    """Executes one cycle of the trading spell with enhanced precision and logic."""
    logger.info(Fore.MAGENTA + Style.BRIGHT + f"\n--- Starting Cycle {cycle_count} ---")
    start_time = time.monotonic()

    # 1. Fetch Market Data
    df = fetch_market_data(CONFIG.symbol, CONFIG.interval, CONFIG.ohlcv_limit)
    if df is None or df.empty:
        logger.error(
            Fore.RED + "Halting cycle: Market data fetch failed or returned empty."
        )
        return

    # Use Decimal for current price from the latest candle's close
    try:
        # Ensure the DataFrame index is sorted if needed (usually is from fetch_ohlcv)
        # df.sort_index(inplace=True)
        current_price_float = df["close"].iloc[-1]
        current_price = Decimal(str(current_price_float))
        last_timestamp = df.index[-1]
        logger.debug(f"Latest candle close: {current_price:.4f} at {last_timestamp}")
    except IndexError:
        logger.error(
            Fore.RED + "Failed to get current price from DataFrame (IndexError)."
        )
        return
    except Exception as e:
        logger.error(Fore.RED + f"Error processing current price: {e}", exc_info=True)
        return

    # 2. Calculate Indicators (returns Decimals)
    indicators = calculate_indicators(df)
    if indicators is None:
        logger.error(Fore.RED + "Halting cycle: Indicator calculation failed.")
        return
    current_atr = indicators.get(
        "atr", Decimal("NaN")
    )  # Keep as Decimal, check for NaN later

    # 3. Get Current State (Positions & Balance as Decimals)
    # Fetch balance first to know available capital
    quote_currency = MARKET_INFO.get("quote", "USDT") if MARKET_INFO else "USDT"
    free_balance, current_equity = get_balance(quote_currency)
    if current_equity is None:
        # Allow proceeding without equity if balance fetch fails, but log warning
        logger.warning(
            Fore.YELLOW
            + f"Failed to fetch current {quote_currency} balance/equity. Status panel may be incomplete."
        )
        current_equity = Decimal("NaN")  # Use NaN to indicate missing data

    # Fetch positions
    positions = get_current_position(CONFIG.symbol)
    if positions is None:
        logger.error(Fore.RED + "Halting cycle: Failed to fetch current positions.")
        # Decide if we should halt or proceed cautiously assuming flat? Halting is safer.
        return

    # Ensure positions dict has expected structure (already done in get_current_position)
    long_pos = positions["long"]
    short_pos = positions["short"]

    # 4. Manage Trailing Stops (pass Decimals)
    # Check if ATR is valid before managing stops
    if current_atr.is_nan():
        logger.warning(f"{Fore.YELLOW}Skipping TSL management: ATR is NaN.")
    else:
        # Manage long TSL if long position exists
        if long_pos["qty"].copy_abs() >= CONFIG.position_qty_epsilon:
            manage_trailing_stop(
                CONFIG.symbol,
                "long",
                long_pos["qty"],
                long_pos["entry_price"],
                current_price,
                current_atr,
            )
        # Manage short TSL if short position exists
        if short_pos["qty"].copy_abs() >= CONFIG.position_qty_epsilon:
            manage_trailing_stop(
                CONFIG.symbol,
                "short",
                short_pos["qty"],
                short_pos["entry_price"],
                current_price,
                current_atr,
            )

    # 5. Generate Trading Signals (pass Decimals)
    signals = generate_signals(indicators, current_price)

    # --- Make a copy of the order tracker state *before* potential trade execution ---
    # This ensures the status panel reflects the state *at the time of decision making*
    # Use deepcopy if tracker contained mutable objects, but dicts of strings/None is fine with .copy()
    order_tracker_snapshot = {
        "long": order_tracker["long"].copy(),
        "short": order_tracker["short"].copy(),
    }

    # 6. Execute Trades based on Signals
    # Check if flat (neither long nor short position significantly open)
    is_flat = (
        long_pos["qty"].copy_abs() < CONFIG.position_qty_epsilon
        and short_pos["qty"].copy_abs() < CONFIG.position_qty_epsilon
    )
    logger.debug(
        f"Position Status: Flat = {is_flat} (Long Qty: {long_pos['qty']}, Short Qty: {short_pos['qty']})"
    )

    trade_executed = False
    if is_flat:
        # Check ATR validity again before attempting trade based on it
        if current_atr.is_nan():
            logger.warning(f"{Fore.YELLOW}Cannot place new order: ATR is NaN.")
        elif signals.get("long"):
            logger.info(
                Fore.GREEN + Style.BRIGHT + "Long signal detected! Attempting entry..."
            )
            trade_executed = place_risked_market_order(
                CONFIG.symbol, "buy", CONFIG.risk_percentage, current_atr
            )
        elif signals.get("short"):
            logger.info(
                Fore.RED + Style.BRIGHT + "Short signal detected! Attempting entry..."
            )
            trade_executed = place_risked_market_order(
                CONFIG.symbol, "sell", CONFIG.risk_percentage, current_atr
            )
        else:
            logger.info("No entry signals detected while flat.")

        # If a trade was attempted, pause briefly to allow exchange state to potentially update
        if trade_executed:
            logger.info("Pausing briefly after trade attempt...")
            time.sleep(
                CONFIG.order_check_delay_seconds + 1
            )  # Small pause slightly longer than order check delay

    elif not is_flat:
        logger.info("Position already open, skipping new entry signals.")
        # --- Potential Exit Logic (Placeholder) ---
        # Example: Exit long if a strong short signal appears?
        # if long_pos['qty'].copy_abs() >= CONFIG.position_qty_epsilon and signals.get("short"):
        #     logger.info(Fore.RED + "Short signal detected while long. Considering exit...")
        #     # close_position(CONFIG.symbol, "long", long_pos['qty']) # Needs a close_position function
        # Example: Exit short if a strong long signal appears?
        # elif short_pos['qty'].copy_abs() >= CONFIG.position_qty_epsilon and signals.get("long"):
        #     logger.info(Fore.GREEN + "Long signal detected while short. Considering exit...")
        #     # close_position(CONFIG.symbol, "short", short_pos['qty']) # Needs a close_position function

    # 7. Display Status Panel
    # Use the state captured *before* trade execution for consistency in the panel for this cycle
    print_status_panel(
        cycle_count,
        last_timestamp,
        current_price,
        indicators,
        positions,
        current_equity,
        signals,
        order_tracker_snapshot,  # Use the snapshot
    )

    end_time = time.monotonic()
    logger.info(
        Fore.MAGENTA
        + f"--- Cycle {cycle_count} Complete (Duration: {end_time - start_time:.2f}s) ---"
    )


def graceful_shutdown() -> None:
    """Dispels active orders and closes open positions gracefully with precision."""
    logger.warning(
        Fore.YELLOW + Style.BRIGHT + "\n=== Initiating Graceful Shutdown Sequence ==="
    )
    termux_notify(
        "Shutdown Started", f"Closing orders/positions for {CONFIG.symbol}..."
    )

    # Ensure exchange object is available
    if "exchange" not in globals() or not hasattr(exchange, "cancel_all_orders"):
        logger.error(
            Fore.RED
            + "Exchange object not available for shutdown. Cannot manage orders/positions."
        )
        return

    # 1. Cancel All Open Orders for the Symbol
    try:
        logger.info(Fore.CYAN + f"Dispelling all open orders for {CONFIG.symbol}...")
        open_orders = []
        try:
            # Fetch open orders first to log IDs before cancelling
            # This includes SL and TSL orders placed by the bot
            open_orders = exchange.fetch_open_orders(CONFIG.symbol)
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
            try:
                # Bybit might require cancel_all_orders(symbol, params={'settleCoin': 'USDT'}) or similar? Check docs.
                # Let's try the standard call first.
                response = exchange.cancel_all_orders(CONFIG.symbol)
                logger.info(
                    Fore.GREEN
                    + f"Cancel all orders command sent. Exchange response: {response}"
                )  # Response varies
                # Add a small delay for cancellations to process
                time.sleep(2)
            except (ccxt.ExchangeError, ccxt.NetworkError) as cancel_err:
                logger.error(
                    Fore.RED
                    + f"Error sending cancel all orders command: {cancel_err}. MANUAL ORDER CHECK REQUIRED."
                )
            except Exception as cancel_exc:
                logger.error(
                    Fore.RED
                    + f"Unexpected error sending cancel all orders command: {cancel_exc}. MANUAL ORDER CHECK REQUIRED."
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
            + f"Unexpected error during order cancellation phase: {e}. MANUAL CHECK REQUIRED.",
            exc_info=True,
        )

    # Add a small delay after cancelling orders before checking/closing positions
    logger.info("Waiting briefly after order cancellation...")
    time.sleep(3)

    # 2. Close Any Open Positions
    try:
        logger.info(Fore.CYAN + "Checking for lingering positions to close...")
        # Fetch final position state using the dedicated function
        positions = get_current_position(CONFIG.symbol)  # Re-fetch after cancellations
        closed_count = 0

        if positions is None:
            logger.error(
                Fore.RED
                + "Could not fetch final positions during shutdown. MANUAL POSITION CHECK REQUIRED."
            )
            # Cannot proceed with automated closure if position state is unknown
            return

        for side, pos_data in positions.items():
            qty = pos_data.get("qty", Decimal("0.0"))
            # Check if quantity is significant using epsilon
            if qty.copy_abs() >= CONFIG.position_qty_epsilon:
                close_side = "sell" if side == "long" else "buy"
                logger.warning(
                    Fore.YELLOW
                    + f"Closing {side} position ({qty} {CONFIG.symbol}) with market order..."
                )
                try:
                    # Format quantity precisely for closure order
                    close_qty_str = format_amount(
                        CONFIG.symbol, qty.copy_abs()
                    )  # Use absolute value for amount
                    close_qty_decimal = Decimal(close_qty_str)

                    # Final check on close quantity validity
                    min_qty_str = (
                        MARKET_INFO.get("limits", {}).get("amount", {}).get("min")
                        if MARKET_INFO
                        else None
                    )
                    min_qty = (
                        Decimal(str(min_qty_str)) if min_qty_str is not None else None
                    )
                    if close_qty_decimal < CONFIG.position_qty_epsilon or (
                        min_qty is not None and close_qty_decimal < min_qty
                    ):
                        logger.error(
                            Fore.RED
                            + f"Cannot close {side} position: Formatted close quantity {close_qty_decimal} is invalid (zero, too small, or below min {min_qty})."
                        )
                        logger.critical(
                            Fore.RED
                            + Style.BRIGHT
                            + f"MANUAL INTERVENTION REQUIRED TO CLOSE {side} POSITION."
                        )
                        continue  # Skip to next side if any

                    close_order = exchange.create_market_order(
                        symbol=CONFIG.symbol,
                        side=close_side,
                        amount=float(close_qty_str),  # CCXT needs float
                        params={"reduceOnly": True},  # Crucial to ensure it only closes
                    )
                    close_order_id = close_order.get("id", "N/A")
                    logger.info(
                        Fore.GREEN
                        + f"Position closure order placed: Side={close_side}, Qty={close_qty_str}, ID={close_order_id}"
                    )
                    closed_count += 1
                    # Add a small delay to allow closure order to process before finishing shutdown
                    time.sleep(CONFIG.order_check_delay_seconds)
                except ccxt.InsufficientFunds:
                    # This might happen if funds are locked in orders that failed to cancel?
                    logger.critical(
                        Fore.RED
                        + Style.BRIGHT
                        + f"Insufficient funds to close {side} position ({qty}). MANUAL INTERVENTION REQUIRED!"
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

        if closed_count == 0:
            logger.info(Fore.GREEN + "No open positions found requiring closure.")
        else:
            logger.info(f"Attempted to close {closed_count} position(s).")

    except Exception as e:
        logger.error(
            Fore.RED
            + f"Error during position closure check phase: {e}. Manual check advised.",
            exc_info=True,
        )

    logger.warning(
        Fore.YELLOW + Style.BRIGHT + "=== Graceful Shutdown Sequence Complete ==="
    )
    termux_notify("Shutdown Complete", f"{CONFIG.symbol} bot stopped.")


# --- Main Spell Invocation ---
if __name__ == "__main__":
    logger.info(
        Back.MAGENTA
        + Fore.WHITE
        + Style.BRIGHT
        + "*** Pyrmethus Termux Trading Spell Activated (v2 Precision) ***"
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
    logger.info(f"Trade Only With Trend (50 EMA): {CONFIG.trade_only_with_trend}")
    logger.info(f"Position Quantity Epsilon: {CONFIG.position_qty_epsilon}")
    logger.info(f"Log File: {LOG_FILENAME}")

    if MARKET_INFO:  # Check if market info loaded successfully during init
        termux_notify("Bot Started", f"Monitoring {CONFIG.symbol} (v2 Precision)")
        logger.info(
            Fore.GREEN
            + Style.BRIGHT
            + "Nexus connection stable. Awaiting market whispers..."
        )
    else:
        # Error should have been logged during init, this is a final check
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
                + f"Cycle {cycle} finished. Resting for {CONFIG.loop_sleep_seconds} seconds..."
            )
            time.sleep(CONFIG.loop_sleep_seconds)

    except KeyboardInterrupt:
        logger.warning(Fore.YELLOW + "\nCtrl+C detected! Initiating shutdown...")
        graceful_shutdown()
    except Exception as e:
        # Catch unexpected errors in the main loop
        logger.critical(
            Fore.RED
            + Style.BRIGHT
            + f"\nFATAL RUNTIME ERROR in Main Loop (Cycle {cycle}): {e}",
            exc_info=True,
        )
        termux_notify("Bot CRASHED", f"{CONFIG.symbol} Error: Check logs!")
        # Attempt cleanup even on unexpected crash
        graceful_shutdown()
        sys.exit(1)  # Exit with error code
    finally:
        # Ensure logs are flushed and handlers closed properly
        logger.info("Flushing logs and shutting down logging system.")
        logging.shutdown()
        print(Fore.MAGENTA + Style.BRIGHT + "Pyrmethus spell has faded.")
