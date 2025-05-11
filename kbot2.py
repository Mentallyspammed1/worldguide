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
import sys
import time
from decimal import Decimal
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
# We will primarily use it for critical financial calculations like position sizing if enabled.
# getcontext().prec = 28 # Example: Set precision to 28 digits (default is usually sufficient)

# --- Arcane Configuration ---

# Summon secrets from the .env scroll
load_dotenv()

# Configure the Ethereal Log Scribe
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
logger.setLevel(logging.INFO)  # Set to DEBUG for more verbose output
stream_handler = logging.StreamHandler(sys.stdout)  # Explicitly use stdout
stream_handler.setFormatter(log_formatter)
if not logger.hasHandlers():
    logger.addHandler(stream_handler)
logger.propagate = False


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

        self.position_qty_epsilon = Decimal(
            "0.000001"
        )  # Threshold for considering position closed (as Decimal)
        self.api_key = self._get_env("BYBIT_API_KEY", None, Fore.RED)
        self.api_secret = self._get_env("BYBIT_API_SECRET", None, Fore.RED)
        self.ohlcv_limit = 200
        self.loop_sleep_seconds = 15
        self.order_check_delay_seconds = 2
        self.order_check_timeout_seconds = 10  # Max time to wait for order status check
        self.max_fetch_retries = 3
        self.trade_only_with_trend = self._get_env(
            "TRADE_ONLY_WITH_TREND", "True", Fore.YELLOW, cast_type=bool
        )  # Only trade in direction of trend_ema

        if not self.api_key or not self.api_secret:
            logger.error(
                Fore.RED
                + Style.BRIGHT
                + "BYBIT_API_KEY or BYBIT_API_SECRET not found in .env scroll!"
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
            log_value = "****" if "SECRET" in key else value
            logger.info(f"{color}Summoned {key}: {log_value}")

        try:
            if value is None:
                return None
            if cast_type == bool:
                return str(value).lower() in ["true", "1", "yes", "y"]
            return cast_type(value)
        except (ValueError, TypeError) as e:
            logger.error(
                f"{Fore.RED}Could not cast {key} ('{value}') to {cast_type.__name__}: {e}. Using default: {default}"
            )
            # Attempt to cast default if value failed
            try:
                if default is None:
                    return None
                if cast_type == bool:
                    return str(default).lower() in ["true", "1", "yes", "y"]
                return cast_type(default)
            except (ValueError, TypeError):
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
            "enableRateLimit": True,
        }
    )
    # Set market type based on config
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
        ][:10]
        logger.info(
            Fore.CYAN
            + f"Available active {CONFIG.market_type} symbols with {CONFIG.symbol.split('/')[1].split(':')[0]} quote (sample): "
            + ", ".join(available_symbols)
        )
        sys.exit(1)
    else:
        MARKET_INFO = exchange.market(CONFIG.symbol)
        logger.info(Fore.CYAN + f"Market spirit for {CONFIG.symbol} acknowledged.")
        # Log key precision and limits using Decimal where appropriate
        price_prec = MARKET_INFO["precision"]["price"]
        amount_prec = MARKET_INFO["precision"]["amount"]
        min_amount = MARKET_INFO["limits"]["amount"]["min"]
        max_amount = MARKET_INFO["limits"]["amount"]["max"]
        contract_size = MARKET_INFO.get(
            "contractSize", "1"
        )  # Default to '1' if not present

        logger.debug(f"Market Precision: Price={price_prec}, Amount={amount_prec}")
        logger.debug(f"Market Limits: Min Amount={min_amount}, Max Amount={max_amount}")
        logger.debug(f"Contract Size: {contract_size}")

        # Validate that we can convert these critical values
        try:
            Decimal(str(price_prec))
            Decimal(str(amount_prec))
            if min_amount is not None:
                Decimal(str(min_amount))
            if max_amount is not None:
                Decimal(str(max_amount))
            Decimal(str(contract_size))
        except Exception as e:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}Failed to parse critical market info (precision/limits/size) as numbers: {e}. Halting."
            )
            sys.exit(1)


except ccxt.AuthenticationError:
    logger.error(
        Fore.RED + Style.BRIGHT + "Authentication failed! Check your API keys."
    )
    sys.exit(1)
except ccxt.ExchangeError as e:
    logger.error(Fore.RED + Style.BRIGHT + f"Exchange Nexus Error: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(
        Fore.RED + Style.BRIGHT + f"Unexpected error during Nexus initialization: {e}",
        exc_info=True,
    )
    sys.exit(1)


# --- Global State Runes ---
order_tracker: dict[str, dict[str, str | None]] = {
    "long": {"sl_id": None, "tsl_id": None},
    "short": {"sl_id": None, "tsl_id": None},
}


# --- Termux Utility Spell ---
def termux_notify(title: str, content: str) -> None:
    """Sends a notification using Termux API (if available)."""
    if not sys.platform.startswith(
        "linux"
    ):  # Basic check if not on Linux (Termux runs on Linux kernel)
        logger.debug("Skipping Termux notification (not on Linux).")
        return
    try:
        # Use `command -v` for better check if command exists
        toast_cmd_check = os.system("command -v termux-toast > /dev/null 2>&1")
        if toast_cmd_check == 0:  # Command exists
            toast_cmd = "termux-toast"
            # Basic sanitization for shell command arguments
            safe_title = title.replace('"', "'").replace("`", "'").replace("$", "")
            safe_content = content.replace('"', "'").replace("`", "'").replace("$", "")
            # Construct command safely - avoid complex shell features in strings
            cmd = f'{toast_cmd} -g middle -c green "{safe_title}: {safe_content}"'
            os.system(cmd)
        else:
            logger.debug("termux-toast command not found. Skipping notification.")
    except Exception as e:
        logger.warning(Fore.YELLOW + f"Could not conjure Termux notification: {e}")


# --- Precision Casting Spells ---


def format_price(symbol: str, price: float | Decimal) -> str:
    """Formats price according to market precision rules using ROUND_DOWN."""
    if MARKET_INFO is None:
        logger.error(f"{Fore.RED}Market info not loaded, cannot format price.")
        return str(float(price))  # Fallback
    try:
        # Use price_to_precision which handles rounding according to exchange rules (often truncate/ROUND_DOWN)
        # CCXT methods usually expect float input
        return exchange.price_to_precision(symbol, float(price))
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting price {price} for {symbol}: {e}")
        return str(float(price))  # Fallback to float string


def format_amount(symbol: str, amount: float | Decimal) -> str:
    """Formats amount according to market precision rules using ROUND_DOWN."""
    if MARKET_INFO is None:
        logger.error(f"{Fore.RED}Market info not loaded, cannot format amount.")
        return str(float(amount))  # Fallback
    try:
        # Use amount_to_precision which handles rounding according to exchange rules (often truncate/ROUND_DOWN)
        # CCXT methods usually expect float input
        return exchange.amount_to_precision(symbol, float(amount))
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting amount {amount} for {symbol}: {e}")
        return str(float(amount))  # Fallback to float string


# --- Core Spell Functions ---


def fetch_market_data(
    symbol: str, timeframe: str, limit: int, retries: int = CONFIG.max_fetch_retries
) -> pd.DataFrame | None:
    """Fetch OHLCV data, handling transient errors with retries."""
    logger.info(
        Fore.CYAN + f"# Channeling market whispers for {symbol} ({timeframe})..."
    )
    for attempt in range(retries):
        try:
            # Check if exchange object is valid
            if not hasattr(exchange, "fetch_ohlcv"):
                logger.error(Fore.RED + "Exchange object not properly initialized.")
                return None

            ohlcv: list[list[int | float]] = exchange.fetch_ohlcv(
                symbol, timeframe, limit=limit
            )
            if not ohlcv:
                logger.warning(
                    Fore.YELLOW
                    + f"Received empty OHLCV data (Attempt {attempt + 1}/{retries})."
                )
                if attempt < retries - 1:
                    time.sleep(1 * (attempt + 1))  # Simple backoff
                    continue
                else:
                    logger.error(
                        Fore.RED
                        + f"Received empty OHLCV data after {retries} attempts."
                    )
                    return None

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            # Convert to numeric, coercing errors (should not happen with valid API data)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df.dropna(
                subset=["open", "high", "low", "close"], inplace=True
            )  # Drop only if price data is missing

            if df.empty:
                logger.error(
                    Fore.RED + "DataFrame is empty after processing OHLCV data."
                )
                return None

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
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
            logger.error(Fore.RED + f"Exchange rejected data request: {e}")
            return None
        except Exception as e:
            logger.error(
                Fore.RED + f"Unexpected shadow encountered fetching data: {e}",
                exc_info=True,
            )
            return None
    return None


def calculate_indicators(df: pd.DataFrame) -> dict[str, Decimal] | None:
    """Calculate technical indicators, returning results as Decimals for precision."""
    logger.info(Fore.CYAN + "# Weaving indicator patterns...")
    if df.empty:
        logger.error(Fore.RED + "Cannot calculate indicators on empty DataFrame.")
        return None
    try:
        # Ensure data is float for calculations, convert to Decimal at the end
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        # EMAs
        fast_ema_series = close.ewm(span=8, adjust=False).mean()
        slow_ema_series = close.ewm(span=12, adjust=False).mean()
        trend_ema_series = close.ewm(span=22, adjust=False).mean()
        confirm_ema_series = close.ewm(span=5, adjust=False).mean()

        # Stochastic Oscillator (%K, %D)
        period = 10
        smooth_k = 3
        smooth_d = 3
        if len(df) < period:
            logger.warning(
                f"Not enough data ({len(df)}) for Stochastic period {period}. Skipping Stoch."
            )
            k_now, d_now = 50.0, 50.0  # Default neutral values
        else:
            low_min = low.rolling(window=period).min()
            high_max = high.rolling(window=period).max()
            # Add epsilon to prevent division by zero if high_max == low_min
            stoch_k_raw = 100 * (close - low_min) / (high_max - low_min + 1e-12)
            stoch_k = stoch_k_raw.rolling(window=smooth_k).mean()
            stoch_d = stoch_k.rolling(window=smooth_d).mean()
            k_now = stoch_k.iloc[-1]
            d_now = stoch_d.iloc[-1]
            if pd.isna(k_now):
                k_now = 50.0  # Handle potential NaN at start
            if pd.isna(d_now):
                d_now = 50.0  # Handle potential NaN at start

        # ATR (Average True Range)
        atr_period = 10
        if len(df) < atr_period + 1:  # Need at least period+1 for shift()
            logger.warning(
                f"Not enough data ({len(df)}) for ATR period {atr_period}. Skipping ATR."
            )
            atr = 0.0  # Default to 0 if cannot calculate
        else:
            tr_df = pd.DataFrame(index=df.index)
            tr_df["hl"] = high - low
            tr_df["hc"] = (high - close.shift()).abs()
            tr_df["lc"] = (low - close.shift()).abs()
            tr_df["tr"] = tr_df[["hl", "hc", "lc"]].max(axis=1)
            # Use Exponential Moving Average for ATR for smoother results, common practice
            atr_series = tr_df["tr"].ewm(alpha=1 / atr_period, adjust=False).mean()
            atr = atr_series.iloc[-1]
            if pd.isna(atr):
                atr = 0.0  # Handle potential NaN at start

        logger.info(Fore.GREEN + "Indicator patterns woven successfully.")
        # Convert final indicator values to Decimal, handling potential NaN from calculations
        # Use '.quantize' for safety if needed, but direct str conversion is usually fine
        return {
            "fast_ema": Decimal(str(fast_ema_series.iloc[-1])).quantize(
                Decimal("0.00000001")
            )
            if not pd.isna(fast_ema_series.iloc[-1])
            else Decimal(0),
            "slow_ema": Decimal(str(slow_ema_series.iloc[-1])).quantize(
                Decimal("0.00000001")
            )
            if not pd.isna(slow_ema_series.iloc[-1])
            else Decimal(0),
            "trend_ema": Decimal(str(trend_ema_series.iloc[-1])).quantize(
                Decimal("0.00000001")
            )
            if not pd.isna(trend_ema_series.iloc[-1])
            else Decimal(0),
            "confirm_ema": Decimal(str(confirm_ema_series.iloc[-1])).quantize(
                Decimal("0.00000001")
            )
            if not pd.isna(confirm_ema_series.iloc[-1])
            else Decimal(0),
            "stoch_k": Decimal(str(k_now)).quantize(Decimal("0.01")),
            "stoch_d": Decimal(str(d_now)).quantize(Decimal("0.01")),
            "atr": Decimal(str(atr)).quantize(Decimal("0.00000001")),
        }
    except Exception as e:
        logger.error(
            Fore.RED + f"Failed to weave indicator patterns: {e}", exc_info=True
        )
        return None


def get_current_position(symbol: str) -> dict[str, dict[str, Any]] | None:
    """Fetch current positions, returning quantities and prices as Decimals."""
    logger.info(Fore.CYAN + f"# Consulting position spirits for {symbol}...")
    try:
        # Bybit specific parameter to get positions for one symbol - use the exact symbol format from MARKET_INFO
        # market_symbol_id = MARKET_INFO['id'] if MARKET_INFO else symbol.replace('/', '').replace(':','')
        # params = {'symbol': market_symbol_id}
        # Fetching without params often works for unified symbols if not in hedge mode
        positions = exchange.fetch_positions(
            symbols=[symbol]
        )  # Fetch for specific unified symbol

        # Initialize with Decimal zero
        pos_dict = {
            "long": {"qty": Decimal("0.0"), "entry_price": Decimal("0.0")},
            "short": {"qty": Decimal("0.0"), "entry_price": Decimal("0.0")},
        }

        if not positions:
            logger.info(Fore.BLUE + f"No open positions found for {symbol}.")
            return pos_dict

        # Filter positions for the exact symbol requested
        symbol_positions = [p for p in positions if p.get("symbol") == symbol]

        if not symbol_positions:
            logger.info(
                Fore.BLUE
                + f"No matching position details found for {symbol} in fetched data."
            )
            return pos_dict

        # Logic assumes non-hedge mode or aggregates hedge mode positions (summing might be complex)
        # For Bybit unified margin, often only one entry per symbol/side exists.
        active_positions_found = 0
        for pos in symbol_positions:
            side = pos.get("side")  # 'long' or 'short'
            contracts_str = pos.get("contracts")  # Amount of contracts/base currency
            entry_price_str = pos.get("entryPrice")

            if side in pos_dict and contracts_str is not None:
                contracts = Decimal(str(contracts_str))
                # Use epsilon to check if effectively zero
                if contracts.copy_abs() < CONFIG.position_qty_epsilon:
                    logger.debug(
                        f"Skipping effectively zero size {side} position for {symbol}."
                    )
                    continue

                entry_price = (
                    Decimal(str(entry_price_str))
                    if entry_price_str is not None
                    else Decimal("0.0")
                )

                # If hedge mode is possible, this might overwrite. Assuming single position per side.
                pos_dict[side]["qty"] = contracts
                pos_dict[side]["entry_price"] = entry_price
                logger.info(
                    Fore.YELLOW
                    + f"Found active {side} position: Qty={contracts}, Entry={entry_price}"
                )
                active_positions_found += 1

        if active_positions_found == 0:
            logger.info(Fore.BLUE + f"No active non-zero positions found for {symbol}.")

        logger.info(Fore.GREEN + "Position spirits consulted.")
        return pos_dict

    except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
        logger.warning(
            Fore.YELLOW
            + f"Network disturbance consulting position spirits: {e}. Cannot reliably get position."
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


def get_balance(currency: str = "USDT") -> tuple[Decimal | None, Decimal | None]:
    """Fetches the free and total balance for a specific currency as Decimals."""
    logger.info(Fore.CYAN + f"# Querying the Vault of {currency}...")
    try:
        balance = exchange.fetch_balance()
        # Use Decimal for balances
        free_balance_str = balance.get("free", {}).get(currency)
        total_balance_str = balance.get("total", {}).get(currency)

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
    except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
        logger.warning(
            Fore.YELLOW
            + f"Network disturbance querying vault: {e}. Cannot assess risk capital."
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
) -> dict | None:
    """Checks order status with retries and timeout. Returns the order dict or None."""
    logger.info(Fore.CYAN + f"Verifying status of order {order_id} for {symbol}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            order_status = exchange.fetch_order(order_id, symbol)
            if order_status:
                status = order_status.get("status")
                logger.info(f"Order {order_id} status: {status}")
                # Return the full order dict if found, regardless of status initially
                return order_status
            else:
                # This case might indicate the order *was* found but the structure is empty/unexpected
                logger.warning(
                    f"fetch_order returned empty/unexpected structure for {order_id}. Retrying..."
                )

        except ccxt.OrderNotFound:
            # Order is definitively not found on the exchange (could be cancelled, filled & archived quickly, or never existed)
            logger.error(Fore.RED + f"Order {order_id} not found by exchange.")
            # Depending on context, this might mean it filled or failed. Returning None indicates it's not 'open'.
            return None  # Explicitly indicate not found

        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
            logger.warning(f"Network issue checking order {order_id}: {e}. Retrying...")
        except ccxt.ExchangeError as e:
            # Exchange error likely means the request failed, not necessarily that the order failed.
            logger.error(f"Exchange error checking order {order_id}: {e}. Retrying...")
            # Consider if specific exchange errors should terminate the check
        except Exception as e:
            logger.error(
                f"Unexpected error checking order {order_id}: {e}", exc_info=True
            )
            # Treat unexpected errors cautiously, maybe retry

        # Wait before retrying
        check_interval = 1  # seconds
        # Ensure we don't sleep past the timeout
        if time.time() - start_time + check_interval < timeout:
            time.sleep(check_interval)
        else:
            break  # Exit loop if next sleep would exceed timeout

    logger.error(
        Fore.RED
        + f"Timed out checking status for order {order_id} after {timeout} seconds."
    )
    return None  # Indicate timeout or persistent failure to get status


def place_risked_market_order(
    symbol: str, side: str, risk_percentage: Decimal, atr: Decimal
) -> bool:
    """Places a market order with calculated size and initial ATR-based stop-loss, using Decimal precision."""
    logger.info(
        Fore.BLUE
        + Style.BRIGHT
        + f"Preparing {side.upper()} market incantation for {symbol}..."
    )

    if MARKET_INFO is None:
        logger.error(Fore.RED + "Market info not available. Cannot place order.")
        return False

    free_balance, _ = get_balance(
        "USDT"
    )  # Assuming USDT is the quote currency for risk calc
    if free_balance is None or free_balance <= Decimal("0"):
        logger.error(
            Fore.RED + "Cannot place order: Invalid or zero available balance."
        )
        return False

    if atr is None or atr <= Decimal("0"):
        logger.error(Fore.RED + f"Cannot place order: Invalid ATR value ({atr}).")
        return False

    try:
        ticker = exchange.fetch_ticker(symbol)
        price_str = ticker.get("last")
        if price_str is None:
            logger.error(Fore.RED + "Cannot fetch current price for sizing.")
            return False
        price = Decimal(str(price_str))

        # Calculate Stop Loss Price based on ATR
        sl_distance_points = CONFIG.sl_atr_multiplier * atr
        if side == "buy":
            sl_price_raw = price - sl_distance_points
        else:  # side == "sell"
            sl_price_raw = price + sl_distance_points

        # Format SL price according to market precision *before* using it in calculations
        sl_price_formatted_str = format_price(symbol, sl_price_raw)
        sl_price = Decimal(sl_price_formatted_str)  # Use the formatted price as Decimal
        logger.debug(
            f"Current Price: {price}, ATR: {atr:.6f}, SL Distance Pts: {sl_distance_points:.6f}"
        )
        logger.debug(f"Raw SL Price: {sl_price_raw}, Formatted SL Price: {sl_price}")

        # Ensure SL is not triggered immediately (e.g., due to large spread or ATR)
        if side == "buy" and sl_price >= price:
            logger.error(
                Fore.RED
                + f"Calculated SL price ({sl_price}) is >= current price ({price}). Aborting."
            )
            return False
        if side == "sell" and sl_price <= price:
            logger.error(
                Fore.RED
                + f"Calculated SL price ({sl_price}) is <= current price ({price}). Aborting."
            )
            return False

        # Calculate Position Size based on Risk (using Decimal)
        risk_amount_usd = free_balance * risk_percentage
        # Stop distance in USD (absolute difference between entry and SL price)
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
        contract_size = Decimal(str(MARKET_INFO.get("contractSize", "1")))
        qty_raw = Decimal("0")

        if CONFIG.market_type == "linear":
            # Qty = (Risk Amount in Quote) / (Stop Distance in Quote) / (Contract Size in Base)
            # Assumes contract value is directly tied to price (e.g., 1 contract = contract_size * price)
            # More simply: Risk Amount / (Stop Distance * Contract Size) if contract size is in base units
            # Example: BTC/USDT, contractSize=1. Risk $10, Stop $100. Qty = 10 / 100 = 0.1 BTC
            # Example: BTC/USDT, contractSize=0.001. Risk $10, Stop $100. Qty = 10 / (100 * 0.001) = 100 contracts
            if contract_size == Decimal("1"):
                qty_raw = risk_amount_usd / stop_distance_usd
            else:
                # If contract size is not 1, need to factor it in.
                # Value per contract move = contract_size * 1 (for linear)
                # Qty in contracts = Risk Amount / (Stop Distance * Contract Size)
                qty_raw = risk_amount_usd / (stop_distance_usd * contract_size)

        elif CONFIG.market_type == "inverse":
            # Qty = (Risk Amount in Quote * Entry Price) / (Stop Distance in Quote) / (Contract Value)
            # Or simplified: Qty (in contracts) = Risk Amount in Base / Stop Distance in Base
            # Risk Amount in Base = risk_amount_usd / price
            # Stop Distance in Base = stop_distance_usd / price (approximation using entry price)
            # Qty ~= (risk_amount_usd / price) / (stop_distance_usd / price) = risk_amount_usd / stop_distance_usd
            # This seems too simple - Inverse contract sizing is complex. Let's use CCXT's built-in if possible, or stick to linear for now.
            # For Bybit inverse (e.g. BTC/USD), size is in contracts (usually 1 USD worth).
            # Qty (Contracts) = Risk Amount (USD) / (Stop Distance (USD/BTC) * Contract Value (BTC/Contract))
            # Contract Value might be fixed (e.g., $1) or variable.
            # Assuming fixed contract value of $1 for simplicity (Check Bybit docs!)
            # Qty (contracts) ~= Risk Amount (USD) / Stop Distance (USD) * Price (USD/BTC) ?? This seems off.
            # Let's assume qty is Risk Amount / Stop Distance for now, similar to linear, and note this needs verification for inverse.
            logger.warning(
                Fore.YELLOW
                + "Inverse contract sizing calculation is simplified and may need adjustment based on specific contract details."
            )
            qty_raw = (
                risk_amount_usd / stop_distance_usd
            )  # Simplified - NEEDS VERIFICATION FOR INVERSE

        else:
            logger.error(f"Unsupported market type for sizing: {CONFIG.market_type}")
            return False

        # Format quantity according to market precision (ROUND_DOWN)
        qty_formatted_str = format_amount(symbol, qty_raw)
        qty = Decimal(qty_formatted_str)
        logger.debug(
            f"Risk Amount: {risk_amount_usd:.4f} USDT, Stop Distance: {stop_distance_usd:.4f} USDT"
        )
        logger.debug(f"Raw Qty: {qty_raw}, Formatted Qty: {qty}")

        # Validate quantity against market limits
        min_qty_str = MARKET_INFO.get("limits", {}).get("amount", {}).get("min")
        max_qty_str = MARKET_INFO.get("limits", {}).get("amount", {}).get("max")
        min_qty = Decimal(str(min_qty_str)) if min_qty_str is not None else None
        max_qty = Decimal(str(max_qty_str)) if max_qty_str is not None else None

        if qty.is_zero() or qty < CONFIG.position_qty_epsilon:
            logger.error(
                Fore.RED
                + f"Calculated quantity ({qty}) is zero or too small after precision formatting. Risk amount, price movement, or ATR might be too small."
            )
            return False
        if min_qty is not None and qty < min_qty:
            logger.error(
                Fore.RED
                + f"Calculated quantity {qty} is below minimum {min_qty}. Cannot place order."
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

        logger.info(
            Fore.YELLOW
            + f"Calculated Order: Side={side.upper()}, Qty={qty}, Entry≈{price:.4f}, SL={sl_price:.4f} (ATR={atr:.4f})"
        )

        # --- Cast the Market Order Spell ---
        logger.info(
            Fore.CYAN + f"Submitting {side.upper()} market order for {qty} {symbol}..."
        )
        order_params = {}  # No extra params needed for basic market order
        order = exchange.create_market_order(
            symbol, side, float(qty), params=order_params
        )  # CCXT expects float amount
        order_id = order.get("id")
        logger.info(Fore.CYAN + f"Market order submitted: ID {order_id}")
        if not order_id:
            logger.error(Fore.RED + "Market order submission failed to return an ID.")
            return False

        # --- Verify Order Fill (Crucial Step) ---
        logger.info(
            f"Waiting {CONFIG.order_check_delay_seconds}s before checking order status..."
        )
        time.sleep(CONFIG.order_check_delay_seconds)  # Allow time for potential fill
        order_status_data = check_order_status(
            order_id, symbol, timeout=CONFIG.order_check_timeout_seconds
        )

        filled_qty = Decimal("0.0")
        average_price = price  # Fallback to estimated entry price
        order_final_status = "unknown"

        if order_status_data:
            order_final_status = order_status_data.get("status", "unknown")
            filled_str = order_status_data.get("filled")
            average_str = order_status_data.get("average")

            if filled_str is not None:
                filled_qty = Decimal(str(filled_str))
            if average_str is not None:
                average_price = Decimal(
                    str(average_str)
                )  # Use actual fill price if available

            if (
                order_final_status == "closed"
            ):  # 'closed' usually means fully filled for market orders
                logger.info(
                    Fore.GREEN
                    + Style.BRIGHT
                    + f"Order {order_id} confirmed filled: {filled_qty} @ {average_price:.4f}"
                )
            elif order_final_status in ["open", "partially_filled"]:
                # Market orders shouldn't stay 'open' long, but handle partial fills
                logger.warning(
                    Fore.YELLOW
                    + f"Order {order_id} partially filled or status unclear: Status '{order_final_status}', Filled {filled_qty}. SL will be based on filled amount."
                )
                if filled_qty < CONFIG.position_qty_epsilon:
                    logger.error(
                        Fore.RED
                        + f"Order {order_id} has status '{order_final_status}' but filled quantity is effectively zero ({filled_qty}). Aborting SL placement."
                    )
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

        # --- Place Initial Stop-Loss Order ---
        position_side = "long" if side == "buy" else "short"
        sl_order_side = "sell" if side == "buy" else "buy"

        # Format SL price and filled quantity correctly for the SL order
        # Use the SL price calculated earlier based on estimated entry
        sl_price_str_for_order = format_price(symbol, sl_price)
        # Use the *actual filled quantity* for the SL order size
        sl_qty_str_for_order = format_amount(symbol, filled_qty)

        sl_params = {
            "stopLossPrice": sl_price_str_for_order,  # Trigger price for the stop market order
            "reduceOnly": True,
            "triggerPrice": sl_price_str_for_order,  # Some exchanges might use this param name or require it
            "triggerBy": CONFIG.sl_trigger_by,  # e.g., 'LastPrice', 'MarkPrice'
            # Bybit specific potentially useful params (check CCXT unification)
            # 'tpslMode': 'Full', # or 'Partial' - affects if TP/SL apply to whole position
            # 'slTriggerBy': CONFIG.sl_trigger_by, # More specific param if available
            # 'positionIdx': 0 # For Bybit unified: 0 for one-way, 1 for long hedge, 2 for short hedge
        }
        logger.info(
            Fore.CYAN
            + f"Placing SL order: Side={sl_order_side}, Qty={sl_qty_str_for_order}, Trigger={sl_price_str_for_order}, TriggerBy={CONFIG.sl_trigger_by}"
        )
        logger.debug(f"SL Params: {sl_params}")

        try:
            # Use create_order with stop type. CCXT standard is often 'stop_market' or 'stop'.
            # Check exchange.has['createStopMarketOrder'] etc. if needed.
            # create_order is the unified method.
            sl_order = exchange.create_order(
                symbol=symbol,
                type="stop_market",  # Use 'stop_market' for market SL, 'stop_limit' for limit SL
                side=sl_order_side,
                amount=float(sl_qty_str_for_order),  # CCXT expects float amount
                price=None,  # Market stop loss doesn't need a limit price
                params=sl_params,
            )
            sl_order_id = sl_order.get("id")
            if not sl_order_id:
                raise ccxt.ExchangeError(
                    "Stop loss order placement did not return an ID."
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
                f"ENTERED {side.upper()} {filled_qty} @ {average_price:.4f}. "
                f"Initial SL @ {sl_price_str_for_order} (ID: {sl_order_id}). TSL pending profit threshold."
            )
            logger.info(Back.BLUE + Fore.WHITE + Style.BRIGHT + entry_msg)
            termux_notify(
                "Trade Entry", f"{side.upper()} {symbol} @ {average_price:.4f}"
            )
            return True

        except ccxt.InsufficientFunds as e:
            logger.error(
                Fore.RED
                + Style.BRIGHT
                + f"Insufficient funds to place stop-loss order: {e}. Position is UNPROTECTED."
            )
            logger.warning(
                Fore.YELLOW + "Attempting emergency closure of unprotected position..."
            )
            try:
                # Use the same filled quantity and opposite side
                close_qty_str = format_amount(symbol, filled_qty)
                exchange.create_market_order(
                    symbol,
                    sl_order_side,
                    float(close_qty_str),
                    params={"reduceOnly": True},
                )
                logger.info(Fore.GREEN + "Emergency closure order placed.")
            except Exception as close_err:
                logger.critical(
                    Fore.RED
                    + Style.BRIGHT
                    + f"EMERGENCY CLOSURE FAILED: {close_err}. MANUAL INTERVENTION REQUIRED!"
                )
            return False  # Signal failure even if closure attempted

        except (ccxt.ExchangeError, ccxt.InvalidOrder) as e:
            logger.error(
                Fore.RED
                + Style.BRIGHT
                + f"Failed to place initial SL order: {e}. Position might be UNPROTECTED."
            )
            logger.warning(
                Fore.YELLOW
                + "Position may be open without Stop Loss due to SL placement error. Consider emergency closure."
            )
            # Optionally trigger emergency closure here as well
            return False  # Signal failure
        except Exception as e:
            logger.error(
                Fore.RED + Style.BRIGHT + f"Unexpected error placing SL: {e}",
                exc_info=True,
            )
            # Optionally trigger emergency closure
            return False

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
        return False
    except Exception as e:
        logger.error(
            Fore.RED + Style.BRIGHT + f"Unexpected error during order placement: {e}",
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
    if position_qty < CONFIG.position_qty_epsilon or entry_price <= Decimal("0"):
        # Clear potentially stale trackers if position is confirmed closed or invalid
        if (
            order_tracker[position_side]["tsl_id"]
            or order_tracker[position_side]["sl_id"]
        ):
            logger.debug(
                f"Position {position_side} closed or invalid, clearing order trackers."
            )
            order_tracker[position_side] = {"sl_id": None, "tsl_id": None}
        return

    has_initial_sl = order_tracker[position_side]["sl_id"] is not None
    has_tsl = order_tracker[position_side]["tsl_id"] is not None

    # If TSL is already active, the exchange handles the trailing. No updates needed from script side for basic TSL.
    if has_tsl:
        logger.debug(
            f"{position_side.upper()} TSL (ID: {order_tracker[position_side]['tsl_id']}) is already active. Exchange is managing trail."
        )
        # Could add logic here to check if the TSL order still exists on the exchange, but adds complexity/API calls.
        return

    # --- Check for TSL Activation Condition ---
    if atr is None or atr <= Decimal("0"):
        logger.warning(
            Fore.YELLOW + "Cannot evaluate TSL activation without valid ATR."
        )
        return

    profit = Decimal("0.0")
    if position_side == "long":
        profit = current_price - entry_price
    else:  # short
        profit = entry_price - current_price

    activation_threshold_points = CONFIG.tsl_activation_atr_multiplier * atr
    logger.debug(
        f"{position_side.upper()} Profit: {profit:.4f}, TSL Activation Threshold (Points): {activation_threshold_points:.4f}"
    )

    # Activate TSL only if profit exceeds the threshold AND we still have the initial SL active (meaning TSL hasn't been set yet)
    if profit > activation_threshold_points and has_initial_sl:
        logger.info(
            Fore.GREEN
            + Style.BRIGHT
            + f"Profit threshold reached for {position_side.upper()} position. Activating TSL."
        )

        # --- Cancel Initial SL before placing TSL ---
        initial_sl_id = order_tracker[position_side]["sl_id"]
        logger.info(
            Fore.CYAN
            + f"Attempting to cancel initial SL (ID: {initial_sl_id}) before placing TSL..."
        )
        try:
            exchange.cancel_order(initial_sl_id, symbol)
            logger.info(
                Fore.GREEN + f"Successfully cancelled initial SL (ID: {initial_sl_id})."
            )
            order_tracker[position_side]["sl_id"] = None  # Mark as cancelled locally
        except ccxt.OrderNotFound:
            logger.warning(
                Fore.YELLOW
                + f"Initial SL (ID: {initial_sl_id}) not found when trying to cancel. Might have been triggered or already cancelled."
            )
            order_tracker[position_side]["sl_id"] = None  # Assume it's gone
        except (ccxt.ExchangeError, ccxt.NetworkError) as e:
            logger.error(
                Fore.RED
                + f"Failed to cancel initial SL (ID: {initial_sl_id}): {e}. Proceeding with TSL placement cautiously, but risk of double orders exists."
            )
            # Decide: Abort TSL? Or place TSL hoping the initial SL is gone? Placing TSL might be safer than no stop.
            # For now, proceed with TSL placement attempt but log the risk.
        except Exception as e:
            logger.error(
                Fore.RED + f"Unexpected error cancelling initial SL: {e}", exc_info=True
            )
            # Proceed with TSL placement attempt

        # --- Place Trailing Stop Loss Order ---
        tsl_order_side = "sell" if position_side == "long" else "buy"
        tsl_qty_str = format_amount(
            symbol, position_qty
        )  # Use current position quantity

        # Convert Decimal percentage (e.g., 0.5) to float for CCXT param
        trail_percent_value = float(CONFIG.trailing_stop_percent)

        # Check CCXT documentation and exchange capabilities for trailing stop parameters
        # Common parameters: 'trailingPercent', 'trailingAmount', 'activationPrice'
        tsl_params = {
            "reduceOnly": True,
            "triggerBy": CONFIG.tsl_trigger_by,  # Use configured trigger type
            "trailingPercent": trail_percent_value,  # CCXT standard parameter for percentage-based trail
            # 'activationPrice': format_price(symbol, current_price) # Optional: Price at which the trail *starts*. Some exchanges require/support this.
            # If not provided, trail might start immediately or based on trigger price rules.
            # Bybit specific params (check if needed/mapped by CCXT):
            # 'tpslMode': 'Full',
            # 'trailingStop': str(CONFIG.trailing_stop_percent / 100), # Bybit might expect percentage value like '0.005' for 0.5% ? Check API docs. CCXT usually handles conversion.
            # 'activePrice': format_price(symbol, current_price), # Bybit's activation price
            # 'positionIdx': 0
        }
        logger.info(
            Fore.CYAN
            + f"Placing TSL order: Side={tsl_order_side}, Qty={tsl_qty_str}, Trail%={trail_percent_value}, TriggerBy={CONFIG.tsl_trigger_by}"
        )
        logger.debug(f"TSL Params: {tsl_params}")

        try:
            # Use create_order with the specific type for trailing stops if available and preferred
            # Check exchange.has['createTrailingStopMarketOrder'] or similar
            # Otherwise, use 'stop_market' or 'market' with trailing params if supported by unified method
            tsl_order = exchange.create_order(
                symbol=symbol,
                type="trailing_stop_market",  # Prefer specific type if unified by CCXT for Bybit
                side=tsl_order_side,
                amount=float(tsl_qty_str),  # CCXT expects float amount
                price=None,  # Market based trail
                params=tsl_params,
            )
            tsl_order_id = tsl_order.get("id")
            if not tsl_order_id:
                raise ccxt.ExchangeError(
                    "Trailing stop order placement did not return an ID."
                )

            order_tracker[position_side]["tsl_id"] = tsl_order_id
            order_tracker[position_side]["sl_id"] = (
                None  # Ensure initial SL ID is cleared
            )
            logger.info(
                Fore.GREEN
                + Style.BRIGHT
                + f"Trailing Stop Loss activated for {position_side.upper()}: ID {tsl_order_id}, Trail: {trail_percent_value}%"
            )
            termux_notify(
                "TSL Activated", f"{position_side.upper()} {symbol} TSL active."
            )

        except (ccxt.ExchangeError, ccxt.InvalidOrder) as e:
            # If 'trailing_stop_market' type fails, maybe the exchange/CCXT requires 'stop_market' with trailing params?
            logger.error(
                Fore.RED
                + Style.BRIGHT
                + f"Failed to place TSL order (tried type 'trailing_stop_market'): {e}"
            )
            logger.warning(
                Fore.YELLOW
                + "Position might be unprotected after failed TSL placement. Initial SL was likely cancelled. MANUAL INTERVENTION MAY BE NEEDED."
            )
            # Reset local tracker as TSL failed
            order_tracker[position_side]["tsl_id"] = None
            # CRITICAL: At this point, the initial SL might be cancelled, and TSL failed.
            # Consider placing a new *regular* stop loss as a fallback?
            # place_fallback_stop_loss(symbol, position_side, position_qty, current_price) # Example function call
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


def print_status_panel(
    cycle: int,
    timestamp: pd.Timestamp,
    price: Decimal,
    indicators: dict[str, Decimal],
    positions: dict[str, dict[str, Any]],
    equity: Decimal,
    signals: dict[str, bool],
    order_tracker_state: dict[
        str, dict[str, str | None]
    ],  # Pass tracker state explicitly
) -> None:
    """Displays the current state using a mystical status panel with Decimal precision."""
    # Market & Indicators
    trend_ema = indicators.get("trend_ema", Decimal(0))
    price_color = (
        Fore.GREEN
        if price > trend_ema
        else Fore.RED
        if price < trend_ema
        else Fore.WHITE
    )
    stoch_k = indicators.get("stoch_k", Decimal(50))
    stoch_d = indicators.get("stoch_d", Decimal(50))
    stoch_color = (
        Fore.GREEN
        if stoch_k < Decimal(25)
        else Fore.RED
        if stoch_k > Decimal(75)
        else Fore.YELLOW
    )
    fast_ema = indicators.get("fast_ema", Decimal(0))
    slow_ema = indicators.get("slow_ema", Decimal(0))
    ema_cross_color = (
        Fore.GREEN
        if fast_ema > slow_ema
        else Fore.RED
        if fast_ema < slow_ema
        else Fore.WHITE
    )

    [
        [Fore.CYAN + "Market", Fore.WHITE + CONFIG.symbol, f"{price_color}{price:.4f}"],
        [
            Fore.CYAN + "ATR",
            f"{Fore.WHITE}{indicators.get('atr', Decimal(0)):.6f}",
            "",
        ],  # More precision for ATR
        [
            Fore.CYAN + "EMA Fast/Slow",
            f"{ema_cross_color}{fast_ema:.4f} / {slow_ema:.4f}",
            f"{Fore.GREEN + ' bullish' if ema_cross_color == Fore.GREEN else Fore.RED + ' bearish' if ema_cross_color == Fore.RED else ''}",
        ],
        [
            Fore.CYAN + "EMA Trend",
            f"{Fore.WHITE}{trend_ema:.4f}",
            f"{price_color}{'(Above)' if price > trend_ema else '(Below)' if price < trend_ema else '(At)'}",
        ],
        [
            Fore.CYAN + "Stoch %K/%D",
            f"{stoch_color}{stoch_k:.2f} / {stoch_d:.2f}",
            f"{Fore.GREEN + ' Oversold' if stoch_color == Fore.GREEN else Fore.RED + ' Overbought' if stoch_color == Fore.RED else ''}",
        ],
    ]

    # Positions & Orders
    long_pos = positions.get(
        "long", {"qty": Decimal("0.0"), "entry_price": Decimal("0.0")}
    )
    short_pos = positions.get(
        "short", {"qty": Decimal("0.0"), "entry_price": Decimal("0.0")}
    )

    # Use the passed tracker state
    long_sl_id = order_tracker_state["long"]["sl_id"]
    long_tsl_id = order_tracker_state["long"]["tsl_id"]
    short_sl_id = order_tracker_state["short"]["sl_id"]
    short_tsl_id = order_tracker_state["short"]["tsl_id"]

    # Determine SL/TSL status strings
    long_stop_status = Fore.RED + "None"
    if long_tsl_id:
        long_stop_status = f"{Fore.GREEN}TSL Active (ID: ...{long_tsl_id[-6:]})"
    elif long_sl_id:
        long_stop_status = f"{Fore.YELLOW}SL Active (ID: ...{long_sl_id[-6:]})"

    short_stop_status = Fore.RED + "None"
    if short_tsl_id:
        short_stop_status = f"{Fore.GREEN}TSL Active (ID: ...{short_tsl_id[-6:]})"
    elif short_sl_id:
        short_stop_status = f"{Fore.YELLOW}SL Active (ID: ...{short_sl_id[-6:]})"

    [
        [Fore.CYAN + "Position", Fore.GREEN + "LONG", Fore.RED + "SHORT"],
        [
            Fore.CYAN + "Quantity",
            f"{Fore.WHITE}{long_pos['qty']}",
            f"{Fore.WHITE}{short_pos['qty']}",
        ],
        [
            Fore.CYAN + "Entry Price",
            f"{Fore.WHITE}{long_pos['entry_price']:.4f}",
            f"{Fore.WHITE}{short_pos['entry_price']:.4f}",
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

    try:
        # Use .get with default Decimal values for safety
        k = indicators.get("stoch_k", Decimal(50))
        indicators.get("stoch_d", Decimal(50))
        fast_ema = indicators.get("fast_ema", Decimal(0))
        slow_ema = indicators.get("slow_ema", Decimal(0))
        trend_ema = indicators.get("trend_ema", Decimal(0))

        # Define conditions using Decimal comparisons
        ema_bullish_cross = fast_ema > slow_ema
        ema_bearish_cross = fast_ema < slow_ema
        price_above_trend = current_price > trend_ema
        price_below_trend = current_price < trend_ema
        # Use Decimal for thresholds
        stoch_oversold = k < Decimal(
            25
        )  # Basic level check, consider adding 'and d < 25' or crossover
        stoch_overbought = k > Decimal(
            75
        )  # Basic level check, consider adding 'and d > 75' or crossover

        # --- Basic Signal Logic ---

        # Long Signal: Bullish EMA cross + Stoch Oversold
        if ema_bullish_cross and stoch_oversold:
            if CONFIG.trade_only_with_trend:
                if price_above_trend:
                    long_signal = True
                    logger.debug(
                        "Long Signal Criteria Met: EMA Cross Bullish, Stoch Oversold, Price Above Trend EMA"
                    )
                else:
                    logger.debug(
                        "Long Signal Blocked: Price Below Trend EMA (Trend Filter ON)"
                    )
            else:  # Trend filter off
                long_signal = True
                logger.debug(
                    "Long Signal Criteria Met: EMA Cross Bullish, Stoch Oversold (Trend Filter OFF)"
                )

        # Short Signal: Bearish EMA cross + Stoch Overbought
        if ema_bearish_cross and stoch_overbought:
            if CONFIG.trade_only_with_trend:
                if price_below_trend:
                    short_signal = True
                    logger.debug(
                        "Short Signal Criteria Met: EMA Cross Bearish, Stoch Overbought, Price Below Trend EMA"
                    )
                else:
                    logger.debug(
                        "Short Signal Blocked: Price Above Trend EMA (Trend Filter ON)"
                    )
            else:  # Trend filter off
                short_signal = True
                logger.debug(
                    "Short Signal Criteria Met: EMA Cross Bearish, Stoch Overbought (Trend Filter OFF)"
                )

        # Refinement Ideas (Not Implemented):
        # - Require Stoch K/D crossover in oversold/overbought zones.
        # - Add volume confirmation.
        # - Add divergence checks.
        # - Check if EMAs are separating (momentum).

    except Exception as e:
        logger.error(f"{Fore.RED}Error generating signals: {e}", exc_info=True)
        return {"long": False, "short": False}

    return {"long": long_signal, "short": short_signal}


def trading_spell_cycle(cycle_count: int) -> None:
    """Executes one cycle of the trading spell with enhanced precision and logic."""
    logger.info(Fore.MAGENTA + Style.BRIGHT + f"\n--- Starting Cycle {cycle_count} ---")

    # 1. Fetch Market Data
    df = fetch_market_data(CONFIG.symbol, CONFIG.interval, CONFIG.ohlcv_limit)
    if df is None or df.empty:
        logger.error(Fore.RED + "Halting cycle: Market data fetch failed.")
        return

    # Use Decimal for current price from the latest candle
    try:
        current_price_float = df["close"].iloc[-1]
        current_price = Decimal(str(current_price_float))
        last_timestamp = df.index[-1]
        logger.debug(f"Latest candle close: {current_price:.4f} at {last_timestamp}")
    except IndexError:
        logger.error(Fore.RED + "Failed to get current price from DataFrame.")
        return
    except Exception as e:
        logger.error(Fore.RED + f"Error processing current price: {e}", exc_info=True)
        return

    # 2. Calculate Indicators (returns Decimals)
    indicators = calculate_indicators(df)
    if indicators is None:
        logger.error(Fore.RED + "Halting cycle: Indicator calculation failed.")
        return
    current_atr = indicators.get("atr")  # Keep as Decimal

    # 3. Get Current State (Positions & Balance as Decimals)
    # Fetch balance first to know available capital
    free_balance, current_equity = get_balance("USDT")  # Assuming USDT quote
    if current_equity is None:
        # Allow proceeding without equity if balance fetch fails, but log warning
        logger.warning(
            Fore.YELLOW
            + "Failed to fetch current balance/equity. Status panel may be incomplete."
        )
        current_equity = Decimal("-1.0")  # Placeholder to indicate missing data

    # Fetch positions
    positions = get_current_position(CONFIG.symbol)
    if positions is None:
        logger.error(Fore.RED + "Halting cycle: Failed to fetch current positions.")
        # Decide if we should halt or proceed cautiously assuming flat? Halting is safer.
        return

    # Ensure positions dict has expected structure
    long_pos = positions.get(
        "long", {"qty": Decimal("0.0"), "entry_price": Decimal("0.0")}
    )
    short_pos = positions.get(
        "short", {"qty": Decimal("0.0"), "entry_price": Decimal("0.0")}
    )

    # 4. Manage Trailing Stops (pass Decimals)
    # Manage long TSL if long position exists
    if long_pos["qty"] >= CONFIG.position_qty_epsilon:
        manage_trailing_stop(
            CONFIG.symbol,
            "long",
            long_pos["qty"],
            long_pos["entry_price"],
            current_price,
            current_atr,
        )
    # Manage short TSL if short position exists
    if short_pos["qty"] >= CONFIG.position_qty_epsilon:
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
    order_tracker_snapshot = {
        "long": order_tracker["long"].copy(),
        "short": order_tracker["short"].copy(),
    }

    # 6. Execute Trades based on Signals
    # Check if flat (neither long nor short position significantly open)
    is_flat = (
        long_pos["qty"] < CONFIG.position_qty_epsilon
        and short_pos["qty"] < CONFIG.position_qty_epsilon
    )
    logger.debug(
        f"Position Status: Flat = {is_flat} (Long Qty: {long_pos['qty']}, Short Qty: {short_pos['qty']})"
    )

    trade_executed = False
    if is_flat:
        if signals.get("long"):
            logger.info(
                Fore.GREEN + Style.BRIGHT + "Long signal detected! Attempting entry."
            )
            trade_executed = place_risked_market_order(
                CONFIG.symbol, "buy", CONFIG.risk_percentage, current_atr
            )

        elif signals.get("short"):
            logger.info(
                Fore.RED + Style.BRIGHT + "Short signal detected! Attempting entry."
            )
            trade_executed = place_risked_market_order(
                CONFIG.symbol, "sell", CONFIG.risk_percentage, current_atr
            )

        # If a trade was attempted, pause briefly to allow exchange state to potentially update before next cycle's fetch
        if trade_executed:
            logger.info("Pausing briefly after trade attempt...")
            time.sleep(2)  # Small pause

    elif not is_flat:
        logger.info("Position already open, skipping new entry signals.")
        # Future enhancement: Add logic here to exit positions based on counter-signals,
        # profit targets, or other exit conditions if desired.
        # Example: if (long position exists and short signal) -> close long position
        # Example: if (short position exists and long signal) -> close short position

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

    logger.info(Fore.MAGENTA + f"--- Cycle {cycle_count} Complete ---")


def graceful_shutdown() -> None:
    """Dispels active orders and closes open positions gracefully with precision."""
    logger.info(
        Fore.YELLOW + Style.BRIGHT + "\nInitiating Graceful Shutdown Sequence..."
    )
    termux_notify("Shutdown", f"Closing orders/positions for {CONFIG.symbol}.")

    # Ensure exchange object is available
    if "exchange" not in globals() or not hasattr(exchange, "cancel_all_orders"):
        logger.error(Fore.RED + "Exchange object not available for shutdown.")
        return

    # 1. Cancel All Open Orders for the Symbol
    try:
        logger.info(Fore.CYAN + f"Dispelling all open orders for {CONFIG.symbol}...")
        # Fetch open orders first to log IDs before cancelling
        open_orders = []
        try:
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
            # cancel_all_orders might return confirmations or throw error
            # response = exchange.cancel_all_orders(CONFIG.symbol) # Standard CCXT
            # Bybit might require cancel_all_orders(symbol, params={'settleCoin': 'USDT'}) or similar? Check docs.
            # Let's try the standard call first.
            response = exchange.cancel_all_orders(CONFIG.symbol)
            logger.info(
                Fore.GREEN + f"Cancel command sent. Exchange response: {response}"
            )  # Response varies by exchange
        else:
            logger.info(Fore.GREEN + "No open orders found for the symbol to cancel.")

        # Clear local tracker regardless of API response, assuming intent was cancellation
        logger.info("Clearing local order tracker.")
        order_tracker["long"] = {"sl_id": None, "tsl_id": None}
        order_tracker["short"] = {"sl_id": None, "tsl_id": None}

    except (ccxt.ExchangeError, ccxt.NetworkError) as e:
        logger.error(
            Fore.RED
            + f"Error dispelling orders: {e}. MANUAL CHECK REQUIRED on exchange."
        )
    except Exception as e:
        logger.error(
            Fore.RED
            + f"Unexpected error dispelling orders: {e}. MANUAL CHECK REQUIRED.",
            exc_info=True,
        )

    # Add a small delay after cancelling orders before checking positions
    time.sleep(2)

    # 2. Close Any Open Positions
    try:
        logger.info(Fore.CYAN + "Checking for lingering positions to close...")
        # Fetch final position state using the dedicated function
        positions = get_current_position(CONFIG.symbol)
        closed_count = 0
        if positions:
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
                        close_qty_str = format_amount(CONFIG.symbol, qty)
                        close_order = exchange.create_market_order(
                            symbol=CONFIG.symbol,
                            side=close_side,
                            amount=float(close_qty_str),  # CCXT needs float
                            params={"reduceOnly": True},
                        )
                        logger.info(
                            Fore.GREEN
                            + f"Position closure order placed: ID {close_order.get('id')}"
                        )
                        closed_count += 1
                        # Add a small delay to allow closure order to process before final log
                        time.sleep(CONFIG.order_check_delay_seconds)
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

        if (
            closed_count == 0 and positions is not None
        ):  # Check positions was successfully fetched
            logger.info(Fore.GREEN + "No open positions found requiring closure.")
        elif positions is None:
            logger.error(
                Fore.RED
                + "Could not fetch positions during shutdown. MANUAL CHECK REQUIRED."
            )

    except Exception as e:
        logger.error(
            Fore.RED
            + f"Error during position closure check: {e}. Manual check advised.",
            exc_info=True,
        )

    logger.info(Fore.YELLOW + Style.BRIGHT + "Graceful Shutdown Sequence Complete.")
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

    if MARKET_INFO:  # Check if market info loaded successfully
        termux_notify("Bot Started", f"Monitoring {CONFIG.symbol} (v2 Precision)")
        logger.info("Awaiting market whispers...")
    else:
        logger.critical(
            Fore.RED
            + Style.BRIGHT
            + "Market info failed to load. Cannot start trading loop."
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
            Fore.RED + Style.BRIGHT + f"\nFatal Runtime Error in Main Loop: {e}",
            exc_info=True,
        )
        termux_notify("Bot CRASHED", f"{CONFIG.symbol} Error: Check logs!")
        graceful_shutdown()  # Attempt cleanup even on unexpected crash
        sys.exit(1)
    finally:
        # Ensure logs are flushed
        logging.shutdown()
