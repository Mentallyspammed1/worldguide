#!/usr/bin/env python

# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v2.0.1 (Enhanced Robustness & Clarity)
# Conjures high-frequency trades on Bybit Futures with enhanced precision and adaptable strategies.

"""High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 2.0.1 (Unified: Selectable Strategies + Precision + Native SL/TSL + Enhancements).

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
- Improved data validation and handling of edge cases.

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
import logging
import os
import subprocess
import sys
import time
import traceback
from decimal import ROUND_HALF_UP, Decimal, DivisionByZero, InvalidOperation, getcontext
from typing import Any

# Third-party Libraries
try:
    import ccxt
    import pandas as pd
    import pandas_ta as ta  # type: ignore[import]
    from colorama import Back, Fore, Style
    from colorama import init as colorama_init
    from dotenv import load_dotenv
except ImportError as e:
    missing_pkg = e.name
    sys.exit(1)

# --- Initializations ---
colorama_init(autoreset=True)
load_dotenv()
getcontext().prec = 18  # Set Decimal precision (adjust as needed for higher precision assets)


# --- Configuration Class ---
class Config:
    """Loads and validates configuration parameters from environment variables.

    Attributes are dynamically set based on environment variables defined below.
    Provides type casting, default values, and validation for required parameters.
    """
    def __init__(self) -> None:
        """Initializes the configuration by loading environment variables."""
        logger.info(f"{Fore.MAGENTA}--- Summoning Configuration Runes ---{Style.RESET_ALL}")
        # --- API Credentials ---
        self.api_key: str | None = self._get_env("BYBIT_API_KEY", required=True, color=Fore.RED)
        self.api_secret: str | None = self._get_env("BYBIT_API_SECRET", required=True, color=Fore.RED)

        # --- Trading Parameters ---
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT", color=Fore.YELLOW)
        self.interval: str = self._get_env("INTERVAL", "1m", color=Fore.YELLOW)
        self.leverage: int = self._get_env("LEVERAGE", 10, cast_type=int, color=Fore.YELLOW)
        self.sleep_seconds: int = self._get_env("SLEEP_SECONDS", 10, cast_type=int, color=Fore.YELLOW)

        # --- Strategy Selection ---
        self.strategy_name: str = self._get_env("STRATEGY_NAME", "DUAL_SUPERTREND", color=Fore.CYAN).upper()
        self.valid_strategies: list[str] = ["DUAL_SUPERTREND", "STOCHRSI_MOMENTUM", "EHLERS_FISHER", "EHLERS_MA_CROSS"]
        if self.strategy_name not in self.valid_strategies:
            logger.critical(f"Invalid STRATEGY_NAME '{self.strategy_name}'. Valid options are: {self.valid_strategies}")
            raise ValueError(f"Invalid STRATEGY_NAME '{self.strategy_name}'. Valid: {self.valid_strategies}")
        logger.info(f"Selected Strategy: {Fore.CYAN}{self.strategy_name}{Style.RESET_ALL}")

        # --- Risk Management ---
        self.risk_per_trade_percentage: Decimal = self._get_env("RISK_PER_TRADE_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN)  # 0.5%
        self.atr_stop_loss_multiplier: Decimal = self._get_env("ATR_STOP_LOSS_MULTIPLIER", "1.5", cast_type=Decimal, color=Fore.GREEN)
        self.max_order_usdt_amount: Decimal = self._get_env("MAX_ORDER_USDT_AMOUNT", "500.0", cast_type=Decimal, color=Fore.GREEN)
        self.required_margin_buffer: Decimal = self._get_env("REQUIRED_MARGIN_BUFFER", "1.05", cast_type=Decimal, color=Fore.GREEN)  # 5% buffer

        # --- Trailing Stop Loss (Exchange Native) ---
        self.trailing_stop_percentage: Decimal = self._get_env("TRAILING_STOP_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN)  # 0.5% trail
        self.trailing_stop_activation_offset_percent: Decimal = self._get_env("TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT", "0.001", cast_type=Decimal, color=Fore.GREEN)  # 0.1% offset

        # --- Dual Supertrend Parameters ---
        self.st_atr_length: int = self._get_env("ST_ATR_LENGTH", 7, cast_type=int, color=Fore.CYAN)
        self.st_multiplier: Decimal = self._get_env("ST_MULTIPLIER", "2.5", cast_type=Decimal, color=Fore.CYAN)
        self.confirm_st_atr_length: int = self._get_env("CONFIRM_ST_ATR_LENGTH", 5, cast_type=int, color=Fore.CYAN)
        self.confirm_st_multiplier: Decimal = self._get_env("CONFIRM_ST_MULTIPLIER", "2.0", cast_type=Decimal, color=Fore.CYAN)

        # --- StochRSI + Momentum Parameters ---
        self.stochrsi_rsi_length: int = self._get_env("STOCHRSI_RSI_LENGTH", 14, cast_type=int, color=Fore.CYAN)
        self.stochrsi_stoch_length: int = self._get_env("STOCHRSI_STOCH_LENGTH", 14, cast_type=int, color=Fore.CYAN)
        self.stochrsi_k_period: int = self._get_env("STOCHRSI_K_PERIOD", 3, cast_type=int, color=Fore.CYAN)
        self.stochrsi_d_period: int = self._get_env("STOCHRSI_D_PERIOD", 3, cast_type=int, color=Fore.CYAN)
        self.stochrsi_overbought: Decimal = self._get_env("STOCHRSI_OVERBOUGHT", "80.0", cast_type=Decimal, color=Fore.CYAN)
        self.stochrsi_oversold: Decimal = self._get_env("STOCHRSI_OVERSOLD", "20.0", cast_type=Decimal, color=Fore.CYAN)
        self.momentum_length: int = self._get_env("MOMENTUM_LENGTH", 5, cast_type=int, color=Fore.CYAN)

        # --- Ehlers Fisher Transform Parameters ---
        self.ehlers_fisher_length: int = self._get_env("EHLERS_FISHER_LENGTH", 10, cast_type=int, color=Fore.CYAN)
        self.ehlers_fisher_signal_length: int = self._get_env("EHLERS_FISHER_SIGNAL_LENGTH", 1, cast_type=int, color=Fore.CYAN)  # Default to 1

        # --- Ehlers MA Cross Parameters ---
        self.ehlers_fast_period: int = self._get_env("EHLERS_FAST_PERIOD", 10, cast_type=int, color=Fore.CYAN)
        self.ehlers_slow_period: int = self._get_env("EHLERS_SLOW_PERIOD", 30, cast_type=int, color=Fore.CYAN)

        # --- Volume Analysis ---
        self.volume_ma_period: int = self._get_env("VOLUME_MA_PERIOD", 20, cast_type=int, color=Fore.YELLOW)
        self.volume_spike_threshold: Decimal = self._get_env("VOLUME_SPIKE_THRESHOLD", "1.5", cast_type=Decimal, color=Fore.YELLOW)
        self.require_volume_spike_for_entry: bool = self._get_env("REQUIRE_VOLUME_SPIKE_FOR_ENTRY", "false", cast_type=bool, color=Fore.YELLOW)

        # --- Order Book Analysis ---
        self.order_book_depth: int = self._get_env("ORDER_BOOK_DEPTH", 10, cast_type=int, color=Fore.YELLOW)
        self.order_book_ratio_threshold_long: Decimal = self._get_env("ORDER_BOOK_RATIO_THRESHOLD_LONG", "1.2", cast_type=Decimal, color=Fore.YELLOW)
        self.order_book_ratio_threshold_short: Decimal = self._get_env("ORDER_BOOK_RATIO_THRESHOLD_SHORT", "0.8", cast_type=Decimal, color=Fore.YELLOW)
        self.fetch_order_book_per_cycle: bool = self._get_env("FETCH_ORDER_BOOK_PER_CYCLE", "false", cast_type=bool, color=Fore.YELLOW)

        # --- ATR Calculation (for Initial SL) ---
        self.atr_calculation_period: int = self._get_env("ATR_CALCULATION_PERIOD", 14, cast_type=int, color=Fore.GREEN)

        # --- Termux SMS Alerts ---
        self.enable_sms_alerts: bool = self._get_env("ENABLE_SMS_ALERTS", "false", cast_type=bool, color=Fore.MAGENTA)
        self.sms_recipient_number: str | None = self._get_env("SMS_RECIPIENT_NUMBER", None, color=Fore.MAGENTA)
        self.sms_timeout_seconds: int = self._get_env("SMS_TIMEOUT_SECONDS", 30, cast_type=int, color=Fore.MAGENTA)

        # --- CCXT / API Parameters ---
        self.default_recv_window: int = 10000
        self.order_book_fetch_limit: int = max(25, self.order_book_depth)  # Ensure fetch limit is at least 25 for L2 OB
        self.shallow_ob_fetch_depth: int = 5
        self.order_fill_timeout_seconds: int = self._get_env("ORDER_FILL_TIMEOUT_SECONDS", 15, cast_type=int, color=Fore.YELLOW)

        # --- Internal Constants ---
        self.SIDE_BUY: str = "buy"
        self.SIDE_SELL: str = "sell"
        self.POS_LONG: str = "Long"
        self.POS_SHORT: str = "Short"
        self.POS_NONE: str = "None"
        self.USDT_SYMBOL: str = "USDT"
        self.RETRY_COUNT: int = 3
        self.RETRY_DELAY_SECONDS: int = 2
        self.API_FETCH_LIMIT_BUFFER: int = 10  # Extra candles to fetch beyond indicator needs
        self.POSITION_QTY_EPSILON: Decimal = Decimal("1e-9")  # Small value to treat quantities near zero
        self.POST_CLOSE_DELAY_SECONDS: int = 3  # Wait time after closing position before next action

        logger.info(f"{Fore.MAGENTA}--- Configuration Runes Summoned Successfully ---{Style.RESET_ALL}")

    def _get_env(self, key: str, default: Any = None, cast_type: type = str, required: bool = False, color: str = Fore.WHITE) -> Any:
        """Fetches an environment variable, casts its type, logs the value, and handles defaults or errors.

        Args:
            key: The environment variable key name.
            default: The default value to use if the environment variable is not set.
            cast_type: The type to cast the environment variable value to (e.g., int, float, bool, Decimal).
            required: If True, raises a ValueError if the environment variable is not set and no default is provided.
            color: The colorama Fore color to use for logging this parameter.

        Returns:
            The environment variable value, cast to the specified type, or the default value.

        Raises:
            ValueError: If a required environment variable is not set.
        """
        value_str = os.getenv(key)
        value = None
        log_source = ""

        if value_str is not None:
            log_source = f"(from env: '{value_str}')"
            try:
                if cast_type == bool:
                    value = value_str.lower() in ['true', '1', 'yes', 'y']
                elif cast_type == Decimal:
                    value = Decimal(value_str)
                elif cast_type is not None:
                    value = cast_type(value_str)
                else:
                    value = value_str  # Keep as string if cast_type is None
            except (ValueError, TypeError, InvalidOperation) as e:
                logger.error(f"{Fore.RED}Invalid type/value for {key}: '{value_str}'. Expected {cast_type.__name__}. Error: {e}. Using default: '{default}'{Style.RESET_ALL}")
                value = default  # Fallback to default on casting error
                log_source = f"(env parse error, using default: '{default}')"
        else:
            value = default
            log_source = f"(not set, using default: '{default}')" if default is not None else "(not set, no default)"

        if value is None and required:
            critical_msg = f"CRITICAL: Required environment variable '{key}' not set and no default value provided."
            logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}{critical_msg}{Style.RESET_ALL}")
            raise ValueError(critical_msg)

        logger.debug(f"{color}Config {key}: {value} {log_source}{Style.RESET_ALL}")
        return value


# --- Logger Setup ---
LOGGING_LEVEL: int = logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]  # Ensure logs go to stdout
)
logger: logging.Logger = logging.getLogger(__name__)

# Custom SUCCESS level and Neon Color Formatting
SUCCESS_LEVEL: int = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Adds a custom 'success' log level."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        # pylint: disable=protected-access
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


logging.Logger.success = log_success  # type: ignore[attr-defined]

# Apply colors only if output is a TTY (console)
if sys.stdout.isatty():
    logging.addLevelName(logging.DEBUG, f"{Fore.CYAN}{logging.getLevelName(logging.DEBUG)}{Style.RESET_ALL}")
    logging.addLevelName(logging.INFO, f"{Fore.BLUE}{logging.getLevelName(logging.INFO)}{Style.RESET_ALL}")
    logging.addLevelName(SUCCESS_LEVEL, f"{Fore.MAGENTA}{logging.getLevelName(SUCCESS_LEVEL)}{Style.RESET_ALL}")
    logging.addLevelName(logging.WARNING, f"{Fore.YELLOW}{logging.getLevelName(logging.WARNING)}{Style.RESET_ALL}")
    logging.addLevelName(logging.ERROR, f"{Fore.RED}{logging.getLevelName(logging.ERROR)}{Style.RESET_ALL}")
    logging.addLevelName(logging.CRITICAL, f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}{logging.getLevelName(logging.CRITICAL)}{Style.RESET_ALL}")

# --- Global Objects ---
try:
    CONFIG = Config()
except ValueError as e:
    logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Configuration Error: {e}{Style.RESET_ALL}")
    sys.exit(1)
except Exception as e:
    logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Unexpected Error initializing configuration: {e}{Style.RESET_ALL}")
    logger.debug(traceback.format_exc())
    sys.exit(1)


# --- Helper Functions ---
def safe_decimal_conversion(value: Any, default: Decimal = Decimal("0.0")) -> Decimal:
    """Safely converts a value to Decimal, returning a default if conversion fails.

    Args:
        value: The value to convert (can be string, float, int, Decimal, etc.).
        default: The Decimal value to return if conversion fails.

    Returns:
        The converted Decimal value or the default.
    """
    if value is None:
        return default
    try:
        # Explicitly convert to string first to handle floats accurately
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        logger.warning(f"Could not convert '{value}' (type: {type(value).__name__}) to Decimal, using default {default}")
        return default


def format_order_id(order_id: str | int | None) -> str:
    """Returns the last 6 characters of an order ID or 'N/A' if None.

    Args:
        order_id: The order ID string or integer.

    Returns:
        A shortened representation of the order ID or 'N/A'.
    """
    return str(order_id)[-6:] if order_id else "N/A"


# --- Precision Formatting ---
def format_price(exchange: ccxt.Exchange, symbol: str, price: float | Decimal | str) -> str:
    """Formats a price according to the market's precision rules using CCXT.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT:USDT').
        price: The price value to format.

    Returns:
        The price formatted as a string according to market rules.
    """
    try:
        # CCXT formatting methods often expect float input, convert Decimal safely
        price_float = float(price)
        return exchange.price_to_precision(symbol, price_float)
    except (ValueError, TypeError, OverflowError, ccxt.ExchangeError, Exception) as e:
        logger.error(f"{Fore.RED}Error formatting price '{price}' for {symbol}: {e}{Style.RESET_ALL}")
        # Fallback to Decimal string representation with normalization
        try:
            return str(Decimal(str(price)).normalize())
        except (InvalidOperation, TypeError, ValueError):
             return str(price)  # Absolute fallback


def format_amount(exchange: ccxt.Exchange, symbol: str, amount: float | Decimal | str) -> str:
    """Formats an amount (quantity) according to the market's precision rules using CCXT.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT:USDT').
        amount: The amount value to format.

    Returns:
        The amount formatted as a string according to market rules.
    """
    try:
        # CCXT formatting methods often expect float input, convert Decimal safely
        amount_float = float(amount)
        return exchange.amount_to_precision(symbol, amount_float)
    except (ValueError, TypeError, OverflowError, ccxt.ExchangeError, Exception) as e:
        logger.error(f"{Fore.RED}Error formatting amount '{amount}' for {symbol}: {e}{Style.RESET_ALL}")
        # Fallback to Decimal string representation with normalization
        try:
            return str(Decimal(str(amount)).normalize())
        except (InvalidOperation, TypeError, ValueError):
            return str(amount)  # Absolute fallback


# --- Termux SMS Alert Function ---
def send_sms_alert(message: str) -> bool:
    """Sends an SMS alert using the Termux:API command-line tool.

    Args:
        message: The text message content to send.

    Returns:
        True if the command executed successfully (return code 0), False otherwise.
    """
    if not CONFIG.enable_sms_alerts:
        logger.debug("SMS alerts disabled via config.")
        return False
    if not CONFIG.sms_recipient_number:
        logger.warning("SMS alerts enabled, but SMS_RECIPIENT_NUMBER is not set in config.")
        return False

    try:
        # Use shlex.quote for message safety if needed, but direct passing is usually fine for simple messages
        # quoted_message = shlex.quote(message)
        command: list[str] = ['termux-sms-send', '-n', CONFIG.sms_recipient_number, message]
        logger.info(f"{Fore.MAGENTA}Attempting SMS to {CONFIG.sms_recipient_number} (Timeout: {CONFIG.sms_timeout_seconds}s): \"{message[:50]}...\"{Style.RESET_ALL}")

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,  # Don't raise exception on non-zero exit code
            timeout=CONFIG.sms_timeout_seconds
        )

        if result.returncode == 0:
            logger.success(f"{Fore.MAGENTA}SMS command executed successfully.{Style.RESET_ALL}")
            return True
        else:
            stderr_msg = result.stderr.strip() if result.stderr else "No stderr output"
            logger.error(f"{Fore.RED}SMS command failed. RC: {result.returncode}, Stderr: {stderr_msg}{Style.RESET_ALL}")
            return False
    except FileNotFoundError:
        logger.error(f"{Fore.RED}SMS failed: 'termux-sms-send' command not found. Is Termux:API app installed and configured?{Style.RESET_ALL}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"{Fore.RED}SMS failed: Command timed out after {CONFIG.sms_timeout_seconds}s.{Style.RESET_ALL}")
        return False
    except Exception as e:
        logger.error(f"{Fore.RED}SMS failed: Unexpected error during execution: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        return False


# --- Exchange Initialization ---
def initialize_exchange() -> ccxt.Exchange | None:
    """Initializes and returns the CCXT Bybit exchange instance using API keys from config.

    Performs basic checks like loading markets and fetching balance.

    Returns:
        A configured CCXT Bybit exchange instance, or None if initialization fails.
    """
    logger.info(f"{Fore.BLUE}Initializing CCXT Bybit connection...{Style.RESET_ALL}")
    if not CONFIG.api_key or not CONFIG.api_secret:
        logger.critical("API Key or Secret is missing in configuration.")
        send_sms_alert("[ScalpBot] CRITICAL: API keys missing. Bot stopped.")
        return None
    try:
        exchange = ccxt.bybit({
            "apiKey": CONFIG.api_key,
            "secret": CONFIG.api_secret,
            "enableRateLimit": True,  # Enable built-in rate limiting
            "options": {
                "defaultType": "linear",  # Default to linear contracts (USDT margined)
                "recvWindow": CONFIG.default_recv_window,
                "adjustForTimeDifference": True,  # Auto-adjust timestamp if needed
                # 'verbose': True, # Uncomment for detailed API request/response logging
            },
        })
        logger.debug("Loading markets (forced reload)...")
        exchange.load_markets(True)  # Force reload to get latest info
        logger.debug("Performing initial balance check...")
        exchange.fetch_balance()  # Check if API keys are valid by fetching balance
        logger.success(f"{Fore.GREEN}{Style.BRIGHT}CCXT Bybit Session Initialized (LIVE SCALPING MODE - EXTREME CAUTION!).{Style.RESET_ALL}")
        send_sms_alert("[ScalpBot] Initialized & authenticated successfully.")
        return exchange

    except ccxt.AuthenticationError as e:
        logger.critical(f"Authentication failed: {e}. Check API keys, IP whitelist, and permissions on Bybit.")
        send_sms_alert(f"[ScalpBot] CRITICAL: Authentication FAILED: {e}. Bot stopped.")
    except ccxt.NetworkError as e:
        logger.critical(f"Network error during initialization: {e}. Check internet connection and Bybit status.")
        send_sms_alert(f"[ScalpBot] CRITICAL: Network Error on Init: {e}. Bot stopped.")
    except ccxt.ExchangeError as e:
        logger.critical(f"Exchange error during initialization: {e}. Check Bybit status or API documentation.")
        send_sms_alert(f"[ScalpBot] CRITICAL: Exchange Error on Init: {e}. Bot stopped.")
    except Exception as e:
        logger.critical(f"Unexpected error during exchange initialization: {e}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[ScalpBot] CRITICAL: Unexpected Init Error: {type(e).__name__}. Bot stopped.")

    return None


# --- Indicator Calculation Functions ---
def calculate_supertrend(df: pd.DataFrame, length: int, multiplier: Decimal, prefix: str = "") -> pd.DataFrame:
    """Calculates the Supertrend indicator using pandas_ta.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
        length: The ATR lookback period for Supertrend.
        multiplier: The ATR multiplier for Supertrend.
        prefix: Optional prefix for the resulting columns (e.g., "confirm_").

    Returns:
        The input DataFrame with added Supertrend columns:
        - f'{prefix}supertrend': The Supertrend line value (Decimal).
        - f'{prefix}trend': Boolean, True if uptrend (price > Supertrend), False otherwise.
        - f'{prefix}st_long': Boolean, True if a long entry signal (trend flipped up) occurred.
        - f'{prefix}st_short': Boolean, True if a short entry signal (trend flipped down) occurred.
    """
    col_prefix = f"{prefix}" if prefix else ""
    target_cols = [f"{col_prefix}supertrend", f"{col_prefix}trend", f"{col_prefix}st_long", f"{col_prefix}st_short"]
    # pandas_ta uses float in the generated column name string representation
    st_col = f"SUPERT_{length}_{float(multiplier)}"
    st_trend_col = f"SUPERTd_{length}_{float(multiplier)}"
    st_long_col = f"SUPERTl_{length}_{float(multiplier)}"
    st_short_col = f"SUPERTs_{length}_{float(multiplier)}"
    required_input_cols = ["high", "low", "close"]
    min_required_len = length + 1  # Need at least 'length' periods for ATR + 1 for comparison

    # Initialize target columns to NA
    for col in target_cols: df[col] = pd.NA

    if df is None or df.empty or not all(c in df.columns for c in required_input_cols) or len(df) < min_required_len:
        logger.warning(f"{Fore.YELLOW}Indicator Calc ({col_prefix}ST): Input invalid or too short (Len: {len(df) if df is not None else 0}, Need: {min_required_len}).{Style.RESET_ALL}")
        return df

    try:
        # pandas_ta expects float multiplier, calculate in place
        df.ta.supertrend(length=length, multiplier=float(multiplier), append=True)

        # Check if pandas_ta created the expected raw columns
        if st_col not in df.columns or st_trend_col not in df.columns:
            raise KeyError(f"pandas_ta failed to create expected raw columns: {st_col}, {st_trend_col}")

        # Convert Supertrend value to Decimal
        df[f"{col_prefix}supertrend"] = df[st_col].apply(lambda x: safe_decimal_conversion(x, default=Decimal('NaN')))
        df[f"{col_prefix}trend"] = df[st_trend_col] == 1  # Boolean: True for uptrend (1), False for downtrend (-1)

        # Calculate flip signals (requires previous trend value)
        prev_trend = df[st_trend_col].shift(1)
        df[f"{col_prefix}st_long"] = (prev_trend == -1) & (df[st_trend_col] == 1)  # Flipped from down to up
        df[f"{col_prefix}st_short"] = (prev_trend == 1) & (df[st_trend_col] == -1)  # Flipped from up to down

        # Drop the raw columns generated by pandas_ta
        raw_st_cols_to_drop = [st_col, st_trend_col, st_long_col, st_short_col]
        df.drop(columns=[col for col in raw_st_cols_to_drop if col in df.columns], inplace=True)

        # Log last calculated values for debugging
        last_st_val = df[f'{col_prefix}supertrend'].iloc[-1]
        if pd.notna(last_st_val):
            last_trend = 'Up' if df[f'{col_prefix}trend'].iloc[-1] else 'Down'
            signal = 'LONG' if df[f'{col_prefix}st_long'].iloc[-1] else ('SHORT' if df[f'{col_prefix}st_short'].iloc[-1] else 'None')
            logger.debug(f"Indicator Calc ({col_prefix}ST({length},{multiplier})): Trend={last_trend}, Val={last_st_val:.4f}, Signal={signal}")
        else:
            logger.debug(f"Indicator Calc ({col_prefix}ST({length},{multiplier})): Resulted in NA for last candle.")

    except KeyError as e:
        logger.error(f"{Fore.RED}Indicator Calc ({col_prefix}ST): Missing column during calculation: {e}{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA  # Ensure columns exist even on error
    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc ({col_prefix}ST): Unexpected error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA  # Ensure columns exist even on error
    return df


def analyze_volume_atr(df: pd.DataFrame, atr_len: int, vol_ma_len: int) -> dict[str, Decimal | None]:
    """Calculates ATR, Volume Moving Average, and checks for volume spikes.

    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume' columns.
        atr_len: The lookback period for ATR calculation.
        vol_ma_len: The lookback period for the Volume Moving Average.

    Returns:
        A dictionary containing:
        - 'atr': Calculated ATR value (Decimal), or None if calculation failed.
        - 'volume_ma': Volume Moving Average (Decimal), or None.
        - 'last_volume': Last candle's volume (Decimal), or None.
        - 'volume_ratio': Ratio of last volume to volume MA (Decimal), or None.
    """
    results: dict[str, Decimal | None] = {"atr": None, "volume_ma": None, "last_volume": None, "volume_ratio": None}
    required_cols = ["high", "low", "close", "volume"]
    min_len = max(atr_len, vol_ma_len) + 1  # Need sufficient lookback

    if df is None or df.empty or not all(c in df.columns for c in required_cols) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Indicator Calc (Vol/ATR): Input invalid or too short (Len: {len(df) if df is not None else 0}, Need: {min_len}).{Style.RESET_ALL}")
        return results

    try:
        # Calculate ATR using pandas_ta
        atr_col = f"ATRr_{atr_len}"  # Default ATR column name from pandas_ta
        df.ta.atr(length=atr_len, append=True)
        if atr_col in df.columns:
            last_atr = df[atr_col].iloc[-1]
            if pd.notna(last_atr):
                results["atr"] = safe_decimal_conversion(last_atr, default=Decimal('NaN'))
            df.drop(columns=[atr_col], errors='ignore', inplace=True)  # Clean up raw column
        else:
             logger.warning(f"{Fore.YELLOW}Indicator Calc (ATR): Column '{atr_col}' not found after calculation.{Style.RESET_ALL}")

        # Calculate Volume MA using pandas rolling mean
        volume_ma_col = 'volume_ma'
        # Use min_periods to get a value even if window isn't full initially
        df[volume_ma_col] = df['volume'].rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2)).mean()
        last_vol_ma = df[volume_ma_col].iloc[-1]
        last_vol = df['volume'].iloc[-1]

        if pd.notna(last_vol_ma): results["volume_ma"] = safe_decimal_conversion(last_vol_ma, default=Decimal('NaN'))
        if pd.notna(last_vol): results["last_volume"] = safe_decimal_conversion(last_vol, default=Decimal('NaN'))

        # Calculate Volume Ratio safely
        if results["volume_ma"] is not None and results["volume_ma"] > CONFIG.POSITION_QTY_EPSILON and results["last_volume"] is not None:
            try:
                results["volume_ratio"] = results["last_volume"] / results["volume_ma"]
            except (DivisionByZero, InvalidOperation):
                 logger.warning("Indicator Calc (Vol/ATR): Division by zero or invalid operation calculating volume ratio.")
                 results["volume_ratio"] = None
        else:
             results["volume_ratio"] = None  # Set to None if MA is zero/negligible or volume is missing

        if volume_ma_col in df.columns: df.drop(columns=[volume_ma_col], errors='ignore', inplace=True)  # Clean up temp column

        # Log results
        atr_str = f"{results['atr']:.5f}" if results['atr'] is not None and not results['atr'].is_nan() else 'N/A'
        vol_ma_str = f"{results['volume_ma']:.2f}" if results['volume_ma'] is not None and not results['volume_ma'].is_nan() else 'N/A'
        last_vol_str = f"{results['last_volume']:.2f}" if results['last_volume'] is not None and not results['last_volume'].is_nan() else 'N/A'
        vol_ratio_str = f"{results['volume_ratio']:.2f}" if results['volume_ratio'] is not None and not results['volume_ratio'].is_nan() else 'N/A'
        logger.debug(f"Indicator Calc: ATR({atr_len})={atr_str}, Vol={last_vol_str}, VolMA({vol_ma_len})={vol_ma_str}, VolRatio={vol_ratio_str}")

    except KeyError as e:
        logger.error(f"{Fore.RED}Indicator Calc (Vol/ATR): Missing column: {e}{Style.RESET_ALL}")
        results = dict.fromkeys(results)
    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc (Vol/ATR): Unexpected error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        results = dict.fromkeys(results)  # Reset on error
    return results


def calculate_stochrsi_momentum(df: pd.DataFrame, rsi_len: int, stoch_len: int, k: int, d: int, mom_len: int) -> pd.DataFrame:
    """Calculates StochRSI (K and D lines) and Momentum indicator using pandas_ta.

    Args:
        df: DataFrame with 'close' column.
        rsi_len: Lookback period for RSI calculation within StochRSI.
        stoch_len: Lookback period for Stochastic calculation within StochRSI.
        k: Smoothing period for StochRSI %K line.
        d: Smoothing period for StochRSI %D line.
        mom_len: Lookback period for Momentum indicator.

    Returns:
        The input DataFrame with added columns:
        - 'stochrsi_k': StochRSI %K line value (Decimal).
        - 'stochrsi_d': StochRSI %D line value (Decimal).
        - 'momentum': Momentum value (Decimal).
    """
    target_cols = ['stochrsi_k', 'stochrsi_d', 'momentum']
    # Estimate minimum length needed - StochRSI needs RSI + Stoch periods + smoothing
    min_len = max(rsi_len + stoch_len + max(k, d), mom_len) + 5  # Add buffer
    for col in target_cols: df[col] = pd.NA  # Initialize columns

    if df is None or df.empty or "close" not in df.columns or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Indicator Calc (StochRSI/Mom): Input invalid or too short (Len: {len(df) if df is not None else 0}, Need ~{min_len}).{Style.RESET_ALL}")
        return df
    try:
        # Calculate StochRSI - use append=False to get predictable column names
        stochrsi_df = df.ta.stochrsi(length=stoch_len, rsi_length=rsi_len, k=k, d=d, append=False)
        k_col = f"STOCHRSIk_{stoch_len}_{rsi_len}_{k}_{d}"
        d_col = f"STOCHRSId_{stoch_len}_{rsi_len}_{k}_{d}"

        if k_col in stochrsi_df.columns:
            df['stochrsi_k'] = stochrsi_df[k_col].apply(lambda x: safe_decimal_conversion(x, default=Decimal('NaN')))
        else:
            logger.warning(f"{Fore.YELLOW}StochRSI K column '{k_col}' not found after calculation.{Style.RESET_ALL}")

        if d_col in stochrsi_df.columns:
            df['stochrsi_d'] = stochrsi_df[d_col].apply(lambda x: safe_decimal_conversion(x, default=Decimal('NaN')))
        else:
             logger.warning(f"{Fore.YELLOW}StochRSI D column '{d_col}' not found after calculation.{Style.RESET_ALL}")

        # Calculate Momentum
        mom_col = f"MOM_{mom_len}"
        df.ta.mom(length=mom_len, append=True)  # Append momentum directly
        if mom_col in df.columns:
            df['momentum'] = df[mom_col].apply(lambda x: safe_decimal_conversion(x, default=Decimal('NaN')))
            df.drop(columns=[mom_col], errors='ignore', inplace=True)  # Clean up raw column
        else:
            logger.warning(f"{Fore.YELLOW}Momentum column '{mom_col}' not found after calculation.{Style.RESET_ALL}")

        # Log last values
        k_val, d_val, mom_val = df['stochrsi_k'].iloc[-1], df['stochrsi_d'].iloc[-1], df['momentum'].iloc[-1]
        if pd.notna(k_val) and pd.notna(d_val) and pd.notna(mom_val):
            logger.debug(f"Indicator Calc (StochRSI/Mom): K={k_val:.2f}, D={d_val:.2f}, Mom={mom_val:.4f}")
        else:
            logger.debug("Indicator Calc (StochRSI/Mom): Resulted in NA for last candle.")

    except KeyError as e:
        logger.error(f"{Fore.RED}Indicator Calc (StochRSI/Mom): Missing column: {e}{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA
    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc (StochRSI/Mom): Unexpected error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA
    return df


def calculate_ehlers_fisher(df: pd.DataFrame, length: int, signal: int) -> pd.DataFrame:
    """Calculates the Ehlers Fisher Transform indicator using pandas_ta.

    Args:
        df: DataFrame with 'high', 'low' columns.
        length: The lookback period for the Fisher Transform.
        signal: The smoothing period for the signal line (usually 1).

    Returns:
        The input DataFrame with added columns:
        - 'ehlers_fisher': Fisher Transform value (Decimal).
        - 'ehlers_signal': Fisher Transform signal line value (Decimal).
    """
    target_cols = ['ehlers_fisher', 'ehlers_signal']
    min_len = length + signal  # Approximate minimum length
    for col in target_cols: df[col] = pd.NA  # Initialize columns

    if df is None or df.empty or not all(c in df.columns for c in ["high", "low"]) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Indicator Calc (EhlersFisher): Input invalid or too short (Len: {len(df) if df is not None else 0}, Need ~{min_len}).{Style.RESET_ALL}")
        return df
    try:
        # Calculate Fisher Transform - use append=False
        fisher_df = df.ta.fisher(length=length, signal=signal, append=False)
        fish_col = f"FISHERT_{length}_{signal}"
        signal_col = f"FISHERTs_{length}_{signal}"

        if fish_col in fisher_df.columns:
            df['ehlers_fisher'] = fisher_df[fish_col].apply(lambda x: safe_decimal_conversion(x, default=Decimal('NaN')))
        else:
            logger.warning(f"{Fore.YELLOW}Ehlers Fisher column '{fish_col}' not found after calculation.{Style.RESET_ALL}")

        if signal_col in fisher_df.columns:
            df['ehlers_signal'] = fisher_df[signal_col].apply(lambda x: safe_decimal_conversion(x, default=Decimal('NaN')))
        else:
            logger.warning(f"{Fore.YELLOW}Ehlers Signal column '{signal_col}' not found after calculation.{Style.RESET_ALL}")

        # Log last values
        fish_val, sig_val = df['ehlers_fisher'].iloc[-1], df['ehlers_signal'].iloc[-1]
        if pd.notna(fish_val) and pd.notna(sig_val):
             logger.debug(f"Indicator Calc (EhlersFisher({length},{signal})): Fisher={fish_val:.4f}, Signal={sig_val:.4f}")
        else:
             logger.debug("Indicator Calc (EhlersFisher): Resulted in NA for last candle.")

    except KeyError as e:
        logger.error(f"{Fore.RED}Indicator Calc (EhlersFisher): Missing column: {e}{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA
    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc (EhlersFisher): Unexpected error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA
    return df


def calculate_ehlers_ma(df: pd.DataFrame, fast_len: int, slow_len: int) -> pd.DataFrame:
    """Calculates Ehlers-style Moving Averages (placeholder using EMA).

    Args:
        df: DataFrame with 'close' column.
        fast_len: Lookback period for the fast moving average.
        slow_len: Lookback period for the slow moving average.

    Returns:
        The input DataFrame with added columns:
        - 'fast_ema': Fast EMA value (Decimal).
        - 'slow_ema': Slow EMA value (Decimal).
    """
    target_cols = ['fast_ema', 'slow_ema']
    min_len = max(fast_len, slow_len) + 5  # Add buffer
    for col in target_cols: df[col] = pd.NA  # Initialize columns

    if df is None or df.empty or "close" not in df.columns or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Indicator Calc (EhlersMA): Input invalid or too short (Len: {len(df) if df is not None else 0}, Need ~{min_len}).{Style.RESET_ALL}")
        return df
    try:
        # WARNING: Placeholder Implementation!
        # pandas_ta.supersmoother might not exist or be reliable.
        # Using standard EMA as a substitute. Replace with a proper Ehlers filter
        # implementation (e.g., from another library or custom code) if true Ehlers MA is needed.
        logger.warning(f"{Fore.YELLOW}Using standard EMA as placeholder for Ehlers Super Smoother MAs. Review if accurate Ehlers MA is required.{Style.RESET_ALL}")
        df['fast_ema'] = df.ta.ema(length=fast_len).apply(lambda x: safe_decimal_conversion(x, default=Decimal('NaN')))
        df['slow_ema'] = df.ta.ema(length=slow_len).apply(lambda x: safe_decimal_conversion(x, default=Decimal('NaN')))

        # Log last values
        fast_val, slow_val = df['fast_ema'].iloc[-1], df['slow_ema'].iloc[-1]
        if pd.notna(fast_val) and pd.notna(slow_val):
            logger.debug(f"Indicator Calc (EhlersMA({fast_len},{slow_len})): Fast={fast_val:.4f}, Slow={slow_val:.4f}")
        else:
             logger.debug("Indicator Calc (EhlersMA): Resulted in NA for last candle.")

    except KeyError as e:
        logger.error(f"{Fore.RED}Indicator Calc (EhlersMA): Missing column: {e}{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA
    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc (EhlersMA): Unexpected error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA
    return df


def analyze_order_book(exchange: ccxt.Exchange, symbol: str, depth: int, fetch_limit: int) -> dict[str, Decimal | None]:
    """Fetches and analyzes the L2 order book for bid/ask pressure and spread.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol.
        depth: The number of price levels (bids/asks) to consider for volume summation.
        fetch_limit: The number of price levels to request from the API (>= depth).

    Returns:
        A dictionary containing:
        - 'bid_ask_ratio': Ratio of cumulative bid volume to ask volume within the specified depth (Decimal), or None.
        - 'spread': Difference between best ask and best bid (Decimal), or None.
        - 'best_bid': Best bid price (Decimal), or None.
        - 'best_ask': Best ask price (Decimal), or None.
    """
    results: dict[str, Decimal | None] = {"bid_ask_ratio": None, "spread": None, "best_bid": None, "best_ask": None}
    logger.debug(f"Order Book: Fetching L2 {symbol} (Depth:{depth}, Limit:{fetch_limit})...")

    if not exchange.has.get('fetchL2OrderBook'):
        logger.warning(f"{Fore.YELLOW}Order Book: fetchL2OrderBook is not supported by {exchange.id}. Cannot analyze.{Style.RESET_ALL}")
        return results
    try:
        # Fetch L2 order book data
        order_book = exchange.fetch_l2_order_book(symbol, limit=fetch_limit)
        bids: list[list[float | str]] = order_book.get('bids', [])
        asks: list[list[float | str]] = order_book.get('asks', [])

        if not bids or not asks:
            logger.warning(f"{Fore.YELLOW}Order Book: Fetched empty bids or asks for {symbol}.{Style.RESET_ALL}")
            return results

        # Extract best bid/ask using safe conversion
        best_bid = safe_decimal_conversion(bids[0][0], default=Decimal('NaN')) if len(bids[0]) > 0 else Decimal('NaN')
        best_ask = safe_decimal_conversion(asks[0][0], default=Decimal('NaN')) if len(asks[0]) > 0 else Decimal('NaN')
        results["best_bid"] = best_bid if not best_bid.is_nan() else None
        results["best_ask"] = best_ask if not best_ask.is_nan() else None

        # Calculate spread
        if results["best_bid"] is not None and results["best_ask"] is not None:
            if results["best_ask"] > results["best_bid"]:  # Sanity check
                results["spread"] = results["best_ask"] - results["best_bid"]
                logger.debug(f"OB: Bid={results['best_bid']:.4f}, Ask={results['best_ask']:.4f}, Spread={results['spread']:.4f}")
            else:
                logger.warning(f"{Fore.YELLOW}Order Book: Best bid ({results['best_bid']}) >= best ask ({results['best_ask']}). Spread calculation invalid.{Style.RESET_ALL}")
                results["spread"] = None
        else:
            logger.debug(f"OB: Bid={results['best_bid'] or 'N/A'}, Ask={results['best_ask'] or 'N/A'} (Spread N/A)")

        # Sum volumes within the specified depth using Decimal
        bid_vol = sum(safe_decimal_conversion(bid[1], default=Decimal('0')) for bid in bids[:depth] if len(bid) > 1)
        ask_vol = sum(safe_decimal_conversion(ask[1], default=Decimal('0')) for ask in asks[:depth] if len(ask) > 1)
        logger.debug(f"OB (Depth {depth}): BidVol={bid_vol:.4f}, AskVol={ask_vol:.4f}")

        # Calculate bid/ask ratio safely
        if ask_vol > CONFIG.POSITION_QTY_EPSILON:
            try:
                results["bid_ask_ratio"] = bid_vol / ask_vol
                logger.debug(f"OB Ratio: {results['bid_ask_ratio']:.3f}")
            except (DivisionByZero, InvalidOperation):
                logger.warning("Order Book: Error calculating OB ratio (division by zero or invalid operation).")
                results["bid_ask_ratio"] = None
        else:
            logger.debug("OB Ratio: N/A (Ask volume zero or negligible)")
            results["bid_ask_ratio"] = None  # Explicitly set to None

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger.warning(f"{Fore.YELLOW}Order Book: API Error fetching for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
    except (IndexError, TypeError, KeyError) as e:
         logger.warning(f"{Fore.YELLOW}Order Book: Error parsing data for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"{Fore.RED}Order Book: Unexpected error for {symbol}: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())

    # Ensure None is returned for keys if any error occurred during calculation
    if any(v is not None and isinstance(v, Decimal) and v.is_nan() for v in results.values()):
        results = {k: (v if not (isinstance(v, Decimal) and v.is_nan()) else None) for k, v in results.items()}

    return results


# --- Data Fetching ---
def get_market_data(exchange: ccxt.Exchange, symbol: str, interval: str, limit: int) -> pd.DataFrame | None:
    """Fetches and prepares OHLCV data from the exchange.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol.
        interval: The timeframe interval (e.g., '1m', '5m').
        limit: The maximum number of candles to fetch.

    Returns:
        A pandas DataFrame containing OHLCV data with a datetime index,
        or None if fetching or processing fails.
    """
    if not exchange.has.get("fetchOHLCV"):
        logger.error(f"{Fore.RED}Data Fetch: Exchange '{exchange.id}' does not support fetchOHLCV.{Style.RESET_ALL}")
        return None
    try:
        logger.debug(f"Data Fetch: Fetching {limit} OHLCV candles for {symbol} ({interval})...")
        # Fetch OHLCV data: [timestamp, open, high, low, close, volume]
        ohlcv: list[list[int | float | str]] = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)

        if not ohlcv:
            logger.warning(f"{Fore.YELLOW}Data Fetch: No OHLCV data returned for {symbol} ({interval}). Might be an invalid symbol or timeframe.{Style.RESET_ALL}")
            return None

        # Create DataFrame
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        if df.empty:
            logger.warning(f"{Fore.YELLOW}Data Fetch: OHLCV data for {symbol} resulted in an empty DataFrame.{Style.RESET_ALL}")
            return None

        # Convert timestamp to datetime and set as index
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
        except Exception as e:
             logger.error(f"{Fore.RED}Data Fetch: Error processing timestamps: {e}{Style.RESET_ALL}")
             return None

        # Convert OHLCV columns to numeric, coercing errors to NaN
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Check for and handle NaNs robustly
        if df.isnull().values.any():
            nan_counts = df.isnull().sum()
            logger.warning(f"{Fore.YELLOW}Data Fetch: OHLCV contains NaNs after conversion:\n{nan_counts[nan_counts > 0]}\nAttempting forward fill...{Style.RESET_ALL}")
            df.ffill(inplace=True)  # Forward fill first (common for missing data)
            if df.isnull().values.any():  # Check again, backfill if needed
                logger.warning(f"{Fore.YELLOW}NaNs remain after ffill, attempting backward fill...{Style.RESET_ALL}")
                df.bfill(inplace=True)
                if df.isnull().values.any():
                    logger.error(f"{Fore.RED}Data Fetch: NaNs persist even after ffill and bfill. Cannot proceed with this data.{Style.RESET_ALL}")
                    return None  # Cannot reliably use data with remaining NaNs

        logger.debug(f"Data Fetch: Successfully processed {len(df)} OHLCV candles for {symbol}.")
        return df

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger.warning(f"{Fore.YELLOW}Data Fetch: API Error fetching OHLCV for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
    except (ValueError, TypeError, KeyError, Exception) as e:
        logger.error(f"{Fore.RED}Data Fetch: Error processing OHLCV data for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())

    return None


# --- Position & Order Management ---
def get_current_position(exchange: ccxt.Exchange, symbol: str) -> dict[str, Any]:
    """Fetches the current position details for a given symbol, focusing on Bybit V5 API structure.

    Assumes One-Way Mode on Bybit.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT:USDT').

    Returns:
        A dictionary containing:
        - 'side': Position side ('Long', 'Short', or 'None').
        - 'qty': Position quantity (Decimal), absolute value.
        - 'entry_price': Average entry price (Decimal).
        Returns default values (side='None', qty=0.0, entry_price=0.0) if no position or error.
    """
    default_pos: dict[str, Any] = {'side': CONFIG.POS_NONE, 'qty': Decimal("0.0"), 'entry_price': Decimal("0.0")}
    market: dict | None = None
    market_id: str | None = None

    try:
        market = exchange.market(symbol)
        market_id = market['id']  # Get the exchange-specific market ID (e.g., 'BTCUSDT')
    except (ccxt.BadSymbol, KeyError, Exception) as e:
        logger.error(f"{Fore.RED}Position Check: Failed to get market info for '{symbol}': {e}{Style.RESET_ALL}")
        return default_pos

    if not market:  # Should not happen if above try succeeded, but check anyway
        logger.error(f"{Fore.RED}Position Check: Market info for '{symbol}' is unexpectedly None.{Style.RESET_ALL}")
        return default_pos

    try:
        if not exchange.has.get('fetchPositions'):
            logger.warning(f"{Fore.YELLOW}Position Check: fetchPositions method not supported by {exchange.id}. Cannot check position.{Style.RESET_ALL}")
            return default_pos

        # Determine category for Bybit V5 API call (linear or inverse)
        category = 'linear' if market.get('linear') else ('inverse' if market.get('inverse') else None)
        if not category:
            logger.warning(f"{Fore.YELLOW}Position Check: Could not determine category (linear/inverse) for {symbol}. Assuming linear.{Style.RESET_ALL}")
            category = 'linear'  # Default assumption

        params = {'category': category}
        logger.debug(f"Position Check: Fetching positions for {symbol} (MarketID: {market_id}) with params: {params}")

        # Fetch positions for the specific symbol
        fetched_positions = exchange.fetch_positions(symbols=[symbol], params=params)

        # Filter for the active position in One-Way mode
        active_pos_data = None
        for pos in fetched_positions:
            pos_info = pos.get('info', {})
            pos_market_id = pos_info.get('symbol')
            # Bybit V5 One-Way mode uses positionIdx 0. Hedge mode uses 1 for Buy, 2 for Sell.
            position_idx = pos_info.get('positionIdx', -1)  # Use -1 default to indicate if not found
            pos_side_v5 = pos_info.get('side', 'None')  # 'Buy' for long, 'Sell' for short, 'None' if flat
            size_str = pos_info.get('size')

            # Match market ID, check for One-Way mode (idx 0), and ensure side is not 'None' (means position exists)
            if pos_market_id == market_id and position_idx == 0 and pos_side_v5 != 'None':
                size = safe_decimal_conversion(size_str)
                # Check if size is significant (greater than epsilon)
                if abs(size) > CONFIG.POSITION_QTY_EPSILON:
                    active_pos_data = pos  # Found the active position for this symbol in One-Way mode
                    break  # Assume only one such position exists per symbol in One-Way mode

        if active_pos_data:
            try:
                info = active_pos_data.get('info', {})
                size = safe_decimal_conversion(info.get('size'))
                # Use 'avgPrice' from info for V5 entry price
                entry_price = safe_decimal_conversion(info.get('avgPrice'))
                # Determine side based on V5 'side' field ('Buy' -> Long, 'Sell' -> Short)
                side = CONFIG.POS_LONG if info.get('side') == 'Buy' else CONFIG.POS_SHORT

                position_qty = abs(size)
                if position_qty <= CONFIG.POSITION_QTY_EPSILON:
                     logger.info(f"Position Check: Found position for {market_id}, but size ({size}) is negligible. Treating as flat.")
                     return default_pos

                logger.info(f"{Fore.YELLOW}Position Check: Found ACTIVE {side} position: Qty={position_qty:.8f} @ Entry={entry_price:.4f}{Style.RESET_ALL}")
                return {'side': side, 'qty': position_qty, 'entry_price': entry_price}
            except (KeyError, TypeError, Exception) as parse_err:
                 logger.warning(f"{Fore.YELLOW}Position Check: Error parsing active position data: {parse_err}. Data: {active_pos_data}{Style.RESET_ALL}")
                 return default_pos  # Return default on parsing error
        else:
            logger.info(f"Position Check: No active One-Way position found for {market_id}.")
            return default_pos

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger.warning(f"{Fore.YELLOW}Position Check: API Error fetching positions for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"{Fore.RED}Position Check: Unexpected error fetching positions for {symbol}: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())

    return default_pos  # Return default if any error occurs


def set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int) -> bool:
    """Sets leverage for a futures symbol, handling Bybit V5 API specifics.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol.
        leverage: The desired leverage value (integer).

    Returns:
        True if leverage was set successfully or already set, False otherwise.
    """
    logger.info(f"{Fore.CYAN}Leverage Setting: Attempting to set {leverage}x for {symbol}...{Style.RESET_ALL}")
    try:
        market = exchange.market(symbol)
        if not market.get('contract'):
            logger.error(f"{Fore.RED}Leverage Setting: Cannot set leverage for non-contract market: {symbol}.{Style.RESET_ALL}")
            return False
    except (ccxt.BadSymbol, KeyError, Exception) as e:
         logger.error(f"{Fore.RED}Leverage Setting: Failed to get market info for '{symbol}': {e}{Style.RESET_ALL}")
         return False

    for attempt in range(CONFIG.RETRY_COUNT):
        try:
            # Bybit V5 requires setting buyLeverage and sellLeverage separately via params
            # The main 'leverage' argument might also be needed depending on CCXT version/implementation
            params = {'buyLeverage': str(leverage), 'sellLeverage': str(leverage)}
            logger.debug(f"Leverage Setting: Calling set_leverage with leverage={leverage}, symbol={symbol}, params={params}")
            response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
            logger.success(f"{Fore.GREEN}Leverage Setting: Successfully set to {leverage}x for {symbol}. Response: {response}{Style.RESET_ALL}")
            return True
        except ccxt.ExchangeError as e:
            # Check for common Bybit messages indicating leverage is already set or not modified
            err_str = str(e).lower()
            # Example error codes/messages from Bybit V5 (these might change):
            # 110044: "Set leverage not modified"
            # Specific string checks:
            if "set leverage not modified" in err_str or "leverage is same as requested" in err_str:
                logger.info(f"{Fore.CYAN}Leverage Setting: Leverage already set to {leverage}x for {symbol}.{Style.RESET_ALL}")
                return True
            logger.warning(f"{Fore.YELLOW}Leverage Setting: Exchange error (Attempt {attempt + 1}/{CONFIG.RETRY_COUNT}): {e}{Style.RESET_ALL}")
            if attempt < CONFIG.RETRY_COUNT - 1:
                time.sleep(CONFIG.RETRY_DELAY_SECONDS)
            else:
                logger.error(f"{Fore.RED}Leverage Setting: Failed after {CONFIG.RETRY_COUNT} attempts due to ExchangeError.{Style.RESET_ALL}")
        except (ccxt.NetworkError, Exception) as e:
            logger.warning(f"{Fore.YELLOW}Leverage Setting: Network/Other error (Attempt {attempt + 1}/{CONFIG.RETRY_COUNT}): {e}{Style.RESET_ALL}")
            if attempt < CONFIG.RETRY_COUNT - 1:
                time.sleep(CONFIG.RETRY_DELAY_SECONDS)
            else:
                logger.error(f"{Fore.RED}Leverage Setting: Failed after {CONFIG.RETRY_COUNT} attempts due to {type(e).__name__}.{Style.RESET_ALL}")
    return False


def close_position(exchange: ccxt.Exchange, symbol: str, position_to_close: dict[str, Any], reason: str = "Signal") -> dict[str, Any] | None:
    """Closes the specified active position by placing a market order with reduceOnly=True.
    Re-validates the position just before closing.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol.
        position_to_close: A dictionary representing the position to close (from get_current_position).
        reason: A string indicating the reason for closing (for logging/alerts).

    Returns:
        The CCXT order dictionary if the close order was successfully placed, None otherwise.
    """
    initial_side = position_to_close.get('side', CONFIG.POS_NONE)
    initial_qty = position_to_close.get('qty', Decimal("0.0"))
    market_base = symbol.split('/')[0] if '/' in symbol else symbol
    logger.info(f"{Fore.YELLOW}Close Position: Initiated for {symbol}. Reason: {reason}. Initial state: {initial_side} Qty={initial_qty:.8f}{Style.RESET_ALL}")

    # === Re-validate the position just before closing ===
    logger.debug("Close Position: Re-validating current position state...")
    live_position = get_current_position(exchange, symbol)
    live_position_side = live_position['side']
    live_amount_to_close = live_position['qty']

    if live_position_side == CONFIG.POS_NONE or live_amount_to_close <= CONFIG.POSITION_QTY_EPSILON:
        logger.warning(f"{Fore.YELLOW}Close Position: Re-validation shows NO active position (or negligible size) for {symbol}. Aborting close attempt.{Style.RESET_ALL}")
        if initial_side != CONFIG.POS_NONE:
            logger.warning(f"{Fore.YELLOW}Close Position: Discrepancy detected (Bot thought position was {initial_side}, but exchange reports None/Zero).{Style.RESET_ALL}")
        return None  # Nothing to close

    if live_position_side != initial_side:
         logger.warning(f"{Fore.YELLOW}Close Position: Discrepancy detected! Initial side was {initial_side}, live side is {live_position_side}. Closing live position.{Style.RESET_ALL}")
         # Continue with closing the actual live position

    # Determine the side needed to close the position
    side_to_execute_close = CONFIG.SIDE_SELL if live_position_side == CONFIG.POS_LONG else CONFIG.SIDE_BUY

    try:
        # Format amount according to market precision
        amount_str = format_amount(exchange, symbol, live_amount_to_close)
        amount_decimal = safe_decimal_conversion(amount_str)  # Convert formatted string back to Decimal for check
        amount_float = float(amount_decimal)  # CCXT create order often expects float

        if amount_decimal <= CONFIG.POSITION_QTY_EPSILON:
            logger.error(f"{Fore.RED}Close Position: Closing amount '{amount_str}' after precision formatting is negligible. Aborting.{Style.RESET_ALL}")
            return None

        logger.warning(f"{Back.YELLOW}{Fore.BLACK}Close Position: Attempting to CLOSE {live_position_side} ({reason}): "
                       f"Exec {side_to_execute_close.upper()} MARKET {amount_str} {symbol} (reduce_only=True)...{Style.RESET_ALL}")

        # Set reduceOnly parameter for closing orders
        params = {'reduceOnly': True}
        order = exchange.create_market_order(
            symbol=symbol,
            side=side_to_execute_close,
            amount=amount_float,
            params=params
        )

        # Parse order response safely using Decimal
        order_id = order.get('id')
        order_id_short = format_order_id(order_id)
        status = order.get('status', 'unknown')
        filled_qty = safe_decimal_conversion(order.get('filled'))
        avg_fill_price = safe_decimal_conversion(order.get('average'))
        cost = safe_decimal_conversion(order.get('cost'))

        logger.success(f"{Fore.GREEN}{Style.BRIGHT}Close Position: Order ({reason}) submitted for {symbol}. "
                       f"ID:...{order_id_short}, Status: {status}, Filled: {filled_qty:.8f}/{amount_str}, AvgFill: {avg_fill_price:.4f}, Cost: {cost:.2f} USDT.{Style.RESET_ALL}")
        # Note: Market orders might fill immediately, but status might be 'open' initially.
        # We don't wait for fill confirmation here, assuming reduceOnly works reliably.

        send_sms_alert(f"[{market_base}] Closed {live_position_side} {amount_str} @ ~{avg_fill_price:.4f} ({reason}). ID:...{order_id_short}")
        return order  # Return the order details

    except ccxt.InsufficientFunds as e:
         logger.error(f"{Fore.RED}Close Position ({reason}): Insufficient funds for {symbol}: {e}{Style.RESET_ALL}")
         send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): Insufficient Funds. Check margin/position.")
    except ccxt.ExchangeError as e:
        # Check for specific Bybit errors indicating the position might already be closed or closing
        err_str = str(e).lower()
        # Example Bybit V5 error codes/messages (may change):
        # 110025: "Position size is zero" (or similar variations)
        # 110053: "The order would not reduce the position size"
        if "position size is zero" in err_str or \
           "order would not reduce position size" in err_str or \
           "position is already zero" in err_str:  # Add more known messages if needed
             logger.warning(f"{Fore.YELLOW}Close Position ({reason}): Exchange indicates position likely already closed/zero: {e}. Assuming closed.{Style.RESET_ALL}")
             # Don't send error SMS, treat as effectively closed.
             return None  # Treat as success (nothing to close) in this specific case
        else:
             logger.error(f"{Fore.RED}Close Position ({reason}): Exchange error for {symbol}: {e}{Style.RESET_ALL}")
             send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): Exchange Error: {type(e).__name__}. Check logs.")
    except (ccxt.NetworkError, ValueError, TypeError, Exception) as e:
        logger.error(f"{Fore.RED}Close Position ({reason}): Failed for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): {type(e).__name__}. Check logs.")

    return None  # Return None if closing failed


def calculate_position_size(
    equity: Decimal,
    risk_per_trade_pct: Decimal,
    entry_price: Decimal,
    stop_loss_price: Decimal,
    leverage: int,
    symbol: str,
    exchange: ccxt.Exchange
) -> tuple[Decimal | None, Decimal | None]:
    """Calculates the position size based on risk percentage, entry/stop prices, and equity.

    Args:
        equity: Total available equity in USDT (Decimal).
        risk_per_trade_pct: The fraction of equity to risk per trade (e.g., 0.01 for 1%).
        entry_price: Estimated entry price (Decimal).
        stop_loss_price: Calculated stop-loss price (Decimal).
        leverage: The leverage used for the trade (int).
        symbol: The market symbol.
        exchange: The CCXT exchange instance.

    Returns:
        A tuple containing:
        - Calculated position quantity (Decimal), formatted to market precision, or None if calculation fails.
        - Estimated required margin for the position (Decimal), or None.
    """
    logger.debug(f"Risk Calc: Equity={equity:.4f}, Risk%={risk_per_trade_pct:.4%}, Entry={entry_price:.4f}, SL={stop_loss_price:.4f}, Lev={leverage}x")

    # --- Input Validation ---
    if not (entry_price > 0 and stop_loss_price > 0):
        logger.error(f"{Fore.RED}Risk Calc Error: Invalid entry price ({entry_price}) or SL price ({stop_loss_price}). Must be positive.{Style.RESET_ALL}")
        return None, None
    price_diff = abs(entry_price - stop_loss_price)
    if price_diff < CONFIG.POSITION_QTY_EPSILON:
        logger.error(f"{Fore.RED}Risk Calc Error: Entry price ({entry_price}) and SL price ({stop_loss_price}) are too close (Diff: {price_diff:.8f}).{Style.RESET_ALL}")
        return None, None
    if not (0 < risk_per_trade_pct < 1):
        logger.error(f"{Fore.RED}Risk Calc Error: Invalid risk percentage: {risk_per_trade_pct:.4%}. Must be between 0 and 1 (exclusive).{Style.RESET_ALL}")
        return None, None
    if equity <= 0:
        logger.error(f"{Fore.RED}Risk Calc Error: Invalid equity: {equity:.4f}. Must be positive.{Style.RESET_ALL}")
        return None, None
    if leverage <= 0:
        logger.error(f"{Fore.RED}Risk Calc Error: Invalid leverage: {leverage}. Must be positive.{Style.RESET_ALL}")
        return None, None

    try:
        # --- Calculation ---
        risk_amount_usdt: Decimal = equity * risk_per_trade_pct
        # For linear contracts (like BTC/USDT:USDT), the value of 1 unit of base currency (BTC) is its price in quote currency (USDT).
        # The risk per unit of the base currency is the price difference between entry and stop-loss.
        # Quantity = (Total Risk Amount) / (Risk Per Unit)
        quantity_raw: Decimal = risk_amount_usdt / price_diff

        # --- Apply Precision ---
        # Format the raw quantity according to market rules *then* convert back to Decimal for further use
        quantity_precise_str = format_amount(exchange, symbol, quantity_raw)
        quantity_precise = safe_decimal_conversion(quantity_precise_str)

        if quantity_precise <= CONFIG.POSITION_QTY_EPSILON:
            logger.warning(f"{Fore.YELLOW}Risk Calc Warning: Calculated quantity ({quantity_precise:.8f}) is negligible or zero. "
                           f"RiskAmt={risk_amount_usdt:.4f}, PriceDiff={price_diff:.4f}. Cannot place order.{Style.RESET_ALL}")
            return None, None

        # --- Calculate Estimated Margin ---
        position_value_usdt = quantity_precise * entry_price
        required_margin = position_value_usdt / Decimal(leverage)

        logger.debug(f"Risk Calc Result: RawQty={quantity_raw:.8f} -> PreciseQty={quantity_precise:.8f}, EstValue={position_value_usdt:.4f}, EstMargin={required_margin:.4f}")
        return quantity_precise, required_margin

    except (DivisionByZero, InvalidOperation, OverflowError, Exception) as e:
        logger.error(f"{Fore.RED}Risk Calc Error: Unexpected exception during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        return None, None


def wait_for_order_fill(exchange: ccxt.Exchange, order_id: str, symbol: str, timeout_seconds: int) -> dict[str, Any] | None:
    """Waits for a specific order to reach a 'closed' (filled) status by polling the exchange.

    Args:
        exchange: The CCXT exchange instance.
        order_id: The ID of the order to wait for.
        symbol: The market symbol of the order.
        timeout_seconds: Maximum time to wait in seconds.

    Returns:
        The filled order dictionary if the order status becomes 'closed' within the timeout,
        None if the order fails, is cancelled, or times out.
    """
    start_time = time.monotonic()
    order_id_short = format_order_id(order_id)
    logger.info(f"{Fore.CYAN}Waiting for order ...{order_id_short} ({symbol}) fill (Timeout: {timeout_seconds}s)...{Style.RESET_ALL}")

    while time.monotonic() - start_time < timeout_seconds:
        try:
            # Fetch the order status
            order = exchange.fetch_order(order_id, symbol)
            status = order.get('status')
            logger.debug(f"Order ...{order_id_short} status: {status}")

            if status == 'closed':
                logger.success(f"{Fore.GREEN}Order ...{order_id_short} confirmed FILLED.{Style.RESET_ALL}")
                return order
            elif status in ['canceled', 'rejected', 'expired']:
                logger.error(f"{Fore.RED}Order ...{order_id_short} reached failure status: '{status}'.{Style.RESET_ALL}")
                return None  # Failed state
            # Continue polling if status is 'open', 'partiallyFilled', None, or other intermediate state

            time.sleep(0.5)  # Poll every 500ms

        except ccxt.OrderNotFound:
            # Can happen briefly after placing if the exchange hasn't registered it yet. Keep trying.
            elapsed = time.monotonic() - start_time
            logger.warning(f"{Fore.YELLOW}Order ...{order_id_short} not found yet (after {elapsed:.1f}s). Retrying...{Style.RESET_ALL}")
            time.sleep(1)  # Wait a bit longer if not found initially
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            elapsed = time.monotonic() - start_time
            logger.warning(f"{Fore.YELLOW}API Error checking order ...{order_id_short} (after {elapsed:.1f}s): {type(e).__name__} - {e}. Retrying...{Style.RESET_ALL}")
            time.sleep(CONFIG.RETRY_DELAY_SECONDS)  # Wait longer on API errors
        except Exception as e:
             elapsed = time.monotonic() - start_time
             logger.error(f"{Fore.RED}Unexpected error checking order ...{order_id_short} (after {elapsed:.1f}s): {e}. Stopping wait.{Style.RESET_ALL}")
             logger.debug(traceback.format_exc())
             return None  # Stop waiting on unexpected errors

    # If loop finishes without returning, it timed out
    logger.error(f"{Fore.RED}Order ...{order_id_short} did NOT fill within the {timeout_seconds}s timeout.{Style.RESET_ALL}")
    return None  # Timeout


def place_risked_market_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,  # CONFIG.SIDE_BUY or CONFIG.SIDE_SELL
    risk_percentage: Decimal,
    current_atr: Decimal | None,
    sl_atr_multiplier: Decimal,
    leverage: int,
    max_order_cap_usdt: Decimal,
    margin_check_buffer: Decimal,
    tsl_percent: Decimal,
    tsl_activation_offset_percent: Decimal
) -> dict[str, Any] | None:
    """Manages the full process of placing a market entry order with risk management,
    waiting for fill, and then placing exchange-native fixed Stop Loss and Trailing Stop Loss.

    Args:
        exchange: CCXT exchange instance.
        symbol: Market symbol.
        side: Order side ('buy' or 'sell').
        risk_percentage: Risk per trade as a decimal (e.g., 0.01 for 1%).
        current_atr: Current ATR value (Decimal) for SL calculation.
        sl_atr_multiplier: Multiplier for ATR to set initial SL distance.
        leverage: Leverage to use.
        max_order_cap_usdt: Maximum position value in USDT (Decimal).
        margin_check_buffer: Buffer multiplier for margin check (e.g., 1.05 for 5%).
        tsl_percent: Trailing stop percentage as a decimal (e.g., 0.005 for 0.5%).
        tsl_activation_offset_percent: Activation offset percentage from entry for TSL (Decimal).

    Returns:
        The filled entry order dictionary if the entry was successful, None otherwise.
        Note: Success/failure of SL/TSL placement is logged/alerted but doesn't change the return value
              if the entry itself was filled. Check logs for SL/TSL status.
    """
    market_base = symbol.split('/')[0] if '/' in symbol else symbol
    log_prefix = f"Place Order ({side.upper()})"
    logger.info(f"{Fore.BLUE}{Style.BRIGHT}{log_prefix}: Initiating for {symbol}...{Style.RESET_ALL}")

    # --- Preliminary Checks ---
    if current_atr is None or current_atr <= Decimal("0"):
        logger.error(f"{Fore.RED}{log_prefix}: Invalid ATR ({current_atr}). Cannot calculate SL or place order.{Style.RESET_ALL}")
        return None
    if side not in [CONFIG.SIDE_BUY, CONFIG.SIDE_SELL]:
        logger.error(f"{Fore.RED}{log_prefix}: Invalid order side '{side}'.{Style.RESET_ALL}")
        return None

    entry_price_estimate: Decimal | None = None
    initial_sl_price_estimate: Decimal | None = None
    final_quantity: Decimal | None = None
    market: dict | None = None
    filled_entry_order: dict[str, Any] | None = None  # Store the filled entry order

    try:
        # === Step 1: Get Balance, Market Info, Limits ===
        logger.debug(f"{log_prefix}: Fetching balance & market details...")
        balance = exchange.fetch_balance()
        market = exchange.market(symbol)  # Fetch market details once
        limits = market.get('limits', {})
        amount_limits = limits.get('amount', {})
        price_limits = limits.get('price', {})
        min_qty_str = amount_limits.get('min')
        max_qty_str = amount_limits.get('max')
        min_price_str = price_limits.get('min')
        min_qty = safe_decimal_conversion(min_qty_str, default=Decimal('0')) if min_qty_str else Decimal('0')
        max_qty = safe_decimal_conversion(max_qty_str) if max_qty_str else None  # Can be None if no max limit
        min_price = safe_decimal_conversion(min_price_str, default=Decimal('0')) if min_price_str else Decimal('0')

        # Get USDT balance (adjust symbol if using different quote currency)
        usdt_balance = balance.get(CONFIG.USDT_SYMBOL, {})
        # Prefer 'total' equity, fallback to 'free' if 'total' isn't available/zero
        usdt_total = safe_decimal_conversion(usdt_balance.get('total'))
        usdt_free = safe_decimal_conversion(usdt_balance.get('free'))
        usdt_equity = usdt_total if usdt_total > 0 else usdt_free

        if usdt_equity <= 0:
            logger.error(f"{Fore.RED}{log_prefix}: Zero or negative equity ({usdt_equity:.4f}). Cannot place order.{Style.RESET_ALL}")
            return None
        if usdt_free < 0:  # Free margin should not be negative
             logger.error(f"{Fore.RED}{log_prefix}: Negative free margin ({usdt_free:.4f}). Cannot place order.{Style.RESET_ALL}")
             return None
        logger.debug(f"{log_prefix}: Equity={usdt_equity:.4f} USDT, Free={usdt_free:.4f} USDT")

        # === Step 2: Estimate Entry Price (for size calculation) ===
        # Use shallow OB fetch for a quick estimate, fallback to ticker
        logger.debug(f"{log_prefix}: Estimating entry price...")
        ob_data = analyze_order_book(exchange, symbol, CONFIG.shallow_ob_fetch_depth, CONFIG.shallow_ob_fetch_depth)
        best_ask = ob_data.get("best_ask")
        best_bid = ob_data.get("best_bid")
        if side == CONFIG.SIDE_BUY and best_ask:
            entry_price_estimate = best_ask
        elif side == CONFIG.SIDE_SELL and best_bid:
            entry_price_estimate = best_bid
        else:
            try:
                ticker = exchange.fetch_ticker(symbol)
                entry_price_estimate = safe_decimal_conversion(ticker.get('last'))
                if not entry_price_estimate or entry_price_estimate <= 0: raise ValueError("Invalid last price from ticker")
                logger.debug(f"{log_prefix}: Used last ticker price for estimate: {entry_price_estimate:.4f}")
            except (ccxt.NetworkError, ccxt.ExchangeError, ValueError, KeyError, Exception) as e:
                logger.error(f"{Fore.RED}{log_prefix}: Failed to fetch valid ticker price for estimate: {e}{Style.RESET_ALL}")
                return None
        logger.info(f"{log_prefix}: Estimated Entry Price ~ {entry_price_estimate:.4f}")

        # === Step 3: Calculate Initial Stop Loss Price (Estimate) ===
        sl_distance = current_atr * sl_atr_multiplier
        if side == CONFIG.SIDE_BUY:
            initial_sl_price_raw = entry_price_estimate - sl_distance
        else:  # side == CONFIG.SIDE_SELL
            initial_sl_price_raw = entry_price_estimate + sl_distance

        # Ensure SL is not below minimum price tick (or zero)
        if min_price > 0 and initial_sl_price_raw < min_price:
             logger.warning(f"{Fore.YELLOW}{log_prefix}: Raw SL price {initial_sl_price_raw:.4f} below min price {min_price:.4f}. Adjusting SL to min price.{Style.RESET_ALL}")
             initial_sl_price_raw = min_price
        elif initial_sl_price_raw <= 0:
            logger.error(f"{Fore.RED}{log_prefix}: Calculated initial SL price is zero or negative ({initial_sl_price_raw:.4f}). Cannot proceed.{Style.RESET_ALL}")
            return None

        # Format estimated SL price
        initial_sl_price_estimate_str = format_price(exchange, symbol, initial_sl_price_raw)
        initial_sl_price_estimate = safe_decimal_conversion(initial_sl_price_estimate_str)
        logger.info(f"{log_prefix}: Calculated Initial SL Price (Estimate) ~ {initial_sl_price_estimate:.4f} (Dist: {sl_distance:.4f})")

        # === Step 4: Calculate Position Size based on Risk ===
        logger.debug(f"{log_prefix}: Calculating position size based on risk...")
        calc_qty, req_margin_estimate = calculate_position_size(
            usdt_equity, risk_percentage, entry_price_estimate, initial_sl_price_estimate, leverage, symbol, exchange
        )
        if calc_qty is None or req_margin_estimate is None:
            logger.error(f"{Fore.RED}{log_prefix}: Failed risk calculation. Cannot determine position size.{Style.RESET_ALL}")
            return None
        final_quantity = calc_qty  # Start with risk-based quantity

        # === Step 5: Apply Max Order Value Cap ===
        pos_value_estimate = final_quantity * entry_price_estimate
        if pos_value_estimate > max_order_cap_usdt:
            logger.warning(f"{Fore.YELLOW}{log_prefix}: Estimated position value {pos_value_estimate:.4f} USDT exceeds cap {max_order_cap_usdt:.4f} USDT. Capping quantity.{Style.RESET_ALL}")
            capped_quantity_raw = max_order_cap_usdt / entry_price_estimate
            # Format the capped quantity according to market precision
            capped_quantity_str = format_amount(exchange, symbol, capped_quantity_raw)
            final_quantity = safe_decimal_conversion(capped_quantity_str)
            # Recalculate estimated margin based on capped quantity
            req_margin_estimate = (final_quantity * entry_price_estimate) / Decimal(leverage)
            logger.info(f"{log_prefix}: Quantity capped to {final_quantity:.8f}, New Est. Margin ~{req_margin_estimate:.4f} USDT")

        # === Step 6: Check Limits & Final Margin Check ===
        if final_quantity <= CONFIG.POSITION_QTY_EPSILON:
            logger.error(f"{Fore.RED}{log_prefix}: Final quantity ({final_quantity:.8f}) is negligible or zero after calculations/caps. Aborting.{Style.RESET_ALL}")
            return None
        if min_qty > 0 and final_quantity < min_qty:
            logger.error(f"{Fore.RED}{log_prefix}: Final quantity {final_quantity:.8f} is below market minimum {min_qty:.8f}. Cannot place order.{Style.RESET_ALL}")
            # Consider adjusting risk % or raising an alert if this happens often
            return None
        if max_qty is not None and final_quantity > max_qty:
            logger.warning(f"{Fore.YELLOW}{log_prefix}: Final quantity {final_quantity:.8f} exceeds market maximum {max_qty:.8f}. Capping to max.{Style.RESET_ALL}")
            final_quantity = max_qty
            # Re-format capped amount (important if max_qty wasn't already precise)
            final_quantity = safe_decimal_conversion(format_amount(exchange, symbol, final_quantity))
            # Recalculate estimated margin based on max quantity
            req_margin_estimate = (final_quantity * entry_price_estimate) / Decimal(leverage)
            logger.info(f"{log_prefix}: Quantity capped to {final_quantity:.8f}, New Est. Margin ~{req_margin_estimate:.4f} USDT")

        # Final margin check with buffer
        final_req_margin_buffered = req_margin_estimate * margin_check_buffer
        if usdt_free < final_req_margin_buffered:
            logger.error(f"{Fore.RED}{log_prefix}: Insufficient FREE margin. Need ~{final_req_margin_buffered:.4f} USDT (incl. buffer), Have {usdt_free:.4f} USDT.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Insufficient Free Margin (Need ~{final_req_margin_buffered:.2f}, Have {usdt_free:.2f})")
            return None
        logger.info(f"{Fore.GREEN}{log_prefix}: Final Order Quantity={final_quantity:.8f}, Est. Value={final_quantity * entry_price_estimate:.4f}, Est. Margin={req_margin_estimate:.4f}. Margin check OK.{Style.RESET_ALL}")

        # === Step 7: Place Entry Market Order ===
        entry_order_id: str | None = None
        try:
            qty_float = float(final_quantity)  # CCXT create order often expects float
            logger.warning(f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}*** {log_prefix}: Placing MARKET ENTRY: {qty_float:.8f} {symbol} ***{Style.RESET_ALL}")
            # Use reduce_only=False explicitly for entry orders
            entry_order = exchange.create_market_order(symbol=symbol, side=side, amount=qty_float, params={'reduce_only': False})
            entry_order_id = entry_order.get('id')
            if not entry_order_id:
                logger.error(f"{Fore.RED}{log_prefix}: Entry order placed but no ID was returned! Response: {entry_order}{Style.RESET_ALL}")
                # Attempt cleanup if possible, but state is uncertain
                return None
            logger.success(f"{Fore.GREEN}{log_prefix}: Market Entry Order submitted. ID: ...{format_order_id(entry_order_id)}. Waiting for fill confirmation...{Style.RESET_ALL}")
        except (ccxt.InsufficientFunds, ccxt.ExchangeError, ccxt.NetworkError, ValueError, Exception) as e:
            logger.error(f"{Fore.RED}{Style.BRIGHT}{log_prefix}: FAILED TO PLACE ENTRY ORDER: {e}{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Entry placement failed: {type(e).__name__}")
            return None  # Stop process if entry order fails

        # === Step 8: Wait for Entry Order Fill ===
        filled_entry_order = wait_for_order_fill(exchange, entry_order_id, symbol, CONFIG.order_fill_timeout_seconds)
        if not filled_entry_order:
            logger.error(f"{Fore.RED}{log_prefix}: Entry order ...{format_order_id(entry_order_id)} did not fill or failed.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Entry ...{format_order_id(entry_order_id)} fill timeout/fail.")
            # Try to cancel the potentially stuck order (best effort)
            try:
                logger.warning(f"{log_prefix}: Attempting to cancel unfilled/failed entry order ...{format_order_id(entry_order_id)}.")
                exchange.cancel_order(entry_order_id, symbol)
            except Exception as cancel_err:
                logger.warning(f"{log_prefix}: Failed to cancel stuck entry order: {cancel_err}")
            return None  # Stop process if entry doesn't fill

        # === Step 9: Extract Actual Fill Details ===
        # CRITICAL: Use details from the *actual filled order*, not estimates
        avg_fill_price = safe_decimal_conversion(filled_entry_order.get('average'))
        filled_qty = safe_decimal_conversion(filled_entry_order.get('filled'))
        cost = safe_decimal_conversion(filled_entry_order.get('cost'))

        # Validate fill details
        if avg_fill_price <= 0 or filled_qty <= CONFIG.POSITION_QTY_EPSILON:
            logger.error(f"{Fore.RED}{log_prefix}: Invalid fill details for entry ...{format_order_id(entry_order_id)}: Price={avg_fill_price}, Qty={filled_qty}. Position state uncertain!{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Invalid fill details for entry ...{format_order_id(entry_order_id)}.")
            # Return the problematic order details, but signal that subsequent steps failed
            return filled_entry_order  # Indicate entry happened but subsequent steps might fail

        logger.success(f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}{log_prefix}: ENTRY CONFIRMED FILLED: ...{format_order_id(entry_order_id)}. FilledQty: {filled_qty:.8f} @ AvgPrice: {avg_fill_price:.4f}, Cost: {cost:.4f} USDT{Style.RESET_ALL}")

        # --- Post-Entry: Place SL and TSL ---
        sl_order_id_short = "N/A"
        tsl_order_id_short = "N/A"
        actual_sl_price_str = "N/A"
        tsl_act_price_str = "N/A"

        # === Step 10: Calculate ACTUAL Stop Loss Price based on Actual Fill ===
        if side == CONFIG.SIDE_BUY:
            actual_sl_price_raw = avg_fill_price - sl_distance
        else:  # side == CONFIG.SIDE_SELL
            actual_sl_price_raw = avg_fill_price + sl_distance

        # Ensure SL is valid after using actual fill price
        if min_price > 0 and actual_sl_price_raw < min_price:
             logger.warning(f"{Fore.YELLOW}{log_prefix}: Actual SL price {actual_sl_price_raw:.4f} below min price {min_price:.4f}. Adjusting SL to min price.{Style.RESET_ALL}")
             actual_sl_price_raw = min_price
        elif actual_sl_price_raw <= 0:
            logger.error(f"{Fore.RED}{Style.BRIGHT}{log_prefix}: Invalid ACTUAL SL price calculated ({actual_sl_price_raw:.4f}) based on fill price {avg_fill_price:.4f}. Cannot place SL!{Style.RESET_ALL}")
            # CRITICAL SITUATION: Position is open without a calculated SL. Attempt emergency close.
            send_sms_alert(f"[{market_base}] CRITICAL ({side.upper()}): Invalid ACTUAL SL price ({actual_sl_price_raw:.4f})! Attempting emergency close.")
            # Determine position side based on entry order side
            position_side = CONFIG.POS_LONG if side == CONFIG.SIDE_BUY else CONFIG.POS_SHORT
            close_position(exchange, symbol, {'side': position_side, 'qty': filled_qty}, reason="Invalid SL Calc Post-Entry")
            return filled_entry_order  # Return filled entry, but indicate failure state

        # Format the valid actual SL price
        actual_sl_price_str = format_price(exchange, symbol, actual_sl_price_raw)
        actual_sl_price_float = float(actual_sl_price_str)  # For CCXT param

        # === Step 11: Place Initial Fixed Stop Loss ===
        try:
            sl_side = CONFIG.SIDE_SELL if side == CONFIG.SIDE_BUY else CONFIG.SIDE_BUY
            # Use the actual filled quantity for the SL order
            sl_qty_str = format_amount(exchange, symbol, filled_qty)
            sl_qty_float = float(sl_qty_str)

            logger.info(f"{Fore.CYAN}{log_prefix}: Placing Initial Fixed SL ({sl_atr_multiplier}*ATR)... Side: {sl_side.upper()}, Qty: {sl_qty_float:.8f}, StopPx: {actual_sl_price_str}{Style.RESET_ALL}")
            # Bybit V5 stop order params: 'stopPrice' (trigger price), 'reduceOnly': True
            sl_params = {'stopPrice': actual_sl_price_float, 'reduceOnly': True}
            # Use 'stopMarket' type for market stop loss order
            sl_order = exchange.create_order(symbol, 'stopMarket', sl_side, sl_qty_float, params=sl_params)
            sl_order_id = sl_order.get('id')
            sl_order_id_short = format_order_id(sl_order_id)
            logger.success(f"{Fore.GREEN}{log_prefix}: Initial Fixed SL order placed. ID: ...{sl_order_id_short}, Trigger: {actual_sl_price_str}{Style.RESET_ALL}")
        except (ccxt.InsufficientFunds, ccxt.ExchangeError, ccxt.NetworkError, ValueError, Exception) as e:
            logger.error(f"{Fore.RED}{Style.BRIGHT}{log_prefix}: FAILED to place Initial Fixed SL order: {e}{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())
            send_sms_alert(f"[{market_base}] ERROR ({side.upper()}): Failed initial SL placement ({type(e).__name__}). Pos unprotected by fixed SL!")
            # Don't necessarily close here, TSL might still work, or manual intervention needed.

        # === Step 12: Place Trailing Stop Loss ===
        try:
            # Calculate TSL activation price based on actual fill price
            act_offset = avg_fill_price * tsl_activation_offset_percent
            if side == CONFIG.SIDE_BUY:
                act_price_raw = avg_fill_price + act_offset
            else:  # side == CONFIG.SIDE_SELL
                act_price_raw = avg_fill_price - act_offset

            # Ensure activation price is valid
            if min_price > 0 and act_price_raw < min_price: act_price_raw = min_price
            if act_price_raw <= 0:
                raise ValueError(f"Invalid TSL activation price calculated: {act_price_raw:.4f}")

            tsl_act_price_str = format_price(exchange, symbol, act_price_raw)
            tsl_act_price_float = float(tsl_act_price_str)
            tsl_side = CONFIG.SIDE_SELL if side == CONFIG.SIDE_BUY else CONFIG.SIDE_BUY
            # Bybit V5 uses 'trailingStop' for percentage distance (e.g., "0.5" for 0.5%)
            # Convert decimal percentage (0.005) to string percentage ("0.5")
            tsl_trail_value_str = str((tsl_percent * Decimal("100")).quantize(Decimal("0.01")))  # Adjust precision as needed by Bybit
            # Use the actual filled quantity for the TSL order
            tsl_qty_str = format_amount(exchange, symbol, filled_qty)
            tsl_qty_float = float(tsl_qty_str)

            logger.info(f"{Fore.CYAN}{log_prefix}: Placing Trailing SL ({tsl_percent:.2%})... Side: {tsl_side.upper()}, Qty: {tsl_qty_float:.8f}, TrailValue: {tsl_trail_value_str}%, ActPx: {tsl_act_price_str}{Style.RESET_ALL}")

            # Bybit V5 TSL parameters via CCXT:
            # 'trailingStop': The trailing percentage/distance as a string (check Bybit docs for exact format).
            # 'activePrice': The price at which the trailing stop activates.
            # 'reduceOnly': True
            tsl_params = {
                'trailingStop': tsl_trail_value_str,  # String percentage for Bybit V5
                'activePrice': tsl_act_price_float,
                'reduceOnly': True,
            }
            # Use 'stopMarket' type with TSL params for Bybit V5 via CCXT (check CCXT Bybit overrides if needed)
            tsl_order = exchange.create_order(symbol, 'stopMarket', tsl_side, tsl_qty_float, params=tsl_params)
            tsl_order_id = tsl_order.get('id')
            tsl_order_id_short = format_order_id(tsl_order_id)
            logger.success(f"{Fore.GREEN}{log_prefix}: Trailing SL order placed. ID: ...{tsl_order_id_short}, Trail: {tsl_trail_value_str}%, ActPx: {tsl_act_price_str}{Style.RESET_ALL}")

            # Final comprehensive SMS Alert after successful entry and SL/TSL placement attempts
            sms_msg = (f"[{market_base}] ENTERED {side.upper()} {filled_qty:.8f} @ {avg_fill_price:.4f}. "
                       f"Init SL ~{actual_sl_price_str} (ID:...{sl_order_id_short}). "
                       f"TSL {tsl_percent:.2%} act@{tsl_act_price_str} (ID:...{tsl_order_id_short}). "
                       f"EntryID:...{format_order_id(entry_order_id)}")
            send_sms_alert(sms_msg)

        except (ccxt.InsufficientFunds, ccxt.ExchangeError, ccxt.NetworkError, ValueError, Exception) as e:
            logger.error(f"{Fore.RED}{Style.BRIGHT}{log_prefix}: FAILED to place Trailing SL order: {e}{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())
            send_sms_alert(f"[{market_base}] ERROR ({side.upper()}): Failed TSL placement ({type(e).__name__}). Pos may lack TSL protection.")
            # If TSL fails but initial SL was placed, the position is still protected initially.

        return filled_entry_order  # Return filled entry order details regardless of SL/TSL placement success/failure

    except (ccxt.InsufficientFunds, ccxt.NetworkError, ccxt.ExchangeError, ValueError, Exception) as e:
        # Catch errors occurring before entry order placement or during initial setup
        logger.error(f"{Fore.RED}{Style.BRIGHT}{log_prefix}: Overall process failed before/during entry: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Pre-entry/Setup failed: {type(e).__name__}")
        return None  # Indicate failure


def cancel_open_orders(exchange: ccxt.Exchange, symbol: str, reason: str = "Cleanup") -> None:
    """Attempts to cancel all open orders for the specified symbol on the exchange.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol for which to cancel orders.
        reason: A string indicating why orders are being cancelled (for logging).
    """
    logger.info(f"{Fore.CYAN}Order Cancel: Attempting for {symbol} (Reason: {reason})...{Style.RESET_ALL}")
    try:
        if not exchange.has.get('fetchOpenOrders'):
            logger.warning(f"{Fore.YELLOW}Order Cancel: fetchOpenOrders not supported by {exchange.id}. Cannot cancel automatically.{Style.RESET_ALL}")
            return

        # Fetch only open orders for the specific symbol
        open_orders = exchange.fetch_open_orders(symbol)

        if not open_orders:
            logger.info(f"{Fore.CYAN}Order Cancel: No open orders found for {symbol}.{Style.RESET_ALL}")
            return

        logger.warning(f"{Fore.YELLOW}Order Cancel: Found {len(open_orders)} open order(s) for {symbol}. Cancelling (Reason: {reason})...{Style.RESET_ALL}")
        cancelled_count = 0
        failed_count = 0
        for order in open_orders:
            order_id = order.get('id')
            order_info_str = f"ID:...{format_order_id(order_id)} ({order.get('type')} {order.get('side')} Qty:{order.get('amount')} Px:{order.get('price') or order.get('stopPrice')})"
            if order_id:
                try:
                    logger.debug(f"Order Cancel: Attempting to cancel order {order_info_str}")
                    exchange.cancel_order(order_id, symbol)
                    logger.info(f"{Fore.CYAN}Order Cancel: Successfully cancelled order {order_info_str}{Style.RESET_ALL}")
                    cancelled_count += 1
                    time.sleep(0.1)  # Small delay between cancel calls to avoid rate limits
                except ccxt.OrderNotFound:
                    # If the order is not found, it might have been filled or cancelled already.
                    logger.warning(f"{Fore.YELLOW}Order Cancel: Order not found (already closed/cancelled?): {order_info_str}{Style.RESET_ALL}")
                    cancelled_count += 1  # Count it as effectively cancelled if not found
                except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                    logger.error(f"{Fore.RED}Order Cancel: FAILED to cancel order {order_info_str}: {type(e).__name__} - {e}{Style.RESET_ALL}")
                    failed_count += 1
                except Exception as e:
                     logger.error(f"{Fore.RED}Order Cancel: Unexpected error cancelling {order_info_str}: {e}{Style.RESET_ALL}")
                     logger.debug(traceback.format_exc())
                     failed_count += 1
            else:
                logger.warning(f"{Fore.YELLOW}Order Cancel: Found an open order without an ID: {order}. Skipping cancellation.{Style.RESET_ALL}")

        log_level = logging.INFO if failed_count == 0 else logging.WARNING
        logger.log(log_level, f"{Fore.CYAN}Order Cancel: Finished for {symbol}. Cancelled/Not Found: {cancelled_count}, Failed: {failed_count}.{Style.RESET_ALL}")

        if failed_count > 0:
            market_base = symbol.split('/')[0] if '/' in symbol else symbol
            send_sms_alert(f"[{market_base}] WARNING: Failed to cancel {failed_count} open order(s) during {reason}. Manual check may be needed.")

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger.error(f"{Fore.RED}Order Cancel: Failed to fetch open orders for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"{Fore.RED}Order Cancel: Unexpected error during open order fetch/cancel process for {symbol}: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())


# --- Strategy Signal Generation ---
def generate_signals(df: pd.DataFrame, strategy_name: str) -> dict[str, Any]:
    """Generates entry and exit signals based on the selected strategy and calculated indicators in the DataFrame.

    Args:
        df: DataFrame containing OHLCV data and calculated indicator columns.
        strategy_name: The name of the strategy to use (from CONFIG.strategy_name).

    Returns:
        A dictionary containing boolean signals:
        - 'enter_long': True to enter a long position.
        - 'enter_short': True to enter a short position.
        - 'exit_long': True to exit an existing long position.
        - 'exit_short': True to exit an existing short position.
        - 'exit_reason': A string describing the reason for the exit signal.
    """
    signals = {'enter_long': False, 'enter_short': False, 'exit_long': False, 'exit_short': False, 'exit_reason': "Strategy Exit"}
    log_prefix = f"Signal Gen ({strategy_name})"

    # Need at least 2 rows for comparisons (current and previous candle)
    if df is None or len(df) < 2:
        logger.debug(f"{log_prefix}: Insufficient data length ({len(df) if df is not None else 0}) for signal generation.")
        return signals

    last = df.iloc[-1]  # Current (latest) candle data
    prev = df.iloc[-2]  # Previous candle data

    try:
        # --- Dual Supertrend Logic ---
        if strategy_name == "DUAL_SUPERTREND":
            # Entry: Primary ST flips long AND confirmation ST is also long (trend=True)
            if pd.notna(last['st_long']) and last['st_long'] and pd.notna(last['confirm_trend']) and last['confirm_trend']:
                signals['enter_long'] = True
            # Entry: Primary ST flips short AND confirmation ST is also short (trend=False)
            if pd.notna(last['st_short']) and last['st_short'] and pd.notna(last['confirm_trend']) and not last['confirm_trend']:
                signals['enter_short'] = True
            # Exit Long: Primary ST flips short
            if pd.notna(last['st_short']) and last['st_short']:
                signals['exit_long'] = True
                signals['exit_reason'] = "Primary ST Short Flip"
            # Exit Short: Primary ST flips long
            if pd.notna(last['st_long']) and last['st_long']:
                signals['exit_short'] = True
                signals['exit_reason'] = "Primary ST Long Flip"

        # --- StochRSI + Momentum Logic ---
        elif strategy_name == "STOCHRSI_MOMENTUM":
            k_now, d_now, mom_now = last.get('stochrsi_k'), last.get('stochrsi_d'), last.get('momentum')
            k_prev, d_prev = prev.get('stochrsi_k'), prev.get('stochrsi_d')

            # Check if all required values are valid Decimals
            required_vals = [k_now, d_now, mom_now, k_prev, d_prev]
            if any(v is None or not isinstance(v, Decimal) or v.is_nan() for v in required_vals):
                logger.debug(f"{log_prefix}: Skipping due to missing/NaN StochRSI/Momentum values.")
                return signals

            # Entry Long: K crosses above D from below, K is below oversold, Momentum is positive
            if k_prev <= d_prev and k_now > d_now and k_now < CONFIG.stochrsi_oversold and mom_now > CONFIG.POSITION_QTY_EPSILON:
                signals['enter_long'] = True
            # Entry Short: K crosses below D from above, K is above overbought, Momentum is negative
            if k_prev >= d_prev and k_now < d_now and k_now > CONFIG.stochrsi_overbought and mom_now < -CONFIG.POSITION_QTY_EPSILON:
                signals['enter_short'] = True
            # Exit Long: K crosses below D
            if k_prev >= d_prev and k_now < d_now:
                signals['exit_long'] = True
                signals['exit_reason'] = "StochRSI K crossed below D"
            # Exit Short: K crosses above D
            if k_prev <= d_prev and k_now > d_now:
                signals['exit_short'] = True
                signals['exit_reason'] = "StochRSI K crossed above D"

        # --- Ehlers Fisher Logic ---
        elif strategy_name == "EHLERS_FISHER":
            fish_now, sig_now = last.get('ehlers_fisher'), last.get('ehlers_signal')
            fish_prev, sig_prev = prev.get('ehlers_fisher'), prev.get('ehlers_signal')

            required_vals = [fish_now, sig_now, fish_prev, sig_prev]
            if any(v is None or not isinstance(v, Decimal) or v.is_nan() for v in required_vals):
                logger.debug(f"{log_prefix}: Skipping due to missing/NaN Ehlers Fisher values.")
                return signals

            # Entry Long: Fisher crosses above Signal line
            if fish_prev <= sig_prev and fish_now > sig_now:
                signals['enter_long'] = True
            # Entry Short: Fisher crosses below Signal line
            if fish_prev >= sig_prev and fish_now < sig_now:
                signals['enter_short'] = True
            # Exit Long: Fisher crosses below Signal line
            if fish_prev >= sig_prev and fish_now < sig_now:
                signals['exit_long'] = True
                signals['exit_reason'] = "Ehlers Fisher crossed below Signal"
            # Exit Short: Fisher crosses above Signal line
            if fish_prev <= sig_prev and fish_now > sig_now:
                signals['exit_short'] = True
                signals['exit_reason'] = "Ehlers Fisher crossed above Signal"

        # --- Ehlers MA Cross Logic (Using EMA placeholder) ---
        elif strategy_name == "EHLERS_MA_CROSS":
            fast_ma_now, slow_ma_now = last.get('fast_ema'), last.get('slow_ema')
            fast_ma_prev, slow_ma_prev = prev.get('fast_ema'), prev.get('slow_ema')

            required_vals = [fast_ma_now, slow_ma_now, fast_ma_prev, slow_ma_prev]
            if any(v is None or not isinstance(v, Decimal) or v.is_nan() for v in required_vals):
                logger.debug(f"{log_prefix}: Skipping due to missing/NaN Ehlers MA (EMA placeholder) values.")
                return signals

            # Entry Long: Fast MA crosses above Slow MA
            if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now:
                signals['enter_long'] = True
            # Entry Short: Fast MA crosses below Slow MA
            if fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now:
                signals['enter_short'] = True
            # Exit Long: Fast MA crosses below Slow MA
            if fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now:
                signals['exit_long'] = True
                signals['exit_reason'] = "Fast MA crossed below Slow MA"
            # Exit Short: Fast MA crosses above Slow MA
            if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now:
                signals['exit_short'] = True
                signals['exit_reason'] = "Fast MA crossed above Slow MA"

        else:
            logger.warning(f"{log_prefix}: Unknown strategy name '{strategy_name}' provided.")

    except KeyError as e:
        logger.error(f"{Fore.RED}{log_prefix} Error: Missing expected indicator column in DataFrame: {e}. Ensure indicators are calculated correctly.{Style.RESET_ALL}")
        # Reset signals to False if a required column is missing
        signals = {k: (v if k == 'exit_reason' else False) for k, v in signals.items()}
    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix} Error: Unexpected exception during signal generation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        signals = {k: (v if k == 'exit_reason' else False) for k, v in signals.items()}

    return signals


# --- Trading Logic ---
def trade_logic(exchange: ccxt.Exchange, symbol: str, df: pd.DataFrame) -> None:
    """Executes the main trading logic for one cycle:
    1. Calculates all indicators.
    2. Checks current position and market state.
    3. Generates strategy signals.
    4. Executes exit or entry actions based on signals and confirmations.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol.
        df: DataFrame containing the latest OHLCV data.
    """
    cycle_time_str = df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z') if not df.empty else "N/A"
    logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== New Check Cycle ({CONFIG.strategy_name}): {symbol} | Candle: {cycle_time_str} =========={Style.RESET_ALL}")

    # --- Determine Minimum Required Data Length ---
    # Find the longest lookback period required by any *potentially* used indicator based on config
    # Add a buffer for safety and potential multi-step calculations
    required_rows = max(
        CONFIG.st_atr_length, CONFIG.confirm_st_atr_length,
        CONFIG.stochrsi_rsi_length + CONFIG.stochrsi_stoch_length + max(CONFIG.stochrsi_k_period, CONFIG.stochrsi_d_period),  # StochRSI lookback estimate
        CONFIG.momentum_length,
        CONFIG.ehlers_fisher_length + CONFIG.ehlers_fisher_signal_length,  # Fisher lookback estimate
        CONFIG.ehlers_fast_period, CONFIG.ehlers_slow_period,
        CONFIG.atr_calculation_period, CONFIG.volume_ma_period
    ) + 10  # General buffer

    if df is None or len(df) < required_rows:
        logger.warning(f"{Fore.YELLOW}Trade Logic: Insufficient data ({len(df) if df is not None else 0} rows, need ~{required_rows} for indicators). Skipping cycle.{Style.RESET_ALL}")
        return

    try:
        # === Step 1: Calculate ALL Required Indicators ===
        # Calculate indicators needed for the selected strategy and potentially for confirmations/SL.
        # It's often simpler to calculate a common set and let the signal function use what it needs.
        logger.debug("Calculating indicators...")
        # Always calculate ATR/Volume for SL and potential confirmation
        vol_atr_data = analyze_volume_atr(df, CONFIG.atr_calculation_period, CONFIG.volume_ma_period)
        current_atr = vol_atr_data.get("atr")

        # Calculate strategy-specific indicators
        if CONFIG.strategy_name == "DUAL_SUPERTREND":
            df = calculate_supertrend(df, CONFIG.st_atr_length, CONFIG.st_multiplier)
            df = calculate_supertrend(df, CONFIG.confirm_st_atr_length, CONFIG.confirm_st_multiplier, prefix="confirm_")
        elif CONFIG.strategy_name == "STOCHRSI_MOMENTUM":
            df = calculate_stochrsi_momentum(df, CONFIG.stochrsi_rsi_length, CONFIG.stochrsi_stoch_length, CONFIG.stochrsi_k_period, CONFIG.stochrsi_d_period, CONFIG.momentum_length)
        elif CONFIG.strategy_name == "EHLERS_FISHER":
            df = calculate_ehlers_fisher(df, CONFIG.ehlers_fisher_length, CONFIG.ehlers_fisher_signal_length)
        elif CONFIG.strategy_name == "EHLERS_MA_CROSS":
            df = calculate_ehlers_ma(df, CONFIG.ehlers_fast_period, CONFIG.ehlers_slow_period)
        # Add other strategy indicator calculations here if needed

        # === Step 2: Validate Base Requirements for Trading ===
        last_candle = df.iloc[-1]
        current_price = safe_decimal_conversion(last_candle.get('close'), default=Decimal('NaN'))

        if current_price.is_nan() or current_price <= 0:
            logger.warning(f"{Fore.YELLOW}Trade Logic: Last candle close price is invalid ({current_price}). Skipping cycle.{Style.RESET_ALL}")
            return

        # Check if ATR is valid for placing new orders (needed for SL calculation)
        can_place_new_order = current_atr is not None and not current_atr.is_nan() and current_atr > Decimal("0")
        if not can_place_new_order:
            logger.warning(f"{Fore.YELLOW}Trade Logic: Invalid ATR ({current_atr}). Cannot calculate SL, new order placement disabled this cycle.{Style.RESET_ALL}")
            # Note: Exits might still be possible if triggered by strategy without needing ATR.

        # === Step 3: Get Current Position & Optional Order Book Data ===
        position = get_current_position(exchange, symbol)
        position_side = position['side']
        position_qty = position['qty']
        position_entry_price = position['entry_price']

        ob_data: dict[str, Decimal | None] | None = None
        if CONFIG.fetch_order_book_per_cycle or CONFIG.require_volume_spike_for_entry:  # Fetch if needed for entry confirmation
             ob_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit)

        # === Step 4: Log Current State ===
        vol_ratio = vol_atr_data.get("volume_ratio")
        vol_spike = vol_ratio is not None and not vol_ratio.is_nan() and vol_ratio > CONFIG.volume_spike_threshold
        bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None
        spread = ob_data.get("spread") if ob_data else None

        atr_str = f"{current_atr:.5f}" if can_place_new_order else 'N/A'
        logger.info(f"State | Price: {current_price:.4f}, ATR({CONFIG.atr_calculation_period}): {atr_str}")
        vol_ratio_str = f"{vol_ratio:.2f}" if vol_ratio is not None and not vol_ratio.is_nan() else 'N/A'
        logger.info(f"State | Volume: Ratio={vol_ratio_str}, Spike={vol_spike} (RequiredForEntry={CONFIG.require_volume_spike_for_entry})")
        ob_ratio_str = f"{bid_ask_ratio:.3f}" if bid_ask_ratio is not None and not bid_ask_ratio.is_nan() else 'N/A'
        spread_str = f"{spread:.4f}" if spread is not None and not spread.is_nan() else 'N/A'
        logger.info(f"State | OrderBook: Ratio={ob_ratio_str}, Spread={spread_str} (Fetched={ob_data is not None})")
        logger.info(f"State | Position: Side={position_side}, Qty={position_qty:.8f}, Entry={position_entry_price:.4f}")

        # === Step 5: Generate Strategy Signals ===
        strategy_signals = generate_signals(df, CONFIG.strategy_name)
        logger.debug(f"Strategy Signals ({CONFIG.strategy_name}): {strategy_signals}")

        # === Step 6: Execute Exit Actions (If in Position) ===
        if position_side != CONFIG.POS_NONE:
            should_exit = False
            exit_reason = ""
            if position_side == CONFIG.POS_LONG and strategy_signals['exit_long'] or position_side == CONFIG.POS_SHORT and strategy_signals['exit_short']:
                should_exit = True
                exit_reason = strategy_signals['exit_reason']

            if should_exit:
                logger.warning(f"{Back.YELLOW}{Fore.BLACK}*** TRADE EXIT SIGNAL: Closing {position_side} due to '{exit_reason}' ***{Style.RESET_ALL}")
                # Cancel existing SL/TSL orders *before* attempting to close the position
                cancel_open_orders(exchange, symbol, f"Cancel SL/TSL before {exit_reason} Exit")
                time.sleep(0.5)  # Short delay to allow cancellations to process

                close_result = close_position(exchange, symbol, position, reason=exit_reason)
                if close_result:
                    logger.info(f"Pausing for {CONFIG.POST_CLOSE_DELAY_SECONDS}s after closing position...")
                    time.sleep(CONFIG.POST_CLOSE_DELAY_SECONDS)
                # Regardless of close success, exit the logic for this cycle after an exit signal
                return
            else:
                # No strategy exit signal, holding position
                logger.info(f"Holding {position_side} position. No strategy exit signal. Waiting for SL/TSL or next signal.")
                return  # End cycle logic, wait for next candle or SL/TSL hit

        # === Step 7: Check & Execute Entry Actions (Only if Flat) ===
        if position_side == CONFIG.POS_NONE:
            if not can_place_new_order:
                logger.warning(f"{Fore.YELLOW}Holding Cash. Cannot enter: Invalid ATR prevents SL calculation.{Style.RESET_ALL}")
                return  # Cannot enter without valid ATR

            logger.debug("Holding Cash. Checking entry signals...")
            enter_long_signal = strategy_signals['enter_long']
            enter_short_signal = strategy_signals['enter_short']

            # --- Check Confirmation Conditions ---
            passes_volume_confirm = not CONFIG.require_volume_spike_for_entry or vol_spike
            vol_log = f"VolConfirm OK (Pass:{passes_volume_confirm}, Spike={vol_spike}, Req={CONFIG.require_volume_spike_for_entry})"

            # Fetch OB now if not fetched earlier and needed for confirmation
            if (enter_long_signal or enter_short_signal) and ob_data is None:
                 logger.debug("Potential entry signal, fetching OB for confirmation...")
                 ob_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit)
                 bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None  # Update ratio

            ob_available = ob_data is not None and bid_ask_ratio is not None and not bid_ask_ratio.is_nan()
            passes_long_ob_confirm = not ob_available or (bid_ask_ratio >= CONFIG.order_book_ratio_threshold_long)
            passes_short_ob_confirm = not ob_available or (bid_ask_ratio <= CONFIG.order_book_ratio_threshold_short)
            # Note: If OB is unavailable, confirmation passes by default (can be configured otherwise if needed)
            ob_log = f"OBConfirm OK (L:{passes_long_ob_confirm},S:{passes_short_ob_confirm}, Ratio={ob_ratio_str}, Available={ob_available})"

            # --- Combine Strategy Signal with Confirmations ---
            final_enter_long = enter_long_signal and passes_volume_confirm and passes_long_ob_confirm
            final_enter_short = enter_short_signal and passes_volume_confirm and passes_short_ob_confirm
            logger.debug(f"Final Entry Check (Long): Strategy={enter_long_signal}, {vol_log}, {ob_log} => Enter={final_enter_long}")
            logger.debug(f"Final Entry Check (Short): Strategy={enter_short_signal}, {vol_log}, {ob_log} => Enter={final_enter_short}")

            # --- Execute Entry ---
            entry_side: str | None = None
            if final_enter_long:
                entry_side = CONFIG.SIDE_BUY
                logger.success(f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}*** TRADE SIGNAL: CONFIRMED LONG ENTRY ({CONFIG.strategy_name}) for {symbol} ***{Style.RESET_ALL}")
            elif final_enter_short:
                entry_side = CONFIG.SIDE_SELL
                logger.success(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}*** TRADE SIGNAL: CONFIRMED SHORT ENTRY ({CONFIG.strategy_name}) for {symbol} ***{Style.RESET_ALL}")

            if entry_side:
                # Cancel any lingering orders before entering (shouldn't be any if flat, but good practice)
                cancel_open_orders(exchange, symbol, f"Before {entry_side.upper()} Entry")
                time.sleep(0.5)

                place_result = place_risked_market_order(
                    exchange=exchange,
                    symbol=symbol,
                    side=entry_side,
                    risk_percentage=CONFIG.risk_per_trade_percentage,
                    current_atr=current_atr,  # Already validated that it's not None/zero
                    sl_atr_multiplier=CONFIG.atr_stop_loss_multiplier,
                    leverage=CONFIG.leverage,
                    max_order_cap_usdt=CONFIG.max_order_usdt_amount,
                    margin_check_buffer=CONFIG.required_margin_buffer,
                    tsl_percent=CONFIG.trailing_stop_percentage,
                    tsl_activation_offset_percent=CONFIG.trailing_stop_activation_offset_percent
                )
                if place_result:
                    pass
                # End cycle after attempting entry
                return
            else:
                 # No confirmed entry signal
                 logger.info("Holding Cash. No confirmed entry signal this cycle.")

    except Exception as e:
        # Catch-all for unexpected errors within the main logic block
        logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}CRITICAL UNEXPECTED ERROR in trade_logic cycle: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        market_base = symbol.split('/')[0] if '/' in symbol else symbol
        send_sms_alert(f"[{market_base}] CRITICAL ERROR in trade_logic: {type(e).__name__}. Check logs!")
    finally:
        # Log end of cycle regardless of outcome
        logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== Cycle Check End: {symbol} =========={Style.RESET_ALL}\n")


# --- Graceful Shutdown ---
def graceful_shutdown(exchange: ccxt.Exchange | None, symbol: str | None) -> None:
    """Attempts to gracefully shut down the bot by cancelling open orders and closing any active position.

    Args:
        exchange: The CCXT exchange instance (can be None if init failed).
        symbol: The market symbol being traded (can be None if setup failed).
    """
    logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}Shutdown requested. Initiating graceful exit sequence...{Style.RESET_ALL}")
    market_base = symbol.split('/')[0] if symbol and '/' in symbol else "Bot"
    send_sms_alert(f"[{market_base}] Shutdown requested. Attempting cleanup...")

    if not exchange or not symbol:
        logger.warning(f"{Fore.YELLOW}Shutdown: Exchange instance or symbol is not available. Cannot perform cleanup.{Style.RESET_ALL}")
        return

    try:
        # --- Step 1: Cancel All Open Orders FIRST ---
        # This prevents SL/TSL orders interfering with the manual close attempt.
        logger.info(f"{Fore.CYAN}Shutdown: Cancelling all open orders for {symbol}...{Style.RESET_ALL}")
        cancel_open_orders(exchange, symbol, reason="Graceful Shutdown")
        time.sleep(CONFIG.RETRY_DELAY_SECONDS)  # Allow time for cancellations to be processed by the exchange

        # --- Step 2: Check and Close Active Position ---
        logger.info(f"{Fore.CYAN}Shutdown: Checking for active position for {symbol}...{Style.RESET_ALL}")
        position = get_current_position(exchange, symbol)

        if position['side'] != CONFIG.POS_NONE and position['qty'] > CONFIG.POSITION_QTY_EPSILON:
            logger.warning(f"{Fore.YELLOW}Shutdown: Active {position['side']} position found (Qty: {position['qty']:.8f}). Attempting to close...{Style.RESET_ALL}")
            close_result = close_position(exchange, symbol, position, reason="Shutdown")

            if close_result:
                logger.info(f"{Fore.CYAN}Shutdown: Close order placed. Waiting {CONFIG.POST_CLOSE_DELAY_SECONDS * 2}s for confirmation...{Style.RESET_ALL}")
                time.sleep(CONFIG.POST_CLOSE_DELAY_SECONDS * 2)  # Wait a bit longer for final check
                # --- Final Confirmation Check ---
                final_pos = get_current_position(exchange, symbol)
                if final_pos['side'] == CONFIG.POS_NONE or final_pos['qty'] <= CONFIG.POSITION_QTY_EPSILON:
                    logger.success(f"{Fore.GREEN}{Style.BRIGHT}Shutdown: Position successfully confirmed CLOSED.{Style.RESET_ALL}")
                    send_sms_alert(f"[{market_base}] Position confirmed CLOSED on shutdown.")
                else:
                    logger.error(f"{Back.RED}{Fore.WHITE}Shutdown Error: FAILED TO CONFIRM position closure after waiting. Final state: {final_pos['side']} Qty={final_pos['qty']:.8f}{Style.RESET_ALL}")
                    send_sms_alert(f"[{market_base}] ERROR: Failed confirm closure! Final: {final_pos['side']} Qty={final_pos['qty']:.8f}. MANUAL CHECK REQUIRED!")
            else:
                # Close order placement failed
                logger.error(f"{Back.RED}{Fore.WHITE}Shutdown Error: Failed to place close order for active position. MANUAL INTERVENTION REQUIRED.{Style.RESET_ALL}")
                send_sms_alert(f"[{market_base}] ERROR: Failed PLACE close order on shutdown. MANUAL CHECK REQUIRED!")
        else:
            # No active position found
            logger.info(f"{Fore.GREEN}Shutdown: No active position found for {symbol}. No close action needed.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] No active position found on shutdown.")

    except Exception as e:
        logger.error(f"{Fore.RED}Shutdown Error: Unexpected error during cleanup sequence: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}] Error during shutdown cleanup sequence: {type(e).__name__}")

    logger.info(f"{Fore.YELLOW}{Style.BRIGHT}--- Scalping Bot Shutdown Sequence Complete ---{Style.RESET_ALL}")


# --- Main Execution ---
def main() -> None:
    """Main function to initialize the bot, set up the exchange and symbol,
    and run the main trading loop. Handles setup errors and graceful shutdown.
    """
    start_time = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Bybit Scalping Spell v2.0.1 Initializing ({start_time}) ---{Style.RESET_ALL}")
    logger.info(f"{Fore.CYAN}--- Strategy Enchantment: {CONFIG.strategy_name} ---{Style.RESET_ALL}")
    logger.info(f"{Fore.GREEN}--- Warding Rune: Initial ATR({CONFIG.atr_calculation_period}) SL + Exchange TSL ({CONFIG.trailing_stop_percentage:.2%}) ---{Style.RESET_ALL}")
    logger.warning(f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}--- !!! LIVE FUTURES SCALPING MODE - EXTREME RISK INVOLVED !!! ---{Style.RESET_ALL}")

    exchange: ccxt.Exchange | None = None
    symbol: str | None = None
    market_base: str = "Bot"  # Default for early alerts
    run_bot: bool = True
    cycle_count: int = 0

    try:
        # === Initialize Exchange ===
        exchange = initialize_exchange()
        if not exchange:
            logger.critical("Exchange initialization failed. Exiting.")
            return  # Exit if exchange setup fails

        # === Setup Symbol and Leverage ===
        try:
            # Allow user input for symbol or use default from config
            sym_input = input(f"{Fore.YELLOW}Enter symbol {Style.DIM}(Default: [{CONFIG.symbol}]){Style.NORMAL}: {Style.RESET_ALL}").strip()
            symbol_to_use = sym_input or CONFIG.symbol

            # Validate symbol and get unified market info from CCXT
            logger.debug(f"Validating symbol: {symbol_to_use}")
            market = exchange.market(symbol_to_use)
            symbol = market['symbol']  # Use the unified symbol (e.g., BTC/USDT:USDT)
            market_base = symbol.split('/')[0] if '/' in symbol else symbol  # For alerts

            # Ensure it's a contract market suitable for leverage/futures
            if not market.get('contract'):
                raise ValueError(f"Market '{symbol}' is not a contract/futures market.")
            logger.info(f"{Fore.GREEN}Using Symbol: {symbol} (Type: {market.get('type', 'N/A')}, ID: {market.get('id')}){Style.RESET_ALL}")

            # Set leverage
            if not set_leverage(exchange, symbol, CONFIG.leverage):
                # set_leverage logs errors internally
                raise RuntimeError(f"Failed to set leverage to {CONFIG.leverage}x for {symbol}.")

        except (ccxt.BadSymbol, KeyError, ValueError, RuntimeError) as e:
            logger.critical(f"Symbol/Leverage setup failed: {e}")
            send_sms_alert(f"[{market_base}] CRITICAL: Symbol/Leverage setup FAILED ({e}). Exiting.")
            return  # Exit if symbol/leverage setup fails
        except Exception as e:
            logger.critical(f"Unexpected error during symbol/leverage setup: {e}")
            logger.debug(traceback.format_exc())
            send_sms_alert(f"[{market_base}] CRITICAL: Unexpected setup error ({type(e).__name__}). Exiting.")
            return

        # === Log Configuration Summary ===
        logger.info(f"{Fore.MAGENTA}{Style.BRIGHT}{'--- Configuration Summary ---':^50}{Style.RESET_ALL}")
        logger.info(f"{Fore.WHITE}Symbol: {symbol}, Interval: {CONFIG.interval}, Leverage: {CONFIG.leverage}x")
        logger.info(f"{Fore.CYAN}Strategy: {CONFIG.strategy_name}")
        # Log relevant strategy parameters based on selection
        if CONFIG.strategy_name == "DUAL_SUPERTREND":
            logger.info(f"  Params: ST={CONFIG.st_atr_length}/{CONFIG.st_multiplier}, ConfirmST={CONFIG.confirm_st_atr_length}/{CONFIG.confirm_st_multiplier}")
        elif CONFIG.strategy_name == "STOCHRSI_MOMENTUM":
            logger.info(f"  Params: StochRSI={CONFIG.stochrsi_rsi_length}/{CONFIG.stochrsi_stoch_length}/{CONFIG.stochrsi_k_period}/{CONFIG.stochrsi_d_period} (OB={CONFIG.stochrsi_overbought},OS={CONFIG.stochrsi_oversold}), Mom={CONFIG.momentum_length}")
        elif CONFIG.strategy_name == "EHLERS_FISHER":
            logger.info(f"  Params: Fisher={CONFIG.ehlers_fisher_length}, Signal={CONFIG.ehlers_fisher_signal_length}")
        elif CONFIG.strategy_name == "EHLERS_MA_CROSS":
            logger.info(f"  Params: FastMA={CONFIG.ehlers_fast_period}, SlowMA={CONFIG.ehlers_slow_period}")
        logger.info(f"{Fore.GREEN}Risk: {CONFIG.risk_per_trade_percentage:.3%} per trade")
        logger.info(f"{Fore.GREEN}Max Position Value (Cap): {CONFIG.max_order_usdt_amount:.2f} USDT")
        logger.info(f"{Fore.GREEN}Initial SL: {CONFIG.atr_stop_loss_multiplier} * ATR({CONFIG.atr_calculation_period})")
        logger.info(f"{Fore.GREEN}Trailing SL: {CONFIG.trailing_stop_percentage:.2%}, Activation Offset: {CONFIG.trailing_stop_activation_offset_percent:.2%}")
        logger.info(f"{Fore.YELLOW}Volume Confirm: {CONFIG.require_volume_spike_for_entry} (MA={CONFIG.volume_ma_period}, Thr={CONFIG.volume_spike_threshold})")
        logger.info(f"{Fore.YELLOW}OB Confirm (Per Cycle): {CONFIG.fetch_order_book_per_cycle} (Depth={CONFIG.order_book_depth}, L>={CONFIG.order_book_ratio_threshold_long}, S<={CONFIG.order_book_ratio_threshold_short})")
        logger.info(f"{Fore.WHITE}Sleep: {CONFIG.sleep_seconds}s, Margin Buffer: {CONFIG.required_margin_buffer:.1%}, Fill Timeout: {CONFIG.order_fill_timeout_seconds}s")
        logger.info(f"{Fore.MAGENTA}SMS Alerts: {CONFIG.enable_sms_alerts} (To: {'*****' + CONFIG.sms_recipient_number[-4:] if CONFIG.sms_recipient_number else 'N/A'})")
        logger.info(f"{Fore.CYAN}Logging Level: {logging.getLevelName(logger.level)}")
        logger.info(f"{Fore.MAGENTA}{Style.BRIGHT}{'----------------------------':^50}{Style.RESET_ALL}")

        send_sms_alert(f"[{market_base}] Bot configured ({CONFIG.strategy_name}, {symbol}, {CONFIG.interval}, {CONFIG.leverage}x). SL: ATR+TSL. Starting main loop.")

        # === Main Trading Loop ===
        while run_bot:
            cycle_start_time = time.monotonic()
            cycle_count += 1
            logger.debug(f"{Fore.CYAN}--- Cycle {cycle_count} Start ({time.strftime('%H:%M:%S')}) ---{Style.RESET_ALL}")

            try:
                # Determine required data length dynamically based on config
                # Fetch enough data for the longest lookback + buffer
                data_limit = max(150,  # Base minimum
                                 CONFIG.st_atr_length * 2, CONFIG.confirm_st_atr_length * 2,
                                 CONFIG.stochrsi_rsi_length + CONFIG.stochrsi_stoch_length + max(CONFIG.stochrsi_k_period, CONFIG.stochrsi_d_period) + 5,
                                 CONFIG.momentum_length * 2,
                                 CONFIG.ehlers_fisher_length * 2 + CONFIG.ehlers_fisher_signal_length,
                                 CONFIG.ehlers_fast_period * 2, CONFIG.ehlers_slow_period * 2,
                                 CONFIG.atr_calculation_period * 2, CONFIG.volume_ma_period * 2
                                 ) + CONFIG.API_FETCH_LIMIT_BUFFER

                # Fetch market data for the current cycle
                df = get_market_data(exchange, symbol, CONFIG.interval, limit=data_limit)

                if df is not None and not df.empty:
                    # Pass a copy of the DataFrame to trade_logic to avoid modifying the original df used elsewhere
                    trade_logic(exchange, symbol, df.copy())
                else:
                    logger.warning(f"{Fore.YELLOW}No valid market data returned for {symbol}. Skipping trade logic this cycle.{Style.RESET_ALL}")
                    # Consider a longer sleep or different handling if data is consistently unavailable

            # --- Robust Error Handling within the Loop ---
            except ccxt.RateLimitExceeded as e:
                logger.warning(f"{Back.YELLOW}{Fore.BLACK}Rate Limit Exceeded: {e}. Sleeping for {CONFIG.sleep_seconds * 5}s...{Style.RESET_ALL}")
                send_sms_alert(f"[{market_base}] WARNING: Rate limit hit! Sleeping longer.")
                time.sleep(CONFIG.sleep_seconds * 5)
            except ccxt.NetworkError as e:
                # Includes connection errors, timeouts, etc. Usually recoverable.
                logger.warning(f"{Fore.YELLOW}Network error: {e}. Check connection. Retrying next cycle after sleep.{Style.RESET_ALL}")
                # Standard sleep is usually sufficient here
                time.sleep(CONFIG.sleep_seconds)
            except ccxt.ExchangeNotAvailable as e:
                # Exchange maintenance or severe issues. Sleep longer.
                logger.error(f"{Back.RED}{Fore.WHITE}Exchange Not Available: {e}. Sleeping for {CONFIG.sleep_seconds * 10}s...{Style.RESET_ALL}")
                send_sms_alert(f"[{market_base}] ERROR: Exchange unavailable! Sleeping much longer.")
                time.sleep(CONFIG.sleep_seconds * 10)
            except ccxt.AuthenticationError as e:
                # API keys likely invalid or expired. Critical - Stop the bot.
                logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Authentication Error: {e}. API keys may be invalid/revoked. Stopping bot NOW.{Style.RESET_ALL}")
                send_sms_alert(f"[{market_base}] CRITICAL: Authentication Error! Bot stopped.")
                run_bot = False  # Terminate the loop
            except ccxt.ExchangeError as e:
                # Catch other specific exchange errors not handled above.
                logger.error(f"{Fore.RED}Unhandled Exchange Error: {e}{Style.RESET_ALL}")
                logger.debug(traceback.format_exc())
                send_sms_alert(f"[{market_base}] ERROR: Unhandled Exchange error: {type(e).__name__}. Check logs.")
                # Standard sleep before retrying
                time.sleep(CONFIG.sleep_seconds)
            except Exception as e:
                # Catch any other unexpected error. Critical - Stop the bot.
                logger.exception(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}!!! UNEXPECTED CRITICAL ERROR in main loop: {e} !!!{Style.RESET_ALL}")
                send_sms_alert(f"[{market_base}] CRITICAL UNEXPECTED ERROR: {type(e).__name__}! Stopping bot NOW.")
                run_bot = False  # Terminate the loop

            # --- Loop Delay ---
            if run_bot:
                elapsed = time.monotonic() - cycle_start_time
                sleep_duration = max(0, CONFIG.sleep_seconds - elapsed)
                logger.debug(f"Cycle {cycle_count} completed in {elapsed:.2f}s. Sleeping for {sleep_duration:.2f}s.")
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

    except KeyboardInterrupt:
        logger.warning(f"\n{Fore.YELLOW}{Style.BRIGHT}KeyboardInterrupt received. Stopping bot...{Style.RESET_ALL}")
        run_bot = False  # Ensure loop terminates if Ctrl+C is pressed during setup/sleep
    finally:
        # --- Graceful Shutdown ---
        # This block executes whether the loop finished normally,
        # was interrupted, or exited due to a critical error.
        graceful_shutdown(exchange, symbol)
        final_alert_market = market_base if market_base != "Bot" else (symbol.split('/')[0] if symbol and '/' in symbol else "Bot")
        send_sms_alert(f"[{final_alert_market}] Bot process terminated.")
        logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Scalping Spell Deactivated ---{Style.RESET_ALL}")


if __name__ == "__main__":
    main()

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v2.0.1 (Syntax Fix)
# Conjures high-frequency trades on Bybit Futures with enhanced precision and adaptable strategies.

"""
High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 2.0.1 (Unified: Selectable Strategies + Precision + Native SL/TSL + Syntax Fix)

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
import logging
import os
import sys
import traceback
from decimal import Decimal, getcontext
from typing import Any

# Third-party Libraries
try:
    import ccxt
    import pandas as pd
    import pandas_ta as ta  # type: ignore[import]
    from colorama import Back, Fore, Style
    from colorama import init as colorama_init
    from dotenv import load_dotenv
except ImportError as e:
    missing_pkg = e.name
    sys.exit(1)

# --- Initializations ---
colorama_init(autoreset=True)
load_dotenv()
getcontext().prec = 18  # Set Decimal precision (adjust as needed)


# --- Configuration Class ---
class Config:
    """Loads and validates configuration parameters from environment variables."""
    def __init__(self) -> None:
        logger.info(f"{Fore.MAGENTA}--- Summoning Configuration Runes ---{Style.RESET_ALL}")
        # --- API Credentials ---
        self.api_key: str | None = self._get_env("BYBIT_API_KEY", required=True, color=Fore.RED)
        self.api_secret: str | None = self._get_env("BYBIT_API_SECRET", required=True, color=Fore.RED)

        # --- Trading Parameters ---
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT", color=Fore.YELLOW)
        self.interval: str = self._get_env("INTERVAL", "1m", color=Fore.YELLOW)
        self.leverage: int = self._get_env("LEVERAGE", 10, cast_type=int, color=Fore.YELLOW)
        self.sleep_seconds: int = self._get_env("SLEEP_SECONDS", 10, cast_type=int, color=Fore.YELLOW)

        # --- Strategy Selection ---
        self.strategy_name: str = self._get_env("STRATEGY_NAME", "DUAL_SUPERTREND", color=Fore.CYAN).upper()
        self.valid_strategies: list[str] = ["DUAL_SUPERTREND", "STOCHRSI_MOMENTUM", "EHLERS_FISHER", "EHLERS_MA_CROSS"]
        if self.strategy_name not in self.valid_strategies:
            raise ValueError(f"Invalid STRATEGY_NAME '{self.strategy_name}'. Valid: {self.valid_strategies}")

        # --- Risk Management ---
        self.risk_per_trade_percentage: Decimal = self._get_env("RISK_PER_TRADE_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN)  # 0.5%
        self.atr_stop_loss_multiplier: Decimal = self._get_env("ATR_STOP_LOSS_MULTIPLIER", "1.5", cast_type=Decimal, color=Fore.GREEN)
        self.max_order_usdt_amount: Decimal = self._get_env("MAX_ORDER_USDT_AMOUNT", "500.0", cast_type=Decimal, color=Fore.GREEN)
        self.required_margin_buffer: Decimal = self._get_env("REQUIRED_MARGIN_BUFFER", "1.05", cast_type=Decimal, color=Fore.GREEN)  # 5% buffer

        # --- Trailing Stop Loss (Exchange Native) ---
        self.trailing_stop_percentage: Decimal = self._get_env("TRAILING_STOP_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN)  # 0.5% trail
        self.trailing_stop_activation_offset_percent: Decimal = self._get_env("TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT", "0.001", cast_type=Decimal, color=Fore.GREEN)  # 0.1% offset

        # --- Dual Supertrend Parameters ---
        self.st_atr_length: int = self._get_env("ST_ATR_LENGTH", 7, cast_type=int, color=Fore.CYAN)
        self.st_multiplier: Decimal = self._get_env("ST_MULTIPLIER", "2.5", cast_type=Decimal, color=Fore.CYAN)
        self.confirm_st_atr_length: int = self._get_env("CONFIRM_ST_ATR_LENGTH", 5, cast_type=int, color=Fore.CYAN)
        self.confirm_st_multiplier: Decimal = self._get_env("CONFIRM_ST_MULTIPLIER", "2.0", cast_type=Decimal, color=Fore.CYAN)

        # --- StochRSI + Momentum Parameters ---
        self.stochrsi_rsi_length: int = self._get_env("STOCHRSI_RSI_LENGTH", 14, cast_type=int, color=Fore.CYAN)
        self.stochrsi_stoch_length: int = self._get_env("STOCHRSI_STOCH_LENGTH", 14, cast_type=int, color=Fore.CYAN)
        self.stochrsi_k_period: int = self._get_env("STOCHRSI_K_PERIOD", 3, cast_type=int, color=Fore.CYAN)
        self.stochrsi_d_period: int = self._get_env("STOCHRSI_D_PERIOD", 3, cast_type=int, color=Fore.CYAN)
        self.stochrsi_overbought: Decimal = self._get_env("STOCHRSI_OVERBOUGHT", "80.0", cast_type=Decimal, color=Fore.CYAN)
        self.stochrsi_oversold: Decimal = self._get_env("STOCHRSI_OVERSOLD", "20.0", cast_type=Decimal, color=Fore.CYAN)
        self.momentum_length: int = self._get_env("MOMENTUM_LENGTH", 5, cast_type=int, color=Fore.CYAN)

        # --- Ehlers Fisher Transform Parameters ---
        self.ehlers_fisher_length: int = self._get_env("EHLERS_FISHER_LENGTH", 10, cast_type=int, color=Fore.CYAN)
        self.ehlers_fisher_signal_length: int = self._get_env("EHLERS_FISHER_SIGNAL_LENGTH", 1, cast_type=int, color=Fore.CYAN)  # Default to 1

        # --- Ehlers MA Cross Parameters ---
        self.ehlers_fast_period: int = self._get_env("EHLERS_FAST_PERIOD", 10, cast_type=int, color=Fore.CYAN)
        self.ehlers_slow_period: int = self._get_env("EHLERS_SLOW_PERIOD", 30, cast_type=int, color=Fore.CYAN)

        # --- Volume Analysis ---
        self.volume_ma_period: int = self._get_env("VOLUME_MA_PERIOD", 20, cast_type=int, color=Fore.YELLOW)
        self.volume_spike_threshold: Decimal = self._get_env("VOLUME_SPIKE_THRESHOLD", "1.5", cast_type=Decimal, color=Fore.YELLOW)
        self.require_volume_spike_for_entry: bool = self._get_env("REQUIRE_VOLUME_SPIKE_FOR_ENTRY", "false", cast_type=bool, color=Fore.YELLOW)

        # --- Order Book Analysis ---
        self.order_book_depth: int = self._get_env("ORDER_BOOK_DEPTH", 10, cast_type=int, color=Fore.YELLOW)
        self.order_book_ratio_threshold_long: Decimal = self._get_env("ORDER_BOOK_RATIO_THRESHOLD_LONG", "1.2", cast_type=Decimal, color=Fore.YELLOW)
        self.order_book_ratio_threshold_short: Decimal = self._get_env("ORDER_BOOK_RATIO_THRESHOLD_SHORT", "0.8", cast_type=Decimal, color=Fore.YELLOW)
        self.fetch_order_book_per_cycle: bool = self._get_env("FETCH_ORDER_BOOK_PER_CYCLE", "false", cast_type=bool, color=Fore.YELLOW)

        # --- ATR Calculation (for Initial SL) ---
        self.atr_calculation_period: int = self._get_env("ATR_CALCULATION_PERIOD", 14, cast_type=int, color=Fore.GREEN)

        # --- Termux SMS Alerts ---
        self.enable_sms_alerts: bool = self._get_env("ENABLE_SMS_ALERTS", "false", cast_type=bool, color=Fore.MAGENTA)
        self.sms_recipient_number: str | None = self._get_env("SMS_RECIPIENT_NUMBER", None, color=Fore.MAGENTA)
        self.sms_timeout_seconds: int = self._get_env("SMS_TIMEOUT_SECONDS", 30, cast_type=int, color=Fore.MAGENTA)

        # --- CCXT / API Parameters ---
        self.default_recv_window: int = 10000
        self.order_book_fetch_limit: int = max(25, self.order_book_depth)
        self.shallow_ob_fetch_depth: int = 5
        self.order_fill_timeout_seconds: int = self._get_env("ORDER_FILL_TIMEOUT_SECONDS", 15, cast_type=int, color=Fore.YELLOW)

        # --- Internal Constants ---
        self.side_buy: str = "buy"
        self.side_sell: str = "sell"
        self.pos_long: str = "Long"
        self.pos_short: str = "Short"
        self.pos_none: str = "None"
        self.usdt_symbol: str = "USDT"
        self.retry_count: int = 3
        self.retry_delay_seconds: int = 2
        self.api_fetch_limit_buffer: int = 10
        self.position_qty_epsilon: Decimal = Decimal("1e-9")
        self.post_close_delay_seconds: int = 3

        logger.info(f"{Fore.MAGENTA}--- Configuration Runes Summoned ---{Style.RESET_ALL}")

    def _get_env(self, key: str, default: Any = None, cast_type: type = str, required: bool = False, color: str = Fore.WHITE) -> Any:
        """Fetches env var, casts type, logs, handles defaults/errors."""
        value = os.getenv(key)
        log_value = f"'{value}'" if value is not None else f"Not Set (Using Default: '{default}')"
        logger.debug(f"{color}Summoning {key}: {log_value}{Style.RESET_ALL}")

        if value is None:
            if required:
                raise ValueError(f"CRITICAL: Required environment variable '{key}' not set.")
            value = default
        elif cast_type == bool:
            value = value.lower() in ['true', '1', 'yes', 'y']
        elif cast_type == Decimal:
            try:
                value = Decimal(value)
            except InvalidOperation:
                logger.error(f"{Fore.RED}Invalid Decimal value for {key}: '{value}'. Using default: '{default}'{Style.RESET_ALL}")
                value = Decimal(str(default)) if default is not None else None
        elif cast_type is not None:
            try:
                value = cast_type(value)
            except (ValueError, TypeError):
                logger.error(f"{Fore.RED}Invalid type for {key}: '{value}'. Expected {cast_type.__name__}. Using default: '{default}'{Style.RESET_ALL}")
                value = default

        if value is None and required:  # Check again if default was None
             raise ValueError(f"CRITICAL: Required environment variable '{key}' has no value or default.")

        return value


# --- Logger Setup ---
LOGGING_LEVEL: int = logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger: logging.Logger = logging.getLogger(__name__)

# Custom SUCCESS level and Neon Color Formatting
SUCCESS_LEVEL: int = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(SUCCESS_LEVEL): self._log(SUCCESS_LEVEL, message, args, **kwargs)  # pylint: disable=protected-access


logging.Logger.success = log_success

if sys.stdout.isatty():
    logging.addLevelName(logging.DEBUG, f"{Fore.CYAN}{logging.getLevelName(logging.DEBUG)}{Style.RESET_ALL}")
    logging.addLevelName(logging.INFO, f"{Fore.BLUE}{logging.getLevelName(logging.INFO)}{Style.RESET_ALL}")
    logging.addLevelName(SUCCESS_LEVEL, f"{Fore.MAGENTA}{logging.getLevelName(SUCCESS_LEVEL)}{Style.RESET_ALL}")
    logging.addLevelName(logging.WARNING, f"{Fore.YELLOW}{logging.getLevelName(logging.WARNING)}{Style.RESET_ALL}")
    logging.addLevelName(logging.ERROR, f"{Fore.RED}{logging.getLevelName(logging.ERROR)}{Style.RESET_ALL}")
    logging.addLevelName(logging.CRITICAL, f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}{logging.getLevelName(logging.CRITICAL)}{Style.RESET_ALL}")

# --- Global Objects ---
try:
    CONFIG = Config()
except ValueError as e:
    logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Configuration Error: {e}{Style.RESET_ALL}")
    sys.exit(1)


# --- Helper Functions ---
def safe_decimal_conversion(value: Any, default: Decimal = Decimal("0.0")) -> Decimal:
    """Safely converts a value to Decimal, returning default if conversion fails."""
    try:
        return Decimal(str(value)) if value is not None else default
    except (InvalidOperation, TypeError, ValueError):
        logger.warning(f"Could not convert '{value}' to Decimal, using default {default}")
        return default


def format_order_id(order_id: str | int | None) -> str:
    """Returns the last 6 characters of an order ID or 'N/A'."""
    return str(order_id)[-6:] if order_id else "N/A"


# --- Precision Formatting ---
def format_price(exchange: ccxt.Exchange, symbol: str, price: float | Decimal) -> str:
    """Formats price according to market precision rules."""
    try:
        # CCXT formatting methods often expect float input
        return exchange.price_to_precision(symbol, float(price))
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting price {price} for {symbol}: {e}{Style.RESET_ALL}")
        return str(Decimal(str(price)).normalize())  # Fallback to Decimal string


def format_amount(exchange: ccxt.Exchange, symbol: str, amount: float | Decimal) -> str:
    """Formats amount according to market precision rules."""
    try:
        # CCXT formatting methods often expect float input
        return exchange.amount_to_precision(symbol, float(amount))
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting amount {amount} for {symbol}: {e}{Style.RESET_ALL}")
        return str(Decimal(str(amount)).normalize())  # Fallback to Decimal string


# --- Termux SMS Alert Function ---
def send_sms_alert(message: str) -> bool:
    """Sends an SMS alert using Termux API."""
    if not CONFIG.enable_sms_alerts:
        logger.debug("SMS alerts disabled.")
        return False
    if not CONFIG.sms_recipient_number:
        logger.warning("SMS alerts enabled, but SMS_RECIPIENT_NUMBER not set.")
        return False
    try:
        # Use shlex.quote for message safety, though direct passing is usually fine
        # quoted_message = shlex.quote(message)
        command: list[str] = ['termux-sms-send', '-n', CONFIG.sms_recipient_number, message]
        logger.info(f"{Fore.MAGENTA}Attempting SMS to {CONFIG.sms_recipient_number} (Timeout: {CONFIG.sms_timeout_seconds}s)...{Style.RESET_ALL}")
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=CONFIG.sms_timeout_seconds)
        if result.returncode == 0:
            logger.success(f"{Fore.MAGENTA}SMS command executed successfully.{Style.RESET_ALL}")
            return True
        else:
            logger.error(f"{Fore.RED}SMS command failed. RC: {result.returncode}, Stderr: {result.stderr.strip()}{Style.RESET_ALL}")
            return False
    except FileNotFoundError:
        logger.error(f"{Fore.RED}SMS failed: 'termux-sms-send' not found. Install Termux:API.{Style.RESET_ALL}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"{Fore.RED}SMS failed: command timed out after {CONFIG.sms_timeout_seconds}s.{Style.RESET_ALL}")
        return False
    except Exception as e:
        logger.error(f"{Fore.RED}SMS failed: Unexpected error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        return False


# --- Exchange Initialization ---
def initialize_exchange() -> ccxt.Exchange | None:
    """Initializes and returns the CCXT Bybit exchange instance."""
    logger.info(f"{Fore.BLUE}Initializing CCXT Bybit connection...{Style.RESET_ALL}")
    if not CONFIG.api_key or not CONFIG.api_secret:
        logger.critical("API keys missing in .env file.")
        send_sms_alert("[ScalpBot] CRITICAL: API keys missing. Bot stopped.")
        return None
    try:
        exchange = ccxt.bybit({
            "apiKey": CONFIG.api_key,
            "secret": CONFIG.api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "linear",  # Assuming USDT perpetuals
                "recvWindow": CONFIG.default_recv_window,
                "adjustForTimeDifference": True,
            },
        })
        logger.debug("Loading markets...")
        exchange.load_markets(True)  # Force reload
        logger.debug("Fetching initial balance...")
        exchange.fetch_balance()  # Initial check
        logger.success(f"{Fore.GREEN}{Style.BRIGHT}CCXT Bybit Session Initialized (LIVE SCALPING MODE - EXTREME CAUTION!).{Style.RESET_ALL}")
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


# --- Indicator Calculation Functions ---
def calculate_supertrend(df: pd.DataFrame, length: int, multiplier: Decimal, prefix: str = "") -> pd.DataFrame:
    """Calculates the Supertrend indicator using pandas_ta, returns Decimal where applicable."""
    col_prefix = f"{prefix}" if prefix else ""
    target_cols = [f"{col_prefix}supertrend", f"{col_prefix}trend", f"{col_prefix}st_long", f"{col_prefix}st_short"]
    st_col = f"SUPERT_{length}_{float(multiplier)}"  # pandas_ta uses float in name
    st_trend_col = f"SUPERTd_{length}_{float(multiplier)}"
    st_long_col = f"SUPERTl_{length}_{float(multiplier)}"
    st_short_col = f"SUPERTs_{length}_{float(multiplier)}"
    required_input_cols = ["high", "low", "close"]

    if df is None or df.empty or not all(c in df.columns for c in required_input_cols) or len(df) < length + 1:
        logger.warning(f"{Fore.YELLOW}Indicator Calc ({col_prefix}ST): Invalid input (Len: {len(df) if df is not None else 0}, Need: {length + 1}).{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA
        return df

    try:
        # pandas_ta expects float multiplier
        df.ta.supertrend(length=length, multiplier=float(multiplier), append=True)
        if st_col not in df.columns or st_trend_col not in df.columns:
            raise KeyError(f"pandas_ta failed to create expected raw columns: {st_col}, {st_trend_col}")

        # Convert Supertrend value to Decimal
        df[f"{col_prefix}supertrend"] = df[st_col].apply(safe_decimal_conversion)
        df[f"{col_prefix}trend"] = df[st_trend_col] == 1  # Boolean
        prev_trend = df[st_trend_col].shift(1)
        df[f"{col_prefix}st_long"] = (prev_trend == -1) & (df[st_trend_col] == 1)  # Boolean
        df[f"{col_prefix}st_short"] = (prev_trend == 1) & (df[st_trend_col] == -1)  # Boolean

        raw_st_cols = [st_col, st_trend_col, st_long_col, st_short_col]
        df.drop(columns=raw_st_cols, errors='ignore', inplace=True)

        last_st_val = df[f'{col_prefix}supertrend'].iloc[-1]
        if pd.notna(last_st_val):
            last_trend = 'Up' if df[f'{col_prefix}trend'].iloc[-1] else 'Down'
            signal = 'LONG' if df[f'{col_prefix}st_long'].iloc[-1] else ('SHORT' if df[f'{col_prefix}st_short'].iloc[-1] else 'None')
            logger.debug(f"Indicator Calc ({col_prefix}ST({length},{multiplier})): Trend={last_trend}, Val={last_st_val:.4f}, Signal={signal}")
        else:
            logger.debug(f"Indicator Calc ({col_prefix}ST({length},{multiplier})): Resulted in NA for last candle.")

    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc ({col_prefix}ST): Error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA
    return df


def analyze_volume_atr(df: pd.DataFrame, atr_len: int, vol_ma_len: int) -> dict[str, Decimal | None]:
    """Calculates ATR, Volume MA, checks spikes. Returns Decimals."""
    results: dict[str, Decimal | None] = {"atr": None, "volume_ma": None, "last_volume": None, "volume_ratio": None}
    required_cols = ["high", "low", "close", "volume"]
    min_len = max(atr_len, vol_ma_len)

    if df is None or df.empty or not all(c in df.columns for c in required_cols) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Indicator Calc (Vol/ATR): Invalid input (Len: {len(df) if df is not None else 0}, Need: {min_len}).{Style.RESET_ALL}")
        return results

    try:
        # Calculate ATR
        atr_col = f"ATRr_{atr_len}"
        df.ta.atr(length=atr_len, append=True)
        if atr_col in df.columns:
            last_atr = df[atr_col].iloc[-1]
            if pd.notna(last_atr): results["atr"] = safe_decimal_conversion(last_atr)
            df.drop(columns=[atr_col], errors='ignore', inplace=True)

        # Calculate Volume MA
        volume_ma_col = 'volume_ma'
        df[volume_ma_col] = df['volume'].rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2)).mean()
        last_vol_ma = df[volume_ma_col].iloc[-1]
        last_vol = df['volume'].iloc[-1]

        if pd.notna(last_vol_ma): results["volume_ma"] = safe_decimal_conversion(last_vol_ma)
        if pd.notna(last_vol): results["last_volume"] = safe_decimal_conversion(last_vol)

        if results["volume_ma"] is not None and results["volume_ma"] > CONFIG.position_qty_epsilon and results["last_volume"] is not None:
            try:
                results["volume_ratio"] = results["last_volume"] / results["volume_ma"]
            except Exception:  # Handles potential division by zero if MA is epsilon
                 results["volume_ratio"] = None

        if volume_ma_col in df.columns: df.drop(columns=[volume_ma_col], errors='ignore', inplace=True)

        # Log results
        atr_str = f"{results['atr']:.5f}" if results['atr'] else 'N/A'
        vol_ma_str = f"{results['volume_ma']:.2f}" if results['volume_ma'] else 'N/A'
        vol_ratio_str = f"{results['volume_ratio']:.2f}" if results['volume_ratio'] else 'N/A'
        logger.debug(f"Indicator Calc: ATR({atr_len})={atr_str}, VolMA({vol_ma_len})={vol_ma_str}, VolRatio={vol_ratio_str}")

    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc (Vol/ATR): Error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        results = dict.fromkeys(results)
    return results


def calculate_stochrsi_momentum(df: pd.DataFrame, rsi_len: int, stoch_len: int, k: int, d: int, mom_len: int) -> pd.DataFrame:
    """Calculates StochRSI and Momentum, returns Decimals."""
    target_cols = ['stochrsi_k', 'stochrsi_d', 'momentum']
    min_len = max(rsi_len + stoch_len, mom_len) + 5  # Add buffer
    if df is None or df.empty or not all(c in df.columns for c in ["close"]) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Indicator Calc (StochRSI/Mom): Invalid input (Len: {len(df) if df is not None else 0}, Need ~{min_len}).{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA
        return df
    try:
        # StochRSI
        stochrsi_df = df.ta.stochrsi(length=stoch_len, rsi_length=rsi_len, k=k, d=d, append=False)
        k_col, d_col = f"STOCHRSIk_{stoch_len}_{rsi_len}_{k}_{d}", f"STOCHRSId_{stoch_len}_{rsi_len}_{k}_{d}"
        if k_col in stochrsi_df.columns: df['stochrsi_k'] = stochrsi_df[k_col].apply(safe_decimal_conversion)
        else: logger.warning("StochRSI K column not found"); df['stochrsi_k'] = pd.NA
        if d_col in stochrsi_df.columns: df['stochrsi_d'] = stochrsi_df[d_col].apply(safe_decimal_conversion)
        else: logger.warning("StochRSI D column not found"); df['stochrsi_d'] = pd.NA

        # Momentum
        mom_col = f"MOM_{mom_len}"
        df.ta.mom(length=mom_len, append=True)
        if mom_col in df.columns:
            df['momentum'] = df[mom_col].apply(safe_decimal_conversion)
            df.drop(columns=[mom_col], errors='ignore', inplace=True)
        else: logger.warning("Momentum column not found"); df['momentum'] = pd.NA

        k_val, d_val, mom_val = df['stochrsi_k'].iloc[-1], df['stochrsi_d'].iloc[-1], df['momentum'].iloc[-1]
        if pd.notna(k_val) and pd.notna(d_val) and pd.notna(mom_val):
            logger.debug(f"Indicator Calc (StochRSI/Mom): K={k_val:.2f}, D={d_val:.2f}, Mom={mom_val:.4f}")
        else:
            logger.debug("Indicator Calc (StochRSI/Mom): Resulted in NA for last candle.")

    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc (StochRSI/Mom): Error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA
    return df


def calculate_ehlers_fisher(df: pd.DataFrame, length: int, signal: int) -> pd.DataFrame:
    """Calculates Ehlers Fisher Transform, returns Decimals."""
    target_cols = ['ehlers_fisher', 'ehlers_signal']
    if df is None or df.empty or not all(c in df.columns for c in ["high", "low"]) or len(df) < length + 1:
        logger.warning(f"{Fore.YELLOW}Indicator Calc (EhlersFisher): Invalid input (Len: {len(df) if df is not None else 0}, Need {length + 1}).{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA
        return df
    try:
        fisher_df = df.ta.fisher(length=length, signal=signal, append=False)
        fish_col, signal_col = f"FISHERT_{length}_{signal}", f"FISHERTs_{length}_{signal}"
        if fish_col in fisher_df.columns: df['ehlers_fisher'] = fisher_df[fish_col].apply(safe_decimal_conversion)
        else: logger.warning("Ehlers Fisher column not found"); df['ehlers_fisher'] = pd.NA
        if signal_col in fisher_df.columns: df['ehlers_signal'] = fisher_df[signal_col].apply(safe_decimal_conversion)
        else: logger.warning("Ehlers Signal column not found"); df['ehlers_signal'] = pd.NA

        fish_val, sig_val = df['ehlers_fisher'].iloc[-1], df['ehlers_signal'].iloc[-1]
        if pd.notna(fish_val) and pd.notna(sig_val):
             logger.debug(f"Indicator Calc (EhlersFisher({length},{signal})): Fisher={fish_val:.4f}, Signal={sig_val:.4f}")
        else:
             logger.debug("Indicator Calc (EhlersFisher): Resulted in NA for last candle.")
    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc (EhlersFisher): Error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA
    return df


def calculate_ehlers_ma(df: pd.DataFrame, fast_len: int, slow_len: int) -> pd.DataFrame:
    """Calculates Ehlers Super Smoother Moving Averages, returns Decimals."""
    target_cols = ['fast_ema', 'slow_ema']
    min_len = max(fast_len, slow_len) + 5  # Add buffer
    if df is None or df.empty or not all(c in df.columns for c in ["close"]) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Indicator Calc (EhlersMA): Invalid input (Len: {len(df) if df is not None else 0}, Need ~{min_len}).{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA
        return df
    try:
        # pandas_ta.supersmoother might not exist, use custom or alternative like Ehlers Filter if needed
        # Assuming ta.ema as a placeholder if supersmoother is unavailable or buggy
        # Replace with actual Ehlers filter implementation if required
        logger.warning(f"{Fore.YELLOW}Using EMA as placeholder for Ehlers Super Smoother. Replace with actual implementation if needed.{Style.RESET_ALL}")
        df['fast_ema'] = df.ta.ema(length=fast_len).apply(safe_decimal_conversion)
        df['slow_ema'] = df.ta.ema(length=slow_len).apply(safe_decimal_conversion)

        fast_val, slow_val = df['fast_ema'].iloc[-1], df['slow_ema'].iloc[-1]
        if pd.notna(fast_val) and pd.notna(slow_val):
            logger.debug(f"Indicator Calc (EhlersMA({fast_len},{slow_len})): Fast={fast_val:.4f}, Slow={slow_val:.4f}")
        else:
             logger.debug("Indicator Calc (EhlersMA): Resulted in NA for last candle.")
    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc (EhlersMA): Error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA
    return df


def analyze_order_book(exchange: ccxt.Exchange, symbol: str, depth: int, fetch_limit: int) -> dict[str, Decimal | None]:
    """Fetches and analyzes L2 order book pressure and spread. Returns Decimals."""
    results: dict[str, Decimal | None] = {"bid_ask_ratio": None, "spread": None, "best_bid": None, "best_ask": None}
    logger.debug(f"Order Book: Fetching L2 {symbol} (Depth:{depth}, Limit:{fetch_limit})...")
    if not exchange.has.get('fetchL2OrderBook'):
        logger.warning(f"{Fore.YELLOW}fetchL2OrderBook not supported by {exchange.id}.{Style.RESET_ALL}")
        return results
    try:
        order_book = exchange.fetch_l2_order_book(symbol, limit=fetch_limit)
        bids: list[list[float | str]] = order_book.get('bids', [])
        asks: list[list[float | str]] = order_book.get('asks', [])

        best_bid = safe_decimal_conversion(bids[0][0]) if bids and len(bids[0]) > 0 else None
        best_ask = safe_decimal_conversion(asks[0][0]) if asks and len(asks[0]) > 0 else None
        results["best_bid"] = best_bid
        results["best_ask"] = best_ask

        if best_bid is not None and best_ask is not None and best_bid > 0 and best_ask > 0:
            results["spread"] = best_ask - best_bid
            logger.debug(f"OB: Bid={best_bid:.4f}, Ask={best_ask:.4f}, Spread={results['spread']:.4f}")
        else:
            logger.debug(f"OB: Bid={best_bid or 'N/A'}, Ask={best_ask or 'N/A'} (Spread N/A)")

        # Sum volumes using Decimal
        bid_vol = sum(safe_decimal_conversion(bid[1]) for bid in bids[:depth] if len(bid) > 1)
        ask_vol = sum(safe_decimal_conversion(ask[1]) for ask in asks[:depth] if len(ask) > 1)
        logger.debug(f"OB (Depth {depth}): BidVol={bid_vol:.4f}, AskVol={ask_vol:.4f}")

        if ask_vol > CONFIG.position_qty_epsilon:
            try:
                results["bid_ask_ratio"] = bid_vol / ask_vol
                logger.debug(f"OB Ratio: {results['bid_ask_ratio']:.3f}")
            except Exception:
                logger.warning("Error calculating OB ratio.")
                results["bid_ask_ratio"] = None
        else:
            logger.debug("OB Ratio: N/A (Ask volume zero or negligible)")

    except (ccxt.NetworkError, ccxt.ExchangeError, IndexError, Exception) as e:
        logger.warning(f"{Fore.YELLOW}Order Book Error for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        results = dict.fromkeys(results)  # Reset on error
    return results


# --- Data Fetching ---
def get_market_data(exchange: ccxt.Exchange, symbol: str, interval: str, limit: int) -> pd.DataFrame | None:
    """Fetches and prepares OHLCV data, ensuring numeric types."""
    if not exchange.has.get("fetchOHLCV"):
        logger.error(f"{Fore.RED}Data Fetch: Exchange '{exchange.id}' does not support fetchOHLCV.{Style.RESET_ALL}")
        return None
    try:
        logger.debug(f"Data Fetch: Fetching {limit} OHLCV candles for {symbol} ({interval})...")
        ohlcv: list[list[int | float | str]] = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
        if not ohlcv:
            logger.warning(f"{Fore.YELLOW}Data Fetch: No OHLCV data returned for {symbol} ({interval}).{Style.RESET_ALL}")
            return None

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)

        # Convert to numeric, coercing errors, check NaNs robustly
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        if df.isnull().values.any():
            nan_counts = df.isnull().sum()
            logger.warning(f"{Fore.YELLOW}Data Fetch: OHLCV contains NaNs after conversion:\n{nan_counts[nan_counts > 0]}\nAttempting ffill...{Style.RESET_ALL}")
            df.ffill(inplace=True)  # Forward fill first
            if df.isnull().values.any():  # Check again, maybe backfill needed?
                logger.warning(f"{Fore.YELLOW}NaNs remain after ffill, attempting bfill...{Style.RESET_ALL}")
                df.bfill(inplace=True)
                if df.isnull().values.any():
                    logger.error(f"{Fore.RED}Data Fetch: NaNs persist after ffill/bfill. Cannot proceed.{Style.RESET_ALL}")
                    return None

        logger.debug(f"Data Fetch: Processed {len(df)} OHLCV candles for {symbol}.")
        return df

    except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
        logger.warning(f"{Fore.YELLOW}Data Fetch: Error fetching OHLCV for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
    return None


# --- Position & Order Management ---
def get_current_position(exchange: ccxt.Exchange, symbol: str) -> dict[str, Any]:
    """Fetches current position details (Bybit V5 focus), returns Decimals."""
    default_pos: dict[str, Any] = {'side': CONFIG.pos_none, 'qty': Decimal("0.0"), 'entry_price': Decimal("0.0")}
    market_id = None
    market = None
    try:
        market = exchange.market(symbol)
        market_id = market['id']
    except Exception as e:
        logger.error(f"{Fore.RED}Position Check: Failed to get market info for '{symbol}': {e}{Style.RESET_ALL}")
        return default_pos

    try:
        if not exchange.has.get('fetchPositions'):
            logger.warning(f"{Fore.YELLOW}Position Check: fetchPositions not supported by {exchange.id}.{Style.RESET_ALL}")
            return default_pos

        # Bybit V5 uses 'category' parameter
        params = {'category': 'linear'} if market.get('linear') else ({'category': 'inverse'} if market.get('inverse') else {})
        logger.debug(f"Position Check: Fetching positions for {symbol} (MarketID: {market_id}) with params: {params}")

        fetched_positions = exchange.fetch_positions(symbols=[symbol], params=params)

        # Bybit V5 might return multiple entries even for one-way mode sometimes, find the active one
        active_pos = None
        for pos in fetched_positions:
            pos_info = pos.get('info', {})
            pos_market_id = pos_info.get('symbol')
            position_idx = pos_info.get('positionIdx', 0)  # 0 for One-Way mode
            pos_side_v5 = pos_info.get('side', 'None')  # 'Buy' for long, 'Sell' for short
            size_str = pos_info.get('size')

            # Filter for the correct symbol and One-Way mode active position
            if pos_market_id == market_id and position_idx == 0 and pos_side_v5 != 'None':
                size = safe_decimal_conversion(size_str)
                if abs(size) > CONFIG.position_qty_epsilon:
                    active_pos = pos  # Found the active position
                    break  # Assume only one active position in One-Way mode

        if active_pos:
            try:
                size = safe_decimal_conversion(active_pos.get('info', {}).get('size'))
                # Use 'avgPrice' from info for V5 entry price
                entry_price = safe_decimal_conversion(active_pos.get('info', {}).get('avgPrice'))
                # Determine side based on V5 'side' field
                side = CONFIG.pos_long if active_pos.get('info', {}).get('side') == 'Buy' else CONFIG.pos_short

                logger.info(f"{Fore.YELLOW}Position Check: Found ACTIVE {side} position: Qty={abs(size):.8f} @ Entry={entry_price:.4f}{Style.RESET_ALL}")
                return {'side': side, 'qty': abs(size), 'entry_price': entry_price}
            except Exception as parse_err:
                 logger.warning(f"{Fore.YELLOW}Position Check: Error parsing active position data: {parse_err}. Data: {active_pos}{Style.RESET_ALL}")
                 return default_pos
        else:
            logger.info(f"Position Check: No active One-Way position found for {market_id}.")
            return default_pos

    except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
        logger.warning(f"{Fore.YELLOW}Position Check: Error fetching positions for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
    return default_pos


def set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int) -> bool:
    """Sets leverage for a futures symbol (Bybit V5 focus)."""
    logger.info(f"{Fore.CYAN}Leverage Setting: Attempting to set {leverage}x for {symbol}...{Style.RESET_ALL}")
    try:
        market = exchange.market(symbol)
        if not market.get('contract'):
            logger.error(f"{Fore.RED}Leverage Setting: Cannot set for non-contract market: {symbol}.{Style.RESET_ALL}")
            return False
    except Exception as e:
         logger.error(f"{Fore.RED}Leverage Setting: Failed to get market info for '{symbol}': {e}{Style.RESET_ALL}")
         return False

    for attempt in range(CONFIG.retry_count):
        try:
            # Bybit V5 requires setting buy and sell leverage separately
            params = {'buyLeverage': str(leverage), 'sellLeverage': str(leverage)}
            response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
            logger.success(f"{Fore.GREEN}Leverage Setting: Set to {leverage}x for {symbol}. Response: {response}{Style.RESET_ALL}")
            return True
        except ccxt.ExchangeError as e:
            # Check for common "already set" messages
            err_str = str(e).lower()
            if "leverage not modified" in err_str or "leverage is same as requested" in err_str:
                logger.info(f"{Fore.CYAN}Leverage Setting: Already set to {leverage}x for {symbol}.{Style.RESET_ALL}")
                return True
            logger.warning(f"{Fore.YELLOW}Leverage Setting: Exchange error (Attempt {attempt + 1}/{CONFIG.retry_count}): {e}{Style.RESET_ALL}")
            if attempt < CONFIG.retry_count - 1: time.sleep(CONFIG.retry_delay_seconds)
            else: logger.error(f"{Fore.RED}Leverage Setting: Failed after {CONFIG.retry_count} attempts.{Style.RESET_ALL}")
        except (ccxt.NetworkError, Exception) as e:
            logger.warning(f"{Fore.YELLOW}Leverage Setting: Network/Other error (Attempt {attempt + 1}/{CONFIG.retry_count}): {e}{Style.RESET_ALL}")
            if attempt < CONFIG.retry_count - 1: time.sleep(CONFIG.retry_delay_seconds)
            else: logger.error(f"{Fore.RED}Leverage Setting: Failed after {CONFIG.retry_count} attempts.{Style.RESET_ALL}")
    return False


def close_position(exchange: ccxt.Exchange, symbol: str, position_to_close: dict[str, Any], reason: str = "Signal") -> dict[str, Any] | None:
    """Closes the specified active position with re-validation, uses Decimal."""
    initial_side = position_to_close.get('side', CONFIG.pos_none)
    initial_qty = position_to_close.get('qty', Decimal("0.0"))
    market_base = symbol.split('/')[0]
    logger.info(f"{Fore.YELLOW}Close Position: Initiated for {symbol}. Reason: {reason}. Initial state: {initial_side} Qty={initial_qty:.8f}{Style.RESET_ALL}")

    # Re-validate the position just before closing
    live_position = get_current_position(exchange, symbol)
    if live_position['side'] == CONFIG.pos_none:
        logger.warning(f"{Fore.YELLOW}Close Position: Re-validation shows NO active position for {symbol}. Aborting.{Style.RESET_ALL}")
        if initial_side != CONFIG.pos_none: logger.warning(f"{Fore.YELLOW}Close Position: Discrepancy detected (was {initial_side}, now None).{Style.RESET_ALL}")
        return None

    live_amount_to_close = live_position['qty']
    live_position_side = live_position['side']
    side_to_execute_close = CONFIG.side_sell if live_position_side == CONFIG.pos_long else CONFIG.side_buy

    try:
        amount_str = format_amount(exchange, symbol, live_amount_to_close)
        amount_float = float(amount_str)  # CCXT create order expects float
        if amount_float <= float(CONFIG.position_qty_epsilon):
            logger.error(f"{Fore.RED}Close Position: Closing amount after precision is negligible ({amount_str}). Aborting.{Style.RESET_ALL}")
            return None

        logger.warning(f"{Back.YELLOW}{Fore.BLACK}Close Position: Attempting to CLOSE {live_position_side} ({reason}): "
                       f"Exec {side_to_execute_close.upper()} MARKET {amount_str} {symbol} (reduce_only=True)...{Style.RESET_ALL}")
        params = {'reduceOnly': True}
        order = exchange.create_market_order(symbol=symbol, side=side_to_execute_close, amount=amount_float, params=params)

        # Parse order response safely using Decimal
        fill_price = safe_decimal_conversion(order.get('average'))
        filled_qty = safe_decimal_conversion(order.get('filled'))
        cost = safe_decimal_conversion(order.get('cost'))
        order_id_short = format_order_id(order.get('id'))

        logger.success(f"{Fore.GREEN}{Style.BRIGHT}Close Position: Order ({reason}) placed for {symbol}. "
                       f"Filled: {filled_qty:.8f}/{amount_str}, AvgFill: {fill_price:.4f}, Cost: {cost:.2f} USDT. ID:...{order_id_short}{Style.RESET_ALL}")
        send_sms_alert(f"[{market_base}] Closed {live_position_side} {amount_str} @ ~{fill_price:.4f} ({reason}). ID:...{order_id_short}")
        return order

    except (ccxt.InsufficientFunds, ccxt.NetworkError, ccxt.ExchangeError, ValueError, Exception) as e:
        logger.error(f"{Fore.RED}Close Position ({reason}): Failed for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        # Check for specific Bybit errors indicating already closed
        err_str = str(e).lower()
        if isinstance(e, ccxt.ExchangeError) and ("order would not reduce position size" in err_str or "position is zero" in err_str or "position size is zero" in err_str):
             logger.warning(f"{Fore.YELLOW}Close Position: Exchange indicates position already closed/closing. Assuming closed.{Style.RESET_ALL}")
             return None  # Treat as success in this case
        send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): {type(e).__name__}. Check logs.")
    return None


def calculate_position_size(equity: Decimal, risk_per_trade_pct: Decimal, entry_price: Decimal, stop_loss_price: Decimal,
                            leverage: int, symbol: str, exchange: ccxt.Exchange) -> tuple[Decimal | None, Decimal | None]:
    """Calculates position size and estimated margin based on risk, using Decimal."""
    logger.debug(f"Risk Calc: Equity={equity:.4f}, Risk%={risk_per_trade_pct:.4%}, Entry={entry_price:.4f}, SL={stop_loss_price:.4f}, Lev={leverage}x")
    if not (entry_price > 0 and stop_loss_price > 0): logger.error(f"{Fore.RED}Risk Calc: Invalid entry/SL price.{Style.RESET_ALL}"); return None, None
    price_diff = abs(entry_price - stop_loss_price)
    if price_diff < CONFIG.position_qty_epsilon: logger.error(f"{Fore.RED}Risk Calc: Entry/SL prices too close ({price_diff:.8f}).{Style.RESET_ALL}"); return None, None
    if not 0 < risk_per_trade_pct < 1: logger.error(f"{Fore.RED}Risk Calc: Invalid risk %: {risk_per_trade_pct:.4%}.{Style.RESET_ALL}"); return None, None
    if equity <= 0: logger.error(f"{Fore.RED}Risk Calc: Invalid equity: {equity:.4f}{Style.RESET_ALL}"); return None, None
    if leverage <= 0: logger.error(f"{Fore.RED}Risk Calc: Invalid leverage: {leverage}{Style.RESET_ALL}"); return None, None

    risk_amount_usdt = equity * risk_per_trade_pct
    # Assuming linear contract where 1 unit = 1 base currency (e.g., 1 BTC)
    # Risk per unit = price_diff
    quantity_raw = risk_amount_usdt / price_diff

    try:
        # Format according to market precision *then* convert back to Decimal
        quantity_precise_str = format_amount(exchange, symbol, quantity_raw)
        quantity_precise = Decimal(quantity_precise_str)
    except Exception as e:
        logger.warning(f"{Fore.YELLOW}Risk Calc: Failed precision formatting for quantity {quantity_raw:.8f}. Using raw. Error: {e}{Style.RESET_ALL}")
        quantity_precise = quantity_raw.quantize(Decimal('1e-8'), rounding=ROUND_HALF_UP)  # Fallback quantization

    if quantity_precise <= CONFIG.position_qty_epsilon:
        logger.warning(f"{Fore.YELLOW}Risk Calc: Calculated quantity negligible ({quantity_precise:.8f}). RiskAmt={risk_amount_usdt:.4f}, PriceDiff={price_diff:.4f}{Style.RESET_ALL}")
        return None, None

    pos_value_usdt = quantity_precise * entry_price
    required_margin = pos_value_usdt / Decimal(leverage)
    logger.debug(f"Risk Calc Result: Qty={quantity_precise:.8f}, EstValue={pos_value_usdt:.4f}, EstMargin={required_margin:.4f}")
    return quantity_precise, required_margin


def wait_for_order_fill(exchange: ccxt.Exchange, order_id: str, symbol: str, timeout_seconds: int) -> dict[str, Any] | None:
    """Waits for a specific order to be filled (status 'closed')."""
    start_time = time.time()
    logger.info(f"{Fore.CYAN}Waiting for order ...{format_order_id(order_id)} ({symbol}) fill (Timeout: {timeout_seconds}s)...{Style.RESET_ALL}")
    while time.time() - start_time < timeout_seconds:
        try:
            order = exchange.fetch_order(order_id, symbol)
            status = order.get('status')
            logger.debug(f"Order ...{format_order_id(order_id)} status: {status}")
            if status == 'closed':
                logger.success(f"{Fore.GREEN}Order ...{format_order_id(order_id)} confirmed FILLED.{Style.RESET_ALL}")
                return order
            elif status in ['canceled', 'rejected', 'expired']:
                logger.error(f"{Fore.RED}Order ...{format_order_id(order_id)} failed with status '{status}'.{Style.RESET_ALL}")
                return None  # Failed state
            # Continue polling if 'open' or 'partially_filled' or None
            time.sleep(0.5)  # Check every 500ms
        except ccxt.OrderNotFound:
            # This might happen briefly after placing, keep trying
            logger.warning(f"{Fore.YELLOW}Order ...{format_order_id(order_id)} not found yet. Retrying...{Style.RESET_ALL}")
            time.sleep(1)
        except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
            logger.warning(f"{Fore.YELLOW}Error checking order ...{format_order_id(order_id)}: {type(e).__name__} - {e}. Retrying...{Style.RESET_ALL}")
            time.sleep(1)  # Wait longer on error
    logger.error(f"{Fore.RED}Order ...{format_order_id(order_id)} did not fill within {timeout_seconds}s timeout.{Style.RESET_ALL}")
    return None  # Timeout


def place_risked_market_order(exchange: ccxt.Exchange, symbol: str, side: str,
                            risk_percentage: Decimal, current_atr: Decimal | None, sl_atr_multiplier: Decimal,
                            leverage: int, max_order_cap_usdt: Decimal, margin_check_buffer: Decimal,
                            tsl_percent: Decimal, tsl_activation_offset_percent: Decimal) -> dict[str, Any] | None:
    """Places market entry, waits for fill, then places exchange-native fixed SL and TSL using Decimal."""
    market_base = symbol.split('/')[0]
    logger.info(f"{Fore.BLUE}{Style.BRIGHT}Place Order: Initiating {side.upper()} for {symbol}...{Style.RESET_ALL}")
    if current_atr is None or current_atr <= Decimal("0"):
        logger.error(f"{Fore.RED}Place Order ({side.upper()}): Invalid ATR ({current_atr}). Cannot place order.{Style.RESET_ALL}")
        return None

    entry_price_estimate: Decimal | None = None
    initial_sl_price_estimate: Decimal | None = None
    final_quantity: Decimal | None = None
    market: dict | None = None

    try:
        # === 1. Get Balance, Market Info, Limits ===
        logger.debug("Fetching balance & market details...")
        balance = exchange.fetch_balance()
        market = exchange.market(symbol)
        limits = market.get('limits', {})
        amount_limits = limits.get('amount', {})
        price_limits = limits.get('price', {})
        min_qty_str = amount_limits.get('min')
        max_qty_str = amount_limits.get('max')
        min_price_str = price_limits.get('min')
        min_qty = safe_decimal_conversion(min_qty_str) if min_qty_str else None
        max_qty = safe_decimal_conversion(max_qty_str) if max_qty_str else None
        min_price = safe_decimal_conversion(min_price_str) if min_price_str else None

        usdt_balance = balance.get(CONFIG.usdt_symbol, {})
        usdt_total = safe_decimal_conversion(usdt_balance.get('total'))
        usdt_free = safe_decimal_conversion(usdt_balance.get('free'))
        usdt_equity = usdt_total if usdt_total > 0 else usdt_free  # Use total if available, else free

        if usdt_equity <= Decimal("0"): logger.error(f"{Fore.RED}Place Order ({side.upper()}): Zero/Invalid equity ({usdt_equity:.4f}).{Style.RESET_ALL}"); return None
        if usdt_free < Decimal("0"): logger.error(f"{Fore.RED}Place Order ({side.upper()}): Invalid free margin ({usdt_free:.4f}).{Style.RESET_ALL}"); return None
        logger.debug(f"Equity={usdt_equity:.4f}, Free={usdt_free:.4f} USDT")

        # === 2. Estimate Entry Price ===
        ob_data = analyze_order_book(exchange, symbol, CONFIG.shallow_ob_fetch_depth, CONFIG.shallow_ob_fetch_depth)
        best_ask = ob_data.get("best_ask")
        best_bid = ob_data.get("best_bid")
        if side == CONFIG.side_buy and best_ask: entry_price_estimate = best_ask
        elif side == CONFIG.side_sell and best_bid: entry_price_estimate = best_bid
        else:
            try: entry_price_estimate = safe_decimal_conversion(exchange.fetch_ticker(symbol).get('last'))
            except Exception as e: logger.error(f"{Fore.RED}Failed to fetch ticker price: {e}{Style.RESET_ALL}"); return None
        if not entry_price_estimate or entry_price_estimate <= 0: logger.error(f"{Fore.RED}Invalid entry price estimate ({entry_price_estimate}).{Style.RESET_ALL}"); return None
        logger.debug(f"Estimated Entry Price ~ {entry_price_estimate:.4f}")

        # === 3. Calculate Initial Stop Loss Price (Estimate) ===
        sl_distance = current_atr * sl_atr_multiplier
        initial_sl_price_raw = (entry_price_estimate - sl_distance) if side == CONFIG.side_buy else (entry_price_estimate + sl_distance)
        if min_price is not None and initial_sl_price_raw < min_price: initial_sl_price_raw = min_price
        if initial_sl_price_raw <= 0: logger.error(f"{Fore.RED}Invalid Initial SL price calc: {initial_sl_price_raw:.4f}{Style.RESET_ALL}"); return None
        initial_sl_price_estimate = safe_decimal_conversion(format_price(exchange, symbol, initial_sl_price_raw))  # Format estimate
        logger.info(f"Calculated Initial SL Price (Estimate) ~ {initial_sl_price_estimate:.4f} (Dist: {sl_distance:.4f})")

        # === 4. Calculate Position Size ===
        calc_qty, req_margin = calculate_position_size(usdt_equity, risk_percentage, entry_price_estimate, initial_sl_price_estimate, leverage, symbol, exchange)
        if calc_qty is None or req_margin is None: logger.error(f"{Fore.RED}Failed risk calculation.{Style.RESET_ALL}"); return None
        final_quantity = calc_qty

        # === 5. Apply Max Order Cap ===
        pos_value = final_quantity * entry_price_estimate
        if pos_value > max_order_cap_usdt:
            logger.warning(f"{Fore.YELLOW}Order value {pos_value:.4f} > Cap {max_order_cap_usdt:.4f}. Capping qty.{Style.RESET_ALL}")
            final_quantity = max_order_cap_usdt / entry_price_estimate
            # Format capped quantity
            final_quantity = safe_decimal_conversion(format_amount(exchange, symbol, final_quantity))
            req_margin = (max_order_cap_usdt / Decimal(leverage))  # Recalculate margin based on cap

        # === 6. Check Limits & Margin ===
        if final_quantity <= CONFIG.position_qty_epsilon: logger.error(f"{Fore.RED}Final Qty negligible: {final_quantity:.8f}{Style.RESET_ALL}"); return None
        if min_qty is not None and final_quantity < min_qty: logger.error(f"{Fore.RED}Final Qty {final_quantity:.8f} < Min {min_qty}{Style.RESET_ALL}"); return None
        if max_qty is not None and final_quantity > max_qty:
            logger.warning(f"{Fore.YELLOW}Final Qty {final_quantity:.8f} > Max {max_qty}. Capping.{Style.RESET_ALL}")
            final_quantity = max_qty
            final_quantity = safe_decimal_conversion(format_amount(exchange, symbol, final_quantity))  # Re-format capped amount

        final_req_margin = (final_quantity * entry_price_estimate) / Decimal(leverage)  # Final margin estimate
        req_margin_buffered = final_req_margin * margin_check_buffer

        if usdt_free < req_margin_buffered:
            logger.error(f"{Fore.RED}Insufficient FREE margin. Need ~{req_margin_buffered:.4f}, Have {usdt_free:.4f}{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Insufficient Free Margin")
            return None
        logger.info(f"{Fore.GREEN}Final Order: Qty={final_quantity:.8f}, EstValue={final_quantity * entry_price_estimate:.4f}, EstMargin={final_req_margin:.4f}. Margin check OK.{Style.RESET_ALL}")

        # === 7. Place Entry Market Order ===
        entry_order: dict[str, Any] | None = None
        order_id: str | None = None
        try:
            qty_float = float(final_quantity)  # CCXT expects float
            logger.warning(f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}*** Placing {side.upper()} MARKET ENTRY: {qty_float:.8f} {symbol} ***{Style.RESET_ALL}")
            entry_order = exchange.create_market_order(symbol=symbol, side=side, amount=qty_float, params={'reduce_only': False})
            order_id = entry_order.get('id')
            if not order_id:
                logger.error(f"{Fore.RED}Entry order placed but no ID returned! Response: {entry_order}{Style.RESET_ALL}")
                return None
            logger.success(f"{Fore.GREEN}Market Entry Order submitted. ID: ...{format_order_id(order_id)}. Waiting for fill...{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"{Fore.RED}{Style.BRIGHT}FAILED TO PLACE ENTRY ORDER: {e}{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Entry placement failed: {type(e).__name__}")
            return None

        # === 8. Wait for Entry Fill ===
        filled_entry = wait_for_order_fill(exchange, order_id, symbol, CONFIG.order_fill_timeout_seconds)
        if not filled_entry:
            logger.error(f"{Fore.RED}Entry order ...{format_order_id(order_id)} did not fill/failed.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Entry ...{format_order_id(order_id)} fill timeout/fail.")
            # Try to cancel the potentially stuck order (Corrected Block)
            try:
                logger.info(f"{Fore.CYAN}Attempting cancellation of potentially stuck order ...{format_order_id(order_id)}.{Style.RESET_ALL}")
                exchange.cancel_order(order_id, symbol)
            except Exception as cancel_e:
                # Log the cancellation error, but proceed to return None as the entry failed
                logger.warning(f"{Fore.YELLOW}Failed to cancel potentially stuck order ...{format_order_id(order_id)}: {cancel_e}{Style.RESET_ALL}")
                pass  # Allow the function to return None below
            return None  # Return None as the entry failed

        # === 9. Extract Fill Details (Crucial: Use Actual Fill) ===
        avg_fill_price = safe_decimal_conversion(filled_entry.get('average'))
        filled_qty = safe_decimal_conversion(filled_entry.get('filled'))
        cost = safe_decimal_conversion(filled_entry.get('cost'))

        if avg_fill_price <= 0 or filled_qty <= CONFIG.position_qty_epsilon:
            logger.error(f"{Fore.RED}Invalid fill details for ...{format_order_id(order_id)}: Price={avg_fill_price}, Qty={filled_qty}{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Invalid fill details ...{format_order_id(order_id)}.")
            return filled_entry  # Return problematic order

        logger.success(f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}ENTRY CONFIRMED: ...{format_order_id(order_id)}. Filled: {filled_qty:.8f} @ {avg_fill_price:.4f}, Cost: {cost:.4f} USDT{Style.RESET_ALL}")

        # === 10. Calculate ACTUAL Stop Loss Price based on Fill ===
        # Recalculate sl_distance using the confirmed ATR value used for sizing
        sl_distance = current_atr * sl_atr_multiplier
        actual_sl_price_raw = (avg_fill_price - sl_distance) if side == CONFIG.side_buy else (avg_fill_price + sl_distance)
        if min_price is not None and actual_sl_price_raw < min_price: actual_sl_price_raw = min_price
        if actual_sl_price_raw <= 0:
            logger.error(f"{Fore.RED}Invalid ACTUAL SL price calc based on fill: {actual_sl_price_raw:.4f}. Cannot place SL!{Style.RESET_ALL}")
            # CRITICAL: Position is open without SL. Attempt emergency close.
            send_sms_alert(f"[{market_base}] CRITICAL ({side.upper()}): Invalid ACTUAL SL price! Attempting emergency close.")
            close_position(exchange, symbol, {'side': side, 'qty': filled_qty}, reason="Invalid SL Calc")
            return filled_entry  # Return filled entry, but indicate failure state
        actual_sl_price_str = format_price(exchange, symbol, actual_sl_price_raw)
        actual_sl_price_float = float(actual_sl_price_str)  # For CCXT param

        # === 11. Place Initial Fixed Stop Loss ===
        sl_order_id = "N/A"
        try:
            sl_side = CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy
            sl_qty_str = format_amount(exchange, symbol, filled_qty)
            sl_qty_float = float(sl_qty_str)

            logger.info(f"{Fore.CYAN}Placing Initial Fixed SL ({sl_atr_multiplier}*ATR)... Side: {sl_side.upper()}, Qty: {sl_qty_float:.8f}, StopPx: {actual_sl_price_str}{Style.RESET_ALL}")
            # Bybit V5 stop order params: stopPrice (trigger), reduceOnly
            sl_params = {'stopPrice': actual_sl_price_float, 'reduceOnly': True}
            sl_order = exchange.create_order(symbol, 'stopMarket', sl_side, sl_qty_float, params=sl_params)
            sl_order_id = format_order_id(sl_order.get('id'))
            logger.success(f"{Fore.GREEN}Initial Fixed SL order placed. ID: ...{sl_order_id}, Trigger: {actual_sl_price_str}{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"{Fore.RED}{Style.BRIGHT}FAILED to place Initial Fixed SL order: {e}{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())
            send_sms_alert(f"[{market_base}] ERROR ({side.upper()}): Failed initial SL placement: {type(e).__name__}")
            # Don't necessarily close here, TSL might still work, or user might want manual intervention

        # === 12. Place Trailing Stop Loss ===
        tsl_order_id = "N/A"
        tsl_act_price_str = "N/A"
        try:
            # Calculate TSL activation price based on actual fill
            act_offset = avg_fill_price * tsl_activation_offset_percent
            act_price_raw = (avg_fill_price + act_offset) if side == CONFIG.side_buy else (avg_fill_price - act_offset)
            if min_price is not None and act_price_raw < min_price: act_price_raw = min_price
            if act_price_raw <= 0: raise ValueError(f"Invalid TSL activation price {act_price_raw:.4f}")

            tsl_act_price_str = format_price(exchange, symbol, act_price_raw)
            tsl_act_price_float = float(tsl_act_price_str)
            tsl_side = CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy
            # Bybit V5 uses 'trailingStop' for percentage distance (e.g., "0.5" for 0.5%)
            tsl_trail_value_str = str(tsl_percent * Decimal("100"))
            tsl_qty_str = format_amount(exchange, symbol, filled_qty)
            tsl_qty_float = float(tsl_qty_str)

            logger.info(f"{Fore.CYAN}Placing Trailing SL ({tsl_percent:.2%})... Side: {tsl_side.upper()}, Qty: {tsl_qty_float:.8f}, Trail%: {tsl_trail_value_str}, ActPx: {tsl_act_price_str}{Style.RESET_ALL}")
            # Bybit V5 TSL params: trailingStop (percent string), activePrice (activation trigger), reduceOnly
            tsl_params = {
                'trailingStop': tsl_trail_value_str,
                'activePrice': tsl_act_price_float,
                'reduceOnly': True,
            }
            # Use 'stopMarket' type with TSL params for Bybit V5 via CCXT
            tsl_order = exchange.create_order(symbol, 'stopMarket', tsl_side, tsl_qty_float, params=tsl_params)
            tsl_order_id = format_order_id(tsl_order.get('id'))
            logger.success(f"{Fore.GREEN}Trailing SL order placed. ID: ...{tsl_order_id}, Trail%: {tsl_trail_value_str}, ActPx: {tsl_act_price_str}{Style.RESET_ALL}")

            # Final comprehensive SMS
            sms_msg = (f"[{market_base}] ENTERED {side.upper()} {filled_qty:.8f} @ {avg_fill_price:.4f}. "
                       f"Init SL ~{actual_sl_price_str}. TSL {tsl_percent:.2%} act@{tsl_act_price_str}. "
                       f"IDs E:...{format_order_id(order_id)}, SL:...{sl_order_id}, TSL:...{tsl_order_id}")
            send_sms_alert(sms_msg)

        except Exception as e:
            logger.error(f"{Fore.RED}{Style.BRIGHT}FAILED to place Trailing SL order: {e}{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())
            send_sms_alert(f"[{market_base}] ERROR ({side.upper()}): Failed TSL placement: {type(e).__name__}")
            # If TSL fails but initial SL was placed, the position is still protected initially.

        return filled_entry  # Return filled entry order details regardless of SL/TSL placement success

    except (ccxt.InsufficientFunds, ccxt.NetworkError, ccxt.ExchangeError, ValueError, Exception) as e:
        logger.error(f"{Fore.RED}{Style.BRIGHT}Place Order ({side.upper()}): Overall process failed: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Overall process failed: {type(e).__name__}")
    return None


def cancel_open_orders(exchange: ccxt.Exchange, symbol: str, reason: str = "Cleanup") -> None:
    """Attempts to cancel all open orders for the specified symbol."""
    logger.info(f"{Fore.CYAN}Order Cancel: Attempting for {symbol} (Reason: {reason})...{Style.RESET_ALL}")
    try:
        if not exchange.has.get('fetchOpenOrders'):
            logger.warning(f"{Fore.YELLOW}Order Cancel: fetchOpenOrders not supported.{Style.RESET_ALL}")
            return
        open_orders = exchange.fetch_open_orders(symbol)
        if not open_orders:
            logger.info(f"{Fore.CYAN}Order Cancel: No open orders found for {symbol}.{Style.RESET_ALL}")
            return

        logger.warning(f"{Fore.YELLOW}Order Cancel: Found {len(open_orders)} open orders for {symbol}. Cancelling...{Style.RESET_ALL}")
        cancelled_count, failed_count = 0, 0
        for order in open_orders:
            order_id = order.get('id')
            order_info = f"...{format_order_id(order_id)} ({order.get('type')} {order.get('side')})"
            if order_id:
                try:
                    exchange.cancel_order(order_id, symbol)
                    logger.info(f"{Fore.CYAN}Order Cancel: Success for {order_info}{Style.RESET_ALL}")
                    cancelled_count += 1
                    time.sleep(0.1)  # Small delay between cancels
                except ccxt.OrderNotFound:
                    logger.warning(f"{Fore.YELLOW}Order Cancel: Not found (already closed/cancelled?): {order_info}{Style.RESET_ALL}")
                    cancelled_count += 1  # Treat as cancelled if not found
                except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
                    logger.error(f"{Fore.RED}Order Cancel: FAILED for {order_info}: {type(e).__name__} - {e}{Style.RESET_ALL}")
                    failed_count += 1
        logger.info(f"{Fore.CYAN}Order Cancel: Finished. Cancelled: {cancelled_count}, Failed: {failed_count}.{Style.RESET_ALL}")
        if failed_count > 0: send_sms_alert(f"[{symbol.split('/')[0]}] WARNING: Failed to cancel {failed_count} orders during {reason}.")
    except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
        logger.error(f"{Fore.RED}Order Cancel: Failed fetching open orders for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")


# --- Strategy Signal Generation ---
def generate_signals(df: pd.DataFrame, strategy_name: str) -> dict[str, Any]:
    """Generates entry/exit signals based on the selected strategy."""
    signals = {'enter_long': False, 'enter_short': False, 'exit_long': False, 'exit_short': False, 'exit_reason': "Strategy Exit"}
    if len(df) < 2: return signals  # Need previous candle for comparisons/crosses

    last = df.iloc[-1]
    prev = df.iloc[-2]

    try:
        # --- Dual Supertrend Logic ---
        if strategy_name == "DUAL_SUPERTREND":
            # Check if columns exist and are not NA before accessing boolean value
            if 'st_long' in last and pd.notna(last['st_long']) and last['st_long'] and \
               'confirm_trend' in last and pd.notna(last['confirm_trend']) and last['confirm_trend']:
                signals['enter_long'] = True
            if 'st_short' in last and pd.notna(last['st_short']) and last['st_short'] and \
               'confirm_trend' in last and pd.notna(last['confirm_trend']) and not last['confirm_trend']:
                signals['enter_short'] = True
            if 'st_short' in last and pd.notna(last['st_short']) and last['st_short']:
                signals['exit_long'] = True; signals['exit_reason'] = "Primary ST Short Flip"
            if 'st_long' in last and pd.notna(last['st_long']) and last['st_long']:
                signals['exit_short'] = True; signals['exit_reason'] = "Primary ST Long Flip"

        # --- StochRSI + Momentum Logic ---
        elif strategy_name == "STOCHRSI_MOMENTUM":
            required_cols = ['stochrsi_k', 'stochrsi_d', 'momentum']
            if not all(col in last and pd.notna(last[col]) for col in required_cols) or \
               not all(col in prev and pd.notna(prev[col]) for col in ['stochrsi_k', 'stochrsi_d']):
                logger.debug("StochRSI/Mom signals skipped due to NA values.")
                return signals  # Not enough data or indicator failed

            k_now, d_now, mom_now = last['stochrsi_k'], last['stochrsi_d'], last['momentum']
            k_prev, d_prev = prev['stochrsi_k'], prev['stochrsi_d']

            if k_prev <= d_prev and k_now > d_now and k_now < CONFIG.stochrsi_oversold and mom_now > CONFIG.position_qty_epsilon: signals['enter_long'] = True
            if k_prev >= d_prev and k_now < d_now and k_now > CONFIG.stochrsi_overbought and mom_now < -CONFIG.position_qty_epsilon: signals['enter_short'] = True
            if k_prev >= d_prev and k_now < d_now: signals['exit_long'] = True; signals['exit_reason'] = "StochRSI K below D"
            if k_prev <= d_prev and k_now > d_now: signals['exit_short'] = True; signals['exit_reason'] = "StochRSI K above D"

        # --- Ehlers Fisher Logic ---
        elif strategy_name == "EHLERS_FISHER":
            required_cols = ['ehlers_fisher', 'ehlers_signal']
            if not all(col in last and pd.notna(last[col]) for col in required_cols) or \
               not all(col in prev and pd.notna(prev[col]) for col in required_cols):
                logger.debug("Ehlers Fisher signals skipped due to NA values.")
                return signals

            fish_now, sig_now = last['ehlers_fisher'], last['ehlers_signal']
            fish_prev, sig_prev = prev['ehlers_fisher'], prev['ehlers_signal']

            if fish_prev <= sig_prev and fish_now > sig_now: signals['enter_long'] = True
            if fish_prev >= sig_prev and fish_now < sig_now: signals['enter_short'] = True
            if fish_prev >= sig_prev and fish_now < sig_now: signals['exit_long'] = True; signals['exit_reason'] = "Ehlers Fisher Short Cross"
            if fish_prev <= sig_prev and fish_now > sig_now: signals['exit_short'] = True; signals['exit_reason'] = "Ehlers Fisher Long Cross"

        # --- Ehlers MA Cross Logic ---
        elif strategy_name == "EHLERS_MA_CROSS":
            required_cols = ['fast_ema', 'slow_ema']
            if not all(col in last and pd.notna(last[col]) for col in required_cols) or \
               not all(col in prev and pd.notna(prev[col]) for col in required_cols):
                logger.debug("Ehlers MA signals skipped due to NA values.")
                return signals

            fast_ma_now, slow_ma_now = last['fast_ema'], last['slow_ema']
            fast_ma_prev, slow_ma_prev = prev['fast_ema'], prev['slow_ema']

            if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now: signals['enter_long'] = True
            if fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now: signals['enter_short'] = True
            if fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now: signals['exit_long'] = True; signals['exit_reason'] = "Ehlers MA Short Cross"
            if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now: signals['exit_short'] = True; signals['exit_reason'] = "Ehlers MA Long Cross"

    except KeyError as e:
        logger.error(f"{Fore.RED}Signal Generation Error: Missing expected column in DataFrame: {e}{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"{Fore.RED}Signal Generation Error: Unexpected exception: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())

    return signals


# --- Trading Logic ---
def trade_logic(exchange: ccxt.Exchange, symbol: str, df: pd.DataFrame) -> None:
    """Executes the main trading logic for one cycle based on selected strategy."""
    cycle_time_str = df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z') if not df.empty else "N/A"
    logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== New Check Cycle ({CONFIG.strategy_name}): {symbol} | Candle: {cycle_time_str} =========={Style.RESET_ALL}")

    # Determine required rows based on the longest lookback needed by any indicator used
    required_rows = max(
        CONFIG.st_atr_length, CONFIG.confirm_st_atr_length,
        CONFIG.stochrsi_rsi_length + CONFIG.stochrsi_stoch_length, CONFIG.momentum_length,  # Estimate
        CONFIG.ehlers_fisher_length, CONFIG.ehlers_fisher_signal_length,
        CONFIG.ehlers_fast_period, CONFIG.ehlers_slow_period,
        CONFIG.atr_calculation_period, CONFIG.volume_ma_period
    ) + 10  # Add buffer

    if df is None or len(df) < required_rows:
        logger.warning(f"{Fore.YELLOW}Trade Logic: Insufficient data ({len(df) if df is not None else 0}, need ~{required_rows}). Skipping.{Style.RESET_ALL}")
        return

    action_taken_this_cycle: bool = False
    try:
        # === 1. Calculate ALL Indicators ===
        # It's often simpler to calculate all potential indicators needed by any strategy
        # and let the signal generation function pick the ones it needs.
        logger.debug("Calculating indicators...")
        df = calculate_supertrend(df, CONFIG.st_atr_length, CONFIG.st_multiplier)
        df = calculate_supertrend(df, CONFIG.confirm_st_atr_length, CONFIG.confirm_st_multiplier, prefix="confirm_")
        df = calculate_stochrsi_momentum(df, CONFIG.stochrsi_rsi_length, CONFIG.stochrsi_stoch_length, CONFIG.stochrsi_k_period, CONFIG.stochrsi_d_period, CONFIG.momentum_length)
        df = calculate_ehlers_fisher(df, CONFIG.ehlers_fisher_length, CONFIG.ehlers_fisher_signal_length)
        df = calculate_ehlers_ma(df, CONFIG.ehlers_fast_period, CONFIG.ehlers_slow_period)
        vol_atr_data = analyze_volume_atr(df, CONFIG.atr_calculation_period, CONFIG.volume_ma_period)
        current_atr = vol_atr_data.get("atr")

        # === 2. Validate Base Requirements ===
        last = df.iloc[-1]
        current_price = safe_decimal_conversion(last.get('close'))  # Use .get for safety
        if pd.isna(current_price) or current_price <= 0:
            logger.warning(f"{Fore.YELLOW}Last candle close price is invalid ({current_price}). Skipping.{Style.RESET_ALL}")
            return
        can_place_order = current_atr is not None and current_atr > Decimal("0")
        if not can_place_order:
            logger.warning(f"{Fore.YELLOW}Invalid ATR ({current_atr}). Cannot calculate SL or place new orders.{Style.RESET_ALL}")

        # === 3. Get Position & Analyze OB ===
        position = get_current_position(exchange, symbol)
        position_side = position['side']
        ob_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit) if CONFIG.fetch_order_book_per_cycle else None

        # === 4. Log State ===
        vol_ratio = vol_atr_data.get("volume_ratio")
        vol_spike = vol_ratio is not None and vol_ratio > CONFIG.volume_spike_threshold
        bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None
        spread = ob_data.get("spread") if ob_data else None

        logger.info(f"State | Price: {current_price:.4f}, ATR({CONFIG.atr_calculation_period}): {current_atr:.5f}" if current_atr else f"State | Price: {current_price:.4f}, ATR({CONFIG.atr_calculation_period}): N/A")
        logger.info(f"State | Volume: Ratio={vol_ratio:.2f if vol_ratio else 'N/A'}, Spike={vol_spike} (Req={CONFIG.require_volume_spike_for_entry})")
        # Log specific strategy indicators
        # ... (Add logging for relevant indicators based on CONFIG.strategy_name if needed, or rely on debug logs from calc functions) ...
        logger.info(f"State | OrderBook: Ratio={bid_ask_ratio:.3f if bid_ask_ratio else 'N/A'}, Spread={spread:.4f if spread else 'N/A'}")
        logger.info(f"State | Position: Side={position_side}, Qty={position['qty']:.8f}, Entry={position['entry_price']:.4f}")

        # === 5. Generate Strategy Signals ===
        strategy_signals = generate_signals(df, CONFIG.strategy_name)
        logger.debug(f"Strategy Signals ({CONFIG.strategy_name}): {strategy_signals}")

        # === 6. Execute Exit Actions ===
        should_exit_long = position_side == CONFIG.pos_long and strategy_signals['exit_long']
        should_exit_short = position_side == CONFIG.pos_short and strategy_signals['exit_short']

        if should_exit_long or should_exit_short:
            exit_reason = strategy_signals['exit_reason']
            logger.warning(f"{Back.YELLOW}{Fore.BLACK}*** TRADE EXIT SIGNAL: Closing {position_side} due to {exit_reason} ***{Style.RESET_ALL}")
            cancel_open_orders(exchange, symbol, f"SL/TSL before {exit_reason} Exit")
            close_result = close_position(exchange, symbol, position, reason=exit_reason)
            if close_result: action_taken_this_cycle = True
            # Add delay after closing before allowing new entry
            if action_taken_this_cycle:
                logger.info(f"Pausing for {CONFIG.post_close_delay_seconds}s after closing position...")
                time.sleep(CONFIG.post_close_delay_seconds)
            return  # Exit cycle after attempting close

        # === 7. Check & Execute Entry Actions (Only if Flat & Can Place Order) ===
        if position_side != CONFIG.pos_none:
             logger.info(f"Holding {position_side} position. Waiting for SL/TSL or Strategy Exit.")
             return
        if not can_place_order:
             logger.warning(f"{Fore.YELLOW}Holding Cash. Cannot enter due to invalid ATR for SL calculation.{Style.RESET_ALL}")
             return

        logger.debug("Checking entry signals...")
        # --- Define Confirmation Conditions ---
        potential_entry = strategy_signals['enter_long'] or strategy_signals['enter_short']
        if not CONFIG.fetch_order_book_per_cycle and potential_entry and ob_data is None:
            logger.debug("Potential entry signal, fetching OB for confirmation...")
            ob_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit)
            bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None  # Update ratio

        # Check OB confirmation only if required
        ob_check_required = potential_entry  # Always check OB if entry signal exists? Or make configurable? Let's assume yes for now.
        ob_available = ob_data is not None and bid_ask_ratio is not None
        passes_long_ob = not ob_check_required or (ob_available and bid_ask_ratio >= CONFIG.order_book_ratio_threshold_long)
        passes_short_ob = not ob_check_required or (ob_available and bid_ask_ratio <= CONFIG.order_book_ratio_threshold_short)
        ob_log = f"OB OK (L:{passes_long_ob},S:{passes_short_ob}, Ratio={bid_ask_ratio:.3f if bid_ask_ratio else 'N/A'}, Req={ob_check_required})"

        # Check Volume confirmation only if required
        vol_check_required = CONFIG.require_volume_spike_for_entry
        passes_volume = not vol_check_required or (vol_spike)
        vol_log = f"Vol OK (Pass:{passes_volume}, Spike={vol_spike}, Req={vol_check_required})"

        # --- Combine Strategy Signal with Confirmations ---
        enter_long = strategy_signals['enter_long'] and passes_long_ob and passes_volume
        enter_short = strategy_signals['enter_short'] and passes_short_ob and passes_volume
        logger.debug(f"Final Entry Check (Long): Strategy={strategy_signals['enter_long']}, {ob_log}, {vol_log} => Enter={enter_long}")
        logger.debug(f"Final Entry Check (Short): Strategy={strategy_signals['enter_short']}, {ob_log}, {vol_log} => Enter={enter_short}")

        # --- Execute ---
        if enter_long:
            logger.success(f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}*** TRADE SIGNAL: CONFIRMED LONG ENTRY ({CONFIG.strategy_name}) for {symbol} ***{Style.RESET_ALL}")
            cancel_open_orders(exchange, symbol, "Before Long Entry")  # Cancel previous SL/TSL just in case
            place_result = place_risked_market_order(
                exchange, symbol, CONFIG.side_buy, CONFIG.risk_per_trade_percentage, current_atr, CONFIG.atr_stop_loss_multiplier,
                CONFIG.leverage, CONFIG.max_order_usdt_amount, CONFIG.required_margin_buffer,
                CONFIG.trailing_stop_percentage, CONFIG.trailing_stop_activation_offset_percent)
            if place_result: action_taken_this_cycle = True

        elif enter_short:
            logger.success(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}*** TRADE SIGNAL: CONFIRMED SHORT ENTRY ({CONFIG.strategy_name}) for {symbol} ***{Style.RESET_ALL}")
            cancel_open_orders(exchange, symbol, "Before Short Entry")
            place_result = place_risked_market_order(
                exchange, symbol, CONFIG.side_sell, CONFIG.risk_per_trade_percentage, current_atr, CONFIG.atr_stop_loss_multiplier,
                CONFIG.leverage, CONFIG.max_order_usdt_amount, CONFIG.required_margin_buffer,
                CONFIG.trailing_stop_percentage, CONFIG.trailing_stop_activation_offset_percent)
            if place_result: action_taken_this_cycle = True

        else:
             if not action_taken_this_cycle: logger.info("No confirmed entry signal. Holding cash.")

    except Exception as e:
        logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}CRITICAL UNEXPECTED ERROR in trade_logic: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{symbol.split('/')[0]}] CRITICAL ERROR in trade_logic: {type(e).__name__}")
    finally:
        logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== Cycle Check End: {symbol} =========={Style.RESET_ALL}\n")


# --- Graceful Shutdown ---
def graceful_shutdown(exchange: ccxt.Exchange | None, symbol: str | None) -> None:
    """Attempts to close position and cancel orders before exiting."""
    logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}Shutdown requested. Initiating graceful exit...{Style.RESET_ALL}")
    market_base = symbol.split('/')[0] if symbol else "Bot"
    send_sms_alert(f"[{market_base}] Shutdown requested. Attempting cleanup...")
    if not exchange or not symbol:
        logger.warning(f"{Fore.YELLOW}Shutdown: Exchange/Symbol not available.{Style.RESET_ALL}")
        return

    try:
        # 1. Cancel All Open Orders
        cancel_open_orders(exchange, symbol, reason="Graceful Shutdown")
        time.sleep(1)  # Allow cancellations to process

        # 2. Check and Close Position
        position = get_current_position(exchange, symbol)
        if position['side'] != CONFIG.pos_none:
            logger.warning(f"{Fore.YELLOW}Shutdown: Active {position['side']} position found (Qty: {position['qty']:.8f}). Closing...{Style.RESET_ALL}")
            close_result = close_position(exchange, symbol, position, reason="Shutdown")
            if close_result:
                logger.info(f"{Fore.CYAN}Shutdown: Close order placed. Waiting {CONFIG.post_close_delay_seconds * 2}s for confirmation...{Style.RESET_ALL}")
                time.sleep(CONFIG.post_close_delay_seconds * 2)
                # Final check
                final_pos = get_current_position(exchange, symbol)
                if final_pos['side'] == CONFIG.pos_none:
                    logger.success(f"{Fore.GREEN}{Style.BRIGHT}Shutdown: Position confirmed CLOSED.{Style.RESET_ALL}")
                    send_sms_alert(f"[{market_base}] Position confirmed CLOSED on shutdown.")
                else:
                    logger.error(f"{Back.RED}{Fore.WHITE}Shutdown: FAILED TO CONFIRM closure. Final state: {final_pos['side']} Qty={final_pos['qty']:.8f}{Style.RESET_ALL}")
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


# --- Main Execution ---
def main() -> None:
    """Main function to initialize, set up, and run the trading loop."""
    start_time = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Bybit Scalping Spell v2.0.1 Initializing ({start_time}) ---{Style.RESET_ALL}")
    logger.info(f"{Fore.CYAN}--- Strategy Enchantment: {CONFIG.strategy_name} ---{Style.RESET_ALL}")
    logger.info(f"{Fore.GREEN}--- Warding Rune: Initial ATR + Exchange Trailing Stop ---{Style.RESET_ALL}")
    logger.warning(f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}--- !!! LIVE FUTURES SCALPING MODE - EXTREME RISK INVOLVED !!! ---{Style.RESET_ALL}")

    exchange: ccxt.Exchange | None = None
    symbol: str | None = None
    run_bot: bool = True
    cycle_count: int = 0

    try:
        # Initialize Exchange
        exchange = initialize_exchange()
        if not exchange: return

        # Setup Symbol and Leverage
        try:
            # Allow user input or use default from config
            sym_input = input(f"{Fore.YELLOW}Enter symbol {Style.DIM}(Default [{CONFIG.symbol}]){Style.NORMAL}: {Style.RESET_ALL}").strip()
            symbol_to_use = sym_input or CONFIG.symbol
            market = exchange.market(symbol_to_use)
            symbol = market['symbol']  # Use the unified symbol from CCXT
            if not market.get('contract'): raise ValueError("Not a contract/futures market")
            logger.info(f"{Fore.GREEN}Using Symbol: {symbol} (Type: {market.get('type')}){Style.RESET_ALL}")
            if not set_leverage(exchange, symbol, CONFIG.leverage): raise RuntimeError("Leverage setup failed")
        except (ccxt.BadSymbol, KeyError, ValueError, RuntimeError) as e:
            logger.critical(f"Symbol/Leverage setup failed: {e}")
            send_sms_alert(f"[ScalpBot] CRITICAL: Symbol/Leverage setup FAILED ({e}). Exiting.")
            return
        except Exception as e:
            logger.critical(f"Unexpected error during setup: {e}")
            send_sms_alert("[ScalpBot] CRITICAL: Unexpected setup error. Exiting.")
            return

        # Log Config Summary
        logger.info(f"{Fore.MAGENTA}--- Configuration Summary ---{Style.RESET_ALL}")
        logger.info(f"{Fore.WHITE}Symbol: {symbol}, Interval: {CONFIG.interval}, Leverage: {CONFIG.leverage}x")
        logger.info(f"{Fore.CYAN}Strategy: {CONFIG.strategy_name}")
        # Log relevant strategy params
        if CONFIG.strategy_name == "DUAL_SUPERTREND": logger.info(f"  Params: ST={CONFIG.st_atr_length}/{CONFIG.st_multiplier}, ConfirmST={CONFIG.confirm_st_atr_length}/{CONFIG.confirm_st_multiplier}")
        elif CONFIG.strategy_name == "STOCHRSI_MOMENTUM": logger.info(f"  Params: StochRSI={CONFIG.stochrsi_rsi_length}/{CONFIG.stochrsi_stoch_length}/{CONFIG.stochrsi_k_period}/{CONFIG.stochrsi_d_period} (OB={CONFIG.stochrsi_overbought},OS={CONFIG.stochrsi_oversold}), Mom={CONFIG.momentum_length}")
        elif CONFIG.strategy_name == "EHLERS_FISHER": logger.info(f"  Params: Fisher={CONFIG.ehlers_fisher_length}, Signal={CONFIG.ehlers_fisher_signal_length}")
        elif CONFIG.strategy_name == "EHLERS_MA_CROSS": logger.info(f"  Params: FastMA={CONFIG.ehlers_fast_period}, SlowMA={CONFIG.ehlers_slow_period}")
        logger.info(f"{Fore.GREEN}Risk: {CONFIG.risk_per_trade_percentage:.3%}/trade, MaxPosValue: {CONFIG.max_order_usdt_amount:.4f} USDT")
        logger.info(f"{Fore.GREEN}Initial SL: {CONFIG.atr_stop_loss_multiplier} * ATR({CONFIG.atr_calculation_period})")
        logger.info(f"{Fore.GREEN}Trailing SL: {CONFIG.trailing_stop_percentage:.2%}, Activation Offset: {CONFIG.trailing_stop_activation_offset_percent:.2%}")
        logger.info(f"{Fore.YELLOW}Vol Confirm: {CONFIG.require_volume_spike_for_entry} (MA={CONFIG.volume_ma_period}, Thr={CONFIG.volume_spike_threshold})")
        logger.info(f"{Fore.YELLOW}OB Confirm: {CONFIG.fetch_order_book_per_cycle} (Depth={CONFIG.order_book_depth}, L>={CONFIG.order_book_ratio_threshold_long}, S<={CONFIG.order_book_ratio_threshold_short})")
        logger.info(f"{Fore.WHITE}Sleep: {CONFIG.sleep_seconds}s, Margin Buffer: {CONFIG.required_margin_buffer:.1%}, SMS: {CONFIG.enable_sms_alerts}")
        logger.info(f"{Fore.CYAN}Logging Level: {logging.getLevelName(logger.level)}")
        logger.info(f"{Fore.MAGENTA}{'-' * 30}{Style.RESET_ALL}")
        market_base = symbol.split('/')[0]
        send_sms_alert(f"[{market_base}] Bot configured ({CONFIG.strategy_name}). SL: ATR+TSL. Starting loop.")

        # Main Trading Loop
        while run_bot:
            cycle_start_time = time.monotonic()
            cycle_count += 1
            logger.debug(f"{Fore.CYAN}--- Cycle {cycle_count} Start ---{Style.RESET_ALL}")
            try:
                # Determine required data length based on longest possible indicator lookback
                data_limit = max(100, CONFIG.st_atr_length * 2, CONFIG.confirm_st_atr_length * 2,
                                 CONFIG.stochrsi_rsi_length + CONFIG.stochrsi_stoch_length + 5, CONFIG.momentum_length * 2,
                                 CONFIG.ehlers_fisher_length * 2, CONFIG.ehlers_fisher_signal_length * 2,
                                 CONFIG.ehlers_fast_period * 2, CONFIG.ehlers_slow_period * 2,
                                 CONFIG.atr_calculation_period * 2, CONFIG.volume_ma_period * 2) + CONFIG.api_fetch_limit_buffer

                df = get_market_data(exchange, symbol, CONFIG.interval, limit=data_limit)

                if df is not None and not df.empty:
                    trade_logic(exchange, symbol, df.copy())  # Pass copy to avoid modifying original in logic
                else:
                    logger.warning(f"{Fore.YELLOW}No valid market data for {symbol}. Skipping cycle.{Style.RESET_ALL}")

            # --- Robust Error Handling ---
            except ccxt.RateLimitExceeded as e:
                logger.warning(f"{Back.YELLOW}{Fore.BLACK}Rate Limit Exceeded: {e}. Sleeping longer...{Style.RESET_ALL}"); time.sleep(CONFIG.sleep_seconds * 5); send_sms_alert(f"[{market_base}] WARNING: Rate limit hit!")
            except ccxt.NetworkError as e:
                logger.warning(f"{Fore.YELLOW}Network error: {e}. Retrying next cycle.{Style.RESET_ALL}")
                time.sleep(CONFIG.sleep_seconds)  # Standard sleep on recoverable network errors
            except ccxt.ExchangeNotAvailable as e:
                logger.error(f"{Back.RED}{Fore.WHITE}Exchange unavailable: {e}. Sleeping much longer...{Style.RESET_ALL}"); time.sleep(CONFIG.sleep_seconds * 10); send_sms_alert(f"[{market_base}] ERROR: Exchange unavailable!")
            except ccxt.AuthenticationError as e:
                logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Authentication Error: {e}. Stopping NOW.{Style.RESET_ALL}"); run_bot = False; send_sms_alert(f"[{market_base}] CRITICAL: Authentication Error! Stopping NOW.")
            except ccxt.ExchangeError as e:  # Catch broader exchange errors
                logger.error(f"{Fore.RED}Unhandled Exchange error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); send_sms_alert(f"[{market_base}] ERROR: Unhandled Exchange error: {type(e).__name__}")
                time.sleep(CONFIG.sleep_seconds)  # Sleep before retrying after general exchange error
            except Exception as e:
                logger.exception(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}!!! UNEXPECTED CRITICAL ERROR: {e} !!!{Style.RESET_ALL}"); run_bot = False; send_sms_alert(f"[{market_base}] CRITICAL UNEXPECTED ERROR: {type(e).__name__}! Stopping NOW.")

            # --- Loop Delay ---
            if run_bot:
                elapsed = time.monotonic() - cycle_start_time
                sleep_dur = max(0, CONFIG.sleep_seconds - elapsed)
                logger.debug(f"Cycle {cycle_count} time: {elapsed:.2f}s. Sleeping: {sleep_dur:.2f}s.")
                if sleep_dur > 0: time.sleep(sleep_dur)

    except KeyboardInterrupt:
        logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}KeyboardInterrupt received. Arcane energies withdrawing...{Style.RESET_ALL}")
        run_bot = False  # Ensure loop terminates
    finally:
        # --- Graceful Shutdown ---
        graceful_shutdown(exchange, symbol)
        market_base_final = symbol.split('/')[0] if symbol else "Bot"
        send_sms_alert(f"[{market_base_final}] Bot process terminated.")
        logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Scalping Spell Deactivated ---{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
