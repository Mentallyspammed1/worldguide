#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v2.1.1 (Enhanced Precision, Strategy Selection, Refined Robustness)
# Conjures high-frequency trades on Bybit Futures with enhanced precision, adaptable strategies, and improved resilience.

"""High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 2.1.1 (Unified: Selectable Strategies + Precision + Native SL/TSL + Enhancements).

Enhancements in v2.1.1:
- Improved documentation and comments.
- More robust error handling across various functions (API calls, calculations).
- Enhanced position detection logic for Bybit V5 (One-Way mode).
- Stricter NaN handling in market data fetching.
- Refined logging messages and color usage for clarity.
- Improved SMS alert content and triggers.
- Safer Decimal conversions and fallback mechanisms.
- Explicit handling of Bybit V5 parameters (e.g., reduceOnly, stop order structure).
- Added warning for Ehlers MA placeholder implementation.
- More robust graceful shutdown sequence.
- Consolidated cycle count logic.

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
- Refined error handling, logging, and robustness checks.

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
from decimal import ROUND_DOWN, Decimal, InvalidOperation, getcontext
from typing import Any, Dict, Optional, Tuple, List

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
    print(f"Error: Missing required package '{missing_pkg}'.")
    print(f"Please install it using: pip install {missing_pkg}")
    # Consider providing installation instructions for all packages
    print("Required packages: ccxt pandas pandas_ta colorama python-dotenv")
    sys.exit(1)

# --- Initializations ---
colorama_init(autoreset=True)
load_dotenv()
getcontext().prec = 18  # Set Decimal precision globally


# --- Logger Setup ---
LOGGING_LEVEL: int = (
    logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO
)
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],  # Ensure logs go to stdout
)
logger: logging.Logger = logging.getLogger(__name__)

# Custom SUCCESS level and Neon Color Formatting
SUCCESS_LEVEL: int = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Logs a message with SUCCESS level."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        # pylint: disable=protected-access
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


logging.Logger.success = log_success  # type: ignore[attr-defined]

# Apply colors only if output is a TTY (supports colors)
if sys.stdout.isatty():
    logging.addLevelName(
        logging.DEBUG,
        f"{Fore.CYAN}{logging.getLevelName(logging.DEBUG)}{Style.RESET_ALL}",
    )
    logging.addLevelName(
        logging.INFO,
        f"{Fore.BLUE}{logging.getLevelName(logging.INFO)}{Style.RESET_ALL}",
    )
    logging.addLevelName(
        SUCCESS_LEVEL,
        f"{Fore.MAGENTA}{logging.getLevelName(SUCCESS_LEVEL)}{Style.RESET_ALL}",
    )
    logging.addLevelName(
        logging.WARNING,
        f"{Fore.YELLOW}{logging.getLevelName(logging.WARNING)}{Style.RESET_ALL}",
    )
    logging.addLevelName(
        logging.ERROR,
        f"{Fore.RED}{logging.getLevelName(logging.ERROR)}{Style.RESET_ALL}",
    )
    logging.addLevelName(
        logging.CRITICAL,
        f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}{logging.getLevelName(logging.CRITICAL)}{Style.RESET_ALL}",
    )


# --- Configuration Class ---
class Config:
    """Loads and validates configuration parameters from environment variables."""

    def __init__(self) -> None:
        logger.info(
            f"{Fore.MAGENTA}--- Summoning Configuration Runes ---{Style.RESET_ALL}"
        )
        # --- API Credentials ---
        self.api_key: Optional[str] = self._get_env(
            "BYBIT_API_KEY", required=True, color=Fore.RED
        )
        self.api_secret: Optional[str] = self._get_env(
            "BYBIT_API_SECRET", required=True, color=Fore.RED
        )

        # --- Trading Parameters ---
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT", color=Fore.YELLOW)
        self.interval: str = self._get_env("INTERVAL", "1m", color=Fore.YELLOW)
        self.leverage: int = self._get_env(
            "LEVERAGE", 10, cast_type=int, color=Fore.YELLOW
        )
        self.sleep_seconds: int = self._get_env(
            "SLEEP_SECONDS", 10, cast_type=int, color=Fore.YELLOW
        )

        # --- Strategy Selection ---
        self.strategy_name: str = self._get_env(
            "STRATEGY_NAME", "DUAL_SUPERTREND", color=Fore.CYAN
        ).upper()
        self.valid_strategies: List[str] = [
            "DUAL_SUPERTREND",
            "STOCHRSI_MOMENTUM",
            "EHLERS_FISHER",
            "EHLERS_MA_CROSS",
        ]
        if self.strategy_name not in self.valid_strategies:
            raise ValueError(
                f"Invalid STRATEGY_NAME '{self.strategy_name}'. Valid options: {self.valid_strategies}"
            )

        # --- Risk Management ---
        self.risk_per_trade_percentage: Decimal = self._get_env(
            "RISK_PER_TRADE_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN
        )  # 0.5% risk per trade
        self.atr_stop_loss_multiplier: Decimal = self._get_env(
            "ATR_STOP_LOSS_MULTIPLIER", "1.5", cast_type=Decimal, color=Fore.GREEN
        )  # Multiplier for ATR-based initial SL
        self.max_order_usdt_amount: Decimal = self._get_env(
            "MAX_ORDER_USDT_AMOUNT", "500.0", cast_type=Decimal, color=Fore.GREEN
        )  # Max position value in USDT
        self.required_margin_buffer: Decimal = self._get_env(
            "REQUIRED_MARGIN_BUFFER", "1.05", cast_type=Decimal, color=Fore.GREEN
        )  # e.g., 1.05 means 5% buffer on required margin

        # --- Trailing Stop Loss (Exchange Native) ---
        self.trailing_stop_percentage: Decimal = self._get_env(
            "TRAILING_STOP_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN
        )  # e.g., 0.005 = 0.5% trail distance
        self.trailing_stop_activation_offset_percent: Decimal = self._get_env(
            "TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT",
            "0.001",
            cast_type=Decimal,
            color=Fore.GREEN,
        )  # e.g., 0.001 = 0.1% offset from entry to activate TSL

        # --- Dual Supertrend Parameters ---
        self.st_atr_length: int = self._get_env(
            "ST_ATR_LENGTH", 7, cast_type=int, color=Fore.CYAN
        )
        self.st_multiplier: Decimal = self._get_env(
            "ST_MULTIPLIER", "2.5", cast_type=Decimal, color=Fore.CYAN
        )
        self.confirm_st_atr_length: int = self._get_env(
            "CONFIRM_ST_ATR_LENGTH", 5, cast_type=int, color=Fore.CYAN
        )
        self.confirm_st_multiplier: Decimal = self._get_env(
            "CONFIRM_ST_MULTIPLIER", "2.0", cast_type=Decimal, color=Fore.CYAN
        )

        # --- StochRSI + Momentum Parameters ---
        self.stochrsi_rsi_length: int = self._get_env(
            "STOCHRSI_RSI_LENGTH", 14, cast_type=int, color=Fore.CYAN
        )
        self.stochrsi_stoch_length: int = self._get_env(
            "STOCHRSI_STOCH_LENGTH", 14, cast_type=int, color=Fore.CYAN
        )
        self.stochrsi_k_period: int = self._get_env(
            "STOCHRSI_K_PERIOD", 3, cast_type=int, color=Fore.CYAN
        )
        self.stochrsi_d_period: int = self._get_env(
            "STOCHRSI_D_PERIOD", 3, cast_type=int, color=Fore.CYAN
        )
        self.stochrsi_overbought: Decimal = self._get_env(
            "STOCHRSI_OVERBOUGHT", "80.0", cast_type=Decimal, color=Fore.CYAN
        )
        self.stochrsi_oversold: Decimal = self._get_env(
            "STOCHRSI_OVERSOLD", "20.0", cast_type=Decimal, color=Fore.CYAN
        )
        self.momentum_length: int = self._get_env(
            "MOMENTUM_LENGTH", 5, cast_type=int, color=Fore.CYAN
        )

        # --- Ehlers Fisher Transform Parameters ---
        self.ehlers_fisher_length: int = self._get_env(
            "EHLERS_FISHER_LENGTH", 10, cast_type=int, color=Fore.CYAN
        )
        self.ehlers_fisher_signal_length: int = self._get_env(
            "EHLERS_FISHER_SIGNAL_LENGTH", 1, cast_type=int, color=Fore.CYAN
        )  # Signal = 1 means Fisher line only

        # --- Ehlers MA Cross Parameters ---
        self.ehlers_fast_period: int = self._get_env(
            "EHLERS_FAST_PERIOD", 10, cast_type=int, color=Fore.CYAN
        )
        self.ehlers_slow_period: int = self._get_env(
            "EHLERS_SLOW_PERIOD", 30, cast_type=int, color=Fore.CYAN
        )

        # --- Volume Analysis ---
        self.volume_ma_period: int = self._get_env(
            "VOLUME_MA_PERIOD", 20, cast_type=int, color=Fore.YELLOW
        )
        self.volume_spike_threshold: Decimal = self._get_env(
            "VOLUME_SPIKE_THRESHOLD", "1.5", cast_type=Decimal, color=Fore.YELLOW
        )  # Ratio of current vol to MA
        self.require_volume_spike_for_entry: bool = self._get_env(
            "REQUIRE_VOLUME_SPIKE_FOR_ENTRY", "false", cast_type=bool, color=Fore.YELLOW
        )

        # --- Order Book Analysis ---
        self.order_book_depth: int = self._get_env(
            "ORDER_BOOK_DEPTH", 10, cast_type=int, color=Fore.YELLOW
        )  # Levels to sum for ratio
        self.order_book_ratio_threshold_long: Decimal = self._get_env(
            "ORDER_BOOK_RATIO_THRESHOLD_LONG",
            "1.2",
            cast_type=Decimal,
            color=Fore.YELLOW,
        )  # Bid/Ask ratio >= this for long
        self.order_book_ratio_threshold_short: Decimal = self._get_env(
            "ORDER_BOOK_RATIO_THRESHOLD_SHORT",
            "0.8",
            cast_type=Decimal,
            color=Fore.YELLOW,
        )  # Bid/Ask ratio <= this for short
        self.fetch_order_book_per_cycle: bool = self._get_env(
            "FETCH_ORDER_BOOK_PER_CYCLE", "false", cast_type=bool, color=Fore.YELLOW
        )  # If false, fetch only on potential entry

        # --- ATR Calculation (for Initial SL) ---
        self.atr_calculation_period: int = self._get_env(
            "ATR_CALCULATION_PERIOD", 14, cast_type=int, color=Fore.GREEN
        )

        # --- Termux SMS Alerts ---
        self.enable_sms_alerts: bool = self._get_env(
            "ENABLE_SMS_ALERTS", "false", cast_type=bool, color=Fore.MAGENTA
        )
        self.sms_recipient_number: Optional[str] = self._get_env(
            "SMS_RECIPIENT_NUMBER", None, color=Fore.MAGENTA
        )
        self.sms_timeout_seconds: int = self._get_env(
            "SMS_TIMEOUT_SECONDS", 30, cast_type=int, color=Fore.MAGENTA
        )

        # --- CCXT / API Parameters ---
        self.default_recv_window: int = (
            10000  # milliseconds (adjust if needed for API timing issues)
        )
        self.order_book_fetch_limit: int = max(
            25, self.order_book_depth
        )  # Min limit often 25 for L2, ensure we fetch enough
        self.shallow_ob_fetch_depth: int = 5  # For quick price estimate
        self.order_fill_timeout_seconds: int = self._get_env(
            "ORDER_FILL_TIMEOUT_SECONDS", 15, cast_type=int, color=Fore.YELLOW
        )  # Wait time for market order fill confirmation

        # --- Internal Constants ---
        self.side_buy: str = "buy"
        self.side_sell: str = "sell"
        self.pos_long: str = "Long"
        self.pos_short: str = "Short"
        self.pos_none: str = "None"
        self.usdt_symbol: str = "USDT"
        self.retry_count: int = 3
        self.retry_delay_seconds: int = 2
        self.api_fetch_limit_buffer: int = (
            10  # Extra candles to fetch beyond indicator needs
        )
        self.position_qty_epsilon: Decimal = Decimal(
            "1e-9"
        )  # Small value to check against zero qty, avoid floating point issues
        self.post_close_delay_seconds: int = (
            3  # Pause after closing before allowing new entry
        )
        self.market_order_fill_check_interval: float = (
            0.5  # Seconds between checks for fill
        )

        logger.info(
            f"{Fore.MAGENTA}--- Configuration Runes Summoned Successfully ---{Style.RESET_ALL}"
        )

    def _get_env(
        self,
        key: str,
        default: Any = None,
        cast_type: type = str,
        required: bool = False,
        color: str = Fore.WHITE,
    ) -> Any:
        """Fetches environment variable, casts type, logs, handles defaults/errors."""
        value = os.getenv(key)
        log_value = (
            f"'{value}'"
            if value is not None
            else f"Not Set (Using Default: '{default}')"
        )
        logger.debug(f"{color}Summoning {key}: {log_value}{Style.RESET_ALL}")

        if value is None:
            if required:
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}CRITICAL: Required environment variable '{key}' not set.{Style.RESET_ALL}"
                )
                raise ValueError(f"Required environment variable '{key}' not set.")
            value = default
        elif cast_type == bool:
            value = value.lower() in ["true", "1", "yes", "y"]
        elif cast_type == Decimal:
            try:
                value = Decimal(value)
            except InvalidOperation:
                logger.error(
                    f"{Fore.RED}Invalid Decimal value for {key}: '{value}'. Using default: '{default}'{Style.RESET_ALL}"
                )
                # Attempt to use default, ensure it's convertible
                try:
                    value = Decimal(str(default)) if default is not None else None
                except InvalidOperation:
                    logger.error(
                        f"{Fore.RED}Default value '{default}' for {key} is also invalid for Decimal.{Style.RESET_ALL}"
                    )
                    value = None

                if required and value is None:
                    raise ValueError(
                        f"Required Decimal env var '{key}' had invalid value and no valid default."
                    )
        elif cast_type is not None:
            try:
                value = cast_type(value)
            except (ValueError, TypeError):
                logger.error(
                    f"{Fore.RED}Invalid type for {key}: '{value}'. Expected {cast_type.__name__}. Using default: '{default}'{Style.RESET_ALL}"
                )
                value = default

        # Final check if required value ended up as None (e.g., if default was None)
        if value is None and required:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}CRITICAL: Required environment variable '{key}' has no value or valid default.{Style.RESET_ALL}"
            )
            raise ValueError(
                f"Required environment variable '{key}' has no value or valid default."
            )

        return value


# --- Global Objects ---
try:
    CONFIG = Config()
except ValueError as e:
    logger.critical(
        f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Configuration Error: {e}{Style.RESET_ALL}"
    )
    sys.exit(1)


# --- Helper Functions ---
def safe_decimal_conversion(value: Any, default: Decimal = Decimal("0.0")) -> Decimal:
    """Safely converts a value to Decimal, returning default if conversion fails or value is None/NaN."""
    if value is None or pd.isna(value):  # Handle potential pandas NaN
        return default
    try:
        # Convert potential floats or other numeric types to string first for precise Decimal conversion
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        logger.warning(
            f"Could not convert '{value}' (type: {type(value).__name__}) to Decimal, using default {default}"
        )
        return default


def format_order_id(order_id: Optional[str | int]) -> str:
    """Returns the last 6 characters of an order ID or 'N/A'."""
    return f"...{str(order_id)[-6:]}" if order_id else "N/A"


def get_market_base_currency(symbol: str) -> str:
    """Extracts the base currency from a symbol like 'BTC/USDT:USDT'."""
    try:
        # Handle both 'BTC/USDT' and 'BTC/USDT:USDT' formats
        return symbol.split(":")[0].split("/")[0]
    except IndexError:
        logger.warning(f"Could not parse base currency from symbol: {symbol}")
        return symbol  # Fallback


# --- Precision Formatting ---
def format_price(
    exchange: ccxt.Exchange, symbol: str, price: float | Decimal | str
) -> str:
    """Formats price according to market precision rules, returning string."""
    try:
        # Convert Decimal to float for ccxt, handle potential exceptions during conversion
        price_decimal = safe_decimal_conversion(price)  # Ensure it's Decimal first
        price_float = float(price_decimal)
        return exchange.price_to_precision(symbol, price_float)
    except (ValueError, TypeError, ccxt.ExchangeError, Exception) as e:
        logger.error(
            f"{Fore.RED}Error formatting price '{price}' for {symbol}: {e}{Style.RESET_ALL}"
        )
        # Fallback: attempt to return Decimal as string, else original string
        if isinstance(price, Decimal):
            return str(price.normalize())  # Remove trailing zeros
        return str(price)


def format_amount(
    exchange: ccxt.Exchange, symbol: str, amount: float | Decimal | str
) -> str:
    """Formats amount according to market precision rules, returning string.
    Crucially, uses ROUND_DOWN for amounts to avoid ordering more than intended/possible.
    """
    try:
        market = exchange.market(symbol)
        precision_amount = market.get("precision", {}).get("amount")
        amount_decimal = safe_decimal_conversion(amount)  # Ensure Decimal

        if precision_amount is not None:
            # Use amount_to_precision if it respects rounding mode, otherwise quantize manually
            # Note: CCXT's amount_to_precision *might* not always round down, so manual quantization is safer
            decimal_places = (
                Decimal(str(precision_amount)).normalize().as_tuple().exponent * -1
            )
            quantizer = Decimal("1e-" + str(decimal_places))
            rounded_amount = amount_decimal.quantize(quantizer, rounding=ROUND_DOWN)
            return (
                f"{rounded_amount:.{decimal_places}f}"  # Format to fixed decimal places
            )
        else:
            # Fallback if precision info is missing (less ideal)
            logger.warning(
                f"Amount precision not found for {symbol}, using default formatting."
            )
            amount_float = float(amount_decimal)
            # Use CCXT's method as a fallback, hoping it does something reasonable
            return exchange.amount_to_precision(symbol, amount_float)

    except (
        ValueError,
        TypeError,
        ccxt.ExchangeError,
        InvalidOperation,
        Exception,
    ) as e:
        logger.error(
            f"{Fore.RED}Error formatting amount '{amount}' for {symbol}: {e}{Style.RESET_ALL}"
        )
        # Fallback: attempt to return Decimal as string (normalized), else original string
        if isinstance(amount, Decimal):
            return str(amount.normalize())
        return str(amount)


# --- Termux SMS Alert Function ---
def send_sms_alert(message: str) -> bool:
    """Sends an SMS alert using Termux API if enabled."""
    if not CONFIG.enable_sms_alerts:
        logger.debug("SMS alerts disabled.")
        return False
    if not CONFIG.sms_recipient_number:
        logger.warning("SMS alerts enabled, but SMS_RECIPIENT_NUMBER not set.")
        return False

    # Sanitize message slightly (optional, Termux usually handles it)
    safe_message = message.replace('"', "'").replace(
        "`", "'"
    )  # Basic quote replacement

    try:
        command: List[str] = [
            "termux-sms-send",
            "-n",
            CONFIG.sms_recipient_number,
            safe_message,
        ]
        logger.info(
            f"{Fore.MAGENTA}Attempting SMS to {CONFIG.sms_recipient_number} (Timeout: {CONFIG.sms_timeout_seconds}s)...{Style.RESET_ALL}"
        )
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=CONFIG.sms_timeout_seconds,
        )

        if result.returncode == 0:
            logger.success(
                f"{Fore.MAGENTA}SMS command executed successfully.{Style.RESET_ALL}"
            )
            return True
        else:
            # Log stderr if available for debugging
            stderr_output = (
                result.stderr.strip() if result.stderr else "No stderr output"
            )
            logger.error(
                f"{Fore.RED}SMS command failed. RC: {result.returncode}, Stderr: {stderr_output}{Style.RESET_ALL}"
            )
            return False
    except FileNotFoundError:
        logger.error(
            f"{Fore.RED}SMS failed: 'termux-sms-send' command not found. Ensure Termux:API package and app are installed and configured.{Style.RESET_ALL}"
        )
        return False
    except subprocess.TimeoutExpired:
        logger.error(
            f"{Fore.RED}SMS failed: command timed out after {CONFIG.sms_timeout_seconds}s.{Style.RESET_ALL}"
        )
        return False
    except Exception as e:
        logger.error(f"{Fore.RED}SMS failed: Unexpected error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        return False


# --- Exchange Initialization ---
def initialize_exchange() -> Optional[ccxt.Exchange]:
    """Initializes and returns the CCXT Bybit exchange instance."""
    logger.info(f"{Fore.BLUE}Initializing CCXT Bybit connection...{Style.RESET_ALL}")
    if not CONFIG.api_key or not CONFIG.api_secret:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}CRITICAL: API keys (BYBIT_API_KEY, BYBIT_API_SECRET) missing in environment.{Style.RESET_ALL}"
        )
        send_sms_alert("[Pyrmethus] CRITICAL: API keys missing. Bot stopped.")
        return None
    try:
        exchange = ccxt.bybit(
            {
                "apiKey": CONFIG.api_key,
                "secret": CONFIG.api_secret,
                "enableRateLimit": True,  # Enable CCXT's built-in rate limiter
                "options": {
                    "defaultType": "linear",  # Explicitly set for USDT perpetuals (Bybit V5)
                    "adjustForTimeDifference": True,  # Auto-sync time with server
                    "recvWindow": CONFIG.default_recv_window,
                    "verbose": LOGGING_LEVEL
                    == logging.DEBUG,  # Enable verbose CCXT logging if bot is in debug mode
                    # V5 specific options if needed, e.g., 'code', check CCXT docs
                },
            }
        )
        # exchange.set_sandbox_mode(True) # Uncomment for testnet usage

        logger.debug("Loading markets (forced update)...")
        exchange.load_markets(True)  # Force reload to get latest info, symbols, limits

        logger.debug("Performing initial balance check for authentication...")
        exchange.fetch_balance()  # Initial connectivity and authentication check

        logger.success(
            f"{Fore.GREEN}{Style.BRIGHT}CCXT Bybit Session Initialized (Targeting V5 API - LIVE SCALPING MODE - EXTREME CAUTION!).{Style.RESET_ALL}"
        )
        send_sms_alert("[Pyrmethus] Initialized & authenticated successfully.")
        return exchange

    except ccxt.AuthenticationError as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Authentication failed: {e}. Check API keys, IP whitelist, and permissions.{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[Pyrmethus] CRITICAL: Authentication FAILED: {e}. Bot stopped."
        )
    except ccxt.NetworkError as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Network error during initialization: {e}. Check connection and Bybit status.{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[Pyrmethus] CRITICAL: Network Error on Init: {e}. Bot stopped."
        )
    except ccxt.ExchangeError as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Exchange error during initialization: {e}. Check Bybit status and API documentation.{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[Pyrmethus] CRITICAL: Exchange Error on Init: {e}. Bot stopped."
        )
    except Exception as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Unexpected error during exchange initialization: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        send_sms_alert(
            f"[Pyrmethus] CRITICAL: Unexpected Init Error: {type(e).__name__}. Bot stopped."
        )
    return None


# --- Indicator Calculation Functions ---
# Note: These functions now return the modified DataFrame and handle NA values more explicitly.


def calculate_supertrend(
    df: pd.DataFrame, length: int, multiplier: Decimal, prefix: str = ""
) -> pd.DataFrame:
    """Calculates the Supertrend indicator using pandas_ta, returns Decimal where applicable."""
    col_prefix = f"{prefix}" if prefix else ""
    # Define consistent internal column names
    trend_col = f"{col_prefix}trend"  # Boolean: True for uptrend, False for downtrend
    long_flip_col = (
        f"{col_prefix}st_long"  # Boolean: True if trend flipped to Long on this candle
    )
    short_flip_col = f"{col_prefix}st_short"  # Boolean: True if trend flipped to Short on this candle
    value_col = f"{col_prefix}supertrend"  # Decimal: The Supertrend line value
    target_cols = [value_col, trend_col, long_flip_col, short_flip_col]

    # pandas_ta column names (may vary slightly based on version)
    pt_st_col = f"SUPERT_{length}_{float(multiplier)}"
    pt_trend_col = f"SUPERTd_{length}_{float(multiplier)}"
    pt_long_col = (
        f"SUPERTl_{length}_{float(multiplier)}"  # Often represents the lower band value
    )
    pt_short_col = (
        f"SUPERTs_{length}_{float(multiplier)}"  # Often represents the upper band value
    )
    raw_pt_cols = [pt_st_col, pt_trend_col, pt_long_col, pt_short_col]

    required_input_cols = ["high", "low", "close"]
    min_len = length + 1  # Need at least 'length' periods + current

    # Initialize target columns with NA
    for col in target_cols:
        df[col] = pd.NA

    if (
        df is None
        or df.empty
        or not all(c in df.columns for c in required_input_cols)
        or len(df) < min_len
    ):
        logger.warning(
            f"{Fore.YELLOW}Indicator Calc ({col_prefix}ST): Insufficient data (Len: {len(df) if df is not None else 0}, Need: {min_len}). Setting cols to NA.{Style.RESET_ALL}"
        )
        return df

    try:
        # Calculate using pandas_ta (expects float multiplier)
        df.ta.supertrend(length=length, multiplier=float(multiplier), append=True)

        # Check if expected columns were created
        if not all(c in df.columns for c in [pt_st_col, pt_trend_col]):
            missing = [c for c in [pt_st_col, pt_trend_col] if c not in df.columns]
            raise KeyError(
                f"pandas_ta failed to create expected raw columns: {missing}"
            )

        # --- Populate our target columns ---
        # Supertrend value (convert to Decimal)
        df[value_col] = df[pt_st_col].apply(safe_decimal_conversion)

        # Trend direction (convert 1/-1 to boolean)
        df[trend_col] = df[pt_trend_col].apply(
            lambda x: True
            if pd.notna(x) and x == 1
            else (False if pd.notna(x) and x == -1 else pd.NA)
        )

        # Trend flips (compare current trend with previous candle's trend)
        prev_trend = df[trend_col].shift(1)
        df[long_flip_col] = (not prev_trend) & (df[trend_col])
        df[short_flip_col] = (prev_trend) & (not df[trend_col])

        # Clean up raw pandas_ta columns
        df.drop(columns=raw_pt_cols, errors="ignore", inplace=True)

        # Log last values for debugging
        last_st_val = df[value_col].iloc[-1]
        if pd.notna(last_st_val):
            last_trend_val = df[trend_col].iloc[-1]
            last_trend_str = (
                "Up"
                if last_trend_val is True
                else ("Down" if last_trend_val is False else "NA")
            )
            signal = (
                "LONG_FLIP"
                if df[long_flip_col].iloc[-1]
                else ("SHORT_FLIP" if df[short_flip_col].iloc[-1] else "No Flip")
            )
            logger.debug(
                f"Indicator Calc ({col_prefix}ST({length},{multiplier})): Trend={last_trend_str}, Val={last_st_val:.4f}, Signal={signal}"
            )
        else:
            logger.debug(
                f"Indicator Calc ({col_prefix}ST({length},{multiplier})): Resulted in NA for last candle."
            )

    except Exception as e:
        logger.error(
            f"{Fore.RED}Indicator Calc ({col_prefix}ST): Error: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        # Ensure target columns are reset to NA on error
        for col in target_cols:
            df[col] = pd.NA
    return df


def analyze_volume_atr(
    df: pd.DataFrame, atr_len: int, vol_ma_len: int
) -> Tuple[pd.DataFrame, Dict[str, Optional[Decimal]]]:
    """Calculates ATR, Volume MA, checks spikes. Returns modified DF and results Dict."""
    results: Dict[str, Optional[Decimal]] = {
        "atr": None,
        "volume_ma": None,
        "last_volume": None,
        "volume_ratio": None,
    }
    required_cols = ["high", "low", "close", "volume"]
    min_len = max(atr_len, vol_ma_len) + 1  # Need +1 for calculations like rolling MA

    if (
        df is None
        or df.empty
        or not all(c in df.columns for c in required_cols)
        or len(df) < min_len
    ):
        logger.warning(
            f"{Fore.YELLOW}Indicator Calc (Vol/ATR): Insufficient data (Len: {len(df) if df is not None else 0}, Need: {min_len}). Returning N/A.{Style.RESET_ALL}"
        )
        return df, results

    try:
        # Calculate ATR using pandas_ta
        atr_col = f"ATRr_{atr_len}"  # pandas_ta default ATR column name
        df.ta.atr(length=atr_len, append=True)
        if atr_col in df.columns:
            last_atr = df[atr_col].iloc[-1]
            if pd.notna(last_atr):
                results["atr"] = safe_decimal_conversion(last_atr)
            df.drop(
                columns=[atr_col], errors="ignore", inplace=True
            )  # Clean up raw column
        else:
            logger.warning(f"ATR column '{atr_col}' not found after calculation.")

        # Calculate Volume MA using pandas rolling mean
        volume_ma_col = f"volume_ma_{vol_ma_len}"
        # Ensure enough periods for calculation, ignore initial NaNs
        df[volume_ma_col] = (
            df["volume"]
            .rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2))
            .mean()
        )
        last_vol_ma = df[volume_ma_col].iloc[-1]
        last_vol = df["volume"].iloc[-1]  # Already numeric from get_market_data

        if pd.notna(last_vol_ma):
            results["volume_ma"] = safe_decimal_conversion(last_vol_ma)
        if pd.notna(last_vol):
            results["last_volume"] = safe_decimal_conversion(last_vol)

        # Calculate volume ratio safely using Decimal
        if (
            results["volume_ma"] is not None
            and results["volume_ma"] > CONFIG.position_qty_epsilon
            and results["last_volume"] is not None
        ):
            try:
                results["volume_ratio"] = results["last_volume"] / results["volume_ma"]
            except (InvalidOperation, ZeroDivisionError) as ratio_err:
                logger.warning(f"Error calculating volume ratio: {ratio_err}")
                results["volume_ratio"] = None
        else:
            results["volume_ratio"] = (
                None  # Cannot calculate ratio if MA is zero/NA or volume is NA
            )

        # Optionally keep the MA column in df or drop it
        # df.drop(columns=[volume_ma_col], errors='ignore', inplace=True) # Uncomment to drop

        # Log results
        atr_str = f"{results['atr']:.5f}" if results["atr"] is not None else "N/A"
        vol_ma_str = (
            f"{results['volume_ma']:.2f}" if results["volume_ma"] is not None else "N/A"
        )
        last_vol_str = (
            f"{results['last_volume']:.2f}"
            if results["last_volume"] is not None
            else "N/A"
        )
        vol_ratio_str = (
            f"{results['volume_ratio']:.2f}"
            if results["volume_ratio"] is not None
            else "N/A"
        )
        logger.debug(
            f"Indicator Calc: ATR({atr_len})={atr_str}, Vol={last_vol_str}, VolMA({vol_ma_len})={vol_ma_str}, VolRatio={vol_ratio_str}"
        )

    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc (Vol/ATR): Error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        results = {key: None for key in results}  # Reset results on error
    return df, results


def calculate_stochrsi_momentum(
    df: pd.DataFrame, rsi_len: int, stoch_len: int, k: int, d: int, mom_len: int
) -> pd.DataFrame:
    """Calculates StochRSI and Momentum, returns modified DF with Decimal values."""
    # Define consistent internal column names
    k_col_internal = "stochrsi_k"
    d_col_internal = "stochrsi_d"
    mom_col_internal = "momentum"
    target_cols = [k_col_internal, d_col_internal, mom_col_internal]

    # pandas_ta column names
    k_col_pt = f"STOCHRSIk_{stoch_len}_{rsi_len}_{k}_{d}"
    d_col_pt = f"STOCHRSId_{stoch_len}_{rsi_len}_{k}_{d}"
    mom_col_pt = f"MOM_{mom_len}"

    required_input_cols = ["close"]
    # Estimate minimum length needed (can be complex for chained indicators)
    min_len = max(rsi_len + stoch_len + k + d, mom_len) + 10  # Add buffer

    # Initialize target columns with NA
    for col in target_cols:
        df[col] = pd.NA

    if (
        df is None
        or df.empty
        or not all(c in df.columns for c in required_input_cols)
        or len(df) < min_len
    ):
        logger.warning(
            f"{Fore.YELLOW}Indicator Calc (StochRSI/Mom): Insufficient data (Len: {len(df) if df is not None else 0}, Need ~{min_len}). Setting cols to NA.{Style.RESET_ALL}"
        )
        return df

    try:
        # StochRSI - Calculate separately to handle potential column name issues
        stochrsi_df = df.ta.stochrsi(
            length=stoch_len, rsi_length=rsi_len, k=k, d=d, append=False
        )
        if k_col_pt in stochrsi_df.columns:
            df[k_col_internal] = stochrsi_df[k_col_pt].apply(safe_decimal_conversion)
        else:
            logger.warning(
                f"StochRSI K column '{k_col_pt}' not found in pandas_ta output."
            )
            df[k_col_internal] = pd.NA
        if d_col_pt in stochrsi_df.columns:
            df[d_col_internal] = stochrsi_df[d_col_pt].apply(safe_decimal_conversion)
        else:
            logger.warning(
                f"StochRSI D column '{d_col_pt}' not found in pandas_ta output."
            )
            df[d_col_internal] = pd.NA

        # Momentum
        df.ta.mom(length=mom_len, append=True)
        if mom_col_pt in df.columns:
            df[mom_col_internal] = df[mom_col_pt].apply(safe_decimal_conversion)
            df.drop(
                columns=[mom_col_pt], errors="ignore", inplace=True
            )  # Clean up raw column
        else:
            logger.warning(
                f"Momentum column '{mom_col_pt}' not found after calculation."
            )
            df[mom_col_internal] = pd.NA

        # Log last values for debugging
        k_val = df[k_col_internal].iloc[-1]
        d_val = df[d_col_internal].iloc[-1]
        mom_val = df[mom_col_internal].iloc[-1]
        k_str = f"{k_val:.2f}" if pd.notna(k_val) else "NA"
        d_str = f"{d_val:.2f}" if pd.notna(d_val) else "NA"
        mom_str = f"{mom_val:.4f}" if pd.notna(mom_val) else "NA"
        logger.debug(
            f"Indicator Calc (StochRSI({rsi_len},{stoch_len},{k},{d})/Mom({mom_len})): K={k_str}, D={d_str}, Mom={mom_str}"
        )

    except Exception as e:
        logger.error(
            f"{Fore.RED}Indicator Calc (StochRSI/Mom): Error: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        for col in target_cols:
            df[col] = pd.NA  # Ensure reset on error
    return df


def calculate_ehlers_fisher(df: pd.DataFrame, length: int, signal: int) -> pd.DataFrame:
    """Calculates Ehlers Fisher Transform, returns modified DF with Decimal values."""
    # Define consistent internal column names
    fish_col_internal = "ehlers_fisher"
    signal_col_internal = "ehlers_signal"
    target_cols = [fish_col_internal, signal_col_internal]

    # pandas_ta column names
    fish_col_pt = f"FISHERT_{length}_{signal}"
    signal_col_pt = f"FISHERTs_{length}_{signal}"

    required_input_cols = ["high", "low"]
    min_len = length + signal + 5  # Conservative buffer

    # Initialize target columns with NA
    for col in target_cols:
        df[col] = pd.NA

    if (
        df is None
        or df.empty
        or not all(c in df.columns for c in required_input_cols)
        or len(df) < min_len
    ):
        logger.warning(
            f"{Fore.YELLOW}Indicator Calc (EhlersFisher): Insufficient data (Len: {len(df) if df is not None else 0}, Need ~{min_len}). Setting cols to NA.{Style.RESET_ALL}"
        )
        return df

    try:
        # Calculate using pandas_ta
        fisher_df = df.ta.fisher(length=length, signal=signal, append=False)

        # Populate internal columns
        if fish_col_pt in fisher_df.columns:
            df[fish_col_internal] = fisher_df[fish_col_pt].apply(
                safe_decimal_conversion
            )
        else:
            logger.warning(
                f"Ehlers Fisher column '{fish_col_pt}' not found in pandas_ta output."
            )
            df[fish_col_internal] = pd.NA

        # Only add signal if length > 0 and column exists
        if signal > 0 and signal_col_pt in fisher_df.columns:
            df[signal_col_internal] = fisher_df[signal_col_pt].apply(
                safe_decimal_conversion
            )
        else:
            df[signal_col_internal] = (
                pd.NA
            )  # Set to NA if signal length is 0 or column missing

        # Log last values for debugging
        fish_val = df[fish_col_internal].iloc[-1]
        sig_val = df[signal_col_internal].iloc[-1]
        fish_str = f"{fish_val:.4f}" if pd.notna(fish_val) else "NA"
        sig_str = (
            f"{sig_val:.4f}"
            if pd.notna(sig_val)
            else ("NA" if signal > 0 else "N/A (SigLen=0)")
        )
        logger.debug(
            f"Indicator Calc (EhlersFisher({length},{signal})): Fisher={fish_str}, Signal={sig_str}"
        )

    except Exception as e:
        logger.error(
            f"{Fore.RED}Indicator Calc (EhlersFisher): Error: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        for col in target_cols:
            df[col] = pd.NA  # Ensure reset on error
    return df


def calculate_ehlers_ma(df: pd.DataFrame, fast_len: int, slow_len: int) -> pd.DataFrame:
    """Placeholder: Calculates EMA instead of Ehlers Super Smoother. Returns modified DF."""
    # Define consistent internal column names
    fast_ma_col = "fast_ema"  # Using EMA as placeholder
    slow_ma_col = "slow_ema"  # Using EMA as placeholder
    target_cols = [fast_ma_col, slow_ma_col]

    required_input_cols = ["close"]
    min_len = max(fast_len, slow_len) + 5  # Add buffer

    # Initialize target columns with NA
    for col in target_cols:
        df[col] = pd.NA

    if (
        df is None
        or df.empty
        or not all(c in df.columns for c in required_input_cols)
        or len(df) < min_len
    ):
        logger.warning(
            f"{Fore.YELLOW}Indicator Calc (EhlersMA - PLACEHOLDER): Insufficient data (Len: {len(df) if df is not None else 0}, Need ~{min_len}). Setting cols to NA.{Style.RESET_ALL}"
        )
        return df

    try:
        # --- PLACEHOLDER WARNING ---
        logger.warning(
            f"{Back.YELLOW}{Fore.BLACK}WARNING: Using standard EMA as placeholder for Ehlers Super Smoother MA. "
            f"This is NOT the true Ehlers implementation. Replace if accurate Ehlers MA is required.{Style.RESET_ALL}"
        )

        # Calculate standard EMA using pandas_ta
        df[fast_ma_col] = df.ta.ema(length=fast_len).apply(safe_decimal_conversion)
        df[slow_ma_col] = df.ta.ema(length=slow_len).apply(safe_decimal_conversion)

        # Log last values for debugging
        fast_val = df[fast_ma_col].iloc[-1]
        slow_val = df[slow_ma_col].iloc[-1]
        fast_str = f"{fast_val:.4f}" if pd.notna(fast_val) else "NA"
        slow_str = f"{slow_val:.4f}" if pd.notna(slow_val) else "NA"
        logger.debug(
            f"Indicator Calc (EhlersMA Placeholder - EMA({fast_len},{slow_len})): Fast={fast_str}, Slow={slow_str}"
        )

    except Exception as e:
        logger.error(
            f"{Fore.RED}Indicator Calc (EhlersMA Placeholder): Error: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        for col in target_cols:
            df[col] = pd.NA  # Ensure reset on error
    return df


def analyze_order_book(
    exchange: ccxt.Exchange, symbol: str, depth: int, fetch_limit: int
) -> Dict[str, Optional[Decimal]]:
    """Fetches and analyzes L2 order book pressure and spread. Returns Decimals."""
    results: Dict[str, Optional[Decimal]] = {
        "bid_ask_ratio": None,
        "spread": None,
        "best_bid": None,
        "best_ask": None,
    }
    logger.debug(
        f"Order Book: Fetching L2 {symbol} (Depth:{depth}, Limit:{fetch_limit})..."
    )

    if not exchange.has.get("fetchL2OrderBook"):
        logger.warning(
            f"{Fore.YELLOW}Order Book: fetchL2OrderBook not supported by {exchange.id}. Cannot analyze.{Style.RESET_ALL}"
        )
        return results
    try:
        order_book = exchange.fetch_l2_order_book(symbol, limit=fetch_limit)
        bids: List[List[float | str]] = order_book.get("bids", [])
        asks: List[List[float | str]] = order_book.get("asks", [])

        if not bids or not asks:
            logger.warning(
                f"{Fore.YELLOW}Order Book: Empty bids or asks returned for {symbol}. OB analysis skipped.{Style.RESET_ALL}"
            )
            return results

        # Get best bid/ask safely
        best_bid = (
            safe_decimal_conversion(bids[0][0])
            if len(bids) > 0 and len(bids[0]) > 0
            else None
        )
        best_ask = (
            safe_decimal_conversion(asks[0][0])
            if len(asks) > 0 and len(asks[0]) > 0
            else None
        )
        results["best_bid"] = best_bid
        results["best_ask"] = best_ask

        # Calculate spread
        if (
            best_bid is not None
            and best_ask is not None
            and best_bid > 0
            and best_ask > 0
        ):
            results["spread"] = best_ask - best_bid
            logger.debug(
                f"OB: Best Bid={best_bid:.4f}, Best Ask={best_ask:.4f}, Spread={results['spread']:.4f}"
            )
        else:
            results["spread"] = None  # Cannot calculate if bid/ask invalid
            logger.debug(
                f"OB: Best Bid={best_bid or 'N/A'}, Best Ask={best_ask or 'N/A'} (Spread N/A)"
            )

        # Sum volumes within the specified depth using Decimal for precision
        # Ensure list elements have at least 2 items (price, volume)
        bid_vol = sum(
            safe_decimal_conversion(bid[1]) for bid in bids[:depth] if len(bid) > 1
        )
        ask_vol = sum(
            safe_decimal_conversion(ask[1]) for ask in asks[:depth] if len(ask) > 1
        )
        logger.debug(
            f"OB (Depth {depth}): Total BidVol={bid_vol:.4f}, Total AskVol={ask_vol:.4f}"
        )

        # Calculate ratio safely, avoid division by zero
        if ask_vol > CONFIG.position_qty_epsilon:  # Use epsilon for safety
            try:
                results["bid_ask_ratio"] = bid_vol / ask_vol
                logger.debug(f"OB Ratio (Bid/Ask): {results['bid_ask_ratio']:.3f}")
            except (InvalidOperation, ZeroDivisionError) as ratio_err:
                logger.warning(f"Error calculating OB ratio: {ratio_err}")
                results["bid_ask_ratio"] = None
        else:
            logger.debug("OB Ratio: N/A (Ask volume zero or negligible)")
            results["bid_ask_ratio"] = None  # Explicitly set to None

    except (ccxt.NetworkError, ccxt.ExchangeError, IndexError, Exception) as e:
        logger.warning(
            f"{Fore.YELLOW}Order Book Analysis Error for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        results = {key: None for key in results}  # Reset results on error
    return results


# --- Data Fetching ---
def get_market_data(
    exchange: ccxt.Exchange, symbol: str, interval: str, limit: int
) -> Optional[pd.DataFrame]:
    """Fetches and prepares OHLCV data, ensuring numeric types and handling NaNs robustly."""
    if not exchange.has.get("fetchOHLCV"):
        logger.error(
            f"{Fore.RED}Data Fetch: Exchange '{exchange.id}' does not support fetchOHLCV.{Style.RESET_ALL}"
        )
        return None
    try:
        logger.debug(
            f"Data Fetch: Fetching {limit} OHLCV candles for {symbol} ({interval})..."
        )
        # FetchOHLCV params: symbol, timeframe, since=None, limit=None, params={}
        ohlcv: List[List[int | float | str]] = exchange.fetch_ohlcv(
            symbol, timeframe=interval, limit=limit
        )

        if not ohlcv:
            logger.warning(
                f"{Fore.YELLOW}Data Fetch: No OHLCV data returned for {symbol} ({interval}). Could be API issue, insufficient history, or incorrect symbol/interval.{Style.RESET_ALL}"
            )
            return None

        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)

        # Convert to numeric, coercing errors to NaN (will be handled below)
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # --- Robust NaN Handling ---
        if df.isnull().values.any():
            initial_nan_counts = df.isnull().sum()
            logger.warning(
                f"{Fore.YELLOW}Data Fetch: OHLCV contains NaNs after initial conversion:\n{initial_nan_counts[initial_nan_counts > 0]}\nAttempting forward fill...{Style.RESET_ALL}"
            )
            df.ffill(
                inplace=True
            )  # Forward fill first (carries last valid observation forward)

            if df.isnull().values.any():  # Check again after ffill
                remaining_nan_counts = df.isnull().sum()
                logger.warning(
                    f"{Fore.YELLOW}NaNs remain after ffill:\n{remaining_nan_counts[remaining_nan_counts > 0]}\nAttempting backward fill...{Style.RESET_ALL}"
                )
                df.bfill(
                    inplace=True
                )  # Back fill if ffill wasn't enough (e.g., NaNs at the start)

                if df.isnull().values.any():  # Final check after bfill
                    final_nan_counts = df.isnull().sum()
                    logger.error(
                        f"{Fore.RED}Data Fetch: Unfillable NaNs persist after ffill and bfill:\n{final_nan_counts[final_nan_counts > 0]}\nCannot reliably use this data. Discarding OHLCV fetch.{Style.RESET_ALL}"
                    )
                    return None  # Unrecoverable NaNs - data quality compromised

        # Final check for valid data types (ensure they are numeric, not object etc.)
        if not all(pd.api.types.is_numeric_dtype(df[col]) for col in numeric_cols):
            logger.error(
                f"{Fore.RED}Data Fetch: Non-numeric data detected in OHLCV columns AFTER processing. Discarding fetch.{Style.RESET_ALL}"
            )
            return None

        # Check if enough data was returned (sometimes API returns fewer than requested)
        if (
            len(df) < limit // 2
        ):  # Heuristic check for significantly less data than requested
            logger.warning(
                f"{Fore.YELLOW}Data Fetch: Received significantly fewer candles ({len(df)}) than requested ({limit}). Potential API issue or lack of history.{Style.RESET_ALL}"
            )
            # Decide if this is critical - for now, proceed if some data exists

        logger.debug(
            f"Data Fetch: Successfully processed {len(df)} OHLCV candles for {symbol}. Last candle: {df.index[-1]}"
        )
        return df

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger.warning(
            f"{Fore.YELLOW}Data Fetch: CCXT Error fetching OHLCV for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
    except Exception as e:
        logger.error(
            f"{Fore.RED}Data Fetch: Unexpected error processing OHLCV for {symbol}: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
    return None


# --- Position & Order Management ---


def get_current_position(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    """Fetches current position details using Bybit V5 API specifics via CCXT.
    Targets One-Way mode (positionIdx=0).
    Returns a dictionary: {'side': Config.pos_long/pos_short/pos_none, 'qty': Decimal, 'entry_price': Decimal}.
    """
    default_pos: Dict[str, Any] = {
        "side": CONFIG.pos_none,
        "qty": Decimal("0.0"),
        "entry_price": Decimal("0.0"),
    }
    market: Optional[Dict] = None
    market_id: Optional[str] = None

    try:
        market = exchange.market(symbol)
        market_id = market["id"]  # Get the exchange-specific market ID (e.g., BTCUSDT)
        if not market or not market_id:
            raise ValueError(f"Market info or ID not found in CCXT for '{symbol}'.")
    except (ccxt.BadSymbol, KeyError, ValueError, Exception) as e:
        logger.error(
            f"{Fore.RED}Position Check: Failed to get market info for '{symbol}': {e}{Style.RESET_ALL}"
        )
        return default_pos

    try:
        if not exchange.has.get("fetchPositions"):
            logger.warning(
                f"{Fore.YELLOW}Position Check: Exchange capability 'fetchPositions' not reported by CCXT for {exchange.id}. Assuming no position.{Style.RESET_ALL}"
            )
            # This might require alternative methods or indicate an issue with CCXT/exchange setup.
            return default_pos

        # Bybit V5 fetchPositions requires 'category' parameter: 'linear' or 'inverse'
        # Determine category based on market info (linear is common for USDT perps)
        category = (
            "linear" if market.get("linear", True) else "inverse"
        )  # Default to linear if unsure
        params = {"category": category}
        logger.debug(
            f"Position Check: Fetching positions for {symbol} (MarketID: {market_id}) with params: {params}"
        )

        # Fetch positions for the specific symbol
        fetched_positions = exchange.fetch_positions(symbols=[symbol], params=params)

        # Filter for the specific symbol and One-Way mode (positionIdx=0)
        active_pos_data = None
        for pos in fetched_positions:
            # Access V5 specific info nested within the 'info' dict provided by CCXT
            pos_info = pos.get("info", {})
            pos_market_id = pos_info.get("symbol")
            # V5 uses positionIdx: 0 for One-Way, 1 for Buy Hedge, 2 for Sell Hedge
            position_idx = int(
                pos_info.get("positionIdx", -1)
            )  # Default to -1 if missing
            # V5 side: 'Buy', 'Sell', or 'None' for flat positions
            pos_side_v5 = (
                str(pos_info.get("side", "None")).strip().capitalize()
            )  # Normalize side
            size_str = pos_info.get("size")  # Position size as string

            logger.debug(
                f"Processing fetched pos: MarketID={pos_market_id}, Idx={position_idx}, SideV5={pos_side_v5}, Size={size_str}"
            )

            # --- Strict Matching Logic ---
            # 1. Match the exact market ID.
            # 2. Ensure it's One-Way mode (positionIdx == 0).
            # 3. Check if the side indicates an actual position ('Buy' or 'Sell').
            # 4. Verify the size is non-zero after conversion.
            if (
                pos_market_id == market_id
                and position_idx == 0
                and pos_side_v5 in ["Buy", "Sell"]
            ):
                size = safe_decimal_conversion(size_str)
                # Use epsilon comparison for floating point inaccuracies
                if abs(size) > CONFIG.position_qty_epsilon:
                    active_pos_data = pos_info  # Store the raw V5 info dict
                    logger.debug(
                        f"Found potential active One-Way position entry: {active_pos_data}"
                    )
                    break  # Assume only one active position in One-Way mode for the symbol

        # --- Parse the Found Active Position ---
        if active_pos_data:
            try:
                size = safe_decimal_conversion(active_pos_data.get("size"))
                # Use 'avgPrice' from V5 info dict for entry price
                entry_price = safe_decimal_conversion(active_pos_data.get("avgPrice"))
                pos_side_v5 = str(active_pos_data.get("side")).capitalize()

                # Convert V5 side ('Buy'/'Sell') to internal representation ('Long'/'Short')
                side = (
                    CONFIG.pos_long
                    if pos_side_v5 == "Buy"
                    else (
                        CONFIG.pos_short if pos_side_v5 == "Sell" else CONFIG.pos_none
                    )
                )

                # Final validation of parsed values
                if (
                    side != CONFIG.pos_none
                    and abs(size) > CONFIG.position_qty_epsilon
                    and entry_price >= Decimal("0")
                ):  # Entry price shouldn't be negative
                    position_details = {
                        "side": side,
                        "qty": abs(size),
                        "entry_price": entry_price,
                    }
                    logger.info(
                        f"{Fore.YELLOW}Position Check: Found ACTIVE {side} position: Qty={position_details['qty']:.8f} @ Entry={position_details['entry_price']:.4f}{Style.RESET_ALL}"
                    )
                    return position_details
                else:
                    # Log if parsing resulted in invalid state (e.g., zero size but side='Buy')
                    logger.warning(
                        f"{Fore.YELLOW}Position Check: Found position data but parsed as invalid/flat (Side:{side}, Qty:{size}, Entry:{entry_price}). Treating as flat.{Style.RESET_ALL}"
                    )
                    return default_pos

            except Exception as parse_err:
                logger.warning(
                    f"{Fore.YELLOW}Position Check: Error parsing active position data: {parse_err}. Raw data: {active_pos_data}{Style.RESET_ALL}"
                )
                return default_pos  # Return default on parsing error
        else:
            # No position matched the strict criteria
            logger.info(
                f"Position Check: No active One-Way (Idx=0) position found for {market_id}."
            )
            return default_pos

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        # Handle API errors during fetch
        logger.warning(
            f"{Fore.YELLOW}Position Check: CCXT Error fetching positions for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
    except Exception as e:
        # Handle unexpected errors
        logger.error(
            f"{Fore.RED}Position Check: Unexpected error fetching/processing positions for {symbol}: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
    return default_pos  # Return default if any error occurs


def set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int) -> bool:
    """Sets leverage for a futures symbol using Bybit V5 specifics via CCXT."""
    logger.info(
        f"{Fore.CYAN}Leverage Setting: Attempting to set {leverage}x for {symbol}...{Style.RESET_ALL}"
    )
    market: Optional[Dict] = None
    try:
        market = exchange.market(symbol)
        if not market or not market.get("contract"):
            logger.error(
                f"{Fore.RED}Leverage Setting: Cannot set leverage for non-contract/non-futures market: {symbol}.{Style.RESET_ALL}"
            )
            return False
    except (ccxt.BadSymbol, KeyError, Exception) as e:
        logger.error(
            f"{Fore.RED}Leverage Setting: Failed to get market info for '{symbol}': {e}{Style.RESET_ALL}"
        )
        return False

    # Determine category (needed for some V5 operations, though set_leverage might handle it)
    category = "linear" if market.get("linear", True) else "inverse"

    for attempt in range(CONFIG.retry_count):
        try:
            # Bybit V5 'setLeverage' potentially needs 'buyLeverage' and 'sellLeverage' in params for unified margin.
            # CCXT `set_leverage` usually handles this abstraction for V5.
            # Providing explicit params as a fallback/safety measure.
            params = {
                "buyLeverage": str(leverage),
                "sellLeverage": str(leverage),
                "category": category,  # Explicitly pass category if required by underlying API call
            }
            # The `leverage` argument to ccxt's `set_leverage` is the primary way.
            response = exchange.set_leverage(
                leverage=leverage, symbol=symbol, params=params
            )

            # Response parsing can be tricky and varies. Often just checking for exceptions is enough.
            # Log a sample of the response for debugging if needed.
            logger.success(
                f"{Fore.GREEN}Leverage Setting: Successfully set to {leverage}x for {symbol}. Response sample: {str(response)[:100]}...{Style.RESET_ALL}"
            )
            return True
        except ccxt.ExchangeError as e:
            err_str = str(e).lower()
            # Bybit V5 specific error code for "leverage not modified" (check Bybit API docs for exact codes)
            # Common code: 110044 - leverage not modified
            if (
                "leverage not modified" in err_str
                or "leverage is same as requested" in err_str
                or "110044" in err_str
            ):
                logger.info(
                    f"{Fore.CYAN}Leverage Setting: Leverage already set to {leverage}x for {symbol}. No change needed.{Style.RESET_ALL}"
                )
                return True
            # Log other exchange errors
            logger.warning(
                f"{Fore.YELLOW}Leverage Setting: Exchange error (Attempt {attempt + 1}/{CONFIG.retry_count}): {e}{Style.RESET_ALL}"
            )
        except (ccxt.NetworkError, Exception) as e:
            # Log network or other unexpected errors
            logger.warning(
                f"{Fore.YELLOW}Leverage Setting: Network/Other error (Attempt {attempt + 1}/{CONFIG.retry_count}): {e}{Style.RESET_ALL}"
            )

        # Wait before retrying
        if attempt < CONFIG.retry_count - 1:
            logger.debug(
                f"Retrying leverage setting in {CONFIG.retry_delay_seconds}s..."
            )
            time.sleep(CONFIG.retry_delay_seconds)
        else:
            logger.error(
                f"{Fore.RED}Leverage Setting: Failed to set leverage for {symbol} to {leverage}x after {CONFIG.retry_count} attempts.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{get_market_base_currency(symbol)}] CRITICAL: Failed to set leverage to {leverage}x for {symbol}!"
            )

    return False  # Failed after all retries


def close_position(
    exchange: ccxt.Exchange,
    symbol: str,
    position_to_close: Dict[str, Any],
    reason: str = "Signal",
) -> Optional[Dict[str, Any]]:
    """Closes the specified active position with re-validation using a market order.
    Uses Decimal for quantity, handles precision, and Bybit V5 `reduceOnly`.
    Returns the executed order dict on success, None on failure or if no position exists.
    """
    initial_side = position_to_close.get("side", CONFIG.pos_none)
    initial_qty = position_to_close.get("qty", Decimal("0.0"))
    market_base = get_market_base_currency(symbol)
    logger.warning(
        f"{Back.YELLOW}{Fore.BLACK}Close Position: Initiated for {symbol}. Reason: {reason}. Expected state: {initial_side} Qty={initial_qty:.8f}{Style.RESET_ALL}"
    )

    # --- Re-validate the position just before closing ---
    live_position = get_current_position(exchange, symbol)
    if live_position["side"] == CONFIG.pos_none:
        logger.warning(
            f"{Fore.YELLOW}Close Position: Re-validation shows NO active position for {symbol}. Aborting closure attempt.{Style.RESET_ALL}"
        )
        # If the bot thought there was a position, log the discrepancy
        if initial_side != CONFIG.pos_none:
            logger.info(
                f"{Fore.CYAN}Close Position: Discrepancy detected (Bot expected {initial_side}, exchange reports None). State likely already updated.{Style.RESET_ALL}"
            )
        return None  # Nothing to close

    # Use the live quantity and side for the closing order
    live_amount_to_close = live_position["qty"]
    live_position_side = live_position["side"]

    # Determine the side needed to close the position (opposite of current position)
    side_to_execute_close = (
        CONFIG.side_sell if live_position_side == CONFIG.pos_long else CONFIG.side_buy
    )

    try:
        # Format amount according to market rules (rounding down) BEFORE converting to float
        amount_str = format_amount(exchange, symbol, live_amount_to_close)
        amount_to_close_precise = Decimal(amount_str)
        amount_float = float(
            amount_to_close_precise
        )  # CCXT create order often expects float

        # Check if the precise amount is valid
        if amount_to_close_precise <= CONFIG.position_qty_epsilon:
            logger.error(
                f"{Fore.RED}Close Position: Closing amount after precision ({amount_str}) is negligible or zero for {symbol}. Aborting closure. Manual check might be needed.{Style.RESET_ALL}"
            )
            # This might happen with extremely small positions or precision issues.
            return None

        logger.warning(
            f"{Back.YELLOW}{Fore.BLACK}Close Position: Attempting to CLOSE {live_position_side} ({reason}): "
            f"Exec {side_to_execute_close.upper()} MARKET {amount_str} {symbol} (reduce_only=True)...{Style.RESET_ALL}"
        )

        # Bybit V5 requires 'reduceOnly': True for closing/reducing orders
        # Category might also be needed depending on CCXT version/implementation
        market = exchange.market(symbol)
        category = "linear" if market.get("linear", True) else "inverse"
        params = {"reduceOnly": True, "category": category}

        # Place the market order to close the position
        order = exchange.create_market_order(
            symbol=symbol,
            side=side_to_execute_close,
            amount=amount_float,
            params=params,
        )

        # --- Parse order response safely using Decimal ---
        order_id = order.get("id")
        status = order.get(
            "status", "unknown"
        )  # Check status if available (market orders often fill immediately)
        fill_price = safe_decimal_conversion(order.get("average"))  # Average fill price
        filled_qty = safe_decimal_conversion(order.get("filled"))  # Amount filled
        cost = safe_decimal_conversion(
            order.get("cost")
        )  # Total cost in quote currency
        fee_cost = safe_decimal_conversion(
            order.get("fee", {}).get("cost", "0.0")
        )  # Fee cost if available
        fee_currency = order.get("fee", {}).get("currency")

        logger.success(
            f"{Fore.GREEN}{Style.BRIGHT}Close Position Order ({reason}) for {symbol} PLACED/FILLED(?). "
            f"ID: {format_order_id(order_id)}, Status: {status}. "
            f"Filled Qty: {filled_qty:.8f}/{amount_str}, AvgFill: {fill_price:.4f}, Cost: {cost:.2f}, Fee: {fee_cost:.4f} {fee_currency or ''}{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[{market_base}] Closed {live_position_side} {amount_str} @ ~{fill_price:.4f} ({reason}). ID:{format_order_id(order_id)}"
        )

        # Optional: Add a short delay and re-check position to be absolutely sure it's closed
        # time.sleep(1.5)
        # final_pos_check = get_current_position(exchange, symbol)
        # if final_pos_check['side'] == CONFIG.pos_none:
        #     logger.info("Post-close check confirms position is flat.")
        # else:
        #     logger.warning(f"Post-close check shows position might still exist! State: {final_pos_check['side']}, Qty: {final_pos_check['qty']:.8f}")

        return order  # Return the order details

    except ccxt.InsufficientFunds as e:
        # This shouldn't happen for a reduceOnly order unless there's a margin issue or bug
        logger.error(
            f"{Fore.RED}Close Position ({reason}): Failed for {symbol} - Insufficient Funds (Unexpected for reduceOnly): {e}{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[{market_base}] ERROR Closing ({reason}): Insufficient Funds reported (Check Margin/State!)."
        )
    except ccxt.ExchangeError as e:
        err_str = str(e).lower()
        # Check for Bybit V5 errors indicating already closed or reduce-only conflict
        # Example codes/messages (check Bybit docs):
        # 110025: Position is zero or closing
        # 110045: Order would not reduce position size (e.g., trying to buy more when short)
        # 30086: order qty exceeds available position size (or similar)
        if (
            "order would not reduce position size" in err_str
            or "position is zero" in err_str
            or "position has been closed" in err_str
            or "size is zero" in err_str
            or "qty exceeds available position size" in err_str
            or "110025" in err_str
            or "110045" in err_str
            or "30086" in err_str
        ):
            logger.warning(
                f"{Fore.YELLOW}Close Position: Exchange indicates position likely already closed/closing or size mismatch ({e}). Assuming closure succeeded or wasn't needed.{Style.RESET_ALL}"
            )
            # Send SMS indicating it was likely already closed if the bot initiated the closure
            send_sms_alert(
                f"[{market_base}] Close ({reason}): Exchange reported position likely already closed/zero."
            )
            return None  # Treat as effectively closed or non-actionable in this case
        else:
            # Log other, potentially more problematic, exchange errors
            logger.error(
                f"{Fore.RED}Close Position ({reason}): Unhandled Exchange error for {symbol}: {e}{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}] ERROR Closing ({reason}): Exchange error: {type(e).__name__}. Check logs."
            )
    except (ccxt.NetworkError, ValueError, TypeError, Exception) as e:
        # Catch network issues, data conversion problems, or other unexpected errors
        logger.error(
            f"{Fore.RED}Close Position ({reason}): Failed for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        send_sms_alert(
            f"[{market_base}] ERROR Closing ({reason}): {type(e).__name__}. Check logs."
        )
    return None  # Indicate failure to close


def calculate_position_size(
    equity: Decimal,
    risk_per_trade_pct: Decimal,
    entry_price: Decimal,
    stop_loss_price: Decimal,
    leverage: int,
    symbol: str,
    exchange: ccxt.Exchange,
) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """Calculates position size (in base currency) and estimated margin based on risk, using Decimal.
    Returns (quantity_precise, required_margin_estimate) or (None, None) on error.
    """
    logger.debug(
        f"Risk Calc: Equity={equity:.4f}, Risk%={risk_per_trade_pct:.4%}, Entry={entry_price:.4f}, SL={stop_loss_price:.4f}, Lev={leverage}x"
    )

    # --- Input Validation ---
    if not (entry_price > 0 and stop_loss_price > 0):
        logger.error(
            f"{Fore.RED}Risk Calc: Invalid entry price ({entry_price}) or SL price ({stop_loss_price}). Must be positive.{Style.RESET_ALL}"
        )
        return None, None
    if entry_price == stop_loss_price:
        logger.error(
            f"{Fore.RED}Risk Calc: Entry price cannot be exactly equal to SL price.{Style.RESET_ALL}"
        )
        return None, None
    price_diff = abs(entry_price - stop_loss_price)
    if price_diff <= CONFIG.position_qty_epsilon:  # Check if difference is negligible
        logger.error(
            f"{Fore.RED}Risk Calc: Entry and SL prices are too close ({price_diff:.8f}). Increase ATR multiplier or check price data.{Style.RESET_ALL}"
        )
        return None, None
    if not 0 < risk_per_trade_pct < 1:
        logger.error(
            f"{Fore.RED}Risk Calc: Invalid risk percentage: {risk_per_trade_pct:.4%}. Must be between 0 and 1 (exclusive).{Style.RESET_ALL}"
        )
        return None, None
    if equity <= 0:
        logger.error(
            f"{Fore.RED}Risk Calc: Invalid equity: {equity:.4f}. Must be positive.{Style.RESET_ALL}"
        )
        return None, None
    if leverage <= 0:
        logger.error(
            f"{Fore.RED}Risk Calc: Invalid leverage: {leverage}. Must be positive.{Style.RESET_ALL}"
        )
        return None, None

    # --- Calculate Risk Amount and Quantity ---
    try:
        risk_amount_usdt = equity * risk_per_trade_pct
        # For linear contracts (like BTC/USDT:USDT), risk per unit of base currency = price_diff in USDT
        # Quantity = Total Risk Amount (USDT) / Risk per Unit (USDT/Base)
        quantity_raw = risk_amount_usdt / price_diff

        # Format the raw quantity according to market precision RULES (rounding down)
        # Then convert back to Decimal for internal use
        quantity_precise_str = format_amount(
            exchange, symbol, quantity_raw
        )  # format_amount handles rounding down
        quantity_precise = Decimal(quantity_precise_str)

    except (ValueError, InvalidOperation, ZeroDivisionError, Exception) as e:
        logger.error(
            f"{Fore.RED}Risk Calc: Error during quantity calculation or formatting for raw qty {quantity_raw if 'quantity_raw' in locals() else 'N/A'}: {e}{Style.RESET_ALL}"
        )
        return None, None

    # Check if calculated quantity is valid after precision formatting
    if quantity_precise <= CONFIG.position_qty_epsilon:
        logger.warning(
            f"{Fore.YELLOW}Risk Calc: Calculated quantity negligible or zero after precision ({quantity_precise:.8f}). RiskAmt={risk_amount_usdt:.4f}, PriceDiff={price_diff:.4f}. Cannot place order.{Style.RESET_ALL}"
        )
        return None, None

    # --- Calculate Estimated Margin ---
    try:
        position_value_usdt = quantity_precise * entry_price
        required_margin_estimate = position_value_usdt / Decimal(leverage)
    except (InvalidOperation, ZeroDivisionError, Exception) as e:
        logger.error(
            f"{Fore.RED}Risk Calc: Error calculating estimated margin: {e}{Style.RESET_ALL}"
        )
        return None, None

    logger.debug(
        f"Risk Calc Result: Qty={quantity_precise:.8f}, RiskAmt={risk_amount_usdt:.4f}, EstValue={position_value_usdt:.4f}, EstMargin={required_margin_estimate:.4f}"
    )
    return quantity_precise, required_margin_estimate


def wait_for_order_fill(
    exchange: ccxt.Exchange, order_id: str, symbol: str, timeout_seconds: int
) -> Optional[Dict[str, Any]]:
    """Waits for a specific order to reach a 'closed' (filled) status via polling.
    Returns the filled order dict or None if timeout or failed status reached.
    """
    start_time = time.time()
    order_id_short = format_order_id(order_id)
    logger.info(
        f"{Fore.CYAN}Waiting for order {order_id_short} ({symbol}) fill (Timeout: {timeout_seconds}s)...{Style.RESET_ALL}"
    )

    while time.time() - start_time < timeout_seconds:
        try:
            # fetch_order is preferred for checking a specific ID
            order = exchange.fetch_order(order_id, symbol)
            status = order.get("status")
            logger.debug(f"Order {order_id_short} status check: {status}")

            if status == "closed":
                logger.success(
                    f"{Fore.GREEN}Order {order_id_short} confirmed FILLED.{Style.RESET_ALL}"
                )
                # Return the complete order structure fetched
                return order
            elif status in ["canceled", "rejected", "expired"]:
                logger.error(
                    f"{Fore.RED}Order {order_id_short} reached final FAILED status: '{status}'.{Style.RESET_ALL}"
                )
                return None  # Indicate failure

            # Continue polling if 'open', 'partially_filled' (market orders usually go straight to closed), or None/unknown
            time.sleep(CONFIG.market_order_fill_check_interval)  # Check frequently

        except ccxt.OrderNotFound:
            # Can happen briefly after placing or if already closed/canceled and pruned by exchange
            elapsed_time = time.time() - start_time
            # Tolerate 'not found' for a short initial period, as the order might still be processing
            if elapsed_time < 5.0:  # Allow 5 seconds grace period
                logger.warning(
                    f"{Fore.YELLOW}Order {order_id_short} not found yet (after {elapsed_time:.1f}s). Retrying...{Style.RESET_ALL}"
                )
                time.sleep(0.75)  # Slightly longer wait if not found initially
            else:
                # If not found after the grace period, assume it failed or was processed differently
                logger.error(
                    f"{Fore.RED}Order {order_id_short} not found after {elapsed_time:.1f}s grace period. Assuming failed/pruned.{Style.RESET_ALL}"
                )
                return None

        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            # Handle temporary API/network issues during polling
            logger.warning(
                f"{Fore.YELLOW}Error checking order {order_id_short}: {type(e).__name__} - {e}. Retrying...{Style.RESET_ALL}"
            )
            time.sleep(1.5)  # Wait a bit longer on API errors before retrying poll
        except Exception as e:
            # Catch unexpected errors during the fetch/check process
            logger.error(
                f"{Fore.RED}Unexpected error checking order {order_id_short}: {e}{Style.RESET_ALL}"
            )
            logger.debug(traceback.format_exc())
            time.sleep(1.5)  # Wait longer on unexpected errors

    # Loop finished without success (timeout)
    logger.error(
        f"{Fore.RED}Order {order_id_short} did NOT fill within {timeout_seconds}s timeout.{Style.RESET_ALL}"
    )
    # Attempt to fetch one last time to log the final status if possible
    try:
        final_order_check = exchange.fetch_order(order_id, symbol)
        logger.warning(
            f"Final status check for timed-out order {order_id_short}: Status='{final_order_check.get('status')}', Filled='{final_order_check.get('filled')}'"
        )
    except Exception as final_e:
        logger.warning(
            f"Could not perform final status check for timed-out order {order_id_short}: {final_e}"
        )
    return None  # Indicate timeout


def place_risked_market_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    risk_percentage: Decimal,
    current_atr: Optional[Decimal],
    sl_atr_multiplier: Decimal,
    leverage: int,
    max_order_cap_usdt: Decimal,
    margin_check_buffer: Decimal,
    tsl_percent: Decimal,
    tsl_activation_offset_percent: Decimal,
) -> Optional[Dict[str, Any]]:
    """Handles the complete process of placing a risk-calculated market entry order,
    waiting for fill, and then placing exchange-native fixed SL and TSL orders.
    Uses Decimal precision throughout calculations. Returns the filled entry order dict on success, None on major failure.
    """
    market_base = get_market_base_currency(symbol)
    action_verb = "LONG" if side == CONFIG.side_buy else "SHORT"
    logger.info(
        f"{Fore.BLUE}{Style.BRIGHT}Place Order: Initiating {action_verb} entry for {symbol} with Risk/SL/TSL...{Style.RESET_ALL}"
    )

    # --- Pre-computation & Validation ---
    if current_atr is None or current_atr <= CONFIG.position_qty_epsilon:
        logger.error(
            f"{Fore.RED}Place Order ({action_verb}): Invalid or zero ATR ({current_atr}). Cannot calculate SL or place order.{Style.RESET_ALL}"
        )
        return None

    market: Optional[Dict] = None
    entry_price_estimate: Optional[Decimal] = None
    initial_sl_price_estimate: Optional[Decimal] = None
    final_quantity: Optional[Decimal] = None
    entry_order_id: Optional[str] = None
    filled_entry_order: Optional[Dict[str, Any]] = None
    sl_order_id: Optional[str] = None
    tsl_order_id: Optional[str] = None
    sl_status: str = "Not Placed"
    tsl_status: str = "Not Placed"
    actual_sl_price_str: str = "N/A"  # For final summary

    try:
        # === 1. Get Balance, Market Info, Limits ===
        logger.debug("Fetching balance & market details...")
        balance = exchange.fetch_balance()
        market = exchange.market(symbol)
        if not market:
            raise ValueError(f"Market {symbol} not found in loaded markets.")

        # Extract limits safely
        limits = market.get("limits", {})
        amount_limits = limits.get("amount", {})
        price_limits = limits.get("price", {})
        min_qty = (
            safe_decimal_conversion(amount_limits.get("min"))
            if amount_limits.get("min")
            else None
        )
        max_qty = (
            safe_decimal_conversion(amount_limits.get("max"))
            if amount_limits.get("max")
            else None
        )
        min_price = (
            safe_decimal_conversion(price_limits.get("min"))
            if price_limits.get("min")
            else None
        )
        logger.debug(
            f"Market Limits: MinQty={min_qty or 'N/A'}, MaxQty={max_qty or 'N/A'}, MinPrice={min_price or 'N/A'}"
        )

        # Get USDT balance (adjust key if using different quote currency)
        usdt_balance = balance.get(CONFIG.usdt_symbol, {})
        # Use 'total' for equity calculation, 'free' for margin check. Handle potential missing keys.
        usdt_total = safe_decimal_conversion(usdt_balance.get("total"))
        usdt_free = safe_decimal_conversion(usdt_balance.get("free"))
        # Bybit V5 might have different structures, check `exchange.fetch_balance()` response if issues arise.
        # Use total if available and positive, otherwise fallback to free (less accurate for equity but safer than zero)
        usdt_equity = usdt_total if usdt_total > 0 else usdt_free

        if usdt_equity <= Decimal("0"):
            logger.error(
                f"{Fore.RED}Place Order ({action_verb}): Zero or invalid equity ({usdt_equity:.4f}). Cannot calculate risk or place order.{Style.RESET_ALL}"
            )
            return None
        if usdt_free < Decimal(
            "0"
        ):  # Free margin shouldn't be negative, but check defensively
            logger.warning(
                f"{Fore.YELLOW}Place Order ({action_verb}): Negative free margin detected ({usdt_free:.4f}). May indicate issues.{Style.RESET_ALL}"
            )
            # Allow proceeding but log warning, as calculation uses equity. Margin check later will fail if truly insufficient.

        logger.debug(
            f"Balance: Equity={usdt_equity:.4f} {CONFIG.usdt_symbol}, Free={usdt_free:.4f} {CONFIG.usdt_symbol}"
        )

        # === 2. Estimate Entry Price (using shallow OB or ticker for market order slippage estimate) ===
        logger.debug("Estimating entry price for calculations...")
        ob_data = analyze_order_book(
            exchange,
            symbol,
            CONFIG.shallow_ob_fetch_depth,
            CONFIG.shallow_ob_fetch_depth,
        )
        best_ask = ob_data.get("best_ask")
        best_bid = ob_data.get("best_bid")
        # For market buy, estimate fill around best ask; for market sell, around best bid
        if side == CONFIG.side_buy and best_ask:
            entry_price_estimate = best_ask
        elif side == CONFIG.side_sell and best_bid:
            entry_price_estimate = best_bid
        else:
            # Fallback to ticker if OB fails or is insufficient
            logger.debug("OB price unavailable for estimate, falling back to ticker.")
            try:
                ticker = exchange.fetch_ticker(symbol)
                # Use 'last' price from ticker as fallback estimate
                entry_price_estimate = safe_decimal_conversion(ticker.get("last"))
                if not entry_price_estimate or entry_price_estimate <= 0:
                    raise ValueError(f"Invalid ticker price: {ticker.get('last')}")
            except (ccxt.NetworkError, ccxt.ExchangeError, ValueError, Exception) as e:
                logger.error(
                    f"{Fore.RED}Place Order ({action_verb}): Failed to fetch ticker or get valid price for estimation: {e}{Style.RESET_ALL}"
                )
                return None  # Cannot proceed without a price estimate

        if not entry_price_estimate or entry_price_estimate <= 0:
            logger.error(
                f"{Fore.RED}Place Order ({action_verb}): Failed to obtain a valid positive entry price estimate ({entry_price_estimate}). Aborting.{Style.RESET_ALL}"
            )
            return None
        logger.debug(f"Estimated Entry Price ~ {entry_price_estimate:.4f}")

        # === 3. Calculate Initial Stop Loss Price (Estimate based on estimated entry) ===
        sl_distance = current_atr * sl_atr_multiplier
        initial_sl_price_raw = (
            (entry_price_estimate - sl_distance)
            if side == CONFIG.side_buy
            else (entry_price_estimate + sl_distance)
        )

        # Ensure SL is valid and respects minimum price
        if min_price is not None and initial_sl_price_raw < min_price:
            logger.warning(
                f"{Fore.YELLOW}Calculated SL price estimate {initial_sl_price_raw:.4f} is below min price {min_price}. Adjusting SL estimate to min price.{Style.RESET_ALL}"
            )
            initial_sl_price_raw = min_price
        if initial_sl_price_raw <= 0:
            logger.error(
                f"{Fore.RED}Place Order ({action_verb}): Invalid Initial SL price calculated ({initial_sl_price_raw:.4f}). Check ATR/multiplier/price data. Aborting.{Style.RESET_ALL}"
            )
            return None

        # Format the estimated SL price using market precision for accurate risk calculation
        initial_sl_price_str_estimate = format_price(
            exchange, symbol, initial_sl_price_raw
        )
        initial_sl_price_estimate = Decimal(initial_sl_price_str_estimate)
        logger.info(
            f"Calculated Initial SL Price (Estimate) ~ {initial_sl_price_estimate:.4f} (ATR Dist: {sl_distance:.4f})"
        )

        # === 4. Calculate Position Size based on Risk & Estimated SL ===
        calc_qty, req_margin_estimate = calculate_position_size(
            usdt_equity,
            risk_percentage,
            entry_price_estimate,
            initial_sl_price_estimate,
            leverage,
            symbol,
            exchange,
        )
        if calc_qty is None or req_margin_estimate is None:
            logger.error(
                f"{Fore.RED}Place Order ({action_verb}): Failed risk calculation. Cannot determine position size.{Style.RESET_ALL}"
            )
            return None
        final_quantity = calc_qty  # Start with risk-based quantity

        # === 5. Apply Max Order Value Cap ===
        pos_value_estimate = final_quantity * entry_price_estimate
        if pos_value_estimate > max_order_cap_usdt:
            logger.warning(
                f"{Fore.YELLOW}Calculated order value {pos_value_estimate:.4f} USDT exceeds cap {max_order_cap_usdt:.4f}. Capping quantity based on max value.{Style.RESET_ALL}"
            )
            # Calculate the quantity that would hit the cap exactly
            capped_quantity_raw = max_order_cap_usdt / entry_price_estimate
            # Format capped quantity according to market rules (round down)
            capped_quantity_str = format_amount(exchange, symbol, capped_quantity_raw)
            final_quantity = Decimal(capped_quantity_str)
            # Recalculate estimated margin based on the capped quantity
            req_margin_estimate = (final_quantity * entry_price_estimate) / Decimal(
                leverage
            )
            logger.info(
                f"Quantity capped to {final_quantity:.8f}. New Est Margin ~ {req_margin_estimate:.4f}"
            )

        # === 6. Check Quantity Limits & Margin Availability ===
        if final_quantity <= CONFIG.position_qty_epsilon:
            logger.error(
                f"{Fore.RED}Place Order ({action_verb}): Final quantity ({final_quantity:.8f}) is negligible or zero after calculations/capping. Aborting.{Style.RESET_ALL}"
            )
            return None
        if min_qty is not None and final_quantity < min_qty:
            logger.error(
                f"{Fore.RED}Place Order ({action_verb}): Final quantity {final_quantity:.8f} is below market minimum {min_qty}. Aborting.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}] ORDER FAIL ({action_verb}): Qty {final_quantity:.8f} < Min {min_qty}"
            )
            return None
        if max_qty is not None and final_quantity > max_qty:
            logger.warning(
                f"{Fore.YELLOW}Final quantity {final_quantity:.8f} exceeds market maximum {max_qty}. Adjusting to max limit.{Style.RESET_ALL}"
            )
            final_quantity = max_qty  # Use the absolute max allowed by exchange
            # Re-format just in case max_qty needs formatting (unlikely but safe)
            final_quantity = Decimal(format_amount(exchange, symbol, final_quantity))
            # Recalculate final margin estimate based on exchange max quantity
            req_margin_estimate = (final_quantity * entry_price_estimate) / Decimal(
                leverage
            )

        # Final margin check using free balance and buffer
        req_margin_buffered = req_margin_estimate * margin_check_buffer
        if usdt_free < req_margin_buffered:
            logger.error(
                f"{Fore.RED}Place Order ({action_verb}): Insufficient FREE margin. Need ~{req_margin_buffered:.4f} (incl. {margin_check_buffer:.1%} buffer), Have {usdt_free:.4f}{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}] ORDER FAIL ({action_verb}): Insufficient Free Margin (Need ~{req_margin_buffered:.2f})"
            )
            return None

        logger.info(
            f"{Fore.GREEN}Pre-order Checks Passed. Final Qty: {final_quantity:.8f}, Est Margin: {req_margin_estimate:.4f}, Buffered Margin Need: {req_margin_buffered:.4f}{Style.RESET_ALL}"
        )

        # === 7. Place Entry Market Order ===
        entry_order_details: Optional[Dict[str, Any]] = None
        try:
            qty_float = float(
                final_quantity
            )  # CCXT requires float for amount in create_order
            logger.warning(
                f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}*** Placing {action_verb} MARKET ENTRY: {qty_float:.8f} {symbol} ***{Style.RESET_ALL}"
            )
            # For Bybit V5, ensure `reduceOnly` is false or omitted for entry orders. Category might be needed.
            category = "linear" if market.get("linear", True) else "inverse"
            entry_params = {"reduceOnly": False, "category": category}
            entry_order_details = exchange.create_market_order(
                symbol=symbol, side=side, amount=qty_float, params=entry_params
            )
            entry_order_id = entry_order_details.get("id")

            if not entry_order_id:
                # This is highly unusual for a successful placement
                logger.error(
                    f"{Fore.RED}{Style.BRIGHT}Entry order placed but NO Order ID returned! Critical issue. Response: {entry_order_details}{Style.RESET_ALL}"
                )
                # Attempt to check position just in case it somehow opened without ID confirmation
                time.sleep(1.5)
                current_pos = get_current_position(exchange, symbol)
                if current_pos["side"] != CONFIG.pos_none:
                    logger.warning(
                        f"{Fore.YELLOW}Position check reveals position exists despite missing entry ID! Qty: {current_pos['qty']}. Cannot reliably manage SL/TSL. Manual check required!{Style.RESET_ALL}"
                    )
                    send_sms_alert(
                        f"[{market_base}] CRITICAL: Position opened ({action_verb}) but entry ID MISSING! Manual check needed!"
                    )
                    # Cannot proceed with SL/TSL reliably, returning None signals failure upstream
                    return None
                # If no position exists either, assume the order truly failed despite lack of error
                raise ValueError(
                    "Entry order placement failed to return an ID and no position found."
                )

            logger.success(
                f"{Fore.GREEN}Market Entry Order submitted. ID: {format_order_id(entry_order_id)}. Waiting for fill confirmation...{Style.RESET_ALL}"
            )

        except (
            ccxt.InsufficientFunds,
            ccxt.ExchangeError,
            ccxt.NetworkError,
            ValueError,
            Exception,
        ) as e:
            logger.error(
                f"{Fore.RED}{Style.BRIGHT}FAILED TO PLACE ENTRY ORDER ({action_verb}): {e}{Style.RESET_ALL}"
            )
            logger.debug(traceback.format_exc())
            send_sms_alert(
                f"[{market_base}] ORDER FAIL ({action_verb}): Entry placement failed: {type(e).__name__}"
            )
            return None  # Stop the process if entry fails

        # === 8. Wait for Entry Order Fill Confirmation ===
        filled_entry_order = wait_for_order_fill(
            exchange, entry_order_id, symbol, CONFIG.order_fill_timeout_seconds
        )
        if not filled_entry_order:
            logger.error(
                f"{Fore.RED}{Style.BRIGHT}Entry order {format_order_id(entry_order_id)} did NOT fill or failed within timeout.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}] ORDER FAIL ({action_verb}): Entry {format_order_id(entry_order_id)} fill timeout/fail."
            )
            # Attempt to cancel the potentially stuck order (might fail if already filled/gone)
            try:
                logger.warning(
                    f"Attempting to cancel potentially unfilled/failed entry order {format_order_id(entry_order_id)}..."
                )
                exchange.cancel_order(entry_order_id, symbol)
                logger.info(
                    f"Cancel request sent for {format_order_id(entry_order_id)}."
                )
            except ccxt.OrderNotFound:
                logger.warning(
                    f"Could not cancel order {format_order_id(entry_order_id)} (likely already filled, cancelled, or rejected)."
                )
            except (ccxt.ExchangeError, ccxt.NetworkError) as cancel_e:
                logger.warning(
                    f"Error sending cancel request for order {format_order_id(entry_order_id)}: {cancel_e}"
                )
            except Exception as cancel_e:
                logger.error(
                    f"Unexpected error cancelling order {format_order_id(entry_order_id)}: {cancel_e}"
                )

            # --- CRITICAL CHECK after failed fill/cancel attempt ---
            # Verify if a position was opened despite the fill failure indication.
            time.sleep(1.5)  # Allow state to potentially update
            current_pos = get_current_position(exchange, symbol)
            if current_pos["side"] != CONFIG.pos_none:
                logger.error(
                    f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}CRITICAL STATE: POSITION OPENED ({current_pos['side']} Qty: {current_pos['qty']}) despite entry fill failure/timeout! Attempting emergency close!{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"[{market_base}] CRITICAL: Position opened ({action_verb}) on FAILED entry fill! Closing NOW."
                )
                # Attempt immediate closure
                close_position(
                    exchange,
                    symbol,
                    current_pos,
                    reason="Emergency Close - Failed Entry Fill",
                )
            # Whether position was found and closed or not, the entry process failed here.
            return None

        # === 9. Extract Actual Fill Details (Crucial: Use Actual Fill Info) ===
        avg_fill_price = safe_decimal_conversion(filled_entry_order.get("average"))
        filled_qty = safe_decimal_conversion(filled_entry_order.get("filled"))
        cost = safe_decimal_conversion(filled_entry_order.get("cost"))
        fee_cost = safe_decimal_conversion(
            filled_entry_order.get("fee", {}).get("cost", "0.0")
        )
        fee_currency = filled_entry_order.get("fee", {}).get("currency")

        # Validate crucial fill details
        if avg_fill_price <= 0 or filled_qty <= CONFIG.position_qty_epsilon:
            logger.error(
                f"{Fore.RED}{Style.BRIGHT}Invalid fill details received for order {format_order_id(entry_order_id)}: Price={avg_fill_price}, Qty={filled_qty}. Cannot proceed with SL/TSL.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}] ORDER FAIL ({action_verb}): Invalid fill details {format_order_id(entry_order_id)}."
            )
            # --- CRITICAL STATE: Position might be open with bad data ---
            logger.error(
                f"{Back.RED}{Fore.WHITE}Attempting emergency close due to invalid fill details...{Style.RESET_ALL}"
            )
            # Re-fetch position state as filled_qty might be unreliable
            current_pos = get_current_position(exchange, symbol)
            if current_pos["side"] != CONFIG.pos_none:
                close_position(
                    exchange,
                    symbol,
                    current_pos,
                    reason="Emergency Close - Invalid Fill Data",
                )
            else:
                logger.warning(
                    "Position seems already closed or wasn't opened despite invalid fill data."
                )
            # Even if closed, signal failure upstream by returning None
            return None

        logger.success(
            f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}ENTRY CONFIRMED: {format_order_id(entry_order_id)}. "
            f"Filled Qty: {filled_qty:.8f} @ AvgPrice: {avg_fill_price:.4f}. Cost: {cost:.4f}, Fee: {fee_cost:.4f} {fee_currency or ''}{Style.RESET_ALL}"
        )

        # --- Post-Fill Actions: Place SL and TSL based on ACTUAL fill ---
        # Use the ACTUAL filled quantity and average price for SL/TSL calculations

        # === 10. Calculate ACTUAL Stop Loss Price based on Actual Fill Price ===
        # Recalculate SL distance using the original ATR value but the actual fill price
        actual_sl_price_raw = (
            (avg_fill_price - sl_distance)
            if side == CONFIG.side_buy
            else (avg_fill_price + sl_distance)
        )
        # Validate and adjust SL price against min price limit
        if min_price is not None and actual_sl_price_raw < min_price:
            logger.warning(
                f"{Fore.YELLOW}Adjusted SL price {actual_sl_price_raw:.4f} is below min price {min_price}. Adjusting SL to min price.{Style.RESET_ALL}"
            )
            actual_sl_price_raw = min_price
        if actual_sl_price_raw <= 0:
            # --- CRITICAL FAILURE --- Position is open without valid SL calculation possible
            logger.error(
                f"{Fore.RED}{Style.BRIGHT}Invalid ACTUAL SL price calculated ({actual_sl_price_raw:.4f}) based on fill price {avg_fill_price:.4f}. CANNOT PLACE SL! Attempting emergency close!{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}] CRITICAL ({action_verb}): Invalid ACTUAL SL price ({actual_sl_price_raw:.4f})! Emergency Closing!"
            )
            # Use filled details for emergency close attempt
            close_position(
                exchange,
                symbol,
                {"side": side, "qty": filled_qty, "entry_price": avg_fill_price},
                reason="Emergency Close - Invalid SL Calc",
            )
            # Return the filled entry order to indicate entry happened, but signal overall failure upstream
            return filled_entry_order  # Let caller know entry occurred but failed after

        # Format the final SL price for the order
        actual_sl_price_str = format_price(exchange, symbol, actual_sl_price_raw)
        actual_sl_price_float = float(
            actual_sl_price_str
        )  # For CCXT param requiring float

        # === 11. Place Initial Fixed Stop Loss (Stop Market Order) ===
        try:
            sl_side = CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy
            # Use the actual filled quantity, formatted correctly
            sl_qty_str = format_amount(exchange, symbol, filled_qty)
            sl_qty_float = float(sl_qty_str)

            if sl_qty_float <= 0:
                raise ValueError(
                    f"Stop Loss quantity is zero or negative ({sl_qty_float})"
                )

            logger.info(
                f"{Fore.CYAN}Placing Initial Fixed SL ({sl_atr_multiplier}*ATR)... TriggerPx: {actual_sl_price_str}, Qty: {sl_qty_float:.8f}{Style.RESET_ALL}"
            )
            # --- Bybit V5 Stop Market Order Params via CCXT ---
            # Required: stopPrice (trigger), reduceOnly=True
            # Optional/Contextual: basePrice (trigger source), triggerDirection, tpslMode, slOrderType='Market'
            category = "linear" if market.get("linear", True) else "inverse"
            sl_params = {
                "stopPrice": actual_sl_price_float,  # The trigger price
                "reduceOnly": True,
                "category": category,
                # Optional: Specify trigger price type if needed (default often 'LastPrice')
                # 'triggerPrice': 'MarkPrice' or 'IndexPrice',
                # Optional: Ensure it's a market stop-loss if exchange requires it
                # 'slOrderType': 'Market', # Or handled by type='StopMarket'/'Stop'
                # Optional: Position mode if needed
                # 'positionIdx': 0 # For One-Way mode
            }
            # Use create_order with type='Stop' or 'StopMarket' (check CCXT docs for Bybit alias)
            # 'Stop' often implies StopMarket for derivatives on CCXT
            sl_order = exchange.create_order(
                symbol=symbol,
                type="Stop",  # Use 'Stop' as type for stop-market on Bybit via CCXT
                side=sl_side,
                amount=sl_qty_float,
                params=sl_params,
            )

            sl_order_id = sl_order.get("id")
            if sl_order_id:
                sl_status = f"Placed (ID: {format_order_id(sl_order_id)}, Trigger: {actual_sl_price_str})"
                logger.success(f"{Fore.GREEN}{sl_status}{Style.RESET_ALL}")
            else:
                # Should ideally not happen if no exception was raised
                sl_status = f"Placement FAILED (No ID returned). Response: {sl_order}"
                logger.error(f"{Fore.RED}{sl_status}{Style.RESET_ALL}")
                send_sms_alert(
                    f"[{market_base}] ERROR ({action_verb}): Failed initial SL placement (NO ID)."
                )

        except (
            ccxt.InsufficientFunds,
            ccxt.ExchangeError,
            ccxt.NetworkError,
            ValueError,
            Exception,
        ) as e:
            sl_status = f"Placement FAILED: {type(e).__name__} - {e}"
            logger.error(
                f"{Fore.RED}{Style.BRIGHT}FAILED to place Initial Fixed SL order: {e}{Style.RESET_ALL}"
            )
            logger.debug(traceback.format_exc())
            send_sms_alert(
                f"[{market_base}] ERROR ({action_verb}): Failed initial SL placement: {type(e).__name__}"
            )
            # Decide if this is critical enough to close the position.
            # For now, log and continue to TSL attempt, but position is less protected.
            # Consider adding an emergency close here if fixed SL fails.

        # === 12. Place Trailing Stop Loss (if percentage > 0) ===
        tsl_trail_value_str = "N/A"  # For final summary
        if (
            tsl_percent > CONFIG.position_qty_epsilon
        ):  # Only place if TSL percentage is meaningful
            try:
                # Calculate TSL activation price based on actual fill price
                act_offset = avg_fill_price * tsl_activation_offset_percent
                # Activation price: Slightly in profit from entry
                act_price_raw = (
                    (avg_fill_price + act_offset)
                    if side == CONFIG.side_buy
                    else (avg_fill_price - act_offset)
                )
                # Validate and adjust activation price
                if min_price is not None and act_price_raw < min_price:
                    act_price_raw = min_price
                if act_price_raw <= 0:
                    raise ValueError(
                        f"Invalid TSL activation price calculated: {act_price_raw:.4f}"
                    )

                tsl_act_price_str = format_price(exchange, symbol, act_price_raw)
                tsl_act_price_float = float(tsl_act_price_str)
                tsl_side = (
                    CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy
                )

                # Bybit V5 uses 'trailingStop' for percentage distance, requires value as string percentage (e.g., "0.5" for 0.5%)
                # Ensure correct formatting (e.g., "0.50" for 0.5%)
                tsl_trail_value_str = f"{(tsl_percent * Decimal('100')).quantize(Decimal('0.01'))}"  # Format to 2 decimal places percentage string

                tsl_qty_str = format_amount(exchange, symbol, filled_qty)
                tsl_qty_float = float(tsl_qty_str)

                if tsl_qty_float <= 0:
                    raise ValueError(
                        f"Trailing Stop Loss quantity is zero or negative ({tsl_qty_float})"
                    )

                logger.info(
                    f"{Fore.CYAN}Placing Trailing SL ({tsl_percent:.2%})... Trail%: {tsl_trail_value_str}%, ActPx: {tsl_act_price_str}, Qty: {tsl_qty_float:.8f}{Style.RESET_ALL}"
                )
                # --- Bybit V5 Trailing Stop Params via CCXT ---
                # Required: trailingStop (percentage string), activePrice (trigger), reduceOnly=True
                category = "linear" if market.get("linear", True) else "inverse"
                tsl_params = {
                    "trailingStop": tsl_trail_value_str,  # e.g., '0.5' for 0.5%
                    "activePrice": tsl_act_price_float,  # Price at which the TSL becomes active and starts trailing
                    "reduceOnly": True,
                    "category": category,
                    # Optional: Ensure market TSL if needed
                    # 'slOrderType': 'Market',
                    # Optional: Position mode
                    # 'positionIdx': 0
                }
                # Use create_order with type='Stop' for TSL as well on Bybit via CCXT
                tsl_order = exchange.create_order(
                    symbol=symbol,
                    type="Stop",  # TSL is often a variant of a Stop order
                    side=tsl_side,
                    amount=tsl_qty_float,
                    params=tsl_params,
                )

                tsl_order_id = tsl_order.get("id")
                if tsl_order_id:
                    tsl_status = f"Placed (ID: {format_order_id(tsl_order_id)}, Trail: {tsl_trail_value_str}%, ActPx: {tsl_act_price_str})"
                    logger.success(f"{Fore.GREEN}{tsl_status}{Style.RESET_ALL}")
                else:
                    tsl_status = (
                        f"Placement FAILED (No ID returned). Response: {tsl_order}"
                    )
                    logger.error(f"{Fore.RED}{tsl_status}{Style.RESET_ALL}")
                    send_sms_alert(
                        f"[{market_base}] ERROR ({action_verb}): Failed TSL placement (NO ID)."
                    )

            except (
                ccxt.InsufficientFunds,
                ccxt.ExchangeError,
                ccxt.NetworkError,
                ValueError,
                Exception,
            ) as e:
                tsl_status = f"Placement FAILED: {type(e).__name__} - {e}"
                logger.error(
                    f"{Fore.RED}{Style.BRIGHT}FAILED to place Trailing SL order: {e}{Style.RESET_ALL}"
                )
                logger.debug(traceback.format_exc())
                send_sms_alert(
                    f"[{market_base}] ERROR ({action_verb}): Failed TSL placement: {type(e).__name__}"
                )
        else:
            # TSL not configured
            tsl_status = "Not Configured (Percentage Zero)"
            logger.info(
                f"{Fore.CYAN}Trailing SL not configured (percentage is zero or less). Skipping TSL placement.{Style.RESET_ALL}"
            )

        # === 13. Final Summary Log & SMS ===
        logger.info(
            f"{Back.BLUE}{Fore.WHITE}--- ORDER PLACEMENT SUMMARY ({action_verb} {symbol}) ---{Style.RESET_ALL}"
        )
        logger.info(
            f"  Entry Order: {format_order_id(entry_order_id)} | Filled Qty: {filled_qty:.8f} @ Avg Price: {avg_fill_price:.4f}"
        )
        logger.info(f"  Fixed SL:    {sl_status}")
        logger.info(f"  Trailing SL: {tsl_status}")
        logger.info(f"{Back.BLUE}{Fore.WHITE}--- END SUMMARY ---{Style.RESET_ALL}")

        # Send comprehensive SMS only if entry was successful
        # Use actual SL price string calculated earlier
        sl_summary = actual_sl_price_str if sl_order_id else "FAIL"
        tsl_summary = (
            "%" + tsl_trail_value_str
            if tsl_order_id
            else ("FAIL" if tsl_percent > 0 else "OFF")
        )
        sms_summary = (
            f"[{market_base}] {action_verb} {filled_qty:.6f}@{avg_fill_price:.3f}. "
            f"SL:{sl_summary}. TSL:{tsl_summary}. E:{format_order_id(entry_order_id)}"
        )
        send_sms_alert(sms_summary)

        # Return the filled entry order details, signalling overall success of getting into position
        return filled_entry_order

    except (
        ccxt.InsufficientFunds,
        ccxt.NetworkError,
        ccxt.ExchangeError,
        ValueError,
        Exception,
    ) as e:
        # Catch errors occurring before entry placement or during setup phases (balance check, price fetch, calc)
        logger.error(
            f"{Fore.RED}{Style.BRIGHT}Place Order ({action_verb}): Overall process failed before/during entry placement: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        send_sms_alert(
            f"[{market_base}] ORDER FAIL ({action_verb}): Pre-entry setup/checks failed: {type(e).__name__}"
        )
        # Ensure we didn't somehow open a position if an entry ID was generated but fill confirmation wasn't reached
        if entry_order_id and not filled_entry_order:
            logger.warning(
                "Checking position status after setup failure but before confirmed fill..."
            )
            time.sleep(1.5)
            current_pos = get_current_position(exchange, symbol)
            if current_pos["side"] != CONFIG.pos_none:
                logger.error(
                    f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}CRITICAL STATE: POSITION OPENED ({current_pos['side']} Qty: {current_pos['qty']}) despite order setup failure! Attempting emergency close!{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"[{market_base}] CRITICAL: Position opened ({action_verb}) on FAILED setup! Closing NOW."
                )
                close_position(
                    exchange,
                    symbol,
                    current_pos,
                    reason="Emergency Close - Failed Order Setup",
                )
    # Indicate overall failure if any exception occurred outside the post-fill SL/TSL placement
    return None


def cancel_open_orders(
    exchange: ccxt.Exchange, symbol: str, reason: str = "Cleanup"
) -> None:
    """Attempts to cancel all open orders (limit, stop, conditional) for the specified symbol."""
    logger.info(
        f"{Fore.CYAN}Order Cancel: Attempting for {symbol} (Reason: {reason})...{Style.RESET_ALL}"
    )
    cancelled_count, failed_count = 0, 0
    market_base = get_market_base_currency(symbol)

    try:
        # Use fetch_open_orders. Bybit V5 might require category, CCXT often handles this.
        # Stops might require fetch_open_orders(params={'stop': True}) or similar on some exchanges/versions.
        # Check CCXT documentation for Bybit V5 specifics if needed.
        if not exchange.has.get("fetchOpenOrders"):
            logger.warning(
                f"{Fore.YELLOW}Order Cancel: fetchOpenOrders capability not reported by CCXT for {exchange.id}. Cannot automatically cancel.{Style.RESET_ALL}"
            )
            return
        if not exchange.has.get("cancelOrder"):
            logger.warning(
                f"{Fore.YELLOW}Order Cancel: cancelOrder capability not reported by CCXT for {exchange.id}. Cannot cancel.{Style.RESET_ALL}"
            )
            return

        logger.debug("Fetching open orders for cancellation...")
        # Add params if needed, e.g., {'category': 'linear'} or potentially type filters like {'stop': True}
        # Fetching without specific filters should get regular and conditional orders on Bybit V5 via CCXT
        market = exchange.market(symbol)
        category = "linear" if market.get("linear", True) else "inverse"
        open_orders = exchange.fetch_open_orders(symbol, params={"category": category})

        if not open_orders:
            logger.info(
                f"{Fore.CYAN}Order Cancel: No open orders found for {symbol}.{Style.RESET_ALL}"
            )
            return

        logger.warning(
            f"{Fore.YELLOW}Order Cancel: Found {len(open_orders)} open orders for {symbol}. Attempting cancellation...{Style.RESET_ALL}"
        )

        for order in open_orders:
            order_id = order.get("id")
            order_type = order.get("type", "N/A")
            order_side = order.get("side", "N/A")
            order_status = order.get("status", "N/A")  # Should be 'open'
            order_stop_price = order.get("stopPrice")  # Check for stop price

            order_info = f"ID: {format_order_id(order_id)} ({order_type} {order_side} Qty:{order.get('amount', 'N/A')} Px:{order.get('price', 'N/A')}"
            if order_stop_price:
                order_info += f" StopPx:{order_stop_price}"
            order_info += f" Status:{order_status})"

            if (
                order_id and order_status == "open"
            ):  # Only attempt to cancel open orders with an ID
                try:
                    logger.debug(f"Cancelling order: {order_info}")
                    # Bybit V5 cancelOrder might need category, CCXT should handle it
                    exchange.cancel_order(
                        order_id, symbol, params={"category": category}
                    )
                    logger.info(
                        f"{Fore.CYAN}Order Cancel: Success for {order_info}{Style.RESET_ALL}"
                    )
                    cancelled_count += 1
                    time.sleep(
                        0.15
                    )  # Small delay between cancellations to avoid potential rate limits

                except ccxt.OrderNotFound:
                    # Order might have been filled/cancelled between fetch and cancel call
                    logger.warning(
                        f"{Fore.YELLOW}Order Cancel: Order not found (already closed/cancelled?): {order_info}{Style.RESET_ALL}"
                    )
                    # Count as 'cancelled' in this context as it's no longer open
                    cancelled_count += 1
                except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                    logger.error(
                        f"{Fore.RED}Order Cancel: FAILED for {order_info}: {type(e).__name__} - {e}{Style.RESET_ALL}"
                    )
                    failed_count += 1
                except Exception as e:
                    logger.error(
                        f"{Fore.RED}Order Cancel: Unexpected error for {order_info}: {e}{Style.RESET_ALL}"
                    )
                    failed_count += 1
            elif not order_id:
                logger.warning(
                    f"Order Cancel: Found open order without ID: {order}"
                )  # Should not happen
            else:
                logger.debug(
                    f"Skipping cancellation for non-open order: {order_info}"
                )  # Log skipped orders

        # Log summary of cancellation attempt
        log_level = logging.INFO if failed_count == 0 else logging.WARNING
        logger.log(
            log_level,
            f"{Fore.CYAN}Order Cancel: Finished for {symbol}. Attempted: {len(open_orders)}, Succeeded/Not Found: {cancelled_count}, Failed: {failed_count}.{Style.RESET_ALL}",
        )
        if failed_count > 0:
            send_sms_alert(
                f"[{market_base}] WARNING: Failed to cancel {failed_count} order(s) during {reason}. Check manually."
            )

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger.error(
            f"{Fore.RED}Order Cancel: Failed fetching open orders for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
    except Exception as e:
        logger.error(
            f"{Fore.RED}Order Cancel: Unexpected error during cancel process: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())


# --- Strategy Signal Generation ---
def generate_signals(df: pd.DataFrame, strategy_name: str) -> Dict[str, Any]:
    """Generates entry/exit signals based on the selected strategy and indicator columns in the DataFrame.
    Returns a dict: {'enter_long': bool, 'enter_short': bool, 'exit_long': bool, 'exit_short': bool, 'exit_reason': str}.
    Assumes indicator columns (e.g., 'trend', 'stochrsi_k') exist and contain appropriate types (bool, Decimal, NA).
    """
    signals = {
        "enter_long": False,
        "enter_short": False,
        "exit_long": False,
        "exit_short": False,
        "exit_reason": "Strategy Exit Signal",
    }
    required_rows = 2  # Need at least current and previous row for comparisons/crosses

    if df is None or len(df) < required_rows:
        logger.warning(
            f"Signal Gen ({strategy_name}): Insufficient data ({len(df) if df is not None else 0} rows, need {required_rows})"
        )
        return signals  # Not enough data to generate signals

    # Use .iloc for reliable access to last two rows
    last = df.iloc[-1]
    prev = df.iloc[-2]

    try:
        # --- Dual Supertrend Logic ---
        if strategy_name == "DUAL_SUPERTREND":
            # Check primary ST flip and confirmation ST trend alignment
            primary_flipped_long = last.get("st_long") is True
            primary_flipped_short = last.get("st_short") is True
            confirm_is_long = last.get("confirm_trend") is True
            confirm_is_short = (
                last.get("confirm_trend") is False
            )  # Explicitly check for False

            # Entry Long: Primary ST flips long AND confirmation ST is also long (or NA treated as permissive here, adjust if needed)
            if primary_flipped_long and confirm_is_long:
                signals["enter_long"] = True
            # Entry Short: Primary ST flips short AND confirmation ST is also short
            if primary_flipped_short and confirm_is_short:
                signals["enter_short"] = True

            # Exit Long: Primary ST flips short
            if primary_flipped_short:
                signals["exit_long"] = True
                signals["exit_reason"] = "Primary ST Short Flip"
            # Exit Short: Primary ST flips long
            if primary_flipped_long:
                signals["exit_short"] = True
                signals["exit_reason"] = "Primary ST Long Flip"

        # --- StochRSI + Momentum Logic ---
        elif strategy_name == "STOCHRSI_MOMENTUM":
            k_now, d_now, mom_now = (
                last.get("stochrsi_k"),
                last.get("stochrsi_d"),
                last.get("momentum"),
            )
            k_prev, d_prev = prev.get("stochrsi_k"), prev.get("stochrsi_d")

            # Check if all needed values are valid Decimals before proceeding
            if not all(
                isinstance(val, Decimal)
                for val in [k_now, d_now, mom_now, k_prev, d_prev]
            ):
                logger.debug(
                    "Signal Gen (StochRSI/Mom): Skipping due to missing/invalid indicator data (NA values)."
                )
                return signals  # Cannot generate signals if data is missing

            # Entry Long: K crosses above D FROM BELOW oversold level, with positive momentum confirmation
            if (
                k_prev <= d_prev
                and k_now > d_now
                and k_now < CONFIG.stochrsi_oversold
                and mom_now > CONFIG.position_qty_epsilon
            ):
                signals["enter_long"] = True
            # Entry Short: K crosses below D FROM ABOVE overbought level, with negative momentum confirmation
            if (
                k_prev >= d_prev
                and k_now < d_now
                and k_now > CONFIG.stochrsi_overbought
                and mom_now < -CONFIG.position_qty_epsilon
            ):
                signals["enter_short"] = True

            # Exit Long: K crosses below D (anywhere - simple exit signal)
            if k_prev >= d_prev and k_now < d_now:
                signals["exit_long"] = True
                signals["exit_reason"] = "StochRSI K crossed below D"
            # Exit Short: K crosses above D (anywhere - simple exit signal)
            if k_prev <= d_prev and k_now > d_now:
                signals["exit_short"] = True
                signals["exit_reason"] = "StochRSI K crossed above D"

        # --- Ehlers Fisher Logic ---
        elif strategy_name == "EHLERS_FISHER":
            fish_now, sig_now = last.get("ehlers_fisher"), last.get("ehlers_signal")
            fish_prev, sig_prev = prev.get("ehlers_fisher"), prev.get("ehlers_signal")

            # Check if Fisher line itself is valid
            if not isinstance(fish_now, Decimal) or not isinstance(fish_prev, Decimal):
                logger.debug(
                    "Signal Gen (EhlersFisher): Skipping due to missing/invalid Fisher line data (NA values)."
                )
                return signals

            # Determine if signal line is valid and should be used for crosses
            use_signal_cross = isinstance(sig_now, Decimal) and isinstance(
                sig_prev, Decimal
            )

            if use_signal_cross:
                # Strategy: Fisher crossing Signal line
                logger.debug(
                    "Signal Gen (EhlersFisher): Using Fisher/Signal cross strategy."
                )
                # Entry Long: Fisher crosses above Signal
                if fish_prev <= sig_prev and fish_now > sig_now:
                    signals["enter_long"] = True
                # Entry Short: Fisher crosses below Signal
                if fish_prev >= sig_prev and fish_now < sig_now:
                    signals["enter_short"] = True
                # Exit Long: Fisher crosses below Signal
                if fish_prev >= sig_prev and fish_now < sig_now:
                    signals["exit_long"] = True
                    signals["exit_reason"] = "Ehlers Fisher crossed below Signal"
                # Exit Short: Fisher crosses above Signal
                if fish_prev <= sig_prev and fish_now > sig_now:
                    signals["exit_short"] = True
                    signals["exit_reason"] = "Ehlers Fisher crossed above Signal"
            else:
                # Strategy: Fisher crossing zero (or previous value if preferred) - used when signal line invalid/disabled
                logger.debug(
                    "Signal Gen (EhlersFisher): Signal line invalid/disabled, using Fisher Zero-Cross strategy."
                )
                zero = Decimal("0.0")
                # Entry Long: Fisher crosses above zero
                if fish_prev <= zero and fish_now > zero:
                    signals["enter_long"] = True
                # Entry Short: Fisher crosses below zero
                if fish_prev >= zero and fish_now < zero:
                    signals["enter_short"] = True
                # Exit Long: Fisher crosses below zero
                if fish_prev >= zero and fish_now < zero:
                    signals["exit_long"] = True
                    signals["exit_reason"] = "Ehlers Fisher crossed below Zero"
                # Exit Short: Fisher crosses above zero
                if fish_prev <= zero and fish_now > zero:
                    signals["exit_short"] = True
                    signals["exit_reason"] = "Ehlers Fisher crossed above Zero"

        # --- Ehlers MA Cross Logic (Placeholder EMA Cross) ---
        elif strategy_name == "EHLERS_MA_CROSS":
            # Using the placeholder EMA columns calculated earlier
            fast_ma_now, slow_ma_now = last.get("fast_ema"), last.get("slow_ema")
            fast_ma_prev, slow_ma_prev = prev.get("fast_ema"), prev.get("slow_ema")

            # Check if all MA values are valid Decimals
            if not all(
                isinstance(val, Decimal)
                for val in [fast_ma_now, slow_ma_now, fast_ma_prev, slow_ma_prev]
            ):
                logger.debug(
                    "Signal Gen (EhlersMA Cross - Placeholder): Skipping due to missing/invalid MA data (NA values)."
                )
                return signals

            # Entry Long: Fast MA crosses above Slow MA
            if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now:
                signals["enter_long"] = True
            # Entry Short: Fast MA crosses below Slow MA
            if fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now:
                signals["enter_short"] = True

            # Exit Long: Fast MA crosses below Slow MA
            if fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now:
                signals["exit_long"] = True
                signals["exit_reason"] = "Fast MA crossed below Slow MA (Placeholder)"
            # Exit Short: Fast MA crosses above Slow MA
            if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now:
                signals["exit_short"] = True
                signals["exit_reason"] = "Fast MA crossed above Slow MA (Placeholder)"

    except KeyError as e:
        # This indicates a programming error - required indicator column wasn't calculated or named correctly
        logger.error(
            f"{Fore.RED}Signal Generation Error: Missing expected indicator column in DataFrame: {e}. Strategy '{strategy_name}' cannot run correctly.{Style.RESET_ALL}"
        )
        # Prevent signals if data is missing
        signals = {
            "enter_long": False,
            "enter_short": False,
            "exit_long": False,
            "exit_short": False,
            "exit_reason": "Missing Indicator Data",
        }
    except Exception as e:
        # Catch any other unexpected errors during signal logic
        logger.error(
            f"{Fore.RED}Signal Generation Error: Unexpected exception during signal calculation for strategy '{strategy_name}': {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        # Prevent signals on unexpected error
        signals = {
            "enter_long": False,
            "enter_short": False,
            "exit_long": False,
            "exit_short": False,
            "exit_reason": "Calculation Error",
        }

    return signals


# --- Trading Logic ---
# Global variable to track cycle count for logging
cycle_count: int = 0


def trade_logic(exchange: ccxt.Exchange, symbol: str, df: pd.DataFrame) -> None:
    """Executes the main trading logic for one cycle based on the selected strategy.
    1. Calculates indicators.
    2. Checks position status.
    3. Generates strategy signals.
    4. Handles exits based on signals.
    5. Handles entries based on signals and confirmations (Volume, Order Book).
    """
    global cycle_count  # Use the global counter
    cycle_time_str = (
        df.index[-1].strftime("%Y-%m-%d %H:%M:%S %Z") if not df.empty else "N/A"
    )
    market_base = get_market_base_currency(symbol)
    logger.info(
        f"{Fore.BLUE}{Style.BRIGHT}========== New Check Cycle [{cycle_count}] ({CONFIG.strategy_name}): {symbol} | Candle: {cycle_time_str} =========={Style.RESET_ALL}"
    )

    # --- Data Sufficiency Check ---
    # Determine minimum required rows based dynamically on the selected strategy's needs
    # This is more efficient than calculating based on *all* possible indicators
    required_lookbacks = [
        CONFIG.atr_calculation_period,
        CONFIG.volume_ma_period,
    ]  # Base lookbacks needed by all
    if CONFIG.strategy_name == "DUAL_SUPERTREND":
        required_lookbacks.extend([CONFIG.st_atr_length, CONFIG.confirm_st_atr_length])
    elif CONFIG.strategy_name == "STOCHRSI_MOMENTUM":
        required_lookbacks.extend(
            [
                CONFIG.stochrsi_rsi_length
                + CONFIG.stochrsi_stoch_length
                + CONFIG.stochrsi_k_period
                + CONFIG.stochrsi_d_period,  # Sum for full StochRSI calc depth
                CONFIG.momentum_length,
            ]
        )
    elif CONFIG.strategy_name == "EHLERS_FISHER":
        required_lookbacks.append(
            CONFIG.ehlers_fisher_length + CONFIG.ehlers_fisher_signal_length
        )
    elif CONFIG.strategy_name == "EHLERS_MA_CROSS":
        required_lookbacks.extend(
            [CONFIG.ehlers_fast_period, CONFIG.ehlers_slow_period]
        )

    # Need max lookback + 1 for current candle + buffer for calculation stability
    required_rows = (
        max(required_lookbacks) + 2
    )  # +2 = current candle + previous for comparisons

    if df is None or len(df) < required_rows:
        logger.warning(
            f"{Fore.YELLOW}Trade Logic: Insufficient data ({len(df) if df is not None else 0} rows, need ~{required_rows} for {CONFIG.strategy_name}). Skipping cycle.{Style.RESET_ALL}"
        )
        return

    action_taken_this_cycle: bool = False  # Track if an entry/exit occurred
    try:
        # === 1. Calculate Required Indicators ===
        logger.debug("Calculating required indicators...")
        # Always calculate ATR/Volume as they are used for SL and confirmation
        df, vol_atr_data = analyze_volume_atr(
            df, CONFIG.atr_calculation_period, CONFIG.volume_ma_period
        )
        current_atr = vol_atr_data.get("atr")

        # Calculate strategy-specific indicators
        if CONFIG.strategy_name == "DUAL_SUPERTREND":
            df = calculate_supertrend(df, CONFIG.st_atr_length, CONFIG.st_multiplier)
            df = calculate_supertrend(
                df,
                CONFIG.confirm_st_atr_length,
                CONFIG.confirm_st_multiplier,
                prefix="confirm_",
            )
        elif CONFIG.strategy_name == "STOCHRSI_MOMENTUM":
            df = calculate_stochrsi_momentum(
                df,
                CONFIG.stochrsi_rsi_length,
                CONFIG.stochrsi_stoch_length,
                CONFIG.stochrsi_k_period,
                CONFIG.stochrsi_d_period,
                CONFIG.momentum_length,
            )
        elif CONFIG.strategy_name == "EHLERS_FISHER":
            df = calculate_ehlers_fisher(
                df, CONFIG.ehlers_fisher_length, CONFIG.ehlers_fisher_signal_length
            )
        elif CONFIG.strategy_name == "EHLERS_MA_CROSS":
            df = calculate_ehlers_ma(
                df, CONFIG.ehlers_fast_period, CONFIG.ehlers_slow_period
            )  # Placeholder EMA

        # === 2. Validate Base Requirements for Trading ===
        last = df.iloc[-1]  # Get the most recent candle data
        current_price = safe_decimal_conversion(
            last.get("close")
        )  # Use .get() for safety
        if current_price <= 0:  # Check if price is valid
            logger.warning(
                f"{Fore.YELLOW}Trade Logic: Last candle close price is invalid or zero ({current_price}). Skipping cycle.{Style.RESET_ALL}"
            )
            return

        # Check if ATR is valid for SL calculation (needed for *new* entries)
        can_calculate_sl = (
            current_atr is not None and current_atr > CONFIG.position_qty_epsilon
        )
        if not can_calculate_sl:
            # Log warning but allow cycle to continue for potential position management/exits
            logger.warning(
                f"{Fore.YELLOW}Trade Logic: Invalid ATR ({current_atr}). Cannot calculate SL for new entries this cycle.{Style.RESET_ALL}"
            )
            # Note: Existing positions with SL/TSL already placed will continue to be managed by the exchange.

        # === 3. Get Position & Analyze Order Book (if needed) ===
        position = get_current_position(exchange, symbol)
        position_side = position["side"]
        position_qty = position["qty"]
        position_entry_price = position["entry_price"]

        # Fetch OB per cycle only if configured, otherwise fetch later if needed for entry confirmation
        ob_data: Optional[Dict[str, Optional[Decimal]]] = None
        if CONFIG.fetch_order_book_per_cycle:
            ob_data = analyze_order_book(
                exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit
            )

        # === 4. Log Current State ===
        vol_ratio = vol_atr_data.get("volume_ratio")
        vol_spike = vol_ratio is not None and vol_ratio > CONFIG.volume_spike_threshold
        bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None
        spread = ob_data.get("spread") if ob_data else None

        atr_log = f"{current_atr:.5f}" if current_atr is not None else "N/A"
        logger.info(
            f"State | Price: {current_price:.4f}, ATR({CONFIG.atr_calculation_period}): {atr_log}"
        )
        vol_ratio_log = f"{vol_ratio:.2f}" if vol_ratio is not None else "N/A"
        logger.info(
            f"State | Volume: Ratio={vol_ratio_log}, Spike={vol_spike} (EntryReq={CONFIG.require_volume_spike_for_entry})"
        )

        # Log key indicators for the *active* strategy
        strat_log = f"State | Strategy ({CONFIG.strategy_name}): "
        try:  # Add specific logging based on strategy, handle potential missing keys safely
            if CONFIG.strategy_name == "DUAL_SUPERTREND":
                st_trend = last.get("trend")
                st_confirm_trend = last.get("confirm_trend")
                st_log = f"ST Trend: {'Up' if st_trend is True else ('Down' if st_trend is False else 'NA')}"
                st_confirm_log = f"Confirm Trend: {'Up' if st_confirm_trend is True else ('Down' if st_confirm_trend is False else 'NA')}"
                strat_log += f"{st_log}, {st_confirm_log}"
            elif CONFIG.strategy_name == "STOCHRSI_MOMENTUM":
                k, d, mom = (
                    last.get("stochrsi_k"),
                    last.get("stochrsi_d"),
                    last.get("momentum"),
                )
                k_str = f"{k:.2f}" if isinstance(k, Decimal) else "NA"
                d_str = f"{d:.2f}" if isinstance(d, Decimal) else "NA"
                mom_str = f"{mom:.4f}" if isinstance(mom, Decimal) else "NA"
                strat_log += f"K={k_str}, D={d_str}, Mom={mom_str}"
            elif CONFIG.strategy_name == "EHLERS_FISHER":
                fish, sig = last.get("ehlers_fisher"), last.get("ehlers_signal")
                fish_str = f"{fish:.4f}" if isinstance(fish, Decimal) else "NA"
                sig_str = f"{sig:.4f}" if isinstance(sig, Decimal) else "NA"
                strat_log += f"Fisher={fish_str}, Signal={sig_str}"
            elif CONFIG.strategy_name == "EHLERS_MA_CROSS":
                fast, slow = last.get("fast_ema"), last.get("slow_ema")
                fast_str = f"{fast:.4f}" if isinstance(fast, Decimal) else "NA"
                slow_str = f"{slow:.4f}" if isinstance(slow, Decimal) else "NA"
                strat_log += f"FastMA={fast_str}, SlowMA={slow_str} (Placeholder)"
            logger.info(strat_log)
        except Exception as log_err:
            logger.warning(f"Could not log strategy indicators: {log_err}")

        ob_ratio_log = f"{bid_ask_ratio:.3f}" if bid_ask_ratio is not None else "N/A"
        spread_log = f"{spread:.4f}" if spread is not None else "N/A"
        logger.info(
            f"State | OrderBook: Ratio={ob_ratio_log}, Spread={spread_log} (Fetched={ob_data is not None})"
        )
        logger.info(
            f"State | Position: Side={position_side}, Qty={position_qty:.8f}, Entry={position_entry_price:.4f}"
        )

        # === 5. Generate Strategy Signals ===
        # Pass a copy of the DataFrame to prevent potential mutation issues if signals modify it
        strategy_signals = generate_signals(df.copy(), CONFIG.strategy_name)
        logger.debug(
            f"Strategy Signals ({CONFIG.strategy_name}): EnterLong={strategy_signals['enter_long']}, EnterShort={strategy_signals['enter_short']}, ExitLong={strategy_signals['exit_long']}, ExitShort={strategy_signals['exit_short']}, Reason='{strategy_signals['exit_reason']}'"
        )

        # === 6. Execute Exit Actions (If in Position) ===
        if position_side != CONFIG.pos_none:
            should_exit = (
                position_side == CONFIG.pos_long and strategy_signals["exit_long"]
            ) or (position_side == CONFIG.pos_short and strategy_signals["exit_short"])

            if should_exit:
                exit_reason = strategy_signals["exit_reason"]
                logger.warning(
                    f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}*** STRATEGY EXIT SIGNAL: Closing {position_side} position due to '{exit_reason}' ***{Style.RESET_ALL}"
                )
                # --- Critical Step: Cancel existing SL/TSL orders BEFORE sending the closing market order ---
                # This prevents the SL/TSL from potentially triggering simultaneously or conflicting.
                cancel_open_orders(
                    exchange, symbol, reason=f"Cancel SL/TSL before {exit_reason} Exit"
                )
                time.sleep(
                    0.75
                )  # Brief pause after cancel request before sending close order

                # Attempt to close the position using the re-validated position details
                close_result = close_position(
                    exchange, symbol, position, reason=exit_reason
                )

                if close_result:
                    action_taken_this_cycle = True
                    logger.info(
                        f"Pausing for {CONFIG.post_close_delay_seconds}s after closing position..."
                    )
                    time.sleep(
                        CONFIG.post_close_delay_seconds
                    )  # Pause to allow state settlement
                else:
                    # This is a potential problem - the strategy indicated exit, but closure failed.
                    logger.error(
                        f"{Fore.RED}{Style.BRIGHT}Failed to execute close order for {position_side} exit signal! Manual check required. Position might still be open.{Style.RESET_ALL}"
                    )
                    send_sms_alert(
                        f"[{market_base}] CRITICAL: Failed to CLOSE {position_side} on signal '{exit_reason}'! Check position!"
                    )
                # Exit the current trade logic cycle after attempting close (successful or not)
                # Avoids trying to enter immediately after an exit signal/attempt.
                return
            else:
                # Still in position, no strategy exit signal this candle
                logger.info(
                    f"Holding {position_side} position. No strategy exit signal. Relying on exchange-managed SL/TSL."
                )
                # No further action needed this cycle if holding.
                return

        # === 7. Check & Execute Entry Actions (Only if Flat) ===
        # If code execution reaches here, it means position_side == CONFIG.pos_none

        # Check if we can place new orders (ATR must be valid for SL calc)
        if not can_calculate_sl:
            logger.warning(
                f"{Fore.YELLOW}Holding Cash. Cannot evaluate new entries because ATR is invalid for SL calculation this cycle.{Style.RESET_ALL}"
            )
            return  # Skip entry checks if SL cannot be calculated

        logger.debug("Checking entry signals (currently flat)...")
        enter_long_signal = strategy_signals["enter_long"]
        enter_short_signal = strategy_signals["enter_short"]
        potential_entry = enter_long_signal or enter_short_signal

        if not potential_entry:
            logger.info("No entry signal generated by strategy. Holding cash.")
            return  # No signal, do nothing

        # --- A potential entry signal exists, check confirmation conditions ---

        # Fetch OB now if not fetched per cycle and a potential entry exists
        if (
            not CONFIG.fetch_order_book_per_cycle
            and potential_entry
            and ob_data is None
        ):
            logger.debug(
                "Potential entry signal detected, fetching Order Book for confirmation..."
            )
            ob_data = analyze_order_book(
                exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit
            )
            # Update local variables if OB was fetched on demand
            bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None
            spread = ob_data.get("spread") if ob_data else None
            ob_ratio_log = (
                f"{bid_ask_ratio:.3f}" if bid_ask_ratio is not None else "N/A"
            )
            spread_log = f"{spread:.4f}" if spread is not None else "N/A"
            logger.info(
                f"State Update | OrderBook (On Demand): Ratio={ob_ratio_log}, Spread={spread_log}"
            )

        # Evaluate Order Book confirmation
        ob_available = ob_data is not None and bid_ask_ratio is not None
        # Confirmation logic: Pass if OB not needed, or if available AND meets threshold
        passes_long_ob = not ob_available or (
            bid_ask_ratio >= CONFIG.order_book_ratio_threshold_long
        )
        passes_short_ob = not ob_available or (
            bid_ask_ratio <= CONFIG.order_book_ratio_threshold_short
        )
        # If OB *is* required for confirmation (e.g., could add a config flag 'REQUIRE_OB_CONFIRMATION'),
        # then failure should block entry. For now, treat missing OB as permissive.
        # Refined logic: Check if OB confirmation is *desired* (always true for now)
        ob_confirm_needed = True  # Can be made configurable
        passes_long_ob_final = not ob_confirm_needed or (
            ob_available and passes_long_ob
        )
        passes_short_ob_final = not ob_confirm_needed or (
            ob_available and passes_short_ob
        )
        ob_log = f"OB Check: Needed={ob_confirm_needed}, Avail={ob_available}, Ratio={ob_ratio_log} -> LongOK={passes_long_ob_final}, ShortOK={passes_short_ob_final}"
        logger.debug(ob_log)

        # Evaluate Volume confirmation
        vol_confirm_needed = CONFIG.require_volume_spike_for_entry
        passes_volume = (
            not vol_confirm_needed or vol_spike
        )  # Passes if not needed OR spike occurred
        vol_log = f"Vol Check: Needed={vol_confirm_needed}, SpikeMet={vol_spike} -> OK={passes_volume}"
        logger.debug(vol_log)

        # --- Combine Strategy Signal with Confirmations ---
        execute_long_entry = (
            enter_long_signal and passes_long_ob_final and passes_volume
        )
        execute_short_entry = (
            enter_short_signal and passes_short_ob_final and passes_volume
        )

        logger.info(
            f"Final Entry Decision | Long: Signal={enter_long_signal}, OB OK={passes_long_ob_final}, Vol OK={passes_volume} => EXECUTE={execute_long_entry}"
        )
        logger.info(
            f"Final Entry Decision | Short: Signal={enter_short_signal}, OB OK={passes_short_ob_final}, Vol OK={passes_volume} => EXECUTE={execute_short_entry}"
        )

        # --- Execute Entry ---
        entry_side: Optional[str] = None
        if execute_long_entry:
            entry_side = CONFIG.side_buy
            logger.success(
                f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}*** TRADE SIGNAL: CONFIRMED LONG ENTRY ({CONFIG.strategy_name}) for {symbol} ***{Style.RESET_ALL}"
            )
        elif execute_short_entry:
            entry_side = CONFIG.side_sell
            logger.success(
                f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}*** TRADE SIGNAL: CONFIRMED SHORT ENTRY ({CONFIG.strategy_name}) for {symbol} ***{Style.RESET_ALL}"
            )

        if entry_side:
            # Cancel any lingering orders (shouldn't exist if flat, but as a safety check)
            cancel_open_orders(exchange, symbol, f"Before {entry_side.upper()} Entry")
            time.sleep(0.5)  # Pause after cancel before placing new order

            # Place the fully managed order (entry + SL + TSL)
            place_result = place_risked_market_order(
                exchange=exchange,
                symbol=symbol,
                side=entry_side,
                risk_percentage=CONFIG.risk_per_trade_percentage,
                current_atr=current_atr,  # Pass the validated ATR
                sl_atr_multiplier=CONFIG.atr_stop_loss_multiplier,
                leverage=CONFIG.leverage,
                max_order_cap_usdt=CONFIG.max_order_usdt_amount,
                margin_check_buffer=CONFIG.required_margin_buffer,
                tsl_percent=CONFIG.trailing_stop_percentage,
                tsl_activation_offset_percent=CONFIG.trailing_stop_activation_offset_percent,
            )
            if place_result:
                action_taken_this_cycle = True
                # No need to sleep after entry, next cycle will handle monitoring/exits
            else:
                # Log if placement failed after confirmation passed
                logger.error(
                    f"{Fore.RED}Entry placement failed for {entry_side.upper()} despite confirmed signal. See logs from place_risked_market_order.{Style.RESET_ALL}"
                )
                # SMS alert for failed placement is handled within place_risked_market_order

        else:
            # Log if there was a signal but confirmations failed
            if potential_entry and not (execute_long_entry or execute_short_entry):
                reason = []
                if enter_long_signal and not passes_long_ob_final:
                    reason.append("OB Long")
                if enter_short_signal and not passes_short_ob_final:
                    reason.append("OB Short")
                if not passes_volume and vol_confirm_needed:
                    reason.append("Volume")
                logger.warning(
                    f"{Fore.YELLOW}Entry signal received but FAILED confirmation checks ({', '.join(reason)}). Holding cash.{Style.RESET_ALL}"
                )
            # If no action was taken and no warning logged, implies no confirmed signal
            elif not action_taken_this_cycle:
                logger.info("No confirmed entry signal this cycle. Holding cash.")

    except Exception as e:
        # Catch-all for unexpected errors within the main trade logic block
        logger.error(
            f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}CRITICAL UNEXPECTED ERROR in trade_logic cycle {cycle_count}: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        send_sms_alert(
            f"[{market_base}] CRITICAL ERROR in trade_logic cycle {cycle_count}: {type(e).__name__}. Check logs!"
        )
    finally:
        # Log cycle end regardless of outcome
        logger.info(
            f"{Fore.BLUE}{Style.BRIGHT}========== Cycle Check End [{cycle_count}]: {symbol} =========={Style.RESET_ALL}\n"
        )


# --- Graceful Shutdown ---
def graceful_shutdown(exchange: Optional[ccxt.Exchange], symbol: Optional[str]) -> None:
    """Attempts to close any open position and cancel all orders before exiting."""
    logger.warning(
        f"\n{Fore.YELLOW}{Style.BRIGHT}Shutdown requested. Initiating graceful exit sequence...{Style.RESET_ALL}"
    )
    market_base = get_market_base_currency(symbol) if symbol else "Bot"
    send_sms_alert(f"[{market_base}] Shutdown initiated. Attempting cleanup...")

    if not exchange or not symbol:
        logger.warning(
            f"{Fore.YELLOW}Shutdown: Exchange instance or symbol not available. Cannot perform API cleanup.{Style.RESET_ALL}"
        )
        logger.info(
            f"{Fore.YELLOW}{Style.BRIGHT}--- Pyrmethus Shutdown Sequence Complete (No API Actions) ---{Style.RESET_ALL}"
        )
        return

    try:
        # === Step 1: Cancel All Open Orders First ===
        # This prevents SL/TSL orders from potentially triggering while we try to close the position manually.
        logger.info("Shutdown: Attempting to cancel all open orders...")
        cancel_open_orders(exchange, symbol, reason="Graceful Shutdown")
        time.sleep(
            1.5
        )  # Allow time for cancellations to be processed by the exchange API

        # === Step 2: Check Current Position Status ===
        logger.info("Shutdown: Checking for active position...")
        # Use get_current_position which handles V5 specifics
        position = get_current_position(exchange, symbol)

        # === Step 3: Close Position if Active ===
        if position["side"] != CONFIG.pos_none:
            logger.warning(
                f"{Fore.YELLOW}Shutdown: Active {position['side']} position found (Qty: {position['qty']:.8f}). Attempting market close...{Style.RESET_ALL}"
            )
            # Use the close_position function which includes reduceOnly=True and V5 logic
            close_result = close_position(
                exchange, symbol, position, reason="Shutdown Request"
            )

            if close_result:
                logger.info(
                    f"{Fore.CYAN}Shutdown: Close order placed. Waiting {CONFIG.post_close_delay_seconds * 2}s for confirmation...{Style.RESET_ALL}"
                )
                time.sleep(
                    CONFIG.post_close_delay_seconds * 2
                )  # Wait longer to allow market closure and state update

                # === Step 4: Final Confirmation Check ===
                logger.info("Shutdown: Performing final position check...")
                final_pos = get_current_position(exchange, symbol)
                if final_pos["side"] == CONFIG.pos_none:
                    logger.success(
                        f"{Fore.GREEN}{Style.BRIGHT}Shutdown: Position successfully confirmed CLOSED.{Style.RESET_ALL}"
                    )
                    send_sms_alert(
                        f"[{market_base}] Position confirmed CLOSED during shutdown."
                    )
                else:
                    # This is a critical situation - manual intervention needed!
                    logger.error(
                        f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Shutdown Error: FAILED TO CONFIRM position closure after placing order. "
                        f"Final state check shows: {final_pos['side']} Qty={final_pos['qty']:.8f}. MANUAL INTERVENTION REQUIRED!{Style.RESET_ALL}"
                    )
                    send_sms_alert(
                        f"[{market_base}] CRITICAL ERROR: Failed to CONFIRM closure on shutdown! Final: {final_pos['side']} Qty={final_pos['qty']:.8f}. CHECK MANUALLY!"
                    )
            else:
                # Failed to even place the close order
                logger.error(
                    f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Shutdown Error: Failed to PLACE close order for active position ({position['side']} Qty: {position['qty']:.8f}). MANUAL INTERVENTION REQUIRED!{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"[{market_base}] CRITICAL ERROR: Failed to PLACE close order on shutdown. CHECK MANUALLY!"
                )
        else:
            # No position was found after cancelling orders
            logger.info(
                f"{Fore.GREEN}Shutdown: No active position found. No closure needed.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}] No active position found during shutdown cleanup."
            )

    except Exception as e:
        # Catch any errors during the shutdown API calls
        logger.error(
            f"{Fore.RED}Shutdown: Error during cleanup sequence: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        send_sms_alert(
            f"[{market_base}] Error during shutdown cleanup: {type(e).__name__}. Manual check advised."
        )
    finally:
        # Log completion of the shutdown sequence
        logger.info(
            f"{Fore.YELLOW}{Style.BRIGHT}--- Pyrmethus Shutdown Sequence Complete ---{Style.RESET_ALL}"
        )


# --- Main Execution ---
def main() -> None:
    """Main function to initialize, set up, and run the trading loop."""
    global cycle_count  # Use the global counter defined outside
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    logger.info(
        f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Bybit Scalping Spell v2.1.1 Initializing ({start_time_str}) ---{Style.RESET_ALL}"
    )
    logger.info(
        f"{Fore.CYAN}--- Strategy Enchantment Selected: {CONFIG.strategy_name} ---{Style.RESET_ALL}"
    )
    logger.info(
        f"{Fore.GREEN}--- Warding Runes Active: Initial ATR Stop Loss + Exchange Native Trailing Stop Loss ---{Style.RESET_ALL}"
    )
    logger.warning(
        f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}--- !!! LIVE FUTURES SCALPING MODE ENGAGED - EXTREME RISK INVOLVED - USE CAUTION !!! ---{Style.RESET_ALL}"
    )

    exchange: Optional[ccxt.Exchange] = None
    symbol: Optional[str] = None
    run_bot: bool = True  # Flag to control the main loop

    try:
        # === Initialization ===
        exchange = initialize_exchange()
        if not exchange:
            logger.critical("Exchange initialization failed. Exiting.")
            # SMS alert handled within initialize_exchange()
            return  # Exit if exchange setup fails

        # === Symbol Validation and Leverage Setup ===
        try:
            # Use configured symbol, ensure it's loaded and is a futures contract
            symbol_to_use = CONFIG.symbol
            logger.info(f"Validating and setting up symbol: {symbol_to_use}")
            market = exchange.market(
                symbol_to_use
            )  # Raises BadSymbol if not found/loaded
            symbol = market[
                "symbol"
            ]  # Use the unified symbol from CCXT (e.g., BTC/USDT:USDT)
            market_base = get_market_base_currency(symbol)  # For concise alerts

            # Verify it's a futures/contract market
            if not market.get("contract"):
                raise ValueError(
                    f"Market '{symbol}' is not a contract/futures market. Check SYMBOL config."
                )
            logger.info(
                f"{Fore.GREEN}Using Symbol: {symbol} (Type: {market.get('type', 'N/A')}, Exchange ID: {market.get('id')}){Style.RESET_ALL}"
            )

            # Set leverage (crucial for futures, check for success)
            if not set_leverage(exchange, symbol, CONFIG.leverage):
                # set_leverage logs errors and sends SMS on failure
                raise RuntimeError(f"Leverage setup failed for {symbol} after retries.")

        except (ccxt.BadSymbol, KeyError, ValueError, RuntimeError) as e:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}Symbol/Leverage setup failed for '{CONFIG.symbol}': {e}{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[Pyrmethus] CRITICAL: Symbol/Leverage setup FAILED ({e}). Exiting."
            )
            return  # Exit if symbol/leverage setup fails
        except Exception as e:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}Unexpected error during symbol/leverage setup: {e}{Style.RESET_ALL}"
            )
            logger.debug(traceback.format_exc())
            send_sms_alert("[Pyrmethus] CRITICAL: Unexpected setup error. Exiting.")
            return

        # === Log Configuration Summary ===
        logger.info(f"{Fore.MAGENTA}--- Configuration Summary ---{Style.RESET_ALL}")
        logger.info(
            f"  Symbol: {symbol}, Interval: {CONFIG.interval}, Leverage: {CONFIG.leverage}x"
        )
        logger.info(f"  Strategy: {CONFIG.strategy_name}")
        # Log relevant strategy params concisely
        if CONFIG.strategy_name == "DUAL_SUPERTREND":
            logger.info(
                f"    Params: ST={CONFIG.st_atr_length}/{CONFIG.st_multiplier}, ConfirmST={CONFIG.confirm_st_atr_length}/{CONFIG.confirm_st_multiplier}"
            )
        elif CONFIG.strategy_name == "STOCHRSI_MOMENTUM":
            logger.info(
                f"    Params: StochRSI={CONFIG.stochrsi_rsi_length}/{CONFIG.stochrsi_stoch_length}/{CONFIG.stochrsi_k_period}/{CONFIG.stochrsi_d_period} (OB={CONFIG.stochrsi_overbought},OS={CONFIG.stochrsi_oversold}), Mom={CONFIG.momentum_length}"
            )
        elif CONFIG.strategy_name == "EHLERS_FISHER":
            logger.info(
                f"    Params: Fisher={CONFIG.ehlers_fisher_length}, Signal={CONFIG.ehlers_fisher_signal_length}"
            )
        elif CONFIG.strategy_name == "EHLERS_MA_CROSS":
            logger.info(
                f"    Params: FastMA={CONFIG.ehlers_fast_period}, SlowMA={CONFIG.ehlers_slow_period} (Placeholder EMA)"
            )
        logger.info(f"  Risk: {CONFIG.risk_per_trade_percentage:.3%} per trade")
        logger.info(
            f"  Sizing: MaxPosValue={CONFIG.max_order_usdt_amount:.2f} USDT, MarginBuffer={CONFIG.required_margin_buffer:.1%}"
        )
        logger.info(
            f"  Initial SL: {CONFIG.atr_stop_loss_multiplier} * ATR({CONFIG.atr_calculation_period})"
        )
        logger.info(
            f"  Trailing SL: {CONFIG.trailing_stop_percentage:.2%} (Trail Dist), {CONFIG.trailing_stop_activation_offset_percent:.2%} (Activation Offset from Entry)"
        )
        logger.info("  Confirmations:")
        logger.info(
            f"    Vol Spike: Required={CONFIG.require_volume_spike_for_entry} (MA={CONFIG.volume_ma_period}, Threshold={CONFIG.volume_spike_threshold}x)"
        )
        logger.info(
            f"    Order Book: FetchPerCycle={CONFIG.fetch_order_book_per_cycle}, Depth={CONFIG.order_book_depth}, Long Ratio>={CONFIG.order_book_ratio_threshold_long}, Short Ratio<={CONFIG.order_book_ratio_threshold_short}"
        )
        logger.info(
            f"  Timing: Sleep={CONFIG.sleep_seconds}s, FillTimeout={CONFIG.order_fill_timeout_seconds}s, PostCloseDelay={CONFIG.post_close_delay_seconds}s"
        )
        logger.info(
            f"  Alerts: SMS Enabled={CONFIG.enable_sms_alerts} (To: {CONFIG.sms_recipient_number or 'Not Set'})"
        )
        logger.info(f"  Logging Level: {logging.getLevelName(logger.level)}")
        logger.info(f"{Fore.MAGENTA}{'-' * 30}{Style.RESET_ALL}")

        # Send startup confirmation SMS
        send_sms_alert(
            f"[{market_base}] Pyrmethus Bot started. Strategy: {CONFIG.strategy_name}. SL: ATR+TSL. Risk: {CONFIG.risk_per_trade_percentage:.2%}. Live Trading Active!"
        )

        # === Main Trading Loop ===
        logger.info(
            f"{Fore.GREEN}{Style.BRIGHT}--- Starting Main Trading Loop ---{Style.RESET_ALL}"
        )
        while run_bot:
            cycle_start_time = time.monotonic()
            cycle_count += 1
            logger.debug(
                f"{Fore.CYAN}--- Cycle {cycle_count} Start ---{Style.RESET_ALL}"
            )

            try:
                # --- Calculate Data Limit ---
                # Determine required data length dynamically based on the *active* strategy + base needs
                active_lookbacks = [
                    CONFIG.atr_calculation_period,
                    CONFIG.volume_ma_period,
                ]
                if CONFIG.strategy_name == "DUAL_SUPERTREND":
                    active_lookbacks.extend(
                        [CONFIG.st_atr_length, CONFIG.confirm_st_atr_length]
                    )
                elif CONFIG.strategy_name == "STOCHRSI_MOMENTUM":
                    active_lookbacks.extend(
                        [
                            CONFIG.stochrsi_rsi_length
                            + CONFIG.stochrsi_stoch_length
                            + CONFIG.stochrsi_k_period
                            + CONFIG.stochrsi_d_period,
                            CONFIG.momentum_length,
                        ]
                    )
                elif CONFIG.strategy_name == "EHLERS_FISHER":
                    active_lookbacks.append(
                        CONFIG.ehlers_fisher_length + CONFIG.ehlers_fisher_signal_length
                    )
                elif CONFIG.strategy_name == "EHLERS_MA_CROSS":
                    active_lookbacks.extend(
                        [CONFIG.ehlers_fast_period, CONFIG.ehlers_slow_period]
                    )
                # Fetch enough data for the longest lookback + buffer for calculations + previous candle access
                data_limit = (
                    max(active_lookbacks) + CONFIG.api_fetch_limit_buffer + 2
                )  # +2 for prev candle access

                # --- Fetch Market Data ---
                df = get_market_data(
                    exchange, symbol, CONFIG.interval, limit=data_limit
                )

                if df is not None and not df.empty:
                    # --- Execute Trade Logic ---
                    # Pass essential objects to the trade logic function
                    trade_logic(
                        exchange, symbol, df
                    )  # df is already a copy if needed internally
                else:
                    # Handle case where data fetching failed
                    logger.warning(
                        f"{Fore.YELLOW}No valid market data returned for {symbol} ({CONFIG.interval}) in cycle {cycle_count}. Skipping trade logic. Check connection or symbol validity.{Style.RESET_ALL}"
                    )
                    # Optional: Add a longer sleep here if data fetching fails repeatedly to avoid spamming API
                    time.sleep(CONFIG.sleep_seconds * 2)  # Example: Double sleep time

            # --- Robust Error Handling within the Loop ---
            except ccxt.RateLimitExceeded as e:
                logger.warning(
                    f"{Back.YELLOW}{Fore.BLACK}Rate Limit Exceeded: {e}. Sleeping for {CONFIG.sleep_seconds * 6}s...{Style.RESET_ALL}"
                )
                time.sleep(CONFIG.sleep_seconds * 6)  # Longer sleep for rate limits
                send_sms_alert(f"[{market_base}] WARNING: Rate limit hit! Pausing.")
            except ccxt.NetworkError as e:
                logger.warning(
                    f"{Fore.YELLOW}Network Error in main loop: {e}. Check connection. Will retry next cycle.{Style.RESET_ALL}"
                )
                # Consider a slightly longer sleep or a connection check helper function if this persists
                time.sleep(
                    CONFIG.sleep_seconds * 1.5
                )  # Slightly longer sleep on network errors
            except ccxt.ExchangeNotAvailable as e:
                # Exchange maintenance or temporary outage
                logger.error(
                    f"{Back.RED}{Fore.WHITE}Exchange Not Available: {e}. Sleeping for {CONFIG.sleep_seconds * 10}s... Check exchange status.{Style.RESET_ALL}"
                )
                time.sleep(CONFIG.sleep_seconds * 10)  # Significantly longer sleep
                send_sms_alert(f"[{market_base}] ERROR: Exchange unavailable! Paused.")
            except ccxt.AuthenticationError as e:
                # This is critical and should stop the bot immediately
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Authentication Error during operation: {e}. Invalid API keys or permissions? Stopping bot NOW.{Style.RESET_ALL}"
                )
                run_bot = False  # Signal loop termination
                send_sms_alert(
                    f"[{market_base}] CRITICAL: Authentication Error! Bot stopping NOW."
                )
            except (
                ccxt.ExchangeError
            ) as e:  # Catch other specific exchange errors not handled above
                logger.error(
                    f"{Fore.RED}Unhandled Exchange Error in main loop cycle {cycle_count}: {e}{Style.RESET_ALL}"
                )
                logger.debug(f"Exchange Error Details: {traceback.format_exc()}")
                send_sms_alert(
                    f"[{market_base}] ERROR: Unhandled Exchange error: {type(e).__name__}. Check logs."
                )
                time.sleep(CONFIG.sleep_seconds * 1.5)  # Sleep before retrying
            except Exception as e:
                # Catch-all for truly unexpected issues within the loop
                logger.exception(
                    f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}!!! UNEXPECTED CRITICAL ERROR IN MAIN LOOP (Cycle {cycle_count}): {e} !!!{Style.RESET_ALL}"
                )
                run_bot = False  # Stop bot on unknown critical errors to prevent unintended actions
                send_sms_alert(
                    f"[{market_base}] CRITICAL UNEXPECTED ERROR: {type(e).__name__}! Bot stopping NOW."
                )

            # --- Loop Delay Calculation ---
            if run_bot:  # Only sleep if the bot hasn't been signaled to stop
                elapsed = time.monotonic() - cycle_start_time
                sleep_duration = max(0, CONFIG.sleep_seconds - elapsed)
                logger.debug(
                    f"Cycle {cycle_count} processed in {elapsed:.2f}s. Sleeping for {sleep_duration:.2f}s."
                )
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

    except KeyboardInterrupt:
        logger.warning(
            f"\n{Fore.YELLOW}{Style.BRIGHT}KeyboardInterrupt received. Initiating graceful shutdown...{Style.RESET_ALL}"
        )
        run_bot = False  # Ensure loop terminates cleanly if somehow still running
    except Exception as e:
        # Catch errors during the initial setup phase (before main loop starts)
        logger.critical(
            f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Critical error during bot initialization phase: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        # Try sending SMS even if config might be partially loaded
        send_sms_alert(
            f"[Pyrmethus] CRITICAL SETUP ERROR: {type(e).__name__}! Bot failed to start."
        )
        run_bot = False  # Ensure shutdown sequence runs even if loop never started
    finally:
        # --- Graceful Shutdown Sequence ---
        # This block executes regardless of how the loop was exited (normal stop, error, interrupt)
        graceful_shutdown(exchange, symbol)  # Pass initialized objects

        # Final termination message
        market_base_final = get_market_base_currency(symbol) if symbol else "Bot"
        send_sms_alert(f"[{market_base_final}] Pyrmethus bot process terminated.")
        logger.info(
            f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Scalping Spell Deactivated ({time.strftime('%Y-%m-%d %H:%M:%S %Z')}) ---{Style.RESET_ALL}"
        )


if __name__ == "__main__":
    # Entry point of the script
    main()
