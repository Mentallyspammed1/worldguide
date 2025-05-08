#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v2.0.1 (Enhanced Robustness & Clarity)
# Conjures high-frequency trades on Bybit Futures with enhanced precision and adaptable strategies.

"""High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 2.0.1 (Unified: Selectable Strategies + Precision + Native SL/TSL + Enhancements)

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
from decimal import (ROUND_HALF_UP, Decimal, DivisionByZero, InvalidOperation,
                     getcontext)
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
    missing_pkg = getattr(e, 'name', 'Unknown')
    print(f"Error: Missing required package '{missing_pkg}'. Please install it (e.g., pip install {missing_pkg})")
    sys.exit(1)

# --- Initializations ---
colorama_init(autoreset=True)
load_dotenv()
getcontext().prec = 18  # Set Decimal precision (adjust as needed for higher precision assets)


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
            critical_msg = f"Invalid STRATEGY_NAME '{self.strategy_name}'. Valid options are: {self.valid_strategies}"
            logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}{critical_msg}{Style.RESET_ALL}")
            raise ValueError(critical_msg)
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
            TypeError: If the default value cannot be cast to the specified type (internal check).
        """
        value_str = os.getenv(key)
        value = None
        log_source = ""
        has_error = False

        if value_str is not None:
            log_source = f"(from env: '{value_str}')"
            try:
                if cast_type == bool:
                    value = value_str.lower() in ['true', '1', 'yes', 'y']
                elif cast_type == Decimal:
                    # Ensure string conversion first for accurate Decimal
                    value = Decimal(str(value_str))
                elif cast_type is not None:
                    value = cast_type(value_str)
                else:
                    value = value_str  # Keep as string if cast_type is None
            except (ValueError, TypeError, InvalidOperation) as e:
                logger.error(f"{Fore.RED}Config Error: Invalid type/value for {key}: '{value_str}'. Expected {cast_type.__name__}. Error: {e}. Using default: '{default}'{Style.RESET_ALL}")
                value = default  # Fallback to default on casting error
                log_source = f"(env parse error, using default: '{default}')"
                has_error = True
        else:
            value = default
            log_source = f"(not set, using default: '{default}')" if default is not None else "(not set, no default)"

        # Validate default value type if used
        if value is None and not has_error and default is not None:
            try:
                if cast_type == bool and not isinstance(default, bool):
                    raise TypeError("Default is not bool")
                if cast_type == Decimal and not isinstance(default, Decimal):
                     # Attempt to convert default to Decimal if possible
                     value = Decimal(str(default))
                elif cast_type not in (bool, Decimal) and not isinstance(default, cast_type):
                    raise TypeError(f"Default is not {cast_type.__name__}")
            except (ValueError, TypeError, InvalidOperation) as e:
                critical_msg = f"Config Error: Default value '{default}' for key '{key}' is incompatible with type {cast_type.__name__}. Error: {e}"
                logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}{critical_msg}{Style.RESET_ALL}")
                raise TypeError(critical_msg) from e

        # Final check for required variables
        if value is None and required:
            critical_msg = f"CRITICAL: Required environment variable '{key}' not set and no default value provided."
            logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}{critical_msg}{Style.RESET_ALL}")
            raise ValueError(critical_msg)

        logger.debug(f"{color}Config {key}: {value} {log_source}{Style.RESET_ALL}")
        return value


# --- Global Objects ---
try:
    CONFIG = Config()
except (ValueError, TypeError) as e: # Catch config errors
    logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Configuration Error: {e}{Style.RESET_ALL}")
    sys.exit(1)
except Exception as e: # Catch unexpected errors during init
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
        The converted Decimal value or the default. Returns default also if input is None.
    """
    if value is None:
        return default
    try:
        # Explicitly convert to string first to handle floats accurately and avoid precision issues
        # Strip potential whitespace
        str_value = str(value).strip()
        # Handle edge case of empty string after stripping
        if not str_value:
            return default
        return Decimal(str_value)
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

    Handles potential input types and CCXT errors gracefully.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT:USDT').
        price: The price value to format.

    Returns:
        The price formatted as a string according to market rules, or a normalized string
        representation of the input Decimal/string on error.
    """
    try:
        # CCXT formatting methods generally expect float input. Convert Decimal safely.
        price_float = float(price)
        return exchange.price_to_precision(symbol, price_float)
    except (ValueError, TypeError, OverflowError) as e:
        logger.error(f"{Fore.RED}Error converting price '{price}' to float for formatting: {e}{Style.RESET_ALL}")
    except (ccxt.ExchangeError, ccxt.BadSymbol, Exception) as e:
        logger.error(f"{Fore.RED}CCXT Error formatting price '{price}' for {symbol}: {e}{Style.RESET_ALL}")

    # Fallback: Attempt to normalize Decimal representation or return original string
    try:
        # Convert to Decimal first for consistent handling
        dec_price = Decimal(str(price))
        # normalize() removes trailing zeros, to_eng_string() avoids scientific notation
        return dec_price.normalize().to_eng_string()
    except (InvalidOperation, TypeError, ValueError):
        logger.warning(f"Could not format price '{price}' using Decimal fallback, returning as string.")
        return str(price) # Absolute fallback


def format_amount(exchange: ccxt.Exchange, symbol: str, amount: float | Decimal | str) -> str:
    """Formats an amount (quantity) according to the market's precision rules using CCXT.

    Handles potential input types and CCXT errors gracefully.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT:USDT').
        amount: The amount value to format.

    Returns:
        The amount formatted as a string according to market rules, or a normalized string
        representation of the input Decimal/string on error.
    """
    try:
        # CCXT formatting methods generally expect float input. Convert Decimal safely.
        amount_float = float(amount)
        return exchange.amount_to_precision(symbol, amount_float)
    except (ValueError, TypeError, OverflowError) as e:
        logger.error(f"{Fore.RED}Error converting amount '{amount}' to float for formatting: {e}{Style.RESET_ALL}")
    except (ccxt.ExchangeError, ccxt.BadSymbol, Exception) as e:
        logger.error(f"{Fore.RED}CCXT Error formatting amount '{amount}' for {symbol}: {e}{Style.RESET_ALL}")

    # Fallback: Attempt to normalize Decimal representation or return original string
    try:
        # Convert to Decimal first for consistent handling
        dec_amount = Decimal(str(amount))
        # normalize() removes trailing zeros, to_eng_string() avoids scientific notation
        return dec_amount.normalize().to_eng_string()
    except (InvalidOperation, TypeError, ValueError):
        logger.warning(f"Could not format amount '{amount}' using Decimal fallback, returning as string.")
        return str(amount) # Absolute fallback


# --- Termux SMS Alert Function ---
def send_sms_alert(message: str) -> bool:
    """Sends an SMS alert using the Termux:API command-line tool.

    Args:
        message: The text message content to send. Limits message length internally.

    Returns:
        True if the command executed successfully (return code 0), False otherwise.
    """
    if not CONFIG.enable_sms_alerts:
        logger.debug("SMS alerts disabled via config.")
        return False
    if not CONFIG.sms_recipient_number:
        logger.warning("SMS alerts enabled, but SMS_RECIPIENT_NUMBER is not set in config.")
        return False

    # Limit message length to avoid issues (SMS standard is 160 chars, but be conservative)
    max_sms_length = 150
    truncated_message = message[:max_sms_length] + "..." if len(message) > max_sms_length else message

    try:
        # Direct passing is usually fine for simple messages, quoting is complex.
        command: list[str] = ['termux-sms-send', '-n', CONFIG.sms_recipient_number, truncated_message]
        log_preview = message[:70].replace('\n', ' ') + ('...' if len(message) > 70 else '')
        logger.info(f"{Fore.MAGENTA}Attempting SMS to {CONFIG.sms_recipient_number} (Timeout: {CONFIG.sms_timeout_seconds}s): \"{log_preview}\"{Style.RESET_ALL}")

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
        # No SMS here as config loading happens first. Log is critical.
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
        send_sms_alert("[Pyrmethus] Initialized & authenticated successfully.")
        return exchange

    except ccxt.AuthenticationError as e:
        logger.critical(f"Authentication failed: {e}. Check API keys, IP whitelist, and permissions on Bybit.")
        send_sms_alert(f"[Pyrmethus] CRITICAL: Authentication FAILED: {e}. Bot stopped.")
    except ccxt.NetworkError as e:
        logger.critical(f"Network error during initialization: {e}. Check internet connection and Bybit status.")
        send_sms_alert(f"[Pyrmethus] CRITICAL: Network Error on Init: {e}. Bot stopped.")
    except ccxt.ExchangeError as e:
        logger.critical(f"Exchange error during initialization: {e}. Check Bybit status or API documentation.")
        send_sms_alert(f"[Pyrmethus] CRITICAL: Exchange Error on Init: {e}. Bot stopped.")
    except Exception as e:
        logger.critical(f"Unexpected error during exchange initialization: {e}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[Pyrmethus] CRITICAL: Unexpected Init Error: {type(e).__name__}. Bot stopped.")

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
        Columns are populated with pd.NA on error or insufficient data.
    """
    col_prefix = f"{prefix}" if prefix else ""
    st_val_col = f"{col_prefix}supertrend"
    trend_col = f"{col_prefix}trend"
    long_col = f"{col_prefix}st_long"
    short_col = f"{col_prefix}st_short"
    target_cols = [st_val_col, trend_col, long_col, short_col]

    # pandas_ta uses float in the generated column name string representation
    raw_st_col = f"SUPERT_{length}_{float(multiplier)}"
    raw_st_trend_col = f"SUPERTd_{length}_{float(multiplier)}"
    raw_st_long_col = f"SUPERTl_{length}_{float(multiplier)}"
    raw_st_short_col = f"SUPERTs_{length}_{float(multiplier)}"
    required_input_cols = ["high", "low", "close"]
    min_required_len = length + 1  # Need at least 'length' periods for ATR + 1 for comparison

    # Initialize target columns to NA first
    for col in target_cols: df[col] = pd.NA

    if df is None or df.empty or not all(c in df.columns for c in required_input_cols) or len(df) < min_required_len:
        logger.warning(f"{Fore.YELLOW}Indicator Calc ({col_prefix}ST): Input invalid or too short (Len: {len(df) if df is not None else 0}, Need: {min_required_len}).{Style.RESET_ALL}")
        return df

    try:
        # pandas_ta expects float multiplier, calculate in place
        df.ta.supertrend(length=length, multiplier=float(multiplier), append=True)

        # Check if pandas_ta created the expected raw columns
        if raw_st_col not in df.columns or raw_st_trend_col not in df.columns:
            logger.error(f"{Fore.RED}Indicator Calc ({col_prefix}ST): pandas_ta failed to create raw columns: {raw_st_col}, {raw_st_trend_col}{Style.RESET_ALL}")
            # Columns are already NA, so just return
            return df

        # Convert Supertrend value to Decimal, handle potential NaN from ta
        df[st_val_col] = df[raw_st_col].apply(lambda x: safe_decimal_conversion(x, default=Decimal('NaN')))
        # Convert trend direction (-1/1) to boolean (True for uptrend 1)
        df[trend_col] = df[raw_st_trend_col].apply(lambda x: True if pd.notna(x) and x == 1 else (False if pd.notna(x) else pd.NA))

        # Calculate flip signals (requires previous trend value, handle NA shifts)
        prev_trend_raw = df[raw_st_trend_col].shift(1)
        current_trend_raw = df[raw_st_trend_col]

        # Flip from down (-1) to up (1)
        df[long_col] = (prev_trend_raw == -1) & (current_trend_raw == 1)
        # Flip from up (1) to down (-1)
        df[short_col] = (prev_trend_raw == 1) & (current_trend_raw == -1)

        # Drop the raw columns generated by pandas_ta
        raw_st_cols_to_drop = [raw_st_col, raw_st_trend_col, raw_st_long_col, raw_st_short_col]
        df.drop(columns=[col for col in raw_st_cols_to_drop if col in df.columns], inplace=True)

        # Log last calculated values for debugging
        last_st_val = df[st_val_col].iloc[-1]
        if isinstance(last_st_val, Decimal) and not last_st_val.is_nan():
            last_trend_bool = df[trend_col].iloc[-1]
            last_trend_str = 'Up' if last_trend_bool is True else ('Down' if last_trend_bool is False else 'N/A')
            signal_long = df[long_col].iloc[-1]
            signal_short = df[short_col].iloc[-1]
            signal_str = 'LONG' if signal_long else ('SHORT' if signal_short else 'None')
            logger.debug(f"Indicator Calc ({col_prefix}ST({length},{multiplier})): Trend={last_trend_str}, Val={last_st_val:.4f}, Signal={signal_str}")
        else:
            logger.debug(f"Indicator Calc ({col_prefix}ST({length},{multiplier})): Resulted in NA/Invalid for last candle.")

    except KeyError as e:
        logger.error(f"{Fore.RED}Indicator Calc ({col_prefix}ST): Missing column during calculation: {e}{Style.RESET_ALL}")
        # Ensure columns are NA on error (already initialized)
    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc ({col_prefix}ST): Unexpected error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        # Ensure columns are NA on error (already initialized)
    return df


def analyze_volume_atr(df: pd.DataFrame, atr_len: int, vol_ma_len: int) -> dict[str, Decimal | None]:
    """Calculates ATR, Volume Moving Average, and checks for volume spikes.

    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume' columns.
        atr_len: The lookback period for ATR calculation.
        vol_ma_len: The lookback period for the Volume Moving Average.

    Returns:
        A dictionary containing:
        - 'atr': Calculated ATR value (Decimal), or None if calculation failed/invalid.
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
            last_atr_raw = df[atr_col].iloc[-1]
            if pd.notna(last_atr_raw):
                # Convert to Decimal, handle potential NaN conversion
                atr_decimal = safe_decimal_conversion(last_atr_raw, default=Decimal('NaN'))
                if not atr_decimal.is_nan():
                    results["atr"] = atr_decimal
            df.drop(columns=[atr_col], errors='ignore', inplace=True)  # Clean up raw column
        else:
             logger.warning(f"{Fore.YELLOW}Indicator Calc (ATR): Column '{atr_col}' not found after calculation.{Style.RESET_ALL}")

        # Calculate Volume MA using pandas rolling mean
        volume_ma_col = 'volume_ma'
        # Use min_periods to get a value even if window isn't full initially
        df[volume_ma_col] = df['volume'].rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2)).mean()
        last_vol_ma_raw = df[volume_ma_col].iloc[-1]
        last_vol_raw = df['volume'].iloc[-1]

        # Convert volume MA and last volume to Decimal, handling NaNs
        if pd.notna(last_vol_ma_raw):
            vol_ma_decimal = safe_decimal_conversion(last_vol_ma_raw, default=Decimal('NaN'))
            if not vol_ma_decimal.is_nan(): results["volume_ma"] = vol_ma_decimal
        if pd.notna(last_vol_raw):
             last_vol_decimal = safe_decimal_conversion(last_vol_raw, default=Decimal('NaN'))
             if not last_vol_decimal.is_nan(): results["last_volume"] = last_vol_decimal

        # Calculate Volume Ratio safely
        if results["volume_ma"] is not None and results["volume_ma"] > CONFIG.POSITION_QTY_EPSILON and results["last_volume"] is not None:
            try:
                results["volume_ratio"] = results["last_volume"] / results["volume_ma"]
            except (DivisionByZero, InvalidOperation):
                 logger.warning(f"{Fore.YELLOW}Indicator Calc (Vol/ATR): Division by zero or invalid operation calculating volume ratio.{Style.RESET_ALL}")
                 results["volume_ratio"] = None
        else:
             results["volume_ratio"] = None  # Set to None if MA is zero/negligible or volume is missing

        if volume_ma_col in df.columns: df.drop(columns=[volume_ma_col], errors='ignore', inplace=True)  # Clean up temp column

        # Log results
        atr_str = f"{results['atr']:.5f}" if results['atr'] is not None else 'N/A'
        vol_ma_str = f"{results['volume_ma']:.2f}" if results['volume_ma'] is not None else 'N/A'
        last_vol_str = f"{results['last_volume']:.2f}" if results['last_volume'] is not None else 'N/A'
        vol_ratio_str = f"{results['volume_ratio']:.2f}" if results['volume_ratio'] is not None else 'N/A'
        logger.debug(f"Indicator Calc: ATR({atr_len})={atr_str}, Vol={last_vol_str}, VolMA({vol_ma_len})={vol_ma_str}, VolRatio={vol_ratio_str}")

    except KeyError as e:
        logger.error(f"{Fore.RED}Indicator Calc (Vol/ATR): Missing column: {e}{Style.RESET_ALL}")
        results = {key: None for key in results} # Reset all to None on error
    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc (Vol/ATR): Unexpected error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        results = {key: None for key in results} # Reset all to None on error
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
        The input DataFrame with added columns (populated with pd.NA on error/insufficient data):
        - 'stochrsi_k': StochRSI %K line value (Decimal or pd.NA).
        - 'stochrsi_d': StochRSI %D line value (Decimal or pd.NA).
        - 'momentum': Momentum value (Decimal or pd.NA).
    """
    k_col_name = 'stochrsi_k'
    d_col_name = 'stochrsi_d'
    mom_col_name = 'momentum'
    target_cols = [k_col_name, d_col_name, mom_col_name]
    # Estimate minimum length needed - StochRSI needs RSI + Stoch periods + smoothing
    min_len = max(rsi_len + stoch_len + max(k, d), mom_len) + 5  # Add buffer
    for col in target_cols: df[col] = pd.NA  # Initialize columns

    if df is None or df.empty or "close" not in df.columns or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Indicator Calc (StochRSI/Mom): Input invalid or too short (Len: {len(df) if df is not None else 0}, Need ~{min_len}).{Style.RESET_ALL}")
        return df
    try:
        # Calculate StochRSI - use append=False to get predictable column names
        stochrsi_df = df.ta.stochrsi(length=stoch_len, rsi_length=rsi_len, k=k, d=d, append=False)
        raw_k_col = f"STOCHRSIk_{stoch_len}_{rsi_len}_{k}_{d}"
        raw_d_col = f"STOCHRSId_{stoch_len}_{rsi_len}_{k}_{d}"

        if raw_k_col in stochrsi_df.columns:
            df[k_col_name] = stochrsi_df[raw_k_col].apply(lambda x: safe_decimal_conversion(x, default=Decimal('NaN')))
            # Replace NaN Decimals with pd.NA for consistency in pandas checks
            df[k_col_name] = df[k_col_name].apply(lambda x: pd.NA if isinstance(x, Decimal) and x.is_nan() else x)
        else:
            logger.warning(f"{Fore.YELLOW}StochRSI K column '{raw_k_col}' not found after calculation.{Style.RESET_ALL}")

        if raw_d_col in stochrsi_df.columns:
            df[d_col_name] = stochrsi_df[raw_d_col].apply(lambda x: safe_decimal_conversion(x, default=Decimal('NaN')))
            df[d_col_name] = df[d_col_name].apply(lambda x: pd.NA if isinstance(x, Decimal) and x.is_nan() else x)
        else:
             logger.warning(f"{Fore.YELLOW}StochRSI D column '{raw_d_col}' not found after calculation.{Style.RESET_ALL}")

        # Calculate Momentum
        raw_mom_col = f"MOM_{mom_len}"
        df.ta.mom(length=mom_len, append=True)  # Append momentum directly
        if raw_mom_col in df.columns:
            df[mom_col_name] = df[raw_mom_col].apply(lambda x: safe_decimal_conversion(x, default=Decimal('NaN')))
            df[mom_col_name] = df[mom_col_name].apply(lambda x: pd.NA if isinstance(x, Decimal) and x.is_nan() else x)
            df.drop(columns=[raw_mom_col], errors='ignore', inplace=True)  # Clean up raw column
        else:
            logger.warning(f"{Fore.YELLOW}Momentum column '{raw_mom_col}' not found after calculation.{Style.RESET_ALL}")

        # Log last values (handle potential pd.NA)
        k_val, d_val, mom_val = df[k_col_name].iloc[-1], df[d_col_name].iloc[-1], df[mom_col_name].iloc[-1]
        k_str = f"{k_val:.2f}" if pd.notna(k_val) else "N/A"
        d_str = f"{d_val:.2f}" if pd.notna(d_val) else "N/A"
        mom_str = f"{mom_val:.4f}" if pd.notna(mom_val) else "N/A"
        logger.debug(f"Indicator Calc (StochRSI/Mom): K={k_str}, D={d_str}, Mom={mom_str}")

    except KeyError as e:
        logger.error(f"{Fore.RED}Indicator Calc (StochRSI/Mom): Missing column: {e}{Style.RESET_ALL}")
        # Columns already initialized to NA
    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc (StochRSI/Mom): Unexpected error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        # Columns already initialized to NA
    return df


def calculate_ehlers_fisher(df: pd.DataFrame, length: int, signal: int) -> pd.DataFrame:
    """Calculates the Ehlers Fisher Transform indicator using pandas_ta.

    Args:
        df: DataFrame with 'high', 'low' columns.
        length: The lookback period for the Fisher Transform.
        signal: The smoothing period for the signal line (usually 1).

    Returns:
        The input DataFrame with added columns (populated with pd.NA on error/insufficient data):
        - 'ehlers_fisher': Fisher Transform value (Decimal or pd.NA).
        - 'ehlers_signal': Fisher Transform signal line value (Decimal or pd.NA).
    """
    fish_col_name = 'ehlers_fisher'
    sig_col_name = 'ehlers_signal'
    target_cols = [fish_col_name, sig_col_name]
    min_len = length + signal  # Approximate minimum length
    for col in target_cols: df[col] = pd.NA  # Initialize columns

    if df is None or df.empty or not all(c in df.columns for c in ["high", "low"]) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Indicator Calc (EhlersFisher): Input invalid or too short (Len: {len(df) if df is not None else 0}, Need ~{min_len}).{Style.RESET_ALL}")
        return df
    try:
        # Calculate Fisher Transform - use append=False
        fisher_df = df.ta.fisher(length=length, signal=signal, append=False)
        raw_fish_col = f"FISHERT_{length}_{signal}"
        raw_signal_col = f"FISHERTs_{length}_{signal}"

        if raw_fish_col in fisher_df.columns:
            df[fish_col_name] = fisher_df[raw_fish_col].apply(lambda x: safe_decimal_conversion(x, default=Decimal('NaN')))
            df[fish_col_name] = df[fish_col_name].apply(lambda x: pd.NA if isinstance(x, Decimal) and x.is_nan() else x)
        else:
            logger.warning(f"{Fore.YELLOW}Ehlers Fisher column '{raw_fish_col}' not found after calculation.{Style.RESET_ALL}")

        if raw_signal_col in fisher_df.columns:
            df[sig_col_name] = fisher_df[raw_signal_col].apply(lambda x: safe_decimal_conversion(x, default=Decimal('NaN')))
            df[sig_col_name] = df[sig_col_name].apply(lambda x: pd.NA if isinstance(x, Decimal) and x.is_nan() else x)
        else:
            logger.warning(f"{Fore.YELLOW}Ehlers Signal column '{raw_signal_col}' not found after calculation.{Style.RESET_ALL}")

        # Log last values (handle potential pd.NA)
        fish_val, sig_val = df[fish_col_name].iloc[-1], df[sig_col_name].iloc[-1]
        fish_str = f"{fish_val:.4f}" if pd.notna(fish_val) else "N/A"
        sig_str = f"{sig_val:.4f}" if pd.notna(sig_val) else "N/A"
        logger.debug(f"Indicator Calc (EhlersFisher({length},{signal})): Fisher={fish_str}, Signal={sig_str}")

    except KeyError as e:
        logger.error(f"{Fore.RED}Indicator Calc (EhlersFisher): Missing column: {e}{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc (EhlersFisher): Unexpected error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
    return df


def calculate_ehlers_ma(df: pd.DataFrame, fast_len: int, slow_len: int) -> pd.DataFrame:
    """Calculates Ehlers-style Moving Averages (placeholder using EMA).

    Args:
        df: DataFrame with 'close' column.
        fast_len: Lookback period for the fast moving average.
        slow_len: Lookback period for the slow moving average.

    Returns:
        The input DataFrame with added columns (populated with pd.NA on error/insufficient data):
        - 'fast_ema': Fast EMA value (Decimal or pd.NA).
        - 'slow_ema': Slow EMA value (Decimal or pd.NA).
    """
    fast_col_name = 'fast_ema'
    slow_col_name = 'slow_ema'
    target_cols = [fast_col_name, slow_col_name]
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

        fast_ema_raw = df.ta.ema(length=fast_len)
        slow_ema_raw = df.ta.ema(length=slow_len)

        df[fast_col_name] = fast_ema_raw.apply(lambda x: safe_decimal_conversion(x, default=Decimal('NaN')))
        df[fast_col_name] = df[fast_col_name].apply(lambda x: pd.NA if isinstance(x, Decimal) and x.is_nan() else x)

        df[slow_col_name] = slow_ema_raw.apply(lambda x: safe_decimal_conversion(x, default=Decimal('NaN')))
        df[slow_col_name] = df[slow_col_name].apply(lambda x: pd.NA if isinstance(x, Decimal) and x.is_nan() else x)

        # Log last values (handle potential pd.NA)
        fast_val, slow_val = df[fast_col_name].iloc[-1], df[slow_col_name].iloc[-1]
        fast_str = f"{fast_val:.4f}" if pd.notna(fast_val) else "N/A"
        slow_str = f"{slow_val:.4f}" if pd.notna(slow_val) else "N/A"
        logger.debug(f"Indicator Calc (EhlersMA/EMA({fast_len},{slow_len})): Fast={fast_str}, Slow={slow_str}")

    except KeyError as e:
        logger.error(f"{Fore.RED}Indicator Calc (EhlersMA/EMA): Missing column: {e}{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc (EhlersMA/EMA): Unexpected error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
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
        - 'bid_ask_ratio': Ratio of cumulative bid volume to ask volume within depth (Decimal), or None.
        - 'spread': Difference between best ask and best bid (Decimal), or None.
        - 'best_bid': Best bid price (Decimal), or None.
        - 'best_ask': Best ask price (Decimal), or None.
        Returns None for values if fetch/parse fails or data is insufficient.
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
            return results # Cannot calculate ratio or spread

        # Extract best bid/ask using safe conversion, handle potential NaN
        best_bid_raw = bids[0][0] if len(bids[0]) > 0 else None
        best_ask_raw = asks[0][0] if len(asks[0]) > 0 else None
        best_bid = safe_decimal_conversion(best_bid_raw, default=Decimal('NaN'))
        best_ask = safe_decimal_conversion(best_ask_raw, default=Decimal('NaN'))

        results["best_bid"] = None if best_bid.is_nan() else best_bid
        results["best_ask"] = None if best_ask.is_nan() else best_ask

        # Calculate spread only if both bid and ask are valid
        if results["best_bid"] is not None and results["best_ask"] is not None:
            if results["best_ask"] > results["best_bid"]:  # Sanity check
                results["spread"] = results["best_ask"] - results["best_bid"]
                logger.debug(f"OB: Bid={results['best_bid']:.4f}, Ask={results['best_ask']:.4f}, Spread={results['spread']:.4f}")
            else:
                logger.warning(f"{Fore.YELLOW}Order Book: Best bid ({results['best_bid']}) >= best ask ({results['best_ask']}). Spread calculation invalid.{Style.RESET_ALL}")
                results["spread"] = None # Invalid spread
        else:
            logger.debug(f"OB: Bid={results['best_bid'] or 'N/A'}, Ask={results['best_ask'] or 'N/A'} (Spread N/A)")

        # Sum volumes within the specified depth using Decimal, default to 0 for safety
        bid_vol = sum(safe_decimal_conversion(bid[1], default=Decimal('0')) for bid in bids[:depth] if len(bid) > 1)
        ask_vol = sum(safe_decimal_conversion(ask[1], default=Decimal('0')) for ask in asks[:depth] if len(ask) > 1)
        logger.debug(f"OB (Depth {depth}): BidVol={bid_vol:.4f}, AskVol={ask_vol:.4f}")

        # Calculate bid/ask ratio safely
        if ask_vol > CONFIG.POSITION_QTY_EPSILON:
            try:
                results["bid_ask_ratio"] = (bid_vol / ask_vol).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP) # Quantize ratio
                logger.debug(f"OB Ratio: {results['bid_ask_ratio']:.3f}")
            except (DivisionByZero, InvalidOperation):
                logger.warning(f"{Fore.YELLOW}Order Book: Error calculating OB ratio (division by zero or invalid operation).{Style.RESET_ALL}")
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

    # Ensure None is returned for keys if any error occurred and values are None
    # (This is already handled by the logic above, but double-check)
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
        or None if fetching or processing fails. Handles NaNs robustly.
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
            # Coerce errors during timestamp conversion, though unlikely with standard API format
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors='coerce')
            if df["timestamp"].isnull().any():
                logger.error(f"{Fore.RED}Data Fetch: Found invalid timestamps after conversion. Cannot proceed.{Style.RESET_ALL}")
                return None
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
            nan_cols = nan_counts[nan_counts > 0]
            logger.warning(f"{Fore.YELLOW}Data Fetch: OHLCV contains NaNs after conversion:\n{nan_cols}\nAttempting forward fill...{Style.RESET_ALL}")
            df.ffill(inplace=True)  # Forward fill first (common for missing data)
            if df.isnull().values.any():  # Check again, backfill if needed (less common)
                nan_counts_after_ffill = df.isnull().sum()
                nan_cols_after_ffill = nan_counts_after_ffill[nan_counts_after_ffill > 0]
                logger.warning(f"{Fore.YELLOW}NaNs remain after ffill:\n{nan_cols_after_ffill}\nAttempting backward fill...{Style.RESET_ALL}")
                df.bfill(inplace=True)
                if df.isnull().values.any():
                    nan_counts_final = df.isnull().sum()
                    nan_cols_final = nan_counts_final[nan_counts_final > 0]
                    logger.error(f"{Fore.RED}Data Fetch: NaNs persist even after ffill and bfill:\n{nan_cols_final}\nCannot proceed with this data.{Style.RESET_ALL}")
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

    Assumes One-Way Mode on Bybit. Uses Decimal for quantity and price.

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

    if not market:  # Should not happen if above try succeeded, but defensive check
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
            category = 'linear'  # Default assumption for USDT pairs

        params = {'category': category, 'symbol': market_id} # Filter by symbol directly in params for Bybit V5
        logger.debug(f"Position Check: Fetching positions with params: {params}")

        # Fetch positions, potentially filtered by the exchange based on params
        fetched_positions = exchange.fetch_positions(symbols=[symbol], params=params) # Still provide symbols= for CCXT layer

        # Filter for the active position in One-Way mode (positionIdx == 0)
        active_pos_data = None
        for pos in fetched_positions:
            pos_info = pos.get('info', {})
            # Match market ID returned by the API, which should match the request
            pos_market_id = pos_info.get('symbol')
            # Bybit V5 One-Way mode uses positionIdx 0. Hedge mode uses 1 for Buy, 2 for Sell.
            position_idx = safe_decimal_conversion(pos_info.get('positionIdx', '-1')).to_integral_value() # Use -1 default to indicate if not found
            pos_side_v5 = pos_info.get('side', 'None')  # 'Buy' for long, 'Sell' for short, 'None' if flat
            size_str = pos_info.get('size')

            # Match market ID, check for One-Way mode (idx 0), and ensure side is not 'None' (means position exists)
            if pos_market_id == market_id and position_idx == 0 and pos_side_v5 != 'None':
                size = safe_decimal_conversion(size_str, default=Decimal('0'))
                # Check if size is significant (greater than epsilon)
                if abs(size) > CONFIG.POSITION_QTY_EPSILON:
                    active_pos_data = pos  # Found the active position for this symbol in One-Way mode
                    break  # Assume only one such position exists per symbol in One-Way mode

        if active_pos_data:
            try:
                info = active_pos_data.get('info', {})
                size = safe_decimal_conversion(info.get('size'), default=Decimal('NaN'))
                # Use 'avgPrice' from info for V5 entry price
                entry_price = safe_decimal_conversion(info.get('avgPrice'), default=Decimal('NaN'))
                # Determine side based on V5 'side' field ('Buy' -> Long, 'Sell' -> Short)
                v5_side = info.get('side')
                side = CONFIG.POS_LONG if v5_side == 'Buy' else (CONFIG.POS_SHORT if v5_side == 'Sell' else CONFIG.POS_NONE)

                if side == CONFIG.POS_NONE or size.is_nan() or entry_price.is_nan():
                     logger.warning(f"{Fore.YELLOW}Position Check: Parsed position data invalid (Side:{side}, Size:{size}, Entry:{entry_price}). Treating as flat.{Style.RESET_ALL}")
                     return default_pos

                position_qty = abs(size)
                if position_qty <= CONFIG.POSITION_QTY_EPSILON:
                     logger.info(f"Position Check: Found position for {market_id}, but size ({size}) is negligible. Treating as flat.")
                     return default_pos

                logger.info(f"{Fore.YELLOW}Position Check: Found ACTIVE {side} position: Qty={position_qty:.8f} @ Entry={entry_price:.4f}{Style.RESET_ALL}")
                return {'side': side, 'qty': position_qty, 'entry_price': entry_price}
            except (KeyError, TypeError, InvalidOperation, Exception) as parse_err:
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
    """Sets leverage for a futures symbol, handling Bybit V5 API specifics and retries.

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

    # Validate leverage input
    if leverage <= 0:
        logger.error(f"{Fore.RED}Leverage Setting: Invalid leverage value requested: {leverage}. Must be positive.{Style.RESET_ALL}")
        return False

    for attempt in range(CONFIG.RETRY_COUNT):
        try:
            # Bybit V5 requires setting buyLeverage and sellLeverage separately via params
            # CCXT typically abstracts this, but providing params explicitly ensures compatibility.
            params = {'buyLeverage': str(leverage), 'sellLeverage': str(leverage)}
            logger.debug(f"Leverage Setting: Calling set_leverage with leverage={leverage}, symbol={symbol}, params={params}")
            response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
            logger.success(f"{Fore.GREEN}Leverage Setting: Successfully set to {leverage}x for {symbol}. Response: {response}{Style.RESET_ALL}")
            return True
        except ccxt.ExchangeError as e:
            # Check for common Bybit messages indicating leverage is already set or not modified
            err_str = str(e).lower()
            # Example error codes/messages from Bybit V5 (these might change):
            # 110044: "Set leverage not modified" / "Leverage not modified"
            # Specific string checks:
            if "leverage not modified" in err_str or "leverage is same as requested" in err_str:
                logger.info(f"{Fore.CYAN}Leverage Setting: Leverage already set to {leverage}x for {symbol}.{Style.RESET_ALL}")
                return True
            logger.warning(f"{Fore.YELLOW}Leverage Setting: Exchange error (Attempt {attempt + 1}/{CONFIG.RETRY_COUNT}): {e}{Style.RESET_ALL}")
            if attempt < CONFIG.RETRY_COUNT - 1:
                time.sleep(CONFIG.RETRY_DELAY_SECONDS)
            else:
                logger.error(f"{Fore.RED}Leverage Setting: Failed after {CONFIG.RETRY_COUNT} attempts due to ExchangeError.{Style.RESET_ALL}")
                send_sms_alert(f"[{symbol.split('/')[0]}] ERROR: Failed to set leverage for {symbol} after retries: {e}")
        except (ccxt.NetworkError, Exception) as e:
            logger.warning(f"{Fore.YELLOW}Leverage Setting: Network/Other error (Attempt {attempt + 1}/{CONFIG.RETRY_COUNT}): {e}{Style.RESET_ALL}")
            if attempt < CONFIG.RETRY_COUNT - 1:
                time.sleep(CONFIG.RETRY_DELAY_SECONDS)
            else:
                logger.error(f"{Fore.RED}Leverage Setting: Failed after {CONFIG.RETRY_COUNT} attempts due to {type(e).__name__}.{Style.RESET_ALL}")
                send_sms_alert(f"[{symbol.split('/')[0]}] ERROR: Failed to set leverage for {symbol} after retries: {type(e).__name__}")
    return False


def close_position(exchange: ccxt.Exchange, symbol: str, position_to_close: dict[str, Any], reason: str = "Signal") -> dict[str, Any] | None:
    """Closes the specified active position by placing a market order with reduceOnly=True.
    Re-validates the position just before closing. Uses Decimal internally.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol.
        position_to_close: A dictionary representing the position state *before* re-validation.
        reason: A string indicating the reason for closing (for logging/alerts).

    Returns:
        The CCXT order dictionary if the close order was successfully placed, None otherwise.
    """
    initial_side = position_to_close.get('side', CONFIG.POS_NONE)
    initial_qty = position_to_close.get('qty', Decimal("0.0"))
    market_base = symbol.split('/')[0] if '/' in symbol else symbol
    logger.info(f"{Fore.YELLOW}Close Position: Initiated for {symbol}. Reason: {reason}. State before re-val: {initial_side} Qty={initial_qty:.8f}{Style.RESET_ALL}")

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
         logger.warning(f"{Fore.YELLOW}Close Position: Discrepancy detected! Initial side was {initial_side}, live side is {live_position_side}. Closing live position based on re-validation.{Style.RESET_ALL}")
         # Continue with closing the actual live position

    # Determine the side needed to close the position
    side_to_execute_close = CONFIG.SIDE_SELL if live_position_side == CONFIG.POS_LONG else CONFIG.SIDE_BUY

    try:
        # Format amount according to market precision
        amount_str = format_amount(exchange, symbol, live_amount_to_close)
        # Convert formatted string back to Decimal for check, then to float for CCXT
        amount_decimal = safe_decimal_conversion(amount_str, default=Decimal('NaN'))
        if amount_decimal.is_nan() or amount_decimal <= CONFIG.POSITION_QTY_EPSILON:
            logger.error(f"{Fore.RED}Close Position: Closing amount '{amount_str}' after precision formatting is negligible or invalid. Aborting.{Style.RESET_ALL}")
            return None
        amount_float = float(amount_decimal)

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
        # Use safe_decimal_conversion for robustness, providing NaN as default
        filled_qty = safe_decimal_conversion(order.get('filled'), default=Decimal('NaN'))
        avg_fill_price = safe_decimal_conversion(order.get('average'), default=Decimal('NaN'))
        cost = safe_decimal_conversion(order.get('cost'), default=Decimal('NaN'))

        # Format for logging, handle potential NaNs
        filled_qty_str = f"{filled_qty:.8f}" if not filled_qty.is_nan() else "N/A"
        avg_fill_price_str = f"{avg_fill_price:.4f}" if not avg_fill_price.is_nan() else "N/A"
        cost_str = f"{cost:.2f}" if not cost.is_nan() else "N/A"

        logger.success(f"{Fore.GREEN}{Style.BRIGHT}Close Position: Order ({reason}) submitted for {symbol}. "
                       f"ID:...{order_id_short}, Status: {status}, Filled: {filled_qty_str}/{amount_str}, AvgFill: {avg_fill_price_str}, Cost: {cost_str} USDT.{Style.RESET_ALL}")
        # Note: Market orders might fill immediately, but status might be 'open' initially.
        # We don't wait for fill confirmation here, assuming reduceOnly works reliably or subsequent checks handle it.

        send_sms_alert(f"[{market_base}] Closed {live_position_side} {amount_str} @ ~{avg_fill_price_str} ({reason}). ID:...{order_id_short}")
        return order  # Return the order details

    except ccxt.InsufficientFunds as e:
         logger.error(f"{Fore.RED}Close Position ({reason}): Insufficient funds for {symbol}: {e}{Style.RESET_ALL}")
         send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): Insufficient Funds. Check margin/position.")
    except ccxt.ExchangeError as e:
        # Check for specific Bybit errors indicating the position might already be closed or closing
        err_str = str(e).lower()
        # Example Bybit V5 error codes/messages (may change):
        # 110025: "Position size is zero" / "position idx not match position side" (when trying to close wrong side due to race condition)
        # 110053: "The order would not reduce the position size"
        # 30086: Order failed maybe because your position has been closed (or similar TP/SL related)
        if "position size is zero" in err_str or \
           "order would not reduce position size" in err_str or \
           "position has been closed" in err_str or \
           "position is already zero" in err_str:  # Add more known messages if needed
             logger.warning(f"{Fore.YELLOW}Close Position ({reason}): Exchange indicates position likely already closed/zero: {e}. Assuming closed.{Style.RESET_ALL}")
             # Don't send error SMS, treat as effectively closed.
             return None  # Treat as success (nothing to close) in this specific case
        else:
             logger.error(f"{Fore.RED}Close Position ({reason}): Exchange error for {symbol}: {e}{Style.RESET_ALL}")
             send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): Exchange Error: {type(e).__name__}. Check logs.")
    except (ccxt.NetworkError, ValueError, TypeError, InvalidOperation, Exception) as e:
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
    log_prefix = "Risk Calc"
    logger.debug(f"{log_prefix}: Equity={equity:.4f}, Risk%={risk_per_trade_pct:.4%}, Entry={entry_price:.4f}, SL={stop_loss_price:.4f}, Lev={leverage}x")

    # --- Input Validation ---
    if not isinstance(equity, Decimal) or equity <= 0:
        logger.error(f"{Fore.RED}{log_prefix} Error: Invalid equity: {equity} (Type: {type(equity).__name__}). Must be positive Decimal.{Style.RESET_ALL}")
        return None, None
    if not isinstance(risk_per_trade_pct, Decimal) or not (0 < risk_per_trade_pct < 1):
        logger.error(f"{Fore.RED}{log_prefix} Error: Invalid risk percentage: {risk_per_trade_pct} (Type: {type(risk_per_trade_pct).__name__}). Must be Decimal between 0 and 1.{Style.RESET_ALL}")
        return None, None
    if not isinstance(entry_price, Decimal) or entry_price <= 0:
        logger.error(f"{Fore.RED}{log_prefix} Error: Invalid entry price: {entry_price} (Type: {type(entry_price).__name__}). Must be positive Decimal.{Style.RESET_ALL}")
        return None, None
    if not isinstance(stop_loss_price, Decimal) or stop_loss_price <= 0:
        logger.error(f"{Fore.RED}{log_prefix} Error: Invalid SL price: {stop_loss_price} (Type: {type(stop_loss_price).__name__}). Must be positive Decimal.{Style.RESET_ALL}")
        return None, None
    if not isinstance(leverage, int) or leverage <= 0:
        logger.error(f"{Fore.RED}{log_prefix} Error: Invalid leverage: {leverage}. Must be positive integer.{Style.RESET_ALL}")
        return None, None

    price_diff = abs(entry_price - stop_loss_price)
    if price_diff < CONFIG.POSITION_QTY_EPSILON: # Use epsilon for comparison
        logger.error(f"{Fore.RED}{log_prefix} Error: Entry price ({entry_price}) and SL price ({stop_loss_price}) are too close (Diff: {price_diff:.8f}).{Style.RESET_ALL}")
        return None, None

    try:
        # --- Calculation ---
        risk_amount_usdt: Decimal = equity * risk_per_trade_pct
        # For linear contracts (like BTC/USDT:USDT), the value of 1 unit of base currency (BTC) is its price in quote currency (USDT).
        # The risk per unit of the base currency is the price difference between entry and stop-loss.
        # Quantity = (Total Risk Amount USDT) / (Risk Per Unit in USDT)
        quantity_raw: Decimal = risk_amount_usdt / price_diff

        # --- Apply Precision ---
        # Format the raw quantity according to market rules *then* convert back to Decimal for further use
        quantity_precise_str = format_amount(exchange, symbol, quantity_raw)
        quantity_precise = safe_decimal_conversion(quantity_precise_str, default=Decimal('NaN'))

        if quantity_precise.is_nan() or quantity_precise <= CONFIG.POSITION_QTY_EPSILON:
            logger.warning(f"{Fore.YELLOW}{log_prefix} Warning: Calculated quantity ({quantity_precise_str} -> {quantity_precise}) is negligible, zero, or invalid after formatting. "
                           f"RiskAmt={risk_amount_usdt:.4f}, PriceDiff={price_diff:.4f}. Cannot place order.{Style.RESET_ALL}")
            return None, None

        # --- Calculate Estimated Margin ---
        # Margin = Position Value / Leverage = (Quantity * Entry Price) / Leverage
        position_value_usdt = quantity_precise * entry_price
        required_margin = (position_value_usdt / Decimal(leverage)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP) # Quantize margin

        logger.debug(f"{log_prefix} Result: RawQty={quantity_raw:.8f} -> PreciseQty={quantity_precise}, EstValue={position_value_usdt:.4f}, EstMargin={required_margin:.4f}")
        return quantity_precise, required_margin

    except (DivisionByZero, InvalidOperation, OverflowError) as e:
        logger.error(f"{Fore.RED}{log_prefix} Error: Calculation error: {e}. Inputs: Equity={equity}, Risk%={risk_per_trade_pct}, Entry={entry_price}, SL={stop_loss_price}{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix} Error: Unexpected exception during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())

    return None, None


def wait_for_order_fill(exchange: ccxt.Exchange, order_id: str, symbol: str, timeout_seconds: int) ->
