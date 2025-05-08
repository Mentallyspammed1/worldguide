```python
#!/usr/bin/env python

# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v2.1.1 (Enhanced Clarity & Arcane Weaving)
# Conjures high-frequency trades on Bybit Futures with enhanced precision, adaptable strategies, and Termux integration.

"""High-Frequency Trading Bot (Scalping) for Bybit USDT Perpetual Futures
Version: 2.1.1 (Unified: Selectable Strategies + Precision + Native SL/TSL + Pyrmethus Enhancements)

Enhancements in v2.1.1:
- Improved type hinting and docstrings.
- More robust error handling in various functions (incl. order placement, position checks).
- Refined logging messages with consistent coloring and detail levels.
- Strengthened validation in Config and helper functions.
- Cleaner handling of Bybit V5 API specifics (position parsing, order params).
- More explicit checks for market type and exchange capabilities.
- Improved graceful shutdown sequence with position confirmation.
- Minor code style and structure improvements.

Features:
- Unified Framework: Select from multiple trading strategies via configuration:
    - "DUAL_SUPERTREND": Classic trend-following confirmation.
    - "STOCHRSI_MOMENTUM": Oscillator-based reversal/momentum detection.
    - "EHLERS_FISHER": Cycle analysis for potential turning points.
    - "EHLERS_MA_CROSS": Moving average crossover (using EMA as placeholder).
- Financial Precision: Leverages Python's `Decimal` type for critical calculations (pricing, sizing, risk).
- Native Risk Management: Places exchange-native Stop Loss (SL) and Trailing Stop Loss (TSL) orders immediately upon entry.
    - Initial SL based on ATR (Average True Range) for volatility adaptation.
    - TSL trails price movement to lock in potential profits.
- Configurable Confirmation Filters: Optional validation using:
    - Volume Spike Analysis: Confirm entry signals with unusual volume activity.
    - Order Book Pressure: Gauge short-term market sentiment via bid/ask depth imbalance.
- Sophisticated Position Sizing: Calculates trade size based on account equity, defined risk percentage, and stop-loss distance. Includes checks against available margin and maximum order caps.
- Termux Integration: Sends SMS alerts for critical events (startup, errors, fills, shutdown) via Termux:API on Android devices.
- Robust Operations: Features detailed logging with color support (Colorama), comprehensive error handling for API/network issues, and a graceful shutdown procedure attempting to close open positions/orders.
- Bybit V5 API Compatibility: Specifically targets Bybit's V5 API via the CCXT library, including stricter position detection logic.

Disclaimer & Warnings:
- **⚠️ EXTREME RISK ⚠️**: This software involves arcane financial energies and is intended for EDUCATIONAL PURPOSES ONLY. High-frequency trading, especially with leverage, carries a SIGNIFICANT RISK OF SUBSTANTIAL FINANCIAL LOSS. Use EXCLUSIVELY AT YOUR OWN ABSOLUTE RISK. Past performance is not indicative of future results.
- **EXCHANGE-NATIVE ORDER DEPENDENCE**: Relies heavily on Bybit's native SL/TSL order types. Functionality is subject to exchange performance, potential slippage during execution, API reliability, and specific order behavior rules defined by Bybit. Unexpected market conditions (e.g., extreme volatility, liquidity gaps) can affect order execution.
- **PARAMETER SENSITIVITY**: Trading parameters (strategy settings, risk levels, SL/TSL values) are highly sensitive and require extensive tuning, backtesting, and forward-testing (paper trading/testnet) before any consideration of live deployment. Default values are examples and likely suboptimal.
- **API RATE LIMITS**: Monitor API usage. Excessive requests can lead to temporary or permanent bans from the exchange. Respect the limits imposed by Bybit's spirits.
- **MARKET SLIPPAGE**: Market orders, used for entry and potentially for SL/TSL execution, are susceptible to slippage, especially during volatile periods. The executed price may differ from the expected price.
- **TEST THOROUGHLY**: **DO NOT DEPLOY WITH REAL FUNDS WITHOUT EXTENSIVE AND RIGOROUS TESTING ON BYBIT'S TESTNET ENVIRONMENT.** Ensure you fully understand the code and its potential risks.
- **TERMUX DEPENDENCY**: SMS alerts require the Termux app and Termux:API add-on to be installed and correctly configured on an Android device.
- **API & LIBRARY UPDATES**: This code targets the Bybit V5 API via CCXT as of its writing. Future changes to the Bybit API or CCXT library may require code modifications to maintain functionality.
"""

# Standard Library Imports - The Foundational Runes
import logging
import os
import subprocess
import sys
import time
import traceback
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, getcontext
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party Libraries - Summoned Essences
try:
    import ccxt
    import pandas as pd
    import pandas_ta as ta  # type: ignore[import] # pandas_ta might not have complete type stubs
    from colorama import Back, Fore, Style
    from colorama import init as colorama_init
    from dotenv import load_dotenv
except ImportError as e:
    # Use raw ANSI codes as Colorama might be the missing package
    missing_pkg = getattr(e, 'name', 'dependency')
    print(f"\033[91m\033[1mCRITICAL ERROR:\033[0m Missing required Python package: '{missing_pkg}'.")
    print("Please install the necessary libraries, e.g., using:")
    print(f"  pip install -r requirements.txt  (if provided)")
    print(f"  pip install ccxt pandas pandas_ta python-dotenv colorama")
    sys.exit(1) # Exit with a non-zero code to indicate failure

# --- Initializations - Preparing the Ritual Chamber ---
colorama_init(autoreset=True)  # Activate Colorama's magic for colored terminal output
load_dotenv()                  # Load secrets from the hidden .env scroll into environment variables
getcontext().prec = 18         # Set Decimal precision for financial exactitude (adjust if needed)


# --- Logger Setup - The Oracle's Voice ---
# Determine logging level from environment variable or default to INFO
LOGGING_LEVEL_STR: str = os.getenv("LOGGING_LEVEL", "INFO").upper()
LOGGING_LEVEL: int = getattr(logging, LOGGING_LEVEL_STR, logging.INFO)

logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)] # Output logs to the console (e.g., Termux)
)
logger: logging.Logger = logging.getLogger(__name__) # Get the root logger

# Custom SUCCESS level and Neon Color Formatting for the Oracle
SUCCESS_LEVEL: int = 25 # Assign a numerical level between INFO (20) and WARNING (30)
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Logs a message with the custom SUCCESS level and mystical flair."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        # pylint: disable=protected-access
        self._log(SUCCESS_LEVEL, message, args, **kwargs)

# Bind the custom log method to the Logger class
logging.Logger.success = log_success # type: ignore[attr-defined]

# Apply vibrant colors if outputting to a TTY (like Termux or most terminals)
if sys.stdout.isatty():
    logging.addLevelName(logging.DEBUG, f"{Fore.CYAN}{Style.DIM}{logging.getLevelName(logging.DEBUG)}{Style.RESET_ALL}")
    logging.addLevelName(logging.INFO, f"{Fore.BLUE}{logging.getLevelName(logging.INFO)}{Style.RESET_ALL}")
    logging.addLevelName(SUCCESS_LEVEL, f"{Fore.MAGENTA}{Style.BRIGHT}{logging.getLevelName(SUCCESS_LEVEL)}{Style.RESET_ALL}")
    logging.addLevelName(logging.WARNING, f"{Fore.YELLOW}{Style.BRIGHT}{logging.getLevelName(logging.WARNING)}{Style.RESET_ALL}")
    logging.addLevelName(logging.ERROR, f"{Fore.RED}{Style.BRIGHT}{logging.getLevelName(logging.ERROR)}{Style.RESET_ALL}")
    logging.addLevelName(logging.CRITICAL, f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}{logging.getLevelName(logging.CRITICAL)}{Style.RESET_ALL}")

logger.info(f"Oracle activated. Logging level set to: {logging.getLevelName(logger.level)}")


# --- Configuration Class - Defining the Spell's Parameters ---
class Config:
    """Loads, validates, and stores configuration parameters from environment variables."""

    def __init__(self) -> None:
        """Initializes and validates the configuration."""
        logger.info(f"{Fore.MAGENTA}--- Summoning Configuration Runes ---{Style.RESET_ALL}")

        # --- API Credentials - Keys to the Exchange Vault ---
        self.api_key: Optional[str] = self._get_env("BYBIT_API_KEY", required=True, sensitive=True, color=Fore.RED)
        self.api_secret: Optional[str] = self._get_env("BYBIT_API_SECRET", required=True, sensitive=True, color=Fore.RED)

        # --- Trading Parameters - Core Incantation Variables ---
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT", color=Fore.YELLOW)
        self.interval: str = self._get_env("INTERVAL", "1m", color=Fore.YELLOW)  # Timeframe focus
        self.leverage: int = self._get_env("LEVERAGE", 10, cast_type=int, min_val=1, max_val=100, color=Fore.YELLOW) # Power multiplier
        self.sleep_seconds: int = self._get_env("SLEEP_SECONDS", 10, cast_type=int, min_val=1, color=Fore.YELLOW) # Pause between observations

        # --- Strategy Selection - Choosing the Path of Magic ---
        self.strategy_name: str = self._get_env("STRATEGY_NAME", "DUAL_SUPERTREND", color=Fore.CYAN).upper()
        self.valid_strategies: List[str] = ["DUAL_SUPERTREND", "STOCHRSI_MOMENTUM", "EHLERS_FISHER", "EHLERS_MA_CROSS"]
        if self.strategy_name not in self.valid_strategies:
            err_msg = f"Invalid STRATEGY_NAME '{self.strategy_name}'. Valid paths: {self.valid_strategies}"
            logger.critical(f"{Back.RED}{Fore.WHITE}{err_msg}{Style.RESET_ALL}")
            raise ValueError(err_msg)
        logger.info(f"{Fore.CYAN}Chosen Strategy Path: {self.strategy_name}{Style.RESET_ALL}")

        # --- Risk Management - Wards Against Ruin ---
        self.risk_per_trade_percentage: Decimal = self._get_env("RISK_PER_TRADE_PERCENTAGE", "0.005", cast_type=Decimal, min_val=Decimal("0.0001"), max_val=Decimal("0.1"), color=Fore.GREEN) # e.g., 0.5% risk
        self.atr_stop_loss_multiplier: Decimal = self._get_env("ATR_STOP_LOSS_MULTIPLIER", "1.5", cast_type=Decimal, min_val=Decimal("0.1"), color=Fore.GREEN) # Volatility-based ward distance
        self.max_order_usdt_amount: Decimal = self._get_env("MAX_ORDER_USDT_AMOUNT", "500.0", cast_type=Decimal, min_val=Decimal("1.0"), color=Fore.GREEN) # Limit on position value
        self.required_margin_buffer: Decimal = self._get_env("REQUIRED_MARGIN_BUFFER", "1.05", cast_type=Decimal, min_val=Decimal("1.0"), color=Fore.GREEN) # e.g., 5% safety margin

        # --- Trailing Stop Loss (Exchange Native) - The Adaptive Shield ---
        self.trailing_stop_percentage: Decimal = self._get_env("TRAILING_STOP_PERCENTAGE", "0.005", cast_type=Decimal, min_val=Decimal("0.0001"), max_val=Decimal("0.1"), color=Fore.GREEN) # e.g., 0.5% trailing distance
        self.trailing_stop_activation_offset_percent: Decimal = self._get_env("TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT", "0.001", cast_type=Decimal, min_val=Decimal("0.0"), max_val=Decimal("0.05"), color=Fore.GREEN) # e.g., 0.1% offset before activation

        # --- Strategy-Specific Parameters - Tuning the Chosen Path ---
        # Dual Supertrend
        self.st_atr_length: int = self._get_env("ST_ATR_LENGTH", 7, cast_type=int, min_val=1, color=Fore.CYAN)
        self.st_multiplier: Decimal = self._get_env("ST_MULTIPLIER", "2.5", cast_type=Decimal, min_val=Decimal("0.1"), color=Fore.CYAN)
        self.confirm_st_atr_length: int = self._get_env("CONFIRM_ST_ATR_LENGTH", 5, cast_type=int, min_val=1, color=Fore.CYAN)
        self.confirm_st_multiplier: Decimal = self._get_env("CONFIRM_ST_MULTIPLIER", "2.0", cast_type=Decimal, min_val=Decimal("0.1"), color=Fore.CYAN)
        # StochRSI + Momentum
        self.stochrsi_rsi_length: int = self._get_env("STOCHRSI_RSI_LENGTH", 14, cast_type=int, min_val=2, color=Fore.CYAN)
        self.stochrsi_stoch_length: int = self._get_env("STOCHRSI_STOCH_LENGTH", 14, cast_type=int, min_val=2, color=Fore.CYAN)
        self.stochrsi_k_period: int = self._get_env("STOCHRSI_K_PERIOD", 3, cast_type=int, min_val=1, color=Fore.CYAN)
        self.stochrsi_d_period: int = self._get_env("STOCHRSI_D_PERIOD", 3, cast_type=int, min_val=1, color=Fore.CYAN)
        self.stochrsi_overbought: Decimal = self._get_env("STOCHRSI_OVERBOUGHT", "80.0", cast_type=Decimal, min_val=Decimal("50"), max_val=Decimal("100"), color=Fore.CYAN)
        self.stochrsi_oversold: Decimal = self._get_env("STOCHRSI_OVERSOLD", "20.0", cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("50"), color=Fore.CYAN)
        self.momentum_length: int = self._get_env("MOMENTUM_LENGTH", 5, cast_type=int, min_val=1, color=Fore.CYAN)
        # Ehlers Fisher Transform
        self.ehlers_fisher_length: int = self._get_env("EHLERS_FISHER_LENGTH", 10, cast_type=int, min_val=2, color=Fore.CYAN)
        self.ehlers_fisher_signal_length: int = self._get_env("EHLERS_FISHER_SIGNAL_LENGTH", 1, cast_type=int, min_val=1, color=Fore.CYAN) # Default to 1 (often just the Fisher line itself)
        # Ehlers MA Cross (Placeholder - uses EMA)
        self.ehlers_fast_period: int = self._get_env("EHLERS_FAST_PERIOD", 10, cast_type=int, min_val=2, color=Fore.CYAN)
        self.ehlers_slow_period: int = self._get_env("EHLERS_SLOW_PERIOD", 30, cast_type=int, min_val=3, color=Fore.CYAN)
        if self.ehlers_fast_period >= self.ehlers_slow_period:
             logger.warning(f"{Fore.YELLOW}EHLERS_FAST_PERIOD ({self.ehlers_fast_period}) >= EHLERS_SLOW_PERIOD ({self.ehlers_slow_period}). Ensure this is intended.{Style.RESET_ALL}")

        # --- Confirmation Filters - Seeking Concordance in the Ether ---
        # Volume Analysis
        self.volume_ma_period: int = self._get_env("VOLUME_MA_PERIOD", 20, cast_type=int, min_val=2, color=Fore.YELLOW)
        self.volume_spike_threshold: Decimal = self._get_env("VOLUME_SPIKE_THRESHOLD", "1.5", cast_type=Decimal, min_val=Decimal("0.1"), color=Fore.YELLOW) # Multiplier over MA
        self.require_volume_spike_for_entry: bool = self._get_env("REQUIRE_VOLUME_SPIKE_FOR_ENTRY", "false", cast_type=bool, color=Fore.YELLOW)
        # Order Book Analysis
        self.order_book_depth: int = self._get_env("ORDER_BOOK_DEPTH", 10, cast_type=int, min_val=1, max_val=50, color=Fore.YELLOW) # Levels to analyze (Bybit typically allows up to 50)
        self.order_book_ratio_threshold_long: Decimal = self._get_env("ORDER_BOOK_RATIO_THRESHOLD_LONG", "1.2", cast_type=Decimal, min_val=Decimal("0.1"), color=Fore.YELLOW) # Bid/Ask ratio for long
        self.order_book_ratio_threshold_short: Decimal = self._get_env("ORDER_BOOK_RATIO_THRESHOLD_SHORT", "0.8", cast_type=Decimal, min_val=Decimal("0.1"), max_val=Decimal("2.0"), color=Fore.YELLOW) # Bid/Ask ratio for short (ensure <= threshold_long)
        self.fetch_order_book_per_cycle: bool = self._get_env("FETCH_ORDER_BOOK_PER_CYCLE", "false", cast_type=bool, color=Fore.YELLOW) # Fetch OB every cycle or only on signal?

        # --- ATR Calculation (for Initial SL) ---
        self.atr_calculation_period: int = self._get_env("ATR_CALCULATION_PERIOD", 14, cast_type=int, min_val=2, color=Fore.GREEN)

        # --- Termux SMS Alerts - Whispers Through the Digital Veil ---
        self.enable_sms_alerts: bool = self._get_env("ENABLE_SMS_ALERTS", "false", cast_type=bool, color=Fore.MAGENTA)
        self.sms_recipient_number: Optional[str] = self._get_env("SMS_RECIPIENT_NUMBER", None, color=Fore.MAGENTA) # No default, must be set if enabled
        self.sms_timeout_seconds: int = self._get_env("SMS_TIMEOUT_SECONDS", 30, cast_type=int, min_val=5, color=Fore.MAGENTA)
        if self.enable_sms_alerts and not self.sms_recipient_number:
            logger.warning(f"{Fore.YELLOW}SMS alerts enabled, but SMS_RECIPIENT_NUMBER rune is missing. Alerts will not be sent.{Style.RESET_ALL}")
            self.enable_sms_alerts = False # Disable if number is missing

        # --- CCXT / API Parameters - Tuning the Connection ---
        self.default_recv_window: int = 10000  # Milliseconds for API request validity (Bybit default is 5000, increased for potential latency)
        self.order_book_fetch_limit: int = max(50, self.order_book_depth) # Ensure sufficient depth fetched (Bybit max is 50 for L2)
        self.shallow_ob_fetch_depth: int = 5  # For quick price estimates
        self.order_fill_timeout_seconds: int = self._get_env("ORDER_FILL_TIMEOUT_SECONDS", 15, cast_type=int, min_val=1, color=Fore.YELLOW) # Wait time for market order fill

        # --- Internal Constants - Fixed Arcane Symbols ---
        self.SIDE_BUY: str = "buy"
        self.SIDE_SELL: str = "sell"
        self.POS_LONG: str = "Long"
        self.POS_SHORT: str = "Short"
        self.POS_NONE: str = "None"
        self.USDT_SYMBOL: str = "USDT"  # The stable anchor currency
        self.RETRY_COUNT: int = 3  # Attempts for certain API calls (e.g., leverage setting)
        self.RETRY_DELAY_SECONDS: int = 2  # Pause between retries
        self.API_FETCH_LIMIT_BUFFER: int = 10  # Extra candles to fetch beyond calculation needs
        self.POSITION_QTY_EPSILON: Decimal = Decimal("1e-9")  # Small value for float/decimal comparisons near zero
        self.POST_CLOSE_DELAY_SECONDS: int = 3  # Brief pause after closing a position before potentially re-entering

        logger.info(f"{Fore.MAGENTA}--- Configuration Runes Summoned and Verified ---{Style.RESET_ALL}")

    # pylint: disable=too-many-arguments,too-many-branches
    def _get_env(
        self,
        key: str,
        default: Any = None,
        cast_type: type = str,
        required: bool = False,
        sensitive: bool = False,
        min_val: Optional[Union[int, float, Decimal]] = None,
        max_val: Optional[Union[int, float, Decimal]] = None,
        color: str = Fore.WHITE
    ) -> Any:
        """
        Fetches an environment variable, casts its type, logs it, validates,
        and handles defaults/errors with arcane grace.

        Args:
            key: The environment variable key name.
            default: The default value if the key is not found.
            cast_type: The type to cast the value to (str, int, float, bool, Decimal).
            required: If True, raises ValueError if the key is not found and no default is provided.
            sensitive: If True, logs the value as '******'.
            min_val: Minimum allowed value for numeric types.
            max_val: Maximum allowed value for numeric types.
            color: Colorama Fore color for logging this parameter.

        Returns:
            The processed value.

        Raises:
            ValueError: If a required variable is missing or validation fails.
        """
        value_str = os.getenv(key)
        raw_value_repr = f"'{value_str}'" if value_str is not None else "Not Set"
        log_value_repr = '******' if sensitive and value_str is not None else raw_value_repr

        if value_str is None:
            if required and default is None:
                err_msg = f"CRITICAL: Required configuration rune '{key}' not found in the environment scroll (.env) and no default specified."
                logger.critical(f"{Back.RED}{Fore.WHITE}{err_msg}{Style.RESET_ALL}")
                raise ValueError(err_msg)
            elif required and default is not None:
                # Required but has a default
                logger.debug(f"{color}Summoning {key}: {log_value_repr} (Using Default: '{default}'){Style.RESET_ALL}")
                value = default # Use the default value directly
            else:
                # Not required, no value set
                 logger.debug(f"{color}Summoning {key}: {log_value_repr} (Using Default: '{default}'){Style.RESET_ALL}")
                 value = default # Use the default value directly
        else:
            # Value is set, attempt casting and validation
            log_default_msg = f"(Default: '{default}')" if default is not None else ""
            logger.debug(f"{color}Summoning {key}: {log_value_repr} {log_default_msg}{Style.RESET_ALL}")
            original_value_str = value_str # Keep original for error messages
            try:
                if cast_type == bool:
                    value = value_str.lower() in ['true', '1', 'yes', 'y']
                elif cast_type == Decimal:
                    value = Decimal(value_str)
                elif cast_type in [int, float]:
                    value = cast_type(value_str) # type: ignore
                else: # Default to string
                    value = value_str

                # Value Validation
                if min_val is not None and value < min_val:
                    err_msg = f"Validation failed for {key}: Value '{original_value_str}' ({value}) is below minimum {min_val}."
                    logger.critical(f"{Back.RED}{Fore.WHITE}{err_msg}{Style.RESET_ALL}")
                    raise ValueError(err_msg)
                if max_val is not None and value > max_val:
                    err_msg = f"Validation failed for {key}: Value '{original_value_str}' ({value}) is above maximum {max_val}."
                    logger.critical(f"{Back.RED}{Fore.WHITE}{err_msg}{Style.RESET_ALL}")
                    raise ValueError(err_msg)

            except (InvalidOperation, ValueError, TypeError) as e:
                # Error during casting or Decimal conversion
                err_msg = f"Invalid value for {key}: '{original_value_str}'. Expected type {cast_type.__name__}. Error: {e}"
                if required and default is None:
                     logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: {err_msg} and no default available.{Style.RESET_ALL}")
                     raise ValueError(err_msg) from e
                else:
                    logger.warning(f"{Fore.YELLOW}{err_msg}. Using default: '{default}'{Style.RESET_ALL}")
                    value = default # Fallback to default on casting/validation error if not strictly required or default exists

        # Final check if value is None after processing (e.g., default was None)
        if value is None and required:
            err_msg = f"CRITICAL: Required configuration rune '{key}' ended up with no value after processing defaults/errors."
            logger.critical(f"{Back.RED}{Fore.WHITE}{err_msg}{Style.RESET_ALL}")
            raise ValueError(err_msg)

        return value

# --- Global Objects - Instantiated Arcana ---
try:
    # Forge the configuration object; this runs the __init__ including validation
    CONFIG = Config()
except ValueError as config_error:
    # Error already logged critically within Config init or _get_env
    logger.info(f"{Fore.RED}Configuration spell failed. Check .env file and logs. Terminating.{Style.RESET_ALL}")
    sys.exit(1) # Exit if configuration is invalid


# --- Helper Functions - Minor Cantrips ---

def safe_decimal_conversion(value: Any, default: Decimal = Decimal("0.0")) -> Decimal:
    """
    Safely converts a value to a Decimal, returning a default if conversion fails.
    Handles None, strings, ints, floats, and existing Decimals.
    """
    if value is None:
        return default
    if isinstance(value, Decimal):
        return value
    try:
        # Convert to string first to handle floats more reliably than direct Decimal(float)
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        logger.warning(f"Could not convert '{value}' (type: {type(value).__name__}) to Decimal, using default {default}")
        return default

def format_order_id(order_id: Optional[Union[str, int]]) -> str:
    """Returns the last 6 characters of an order ID or 'N/A' for brevity and safety."""
    if order_id is None:
        return "N/A"
    order_id_str = str(order_id)
    return f"...{order_id_str[-6:]}" if len(order_id_str) > 6 else order_id_str

# --- Precision Formatting - Shaping the Numbers ---

def format_price(exchange: ccxt.Exchange, symbol: str, price: Union[float, Decimal]) -> str:
    """
    Formats a price according to the market's precision rules using CCXT.
    Handles potential errors and Decimal input.
    """
    try:
        # CCXT formatting methods typically expect float input
        price_float = float(price)
        return exchange.price_to_precision(symbol, price_float)
    except AttributeError:
         logger.error(f"{Fore.RED}Exchange object missing 'price_to_precision' method for {symbol}.{Style.RESET_ALL}")
         # Fallback: Use Decimal's normalize to remove trailing zeros, adjust precision if needed
         return str(safe_decimal_conversion(price).quantize(Decimal('1e-8'), rounding=ROUND_HALF_UP).normalize())
    except (ccxt.ExchangeError, ValueError, TypeError) as e:
        logger.error(f"{Fore.RED}Error shaping price {price} for {symbol}: {e}{Style.RESET_ALL}")
        # Fallback: Use Decimal's normalize as a best effort
        return str(safe_decimal_conversion(price).normalize())

def format_amount(exchange: ccxt.Exchange, symbol: str, amount: Union[float, Decimal]) -> str:
    """
    Formats an amount (quantity) according to the market's precision rules using CCXT.
    Handles potential errors and Decimal input.
    """
    try:
        # CCXT formatting methods typically expect float input
        amount_float = float(amount)
        return exchange.amount_to_precision(symbol, amount_float)
    except AttributeError:
         logger.error(f"{Fore.RED}Exchange object missing 'amount_to_precision' method for {symbol}.{Style.RESET_ALL}")
         # Fallback: Use Decimal's normalize, adjust precision if needed
         return str(safe_decimal_conversion(amount).quantize(Decimal('1e-8'), rounding=ROUND_HALF_UP).normalize())
    except (ccxt.ExchangeError, ValueError, TypeError) as e:
        logger.error(f"{Fore.RED}Error shaping amount {amount} for {symbol}: {e}{Style.RESET_ALL}")
        # Fallback: Use Decimal's normalize as a best effort
        return str(safe_decimal_conversion(amount).normalize())

# --- Termux SMS Alert Function - Sending Whispers ---

def send_sms_alert(message: str) -> bool:
    """
    Sends an SMS alert using the Termux:API, whispering through the digital veil.

    Args:
        message: The text message content.

    Returns:
        True if the SMS command was executed successfully (return code 0), False otherwise.
    """
    if not CONFIG.enable_sms_alerts:
        logger.debug("SMS alerts disabled by configuration.")
        return False
    # This check should be redundant due to config validation, but belts and braces
    if not CONFIG.sms_recipient_number:
        logger.warning(f"{Fore.YELLOW}SMS alerts enabled, but SMS_RECIPIENT_NUMBER rune is missing.{Style.RESET_ALL}")
        return False

    try:
        # Prepare the command spell
        command: List[str] = ['termux-sms-send', '-n', CONFIG.sms_recipient_number, message]
        logger.info(f"{Fore.MAGENTA}Dispatching SMS whisper to {CONFIG.sms_recipient_number} (Timeout: {CONFIG.sms_timeout_seconds}s)...{Style.RESET_ALL}")
        logger.debug(f"SMS Content: {message}")

        # Execute the spell via subprocess
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False, # Don't raise exception on non-zero exit code, handle it manually
            timeout=CONFIG.sms_timeout_seconds
        )

        if result.returncode == 0:
            logger.success(f"{Fore.MAGENTA}SMS whisper dispatched successfully.{Style.RESET_ALL}")
            return True
        else:
            # Log failure details
            stderr_output = result.stderr.strip() if result.stderr else "No stderr output"
            logger.error(f"{Fore.RED}SMS whisper failed. Return Code: {result.returncode}, Stderr: {stderr_output}{Style.RESET_ALL}")
            return False
    except FileNotFoundError:
        logger.error(f"{Fore.RED}SMS failed: 'termux-sms-send' command not found. Ensure Termux:API is installed and configured.{Style.RESET_ALL}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"{Fore.RED}SMS failed: Command timed out after {CONFIG.sms_timeout_seconds}s.{Style.RESET_ALL}")
        return False
    except Exception as e:
        logger.error(f"{Fore.RED}SMS failed: Unexpected disturbance during dispatch: {e}{Style.RESET_ALL}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return False

# --- Exchange Initialization - Opening the Portal ---

def initialize_exchange() -> Optional[ccxt.Exchange]:
    """
    Initializes and returns the CCXT Bybit exchange instance, opening a portal.
    Performs basic checks for connectivity and authentication.

    Returns:
        An initialized ccxt.Exchange object, or None if initialization fails.
    """
    logger.info(f"{Fore.BLUE}Opening portal to Bybit via CCXT...{Style.RESET_ALL}")
    # API Key/Secret presence checked during Config init, but double-check here
    if not CONFIG.api_key or not CONFIG.api_secret:
        logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: API Key/Secret runes missing. Cannot open portal.{Style.RESET_ALL}")
        send_sms_alert("[Pyrmethus] CRITICAL: API keys missing. Spell failed.")
        return None

    try:
        # Forging the connection using credentials from Config
        exchange_options = {
            "apiKey": CONFIG.api_key,
            "secret": CONFIG.api_secret,
            "enableRateLimit": True,  # Respect the exchange spirits' rate limits
            "options": {
                "defaultType": "linear",  # Crucial for USDT perpetuals on Bybit V5
                "recvWindow": CONFIG.default_recv_window,
                "adjustForTimeDifference": True, # Attempt to sync clock with server
                # Consider adding 'brokerId' or 'Referer' if applicable/provided
            },
        }
        exchange = ccxt.bybit(exchange_options)

        # Test connectivity and authentication
        logger.debug("Loading market structures from Bybit...")
        exchange.load_markets(True)  # Force reload for fresh data
        logger.debug("Market structures loaded.")

        logger.debug("Checking initial balance for authentication verification...")
        # Fetching balance implicitly tests API key validity
        balance = exchange.fetch_balance(params={'category': 'linear'}) # V5 needs category
        logger.debug(f"Initial balance check successful. USDT Available: {balance.get('USDT', {}).get('free', 'N/A')}")

        logger.success(f"{Fore.GREEN}{Style.BRIGHT}Portal to Bybit Opened (LIVE SCALPING MODE - EXTREME CAUTION!).{Style.RESET_ALL}")
        send_sms_alert("[Pyrmethus] Portal opened & authenticated successfully.")
        return exchange

    except ccxt.AuthenticationError as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}Authentication failed: {e}. Check API keys, IP whitelist, and permissions.{Style.RESET_ALL}")
        send_sms_alert(f"[Pyrmethus] CRITICAL: Authentication FAILED: {e}. Spell failed.")
    except ccxt.NetworkError as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}Network disturbance during portal opening: {e}. Check internet connection and Bybit status.{Style.RESET_ALL}")
        send_sms_alert(f"[Pyrmethus] CRITICAL: Network Error on Init: {e}. Spell failed.")
    except ccxt.ExchangeNotAvailable as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}Exchange Not Available: {e}. Bybit might be down for maintenance.{Style.RESET_ALL}")
        send_sms_alert(f"[Pyrmethus] CRITICAL: Exchange Unavailable on Init: {e}. Spell failed.")
    except ccxt.ExchangeError as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}Exchange spirit rejected portal opening: {e}. Check Bybit status or API documentation.{Style.RESET_ALL}")
        send_sms_alert(f"[Pyrmethus] CRITICAL: Exchange Error on Init: {e}. Spell failed.")
    except Exception as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}Unexpected chaos during portal opening: {e}{Style.RESET_ALL}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        send_sms_alert(f"[Pyrmethus] CRITICAL: Unexpected Init Error: {type(e).__name__}. Spell failed.")

    return None # Return None if any exception occurred

# --- Indicator Calculation Functions - Scrying the Market ---

def calculate_supertrend(
    df: pd.DataFrame, length: int, multiplier: Decimal, prefix: str = ""
) -> pd.DataFrame:
    """
    Calculates the Supertrend indicator using pandas_ta.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
        length: ATR period for Supertrend.
        multiplier: Multiplier for ATR.
        prefix: Optional prefix for the resulting columns (e.g., "confirm_").

    Returns:
        DataFrame with added columns:
        - f'{prefix}supertrend': The Supertrend line value (Decimal).
        - f'{prefix}trend': Boolean trend direction (True=Up, False=Down).
        - f'{prefix}st_long': Boolean signal for trend flipping to Long.
        - f'{prefix}st_short': Boolean signal for trend flipping to Short.
    """
    col_prefix = f"{prefix}" if prefix else ""
    target_cols = [f"{col_prefix}supertrend", f"{col_prefix}trend", f"{col_prefix}st_long", f"{col_prefix}st_short"]
    # Define expected column names from pandas_ta (uses float in name)
    st_float_mult = float(multiplier)
    st_col = f"SUPERT_{length}_{st_float_mult}"
    st_trend_col = f"SUPERTd_{length}_{st_float_mult}"
    st_long_col = f"SUPERTl_{length}_{st_float_mult}"
    st_short_col = f"SUPERTs_{length}_{st_float_mult}"
    raw_st_cols = [st_col, st_trend_col, st_long_col, st_short_col]
    required_input_cols = ["high", "low", "close"]

    # Input validation
    if df is None or df.empty or not all(c in df.columns for c in required_input_cols) or len(df) < length + 1:
        logger.warning(f"{Fore.YELLOW}Scrying ({col_prefix}ST): Insufficient data (Len: {len(df) if df is not None else 0}, Need: {length + 1}).{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA # Add empty columns if they don't exist
        return df

    try:
        # pandas_ta expects float multiplier
        logger.debug(f"Scrying ({col_prefix}ST): Calculating with length={length}, multiplier={st_float_mult}")
        # Calculate using pandas_ta, appending results to the DataFrame
        df.ta.supertrend(length=length, multiplier=st_float_mult, append=True)

        # Check if pandas_ta created the expected raw columns
        if not all(c in df.columns for c in [st_col, st_trend_col, st_long_col, st_short_col]):
             # Find missing columns for better error message
            missing_cols = [c for c in raw_st_cols if c not in df.columns]
            raise KeyError(f"pandas_ta failed to create expected raw Supertrend columns: {missing_cols}")

        # Process the raw results: Convert values, interpret trends/signals
        df[f"{col_prefix}supertrend"] = df[st_col].apply(safe_decimal_conversion)
        # Trend: 1 for Uptrend, -1 for Downtrend. Convert to boolean.
        df[f"{col_prefix}trend"] = df[st_trend_col] == 1
        # Signals: pandas_ta st_long/st_short seem reversed or based on previous candle, calculate manually
        prev_trend_direction = df[st_trend_col].shift(1)
        # Long flip: Previous trend was down (-1), current trend is up (1)
        df[f"{col_prefix}st_long"] = (prev_trend_direction == -1) & (df[st_trend_col] == 1)
        # Short flip: Previous trend was up (1), current trend is down (-1)
        df[f"{col_prefix}st_short"] = (prev_trend_direction == 1) & (df[st_trend_col] == -1)

        # Log the latest readings
        last_st_val = df[f'{col_prefix}supertrend'].iloc[-1]
        if pd.notna(last_st_val):
            last_trend_bool = df[f'{col_prefix}trend'].iloc[-1]
            last_trend_str = 'Up' if last_trend_bool else 'Down'
            trend_color = Fore.GREEN if last_trend_bool else Fore.RED
            signal_str = ('LONG FLIP' if df[f'{col_prefix}st_long'].iloc[-1] else
                          ('SHORT FLIP' if df[f'{col_prefix}st_short'].iloc[-1] else 'Hold'))
            signal_color = Fore.GREEN if signal_str == 'LONG FLIP' else (Fore.RED if signal_str == 'SHORT FLIP' else Fore.CYAN)
            logger.debug(f"Scrying ({col_prefix}ST({length},{multiplier})): Trend={trend_color}{last_trend_str}{Style.RESET_ALL}, Val={last_st_val:.4f}, Signal={signal_color}{signal_str}{Style.RESET_ALL}")
        else:
            logger.debug(f"Scrying ({col_prefix}ST({length},{multiplier})): Resulted in NA for last candle.")

    except Exception as e:
        logger.error(f"{Fore.RED}Scrying ({col_prefix}ST): Error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        for col in target_cols: df[col] = pd.NA # Nullify results on error
    finally:
        # Clean up raw columns from pandas_ta regardless of success/failure
        cols_to_drop = [c for c in raw_st_cols if c in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)

    return df

def analyze_volume_atr(
    df: pd.DataFrame, atr_len: int, vol_ma_len: int
) -> Dict[str, Optional[Decimal]]:
    """
    Calculates ATR, Volume Moving Average, and checks for volume spikes.

    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume' columns.
        atr_len: Period for ATR calculation.
        vol_ma_len: Period for Volume Moving Average.

    Returns:
        A dictionary containing:
        - 'atr': Average True Range value (Decimal) or None.
        - 'volume_ma': Volume Moving Average value (Decimal) or None.
        - 'last_volume': Last candle's volume (Decimal) or None.
        - 'volume_ratio': Ratio of last volume to volume MA (Decimal) or None.
    """
    results: Dict[str, Optional[Decimal]] = {"atr": None, "volume_ma": None, "last_volume": None, "volume_ratio": None}
    required_cols = ["high", "low", "close", "volume"]
    min_len = max(atr_len, vol_ma_len) + 1 # Need at least period+1 data points

    # Input validation
    if df is None or df.empty or not all(c in df.columns for c in required_cols) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Scrying (Vol/ATR): Insufficient data (Len: {len(df) if df is not None else 0}, Need: {min_len}).{Style.RESET_ALL}")
        return results

    try:
        # Calculate ATR (Average True Range) - Measure of volatility
        logger.debug(f"Scrying (ATR): Calculating with length={atr_len}")
        atr_col = f"ATRr_{atr_len}" # pandas_ta default name
        df.ta.atr(length=atr_len, append=True)
        if atr_col in df.columns:
            last_atr_raw = df[atr_col].iloc[-1]
            if pd.notna(last_atr_raw):
                results["atr"] = safe_decimal_conversion(last_atr_raw)
            df.drop(columns=[atr_col], errors='ignore', inplace=True) # Clean up raw column
        else:
             logger.warning(f"{Fore.YELLOW}Scrying (ATR): Column '{atr_col}' not found after calculation.{Style.RESET_ALL}")

        # Calculate Volume Moving Average and Ratio - Measure of market energy
        logger.debug(f"Scrying (Volume): Calculating MA with length={vol_ma_len}")
        volume_ma_col = 'volume_ma' # Temporary column name
        # Use rolling mean, require at least half the window period for calculation
        df[volume_ma_col] = df['volume'].rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2)).mean()
        last_vol_ma_raw = df[volume_ma_col].iloc[-1]
        last_vol_raw = df['volume'].iloc[-1]

        if pd.notna(last_vol_ma_raw): results["volume_ma"] = safe_decimal_conversion(last_vol_ma_raw)
        if pd.notna(last_vol_raw): results["last_volume"] = safe_decimal_conversion(last_vol_raw)

        # Calculate Volume Ratio (Last Volume / Volume MA)
        if results["volume_ma"] is not None and results["volume_ma"] > CONFIG.POSITION_QTY_EPSILON and results["last_volume"] is not None:
            try:
                results["volume_ratio"] = results["last_volume"] / results["volume_ma"]
            except InvalidOperation: # Handles potential issues like division by very small number if MA is epsilon
                 logger.warning(f"Could not calculate volume ratio ({results['last_volume']} / {results['volume_ma']})")
                 results["volume_ratio"] = None
        else:
            results["volume_ratio"] = None # Cannot calculate ratio if MA is zero/negligible or volume is missing

        # Clean up temporary MA column
        if volume_ma_col in df.columns:
            df.drop(columns=[volume_ma_col], errors='ignore', inplace=True)

        # Log results
        atr_str = f"{results['atr']:.5f}" if results['atr'] else 'N/A'
        vol_ma_str = f"{results['volume_ma']:.2f}" if results['volume_ma'] else 'N/A'
        last_vol_str = f"{results['last_volume']:.2f}" if results['last_volume'] else 'N/A'
        vol_ratio_str = f"{results['volume_ratio']:.2f}x" if results['volume_ratio'] else 'N/A'
        logger.debug(f"Scrying Results: ATR({atr_len})={Fore.CYAN}{atr_str}{Style.RESET_ALL}, LastVol={last_vol_str}, VolMA({vol_ma_len})={vol_ma_str}, VolRatio={Fore.YELLOW}{vol_ratio_str}{Style.RESET_ALL}")

    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (Vol/ATR): Error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        results = {k: None for k in results} # Nullify results on error

    return results

def calculate_stochrsi_momentum(
    df: pd.DataFrame, rsi_len: int, stoch_len: int, k: int, d: int, mom_len: int
) -> pd.DataFrame:
    """
    Calculates Stochastic RSI and Momentum indicators.

    Args:
        df: DataFrame with 'close' column.
        rsi_len: RSI period for StochRSI.
        stoch_len: Stochastic period for StochRSI.
        k: %K period for StochRSI.
        d: %D period for StochRSI.
        mom_len: Period for Momentum.

    Returns:
        DataFrame with added columns:
        - 'stochrsi_k': %K line value (Decimal).
        - 'stochrsi_d': %D line value (Decimal).
        - 'momentum': Momentum value (Decimal).
    """
    target_cols = ['stochrsi_k', 'stochrsi_d', 'momentum']
    # StochRSI needs RSI length + Stoch length. Momentum needs its length. Add buffer.
    min_len = max(rsi_len + stoch_len, mom_len) + d + 5 # Add d period and buffer
    stochrsi_raw_k_col = f"STOCHRSIk_{stoch_len}_{rsi_len}_{k}_{d}"
    stochrsi_raw_d_col = f"STOCHRSId_{stoch_len}_{rsi_len}_{k}_{d}"
    mom_raw_col = f"MOM_{mom_len}"

    # Input validation
    if df is None or df.empty or "close" not in df.columns or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Scrying (StochRSI/Mom): Insufficient data (Len: {len(df) if df is not None else 0}, Need ~{min_len}).{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA
        return df

    try:
        # Calculate StochRSI
        logger.debug(f"Scrying (StochRSI): Calculating with RSI={rsi_len}, Stoch={stoch_len}, K={k}, D={d}")
        # Use append=False first to avoid potential column name conflicts if run multiple times
        stochrsi_df = df.ta.stochrsi(length=stoch_len, rsi_length=rsi_len, k=k, d=d, append=False)
        if stochrsi_raw_k_col in stochrsi_df.columns:
            df['stochrsi_k'] = stochrsi_df[stochrsi_raw_k_col].apply(safe_decimal_conversion)
        else:
            logger.warning(f"{Fore.YELLOW}StochRSI K column '{stochrsi_raw_k_col}' not found after calculation.{Style.RESET_ALL}")
            df['stochrsi_k'] = pd.NA
        if stochrsi_raw_d_col in stochrsi_df.columns:
            df['stochrsi_d'] = stochrsi_df[stochrsi_raw_d_col].apply(safe_decimal_conversion)
        else:
            logger.warning(f"{Fore.YELLOW}StochRSI D column '{stochrsi_raw_d_col}' not found after calculation.{Style.RESET_ALL}")
            df['stochrsi_d'] = pd.NA

        # Calculate Momentum
        logger.debug(f"Scrying (Momentum): Calculating with length={mom_len}")
        df.ta.mom(length=mom_len, append=True) # Append momentum directly
        if mom_raw_col in df.columns:
            df['momentum'] = df[mom_raw_col].apply(safe_decimal_conversion)
            df.drop(columns=[mom_raw_col], errors='ignore', inplace=True) # Clean up raw column
        else:
            logger.warning(f"{Fore.YELLOW}Momentum column '{mom_raw_col}' not found after calculation.{Style.RESET_ALL}")
            df['momentum'] = pd.NA

        # Log latest values
        k_val, d_val, mom_val = df['stochrsi_k'].iloc[-1], df['stochrsi_d'].iloc[-1], df['momentum'].iloc[-1]
        if pd.notna(k_val) and pd.notna(d_val) and pd.notna(mom_val):
            k_cond = "OB" if k_val > CONFIG.stochrsi_overbought else ("OS" if k_val < CONFIG.stochrsi_oversold else "Mid")
            d_cond = "OB" if d_val > CONFIG.stochrsi_overbought else ("OS" if d_val < CONFIG.stochrsi_oversold else "Mid")
            mom_dir = "Up" if mom_val > CONFIG.POSITION_QTY_EPSILON else ("Down" if mom_val < -CONFIG.POSITION_QTY_EPSILON else "Flat")

            k_color = Fore.RED if k_cond=="OB" else (Fore.GREEN if k_cond=="OS" else Fore.CYAN)
            d_color = Fore.RED if d_cond=="OB" else (Fore.GREEN if d_cond=="OS" else Fore.CYAN)
            mom_color = Fore.GREEN if mom_dir=="Up" else (Fore.RED if mom_dir=="Down" else Fore.WHITE)

            logger.debug(f"Scrying (StochRSI/Mom): K={k_color}{k_val:.2f} ({k_cond}){Style.RESET_ALL}, D={d_color}{d_val:.2f} ({d_cond}){Style.RESET_ALL}, Mom={mom_color}{mom_val:.4f} ({mom_dir}){Style.RESET_ALL}")
        else:
            logger.debug("Scrying (StochRSI/Mom): Resulted in NA for last candle.")

    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (StochRSI/Mom): Error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        for col in target_cols: df[col] = pd.NA # Nullify results on error

    return df

def calculate_ehlers_fisher(df: pd.DataFrame, length: int, signal: int) -> pd.DataFrame:
    """
    Calculates the Ehlers Fisher Transform indicator.

    Args:
        df: DataFrame with 'high', 'low' columns.
        length: Period for the Fisher calculation.
        signal: Period for the signal line (usually 1).

    Returns:
        DataFrame with added columns:
        - 'ehlers_fisher': Fisher Transform line value (Decimal).
        - 'ehlers_signal': Signal line value (Decimal).
    """
    target_cols = ['ehlers_fisher', 'ehlers_signal']
    min_len = length + signal + 5 # Add buffer
    fish_raw_col = f"FISHERT_{length}_{signal}"
    signal_raw_col = f"FISHERTs_{length}_{signal}"

    # Input validation
    if df is None or df.empty or not all(c in df.columns for c in ["high", "low"]) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Scrying (EhlersFisher): Insufficient data (Len: {len(df) if df is not None else 0}, Need ~{min_len}).{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA
        return df

    try:
        logger.debug(f"Scrying (EhlersFisher): Calculating with length={length}, signal={signal}")
        # Calculate separately first
        fisher_df = df.ta.fisher(length=length, signal=signal, append=False)
        if fish_raw_col in fisher_df.columns:
             df['ehlers_fisher'] = fisher_df[fish_raw_col].apply(safe_decimal_conversion)
        else:
             logger.warning(f"{Fore.YELLOW}Ehlers Fisher column '{fish_raw_col}' not found after calculation.{Style.RESET_ALL}")
             df['ehlers_fisher'] = pd.NA
        if signal_raw_col in fisher_df.columns:
            df['ehlers_signal'] = fisher_df[signal_raw_col].apply(safe_decimal_conversion)
        else:
            # Signal is often just the Fisher line shifted, might not exist if signal=1
            logger.debug(f"Ehlers Signal column '{signal_raw_col}' not found (expected if signal=1). Using Fisher value.")
            df['ehlers_signal'] = df['ehlers_fisher'] # Use Fisher if signal line is missing (common for signal=1)

        # Log latest values
        fish_val, sig_val = df['ehlers_fisher'].iloc[-1], df['ehlers_signal'].iloc[-1]
        if pd.notna(fish_val) and pd.notna(sig_val):
             cross_state = "Above" if fish_val > sig_val else ("Below" if fish_val < sig_val else "Cross")
             cross_color = Fore.GREEN if cross_state == "Above" else (Fore.RED if cross_state == "Below" else Fore.YELLOW)
             logger.debug(f"Scrying (EhlersFisher({length},{signal})): Fisher={Fore.CYAN}{fish_val:.4f}{Style.RESET_ALL}, Signal={Fore.MAGENTA}{sig_val:.4f}{Style.RESET_ALL} ({cross_color}{cross_state}{Style.RESET_ALL})")
        else:
             logger.debug("Scrying (EhlersFisher): Resulted in NA for last candle.")

    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (EhlersFisher): Error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        for col in target_cols: df[col] = pd.NA # Nullify results on error

    return df

def calculate_ehlers_ma(df: pd.DataFrame, fast_len: int, slow_len: int) -> pd.DataFrame:
    """
    Calculates 'Ehlers' Moving Averages (Placeholder: Uses standard EMA).

    Args:
        df: DataFrame with 'close' column.
        fast_len: Period for the fast EMA.
        slow_len: Period for the slow EMA.

    Returns:
        DataFrame with added columns:
        - 'fast_ema': Fast Exponential Moving Average value (Decimal).
        - 'slow_ema': Slow Exponential Moving Average value (Decimal).
    """
    target_cols = ['fast_ema', 'slow_ema']
    min_len = max(fast_len, slow_len) + 5 # Add buffer for EMA calculation stability

    # Input validation
    if df is None or df.empty or "close" not in df.columns or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Scrying (EhlersMA/EMA): Insufficient data (Len: {len(df) if df is not None else 0}, Need ~{min_len}).{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA
        return df

    try:
        # *** PYRMETHUS NOTE: Using standard EMA as a placeholder. ***
        # The true Ehlers Super Smoother or Instantaneous Trendline involve more complex filters.
        # If `pandas_ta` offers a suitable Ehlers filter (e.g., `supersmoother`, `itrend`), consider using it.
        # Otherwise, implement the filter manually or accept EMA as an approximation for this strategy path.
        logger.warning(f"{Fore.YELLOW}{Style.DIM}Scrying (EhlersMA): Using EMA as placeholder for Ehlers Moving Average. Verify strategy suitability.{Style.RESET_ALL}")
        logger.debug(f"Scrying (EhlersMA - EMA Placeholder): Calculating Fast EMA({fast_len}), Slow EMA({slow_len})")

        # Calculate EMAs using pandas_ta
        df['fast_ema'] = df.ta.ema(length=fast_len).apply(safe_decimal_conversion)
        df['slow_ema'] = df.ta.ema(length=slow_len).apply(safe_decimal_conversion)

        # Log latest values
        fast_val, slow_val = df['fast_ema'].iloc[-1], df['slow_ema'].iloc[-1]
        if pd.notna(fast_val) and pd.notna(slow_val):
            cross_state = "Fast Above" if fast_val > slow_val else ("Fast Below" if fast_val < slow_val else "Cross")
            cross_color = Fore.GREEN if cross_state == "Fast Above" else (Fore.RED if cross_state == "Fast Below" else Fore.YELLOW)
            logger.debug(f"Scrying (EhlersMA/EMA({fast_len},{slow_len})): Fast={Fore.GREEN}{fast_val:.4f}{Style.RESET_ALL}, Slow={Fore.RED}{slow_val:.4f}{Style.RESET_ALL} ({cross_color}{cross_state}{Style.RESET_ALL})")
        else:
             logger.debug("Scrying (EhlersMA/EMA): Resulted in NA for last candle.")

    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (EhlersMA/EMA): Error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        for col in target_cols: df[col] = pd.NA # Nullify results on error

    return df

def analyze_order_book(
    exchange: ccxt.Exchange, symbol: str, depth: int, fetch_limit: int
) -> Dict[str, Optional[Decimal]]:
    """
    Fetches and analyzes L2 order book pressure and spread.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The market symbol.
        depth: Number of price levels to sum volume for ratio calculation.
        fetch_limit: Number of levels to request from the API (>= depth).

    Returns:
        A dictionary containing:
        - 'bid_ask_ratio': Ratio of cumulative bid volume to ask volume at specified depth (Decimal) or None.
        - 'spread': Difference between best ask and best bid (Decimal) or None.
        - 'best_bid': Best bid price (Decimal) or None.
        - 'best_ask': Best ask price (Decimal) or None.
    """
    results: Dict[str, Optional[Decimal]] = {"bid_ask_ratio": None, "spread": None, "best_bid": None, "best_ask": None}
    logger.debug(f"Order Book Scrying: Fetching L2 {symbol} (Depth:{depth}, Limit:{fetch_limit})...")

    # Check if the exchange supports fetching L2 order book
    if not exchange.has.get('fetchL2OrderBook'):
        logger.warning(f"{Fore.YELLOW}Order Book Scrying: fetchL2OrderBook not supported by {exchange.id}. Cannot peer into depth.{Style.RESET_ALL}")
        return results

    try:
        # Fetching the order book's current state
        # Bybit V5 uses fetchL2OrderBook, limit constraints apply (e.g., max 50)
        order_book = exchange.fetch_l2_order_book(symbol, limit=fetch_limit)
        # Order book structure: {'bids': [[price, amount], ...], 'asks': [[price, amount], ...], ...}
        bids: List[List[Union[float, str]]] = order_book.get('bids', [])
        asks: List[List[Union[float, str]]] = order_book.get('asks', [])

        # Extract best bid/ask with Decimal precision
        # Bids are sorted high to low, Asks low to high
        best_bid = safe_decimal_conversion(bids[0][0]) if bids and len(bids[0]) > 0 else None
        best_ask = safe_decimal_conversion(asks[0][0]) if asks and len(asks[0]) > 0 else None
        results["best_bid"] = best_bid
        results["best_ask"] = best_ask

        # Calculate spread
        if best_bid is not None and best_ask is not None and best_bid > 0 and best_ask > 0:
            # Ensure bid is not higher than ask (can happen in crossed books, though rare)
            if best_ask >= best_bid:
                results["spread"] = best_ask - best_bid
                logger.debug(f"OB Scrying: Best Bid={Fore.GREEN}{best_bid:.4f}{Style.RESET_ALL}, Ask={Fore.RED}{best_ask:.4f}{Style.RESET_ALL}, Spread={Fore.YELLOW}{results['spread']:.4f}{Style.RESET_ALL}")
            else:
                logger.warning(f"{Fore.YELLOW}OB Scrying: Crossed book detected (Bid {best_bid} > Ask {best_ask}). Spread calculation skipped.{Style.RESET_ALL}")
                results["spread"] = Decimal("0.0") # Or None, depending on desired handling
        else:
            logger.debug(f"OB Scrying: Best Bid={best_bid or 'N/A'}, Ask={best_ask or 'N/A'} (Spread N/A)")

        # Sum volumes within the specified depth using Decimal
        # Take min(depth, len(bids/asks)) to avoid IndexError if OB is shallower than requested depth
        bid_vol = sum(safe_decimal_conversion(bid[1]) for bid in bids[:min(depth, len(bids))] if len(bid) > 1)
        ask_vol = sum(safe_decimal_conversion(ask[1]) for ask in asks[:min(depth, len(asks))] if len(ask) > 1)
        logger.debug(f"OB Scrying (Depth {depth}): Total BidVol={Fore.GREEN}{bid_vol:.4f}{Style.RESET_ALL}, Total AskVol={Fore.RED}{ask_vol:.4f}{Style.RESET_ALL}")

        # Calculate Bid/Ask Volume Ratio
        if ask_vol > CONFIG.POSITION_QTY_EPSILON: # Avoid division by zero or negligible volume
            try:
                results["bid_ask_ratio"] = bid_vol / ask_vol
                ratio_str = f"{results['bid_ask_ratio']:.3f}"
                ratio_color = Fore.GREEN if results["bid_ask_ratio"] >= CONFIG.order_book_ratio_threshold_long else \
                              (Fore.RED if results["bid_ask_ratio"] <= CONFIG.order_book_ratio_threshold_short else Fore.YELLOW)
                logger.debug(f"OB Scrying Ratio (Bid/Ask): {ratio_color}{ratio_str}{Style.RESET_ALL}")
            except (InvalidOperation, Exception) as e:
                logger.warning(f"{Fore.YELLOW}Error calculating OB ratio ({bid_vol}/{ask_vol}): {e}{Style.RESET_ALL}")
                results["bid_ask_ratio"] = None
        else:
            logger.debug(f"OB Scrying Ratio: N/A (Ask volume zero or negligible at depth {depth})")
            # Ratio is effectively infinite if bids exist and asks are zero, handle as needed
            results["bid_ask_ratio"] = None # Or set to a very large number if that signals strong bids

    except (ccxt.NetworkError, ccxt.ExchangeError, IndexError) as e:
        logger.warning(f"{Fore.YELLOW}Order Book Scrying Error for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(f"Traceback (OB Scrying):\n{traceback.format_exc()}")
        results = {k: None for k in results} # Reset results on error
    except Exception as e:
        logger.error(f"{Fore.RED}Unexpected error during Order Book Scrying for {symbol}: {e}{Style.RESET_ALL}")
        logger.debug(f"Traceback (OB Scrying):\n{traceback.format_exc()}")
        results = {k: None for k in results}

    return results

# --- Data Fetching - Gathering Etheric Data Streams ---

def get_market_data(
    exchange: ccxt.Exchange, symbol: str, interval: str, limit: int
) -> Optional[pd.DataFrame]:
    """
    Fetches and prepares OHLCV data from the exchange.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The market symbol.
        interval: The timeframe interval (e.g., '1m', '5m').
        limit: The number of candles to fetch.

    Returns:
        A pandas DataFrame containing OHLCV data with a datetime index,
        or None if fetching or processing fails.
    """
    if not exchange.has.get("fetchOHLCV"):
        logger.error(f"{Fore.RED}Data Fetch: Exchange '{exchange.id}' does not support fetchOHLCV.{Style.RESET_ALL}")
        return None

    try:
        logger.debug(f"Data Fetch: Gathering {limit} OHLCV candles for {symbol} ({interval})...")
        # Channeling the data stream from the exchange
        # Bybit V5 fetchOHLCV requires 'category' for perpetuals (linear/inverse)
        market = exchange.market(symbol)
        params = {'category': 'linear'} if market.get('linear') else ({'category': 'inverse'} if market.get('inverse') else {})
        if not params: logger.warning(f"Could not determine category (linear/inverse) for {symbol}, fetchOHLCV might fail.")

        ohlcv: List[List[Union[int, float, str]]] = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit, params=params)

        if not ohlcv:
            logger.warning(f"{Fore.YELLOW}Data Fetch: No OHLCV data returned for {symbol} ({interval}) with params {params}. Market asleep or issue with request?{Style.RESET_ALL}")
            return None

        # Weaving data into a DataFrame structure
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Data Cleaning and Type Conversion
        # 1. Convert timestamp to datetime and set as index
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True) # Assume UTC timestamps from exchange
            df.set_index("timestamp", inplace=True)
        except Exception as time_e:
            logger.error(f"{Fore.RED}Data Fetch: Error converting timestamp column: {time_e}{Style.RESET_ALL}")
            return None # Cannot proceed without valid timestamps

        # 2. Convert OHLCV columns to numeric, coercing errors to NaN
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 3. Check for and handle gaps (NaNs) introduced by coercion or missing data
        if df.isnull().values.any():
            nan_counts = df.isnull().sum()
            logger.warning(f"{Fore.YELLOW}Data Fetch: OHLCV contains gaps (NaNs) after conversion/fetch:\n{nan_counts[nan_counts > 0]}\nAttempting forward fill...{Style.RESET_ALL}")
            df.ffill(inplace=True) # Fill gaps with the previous valid observation

            # Check again after forward fill (handles NaNs except at the beginning)
            if df.isnull().values.any():
                nan_counts_after_ffill = df.isnull().sum()
                logger.warning(f"{Fore.YELLOW}Gaps remain after ffill (likely at the start):\n{nan_counts_after_ffill[nan_counts_after_ffill > 0]}\nAttempting backward fill...{Style.RESET_ALL}")
                df.bfill(inplace=True) # Fill remaining gaps (usually at start) with the next valid observation

                # Final check - if NaNs still exist, the dataset is likely unusable
                if df.isnull().values.any():
                    logger.error(f"{Fore.RED}Data Fetch: Gaps persist after ffill/bfill. Cannot proceed with unreliable data.{Style.RESET_ALL}")
                    return None # Cannot proceed if gaps remain

        # 4. Ensure columns are suitable for Decimal conversion later if needed (they should be float64 now)
        # logger.debug(f"Data types after cleaning:\n{df.dtypes}")

        logger.debug(f"Data Fetch: Woven {len(df)} cleaned OHLCV candles for {symbol}.")
        return df

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger.warning(f"{Fore.YELLOW}Data Fetch: Disturbance gathering OHLCV for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
    except KeyError as e:
         logger.error(f"{Fore.RED}Data Fetch: Market structure issue for {symbol}: {e}. Was the market loaded correctly?{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"{Fore.RED}Data Fetch: Unexpected error processing OHLCV for {symbol}: {e}{Style.RESET_ALL}")
        logger.debug(f"Traceback (Data Fetch):\n{traceback.format_exc()}")

    return None # Return None if any error occurred

# --- Position & Order Management - Manipulating Market Presence ---

def get_current_position(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    """
    Fetches current position details for a specific symbol using Bybit V5 API specifics.
    Focuses on 'One-Way' position mode (positionIdx=0).

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The unified market symbol (e.g., 'BTC/USDT:USDT').

    Returns:
        A dictionary representing the position:
        {'side': 'Long', 'Short', or 'None',
         'qty': Position size as Decimal (absolute value),
         'entry_price': Average entry price as Decimal}
        Returns default (None, 0.0, 0.0) if no position or on error.
    """
    default_pos: Dict[str, Any] = {'side': CONFIG.POS_NONE, 'qty': Decimal("0.0"), 'entry_price': Decimal("0.0")}
    market: Optional[Dict[str, Any]] = None
    market_id: Optional[str] = None # Exchange-specific market ID (e.g., 'BTCUSDT')

    try:
        # Identify the market structure to get the ID and type
        market = exchange.market(symbol)
        market_id = market['id']
        category = 'linear' if market.get('linear') else ('inverse' if market.get('inverse') else None)
        if not category:
             logger.error(f"{Fore.RED}Position Check: Cannot determine category (linear/inverse) for market {symbol}. Aborting check.{Style.RESET_ALL}")
             return default_pos

    except (ccxt.BadSymbol, KeyError) as e:
        logger.error(f"{Fore.RED}Position Check: Failed to identify market structure for '{symbol}': {e}{Style.RESET_ALL}")
        return default_pos
    except Exception as e:
         logger.error(f"{Fore.RED}Position Check: Unexpected error getting market info for '{symbol}': {e}{Style.RESET_ALL}")
         return default_pos

    try:
        # Check if the exchange instance supports fetching positions
        if not exchange.has.get('fetchPositions'):
            logger.warning(f"{Fore.YELLOW}Position Check: fetchPositions spell not available for {exchange.id}. Cannot determine position.{Style.RESET_ALL}")
            return default_pos

        # Bybit V5 requires 'category' and optionally 'symbol' or 'settleCoin'
        params = {'category': category}
        # Fetching for a specific symbol is generally more efficient if supported well
        logger.debug(f"Position Check: Querying positions for {symbol} (MarketID: {market_id}, Category: {category})...")

        # Summon position data from the exchange for the specific symbol
        # Note: ccxt might internally fetch all and filter, or pass symbol to API if supported
        fetched_positions = exchange.fetch_positions(symbols=[symbol], params=params)

        # Find the active position in 'One-Way' mode (positionIdx=0) for the target symbol
        active_pos_info = None
        for pos_data in fetched_positions:
            # CCXT structure might vary, but 'info' usually holds raw data
            info = pos_data.get('info', {})
            pos_market_id = info.get('symbol')
            pos_idx = int(info.get('positionIdx', -1)) # Default to -1 if missing
            pos_side_v5 = info.get('side', 'None') # V5: 'Buy' (Long), 'Sell' (Short), 'None' (Flat)
            size_str = info.get('size', '0')

            # Filter: Match market ID, One-Way mode (idx=0), and ensure it's not a flat entry ('None')
            if pos_market_id == market_id and pos_idx == 0 and pos_side_v5 != 'None':
                 # Check if the size is effectively non-zero
                size = safe_decimal_conversion(size_str)
                if abs(size) > CONFIG.POSITION_QTY_EPSILON:
                    active_pos_info = info # Found the active position raw data
                    logger.debug(f"Found potential active position entry (idx=0): {active_pos_info}")
                    break # Assume only one active position for the symbol in One-Way mode

        # Parse the details from the found active position
        if active_pos_info:
            try:
                size = safe_decimal_conversion(active_pos_info.get('size'))
                # V5 uses 'avgPrice' for the average entry price of the position
                entry_price = safe_decimal_conversion(active_pos_info.get('avgPrice'))
                # Determine side based on V5 'side' field
                side = CONFIG.POS_LONG if active_pos_info.get('side') == 'Buy' else CONFIG.POS_SHORT

                # Validate parsed values
                if size <= CONFIG.POSITION_QTY_EPSILON or entry_price <= Decimal("0"):
                     logger.warning(f"{Fore.YELLOW}Position Check: Parsed active position has invalid size ({size}) or entry price ({entry_price}). Treating as flat. Data: {active_pos_info}{Style.RESET_ALL}")
                     return default_pos

                pos_color = Fore.GREEN if side == CONFIG.POS_LONG else Fore.RED
                logger.info(f"{pos_color}Position Check: Found ACTIVE {side} position: Qty={size:.8f} @ Entry={entry_price:.4f}{Style.RESET_ALL}")
                # Return absolute quantity
                return {'side': side, 'qty': abs(size), 'entry_price': entry_price}
            except Exception as parse_err:
                 logger.warning(f"{Fore.YELLOW}Position Check: Error parsing details from active position data: {parse_err}. Data: {active_pos_info}{Style.RESET_ALL}")
                 return default_pos # Return default on parsing error
        else:
            # No position matching the criteria was found
            logger.info(f"{Fore.BLUE}Position Check: No active One-Way (idx=0) position found for {market_id}. Currently Flat.{Style.RESET_ALL}")
            return default_pos

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger.warning(f"{Fore.YELLOW}Position Check: Disturbance querying positions for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        # Potentially transient error, return default but log warning
    except Exception as e:
        logger.error(f"{Fore.RED}Position Check: Unexpected error querying positions for {symbol}: {e}{Style.RESET_ALL}")
        logger.debug(f"Traceback (Position Check):\n{traceback.format_exc()}")

    return default_pos # Return default on API error or unexpected exception

def set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int) -> bool:
    """
    Sets the leverage for a given futures symbol using Bybit V5 API specifics.
    Attempts retries on failure.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The unified market symbol.
        leverage: The desired leverage multiplier (integer).

    Returns:
        True if leverage was set successfully or confirmed to be already set, False otherwise.
    """
    logger.info(f"{Fore.CYAN}Leverage Conjuring: Attempting to set {leverage}x for {symbol}...{Style.RESET_ALL}")
    market: Optional[Dict[str, Any]] = None
    market_id: Optional[str] = None

    try:
        # Verify it's a contract market where leverage applies
        market = exchange.market(symbol)
        market_id = market['id']
        if not market.get('contract'):
            logger.error(f"{Fore.RED}Leverage Conjuring Failed: Cannot set leverage for non-contract market: {symbol}.{Style.RESET_ALL}")
            return False
        category = 'linear' if market.get('linear') else ('inverse' if market.get('inverse') else None)
        if not category:
             logger.error(f"{Fore.RED}Leverage Conjuring Failed: Cannot determine category for {symbol}.{Style.RESET_ALL}")
             return False

    except (ccxt.BadSymbol, KeyError) as e:
         logger.error(f"{Fore.RED}Leverage Conjuring Failed: Failed to identify market structure for '{symbol}': {e}{Style.RESET_ALL}")
         return False
    except Exception as e:
         logger.error(f"{Fore.RED}Leverage Conjuring Failed: Unexpected error getting market info for '{symbol}': {e}{Style.RESET_ALL}")
         return False

    # Check if setLeverage is supported
    if not exchange.has.get('setLeverage'):
         logger.error(f"{Fore.RED}Leverage Conjuring Failed: Exchange adapter does not support setLeverage for {exchange.id}.{Style.RESET_ALL}")
         return False

    # Prepare Bybit V5 parameters: requires setting buy and sell leverage separately
    # Leverage value must be passed as a string for Bybit V5 via ccxt params
    leverage_str = str(leverage)
    params = {'buyLeverage': leverage_str, 'sellLeverage': leverage_str, 'category': category}

    for attempt in range(CONFIG.RETRY_COUNT):
        try:
            logger.debug(f"Attempt {attempt + 1}/{CONFIG.RETRY_COUNT}: Setting leverage for {market_id} with params: {params}")
            # The `leverage` argument to ccxt's set_leverage might also be needed alongside params
            response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
            # Successful response might be minimal or just confirm the setting
            logger.success(f"{Fore.GREEN}Leverage Conjuring: Successfully set to {leverage}x for {symbol}. Response: {response}{Style.RESET_ALL}")
            return True

        except ccxt.ExchangeError as e:
            # Check for common "already set" or "no modification needed" messages (case-insensitive)
            err_str = str(e).lower()
            common_success_msgs = ["leverage not modified", "same as requested", "same leverage", "success"] # Add "success" if Bybit returns it even without change
            if any(msg in err_str for msg in common_success_msgs):
                logger.info(f"{Fore.CYAN}Leverage Conjuring: Confirmed already set to {leverage}x for {symbol}. ({e}){Style.RESET_ALL}")
                return True

            # Log other exchange errors and decide whether to retry
            logger.warning(f"{Fore.YELLOW}Leverage Conjuring: Exchange resistance (Attempt {attempt + 1}/{CONFIG.RETRY_COUNT}) for {symbol}: {e}{Style.RESET_ALL}")
            if attempt < CONFIG.RETRY_COUNT - 1:
                logger.debug(f"Retrying leverage setting in {CONFIG.RETRY_DELAY_SECONDS}s...")
                time.sleep(CONFIG.RETRY_DELAY_SECONDS)
            else:
                logger.error(f"{Fore.RED}Leverage Conjuring Failed: Max retries ({CONFIG.RETRY_COUNT}) reached for {symbol}. Last error: {e}{Style.RESET_ALL}")
                # Consider sending SMS alert on final failure
                # send_sms_alert(f"[Pyrmethus] ERROR: Failed to set leverage {leverage}x for {symbol} after retries.")
                return False # Failed after retries

        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            logger.warning(f"{Fore.YELLOW}Leverage Conjuring: Network/Timeout disturbance (Attempt {attempt + 1}/{CONFIG.RETRY_COUNT}) for {symbol}: {e}{Style.RESET_ALL}")
            if attempt < CONFIG.RETRY_COUNT - 1:
                 logger.debug(f"Retrying leverage setting in {CONFIG.RETRY_DELAY_SECONDS}s...")
                 time.sleep(CONFIG.RETRY_DELAY_SECONDS)
            else:
                 logger.error(f"{Fore.RED}Leverage Conjuring Failed: Max retries ({CONFIG.RETRY_COUNT}) reached due to network/timeout issues for {symbol}. Last error: {e}{Style.RESET_ALL}")
                 return False # Failed after retries
        except Exception as e:
             logger.error(f"{Fore.RED}Leverage Conjuring Failed: Unexpected error on attempt {attempt + 1} for {symbol}: {e}{Style.RESET_ALL}")
             logger.debug(f"Traceback (Leverage Setting):\n{traceback.format_exc()}")
             # Stop retrying on unexpected errors
             return False

    return False # Should not be reached, but indicates failure

def close_position(
    exchange: ccxt.Exchange, symbol: str, position_to_close: Dict[str, Any], reason: str = "Signal"
) -> Optional[Dict[str, Any]]:
    """
    Closes the specified active position by placing a market order with reduceOnly flag.
    Includes re-validation of the position before attempting closure.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The unified market symbol.
        position_to_close: The position dictionary (from get_current_position) to close.
        reason: A string indicating the reason for closing (for logging/alerts).

    Returns:
        The filled market close order dictionary from CCXT on success, or None if:
        - No active position was found upon re-validation.
        - The closing order placement failed.
        - The position size was negligible.
    """
    initial_side = position_to_close.get('side', CONFIG.POS_NONE)
    initial_qty = position_to_close.get('qty', Decimal("0.0"))
    market_base = symbol.split('/')[0] # For concise alerts

    logger.info(f"{Fore.YELLOW}Banish Position Ritual: Initiated for {symbol}. Reason: {reason}. Expected state: {initial_side} Qty={initial_qty:.8f}{Style.RESET_ALL}")

    # === Re-validate the position just before closing ===
    logger.debug(f"Banish Position: Re-validating live position state for {symbol}...")
    live_position = get_current_position(exchange, symbol)
    live_position_side = live_position['side']
    live_amount_to_close = live_position['qty']

    if live_position_side == CONFIG.POS_NONE:
        logger.warning(f"{Fore.YELLOW}Banish Position: Re-validation shows NO active position for {symbol}. Aborting banishment ritual.{Style.RESET_ALL}")
        if initial_side != CONFIG.POS_NONE:
            # Log if the expected state doesn't match the live state (e.g., SL/TSL triggered between cycles)
            logger.warning(f"{Fore.YELLOW}Banish Position: Discrepancy detected! Expected {initial_side}, but position is now {live_position_side}.{Style.RESET_ALL}")
        return None # Nothing to close

    # Check if live amount is valid
    if live_amount_to_close <= CONFIG.POSITION_QTY_EPSILON:
         logger.warning(f"{Fore.YELLOW}Banish Position: Re-validated position quantity ({live_amount_to_close:.8f}) is negligible. Assuming already closed or closing.{Style.RESET_ALL}")
         return None # Treat negligible amount as effectively closed

    # Determine the market order side needed to close the position
    side_to_execute_close = CONFIG.SIDE_SELL if live_position_side == CONFIG.POS_LONG else CONFIG.SIDE_BUY

    try:
        # Format amount according to market rules
        amount_str = format_amount(exchange, symbol, live_amount_to_close)
        # Convert formatted string amount to float for CCXT create_order amount parameter
        # Ensure conversion is safe
        try:
            amount_float = float(amount_str)
            if amount_float <= float(CONFIG.POSITION_QTY_EPSILON): # Double check after float conversion
                 raise ValueError("Amount negligible after float conversion")
        except ValueError:
             logger.error(f"{Fore.RED}Banish Position: Closing amount '{amount_str}' became invalid/negligible after float conversion. Aborting.{Style.RESET_ALL}")
             return None

        # Prepare parameters for the closing order
        # Bybit V5 requires 'category' and 'reduceOnly' for closing orders
        market = exchange.market(symbol)
        category = 'linear' if market.get('linear') else ('inverse' if market.get('inverse') else None)
        if not category:
             logger.error(f"{Fore.RED}Banish Position: Cannot determine category for {symbol}. Aborting close.{Style.RESET_ALL}")
             return None
        params = {'reduceOnly': True, 'category': category}

        # Execute the market close order
        close_action_color = Back.YELLOW
        logger.warning(f"{close_action_color}{Fore.BLACK}{Style.BRIGHT}Banish Position ({reason}): Attempting to CLOSE {live_position_side} "
                       f"by executing {side_to_execute_close.upper()} MARKET {amount_str} {symbol} (reduceOnly=True)...{Style.RESET_ALL}")

        order = exchange.create_market_order(
            symbol=symbol,
            side=side_to_execute_close,
            amount=amount_float,
            params=params
        )

        # --- Process successful order placement ---
        # Parse order response safely using Decimal for financial fields
        fill_price = safe_decimal_conversion(order.get('average')) # 'average' is often the avg fill price
        filled_qty = safe_decimal_conversion(order.get('filled'))
        cost = safe_decimal_conversion(order.get('cost')) # Cost in quote currency (USDT)
        order_id_short = format_order_id(order.get('id'))

        # Log success with details
        # Check if filled quantity matches expected (allowing for small tolerance)
        qty_match_tolerance = CONFIG.POSITION_QTY_EPSILON * Decimal('10') # Slightly larger tolerance for fill vs request
        if abs(filled_qty - live_amount_to_close) > qty_match_tolerance:
             logger.warning(f"{Fore.YELLOW}Banish Position: Filled quantity ({filled_qty:.8f}) differs significantly from expected close amount ({live_amount_to_close:.8f}). Order ID: {order_id_short}{Style.RESET_ALL}")

        # Determine log color based on expected action vs actual execution
        close_log_color = Fore.GREEN # Assume success unless logic error found
        logger.success(f"{close_log_color}{Style.BRIGHT}Banish Position: Order ({reason}) for {symbol} successfully placed & likely filled. "
                       f"FilledQty: {filled_qty:.8f}, AvgFillPx: {fill_price:.4f}, Cost: {cost:.2f} USDT. ID:{order_id_short}{Style.RESET_ALL}")
        send_sms_alert(f"[{market_base}] BANISHED {live_position_side} {amount_str} @ ~{fill_price:.4f} ({reason}). ID:{order_id_short}")

        # Return the filled close order details dictionary
        return order

    except ccxt.InsufficientFunds as e:
         # This might happen if margin calculation was off or balance changed rapidly
         logger.error(f"{Fore.RED}Banish Position ({reason}): FAILED for {symbol} due to Insufficient Funds: {e}{Style.RESET_ALL}")
         send_sms_alert(f"[{market_base}] ERROR Banishing ({reason}): INSUFFICIENT FUNDS! Check account.")
         # Critical situation, position might remain open without enough margin
    except ccxt.ExchangeError as e:
        # Check for specific Bybit errors indicating already closed or zero position
        err_str = str(e).lower()
        # V5 common errors: 'order qty is not greater than 0', 'position size is zero', 'reduce-only rule violated' (if somehow size increased?)
        # 'order would not reduce position size' (common if already closing/closed)
        # 'The position size is zero' (110025)
        # 'Order quantity cannot be greater than configurable position size' (110043) -> indicates maybe SL/TP already triggered reducing size
        already_closed_errors = ["order qty is not greater than 0", "position size is zero", "order would not reduce position size", "110025", "110043"]
        if any(indicator in err_str for indicator in already_closed_errors):
             logger.warning(f"{Fore.YELLOW}Banish Position: Exchange indicates position already closed/closing or size mismatch (e.g., SL/TP triggered): {e}. Assuming banished.{Style.RESET_ALL}")
             return None # Treat as success (or non-actionable) in this specific case
        else:
            # Other exchange errors
            logger.error(f"{Fore.RED}Banish Position ({reason}): FAILED for {symbol} due to Exchange Error: {e}{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] ERROR Banishing ({reason}): Exchange Error {type(e).__name__}. Check logs.")
    except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
         logger.error(f"{Fore.RED}Banish Position ({reason}): FAILED for {symbol} due to Network/Timeout Error: {e}{Style.RESET_ALL}")
         # Might retry later if implemented, but for now signal failure
         send_sms_alert(f"[{market_base}] ERROR Banishing ({reason}): Network/Timeout. Check connection.")
    except ValueError as e: # Catch the float conversion error explicitly
         logger.error(f"{Fore.RED}Banish Position ({reason}): FAILED for {symbol} due to Value Error: {e}{Style.RESET_ALL}")
         # Error logged during float conversion attempt
    except Exception as e:
        # Catch any other unexpected exceptions
        logger.error(f"{Fore.RED}Banish Position ({reason}): FAILED for {symbol} due to Unexpected Error: {e}{Style.RESET_ALL}")
        logger.debug(f"Traceback (Close Position):\n{traceback.format_exc()}")
        send_sms_alert(f"[{market_base}] ERROR Banishing ({reason}): Unexpected {type(e).__name__}. Check logs.")

    return None # Indicate failure if any exception occurred

def calculate_position_size(
    equity: Decimal,
    risk_per_trade_pct: Decimal,
    entry_price: Decimal,
    stop_loss_price: Decimal,
    leverage: int,
    symbol: str,
    exchange: ccxt.Exchange
) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """
    Calculates the position size in base currency based on risk percentage,
    entry/stop prices, and equity. Uses Decimal precision throughout.

    Args:
        equity: Total account equity (in USDT).
        risk_per_trade_pct: Desired risk per trade as a decimal (e.g., 0.01 for 1%).
        entry_price: Estimated entry price.
        stop_loss_price: Calculated stop-loss price.
        leverage: The leverage being used.
        symbol: The market symbol (for fetching precision).
        exchange: Initialized CCXT exchange object.

    Returns:
        A tuple containing:
        - Calculated position size (Decimal) formatted to market precision, or None if calculation fails.
        - Estimated margin required for the position (Decimal), or None.
    """
    logger.debug(f"Risk Calc: Equity={equity:.4f} USDT, Risk%={risk_per_trade_pct:.4%}, Entry={entry_price:.4f}, SL={stop_loss_price:.4f}, Lev={leverage}x")

    # --- Input Validation ---
    if not (isinstance(equity, Decimal) and equity > Decimal("0")):
        logger.error(f"{Fore.RED}Risk Calc Error: Invalid equity ({equity}). Must be positive Decimal.{Style.RESET_ALL}")
        return None, None
    if not (isinstance(risk_per_trade_pct, Decimal) and 0 < risk_per_trade_pct < 1):
        logger.error(f"{Fore.RED}Risk Calc Error: Invalid risk percentage ({risk_per_trade_pct:.4%}). Must be Decimal between 0 and 1.{Style.RESET_ALL}")
        return None, None
    if not (isinstance(entry_price, Decimal) and entry_price > Decimal("0")):
        logger.error(f"{Fore.RED}Risk Calc Error: Invalid entry price ({entry_price}). Must be positive Decimal.{Style.RESET_ALL}")
        return None, None
    if not (isinstance(stop_loss_price, Decimal) and stop_loss_price > Decimal("0")):
        logger.error(f"{Fore.RED}Risk Calc Error: Invalid stop loss price ({stop_loss_price}). Must be positive Decimal.{Style.RESET_ALL}")
        return None, None
    if not (isinstance(leverage, int) and leverage > 0):
         logger.error(f"{Fore.RED}Risk Calc Error: Invalid leverage ({leverage}). Must be positive integer.{Style.RESET_ALL}")
         return None, None

    price_diff = abs(entry_price - stop_loss_price)
    if price_diff < CONFIG.POSITION_QTY_EPSILON: # Use epsilon for comparing price difference to zero
        logger.error(f"{Fore.RED}Risk Calc Error: Entry and SL prices are too close ({price_diff:.8f}). Cannot calculate risk accurately.{Style.RESET_ALL}")
        return None, None

    # --- Calculation ---
    # 1. Calculate the maximum USDT amount to risk on this trade
    risk_amount_usdt: Decimal = equity * risk_per_trade_pct
    logger.debug(f"Risk Calc: Max Risk Amount = {risk_amount_usdt:.4f} USDT")

    # 2. Calculate position size in base currency (e.g., BTC for BTC/USDT)
    # For linear contracts (like USDT perpetuals), the value of 1 unit of base currency changes with price.
    # The risk per unit of the asset is the price difference between entry and stop-loss.
    # Quantity = Total Risk Amount (USDT) / Risk per Unit (USDT)
    try:
        quantity_raw: Decimal = risk_amount_usdt / price_diff
    except InvalidOperation as e:
         logger.error(f"{Fore.RED}Risk Calc Error: Decimal operation failed during quantity calculation ({risk_amount_usdt} / {price_diff}): {e}{Style.RESET_ALL}")
         return None, None

    logger.debug(f"Risk Calc: Raw Quantity = {quantity_raw:.8f}")

    # --- Apply Market Precision ---
    try:
        # Format the raw quantity according to market precision rules via CCXT
        quantity_precise_str: str = format_amount(exchange, symbol, quantity_raw)
        # Convert the precision-formatted string back to Decimal
        quantity_precise: Decimal = Decimal(quantity_precise_str)
        logger.debug(f"Risk Calc: Precise Quantity (after formatting) = {quantity_precise:.8f}")
    except (ValueError, InvalidOperation) as e:
        logger.warning(f"{Fore.YELLOW}Risk Calc Warning: Failed to format quantity {quantity_raw:.8f} using exchange precision or convert back to Decimal. Error: {e}. Using raw value with basic quantization.{Style.RESET_ALL}")
        # Fallback: Quantize raw value to a reasonable number of decimal places if formatting fails
        # Determine a sensible fallback precision (e.g., 8 decimal places)
        fallback_precision = Decimal('1e-8')
        quantity_precise = quantity_raw.quantize(fallback_precision, rounding=ROUND_HALF_UP)
        logger.debug(f"Risk Calc: Fallback Quantized Quantity = {quantity_precise:.8f}")
    except Exception as e:
         logger.error(f"{Fore.RED}Risk Calc Error: Unexpected error during quantity precision shaping: {e}{Style.RESET_ALL}")
         return None, None


    # --- Final Checks & Margin Estimation ---
    if quantity_precise <= CONFIG.POSITION_QTY_EPSILON:
        logger.warning(f"{Fore.YELLOW}Risk Calc Warning: Calculated position quantity is negligible ({quantity_precise:.8f}) after precision formatting. RiskAmt={risk_amount_usdt:.4f}, PriceDiff={price_diff:.4f}. Cannot place order.{Style.RESET_ALL}")
        return None, None

    # Estimate position value and margin required (using the precise quantity)
    try:
        position_value_usdt: Decimal = quantity_precise * entry_price
        required_margin: Decimal = position_value_usdt / Decimal(leverage)
        logger.debug(f"Risk Calc Result: Quantity={Fore.CYAN}{quantity_precise:.8f}{Style.RESET_ALL}, Est. Value={position_value_usdt:.4f} USDT, Est. Margin={required_margin:.4f} USDT")
    except InvalidOperation as e:
         logger.error(f"{Fore.RED}Risk Calc Error: Decimal operation failed during final value/margin estimation: {e}{Style.RESET_ALL}")
         return None, None

    return quantity_precise, required_margin

def wait_for_order_fill(
    exchange: ccxt.Exchange, order_id: str, symbol: str, timeout_seconds: int
) -> Optional[Dict[str, Any]]:
    """
    Waits for a specific order to reach a 'closed' (usually filled for market orders)
    or failed ('canceled', 'rejected', 'expired') status.

    Args:
        exchange: Initialized CCXT exchange object.
        order_id: The ID of the order to monitor.
        symbol: The market symbol.
        timeout_seconds: Maximum time to wait in seconds.

    Returns:
        The order dictionary from CCXT if the order is confirmed filled ('closed').
        None if the order reaches a failed state or the timeout is exceeded.
    """
    start_time = time.monotonic() # Use monotonic clock for reliable time difference measurement
    order_id_short = format_order_id(order_id)
    logger.info(f"{Fore.CYAN}Observing order {order_id_short} ({symbol}) for fill (Timeout: {timeout_seconds}s)...{Style.RESET_ALL}")

    while time.monotonic() - start_time < timeout_seconds:
        try:
            # Query the order's status from the exchange
            # Bybit V5: fetchOrder requires category
            market = exchange.market(symbol)
            category = 'linear' if market.get('linear') else ('inverse' if market.get('inverse') else None)
            if not category: raise ValueError(f"Cannot determine category for {symbol} in wait_for_order_fill")
            params={'category': category}

            order = exchange.fetch_order(order_id, symbol, params=params)
            status = order.get('status') # Common statuses: 'open', 'closed', 'canceled', 'rejected', 'expired'
            logger.debug(f"Order {order_id_short} status check: {status}")

            if status == 'closed':
                # 'closed' usually means fully filled for market orders, or canceled/rejected and fully resolved.
                # We rely on the calling function (place_risked_market_order) to have placed a market order
                # and expect 'closed' to mean filled in that context.
                logger.success(f"{Fore.GREEN}Order {order_id_short} confirmed FILLED (status 'closed').{Style.RESET_ALL}")
                return order # Success - return the order details
            elif status in ['canceled', 'rejected', 'expired']:
                logger.error(f"{Fore.RED}Order {order_id_short} reached FAILED status: '{status}'.{Style.RESET_ALL}")
                # Log relevant info if available
                reason = order.get('info', {}).get('rejectReason', 'N/A')
                logger.error(f"Order {order_id_short} Failure Reason (if available): {reason}")
                return None # Failed state
            elif status == 'open':
                # Order is still open (or partially filled), continue polling
                logger.debug(f"Order {order_id_short} still 'open', continuing observation...")
            else:
                 # Unknown or unexpected status
                 logger.warning(f"{Fore.YELLOW}Order {order_id_short} has unexpected status: '{status}'. Continuing observation.{Style.RESET_ALL}")

            # Wait before the next poll
            time.sleep(0.75) # Check roughly every 750ms

        except ccxt.OrderNotFound:
            # This can happen briefly after placing, especially on busy exchanges. Keep trying.
            logger.warning(f"{Fore.YELLOW}Order {order_id_short} not found yet by exchange spirits. Retrying...{Style.RESET_ALL}")
            time.sleep(1.5) # Wait a bit longer if not found initially
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeError) as e:
            # Handle recoverable API errors
            logger.warning(f"{Fore.YELLOW}Disturbance checking order {order_id_short}: {type(e).__name__} - {e}. Retrying...{Style.RESET_ALL}")
            time.sleep(2) # Wait longer on error before retrying
        except ValueError as e: # Catch category error
             logger.error(f"{Fore.RED}Error in wait_for_order_fill: {e}{Style.RESET_ALL}")
             return None # Cannot proceed without category
        except Exception as e:
             logger.error(f"{Fore.RED}Unexpected error checking order {order_id_short}: {e}{Style.RESET_ALL}")
             logger.debug(f"Traceback (Wait Order Fill):\n{traceback.format_exc()}")
             time.sleep(2) # Wait before retrying unexpected error

    # If the loop finishes without returning, it timed out
    logger.error(f"{Fore.RED}Order {order_id_short} did not reach 'closed' or failed status within {timeout_seconds}s timeout.{Style.RESET_ALL}")
    # Consider attempting to cancel the order here if it timed out in 'open' state
    # try: exchange.cancel_order(order_id, symbol) except Exception: pass
    return None # Timeout failure

# pylint: disable=too-many-locals, too-many-statements, too-many-branches
def place_risked_market_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str, # Use CONFIG.SIDE_BUY or CONFIG.SIDE_SELL
    risk_percentage: Decimal,
    current_atr: Optional[Decimal],
    sl_atr_multiplier: Decimal,
    leverage: int,
    max_order_cap_usdt: Decimal,
    margin_check_buffer: Decimal,
    tsl_percent: Decimal,
    tsl_activation_offset_percent: Decimal
) -> Optional[Dict[str, Any]]:
    """
    Orchestrates placing a market entry order with calculated risk,
    waits for it to fill, and then places exchange-native fixed Stop Loss (SL)
    and Trailing Stop Loss (TSL) orders. Uses Decimal precision throughout.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The unified market symbol.
        side: 'buy' or 'sell'.
        risk_percentage: Risk per trade as a decimal (e.g., 0.01).
        current_atr: Current ATR value (Decimal) for SL calculation.
        sl_atr_multiplier: Multiplier for ATR to determine SL distance.
        leverage: Leverage to use.
        max_order_cap_usdt: Maximum position value allowed in USDT.
        margin_check_buffer: Buffer multiplier for margin check (e.g., 1.05 for 5%).
        tsl_percent: Trailing stop percentage as a decimal (e.g., 0.005 for 0.5%).
        tsl_activation_offset_percent: Activation price offset percentage (Decimal).

    Returns:
        The filled entry order dictionary from CCXT on success.
        None if any critical step fails (validation, order placement, fill confirmation, SL/TSL placement).
    """
    market_base = symbol.split('/')[0] # For concise alerts
    order_side_color = Fore.GREEN if side == CONFIG.SIDE_BUY else Fore.RED
    logger.info(f"{order_side_color}{Style.BRIGHT}Place Order Ritual: Initiating {side.upper()} for {symbol}...{Style.RESET_ALL}")

    # --- Pre-computation & Validation ---
    if current_atr is None or not isinstance(current_atr, Decimal) or current_atr <= Decimal("0"):
        logger.error(f"{Fore.RED}Place Order ({side.upper()}) Failed: Invalid or missing ATR ({current_atr}). Cannot calculate SL distance.{Style.RESET_ALL}")
        return None
    if side not in [CONFIG.SIDE_BUY, CONFIG.SIDE_SELL]:
        logger.error(f"{Fore.RED}Place Order Failed: Invalid side '{side}'. Must be '{CONFIG.SIDE_BUY}' or '{CONFIG.SIDE_SELL}'.{Style.RESET_ALL}")
        return None

    entry_price_estimate: Optional[Decimal] = None
    initial_sl_price_estimate: Optional[Decimal] = None
    final_quantity: Optional[Decimal] = None
    market: Optional[Dict[str, Any]] = None
    category: Optional[str] = None # linear / inverse

    try:
        # === 1. Gather Resources: Balance, Market Info, Limits ===
        logger.debug("Gathering resources: Balance, Market Structure, Limits...")
        market = exchange.market(symbol)
        market_id = market['id']
        category = 'linear' if market.get('linear') else ('inverse' if market.get('inverse') else None)
        if not category: raise ValueError(f"Cannot determine category for {symbol}")

        # Fetch balance using the determined category
        balance = exchange.fetch_balance(params={'category': category})
        limits = market.get('limits', {})
        amount_limits = limits.get('amount', {})
        price_limits = limits.get('price', {})
        min_qty_str = amount_limits.get('min')
        max_qty_str = amount_limits.get('max')
        min_price_str = price_limits.get('min')

        # Use safe conversion for limits, store as Decimal or None
        min_qty = safe_decimal_conversion(min_qty_str) if min_qty_str else None
        max_qty = safe_decimal_conversion(max_qty_str) if max_qty_str else None
        min_price = safe_decimal_conversion(min_price_str) if min_price_str else None
        logger.debug(f"Market Limits: MinQty={min_qty}, MaxQty={max_qty}, MinPrice={min_price}")

        # Extract USDT balance details (assuming USDT margined contract)
        usdt_balance = balance.get(CONFIG.USDT_SYMBOL, {})
        # Bybit V5 balance fields might differ, check API/CCXT response structure
        # Common fields: 'total' (equity), 'free' (available margin)
        usdt_total_equity = safe_decimal_conversion(usdt_balance.get('total'))
        usdt_free_margin = safe_decimal_conversion(usdt_balance.get('free'))

        if usdt_total_equity <= Decimal("0"):
            logger.error(f"{Fore.RED}Place Order ({side.upper()}) Failed: Zero or Invalid Total Equity ({usdt_total_equity:.4f} {CONFIG.USDT_SYMBOL}). Cannot proceed.{Style.RESET_ALL}")
            return None
        if usdt_free_margin < Decimal("0"): # Free margin shouldn't be negative
            logger.error(f"{Fore.RED}Place Order ({side.upper()}) Failed: Negative Free Margin ({usdt_free_margin:.4f} {CONFIG.USDT_SYMBOL}). Cannot proceed.{Style.RESET_ALL}")
            return None
        logger.debug(f"Resources: Equity={usdt_total_equity:.4f}, FreeMargin={usdt_free_margin:.4f} {CONFIG.USDT_SYMBOL}")

        # === 2. Estimate Entry Price - Peering into the immediate future ===
        logger.debug("Estimating entry price...")
        # Use shallow order book fetch for a quick estimate
        ob_data = analyze_order_book(exchange, symbol, CONFIG.shallow_ob_fetch_depth, CONFIG.shallow_ob_fetch_depth)
        best_ask = ob_data.get("best_ask")
        best_bid = ob_data.get("best_bid")

        if side == CONFIG.SIDE_BUY and best_ask:
            entry_price_estimate = best_ask
            logger.debug(f"Using best ask for BUY entry estimate: {entry_price_estimate:.4f}")
        elif side == CONFIG.SIDE_SELL and best_bid:
            entry_price_estimate = best_bid
            logger.debug(f"Using best bid for SELL entry estimate: {entry_price_estimate:.4f}")
        else:
            # Fallback: Fetch last traded price if OB data is unavailable or invalid
            logger.warning(f"{Fore.YELLOW}Order book estimate failed (Ask: {best_ask}, Bid: {best_bid}). Fetching ticker last price...{Style.RESET_ALL}")
            try:
                ticker = exchange.fetch_ticker(symbol)
                entry_price_estimate = safe_decimal_conversion(ticker.get('last'))
                if entry_price_estimate <= 0: raise ValueError("Ticker price invalid")
                logger.debug(f"Using ticker last price for estimate: {entry_price_estimate:.4f}")
            except (ccxt.NetworkError, ccxt.ExchangeError, ValueError, Exception) as e:
                logger.error(f"{Fore.RED}Failed to fetch or use ticker price for estimate: {e}{Style.RESET_ALL}")
                return None # Cannot proceed without entry price estimate

        if not entry_price_estimate or entry_price_estimate <= 0: # Should be caught above, but double check
             logger.error(f"{Fore.RED}Invalid entry price estimate ({entry_price_estimate}). Cannot proceed.{Style.RESET_ALL}")
             return None

        # === 3. Calculate Initial Stop Loss Price (Estimate) - The First Ward ===
        logger.debug("Calculating initial SL price estimate...")
        sl_distance = current_atr * sl_atr_multiplier
        initial_sl_price_raw = (entry_price_estimate - sl_distance) if side == CONFIG.SIDE_BUY else (entry_price_estimate + sl_distance)

        # Ensure SL price is valid (positive) and respects minimum price limit if applicable
        if initial_sl_price_raw <= 0:
            logger.error(f"{Fore.RED}Invalid Initial SL price calculation resulted in non-positive value ({initial_sl_price_raw:.4f}). Cannot proceed.{Style.RESET_ALL}")
            return None
        if min_price is not None and initial_sl_price_raw < min_price:
            logger.warning(f"{Fore.YELLOW}Initial SL price estimate {initial_sl_price_raw:.4f} below min price {min_price}. Adjusting SL to min price.{Style.RESET_ALL}")
            initial_sl_price_raw = min_price

        # Format the estimated SL price according to market rules for calculation step
        initial_sl_price_estimate_str = format_price(exchange, symbol, initial_sl_price_raw)
        initial_sl_price_estimate = safe_decimal_conversion(initial_sl_price_estimate_str) # Back to Decimal
        if initial_sl_price_estimate <= 0: # Check again after formatting
             logger.error(f"{Fore.RED}Invalid Initial SL price after formatting ({initial_sl_price_estimate_str}). Cannot proceed.{Style.RESET_ALL}")
             return None

        logger.info(f"Calculated Initial SL Price (Estimate) ~ {Fore.YELLOW}{initial_sl_price_estimate:.4f}{Style.RESET_ALL} (ATR: {current_atr:.4f}, Mult: {sl_atr_multiplier}, Dist: {sl_distance:.4f})")

        # === 4. Calculate Position Size - Determining the Energy Input ===
        logger.debug("Calculating position size based on risk...")
        calc_qty, est_req_margin = calculate_position_size(
            usdt_total_equity, risk_percentage, entry_price_estimate, initial_sl_price_estimate,
            leverage, symbol, exchange
        )
        if calc_qty is None or est_req_margin is None:
            logger.error(f"{Fore.RED}Failed risk calculation. Cannot determine position size.{Style.RESET_ALL}")
            return None
        final_quantity = calc_qty # Start with the risk-calculated quantity

        # === 5. Apply Max Order Cap - Limiting the Power ===
        logger.debug("Applying max order cap...")
        pos_value_estimate = final_quantity * entry_price_estimate
        logger.debug(f"Estimated position value before cap: {pos_value_estimate:.4f} USDT")
        if pos_value_estimate > max_order_cap_usdt:
            logger.warning(f"{Fore.YELLOW}Calculated order value {pos_value_estimate:.4f} USDT > Cap {max_order_cap_usdt:.4f} USDT. Capping quantity.{Style.RESET_ALL}")
            # Recalculate quantity based on the cap
            try:
                final_quantity = max_order_cap_usdt / entry_price_estimate
                # Format the capped quantity according to market rules *immediately*
                final_quantity_str = format_amount(exchange, symbol, final_quantity)
                final_quantity = safe_decimal_conversion(final_quantity_str)
                if final_quantity <= CONFIG.POSITION_QTY_EPSILON:
                    raise ValueError("Quantity negligible after capping and formatting")
                # Recalculate estimated margin based on the capped quantity
                est_req_margin = (final_quantity * entry_price_estimate) / Decimal(leverage) # Use the new final_quantity
                logger.info(f"Capped Qty: {final_quantity:.8f}, New Est. Value: ~{max_order_cap_usdt:.4f}, New Est. Margin: {est_req_margin:.4f}")
            except (InvalidOperation, ValueError) as e:
                 logger.error(f"{Fore.RED}Error applying max order cap or formatting capped quantity: {e}. Aborting.{Style.RESET_ALL}")
                 return None

        # === 6. Check Limits & Margin Availability - Final Preparations ===
        logger.debug("Checking final quantity against limits and available margin...")
        if final_quantity <= CONFIG.POSITION_QTY_EPSILON: # Check again after potential capping
            logger.error(f"{Fore.RED}Final Quantity negligible after capping/formatting: {final_quantity:.8f}. Aborting.{Style.RESET_ALL}")
            return None

        # Check against minimum order size
        if min_qty is not None and final_quantity < min_qty:
            logger.error(f"{Fore.RED}Final Quantity {final_quantity:.8f} < Market Min Allowed {min_qty}. Cannot place order.{Style.RESET_ALL}")
            # Provide context for failure
            logger.error(f"  (Calculated from: Equity={usdt_total_equity:.2f}, Risk={risk_percentage:.3%}, Entry={entry_price_estimate:.4f}, SL={initial_sl_price_estimate:.4f}, Cap={max_order_cap_usdt})")
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Final Qty {final_quantity:.8f} < Min Qty {min_qty}.")
            return None
        # Check against maximum order size (should be handled by cap, but good failsafe)
        if max_qty is not None and final_quantity > max_qty:
            # This shouldn't happen if capping logic is correct, but log a warning if it does
            logger.warning(f"{Fore.YELLOW}Final Quantity {final_quantity:.8f} > Market Max Allowed {max_qty}. This might indicate an issue. Clamping to max.{Style.RESET_ALL}")
            final_quantity = max_qty
            # Re-format and re-calculate margin if clamped here (unlikely path)
            final_quantity_str = format_amount(exchange, symbol, final_quantity)
            final_quantity = safe_decimal_conversion(final_quantity_str)
            est_req_margin = (final_quantity * entry_price_estimate) / Decimal(leverage)

        # Final margin calculation based on the definitive final_quantity
        final_req_margin = est_req_margin # Use the margin calculated based on final_quantity
        req_margin_buffered = final_req_margin * margin_check_buffer # Add safety buffer

        # Check if sufficient FREE margin is available (use free margin for placement check)
        logger.debug(f"Final Margin Check: Need ~{final_req_margin:.4f} (Buffered: {req_margin_buffered:.4f}), Available Free: {usdt_free_margin:.4f}")
        if usdt_free_margin < req_margin_buffered:
            logger.error(f"{Fore.RED}Insufficient FREE margin. Need ~{req_margin_buffered:.4f} {CONFIG.USDT_SYMBOL} (incl. {margin_check_buffer:.1%} buffer), Have {usdt_free_margin:.4f}{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Insufficient Free Margin (Need ~{req_margin_buffered:.2f})")
            return None

        logger.info(f"{Fore.GREEN}Final Order Details Check OK: Side={side.upper()}, Qty={final_quantity:.8f}, EstValue={final_quantity * entry_price_estimate:.4f}, EstMargin={final_req_margin:.4f}{Style.RESET_ALL}")

        # === 7. Place Entry Market Order - Unleashing the Energy ===
        entry_order: Optional[Dict[str, Any]] = None
        order_id: Optional[str] = None
        try:
            # Convert final Decimal quantity to float for CCXT amount parameter
            qty_float = float(final_quantity)

            entry_action_color = Back.GREEN if side == CONFIG.SIDE_BUY else Back.RED
            text_color = Fore.BLACK if side == CONFIG.SIDE_BUY else Fore.WHITE
            logger.warning(f"{entry_action_color}{text_color}{Style.BRIGHT}*** Placing {side.upper()} MARKET ENTRY: {qty_float:.8f} {symbol} ***{Style.RESET_ALL}")

            # Prepare parameters for Bybit V5 market entry
            entry_params = {'category': category, 'reduceOnly': False} # Ensure reduceOnly is False for entry

            # Create the market order
            entry_order = exchange.create_market_order(
                symbol=symbol,
                side=side,
                amount=qty_float,
                params=entry_params
            )

            order_id = entry_order.get('id')
            if not order_id:
                # This is unexpected and highly problematic
                logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Entry order placed but NO Order ID returned! Cannot track or place SL/TSL. Response: {entry_order}{Style.RESET_ALL}")
                send_sms_alert(f"[{market_base}] CRITICAL ORDER FAIL ({side.upper()}): Entry placed but NO ID received!")
                # Attempt to query recent orders or position? Difficult. Manual intervention required.
                # We cannot safely proceed without the ID.
                return None # Cannot proceed

            order_id_short = format_order_id(order_id)
            logger.success(f"{Fore.GREEN}Market Entry Order submitted. ID: {order_id_short}. Awaiting fill confirmation...{Style.RESET_ALL}")

        except (ccxt.InsufficientFunds, ccxt.ExchangeError, ccxt.NetworkError, ValueError) as e:
            # Catch specific errors during order placement
            logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}FAILED TO PLACE ENTRY ORDER ({side.upper()} {symbol}): {type(e).__name__} - {e}{Style.RESET_ALL}")
            logger.debug(f"Traceback (Place Entry Order):\n{traceback.format_exc()}")
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Entry placement failed: {type(e).__name__}")
            return None # Failed to place entry
        except Exception as e:
             # Catch unexpected errors
             logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}UNEXPECTED CRITICAL ERROR Placing Entry Order: {e}{Style.RESET_ALL}")
             logger.debug(f"Traceback (Place Entry Order):\n{traceback.format_exc()}")
             send_sms_alert(f"[{market_base}] CRITICAL ORDER FAIL ({side.upper()}): Unexpected entry placement error: {type(e).__name__}")
             return None


        # === 8. Wait for Entry Fill Confirmation - Observing the Impact ===
        logger.debug(f"Waiting for entry order {order_id_short} to fill...")
        filled_entry = wait_for_order_fill(exchange, order_id, symbol, CONFIG.order_fill_timeout_seconds)

        if not filled_entry:
            logger.error(f"{Fore.RED}Entry order {order_id_short} did not fill or failed confirmation within timeout.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Entry {order_id_short} fill timeout/fail.")
            # Try to cancel the potentially stuck order (might fail if already filled/gone/rejected)
            try:
                logger.warning(f"Attempting to cancel potentially stuck/unconfirmed order {order_id_short}...")
                cancel_params={'category': category}
                exchange.cancel_order(order_id, symbol, params=cancel_params)
                logger.info(f"Cancel request sent for order {order_id_short}.")
            except ccxt.OrderNotFound:
                 logger.warning(f"Order {order_id_short} not found for cancellation (likely already closed/rejected).")
            except Exception as cancel_e:
                logger.warning(f"Could not cancel order {order_id_short} (may be filled or already gone): {cancel_e}")
            # Whether cancel worked or not, we can't proceed reliably without a confirmed fill
            return None # Cannot proceed

        # === 9. Extract Actual Fill Details - Reading the Result ===
        logger.debug(f"Extracting fill details for order {order_id_short}...")
        # Use 'average' for average fill price, 'filled' for quantity, 'cost' for USDT cost
        avg_fill_price = safe_decimal_conversion(filled_entry.get('average'))
        filled_qty = safe_decimal_conversion(filled_entry.get('filled'))
        cost = safe_decimal_conversion(filled_entry.get('cost')) # Total cost in quote currency (USDT)
        fee = safe_decimal_conversion(filled_entry.get('fee', {}).get('cost')) # Extract fee if available

        # Validate fill details
        if avg_fill_price <= 0 or filled_qty <= CONFIG.POSITION_QTY_EPSILON:
            logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Invalid fill details received for order {order_id_short}: Price={avg_fill_price}, Qty={filled_qty}. Position state UNKNOWN! Manual check REQUIRED.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] CRITICAL ORDER FAIL ({side.upper()}): Invalid fill details {order_id_short}! MANUAL CHECK!")
            # Position might be open with bad data. We cannot safely place SL/TSL.
            # Return the problematic order details for logging, but signal overall failure.
            # Do NOT proceed to SL/TSL placement.
            return None # Treat as failure because we can't place stops

        # Check filled quantity against requested quantity (allow small tolerance)
        qty_discrepancy = abs(filled_qty - final_quantity)
        if qty_discrepancy > CONFIG.POSITION_QTY_EPSILON * Decimal('10'): # Check against original *requested* quantity
             logger.warning(f"{Fore.YELLOW}Fill quantity discrepancy: Requested={final_quantity:.8f}, Filled={filled_qty:.8f}. Discrepancy={qty_discrepancy:.8f}. ID: {order_id_short}{Style.RESET_ALL}")
             # Proceed, but use the ACTUAL filled quantity for SL/TSL

        fill_action_color = Back.GREEN if side == CONFIG.SIDE_BUY else Back.RED
        text_color = Fore.BLACK if side == CONFIG.SIDE_BUY else Fore.WHITE
        logger.success(f"{fill_action_color}{text_color}{Style.BRIGHT}ENTRY CONFIRMED: {order_id_short}. Filled: {filled_qty:.8f} {market_base} @ {avg_fill_price:.4f} USDT, Cost: {cost:.4f} USDT, Fee: ~{fee:.4f} USDT{Style.RESET_ALL}")

        # --- Post-Fill: Place Stop Loss and Trailing Stop ---
        # Use the ACTUAL filled quantity and average price for SL/TSL calculations

        # === 10. Calculate ACTUAL Stop Loss Price - Setting the Ward ===
        logger.debug("Calculating ACTUAL stop loss price based on fill...")
        # SL distance remains the same as calculated before based on ATR
        # actual_sl_distance = current_atr * sl_atr_multiplier # Recalculate or use stored sl_distance
        actual_sl_price_raw = (avg_fill_price - sl_distance) if side == CONFIG.SIDE_BUY else (avg_fill_price + sl_distance)

        # Apply min price constraint again based on actual fill
        if actual_sl_price_raw <= 0:
             logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Invalid ACTUAL SL price ({actual_sl_price_raw:.4f}) calculated based on fill price {avg_fill_price:.4f}. Cannot place SL! Position unprotected!{Style.RESET_ALL}")
             send_sms_alert(f"[{market_base}] CRITICAL ({side.upper()}): Invalid ACTUAL SL price! Position UNPROTECTED. Attempting emergency close.")
             # Attempt emergency close of the *actual* filled position
             close_reason = "Invalid SL Calc Post-Fill"
             # Need to reconstruct position dict with actual fill data
             pos_to_close = {'side': CONFIG.POS_LONG if side == CONFIG.SIDE_BUY else CONFIG.POS_SHORT, 'qty': filled_qty, 'entry_price': avg_fill_price}
             close_position(exchange, symbol, pos_to_close, reason=close_reason)
             return filled_entry # Return filled entry, but signal overall failure state

        if min_price is not None and actual_sl_price_raw < min_price:
             logger.warning(f"{Fore.YELLOW}Actual SL price {actual_sl_price_raw:.4f} below min price {min_price}. Adjusting SL to min price.{Style.RESET_ALL}")
             actual_sl_price_raw = min_price

        # Format the final SL price according to market rules
        actual_sl_price_str = format_price(exchange, symbol, actual_sl_price_raw)
        try:
            actual_sl_price_float = float(actual_sl_price_str) # For CCXT param
            if actual_sl_price_float <= 0: raise ValueError("SL price non-positive after formatting")
        except ValueError as e:
            logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Actual SL price '{actual_sl_price_str}' became invalid after formatting: {e}. Cannot place SL! Position unprotected!{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] CRITICAL ({side.upper()}): Invalid FORMATTED SL price! Pos UNPROTECTED. Attempt E-Close.")
            pos_to_close = {'side': CONFIG.POS_LONG if side == CONFIG.SIDE_BUY else CONFIG.POS_SHORT, 'qty': filled_qty, 'entry_price': avg_fill_price}
            close_position(exchange, symbol, pos_to_close, reason="Invalid Formatted SL")
            return filled_entry

        logger.info(f"Calculated ACTUAL Initial SL Price: {Fore.YELLOW}{actual_sl_price_str}{Style.RESET_ALL}")

        # === 11. Place Initial Fixed Stop Loss Order - The Static Ward ===
        sl_order_id_short = "N/A (Failed)"
        initial_sl_placed = False
        try:
            sl_side = CONFIG.SIDE_SELL if side == CONFIG.SIDE_BUY else CONFIG.SIDE_BUY # Opposite side for SL
            # Use actual filled quantity, formatted
            sl_qty_str = format_amount(exchange, symbol, filled_qty)
            sl_qty_float = float(sl_qty_str)
            if sl_qty_float <= float(CONFIG.POSITION_QTY_EPSILON):
                 raise ValueError(f"Stop loss quantity negligible after formatting: {sl_qty_str}")

            logger.info(f"{Fore.CYAN}Weaving Initial Fixed SL ({sl_atr_multiplier}*ATR)... Side: {sl_side.upper()}, Qty: {sl_qty_float:.8f}, TriggerPx: {actual_sl_price_str}{Style.RESET_ALL}")

            # Bybit V5 stop order params via CCXT:
            # type='stopMarket' or 'stopLimit' (using Market here)
            # side=opposite of entry
            # amount=filled quantity
            # params={'stopPrice': trigger_price_float, 'reduceOnly': True, 'category': category}
            sl_params = {'stopPrice': actual_sl_price_float, 'reduceOnly': True, 'category': category}

            # Create the Stop Market order
            sl_order = exchange.create_order(symbol, 'stopMarket', sl_side, sl_qty_float, params=sl_params)
            sl_order_id = sl_order.get('id')
            if not sl_order_id: raise ValueError("Stop Loss order placement did not return an ID.")

            sl_order_id_short = format_order_id(sl_order_id)
            logger.success(f"{Fore.GREEN}Initial Fixed SL ward placed successfully. ID: {sl_order_id_short}, Trigger: {actual_sl_price_str}{Style.RESET_ALL}")
            initial_sl_placed = True

        except (ccxt.ExchangeError, ccxt.NetworkError, ValueError) as e:
            # Handle failure to place initial SL
            logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}FAILED to place Initial Fixed SL ward: {type(e).__name__} - {e}{Style.RESET_ALL}")
            logger.debug(f"Traceback (Place Initial SL):\n{traceback.format_exc()}")
            send_sms_alert(f"[{market_base}] ERROR ({side.upper()}): Failed initial SL placement: {type(e).__name__}. Check position manually!")
            # Decide if we should proceed to TSL placement if initial SL fails.
            # For safety, maybe don't place TSL if initial SL failed, as the core protection is missing.
            # However, TSL *might* still work. Current choice: Log error and continue to TSL attempt.
        except Exception as e:
             logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}UNEXPECTED CRITICAL ERROR Placing Initial SL: {e}{Style.RESET_ALL}")
             logger.debug(f"Traceback (Place Initial SL):\n{traceback.format_exc()}")
             send_sms_alert(f"[{market_base}] CRITICAL ERROR ({side.upper()}): Unexpected initial SL placement error: {type(e).__name__}. Check manually!")
             # Stop trying to place orders for this entry.

        # === 12. Place Trailing Stop Loss Order - The Adaptive Shield ===
        tsl_order_id_short = "N/A (Not Placed/Failed)"
        tsl_placed = False
        tsl_act_price_str = "N/A"
        tsl_trail_value_str = "N/A"
        try:
            # Calculate TSL activation price based on actual fill price
            act_offset = avg_fill_price * tsl_activation_offset_percent
            act_price_raw = (avg_fill_price + act_offset) if side == CONFIG.SIDE_BUY else (avg_fill_price - act_offset)

            # Apply min price constraint to activation price
            if act_price_raw <= 0: raise ValueError(f"Invalid TSL activation price calculation ({act_price_raw:.4f})")
            if min_price is not None and act_price_raw < min_price:
                logger.warning(f"{Fore.YELLOW}TSL activation price {act_price_raw:.4f} below min price {min_price}. Adjusting to min price.{Style.RESET_ALL}")
                act_price_raw = min_price

            # Format activation price
            tsl_act_price_str = format_price(exchange, symbol, act_price_raw)
            tsl_act_price_float = float(tsl_act_price_str)
            if tsl_act_price_float <= 0: raise ValueError("TSL activation price non-positive after formatting")

            # Prepare TSL parameters
            tsl_side = CONFIG.SIDE_SELL if side == CONFIG.SIDE_BUY else CONFIG.SIDE_BUY # Opposite side
            # Bybit V5 uses 'trailingStop' param for percentage distance (e.g., "0.5" for 0.5%)
            # Convert our decimal percentage (e.g., 0.005) to percentage string (e.g., "0.5")
            # Ensure trailing percentage is within Bybit's limits (e.g., 0.1% to 10%) - check Bybit docs!
            # Assuming limits are handled by Config validation
            tsl_trail_value_str = str((tsl_percent * Decimal("100")).normalize()) # e.g., Decimal('0.005') -> '0.5'

            # Use actual filled quantity, formatted
            tsl_qty_str = format_amount(exchange, symbol, filled_qty)
            tsl_qty_float = float(tsl_qty_str)
            if tsl_qty_float <= float(CONFIG.POSITION_QTY_EPSILON):
                 raise ValueError(f"Trailing stop loss quantity negligible after formatting: {tsl_qty_str}")

            logger.info(f"{Fore.CYAN}Weaving Trailing SL ({tsl_percent:.2%})... Side: {tsl_side.upper()}, Qty: {tsl_qty_float:.8f}, Trail%: {tsl_trail_value_str}%, ActPx: {tsl_act_price_str}{Style.RESET_ALL}")

            # Bybit V5 TSL parameters via CCXT (using create_order with 'stopMarket' type):
            # 'trailingStop': Percentage value as a string (e.g., "0.5")
            # 'activePrice': Activation trigger price (float)
            # 'reduceOnly': Must be True
            # 'category': linear/inverse
            tsl_params = {
                'trailingStop': tsl_trail_value_str,
                'activePrice': tsl_act_price_float,
                'reduceOnly': True,
                'category': category,
                # 'tpslMode': 'Full' # Optional: Ensure it applies to the whole position if needed
            }

            # Create the Trailing Stop Market order
            tsl_order = exchange.create_order(symbol, 'stopMarket', tsl_side, tsl_qty_float, params=tsl_params)
            tsl_order_id = tsl_order.get('id')
            if not tsl_order_id: raise ValueError("Trailing Stop Loss order placement did not return an ID.")

            tsl_order_id_short = format_order_id(tsl_order_id)
            logger.success(f"{Fore.GREEN}Trailing SL shield placed successfully. ID: {tsl_order_id_short}, Trail%: {tsl_trail_value_str}%, ActPx: {tsl_act_price_str}{Style.RESET_ALL}")
            tsl_placed = True

        except (ccxt.ExchangeError, ccxt.NetworkError, ValueError) as e:
            logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}FAILED to place Trailing SL shield: {type(e).__name__} - {e}{Style.RESET_ALL}")
            logger.debug(f"Traceback (Place TSL):\n{traceback.format_exc()}")
            send_sms_alert(f"[{market_base}] ERROR ({side.upper()}): Failed TSL placement: {type(e).__name__}. Check manually!")
            # Position might be open with only the initial fixed SL (if that succeeded).
        except Exception as e:
             logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}UNEXPECTED CRITICAL ERROR Placing TSL: {e}{Style.RESET_ALL}")
             logger.debug(f"Traceback (Place TSL):\n{traceback.format_exc()}")
             send_sms_alert(f"[{market_base}] CRITICAL ERROR ({side.upper()}): Unexpected TSL placement error: {type(e).__name__}. Check manually!")

        # === Final Comprehensive SMS Alert (only if entry was successful) ===
        if filled_entry: # Check if entry fill was successful before sending summary
             sl_status = f"InitSL:{sl_order_id_short} @{actual_sl_price_str}" if initial_sl_placed else "InitSL:FAILED"
             tsl_status = f"TSL:{tsl_order_id_short} {tsl_trail_value_str}% Act@{tsl_act_price_str}" if tsl_placed else "TSL:FAILED"
             sms_msg = (f"[{market_base}] ENTERED {side.upper()} {filled_qty:.8f} @ {avg_fill_price:.4f}. "
                        f"{sl_status}. {tsl_status}. EntryID:{format_order_id(order_id)}")
             send_sms_alert(sms_msg)

        # Return the details of the successfully filled entry order
        # We return the entry order even if SL/TSL placement failed,
        # as the entry itself was successful. The calling logic might need this info.
        # The errors in SL/TSL placement are logged and alerted.
        return filled_entry

    except ValueError as e: # Catch category determination error, etc.
        logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Place Order Ritual ({side.upper()}): Prerequisite failed: {e}{Style.RESET_ALL}")
    except (ccxt.InsufficientFunds, ccxt.NetworkError, ccxt.ExchangeError) as e:
        # Catch errors occurring during setup (balance fetch, etc.) before order placement
        logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Place Order Ritual ({side.upper()}): Setup failed: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(f"Traceback (Place Order Setup):\n{traceback.format_exc()}")
        # Don't send SMS here as it might be redundant with more specific errors later
    except Exception as e:
        # Catch-all for unexpected errors during the setup phase
        logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Place Order Ritual ({side.upper()}): Unexpected critical error during setup: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(f"Traceback (Place Order Setup):\n{traceback.format_exc()}")
        send_sms_alert(f"[{market_base}] CRITICAL ORDER FAIL ({side.upper()}): Unexpected setup error: {type(e).__name__}")

    # If any exception occurred before returning the filled_entry order
    return None # Indicate failure of the overall process

def cancel_open_orders(exchange: ccxt.Exchange, symbol: str, reason: str = "Cleanup") -> None:
    """
    Attempts to cancel all open orders for the specified symbol.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The unified market symbol.
        reason: Context for logging why orders are being cancelled.
    """
    logger.info(f"{Fore.CYAN}Order Cleanup Ritual: Initiating for {symbol} (Reason: {reason})...{Style.RESET_ALL}")
    try:
        # Check capability
        if not exchange.has.get('fetchOpenOrders'):
            logger.warning(f"{Fore.YELLOW}Order Cleanup: fetchOpenOrders spell not available for {exchange.id}. Cannot perform cleanup.{Style.RESET_ALL}")
            return
        if not exchange.has.get('cancelOrder'):
            logger.warning(f"{Fore.YELLOW}Order Cleanup: cancelOrder spell not available for {exchange.id}. Cannot perform cleanup.{Style.RESET_ALL}")
            return

        # Bybit V5 requires category for fetching orders
        market = exchange.market(symbol)
        category = 'linear' if market.get('linear') else ('inverse' if market.get('inverse') else None)
        if not category: raise ValueError(f"Cannot determine category for {symbol}")
        params = {'category': category}

        # Summon list of open orders for the specific symbol
        logger.debug(f"Fetching open orders for {symbol} with params: {params}")
        open_orders = exchange.fetch_open_orders(symbol, params=params)

        if not open_orders:
            logger.info(f"{Fore.CYAN}Order Cleanup: No open orders found for {symbol}.{Style.RESET_ALL}")
            return

        logger.warning(f"{Fore.YELLOW}Order Cleanup: Found {len(open_orders)} open orders for {symbol}. Attempting cancellation...{Style.RESET_ALL}")
        cancelled_count, failed_count = 0, 0

        for order in open_orders:
            order_id = order.get('id')
            order_type = order.get('type', 'N/A')
            order_side = order.get('side', 'N/A')
            order_status = order.get('status', 'N/A') # Check status before trying to cancel
            order_id_short = format_order_id(order_id)
            order_info = f"{order_id_short} ({order_type} {order_side}, Status: {order_status})"

            if order_id and order_status == 'open': # Only attempt to cancel open orders
                try:
                    logger.debug(f"Cancelling order {order_info}...")
                    # Bybit V5 cancelOrder also requires category
                    cancel_params = {'category': category}
                    exchange.cancel_order(order_id, symbol, params=cancel_params)
                    logger.info(f"{Fore.CYAN}Order Cleanup: Cancellation request sent for {order_info}{Style.RESET_ALL}")
                    cancelled_count += 1
                    # Add a small delay between cancels to potentially avoid rate limits
                    time.sleep(0.15)
                except ccxt.OrderNotFound:
                    # Order might have been filled or cancelled just before this attempt
                    logger.warning(f"{Fore.YELLOW}Order Cleanup: Order {order_info} not found during cancellation (already closed/cancelled?).{Style.RESET_ALL}")
                    # Consider it 'handled' if not found
                except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                    logger.error(f"{Fore.RED}Order Cleanup: FAILED to cancel {order_info}: {type(e).__name__} - {e}{Style.RESET_ALL}")
                    failed_count += 1
                except Exception as e:
                     logger.error(f"{Fore.RED}Order Cleanup: Unexpected error cancelling {order_info}: {e}{Style.RESET_ALL}")
                     logger.debug(f"Traceback (Cancel Order):\n{traceback.format_exc()}")
                     failed_count += 1
            elif not order_id:
                logger.error(f"{Fore.RED}Order Cleanup: Found order with no ID: {order}. Cannot cancel.{Style.RESET_ALL}")
                failed_count += 1
            else:
                 # Order exists but is not 'open' (e.g., 'closed', 'canceled') - no action needed
                 logger.debug(f"Order Cleanup: Skipping order {order_info} as status is not 'open'.")


        log_level = logging.INFO if failed_count == 0 else logging.WARNING
        log_color = Fore.CYAN if failed_count == 0 else Fore.YELLOW
        logger.log(log_level, f"{log_color}Order Cleanup Ritual Finished for {symbol}. Requests Sent: {cancelled_count}, Failures: {failed_count}.{Style.RESET_ALL}")

        if failed_count > 0:
            send_sms_alert(f"[{symbol.split('/')[0]}] WARNING: Failed to cancel {failed_count} orders during {reason}.")

    except ValueError as e: # Catch category error
        logger.error(f"{Fore.RED}Order Cleanup: Error determining category for {symbol}: {e}{Style.RESET_ALL}")
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger.error(f"{Fore.RED}Order Cleanup: Failed fetching open orders for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"{Fore.RED}Order Cleanup: Unexpected error during cleanup for {symbol}: {e}{Style.RESET_ALL}")
        logger.debug(f"Traceback (Cancel Open Orders):\n{traceback.format_exc()}")

# --- Strategy Signal Generation - Interpreting the Omens ---

def generate_signals(df: pd.DataFrame, strategy_name: str) -> Dict[str, Any]:
    """
    Generates entry and exit signals based on the selected strategy's interpretation
    of the calculated indicators in the DataFrame.

    Args:
        df: DataFrame containing OHLCV data and calculated indicator columns.
        strategy_name: The name of the strategy to use (from CONFIG).

    Returns:
        A dictionary containing boolean signals and exit reason:
        {'enter_long': bool, 'enter_short': bool, 'exit_long': bool, 'exit_short': bool, 'exit_reason': str}
    """
    signals: Dict[str, Any] = {'enter_long': False, 'enter_short': False, 'exit_long': False, 'exit_short': False, 'exit_reason': "Strategy Exit"}
    required_cols_map = {
        "DUAL_SUPERTREND": ['st_long', 'st_short', 'confirm_trend'],
        "STOCHRSI_MOMENTUM": ['stochrsi_k', 'stochrsi_d', 'momentum'],
        "EHLERS_FISHER": ['ehlers_fisher', 'ehlers_signal'],
        "EHLERS_MA_CROSS": ['fast_ema', 'slow_ema']
    }

    # Basic validation
    if df is None or df.empty or len(df) < 2:
        logger.debug("Signal Gen: Insufficient data length (< 2). No signals generated.")
        return signals
    if strategy_name not in required_cols_map:
         logger.error(f"Signal Gen: Unknown strategy '{strategy_name}'. No signals generated.")
         return signals

    # Check if required columns for the *selected* strategy exist and have data
    required_cols = required_cols_map[strategy_name]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"{Fore.YELLOW}Signal Gen ({strategy_name}): Missing required columns: {missing_cols}. No signals generated.{Style.RESET_ALL}")
        return signals

    # Check for NaNs in the required columns on the last two rows needed for comparison
    if df.iloc[-2:][required_cols].isnull().values.any():
         logger.debug(f"Signal Gen ({strategy_name}): NaN values found in required columns in last 2 rows. No signals generated.")
         # Log specific NaNs
         nan_check = df.iloc[-2:][required_cols].isnull()
         logger.debug(f"NaN Check:\n{nan_check}")
         return signals

    # Get data for the last two candles (latest closed = -1, previous = -2)
    last = df.iloc[-1]
    prev = df.iloc[-2]

    try:
        # --- Dual Supertrend Logic ---
        if strategy_name == "DUAL_SUPERTREND":
            # Enter Long: Primary ST flips long (last['st_long'] is True) AND Confirmation ST is already in uptrend (last['confirm_trend'] is True)
            if last['st_long'] and last['confirm_trend']:
                signals['enter_long'] = True
                logger.debug(f"Signal Gen (DualST): Long Entry - Primary ST Long Flip & Confirm ST Up.")
            # Enter Short: Primary ST flips short (last['st_short'] is True) AND Confirmation ST is already in downtrend (last['confirm_trend'] is False)
            elif last['st_short'] and not last['confirm_trend']:
                signals['enter_short'] = True
                logger.debug(f"Signal Gen (DualST): Short Entry - Primary ST Short Flip & Confirm ST Down.")

            # Exit Long: Primary ST flips short
            if last['st_short']:
                signals['exit_long'] = True
                signals['exit_reason'] = "Primary ST Short Flip"
                logger.debug(f"Signal Gen (DualST): Long Exit - Primary ST Short Flip.")
            # Exit Short: Primary ST flips long
            if last['st_long']:
                signals['exit_short'] = True
                signals['exit_reason'] = "Primary ST Long Flip"
                logger.debug(f"Signal Gen (DualST): Short Exit - Primary ST Long Flip.")

        # --- StochRSI + Momentum Logic ---
        elif strategy_name == "STOCHRSI_MOMENTUM":
            k_now, d_now, mom_now = last['stochrsi_k'], last['stochrsi_d'], last['momentum']
            k_prev, d_prev = prev['stochrsi_k'], prev['stochrsi_d']

            # Enter Long: K crosses above D from below, K is currently oversold, AND Momentum is positive
            if k_prev <= d_prev and k_now > d_now and k_now < CONFIG.stochrsi_oversold and mom_now > CONFIG.POSITION_QTY_EPSILON:
                signals['enter_long'] = True
                logger.debug(f"Signal Gen (StochRSI/Mom): Long Entry - K/D Cross Up ({k_prev:.1f}<={d_prev:.1f}, {k_now:.1f}>{d_now:.1f}), K Oversold ({k_now:.1f}<{CONFIG.stochrsi_oversold}), Mom Positive ({mom_now:.2f}>0)")
            # Enter Short: K crosses below D from above, K is currently overbought, AND Momentum is negative
            elif k_prev >= d_prev and k_now < d_now and k_now > CONFIG.stochrsi_overbought and mom_now < -CONFIG.POSITION_QTY_EPSILON:
                signals['enter_short'] = True
                logger.debug(f"Signal Gen (StochRSI/Mom): Short Entry - K/D Cross Down ({k_prev:.1f}>={d_prev:.1f}, {k_now:.1f}<{d_now:.1f}), K Overbought ({k_now:.1f}>{CONFIG.stochrsi_overbought}), Mom Negative ({mom_now:.2f}<0)")

            # Exit Long: K crosses below D (regardless of OB/OS or Momentum)
            if k_prev >= d_prev and k_now < d_now:
                signals['exit_long'] = True
                signals['exit_reason'] = "StochRSI K below D"
                logger.debug(f"Signal Gen (StochRSI/Mom): Long Exit - K/D Cross Down ({k_prev:.1f}>={d_prev:.1f}, {k_now:.1f}<{d_now:.1f})")
            # Exit Short: K crosses above D (regardless of OB/OS or Momentum)
            if k_prev <= d_prev and k_now > d_now:
                signals['exit_short'] = True
                signals['exit_reason'] = "StochRSI K above D"
                logger.debug(f"Signal Gen (StochRSI/Mom): Short Exit - K/D Cross Up ({k_prev:.1f}<={d_prev:.1f}, {k_now:.1f}>{d_now:.1f})")

        # --- Ehlers Fisher Logic ---
        elif strategy_name == "EHLERS_FISHER":
            fish_now, sig_now = last['ehlers_fisher'], last['ehlers_signal']
            fish_prev, sig_prev = prev['ehlers_fisher'], prev['ehlers_signal']

            # Enter Long: Fisher line crosses above Signal line
            if fish_prev <= sig_prev and fish_now > sig_now:
                signals['enter_long'] = True
                logger.debug(f"Signal Gen (EhlersFisher): Long Entry - Fisher/Signal Cross Up ({fish_prev:.3f}<={sig_prev:.3f}, {fish_now:.3f}>{sig_now:.3f})")
            # Enter Short: Fisher line crosses below Signal line
            elif fish_prev >= sig_prev and fish_now < sig_now:
                signals['enter_short'] = True
                logger.debug(f"Signal Gen (EhlersFisher): Short Entry - Fisher/Signal Cross Down ({fish_prev:.3f}>={sig_prev:.3f}, {fish_now:.3f}<{sig_now:.3f})")

            # Exit Long: Fisher crosses below Signal line (Opposite of entry)
            if fish_prev >= sig_prev and fish_now < sig_now:
                signals['exit_long'] = True
                signals['exit_reason'] = "Ehlers Fisher Short Cross"
                logger.debug(f"Signal Gen (EhlersFisher): Long Exit - Fisher/Signal Cross Down ({fish_prev:.3f}>={sig_prev:.3f}, {fish_now:.3f}<{sig_now:.3f})")
            # Exit Short: Fisher crosses above Signal line (Opposite of entry)
            if fish_prev <= sig_prev and fish_now > sig_now:
                signals['exit_short'] = True
                signals['exit_reason'] = "Ehlers Fisher Long Cross"
                logger.debug(f"Signal Gen (EhlersFisher): Short Exit - Fisher/Signal Cross Up ({fish_prev:.3f}<={sig_prev:.3f}, {fish_now:.3f}>{sig_now:.3f})")

        # --- Ehlers MA Cross Logic (Using EMA Placeholder) ---
        elif strategy_name == "EHLERS_MA_CROSS":
            fast_ma_now, slow_ma_now = last['fast_ema'], last['slow_ema']
            fast_ma_prev, slow_ma_prev = prev['fast_ema'], prev['slow_ema']

            # Enter Long: Fast MA crosses above Slow MA
            if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now:
                signals['enter_long'] = True
                logger.debug(f"Signal Gen (EhlersMA/EMA): Long Entry - Fast/Slow Cross Up ({fast_ma_prev:.3f}<={slow_ma_prev:.3f}, {fast_ma_now:.3f}>{slow_ma_now:.3f})")
            # Enter Short: Fast MA crosses below Slow MA
            elif fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now:
                signals['enter_short'] = True
                logger.debug(f"Signal Gen (EhlersMA/EMA): Short Entry - Fast/Slow Cross Down ({fast_ma_prev:.3f}>={slow_ma_prev:.3f}, {fast_ma_now:.3f}<{slow_ma_now:.3f})")

            # Exit Long: Fast MA crosses below Slow MA (Opposite of entry)
            if fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now:
                signals['exit_long'] = True
                signals['exit_reason'] = "Ehlers MA Short Cross (EMA)"
                logger.debug(f"Signal Gen (EhlersMA/EMA): Long Exit - Fast/Slow Cross Down ({fast_ma_prev:.3f}>={slow_ma_prev:.3f}, {fast_ma_now:.3f}<{slow_ma_now:.3f})")
            # Exit Short: Fast MA crosses above Slow MA (Opposite of entry)
            if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now:
                signals['exit_short'] = True
                signals['exit_reason'] = "Ehlers MA Long Cross (EMA)"
                logger.debug(f"Signal Gen (EhlersMA/EMA): Short Exit - Fast/Slow Cross Up ({fast_ma_prev:.3f}<={slow_ma_prev:.3f}, {fast_ma_now:.3f}>{slow_ma_now:.3f})")

    except KeyError as e:
        # This should be caught by the initial column check, but as a safeguard
        logger.error(f"{Fore.RED}Signal Generation Error ({strategy_name}): Missing expected column during logic execution: {e}. Signals reset.{Style.RESET_ALL}")
        signals = {'enter_long': False, 'enter_short': False, 'exit_long': False, 'exit_short': False, 'exit_reason': "Error"}
    except Exception as e:
        logger.error(f"{Fore.RED}Signal Generation Error ({strategy_name}): Unexpected disturbance: {e}{Style.RESET_ALL}")
        logger.debug(f"Traceback (Signal Generation):\n{traceback.format_exc()}")
        signals = {'enter_long': False, 'enter_short': False, 'exit_long': False, 'exit_short': False, 'exit_reason': "Error"}

    # Final check for conflicting signals (should not happen with current logic, but good practice)
    if signals['enter_long'] and signals['enter_short']:
        logger.warning(f"{Fore.YELLOW}Signal Gen ({strategy_name}): Conflicting Enter Long and Enter Short signals generated simultaneously. Prioritizing neither.{Style.RESET_ALL}")
        signals['enter_long'], signals['enter_short'] = False, False
    # Similar check for exits might be needed if logic changes

    # Log generated signals if any are True
    if any(signals.values()): # Checks if any boolean signal is True
         active_signals = {k:v for k,v in signals.items() if isinstance(v, bool) and v}
         if active_signals: # Check if there are any active boolean signals
              logger.info(f"Strategy Signals ({strategy_name}) Generated: {active_signals}")
         elif signals['exit_reason'] != "Strategy Exit": # Log if only exit reason changed
              logger.info(f"Strategy Signals ({strategy_name}): Exit Reason Set - '{signals['exit_reason']}'")

    return signals

# --- Trading Logic - The Core Spell Weaving ---

def trade_logic(exchange: ccxt.Exchange, symbol: str, df: pd.DataFrame) -> None:
    """
    Executes the main trading logic for one cycle:
    1. Calculates all required indicators.
    2. Checks current position and market conditions (ATR, Volume, Order Book).
    3. Generates signals based on the selected strategy.
    4. Executes exit orders if required.
    5. Executes entry orders (including SL/TSL) if conditions are met and flat.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The unified market symbol.
        df: DataFrame containing OHLCV data (should not be empty).
    """
    # Ensure DataFrame is not empty before proceeding
    if df is None or df.empty:
         logger.error("Trade Logic Error: Received empty DataFrame. Skipping cycle.")
         return

    try:
        cycle_time_str = df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z') if not df.empty else "N/A"
        logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== New Weaving Cycle ({CONFIG.strategy_name}): {symbol} | Candle: {cycle_time_str} =========={Style.RESET_ALL}")

        # === 1. Calculate ALL Indicators - Scry the full spectrum ===
        # It's simpler to calculate all potential indicators required by any strategy
        # and let the signal generation function pick the ones relevant to the selected strategy.
        logger.debug("Calculating all potential indicators...")
        df = calculate_supertrend(df, CONFIG.st_atr_length, CONFIG.st_multiplier)
        df = calculate_supertrend(df, CONFIG.confirm_st_atr_length, CONFIG.confirm_st_multiplier, prefix="confirm_")
        df = calculate_stochrsi_momentum(df, CONFIG.stochrsi_rsi_length, CONFIG.stochrsi_stoch_length, CONFIG.stochrsi_k_period, CONFIG.stochrsi_d_period, CONFIG.momentum_length)
        df = calculate_ehlers_fisher(df, CONFIG.ehlers_fisher_length, CONFIG.ehlers_fisher_signal_length)
        df = calculate_ehlers_ma(df, CONFIG.ehlers_fast_period, CONFIG.ehlers_slow_period) # Placeholder EMA
        # Calculate Vol/ATR separately as it returns a dict
        vol_atr_data = analyze_volume_atr(df, CONFIG.atr_calculation_period, CONFIG.volume_ma_period)
        current_atr = vol_atr_data.get("atr")

        # === 2. Validate Base Requirements - Ensure stable ground ===
        logger.debug("Validating core requirements...")
        last_candle = df.iloc[-1]
        current_price = safe_decimal_conversion(last_candle.get('close')) # Get latest close price
        if current_price <= Decimal("0"): # Check if close price is valid
            logger.warning(f"{Fore.YELLOW}Trade Logic: Last candle close price is invalid ({current_price}). Skipping cycle.{Style.RESET_ALL}")
            return

        # Can we place a new order? Requires valid ATR for SL calculation.
        can_place_new_order = current_atr is not None and current_atr > Decimal("0")
        if not can_place_new_order:
            logger.warning(f"{Fore.YELLOW}Trade Logic: Invalid ATR ({current_atr}). Cannot calculate SL distances or place new entry orders this cycle.{Style.RESET_ALL}")
            # Note: Existing position might still be managed by exchange-native SL/TSL or exit signals not requiring ATR.

        # === 3. Get Position & Analyze Order Book (if configured) ===
        logger.debug("Getting current position and analyzing order book (if configured)...")
        position = get_current_position(exchange, symbol) # Check current market presence
        position_side = position['side']
        position_qty = position['qty']
        position_entry = position['entry_price']

        # Fetch OB data if configured for every cycle, OR initialize to None
        ob_data: Optional[Dict[str, Optional[Decimal]]] = None
        if CONFIG.fetch_order_book_per_cycle:
            ob_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit)

        # === 4. Log Current State - The Oracle Reports ===
        logger.debug("Logging current market state...")
        vol_ratio = vol_atr_data.get("volume_ratio")
        # Check for volume spike based on threshold
        vol_spike = (vol_ratio is not None and vol_atr_data.get("last_volume") is not None and
                     vol_ratio > CONFIG.volume_spike_threshold and
                     vol_atr_data["last_volume"] > 0) # Ensure vol > 0 for a spike

        # Extract OB data if fetched
        bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None
        spread = ob_data.get("spread") if ob_data else None

        # Log core state
        atr_str = f"{current_atr:.5f}" if current_atr else "N/A"
        logger.info(f"State | Price: {Fore.CYAN}{current_price:.4f}{Style.RESET_ALL}, ATR({CONFIG.atr_calculation_period}): {Fore.MAGENTA}{atr_str}{Style.RESET_ALL} (Orderable: {can_place_new_order})")
        # Log confirmation states
        vol_ratio_str = f"{vol_ratio:.2f}x" if vol_ratio else "N/A"
        vol_spike_str = f"{Fore.GREEN}YES{Style.RESET_ALL}" if vol_spike else f"{Fore.RED}NO{Style.RESET_ALL}"
        logger.info(f"State | Volume: Ratio={Fore.YELLOW}{vol_ratio_str}{Style.RESET_ALL}, Spike={vol_spike_str} (Required for Entry: {CONFIG.require_volume_spike_for_entry})")
        ob_ratio_str = f"{bid_ask_ratio:.3f}" if bid_ask_ratio else "N/A"
        ob_spread_str = f"{spread:.4f}" if spread else "N/A"
        ob_fetched_str = "Yes" if ob_data is not None else ("No" if not CONFIG.fetch_order_book_per_cycle else "Failed")
        logger.info(f"State | OrderBook: Ratio={Fore.YELLOW}{ob_ratio_str}{Style.RESET_ALL}, Spread={ob_spread_str} (Fetched This Cycle: {ob_fetched_str})")
        # Log position state
        pos_color = Fore.GREEN if position_side == CONFIG.POS_LONG else (Fore.RED if position_side == CONFIG.POS_SHORT else Fore.BLUE)
        logger.info(f"State | Position: Side={pos_color}{position_side}{Style.RESET_ALL}, Qty={position_qty:.8f}, Entry={position_entry:.4f}")

        # === 5. Generate Strategy Signals - Interpret the Omens ===
        logger.debug(f"Generating signals for strategy: {CONFIG.strategy_name}...")
        strategy_signals = generate_signals(df, CONFIG.strategy_name)
        # Logging of signals happens within generate_signals

        # === 6. Execute Exit Actions - If the Omens Demand Retreat ===
        action_taken_this_cycle: bool = False # Track if an entry/exit order was placed
        if position_side != CONFIG.POS_NONE:
            should_exit_long = position_side == CONFIG.POS_LONG and strategy_signals['exit_long']
            should_exit_short = position_side == CONFIG.POS_SHORT and strategy_signals['exit_short']

            if should_exit_long or should_exit_short:
                exit_reason = strategy_signals.get('exit_reason', "Strategy Exit Signal")
                exit_side_color = Back.YELLOW
                logger.warning(f"{exit_side_color}{Fore.BLACK}{Style.BRIGHT}*** STRATEGY EXIT SIGNAL: Closing {position_side} due to '{exit_reason}' ***{Style.RESET_ALL}")

                # 1. Cancel existing open orders (SL/TSL) associated with this position before closing market.
                # This is crucial to prevent the SL/TSL from potentially interfering or executing after the close.
                cancel_open_orders(exchange, symbol, f"Pre-Exit Cleanup ({exit_reason})")
                time.sleep(0.5) # Small pause after cancel request before placing market close

                # 2. Attempt to close the position with a market order
                close_result = close_position(exchange, symbol, position, reason=exit_reason)
                if close_result:
                    action_taken_this_cycle = True
                    logger.info(f"Exit successful. Pausing for {CONFIG.POST_CLOSE_DELAY_SECONDS}s before checking for new entries...")
                    time.sleep(CONFIG.POST_CLOSE_DELAY_SECONDS)
                else:
                     # Close failed, logged within close_position. Position might still be open.
                     logger.error(f"{Fore.RED}Strategy exit FAILED for {position_side} position. Manual check may be required.{Style.RESET_ALL}")
                     # Don't proceed to check for entries if exit failed.

                # Regardless of close success/failure, return here to prevent immediate re-entry in the same cycle.
                # The next cycle will re-evaluate the (potentially still open or now closed) position.
                logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== Cycle Weaving End (After Exit Attempt): {symbol} =========={Style.RESET_ALL}\n")
                return

        # === 7. Check & Execute Entry Actions (Only if Flat & Can Place Order) ===
        # If we reach here, we are either flat or holding a position without an exit signal.
        if position_side != CONFIG.POS_NONE:
             logger.info(f"Holding {pos_color}{position_side}{Style.RESET_ALL} position. No strategy exit signal this cycle. Awaiting SL/TSL or next signal.")
             logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== Cycle Weaving End (Holding Position): {symbol} =========={Style.RESET_ALL}\n")
             return # Do nothing more this cycle if holding position

        # --- We are FLAT. Check for Entry ---
        logger.info("Position is Flat. Checking for entry signals...")
        if not can_place_new_order:
             logger.warning(f"{Fore.YELLOW}Holding Cash. Cannot enter: Invalid ATR ({current_atr}) prevents required SL calculation.{Style.RESET_ALL}")
             logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== Cycle Weaving End (Cannot Enter): {symbol} =========={Style.RESET_ALL}\n")
             return # Cannot enter without valid ATR for SL

        # Check if there's a potential strategy entry signal
        potential_entry_side: Optional[str] = None
        if strategy_signals['enter_long']: potential_entry_side = CONFIG.SIDE_BUY
        elif strategy_signals['enter_short']: potential_entry_side = CONFIG.SIDE_SELL

        if not potential_entry_side:
            logger.info("No strategy entry signal generated this cycle. Holding cash.")
            logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== Cycle Weaving End (No Entry Signal): {symbol} =========={Style.RESET_ALL}\n")
            return # No entry signal

        # --- Potential Entry Signal Exists: Evaluate Confirmation Filters ---
        logger.info(f"Potential Entry Signal: {potential_entry_side.upper()}. Evaluating confirmation filters...")

        # Fetch OB data now if not fetched per cycle AND confirmation is needed
        if ob_data is None: # Check if we need to fetch it now
            # Determine if OB check is relevant for *any* entry condition based on config/strategy needs
            # For now, assume we always fetch if not fetched per cycle and a signal exists
            logger.debug("OB not fetched yet, fetching now for entry confirmation...")
            ob_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit)
            bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None # Update ratio
            ob_log_ratio = f"{bid_ask_ratio:.3f}" if bid_ask_ratio else "N/A"
            logger.debug(f"OB Data Fetched On-Demand: Ratio={ob_log_ratio}")

        # Evaluate Volume Confirmation
        volume_confirmation_met = not CONFIG.require_volume_spike_for_entry or vol_spike
        vol_log = f"Volume OK? (Pass:{volume_confirmation_met}, Spike={vol_spike}, Req={CONFIG.require_volume_spike_for_entry})"

        # Evaluate Order Book Confirmation
        ob_confirmation_met = False
        ob_available = ob_data is not None and bid_ask_ratio is not None
        if not ob_available:
             ob_log = f"OB OK? (N/A - Data Unavailable)"
             # If OB data is unavailable, treat confirmation as failed if it were required?
             # Or allow entry without OB? Current logic: assume OK if unavailable, adjust if needed.
             # More strict: ob_confirmation_met = not config_requires_ob_check (need new config?)
             ob_confirmation_met = True # Allow entry if OB unavailable for now
        elif potential_entry_side == CONFIG.SIDE_BUY:
            ob_confirmation_met = bid_ask_ratio >= CONFIG.order_book_ratio_threshold_long
            ob_log_ratio = f"{bid_ask_ratio:.3f}"
            ob_log = f"OB OK? (L Pass:{ob_confirmation_met}, Ratio={ob_log_ratio} >= {CONFIG.order_book_ratio_threshold_long})"
        elif potential_entry_side == CONFIG.SIDE_SELL:
            ob_confirmation_met = bid_ask_ratio <= CONFIG.order_book_ratio_threshold_short
            ob_log_ratio = f"{bid_ask_ratio:.3f}"
            ob_log = f"OB OK? (S Pass:{ob_confirmation_met}, Ratio={ob_log_ratio} <= {CONFIG.order_book_ratio_threshold_short})"

        # --- Combine Strategy Signal with Confirmations ---
        final_entry_decision = potential_entry_side is not None and volume_confirmation_met and ob_confirmation_met

        logger.debug(f"Final Entry Check ({potential_entry_side.upper()} Signal): {vol_log}, {ob_log}")
        if final_entry_decision:
             entry_action_color = Back.GREEN if potential_entry_side == CONFIG.SIDE_BUY else Back.RED
             text_color = Fore.BLACK if potential_entry_side == CONFIG.SIDE_BUY else Fore.WHITE
             logger.success(f"{entry_action_color}{text_color}{Style.BRIGHT}*** TRADE ENTRY CONFIRMED: {potential_entry_side.upper()} ({CONFIG.strategy_name}) for {symbol} ***{Style.RESET_ALL}")

             # Cancel any stray orders *before* entering (safety measure)
             cancel_open_orders(exchange, symbol, f"Pre-{potential_entry_side.upper()} Entry")
             time.sleep(0.5) # Small pause after cancel

             # Place the risked market order with SL and TSL
             place_result = place_risked_market_order(
                 exchange=exchange,
                 symbol=symbol,
                 side=potential_entry_side,
                 risk_percentage=CONFIG.risk_per_trade_percentage,
                 current_atr=current_atr, # Already validated as non-None, positive
                 sl_atr_multiplier=CONFIG.atr_stop_loss_multiplier,
                 leverage=CONFIG.leverage,
                 max_order_cap_usdt=CONFIG.max_order_usdt_amount,
                 margin_check_buffer=CONFIG.required_margin_buffer,
                 tsl_percent=CONFIG.trailing_stop_percentage,
                 tsl_activation_offset_percent=CONFIG.trailing_stop_activation_offset_percent
             )
             if place_result:
                 action_taken_this_cycle = True
                 # No post-entry delay needed here, next cycle starts after sleep
             else:
                 # Entry placement failed, logged within place_risked_market_order
                 logger.error(f"{Fore.RED}Confirmed entry signal for {potential_entry_side.upper()} but order placement failed.{Style.RESET_ALL}")
                 # Action already logged, no need to repeat SMS alert maybe

        elif potential_entry_side: # Signal was present but filters failed
            logger.info(f"Strategy signal for {potential_entry_side.upper()} present but confirmation filters not met ({vol_log}, {ob_log}). Holding cash.")
        # else: # No potential signal case already handled above

    except Exception as e:
        # Catch-all for unexpected errors within the main trade logic function for this cycle
        logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}CRITICAL UNEXPECTED ERROR in trade_logic cycle: {e}{Style.RESET_ALL}")
        logger.debug(f"Traceback (Trade Logic Cycle):\n{traceback.format_exc()}")
        send_sms_alert(f"[{symbol.split('/')[0]}] CRITICAL ERROR in trade_logic: {type(e).__name__}. Check logs!")
        # Allow the main loop to continue to the next cycle, unless it's handled there

    finally:
        # Mark the end of the cycle clearly
        logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== Cycle Weaving End: {symbol} =========={Style.RESET_ALL}\n")

# --- Graceful Shutdown - Withdrawing the Arcane Energies ---

def graceful_shutdown(exchange: Optional[ccxt.Exchange], symbol: Optional[str]) -> None:
    """
    Attempts to close any open position and cancel all open orders for the symbol
    before exiting the script. Ensures a cleaner withdrawal from the market.
    """
    logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}Shutdown requested. Withdrawing arcane energies gracefully...{Style.RESET_ALL}")
    market_base = symbol.split('/')[0] if symbol else "Bot"
    send_sms_alert(f"[{market_base}] Shutdown initiated. Attempting cleanup...")

    if not exchange or not symbol:
        logger.warning(f"{Fore.YELLOW}Shutdown: Exchange portal or symbol not defined. Cannot perform cleanup.{Style.RESET_ALL}")
        return

    try:
        # 1. Cancel All Open Orders - Dispel residual intents first
        logger.info("Shutdown Step 1: Cancelling all open orders...")
        # Use the dedicated function which handles category etc.
        cancel_open_orders(exchange, symbol, reason="Graceful Shutdown")
        # Wait briefly for cancellations to process before checking position
        time.sleep(1.5)

        # 2. Check and Close Existing Position - Banish final presence
        logger.info("Shutdown Step 2: Checking for active position...")
        position = get_current_position(exchange, symbol)

        if position['side'] != CONFIG.POS_NONE:
            pos_color = Fore.GREEN if position['side'] == CONFIG.POS_LONG else Fore.RED
            logger.warning(f"{Fore.YELLOW}Shutdown: Active {pos_color}{position['side']}{Style.RESET_ALL} position found (Qty: {position['qty']:.8f}). Attempting banishment...{Style.RESET_ALL}")

            # Attempt to close the position using the dedicated function
            close_result = close_position(exchange, symbol, position, reason="Shutdown")

            if close_result:
                # Close order was placed, wait longer to allow fill confirmation attempt
                wait_time = CONFIG.POST_CLOSE_DELAY_SECONDS * 2
                logger.info(f"{Fore.CYAN}Shutdown: Close order placed. Waiting {wait_time}s for market confirmation...{Style.RESET_ALL}")
                time.sleep(wait_time)

                # Final check after waiting: Did the position actually close?
                logger.info("Shutdown: Final position check after closure attempt...")
                final_pos = get_current_position(exchange, symbol)
                if final_pos['side'] == CONFIG.POS_NONE:
                    logger.success(f"{Fore.GREEN}{Style.BRIGHT}Shutdown: Position confirmed BANISHED. Clean exit.{Style.RESET_ALL}")
                    send_sms_alert(f"[{market_base}] Position confirmed CLOSED on shutdown.")
                else:
                    # This is problematic - position might still be open despite close attempt
                    final_pos_color = Fore.GREEN if final_pos['side'] == CONFIG.POS_LONG else Fore.RED
                    logger.critical(f"{Back.RED}{Fore.WHITE}Shutdown CRITICAL: FAILED TO CONFIRM position closure after waiting! "
                                   f"Final state: {final_pos_color}{final_pos['side']}{Style.RESET_ALL} Qty={final_pos['qty']:.8f}. "
                                   f"MANUAL CHECK REQUIRED!{Style.RESET_ALL}")
                    send_sms_alert(f"[{market_base}] CRITICAL SHUTDOWN ERROR: Failed confirm closure! Final: {final_pos['side']} Qty={final_pos['qty']:.8f}. MANUAL CHECK!")
            else:
                # close_position function failed to even place the order
                logger.critical(f"{Back.RED}{Fore.WHITE}Shutdown CRITICAL: Failed to PLACE close order for active position. MANUAL INTERVENTION REQUIRED!{Style.RESET_ALL}")
                send_sms_alert(f"[{market_base}] CRITICAL SHUTDOWN ERROR: Failed PLACE close order. MANUAL CHECK!")
        else:
            # No position was found initially after cancelling orders
            logger.info(f"{Fore.GREEN}Shutdown: No active position found after order cancellation. Clean exit pathway.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] No active position found on shutdown.")

    except Exception as e:
        # Catch errors during the shutdown sequence itself
        logger.error(f"{Fore.RED}Shutdown Sequence Error: An error occurred during cleanup: {e}{Style.RESET_ALL}")
        logger.debug(f"Traceback (Graceful Shutdown):\n{traceback.format_exc()}")
        send_sms_alert(f"[{market_base}] Error during shutdown cleanup: {type(e).__name__}. Check manually!")

    logger.info(f"{Fore.YELLOW}{Style.BRIGHT}--- Pyrmethus Scalping Spell Shutdown Complete ---{Style.RESET_ALL}")


# --- Main Execution - Igniting the Spell ---

def main() -> None:
    """
    Main function to initialize the bot, set up the exchange connection and parameters,
    and run the main trading loop. Handles overall error management and graceful shutdown.
    """
    start_time_str = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Bybit Scalping Spell v{__import__('__main__').__doc__.splitlines()[1].split(' ')[1]} Initializing ({start_time_str}) ---{Style.RESET_ALL}") # Read version dynamically
    logger.info(f"{Fore.CYAN}--- Strategy Enchantment Selected: {CONFIG.strategy_name} ---{Style.RESET_ALL}")
    logger.info(f"{Fore.GREEN}--- Protective Wards: Initial ATR-Stop + Exchange Trailing Stop ---{Style.RESET_ALL}")
    logger.warning(f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}--- !!! LIVE FUTURES SCALPING MODE - EXTREME RISK - HANDLE WITH CARE !!! ---{Style.RESET_ALL}")

    exchange: Optional[ccxt.Exchange] = None
    symbol: Optional[str] = None # The specific unified market symbol (e.g., BTC/USDT:USDT)
    run_bot: bool = True         # Controls the main loop execution
    cycle_count: int = 0         # Tracks the number of iterations

    try:
        # === Initialize Exchange Portal ===
        exchange = initialize_exchange()
        if not exchange:
            logger.critical("Failed to open exchange portal. Spell cannot proceed. Terminating.")
            # SMS alert sent within initialize_exchange on critical failure
            return # Exit script

        # === Setup Symbol and Leverage - Focusing the Spell ===
        try:
            # Allow user input for symbol, falling back to config default
            default_sym = CONFIG.symbol
            sym_input = input(f"{Fore.YELLOW}Enter target symbol {Style.DIM}(e.g., BTC/USDT:USDT, Default [{default_sym}]){Style.NORMAL}: {Style.RESET_ALL}").strip()
            symbol_to_use = sym_input if sym_input else default_sym

            # Validate the symbol using exchange.market() and ensure it's a usable contract market
            logger.debug(f"Validating target symbol: {symbol_to_use}")
            market = exchange.market(symbol_to_use) # Throws BadSymbol if invalid
            symbol = market['symbol'] # Use the precise, unified symbol recognized by CCXT

            # Ensure it's a futures/contract market (swap or future) and linear (USDT margined)
            if not market.get('contract'):
                raise ValueError(f"Market '{symbol}' is not a contract/futures market.")
            if not market.get('linear'):
                 raise ValueError(f"Market '{symbol}' is not a linear (USDT-margined) contract. This bot currently supports linear contracts only.")
            logger.info(f"{Fore.GREEN}Focusing spell on Symbol: {symbol} (Type: {market.get('type', 'N/A')}, Linear: {market.get('linear')}){Style.RESET_ALL}")

            # Set the desired leverage
            if not set_leverage(exchange, symbol, CONFIG.leverage):
                # Error logged within set_leverage
                raise RuntimeError(f"Leverage conjuring failed for {symbol} at {CONFIG.leverage}x. Cannot proceed safely.")

        except (ccxt.BadSymbol, KeyError, ValueError, RuntimeError) as e:
            logger.critical(f"{Back.RED}{Fore.WHITE}Symbol/Leverage setup failed: {e}{Style.RESET_ALL}")
            send_sms_alert(f"[Pyrmethus] CRITICAL: Symbol/Leverage setup FAILED ({e}). Exiting.")
            return # Exit script if symbol/leverage setup fails
        except Exception as e:
            logger.critical(f"{Back.RED}{Fore.WHITE}Unexpected error during spell focus (Symbol/Leverage) setup: {e}{Style.RESET_ALL}")
            logger.debug(f"Traceback (Symbol/Leverage Setup):\n{traceback.format_exc()}")
            send_sms_alert("[Pyrmethus] CRITICAL: Unexpected setup error. Exiting.")
            return # Exit script

        # === Log Configuration Summary - Reciting the Parameters ===
        logger.info(f"{Fore.MAGENTA}{'=' * 15} Spell Configuration Summary {'=' * 15}{Style.RESET_ALL}")
        logger.info(f"{Fore.WHITE}Target Symbol       : {symbol}")
        logger.info(f"{Fore.WHITE}Time Interval       : {CONFIG.interval}")
        logger.info(f"{Fore.WHITE}Leverage            : {CONFIG.leverage}x")
        logger.info(f"{Fore.CYAN}Strategy Path       : {CONFIG.strategy_name}")
        # Log relevant strategy parameters for clarity
        strategy_params_log = "  Strategy Params     : "
        if CONFIG.strategy_name == "DUAL_SUPERTREND": strategy_params_log += f"ST={CONFIG.st_atr_length}/{CONFIG.st_multiplier}, ConfirmST={CONFIG.confirm_st_atr_length}/{CONFIG.confirm_st_multiplier}"
        elif CONFIG.strategy_name == "STOCHRSI_MOMENTUM": strategy_params_log += f"StochRSI={CONFIG.stochrsi_rsi_length}/{CONFIG.stochrsi_stoch_length}/{CONFIG.stochrsi_k_period}/{CONFIG.stochrsi_d_period} (OB={CONFIG.stochrsi_overbought},OS={CONFIG.stochrsi_oversold}), Mom={CONFIG.momentum_length}"
        elif CONFIG.strategy_name == "EHLERS_FISHER": strategy_params_log += f"Fisher={CONFIG.ehlers_fisher_length}, Signal={CONFIG.ehlers_fisher_signal_length}"
        elif CONFIG.strategy_name == "EHLERS_MA_CROSS": strategy_params_log += f"FastMA(EMA)={CONFIG.ehlers_fast_period}, SlowMA(EMA)={CONFIG.ehlers_slow_period} (EMA Placeholder)"
        logger.info(strategy_params_log)
        logger.info(f"{Fore.GREEN}Risk Per Trade      : {CONFIG.risk_per_trade_percentage:.3%}")
        logger.info(f"{Fore.GREEN}Max Position Value  : {CONFIG.max_order_usdt_amount:.2f} USDT")
        logger.info(f"{Fore.GREEN}Initial SL ATR Mult : {CONFIG.atr_stop_loss_multiplier} (ATR Period: {CONFIG.atr_calculation_period})")
        logger.info(f"{Fore.GREEN}Trailing SL Percent : {CONFIG.trailing_stop_percentage:.2%}")
        logger.info(f"{Fore.GREEN}TSL Activation Offset: {CONFIG.trailing_stop_activation_offset_percent:.2%}")
        logger.info(f"{Fore.YELLOW}Volume Filter       : Required={CONFIG.require_volume_spike_for_entry} (MA={CONFIG.volume_ma_period}, Thr={CONFIG.volume_spike_threshold}x)")
        logger.info(f"{Fore.YELLOW}Order Book Filter   : Fetch Per Cycle={CONFIG.fetch_order_book_per_cycle} (Depth={CONFIG.order_book_depth}, L>={CONFIG.order_book_ratio_threshold_long}, S<={CONFIG.order_book_ratio_threshold_short})")
        logger.info(f"{Fore.WHITE}Cycle Sleep         : {CONFIG.sleep_seconds}s")
        logger.info(f"{Fore.WHITE}Margin Buffer Check : {CONFIG.required_margin_buffer:.1%}")
        logger.info(f"{Fore.MAGENTA}SMS Alerts Enabled  : {CONFIG.enable_sms_alerts} (To: {CONFIG.sms_recipient_number or 'N/A'})")
        logger.info(f"{Fore.CYAN}Logging Level       : {logging.getLevelName(logger.level)}")
        logger.info(f"{Fore.MAGENTA}{'=' * 52}{Style.RESET_ALL}")

        market_base = symbol.split('/')[0] # For SMS brevity
        send_sms_alert(f"[{market_base}] Pyrmethus Configured. Strategy: {CONFIG.strategy_name}. SL: ATR+TSL. Starting main loop.")

        # === Main Trading Loop - The Continuous Weaving ===
        while run_bot:
            cycle_start_time = time.monotonic()
            cycle_count += 1
            logger.debug(f"{Fore.CYAN}--- Cycle {cycle_count} Weaving Start (Time: {time.strftime('%H:%M:%S')}) ---{Style.RESET_ALL}")

            try:
                # Determine required data length based on longest possible indicator lookback + buffer
                # Ensure calculation uses integer periods from config
                data_limit = max(
                    100, # Base minimum
                    CONFIG.st_atr_length * 2, CONFIG.confirm_st_atr_length * 2,
                    # StochRSI needs sum of lengths + d + buffer
                    CONFIG.stochrsi_rsi_length + CONFIG.stochrsi_stoch_length + CONFIG.stochrsi_d_period + 5,
                    CONFIG.momentum_length * 2,
                    CONFIG.ehlers_fisher_length * 2 + CONFIG.ehlers_fisher_signal_length,
                    CONFIG.ehlers_fast_period * 2, CONFIG.ehlers_slow_period * 2,
                    CONFIG.atr_calculation_period * 2, CONFIG.volume_ma_period * 2
                ) + CONFIG.API_FETCH_LIMIT_BUFFER # Add safety buffer
                logger.debug(f"Calculated data fetch limit: {data_limit} candles")

                # --- Gather fresh market data ---
                # Ensure exchange and symbol are valid before fetching
                if not exchange or not symbol:
                     logger.critical("Exchange or Symbol became invalid within loop. Stopping.")
                     run_bot = False
                     continue

                df = get_market_data(exchange, symbol, CONFIG.interval, limit=data_limit)

                # --- Process data and execute trading logic if data is valid ---
                if df is not None and not df.empty:
                    # Pass a copy of the DataFrame to trade_logic to prevent modifications
                    # from affecting subsequent calculations if df is reused (though it's refetched each cycle here)
                    trade_logic(exchange, symbol, df.copy())
                else:
                    logger.warning(f"{Fore.YELLOW}No valid market data received for {symbol} in cycle {cycle_count}. Skipping trade logic.{Style.RESET_ALL}")
                    # Consider adding a longer sleep or specific action if data is consistently missing

            # --- Robust Error Handling within the Loop ---
            except ccxt.RateLimitExceeded as e:
                logger.warning(f"{Back.YELLOW}{Fore.BLACK}RATE LIMIT EXCEEDED: {e}. The exchange spirits demand patience. Sleeping longer...{Style.RESET_ALL}")
                send_sms_alert(f"[{market_base}] WARNING: Rate limit hit! Pausing significantly.")
                time.sleep(CONFIG.sleep_seconds * 10) # Sleep much longer

            except ccxt.NetworkError as e:
                # Includes connection errors, timeouts, etc. Often transient.
                logger.warning(f"{Fore.YELLOW}Network Disturbance: {e}. Retrying next cycle after standard sleep.{Style.RESET_ALL}")
                # Standard sleep will apply at the end of the loop

            except ccxt.ExchangeNotAvailable as e:
                # Exchange might be down for maintenance or experiencing issues.
                logger.error(f"{Back.RED}{Fore.WHITE}Exchange Unavailable: {e}. Portal temporarily closed? Sleeping significantly longer...{Style.RESET_ALL}")
                send_sms_alert(f"[{market_base}] ERROR: Exchange unavailable ({type(e).__name__})! Long pause.")
                time.sleep(CONFIG.sleep_seconds * 15) # Wait a significant time

            except ccxt.AuthenticationError as e:
                # API keys might have been revoked, expired, or IP banned. Critical.
                logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}AUTHENTICATION ERROR: {e}. Spell broken! API keys invalid or permissions changed. Stopping NOW.{Style.RESET_ALL}")
                send_sms_alert(f"[{market_base}] CRITICAL: Authentication Error! Stopping NOW.")
                run_bot = False # Stop the bot immediately

            except ccxt.ExchangeError as e:
                # Catch other specific exchange errors not handled above (e.g., invalid order params, insufficient margin *during* operation)
                logger.error(f"{Fore.RED}Unhandled Exchange Error in main loop: {e}{Style.RESET_ALL}")
                logger.debug(f"Traceback (Main Loop ExchangeError):\n{traceback.format_exc()}")
                send_sms_alert(f"[{market_base}] ERROR: Unhandled Exchange error: {type(e).__name__}. Check logs.")
                # Continue loop after standard sleep, assuming it might be recoverable or position needs managing

            except Exception as e:
                # Catch-all for truly unexpected issues within the main loop's try block
                logger.exception(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}!!! UNEXPECTED CRITICAL CHAOS IN MAIN LOOP: {e} !!! Stopping spell!{Style.RESET_ALL}")
                # logger.exception automatically logs the traceback at ERROR level
                send_sms_alert(f"[{market_base}] CRITICAL UNEXPECTED ERROR: {type(e).__name__}! Stopping NOW.")
                run_bot = False # Stop the bot on unknown critical errors

            # --- Loop Delay - Controlling the Rhythm ---
            if run_bot:
                elapsed = time.monotonic() - cycle_start_time
                sleep_dur = max(0, CONFIG.sleep_seconds - elapsed)
                logger.debug(f"Cycle {cycle_count} completed in {elapsed:.2f}s. Sleeping for {sleep_dur:.2f}s.")
                if sleep_dur > 0:
                    time.sleep(sleep_dur)

    except KeyboardInterrupt:
        logger.warning(f"\n{Fore.YELLOW}{Style.BRIGHT}KeyboardInterrupt received. User requests withdrawal of arcane energies...{Style.RESET_ALL}")
        run_bot = False # Signal the loop to terminate gracefully
    except Exception as e:
         # Catch errors during initial setup *before* the main loop starts
         logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL ERROR during initial setup phase: {e}{Style.RESET_ALL}")
         logger.debug(f"Traceback (Initial Setup Phase):\n{traceback.format_exc()}")
         send_sms_alert(f"[Pyrmethus] CRITICAL: Bot failed during initial setup: {type(e).__name__}. Cannot start.")
         run_bot = False # Ensure shutdown is attempted even if loop never ran

    finally:
        # --- Graceful Shutdown Sequence ---
        # This block executes whether the loop finished normally, was interrupted (KeyboardInterrupt),
        # or hit a critical error that set run_bot=False.
        # Pass the potentially valid exchange and symbol objects to the shutdown function.
        graceful_shutdown(exchange, symbol)
        final_market_base = symbol.split('/')[0] if symbol else "Bot"
        # Optional: Alert on final termination (might not send if Termux process killed abruptly)
        # send_sms_alert(f"[{final_market_base}] Pyrmethus process terminated.")
        logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Scalping Spell Deactivated ---{Style.RESET_ALL}")


if __name__ == "__main__":
    # Ensure the spell is cast only when invoked directly as a script
    main()
```
