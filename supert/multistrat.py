#!/usr/bin/env python

# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v2.2.3 (Fortified Configuration & Enhanced Clarity/Robustness + Ehlers SSF MA + Enhanced Indicator Logging)
# Conjures high-frequency trades on Bybit Futures with enhanced precision, adaptable strategies, and Termux integration.

"""High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 2.2.3 (Unified: Selectable Strategies + Precision + Native SL/TSL + Fortified Config + Pyrmethus Enhancements + Robustness + Ehlers SSF MA + Enhanced Indicator Logging).

Features:
- Multiple strategies selectable via config: "DUAL_SUPERTREND", "STOCHRSI_MOMENTUM", "EHLERS_FISHER", "EHLERS_MA_CROSS" (using Ehlers Super Smoother Filter).
- Enhanced Precision: Uses Decimal for critical financial calculations.
- Fortified Configuration Loading: Correctly handles type casting for environment variables and default values.
- Exchange-native Trailing Stop Loss (TSL) placed immediately after entry (Bybit V5).
- Exchange-native fixed Stop Loss (based on ATR) placed immediately after entry (Bybit V5).
- ATR for volatility measurement and initial Stop-Loss calculation.
- Optional Volume spike and Order Book pressure confirmation filters.
- Risk-based position sizing with margin checks and configurable cap.
- Termux SMS alerts for critical events and trade actions (with Termux:API command check).
- Robust error handling (CCXT exceptions, validation) and detailed logging with vibrant Neon color support via Colorama.
- Graceful shutdown on KeyboardInterrupt or critical errors, attempting position/order closing.
- Stricter position detection logic tailored for Bybit V5 API (One-Way Mode).
- NaN handling in fetched OHLCV data.
- Re-validation of position state before closing.
- Enhanced Indicator Logging: Comprehensive output of key indicator values each cycle.

Disclaimer:
- **EXTREME RISK**: Trading futures, especially with leverage and automation, is extremely risky. This script is for EDUCATIONAL PURPOSES ONLY. You can lose all your capital and more. Use at your own absolute risk.
- **EXCHANGE-NATIVE SL/TSL DEPENDENCE**: Relies entirely on Bybit's native SL/TSL order execution. Performance is subject to exchange conditions, potential slippage, API reliability, and order book liquidity. These orders are NOT guaranteed to execute at the exact trigger price.
- **PARAMETER SENSITIVITY**: Bot performance is highly sensitive to parameter tuning (strategy settings, risk, SL/TSL percentages, filters). Requires significant backtesting and forward testing on TESTNET.
- **API RATE LIMITS**: Monitor API usage. Excessive requests can lead to temporary or permanent bans from the exchange.
- **SLIPPAGE**: Market orders used for entry and potentially for SL/TSL execution are prone to slippage, especially during volatile market conditions.
- **TEST THOROUGHLY**: **DO NOT RUN WITH REAL FUNDS WITHOUT EXTENSIVE AND SUCCESSFUL TESTNET/DEMO TESTING.** Understand every part of the code before considering live deployment.
- **TERMUX DEPENDENCY**: Requires Termux environment and Termux:API package (`pkg install termux-api`) for SMS alerts. Ensure it's correctly installed and configured.
- **API CHANGES**: This code targets the Bybit V5 API via CCXT. Exchange API updates may break functionality. Keep CCXT updated (`pip install -U ccxt`).
"""

# Standard Library Imports - The Foundational Runes
import logging
import os
import shutil  # For checking command existence
import subprocess
import sys
import time
import traceback
from decimal import ROUND_HALF_UP, Decimal, DivisionByZero, InvalidOperation, getcontext
from typing import Any

# Third-party Libraries - Summoned Essences
try:
    import ccxt
    import pandas as pd
    import pandas_ta as ta  # type: ignore[import]
    from colorama import Back, Fore, Style
    from colorama import init as colorama_init
    from dotenv import load_dotenv
except ImportError as e:
    missing_pkg = e.name
    # Use Colorama's raw codes here as it might not be initialized yet
    sys.stderr.write(
        f"\033[91mCRITICAL ERROR: Missing required Python package: '{missing_pkg}'.\033[0m\n"
    )
    sys.stderr.write(
        f"\033[91mPlease install it by running: pip install {missing_pkg}\033[0m\n"
    )
    if missing_pkg == "pandas_ta":
        sys.stderr.write(
            f"\033[91mFor pandas_ta, you might also need TA-Lib. See pandas_ta documentation.\033[0m\n"
        )
    sys.exit(1)

# --- Initializations - Preparing the Ritual Chamber ---
colorama_init(autoreset=True)  # Activate Colorama's magic for vibrant logs
load_dotenv()  # Load secrets from the hidden .env scroll (if present)
getcontext().prec = (
    18  # Set Decimal precision for financial exactitude (adjust if needed)
)


# --- Configuration Class - Defining the Spell's Parameters ---
class Config:
    """Loads and validates configuration parameters from environment variables."""

    def __init__(self) -> None:
        logger.info(
            f"{Fore.MAGENTA}--- Summoning Configuration Runes ---{Style.RESET_ALL}"
        )
        # --- API Credentials - Keys to the Exchange Vault ---
        self.api_key: str | None = self._get_env(
            "BYBIT_API_KEY", required=True, color=Fore.RED
        )
        self.api_secret: str | None = self._get_env(
            "BYBIT_API_SECRET", required=True, color=Fore.RED
        )

        # --- Trading Parameters - Core Incantation Variables ---
        self.symbol: str = self._get_env(
            "SYMBOL", "FARTCOIN/USDT:USDT", color=Fore.YELLOW
        )  # Target market (CCXT unified format)
        self.interval: str = self._get_env(
            "INTERVAL", "1m", color=Fore.YELLOW
        )  # Chart timeframe (e.g., '1m', '5m', '1h')
        self.leverage: int = self._get_env(
            "LEVERAGE", 25, cast_type=int, color=Fore.YELLOW
        )  # Desired leverage multiplier
        self.sleep_seconds: int = self._get_env(
            "SLEEP_SECONDS", 10, cast_type=int, color=Fore.YELLOW
        )  # Pause between trading cycles (seconds)

        # --- Strategy Selection - Choosing the Path of Magic ---
        self.strategy_name: str = self._get_env(
            "STRATEGY_NAME", "STOCHRSI_MOMENTUM", color=Fore.CYAN
        ).upper()
        self.valid_strategies: list[str] = [
            "DUAL_SUPERTREND",
            "STOCHRSI_MOMENTUM",
            "EHLERS_FISHER",
            "EHLERS_MA_CROSS",
        ]
        if self.strategy_name not in self.valid_strategies:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}Invalid STRATEGY_NAME '{self.strategy_name}'. Valid paths: {self.valid_strategies}{Style.RESET_ALL}"
            )
            raise ValueError(f"Invalid STRATEGY_NAME '{self.strategy_name}'.")
        logger.info(
            f"{Fore.CYAN}Chosen Strategy Path: {self.strategy_name}{Style.RESET_ALL}"
        )

        # --- Risk Management - Wards Against Ruin ---
        self.risk_per_trade_percentage: Decimal = self._get_env(
            "RISK_PER_TRADE_PERCENTAGE", "0.01", cast_type=Decimal, color=Fore.GREEN
        )  # e.g., 0.005 = 0.5% of equity per trade
        self.atr_stop_loss_multiplier: Decimal = self._get_env(
            "ATR_STOP_LOSS_MULTIPLIER", "1.3", cast_type=Decimal, color=Fore.GREEN
        )  # Multiplier for ATR to set initial SL distance
        self.max_order_usdt_amount: Decimal = self._get_env(
            "MAX_ORDER_USDT_AMOUNT", "50.0", cast_type=Decimal, color=Fore.GREEN
        )  # Maximum position value in USDT (overrides risk calc if needed)
        self.required_margin_buffer: Decimal = self._get_env(
            "REQUIRED_MARGIN_BUFFER", "1.05", cast_type=Decimal, color=Fore.GREEN
        )  # e.g., 1.05 = Require 5% extra free margin than estimated

        # --- Trailing Stop Loss (Exchange Native - Bybit V5) - The Adaptive Shield ---
        self.trailing_stop_percentage: Decimal = self._get_env(
            "TRAILING_STOP_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN
        )  # e.g., 0.005 = 0.5% trailing distance from high/low water mark
        self.trailing_stop_activation_offset_percent: Decimal = self._get_env(
            "TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT",
            "0.001",
            cast_type=Decimal,
            color=Fore.GREEN,
        )  # e.g., 0.001 = 0.1% price movement in profit before TSL activates

        # --- Strategy-Specific Parameters - Tuning the Chosen Path ---
        # Dual Supertrend
        self.st_atr_length: int = self._get_env(
            "ST_ATR_LENGTH", 1, cast_type=int, color=Fore.CYAN
        )  # Primary Supertrend ATR period
        self.st_multiplier: Decimal = self._get_env(
            "ST_MULTIPLIER", "0.6", cast_type=Decimal, color=Fore.CYAN
        )  # Primary Supertrend ATR multiplier
        self.confirm_st_atr_length: int = self._get_env(
            "CONFIRM_ST_ATR_LENGTH", 3, cast_type=int, color=Fore.CYAN
        )  # Confirmation Supertrend ATR period
        self.confirm_st_multiplier: Decimal = self._get_env(
            "CONFIRM_ST_MULTIPLIER", "0.2", cast_type=Decimal, color=Fore.CYAN
        )  # Confirmation Supertrend ATR multiplier
        # StochRSI + Momentum
        self.stochrsi_rsi_length: int = self._get_env(
            "STOCHRSI_RSI_LENGTH", 12, cast_type=int, color=Fore.CYAN
        )  # StochRSI: RSI period
        self.stochrsi_stoch_length: int = self._get_env(
            "STOCHRSI_STOCH_LENGTH", 11, cast_type=int, color=Fore.CYAN
        )  # StochRSI: Stochastic period
        self.stochrsi_k_period: int = self._get_env(
            "STOCHRSI_K_PERIOD", 3, cast_type=int, color=Fore.CYAN
        )  # StochRSI: %K smoothing period
        self.stochrsi_d_period: int = self._get_env(
            "STOCHRSI_D_PERIOD", 3, cast_type=int, color=Fore.CYAN
        )  # StochRSI: %D smoothing period (signal line)
        self.stochrsi_overbought: Decimal = self._get_env(
            "STOCHRSI_OVERBOUGHT", "70.0", cast_type=Decimal, color=Fore.CYAN
        )  # StochRSI overbought level
        self.stochrsi_oversold: Decimal = self._get_env(
            "STOCHRSI_OVERSOLD", "30.0", cast_type=Decimal, color=Fore.CYAN
        )  # StochRSI oversold level
        self.momentum_length: int = self._get_env(
            "MOMENTUM_LENGTH", 5, cast_type=int, color=Fore.CYAN
        )  # Momentum indicator period
        # Ehlers Fisher Transform
        self.ehlers_fisher_length: int = self._get_env(
            "EHLERS_FISHER_LENGTH", 10, cast_type=int, color=Fore.CYAN
        )  # Fisher Transform calculation period
        self.ehlers_fisher_signal_length: int = self._get_env(
            "EHLERS_FISHER_SIGNAL_LENGTH", 1, cast_type=int, color=Fore.CYAN
        )  # Fisher Transform signal line period (often 1 for just the trigger line)
        # Ehlers MA Cross (Super Smoother Filter)
        self.ehlers_fast_period: int = self._get_env(
            "EHLERS_FAST_PERIOD", 10, cast_type=int, color=Fore.CYAN
        )  # Fast Ehlers Super Smoother Filter (SSF) period
        self.ehlers_slow_period: int = self._get_env(
            "EHLERS_SLOW_PERIOD", 30, cast_type=int, color=Fore.CYAN
        )  # Slow Ehlers Super Smoother Filter (SSF) period
        self.ehlers_ssf_poles: int = self._get_env(
            "EHLERS_SSF_POLES", 2, cast_type=int, color=Fore.CYAN
        )  # Poles for Ehlers Super Smoother Filter (typically 2 or 3)

        # --- Confirmation Filters - Seeking Concordance in the Ether ---
        # Volume Analysis
        self.volume_ma_period: int = self._get_env(
            "VOLUME_MA_PERIOD", 20, cast_type=int, color=Fore.YELLOW
        )  # Moving average period for volume
        self.volume_spike_threshold: Decimal = self._get_env(
            "VOLUME_SPIKE_THRESHOLD", "1.5", cast_type=Decimal, color=Fore.YELLOW
        )  # Multiplier over MA to consider a 'spike' (e.g., 1.5 = 150% of MA)
        self.require_volume_spike_for_entry: bool = self._get_env(
            "REQUIRE_VOLUME_SPIKE_FOR_ENTRY", "false", cast_type=bool, color=Fore.YELLOW
        )  # Require volume spike for entry signal?
        # Order Book Analysis
        self.order_book_depth: int = self._get_env(
            "ORDER_BOOK_DEPTH", 20, cast_type=int, color=Fore.YELLOW
        )  # Number of bid/ask levels to analyze
        self.order_book_ratio_threshold_long: Decimal = self._get_env(
            "ORDER_BOOK_RATIO_THRESHOLD_LONG",
            "1.2",
            cast_type=Decimal,
            color=Fore.YELLOW,
        )  # Min Bid/Ask volume ratio for long confirmation
        self.order_book_ratio_threshold_short: Decimal = self._get_env(
            "ORDER_BOOK_RATIO_THRESHOLD_SHORT",
            "0.8",
            cast_type=Decimal,
            color=Fore.YELLOW,
        )  # Max Bid/Ask volume ratio for short confirmation (ratio = Bids/Asks)
        self.fetch_order_book_per_cycle: bool = self._get_env(
            "FETCH_ORDER_BOOK_PER_CYCLE", "true", cast_type=bool, color=Fore.YELLOW
        )  # Fetch OB every cycle (more API calls) or only when signal occurs?

        # --- ATR Calculation (for Initial SL) ---
        self.atr_calculation_period: int = self._get_env(
            "ATR_CALCULATION_PERIOD", 14, cast_type=int, color=Fore.GREEN
        )  # Period for ATR calculation used in SL

        # --- Termux SMS Alerts - Whispers Through the Digital Veil ---
        self.enable_sms_alerts: bool = self._get_env(
            "ENABLE_SMS_ALERTS", "true", cast_type=bool, color=Fore.MAGENTA
        )  # Enable/disable SMS alerts globally
        self.sms_recipient_number: str | None = self._get_env(
            "SMS_RECIPIENT_NUMBER", +16364866381, color=Fore.MAGENTA
        )  # Recipient phone number for alerts
        self.sms_timeout_seconds: int = self._get_env(
            "SMS_TIMEOUT_SECONDS", 30, cast_type=int, color=Fore.MAGENTA
        )  # Max time to wait for SMS command (seconds)

        # --- CCXT / API Parameters - Tuning the Connection ---
        self.default_recv_window: int = (
            13000  # Milliseconds for API request validity (Bybit default is 5000)
        )
        self.order_book_fetch_limit: int = max(
            25, self.order_book_depth
        )  # How many levels to fetch (ensure >= depth needed)
        self.shallow_ob_fetch_depth: int = (
            5  # Depth for quick price estimates (used in order placement)
        )
        self.order_fill_timeout_seconds: int = self._get_env(
            "ORDER_FILL_TIMEOUT_SECONDS", 15, cast_type=int, color=Fore.YELLOW
        )  # Max time to wait for market order fill confirmation (seconds)

        # --- Internal Constants - Fixed Arcane Symbols ---
        self.side_buy: str = "buy"
        self.side_sell: str = "sell"
        self.pos_long: str = "Long"
        self.pos_short: str = "Short"
        self.pos_none: str = "None"
        self.usdt_symbol: str = "USDT"  # The stable anchor currency
        self.retry_count: int = 3  # Default attempts for certain retryable API calls
        self.retry_delay_seconds: int = 2  # Default pause between retries (seconds)
        self.api_fetch_limit_buffer: int = (
            10  # Extra candles to fetch beyond indicator needs
        )
        self.position_qty_epsilon: Decimal = Decimal(
            "1e-9"
        )  # Small value for float comparisons involving position size
        self.post_close_delay_seconds: int = (
            3  # Brief pause after successfully closing a position (seconds)
        )

        logger.info(
            f"{Fore.MAGENTA}--- Configuration Runes Summoned and Verified ---{Style.RESET_ALL}"
        )

    def _get_env(
        self,
        key: str,
        default: Any = None,
        cast_type: type = str,
        required: bool = False,
        color: str = Fore.WHITE,
    ) -> Any:
        """Fetches env var, casts type (including defaults), logs, handles defaults/errors with arcane grace.
        Ensures that default values are also cast to the specified type. Handles bool, int, Decimal, float, str.
        """
        value_str = os.getenv(key)  # Get raw string value from environment
        source = "Env Var"
        value_to_cast: Any = None  # Variable to hold the value before casting

        if value_str is None:
            # Environment variable not found
            if required:
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}CRITICAL: Required configuration rune '{key}' not found in the environment scroll (.env).{Style.RESET_ALL}"
                )
                raise ValueError(f"Required environment variable '{key}' not set.")
            logger.debug(
                f"{color}Summoning {key}: Not Set. Using Default: '{default}'{Style.RESET_ALL}"
            )
            value_to_cast = default  # Assign default, needs casting below
            source = "Default"
        else:
            # Environment variable found
            logger.debug(
                f"{color}Summoning {key}: Found Env Value: '{value_str}'{Style.RESET_ALL}"
            )
            value_to_cast = value_str  # Assign found string, needs casting below

        # --- Attempt Casting (applies to both env var value and default value) ---
        if value_to_cast is None:
            # This handles cases where default=None or env var was explicitly empty and default was None
            if required:
                # This should have been caught earlier if env var was missing, but catches required=True with default=None
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}CRITICAL: Required configuration rune '{key}' has no value (from env or default).{Style.RESET_ALL}"
                )
                raise ValueError(
                    f"Required environment variable '{key}' resolved to None."
                )
            else:
                # Not required and value is None, return None directly
                logger.debug(
                    f"{color}Final value for {key}: None (Type: NoneType) (Source: {source}){Style.RESET_ALL}"
                )
                return None

        final_value: Any = None
        try:
            raw_value_str = str(
                value_to_cast
            )  # Ensure we have a string representation for casting checks
            if cast_type == bool:
                final_value = raw_value_str.lower() in ["true", "1", "yes", "y"]
            elif cast_type == Decimal:
                final_value = Decimal(raw_value_str)
            elif cast_type == int:
                final_value = int(
                    Decimal(raw_value_str)
                )  # Cast via Decimal to handle "10.0" -> 10
            elif cast_type == float:
                final_value = float(raw_value_str)
            elif cast_type == str:
                final_value = raw_value_str  # Already string or cast to string
            else:
                # Should not happen if using supported types, but good practice
                logger.warning(
                    f"Unsupported cast_type '{cast_type.__name__}' for key '{key}'. Returning raw value."
                )
                final_value = value_to_cast  # Return the original value

        except (ValueError, TypeError, InvalidOperation) as e:
            # Casting failed! Log error and attempt to use default, casting it carefully.
            logger.error(
                f"{Fore.RED}Invalid type/value for {key}: '{value_to_cast}' (Source: {source}). Expected {cast_type.__name__}. Error: {e}. Attempting to use default '{default}'.{Style.RESET_ALL}"
            )
            if default is None:
                # If default is also None, and required, we have a critical problem.
                if required:
                    logger.critical(
                        f"{Back.RED}{Fore.WHITE}CRITICAL: Failed to cast value for required key '{key}' and default is None.{Style.RESET_ALL}"
                    )
                    raise ValueError(
                        f"Required env var '{key}' failed casting and has no valid default."
                    )
                else:
                    # Not required, default is None, casting failed -> return None
                    logger.warning(
                        f"{Fore.YELLOW}Casting failed for {key}, default is None. Final value: None{Style.RESET_ALL}"
                    )
                    return None
            else:
                # Try casting the default value itself
                source = "Default (Fallback)"
                logger.debug(
                    f"Attempting to cast fallback default value '{default}' for key '{key}' to {cast_type.__name__}"
                )
                try:
                    default_str = str(default)
                    if cast_type == bool:
                        final_value = default_str.lower() in ["true", "1", "yes", "y"]
                    elif cast_type == Decimal:
                        final_value = Decimal(default_str)
                    elif cast_type == int:
                        final_value = int(Decimal(default_str))
                    elif cast_type == float:
                        final_value = float(default_str)
                    elif cast_type == str:
                        final_value = default_str
                    else:
                        final_value = default  # Fallback to raw default if type unknown

                    logger.warning(
                        f"{Fore.YELLOW}Successfully used casted default value for {key}: '{final_value}'{Style.RESET_ALL}"
                    )
                except (ValueError, TypeError, InvalidOperation) as e_default:
                    # If even the default fails casting, it's a critical config issue
                    logger.critical(
                        f"{Back.RED}{Fore.WHITE}CRITICAL: Failed to cast BOTH provided value ('{value_to_cast}') AND default value ('{default}') for key '{key}' to {cast_type.__name__}. Error on default: {e_default}{Style.RESET_ALL}"
                    )
                    raise ValueError(
                        f"Configuration error: Cannot cast value or default for key '{key}' to {cast_type.__name__}."
                    )

        # Log the final type and value being used
        logger.debug(
            f"{color}Using final value for {key}: {final_value} (Type: {type(final_value).__name__}) (Source: {source}){Style.RESET_ALL}"
        )
        return final_value


# --- Logger Setup - The Oracle's Voice ---
LOGGING_LEVEL: int = (
    logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO
)
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],  # Output to the Termux console (or wherever stdout goes)
)
logger: logging.Logger = logging.getLogger(__name__)

# Custom SUCCESS level and Neon Color Formatting for the Oracle
SUCCESS_LEVEL: int = 25  # Between INFO and WARNING
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Logs a success message with mystical flair."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        # pylint: disable=protected-access
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


logging.Logger.success = log_success  # type: ignore[attr-defined] # Add method to Logger class

# Apply colors if outputting to a TTY (like Termux console)
if sys.stdout.isatty():
    logging.addLevelName(
        logging.DEBUG,
        f"{Fore.CYAN}{Style.DIM}{logging.getLevelName(logging.DEBUG)}{Style.RESET_ALL}",
    )  # Dim Cyan for Debug
    logging.addLevelName(
        logging.INFO,
        f"{Fore.BLUE}{logging.getLevelName(logging.INFO)}{Style.RESET_ALL}",
    )  # Blue for Info
    logging.addLevelName(
        SUCCESS_LEVEL,
        f"{Fore.MAGENTA}{Style.BRIGHT}{logging.getLevelName(SUCCESS_LEVEL)}{Style.RESET_ALL}",
    )  # Bright Magenta for Success
    logging.addLevelName(
        logging.WARNING,
        f"{Fore.YELLOW}{Style.BRIGHT}{logging.getLevelName(logging.WARNING)}{Style.RESET_ALL}",
    )  # Bright Yellow for Warning
    logging.addLevelName(
        logging.ERROR,
        f"{Fore.RED}{Style.BRIGHT}{logging.getLevelName(logging.ERROR)}{Style.RESET_ALL}",
    )  # Bright Red for Error
    logging.addLevelName(
        logging.CRITICAL,
        f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}{logging.getLevelName(logging.CRITICAL)}{Style.RESET_ALL}",
    )  # White on Red for Critical

# --- Global Objects - Instantiated Arcana ---
try:
    CONFIG = Config()  # Forge the configuration object
except ValueError as config_error:
    # Error should have been logged within Config init or _get_env
    logger.critical(
        f"{Back.RED}{Fore.WHITE}Configuration loading failed. Cannot continue spellcasting. Error: {config_error}{Style.RESET_ALL}"
    )
    sys.exit(1)
except Exception as general_config_error:
    # Catch any other unexpected error during config init
    logger.critical(
        f"{Back.RED}{Fore.WHITE}Unexpected critical error during configuration: {general_config_error}{Style.RESET_ALL}"
    )
    logger.debug(traceback.format_exc())
    sys.exit(1)


# --- Helper Functions - Minor Cantrips ---
def safe_decimal_conversion(value: Any, default: Decimal = Decimal("0.0")) -> Decimal:
    """Safely converts a value to Decimal, returning default if conversion fails or value is None."""
    if value is None:
        return default
    try:
        # Use str(value) to handle potential float inputs more reliably before Decimal conversion
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        logger.warning(
            f"Could not convert '{value}' (type: {type(value).__name__}) to Decimal, using default {default}"
        )
        return default


def format_order_id(order_id: str | int | None) -> str:
    """Returns the last 6 characters of an order ID or 'N/A' for brevity."""
    return str(order_id)[-6:] if order_id else "N/A"


def _format_for_log(value: Any, precision: int = 4, is_bool_trend: bool = False) -> str:
    """Helper function to format values for logging, especially indicators."""
    if pd.isna(value) or value is None:
        return "N/A"
    if is_bool_trend:  # For boolean trend indicators (True=Up, False=Down)
        if value is True:
            return "Up"
        if value is False:
            return "Down"
    if isinstance(value, Decimal):
        return f"{value:.{precision}f}"
    if isinstance(value, (float, int)):
        # Ensure float conversion for consistent formatting
        return f"{float(value):.{precision}f}"
    if isinstance(value, bool):  # For non-trend booleans (e.g. flip signals)
        return str(value)
    return str(value)  # Fallback for other types


# --- Precision Formatting - Shaping the Numbers for the Exchange ---
def format_price(exchange: ccxt.Exchange, symbol: str, price: float | Decimal) -> str:
    """Formats price according to market precision rules using CCXT."""
    try:
        # CCXT formatting methods typically expect float input
        price_float = float(price)
        return exchange.price_to_precision(symbol, price_float)
    except (ccxt.ExchangeError, ValueError, TypeError) as e:
        logger.error(
            f"{Fore.RED}Error shaping price {price} for {symbol}: {e}. Using raw Decimal string.{Style.RESET_ALL}"
        )
        # Fallback: return a normalized string representation of the Decimal
        return str(Decimal(str(price)).normalize())
    except Exception as e_unexp:
        logger.error(
            f"{Fore.RED}Unexpected error shaping price {price} for {symbol}: {e_unexp}. Using raw Decimal string.{Style.RESET_ALL}"
        )
        return str(Decimal(str(price)).normalize())


def format_amount(exchange: ccxt.Exchange, symbol: str, amount: float | Decimal) -> str:
    """Formats amount according to market precision rules using CCXT."""
    try:
        # CCXT formatting methods typically expect float input
        amount_float = float(amount)
        return exchange.amount_to_precision(symbol, amount_float)
    except (ccxt.ExchangeError, ValueError, TypeError) as e:
        logger.error(
            f"{Fore.RED}Error shaping amount {amount} for {symbol}: {e}. Using raw Decimal string.{Style.RESET_ALL}"
        )
        # Fallback: return a normalized string representation of the Decimal
        return str(Decimal(str(amount)).normalize())
    except Exception as e_unexp:
        logger.error(
            f"{Fore.RED}Unexpected error shaping amount {amount} for {symbol}: {e_unexp}. Using raw Decimal string.{Style.RESET_ALL}"
        )
        return str(Decimal(str(amount)).normalize())


# --- Termux SMS Alert Function - Sending Whispers ---
_termux_sms_command_exists: bool | None = (
    None  # Cache the result of checking command existence
)


def send_sms_alert(message: str) -> bool:
    """Sends an SMS alert using Termux API, if enabled and available."""
    global _termux_sms_command_exists

    if not CONFIG.enable_sms_alerts:
        logger.debug("SMS alerts disabled by configuration.")
        return False

    # Check for command existence only once
    if _termux_sms_command_exists is None:
        _termux_sms_command_exists = shutil.which("termux-sms-send") is not None
        if not _termux_sms_command_exists:
            logger.warning(
                f"{Fore.YELLOW}SMS alerts enabled, but 'termux-sms-send' command not found. "
                f"Ensure Termux:API is installed (`pkg install termux-api`) and configured.{Style.RESET_ALL}"
            )

    if not _termux_sms_command_exists:
        return False  # Don't proceed if command is missing

    if not CONFIG.sms_recipient_number:
        logger.warning(
            f"{Fore.YELLOW}SMS alerts enabled, but SMS_RECIPIENT_NUMBER is not set in config.{Style.RESET_ALL}"
        )
        return False

    try:
        # Prepare the command spell, ensuring message is treated as a single argument
        # Using shlex.quote defensively, though termux-sms-send might handle spaces fine.
        # However, the API expects the message as the last argument(s).
        command: list[str] = [
            "termux-sms-send",
            "-n",
            CONFIG.sms_recipient_number,
            message,
        ]
        logger.info(
            f"{Fore.MAGENTA}Dispatching SMS whisper to {CONFIG.sms_recipient_number} (Timeout: {CONFIG.sms_timeout_seconds}s)...{Style.RESET_ALL}"
        )

        # Execute the spell via subprocess
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=CONFIG.sms_timeout_seconds,
        )

        if result.returncode == 0:
            logger.success(
                f"{Fore.MAGENTA}SMS whisper dispatched successfully.{Style.RESET_ALL}"
            )
            return True
        else:
            # Log error details from stderr if available
            error_details = (
                result.stderr.strip() if result.stderr else "No stderr output"
            )
            logger.error(
                f"{Fore.RED}SMS whisper failed. Return Code: {result.returncode}, Stderr: {error_details}{Style.RESET_ALL}"
            )
            return False
    except FileNotFoundError:
        # This shouldn't happen due to the check above, but handle defensively
        logger.error(
            f"{Fore.RED}SMS failed: 'termux-sms-send' command vanished unexpectedly? Ensure Termux:API is installed.{Style.RESET_ALL}"
        )
        _termux_sms_command_exists = False  # Update cache
        return False
    except subprocess.TimeoutExpired:
        logger.error(
            f"{Fore.RED}SMS failed: Command timed out after {CONFIG.sms_timeout_seconds}s.{Style.RESET_ALL}"
        )
        return False
    except Exception as e:
        logger.error(
            f"{Fore.RED}SMS failed: Unexpected disturbance during dispatch: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        return False


# --- Exchange Initialization - Opening the Portal ---
def initialize_exchange() -> ccxt.Exchange | None:
    """Initializes and returns the CCXT Bybit exchange instance, handling authentication and basic checks."""
    logger.info(f"{Fore.BLUE}Opening portal to Bybit via CCXT...{Style.RESET_ALL}")
    if not CONFIG.api_key or not CONFIG.api_secret:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}CRITICAL: API Key/Secret runes missing in configuration. Cannot open portal.{Style.RESET_ALL}"
        )
        send_sms_alert("[Pyrmethus] CRITICAL: API keys missing. Spell failed.")
        return None
    try:
        # Forging the connection with Bybit V5 defaults
        exchange = ccxt.bybit(
            {
                "apiKey": CONFIG.api_key,
                "secret": CONFIG.api_secret,
                "enableRateLimit": True,  # Respect the exchange spirits' limits
                "options": {
                    "defaultType": "linear",  # Assume USDT perpetuals unless symbol specifies otherwise
                    "adjustForTimeDifference": True,  # Sync client time with server time
                    # V5 API specific options might be added here if needed, but CCXT handles most.
                },
                # Explicitly set API version if necessary, though CCXT usually defaults correctly
                # 'options': {'api-version': 'v5'}, # Uncomment if explicit V5 needed
                "recvWindow": CONFIG.default_recv_window,  # Set receive window
            }
        )
        # exchange.set_sandbox_mode(True) # Uncomment for Bybit Testnet

        logger.debug("Loading market structures from Bybit...")
        exchange.load_markets(True)  # Force reload for fresh data
        logger.debug(f"Loaded {len(exchange.markets)} market structures.")

        logger.debug("Performing initial balance check for authentication...")
        # Fetch balance to confirm API keys are valid and connection works
        exchange.fetch_balance(
            params={"category": "linear"}
        )  # Specify category for V5 balance check
        logger.success(
            f"{Fore.GREEN}{Style.BRIGHT}Portal to Bybit Opened & Authenticated (Targeting V5 API).{Style.RESET_ALL}"
        )
        logger.warning(
            f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}!!! LIVE SCALPING MODE ACTIVE - EXTREME CAUTION ADVISED !!!{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[Pyrmethus/{CONFIG.strategy_name}] Portal opened & authenticated."
        )
        return exchange

    except ccxt.AuthenticationError as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Authentication failed: {e}. Check API keys, permissions, and IP whitelist on Bybit.{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[Pyrmethus] CRITICAL: Authentication FAILED: {e}. Spell failed."
        )
    except ccxt.NetworkError as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Network disturbance during portal opening: {e}. Check internet connection and Bybit status.{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[Pyrmethus] CRITICAL: Network Error on Init: {e}. Spell failed."
        )
    except ccxt.ExchangeError as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Exchange spirit rejected portal opening: {e}. Check Bybit status, API documentation, or account status.{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[Pyrmethus] CRITICAL: Exchange Error on Init: {e}. Spell failed."
        )
    except Exception as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Unexpected chaos during portal opening: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        send_sms_alert(
            f"[Pyrmethus] CRITICAL: Unexpected Init Error: {type(e).__name__}. Spell failed."
        )

    return None  # Return None if initialization failed


# --- Indicator Calculation Functions - Scrying the Market ---


def calculate_supertrend(
    df: pd.DataFrame, length: int, multiplier: Decimal, prefix: str = ""
) -> pd.DataFrame:
    """Calculates the Supertrend indicator using pandas_ta, returning Decimal where applicable."""
    col_prefix = f"{prefix}" if prefix else ""
    target_cols = [
        f"{col_prefix}supertrend",
        f"{col_prefix}trend",
        f"{col_prefix}st_long",
        f"{col_prefix}st_short",
    ]
    # pandas_ta column naming convention (may vary slightly with versions)
    st_col = f"SUPERT_{length}_{float(multiplier)}"
    st_trend_col = f"SUPERTd_{length}_{float(multiplier)}"
    st_long_col = f"SUPERTl_{length}_{float(multiplier)}"  # Long signal column
    st_short_col = f"SUPERTs_{length}_{float(multiplier)}"  # Short signal column
    raw_st_cols = [
        st_col,
        st_trend_col,
        st_long_col,
        st_short_col,
    ]  # Columns pandas_ta creates

    required_input_cols = ["high", "low", "close"]
    min_len = length + 1  # Minimum data length required

    if (
        df is None
        or df.empty
        or not all(c in df.columns for c in required_input_cols)
        or len(df) < min_len
    ):
        logger.warning(
            f"{Fore.YELLOW}Scrying ({col_prefix}ST): Insufficient data (Len: {len(df) if df is not None else 0}, Need: {min_len}).{Style.RESET_ALL}"
        )
        for col in target_cols:  # Use a copy to avoid SettingWithCopyWarning
            if df is not None:
                df[col] = pd.NA
        return df if df is not None else pd.DataFrame()

    try:
        # pandas_ta expects float multiplier for calculation and column naming
        logger.debug(
            f"Scrying ({col_prefix}ST): Calculating with length={length}, multiplier={float(multiplier)}"
        )
        # Operate on a copy if df might be a slice, or ensure df is the full DataFrame
        df_copy = df.copy()
        df_copy.ta.supertrend(length=length, multiplier=float(multiplier), append=True)

        # Check if pandas_ta created the expected columns
        if not all(
            c in df_copy.columns
            for c in [st_col, st_trend_col, st_long_col, st_short_col]
        ):
            # Find which columns are missing
            missing = [c for c in raw_st_cols if c not in df_copy.columns]
            raise KeyError(
                f"pandas_ta failed to create expected raw columns for {col_prefix}ST: {missing}"
            )

        # Convert Supertrend value to Decimal, interpret trend and flips
        df[f"{col_prefix}supertrend"] = df_copy[st_col].apply(safe_decimal_conversion)
        # Trend: 1 = Uptrend, -1 = Downtrend. Convert to boolean: True for Up, False for Down.
        df[f"{col_prefix}trend"] = df_copy[st_trend_col] == 1
        # Flip Signals (pandas_ta provides these directly in recent versions)
        # st_long_col (SUPERTl): 1.0 when trend flips Long, NaN otherwise
        # st_short_col (SUPERTs): -1.0 when trend flips Short, NaN otherwise
        df[f"{col_prefix}st_long"] = (
            df_copy[st_long_col] == 1.0
        )  # True if flipped Long this candle
        df[f"{col_prefix}st_short"] = (
            df_copy[st_short_col] == -1.0
        )  # True if flipped Short this candle

        # Clean up raw columns created by pandas_ta from the original df IF they somehow got there
        # This is defensive; they should only be on df_copy if not appended to original by mistake
        # However, append=True was used, so they are on df_copy which becomes df.
        # The assignment `df[...] = df_copy[...]` copies the new columns.
        # We need to drop the raw TA-Lib named columns from `df` after processing.
        # df.drop(columns=raw_st_cols, errors="ignore", inplace=True) # This was original
        # The raw_st_cols are from df_copy, which was used with append=True.
        # If df passed in is the main df, then append=True modified it.
        # Let's ensure df is modified directly by pandas_ta, then drop.

        # Re-do with direct modification of df to simplify column management
        df.ta.supertrend(length=length, multiplier=float(multiplier), append=True)
        if not all(
            c in df.columns for c in [st_col, st_trend_col, st_long_col, st_short_col]
        ):
            missing = [c for c in raw_st_cols if c not in df.columns]
            raise KeyError(
                f"pandas_ta failed to create expected raw columns for {col_prefix}ST (direct mod): {missing}"
            )

        df_orig_cols = (
            df.columns.tolist()
        )  # Save original columns before adding target ones

        df[f"{col_prefix}supertrend"] = df[st_col].apply(safe_decimal_conversion)
        df[f"{col_prefix}trend"] = df[st_trend_col] == 1
        df[f"{col_prefix}st_long"] = df[st_long_col] == 1.0
        df[f"{col_prefix}st_short"] = df[st_short_col] == -1.0

        # Drop only the pandas_ta specific columns, not user-defined ones if they conflict
        cols_to_drop = [
            c for c in raw_st_cols if c in df.columns and c not in target_cols
        ]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)

        # Log the latest reading for debugging
        if not df.empty:
            last_st_val = df[f"{col_prefix}supertrend"].iloc[-1]
            if pd.notna(last_st_val):
                last_trend_val = df[f"{col_prefix}trend"].iloc[-1]
                last_trend = (
                    "Up"
                    if last_trend_val is True
                    else ("Down" if last_trend_val is False else "N/A")
                )

                signal_long = df[f"{col_prefix}st_long"].iloc[-1]
                signal_short = df[f"{col_prefix}st_short"].iloc[-1]
                signal = (
                    "LONG FLIP"
                    if signal_long
                    else ("SHORT FLIP" if signal_short else "Hold")
                )
                trend_color = Fore.GREEN if last_trend == "Up" else Fore.RED
                logger.debug(
                    f"Scrying ({col_prefix}ST({length},{multiplier})): Trend={trend_color}{last_trend}{Style.RESET_ALL}, Val={last_st_val:.4f}, Signal={signal}"
                )
            else:
                logger.debug(
                    f"Scrying ({col_prefix}ST({length},{multiplier})): Resulted in NA for last candle."
                )

    except KeyError as e:
        logger.error(
            f"{Fore.RED}Scrying ({col_prefix}ST): Error accessing column - likely pandas_ta issue or data problem: {e}{Style.RESET_ALL}"
        )
        for col in target_cols:
            df[col] = pd.NA
    except Exception as e:
        logger.error(
            f"{Fore.RED}Scrying ({col_prefix}ST): Unexpected error during calculation: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        for col in target_cols:
            df[col] = pd.NA
    return df


def analyze_volume_atr(
    df: pd.DataFrame, atr_len: int, vol_ma_len: int
) -> dict[str, Decimal | None]:
    """Calculates ATR, Volume MA, checks spikes. Returns Dict with Decimals for volatility and volume metrics."""
    results: dict[str, Decimal | None] = {
        "atr": None,
        "volume_ma": None,
        "last_volume": None,
        "volume_ratio": None,
    }
    required_cols = ["high", "low", "close", "volume"]
    min_len = max(atr_len, vol_ma_len) + 1

    if (
        df is None
        or df.empty
        or not all(c in df.columns for c in required_cols)
        or len(df) < min_len
    ):
        logger.warning(
            f"{Fore.YELLOW}Scrying (Vol/ATR): Insufficient data (Len: {len(df) if df is not None else 0}, Need: {min_len}).{Style.RESET_ALL}"
        )
        return results

    try:
        logger.debug(f"Scrying (ATR): Calculating with length={atr_len}")
        atr_col = f"ATRr_{atr_len}"
        df.ta.atr(length=atr_len, append=True)  # Modifies df directly
        if atr_col in df.columns and not df.empty:
            last_atr = df[atr_col].iloc[-1]
            if pd.notna(last_atr):
                results["atr"] = safe_decimal_conversion(last_atr)
            df.drop(columns=[atr_col], errors="ignore", inplace=True)
        else:
            logger.warning(
                f"ATR column '{atr_col}' not found or df empty after calculation."
            )

        logger.debug(f"Scrying (Volume): Calculating MA with length={vol_ma_len}")
        volume_ma_col = f"volume_sma_{vol_ma_len}"
        df[volume_ma_col] = (
            df["volume"]
            .rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2))
            .mean()
        )
        if not df.empty:
            last_vol_ma = df[volume_ma_col].iloc[-1]
            last_vol = df["volume"].iloc[-1]

            if pd.notna(last_vol_ma):
                results["volume_ma"] = safe_decimal_conversion(last_vol_ma)
            if pd.notna(last_vol):
                results["last_volume"] = safe_decimal_conversion(last_vol)

            if (
                results["volume_ma"] is not None
                and results["volume_ma"] > CONFIG.position_qty_epsilon
                and results["last_volume"] is not None
            ):
                try:
                    results["volume_ratio"] = (
                        results["last_volume"] / results["volume_ma"]
                    )
                except DivisionByZero:
                    logger.warning("Division by zero calculating volume ratio.")
                    results["volume_ratio"] = None
            else:
                results["volume_ratio"] = None

        if volume_ma_col in df.columns:
            df.drop(columns=[volume_ma_col], errors="ignore", inplace=True)

        atr_str = f"{results['atr']:.5f}" if results["atr"] else "N/A"
        vol_ma_str = f"{results['volume_ma']:.2f}" if results["volume_ma"] else "N/A"
        last_vol_str = (
            f"{results['last_volume']:.2f}" if results["last_volume"] else "N/A"
        )
        vol_ratio_str = (
            f"{results['volume_ratio']:.2f}" if results["volume_ratio"] else "N/A"
        )
        logger.debug(
            f"Scrying Results: ATR({atr_len})={Fore.CYAN}{atr_str}{Style.RESET_ALL}, "
            f"LastVol={last_vol_str}, VolMA({vol_ma_len})={vol_ma_str}, VolRatio={Fore.YELLOW}{vol_ratio_str}{Style.RESET_ALL}"
        )

    except Exception as e:
        logger.error(
            f"{Fore.RED}Scrying (Vol/ATR): Unexpected error: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        results = {key: None for key in results}  # Nullify on error
    return results


def calculate_stochrsi_momentum(
    df: pd.DataFrame, rsi_len: int, stoch_len: int, k: int, d: int, mom_len: int
) -> pd.DataFrame:
    """Calculates StochRSI and Momentum, gauging overbought/oversold conditions and trend strength."""
    target_cols = ["stochrsi_k", "stochrsi_d", "momentum"]
    min_len = max(rsi_len + stoch_len + d, mom_len) + 5

    if df is None or df.empty or "close" not in df.columns or len(df) < min_len:
        logger.warning(
            f"{Fore.YELLOW}Scrying (StochRSI/Mom): Insufficient data (Len: {len(df) if df is not None else 0}, Need ~{min_len}).{Style.RESET_ALL}"
        )
        if df is not None:
            for col in target_cols:
                df[col] = pd.NA
        return df if df is not None else pd.DataFrame()

    try:
        logger.debug(
            f"Scrying (StochRSI): Calculating with RSI={rsi_len}, Stoch={stoch_len}, K={k}, D={d}"
        )
        stochrsi_df = df.ta.stochrsi(
            length=stoch_len, rsi_length=rsi_len, k=k, d=d, append=False
        )
        k_col_ta = f"STOCHRSIk_{stoch_len}_{rsi_len}_{k}_{d}"
        d_col_ta = f"STOCHRSId_{stoch_len}_{rsi_len}_{k}_{d}"

        if k_col_ta in stochrsi_df.columns:
            df["stochrsi_k"] = stochrsi_df[k_col_ta].apply(safe_decimal_conversion)
        else:
            logger.warning(f"StochRSI K column '{k_col_ta}' not found.")
            df["stochrsi_k"] = pd.NA
        if d_col_ta in stochrsi_df.columns:
            df["stochrsi_d"] = stochrsi_df[d_col_ta].apply(safe_decimal_conversion)
        else:
            logger.warning(f"StochRSI D column '{d_col_ta}' not found.")
            df["stochrsi_d"] = pd.NA

        logger.debug(f"Scrying (Momentum): Calculating with length={mom_len}")
        mom_col_ta = f"MOM_{mom_len}"
        df.ta.mom(length=mom_len, append=True)  # Modifies df
        if mom_col_ta in df.columns:
            df["momentum"] = df[mom_col_ta].apply(safe_decimal_conversion)
            df.drop(columns=[mom_col_ta], errors="ignore", inplace=True)
        else:
            logger.warning(f"Momentum column '{mom_col_ta}' not found.")
            df["momentum"] = pd.NA

        if not df.empty:
            k_val = df["stochrsi_k"].iloc[-1]
            d_val = df["stochrsi_d"].iloc[-1]
            mom_val = df["momentum"].iloc[-1]

            if pd.notna(k_val) and pd.notna(d_val) and pd.notna(mom_val):
                k_color = (
                    Fore.RED
                    if k_val > CONFIG.stochrsi_overbought
                    else (Fore.GREEN if k_val < CONFIG.stochrsi_oversold else Fore.CYAN)
                )
                d_color = (
                    Fore.RED
                    if d_val > CONFIG.stochrsi_overbought
                    else (Fore.GREEN if d_val < CONFIG.stochrsi_oversold else Fore.CYAN)
                )
                mom_color = (
                    Fore.GREEN
                    if mom_val > CONFIG.position_qty_epsilon
                    else (
                        Fore.RED
                        if mom_val < -CONFIG.position_qty_epsilon
                        else Fore.WHITE
                    )
                )
                logger.debug(
                    f"Scrying (StochRSI/Mom): K={k_color}{k_val:.2f}{Style.RESET_ALL}, D={d_color}{d_val:.2f}{Style.RESET_ALL}, Mom={mom_color}{mom_val:.4f}{Style.RESET_ALL}"
                )
            else:
                logger.debug("Scrying (StochRSI/Mom): NA values on last candle.")

    except Exception as e:
        logger.error(
            f"{Fore.RED}Scrying (StochRSI/Mom): Unexpected error: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        for col in target_cols:
            df[col] = pd.NA
    return df


def calculate_ehlers_fisher(df: pd.DataFrame, length: int, signal: int) -> pd.DataFrame:
    """Calculates Ehlers Fisher Transform indicator, seeking cyclical turning points."""
    target_cols = ["ehlers_fisher", "ehlers_signal"]
    required_input_cols = ["high", "low"]
    min_len = length + signal

    if (
        df is None
        or df.empty
        or not all(c in df.columns for c in required_input_cols)
        or len(df) < min_len
    ):
        logger.warning(
            f"{Fore.YELLOW}Scrying (EhlersFisher): Insufficient data (Len: {len(df) if df is not None else 0}, Need ~{min_len}).{Style.RESET_ALL}"
        )
        if df is not None:
            for col in target_cols:
                df[col] = pd.NA
        return df if df is not None else pd.DataFrame()

    try:
        logger.debug(
            f"Scrying (EhlersFisher): Calculating with length={length}, signal={signal}"
        )
        fisher_df = df.ta.fisher(length=length, signal=signal, append=False)
        fish_col_ta = f"FISHERT_{length}_{signal}"
        signal_col_ta = f"FISHERTs_{length}_{signal}"

        if fish_col_ta in fisher_df.columns:
            df["ehlers_fisher"] = fisher_df[fish_col_ta].apply(safe_decimal_conversion)
        else:
            logger.warning(f"Ehlers Fisher column '{fish_col_ta}' not found.")
            df["ehlers_fisher"] = pd.NA
        if signal_col_ta in fisher_df.columns:
            df["ehlers_signal"] = fisher_df[signal_col_ta].apply(
                safe_decimal_conversion
            )
        else:
            logger.warning(f"Ehlers Signal column '{signal_col_ta}' not found.")
            df["ehlers_signal"] = pd.NA

        if not df.empty:
            fish_val = df["ehlers_fisher"].iloc[-1]
            sig_val = df["ehlers_signal"].iloc[-1]
            if pd.notna(fish_val) and pd.notna(sig_val):
                logger.debug(
                    f"Scrying (EhlersFisher({length},{signal})): Fisher={Fore.CYAN}{fish_val:.4f}{Style.RESET_ALL}, Signal={Fore.MAGENTA}{sig_val:.4f}{Style.RESET_ALL}"
                )
            else:
                logger.debug("Scrying (EhlersFisher): NA values on last candle.")
    except Exception as e:
        logger.error(
            f"{Fore.RED}Scrying (EhlersFisher): Unexpected error: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        for col in target_cols:
            df[col] = pd.NA
    return df


def calculate_ehlers_ma(
    df: pd.DataFrame, fast_len: int, slow_len: int, poles: int
) -> pd.DataFrame:
    """Calculates Ehlers Super Smoother Filter (SSF) MAs for fast and slow periods."""
    target_cols = ["ehlers_ssf_fast", "ehlers_ssf_slow"]
    min_len = max(fast_len, slow_len) + poles + 5

    if df is None or df.empty or "close" not in df.columns or len(df) < min_len:
        logger.warning(
            f"{Fore.YELLOW}Scrying (Ehlers SSF MA): Insufficient data (Len: {len(df) if df is not None else 0}, Need ~{min_len}).{Style.RESET_ALL}"
        )
        if df is not None:
            for col in target_cols:
                df[col] = pd.NA
        return df if df is not None else pd.DataFrame()

    try:
        logger.debug(
            f"Scrying (Ehlers SSF MA): Calculating Fast SSF({fast_len}, poles={poles}), Slow SSF({slow_len}, poles={poles})"
        )
        # pandas_ta.ssf returns a Series, ensure it's assigned correctly
        df["ehlers_ssf_fast"] = df.ta.ssf(
            length=fast_len, poles=poles, append=False
        ).apply(safe_decimal_conversion)
        df["ehlers_ssf_slow"] = df.ta.ssf(
            length=slow_len, poles=poles, append=False
        ).apply(safe_decimal_conversion)

        if not df.empty:
            fast_val = df["ehlers_ssf_fast"].iloc[-1]
            slow_val = df["ehlers_ssf_slow"].iloc[-1]
            if pd.notna(fast_val) and pd.notna(slow_val):
                logger.debug(
                    f"Scrying (Ehlers SSF MA({fast_len},{slow_len},p{poles})): Fast={Fore.GREEN}{fast_val:.4f}{Style.RESET_ALL}, Slow={Fore.RED}{slow_val:.4f}{Style.RESET_ALL}"
                )
            else:
                logger.debug("Scrying (Ehlers SSF MA): NA values on last candle.")
    except Exception as e:
        logger.error(
            f"{Fore.RED}Scrying (Ehlers SSF MA): Unexpected error: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        for col in target_cols:
            df[col] = pd.NA
    return df


def analyze_order_book(
    exchange: ccxt.Exchange, symbol: str, depth: int, fetch_limit: int
) -> dict[str, Decimal | None]:
    """Fetches and analyzes L2 order book pressure (Bid/Ask volume ratio) and spread."""
    results: dict[str, Decimal | None] = {
        "bid_ask_ratio": None,
        "spread": None,
        "best_bid": None,
        "best_ask": None,
    }
    logger.debug(
        f"Order Book Scrying: Fetching L2 {symbol} (Depth:{depth}, Limit:{fetch_limit})..."
    )

    if not exchange.has.get("fetchL2OrderBook"):
        logger.warning(
            f"{Fore.YELLOW}Order Book Scrying: fetchL2OrderBook method not supported by {exchange.id}. Cannot analyze depth.{Style.RESET_ALL}"
        )
        return results

    try:
        order_book = exchange.fetch_l2_order_book(symbol, limit=fetch_limit)
        bids: list[list[float | str]] = order_book.get("bids", [])
        asks: list[list[float | str]] = order_book.get("asks", [])

        if not bids or not asks:
            logger.warning(
                f"{Fore.YELLOW}Order Book Scrying: Empty bids or asks received for {symbol}. Cannot analyze.{Style.RESET_ALL}"
            )
            return results

        best_bid = (
            safe_decimal_conversion(bids[0][0]) if bids and len(bids[0]) > 0 else None
        )
        best_ask = (
            safe_decimal_conversion(asks[0][0]) if asks and len(asks[0]) > 0 else None
        )
        results["best_bid"] = best_bid
        results["best_ask"] = best_ask

        if (
            best_bid is not None
            and best_ask is not None
            and best_bid > 0
            and best_ask > 0
        ):
            results["spread"] = best_ask - best_bid
            logger.debug(
                f"OB Scrying: Best Bid={Fore.GREEN}{_format_for_log(best_bid, 4)}{Style.RESET_ALL}, Best Ask={Fore.RED}{_format_for_log(best_ask, 4)}{Style.RESET_ALL}, Spread={Fore.YELLOW}{_format_for_log(results['spread'], 4)}{Style.RESET_ALL}"
            )
        else:
            logger.debug(
                f"OB Scrying: Best Bid={_format_for_log(best_bid)}, Best Ask={_format_for_log(best_ask)} (Spread calculation skipped)"
            )

        bid_vol = sum(
            safe_decimal_conversion(bid[1])
            for bid in bids[: min(depth, len(bids))]
            if len(bid) > 1
        )
        ask_vol = sum(
            safe_decimal_conversion(ask[1])
            for ask in asks[: min(depth, len(asks))]
            if len(ask) > 1
        )
        logger.debug(
            f"OB Scrying (Depth {depth}): Total BidVol={Fore.GREEN}{bid_vol:.4f}{Style.RESET_ALL}, Total AskVol={Fore.RED}{ask_vol:.4f}{Style.RESET_ALL}"
        )

        if ask_vol > CONFIG.position_qty_epsilon:
            try:
                results["bid_ask_ratio"] = bid_vol / ask_vol
                ratio_color = (
                    Fore.GREEN
                    if results["bid_ask_ratio"]
                    >= CONFIG.order_book_ratio_threshold_long
                    else (
                        Fore.RED
                        if results["bid_ask_ratio"]
                        <= CONFIG.order_book_ratio_threshold_short
                        else Fore.YELLOW
                    )
                )
                logger.debug(
                    f"OB Scrying Ratio (Bids/Asks): {ratio_color}{results['bid_ask_ratio']:.3f}{Style.RESET_ALL}"
                )
            except (DivisionByZero, Exception) as e:
                logger.warning(
                    f"{Fore.YELLOW}Error calculating OB ratio: {e}{Style.RESET_ALL}"
                )
                results["bid_ask_ratio"] = None
        else:
            logger.debug(f"OB Scrying Ratio: N/A (Ask volume near zero: {ask_vol:.4f})")
            results["bid_ask_ratio"] = None

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger.warning(
            f"{Fore.YELLOW}Order Book Scrying Error for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
    except IndexError:
        logger.warning(
            f"{Fore.YELLOW}Order Book Scrying Error: Index out of bounds for {symbol}. Malformed OB data?{Style.RESET_ALL}"
        )
    except Exception as e:
        logger.warning(
            f"{Fore.YELLOW}Unexpected Order Book Scrying Error for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())

    results.setdefault("bid_ask_ratio", None)
    results.setdefault("spread", None)
    results.setdefault("best_bid", None)
    results.setdefault("best_ask", None)
    return results


# --- Data Fetching - Gathering Etheric Data Streams ---
def get_market_data(
    exchange: ccxt.Exchange, symbol: str, interval: str, limit: int
) -> pd.DataFrame | None:
    """Fetches and prepares OHLCV data as a pandas DataFrame, ensuring numeric types and handling NaNs."""
    if not exchange.has.get("fetchOHLCV"):
        logger.error(
            f"{Fore.RED}Data Fetch: Exchange '{exchange.id}' does not support fetchOHLCV method.{Style.RESET_ALL}"
        )
        return None
    try:
        logger.debug(
            f"Data Fetch: Gathering {limit} OHLCV candles for {symbol} ({interval})..."
        )
        ohlcv: list[list[int | float | str]] = exchange.fetch_ohlcv(
            symbol, timeframe=interval, limit=limit
        )

        if not ohlcv:
            logger.warning(
                f"{Fore.YELLOW}Data Fetch: No OHLCV data returned for {symbol} ({interval}). Market might be inactive or API issue.{Style.RESET_ALL}"
            )
            return None

        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        initial_nan_count = df.isnull().sum().sum()
        if initial_nan_count > 0:
            nan_counts_per_col = df.isnull().sum()
            logger.warning(
                f"{Fore.YELLOW}Data Fetch: Found {initial_nan_count} NaN values in OHLCV data after conversion:\n"
                f"{nan_counts_per_col[nan_counts_per_col > 0]}\nAttempting forward fill (ffill)...{Style.RESET_ALL}"
            )
            df.ffill(inplace=True)

            remaining_nan_count = df.isnull().sum().sum()
            if remaining_nan_count > 0:
                logger.warning(
                    f"{Fore.YELLOW}NaNs remain after ffill ({remaining_nan_count}). Attempting backward fill (bfill)...{Style.RESET_ALL}"
                )
                df.bfill(inplace=True)

                final_nan_count = df.isnull().sum().sum()
                if final_nan_count > 0:
                    logger.error(
                        f"{Fore.RED}Data Fetch: Unfillable NaN values ({final_nan_count}) remain after ffill and bfill. "
                        f"Data quality insufficient for {symbol}. Skipping cycle.{Style.RESET_ALL}"
                    )
                    return None

        logger.debug(
            f"Data Fetch: Successfully woven and cleaned {len(df)} OHLCV candles for {symbol}."
        )
        return df

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger.warning(
            f"{Fore.YELLOW}Data Fetch: Disturbance gathering OHLCV for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
    except Exception as e:
        logger.error(
            f"{Fore.RED}Data Fetch: Unexpected error processing OHLCV for {symbol}: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())

    return None


# --- Position & Order Management - Manipulating Market Presence ---
def get_current_position(exchange: ccxt.Exchange, symbol: str) -> dict[str, Any]:
    """Fetches current position details using Bybit V5 API specifics (fetchPositions).
    Assumes One-Way Mode. Returns position side ('Long', 'Short', 'None'), quantity (Decimal), and entry price (Decimal).
    """
    default_pos: dict[str, Any] = {
        "side": CONFIG.pos_none,
        "qty": Decimal("0.0"),
        "entry_price": Decimal("0.0"),
    }
    market_id: str | None = None
    market: dict[str, Any] | None = None

    try:
        market = exchange.market(symbol)
        market_id = market["id"]
        category = (
            "linear"
            if market.get("linear")
            else ("inverse" if market.get("inverse") else None)
        )
        if not category:
            logger.error(
                f"{Fore.RED}Position Check: Could not determine category for market '{symbol}'.{Style.RESET_ALL}"
            )
            return default_pos
    except (ccxt.BadSymbol, KeyError) as e:
        logger.error(
            f"{Fore.RED}Position Check: Failed to identify market structure for '{symbol}': {e}{Style.RESET_ALL}"
        )
        return default_pos
    except Exception as e_market:
        logger.error(
            f"{Fore.RED}Position Check: Unexpected error getting market info for '{symbol}': {e_market}{Style.RESET_ALL}"
        )
        return default_pos

    try:
        if not exchange.has.get("fetchPositions"):
            logger.warning(
                f"{Fore.YELLOW}Position Check: Exchange '{exchange.id}' does not support fetchPositions method.{Style.RESET_ALL}"
            )
            return default_pos

        params = {"category": category, "symbol": market_id}
        logger.debug(
            f"Position Check: Querying V5 positions for {symbol} (MarketID: {market_id}, Category: {category})..."
        )
        fetched_positions = exchange.fetch_positions(symbols=[symbol], params=params)

        active_pos_info = None
        for pos in fetched_positions:
            pos_info = pos.get("info", {})
            pos_market_id_from_api = pos_info.get("symbol")
            position_idx = int(pos_info.get("positionIdx", -1))
            pos_side_v5 = pos_info.get("side", "None")
            size_str = pos_info.get("size", "0")

            if pos_market_id_from_api == market_id and position_idx == 0:
                size_dec = safe_decimal_conversion(size_str)
                if (
                    abs(size_dec) > CONFIG.position_qty_epsilon
                    and pos_side_v5 != "None"
                ):
                    active_pos_info = pos_info
                    logger.debug(
                        f"Found active V5 position candidate: Idx={position_idx}, Side={pos_side_v5}, Size={size_str}"
                    )
                    break

        if active_pos_info:
            try:
                size = safe_decimal_conversion(active_pos_info.get("size"))
                entry_price = safe_decimal_conversion(active_pos_info.get("avgPrice"))
                side = (
                    CONFIG.pos_long
                    if active_pos_info.get("side") == "Buy"
                    else CONFIG.pos_short
                )

                if size <= CONFIG.position_qty_epsilon or entry_price <= 0:
                    logger.warning(
                        f"{Fore.YELLOW}Position Check: Found active V5 pos with invalid size/entry: Size={size}, Entry={entry_price}. Treating as flat.{Style.RESET_ALL}"
                    )
                    return default_pos

                pos_color = Fore.GREEN if side == CONFIG.pos_long else Fore.RED
                logger.info(
                    f"{pos_color}Position Check: Found ACTIVE {side} position: Qty={size:.8f} @ Entry={entry_price:.4f}{Style.RESET_ALL}"
                )
                return {"side": side, "qty": size, "entry_price": entry_price}
            except Exception as parse_err:
                logger.warning(
                    f"{Fore.YELLOW}Position Check: Error parsing active V5 position data: {parse_err}. Data: {active_pos_info}{Style.RESET_ALL}"
                )
                return default_pos
        else:
            logger.info(
                f"{Fore.BLUE}Position Check: No active One-Way (positionIdx=0) position found for {market_id}. Currently Flat.{Style.RESET_ALL}"
            )
            return default_pos

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger.warning(
            f"{Fore.YELLOW}Position Check: Disturbance querying V5 positions for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
    except Exception as e_pos:
        logger.error(
            f"{Fore.RED}Position Check: Unexpected error querying V5 positions for {symbol}: {e_pos}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())

    return default_pos


def set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int) -> bool:
    """Sets leverage for a futures symbol using Bybit V5 API specifics."""
    logger.info(
        f"{Fore.CYAN}Leverage Conjuring: Attempting to set {leverage}x for {symbol}...{Style.RESET_ALL}"
    )
    try:
        market = exchange.market(symbol)
        if not market.get("contract"):
            logger.error(
                f"{Fore.RED}Leverage Conjuring Error: Cannot set leverage for non-contract market: {symbol}.{Style.RESET_ALL}"
            )
            return False
        params = {"buyLeverage": str(leverage), "sellLeverage": str(leverage)}
        logger.debug(f"Using V5 params for set_leverage: {params}")
    except (ccxt.BadSymbol, KeyError) as e:
        logger.error(
            f"{Fore.RED}Leverage Conjuring Error: Failed to identify market structure for '{symbol}': {e}{Style.RESET_ALL}"
        )
        return False
    except Exception as e_market:
        logger.error(
            f"{Fore.RED}Leverage Conjuring Error: Unexpected error getting market info for '{symbol}': {e_market}{Style.RESET_ALL}"
        )
        return False

    for attempt in range(CONFIG.retry_count + 1):
        try:
            response = exchange.set_leverage(
                leverage=leverage, symbol=symbol, params=params
            )
            logger.success(
                f"{Fore.GREEN}Leverage Conjuring: Successfully set to {leverage}x for {symbol}. Response: {response}{Style.RESET_ALL}"
            )
            return True
        except ccxt.ExchangeError as e:
            err_str = str(e).lower()
            if (
                "leverage not modified" in err_str
                or "same leverage" in err_str
                or "110044" in err_str
            ):
                logger.info(
                    f"{Fore.CYAN}Leverage Conjuring: Leverage already set to {leverage}x for {symbol}.{Style.RESET_ALL}"
                )
                return True
            elif "cannot be lower than 1" in err_str and leverage < 1:
                logger.error(
                    f"{Fore.RED}Leverage Conjuring: Invalid leverage value ({leverage}) requested.{Style.RESET_ALL}"
                )
                return False

            logger.warning(
                f"{Fore.YELLOW}Leverage Conjuring: Exchange resistance (Attempt {attempt + 1}/{CONFIG.retry_count + 1}): {e}{Style.RESET_ALL}"
            )
            if attempt >= CONFIG.retry_count:
                logger.error(
                    f"{Fore.RED}Leverage Conjuring: Failed after {CONFIG.retry_count + 1} attempts.{Style.RESET_ALL}"
                )
                break
            time.sleep(CONFIG.retry_delay_seconds * (attempt + 1))
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            logger.warning(
                f"{Fore.YELLOW}Leverage Conjuring: Network/Timeout disturbance (Attempt {attempt + 1}/{CONFIG.retry_count + 1}): {e}{Style.RESET_ALL}"
            )
            if attempt >= CONFIG.retry_count:
                logger.error(
                    f"{Fore.RED}Leverage Conjuring: Failed due to network issues after {CONFIG.retry_count + 1} attempts.{Style.RESET_ALL}"
                )
                break
            time.sleep(CONFIG.retry_delay_seconds * (attempt + 1))
        except Exception as e_unexp:
            logger.error(
                f"{Fore.RED}Leverage Conjuring: Unexpected error (Attempt {attempt + 1}/{CONFIG.retry_count + 1}): {e_unexp}{Style.RESET_ALL}"
            )
            logger.debug(traceback.format_exc())
            if attempt >= CONFIG.retry_count:
                logger.error(
                    f"{Fore.RED}Leverage Conjuring: Failed due to unexpected error after {CONFIG.retry_count + 1} attempts.{Style.RESET_ALL}"
                )
                break
            time.sleep(CONFIG.retry_delay_seconds * (attempt + 1))
    return False


def close_position(
    exchange: ccxt.Exchange,
    symbol: str,
    position_to_close: dict[str, Any],
    reason: str = "Signal",
) -> dict[str, Any] | None:
    """Closes the specified active position by placing a market order with reduceOnly=True.
    Re-validates the position just before closing.
    """
    initial_side = position_to_close.get("side", CONFIG.pos_none)
    initial_qty = position_to_close.get("qty", Decimal("0.0"))
    market_base = symbol.split("/")[0].split(":")[0]
    logger.info(
        f"{Fore.YELLOW}Banish Position Ritual: Initiated for {symbol}. Reason: {reason}. Initial state: {initial_side} Qty={initial_qty:.8f}{Style.RESET_ALL}"
    )

    logger.debug("Banish Position: Re-validating live position state...")
    live_position = get_current_position(exchange, symbol)
    live_position_side = live_position["side"]
    live_amount_to_close = live_position["qty"]

    if (
        live_position_side == CONFIG.pos_none
        or live_amount_to_close <= CONFIG.position_qty_epsilon
    ):
        logger.warning(
            f"{Fore.YELLOW}Banish Position: Re-validation shows NO active position (or negligible size: {live_amount_to_close:.8f}) for {symbol}. Aborting banishment.{Style.RESET_ALL}"
        )
        if initial_side != CONFIG.pos_none:
            logger.warning(
                f"{Fore.YELLOW}Banish Position: Discrepancy detected! Initial check showed {initial_side}, but now it's None/Zero.{Style.RESET_ALL}"
            )
        return None

    side_to_execute_close = (
        CONFIG.side_sell if live_position_side == CONFIG.pos_long else CONFIG.side_buy
    )

    try:
        amount_str = format_amount(exchange, symbol, live_amount_to_close)
        amount_dec = Decimal(amount_str)
        amount_float = float(amount_dec)

        if amount_dec <= CONFIG.position_qty_epsilon:
            logger.error(
                f"{Fore.RED}Banish Position: Closing amount negligible ({amount_str}) after precision shaping. Cannot close. Manual check advised.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] ERROR Banishing ({reason}): Negligible close amount {amount_str}. MANUAL CHECK!"
            )
            return None

        close_color = Back.YELLOW
        logger.warning(
            f"{close_color}{Fore.BLACK}{Style.BRIGHT}Banish Position: Attempting to CLOSE {live_position_side} ({reason}): Exec {side_to_execute_close.upper()} MARKET {amount_str} {symbol} (reduce_only=True)...{Style.RESET_ALL}"
        )
        params = {"reduceOnly": True}
        order = exchange.create_market_order(
            symbol=symbol,
            side=side_to_execute_close,
            amount=amount_float,
            params=params,
        )

        fill_price_avg = safe_decimal_conversion(order.get("average"))
        filled_qty = safe_decimal_conversion(order.get("filled"))
        cost = safe_decimal_conversion(order.get("cost"))
        order_id_short = format_order_id(order.get("id"))
        status = order.get("status", "unknown")

        if abs(filled_qty - amount_dec) < CONFIG.position_qty_epsilon:
            logger.success(
                f"{Fore.GREEN}{Style.BRIGHT}Banish Position: Order ({reason}) appears FILLED for {symbol}. Filled: {filled_qty:.8f}, AvgFill: {fill_price_avg:.4f}, Cost: {cost:.2f} USDT. ID:...{order_id_short}, Status: {status}{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] BANISHED {live_position_side} {amount_str} @ ~{fill_price_avg:.4f} ({reason}). ID:...{order_id_short}"
            )
            return order
        else:
            logger.warning(
                f"{Fore.YELLOW}Banish Position: Order ({reason}) status uncertain. Expected {amount_dec:.8f}, Filled: {filled_qty:.8f}. ID:...{order_id_short}, Status: {status}. Re-checking position state soon.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] WARNING Banishing ({reason}): Fill uncertain (Exp:{amount_dec:.8f}, Got:{filled_qty:.8f}). ID:...{order_id_short}"
            )
            return order

    except ccxt.InsufficientFunds as e:
        logger.error(
            f"{Fore.RED}Banish Position ({reason}): Insufficient funds for {symbol}: {e}. This might indicate margin issues or incorrect state.{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[{market_base}/{CONFIG.strategy_name}] ERROR Banishing ({reason}): Insufficient Funds! Check account."
        )
    except ccxt.ExchangeError as e:
        err_str = str(e).lower()
        if (
            "order quantity exceeds open position size" in err_str
            or "position is zero" in err_str
            or "position size is zero" in err_str
            or "order would not reduce position size" in err_str
            or "110007" in err_str
            or "110015" in err_str
        ):
            logger.warning(
                f"{Fore.YELLOW}Banish Position: Exchange indicates position likely already closed/closing or zero size ({e}). Assuming banished.{Style.RESET_ALL}"
            )
            return None
        else:
            logger.error(
                f"{Fore.RED}Banish Position ({reason}): Unhandled Exchange Error for {symbol}: {e}{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] ERROR Banishing ({reason}): Exchange Error: {e}. Check logs."
            )
    except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
        logger.error(
            f"{Fore.RED}Banish Position ({reason}): Network/Timeout Error for {symbol}: {e}. Position state uncertain.{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[{market_base}/{CONFIG.strategy_name}] ERROR Banishing ({reason}): Network/Timeout Error. Check position manually."
        )
    except ValueError as e:
        logger.error(
            f"{Fore.RED}Banish Position ({reason}): Value Error (likely formatting/conversion) for {symbol}: {e}{Style.RESET_ALL}"
        )
    except Exception as e:
        logger.error(
            f"{Fore.RED}Banish Position ({reason}): Unexpected Error for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        send_sms_alert(
            f"[{market_base}/{CONFIG.strategy_name}] ERROR Banishing ({reason}): Unexpected Error {type(e).__name__}. Check logs."
        )

    return None


def calculate_position_size(
    equity: Decimal,
    risk_per_trade_pct: Decimal,
    entry_price: Decimal,
    stop_loss_price: Decimal,
    leverage: int,
    symbol: str,
    exchange: ccxt.Exchange,
) -> tuple[Decimal | None, Decimal | None]:
    """Calculates position size based on risk percentage, entry/SL prices, and leverage.
    Returns (position_quantity, estimated_margin_required) as Decimals, or (None, None) on failure.
    """
    logger.debug(
        f"Risk Calc: Equity={equity:.4f}, Risk%={risk_per_trade_pct:.4%}, Entry={entry_price:.4f}, SL={stop_loss_price:.4f}, Lev={leverage}x"
    )

    if not (entry_price > 0 and stop_loss_price > 0):
        logger.error(
            f"{Fore.RED}Risk Calc Error: Invalid entry/SL price (<= 0). Entry={entry_price}, SL={stop_loss_price}.{Style.RESET_ALL}"
        )
        return None, None
    price_diff = abs(entry_price - stop_loss_price)
    if price_diff < CONFIG.position_qty_epsilon:
        logger.error(
            f"{Fore.RED}Risk Calc Error: Entry and SL prices are too close ({price_diff:.8f}). Cannot calculate safe size.{Style.RESET_ALL}"
        )
        return None, None
    if not 0 < risk_per_trade_pct < 1:
        logger.error(
            f"{Fore.RED}Risk Calc Error: Invalid risk percentage: {risk_per_trade_pct:.4%}. Must be between 0 and 1.{Style.RESET_ALL}"
        )
        return None, None
    if equity <= 0:
        logger.error(
            f"{Fore.RED}Risk Calc Error: Invalid equity: {equity:.4f}. Cannot calculate risk.{Style.RESET_ALL}"
        )
        return None, None
    if leverage <= 0:
        logger.error(
            f"{Fore.RED}Risk Calc Error: Invalid leverage: {leverage}. Must be > 0.{Style.RESET_ALL}"
        )
        return None, None

    try:
        risk_amount_usdt = equity * risk_per_trade_pct
        quantity_raw = risk_amount_usdt / price_diff
        logger.debug(
            f"Risk Calc: RiskAmt={risk_amount_usdt:.4f} USDT, PriceDiff={price_diff:.4f} USDT, Raw Qty={quantity_raw:.8f}"
        )
    except DivisionByZero:
        logger.error(
            f"{Fore.RED}Risk Calc Error: Division by zero calculating raw quantity (price_diff is zero?).{Style.RESET_ALL}"
        )
        return None, None
    except Exception as calc_err:
        logger.error(
            f"{Fore.RED}Risk Calc Error: Unexpected error during raw quantity calculation: {calc_err}{Style.RESET_ALL}"
        )
        return None, None

    try:
        quantity_precise_str = format_amount(exchange, symbol, quantity_raw)
        quantity_precise = Decimal(quantity_precise_str)
        logger.debug(f"Risk Calc: Precise Qty after formatting={quantity_precise:.8f}")
    except Exception as fmt_err:
        logger.warning(
            f"{Fore.YELLOW}Risk Calc Warning: Failed precision shaping for quantity {quantity_raw:.8f}. Error: {fmt_err}. Using raw value with fallback quantization.{Style.RESET_ALL}"
        )
        try:
            quantity_precise = quantity_raw.quantize(
                Decimal("1e-8"), rounding=ROUND_HALF_UP
            )
            logger.debug(f"Risk Calc: Fallback Quantized Qty={quantity_precise:.8f}")
        except Exception as q_err:
            logger.error(
                f"{Fore.RED}Risk Calc Error: Failed fallback quantization for quantity {quantity_raw:.8f}: {q_err}{Style.RESET_ALL}"
            )
            return None, None

    if quantity_precise <= CONFIG.position_qty_epsilon:
        logger.warning(
            f"{Fore.YELLOW}Risk Calc Warning: Calculated quantity negligible ({quantity_precise:.8f}). Cannot place order.{Style.RESET_ALL}"
        )
        return None, None

    try:
        pos_value_usdt = quantity_precise * entry_price
        required_margin = pos_value_usdt / Decimal(leverage)
        logger.debug(
            f"Risk Calc Result: Qty={Fore.CYAN}{quantity_precise:.8f}{Style.RESET_ALL}, Est. Pos Value={pos_value_usdt:.4f} USDT, Est. Margin Req.={required_margin:.4f} USDT"
        )
        return quantity_precise, required_margin
    except (DivisionByZero, Exception) as margin_err:
        logger.error(
            f"{Fore.RED}Risk Calc Error: Failed calculating estimated margin: {margin_err}{Style.RESET_ALL}"
        )
        return None, None


def wait_for_order_fill(
    exchange: ccxt.Exchange, order_id: str, symbol: str, timeout_seconds: int
) -> dict[str, Any] | None:
    """Waits for a specific order ID to reach a 'closed' (filled for market) or failed status.
    Returns the final order dict if filled/failed, None if timed out or error occurred during check.
    """
    start_time = time.time()
    order_id_short = format_order_id(order_id)
    logger.info(
        f"{Fore.CYAN}Observing order ...{order_id_short} ({symbol}) for fill/final status (Timeout: {timeout_seconds}s)...{Style.RESET_ALL}"
    )

    while time.time() - start_time < timeout_seconds:
        try:
            order = exchange.fetch_order(order_id, symbol)
            status = order.get("status")
            filled_qty = safe_decimal_conversion(order.get("filled"))
            logger.debug(
                f"Order ...{order_id_short} status: {status}, Filled: {filled_qty:.8f}"
            )

            if status == "closed":
                logger.success(
                    f"{Fore.GREEN}Order ...{order_id_short} confirmed FILLED/CLOSED.{Style.RESET_ALL}"
                )
                return order
            elif status in ["canceled", "rejected", "expired"]:
                logger.error(
                    f"{Fore.RED}Order ...{order_id_short} reached FAILED status: '{status}'.{Style.RESET_ALL}"
                )
                return order
            time.sleep(0.75)
        except ccxt.OrderNotFound:
            elapsed = time.time() - start_time
            logger.warning(
                f"{Fore.YELLOW}Order ...{order_id_short} not found yet ({elapsed:.1f}s elapsed). Retrying...{Style.RESET_ALL}"
            )
            time.sleep(1.5)
        except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
            elapsed = time.time() - start_time
            logger.warning(
                f"{Fore.YELLOW}Disturbance checking order ...{order_id_short} ({elapsed:.1f}s elapsed): {type(e).__name__} - {e}. Retrying...{Style.RESET_ALL}"
            )
            time.sleep(2)
        except Exception as e_unexp:
            elapsed = time.time() - start_time
            logger.error(
                f"{Fore.RED}Unexpected error checking order ...{order_id_short} ({elapsed:.1f}s elapsed): {e_unexp}. Stopping check.{Style.RESET_ALL}"
            )
            logger.debug(traceback.format_exc())
            return None

    logger.error(
        f"{Fore.RED}Order ...{order_id_short} did NOT reach final status within {timeout_seconds}s timeout.{Style.RESET_ALL}"
    )
    try:
        final_check_order = exchange.fetch_order(order_id, symbol)
        logger.warning(
            f"Final status check for timed-out order ...{order_id_short}: {final_check_order.get('status')}"
        )
        return final_check_order
    except Exception as final_e:
        logger.error(
            f"Failed final status check for timed-out order ...{order_id_short}: {final_e}"
        )
        return None


def place_risked_market_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    risk_percentage: Decimal,
    current_atr: Decimal | None,
    sl_atr_multiplier: Decimal,
    leverage: int,
    max_order_cap_usdt: Decimal,
    margin_check_buffer: Decimal,
    tsl_percent: Decimal,
    tsl_activation_offset_percent: Decimal,
) -> dict[str, Any] | None:
    """Core order placement logic."""
    market_base = symbol.split("/")[0].split(":")[0]
    order_side_color = Fore.GREEN if side == CONFIG.side_buy else Fore.RED
    logger.info(
        f"{order_side_color}{Style.BRIGHT}Place Order Ritual: Initiating {side.upper()} for {symbol}...{Style.RESET_ALL}"
    )

    if current_atr is None or current_atr <= Decimal("0"):
        logger.error(
            f"{Fore.RED}Place Order Error ({side.upper()}): Invalid ATR ({current_atr}). Cannot calculate SL.{Style.RESET_ALL}"
        )
        return None

    entry_price_estimate: Decimal | None = None
    initial_sl_price_estimate: Decimal | None = None
    final_quantity: Decimal | None = None
    market: dict | None = None
    min_qty: Decimal | None = None
    min_price: Decimal | None = None

    try:
        logger.debug("Gathering resources: Balance, Market Structure, Limits...")
        balance_params = {"category": "linear"}
        balance = exchange.fetch_balance(params=balance_params)
        market = exchange.market(symbol)
        limits = market.get("limits", {})
        amount_limits = limits.get("amount", {})
        price_limits = limits.get("price", {})
        min_qty_str = amount_limits.get("min")
        max_qty_str = amount_limits.get("max")
        min_price_str = price_limits.get("min")
        min_qty = safe_decimal_conversion(min_qty_str) if min_qty_str else None
        max_qty = safe_decimal_conversion(max_qty_str) if max_qty_str else None
        min_price = safe_decimal_conversion(min_price_str) if min_price_str else None

        usdt_balance = balance.get(CONFIG.usdt_symbol, {})
        usdt_total = safe_decimal_conversion(usdt_balance.get("total"))
        usdt_free = safe_decimal_conversion(usdt_balance.get("free"))
        usdt_equity = (
            usdt_total if usdt_total is not None and usdt_total > 0 else usdt_free
        )

        if usdt_equity is None or usdt_equity <= Decimal("0"):
            logger.error(
                f"{Fore.RED}Place Order Error ({side.upper()}): Zero or Invalid Equity ({usdt_equity}).{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Zero/Invalid Equity ({usdt_equity:.2f})"
            )
            return None
        if usdt_free is None or usdt_free < Decimal("0"):
            logger.error(
                f"{Fore.RED}Place Order Error ({side.upper()}): Invalid Free Margin ({usdt_free}).{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Invalid Free Margin ({usdt_free:.2f})"
            )
            return None
        logger.debug(
            f"Resources: Equity={usdt_equity:.4f}, Free Margin={usdt_free:.4f} {CONFIG.usdt_symbol}"
        )
        if min_qty:
            logger.debug(f"Market Limits: Min Qty={min_qty:.8f}")
        if max_qty:
            logger.debug(f"Market Limits: Max Qty={max_qty:.8f}")
        if min_price:
            logger.debug(f"Market Limits: Min Price={min_price:.8f}")

        logger.debug("Estimating entry price using shallow order book...")
        ob_data = analyze_order_book(
            exchange,
            symbol,
            CONFIG.shallow_ob_fetch_depth,
            CONFIG.shallow_ob_fetch_depth,
        )
        best_ask = ob_data.get("best_ask")
        best_bid = ob_data.get("best_bid")
        if side == CONFIG.side_buy and best_ask and best_ask > 0:
            entry_price_estimate = best_ask
        elif side == CONFIG.side_sell and best_bid and best_bid > 0:
            entry_price_estimate = best_bid
        else:
            logger.warning(
                f"{Fore.YELLOW}Shallow OB failed for entry price estimate. Fetching ticker...{Style.RESET_ALL}"
            )
            try:
                ticker = exchange.fetch_ticker(symbol)
                last_price = safe_decimal_conversion(ticker.get("last"))
                if last_price > 0:
                    entry_price_estimate = last_price
                else:
                    raise ValueError("Ticker price invalid")
            except Exception as e:
                logger.error(
                    f"{Fore.RED}Place Order Error ({side.upper()}): Failed to get entry price estimate: {e}{Style.RESET_ALL}"
                )
                return None
        logger.debug(f"Estimated Entry Price ~ {entry_price_estimate:.4f}")

        sl_distance = current_atr * sl_atr_multiplier
        initial_sl_price_raw = (
            (entry_price_estimate - sl_distance)
            if side == CONFIG.side_buy
            else (entry_price_estimate + sl_distance)
        )

        if initial_sl_price_raw <= 0:
            logger.error(
                f"{Fore.RED}Place Order Error ({side.upper()}): Initial SL price calc invalid ({initial_sl_price_raw:.4f}).{Style.RESET_ALL}"
            )
            return None
        if min_price is not None and initial_sl_price_raw < min_price:
            initial_sl_price_raw = min_price
        try:
            initial_sl_price_estimate_str = format_price(
                exchange, symbol, initial_sl_price_raw
            )
            initial_sl_price_estimate = Decimal(initial_sl_price_estimate_str)
            if initial_sl_price_estimate <= 0:
                raise ValueError("Formatted SL estimate invalid")
            logger.info(
                f"Calculated Initial SL Price (Estimate) ~ {Fore.YELLOW}{initial_sl_price_estimate:.4f}{Style.RESET_ALL}"
            )
        except Exception as e:
            logger.error(
                f"{Fore.RED}Place Order Error ({side.upper()}): Failed to format initial SL estimate: {e}{Style.RESET_ALL}"
            )
            return None

        calc_qty, req_margin_est = calculate_position_size(
            usdt_equity,
            risk_percentage,
            entry_price_estimate,
            initial_sl_price_estimate,
            leverage,
            symbol,
            exchange,
        )
        if calc_qty is None or req_margin_est is None:
            logger.error(
                f"{Fore.RED}Place Order Error ({side.upper()}): Failed risk calculation.{Style.RESET_ALL}"
            )
            return None
        final_quantity = calc_qty

        pos_value_estimate = final_quantity * entry_price_estimate
        if pos_value_estimate > max_order_cap_usdt:
            logger.warning(
                f"{Fore.YELLOW}Calculated order value {pos_value_estimate:.4f} exceeds Max Cap {max_order_cap_usdt:.4f}. Capping.{Style.RESET_ALL}"
            )
            try:
                final_quantity_capped = max_order_cap_usdt / entry_price_estimate
                final_quantity_str = format_amount(
                    exchange, symbol, final_quantity_capped
                )
                final_quantity = Decimal(final_quantity_str)
                req_margin_est = (
                    final_quantity * entry_price_estimate / Decimal(leverage)
                )
                logger.info(
                    f"Quantity capped to: {final_quantity:.8f}, New Est. Margin: {req_margin_est:.4f} USDT"
                )
            except Exception as cap_err:
                logger.error(
                    f"{Fore.RED}Place Order Error ({side.upper()}): Failed to cap quantity: {cap_err}{Style.RESET_ALL}"
                )
                return None

        if final_quantity <= CONFIG.position_qty_epsilon:
            logger.error(
                f"{Fore.RED}Place Order Error ({side.upper()}): Final Quantity negligible ({final_quantity:.8f}).{Style.RESET_ALL}"
            )
            return None
        if min_qty is not None and final_quantity < min_qty:
            logger.error(
                f"{Fore.RED}Place Order Error ({side.upper()}): Qty {final_quantity:.8f} < Min {min_qty:.8f}.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Qty {final_quantity:.8f} < Min {min_qty:.8f}"
            )
            return None
        if max_qty is not None and final_quantity > max_qty:
            logger.warning(
                f"{Fore.YELLOW}Qty {final_quantity:.8f} > Max {max_qty:.8f}. Adjusting to max.{Style.RESET_ALL}"
            )
            final_quantity = max_qty
            try:
                final_quantity_str = format_amount(exchange, symbol, final_quantity)
                final_quantity = Decimal(final_quantity_str)
                req_margin_est = (
                    final_quantity * entry_price_estimate / Decimal(leverage)
                )
            except Exception as max_fmt_err:
                logger.error(
                    f"{Fore.RED}Place Order Error ({side.upper()}): Failed format max-capped qty: {max_fmt_err}{Style.RESET_ALL}"
                )
                return None

        req_margin_buffered = req_margin_est * margin_check_buffer
        logger.debug(
            f"Final Margin Check: Need ~{req_margin_est:.4f} (Buffered: {req_margin_buffered:.4f}), Have Free: {usdt_free:.4f}"
        )
        if usdt_free < req_margin_buffered:
            logger.error(
                f"{Fore.RED}Place Order Error ({side.upper()}): Insufficient FREE margin. Need ~{req_margin_buffered:.4f}, Have {usdt_free:.4f}{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Insufficient Free Margin (Need ~{req_margin_buffered:.2f}, Have {usdt_free:.2f})"
            )
            return None
        logger.info(
            f"{Fore.GREEN}Final Order Details Pre-Submission: Side={side.upper()}, Qty={final_quantity:.8f}, EstValue={final_quantity * entry_price_estimate:.4f}, EstMargin={req_margin_est:.4f}. Margin OK.{Style.RESET_ALL}"
        )

        entry_order: dict[str, Any] | None = None
        order_id: str | None = None
        try:
            qty_float = float(final_quantity)
            entry_bg_color = Back.GREEN if side == CONFIG.side_buy else Back.RED
            entry_fg_color = Fore.BLACK if side == CONFIG.side_buy else Fore.WHITE
            logger.warning(
                f"{entry_bg_color}{entry_fg_color}{Style.BRIGHT}*** Placing {side.upper()} MARKET ENTRY Order: {qty_float:.8f} {symbol} ***{Style.RESET_ALL}"
            )
            entry_order = exchange.create_market_order(
                symbol=symbol, side=side, amount=qty_float, params={"reduceOnly": False}
            )
            order_id = entry_order.get("id")
            if not order_id:
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}CRITICAL FAILURE: Entry order NO Order ID! MANUAL INTERVENTION! Response: {entry_order}{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"[{market_base}/{CONFIG.strategy_name}] CRITICAL ORDER FAIL ({side.upper()}): Entry NO ID! MANUAL CHECK!"
                )
                return None
            logger.success(
                f"{Fore.GREEN}Market Entry Order submitted. ID: ...{format_order_id(order_id)}. Awaiting fill...{Style.RESET_ALL}"
            )
        except (ccxt.InsufficientFunds, ccxt.ExchangeError, ccxt.NetworkError) as e:
            logger.error(
                f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}FAILED TO PLACE ENTRY ORDER: {type(e).__name__} - {e}{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Entry placement failed: {type(e).__name__}"
            )
            return None
        except Exception as e_place:
            logger.error(
                f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}UNEXPECTED ERROR Placing Entry Order: {e_place}{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Unexpected entry error: {type(e_place).__name__}"
            )
            return None

        filled_entry = wait_for_order_fill(
            exchange, order_id, symbol, CONFIG.order_fill_timeout_seconds
        )
        if not filled_entry or filled_entry.get("status") != "closed":
            final_status = filled_entry.get("status") if filled_entry else "timeout"
            logger.error(
                f"{Fore.RED}Entry order ...{format_order_id(order_id)} not confirmed filled (Status: {final_status}). Uncertain state.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Entry ...{format_order_id(order_id)} fill {final_status}. MANUAL CHECK!"
            )
            try:
                logger.warning(
                    f"Attempting to cancel unconfirmed order ...{format_order_id(order_id)}"
                )
                exchange.cancel_order(order_id, symbol)
                logger.info(
                    f"Cancel request sent for order ...{format_order_id(order_id)}"
                )
            except Exception as cancel_e:
                logger.warning(
                    f"Could not cancel order ...{format_order_id(order_id)}: {cancel_e}"
                )
            return None

        avg_fill_price = safe_decimal_conversion(filled_entry.get("average"))
        filled_qty = safe_decimal_conversion(filled_entry.get("filled"))
        cost = safe_decimal_conversion(filled_entry.get("cost"))

        if filled_qty <= CONFIG.position_qty_epsilon:
            logger.error(
                f"{Back.RED}{Fore.WHITE}CRITICAL: Invalid fill quantity ({filled_qty}) for ...{format_order_id(order_id)}. MANUAL CHECK!{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] CRITICAL ORDER FAIL ({side.upper()}): Invalid fill qty {filled_qty} for ...{format_order_id(order_id)}! MANUAL CHECK!"
            )
            return filled_entry
        if avg_fill_price <= 0 and cost > 0 and filled_qty > 0:
            avg_fill_price = cost / filled_qty
            logger.warning(
                f"{Fore.YELLOW}Fill price 'average' missing/zero. Estimated from cost/filled: {avg_fill_price:.4f}{Style.RESET_ALL}"
            )
        if avg_fill_price <= 0:
            logger.error(
                f"{Back.RED}{Fore.WHITE}CRITICAL: Invalid fill price ({avg_fill_price}) for ...{format_order_id(order_id)}. MANUAL CHECK!{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] CRITICAL ORDER FAIL ({side.upper()}): Invalid fill price {avg_fill_price} for ...{format_order_id(order_id)}! MANUAL CHECK!"
            )
            return filled_entry

        fill_bg_color = Back.GREEN if side == CONFIG.side_buy else Back.RED
        fill_fg_color = Fore.BLACK if side == CONFIG.side_buy else Fore.WHITE
        logger.success(
            f"{fill_bg_color}{fill_fg_color}{Style.BRIGHT}ENTRY CONFIRMED: ...{format_order_id(order_id)}. Filled: {filled_qty:.8f} @ Avg: {avg_fill_price:.4f}, Cost: {cost:.4f} USDT{Style.RESET_ALL}"
        )

        actual_sl_price_raw = (
            (avg_fill_price - sl_distance)
            if side == CONFIG.side_buy
            else (avg_fill_price + sl_distance)
        )
        if actual_sl_price_raw <= 0:
            logger.error(
                f"{Back.RED}{Fore.WHITE}CRITICAL FAILURE: ACTUAL SL price invalid ({actual_sl_price_raw:.4f}). Cannot place SL!{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] CRITICAL ({side.upper()}): Invalid ACTUAL SL price ({actual_sl_price_raw:.4f})! Emergency close attempt."
            )
            close_position(
                exchange,
                symbol,
                {"side": side, "qty": filled_qty},
                reason="Invalid SL Calc Post-Entry",
            )
            return filled_entry
        if min_price is not None and actual_sl_price_raw < min_price:
            actual_sl_price_raw = min_price
        try:
            actual_sl_price_str = format_price(exchange, symbol, actual_sl_price_raw)
            actual_sl_price_float = float(actual_sl_price_str)
            if actual_sl_price_float <= 0:
                raise ValueError("Formatted actual SL price invalid")
            logger.info(
                f"Calculated ACTUAL Fixed SL Trigger Price: {Fore.YELLOW}{actual_sl_price_str}{Style.RESET_ALL}"
            )
        except Exception as e:
            logger.error(
                f"{Back.RED}{Fore.WHITE}CRITICAL FAILURE: Failed to format ACTUAL SL price '{actual_sl_price_raw:.4f}': {e}. Cannot place SL!{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] CRITICAL ({side.upper()}): Failed format ACTUAL SL! Emergency close attempt."
            )
            close_position(
                exchange,
                symbol,
                {"side": side, "qty": filled_qty},
                reason="Invalid SL Format Post-Entry",
            )
            return filled_entry

        sl_order_id_short = "N/A"
        sl_placed_successfully = False
        try:
            sl_side = CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy
            sl_qty_str = format_amount(exchange, symbol, filled_qty)
            sl_qty_float = float(sl_qty_str)
            if sl_qty_float <= float(CONFIG.position_qty_epsilon):
                raise ValueError(f"Formatted SL quantity negligible: {sl_qty_float}")
            logger.info(
                f"{Fore.CYAN}Weaving Initial Fixed SL... Side: {sl_side.upper()}, Qty: {sl_qty_float:.8f}, TriggerPx: {actual_sl_price_str}{Style.RESET_ALL}"
            )
            sl_params = {"stopPrice": actual_sl_price_float, "reduceOnly": True}
            sl_order = exchange.create_order(
                symbol, "stopMarket", sl_side, sl_qty_float, params=sl_params
            )
            sl_order_id_short = format_order_id(sl_order.get("id"))
            logger.success(
                f"{Fore.GREEN}Initial Fixed SL ward placed. ID: ...{sl_order_id_short}, Trigger: {actual_sl_price_str}{Style.RESET_ALL}"
            )
            sl_placed_successfully = True
        except Exception as e:
            logger.error(
                f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}FAILED to place Initial Fixed SL ward: {e}{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] ERROR ({side.upper()}): Failed initial FIXED SL: {type(e).__name__}"
            )

        tsl_order_id_short = "N/A"
        tsl_act_price_str = "N/A"
        tsl_placed_successfully = False
        try:
            act_offset = avg_fill_price * tsl_activation_offset_percent
            act_price_raw = (
                (avg_fill_price + act_offset)
                if side == CONFIG.side_buy
                else (avg_fill_price - act_offset)
            )
            if act_price_raw <= 0:
                raise ValueError(f"Invalid TSL activation price: {act_price_raw:.4f}")
            if min_price is not None and act_price_raw < min_price:
                act_price_raw = min_price
            tsl_act_price_str = format_price(exchange, symbol, act_price_raw)
            tsl_act_price_float = float(tsl_act_price_str)
            if tsl_act_price_float <= 0:
                raise ValueError("Formatted TSL activation price invalid")

            tsl_side = CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy
            tsl_trail_value_str = str((tsl_percent * Decimal("100")).normalize())
            tsl_qty_str = format_amount(exchange, symbol, filled_qty)
            tsl_qty_float = float(tsl_qty_str)
            if tsl_qty_float <= float(CONFIG.position_qty_epsilon):
                raise ValueError(f"Formatted TSL quantity negligible: {tsl_qty_float}")
            logger.info(
                f"{Fore.CYAN}Weaving Trailing SL ({tsl_percent:.2%})... Side: {tsl_side.upper()}, Qty: {tsl_qty_float:.8f}, Trail%: {tsl_trail_value_str}, ActPx: {tsl_act_price_str}{Style.RESET_ALL}"
            )
            tsl_params = {
                "trailingStop": tsl_trail_value_str,
                "activePrice": tsl_act_price_float,
                "reduceOnly": True,
            }
            tsl_order = exchange.create_order(
                symbol, "stopMarket", tsl_side, tsl_qty_float, params=tsl_params
            )
            tsl_order_id_short = format_order_id(tsl_order.get("id"))
            logger.success(
                f"{Fore.GREEN}Trailing SL shield placed. ID: ...{tsl_order_id_short}, Trail%: {tsl_trail_value_str}, ActPx: {tsl_act_price_str}{Style.RESET_ALL}"
            )
            tsl_placed_successfully = True
        except Exception as e:
            logger.error(
                f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}FAILED to place Trailing SL shield: {e}{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] ERROR ({side.upper()}): Failed TSL: {type(e).__name__}"
            )

        if sl_placed_successfully or tsl_placed_successfully:
            sms_msg = (
                f"[{market_base}/{CONFIG.strategy_name}] ENTERED {side.upper()} {filled_qty:.8f} @ {avg_fill_price:.4f}. "
                f"SL: {'~' + actual_sl_price_str if sl_placed_successfully else 'FAIL'}(ID:...{sl_order_id_short}). "
                f"TSL: {tsl_percent:.2%}@{'~' + tsl_act_price_str if tsl_placed_successfully else 'FAIL'}(ID:...{tsl_order_id_short}). "
                f"EntryID:...{format_order_id(order_id)}"
            )
            send_sms_alert(sms_msg)
        elif not sl_placed_successfully and not tsl_placed_successfully:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}CRITICAL WARNING: Entry OK but BOTH SL & TSL FAILED! POS UNPROTECTED! EntryID:...{format_order_id(order_id)}{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] CRITICAL ({side.upper()}): Entry OK BUT BOTH SL & TSL FAILED! POS UNPROTECTED! EntryID:...{format_order_id(order_id)}"
            )
        return filled_entry

    except (
        ccxt.InsufficientFunds,
        ccxt.NetworkError,
        ccxt.ExchangeError,
        ValueError,
        InvalidOperation,
    ) as e:
        logger.error(
            f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Place Order Ritual ({side.upper()}): Pre-entry process failed: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Pre-entry setup failed: {type(e).__name__}"
        )
    except Exception as e_overall:
        logger.error(
            f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Place Order Ritual ({side.upper()}): Unexpected overall failure: {type(e_overall).__name__} - {e_overall}{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Unexpected overall error: {type(e_overall).__name__}"
        )
    return None


def cancel_open_orders(
    exchange: ccxt.Exchange, symbol: str, reason: str = "Cleanup"
) -> int:
    """Attempts to cancel all open orders for the specified symbol.
    Returns the number of orders successfully cancelled or confirmed closed.
    """
    logger.info(
        f"{Fore.CYAN}Order Cleanup Ritual: Initiating for {symbol} (Reason: {reason})...{Style.RESET_ALL}"
    )
    cancelled_count: int = 0
    failed_count: int = 0
    market_base = symbol.split("/")[0].split(":")[0]

    try:
        if not exchange.has.get("fetchOpenOrders"):
            logger.warning(
                f"{Fore.YELLOW}Order Cleanup: fetchOpenOrders spell not available for {exchange.id}. Cannot perform automated cleanup.{Style.RESET_ALL}"
            )
            return 0

        open_orders = exchange.fetch_open_orders(symbol)
        if not open_orders:
            logger.info(
                f"{Fore.CYAN}Order Cleanup: No open orders found for {symbol}.{Style.RESET_ALL}"
            )
            return 0

        logger.warning(
            f"{Fore.YELLOW}Order Cleanup: Found {len(open_orders)} open order(s) for {symbol}. Attempting cancellation...{Style.RESET_ALL}"
        )
        for order in open_orders:
            order_id = order.get("id")
            order_info_str = f"ID:...{format_order_id(order_id)} ({order.get('type', 'N/A')} {order.get('side', 'N/A')} Qty:{order.get('amount', 'N/A')} Px:{order.get('price', 'N/A')})"
            if order_id:
                try:
                    logger.debug(f"Cancelling order {order_info_str}...")
                    exchange.cancel_order(order_id, symbol)
                    logger.info(
                        f"{Fore.CYAN}Order Cleanup: Cancel request successful for {order_info_str}{Style.RESET_ALL}"
                    )
                    cancelled_count += 1
                    time.sleep(0.15)
                except ccxt.OrderNotFound:
                    logger.warning(
                        f"{Fore.YELLOW}Order Cleanup: Order not found (already closed/cancelled?): {order_info_str}{Style.RESET_ALL}"
                    )
                    cancelled_count += 1
                except (
                    ccxt.NetworkError,
                    ccxt.ExchangeError,
                    ccxt.RequestTimeout,
                ) as e:
                    logger.error(
                        f"{Fore.RED}Order Cleanup: FAILED to cancel {order_info_str}: {type(e).__name__} - {e}{Style.RESET_ALL}"
                    )
                    failed_count += 1
                except Exception as e_cancel:
                    logger.error(
                        f"{Fore.RED}Order Cleanup: UNEXPECTED error cancelling {order_info_str}: {type(e_cancel).__name__} - {e_cancel}{Style.RESET_ALL}"
                    )
                    failed_count += 1
            else:
                logger.error(
                    f"{Fore.RED}Order Cleanup: Found open order with no ID: {order}. Cannot cancel.{Style.RESET_ALL}"
                )
                failed_count += 1

        logger.info(
            f"{Fore.CYAN}Order Cleanup Ritual Finished for {symbol}. Successfully Cancelled/Closed: {cancelled_count}, Failed: {failed_count}.{Style.RESET_ALL}"
        )
        if failed_count > 0:
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] WARNING: Failed to cancel {failed_count} order(s) during {reason}. Check manually."
            )
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
        logger.error(
            f"{Fore.RED}Order Cleanup: Failed fetching open orders for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
    except Exception as e_fetch:
        logger.error(
            f"{Fore.RED}Order Cleanup: Unexpected error fetching open orders for {symbol}: {type(e_fetch).__name__} - {e_fetch}{Style.RESET_ALL}"
        )
    return cancelled_count


# --- Strategy Signal Generation - Interpreting the Omens ---
def generate_signals(df: pd.DataFrame, strategy_name: str) -> dict[str, Any]:
    """Generates entry/exit signals based on the selected strategy's interpretation of indicator values
    from the provided DataFrame. Requires at least 2 rows for comparisons.
    Returns a dict: {'enter_long': bool, 'enter_short': bool, 'exit_long': bool, 'exit_short': bool, 'exit_reason': str}.
    """
    signals = {
        "enter_long": False,
        "enter_short": False,
        "exit_long": False,
        "exit_short": False,
        "exit_reason": "Strategy Exit Signal",
    }
    required_rows = 2
    if df is None or len(df) < required_rows:
        logger.debug(
            f"Signal Gen ({strategy_name}): Insufficient data ({len(df) if df is not None else 0} rows, need {required_rows})."
        )
        return signals

    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]
    except IndexError:
        logger.error(
            f"Signal Gen ({strategy_name}): Error accessing DataFrame rows (len: {len(df)})."
        )
        return signals

    try:
        if strategy_name == "DUAL_SUPERTREND":
            primary_long_flip = last.get("st_long", False)
            primary_short_flip = last.get("st_short", False)
            confirm_is_up = last.get("confirm_trend", pd.NA)
            if pd.isna(confirm_is_up):
                logger.warning(
                    f"Signal Gen ({strategy_name}): Confirmation trend is NA."
                )
                return signals
            if primary_long_flip and confirm_is_up:
                signals["enter_long"] = True
            if primary_short_flip and not confirm_is_up:
                signals["enter_short"] = True
            if primary_short_flip:
                signals["exit_long"] = True
                signals["exit_reason"] = "Primary ST Flipped Short"
            if primary_long_flip:
                signals["exit_short"] = True
                signals["exit_reason"] = "Primary ST Flipped Long"

        elif strategy_name == "STOCHRSI_MOMENTUM":
            k_now, d_now, mom_now = (
                last.get("stochrsi_k", pd.NA),
                last.get("stochrsi_d", pd.NA),
                last.get("momentum", pd.NA),
            )
            k_prev, d_prev = (
                prev.get("stochrsi_k", pd.NA),
                prev.get("stochrsi_d", pd.NA),
            )
            if any(pd.isna(v) for v in [k_now, d_now, mom_now, k_prev, d_prev]):
                logger.debug(
                    f"Signal Gen ({strategy_name}): Skipping due to NA StochRSI/Mom values."
                )
                return signals
            if (
                k_prev <= d_prev
                and k_now > d_now
                and k_now < CONFIG.stochrsi_oversold
                and mom_now > CONFIG.position_qty_epsilon
            ):
                signals["enter_long"] = True
            if (
                k_prev >= d_prev
                and k_now < d_now
                and k_now > CONFIG.stochrsi_overbought
                and mom_now < -CONFIG.position_qty_epsilon
            ):
                signals["enter_short"] = True
            if k_prev >= d_prev and k_now < d_now:
                signals["exit_long"] = True
                signals["exit_reason"] = "StochRSI K crossed below D"
            if k_prev <= d_prev and k_now > d_now:
                signals["exit_short"] = True
                signals["exit_reason"] = "StochRSI K crossed above D"

        elif strategy_name == "EHLERS_FISHER":
            fish_now, sig_now = (
                last.get("ehlers_fisher", pd.NA),
                last.get("ehlers_signal", pd.NA),
            )
            fish_prev, sig_prev = (
                prev.get("ehlers_fisher", pd.NA),
                prev.get("ehlers_signal", pd.NA),
            )
            if any(pd.isna(v) for v in [fish_now, sig_now, fish_prev, sig_prev]):
                logger.debug(
                    f"Signal Gen ({strategy_name}): Skipping due to NA Ehlers Fisher values."
                )
                return signals
            if fish_prev <= sig_prev and fish_now > sig_now:
                signals["enter_long"] = True
            if fish_prev >= sig_prev and fish_now < sig_now:
                signals["enter_short"] = True
            if fish_prev >= sig_prev and fish_now < sig_now:
                signals["exit_long"] = True
                signals["exit_reason"] = "Ehlers Fisher crossed Short"
            if fish_prev <= sig_prev and fish_now > sig_now:
                signals["exit_short"] = True
                signals["exit_reason"] = "Ehlers Fisher crossed Long"

        elif strategy_name == "EHLERS_MA_CROSS":
            fast_ma_now, slow_ma_now = (
                last.get("ehlers_ssf_fast", pd.NA),
                last.get("ehlers_ssf_slow", pd.NA),
            )
            fast_ma_prev, slow_ma_prev = (
                prev.get("ehlers_ssf_fast", pd.NA),
                prev.get("ehlers_ssf_slow", pd.NA),
            )
            if any(
                pd.isna(v)
                for v in [fast_ma_now, slow_ma_now, fast_ma_prev, slow_ma_prev]
            ):
                logger.debug(
                    f"Signal Gen ({strategy_name}): Skipping due to NA Ehlers SSF MA values."
                )
                return signals
            if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now:
                signals["enter_long"] = True
            if fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now:
                signals["enter_short"] = True
            if fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now:
                signals["exit_long"] = True
                signals["exit_reason"] = "Fast Ehlers SSF MA crossed below Slow"
            if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now:
                signals["exit_short"] = True
                signals["exit_reason"] = "Fast Ehlers SSF MA crossed above Slow"
    except KeyError as e:
        logger.error(
            f"{Fore.RED}Signal Generation Error ({strategy_name}): Missing indicator column: {e}.{Style.RESET_ALL}"
        )
    except Exception as e:
        logger.error(
            f"{Fore.RED}Signal Generation Error ({strategy_name}): Unexpected error: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())

    if (
        signals["enter_long"]
        or signals["enter_short"]
        or signals["exit_long"]
        or signals["exit_short"]
    ):
        active_signals = {k: v for k, v in signals.items() if isinstance(v, bool) and v}
        logger.debug(f"Strategy Signals ({strategy_name}): {active_signals}")
    return signals


# --- Trading Logic - The Core Spell Weaving ---
def trade_logic(exchange: ccxt.Exchange, symbol: str, df: pd.DataFrame) -> None:
    """Executes the main trading logic for one cycle."""
    cycle_time_str = (
        df.index[-1].strftime("%Y-%m-%d %H:%M:%S %Z") if not df.empty else "N/A"
    )
    logger.info(
        f"{Fore.BLUE}{Style.BRIGHT}========== New Weaving Cycle ({CONFIG.strategy_name}): {symbol} | Candle: {cycle_time_str} =========={Style.RESET_ALL}"
    )

    indicator_periods = [
        CONFIG.st_atr_length,
        CONFIG.confirm_st_atr_length,
        CONFIG.stochrsi_rsi_length,
        CONFIG.stochrsi_stoch_length,
        CONFIG.momentum_length,
        CONFIG.ehlers_fisher_length,
        CONFIG.ehlers_fisher_signal_length,
        CONFIG.ehlers_fast_period,
        CONFIG.ehlers_slow_period,
        CONFIG.ehlers_ssf_poles,
        CONFIG.atr_calculation_period,
        CONFIG.volume_ma_period,
    ]
    stochrsi_lookback = (
        CONFIG.stochrsi_rsi_length
        + CONFIG.stochrsi_stoch_length
        + CONFIG.stochrsi_d_period
        + 5
    )
    ehlers_ssf_lookback = (
        max(CONFIG.ehlers_fast_period, CONFIG.ehlers_slow_period)
        + CONFIG.ehlers_ssf_poles
        + 5
    )
    required_rows = max(
        max(indicator_periods) + 10 if indicator_periods else 0,
        stochrsi_lookback,
        ehlers_ssf_lookback,
        50,
    )

    if df is None or len(df) < required_rows:
        logger.warning(
            f"{Fore.YELLOW}Trade Logic Skipped: Insufficient data ({len(df) if df is not None else 0} rows, need ~{required_rows}).{Style.RESET_ALL}"
        )
        return

    action_taken_this_cycle: bool = False
    try:
        logger.debug("Calculating all potential indicators...")
        df = calculate_supertrend(df, CONFIG.st_atr_length, CONFIG.st_multiplier)
        df = calculate_supertrend(
            df,
            CONFIG.confirm_st_atr_length,
            CONFIG.confirm_st_multiplier,
            prefix="confirm_",
        )
        df = calculate_stochrsi_momentum(
            df,
            CONFIG.stochrsi_rsi_length,
            CONFIG.stochrsi_stoch_length,
            CONFIG.stochrsi_k_period,
            CONFIG.stochrsi_d_period,
            CONFIG.momentum_length,
        )
        df = calculate_ehlers_fisher(
            df, CONFIG.ehlers_fisher_length, CONFIG.ehlers_fisher_signal_length
        )
        df = calculate_ehlers_ma(
            df,
            CONFIG.ehlers_fast_period,
            CONFIG.ehlers_slow_period,
            CONFIG.ehlers_ssf_poles,
        )
        vol_atr_data = analyze_volume_atr(
            df, CONFIG.atr_calculation_period, CONFIG.volume_ma_period
        )
        current_atr = vol_atr_data.get("atr")

        last_candle_data = (
            df.iloc[-1] if not df.empty else pd.Series()
        )  # Ensure last_candle_data is a Series
        current_price = safe_decimal_conversion(last_candle_data.get("close"))

        if pd.isna(current_price) or current_price <= 0:
            logger.warning(
                f"{Fore.YELLOW}Trade Logic Skipped: Last candle close price invalid ({current_price}).{Style.RESET_ALL}"
            )
            return
        can_place_new_order = current_atr is not None and current_atr > Decimal("0")
        if not can_place_new_order:
            logger.warning(
                f"{Fore.YELLOW}Invalid ATR ({current_atr}). Cannot calculate SL or place new entry orders.{Style.RESET_ALL}"
            )

        position = get_current_position(exchange, symbol)
        position_side, position_qty, position_entry = (
            position["side"],
            position["qty"],
            position["entry_price"],
        )

        ob_data = None
        if CONFIG.fetch_order_book_per_cycle:
            ob_data = analyze_order_book(
                exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit
            )

        # === Enhanced Indicator Output ===
        logger.info(f"{Fore.MAGENTA}--- Indicator Snapshot ---{Style.RESET_ALL}")
        logger.info(f"  Market State:")
        logger.info(f"    Close Price: {_format_for_log(current_price, 4)}")
        logger.info(
            f"    ATR({CONFIG.atr_calculation_period}): {_format_for_log(current_atr, 5)}"
        )

        vol_ratio = vol_atr_data.get("volume_ratio")
        is_vol_spike = (
            vol_ratio is not None and vol_ratio > CONFIG.volume_spike_threshold
        )
        logger.info(
            f"    Volume: Ratio={_format_for_log(vol_ratio, 2)}, Spike={is_vol_spike} (Thr={CONFIG.volume_spike_threshold}, ReqForEntry={CONFIG.require_volume_spike_for_entry})"
        )

        bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None
        spread = ob_data.get("spread") if ob_data else None
        if ob_data:
            ob_ratio_color = (
                Fore.GREEN
                if bid_ask_ratio
                and bid_ask_ratio >= CONFIG.order_book_ratio_threshold_long
                else (
                    Fore.RED
                    if bid_ask_ratio
                    and bid_ask_ratio <= CONFIG.order_book_ratio_threshold_short
                    else Fore.YELLOW
                )
            )
            logger.info(
                f"    Order Book: Ratio(B/A)={ob_ratio_color}{_format_for_log(bid_ask_ratio, 3)}{Style.RESET_ALL} (L≥{CONFIG.order_book_ratio_threshold_long},S≤{CONFIG.order_book_ratio_threshold_short}), Spread={_format_for_log(spread, 4)}"
            )
        else:
            logger.info(
                f"    Order Book: Not fetched for this snapshot (FetchPerCycle={CONFIG.fetch_order_book_per_cycle})"
            )

        # Strategy Specific Indicators
        strat_name = CONFIG.strategy_name
        logger.info(f"  Strategy Indicators ({strat_name}):")
        if strat_name == "DUAL_SUPERTREND":
            st_val = last_candle_data.get("supertrend", pd.NA)
            st_trend_is_up = last_candle_data.get("trend", pd.NA)
            st_long_flip = last_candle_data.get("st_long", False)
            st_short_flip = last_candle_data.get("st_short", False)
            confirm_st_val = last_candle_data.get("confirm_supertrend", pd.NA)
            confirm_st_trend_is_up = last_candle_data.get("confirm_trend", pd.NA)
            st_flip_str = (
                "Long" if st_long_flip else ("Short" if st_short_flip else "None")
            )
            logger.info(
                f"    Primary ST({CONFIG.st_atr_length},{CONFIG.st_multiplier}): Val={_format_for_log(st_val)}, Trend={_format_for_log(st_trend_is_up, is_bool_trend=True)}, Flip={st_flip_str}"
            )
            logger.info(
                f"    Confirm ST({CONFIG.confirm_st_atr_length},{CONFIG.confirm_st_multiplier}): Val={_format_for_log(confirm_st_val)}, Trend={_format_for_log(confirm_st_trend_is_up, is_bool_trend=True)}"
            )
        elif strat_name == "STOCHRSI_MOMENTUM":
            k, d, mom = (
                last_candle_data.get("stochrsi_k", pd.NA),
                last_candle_data.get("stochrsi_d", pd.NA),
                last_candle_data.get("momentum", pd.NA),
            )
            logger.info(
                f"    StochRSI K({CONFIG.stochrsi_k_period}): {_format_for_log(k, 2)}, D({CONFIG.stochrsi_d_period}): {_format_for_log(d, 2)} (OB:{CONFIG.stochrsi_overbought},OS:{CONFIG.stochrsi_oversold})"
            )
            logger.info(
                f"    Momentum({CONFIG.momentum_length}): {_format_for_log(mom, 4)}"
            )
        elif strat_name == "EHLERS_FISHER":
            fisher, signal_val = (
                last_candle_data.get("ehlers_fisher", pd.NA),
                last_candle_data.get("ehlers_signal", pd.NA),
            )
            logger.info(
                f"    Fisher({CONFIG.ehlers_fisher_length}): {_format_for_log(fisher, 4)}, Signal({CONFIG.ehlers_fisher_signal_length}): {_format_for_log(signal_val, 4)}"
            )
        elif strat_name == "EHLERS_MA_CROSS":
            ssf_fast, ssf_slow = (
                last_candle_data.get("ehlers_ssf_fast", pd.NA),
                last_candle_data.get("ehlers_ssf_slow", pd.NA),
            )
            logger.info(
                f"    Fast SSF MA({CONFIG.ehlers_fast_period},P{CONFIG.ehlers_ssf_poles}): {_format_for_log(ssf_fast, 4)}"
            )
            logger.info(
                f"    Slow SSF MA({CONFIG.ehlers_slow_period},P{CONFIG.ehlers_ssf_poles}): {_format_for_log(ssf_slow, 4)}"
            )

        pos_color = (
            Fore.GREEN
            if position_side == CONFIG.pos_long
            else (Fore.RED if position_side == CONFIG.pos_short else Fore.BLUE)
        )
        logger.info(
            f"  Position State: Side={pos_color}{position_side}{Style.RESET_ALL}, Qty={position_qty:.8f}, Entry={position_entry:.4f}"
        )
        logger.info(f"{Fore.MAGENTA}--------------------------{Style.RESET_ALL}")

        strategy_signals = generate_signals(df, CONFIG.strategy_name)
        should_exit_long = (
            position_side == CONFIG.pos_long and strategy_signals["exit_long"]
        )
        should_exit_short = (
            position_side == CONFIG.pos_short and strategy_signals["exit_short"]
        )

        if should_exit_long or should_exit_short:
            exit_reason = strategy_signals["exit_reason"]
            logger.warning(
                f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}*** STRATEGY EXIT SIGNAL: Closing {position_side} (Reason: {exit_reason}) ***{Style.RESET_ALL}"
            )
            action_taken_this_cycle = True
            logger.info("Performing pre-exit order cleanup...")
            cancel_open_orders(exchange, symbol, f"Pre-Exit Cleanup ({exit_reason})")
            time.sleep(0.75)
            close_result = close_position(
                exchange, symbol, position, reason=exit_reason
            )
            if close_result:
                logger.info(
                    f"Position close order placed for {position_side}. Pausing..."
                )
                time.sleep(CONFIG.post_close_delay_seconds)
            else:
                logger.error(
                    f"{Fore.RED}Failed to place position close order for {position_side}. Manual check advised.{Style.RESET_ALL}"
                )
                time.sleep(CONFIG.sleep_seconds // 2)
            logger.info("Exiting trade logic cycle after processing exit signal.")
            return

        if position_side != CONFIG.pos_none:
            logger.info(
                f"Holding {pos_color}{position_side}{Style.RESET_ALL} position. Awaiting Exchange SL/TSL or Strategy Exit."
            )
            return

        if not can_place_new_order:
            logger.warning(
                f"{Fore.YELLOW}Holding Cash. Cannot evaluate entry: Invalid ATR ({current_atr}).{Style.RESET_ALL}"
            )
            return

        logger.debug("Position is Flat. Checking strategy entry signals...")
        potential_entry_signal = (
            strategy_signals["enter_long"] or strategy_signals["enter_short"]
        )
        if not potential_entry_signal:
            logger.info("Holding Cash. No entry signal generated by strategy.")
            return

        logger.debug("Potential entry signal. Evaluating confirmation filters...")
        if ob_data is None:  # Fetch OB only if needed and not already fetched
            logger.debug("Fetching Order Book data for entry confirmation...")
            ob_data = analyze_order_book(
                exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit
            )
            bid_ask_ratio = (
                ob_data.get("bid_ask_ratio") if ob_data else None
            )  # Update after fetch

        ob_confirm_long, ob_confirm_short = False, False
        if bid_ask_ratio is not None:
            ob_confirm_long = bid_ask_ratio >= CONFIG.order_book_ratio_threshold_long
            ob_confirm_short = bid_ask_ratio <= CONFIG.order_book_ratio_threshold_short
        logger.debug(
            f"OB Confirm: Long={ob_confirm_long}, Short={ob_confirm_short} (Ratio: {_format_for_log(bid_ask_ratio, 3)})"
        )
        vol_confirm = not CONFIG.require_volume_spike_for_entry or is_vol_spike
        logger.debug(
            f"Vol Confirm: {vol_confirm} (Spike: {is_vol_spike}, Required: {CONFIG.require_volume_spike_for_entry})"
        )

        final_enter_long = (
            strategy_signals["enter_long"] and ob_confirm_long and vol_confirm
        )
        final_enter_short = (
            strategy_signals["enter_short"] and ob_confirm_short and vol_confirm
        )

        if strategy_signals["enter_long"]:
            logger.debug(
                f"Final Entry (Long): Strat={strategy_signals['enter_long']}, OB={ob_confirm_long}, Vol={vol_confirm} => {Fore.GREEN if final_enter_long else Fore.RED}{final_enter_long}{Style.RESET_ALL}"
            )
        if strategy_signals["enter_short"]:
            logger.debug(
                f"Final Entry (Short): Strat={strategy_signals['enter_short']}, OB={ob_confirm_short}, Vol={vol_confirm} => {Fore.GREEN if final_enter_short else Fore.RED}{final_enter_short}{Style.RESET_ALL}"
            )

        entry_params = {
            "exchange": exchange,
            "symbol": symbol,
            "risk_percentage": CONFIG.risk_per_trade_percentage,
            "current_atr": current_atr,
            "sl_atr_multiplier": CONFIG.atr_stop_loss_multiplier,
            "leverage": CONFIG.leverage,
            "max_order_cap_usdt": CONFIG.max_order_usdt_amount,
            "margin_check_buffer": CONFIG.required_margin_buffer,
            "tsl_percent": CONFIG.trailing_stop_percentage,
            "tsl_activation_offset_percent": CONFIG.trailing_stop_activation_offset_percent,
        }
        if final_enter_long:
            logger.success(
                f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}*** CONFIRMED LONG ENTRY SIGNAL ({CONFIG.strategy_name}) for {symbol} ***{Style.RESET_ALL}"
            )
            action_taken_this_cycle = True
            cancel_open_orders(exchange, symbol, "Pre-Long Entry Cleanup")
            time.sleep(0.5)
            place_result = place_risked_market_order(
                side=CONFIG.side_buy, **entry_params
            )
            if place_result:
                logger.info("Long entry process initiated.")
            else:
                logger.error(f"{Fore.RED}Long entry process failed.{Style.RESET_ALL}")
        elif final_enter_short:
            logger.success(
                f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}*** CONFIRMED SHORT ENTRY SIGNAL ({CONFIG.strategy_name}) for {symbol} ***{Style.RESET_ALL}"
            )
            action_taken_this_cycle = True
            cancel_open_orders(exchange, symbol, "Pre-Short Entry Cleanup")
            time.sleep(0.5)
            place_result = place_risked_market_order(
                side=CONFIG.side_sell, **entry_params
            )
            if place_result:
                logger.info("Short entry process initiated.")
            else:
                logger.error(f"{Fore.RED}Short entry process failed.{Style.RESET_ALL}")
        else:
            if potential_entry_signal and not action_taken_this_cycle:
                logger.info(
                    "Holding Cash. Strategy signal present but confirmation filters not met."
                )

    except Exception as e:
        logger.error(
            f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}CRITICAL UNEXPECTED ERROR in trade_logic cycle: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        send_sms_alert(
            f"[{symbol.split('/')[0].split(':')[0]}/{CONFIG.strategy_name}] CRITICAL ERROR in trade_logic: {type(e).__name__}. Check logs!"
        )
    finally:
        logger.info(
            f"{Fore.BLUE}{Style.BRIGHT}========== Cycle Weaving End: {symbol} =========={Style.RESET_ALL}\n"
        )


# --- Graceful Shutdown - Withdrawing the Arcane Energies ---
def graceful_shutdown(exchange: ccxt.Exchange | None, symbol: str | None) -> None:
    """Attempts to cancel all open orders and close any existing position before exiting."""
    logger.warning(
        f"\n{Fore.YELLOW}{Style.BRIGHT}Shutdown requested. Withdrawing arcane energies gracefully...{Style.RESET_ALL}"
    )
    market_base = symbol.split("/")[0].split(":")[0] if symbol else "Bot"
    send_sms_alert(
        f"[{market_base}/{CONFIG.strategy_name}] Shutdown initiated. Attempting cleanup..."
    )

    if not exchange or not symbol:
        logger.warning(
            f"{Fore.YELLOW}Shutdown: Exchange portal or symbol not defined. Cannot perform automated cleanup.{Style.RESET_ALL}"
        )
        return

    try:
        logger.warning("Shutdown Step 1: Cancelling all open orders...")
        cancelled_count = cancel_open_orders(
            exchange, symbol, reason="Graceful Shutdown"
        )
        logger.info(f"Shutdown Step 1: Cancelled {cancelled_count} open orders.")
        time.sleep(1.5)

        logger.warning("Shutdown Step 2: Checking for active position to close...")
        position = get_current_position(exchange, symbol)

        if (
            position["side"] != CONFIG.pos_none
            and position["qty"] > CONFIG.position_qty_epsilon
        ):
            pos_color = Fore.GREEN if position["side"] == CONFIG.pos_long else Fore.RED
            logger.warning(
                f"{Fore.YELLOW}Shutdown Step 2: Active {pos_color}{position['side']}{Style.RESET_ALL} position (Qty: {position['qty']:.8f}). Attempting market close...{Style.RESET_ALL}"
            )
            close_result = close_position(exchange, symbol, position, reason="Shutdown")

            if close_result:
                logger.info(
                    f"{Fore.CYAN}Shutdown Step 2: Close order placed. Waiting {CONFIG.post_close_delay_seconds * 2}s...{Style.RESET_ALL}"
                )
                time.sleep(CONFIG.post_close_delay_seconds * 2)
                logger.warning("Shutdown Step 3: Final position confirmation check...")
                final_pos = get_current_position(exchange, symbol)
                if (
                    final_pos["side"] == CONFIG.pos_none
                    or final_pos["qty"] <= CONFIG.position_qty_epsilon
                ):
                    logger.success(
                        f"{Fore.GREEN}{Style.BRIGHT}Shutdown Step 3: Position confirmed CLOSED/FLAT.{Style.RESET_ALL}"
                    )
                    send_sms_alert(
                        f"[{market_base}/{CONFIG.strategy_name}] Position confirmed CLOSED on shutdown."
                    )
                else:
                    final_pos_color = (
                        Fore.GREEN if final_pos["side"] == CONFIG.pos_long else Fore.RED
                    )
                    logger.critical(
                        f"{Back.RED}{Fore.WHITE}CRITICAL SHUTDOWN FAILURE: FAILED TO CONFIRM closure! Final: {final_pos_color}{final_pos['side']}{Style.RESET_ALL} Qty={final_pos['qty']:.8f}. MANUAL INTERVENTION!{Style.RESET_ALL}"
                    )
                    send_sms_alert(
                        f"[{market_base}/{CONFIG.strategy_name}] CRITICAL SHUTDOWN ERROR: Failed confirm closure! Final: {final_pos['side']} Qty={final_pos['qty']:.8f}. MANUAL CHECK!"
                    )
            else:
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}CRITICAL SHUTDOWN FAILURE: Failed to place close order. Position likely still open! MANUAL INTERVENTION!{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"[{market_base}/{CONFIG.strategy_name}] CRITICAL SHUTDOWN ERROR: Failed PLACE close order. MANUAL CHECK!"
                )
        else:
            logger.info(
                f"{Fore.GREEN}Shutdown Step 2: No active position found. Clean exit state.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] No active position on shutdown."
            )
    except Exception as e:
        logger.error(
            f"{Fore.RED}Shutdown Error: Unexpected error during cleanup: {e}{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[{market_base}/{CONFIG.strategy_name}] Error during shutdown cleanup: {type(e).__name__}. Check logs & position manually."
        )
    logger.info(
        f"{Fore.YELLOW}{Style.BRIGHT}--- Pyrmethus Scalping Spell Shutdown Sequence Complete ---{Style.RESET_ALL}"
    )


# --- Main Execution - Igniting the Spell ---
def main() -> None:
    """Main function: Initializes components, sets up the market, and runs the trading loop."""
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    logger.info(
        f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Bybit Scalping Spell v2.2.3 Initializing ({start_time_str}) ---{Style.RESET_ALL}"
    )
    logger.info(
        f"{Fore.CYAN}--- Strategy Enchantment Selected: {CONFIG.strategy_name} ---{Style.RESET_ALL}"
    )
    logger.info(
        f"{Fore.GREEN}--- Protective Wards Activated: Initial ATR-Stop + Exchange Trailing Stop (Bybit V5 Native) ---{Style.RESET_ALL}"
    )
    logger.warning(
        f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}--- !!! LIVE FUTURES SCALPING MODE - EXTREME RISK - EDUCATIONAL USE ONLY !!! ---{Style.RESET_ALL}"
    )

    exchange: ccxt.Exchange | None = None
    symbol_unified: str | None = None
    run_bot: bool = True
    cycle_count: int = 0

    try:
        exchange = initialize_exchange()
        if not exchange:
            logger.critical(
                "Failed to open exchange portal. Spell cannot proceed. Exiting."
            )
            return

        try:
            symbol_to_use = CONFIG.symbol
            logger.info(f"Attempting to focus spell on symbol: {symbol_to_use}")
            market = exchange.market(symbol_to_use)
            symbol_unified = market["symbol"]
            market_type, is_contract, is_linear = (
                market.get("type", "unknown"),
                market.get("contract", False),
                market.get("linear", False),
            )
            if not is_contract:
                raise ValueError(
                    f"Market '{symbol_unified}' (Type: {market_type}) is not a contract market."
                )
            if not is_linear:
                logger.warning(
                    f"{Fore.YELLOW}Market '{symbol_unified}' not detected as LINEAR. Ensure compatibility.{Style.RESET_ALL}"
                )
            logger.info(
                f"{Fore.GREEN}Spell focused on Symbol: {symbol_unified} (Type: {market_type}, Contract: {is_contract}, Linear: {is_linear}){Style.RESET_ALL}"
            )
            if not set_leverage(exchange, symbol_unified, CONFIG.leverage):
                raise RuntimeError(f"Leverage conjuring failed for {symbol_unified}.")
        except (ccxt.BadSymbol, KeyError, ValueError, RuntimeError) as e:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}Symbol/Leverage setup failed: {e}{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[Pyrmethus/{CONFIG.strategy_name}] CRITICAL: Symbol/Leverage setup FAILED ({e}). Exiting."
            )
            return
        except Exception as e_setup:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}Unexpected error during spell focus setup: {e_setup}{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[Pyrmethus/{CONFIG.strategy_name}] CRITICAL: Unexpected setup error. Exiting."
            )
            return

        logger.info(
            f"{Fore.MAGENTA}--- Spell Configuration Summary ---{Style.RESET_ALL}"
        )
        logger.info(
            f"{Fore.WHITE}Symbol: {symbol_unified}, Interval: {CONFIG.interval}, Leverage: {CONFIG.leverage}x"
        )
        logger.info(f"{Fore.CYAN}Strategy Path: {CONFIG.strategy_name}")
        if CONFIG.strategy_name == "DUAL_SUPERTREND":
            logger.info(
                f"  Params: ST={CONFIG.st_atr_length}/{CONFIG.st_multiplier}, ConfirmST={CONFIG.confirm_st_atr_length}/{CONFIG.confirm_st_multiplier}"
            )
        elif CONFIG.strategy_name == "STOCHRSI_MOMENTUM":
            logger.info(
                f"  Params: StochRSI={CONFIG.stochrsi_rsi_length}/{CONFIG.stochrsi_stoch_length}/{CONFIG.stochrsi_k_period}/{CONFIG.stochrsi_d_period} (OB={CONFIG.stochrsi_overbought},OS={CONFIG.stochrsi_oversold}), Mom={CONFIG.momentum_length}"
            )
        elif CONFIG.strategy_name == "EHLERS_FISHER":
            logger.info(
                f"  Params: Fisher={CONFIG.ehlers_fisher_length}, Signal={CONFIG.ehlers_fisher_signal_length}"
            )
        elif CONFIG.strategy_name == "EHLERS_MA_CROSS":
            logger.info(
                f"  Params: FastSSF={CONFIG.ehlers_fast_period}, SlowSSF={CONFIG.ehlers_slow_period}, Poles={CONFIG.ehlers_ssf_poles} {Fore.GREEN}(Ehlers SuperSmoother){Style.RESET_ALL}"
            )
        logger.info(
            f"{Fore.GREEN}Risk Ward: {CONFIG.risk_per_trade_percentage:.3%} equity/trade, Max Pos Value: {CONFIG.max_order_usdt_amount:.4f} USDT"
        )
        logger.info(
            f"{Fore.GREEN}Initial SL Ward: {CONFIG.atr_stop_loss_multiplier} * ATR({CONFIG.atr_calculation_period})"
        )
        logger.info(
            f"{Fore.GREEN}Trailing SL Shield: {CONFIG.trailing_stop_percentage:.2%}, Activation Offset: {CONFIG.trailing_stop_activation_offset_percent:.2%}"
        )
        logger.info(
            f"{Fore.YELLOW}Volume Filter: EntryRequiresSpike={CONFIG.require_volume_spike_for_entry} (MA={CONFIG.volume_ma_period}, SpikeThr={CONFIG.volume_spike_threshold}x)"
        )
        logger.info(
            f"{Fore.YELLOW}Order Book Filter: FetchPerCycle={CONFIG.fetch_order_book_per_cycle} (Depth={CONFIG.order_book_depth}, L>={CONFIG.order_book_ratio_threshold_long}, S<={CONFIG.order_book_ratio_threshold_short})"
        )
        margin_buffer_percent = (CONFIG.required_margin_buffer - Decimal(1)) * Decimal(
            100
        )
        logger.info(
            f"{Fore.WHITE}Timing: Sleep={CONFIG.sleep_seconds}s | API: RecvWin={CONFIG.default_recv_window}ms, FillTimeout={CONFIG.order_fill_timeout_seconds}s"
        )
        logger.info(
            f"{Fore.WHITE}Other: Margin Buffer={margin_buffer_percent:.1f}%, SMS Alerts={CONFIG.enable_sms_alerts}"
        )
        logger.info(
            f"{Fore.CYAN}Oracle Verbosity (Log Level): {logging.getLevelName(logger.level)}"
        )
        logger.info(f"{Fore.MAGENTA}{'-' * 30}{Style.RESET_ALL}")

        market_base = symbol_unified.split("/")[0].split(":")[0]
        send_sms_alert(
            f"[{market_base}/{CONFIG.strategy_name}] Pyrmethus Bot v2.2.3 Initialized. Symbol: {symbol_unified}, Strat: {CONFIG.strategy_name}. Starting main loop."
        )

        while run_bot:
            cycle_start_time = time.monotonic()
            cycle_count += 1
            logger.debug(
                f"{Fore.CYAN}--- Cycle {cycle_count} Weaving Start ({time.strftime('%H:%M:%S')}) ---{Style.RESET_ALL}"
            )
            try:
                indicator_periods_main = [
                    CONFIG.st_atr_length,
                    CONFIG.confirm_st_atr_length,
                    CONFIG.stochrsi_rsi_length,
                    CONFIG.stochrsi_stoch_length,
                    CONFIG.momentum_length,
                    CONFIG.ehlers_fisher_length,
                    CONFIG.ehlers_fast_period,
                    CONFIG.ehlers_slow_period,
                    CONFIG.atr_calculation_period,
                    CONFIG.volume_ma_period,
                ]  # Note: k,d,signal_length,poles are not direct price lookbacks for this list
                stochrsi_lookback_main = (
                    CONFIG.stochrsi_rsi_length
                    + CONFIG.stochrsi_stoch_length
                    + CONFIG.stochrsi_d_period
                    + 5
                )
                ehlers_ssf_lookback_main = (
                    max(CONFIG.ehlers_fast_period, CONFIG.ehlers_slow_period)
                    + CONFIG.ehlers_ssf_poles
                    + 5
                )
                data_limit = (
                    max(
                        max(indicator_periods_main) + 15
                        if indicator_periods_main
                        else 0,
                        stochrsi_lookback_main,
                        ehlers_ssf_lookback_main,
                        100,
                    )
                    + CONFIG.api_fetch_limit_buffer
                )

                df = get_market_data(
                    exchange, symbol_unified, CONFIG.interval, limit=data_limit
                )
                if df is not None and not df.empty:
                    trade_logic(exchange, symbol_unified, df.copy())
                else:
                    logger.warning(
                        f"{Fore.YELLOW}Skipping trade logic: invalid/missing market data for {symbol_unified}.{Style.RESET_ALL}"
                    )
            except ccxt.RateLimitExceeded as e:
                logger.warning(
                    f"{Back.YELLOW}{Fore.BLACK}Rate Limit Exceeded: {e}. Sleeping longer...{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"[{market_base}/{CONFIG.strategy_name}] WARNING: Rate limit exceeded! Pausing {CONFIG.sleep_seconds * 6}s."
                )
                time.sleep(CONFIG.sleep_seconds * 6)
            except ccxt.NetworkError as e:
                logger.warning(
                    f"{Fore.YELLOW}Network disturbance in main loop: {e}. Retrying next cycle.{Style.RESET_ALL}"
                )
                time.sleep(CONFIG.sleep_seconds * 2)
            except ccxt.ExchangeNotAvailable as e:
                logger.error(
                    f"{Back.RED}{Fore.WHITE}Exchange Not Available: {e}. Sleeping much longer...{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"[{market_base}/{CONFIG.strategy_name}] ERROR: Exchange unavailable ({type(e).__name__})! Long pause (60s)."
                )
                time.sleep(60)
            except ccxt.AuthenticationError as e:
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}FATAL: Authentication Error: {e}. API keys invalid. Stopping NOW.{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"[{market_base}/{CONFIG.strategy_name}] CRITICAL: Authentication Error! Bot stopping NOW."
                )
                run_bot = False
            except ccxt.ExchangeError as e:
                logger.error(
                    f"{Fore.RED}Unhandled Exchange Error in main loop: {e}{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"[{market_base}/{CONFIG.strategy_name}] ERROR: Unhandled Exchange error: {type(e).__name__}."
                )
                time.sleep(CONFIG.sleep_seconds)
            except Exception as e:
                logger.exception(
                    f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}!!! UNEXPECTED CRITICAL CHAOS in Main Loop: {e} !!! Stopping spell!{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"[{market_base}/{CONFIG.strategy_name}] CRITICAL UNEXPECTED ERROR: {type(e).__name__}! Bot stopping NOW."
                )
                run_bot = False

            if run_bot:
                elapsed = time.monotonic() - cycle_start_time
                sleep_duration = max(0, CONFIG.sleep_seconds - elapsed)
                logger.debug(
                    f"Cycle {cycle_count} duration: {elapsed:.2f}s. Sleeping for {sleep_duration:.2f}s."
                )
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
    except KeyboardInterrupt:
        logger.warning(
            f"\n{Fore.YELLOW}{Style.BRIGHT}KeyboardInterrupt detected. Withdrawing energies...{Style.RESET_ALL}"
        )
        run_bot = False
    except Exception as startup_error:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}CRITICAL ERROR during bot startup: {startup_error}{Style.RESET_ALL}"
        )
        if CONFIG and CONFIG.enable_sms_alerts and CONFIG.sms_recipient_number:
            send_sms_alert(
                f"[Pyrmethus] CRITICAL STARTUP ERROR: {type(startup_error).__name__}. Bot failed."
            )
        run_bot = False
    finally:
        graceful_shutdown(exchange, symbol_unified)
        logger.info(
            f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Scalping Spell Deactivated ---{Style.RESET_ALL}"
        )


if __name__ == "__main__":
    main()
