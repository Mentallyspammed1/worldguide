#!/usr/bin/env python

# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v2.2.1 (Fortified Configuration & Enhanced Clarity/Robustness)
# Conjures high-frequency trades on Bybit Futures with enhanced precision, adaptable strategies, and Termux integration.

"""High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 2.2.1 (Unified: Selectable Strategies + Precision + Native SL/TSL + Fortified Config + Pyrmethus Enhancements + Robustness).

Features:
- Multiple strategies selectable via config: "DUAL_SUPERTREND", "STOCHRSI_MOMENTUM", "EHLERS_FISHER", "EHLERS_MA_CROSS".
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
    sys.exit(1)

# --- Initializations - Preparing the Ritual Chamber ---
colorama_init(autoreset=True)  # Activate Colorama's magic for vibrant logs
load_dotenv()  # Load secrets from the hidden .env scroll (if present)
getcontext().prec = 18  # Set Decimal precision for financial exactitude (adjust if needed)


# --- Configuration Class - Defining the Spell's Parameters ---
class Config:
    """Loads and validates configuration parameters from environment variables."""

    def __init__(self) -> None:
        logger.info(f"{Fore.MAGENTA}--- Summoning Configuration Runes ---{Style.RESET_ALL}")
        # --- API Credentials - Keys to the Exchange Vault ---
        self.api_key: str | None = self._get_env("BYBIT_API_KEY", required=True, color=Fore.RED)
        self.api_secret: str | None = self._get_env("BYBIT_API_SECRET", required=True, color=Fore.RED)

        # --- Trading Parameters - Core Incantation Variables ---
        self.symbol: str = self._get_env(
            "SYMBOL", "BTC/USDT:USDT", color=Fore.YELLOW
        )  # Target market (CCXT unified format)
        self.interval: str = self._get_env(
            "INTERVAL", "1m", color=Fore.YELLOW
        )  # Chart timeframe (e.g., '1m', '5m', '1h')
        self.leverage: int = self._get_env(
            "LEVERAGE", 10, cast_type=int, color=Fore.YELLOW
        )  # Desired leverage multiplier
        self.sleep_seconds: int = self._get_env(
            "SLEEP_SECONDS", 10, cast_type=int, color=Fore.YELLOW
        )  # Pause between trading cycles (seconds)

        # --- Strategy Selection - Choosing the Path of Magic ---
        self.strategy_name: str = self._get_env("STRATEGY_NAME", "DUAL_SUPERTREND", color=Fore.CYAN).upper()
        self.valid_strategies: list[str] = ["DUAL_SUPERTREND", "STOCHRSI_MOMENTUM", "EHLERS_FISHER", "EHLERS_MA_CROSS"]
        if self.strategy_name not in self.valid_strategies:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}Invalid STRATEGY_NAME '{self.strategy_name}'. Valid paths: {self.valid_strategies}{Style.RESET_ALL}"
            )
            raise ValueError(f"Invalid STRATEGY_NAME '{self.strategy_name}'.")
        logger.info(f"{Fore.CYAN}Chosen Strategy Path: {self.strategy_name}{Style.RESET_ALL}")

        # --- Risk Management - Wards Against Ruin ---
        self.risk_per_trade_percentage: Decimal = self._get_env(
            "RISK_PER_TRADE_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN
        )  # e.g., 0.005 = 0.5% of equity per trade
        self.atr_stop_loss_multiplier: Decimal = self._get_env(
            "ATR_STOP_LOSS_MULTIPLIER", "1.5", cast_type=Decimal, color=Fore.GREEN
        )  # Multiplier for ATR to set initial SL distance
        self.max_order_usdt_amount: Decimal = self._get_env(
            "MAX_ORDER_USDT_AMOUNT", "500.0", cast_type=Decimal, color=Fore.GREEN
        )  # Maximum position value in USDT (overrides risk calc if needed)
        self.required_margin_buffer: Decimal = self._get_env(
            "REQUIRED_MARGIN_BUFFER", "1.05", cast_type=Decimal, color=Fore.GREEN
        )  # e.g., 1.05 = Require 5% extra free margin than estimated

        # --- Trailing Stop Loss (Exchange Native - Bybit V5) - The Adaptive Shield ---
        self.trailing_stop_percentage: Decimal = self._get_env(
            "TRAILING_STOP_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN
        )  # e.g., 0.005 = 0.5% trailing distance from high/low water mark
        self.trailing_stop_activation_offset_percent: Decimal = self._get_env(
            "TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT", "0.001", cast_type=Decimal, color=Fore.GREEN
        )  # e.g., 0.001 = 0.1% price movement in profit before TSL activates

        # --- Strategy-Specific Parameters - Tuning the Chosen Path ---
        # Dual Supertrend
        self.st_atr_length: int = self._get_env(
            "ST_ATR_LENGTH", 7, cast_type=int, color=Fore.CYAN
        )  # Primary Supertrend ATR period
        self.st_multiplier: Decimal = self._get_env(
            "ST_MULTIPLIER", "2.5", cast_type=Decimal, color=Fore.CYAN
        )  # Primary Supertrend ATR multiplier
        self.confirm_st_atr_length: int = self._get_env(
            "CONFIRM_ST_ATR_LENGTH", 5, cast_type=int, color=Fore.CYAN
        )  # Confirmation Supertrend ATR period
        self.confirm_st_multiplier: Decimal = self._get_env(
            "CONFIRM_ST_MULTIPLIER", "2.0", cast_type=Decimal, color=Fore.CYAN
        )  # Confirmation Supertrend ATR multiplier
        # StochRSI + Momentum
        self.stochrsi_rsi_length: int = self._get_env(
            "STOCHRSI_RSI_LENGTH", 14, cast_type=int, color=Fore.CYAN
        )  # StochRSI: RSI period
        self.stochrsi_stoch_length: int = self._get_env(
            "STOCHRSI_STOCH_LENGTH", 14, cast_type=int, color=Fore.CYAN
        )  # StochRSI: Stochastic period
        self.stochrsi_k_period: int = self._get_env(
            "STOCHRSI_K_PERIOD", 3, cast_type=int, color=Fore.CYAN
        )  # StochRSI: %K smoothing period
        self.stochrsi_d_period: int = self._get_env(
            "STOCHRSI_D_PERIOD", 3, cast_type=int, color=Fore.CYAN
        )  # StochRSI: %D smoothing period (signal line)
        self.stochrsi_overbought: Decimal = self._get_env(
            "STOCHRSI_OVERBOUGHT", "80.0", cast_type=Decimal, color=Fore.CYAN
        )  # StochRSI overbought level
        self.stochrsi_oversold: Decimal = self._get_env(
            "STOCHRSI_OVERSOLD", "20.0", cast_type=Decimal, color=Fore.CYAN
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
        # Ehlers MA Cross (Placeholder - see function note)
        self.ehlers_fast_period: int = self._get_env(
            "EHLERS_FAST_PERIOD", 10, cast_type=int, color=Fore.CYAN
        )  # Fast EMA period (placeholder)
        self.ehlers_slow_period: int = self._get_env(
            "EHLERS_SLOW_PERIOD", 30, cast_type=int, color=Fore.CYAN
        )  # Slow EMA period (placeholder)

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
            "ORDER_BOOK_DEPTH", 10, cast_type=int, color=Fore.YELLOW
        )  # Number of bid/ask levels to analyze
        self.order_book_ratio_threshold_long: Decimal = self._get_env(
            "ORDER_BOOK_RATIO_THRESHOLD_LONG", "1.2", cast_type=Decimal, color=Fore.YELLOW
        )  # Min Bid/Ask volume ratio for long confirmation
        self.order_book_ratio_threshold_short: Decimal = self._get_env(
            "ORDER_BOOK_RATIO_THRESHOLD_SHORT", "0.8", cast_type=Decimal, color=Fore.YELLOW
        )  # Max Bid/Ask volume ratio for short confirmation (ratio = Bids/Asks)
        self.fetch_order_book_per_cycle: bool = self._get_env(
            "FETCH_ORDER_BOOK_PER_CYCLE", "false", cast_type=bool, color=Fore.YELLOW
        )  # Fetch OB every cycle (more API calls) or only when signal occurs?

        # --- ATR Calculation (for Initial SL) ---
        self.atr_calculation_period: int = self._get_env(
            "ATR_CALCULATION_PERIOD", 14, cast_type=int, color=Fore.GREEN
        )  # Period for ATR calculation used in SL

        # --- Termux SMS Alerts - Whispers Through the Digital Veil ---
        self.enable_sms_alerts: bool = self._get_env(
            "ENABLE_SMS_ALERTS", "false", cast_type=bool, color=Fore.MAGENTA
        )  # Enable/disable SMS alerts globally
        self.sms_recipient_number: str | None = self._get_env(
            "SMS_RECIPIENT_NUMBER", None, color=Fore.MAGENTA
        )  # Recipient phone number for alerts
        self.sms_timeout_seconds: int = self._get_env(
            "SMS_TIMEOUT_SECONDS", 30, cast_type=int, color=Fore.MAGENTA
        )  # Max time to wait for SMS command (seconds)

        # --- CCXT / API Parameters - Tuning the Connection ---
        self.default_recv_window: int = 10000  # Milliseconds for API request validity (Bybit default is 5000)
        self.order_book_fetch_limit: int = max(
            25, self.order_book_depth
        )  # How many levels to fetch (ensure >= depth needed)
        self.shallow_ob_fetch_depth: int = 5  # Depth for quick price estimates (used in order placement)
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
        self.api_fetch_limit_buffer: int = 10  # Extra candles to fetch beyond indicator needs
        self.position_qty_epsilon: Decimal = Decimal(
            "1e-9"
        )  # Small value for float comparisons involving position size
        self.post_close_delay_seconds: int = 3  # Brief pause after successfully closing a position (seconds)

        logger.info(f"{Fore.MAGENTA}--- Configuration Runes Summoned and Verified ---{Style.RESET_ALL}")

    def _get_env(
        self, key: str, default: Any = None, cast_type: type = str, required: bool = False, color: str = Fore.WHITE
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
            logger.debug(f"{color}Summoning {key}: Not Set. Using Default: '{default}'{Style.RESET_ALL}")
            value_to_cast = default  # Assign default, needs casting below
            source = "Default"
        else:
            # Environment variable found
            logger.debug(f"{color}Summoning {key}: Found Env Value: '{value_str}'{Style.RESET_ALL}")
            value_to_cast = value_str  # Assign found string, needs casting below

        # --- Attempt Casting (applies to both env var value and default value) ---
        if value_to_cast is None:
            # This handles cases where default=None or env var was explicitly empty and default was None
            if required:
                # This should have been caught earlier if env var was missing, but catches required=True with default=None
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}CRITICAL: Required configuration rune '{key}' has no value (from env or default).{Style.RESET_ALL}"
                )
                raise ValueError(f"Required environment variable '{key}' resolved to None.")
            else:
                # Not required and value is None, return None directly
                logger.debug(f"{color}Final value for {key}: None (Type: NoneType) (Source: {source}){Style.RESET_ALL}")
                return None

        final_value: Any = None
        try:
            raw_value_str = str(value_to_cast)  # Ensure we have a string representation for casting checks
            if cast_type == bool:
                final_value = raw_value_str.lower() in ["true", "1", "yes", "y"]
            elif cast_type == Decimal:
                final_value = Decimal(raw_value_str)
            elif cast_type == int:
                final_value = int(Decimal(raw_value_str))  # Cast via Decimal to handle "10.0" -> 10
            elif cast_type == float:
                final_value = float(raw_value_str)
            elif cast_type == str:
                final_value = raw_value_str  # Already string or cast to string
            else:
                # Should not happen if using supported types, but good practice
                logger.warning(f"Unsupported cast_type '{cast_type.__name__}' for key '{key}'. Returning raw value.")
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
                    raise ValueError(f"Required env var '{key}' failed casting and has no valid default.")
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
LOGGING_LEVEL: int = logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],  # Output to the Termux console (or wherever stdout goes)
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
        logging.DEBUG, f"{Fore.CYAN}{Style.DIM}{logging.getLevelName(logging.DEBUG)}{Style.RESET_ALL}"
    )  # Dim Cyan for Debug
    logging.addLevelName(
        logging.INFO, f"{Fore.BLUE}{logging.getLevelName(logging.INFO)}{Style.RESET_ALL}"
    )  # Blue for Info
    logging.addLevelName(
        SUCCESS_LEVEL, f"{Fore.MAGENTA}{Style.BRIGHT}{logging.getLevelName(SUCCESS_LEVEL)}{Style.RESET_ALL}"
    )  # Bright Magenta for Success
    logging.addLevelName(
        logging.WARNING, f"{Fore.YELLOW}{Style.BRIGHT}{logging.getLevelName(logging.WARNING)}{Style.RESET_ALL}"
    )  # Bright Yellow for Warning
    logging.addLevelName(
        logging.ERROR, f"{Fore.RED}{Style.BRIGHT}{logging.getLevelName(logging.ERROR)}{Style.RESET_ALL}"
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
_termux_sms_command_exists: bool | None = None  # Cache the result of checking command existence


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
        command: list[str] = ["termux-sms-send", "-n", CONFIG.sms_recipient_number, message]
        logger.info(
            f"{Fore.MAGENTA}Dispatching SMS whisper to {CONFIG.sms_recipient_number} (Timeout: {CONFIG.sms_timeout_seconds}s)...{Style.RESET_ALL}"
        )

        # Execute the spell via subprocess
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=CONFIG.sms_timeout_seconds
        )

        if result.returncode == 0:
            logger.success(f"{Fore.MAGENTA}SMS whisper dispatched successfully.{Style.RESET_ALL}")
            return True
        else:
            # Log error details from stderr if available
            error_details = result.stderr.strip() if result.stderr else "No stderr output"
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
        logger.error(f"{Fore.RED}SMS failed: Command timed out after {CONFIG.sms_timeout_seconds}s.{Style.RESET_ALL}")
        return False
    except Exception as e:
        logger.error(f"{Fore.RED}SMS failed: Unexpected disturbance during dispatch: {e}{Style.RESET_ALL}")
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
        exchange.fetch_balance(params={"category": "linear"})  # Specify category for V5 balance check
        logger.success(
            f"{Fore.GREEN}{Style.BRIGHT}Portal to Bybit Opened & Authenticated (Targeting V5 API).{Style.RESET_ALL}"
        )
        logger.warning(
            f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}!!! LIVE SCALPING MODE ACTIVE - EXTREME CAUTION ADVISED !!!{Style.RESET_ALL}"
        )
        send_sms_alert(f"[Pyrmethus/{CONFIG.strategy_name}] Portal opened & authenticated.")
        return exchange

    except ccxt.AuthenticationError as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Authentication failed: {e}. Check API keys, permissions, and IP whitelist on Bybit.{Style.RESET_ALL}"
        )
        send_sms_alert(f"[Pyrmethus] CRITICAL: Authentication FAILED: {e}. Spell failed.")
    except ccxt.NetworkError as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Network disturbance during portal opening: {e}. Check internet connection and Bybit status.{Style.RESET_ALL}"
        )
        send_sms_alert(f"[Pyrmethus] CRITICAL: Network Error on Init: {e}. Spell failed.")
    except ccxt.ExchangeError as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Exchange spirit rejected portal opening: {e}. Check Bybit status, API documentation, or account status.{Style.RESET_ALL}"
        )
        send_sms_alert(f"[Pyrmethus] CRITICAL: Exchange Error on Init: {e}. Spell failed.")
    except Exception as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}Unexpected chaos during portal opening: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[Pyrmethus] CRITICAL: Unexpected Init Error: {type(e).__name__}. Spell failed.")

    return None  # Return None if initialization failed


# --- Indicator Calculation Functions - Scrying the Market ---


def calculate_supertrend(df: pd.DataFrame, length: int, multiplier: Decimal, prefix: str = "") -> pd.DataFrame:
    """Calculates the Supertrend indicator using pandas_ta, returning Decimal where applicable."""
    col_prefix = f"{prefix}" if prefix else ""
    target_cols = [f"{col_prefix}supertrend", f"{col_prefix}trend", f"{col_prefix}st_long", f"{col_prefix}st_short"]
    # pandas_ta column naming convention (may vary slightly with versions)
    st_col = f"SUPERT_{length}_{float(multiplier)}"
    st_trend_col = f"SUPERTd_{length}_{float(multiplier)}"
    st_long_col = f"SUPERTl_{length}_{float(multiplier)}"  # Long signal column
    st_short_col = f"SUPERTs_{length}_{float(multiplier)}"  # Short signal column
    raw_st_cols = [st_col, st_trend_col, st_long_col, st_short_col]  # Columns pandas_ta creates

    required_input_cols = ["high", "low", "close"]
    min_len = length + 1  # Minimum data length required

    if df is None or df.empty or not all(c in df.columns for c in required_input_cols) or len(df) < min_len:
        logger.warning(
            f"{Fore.YELLOW}Scrying ({col_prefix}ST): Insufficient data (Len: {len(df) if df is not None else 0}, Need: {min_len}).{Style.RESET_ALL}"
        )
        for col in target_cols:
            df[col] = pd.NA  # Assign NA to expected output columns
        return df

    try:
        # pandas_ta expects float multiplier for calculation and column naming
        logger.debug(f"Scrying ({col_prefix}ST): Calculating with length={length}, multiplier={float(multiplier)}")
        df.ta.supertrend(length=length, multiplier=float(multiplier), append=True)

        # Check if pandas_ta created the expected columns
        if not all(c in df.columns for c in [st_col, st_trend_col, st_long_col, st_short_col]):
            # Find which columns are missing
            missing = [c for c in raw_st_cols if c not in df.columns]
            raise KeyError(f"pandas_ta failed to create expected raw columns for {col_prefix}ST: {missing}")

        # Convert Supertrend value to Decimal, interpret trend and flips
        df[f"{col_prefix}supertrend"] = df[st_col].apply(safe_decimal_conversion)
        # Trend: 1 = Uptrend, -1 = Downtrend. Convert to boolean: True for Up, False for Down.
        df[f"{col_prefix}trend"] = df[st_trend_col] == 1
        # Flip Signals (pandas_ta provides these directly in recent versions)
        # st_long_col (SUPERTl): 1.0 when trend flips Long, NaN otherwise
        # st_short_col (SUPERTs): -1.0 when trend flips Short, NaN otherwise
        df[f"{col_prefix}st_long"] = df[st_long_col] == 1.0  # True if flipped Long this candle
        df[f"{col_prefix}st_short"] = df[st_short_col] == -1.0  # True if flipped Short this candle

        # Clean up raw columns created by pandas_ta
        df.drop(columns=raw_st_cols, errors="ignore", inplace=True)

        # Log the latest reading for debugging
        last_st_val = df[f"{col_prefix}supertrend"].iloc[-1]
        if pd.notna(last_st_val):
            last_trend = "Up" if df[f"{col_prefix}trend"].iloc[-1] else "Down"
            signal = (
                "LONG FLIP"
                if df[f"{col_prefix}st_long"].iloc[-1]
                else ("SHORT FLIP" if df[f"{col_prefix}st_short"].iloc[-1] else "Hold")
            )
            trend_color = Fore.GREEN if last_trend == "Up" else Fore.RED
            logger.debug(
                f"Scrying ({col_prefix}ST({length},{multiplier})): Trend={trend_color}{last_trend}{Style.RESET_ALL}, Val={last_st_val:.4f}, Signal={signal}"
            )
        else:
            logger.debug(f"Scrying ({col_prefix}ST({length},{multiplier})): Resulted in NA for last candle.")

    except KeyError as e:
        logger.error(
            f"{Fore.RED}Scrying ({col_prefix}ST): Error accessing column - likely pandas_ta issue or data problem: {e}{Style.RESET_ALL}"
        )
        for col in target_cols:
            df[col] = pd.NA  # Nullify results on error
    except Exception as e:
        logger.error(f"{Fore.RED}Scrying ({col_prefix}ST): Unexpected error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols:
            df[col] = pd.NA  # Nullify results on error
    return df


def analyze_volume_atr(df: pd.DataFrame, atr_len: int, vol_ma_len: int) -> dict[str, Decimal | None]:
    """Calculates ATR, Volume MA, checks spikes. Returns Dict with Decimals for volatility and volume metrics."""
    results: dict[str, Decimal | None] = {"atr": None, "volume_ma": None, "last_volume": None, "volume_ratio": None}
    required_cols = ["high", "low", "close", "volume"]
    min_len = max(atr_len, vol_ma_len) + 1  # Need at least period+1 for reliable calculation

    if df is None or df.empty or not all(c in df.columns for c in required_cols) or len(df) < min_len:
        logger.warning(
            f"{Fore.YELLOW}Scrying (Vol/ATR): Insufficient data (Len: {len(df) if df is not None else 0}, Need: {min_len}).{Style.RESET_ALL}"
        )
        return results

    try:
        # Calculate ATR (Average True Range) - Measure of volatility
        logger.debug(f"Scrying (ATR): Calculating with length={atr_len}")
        atr_col = f"ATRr_{atr_len}"  # pandas_ta standard name for ATR
        df.ta.atr(length=atr_len, append=True)
        if atr_col in df.columns:
            # Convert last ATR value to Decimal
            last_atr = df[atr_col].iloc[-1]
            if pd.notna(last_atr):
                results["atr"] = safe_decimal_conversion(last_atr)
            # Clean up the raw ATR column added by pandas_ta
            df.drop(columns=[atr_col], errors="ignore", inplace=True)
        else:
            logger.warning(f"ATR column '{atr_col}' not found after calculation.")

        # Calculate Volume Moving Average and Ratio - Measure of market energy
        logger.debug(f"Scrying (Volume): Calculating MA with length={vol_ma_len}")
        volume_ma_col = f"volume_sma_{vol_ma_len}"  # Use a distinct name
        # Use pandas rolling mean for SMA of volume
        df[volume_ma_col] = df["volume"].rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2)).mean()
        last_vol_ma = df[volume_ma_col].iloc[-1]
        last_vol = df["volume"].iloc[-1]  # Get the most recent volume bar

        if pd.notna(last_vol_ma):
            results["volume_ma"] = safe_decimal_conversion(last_vol_ma)
        if pd.notna(last_vol):
            results["last_volume"] = safe_decimal_conversion(last_vol)

        # Calculate Volume Ratio (Last Volume / Volume MA)
        if (
            results["volume_ma"] is not None
            and results["volume_ma"] > CONFIG.position_qty_epsilon
            and results["last_volume"] is not None
        ):
            try:
                results["volume_ratio"] = results["last_volume"] / results["volume_ma"]
            except DivisionByZero:
                logger.warning("Division by zero encountered calculating volume ratio (Volume MA near zero).")
                results["volume_ratio"] = None
            except Exception as ratio_err:
                logger.warning(f"Error calculating volume ratio: {ratio_err}")
                results["volume_ratio"] = None
        else:
            results["volume_ratio"] = None  # Cannot calculate ratio if MA is zero/negligible or volume is NA

        # Clean up the volume MA column
        if volume_ma_col in df.columns:
            df.drop(columns=[volume_ma_col], errors="ignore", inplace=True)

        # Log calculated results
        atr_str = f"{results['atr']:.5f}" if results["atr"] else "N/A"
        vol_ma_str = f"{results['volume_ma']:.2f}" if results["volume_ma"] else "N/A"
        last_vol_str = f"{results['last_volume']:.2f}" if results["last_volume"] else "N/A"
        vol_ratio_str = f"{results['volume_ratio']:.2f}" if results["volume_ratio"] else "N/A"
        logger.debug(
            f"Scrying Results: ATR({atr_len})={Fore.CYAN}{atr_str}{Style.RESET_ALL}, "
            f"LastVol={last_vol_str}, VolMA({vol_ma_len})={vol_ma_str}, VolRatio={Fore.YELLOW}{vol_ratio_str}{Style.RESET_ALL}"
        )

    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (Vol/ATR): Unexpected error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        results = dict.fromkeys(results)  # Nullify all results on error
    return results


def calculate_stochrsi_momentum(
    df: pd.DataFrame, rsi_len: int, stoch_len: int, k: int, d: int, mom_len: int
) -> pd.DataFrame:
    """Calculates StochRSI and Momentum, gauging overbought/oversold conditions and trend strength."""
    target_cols = ["stochrsi_k", "stochrsi_d", "momentum"]
    # StochRSI requires RSI length + Stoch length + k + d periods approximately
    min_len = max(rsi_len + stoch_len + d, mom_len) + 5  # Add buffer for calculation stability
    if df is None or df.empty or not all(c in df.columns for c in ["close"]) or len(df) < min_len:
        logger.warning(
            f"{Fore.YELLOW}Scrying (StochRSI/Mom): Insufficient data (Len: {len(df) if df is not None else 0}, Need ~{min_len}).{Style.RESET_ALL}"
        )
        for col in target_cols:
            df[col] = pd.NA
        return df
    try:
        # Calculate StochRSI using pandas_ta
        logger.debug(f"Scrying (StochRSI): Calculating with RSI={rsi_len}, Stoch={stoch_len}, K={k}, D={d}")
        # Calculate separately first to handle potential column naming issues
        stochrsi_df = df.ta.stochrsi(length=stoch_len, rsi_length=rsi_len, k=k, d=d, append=False)
        # Standard pandas_ta column names for StochRSI %K and %D
        k_col = f"STOCHRSIk_{stoch_len}_{rsi_len}_{k}_{d}"
        d_col = f"STOCHRSId_{stoch_len}_{rsi_len}_{k}_{d}"

        if k_col in stochrsi_df.columns:
            df["stochrsi_k"] = stochrsi_df[k_col].apply(safe_decimal_conversion)
        else:
            logger.warning(f"StochRSI K column '{k_col}' not found after calculation.")
            df["stochrsi_k"] = pd.NA
        if d_col in stochrsi_df.columns:
            df["stochrsi_d"] = stochrsi_df[d_col].apply(safe_decimal_conversion)
        else:
            logger.warning(f"StochRSI D column '{d_col}' not found after calculation.")
            df["stochrsi_d"] = pd.NA

        # Calculate Momentum using pandas_ta
        logger.debug(f"Scrying (Momentum): Calculating with length={mom_len}")
        mom_col = f"MOM_{mom_len}"  # Standard pandas_ta name
        df.ta.mom(length=mom_len, append=True)
        if mom_col in df.columns:
            df["momentum"] = df[mom_col].apply(safe_decimal_conversion)
            # Clean up raw momentum column
            df.drop(columns=[mom_col], errors="ignore", inplace=True)
        else:
            logger.warning(f"Momentum column '{mom_col}' not found after calculation.")
            df["momentum"] = pd.NA

        # Log latest values for debugging
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
                else (Fore.RED if mom_val < -CONFIG.position_qty_epsilon else Fore.WHITE)
            )
            logger.debug(
                f"Scrying (StochRSI/Mom): K={k_color}{k_val:.2f}{Style.RESET_ALL}, D={d_color}{d_val:.2f}{Style.RESET_ALL}, Mom={mom_color}{mom_val:.4f}{Style.RESET_ALL}"
            )
        else:
            logger.debug("Scrying (StochRSI/Mom): Resulted in NA for one or more values on last candle.")

    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (StochRSI/Mom): Unexpected error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols:
            df[col] = pd.NA  # Nullify results on error
    return df


def calculate_ehlers_fisher(df: pd.DataFrame, length: int, signal: int) -> pd.DataFrame:
    """Calculates Ehlers Fisher Transform indicator, seeking cyclical turning points."""
    target_cols = ["ehlers_fisher", "ehlers_signal"]
    required_input_cols = ["high", "low"]
    min_len = length + signal  # Need roughly length + signal periods
    if df is None or df.empty or not all(c in df.columns for c in required_input_cols) or len(df) < min_len:
        logger.warning(
            f"{Fore.YELLOW}Scrying (EhlersFisher): Insufficient data (Len: {len(df) if df is not None else 0}, Need ~{min_len}).{Style.RESET_ALL}"
        )
        for col in target_cols:
            df[col] = pd.NA
        return df
    try:
        logger.debug(f"Scrying (EhlersFisher): Calculating with length={length}, signal={signal}")
        # Calculate separately first
        fisher_df = df.ta.fisher(length=length, signal=signal, append=False)
        # Standard pandas_ta column names
        fish_col = f"FISHERT_{length}_{signal}"
        signal_col = f"FISHERTs_{length}_{signal}"

        if fish_col in fisher_df.columns:
            df["ehlers_fisher"] = fisher_df[fish_col].apply(safe_decimal_conversion)
        else:
            logger.warning(f"Ehlers Fisher column '{fish_col}' not found after calculation.")
            df["ehlers_fisher"] = pd.NA
        if signal_col in fisher_df.columns:
            df["ehlers_signal"] = fisher_df[signal_col].apply(safe_decimal_conversion)
        else:
            logger.warning(f"Ehlers Signal column '{signal_col}' not found after calculation.")
            df["ehlers_signal"] = pd.NA

        # Log latest values for debugging
        fish_val = df["ehlers_fisher"].iloc[-1]
        sig_val = df["ehlers_signal"].iloc[-1]
        if pd.notna(fish_val) and pd.notna(sig_val):
            logger.debug(
                f"Scrying (EhlersFisher({length},{signal})): Fisher={Fore.CYAN}{fish_val:.4f}{Style.RESET_ALL}, Signal={Fore.MAGENTA}{sig_val:.4f}{Style.RESET_ALL}"
            )
        else:
            logger.debug("Scrying (EhlersFisher): Resulted in NA for one or more values on last candle.")
    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (EhlersFisher): Unexpected error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols:
            df[col] = pd.NA  # Nullify results on error
    return df


def calculate_ehlers_ma(df: pd.DataFrame, fast_len: int, slow_len: int) -> pd.DataFrame:
    """Placeholder: Calculates standard EMAs instead of Ehlers Super Smoother MAs.
    Requires verification or replacement with a proper Ehlers Super Smoother implementation if strict adherence is needed.
    """
    target_cols = ["fast_ema", "slow_ema"]
    min_len = max(fast_len, slow_len) + 5  # Buffer for EMA calculation stability
    if df is None or df.empty or not all(c in df.columns for c in ["close"]) or len(df) < min_len:
        logger.warning(
            f"{Fore.YELLOW}Scrying (EhlersMA - EMA): Insufficient data (Len: {len(df) if df is not None else 0}, Need ~{min_len}).{Style.RESET_ALL}"
        )
        for col in target_cols:
            df[col] = pd.NA
        return df
    try:
        # *** PYRMETHUS NOTE / WARNING ***
        logger.warning(
            f"{Fore.YELLOW}{Style.DIM}Scrying (EhlersMA): Using standard EMA as placeholder for Ehlers Super Smoother. "
            f"This strategy path may not perform as intended. Verify indicator suitability or implement actual Ehlers Super Smoother.{Style.RESET_ALL}"
        )

        logger.debug(f"Scrying (EhlersMA - EMA Placeholder): Calculating Fast EMA({fast_len}), Slow EMA({slow_len})")
        # Use pandas_ta standard EMA calculation
        df["fast_ema"] = df.ta.ema(length=fast_len).apply(safe_decimal_conversion)
        df["slow_ema"] = df.ta.ema(length=slow_len).apply(safe_decimal_conversion)

        # Log latest values for debugging
        fast_val = df["fast_ema"].iloc[-1]
        slow_val = df["slow_ema"].iloc[-1]
        if pd.notna(fast_val) and pd.notna(slow_val):
            logger.debug(
                f"Scrying (EhlersMA({fast_len},{slow_len}) - EMA): Fast={Fore.GREEN}{fast_val:.4f}{Style.RESET_ALL}, Slow={Fore.RED}{slow_val:.4f}{Style.RESET_ALL}"
            )
        else:
            logger.debug("Scrying (EhlersMA - EMA): Resulted in NA for one or more values on last candle.")
    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (EhlersMA - EMA): Unexpected error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols:
            df[col] = pd.NA  # Nullify results on error
    return df


def analyze_order_book(exchange: ccxt.Exchange, symbol: str, depth: int, fetch_limit: int) -> dict[str, Decimal | None]:
    """Fetches and analyzes L2 order book pressure (Bid/Ask volume ratio) and spread."""
    results: dict[str, Decimal | None] = {"bid_ask_ratio": None, "spread": None, "best_bid": None, "best_ask": None}
    logger.debug(f"Order Book Scrying: Fetching L2 {symbol} (Depth:{depth}, Limit:{fetch_limit})...")

    # Check if the exchange supports fetching L2 order book
    if not exchange.has.get("fetchL2OrderBook"):
        logger.warning(
            f"{Fore.YELLOW}Order Book Scrying: fetchL2OrderBook method not supported by {exchange.id}. Cannot analyze depth.{Style.RESET_ALL}"
        )
        return results

    try:
        # Fetching the order book's current state
        order_book = exchange.fetch_l2_order_book(symbol, limit=fetch_limit)
        bids: list[list[float | str]] = order_book.get("bids", [])  # List of [price, amount]
        asks: list[list[float | str]] = order_book.get("asks", [])  # List of [price, amount]

        if not bids or not asks:
            logger.warning(
                f"{Fore.YELLOW}Order Book Scrying: Empty bids or asks received for {symbol}. Cannot analyze.{Style.RESET_ALL}"
            )
            return results

        # Extract best bid/ask with Decimal precision
        best_bid = safe_decimal_conversion(bids[0][0]) if bids and len(bids[0]) > 0 else None
        best_ask = safe_decimal_conversion(asks[0][0]) if asks and len(asks[0]) > 0 else None
        results["best_bid"] = best_bid
        results["best_ask"] = best_ask

        # Calculate spread
        if best_bid is not None and best_ask is not None and best_bid > 0 and best_ask > 0:
            results["spread"] = best_ask - best_bid
            logger.debug(
                f"OB Scrying: Best Bid={Fore.GREEN}{best_bid:.4f}{Style.RESET_ALL}, Best Ask={Fore.RED}{best_ask:.4f}{Style.RESET_ALL}, Spread={Fore.YELLOW}{results['spread']:.4f}{Style.RESET_ALL}"
            )
        else:
            logger.debug(
                f"OB Scrying: Best Bid={best_bid or 'N/A'}, Best Ask={best_ask or 'N/A'} (Spread calculation skipped)"
            )

        # Sum total volume within the specified depth using Decimal for precision
        # Ensure list slicing doesn't go out of bounds
        bid_vol = sum(safe_decimal_conversion(bid[1]) for bid in bids[: min(depth, len(bids))] if len(bid) > 1)
        ask_vol = sum(safe_decimal_conversion(ask[1]) for ask in asks[: min(depth, len(asks))] if len(ask) > 1)
        logger.debug(
            f"OB Scrying (Depth {depth}): Total BidVol={Fore.GREEN}{bid_vol:.4f}{Style.RESET_ALL}, Total AskVol={Fore.RED}{ask_vol:.4f}{Style.RESET_ALL}"
        )

        # Calculate Bid/Ask Volume Ratio (Bids / Asks)
        if ask_vol > CONFIG.position_qty_epsilon:  # Avoid division by zero or near-zero
            try:
                results["bid_ask_ratio"] = bid_vol / ask_vol
                # Determine color based on configured thresholds
                ratio_color = (
                    Fore.GREEN
                    if results["bid_ask_ratio"] >= CONFIG.order_book_ratio_threshold_long
                    else (
                        Fore.RED if results["bid_ask_ratio"] <= CONFIG.order_book_ratio_threshold_short else Fore.YELLOW
                    )
                )
                logger.debug(
                    f"OB Scrying Ratio (Bids/Asks): {ratio_color}{results['bid_ask_ratio']:.3f}{Style.RESET_ALL}"
                )
            except (DivisionByZero, Exception) as e:
                logger.warning(f"{Fore.YELLOW}Error calculating OB ratio: {e}{Style.RESET_ALL}")
                results["bid_ask_ratio"] = None
        else:
            logger.debug(
                f"OB Scrying Ratio: N/A (Ask volume within depth {depth} is zero or negligible: {ask_vol:.4f})"
            )
            results["bid_ask_ratio"] = None  # Set explicitly to None

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger.warning(f"{Fore.YELLOW}Order Book Scrying Error for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
    except IndexError:
        logger.warning(
            f"{Fore.YELLOW}Order Book Scrying Error: Index out of bounds accessing bids/asks for {symbol}. OB data might be malformed.{Style.RESET_ALL}"
        )
    except Exception as e:
        logger.warning(
            f"{Fore.YELLOW}Unexpected Order Book Scrying Error for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())

    # Ensure results dictionary keys exist even if errors occurred
    results.setdefault("bid_ask_ratio", None)
    results.setdefault("spread", None)
    results.setdefault("best_bid", None)
    results.setdefault("best_ask", None)
    return results


# --- Data Fetching - Gathering Etheric Data Streams ---
def get_market_data(exchange: ccxt.Exchange, symbol: str, interval: str, limit: int) -> pd.DataFrame | None:
    """Fetches and prepares OHLCV data as a pandas DataFrame, ensuring numeric types and handling NaNs."""
    if not exchange.has.get("fetchOHLCV"):
        logger.error(
            f"{Fore.RED}Data Fetch: Exchange '{exchange.id}' does not support fetchOHLCV method.{Style.RESET_ALL}"
        )
        return None
    try:
        logger.debug(f"Data Fetch: Gathering {limit} OHLCV candles for {symbol} ({interval})...")
        # Channeling the data stream from the exchange
        ohlcv: list[list[int | float | str]] = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)

        if not ohlcv:
            logger.warning(
                f"{Fore.YELLOW}Data Fetch: No OHLCV data returned for {symbol} ({interval}). Market might be inactive or API issue.{Style.RESET_ALL}"
            )
            return None

        # Weaving data into a DataFrame structure
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Convert timestamp to datetime objects (UTC) and set as index
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)

        # Ensure OHLCV columns are numeric, coercing errors to NaN
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # --- Robust NaN Handling ---
        initial_nan_count = df.isnull().sum().sum()
        if initial_nan_count > 0:
            nan_counts_per_col = df.isnull().sum()
            logger.warning(
                f"{Fore.YELLOW}Data Fetch: Found {initial_nan_count} NaN values in OHLCV data after conversion:\n"
                f"{nan_counts_per_col[nan_counts_per_col > 0]}\nAttempting forward fill (ffill)...{Style.RESET_ALL}"
            )
            df.ffill(inplace=True)  # Fill NaNs with the previous valid observation

            # Check if NaNs remain (likely at the beginning of the series)
            remaining_nan_count = df.isnull().sum().sum()
            if remaining_nan_count > 0:
                logger.warning(
                    f"{Fore.YELLOW}NaNs remain after ffill ({remaining_nan_count}). Attempting backward fill (bfill)...{Style.RESET_ALL}"
                )
                df.bfill(inplace=True)  # Fill remaining NaNs with the next valid observation

                # Final check: if NaNs still exist, data is likely too gappy at start/end
                final_nan_count = df.isnull().sum().sum()
                if final_nan_count > 0:
                    logger.error(
                        f"{Fore.RED}Data Fetch: Unfillable NaN values ({final_nan_count}) remain after ffill and bfill. "
                        f"Data quality insufficient for {symbol}. Skipping cycle.{Style.RESET_ALL}"
                    )
                    return None  # Cannot proceed with unreliable data

        logger.debug(f"Data Fetch: Successfully woven and cleaned {len(df)} OHLCV candles for {symbol}.")
        return df

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger.warning(
            f"{Fore.YELLOW}Data Fetch: Disturbance gathering OHLCV for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
    except Exception as e:
        logger.error(f"{Fore.RED}Data Fetch: Unexpected error processing OHLCV for {symbol}: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())

    return None  # Return None if any error occurred


# --- Position & Order Management - Manipulating Market Presence ---
def get_current_position(exchange: ccxt.Exchange, symbol: str) -> dict[str, Any]:
    """Fetches current position details using Bybit V5 API specifics (fetchPositions).
    Assumes One-Way Mode. Returns position side ('Long', 'Short', 'None'), quantity (Decimal), and entry price (Decimal).
    """
    default_pos: dict[str, Any] = {"side": CONFIG.pos_none, "qty": Decimal("0.0"), "entry_price": Decimal("0.0")}
    market_id: str | None = None
    market: dict[str, Any] | None = None

    try:
        # Get market details to determine category (linear/inverse) and ID
        market = exchange.market(symbol)
        market_id = market["id"]  # The exchange's specific ID for the symbol (e.g., BTCUSDT)
        category = "linear" if market.get("linear") else ("inverse" if market.get("inverse") else None)
        if not category:
            logger.error(
                f"{Fore.RED}Position Check: Could not determine category (linear/inverse) for market '{symbol}'.{Style.RESET_ALL}"
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
            # Attempt fetchMyTrades or other methods as fallback if needed, but V5 relies on fetchPositions
            return default_pos

        # Bybit V5 fetchPositions requires 'category' and optionally 'symbol'
        params = {"category": category, "symbol": market_id}
        logger.debug(
            f"Position Check: Querying V5 positions for {symbol} (MarketID: {market_id}, Category: {category})..."
        )

        # Summon position data from the exchange
        fetched_positions = exchange.fetch_positions(
            symbols=[symbol], params=params
        )  # Pass symbol for filtering if supported

        # Bybit V5 fetchPositions returns a list. In One-Way mode, we look for the entry with positionIdx = 0.
        # The 'side' field in the info dict indicates 'Buy' (Long) or 'Sell' (Short) if a position exists.
        active_pos_info = None
        for pos in fetched_positions:
            pos_info = pos.get("info", {})
            pos_market_id = pos_info.get("symbol")
            position_idx = int(pos_info.get("positionIdx", -1))  # Default to -1 if missing
            pos_side_v5 = pos_info.get("side", "None")  # 'Buy', 'Sell', or 'None'
            size_str = pos_info.get("size", "0")  # Position size as string

            # Check if this entry matches our symbol and is the active One-Way position
            if pos_market_id == market_id and position_idx == 0:
                # Check if size is non-zero and side indicates an open position
                size_dec = safe_decimal_conversion(size_str)
                if abs(size_dec) > CONFIG.position_qty_epsilon and pos_side_v5 != "None":
                    active_pos_info = pos_info  # Found the active position data
                    logger.debug(
                        f"Found active V5 position candidate: Idx={position_idx}, Side={pos_side_v5}, Size={size_str}"
                    )
                    break  # Assume only one active position in One-Way mode

        if active_pos_info:
            try:
                # Parse details from the active position info
                size = safe_decimal_conversion(active_pos_info.get("size"))
                # Use 'avgPrice' from V5 info dict for the entry price
                entry_price = safe_decimal_conversion(active_pos_info.get("avgPrice"))
                # Determine side based on V5 'side' field ('Buy' or 'Sell')
                side = CONFIG.pos_long if active_pos_info.get("side") == "Buy" else CONFIG.pos_short

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
                return default_pos  # Return default on parsing error
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

    return default_pos  # Return default on API error or if no active position found


def set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int) -> bool:
    """Sets leverage for a futures symbol using Bybit V5 API specifics."""
    logger.info(f"{Fore.CYAN}Leverage Conjuring: Attempting to set {leverage}x for {symbol}...{Style.RESET_ALL}")
    try:
        # Verify it's a contract market where leverage applies
        market = exchange.market(symbol)
        if not market.get("contract"):
            logger.error(
                f"{Fore.RED}Leverage Conjuring Error: Cannot set leverage for non-contract market: {symbol}.{Style.RESET_ALL}"
            )
            return False
        # Bybit V5 requires setting buy and sell leverage separately for the symbol
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

    for attempt in range(CONFIG.retry_count + 1):  # Add 1 to retry_count for number of attempts
        try:
            response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
            # Response format varies, log it for debugging. Success is usually indicated by lack of error.
            logger.success(
                f"{Fore.GREEN}Leverage Conjuring: Successfully set to {leverage}x for {symbol}. Response: {response}{Style.RESET_ALL}"
            )
            return True
        except ccxt.ExchangeError as e:
            # Check for common Bybit errors indicating leverage is already set or not modified
            err_str = str(e).lower()
            # Example error codes/messages from Bybit V5 (adjust if needed):
            # 110044: Leverage not modified
            # 110025: Leverage cannot be lower than 1
            # (Add other relevant codes/messages here)
            if "leverage not modified" in err_str or "same leverage" in err_str or "110044" in err_str:
                logger.info(
                    f"{Fore.CYAN}Leverage Conjuring: Leverage already set to {leverage}x for {symbol}.{Style.RESET_ALL}"
                )
                return True
            elif "cannot be lower than 1" in err_str and leverage < 1:
                logger.error(
                    f"{Fore.RED}Leverage Conjuring: Invalid leverage value ({leverage}) requested.{Style.RESET_ALL}"
                )
                return False  # Don't retry invalid value

            # Log other exchange errors and decide whether to retry
            logger.warning(
                f"{Fore.YELLOW}Leverage Conjuring: Exchange resistance (Attempt {attempt + 1}/{CONFIG.retry_count + 1}): {e}{Style.RESET_ALL}"
            )
            if attempt >= CONFIG.retry_count:
                logger.error(
                    f"{Fore.RED}Leverage Conjuring: Failed after {CONFIG.retry_count + 1} attempts.{Style.RESET_ALL}"
                )
                break  # Exit loop after final attempt
            time.sleep(CONFIG.retry_delay_seconds * (attempt + 1))  # Exponential backoff might be better

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
            # Decide if this is retryable or fatal
            if attempt >= CONFIG.retry_count:
                logger.error(
                    f"{Fore.RED}Leverage Conjuring: Failed due to unexpected error after {CONFIG.retry_count + 1} attempts.{Style.RESET_ALL}"
                )
                break
            time.sleep(CONFIG.retry_delay_seconds * (attempt + 1))

    return False  # Failed to set leverage after all attempts


def close_position(
    exchange: ccxt.Exchange, symbol: str, position_to_close: dict[str, Any], reason: str = "Signal"
) -> dict[str, Any] | None:
    """Closes the specified active position by placing a market order with reduceOnly=True.
    Re-validates the position just before closing.
    """
    initial_side = position_to_close.get("side", CONFIG.pos_none)
    initial_qty = position_to_close.get("qty", Decimal("0.0"))
    market_base = symbol.split("/")[0].split(":")[0]  # For concise alerts (e.g., BTC from BTC/USDT:USDT)
    logger.info(
        f"{Fore.YELLOW}Banish Position Ritual: Initiated for {symbol}. Reason: {reason}. Initial state: {initial_side} Qty={initial_qty:.8f}{Style.RESET_ALL}"
    )

    # --- Re-validate the position just before closing ---
    logger.debug("Banish Position: Re-validating live position state...")
    live_position = get_current_position(exchange, symbol)
    live_position_side = live_position["side"]
    live_amount_to_close = live_position["qty"]

    if live_position_side == CONFIG.pos_none or live_amount_to_close <= CONFIG.position_qty_epsilon:
        logger.warning(
            f"{Fore.YELLOW}Banish Position: Re-validation shows NO active position (or negligible size: {live_amount_to_close:.8f}) for {symbol}. Aborting banishment.{Style.RESET_ALL}"
        )
        if initial_side != CONFIG.pos_none:
            # This indicates a potential state discrepancy between cycles
            logger.warning(
                f"{Fore.YELLOW}Banish Position: Discrepancy detected! Initial check showed {initial_side}, but now it's None/Zero.{Style.RESET_ALL}"
            )
        return None  # Nothing to close

    # Determine the opposite side needed to close the position
    side_to_execute_close = CONFIG.side_sell if live_position_side == CONFIG.pos_long else CONFIG.side_buy

    try:
        # Format amount according to market rules
        amount_str = format_amount(exchange, symbol, live_amount_to_close)
        amount_dec = Decimal(amount_str)  # Convert formatted string back to Decimal for check
        amount_float = float(amount_dec)  # CCXT create order often expects float

        # Check if the amount is valid after formatting
        if amount_dec <= CONFIG.position_qty_epsilon:
            logger.error(
                f"{Fore.RED}Banish Position: Closing amount negligible ({amount_str}) after precision shaping. Cannot close. Manual check advised.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] ERROR Banishing ({reason}): Negligible close amount {amount_str}. MANUAL CHECK!"
            )
            return None

        # Execute the market close order with reduceOnly flag
        close_color = Back.YELLOW
        logger.warning(
            f"{close_color}{Fore.BLACK}{Style.BRIGHT}Banish Position: Attempting to CLOSE {live_position_side} ({reason}): "
            f"Exec {side_to_execute_close.upper()} MARKET {amount_str} {symbol} (reduce_only=True)...{Style.RESET_ALL}"
        )
        params = {"reduceOnly": True}  # Crucial: ensures this order only reduces/closes position
        order = exchange.create_market_order(
            symbol=symbol, side=side_to_execute_close, amount=amount_float, params=params
        )

        # Order placed, parse response safely using Decimal
        # Note: 'average' might be None immediately for market orders, 'price' might be 0.
        # Rely on 'filled' and 'cost' primarily. Fetch order later if exact avg price needed.
        fill_price_avg = safe_decimal_conversion(order.get("average"))  # May be None initially
        filled_qty = safe_decimal_conversion(order.get("filled"))
        cost = safe_decimal_conversion(order.get("cost"))
        order_id_short = format_order_id(order.get("id"))
        status = order.get("status", "unknown")

        # Log success based on filled quantity vs expected quantity
        if abs(filled_qty - amount_dec) < CONFIG.position_qty_epsilon:
            logger.success(
                f"{Fore.GREEN}{Style.BRIGHT}Banish Position: Order ({reason}) appears FILLED for {symbol}. "
                f"Filled: {filled_qty:.8f}, AvgFill: {fill_price_avg:.4f}, Cost: {cost:.2f} USDT. ID:...{order_id_short}, Status: {status}{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] BANISHED {live_position_side} {amount_str} @ ~{fill_price_avg:.4f} ({reason}). ID:...{order_id_short}"
            )
            return order  # Return the order details
        else:
            # Partial fill or zero fill on market close order is unusual but possible
            logger.warning(
                f"{Fore.YELLOW}Banish Position: Order ({reason}) status uncertain. Expected {amount_dec:.8f}, Filled: {filled_qty:.8f}. "
                f"ID:...{order_id_short}, Status: {status}. Re-checking position state soon.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] WARNING Banishing ({reason}): Fill uncertain (Exp:{amount_dec:.8f}, Got:{filled_qty:.8f}). ID:...{order_id_short}"
            )
            # Return order details, but the caller should re-verify position state
            return order

    except ccxt.InsufficientFunds as e:
        logger.error(
            f"{Fore.RED}Banish Position ({reason}): Insufficient funds for {symbol}: {e}. This might indicate margin issues or incorrect state.{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[{market_base}/{CONFIG.strategy_name}] ERROR Banishing ({reason}): Insufficient Funds! Check account."
        )
    except ccxt.ExchangeError as e:
        # Check for specific Bybit V5 errors indicating position already closed or issues with reduceOnly
        err_str = str(e).lower()
        # Example error codes/messages (adjust based on Bybit V5 docs):
        # 110007: Order quantity exceeds open position size
        # 110015: Reduce-only rule violation
        # 1100XX: Position size related errors
        # Check for messages indicating the position is already gone or the order wouldn't reduce it
        if (
            "order quantity exceeds open position size" in err_str
            or "position is zero" in err_str
            or "position size is zero" in err_str
            or "order would not reduce position size" in err_str
            or "110007" in err_str
            or "110015" in err_str
        ):  # Add relevant error codes
            logger.warning(
                f"{Fore.YELLOW}Banish Position: Exchange indicates position likely already closed/closing or zero size ({e}). Assuming banished.{Style.RESET_ALL}"
            )
            return None  # Treat as success (or non-actionable) in this case
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

    return None  # Indicate failure to close if an error occurred


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

    # --- Input Validation ---
    if not (entry_price > 0 and stop_loss_price > 0):
        logger.error(
            f"{Fore.RED}Risk Calc Error: Invalid entry/SL price (<= 0). Entry={entry_price}, SL={stop_loss_price}.{Style.RESET_ALL}"
        )
        return None, None
    price_diff = abs(entry_price - stop_loss_price)
    if price_diff < CONFIG.position_qty_epsilon:  # Use epsilon for comparison
        logger.error(
            f"{Fore.RED}Risk Calc Error: Entry and SL prices are too close ({price_diff:.8f}). Cannot calculate safe size.{Style.RESET_ALL}"
        )
        return None, None
    if not 0 < risk_per_trade_pct < 1:
        logger.error(
            f"{Fore.RED}Risk Calc Error: Invalid risk percentage: {risk_per_trade_pct:.4%}. Must be between 0 and 1 (e.g., 0.01 for 1%).{Style.RESET_ALL}"
        )
        return None, None
    if equity <= 0:
        logger.error(
            f"{Fore.RED}Risk Calc Error: Invalid equity: {equity:.4f}. Cannot calculate risk.{Style.RESET_ALL}"
        )
        return None, None
    if leverage <= 0:
        logger.error(f"{Fore.RED}Risk Calc Error: Invalid leverage: {leverage}. Must be > 0.{Style.RESET_ALL}")
        return None, None

    # --- Calculation ---
    try:
        risk_amount_usdt = equity * risk_per_trade_pct  # Max USDT amount to risk on this trade
        # Assuming linear contract where 1 unit = 1 base currency (e.g., 1 BTC)
        # Risk per unit of the asset = price_diff (difference between entry and stop loss in USDT)
        # Quantity = Total Risk Amount (USDT) / Risk per Unit (USDT)
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

    # --- Apply Market Precision using CCXT formatter ---
    try:
        # Format according to market precision *then* convert back to Decimal
        quantity_precise_str = format_amount(exchange, symbol, quantity_raw)
        quantity_precise = Decimal(quantity_precise_str)
        logger.debug(f"Risk Calc: Precise Qty after formatting={quantity_precise:.8f}")
    except Exception as fmt_err:
        logger.warning(
            f"{Fore.YELLOW}Risk Calc Warning: Failed precision shaping for quantity {quantity_raw:.8f}. Error: {fmt_err}. Using raw value with fallback quantization.{Style.RESET_ALL}"
        )
        # Fallback: Quantize raw value to a reasonable number of decimal places if formatting fails
        # Determine appropriate decimal places based on market or a default (e.g., 8)
        # For simplicity, using 8 decimal places as a fallback. A better approach might inspect market['precision']['amount']
        try:
            quantity_precise = quantity_raw.quantize(Decimal("1e-8"), rounding=ROUND_HALF_UP)
            logger.debug(f"Risk Calc: Fallback Quantized Qty={quantity_precise:.8f}")
        except Exception as q_err:
            logger.error(
                f"{Fore.RED}Risk Calc Error: Failed fallback quantization for quantity {quantity_raw:.8f}: {q_err}{Style.RESET_ALL}"
            )
            return None, None

    # --- Final Checks & Margin Estimation ---
    if quantity_precise <= CONFIG.position_qty_epsilon:
        logger.warning(
            f"{Fore.YELLOW}Risk Calc Warning: Calculated quantity negligible ({quantity_precise:.8f}) after formatting/quantization. "
            f"RiskAmt={risk_amount_usdt:.4f}, PriceDiff={price_diff:.4f}. Cannot place order.{Style.RESET_ALL}"
        )
        return None, None

    # Estimate position value and margin required based on the precise quantity
    try:
        pos_value_usdt = quantity_precise * entry_price
        required_margin = pos_value_usdt / Decimal(leverage)
        logger.debug(
            f"Risk Calc Result: Qty={Fore.CYAN}{quantity_precise:.8f}{Style.RESET_ALL}, Est. Pos Value={pos_value_usdt:.4f} USDT, Est. Margin Req.={required_margin:.4f} USDT"
        )
        return quantity_precise, required_margin
    except (DivisionByZero, Exception) as margin_err:
        logger.error(f"{Fore.RED}Risk Calc Error: Failed calculating estimated margin: {margin_err}{Style.RESET_ALL}")
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
            # Query the order's current status
            # Note: fetchOrder might require params for specific order types on some exchanges/versions
            order = exchange.fetch_order(order_id, symbol)
            status = order.get("status")
            filled_qty = safe_decimal_conversion(order.get("filled"))
            logger.debug(f"Order ...{order_id_short} status: {status}, Filled: {filled_qty:.8f}")

            if (
                status == "closed"
            ):  # 'closed' typically means fully filled for market orders, or fully executed for limit/stop
                logger.success(f"{Fore.GREEN}Order ...{order_id_short} confirmed FILLED/CLOSED.{Style.RESET_ALL}")
                return order  # Success, return final order state
            elif status in ["canceled", "rejected", "expired"]:
                logger.error(f"{Fore.RED}Order ...{order_id_short} reached FAILED status: '{status}'.{Style.RESET_ALL}")
                return order  # Failed state, return final order state
            # Continue polling if 'open', 'partially_filled', or None/unknown status

            time.sleep(0.75)  # Check slightly less frequently than every 500ms

        except ccxt.OrderNotFound:
            # This can happen briefly after placing, especially on busy exchanges. Keep trying within timeout.
            elapsed = time.time() - start_time
            logger.warning(
                f"{Fore.YELLOW}Order ...{order_id_short} not found yet by exchange spirits ({elapsed:.1f}s elapsed). Retrying...{Style.RESET_ALL}"
            )
            # Don't sleep excessively here, just continue the loop after a short pause
            time.sleep(1.5)  # Wait a bit longer if not found initially
        except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
            elapsed = time.time() - start_time
            logger.warning(
                f"{Fore.YELLOW}Disturbance checking order ...{order_id_short} ({elapsed:.1f}s elapsed): {type(e).__name__} - {e}. Retrying...{Style.RESET_ALL}"
            )
            time.sleep(2)  # Wait longer on error before retrying
        except Exception as e_unexp:
            elapsed = time.time() - start_time
            logger.error(
                f"{Fore.RED}Unexpected error checking order ...{order_id_short} ({elapsed:.1f}s elapsed): {e_unexp}. Stopping check.{Style.RESET_ALL}"
            )
            logger.debug(traceback.format_exc())
            return None  # Stop checking on unexpected error

    # If the loop finishes without returning, it timed out
    logger.error(
        f"{Fore.RED}Order ...{order_id_short} did NOT reach final status within {timeout_seconds}s timeout.{Style.RESET_ALL}"
    )
    # Optionally try fetching one last time outside the loop
    try:
        final_check_order = exchange.fetch_order(order_id, symbol)
        logger.warning(f"Final status check for timed-out order ...{order_id_short}: {final_check_order.get('status')}")
        return final_check_order  # Return whatever the final status was
    except Exception as final_e:
        logger.error(f"Failed final status check for timed-out order ...{order_id_short}: {final_e}")
        return None  # Timeout failure


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
    """Core order placement logic:
    1. Calculates size based on risk and ATR-based SL estimate.
    2. Caps size if needed.
    3. Checks margin.
    4. Places MARKET entry order.
    5. Waits for fill confirmation.
    6. Calculates ACTUAL SL price based on fill.
    7. Places exchange-native FIXED SL order (stopMarket).
    8. Places exchange-native TRAILING SL order (stopMarket with trailing params).
    Returns the filled entry order dict on success, None on failure at any critical step.
    """
    market_base = symbol.split("/")[0].split(":")[0]  # For concise alerts
    order_side_color = Fore.GREEN if side == CONFIG.side_buy else Fore.RED
    logger.info(
        f"{order_side_color}{Style.BRIGHT}Place Order Ritual: Initiating {side.upper()} for {symbol}...{Style.RESET_ALL}"
    )

    # --- Pre-computation & Validation ---
    if current_atr is None or current_atr <= Decimal("0"):
        logger.error(
            f"{Fore.RED}Place Order Error ({side.upper()}): Invalid ATR ({current_atr}) provided. Cannot calculate SL distance.{Style.RESET_ALL}"
        )
        return None

    entry_price_estimate: Decimal | None = None
    initial_sl_price_estimate: Decimal | None = None
    final_quantity: Decimal | None = None
    market: dict | None = None
    min_qty: Decimal | None = None
    min_price: Decimal | None = None

    try:
        # === 1. Gather Resources: Balance, Market Info, Limits ===
        logger.debug("Gathering resources: Balance, Market Structure, Limits...")
        # Fetch available balance (ensure correct category for Bybit V5)
        balance_params = {"category": "linear"}  # Assuming linear for USDT futures
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

        # Extract USDT balance details (use 'total' for equity, 'free' for margin check)
        usdt_balance = balance.get(CONFIG.usdt_symbol, {})
        usdt_total = safe_decimal_conversion(usdt_balance.get("total"))  # Total equity (incl. PnL)
        usdt_free = safe_decimal_conversion(usdt_balance.get("free"))  # Available for new orders/margin
        # Use total equity for risk calculation if available and positive, otherwise fall back to free (less ideal)
        usdt_equity = usdt_total if usdt_total is not None and usdt_total > 0 else usdt_free

        if usdt_equity is None or usdt_equity <= Decimal("0"):
            logger.error(
                f"{Fore.RED}Place Order Error ({side.upper()}): Zero or Invalid Equity ({usdt_equity}). Cannot calculate risk.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Zero/Invalid Equity ({usdt_equity:.2f})"
            )
            return None
        if usdt_free is None or usdt_free < Decimal("0"):  # Free margin shouldn't be negative
            logger.error(
                f"{Fore.RED}Place Order Error ({side.upper()}): Invalid Free Margin ({usdt_free}). Cannot place orders.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Invalid Free Margin ({usdt_free:.2f})"
            )
            return None
        logger.debug(f"Resources: Equity={usdt_equity:.4f}, Free Margin={usdt_free:.4f} {CONFIG.usdt_symbol}")
        if min_qty:
            logger.debug(f"Market Limits: Min Qty={min_qty:.8f}")
        if max_qty:
            logger.debug(f"Market Limits: Max Qty={max_qty:.8f}")
        if min_price:
            logger.debug(f"Market Limits: Min Price={min_price:.8f}")

        # === 2. Estimate Entry Price - Peering into the immediate future ===
        logger.debug("Estimating entry price using shallow order book...")
        ob_data = analyze_order_book(exchange, symbol, CONFIG.shallow_ob_fetch_depth, CONFIG.shallow_ob_fetch_depth)
        best_ask = ob_data.get("best_ask")
        best_bid = ob_data.get("best_bid")
        if side == CONFIG.side_buy and best_ask and best_ask > 0:
            entry_price_estimate = best_ask  # Estimate buying at best ask
        elif side == CONFIG.side_sell and best_bid and best_bid > 0:
            entry_price_estimate = best_bid  # Estimate selling at best bid
        else:
            # Fallback: Fetch last traded price if OB data is unreliable
            logger.warning(
                f"{Fore.YELLOW}Shallow OB failed for entry price estimate (Ask:{best_ask}, Bid:{best_bid}). Fetching ticker...{Style.RESET_ALL}"
            )
            try:
                ticker = exchange.fetch_ticker(symbol)
                last_price = safe_decimal_conversion(ticker.get("last"))
                if last_price > 0:
                    entry_price_estimate = last_price
                    logger.debug(f"Using ticker last price for estimate: {entry_price_estimate}")
                else:
                    raise ValueError("Ticker price invalid")
            except (ccxt.NetworkError, ccxt.ExchangeError, ValueError, Exception) as e:
                logger.error(
                    f"{Fore.RED}Place Order Error ({side.upper()}): Failed to get valid entry price estimate from OB or Ticker: {e}{Style.RESET_ALL}"
                )
                return None
        logger.debug(f"Estimated Entry Price ~ {entry_price_estimate:.4f}")

        # === 3. Calculate Initial Stop Loss Price (Estimate) - The First Ward ===
        sl_distance = current_atr * sl_atr_multiplier
        initial_sl_price_raw = (
            (entry_price_estimate - sl_distance) if side == CONFIG.side_buy else (entry_price_estimate + sl_distance)
        )

        # Ensure SL price estimate is valid and respects minimum price if applicable
        if initial_sl_price_raw <= 0:
            logger.error(
                f"{Fore.RED}Place Order Error ({side.upper()}): Initial SL price calculation resulted in zero or negative value ({initial_sl_price_raw:.4f}). Cannot proceed.{Style.RESET_ALL}"
            )
            return None
        if min_price is not None and initial_sl_price_raw < min_price:
            logger.warning(
                f"{Fore.YELLOW}Initial SL price estimate {initial_sl_price_raw:.4f} is below market min price {min_price}. Adjusting SL estimate to min price.{Style.RESET_ALL}"
            )
            initial_sl_price_raw = min_price  # Adjust SL estimate upwards to meet minimum

        # Format the estimated SL price according to market rules
        try:
            initial_sl_price_estimate_str = format_price(exchange, symbol, initial_sl_price_raw)
            initial_sl_price_estimate = Decimal(initial_sl_price_estimate_str)
            if initial_sl_price_estimate <= 0:
                raise ValueError("Formatted SL price estimate invalid")
            logger.info(
                f"Calculated Initial SL Price (Estimate) ~ {Fore.YELLOW}{initial_sl_price_estimate:.4f}{Style.RESET_ALL} (ATR: {current_atr:.4f}, Multiplier: {sl_atr_multiplier}, Dist: {sl_distance:.4f})"
            )
        except (ValueError, InvalidOperation) as e:
            logger.error(
                f"{Fore.RED}Place Order Error ({side.upper()}): Failed to format/validate initial SL price estimate '{initial_sl_price_raw:.4f}': {e}{Style.RESET_ALL}"
            )
            return None

        # === 4. Calculate Position Size - Determining the Energy Input ===
        calc_qty, req_margin_est = calculate_position_size(
            usdt_equity, risk_percentage, entry_price_estimate, initial_sl_price_estimate, leverage, symbol, exchange
        )
        if calc_qty is None or req_margin_est is None:
            logger.error(
                f"{Fore.RED}Place Order Error ({side.upper()}): Failed risk calculation. Cannot determine position size.{Style.RESET_ALL}"
            )
            return None
        final_quantity = calc_qty  # Start with the risk-based quantity

        # === 5. Apply Max Order Cap - Limiting the Power ===
        pos_value_estimate = final_quantity * entry_price_estimate
        logger.debug(
            f"Estimated position value based on risk calc: {pos_value_estimate:.4f} USDT (Cap: {max_order_cap_usdt:.4f} USDT)"
        )
        if pos_value_estimate > max_order_cap_usdt:
            logger.warning(
                f"{Fore.YELLOW}Calculated order value {pos_value_estimate:.4f} exceeds Max Cap {max_order_cap_usdt:.4f}. Capping quantity.{Style.RESET_ALL}"
            )
            try:
                final_quantity_capped = max_order_cap_usdt / entry_price_estimate
                # Format the capped quantity according to market rules *then* convert back
                final_quantity_str = format_amount(exchange, symbol, final_quantity_capped)
                final_quantity = Decimal(final_quantity_str)
                # Recalculate estimated margin based on the capped quantity
                req_margin_est = final_quantity * entry_price_estimate / Decimal(leverage)
                logger.info(
                    f"Quantity capped to: {final_quantity:.8f}, New Est. Margin Req.: {req_margin_est:.4f} USDT"
                )
            except (DivisionByZero, ValueError, InvalidOperation, Exception) as cap_err:
                logger.error(
                    f"{Fore.RED}Place Order Error ({side.upper()}): Failed to calculate or format capped quantity: {cap_err}{Style.RESET_ALL}"
                )
                return None

        # === 6. Check Limits & Margin Availability - Final Preparations ===
        if final_quantity <= CONFIG.position_qty_epsilon:
            logger.error(
                f"{Fore.RED}Place Order Error ({side.upper()}): Final Quantity negligible ({final_quantity:.8f}) after risk calc/capping. Cannot place order.{Style.RESET_ALL}"
            )
            return None
        # Check against minimum order size
        if min_qty is not None and final_quantity < min_qty:
            logger.error(
                f"{Fore.RED}Place Order Error ({side.upper()}): Final Quantity {final_quantity:.8f} is less than market minimum allowed {min_qty:.8f}. Cannot place order.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Qty {final_quantity:.8f} < Min {min_qty:.8f}"
            )
            return None
        # Check against maximum order size (should be handled by cap, but double-check)
        if max_qty is not None and final_quantity > max_qty:
            logger.warning(
                f"{Fore.YELLOW}Final Quantity {final_quantity:.8f} exceeds market maximum {max_qty:.8f}. Adjusting down to max allowed.{Style.RESET_ALL}"
            )
            final_quantity = max_qty
            # Re-format capped amount one last time
            try:
                final_quantity_str = format_amount(exchange, symbol, final_quantity)
                final_quantity = Decimal(final_quantity_str)
                # Recalculate margin again if qty changed due to max limit
                req_margin_est = final_quantity * entry_price_estimate / Decimal(leverage)
            except Exception as max_fmt_err:
                logger.error(
                    f"{Fore.RED}Place Order Error ({side.upper()}): Failed to format max-capped quantity: {max_fmt_err}{Style.RESET_ALL}"
                )
                return None

        # Final margin calculation based on potentially adjusted final_quantity
        # Use the most up-to-date req_margin_est calculated above
        req_margin_buffered = req_margin_est * margin_check_buffer  # Add safety buffer
        logger.debug(
            f"Final Margin Check: Need ~{req_margin_est:.4f} (Buffered: {req_margin_buffered:.4f}), Have Free: {usdt_free:.4f}"
        )

        # Check if sufficient free margin is available
        if usdt_free < req_margin_buffered:
            logger.error(
                f"{Fore.RED}Place Order Error ({side.upper()}): Insufficient FREE margin. Need ~{req_margin_buffered:.4f} (incl. {margin_check_buffer:.1%} buffer), Have {usdt_free:.4f}{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Insufficient Free Margin (Need ~{req_margin_buffered:.2f}, Have {usdt_free:.2f})"
            )
            return None
        logger.info(
            f"{Fore.GREEN}Final Order Details Pre-Submission: Side={side.upper()}, Qty={final_quantity:.8f}, EstValue={final_quantity * entry_price_estimate:.4f}, EstMargin={req_margin_est:.4f}. Margin check OK.{Style.RESET_ALL}"
        )

        # === 7. Place Entry Market Order - Unleashing the Energy ===
        entry_order: dict[str, Any] | None = None
        order_id: str | None = None
        try:
            qty_float = float(final_quantity)  # CCXT expects float for amount
            entry_bg_color = Back.GREEN if side == CONFIG.side_buy else Back.RED
            entry_fg_color = Fore.BLACK if side == CONFIG.side_buy else Fore.WHITE
            logger.warning(
                f"{entry_bg_color}{entry_fg_color}{Style.BRIGHT}*** Placing {side.upper()} MARKET ENTRY Order: {qty_float:.8f} {symbol} ***{Style.RESET_ALL}"
            )

            # Create the market order (ensure it's NOT reduceOnly for entry)
            entry_order = exchange.create_market_order(
                symbol=symbol, side=side, amount=qty_float, params={"reduceOnly": False}
            )
            order_id = entry_order.get("id")

            if not order_id:
                # This is highly unexpected and problematic if the order was actually placed
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}CRITICAL FAILURE: Entry order potentially placed but NO Order ID received from exchange! Position state unknown. MANUAL INTERVENTION REQUIRED! Response: {entry_order}{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"[{market_base}/{CONFIG.strategy_name}] CRITICAL ORDER FAIL ({side.upper()}): Entry placed but NO ID received! MANUAL CHECK!"
                )
                # Cannot proceed reliably without the order ID to track fill and place SL/TSL
                return None
            logger.success(
                f"{Fore.GREEN}Market Entry Order submitted. ID: ...{format_order_id(order_id)}. Awaiting fill confirmation...{Style.RESET_ALL}"
            )

        except (ccxt.InsufficientFunds, ccxt.ExchangeError, ccxt.NetworkError) as e:
            logger.error(
                f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}FAILED TO PLACE ENTRY ORDER: {type(e).__name__} - {e}{Style.RESET_ALL}"
            )
            logger.debug(traceback.format_exc())
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Entry placement failed: {type(e).__name__}"
            )
            return None  # Failed to place entry, cannot proceed
        except Exception as e_place:
            logger.error(
                f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}UNEXPECTED ERROR Placing Entry Order: {e_place}{Style.RESET_ALL}"
            )
            logger.debug(traceback.format_exc())
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Unexpected entry placement error: {type(e_place).__name__}"
            )
            return None

        # === 8. Wait for Entry Fill Confirmation - Observing the Impact ===
        filled_entry = wait_for_order_fill(exchange, order_id, symbol, CONFIG.order_fill_timeout_seconds)
        if not filled_entry or filled_entry.get("status") != "closed":
            # Handle timeout or failure status from wait_for_order_fill
            final_status = filled_entry.get("status") if filled_entry else "timeout"
            logger.error(
                f"{Fore.RED}Entry order ...{format_order_id(order_id)} did not confirm filled (Status: {final_status}). Position state uncertain.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Entry ...{format_order_id(order_id)} fill timeout/fail (Status: {final_status}). MANUAL CHECK!"
            )
            # Attempt to cancel the potentially stuck/unfilled order
            try:
                logger.warning(
                    f"Attempting to cancel potentially stuck/unconfirmed order ...{format_order_id(order_id)}"
                )
                exchange.cancel_order(order_id, symbol)
                logger.info(f"Cancel request sent for order ...{format_order_id(order_id)}")
            except Exception as cancel_e:
                logger.warning(
                    f"Could not cancel order ...{format_order_id(order_id)} (may be filled, already gone, or error): {cancel_e}"
                )
            # Even if cancel fails, we cannot proceed to SL/TSL placement without confirmed entry
            return None

        # === 9. Extract Actual Fill Details - Reading the Result ===
        # Use 'average' if available and valid, otherwise calculate from cost/filled if possible
        avg_fill_price = safe_decimal_conversion(filled_entry.get("average"))
        filled_qty = safe_decimal_conversion(filled_entry.get("filled"))
        cost = safe_decimal_conversion(filled_entry.get("cost"))  # Total cost in quote currency (USDT)

        # Validate fill details - ensure quantity is non-zero and price is sensible
        if filled_qty <= CONFIG.position_qty_epsilon:
            logger.error(
                f"{Back.RED}{Fore.WHITE}CRITICAL: Invalid fill quantity ({filled_qty}) for order ...{format_order_id(order_id)}. Position state unknown! MANUAL CHECK REQUIRED!{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] CRITICAL ORDER FAIL ({side.upper()}): Invalid fill qty {filled_qty} for ...{format_order_id(order_id)}! MANUAL CHECK!"
            )
            return filled_entry  # Return problematic order details, but signal failure state

        if avg_fill_price <= 0 and cost > 0 and filled_qty > 0:
            # Try to estimate fill price from cost/qty if average is missing/zero
            avg_fill_price = cost / filled_qty
            logger.warning(
                f"{Fore.YELLOW}Fill price 'average' was missing/zero. Estimated from cost/filled: {avg_fill_price:.4f}{Style.RESET_ALL}"
            )

        if avg_fill_price <= 0:
            logger.error(
                f"{Back.RED}{Fore.WHITE}CRITICAL: Invalid fill price ({avg_fill_price}) for order ...{format_order_id(order_id)}. Cannot calculate SL accurately! MANUAL CHECK REQUIRED!{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] CRITICAL ORDER FAIL ({side.upper()}): Invalid fill price {avg_fill_price} for ...{format_order_id(order_id)}! MANUAL CHECK!"
            )
            # Position is likely open but SL cannot be placed reliably. Return filled entry but signal issue.
            return filled_entry  # Allow shutdown handler to potentially close, but mark this step as problematic

        fill_bg_color = Back.GREEN if side == CONFIG.side_buy else Back.RED
        fill_fg_color = Fore.BLACK if side == CONFIG.side_buy else Fore.WHITE
        logger.success(
            f"{fill_bg_color}{fill_fg_color}{Style.BRIGHT}ENTRY CONFIRMED: ...{format_order_id(order_id)}. Filled: {filled_qty:.8f} @ Avg: {avg_fill_price:.4f}, Cost: {cost:.4f} USDT{Style.RESET_ALL}"
        )

        # === 10. Calculate ACTUAL Stop Loss Price - Setting the Ward ===
        # Use the actual average fill price and the original ATR distance for SL calculation
        actual_sl_price_raw = (
            (avg_fill_price - sl_distance) if side == CONFIG.side_buy else (avg_fill_price + sl_distance)
        )

        # Apply min price constraint again based on actual fill
        if actual_sl_price_raw <= 0:
            logger.error(
                f"{Back.RED}{Fore.WHITE}CRITICAL FAILURE: ACTUAL SL price calculation resulted in zero or negative value ({actual_sl_price_raw:.4f}) based on fill price {avg_fill_price:.4f}. Cannot place SL!{Style.RESET_ALL}"
            )
            # Position is open without SL protection. Attempt emergency close.
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] CRITICAL ({side.upper()}): Invalid ACTUAL SL price ({actual_sl_price_raw:.4f})! Attempting emergency close."
            )
            # Use the confirmed fill details to attempt closure
            close_position(exchange, symbol, {"side": side, "qty": filled_qty}, reason="Invalid SL Calc Post-Entry")
            return filled_entry  # Return filled entry, but signal overall failure state

        if min_price is not None and actual_sl_price_raw < min_price:
            logger.warning(
                f"{Fore.YELLOW}Actual SL price {actual_sl_price_raw:.4f} is below market min price {min_price}. Adjusting SL to min price.{Style.RESET_ALL}"
            )
            actual_sl_price_raw = min_price  # Adjust SL upwards

        # Format the final SL price
        try:
            actual_sl_price_str = format_price(exchange, symbol, actual_sl_price_raw)
            actual_sl_price_float = float(actual_sl_price_str)  # For CCXT param
            if actual_sl_price_float <= 0:
                raise ValueError("Formatted actual SL price invalid")
            logger.info(
                f"Calculated ACTUAL Fixed SL Trigger Price: {Fore.YELLOW}{actual_sl_price_str}{Style.RESET_ALL}"
            )
        except (ValueError, InvalidOperation, TypeError) as e:
            logger.error(
                f"{Back.RED}{Fore.WHITE}CRITICAL FAILURE: Failed to format/validate ACTUAL SL price '{actual_sl_price_raw:.4f}': {e}. Cannot place SL!{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] CRITICAL ({side.upper()}): Failed format/validate ACTUAL SL price! Attempting emergency close."
            )
            close_position(exchange, symbol, {"side": side, "qty": filled_qty}, reason="Invalid SL Format Post-Entry")
            return filled_entry

        # === 11. Place Initial Fixed Stop Loss Order - The Static Ward ===
        sl_order_id_short = "N/A"
        sl_placed_successfully = False
        try:
            sl_side = CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy  # Opposite side for SL
            sl_qty_str = format_amount(exchange, symbol, filled_qty)  # Use actual filled quantity
            sl_qty_float = float(sl_qty_str)

            if sl_qty_float <= float(CONFIG.position_qty_epsilon):
                raise ValueError(f"Formatted SL quantity negligible: {sl_qty_float}")

            logger.info(
                f"{Fore.CYAN}Weaving Initial Fixed SL ({sl_atr_multiplier}*ATR)... Side: {sl_side.upper()}, Qty: {sl_qty_float:.8f}, TriggerPx: {actual_sl_price_str}{Style.RESET_ALL}"
            )
            # Bybit V5 stop order params using stopMarket type via CCXT:
            # 'stopPrice': The trigger price (float)
            # 'reduceOnly': Must be true for SL/TP orders (boolean)
            sl_params = {"stopPrice": actual_sl_price_float, "reduceOnly": True}
            sl_order = exchange.create_order(symbol, "stopMarket", sl_side, sl_qty_float, params=sl_params)
            sl_order_id_short = format_order_id(sl_order.get("id"))
            logger.success(
                f"{Fore.GREEN}Initial Fixed SL ward placed. ID: ...{sl_order_id_short}, Trigger: {actual_sl_price_str}{Style.RESET_ALL}"
            )
            sl_placed_successfully = True
        except (ValueError, ccxt.ExchangeError, ccxt.NetworkError) as e:
            logger.error(
                f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}FAILED to place Initial Fixed SL ward: {e}{Style.RESET_ALL}"
            )
            logger.debug(traceback.format_exc())
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] ERROR ({side.upper()}): Failed initial FIXED SL placement: {type(e).__name__}"
            )
            # Continue to TSL placement attempt, but log the failure. Position might still be partially protected by TSL.
        except Exception as e_sl:
            logger.error(
                f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}UNEXPECTED ERROR placing Initial Fixed SL ward: {e_sl}{Style.RESET_ALL}"
            )
            logger.debug(traceback.format_exc())
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] ERROR ({side.upper()}): Unexpected initial Fixed SL error: {type(e_sl).__name__}"
            )

        # === 12. Place Trailing Stop Loss Order - The Adaptive Shield ===
        tsl_order_id_short = "N/A"
        tsl_act_price_str = "N/A"
        tsl_placed_successfully = False
        try:
            # Calculate TSL activation price based on actual fill price and offset percentage
            act_offset = avg_fill_price * tsl_activation_offset_percent
            act_price_raw = (avg_fill_price + act_offset) if side == CONFIG.side_buy else (avg_fill_price - act_offset)

            # Apply min price constraint to activation price
            if act_price_raw <= 0:
                raise ValueError(f"Invalid TSL activation price calculated: {act_price_raw:.4f}")
            if min_price is not None and act_price_raw < min_price:
                logger.warning(
                    f"{Fore.YELLOW}TSL activation price {act_price_raw:.4f} is below market min price {min_price}. Adjusting activation to min price.{Style.RESET_ALL}"
                )
                act_price_raw = min_price

            # Format activation price
            tsl_act_price_str = format_price(exchange, symbol, act_price_raw)
            tsl_act_price_float = float(tsl_act_price_str)
            if tsl_act_price_float <= 0:
                raise ValueError("Formatted TSL activation price invalid")

            # Prepare TSL parameters for Bybit V5 via CCXT
            tsl_side = CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy  # Opposite side
            # Bybit V5 uses 'trailingStop' for percentage distance (e.g., "0.5" for 0.5%)
            # Convert our decimal percentage (e.g., 0.005) to percentage string (e.g., "0.5")
            tsl_trail_value_str = str((tsl_percent * Decimal("100")).normalize())
            tsl_qty_str = format_amount(exchange, symbol, filled_qty)  # Use actual filled quantity
            tsl_qty_float = float(tsl_qty_str)

            if tsl_qty_float <= float(CONFIG.position_qty_epsilon):
                raise ValueError(f"Formatted TSL quantity negligible: {tsl_qty_float}")

            logger.info(
                f"{Fore.CYAN}Weaving Trailing SL ({tsl_percent:.2%})... Side: {tsl_side.upper()}, Qty: {tsl_qty_float:.8f}, Trail%: {tsl_trail_value_str}, ActPx: {tsl_act_price_str}{Style.RESET_ALL}"
            )

            # Bybit V5 TSL params via CCXT using 'stopMarket' type:
            # 'trailingStop': Percentage value as a string (e.g., "0.5" for 0.5%)
            # 'activePrice': Activation trigger price (float)
            # 'reduceOnly': Must be True (boolean)
            tsl_params = {
                "trailingStop": tsl_trail_value_str,
                "activePrice": tsl_act_price_float,
                "reduceOnly": True,
            }
            tsl_order = exchange.create_order(symbol, "stopMarket", tsl_side, tsl_qty_float, params=tsl_params)
            tsl_order_id_short = format_order_id(tsl_order.get("id"))
            logger.success(
                f"{Fore.GREEN}Trailing SL shield placed. ID: ...{tsl_order_id_short}, Trail%: {tsl_trail_value_str}, ActPx: {tsl_act_price_str}{Style.RESET_ALL}"
            )
            tsl_placed_successfully = True

        except (ValueError, ccxt.ExchangeError, ccxt.NetworkError) as e:
            logger.error(
                f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}FAILED to place Trailing SL shield: {e}{Style.RESET_ALL}"
            )
            logger.debug(traceback.format_exc())
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] ERROR ({side.upper()}): Failed TSL placement: {type(e).__name__}"
            )
            # If TSL fails but initial SL was placed, the position is still protected initially.
        except Exception as e_tsl:
            logger.error(
                f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}UNEXPECTED ERROR placing Trailing SL shield: {e_tsl}{Style.RESET_ALL}"
            )
            logger.debug(traceback.format_exc())
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] ERROR ({side.upper()}): Unexpected TSL error: {type(e_tsl).__name__}"
            )

        # === 13. Final Confirmation & Alert ===
        if sl_placed_successfully or tsl_placed_successfully:
            # Send comprehensive SMS only if at least one protective order was placed
            sms_msg = (
                f"[{market_base}/{CONFIG.strategy_name}] ENTERED {side.upper()} {filled_qty:.8f} @ {avg_fill_price:.4f}. "
                f"SL: {'~' + actual_sl_price_str if sl_placed_successfully else 'FAIL'}(ID:...{sl_order_id_short}). "
                f"TSL: {tsl_percent:.2%}@{'~' + tsl_act_price_str if tsl_placed_successfully else 'FAIL'}(ID:...{tsl_order_id_short}). "
                f"EntryID:...{format_order_id(order_id)}"
            )
            send_sms_alert(sms_msg)
        elif not sl_placed_successfully and not tsl_placed_successfully:
            # Critical situation: Entry confirmed, but NO protective orders placed.
            logger.critical(
                f"{Back.RED}{Fore.WHITE}CRITICAL WARNING: Entry confirmed but BOTH Fixed SL and Trailing SL placement failed! Position is UNPROTECTED. Manual intervention likely required.{Style.RESET_ALL}"
            )
            # Send critical alert
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name}] CRITICAL ({side.upper()}): Entry OK BUT BOTH SL & TSL FAILED! POS UNPROTECTED! EntryID:...{format_order_id(order_id)}"
            )
            # Consider attempting emergency close again here? Or rely on shutdown handler? For now, just alert critically.

        # Return the details of the successfully filled entry order, even if subsequent SL/TSL failed (position is open)
        return filled_entry

    except (ccxt.InsufficientFunds, ccxt.NetworkError, ccxt.ExchangeError, ValueError, InvalidOperation) as e:
        # Catch errors occurring during the setup phase (before placing entry order)
        logger.error(
            f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Place Order Ritual ({side.upper()}): Pre-entry process failed: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        send_sms_alert(
            f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Pre-entry setup failed: {type(e).__name__}"
        )
    except Exception as e_overall:
        # Catch any other unexpected errors during the entire process
        logger.error(
            f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Place Order Ritual ({side.upper()}): Unexpected overall failure: {type(e_overall).__name__} - {e_overall}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        send_sms_alert(
            f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Unexpected overall error: {type(e_overall).__name__}"
        )

    return None  # Indicate failure of the overall process if any critical step failed before returning entry order


def cancel_open_orders(exchange: ccxt.Exchange, symbol: str, reason: str = "Cleanup") -> int:
    """Attempts to cancel all open orders for the specified symbol.
    Returns the number of orders successfully cancelled or confirmed closed.
    """
    logger.info(f"{Fore.CYAN}Order Cleanup Ritual: Initiating for {symbol} (Reason: {reason})...{Style.RESET_ALL}")
    cancelled_count: int = 0
    failed_count: int = 0
    market_base = symbol.split("/")[0].split(":")[0]

    try:
        if not exchange.has.get("fetchOpenOrders"):
            logger.warning(
                f"{Fore.YELLOW}Order Cleanup: fetchOpenOrders spell not available for {exchange.id}. Cannot perform automated cleanup.{Style.RESET_ALL}"
            )
            return 0  # Cannot cancel

        # Summon list of open orders for the specific symbol
        open_orders = exchange.fetch_open_orders(symbol)
        if not open_orders:
            logger.info(f"{Fore.CYAN}Order Cleanup: No open orders found for {symbol}.{Style.RESET_ALL}")
            return 0  # No orders to cancel

        logger.warning(
            f"{Fore.YELLOW}Order Cleanup: Found {len(open_orders)} open order(s) for {symbol}. Attempting cancellation...{Style.RESET_ALL}"
        )
        for order in open_orders:
            order_id = order.get("id")
            order_info_str = f"ID:...{format_order_id(order_id)} ({order.get('type', 'N/A')} {order.get('side', 'N/A')} Qty:{order.get('amount', 'N/A')} Px:{order.get('price', 'N/A')})"
            if order_id:
                try:
                    # Cast the cancel spell
                    logger.debug(f"Cancelling order {order_info_str}...")
                    exchange.cancel_order(order_id, symbol)
                    logger.info(
                        f"{Fore.CYAN}Order Cleanup: Cancel request successful for {order_info_str}{Style.RESET_ALL}"
                    )
                    cancelled_count += 1
                    time.sleep(0.15)  # Small delay between cancels to avoid strict rate limits
                except ccxt.OrderNotFound:
                    # Order might have been filled or cancelled just before this attempt
                    logger.warning(
                        f"{Fore.YELLOW}Order Cleanup: Order not found (already closed/cancelled?): {order_info_str}{Style.RESET_ALL}"
                    )
                    cancelled_count += 1  # Treat as effectively cancelled if not found
                except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
                    logger.error(
                        f"{Fore.RED}Order Cleanup: FAILED to cancel {order_info_str}: {type(e).__name__} - {e}{Style.RESET_ALL}"
                    )
                    failed_count += 1
                except Exception as e_cancel:
                    logger.error(
                        f"{Fore.RED}Order Cleanup: UNEXPECTED error cancelling {order_info_str}: {type(e_cancel).__name__} - {e_cancel}{Style.RESET_ALL}"
                    )
                    logger.debug(traceback.format_exc())
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
        logger.debug(traceback.format_exc())

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
        return signals  # Not enough data for comparisons

    # Access last (current closed) and previous candle data safely
    try:
        last = df.iloc[-1]  # Latest closed candle's data
        prev = df.iloc[-2]  # Previous candle's data
    except IndexError:
        logger.error(f"Signal Gen ({strategy_name}): Error accessing DataFrame rows (len: {len(df)}).")
        return signals

    try:
        # --- Dual Supertrend Logic ---
        if strategy_name == "DUAL_SUPERTREND":
            # Check necessary columns exist and are not NA
            primary_long_flip = last.get("st_long", False)  # True if primary ST flipped long this candle
            primary_short_flip = last.get("st_short", False)  # True if primary ST flipped short this candle
            confirm_is_up = last.get("confirm_trend", pd.NA)  # True if confirmation ST is currently up

            if pd.isna(confirm_is_up):
                logger.warning(f"Signal Gen ({strategy_name}): Confirmation trend is NA. Cannot generate signals.")
                return signals

            # Enter Long: Primary ST flips long AND Confirmation ST is in uptrend
            if primary_long_flip and confirm_is_up:
                signals["enter_long"] = True
            # Enter Short: Primary ST flips short AND Confirmation ST is in downtrend
            if primary_short_flip and not confirm_is_up:
                signals["enter_short"] = True
            # Exit Long: Primary ST flips short (regardless of confirmation trend)
            if primary_short_flip:
                signals["exit_long"] = True
                signals["exit_reason"] = "Primary ST Flipped Short"
            # Exit Short: Primary ST flips long (regardless of confirmation trend)
            if primary_long_flip:
                signals["exit_short"] = True
                signals["exit_reason"] = "Primary ST Flipped Long"

        # --- StochRSI + Momentum Logic ---
        elif strategy_name == "STOCHRSI_MOMENTUM":
            k_now = last.get("stochrsi_k", pd.NA)
            d_now = last.get("stochrsi_d", pd.NA)
            mom_now = last.get("momentum", pd.NA)
            k_prev = prev.get("stochrsi_k", pd.NA)
            d_prev = prev.get("stochrsi_d", pd.NA)

            # Check if all necessary indicator values are present (not NA)
            if any(pd.isna(v) for v in [k_now, d_now, mom_now, k_prev, d_prev]):
                logger.debug(
                    f"Signal Gen ({strategy_name}): Skipping due to NA values (k={k_now}, d={d_now}, mom={mom_now}, k_prev={k_prev}, d_prev={d_prev})"
                )
                return signals

            # Enter Long: K crosses above D from below, K is currently below oversold level, Momentum is positive
            if (
                k_prev <= d_prev
                and k_now > d_now
                and k_now < CONFIG.stochrsi_oversold
                and mom_now > CONFIG.position_qty_epsilon
            ):
                signals["enter_long"] = True
            # Enter Short: K crosses below D from above, K is currently above overbought level, Momentum is negative
            if (
                k_prev >= d_prev
                and k_now < d_now
                and k_now > CONFIG.stochrsi_overbought
                and mom_now < -CONFIG.position_qty_epsilon
            ):
                signals["enter_short"] = True
            # Exit Long: K crosses below D (potential downward momentum)
            if k_prev >= d_prev and k_now < d_now:
                signals["exit_long"] = True
                signals["exit_reason"] = "StochRSI K crossed below D"
            # Exit Short: K crosses above D (potential upward momentum)
            if k_prev <= d_prev and k_now > d_now:
                signals["exit_short"] = True
                signals["exit_reason"] = "StochRSI K crossed above D"

        # --- Ehlers Fisher Logic ---
        elif strategy_name == "EHLERS_FISHER":
            fish_now = last.get("ehlers_fisher", pd.NA)
            sig_now = last.get("ehlers_signal", pd.NA)
            fish_prev = prev.get("ehlers_fisher", pd.NA)
            sig_prev = prev.get("ehlers_signal", pd.NA)

            if any(pd.isna(v) for v in [fish_now, sig_now, fish_prev, sig_prev]):
                logger.debug(
                    f"Signal Gen ({strategy_name}): Skipping due to NA values (fish={fish_now}, sig={sig_now}, fish_prev={fish_prev}, sig_prev={sig_prev})"
                )
                return signals

            # Enter Long: Fisher line crosses above Signal line
            if fish_prev <= sig_prev and fish_now > sig_now:
                signals["enter_long"] = True
            # Enter Short: Fisher line crosses below Signal line
            if fish_prev >= sig_prev and fish_now < sig_now:
                signals["enter_short"] = True
            # Exit Long: Fisher crosses below Signal line (same condition as enter short)
            if fish_prev >= sig_prev and fish_now < sig_now:
                signals["exit_long"] = True
                signals["exit_reason"] = "Ehlers Fisher crossed Short"
            # Exit Short: Fisher crosses above Signal line (same condition as enter long)
            if fish_prev <= sig_prev and fish_now > sig_now:
                signals["exit_short"] = True
                signals["exit_reason"] = "Ehlers Fisher crossed Long"

        # --- Ehlers MA Cross Logic (Using EMA Placeholder) ---
        elif strategy_name == "EHLERS_MA_CROSS":
            fast_ma_now = last.get("fast_ema", pd.NA)
            slow_ma_now = last.get("slow_ema", pd.NA)
            fast_ma_prev = prev.get("fast_ema", pd.NA)
            slow_ma_prev = prev.get("slow_ema", pd.NA)

            if any(pd.isna(v) for v in [fast_ma_now, slow_ma_now, fast_ma_prev, slow_ma_prev]):
                logger.debug(
                    f"Signal Gen ({strategy_name} - EMA): Skipping due to NA values (fast={fast_ma_now}, slow={slow_ma_now}, fast_prev={fast_ma_prev}, slow_prev={slow_ma_prev})"
                )
                return signals

            # Enter Long: Fast MA crosses above Slow MA
            if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now:
                signals["enter_long"] = True
            # Enter Short: Fast MA crosses below Slow MA
            if fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now:
                signals["enter_short"] = True
            # Exit Long: Fast MA crosses below Slow MA (same condition as enter short)
            if fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now:
                signals["exit_long"] = True
                signals["exit_reason"] = "Fast EMA crossed below Slow EMA"
            # Exit Short: Fast MA crosses above Slow MA (same condition as enter long)
            if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now:
                signals["exit_short"] = True
                signals["exit_reason"] = "Fast EMA crossed above Slow EMA"

    except KeyError as e:
        logger.error(
            f"{Fore.RED}Signal Generation Error ({strategy_name}): Missing expected indicator column in DataFrame: {e}. Check indicator calculation functions.{Style.RESET_ALL}"
        )
    except Exception as e:
        logger.error(
            f"{Fore.RED}Signal Generation Error ({strategy_name}): Unexpected disturbance during signal evaluation: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())

    # Log generated signals only if a signal is active
    if signals["enter_long"] or signals["enter_short"] or signals["exit_long"] or signals["exit_short"]:
        active_signals = {k: v for k, v in signals.items() if isinstance(v, bool) and v}
        logger.debug(f"Strategy Signals ({strategy_name}): {active_signals}")
    return signals


# --- Trading Logic - The Core Spell Weaving ---
def trade_logic(exchange: ccxt.Exchange, symbol: str, df: pd.DataFrame) -> None:
    """Executes the main trading logic for one cycle:
    1. Calculates indicators.
    2. Checks position state.
    3. Checks filters (Volume, Order Book).
    4. Generates strategy signals.
    5. Executes exit or entry actions based on signals and filters.
    """
    cycle_time_str = df.index[-1].strftime("%Y-%m-%d %H:%M:%S %Z") if not df.empty else "N/A"
    logger.info(
        f"{Fore.BLUE}{Style.BRIGHT}========== New Weaving Cycle ({CONFIG.strategy_name}): {symbol} | Candle: {cycle_time_str} =========={Style.RESET_ALL}"
    )

    # --- Pre-computation: Determine max lookback needed for ANY indicator ---
    # This ensures we always request enough data, regardless of the active strategy. Add generous buffer.
    indicator_periods = [
        CONFIG.st_atr_length,
        CONFIG.confirm_st_atr_length,
        CONFIG.stochrsi_rsi_length,
        CONFIG.stochrsi_stoch_length,
        CONFIG.stochrsi_k_period,
        CONFIG.stochrsi_d_period,
        CONFIG.momentum_length,
        CONFIG.ehlers_fisher_length,
        CONFIG.ehlers_fisher_signal_length,
        CONFIG.ehlers_fast_period,
        CONFIG.ehlers_slow_period,
        CONFIG.atr_calculation_period,
        CONFIG.volume_ma_period,
    ]
    # Add buffer for calculation stability (e.g., SMA needs window size, StochRSI needs sum of its periods)
    # A simple max + buffer is usually sufficient.
    required_rows_strict = max(indicator_periods) + 10  # Base requirement + buffer
    # StochRSI might need more lookback: rsi_len + stoch_len + d_len
    stochrsi_lookback = CONFIG.stochrsi_rsi_length + CONFIG.stochrsi_stoch_length + CONFIG.stochrsi_d_period + 5
    required_rows = max(required_rows_strict, stochrsi_lookback, 50)  # Ensure at least 50 rows minimum

    if df is None or len(df) < required_rows:
        logger.warning(
            f"{Fore.YELLOW}Trade Logic Skipped: Insufficient data ({len(df) if df is not None else 0} rows, need ~{required_rows}). Check data fetching.{Style.RESET_ALL}"
        )
        return

    action_taken_this_cycle: bool = False  # Track if an entry/exit order was placed/attempted

    try:
        # === 1. Calculate ALL Potential Indicators - Scry the full spectrum ===
        # Calculate all indicators needed by any strategy. Signal generation will pick the relevant ones.
        logger.debug("Calculating all potential indicators...")
        df = calculate_supertrend(df, CONFIG.st_atr_length, CONFIG.st_multiplier)
        df = calculate_supertrend(df, CONFIG.confirm_st_atr_length, CONFIG.confirm_st_multiplier, prefix="confirm_")
        df = calculate_stochrsi_momentum(
            df,
            CONFIG.stochrsi_rsi_length,
            CONFIG.stochrsi_stoch_length,
            CONFIG.stochrsi_k_period,
            CONFIG.stochrsi_d_period,
            CONFIG.momentum_length,
        )
        df = calculate_ehlers_fisher(df, CONFIG.ehlers_fisher_length, CONFIG.ehlers_fisher_signal_length)
        df = calculate_ehlers_ma(df, CONFIG.ehlers_fast_period, CONFIG.ehlers_slow_period)  # Uses EMA placeholder
        vol_atr_data = analyze_volume_atr(df, CONFIG.atr_calculation_period, CONFIG.volume_ma_period)
        current_atr = vol_atr_data.get("atr")  # Crucial for SL calculation

        # === 2. Validate Base Requirements - Ensure stable ground ===
        last_candle = df.iloc[-1]
        current_price = safe_decimal_conversion(last_candle.get("close"))  # Get latest close price

        if pd.isna(current_price) or current_price <= 0:
            logger.warning(
                f"{Fore.YELLOW}Trade Logic Skipped: Last candle close price is invalid ({current_price}).{Style.RESET_ALL}"
            )
            return
        # Can we place a new order? Requires a valid ATR for SL calculation.
        can_place_new_order = current_atr is not None and current_atr > Decimal("0")
        if not can_place_new_order:
            logger.warning(
                f"{Fore.YELLOW}Invalid ATR ({current_atr}) calculated. Cannot calculate SL or place new entry orders this cycle.{Style.RESET_ALL}"
            )
            # Note: Existing position management (exits based on strategy signals) might still be possible.

        # === 3. Get Current Position & Analyze Order Book (conditionally) ===
        position = get_current_position(exchange, symbol)  # Check current market presence
        position_side = position["side"]
        position_qty = position["qty"]
        position_entry = position["entry_price"]

        # Fetch OB data if configured for every cycle, otherwise it will be fetched later if needed for entry confirmation
        ob_data = None
        if CONFIG.fetch_order_book_per_cycle:
            ob_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit)

        # === 4. Log Current State - The Oracle Reports ===
        vol_ratio = vol_atr_data.get("volume_ratio")
        is_vol_spike = vol_ratio is not None and vol_ratio > CONFIG.volume_spike_threshold
        bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None
        spread = ob_data.get("spread") if ob_data else None

        # Log core state
        atr_str = f"{current_atr:.5f}" if current_atr else f"{Fore.RED}N/A{Style.RESET_ALL}"
        logger.info(
            f"State | Price: {Fore.CYAN}{current_price:.4f}{Style.RESET_ALL}, ATR({CONFIG.atr_calculation_period}): {Fore.MAGENTA}{atr_str}{Style.RESET_ALL}"
        )

        # Log filter states
        vol_ratio_str = f"{vol_ratio:.2f}" if vol_ratio else "N/A"
        vol_spike_str = f"{Fore.GREEN}YES{Style.RESET_ALL}" if is_vol_spike else f"{Fore.RED}NO{Style.RESET_ALL}"
        logger.info(
            f"State | Volume: Ratio={Fore.YELLOW}{vol_ratio_str}{Style.RESET_ALL}, Spike={vol_spike_str} (Threshold={CONFIG.volume_spike_threshold}, RequiredForEntry={CONFIG.require_volume_spike_for_entry})"
        )

        ob_ratio_str = f"{bid_ask_ratio:.3f}" if bid_ask_ratio else "N/A"
        ob_spread_str = f"{spread:.4f}" if spread else "N/A"
        ob_ratio_color = (
            Fore.GREEN
            if bid_ask_ratio and bid_ask_ratio >= CONFIG.order_book_ratio_threshold_long
            else (
                Fore.RED if bid_ask_ratio and bid_ask_ratio <= CONFIG.order_book_ratio_threshold_short else Fore.YELLOW
            )
        )
        logger.info(
            f"State | OrderBook: Ratio(B/A)={ob_ratio_color}{ob_ratio_str}{Style.RESET_ALL} (L >= {CONFIG.order_book_ratio_threshold_long}, S <= {CONFIG.order_book_ratio_threshold_short}), Spread={ob_spread_str} (Fetched={ob_data is not None})"
        )

        # Log position state
        pos_color = (
            Fore.GREEN
            if position_side == CONFIG.pos_long
            else (Fore.RED if position_side == CONFIG.pos_short else Fore.BLUE)
        )
        logger.info(
            f"State | Position: Side={pos_color}{position_side}{Style.RESET_ALL}, Qty={position_qty:.8f}, Entry={position_entry:.4f}"
        )

        # === 5. Generate Strategy Signals - Interpret the Omens ===
        strategy_signals = generate_signals(df, CONFIG.strategy_name)

        # === 6. Execute Exit Actions - If the Omens Demand Retreat ===
        # Check if we are currently in a position that matches an exit signal
        should_exit_long = position_side == CONFIG.pos_long and strategy_signals["exit_long"]
        should_exit_short = position_side == CONFIG.pos_short and strategy_signals["exit_short"]

        if should_exit_long or should_exit_short:
            exit_reason = strategy_signals["exit_reason"]
            exit_side_color = Back.YELLOW
            logger.warning(
                f"{exit_side_color}{Fore.BLACK}{Style.BRIGHT}*** STRATEGY EXIT SIGNAL: Attempting to Close {position_side} Position (Reason: {exit_reason}) ***{Style.RESET_ALL}"
            )
            action_taken_this_cycle = True  # Mark that we are attempting an action

            # --- Pre-Exit Cleanup: Cancel existing SL/TP orders ---
            # It's crucial to cancel stops BEFORE sending the market close order
            # to avoid potential race conditions or conflicts.
            logger.info("Performing pre-exit order cleanup...")
            cancel_open_orders(exchange, symbol, f"Pre-Exit Cleanup ({exit_reason})")
            time.sleep(0.75)  # Small pause after cancel request before placing market close

            # --- Attempt to close the position ---
            close_result = close_position(exchange, symbol, position, reason=exit_reason)
            if close_result:
                # Position close order placed (might not be fully confirmed yet)
                logger.info(f"Position close order placed for {position_side}. Pausing briefly...")
                time.sleep(CONFIG.post_close_delay_seconds)  # Pause after successful close attempt
            else:
                # Close attempt failed (error logged in close_position)
                logger.error(
                    f"{Fore.RED}Failed to place position close order for {position_side}. Manual check advised.{Style.RESET_ALL}"
                )
                # Still pause briefly even on failure to avoid rapid retries if issue persists
                time.sleep(CONFIG.sleep_seconds // 2)

            # --- Exit the current logic cycle ---
            # Regardless of close success/failure, we don't want to evaluate entry signals immediately after an exit signal.
            logger.info("Exiting trade logic cycle after processing exit signal.")
            return

        # === 7. Check & Execute Entry Actions (Only if Currently Flat) ===
        if position_side != CONFIG.pos_none:
            logger.info(
                f"Holding {pos_color}{position_side}{Style.RESET_ALL} position. Awaiting Exchange SL/TSL or Strategy Exit signal."
            )
            return  # Do nothing more if already in a position

        # --- Check if we can place a new order (requires valid ATR) ---
        if not can_place_new_order:
            logger.warning(
                f"{Fore.YELLOW}Holding Cash. Cannot evaluate entry: Invalid ATR ({current_atr}) prevents SL calculation.{Style.RESET_ALL}"
            )
            return  # Cannot enter without valid ATR for initial SL

        # --- Evaluate Entry Conditions ---
        logger.debug("Position is Flat. Checking strategy entry signals...")
        potential_entry_signal = strategy_signals["enter_long"] or strategy_signals["enter_short"]

        if not potential_entry_signal:
            logger.info("Holding Cash. No entry signal generated by strategy.")
            return  # No base signal, do nothing

        # --- Apply Confirmation Filters (Volume & Order Book) ---
        logger.debug("Potential entry signal found. Evaluating confirmation filters...")

        # Fetch OB data now if not fetched per cycle AND there's a potential entry signal
        if ob_data is None:
            logger.debug("Fetching Order Book data for entry confirmation...")
            ob_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit)
            bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None  # Update ratio after fetch

        # Evaluate Order Book Filter
        ob_confirm_long = False
        ob_confirm_short = False
        if bid_ask_ratio is not None:
            ob_confirm_long = bid_ask_ratio >= CONFIG.order_book_ratio_threshold_long
            ob_confirm_short = bid_ask_ratio <= CONFIG.order_book_ratio_threshold_short
        ob_log_ratio_str = f"{bid_ask_ratio:.3f}" if bid_ask_ratio is not None else "N/A"
        ob_filter_log = (
            f"OB Confirm Long: {ob_confirm_long} (Ratio: {ob_log_ratio_str} >= {CONFIG.order_book_ratio_threshold_long}), "
            f"OB Confirm Short: {ob_confirm_short} (Ratio: {ob_log_ratio_str} <= {CONFIG.order_book_ratio_threshold_short})"
        )
        logger.debug(ob_filter_log)

        # Evaluate Volume Filter
        vol_confirm = not CONFIG.require_volume_spike_for_entry or is_vol_spike
        vol_filter_log = (
            f"Vol Confirm: {vol_confirm} (Spike: {is_vol_spike}, Required: {CONFIG.require_volume_spike_for_entry})"
        )
        logger.debug(vol_filter_log)

        # --- Combine Strategy Signal with Confirmations ---
        final_enter_long = strategy_signals["enter_long"] and ob_confirm_long and vol_confirm
        final_enter_short = strategy_signals["enter_short"] and ob_confirm_short and vol_confirm

        # Log final entry decision logic
        if strategy_signals["enter_long"]:
            logger.debug(
                f"Final Entry Check (Long): Strategy={strategy_signals['enter_long']}, OB OK={ob_confirm_long}, Vol OK={vol_confirm} => {Fore.GREEN if final_enter_long else Fore.RED}Enter={final_enter_long}{Style.RESET_ALL}"
            )
        if strategy_signals["enter_short"]:
            logger.debug(
                f"Final Entry Check (Short): Strategy={strategy_signals['enter_short']}, OB OK={ob_confirm_short}, Vol OK={vol_confirm} => {Fore.GREEN if final_enter_short else Fore.RED}Enter={final_enter_short}{Style.RESET_ALL}"
            )

        # --- Execute Entry ---
        if final_enter_long:
            entry_bg_color = Back.GREEN
            logger.success(
                f"{entry_bg_color}{Fore.BLACK}{Style.BRIGHT}*** CONFIRMED LONG ENTRY SIGNAL ({CONFIG.strategy_name}) for {symbol} ***{Style.RESET_ALL}"
            )
            action_taken_this_cycle = True
            # Cancel any stray orders before entering (belt-and-suspenders)
            cancel_open_orders(exchange, symbol, "Pre-Long Entry Cleanup")
            time.sleep(0.5)  # Small pause after cancel
            # Place the risked order with SL and TSL
            place_result = place_risked_market_order(
                exchange=exchange,
                symbol=symbol,
                side=CONFIG.side_buy,
                risk_percentage=CONFIG.risk_per_trade_percentage,
                current_atr=current_atr,
                sl_atr_multiplier=CONFIG.atr_stop_loss_multiplier,
                leverage=CONFIG.leverage,
                max_order_cap_usdt=CONFIG.max_order_usdt_amount,
                margin_check_buffer=CONFIG.required_margin_buffer,
                tsl_percent=CONFIG.trailing_stop_percentage,
                tsl_activation_offset_percent=CONFIG.trailing_stop_activation_offset_percent,
            )
            if place_result:
                logger.info("Long entry order placement process initiated.")
            else:
                logger.error(f"{Fore.RED}Long entry order placement process failed.{Style.RESET_ALL}")

        elif final_enter_short:
            entry_bg_color = Back.RED
            logger.success(
                f"{entry_bg_color}{Fore.WHITE}{Style.BRIGHT}*** CONFIRMED SHORT ENTRY SIGNAL ({CONFIG.strategy_name}) for {symbol} ***{Style.RESET_ALL}"
            )
            action_taken_this_cycle = True
            # Cancel any stray orders before entering
            cancel_open_orders(exchange, symbol, "Pre-Short Entry Cleanup")
            time.sleep(0.5)  # Small pause after cancel
            # Place the risked order with SL and TSL
            place_result = place_risked_market_order(
                exchange=exchange,
                symbol=symbol,
                side=CONFIG.side_sell,
                risk_percentage=CONFIG.risk_per_trade_percentage,
                current_atr=current_atr,
                sl_atr_multiplier=CONFIG.atr_stop_loss_multiplier,
                leverage=CONFIG.leverage,
                max_order_cap_usdt=CONFIG.max_order_usdt_amount,
                margin_check_buffer=CONFIG.required_margin_buffer,
                tsl_percent=CONFIG.trailing_stop_percentage,
                tsl_activation_offset_percent=CONFIG.trailing_stop_activation_offset_percent,
            )
            if place_result:
                logger.info("Short entry order placement process initiated.")
            else:
                logger.error(f"{Fore.RED}Short entry order placement process failed.{Style.RESET_ALL}")

        else:
            # Log if a base signal existed but filters blocked it
            if potential_entry_signal and not action_taken_this_cycle:
                logger.info("Holding Cash. Strategy signal present but confirmation filters (Volume/OB) not met.")
            # No logging needed here if potential_entry_signal was false initially

    except Exception as e:
        # Catch-all for unexpected errors within the main logic loop for this cycle
        logger.error(
            f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}CRITICAL UNEXPECTED ERROR in trade_logic cycle: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        send_sms_alert(
            f"[{symbol.split('/')[0].split(':')[0]}/{CONFIG.strategy_name}] CRITICAL ERROR in trade_logic cycle: {type(e).__name__}. Check logs!"
        )
    finally:
        # Mark the end of the cycle clearly
        logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== Cycle Weaving End: {symbol} =========={Style.RESET_ALL}\n")


# --- Graceful Shutdown - Withdrawing the Arcane Energies ---
def graceful_shutdown(exchange: ccxt.Exchange | None, symbol: str | None) -> None:
    """Attempts to cancel all open orders and close any existing position before exiting."""
    logger.warning(
        f"\n{Fore.YELLOW}{Style.BRIGHT}Shutdown requested. Withdrawing arcane energies gracefully...{Style.RESET_ALL}"
    )
    market_base = symbol.split("/")[0].split(":")[0] if symbol else "Bot"
    send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] Shutdown initiated. Attempting cleanup...")

    if not exchange or not symbol:
        logger.warning(
            f"{Fore.YELLOW}Shutdown: Exchange portal or symbol not defined. Cannot perform automated cleanup.{Style.RESET_ALL}"
        )
        return

    try:
        # 1. Cancel All Open Orders - Dispel residual intents (SL, TSL, pending entries)
        logger.warning("Shutdown Step 1: Cancelling all open orders...")
        cancelled_count = cancel_open_orders(exchange, symbol, reason="Graceful Shutdown")
        logger.info(f"Shutdown Step 1: Cancelled {cancelled_count} open orders.")
        time.sleep(1.5)  # Allow cancellations time to process on the exchange side

        # 2. Check and Close Existing Position - Banish final market presence
        logger.warning("Shutdown Step 2: Checking for active position to close...")
        position = get_current_position(exchange, symbol)  # Get final position state

        if position["side"] != CONFIG.pos_none and position["qty"] > CONFIG.position_qty_epsilon:
            pos_color = Fore.GREEN if position["side"] == CONFIG.pos_long else Fore.RED
            logger.warning(
                f"{Fore.YELLOW}Shutdown Step 2: Active {pos_color}{position['side']}{Style.RESET_ALL} position found (Qty: {position['qty']:.8f}). Attempting market close...{Style.RESET_ALL}"
            )
            close_result = close_position(exchange, symbol, position, reason="Shutdown")

            if close_result:
                logger.info(
                    f"{Fore.CYAN}Shutdown Step 2: Close order placed. Waiting {CONFIG.post_close_delay_seconds * 2}s for confirmation...{Style.RESET_ALL}"
                )
                time.sleep(CONFIG.post_close_delay_seconds * 2)  # Wait a bit longer to be reasonably sure it closed

                # --- Final Confirmation Check ---
                logger.warning("Shutdown Step 3: Final position confirmation check...")
                final_pos = get_current_position(exchange, symbol)
                if final_pos["side"] == CONFIG.pos_none or final_pos["qty"] <= CONFIG.position_qty_epsilon:
                    logger.success(
                        f"{Fore.GREEN}{Style.BRIGHT}Shutdown Step 3: Position confirmed CLOSED/FLAT.{Style.RESET_ALL}"
                    )
                    send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] Position confirmed CLOSED on shutdown.")
                else:
                    # This is a critical issue - manual intervention likely needed
                    final_pos_color = Fore.GREEN if final_pos["side"] == CONFIG.pos_long else Fore.RED
                    logger.critical(
                        f"{Back.RED}{Fore.WHITE}CRITICAL SHUTDOWN FAILURE: FAILED TO CONFIRM position closure after waiting! "
                        f"Final state: {final_pos_color}{final_pos['side']}{Style.RESET_ALL} Qty={final_pos['qty']:.8f}. "
                        f"MANUAL INTERVENTION REQUIRED on Bybit!{Style.RESET_ALL}"
                    )
                    send_sms_alert(
                        f"[{market_base}/{CONFIG.strategy_name}] CRITICAL SHUTDOWN ERROR: Failed confirm closure! Final: {final_pos['side']} Qty={final_pos['qty']:.8f}. MANUAL CHECK!"
                    )
            else:
                # Close order placement failed initially
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}CRITICAL SHUTDOWN FAILURE: Failed to place position close order during shutdown. "
                    f"Position likely still open! MANUAL INTERVENTION REQUIRED on Bybit!{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"[{market_base}/{CONFIG.strategy_name}] CRITICAL SHUTDOWN ERROR: Failed PLACE close order. MANUAL CHECK!"
                )
        else:
            logger.info(f"{Fore.GREEN}Shutdown Step 2: No active position found. Clean exit state.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] No active position found on shutdown.")

    except Exception as e:
        logger.error(f"{Fore.RED}Shutdown Error: Unexpected error during cleanup sequence: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
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
        f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Bybit Scalping Spell v2.2.1 Initializing ({start_time_str}) ---{Style.RESET_ALL}"
    )
    logger.info(f"{Fore.CYAN}--- Strategy Enchantment Selected: {CONFIG.strategy_name} ---{Style.RESET_ALL}")
    logger.info(
        f"{Fore.GREEN}--- Protective Wards Activated: Initial ATR-Stop + Exchange Trailing Stop (Bybit V5 Native) ---{Style.RESET_ALL}"
    )
    logger.warning(
        f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}--- !!! LIVE FUTURES SCALPING MODE - EXTREME RISK - EDUCATIONAL USE ONLY !!! ---{Style.RESET_ALL}"
    )

    exchange: ccxt.Exchange | None = None
    symbol_unified: str | None = None  # The specific market symbol confirmed by CCXT (e.g., BTC/USDT:USDT)
    run_bot: bool = True  # Controls the main trading loop
    cycle_count: int = 0  # Tracks the number of iterations

    try:
        # === Initialize Exchange Portal ===
        exchange = initialize_exchange()
        if not exchange:
            logger.critical("Failed to open exchange portal. Spell cannot proceed. Exiting.")
            # SMS alert sent within initialize_exchange on failure
            return  # Exit script

        # === Setup Symbol and Leverage - Focusing the Spell ===
        try:
            # Use symbol from config directly
            symbol_to_use = CONFIG.symbol
            logger.info(f"Attempting to focus spell on symbol: {symbol_to_use}")

            # Validate symbol and get unified representation from CCXT
            market = exchange.market(symbol_to_use)
            symbol_unified = market["symbol"]  # Use the precise symbol recognized by CCXT (e.g., BTC/USDT:USDT)

            # Ensure it's a futures/contract market suitable for leverage
            market_type = market.get("type", "unknown")
            is_contract = market.get("contract", False)
            is_linear = market.get("linear", False)  # Check if it's linear (USDT margined)

            if not is_contract:
                raise ValueError(f"Market '{symbol_unified}' (Type: {market_type}) is not a contract/futures market.")
            if not is_linear:
                logger.warning(
                    f"{Fore.YELLOW}Market '{symbol_unified}' is not detected as LINEAR (USDT margined). Ensure compatibility with bot logic.{Style.RESET_ALL}"
                )
                # Allow proceeding but warn user

            logger.info(
                f"{Fore.GREEN}Spell successfully focused on Symbol: {symbol_unified} (Type: {market_type}, Contract: {is_contract}, Linear: {is_linear}){Style.RESET_ALL}"
            )

            # Set the desired leverage for the focused symbol
            if not set_leverage(exchange, symbol_unified, CONFIG.leverage):
                raise RuntimeError(f"Leverage conjuring failed for {symbol_unified}. Cannot proceed safely.")

        except (ccxt.BadSymbol, KeyError, ValueError, RuntimeError) as e:
            logger.critical(f"{Back.RED}{Fore.WHITE}Symbol/Leverage setup failed: {e}{Style.RESET_ALL}")
            send_sms_alert(f"[Pyrmethus/{CONFIG.strategy_name}] CRITICAL: Symbol/Leverage setup FAILED ({e}). Exiting.")
            return  # Exit script
        except Exception as e_setup:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}Unexpected error during spell focus (Symbol/Leverage) setup: {e_setup}{Style.RESET_ALL}"
            )
            logger.debug(traceback.format_exc())
            send_sms_alert(f"[Pyrmethus/{CONFIG.strategy_name}] CRITICAL: Unexpected setup error. Exiting.")
            return  # Exit script

        # === Log Configuration Summary - Reciting the Parameters ===
        logger.info(f"{Fore.MAGENTA}--- Spell Configuration Summary ---{Style.RESET_ALL}")
        logger.info(f"{Fore.WHITE}Symbol: {symbol_unified}, Interval: {CONFIG.interval}, Leverage: {CONFIG.leverage}x")
        logger.info(f"{Fore.CYAN}Strategy Path: {CONFIG.strategy_name}")
        # Log relevant strategy parameters for clarity
        if CONFIG.strategy_name == "DUAL_SUPERTREND":
            logger.info(
                f"  Params: ST={CONFIG.st_atr_length}/{CONFIG.st_multiplier}, ConfirmST={CONFIG.confirm_st_atr_length}/{CONFIG.confirm_st_multiplier}"
            )
        elif CONFIG.strategy_name == "STOCHRSI_MOMENTUM":
            logger.info(
                f"  Params: StochRSI={CONFIG.stochrsi_rsi_length}/{CONFIG.stochrsi_stoch_length}/{CONFIG.stochrsi_k_period}/{CONFIG.stochrsi_d_period} (OB={CONFIG.stochrsi_overbought},OS={CONFIG.stochrsi_oversold}), Mom={CONFIG.momentum_length}"
            )
        elif CONFIG.strategy_name == "EHLERS_FISHER":
            logger.info(f"  Params: Fisher={CONFIG.ehlers_fisher_length}, Signal={CONFIG.ehlers_fisher_signal_length}")
        elif CONFIG.strategy_name == "EHLERS_MA_CROSS":
            logger.info(
                f"  Params: FastMA(EMA)={CONFIG.ehlers_fast_period}, SlowMA(EMA)={CONFIG.ehlers_slow_period} {Fore.YELLOW}(EMA Placeholder){Style.RESET_ALL}"
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
        # Format margin buffer as percentage
        margin_buffer_percent = (CONFIG.required_margin_buffer - Decimal(1)) * Decimal(100)
        logger.info(
            f"{Fore.WHITE}Timing: Sleep={CONFIG.sleep_seconds}s | API: RecvWin={CONFIG.default_recv_window}ms, FillTimeout={CONFIG.order_fill_timeout_seconds}s"
        )
        logger.info(
            f"{Fore.WHITE}Other: Margin Buffer={margin_buffer_percent:.1f}%, SMS Alerts={CONFIG.enable_sms_alerts}"
        )
        logger.info(f"{Fore.CYAN}Oracle Verbosity (Log Level): {logging.getLevelName(logger.level)}")
        logger.info(f"{Fore.MAGENTA}{'-' * 30}{Style.RESET_ALL}")

        market_base = symbol_unified.split("/")[0].split(":")[0]  # For SMS brevity
        send_sms_alert(
            f"[{market_base}/{CONFIG.strategy_name}] Pyrmethus Bot v2.2.1 Initialized. Symbol: {symbol_unified}, Strat: {CONFIG.strategy_name}. Starting main loop."
        )

        # === Main Trading Loop - The Continuous Weaving ===
        while run_bot:
            cycle_start_time = time.monotonic()
            cycle_count += 1
            logger.debug(
                f"{Fore.CYAN}--- Cycle {cycle_count} Weaving Start ({time.strftime('%H:%M:%S')}) ---{Style.RESET_ALL}"
            )

            try:
                # --- Calculate required data length dynamically ---
                # Ensure enough data for the longest lookback period of any indicator used across all strategies + buffer
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
                    CONFIG.atr_calculation_period,
                    CONFIG.volume_ma_period,
                ]
                # Add buffer for calculation stability (e.g., SMA needs window size, StochRSI needs sum of its periods)
                stochrsi_lookback = (
                    CONFIG.stochrsi_rsi_length + CONFIG.stochrsi_stoch_length + CONFIG.stochrsi_d_period + 5
                )
                data_limit = (
                    max(max(indicator_periods) + 15, stochrsi_lookback, 100) + CONFIG.api_fetch_limit_buffer
                )  # Ensure reasonable min + buffer

                # --- Gather fresh market data ---
                df = get_market_data(exchange, symbol_unified, CONFIG.interval, limit=data_limit)

                # --- Process data and execute logic if data is valid ---
                if df is not None and not df.empty:
                    # Pass a copy to trade_logic to prevent modifications affecting subsequent cycles if df were reused
                    trade_logic(exchange, symbol_unified, df.copy())
                else:
                    # Error/Warning logged within get_market_data if fetching failed
                    logger.warning(
                        f"{Fore.YELLOW}Skipping trade logic this cycle due to invalid/missing market data for {symbol_unified}.{Style.RESET_ALL}"
                    )

            # --- Robust Error Handling within the Loop ---
            except ccxt.RateLimitExceeded as e:
                logger.warning(
                    f"{Back.YELLOW}{Fore.BLACK}Rate Limit Exceeded: {e}. The exchange spirits demand patience. Sleeping longer...{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"[{market_base}/{CONFIG.strategy_name}] WARNING: Rate limit exceeded! Pausing {CONFIG.sleep_seconds * 6}s."
                )
                time.sleep(CONFIG.sleep_seconds * 6)  # Sleep much longer
            except ccxt.NetworkError as e:
                # Transient network issues, usually recoverable
                logger.warning(
                    f"{Fore.YELLOW}Network disturbance in main loop: {e}. Retrying next cycle.{Style.RESET_ALL}"
                )
                # Optional: Send SMS on repeated network errors?
                time.sleep(CONFIG.sleep_seconds * 2)  # Slightly longer sleep for network issues
            except ccxt.ExchangeNotAvailable as e:
                # Exchange might be down for maintenance or unavailable
                logger.error(
                    f"{Back.RED}{Fore.WHITE}Exchange Not Available: {e}. Portal temporarily closed. Sleeping much longer...{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"[{market_base}/{CONFIG.strategy_name}] ERROR: Exchange unavailable ({type(e).__name__})! Long pause (60s)."
                )
                time.sleep(60)  # Wait a significant time
            except ccxt.AuthenticationError as e:
                # API keys might have been revoked, expired, or IP changed
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}FATAL: Authentication Error: {e}. API keys invalid or permissions changed. Spell broken! Stopping NOW.{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"[{market_base}/{CONFIG.strategy_name}] CRITICAL: Authentication Error! Bot stopping NOW. Check API keys/IP."
                )
                run_bot = False  # Stop the bot immediately
            except ccxt.ExchangeError as e:  # Catch other specific exchange errors not handled elsewhere
                logger.error(f"{Fore.RED}Unhandled Exchange Error in main loop: {e}{Style.RESET_ALL}")
                logger.debug(traceback.format_exc())
                send_sms_alert(
                    f"[{market_base}/{CONFIG.strategy_name}] ERROR: Unhandled Exchange error: {type(e).__name__}. Check logs."
                )
                time.sleep(CONFIG.sleep_seconds)  # Standard sleep before retrying after general exchange error
            except Exception as e:
                # Catch-all for truly unexpected issues in the main loop/trade_logic
                logger.exception(
                    f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}!!! UNEXPECTED CRITICAL CHAOS in Main Loop: {e} !!! Stopping spell!{Style.RESET_ALL}"
                )
                # logger.exception provides full traceback automatically
                send_sms_alert(
                    f"[{market_base}/{CONFIG.strategy_name}] CRITICAL UNEXPECTED ERROR: {type(e).__name__}! Bot stopping NOW. Check logs!"
                )
                run_bot = False  # Stop the bot on unknown critical errors

            # --- Loop Delay - Controlling the Rhythm ---
            if run_bot:
                cycle_end_time = time.monotonic()
                elapsed = cycle_end_time - cycle_start_time
                sleep_duration = max(0, CONFIG.sleep_seconds - elapsed)
                logger.debug(f"Cycle {cycle_count} duration: {elapsed:.2f}s. Sleeping for {sleep_duration:.2f}s.")
                if sleep_duration > 0:
                    time.sleep(sleep_duration)  # Wait for the configured interval before next cycle

    except KeyboardInterrupt:
        logger.warning(
            f"\n{Fore.YELLOW}{Style.BRIGHT}KeyboardInterrupt detected. User requests withdrawal of arcane energies...{Style.RESET_ALL}"
        )
        run_bot = False  # Signal the loop to terminate gracefully
    except Exception as startup_error:
        # Catch critical errors during initial setup (before the main loop starts)
        logger.critical(
            f"{Back.RED}{Fore.WHITE}CRITICAL ERROR during bot startup sequence: {startup_error}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        # Attempt to send SMS if config allows, otherwise just log
        if CONFIG and CONFIG.enable_sms_alerts and CONFIG.sms_recipient_number:
            send_sms_alert(f"[Pyrmethus] CRITICAL STARTUP ERROR: {type(startup_error).__name__}. Bot failed to start.")
        run_bot = False  # Ensure bot doesn't run
    finally:
        # --- Graceful Shutdown Sequence ---
        # This will run whether the loop finished normally, was interrupted by Ctrl+C,
        # or hit a critical error that set run_bot=False.
        graceful_shutdown(exchange, symbol_unified)
        symbol_unified.split("/")[0].split(":")[0] if symbol_unified else "Bot"
        # Final SMS may not send if Termux process is killed abruptly, but attempt it.
        # send_sms_alert(f"[{market_base_final}/{CONFIG.strategy_name}] Bot process terminated.") # Optional: Alert on final termination
        logger.info(
            f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Scalping Spell Deactivated ---{Style.RESET_ALL}"
        )


if __name__ == "__main__":
    # Ensure the spell is cast only when invoked directly
    main()
