#!/usr/bin/env python

# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v2.2 (Fortified Configuration & Clarity)
# Conjures high-frequency trades on Bybit Futures with enhanced precision, adaptable strategies, and Termux integration.

"""High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 2.2.0 (Unified: Selectable Strategies + Precision + Native SL/TSL + Fortified Config + Pyrmethus Enhancements).

Features:
- Multiple strategies selectable via config: "DUAL_SUPERTREND", "STOCHRSI_MOMENTUM", "EHLERS_FISHER", "EHLERS_MA_CROSS".
- Enhanced Precision: Uses Decimal for critical financial calculations.
- Fortified Configuration Loading: Correctly handles type casting for default values.
- Exchange-native Trailing Stop Loss (TSL) placed immediately after entry.
- Exchange-native fixed Stop Loss placed immediately after entry.
- ATR for volatility measurement and initial Stop-Loss calculation.
- Optional Volume spike and Order Book pressure confirmation.
- Risk-based position sizing with margin checks.
- Termux SMS alerts for critical events and trade actions (with Termux:API check).
- Robust error handling and logging with vibrant Neon color support via Colorama.
- Graceful shutdown on KeyboardInterrupt with position/order closing attempt.
- Stricter position detection logic (Bybit V5 API).

Disclaimer:
- **EXTREME RISK**: Arcane energies are volatile. Educational purposes ONLY. High-risk. Use at own absolute risk.
- **EXCHANGE-NATIVE SL/TSL DEPENDENCE**: Relies on exchange-native orders. Subject to exchange performance, slippage, API reliability.
- Parameter Sensitivity: Requires significant tuning and testing in the astral plane (testnet).
- API Rate Limits: Monitor usage lest the exchange spirits grow wary.
- Slippage: Market orders are prone to slippage in turbulent ether.
- Test Thoroughly: **DO NOT RUN LIVE WITHOUT EXTENSIVE TESTNET/DEMO TESTING.**
- Termux Dependency: Requires Termux:API for SMS communication scrolls. Ensure `pkg install termux-api`.
- API Changes: Code targets Bybit V5 via CCXT, updates may be needed as the digital cosmos shifts.
"""

# Standard Library Imports - The Foundational Runes
import logging
import os
import shutil  # For checking command existence
import subprocess
import sys
import time
import traceback
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, getcontext
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
colorama_init(autoreset=True)  # Activate Colorama's magic
load_dotenv()  # Load secrets from the hidden .env scroll
getcontext().prec = 18  # Set Decimal precision for financial exactitude


# --- Configuration Class - Defining the Spell's Parameters ---
class Config:
    """Loads and validates configuration parameters from environment variables."""
    def __init__(self) -> None:
        logger.info(f"{Fore.MAGENTA}--- Summoning Configuration Runes ---{Style.RESET_ALL}")
        # --- API Credentials - Keys to the Exchange Vault ---
        self.api_key: str | None = self._get_env("BYBIT_API_KEY", required=True, color=Fore.RED)
        self.api_secret: str | None = self._get_env("BYBIT_API_SECRET", required=True, color=Fore.RED)

        # --- Trading Parameters - Core Incantation Variables ---
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT", color=Fore.YELLOW)
        self.interval: str = self._get_env("INTERVAL", "1m", color=Fore.YELLOW)  # Timeframe focus
        self.leverage: int = self._get_env("LEVERAGE", 10, cast_type=int, color=Fore.YELLOW)  # Power multiplier
        self.sleep_seconds: int = self._get_env("SLEEP_SECONDS", 10, cast_type=int, color=Fore.YELLOW)  # Pause between observations

        # --- Strategy Selection - Choosing the Path of Magic ---
        self.strategy_name: str = self._get_env("STRATEGY_NAME", "DUAL_SUPERTREND", color=Fore.CYAN).upper()
        self.valid_strategies: list[str] = ["DUAL_SUPERTREND", "STOCHRSI_MOMENTUM", "EHLERS_FISHER", "EHLERS_MA_CROSS"]
        if self.strategy_name not in self.valid_strategies:
            logger.critical(f"{Back.RED}{Fore.WHITE}Invalid STRATEGY_NAME '{self.strategy_name}'. Valid paths: {self.valid_strategies}{Style.RESET_ALL}")
            raise ValueError(f"Invalid STRATEGY_NAME '{self.strategy_name}'.")
        logger.info(f"{Fore.CYAN}Chosen Strategy Path: {self.strategy_name}{Style.RESET_ALL}")

        # --- Risk Management - Wards Against Ruin ---
        self.risk_per_trade_percentage: Decimal = self._get_env("RISK_PER_TRADE_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN)  # 0.5% risk per venture
        self.atr_stop_loss_multiplier: Decimal = self._get_env("ATR_STOP_LOSS_MULTIPLIER", "1.5", cast_type=Decimal, color=Fore.GREEN)  # Volatility-based ward distance
        self.max_order_usdt_amount: Decimal = self._get_env("MAX_ORDER_USDT_AMOUNT", "500.0", cast_type=Decimal, color=Fore.GREEN)  # Limit on position value
        self.required_margin_buffer: Decimal = self._get_env("REQUIRED_MARGIN_BUFFER", "1.05", cast_type=Decimal, color=Fore.GREEN)  # 5% safety margin

        # --- Trailing Stop Loss (Exchange Native) - The Adaptive Shield ---
        self.trailing_stop_percentage: Decimal = self._get_env("TRAILING_STOP_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN)  # 0.5% trailing distance
        self.trailing_stop_activation_offset_percent: Decimal = self._get_env("TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT", "0.001", cast_type=Decimal, color=Fore.GREEN)  # 0.1% offset before activation

        # --- Strategy-Specific Parameters - Tuning the Chosen Path ---
        # Dual Supertrend
        self.st_atr_length: int = self._get_env("ST_ATR_LENGTH", 7, cast_type=int, color=Fore.CYAN)
        self.st_multiplier: Decimal = self._get_env("ST_MULTIPLIER", "2.5", cast_type=Decimal, color=Fore.CYAN)
        self.confirm_st_atr_length: int = self._get_env("CONFIRM_ST_ATR_LENGTH", 5, cast_type=int, color=Fore.CYAN)
        self.confirm_st_multiplier: Decimal = self._get_env("CONFIRM_ST_MULTIPLIER", "2.0", cast_type=Decimal, color=Fore.CYAN)
        # StochRSI + Momentum
        self.stochrsi_rsi_length: int = self._get_env("STOCHRSI_RSI_LENGTH", 14, cast_type=int, color=Fore.CYAN)
        self.stochrsi_stoch_length: int = self._get_env("STOCHRSI_STOCH_LENGTH", 14, cast_type=int, color=Fore.CYAN)
        self.stochrsi_k_period: int = self._get_env("STOCHRSI_K_PERIOD", 3, cast_type=int, color=Fore.CYAN)
        self.stochrsi_d_period: int = self._get_env("STOCHRSI_D_PERIOD", 3, cast_type=int, color=Fore.CYAN)
        self.stochrsi_overbought: Decimal = self._get_env("STOCHRSI_OVERBOUGHT", "80.0", cast_type=Decimal, color=Fore.CYAN)
        self.stochrsi_oversold: Decimal = self._get_env("STOCHRSI_OVERSOLD", "20.0", cast_type=Decimal, color=Fore.CYAN)
        self.momentum_length: int = self._get_env("MOMENTUM_LENGTH", 5, cast_type=int, color=Fore.CYAN)
        # Ehlers Fisher Transform
        self.ehlers_fisher_length: int = self._get_env("EHLERS_FISHER_LENGTH", 10, cast_type=int, color=Fore.CYAN)
        self.ehlers_fisher_signal_length: int = self._get_env("EHLERS_FISHER_SIGNAL_LENGTH", 1, cast_type=int, color=Fore.CYAN)  # Default to 1 (often just the Fisher line itself)
        # Ehlers MA Cross (Placeholder - see function note)
        self.ehlers_fast_period: int = self._get_env("EHLERS_FAST_PERIOD", 10, cast_type=int, color=Fore.CYAN)
        self.ehlers_slow_period: int = self._get_env("EHLERS_SLOW_PERIOD", 30, cast_type=int, color=Fore.CYAN)

        # --- Confirmation Filters - Seeking Concordance in the Ether ---
        # Volume Analysis
        self.volume_ma_period: int = self._get_env("VOLUME_MA_PERIOD", 20, cast_type=int, color=Fore.YELLOW)
        self.volume_spike_threshold: Decimal = self._get_env("VOLUME_SPIKE_THRESHOLD", "1.5", cast_type=Decimal, color=Fore.YELLOW)  # Multiplier over MA
        self.require_volume_spike_for_entry: bool = self._get_env("REQUIRE_VOLUME_SPIKE_FOR_ENTRY", "false", cast_type=bool, color=Fore.YELLOW)
        # Order Book Analysis
        self.order_book_depth: int = self._get_env("ORDER_BOOK_DEPTH", 10, cast_type=int, color=Fore.YELLOW)  # Levels to analyze
        self.order_book_ratio_threshold_long: Decimal = self._get_env("ORDER_BOOK_RATIO_THRESHOLD_LONG", "1.2", cast_type=Decimal, color=Fore.YELLOW)  # Bid/Ask ratio for long
        self.order_book_ratio_threshold_short: Decimal = self._get_env("ORDER_BOOK_RATIO_THRESHOLD_SHORT", "0.8", cast_type=Decimal, color=Fore.YELLOW)  # Bid/Ask ratio for short
        self.fetch_order_book_per_cycle: bool = self._get_env("FETCH_ORDER_BOOK_PER_CYCLE", "false", cast_type=bool, color=Fore.YELLOW)  # Fetch OB every cycle or only on signal?

        # --- ATR Calculation (for Initial SL) ---
        self.atr_calculation_period: int = self._get_env("ATR_CALCULATION_PERIOD", 14, cast_type=int, color=Fore.GREEN)

        # --- Termux SMS Alerts - Whispers Through the Digital Veil ---
        self.enable_sms_alerts: bool = self._get_env("ENABLE_SMS_ALERTS", "false", cast_type=bool, color=Fore.MAGENTA)
        self.sms_recipient_number: str | None = self._get_env("SMS_RECIPIENT_NUMBER", None, color=Fore.MAGENTA)
        self.sms_timeout_seconds: int = self._get_env("SMS_TIMEOUT_SECONDS", 30, cast_type=int, color=Fore.MAGENTA)

        # --- CCXT / API Parameters - Tuning the Connection ---
        self.default_recv_window: int = 10000  # Milliseconds for API request validity
        self.order_book_fetch_limit: int = max(25, self.order_book_depth)  # Ensure sufficient depth fetched
        self.shallow_ob_fetch_depth: int = 5  # For quick price estimates
        self.order_fill_timeout_seconds: int = self._get_env("ORDER_FILL_TIMEOUT_SECONDS", 15, cast_type=int, color=Fore.YELLOW)  # Wait time for market order fill

        # --- Internal Constants - Fixed Arcane Symbols ---
        self.side_buy: str = "buy"
        self.side_sell: str = "sell"
        self.pos_long: str = "Long"
        self.pos_short: str = "Short"
        self.pos_none: str = "None"
        self.usdt_symbol: str = "USDT"  # The stable anchor
        self.retry_count: int = 3  # Attempts for certain API calls
        self.retry_delay_seconds: int = 2  # Pause between retries
        self.api_fetch_limit_buffer: int = 10  # Extra candles to fetch
        self.position_qty_epsilon: Decimal = Decimal("1e-9")  # Small value for float comparisons
        self.post_close_delay_seconds: int = 3  # Brief pause after closing a position

        logger.info(f"{Fore.MAGENTA}--- Configuration Runes Summoned and Verified ---{Style.RESET_ALL}")

    def _get_env(self, key: str, default: Any = None, cast_type: type = str, required: bool = False, color: str = Fore.WHITE) -> Any:
        """Fetches env var, casts type (including defaults), logs, handles defaults/errors with arcane grace.
        Ensures that default values are also cast to the specified type.
        """
        value_str = os.getenv(key)  # Get raw string value from environment
        final_value: Any = None  # Variable to hold the final, potentially casted value

        if value_str is None:
            # Environment variable not found, use the default
            if required:
                logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Required configuration rune '{key}' not found in the environment scroll (.env).{Style.RESET_ALL}")
                raise ValueError(f"Required environment variable '{key}' not set.")
            logger.debug(f"{color}Summoning {key}: Not Set (Using Default: '{default}'){Style.RESET_ALL}")
            final_value = default  # Assign default, still needs casting below
        else:
            # Environment variable found
            logger.debug(f"{color}Summoning {key}: Found Env Value '{value_str}'{Style.RESET_ALL}")
            final_value = value_str  # Assign found string, needs casting below

        # --- Attempt Casting (applies to both env var value and default value) ---
        if final_value is None and required:
            # This case handles if the default was None and it was required
            logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Required configuration rune '{key}' has no value or default.{Style.RESET_ALL}")
            raise ValueError(f"Required environment variable '{key}' has no value or default.")
        elif final_value is None:
            # Not required and no value/default, return None
            return None

        # Proceed with casting if a value (from env or default) exists
        try:
            if cast_type == bool:
                # Handle boolean casting explicitly
                final_value = str(final_value).lower() in ['true', '1', 'yes', 'y']
            elif cast_type == Decimal:
                # Handle Decimal casting
                final_value = Decimal(str(final_value))
            elif cast_type == int:
                # Handle Integer casting
                final_value = int(str(final_value))
            elif cast_type == float:
                 # Handle Float casting
                final_value = float(str(final_value))
            elif cast_type == str:
                # Already string or cast to string
                final_value = str(final_value)
            # Add other types if needed
            # else: No specific cast needed beyond string, leave as is (or handle error)

        except (ValueError, TypeError, InvalidOperation) as e:
            # Casting failed! Log error and use default (if possible), but try casting default again
            logger.error(f"{Fore.RED}Invalid type/value for {key}: '{final_value}' (from env or default). Expected {cast_type.__name__}. Error: {e}. Attempting default '{default}' again.{Style.RESET_ALL}")
            if default is None:
                 # If default is also None, and required, we have a problem (handled above)
                 # If not required, return None
                 if required:  # Should have been caught earlier, but double check
                     raise ValueError(f"Required env var '{key}' failed casting and has no valid default.")
                 return None
            else:
                # Try casting the default value one more time
                try:
                    if cast_type == bool: final_value = str(default).lower() in ['true', '1', 'yes', 'y']
                    elif cast_type == Decimal: final_value = Decimal(str(default))
                    elif cast_type == int: final_value = int(str(default))
                    elif cast_type == float: final_value = float(str(default))
                    elif cast_type == str: final_value = str(default)
                    # else: leave default as is
                    logger.warning(f"{Fore.YELLOW}Successfully used casted default value for {key}: '{final_value}'{Style.RESET_ALL}")
                except (ValueError, TypeError, InvalidOperation) as e_default:
                    # If even the default fails casting, it's a critical config issue
                    logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Failed to cast BOTH provided value and default value for {key}. Default='{default}', Type={cast_type.__name__}. Error: {e_default}{Style.RESET_ALL}")
                    raise ValueError(f"Configuration error: Cannot cast value or default for key '{key}' to {cast_type.__name__}.")

        # Log the final type and value being used
        logger.debug(f"{color}Using final value for {key}: {final_value} (Type: {type(final_value).__name__}){Style.RESET_ALL}")
        return final_value


# --- Logger Setup - The Oracle's Voice ---
LOGGING_LEVEL: int = logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]  # Output to the Termux console
)
logger: logging.Logger = logging.getLogger(__name__)

# Custom SUCCESS level and Neon Color Formatting for the Oracle
SUCCESS_LEVEL: int = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Logs a success message with mystical flair."""
    if self.isEnabledFor(SUCCESS_LEVEL): self._log(SUCCESS_LEVEL, message, args, **kwargs)  # pylint: disable=protected-access


logging.Logger.success = log_success  # type: ignore

# Apply colors if outputting to a TTY (like Termux)
if sys.stdout.isatty():
    logging.addLevelName(logging.DEBUG, f"{Fore.CYAN}{Style.DIM}{logging.getLevelName(logging.DEBUG)}{Style.RESET_ALL}")  # Dim Cyan for Debug
    logging.addLevelName(logging.INFO, f"{Fore.BLUE}{logging.getLevelName(logging.INFO)}{Style.RESET_ALL}")  # Blue for Info
    logging.addLevelName(SUCCESS_LEVEL, f"{Fore.MAGENTA}{Style.BRIGHT}{logging.getLevelName(SUCCESS_LEVEL)}{Style.RESET_ALL}")  # Bright Magenta for Success
    logging.addLevelName(logging.WARNING, f"{Fore.YELLOW}{Style.BRIGHT}{logging.getLevelName(logging.WARNING)}{Style.RESET_ALL}")  # Bright Yellow for Warning
    logging.addLevelName(logging.ERROR, f"{Fore.RED}{Style.BRIGHT}{logging.getLevelName(logging.ERROR)}{Style.RESET_ALL}")  # Bright Red for Error
    logging.addLevelName(logging.CRITICAL, f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}{logging.getLevelName(logging.CRITICAL)}{Style.RESET_ALL}")  # White on Red for Critical

# --- Global Objects - Instantiated Arcana ---
try:
    CONFIG = Config()  # Forge the configuration object
except ValueError:
    # Error already logged within Config init or _get_env
    sys.exit(1)


# --- Helper Functions - Minor Cantrips ---
def safe_decimal_conversion(value: Any, default: Decimal = Decimal("0.0")) -> Decimal:
    """Safely converts a value to Decimal, returning default if conversion fails."""
    try:
        return Decimal(str(value)) if value is not None else default
    except (InvalidOperation, TypeError, ValueError):
        logger.warning(f"Could not convert '{value}' to Decimal, using default {default}")
        return default


def format_order_id(order_id: str | int | None) -> str:
    """Returns the last 6 characters of an order ID or 'N/A' for brevity."""
    return str(order_id)[-6:] if order_id else "N/A"


# --- Precision Formatting - Shaping the Numbers ---
def format_price(exchange: ccxt.Exchange, symbol: str, price: float | Decimal) -> str:
    """Formats price according to market precision rules, guided by the exchange spirits."""
    try:
        # CCXT formatting methods often expect float input
        return exchange.price_to_precision(symbol, float(price))
    except Exception as e:
        logger.error(f"{Fore.RED}Error shaping price {price} for {symbol}: {e}{Style.RESET_ALL}")
        return str(Decimal(str(price)).normalize())  # Fallback to Decimal string representation


def format_amount(exchange: ccxt.Exchange, symbol: str, amount: float | Decimal) -> str:
    """Formats amount according to market precision rules, guided by the exchange spirits."""
    try:
        # CCXT formatting methods often expect float input
        return exchange.amount_to_precision(symbol, float(amount))
    except Exception as e:
        logger.error(f"{Fore.RED}Error shaping amount {amount} for {symbol}: {e}{Style.RESET_ALL}")
        return str(Decimal(str(amount)).normalize())  # Fallback to Decimal string representation


# --- Termux SMS Alert Function - Sending Whispers ---
_termux_sms_command_exists: bool | None = None  # Cache check result


def send_sms_alert(message: str) -> bool:
    """Sends an SMS alert using Termux API, a whisper through the digital veil."""
    global _termux_sms_command_exists

    if not CONFIG.enable_sms_alerts:
        logger.debug("SMS alerts disabled by configuration.")
        return False

    # Check for command existence only once
    if _termux_sms_command_exists is None:
        _termux_sms_command_exists = shutil.which('termux-sms-send') is not None
        if not _termux_sms_command_exists:
             logger.warning(f"{Fore.YELLOW}SMS failed: 'termux-sms-send' command not found. Ensure Termux:API is installed (`pkg install termux-api`) and configured.{Style.RESET_ALL}")

    if not _termux_sms_command_exists:
        return False  # Don't proceed if command is missing

    if not CONFIG.sms_recipient_number:
        logger.warning(f"{Fore.YELLOW}SMS alerts enabled, but SMS_RECIPIENT_NUMBER rune is missing.{Style.RESET_ALL}")
        return False

    try:
        # Prepare the command spell
        command: list[str] = ['termux-sms-send', '-n', CONFIG.sms_recipient_number, message]
        logger.info(f"{Fore.MAGENTA}Dispatching SMS whisper to {CONFIG.sms_recipient_number} (Timeout: {CONFIG.sms_timeout_seconds}s)...{Style.RESET_ALL}")
        # Execute the spell via subprocess
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=CONFIG.sms_timeout_seconds)
        if result.returncode == 0:
            logger.success(f"{Fore.MAGENTA}SMS whisper dispatched successfully.{Style.RESET_ALL}")
            return True
        else:
            logger.error(f"{Fore.RED}SMS whisper failed. RC: {result.returncode}, Stderr: {result.stderr.strip()}{Style.RESET_ALL}")
            return False
    except FileNotFoundError:
        # This shouldn't happen due to the check above, but handle defensively
        logger.error(f"{Fore.RED}SMS failed: 'termux-sms-send' vanished unexpectedly?{Style.RESET_ALL}")
        _termux_sms_command_exists = False  # Update cache
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"{Fore.RED}SMS failed: Command timed out after {CONFIG.sms_timeout_seconds}s.{Style.RESET_ALL}")
        return False
    except Exception as e:
        logger.error(f"{Fore.RED}SMS failed: Unexpected disturbance: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        return False


# --- Exchange Initialization - Opening the Portal ---
def initialize_exchange() -> ccxt.Exchange | None:
    """Initializes and returns the CCXT Bybit exchange instance, opening a portal."""
    logger.info(f"{Fore.BLUE}Opening portal to Bybit via CCXT...{Style.RESET_ALL}")
    if not CONFIG.api_key or not CONFIG.api_secret:
        logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: API Key/Secret runes missing. Cannot open portal.{Style.RESET_ALL}")
        send_sms_alert("[ScalpBot] CRITICAL: API keys missing. Spell failed.")
        return None
    try:
        # Forging the connection
        exchange = ccxt.bybit({
            "apiKey": CONFIG.api_key,
            "secret": CONFIG.api_secret,
            "enableRateLimit": True,  # Respect the exchange spirits' limits
            "options": {
                "defaultType": "linear",  # Assuming USDT perpetuals
                "recvWindow": CONFIG.default_recv_window,
                "adjustForTimeDifference": True,  # Sync with exchange time
            },
        })
        logger.debug("Loading market structures...")
        exchange.load_markets(True)  # Force reload for fresh data
        logger.debug("Checking initial balance...")
        exchange.fetch_balance()  # Initial check for authentication success
        logger.success(f"{Fore.GREEN}{Style.BRIGHT}Portal to Bybit Opened (LIVE SCALPING MODE - EXTREME CAUTION!).{Style.RESET_ALL}")
        send_sms_alert(f"[ScalpBot/{CONFIG.strategy_name}] Portal opened & authenticated.")  # Added strategy name
        return exchange
    except ccxt.AuthenticationError as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}Authentication failed: {e}. Check keys/IP/permissions.{Style.RESET_ALL}")
        send_sms_alert(f"[ScalpBot] CRITICAL: Authentication FAILED: {e}. Spell failed.")
    except ccxt.NetworkError as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}Network disturbance during portal opening: {e}. Check connection/Bybit status.{Style.RESET_ALL}")
        send_sms_alert(f"[ScalpBot] CRITICAL: Network Error on Init: {e}. Spell failed.")
    except ccxt.ExchangeError as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}Exchange spirit rejected portal opening: {e}. Check Bybit status/API docs.{Style.RESET_ALL}")
        send_sms_alert(f"[ScalpBot] CRITICAL: Exchange Error on Init: {e}. Spell failed.")
    except Exception as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}Unexpected chaos during portal opening: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[ScalpBot] CRITICAL: Unexpected Init Error: {type(e).__name__}. Spell failed.")
    return None


# --- Indicator Calculation Functions - Scrying the Market ---
# (Indicator functions remain largely the same as v2.1, focusing on clarity and Decimal usage)
def calculate_supertrend(df: pd.DataFrame, length: int, multiplier: Decimal, prefix: str = "") -> pd.DataFrame:
    """Calculates the Supertrend indicator using pandas_ta, returning Decimal where applicable."""
    col_prefix = f"{prefix}" if prefix else ""
    target_cols = [f"{col_prefix}supertrend", f"{col_prefix}trend", f"{col_prefix}st_long", f"{col_prefix}st_short"]
    st_col = f"SUPERT_{length}_{float(multiplier)}"  # pandas_ta uses float in name
    st_trend_col = f"SUPERTd_{length}_{float(multiplier)}"
    st_long_col = f"SUPERTl_{length}_{float(multiplier)}"
    st_short_col = f"SUPERTs_{length}_{float(multiplier)}"
    required_input_cols = ["high", "low", "close"]

    if df is None or df.empty or not all(c in df.columns for c in required_input_cols) or len(df) < length + 1:
        logger.warning(f"{Fore.YELLOW}Scrying ({col_prefix}ST): Insufficient data (Len: {len(df) if df is not None else 0}, Need: {length + 1}).{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA
        return df

    try:
        # pandas_ta expects float multiplier
        logger.debug(f"Scrying ({col_prefix}ST): Calculating with length={length}, multiplier={float(multiplier)}")
        df.ta.supertrend(length=length, multiplier=float(multiplier), append=True)
        if st_col not in df.columns or st_trend_col not in df.columns:
            raise KeyError(f"pandas_ta failed to create expected raw columns: {st_col}, {st_trend_col}")

        # Convert Supertrend value to Decimal, interpret trend
        df[f"{col_prefix}supertrend"] = df[st_col].apply(safe_decimal_conversion)
        df[f"{col_prefix}trend"] = df[st_trend_col] == 1  # Boolean: True for Uptrend (1), False for Downtrend (-1)
        prev_trend = df[st_trend_col].shift(1)
        df[f"{col_prefix}st_long"] = (prev_trend == -1) & (df[st_trend_col] == 1)  # Boolean: Trend flipped to Long
        df[f"{col_prefix}st_short"] = (prev_trend == 1) & (df[st_trend_col] == -1)  # Boolean: Trend flipped to Short

        # Clean up raw columns from pandas_ta
        raw_st_cols = [st_col, st_trend_col, st_long_col, st_short_col]
        df.drop(columns=raw_st_cols, errors='ignore', inplace=True)

        # Log the latest reading
        last_st_val = df[f'{col_prefix}supertrend'].iloc[-1]
        if pd.notna(last_st_val):
            last_trend = 'Up' if df[f'{col_prefix}trend'].iloc[-1] else 'Down'
            signal = 'LONG FLIP' if df[f'{col_prefix}st_long'].iloc[-1] else ('SHORT FLIP' if df[f'{col_prefix}st_short'].iloc[-1] else 'Hold')
            logger.debug(f"Scrying ({col_prefix}ST({length},{multiplier})): Trend={Fore.GREEN if last_trend == 'Up' else Fore.RED}{last_trend}{Style.RESET_ALL}, Val={last_st_val:.4f}, Signal={signal}")
        else:
            logger.debug(f"Scrying ({col_prefix}ST({length},{multiplier})): Resulted in NA for last candle.")

    except Exception as e:
        logger.error(f"{Fore.RED}Scrying ({col_prefix}ST): Error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA  # Nullify results on error
    return df


def analyze_volume_atr(df: pd.DataFrame, atr_len: int, vol_ma_len: int) -> dict[str, Decimal | None]:
    """Calculates ATR, Volume MA, checks spikes. Returns Decimals representing volatility and energy."""
    results: dict[str, Decimal | None] = {"atr": None, "volume_ma": None, "last_volume": None, "volume_ratio": None}
    required_cols = ["high", "low", "close", "volume"]
    min_len = max(atr_len, vol_ma_len) + 1  # Need at least period+1

    if df is None or df.empty or not all(c in df.columns for c in required_cols) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Scrying (Vol/ATR): Insufficient data (Len: {len(df) if df is not None else 0}, Need: {min_len}).{Style.RESET_ALL}")
        return results

    try:
        # Calculate ATR (Average True Range) - Measure of volatility
        logger.debug(f"Scrying (ATR): Calculating with length={atr_len}")
        atr_col = f"ATRr_{atr_len}"
        df.ta.atr(length=atr_len, append=True)
        if atr_col in df.columns:
            last_atr = df[atr_col].iloc[-1]
            if pd.notna(last_atr): results["atr"] = safe_decimal_conversion(last_atr)
            df.drop(columns=[atr_col], errors='ignore', inplace=True)  # Clean up raw column

        # Calculate Volume Moving Average and Ratio - Measure of market energy
        logger.debug(f"Scrying (Volume): Calculating MA with length={vol_ma_len}")
        volume_ma_col = 'volume_ma'
        df[volume_ma_col] = df['volume'].rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2)).mean()
        last_vol_ma = df[volume_ma_col].iloc[-1]
        last_vol = df['volume'].iloc[-1]

        if pd.notna(last_vol_ma): results["volume_ma"] = safe_decimal_conversion(last_vol_ma)
        if pd.notna(last_vol): results["last_volume"] = safe_decimal_conversion(last_vol)

        # Calculate Volume Ratio (Last Volume / Volume MA)
        if results["volume_ma"] is not None and results["volume_ma"] > CONFIG.position_qty_epsilon and results["last_volume"] is not None:
            try:
                results["volume_ratio"] = results["last_volume"] / results["volume_ma"]
            except Exception:  # Handles potential division by zero if MA is epsilon
                 results["volume_ratio"] = None
        else:
            results["volume_ratio"] = None  # Cannot calculate ratio

        if volume_ma_col in df.columns: df.drop(columns=[volume_ma_col], errors='ignore', inplace=True)  # Clean up MA column

        # Log results
        atr_str = f"{results['atr']:.5f}" if results['atr'] else 'N/A'
        vol_ma_str = f"{results['volume_ma']:.2f}" if results['volume_ma'] else 'N/A'
        vol_ratio_str = f"{results['volume_ratio']:.2f}" if results['volume_ratio'] else 'N/A'
        logger.debug(f"Scrying Results: ATR({atr_len})={Fore.CYAN}{atr_str}{Style.RESET_ALL}, VolMA({vol_ma_len})={vol_ma_str}, VolRatio={Fore.YELLOW}{vol_ratio_str}{Style.RESET_ALL}")

    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (Vol/ATR): Error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        results = dict.fromkeys(results)  # Nullify results on error
    return results


def calculate_stochrsi_momentum(df: pd.DataFrame, rsi_len: int, stoch_len: int, k: int, d: int, mom_len: int) -> pd.DataFrame:
    """Calculates StochRSI and Momentum, gauging overbought/oversold and trend strength."""
    target_cols = ['stochrsi_k', 'stochrsi_d', 'momentum']
    min_len = max(rsi_len + stoch_len, mom_len) + 5  # Add buffer for calculation stability
    if df is None or df.empty or not all(c in df.columns for c in ["close"]) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Scrying (StochRSI/Mom): Insufficient data (Len: {len(df) if df is not None else 0}, Need ~{min_len}).{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA
        return df
    try:
        # Calculate StochRSI
        logger.debug(f"Scrying (StochRSI): Calculating with RSI={rsi_len}, Stoch={stoch_len}, K={k}, D={d}")
        stochrsi_df = df.ta.stochrsi(length=stoch_len, rsi_length=rsi_len, k=k, d=d, append=False)  # Calculate separately
        k_col, d_col = f"STOCHRSIk_{stoch_len}_{rsi_len}_{k}_{d}", f"STOCHRSId_{stoch_len}_{rsi_len}_{k}_{d}"
        if k_col in stochrsi_df.columns: df['stochrsi_k'] = stochrsi_df[k_col].apply(safe_decimal_conversion)
        else: logger.warning("StochRSI K column not found after calculation"); df['stochrsi_k'] = pd.NA
        if d_col in stochrsi_df.columns: df['stochrsi_d'] = stochrsi_df[d_col].apply(safe_decimal_conversion)
        else: logger.warning("StochRSI D column not found after calculation"); df['stochrsi_d'] = pd.NA

        # Calculate Momentum
        logger.debug(f"Scrying (Momentum): Calculating with length={mom_len}")
        mom_col = f"MOM_{mom_len}"
        df.ta.mom(length=mom_len, append=True)
        if mom_col in df.columns:
            df['momentum'] = df[mom_col].apply(safe_decimal_conversion)
            df.drop(columns=[mom_col], errors='ignore', inplace=True)  # Clean up raw column
        else: logger.warning("Momentum column not found after calculation"); df['momentum'] = pd.NA

        # Log latest values
        k_val, d_val, mom_val = df['stochrsi_k'].iloc[-1], df['stochrsi_d'].iloc[-1], df['momentum'].iloc[-1]
        if pd.notna(k_val) and pd.notna(d_val) and pd.notna(mom_val):
            k_color = Fore.RED if k_val > CONFIG.stochrsi_overbought else (Fore.GREEN if k_val < CONFIG.stochrsi_oversold else Fore.CYAN)
            d_color = Fore.RED if d_val > CONFIG.stochrsi_overbought else (Fore.GREEN if d_val < CONFIG.stochrsi_oversold else Fore.CYAN)
            mom_color = Fore.GREEN if mom_val > 0 else (Fore.RED if mom_val < 0 else Fore.WHITE)
            logger.debug(f"Scrying (StochRSI/Mom): K={k_color}{k_val:.2f}{Style.RESET_ALL}, D={d_color}{d_val:.2f}{Style.RESET_ALL}, Mom={mom_color}{mom_val:.4f}{Style.RESET_ALL}")
        else:
            logger.debug("Scrying (StochRSI/Mom): Resulted in NA for last candle.")

    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (StochRSI/Mom): Error during calculation: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA  # Nullify results on error
    return df


def calculate_ehlers_fisher(df: pd.DataFrame, length: int, signal: int) -> pd.DataFrame:
    """Calculates Ehlers Fisher Transform, seeking cyclical turning points."""
    target_cols = ['ehlers_fisher', 'ehlers_signal']
    if df is None or df.empty or not all(c in df.columns for c in ["high", "low"]) or len(df) < length + 1:
        logger.warning(f"{Fore.YELLOW}Scrying (EhlersFisher): Insufficient data (Len: {len(df) if df is not None else 0}, Need {length + 1}).{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA
        return df
    try:
        logger.debug(f"Scrying (EhlersFisher): Calculating with length={length}, signal={signal}")
        fisher_df = df.ta.fisher(length=length, signal=signal, append=False)  # Calculate separately
        fish_col, signal_col = f"FISHERT_{length}_{signal}", f"FISHERTs_{length}_{signal}"
        if fish_col in fisher_df.columns: df['ehlers_fisher'] = fisher_df[fish_col].apply(safe_decimal_conversion)
        else: logger.warning("Ehlers Fisher column not found after calculation"); df['ehlers_fisher'] = pd.NA
        if signal_col in fisher_df.columns: df['ehlers_signal'] = fisher_df[signal_col].apply(safe_decimal_conversion)
        else: logger.warning("Ehlers Signal column not found after calculation"); df['ehlers_signal'] = pd.NA

        # Log latest values
        fish_val, sig_val = df['ehlers_fisher'].iloc[-1], df['ehlers_signal'].iloc[-1]
        if pd.notna(fish_val) and pd.notna(sig_val):
             logger.debug(f"Scrying (EhlersFisher({length},{signal})): Fisher={Fore.CYAN}{fish_val:.4f}{Style.RESET_ALL}, Signal={Fore.MAGENTA}{sig_val:.4f}{Style.RESET_ALL}")
        else:
             logger.debug("Scrying (EhlersFisher): Resulted in NA for last candle.")
    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (EhlersFisher): Error during calculation: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA  # Nullify results on error
    return df


def calculate_ehlers_ma(df: pd.DataFrame, fast_len: int, slow_len: int) -> pd.DataFrame:
    """Calculates Ehlers Super Smoother Moving Averages (Placeholder: Uses EMA)."""
    target_cols = ['fast_ema', 'slow_ema']
    min_len = max(fast_len, slow_len) + 5  # Add buffer for calculation stability
    if df is None or df.empty or not all(c in df.columns for c in ["close"]) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Scrying (EhlersMA): Insufficient data (Len: {len(df) if df is not None else 0}, Need ~{min_len}).{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA
        return df
    try:
        # *** PYRMETHUS NOTE: Using standard EMA as a placeholder. ***
        # The true Ehlers Super Smoother involves a more complex filter.
        # If `pandas_ta.supersmoother` exists and is reliable, use it. Otherwise, implement the filter manually
        # or accept EMA as an approximation for this strategy path.
        logger.warning(f"{Fore.YELLOW}{Style.DIM}Scrying (EhlersMA): Using EMA as placeholder for Ehlers Super Smoother. Verify suitability.{Style.RESET_ALL}")
        logger.debug(f"Scrying (EhlersMA - EMA Placeholder): Calculating Fast EMA({fast_len}), Slow EMA({slow_len})")
        df['fast_ema'] = df.ta.ema(length=fast_len).apply(safe_decimal_conversion)
        df['slow_ema'] = df.ta.ema(length=slow_len).apply(safe_decimal_conversion)

        # Log latest values
        fast_val, slow_val = df['fast_ema'].iloc[-1], df['slow_ema'].iloc[-1]
        if pd.notna(fast_val) and pd.notna(slow_val):
            logger.debug(f"Scrying (EhlersMA({fast_len},{slow_len})): Fast={Fore.GREEN}{fast_val:.4f}{Style.RESET_ALL}, Slow={Fore.RED}{slow_val:.4f}{Style.RESET_ALL}")
        else:
             logger.debug("Scrying (EhlersMA): Resulted in NA for last candle.")
    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (EhlersMA): Error during calculation: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA  # Nullify results on error
    return df


def analyze_order_book(exchange: ccxt.Exchange, symbol: str, depth: int, fetch_limit: int) -> dict[str, Decimal | None]:
    """Fetches and analyzes L2 order book pressure and spread, peering into market intent."""
    results: dict[str, Decimal | None] = {"bid_ask_ratio": None, "spread": None, "best_bid": None, "best_ask": None}
    logger.debug(f"Order Book Scrying: Fetching L2 {symbol} (Depth:{depth}, Limit:{fetch_limit})...")
    if not exchange.has.get('fetchL2OrderBook'):
        logger.warning(f"{Fore.YELLOW}Order Book Scrying: fetchL2OrderBook not supported by {exchange.id}. Cannot peer into depth.{Style.RESET_ALL}")
        return results
    try:
        # Fetching the order book's current state
        order_book = exchange.fetch_l2_order_book(symbol, limit=fetch_limit)
        bids: list[list[float | str]] = order_book.get('bids', [])
        asks: list[list[float | str]] = order_book.get('asks', [])

        # Extract best bid/ask with Decimal precision
        best_bid = safe_decimal_conversion(bids[0][0]) if bids and len(bids[0]) > 0 else None
        best_ask = safe_decimal_conversion(asks[0][0]) if asks and len(asks[0]) > 0 else None
        results["best_bid"] = best_bid
        results["best_ask"] = best_ask

        # Calculate spread
        if best_bid is not None and best_ask is not None and best_bid > 0 and best_ask > 0:
            results["spread"] = best_ask - best_bid
            logger.debug(f"OB Scrying: Best Bid={Fore.GREEN}{best_bid:.4f}{Style.RESET_ALL}, Ask={Fore.RED}{best_ask:.4f}{Style.RESET_ALL}, Spread={Fore.YELLOW}{results['spread']:.4f}{Style.RESET_ALL}")
        else:
            logger.debug(f"OB Scrying: Bid={best_bid or 'N/A'}, Ask={best_ask or 'N/A'} (Spread N/A)")

        # Sum volumes within the specified depth using Decimal
        bid_vol = sum(safe_decimal_conversion(bid[1]) for bid in bids[:depth] if len(bid) > 1)
        ask_vol = sum(safe_decimal_conversion(ask[1]) for ask in asks[:depth] if len(ask) > 1)
        logger.debug(f"OB Scrying (Depth {depth}): BidVol={Fore.GREEN}{bid_vol:.4f}{Style.RESET_ALL}, AskVol={Fore.RED}{ask_vol:.4f}{Style.RESET_ALL}")

        # Calculate Bid/Ask Volume Ratio
        if ask_vol > CONFIG.position_qty_epsilon:
            try:
                results["bid_ask_ratio"] = bid_vol / ask_vol
                ratio_color = Fore.GREEN if results["bid_ask_ratio"] >= CONFIG.order_book_ratio_threshold_long else (Fore.RED if results["bid_ask_ratio"] <= CONFIG.order_book_ratio_threshold_short else Fore.YELLOW)
                logger.debug(f"OB Scrying Ratio: {ratio_color}{results['bid_ask_ratio']:.3f}{Style.RESET_ALL}")
            except Exception as e:
                logger.warning(f"Error calculating OB ratio: {e}")
                results["bid_ask_ratio"] = None
        else:
            logger.debug("OB Scrying Ratio: N/A (Ask volume zero or negligible)")

    except (ccxt.NetworkError, ccxt.ExchangeError, IndexError, Exception) as e:
        logger.warning(f"{Fore.YELLOW}Order Book Scrying Error for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        results = dict.fromkeys(results)  # Reset on error
    return results


# --- Data Fetching - Gathering Etheric Data Streams ---
def get_market_data(exchange: ccxt.Exchange, symbol: str, interval: str, limit: int) -> pd.DataFrame | None:
    """Fetches and prepares OHLCV data, ensuring numeric types and handling gaps."""
    if not exchange.has.get("fetchOHLCV"):
        logger.error(f"{Fore.RED}Data Fetch: Exchange '{exchange.id}' does not support fetchOHLCV.{Style.RESET_ALL}")
        return None
    try:
        logger.debug(f"Data Fetch: Gathering {limit} OHLCV candles for {symbol} ({interval})...")
        # Channeling the data stream
        ohlcv: list[list[int | float | str]] = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
        if not ohlcv:
            logger.warning(f"{Fore.YELLOW}Data Fetch: No OHLCV data returned for {symbol} ({interval}). Market asleep?{Style.RESET_ALL}")
            return None

        # Weaving data into a DataFrame structure
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)  # Time magic
        df.set_index("timestamp", inplace=True)

        # Convert to numeric, coercing errors, check NaNs robustly
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Ensure numeric essence

        # Check for and handle gaps (NaNs)
        if df.isnull().values.any():
            nan_counts = df.isnull().sum()
            logger.warning(f"{Fore.YELLOW}Data Fetch: OHLCV contains gaps (NaNs) after conversion:\n{nan_counts[nan_counts > 0]}\nAttempting forward fill...{Style.RESET_ALL}")
            df.ffill(inplace=True)  # Attempt to fill gaps with previous values
            if df.isnull().values.any():  # Check again
                logger.warning(f"{Fore.YELLOW}Gaps remain after ffill, attempting backward fill...{Style.RESET_ALL}")
                df.bfill(inplace=True)  # Attempt to fill remaining gaps with next values
                if df.isnull().values.any():
                    logger.error(f"{Fore.RED}Data Fetch: Gaps persist after ffill/bfill. Cannot proceed with unreliable data.{Style.RESET_ALL}")
                    return None  # Cannot proceed if gaps remain at start/end

        logger.debug(f"Data Fetch: Woven {len(df)} OHLCV candles for {symbol}.")
        return df

    except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
        logger.warning(f"{Fore.YELLOW}Data Fetch: Disturbance gathering OHLCV for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
    return None


# --- Position & Order Management - Manipulating Market Presence ---
def get_current_position(exchange: ccxt.Exchange, symbol: str) -> dict[str, Any]:
    """Fetches current position details (Bybit V5 focus), returns Decimals representing market presence."""
    default_pos: dict[str, Any] = {'side': CONFIG.pos_none, 'qty': Decimal("0.0"), 'entry_price': Decimal("0.0")}
    market_id = None
    market = None
    try:
        # Identify the market structure
        market = exchange.market(symbol)
        market_id = market['id']  # The exchange's name for the symbol
    except Exception as e:
        logger.error(f"{Fore.RED}Position Check: Failed to identify market structure for '{symbol}': {e}{Style.RESET_ALL}")
        return default_pos

    try:
        if not exchange.has.get('fetchPositions'):
            logger.warning(f"{Fore.YELLOW}Position Check: fetchPositions spell not available for {exchange.id}.{Style.RESET_ALL}")
            return default_pos

        # Bybit V5 requires 'category' parameter (linear/inverse)
        params = {'category': 'linear'} if market.get('linear') else ({'category': 'inverse'} if market.get('inverse') else {})
        logger.debug(f"Position Check: Querying positions for {symbol} (MarketID: {market_id}) with params: {params}")

        # Summon position data from the exchange
        fetched_positions = exchange.fetch_positions(symbols=[symbol], params=params)

        # Bybit V5 might return multiple entries; find the active one for One-Way mode (positionIdx=0)
        active_pos = None
        for pos in fetched_positions:
            pos_info = pos.get('info', {})
            pos_market_id = pos_info.get('symbol')
            position_idx = pos_info.get('positionIdx', -1)  # Use -1 default to catch issues
            pos_side_v5 = pos_info.get('side', 'None')  # 'Buy' for long, 'Sell' for short, 'None' if flat
            size_str = pos_info.get('size')

            # Filter for the correct symbol and One-Way mode active position
            if pos_market_id == market_id and position_idx == 0 and pos_side_v5 != 'None':
                size = safe_decimal_conversion(size_str)
                if abs(size) > CONFIG.position_qty_epsilon:  # Check if size is non-negligible
                    active_pos = pos  # Found the active position
                    logger.debug(f"Found potential active position entry: {pos_info}")
                    break  # Assume only one active position in One-Way mode

        if active_pos:
            try:
                # Parse details from the active position info
                size = safe_decimal_conversion(active_pos.get('info', {}).get('size'))
                # Use 'avgPrice' from info for V5 entry price
                entry_price = safe_decimal_conversion(active_pos.get('info', {}).get('avgPrice'))
                # Determine side based on V5 'side' field ('Buy' or 'Sell')
                side = CONFIG.pos_long if active_pos.get('info', {}).get('side') == 'Buy' else CONFIG.pos_short

                pos_color = Fore.GREEN if side == CONFIG.pos_long else Fore.RED
                logger.info(f"{pos_color}Position Check: Found ACTIVE {side} position: Qty={abs(size):.8f} @ Entry={entry_price:.4f}{Style.RESET_ALL}")
                return {'side': side, 'qty': abs(size), 'entry_price': entry_price}
            except Exception as parse_err:
                 logger.warning(f"{Fore.YELLOW}Position Check: Error parsing active position data: {parse_err}. Data: {active_pos}{Style.RESET_ALL}")
                 return default_pos  # Return default on parsing error
        else:
            logger.info(f"{Fore.BLUE}Position Check: No active One-Way position found for {market_id}. Currently Flat.{Style.RESET_ALL}")
            return default_pos

    except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
        logger.warning(f"{Fore.YELLOW}Position Check: Disturbance querying positions for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
    return default_pos  # Return default on API error


def set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int) -> bool:
    """Sets leverage for a futures symbol (Bybit V5 focus), adjusting the power multiplier."""
    logger.info(f"{Fore.CYAN}Leverage Conjuring: Attempting to set {leverage}x for {symbol}...{Style.RESET_ALL}")
    try:
        # Verify it's a contract market where leverage applies
        market = exchange.market(symbol)
        if not market.get('contract'):
            logger.error(f"{Fore.RED}Leverage Conjuring: Cannot set leverage for non-contract market: {symbol}.{Style.RESET_ALL}")
            return False
    except Exception as e:
         logger.error(f"{Fore.RED}Leverage Conjuring: Failed to identify market structure for '{symbol}': {e}{Style.RESET_ALL}")
         return False

    for attempt in range(CONFIG.retry_count):
        try:
            # Bybit V5 requires setting buy and sell leverage separately
            params = {'buyLeverage': str(leverage), 'sellLeverage': str(leverage)}
            response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
            logger.success(f"{Fore.GREEN}Leverage Conjuring: Set to {leverage}x for {symbol}. Response: {response}{Style.RESET_ALL}")
            return True
        except ccxt.ExchangeError as e:
            # Check for common "already set" or "no modification needed" messages
            err_str = str(e).lower()
            if "leverage not modified" in err_str or "same as requested" in err_str or "same leverage" in err_str:
                logger.info(f"{Fore.CYAN}Leverage Conjuring: Already set to {leverage}x for {symbol}.{Style.RESET_ALL}")
                return True
            logger.warning(f"{Fore.YELLOW}Leverage Conjuring: Exchange resistance (Attempt {attempt + 1}/{CONFIG.retry_count}): {e}{Style.RESET_ALL}")
            if attempt < CONFIG.retry_count - 1: time.sleep(CONFIG.retry_delay_seconds)
            else: logger.error(f"{Fore.RED}Leverage Conjuring: Failed after {CONFIG.retry_count} attempts.{Style.RESET_ALL}")
        except (ccxt.NetworkError, Exception) as e:
            logger.warning(f"{Fore.YELLOW}Leverage Conjuring: Network/Other disturbance (Attempt {attempt + 1}/{CONFIG.retry_count}): {e}{Style.RESET_ALL}")
            if attempt < CONFIG.retry_count - 1: time.sleep(CONFIG.retry_delay_seconds)
            else: logger.error(f"{Fore.RED}Leverage Conjuring: Failed after {CONFIG.retry_count} attempts.{Style.RESET_ALL}")
    return False  # Failed after retries


def close_position(exchange: ccxt.Exchange, symbol: str, position_to_close: dict[str, Any], reason: str = "Signal") -> dict[str, Any] | None:
    """Closes the specified active position with re-validation, banishing market presence."""
    initial_side = position_to_close.get('side', CONFIG.pos_none)
    initial_qty = position_to_close.get('qty', Decimal("0.0"))
    market_base = symbol.split('/')[0]  # For concise alerts
    logger.info(f"{Fore.YELLOW}Banish Position: Initiated for {symbol}. Reason: {reason}. Initial state: {initial_side} Qty={initial_qty:.8f}{Style.RESET_ALL}")

    # Re-validate the position just before closing - ensure it still exists
    live_position = get_current_position(exchange, symbol)
    if live_position['side'] == CONFIG.pos_none:
        logger.warning(f"{Fore.YELLOW}Banish Position: Re-validation shows NO active position for {symbol}. Aborting banishment.{Style.RESET_ALL}")
        if initial_side != CONFIG.pos_none: logger.warning(f"{Fore.YELLOW}Banish Position: Discrepancy detected (was {initial_side}, now None).{Style.RESET_ALL}")
        return None  # Nothing to close

    live_amount_to_close = live_position['qty']
    live_position_side = live_position['side']
    # Determine the opposite side needed to close the position
    side_to_execute_close = CONFIG.side_sell if live_position_side == CONFIG.pos_long else CONFIG.side_buy

    try:
        # Format amount according to market rules
        amount_str = format_amount(exchange, symbol, live_amount_to_close)
        amount_float = float(amount_str)  # CCXT create order often expects float

        # Check if the amount is negligible after formatting
        if amount_float <= float(CONFIG.position_qty_epsilon):
            logger.error(f"{Fore.RED}Banish Position: Closing amount negligible ({amount_str}) after precision shaping. Aborting.{Style.RESET_ALL}")
            return None

        # Execute the market close order with reduceOnly flag
        logger.warning(f"{Back.YELLOW}{Fore.BLACK}Banish Position: Attempting to CLOSE {live_position_side} ({reason}): "
                       f"Exec {side_to_execute_close.upper()} MARKET {amount_str} {symbol} (reduce_only=True)...{Style.RESET_ALL}")
        params = {'reduceOnly': True}  # Ensure this order only closes, not opens
        order = exchange.create_market_order(symbol=symbol, side=side_to_execute_close, amount=amount_float, params=params)

        # Parse order response safely using Decimal
        fill_price = safe_decimal_conversion(order.get('average'))
        filled_qty = safe_decimal_conversion(order.get('filled'))
        cost = safe_decimal_conversion(order.get('cost'))
        order_id_short = format_order_id(order.get('id'))

        close_color = Fore.GREEN if (live_position_side == CONFIG.pos_long and side_to_execute_close == CONFIG.side_sell) or \
                                     (live_position_side == CONFIG.pos_short and side_to_execute_close == CONFIG.side_buy) else Fore.YELLOW  # Should always be green if logic is right

        logger.success(f"{close_color}{Style.BRIGHT}Banish Position: Order ({reason}) placed for {symbol}. "
                       f"Filled: {filled_qty:.8f}/{amount_str}, AvgFill: {fill_price:.4f}, Cost: {cost:.2f} USDT. ID:...{order_id_short}{Style.RESET_ALL}")
        send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] BANISHED {live_position_side} {amount_str} @ ~{fill_price:.4f} ({reason}). ID:...{order_id_short}")
        return order  # Return the filled close order details

    except (ccxt.InsufficientFunds, ccxt.NetworkError, ccxt.ExchangeError, ValueError, Exception) as e:
        logger.error(f"{Fore.RED}Banish Position ({reason}): Failed for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        # Check for specific Bybit errors indicating already closed or zero position
        err_str = str(e).lower()
        if isinstance(e, ccxt.ExchangeError) and ("order would not reduce position size" in err_str or "position is zero" in err_str or "position size is zero" in err_str or "cannot be less than" in err_str):  # Added common size error
             logger.warning(f"{Fore.YELLOW}Banish Position: Exchange indicates position already closed/closing or zero size. Assuming banished.{Style.RESET_ALL}")
             return None  # Treat as success (or non-actionable) in this case
        send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ERROR Banishing ({reason}): {type(e).__name__}. Check logs.")
    return None  # Failed to close


def calculate_position_size(equity: Decimal, risk_per_trade_pct: Decimal, entry_price: Decimal, stop_loss_price: Decimal,
                            leverage: int, symbol: str, exchange: ccxt.Exchange) -> tuple[Decimal | None, Decimal | None]:
    """Calculates position size and estimated margin based on risk, using Decimal precision."""
    logger.debug(f"Risk Calc: Equity={equity:.4f}, Risk%={risk_per_trade_pct:.4%}, Entry={entry_price:.4f}, SL={stop_loss_price:.4f}, Lev={leverage}x")
    # --- Input Validation ---
    if not (entry_price > 0 and stop_loss_price > 0):
        logger.error(f"{Fore.RED}Risk Calc: Invalid entry/SL price (<= 0).{Style.RESET_ALL}"); return None, None
    price_diff = abs(entry_price - stop_loss_price)
    if price_diff < CONFIG.position_qty_epsilon:
        logger.error(f"{Fore.RED}Risk Calc: Entry/SL prices too close ({price_diff:.8f}). Cannot calculate risk.{Style.RESET_ALL}"); return None, None
    if not 0 < risk_per_trade_pct < 1:
        logger.error(f"{Fore.RED}Risk Calc: Invalid risk percentage: {risk_per_trade_pct:.4%}. Must be between 0 and 1.{Style.RESET_ALL}"); return None, None
    if equity <= 0:
        logger.error(f"{Fore.RED}Risk Calc: Invalid equity: {equity:.4f}.{Style.RESET_ALL}"); return None, None
    if leverage <= 0:
        logger.error(f"{Fore.RED}Risk Calc: Invalid leverage: {leverage}.{Style.RESET_ALL}"); return None, None

    # --- Calculation ---
    risk_amount_usdt = equity * risk_per_trade_pct  # Max USDT amount to risk on this trade
    # Assuming linear contract where 1 unit = 1 base currency (e.g., 1 BTC)
    # Risk per unit of the asset = price_diff (difference between entry and stop loss)
    # Quantity = Total Risk Amount / Risk per Unit
    quantity_raw = risk_amount_usdt / price_diff

    # --- Apply Market Precision ---
    try:
        # Format according to market precision *then* convert back to Decimal
        quantity_precise_str = format_amount(exchange, symbol, quantity_raw)
        quantity_precise = Decimal(quantity_precise_str)
        logger.debug(f"Risk Calc: Raw Qty={quantity_raw:.8f}, Precise Qty={quantity_precise:.8f}")
    except Exception as e:
        logger.warning(f"{Fore.YELLOW}Risk Calc: Failed precision shaping for quantity {quantity_raw:.8f}. Using raw with fallback quantization. Error: {e}{Style.RESET_ALL}")
        # Fallback: Quantize to a reasonable number of decimal places if formatting fails
        quantity_precise = quantity_raw.quantize(Decimal('1e-8'), rounding=ROUND_HALF_UP)

    # --- Final Checks & Margin Estimation ---
    if quantity_precise <= CONFIG.position_qty_epsilon:
        logger.warning(f"{Fore.YELLOW}Risk Calc: Calculated quantity negligible ({quantity_precise:.8f}). RiskAmt={risk_amount_usdt:.4f}, PriceDiff={price_diff:.4f}{Style.RESET_ALL}")
        return None, None

    # Estimate position value and margin required
    pos_value_usdt = quantity_precise * entry_price
    required_margin = pos_value_usdt / Decimal(leverage)
    logger.debug(f"Risk Calc Result: Qty={Fore.CYAN}{quantity_precise:.8f}{Style.RESET_ALL}, EstValue={pos_value_usdt:.4f}, EstMargin={required_margin:.4f}")
    return quantity_precise, required_margin


def wait_for_order_fill(exchange: ccxt.Exchange, order_id: str, symbol: str, timeout_seconds: int) -> dict[str, Any] | None:
    """Waits for a specific order to be filled (status 'closed'), observing the order's fate."""
    start_time = time.time()
    order_id_short = format_order_id(order_id)
    logger.info(f"{Fore.CYAN}Observing order ...{order_id_short} ({symbol}) for fill (Timeout: {timeout_seconds}s)...{Style.RESET_ALL}")
    while time.time() - start_time < timeout_seconds:
        try:
            # Query the order's status
            order = exchange.fetch_order(order_id, symbol)
            status = order.get('status')
            logger.debug(f"Order ...{order_id_short} status: {status}")

            if status == 'closed':  # 'closed' usually means fully filled for market orders
                logger.success(f"{Fore.GREEN}Order ...{order_id_short} confirmed FILLED.{Style.RESET_ALL}")
                return order  # Success
            elif status in ['canceled', 'rejected', 'expired']:
                logger.error(f"{Fore.RED}Order ...{order_id_short} failed with status '{status}'.{Style.RESET_ALL}")
                return None  # Failed state
            # Continue polling if 'open', 'partially_filled', or None/unknown status

            time.sleep(0.5)  # Check every 500ms

        except ccxt.OrderNotFound:
            # This might happen briefly after placing, especially on busy exchanges. Keep trying.
            logger.warning(f"{Fore.YELLOW}Order ...{order_id_short} not found yet by exchange spirits. Retrying...{Style.RESET_ALL}")
            time.sleep(1)  # Wait a bit longer if not found initially
        except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
            logger.warning(f"{Fore.YELLOW}Disturbance checking order ...{order_id_short}: {type(e).__name__} - {e}. Retrying...{Style.RESET_ALL}")
            time.sleep(1)  # Wait longer on error before retrying

    # If the loop finishes without returning, it timed out
    logger.error(f"{Fore.RED}Order ...{order_id_short} did not fill within {timeout_seconds}s timeout.{Style.RESET_ALL}")
    return None  # Timeout failure


def place_risked_market_order(exchange: ccxt.Exchange, symbol: str, side: str,
                            risk_percentage: Decimal, current_atr: Decimal | None, sl_atr_multiplier: Decimal,
                            leverage: int, max_order_cap_usdt: Decimal, margin_check_buffer: Decimal,
                            tsl_percent: Decimal, tsl_activation_offset_percent: Decimal) -> dict[str, Any] | None:
    """Places market entry, waits for fill, then places exchange-native fixed SL and TSL using Decimal precision."""
    market_base = symbol.split('/')[0]  # For concise alerts
    order_side_color = Fore.GREEN if side == CONFIG.side_buy else Fore.RED
    logger.info(f"{order_side_color}{Style.BRIGHT}Place Order Ritual: Initiating {side.upper()} for {symbol}...{Style.RESET_ALL}")

    # --- Pre-computation & Validation ---
    if current_atr is None or current_atr <= Decimal("0"):
        logger.error(f"{Fore.RED}Place Order ({side.upper()}): Invalid ATR ({current_atr}). Cannot calculate SL distance.{Style.RESET_ALL}")
        return None

    entry_price_estimate: Decimal | None = None
    initial_sl_price_estimate: Decimal | None = None
    final_quantity: Decimal | None = None
    market: dict | None = None

    try:
        # === 1. Gather Resources: Balance, Market Info, Limits ===
        logger.debug("Gathering resources: Balance, Market Structure, Limits...")
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

        # Extract USDT balance details
        usdt_balance = balance.get(CONFIG.usdt_symbol, {})
        usdt_total = safe_decimal_conversion(usdt_balance.get('total'))  # Total equity including PnL
        usdt_free = safe_decimal_conversion(usdt_balance.get('free'))   # Available for new orders
        # Use total equity for risk calculation if available, otherwise fall back to free
        usdt_equity = usdt_total if usdt_total > 0 else usdt_free

        if usdt_equity <= Decimal("0"):
            logger.error(f"{Fore.RED}Place Order ({side.upper()}): Zero or Invalid equity ({usdt_equity:.4f}). Cannot proceed.{Style.RESET_ALL}"); return None
        if usdt_free < Decimal("0"):  # Free margin shouldn't be negative
            logger.error(f"{Fore.RED}Place Order ({side.upper()}): Invalid free margin ({usdt_free:.4f}). Cannot proceed.{Style.RESET_ALL}"); return None
        logger.debug(f"Resources: Equity={usdt_equity:.4f}, Free={usdt_free:.4f} {CONFIG.usdt_symbol}")

        # === 2. Estimate Entry Price - Peering into the immediate future ===
        # Use shallow order book fetch for a quick estimate
        ob_data = analyze_order_book(exchange, symbol, CONFIG.shallow_ob_fetch_depth, CONFIG.shallow_ob_fetch_depth)
        best_ask = ob_data.get("best_ask")
        best_bid = ob_data.get("best_bid")
        if side == CONFIG.side_buy and best_ask: entry_price_estimate = best_ask  # Estimate buying at best ask
        elif side == CONFIG.side_sell and best_bid: entry_price_estimate = best_bid  # Estimate selling at best bid
        else:
            # Fallback: Fetch last traded price if OB data is unavailable
            try:
                ticker = exchange.fetch_ticker(symbol)
                entry_price_estimate = safe_decimal_conversion(ticker.get('last'))
                logger.debug(f"Using ticker last price for estimate: {entry_price_estimate}")
            except Exception as e:
                logger.error(f"{Fore.RED}Failed to fetch ticker price for estimate: {e}{Style.RESET_ALL}"); return None
        if not entry_price_estimate or entry_price_estimate <= 0:
            logger.error(f"{Fore.RED}Invalid entry price estimate ({entry_price_estimate}). Cannot proceed.{Style.RESET_ALL}"); return None
        logger.debug(f"Estimated Entry Price ~ {entry_price_estimate:.4f}")

        # === 3. Calculate Initial Stop Loss Price (Estimate) - The First Ward ===
        sl_distance = current_atr * sl_atr_multiplier  # Calculate stop distance based on volatility
        initial_sl_price_raw = (entry_price_estimate - sl_distance) if side == CONFIG.side_buy else (entry_price_estimate + sl_distance)
        # Ensure SL price respects minimum price limit if applicable
        if min_price is not None and initial_sl_price_raw < min_price:
            logger.warning(f"{Fore.YELLOW}Initial SL price {initial_sl_price_raw:.4f} below min price {min_price}. Adjusting SL to min price.{Style.RESET_ALL}")
            initial_sl_price_raw = min_price
        if initial_sl_price_raw <= 0:
            logger.error(f"{Fore.RED}Invalid Initial SL price calculation resulted in {initial_sl_price_raw:.4f}. Cannot proceed.{Style.RESET_ALL}"); return None
        # Format the estimated SL price according to market rules
        initial_sl_price_estimate = safe_decimal_conversion(format_price(exchange, symbol, initial_sl_price_raw))
        logger.info(f"Calculated Initial SL Price (Estimate) ~ {Fore.YELLOW}{initial_sl_price_estimate:.4f}{Style.RESET_ALL} (ATR Dist: {sl_distance:.4f})")

        # === 4. Calculate Position Size - Determining the Energy Input ===
        calc_qty, req_margin = calculate_position_size(usdt_equity, risk_percentage, entry_price_estimate, initial_sl_price_estimate, leverage, symbol, exchange)
        if calc_qty is None or req_margin is None:
            logger.error(f"{Fore.RED}Failed risk calculation. Cannot determine position size.{Style.RESET_ALL}"); return None
        final_quantity = calc_qty

        # === 5. Apply Max Order Cap - Limiting the Power ===
        pos_value_estimate = final_quantity * entry_price_estimate
        if pos_value_estimate > max_order_cap_usdt:
            logger.warning(f"{Fore.YELLOW}Calculated order value {pos_value_estimate:.4f} > Cap {max_order_cap_usdt:.4f}. Capping quantity.{Style.RESET_ALL}")
            final_quantity = max_order_cap_usdt / entry_price_estimate
            # Format the capped quantity according to market rules
            final_quantity = safe_decimal_conversion(format_amount(exchange, symbol, final_quantity))
            # Recalculate estimated margin based on the capped quantity
            req_margin = (max_order_cap_usdt / Decimal(leverage))
            logger.info(f"Capped Qty: {final_quantity:.8f}, New Est. Margin: {req_margin:.4f}")

        # === 6. Check Limits & Margin Availability - Final Preparations ===
        if final_quantity <= CONFIG.position_qty_epsilon:
            logger.error(f"{Fore.RED}Final Quantity negligible after capping/formatting: {final_quantity:.8f}{Style.RESET_ALL}"); return None
        # Check against minimum order size
        if min_qty is not None and final_quantity < min_qty:
            logger.error(f"{Fore.RED}Final Quantity {final_quantity:.8f} < Min Allowed {min_qty}. Cannot place order.{Style.RESET_ALL}"); return None
        # Check against maximum order size (though capping should handle this, double-check)
        if max_qty is not None and final_quantity > max_qty:
            logger.warning(f"{Fore.YELLOW}Final Quantity {final_quantity:.8f} > Max Allowed {max_qty}. Capping to max.{Style.RESET_ALL}")
            final_quantity = max_qty
            # Re-format capped amount one last time
            final_quantity = safe_decimal_conversion(format_amount(exchange, symbol, final_quantity))

        # Final margin calculation based on potentially adjusted final_quantity
        final_req_margin = (final_quantity * entry_price_estimate) / Decimal(leverage)
        req_margin_buffered = final_req_margin * margin_check_buffer  # Add safety buffer
        logger.debug(f"Final Margin Check: Need ~{final_req_margin:.4f} (Buffered: {req_margin_buffered:.4f}), Have Free: {usdt_free:.4f}")

        # Check if sufficient free margin is available
        if usdt_free < req_margin_buffered:
            logger.error(f"{Fore.RED}Insufficient FREE margin. Need ~{req_margin_buffered:.4f} (incl. buffer), Have {usdt_free:.4f}{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Insufficient Free Margin (Need ~{req_margin_buffered:.2f})")
            return None
        logger.info(f"{Fore.GREEN}Final Order Details: Qty={final_quantity:.8f}, EstValue={final_quantity * entry_price_estimate:.4f}, EstMargin={final_req_margin:.4f}. Margin check OK.{Style.RESET_ALL}")

        # === 7. Place Entry Market Order - Unleashing the Energy ===
        entry_order: dict[str, Any] | None = None
        order_id: str | None = None
        try:
            qty_float = float(final_quantity)  # CCXT expects float for amount
            entry_color = Back.GREEN if side == CONFIG.side_buy else Back.RED
            logger.warning(f"{entry_color}{Fore.BLACK}{Style.BRIGHT}*** Placing {side.upper()} MARKET ENTRY: {qty_float:.8f} {symbol} ***{Style.RESET_ALL}")
            # Create the market order (not reduceOnly)
            entry_order = exchange.create_market_order(symbol=symbol, side=side, amount=qty_float, params={'reduce_only': False})
            order_id = entry_order.get('id')
            if not order_id:
                # This is unexpected and problematic
                logger.error(f"{Back.RED}{Fore.WHITE}Entry order placed but NO ID returned! Cannot track. Response: {entry_order}{Style.RESET_ALL}")
                send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] CRITICAL ORDER FAIL ({side.upper()}): Entry placed but NO ID received!")
                # Attempt to find position manually? Difficult. Best to stop or handle manually.
                return None  # Cannot proceed without order ID
            logger.success(f"{Fore.GREEN}Market Entry Order submitted. ID: ...{format_order_id(order_id)}. Awaiting confirmation...{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}FAILED TO PLACE ENTRY ORDER: {e}{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())
            send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Entry placement failed: {type(e).__name__}")
            return None  # Failed to place entry

        # === 8. Wait for Entry Fill Confirmation - Observing the Impact ===
        filled_entry = wait_for_order_fill(exchange, order_id, symbol, CONFIG.order_fill_timeout_seconds)
        if not filled_entry:
            logger.error(f"{Fore.RED}Entry order ...{format_order_id(order_id)} did not fill or failed confirmation.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Entry ...{format_order_id(order_id)} fill timeout/fail.")
            # Try to cancel the potentially stuck order (might fail if already filled/gone)
            try:
                logger.warning(f"Attempting to cancel potentially stuck/unconfirmed order ...{format_order_id(order_id)}")
                exchange.cancel_order(order_id, symbol)
            except Exception as cancel_e:
                logger.warning(f"Could not cancel order ...{format_order_id(order_id)} (may be filled or already gone): {cancel_e}")
            return None  # Cannot proceed without confirmed entry

        # === 9. Extract Actual Fill Details - Reading the Result ===
        avg_fill_price = safe_decimal_conversion(filled_entry.get('average'))
        filled_qty = safe_decimal_conversion(filled_entry.get('filled'))
        cost = safe_decimal_conversion(filled_entry.get('cost'))  # Total cost in quote currency (USDT)

        # Validate fill details
        if avg_fill_price <= 0 or filled_qty <= CONFIG.position_qty_epsilon:
            logger.error(f"{Back.RED}{Fore.WHITE}Invalid fill details for ...{format_order_id(order_id)}: Price={avg_fill_price}, Qty={filled_qty}. Position state unknown!{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] CRITICAL ORDER FAIL ({side.upper()}): Invalid fill details ...{format_order_id(order_id)}!")
            # Position might be open with bad data. Manual check needed.
            return filled_entry  # Return problematic order details

        fill_color = Back.GREEN if side == CONFIG.side_buy else Back.RED
        logger.success(f"{fill_color}{Fore.BLACK}{Style.BRIGHT}ENTRY CONFIRMED: ...{format_order_id(order_id)}. Filled: {filled_qty:.8f} @ {avg_fill_price:.4f}, Cost: {cost:.4f} USDT{Style.RESET_ALL}")

        # === 10. Calculate ACTUAL Stop Loss Price - Setting the Ward ===
        # Use the actual average fill price for SL calculation
        actual_sl_price_raw = (avg_fill_price - sl_distance) if side == CONFIG.side_buy else (avg_fill_price + sl_distance)
        # Apply min price constraint again based on actual fill
        if min_price is not None and actual_sl_price_raw < min_price:
             logger.warning(f"{Fore.YELLOW}Actual SL price {actual_sl_price_raw:.4f} below min price {min_price}. Adjusting SL to min price.{Style.RESET_ALL}")
             actual_sl_price_raw = min_price
        if actual_sl_price_raw <= 0:
            logger.error(f"{Back.RED}{Fore.WHITE}CRITICAL: Invalid ACTUAL SL price ({actual_sl_price_raw:.4f}) calculated based on fill price {avg_fill_price:.4f}. Cannot place SL!{Style.RESET_ALL}")
            # Position is open without SL protection. Attempt emergency close.
            send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] CRITICAL ({side.upper()}): Invalid ACTUAL SL price! Attempting emergency close.")
            close_position(exchange, symbol, {'side': side, 'qty': filled_qty}, reason="Invalid SL Calc")  # Use filled details
            return filled_entry  # Return filled entry, but signal failure state
        # Format the final SL price
        actual_sl_price_str = format_price(exchange, symbol, actual_sl_price_raw)
        actual_sl_price_float = float(actual_sl_price_str)  # For CCXT param

        # === 11. Place Initial Fixed Stop Loss Order - The Static Ward ===
        sl_order_id = "N/A"
        try:
            sl_side = CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy  # Opposite side for SL
            sl_qty_str = format_amount(exchange, symbol, filled_qty)  # Use actual filled quantity
            sl_qty_float = float(sl_qty_str)

            logger.info(f"{Fore.CYAN}Weaving Initial Fixed SL ({sl_atr_multiplier}*ATR)... Side: {sl_side.upper()}, Qty: {sl_qty_float:.8f}, TriggerPx: {actual_sl_price_str}{Style.RESET_ALL}")
            # Bybit V5 stop order params: stopPrice (trigger price), reduceOnly (must be true for SL/TP)
            sl_params = {'stopPrice': actual_sl_price_float, 'reduceOnly': True}
            # Use 'stopMarket' type for market execution when trigger price is hit
            sl_order = exchange.create_order(symbol, 'stopMarket', sl_side, sl_qty_float, params=sl_params)
            sl_order_id = format_order_id(sl_order.get('id'))
            logger.success(f"{Fore.GREEN}Initial Fixed SL ward placed. ID: ...{sl_order_id}, Trigger: {actual_sl_price_str}{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}FAILED to place Initial Fixed SL ward: {e}{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())
            send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ERROR ({side.upper()}): Failed initial SL placement: {type(e).__name__}")
            # Don't necessarily close here, TSL might still work, or user might want manual intervention. Logged failure is key.

        # === 12. Place Trailing Stop Loss Order - The Adaptive Shield ===
        tsl_order_id = "N/A"
        tsl_act_price_str = "N/A"
        try:
            # Calculate TSL activation price based on actual fill price
            act_offset = avg_fill_price * tsl_activation_offset_percent
            act_price_raw = (avg_fill_price + act_offset) if side == CONFIG.side_buy else (avg_fill_price - act_offset)
            # Apply min price constraint to activation price
            if min_price is not None and act_price_raw < min_price:
                logger.warning(f"{Fore.YELLOW}TSL activation price {act_price_raw:.4f} below min price {min_price}. Adjusting to min price.{Style.RESET_ALL}")
                act_price_raw = min_price
            if act_price_raw <= 0: raise ValueError(f"Invalid TSL activation price {act_price_raw:.4f}")

            tsl_act_price_str = format_price(exchange, symbol, act_price_raw)  # Format activation price
            tsl_act_price_float = float(tsl_act_price_str)  # For CCXT param
            tsl_side = CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy  # Opposite side
            # Bybit V5 uses 'trailingStop' for percentage distance (e.g., "0.5" for 0.5%)
            # Convert our decimal percentage (e.g., 0.005) to percentage string (e.g., "0.5")
            tsl_trail_value_str = str((tsl_percent * Decimal("100")).normalize())
            tsl_qty_str = format_amount(exchange, symbol, filled_qty)  # Use actual filled quantity
            tsl_qty_float = float(tsl_qty_str)

            logger.info(f"{Fore.CYAN}Weaving Trailing SL ({tsl_percent:.2%})... Side: {tsl_side.upper()}, Qty: {tsl_qty_float:.8f}, Trail%: {tsl_trail_value_str}, ActPx: {tsl_act_price_str}{Style.RESET_ALL}")
            # Bybit V5 TSL params via CCXT:
            # 'trailingStop': Percentage value as a string (e.g., "0.5")
            # 'activePrice': Activation trigger price (float)
            # 'reduceOnly': Must be True
            tsl_params = {
                'trailingStop': tsl_trail_value_str,
                'activePrice': tsl_act_price_float,
                'reduceOnly': True,
            }
            # Use 'stopMarket' type with TSL params for Bybit V5 via CCXT
            tsl_order = exchange.create_order(symbol, 'stopMarket', tsl_side, tsl_qty_float, params=tsl_params)
            tsl_order_id = format_order_id(tsl_order.get('id'))
            logger.success(f"{Fore.GREEN}Trailing SL shield placed. ID: ...{tsl_order_id}, Trail%: {tsl_trail_value_str}, ActPx: {tsl_act_price_str}{Style.RESET_ALL}")

            # --- Final Comprehensive SMS Alert ---
            sms_msg = (f"[{market_base}/{CONFIG.strategy_name}] ENTERED {side.upper()} {filled_qty:.8f} @ {avg_fill_price:.4f}. "
                       f"Init SL ~{actual_sl_price_str}. TSL {tsl_percent:.2%} act@{tsl_act_price_str}. "
                       f"IDs E:...{format_order_id(order_id)}, SL:...{sl_order_id}, TSL:...{tsl_order_id}")
            send_sms_alert(sms_msg)

        except Exception as e:
            logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}FAILED to place Trailing SL shield: {e}{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())
            send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ERROR ({side.upper()}): Failed TSL placement: {type(e).__name__}")
            # If TSL fails but initial SL was placed, the position is still protected initially.

        # Return the details of the successfully filled entry order
        return filled_entry

    except (ccxt.InsufficientFunds, ccxt.NetworkError, ccxt.ExchangeError, ValueError, Exception) as e:
        # Catch errors occurring before order placement or during setup
        logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Place Order Ritual ({side.upper()}): Overall process failed: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Overall process failed: {type(e).__name__}")
    return None  # Indicate failure of the overall process


def cancel_open_orders(exchange: ccxt.Exchange, symbol: str, reason: str = "Cleanup") -> None:
    """Attempts to cancel all open orders for the specified symbol, clearing residual energies."""
    logger.info(f"{Fore.CYAN}Order Cleanup: Attempting for {symbol} (Reason: {reason})...{Style.RESET_ALL}")
    try:
        if not exchange.has.get('fetchOpenOrders'):
            logger.warning(f"{Fore.YELLOW}Order Cleanup: fetchOpenOrders spell not available.{Style.RESET_ALL}")
            return
        # Summon list of open orders
        open_orders = exchange.fetch_open_orders(symbol)
        if not open_orders:
            logger.info(f"{Fore.CYAN}Order Cleanup: No open orders found for {symbol}.{Style.RESET_ALL}")
            return

        logger.warning(f"{Fore.YELLOW}Order Cleanup: Found {len(open_orders)} open orders for {symbol}. Cancelling...{Style.RESET_ALL}")
        cancelled_count, failed_count = 0, 0
        for order in open_orders:
            order_id = order.get('id')
            order_info = f"...{format_order_id(order_id)} ({order.get('type')} {order.get('side')})"
            if order_id:
                try:
                    # Cast the cancel spell
                    exchange.cancel_order(order_id, symbol)
                    logger.info(f"{Fore.CYAN}Order Cleanup: Success for {order_info}{Style.RESET_ALL}")
                    cancelled_count += 1
                    time.sleep(0.1)  # Small delay between cancels to avoid rate limits
                except ccxt.OrderNotFound:
                    # Order might have been filled or cancelled just before this attempt
                    logger.warning(f"{Fore.YELLOW}Order Cleanup: Not found (already closed/cancelled?): {order_info}{Style.RESET_ALL}")
                    cancelled_count += 1  # Treat as cancelled if not found
                except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
                    logger.error(f"{Fore.RED}Order Cleanup: FAILED for {order_info}: {type(e).__name__} - {e}{Style.RESET_ALL}")
                    failed_count += 1
            else:
                logger.error(f"{Fore.RED}Order Cleanup: Found order with no ID: {order}. Cannot cancel.{Style.RESET_ALL}")
                failed_count += 1

        logger.info(f"{Fore.CYAN}Order Cleanup: Finished. Cancelled: {cancelled_count}, Failed: {failed_count}.{Style.RESET_ALL}")
        if failed_count > 0:
            send_sms_alert(f"[{symbol.split('/')[0]}/{CONFIG.strategy_name}] WARNING: Failed to cancel {failed_count} orders during {reason}.")
    except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
        logger.error(f"{Fore.RED}Order Cleanup: Failed fetching open orders for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")


# --- Strategy Signal Generation - Interpreting the Omens ---
def generate_signals(df: pd.DataFrame, strategy_name: str) -> dict[str, Any]:
    """Generates entry/exit signals based on the selected strategy's interpretation of indicators."""
    signals = {'enter_long': False, 'enter_short': False, 'exit_long': False, 'exit_short': False, 'exit_reason': "Strategy Exit"}
    if len(df) < 2: return signals  # Need previous candle for comparisons/crosses

    last = df.iloc[-1]  # Current (latest closed) candle's data
    prev = df.iloc[-2]  # Previous candle's data

    try:
        # --- Dual Supertrend Logic ---
        if strategy_name == "DUAL_SUPERTREND":
            # Enter Long: Primary ST flips long AND Confirmation ST is in uptrend
            if pd.notna(last.get('st_long')) and last['st_long'] and pd.notna(last.get('confirm_trend')) and last['confirm_trend']:
                signals['enter_long'] = True
            # Enter Short: Primary ST flips short AND Confirmation ST is in downtrend
            if pd.notna(last.get('st_short')) and last['st_short'] and pd.notna(last.get('confirm_trend')) and not last['confirm_trend']:
                signals['enter_short'] = True
            # Exit Long: Primary ST flips short
            if pd.notna(last.get('st_short')) and last['st_short']:
                signals['exit_long'] = True; signals['exit_reason'] = "Primary ST Short Flip"
            # Exit Short: Primary ST flips long
            if pd.notna(last.get('st_long')) and last['st_long']:
                signals['exit_short'] = True; signals['exit_reason'] = "Primary ST Long Flip"

        # --- StochRSI + Momentum Logic ---
        elif strategy_name == "STOCHRSI_MOMENTUM":
            k_now, d_now, mom_now = last.get('stochrsi_k'), last.get('stochrsi_d'), last.get('momentum')
            k_prev, d_prev = prev.get('stochrsi_k'), prev.get('stochrsi_d')
            # Check if all necessary indicator values are present
            if any(pd.isna(v) for v in [k_now, d_now, mom_now, k_prev, d_prev]): return signals

            # Enter Long: K crosses above D from below, K is oversold, Momentum is positive
            if k_prev <= d_prev and k_now > d_now and k_now < CONFIG.stochrsi_oversold and mom_now > CONFIG.position_qty_epsilon:
                signals['enter_long'] = True
            # Enter Short: K crosses below D from above, K is overbought, Momentum is negative
            if k_prev >= d_prev and k_now < d_now and k_now > CONFIG.stochrsi_overbought and mom_now < -CONFIG.position_qty_epsilon:
                signals['enter_short'] = True
            # Exit Long: K crosses below D
            if k_prev >= d_prev and k_now < d_now:
                signals['exit_long'] = True; signals['exit_reason'] = "StochRSI K below D"
            # Exit Short: K crosses above D
            if k_prev <= d_prev and k_now > d_now:
                signals['exit_short'] = True; signals['exit_reason'] = "StochRSI K above D"

        # --- Ehlers Fisher Logic ---
        elif strategy_name == "EHLERS_FISHER":
            fish_now, sig_now = last.get('ehlers_fisher'), last.get('ehlers_signal')
            fish_prev, sig_prev = prev.get('ehlers_fisher'), prev.get('ehlers_signal')
            if any(pd.isna(v) for v in [fish_now, sig_now, fish_prev, sig_prev]): return signals

            # Enter Long: Fisher crosses above Signal line
            if fish_prev <= sig_prev and fish_now > sig_now:
                signals['enter_long'] = True
            # Enter Short: Fisher crosses below Signal line
            if fish_prev >= sig_prev and fish_now < sig_now:
                signals['enter_short'] = True
            # Exit Long: Fisher crosses below Signal line
            if fish_prev >= sig_prev and fish_now < sig_now:
                signals['exit_long'] = True; signals['exit_reason'] = "Ehlers Fisher Short Cross"
            # Exit Short: Fisher crosses above Signal line
            if fish_prev <= sig_prev and fish_now > sig_now:
                signals['exit_short'] = True; signals['exit_reason'] = "Ehlers Fisher Long Cross"

        # --- Ehlers MA Cross Logic (Using EMA Placeholder) ---
        elif strategy_name == "EHLERS_MA_CROSS":
            fast_ma_now, slow_ma_now = last.get('fast_ema'), last.get('slow_ema')
            fast_ma_prev, slow_ma_prev = prev.get('fast_ema'), prev.get('slow_ema')
            if any(pd.isna(v) for v in [fast_ma_now, slow_ma_now, fast_ma_prev, slow_ma_prev]): return signals

            # Enter Long: Fast MA crosses above Slow MA
            if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now:
                signals['enter_long'] = True
            # Enter Short: Fast MA crosses below Slow MA
            if fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now:
                signals['enter_short'] = True
            # Exit Long: Fast MA crosses below Slow MA
            if fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now:
                signals['exit_long'] = True; signals['exit_reason'] = "Ehlers MA Short Cross (EMA)"
            # Exit Short: Fast MA crosses above Slow MA
            if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now:
                signals['exit_short'] = True; signals['exit_reason'] = "Ehlers MA Long Cross (EMA)"

    except KeyError as e:
        logger.error(f"{Fore.RED}Signal Generation Error: Missing expected indicator column in DataFrame: {e}. Strategy: {strategy_name}{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"{Fore.RED}Signal Generation Error: Unexpected disturbance: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())

    # Log generated signals for debugging if any signal is active
    if signals['enter_long'] or signals['enter_short'] or signals['exit_long'] or signals['exit_short']:
        logger.debug(f"Strategy Signals ({strategy_name}): {signals}")
    return signals


# --- Trading Logic - The Core Spell Weaving ---
def trade_logic(exchange: ccxt.Exchange, symbol: str, df: pd.DataFrame) -> None:
    """Executes the main trading logic for one cycle based on selected strategy and market conditions."""
    cycle_time_str = df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z') if not df.empty else "N/A"
    logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== New Weaving Cycle ({CONFIG.strategy_name}): {symbol} | Candle: {cycle_time_str} =========={Style.RESET_ALL}")

    # Determine required rows based on the longest lookback needed by any indicator used in *any* strategy + buffers
    # This ensures enough data regardless of the selected strategy.
    required_rows = max(
        CONFIG.st_atr_length, CONFIG.confirm_st_atr_length,
        CONFIG.stochrsi_rsi_length + CONFIG.stochrsi_stoch_length, CONFIG.momentum_length,  # Estimate lookback needed
        CONFIG.ehlers_fisher_length, CONFIG.ehlers_fisher_signal_length,
        CONFIG.ehlers_fast_period, CONFIG.ehlers_slow_period,
        CONFIG.atr_calculation_period, CONFIG.volume_ma_period
    ) + 10  # Add a safety buffer

    if df is None or len(df) < required_rows:
        logger.warning(f"{Fore.YELLOW}Trade Logic: Insufficient data ({len(df) if df is not None else 0}, need ~{required_rows}). Skipping cycle.{Style.RESET_ALL}")
        return

    action_taken_this_cycle: bool = False  # Track if an entry/exit order was placed
    try:
        # === 1. Calculate ALL Indicators - Scry the full spectrum ===
        # It's often simpler to calculate all potential indicators needed by any strategy
        # and let the signal generation function pick the ones it needs based on CONFIG.strategy_name.
        logger.debug("Calculating all potential indicators...")
        df = calculate_supertrend(df, CONFIG.st_atr_length, CONFIG.st_multiplier)
        df = calculate_supertrend(df, CONFIG.confirm_st_atr_length, CONFIG.confirm_st_multiplier, prefix="confirm_")
        df = calculate_stochrsi_momentum(df, CONFIG.stochrsi_rsi_length, CONFIG.stochrsi_stoch_length, CONFIG.stochrsi_k_period, CONFIG.stochrsi_d_period, CONFIG.momentum_length)
        df = calculate_ehlers_fisher(df, CONFIG.ehlers_fisher_length, CONFIG.ehlers_fisher_signal_length)
        df = calculate_ehlers_ma(df, CONFIG.ehlers_fast_period, CONFIG.ehlers_slow_period)  # Placeholder EMA
        vol_atr_data = analyze_volume_atr(df, CONFIG.atr_calculation_period, CONFIG.volume_ma_period)
        current_atr = vol_atr_data.get("atr")  # Crucial for SL calculation

        # === 2. Validate Base Requirements - Ensure stable ground ===
        last = df.iloc[-1]
        current_price = safe_decimal_conversion(last.get('close'))  # Get latest close price
        if pd.isna(current_price) or current_price <= 0:
            logger.warning(f"{Fore.YELLOW}Last candle close price is invalid ({current_price}). Skipping cycle.{Style.RESET_ALL}")
            return
        # Can we place an order? Requires valid ATR for SL calculation.
        can_place_order = current_atr is not None and current_atr > Decimal("0")
        if not can_place_order:
            logger.warning(f"{Fore.YELLOW}Invalid ATR ({current_atr}). Cannot calculate SL or place new orders this cycle.{Style.RESET_ALL}")

        # === 3. Get Position & Analyze Order Book (if configured) ===
        position = get_current_position(exchange, symbol)  # Check current market presence
        position_side = position['side']
        position_qty = position['qty']
        position_entry = position['entry_price']
        # Fetch OB data if configured for every cycle, or later if needed for confirmation
        ob_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit) if CONFIG.fetch_order_book_per_cycle else None

        # === 4. Log Current State - The Oracle Reports ===
        vol_ratio = vol_atr_data.get("volume_ratio")
        vol_spike = vol_ratio is not None and vol_ratio > CONFIG.volume_spike_threshold
        bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None
        spread = ob_data.get("spread") if ob_data else None

        # Log core state
        atr_str = f"{current_atr:.5f}" if current_atr else "N/A"
        logger.info(f"State | Price: {Fore.CYAN}{current_price:.4f}{Style.RESET_ALL}, ATR({CONFIG.atr_calculation_period}): {Fore.MAGENTA}{atr_str}{Style.RESET_ALL}")
        # Log confirmation states
        vol_ratio_str = f"{vol_ratio:.2f}" if vol_ratio else "N/A"
        vol_spike_str = f"{Fore.GREEN}YES{Style.RESET_ALL}" if vol_spike else f"{Fore.RED}NO{Style.RESET_ALL}"
        logger.info(f"State | Volume: Ratio={Fore.YELLOW}{vol_ratio_str}{Style.RESET_ALL}, Spike={vol_spike_str} (Req={CONFIG.require_volume_spike_for_entry})")
        ob_ratio_str = f"{bid_ask_ratio:.3f}" if bid_ask_ratio else "N/A"
        ob_spread_str = f"{spread:.4f}" if spread else "N/A"
        logger.info(f"State | OrderBook: Ratio={Fore.YELLOW}{ob_ratio_str}{Style.RESET_ALL}, Spread={ob_spread_str} (Fetched={ob_data is not None})")
        # Log position state
        pos_color = Fore.GREEN if position_side == CONFIG.pos_long else (Fore.RED if position_side == CONFIG.pos_short else Fore.BLUE)
        logger.info(f"State | Position: Side={pos_color}{position_side}{Style.RESET_ALL}, Qty={position_qty:.8f}, Entry={position_entry:.4f}")

        # === 5. Generate Strategy Signals - Interpret the Omens ===
        strategy_signals = generate_signals(df, CONFIG.strategy_name)
        # Log if any signal is generated
        # (Logging moved to within generate_signals for brevity here)

        # === 6. Execute Exit Actions - If the Omens Demand Retreat ===
        should_exit_long = position_side == CONFIG.pos_long and strategy_signals['exit_long']
        should_exit_short = position_side == CONFIG.pos_short and strategy_signals['exit_short']

        if should_exit_long or should_exit_short:
            exit_reason = strategy_signals['exit_reason']
            exit_side_color = Back.YELLOW
            logger.warning(f"{exit_side_color}{Fore.BLACK}{Style.BRIGHT}*** TRADE EXIT SIGNAL: Closing {position_side} due to {exit_reason} ***{Style.RESET_ALL}")
            # Cancel existing SL/TSL orders before placing market close
            cancel_open_orders(exchange, symbol, f"Pre-Exit Cleanup ({exit_reason})")
            time.sleep(0.5)  # Small pause after cancel before closing
            # Attempt to close the position
            close_result = close_position(exchange, symbol, position, reason=exit_reason)
            if close_result:
                action_taken_this_cycle = True
                # Add delay after closing before allowing new entry
                logger.info(f"Pausing for {CONFIG.post_close_delay_seconds}s after closing position...")
                time.sleep(CONFIG.post_close_delay_seconds)
            # Exit cycle immediately after attempting close, regardless of success
            # This prevents trying to enter immediately after an exit signal in the same cycle.
            return

        # === 7. Check & Execute Entry Actions (Only if Flat & Can Place Order) ===
        if position_side != CONFIG.pos_none:
             logger.info(f"Holding {pos_color}{position_side}{Style.RESET_ALL} position. Awaiting SL/TSL or Strategy Exit signal.")
             return  # Do nothing if already in a position
        if not can_place_order:
             logger.warning(f"{Fore.YELLOW}Holding Cash. Cannot enter: Invalid ATR ({current_atr}) prevents SL calculation.{Style.RESET_ALL}")
             return  # Do nothing if we can't calculate SL

        # --- Check Entry Conditions ---
        logger.debug("Position is Flat. Checking entry signals...")
        potential_entry = strategy_signals['enter_long'] or strategy_signals['enter_short']

        # Fetch OB data now if not fetched per cycle AND there's a potential entry signal AND OB confirmation is desired
        # (Assuming OB check is always desired if a signal exists, adjust logic if needed)
        ob_check_required = potential_entry  # Check OB if there's any entry signal?
        if ob_check_required and ob_data is None:
            logger.debug("Potential entry signal and OB not fetched yet, fetching OB for confirmation...")
            ob_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit)
            bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None  # Update ratio

        # Evaluate Confirmation Filters
        ob_available = ob_data is not None and bid_ask_ratio is not None
        # Long OB Confirmation: Ratio >= Threshold OR check not required
        passes_long_ob = not ob_check_required or (ob_available and bid_ask_ratio >= CONFIG.order_book_ratio_threshold_long)
        # Short OB Confirmation: Ratio <= Threshold OR check not required
        passes_short_ob = not ob_check_required or (ob_available and bid_ask_ratio <= CONFIG.order_book_ratio_threshold_short)
        ob_log_ratio = f"{bid_ask_ratio:.3f}" if bid_ask_ratio else "N/A"
        ob_log = f"OB OK? (L:{passes_long_ob}, S:{passes_short_ob}, Ratio={ob_log_ratio}, Req={ob_check_required})"

        # Volume Confirmation
        vol_check_required = CONFIG.require_volume_spike_for_entry
        passes_volume = not vol_check_required or (vol_spike)
        vol_log = f"Vol OK? (Pass:{passes_volume}, Spike={vol_spike}, Req={vol_check_required})"

        # --- Combine Strategy Signal with Confirmations ---
        enter_long = strategy_signals['enter_long'] and passes_long_ob and passes_volume
        enter_short = strategy_signals['enter_short'] and passes_short_ob and passes_volume

        # Log final entry decision logic
        if strategy_signals['enter_long'] or strategy_signals['enter_short']:  # Only log if there was a base signal
            logger.debug(f"Final Entry Check (Long): Strategy={strategy_signals['enter_long']}, {ob_log}, {vol_log} => {Fore.GREEN if enter_long else Fore.RED}Enter={enter_long}{Style.RESET_ALL}")
            logger.debug(f"Final Entry Check (Short): Strategy={strategy_signals['enter_short']}, {ob_log}, {vol_log} => {Fore.GREEN if enter_short else Fore.RED}Enter={enter_short}{Style.RESET_ALL}")

        # --- Execute Entry ---
        if enter_long:
            entry_side_color = Back.GREEN
            logger.success(f"{entry_side_color}{Fore.BLACK}{Style.BRIGHT}*** TRADE SIGNAL: CONFIRMED LONG ENTRY ({CONFIG.strategy_name}) for {symbol} ***{Style.RESET_ALL}")
            # Cancel any stray orders before entering
            cancel_open_orders(exchange, symbol, "Pre-Long Entry")
            time.sleep(0.5)  # Small pause after cancel
            # Place the risked order with SL and TSL
            place_result = place_risked_market_order(
                exchange, symbol, CONFIG.side_buy, CONFIG.risk_per_trade_percentage, current_atr, CONFIG.atr_stop_loss_multiplier,
                CONFIG.leverage, CONFIG.max_order_usdt_amount, CONFIG.required_margin_buffer,
                CONFIG.trailing_stop_percentage, CONFIG.trailing_stop_activation_offset_percent)
            if place_result: action_taken_this_cycle = True

        elif enter_short:
            entry_side_color = Back.RED
            logger.success(f"{entry_side_color}{Fore.WHITE}{Style.BRIGHT}*** TRADE SIGNAL: CONFIRMED SHORT ENTRY ({CONFIG.strategy_name}) for {symbol} ***{Style.RESET_ALL}")
            # Cancel any stray orders before entering
            cancel_open_orders(exchange, symbol, "Pre-Short Entry")
            time.sleep(0.5)  # Small pause after cancel
            # Place the risked order with SL and TSL
            place_result = place_risked_market_order(
                exchange, symbol, CONFIG.side_sell, CONFIG.risk_per_trade_percentage, current_atr, CONFIG.atr_stop_loss_multiplier,
                CONFIG.leverage, CONFIG.max_order_usdt_amount, CONFIG.required_margin_buffer,
                CONFIG.trailing_stop_percentage, CONFIG.trailing_stop_activation_offset_percent)
            if place_result: action_taken_this_cycle = True

        else:
             # Log if no entry signal met confirmations
             if potential_entry and not action_taken_this_cycle:
                 logger.info("Strategy signal present but confirmation filters not met. Holding cash.")
             elif not action_taken_this_cycle:
                 logger.info("No entry signal generated by strategy. Holding cash.")

    except Exception as e:
        # Catch-all for unexpected errors within the main logic loop
        logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}CRITICAL UNEXPECTED ERROR in trade_logic: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{symbol.split('/')[0]}/{CONFIG.strategy_name}] CRITICAL ERROR in trade_logic: {type(e).__name__}. Check logs!")
    finally:
        # Mark the end of the cycle clearly
        logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== Cycle Weaving End: {symbol} =========={Style.RESET_ALL}\n")


# --- Graceful Shutdown - Withdrawing the Arcane Energies ---
def graceful_shutdown(exchange: ccxt.Exchange | None, symbol: str | None) -> None:
    """Attempts to close position and cancel orders before exiting, ensuring a clean withdrawal."""
    logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}Shutdown requested. Withdrawing arcane energies gracefully...{Style.RESET_ALL}")
    market_base = symbol.split('/')[0] if symbol else "Bot"
    send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] Shutdown initiated. Attempting cleanup...")

    if not exchange or not symbol:
        logger.warning(f"{Fore.YELLOW}Shutdown: Exchange portal or symbol not defined. Cannot perform cleanup.{Style.RESET_ALL}")
        return

    try:
        # 1. Cancel All Open Orders - Dispel residual intents
        logger.info("Shutdown: Cancelling all open orders...")
        cancel_open_orders(exchange, symbol, reason="Graceful Shutdown")
        time.sleep(1)  # Allow cancellations to process

        # 2. Check and Close Existing Position - Banish final presence
        logger.info("Shutdown: Checking for active position...")
        position = get_current_position(exchange, symbol)
        if position['side'] != CONFIG.pos_none:
            pos_color = Fore.GREEN if position['side'] == CONFIG.pos_long else Fore.RED
            logger.warning(f"{Fore.YELLOW}Shutdown: Active {pos_color}{position['side']}{Style.RESET_ALL} position found (Qty: {position['qty']:.8f}). Attempting banishment...{Style.RESET_ALL}")
            close_result = close_position(exchange, symbol, position, reason="Shutdown")
            if close_result:
                logger.info(f"{Fore.CYAN}Shutdown: Close order placed. Waiting {CONFIG.post_close_delay_seconds * 2}s for confirmation...{Style.RESET_ALL}")
                time.sleep(CONFIG.post_close_delay_seconds * 2)  # Wait longer to be sure
                # Final check after waiting
                final_pos = get_current_position(exchange, symbol)
                if final_pos['side'] == CONFIG.pos_none:
                    logger.success(f"{Fore.GREEN}{Style.BRIGHT}Shutdown: Position confirmed BANISHED.{Style.RESET_ALL}")
                    send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] Position confirmed CLOSED on shutdown.")
                else:
                    # This is bad - manual intervention likely needed
                    logger.error(f"{Back.RED}{Fore.WHITE}Shutdown: FAILED TO CONFIRM position closure. Final state: {final_pos['side']} Qty={final_pos['qty']:.8f}. MANUAL CHECK REQUIRED!{Style.RESET_ALL}")
                    send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] CRITICAL ERROR: Failed confirm closure! Final: {final_pos['side']} Qty={final_pos['qty']:.8f}. MANUAL CHECK!")
            else:
                # Close order placement failed
                logger.error(f"{Back.RED}{Fore.WHITE}Shutdown: Failed to place close order. MANUAL INTERVENTION REQUIRED!{Style.RESET_ALL}")
                send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] CRITICAL ERROR: Failed PLACE close order on shutdown. MANUAL CHECK!")
        else:
            logger.info(f"{Fore.GREEN}Shutdown: No active position found. Clean exit.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] No active position found on shutdown.")

    except Exception as e:
        logger.error(f"{Fore.RED}Shutdown: Error during cleanup sequence: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] Error during shutdown cleanup: {type(e).__name__}")

    logger.info(f"{Fore.YELLOW}{Style.BRIGHT}--- Scalping Spell Shutdown Complete ---{Style.RESET_ALL}")


# --- Main Execution - Igniting the Spell ---
def main() -> None:
    """Main function to initialize, set up, and run the trading loop."""
    start_time = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Bybit Scalping Spell v2.2 Initializing ({start_time}) ---{Style.RESET_ALL}")
    logger.info(f"{Fore.CYAN}--- Strategy Enchantment Selected: {CONFIG.strategy_name} ---{Style.RESET_ALL}")
    logger.info(f"{Fore.GREEN}--- Protective Wards: Initial ATR-Stop + Exchange Trailing Stop ---{Style.RESET_ALL}")
    logger.warning(f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}--- !!! LIVE FUTURES SCALPING MODE - EXTREME RISK - HANDLE WITH CARE !!! ---{Style.RESET_ALL}")

    exchange: ccxt.Exchange | None = None
    symbol: str | None = None  # The specific market symbol (e.g., BTC/USDT:USDT)
    run_bot: bool = True  # Controls the main loop
    cycle_count: int = 0  # Tracks the number of iterations

    try:
        # === Initialize Exchange Portal ===
        exchange = initialize_exchange()
        if not exchange:
            logger.critical("Failed to open exchange portal. Spell cannot proceed.")
            return  # Exit if connection failed

        # === Setup Symbol and Leverage - Focusing the Spell ===
        try:
            # Allow user input for symbol, falling back to config default
            sym_input = input(f"{Fore.YELLOW}Enter target symbol {Style.DIM}(Default [{CONFIG.symbol}]){Style.NORMAL}: {Style.RESET_ALL}").strip()
            symbol_to_use = sym_input or CONFIG.symbol
            # Validate and get unified symbol from CCXT
            market = exchange.market(symbol_to_use)
            symbol = market['symbol']  # Use the precise symbol recognized by CCXT
            # Ensure it's a futures/contract market
            if not market.get('contract'):
                raise ValueError(f"Market '{symbol}' is not a contract/futures market.")
            logger.info(f"{Fore.GREEN}Focusing spell on Symbol: {symbol} (Type: {market.get('type')}, Linear: {market.get('linear')}){Style.RESET_ALL}")
            # Set the desired leverage
            if not set_leverage(exchange, symbol, CONFIG.leverage):
                raise RuntimeError("Leverage conjuring failed. Cannot proceed.")
        except (ccxt.BadSymbol, KeyError, ValueError, RuntimeError) as e:
            logger.critical(f"{Back.RED}{Fore.WHITE}Symbol/Leverage setup failed: {e}{Style.RESET_ALL}")
            send_sms_alert(f"[ScalpBot/{CONFIG.strategy_name}] CRITICAL: Symbol/Leverage setup FAILED ({e}). Exiting.")
            return
        except Exception as e:
            logger.critical(f"{Back.RED}{Fore.WHITE}Unexpected error during spell focus setup: {e}{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())
            send_sms_alert(f"[ScalpBot/{CONFIG.strategy_name}] CRITICAL: Unexpected setup error. Exiting.")
            return

        # === Log Configuration Summary - Reciting the Parameters ===
        logger.info(f"{Fore.MAGENTA}--- Spell Configuration Summary ---{Style.RESET_ALL}")
        logger.info(f"{Fore.WHITE}Symbol: {symbol}, Interval: {CONFIG.interval}, Leverage: {CONFIG.leverage}x")
        logger.info(f"{Fore.CYAN}Strategy Path: {CONFIG.strategy_name}")
        # Log relevant strategy parameters for clarity
        if CONFIG.strategy_name == "DUAL_SUPERTREND": logger.info(f"  Params: ST={CONFIG.st_atr_length}/{CONFIG.st_multiplier}, ConfirmST={CONFIG.confirm_st_atr_length}/{CONFIG.confirm_st_multiplier}")
        elif CONFIG.strategy_name == "STOCHRSI_MOMENTUM": logger.info(f"  Params: StochRSI={CONFIG.stochrsi_rsi_length}/{CONFIG.stochrsi_stoch_length}/{CONFIG.stochrsi_k_period}/{CONFIG.stochrsi_d_period} (OB={CONFIG.stochrsi_overbought},OS={CONFIG.stochrsi_oversold}), Mom={CONFIG.momentum_length}")
        elif CONFIG.strategy_name == "EHLERS_FISHER": logger.info(f"  Params: Fisher={CONFIG.ehlers_fisher_length}, Signal={CONFIG.ehlers_fisher_signal_length}")
        elif CONFIG.strategy_name == "EHLERS_MA_CROSS": logger.info(f"  Params: FastMA(EMA)={CONFIG.ehlers_fast_period}, SlowMA(EMA)={CONFIG.ehlers_slow_period}")
        logger.info(f"{Fore.GREEN}Risk Ward: {CONFIG.risk_per_trade_percentage:.3%}/trade, Max Pos Value: {CONFIG.max_order_usdt_amount:.4f} USDT")
        logger.info(f"{Fore.GREEN}Initial SL Ward: {CONFIG.atr_stop_loss_multiplier} * ATR({CONFIG.atr_calculation_period})")
        logger.info(f"{Fore.GREEN}Trailing SL Shield: {CONFIG.trailing_stop_percentage:.2%}, Activation Offset: {CONFIG.trailing_stop_activation_offset_percent:.2%}")
        logger.info(f"{Fore.YELLOW}Volume Filter: {CONFIG.require_volume_spike_for_entry} (MA={CONFIG.volume_ma_period}, Thr={CONFIG.volume_spike_threshold})")
        logger.info(f"{Fore.YELLOW}Order Book Filter: {CONFIG.fetch_order_book_per_cycle} (Depth={CONFIG.order_book_depth}, L>={CONFIG.order_book_ratio_threshold_long}, S<={CONFIG.order_book_ratio_threshold_short})")
        # *** THE FIX IS HERE: Ensure CONFIG.required_margin_buffer is Decimal before formatting ***
        logger.info(f"{Fore.WHITE}Timing: Sleep={CONFIG.sleep_seconds}s, Margin Buffer={CONFIG.required_margin_buffer:.1%}, SMS Alerts={CONFIG.enable_sms_alerts}{Style.RESET_ALL}")
        logger.info(f"{Fore.CYAN}Oracle Verbosity (Log Level): {logging.getLevelName(logger.level)}")
        logger.info(f"{Fore.MAGENTA}{'-' * 30}{Style.RESET_ALL}")
        market_base = symbol.split('/')[0]  # For SMS brevity
        send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] Bot Configured. SL: ATR+TSL. Starting main loop.")

        # === Main Trading Loop - The Continuous Weaving ===
        while run_bot:
            cycle_start_time = time.monotonic()
            cycle_count += 1
            logger.debug(f"{Fore.CYAN}--- Cycle {cycle_count} Weaving Start ---{Style.RESET_ALL}")
            try:
                # Determine required data length based on longest possible indicator lookback + buffer
                data_limit = max(100,  # Base minimum
                                 CONFIG.st_atr_length * 2, CONFIG.confirm_st_atr_length * 2,
                                 CONFIG.stochrsi_rsi_length + CONFIG.stochrsi_stoch_length + 5, CONFIG.momentum_length * 2,
                                 CONFIG.ehlers_fisher_length * 2, CONFIG.ehlers_fisher_signal_length * 2,
                                 CONFIG.ehlers_fast_period * 2, CONFIG.ehlers_slow_period * 2,
                                 CONFIG.atr_calculation_period * 2, CONFIG.volume_ma_period * 2
                                 ) + CONFIG.api_fetch_limit_buffer  # Add buffer for safety

                # Gather fresh market data
                df = get_market_data(exchange, symbol, CONFIG.interval, limit=data_limit)

                # Process data and execute logic if data is valid
                if df is not None and not df.empty:
                    trade_logic(exchange, symbol, df.copy())  # Pass copy to avoid modifying original df in logic
                else:
                    logger.warning(f"{Fore.YELLOW}No valid market data received for {symbol}. Skipping trade logic this cycle.{Style.RESET_ALL}")

            # --- Robust Error Handling within the Loop ---
            except ccxt.RateLimitExceeded as e:
                logger.warning(f"{Back.YELLOW}{Fore.BLACK}Rate Limit Exceeded: {e}. The exchange spirits demand patience. Sleeping longer...{Style.RESET_ALL}")
                time.sleep(CONFIG.sleep_seconds * 5)  # Sleep much longer
                send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] WARNING: Rate limit hit! Pausing.")
            except ccxt.NetworkError as e:
                # Transient network issues, retry next cycle
                logger.warning(f"{Fore.YELLOW}Network disturbance: {e}. Retrying next cycle.{Style.RESET_ALL}")
                time.sleep(CONFIG.sleep_seconds)  # Standard sleep on recoverable network errors
            except ccxt.ExchangeNotAvailable as e:
                # Exchange might be down for maintenance
                logger.error(f"{Back.RED}{Fore.WHITE}Exchange unavailable: {e}. Portal temporarily closed. Sleeping much longer...{Style.RESET_ALL}")
                time.sleep(CONFIG.sleep_seconds * 10)  # Wait a significant time
                send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ERROR: Exchange unavailable! Long pause.")
            except ccxt.AuthenticationError as e:
                # API keys might have been revoked or expired
                logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Authentication Error: {e}. Spell broken! Stopping NOW.{Style.RESET_ALL}")
                run_bot = False  # Stop the bot immediately
                send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] CRITICAL: Authentication Error! Stopping NOW.")
            except ccxt.ExchangeError as e:  # Catch other specific exchange errors
                logger.error(f"{Fore.RED}Unhandled Exchange Error: {e}{Style.RESET_ALL}")
                logger.debug(traceback.format_exc())
                send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ERROR: Unhandled Exchange error: {type(e).__name__}")
                time.sleep(CONFIG.sleep_seconds)  # Sleep before retrying after general exchange error
            except Exception as e:
                # Catch-all for truly unexpected issues
                logger.exception(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}!!! UNEXPECTED CRITICAL CHAOS: {e} !!! Stopping spell!{Style.RESET_ALL}")
                run_bot = False  # Stop the bot on unknown critical errors
                send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] CRITICAL UNEXPECTED ERROR: {type(e).__name__}! Stopping NOW.")

            # --- Loop Delay - Controlling the Rhythm ---
            if run_bot:
                elapsed = time.monotonic() - cycle_start_time
                sleep_dur = max(0, CONFIG.sleep_seconds - elapsed)
                logger.debug(f"Cycle {cycle_count} duration: {elapsed:.2f}s. Sleeping for {sleep_dur:.2f}s.")
                if sleep_dur > 0:
                    time.sleep(sleep_dur)  # Wait for the configured interval

    except KeyboardInterrupt:
        logger.warning(f"\n{Fore.YELLOW}{Style.BRIGHT}KeyboardInterrupt received. User requests withdrawal of arcane energies...{Style.RESET_ALL}")
        run_bot = False  # Signal the loop to terminate
    finally:
        # --- Graceful Shutdown Sequence ---
        # This will run whether the loop finished normally, was interrupted, or hit a critical error that set run_bot=False
        graceful_shutdown(exchange, symbol)
        symbol.split('/')[0] if symbol else "Bot"
        # Final SMS may not send if Termux process is killed abruptly, but attempt it.
        # send_sms_alert(f"[{market_base_final}] Bot process terminated.") # Optional: Alert on final termination
        logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Scalping Spell Deactivated ---{Style.RESET_ALL}")


if __name__ == "__main__":
    # Ensure the spell is cast only when invoked directly
    main()
